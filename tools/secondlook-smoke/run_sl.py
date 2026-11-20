"""Secondlook live-QA integration harness.

Drives real Frigate code on looped mp4 clips:
  ffmpeg (yuv420p @ detect fps) -> SHM ring (SharedMemoryFrameManager)
  -> primary OpenVINO yolo on regions (create_tensor_input / post_process_yolo /
     is_object_filtered / reduce_detections) -> NorfairTracker.match_and_update
  -> CameraState.update (TrackedObject.update -> patched _is_false_positive)
  -> EmbeddingMaintainer._process_frame_updates (SecondlookProcessor, real crop
     from SHM, real OpenVINO SL infer on worker thread)
  -> EmbeddingMaintainer._process_deferred_results -> EventMetadataPublisher payload
  -> TrackedObjectProcessor.set_secondlook -> TrackedObject.apply_secondlook

The three "processes" are serialized in one Python process but use wall-clock
frame times and real-time pacing, so hold_seconds vs SL latency is realistic.
"""
from __future__ import annotations

import argparse
import logging
import statistics
import subprocess
import sys
import os
import time
from types import SimpleNamespace

import numpy as np

from cfg import build_config, LABELS, MODEL

log = logging.getLogger("harness")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--cam", default="test_loop")
    ap.add_argument("--save-and-review", choices=["on", "off"], default="on")
    ap.add_argument("--sl", choices=["on", "off"], default="on")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--primary-scale", type=float, default=1.0,
                    help="multiply primary scores (simulate weaker primary)")
    ap.add_argument("--inject-fp", default=None,
                    help="label:score:x1,y1,x2,y2 constant fake primary detection")
    ap.add_argument("--queue-max", type=int, default=8)
    ap.add_argument("--hold", type=float, default=1.0)
    ap.add_argument("--sl-delay-ms", type=float, default=0.0,
                    help="extra artificial SL latency (test hold timeout)")
    ap.add_argument("--sl-model", default=None, help="override SL model path (e.g. missing -> mock)")
    ap.add_argument("--sl-plus-metadata", default=None,
                    help="Frigate+ model info JSON for --sl-model (Frigate+ path)")
    ap.add_argument("--no-hit-factor", default="default")
    ap.add_argument("--min-iou", default="default")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("frigate.config", "frigate.detectors", "frigate.util"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    logging.getLogger("frigate.data_processing.real_time.secondlook").setLevel(logging.DEBUG)
    logging.getLogger("frigate.secondlook").setLevel(logging.DEBUG)

    W, H = 1280, 720
    sl_over = {"queue_max": args.queue_max, "hold_seconds": args.hold,
               **({} if args.no_hit_factor == "default" else {"no_hit_factor": None if args.no_hit_factor == "none" else float(args.no_hit_factor)}),
               **({} if args.min_iou == "default" else {"min_box_iou": None if args.min_iou == "none" else float(args.min_iou)})}
    sl_over["save_and_review"] = {
        "enabled": args.save_and_review == "on",
        "database": {"path": os.path.join(os.environ.get("SECONDLOOK_SMOKE_OUTPUT", os.path.dirname(__file__)), "logs", "examples", "secondlook_examples.db")},
        "images": {"path": os.path.join(os.environ.get("SECONDLOOK_SMOKE_OUTPUT", os.path.dirname(__file__)), "logs", "examples")},
    }
    if args.sl_model:
        sl_over["model"] = {"path": args.sl_model,
                            "labelmap_path": LABELS,
                            "plus_metadata_path": args.sl_plus_metadata}
    config = build_config(sl_enabled=args.sl == "on", cam=args.cam, width=W, height=H,
                          fps=args.fps, sl_overrides=sl_over)
    cam_cfg = config.cameras[args.cam]
    model_cfg = config.model

    # --- frigate imports after config (heavy) ---
    import openvino as ov
    from peewee import SqliteDatabase
    from frigate.camera import PTZMetrics
    from frigate.camera.state import CameraState
    from frigate.embeddings.maintainer import EmbeddingMaintainer
    from frigate.models import Event
    from frigate.track.norfair_tracker import NorfairTracker
    from frigate.track.object_processing import TrackedObjectProcessor
    from frigate.util.image import SharedMemoryFrameManager
    from frigate.util.model import post_process_yolo
    from frigate.util.object import create_tensor_input, is_object_filtered, reduce_detections

    db = SqliteDatabase(":memory:")
    db.bind([Event])
    db.create_tables([Event])

    # --- primary detector (same onnx @640 on CPU; static shape) ---
    core = ov.Core()
    pm = core.read_model(MODEL)
    primary = core.compile_model(pm, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
    preq = primary.create_infer_request()
    labelmap = model_cfg.merged_labelmap

    def primary_detect(yuv, region):
        tensor = create_tensor_input(yuv, model_cfg, region)
        if tensor.ndim == 4 and tensor.shape[-1] == 3:  # nhwc -> nchw
            tensor = np.transpose(tensor, (0, 3, 1, 2))
        tensor = np.ascontiguousarray(tensor.astype(np.float32) / 255.0)
        preq.infer({0: tensor})
        out = [preq.get_output_tensor(i).data.copy() for i in range(len(primary.outputs))]
        dets = post_process_yolo(out, model_cfg.width, model_cfg.height)
        res = []
        size = region[2] - region[0]
        for d in dets:
            if d[1] <= 0:
                continue
            label = labelmap[int(d[0])]
            score = float(min(1.0, d[1] * args.primary_scale))
            x_min = int(max(0, d[3] * size + region[0])); y_min = int(max(0, d[2] * size + region[1]))
            x_max = int(min(W - 1, d[5] * size + region[0])); y_max = int(min(H - 1, d[4] * size + region[1]))
            if x_min >= W - 1 or y_min >= H - 1:
                continue
            w, h = x_max - x_min, y_max - y_min
            det = (label, score, (x_min, y_min, x_max, y_max), w * h, w / max(1, h), region)
            if is_object_filtered(det, cam_cfg.objects.track, cam_cfg.objects.filters):
                continue
            res.append(det)
        return res

    regions = [(0, 0, H, H), (W - H, 0, W, H)]
    inject = []
    for spec in (args.inject_fp.split(";") if args.inject_fp else []):
        lab, sc, box = spec.split(":")
        b = tuple(int(v) for v in box.split(","))
        inject.append((lab, float(sc), b, (b[2] - b[0]) * (b[3] - b[1]),
                       (b[2] - b[0]) / max(1, b[3] - b[1]), (0, 0, H, H)))

    # --- SHM ring like capture process ---
    fm_cap = SharedMemoryFrameManager()
    ring = 10
    fsize = W * H * 3 // 2
    names = [f"{args.cam}_frame{i}" for i in range(ring)]
    for n in names:
        fm_cap.create(n, fsize)

    tracker = NorfairTracker(cam_cfg, PTZMetrics(autotracker_enabled=False))
    fm_main = SharedMemoryFrameManager()
    cstate = CameraState(args.cam, config, fm_main, None)

    # Save and review uses existing local paths; disabled means no store writes.
    from frigate.secondlook.save_and_review import configure_save_and_review
    from frigate.secondlook.save_and_review_deposit import save_track_for_review
    if configure_save_and_review(config.secondlook.save_and_review) is not None:
        log.info("save-and-review store db=%s", config.secondlook.save_and_review.database.path)

    # --- embeddings side ---
    published: list = []

    class Pub:
        def publish(self, payload, topic):
            published.append((topic, payload))

    class Sub:
        item = None

        def check_for_update(self):
            it, self.item = self.item, None
            return ("video", it) if it is not None else (None, None)

    procs = []
    sl_metrics = SimpleNamespace(secondlook={})
    if config.secondlook.enabled:
        from frigate.data_processing.real_time.secondlook import SecondlookProcessor
        proc = SecondlookProcessor(config, Pub(), sl_metrics)
        # baseline snapshot for the same per-camera rate code stats/util uses
        from frigate.secondlook.metrics import camera_rates
        camera_rates(config, sl_metrics)
        if args.sl_delay_ms > 0:
            orig = proc.model.infer

            def slow(crop, _o=orig):
                time.sleep(args.sl_delay_ms / 1000.0)
                dets, lat = _o(crop)
                return dets, lat + args.sl_delay_ms
            proc.model.infer = slow
        procs.append(proc)
        log.info("SL model: mock=%s device=%s path=%s", proc.model.mock, proc.model.device,
                 proc.model.model_path)
    emb = SimpleNamespace(config=config, detection_subscriber=Sub(),
                          frame_manager=SharedMemoryFrameManager(),
                          realtime_processors=procs, event_metadata_publisher=Pub())
    emb.event_metadata_publisher.publish = lambda payload, topic: published.append((topic, payload))
    tp = SimpleNamespace(camera_states={args.cam: cstate})

    # --- bookkeeping ---
    tracks: dict[str, dict] = {}

    def on_start(cam, obj, frame_name):
        frame_time = obj.obj_data["frame_time"]
        tracks[obj.obj_data["id"]] = {"label": obj.obj_data["label"], "start": frame_time,
                                      "release": None, "held": 0, "factors": [], "lat": [],
                                      "fallbacks": 0, "frames": 0, "end": None}
        log.info("NEW   %s %s score=%.2f", obj.obj_data["id"][-6:], obj.obj_data["label"],
                 obj.obj_data["score"])

    def on_end(cam, obj, frame_name):
        frame_time = obj.obj_data.get("end_time", obj.obj_data["frame_time"])
        oid = obj.obj_data["id"]
        tracks.setdefault(oid, {})["end"] = frame_time
        for p in procs:
            p.expire_object(oid, cam)
        try:
            ex_id = save_track_for_review(obj, config.secondlook)
            if ex_id:
                tracks[oid]["example_id"] = ex_id
                log.info("EXAMPLE %s id=%s", oid[-6:], ex_id)
        except Exception:
            log.exception("example deposit failed")
        log.info("END   %s", oid[-6:])

    cstate.on("start", on_start)
    cstate.on("end", on_end)

    def apply_metadata():
        while published:
            topic, payload = published.pop(0)
            oid, sl = payload
            TrackedObjectProcessor.set_secondlook(tp, oid, sl)
            t = tracks.get(oid)
            if t is not None:
                if sl.get("factor") is not None:
                    t["factors"].append(round(sl["factor"], 3))
                if sl.get("applied_by") == "primary_fallback":
                    t["fallbacks"] += 1
                if not sl.get("pending"):
                    t["lat"].append(sl.get("latency_ms") or 0.0)
            log.info("SLRES %s factor=%s prov=%s sl=%s prim=%s lat=%.1fms by=%s pending=%s",
                     oid[-6:], None if sl["factor"] is None else f"{sl['factor']:.3f}",
                     sl["provisional"], None if sl["sl_score"] is None else f"{sl['sl_score']:.3f}",
                     None if sl["primary_score"] is None else f"{sl['primary_score']:.3f}",
                     sl["latency_ms"] or 0.0, sl["applied_by"], sl["pending"])

    def drain():
        if procs:
            EmbeddingMaintainer._process_deferred_results(emb)
        apply_metadata()

    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-stream_loop", "-1", "-i", args.clip,
         "-vf", f"fps={args.fps},scale={W}:{H}", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, bufsize=fsize * 4)

    t0 = time.time()
    period = 1.0 / args.fps
    nframe = 0
    prim_lat = []
    try:
        while time.time() - t0 < args.seconds:
            raw = ff.stdout.read(fsize)
            if len(raw) < fsize:
                break
            frame_time = time.time()
            slot = nframe % ring
            name = names[slot]
            buf = fm_cap.write(name)
            np.ndarray((H * 3 // 2, W), np.uint8, buffer=buf)[:] = np.frombuffer(raw, np.uint8).reshape(H * 3 // 2, W)
            yuv = fm_main.get(name, cam_cfg.frame_shape_yuv)

            # main loop drains metadata first (TrackedObjectProcessor.run)
            drain()

            # camera process: primary + tracker
            ts = time.perf_counter()
            dets = []
            for r in regions:
                dets += primary_detect(yuv, r)
            dets += inject
            dets = reduce_detections(cam_cfg.frame_shape, dets)
            prim_lat.append((time.perf_counter() - ts) * 1000)
            tracker.match_and_update(name, frame_time, dets)
            detections = {o["id"]: {**o, "attributes": []} for o in tracker.tracked_objects.values()}

            # main: CameraState.update -> TrackedObject.update -> false_positive check
            prev_fp = {k: o.false_positive for k, o in cstate.tracked_objects.items()}
            cstate.update(name, frame_time, detections, [], regions)
            for oid, o in cstate.tracked_objects.items():
                t = tracks.get(oid)
                if t is None:
                    continue
                t["frames"] += 1
                sl = o.obj_data.get("secondlook") or {}
                holding = (o.false_positive and o.secondlook_pending_since is not None
                           and sl.get("factor") is None)
                if holding:
                    t["held"] += 1
                if prev_fp.get(oid, True) and not o.false_positive and t["release"] is None:
                    t["release"] = frame_time
                    log.info("ALERT %s %s released after %.2fs (median raw=%.3f computed=%.3f factor=%s by=%s)",
                             oid[-6:], o.obj_data["label"], frame_time - t["start"],
                             statistics.median(o.score_history), o.computed_score,
                             sl.get("factor"), sl.get("applied_by", "stock"))
                if args.verbose:
                    log.info("FRAME %s %s hit=%s p=%.2f hist_med=%.3f comp=%.3f f=%s pend=%s fp=%s",
                             oid[-6:], o.obj_data["label"], o.obj_data["frame_time"] == frame_time,
                             o.obj_data["score"], statistics.median(o.score_history),
                             o.computed_score, sl.get("factor"),
                             o.secondlook_pending_since is not None, o.false_positive)
            tracked = [o.to_dict() for o in cstate.tracked_objects.values()]

            # embeddings: per-frame update
            emb.detection_subscriber.item = (args.cam, name, frame_time, tracked, [], regions)
            if procs:
                EmbeddingMaintainer._process_frame_updates(emb)

            nframe += 1
            # pace; drain results as they arrive (separate processes in Frigate)
            while time.time() < frame_time + period:
                drain()
                time.sleep(0.005)
    finally:
        # Exercise real tracker disappearance + CameraState end callbacks, not a
        # forced end before the median buffer accumulates missed-frame zeros.
        try:
            for step in range(cam_cfg.detect.max_disappeared * 3 + 30):
                drain()
                expired_time = time.time() + step / args.fps
                tracker.match_and_update(names[0], expired_time, [])
                stale = {o["id"]: {**o, "attributes": []} for o in tracker.tracked_objects.values()}
                cstate.update(names[0], expired_time, stale, [], [])
                if not tracker.tracked_objects:
                    break
                time.sleep(0.01)
        except Exception:
            log.exception("natural track expiry failed")
            raise
        ff.kill()
        for p in procs:
            p.shutdown()
        for n in names:
            fm_cap.delete(n)

    print("\n=== summary ===")
    print(f"secondlook_metrics={sl_metrics.secondlook}")
    if config.secondlook.enabled:
        from frigate.secondlook.metrics import camera_rates, secondlook_stats
        print(f"secondlook_camera_rates={camera_rates(config, sl_metrics)}")
        stats = secondlook_stats(config, sl_metrics)
        print(f"secondlook_accuracy={stats['accuracy']}")
    print(f"clip={args.clip} sl={args.sl} frames={nframe} primary_ms_med={statistics.median(prim_lat):.1f}")
    for oid, t in tracks.items():
        if "label" not in t:
            continue
        rel = f"{t['release'] - t['start']:.2f}s" if t["release"] else "never(FP)"
        lat = f"{statistics.median(t['lat']):.0f}ms" if t["lat"] else "-"
        ex = t.get("example_id")
        ex_s = f" example={ex[:8]}" if ex else ""
        print(f"{oid[-6:]} {t['label']:6s} frames={t['frames']:3d} release={rel:10s} held_frames={t['held']:2d} "
              f"factors={t['factors'][:6]}{'...' if len(t['factors']) > 6 else ''} sl_lat_med={lat} fallbacks={t['fallbacks']}{ex_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
