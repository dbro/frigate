"""Secondlook live QA processor — crop-tied OpenVINO YOLO score adjustment."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from frigate.comms.event_metadata_updater import EventMetadataPublisher
from frigate.config import FrigateConfig
from frigate.config.classification import CameraSecondlookConfig
from frigate.data_processing.real_time.api import DeferredRealtimeProcessorApi
from frigate.data_processing.types import DataProcessorMetrics
from frigate.secondlook.factor import compute_factor
from frigate.secondlook.evidence import source_evidence, finish_evidence
from frigate.secondlook.match import box_iou, select_misclass
from frigate.secondlook.model import SecondlookModel
from frigate.secondlook.model_id import file_md5
from frigate.util.image import calculate_region, yuv_region_2_rgb

logger = logging.getLogger(__name__)


@dataclass
class TrackSecondlookState:
    requested_frame1: bool = False
    in_flight: bool = False
    factor: float | None = None
    provisional: bool = False
    hit_count_in_window: int = 0
    best_hit_score: float = 0.0
    best_hit_crop: np.ndarray | None = None
    best_hit_box_norm: list[float] | None = None
    best_hit_primary: float = 0.0
    last_frame_time: float = 0.0
    best_evidence: dict | None = None
    best_frame_time: float = 0.0


def _required_labels(config: FrigateConfig) -> set[str]:
    """Labels any SL-enabled camera will ask the model about."""
    labels: set[str] = set()
    for camera in config.cameras.values():
        if camera.enabled and camera.secondlook.enabled:
            objects = camera.secondlook.objects
            labels.update(config.secondlook.objects if objects is None else objects)
    return labels


class SecondlookProcessor(DeferredRealtimeProcessorApi):
    """Per-frame crop-tied Secondlook for tracked objects (including FPs)."""

    def __init__(
        self,
        config: FrigateConfig,
        event_metadata_publisher: EventMetadataPublisher,
        metrics: DataProcessorMetrics,
    ) -> None:
        sl = config.secondlook
        super().__init__(config, metrics, max_queue=sl.queue_max)
        self.event_metadata_publisher = event_metadata_publisher
        self.sl_config = sl
        self.model = SecondlookModel(
            model_path=sl.model.path,
            labelmap_path=sl.model.labelmap_path,
            device=sl.device.value if hasattr(sl.device, "value") else str(sl.device),
            input_size=sl.model.input_size,
            plus_metadata_path=sl.model.plus_metadata_path,
            required_labels=_required_labels(config),
            min_score=sl.model.min_score,
            nms_threshold=sl.model.nms_threshold,
            max_detections=sl.model.max_detections,
            nms_class_groups=sl.model.nms_class_groups,
        )
        self._tracks: dict[str, TrackSecondlookState] = {}
        # model_id stays the basename (display label, older rows); model_md5
        # is the identity, hashed once at startup
        self.model_id = os_path_basename(sl.model.path)
        self.model_md5 = file_md5(sl.model.path)
        self._samples = 0
        self._queue_dropped = 0
        self._results_dropped = 0
        # Secondlook items: frames waiting in the bounded queue + the one in
        # process. queue_rises[n] counts how often the items rose to n, so stats
        # can derive each interval's peak without resetting anything here.
        self._queue_items = 0
        self._queue_rises: dict[int, int] = {}
        self._items_lock = threading.Lock()
        # cumulative per-camera counts; rates are derived in stats snapshots
        self._camera_counts: dict[str, dict[str, int]] = {}
        if metrics is not None:
            metrics.secondlook.update(
                available=self.model.available and not self.model.mock,
                device=self.model.device,
                samples=0,
                queue_items=0,
                model_source=self.model.source,
                model_refusal=self.model.refusal,
            )
        self._pending_results = deque(maxlen=max(4, sl.queue_max * 2))

    def _enqueue_task(self, task):
        task["enqueued_at"] = time.perf_counter()
        with self._items_lock:
            # under the lock, so the worker cannot finish (and decrement) first
            accepted = super()._enqueue_task(task)
            if accepted and task.get("kind") == "infer":
                self._change_queue_items(1)
        if not accepted:
            self._queue_dropped += 1
            if self.metrics is not None:
                self.metrics.secondlook["queue_dropped"] = self._queue_dropped
        return accepted

    def _change_queue_items(self, delta: int) -> None:
        """Caller holds _items_lock."""
        self._queue_items += delta
        if delta > 0:
            self._queue_rises[self._queue_items] = (
                self._queue_rises.get(self._queue_items, 0) + 1
            )
        if self.metrics is not None:
            self.metrics.secondlook.update(
                queue_items=self._queue_items, queue_rises=dict(self._queue_rises)
            )

    def _count(self, camera: str, field: str) -> None:
        counts = self._camera_counts.setdefault(
            camera, {"analysed": 0, "skipped": 0}
        )
        counts[field] += 1
        if self.metrics is not None:
            self.metrics.secondlook["cameras"] = {
                name: dict(values) for name, values in self._camera_counts.items()
            }

    def _emit_result(self, result):
        with self._results_lock:
            if (
                self._pending_results.maxlen
                and len(self._pending_results) == self._pending_results.maxlen
            ):
                self._results_dropped += 1
                if self.metrics is not None:
                    self.metrics.secondlook["results_dropped"] = self._results_dropped
            self._pending_results.append(result)

    def _camera_sl(self, camera: str) -> CameraSecondlookConfig | None:
        cam = self.config.cameras.get(camera)
        if cam is None:
            return None
        return cam.secondlook

    def _objects_for(self, camera: str) -> list[str]:
        cam_sl = self._camera_sl(camera)
        if cam_sl is None:
            return []
        if cam_sl.objects is not None:
            return cam_sl.objects
        return self.sl_config.objects

    def process_frame(self, obj_data: dict[str, Any], frame: np.ndarray) -> None:
        """Gating + crop copy + enqueue. Heavy work is on the worker thread.

        Called from EmbeddingMaintainer for each tracked object on each video
        frame (including false positives). Also safe if called from the sparse
        event update path — stationary / ended objects are skipped.
        """
        camera = obj_data.get("camera")
        if not camera:
            return

        cam_sl = self._camera_sl(camera)
        if cam_sl is None or not cam_sl.enabled:
            return

        if not self.config.secondlook.enabled:
            return

        label = obj_data.get("label")
        if label not in self._objects_for(camera):
            return

        if obj_data.get("end_time") is not None:
            return

        if obj_data.get("stationary"):
            return

        object_id = obj_data["id"]
        frame_time = float(obj_data.get("frame_time") or 0.0)
        score = float(obj_data.get("score") or 0.0)
        box = obj_data.get("box")
        if not box or len(box) != 4:
            return

        state = self._tracks.setdefault(object_id, TrackSecondlookState())

        # Only real hits on this frame participate in window / frame-1 crop.
        # Callers pass objects whose frame_time matches the published frame.
        is_hit = score > 0.0

        # A) Frame-1 request
        if not state.requested_frame1 and not state.in_flight and is_hit:
            cropped = self._copy_crop(frame, box)
            if cropped is None:
                self._count(camera, "skipped")
                return
            crop, track_box_norm, region = cropped
            task = {
                "kind": "infer",
                "object_id": object_id,
                "camera": camera,
                "label": label,
                "primary_score": score,
                "frame_time": frame_time,
                "crop": crop,
                "evidence": source_evidence(frame, box, region, frame_time, score),
                "track_box_norm": track_box_norm,
                "provisional": True,
                "reason": "frame1",
            }
            # Stage "pending" before enqueue so it can never land after the
            # worker's final result for this track.
            self._emit_pending(object_id, camera, frame_time)
            if self._enqueue_task(task):
                state.requested_frame1 = True
                state.in_flight = True
            else:
                # Concurrency fail-open
                self._count(camera, "skipped")
                state.requested_frame1 = True
                self._emit_result(
                    {
                        "type": "secondlook",
                        "object_id": object_id,
                        "camera": camera,
                        "factor": None,
                        "provisional": False,
                        "sl_score": None,
                        "primary_score": score,
                        "frame_time": frame_time,
                        "model_id": self.model_id,
                        "model_md5": getattr(self, "model_md5", None),
                        "latency_ms": 0.0,
                        "applied_by": "primary_fallback",
                        "running": False,
                        "pending": False,
                    }
                )
            return

        # D) Refresh window — accumulate real hits; keep max-score crop
        if not is_hit:
            return

        if score >= state.best_hit_score or state.best_hit_crop is None:
            cropped = self._copy_crop(frame, box)
            if cropped is not None:
                state.best_frame_time = frame_time
                state.best_hit_score = score
                state.best_hit_primary = score
                state.best_hit_crop, state.best_hit_box_norm, region = cropped
                state.best_evidence = source_evidence(
                    frame, box, region, frame_time, score
                )

        state.hit_count_in_window += 1
        state.last_frame_time = frame_time

        if (
            state.hit_count_in_window >= self.sl_config.refresh_every_n
            and not state.in_flight
            and state.best_hit_crop is not None
        ):
            task = {
                "kind": "infer",
                "object_id": object_id,
                "camera": camera,
                "label": label,
                "primary_score": state.best_hit_primary,
                "frame_time": state.best_frame_time,
                "evidence": state.best_evidence,
                "crop": state.best_hit_crop,
                "track_box_norm": state.best_hit_box_norm,
                "provisional": False,
                "reason": "refresh",
            }
            if self._enqueue_task(task):
                state.in_flight = True
                # reset window after successful enqueue
                state.hit_count_in_window = 0
                state.best_hit_score = 0.0
                state.best_hit_primary = 0.0
                state.best_hit_crop = None
            else:
                # fail-open for this refresh; keep existing factor
                self._count(camera, "skipped")
                state.hit_count_in_window = 0
                state.best_hit_score = 0.0
                state.best_hit_primary = 0.0
                state.best_hit_crop = None
                logger.debug(
                    "Secondlook refresh queue full for %s — keeping prior factor",
                    object_id,
                )

    def _copy_crop(
        self, frame: np.ndarray, box: list[int]
    ) -> tuple[np.ndarray, list[float], tuple[int, int, int, int]] | None:
        """Crop track box via Frigate calculate_region for SL -> RGB uint8.

        Uses the same helper Frigate uses for detect regions: square pad
        around the track box with context_scale as multiplier, floored at
        the SL model input size so tiny boxes get more scene context rather
        than an upscaled postage stamp. The frame is I420 (height * 3/2
        rows), so region math uses luma (height, width). yuv_region_2_rgb
        pads with black if the region exceeds the frame.
        """
        try:
            xmin, ymin, xmax, ymax = [int(v) for v in box]
            height = frame.shape[0] * 2 // 3
            width = frame.shape[1]
            region = calculate_region(
                (height, width),
                xmin,
                ymin,
                xmax,
                ymax,
                self.model.input_size,
                self.sl_config.context_scale,
            )
            if region[2] - region[0] < 4 or region[3] - region[1] < 4:
                return None
            crop = yuv_region_2_rgb(frame, region)
            if crop is None or crop.size == 0:
                return None
            # track box in crop-normalized [ymin, xmin, ymax, xmax] (SL box_norm
            # convention); crop is square with side = region height.
            side = float(crop.shape[0])
            track_box_norm = [
                (ymin - region[1]) / side,
                (xmin - region[0]) / side,
                (ymax - region[1]) / side,
                (xmax - region[0]) / side,
            ]
            return np.ascontiguousarray(crop), track_box_norm, region
        except Exception:
            logger.debug("Secondlook crop failed", exc_info=True)
            return None

    def _emit_pending(self, object_id: str, camera: str, frame_time: float) -> None:
        self._emit_result(
            {
                "type": "secondlook",
                "object_id": object_id,
                "camera": camera,
                "factor": None,
                "provisional": True,
                "sl_score": None,
                "primary_score": None,
                "frame_time": frame_time,
                "model_id": self.model_id,
                "model_md5": getattr(self, "model_md5", None),
                "latency_ms": 0.0,
                "applied_by": "secondlook_pending",
                "running": True,
                "pending": True,
            }
        )

    def _process_task(self, task: Any) -> None:
        if not isinstance(task, dict):
            return
        kind = task.get("kind")
        if kind == "expire":
            self._tracks.pop(task["object_id"], None)
            return
        if kind != "infer":
            return
        try:
            self._analyse(task)
        finally:
            # an item leaves the Secondlook queue only once it was processed
            with self._items_lock:
                self._change_queue_items(-1)

    def _analyse(self, task: dict[str, Any]) -> None:
        started = time.perf_counter()
        object_id = task["object_id"]
        camera = task["camera"]
        label = task["label"]
        primary_score = float(task["primary_score"])
        frame_time = float(task["frame_time"])
        crop = task["crop"]
        provisional = bool(task.get("provisional", False))

        # fail_open: no real inference behind this answer (model refused or
        # unavailable, or inference / post-processing raised); the primary
        # decides alone, as with no factor
        fail_open = not getattr(self.model, "available", True) or bool(
            getattr(self.model, "mock", False)
        )
        try:
            detections, latency_ms = self.model.infer(crop)
        except Exception:
            logger.exception("Secondlook inference failed for %s", object_id)
            detections, latency_ms, fail_open = [], 0.0, True
        if getattr(self.model, "last_failed", False):
            fail_open = True
        min_iou = self.sl_config.min_box_iou
        track_box_norm = task.get("track_box_norm")
        if min_iou is not None and track_box_norm is not None:
            # optional: only SL boxes that overlap this track's own box count,
            # so another same-class object inside the context pad can't vouch
            detections = [
                d
                for d in detections
                if box_iou(d["box_norm"], track_box_norm) >= min_iou
            ]
        best = self.model.best_same_class(
            detections, label, self.sl_config.score_threshold
        )

        factor = None
        sl_score = None
        sl_box = None
        applied_by = "primary_fallback"
        # Misclass is review-only and does not change factor / no-hit / override.
        misclass_label, misclass_score, misclass_box = select_misclass(
            detections,
            label,
            best,
            self.sl_config.score_threshold,
            track_box_norm,
        )

        if best is not None:
            sl_score = float(best["score"])
            sl_box = best.get("box_norm")
            factor = compute_factor(
                sl_score,
                primary_score,
                factor_min=self.sl_config.factor_min,
                factor_max=self.sl_config.factor_max,
            )
            if factor is not None:
                applied_by = "secondlook_scale"
        elif (
            self.sl_config.no_hit_factor is not None
            and not self.model.mock
            and not fail_open
        ):
            # optional: SL saw no same-class object on the crop -> scale down
            factor = self.sl_config.no_hit_factor
            applied_by = "secondlook_no_hit"

        state = self._tracks.get(object_id)
        if state is not None:
            state.in_flight = False
            if factor is not None:
                state.factor = factor
                state.provisional = provisional

        logger.debug(
            "Secondlook %s %s label=%s primary=%.3f sl=%s factor=%s lat=%.1fms (%s)",
            task.get("reason"),
            object_id,
            label,
            primary_score,
            f"{sl_score:.3f}" if sl_score is not None else None,
            f"{factor:.3f}" if factor is not None else None,
            latency_ms,
            applied_by,
        )

        if self.metrics is not None and self.model.available and not self.model.mock:
            self._count(camera, "analysed")
            self._samples += 1
            self.metrics.secondlook.update(
                samples=self._samples,
                last_updated=time.time(),
                inference_ms=latency_ms,
                processing_ms=(time.perf_counter() - started) * 1000,
                queue_ms=max(0, (started - task.get("enqueued_at", started)) * 1000),
            )
        self._emit_result(
            {
                "type": "secondlook",
                "object_id": object_id,
                "camera": camera,
                "factor": factor,
                "provisional": provisional if factor is not None else False,
                "sl_score": sl_score,
                "sl_box": sl_box,
                "primary_score": primary_score,
                "frame_time": frame_time,
                "model_id": self.model_id,
                "model_md5": getattr(self, "model_md5", None),
                "latency_ms": latency_ms,
                "applied_by": applied_by,
                "running": False,
                "pending": False,
                # a real analysis ran (not a queue-full fail-open)
                "analysed": True,
                "fail_open": fail_open,
                "misclass_label": misclass_label,
                "misclass_score": misclass_score,
                "misclass_box": misclass_box,
                "evidence": finish_evidence(
                    task.get("evidence") if not self.model.mock else None,
                    misclass_box if misclass_label else sl_box,
                    misclass_label or label,
                    misclass_score if misclass_label else sl_score,
                    self.model_id,
                    camera=camera,
                    primary_label=label,
                ),
            }
        )

    def handle_request(self, topic: str, request_data: dict[str, Any]) -> dict | None:
        return None

    def expire_object(self, object_id: str, camera: str) -> None:
        self._enqueue_task({"kind": "expire", "object_id": object_id})
        # also drop locally so we don't keep growing if queue is busy
        self._tracks.pop(object_id, None)

    CONFIG_UPDATE_TOPIC = "config/secondlook"

    def update_config(self, topic: str, payload: Any) -> None:
        if topic != self.CONFIG_UPDATE_TOPIC:
            return
        if payload is not None:
            self.sl_config = payload
            self.config.secondlook = payload



def os_path_basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]
