"""Decision/provenance/API regression tests against real local pipeline pieces."""

import base64
import datetime
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from playhouse.sqlite_ext import SqliteExtDatabase

from frigate.secondlook.evidence import source_evidence, finish_evidence, full_frame_box
from frigate.secondlook.save_and_review import (
    configure_save_and_review,
    SaveAndReviewStore,
)
from frigate.secondlook.save_and_review_deposit import save_track_for_review
from frigate.test.test_save_and_review import (
    make_config,
    make_track,
    TestSaveAndReviewAPI,
)


class TestDecisionHistory(unittest.TestCase):
    def test_suppression_survives_missed_frames(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(True, td)
            store = configure_save_and_review(config.secondlook.save_and_review)
            obj = make_track(config)
            evidence = finish_evidence(
                source_evidence(
                    np.zeros((720, 640), np.uint8),
                    [100, 100, 200, 300],
                    [0, 0, 640, 640],
                    1000,
                    0.9,
                ),
                [0.1, 0.1, 0.4, 0.4],
                "person",
                0.1,
                "sl",
            )
            obj.apply_secondlook({"factor": 0.1, "evidence": evidence, "sl_score": 0.1})
            data = dict(obj.obj_data)
            data["frame_time"] = 1001
            obj.update(1001, data, False)
            for i in range(20):
                obj.update(1002 + i, data, False)
            self.assertEqual(obj.score_history, [0.0] * 10)
            rec = store.get(save_track_for_review(obj, config.secondlook))
            self.assertEqual(rec.kind, "false_positive")
            self.assertEqual(rec.frame_time, 1000)
            self.assertEqual(rec.evidence["width"], 640)
            self.assertEqual(
                Path(rec.still_path).read_bytes(), base64.b64decode(evidence["jpeg"])
            )

    def test_fn_evidence_survives_and_stock_tp_does_not_become_fn(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(True, td)
            store = configure_save_and_review(config.secondlook.save_and_review)
            fn = make_track(config, 0.5, 2)
            data = dict(fn.obj_data)
            for i in range(20):
                fn.update(1001 + i, data, False)
            self.assertEqual(
                store.get(save_track_for_review(fn, config.secondlook)).kind,
                "false_negative",
            )
            tp = make_track(config, 0.9, 1)
            tp.apply_secondlook({"factor": 2})
            data = dict(tp.obj_data)
            for i in range(20):
                tp.update(1001 + i, data, False)
            self.assertIsNone(save_track_for_review(tp, config.secondlook))

    def test_crop_padding_non_square_and_coherent_source(self):
        self.assertEqual(
            full_frame_box([0.1, 0.2, 0.9, 0.8], [-100, -50, 300, 350], 640, 480),
            [0, 0, 220.0, 310.0],
        )
        src = source_evidence(
            np.zeros((720, 640), np.uint8),
            [10, 20, 30, 40],
            [-100, -50, 300, 350],
            12,
            0.8,
        )
        result = finish_evidence(src, [0.1, 0.2, 0.9, 0.8], "truck", 0.9, "model")
        self.assertEqual(result["frame_time"], 12)
        self.assertEqual((result["width"], result["height"]), (640, 480))
        self.assertEqual(result["primary_box"], [10, 20, 30, 40])


class TestWindowProvenance(unittest.TestCase):
    def test_refresh_uses_maximum_not_last_frame_with_delayed_result(self):
        from frigate.data_processing.real_time.secondlook import SecondlookProcessor

        with tempfile.TemporaryDirectory() as td:
            config = make_config(True, td)
            config.secondlook.refresh_every_n = 2
            proc = SecondlookProcessor.__new__(SecondlookProcessor)
            proc.config = config
            proc.sl_config = config.secondlook
            proc.model = SimpleNamespace(input_size=640)
            proc._tracks = {}
            proc.model_id = "sl"
            tasks = []
            proc._emit_pending = lambda *a: None
            proc._enqueue_task = lambda t: tasks.append(t) or True
            image = np.zeros((720, 640), np.uint8)
            obj = {
                "id": "track",
                "camera": "cam",
                "label": "person",
                "score": 0.8,
                "box": [10, 20, 100, 200],
                "frame_time": 1,
            }
            proc.process_frame(obj, image)
            proc._tracks["track"].in_flight = False
            proc.process_frame({**obj, "score": 0.95, "frame_time": 2}, image)
            image[:, :] = 100
            proc.process_frame(
                {**obj, "score": 0.7, "frame_time": 3, "box": [300, 20, 400, 200]},
                image,
            )
            task = tasks[-1]
            self.assertEqual(task["frame_time"], 2)
            self.assertEqual(task["evidence"]["frame_time"], 2)
            self.assertEqual(task["evidence"]["primary_box"], obj["box"])
            result = finish_evidence(
                task["evidence"], [0.1, 0.1, 0.9, 0.9], "person", 0.9, "sl"
            )
            self.assertEqual(result["frame_time"], 2)
            self.assertEqual(result["jpeg"], task["evidence"]["jpeg"])


class TestPagination(TestSaveAndReviewAPI):
    def test_all_history_counts_dates_zones_and_over_200(self):
        for i in range(221):
            self.store.deposit(
                track_id=str(i),
                camera="cam",
                kind="misclass",
                label="car",
                misclass_label="truck",
                jpeg_bytes=b"jpeg",
                frame_time=i,
                zones=["inside" if i % 2 else "outside"],
            )
        url = "/secondlook/save_and_review"
        first = self.client.get(url, params={"kind": "misclass", "limit": 50}).json()
        self.assertEqual(first["total"], 221)
        self.assertEqual(first["unreviewed"], 221)
        ids = []
        for offset in range(0, 221, 50):
            page = self.client.get(
                url, params={"kind": "misclass", "limit": 50, "offset": offset}
            ).json()
            ids.extend(r["id"] for r in page["examples"])
        self.assertEqual(len(set(ids)), 221)
        self.client.post(f"{url}/{ids[0]}/mark", json={"mark": "ignore"})
        filtered = self.client.get(
            url,
            params={
                "kind": "misclass",
                "after": 100,
                "before": 200,
                "zones": "inside",
                "reviewed": "false",
            },
        ).json()
        self.assertEqual(filtered["total"], 50)
        self.assertEqual(filtered["unreviewed"], 50)
        self.assertEqual(len(filtered["examples"]), 50)
        self.assertFalse(filtered["has_more"])


class TestRetention(unittest.TestCase):
    def test_real_stock_record_event_review_cleanup_preserves_examples(self):
        from frigate.models import (
            Event,
            Timeline,
            Recordings,
            ReviewSegment,
            UserReviewStatus,
            RecordingsToDelete,
            Previews,
        )
        from frigate.events.cleanup import EventCleanup
        from frigate.record.cleanup import RecordingCleanup

        models = [
            Event,
            Timeline,
            Recordings,
            ReviewSegment,
            UserReviewStatus,
            RecordingsToDelete,
            Previews,
        ]
        with tempfile.TemporaryDirectory() as td:
            config = make_config(True, td)
            clips = Path(td) / "clips"
            clips.mkdir()
            config.secondlook.save_and_review.images.path = str(
                clips / "secondlook_examples"
            )
            store = configure_save_and_review(config.secondlook.save_and_review)
            refs = []
            for mark in ("primary_right", "secondlook_right", "ignore", None):
                r = store.deposit(
                    track_id="event",
                    camera="cam",
                    kind="false_negative",
                    label="person",
                    jpeg_bytes=b"independent JPEG",
                    event_id="event",
                    has_clip=True,
                )
                store.set_mark(r.id, mark)
                refs.append((r.id, r.still_path, mark))
            db = SqliteExtDatabase(str(Path(td) / "stock.db"))
            with db.bind_ctx(models):
                db.create_tables(models)
                snapshot = clips / "cam-event.jpg"
                snapshot.write_bytes(b"stock snapshot")
                Event.create(
                    id="event",
                    camera="cam",
                    label="person",
                    start_time=1,
                    end_time=2,
                    top_score=0.9,
                    score=0.9,
                    false_positive=False,
                    zones=[],
                    thumbnail="",
                    region=[],
                    box=[],
                    area=1,
                    plus_id="",
                    model_hash="",
                    detector_type="",
                    model_type="",
                    data={},
                )
                thumb = clips / "review.webp"
                thumb.write_bytes(b"stock review")
                ReviewSegment.create(
                    id="review",
                    camera="cam",
                    start_time=1,
                    end_time=2,
                    severity="alert",
                    thumb_path=str(thumb),
                    data={},
                )
                recording = Path(td) / "old.mp4"
                recording.write_bytes(b"stock recording")
                Recordings.create(
                    id="record",
                    camera="cam",
                    path=str(recording),
                    start_time=1,
                    end_time=2,
                    duration=1,
                    motion=0,
                    objects=0,
                )
                cleanup = RecordingCleanup(config, threading.Event())
                cleanup.expire_review_segments(
                    config.cameras["cam"], datetime.datetime.now(datetime.timezone.utc)
                )
                cleanup.expire_existing_camera_recordings(
                    1e10, 1e10, config.cameras["cam"], []
                )
                stop = Mock()
                stop.wait.side_effect = [False, True]
                with (
                    patch("frigate.events.cleanup.CLIPS_DIR", str(clips)),
                    patch("frigate.util.file.CLIPS_DIR", str(clips)),
                ):
                    EventCleanup(config, stop, Mock()).run()
                self.assertEqual(Event.select().count(), 0)
                self.assertEqual(ReviewSegment.select().count(), 0)
                self.assertEqual(Recordings.select().count(), 0)
                self.assertFalse(snapshot.exists())
                self.assertFalse(thumb.exists())
                self.assertFalse(recording.exists())
            reopened = SaveAndReviewStore(store.db_path, store.images_path)
            for id, path, mark in refs:
                self.assertEqual(reopened.get(id).mark, mark)
                self.assertEqual(Path(path).read_bytes(), b"independent JPEG")
                from fastapi import FastAPI
                from fastapi.testclient import TestClient
                from frigate.api.save_and_review import router

                app = FastAPI()
                app.frigate_config = config
                app.include_router(router)
                with TestClient(app) as client:
                    self.assertEqual(
                        client.get("/secondlook/save_and_review/" + id).json()["mark"],
                        mark,
                    )
                    self.assertEqual(
                        client.get(
                            "/secondlook/save_and_review/" + id + "/still.jpg"
                        ).content,
                        b"independent JPEG",
                    )
            db.close()


class TestExistingData(unittest.TestCase):
    def test_additive_schema_preserves_old_marks_images_and_unavailable_overlay(self):
        import sqlite3
        from frigate.secondlook.save_and_review import _SCHEMA

        with tempfile.TemporaryDirectory() as td:
            image = Path(td) / "old.jpg"
            image.write_bytes(b"original")
            path = str(Path(td) / "old.db")
            with sqlite3.connect(path) as conn:
                conn.executescript(_SCHEMA)
                conn.execute(
                    "INSERT INTO examples (id,track_id,camera,kind,label,still_path,mark,created_at) VALUES ('old','track','cam','misclass','car',?,'reject',1)",
                    (str(image),),
                )
            store = SaveAndReviewStore(path, td)
            rec = store.get("old")
            self.assertEqual(rec.mark, "ignore")
            self.assertEqual(rec.legacy_mark, "reject")
            self.assertIsNone(rec.evidence)
            self.assertEqual(image.read_bytes(), b"original")


class TestLosslessGrouping(unittest.TestCase):
    def test_exact_verdict_not_vague_appearance_and_individual_marks(self):
        with tempfile.TemporaryDirectory() as td:
            store = SaveAndReviewStore(str(Path(td) / "saved.db"), td)
            args = dict(
                camera="cam",
                kind="misclass",
                label="car",
                misclass_label="truck",
                jpeg_bytes=b"exact JPEG",
                primary_score=0.8,
                box=[1, 2, 3, 4],
            )
            a = store.deposit(track_id="a", **args)
            b = store.deposit(track_id="b", **args)
            other = store.deposit(track_id="c", **{**args, "box": [2, 3, 4, 5]})
            self.assertEqual(a.group_key, b.group_key)
            self.assertNotEqual(a.group_key, other.group_key)
            store.set_mark(a.id, "secondlook_right")
            self.assertIsNone(store.get(b.id).mark)
            self.assertEqual(len(store.list()), 3)
            self.assertEqual(Path(b.still_path).read_bytes(), b"exact JPEG")


class TestTimingAndDebug(unittest.TestCase):
    def test_metrics_states_are_not_fake_zero_inferences(self):
        from frigate.secondlook.metrics import secondlook_stats

        with tempfile.TemporaryDirectory() as td:
            config = make_config(False, td)
            metrics = SimpleNamespace(
                secondlook={
                    "available": True,
                    "samples": 1,
                    "inference_ms": 12,
                    "processing_ms": 14,
                    "queue_ms": 3,
                    "last_updated": 123,
                }
            )
            self.assertEqual(secondlook_stats(config, metrics)["inference_ms"], 12)
            config.secondlook.enabled = False
            self.assertIsNone(secondlook_stats(config, metrics)["inference_ms"])
            config.secondlook.enabled = True
            metrics.secondlook["available"] = False
            self.assertIsNone(secondlook_stats(config, metrics)["inference_ms"])
            metrics.secondlook.update(available=True, samples=0)
            self.assertIsNone(secondlook_stats(config, metrics)["inference_ms"])

    def test_real_worker_timing_and_bounded_results(self):
        import time
        from collections import deque
        from frigate.data_processing.real_time.secondlook import SecondlookProcessor

        with tempfile.TemporaryDirectory() as td:
            proc = SecondlookProcessor.__new__(SecondlookProcessor)
            proc.sl_config = make_config(True, td).secondlook
            proc._tracks = {}
            proc.model_id = "test"
            proc._samples = 0
            proc._results_dropped = 0
            proc._camera_counts = {}
            proc.metrics = SimpleNamespace(secondlook={})
            proc._pending_results = deque(maxlen=2)
            proc._results_lock = threading.Lock()
            proc._queue_items = 1
            proc._queue_rises = {}
            proc._items_lock = threading.Lock()

            def infer(crop):
                time.sleep(0.002)
                return [], 1

            proc.model = SimpleNamespace(
                available=True, mock=False, infer=infer, best_same_class=lambda *a: None
            )
            task = {
                "kind": "infer",
                "object_id": "x",
                "camera": "cam",
                "label": "person",
                "primary_score": 0.8,
                "frame_time": 1,
                "crop": np.zeros((10, 10, 3)),
                "enqueued_at": time.perf_counter() - 0.01,
            }
            proc._process_task(task)
            self.assertGreaterEqual(proc.metrics.secondlook["queue_ms"], 10)
            self.assertGreaterEqual(proc.metrics.secondlook["processing_ms"], 2)
            self.assertEqual(proc.metrics.secondlook["inference_ms"], 1)
            for i in range(10):
                proc._emit_result({"n": i})
            self.assertEqual(len(proc._pending_results), 2)
            self.assertEqual(proc.metrics.secondlook["results_dropped"], 9)
            self.assertEqual(proc.metrics.secondlook["samples"], 1)
            proc.model.mock = True
            proc._process_task(task)
            self.assertEqual(proc.metrics.secondlook["samples"], 1)

    def test_debug_independent_of_saving_and_camera_authorized(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from frigate.api.auth import get_allowed_cameras_for_filter
        from frigate.api.secondlook_debug import router

        with tempfile.TemporaryDirectory() as td:
            config = make_config(False, td)
            app = FastAPI()
            app.frigate_config = config
            state = SimpleNamespace(latest_secondlook=None)
            app.detected_frames_processor = SimpleNamespace(
                camera_states={"cam": state}
            )
            app.dependency_overrides[get_allowed_cameras_for_filter] = lambda: ["cam"]
            app.include_router(router)
            with TestClient(app) as client:
                self.assertEqual(client.get("/other/secondlook").status_code, 403)
                self.assertEqual(
                    client.get("/cam/secondlook").json()["state"], "no_result"
                )
                state.latest_secondlook = {"jpeg": "bounded-image", "frame_time": 1}
                self.assertEqual(
                    client.get("/cam/secondlook").json()["frame"],
                    state.latest_secondlook,
                )
                config.secondlook.enabled = False
                self.assertEqual(
                    client.get("/cam/secondlook").json(),
                    {"state": "disabled", "frame": None},
                )
                config.secondlook.enabled = True
                config.cameras["cam"].secondlook.enabled = False
                self.assertEqual(
                    client.get("/cam/secondlook").json()["state"], "disabled"
                )

    def test_manager_exports_worker_values_across_processes(self):
        from multiprocessing import Manager, Process
        from frigate.data_processing.types import DataProcessorMetrics
        from frigate.secondlook.metrics import secondlook_stats

        with Manager() as manager, tempfile.TemporaryDirectory() as td:
            metrics = DataProcessorMetrics(manager, [])
            worker = Process(
                target=metrics.secondlook.update,
                kwargs={"available": True, "samples": 1, "inference_ms": 12},
            )
            worker.start()
            worker.join(5)
            if worker.is_alive():
                worker.terminate()
                worker.join()
                self.fail("metrics worker hung")
            self.assertEqual(worker.exitcode, 0)
            self.assertEqual(
                secondlook_stats(make_config(False, td), metrics)["inference_ms"], 12
            )

    def test_failed_deposit_is_diagnosed_without_mutating_existing_data(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(True, td)
            store = configure_save_and_review(config.secondlook.save_and_review)
            with patch.object(
                store, "deposit", side_effect=OSError("simulated disk full")
            ):
                with self.assertRaises(OSError):
                    save_track_for_review(make_track(config), config.secondlook)
            self.assertEqual(store.diagnostics["deposit_failed"], 1)
            self.assertEqual(store.list(), [])

    def test_older_debug_result_cannot_replace_newer_analysed_frame(self):
        from peewee import DoesNotExist
        from frigate.track.object_processing import TrackedObjectProcessor
        from frigate.models import Event

        obj = SimpleNamespace(obj_data={"label": "person"}, apply_secondlook=Mock())
        state = SimpleNamespace(tracked_objects={"track": obj}, latest_secondlook=None)
        processor = SimpleNamespace(camera_states={"cam": state})
        with patch.object(Event, "get", side_effect=DoesNotExist):
            TrackedObjectProcessor.set_secondlook(
                processor, "track", {"evidence": {"frame_time": 20, "jpeg": "new"}}
            )
            TrackedObjectProcessor.set_secondlook(
                processor, "track", {"evidence": {"frame_time": 10, "jpeg": "old"}}
            )
        self.assertEqual(state.latest_secondlook["jpeg"], "new")

    def test_non_sl_camera_does_not_pollute_decision_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(True, td)
            config.cameras["cam"].secondlook.enabled = False
            store = configure_save_and_review(config.secondlook.save_and_review)
            make_track(config, 0.9, 1)
            self.assertEqual(dict(store.diagnostics), {})

    def test_latest_debug_result_survives_track_expiry_during_inference(self):
        from peewee import DoesNotExist
        from frigate.track.object_processing import TrackedObjectProcessor
        from frigate.models import Event

        state = SimpleNamespace(tracked_objects={}, latest_secondlook=None)
        processor = SimpleNamespace(camera_states={"cam": state})
        evidence = {
            "camera": "cam",
            "label": "person",
            "jpeg": "final-still",
            "frame_time": 30,
        }
        with patch.object(Event, "get", side_effect=DoesNotExist):
            TrackedObjectProcessor.set_secondlook(
                processor, "expired", {"evidence": evidence}
            )
        self.assertEqual(state.latest_secondlook, evidence)
