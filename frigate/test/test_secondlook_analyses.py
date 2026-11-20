"""Per-object Secondlook analyses: one row per track for every outcome,
idempotent writes, hourly reconciliation, model MD5s, migration from the
earlier schema (before the analyses table), retention and the summary API."""

import hashlib
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from peewee import DoesNotExist

from frigate.api.auth import get_allowed_cameras_for_filter
from frigate.api.save_and_review import router
from frigate.config.classification import (
    SecondlookSaveAndReviewConfig,
    SecondlookSaveAndReviewRetentionConfig,
)
from frigate.data_processing.real_time.secondlook import SecondlookProcessor
from frigate.models import Event
from frigate.secondlook import analyses, hourly
from frigate.secondlook import late as secondlook_late
from frigate.secondlook import metrics as sl_metrics
from frigate.secondlook.model_id import file_md5
from frigate.secondlook.retention import run_retention
from frigate.secondlook.save_and_review import SaveAndReviewStore, configure_save_and_review
from frigate.test.test_save_and_review import make_config
from frigate.test.test_secondlook_outcomes import new_track, step
from frigate.track.object_processing import (
    TrackedObjectProcessor,
    finish_expired_secondlook_tracks,
    finish_secondlook_track,
)

MD5 = "0123456789abcdef0123456789abcdef"

# examples + hourly_summary exactly as the build before the analyses table created them
OLD_SCHEMA = """CREATE TABLE IF NOT EXISTS examples (
    id TEXT PRIMARY KEY,
    track_id TEXT NOT NULL,
    camera TEXT NOT NULL,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    misclass_label TEXT,
    primary_score REAL,
    sl_score REAL,
    factor REAL,
    threshold REAL,
    raw_median REAL,
    box TEXT,
    sl_box TEXT,
    frame_time REAL,
    model_id TEXT,
    primary_model_id TEXT,
    still_path TEXT NOT NULL,
    event_id TEXT,
    has_clip INTEGER NOT NULL DEFAULT 0,
    zones TEXT,
    mark TEXT,
    created_at REAL NOT NULL,
    marked_at REAL
);
CREATE INDEX IF NOT EXISTS idx_examples_camera_time ON examples(camera, frame_time);
CREATE INDEX IF NOT EXISTS idx_examples_kind ON examples(kind);
CREATE INDEX IF NOT EXISTS idx_examples_mark ON examples(mark);
CREATE INDEX IF NOT EXISTS idx_examples_label ON examples(label);
-- Corrections order and time window (expression index; no data change)
CREATE INDEX IF NOT EXISTS idx_examples_time ON examples(COALESCE(frame_time, created_at));
-- hourly disagreement summary (secondlook/hourly.py); separate from examples
CREATE TABLE IF NOT EXISTS hourly_summary (
    hour_start INTEGER NOT NULL,
    local_hour TEXT NOT NULL,
    camera TEXT NOT NULL,
    label TEXT NOT NULL,
    objects_analysed INTEGER NOT NULL DEFAULT 0,
    false_positive INTEGER NOT NULL DEFAULT 0,
    false_negative INTEGER NOT NULL DEFAULT 0,
    misclass INTEGER NOT NULL DEFAULT 0,
    fn_rescued INTEGER NOT NULL DEFAULT 0,
    fn_not_rescued INTEGER NOT NULL DEFAULT 0,
    fp_suppressed INTEGER NOT NULL DEFAULT 0,
    fp_not_suppressed INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (hour_start, camera, label)
);
"""


def final(factor, applied_by="secondlook_scale", **extra):
    return {
        "factor": factor,
        "sl_score": 0.8 if applied_by == "secondlook_scale" else None,
        "primary_score": 0.5,
        "applied_by": applied_by,
        "analysed": True,
        "model_id": "sl.onnx",
        "model_md5": MD5,
        "latency_ms": 12.5,
        **extra,
    }


PENDING = {"pending": True, "applied_by": "secondlook_pending", "model_id": "sl.onnx", "model_md5": MD5}
DROPPED = {"factor": None, "applied_by": "primary_fallback", "pending": False,
           "model_id": "sl.onnx", "model_md5": MD5}


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = make_config(True, self.tmp.name)
        self.config.model.compute_model_hash()  # as FrigateConfig does at startup
        self.store = configure_save_and_review(self.config.secondlook.save_and_review)
        for key in sl_metrics._accuracy:
            sl_metrics._accuracy[key] = 0
        secondlook_late._parked.clear()

    def tearDown(self):
        configure_save_and_review(SecondlookSaveAndReviewConfig(enabled=False))
        secondlook_late._parked.clear()
        self.tmp.cleanup()

    def end(self, obj):
        finish_secondlook_track(obj, self.config.secondlook)

    def rows(self):
        with sqlite3.connect(self.store.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return {r["track_id"]: dict(r) for r in conn.execute("SELECT * FROM analyses")}

    def totals(self):
        hours = hourly.series(self.store, hours=2)
        return {c: sum(h[c] for h in hours) for c in hourly.COLUMNS}

    def track(self, score, track_id, *payloads, frames=3):
        obj = new_track(self.config, score, track_id)
        for p in payloads:
            obj.apply_secondlook(dict(p))
        for i in range(frames):
            step(obj, 1000.2 + i * 0.2)
        obj.obj_data["end_time"] = 1001.0
        return obj


class TestOneRowPerOutcome(Case):
    def test_each_outcome_once_and_hourly_reconciles(self):
        agree = self.track(0.9, "t-agree", PENDING, final(1.0))
        fp = self.track(0.9, "t-fp", PENDING, final(0.2))
        fn = self.track(0.5, "t-fn", PENDING, final(2.0))
        no_hit = self.track(0.9, "t-nohit", PENDING, final(0.0, "secondlook_no_hit"))
        no_hit_null = self.track(0.9, "t-nohit-null", PENDING, final(None, "primary_fallback"))
        fail_open = self.track(0.9, "t-failopen", PENDING, final(None, "primary_fallback", fail_open=True))
        dropped = self.track(0.9, "t-dropped", PENDING, DROPPED)
        misclass = self.track(0.9, "t-misclass", PENDING, final(None, "primary_fallback", misclass_label="dog", misclass_score=0.7))
        # forced timeout: pending, never answered, finished after the late window
        silent = self.track(0.9, "t-timeout", PENDING, frames=10)
        ended = [agree, fp, fn, no_hit, no_hit_null, fail_open, dropped, misclass]
        for obj in ended + ended:  # finished twice: still one row each
            self.end(obj)
        self.assertTrue(silent.awaiting_secondlook())
        secondlook_late.park("t-timeout", silent, now=time.monotonic() - 1000)
        finish_expired_secondlook_tracks(self.config.secondlook)
        finish_expired_secondlook_tracks(self.config.secondlook)

        rows = self.rows()
        got = {t: (r["outcome"], r["live"], r["live_outcome"]) for t, r in rows.items()}
        self.assertEqual(got, {
            "t-agree": ("result", "agree", None),
            "t-fp": ("result", "false_positive", "fp_suppressed"),
            "t-fn": ("result", "false_negative", "fn_rescued"),
            "t-nohit": ("no_hit", "false_positive", "fp_suppressed"),
            "t-nohit-null": ("no_hit", "none", None),
            "t-failopen": ("fail_open", "none", None),
            "t-dropped": ("dropped", "none", None),
            "t-misclass": ("no_hit", "misclass", None),
            "t-timeout": ("timeout", "none", None),
        })
        self.assertEqual(len(rows), 9)
        r = rows["t-agree"]
        self.assertEqual((r["factor"], r["sl_score"], r["applied_by"]), (1.0, 0.8, "secondlook_scale"))
        self.assertEqual((r["sl_model_md5"], r["sl_model_name"]), (MD5, "sl.onnx"))
        self.assertEqual(r["primary_model_id"], self.config.model.model_hash)
        self.assertEqual(r["event_id"], "t-agree")
        self.assertEqual(r["threshold"], 0.7)
        self.assertAlmostEqual(r["primary_score"], 0.9)
        self.assertAlmostEqual(r["raw_median"], 0.9)
        self.assertEqual(json.loads(r["box"]), [100, 100, 200, 300])
        self.assertEqual(r["size_bucket"], 3)  # 20000 / (640 * 480) = 6.5%: 5-10%
        self.assertEqual((r["start_time"], r["end_time"]), (1000.0, 1001.0))
        self.assertEqual(r["latency_ms"], 12.5)
        self.assertIsNone(rows["t-fp"]["event_id"])  # suppressed: never a detection
        self.assertEqual(rows["t-fn"]["event_id"], "t-fn")
        self.assertIsNone(rows["t-fn"]["raw_median"])  # primary never crossed
        self.assertEqual(rows["t-nohit"]["factor"], 0.0)
        self.assertIsNone(rows["t-nohit-null"]["factor"])
        self.assertEqual(rows["t-misclass"]["misclass_label"], "dog")
        t = rows["t-timeout"]
        self.assertEqual((t["answers"], t["factor"], t["hold_expired"], t["event_id"]),
                         (0, None, 1, "t-timeout"))  # hold over: primary released it

        totals = self.totals()
        self.assertEqual(totals["tracks_finished"], 9)
        self.assertEqual(
            {o: totals[c] for o, c in hourly.ANSWER_COLUMNS.items()},
            {"result": 3, "no_hit": 3, "fail_open": 1, "timeout": 1, "dropped": 1},
        )
        self.assertEqual(totals["agreements"], 1)
        # reconciliation with the counters the hourly table had before
        self.assertEqual(totals["objects_analysed"], 3 + 3 + 1)  # result + no_hit + fail_open
        self.assertEqual(totals["fp_suppressed"], sum(r["live_outcome"] == "fp_suppressed" for r in rows.values()))
        self.assertEqual(totals["fn_rescued"], 1)
        self.assertEqual(totals["misclass"], 1)

    def test_late_answer_and_dropped_then_refresh(self):
        late = self.track(0.5, "t-late", PENDING)
        secondlook_late.park("t-late", late)
        with patch.object(Event, "get", side_effect=DoesNotExist):
            TrackedObjectProcessor.set_secondlook(
                SimpleNamespace(camera_states={}, config=self.config), "t-late", final(2.0)
            )
        # first request dropped, a refresh answered later in the track
        refreshed = self.track(0.9, "t-refresh", PENDING, DROPPED, final(1.0))
        self.end(refreshed)
        rows = self.rows()
        self.assertEqual((rows["t-late"]["outcome"], rows["t-late"]["answer_after_end"],
                          rows["t-late"]["live_outcome"]), ("result", 1, "fn_not_rescued"))
        self.assertEqual((rows["t-refresh"]["outcome"], rows["t-refresh"]["answers"],
                          rows["t-refresh"]["live"]), ("result", 2, "agree"))

    def test_never_asked_has_no_row_and_disabled_store_writes_nothing(self):
        obj = self.track(0.9, "t-never")
        self.end(obj)
        self.assertEqual(self.rows(), {})
        configure_save_and_review(SecondlookSaveAndReviewConfig(enabled=False))
        obj = self.track(0.9, "t-off", PENDING, final(1.0))
        self.end(obj)
        self.assertFalse(analyses.record_track(obj))

    def test_write_is_idempotent_across_restart(self):
        obj = self.track(0.9, "t-agree", PENDING, final(1.0))
        self.end(obj)
        row = analyses.build_row(obj)
        # a new object for the same track (e.g. after a restart) and a reopened store
        reopened = SaveAndReviewStore(self.store.db_path, self.store.images_path)
        self.assertFalse(analyses.write(reopened, row))
        self.assertFalse(analyses.write(self.store, row))
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.totals()["tracks_finished"], 1)

    def test_example_rows_get_the_model_md5(self):
        fp = self.track(0.9, "t-fp", PENDING, final(0.2))
        self.end(fp)
        (item,) = self.store.list()
        self.assertEqual((item.sl_model_md5, item.sl_model_name, item.model_id),
                         (MD5, "sl.onnx", "sl.onnx"))
        self.assertEqual(item.to_dict()["sl_model_md5"], MD5)

    def test_hourly_rows_are_kept_apart_by_model(self):
        a = self.track(0.9, "t-a", PENDING, final(1.0))
        b = self.track(0.9, "t-b", {**PENDING, "model_md5": "f" * 32},
                       final(1.0, model_md5="f" * 32))
        self.end(a)
        self.end(b)
        self.assertEqual(hourly.series(self.store, hours=1, sl_model_md5=MD5)[0]["agreements"], 1)
        self.assertEqual(hourly.series(self.store, hours=1)[0]["agreements"], 2)
        self.assertEqual({m["sl_model_md5"] for m in hourly.models(self.store)}, {MD5, "f" * 32})


class TestProcessorPayloads(unittest.TestCase):
    """The worker's payloads carry the MD5 and say when it failed open."""

    def processor(self, td, model, queue_max=1):
        proc = SecondlookProcessor.__new__(SecondlookProcessor)
        proc.config = make_config(True, td)
        proc.sl_config = proc.config.secondlook
        proc._task_queue = Queue(maxsize=queue_max)
        proc._queue_items = 0
        proc._queue_rises = {}
        proc._items_lock = threading.Lock()
        proc._queue_dropped = 0
        proc._tracks = {}
        proc.model_id = "sl.onnx"
        proc.model_md5 = MD5
        proc._samples = 0
        proc._camera_counts = {}
        proc.emitted = []
        proc._emit_result = proc.emitted.append
        proc.metrics = SimpleNamespace(secondlook={})
        proc.model = model
        return proc

    def model(self, **extra):
        return SimpleNamespace(**{"available": True, "mock": False, "input_size": 320,
                                  "infer": lambda crop: ([], 1.0),
                                  "best_same_class": lambda *a: None, **extra})

    def task(self):
        return {"kind": "infer", "object_id": "x", "camera": "cam", "label": "person",
                "primary_score": 0.8, "frame_time": 1.0, "crop": np.zeros((8, 8, 3))}

    def test_queue_drop_emits_the_fail_open_notice_with_md5(self):
        with tempfile.TemporaryDirectory() as td:
            proc = self.processor(td, self.model(), queue_max=1)
            proc._task_queue.put_nowait({"kind": "expire"})  # queue full
            frame = np.zeros((720, 640), dtype=np.uint8)
            proc.process_frame({"id": "x", "camera": "cam", "label": "person",
                                "frame_time": 1.0, "score": 0.8,
                                "box": [100, 100, 200, 300]}, frame)
            pending, notice = proc.emitted
            self.assertTrue(pending["pending"])
            self.assertEqual(analyses.answer_kind(notice), "dropped")
            self.assertEqual((notice["model_md5"], pending["model_md5"]), (MD5, MD5))

    def test_refused_model_and_failed_inference_fail_open_without_no_hit_factor(self):
        with tempfile.TemporaryDirectory() as td:
            for model in (
                self.model(mock=True),
                self.model(infer=lambda crop: (_ for _ in ()).throw(RuntimeError("boom"))),
                self.model(last_failed=True),
            ):
                proc = self.processor(td, model)
                proc._analyse(self.task())
                (out,) = proc.emitted
                self.assertTrue(out["fail_open"])
                self.assertIsNone(out["factor"])  # no_hit_factor 0.0 not applied
                self.assertEqual(analyses.answer_kind(out), "fail_open")
            proc = self.processor(td, self.model())
            proc._analyse(self.task())
            (out,) = proc.emitted
            self.assertFalse(out["fail_open"])
            self.assertEqual((out["factor"], out["applied_by"]), (0.0, "secondlook_no_hit"))
            self.assertEqual((analyses.answer_kind(out), out["model_md5"]), ("no_hit", MD5))

    def test_file_md5_matches_hashlib_and_detector_loop(self):
        with tempfile.TemporaryDirectory() as td:
            path = f"{td}/m.onnx"
            data = np.random.default_rng(0).bytes(100_000)
            with open(path, "wb") as f:
                f.write(data)
            self.assertEqual(file_md5(path), hashlib.md5(data).hexdigest())
            self.assertIsNone(file_md5(f"{td}/missing.onnx"))


class TestLiveWiring(unittest.TestCase):
    """Two hand-offs a live boot showed missing (answer keys and the detector model hash)."""

    def test_embeddings_maintainer_forwards_md5_and_fail_open(self):
        from frigate.embeddings.maintainer import EmbeddingMaintainer

        sent = []
        maint = EmbeddingMaintainer.__new__(EmbeddingMaintainer)
        result = {"type": "secondlook", "object_id": "x", **final(None, "primary_fallback", fail_open=True)}
        maint.realtime_processors = [SimpleNamespace(drain_results=lambda: [result])]
        maint.event_metadata_publisher = SimpleNamespace(publish=lambda payload, topic: sent.append(payload))
        maint._process_deferred_results()
        ((object_id, payload),) = sent
        self.assertEqual((object_id, payload["model_md5"], payload["fail_open"]), ("x", MD5, True))
        self.assertEqual(analyses.answer_kind(payload), "fail_open")

    def test_tracked_objects_get_the_detector_model_md5(self):
        from frigate.camera.state import primary_model_hash

        with tempfile.TemporaryDirectory() as td:
            config = make_config(True, td)
            detector = next(iter(config.detectors.values()))
            self.assertEqual(primary_model_hash(config), detector.model.model_hash)
            self.assertRegex(primary_model_hash(config), "^[0-9a-f]{32}$")
            self.assertIsNone(primary_model_hash(SimpleNamespace(detectors={})))


class TestMigrationFromPreAnalysesSchema(unittest.TestCase):
    def test_old_db_opens_keeps_rows_and_copies_hourly_once(self):
        with tempfile.TemporaryDirectory() as td:
            db = f"{td}/secondlook_examples.db"
            conn = sqlite3.connect(db)
            conn.executescript(OLD_SCHEMA)
            conn.execute("ALTER TABLE examples ADD COLUMN evidence TEXT")
            conn.execute("ALTER TABLE examples ADD COLUMN group_key TEXT")
            conn.execute("ALTER TABLE examples ADD COLUMN legacy_mark TEXT")
            now = time.time()
            conn.execute(
                "INSERT INTO examples (id, track_id, camera, kind, label, model_id, "
                "still_path, created_at, frame_time, mark, marked_at) "
                "VALUES ('e1', 't1', 'cam', 'false_positive', 'person', 'old.onnx', "
                "'/nope.jpg', ?, ?, 'primary_right', ?)", (now, now, now))
            hour, local = hourly.hour_bucket(now, None)
            conn.execute(
                "INSERT INTO hourly_summary VALUES (?, ?, 'cam', 'person', 40, 3, 2, 1, 2, 0, 2, 1)",
                (hour, local))
            conn.commit()
            conn.close()
            for _ in range(3):  # reopening is idempotent
                store = SaveAndReviewStore(db, f"{td}/img")
            rec = store.get("e1")
            self.assertEqual((rec.model_id, rec.sl_model_md5, rec.mark), ("old.onnx", None, "primary_right"))
            (h,) = [x for x in hourly.series(store, hours=1)]
            self.assertEqual((h["objects_analysed"], h["false_positive"], h["fp_not_suppressed"],
                              h["tracks_finished"]), (40, 3, 1, 0))
            self.assertEqual(hourly.models(store)[0]["sl_model_md5"], "")
            hourly.add(store, "cam", "person", hourly.ANALYSED, now=now, sl_model_md5=MD5)
            self.assertEqual(hourly.series(store, hours=1)[0]["objects_analysed"], 41)
            self.assertEqual(hourly.series(store, hours=1, sl_model_md5="")[0]["objects_analysed"], 40)
            with sqlite3.connect(db) as c:
                # the old table is left as it was
                self.assertEqual(c.execute("SELECT objects_analysed FROM hourly_summary").fetchone()[0], 40)
                c.execute("DELETE FROM hourly_counts")  # as retention would
            SaveAndReviewStore(db, f"{td}/img")  # never copied back
            self.assertEqual(hourly.series(store, hours=1)[0]["objects_analysed"], 0)


class TestRetention(Case):
    def test_analyses_older_than_analyses_days(self):
        now = time.time()
        for i, days in enumerate((120, 91, 89, 1)):
            obj = self.track(0.9, f"t{i}", PENDING, final(1.0))
            row = analyses.build_row(obj, now=now)
            row["end_time"] = now - days * 86400
            analyses.write(self.store, row)
        keep = SecondlookSaveAndReviewRetentionConfig(analyses_days=0)
        self.assertEqual(run_retention(self.store, keep, now=now)["deleted_analyses_rows"], 0)
        out = run_retention(self.store, SecondlookSaveAndReviewRetentionConfig(), now=now)
        self.assertEqual(out["deleted_analyses_rows"], 2)
        self.assertEqual(sorted(self.rows()), ["t2", "t3"])
        short = SecondlookSaveAndReviewRetentionConfig(analyses_days=30)
        self.assertEqual(run_retention(self.store, short, now=now)["deleted_analyses_rows"], 1)


class TestSummaryAPI(Case):
    def test_counts_histograms_filters_and_cameras(self):
        for tid, payloads in (
            ("a", (PENDING, final(1.0))),
            ("b", (PENDING, final(0.2))),
            ("c", (PENDING, DROPPED)),
            ("d", (PENDING, final(0.0, "secondlook_no_hit"))),
        ):
            self.end(self.track(0.9, tid, *payloads))
        row = analyses.build_row(self.track(0.9, "hidden", PENDING, final(1.0)))
        row["camera"] = "hidden"
        analyses.write(self.store, row)
        app = FastAPI()
        app.frigate_config = self.config
        app.dependency_overrides[get_allowed_cameras_for_filter] = lambda: ["cam"]
        app.include_router(router)
        with TestClient(app) as client:
            body = client.get("/secondlook/analyses/summary").json()
            self.assertEqual(body["total"], 4)
            self.assertEqual(body["outcomes"], {"result": 2, "no_hit": 1, "fail_open": 0,
                                                "timeout": 0, "dropped": 1})
            self.assertEqual(body["live"]["agree"], 1)
            self.assertEqual(body["live_outcomes"], {"fp_suppressed": 2})
            fh = body["factor_histogram"]
            self.assertEqual(sum(fh["counts"]) + fh["none"], 4)
            self.assertEqual((fh["counts"][0], fh["counts"][4], fh["none"]), (2, 1, 1))  # 0.0, 0.2 | 1.0 | none
            sh = body["sl_score_histogram"]
            self.assertEqual((sh["counts"][16], sh["none"]), (2, 2))  # 0.8 -> 0.80-0.85
            self.assertEqual(body["primary_score_histogram"]["counts"][18], 4)  # 0.9
            self.assertEqual(body["size_buckets"]["counts"][3], 4)
            self.assertEqual(body["models"][0]["sl_model_md5"], MD5)
            self.assertEqual(client.get("/secondlook/analyses/summary?camera=hidden").json()["total"], 0)
            self.assertEqual(client.get("/secondlook/analyses/summary?outcome=dropped").json()["total"], 1)
            self.assertEqual(client.get(f"/secondlook/analyses/summary?sl_model={MD5}&live=agree").json()["total"], 1)
            self.assertEqual(client.get("/secondlook/analyses/summary?sl_model=x").json()["total"], 0)
            self.assertEqual(client.get("/secondlook/analyses/summary?after=1e12").json()["total"], 0)
            self.assertEqual(client.get("/secondlook/analyses/summary?outcome=bad").status_code, 400)
            h = client.get("/secondlook/disagreements/hourly").json()
            self.assertEqual(sum(x["tracks_finished"] for x in h["hours"]), 4)
            self.assertEqual(sum(x["dropped"] for x in h["hours"]), 1)
            self.assertEqual(h["models"][0]["sl_model_md5"], MD5)
            h = client.get("/secondlook/disagreements/hourly?sl_model=other").json()
            self.assertEqual(sum(x["tracks_finished"] for x in h["hours"]), 0)


if __name__ == "__main__":
    unittest.main()
