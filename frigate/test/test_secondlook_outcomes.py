"""Live outcome of Secondlook disagreements: rescued / suppressed or not, late
answers, once per track, hourly migration and the Review Event lookup."""

import sqlite3
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from peewee import DoesNotExist, SqliteDatabase

from frigate.api.auth import get_allowed_cameras_for_filter
from frigate.api.save_and_review import router
from frigate.config.classification import SecondlookSaveAndReviewConfig
from frigate.models import Event
from frigate.secondlook import hourly
from frigate.secondlook import late as secondlook_late
from frigate.secondlook import metrics as sl_metrics
from frigate.secondlook.live_outcome import event_retention_days, live_outcomes
from frigate.secondlook.save_and_review import (
    SaveAndReviewStore,
    configure_save_and_review,
)
from frigate.test.test_save_and_review import make_config
from frigate.track.object_processing import (
    TrackedObjectProcessor,
    finish_expired_secondlook_tracks,
    finish_secondlook_track,
)
from frigate.track.tracked_object import TrackedObject

import numpy as np


def new_track(config, score, track_id="1000.0-abcdef"):
    data = {
        "id": track_id,
        "label": "person",
        "frame_time": 1000.0,
        "start_time": 1000.0,
        "score": score,
        "box": [100, 100, 200, 300],
        "area": 20000,
        "ratio": 0.5,
        "region": [0, 0, 640, 480],
        "centroid": [150, 200],
        "position_changes": 1,
        "motionless_count": 0,
        "attributes": [],
    }
    obj = TrackedObject(
        config.model, config.cameras["cam"], config.ui, {}, {**data, "score_history": [score] * 10}
    )
    obj.capture_save_and_review_frame(np.zeros((720, 640), dtype=np.uint8))
    return obj


def step(obj, t):
    data = dict(obj.obj_data)
    data["frame_time"] = t
    obj.update(t, data, False)


def answer(obj, factor, **extra):
    obj.apply_secondlook(
        {"factor": factor, "primary_score": obj.obj_data["score"], "sl_score": 0.5,
         "analysed": True, **extra}
    )


class OutcomeCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = make_config(True, self.tmp.name)
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

    def hour(self):
        return hourly.series(self.store, hours=1)[0]

    def items(self):
        return {(r.track_id, r.kind, r.event_id) for r in self.store.list()}


class TestOutcomes(OutcomeCase):
    def test_four_outcomes_counted_once_and_saved_for_review(self):
        # FN rescued: primary 0.5 never crosses 0.7, SL x2 releases it
        rescued = new_track(self.config, 0.5, "1.0-rescued")
        answer(rescued, 2.0)
        step(rescued, 1000.2)
        # FN not rescued: x2 would release it, but the answer came after the
        # last frame of the track
        weak = new_track(self.config, 0.5, "1.0-weak")
        answer(weak, 2.0)
        # FP suppressed: primary 0.9 would release, SL x0 holds it back
        suppressed = new_track(self.config, 0.9, "1.0-suppressed")
        answer(suppressed, 0.0)
        step(suppressed, 1000.2)
        # FP not suppressed: no answer within hold_seconds -> released, then x0
        slow = new_track(self.config, 0.9, "1.0-slow")
        step(slow, 1000.2)
        step(slow, 1001.5)
        self.assertFalse(slow.false_positive)
        answer(slow, 0.0)
        step(slow, 1001.7)

        for obj in (rescued, weak, suppressed, slow, rescued, slow):  # end twice
            self.end(obj)

        self.assertEqual(
            [o.secondlook_outcome for o in (rescued, weak, suppressed, slow)],
            ["fn_rescued", "fn_not_rescued", "fp_suppressed", "fp_not_suppressed"],
        )
        row = self.hour()
        self.assertEqual(
            {k: row[k] for k in ("false_negative", "fn_rescued", "fn_not_rescued",
                                 "false_positive", "fp_suppressed", "fp_not_suppressed",
                                 "objects_analysed")},
            {"false_negative": 2, "fn_rescued": 1, "fn_not_rescued": 1,
             "false_positive": 2, "fp_suppressed": 1, "fp_not_suppressed": 1,
             "objects_analysed": 4},
        )
        acc = sl_metrics.secondlook_stats(self.config, None)["accuracy"]
        self.assertEqual(acc["false_negative"], acc["fn_rescued"] + acc["fn_not_rescued"])
        self.assertEqual(acc["false_positive"], acc["fp_suppressed"] + acc["fp_not_suppressed"])
        self.assertEqual((acc["false_negative"], acc["false_positive"]), (2, 2))
        # every case is a Review item; released ones carry their event id
        self.assertEqual(
            self.items(),
            {
                ("1.0-rescued", "false_negative", "1.0-rescued"),
                ("1.0-weak", "false_negative", None),
                ("1.0-suppressed", "false_positive", None),
                ("1.0-slow", "false_positive", "1.0-slow"),
            },
        )

    def test_primary_crossing_later_means_not_a_false_negative(self):
        obj = new_track(self.config, 0.5)
        answer(obj, 2.0)
        step(obj, 1000.2)
        for i in range(10):  # primary alone gets there later
            obj.obj_data["score"] = 0.95
            step(obj, 1000.4 + i * 0.2)
        self.end(obj)
        self.assertIsNone(obj.secondlook_outcome)
        self.assertEqual(self.hour()["false_negative"], 0)


class TestCounterfactual(OutcomeCase):
    """FP/FN only when the factor would flip whether the object becomes a detection."""

    def track(self, score, factor, track_id, threshold=None, frames=3):
        if threshold is not None:
            self.config.cameras["cam"].objects.filters["person"].threshold = threshold
        obj = new_track(self.config, score, track_id)
        answer(obj, factor)
        for i in range(frames):
            step(obj, 1000.2 + i * 0.2)
        self.end(obj)
        return obj

    def assert_nothing_counted(self, obj):
        self.assertIsNone(obj.secondlook_outcome)
        row = self.hour()
        self.assertEqual(
            (row["false_positive"], row["false_negative"],
             row["fp_not_suppressed"], row["fn_not_rescued"]),
            (0, 0, 0, 0),
        )
        self.assertEqual(self.items(), set())

    def test_reported_case_087_069_is_not_a_false_positive(self):
        # P 0.87, SL 0.69 -> factor 0.79 (< 0.8); with a 0.65 threshold the
        # scaled median 0.69 stays above it: released correctly, no alarm
        obj = self.track(0.87, 0.69 / 0.87, "1.0-real", threshold=0.65)
        self.assertFalse(obj.false_positive)
        self.assert_nothing_counted(obj)

    def test_same_scores_at_threshold_07_would_suppress(self):
        # 0.87 x 0.79 = 0.69 < 0.7: SL holds it back -> an FP
        obj = self.track(0.87, 0.69 / 0.87, "1.0-strict", threshold=0.7)
        self.assertEqual(obj.secondlook_outcome, "fp_suppressed")

    def test_clear_false_positive_is_suppressed(self):
        obj = self.track(0.9, 0.3, "1.0-clear")
        self.assertTrue(obj.false_positive)
        self.assertEqual(obj.secondlook_outcome, "fp_suppressed")
        self.assertEqual(self.hour()["fp_suppressed"], 1)
        self.assertEqual(self.items(), {("1.0-clear", "false_positive", None)})

    def test_would_suppress_but_answer_after_release_is_not_suppressed(self):
        obj = new_track(self.config, 0.9, "1.0-late-fp")
        step(obj, 1000.2)
        step(obj, 1001.5)  # hold over, released by the primary
        self.assertFalse(obj.false_positive)
        answer(obj, 0.3)
        step(obj, 1001.7)
        self.end(obj)
        self.assertEqual(obj.secondlook_outcome, "fp_not_suppressed")
        self.assertEqual(self.hour()["fp_not_suppressed"], 1)

    def test_peak_window_decides_not_the_padded_end_window(self):
        # primary 0.9 then the object leaves (0.0 pads); x0.3 holds back the
        # peak, so still an FP even though the end window is low
        obj = new_track(self.config, 0.9, "1.0-leaves")
        answer(obj, 0.3)
        step(obj, 1000.2)
        for i in range(10):
            obj.obj_data["score"] = 0.0
            step(obj, 1000.4 + i * 0.2)
        self.end(obj)
        self.assertEqual(obj.secondlook_outcome, "fp_suppressed")

    def test_fn_scaled_median_still_below_threshold_is_not_a_false_negative(self):
        # 0.3 x 1.5 = 0.45 < 0.7: counted as FN (not rescued) before this change
        obj = self.track(0.3, 1.5, "1.0-weak-fn")
        self.assert_nothing_counted(obj)

    def test_fn_scaled_to_threshold_is_rescued(self):
        obj = self.track(0.5, 1.5, "1.0-fn")  # 0.5 x 1.5 = 0.75 >= 0.7
        self.assertFalse(obj.false_positive)
        self.assertEqual(obj.secondlook_outcome, "fn_rescued")

    def test_fn_that_would_be_rescued_but_answer_after_track_end(self):
        obj = new_track(self.config, 0.5, "1.0-fn-late")
        obj.apply_secondlook({"pending": True})
        step(obj, 1000.2)
        secondlook_late.park("1.0-fn-late", obj)
        with patch.object(Event, "get", side_effect=DoesNotExist):
            TrackedObjectProcessor.set_secondlook(
                SimpleNamespace(camera_states={}, config=self.config),
                "1.0-fn-late", {"factor": 1.5, "analysed": True},
            )
        self.assertEqual(obj.secondlook_outcome, "fn_not_rescued")
        self.assertEqual(self.hour()["fn_not_rescued"], 1)


class TestLateAnswers(OutcomeCase):
    def processor(self):
        return SimpleNamespace(camera_states={}, config=self.config)

    def test_answer_after_track_end_is_counted_not_rescued(self):
        obj = new_track(self.config, 0.5, "1.0-late")
        obj.apply_secondlook({"pending": True})
        step(obj, 1000.2)
        self.assertTrue(obj.awaiting_secondlook())
        secondlook_late.park("1.0-late", obj)  # what end() does
        with patch.object(Event, "get", side_effect=DoesNotExist):
            TrackedObjectProcessor.set_secondlook(
                self.processor(), "1.0-late",
                {"factor": 2.0, "sl_score": 0.9, "primary_score": 0.5, "analysed": True},
            )
        self.assertEqual(obj.secondlook_outcome, "fn_not_rescued")
        self.assertEqual(self.hour()["fn_not_rescued"], 1)
        self.assertEqual(self.hour()["objects_analysed"], 1)
        self.assertEqual(self.items(), {("1.0-late", "false_negative", None)})
        self.assertEqual(secondlook_late._parked, {})

    def test_late_fp_after_release_is_not_suppressed(self):
        obj = new_track(self.config, 0.9, "1.0-latefp")
        obj.apply_secondlook({"pending": True})
        step(obj, 1000.2)
        step(obj, 1001.5)  # hold over: released
        secondlook_late.park("1.0-latefp", obj)
        with patch.object(Event, "get", side_effect=DoesNotExist):
            TrackedObjectProcessor.set_secondlook(
                self.processor(), "1.0-latefp", {"factor": 0.0, "analysed": True},
            )
        self.assertEqual(obj.secondlook_outcome, "fp_not_suppressed")

    def test_never_answered_track_is_finished_after_ttl(self):
        obj = new_track(self.config, 0.5, "1.0-silent")
        obj.apply_secondlook({"pending": True})
        secondlook_late.park("1.0-silent", obj, now=time.monotonic() - 1000)
        finish_expired_secondlook_tracks(self.config.secondlook)
        self.assertTrue(obj.secondlook_finished)
        self.assertIsNone(obj.secondlook_outcome)  # no answer: no disagreement

    def test_holding_window_is_15_seconds_and_keeps_the_500_cap(self):
        self.assertEqual(secondlook_late.LATE_TTL_SECONDS, 15.0)
        self.assertEqual(secondlook_late.MAX_PARKED, 500)
        secondlook_late.park("young", "y", now=100.0)
        self.assertEqual(secondlook_late.expired(now=114.9), [])
        self.assertEqual(secondlook_late.expired(now=115.0), ["y"])
        for i in range(501):
            secondlook_late.park(f"t{i}", i, now=200.0)
        self.assertEqual(secondlook_late.expired(now=201.0), [0])  # oldest over cap


class TestHourlyMigration(unittest.TestCase):
    def test_existing_rows_keep_totals_and_become_acted_on(self):
        with tempfile.TemporaryDirectory() as td:
            db = f"{td}/e.db"
            conn = sqlite3.connect(db)
            conn.executescript(
                """CREATE TABLE hourly_summary (hour_start INTEGER NOT NULL,
                local_hour TEXT NOT NULL, camera TEXT NOT NULL, label TEXT NOT NULL,
                objects_analysed INTEGER NOT NULL DEFAULT 0,
                false_positive INTEGER NOT NULL DEFAULT 0,
                false_negative INTEGER NOT NULL DEFAULT 0,
                misclass INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (hour_start, camera, label));
                INSERT INTO hourly_summary VALUES (3600, 'x', 'cam', 'person', 9, 2, 3, 1);"""
            )
            conn.close()
            for _ in range(2):  # idempotent
                store = SaveAndReviewStore(db, f"{td}/img")
            with sqlite3.connect(db) as conn:
                conn.row_factory = sqlite3.Row
                row = dict(conn.execute("SELECT * FROM hourly_summary").fetchone())
            self.assertEqual(
                row,
                {"hour_start": 3600, "local_hour": "x", "camera": "cam", "label": "person",
                 "objects_analysed": 9, "false_positive": 2, "false_negative": 3,
                 "misclass": 1, "fn_rescued": 3, "fn_not_rescued": 0,
                 "fp_suppressed": 2, "fp_not_suppressed": 0},
            )
            del store


class TestEventLookup(OutcomeCase):
    def setUp(self):
        super().setUp()
        # a file: the API runs the lookup on another thread
        self.db = SqliteDatabase(f"{self.tmp.name}/frigate.db")
        self.db.bind([Event])
        self.db.connect()
        # Frigate's real table allows end_time NULL while an event is open
        self.db.execute_sql("CREATE TABLE event (id TEXT PRIMARY KEY, end_time REAL)")
        cam = self.config.cameras["cam"]
        cam.snapshots.enabled = True
        cam.snapshots.retain.default = 10

    def tearDown(self):
        self.db.close()
        super().tearDown()

    def item(self, track_id, kind, released, age_days=0.0):
        rec = self.store.deposit(
            track_id=track_id, camera="cam", kind=kind, label="person",
            jpeg_bytes=b"j", frame_time=time.time() - age_days * 86400,
            event_id=track_id if released else None,
        )
        return rec

    def event(self, track_id, end_time=1.0):
        self.db.execute_sql("INSERT INTO event VALUES (?, ?)", (track_id, end_time))

    def test_event_present_absent_retention_and_in_progress(self):
        recs = {
            "rescued": self.item("t-res", "false_negative", True),
            "not_rescued": self.item("t-nres", "false_negative", False),
            "not_suppressed": self.item("t-nsup", "false_positive", True),
            "suppressed": self.item("t-sup", "false_positive", False),
            "misclass": self.item("t-mc", "misclass", True),
            "old": self.item("t-old", "false_negative", True, age_days=11),
            "open": self.item("t-open", "false_negative", True),
            "cleaned": self.item("t-gone", "false_negative", True),
        }
        for t in ("t-res", "t-nsup", "t-mc", "t-old"):
            self.event(t)
        self.event("t-open", None)
        got = live_outcomes(self.config, recs.values())
        self.assertEqual(
            {name: got[r.id] for name, r in recs.items()},
            {"rescued": "rescued", "not_rescued": "not_rescued",
             "not_suppressed": "not_suppressed", "suppressed": "suppressed",
             "misclass": "none", "old": "none", "open": "none", "cleaned": "none"},
        )

    def test_hourly_outcome_and_event_lookup_agree(self):
        rescued = new_track(self.config, 0.5, "1.0-rescued")
        answer(rescued, 2.0)
        step(rescued, 1000.2)
        weak = new_track(self.config, 0.5, "1.0-weak")
        answer(weak, 2.0)  # after the last frame: not released
        for obj in (rescued, weak):
            self.end(obj)
            if not obj.false_positive:  # Frigate wrote an Event for it
                self.event(obj.obj_data["id"])
        # these tracks ran at frame time ~1000 s: judge them from then
        got = live_outcomes(self.config, self.store.list(), now=1100.0)
        by_track = {r.track_id: got[r.id] for r in self.store.list()}
        self.assertEqual(by_track, {"1.0-rescued": "rescued", "1.0-weak": "not_rescued"})

    def test_no_retention_means_no_outcome(self):
        cam = self.config.cameras["cam"]
        cam.snapshots.enabled = False
        cam.record.enabled = False
        self.assertEqual(event_retention_days(cam, "person"), 0)
        rec = self.item("t-res", "false_negative", True)
        self.event("t-res")
        self.assertEqual(live_outcomes(self.config, [rec])[rec.id], "none")

    def test_review_api_returns_live_outcome(self):
        rec = self.item("t-res", "false_negative", True)
        self.event("t-res")
        app = FastAPI()
        app.frigate_config = self.config
        app.dependency_overrides[get_allowed_cameras_for_filter] = lambda: ["cam"]
        app.include_router(router)
        with TestClient(app) as client:
            listed = client.get("/secondlook/save_and_review").json()["examples"]
            self.assertEqual(listed[0]["live_outcome"], "rescued")
            one = client.get(f"/secondlook/save_and_review/{rec.id}").json()
            self.assertEqual(one["live_outcome"], "rescued")
            marked = client.post(
                f"/secondlook/save_and_review/{rec.id}/mark", json={"mark": "ignore"}
            ).json()["example"]
            self.assertEqual(marked["live_outcome"], "rescued")


if __name__ == "__main__":
    unittest.main()
