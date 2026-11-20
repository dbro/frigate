"""Hourly Secondlook disagreement summary: counting, local-hour buckets, persistence."""

import os
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from frigate.api.auth import get_allowed_cameras_for_filter
from frigate.api.save_and_review import router
from frigate.config.classification import (
    SecondlookSaveAndReviewConfig,
    SecondlookSaveAndReviewRetentionConfig,
)
from frigate.secondlook import hourly
from frigate.secondlook.retention import run_retention
from frigate.secondlook.save_and_review import (
    SaveAndReviewStore,
    configure_save_and_review,
)
from frigate.test.test_save_and_review import make_config, make_track


def utc(*args) -> float:
    return datetime(*args, tzinfo=timezone.utc).timestamp()


def latest(store) -> dict:
    return hourly.series(store, hours=1)[0]


def result(**extra) -> dict:
    return {"factor": 1.0, "primary_score": 0.9, "sl_score": 0.9, **extra}


class LiveCase(unittest.TestCase):
    def tearDown(self):
        configure_save_and_review(SecondlookSaveAndReviewConfig(enabled=False))


class TestLiveCounting(LiveCase):
    def test_objects_analysed_once_per_track_disagreements_once_per_kind(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(True, td)
            store = configure_save_and_review(config.secondlook.save_and_review)
            one = make_track(config, 0.9, 1)
            for _ in range(5):  # refresh analyses of the same object
                one.apply_secondlook(result(analysed=True))
            fallback = make_track(config, 0.9, 1)  # queue-full fail-open only
            fallback.apply_secondlook(result(factor=None, applied_by="primary_fallback"))
            fp = make_track(config, 0.9, 0.0)
            fp.apply_secondlook(result(factor=0.0, analysed=True))
            for i in range(20):
                fp.update(1001 + i, dict(fp.obj_data), False)
            mc = make_track(config, 0.9, 1)
            for _ in range(3):
                mc.apply_secondlook(result(analysed=True, misclass_label="dog"))
                mc.update(1001, dict(mc.obj_data), False)
            for track in (one, fallback, fp, mc, fp):  # tracks end
                track.finish_secondlook()
            row = latest(store)
            self.assertEqual(
                {k: row[k] for k in hourly.LEGACY_COLUMNS},
                {
                    "objects_analysed": 3,
                    "false_positive": 1,
                    "false_negative": 0,
                    "misclass": 1,
                    "fn_rescued": 0,
                    "fn_not_rescued": 0,
                    "fp_suppressed": 1,
                    "fp_not_suppressed": 0,
                },
            )
            with sqlite3.connect(store.db_path) as conn:
                self.assertEqual(
                    conn.execute(
                        "SELECT DISTINCT camera, label FROM hourly_counts"
                    ).fetchall(),
                    [("cam", "person")],
                )

    def test_disabled_writes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(False, td)
            configure_save_and_review(config.secondlook.save_and_review)
            fp = make_track(config, 0.9, 0.0)
            fp.apply_secondlook(result(factor=0.0, analysed=True))
            for i in range(20):
                fp.update(1001 + i, dict(fp.obj_data), False)
            self.assertEqual(list(Path(td).iterdir()), [])


class TestBuckets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SaveAndReviewStore(f"{self.tmp.name}/e.db", f"{self.tmp.name}/img")
        self.store.timezone = "Europe/Berlin"

    def tearDown(self):
        self.tmp.cleanup()

    def test_local_hour_and_dst_fall_back_and_spring_forward(self):
        self.assertEqual(
            hourly.hour_bucket(utc(2026, 11, 11, 11, 59), "Europe/Berlin"),
            (int(utc(2026, 11, 11, 11)), "2026-11-11T12:00+01:00"),
        )
        self.assertEqual(
            hourly.hour_bucket(utc(2026, 11, 11, 11, 59), "America/New_York")[1],
            "2026-11-11T06:00-05:00",
        )
        # 25 Oct 2026: 02:00-03:00 local happens twice -> two buckets
        hourly.add(self.store, "cam", "person", "false_positive", now=utc(2026, 10, 25, 0, 30))
        hourly.add(self.store, "cam", "person", "misclass", now=utc(2026, 10, 25, 1, 30))
        hours = hourly.series(self.store, now=utc(2026, 10, 25, 2, 10), hours=4)
        self.assertEqual(
            [h["local_hour"] for h in hours],
            [
                "2026-10-25T01:00+02:00",
                "2026-10-25T02:00+02:00",
                "2026-10-25T02:00+01:00",
                "2026-10-25T03:00+01:00",
            ],
        )
        self.assertEqual([h["false_positive"] for h in hours], [0, 1, 0, 0])
        self.assertEqual([h["misclass"] for h in hours], [0, 0, 1, 0])
        # 29 Mar 2026: 02:00 local does not exist
        spring = hourly.series(self.store, now=utc(2026, 3, 29, 1, 30), hours=2)
        self.assertEqual(
            [h["local_hour"] for h in spring],
            ["2026-03-29T01:00+01:00", "2026-03-29T03:00+02:00"],
        )

    def test_container_local_time_when_no_ui_timezone(self):
        old = os.environ.get("TZ")
        os.environ["TZ"] = "Asia/Kolkata"
        time.tzset()
        try:
            self.assertEqual(
                hourly.hour_bucket(utc(2026, 11, 11, 11, 59), None)[1],
                "2026-11-11T17:00+05:30",
            )
        finally:
            if old is None:
                del os.environ["TZ"]
            else:
                os.environ["TZ"] = old
            time.tzset()

    def test_168_hours_with_zero_filled_gaps_and_camera_filter(self):
        now = utc(2026, 11, 11, 11, 30)
        hourly.add(self.store, "cam", "person", "objects_analysed", now=now)
        hourly.add(self.store, "cam", "car", "objects_analysed", now=now)
        hourly.add(self.store, "other", "car", "objects_analysed", now=now)
        hourly.add(self.store, "cam", "car", "objects_analysed", now=now - 168 * 3600)
        hours = hourly.series(self.store, now=now)
        self.assertEqual(len(hours), 168)
        self.assertEqual(hours[-1]["objects_analysed"], 3)
        self.assertEqual(sum(h["objects_analysed"] for h in hours), 3)
        self.assertEqual(hours[0]["hour_start"], hours[-1]["hour_start"] - 167 * 3600)
        self.assertEqual(
            hourly.series(self.store, now=now, cameras=["cam"])[-1]["objects_analysed"], 2
        )
        self.assertEqual(
            hourly.series(self.store, now=now, cameras=[])[-1]["objects_analysed"], 0
        )

    def test_survives_restart(self):
        hourly.add(self.store, "cam", "person", "objects_analysed")
        hourly.add(self.store, "cam", "person", "false_negative")
        reopened = SaveAndReviewStore(self.store.db_path, self.store.images_path)
        hourly.add(reopened, "cam", "person", "objects_analysed")
        row = latest(reopened)
        self.assertEqual((row["objects_analysed"], row["false_negative"]), (2, 1))

    def test_retention_drops_summary_rows_older_than_summary_days(self):
        now = utc(2026, 11, 11, 11, 30)
        for days in (95, 89):
            hourly.add(self.store, "cam", "person", "misclass", now=now - days * 86400)
        accepted = self.store.deposit(
            track_id="t", camera="cam", kind="misclass", label="person",
            jpeg_bytes=b"j", frame_time=now - 400 * 86400,
        )
        self.store.set_mark(accepted.id, "primary_right")
        keep = SecondlookSaveAndReviewRetentionConfig(summary_days=0)
        self.assertEqual(run_retention(self.store, keep, now=now)["deleted_summary_rows"], 0)
        out = run_retention(self.store, SecondlookSaveAndReviewRetentionConfig(), now=now)
        self.assertEqual(out["deleted_summary_rows"], 1)
        with sqlite3.connect(self.store.db_path) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM hourly_counts").fetchone()[0], 1)
        self.assertEqual(self.store.get(accepted.id).mark, "primary_right")

    def test_existing_examples_untouched_by_migration(self):
        rec = self.store.deposit(
            track_id="t", camera="cam", kind="misclass", label="person",
            jpeg_bytes=b"j", frame_time=1.0,
        )
        with sqlite3.connect(self.store.db_path) as conn:
            conn.execute("DROP TABLE hourly_summary")
        reopened = SaveAndReviewStore(self.store.db_path, self.store.images_path)
        self.assertEqual(reopened.get(rec.id).frame_time, 1.0)
        self.assertEqual(latest(reopened)["objects_analysed"], 0)


class TestHourlyAPI(LiveCase):
    def test_endpoint_and_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(True, td)
            store = configure_save_and_review(config.secondlook.save_and_review)
            hourly.add(store, "cam", "person", "objects_analysed")
            hourly.add(store, "hidden", "person", "objects_analysed")
            app = FastAPI()
            app.frigate_config = config
            app.dependency_overrides[get_allowed_cameras_for_filter] = lambda: ["cam"]
            app.include_router(router)
            with TestClient(app) as client:
                body = client.get("/secondlook/disagreements/hourly").json()
                self.assertEqual(len(body["hours"]), 168)
                self.assertEqual(body["hours"][-1]["objects_analysed"], 1)
                self.assertEqual(body["hours"][0]["objects_analysed"], 0)
                config.secondlook.save_and_review.enabled = False
                self.assertEqual(
                    client.get("/secondlook/disagreements/hourly").status_code, 404
                )


class TestTimingPrecision(unittest.TestCase):
    def test_ms_rounded_to_two_decimals_like_detector_inference_speed(self):
        from types import SimpleNamespace

        from frigate.secondlook import metrics as sl_metrics

        with tempfile.TemporaryDirectory() as td:
            metrics = SimpleNamespace(
                secondlook={
                    "available": True,
                    "samples": 1,
                    "inference_ms": 7.0049,
                    "processing_ms": 123.456789,
                    "queue_ms": 0.1,
                }
            )
            stats = sl_metrics.secondlook_stats(make_config(False, td), metrics)
            self.assertEqual(
                (stats["inference_ms"], stats["processing_ms"], stats["queue_ms"]),
                (7.0, 123.46, 0.1),
            )


if __name__ == "__main__":
    unittest.main()
