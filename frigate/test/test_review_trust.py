"""Tests for Review Trust Overview, Human Feedback, Golden Clips, and Alert Timing."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from peewee import SqliteDatabase

from frigate.api import trust as trust_api
from frigate.api.auth import (
    get_allowed_cameras_for_filter,
    get_current_user,
)
from frigate.models import Event, Recordings, ReviewSegment, UserReviewStatus
from frigate.review.maintainer import PendingReviewSegment, ReviewSegmentMaintainer
from frigate.review.trust_store import ReviewTrustStore
from frigate.review.types import SeverityEnum
from frigate.secondlook.save_and_review import SaveAndReviewStore


class TestReviewTrustStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, "review_trust.db")
        self.exports_dir = os.path.join(self.tmp.name, "exports")
        self.store = ReviewTrustStore(self.db_path, self.exports_dir)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_feedback_set_get_update_clear(self) -> None:
        created = self.store.set_feedback(
            review_id="rev-1",
            camera="front_door",
            severity="alert",
            verdict="fp",
            start_time=1000.0,
            reason="shadow",
        )
        self.assertEqual(created["verdict"], "fp")
        self.assertEqual(created["reason"], "shadow")

        updated = self.store.set_feedback(
            review_id="rev-1",
            camera="front_door",
            severity="alert",
            verdict="misclassified",
            start_time=1000.0,
            corrected_label="cat",
            reason="actually a cat",
        )
        self.assertEqual(updated["verdict"], "misclassified")
        self.assertEqual(updated["corrected_label"], "cat")

        fb_map = self.store.get_feedback_map(900.0, 1100.0, cameras=["front_door"])
        self.assertIn("rev-1", fb_map)
        self.assertEqual(fb_map["rev-1"]["verdict"], "misclassified")
        self.assertEqual(
            self.store.get_feedback_map(900.0, 1100.0, cameras=["back_yard"]), {}
        )

        self.assertTrue(self.store.clear_feedback("rev-1"))
        self.assertFalse(self.store.clear_feedback("rev-1"))
        self.assertEqual(self.store.get_feedback_map(900.0, 1100.0), {})

        with self.assertRaises(ValueError):
            self.store.set_feedback(
                review_id="rev-bad",
                camera="front_door",
                severity="alert",
                verdict="invalid",
                start_time=1000.0,
            )

    def test_golden_clips_sidecar_coverage_and_delete(self) -> None:
        source_mp4 = os.path.join(self.tmp.name, "sample.mp4")
        Path(source_mp4).write_bytes(b"fake-mp4-bytes")

        clip1 = self.store.create_golden_clip(
            review_id="rev-10",
            camera="front_door",
            severity="alert",
            start_time=2000.0,
            end_time=2015.0,
            kind="tp",
            object_label="person",
            condition_tags=["day", "wind"],
            source_video_path=source_mp4,
            notes="Clear daytime approach",
        )
        self.assertTrue(os.path.isfile(clip1["sidecar_path"]))
        self.assertTrue(os.path.isfile(clip1["video_path"]))
        with open(clip1["sidecar_path"], encoding="utf-8") as f:
            sidecar = json.load(f)
        self.assertEqual(sidecar["review_id"], "rev-10")
        self.assertEqual(sidecar["condition_tags"], ["day", "wind"])

        clip2 = self.store.create_golden_clip(
            review_id=None,
            camera="back_yard",
            severity="detection",
            start_time=2100.0,
            end_time=2110.0,
            kind="background",
            condition_tags=["night_ir", "rain", "wind"],
        )
        self.assertTrue(os.path.isfile(clip2["sidecar_path"]))

        summary = self.store.get_golden_summary()
        self.assertEqual(summary["total"], 2)
        self.assertEqual(
            summary["coverage"],
            {"day": 1, "night_ir": 1, "rain": 1, "wind": 2, "snow": 0},
        )
        self.assertEqual(len(self.store.list_golden_clips(cameras=["front_door"])), 1)
        self.assertEqual(len(self.store.list_golden_clips(severity="detection")), 1)

        self.assertTrue(self.store.delete_golden_clip(clip1["id"]))
        self.assertFalse(os.path.exists(clip1["sidecar_path"]))
        self.assertFalse(os.path.exists(clip1["video_path"]))
        self.assertEqual(self.store.get_golden_summary()["total"], 1)

    def test_alert_timings_crud(self) -> None:
        rec = self.store.record_alert_timing(
            review_id="alert-1",
            camera="front_door",
            alert_time=5000.0,
            discovery_ms=0.0,
            resource_busy_ms=14.5,
            confirmation_ms=420.0,
        )
        self.assertAlmostEqual(rec["total_ms"], 434.5)

        self.store.record_alert_timing(
            review_id="alert-2",
            camera="back_yard",
            alert_time=5100.0,
            discovery_ms=10.0,
            resource_busy_ms=20.0,
            confirmation_ms=500.0,
        )
        rows = self.store.get_timings_in_range(4900.0, 5200.0, cameras=["front_door"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["review_id"], "alert-1")


class TestReviewTrustAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.frigate_db_path = os.path.join(self.tmp.name, "frigate.db")
        self.db = SqliteDatabase(self.frigate_db_path)
        self.models = [Event, Recordings, ReviewSegment, UserReviewStatus]
        self.db.bind(self.models, bind_refs=False, bind_backrefs=False)
        self.db.connect()
        self.db.create_tables(self.models)

        self.trust_store = ReviewTrustStore(
            os.path.join(self.tmp.name, "review_trust.db"),
            os.path.join(self.tmp.name, "exports"),
        )
        self.sl_store = SaveAndReviewStore(
            os.path.join(self.tmp.name, "secondlook_examples.db"),
            os.path.join(self.tmp.name, "sl_images"),
        )

        self.app = FastAPI()
        # Mount trust router BEFORE a /review/{review_id} wildcard route so
        # /review/overview & /review/golden_clips never collide with it.
        wildcard_router = APIRouter()

        @wildcard_router.get("/review/{review_id}")
        def _wildcard_review(review_id: str):
            return {"wildcard": review_id}

        self.app.include_router(trust_api.router)
        self.app.include_router(wildcard_router)
        self.app.review_trust_store = self.trust_store
        self.app.secondlook_store = self.sl_store
        self.app.frigate_config = SimpleNamespace(
            database=SimpleNamespace(path=self.frigate_db_path),
            cameras={"front_door": SimpleNamespace(), "back_yard": SimpleNamespace()},
            auth=SimpleNamespace(roles={"admin": []}),
            proxy=SimpleNamespace(separator=","),
        )
        self.app.dependency_overrides[get_allowed_cameras_for_filter] = lambda: [
            "front_door",
            "back_yard",
        ]
        self.app.dependency_overrides[get_current_user] = lambda: {
            "username": "admin",
            "role": "admin",
        }
        self.client = TestClient(
            self.app,
            headers={"remote-user": "admin", "remote-role": "admin"},
        )

    def tearDown(self) -> None:
        self.client.close()
        self.db.close()
        self.tmp.cleanup()

    def _insert_segment(
        self,
        seg_id: str,
        *,
        camera: str = "front_door",
        severity: str = SeverityEnum.alert.value,
        start_time: float,
        end_time: float | None = None,
        objects: list[str] | None = None,
        zones: list[str] | None = None,
        sub_labels: list[str] | None = None,
        score: float | None = None,
    ) -> None:
        data: dict[str, object] = {
            "detections": [f"ev-{seg_id}"],
            "objects": objects or ["person"],
            "zones": zones or [],
            "sub_labels": sub_labels or [],
            "audio": [],
        }
        if score is not None:
            data["score"] = score
        ReviewSegment.insert(
            id=seg_id,
            camera=camera,
            start_time=start_time,
            end_time=end_time if end_time is not None else start_time + 5.0,
            severity=severity,
            thumb_path=f"/tmp/thumb-{seg_id}.webp",
            data=data,
        ).execute()

    def _insert_sl_analysis(
        self,
        *,
        track_id: str,
        camera: str = "front_door",
        label: str = "person",
        end_time: float,
        live: str,
        live_outcome: str | None = None,
    ) -> None:
        with self.sl_store._connect() as conn:
            conn.execute(
                """
                INSERT INTO analyses (
                    camera, track_id, label, end_time, outcome, live,
                    live_outcome, written_at
                ) VALUES (?, ?, ?, ?, 'result', ?, ?, ?)
                """,
                (camera, track_id, label, end_time, live, live_outcome, end_time),
            )
            conn.commit()

    def test_30_bar_bucketing_across_all_bar_sizes(self) -> None:
        expected_seconds = {
            "1m": 60,
            "10m": 600,
            "1h": 3600,
            "4h": 14400,
            "24h": 86400,
        }
        end_ts = 1_700_000_000.0
        for bar_size, bar_sec in expected_seconds.items():
            ReviewSegment.delete().execute()
            seg_id = f"seg-{bar_size}"
            # Put one segment right inside the very first bar [window_start, window_start + bar_sec)
            first_bar_ts = end_ts - (30 * bar_sec) + (bar_sec * 0.5)
            self._insert_segment(seg_id, start_time=first_bar_ts)

            resp = self.client.get(
                "/review/overview",
                params={"bar_size": bar_size, "end_time": end_ts},
            )
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["num_bars"], 30)
            self.assertEqual(body["bar_seconds"], bar_sec)
            self.assertEqual(len(body["buckets"]), 30)
            self.assertEqual(body["buckets"][0]["above"]["tp"], 1)
            # Last bar should not contain the segment placed in bar 0
            self.assertEqual(body["buckets"][29]["above"]["tp"], 0)

    def test_no_double_counting_and_feedback_moves_above_to_below(self) -> None:
        end_ts = 1_700_000_000.0
        # Bar 29 starts at end_ts - 3600
        t = end_ts - 1800.0
        self._insert_segment("a1", start_time=t, severity="alert")
        self._insert_segment("a2", start_time=t + 10, severity="alert")
        self._insert_segment("a3", start_time=t + 20, severity="alert")
        self._insert_segment("a4", start_time=t + 30, severity="alert")

        # Before feedback: all 4 are above.tp
        body0 = self.client.get(
            "/review/overview", params={"severity": "alert", "end_time": end_ts}
        ).json()
        bar = body0["buckets"][29]
        self.assertEqual(bar["above"]["total"], 4)
        self.assertEqual(bar["above"]["tp"], 4)
        self.assertEqual(bar["below"]["total"], 0)

        # Mark a1 as fp, a2 as misclassified, a3 as good; leave a4 unmarked
        self.assertEqual(
            self.client.post(
                "/review/a1/feedback", json={"verdict": "fp", "reason": "headlight"}
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.post(
                "/review/a2/feedback",
                json={"verdict": "misclassified", "corrected_label": "dog"},
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.post(
                "/review/a3/feedback", json={"verdict": "good"}
            ).status_code,
            200,
        )

        body1 = self.client.get(
            "/review/overview", params={"severity": "alert", "end_time": end_ts}
        ).json()
        bar1 = body1["buckets"][29]
        # a3 (good) + a4 (tp) stay above; a1 (fp) + a2 (misclassified) move below
        self.assertEqual(bar1["above"]["good"], 1)
        self.assertEqual(bar1["above"]["tp"], 1)
        self.assertEqual(bar1["above"]["total"], 2)
        self.assertEqual(bar1["below"]["fp"], 1)
        self.assertEqual(bar1["below"]["misclassified"], 1)
        self.assertEqual(bar1["below"]["fn"], 0)
        self.assertEqual(bar1["below"]["total"], 2)
        self.assertEqual(set(body1["feedback_by_review_id"].keys()), {"a1", "a2", "a3"})

        # Clearing a1 moves it back to above.tp
        self.assertEqual(
            self.client.post(
                "/review/a1/feedback", json={"verdict": "clear"}
            ).status_code,
            200,
        )
        body2 = self.client.get(
            "/review/overview", params={"severity": "alert", "end_time": end_ts}
        ).json()
        bar2 = body2["buckets"][29]
        self.assertEqual(bar2["above"]["tp"], 2)
        self.assertEqual(bar2["above"]["good"], 1)
        self.assertEqual(bar2["above"]["total"], 3)
        self.assertEqual(bar2["below"]["fp"], 0)
        self.assertEqual(bar2["below"]["total"], 1)

    def test_alerts_vs_detections_secondlook_background_and_timing(self) -> None:
        end_ts = 1_700_000_000.0
        t = end_ts - 1200.0  # Bar 29 in 1h view

        # 1 alert segment + 1 human misclassified alert segment
        self._insert_segment(
            "al-1",
            start_time=t,
            severity="alert",
            objects=["person"],
            zones=["porch"],
            score=0.85,
        )
        self._insert_segment(
            "al-2",
            start_time=t + 30,
            severity="alert",
            objects=["car"],
            zones=["driveway"],
            score=0.90,
        )
        self.trust_store.set_feedback(
            review_id="al-2",
            camera="front_door",
            severity="alert",
            verdict="misclassified",
            start_time=t + 30,
            corrected_label="truck",
        )

        # 1 detection segment
        self._insert_segment(
            "det-1",
            start_time=t + 60,
            severity="detection",
            objects=["cat"],
            zones=["yard"],
            score=0.65,
        )

        # Secondlook automatic metrics in same bucket:
        # rescued_fn (alerts view), fp_suppressed (detections view), misclass (both views)
        self._insert_sl_analysis(
            track_id="t-fn",
            end_time=t + 10,
            live="false_negative",
            live_outcome="fn_rescued",
        )
        self._insert_sl_analysis(
            track_id="t-fp",
            end_time=t + 20,
            live="false_positive",
            live_outcome="fp_suppressed",
        )
        self._insert_sl_analysis(
            track_id="t-mc", end_time=t + 30, live="misclass", live_outcome=None
        )

        # Hollow background clip (counted above in detections view) + golden alert clip
        self.trust_store.create_golden_clip(
            review_id=None,
            camera="front_door",
            severity="detection",
            start_time=t + 40,
            kind="background",
            condition_tags=["rain"],
        )
        self.trust_store.create_golden_clip(
            review_id="al-1",
            camera="front_door",
            severity="alert",
            start_time=t,
            kind="tp",
            condition_tags=["day"],
        )

        # Alert timing rows in bar 29
        self.trust_store.record_alert_timing(
            review_id="al-1",
            camera="front_door",
            alert_time=t,
            discovery_ms=0.0,
            resource_busy_ms=10.0,
            confirmation_ms=100.0,
        )
        self.trust_store.record_alert_timing(
            review_id="al-2",
            camera="front_door",
            alert_time=t + 30,
            discovery_ms=0.0,
            resource_busy_ms=30.0,
            confirmation_ms=300.0,
        )

        # Alerts view
        alert_resp = self.client.get(
            "/review/overview", params={"severity": "alert", "end_time": end_ts}
        )
        self.assertEqual(alert_resp.status_code, 200)
        alert_body = alert_resp.json()
        self.assertEqual(alert_body["golden_total"], 2)
        bar_a = alert_body["buckets"][29]
        self.assertEqual(bar_a["above"]["tp"], 1)
        self.assertEqual(bar_a["above"]["rescued_fn"], 1)
        self.assertEqual(bar_a["above"]["auto_suppressed_fp"], 0)
        self.assertEqual(bar_a["above"]["auto_misclassified"], 1)
        self.assertEqual(bar_a["above"]["background_clips"], 0)
        self.assertEqual(bar_a["above"]["total"], 3)
        self.assertEqual(bar_a["below"]["misclassified"], 1)
        self.assertEqual(bar_a["below"]["total"], 1)
        self.assertEqual(bar_a["timing"]["count"], 2)
        self.assertAlmostEqual(bar_a["timing"]["avg_discovery_ms"], 0.0)
        self.assertAlmostEqual(bar_a["timing"]["avg_resource_busy_ms"], 20.0)
        self.assertAlmostEqual(bar_a["timing"]["avg_confirmation_ms"], 200.0)
        self.assertAlmostEqual(bar_a["timing"]["avg_total_ms"], 220.0)
        self.assertAlmostEqual(bar_a["timing"]["slowest_ms"], 330.0)
        self.assertEqual(bar_a["timing"]["slowest_review_id"], "al-2")

        # Detections view
        det_resp = self.client.get(
            "/review/overview", params={"severity": "detection", "end_time": end_ts}
        )
        self.assertEqual(det_resp.status_code, 200)
        bar_d = det_resp.json()["buckets"][29]
        self.assertEqual(bar_d["above"]["tp"], 1)
        self.assertEqual(bar_d["above"]["rescued_fn"], 0)
        self.assertEqual(bar_d["above"]["auto_suppressed_fp"], 1)
        self.assertEqual(bar_d["above"]["auto_misclassified"], 1)
        self.assertEqual(bar_d["above"]["background_clips"], 1)
        self.assertEqual(bar_d["above"]["total"], 4)

    def test_filters_and_golden_clip_endpoints(self) -> None:
        end_ts = 1_700_000_000.0
        t = end_ts - 500.0
        self._insert_segment(
            "f1",
            camera="front_door",
            severity="alert",
            start_time=t,
            objects=["person-verified"],
            zones=["porch"],
            sub_labels=["Alice"],
            score=0.92,
        )
        self._insert_segment(
            "f2",
            camera="back_yard",
            severity="alert",
            start_time=t + 10,
            objects=["car"],
            zones=["driveway"],
            sub_labels=["Bob"],
            score=0.45,
        )

        resp = self.client.get(
            "/review/overview",
            params={
                "severity": "alert",
                "end_time": end_ts,
                "cameras": "front_door",
                "labels": "person",
                "zones": "porch",
                "sub_labels": "Alice",
                "min_score": 0.8,
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["buckets"][29]["above"]["tp"], 1)

        # Save golden clip via POST /review/{review_id}/golden (no recordings on disk -> sidecar still saved)
        g_resp = self.client.post(
            "/review/f1/golden",
            json={
                "kind": "tp",
                "condition_tags": ["day", "rain"],
                "notes": "Verified Alice on porch",
            },
        )
        self.assertEqual(g_resp.status_code, 201)
        clip = g_resp.json()["clip"]
        self.assertEqual(clip["object_label"], "person")
        self.assertTrue(os.path.isfile(clip["sidecar_path"]))

        list_resp = self.client.get("/review/golden_clips")
        self.assertEqual(list_resp.status_code, 200)
        self.assertEqual(list_resp.json()["total"], 1)
        self.assertEqual(list_resp.json()["coverage"]["day"], 1)
        self.assertEqual(list_resp.json()["coverage"]["rain"], 1)

        del_resp = self.client.delete(f"/review/golden_clips/{clip['id']}")
        self.assertEqual(del_resp.status_code, 200)
        self.assertTrue(del_resp.json()["success"])
        self.assertEqual(self.client.get("/review/golden_clips").json()["total"], 0)

    def test_maintainer_records_alert_timing_with_discovery_placeholder(self) -> None:
        maintainer = ReviewSegmentMaintainer.__new__(ReviewSegmentMaintainer)
        maintainer.trust_store = self.trust_store

        segment = PendingReviewSegment(
            camera="front_door",
            frame_time=6000.5,
            severity=SeverityEnum.alert,
            detections={"obj-1": "person"},
            sub_labels={},
            zones=["porch"],
            audio=set(),
        )
        maintainer._record_alert_timing(
            segment,
            alert_time=6000.5,
            alert_objects=[
                {
                    "id": "obj-1",
                    "start_time": 6000.0,
                    "accumulated_resource_busy_ms": 27.5,
                }
            ],
        )

        rows = self.trust_store.get_timings_in_range(5999.0, 6002.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["review_id"], segment.id)
        self.assertAlmostEqual(rows[0]["discovery_ms"], 0.0)
        self.assertAlmostEqual(rows[0]["resource_busy_ms"], 27.5)
        self.assertAlmostEqual(rows[0]["confirmation_ms"], 500.0)
        self.assertAlmostEqual(rows[0]["total_ms"], 527.5)


if __name__ == "__main__":
    unittest.main()
