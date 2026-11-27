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
        event_id: str | None = None,
        camera: str = "front_door",
        label: str = "person",
        end_time: float,
        live: str,
        live_outcome: str | None = None,
        misclass_label: str | None = None,
        primary_score: float | None = None,
        sl_score: float | None = None,
        factor: float | None = None,
    ) -> None:
        with self.sl_store._connect() as conn:
            conn.execute(
                """
                INSERT INTO analyses (
                    camera, track_id, event_id, label, end_time, outcome, live,
                    live_outcome, misclass_label, primary_score, sl_score, factor,
                    written_at
                ) VALUES (?, ?, ?, ?, ?, 'result', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera,
                    track_id,
                    event_id,
                    label,
                    end_time,
                    live,
                    live_outcome,
                    misclass_label,
                    primary_score,
                    sl_score,
                    factor,
                    end_time,
                ),
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

        # 4 alert segments:
        # - al-1: normal alert
        # - al-2: human misclassified alert
        # - al-3: rescued FN alert (matched via ev-al-3)
        # - al-4: Secondlook auto-misclassified alert (matched via ev-al-4)
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
        self._insert_segment(
            "al-3",
            start_time=t + 40,
            severity="alert",
            objects=["person"],
            zones=["porch"],
            score=0.72,
        )
        self._insert_segment(
            "al-4",
            start_time=t + 50,
            severity="alert",
            objects=["dog"],
            zones=["yard"],
            score=0.78,
        )
        self.trust_store.set_feedback(
            review_id="al-2",
            camera="front_door",
            severity="alert",
            verdict="misclassified",
            start_time=t + 30,
            corrected_label="truck",
        )

        # 2 detection segments (det-1 normal, det-2 auto-misclassified via ev-det-2)
        self._insert_segment(
            "det-1",
            start_time=t + 60,
            severity="detection",
            objects=["cat"],
            zones=["yard"],
            score=0.65,
        )
        self._insert_segment(
            "det-2",
            start_time=t + 70,
            severity="detection",
            objects=["cat"],
            zones=["yard"],
            score=0.60,
        )

        # Secondlook analyses:
        # - ev-al-3 rescued FN -> reclassifies al-3 (NO double counting!)
        # - ev-al-4 misclass -> reclassifies al-4 (NO double counting!)
        # - t-fp suppressed FP (event_id=None) -> Detections above.auto_suppressed_fp
        # - t-unlinked-mc suppressed misclass (event_id=None) -> must NOT inflate Alerts bar!
        # - ev-det-2 misclass -> reclassifies det-2 in Detections view
        self._insert_sl_analysis(
            track_id="ev-al-3",
            event_id="ev-al-3",
            end_time=t + 45,
            live="false_negative",
            live_outcome="fn_rescued",
            primary_score=0.45,
            sl_score=0.82,
            factor=1.5,
        )
        self._insert_sl_analysis(
            track_id="ev-al-4",
            event_id="ev-al-4",
            label="dog",
            end_time=t + 55,
            live="misclass",
            live_outcome=None,
            misclass_label="cat",
        )
        self._insert_sl_analysis(
            track_id="t-fp",
            event_id=None,
            end_time=t + 20,
            live="false_positive",
            live_outcome="fp_suppressed",
        )
        self._insert_sl_analysis(
            track_id="t-unlinked-mc",
            event_id=None,
            end_time=t + 25,
            live="misclass",
            live_outcome=None,
            misclass_label="bird",
        )
        self._insert_sl_analysis(
            track_id="ev-det-2",
            event_id="ev-det-2",
            label="cat",
            end_time=t + 75,
            live="misclass",
            live_outcome=None,
            misclass_label="dog",
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

        # Alert timing rows in bar 29 (pure consecutive timestamp differences)
        self.trust_store.record_alert_timing(
            review_id="al-1",
            camera="front_door",
            alert_time=t,
            first_visible_time=t - 0.35,
            first_detected_time=t - 0.25,
            track_start_time=t - 0.15,
            confirmed_time=t - 0.05,
            discovery_ms=100.0,
            tracking_ms=100.0,
            confirmation_ms=100.0,
            zone_delay_ms=50.0,
            reverse_norfair_status="done",
        )
        self.trust_store.record_alert_timing(
            review_id="al-2",
            camera="front_door",
            alert_time=t + 30,
            first_visible_time=t + 29.4,
            first_detected_time=t + 29.6,
            track_start_time=t + 29.7,
            confirmed_time=t + 30.0,
            discovery_ms=200.0,
            tracking_ms=100.0,
            confirmation_ms=300.0,
            zone_delay_ms=0.0,
            reverse_norfair_status="done",
        )
        # Suppressed FP timing row attached to t-fp in Detections view
        self.trust_store.record_alert_timing(
            review_id="sl_t-fp",
            camera="front_door",
            severity="suppressed_fp",
            alert_time=t + 20,
            discovery_ms=50.0,
            tracking_ms=120.0,
            confirmation_ms=480.0,
            zone_delay_ms=0.0,
            reverse_norfair_status="done",
        )

        # Alerts view: total bar count == 4 (1 normal + 1 rescued_fn + 1 auto_misclass above, 1 human misclass below)
        alert_resp = self.client.get(
            "/review/overview", params={"severity": "alert", "end_time": end_ts}
        )
        self.assertEqual(alert_resp.status_code, 200)
        alert_body = alert_resp.json()
        self.assertEqual(alert_body["golden_total"], 2)
        bar_a = alert_body["buckets"][29]
        self.assertEqual(bar_a["above"]["alerts"], 1)
        self.assertEqual(bar_a["above"]["rescued_fn"], 1)
        self.assertEqual(bar_a["above"]["auto_suppressed_fp"], 0)
        self.assertEqual(bar_a["above"]["auto_misclassified"], 1)
        self.assertEqual(bar_a["above"]["background_clips"], 0)
        self.assertEqual(bar_a["above"]["total"], 3)
        self.assertEqual(bar_a["below"]["misclassified"], 1)
        self.assertEqual(bar_a["below"]["total"], 1)
        self.assertEqual(bar_a["timing"]["count"], 2)
        self.assertAlmostEqual(bar_a["timing"]["avg_discovery_ms"], 150.0)
        self.assertAlmostEqual(bar_a["timing"]["avg_tracking_ms"], 100.0)
        self.assertAlmostEqual(bar_a["timing"]["avg_confirmation_ms"], 200.0)
        self.assertAlmostEqual(bar_a["timing"]["avg_zone_delay_ms"], 25.0)
        self.assertAlmostEqual(bar_a["timing"]["avg_total_ms"], 475.0)
        self.assertAlmostEqual(bar_a["timing"]["slowest_ms"], 600.0)
        self.assertEqual(bar_a["timing"]["slowest_review_id"], "al-2")

        # Per-clip metadata matches chart categories 1-to-1
        meta_a = alert_body["review_meta_by_id"]
        self.assertEqual(meta_a["al-1"]["category"], "alerts")
        self.assertEqual(meta_a["al-2"]["category"], "human_misclassified")
        self.assertEqual(meta_a["al-3"]["category"], "rescued_fn")
        self.assertEqual(meta_a["al-4"]["category"], "auto_misclassified")
        self.assertEqual(meta_a["al-4"]["secondlook"]["misclass_label"], "cat")
        self.assertAlmostEqual(meta_a["al-1"]["timing"]["confirmation_ms"], 100.0)
        self.assertAlmostEqual(meta_a["al-1"]["timing"]["tracking_ms"], 100.0)
        self.assertAlmostEqual(meta_a["al-1"]["timing"]["zone_delay_ms"], 50.0)

        # Single-review timing endpoint for DetailStream sidebar
        t_resp = self.client.get("/review/al-1/timing")
        self.assertEqual(t_resp.status_code, 200)
        t_body = t_resp.json()
        self.assertTrue(t_body["available"])
        self.assertAlmostEqual(t_body["discovery_ms"], 100.0)
        self.assertAlmostEqual(t_body["tracking_ms"], 100.0)
        self.assertAlmostEqual(t_body["confirmation_ms"], 100.0)
        self.assertAlmostEqual(t_body["zone_delay_ms"], 50.0)
        self.assertAlmostEqual(t_body["total_ms"], 350.0)
        self.assertAlmostEqual(t_body["first_detected_time"], t - 0.25, places=3)

        # Detections view: 1 normal det + 1 auto_misclass det + 1 suppressed FP + 1 background clip
        det_resp = self.client.get(
            "/review/overview", params={"severity": "detection", "end_time": end_ts}
        )
        self.assertEqual(det_resp.status_code, 200)
        det_body = det_resp.json()
        bar_d = det_body["buckets"][29]
        self.assertEqual(bar_d["above"]["detections"], 1)
        self.assertEqual(bar_d["above"]["rescued_fn"], 0)
        self.assertEqual(bar_d["above"]["auto_suppressed_fp"], 1)
        self.assertEqual(bar_d["above"]["auto_misclassified"], 1)
        self.assertEqual(bar_d["above"]["background_clips"], 1)
        self.assertEqual(bar_d["above"]["total"], 4)
        self.assertEqual(bar_d["timing"]["segment_count"], 2)
        self.assertEqual(det_body["review_meta_by_id"]["det-2"]["category"], "auto_misclassified")
        self.assertEqual(len(det_body["suppressed_fp_items"]), 1)
        self.assertEqual(det_body["suppressed_fp_items"][0]["track_id"], "t-fp")
        self.assertIsNotNone(det_body["suppressed_fp_items"][0]["timing"])
        self.assertAlmostEqual(
            det_body["suppressed_fp_items"][0]["timing"]["total_ms"], 650.0
        )
        # Alerts view should not include suppressed_fp_items
        self.assertEqual(alert_body["suppressed_fp_items"], [])
        self.assertEqual(bar_a["timing"]["segment_count"], 4)

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

    def test_maintainer_records_alert_timing_lifecycle_and_reverse_update(self) -> None:
        maintainer = ReviewSegmentMaintainer.__new__(ReviewSegmentMaintainer)
        maintainer.trust_store = self.trust_store

        segment = PendingReviewSegment(
            camera="front_door",
            frame_time=6000.8,
            severity=SeverityEnum.alert,
            detections={"obj-1": "person"},
            sub_labels={},
            zones=["porch"],
            audio=set(),
        )
        maintainer._record_alert_timing(
            segment,
            alert_time=6000.8,
            alert_objects=[
                {
                    "id": "obj-1",
                    "first_detect_time": 6000.0,
                    "start_time": 6000.15,
                    "confirmed_time": 6000.55,
                    "secondlook_request_time": 6000.20,
                    "secondlook_answer_time": 6000.48,
                    "accumulated_resource_busy_ms": 27.5,
                }
            ],
        )

        rows = self.trust_store.get_timings_in_range(5999.0, 6002.0, severity="alert")
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["review_id"], segment.id)
        self.assertEqual(row["severity"], "alert")
        self.assertAlmostEqual(row["discovery_ms"], 0.0)
        self.assertAlmostEqual(row["tracking_ms"], 150.0, places=2)
        self.assertAlmostEqual(row["confirmation_ms"], 400.0, places=2)
        self.assertAlmostEqual(row["zone_delay_ms"], 250.0, places=2)
        # Pure telescoping sum: 0 + 150 + 400 + 250 = 800ms (resource_busy_ms NOT double-counted)
        self.assertAlmostEqual(row["total_ms"], 800.0, places=2)
        self.assertAlmostEqual(row["secondlook_request_time"], 6000.20, places=3)
        self.assertAlmostEqual(row["secondlook_answer_time"], 6000.48, places=3)

        # Reverse Norfair asynchronously updates discovery_ms -> total_ms becomes 1150ms
        updated = self.trust_store.update_discovery_timing(
            segment.id,
            first_visible_time=5999.65,
            discovery_ms=350.0,
            status="done",
        )
        self.assertIsNotNone(updated)
        assert updated is not None
        self.assertEqual(updated["reverse_norfair_status"], "done")
        self.assertAlmostEqual(updated["first_visible_time"], 5999.65, places=3)
        self.assertAlmostEqual(updated["discovery_ms"], 350.0, places=2)
        self.assertAlmostEqual(updated["total_ms"], 1150.0, places=2)

        # A subsequent upsert must preserve already-completed Reverse Norfair discovery
        again = self.trust_store.record_alert_timing(
            review_id=segment.id,
            camera="front_door",
            alert_time=6000.8,
            first_detected_time=6000.0,
            track_start_time=6000.15,
            confirmed_time=6000.55,
            discovery_ms=0.0,
            tracking_ms=150.0,
            confirmation_ms=400.0,
            zone_delay_ms=250.0,
        )
        self.assertEqual(again["reverse_norfair_status"], "done")
        self.assertAlmostEqual(again["discovery_ms"], 350.0, places=2)
        self.assertAlmostEqual(again["total_ms"], 1150.0, places=2)

    def test_detection_timing_lifecycle_and_outlier_guard(self) -> None:
        maintainer = ReviewSegmentMaintainer.__new__(ReviewSegmentMaintainer)
        maintainer.trust_store = self.trust_store
        end_ts = 1_700_000_000.0
        t = end_ts - 600.0

        # Detection segment records tracking_ms + confirmation_ms, with zone_delay_ms=0.0
        det_seg = PendingReviewSegment(
            camera="front_door",
            frame_time=t,
            severity=SeverityEnum.detection,
            detections={"obj-det": "cat"},
            sub_labels={},
            zones=[],
            audio=set(),
        )
        self._insert_segment(det_seg.id, start_time=t, severity="detection")
        maintainer._record_alert_timing(
            det_seg,
            alert_time=t,
            alert_objects=[
                {
                    "id": "obj-det",
                    "first_detect_time": t - 0.6,
                    "start_time": t - 0.4,
                    "confirmed_time": t,
                    "accumulated_resource_busy_ms": 42.0,
                }
            ],
            severity="detection",
        )

        # Stationary object reactivated after 700s uses last_active_time (0.4s)
        reactivated_seg = PendingReviewSegment(
            camera="front_door",
            frame_time=t + 10.0,
            severity=SeverityEnum.alert,
            detections={"obj-car": "car"},
            sub_labels={},
            zones=["driveway"],
            audio=set(),
        )
        self._insert_segment(reactivated_seg.id, start_time=t + 10.0, severity="alert")
        maintainer._record_alert_timing(
            reactivated_seg,
            alert_time=t + 10.0,
            alert_objects=[
                {
                    "id": "obj-car",
                    "start_time": t - 700.0,
                    "last_active_time": t + 9.6,
                    "accumulated_resource_busy_ms": 18.0,
                }
            ],
            severity="alert",
        )

        # Stale >30s track with no confirmed_time or recent last_active_time is skipped
        stale_seg = PendingReviewSegment(
            camera="front_door",
            frame_time=t + 20.0,
            severity=SeverityEnum.alert,
            detections={"obj-stale": "car"},
            sub_labels={},
            zones=["driveway"],
            audio=set(),
        )
        self._insert_segment(stale_seg.id, start_time=t + 20.0, severity="alert")
        maintainer._record_alert_timing(
            stale_seg,
            alert_time=t + 20.0,
            alert_objects=[
                {
                    "id": "obj-stale",
                    "start_time": t - 750.0,
                    "accumulated_resource_busy_ms": 10.0,
                }
            ],
            severity="alert",
        )

        # Simulate a legacy pre-fix 720s outlier already in review_trust.db
        self._insert_segment("legacy-720s", start_time=t + 30.0, severity="alert")
        self.trust_store.record_alert_timing(
            review_id="legacy-720s",
            camera="front_door",
            alert_time=t + 30.0,
            discovery_ms=0.0,
            tracking_ms=0.0,
            confirmation_ms=720_000.0,
            severity="alert",
        )

        # Alert overview ignores legacy 720s outlier and sees only reactivated_seg (400ms)
        alert_body = self.client.get(
            "/review/overview", params={"severity": "alert", "end_time": end_ts}
        ).json()
        alert_timing = alert_body["buckets"][29]["timing"]
        self.assertEqual(alert_timing["count"], 1)
        self.assertAlmostEqual(alert_timing["avg_tracking_ms"], 0.0, places=1)
        self.assertAlmostEqual(alert_timing["avg_confirmation_ms"], 400.0, places=1)
        self.assertAlmostEqual(alert_timing["avg_zone_delay_ms"], 0.0, places=1)
        self.assertAlmostEqual(alert_timing["avg_total_ms"], 400.0, places=1)
        self.assertEqual(alert_timing["slowest_review_id"], reactivated_seg.id)

        # Detection overview sees det_seg with tracking_ms=200ms, confirmation_ms=400ms, zone_delay_ms=0ms
        det_body = self.client.get(
            "/review/overview", params={"severity": "detection", "end_time": end_ts}
        ).json()
        det_timing = det_body["buckets"][29]["timing"]
        self.assertEqual(det_timing["count"], 1)
        self.assertAlmostEqual(det_timing["avg_tracking_ms"], 200.0, places=1)
        self.assertAlmostEqual(det_timing["avg_confirmation_ms"], 400.0, places=1)
        self.assertAlmostEqual(det_timing["avg_zone_delay_ms"], 0.0, places=1)
        self.assertAlmostEqual(det_timing["avg_total_ms"], 600.0, places=1)

    def test_legacy_db_migrates_alert_timings_severity_column(self) -> None:
        import sqlite3

        legacy_path = os.path.join(self.tmp.name, "legacy_trust.db")
        with sqlite3.connect(legacy_path) as conn:
            conn.execute(
                """
                CREATE TABLE alert_timings (
                    review_id TEXT PRIMARY KEY,
                    camera TEXT NOT NULL,
                    alert_time REAL NOT NULL,
                    discovery_ms REAL NOT NULL DEFAULT 0.0,
                    resource_busy_ms REAL NOT NULL DEFAULT 0.0,
                    confirmation_ms REAL NOT NULL DEFAULT 0.0,
                    total_ms REAL NOT NULL DEFAULT 0.0
                )
                """
            )
            conn.execute(
                """
                INSERT INTO alert_timings (
                    review_id, camera, alert_time, discovery_ms,
                    resource_busy_ms, confirmation_ms, total_ms
                ) VALUES ('old-1', 'front_door', 1000.0, 0.0, 15.0, 300.0, 315.0)
                """
            )
            conn.commit()

        migrated = ReviewTrustStore(legacy_path, os.path.join(self.tmp.name, "exp2"))
        rows = migrated.get_timings_in_range(900.0, 1100.0, severity="alert")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["review_id"], "old-1")
        self.assertEqual(rows[0]["severity"], "alert")
        self.assertAlmostEqual(rows[0]["tracking_ms"], 0.0)
        self.assertAlmostEqual(rows[0]["zone_delay_ms"], 0.0)


class TestReverseNorfair(unittest.TestCase):
    def test_ring_buffer_sampling_and_reverse_order(self) -> None:
        import numpy as np

        from frigate.review.reverse_norfair import CameraFrameRingBuffer

        buf = CameraFrameRingBuffer(max_frames=5, min_interval=0.2)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        self.assertTrue(buf.push_rgb("cam1", 100.0, frame))
        # Too soon (< 0.2s) -> ignored
        self.assertFalse(buf.push_rgb("cam1", 100.05, frame))
        self.assertTrue(buf.push_rgb("cam1", 100.25, frame))
        self.assertTrue(buf.push_rgb("cam1", 100.50, frame))
        self.assertTrue(buf.push_rgb("cam1", 100.75, frame))

        snap = buf.snapshot_before("cam1", 100.75, lookback_sec=2.0)
        times = [entry[0] for entry in snap]
        self.assertEqual(times, [100.50, 100.25, 100.0])

    def test_reverse_norfair_walks_backward_stops_on_misses_and_preempts(self) -> None:
        import numpy as np

        from frigate.review.reverse_norfair import run_reverse_norfair

        # Synthetic 640x480 camera frames where a person (rendered as a bright box)
        # walks right (+20px/frame):
        # t=100.0..100.4 (not visible), t=100.6 (x=160..220), t=100.8 (x=180..240),
        # t=101.0 (x=200..260), first_detect_time=101.2 (seed_box=[220,150,280,310],
        # seed_box_later=[240,150,300,310]).
        width, height = 640, 480
        box_by_ts: dict[float, list[int] | None] = {
            100.0: None,
            100.2: None,
            100.4: None,
            100.6: [160, 150, 220, 310],  # earliest visible!
            100.8: [180, 150, 240, 310],
            101.0: [200, 150, 260, 310],
        }
        frames_rev = []
        for ts in [101.0, 100.8, 100.6, 100.4, 100.2, 100.0]:
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            box = box_by_ts[ts]
            if box is not None:
                x1, y1, x2, y2 = box
                rgb[y1:y2, x1:x2] = 255
            frames_rev.append((ts, rgb, width, height))

        # Mock detector finds the bright box inside whatever crop Reverse Norfair
        # passes, exercising real crop + box_norm -> full-frame coordinate mapping!
        def mock_detector(crop: np.ndarray) -> tuple[list[dict[str, object]], float]:
            ys, xs = np.where(crop[:, :, 0] > 128)
            if len(xs) == 0:
                return [], 1.0
            ch, cw = crop.shape[:2]
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            return [
                {
                    "label": "person",
                    "score": 0.72,
                    "box_norm": [y1 / ch, x1 / cw, y2 / ch, x2 / cw],
                }
            ], 1.0

        res = run_reverse_norfair(
            frames_rev=frames_rev,
            label="person",
            first_detect_time=101.2,
            seed_box=[220, 150, 280, 310],
            seed_box_later=[240, 150, 300, 310],
            detector_fn=mock_detector,
            input_size=320,
            context_scale=2.5,
            min_score=0.25,
            max_missed_frames=2,
        )
        self.assertFalse(res.preempted)
        self.assertEqual(res.frames_matched, 3)
        # Checked 3 matched frames + stopped after 2 consecutive misses (100.4, 100.2) = 5 frames checked
        self.assertEqual(res.frames_checked, 5)
        self.assertAlmostEqual(res.first_visible_time, 100.6, places=3)
        self.assertAlmostEqual(res.discovery_ms, 600.0, places=1)
        self.assertEqual(res.earliest_box, [160, 150, 220, 310])

        # Preemption check: stops immediately when live work arrives
        calls = 0

        def preempt_after_one() -> bool:
            return calls >= 1

        def counting_detector(crop: np.ndarray) -> tuple[list[dict[str, object]], float]:
            nonlocal calls
            calls += 1
            return mock_detector(crop)

        res_preempt = run_reverse_norfair(
            frames_rev=frames_rev,
            label="person",
            first_detect_time=101.2,
            seed_box=[220, 150, 280, 310],
            detector_fn=counting_detector,
            input_size=320,
            context_scale=2.5,
            should_preempt=preempt_after_one,
        )
        self.assertTrue(res_preempt.preempted)
        self.assertEqual(res_preempt.frames_checked, 1)
        self.assertAlmostEqual(res_preempt.first_visible_time, 101.0, places=3)


if __name__ == "__main__":
    unittest.main()
