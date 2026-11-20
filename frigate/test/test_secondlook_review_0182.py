"""0182 batch: per-camera SL rates, saved-data disk use, small-edge filter, verdicts."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from frigate.secondlook import metrics as sl_metrics
from frigate.secondlook.save_and_review import _SCHEMA, SaveAndReviewStore
from frigate.test.test_save_and_review import TestSaveAndReviewAPI, make_config


def evidence(box, width=1280, height=720):
    return {"version": 1, "width": width, "height": height, "primary_box": box}


class TestCameraRates(unittest.TestCase):
    def setUp(self):
        sl_metrics._previous_counts.clear()

    def test_rates_from_cumulative_counts_and_honest_absence(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(False, td)
            values = {"available": True, "cameras": {}}
            metrics = SimpleNamespace(secondlook=values)
            # first snapshot establishes a baseline; no fabricated zero rate
            self.assertEqual(sl_metrics.camera_rates(config, metrics, now=10.0), {})
            values["cameras"] = {"cam": {"analysed": 30, "skipped": 6}}
            self.assertEqual(
                sl_metrics.camera_rates(config, metrics, now=25.0),
                {"cam": {"secondlook_analysed_ps": 2.0, "secondlook_skipped_ps": 0.4}},
            )
            values["available"] = False
            self.assertEqual(sl_metrics.camera_rates(config, metrics, now=40.0), {})
            config.secondlook.enabled = False
            values["available"] = True
            self.assertEqual(sl_metrics.camera_rates(config, metrics, now=55.0), {})

    def test_processor_counts_queue_full_and_crop_failure_as_skipped(self):
        from frigate.data_processing.real_time.secondlook import SecondlookProcessor

        with tempfile.TemporaryDirectory() as td:
            config = make_config(True, td)
            proc = SecondlookProcessor.__new__(SecondlookProcessor)
            proc.config = config
            proc.sl_config = config.secondlook
            proc.model = SimpleNamespace(input_size=640)
            proc._tracks = {}
            proc.model_id = "sl"
            proc._camera_counts = {}
            proc.metrics = SimpleNamespace(secondlook={})
            proc._emit_pending = lambda *a: None
            proc._enqueue_task = lambda t: False  # bounded queue full
            proc._emit_result = lambda r: None
            image = np.zeros((720, 640), np.uint8)
            obj = {
                "id": "t",
                "camera": "cam",
                "label": "person",
                "score": 0.8,
                "box": [10, 20, 100, 200],
                "frame_time": 1,
            }
            proc.process_frame(obj, image)
            self.assertEqual(
                proc.metrics.secondlook["cameras"]["cam"],
                {"analysed": 0, "skipped": 1},
            )
            proc._copy_crop = lambda *a: None
            proc.process_frame({**obj, "id": "u"}, image)
            self.assertEqual(proc.metrics.secondlook["cameras"]["cam"]["skipped"], 2)


class TestSavedDataUsage(unittest.TestCase):
    def test_database_wal_and_images_counted_and_cached(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "s.db"
            db.write_bytes(b"x" * 100)
            Path(str(db) + "-wal").write_bytes(b"x" * 10)
            images = Path(td) / "images" / "cam"
            images.mkdir(parents=True)
            (images / "a.jpg").write_bytes(b"y" * 7)
            (images / "b.jpg").write_bytes(b"y" * 3)
            usage = sl_metrics.saved_data_usage(str(db), str(images.parent), now=1.0)
            self.assertEqual(
                usage, {"database_bytes": 110, "images_bytes": 10, "image_files": 2}
            )
            (images / "c.jpg").write_bytes(b"y")
            self.assertEqual(
                sl_metrics.saved_data_usage(str(db), str(images.parent), now=30.0),
                usage,
            )
            self.assertEqual(
                sl_metrics.saved_data_usage(str(db), str(images.parent), now=100.0)[
                    "image_files"
                ],
                3,
            )


class TestEdgeSmallFilter(TestSaveAndReviewAPI):
    def test_filter_hides_only_small_edge_boxes_and_keeps_data(self):
        store = self.store
        common = dict(camera="cam", kind="false_positive", label="person")
        # 40x40 at the left edge — below 1/20 of both sides
        small_edge = store.deposit(
            track_id="a", jpeg_bytes=b"1", evidence=evidence([0, 300, 40, 340]), **common
        )
        # 60x200 at the left edge: width 60 < 1280/20, should be hidden
        tall_thin_edge = store.deposit(
            track_id="t", jpeg_bytes=b"6", evidence=evidence([0, 200, 60, 400]), **common
        )
        # 100x100 at the left edge: both sides >= 1/20, stays visible
        square_edge = store.deposit(
            track_id="s", jpeg_bytes=b"7", evidence=evidence([0, 200, 100, 300]), **common
        )
        big_edge = store.deposit(
            track_id="b", jpeg_bytes=b"2", evidence=evidence([0, 100, 300, 700]), **common
        )
        # 40x40 in the centre — small, but not at the edge
        small_centre = store.deposit(
            track_id="c", jpeg_bytes=b"3", evidence=evidence([600, 300, 640, 340]), **common
        )
        right_edge = store.deposit(
            track_id="d",
            jpeg_bytes=b"4",
            evidence=evidence([1240, 300, 1277, 340]),
            **common,
        )
        no_size = store.deposit(track_id="e", jpeg_bytes=b"5", **common)
        filtered = self.client.get(
            "/secondlook/save_and_review", params={"hide_edge_small": True}
        ).json()
        ids = {e["id"] for e in filtered["examples"]}
        self.assertNotIn(small_edge.id, ids)
        self.assertNotIn(tall_thin_edge.id, ids)
        self.assertNotIn(right_edge.id, ids)
        for kept in (square_edge, big_edge, small_centre, no_size):
            self.assertIn(kept.id, ids)
        self.assertIn(self.example_id, ids)
        self.assertEqual(filtered["hidden_edge_small"], 3)
        self.assertEqual(filtered["total"], len(ids))
        self.assertEqual(filtered["unreviewed"], len(ids))
        everything = self.client.get("/secondlook/save_and_review").json()
        self.assertEqual(everything["total"], len(ids) + 3)
        self.assertEqual(everything["hidden_edge_small"], 0)
        self.assertTrue(Path(small_edge.still_path).exists())

    def test_cutoffs_come_from_config(self):
        self.config.secondlook.save_and_review.review.small_dim_fraction = 0.5
        big_edge = self.store.deposit(
            track_id="b",
            camera="cam",
            kind="false_positive",
            label="person",
            jpeg_bytes=b"2",
            evidence=evidence([0, 100, 300, 700]),
        )
        ids = {
            e["id"]
            for e in self.client.get(
                "/secondlook/save_and_review", params={"hide_edge_small": True}
            ).json()["examples"]
        }
        self.assertNotIn(big_edge.id, ids)

    def test_defaults(self):
        review = self.config.secondlook.save_and_review.review
        self.assertTrue(review.hide_edge_small)
        self.assertEqual(review.edge_margin_px, 4)
        self.assertEqual(review.small_dim_fraction, 0.05)
        self.assertEqual(review.group_gap_seconds, 60.0)

    def test_legacy_small_area_fraction_is_ignored(self):
        from frigate.config.classification import SecondlookSaveAndReviewReviewConfig

        with self.assertLogs("frigate.config.classification", level="WARNING") as logs:
            review = SecondlookSaveAndReviewReviewConfig(small_area_fraction=0.005)
        self.assertEqual(review.small_dim_fraction, 0.05)
        self.assertFalse(hasattr(review, "small_area_fraction"))
        self.assertTrue(any("small_area_fraction" in line for line in logs.output))


class TestVerdicts(TestSaveAndReviewAPI):
    def test_three_verdicts_editable_and_old_values_refused(self):
        base = f"/secondlook/save_and_review/{self.example_id}/mark"
        for mark in ("primary_right", "secondlook_right", "ignore", None):
            response = self.client.post(base, json={"mark": mark})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(self.store.get(self.example_id).mark, mark)
        for old in ("accept", "reject"):
            self.assertEqual(self.client.post(base, json={"mark": old}).status_code, 400)
        self.assertEqual(
            self.client.get(
                "/secondlook/save_and_review", params={"mark": "primary_right"}
            ).json()["total"],
            0,
        )


class TestLegacyMarkMigration(unittest.TestCase):
    def test_mapping_is_lossless_idempotent_and_backed_up(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "old.db")
            with sqlite3.connect(path) as conn:
                conn.executescript(_SCHEMA)
                for id, mark, marked_at in (
                    ("a", "accept", 5.0),
                    ("r", "reject", 6.0),
                    ("u", None, None),
                ):
                    conn.execute(
                        "INSERT INTO examples (id,track_id,camera,kind,label,still_path,mark,created_at,marked_at)"
                        " VALUES (?,?,?,?,?,?,?,?,?)",
                        (id, id, "cam", "misclass", "car", f"{td}/{id}.jpg", mark, 1, marked_at),
                    )
            store = SaveAndReviewStore(path, td)
            got = {id: store.get(id) for id in "aru"}
            self.assertEqual(
                [(got[i].mark, got[i].legacy_mark, got[i].marked_at) for i in "aru"],
                [("secondlook_right", "accept", 5.0), ("ignore", "reject", 6.0), (None, None, None)],
            )
            with sqlite3.connect(path + ".pre-verdict.bak") as backup:
                self.assertEqual(
                    dict(backup.execute("SELECT id, mark FROM examples")),
                    {"a": "accept", "r": "reject", "u": None},
                )
            store.set_mark("a", "primary_right")
            reopened = SaveAndReviewStore(path, td)
            self.assertEqual(reopened.get("a").mark, "primary_right")
            self.assertEqual(reopened.get("a").legacy_mark, "accept")
            self.assertEqual(len(reopened.list()), 3)


if __name__ == "__main__":
    unittest.main()
