"""Retention cleanup of the Secondlook save-and-review store, on real SQLite + files."""

import os
import sqlite3
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from frigate.config.classification import SecondlookSaveAndReviewRetentionConfig
from frigate.secondlook import metrics as sl_metrics
from frigate.secondlook.retention import (
    GIB,
    run_retention,
    start_retention,
)
from frigate.secondlook.save_and_review import SaveAndReviewStore
from frigate.test.test_save_and_review import make_config

DAY = 86400
NOW = 2_000_000_000.0


def retention(**kwargs):
    return SecondlookSaveAndReviewRetentionConfig(**kwargs)


class RetentionCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.store = SaveAndReviewStore(
            str(self.dir / "db" / "examples.db"), str(self.dir / "images")
        )

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, captured_days_ago, mark=None, marked_days_ago=None, size=10):
        rec = self.store.deposit(
            track_id="t",
            camera="cam",
            kind="false_positive",
            label="person",
            jpeg_bytes=b"x" * size,
            frame_time=NOW - captured_days_ago * DAY,
        )
        with sqlite3.connect(self.store.db_path) as conn:
            conn.execute(
                "UPDATE examples SET mark = ?, marked_at = ?, created_at = ? WHERE id = ?",
                (
                    mark,
                    None if marked_days_ago is None else NOW - marked_days_ago * DAY,
                    NOW - captured_days_ago * DAY,
                    rec.id,
                ),
            )
        return rec

    def ids(self):
        with sqlite3.connect(self.store.db_path) as conn:
            return {r[0] for r in conn.execute("SELECT id FROM examples")}


class TestAgeRules(RetentionCase):
    def test_unreviewed_by_capture_ignored_by_mark_accepted_never(self):
        old_unreviewed = self.add(31)
        new_unreviewed = self.add(29)
        # captured long ago, but ignored only 3 days ago: kept
        recently_ignored = self.add(100, "ignore", 3)
        old_ignored = self.add(10, "ignore", 8)
        accepted = [
            self.add(400, "primary_right", 400),
            self.add(400, "secondlook_right", 400),
        ]
        result = run_retention(self.store, retention(), now=NOW)
        self.assertEqual(
            self.ids(),
            {new_unreviewed.id, recently_ignored.id} | {r.id for r in accepted},
        )
        self.assertEqual((result["deleted_rows"], result["deleted_files"]), (2, 2))
        self.assertFalse(Path(old_unreviewed.still_path).exists())
        self.assertFalse(Path(old_ignored.still_path).exists())
        for rec in accepted:
            self.assertTrue(Path(rec.still_path).exists())

    def test_zero_or_null_disables_each_rule(self):
        self.add(1000)
        self.add(1000, "ignore", 1000)
        cfg = retention(
            unreviewed_days=0, ignored_days=None, summary_days=0, analyses_days=0
        )
        self.assertEqual(run_retention(self.store, cfg, now=NOW)["deleted_rows"], 0)
        self.assertEqual(len(self.ids()), 2)
        self.assertIsNone(start_retention(self.store, cfg))


class TestSizeCap(RetentionCase):
    def test_oldest_unreviewed_then_ignored_never_accepted(self):
        accepted = self.add(50, "secondlook_right", 50, size=4000)
        ignored_old = self.add(5, "ignore", 2, size=1000)
        ignored_new = self.add(5, "ignore", 1, size=1000)
        unreviewed_old = self.add(3, size=1000)
        unreviewed_new = self.add(1, size=1000)
        db_bytes = os.path.getsize(self.store.db_path)
        images = 8000
        # one unreviewed must go
        cfg = retention(max_size_gb=(db_bytes + images - 500) / GIB)
        run_retention(self.store, cfg, now=NOW)
        self.assertNotIn(unreviewed_old.id, self.ids())
        self.assertIn(unreviewed_new.id, self.ids())
        # all unreviewed, then the oldest-marked ignored
        cfg = retention(max_size_gb=(db_bytes + images - 2500) / GIB)
        run_retention(self.store, cfg, now=NOW)
        self.assertEqual(self.ids(), {accepted.id, ignored_new.id})
        # a cap below the accepted data keeps every accepted row anyway
        run_retention(self.store, retention(max_size_gb=1e-12), now=NOW)
        self.assertEqual(self.ids(), {accepted.id})
        self.assertTrue(Path(accepted.still_path).exists())
        self.assertFalse(Path(ignored_old.still_path).exists())


class TestJobThread(RetentionCase):
    def test_runs_at_start_and_stops(self):
        old = self.add(10_000)
        stop = threading.Event()
        stop.set()  # run once, then the daily wait returns immediately
        thread = start_retention(self.store, retention(), stop_event=stop)
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(thread.name, "secondlook_retention")
        self.assertNotIn(old.id, self.ids())
        self.assertEqual(self.store.last_retention["deleted_rows"], 1)


class TestFilesAndRepeat(RetentionCase):
    def test_missing_file_does_not_stop_the_job(self):
        gone = self.add(40)
        present = self.add(40)
        os.remove(gone.still_path)
        result = run_retention(self.store, retention(), now=NOW)
        self.assertEqual(self.ids(), set())
        self.assertEqual(
            (result["deleted_rows"], result["deleted_files"], result["missing_files"]),
            (2, 1, 1),
        )
        self.assertFalse(Path(present.still_path).exists())

    def test_orphan_sweep_only_touches_old_store_files_in_images_dir(self):
        kept = self.add(1)
        cam = Path(self.store.images_path) / "cam"
        orphan = cam / f"{uuid.uuid4()}.jpg"
        fresh_orphan = cam / f"{uuid.uuid4()}.jpg"
        other = [cam / "notes.txt", cam / "snapshot.jpg", Path(self.store.images_path) / f"{uuid.uuid4()}.jpg"]
        outside_dir = self.dir / "elsewhere"
        outside_dir.mkdir()
        outside = outside_dir / f"{uuid.uuid4()}.jpg"
        (Path(self.store.images_path) / "linked").symlink_to(outside_dir)
        for path in [orphan, fresh_orphan, *other, outside]:
            path.write_bytes(b"x")
        old = time.time() - 2 * 3600
        for path in [orphan, *other, outside, Path(kept.still_path)]:
            os.utime(path, (old, old))
        result = run_retention(self.store, retention(), now=time.time())
        self.assertEqual(result["deleted_orphans"], 1)
        self.assertFalse(orphan.exists())
        for path in [fresh_orphan, *other, outside, Path(kept.still_path)]:
            self.assertTrue(path.exists(), path)

    def test_rerun_after_crash_between_commit_and_file_removal(self):
        victim = self.add(40)
        keep = self.add(1)
        with patch("frigate.secondlook.retention.os.remove", side_effect=SystemExit):
            with self.assertRaises(SystemExit):
                run_retention(self.store, retention(), now=NOW)
        # row gone (committed), its file left behind as an orphan
        self.assertEqual(self.ids(), {keep.id})
        self.assertTrue(Path(victim.still_path).exists())
        old = time.time() - 2 * 3600
        os.utime(victim.still_path, (old, old))
        first = run_retention(self.store, retention(), now=time.time())
        self.assertEqual((first["deleted_rows"], first["deleted_orphans"]), (0, 1))
        self.assertFalse(Path(victim.still_path).exists())
        second = run_retention(self.store, retention(), now=time.time())
        self.assertEqual(
            (second["deleted_rows"], second["deleted_files"], second["deleted_orphans"]),
            (0, 0, 0),
        )
        self.assertTrue(Path(keep.still_path).exists())


class TestMigrationJobAndStats(unittest.TestCase):
    def test_marked_rows_without_marked_at_get_a_full_period_from_now(self):
        with tempfile.TemporaryDirectory() as td:
            store = SaveAndReviewStore(f"{td}/e.db", f"{td}/img")
            rec = store.deposit(
                track_id="t", camera="cam", kind="misclass", label="car",
                jpeg_bytes=b"jpeg", frame_time=1.0,
            )
            with sqlite3.connect(store.db_path) as conn:
                conn.execute(
                    "UPDATE examples SET mark='ignore', marked_at=NULL, created_at=1 WHERE id=?",
                    (rec.id,),
                )
            before = time.time()
            reopened = SaveAndReviewStore(f"{td}/e.db", f"{td}/img")
            row = reopened.get(rec.id)
            self.assertEqual(row.mark, "ignore")
            self.assertGreaterEqual(row.marked_at, before)
            run_retention(reopened, retention())
            self.assertIsNotNone(reopened.get(rec.id))

    def test_job_starts_only_when_save_and_review_is_enabled(self):
        from frigate.app import FrigateApp

        with tempfile.TemporaryDirectory() as td:
            for enabled in (False, True):
                app = SimpleNamespace(config=make_config(enabled, td))
                with patch("frigate.app.start_retention") as start:
                    FrigateApp.init_secondlook_save_and_review(app)
                self.assertEqual(start.called, enabled)

    def test_last_run_in_storage_stats(self):
        from frigate.secondlook.save_and_review import configure_save_and_review

        with tempfile.TemporaryDirectory() as td:
            config = make_config(True, td)
            store = configure_save_and_review(config.secondlook.save_and_review)
            stats = sl_metrics.secondlook_stats(config, None)
            self.assertIsNone(stats["storage"]["retention"])
            result = run_retention(store, config.secondlook.save_and_review.retention)
            stats = sl_metrics.secondlook_stats(config, None)
            self.assertEqual(stats["storage"]["retention"], result)


if __name__ == "__main__":
    unittest.main()
