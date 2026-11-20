"""Live accuracy counters, Secondlook queue items and per-interval stats."""

import tempfile
import threading
import time
import unittest
from queue import Queue
from types import SimpleNamespace

import numpy as np

from frigate.data_processing.real_time.secondlook import SecondlookProcessor
from frigate.secondlook import metrics as sl_metrics
from frigate.test.test_save_and_review import make_config, make_track


class TestAccuracyCounters(unittest.TestCase):
    def setUp(self):
        for kind in sl_metrics._accuracy:
            sl_metrics._accuracy[kind] = 0

    def test_counted_once_per_track_and_kind_even_when_saving_is_off(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(False, td)  # save_and_review disabled
            fp = make_track(config, 0.9, 0.0)
            fn = make_track(config, 0.5, 2)
            data = dict(fn.obj_data)
            for i in range(20):
                fn.update(1001 + i, data, False)
                fp.update(1001 + i, dict(fp.obj_data), False)
            mc = make_track(config, 0.9, 1)
            mc.apply_secondlook({"factor": 1, "misclass_label": "dog"})
            mc.update(1001, dict(mc.obj_data), False)
            stock = make_track(config, 0.9, 1)  # SL agreed: not a correction
            stock.update(1001, dict(stock.obj_data), False)
            # FP / FN are final at track end; ending twice counts once
            for track in (fp, fn, mc, stock, fp, fn):
                track.finish_secondlook()

            stats = sl_metrics.secondlook_stats(config, None)
            self.assertEqual(
                stats["accuracy"],
                {
                    "false_positive": 1,
                    "false_negative": 1,
                    "misclass": 1,
                    "fn_rescued": 1,
                    "fn_not_rescued": 0,
                    "fp_suppressed": 1,
                    "fp_not_suppressed": 0,
                },
            )
            self.assertEqual(fp.save_and_review_decisions, {})

    def test_absent_when_secondlook_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            stats = sl_metrics.secondlook_stats(make_config(False, td, live=False), None)
            self.assertNotIn("accuracy", stats)


class TestQueueItems(unittest.TestCase):
    def make_processor(self, td, queue_max=2):
        proc = SecondlookProcessor.__new__(SecondlookProcessor)
        proc.sl_config = make_config(True, td).secondlook
        proc._task_queue = Queue(maxsize=queue_max)
        proc._queue_items = 0
        proc._queue_rises = {}
        proc._items_lock = threading.Lock()
        proc._queue_dropped = 0
        proc._tracks = {}
        proc.model_id = "sl"
        proc._samples = 0
        proc._camera_counts = {}
        proc._emit_result = lambda r: None
        proc.metrics = SimpleNamespace(secondlook={})
        proc.model = SimpleNamespace(
            available=True,
            mock=False,
            infer=lambda crop: ([], 1),
            best_same_class=lambda *a: None,
        )
        return proc

    def test_items_include_the_one_in_process_and_record_rises(self):
        with tempfile.TemporaryDirectory() as td:
            proc = self.make_processor(td)
            task = {
                "kind": "infer",
                "object_id": "x",
                "camera": "cam",
                "label": "person",
                "primary_score": 0.8,
                "frame_time": 1,
                "crop": np.zeros((10, 10, 3)),
            }
            self.assertTrue(proc._enqueue_task(dict(task)))
            self.assertTrue(proc._enqueue_task(dict(task)))
            self.assertFalse(proc._enqueue_task(dict(task)))  # full: dropped
            self.assertFalse(proc._enqueue_task({"kind": "expire", "object_id": "x"}))
            sl = proc.metrics.secondlook
            self.assertEqual((sl["queue_items"], sl["queue_dropped"]), (2, 2))
            self.assertEqual(sl["queue_rises"], {1: 1, 2: 1})
            seen = []
            proc.model.infer = lambda crop: (seen.append(sl["queue_items"]), ([], 1))[1]
            proc._process_task(proc._task_queue.get_nowait())
            self.assertEqual(seen, [2])  # still counted while in process
            self.assertEqual(sl["queue_items"], 1)
            proc._process_task(proc._task_queue.get_nowait())
            self.assertEqual(sl["queue_items"], 0)


class TestInterval(unittest.TestCase):
    def setUp(self):
        sl_metrics._previous_interval = None

    def test_completed_dropped_and_peak_items_per_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(False, td)
            values = {"available": True, "samples": 5, "queue_dropped": 1}
            metrics = SimpleNamespace(secondlook=values)

            def interval(**update):
                values.update(update)
                return sl_metrics.secondlook_stats(config, metrics)["interval"]

            self.assertIsNone(interval())  # first snapshot has no interval
            # one analysis at a time: busy without queuing
            self.assertEqual(
                interval(samples=7, queue_rises={1: 2}),
                {"completed": 2, "dropped": 0, "queue_peak": 1},
            )
            # idle: no completions (chart gap) and peak 0 (not a gap)
            self.assertEqual(
                interval(), {"completed": 0, "dropped": 0, "queue_peak": 0}
            )
            # backlog that drained, with drops
            rises = {level: 1 for level in range(1, 10)}
            rises[1] = 3
            self.assertEqual(
                interval(samples=16, queue_dropped=4, queue_rises=rises),
                {"completed": 9, "dropped": 3, "queue_peak": 9},
            )
            # items still present at the start of an interval count
            values["queue_items"] = 3
            interval()
            self.assertEqual(interval()["queue_peak"], 3)

    def test_stats_expose_items_and_capacity_only_with_a_real_model(self):
        with tempfile.TemporaryDirectory() as td:
            config = make_config(False, td)
            metrics = SimpleNamespace(
                secondlook={"available": True, "samples": 3, "queue_items": 4}
            )
            stats = sl_metrics.secondlook_stats(config, metrics)
            self.assertEqual(stats["queue_items"], 4)
            self.assertEqual(stats["queue_max"], config.secondlook.queue_max)
            metrics.secondlook["available"] = False
            stats = sl_metrics.secondlook_stats(config, metrics)
            self.assertIsNone(stats["queue_items"])
            self.assertIsNone(stats["interval"])


if __name__ == "__main__":
    unittest.main()
