"""Unit tests for Secondlook factor math and the false_positive check (become a detection or not)."""

from __future__ import annotations

import unittest

from frigate.secondlook.factor import (
    adjusted_median_scores,
    clamp_factor,
    compute_factor,
)
from frigate.secondlook.detection_check import decide_false_positive


class TestSecondlookFactor(unittest.TestCase):
    def test_max_max_scale_down(self):
        self.assertAlmostEqual(compute_factor(0.4, 0.8), 0.5)

    def test_max_max_scale_up(self):
        self.assertAlmostEqual(compute_factor(0.9, 0.5), 1.8)

    def test_clamp_high(self):
        self.assertAlmostEqual(compute_factor(1.0, 0.2), 2.0)

    def test_clamp_low(self):
        self.assertAlmostEqual(compute_factor(0.1, 1.0), 0.3)

    def test_no_primary(self):
        self.assertIsNone(compute_factor(0.9, 0.0))

    def test_no_sl(self):
        self.assertIsNone(compute_factor(0.0, 0.8))

    def test_clamp_helper(self):
        self.assertEqual(clamp_factor(0.1), 0.3)
        self.assertEqual(clamp_factor(3.0), 2.0)
        self.assertEqual(clamp_factor(1.2), 1.2)

    def test_adjusted_median_unset_factor(self):
        hist = [0.6, 0.7, 0.0, 0.8]
        self.assertEqual(adjusted_median_scores(hist, None), hist)

    def test_adjusted_median_with_factor(self):
        self.assertEqual(adjusted_median_scores([0.6, 0.8, 0.0], 0.5), [0.3, 0.4, 0.0])


class TestSecondlookDetectionCheck(unittest.TestCase):
    def test_latched_tp(self):
        fp, _ = decide_false_positive(
            already_true_positive=True,
            secondlook_enabled=True,
            score_history=[0.1] * 10,
            threshold=0.7,
            factor=0.3,
            pending_since=None,
            frame_time=1.0,
            hold_seconds=1.0,
            running=False,
            pending=False,
        )
        self.assertFalse(fp)

    def test_disabled_stock(self):
        fp, _ = decide_false_positive(
            already_true_positive=False,
            secondlook_enabled=False,
            score_history=[0.6] * 10,
            threshold=0.7,
            factor=None,
            pending_since=None,
            frame_time=1.0,
            hold_seconds=1.0,
            running=False,
            pending=False,
        )
        self.assertTrue(fp)

    def test_hold_while_null(self):
        fp, pending = decide_false_positive(
            already_true_positive=False,
            secondlook_enabled=True,
            score_history=[0.8] * 10,
            threshold=0.7,
            factor=None,
            pending_since=None,
            frame_time=1000.0,
            hold_seconds=1.0,
            running=True,
            pending=True,
        )
        self.assertTrue(fp)
        self.assertEqual(pending, 1000.0)

    def test_timeout_fail_open(self):
        fp, pending = decide_false_positive(
            already_true_positive=False,
            secondlook_enabled=True,
            score_history=[0.8] * 10,
            threshold=0.7,
            factor=None,
            pending_since=1000.0,
            frame_time=1002.0,
            hold_seconds=1.0,
            running=False,
            pending=True,
        )
        self.assertFalse(fp)
        self.assertIsNone(pending)

    def test_scale_down_stays_fp(self):
        fp, pending = decide_false_positive(
            already_true_positive=False,
            secondlook_enabled=True,
            score_history=[0.75] * 10,
            threshold=0.7,
            factor=0.5,
            pending_since=None,
            frame_time=1.0,
            hold_seconds=1.0,
            running=False,
            pending=False,
        )
        self.assertTrue(fp)
        self.assertIsNone(pending)

    def test_scale_up_crosses(self):
        fp, _ = decide_false_positive(
            already_true_positive=False,
            secondlook_enabled=True,
            score_history=[0.5] * 10,
            threshold=0.7,
            factor=1.6,
            pending_since=None,
            frame_time=1.0,
            hold_seconds=1.0,
            running=False,
            pending=False,
        )
        self.assertFalse(fp)

    def test_factor_unset_equals_one(self):
        # after fail-open clear, unscaled 0.8 passes
        fp, _ = decide_false_positive(
            already_true_positive=False,
            secondlook_enabled=True,
            score_history=[0.8] * 10,
            threshold=0.7,
            factor=None,
            pending_since=None,
            frame_time=1002.0,
            hold_seconds=1.0,
            running=False,
            pending=False,
        )
        # arms pending on first call with factor None
        # with pending_since None it will arm and hold!
        # To test unset=1.0 after fail-open, pass pending_since already expired path:
        # Actually with pending_since None and pending False and running False,
        # decide arms and holds. So for "factor unset after timeout" use expired pending.
        fp2, p2 = decide_false_positive(
            already_true_positive=False,
            secondlook_enabled=True,
            score_history=[0.8] * 10,
            threshold=0.7,
            factor=None,
            pending_since=1000.0,
            frame_time=1002.0,
            hold_seconds=1.0,
            running=False,
            pending=False,
        )
        self.assertFalse(fp2)
        self.assertIsNone(p2)


class TestSecondlookHoldOnce(unittest.TestCase):
    def _call(self, **kw):
        base = dict(
            already_true_positive=False,
            secondlook_enabled=True,
            score_history=[0.8] * 10,
            threshold=0.7,
            factor=None,
            pending_since=None,
            frame_time=1000.0,
            hold_seconds=1.0,
        )
        base.update(kw)
        return decide_false_positive(**base)

    def test_hold_arms_without_running_flag(self):
        # the hold arms before the embeddings "pending" payload can arrive
        fp, pending = self._call()
        self.assertTrue(fp)
        self.assertEqual(pending, 1000.0)

    def test_hold_done_does_not_rearm(self):
        # after a no-hit SL result (factor unset) the hold must not re-arm
        fp, pending = self._call(hold_done=True, frame_time=1000.4)
        self.assertFalse(fp)
        self.assertIsNone(pending)

    def test_hold_done_still_fp_when_low(self):
        fp, _ = self._call(hold_done=True, score_history=[0.6] * 10)
        self.assertTrue(fp)

    def test_factor_wins_over_hold(self):
        fp, pending = self._call(factor=0.5, pending_since=1000.0, frame_time=1000.2)
        self.assertTrue(fp)  # 0.8*0.5 < 0.7
        self.assertIsNone(pending)



class TestClassifyExample(unittest.TestCase):
    def test_false_positive(self):
        from frigate.secondlook.save_and_review import classify_example, KIND_FALSE_POSITIVE

        kind = classify_example(
            raw_median=0.85,
            threshold=0.7,
            factor=0.5,
            final_false_positive=True,
        )
        self.assertEqual(kind, KIND_FALSE_POSITIVE)

    def test_false_negative(self):
        from frigate.secondlook.save_and_review import classify_example, KIND_FALSE_NEGATIVE

        kind = classify_example(
            raw_median=0.55,
            threshold=0.7,
            factor=1.5,
            final_false_positive=False,
        )
        self.assertEqual(kind, KIND_FALSE_NEGATIVE)

    def test_misclass_wins(self):
        from frigate.secondlook.save_and_review import classify_example, KIND_MISCLASS

        kind = classify_example(
            raw_median=0.85,
            threshold=0.7,
            factor=0.4,
            final_false_positive=True,
            misclass_label="dog",
            primary_label="person",
        )
        self.assertEqual(kind, KIND_MISCLASS)

    def test_near_tie_skipped(self):
        from frigate.secondlook.save_and_review import classify_example

        self.assertIsNone(
            classify_example(
                raw_median=0.85,
                threshold=0.7,
                factor=0.9,
                final_false_positive=True,
            )
        )
        self.assertIsNone(
            classify_example(
                raw_median=0.55,
                threshold=0.7,
                factor=1.1,
                final_false_positive=False,
            )
        )

    def test_fp_requires_stayed_fp(self):
        from frigate.secondlook.save_and_review import classify_example

        self.assertIsNone(
            classify_example(
                raw_median=0.85,
                threshold=0.7,
                factor=0.4,
                final_false_positive=False,
            )
        )

    def test_fn_requires_became_detection(self):
        from frigate.secondlook.save_and_review import classify_example

        self.assertIsNone(
            classify_example(
                raw_median=0.55,
                threshold=0.7,
                factor=1.5,
                final_false_positive=True,
            )
        )

    def test_gap_edges(self):
        from frigate.secondlook.save_and_review import (
            classify_example,
            KIND_FALSE_POSITIVE,
            KIND_FALSE_NEGATIVE,
        )

        self.assertEqual(
            classify_example(
                raw_median=0.8,
                threshold=0.7,
                factor=0.799,
                final_false_positive=True,
            ),
            KIND_FALSE_POSITIVE,
        )
        self.assertIsNone(
            classify_example(
                raw_median=0.8,
                threshold=0.7,
                factor=0.8,
                final_false_positive=True,
            )
        )
        self.assertEqual(
            classify_example(
                raw_median=0.5,
                threshold=0.7,
                factor=1.201,
                final_false_positive=False,
            ),
            KIND_FALSE_NEGATIVE,
        )


class TestSaveAndReviewStore(unittest.TestCase):
    def setUp(self):
        import tempfile
        from frigate.secondlook.save_and_review import SaveAndReviewStore

        self._tmpdir = tempfile.TemporaryDirectory()
        db = f"{self._tmpdir.name}/examples.db"
        stills = f"{self._tmpdir.name}/stills"
        self.store = SaveAndReviewStore(db, stills)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_deposit_and_mark(self):
        from frigate.secondlook.save_and_review import KIND_FALSE_POSITIVE, MARK_SECONDLOOK_RIGHT

        jpeg = b"\xff\xd8\xfffakejpeg"
        rec = self.store.deposit(
            track_id="cam-1",
            camera="front",
            kind=KIND_FALSE_POSITIVE,
            label="person",
            jpeg_bytes=jpeg,
            primary_score=0.9,
            sl_score=0.2,
            factor=0.0,
            threshold=0.7,
            raw_median=0.85,
            box=[10, 20, 30, 40],
            frame_time=1000.0,
            model_id="yolov9-c-640.onnx",
        )
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec.kind, KIND_FALSE_POSITIVE)
        self.assertIsNone(rec.mark)
        listed = self.store.list(cameras=["front"], reviewed=False)
        self.assertEqual(len(listed), 1)
        marked = self.store.set_mark(rec.id, MARK_SECONDLOOK_RIGHT)
        assert marked is not None
        self.assertEqual(marked.mark, MARK_SECONDLOOK_RIGHT)
        self.assertEqual(self.store.count_unreviewed(), 0)
        cleared = self.store.set_mark(rec.id, None)
        assert cleared is not None
        self.assertIsNone(cleared.mark)


def _det(label: str, score: float, box: list[float]) -> dict:
    return {"label": label, "score": score, "box_norm": box}


class TestSelectMisclass(unittest.TestCase):
    """Misclass is recorded only when there is no same-class SL partner.

    Boxes are [ymin, xmin, ymax, xmax] in crop-normalized coordinates.
    The primary person box is [0.20, 0.30, 0.70, 0.70].
    """

    PERSON = [0.20, 0.30, 0.70, 0.70]
    # IoU vs PERSON: ~0.91
    PERSON_HIT = [0.21, 0.31, 0.69, 0.69]
    # Bicycle under a rider, IoU ~0.31 (above default min_box_iou 0.3, below 0.5)
    BIKE_UNDER = [0.45, 0.32, 0.95, 0.68]
    # Enclosing car, IoU ~0.36
    CAR_AROUND = [0.10, 0.15, 0.90, 0.85]
    # Tight same-box swap, IoU ~0.94
    CAT_SAME = [0.20, 0.30, 0.72, 0.71]
    # Tight bicycle-as-person swap, IoU ~0.89
    BIKE_TIGHT = [0.22, 0.31, 0.71, 0.69]
    THRESHOLD = 0.25

    def _select(self, detections, same_class, box=None):
        from frigate.secondlook.match import select_misclass

        return select_misclass(
            detections,
            "person",
            same_class,
            self.THRESHOLD,
            self.PERSON if box is None else box,
        )

    def test_person_and_bicycle_with_person_hit_is_not_misclass(self):
        person = _det("person", 0.80, self.PERSON_HIT)
        bike = _det("bicycle", 0.84, self.BIKE_UNDER)
        label, score, box = self._select([person, bike], same_class=person)
        self.assertEqual((label, score, box), (None, None, None))

    def test_person_and_car_with_person_hit_is_not_misclass(self):
        person = _det("person", 0.80, self.PERSON_HIT)
        car = _det("car", 0.90, self.CAR_AROUND)
        label, score, box = self._select([person, car], same_class=person)
        self.assertEqual((label, score, box), (None, None, None))

    def test_person_primary_with_only_animal_is_misclass(self):
        cat = _det("cat", 0.81, self.CAT_SAME)
        label, score, box = self._select([cat], same_class=None)
        self.assertEqual(label, "cat")
        self.assertAlmostEqual(score, 0.81)
        self.assertEqual(box, self.CAT_SAME)

    def test_loose_bicycle_without_person_hit_is_not_misclass(self):
        # Overlaps enough for the 0.3 same-class filter, but not closely
        bike = _det("bicycle", 0.84, self.BIKE_UNDER)
        label, score, box = self._select([bike], same_class=None)
        self.assertEqual((label, score, box), (None, None, None))

    def test_enclosing_car_without_person_hit_is_not_misclass(self):
        car = _det("car", 0.90, self.CAR_AROUND)
        label, score, box = self._select([car], same_class=None)
        self.assertEqual((label, score, box), (None, None, None))

    def test_tight_bicycle_swap_without_person_hit_is_misclass(self):
        bike = _det("bicycle", 0.84, self.BIKE_TIGHT)
        label, score, box = self._select([bike], same_class=None)
        self.assertEqual(label, "bicycle")
        self.assertAlmostEqual(score, 0.84)
        self.assertEqual(box, self.BIKE_TIGHT)

    def test_weak_same_class_below_threshold_still_allows_misclass(self):
        # best_same_class would return None for a 0.20 person (threshold 0.25)
        weak = _det("person", 0.20, self.PERSON_HIT)
        cat = _det("cat", 0.81, self.CAT_SAME)
        label, score, box = self._select([weak, cat], same_class=None)
        self.assertEqual(label, "cat")
        self.assertAlmostEqual(score, 0.81)
        self.assertEqual(box, self.CAT_SAME)

    def test_same_class_at_threshold_is_confirmation(self):
        person = _det("person", self.THRESHOLD, self.PERSON_HIT)
        bike = _det("bicycle", 0.84, self.BIKE_TIGHT)
        label, score, box = self._select([person, bike], same_class=person)
        self.assertEqual((label, score, box), (None, None, None))

    def test_missing_primary_box_is_not_misclass(self):
        from frigate.secondlook.match import select_misclass

        cat = _det("cat", 0.81, self.CAT_SAME)
        label, score, box = select_misclass(
            [cat], "person", None, self.THRESHOLD, None
        )
        self.assertEqual((label, score, box), (None, None, None))


class TestHideEdgeSmallList(unittest.TestCase):
    """list() hide_edge_small uses a min-dimension test, not area."""

    def setUp(self):
        import tempfile
        from frigate.secondlook.save_and_review import SaveAndReviewStore

        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = SaveAndReviewStore(
            f"{self._tmpdir.name}/examples.db",
            f"{self._tmpdir.name}/stills",
        )

    def tearDown(self):
        self._tmpdir.cleanup()

    def _insert(self, track_id, box=None, width=1280, height=720):
        import json
        import time
        import uuid
        from pathlib import Path

        example_id = str(uuid.uuid4())
        still = Path(self.store.images_path) / "cam" / f"{example_id}.jpg"
        still.parent.mkdir(parents=True, exist_ok=True)
        still.write_bytes(b"x")
        evidence = None
        if box is not None:
            evidence = json.dumps(
                {"version": 1, "width": width, "height": height, "primary_box": box}
            )
        with self.store._lock, self.store._connect() as conn:
            conn.execute(
                "INSERT INTO examples (id, track_id, camera, kind, label, "
                "still_path, has_clip, created_at, evidence) "
                "VALUES (?, ?, 'cam', 'false_positive', 'person', ?, 0, ?, ?)",
                (example_id, track_id, str(still), time.time(), evidence),
            )
        return example_id

    def test_dimension_rule_on_1280x720(self):
        # 60x200 at the edge is hidden (width 60 < 1280/20)
        tall_thin_edge = self._insert("t", [0, 200, 60, 400])
        # 100x100 at the edge stays visible (both sides >= 1/20)
        square_edge = self._insert("s", [0, 200, 100, 300])
        # small box not at the edge stays visible
        small_centre = self._insert("c", [600, 300, 640, 340])
        # rows without width/height stay visible
        no_size = self._insert("e", box=None)
        ids = {row.id for row in self.store.list(hide_edge_small=(4, 0.05))}
        self.assertNotIn(tall_thin_edge, ids)
        self.assertIn(square_edge, ids)
        self.assertIn(small_centre, ids)
        self.assertIn(no_size, ids)
        self.assertEqual(self.store.list(hide_edge_small=(4, 0.05), count_only=True), 3)
        self.assertEqual(self.store.list(count_only=True), 4)


if __name__ == "__main__":
    unittest.main()
