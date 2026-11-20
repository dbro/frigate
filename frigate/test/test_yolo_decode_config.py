"""Configurable YOLO decode: min_score, nms_threshold, max_detections, nms_class_groups."""

import unittest

import numpy as np
from pydantic import ValidationError

from frigate.config.classification import SecondlookModelConfig
from frigate.detectors.plugins.onnx import ONNXDetectorConfig
from frigate.detectors.plugins.openvino import OvDetectorConfig
from frigate.util.model import build_nms_group_lookup, post_process_yolo

LABELS = {0: "person", 1: "bicycle", 2: "car", 3: "truck", 4: "dog"}
GROUPS = ["person", ["car", "truck"], "bicycle"]
# (cx, cy, w, h, class, score)
BOXES = [
    (160, 170, 80, 150, 0, 0.90),  # person on a bicycle (IoU 0.67 with the bicycle)
    (160, 195, 80, 100, 1, 0.80),
    (100, 100, 120, 80, 2, 0.85),  # car / truck duplicate (IoU 0.94)
    (104, 100, 120, 80, 3, 0.60),
    (250, 250, 40, 40, 4, 0.70),  # same-class duplicate
    (252, 250, 40, 40, 4, 0.50),
    (40, 40, 20, 20, 4, 0.30),  # below the default min_score
]


def head() -> np.ndarray:
    p = np.zeros((1, 4 + len(LABELS), 100), np.float32)
    for j, (cx, cy, w, h, c, s) in enumerate(BOXES):
        p[0, :4, j] = (cx, cy, w, h)
        p[0, 4 + c, j] = s
    return p


def names(d: np.ndarray) -> list[str]:
    return [f"{LABELS[int(r[0])]} {r[1]:.2f}" for r in d[d[:, 1] > 0]]


class TestYoloDecode(unittest.TestCase):
    def test_defaults_are_the_old_fixed_values(self):
        d = post_process_yolo([head()], 320, 320)
        self.assertEqual(d.shape, (20, 6))
        self.assertEqual(names(d), ["person 0.90", "car 0.85", "dog 0.70"])
        explicit = post_process_yolo(
            [head()], 320, 320, min_score=0.4, nms_threshold=0.4, max_detections=20
        )
        self.assertEqual(d.tobytes(), explicit.tobytes())

    def test_groups_keep_boxes_of_other_groups(self):
        lookup = build_nms_group_lookup(GROUPS, LABELS)
        self.assertEqual(lookup.tolist(), [1, 3, 2, 2, 0])
        d = post_process_yolo([head()], 320, 320, class_groups=lookup)
        self.assertEqual(
            names(d), ["person 0.90", "car 0.85", "bicycle 0.80", "dog 0.70"]
        )

    def test_min_score_threshold_and_max_detections(self):
        low = post_process_yolo([head()], 320, 320, min_score=0.25)
        self.assertIn("dog 0.30", names(low))
        loose = post_process_yolo([head()], 320, 320, nms_threshold=0.7)
        self.assertIn("bicycle 0.80", names(loose))
        three = post_process_yolo([head()], 320, 320, max_detections=2)
        self.assertEqual((three.shape, len(names(three))), ((20, 6), 2))
        self.assertEqual(
            post_process_yolo([head()], 320, 320, max_detections=50).shape, (50, 6)
        )

    def test_unknown_names_warn_and_unset_means_none(self):
        self.assertIsNone(build_nms_group_lookup(None, LABELS))
        with self.assertLogs("frigate.util.model", level="WARNING") as logs:
            lookup = build_nms_group_lookup(["person", "fox"], LABELS)
        self.assertIn("'fox' is not a label of this model", logs.output[0])
        self.assertEqual(lookup.tolist(), [1, 0, 0, 0, 0])

    def test_config_validation(self):
        for cls, extra in (
            (OvDetectorConfig, {"type": "openvino"}),
            (ONNXDetectorConfig, {"type": "onnx"}),
            (SecondlookModelConfig, {}),
        ):
            d = cls(**extra)
            self.assertEqual(
                (d.min_score, d.nms_threshold, d.max_detections, d.nms_class_groups),
                (0.4, 0.4, 20, None),
            )
            for bad in (
                {"nms_class_groups": ["person", ["car", "person"]]},
                {"nms_class_groups": [[]]},
                {"min_score": 0},
                {"nms_threshold": 1.5},
                {"max_detections": 301},
            ):
                with self.assertRaises(ValidationError, msg=f"{cls.__name__} {bad}"):
                    cls(**extra, **bad)
            ok = cls(**extra, nms_class_groups=GROUPS + [["dog", "cat"]])
            self.assertEqual(ok.nms_class_groups[1], ["car", "truck"])
        with self.assertRaises(ValidationError):
            SecondlookModelConfig(min_scor=0.2)


if __name__ == "__main__":
    unittest.main()
