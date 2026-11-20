"""Frigate+ YOLOv9 ONNX support for Secondlook: metadata and compatibility check.

These tests use SYNTHETIC ONNX fixtures shaped like a Frigate+ YOLOv9 ONNX
export (single [1, 4 + classes, anchors] output). No real Frigate+ model file
is available in this environment.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:  # pragma: no cover - fixture builder only
    onnx = None

try:
    import openvino  # noqa: F401
except ImportError:  # pragma: no cover
    openvino = None

from frigate.secondlook.model import SecondlookModel, plus_compatibility_problems

PLUS_LABELS = {"0": "person", "1": "face", "2": "car", "3": "dog", "4": "cat"}


def plus_info(**overrides):
    info = {
        "id": "synthetic",
        "name": "synthetic yolov9s",
        "type": "yolo-generic",
        "width": 320,
        "height": 320,
        "inputShape": "nchw",
        "pixelFormat": "rgb",
        "inputDataType": "float",
        "labelMap": PLUS_LABELS,
        "supportedDetectors": ["onnx", "openvino"],
    }
    info.update(overrides)
    return info


def write_yolo_onnx(path, size=320, classes=5, hit_class=2, layout="nchw"):
    """Constant YOLO head: one centred box of hit_class at 0.9, independent of pixels."""
    anchors = 100  # anchors > channels, as in real exports (2100 at 320)
    pred = np.zeros((1, 4 + classes, anchors), np.float32)
    pred[0, :4, 0] = [size / 2, size / 2, size / 2, size / 2]
    pred[0, 4 + hit_class, 0] = 0.9
    shape = [1, 3, size, size] if layout == "nchw" else [1, size, size, 3]
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
            helper.make_node("Mul", ["mean", "zero"], ["nothing"]),
            helper.make_node("Add", ["pred", "nothing"], ["output0"]),
        ],
        "synthetic_yolov9",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)],
        [
            helper.make_tensor_value_info(
                "output0", TensorProto.FLOAT, [1, 4 + classes, anchors]
            )
        ],
        [
            numpy_helper.from_array(pred, "pred"),
            numpy_helper.from_array(np.zeros((), np.float32), "zero"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)


@unittest.skipIf(onnx is None or openvino is None, "onnx/openvino not installed")
class TestPlusModelLoading(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        # Frigate+ cache layout: model file has no extension, info is <id>.json
        self.model = self.dir / "synthetic"
        self.info = self.dir / "synthetic.json"

    def tearDown(self):
        self.temp.cleanup()

    def load(self, info=None, required=("person", "car"), **onnx_args):
        write_yolo_onnx(str(self.model), **onnx_args)
        self.info.write_text(json.dumps(plus_info() if info is None else info))
        return SecondlookModel(
            model_path=str(self.model),
            labelmap_path=str(self.dir / "unused-coco-labelmap.txt"),
            device="CPU",
            input_size=640,
            plus_metadata_path=str(self.info),
            required_labels=required,
        )

    def test_labels_and_input_come_from_metadata(self):
        model = self.load()
        self.assertIsNone(model.refusal)
        self.assertFalse(model.mock)
        self.assertEqual(model.source, "frigate_plus")
        self.assertEqual(model.input_size, 320)  # not the configured 640
        self.assertEqual(model.labelmap[2], "car")
        detections, _ = model.infer(np.zeros((200, 150, 3), np.uint8))
        self.assertEqual([d["label"] for d in detections], ["car"])
        self.assertAlmostEqual(detections[0]["score"], 0.9, places=5)

    def test_nhwc_bgr_denorm_supported(self):
        model = self.load(
            plus_info(inputShape="nhwc", pixelFormat="bgr", inputDataType="float_denorm"),
            layout="nhwc",
        )
        self.assertIsNone(model.refusal)
        tensor = model.preprocess(np.full((10, 10, 3), [255, 0, 0], np.uint8))
        self.assertEqual(tensor.shape, (1, 320, 320, 3))
        self.assertEqual(tensor[0, 0, 0].tolist(), [0.0, 0.0, 255.0])  # RGB->BGR, no /255

    def assert_refused(self, model, fragment):
        self.assertTrue(model.mock, "refused model must not run inference")
        self.assertIn(fragment, model.refusal)
        self.assertEqual(model.infer(np.zeros((20, 20, 3), np.uint8)), ([], 0.0))

    def test_refusals(self):
        cases = [
            (plus_info(type="yolonas"), "unsupported"),
            (plus_info(supportedDetectors=["edgetpu"]), "not an ONNX/OpenVINO"),
            (plus_info(width=320, height=240), "square"),
            (plus_info(inputDataType="int"), "data type"),
            (plus_info(labelMap={"0": "person", "1": "car"}), "channels"),
            (plus_info(width=640, height=640), "input shape"),
            ({k: v for k, v in plus_info().items() if k != "labelMap"}, "missing labelMap"),
        ]
        for info, fragment in cases:
            with self.subTest(fragment=fragment):
                self.assert_refused(self.load(info), fragment)

    def test_missing_configured_label_refused_not_suppressed(self):
        self.assert_refused(
            self.load(required=("person", "bicycle")), "configured Secondlook objects: bicycle"
        )

    def test_missing_file_and_unreadable_info(self):
        self.info.write_text("{not json")
        model = SecondlookModel(
            str(self.model), "", "CPU", plus_metadata_path=str(self.info)
        )
        self.assert_refused(model, "unreadable")
        self.info.write_text(json.dumps(plus_info()))
        model = SecondlookModel(
            str(self.model), "", "CPU", plus_metadata_path=str(self.info)
        )
        self.assert_refused(model, "model file missing")


class TestPlusUrl(unittest.TestCase):
    def test_plus_url_resolves_to_local_cache_and_runs_same_check(self):
        from unittest.mock import patch

        from frigate.config.classification import SecondlookModelConfig

        with tempfile.TemporaryDirectory() as td:
            with patch("frigate.config.classification.MODEL_CACHE_DIR", td):
                cfg = SecondlookModelConfig(path="plus://abc123")
                kept = SecondlookModelConfig(
                    path="plus://abc123", plus_metadata_path="/other.json"
                )
            self.assertEqual(cfg.path, f"{td}/abc123")
            self.assertEqual(cfg.plus_metadata_path, f"{td}/abc123.json")
            self.assertEqual(kept.plus_metadata_path, "/other.json")
            # nothing downloaded: a missing file is refused, Secondlook fails open
            write_yolo_onnx(cfg.path)
            Path(cfg.plus_metadata_path).write_text(json.dumps(plus_info()))
            model = SecondlookModel(
                model_path=cfg.path,
                labelmap_path=cfg.labelmap_path,
                device="CPU",
                input_size=cfg.input_size,
                plus_metadata_path=cfg.plus_metadata_path,
                required_labels=("person", "car"),
            )
            self.assertIsNone(model.refusal)
            self.assertEqual(model.source, "frigate_plus")
            Path(cfg.path).unlink()
            missing = SecondlookModel(
                model_path=cfg.path,
                labelmap_path=cfg.labelmap_path,
                device="CPU",
                input_size=cfg.input_size,
                plus_metadata_path=cfg.plus_metadata_path,
                required_labels=("person", "car"),
            )
            self.assertIn("model file missing", missing.refusal)
            self.assertFalse(Path(cfg.path).exists())

    def test_bad_plus_ids_rejected(self):
        from pydantic import ValidationError

        from frigate.config.classification import SecondlookModelConfig

        for bad in ("plus://", "plus://a/b", "plus://.."):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                SecondlookModelConfig(path=bad)


class TestPlusCompatibilityRules(unittest.TestCase):
    def test_valid_record_has_no_problems(self):
        self.assertEqual(plus_compatibility_problems(plus_info()), [])
        self.assertEqual(
            plus_compatibility_problems(plus_info(supportedDetectors=None)), []
        )

    def test_not_an_object(self):
        self.assertTrue(plus_compatibility_problems(["x"]))


class TestRefusedModelFailsOpen(unittest.TestCase):
    def test_refused_model_never_applies_no_hit_suppression(self):
        import threading
        import time
        from collections import deque

        from frigate.data_processing.real_time.secondlook import SecondlookProcessor
        from frigate.test.test_save_and_review import make_config

        with tempfile.TemporaryDirectory() as td:
            info = Path(td) / "m.json"
            info.write_text(json.dumps(plus_info(type="yolonas")))
            model = SecondlookModel(
                str(Path(td) / "m"), "", "CPU", plus_metadata_path=str(info)
            )
            self.assertTrue(model.mock)
            proc = SecondlookProcessor.__new__(SecondlookProcessor)
            proc.sl_config = make_config(True, td).secondlook
            self.assertEqual(proc.sl_config.no_hit_factor, 0.0)
            proc._tracks = {}
            proc.model = model
            proc.model_id = "m"
            proc._samples = proc._results_dropped = 0
            proc._camera_counts = {}
            proc.metrics = SimpleNamespace(secondlook={})
            proc._pending_results = deque(maxlen=4)
            proc._results_lock = threading.Lock()
            proc._queue_items = 1
            proc._queue_rises = {}
            proc._items_lock = threading.Lock()
            proc._process_task(
                {
                    "kind": "infer",
                    "object_id": "x",
                    "camera": "cam",
                    "label": "person",
                    "primary_score": 0.8,
                    "frame_time": 1,
                    "crop": np.zeros((10, 10, 3), np.uint8),
                    "enqueued_at": time.perf_counter(),
                }
            )
            result = proc._pending_results[-1]
            self.assertIsNone(result["factor"])
            self.assertEqual(result["applied_by"], "primary_fallback")
            self.assertEqual(proc._samples, 0)


if __name__ == "__main__":
    unittest.main()
