"""OpenVINO runner for the Secondlook YOLO model.

Two model sources:
- COCO YOLOv9 ONNX (default): labels from labelmap_path, square input_size.
- Frigate+ YOLOv9 ONNX from a local file: labels, input size, tensor layout,
  pixel format and dtype from the Frigate+ model info JSON (plus_metadata_path).

A startup compatibility check refuses an unsuitable model. A refused model runs
as the mock runner: no inference, factors stay unset, primary scores pass
unscaled (fail-open). It never suppresses detections.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Iterable

import cv2
import numpy as np

from frigate.util.builtin import load_labels
from frigate.util.model import build_nms_group_lookup, post_process_yolo

logger = logging.getLogger(__name__)

PLUS_SUPPORTED_TYPES = ("yolo-generic",)
PLUS_SUPPORTED_DETECTORS = ("onnx", "openvino")


def plus_compatibility_problems(info: Any) -> list[str]:
    """Reasons a Frigate+ model info record is unsuitable for Secondlook."""
    if not isinstance(info, dict):
        return ["Frigate+ model info is not a JSON object"]
    problems = []
    required = ("type", "width", "height", "inputShape", "pixelFormat", "labelMap")
    missing = [key for key in required if key not in info]
    if missing:
        return ["Frigate+ model info missing " + ", ".join(missing)]
    if info["type"] not in PLUS_SUPPORTED_TYPES:
        problems.append(
            f"model type {info['type']!r} unsupported (only YOLOv9 'yolo-generic'; "
            "YOLO-NAS and MobileDet are not supported)"
        )
    detectors = info.get("supportedDetectors")
    if detectors is not None and not set(detectors) & set(PLUS_SUPPORTED_DETECTORS):
        problems.append(
            f"model built for {detectors}, not an ONNX/OpenVINO variant "
            "(Coral/EdgeTPU, Hailo and RKNN variants are not supported)"
        )
    if info["width"] != info["height"] or not isinstance(info["width"], int):
        problems.append(
            f"input {info['width']}x{info['height']} must be square (Secondlook crops are square)"
        )
    if info["inputShape"] not in ("nchw", "nhwc"):
        problems.append(f"input layout {info['inputShape']!r} unsupported")
    if info["pixelFormat"] not in ("rgb", "bgr"):
        problems.append(f"pixel format {info['pixelFormat']!r} unsupported")
    if info.get("inputDataType") not in (None, "float", "float_denorm"):
        problems.append(f"input data type {info['inputDataType']!r} unsupported")
    label_map = info["labelMap"]
    if not isinstance(label_map, dict) or not label_map:
        problems.append("labelMap empty")
    else:
        try:
            [int(k) for k in label_map]
        except (TypeError, ValueError):
            problems.append("labelMap keys must be class indexes")
    return problems


class SecondlookModel:
    """Load and run a YOLOv9 ONNX via OpenVINO; mock-friendly when unusable."""

    def __init__(
        self,
        model_path: str,
        labelmap_path: str,
        device: str = "GPU",
        input_size: int = 640,
        plus_metadata_path: str | None = None,
        required_labels: Iterable[str] = (),
        min_score: float = 0.4,
        nms_threshold: float = 0.4,
        max_detections: int = 20,
        nms_class_groups: list | None = None,
    ) -> None:
        self.model_path = model_path
        self.labelmap_path = labelmap_path
        self.plus_metadata_path = plus_metadata_path
        self.source = "frigate_plus" if plus_metadata_path else "coco"
        self.device = device
        self.input_size = input_size
        self.input_tensor = "nchw"
        self.pixel_format = "rgb"
        self.input_dtype = "float"
        self.required_labels = sorted(set(required_labels))
        # YOLO post-processing; Secondlook doesn't use Frigate's 20-row
        # detector buffer, so max_detections may exceed 20 (config caps 300).
        self.min_score = float(min_score)
        self.nms_threshold = float(nms_threshold)
        self.max_detections = int(max_detections)
        self.nms_class_groups = nms_class_groups
        self.class_groups: np.ndarray | None = None
        self.available = False
        self.mock = False
        self.refusal: str | None = None
        self.labelmap: dict[int, str] = {}
        self._compiled = None
        self._req = None
        self._input_name = "images"
        self._load()

    def _refuse(self, reason: str) -> None:
        self.refusal = reason
        self.mock = True
        self.available = True
        self._compiled = self._req = None
        logger.error(
            "Secondlook model refused (%s): %s. Secondlook will not adjust "
            "scores; primary detections pass unscaled (fail-open).",
            self.model_path,
            reason,
        )

    def _load_coco_labelmap(self) -> None:
        if not os.path.isfile(self.labelmap_path):
            logger.warning(
                "Secondlook labelmap missing at %s — class names unavailable",
                self.labelmap_path,
            )
            return
        try:
            # indexed=None auto-detects "0 person" vs one-label-per-line.
            # indexed=False would keep the index in the name ("0 person")
            # so best_same_class() never matched and factor stayed unset.
            self.labelmap = load_labels(self.labelmap_path, prefill=0)
        except Exception:
            # also accept plain one-label-per-line coco80
            self.labelmap = {}
            with open(self.labelmap_path) as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2 and parts[0].isdigit():
                        self.labelmap[int(parts[0])] = parts[1]
                    else:
                        self.labelmap[i] = parts[-1]

    def _load_plus_metadata(self) -> str | None:
        """Apply Frigate+ model info; return a refusal reason or None."""
        try:
            with open(self.plus_metadata_path) as f:
                info = json.load(f)
        except (OSError, ValueError) as e:
            return f"Frigate+ model info unreadable at {self.plus_metadata_path}: {e}"
        problems = plus_compatibility_problems(info)
        if problems:
            return "; ".join(problems)
        self.labelmap = {int(k): str(v) for k, v in info["labelMap"].items()}
        self.input_size = int(info["width"])
        self.input_tensor = info["inputShape"]
        self.pixel_format = info["pixelFormat"]
        self.input_dtype = info.get("inputDataType") or "float"
        return None

    def _check_io(self, compiled) -> str | None:
        shape = compiled.inputs[0].get_partial_shape()
        if shape.is_static:
            dims = [d.get_length() for d in shape]
            size = self.input_size
            expected = (
                [1, 3, size, size] if self.input_tensor == "nchw" else [1, size, size, 3]
            )
            if dims != expected:
                return f"model input shape {dims} does not match expected {expected}"
        if self.source != "frigate_plus":
            return None
        if len(compiled.outputs) != 1:
            return (
                f"model has {len(compiled.outputs)} outputs; Secondlook supports the "
                "single-output YOLOv9 ONNX head"
            )
        out = compiled.outputs[0].get_partial_shape()
        if out.is_static:
            dims = [d.get_length() for d in out]
            channels = 4 + max(self.labelmap) + 1
            if len(dims) != 3 or channels not in dims[1:]:
                return (
                    f"model output shape {dims} does not carry {channels} channels "
                    f"(4 box + {channels - 4} classes from labelMap)"
                )
        return None

    def _load(self) -> None:
        if self.plus_metadata_path:
            reason = self._load_plus_metadata()
            if reason:
                self._refuse(reason)
                return
        else:
            self._load_coco_labelmap()

        if not os.path.isfile(self.model_path):
            if self.source == "frigate_plus":
                self._refuse(f"Frigate+ model file missing at {self.model_path}")
                return
            logger.warning(
                "Secondlook model missing at %s — using mock runner "
                "(no real inference; factors stay unset)",
                self.model_path,
            )
            self.mock = True
            self.available = True
            return

        labels = set(self.labelmap.values())
        missing = [label for label in self.required_labels if label not in labels]
        if missing:
            # Without the label every SL result is a miss and no_hit_factor
            # would suppress real detections, so refuse instead.
            self._refuse(
                "model labels do not include configured Secondlook objects: "
                + ", ".join(missing)
            )
            return

        try:
            import openvino as ov
        except ImportError:
            if self.source == "frigate_plus":
                self._refuse("OpenVINO not installed")
                return
            logger.warning(
                "OpenVINO not installed — Secondlook model running in mock mode"
            )
            self.mock = True
            self.available = True
            return

        try:
            core = ov.Core()
            devices = list(core.available_devices)
            device = self.device
            if device not in devices and device.upper() not in [
                d.upper() for d in devices
            ]:
                if "CPU" in devices:
                    logger.warning(
                        "Secondlook device %s unavailable (%s); falling back to CPU",
                        device,
                        devices,
                    )
                    device = "CPU"
                else:
                    raise RuntimeError(f"No OpenVINO devices: {devices}")

            model = core.read_model(self.model_path)
            if device in ("GPU", "AUTO", "NPU"):
                try:
                    core.set_property(device, {"PERFORMANCE_HINT": "LATENCY"})
                except Exception:
                    pass
            compiled = core.compile_model(model, device)
            reason = self._check_io(compiled)
            if reason:
                self._refuse(reason)
                return
            self._compiled = compiled
            self._req = self._compiled.create_infer_request()
            self._input_name = self._compiled.inputs[0].get_any_name()
            self.class_groups = build_nms_group_lookup(
                self.nms_class_groups, self.labelmap
            )
            self.device = device
            self.available = True
            self.mock = False
            logger.info(
                "Secondlook OpenVINO model loaded: %s (%s, %dx%d %s %s) on %s",
                self.model_path,
                self.source,
                self.input_size,
                self.input_size,
                self.input_tensor,
                self.pixel_format,
                device,
            )
        except Exception as e:
            if self.source == "frigate_plus":
                self._refuse(f"OpenVINO could not load the model: {e}")
                return
            logger.exception(
                "Failed to load Secondlook model; falling back to mock mode"
            )
            self.mock = True
            self.available = True

    def preprocess(self, bgr_or_rgb: np.ndarray, is_rgb: bool = True) -> np.ndarray:
        """Square resize to the model input in its pixel format, layout and dtype."""
        img = bgr_or_rgb
        if is_rgb != (self.pixel_format == "rgb"):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        size = self.input_size
        resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32)
        if self.input_dtype == "float":
            tensor /= 255.0
        if self.input_tensor == "nchw":
            tensor = np.transpose(tensor, (2, 0, 1))
        return np.ascontiguousarray(tensor[None, ...])

    def infer(self, rgb_crop: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        """Run detection on an RGB crop.

        Returns (detections, latency_ms). Each detection:
          {label, score, box_norm} where box_norm is [ymin,xmin,ymax,xmax] in 0..1
          relative to the crop (post_process_yolo convention).
        """
        # True when this call had no usable inference (read by the processor)
        self.last_failed = not self.available or self.mock
        if not self.available:
            return [], 0.0

        if self.mock:
            # No weights: pretend no same-class hit so factor stays unset.
            return [], 0.0

        tensor = self.preprocess(rgb_crop, is_rgb=True)
        t0 = time.perf_counter()
        self._req.infer({self._input_name: tensor})
        outputs = [
            self._req.get_output_tensor(i).data.copy()
            for i in range(len(self._compiled.outputs))
        ]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        try:
            dets = post_process_yolo(
                outputs,
                self.input_size,
                self.input_size,
                min_score=self.min_score,
                nms_threshold=self.nms_threshold,
                max_detections=self.max_detections,
                class_groups=self.class_groups,
            )
        except Exception:
            logger.exception("Secondlook post_process_yolo failed")
            self.last_failed = True
            return [], latency_ms

        results: list[dict[str, Any]] = []
        for row in dets:
            class_id = int(row[0])
            score = float(row[1])
            if score <= 0:
                continue
            label = self.labelmap.get(class_id, str(class_id))
            results.append(
                {
                    "label": label,
                    "score": score,
                    "box_norm": [
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                        float(row[5]),
                    ],
                    "class_id": class_id,
                }
            )
        return results, latency_ms

    def best_same_class(
        self, detections: list[dict[str, Any]], label: str, min_score: float
    ) -> dict[str, Any] | None:
        best = None
        for d in detections:
            if d["label"] != label:
                continue
            if d["score"] < min_score:
                continue
            if best is None or d["score"] > best["score"]:
                best = d
        return best
