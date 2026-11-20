"""Build a FrigateConfig for the Secondlook harness."""
from __future__ import annotations
import copy
import os
from frigate.config import FrigateConfig

# Directory with models/ and secondlook-assets/ (see deploy/secondlook/VALIDATION.md).
TERN = os.environ.get("SECONDLOOK_ASSETS", os.path.join(os.path.dirname(__file__), "assets"))
MODEL = f"{TERN}/models/yolov9-cpu-640.onnx"  # primary (t-class)
SL_MODEL = os.environ.get("SL_MODEL", f"{TERN}/models/yolov9-m-640.onnx")  # Secondlook
LABELS = f"{TERN}/secondlook-assets/coco80_labelmap.txt"


def build_config(*, sl_enabled: bool, cam: str, width=1280, height=720, fps=5,
                 sl_overrides: dict | None = None, objects=("person", "car")) -> FrigateConfig:
    sl = {
        "enabled": sl_enabled,
        "device": "CPU",
        "model": {"path": SL_MODEL, "labelmap_path": LABELS, "input_size": 640},
        "objects": list(objects),
    }
    if sl_overrides:
        sl.update(sl_overrides)
    raw = {
        "mqtt": {"enabled": False},
        "detectors": {"cpu": {"type": "openvino", "device": "CPU"}},
        "model": {
            "path": MODEL, "labelmap_path": LABELS, "width": 640, "height": 640,
            "model_type": "yolo-generic", "input_tensor": "nchw",
            "input_pixel_format": "rgb", "input_dtype": "float",
        },
        "objects": {"track": list(objects)},
        "secondlook": sl,
        "cameras": {
            cam: {
                "ffmpeg": {"inputs": [{"path": "/dev/null", "roles": ["detect"]}]},
                "detect": {"width": width, "height": height, "fps": fps},
                "record": {"enabled": False},
                "snapshots": {"enabled": False},
                "secondlook": {"enabled": sl_enabled},
            }
        },
    }
    return FrigateConfig.parse_object(copy.deepcopy(raw))
