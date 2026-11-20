"""Read-only import/model preflight; never starts Frigate, accesses cameras or writes DBs."""

import argparse
import hashlib
import importlib
import sqlite3
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Optional ONNX path to load with OpenVINO on --device")
parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
parser.add_argument("--detector", default=None,
                    help="Optional custom detector plugin to import-check, e.g. a bind-mounted "
                         "/opt/frigate/frigate/detectors/plugins/edgetpu_tfl.py")
parser.add_argument("--labelmap", default="/models/secondlook/coco80_labelmap.txt",
                    help="COCO labelmap for --model (ignored with --plus-metadata)")
parser.add_argument("--plus-metadata", help="Frigate+ model info JSON for a Frigate+ YOLOv9 ONNX --model")
parser.add_argument("--objects", default="person,car,dog,cat",
                    help="Comma-separated Secondlook objects the model must label")
args = parser.parse_args()

from frigate.version import VERSION
if not VERSION.startswith("0.18.2-secondlook-"):
    raise SystemExit(f"Unexpected image version: {VERSION}")
for name in (
    "frigate.config", "frigate.secondlook.save_and_review",
    "frigate.secondlook.save_and_review_deposit",
    "frigate.data_processing.real_time.secondlook",
    "frigate.embeddings.maintainer", "frigate.track.object_processing",
    "frigate.api.fastapi_app",
):
    importlib.import_module(name)
    print(f"IMPORT OK {name}")

with sqlite3.connect(":memory:") as db:
    assert db.execute("SELECT value FROM json_each('[1]')").fetchone() == (1,)
print("SQLITE JSON FILTER SUPPORT OK")

web = Path("/opt/frigate/web")
if "/BASE_PATH/" not in (web / "index.html").read_text():
    raise SystemExit("Production web BASE_PATH convention is missing")
for filename in ("editor.worker.bundle.js", "yaml.worker.bundle.js"):
    if not (web / "assets" / filename).is_file():
        raise SystemExit(f"Relocated Monaco worker missing: {filename}")
if (web / "BASE_PATH").exists():
    raise SystemExit("Monaco relocation was not completed")
print("PRODUCTION WEB LAYOUT OK (not an nginx/browser smoke)")

# Optional: import a custom (for example bind-mounted) detector plugin, so a
# plugin that does not load with this image is found before switching over.
if args.detector:
    plugin = Path(args.detector)
    if not plugin.is_file():
        raise SystemExit(f"Custom detector plugin is missing: {plugin}")
    importlib.import_module(f"frigate.detectors.plugins.{plugin.stem}")
    print("CUSTOM DETECTOR IMPORT OK sha256=" + hashlib.sha256(plugin.read_bytes()).hexdigest())

import openvino as ov
core = ov.Core()
print(f"VERSION {VERSION}")
print(f"OPENVINO {ov.__version__} devices={core.available_devices}")
if args.model:
    if not Path(args.model).is_file():
        raise SystemExit("Selected Secondlook model is missing")
    from frigate.secondlook.model import SecondlookModel

    # Same loader and startup compatibility check Frigate runs.
    model = SecondlookModel(
        args.model,
        args.labelmap,
        device=args.device,
        plus_metadata_path=args.plus_metadata,
        required_labels=[o for o in args.objects.split(",") if o],
    )
    if model.refusal:
        raise SystemExit(f"Secondlook would refuse this model (fail-open): {model.refusal}")
    if model.mock:
        raise SystemExit("Secondlook model did not load (would run as mock, no inference)")
    if model.device != args.device:
        raise SystemExit(f"Model fell back to {model.device}, not {args.device}")
    print(
        f"MODEL COMPATIBLE source={model.source} input={model.input_size} "
        f"layout={model.input_tensor} pixels={model.pixel_format} device={model.device}"
    )
print("Preflight passed (imports/model only; no camera inference or live-service test)")
