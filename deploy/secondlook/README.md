# Secondlook on Frigate 0.18.2: build and run

This directory builds a Docker image of Frigate 0.18.2 with Secondlook. For
what Secondlook does and all of its config keys, see
[`docs/SECONDLOOK.md`](../../docs/SECONDLOOK.md). For tests and validation, see
[`VALIDATION.md`](VALIDATION.md).

Files here:

| File | Purpose |
| --- | --- |
| `Dockerfile` | the official `ghcr.io/blakeblackshear/frigate:0.18.2` image (pinned by digest) with this fork's `frigate/` Python package and a fresh web UI build |
| `preflight.py` | read-only check run inside the new image before switching over: imports, web UI layout, and optionally that a given Secondlook model loads and would be accepted |
| `secondlook.yaml` | example config to merge into an existing Frigate config |
| `coco80_labelmap.txt` | COCO-80 labelmap for a stock COCO YOLOv9 ONNX model |

No model files are included.

## 1. Build the image

From the repository root (needs network access to pull `node:20` and the
Frigate base image, and to run `npm install`):

```bash
docker build -f deploy/secondlook/Dockerfile \
  --build-arg SECONDLOOK_REVISION=$(git rev-parse --short HEAD) \
  -t frigate-secondlook:0.18.2-$(git rev-parse --short HEAD) .
```

Only `frigate/` and the web UI are replaced. Native libraries, s6, nginx,
FFmpeg and the Python dependencies are the stock 0.18.2 ones, so the image runs
on the same hardware and with the same detectors as stock 0.18.2. The
version reads `0.18.2-secondlook-<revision>`. The image is built for the
platform you build on; the base digest is a multi-platform manifest.

## 2. Get a Secondlook model

Secondlook needs a YOLO (`yolo-generic`) ONNX model whose labels include every
object you enable it for. Either:

- **COCO YOLOv9**: export a YOLOv9 model (for example yolov9-c or yolov9-m at
  640x640) to ONNX the same way as for Frigate's ONNX/OpenVINO detectors (see
  Frigate's object detector docs), and put it with `coco80_labelmap.txt` in a
  directory mounted into the container, for example
  `/config/model_cache/secondlook/`.
- **Frigate+ YOLOv9 ONNX**: a Frigate+ YOLOv9 model already in
  `/config/model_cache` (for example because the primary detector uses it).
  Use `path: plus://<model_id>`; labels and input size come from its model
  info file. Secondlook never downloads models.

## 3. Check the image before switching

Run the preflight in the new image with your model directory mounted
read-only. It does not start Frigate, touch cameras or write any database:

```bash
docker run --rm --network none --read-only --tmpfs /tmp \
  -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 \
  -v /path/to/config/model_cache:/config/model_cache:ro \
  frigate-secondlook:0.18.2-<revision> /opt/frigate/secondlook_preflight.py \
  --model /config/model_cache/secondlook/yolov9-c-640.onnx \
  --labelmap /config/model_cache/secondlook/coco80_labelmap.txt \
  --objects person,car --device CPU
```

- For a Frigate+ model use `--model /config/model_cache/<model_id>` and
  `--plus-metadata /config/model_cache/<model_id>.json` instead of
  `--labelmap`.
- For an Intel iGPU add `--device /dev/dri:/dev/dri` before the image name
  and `--device GPU` after the script.
- If you bind-mount a custom detector plugin, mount it the same way and pass
  `--detector /opt/frigate/frigate/detectors/plugins/<plugin>.py` to check
  that it still imports.
- `--objects` must list every object your Secondlook cameras use. A "would
  refuse this model" line gives the reason; Frigate would log the same reason
  and run without Secondlook adjustment.

## 4. Configure and run

Merge `secondlook.yaml` into your existing config: keep your `detectors:`,
`model:` and cameras, and enable Secondlook globally and then on the cameras
you want. Global settings inherit into cameras, so to try it on one camera
first, set `secondlook.enabled: false` under the others. Then change only the
`image:` line of your Compose service (or `docker run` command) to the new
tag, keep all existing volumes, devices and ports, and recreate the
container:

```bash
docker compose up -d --no-deps --no-build --pull never frigate
docker logs --since 2m frigate
```

Do not bind-mount the whole `/opt/frigate/frigate` or `/opt/frigate/web`
directories: they would hide the image's files. Hard-refresh the browser
afterwards.

What to check:

- The log says the Secondlook model loaded on the intended device (not a
  refusal or a fallback to CPU).
- System → Cameras shows Secondlook analysed/s for the Secondlook cameras.
- A known false positive no longer becomes a detection; a real object still
  does.
- After a tracked object ends, items where Secondlook changed the outcome
  appear under Review → Corrections.

Switches:

- `secondlook.enabled: false`: no live score adjustment; Frigate behaves as
  stock 0.18.2. Corrections stays available for saved data.
- `secondlook.save_and_review.enabled: false`: nothing is saved and
  Corrections is hidden; live adjustment continues.

Restart Frigate after changing either.

## 5. Data and backups

Secondlook does not change Frigate's own database schema. It keeps its data in
a separate SQLite file (`secondlook.save_and_review.database.path`, default
`/config/secondlook_examples.db`) and a JPEG directory
(`images.path`, default `/media/frigate/clips/secondlook_examples`). Stock
recording, event and review cleanup does not delete them; the Secondlook
retention settings do (see `docs/SECONDLOOK.md`). Back up both together while
Frigate is stopped.

## 6. Rollback

Stop the container, set `image:` back to the previous tag, restore the config
backup if you changed it, and start it again. Stock Frigate ignores the
separate Secondlook database and images; keep them if you may switch back.
A stock image does not accept the `secondlook:` config section, so remove it
from the config when going back to stock Frigate.
