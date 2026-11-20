# Secondlook: tests and validation

How to check this fork. Everything below runs without cameras or a running
Frigate unless it says otherwise.

## Backend unit tests

The tests use Python's `unittest`, like the rest of Frigate. The simplest way
to get Frigate's exact Python dependencies is the official 0.18.2 image, with
this checkout's `frigate/` package mounted over the stock one. Frigate
generates `frigate/version.py` at build time, so create one first (it is not
tracked in git):

```bash
echo 'VERSION = "0.18.2-secondlook-dev"' > frigate/version.py

docker run --rm --network none --entrypoint python3 -w /opt/frigate \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD/frigate:/opt/frigate/frigate" \
  ghcr.io/blakeblackshear/frigate:0.18.2 -m unittest \
  frigate.test.test_config \
  frigate.test.test_save_and_review \
  frigate.test.test_yolo_decode_config \
  frigate.test.test_secondlook \
  frigate.test.test_secondlook_accuracy_queue \
  frigate.test.test_secondlook_analyses \
  frigate.test.test_secondlook_clusters \
  frigate.test.test_secondlook_hourly \
  frigate.test.test_secondlook_outcomes \
  frigate.test.test_secondlook_plus_model \
  frigate.test.test_secondlook_release_ipc \
  frigate.test.test_secondlook_retention \
  frigate.test.test_secondlook_review_0182 \
  frigate.test.test_secondlook_revision
```

The same works inside a built Secondlook image (it keeps `frigate/test`), or
in a local Python 3.11 virtualenv with `docker/main/requirements*.txt`
installed. Lines logged at ERROR level during the run (for example "Secondlook
model refused ...") are expected: the tests exercise those failure paths.

The whole Frigate suite runs with
`python3 -m unittest discover -s frigate/test -t .` in the same container.
Some stock tests fail in this setup without network access or a running
go2rtc (go2rtc stream checks, sqlite-vec, the latest-frame fallback image);
they fail the same way on unmodified v0.18.2.

What the Secondlook tests cover, in short:

- the live decision: hold while the first answer is pending, factor scaling of
  the score buffer before detection, fail-open on timeout, full queue, refused
  model or inference errors;
- model loading and the compatibility check for COCO and Frigate+ YOLOv9 ONNX
  models (synthetic ONNX fixtures), and the configurable YOLO decode;
- saving at track end, the separate SQLite store and its migrations, verdicts
  and the old accept/reject mapping, filters, paging, grouping, mask
  suggestions (checked by loading the YAML with `FrigateConfig` and running
  `is_object_filtered`), per-object analyses, hourly counts across DST, and
  retention, including that stock recording, event and review cleanup does
  not touch Secondlook data;
- stats and metrics export.

## Web UI

From `web/`:

```bash
npm install
npx tsc --noEmit
npm run lint          # ESLint plus the e2e spec lint
npm run build         # production build (base path /BASE_PATH/)
```

The Playwright tests run against a build served by `vite preview`, with the
API and websocket mocked (no Frigate backend):

```bash
npx playwright install chromium
npm run e2e:build     # build with base path /
npx playwright test --config e2e/playwright.config.ts \
  save-and-review secondlook-debug-metrics secondlook-groups \
  secondlook-lightbox secondlook-review-metrics
```

Each spec runs on a desktop and a mobile project. Some tests save
screenshots to `SECONDLOOK_SCREENS` (default
`web/test-results/secondlook-screens`). `secondlook-groups` uses a real camera
frame from `SECONDLOOK_FRAME` if set, else a generated placeholder.

## Image build and preflight

Build the image as in [`README.md`](README.md), then run
`/opt/frigate/secondlook_preflight.py` in it. Without `--model` it checks
the imports, SQLite JSON support and the production web layout; with
`--model` it also loads the model with the same loader and compatibility
check Frigate uses, on the chosen OpenVINO device.

## Pipeline harness (optional)

`tools/secondlook-smoke/run_sl.py` drives the real Frigate pieces (frame
manager, primary OpenVINO YOLO on regions, Norfair tracker, camera state,
the Secondlook processor and the save step) on a looped video clip in one
process, with real-time pacing. It needs OpenVINO, ffmpeg and Frigate's Python
dependencies (for example inside the Frigate image), plus a directory set by
`SECONDLOOK_ASSETS` containing:

- `models/yolov9-cpu-640.onnx`: the primary model (any YOLO-generic ONNX
  640x640);
- `models/yolov9-m-640.onnx`: the Secondlook model (or set `SL_MODEL`);
- `secondlook-assets/coco80_labelmap.txt`: copy from `deploy/secondlook/`.

```bash
export SECONDLOOK_ASSETS=/path/to/assets
export SECONDLOOK_SMOKE_OUTPUT=$(mktemp -d)   # fresh directory per run
cd tools/secondlook-smoke
python3 run_sl.py --clip /path/to/clip.mp4 --seconds 16 \
  --inject-fp 'person:0.75:350,212,462,519'
```

Useful variations:

- false positive: `--inject-fp label:score:x1,y1,x2,y2` adds a constant fake
  primary detection; Secondlook should keep it from becoming a detection and
  save it as a false positive after the track ends;
- false negative: `--primary-scale .75` weakens the primary; Secondlook
  should let the real object become a detection;
- `--sl off` and `--save-and-review off` for baselines;
- timeout: `--sl-delay-ms 1500 --hold .2 --queue-max 1`;
- Frigate+ model: `--sl-model <file> --sl-plus-metadata <file>.json`.

The harness runs the pipeline in one process. It does not test the separate
capture and detector processes, nginx, or Coral and GPU hardware.

## Live checks

On a real installation, after switching to the image (README step 4): the
Secondlook model loads on the intended device, System → Cameras shows
analysed/s, a known false positive no longer becomes a detection while a real
object still does, items appear in Review → Corrections after their tracked
objects end, and the hourly charts under System → General fill in.

## Known limits

- Corrections grouping uses camera, kind, classes, time and box position and
  size, not appearance, so two different objects can share a group. Each item
  keeps its own verdict.
- The live accuracy counters in `/api/stats` are a snapshot and can lag the
  database counts by one stats interval.
