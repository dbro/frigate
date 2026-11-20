# Secondlook

Secondlook is an addition to Frigate 0.18.2. A second object detection model
checks tracked objects before they become a detection. It runs on a crop
around each tracked object and turns its answer into a factor that scales the
primary detector's scores in the score buffer before detection. A clear
second answer can stop a false positive from becoming a detection, or let a
weak but real object become one. When Secondlook has no answer in time, the
tracked object is handled exactly as stock Frigate would handle it
(fail-open).

It is off by default. With `secondlook.enabled: false`, Frigate behaves as
stock 0.18.2.

## How it works

1. When a tracked object of a configured label is first seen, Secondlook
   crops the detect-stream frame around its box using Frigate's
   `calculate_region` (scaled by `context_scale`, never smaller than the
   model input), and runs the second model on it with OpenVINO.
2. If the second model finds a box of the same class that overlaps the
   tracked object's box (`min_box_iou`) with a score of at least
   `score_threshold`, the factor is `secondlook score / primary score`,
   clamped to `factor_min`..`factor_max`. If it finds none, the factor is
   `no_hit_factor` (default 0.0, so the object does not become a detection
   until a later answer finds it).
3. Frigate's usual rule still decides: the median of the last scores in the
   score buffer before detection must reach the camera's threshold. With
   Secondlook, each score in that buffer is multiplied by the factor first.
4. While the first answer is pending, the tracked object waits at most
   `hold_seconds`. After that, or if the queue is full, the model is missing,
   or inference fails, the primary scores are used unscaled.
5. Every `refresh_every_n` real hits, Secondlook looks again at the crop with
   the highest primary score and updates the factor.

Secondlook does not train or change any model and does not change Frigate's
database schema. Its own data goes into a separate SQLite file and JPEG
directory.

## Corrections Review

When `save_and_review` is on, every tracked object where Secondlook changed
the outcome, or disagreed on the class, is saved at track end as a full-frame
JPEG with both boxes, plus a row in the separate database:

- **False positive**: the primary alone would have made it a detection;
  Secondlook did not agree.
- **False negative**: the primary alone stayed under the threshold;
  Secondlook found the object.
- **Misclass**: Secondlook found a different class in the same place and no
  same-class partner.

The **Corrections** tab in Review lists them with filters (kind, camera,
class, zone, reviewed, suppressed or rescued), short-term and long-term
grouping, a full-size view, and verdicts: **Primary right**, **Secondlook
right** and **Ignore**. A long-term false-positive group can suggest an object
filter mask as YAML to paste into the config (nothing is written for you).
A daily retention job deletes old unreviewed and ignored items; items marked
Primary right or Secondlook right are kept.

Every object Secondlook was asked about also gets one small row (no image) in
an `analyses` table, and hourly counts are kept for the charts.

Details: `docs/docs/configuration/secondlook-save-and-review.md`.

## Metrics

- **System → General**: Secondlook disagreements per hour for the last 7 days
  (false positives suppressed or not, false negatives rescued or not,
  misclass; counts or rates), Secondlook answers per hour (agreed, no hit,
  timed out, dropped, failed open), processing time and queue charts, and the
  model status (including why a model was refused).
- **System → Cameras**: analysed/s and skipped/s per camera.
- **System → Storage**: disk used by the Secondlook database and images.
- **Live → Debug**: an inset with the latest analysed crop and its boxes.
- `/api/stats` has a `secondlook` section; the hourly and per-object data are
  at `GET /api/secondlook/disagreements/hourly` and
  `GET /api/secondlook/analyses/summary`.

## Config

Global (`secondlook:`), with defaults:

```yaml
secondlook:
  enabled: false            # cameras must enable it too
  device: GPU               # OpenVINO device; CPU also works
  model:
    path: /config/model_cache/secondlook/yolov9-c-640.onnx
    labelmap_path: /config/model_cache/secondlook/labelmap.txt
    input_size: 640
    plus_metadata_path: null  # Frigate+ YOLOv9 ONNX: path to its model info JSON
    # path: plus://<model_id> is a short form for a Frigate+ model already in
    # /config/model_cache (Secondlook never downloads)
    # YOLO decode (also available under openvino/onnx detectors):
    # min_score: 0.4, nms_threshold: 0.4, max_detections: 20, nms_class_groups: unset
  objects: [person, car, dog, cat]
  hold_seconds: 1.0         # max wait for the first answer
  refresh_every_n: 10
  context_scale: 1.5        # crop size around the box
  score_threshold: 0.25     # min Secondlook score for a hit
  min_box_iou: 0.3          # null: any same-class box in the crop counts
  no_hit_factor: 0.0        # null: no factor when nothing is found
  factor_min: 0.3
  factor_max: 2.0
  queue_max: 8              # waiting inference jobs; overflow fails open
  gap_factor_low: 0.8       # save as false positive below this factor
  gap_factor_high: 1.2      # save as false negative above this factor
  paired_filters:           # applied only to Secondlook cameras and labels
    min_score: 0.3          # overrides objects.filters.<label>.min_score
    min_initialized: null   # optional detect.min_initialized override
  save_and_review:
    enabled: true
    database:
      path: /config/secondlook_examples.db
    images:
      path: /media/frigate/clips/secondlook_examples
    review:
      group_gap_seconds: 60
      hide_edge_small: true
      edge_margin_px: 4
      small_dim_fraction: 0.05
    retention:              # 0 or null disables a rule
      unreviewed_days: 30
      ignored_days: 7
      max_size_gb: null
      summary_days: 90
      analyses_days: 90
```

Per camera (`cameras.<name>.secondlook:`): `enabled` (default false),
`objects` (null inherits the global list), `hold_seconds`, `factor_min`,
`factor_max`.

The model must be a YOLO (`yolo-generic`) ONNX model whose labels include
every Secondlook object. A model that does not fit is refused at startup with
the reason logged and shown under System → General, and Frigate runs as if
Secondlook were off.

## Build and run

See `deploy/secondlook/README.md`. Tests and validation are described in
`deploy/secondlook/VALIDATION.md`.
