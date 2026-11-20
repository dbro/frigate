---
title: Secondlook save and review
---

Save and review saves data suitable for future fine-tuning and makes it available
for human review in the **Corrections** tab. It does not train/fine-tune models or
perform online corrections.

```yaml
secondlook:
  enabled: true # live score adjustment; cameras must also enable Secondlook
  save_and_review:
    enabled: true # saving AND Corrections UI/API; defaults to true
    database:
      path: /config/secondlook_examples.db
    images:
      path: /media/frigate/clips/secondlook_examples
    review: # Corrections display only; nothing is deleted
      group_gap_seconds: 60 # see Grouping
      hide_edge_small: true # default state of the small-edge toggle
      edge_margin_px: 4
      small_dim_fraction: 0.05
    retention: # daily cleanup; 0 or null disables a rule
      unreviewed_days: 30
      ignored_days: 7
      max_size_gb: null
      summary_days: 90 # hourly summary rows
      analyses_days: 90 # per-object analyses rows (no images)
```

### Retention

- Runs once at startup, then every 24 h, in a background thread of the main
  Frigate process (where the store lives). No job when `save_and_review` is
  disabled or all five rules are 0/null; nothing is deleted then.
- `unreviewed_days` (default 30): decisions with no verdict, aged from capture
  (frame time, else save time).
- `ignored_days` (default 7): **Ignore** decisions, aged from when they were
  last marked (`marked_at`), not from capture.
- `max_size_gb` (default off, GiB): if database + images exceed it, the oldest
  unreviewed (by capture), then the oldest ignored (by mark) are deleted until
  under it. The SQLite file does not shrink after deletes, so set it with room.
- `summary_days` (default 90): rows of the hourly summary (`hourly_counts`
  table: objects analysed, FP/FN/misclass and the per-object answer counts per
  local hour, camera, label and model pair; also rows of the earlier
  `hourly_summary` table) older than this. Separate from saved decisions.
- `analyses_days` (default 90): rows of the per-object `analyses` table whose
  tracked object ended more than this many days ago. About 0.53 MiB per 1,000
  tracked objects (no images), so about 2.7 MB a day at 5,000 objects a day.
- **Primary right / Secondlook right are never deleted**, not even by the cap.
- Rows are deleted in one transaction, then their own JPEGs; a missing JPEG is
  counted, not an error. JPEGs without a row are swept only when they are
  `<images.path>/<camera>/<uuid>.jpg`, not symlinks, and older than 1 h.
  A crash mid-run is cleaned up by the next run. Every run logs its counts;
  the last run is under `secondlook.storage.retention` in `/api/stats` and on
  System → Storage. There is no export yet: copy images out before they age.
- Migration: none needed for the table. Marked rows without `marked_at` (none
  expected) get `marked_at` = first start, so they get a full period.

Set `secondlook.save_and_review.enabled: false` to stop saving and hide
Corrections. Its API is unavailable while disabled, including image and mark
endpoints. Live Secondlook score adjustment continues independently. Saved
records, images, and global verdicts are preserved, so re-enabling
restores access. Reviewing existing data does not require live Secondlook to be
running. Restart Frigate after changing these settings.

The API prefix is `/secondlook/save_and_review` (list, `/{id}`, `/{id}/still.jpg`,
and `/{id}/mark`).

## Per-object analyses

Every tracked object Secondlook was asked about gets one row in the `analyses`
table when it finishes: at track end, when a late answer for an ended track
arrives, or 15 s after the end without one. Agreements are stored as well as
disagreements, so agreement counts, score and factor distributions and model
comparisons on the same objects can be read back. No images.

`outcome`, the first that applies over all of the object's answers:

| outcome | meaning | factor |
| --- | --- | --- |
| `result` | a real analysis found a same-class box on the crop | applied |
| `no_hit` | real analyses ran; none found a same-class box above `score_threshold` | `no_hit_factor` (default 0.0) or null |
| `fail_open` | answers arrived without a real inference behind them (model refused or unavailable, inference or post-processing raised) | null; `no_hit_factor` not applied |
| `timeout` | a request was accepted but no answer reached the object before it finished (track end + 15 s; also an answer lost from the bounded result queue) | null |
| `dropped` | the first request was refused because the Secondlook queue was full, and no later answer arrived | null |

`live` says whether the factor changed what happened, judged at track end with
the same rules as the hourly chart: `misclass` first (Secondlook saw another
class in the same place), then `false_positive` / `false_negative` with
`live_outcome` = `fp_suppressed`, `fp_not_suppressed`, `fn_rescued` or
`fn_not_rescued`, else `agree` when a factor was applied at some point, else
`none`.

Other columns: `event_id` (set when the object became a detection), start and
end time, number of answers, `factor` (in force at the end) with `factor_min`
and `factor_max`, `applied_by`, `sl_score` (last same-class score),
`primary_score` (peak raw median of the 10-score window the release check
uses), `raw_median` (at the first threshold crossing, else null),
`scaled_median` (peak window with the factor), `threshold`, `box` (full-frame
pixels), `size_bucket` (0–4: <1%, 1–2.5%, 2.5–5%, 5–10%, >10% of the frame),
`misclass_label`, `hold_expired`, `answer_after_end`, `latency_ms`,
`sl_model_md5`, `sl_model_name`, `primary_model_id`.

The write is `INSERT OR IGNORE` on (camera, track id), with the row's hourly
counts in the same transaction only when it was inserted, so retries and
restarts never add a second row or count twice. It runs on the
TrackedObjectProcessor thread like the other Secondlook writes.

Model ids are MD5s: the Secondlook processor hashes its model file once at
startup and sends the MD5 with every answer (`sl_model_md5`, with the file name
as `sl_model_name`; saved Review decisions get both too). `primary_model_id` is
the first detector's `model_hash`, as stored in `Event.model_hash`. The hourly
table `hourly_counts` is keyed by both, so hours after a model swap are kept
apart; rows from before (copied once from `hourly_summary`) have empty ids.

API:

- `GET /api/secondlook/disagreements/hourly?primary_model=&sl_model=`: the 168
  hours (adds `tracks_finished`, `agreements`, `results`, `no_hit`,
  `timeouts`, `dropped`, `fail_open`) and the model pairs in the table. By
  default all models are summed; `""` selects rows with an unknown model.
- `GET /api/secondlook/analyses/summary`: counts by `outcome`, `live` and
  `live_outcome`; factor, Secondlook score, primary score and size histograms;
  the models seen. Filters: `camera` (comma-separated), `label`, `sl_model`,
  `primary_model`, `outcome`, `live`, `after`, `before` (track end, epoch s).

System → General shows **Secondlook answers per hour, last 7 days** below the
disagreements chart: agreed, no hit, timed out, dropped and failed open as rates
of the objects finished that hour (or counts), each tooltip value with its
numerator and denominator.

## Verdicts

Each card has three buttons. **Primary right** and **Secondlook right** record
which model's answer is the training label for that example; **Ignore** excludes
it from training. They do not change live detection. Click the selected verdict
again to clear it, or pick another to edit it later. The Primary button uses the
cyan primary box colour and the Secondlook button the orange Secondlook box colour.
Each button also states that model's answer (for misclass, the two classes; for a
false positive, `person` vs `not person`; for a false negative, `not person (below
threshold)` vs `person`). The API values are `primary_right`, `secondlook_right`,
`ignore`, or `null`. Training and export jobs remain outside this feature.

Earlier `accept`/`reject` marks are mapped once, when the store opens:

| Old mark | New verdict | Why |
| --- | --- | --- |
| `accept` | `secondlook_right` | accept meant "include this saved correction"; every saved correction is Secondlook's answer disagreeing with primary |
| `reject` | `ignore` | reject meant "exclude from training"; it never said primary was right, so it is not mapped to `primary_right` |
| unreviewed | unreviewed | unchanged |

The mapping is lossless: the original value is kept in a new nullable
`legacy_mark` column (shown on the card as "Migrated from earlier ... mark"),
`marked_at` is unchanged, and before the first mapping the database is copied to
`<database path>.pre-verdict.bak` with SQLite's backup API. No rows or images are
removed. To roll back to an image that only knows accept/reject, restore that
backup, or run
`UPDATE examples SET mark = legacy_mark WHERE legacy_mark IS NOT NULL;`
(verdicts made after the upgrade have no accept/reject equivalent; set them back
to NULL or keep the backup).

## Building and running

This fork is based on Frigate v0.18.2. For what Secondlook is and all of its
config keys, see `docs/SECONDLOOK.md`; to build and run the image, see
`deploy/secondlook/README.md`. Both are in the source tree.

## Durable Review and frame provenance

Corrections defaults to all saved history; explicit Review dates remain optional.
Counts use the same camera, date, label, zone, kind and mark scope as the list.
Load more traverses the complete result set, not a 200-record subset.

Saving remains track-end only. A bounded counterfactual raw-primary latch records
whether the primary's median buffer ever crossed threshold. A qualifying SL
suppression or release retains its evidence through missed-frame zeros; the actual
true-positive latch remains irreversible. The selected example is one qualifying
window's maximum-primary source frame, not an unrelated later still/result pair.

New rows carry source dimensions/time, crop region and corresponding primary/SL
boxes in full-frame XYXY pixels. The image and result travel together. Refresh
reports the selected maximum's time, not the last window frame. Historical rows
are retained unchanged and explicitly show overlay unavailable where alignment
cannot be established. Schema additions are nullable and do not move image paths.

Examples are JPEGs on disk plus metadata/marks in separate SQLite. Stock recording,
Event and Review cleanup must not remove them. Local regression exercises those
real cleanup paths and checks the saved row, JPEG and marks afterwards.

The list API also exposes fixed-key, process-lifetime decision/deposit counters.
They reset at initialization/restart and contain no unbounded per-track history.

## Grouping

Corrections has a **Grouping** toggle with three modes. Every row, image and
verdict is kept and marked individually in all of them; grouping never merges,
shares or deletes anything.

**No grouping** shows every saved decision on its own card.

**Short-term** (the default) is a proximity rule over the loaded, newest-first
list: same camera, same kind, same primary class and (for misclass) same
Secondlook class, with consecutive frame times at most
`review.group_gap_seconds` apart (default 60 s; 0 groups only equal times).
No appearance or image similarity is used, so two different objects of the same
class on the same camera within the gap can share a group. A group header shows
the count, time span and how many already have a verdict; collapsed, it shows the
newest card; expanded, every card. Pagination and counts always refer to
individual records, and a run that continues past "Load more" becomes one group
once the next page is loaded. (Earlier builds grouped only byte-identical JPEGs,
which separate decisions never have.)

**Long-term** groups every saved decision that matches the filters, on the
server, by fixed bins:

| Field | Values |
| --- | --- |
| Camera | as saved |
| Kind | false positive, false negative, misclass |
| Outcome | acted on (FP suppressed, FN rescued) or not, from the stored `event_id`; none for misclass |
| Class | primary class; for misclass also Secondlook's class |
| Place | cell of the box centre on an 8 × 6 grid (8 columns, 6 rows) of the full frame |
| Size | box area / frame area: <1%, 1–2.5%, 2.5–5%, 5–10%, >10% |

The box is the saved primary box in full-frame pixels; the frame size comes from
the saved evidence, or for older rows from the camera's `detect` width and
height. Rows with neither form a "location unknown" group. Bins that share all
fields but the place and sit in touching cells (sides or corners) are merged, so
a box on a cell edge does not split its group; a spread-out error can therefore
form one large group, which the small grid map on each group shows. Groups are
listed by count, with first and last seen, up to 6 sample pictures, **Show all**
(the member cards, marked one by one as usual) and bulk **Secondlook right** /
**Ignore** for the members without a verdict (existing verdicts are kept; the
page asks first). The **Time window** select (all saved, 30 days, 7 days, 24
hours) narrows the rows grouped. The primary score range and the Secondlook
factor range are planned as later key fields (`KEY_FIELDS` in
`frigate/secondlook/clusters.py`).

## Rescued / suppressed filter

With **False positive** or **False negative** selected, a second toggle filters
by what the live pipeline did: **Suppressed / Not suppressed** for FP, **Rescued /
Not rescued** for FN. It uses the `event_id` stored when the item was saved (set
only when the tracked object became a detection): FN with it = rescued, FP
without it = suppressed. It is applied in the database query, so counts and
paging are exact, and it works for every saved row. The badge on a card comes
from Frigate's Event table at read time and can be absent for old items whose
Event was cleaned up; the filter does not depend on that.

## Suggested object filter mask

For a long-term **false positive** group with a known place, **Suggest mask**
shows a polygon over a saved frame and the YAML to add under
`cameras.<camera>.objects.filters.<label>.mask` (Frigate 0.18 named masks:
`friendly_name`, `enabled`, `coordinates` as relative `x,y` pairs). **Whole
boxes** covers every box in the group plus a 2% margin (the convex hull, so it
covers their union); **Bottom centres only** covers just the point Frigate tests
(the bottom centre of each box), which gives a smaller mask. Frigate drops any
object of that class whose bottom centre falls inside an object filter mask, real
ones included, so check the preview before using it. Nothing is written to the
config: copy the YAML, merge it under the camera and restart. Coordinates are
written with three decimals, and 1 as `1.0`, because 0.18.2 reads a value such
as `1.000` as a pixel coordinate.

## Small boxes at the frame edge

The **Hide small boxes touching the frame edge** toggle (default from
`review.hide_edge_small`, on) hides decisions whose saved primary box has any side
within `edge_margin_px` (4) source pixels of the frame edge **and** is small:
its width is below `small_dim_fraction` (0.05, 1/20) of the frame width, or its
height is below that fraction of the frame height (about 64 px wide or 36 px
tall on 1280x720). It is a filter only: the records are still saved, the list,
tab count and unreviewed count use the same filter, and the toolbar says how
many are hidden. Records saved without frame size and box (older history) are
never hidden. A leftover `small_area_fraction` key in existing YAML is ignored.

## Resource metrics and analysed-frame Debug

**Cameras** shows, per Secondlook camera, *analysed/s* (real Secondlook
inferences completed) and *skipped/s* (analyses Secondlook requested but did not
run: the bounded worker queue was full, so a first-frame request failed open to
the unscaled primary or a refresh kept the prior factor; or the crop could not be
taken). Frames Secondlook never asks about (stationary objects, labels not
configured, between refreshes) are not counted. Rates are counts between stats
snapshots and appear from the second snapshot; cameras without Secondlook or
without a real model show no graph.

**Storage** shows disk used by Secondlook saved data: the SQLite file (including
`-wal`/`-shm`) and the JPEG directory, rescanned at most once a minute.

**General** keeps two timing charts from the last completed **real** SL worker
sample (milliseconds): `processing_ms`, the worker time per analysis
(preprocessing, inference, matching; excluding frame/JPEG preparation and
evidence/IPC), and `queue_ms`, enqueue-to-worker-start wait. Processing is shown
instead of inference because it is the worker's service time: one worker can
finish at most about 1000 / processing analyses per second, and requests beyond
that fill `queue_max` and become skipped. Inference is a subset of processing,
so it is still exported in `/api/stats` (`inference_ms`) but not charted. Samples/last-update distinguish idle
from active inference. Missing model/no completed sample uses null, not fake zero.
CPU/memory are the shared embeddings process, not attributable per model.
Temperature remains device-wide where hardware reports it.

Live camera Debugging has a **Secondlook analysed frame** toggle. Its inset shows
the latest analysed still with its own boxes, class/score, age and stale notice.
It does not draw secondary boxes on current moving video. It is independent of
save/review, camera-authorized, capped at one result/camera and 4 MiB JPEG/result;
queued results are bounded. Disabled/no-result states are explicit.

The list API includes fixed-key in-memory diagnostics (reset on process restart):
primary threshold crossings, actual releases, qualifying kinds, results/no-hit/
unset-factor/misclass, hold timeouts, ended tracks, deposited kinds and skip/failure
reasons. Worker stats add queue/result drops. These explain eligibility without
persisting per-track debug histories. No-FP alone is not proof of a defect.

## Frigate+ YOLOv9 model for Secondlook

Secondlook can use a Frigate+ **YOLOv9 ONNX** model already on disk:

```yaml
secondlook:
  model:
    path: /config/model_cache/<model_id>
    plus_metadata_path: /config/model_cache/<model_id>.json
```

`<model_id>.json` is the Frigate+ model info file Frigate writes next to a
downloaded Frigate+ model. Labels (`labelMap`), input size (`width`/`height`),
layout (`inputShape`), pixel format (`pixelFormat`) and dtype (`inputDataType`)
are read from it; `labelmap_path` and `input_size` are ignored.
`path: plus://<model_id>` is the short form of the two lines above (an explicit
`plus_metadata_path` wins). Secondlook never downloads: the files must already
be in `/config/model_cache`.

At startup Secondlook refuses a model that is not suitable and logs why:
type other than `yolo-generic` (YOLO-NAS, MobileDet), a variant not built for
ONNX/OpenVINO (Coral/EdgeTPU, Hailo, RKNN), a non-square or integer input, an
input shape that differs from the metadata, more than one output, an output whose
channel count does not match the labels, or labels that do not include every
object a Secondlook camera is configured for. The last check applies to the
default COCO model too: a missing label would make every Secondlook result a
miss and `no_hit_factor` would suppress real detections. A refused model never
runs; factors stay unset and primary scores pass unscaled (fail-open), and
General shows the refusal reason.

Tested with synthetic ONNX fixtures, with COCO YOLOv9-s weights behind a
hand-written Frigate+-format metadata file, and with a real Frigate+ YOLOv9-c
(640x640) model on CPU. Not tested on GPU with a Frigate+ model.
