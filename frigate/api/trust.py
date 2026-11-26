"""Review Trust Overview, Human Feedback, and Golden Clips API endpoints."""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from functools import reduce
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from peewee import DatabaseError, DoesNotExist, operator
from pydantic import BaseModel, Field

from frigate.api.auth import (
    allow_any_authenticated,
    get_allowed_cameras_for_filter,
    require_camera_access,
)
from frigate.api.defs.tags import Tags
from frigate.const import EXPORT_DIR
from frigate.models import Event, Recordings, ReviewSegment, UserReviewStatus
from frigate.record.export import PlaybackSourceEnum, RecordingExporter
from frigate.review.trust_store import (
    VALID_GOLDEN_KINDS,
    VALID_VERDICTS,
    ReviewTrustStore,
    default_trust_db_path,
    get_review_trust_store,
    init_review_trust_store,
)
from frigate.secondlook.save_and_review import get_save_and_review_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.review])

BAR_SECONDS_MAP: dict[str, int] = {
    "1m": 60,
    "10m": 600,
    "1h": 3600,
    "4h": 14400,
    "24h": 86400,
}

NUM_BARS = 30


class FeedbackBody(BaseModel):
    verdict: Literal["good", "fp", "fn", "misclassified", "clear"]
    corrected_label: str | None = None
    reason: str | None = None


class GoldenClipCreateBody(BaseModel):
    kind: Literal["tp", "fp", "fn", "misclassified", "background"] = "tp"
    object_label: str | None = None
    condition_tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    export_video: bool = True


def _get_or_init_trust_store(request: Request) -> ReviewTrustStore:
    store = getattr(request.app, "review_trust_store", None)
    if store is not None:
        return store
    singleton = get_review_trust_store()
    if singleton is not None:
        request.app.review_trust_store = singleton
        return singleton

    config = getattr(request.app, "frigate_config", None)
    db_path = (
        default_trust_db_path(config.database.path)
        if config is not None
        else "/config/review_trust.db"
    )
    store = init_review_trust_store(db_path, EXPORT_DIR)
    request.app.review_trust_store = store
    return store


def _get_secondlook_store(request: Request):
    """Look up Secondlook store via request.app first, then module singleton."""
    app_store = getattr(request.app, "secondlook_store", None)
    if app_store is not None:
        return app_store
    return get_save_and_review_store()


def _parse_csv_filter(value: str) -> list[str] | None:
    if not value or value == "all":
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


def _segment_labels(data: dict[str, Any]) -> list[str]:
    objects = data.get("objects") or []
    labels: list[str] = []
    for obj in objects:
        s = str(obj)
        if s.endswith("-verified"):
            labels.append(s[: -len("-verified")])
        labels.append(s)
    for aud in data.get("audio") or []:
        labels.append(str(aud))
    return labels


def _segment_score(
    data: dict[str, Any], event_scores: dict[str, float]
) -> float | None:
    raw_score = data.get("score")
    if raw_score is not None:
        try:
            return float(raw_score)
        except (TypeError, ValueError):
            pass
    scores = [
        event_scores[eid]
        for eid in (data.get("detections") or [])
        if eid in event_scores
    ]
    if scores:
        return max(scores)
    return None


def _fetch_event_scores(event_ids: set[str]) -> dict[str, float]:
    if not event_ids:
        return {}
    scores: dict[str, float] = {}
    id_list = sorted(event_ids)
    try:
        for i in range(0, len(id_list), 500):
            chunk = id_list[i : i + 500]
            for ev in Event.select(
                Event.id, Event.top_score, Event.score, Event.data
            ).where(Event.id << chunk):
                val = ev.top_score
                if val is None or val == 0:
                    val = ev.score
                if (val is None or val == 0) and isinstance(ev.data, dict):
                    val = ev.data.get("top_score") or ev.data.get("score")
                if val is not None:
                    scores[ev.id] = float(val)
    except DatabaseError:
        logger.debug("Event score lookup skipped or failed", exc_info=True)
    return scores


def _empty_bucket(
    index: int, bucket_start: float, bucket_end: float
) -> dict[str, Any]:
    return {
        "index": index,
        "start_time": bucket_start,
        "end_time": bucket_end,
        "above": {
            "total": 0,
            "alerts": 0,
            "detections": 0,
            "tp": 0,
            "good": 0,
            "rescued_fn": 0,
            "auto_suppressed_fp": 0,
            "auto_misclassified": 0,
            "background_clips": 0,
        },
        "below": {
            "total": 0,
            "fp": 0,
            "fn": 0,
            "misclassified": 0,
        },
        "golden_clips": [],
        "timing": {
            "count": 0,
            "avg_discovery_ms": 0.0,
            "avg_resource_busy_ms": 0.0,
            "avg_confirmation_ms": 0.0,
            "avg_total_ms": 0.0,
            "slowest_ms": 0.0,
            "slowest_review_id": None,
        },
    }


def _bucket_index(ts: float, window_start: float, bar_seconds: int) -> int | None:
    if ts < window_start:
        return None
    idx = int((ts - window_start) // bar_seconds)
    if 0 <= idx < NUM_BARS:
        return idx
    return None


def _query_secondlook_bucket_counts(
    sl_store: Any,
    *,
    severity: str,
    window_start: float,
    window_end: float,
    bar_seconds: int,
    cameras: list[str],
    label_filter: list[str] | None,
    zone_filter: list[str] | None,
) -> tuple[dict[int, dict[str, int]], dict[str, int]]:
    """Aggregate Secondlook analyses / examples across the 30 buckets.

    Returns ``(per_bucket_counts, summary_totals)``.
    In ``alert`` view: counts ``rescued_fn`` (``live_outcome == 'fn_rescued'``)
    and ``auto_misclassified`` (``live == 'misclass'``).
    In ``detection`` view: counts ``auto_suppressed_fp`` (``live_outcome == 'fp_suppressed'``)
    and ``auto_misclassified`` (``live == 'misclass'``).
    """
    per_bucket: dict[int, dict[str, int]] = {}
    totals = {"rescued_fn": 0, "auto_suppressed_fp": 0, "auto_misclassified": 0}
    if sl_store is None or not cameras:
        return per_bucket, totals

    clauses = ["end_time >= ?", "end_time < ?"]
    params: list[Any] = [window_start, window_end]
    clauses.append(f"camera IN ({','.join('?' * len(cameras))})")
    params.extend(cameras)

    if label_filter:
        clauses.append(f"label IN ({','.join('?' * len(label_filter))})")
        params.extend(label_filter)

    sql = (
        "SELECT end_time, live, live_outcome FROM analyses "
        f"WHERE {' AND '.join(clauses)}"
    )

    try:
        with sl_store._lock, sl_store._connect() as conn:
            has_analyses = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='analyses'"
            ).fetchone()
            if has_analyses:
                rows = conn.execute(sql, params).fetchall()
            else:
                rows = []
    except sqlite3.Error:
        logger.debug("Secondlook analyses lookup failed", exc_info=True)
        return per_bucket, totals

    for row in rows:
        end_ts = float(row["end_time"])
        idx = _bucket_index(end_ts, window_start, bar_seconds)
        if idx is None:
            continue
        live = row["live"]
        live_outcome = row["live_outcome"]
        counts = per_bucket.setdefault(
            idx, {"rescued_fn": 0, "auto_suppressed_fp": 0, "auto_misclassified": 0}
        )
        if live == "misclass":
            counts["auto_misclassified"] += 1
            totals["auto_misclassified"] += 1
        elif severity == "alert" and live_outcome == "fn_rescued":
            counts["rescued_fn"] += 1
            totals["rescued_fn"] += 1
        elif severity == "detection" and live_outcome == "fp_suppressed":
            counts["auto_suppressed_fp"] += 1
            totals["auto_suppressed_fp"] += 1

    return per_bucket, totals


@router.get(
    "/review/overview",
    dependencies=[Depends(allow_any_authenticated())],
)
async def get_review_overview(
    request: Request,
    severity: Literal["alert", "detection"] = "alert",
    bar_size: Literal["1m", "10m", "1h", "4h", "24h"] = "1h",
    end_time: float | None = None,
    cameras: str = "all",
    labels: str = "all",
    zones: str = "all",
    sub_labels: str = "all",
    min_score: float | None = None,
    max_score: float | None = None,
    reviewed: int | None = None,
    allowed_cameras: list[str] = Depends(get_allowed_cameras_for_filter),
):
    """Return 30-bar Trust Overview timeline data for Alerts or Detections."""
    bar_seconds = BAR_SECONDS_MAP[bar_size]
    resolved_end = float(end_time) if end_time is not None else time.time()
    window_start = resolved_end - (NUM_BARS * bar_seconds)

    if cameras != "all":
        requested = {c.strip() for c in cameras.split(",") if c.strip()}
        camera_list = [c for c in allowed_cameras if c in requested]
    else:
        camera_list = list(allowed_cameras)

    label_filter = _parse_csv_filter(labels)
    zone_filter = _parse_csv_filter(zones)
    sub_label_filter = _parse_csv_filter(sub_labels)

    buckets = [
        _empty_bucket(
            i,
            window_start + i * bar_seconds,
            window_start + (i + 1) * bar_seconds,
        )
        for i in range(NUM_BARS)
    ]

    trust_store = _get_or_init_trust_store(request)
    sl_store = _get_secondlook_store(request)

    if not camera_list:
        golden_summary = trust_store.get_golden_summary(cameras=[])
        return JSONResponse(
            content={
                "severity": severity,
                "bar_size": bar_size,
                "bar_seconds": bar_seconds,
                "num_bars": NUM_BARS,
                "start_time": window_start,
                "end_time": resolved_end,
                "golden_total": golden_summary["total"],
                "unreviewed_fn_count": 0,
                "buckets": buckets,
                "feedback_by_review_id": {},
            }
        )

    # Query ReviewSegments inside [window_start, resolved_end)
    clauses = [
        ReviewSegment.camera << camera_list,
        ReviewSegment.severity == severity,
        ReviewSegment.start_time >= window_start,
        ReviewSegment.start_time < resolved_end,
    ]
    rows = list(
        ReviewSegment.select(
            ReviewSegment.id,
            ReviewSegment.camera,
            ReviewSegment.start_time,
            ReviewSegment.end_time,
            ReviewSegment.severity,
            ReviewSegment.data,
        )
        .where(reduce(operator.and_, clauses))
        .dicts()
    )

    # Optional UserReviewStatus lookup when reviewed filter is requested
    reviewed_ids: set[str] | None = None
    if reviewed in (0, 1) and rows:
        seg_ids = [r["id"] for r in rows]
        try:
            user_id = request.headers.get("remote-user") or "admin"
            reviewed_ids = {
                s.review_segment_id
                for s in UserReviewStatus.select(UserReviewStatus.review_segment).where(
                    (UserReviewStatus.review_segment << seg_ids)
                    & (UserReviewStatus.user_id == user_id)
                    & (UserReviewStatus.has_been_reviewed == True)
                )
            }
        except DatabaseError:
            reviewed_ids = set()

    # Load event scores only when score filtering is requested
    event_scores: dict[str, float] = {}
    if min_score is not None or max_score is not None:
        all_det_ids: set[str] = set()
        for r in rows:
            data = r.get("data") or {}
            if data.get("score") is None:
                for eid in data.get("detections") or []:
                    all_det_ids.add(str(eid))
        event_scores = _fetch_event_scores(all_det_ids)

    filtered_segments: list[dict[str, Any]] = []
    for r in rows:
        data = r.get("data") or {}
        if not isinstance(data, dict):
            data = {}

        if label_filter is not None:
            seg_labels = _segment_labels(data)
            if not any(lbl in seg_labels for lbl in label_filter):
                continue

        if zone_filter is not None:
            seg_zones = [str(z) for z in (data.get("zones") or [])]
            if not any(z in seg_zones for z in zone_filter):
                continue

        if sub_label_filter is not None:
            seg_subs = [str(s) for s in (data.get("sub_labels") or [])]
            if not any(sub in seg_subs for sub in sub_label_filter):
                continue

        if min_score is not None or max_score is not None:
            score_val = _segment_score(data, event_scores)
            if score_val is None:
                continue
            if min_score is not None and score_val < min_score:
                continue
            if max_score is not None and score_val > max_score:
                continue

        if reviewed_ids is not None:
            is_rev = r["id"] in reviewed_ids
            if reviewed == 1 and not is_rev:
                continue
            if reviewed == 0 and is_rev:
                continue

        filtered_segments.append(r)

    # Human feedback map across the window (for filtered cameras)
    feedback_map = trust_store.get_feedback_map(
        window_start, resolved_end, cameras=camera_list
    )

    # Bucket filtered segments with strict no-double-counting:
    # - verdict in ('fp', 'fn', 'misclassified') -> counted ONLY in below, NOT in above
    # - verdict == 'good' -> counted in above (both above.good and above.total)
    # - no verdict -> counted in above (above.tp and above.total)
    feedback_by_review_id: dict[str, dict[str, Any]] = {}
    for seg in filtered_segments:
        seg_id = seg["id"]
        start_ts = float(seg["start_time"])
        idx = _bucket_index(start_ts, window_start, bar_seconds)
        if idx is None:
            continue
        bucket = buckets[idx]
        fb = feedback_map.get(seg_id)
        above_key = "alerts" if severity == "alert" else "detections"
        if fb is not None:
            feedback_by_review_id[seg_id] = fb
            verdict = fb["verdict"]
            if verdict in ("fp", "fn", "misclassified"):
                bucket["below"][verdict] += 1
                bucket["below"]["total"] += 1
                continue
            if verdict == "good":
                bucket["above"]["good"] += 1
                bucket["above"][above_key] += 1
                bucket["above"]["total"] += 1
                continue

        bucket["above"]["tp"] += 1
        bucket["above"][above_key] += 1
        bucket["above"]["total"] += 1

    # Include any human-flagged FN feedback in this window that was NOT already
    # matched to a ReviewSegment in this severity/query (for instance, an FN
    # logged on a detection segment or an independent FN marker).
    seen_ids = {seg["id"] for seg in filtered_segments}
    for rev_id, fb in feedback_map.items():
        if rev_id in seen_ids:
            continue
        if fb["severity"] != severity:
            continue
        idx = _bucket_index(float(fb["start_time"]), window_start, bar_seconds)
        if idx is None:
            continue
        verdict = fb["verdict"]
        if verdict in ("fp", "fn", "misclassified"):
            feedback_by_review_id[rev_id] = fb
            buckets[idx]["below"][verdict] += 1
            buckets[idx]["below"]["total"] += 1

    # Secondlook automatic metrics above the line
    sl_per_bucket, _ = _query_secondlook_bucket_counts(
        sl_store,
        severity=severity,
        window_start=window_start,
        window_end=resolved_end,
        bar_seconds=bar_seconds,
        cameras=camera_list,
        label_filter=label_filter,
        zone_filter=zone_filter,
    )
    for idx, sl_counts in sl_per_bucket.items():
        above = buckets[idx]["above"]
        above["rescued_fn"] += sl_counts["rescued_fn"]
        above["auto_suppressed_fp"] += sl_counts["auto_suppressed_fp"]
        above["auto_misclassified"] += sl_counts["auto_misclassified"]
        above["total"] += (
            sl_counts["rescued_fn"]
            + sl_counts["auto_suppressed_fp"]
            + sl_counts["auto_misclassified"]
        )

    # Golden clips inside window (per-bucket list + background_clips in detection view)
    golden_in_window = trust_store.list_golden_clips(
        start_time=window_start,
        end_time=resolved_end,
        cameras=camera_list,
    )
    for clip in golden_in_window:
        idx = _bucket_index(float(clip["start_time"]), window_start, bar_seconds)
        if idx is None:
            continue
        if clip["kind"] == "background" and severity == "detection":
            buckets[idx]["above"]["background_clips"] += 1
            buckets[idx]["above"]["total"] += 1
        if clip["severity"] == severity or clip["kind"] == "background":
            buckets[idx]["golden_clips"].append(clip)

    # Alert timings per bucket (averages + slowest)
    timings = trust_store.get_timings_in_range(
        window_start, resolved_end, cameras=camera_list
    )
    timings_by_bucket: dict[int, list[dict[str, Any]]] = {}
    for row in timings:
        idx = _bucket_index(float(row["alert_time"]), window_start, bar_seconds)
        if idx is not None:
            timings_by_bucket.setdefault(idx, []).append(row)

    for idx, rows_in_bucket in timings_by_bucket.items():
        n = len(rows_in_bucket)
        slowest = max(rows_in_bucket, key=lambda item: item["total_ms"])
        buckets[idx]["timing"] = {
            "count": n,
            "avg_discovery_ms": round(
                sum(r["discovery_ms"] for r in rows_in_bucket) / n, 2
            ),
            "avg_resource_busy_ms": round(
                sum(r["resource_busy_ms"] for r in rows_in_bucket) / n, 2
            ),
            "avg_confirmation_ms": round(
                sum(r["confirmation_ms"] for r in rows_in_bucket) / n, 2
            ),
            "avg_total_ms": round(sum(r["total_ms"] for r in rows_in_bucket) / n, 2),
            "slowest_ms": round(float(slowest["total_ms"]), 2),
            "slowest_review_id": slowest["review_id"],
        }

    golden_summary = trust_store.get_golden_summary(cameras=camera_list)
    unreviewed_fn_count = trust_store.count_unreviewed_fn(
        start_time=window_start,
        end_time=resolved_end,
        cameras=camera_list,
    )

    return JSONResponse(
        content={
            "severity": severity,
            "bar_size": bar_size,
            "bar_seconds": bar_seconds,
            "num_bars": NUM_BARS,
            "start_time": window_start,
            "end_time": resolved_end,
            "golden_total": golden_summary["total"],
            "unreviewed_fn_count": unreviewed_fn_count,
            "buckets": buckets,
            "feedback_by_review_id": feedback_by_review_id,
        }
    )


@router.post(
    "/review/{review_id}/feedback",
    dependencies=[Depends(allow_any_authenticated())],
)
async def set_review_feedback(
    request: Request,
    review_id: str,
    body: FeedbackBody,
):
    """Set or clear human feedback verdict for a review segment."""
    try:
        segment = ReviewSegment.get(ReviewSegment.id == review_id)
    except DoesNotExist:
        return JSONResponse(
            content={"success": False, "message": f"Review {review_id} not found"},
            status_code=404,
        )

    await require_camera_access(segment.camera, request=request)
    trust_store = _get_or_init_trust_store(request)

    if body.verdict == "clear":
        cleared = trust_store.clear_feedback(review_id)
        return JSONResponse(
            content={
                "success": True,
                "cleared": cleared,
                "review_id": review_id,
                "feedback": None,
            }
        )

    if body.verdict not in VALID_VERDICTS:
        return JSONResponse(
            content={"success": False, "message": f"Invalid verdict: {body.verdict}"},
            status_code=400,
        )

    feedback = trust_store.set_feedback(
        review_id=review_id,
        camera=segment.camera,
        severity=str(segment.severity),
        verdict=body.verdict,
        start_time=float(segment.start_time),
        corrected_label=body.corrected_label,
        reason=body.reason,
    )
    return JSONResponse(
        content={
            "success": True,
            "review_id": review_id,
            "feedback": feedback,
        }
    )


@router.get(
    "/review/golden_clips",
    dependencies=[Depends(allow_any_authenticated())],
)
async def list_golden_clips(
    request: Request,
    cameras: str = "all",
    allowed_cameras: list[str] = Depends(get_allowed_cameras_for_filter),
):
    """List Golden Clips and condition coverage counts."""
    trust_store = _get_or_init_trust_store(request)
    if cameras != "all":
        requested = {c.strip() for c in cameras.split(",") if c.strip()}
        camera_list = [c for c in allowed_cameras if c in requested]
    else:
        camera_list = list(allowed_cameras)

    summary = trust_store.get_golden_summary(cameras=camera_list)
    return JSONResponse(content=summary)


def _try_export_golden_clip(
    request: Request,
    *,
    clip_id: str,
    camera: str,
    start_time: float,
    end_time: float | None,
) -> tuple[str | None, str | None]:
    """Attempt FFmpeg clip export from Recordings if recordings exist on disk.

    Returns ``(export_id, video_path)`` or ``(None, None)`` when unavailable.
    """
    config = getattr(request.app, "frigate_config", None)
    if config is None or camera not in config.cameras:
        return None, None

    seg_end = float(end_time) if end_time is not None else float(start_time) + 10.0
    if seg_end <= start_time:
        seg_end = float(start_time) + 1.0

    try:
        rec_count = (
            Recordings.select()
            .where(
                Recordings.camera == camera,
                Recordings.start_time.between(start_time, seg_end)
                | Recordings.end_time.between(start_time, seg_end)
                | (
                    (start_time > Recordings.start_time)
                    & (seg_end < Recordings.end_time)
                ),
            )
            .count()
        )
    except DatabaseError:
        return None, None

    if rec_count <= 0:
        return None, None

    export_id = f"golden_{camera}_{clip_id[-8:]}"
    try:
        exporter = RecordingExporter(
            config=config,
            id=export_id,
            camera=camera,
            name=f"Golden Clip {clip_id}",
            image=None,
            start_time=int(start_time),
            end_time=int(seg_end),
            playback_source=PlaybackSourceEnum.recordings,
        )
        exporter.start()
        return export_id, None
    except (OSError, RuntimeError):
        logger.debug("Golden clip recording export failed to start", exc_info=True)
        return None, None


@router.post(
    "/review/{review_id}/golden",
    dependencies=[Depends(allow_any_authenticated())],
)
async def create_golden_clip_from_review(
    request: Request,
    review_id: str,
    body: GoldenClipCreateBody,
):
    """Save a Golden Clip (and paired JSON sidecar) from a ReviewSegment."""
    try:
        segment = ReviewSegment.get(ReviewSegment.id == review_id)
    except DoesNotExist:
        return JSONResponse(
            content={"success": False, "message": f"Review {review_id} not found"},
            status_code=404,
        )

    await require_camera_access(segment.camera, request=request)
    if body.kind not in VALID_GOLDEN_KINDS:
        return JSONResponse(
            content={"success": False, "message": f"Invalid kind: {body.kind}"},
            status_code=400,
        )

    trust_store = _get_or_init_trust_store(request)
    data = segment.data if isinstance(segment.data, dict) else {}
    inferred_label = body.object_label
    if not inferred_label and data.get("objects"):
        first_obj = str(data["objects"][0])
        inferred_label = first_obj.removesuffix("-verified")

    clip_id = f"gc_{uuid.uuid4().hex[:12]}"
    export_id: str | None = None
    video_path: str | None = None
    if body.export_video:
        export_id, video_path = _try_export_golden_clip(
            request,
            clip_id=clip_id,
            camera=segment.camera,
            start_time=float(segment.start_time),
            end_time=float(segment.end_time) if segment.end_time is not None else None,
        )

    clip = trust_store.create_golden_clip(
        clip_id=clip_id,
        review_id=review_id,
        camera=segment.camera,
        severity=str(segment.severity),
        start_time=float(segment.start_time),
        end_time=float(segment.end_time) if segment.end_time is not None else None,
        kind=body.kind,
        object_label=inferred_label,
        condition_tags=body.condition_tags,
        export_id=export_id,
        video_path=video_path,
        notes=body.notes,
    )
    return JSONResponse(content={"success": True, "clip": clip}, status_code=201)


@router.delete(
    "/review/golden_clips/{clip_id}",
    dependencies=[Depends(allow_any_authenticated())],
)
async def delete_golden_clip(
    request: Request,
    clip_id: str,
):
    """Delete a Golden Clip by ID."""
    trust_store = _get_or_init_trust_store(request)
    existing = trust_store.get_golden_clip(clip_id)
    if existing is None:
        return JSONResponse(
            content={"success": False, "message": f"Golden clip {clip_id} not found"},
            status_code=404,
        )

    await require_camera_access(existing["camera"], request=request)
    deleted = trust_store.delete_golden_clip(clip_id)
    return JSONResponse(content={"success": deleted, "id": clip_id})
