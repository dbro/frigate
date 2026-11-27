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
    MAX_VALID_TIMING_MS,
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


def _fetch_event_details(event_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Batch-fetch Event label/sub_label/score metadata for review segments."""
    if not event_ids:
        return {}
    details: dict[str, dict[str, Any]] = {}
    id_list = sorted(event_ids)
    try:
        for i in range(0, len(id_list), 500):
            chunk = id_list[i : i + 500]
            for ev in Event.select(
                Event.id,
                Event.label,
                Event.sub_label,
                Event.top_score,
                Event.score,
                Event.start_time,
                Event.end_time,
                Event.data,
            ).where(Event.id << chunk):
                ev_data = ev.data if isinstance(ev.data, dict) else {}
                top_score = ev.top_score
                if (top_score is None or top_score == 0) and ev_data.get("top_score"):
                    top_score = ev_data.get("top_score")
                score = ev.score
                if (score is None or score == 0) and ev_data.get("score"):
                    score = ev_data.get("score")
                effective_score = (
                    float(top_score)
                    if top_score is not None and float(top_score) > 0
                    else (float(score) if score is not None else None)
                )
                details[ev.id] = {
                    "id": ev.id,
                    "label": ev.label,
                    "sub_label": ev.sub_label,
                    "score": round(float(score), 4) if score is not None else None,
                    "top_score": (
                        round(float(top_score), 4)
                        if top_score is not None
                        else None
                    ),
                    "effective_score": effective_score,
                }
    except DatabaseError:
        logger.debug("Event details lookup skipped or failed", exc_info=True)
    return details


def _fetch_event_scores(event_ids: set[str]) -> dict[str, float]:
    details = _fetch_event_details(event_ids)
    return {
        eid: float(info["effective_score"])
        for eid, info in details.items()
        if info.get("effective_score") is not None
    }


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
            "segment_count": 0,
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


def _query_secondlook_analyses(
    sl_store: Any,
    *,
    window_start: float,
    window_end: float,
    cameras: list[str],
    label_filter: list[str] | None,
    zone_filter: list[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Load Secondlook ``analyses`` rows for matching against ReviewSegments.

    Returns ``(by_event_id, suppressed_fp_items)`` where:
    - ``by_event_id`` maps ``event_id`` (padded +-1h around the window so tracks
      ending slightly after a segment start still match) to Secondlook details.
    - ``suppressed_fp_items`` holds ``live_outcome == 'fp_suppressed'`` items
      whose timestamp falls in ``[window_start, window_end)``, enriched with
      their saved ``examples`` still image ID when available so the Detections
      grid can render their thumbnails directly.
    """
    by_event_id: dict[str, dict[str, Any]] = {}
    suppressed_fp_items: list[dict[str, Any]] = []
    if sl_store is None or not cameras:
        return by_event_id, suppressed_fp_items

    pad = 3600.0
    clauses = ["end_time >= ?", "end_time < ?"]
    params: list[Any] = [window_start - pad, window_end + pad]
    clauses.append(f"camera IN ({','.join('?' * len(cameras))})")
    params.extend(cameras)

    if label_filter:
        clauses.append(f"label IN ({','.join('?' * len(label_filter))})")
        params.extend(label_filter)

    sql = (
        "SELECT camera, event_id, track_id, label, start_time, end_time, live, "
        "live_outcome, misclass_label, primary_score, sl_score, factor "
        f"FROM analyses WHERE {' AND '.join(clauses)}"
    )

    examples_by_track: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        with sl_store._lock, sl_store._connect() as conn:
            has_analyses = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='analyses'"
            ).fetchone()
            if has_analyses:
                rows = conn.execute(sql, params).fetchall()
            else:
                rows = []

            has_examples = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='examples'"
            ).fetchone()
            if has_examples and rows:
                suppressed_tracks = [
                    (str(r["camera"]), str(r["track_id"]))
                    for r in rows
                    if r["live_outcome"] == "fp_suppressed"
                ]
                if suppressed_tracks:
                    ex_clauses = [
                        "kind = 'false_positive'",
                        f"camera IN ({','.join('?' * len(cameras))})",
                        "COALESCE(frame_time, created_at) >= ?",
                        "COALESCE(frame_time, created_at) < ?",
                    ]
                    ex_params: list[Any] = [
                        *cameras,
                        window_start - pad,
                        window_end + pad,
                    ]
                    ex_rows = conn.execute(
                        "SELECT id, camera, track_id, frame_time, still_path, "
                        "zones, mark, created_at "
                        f"FROM examples WHERE {' AND '.join(ex_clauses)} "
                        "ORDER BY created_at DESC",
                        ex_params,
                    ).fetchall()
                    for ex in ex_rows:
                        key = (str(ex["camera"]), str(ex["track_id"]))
                        if key not in examples_by_track:
                            raw_zones = ex["zones"]
                            parsed_zones: list[str] = []
                            if raw_zones:
                                try:
                                    import json as _json

                                    z = _json.loads(raw_zones)
                                    if isinstance(z, list):
                                        parsed_zones = [str(v) for v in z]
                                except (ValueError, TypeError):
                                    pass
                            examples_by_track[key] = {
                                "example_id": str(ex["id"]),
                                "frame_time": (
                                    float(ex["frame_time"])
                                    if ex["frame_time"] is not None
                                    else None
                                ),
                                "has_still": bool(ex["still_path"]),
                                "zones": parsed_zones,
                                "mark": ex["mark"],
                            }
    except sqlite3.Error:
        logger.debug("Secondlook analyses lookup failed", exc_info=True)
        return by_event_id, suppressed_fp_items

    for row in rows:
        cam = str(row["camera"])
        track_id = str(row["track_id"])
        start_ts = (
            float(row["start_time"]) if row["start_time"] is not None else None
        )
        end_ts = float(row["end_time"])
        live = row["live"]
        live_outcome = row["live_outcome"]
        event_id = row["event_id"]
        info = {
            "camera": cam,
            "event_id": event_id,
            "track_id": track_id,
            "label": row["label"],
            "start_time": start_ts,
            "end_time": end_ts,
            "live": live,
            "live_outcome": live_outcome,
            "misclass_label": row["misclass_label"],
            "primary_score": (
                round(float(row["primary_score"]), 4)
                if row["primary_score"] is not None
                else None
            ),
            "sl_score": (
                round(float(row["sl_score"]), 4)
                if row["sl_score"] is not None
                else None
            ),
            "factor": (
                round(float(row["factor"]), 3)
                if row["factor"] is not None
                else None
            ),
        }
        if event_id:
            by_event_id[str(event_id)] = info
        if live_outcome == "fp_suppressed":
            ex_info = examples_by_track.get((cam, track_id))
            item_ts = (
                ex_info["frame_time"]
                if ex_info and ex_info["frame_time"] is not None
                else (start_ts if start_ts is not None else end_ts)
            )
            if not (window_start <= item_ts < window_end):
                continue
            item_zones = ex_info["zones"] if ex_info else []
            if zone_filter is not None and item_zones:
                if not any(z in item_zones for z in zone_filter):
                    continue
            duration_sec = (
                round(max(0.0, end_ts - start_ts), 2)
                if start_ts is not None
                else None
            )
            suppressed_fp_items.append(
                {
                    "id": ex_info["example_id"] if ex_info else f"sl_{track_id}",
                    "example_id": ex_info["example_id"] if ex_info else None,
                    "track_id": track_id,
                    "camera": cam,
                    "label": row["label"],
                    "start_time": item_ts,
                    "end_time": end_ts,
                    "duration_sec": duration_sec,
                    "primary_score": info["primary_score"],
                    "sl_score": info["sl_score"],
                    "factor": info["factor"],
                    "zones": item_zones,
                    "mark": ex_info["mark"] if ex_info else None,
                    "has_still": bool(ex_info and ex_info["has_still"]),
                }
            )

    suppressed_fp_items.sort(key=lambda item: item["start_time"], reverse=True)
    return by_event_id, suppressed_fp_items


def _normalize_timing_row(
    row: dict[str, Any], severity: str
) -> dict[str, Any] | None:
    """Apply detection zero-confirmation and >30s outlier guard to a timing row."""
    disc_ms = max(0.0, float(row["discovery_ms"]))
    busy_ms = max(0.0, float(row["resource_busy_ms"]))
    conf_ms = (
        0.0
        if severity == "detection"
        else max(0.0, float(row["confirmation_ms"]))
    )
    total_ms = disc_ms + busy_ms + conf_ms
    if total_ms > MAX_VALID_TIMING_MS:
        return None
    alert_time = float(row["alert_time"])
    first_detected = float(
        row.get("first_detected_time")
        if row.get("first_detected_time") is not None
        else max(0.0, alert_time - (conf_ms / 1000.0))
    )
    return {
        "review_id": row["review_id"],
        "severity": severity,
        "alert_time": alert_time,
        "first_detected_time": first_detected,
        "discovery_ms": round(disc_ms, 2),
        "resource_busy_ms": round(busy_ms, 2),
        "confirmation_ms": round(conf_ms, 2),
        "total_ms": round(total_ms, 2),
    }


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
                "review_meta_by_id": {},
                "suppressed_fp_items": [],
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

    # Optional UserReviewStatus lookup when unreviewed-only filter (reviewed=0) is requested.
    # Matches stock frigate/api/review.py: reviewed=0 shows only unreviewed items;
    # reviewed=1 (or None) shows ALL items (both reviewed and unreviewed).
    reviewed_ids: set[str] | None = None
    if reviewed == 0 and rows:
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

    # Batch-fetch Event details (labels, sub_labels, scores) for all segments
    all_det_ids: set[str] = set()
    for r in rows:
        data = r.get("data") or {}
        if isinstance(data, dict):
            for eid in data.get("detections") or []:
                all_det_ids.add(str(eid))
    event_details = _fetch_event_details(all_det_ids)
    event_scores = {
        eid: float(info["effective_score"])
        for eid, info in event_details.items()
        if info.get("effective_score") is not None
    }

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

        if reviewed_ids is not None and r["id"] in reviewed_ids:
            continue

        filtered_segments.append(r)

    # Human feedback map across the window (for filtered cameras)
    feedback_map = trust_store.get_feedback_map(
        window_start, resolved_end, cameras=camera_list
    )

    # Secondlook analyses lookup by event_id so each ReviewSegment is classified
    # into EXACTLY ONE category (never double-counted across alerts + rescued_fn
    # / auto_misclassified).
    sl_by_event_id, suppressed_fp_items = _query_secondlook_analyses(
        sl_store,
        window_start=window_start,
        window_end=resolved_end,
        cameras=camera_list,
        label_filter=label_filter,
        zone_filter=zone_filter,
    )

    # Per-review timing lookup for both thumbnail info badges AND timing chart
    # (ensures timing bars share exact same segment filtering & bucket index).
    seg_timings_raw = trust_store.get_timings_for_ids(
        [seg["id"] for seg in filtered_segments]
    )

    # Bucket filtered segments with strict 1-to-1 classification (no double counting):
    # 1. Human verdict in ('fp', 'fn', 'misclassified') -> counted ONLY below
    # 2. Secondlook live == 'misclass' on a segment track -> above.auto_misclassified
    # 3. Secondlook live_outcome == 'fn_rescued' (alert view) -> above.rescued_fn
    # 4. Otherwise -> above.alerts (alert view) or above.detections (detection view)
    feedback_by_review_id: dict[str, dict[str, Any]] = {}
    review_meta_by_id: dict[str, dict[str, Any]] = {}
    timings_by_bucket: dict[int, list[dict[str, Any]]] = {}
    above_key = "alerts" if severity == "alert" else "detections"

    for seg in filtered_segments:
        seg_id = seg["id"]
        start_ts = float(seg["start_time"])
        end_ts = float(seg["end_time"]) if seg.get("end_time") is not None else None
        idx = _bucket_index(start_ts, window_start, bar_seconds)
        if idx is None:
            continue
        bucket = buckets[idx]
        bucket["timing"]["segment_count"] += 1
        data = seg.get("data") if isinstance(seg.get("data"), dict) else {}
        det_ids = [str(eid) for eid in (data.get("detections") or [])]

        # Find highest-priority Secondlook match among this segment's tracks
        sl_match: dict[str, Any] | None = None
        for eid in det_ids:
            cand = sl_by_event_id.get(eid)
            if cand is None:
                continue
            if cand["live"] == "misclass":
                sl_match = cand
                break
            if cand["live_outcome"] == "fn_rescued" and sl_match is None:
                sl_match = cand
            elif sl_match is None:
                sl_match = cand

        fb = feedback_map.get(seg_id)
        is_good = False
        category = above_key

        if fb is not None:
            feedback_by_review_id[seg_id] = fb
            verdict = fb["verdict"]
            if verdict in ("fp", "fn", "misclassified"):
                bucket["below"][verdict] += 1
                bucket["below"]["total"] += 1
                category = (
                    "human_misclassified" if verdict == "misclassified" else verdict
                )
            else:
                is_good = verdict == "good"
                if sl_match is not None and sl_match["live"] == "misclass":
                    bucket["above"]["auto_misclassified"] += 1
                    bucket["above"]["total"] += 1
                    category = "auto_misclassified"
                elif (
                    severity == "alert"
                    and sl_match is not None
                    and sl_match["live_outcome"] == "fn_rescued"
                ):
                    bucket["above"]["rescued_fn"] += 1
                    bucket["above"]["total"] += 1
                    category = "rescued_fn"
                else:
                    bucket["above"]["good"] += 1
                    bucket["above"][above_key] += 1
                    bucket["above"]["total"] += 1
                    category = above_key
        else:
            if sl_match is not None and sl_match["live"] == "misclass":
                bucket["above"]["auto_misclassified"] += 1
                bucket["above"]["total"] += 1
                category = "auto_misclassified"
            elif (
                severity == "alert"
                and sl_match is not None
                and sl_match["live_outcome"] == "fn_rescued"
            ):
                bucket["above"]["rescued_fn"] += 1
                bucket["above"]["total"] += 1
                category = "rescued_fn"
            else:
                bucket["above"]["tp"] += 1
                bucket["above"][above_key] += 1
                bucket["above"]["total"] += 1
                category = above_key

        # Build enriched object list with scores from Event table (falling back to segment data)
        clip_objects: list[dict[str, Any]] = []
        for eid in det_ids:
            ev_info = event_details.get(eid)
            if ev_info is not None:
                clip_objects.append(
                    {
                        "id": eid,
                        "label": ev_info["label"],
                        "sub_label": ev_info["sub_label"],
                        "score": ev_info["score"],
                        "top_score": ev_info["top_score"],
                    }
                )
        if not clip_objects:
            for obj_lbl in data.get("objects") or []:
                clip_objects.append(
                    {
                        "id": None,
                        "label": str(obj_lbl).removesuffix("-verified"),
                        "sub_label": None,
                        "score": None,
                        "top_score": None,
                    }
                )

        raw_timing = seg_timings_raw.get(seg_id)
        norm_timing = (
            _normalize_timing_row(raw_timing, severity)
            if raw_timing is not None
            else None
        )
        if norm_timing is not None:
            timings_by_bucket.setdefault(idx, []).append(norm_timing)

        review_meta_by_id[seg_id] = {
            "category": category,
            "is_good": is_good,
            "duration_sec": (
                round(max(0.0, end_ts - start_ts), 2) if end_ts is not None else None
            ),
            "zones": [str(z) for z in (data.get("zones") or [])],
            "objects": clip_objects,
            "timing": norm_timing,
            "secondlook": (
                {
                    "live": sl_match["live"],
                    "live_outcome": sl_match["live_outcome"],
                    "label": sl_match["label"],
                    "misclass_label": sl_match["misclass_label"],
                    "primary_score": sl_match["primary_score"],
                    "sl_score": sl_match["sl_score"],
                    "factor": sl_match["factor"],
                }
                if sl_match is not None
                and (
                    sl_match["live"] == "misclass"
                    or sl_match["live_outcome"] == "fn_rescued"
                )
                else None
            ),
        }

    # Include any human-flagged feedback in this window that was NOT already
    # matched to a ReviewSegment in this severity/query.
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

    # Detections view only: stack Secondlook auto-suppressed FPs above the line.
    # Suppressed FPs never generate a Frigate ReviewSegment (event_id is NULL)
    # and are returned in suppressed_fp_items so thumbnails appear below.
    if severity == "detection":
        for sl_item in suppressed_fp_items:
            idx = _bucket_index(sl_item["start_time"], window_start, bar_seconds)
            if idx is None:
                continue
            buckets[idx]["above"]["auto_suppressed_fp"] += 1
            buckets[idx]["above"]["total"] += 1

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

    # Also include any timing rows whose review_id wasn't matched via
    # filtered_segments only when no segment-level filters are active (covers
    # live in-progress or pruned segments while preserving exact alignment when
    # filtered segments exist).
    has_segment_filters = any(
        [
            label_filter,
            zone_filter,
            sub_label_filter,
            min_score is not None,
            max_score is not None,
            reviewed == 0,
        ]
    )
    if not has_segment_filters:
        all_window_timings = trust_store.get_timings_in_range(
            window_start, resolved_end, cameras=camera_list, severity=severity
        )
        for row in all_window_timings:
            if row["review_id"] in seen_ids:
                continue
            norm = _normalize_timing_row(row, severity)
            if norm is None:
                continue
            idx = _bucket_index(norm["alert_time"], window_start, bar_seconds)
            if idx is not None:
                timings_by_bucket.setdefault(idx, []).append(norm)

    for idx, rows_in_bucket in timings_by_bucket.items():
        n = len(rows_in_bucket)
        slowest = max(rows_in_bucket, key=lambda item: item["total_ms"])
        seg_cnt = buckets[idx]["timing"]["segment_count"]
        buckets[idx]["timing"] = {
            "count": n,
            "segment_count": max(seg_cnt, n),
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
            "review_meta_by_id": review_meta_by_id,
            "suppressed_fp_items": suppressed_fp_items if severity == "detection" else [],
        }
    )


@router.get(
    "/review/{review_id}/timing",
    dependencies=[Depends(allow_any_authenticated())],
)
async def get_review_timing(
    request: Request,
    review_id: str,
):
    """Return Alert/Detection Timing breakdown for a single review segment."""
    try:
        segment = ReviewSegment.get(ReviewSegment.id == review_id)
    except DoesNotExist:
        return JSONResponse(
            content={"success": False, "message": f"Review {review_id} not found"},
            status_code=404,
        )

    await require_camera_access(segment.camera, request=request)
    trust_store = _get_or_init_trust_store(request)
    sev = "detection" if str(segment.severity) == "detection" else "alert"
    raw = trust_store.get_timing_for_review(review_id)
    norm = _normalize_timing_row(raw, sev) if raw is not None else None
    if norm is None:
        return JSONResponse(
            content={
                "review_id": review_id,
                "camera": segment.camera,
                "severity": sev,
                "available": False,
                "alert_time": float(segment.start_time),
                "first_detected_time": float(segment.start_time),
                "discovery_ms": 0.0,
                "resource_busy_ms": 0.0,
                "confirmation_ms": 0.0,
                "total_ms": 0.0,
            }
        )
    return JSONResponse(
        content={
            **norm,
            "camera": segment.camera,
            "available": True,
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
