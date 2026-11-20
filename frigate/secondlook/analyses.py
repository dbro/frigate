"""One row per tracked object Secondlook was asked about: the ``analyses`` table.

Written once when the track is finished (at track end, when a late answer for
an ended track is applied, or when the 15 s late window expires without one;
see track/object_processing.finish_secondlook_track). No images. Agreements
are stored as well as disagreements, so agreement counts, score and factor
distributions and model-against-model comparisons on the same tracks can be
read back. Objects Secondlook was never asked about (another label, stationary,
crop failed on every frame) have no row.

outcome: what Secondlook's answers were, by precedence over every answer the
track received (the first that applies):

- ``result``: at least one real analysis found a same-class box on the crop
  (applied_by ``secondlook_scale``); a factor was applied.
- ``no_hit``: real analyses ran, none found a same-class box above
  ``score_threshold``. The factor is ``no_hit_factor`` (default 0.0), or null
  when that is null.
- ``fail_open``: answers arrived, but none had a real inference behind it (the
  model was refused or unavailable, or inference / post-processing raised);
  no factor, the primary decided alone.
- ``timeout``: a request was accepted (the pending notice reached the track)
  but no answer reached it before it was finished: not before track end nor
  within the 15 s late window (or the answer was lost from the bounded result
  queue). No factor.
- ``dropped``: the track's first request was refused because the Secondlook
  queue was full (the queue-full fail-open notice) and no later answer
  arrived. No factor.

live: did the factor change what the pipeline did, at track end (the same
rules as the hourly summary and Review): ``misclass`` (Secondlook saw another
class in the same place; checked first, as for saved Review items), then
``false_positive`` / ``false_negative`` with ``live_outcome`` fp_suppressed /
fp_not_suppressed / fn_rescued / fn_not_rescued, else ``agree`` when a factor
was applied at some point but did not flip the decision, else ``none`` (no
factor was ever applied: no_hit with a null no_hit_factor, fail_open,
timeout, dropped).

The insert is ``INSERT OR IGNORE`` on UNIQUE(camera, track_id), and the
hourly counts for the row (tracks_finished, its outcome, agreements) are
added in the same transaction only when the row was inserted, so a retry or
a second finish never writes a second row or counts twice. Like the hourly
summary, the write runs straight through on the TrackedObjectProcessor
thread (never in a detector process).
"""

from __future__ import annotations

import json
import logging
import time
from statistics import median
from typing import Any

from frigate.secondlook import hourly
from frigate.secondlook.clusters import SIZE_EDGES, SIZE_LABELS
from frigate.secondlook.factor import adjusted_median_scores
from frigate.secondlook.save_and_review import (
    SaveAndReviewStore,
    get_save_and_review_store,
)

logger = logging.getLogger(__name__)

OUTCOMES = ("result", "no_hit", "fail_open", "timeout", "dropped")
LIVE = ("agree", "false_positive", "false_negative", "misclass", "none")

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY,
    camera TEXT NOT NULL,
    track_id TEXT NOT NULL,
    label TEXT NOT NULL,
    event_id TEXT,
    start_time REAL,
    end_time REAL NOT NULL,
    outcome TEXT NOT NULL,
    answers INTEGER NOT NULL DEFAULT 0,
    factor REAL,
    factor_min REAL,
    factor_max REAL,
    applied_by TEXT,
    sl_score REAL,
    primary_score REAL,
    raw_median REAL,
    scaled_median REAL,
    threshold REAL,
    size_bucket INTEGER,
    box TEXT,
    live TEXT NOT NULL,
    live_outcome TEXT,
    misclass_label TEXT,
    hold_expired INTEGER NOT NULL DEFAULT 0,
    answer_after_end INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL,
    sl_model_md5 TEXT,
    sl_model_name TEXT,
    primary_model_id TEXT,
    written_at REAL NOT NULL,
    UNIQUE (camera, track_id)
);
CREATE INDEX IF NOT EXISTS idx_analyses_time ON analyses(end_time);
CREATE INDEX IF NOT EXISTS idx_analyses_camera_label ON analyses(camera, label, end_time);
CREATE INDEX IF NOT EXISTS idx_analyses_models ON analyses(sl_model_md5, primary_model_id);
"""

COLUMNS = (
    "camera", "track_id", "label", "event_id", "start_time", "end_time",
    "outcome", "answers", "factor", "factor_min", "factor_max", "applied_by",
    "sl_score", "primary_score", "raw_median", "scaled_median", "threshold",
    "size_bucket", "box", "live", "live_outcome", "misclass_label",
    "hold_expired", "answer_after_end", "latency_ms", "sl_model_md5",
    "sl_model_name", "primary_model_id", "written_at",
)


def size_bucket(box: list | None, width: int | None, height: int | None) -> int | None:
    """Index into clusters.SIZE_LABELS by box area / frame area (same edges)."""
    if not box or not width or not height:
        return None
    x1, y1, x2, y2 = (float(v) for v in box)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (float(width) * float(height))
    for i, edge in enumerate(SIZE_EDGES):
        if area < edge:
            return i
    return len(SIZE_EDGES)


def outcome_of(answers: dict[str, int], asked: bool) -> str | None:
    """Precedence over the answers a track received (see module doc)."""
    for kind in ("result", "no_hit", "fail_open"):
        if answers.get(kind):
            return kind
    if answers.get("dropped"):
        return "dropped"
    if asked:
        return "timeout"
    return None


def answer_kind(payload: dict[str, Any]) -> str:
    """Kind of one final (non-pending) Secondlook payload."""
    if not payload.get("analysed") and payload.get("factor") is None:
        return "dropped"  # queue-full fail-open notice: no analysis ran
    if payload.get("fail_open"):
        return "fail_open"
    # a same-class hit gives a factor (applied_by secondlook_scale); no hit
    # gives no_hit_factor (secondlook_no_hit) or none
    if payload.get("factor") is None or payload.get("applied_by") == "secondlook_no_hit":
        return "no_hit"
    return "result"


def build_row(obj: Any, now: float | None = None) -> dict[str, Any] | None:
    """The analyses row for a finished track, or None if SL was never asked."""
    answers = getattr(obj, "secondlook_answers", None) or {}
    outcome = outcome_of(answers, bool(getattr(obj, "secondlook_asked", False)))
    if outcome is None:
        return None
    now = time.time() if now is None else now
    data = obj.obj_data
    label = data.get("label") or "unknown"
    camera = obj.camera_config
    threshold = camera.objects.filters[label].threshold if label in camera.objects.filters else None
    sl = data.get("secondlook") or {}
    live_outcome = getattr(obj, "secondlook_outcome", None)
    factor_seen = getattr(obj, "secondlook_factor_max", None) is not None
    misclass = "misclass" in getattr(obj, "secondlook_counted_kinds", set())
    if misclass:
        live = "misclass"
    elif live_outcome:
        live = "false_negative" if live_outcome.startswith("fn_") else "false_positive"
    elif factor_seen:
        live = "agree"
    else:
        live = "none"
    box = getattr(obj, "save_and_review_box", None) or data.get("box")
    factor = getattr(obj, "secondlook_factor", None)
    history = getattr(obj, "primary_peak_history", None) or []
    scaled = (
        median(adjusted_median_scores(history, factor))
        if factor is not None and history
        else None
    )
    peak = getattr(obj, "primary_peak_median", None)
    sl_score = getattr(obj, "secondlook_last_sl_score", None)
    return {
        "camera": camera.name,
        "track_id": str(data.get("id")),
        "label": label,
        "event_id": None if obj.false_positive else str(data.get("id")),
        "start_time": data.get("start_time"),
        "end_time": float(data.get("end_time") or data.get("frame_time") or now),
        "outcome": outcome,
        "answers": int(sum(answers.values())),
        "factor": factor,
        "factor_min": getattr(obj, "secondlook_factor_min", None),
        "factor_max": getattr(obj, "secondlook_factor_max", None),
        "applied_by": sl.get("applied_by"),
        "sl_score": sl_score,
        # the raw median of the score buffer before detection (10 scores), at the track's peak
        "primary_score": peak if peak is not None and peak >= 0 else None,
        # raw median when the primary first crossed the threshold, else null
        "raw_median": obj.primary_crossing_median
        if getattr(obj, "primary_would_release", False)
        else None,
        "scaled_median": scaled,
        "threshold": threshold,
        "size_bucket": size_bucket(box, camera.detect.width, camera.detect.height),
        "box": json.dumps([int(v) for v in box]) if box else None,
        "live": live,
        "live_outcome": live_outcome,
        "misclass_label": sl.get("misclass_label") if misclass else None,
        "hold_expired": 1 if getattr(obj, "secondlook_hold_expired", False) else 0,
        "answer_after_end": 1 if getattr(obj, "secondlook_answer_after_end", False) else 0,
        "latency_ms": sl.get("latency_ms") if answers else None,
        "sl_model_md5": getattr(obj, "secondlook_model_md5", None),
        "sl_model_name": getattr(obj, "secondlook_model_name", None),
        "primary_model_id": getattr(obj, "primary_model_id", None),
        "written_at": now,
    }


def write(store: SaveAndReviewStore, row: dict[str, Any]) -> bool:
    """Insert the row once; add its hourly counts only if it was inserted."""
    names = ", ".join(COLUMNS)
    marks = ", ".join("?" for _ in COLUMNS)
    columns = [hourly.TRACKS_FINISHED, hourly.ANSWER_COLUMNS[row["outcome"]]]
    if row["live"] == "agree":
        columns.append(hourly.AGREEMENTS)
    with store._lock:
        conn = store._connect()
        try:
            with conn:
                inserted = conn.execute(
                    f"INSERT OR IGNORE INTO analyses ({names}) VALUES ({marks})",
                    [row[c] for c in COLUMNS],
                ).rowcount
                if inserted:
                    hourly.add_in(
                        conn, store.timezone, row["camera"], row["label"], *columns,
                        now=row["written_at"],
                        primary_model_id=row["primary_model_id"],
                        sl_model_md5=row["sl_model_md5"],
                    )
        finally:
            conn.close()
    return bool(inserted)


def record_track(obj: Any) -> bool:
    """Live hook at track finish; never breaks tracking. True if a row was added."""
    if getattr(obj, "secondlook_analysis_written", False):
        return False
    if not getattr(obj, "_secondlook_enabled_for_label", lambda: False)():
        return False
    store = get_save_and_review_store()
    if store is None:
        return False
    try:
        row = build_row(obj)
        if row is None:
            return False
        if row["sl_model_md5"] is None:
            row["sl_model_md5"] = store.sl_model_md5
        if row["sl_model_name"] is None:
            row["sl_model_name"] = store.sl_model_name
        added = write(store, row)
        obj.secondlook_analysis_written = True
        return added
    except Exception:
        logger.exception("Secondlook analyses write failed for %s", obj.obj_data.get("id"))
        return False


# ---- read side (API) -------------------------------------------------------

FACTOR_EDGES = (0.25, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0)
SCORE_BINS = 20  # 0.05 wide over 0..1


def _factor_labels() -> list[str]:
    edges = (0.0, *FACTOR_EDGES)
    labels = [f"{lo:g}–{hi:g}" for lo, hi in zip(edges, edges[1:])]
    return [*labels, f"≥{FACTOR_EDGES[-1]:g}"]


def _where(filters: dict[str, Any]) -> tuple[str, list]:
    clauses, params = [], []
    cameras = filters.get("cameras")
    if cameras is not None:
        clauses.append(f"camera IN ({','.join('?' * len(cameras)) or 'NULL'})")
        params += list(cameras)
    for key, column in (
        ("label", "label"),
        ("sl_model", "sl_model_md5"),
        ("primary_model", "primary_model_id"),
        ("outcome", "outcome"),
        ("live", "live"),
    ):
        if filters.get(key) is not None:
            clauses.append(f"{column} = ?")
            params.append(filters[key])
    if filters.get("after") is not None:
        clauses.append("end_time >= ?")
        params.append(float(filters["after"]))
    if filters.get("before") is not None:
        clauses.append("end_time < ?")
        params.append(float(filters["before"]))
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def _score_histogram(conn, column: str, where: str, params: list) -> dict:
    counts = [0] * SCORE_BINS
    missing = 0
    for bucket, n in conn.execute(
        f"SELECT CASE WHEN {column} IS NULL THEN -1 "
        f"ELSE MIN({SCORE_BINS - 1}, MAX(0, CAST({column} * {SCORE_BINS} AS INTEGER))) END, "
        f"COUNT(*) FROM analyses{where} GROUP BY 1",
        params,
    ):
        if bucket < 0:
            missing = n
        else:
            counts[bucket] = n
    return {
        "edges": [round(i / SCORE_BINS, 2) for i in range(SCORE_BINS + 1)],
        "counts": counts,
        "none": missing,
    }


def summary(store: SaveAndReviewStore, **filters: Any) -> dict[str, Any]:
    """Counts by outcome and live classification, factor and score histograms,
    and the models seen; all over the rows matching the filters."""
    where, params = _where(filters)
    factor_case = " ".join(
        f"WHEN factor < {edge!r} THEN {i}" for i, edge in enumerate(FACTOR_EDGES)
    )
    with store._lock:
        conn = store._connect()
        try:
            total = conn.execute(f"SELECT COUNT(*) FROM analyses{where}", params).fetchone()[0]
            by_outcome = dict.fromkeys(OUTCOMES, 0)
            by_outcome.update(dict(conn.execute(
                f"SELECT outcome, COUNT(*) FROM analyses{where} GROUP BY outcome", params
            ).fetchall()))
            by_live = dict.fromkeys(LIVE, 0)
            by_live.update(dict(conn.execute(
                f"SELECT live, COUNT(*) FROM analyses{where} GROUP BY live", params
            ).fetchall()))
            live_outcomes = dict(conn.execute(
                f"SELECT live_outcome, COUNT(*) FROM analyses{where} "
                f"{'AND' if where else 'WHERE'} live_outcome IS NOT NULL GROUP BY live_outcome",
                params,
            ).fetchall())
            factor_counts = [0] * (len(FACTOR_EDGES) + 1)
            factor_none = 0
            for bucket, n in conn.execute(
                f"SELECT CASE WHEN factor IS NULL THEN -1 {factor_case} "
                f"ELSE {len(FACTOR_EDGES)} END, COUNT(*) FROM analyses{where} GROUP BY 1",
                params,
            ):
                if bucket < 0:
                    factor_none = n
                else:
                    factor_counts[bucket] = n
            sizes = [0] * len(SIZE_LABELS)
            size_none = 0
            for bucket, n in conn.execute(
                f"SELECT size_bucket, COUNT(*) FROM analyses{where} GROUP BY 1", params
            ):
                if bucket is None:
                    size_none = n
                else:
                    sizes[int(bucket)] = n
            sl_scores = _score_histogram(conn, "sl_score", where, params)
            primary_scores = _score_histogram(conn, "primary_score", where, params)
            models = [
                {
                    "sl_model_md5": r[0],
                    "sl_model_name": r[1],
                    "primary_model_id": r[2],
                    "count": r[3],
                    "first": r[4],
                    "last": r[5],
                }
                for r in conn.execute(
                    "SELECT sl_model_md5, MAX(sl_model_name), primary_model_id, COUNT(*), "
                    f"MIN(end_time), MAX(end_time) FROM analyses{where} "
                    "GROUP BY sl_model_md5, primary_model_id ORDER BY MAX(end_time) DESC",
                    params,
                )
            ]
        finally:
            conn.close()
    return {
        "total": total,
        "outcomes": by_outcome,
        "live": by_live,
        "live_outcomes": live_outcomes,
        "factor_histogram": {
            "labels": _factor_labels(),
            "edges": [0.0, *FACTOR_EDGES],
            "counts": factor_counts,
            "none": factor_none,
        },
        "sl_score_histogram": sl_scores,
        "primary_score_histogram": primary_scores,
        "size_buckets": {"labels": list(SIZE_LABELS), "counts": sizes, "none": size_none},
        "models": models,
    }


def delete_older(conn, before: float) -> int:
    """Retention: rows whose track ended before ``before`` (caller's transaction)."""
    return conn.execute("DELETE FROM analyses WHERE end_time < ?", (before,)).rowcount
