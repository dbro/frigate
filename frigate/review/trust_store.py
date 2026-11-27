"""SQLite store for Review Trust Overview, Human Feedback, Golden Clips, and Alert Timing.

Stored in a separate SQLite database (``review_trust.db`` beside Frigate's DB)
so Frigate's stock ``frigate.db`` schema remains untouched.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import threading
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

VALID_VERDICTS = ("good", "fp", "fn", "misclassified")
VALID_GOLDEN_KINDS = ("tp", "fp", "fn", "misclassified", "background")
COVERAGE_CONDITIONS = ("day", "night_ir", "rain", "wind", "snow")
# Upper bound for plausible live alert/detection timing samples (30 seconds).
# Samples above this reflect stale stationary or loitering track ages rather
# than live confirmation or detector queue latency.
MAX_VALID_TIMING_MS = 30_000.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_feedback (
    review_id TEXT PRIMARY KEY,
    camera TEXT NOT NULL,
    severity TEXT NOT NULL,
    verdict TEXT NOT NULL,
    corrected_label TEXT,
    reason TEXT,
    start_time REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_time ON review_feedback(start_time);
CREATE INDEX IF NOT EXISTS idx_feedback_camera_time ON review_feedback(camera, start_time);

CREATE TABLE IF NOT EXISTS golden_clips (
    id TEXT PRIMARY KEY,
    review_id TEXT,
    camera TEXT NOT NULL,
    severity TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL,
    kind TEXT NOT NULL,
    object_label TEXT,
    condition_tags TEXT,
    export_id TEXT,
    video_path TEXT,
    sidecar_path TEXT NOT NULL,
    notes TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_golden_time ON golden_clips(start_time);
CREATE INDEX IF NOT EXISTS idx_golden_camera_time ON golden_clips(camera, start_time);
CREATE INDEX IF NOT EXISTS idx_golden_kind ON golden_clips(kind);

CREATE TABLE IF NOT EXISTS alert_timings (
    review_id TEXT PRIMARY KEY,
    camera TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'alert',
    alert_time REAL NOT NULL,
    first_visible_time REAL,
    first_detected_time REAL,
    track_start_time REAL,
    confirmed_time REAL,
    secondlook_request_time REAL,
    secondlook_answer_time REAL,
    discovery_ms REAL NOT NULL DEFAULT 0.0,
    tracking_ms REAL NOT NULL DEFAULT 0.0,
    resource_busy_ms REAL NOT NULL DEFAULT 0.0,
    confirmation_ms REAL NOT NULL DEFAULT 0.0,
    zone_delay_ms REAL NOT NULL DEFAULT 0.0,
    pipeline_lag_ms REAL NOT NULL DEFAULT 0.0,
    total_ms REAL NOT NULL DEFAULT 0.0,
    reverse_norfair_status TEXT NOT NULL DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_timings_time ON alert_timings(alert_time);
CREATE INDEX IF NOT EXISTS idx_timings_camera_time ON alert_timings(camera, alert_time);
"""

_ALERT_TIMINGS_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("severity", "TEXT NOT NULL DEFAULT 'alert'"),
    ("first_visible_time", "REAL"),
    ("first_detected_time", "REAL"),
    ("track_start_time", "REAL"),
    ("confirmed_time", "REAL"),
    ("secondlook_request_time", "REAL"),
    ("secondlook_answer_time", "REAL"),
    ("tracking_ms", "REAL NOT NULL DEFAULT 0.0"),
    ("zone_delay_ms", "REAL NOT NULL DEFAULT 0.0"),
    ("pipeline_lag_ms", "REAL NOT NULL DEFAULT 0.0"),
    ("reverse_norfair_status", "TEXT NOT NULL DEFAULT 'pending'"),
)


def _parse_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _feedback_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "review_id": row["review_id"],
        "camera": row["camera"],
        "severity": row["severity"],
        "verdict": row["verdict"],
        "corrected_label": row["corrected_label"],
        "reason": row["reason"],
        "start_time": float(row["start_time"]),
        "updated_at": float(row["updated_at"]),
    }


def _golden_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "review_id": row["review_id"],
        "camera": row["camera"],
        "severity": row["severity"],
        "start_time": float(row["start_time"]),
        "end_time": float(row["end_time"]) if row["end_time"] is not None else None,
        "kind": row["kind"],
        "object_label": row["object_label"],
        "condition_tags": _parse_json_list(row["condition_tags"]),
        "export_id": row["export_id"],
        "video_path": row["video_path"],
        "sidecar_path": row["sidecar_path"],
        "notes": row["notes"],
        "created_at": float(row["created_at"]),
    }


def _timing_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    keys = set(row.keys())
    severity = row["severity"] if "severity" in keys else "alert"
    alert_time = float(row["alert_time"])
    disc_ms = float(row["discovery_ms"])
    tracking_ms = float(row["tracking_ms"]) if "tracking_ms" in keys else 0.0
    busy_ms = float(row["resource_busy_ms"])
    conf_ms = float(row["confirmation_ms"])
    zone_ms = float(row["zone_delay_ms"]) if "zone_delay_ms" in keys else 0.0
    lag_ms = float(row["pipeline_lag_ms"]) if "pipeline_lag_ms" in keys else 0.0

    raw_confirmed = row["confirmed_time"] if "confirmed_time" in keys else None
    confirmed_time = (
        float(raw_confirmed)
        if raw_confirmed is not None
        else max(0.0, alert_time - (zone_ms / 1000.0))
    )
    raw_track_start = row["track_start_time"] if "track_start_time" in keys else None
    track_start_time = (
        float(raw_track_start)
        if raw_track_start is not None
        else max(0.0, confirmed_time - (conf_ms / 1000.0))
    )
    raw_first_det = (
        row["first_detected_time"] if "first_detected_time" in keys else None
    )
    first_detected_time = (
        float(raw_first_det)
        if raw_first_det is not None
        else max(0.0, track_start_time - (tracking_ms / 1000.0))
    )
    raw_first_vis = row["first_visible_time"] if "first_visible_time" in keys else None
    first_visible_time = (
        float(raw_first_vis)
        if raw_first_vis is not None
        else max(0.0, first_detected_time - (disc_ms / 1000.0))
    )
    raw_sl_req = (
        row["secondlook_request_time"] if "secondlook_request_time" in keys else None
    )
    raw_sl_ans = (
        row["secondlook_answer_time"] if "secondlook_answer_time" in keys else None
    )
    rn_status = (
        str(row["reverse_norfair_status"])
        if "reverse_norfair_status" in keys and row["reverse_norfair_status"]
        else "pending"
    )
    return {
        "review_id": row["review_id"],
        "camera": row["camera"],
        "severity": severity,
        "alert_time": alert_time,
        "first_visible_time": first_visible_time,
        "first_detected_time": first_detected_time,
        "track_start_time": track_start_time,
        "confirmed_time": confirmed_time,
        "secondlook_request_time": (
            float(raw_sl_req) if raw_sl_req is not None else None
        ),
        "secondlook_answer_time": (
            float(raw_sl_ans) if raw_sl_ans is not None else None
        ),
        "discovery_ms": disc_ms,
        "tracking_ms": tracking_ms,
        "resource_busy_ms": busy_ms,
        "confirmation_ms": conf_ms,
        "zone_delay_ms": zone_ms,
        "pipeline_lag_ms": lag_ms,
        "total_ms": float(row["total_ms"]),
        "reverse_norfair_status": rn_status,
    }


class ReviewTrustStore:
    """Thread-safe SQLite store for human feedback, golden clips, and alert timings."""

    def __init__(self, db_path: str, exports_dir: str) -> None:
        self.db_path = db_path
        self.exports_dir = exports_dir
        self.golden_clips_dir = os.path.join(exports_dir, "golden_clips")
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        os.makedirs(self.golden_clips_dir, exist_ok=True)
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)
            cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(alert_timings)").fetchall()
            }
            migrated = False
            for col_name, col_def in _ALERT_TIMINGS_MIGRATIONS:
                if col_name not in cols:
                    conn.execute(
                        f"ALTER TABLE alert_timings ADD COLUMN {col_name} {col_def}"
                    )
                    migrated = True
            if migrated:
                conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ---- Human Feedback -----------------------------------------------------

    def set_feedback(
        self,
        review_id: str,
        camera: str,
        severity: str,
        verdict: str,
        start_time: float,
        corrected_label: str | None = None,
        reason: str | None = None,
        updated_at: float | None = None,
    ) -> dict[str, Any]:
        """Insert or update human feedback for a review segment."""
        if verdict not in VALID_VERDICTS:
            raise ValueError(
                f"verdict must be one of {', '.join(VALID_VERDICTS)}; got {verdict!r}"
            )
        now = time.time() if updated_at is None else float(updated_at)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                    INSERT INTO review_feedback (
                        review_id, camera, severity, verdict, corrected_label,
                        reason, start_time, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(review_id) DO UPDATE SET
                        camera = excluded.camera,
                        severity = excluded.severity,
                        verdict = excluded.verdict,
                        corrected_label = excluded.corrected_label,
                        reason = excluded.reason,
                        start_time = excluded.start_time,
                        updated_at = excluded.updated_at
                    """,
                (
                    review_id,
                    camera,
                    severity,
                    verdict,
                    corrected_label,
                    reason,
                    float(start_time),
                    now,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM review_feedback WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        assert row is not None
        return _feedback_row_to_dict(row)

    def clear_feedback(self, review_id: str) -> bool:
        """Delete human feedback for a review segment. Returns True if a row was removed."""
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM review_feedback WHERE review_id = ?",
                (review_id,),
            )
            conn.commit()
            return cur.rowcount > 0

    def get_feedback_map(
        self,
        start_time: float,
        end_time: float,
        cameras: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return {review_id: feedback_dict} for segments starting in [start_time, end_time]."""
        clauses = ["start_time >= ?", "start_time <= ?"]
        params: list[Any] = [float(start_time), float(end_time)]
        if cameras is not None:
            if not cameras:
                return {}
            clauses.append(f"camera IN ({','.join('?' * len(cameras))})")
            params.extend(cameras)

        sql = f"SELECT * FROM review_feedback WHERE {' AND '.join(clauses)}"
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return {row["review_id"]: _feedback_row_to_dict(row) for row in rows}

    def get_feedback_for_ids(self, review_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return {review_id: feedback_dict} for the requested review IDs."""
        if not review_ids:
            return {}
        out: dict[str, dict[str, Any]] = {}
        with self._lock, self._connect() as conn:
            for i in range(0, len(review_ids), 500):
                chunk = review_ids[i : i + 500]
                rows = conn.execute(
                    f"SELECT * FROM review_feedback WHERE review_id IN ({','.join('?' * len(chunk))})",
                    chunk,
                ).fetchall()
                for row in rows:
                    out[row["review_id"]] = _feedback_row_to_dict(row)
        return out

    def count_unreviewed_fn(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        cameras: list[str] | None = None,
    ) -> int:
        """Count human-flagged false negatives (placeholder until paired with reviewed flag)."""
        clauses = ["verdict = 'fn'"]
        params: list[Any] = []
        if start_time is not None:
            clauses.append("start_time >= ?")
            params.append(float(start_time))
        if end_time is not None:
            clauses.append("start_time <= ?")
            params.append(float(end_time))
        if cameras is not None:
            if not cameras:
                return 0
            clauses.append(f"camera IN ({','.join('?' * len(cameras))})")
            params.extend(cameras)
        sql = f"SELECT COUNT(*) FROM review_feedback WHERE {' AND '.join(clauses)}"
        with self._lock, self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row[0]) if row else 0

    # ---- Alert & Detection Timings -----------------------------------------

    def record_alert_timing(
        self,
        review_id: str,
        camera: str,
        alert_time: float,
        discovery_ms: float = 0.0,
        resource_busy_ms: float = 0.0,
        confirmation_ms: float = 0.0,
        total_ms: float | None = None,
        severity: str = "alert",
        first_visible_time: float | None = None,
        first_detected_time: float | None = None,
        track_start_time: float | None = None,
        confirmed_time: float | None = None,
        secondlook_request_time: float | None = None,
        secondlook_answer_time: float | None = None,
        tracking_ms: float | None = None,
        zone_delay_ms: float = 0.0,
        pipeline_lag_ms: float = 0.0,
        reverse_norfair_status: str = "pending",
    ) -> dict[str, Any]:
        """Record pure timestamp-difference latency breakdown for a segment or suppressed FP."""
        sev = (
            severity
            if severity in ("alert", "detection", "suppressed_fp")
            else "alert"
        )
        disc = max(0.0, float(discovery_ms))
        trk = (
            max(0.0, float(tracking_ms))
            if tracking_ms is not None
            else 0.0
        )
        busy = max(0.0, float(resource_busy_ms))
        conf = max(0.0, float(confirmation_ms))
        zone = 0.0 if sev != "alert" else max(0.0, float(zone_delay_ms))
        lag = max(0.0, float(pipeline_lag_ms))

        alert_ts = float(alert_time)
        resolved_confirmed = (
            float(confirmed_time)
            if confirmed_time is not None
            else max(0.0, alert_ts - (zone / 1000.0))
        )
        resolved_track_start = (
            float(track_start_time)
            if track_start_time is not None
            else max(0.0, resolved_confirmed - (conf / 1000.0))
        )
        resolved_first_det = (
            float(first_detected_time)
            if first_detected_time is not None
            else max(0.0, resolved_track_start - (trk / 1000.0))
        )
        resolved_first_vis = (
            float(first_visible_time)
            if first_visible_time is not None
            else max(0.0, resolved_first_det - (disc / 1000.0))
        )
        # Recompute pure telescoping total from consecutive stage delays:
        # discovery (t1 - t0) + tracking (t2 - t1) + confirmation (t4 - t2) + zone (t5 - t4)
        # If caller explicitly passed legacy resource_busy_ms without tracking_ms/total_ms,
        # keep backward-compatible total calculation.
        if total_ms is not None:
            total = max(0.0, float(total_ms))
        elif tracking_ms is not None:
            total = disc + trk + conf + zone
        else:
            total = disc + busy + conf + zone

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                    INSERT INTO alert_timings (
                        review_id, camera, severity, alert_time,
                        first_visible_time, first_detected_time,
                        track_start_time, confirmed_time,
                        secondlook_request_time, secondlook_answer_time,
                        discovery_ms, tracking_ms, resource_busy_ms,
                        confirmation_ms, zone_delay_ms, pipeline_lag_ms,
                        total_ms, reverse_norfair_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(review_id) DO UPDATE SET
                        camera = excluded.camera,
                        severity = excluded.severity,
                        alert_time = excluded.alert_time,
                        first_visible_time = COALESCE(
                            CASE
                                WHEN alert_timings.reverse_norfair_status = 'done'
                                THEN alert_timings.first_visible_time
                                ELSE excluded.first_visible_time
                            END,
                            excluded.first_visible_time
                        ),
                        first_detected_time = excluded.first_detected_time,
                        track_start_time = excluded.track_start_time,
                        confirmed_time = excluded.confirmed_time,
                        secondlook_request_time = excluded.secondlook_request_time,
                        secondlook_answer_time = excluded.secondlook_answer_time,
                        discovery_ms = CASE
                            WHEN alert_timings.reverse_norfair_status = 'done'
                            THEN alert_timings.discovery_ms
                            ELSE excluded.discovery_ms
                        END,
                        tracking_ms = excluded.tracking_ms,
                        resource_busy_ms = excluded.resource_busy_ms,
                        confirmation_ms = excluded.confirmation_ms,
                        zone_delay_ms = excluded.zone_delay_ms,
                        pipeline_lag_ms = excluded.pipeline_lag_ms,
                        total_ms = CASE
                            WHEN alert_timings.reverse_norfair_status = 'done'
                            THEN alert_timings.discovery_ms + excluded.tracking_ms + excluded.confirmation_ms + excluded.zone_delay_ms
                            ELSE excluded.total_ms
                        END,
                        reverse_norfair_status = CASE
                            WHEN alert_timings.reverse_norfair_status = 'done'
                            THEN 'done'
                            ELSE excluded.reverse_norfair_status
                        END
                    """,
                (
                    review_id,
                    camera,
                    sev,
                    alert_ts,
                    resolved_first_vis,
                    resolved_first_det,
                    resolved_track_start,
                    resolved_confirmed,
                    (
                        float(secondlook_request_time)
                        if secondlook_request_time is not None
                        else None
                    ),
                    (
                        float(secondlook_answer_time)
                        if secondlook_answer_time is not None
                        else None
                    ),
                    disc,
                    trk,
                    busy,
                    conf,
                    zone,
                    lag,
                    total,
                    reverse_norfair_status,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM alert_timings WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        assert row is not None
        return _timing_row_to_dict(row)

    def update_discovery_timing(
        self,
        review_id: str,
        *,
        first_visible_time: float,
        discovery_ms: float,
        status: str = "done",
    ) -> dict[str, Any] | None:
        """Update Reverse Norfair discovery result on an existing timing row."""
        disc = max(0.0, float(discovery_ms))
        vis = float(first_visible_time)
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM alert_timings WHERE review_id = ?",
                (review_id,),
            ).fetchone()
            if existing is None:
                return None
            trk = float(existing["tracking_ms"])
            conf = float(existing["confirmation_ms"])
            zone = float(existing["zone_delay_ms"])
            new_total = disc + trk + conf + zone
            conn.execute(
                """
                UPDATE alert_timings
                SET first_visible_time = ?,
                    discovery_ms = ?,
                    total_ms = ?,
                    reverse_norfair_status = ?
                WHERE review_id = ?
                """,
                (vis, disc, new_total, status, review_id),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM alert_timings WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        return _timing_row_to_dict(row) if row else None

    def get_timing_for_review(self, review_id: str) -> dict[str, Any] | None:
        """Fetch a single timing row by review_id, or None if not recorded."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM alert_timings WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        return _timing_row_to_dict(row) if row else None

    def get_timings_for_ids(self, review_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Return {review_id: timing_dict} for the requested review IDs."""
        if not review_ids:
            return {}
        out: dict[str, dict[str, Any]] = {}
        with self._lock, self._connect() as conn:
            for i in range(0, len(review_ids), 500):
                chunk = review_ids[i : i + 500]
                rows = conn.execute(
                    f"SELECT * FROM alert_timings WHERE review_id IN ({','.join('?' * len(chunk))})",
                    chunk,
                ).fetchall()
                for row in rows:
                    out[row["review_id"]] = _timing_row_to_dict(row)
        return out

    def get_timings_in_range(
        self,
        start_time: float,
        end_time: float,
        cameras: list[str] | None = None,
        severity: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return timing rows within [start_time, end_time], optionally filtered by severity."""
        clauses = ["alert_time >= ?", "alert_time <= ?"]
        params: list[Any] = [float(start_time), float(end_time)]
        if cameras is not None:
            if not cameras:
                return []
            clauses.append(f"camera IN ({','.join('?' * len(cameras))})")
            params.extend(cameras)
        if severity is not None:
            clauses.append("severity = ?")
            params.append(severity)

        sql = (
            f"SELECT * FROM alert_timings WHERE {' AND '.join(clauses)} "
            "ORDER BY alert_time ASC"
        )
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_timing_row_to_dict(row) for row in rows]

    # ---- Golden Clips -------------------------------------------------------

    def create_golden_clip(
        self,
        *,
        camera: str,
        severity: str,
        start_time: float,
        end_time: float | None = None,
        kind: str = "tp",
        review_id: str | None = None,
        object_label: str | None = None,
        condition_tags: list[str] | None = None,
        export_id: str | None = None,
        video_path: str | None = None,
        source_video_path: str | None = None,
        notes: str | None = None,
        clip_id: str | None = None,
        created_at: float | None = None,
    ) -> dict[str, Any]:
        """Persist a Golden Clip row and write its paired ``<id>.json`` sidecar."""
        if kind not in VALID_GOLDEN_KINDS:
            raise ValueError(
                f"kind must be one of {', '.join(VALID_GOLDEN_KINDS)}; got {kind!r}"
            )
        cid = clip_id or f"gc_{uuid.uuid4().hex[:12]}"
        now = time.time() if created_at is None else float(created_at)
        tags = [str(t).strip() for t in (condition_tags or []) if str(t).strip()]

        os.makedirs(self.golden_clips_dir, exist_ok=True)
        resolved_video_path = video_path
        if source_video_path and os.path.isfile(source_video_path):
            ext = os.path.splitext(source_video_path)[1] or ".mp4"
            dest_video_path = os.path.join(self.golden_clips_dir, f"{cid}{ext}")
            if os.path.abspath(source_video_path) != os.path.abspath(dest_video_path):
                shutil.copy2(source_video_path, dest_video_path)
            resolved_video_path = dest_video_path

        sidecar_path = os.path.join(self.golden_clips_dir, f"{cid}.json")
        sidecar_payload = {
            "id": cid,
            "review_id": review_id,
            "camera": camera,
            "severity": severity,
            "start_time": float(start_time),
            "end_time": float(end_time) if end_time is not None else None,
            "kind": kind,
            "object_label": object_label,
            "condition_tags": tags,
            "export_id": export_id,
            "video_path": resolved_video_path,
            "notes": notes,
            "created_at": now,
        }
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(sidecar_payload, f, indent=2, sort_keys=True)

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                    INSERT INTO golden_clips (
                        id, review_id, camera, severity, start_time, end_time,
                        kind, object_label, condition_tags, export_id,
                        video_path, sidecar_path, notes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                (
                    cid,
                    review_id,
                    camera,
                    severity,
                    float(start_time),
                    float(end_time) if end_time is not None else None,
                    kind,
                    object_label,
                    json.dumps(tags),
                    export_id,
                    resolved_video_path,
                    sidecar_path,
                    notes,
                    now,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM golden_clips WHERE id = ?", (cid,)
            ).fetchone()
        assert row is not None
        return _golden_row_to_dict(row)

    def get_golden_clip(self, clip_id: str) -> dict[str, Any] | None:
        """Fetch a single golden clip record by ID."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM golden_clips WHERE id = ?", (clip_id,)
            ).fetchone()
        return _golden_row_to_dict(row) if row else None

    def list_golden_clips(
        self,
        start_time: float | None = None,
        end_time: float | None = None,
        cameras: list[str] | None = None,
        severity: str | None = None,
    ) -> list[dict[str, Any]]:
        """List golden clips optionally filtered by time window, cameras, and severity."""
        clauses: list[str] = []
        params: list[Any] = []
        if start_time is not None:
            clauses.append("start_time >= ?")
            params.append(float(start_time))
        if end_time is not None:
            clauses.append("start_time <= ?")
            params.append(float(end_time))
        if cameras is not None:
            if not cameras:
                return []
            clauses.append(f"camera IN ({','.join('?' * len(cameras))})")
            params.extend(cameras)
        if severity is not None:
            clauses.append("severity = ?")
            params.append(severity)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM golden_clips{where} ORDER BY start_time DESC, created_at DESC"
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_golden_row_to_dict(row) for row in rows]

    def delete_golden_clip(self, clip_id: str) -> bool:
        """Delete a golden clip row and remove its sidecar (and copied clip inside golden_clips_dir)."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT sidecar_path, video_path FROM golden_clips WHERE id = ?",
                (clip_id,),
            ).fetchone()
            if row is None:
                return False
            conn.execute("DELETE FROM golden_clips WHERE id = ?", (clip_id,))
            conn.commit()

        sidecar_path = row["sidecar_path"]
        video_path = row["video_path"]
        if sidecar_path:
            try:
                os.remove(sidecar_path)
            except OSError:
                pass
        if video_path and os.path.abspath(video_path).startswith(
            os.path.abspath(self.golden_clips_dir) + os.sep
        ):
            try:
                os.remove(video_path)
            except OSError:
                pass
        return True

    def get_golden_summary(
        self,
        cameras: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return total count, condition_tags coverage counts, and clips list."""
        clips = self.list_golden_clips(cameras=cameras)
        coverage = {tag: 0 for tag in COVERAGE_CONDITIONS}
        for clip in clips:
            clip_tags = set(clip.get("condition_tags") or [])
            for tag in COVERAGE_CONDITIONS:
                if tag in clip_tags:
                    coverage[tag] += 1
        return {
            "total": len(clips),
            "coverage": coverage,
            "clips": clips,
        }


# Process-wide store singleton (mirrors Secondlook's SaveAndReviewStore pattern)
_trust_store: ReviewTrustStore | None = None
_trust_store_lock = threading.Lock()


def init_review_trust_store(db_path: str, exports_dir: str) -> ReviewTrustStore:
    """Initialize and register the process-wide ReviewTrustStore."""
    global _trust_store
    with _trust_store_lock:
        _trust_store = ReviewTrustStore(db_path, exports_dir)
        return _trust_store


def get_review_trust_store() -> ReviewTrustStore | None:
    """Return the process-wide ReviewTrustStore, or None if not initialized."""
    return _trust_store


def default_trust_db_path(frigate_db_path: str) -> str:
    """Derive ``review_trust.db`` beside Frigate's configured database path."""
    db_dir = os.path.dirname(frigate_db_path) or "."
    return os.path.join(db_dir, "review_trust.db")
