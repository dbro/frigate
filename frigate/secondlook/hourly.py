"""Hourly Secondlook summary, persisted in the save-and-review DB.

One row per (local hour, camera, label, primary model MD5, Secondlook model
MD5) in ``hourly_counts``, so hours after a model swap do not mix models:

- objects_analysed: each tracked object once, on its first Secondlook answer
  from a real analysis request (not a queue-full fail-open).
- false_positive / false_negative / misclass disagreements, once per track
  and kind (as count_correction), with FN split into rescued / not rescued and
  FP into suppressed / not suppressed (counted with the total when the
  outcome is final, at track end).
- tracks_finished and its parts by answer outcome (results, no_hit, timeouts,
  dropped, fail_open; they sum to tracks_finished) plus agreements: written
  in the same transaction as the track's ``analyses`` row (secondlook/
  analyses.py), so they reconcile with that table exactly until retention
  removes rows from one of them.

Written straight through by the TrackedObjectProcessor thread (one small
UPSERT per event), so nothing is lost on restart or crash. Only when
save_and_review is on.

``hourly_summary`` (no model dimension, unique key hour/camera/label) is the
table earlier builds wrote. Its rows are copied once into hourly_counts with
empty model ids ("unknown") and it is then left as it was and no longer
written.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from frigate.secondlook.metrics import ACCURACY_KINDS, OUTCOMES
from frigate.secondlook.save_and_review import (
    SaveAndReviewStore,
    get_save_and_review_store,
)

logger = logging.getLogger(__name__)

ANALYSED = "objects_analysed"
TRACKS_FINISHED = "tracks_finished"
AGREEMENTS = "agreements"
# per analyses.outcome; they sum to tracks_finished
ANSWER_COLUMNS = {
    "result": "results",
    "no_hit": "no_hit",
    "timeout": "timeouts",
    "dropped": "dropped",
    "fail_open": "fail_open",
}
LEGACY_COLUMNS = (ANALYSED, *ACCURACY_KINDS, *OUTCOMES)
COLUMNS = (
    *LEGACY_COLUMNS,
    TRACKS_FINISHED,
    AGREEMENTS,
    *ANSWER_COLUMNS.values(),
)
HOURS = 168
# NOT NULL key parts: "" = unknown (NULL would never conflict in the UPSERT)
UNKNOWN_MODEL = ""

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS hourly_counts ("
    "hour_start INTEGER NOT NULL, local_hour TEXT NOT NULL, "
    "camera TEXT NOT NULL, label TEXT NOT NULL, "
    "primary_model_id TEXT NOT NULL DEFAULT '', "
    "sl_model_md5 TEXT NOT NULL DEFAULT '', "
    + ", ".join(f"{c} INTEGER NOT NULL DEFAULT 0" for c in COLUMNS)
    + ", PRIMARY KEY (hour_start, camera, label, primary_model_id, sl_model_md5))"
)


def _zone(name: str | None):
    if not name:
        return None  # astimezone(None) = container local time, DST aware
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning("Secondlook hourly: unknown time zone %r, using local", name)
        return None


def hour_bucket(ts: float, zone_name: str | None) -> tuple[int, str]:
    """Start of the local clock hour holding ts: (epoch seconds, ISO with offset).

    Epoch keys keep the repeated hour at a DST fall-back as two buckets.
    """
    local = datetime.fromtimestamp(ts, timezone.utc).astimezone(_zone(zone_name))
    start = local.replace(minute=0, second=0, microsecond=0)
    return int(start.timestamp()), start.isoformat(timespec="minutes")


def add(
    store: SaveAndReviewStore,
    camera: str,
    label: str,
    *columns: str,
    now: float | None = None,
    primary_model_id: str | None = None,
    sl_model_md5: str | None = None,
) -> None:
    """Add 1 to each column (e.g. false_negative + fn_rescued) in one UPSERT."""
    with store._lock:
        conn = store._connect()
        try:
            with conn:
                add_in(
                    conn, store.timezone, camera, label, *columns, now=now,
                    primary_model_id=primary_model_id, sl_model_md5=sl_model_md5,
                )
        finally:
            conn.close()


def add_in(
    conn,
    zone: str | None,
    camera: str,
    label: str,
    *columns: str,
    now: float | None = None,
    primary_model_id: str | None = None,
    sl_model_md5: str | None = None,
) -> None:
    """add() inside the caller's transaction (analyses writes its row with it)."""
    if not columns or any(c not in COLUMNS for c in columns):
        raise ValueError(columns)
    hour_start, local_hour = hour_bucket(time.time() if now is None else now, zone)
    names = ", ".join(columns)
    ones = ", ".join("1" for _ in columns)
    sets = ", ".join(f"{c} = {c} + 1" for c in columns)
    conn.execute(
        f"""INSERT INTO hourly_counts
            (hour_start, local_hour, camera, label, primary_model_id,
             sl_model_md5, {names})
            VALUES (?, ?, ?, ?, ?, ?, {ones})
            ON CONFLICT (hour_start, camera, label, primary_model_id, sl_model_md5)
            DO UPDATE SET {sets}""",
        (
            hour_start,
            local_hour,
            camera,
            label,
            primary_model_id or UNKNOWN_MODEL,
            sl_model_md5 or UNKNOWN_MODEL,
        ),
    )


def migrate(conn) -> None:
    """Create hourly_counts; copy hourly_summary into it once (models unknown).

    A marker in secondlook_meta makes the copy one-time, so rows retention
    later deletes from hourly_counts are never copied back.
    """
    conn.execute(SCHEMA)
    columns = {r[1] for r in conn.execute("PRAGMA table_info(hourly_counts)")}
    for name in COLUMNS:
        if name not in columns:  # a later build adds a count
            conn.execute(
                f"ALTER TABLE hourly_counts ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0"
            )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS secondlook_meta (key TEXT PRIMARY KEY, value TEXT)"
    )
    if conn.execute(
        "SELECT 1 FROM secondlook_meta WHERE key = 'hourly_counts_copied'"
    ).fetchone():
        return
    names = ", ".join(LEGACY_COLUMNS)
    with conn:
        copied = conn.execute(
            f"""INSERT OR IGNORE INTO hourly_counts
                (hour_start, local_hour, camera, label, primary_model_id,
                 sl_model_md5, {names})
                SELECT hour_start, local_hour, camera, label, '', '', {names}
                FROM hourly_summary"""
        ).rowcount
        conn.execute(
            "INSERT INTO secondlook_meta (key, value) VALUES ('hourly_counts_copied', ?)",
            (str(time.time()),),
        )
    if copied:
        logger.info(
            "Secondlook hourly summary: copied %d earlier rows to hourly_counts "
            "(model unknown)",
            copied,
        )


def record(
    camera: str,
    label: str,
    *columns: str,
    primary_model_id: str | None = None,
    sl_model_md5: str | None = None,
) -> None:
    """Live hook: count if the store is open; never breaks tracking."""
    store = get_save_and_review_store()
    if store is None:
        return
    try:
        add(
            store, camera, label, *columns,
            primary_model_id=primary_model_id, sl_model_md5=sl_model_md5,
        )
    except Exception:
        logger.exception("Secondlook hourly summary write failed")


def series(
    store: SaveAndReviewStore,
    now: float | None = None,
    hours: int = HOURS,
    cameras: list[str] | None = None,
    primary_model_id: str | None = None,
    sl_model_md5: str | None = None,
) -> list[dict]:
    """The last `hours` local hours, oldest first, summed over camera and label
    (and model, unless one is given; "" selects rows with an unknown model).

    Every hour is present; hours with nothing analysed have objects_analysed 0.
    """
    current, _ = hour_bucket(time.time() if now is None else now, store.timezone)
    starts = [current - 3600 * k for k in range(hours - 1, -1, -1)]
    where = "hour_start >= ? AND hour_start <= ?"
    params: list = [starts[0], current]
    if cameras is not None:
        # no allowed cameras -> IN (NULL) matches nothing
        where += f" AND camera IN ({','.join('?' * len(cameras)) or 'NULL'})"
        params += cameras
    if primary_model_id is not None:
        where += " AND primary_model_id = ?"
        params.append(primary_model_id)
    if sl_model_md5 is not None:
        where += " AND sl_model_md5 = ?"
        params.append(sl_model_md5)
    sums = ", ".join(f"SUM({c})" for c in COLUMNS)
    with store._lock:
        conn = store._connect()
        try:
            rows = {
                r[0]: r[1:]
                for r in conn.execute(
                    f"SELECT hour_start, {sums} FROM hourly_counts "
                    f"WHERE {where} GROUP BY hour_start",
                    params,
                )
            }
        finally:
            conn.close()
    out = []
    for start in starts:
        values = rows.get(start, (0,) * len(COLUMNS))
        out.append(
            {
                "hour_start": start,
                "local_hour": hour_bucket(start, store.timezone)[1],
                **{c: int(v or 0) for c, v in zip(COLUMNS, values)},
            }
        )
    return out


def models(store: SaveAndReviewStore, cameras: list[str] | None = None) -> list[dict]:
    """Model pairs in the hourly table, newest first ("" = unknown)."""
    where, params = "", []
    if cameras is not None:
        where = f" WHERE camera IN ({','.join('?' * len(cameras)) or 'NULL'})"
        params = list(cameras)
    with store._lock:
        conn = store._connect()
        try:
            rows = conn.execute(
                "SELECT primary_model_id, sl_model_md5, MIN(hour_start), "
                f"MAX(hour_start) FROM hourly_counts{where} "
                "GROUP BY primary_model_id, sl_model_md5 ORDER BY MAX(hour_start) DESC",
                params,
            ).fetchall()
        finally:
            conn.close()
    return [
        {
            "primary_model_id": r[0],
            "sl_model_md5": r[1],
            "first_hour": r[2],
            "last_hour": r[3],
        }
        for r in rows
    ]
