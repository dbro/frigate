"""Retention cleanup for the Secondlook save-and-review store.

Deletes hourly summary rows older than summary_days, per-object analyses rows
whose object ended more than analyses_days ago, unreviewed decisions by
capture age, Ignore decisions by age since they
were last marked, and (optionally) the oldest of those over a size cap. Accepted
decisions (primary_right / secondlook_right) are never selected. Rows go in one
transaction, then their own JPEGs; a crash in between leaves orphan JPEGs that
the next run's sweep removes, so every run is safe to repeat.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import TYPE_CHECKING, Any

from frigate.secondlook.analyses import delete_older
from frigate.secondlook.metrics import _directory_usage, _file_bytes
from frigate.secondlook.save_and_review import MARK_IGNORE, SaveAndReviewStore

if TYPE_CHECKING:
    from frigate.config.classification import SecondlookSaveAndReviewRetentionConfig

logger = logging.getLogger(__name__)

DAY = 86400.0
GIB = 1024**3
# A file the store wrote: <images.path>/<camera>/<uuid4>.jpg
STILL_NAME = re.compile(r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.jpg$")
# deposit writes the JPEG just before its row; never sweep a fresh file
ORPHAN_MIN_AGE_SECONDS = 3600.0
_CHUNK = 500


def _select(conn, sql: str, params: tuple) -> list[tuple[str, str]]:
    return [(r["id"], r["still_path"]) for r in conn.execute(sql, params)]


def run_retention(
    store: SaveAndReviewStore,
    config: SecondlookSaveAndReviewRetentionConfig,
    now: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    unreviewed = "SELECT id, still_path FROM examples WHERE mark IS NULL"
    ignored = "SELECT id, still_path FROM examples WHERE mark = ?"
    by_capture = " ORDER BY COALESCE(frame_time, created_at), id"
    by_mark = " ORDER BY COALESCE(marked_at, created_at), id"
    with store._lock:
        conn = store._connect()
        try:
            with conn:  # one transaction for every row deleted in this run
                victims: dict[str, str] = {}
                if config.unreviewed_days:
                    victims.update(_select(
                        conn,
                        unreviewed + " AND COALESCE(frame_time, created_at) < ?",
                        (now - config.unreviewed_days * DAY,),
                    ))
                if config.ignored_days:
                    victims.update(_select(
                        conn,
                        ignored + " AND COALESCE(marked_at, created_at) < ?",
                        (MARK_IGNORE, now - config.ignored_days * DAY),
                    ))
                if config.max_size_gb:
                    excess = (
                        _file_bytes(store.db_path)
                        + _directory_usage(store.images_path)[0]
                        - config.max_size_gb * GIB
                        - sum(_file_bytes(p) for p in victims.values())
                    )
                    candidates = _select(conn, unreviewed + by_capture, ()) + _select(
                        conn, ignored + by_mark, (MARK_IGNORE,)
                    )
                    for example_id, path in candidates:
                        if excess <= 0:
                            break
                        if example_id not in victims:
                            victims[example_id] = path
                            excess -= _file_bytes(path)
                summary_rows = 0
                if config.summary_days:
                    cutoff = now - config.summary_days * DAY
                    # hourly_counts is the live table; hourly_summary holds
                    # only rows from earlier builds
                    summary_rows = conn.execute(
                        "DELETE FROM hourly_counts WHERE hour_start < ?", (cutoff,)
                    ).rowcount
                    conn.execute(
                        "DELETE FROM hourly_summary WHERE hour_start < ?", (cutoff,)
                    )
                analyses_rows = 0
                analyses_days = getattr(config, "analyses_days", None)
                if analyses_days:
                    analyses_rows = delete_older(conn, now - analyses_days * DAY)
                ids = list(victims)
                for i in range(0, len(ids), _CHUNK):
                    chunk = ids[i : i + _CHUNK]
                    conn.execute(
                        f"DELETE FROM examples WHERE id IN ({','.join('?' * len(chunk))})",
                        chunk,
                    )
            known = {r["id"] for r in conn.execute("SELECT id FROM examples")}
        finally:
            conn.close()

    deleted_files = missing_files = 0
    for example_id, path in victims.items():
        # only the row's own store file, never an arbitrary path from the DB
        if os.path.basename(path) != f"{example_id}.jpg":
            continue
        try:
            os.remove(path)
            deleted_files += 1
        except FileNotFoundError:
            missing_files += 1
        except OSError as e:
            logger.warning("Secondlook retention could not delete %s: %s", path, e)
    orphans = _sweep_orphans(store.images_path, known, now)

    result = {
        "last_run": now,
        "deleted_rows": len(victims),
        "deleted_files": deleted_files,
        "missing_files": missing_files,
        "deleted_orphans": orphans,
        "deleted_summary_rows": summary_rows,
        "deleted_analyses_rows": analyses_rows,
    }
    store.last_retention = result
    logger.info(
        "Secondlook retention: deleted %d rows, %d images (%d already missing), "
        "%d orphan images, %d hourly summary rows, %d analyses rows",
        len(victims),
        deleted_files,
        missing_files,
        orphans,
        summary_rows,
        analyses_rows,
    )
    return result


def _sweep_orphans(images_path: str, known: set[str], now: float) -> int:
    """Delete <images.path>/<camera>/<uuid>.jpg files without a row. Nothing else."""
    removed = 0
    try:
        cameras = [e for e in os.scandir(images_path) if e.is_dir(follow_symlinks=False)]
    except OSError:
        return 0
    for camera in cameras:
        try:
            entries = list(os.scandir(camera.path))
        except OSError:
            continue
        for entry in entries:
            match = STILL_NAME.match(entry.name)
            try:
                if (
                    not match
                    or match.group(1) in known
                    or not entry.is_file(follow_symlinks=False)
                    or now - entry.stat(follow_symlinks=False).st_mtime
                    < ORPHAN_MIN_AGE_SECONDS
                ):
                    continue
                os.remove(entry.path)
                removed += 1
            except FileNotFoundError:
                continue
            except OSError as e:
                logger.warning("Secondlook retention could not delete %s: %s", entry.path, e)
    return removed


def start_retention(
    store: SaveAndReviewStore,
    config: SecondlookSaveAndReviewRetentionConfig,
    stop_event: threading.Event | None = None,
) -> threading.Thread | None:
    """Run now, then daily, in a daemon thread. None when every rule is off."""
    if not (
        config.unreviewed_days
        or config.ignored_days
        or config.max_size_gb
        or config.summary_days
        or getattr(config, "analyses_days", None)
    ):
        logger.info("Secondlook retention disabled; nothing is deleted")
        return None
    stop_event = stop_event or threading.Event()

    def loop() -> None:
        while True:
            try:
                run_retention(store, config)
            except Exception:
                logger.exception("Secondlook retention run failed; retrying next run")
            if stop_event.wait(DAY):
                return

    thread = threading.Thread(target=loop, name="secondlook_retention", daemon=True)
    thread.start()
    return thread
