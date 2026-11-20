"""Save-and-review store + fine-tune data classification for Secondlook Corrections.

Separate SQLite DB (not frigate.db). The human verdict records which model's
answer is the future training label (primary_right / secondlook_right) or
excludes the example from training (ignore). Retention: secondlook/retention.py.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from frigate.secondlook import clusters
from frigate.secondlook.clusters import OUTCOME_SQL, OUTCOMES

if TYPE_CHECKING:
    from frigate.config.classification import SecondlookSaveAndReviewConfig

logger = logging.getLogger(__name__)

KIND_FALSE_POSITIVE = "false_positive"
KIND_FALSE_NEGATIVE = "false_negative"
KIND_MISCLASS = "misclass"
KINDS = (KIND_FALSE_POSITIVE, KIND_FALSE_NEGATIVE, KIND_MISCLASS)

MARK_PRIMARY_RIGHT = "primary_right"
MARK_SECONDLOOK_RIGHT = "secondlook_right"
MARK_IGNORE = "ignore"
MARKS = (MARK_PRIMARY_RIGHT, MARK_SECONDLOOK_RIGHT, MARK_IGNORE)

# Marks written before the three-way verdict. accept meant "include this saved
# correction in training", and every saved correction is Secondlook's answer
# disagreeing with primary, so accept -> secondlook_right. reject meant
# "exclude from training" without saying primary was right, so reject -> ignore
# (never primary_right, which would invent a training label). The original value
# is kept in legacy_mark and a one-time SQLite backup is written first.
LEGACY_MARK_MAP = {"accept": MARK_SECONDLOOK_RIGHT, "reject": MARK_IGNORE}


@dataclass
class ExampleRecord:
    id: str
    track_id: str
    camera: str
    kind: str
    label: str
    misclass_label: str | None
    primary_score: float | None
    sl_score: float | None
    factor: float | None
    threshold: float | None
    raw_median: float | None
    box: list[int] | None
    sl_box: list[float] | None
    frame_time: float | None
    model_id: str | None
    primary_model_id: str | None
    still_path: str
    event_id: str | None
    has_clip: bool
    zones: list[str]
    mark: str | None
    created_at: float
    marked_at: float | None
    evidence: dict | None = None
    group_key: str | None = None
    legacy_mark: str | None = None
    # Secondlook model identity (null on rows from before it was stored;
    # model_id keeps the basename those rows have)
    sl_model_md5: str | None = None
    sl_model_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "track_id": self.track_id,
            "camera": self.camera,
            "kind": self.kind,
            "label": self.label,
            "misclass_label": self.misclass_label,
            "primary_score": self.primary_score,
            "sl_score": self.sl_score,
            "factor": self.factor,
            "threshold": self.threshold,
            "raw_median": self.raw_median,
            "box": self.box,
            "sl_box": self.sl_box,
            "frame_time": self.frame_time,
            "model_id": self.model_id,
            "primary_model_id": self.primary_model_id,
            "still_path": self.still_path,
            "event_id": self.event_id,
            "has_clip": self.has_clip,
            "zones": self.zones,
            "mark": self.mark,
            "created_at": self.created_at,
            "marked_at": self.marked_at,
            "evidence": self.evidence,
            "group_key": self.group_key,
            "legacy_mark": self.legacy_mark,
            "sl_model_md5": self.sl_model_md5,
            "sl_model_name": self.sl_model_name,
        }


def classify_example(
    *,
    raw_median: float,
    threshold: float,
    factor: float | None,
    final_false_positive: bool,
    gap_low: float = 0.8,
    gap_high: float = 1.2,
    misclass_label: str | None = None,
    primary_label: str | None = None,
) -> str | None:
    """Return example kind or None if not a corrected decision / significant gap.

    Misclass wins when misclass_label is set and differs from primary. That
    label is only set when Secondlook found no same-class partner and a
    different-class box overlaps the primary closely.
    FP: primary would clear threshold, SL scale kept it false_positive, factor < gap_low.
    FN: primary alone under threshold, SL scale made a detection, factor > gap_high.
    """
    if misclass_label and primary_label and misclass_label != primary_label:
        return KIND_MISCLASS

    if factor is None:
        return None

    if factor < gap_low and raw_median >= threshold and final_false_positive:
        return KIND_FALSE_POSITIVE

    if factor > gap_high and raw_median < threshold and not final_false_positive:
        return KIND_FALSE_NEGATIVE

    return None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS examples (
    id TEXT PRIMARY KEY,
    track_id TEXT NOT NULL,
    camera TEXT NOT NULL,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    misclass_label TEXT,
    primary_score REAL,
    sl_score REAL,
    factor REAL,
    threshold REAL,
    raw_median REAL,
    box TEXT,
    sl_box TEXT,
    frame_time REAL,
    model_id TEXT,
    primary_model_id TEXT,
    still_path TEXT NOT NULL,
    event_id TEXT,
    has_clip INTEGER NOT NULL DEFAULT 0,
    zones TEXT,
    mark TEXT,
    created_at REAL NOT NULL,
    marked_at REAL
);
CREATE INDEX IF NOT EXISTS idx_examples_camera_time ON examples(camera, frame_time);
CREATE INDEX IF NOT EXISTS idx_examples_kind ON examples(kind);
CREATE INDEX IF NOT EXISTS idx_examples_mark ON examples(mark);
CREATE INDEX IF NOT EXISTS idx_examples_label ON examples(label);
-- Corrections order and time window (expression index; no data change)
CREATE INDEX IF NOT EXISTS idx_examples_time ON examples(COALESCE(frame_time, created_at));
-- hourly disagreement summary (secondlook/hourly.py); separate from examples
CREATE TABLE IF NOT EXISTS hourly_summary (
    hour_start INTEGER NOT NULL,
    local_hour TEXT NOT NULL,
    camera TEXT NOT NULL,
    label TEXT NOT NULL,
    objects_analysed INTEGER NOT NULL DEFAULT 0,
    false_positive INTEGER NOT NULL DEFAULT 0,
    false_negative INTEGER NOT NULL DEFAULT 0,
    misclass INTEGER NOT NULL DEFAULT 0,
    fn_rescued INTEGER NOT NULL DEFAULT 0,
    fn_not_rescued INTEGER NOT NULL DEFAULT 0,
    fp_suppressed INTEGER NOT NULL DEFAULT 0,
    fp_not_suppressed INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (hour_start, camera, label)
);
"""


class SaveAndReviewStore:
    """Thread-safe SQLite store for fine-tune examples."""

    def __init__(self, db_path: str, images_path: str) -> None:
        self.db_path = db_path
        self.images_path = images_path
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        os.makedirs(images_path, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            columns = {r[1] for r in conn.execute("PRAGMA table_info(examples)")}
            for name in (
                "evidence", "group_key", "legacy_mark", "sl_model_md5", "sl_model_name",
            ):
                if name not in columns:
                    conn.execute(f"ALTER TABLE examples ADD COLUMN {name} TEXT")
            self._migrate_legacy_marks(conn)
            self._migrate_hourly_outcomes(conn)
            # per-track analyses and the model-keyed hourly table
            from frigate.secondlook import analyses, hourly

            conn.executescript(analyses.SCHEMA)
            hourly.migrate(conn)
            # retention ages ignored rows from marked_at: give marked rows
            # without one (none expected) a full period from now
            with conn:
                conn.execute(
                    "UPDATE examples SET marked_at = ? "
                    "WHERE mark IS NOT NULL AND marked_at IS NULL",
                    (time.time(),),
                )
        self.last_retention: dict[str, Any] | None = None
        # IANA zone for hourly buckets; None = the container's local time
        self.timezone: str | None = None
        # Secondlook model for rows written without one in their answers
        # (set at startup from the config; see app.init_secondlook_save_and_review)
        self.sl_model_md5: str | None = None
        self.sl_model_name: str | None = None
        from collections import Counter
        self.diagnostics = Counter()

    def _migrate_hourly_outcomes(self, conn: sqlite3.Connection) -> None:
        """Add the live-outcome counts to an hourly table from before them.

        Earlier builds counted FN only when Secondlook released the object and
        FP only while it held one back, so existing totals go to fn_rescued /
        fp_suppressed; totals stay equal to the sum of their two parts.
        """
        columns = {r[1] for r in conn.execute("PRAGMA table_info(hourly_summary)")}
        added = [
            name
            for name in ("fn_rescued", "fn_not_rescued", "fp_suppressed", "fp_not_suppressed")
            if name not in columns
        ]
        if not added:
            return
        with conn:
            for name in added:
                conn.execute(
                    f"ALTER TABLE hourly_summary ADD COLUMN {name} "
                    "INTEGER NOT NULL DEFAULT 0"
                )
            if "fn_rescued" in added:
                conn.execute("UPDATE hourly_summary SET fn_rescued = false_negative")
            if "fp_suppressed" in added:
                conn.execute("UPDATE hourly_summary SET fp_suppressed = false_positive")
        logger.info(
            "Secondlook hourly summary: added %s; earlier FN/FP counted as "
            "rescued/suppressed",
            ", ".join(added),
        )

    def _migrate_legacy_marks(self, conn: sqlite3.Connection) -> None:
        legacy = conn.execute(
            "SELECT COUNT(*) FROM examples WHERE mark IN ('accept', 'reject')"
        ).fetchone()[0]
        if not legacy:
            return
        conn.commit()
        backup_path = self.db_path + ".pre-verdict.bak"
        if not os.path.exists(backup_path):
            backup = sqlite3.connect(backup_path)
            try:
                conn.backup(backup)
            finally:
                backup.close()
        with conn:
            conn.execute(
                """UPDATE examples SET legacy_mark = mark,
                       mark = CASE mark WHEN 'accept' THEN ? ELSE ? END
                   WHERE mark IN ('accept', 'reject')""",
                (MARK_SECONDLOOK_RIGHT, MARK_IGNORE),
            )
        logger.info(
            "Secondlook save_and_review: mapped %d accept/reject marks to "
            "verdicts (accept->secondlook_right, reject->ignore); originals kept "
            "in legacy_mark, backup %s",
            legacy,
            backup_path,
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def deposit(
        self,
        *,
        track_id: str,
        camera: str,
        kind: str,
        label: str,
        jpeg_bytes: bytes,
        misclass_label: str | None = None,
        primary_score: float | None = None,
        sl_score: float | None = None,
        factor: float | None = None,
        threshold: float | None = None,
        raw_median: float | None = None,
        box: list[int] | None = None,
        sl_box: list[float] | None = None,
        frame_time: float | None = None,
        model_id: str | None = None,
        primary_model_id: str | None = None,
        sl_model_md5: str | None = None,
        event_id: str | None = None,
        has_clip: bool = False,
        evidence: dict | None = None,
        zones: list[str] | None = None,
    ) -> ExampleRecord | None:
        if kind not in KINDS:
            logger.warning("Unknown example kind %s — skip deposit", kind)
            return None
        if not jpeg_bytes:
            logger.warning("No JPEG for track %s — skip deposit", track_id)
            return None

        example_id = str(uuid.uuid4())
        cam_dir = os.path.join(self.images_path, camera)
        os.makedirs(cam_dir, exist_ok=True)
        still_path = os.path.join(cam_dir, f"{example_id}.jpg")
        with open(still_path, "wb") as f:
            f.write(jpeg_bytes)

        created_at = time.time()
        zones = zones or []
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO examples (
                        id, track_id, camera, kind, label, misclass_label,
                        primary_score, sl_score, factor, threshold, raw_median,
                        box, sl_box, frame_time, model_id, primary_model_id,
                        still_path, event_id, has_clip, zones, mark, created_at, marked_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL)
                    """,
                    (
                        example_id,
                        track_id,
                        camera,
                        kind,
                        label,
                        misclass_label,
                        primary_score,
                        sl_score,
                        factor,
                        threshold,
                        raw_median,
                        json.dumps(box) if box is not None else None,
                        json.dumps(sl_box) if sl_box is not None else None,
                        frame_time,
                        model_id,
                        primary_model_id,
                        still_path,
                        event_id,
                        1 if has_clip else 0,
                        json.dumps(zones),
                        created_at,
                    ),
                )
                from frigate.secondlook.evidence import group_key
                conn.execute(
                    "UPDATE examples SET evidence=?, group_key=?, sl_model_md5=?, "
                    "sl_model_name=? WHERE id=?", (
                    json.dumps(evidence) if evidence else None,
                    group_key(jpeg_bytes, camera, kind, label, misclass_label, model_id, primary_model_id,
                              [box, sl_box, primary_score, sl_score, factor]),
                    sl_model_md5,
                    model_id,
                    example_id,
                ))
                conn.commit()

        logger.info(
            "Secondlook example deposited id=%s kind=%s camera=%s track=%s factor=%s",
            example_id,
            kind,
            camera,
            track_id,
            factor,
        )
        return self.get(example_id)

    def get(self, example_id: str) -> ExampleRecord | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM examples WHERE id = ?", (example_id,)
                ).fetchone()
        return _row_to_record(row) if row else None

    def set_mark(self, example_id: str, mark: str | None) -> ExampleRecord | None:
        if mark is not None and mark not in MARKS:
            raise ValueError(
                f"mark must be one of {', '.join(MARKS)} or null; got {mark!r}"
            )
        marked_at = time.time() if mark is not None else None
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "UPDATE examples SET mark = ?, marked_at = ? WHERE id = ?",
                    (mark, marked_at, example_id),
                )
                conn.commit()
                if cur.rowcount == 0:
                    return None
        return self.get(example_id)

    def _where(
        self,
        *,
        cameras: list[str] | None = None,
        after: float | None = None,
        before: float | None = None,
        labels: list[str] | None = None,
        zones: list[str] | None = None,
        kind: str | None = None,
        mark: str | None = None,
        reviewed: bool | None = None,
        hide_edge_small: tuple[int, float] | None = None,
        outcome: str | None = None,
        group: dict[str, Any] | None = None,
        frame_sizes: dict[str, tuple[int, int]] | None = None,
    ) -> tuple[str, str, list[Any]]:
        """(FROM source, WHERE text, params) for the shared Corrections filters.

        The source is the plain table, or for a long-term group the table plus
        its derived bin columns (clusters.derived_sql) aliased back to examples.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if cameras is not None:
            if not cameras:
                clauses.append("0")
            clauses.append(f"camera IN ({','.join('?' * len(cameras))})")
            params.extend(cameras)
        if after is not None:
            clauses.append("COALESCE(frame_time, created_at) >= ?")
            params.append(after)
        if before is not None:
            clauses.append("COALESCE(frame_time, created_at) <= ?")
            params.append(before)
        if labels:
            clauses.append(f"label IN ({','.join('?' * len(labels))})")
            params.extend(labels)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if outcome is not None:
            # From the stored event_id, so it works on every row ever saved
            # (no migration). Misclass has no outcome and drops out.
            if outcome not in OUTCOMES:
                raise ValueError(f"outcome must be one of {', '.join(OUTCOMES)}")
            clauses.append(f"{OUTCOME_SQL} = ?")
            params.append(outcome)
        if mark is not None:
            clauses.append("mark = ?")
            params.append(mark)
        if reviewed is True:
            clauses.append("mark IS NOT NULL")
        elif reviewed is False:
            clauses.append("mark IS NULL")

        if zones:
            clauses.append("EXISTS (SELECT 1 FROM json_each(examples.zones) WHERE value IN (" + ",".join("?" * len(zones)) + "))")
            params.extend(zones)
        if hide_edge_small is not None:
            # Display filter only. Rows without saved frame size/box evaluate to
            # NULL and stay visible.
            margin, fraction = hide_edge_small
            # Small = width < fraction * frame width OR height < fraction *
            # frame height (default 0.05). Rows without frame size stay visible.
            clauses.append(
                """NOT COALESCE((
                    (json_extract(evidence, '$.primary_box[0]') <= ?
                     OR json_extract(evidence, '$.primary_box[1]') <= ?
                     OR json_extract(evidence, '$.primary_box[2]') >= json_extract(evidence, '$.width') - ?
                     OR json_extract(evidence, '$.primary_box[3]') >= json_extract(evidence, '$.height') - ?)
                    AND (
                      (json_extract(evidence, '$.primary_box[2]') - json_extract(evidence, '$.primary_box[0]'))
                        < ? * json_extract(evidence, '$.width')
                      OR (json_extract(evidence, '$.primary_box[3]') - json_extract(evidence, '$.primary_box[1]'))
                        < ? * json_extract(evidence, '$.height')
                    )
                ), 0)"""
            )
            params.extend([margin, margin, margin, margin, fraction, fraction])
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        if group is None:
            return "examples", where, params
        # plain filters inside (indexed), the group's derived columns outside
        group_where, group_params = clusters.group_clauses(group)
        source = f"({clusters.derived_sql(frame_sizes or {}, where)}) AS examples"
        return source, " WHERE " + " AND ".join(group_where), [*params, *group_params]

    def list(
        self,
        *,
        limit: int = 100,
        count_only: bool = False,
        offset: int = 0,
        **filters: Any,
    ) -> list[ExampleRecord]:
        source, where, params = self._where(**filters)
        if count_only:
            with self._lock, self._connect() as conn:
                return conn.execute(f"SELECT COUNT(*) FROM {source}{where}", params).fetchone()[0]
        sql = (
            f"SELECT * FROM {source}{where} "
            "ORDER BY COALESCE(frame_time, created_at) DESC, id DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()

        records = [_row_to_record(r) for r in rows]

        return records

    def groups(
        self,
        *,
        frame_sizes: dict[str, tuple[int, int]],
        samples: int = 6,
        **filters: Any,
    ) -> dict[str, Any]:
        """Long-term groups (clusters.py) over every row matching the filters.

        One GROUP BY for the bins and one ranked query for the newest samples
        per bin; touching cells are then merged in Python (a few hundred bins
        at most).
        """
        filters.pop("group", None)
        _, where, params = self._where(**filters)
        derived = f"({clusters.derived_sql(frame_sizes, where)}) AS examples"
        keys = ", ".join(f"{f.sql} AS {f.name}" for f in clusters.KEY_FIELDS)
        names = ", ".join(clusters.KEY_NAMES)
        with self._lock:
            conn = self._connect()
            try:
                # one pass over the filtered rows; bins and samples read the
                # small temp table (per connection, dropped with it)
                conn.execute(
                    f"CREATE TEMP TABLE sl_bins AS SELECT id, {keys}, grid_col, "
                    "grid_row, t, mark, rx1, ry1, rx2, ry2 "
                    f"FROM {derived}",
                    params,
                )
                bins = [
                    dict(r)
                    for r in conn.execute(
                        f"SELECT {names}, grid_col, grid_row, COUNT(*) AS count, "
                        "SUM(mark IS NULL) AS unreviewed, MIN(t) AS first, "
                        "MAX(t) AS last, MIN(rx1) AS rx1, MIN(ry1) AS ry1, "
                        "MAX(rx2) AS rx2, MAX(ry2) AS ry2 FROM sl_bins "
                        f"GROUP BY {names}, grid_col, grid_row"
                    )
                ]
                ranked = [
                    dict(r)
                    for r in conn.execute(
                        f"SELECT * FROM (SELECT id, {names}, grid_col, grid_row, t, "
                        f"ROW_NUMBER() OVER (PARTITION BY {names}, grid_col, grid_row "
                        "ORDER BY t DESC, id DESC) AS rank FROM sl_bins) "
                        "WHERE rank <= ?",
                        (samples,),
                    )
                ]
                details: dict[str, dict[str, Any]] = {}
                ids = [r["id"] for r in ranked]
                for i in range(0, len(ids), 500):
                    chunk = ids[i : i + 500]
                    for r in conn.execute(
                        "SELECT id, kind, label, misclass_label, camera, "
                        "primary_score, sl_score, factor, mark, box, evidence "
                        f"FROM examples WHERE id IN ({','.join('?' * len(chunk))})",
                        chunk,
                    ):
                        details[r["id"]] = dict(r)
            finally:
                conn.close()
        groups = clusters.merge_bins(bins)
        index: dict[tuple, dict[str, Any]] = {}
        for g in groups:
            for cell in [tuple(c) for c in g["cells"]] or [(None, None)]:
                index[(*(g[n] for n in clusters.KEY_NAMES), *cell)] = g
        picked: dict[str, list[dict[str, Any]]] = {}
        for r in ranked:
            g = index[(*(r[n] for n in clusters.KEY_NAMES), r["grid_col"], r["grid_row"])]
            picked.setdefault(g["key"], []).append({**details[r["id"]], "t": r["t"]})
        for g in groups:
            rows = sorted(picked.get(g["key"], []), key=lambda r: (-r["t"], r["id"]))[:samples]
            g["samples"] = [
                {
                    "id": r["id"],
                    "time": r["t"],
                    "kind": r["kind"],
                    "label": r["label"],
                    "misclass_label": r["misclass_label"],
                    "camera": r["camera"],
                    "primary_score": r["primary_score"],
                    "sl_score": r["sl_score"],
                    "factor": r["factor"],
                    "mark": r["mark"],
                    "box": json.loads(r["box"]) if r["box"] else None,
                    "evidence": json.loads(r["evidence"]) if r["evidence"] else None,
                }
                for r in rows
            ]
        return {"groups": groups, "rows": sum(g["count"] for g in groups), "bins": len(bins)}

    def group_boxes(
        self, *, frame_sizes: dict[str, tuple[int, int]], group: dict[str, Any], **filters: Any
    ) -> list[dict[str, Any]]:
        """Relative boxes (0-1) of every located member of one group, newest first."""
        source, where, params = self._where(group=group, frame_sizes=frame_sizes, **filters)
        sql = (
            f"SELECT id, rx1, ry1, rx2, ry2, t FROM {source}{where} "
            "AND rx1 IS NOT NULL ORDER BY t DESC, id DESC"
        )
        with self._lock, self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, params)]

    def mark_matching(self, verdict: str, **filters: Any) -> int:
        """Bulk verdict on every unreviewed row matching the filters; rows
        that already have a verdict keep it."""
        if verdict not in MARKS:
            raise ValueError(f"mark must be one of {', '.join(MARKS)}; got {verdict!r}")
        filters["reviewed"] = False
        filters.pop("mark", None)
        source, where, params = self._where(**filters)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE examples SET mark = ?, marked_at = ? "
                f"WHERE mark IS NULL AND id IN (SELECT id FROM {source}{where})",
                [verdict, time.time(), *params],
            )
            conn.commit()
            return cur.rowcount

    def count_unreviewed(self, cameras: list[str] | None = None) -> int:
        clauses = ["mark IS NULL"]
        params: list[Any] = []
        if cameras:
            clauses.append(f"camera IN ({','.join('?' * len(cameras))})")
            params.extend(cameras)
        sql = f"SELECT COUNT(*) AS c FROM examples WHERE {' AND '.join(clauses)}"
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(sql, params).fetchone()
        return int(row["c"]) if row else 0


def _row_to_record(row: sqlite3.Row) -> ExampleRecord:
    def _json_list(val: str | None) -> list:
        if not val:
            return []
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []

    keys = row.keys()
    box = _json_list(row["box"])
    sl_box = _json_list(row["sl_box"])
    zones = _json_list(row["zones"])
    grouping = row["group_key"]
    if not grouping or not grouping.startswith("v2:"):
        try:
            from frigate.secondlook.evidence import existing_image_digest, group_digest
            stat = os.stat(row["still_path"])
            grouping = group_digest(existing_image_digest(row["still_path"], stat.st_mtime_ns, stat.st_size),
                                    row["camera"], row["kind"], row["label"], row["misclass_label"],
                                    row["model_id"], row["primary_model_id"],
                                    [box or None, sl_box or None, row["primary_score"], row["sl_score"], row["factor"]])
        except OSError:
            grouping = None
    return ExampleRecord(
        id=row["id"],
        track_id=row["track_id"],
        camera=row["camera"],
        kind=row["kind"],
        label=row["label"],
        misclass_label=row["misclass_label"],
        primary_score=row["primary_score"],
        sl_score=row["sl_score"],
        factor=row["factor"],
        threshold=row["threshold"],
        raw_median=row["raw_median"],
        box=box or None,
        sl_box=sl_box or None,
        frame_time=row["frame_time"],
        model_id=row["model_id"],
        primary_model_id=row["primary_model_id"],
        still_path=row["still_path"],
        event_id=row["event_id"],
        has_clip=bool(row["has_clip"]),
        zones=zones,
        mark=row["mark"],
        created_at=row["created_at"],
        marked_at=row["marked_at"],
        evidence=json.loads(row["evidence"]) if row["evidence"] else None,
        group_key=grouping,
        legacy_mark=row["legacy_mark"],
        sl_model_md5=row["sl_model_md5"] if "sl_model_md5" in keys else None,
        sl_model_name=row["sl_model_name"] if "sl_model_name" in keys else None,
    )


# Process-wide store (lazy). Set via init_save_and_review_store from app/startup.
_store: SaveAndReviewStore | None = None
_store_lock = threading.Lock()


def init_save_and_review_store(db_path: str, images_path: str) -> SaveAndReviewStore:
    global _store
    with _store_lock:
        _store = SaveAndReviewStore(db_path, images_path)
        return _store


def get_save_and_review_store() -> SaveAndReviewStore | None:
    return _store


def configure_save_and_review(
    config: SecondlookSaveAndReviewConfig,
) -> SaveAndReviewStore | None:
    """Initialize only when enabled; disabling neither opens nor deletes saved data."""
    global _store
    if not config.enabled:
        with _store_lock:
            _store = None
        return None
    return init_save_and_review_store(config.database.path, config.images.path)
