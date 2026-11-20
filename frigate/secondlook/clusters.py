"""Long-term grouping of saved Secondlook decisions into fixed bins.

A bin key is: camera, error kind, outcome (acted on or not), primary class
(plus Secondlook's class for misclass), the cell of the box centre on an
8 x 6 grid of the full frame, and a size bucket by box area as a fraction of
the frame area. Bins with the same key apart from the cell are merged when
their cells touch (sides or corners), so a box that straddles a cell edge
does not split its group. All bins come from one GROUP BY over the filtered
rows; only the small bin list is merged in Python.

Boxes: the ``box`` column is the primary box in full-frame XYXY pixels (the
same values as ``evidence.primary_box`` when evidence exists). Frame size:
``evidence.width``/``evidence.height``, else the camera's detect resolution
from the config. Rows with neither (or no box) form "location unknown" groups.

Later key fields (primary score range, Secondlook factor range) plug in as one
more KeyField entry with its SQL bucket expression; the merge, group key,
drill-in filter, bulk verdicts and mask use KEY_FIELDS generically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

GRID_COLS = 8
GRID_ROWS = 6
# box area / frame area: <1%, 1-2.5%, 2.5-5%, 5-10%, >=10%
SIZE_EDGES = (0.01, 0.025, 0.05, 0.10)
SIZE_LABELS = ("<1%", "1–2.5%", "2.5–5%", "5–10%", ">10%")

OUTCOME_ACTED = "acted"
OUTCOME_NOT_ACTED = "not_acted"
OUTCOMES = (OUTCOME_ACTED, OUTCOME_NOT_ACTED)

# event_id is stored only when the object became a detection (released at
# track end). FN + event_id: rescued; FP without event_id: suppressed.
# Misclass is review only, so it has no outcome (NULL). Base columns only, so
# the plain list filter uses it without the derived columns below.
OUTCOME_SQL = (
    "(CASE kind"
    " WHEN 'false_negative' THEN CASE WHEN event_id IS NOT NULL THEN 'acted' ELSE 'not_acted' END"
    " WHEN 'false_positive' THEN CASE WHEN event_id IS NULL THEN 'acted' ELSE 'not_acted' END"
    " END)"
)


@dataclass(frozen=True)
class KeyField:
    name: str
    sql: str  # expression over the derived row (see derived_sql)


def bucket_sql(expr: str, edges: tuple[float, ...]) -> str:
    """Bucket index of expr: i when below edges[i], len(edges) above all."""
    whens = " ".join(f"WHEN {expr} < {edge!r} THEN {i}" for i, edge in enumerate(edges))
    return f"(CASE WHEN {expr} IS NULL THEN NULL {whens} ELSE {len(edges)} END)"


# Order is the order of the group key. The grid cell is not here: it is the
# merged dimension. Hooks for later (not built: the UI labels and the bucket
# edges still need deciding), one entry each:
#   KeyField("score_range", bucket_sql("primary_score", (0.5, 0.6, 0.7, 0.8, 0.9))),
#   KeyField("factor_range", bucket_sql("factor", (0.25, 0.5, 0.8, 1.2, 1.5, 2.0))),
KEY_FIELDS: tuple[KeyField, ...] = (
    KeyField("camera", "camera"),
    KeyField("kind", "kind"),
    KeyField("outcome", "outcome"),
    KeyField("label", "label"),
    KeyField("misclass_label", "(CASE WHEN kind = 'misclass' THEN misclass_label END)"),
    KeyField("size", "size"),
)
KEY_NAMES = tuple(f.name for f in KEY_FIELDS)


def _literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def frame_size_sql(frame_sizes: dict[str, tuple[int, int]], axis: int) -> str:
    """CASE camera WHEN 'cam' THEN <detect width/height> ... END (ints only)."""
    whens = " ".join(
        f"WHEN {_literal(cam)} THEN {int(size[axis])}"
        for cam, size in sorted(frame_sizes.items())
        if size and size[axis]
    )
    return f"(CASE camera {whens} END)" if whens else "NULL"


def derived_sql(frame_sizes: dict[str, tuple[int, int]], where: str = "") -> str:
    """Rows of examples matching ``where`` (plain column filters, applied to the
    table so its indexes apply) plus outcome, t, rx1..ry2 (box / frame, 0-1),
    grid_col, grid_row and size. The caller aliases it back to ``examples``.

    The first step is MATERIALIZED so each row's JSON is read once; otherwise
    SQLite inlines rx1..ry2 into every use in the cell and size expressions.
    """
    width = (
        f"COALESCE(json_extract(evidence, '$.width'), {frame_size_sql(frame_sizes, 0)})"
    )
    height = f"COALESCE(json_extract(evidence, '$.height'), {frame_size_sql(frame_sizes, 1)})"
    return (
        "WITH d AS MATERIALIZED (SELECT examples.*, "
        f"{OUTCOME_SQL} AS outcome, "
        "COALESCE(frame_time, created_at) AS t, "
        f"json_extract(box, '$[0]') * 1.0 / {width} AS rx1, "
        f"json_extract(box, '$[1]') * 1.0 / {height} AS ry1, "
        f"json_extract(box, '$[2]') * 1.0 / {width} AS rx2, "
        f"json_extract(box, '$[3]') * 1.0 / {height} AS ry2 "
        f"FROM examples{where}) "
        "SELECT d.*, "
        f"MIN({GRID_COLS - 1}, MAX(0, CAST((rx1 + rx2) / 2.0 * {GRID_COLS} AS INTEGER))) AS grid_col, "
        f"MIN({GRID_ROWS - 1}, MAX(0, CAST((ry1 + ry2) / 2.0 * {GRID_ROWS} AS INTEGER))) AS grid_row, "
        f"{bucket_sql('((rx2 - rx1) * (ry2 - ry1))', SIZE_EDGES)} AS size "
        "FROM d"
    )


def parse_group(text: str) -> dict[str, Any]:
    """Validate a group key sent back by the client (as made by merge_bins)."""
    try:
        group = json.loads(text)
    except (TypeError, json.JSONDecodeError) as e:
        raise ValueError("group must be a JSON group key") from e
    if not isinstance(group, dict) or set(group) != {*KEY_NAMES, "cells"}:
        raise ValueError("group key fields do not match")
    cells = group["cells"]
    if not isinstance(cells, list) or not all(
        isinstance(c, list)
        and len(c) == 2
        and all(isinstance(v, int) and not isinstance(v, bool) for v in c)
        and 0 <= c[0] < GRID_COLS
        and 0 <= c[1] < GRID_ROWS
        for c in cells
    ):
        raise ValueError("group cells must be [column, row] pairs on the grid")
    for name in KEY_NAMES:
        value = group[name]
        if value is not None and not isinstance(value, (str, int)):
            raise ValueError(f"group {name} must be a string, integer or null")
    return group


def group_clauses(group: dict[str, Any]) -> tuple[list[str], list[Any]]:
    """WHERE clauses (over derived_sql) selecting exactly one merged group."""
    clauses: list[str] = []
    params: list[Any] = []
    for field in KEY_FIELDS:
        value = group[field.name]
        if value is None:
            clauses.append(f"{field.sql} IS NULL")
        else:
            clauses.append(f"{field.sql} = ?")
            params.append(value)
    if group["cells"]:
        clauses.append(
            "(grid_col, grid_row) IN (VALUES "
            + ", ".join("(?, ?)" for _ in group["cells"])
            + ")"
        )
        for c, r in group["cells"]:
            params.extend([c, r])
    else:
        clauses.append("grid_col IS NULL")
    return clauses, params


def _components(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Connected components of grid cells touching by a side or a corner."""
    left = set(cells)
    out = []
    while left:
        start = left.pop()
        stack, comp = [start], [start]
        while stack:
            c, r = stack.pop()
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    n = (c + dc, r + dr)
                    if n in left:
                        left.remove(n)
                        stack.append(n)
                        comp.append(n)
        out.append(sorted(comp))
    return out


def merge_bins(bins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge touching cells with the same other key fields into groups.

    Each bin: {key fields..., grid_col, grid_row, count, unreviewed, first,
    last, rx1, ry1, rx2, ry2}. Groups come back sorted by count (most first),
    then most recent.
    """
    by_key: dict[tuple, dict[tuple[int, int] | None, dict]] = {}
    for b in bins:
        key = tuple(b[name] for name in KEY_NAMES)
        cell = (
            None if b["grid_col"] is None else (int(b["grid_col"]), int(b["grid_row"]))
        )
        by_key.setdefault(key, {})[cell] = b
    groups = []
    for key, cells in by_key.items():
        parts: list[list] = [[None]] if None in cells else []
        parts += _components({c for c in cells if c is not None})
        for part in parts:
            members = [cells[c] for c in part]
            located = [m for m in members if m["rx1"] is not None]
            group_key = {
                **dict(zip(KEY_NAMES, key)),
                "cells": [list(c) for c in part if c is not None],
            }
            groups.append(
                {
                    "key": json.dumps(group_key, separators=(",", ":"), sort_keys=True),
                    **group_key,
                    "count": sum(m["count"] for m in members),
                    # per cell, for the map of where the group's boxes sit
                    "cell_counts": [
                        [c[0], c[1], cells[c]["count"]] for c in part if c is not None
                    ],
                    "unreviewed": sum(m["unreviewed"] for m in members),
                    "first_seen": min(m["first"] for m in members),
                    "last_seen": max(m["last"] for m in members),
                    "extent": [
                        min(m["rx1"] for m in located),
                        min(m["ry1"] for m in located),
                        max(m["rx2"] for m in located),
                        max(m["ry2"] for m in located),
                    ]
                    if located
                    else None,
                }
            )
    groups.sort(key=lambda g: (-g["count"], -g["last_seen"], g["key"]))
    return groups


# --- suggested object filter mask -------------------------------------------


def _hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Convex hull (monotone chain)."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _clip(v: float) -> float:
    return min(1.0, max(0.0, v))


def mask_polygon(
    boxes: list[tuple[float, float, float, float]],
    margin: float = 0.02,
    points: str = "boxes",
) -> list[tuple[float, float]]:
    """Polygon (relative 0-1) covering every box plus a margin.

    points="boxes": convex hull of the boxes plus margin, so it covers their
    union. points="bottom": the hull around each box's bottom centre only,
    the one point Frigate tests against an object filter mask.
    """
    corners: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in boxes:
        if points == "bottom":
            x1 = x2 = (x1 + x2) / 2
            y1 = y2
        corners += [
            (_clip(x), _clip(y))
            for x in (x1 - margin, x2 + margin)
            for y in (y1 - margin, y2 + margin)
        ]
    return _hull(corners)


def format_coordinate(v: float) -> str:
    """Three decimals, but 1 as "1.0".

    Frigate 0.18.2 treats a mask as pixels when any value compares above the
    string "1.0" (util/config.py get_relative_coordinates), so "1.000" would be
    read as a pixel value and fail to parse.
    """
    text = f"{_clip(v):.3f}"
    return "1.0" if text == "1.000" else text


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    n = len(polygon)
    return (
        abs(
            sum(
                polygon[i][0] * polygon[(i + 1) % n][1]
                - polygon[(i + 1) % n][0] * polygon[i][1]
                for i in range(n)
            )
        )
        / 2
    )


def point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(polygon)
    for i in range(n):
        (x1, y1), (x2, y2) = polygon[i], polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def mask_yaml(
    camera: str, label: str, name: str, friendly: str, coordinates: str
) -> str:
    """cameras.<cam>.objects.filters.<label>.mask.<name>: 0.18 named-mask form."""
    return (
        "cameras:\n"
        f"  {camera}:\n"
        "    objects:\n"
        "      filters:\n"
        f"        {label}:\n"
        "          mask:\n"
        f"            {name}:\n"
        f"              friendly_name: {json.dumps(friendly)}\n"
        "              enabled: true\n"
        f'              coordinates: "{coordinates}"\n'
    )
