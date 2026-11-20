"""Secondlook Corrections — fine-tune examples API."""

from __future__ import annotations

import hashlib
import json
import logging
import os

from fastapi import APIRouter, Request
from fastapi.params import Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from frigate.api.auth import get_allowed_cameras_for_filter
from frigate.api.defs.tags import Tags
from frigate.secondlook import analyses, clusters, hourly
from frigate.secondlook.clusters import OUTCOMES
from frigate.secondlook.live_outcome import live_outcomes
from frigate.secondlook.save_and_review import (
    KINDS,
    MARKS,
    get_save_and_review_store,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.secondlook])


class MarkBody(BaseModel):
    mark: str | None = Field(
        default=None,
        description="primary_right or secondlook_right (that model's answer is the training label), ignore (exclude from training), or null (unreviewed)",
    )


def _store_or_503(request: Request):
    if not request.app.frigate_config.secondlook.save_and_review.enabled:
        return None, JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": "Secondlook save and review is disabled",
            },
        )
    store = get_save_and_review_store()
    if store is None:
        return None, JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": "Secondlook save-and-review store is not initialized",
            },
        )
    return store, None


def _with_outcomes(request: Request, rows) -> list[dict]:
    """Item dicts plus live_outcome (read-time Event lookup, not stored)."""
    outcomes = live_outcomes(request.app.frigate_config, rows)
    return [{**r.to_dict(), "live_outcome": outcomes[r.id]} for r in rows]


@router.get("/secondlook/disagreements/hourly")
def hourly_disagreements(
    request: Request,
    primary_model: str | None = None,
    sl_model: str | None = None,
    allowed_cameras: list[str] = Depends(get_allowed_cameras_for_filter),
):
    """Last 168 local hours, oldest first, summed over camera and label.

    primary_model / sl_model (MD5; "" = rows with an unknown model, i.e. from
    before models were stored) limit it to one model; by default all models
    are summed. ``models`` lists the pairs in the table.

    A dedicated endpoint, not /api/stats: stats are pushed every few seconds
    and kept as history, and the week changes only hourly.
    """
    store, err = _store_or_503(request)
    if err:
        return err
    cameras = list(allowed_cameras)
    return JSONResponse(
        content={
            "timezone": store.timezone,
            "hours": hourly.series(
                store,
                cameras=cameras,
                primary_model_id=primary_model,
                sl_model_md5=sl_model,
            ),
            "models": hourly.models(store, cameras=cameras),
        }
    )


@router.get("/secondlook/analyses/summary")
def analyses_summary(
    request: Request,
    camera: str | None = None,
    label: str | None = None,
    sl_model: str | None = None,
    primary_model: str | None = None,
    outcome: str | None = None,
    live: str | None = None,
    after: float | None = None,
    before: float | None = None,
    allowed_cameras: list[str] = Depends(get_allowed_cameras_for_filter),
):
    """Per-object Secondlook analyses: counts by outcome and live result,
    factor / score / size histograms and the models seen. Filters: camera
    (comma-separated), label, sl_model (MD5), primary_model (MD5), outcome,
    live, after / before (track end, epoch seconds)."""
    store, err = _store_or_503(request)
    if err:
        return err
    if outcome is not None and outcome not in analyses.OUTCOMES:
        return _bad(f"outcome must be one of {', '.join(analyses.OUTCOMES)}")
    if live is not None and live not in analyses.LIVE:
        return _bad(f"live must be one of {', '.join(analyses.LIVE)}")
    cameras = list(allowed_cameras)
    if camera:
        wanted = [c for c in camera.split(",") if c]
        cameras = [c for c in wanted if c in cameras]
    return JSONResponse(
        content=analyses.summary(
            store,
            cameras=cameras,
            label=label,
            sl_model=sl_model,
            primary_model=primary_model,
            outcome=outcome,
            live=live,
            after=after,
            before=before,
        )
    )


def _frame_sizes(config) -> dict[str, tuple[int, int]]:
    """Detect resolution per camera: the frame size for rows saved without evidence."""
    return {
        name: (cam.detect.width, cam.detect.height)
        for name, cam in config.cameras.items()
        if cam.detect.width and cam.detect.height
    }


def _bad(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status, content={"success": False, "message": message})


def _filters(
    request: Request,
    allowed_cameras: list[str],
    *,
    cameras: str | None,
    after: float | None,
    before: float | None,
    labels: str | None,
    zones: str | None,
    kind: str | None,
    outcome: str | None,
    mark: str | None,
    reviewed: bool | None,
    hide_edge_small: bool,
    group: str | None = None,
) -> tuple[dict | None, JSONResponse | None]:
    """Corrections filters shared by the list, groups, bulk verdict and mask routes."""
    cam_list = [c for c in (cameras.split(",") if cameras else []) if c] or None
    if cam_list:
        cam_list = [c for c in cam_list if c in allowed_cameras]
    else:
        cam_list = list(allowed_cameras)

    if kind and kind not in KINDS:
        return None, _bad(f"kind must be one of {KINDS}")
    if outcome is not None and outcome not in OUTCOMES:
        return None, _bad(f"outcome must be one of {OUTCOMES}")
    if mark is not None and mark not in MARKS:
        return None, _bad(f"mark must be one of {MARKS}")

    label_list = [x for x in (labels.split(",") if labels else []) if x] or None
    zone_list = [x for x in (zones.split(",") if zones else []) if x] or None

    filters = dict(
        cameras=cam_list,
        after=after,
        before=before,
        labels=label_list,
        zones=zone_list,
        kind=kind,
        outcome=outcome,
        mark=mark,
        reviewed=reviewed,
    )
    if hide_edge_small:
        review = request.app.frigate_config.secondlook.save_and_review.review
        filters["hide_edge_small"] = (
            review.edge_margin_px,
            review.small_dim_fraction,
        )
    if group is not None:
        try:
            filters["group"] = clusters.parse_group(group)
        except ValueError as e:
            return None, _bad(str(e))
        filters["frame_sizes"] = _frame_sizes(request.app.frigate_config)
    return filters, None


@router.get("/secondlook/save_and_review")
def list_examples(
    request: Request,
    cameras: str | None = None,
    after: float | None = None,
    before: float | None = None,
    labels: str | None = None,
    zones: str | None = None,
    kind: str | None = None,
    outcome: str | None = None,
    mark: str | None = None,
    reviewed: bool | None = None,
    hide_edge_small: bool = False,
    group: str | None = None,
    limit: int = 100,
    offset: int = 0,
    allowed_cameras: list[str] = Depends(get_allowed_cameras_for_filter),
):
    """outcome: acted (FN rescued, FP suppressed) or not_acted, from the stored
    event_id. group: a long-term group key from /groups (drill-in)."""
    store, err = _store_or_503(request)
    if err:
        return err
    filters, err = _filters(
        request,
        allowed_cameras,
        cameras=cameras,
        after=after,
        before=before,
        labels=labels,
        zones=zones,
        kind=kind,
        outcome=outcome,
        mark=mark,
        reviewed=reviewed,
        hide_edge_small=hide_edge_small,
        group=group,
    )
    if err:
        return err

    rows = store.list(**filters, limit=min(max(limit, 1), 500), offset=max(offset, 0))
    total = store.list(**filters, count_only=True)
    unreviewed_filters = {**filters, "reviewed": False}
    unreviewed_filters.pop("mark", None)
    hidden = 0
    if hide_edge_small:
        unfiltered = {k: v for k, v in filters.items() if k != "hide_edge_small"}
        hidden = store.list(**unfiltered, count_only=True) - total

    return JSONResponse(
        content={
            "examples": _with_outcomes(request, rows),
            "unreviewed": store.list(**unreviewed_filters, count_only=True),
            "total": total,
            "has_more": offset + len(rows) < total,
            "hidden_edge_small": hidden,
            "diagnostics": dict(store.diagnostics),
        }
    )


@router.get("/secondlook/save_and_review/groups")
def list_groups(
    request: Request,
    cameras: str | None = None,
    after: float | None = None,
    before: float | None = None,
    labels: str | None = None,
    zones: str | None = None,
    kind: str | None = None,
    outcome: str | None = None,
    reviewed: bool | None = None,
    hide_edge_small: bool = False,
    limit: int = 50,
    offset: int = 0,
    allowed_cameras: list[str] = Depends(get_allowed_cameras_for_filter),
):
    """Long-term groups: fixed bins (camera, kind, outcome, class, 8x6 cell,
    size bucket) with touching cells merged, most members first."""
    store, err = _store_or_503(request)
    if err:
        return err
    filters, err = _filters(
        request,
        allowed_cameras,
        cameras=cameras,
        after=after,
        before=before,
        labels=labels,
        zones=zones,
        kind=kind,
        outcome=outcome,
        mark=None,
        reviewed=reviewed,
        hide_edge_small=hide_edge_small,
        group=None,
    )
    if err:
        return err

    result = store.groups(frame_sizes=_frame_sizes(request.app.frigate_config), **filters)
    groups = result["groups"]
    start = max(offset, 0)
    page = groups[start : start + min(max(limit, 1), 200)]
    return JSONResponse(
        content={
            "groups": page,
            "total_groups": len(groups),
            "total_rows": result["rows"],
            "has_more": start + len(page) < len(groups),
            "grid": {"columns": clusters.GRID_COLS, "rows": clusters.GRID_ROWS},
            "size_labels": list(clusters.SIZE_LABELS),
        }
    )


class GroupMarkBody(BaseModel):
    mark: str = Field(
        description="Verdict for every unreviewed member: primary_right, secondlook_right or ignore"
    )


@router.post("/secondlook/save_and_review/groups/mark")
def mark_group(
    request: Request,
    body: GroupMarkBody,
    group: str,
    cameras: str | None = None,
    after: float | None = None,
    before: float | None = None,
    labels: str | None = None,
    zones: str | None = None,
    kind: str | None = None,
    outcome: str | None = None,
    hide_edge_small: bool = False,
    allowed_cameras: list[str] = Depends(get_allowed_cameras_for_filter),
):
    """Bulk verdict for the unreviewed members of one long-term group."""
    store, err = _store_or_503(request)
    if err:
        return err
    filters, err = _filters(
        request,
        allowed_cameras,
        cameras=cameras,
        after=after,
        before=before,
        labels=labels,
        zones=zones,
        kind=kind,
        outcome=outcome,
        mark=None,
        reviewed=False,
        hide_edge_small=hide_edge_small,
        group=group,
    )
    if err:
        return err

    try:
        updated = store.mark_matching(body.mark, **filters)
    except ValueError as e:
        return _bad(str(e))
    return JSONResponse(content={"success": True, "updated": updated})


@router.get("/secondlook/save_and_review/groups/mask")
def suggest_mask(
    request: Request,
    group: str,
    margin: float = 0.02,
    points: str = "boxes",
    cameras: str | None = None,
    after: float | None = None,
    before: float | None = None,
    labels: str | None = None,
    zones: str | None = None,
    kind: str | None = None,
    outcome: str | None = None,
    reviewed: bool | None = None,
    hide_edge_small: bool = False,
    allowed_cameras: list[str] = Depends(get_allowed_cameras_for_filter),
):
    """Suggested object filter mask for one false positive group. Nothing is
    written: the response carries YAML to paste into the config by hand."""
    store, err = _store_or_503(request)
    if err:
        return err
    filters, err = _filters(
        request,
        allowed_cameras,
        cameras=cameras,
        after=after,
        before=before,
        labels=labels,
        zones=zones,
        kind=kind,
        outcome=outcome,
        mark=None,
        reviewed=reviewed,
        hide_edge_small=hide_edge_small,
        group=group,
    )
    if err:
        return err

    key = filters["group"]
    if key["kind"] != "false_positive" or not key["cells"]:
        return _bad("A mask is suggested only for a located false positive group")
    if points not in ("boxes", "bottom"):
        return _bad("points must be boxes or bottom")
    margin = min(max(margin, 0.0), 0.2)
    members = store.group_boxes(**filters)
    if not members:
        return _bad("No located members in this group", 404)
    boxes = [(m["rx1"], m["ry1"], m["rx2"], m["ry2"]) for m in members]
    polygon = clusters.mask_polygon(boxes, margin=margin, points=points)
    coordinates = ",".join(
        f"{clusters.format_coordinate(x)},{clusters.format_coordinate(y)}"
        for x, y in polygon
    )
    values = [float(v) for v in coordinates.split(",")]
    rounded = list(zip(values[0::2], values[1::2]))
    camera, label = key["camera"], key["label"]
    digest = hashlib.sha256(
        json.dumps(key, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()[:6]
    name = f"secondlook_{label}_{digest}"
    friendly = f"Secondlook suggestion: {label} false positives ({len(members)})"
    cam_config = request.app.frigate_config.cameras.get(camera)
    existing = []
    if cam_config is not None and label in cam_config.objects.filters:
        existing = sorted(cam_config.objects.filters[label].mask or {})
    inside = sum(
        clusters.point_in_polygon((x1 + x2) / 2, y2, rounded) for x1, _, x2, y2 in boxes
    )
    return JSONResponse(
        content={
            "camera": camera,
            "label": label,
            "name": name,
            "friendly_name": friendly,
            "members": len(members),
            "points": points,
            "margin": margin,
            "polygon": [list(p) for p in rounded],
            "coordinates": coordinates,
            "area_fraction": clusters.polygon_area(rounded),
            "bottom_centres_inside": inside,
            "existing_masks": existing,
            "yaml": clusters.mask_yaml(camera, label, name, friendly, coordinates),
            "boxes": [list(b) for b in boxes[:300]],
            "preview_id": members[0]["id"],
        }
    )


@router.get("/secondlook/save_and_review/{example_id}")
def get_example(request: Request, example_id: str):
    store, err = _store_or_503(request)
    if err:
        return err
    rec = store.get(example_id)
    if rec is None:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "Example not found"},
        )
    return JSONResponse(content=_with_outcomes(request, [rec])[0])


@router.get("/secondlook/save_and_review/{example_id}/still.jpg")
def get_still(request: Request, example_id: str):
    store, err = _store_or_503(request)
    if err:
        return err
    rec = store.get(example_id)
    if rec is None or not os.path.isfile(rec.still_path):
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "Still not found"},
        )
    return FileResponse(rec.still_path, media_type="image/jpeg")


@router.post("/secondlook/save_and_review/{example_id}/mark")
def mark_example(request: Request, example_id: str, body: MarkBody):
    store, err = _store_or_503(request)
    if err:
        return err
    try:
        rec = store.set_mark(example_id, body.mark)
    except ValueError as e:
        return JSONResponse(
            status_code=400, content={"success": False, "message": str(e)}
        )
    if rec is None:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "Example not found"},
        )
    return JSONResponse(
        content={"success": True, "example": _with_outcomes(request, [rec])[0]}
    )
