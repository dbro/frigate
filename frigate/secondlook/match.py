"""Same-class confirmation and misclass selection for Secondlook."""

from __future__ import annotations

from typing import Any

# Stricter than the default same-class min_box_iou (0.3). A bicycle under a
# rider or a car around a person typically overlaps loosely; a swapped class
# (cat called person) occupies nearly the same box.
MISCLASS_MIN_IOU = 0.5


def box_iou(a: list[float], b: list[float]) -> float:
    """IoU of two [ymin, xmin, ymax, xmax] boxes in the same coordinates."""
    iy1, ix1 = max(a[0], b[0]), max(a[1], b[1])
    iy2, ix2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, iy2 - iy1) * max(0.0, ix2 - ix1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


def select_misclass(
    detections: list[dict[str, Any]],
    primary_label: str,
    same_class: dict[str, Any] | None,
    score_threshold: float,
    track_box_norm: list[float] | None,
    min_iou: float = MISCLASS_MIN_IOU,
) -> tuple[str | None, float | None, list[float] | None]:
    """Choose a misclass label, or all-None if this is not a misclass.

    Same-class matching is exact label equality, using the same score
    threshold as best_same_class. A same-class hit confirms the primary box
    and other overlapping classes are ignored. Only if there is no such hit
    may a different-class detection be a misclass candidate, and then only
    when its score meets score_threshold and its box overlaps the primary
    box at min_iou. Other primary tracks in the same frame are not available
    at this point, so a box that is the same-class partner of another
    tracked object is not excluded (follow-up).
    """
    if same_class is not None or not detections or track_box_norm is None:
        return None, None, None
    candidates: list[dict[str, Any]] = []
    for detection in detections:
        label = detection.get("label")
        score = float(detection.get("score") or 0.0)
        box = detection.get("box_norm")
        if (
            not label
            or label == primary_label
            or score < score_threshold
            or not box
            or box_iou(box, track_box_norm) < min_iou
        ):
            continue
        candidates.append(detection)
    if not candidates:
        return None, None, None
    top = max(candidates, key=lambda d: float(d.get("score") or 0.0))
    return top["label"], float(top["score"]), top.get("box_norm")
