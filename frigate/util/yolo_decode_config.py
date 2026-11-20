"""Shared config fields for YOLO post-processing (score cutoff, NMS, groups).

Kept free of heavy imports (pydantic only) so both detector configs and the
Secondlook config can inherit it without import cycles.
"""

from pydantic import BaseModel, Field, field_validator

# Frigate's detector result buffer is fixed at 20 rows of
# [class_id, score, ymin, xmin, ymax, xmax] (frigate/object_detection/base.py)
FRIGATE_RESULT_ROWS = 20
MAX_DETECTIONS_LIMIT = 300


def check_nms_class_groups(v):
    """Reject a class name listed more than once (across or within groups)."""
    if v is None:
        return v
    seen = set()
    for group in v:
        names = [group] if isinstance(group, str) else group
        if not names:
            raise ValueError("nms_class_groups: empty group")
        for name in names:
            if name in seen:
                raise ValueError(
                    f"nms_class_groups: class '{name}' is listed more than once"
                )
            seen.add(name)
    return v


class YoloDecodeConfig(BaseModel):
    """Mixin: post-processing settings for yolo-generic models."""

    min_score: float = Field(
        default=0.4,
        gt=0.0,
        lt=1.0,
        title="Minimum Score",
        description="Minimum class score for a candidate box to survive YOLO post-processing (yolo-generic models).",
    )
    nms_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        title="NMS IoU Threshold",
        description="Overlap (IoU) above which the lower-scoring box is removed by NMS (yolo-generic models).",
    )
    max_detections: int = Field(
        default=FRIGATE_RESULT_ROWS,
        ge=1,
        le=MAX_DETECTIONS_LIMIT,
        title="Max Detections",
        description=f"Maximum number of boxes kept per frame after NMS. Frigate detectors use at most {FRIGATE_RESULT_ROWS} (yolo-generic models).",
    )
    nms_class_groups: list[str | list[str]] | None = Field(
        default=None,
        title="NMS Class Groups",
        description="Groups of class names for NMS. Boxes only suppress boxes in the same group; unlisted classes share one group. Unset = all classes in one group (yolo-generic models).",
    )

    @field_validator("nms_class_groups")
    @classmethod
    def _check_nms_class_groups(cls, v):
        return check_nms_class_groups(v)
