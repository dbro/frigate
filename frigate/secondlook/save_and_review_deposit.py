"""Deposit a fine-tune example at track end when SL corrected the decision."""

from __future__ import annotations

import logging
from typing import Any

from frigate.secondlook.save_and_review import (
    get_save_and_review_store,
)

logger = logging.getLogger(__name__)


def save_track_for_review(obj: Any, sl_config: Any) -> str | None:
    """If this track is a corrected FP/FN/misclass, write one example. Return id or None."""
    if sl_config is None or not sl_config.save_and_review.enabled:
        return None
    if not getattr(obj, "_secondlook_enabled_for_label", lambda: False)():
        return None

    store = get_save_and_review_store()
    if store is None:
        return None

    if getattr(obj, "save_and_review_deposited", False):
        return None
    store.diagnostics["tracks_ended"] += 1
    jpeg = getattr(obj, "save_and_review_jpeg", None)
    for choice in getattr(obj, "save_and_review_decisions", {}).values():
        if choice.get("evidence"):
            from frigate.secondlook.evidence import jpeg_bytes

            jpeg = jpeg_bytes(choice["evidence"])
            break
    if not jpeg:
        # fall back: try encoding from thumbnail frame cache if still present
        jpeg = _jpeg_from_thumbnail(obj)
    if not jpeg:
        store.diagnostics["no_still_frame"] += 1
        logger.debug(
            "Secondlook deposit skip %s — no still frame",
            obj.obj_data.get("id"),
        )
        return None

    choices = getattr(obj, "save_and_review_decisions", {})
    # FP / FN by the final live outcome, acted on or not (not_rescued and
    # not_suppressed are saved too, to investigate why)
    finish = getattr(obj, "finish_secondlook", None)
    outcome = (finish() if finish else getattr(obj, "secondlook_outcome", None)) or ""
    final = (
        "false_negative"
        if outcome.startswith("fn_")
        else "false_positive"
        if outcome.startswith("fp_")
        else None
    )
    kind = "misclass" if "misclass" in choices else (final if final in choices else None)
    if kind is None:
        reason = (
            "already_released_or_no_qualifying_decision"
            if not obj.false_positive
            else "no_qualifying_suppression"
        )
        store.diagnostics[reason] += 1
        logger.debug(
            "Secondlook deposit skip %s reason=%s raw_primary_latched=%s",
            obj.obj_data.get("id"),
            reason,
            getattr(obj, "primary_would_release", False),
        )
        return None
    candidate = choices[kind]
    sl = candidate["sl"]
    evidence = candidate["evidence"]
    label = obj.obj_data.get("label") or "unknown"
    misclass_label = sl.get("misclass_label")
    threshold = candidate["threshold"]
    raw_median = candidate["raw_median"]
    factor = candidate["factor"]
    if evidence:
        from frigate.secondlook.evidence import jpeg_bytes, metadata

        jpeg = jpeg_bytes(evidence)
        box = evidence["primary_box"]
        sl_box = evidence["sl_box"]
        frame_time = evidence["frame_time"]
        primary_score = evidence["primary_score"]
        sl_score = evidence["secondary_score"]
        saved_evidence = {
            **metadata(evidence),
            "decision_time": candidate["decision_time"],
        }
    else:
        box = getattr(obj, "save_and_review_box", None)
        sl_box = None
        frame_time = getattr(obj, "save_and_review_frame_time", None)
        primary_score = sl.get("primary_score")
        sl_score = sl.get("misclass_score") if misclass_label else sl.get("sl_score")
        saved_evidence = None

    try:
        rec = store.deposit(
            track_id=str(obj.obj_data.get("id")),
            camera=obj.camera_config.name,
            kind=kind,
            label=label,
            jpeg_bytes=jpeg,
            misclass_label=misclass_label,
            primary_score=float(primary_score) if primary_score is not None else None,
            sl_score=float(sl_score) if sl_score is not None else None,
            factor=float(factor) if factor is not None else None,
            threshold=threshold,
            raw_median=raw_median,
            box=[int(v) for v in box] if box else None,
            sl_box=[float(v) for v in sl_box] if sl_box else None,
            frame_time=float(frame_time) if frame_time is not None else None,
            model_id=sl.get("model_id"),
            sl_model_md5=sl.get("model_md5")
            or getattr(obj, "secondlook_model_md5", None)
            or store.sl_model_md5,
            primary_model_id=getattr(obj, "primary_model_id", None),
            evidence=saved_evidence,
            event_id=None if obj.false_positive else str(obj.obj_data.get("id")),
            has_clip=bool(getattr(obj, "has_clip", False)) and not obj.false_positive,
            zones=list(getattr(obj, "entered_zones", []) or []),
        )
    except Exception:
        store.diagnostics["deposit_failed"] += 1
        raise
    store.diagnostics["deposited_" + kind if rec else "deposit_failed"] += 1
    if rec:
        obj.save_and_review_deposited = True
    return rec.id if rec else None


def _jpeg_from_thumbnail(obj: Any) -> bytes | None:
    try:
        import cv2

        if obj.thumbnail_data is None:
            return None
        frame_time = obj.thumbnail_data["frame_time"]
        yuv = obj.frame_cache[frame_time]["frame"]
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            return buf.tobytes()
    except Exception:
        logger.debug("thumbnail JPEG fallback failed", exc_info=True)
    return None
