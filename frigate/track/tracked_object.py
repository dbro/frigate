"""Object attribute."""

import logging
import math
import os
from collections import defaultdict
from statistics import median
from typing import Any, cast

import cv2
import numpy as np

from frigate.config import (
    CameraConfig,
    FilterConfig,
    UIConfig,
)
from frigate.const import CLIPS_DIR, REPLAY_CAMERA_PREFIX, THUMB_DIR
from frigate.detectors.detector_config import ModelConfig
from frigate.review.types import SeverityEnum
from frigate.util.builtin import sanitize_float
from frigate.util.image import (
    area,
    get_snapshot_bytes,
    is_better_thumbnail,
)
from frigate.util.object import box_inside
from frigate.secondlook.factor import adjusted_median_scores
from frigate.secondlook.detection_check import decide_false_positive
from frigate.secondlook import hourly
from frigate.secondlook.metrics import count_correction, outcome_kind
from frigate.util.velocity import calculate_real_world_speed

logger = logging.getLogger(__name__)


# In most cases objects that loiter in a loitering zone should alert,
# but can still be expected to stay stationary for extended periods of time
# (ex: car loitering on the street vs when a known person parks on the street)
# person is the main object that should keep alerts going as long as they loiter
# even if they are stationary.
EXTENDED_LOITERING_OBJECTS = ["person"]


class TrackedObject:
    def __init__(
        self,
        model_config: ModelConfig,
        camera_config: CameraConfig,
        ui_config: UIConfig,
        frame_cache: dict[float, dict[str, Any]],
        obj_data: dict[str, Any],
    ) -> None:
        # set the score history then remove as it is not part of object state
        self.score_history: list[float] = obj_data["score_history"]
        del obj_data["score_history"]

        self.obj_data = obj_data
        self.primary_model_id = getattr(model_config, "model_hash", None)
        self.colormap = model_config.colormap
        self.logos = model_config.all_attribute_logos
        self.thumbnail_attributes = [
            attr
            for attr in model_config.attributes_map.get(obj_data["label"], [])
            if attr in model_config.non_logo_attributes
        ]
        self.camera_config = camera_config
        self.ui_config = ui_config
        self.frame_cache = frame_cache
        self.zone_presence: dict[str, int] = {}
        self.zone_loitering: dict[str, int] = {}
        self.current_zones: list[str] = []
        self.entered_zones: list[str] = []
        self.new_zone_entered: bool = False
        self.attributes: dict[str, float] = defaultdict(float)
        self.false_positive = True
        self.has_clip = False
        self.has_snapshot = False
        self.top_score = self.computed_score = 0.0
        # Secondlook live QA state (local patch)
        self.secondlook_pending_since: float | None = None
        self.secondlook_running: bool = False
        self.secondlook_factor: float | None = None
        self.primary_would_release = False
        self.secondlook_counted_kinds: set[str] = set()
        self.secondlook_analysed_counted = False
        # extremes of every Secondlook factor seen; the outcome is decided once
        # at track end (or when a late answer for an ended track arrives)
        self.secondlook_factor_min: float | None = None
        self.secondlook_factor_max: float | None = None
        self.secondlook_result_seen = False
        self.secondlook_outcome: str | None = None
        self.secondlook_finished = False
        # per-track analyses row (secondlook/analyses.py): request seen, answers
        # by kind, model identity from the answers
        self.secondlook_asked = False
        self.secondlook_answers: dict[str, int] = {}
        self.secondlook_model_md5: str | None = None
        self.secondlook_model_name: str | None = None
        self.secondlook_last_sl_score: float | None = None
        self.secondlook_hold_expired = False
        self.secondlook_answer_after_end = False
        self.secondlook_analysis_written = False
        self.primary_crossing_median = 0.0
        # the score buffer before detection at the primary's highest median of the track:
        # where a factor would first release (FN) or must hold back (FP)
        self.primary_peak_history: list[float] = list(self.score_history)
        self.primary_peak_median = (
            median(self.score_history) if self.score_history else -1.0
        )
        self.save_and_review_decisions: dict[str, dict] = {}
        self.secondlook_evidence: dict | None = None
        self.secondlook_hold_done: bool = False
        self._secondlook_frame_time: float | None = None
        # Fine-tune example candidate (max-primary full frame during track)
        self.save_and_review_best_primary: float = 0.0
        self.save_and_review_box: list[int] | None = None
        self.save_and_review_frame_time: float | None = None
        self.save_and_review_jpeg: bytes | None = None
        self.save_and_review_needs_frame: bool = False
        self.thumbnail_data: dict[str, Any] | None = None
        self.last_updated: float = 0
        self.last_published: float = 0
        self.frame = None
        self.active = True
        self.pending_loitering = False
        self.speed_history: list[float] = []
        self.current_estimated_speed: float = 0
        self.average_estimated_speed: float = 0
        self.velocity_angle = 0
        self.path_data: list[tuple[Any, float]] = []
        self.accumulated_resource_busy_ms: float = float(
            obj_data.get("frame_resource_busy_ms") or 0.0
        )
        self.first_detect_time: float = float(
            obj_data.get("first_detect_time") or obj_data.get("start_time") or 0.0
        )
        self.first_detect_box: list[int] = list(
            obj_data.get("first_detect_box") or obj_data.get("box") or []
        )
        self.last_active_time: float = float(obj_data.get("start_time") or 0.0)
        self.confirmed_time: float | None = None
        self.secondlook_request_time: float | None = None
        self.secondlook_answer_time: float | None = None
        self.previous = self.to_dict()

    @property
    def max_severity(self) -> str | None:
        review_config = self.camera_config.review

        if (
            self.camera_config.review.alerts.enabled
            and self.obj_data["label"] in review_config.alerts.labels
            and (
                not review_config.alerts.required_zones
                or set(self.entered_zones) & set(review_config.alerts.required_zones)
            )
        ):
            return SeverityEnum.alert

        if (
            self.camera_config.review.detections.enabled
            and (
                not review_config.detections.labels
                or self.obj_data["label"] in review_config.detections.labels
            )
            and (
                not review_config.detections.required_zones
                or set(self.entered_zones)
                & set(review_config.detections.required_zones)
            )
        ):
            return SeverityEnum.detection

        return None

    def _secondlook_enabled_for_label(self) -> bool:
        cam_sl = getattr(self.camera_config, "secondlook", None)
        if cam_sl is None or not cam_sl.enabled:
            return False
        objects = cam_sl.objects
        if objects is None:
            # objects inherited at config merge; if still None, treat as disabled list
            return False
        return self.obj_data["label"] in objects

    def _is_false_positive(self) -> bool:
        threshold = self.camera_config.objects.filters[self.obj_data["label"]].threshold
        sl_enabled = self._secondlook_enabled_for_label()
        if not sl_enabled:
            # stock: once a true positive, always a true positive
            if not self.false_positive:
                return False
            return self.computed_score < threshold

        hold_seconds = float(
            getattr(self.camera_config.secondlook, "hold_seconds", 1.0)
        )
        # check the frame being processed (obj_data is merged after this check)
        now = float(
            self._secondlook_frame_time or self.obj_data.get("frame_time") or 0.0
        )

        is_fp, new_pending = decide_false_positive(
            already_true_positive=not self.false_positive,
            secondlook_enabled=True,
            score_history=self.score_history,
            threshold=threshold,
            factor=self.secondlook_factor,
            pending_since=self.secondlook_pending_since,
            frame_time=now,
            hold_seconds=hold_seconds,
            running=self.secondlook_running,
            hold_done=self.secondlook_hold_done,
        )
        if self.false_positive and self.secondlook_factor is None:
            if new_pending is None:
                if (
                    not self.secondlook_hold_done
                    and self.secondlook_pending_since is not None
                ):
                    self.secondlook_hold_expired = True
                if (
                    not self.secondlook_hold_done
                    and self.secondlook_pending_since is not None
                    and self.camera_config.secondlook._save_and_review_enabled
                ):
                    from frigate.secondlook.save_and_review import (
                        get_save_and_review_store,
                    )

                    store = get_save_and_review_store()
                    if store is not None:
                        store.diagnostics["hold_timeouts"] += 1
                # hold expired (or already done): never re-arm for this track
                self.secondlook_pending_since = None
                self.secondlook_hold_done = True
            else:
                self.secondlook_pending_since = new_pending
                if self.secondlook_request_time is None and new_pending is not None:
                    self.secondlook_request_time = float(new_pending)
        return is_fp

    def _remember_correction(self, raw_median, threshold, was_fp, decision_time):
        sl = self.obj_data.get("secondlook") or {}
        factor = self.secondlook_factor
        kind = None
        # FP / FN candidates whether or not the pipeline acted on them (the
        # outcome is final at track end, see finish_secondlook); misclass has
        # no live effect and is counted here as before
        if sl.get("misclass_label"):
            kind = "misclass"
        elif (
            self.primary_would_release
            and factor is not None
            and factor < self.camera_config.secondlook._gap_factor_low
        ):
            kind = "false_positive"
        elif (
            not self.primary_would_release
            and factor is not None
            and factor > self.camera_config.secondlook._gap_factor_high
        ):
            kind = "false_negative"
        if kind == "misclass" and kind not in self.secondlook_counted_kinds:
            # live accuracy counter: once per track and kind, whether or not
            # examples are saved
            self.secondlook_counted_kinds.add(kind)
            count_correction(kind)
            if self.camera_config.secondlook._save_and_review_enabled:
                self._hourly(kind)
        if kind and self.camera_config.secondlook._save_and_review_enabled:
            candidate = {
                "evidence": self.secondlook_evidence,
                "sl": dict(sl),
                "raw_median": self.primary_crossing_median
                if kind == "false_positive"
                else raw_median,
                "threshold": threshold,
                "decision_time": decision_time,
                "factor": factor,
            }
            old = self.save_and_review_decisions.get(kind)
            score = (self.secondlook_evidence or {}).get("primary_score", 0)
            if old is None:
                from frigate.secondlook.save_and_review import get_save_and_review_store

                store = get_save_and_review_store()
                if store is not None:
                    store.diagnostics["qualifying_" + kind] += 1
            if old is None or score >= (old.get("evidence") or {}).get(
                "primary_score", 0
            ):
                self.save_and_review_decisions[kind] = candidate

    def compute_score(self) -> float:
        """get median of scores for object."""
        if self.secondlook_factor is not None and self._secondlook_enabled_for_label():
            return median(
                adjusted_median_scores(self.score_history, self.secondlook_factor)
            )
        return median(self.score_history)

    def apply_secondlook(self, payload: dict[str, Any]) -> None:
        """Store Secondlook metadata from the embeddings process.

        A pending payload only marks SL as running. Any final payload ends the
        hold: factor set -> scale; factor None (no same-class hit, or
        queue-full fail-open) -> unset = 1.0, per the configured rules.
        """
        self.secondlook_asked = True
        if payload.get("model_md5"):
            self.secondlook_model_md5 = payload["model_md5"]
        if payload.get("model_id"):
            self.secondlook_model_name = payload["model_id"]
        if payload.get("pending"):
            if not self.secondlook_hold_done:
                self.obj_data["secondlook"] = payload
                self.secondlook_running = True
                if self.secondlook_request_time is None and payload.get("frame_time") is not None:
                    self.secondlook_request_time = float(payload["frame_time"])
            return

        if (
            self.camera_config.secondlook._save_and_review_enabled
            and self._secondlook_enabled_for_label()
        ):
            from frigate.secondlook.save_and_review import get_save_and_review_store

            store = get_save_and_review_store()
            if store is not None:
                store.diagnostics["results_received"] += 1
                if payload.get("factor") is None:
                    store.diagnostics["results_unset_factor"] += 1
                if payload.get("applied_by") == "secondlook_no_hit":
                    store.diagnostics["results_no_same_class"] += 1
                if payload.get("misclass_label"):
                    store.diagnostics["results_diff_class"] += 1
        if (
            payload.get("analysed")
            and not self.secondlook_analysed_counted
            and self.camera_config.secondlook._save_and_review_enabled
        ):
            # hourly summary: each tracked object once, on its first result
            self.secondlook_analysed_counted = True
            self._hourly(hourly.ANALYSED)
        from frigate.secondlook.analyses import answer_kind

        kind = answer_kind(payload)
        self.secondlook_answers[kind] = self.secondlook_answers.get(kind, 0) + 1
        if payload.get("sl_score") is not None:
            self.secondlook_last_sl_score = float(payload["sl_score"])
        self.secondlook_evidence = payload.get("evidence")
        self.obj_data["secondlook"] = {
            k: v for k, v in payload.items() if k != "evidence"
        }

        self.secondlook_running = False
        self.secondlook_pending_since = None
        self.secondlook_hold_done = True
        self.secondlook_result_seen = True
        if self.secondlook_answer_time is None:
            ans_ts = (
                self._secondlook_frame_time
                or self.obj_data.get("frame_time")
                or payload.get("frame_time")
            )
            if ans_ts is not None:
                self.secondlook_answer_time = float(ans_ts)
        factor = payload.get("factor")
        self.secondlook_factor = None if factor is None else float(factor)
        if self.secondlook_factor is not None:
            f = self.secondlook_factor
            lo, hi = self.secondlook_factor_min, self.secondlook_factor_max
            self.secondlook_factor_min = f if lo is None else min(lo, f)
            self.secondlook_factor_max = f if hi is None else max(hi, f)

    def _hourly(self, *columns: str) -> None:
        hourly.record(
            self.camera_config.name,
            self.obj_data["label"],
            *columns,
            primary_model_id=self.primary_model_id,
            sl_model_md5=self.secondlook_model_md5,
        )

    def awaiting_secondlook(self) -> bool:
        """Analysis requested (pending seen) but no answer yet."""
        return (
            self._secondlook_enabled_for_label()
            and self.secondlook_running
            and not self.secondlook_result_seen
        )

    def _scaled_peak_median(self, factor: float) -> float:
        """Median of the peak score-buffer window with the factor applied."""
        return median(adjusted_median_scores(self.primary_peak_history, factor))

    def live_outcome(self) -> str | None:
        """Did Secondlook's disagreement change what the pipeline did?

        A disagreement counts only if the factor flips whether it becomes a detection.
        Release is permanent and happens at the first frame whose (scaled)
        median reaches the threshold, so the deciding window is the track's
        peak raw-median window: FN = primary never reached the threshold, the
        factor is above gap_factor_high and the scaled peak median reaches it;
        FP = primary reached it, the factor is below gap_factor_low and the
        scaled peak median stays below it. Near-agreements are neither.
        released = published as a detection (never un-released): FN rescued
        if released, FP suppressed if never released; answers too late to
        matter land in the not_ halves.
        """
        if not self._secondlook_enabled_for_label() or not self.primary_peak_history:
            return None
        released = not self.false_positive
        sl = self.camera_config.secondlook
        threshold = self.camera_config.objects.filters[self.obj_data["label"]].threshold
        high, low = self.secondlook_factor_max, self.secondlook_factor_min
        if (
            not self.primary_would_release
            and high is not None
            and high > sl._gap_factor_high
            and self._scaled_peak_median(high) >= threshold
        ):
            return "fn_rescued" if released else "fn_not_rescued"
        if (
            self.primary_would_release
            and low is not None
            and low < sl._gap_factor_low
            and self._scaled_peak_median(low) < threshold
        ):
            return "fp_not_suppressed" if released else "fp_suppressed"
        return None

    def finish_secondlook(self) -> str | None:
        """Count the final FN/FP outcome once per track (live + hourly)."""
        if self.secondlook_finished:
            return self.secondlook_outcome
        self.secondlook_finished = True
        if self._secondlook_enabled_for_label():
            # an answer applied after the last update has no candidate yet
            threshold = self.camera_config.objects.filters[
                self.obj_data["label"]
            ].threshold
            self._remember_correction(
                median(self.score_history) if self.score_history else 0.0,
                threshold,
                self.false_positive,
                self.obj_data.get("frame_time"),
            )
        outcome = self.live_outcome()
        self.secondlook_outcome = outcome
        if outcome is None:
            return None
        kind = outcome_kind(outcome)
        if kind not in self.secondlook_counted_kinds:
            self.secondlook_counted_kinds.add(kind)
            count_correction(kind, outcome)
            if self.camera_config.secondlook._save_and_review_enabled:
                self._hourly(kind, outcome)
        return outcome

    def update(
        self, current_frame_time: float, obj_data: dict[str, Any], has_valid_frame: bool
    ) -> tuple[bool, bool, bool, bool]:
        thumb_update = False
        significant_change = False
        path_update = False
        autotracker_update = False
        # if the object is not in the current frame, add a 0.0 to the score history
        if obj_data["frame_time"] != current_frame_time:
            self.score_history.append(0.0)
        else:
            self.score_history.append(obj_data["score"])

        # only keep the last 10 scores
        if len(self.score_history) > 10:
            self.score_history = self.score_history[-10:]

        # Accumulate detector queue wait during object birth/confirmation only
        if self.false_positive and obj_data.get("frame_resource_busy_ms") is not None:
            self.accumulated_resource_busy_ms += float(
                obj_data["frame_resource_busy_ms"]
            )

        # Max-primary frame for Corrections fine-tune still (track end deposit)
        if (
            self._secondlook_enabled_for_label()
            and self.camera_config.secondlook._save_and_review_enabled
            and obj_data.get("frame_time") == current_frame_time
        ):
            hit = float(obj_data.get("score") or 0.0)
            if hit > 0.0 and hit >= self.save_and_review_best_primary:
                self.save_and_review_best_primary = hit
                self.save_and_review_box = (
                    list(obj_data["box"]) if obj_data.get("box") else None
                )
                self.save_and_review_frame_time = current_frame_time
                self.save_and_review_needs_frame = True

        # calculate if this is a false positive
        self._secondlook_frame_time = current_frame_time
        self.computed_score = self.compute_score()
        if self.computed_score > self.top_score:
            self.top_score = self.computed_score
        was_false_positive = self.false_positive
        raw_median = median(self.score_history) if self.score_history else 0.0
        if raw_median > self.primary_peak_median:
            self.primary_peak_median = raw_median
            self.primary_peak_history = list(self.score_history)
        threshold = self.camera_config.objects.filters[self.obj_data["label"]].threshold
        was_primary = self.primary_would_release
        self.primary_would_release |= raw_median >= threshold
        if not was_primary and self.primary_would_release:
            self.primary_crossing_median = raw_median
        self.false_positive = self._is_false_positive()
        if was_false_positive and not self.false_positive and self.confirmed_time is None:
            self.confirmed_time = float(current_frame_time)
        if (
            self.camera_config.secondlook._save_and_review_enabled
            and self._secondlook_enabled_for_label()
        ):
            from frigate.secondlook.save_and_review import get_save_and_review_store

            store = get_save_and_review_store()
            if store is not None:
                if not was_primary and self.primary_would_release:
                    store.diagnostics["raw_primary_threshold_crossings"] += 1
                if was_false_positive and not self.false_positive:
                    store.diagnostics["actual_releases"] += 1
        self._remember_correction(
            raw_median, threshold, was_false_positive, current_frame_time
        )
        was_active = self.active
        self.active = self.is_active()
        if not was_active and self.active:
            self.last_active_time = float(current_frame_time)

        if not self.false_positive and has_valid_frame:
            # determine if this frame is a better thumbnail
            if self.thumbnail_data is None or is_better_thumbnail(
                self.thumbnail_attributes,
                self.thumbnail_data,
                obj_data,
                self.camera_config.frame_shape,
            ):
                if obj_data["frame_time"] == current_frame_time:
                    self.thumbnail_data = {
                        "frame_time": obj_data["frame_time"],
                        "box": obj_data["box"],
                        "area": obj_data["area"],
                        "region": obj_data["region"],
                        "score": obj_data["score"],
                        "attributes": obj_data["attributes"],
                        "current_estimated_speed": self.current_estimated_speed,
                        "velocity_angle": self.velocity_angle,
                        "path_data": self.path_data.copy(),
                        "recognized_license_plate": obj_data.get(
                            "recognized_license_plate"
                        ),
                        "recognized_license_plate_score": obj_data.get(
                            "recognized_license_plate_score"
                        ),
                    }
                    thumb_update = True
                else:
                    logger.debug(
                        f"{self.camera_config.name}: Object frame time {obj_data['frame_time']} is not equal to the current frame time {current_frame_time}, not updating thumbnail"
                    )

        # check zones
        current_zones = []
        bottom_center = (obj_data["centroid"][0], obj_data["box"][3])
        in_loitering_zone = False
        in_speed_zone = False

        # check each zone
        for name, zone in self.camera_config.zones.items():
            # skip disabled zones
            if not zone.enabled:
                continue

            # if the zone is not for this object type, skip
            if len(zone.objects) > 0 and obj_data["label"] not in zone.objects:
                continue
            contour = zone.contour
            zone_score = self.zone_presence.get(name, 0) + 1

            # check if the object is in the zone
            if cv2.pointPolygonTest(contour, bottom_center, False) >= 0:
                # if the object passed the filters once, dont apply again
                if name in self.current_zones or not zone_filtered(self, zone.filters):
                    # Calculate speed first if this is a speed zone
                    if (
                        zone.distances
                        and obj_data["frame_time"] == current_frame_time
                        and self.active
                    ):
                        speed_magnitude, self.velocity_angle = (
                            calculate_real_world_speed(
                                zone.contour,
                                zone.distances,
                                self.obj_data["estimate_velocity"],
                                bottom_center,
                                self.camera_config.detect.fps,
                            )
                        )

                        # users can configure speed zones incorrectly, so sanitize speed_magnitude
                        # and velocity_angle in case the values come back as inf or NaN
                        speed_magnitude = sanitize_float(speed_magnitude)
                        self.velocity_angle = sanitize_float(self.velocity_angle)

                        if self.ui_config.unit_system == "metric":
                            self.current_estimated_speed = (
                                speed_magnitude * 3.6
                            )  # m/s to km/h
                        else:
                            self.current_estimated_speed = (
                                speed_magnitude * 0.681818
                            )  # ft/s to mph

                        self.speed_history.append(self.current_estimated_speed)
                        if len(self.speed_history) > 10:
                            self.speed_history = self.speed_history[-10:]

                        self.average_estimated_speed = sum(self.speed_history) / len(
                            self.speed_history
                        )

                        # we've exceeded the speed threshold on the zone
                        # or we don't have a speed threshold set
                        if (
                            zone.speed_threshold is None
                            or self.average_estimated_speed > zone.speed_threshold
                        ):
                            in_speed_zone = True

                        logger.debug(
                            f"Camera: {self.camera_config.name}, tracked object ID: {self.obj_data['id']}, "
                            f"zone: {name}, pixel velocity: {str(tuple(np.round(self.obj_data['estimate_velocity']).flatten().astype(int)))}, "
                            f"speed magnitude: {speed_magnitude}, velocity angle: {self.velocity_angle}, "
                            f"estimated speed: {self.current_estimated_speed:.1f}, "
                            f"average speed: {self.average_estimated_speed:.1f}, "
                            f"length: {len(self.speed_history)}"
                        )

                    # Check zone entry conditions - for speed zones, require both inertia and speed
                    if zone_score >= zone.inertia:
                        if zone.distances and not in_speed_zone:
                            continue  # Skip zone entry for speed zones until speed threshold met

                        # if the zone has loitering time, and the object is an extended loiter object
                        # always mark it as loitering actively
                        if (
                            self.obj_data["label"] in EXTENDED_LOITERING_OBJECTS
                            and zone.loitering_time > 0
                        ):
                            in_loitering_zone = True

                        loitering_score = self.zone_loitering.get(name, 0) + 1

                        # loitering time is configured as seconds, convert to count of frames
                        if loitering_score >= (
                            self.camera_config.zones[name].loitering_time
                            * self.camera_config.detect.fps
                        ):
                            current_zones.append(name)

                            if name not in self.entered_zones:
                                self.entered_zones.append(name)
                                self.new_zone_entered = True
                        else:
                            self.zone_loitering[name] = loitering_score

                            # this object is pending loitering but has not entered the zone yet
                            if zone.loitering_time > 0:
                                in_loitering_zone = True
                    else:
                        self.zone_presence[name] = zone_score
            else:
                # once an object has a zone inertia of 3+ it is not checked anymore
                if 0 < zone_score < zone.inertia:
                    self.zone_presence[name] = zone_score - 1

            # Reset speed if not in speed zone
            if zone.distances and name not in current_zones:
                self.current_estimated_speed = 0

        # update loitering status
        self.pending_loitering = in_loitering_zone

        # maintain attributes
        for attr in obj_data["attributes"]:
            if self.attributes[attr["label"]] < attr["score"]:
                self.attributes[attr["label"]] = attr["score"]

        # populate the sub_label for object with highest scoring logo
        if self.obj_data["label"] in ["car", "motorcycle", "package", "person"]:
            recognized_logos = {
                k: self.attributes[k] for k in self.logos if k in self.attributes
            }
            if len(recognized_logos) > 0:
                max_logo = max(recognized_logos, key=recognized_logos.get)  # type: ignore[arg-type]

                # don't overwrite sub label if it is already set
                if (
                    self.obj_data.get("sub_label") is None
                    or self.obj_data["sub_label"][0] == max_logo
                ):
                    self.obj_data["sub_label"] = (max_logo, recognized_logos[max_logo])

        # check for significant change
        if not self.false_positive:
            # if the zones changed, signal an update
            if set(self.current_zones) != set(current_zones):
                significant_change = True

            # if the position changed, signal an update
            if self.obj_data["position_changes"] != obj_data["position_changes"]:
                significant_change = True

            # disappearance of a per-frame attribute can be caused by detection
            # skipping the object on a frame (stationary objects on non-interval
            # frames), so only flag when a new attribute label appears
            prev_labels = {a["label"] for a in self.obj_data["attributes"]}
            curr_labels = {a["label"] for a in obj_data["attributes"]}
            if curr_labels - prev_labels:
                significant_change = True

            # if the state changed between stationary and active
            if self.previous["active"] != self.active:
                significant_change = True

            # update at least once per minute
            if self.obj_data["frame_time"] - self.previous["frame_time"] > 60:
                significant_change = True

            # update autotrack at most 3 objects per second
            if self.obj_data["frame_time"] - self.previous["frame_time"] >= (1 / 3):
                autotracker_update = True

            # update path
            width = self.camera_config.detect.width
            height = self.camera_config.detect.height

            if width is not None and height is not None:
                bottom_center = (
                    round(obj_data["centroid"][0] / width, 4),
                    round(obj_data["box"][3] / height, 4),
                )

                # calculate a reasonable movement threshold (e.g., 5% of the frame diagonal)
                threshold = 0.05 * math.sqrt(width**2 + height**2) / max(width, height)

                if not self.path_data:
                    self.path_data.append((bottom_center, obj_data["frame_time"]))
                    path_update = True
                elif (
                    math.dist(self.path_data[-1][0], bottom_center) >= threshold
                    or len(self.path_data) == 1
                ):
                    # check Euclidean distance before appending
                    self.path_data.append((bottom_center, obj_data["frame_time"]))
                    path_update = True
                    logger.debug(
                        f"Point tracking: {obj_data['id']}, {bottom_center}, {obj_data['frame_time']}"
                    )

        self.obj_data.update(obj_data)
        self.current_zones = current_zones
        logger.debug(
            f"{self.camera_config.name}: Updating {obj_data['id']}: thumb update? {thumb_update}, significant change? {significant_change}, path update? {path_update}, autotracker update? {autotracker_update} "
        )
        return (thumb_update, significant_change, path_update, autotracker_update)

    def to_dict(self) -> dict[str, Any]:
        # Tracking internals excluded from output (centroid, estimate, estimate_velocity)
        _EXCLUDED_OBJ_DATA_KEYS = {
            "centroid",
            "estimate",
            "estimate_velocity",
        }

        event: dict[str, Any] = {
            "id": self.obj_data["id"],
            "camera": self.camera_config.name,
            "frame_time": self.obj_data["frame_time"],
            "snapshot": self.thumbnail_data,
            "snapshot_clean": True,
            "label": self.obj_data["label"],
            "sub_label": self.obj_data.get("sub_label"),
            "top_score": self.top_score,
            "false_positive": self.false_positive,
            "start_time": self.obj_data["start_time"],
            "end_time": self.obj_data.get("end_time", None),
            "score": self.obj_data["score"],
            "computed_score": self.computed_score,
            "box": self.obj_data["box"],
            "area": self.obj_data["area"],
            "ratio": self.obj_data["ratio"],
            "region": self.obj_data["region"],
            "active": self.active,
            "stationary": not self.active,
            "motionless_count": self.obj_data["motionless_count"],
            "position_changes": self.obj_data["position_changes"],
            "current_zones": self.current_zones.copy(),
            "entered_zones": self.entered_zones.copy(),
            "has_clip": self.has_clip,
            "has_snapshot": self.has_snapshot,
            "attributes": self.attributes,
            "current_attributes": self.obj_data["attributes"],
            "pending_loitering": self.pending_loitering,
            "max_severity": self.max_severity,
            "current_estimated_speed": self.current_estimated_speed,
            "average_estimated_speed": self.average_estimated_speed,
            "velocity_angle": self.velocity_angle,
            "path_data": self.path_data.copy(),
            "recognized_license_plate": self.obj_data.get("recognized_license_plate"),
            "accumulated_resource_busy_ms": self.accumulated_resource_busy_ms,
            "first_detect_time": self.first_detect_time,
            "first_detect_box": self.first_detect_box.copy(),
            "last_active_time": self.last_active_time,
            "confirmed_time": self.confirmed_time,
            "secondlook_request_time": self.secondlook_request_time,
            "secondlook_answer_time": self.secondlook_answer_time,
        }

        # Add any other obj_data keys (e.g. custom attribute fields) not yet included
        for key, value in self.obj_data.items():
            if key not in _EXCLUDED_OBJ_DATA_KEYS and key not in event:
                event[key] = value

        return event

    def is_active(self) -> bool:
        return not self.is_stationary()

    def is_stationary(self) -> bool:
        count = cast(int | float, self.obj_data["motionless_count"])
        return count > (self.camera_config.detect.stationary.threshold or 50)

    def get_thumbnail(self, ext: str) -> bytes | None:
        img_bytes, _ = self.get_img_bytes(
            ext, timestamp=False, bounding_box=False, crop=True, height=175
        )

        if img_bytes:
            return img_bytes
        else:
            _, img = cv2.imencode(f".{ext}", np.zeros((175, 175, 3), np.uint8))
            return img.tobytes()

    def get_clean_webp(self) -> bytes | None:
        webp_bytes, _ = self.get_img_bytes(
            ext="webp",
            timestamp=False,
            bounding_box=False,
            crop=False,
            height=None,
            quality=self.camera_config.snapshots.quality,
        )
        return webp_bytes

    def get_img_bytes(
        self,
        ext: str,
        timestamp: bool = False,
        bounding_box: bool = False,
        crop: bool = False,
        height: int | None = None,
        quality: int | None = None,
    ) -> tuple[bytes | None, float | None]:
        if self.thumbnail_data is None:
            return None, None

        try:
            frame_time = self.thumbnail_data["frame_time"]
            best_frame = cv2.cvtColor(
                self.frame_cache[frame_time]["frame"],
                cv2.COLOR_YUV2BGR_I420,
            )
        except KeyError:
            logger.warning(
                f"Unable to create snapshot because frame {frame_time} is not in the cache"
            )
            return None, None

        return get_snapshot_bytes(
            best_frame,
            frame_time,
            ext=ext,
            timestamp=timestamp,
            bounding_box=bounding_box,
            crop=crop,
            height=height,
            quality=quality,
            label=self.obj_data["label"],
            box=self.thumbnail_data["box"],
            score=self.thumbnail_data["score"],
            area=self.thumbnail_data["area"],
            attributes=self.thumbnail_data["attributes"],
            color=self.colormap.get(self.obj_data["label"], (255, 255, 255)),
            timestamp_style=self.camera_config.timestamp_style,
            estimated_speed=self.thumbnail_data["current_estimated_speed"],
        )

    def write_snapshot_to_disk(self) -> None:
        webp_bytes = self.get_clean_webp()
        if webp_bytes is None:
            logger.warning(f"Unable to save snapshot for {self.obj_data['id']}.")
        else:
            with open(
                os.path.join(
                    CLIPS_DIR,
                    f"{self.camera_config.name}-{self.obj_data['id']}-clean.webp",
                ),
                "wb",
            ) as p:
                p.write(webp_bytes)

    def capture_save_and_review_frame(self, yuv_frame: np.ndarray) -> None:
        """Encode full detect frame JPEG for Corrections deposit (max-primary)."""
        if not self.camera_config.secondlook._save_and_review_enabled:
            return
        try:
            bgr = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                self.save_and_review_jpeg = buf.tobytes()
                self.save_and_review_needs_frame = False
        except Exception:
            logger.debug(
                "Secondlook example frame encode failed for %s",
                self.obj_data.get("id"),
                exc_info=True,
            )

    def write_thumbnail_to_disk(self) -> None:
        if not self.camera_config.name:
            return

        if self.camera_config.name.startswith(REPLAY_CAMERA_PREFIX):
            return

        directory = os.path.join(THUMB_DIR, self.camera_config.name)

        os.makedirs(directory, exist_ok=True)

        thumb_bytes = self.get_thumbnail("webp")

        if thumb_bytes:
            with open(
                os.path.join(directory, f"{self.obj_data['id']}.webp"), "wb"
            ) as f:
                f.write(thumb_bytes)


def zone_filtered(obj: TrackedObject, object_config: dict[str, FilterConfig]) -> bool:
    object_name = obj.obj_data["label"]

    if object_name in object_config:
        obj_settings = object_config[object_name]

        # if the min area is larger than the
        # detected object, don't add it to detected objects
        if obj_settings.min_area > obj.obj_data["area"]:
            return True

        # if the detected object is larger than the
        # max area, don't add it to detected objects
        if obj_settings.max_area < obj.obj_data["area"]:
            return True

        # if the score is lower than the threshold, skip
        if obj_settings.threshold > obj.computed_score:
            return True

        # if the object is not proportionally wide enough
        if obj_settings.min_ratio > obj.obj_data["ratio"]:
            return True

        # if the object is proportionally too wide
        if obj_settings.max_ratio < obj.obj_data["ratio"]:
            return True

    return False


class TrackedObjectAttribute:
    def __init__(self, raw_data: tuple) -> None:
        self.label = raw_data[0]
        self.score = raw_data[1]
        self.box = raw_data[2]
        self.area = raw_data[3]
        self.ratio = raw_data[4]
        self.region = raw_data[5]

    def get_tracking_data(self) -> dict[str, Any]:
        """Return data saved to the object."""
        return {
            "label": self.label,
            "score": self.score,
            "box": self.box,
        }

    def find_best_object(self, objects: list[dict[str, Any]]) -> str | None:
        """Find the best attribute for each object and return its ID."""
        best_object_area: float | None = None
        best_object_id: str | None = None
        best_object_label: str | None = None

        for obj in objects:
            if not box_inside(obj["box"], self.box):
                continue

            object_area = area(obj["box"])

            # if multiple objects have the same attribute then they
            # are overlapping, it is most likely that the smaller object
            # is the one with the attribute
            if best_object_area is None:
                best_object_area = object_area
                best_object_id = obj["id"]
                best_object_label = obj["label"]
            else:
                if best_object_label == obj["label"]:
                    # if multiple objects of the same type are overlapping
                    # then the attribute will not be assigned
                    return None
                elif object_area < best_object_area:
                    # if a car and person are overlapping then assign the label to the smaller object (which should be the person)
                    best_object_area = object_area
                    best_object_id = obj["id"]
                    best_object_label = obj["label"]

        return best_object_id
