"""Reverse Norfair: backward object tracking to estimate pre-detection discovery delay.

When a new object is detected at ``t_first_detect``, Reverse Norfair walks
backward through recently buffered camera frames (newest -> oldest) using
Frigate's Norfair ``Tracker`` (``frigate_distance`` + ``OptimizedKalmanFilterFactory``)
and an asynchronous detector callable (e.g. ``SecondlookModel.infer``).

The earliest frame where the reverse tracker still matches the object is
``first_visible_time``, giving:
    discovery_ms = (first_detected_time - first_visible_time) * 1000.0
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np
from norfair import Detection, OptimizedKalmanFilterFactory, Tracker

from frigate.track.norfair_tracker import frigate_distance
from frigate.util.image import calculate_region

logger = logging.getLogger(__name__)

# Detector callable signature: (rgb_crop) -> (detections, latency_ms)
# Each detection dict: {"label": str, "score": float, "box_norm": [ymin, xmin, ymax, xmax]}
DetectorFn = Callable[[np.ndarray], tuple[list[dict[str, Any]], float]]


@dataclass
class ReverseNorfairResult:
    """Output of a backward Norfair walk for a single object."""

    first_visible_time: float
    discovery_ms: float
    frames_checked: int
    frames_matched: int
    earliest_box: list[int] | None
    preempted: bool = False


class CameraFrameRingBuffer:
    """Thread-safe per-camera bounded ring buffer of JPEG-compressed detect frames.

    Stores at most ``max_frames`` frames per camera sampled no faster than
    ``min_interval`` seconds (default ~5.5 fps -> ~6s lookback at ~1 MB RAM/cam).
    """

    def __init__(
        self,
        *,
        max_frames: int = 30,
        min_interval: float = 0.18,
        jpeg_quality: int = 75,
    ) -> None:
        self.max_frames = max(2, int(max_frames))
        self.min_interval = max(0.0, float(min_interval))
        self.jpeg_quality = int(jpeg_quality)
        self._lock = threading.Lock()
        self._buffers: dict[str, deque[tuple[float, bytes, int, int]]] = {}
        self._last_time: dict[str, float] = {}

    def push_yuv(self, camera: str, frame_time: float, yuv_frame: np.ndarray) -> bool:
        """Compress and append a YUV I420 frame if ``min_interval`` has elapsed."""
        ts = float(frame_time)
        with self._lock:
            prev = self._last_time.get(camera)
            if prev is not None and (ts - prev) < self.min_interval:
                return False
            self._last_time[camera] = ts

        try:
            bgr = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_I420)
            height, width = bgr.shape[:2]
            ok, encoded = cv2.imencode(
                ".jpg",
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                return False
            jpeg_bytes = encoded.tobytes()
        except Exception:
            logger.debug("Reverse Norfair frame buffer encode failed", exc_info=True)
            return False

        with self._lock:
            buf = self._buffers.get(camera)
            if buf is None:
                buf = deque(maxlen=self.max_frames)
                self._buffers[camera] = buf
            buf.append((ts, jpeg_bytes, width, height))
        return True

    def push_rgb(self, camera: str, frame_time: float, rgb_frame: np.ndarray) -> bool:
        """Compress and append an RGB frame (primarily for testing/non-YUV callers)."""
        ts = float(frame_time)
        with self._lock:
            prev = self._last_time.get(camera)
            if prev is not None and (ts - prev) < self.min_interval:
                return False
            self._last_time[camera] = ts

        try:
            bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            height, width = bgr.shape[:2]
            ok, encoded = cv2.imencode(
                ".jpg",
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                return False
            jpeg_bytes = encoded.tobytes()
        except Exception:
            logger.debug("Reverse Norfair RGB frame encode failed", exc_info=True)
            return False

        with self._lock:
            buf = self._buffers.get(camera)
            if buf is None:
                buf = deque(maxlen=self.max_frames)
                self._buffers[camera] = buf
            buf.append((ts, jpeg_bytes, width, height))
        return True

    def snapshot_before(
        self,
        camera: str,
        before_time: float,
        *,
        lookback_sec: float = 6.0,
    ) -> list[tuple[float, bytes, int, int]]:
        """Return buffered frames strictly before ``before_time``, newest first."""
        cutoff = float(before_time) - max(0.1, float(lookback_sec))
        with self._lock:
            buf = self._buffers.get(camera)
            if not buf:
                return []
            items = [
                entry
                for entry in buf
                if cutoff <= entry[0] < float(before_time) - 1e-4
            ]
        items.sort(key=lambda entry: entry[0], reverse=True)
        return items


def _box_to_points(box: list[int] | tuple[int, int, int, int]) -> np.ndarray:
    return np.array([[float(box[0]), float(box[1])], [float(box[2]), float(box[3])]])


def _estimate_to_box(
    estimate: np.ndarray, width: int, height: int
) -> list[int]:
    x1 = int(max(0, min(width - 1, round(float(estimate[0][0])))))
    y1 = int(max(0, min(height - 1, round(float(estimate[0][1])))))
    x2 = int(max(x1 + 2, min(width, round(float(estimate[1][0])))))
    y2 = int(max(y1 + 2, min(height, round(float(estimate[1][1])))))
    return [x1, y1, x2, y2]


def _crop_rgb_for_box(
    rgb_frame: np.ndarray,
    box: list[int],
    *,
    input_size: int = 640,
    context_scale: float = 2.5,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    height, width = rgb_frame.shape[:2]
    xmin, ymin, xmax, ymax = box
    region = calculate_region(
        (height, width),
        xmin,
        ymin,
        xmax,
        ymax,
        input_size,
        context_scale,
    )
    rx1, ry1, rx2, ry2 = (int(v) for v in region)
    if rx2 - rx1 < 4 or ry2 - ry1 < 4:
        return None
    crop = rgb_frame[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return None
    return np.ascontiguousarray(crop), (rx1, ry1, rx2, ry2)


def run_reverse_norfair(
    *,
    frames_rev: list[tuple[float, bytes | np.ndarray, int, int]],
    label: str,
    first_detect_time: float,
    seed_box: list[int],
    detector_fn: DetectorFn,
    seed_box_later: list[int] | None = None,
    input_size: int = 640,
    context_scale: float = 2.5,
    min_score: float = 0.25,
    distance_threshold: float = 2.5,
    max_missed_frames: int = 3,
    should_preempt: Callable[[], bool] | None = None,
) -> ReverseNorfairResult:
    """Walk backward in time with Norfair to find when ``label`` first appeared.

    Args:
        frames_rev: Frames strictly before ``first_detect_time``, ordered newest
            to oldest. Each item is ``(frame_time, jpeg_bytes_or_rgb_ndarray, width, height)``.
        label: Object class label to match (e.g. ``"person"``).
        first_detect_time: Timestamp of the earliest forward primary hit.
        seed_box: ``[xmin, ymin, xmax, ymax]`` pixel box at ``first_detect_time``.
        detector_fn: Callable returning ``(detections, latency_ms)`` on an RGB crop.
        seed_box_later: Optional later box (e.g. at ``track_start_time``) fed
            first so Norfair's Kalman filter initializes with reverse velocity.
        input_size: Square model input size for crop sizing.
        context_scale: Context multiplier around the predicted box.
        min_score: Minimum detector confidence to accept during reverse search.
        distance_threshold: Norfair ``frigate_distance`` threshold.
        max_missed_frames: Stop walking backward after this many consecutive misses.
        should_preempt: Optional callback checked before each frame; returns
            True to abort early so live inference is never delayed.
    """
    earliest_time = float(first_detect_time)
    earliest_box: list[int] | None = [int(v) for v in seed_box]
    if not frames_rev:
        return ReverseNorfairResult(
            first_visible_time=earliest_time,
            discovery_ms=0.0,
            frames_checked=0,
            frames_matched=0,
            earliest_box=earliest_box,
        )

    tracker = Tracker(
        distance_function=frigate_distance,
        distance_threshold=distance_threshold,
        initialization_delay=0,
        hit_counter_max=max(1, int(max_missed_frames)),
        filter_factory=OptimizedKalmanFilterFactory(R=3.4),
    )

    # Seed reverse tracker (if a later box is known, feed it first so Kalman
    # velocity naturally points backward toward where the object came from).
    if seed_box_later is not None and seed_box_later != seed_box:
        tracker.update(
            detections=[
                Detection(
                    points=_box_to_points(seed_box_later),
                    label=label,
                    data={"box": [int(v) for v in seed_box_later], "score": 1.0},
                )
            ]
        )

    tracked = tracker.update(
        detections=[
            Detection(
                points=_box_to_points(seed_box),
                label=label,
                data={"box": [int(v) for v in seed_box], "score": 1.0},
            )
        ]
    )
    if not tracked:
        return ReverseNorfairResult(
            first_visible_time=earliest_time,
            discovery_ms=0.0,
            frames_checked=0,
            frames_matched=0,
            earliest_box=earliest_box,
        )

    target_id = tracked[0].id
    frames_checked = 0
    frames_matched = 0
    consecutive_misses = 0
    preempted = False

    for frame_ts, frame_payload, width, height in frames_rev:
        if should_preempt is not None and should_preempt():
            preempted = True
            break

        target = next((t for t in tracked if t.id == target_id), None)
        if target is None:
            break

        pred_box = _estimate_to_box(target.estimate, width, height)

        if isinstance(frame_payload, (bytes, bytearray)):
            buf = np.frombuffer(frame_payload, dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame_payload

        crop_res = _crop_rgb_for_box(
            rgb_frame,
            pred_box,
            input_size=input_size,
            context_scale=context_scale,
        )
        if crop_res is None:
            continue
        rgb_crop, (rx1, ry1, rx2, ry2) = crop_res
        crop_w = float(rx2 - rx1)
        crop_h = float(ry2 - ry1)

        frames_checked += 1
        try:
            raw_dets, _ = detector_fn(rgb_crop)
        except Exception:
            logger.debug("Reverse Norfair detector_fn failed", exc_info=True)
            break

        norfair_dets: list[Detection] = []
        for d in raw_dets:
            if d.get("label") != label:
                continue
            score = float(d.get("score") or 0.0)
            if score < min_score:
                continue
            box_norm = d.get("box_norm")
            if not box_norm or len(box_norm) < 4:
                continue
            # box_norm is [ymin, xmin, ymax, xmax] normalized in 0..1 of crop
            dy1, dx1, dy2, dx2 = (float(v) for v in box_norm[:4])
            fx1 = int(max(0, min(width - 1, round(rx1 + dx1 * crop_w))))
            fy1 = int(max(0, min(height - 1, round(ry1 + dy1 * crop_h))))
            fx2 = int(max(fx1 + 1, min(width, round(rx1 + dx2 * crop_w))))
            fy2 = int(max(fy1 + 1, min(height, round(ry1 + dy2 * crop_h))))
            box_px = [fx1, fy1, fx2, fy2]
            norfair_dets.append(
                Detection(
                    points=_box_to_points(box_px),
                    label=label,
                    data={"box": box_px, "score": score, "frame_time": frame_ts},
                )
            )

        tracked = tracker.update(detections=norfair_dets)
        target = next((t for t in tracked if t.id == target_id), None)
        if (
            target is not None
            and target.last_detection is not None
            and target.last_detection.data.get("frame_time") == frame_ts
        ):
            earliest_time = float(frame_ts)
            earliest_box = list(target.last_detection.data["box"])
            frames_matched += 1
            consecutive_misses = 0
        else:
            consecutive_misses += 1
            if consecutive_misses >= max_missed_frames:
                break

    discovery_ms = max(0.0, (float(first_detect_time) - earliest_time) * 1000.0)
    return ReverseNorfairResult(
        first_visible_time=earliest_time,
        discovery_ms=round(discovery_ms, 2),
        frames_checked=frames_checked,
        frames_matched=frames_matched,
        earliest_box=earliest_box,
        preempted=preempted,
    )
