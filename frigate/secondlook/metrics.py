"""Secondlook metrics: last real worker sample, per-camera rates and saved-data disk use.

No per-model process-resource claims: CPU/memory belong to the shared embeddings
process and temperature is device-wide.
"""

import os
import threading
import time

# Per-camera counters are cumulative in the worker; rates are derived here from
# consecutive stats snapshots (one StatsEmitter thread calls stats_snapshot).
_rate_lock = threading.Lock()
_previous_counts: dict[str, tuple[float, int, int]] = {}

# Corrected decisions since Frigate started (live, reset on restart). Incremented
# by the TrackedObjectProcessor thread and read by the StatsEmitter thread; both
# live in the main Frigate process, so no IPC is needed (unlike queue_dropped,
# which comes from the embeddings process through DataProcessorMetrics).
ACCURACY_KINDS = ("false_positive", "false_negative", "misclass")
# Did the live pipeline act on a disagreement? Subsets of FN and FP:
# false_negative = fn_rescued + fn_not_rescued, false_positive =
# fp_suppressed + fp_not_suppressed (both counted once, when final).
OUTCOMES = ("fn_rescued", "fn_not_rescued", "fp_suppressed", "fp_not_suppressed")
_accuracy = dict.fromkeys((*ACCURACY_KINDS, *OUTCOMES), 0)
_accuracy_lock = threading.Lock()


def outcome_kind(outcome: str) -> str:
    return "false_negative" if outcome.startswith("fn_") else "false_positive"


def count_correction(kind: str, outcome: str | None = None) -> None:
    with _accuracy_lock:
        _accuracy[kind] += 1
        if outcome is not None:
            _accuracy[outcome] += 1


def _round2(value):
    return None if value is None else round(float(value), 2)


_STORAGE_TTL_SECONDS = 60.0
_storage_cache: dict[tuple[str, str], tuple[float, dict]] = {}


def secondlook_stats(config, metrics):
    enabled = bool(config.secondlook.enabled)
    values = dict(getattr(metrics, "secondlook", {})) if metrics else {}
    available = enabled and bool(values.get("available", False))
    sampled = available and values.get("samples", 0) > 0
    stats = {
        "enabled": enabled,
        "available": available,
        "samples": values.get("samples", 0) if enabled else 0,
        "last_updated": values.get("last_updated") if sampled else None,
        # 2 decimals, like detector inference_speed
        **{
            key: _round2(values.get(key)) if sampled else None
            for key in ("inference_ms", "processing_ms", "queue_ms")
        },
        "device": values.get("device") if enabled else None,
        "model_source": values.get("model_source") if enabled else None,
        "model_refusal": values.get("model_refusal") if enabled else None,
        "resource_scope": "shared embeddings process; temperature is device-wide",
        "queue_dropped": values.get("queue_dropped", 0),
        "queue_items": values.get("queue_items", 0) if available else None,
        "queue_max": config.secondlook.queue_max,
        "interval": _interval(values) if available else None,
        "results_dropped": values.get("results_dropped", 0),
    }
    if enabled:
        with _accuracy_lock:
            stats["accuracy"] = dict(_accuracy)
    if config.secondlook.save_and_review.enabled:
        from frigate.secondlook.save_and_review import get_save_and_review_store

        store = get_save_and_review_store()
        stats["storage"] = {
            **saved_data_usage(
                config.secondlook.save_and_review.database.path,
                config.secondlook.save_and_review.images.path,
            ),
            # last cleanup run in this Frigate process (None until it ran)
            "retention": store.last_retention if store else None,
        }
    return stats


_previous_interval: tuple[int, int, int, dict[int, int]] | None = None


def _interval(values) -> dict | None:
    """Since the previous stats snapshot: analyses completed, queue drops, and
    peak items (waiting + in process; 1 = busy without queuing, 0 = idle).

    Called once per snapshot by the StatsEmitter thread; None on the first one.
    """
    global _previous_interval
    samples = int(values.get("samples", 0))
    dropped = int(values.get("queue_dropped", 0))
    items = int(values.get("queue_items", 0))
    rises = dict(values.get("queue_rises") or {})
    with _rate_lock:
        previous, _previous_interval = (
            _previous_interval,
            (samples, dropped, items, rises),
        )
    if previous is None:
        return None
    risen = [level for level, n in rises.items() if n > previous[3].get(level, 0)]
    return {
        "completed": max(0, samples - previous[0]),
        "dropped": max(0, dropped - previous[1]),
        "queue_peak": max([previous[2], items, *risen]),
    }


def camera_rates(config, metrics, now=None):
    """Per-camera SL analyses/s and skipped/s since the previous snapshot.

    analysed: real (non-mock) SL inferences completed for that camera.
    skipped: SL analyses requested by the processor but NOT performed, because the
    bounded worker queue was full (frame-1 request fails open to the unscaled
    primary; refresh request keeps the prior factor) or the crop could not be
    taken. Frames/objects Secondlook never requests (stationary, label not
    configured, no primary hit, between refreshes) are not counted as skipped.
    """
    if not config.secondlook.enabled or metrics is None:
        return {}
    values = dict(getattr(metrics, "secondlook", {}))
    if not values.get("available"):
        return {}
    counts = dict(values.get("cameras") or {})
    now = time.monotonic() if now is None else now
    rates = {}
    with _rate_lock:
        for name, camera in config.cameras.items():
            if not camera.enabled or not camera.secondlook.enabled:
                _previous_counts.pop(name, None)
                continue
            current = counts.get(name) or {}
            analysed = int(current.get("analysed", 0))
            skipped = int(current.get("skipped", 0))
            previous = _previous_counts.get(name)
            _previous_counts[name] = (now, analysed, skipped)
            if previous is None or now <= previous[0]:
                continue
            elapsed = now - previous[0]
            rates[name] = {
                "secondlook_analysed_ps": round(
                    max(0, analysed - previous[1]) / elapsed, 2
                ),
                "secondlook_skipped_ps": round(
                    max(0, skipped - previous[2]) / elapsed, 2
                ),
            }
    return rates


def _file_bytes(path):
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


def _directory_usage(path):
    total = files = 0
    pending = [path]
    while pending:
        try:
            entries = os.scandir(pending.pop())
        except OSError:
            continue
        with entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                        files += 1
                except OSError:
                    continue
    return total, files


def saved_data_usage(db_path, images_path, now=None):
    """Bytes used by the separate SQLite store (+WAL/SHM) and JPEG directory.

    Cached for 60 s so the 15 s stats loop does not rescan large directories.
    """
    now = time.monotonic() if now is None else now
    key = (db_path, images_path)
    cached = _storage_cache.get(key)
    if cached and now - cached[0] < _STORAGE_TTL_SECONDS:
        return cached[1]
    images_bytes, image_files = _directory_usage(images_path)
    usage = {
        "database_bytes": sum(
            _file_bytes(db_path + suffix) for suffix in ("", "-wal", "-shm")
        ),
        "images_bytes": images_bytes,
        "image_files": image_files,
    }
    _storage_cache[key] = (now, usage)
    return usage
