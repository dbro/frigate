"""Review items: did the live pipeline act on Secondlook? Read-time Event lookup.

An Event row (Event.id == track id) exists once Frigate released the object as
a detection. Frigate's DB is a separate SQLite file, so this is a lookup when
items are read, not a stored column or foreign key.

  false_negative + event -> rescued      false_negative, no event -> not_rescued
  false_positive + event -> not_suppressed   false_positive, no event -> suppressed
  misclass -> none

"none" also when the answer is not reliable: the item is older than the
shortest Event retention for its camera/label (its Event may already be
gone), the Event is still in progress, or the item itself recorded a release
(event_id) whose Event is gone (deleted by Event cleanup).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

DAY = 86400.0
OUTCOMES = {
    ("false_negative", True): "rescued",
    ("false_negative", False): "not_rescued",
    ("false_positive", True): "not_suppressed",
    ("false_positive", False): "suppressed",
}


def event_retention_days(camera_config: Any, label: str) -> float:
    """Shortest time an Event of this camera/label is kept (0 = not reliably).

    Event cleanup keeps a row while it has a clip or a snapshot; take the
    shortest of the enabled ones, since either may be the only one it had.
    """
    days = []
    snapshots = getattr(camera_config, "snapshots", None)
    if snapshots is not None and snapshots.enabled:
        days.append(float(snapshots.retain.objects.get(label, snapshots.retain.default)))
    record = getattr(camera_config, "record", None)
    if record is not None and record.enabled:
        days.append(float(min(record.alerts.retain.days, record.detections.retain.days)))
    return min(days) if days else 0.0


def _events(ids: list[str]) -> dict[str, float | None]:
    """{event id: end_time} from Frigate's Event table."""
    from frigate.models import Event

    out: dict[str, float | None] = {}
    for i in range(0, len(ids), 500):
        for event in Event.select(Event.id, Event.end_time).where(
            Event.id.in_(ids[i : i + 500])
        ):
            out[event.id] = event.end_time
    return out


def live_outcomes(
    config: Any,
    records: Iterable[Any],
    now: float | None = None,
    lookup: Callable[[list[str]], dict[str, float | None]] = _events,
) -> dict[str, str]:
    """{item id: rescued | not_rescued | suppressed | not_suppressed | none}."""
    now = time.time() if now is None else now
    records = list(records)
    out = {r.id: "none" for r in records}
    candidates = []
    for r in records:
        camera = config.cameras.get(r.camera)
        if r.kind not in ("false_negative", "false_positive") or camera is None:
            continue
        days = event_retention_days(camera, r.label)
        when = r.frame_time if r.frame_time is not None else r.created_at
        if days > 0 and when >= now - days * DAY:
            candidates.append(r)
    if not candidates:
        return out
    try:
        events = lookup(sorted({r.track_id for r in candidates}))
    except Exception:
        logger.debug("Secondlook live outcome: Event lookup failed", exc_info=True)
        return out
    for r in candidates:
        released = r.track_id in events
        if released and events[r.track_id] is None:
            continue  # in progress
        if not released and r.event_id:
            continue  # it was released; its Event has been cleaned up
        out[r.id] = OUTCOMES[(r.kind, released)]
    return out
