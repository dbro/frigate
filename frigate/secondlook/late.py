"""Ended tracks still waiting for their Secondlook answer.

Analysis runs in the embeddings process; a track can end before its answer
reaches the main process. Such a track is parked here (a reference; frames are
in the camera's shared cache) and finished when the answer arrives, or without
it after LATE_TTL_SECONDS. Both paths run on the TrackedObjectProcessor thread.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

LATE_TTL_SECONDS = 15.0
MAX_PARKED = 500

_parked: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()


def park(object_id: str, obj: Any, now: float | None = None) -> None:
    _parked[object_id] = (time.monotonic() if now is None else now, obj)


def take(object_id: str) -> Any | None:
    entry = _parked.pop(object_id, None)
    return entry[1] if entry else None


def expired(now: float | None = None) -> list[Any]:
    """Pop tracks parked too long (oldest first), or beyond MAX_PARKED."""
    now = time.monotonic() if now is None else now
    out = []
    while _parked:
        object_id, (since, obj) = next(iter(_parked.items()))
        if now - since < LATE_TTL_SECONDS and len(_parked) <= MAX_PARKED:
            break
        _parked.popitem(last=False)
        out.append(obj)
    return out
