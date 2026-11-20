"""Pure false_positive check: does a tracked object become a detection?

Applies the Secondlook hold and the factor-adjusted median of the score
buffer before detection.
"""

from __future__ import annotations

from statistics import median

from .factor import adjusted_median_scores


def decide_false_positive(
    *,
    already_true_positive: bool,
    secondlook_enabled: bool,
    score_history: list[float],
    threshold: float,
    factor: float | None,
    pending_since: float | None,
    frame_time: float,
    hold_seconds: float,
    running: bool = False,
    pending: bool = False,
    hold_done: bool = False,
) -> tuple[bool, float | None]:
    """Return (is_false_positive, new_pending_since).

    Hold-while-null runs once per track: it arms on the first false_positive check with
    no factor and lasts at most ``hold_seconds`` of frame time. It ends early
    when any SL result arrives (caller sets ``hold_done``) and never re-arms
    after it ends, so a no-hit / fail-open result does not cause repeated
    holds. ``new_pending_since`` is None once the hold is over; the caller
    should then mark the hold done.

    ``running`` / ``pending`` are informational (the processor's view); the
    hold does not depend on them because the hold arms before the pending
    payload can arrive from the embeddings process.
    """
    if already_true_positive:
        return False, pending_since

    computed = median(score_history) if score_history else 0.0

    if not secondlook_enabled:
        return computed < threshold, pending_since

    if factor is not None:
        adjusted = adjusted_median_scores(score_history, float(factor))
        return median(adjusted) < threshold, None

    if hold_done:
        # fail-open: unscaled primary vs stock threshold
        return computed < threshold, None

    new_pending = frame_time if pending_since is None else pending_since
    if (frame_time - new_pending) <= hold_seconds:
        return True, new_pending

    # timed out → fail-open, clear pending (caller marks hold done)
    return computed < threshold, None
