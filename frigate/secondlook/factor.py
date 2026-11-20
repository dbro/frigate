"""Pure factor helpers for Secondlook live score adjustment."""

from __future__ import annotations


def clamp_factor(value: float, lo: float = 0.3, hi: float = 2.0) -> float:
    """Clamp a raw scale factor into [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def compute_factor(
    sl_max: float,
    primary_max: float,
    *,
    factor_min: float = 0.3,
    factor_max: float = 2.0,
) -> float | None:
    """Return clamp(sl_max / primary_max) or None when inputs are invalid.

    primary_max must be a real hit (> 0). A missing / weak SL hit should not
    call this — leave factor unset so the false_positive check treats it as 1.0.
    """
    if primary_max <= 0 or sl_max <= 0:
        return None
    return clamp_factor(sl_max / primary_max, factor_min, factor_max)


def adjusted_median_scores(
    score_history: list[float],
    factor: float | None,
) -> list[float]:
    """Return scores for median: raw * factor when factor is set, else raw.

    Miss-pads (0.0) are left as 0.0 — do not specially multiply them.
    """
    if factor is None:
        return list(score_history)
    return [s * factor for s in score_history]
