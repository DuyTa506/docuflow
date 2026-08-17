"""Shared live-rate smoothing and publication guards."""

from __future__ import annotations

from typing import Optional


def ema(previous: Optional[float], sample: float, alpha: float) -> float:
    if previous is None:
        return max(0.0, sample)
    alpha = max(0.0, min(1.0, alpha))
    return alpha * max(0.0, sample) + (1.0 - alpha) * max(0.0, previous)


def clamp_step(previous: Optional[float], candidate: float, max_ratio: float) -> float:
    if previous is None or previous <= 0:
        return max(0.0, candidate)
    ratio = max(0.0, max_ratio)
    return min(previous * (1 + ratio), max(previous * (1 - ratio), candidate))


def exceeds_hysteresis(
    previous: Optional[float],
    candidate: float,
    *,
    ratio: float,
    seconds: float,
) -> bool:
    if previous is None:
        return True
    return abs(candidate - previous) >= max(abs(previous) * max(0.0, ratio), max(0.0, seconds))


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * max(0.0, min(1.0, quantile))
    lo = int(pos)
    hi = min(len(ordered) - 1, lo + 1)
    fraction = pos - lo
    return ordered[lo] * (1 - fraction) + ordered[hi] * fraction


def winsorize(values: list[float], lower: float = 0.05, upper: float = 0.95) -> list[float]:
    if len(values) < 4:
        return list(values)
    floor = percentile(values, lower)
    ceiling = percentile(values, upper)
    return [min(ceiling, max(floor, value)) for value in values]
