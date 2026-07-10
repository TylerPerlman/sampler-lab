"""Stable scalar normal-tail calculations used by rare-event diagnostics."""

from __future__ import annotations

import math

import numpy as np

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def standard_normal_log_upper_tail(x: float) -> float:
    """Return ``log P(Z >= x)`` for a standard normal ``Z``.

    ``erfc`` is accurate in the central region.  In the far positive tail an
    asymptotic Mills-ratio expansion avoids underflow.  Negative arguments are
    handled by complementing the corresponding positive-tail probability.
    """

    if not np.isfinite(x):
        if x == float("inf"):
            return float("-inf")
        if x == float("-inf"):
            return 0.0
        raise ValueError("x may not be nan")
    if x < 0.0:
        positive_log_tail = standard_normal_log_upper_tail(-x)
        if positive_log_tail == float("-inf"):
            return 0.0
        return float(math.log1p(-math.exp(positive_log_tail)))
    if x < 8.0:
        return float(math.log(0.5 * math.erfc(x / math.sqrt(2.0))))

    inverse_square = 1.0 / (x * x)
    term = 1.0
    series = 1.0
    previous_magnitude = abs(term)
    for order in range(1, 20):
        term *= -(2 * order - 1) * inverse_square
        magnitude = abs(term)
        if magnitude > previous_magnitude:
            break
        candidate = series + term
        if candidate <= 0.0:
            break
        series = candidate
        previous_magnitude = magnitude
    return float(-0.5 * x * x - math.log(x) - _LOG_SQRT_2PI + math.log(series))


def standard_normal_upper_tail(x: float) -> float:
    """Return ``P(Z >= x)`` while preserving the log-tail implementation."""

    log_value = standard_normal_log_upper_tail(x)
    return 0.0 if log_value < math.log(np.finfo(np.float64).tiny) else float(math.exp(log_value))


def log_add_exp(left: float, right: float) -> float:
    """Stable scalar ``log(exp(left) + exp(right))``."""

    if left == float("-inf"):
        return right
    if right == float("-inf"):
        return left
    maximum = max(left, right)
    return float(maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum)))


def log_cosh(x: float) -> float:
    """Stable scalar logarithm of ``cosh(x)``."""

    absolute = abs(x)
    return float(absolute + math.log1p(math.exp(-2.0 * absolute)) - math.log(2.0))
