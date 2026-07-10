"""Deterministic adaptation schedules and diminishing-step diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RobbinsMonroSchedule:
    """Power-law stochastic-approximation schedule.

    ``rate(t) = initial_rate / (offset + t + 1) ** exponent``.
    Exponents in ``(1/2, 1]`` satisfy the usual square-summable but not summable
    conditions used by Robbins--Monro algorithms.
    """

    initial_rate: float = 0.05
    exponent: float = 0.6
    offset: float = 10.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_rate) or self.initial_rate <= 0.0:
            raise ValueError("initial_rate must be positive and finite")
        if not np.isfinite(self.exponent) or not 0.5 < self.exponent <= 1.0:
            raise ValueError("exponent must lie in (0.5, 1]")
        if not np.isfinite(self.offset) or self.offset < 0.0:
            raise ValueError("offset must be nonnegative and finite")

    def rate(self, step: int) -> float:
        """Return the adaptation rate at zero-indexed ``step``."""

        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step < 0:
            raise ValueError("step must be nonnegative")
        return float(self.initial_rate / (self.offset + step + 1.0) ** self.exponent)

    def rates(self, n_steps: int) -> Array:
        """Return the first ``n_steps`` adaptation rates."""

        if isinstance(n_steps, bool) or not isinstance(n_steps, int):
            raise TypeError("n_steps must be an integer")
        if n_steps < 0:
            raise ValueError("n_steps must be nonnegative")
        indices = np.arange(1, n_steps + 1, dtype=np.float64)
        return np.asarray(
            self.initial_rate / (self.offset + indices) ** self.exponent,
            dtype=np.float64,
        )


def diminishing_ratio(rates: NDArray[np.float64]) -> float:
    """Return the final-to-initial ratio for a positive adaptation-rate sequence."""

    values = np.asarray(rates, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("rates must be a nonempty vector")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("rates must be positive and finite")
    return float(values[-1] / values[0])


def is_nonincreasing(values: NDArray[np.float64], *, tolerance: float = 0.0) -> bool:
    """Check whether a finite one-dimensional sequence is nonincreasing."""

    sequence = np.asarray(values, dtype=np.float64)
    if sequence.ndim != 1 or not np.all(np.isfinite(sequence)):
        raise ValueError("values must be a finite vector")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be nonnegative and finite")
    return bool(np.all(np.diff(sequence) <= tolerance))
