"""Validated interpolation schedules for annealed Monte Carlo methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True, init=False)
class AnnealingSchedule:
    """A strictly increasing path parameter running from zero to one.

    The array contains both endpoints. Consequently ``n_steps`` is one less
    than the number of stored values.
    """

    values: Array

    def __init__(self, values: ArrayLike) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or array.size < 2:
            raise ValueError("an annealing schedule must contain at least two values")
        if not np.all(np.isfinite(array)):
            raise ValueError("schedule values must be finite")
        if not np.isclose(array[0], 0.0, atol=1e-14, rtol=0.0):
            raise ValueError("an annealing schedule must start at zero")
        if not np.isclose(array[-1], 1.0, atol=1e-14, rtol=0.0):
            raise ValueError("an annealing schedule must end at one")
        if np.any(np.diff(array) <= 0.0):
            raise ValueError("schedule values must be strictly increasing")
        if np.any((array < 0.0) | (array > 1.0)):
            raise ValueError("schedule values must lie in [0, 1]")
        copied = np.array(array, dtype=np.float64, copy=True)
        copied[0] = 0.0
        copied[-1] = 1.0
        copied.setflags(write=False)
        object.__setattr__(self, "values", copied)

    @property
    def n_steps(self) -> int:
        """Number of annealing increments."""

        return int(self.values.size - 1)

    @property
    def increments(self) -> Array:
        """Successive path-parameter increments."""

        result = np.diff(self.values)
        result.setflags(write=False)
        return result

    @classmethod
    def linear(cls, n_steps: int) -> AnnealingSchedule:
        """Equally spaced path parameters."""

        steps = _validate_step_count(n_steps)
        return cls(np.linspace(0.0, 1.0, steps + 1, dtype=np.float64))

    @classmethod
    def power(cls, n_steps: int, exponent: float) -> AnnealingSchedule:
        """Return ``beta_k = (k / n_steps) ** exponent``.

        Exponents larger than one spend more stages near the initial law;
        exponents below one spend more stages near the final law.
        """

        steps = _validate_step_count(n_steps)
        if not np.isfinite(exponent) or exponent <= 0.0:
            raise ValueError("exponent must be finite and positive")
        grid = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
        return cls(grid ** float(exponent))


def _validate_step_count(n_steps: int) -> int:
    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise TypeError("n_steps must be an integer")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    return n_steps
