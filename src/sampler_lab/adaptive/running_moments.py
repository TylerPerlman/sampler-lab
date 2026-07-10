"""Stable online means and covariance matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


def _as_vector(value: ArrayLike, *, dimension: int | None = None) -> Array:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("observations must be nonempty finite vectors")
    if dimension is not None and vector.size != dimension:
        raise ValueError("observation dimension changed")
    return vector


@dataclass(frozen=True, slots=True)
class RunningMomentsSnapshot:
    """Immutable snapshot of online vector moments."""

    count: int
    mean: Array
    covariance: Array


class RunningMoments:
    """Welford-style online mean and full sample covariance."""

    def __init__(self, dimension: int) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError("dimension must be an integer")
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._count = 0
        self._mean = np.zeros(dimension, dtype=np.float64)
        self._m2 = np.zeros((dimension, dimension), dtype=np.float64)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> Array:
        return self._mean.copy()

    def update(self, observation: ArrayLike) -> None:
        """Incorporate one observation."""

        value = _as_vector(observation, dimension=self._dimension)
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta_after = value - self._mean
        self._m2 += np.outer(delta, delta_after)

    def update_batch(self, observations: ArrayLike) -> None:
        """Incorporate a two-dimensional batch in row order."""

        values = np.asarray(observations, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self._dimension:
            raise ValueError("observations must have shape (n, dimension)")
        if not np.all(np.isfinite(values)):
            raise ValueError("observations must be finite")
        for value in values:
            self.update(value)

    def covariance(self, *, ddof: int = 1) -> Array:
        """Return covariance with the requested degrees-of-freedom correction."""

        if isinstance(ddof, bool) or not isinstance(ddof, int):
            raise TypeError("ddof must be an integer")
        denominator = self._count - ddof
        if denominator <= 0:
            raise ValueError("not enough observations for the requested ddof")
        covariance = self._m2 / denominator
        return np.asarray(0.5 * (covariance + covariance.T), dtype=np.float64)

    def population_covariance(self) -> Array:
        """Return the population covariance of observations seen so far."""

        return self.covariance(ddof=0)

    def snapshot(self, *, ddof: int = 1) -> RunningMomentsSnapshot:
        """Return a detached immutable snapshot."""

        covariance = self.covariance(ddof=ddof)
        mean = self.mean
        mean.setflags(write=False)
        covariance.setflags(write=False)
        return RunningMomentsSnapshot(self._count, mean, covariance)
