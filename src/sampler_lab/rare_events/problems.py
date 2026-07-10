"""Exactly solvable small-noise Gaussian rare-event problems."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.linalg import as_positive_definite
from sampler_lab.core.numerics import validate_size
from sampler_lab.rare_events.normal import (
    log_add_exp,
    standard_normal_log_upper_tail,
    standard_normal_upper_tail,
)

Array = NDArray[np.float64]


def _validate_epsilon(epsilon: float) -> float:
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    return float(epsilon)


@dataclass(slots=True)
class GaussianHalfspaceRareEvent:
    """Estimate ``P(a^T X >= b)`` for ``X ~ N(0, epsilon * C)``."""

    direction: ArrayLike
    threshold: float
    covariance: ArrayLike
    _direction: Array = field(init=False, repr=False)
    _covariance: Array = field(init=False, repr=False)
    _precision: Array = field(init=False, repr=False)
    _cholesky: Array = field(init=False, repr=False)
    _directional_variance: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._direction = np.asarray(self.direction, dtype=np.float64)
        if self._direction.ndim != 1 or self._direction.size == 0:
            raise ValueError("direction must be a nonempty vector")
        if not np.all(np.isfinite(self._direction)) or np.all(self._direction == 0.0):
            raise ValueError("direction must be finite and nonzero")
        if not np.isfinite(self.threshold) or self.threshold <= 0.0:
            raise ValueError("threshold must be positive and finite")
        self._covariance = as_positive_definite(self.covariance).copy()
        if self._covariance.shape != (self._direction.size, self._direction.size):
            raise ValueError("covariance shape must match direction dimension")
        self._cholesky = np.asarray(np.linalg.cholesky(self._covariance), dtype=np.float64)
        self._precision = np.asarray(
            np.linalg.solve(self._covariance, np.eye(self._direction.size)),
            dtype=np.float64,
        )
        self._directional_variance = float(self._direction @ self._covariance @ self._direction)

    @property
    def dimension(self) -> int:
        """Ambient dimension."""

        return int(self._direction.size)

    @property
    def direction_vector(self) -> Array:
        """Detached event normal vector."""

        return self._direction.copy()

    @property
    def covariance_matrix(self) -> Array:
        """Detached base covariance ``C`` before multiplication by ``epsilon``."""

        return self._covariance.copy()

    @property
    def precision_matrix(self) -> Array:
        """Detached inverse base covariance."""

        return self._precision.copy()

    @property
    def directional_variance(self) -> float:
        """Return ``a^T C a``."""

        return self._directional_variance

    @property
    def rate(self) -> float:
        """Large-deviation rate at the dominating boundary point."""

        return float(self.threshold**2 / (2.0 * self._directional_variance))

    @property
    def dominant_point(self) -> Array:
        """Minimum-rate point satisfying ``a^T x = b``."""

        return self.threshold * (self._covariance @ self._direction) / self._directional_variance

    def standardized_threshold(self, epsilon: float) -> float:
        """Return the one-dimensional normal threshold."""

        epsilon = _validate_epsilon(epsilon)
        return float(self.threshold / np.sqrt(epsilon * self._directional_variance))

    def exact_log_probability(self, epsilon: float) -> float:
        """Exact logarithmic rare-event probability."""

        return standard_normal_log_upper_tail(self.standardized_threshold(epsilon))

    def exact_probability(self, epsilon: float) -> float:
        """Exact rare-event probability."""

        return standard_normal_upper_tail(self.standardized_threshold(epsilon))

    def event(self, samples: ArrayLike) -> NDArray[np.bool_]:
        """Evaluate the halfspace event for a batch of samples."""

        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("samples must have shape (n, dimension)")
        return np.asarray(values @ self._direction >= self.threshold, dtype=np.bool_)

    def sample_target(
        self,
        rng: np.random.Generator,
        size: int,
        epsilon: float,
        *,
        counter: OperationCounter | None = None,
    ) -> Array:
        """Draw exact target samples using a stored Cholesky factor."""

        size = validate_size(size)
        if size == 0:
            return np.empty((0, self.dimension), dtype=np.float64)
        epsilon = _validate_epsilon(epsilon)
        normals = rng.normal(size=(size, self.dimension))
        if counter is not None:
            counter.increment("normal_draws", size * self.dimension)
        return np.asarray(np.sqrt(epsilon) * normals @ self._cholesky.T, dtype=np.float64)

    def log_target_density(self, samples: ArrayLike, epsilon: float) -> Array:
        """Evaluate the normalized target log density for a batch."""

        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("samples must have shape (n, dimension)")
        epsilon = _validate_epsilon(epsilon)
        sign, log_determinant = np.linalg.slogdet(self._covariance)
        if sign <= 0.0:  # pragma: no cover - guaranteed by construction
            raise RuntimeError("covariance determinant must be positive")
        quadratic = np.einsum("ni,ij,nj->n", values, self._precision, values)
        normalizer = 0.5 * (self.dimension * np.log(2.0 * np.pi * epsilon) + log_determinant)
        return np.asarray(-0.5 * quadratic / epsilon - normalizer, dtype=np.float64)


@dataclass(slots=True)
class GaussianTwoSidedRareEvent:
    """Estimate ``P(|a^T X| >= b)`` for ``X ~ N(0, epsilon * C)``."""

    direction: ArrayLike
    threshold: float
    covariance: ArrayLike
    _halfspace: GaussianHalfspaceRareEvent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._halfspace = GaussianHalfspaceRareEvent(
            direction=self.direction,
            threshold=self.threshold,
            covariance=self.covariance,
        )

    @property
    def dimension(self) -> int:
        return self._halfspace.dimension

    @property
    def direction_vector(self) -> Array:
        return self._halfspace.direction_vector

    @property
    def covariance_matrix(self) -> Array:
        return self._halfspace.covariance_matrix

    @property
    def precision_matrix(self) -> Array:
        return self._halfspace.precision_matrix

    @property
    def directional_variance(self) -> float:
        return self._halfspace.directional_variance

    @property
    def rate(self) -> float:
        return self._halfspace.rate

    @property
    def dominant_points(self) -> tuple[Array, Array]:
        point = self._halfspace.dominant_point
        return point, -point

    @property
    def dominant_point(self) -> Array:
        """Positive dominating point, useful for deliberate single-twist failure tests."""

        return self._halfspace.dominant_point

    def standardized_threshold(self, epsilon: float) -> float:
        return self._halfspace.standardized_threshold(epsilon)

    def exact_log_probability(self, epsilon: float) -> float:
        return float(math_log_two() + self._halfspace.exact_log_probability(epsilon))

    def exact_probability(self, epsilon: float) -> float:
        return float(2.0 * self._halfspace.exact_probability(epsilon))

    def event(self, samples: ArrayLike) -> NDArray[np.bool_]:
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("samples must have shape (n, dimension)")
        return np.asarray(
            np.abs(values @ self.direction_vector) >= self.threshold,
            dtype=np.bool_,
        )

    def sample_target(
        self,
        rng: np.random.Generator,
        size: int,
        epsilon: float,
        *,
        counter: OperationCounter | None = None,
    ) -> Array:
        return self._halfspace.sample_target(rng, size, epsilon, counter=counter)

    def log_target_density(self, samples: ArrayLike, epsilon: float) -> Array:
        return self._halfspace.log_target_density(samples, epsilon)

    def shifted_event_log_probability(self, mean: ArrayLike, epsilon: float) -> float:
        """Log event probability for ``N(mean, epsilon C)``."""

        mean_array = np.asarray(mean, dtype=np.float64)
        if mean_array.shape != (self.dimension,):
            raise ValueError("mean has the wrong shape")
        epsilon = _validate_epsilon(epsilon)
        projected_mean = float(self.direction_vector @ mean_array)
        scale = float(np.sqrt(epsilon * self.directional_variance))
        upper = standard_normal_log_upper_tail((self.threshold - projected_mean) / scale)
        lower = standard_normal_log_upper_tail((self.threshold + projected_mean) / scale)
        return log_add_exp(upper, lower)


def math_log_two() -> float:
    """Small helper kept separate to make the two-sided factor explicit in profiles."""

    return float(np.log(2.0))


RareGaussianProblem = GaussianHalfspaceRareEvent | GaussianTwoSidedRareEvent


__all__ = [
    "GaussianHalfspaceRareEvent",
    "GaussianTwoSidedRareEvent",
    "RareGaussianProblem",
]
