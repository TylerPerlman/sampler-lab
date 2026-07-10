"""Linear exponential twisting for small-noise Gaussian rare events."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.linalg import as_positive_definite
from sampler_lab.core.numerics import validate_size
from sampler_lab.rare_events.normal import log_add_exp, standard_normal_log_upper_tail
from sampler_lab.rare_events.problems import (
    GaussianHalfspaceRareEvent,
    RareGaussianProblem,
)
from sampler_lab.rare_events.relative_error import (
    RareEventEstimate,
    estimate_from_log_contributions,
)

Array = NDArray[np.float64]


def _validate_epsilon(epsilon: float) -> float:
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    return float(epsilon)


@dataclass(slots=True)
class GaussianShiftProposal:
    """Gaussian proposal ``N(m, epsilon C)`` for target ``N(0, epsilon C)``."""

    mean_shift: ArrayLike
    covariance: ArrayLike
    epsilon: float
    _mean: Array = field(init=False, repr=False)
    _covariance: Array = field(init=False, repr=False)
    _precision: Array = field(init=False, repr=False)
    _cholesky: Array = field(init=False, repr=False)
    _log_determinant: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._mean = np.asarray(self.mean_shift, dtype=np.float64)
        if self._mean.ndim != 1 or self._mean.size == 0:
            raise ValueError("mean_shift must be a nonempty vector")
        if not np.all(np.isfinite(self._mean)):
            raise ValueError("mean_shift must be finite")
        self._covariance = as_positive_definite(self.covariance).copy()
        if self._covariance.shape != (self._mean.size, self._mean.size):
            raise ValueError("covariance shape must match mean_shift")
        self.epsilon = _validate_epsilon(self.epsilon)
        self._precision = np.asarray(
            np.linalg.solve(self._covariance, np.eye(self._mean.size)),
            dtype=np.float64,
        )
        self._cholesky = np.asarray(np.linalg.cholesky(self._covariance), dtype=np.float64)
        sign, log_determinant = np.linalg.slogdet(self._covariance)
        if sign <= 0.0:  # pragma: no cover
            raise RuntimeError("covariance determinant must be positive")
        self._log_determinant = float(log_determinant)

    @property
    def dimension(self) -> int:
        return int(self._mean.size)

    @property
    def mean_vector(self) -> Array:
        return self._mean.copy()

    @property
    def covariance_matrix(self) -> Array:
        return self._covariance.copy()

    def sample(
        self,
        rng: np.random.Generator,
        size: int,
        *,
        counter: OperationCounter | None = None,
    ) -> Array:
        """Draw proposal samples."""

        size = validate_size(size)
        if size == 0:
            return np.empty((0, self.dimension), dtype=np.float64)
        noise = rng.normal(size=(size, self.dimension))
        if counter is not None:
            counter.increment("normal_draws", size * self.dimension)
        return np.asarray(
            self._mean + np.sqrt(self.epsilon) * noise @ self._cholesky.T,
            dtype=np.float64,
        )

    def log_density(self, samples: ArrayLike) -> Array:
        """Evaluate normalized proposal log densities."""

        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("samples must have shape (n, dimension)")
        centered = values - self._mean
        quadratic = np.einsum("ni,ij,nj->n", centered, self._precision, centered)
        normalizer = 0.5 * (
            self.dimension * math.log(2.0 * math.pi * self.epsilon) + self._log_determinant
        )
        return np.asarray(-0.5 * quadratic / self.epsilon - normalizer, dtype=np.float64)

    def target_log_density(self, samples: ArrayLike) -> Array:
        """Evaluate the matching centered target density."""

        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("samples must have shape (n, dimension)")
        quadratic = np.einsum("ni,ij,nj->n", values, self._precision, values)
        normalizer = 0.5 * (
            self.dimension * math.log(2.0 * math.pi * self.epsilon) + self._log_determinant
        )
        return np.asarray(-0.5 * quadratic / self.epsilon - normalizer, dtype=np.float64)

    def log_weights(self, samples: ArrayLike) -> Array:
        """Return exact target-to-proposal log weights."""

        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("samples must have shape (n, dimension)")
        linear = values @ (self._precision @ self._mean)
        quadratic = float(self._mean @ self._precision @ self._mean)
        return np.asarray(-linear / self.epsilon + 0.5 * quadratic / self.epsilon)


def optimal_linear_shift(problem: RareGaussianProblem) -> Array:
    """Return one dominating point for a Gaussian linear rare event."""

    return problem.dominant_point.copy()


def exact_shifted_log_second_moment(
    problem: RareGaussianProblem,
    mean_shift: ArrayLike,
    epsilon: float,
) -> float:
    """Exact second moment of the shifted-Gaussian importance contribution."""

    epsilon = _validate_epsilon(epsilon)
    mean = np.asarray(mean_shift, dtype=np.float64)
    if mean.shape != (problem.dimension,):
        raise ValueError("mean_shift has the wrong shape")
    precision = problem.precision_matrix
    exponential_factor = float(mean @ precision @ mean / epsilon)
    projected_mean = float(problem.direction_vector @ mean)
    scale = math.sqrt(epsilon * problem.directional_variance)

    upper = standard_normal_log_upper_tail((problem.threshold + projected_mean) / scale)
    if isinstance(problem, GaussianHalfspaceRareEvent):
        event_log_probability = upper
    else:
        lower = standard_normal_log_upper_tail((problem.threshold - projected_mean) / scale)
        event_log_probability = log_add_exp(upper, lower)
    return float(exponential_factor + event_log_probability)


def estimate_with_shift(
    problem: RareGaussianProblem,
    proposal: GaussianShiftProposal,
    rng: np.random.Generator,
    size: int,
    *,
    counter: OperationCounter | None = None,
) -> RareEventEstimate:
    """Estimate a Gaussian rare-event probability under a shifted proposal."""

    if proposal.dimension != problem.dimension:
        raise ValueError("proposal and problem dimensions differ")
    if not np.allclose(proposal.covariance_matrix, problem.covariance_matrix):
        raise ValueError("proposal and problem base covariances differ")
    size = validate_size(size)
    if size == 0:
        raise ValueError("size must be positive")
    samples = proposal.sample(rng, size, counter=counter)
    event = problem.event(samples)
    log_weights = proposal.log_weights(samples)
    if counter is not None:
        counter.increment("log_density_evaluations", size)
        counter.increment("proposal_density_evaluations", size)
    log_contributions = np.where(event, log_weights, float("-inf"))
    return estimate_from_log_contributions(
        log_contributions,
        event_count=int(np.sum(event)),
        counter=counter,
    )


__all__ = [
    "GaussianShiftProposal",
    "estimate_with_shift",
    "exact_shifted_log_second_moment",
    "optimal_linear_shift",
]
