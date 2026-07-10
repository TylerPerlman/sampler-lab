"""Gaussian temperature broadening for small-noise rare-event importance sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.linalg import as_positive_definite
from sampler_lab.core.numerics import validate_size
from sampler_lab.rare_events.normal import standard_normal_log_upper_tail
from sampler_lab.rare_events.problems import (
    GaussianTwoSidedRareEvent,
    RareGaussianProblem,
)
from sampler_lab.rare_events.relative_error import (
    RareEventEstimate,
    estimate_from_log_contributions,
)

Array = NDArray[np.float64]


@dataclass(slots=True)
class GaussianTemperedProposal:
    """Proposal ``N(0, temperature * epsilon * C)`` for ``N(0, epsilon C)``."""

    covariance: ArrayLike
    epsilon: float
    temperature: float
    _covariance: Array = field(init=False, repr=False)
    _precision: Array = field(init=False, repr=False)
    _cholesky: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._covariance = as_positive_definite(self.covariance).copy()
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive and finite")
        if not np.isfinite(self.temperature) or self.temperature < 1.0:
            raise ValueError("temperature must be finite and at least one")
        self.epsilon = float(self.epsilon)
        self.temperature = float(self.temperature)
        self._precision = np.asarray(
            np.linalg.solve(self._covariance, np.eye(self._covariance.shape[0])),
            dtype=np.float64,
        )
        self._cholesky = np.asarray(np.linalg.cholesky(self._covariance), dtype=np.float64)

    @property
    def dimension(self) -> int:
        return int(self._covariance.shape[0])

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
        size = validate_size(size)
        if size == 0:
            return np.empty((0, self.dimension), dtype=np.float64)
        normals = rng.normal(size=(size, self.dimension))
        if counter is not None:
            counter.increment("normal_draws", size * self.dimension)
        scale = math.sqrt(self.temperature * self.epsilon)
        return np.asarray(scale * normals @ self._cholesky.T, dtype=np.float64)

    def log_weights(self, samples: ArrayLike) -> Array:
        """Exact target-to-tempered-proposal log weights."""

        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("samples must have shape (n, dimension)")
        quadratic = np.einsum("ni,ij,nj->n", values, self._precision, values)
        log_normalizer_ratio = 0.5 * self.dimension * math.log(self.temperature)
        exponent = -0.5 * (1.0 - 1.0 / self.temperature) * quadratic / self.epsilon
        return np.asarray(log_normalizer_ratio + exponent, dtype=np.float64)


def fixed_scale_temperature(epsilon: float, *, proposal_scale: float = 1.0) -> float:
    """Choose temperature so proposal covariance stays near ``proposal_scale * C``."""

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not np.isfinite(proposal_scale) or proposal_scale <= 0.0:
        raise ValueError("proposal_scale must be positive and finite")
    return float(max(1.0, proposal_scale / epsilon))


def exact_tempered_log_second_moment(
    problem: RareGaussianProblem,
    temperature: float,
    epsilon: float,
) -> float:
    """Exact second moment for a centered Gaussian temperature proposal."""

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if not np.isfinite(temperature) or temperature < 1.0:
        raise ValueError("temperature must be finite and at least one")
    rate_multiplier = 2.0 - 1.0 / temperature
    log_factor = 0.5 * problem.dimension * (math.log(temperature) - math.log(rate_multiplier))
    standardized = problem.standardized_threshold(epsilon) * math.sqrt(rate_multiplier)
    event_log_probability = standard_normal_log_upper_tail(standardized)
    if isinstance(problem, GaussianTwoSidedRareEvent):
        event_log_probability += math.log(2.0)
    return float(log_factor + event_log_probability)


def estimate_with_tempering(
    problem: RareGaussianProblem,
    proposal: GaussianTemperedProposal,
    rng: np.random.Generator,
    size: int,
    *,
    counter: OperationCounter | None = None,
) -> RareEventEstimate:
    """Estimate a rare-event probability with a broadened Gaussian proposal."""

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
    return estimate_from_log_contributions(
        np.where(event, log_weights, float("-inf")),
        event_count=int(np.sum(event)),
        counter=counter,
    )


__all__ = [
    "GaussianTemperedProposal",
    "estimate_with_tempering",
    "exact_tempered_log_second_moment",
    "fixed_scale_temperature",
]
