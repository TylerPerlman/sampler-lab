"""Mixtures of Gaussian twists for rare sets with multiple dominating points."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.linalg import as_positive_definite
from sampler_lab.core.numerics import logsumexp, validate_size
from sampler_lab.rare_events.normal import log_cosh
from sampler_lab.rare_events.problems import GaussianTwoSidedRareEvent, RareGaussianProblem
from sampler_lab.rare_events.relative_error import (
    RareEventEstimate,
    estimate_from_log_contributions,
)

Array = NDArray[np.float64]


@dataclass(slots=True)
class GaussianShiftMixtureProposal:
    """Finite mixture of ``N(mean_k, epsilon C)`` proposals."""

    means: ArrayLike
    weights: ArrayLike
    covariance: ArrayLike
    epsilon: float
    _means: Array = field(init=False, repr=False)
    _weights: Array = field(init=False, repr=False)
    _log_weights: Array = field(init=False, repr=False)
    _covariance: Array = field(init=False, repr=False)
    _precision: Array = field(init=False, repr=False)
    _cholesky: Array = field(init=False, repr=False)
    _log_determinant: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._means = np.asarray(self.means, dtype=np.float64)
        if self._means.ndim != 2 or self._means.shape[0] == 0 or self._means.shape[1] == 0:
            raise ValueError("means must have shape (components, dimension)")
        if not np.all(np.isfinite(self._means)):
            raise ValueError("means must be finite")
        self._weights = np.asarray(self.weights, dtype=np.float64)
        if self._weights.shape != (self._means.shape[0],):
            raise ValueError("weights must match the number of means")
        if np.any(~np.isfinite(self._weights)) or np.any(self._weights <= 0.0):
            raise ValueError("weights must be positive and finite")
        self._weights = self._weights / np.sum(self._weights)
        self._log_weights = np.log(self._weights)
        self._covariance = as_positive_definite(self.covariance).copy()
        if self._covariance.shape != (self._means.shape[1], self._means.shape[1]):
            raise ValueError("covariance shape must match mean dimension")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive and finite")
        self.epsilon = float(self.epsilon)
        self._precision = np.asarray(
            np.linalg.solve(self._covariance, np.eye(self._covariance.shape[0])),
            dtype=np.float64,
        )
        self._cholesky = np.asarray(np.linalg.cholesky(self._covariance), dtype=np.float64)
        sign, log_determinant = np.linalg.slogdet(self._covariance)
        if sign <= 0.0:  # pragma: no cover
            raise RuntimeError("covariance determinant must be positive")
        self._log_determinant = float(log_determinant)

    @property
    def dimension(self) -> int:
        return int(self._means.shape[1])

    @property
    def n_components(self) -> int:
        return int(self._means.shape[0])

    @property
    def means_matrix(self) -> Array:
        return self._means.copy()

    @property
    def mixture_weights(self) -> Array:
        return self._weights.copy()

    @property
    def covariance_matrix(self) -> Array:
        return self._covariance.copy()

    def sample(
        self,
        rng: np.random.Generator,
        size: int,
        *,
        counter: OperationCounter | None = None,
    ) -> tuple[Array, NDArray[np.int64]]:
        """Draw mixture samples and return their component labels."""

        size = validate_size(size)
        if size == 0:
            return (
                np.empty((0, self.dimension), dtype=np.float64),
                np.empty(0, dtype=np.int64),
            )
        uniforms = rng.random(size)
        cumulative = np.cumsum(self._weights)
        components = np.searchsorted(cumulative, uniforms, side="right").astype(np.int64)
        components = np.minimum(components, self.n_components - 1)
        normals = rng.normal(size=(size, self.dimension))
        if counter is not None:
            counter.increment("uniform_draws", size)
            counter.increment("normal_draws", size * self.dimension)
        samples = self._means[components] + math.sqrt(self.epsilon) * normals @ self._cholesky.T
        return np.asarray(samples, dtype=np.float64), components

    def component_log_densities(self, samples: ArrayLike) -> Array:
        """Return an ``(n, components)`` matrix of normalized component log densities."""

        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("samples must have shape (n, dimension)")
        displacements = values[:, None, :] - self._means[None, :, :]
        quadratic = np.einsum("nki,ij,nkj->nk", displacements, self._precision, displacements)
        normalizer = 0.5 * (
            self.dimension * math.log(2.0 * math.pi * self.epsilon) + self._log_determinant
        )
        return np.asarray(-0.5 * quadratic / self.epsilon - normalizer, dtype=np.float64)

    def log_density(self, samples: ArrayLike) -> Array:
        """Evaluate the normalized mixture log density."""

        components = self.component_log_densities(samples) + self._log_weights[None, :]
        return np.asarray(logsumexp(components, axis=1), dtype=np.float64)

    def target_log_density(self, samples: ArrayLike) -> Array:
        """Evaluate the matching centered Gaussian target density."""

        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("samples must have shape (n, dimension)")
        quadratic = np.einsum("ni,ij,nj->n", values, self._precision, values)
        normalizer = 0.5 * (
            self.dimension * math.log(2.0 * math.pi * self.epsilon) + self._log_determinant
        )
        return np.asarray(-0.5 * quadratic / self.epsilon - normalizer, dtype=np.float64)

    def log_weights(self, samples: ArrayLike) -> Array:
        """Return exact target-to-mixture log weights."""

        return self.target_log_density(samples) - self.log_density(samples)


def symmetric_twist_mixture(
    problem: GaussianTwoSidedRareEvent,
    epsilon: float,
) -> GaussianShiftMixtureProposal:
    """Place equal Gaussian components at both dominating points."""

    positive, negative = problem.dominant_points
    return GaussianShiftMixtureProposal(
        means=np.stack((positive, negative)),
        weights=np.asarray([0.5, 0.5]),
        covariance=problem.covariance_matrix,
        epsilon=epsilon,
    )


def exact_symmetric_mixture_log_second_moment(
    problem: GaussianTwoSidedRareEvent,
    epsilon: float,
    *,
    quadrature_order: int = 192,
    tail_limit: float = 40.0,
) -> float:
    """Compute the exact second moment for the equal two-dominating-point mixture.

    Orthogonal Gaussian coordinates integrate out.  A change of variables reduces the
    remaining half-line integral to a smooth exponentially decaying integral on ``[0, inf)``.
    Gauss--Legendre quadrature on ``[0, tail_limit]`` is deterministic and stable even when
    the probability itself is far below floating-point range.
    """

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive")
    if not np.isfinite(tail_limit) or tail_limit <= 0.0:
        raise ValueError("tail_limit must be positive and finite")
    z = problem.standardized_threshold(epsilon)
    rate_scale = z * z
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    t = 0.5 * tail_limit * (nodes + 1.0)
    scaled_weights = 0.5 * tail_limit * weights
    log_integrand = np.asarray(
        [
            -float(value)
            - float(value) ** 2 / (2.0 * rate_scale)
            - log_cosh(rate_scale + float(value))
            for value in t
        ],
        dtype=np.float64,
    )
    log_integral = float(logsumexp(np.log(scaled_weights) + log_integrand))
    log_prefactor = math.log(2.0) - math.log(z) - 0.5 * math.log(2.0 * math.pi)
    return float(log_prefactor + log_integral)


def estimate_with_mixture(
    problem: RareGaussianProblem,
    proposal: GaussianShiftMixtureProposal,
    rng: np.random.Generator,
    size: int,
    *,
    counter: OperationCounter | None = None,
) -> RareEventEstimate:
    """Estimate a Gaussian rare-event probability under a finite shift mixture."""

    if proposal.dimension != problem.dimension:
        raise ValueError("proposal and problem dimensions differ")
    if not np.allclose(proposal.covariance_matrix, problem.covariance_matrix):
        raise ValueError("proposal and problem base covariances differ")
    size = validate_size(size)
    if size == 0:
        raise ValueError("size must be positive")
    samples, _components = proposal.sample(rng, size, counter=counter)
    event = problem.event(samples)
    log_weights = proposal.log_weights(samples)
    if counter is not None:
        counter.increment("log_density_evaluations", size)
        counter.increment("proposal_density_evaluations", size * proposal.n_components)
    return estimate_from_log_contributions(
        np.where(event, log_weights, float("-inf")),
        event_count=int(np.sum(event)),
        counter=counter,
    )


__all__ = [
    "GaussianShiftMixtureProposal",
    "estimate_with_mixture",
    "exact_symmetric_mixture_log_second_moment",
    "symmetric_twist_mixture",
]
