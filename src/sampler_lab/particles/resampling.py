"""Particle resampling schemes and offspring diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _normalized_weights(weights: ArrayLike) -> Array:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("at least one weight must be positive")
    return values / total


def _target_size(n_particles: int | None, default: int) -> int:
    result = default if n_particles is None else n_particles
    if isinstance(result, bool) or not isinstance(result, int):
        raise TypeError("n_particles must be an integer")
    if result <= 0:
        raise ValueError("n_particles must be positive")
    return result


def offspring_counts(parent_indices: ArrayLike, n_parents: int) -> IntArray:
    """Count offspring assigned to each parent."""

    if isinstance(n_parents, bool) or not isinstance(n_parents, int):
        raise TypeError("n_parents must be an integer")
    if n_parents <= 0:
        raise ValueError("n_parents must be positive")
    indices = np.asarray(parent_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("parent_indices must be one-dimensional")
    if np.any(indices < 0) or np.any(indices >= n_parents):
        raise IndexError("parent index out of range")
    return np.bincount(indices, minlength=n_parents).astype(np.int64, copy=False)


@dataclass(frozen=True, slots=True)
class ResamplingDiagnostics:
    """Observed offspring allocation from one resampling operation."""

    n_parents: int
    n_offspring: int
    n_unique_parents: int
    unique_parent_fraction: float
    max_offspring: int
    offspring_variance: float


def resampling_diagnostics(parent_indices: ArrayLike, n_parents: int) -> ResamplingDiagnostics:
    """Summarize an observed parent-index vector."""

    counts = offspring_counts(parent_indices, n_parents)
    n_offspring = int(np.sum(counts))
    unique = int(np.count_nonzero(counts))
    return ResamplingDiagnostics(
        n_parents=n_parents,
        n_offspring=n_offspring,
        n_unique_parents=unique,
        unique_parent_fraction=unique / n_parents,
        max_offspring=int(np.max(counts)),
        offspring_variance=float(np.var(counts)),
    )


def multinomial_conditional_variances(weights: ArrayLike, n_particles: int) -> Array:
    """Return ``Var[N_i | weights]`` for multinomial resampling."""

    normalized = _normalized_weights(weights)
    target = _target_size(n_particles, normalized.size)
    return target * normalized * (1.0 - normalized)


def minimal_conditional_variances(weights: ArrayLike, n_particles: int) -> Array:
    """Minimum possible marginal offspring variances for unbiased integer counts."""

    normalized = _normalized_weights(weights)
    target = _target_size(n_particles, normalized.size)
    expected = target * normalized
    fractional = expected - np.floor(expected)
    return np.asarray(fractional * (1.0 - fractional), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class MultinomialResampler:
    """Independent categorical parent draws with fixed population size."""

    def resample(
        self,
        weights: Array,
        rng: np.random.Generator,
        n_particles: int | None = None,
    ) -> IntArray:
        normalized = _normalized_weights(weights)
        target = _target_size(n_particles, normalized.size)
        return np.asarray(rng.choice(normalized.size, size=target, p=normalized), dtype=np.int64)


@dataclass(frozen=True, slots=True)
class SystematicResampler:
    """One-random-number fixed-population systematic resampling."""

    def resample(
        self,
        weights: Array,
        rng: np.random.Generator,
        n_particles: int | None = None,
    ) -> IntArray:
        normalized = _normalized_weights(weights)
        target = _target_size(n_particles, normalized.size)
        offset = float(rng.random())
        positions = (np.arange(target, dtype=np.float64) + offset) / target
        cumulative = np.cumsum(normalized)
        cumulative[-1] = 1.0
        return np.searchsorted(cumulative, positions, side="right").astype(np.int64)


@dataclass(frozen=True, slots=True)
class BernoulliResampler:
    """Independent floor-plus-Bernoulli offspring counts.

    The expected population size is ``n_particles`` but the realized size is
    random. Each expected offspring count is split into its integer floor and an independent
    Bernoulli draw for the fractional remainder, so the expected count is unchanged. An empty
    population is possible and is deliberately not retried.
    """

    def resample(
        self,
        weights: Array,
        rng: np.random.Generator,
        n_particles: int | None = None,
    ) -> IntArray:
        normalized = _normalized_weights(weights)
        target = _target_size(n_particles, normalized.size)
        expected = target * normalized
        base = np.floor(expected).astype(np.int64)
        fractional = expected - base
        counts = base + (rng.random(normalized.size) < fractional).astype(np.int64)
        return np.repeat(np.arange(normalized.size, dtype=np.int64), counts)
