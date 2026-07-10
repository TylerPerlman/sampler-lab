"""Diagnostics for normalized importance weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.numerics import normalize_log_weights

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WeightDiagnostics:
    """Scale-free diagnostics for a collection of nonnegative weights.

    ``entropy`` is the Shannon entropy of the normalized weights. ``perplexity``
    is ``exp(entropy)`` and therefore has the units of an effective particle
    count. ``coefficient_of_variation_squared`` uses the empirical mean weight
    as the normalizing constant and equals ``n / ESS - 1``.
    """

    effective_sample_size: float
    ess_fraction: float
    max_normalized_weight: float
    entropy: float
    normalized_entropy: float
    perplexity: float
    coefficient_of_variation_squared: float
    n_positive: int
    n_weights: int


def diagnostics_from_normalized_weights(weights: ArrayLike) -> WeightDiagnostics:
    """Summarize one-dimensional normalized nonnegative weights."""

    normalized = np.asarray(weights, dtype=np.float64)
    if normalized.ndim != 1 or normalized.size == 0:
        raise ValueError("weights must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(normalized)) or np.any(normalized < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    total = float(np.sum(normalized))
    if not np.isclose(total, 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("weights must sum to one")

    # Renormalize tiny floating-point drift before evaluating identities that
    # should agree algebraically.
    normalized = normalized / total
    squared_sum = float(normalized @ normalized)
    effective_sample_size = 1.0 / squared_sum
    positive = normalized > 0.0
    entropy = float(-np.sum(normalized[positive] * np.log(normalized[positive])))
    n_weights = int(normalized.size)
    log_n = float(np.log(n_weights))
    normalized_entropy = 1.0 if n_weights == 1 else entropy / log_n

    return WeightDiagnostics(
        effective_sample_size=effective_sample_size,
        ess_fraction=effective_sample_size / n_weights,
        max_normalized_weight=float(np.max(normalized)),
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        perplexity=float(np.exp(entropy)),
        coefficient_of_variation_squared=n_weights * squared_sum - 1.0,
        n_positive=int(np.count_nonzero(positive)),
        n_weights=n_weights,
    )


def weight_diagnostics(log_weights: ArrayLike) -> WeightDiagnostics:
    """Normalize log weights stably and return scale-free diagnostics."""

    normalized, _ = normalize_log_weights(log_weights)
    return diagnostics_from_normalized_weights(normalized)
