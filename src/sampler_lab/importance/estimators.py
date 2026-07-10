"""Known-normalization and self-normalized importance estimators."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.numerics import logsumexp, normalize_log_weights
from sampler_lab.core.results import ImportanceEstimate
from sampler_lab.diagnostics.weighted import (
    WeightDiagnostics,
    diagnostics_from_normalized_weights,
)

Array = NDArray[np.float64]


def _observable_values(
    samples: ArrayLike,
    observable: Callable[[Array], ArrayLike] | None,
) -> Array:
    sample_array = np.asarray(samples, dtype=np.float64)
    values = (
        sample_array
        if observable is None
        else np.asarray(observable(sample_array), dtype=np.float64)
    )
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("observable must produce a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("observable values must be finite")
    return values


def _validated_log_weights(log_weights: ArrayLike, shape: tuple[int, ...]) -> Array:
    weights = np.asarray(log_weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape != shape:
        raise ValueError("log_weights must match the scalar observations")
    # This performs the complete nan/+inf/all-zero validation in one place.
    normalize_log_weights(weights)
    return weights


def _common_result_fields(
    log_weights: Array,
) -> tuple[Array, float, WeightDiagnostics]:
    normalized, _ = normalize_log_weights(log_weights)
    log_mean_weight = float(logsumexp(log_weights) - np.log(log_weights.size))
    diagnostics = diagnostics_from_normalized_weights(normalized)
    return normalized, log_mean_weight, diagnostics


def standard_importance_estimate(
    samples: ArrayLike,
    log_weights: ArrayLike,
    observable: Callable[[Array], ArrayLike] | None = None,
) -> ImportanceEstimate:
    """Estimate ``E_q[w(X) f(X)]`` when the target normalization is known.

    ``log_weights`` should be ``log p(X) - log q(X)`` for normalized ``p`` and
    ``q``. The standard error is the IID sample standard deviation of the
    weighted contributions divided by ``sqrt(n)``.
    """

    values = _observable_values(samples, observable)
    weights = _validated_log_weights(log_weights, values.shape)
    _normalized, log_mean_weight, diagnostics = _common_result_fields(weights)

    finite_weights = weights[np.isfinite(weights)]
    shift = float(np.max(finite_weights))
    scale = float(np.exp(shift))
    if not np.isfinite(scale):
        raise OverflowError("importance estimate is not representable in float64")
    scaled_contributions = np.exp(weights - shift) * values
    contributions = scale * scaled_contributions
    estimate = float(np.mean(contributions))
    n_samples = int(values.size)
    standard_error = (
        float(np.std(contributions, ddof=1) / np.sqrt(n_samples))
        if n_samples >= 2
        else float("nan")
    )

    return ImportanceEstimate(
        value=estimate,
        standard_error=standard_error,
        n_samples=n_samples,
        effective_sample_size=diagnostics.effective_sample_size,
        max_normalized_weight=diagnostics.max_normalized_weight,
        weight_entropy=diagnostics.entropy,
        log_mean_weight=log_mean_weight,
        delta_method_bias=None,
        self_normalized=False,
    )


def self_normalized_importance_estimate(
    samples: ArrayLike,
    log_weights: ArrayLike,
    observable: Callable[[Array], ArrayLike] | None = None,
) -> ImportanceEstimate:
    """Estimate a target expectation from unnormalized importance weights.

    The standard error uses the plug-in delta-method variance

    ``n/(n-1) * sum_i omega_i^2 (f_i - estimate)^2``.

    The reported bias approximation is the usual second-order ratio-estimator
    approximation and is ``nan`` for a single observation.
    """

    values = _observable_values(samples, observable)
    weights = _validated_log_weights(log_weights, values.shape)
    normalized, log_mean_weight, diagnostics = _common_result_fields(weights)

    estimate = float(normalized @ values)
    n_samples = int(values.size)
    if n_samples == 1 or diagnostics.n_positive < 2:
        standard_error = float("nan")
        bias_approximation = float("nan")
    else:
        centered = values - estimate
        variance_estimate = (
            n_samples / (n_samples - 1) * float(np.sum(np.square(normalized * centered)))
        )
        standard_error = float(np.sqrt(max(0.0, variance_estimate)))

        # Any common scale in the unnormalized weights cancels from this ratio.
        finite_weights = weights[np.isfinite(weights)]
        shift = float(np.max(finite_weights))
        scaled_weights = np.exp(weights - shift)
        numerator_terms = scaled_weights * values
        mean_weight = float(np.mean(scaled_weights))
        variance_weight = float(np.var(scaled_weights, ddof=1))
        covariance = float(np.cov(numerator_terms, scaled_weights, ddof=1)[0, 1])
        bias_approximation = (estimate * variance_weight - covariance) / (
            n_samples * mean_weight**2
        )

    return ImportanceEstimate(
        value=estimate,
        standard_error=standard_error,
        n_samples=n_samples,
        effective_sample_size=diagnostics.effective_sample_size,
        max_normalized_weight=diagnostics.max_normalized_weight,
        weight_entropy=diagnostics.entropy,
        log_mean_weight=log_mean_weight,
        delta_method_bias=bias_approximation,
        self_normalized=True,
    )
