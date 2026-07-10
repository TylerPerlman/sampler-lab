"""Absolute, relative, and exponential-rate diagnostics for rare-event estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.numerics import logsumexp

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RareEventEstimate:
    """A nonnegative IID estimator summarized in both linear and log domains."""

    value: float
    log_value: float
    standard_error: float
    relative_standard_error: float
    sample_variance: float
    relative_variance: float
    log_second_moment: float
    log_relative_second_moment: float
    n_samples: int
    event_count: int
    contribution_effective_sample_size: float
    max_normalized_contribution: float
    operation_counts: dict[str, int | dict[str, int]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExponentialRateFit:
    """Least-squares fit of ``log(relative variance)`` against ``1 / epsilon``."""

    slope: float
    intercept: float
    r_squared: float
    n_points: int


def _linear_from_log(log_value: float) -> float:
    if log_value == float("-inf"):
        return 0.0
    if log_value > math.log(np.finfo(np.float64).max):
        return float("inf")
    if log_value < math.log(np.finfo(np.float64).tiny):
        return 0.0
    return float(math.exp(log_value))


def _log_expm1(log_value: float) -> float:
    """Return ``log(exp(log_value) - 1)`` for nonnegative ``log_value``."""

    if log_value < 0.0 and not math.isclose(log_value, 0.0, abs_tol=1e-14):
        raise ValueError("log_value must be nonnegative")
    if math.isclose(log_value, 0.0, abs_tol=1e-15):
        return float("-inf")
    if log_value <= 1e-8:
        return float(math.log(math.expm1(log_value)))
    if log_value > 50.0:
        return float(log_value + math.log1p(-math.exp(-log_value)))
    return float(math.log(math.expm1(log_value)))


def estimate_from_log_contributions(
    log_contributions: ArrayLike,
    *,
    event_count: int | None = None,
    counter: OperationCounter | None = None,
) -> RareEventEstimate:
    """Summarize nonnegative IID contributions represented by logarithms.

    Zero contributions are represented by ``-inf``.  The contribution ESS is
    ``(sum y_i)^2 / sum y_i^2`` and is deliberately distinct from ordinary
    importance-weight ESS: it measures how many observations actually support this estimator.
    """

    values = np.asarray(log_contributions, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("log_contributions must be a nonempty vector")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("log_contributions may contain only finite values or -inf")
    n_samples = int(values.size)
    positive = np.isfinite(values)
    inferred_event_count = int(np.sum(positive))
    if event_count is None:
        event_count = inferred_event_count
    if event_count < 0 or event_count > n_samples:
        raise ValueError("event_count must lie between zero and n_samples")

    if not np.any(positive):
        return RareEventEstimate(
            value=0.0,
            log_value=float("-inf"),
            standard_error=0.0,
            relative_standard_error=float("inf"),
            sample_variance=0.0,
            relative_variance=float("inf"),
            log_second_moment=float("-inf"),
            log_relative_second_moment=float("inf"),
            n_samples=n_samples,
            event_count=event_count,
            contribution_effective_sample_size=0.0,
            max_normalized_contribution=0.0,
            operation_counts={} if counter is None else counter.snapshot(),
        )

    log_sum = float(logsumexp(values))
    log_sum_squares = float(logsumexp(2.0 * values))
    log_mean = log_sum - math.log(n_samples)
    log_second_moment = log_sum_squares - math.log(n_samples)
    log_relative_second_moment = max(0.0, log_second_moment - 2.0 * log_mean)
    log_relative_variance = _log_expm1(log_relative_second_moment)
    relative_variance = _linear_from_log(log_relative_variance)
    relative_standard_error = (
        float(math.sqrt(relative_variance / n_samples))
        if np.isfinite(relative_variance)
        else float("inf")
    )

    value = _linear_from_log(log_mean)
    second_moment = _linear_from_log(log_second_moment)
    population_variance = max(0.0, second_moment - value * value)
    sample_variance = (
        float(population_variance * n_samples / (n_samples - 1)) if n_samples > 1 else float("nan")
    )
    standard_error = (
        float(math.sqrt(sample_variance / n_samples))
        if n_samples > 1 and np.isfinite(sample_variance)
        else float("nan")
    )
    log_contribution_ess = min(math.log(n_samples), 2.0 * log_sum - log_sum_squares)
    contribution_ess = float(math.exp(log_contribution_ess))
    max_normalized = float(math.exp(float(np.max(values[positive])) - log_sum))

    return RareEventEstimate(
        value=value,
        log_value=log_mean,
        standard_error=standard_error,
        relative_standard_error=relative_standard_error,
        sample_variance=sample_variance,
        relative_variance=relative_variance,
        log_second_moment=log_second_moment,
        log_relative_second_moment=log_relative_second_moment,
        n_samples=n_samples,
        event_count=event_count,
        contribution_effective_sample_size=contribution_ess,
        max_normalized_contribution=max_normalized,
        operation_counts={} if counter is None else counter.snapshot(),
    )


def exact_relative_error(
    *,
    log_probability: float,
    log_second_moment: float,
    n_samples: int,
) -> tuple[float, float, float]:
    """Return exact relative variance, relative standard error, and log ratio."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if log_probability == float("-inf"):
        raise ValueError("probability must be positive")
    log_relative_second_moment = max(0.0, log_second_moment - 2.0 * log_probability)
    log_relative_variance = _log_expm1(log_relative_second_moment)
    relative_variance = _linear_from_log(log_relative_variance)
    relative_standard_error = (
        float(math.sqrt(relative_variance / n_samples))
        if np.isfinite(relative_variance)
        else float("inf")
    )
    return relative_variance, relative_standard_error, log_relative_second_moment


def scaled_log_relative_variance(epsilon: float, log_relative_variance: float) -> float:
    """Return ``epsilon * log(relative variance)`` for asymptotic comparisons."""

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    if np.isnan(log_relative_variance):
        raise ValueError("log_relative_variance may not be nan")
    return float(epsilon * log_relative_variance)


def fit_exponential_relative_variance_rate(
    epsilons: ArrayLike,
    log_relative_variances: ArrayLike,
) -> ExponentialRateFit:
    """Fit the exponential growth rate from multiple small-noise levels."""

    epsilon_array = np.asarray(epsilons, dtype=np.float64)
    log_array = np.asarray(log_relative_variances, dtype=np.float64)
    if epsilon_array.ndim != 1 or log_array.shape != epsilon_array.shape:
        raise ValueError("epsilons and log_relative_variances must be matching vectors")
    if epsilon_array.size < 2:
        raise ValueError("at least two points are required")
    if np.any(~np.isfinite(epsilon_array)) or np.any(epsilon_array <= 0.0):
        raise ValueError("epsilons must be positive and finite")
    if np.any(~np.isfinite(log_array)):
        raise ValueError("log_relative_variances must be finite")
    predictor = 1.0 / epsilon_array
    design = np.column_stack((predictor, np.ones_like(predictor)))
    coefficients, *_ = np.linalg.lstsq(design, log_array, rcond=None)
    fitted = design @ coefficients
    residual = float(np.sum(np.square(log_array - fitted)))
    total = float(np.sum(np.square(log_array - np.mean(log_array))))
    r_squared = 1.0 if total == 0.0 else max(0.0, 1.0 - residual / total)
    return ExponentialRateFit(
        slope=float(coefficients[0]),
        intercept=float(coefficients[1]),
        r_squared=float(r_squared),
        n_points=int(epsilon_array.size),
    )


__all__ = [
    "ExponentialRateFit",
    "RareEventEstimate",
    "estimate_from_log_contributions",
    "exact_relative_error",
    "fit_exponential_relative_variance_rate",
    "scaled_log_relative_variance",
]
