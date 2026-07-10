"""Shared distributional, weighting, and mode-mixing metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.core.numerics import logsumexp
from sampler_lab.diagnostics.time_series import empirical_effective_sample_size

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]
OperationSnapshot = dict[str, int | dict[str, int]]


def _as_samples(samples: ArrayLike) -> Array:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) <= 0 or not np.all(np.isfinite(values)):
        raise ValueError("samples must be a nonempty finite matrix")
    return values


def normalized_log_weights(log_weights: ArrayLike, *, n_samples: int | None = None) -> Array:
    """Normalize finite-or-negative-infinite log weights in a stable manner."""

    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("log_weights must be a nonempty vector")
    if n_samples is not None and values.shape != (n_samples,):
        raise ValueError("log_weights must contain one value per sample")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("log_weights may contain finite values or -inf only")
    if not np.any(np.isfinite(values)):
        raise ValueError("at least one log weight must be finite")
    normalized = np.exp(values - float(logsumexp(values)))
    return np.asarray(normalized, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class DistributionAccuracy:
    """Moment and reference-sample errors for one sample collection."""

    mean_l2_error: float
    standardized_mean_error: float
    covariance_frobenius_error: float
    relative_covariance_error: float
    imq_mmd: float | None
    mode_occupancy_l1_error: float | None


@dataclass(frozen=True, slots=True)
class ModeMixingMetrics:
    """Diagnostics for a binary mode-label trajectory."""

    n_switches: int
    n_round_trips: int
    first_passage_time: int | None
    longest_residence: int
    occupancy_first: float
    occupancy_second: float
    mode_indicator_ess: float | None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Machine-readable common benchmark result.

    ``operation_counts`` always refers to the frozen evaluation run. Training costs
    are carried separately so learned or adaptive methods cannot hide warmup work.
    """

    method: str
    target: str
    n_samples: int
    exact_after_freeze: bool
    distribution: DistributionAccuracy
    mode_mixing: ModeMixingMetrics | None
    acceptance_rate: float | None
    operation_counts: OperationSnapshot
    diagnostics: dict[str, float]
    output_semantics: str = "unweighted-samples"
    training_operation_counts: OperationSnapshot = field(default_factory=dict)
    training_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    replicate: int | None = None

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)


def _imq_kernel_matrix(
    first: Array,
    second: Array,
    *,
    c: float = 1.0,
    beta: float = -0.5,
) -> Array:
    squared = np.sum((first[:, None, :] - second[None, :, :]) ** 2, axis=2)
    return np.asarray((c * c + squared) ** beta, dtype=np.float64)


def _deterministic_prefix(values: Array, maximum_points: int) -> Array:
    return np.asarray(values[:maximum_points], dtype=np.float64)


def imq_mmd(samples: ArrayLike, reference: ArrayLike, *, maximum_points: int = 500) -> float:
    """Biased IMQ maximum mean discrepancy with deterministic subsampling."""

    first = _as_samples(samples)
    second = _as_samples(reference)
    if first.shape[1] != second.shape[1]:
        raise ValueError("sample dimensions must match")
    if maximum_points <= 0:
        raise ValueError("maximum_points must be positive")
    first = _deterministic_prefix(first, maximum_points)
    second = _deterministic_prefix(second, maximum_points)
    value = float(
        np.mean(_imq_kernel_matrix(first, first))
        + np.mean(_imq_kernel_matrix(second, second))
        - 2.0 * np.mean(_imq_kernel_matrix(first, second))
    )
    return float(np.sqrt(max(0.0, value)))


def weighted_imq_mmd(
    samples: ArrayLike,
    log_weights: ArrayLike,
    reference: ArrayLike,
    *,
    maximum_points: int = 500,
) -> float:
    """Biased IMQ MMD for a self-normalized weighted empirical measure."""

    first = _as_samples(samples)
    second = _as_samples(reference)
    if first.shape[1] != second.shape[1]:
        raise ValueError("sample dimensions must match")
    if maximum_points <= 0:
        raise ValueError("maximum_points must be positive")
    weights = normalized_log_weights(log_weights, n_samples=first.shape[0])
    keep = min(maximum_points, first.shape[0])
    indices = np.argsort(-weights, kind="stable")[:keep]
    first = first[indices]
    weights = weights[indices]
    weight_total = float(np.sum(weights))
    if weight_total <= 0.0:
        raise ValueError("deterministic MMD prefix contains no positive weight")
    weights = weights / weight_total
    second = _deterministic_prefix(second, maximum_points)
    reference_weights = np.full(second.shape[0], 1.0 / second.shape[0], dtype=np.float64)
    value = float(
        weights @ _imq_kernel_matrix(first, first) @ weights
        + reference_weights @ _imq_kernel_matrix(second, second) @ reference_weights
        - 2.0 * weights @ _imq_kernel_matrix(first, second) @ reference_weights
    )
    return float(np.sqrt(max(0.0, value)))


def _accuracy_from_moments(
    *,
    sample_mean: Array,
    sample_covariance: Array,
    exact_mean: Array,
    exact_covariance: Array,
    mmd: float | None,
    occupancy_error: float | None,
) -> DistributionAccuracy:
    mean_error = float(np.linalg.norm(sample_mean - exact_mean))
    scale = float(np.sqrt(max(np.trace(exact_covariance), 1e-15)))
    covariance_error = float(np.linalg.norm(sample_covariance - exact_covariance, ord="fro"))
    covariance_norm = float(np.linalg.norm(exact_covariance, ord="fro"))
    covariance_scale = max(covariance_norm, 1e-15)
    return DistributionAccuracy(
        mean_l2_error=mean_error,
        standardized_mean_error=mean_error / scale,
        covariance_frobenius_error=covariance_error,
        relative_covariance_error=covariance_error / covariance_scale,
        imq_mmd=mmd,
        mode_occupancy_l1_error=occupancy_error,
    )


def _mode_occupancy_error(
    labels: ArrayLike | None,
    probabilities: ArrayLike | None,
    weights: Array | None = None,
) -> float | None:
    if labels is None and probabilities is None:
        return None
    if labels is None or probabilities is None:
        raise ValueError("mode labels and exact probabilities must be supplied together")
    label_values = np.asarray(labels, dtype=np.int64)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    if probability_values.ndim != 1 or probability_values.size == 0:
        raise ValueError("exact mode probabilities must be a nonempty vector")
    if label_values.ndim != 1:
        raise ValueError("mode labels must be one-dimensional")
    if np.any(label_values < 0) or np.any(label_values >= probability_values.size):
        raise ValueError("mode label is outside the supplied probability range")
    observed: Array
    if weights is None:
        observed = np.asarray(
            np.bincount(label_values, minlength=probability_values.size) / label_values.size,
            dtype=np.float64,
        )
    else:
        if weights.shape != label_values.shape:
            raise ValueError("mode labels and weights must have matching shapes")
        observed = np.asarray(
            np.bincount(
                label_values,
                weights=weights,
                minlength=probability_values.size,
            ),
            dtype=np.float64,
        )
    return float(np.sum(np.abs(observed - probability_values)))


def distribution_accuracy(
    samples: ArrayLike,
    *,
    exact_mean: ArrayLike,
    exact_covariance: ArrayLike,
    reference_samples: ArrayLike | None = None,
    mode_labels: ArrayLike | None = None,
    exact_mode_probabilities: ArrayLike | None = None,
) -> DistributionAccuracy:
    """Compute common moment, reference, and mode-mass errors."""

    values = _as_samples(samples)
    mean = np.asarray(exact_mean, dtype=np.float64)
    covariance = np.asarray(exact_covariance, dtype=np.float64)
    if mean.shape != (values.shape[1],) or covariance.shape != (values.shape[1], values.shape[1]):
        raise ValueError("exact moments must match sample dimension")
    sample_mean = np.mean(values, axis=0)
    sample_covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=0))
    mmd = None if reference_samples is None else imq_mmd(values, reference_samples)
    occupancy_error = _mode_occupancy_error(mode_labels, exact_mode_probabilities)
    return _accuracy_from_moments(
        sample_mean=sample_mean,
        sample_covariance=sample_covariance,
        exact_mean=mean,
        exact_covariance=covariance,
        mmd=mmd,
        occupancy_error=occupancy_error,
    )


def weighted_distribution_accuracy(
    samples: ArrayLike,
    log_weights: ArrayLike,
    *,
    exact_mean: ArrayLike,
    exact_covariance: ArrayLike,
    reference_samples: ArrayLike | None = None,
    mode_labels: ArrayLike | None = None,
    exact_mode_probabilities: ArrayLike | None = None,
) -> DistributionAccuracy:
    """Compute accuracy for a self-normalized weighted empirical measure."""

    values = _as_samples(samples)
    weights = normalized_log_weights(log_weights, n_samples=values.shape[0])
    mean = np.asarray(exact_mean, dtype=np.float64)
    covariance = np.asarray(exact_covariance, dtype=np.float64)
    if mean.shape != (values.shape[1],) or covariance.shape != (values.shape[1], values.shape[1]):
        raise ValueError("exact moments must match sample dimension")
    sample_mean = weights @ values
    centered = values - sample_mean
    sample_covariance = (centered.T * weights) @ centered
    mmd = (
        None
        if reference_samples is None
        else weighted_imq_mmd(values, log_weights, reference_samples)
    )
    occupancy_error = _mode_occupancy_error(
        mode_labels,
        exact_mode_probabilities,
        weights,
    )
    return _accuracy_from_moments(
        sample_mean=np.asarray(sample_mean, dtype=np.float64),
        sample_covariance=np.asarray(sample_covariance, dtype=np.float64),
        exact_mean=mean,
        exact_covariance=covariance,
        mmd=mmd,
        occupancy_error=occupancy_error,
    )


def binary_mode_mixing(labels: ArrayLike) -> ModeMixingMetrics:
    """Compute switch, round-trip, residence, and ESS diagnostics for labels 0/1."""

    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or values.size == 0 or np.any((values < 0) | (values > 1)):
        raise ValueError("labels must be a nonempty binary vector")
    switches = np.flatnonzero(values[1:] != values[:-1]) + 1
    first_passage = int(switches[0]) if switches.size else None
    run_boundaries = np.concatenate(([0], switches, [values.size]))
    longest_residence = int(np.max(np.diff(run_boundaries)))
    n_round_trips = int(switches.size // 2)
    occupancy_first = float(np.mean(values == 0))
    indicator = values.astype(np.float64)
    ess = None if np.all(values == values[0]) else empirical_effective_sample_size(indicator)
    return ModeMixingMetrics(
        n_switches=int(switches.size),
        n_round_trips=n_round_trips,
        first_passage_time=first_passage,
        longest_residence=longest_residence,
        occupancy_first=occupancy_first,
        occupancy_second=1.0 - occupancy_first,
        mode_indicator_ess=ess,
    )
