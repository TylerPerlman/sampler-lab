"""Exact-reference continuous targets and common sample evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sampler_lab.benchmarks.capabilities import TargetCapabilities
from sampler_lab.benchmarks.metrics import (
    BenchmarkResult,
    binary_mode_mixing,
    distribution_accuracy,
    normalized_log_weights,
    weighted_distribution_accuracy,
)
from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.protocols import TwiceDifferentiableLogDensity
from sampler_lab.models.bimodal_funnel import BimodalFunnelTarget
from sampler_lab.models.funnel import FunnelTarget, seeded_orthogonal_matrix
from sampler_lab.models.gaussian import GaussianTarget
from sampler_lab.models.gaussian_mixture import GaussianMixtureTarget

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


class ExactContinuousTarget(TwiceDifferentiableLogDensity, Protocol):
    @property
    def mean_vector(self) -> Array: ...

    @property
    def covariance_matrix(self) -> Array: ...

    def sample(self, rng: np.random.Generator, size: int) -> Array: ...


@dataclass(frozen=True, slots=True)
class ContinuousTargetCase:
    """Benchmark target with exact references and optional mode labels."""

    name: str
    target: ExactContinuousTarget
    capabilities: TargetCapabilities
    mode_labeler: Callable[[Array], IntArray] | None = None
    exact_mode_probabilities: Array | None = None

    def reference_samples(self, rng: np.random.Generator, size: int) -> Array:
        return np.asarray(self.target.sample(rng, size), dtype=np.float64)


def correlated_gaussian_case(
    *,
    dimension: int = 8,
    condition_number: float = 25.0,
    seed: int = 2022,
) -> ContinuousTargetCase:
    rotation = seeded_orthogonal_matrix(dimension, seed)
    eigenvalues = np.geomspace(1.0, condition_number, dimension)
    covariance = rotation @ np.diag(eigenvalues) @ rotation.T
    target = GaussianTarget(np.zeros(dimension), covariance)
    return ContinuousTargetCase(
        "correlated-gaussian",
        target,
        TargetCapabilities(multimodal=False),
    )


def separated_gaussian_mixture_case(
    *,
    dimension: int = 8,
    separation: float = 10.0,
    condition_number: float = 25.0,
    seed: int = 2022,
) -> ContinuousTargetCase:
    rotation = seeded_orthogonal_matrix(dimension, seed)
    eigenvalues = np.geomspace(1.0, condition_number, dimension)
    first_covariance = rotation @ np.diag(eigenvalues) @ rotation.T
    second_rotation = seeded_orthogonal_matrix(dimension, seed + 1)
    second_covariance = second_rotation @ np.diag(eigenvalues[::-1]) @ second_rotation.T
    direction = rotation[:, 0]
    means = np.stack((-0.5 * separation * direction, 0.5 * separation * direction))
    target = GaussianMixtureTarget(
        np.array([0.5, 0.5]),
        means,
        np.stack((first_covariance, second_covariance)),
    )
    return ContinuousTargetCase(
        "separated-anisotropic-gaussian-mixture",
        target,
        TargetCapabilities(multimodal=True),
        mode_labeler=target.mode_labels,
        exact_mode_probabilities=np.array([0.5, 0.5]),
    )


def anisotropic_funnel_case(
    *,
    dimension: int = 10,
    sigma_v: float = 3.0,
    anisotropy_ratio: float = 20.0,
    seed: int = 2022,
) -> ContinuousTargetCase:
    target = FunnelTarget(
        dimension,
        sigma_v=sigma_v,
        scales=np.geomspace(1.0, anisotropy_ratio, dimension - 1),
        rotation=seeded_orthogonal_matrix(dimension, seed),
    )
    return ContinuousTargetCase(
        "rotated-anisotropic-funnel",
        target,
        TargetCapabilities(multimodal=False),
    )


def bimodal_funnel_case(
    *,
    dimension: int = 10,
    separation: float = 12.0,
    sigma_v: float = 3.0,
    anisotropy_ratio: float = 20.0,
    seed: int = 2022,
) -> ContinuousTargetCase:
    target = BimodalFunnelTarget(
        dimension=dimension,
        separation=separation,
        sigma_v=sigma_v,
        anisotropy_ratio=anisotropy_ratio,
        seed=seed,
    )
    return ContinuousTargetCase(
        "bimodal-anisotropic-funnel",
        target,
        TargetCapabilities(multimodal=True),
        mode_labeler=target.mode_labels,
        exact_mode_probabilities=np.array([0.5, 0.5]),
    )


def default_continuous_cases() -> tuple[ContinuousTargetCase, ...]:
    return (
        correlated_gaussian_case(),
        separated_gaussian_mixture_case(),
        anisotropic_funnel_case(),
        bimodal_funnel_case(),
    )


def _funnel_latent_v(case: ContinuousTargetCase, values: Array) -> Array | None:
    target = case.target
    if isinstance(target, FunnelTarget):
        return np.asarray([target.to_latent(value)[0] for value in values], dtype=np.float64)
    if isinstance(target, BimodalFunnelTarget):
        labels = target.mode_labels(values)
        return np.asarray(
            [
                target.components[int(label)].to_latent(value)[0]
                for value, label in zip(values, labels, strict=True)
            ],
            dtype=np.float64,
        )
    return None


def _target_specific_diagnostics(
    case: ContinuousTargetCase,
    values: Array,
    reference: Array,
    weights: Array | None = None,
) -> dict[str, float]:
    observed_v = _funnel_latent_v(case, values)
    reference_v = _funnel_latent_v(case, reference)
    if observed_v is None or reference_v is None:
        return {}
    target = case.target
    sigma_v = target.sigma_v if isinstance(target, (FunnelTarget, BimodalFunnelTarget)) else 1.0
    normalized = (
        np.full(values.shape[0], 1.0 / values.shape[0], dtype=np.float64)
        if weights is None
        else weights
    )
    neck = float(np.sum(normalized * (observed_v < -sigma_v)))
    wide = float(np.sum(normalized * (observed_v > sigma_v)))
    exact_neck = float(np.mean(reference_v < -sigma_v))
    exact_wide = float(np.mean(reference_v > sigma_v))
    return {
        "funnel_neck_probability": neck,
        "funnel_neck_reference": exact_neck,
        "funnel_neck_abs_error": abs(neck - exact_neck),
        "funnel_wide_probability": wide,
        "funnel_wide_reference": exact_wide,
        "funnel_wide_abs_error": abs(wide - exact_wide),
        "funnel_latent_v_mean_abs_error": abs(
            float(normalized @ observed_v) - float(np.mean(reference_v))
        ),
    }


def evaluate_samples(
    *,
    method: str,
    case: ContinuousTargetCase,
    samples: Array,
    reference_samples: Array,
    exact_after_freeze: bool,
    acceptance_rate: float | None = None,
    operation_counter: OperationCounter | None = None,
    diagnostics: dict[str, float] | None = None,
    compute_mode_mixing: bool = True,
    output_semantics: str = "unweighted-samples",
    training_operation_counter: OperationCounter | None = None,
    training_seconds: float = 0.0,
    evaluation_seconds: float = 0.0,
    replicate: int | None = None,
) -> BenchmarkResult:
    """Evaluate any unweighted continuous sample representation."""

    values = np.asarray(samples, dtype=np.float64)
    labels = case.mode_labeler(values) if case.mode_labeler is not None else None
    accuracy = distribution_accuracy(
        values,
        exact_mean=case.target.mean_vector,
        exact_covariance=case.target.covariance_matrix,
        reference_samples=reference_samples,
        mode_labels=labels,
        exact_mode_probabilities=case.exact_mode_probabilities,
    )
    mixing = binary_mode_mixing(labels) if labels is not None and compute_mode_mixing else None
    combined_diagnostics = _target_specific_diagnostics(case, values, reference_samples)
    combined_diagnostics.update(diagnostics or {})
    return BenchmarkResult(
        method=method,
        target=case.name,
        n_samples=values.shape[0],
        exact_after_freeze=exact_after_freeze,
        distribution=accuracy,
        mode_mixing=mixing,
        acceptance_rate=acceptance_rate,
        operation_counts=(operation_counter or OperationCounter()).snapshot(),
        diagnostics=combined_diagnostics,
        output_semantics=output_semantics,
        training_operation_counts=(training_operation_counter or OperationCounter()).snapshot(),
        training_seconds=float(training_seconds),
        evaluation_seconds=float(evaluation_seconds),
        replicate=replicate,
    )


def evaluate_weighted_samples(
    *,
    method: str,
    case: ContinuousTargetCase,
    samples: Array,
    log_weights: Array,
    reference_samples: Array,
    exact_after_freeze: bool,
    operation_counter: OperationCounter | None = None,
    training_operation_counter: OperationCounter | None = None,
    diagnostics: dict[str, float] | None = None,
    training_seconds: float = 0.0,
    evaluation_seconds: float = 0.0,
    replicate: int | None = None,
    output_semantics: str = "weighted-samples",
) -> BenchmarkResult:
    """Evaluate a self-normalized weighted continuous sample representation."""

    values = np.asarray(samples, dtype=np.float64)
    normalized = normalized_log_weights(log_weights, n_samples=values.shape[0])
    labels = case.mode_labeler(values) if case.mode_labeler is not None else None
    accuracy = weighted_distribution_accuracy(
        values,
        log_weights,
        exact_mean=case.target.mean_vector,
        exact_covariance=case.target.covariance_matrix,
        reference_samples=reference_samples,
        mode_labels=labels,
        exact_mode_probabilities=case.exact_mode_probabilities,
    )
    combined_diagnostics = _target_specific_diagnostics(
        case,
        values,
        reference_samples,
        normalized,
    )
    combined_diagnostics.update(diagnostics or {})
    combined_diagnostics.setdefault(
        "weight_effective_sample_size",
        float(1.0 / np.sum(normalized * normalized)),
    )
    combined_diagnostics.setdefault("max_normalized_weight", float(np.max(normalized)))
    return BenchmarkResult(
        method=method,
        target=case.name,
        n_samples=values.shape[0],
        exact_after_freeze=exact_after_freeze,
        distribution=accuracy,
        mode_mixing=None,
        acceptance_rate=None,
        operation_counts=(operation_counter or OperationCounter()).snapshot(),
        diagnostics=combined_diagnostics,
        output_semantics=output_semantics,
        training_operation_counts=(training_operation_counter or OperationCounter()).snapshot(),
        training_seconds=float(training_seconds),
        evaluation_seconds=float(evaluation_seconds),
        replicate=replicate,
    )
