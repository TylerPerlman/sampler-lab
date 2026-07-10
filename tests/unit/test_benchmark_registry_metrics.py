from __future__ import annotations

import json

import numpy as np
import pytest

from sampler_lab.benchmarks import (
    SamplerCapabilities,
    TargetCapabilities,
    binary_mode_mixing,
    check_compatibility,
    correlated_gaussian_case,
    default_continuous_cases,
    default_continuous_registry,
    evaluate_samples,
)


def test_capability_checks_report_all_missing_requirements() -> None:
    sampler = SamplerCapabilities(
        requires_normalized_density=True,
        requires_gradient=True,
        requires_hessian=True,
        requires_conditionals=True,
        requires_initial_reference_samples=True,
    )
    target = TargetCapabilities(
        normalized_density=False,
        gradient=False,
        hessian=False,
        conditionals=False,
        direct_samples=False,
    )
    result = check_compatibility(sampler, target)
    assert not result.compatible
    assert len(result.reasons) == 5


def test_default_registry_is_capability_aware() -> None:
    registry = default_continuous_registry()
    assert len(registry.sampler_names) >= 10
    assert set(registry.target_names) == {
        "bimodal-anisotropic-funnel",
        "correlated-gaussian",
        "rotated-anisotropic-funnel",
        "separated-anisotropic-gaussian-mixture",
    }
    assert registry.compatibility("hmc", "bimodal-anisotropic-funnel").compatible
    result = registry.compatibility("reverse-kl", "bimodal-anisotropic-funnel")
    assert not result.compatible
    assert result.reasons == ("sampler adapter does not support multimodal targets",)
    with pytest.raises(KeyError, match="unknown sampler"):
        registry.compatibility("teleportation", "correlated-gaussian")


def test_binary_mode_metrics_count_round_trips_and_residence() -> None:
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=np.int64)
    result = binary_mode_mixing(labels)
    assert result.n_switches == 4
    assert result.n_round_trips == 2
    assert result.first_passage_time == 2
    assert result.longest_residence == 2
    assert result.occupancy_first == pytest.approx(5.0 / 8.0)
    assert result.mode_indicator_ess is not None


def test_default_target_suite_has_exact_reference_access() -> None:
    cases = default_continuous_cases()
    assert tuple(case.name for case in cases) == (
        "correlated-gaussian",
        "separated-anisotropic-gaussian-mixture",
        "rotated-anisotropic-funnel",
        "bimodal-anisotropic-funnel",
    )
    rng = np.random.default_rng(13)
    for case in cases:
        samples = case.reference_samples(rng, 5)
        assert samples.shape == (5, case.target.mean_vector.size)
        assert np.all(np.isfinite(samples))


def test_direct_oracle_benchmark_is_accurate_and_json_serializable() -> None:
    case = correlated_gaussian_case(dimension=3, condition_number=4.0)
    rng = np.random.default_rng(14)
    samples = case.reference_samples(rng, 30_000)
    reference = case.reference_samples(rng, 500)
    result = evaluate_samples(
        method="direct-oracle",
        case=case,
        samples=samples,
        reference_samples=reference,
        exact_after_freeze=True,
    )
    assert result.distribution.standardized_mean_error < 0.02
    assert result.distribution.relative_covariance_error < 0.03
    payload = json.loads(result.to_json())
    assert payload["method"] == "direct-oracle"
    assert payload["n_samples"] == 30_000
