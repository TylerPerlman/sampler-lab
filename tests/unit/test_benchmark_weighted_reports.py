from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from sampler_lab.benchmarks import (
    BenchmarkConfig,
    default_continuous_adapters,
    default_continuous_registry,
    normalized_log_weights,
    pareto_frontier,
    run_benchmark_suite,
    weighted_distribution_accuracy,
    write_report_bundle,
)


def test_normalized_log_weights_are_shift_invariant() -> None:
    values = np.log([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        normalized_log_weights(values),
        normalized_log_weights(values + 1000.0),
    )


def test_weighted_distribution_metrics_match_replicated_empirical_measure() -> None:
    samples = np.array([[0.0], [2.0]])
    log_weights = np.log([0.75, 0.25])
    weighted = weighted_distribution_accuracy(
        samples,
        log_weights,
        exact_mean=np.array([0.5]),
        exact_covariance=np.array([[0.75]]),
    )
    replicated = np.repeat(samples, [3, 1], axis=0)
    assert weighted.mean_l2_error == pytest.approx(0.0)
    assert weighted.covariance_frobenius_error == pytest.approx(0.0)
    assert np.mean(replicated) == pytest.approx(0.5)


def test_adapter_names_match_public_registry() -> None:
    names = tuple(adapter.name for adapter in default_continuous_adapters())
    assert len(names) == len(set(names))
    assert {"direct-oracle", "importance", "hmc", "svgd", "policy-gradient-mh"} <= set(names)
    assert set(names) == set(default_continuous_registry().sampler_names)


def test_tiny_replicated_suite_writes_json_csv_markdown_bundle(tmp_path) -> None:
    config = BenchmarkConfig(
        n_samples=80,
        warmup_steps=20,
        reference_samples=80,
        n_walkers=8,
        variational_steps=4,
        policy_updates=3,
        policy_rollout_length=2,
        svgd_particles=8,
        svgd_steps=2,
        annealing_particles=16,
        annealing_steps=3,
    )
    suite = run_benchmark_suite(
        config=config,
        n_replicates=2,
        target_names=("correlated-gaussian",),
        method_names=("direct-oracle", "importance", "random-walk-mh"),
        fail_fast=True,
    )
    assert len(suite.results) == 6
    assert not suite.failures
    assert len(suite.aggregates) == 3
    paths = write_report_bundle(suite, tmp_path)
    assert all(path.exists() for path in paths)
    payload = json.loads((tmp_path / "benchmark_results.json").read_text())
    assert len(payload["results"]) == 6
    with (tmp_path / "benchmark_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    with (tmp_path / "benchmark_pairings.csv").open(newline="") as handle:
        pairings = list(csv.DictReader(handle))
    assert len(pairings) == 3
    assert {row["status"] for row in pairings} == {"success"}
    frontier = pareto_frontier(suite.aggregates, target="correlated-gaussian")
    assert frontier


def test_capability_exclusion_is_preserved_in_suite() -> None:
    config = BenchmarkConfig(
        n_samples=20,
        warmup_steps=5,
        reference_samples=20,
        n_walkers=6,
        variational_steps=2,
        policy_updates=2,
        policy_rollout_length=2,
        svgd_particles=4,
        svgd_steps=1,
        annealing_particles=8,
        annealing_steps=2,
    )
    suite = run_benchmark_suite(
        config=config,
        n_replicates=1,
        target_names=("bimodal-anisotropic-funnel",),
        method_names=("reverse-kl",),
    )
    assert not suite.results
    assert suite.exclusions[0].reasons == ("sampler adapter does not support multimodal targets",)


def test_annealed_output_preserves_weighted_particle_semantics() -> None:
    config = BenchmarkConfig(
        n_samples=20,
        warmup_steps=5,
        reference_samples=20,
        n_walkers=6,
        variational_steps=2,
        policy_updates=2,
        policy_rollout_length=2,
        svgd_particles=4,
        svgd_steps=1,
        annealing_particles=8,
        annealing_steps=2,
    )
    suite = run_benchmark_suite(
        config=config,
        n_replicates=1,
        target_names=("correlated-gaussian",),
        method_names=("annealed-smc",),
        fail_fast=True,
    )
    assert suite.results[0].output_semantics == "weighted-particles"
    assert not suite.results[0].exact_after_freeze
