from __future__ import annotations

import pytest

from sampler_lab.benchmarks import BenchmarkConfig, default_continuous_adapters, run_benchmark_suite


@pytest.mark.statistical
def test_all_compatible_adapters_run_on_bimodal_funnel_capstone() -> None:
    config = BenchmarkConfig(
        n_samples=120,
        warmup_steps=30,
        reference_samples=120,
        n_walkers=12,
        variational_steps=6,
        policy_updates=4,
        policy_rollout_length=3,
        svgd_particles=16,
        svgd_steps=2,
        annealing_particles=24,
        annealing_steps=3,
        seed=701,
    )
    suite = run_benchmark_suite(
        config=config,
        n_replicates=1,
        target_names=("bimodal-anisotropic-funnel",),
        fail_fast=True,
        seed=701,
    )
    compatible = {
        adapter.name for adapter in default_continuous_adapters() if adapter.name != "reverse-kl"
    }
    observed = {result.method for result in suite.results}
    assert observed == compatible
    assert not suite.failures
    assert {result.output_semantics for result in suite.results} >= {
        "iid-samples",
        "weighted-samples",
        "markov-chain",
        "ensemble-chain",
        "approximate-particles",
    }
