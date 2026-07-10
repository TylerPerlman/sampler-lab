from __future__ import annotations

from sampler_lab.policy_experiments import (
    run_learning_benchmark,
    run_objective_gaming_study,
)


def test_mixing_rewards_avoid_acceptance_only_scale_collapse() -> None:
    rows = run_objective_gaming_study(
        n_updates=60,
        rollout_length=12,
        n_evaluation_steps=3_000,
        seed=2022,
    )
    by_name = {row.objective: row for row in rows}
    acceptance = by_name["acceptance"]
    contrastive = by_name["contrastive lower bound"]
    assert contrastive.mean_scale > acceptance.mean_scale + 0.4
    assert contrastive.effective_sample_size > 2.0 * acceptance.effective_sample_size


def test_separated_mixture_benchmark_exposes_distinct_failure_modes() -> None:
    rows = run_learning_benchmark(
        n_samples=1_000,
        n_warmup=400,
        policy_updates=40,
        variational_steps=50,
        svgd_particles=64,
        svgd_steps=20,
        seed=2023,
    )
    by_name = {row.method: row for row in rows}
    expected = {
        "direct oracle",
        "isotropic RWM",
        "adaptive covariance RWM",
        "policy-gradient MH",
        "reverse-KL Gaussian",
        "reverse-KL independence MH",
        "SVGD particles",
    }
    assert set(by_name) == expected
    assert by_name["direct oracle"].distribution.mode_occupancy_l1_error < 0.1
    assert by_name["reverse-KL Gaussian"].distribution.mode_occupancy_l1_error > 0.8
    policy_mixing = by_name["policy-gradient MH"].mode_mixing
    rwm_mixing = by_name["isotropic RWM"].mode_mixing
    assert policy_mixing is not None and rwm_mixing is not None
    assert policy_mixing.n_switches >= 3 * rwm_mixing.n_switches
    assert by_name["policy-gradient MH"].exact_after_freeze
    assert not by_name["SVGD particles"].exact_after_freeze
