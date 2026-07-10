import pytest

from sampler_lab.experiments import (
    run_langevin_gaussian_experiment,
    run_metropolis_generator_experiment,
)


@pytest.mark.statistical
def test_small_step_metropolis_generator_matches_langevin_limit() -> None:
    rows = run_metropolis_generator_experiment(
        step_sizes=(0.1, 0.03, 0.01),
        n_replications=20_000,
        state=0.75,
        seed=2022,
    )

    assert all(abs(row.z_score) < 3.0 for row in rows)
    assert all(row.limiting_generator == pytest.approx(-0.75) for row in rows)


@pytest.mark.statistical
def test_covariance_preconditioning_removes_gaussian_conditioning_penalty() -> None:
    rows = run_langevin_gaussian_experiment(
        condition_numbers=(1.0, 20.0),
        n_samples=6_000,
        burn_in=1_000,
        step_fraction=0.25,
        seed=2023,
    )
    by_key = {(row.condition_number, row.method): row for row in rows}

    identity_well = by_key[(1.0, "ULA identity")]
    identity_stiff = by_key[(20.0, "ULA identity")]
    preconditioned_stiff = by_key[(20.0, "ULA covariance")]
    mala_stiff = by_key[(20.0, "MALA covariance")]

    assert identity_well.exact_iat == pytest.approx(3.0)
    assert identity_stiff.exact_iat == pytest.approx(79.0)
    assert preconditioned_stiff.exact_iat == pytest.approx(3.0)
    assert identity_stiff.exact_iat is not None
    assert preconditioned_stiff.exact_iat is not None
    assert identity_stiff.exact_iat > 20.0 * preconditioned_stiff.exact_iat
    assert preconditioned_stiff.exact_stationary_variance == pytest.approx(4.0 / 3.0)
    assert mala_stiff.exact_stationary_variance == pytest.approx(1.0)
    assert mala_stiff.acceptance_rate is not None
    assert mala_stiff.acceptance_rate > 0.8
    assert mala_stiff.empirical_variance == pytest.approx(1.0, abs=0.08)
