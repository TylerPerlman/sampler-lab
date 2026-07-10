import pytest

from sampler_lab.experiments import (
    run_hamiltonian_gaussian_experiment,
    run_xy_dynamics_experiment,
)


@pytest.mark.statistical
def test_precision_mass_removes_hmc_condition_number_cost() -> None:
    rows = run_hamiltonian_gaussian_experiment(
        condition_numbers=(1.0, 25.0),
        n_samples=1_500,
        burn_in=300,
        trajectory_time=1.0,
        step_fraction=0.25,
        friction=1.0,
        seed=91,
    )
    by_key = {(row.condition_number, row.method): row for row in rows}

    identity = by_key[(25.0, "HMC identity mass")]
    precision = by_key[(25.0, "HMC precision mass")]
    assert identity.n_leapfrog_steps > precision.n_leapfrog_steps
    assert identity.gradient_evaluations > 2.0 * precision.gradient_evaluations
    assert precision.acceptance_rate is not None
    assert precision.acceptance_rate > 0.9
    assert precision.empirical_variance == pytest.approx(1.0, abs=0.15)


@pytest.mark.statistical
def test_xy_experiment_reports_exact_reference_and_corrected_methods() -> None:
    rows = run_xy_dynamics_experiment(
        n_samples=2_000,
        burn_in=300,
        concentration=1.5,
        seed=92,
    )

    assert {row.method for row in rows} == {
        "HMC",
        "BAOAB",
        "Metropolized underdamped",
    }
    assert all(row.exact_mean_cosine == pytest.approx(0.5961332388312908) for row in rows)
    assert all(row.gradient_evaluations > 0 for row in rows)
    assert all(row.ess_per_thousand_gradients > 0.0 for row in rows)
