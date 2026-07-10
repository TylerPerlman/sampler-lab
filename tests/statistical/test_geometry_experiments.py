import pytest

from sampler_lab.experiments import (
    run_conditioning_geometry_experiment,
    run_rosenbrock_geometry_experiment,
)


@pytest.mark.statistical
def test_conditioning_experiment_exposes_affine_geometry_advantage() -> None:
    rows = run_conditioning_geometry_experiment(
        condition_numbers=(30.0,),
        n_samples=2500,
        burn_in=300,
        seed=19,
    )
    by_method = {row.method: row for row in rows}

    assert (
        by_method["isotropic RWM"].empirical_iat > 2.0 * by_method["covariance RWM"].empirical_iat
    )
    assert (
        by_method["isotropic RWM"].empirical_iat
        > 3.0 * by_method["stochastic Newton"].empirical_iat
    )
    assert by_method["stochastic Newton"].acceptance_rate > 0.75


@pytest.mark.statistical
def test_rosenbrock_experiment_reports_all_requested_method_families() -> None:
    rows = run_rosenbrock_geometry_experiment(
        n_samples=500,
        burn_in=100,
        n_walkers=12,
        seed=21,
    )
    methods = {row.method for row in rows}

    assert methods == {
        "isotropic RWM",
        "fixed-covariance RWM",
        "stochastic Newton",
        "stretch ensemble",
        "walk ensemble",
    }
    ensemble_rows = [row for row in rows if "ensemble" in row.method]
    assert all(row.effective_sample_size > 0.0 for row in ensemble_rows)
    assert all(0.0 < row.acceptance_rate < 1.0 for row in rows)
