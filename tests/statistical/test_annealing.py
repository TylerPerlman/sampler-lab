import pytest

from sampler_lab.experiments import run_annealing_experiment


@pytest.mark.statistical
def test_small_ising_annealing_matches_exact_partition_ratio_and_magnetization() -> None:
    rows = run_annealing_experiment(
        lattice_size=2,
        target_beta=0.6,
        path_steps=(8, 16),
        n_particles=4_000,
        sweeps_per_stage=1,
        ess_threshold=0.8,
        seed=2022,
    )

    assert len(rows) == 4
    for row in rows:
        assert row.log_ratio_estimate == pytest.approx(row.exact_log_ratio, abs=0.07)
        assert row.mean_absolute_magnetization == pytest.approx(
            row.exact_mean_absolute_magnetization,
            abs=0.04,
        )
        assert 0.0 < row.minimum_ess_fraction <= 1.0
        assert 0 < row.unique_initial_ancestors <= row.n_particles
        assert row.spin_updates == row.n_particles * 4 * row.n_steps

    resampled = [row for row in rows if row.method == "ESS-resampled SMC"]
    assert any(row.resampling_steps > 0 for row in resampled)
