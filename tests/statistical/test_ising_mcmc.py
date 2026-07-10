import pytest

from sampler_lab.experiments import run_ising_experiment


@pytest.mark.statistical
def test_small_ising_mcmc_methods_match_exact_absolute_magnetization() -> None:
    rows = run_ising_experiment(
        lattice_sizes=(2,),
        betas=(0.4,),
        n_sweeps=8_000,
        burn_in_sweeps=1_000,
        seed=2022,
    )

    assert len(rows) == 3
    for row in rows:
        assert row.exact_mean_absolute_magnetization is not None
        assert row.mean_absolute_magnetization == pytest.approx(
            row.exact_mean_absolute_magnetization,
            abs=0.04,
        )
        assert row.spin_updates == 9_000 * 4
        assert row.effective_sample_size > 0.0
        assert row.ess_per_spin_update > 0.0
        assert row.ess_per_second > 0.0

    metropolis = next(row for row in rows if row.method == "single-spin Metropolis")
    assert metropolis.acceptance_rate is not None
    assert 0.0 < metropolis.acceptance_rate < 1.0
