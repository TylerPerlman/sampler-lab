import pytest

from sampler_lab.experiments import run_markov_theory_experiment


@pytest.mark.statistical
def test_markov_laboratory_matches_exact_finite_sample_errors() -> None:
    rows = run_markov_theory_experiment(
        n_states=8,
        n_samples=80,
        n_replicates=2_000,
        seed=2022,
    )
    reversible, directed, cycle = rows

    assert reversible.reversible
    assert not directed.reversible
    assert directed.integrated_autocorrelation_time < reversible.integrated_autocorrelation_time
    assert cycle.period == 8
    assert cycle.integrated_autocorrelation_time == pytest.approx(0.0, abs=1e-12)
    assert cycle.exact_finite_sample_standard_error == pytest.approx(0.0, abs=1e-12)

    for row in (reversible, directed):
        assert row.empirical_standard_error == pytest.approx(
            row.exact_finite_sample_standard_error,
            rel=0.08,
        )
