import numpy as np
import pytest

from sampler_lab.markov import (
    asymptotic_variance,
    autocorrelations,
    finite_sample_mean_variance,
    integrated_autocorrelation_time,
    solve_poisson_equation,
)
from sampler_lab.models import two_state_chain


def test_two_state_poisson_solution_and_exact_iat() -> None:
    chain = two_state_chain(0.2, 0.3)
    observable = np.array([0.0, 1.0])
    result = solve_poisson_equation(chain, observable)

    assert chain.invariant_distribution() == pytest.approx([0.6, 0.4])
    assert result.stationary_mean == pytest.approx(0.4)
    assert result.residual_norm < 1e-13
    assert result.centering_residual < 1e-13
    assert chain.generator @ result.solution == pytest.approx(result.centered_forcing)

    # The nonconstant eigenvalue is lambda = 1 - 0.2 - 0.3 = 0.5.
    assert autocorrelations(chain, observable, 5) == pytest.approx(0.5 ** np.arange(6))
    assert integrated_autocorrelation_time(chain, observable) == pytest.approx(3.0)
    assert asymptotic_variance(chain, observable) == pytest.approx(0.24 * 3.0)


def test_finite_sample_mean_variance_matches_covariance_sum() -> None:
    chain = two_state_chain(0.2, 0.3)
    observable = np.array([0.0, 1.0])
    n_samples = 4
    eigenvalue = 0.5
    stationary_variance = 0.24
    expected = (
        stationary_variance
        / n_samples**2
        * (
            n_samples
            + 2.0 * sum((n_samples - lag) * eigenvalue**lag for lag in range(1, n_samples))
        )
    )
    assert finite_sample_mean_variance(chain, observable, n_samples) == pytest.approx(expected)


def test_periodic_alternation_has_zero_asymptotic_variance() -> None:
    chain = two_state_chain(1.0, 1.0)
    observable = np.array([-1.0, 1.0])

    assert chain.period == 2
    assert autocorrelations(chain, observable, 5) == pytest.approx(
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    )
    assert asymptotic_variance(chain, observable) == pytest.approx(0.0, abs=1e-14)
    assert integrated_autocorrelation_time(chain, observable) == pytest.approx(0.0, abs=1e-14)
    assert finite_sample_mean_variance(chain, observable, 6) == pytest.approx(0.0, abs=1e-14)
    assert finite_sample_mean_variance(chain, observable, 5) == pytest.approx(1.0 / 25.0)


def test_reversible_worst_case_iat_matches_two_state_mode() -> None:
    chain = two_state_chain(0.2, 0.3)
    summary = chain.spectral_summary()

    assert summary.poincare_gap == pytest.approx(0.5)
    assert summary.absolute_spectral_gap == pytest.approx(0.5)
    assert summary.worst_case_iat == pytest.approx(3.0)


def test_constant_observable_has_zero_asymptotic_variance_but_no_iat() -> None:
    chain = two_state_chain(0.2, 0.3)
    observable = np.ones(2)

    assert asymptotic_variance(chain, observable) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="constant observable"):
        integrated_autocorrelation_time(chain, observable)


def test_poisson_analysis_rejects_multiple_stationary_classes() -> None:
    chain = two_state_chain(0.0, 0.0)
    with pytest.raises(ValueError, match="unique invariant"):
        solve_poisson_equation(chain, [0.0, 1.0])


def test_poisson_martingale_decomposition_is_algebraically_exact() -> None:
    from sampler_lab.markov import poisson_martingale_decomposition

    chain = two_state_chain(0.2, 0.3)
    states = np.array([0, 1, 1, 0, 0, 1], dtype=np.int64)
    decomposition = poisson_martingale_decomposition(chain, [0.0, 1.0], states)

    assert decomposition.increments.shape == (states.size - 1,)
    assert decomposition.centered_sum == pytest.approx(
        decomposition.boundary_term - decomposition.martingale_term
    )
    assert decomposition.residual == pytest.approx(0.0, abs=1e-14)
