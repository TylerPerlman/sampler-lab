import numpy as np
import pytest

from sampler_lab.dynamics import (
    MetropolisAdjustedLangevinKernel,
    UnadjustedLangevinKernel,
    estimate_poisson_invariant_bias,
    gaussian_ula_analysis,
)
from sampler_lab.mcmc import run_chain
from sampler_lab.models import GaussianTarget


@pytest.mark.statistical
def test_ula_reproduces_its_biased_gaussian_stationary_variance() -> None:
    target = GaussianTarget([0.0], [[1.0]])
    step_size = 0.4
    analysis = gaussian_ula_analysis(target, step_size)
    assert analysis.stationary_covariance is not None
    stationary_variance = float(analysis.stationary_covariance[0, 0])
    initial = np.array([np.sqrt(stationary_variance) * np.random.default_rng(1).normal()])
    trajectory = run_chain(
        UnadjustedLangevinKernel(target, step_size),
        initial,
        np.random.default_rng(2),
        n_steps=30_000,
    )
    variance = float(np.var(trajectory.states[1_000:, 0]))

    assert variance == pytest.approx(stationary_variance, abs=0.045)
    assert abs(variance - 1.0) > 0.12


@pytest.mark.statistical
def test_mala_removes_ula_stationary_bias_on_standard_normal() -> None:
    target = GaussianTarget([0.0], [[1.0]])
    trajectory = run_chain(
        MetropolisAdjustedLangevinKernel(target, 0.4),
        np.array([0.0]),
        np.random.default_rng(2022),
        n_steps=30_000,
    )
    samples = trajectory.states[1_000:, 0]

    assert float(np.mean(samples)) == pytest.approx(0.0, abs=0.035)
    assert float(np.var(samples)) == pytest.approx(1.0, abs=0.05)
    assert trajectory.acceptance_rate is not None
    assert trajectory.acceptance_rate > 0.9


@pytest.mark.statistical
def test_poisson_identity_estimates_ula_quadratic_bias() -> None:
    target = GaussianTarget([0.0], [[1.0]])
    step_size = 0.4
    analysis = gaussian_ula_analysis(target, step_size)
    assert analysis.stationary_covariance is not None
    stationary_variance = float(analysis.stationary_covariance[0, 0])
    states = np.random.default_rng(10).normal(scale=np.sqrt(stationary_variance), size=(8_000, 1))
    estimate = estimate_poisson_invariant_bias(
        UnadjustedLangevinKernel(target, step_size),
        poisson_solution=lambda x: -0.5 * float(x[0] ** 2),
        continuous_generator_on_solution=lambda x: float(x[0] ** 2 - 1.0),
        stationary_samples=states,
        step_size=step_size,
        rng=np.random.default_rng(11),
    )

    assert estimate.value == pytest.approx(stationary_variance - 1.0, abs=0.035)
