import numpy as np
import pytest

from sampler_lab.core.protocols import MarkovKernel
from sampler_lab.dynamics import (
    HamiltonianMonteCarloKernel,
    HamiltonianSystem,
    MassMatrix,
    MetropolizedUnderdampedLangevinKernel,
    PhaseSpaceState,
    UnderdampedLangevinKernel,
    leapfrog_integrate,
)
from sampler_lab.mcmc import run_chain
from sampler_lab.models import GaussianTarget, XYModel, wrap_angles


@pytest.mark.statistical
def test_hmc_recovers_correlated_gaussian_law() -> None:
    covariance = np.array([[1.0, 0.3], [0.3, 2.0]])
    target = GaussianTarget([0.0, 0.0], covariance)
    trajectory = run_chain(
        HamiltonianMonteCarloKernel(
            target,
            step_size=0.5,
            n_leapfrog_steps=3,
            mass_matrix=target.precision_matrix,
        ),
        np.zeros(2),
        np.random.default_rng(2022),
        n_steps=10_000,
    )
    samples = trajectory.states[1_000:]

    np.testing.assert_allclose(np.mean(samples, axis=0), [0.0, 0.0], atol=0.05)
    np.testing.assert_allclose(np.cov(samples.T), covariance, atol=0.09)
    assert trajectory.acceptance_rate is not None
    assert trajectory.acceptance_rate > 0.94


@pytest.mark.statistical
def test_leapfrog_global_energy_error_is_second_order() -> None:
    target = GaussianTarget([0.0], [[1.0]])
    system = HamiltonianSystem(target, MassMatrix.identity(1))
    rng = np.random.default_rng(5)
    phase_states = [PhaseSpaceState(rng.normal(size=1), rng.normal(size=1)) for _ in range(400)]

    def rms_error(step_size: float) -> float:
        n_steps = round(1.2 / step_size)
        errors = [
            leapfrog_integrate(system, state, step_size, n_steps).energy_error
            for state in phase_states
        ]
        return float(np.sqrt(np.mean(np.square(errors))))

    coarse = rms_error(0.2)
    fine = rms_error(0.1)

    assert coarse / fine == pytest.approx(4.0, rel=0.25)


@pytest.mark.statistical
def test_baoab_and_metropolized_underdamped_recover_gaussian_phase_law() -> None:
    target = GaussianTarget([0.0], [[1.0]])
    configurations: tuple[MarkovKernel, ...] = (
        UnderdampedLangevinKernel(target, 0.2, friction=1.0),
        MetropolizedUnderdampedLangevinKernel(target, 0.5, friction=1.0),
    )
    for seed, kernel in enumerate(configurations, start=2023):
        trajectory = run_chain(
            kernel,
            np.array([0.0, 0.0]),
            np.random.default_rng(seed),
            n_steps=15_000,
        )
        phase_samples = trajectory.states[1_000:]
        np.testing.assert_allclose(np.mean(phase_samples, axis=0), [0.0, 0.0], atol=0.06)
        np.testing.assert_allclose(np.cov(phase_samples.T), np.eye(2), atol=0.08)

    assert (
        configurations[1].step(np.array([0.0, 0.0]), np.random.default_rng(9)).accepted is not None
    )


@pytest.mark.statistical
def test_hmc_recovers_exact_single_site_xy_response() -> None:
    model = XYModel(
        size=1,
        inverse_temperature=1.0,
        coupling=0.0,
        external_field=1.5,
    )
    trajectory = run_chain(
        HamiltonianMonteCarloKernel(
            model,
            step_size=0.35,
            n_leapfrog_steps=4,
            position_map=wrap_angles,
        ),
        np.array([0.0]),
        np.random.default_rng(2024),
        n_steps=10_000,
    )
    angles = trajectory.states[1_000:, 0]

    assert float(np.mean(np.cos(angles))) == pytest.approx(
        model.exact_single_site_mean_cosine(),
        abs=0.025,
    )
    assert float(np.mean(np.sin(angles))) == pytest.approx(0.0, abs=0.025)
    assert trajectory.acceptance_rate is not None
    assert trajectory.acceptance_rate > 0.95
