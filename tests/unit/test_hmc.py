import numpy as np
import pytest

from sampler_lab import OperationCounter
from sampler_lab.dynamics import (
    HamiltonianMonteCarloKernel,
    HamiltonianPhaseDensity,
    HamiltonianSystem,
    LeapfrogMomentumFlipInvolution,
    MassMatrix,
    PersistentHamiltonianKernel,
    partially_refresh_momentum,
)
from sampler_lab.models import GaussianTarget


def test_partial_refresh_matches_gaussian_ar1_formula() -> None:
    mass = MassMatrix([[4.0, 0.0], [0.0, 1.0]])
    momentum = np.array([1.0, -2.0])
    persistence = 0.6
    expected_rng = np.random.default_rng(8)
    innovation = mass.cholesky @ expected_rng.normal(size=2)
    expected = persistence * momentum + np.sqrt(1.0 - persistence**2) * innovation

    counter = OperationCounter()
    actual = partially_refresh_momentum(
        momentum,
        mass,
        np.random.default_rng(8),
        persistence=persistence,
        counter=counter,
    )

    np.testing.assert_allclose(actual, expected)
    assert counter.normal_draws == 2


def test_leapfrog_momentum_flip_map_is_an_involution() -> None:
    target = GaussianTarget([0.0], [[1.0]])
    system = HamiltonianSystem(target, MassMatrix.identity(1))
    involution = LeapfrogMomentumFlipInvolution(system, 0.2, 7)
    state = np.array([0.7, -1.1])

    twice = involution.apply(involution.apply(state))
    np.testing.assert_allclose(twice, state, atol=1e-12)


def test_hamiltonian_phase_density_matches_target_minus_kinetic_energy() -> None:
    target = GaussianTarget([0.0], [[2.0]])
    mass = MassMatrix([[3.0]])
    density = HamiltonianPhaseDensity(target, mass)
    state = np.array([1.0, 1.5])

    expected = target.log_prob(state[:1]) - mass.kinetic_energy(state[1:])
    assert density.log_prob(state) == pytest.approx(expected)


def test_hmc_records_energy_acceptance_and_operation_costs() -> None:
    target = GaussianTarget([0.0, 0.0], np.eye(2))
    counter = OperationCounter()
    kernel = HamiltonianMonteCarloKernel(
        target,
        step_size=0.2,
        n_leapfrog_steps=4,
        counter=counter,
    )
    transition = kernel.step(np.array([0.5, -0.2]), np.random.default_rng(4))

    assert transition.accepted is not None
    assert transition.log_acceptance_ratio == pytest.approx(-transition.diagnostics["energy_error"])
    assert counter.normal_draws == 2
    assert counter.uniform_draws == 1
    assert counter.gradient_evaluations == 5
    assert counter.log_density_evaluations == 2


def test_persistent_hmc_flips_refreshed_momentum_on_rejection() -> None:
    target = GaussianTarget([0.0], [[1.0]])
    kernel = PersistentHamiltonianKernel(
        target,
        step_size=3.0,
        n_leapfrog_steps=1,
        momentum_persistence=1.0,
    )
    transition = kernel.step(np.array([2.0, 1.0]), np.random.default_rng(0))

    assert not transition.accepted
    np.testing.assert_allclose(transition.state, [2.0, -1.0])
    assert transition.diagnostics["rejection_momentum_flipped"] == pytest.approx(1.0)


def test_full_refresh_ignores_input_momentum_distributionally() -> None:
    target = GaussianTarget([0.0], [[1.0]])
    first = PersistentHamiltonianKernel(
        target,
        step_size=0.1,
        n_leapfrog_steps=1,
        momentum_persistence=0.0,
    ).step(np.array([0.2, -100.0]), np.random.default_rng(12))
    second = PersistentHamiltonianKernel(
        target,
        step_size=0.1,
        n_leapfrog_steps=1,
        momentum_persistence=0.0,
    ).step(np.array([0.2, 100.0]), np.random.default_rng(12))

    np.testing.assert_allclose(first.state, second.state)
