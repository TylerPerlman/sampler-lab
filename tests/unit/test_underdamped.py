import numpy as np
import pytest

from sampler_lab import OperationCounter
from sampler_lab.dynamics import (
    HamiltonianSystem,
    MassMatrix,
    MetropolizedUnderdampedLangevinKernel,
    PhaseSpaceState,
    UnderdampedLangevinKernel,
    leapfrog_integrate,
    ornstein_uhlenbeck_momentum_step,
    ornstein_uhlenbeck_persistence,
    underdamped_generator_value,
)
from sampler_lab.models import GaussianTarget


def test_ou_persistence_and_exact_step_formula() -> None:
    mass = MassMatrix([[4.0]])
    momentum = np.array([2.0])
    friction = 0.7
    duration = 0.3
    persistence = np.exp(-friction * duration)
    expected_rng = np.random.default_rng(7)
    innovation = mass.cholesky @ expected_rng.normal(size=1)
    expected = persistence * momentum + np.sqrt(1.0 - persistence**2) * innovation

    counter = OperationCounter()
    actual = ornstein_uhlenbeck_momentum_step(
        momentum,
        mass,
        np.random.default_rng(7),
        friction=friction,
        duration=duration,
        counter=counter,
    )

    assert ornstein_uhlenbeck_persistence(friction, duration) == pytest.approx(persistence)
    np.testing.assert_allclose(actual, expected)
    assert counter.normal_draws == 1


def test_baoab_with_zero_friction_equals_one_leapfrog_step() -> None:
    target = GaussianTarget([0.0], [[1.0]])
    mass = MassMatrix([[2.0]])
    state = PhaseSpaceState([0.7], [-0.4])
    step_size = 0.2
    system = HamiltonianSystem(target, mass)
    expected = leapfrog_integrate(system, state, step_size, 1).final_state

    transition = UnderdampedLangevinKernel(
        target,
        step_size,
        friction=0.0,
        mass_matrix=mass,
    ).step(state.as_array(), np.random.default_rng(2))

    np.testing.assert_allclose(transition.state, expected.as_array(), atol=1e-12)


def test_underdamped_generator_separates_hamiltonian_and_ou_parts() -> None:
    target = GaussianTarget([0.0], [[1.0]])
    state = PhaseSpaceState([2.0], [3.0])
    mass = MassMatrix([[2.0]])
    # f(q,p)=q^2+p^2: gradients are 2q and 2p, Hess_pp=2.
    result = underdamped_generator_value(
        target,
        state,
        gradient_position=[4.0],
        gradient_momentum=[6.0],
        hessian_momentum=[[2.0]],
        mass=mass,
        friction=0.5,
    )

    expected_hamiltonian = (3.0 / 2.0) * 4.0 + (-2.0) * 6.0
    expected_ou = -0.5 * 3.0 * 6.0 + 0.5 * 2.0 * 2.0
    assert result.hamiltonian == pytest.approx(expected_hamiltonian)
    assert result.ornstein_uhlenbeck == pytest.approx(expected_ou)
    assert result.total == pytest.approx(expected_hamiltonian + expected_ou)


def test_metropolized_underdamped_exposes_transformed_rejection() -> None:
    kernel = MetropolizedUnderdampedLangevinKernel(
        GaussianTarget([0.0], [[1.0]]),
        step_size=3.0,
        friction=0.0,
    )
    transition = kernel.step(np.array([2.0, 1.0]), np.random.default_rng(0))

    assert not transition.accepted
    np.testing.assert_allclose(transition.state, [2.0, -1.0])
