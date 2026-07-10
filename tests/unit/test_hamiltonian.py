import numpy as np
import pytest

from sampler_lab import OperationCounter
from sampler_lab.dynamics import (
    ConstantSkewMatrix,
    FunctionalSkewMatrix,
    HamiltonianSystem,
    MassMatrix,
    PhaseSpaceState,
    canonical_symplectic_matrix,
    conservative_skew_drift,
    finite_difference_jacobian,
    gaussian_exact_flow_matrix,
    gaussian_hamiltonian_analysis,
    gaussian_hamiltonian_frequencies,
    gaussian_leapfrog_matrix,
    leapfrog_integrate,
    leapfrog_map,
    leapfrog_reversibility_error,
    symplecticity_error,
    volume_preservation_error,
)
from sampler_lab.models import GaussianTarget


def test_phase_space_state_round_trip_and_immutability() -> None:
    state = PhaseSpaceState([1.0, 2.0], [3.0, 4.0])
    restored = PhaseSpaceState.from_array(state.as_array())

    np.testing.assert_allclose(restored.position, state.position)
    np.testing.assert_allclose(restored.momentum, state.momentum)
    with pytest.raises(ValueError):
        state.position[0] = 0.0


def test_mass_matrix_kinetic_energy_velocity_and_sampling_formula() -> None:
    mass = MassMatrix([[4.0, 0.0], [0.0, 1.0]])
    momentum = np.array([2.0, -1.0])

    assert mass.kinetic_energy(momentum) == pytest.approx(1.0)
    np.testing.assert_allclose(mass.velocity(momentum), [0.5, -1.0])

    expected_rng = np.random.default_rng(3)
    expected = mass.cholesky @ expected_rng.normal(size=2)
    counter = OperationCounter()
    actual = mass.sample_momentum(np.random.default_rng(3), counter=counter)
    np.testing.assert_allclose(actual, expected)
    assert counter.normal_draws == 2


def test_canonical_hamiltonian_vector_field_is_energy_orthogonal() -> None:
    target = GaussianTarget([0.0, 0.0], [[2.0, 0.3], [0.3, 1.0]])
    system = HamiltonianSystem(target, MassMatrix([[1.5, 0.1], [0.1, 0.8]]))
    state = PhaseSpaceState([0.6, -0.2], [1.1, 0.4])
    field = system.vector_field(state)
    gradient_hamiltonian = np.concatenate(
        (system.potential_gradient(state.position), system.mass.velocity(state.momentum))
    )

    assert gradient_hamiltonian @ field.as_array() == pytest.approx(0.0, abs=1e-12)
    omega = canonical_symplectic_matrix(2)
    np.testing.assert_allclose(field.as_array(), omega @ gradient_hamiltonian)


def test_skew_drift_includes_row_divergence_correction() -> None:
    state = np.array([2.0, 3.0])
    gradient = state.copy()
    field = FunctionalSkewMatrix(
        matrix_function=lambda x: np.array([[0.0, x[0]], [-x[0], 0.0]]),
        divergence_function=lambda _x: np.array([0.0, -1.0]),
    )

    drift = conservative_skew_drift(state, gradient, field)
    np.testing.assert_allclose(drift, [6.0, -3.0])

    constant = ConstantSkewMatrix([[0.0, 1.0], [-1.0, 0.0]])
    np.testing.assert_allclose(
        conservative_skew_drift(state, gradient, constant),
        [3.0, -2.0],
    )


def test_one_dimensional_leapfrog_matches_velocity_verlet_formula() -> None:
    target = GaussianTarget([0.0], [[1.0]])
    system = HamiltonianSystem(target, MassMatrix.identity(1))
    state = PhaseSpaceState([1.2], [-0.7])
    step_size = 0.3

    result = leapfrog_integrate(system, state, step_size, 1)
    p_half = -0.7 - 0.5 * step_size * 1.2
    q_new = 1.2 + step_size * p_half
    p_new = p_half - 0.5 * step_size * q_new

    assert result.final_state.position[0] == pytest.approx(q_new)
    assert result.final_state.momentum[0] == pytest.approx(p_new)
    assert result.energy_error == pytest.approx(
        0.5 * (q_new**2 + p_new**2) - 0.5 * (1.2**2 + 0.7**2)
    )


def test_gaussian_leapfrog_matrix_matches_numerical_integrator() -> None:
    target = GaussianTarget([0.5, -0.25], [[1.0, 0.2], [0.2, 0.7]])
    mass = MassMatrix(target.precision_matrix)
    system = HamiltonianSystem(target, mass)
    state = PhaseSpaceState([0.8, -0.1], [0.4, -1.2])
    step_size = 0.15
    n_steps = 5

    result = leapfrog_integrate(system, state, step_size, n_steps)
    matrix = gaussian_leapfrog_matrix(
        target,
        step_size,
        n_steps,
        mass=mass,
    )
    centered = np.concatenate((state.position - target.mean_vector, state.momentum))
    expected = matrix @ centered

    np.testing.assert_allclose(
        result.final_state.position - target.mean_vector,
        expected[:2],
        atol=1e-12,
    )
    np.testing.assert_allclose(result.final_state.momentum, expected[2:], atol=1e-12)


def test_leapfrog_is_reversible_volume_preserving_and_symplectic() -> None:
    target = GaussianTarget([0.0, 0.0], [[1.0, 0.1], [0.1, 0.5]])
    system = HamiltonianSystem(target, MassMatrix(target.precision_matrix))
    state = PhaseSpaceState([0.4, -0.7], [1.2, 0.3])
    step_size = 0.12
    n_steps = 6

    error = leapfrog_reversibility_error(system, state, step_size, n_steps)
    jacobian = finite_difference_jacobian(
        lambda value: leapfrog_map(system, value, step_size, n_steps),
        state.as_array(),
    )

    assert error < 1e-12
    assert volume_preservation_error(jacobian) < 1e-8
    assert symplecticity_error(jacobian) < 1e-8


def test_exact_gaussian_flow_conserves_energy_and_is_symplectic() -> None:
    target = GaussianTarget([0.0, 0.0], [[1.0, 0.2], [0.2, 0.6]])
    mass = MassMatrix([[2.0, 0.1], [0.1, 1.0]])
    flow = gaussian_exact_flow_matrix(target, 0.73, mass=mass)
    state = np.array([0.4, -0.8, 1.1, 0.2])
    final = flow @ state

    initial_energy = 0.5 * state[:2] @ target.precision_matrix @ state[:2]
    initial_energy += mass.kinetic_energy(state[2:])
    final_energy = 0.5 * final[:2] @ target.precision_matrix @ final[:2]
    final_energy += mass.kinetic_energy(final[2:])

    assert final_energy == pytest.approx(initial_energy, abs=1e-12)
    assert symplecticity_error(flow) < 1e-12
    assert volume_preservation_error(flow) < 1e-12


def test_precision_mass_equalizes_gaussian_frequencies_and_stability() -> None:
    target = GaussianTarget([0.0, 0.0], np.diag([1.0, 0.01]))

    identity_frequencies = gaussian_hamiltonian_frequencies(target)
    precision_frequencies = gaussian_hamiltonian_frequencies(
        target,
        target.precision_matrix,
    )
    analysis = gaussian_hamiltonian_analysis(
        target,
        0.5,
        4,
        mass=target.precision_matrix,
    )

    np.testing.assert_allclose(identity_frequencies, [1.0, 10.0])
    np.testing.assert_allclose(precision_frequencies, [1.0, 1.0])
    assert analysis.maximum_stable_step_size == pytest.approx(2.0)
    assert analysis.stable
