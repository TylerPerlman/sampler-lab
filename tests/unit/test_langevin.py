import numpy as np
import pytest

from sampler_lab import OperationCounter
from sampler_lab.dynamics import (
    ConstantPreconditioner,
    FunctionalPreconditioner,
    MetropolisAdjustedLangevinKernel,
    UnadjustedLangevinKernel,
    gaussian_log_transition_density,
    overdamped_langevin_drift,
)
from sampler_lab.models import GaussianTarget


def test_constant_preconditioned_langevin_drift() -> None:
    target = GaussianTarget([0.0, 0.0], [[2.0, 0.0], [0.0, 0.5]])
    matrix = np.array([[2.0, 0.0], [0.0, 0.25]])
    drift = overdamped_langevin_drift(
        target,
        np.array([2.0, -1.0]),
        preconditioner=matrix,
    )

    np.testing.assert_allclose(drift, [-2.0, 0.5])


def test_position_dependent_divergence_correction_is_explicit() -> None:
    target = GaussianTarget([0.0], [[1.0]])
    geometry = FunctionalPreconditioner(
        matrix_function=lambda x: np.array([[1.0 + x[0] ** 2]]),
        divergence_function=lambda x: np.array([2.0 * x[0]]),
    )

    corrected = overdamped_langevin_drift(
        target,
        np.array([2.0]),
        preconditioner=geometry,
    )
    uncorrected = overdamped_langevin_drift(
        target,
        np.array([2.0]),
        preconditioner=geometry,
        include_divergence=False,
    )

    assert corrected[0] == pytest.approx(-6.0)
    assert uncorrected[0] == pytest.approx(-10.0)


def test_ula_step_matches_its_euler_formula() -> None:
    target = GaussianTarget([0.0], [[1.0]])
    step_size = 0.2
    kernel = UnadjustedLangevinKernel(target, step_size)
    expected_rng = np.random.default_rng(7)
    normal = float(expected_rng.normal())
    transition = kernel.step(np.array([1.5]), np.random.default_rng(7))

    expected = 1.5 - step_size * 1.5 + np.sqrt(2.0 * step_size) * normal
    assert transition.state[0] == pytest.approx(expected)
    assert transition.accepted is None


def test_full_gaussian_transition_density() -> None:
    value = gaussian_log_transition_density(
        destination=np.array([2.0]),
        mean=np.array([1.0]),
        covariance_cholesky=np.array([[2.0]]),
        log_determinant_covariance=np.log(4.0),
    )

    expected = -0.5 * (np.log(2.0 * np.pi) + np.log(4.0) + 0.25)
    assert value == pytest.approx(expected)


def test_mala_records_asymmetric_proposal_terms_and_costs() -> None:
    counter = OperationCounter()
    target = GaussianTarget([0.0], [[1.0]])
    kernel = MetropolisAdjustedLangevinKernel(target, 0.4, counter=counter)
    transition = kernel.step(np.array([1.0]), np.random.default_rng(11))

    diagnostics = transition.diagnostics
    manual_ratio = (
        diagnostics["proposed_log_target"]
        - diagnostics["current_log_target"]
        + diagnostics["reverse_log_proposal"]
        - diagnostics["forward_log_proposal"]
    )
    assert transition.log_acceptance_ratio == pytest.approx(manual_ratio)
    assert transition.accepted is not None
    assert counter.gradient_evaluations == 2
    assert counter.log_density_evaluations == 2
    assert counter.proposal_density_evaluations == 2
    assert counter.normal_draws == 1
    assert counter.uniform_draws == 1


def test_constant_preconditioner_rejects_dimension_mismatch() -> None:
    geometry = ConstantPreconditioner(np.eye(2))
    with pytest.raises(ValueError, match="dimension"):
        geometry.matrix_at(np.array([0.0]))


class PositiveHalfNormalTarget:
    def log_prob(self, x: np.ndarray) -> float:
        return float(-0.5 * x[0] ** 2) if x[0] >= 0.0 else float("-inf")

    def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
        if x[0] < 0.0:
            raise AssertionError("gradient must not be evaluated outside support")
        return np.array([-x[0]])


def test_mala_rejects_out_of_support_proposal_without_gradient_evaluation() -> None:
    counter = OperationCounter()
    kernel = MetropolisAdjustedLangevinKernel(
        PositiveHalfNormalTarget(),
        1.0,
        counter=counter,
    )
    transition = kernel.step(np.array([0.0]), np.random.default_rng(4))

    assert not transition.accepted
    assert transition.state[0] == pytest.approx(0.0)
    assert transition.log_acceptance_ratio == float("-inf")
    assert counter.gradient_evaluations == 1
    assert counter.proposal_density_evaluations == 1


def test_functional_preconditioner_is_factored_once_per_ula_state() -> None:
    calls = 0

    def matrix_function(x: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.array([[1.0 + x[0] ** 2]])

    geometry = FunctionalPreconditioner(
        matrix_function=matrix_function,
        divergence_function=lambda x: np.array([2.0 * x[0]]),
    )
    counter = OperationCounter()
    kernel = UnadjustedLangevinKernel(
        GaussianTarget([0.0], [[1.0]]),
        0.1,
        preconditioner=geometry,
        counter=counter,
    )
    kernel.step(np.array([0.5]), np.random.default_rng(1))

    assert calls == 1
    assert counter.matrix_factorizations == 1
