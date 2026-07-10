import numpy as np
import pytest

from sampler_lab import OperationCounter
from sampler_lab.geometry import (
    AffineMap,
    AffineTransformedTarget,
    MetropolizedStochasticNewtonKernel,
    StochasticNewtonProposal,
    finite_difference_hessian_from_gradient,
    repair_positive_definite,
)
from sampler_lab.models import GaussianTarget, RosenbrockTarget


def test_positive_definite_repair_modes() -> None:
    matrix = np.array([[2.0, 0.0], [0.0, -3.0]])

    clipped = repair_positive_definite(matrix, method="clip", minimum_eigenvalue=0.2)
    absolute = repair_positive_definite(matrix, method="absolute", minimum_eigenvalue=0.2)

    np.testing.assert_allclose(np.linalg.eigvalsh(clipped.matrix), [0.2, 2.0])
    np.testing.assert_allclose(np.linalg.eigvalsh(absolute.matrix), [2.0, 3.0])
    with pytest.raises(np.linalg.LinAlgError):
        repair_positive_definite(matrix, method="raise")


def test_rosenbrock_analytic_hessian_matches_gradient_differences() -> None:
    target = RosenbrockTarget()
    point = np.array([-0.7, 1.3])

    approximate = finite_difference_hessian_from_gradient(target.grad_log_prob, point)

    np.testing.assert_allclose(approximate, target.hessian_log_prob(point), rtol=2e-6, atol=2e-6)


def test_stochastic_newton_gaussian_local_geometry_is_exact() -> None:
    covariance = np.array([[3.0, 0.5], [0.5, 1.0]])
    target = GaussianTarget([1.0, -2.0], covariance)
    proposal = StochasticNewtonProposal(target, 0.25, repair_method="raise")
    state = np.array([4.0, 0.0])

    evaluation = proposal.evaluate_at(state)

    expected_mean = state + 0.25 * covariance @ target.grad_log_prob(state)
    np.testing.assert_allclose(evaluation.metric, covariance)
    np.testing.assert_allclose(evaluation.mean, expected_mean)
    np.testing.assert_allclose(evaluation.covariance, 0.5 * covariance)
    assert evaluation.repair_norm == pytest.approx(0.0, abs=1e-12)


def test_stochastic_newton_parameters_transform_affinely() -> None:
    target = GaussianTarget([0.5, -1.0], [[2.0, 0.3], [0.3, 0.8]])
    mapping = AffineMap([[1.2, 0.4], [-0.7, 2.0]], [3.0, -2.0])
    transformed_target = AffineTransformedTarget(target, mapping)
    original = StochasticNewtonProposal(target, 0.3, repair_method="raise")
    transformed = StochasticNewtonProposal(transformed_target, 0.3, repair_method="raise")
    state = np.array([1.1, -0.4])

    base_evaluation = original.evaluate_at(state)
    transformed_evaluation = transformed.evaluate_at(mapping.forward(state))

    np.testing.assert_allclose(
        transformed_evaluation.mean,
        mapping.forward(base_evaluation.mean),
        atol=1e-11,
    )
    np.testing.assert_allclose(
        transformed_evaluation.metric,
        mapping.transform_covariance(base_evaluation.metric),
        atol=1e-11,
    )
    np.testing.assert_allclose(
        transformed_evaluation.covariance,
        mapping.transform_covariance(base_evaluation.covariance),
        atol=1e-11,
    )


def test_metropolized_stochastic_newton_reuses_endpoint_geometry() -> None:
    target = GaussianTarget([0.0, 0.0], np.eye(2))
    counter = OperationCounter()
    kernel = MetropolizedStochasticNewtonKernel(
        target,
        0.4,
        repair_method="raise",
        counter=counter,
    )

    transition = kernel.step(np.array([1.0, -1.0]), np.random.default_rng(8))

    assert transition.accepted is not None
    assert counter.gradient_evaluations == 2
    assert counter.hessian_evaluations == 2
    assert counter.matrix_factorizations == 2
    assert counter.proposal_density_evaluations == 2
    assert counter.log_density_evaluations == 2
