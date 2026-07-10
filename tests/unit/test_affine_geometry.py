import numpy as np
import pytest

from sampler_lab.geometry import (
    AffineMap,
    AffineTransformedTarget,
    affine_equivariance_error,
    gaussian_conditional,
    gaussian_whitening_map,
    matrix_condition_number,
    whiten_gaussian_target,
)
from sampler_lab.models import GaussianTarget


def test_affine_map_round_trip_for_points_and_batches() -> None:
    mapping = AffineMap([[2.0, 1.0], [-1.0, 3.0]], [0.5, -2.0])
    points = np.array([[1.0, 2.0], [-3.0, 0.5]])

    transformed = mapping.forward(points)
    recovered = mapping.inverse(transformed)

    np.testing.assert_allclose(recovered, points)
    assert mapping.log_abs_determinant == pytest.approx(np.log(7.0))
    assert affine_equivariance_error(points, transformed, mapping) < 1e-12


def test_affine_target_matches_explicit_pushforward_gaussian() -> None:
    base = GaussianTarget([1.0, -2.0], [[2.0, 0.4], [0.4, 0.8]])
    mapping = AffineMap([[1.5, -0.3], [0.7, 2.0]], [-1.0, 0.25])
    transformed = AffineTransformedTarget(base, mapping)
    explicit = GaussianTarget(
        mapping.forward(base.mean_vector),
        mapping.transform_covariance(base.covariance_matrix),
    )
    point = np.array([0.8, -1.4])

    assert transformed.log_prob(point) == pytest.approx(explicit.log_prob(point))
    np.testing.assert_allclose(transformed.grad_log_prob(point), explicit.grad_log_prob(point))
    np.testing.assert_allclose(
        transformed.hessian_log_prob(point),
        explicit.hessian_log_prob(point),
    )


def test_gaussian_whitening_produces_identity_geometry() -> None:
    target = GaussianTarget([2.0, -1.0], [[9.0, 1.2], [1.2, 0.5]])
    mapping = gaussian_whitening_map(target)
    whitened = whiten_gaussian_target(target)
    explicit = GaussianTarget([0.0, 0.0], np.eye(2))

    np.testing.assert_allclose(mapping.forward(target.mean_vector), np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(
        mapping.transform_covariance(target.covariance_matrix),
        np.eye(2),
        atol=1e-12,
    )
    point = np.array([0.4, -1.1])
    assert whitened.log_prob(point) == pytest.approx(explicit.log_prob(point))
    assert matrix_condition_number(target.covariance_matrix) > 1.0
    assert matrix_condition_number(np.eye(2)) == pytest.approx(1.0)


def test_gaussian_conditional_matches_block_formula() -> None:
    target = GaussianTarget(
        [1.0, 2.0, -1.0],
        [[4.0, 1.0, 0.5], [1.0, 3.0, -0.2], [0.5, -0.2, 2.0]],
    )
    result = gaussian_conditional(target, [1], [5.0])

    covariance = target.covariance_matrix
    expected_mean = np.array([1.0, -1.0]) + covariance[np.ix_([0, 2], [1])].ravel() * (
        (5.0 - 2.0) / covariance[1, 1]
    )
    expected_covariance = covariance[np.ix_([0, 2], [0, 2])] - (
        covariance[np.ix_([0, 2], [1])] @ covariance[np.ix_([1], [0, 2])] / covariance[1, 1]
    )

    np.testing.assert_array_equal(result.remaining_indices, [0, 2])
    np.testing.assert_allclose(result.target.mean_vector, expected_mean)
    np.testing.assert_allclose(result.target.covariance_matrix, expected_covariance)
