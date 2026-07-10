from __future__ import annotations

import numpy as np

from sampler_lab.models.bimodal_funnel import BimodalFunnelTarget
from sampler_lab.models.funnel import FunnelTarget, seeded_orthogonal_matrix
from sampler_lab.models.gaussian_mixture import GaussianMixtureTarget


def _finite_difference_gradient(
    target: object, point: np.ndarray, epsilon: float = 1e-6
) -> np.ndarray:
    gradient = np.empty_like(point)
    for index in range(point.size):
        plus = point.copy()
        minus = point.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (target.log_prob(plus) - target.log_prob(minus)) / (2.0 * epsilon)  # type: ignore[attr-defined]
    return gradient


def _finite_difference_hessian(
    target: object, point: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    hessian = np.empty((point.size, point.size), dtype=np.float64)
    for index in range(point.size):
        plus = point.copy()
        minus = point.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        hessian[:, index] = (
            target.grad_log_prob(plus) - target.grad_log_prob(minus)  # type: ignore[attr-defined]
        ) / (2.0 * epsilon)
    return hessian


def test_gaussian_mixture_derivatives_match_finite_differences() -> None:
    target = GaussianMixtureTarget(
        np.array([0.4, 0.6]),
        np.array([[-2.0, 0.5], [1.0, -1.5]]),
        np.array([[[1.0, 0.2], [0.2, 0.5]], [[0.7, -0.1], [-0.1, 1.4]]]),
    )
    point = np.array([0.3, -0.2])
    np.testing.assert_allclose(
        target.grad_log_prob(point),
        _finite_difference_gradient(target, point),
        atol=2e-9,
    )
    np.testing.assert_allclose(
        target.hessian_log_prob(point),
        _finite_difference_hessian(target, point),
        atol=2e-7,
    )


def test_gaussian_mixture_exact_moments_match_direct_samples() -> None:
    target = GaussianMixtureTarget(
        np.array([0.25, 0.75]),
        np.array([[-2.0, 1.0], [1.0, -0.5]]),
        np.array([np.eye(2), np.diag([2.0, 0.5])]),
    )
    samples = target.sample(np.random.default_rng(2), 150_000)
    np.testing.assert_allclose(np.mean(samples, axis=0), target.mean_vector, atol=0.02)
    np.testing.assert_allclose(
        np.cov(samples, rowvar=False, ddof=0), target.covariance_matrix, atol=0.04
    )


def test_funnel_centered_noncentered_round_trip_and_derivatives() -> None:
    rotation = seeded_orthogonal_matrix(4, 12)
    target = FunnelTarget(
        4,
        sigma_v=1.5,
        scales=np.array([0.5, 1.0, 3.0]),
        rotation=rotation,
        location=np.array([1.0, -0.5, 0.2, 2.0]),
    )
    point = target.from_noncentered(np.array([0.4, -0.2, 1.1, 0.7]))
    np.testing.assert_allclose(
        target.from_noncentered(target.to_noncentered(point)), point, atol=1e-13
    )
    np.testing.assert_allclose(
        target.grad_log_prob(point),
        _finite_difference_gradient(target, point),
        atol=3e-8,
    )
    np.testing.assert_allclose(
        target.hessian_log_prob(point),
        _finite_difference_hessian(target, point),
        atol=3e-6,
    )


def test_funnel_direct_samples_match_exact_mean_and_covariance() -> None:
    target = FunnelTarget(
        4,
        sigma_v=1.0,
        scales=np.array([0.5, 1.0, 2.0]),
        rotation=seeded_orthogonal_matrix(4, 5),
    )
    samples = target.sample(np.random.default_rng(6), 200_000)
    np.testing.assert_allclose(np.mean(samples, axis=0), target.mean_vector, atol=0.02)
    np.testing.assert_allclose(
        np.cov(samples, rowvar=False, ddof=0),
        target.covariance_matrix,
        atol=0.06,
    )


def test_bimodal_funnel_derivatives_and_mode_labels() -> None:
    target = BimodalFunnelTarget(dimension=5, separation=8.0, sigma_v=1.2, anisotropy_ratio=4.0)
    point = np.array([0.3, -0.4, 0.5, 0.2, -0.1])
    np.testing.assert_allclose(
        target.grad_log_prob(point),
        _finite_difference_gradient(target, point),
        atol=5e-8,
    )
    np.testing.assert_allclose(
        target.hessian_log_prob(point),
        _finite_difference_hessian(target, point),
        atol=8e-6,
    )
    samples, labels = target.sample_with_labels(np.random.default_rng(8), 2000)
    predicted = target.mode_labels(samples)
    assert np.mean(predicted == labels) > 0.9
