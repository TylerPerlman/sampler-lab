from __future__ import annotations

import numpy as np

from sampler_lab.learning.optimizers import Adam
from sampler_lab.learning.stein import (
    IMQKernel,
    kernel_stein_discrepancy,
    run_svgd,
    stein_kernel_value,
    svgd_direction,
)
from sampler_lab.learning.variational import (
    DiagonalGaussianVariational,
    fit_forward_kl_diagonal_gaussian,
    fit_reverse_kl_diagonal_gaussian,
)
from sampler_lab.models.gaussian import GaussianTarget


def test_reverse_kl_gradient_matches_common_random_number_finite_difference() -> None:
    target = GaussianTarget(np.array([0.5, -1.0]), np.diag([2.0, 0.3]))
    family = DiagonalGaussianVariational(np.array([-0.2, 0.4]), np.log([0.8, 1.3]))
    noise = np.random.default_rng(3).normal(size=(5000, 2))
    analytic = family.reverse_kl_estimate(target, noise).gradient
    original = family.parameters
    numerical = np.empty_like(original)
    epsilon = 1e-5
    for index in range(original.size):
        plus = original.copy()
        minus = original.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        family.set_parameters(plus)
        plus_value = family.reverse_kl_estimate(target, noise).value
        family.set_parameters(minus)
        minus_value = family.reverse_kl_estimate(target, noise).value
        numerical[index] = (plus_value - minus_value) / (2.0 * epsilon)
    family.set_parameters(original)
    np.testing.assert_allclose(analytic, numerical, atol=5e-8)


def test_reverse_kl_fit_recovers_diagonal_gaussian() -> None:
    target = GaussianTarget(np.array([1.0, -2.0]), np.diag([0.5, 3.0]))
    family = DiagonalGaussianVariational(np.zeros(2), np.zeros(2))
    result = fit_reverse_kl_diagonal_gaussian(
        target,
        family,
        np.random.default_rng(7),
        n_steps=350,
        batch_size=256,
        optimizer=Adam(learning_rate=0.03),
    )
    np.testing.assert_allclose(result.approximation.mean, np.array([1.0, -2.0]), atol=0.12)
    np.testing.assert_allclose(
        result.approximation.scale,
        np.sqrt(np.array([0.5, 3.0])),
        rtol=0.12,
    )
    assert result.approximate


def test_forward_kl_fit_matches_reference_moments() -> None:
    samples = np.array([[0.0, 1.0], [2.0, 5.0], [4.0, 3.0]])
    fit = fit_forward_kl_diagonal_gaussian(samples)
    np.testing.assert_allclose(fit.mean, np.mean(samples, axis=0))
    np.testing.assert_allclose(fit.scale, np.std(samples, axis=0, ddof=0))


def test_imq_derivatives_match_finite_difference() -> None:
    kernel = IMQKernel(c=1.3, beta=-0.4)
    x = np.array([0.4, -0.7])
    y = np.array([-0.2, 0.1])
    analytic = kernel.gradient_first(x, y)
    numerical = np.empty_like(x)
    epsilon = 1e-6
    for index in range(x.size):
        plus = x.copy()
        minus = x.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        numerical[index] = (kernel.value(plus, y) - kernel.value(minus, y)) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, numerical, atol=1e-9)


def test_stein_kernel_is_symmetric() -> None:
    kernel = IMQKernel()
    x = np.array([0.3, -0.2])
    y = np.array([-0.4, 1.1])
    score_x = -x
    score_y = -y
    np.testing.assert_allclose(
        stein_kernel_value(x, y, score_x, score_y, kernel=kernel),
        stein_kernel_value(y, x, score_y, score_x, kernel=kernel),
        atol=1e-14,
    )


def test_svgd_repulsion_sums_to_zero_without_scores() -> None:
    class FlatTarget:
        def log_prob(self, x: np.ndarray) -> float:
            return 0.0

        def grad_log_prob(self, x: np.ndarray) -> np.ndarray:
            return np.zeros_like(x)

    particles = np.array([[-1.0], [0.0], [2.0]])
    direction = svgd_direction(particles, FlatTarget())
    np.testing.assert_allclose(np.sum(direction, axis=0), 0.0, atol=1e-14)


def test_svgd_reduces_v_statistic_ksd_on_shifted_gaussian_particles() -> None:
    target = GaussianTarget(np.zeros(1), np.eye(1))
    initial = np.linspace(2.0, 4.0, 20)[:, None]
    initial_ksd = kernel_stein_discrepancy(initial, target, unbiased=False)
    result = run_svgd(initial, target, n_steps=100, step_size=0.05)
    final_ksd = kernel_stein_discrepancy(result.particles, target, unbiased=False)
    assert final_ksd < initial_ksd
    assert result.approximate
