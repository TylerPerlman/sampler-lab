import numpy as np
import pytest

from sampler_lab.dynamics import (
    gaussian_quadratic_expectation,
    gaussian_ula_analysis,
    linear_gaussian_iat,
    solve_discrete_lyapunov,
)
from sampler_lab.models import GaussianTarget


def test_scalar_gaussian_ula_stability_and_stationary_bias() -> None:
    target = GaussianTarget([0.0], [[2.0]])
    analysis = gaussian_ula_analysis(target, 0.5)

    assert analysis.maximum_stable_step_size == pytest.approx(4.0)
    assert analysis.spectral_radius == pytest.approx(0.75)
    assert analysis.stable
    assert analysis.stationary_covariance is not None
    assert analysis.stationary_covariance[0, 0] == pytest.approx(2.0 / (1.0 - 0.125))
    assert analysis.covariance_bias is not None
    assert analysis.covariance_bias[0, 0] > 0.0
    assert analysis.kl_stationary_to_target is not None
    assert analysis.kl_stationary_to_target > 0.0


def test_unstable_gaussian_ula_has_no_stationary_covariance() -> None:
    target = GaussianTarget([0.0], [[2.0]])
    analysis = gaussian_ula_analysis(target, 4.0)

    assert not analysis.stable
    assert analysis.spectral_radius == pytest.approx(1.0)
    assert analysis.stationary_covariance is None
    assert analysis.kl_stationary_to_target is None


def test_covariance_preconditioning_removes_condition_number_from_stability() -> None:
    covariance = np.diag([1.0, 0.01])
    target = GaussianTarget([0.0, 0.0], covariance)
    analysis = gaussian_ula_analysis(target, 0.5, preconditioner=covariance)

    assert analysis.maximum_stable_step_size == pytest.approx(2.0)
    np.testing.assert_allclose(analysis.transition_matrix, 0.5 * np.eye(2), atol=1e-12)
    assert analysis.stationary_covariance is not None
    np.testing.assert_allclose(
        analysis.stationary_covariance,
        covariance / (1.0 - 0.25),
        atol=1e-12,
    )


def test_discrete_lyapunov_solver_satisfies_fixed_point() -> None:
    transition = np.array([[0.8, 0.1], [0.0, 0.5]])
    noise = np.array([[0.4, 0.0], [0.0, 0.2]])
    covariance = solve_discrete_lyapunov(transition, noise)

    np.testing.assert_allclose(
        covariance,
        transition @ covariance @ transition.T + noise,
        atol=1e-12,
    )


def test_linear_gaussian_iat_matches_scalar_ar1_formula() -> None:
    coefficient = 0.8
    iat = linear_gaussian_iat(
        np.array([[coefficient]]),
        np.array([[2.0]]),
        np.array([1.0]),
    )

    assert iat == pytest.approx((1.0 + coefficient) / (1.0 - coefficient))


def test_gaussian_quadratic_expectation() -> None:
    value = gaussian_quadratic_expectation(
        mean=[1.0, -1.0],
        covariance=[[2.0, 0.0], [0.0, 3.0]],
        quadratic=[[2.0, 0.0], [0.0, 1.0]],
        linear=[1.0, 2.0],
        constant=4.0,
    )

    assert value == pytest.approx(13.0)
