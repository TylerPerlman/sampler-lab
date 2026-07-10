import numpy as np
import pytest

from sampler_lab.models import (
    XYModel,
    modified_bessel_i0_i1,
    periodic_angle_difference,
    von_mises_mean_cosine,
    wrap_angles,
)


def finite_difference_gradient(
    model: XYModel, state: np.ndarray, epsilon: float = 1e-6
) -> np.ndarray:
    gradient = np.empty_like(state)
    for index in range(state.size):
        direction = np.zeros_like(state)
        direction[index] = epsilon
        gradient[index] = (
            model.log_prob(state + direction) - model.log_prob(state - direction)
        ) / (2.0 * epsilon)
    return gradient


def test_angle_wrapping_and_periodic_difference() -> None:
    angles = np.array([-4.0 * np.pi, -np.pi, 0.0, np.pi, 5.0 * np.pi / 2.0])
    wrapped = wrap_angles(angles)

    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)
    np.testing.assert_allclose(wrapped, [0.0, -np.pi, 0.0, -np.pi, np.pi / 2.0])
    np.testing.assert_allclose(
        periodic_angle_difference([np.pi - 0.1], [-np.pi + 0.1]),
        [-0.2],
    )


def test_xy_gradient_matches_finite_differences() -> None:
    model = XYModel(size=3, inverse_temperature=0.8, coupling=1.2, external_field=0.4)
    state = np.linspace(-1.1, 1.3, model.dimension)

    np.testing.assert_allclose(
        model.grad_log_prob(state),
        finite_difference_gradient(model, state),
        atol=2e-8,
    )


def test_zero_field_xy_model_is_globally_rotation_invariant() -> None:
    model = XYModel(size=4, inverse_temperature=1.1, external_field=0.0)
    state = np.random.default_rng(2).uniform(-np.pi, np.pi, size=model.dimension)
    shifted = state + 0.73

    assert model.log_prob(shifted) == pytest.approx(model.log_prob(state), abs=1e-12)
    np.testing.assert_allclose(model.grad_log_prob(shifted), model.grad_log_prob(state), atol=1e-12)


def test_bessel_series_and_single_site_exact_response() -> None:
    i0, i1 = modified_bessel_i0_i1(1.5)
    assert i0 == pytest.approx(float(np.i0(1.5)), rel=1e-14)
    assert i1 / i0 == pytest.approx(von_mises_mean_cosine(1.5), rel=1e-14)

    model = XYModel(
        size=1,
        inverse_temperature=2.0,
        coupling=3.0,
        external_field=0.75,
    )
    assert model.exact_single_site_mean_cosine() == pytest.approx(i1 / i0)


def test_xy_magnetization_observables() -> None:
    model = XYModel(size=2, inverse_temperature=1.0)
    aligned = np.zeros(model.dimension)
    alternating = np.array([0.0, np.pi, 0.0, np.pi])

    np.testing.assert_allclose(model.magnetization_vector(aligned), [1.0, 0.0])
    assert model.absolute_magnetization(aligned) == pytest.approx(1.0)
    assert model.absolute_magnetization(alternating) == pytest.approx(0.0, abs=1e-12)
