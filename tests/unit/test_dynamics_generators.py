from dataclasses import dataclass

import numpy as np
import pytest

from sampler_lab.core.results import Transition
from sampler_lab.dynamics import (
    EulerMaruyamaKernel,
    diffusion_generator_value,
    estimate_discrete_generator,
    estimate_local_moments,
)


@dataclass(frozen=True)
class DeterministicShiftKernel:
    shift: np.ndarray

    def step(self, state: np.ndarray, rng: np.random.Generator) -> Transition:
        return Transition(state=state + self.shift)


def test_discrete_generator_for_deterministic_shift() -> None:
    kernel = DeterministicShiftKernel(np.array([0.2, -0.1]))
    estimate = estimate_discrete_generator(
        kernel,
        lambda x: float(x @ x),
        np.array([1.0, 2.0]),
        step_size=0.5,
        n_replications=4,
        rng=np.random.default_rng(1),
    )

    expected = ((np.array([1.2, 1.9]) @ np.array([1.2, 1.9])) - 5.0) / 0.5
    assert estimate.value == pytest.approx(expected)
    assert estimate.standard_error == pytest.approx(0.0)
    assert estimate.mean_next_value == pytest.approx(5.05)


def test_diffusion_generator_value_for_quadratic_function() -> None:
    value = diffusion_generator_value(
        drift=lambda x: -x,
        covariance=lambda x: 2.0 * np.eye(x.size),
        gradient=lambda x: 2.0 * x,
        hessian=lambda x: 2.0 * np.eye(x.size),
        x=np.array([1.0, 2.0]),
    )

    assert value == pytest.approx(-6.0)


def test_euler_maruyama_local_moments_match_coefficients() -> None:
    step_size = 0.01
    kernel = EulerMaruyamaKernel(
        drift=lambda x: np.array([-2.0 * x[0]]),
        diffusion_factor=lambda x: np.array([[3.0]]),
        step_size=step_size,
    )
    estimate = estimate_local_moments(
        kernel,
        np.array([1.5]),
        step_size,
        n_replications=40_000,
        rng=np.random.default_rng(2022),
    )

    assert estimate.drift[0] == pytest.approx(-3.0, abs=0.16)
    assert estimate.second_moment_rate[0, 0] == pytest.approx(9.09, abs=0.18)
    assert estimate.centered_covariance_rate[0, 0] == pytest.approx(9.0, abs=0.18)


def test_generator_validation_rejects_scalar_state() -> None:
    kernel = DeterministicShiftKernel(np.array([1.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        estimate_discrete_generator(
            kernel,
            lambda x: float(x[0]),
            np.array(0.0),
            0.1,
            10,
            np.random.default_rng(1),
        )
