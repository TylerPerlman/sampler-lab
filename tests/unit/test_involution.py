import numpy as np
import pytest

from sampler_lab.dynamics import (
    FunctionalInvolution,
    InvolutiveMetropolisKernel,
    MomentumFlipInvolution,
    involution_error,
    log_involutive_metropolis_ratio,
)


class PositiveExponentialTarget:
    def log_prob(self, state: np.ndarray) -> float:
        return float(-state[0]) if state[0] > 0.0 else float("-inf")


def test_momentum_flip_is_an_involution() -> None:
    flip = MomentumFlipInvolution()
    state = np.array([1.0, 2.0, 3.0, 4.0])

    np.testing.assert_allclose(flip.apply(state), [1.0, 2.0, -3.0, -4.0])
    assert involution_error(flip, state) == pytest.approx(0.0)
    assert flip.log_abs_det_jacobian(state) == pytest.approx(0.0)


def test_non_volume_preserving_involution_uses_jacobian() -> None:
    reciprocal = FunctionalInvolution(
        function=lambda state: np.array([1.0 / state[0]]),
        log_abs_det_jacobian_function=lambda state: -2.0 * np.log(state[0]),
    )
    state = np.array([0.1])

    assert involution_error(reciprocal, state) < 1e-12
    expected = -10.0 - (-0.1) - 2.0 * np.log(0.1)
    assert log_involutive_metropolis_ratio(
        current_log_density=-0.1,
        proposed_log_density=-10.0,
        log_abs_det_jacobian=-2.0 * np.log(0.1),
    ) == pytest.approx(expected)

    transition = InvolutiveMetropolisKernel(
        PositiveExponentialTarget(),
        reciprocal,
    ).step(state, np.random.default_rng(0))
    assert not transition.accepted
    np.testing.assert_allclose(transition.state, state)
    assert transition.log_acceptance_ratio == pytest.approx(expected)
