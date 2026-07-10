import numpy as np
import pytest

from sampler_lab.models import RosenbrockTarget


def test_rosenbrock_mode_gradient_and_exact_moments() -> None:
    target = RosenbrockTarget()

    assert target.log_prob(target.mode) == pytest.approx(0.0)
    np.testing.assert_allclose(target.grad_log_prob(target.mode), np.zeros(2))
    np.testing.assert_allclose(target.exact_mean(), [1.0, 11.0])
    np.testing.assert_allclose(target.exact_covariance(), [[10.0, 20.0], [20.0, 240.1]])


def test_rosenbrock_exact_sampler_respects_hierarchical_residual() -> None:
    target = RosenbrockTarget()
    samples = target.sample_exact(np.random.default_rng(2022), 50_000)
    residual = samples[:, 1] - samples[:, 0] ** 2

    assert float(np.mean(samples[:, 0])) == pytest.approx(target.location, abs=0.04)
    assert float(np.var(samples[:, 0])) == pytest.approx(target.x_variance, abs=0.15)
    assert float(np.mean(residual)) == pytest.approx(0.0, abs=0.004)
    assert float(np.var(residual)) == pytest.approx(target.conditional_y_variance, abs=0.003)
