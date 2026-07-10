import numpy as np
import pytest

from sampler_lab.models import GaussianTarget


def test_gaussian_target_derivatives() -> None:
    target = GaussianTarget(mean=[1.0, -1.0], covariance=[[2.0, 0.0], [0.0, 0.5]])
    point = np.array([3.0, 0.0])
    np.testing.assert_allclose(target.grad_log_prob(point), [-1.0, -2.0])
    np.testing.assert_allclose(target.hessian_log_prob(point), [[-0.5, 0.0], [0.0, -2.0]])
    assert np.isfinite(target.log_prob(point))


def test_gaussian_target_rejects_bad_covariance() -> None:
    with pytest.raises(np.linalg.LinAlgError):
        GaussianTarget(mean=[0.0, 0.0], covariance=[[1.0, 2.0], [2.0, 1.0]])
