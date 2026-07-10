import numpy as np
import pytest

from sampler_lab.estimators import error_metrics, iid_estimate


def test_iid_estimate_known_values() -> None:
    result = iid_estimate([1.0, 2.0, 3.0, 4.0])
    assert result.value == 2.5
    assert result.sample_variance == pytest.approx(5.0 / 3.0)
    assert result.standard_error == pytest.approx(np.sqrt((5.0 / 3.0) / 4.0))


def test_iid_estimate_vector_observable() -> None:
    samples = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = iid_estimate(samples, lambda x: np.sum(x, axis=1))
    assert result.value == 5.0


def test_error_metrics_decomposition() -> None:
    result = error_metrics([1.0, 2.0, 3.0], truth=1.5)
    assert result.bias == pytest.approx(0.5)
    assert result.mse == pytest.approx(result.variance + result.bias**2)
