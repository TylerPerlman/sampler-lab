import numpy as np
import pytest

from sampler_lab.estimators import estimate_normalization_ratio
from sampler_lab.importance import (
    chi_squared_divergence_estimate,
    gaussian_scale_chi_squared,
    product_chi_squared,
    renyi_divergence_order_two_estimate,
)


def test_normalization_ratio_known_weights() -> None:
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    result = estimate_normalization_ratio(np.log(weights))
    assert result.value == pytest.approx(2.5)
    assert result.log_value == pytest.approx(np.log(2.5))
    expected_relative_se = np.std(weights, ddof=1) / np.sqrt(4.0) / np.mean(weights)
    assert result.relative_standard_error == pytest.approx(expected_relative_se)
    assert result.standard_error == pytest.approx(2.5 * expected_relative_se)


def test_normalization_ratio_retains_log_when_linear_value_overflows() -> None:
    result = estimate_normalization_ratio([1_000.0, 1_000.0])
    assert result.log_value == pytest.approx(1_000.0)
    assert np.isinf(result.value)
    assert result.relative_standard_error == pytest.approx(0.0)


def test_scale_invariant_chi_squared_estimate() -> None:
    log_weights = np.log([1.0, 2.0, 3.0])
    first = chi_squared_divergence_estimate(log_weights)
    second = chi_squared_divergence_estimate(log_weights + 900.0)
    expected = 3.0 * (1.0 + 4.0 + 9.0) / 36.0 - 1.0
    assert first == pytest.approx(expected)
    assert second == pytest.approx(first)
    assert renyi_divergence_order_two_estimate(log_weights) == pytest.approx(np.log1p(expected))


def test_product_and_gaussian_chi_squared_formulas() -> None:
    assert product_chi_squared(0.25, 3) == pytest.approx(1.25**3 - 1.0)
    assert gaussian_scale_chi_squared(5) == pytest.approx(0.0)
    one_dimensional = gaussian_scale_chi_squared(1, proposal_scale=1.25)
    five_dimensional = gaussian_scale_chi_squared(5, proposal_scale=1.25)
    assert five_dimensional == pytest.approx((1.0 + one_dimensional) ** 5 - 1.0)
    assert np.isinf(gaussian_scale_chi_squared(1, proposal_scale=0.5))
