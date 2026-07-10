import numpy as np
import pytest

from sampler_lab.diagnostics import (
    diagnostics_from_normalized_weights,
    weight_diagnostics,
)
from sampler_lab.estimators import weighted_mean, weighted_variance
from sampler_lab.importance import (
    self_normalized_importance_estimate,
    standard_importance_estimate,
)


def test_weighted_moments_known_values() -> None:
    values = np.array([1.0, 3.0, 5.0])
    weights = np.array([0.25, 0.5, 0.25])
    assert weighted_mean(values, weights) == pytest.approx(3.0)
    assert weighted_variance(values, weights) == pytest.approx(2.0)
    assert weighted_variance(values, weights, unbiased=True) == pytest.approx(3.2)


def test_weight_diagnostics_uniform_and_degenerate() -> None:
    uniform = diagnostics_from_normalized_weights(np.full(4, 0.25))
    assert uniform.effective_sample_size == pytest.approx(4.0)
    assert uniform.ess_fraction == pytest.approx(1.0)
    assert uniform.normalized_entropy == pytest.approx(1.0)
    assert uniform.coefficient_of_variation_squared == pytest.approx(0.0)

    concentrated = weight_diagnostics([0.0, -np.inf, -np.inf, -np.inf])
    assert concentrated.effective_sample_size == pytest.approx(1.0)
    assert concentrated.max_normalized_weight == pytest.approx(1.0)
    assert concentrated.coefficient_of_variation_squared == pytest.approx(3.0)
    assert concentrated.n_positive == 1


def test_standard_importance_estimator_known_contributions() -> None:
    values = np.array([1.0, 2.0, 4.0])
    weights = np.array([0.5, 1.0, 1.5])
    result = standard_importance_estimate(values, np.log(weights))
    contributions = weights * values
    assert result.value == pytest.approx(float(np.mean(contributions)))
    assert result.standard_error == pytest.approx(
        float(np.std(contributions, ddof=1) / np.sqrt(3.0))
    )
    assert result.self_normalized is False


def test_self_normalized_estimator_is_log_shift_invariant() -> None:
    values = np.array([-2.0, 1.0, 7.0])
    log_weights = np.log([1.0, 2.0, 3.0])
    first = self_normalized_importance_estimate(values, log_weights)
    second = self_normalized_importance_estimate(values, log_weights + 1_000.0)
    expected = (-2.0 + 2.0 + 21.0) / 6.0
    assert first.value == pytest.approx(expected)
    assert second.value == pytest.approx(first.value)
    assert second.standard_error == pytest.approx(first.standard_error)
    assert second.effective_sample_size == pytest.approx(first.effective_sample_size)
    assert second.log_mean_weight == pytest.approx(first.log_mean_weight + 1_000.0)


def test_self_normalized_equal_weights_matches_sample_mean_error() -> None:
    values = np.array([1.0, 2.0, 4.0, 5.0])
    result = self_normalized_importance_estimate(values, np.zeros(values.size))
    assert result.value == pytest.approx(float(np.mean(values)))
    assert result.standard_error == pytest.approx(
        float(np.std(values, ddof=1) / np.sqrt(values.size))
    )


def test_self_normalized_error_is_undefined_with_one_positive_weight() -> None:
    result = self_normalized_importance_estimate(
        [1.0, 2.0, 3.0],
        [0.0, -np.inf, -np.inf],
    )
    assert result.value == pytest.approx(1.0)
    assert result.standard_error is not None
    assert result.delta_method_bias is not None
    assert np.isnan(result.standard_error)
    assert np.isnan(result.delta_method_bias)
