import numpy as np
import pytest

from sampler_lab.diagnostics import (
    empirical_autocorrelations,
    empirical_effective_sample_size,
    empirical_integrated_autocorrelation_time,
)


def test_iid_noise_has_iat_near_one() -> None:
    values = np.random.default_rng(10).normal(size=30_000)
    iat = empirical_integrated_autocorrelation_time(values)

    assert iat == pytest.approx(1.0, abs=0.08)
    assert empirical_effective_sample_size(values) == pytest.approx(values.size / iat)


def test_ar1_iat_matches_closed_form() -> None:
    rng = np.random.default_rng(11)
    coefficient = 0.8
    values = np.empty(80_000)
    values[0] = rng.normal()
    noise_scale = np.sqrt(1.0 - coefficient**2)
    for index in range(1, values.size):
        values[index] = coefficient * values[index - 1] + noise_scale * rng.normal()

    estimated = empirical_integrated_autocorrelation_time(values)
    exact = (1.0 + coefficient) / (1.0 - coefficient)
    correlations = empirical_autocorrelations(values, max_lag=3)

    assert correlations[1] == pytest.approx(coefficient, abs=0.015)
    assert estimated == pytest.approx(exact, rel=0.12)


def test_constant_sequence_has_undefined_autocorrelations() -> None:
    with pytest.raises(ValueError):
        empirical_autocorrelations(np.ones(20))
