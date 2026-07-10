import numpy as np
import pytest

from sampler_lab.importance import (
    gaussian_product_log_weights,
    gaussian_weight_collapse_experiment,
)


def test_gaussian_product_log_weights_equal_scales_are_zero() -> None:
    samples = np.array([[1.0, -2.0], [0.5, 0.25]])
    np.testing.assert_allclose(gaussian_product_log_weights(samples), 0.0)


def test_gaussian_weight_collapse_exact_diagnostics() -> None:
    rows = gaussian_weight_collapse_experiment(
        np.random.default_rng(33),
        dimensions=[1, 5, 10],
        n_samples=50_000,
        proposal_scale=1.25,
    )
    assert [row.dimension for row in rows] == [1, 5, 10]
    assert rows[0].asymptotic_ess_fraction > rows[1].asymptotic_ess_fraction
    assert rows[1].asymptotic_ess_fraction > rows[2].asymptotic_ess_fraction
    for row in rows:
        assert row.ess_fraction == pytest.approx(row.asymptotic_ess_fraction, rel=0.08)
