import numpy as np
import pytest

from sampler_lab.core.counters import OperationCounter
from sampler_lab.exact import generalized_inverse_discrete, inverse_cdf_sample


def test_inverse_cdf_reproducible_and_counted() -> None:
    counter = OperationCounter()
    sample = inverse_cdf_sample(
        np.random.default_rng(9),
        lambda u: -np.log1p(-u),
        8,
        counter=counter,
    )
    expected_uniforms = np.random.default_rng(9).random(8)
    np.testing.assert_allclose(sample, -np.log1p(-expected_uniforms))
    assert counter.uniform_draws == 8


def test_inverse_cdf_requires_shape_preservation() -> None:
    with pytest.raises(ValueError, match="preserve"):
        inverse_cdf_sample(np.random.default_rng(1), lambda u: np.array([0.0]), 3)


def test_generalized_inverse_discrete_support() -> None:
    sample = generalized_inverse_discrete(
        np.random.default_rng(7),
        values=[-1.0, 4.0, 9.0],
        probabilities=[1.0, 2.0, 1.0],
        size=1_000,
    )
    assert set(np.unique(sample)) == {-1.0, 4.0, 9.0}
