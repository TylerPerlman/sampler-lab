import numpy as np
import pytest

from sampler_lab.core.protocols import Resampler
from sampler_lab.particles import (
    BernoulliResampler,
    MultinomialResampler,
    SystematicResampler,
    minimal_conditional_variances,
    multinomial_conditional_variances,
    offspring_counts,
    resampling_diagnostics,
)


@pytest.mark.parametrize("resampler", [SystematicResampler(), BernoulliResampler()])
def test_low_variance_offspring_are_floor_or_ceiling(resampler: Resampler) -> None:
    weights = np.array([0.05, 0.15, 0.3, 0.5])
    target = 10
    indices = resampler.resample(weights, np.random.default_rng(17), target)
    counts = offspring_counts(indices, len(weights))
    expected = target * weights

    assert np.all(counts >= np.floor(expected))
    assert np.all(counts <= np.ceil(expected))
    if isinstance(resampler, SystematicResampler):
        assert indices.size == target


def test_multinomial_has_fixed_population_and_valid_indices() -> None:
    indices = MultinomialResampler().resample(
        np.array([0.1, 0.2, 0.7]),
        np.random.default_rng(5),
        50,
    )
    assert indices.shape == (50,)
    assert np.all((indices >= 0) & (indices < 3))


def test_conditional_variance_formulas() -> None:
    weights = np.array([0.1, 0.2, 0.7])
    multinomial = multinomial_conditional_variances(weights, 10)
    minimum = minimal_conditional_variances(weights, 10)

    assert multinomial == pytest.approx(10 * weights * (1.0 - weights))
    assert minimum == pytest.approx([0.0, 0.0, 0.0])

    noninteger = minimal_conditional_variances([0.15, 0.85], 3)
    assert noninteger == pytest.approx([0.45 * 0.55, 0.55 * 0.45])


def test_resampling_schemes_are_empirically_unbiased() -> None:
    weights = np.array([0.08, 0.27, 0.65])
    target = 7
    repeats = 8_000

    resamplers: tuple[Resampler, ...] = (
        MultinomialResampler(),
        SystematicResampler(),
        BernoulliResampler(),
    )
    for seed, resampler in enumerate(resamplers, start=1):
        rng = np.random.default_rng(seed)
        accumulated = np.zeros(weights.size)
        for _ in range(repeats):
            indices = resampler.resample(weights, rng, target)
            accumulated += offspring_counts(indices, weights.size)
        empirical = accumulated / repeats
        assert empirical == pytest.approx(target * weights, abs=0.025)


def test_resampling_diagnostics_count_coalescence() -> None:
    summary = resampling_diagnostics([0, 0, 2, 2, 2], n_parents=4)
    assert summary.n_offspring == 5
    assert summary.n_unique_parents == 2
    assert summary.unique_parent_fraction == pytest.approx(0.5)
    assert summary.max_offspring == 3
