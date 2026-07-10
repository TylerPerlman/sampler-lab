import numpy as np
import pytest

from sampler_lab.importance import gaussian_tail_experiment

pytestmark = pytest.mark.statistical


def test_shifted_proposal_improves_gaussian_tail_precision() -> None:
    crude, shifted = gaussian_tail_experiment(
        np.random.default_rng(2022),
        threshold=3.5,
        n_samples=100_000,
        proposal_means=[0.0, 3.5],
    )
    assert shifted.standard_error < crude.standard_error / 5.0
    assert shifted.relative_standard_error < 0.02
    assert abs(shifted.estimate - shifted.truth) < 5.0 * shifted.standard_error
    assert shifted.event_count > 40_000
