import numpy as np
import pytest

from sampler_lab.ensemble import (
    EnsembleState,
    ensemble_effective_sample_size,
    mean_cross_walker_correlation,
)
from sampler_lab.models import GaussianTarget


def test_ensemble_state_caches_product_target_and_affine_rank() -> None:
    walkers = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    state = EnsembleState.from_target(walkers, GaussianTarget([0.0, 0.0], np.eye(2)))

    assert state.n_walkers == 4
    assert state.dimension == 2
    assert state.affine_span_rank == 2
    assert state.has_full_affine_span
    with pytest.raises(ValueError):
        state.walkers[0, 0] = 3.0


def test_degenerate_ensemble_reports_missing_affine_direction() -> None:
    state = EnsembleState(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        [0.0, 0.0, 0.0],
    )

    assert state.affine_span_rank == 1
    assert not state.has_full_affine_span


def test_ensemble_ess_uses_ensemble_average_time_series() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(4000, 8))

    result = ensemble_effective_sample_size(values)

    assert result.integrated_autocorrelation_time == pytest.approx(1.0, abs=0.12)
    assert result.effective_sample_size == pytest.approx(32_000, rel=0.12)
    assert mean_cross_walker_correlation(values) == pytest.approx(0.0, abs=0.02)
