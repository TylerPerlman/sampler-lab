from dataclasses import dataclass

import numpy as np
import pytest

from sampler_lab.particles import (
    ParticleExtinctionError,
    PropagationResult,
    SystematicResampler,
    sequential_importance_sampling,
)


@dataclass(frozen=True)
class TwoParticleProposal:
    def propose(
        self,
        particles: np.ndarray,
        step: int,
        rng: np.random.Generator,
    ) -> PropagationResult:
        del rng
        factors = np.array([1.0, 3.0]) if step == 1 else np.array([2.0, 4.0])
        return PropagationResult(particles + 1.0, np.log(factors))


@dataclass(frozen=True)
class CollapseProposal:
    def propose(
        self,
        particles: np.ndarray,
        step: int,
        rng: np.random.Generator,
    ) -> PropagationResult:
        del step, rng
        increments = np.full(particles.shape[0], -np.inf)
        increments[0] = 0.0
        return PropagationResult(particles, increments)


@dataclass(frozen=True)
class ExtinctionProposal:
    def propose(
        self,
        particles: np.ndarray,
        step: int,
        rng: np.random.Generator,
    ) -> PropagationResult:
        del step, rng
        return PropagationResult(particles, np.full(particles.shape[0], -np.inf))


def test_sis_normalizing_constant_telescopes_without_resampling() -> None:
    result = sequential_importance_sampling(
        np.array([[0.0], [1.0]]),
        2,
        TwoParticleProposal(),
        np.random.default_rng(1),
    )

    # Full trajectory weights are 1*2 and 3*4, whose initial-sample mean is 7.
    assert result.normalizing_constant_estimate == pytest.approx(7.0)
    assert result.incremental_log_normalizers == pytest.approx([np.log(2.0), np.log(3.5)])
    assert result.final_weighted_cloud.weights == pytest.approx([1.0 / 7.0, 6.0 / 7.0])
    assert result.resampled.tolist() == [False, False]
    assert result.ancestry.final_to_initial() == pytest.approx([0, 1])


def test_ess_triggered_resampling_preserves_pre_resampling_diagnostics() -> None:
    result = sequential_importance_sampling(
        np.arange(4.0)[:, None],
        1,
        CollapseProposal(),
        np.random.default_rng(4),
        resampler=SystematicResampler(),
        resample_ess_fraction=0.5,
    )

    assert result.resampled.tolist() == [True]
    assert result.weighted_clouds[0].effective_sample_size == pytest.approx(1.0)
    assert result.final_cloud.weights == pytest.approx(np.full(4, 0.25))
    assert result.final_cloud.particles[:, 0] == pytest.approx(np.zeros(4))
    assert result.ancestry.unique_ancestor_counts() == pytest.approx([1, 4])


def test_sequential_importance_reports_total_extinction() -> None:
    with pytest.raises(ParticleExtinctionError, match="step 1"):
        sequential_importance_sampling(
            np.zeros((3, 1)),
            1,
            ExtinctionProposal(),
            np.random.default_rng(8),
        )
