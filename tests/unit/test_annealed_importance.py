import numpy as np
import pytest

from sampler_lab.annealing import (
    AnnealingSchedule,
    FunctionalAnnealingPath,
    IdentityPopulationTransition,
    annealed_importance_sampling,
    jarzynski_estimate,
    reweight_cloud,
)
from sampler_lab.particles import ParticleCloud, SystematicResampler


def test_constant_work_path_gives_exact_ratio_and_free_energy() -> None:
    shift = 2.3
    path = FunctionalAnnealingPath(lambda x, beta: beta * shift)
    result = annealed_importance_sampling(
        np.arange(6.0)[:, None],
        path,
        AnnealingSchedule.linear(5),
        IdentityPopulationTransition(),
        np.random.default_rng(7),
    )

    assert result.log_normalizing_constant_ratio == pytest.approx(shift)
    assert result.normalizing_constant_ratio == pytest.approx(np.exp(shift))
    assert result.delta_free_energy == pytest.approx(-shift)
    assert result.incremental_log_normalizers == pytest.approx(np.full(5, shift / 5.0))
    assert result.cumulative_trajectory_log_weights[-1] == pytest.approx(np.full(6, shift))
    assert result.ess_history == pytest.approx(np.full(5, 6.0))
    assert result.iid_jarzynski_estimate().ratio.log_value == pytest.approx(shift)
    assert result.final_reduced_work == pytest.approx(np.full(6, -shift))


def test_resampled_constant_work_path_preserves_telescoping_ratio() -> None:
    shift = -1.7
    result = annealed_importance_sampling(
        np.arange(8.0)[:, None],
        FunctionalAnnealingPath(lambda x, beta: beta * shift),
        AnnealingSchedule.power(4, 2.0),
        IdentityPopulationTransition(),
        np.random.default_rng(11),
        resampler=SystematicResampler(),
        resample_every_step=True,
    )

    assert result.log_normalizing_constant_ratio == pytest.approx(shift)
    assert result.resampled.tolist() == [True, True, True, True]
    assert result.final_cloud.weights == pytest.approx(np.full(8, 1.0 / 8.0))
    assert result.ancestry.population_sizes == (8, 8, 8, 8, 8)
    with pytest.raises(ValueError, match="invalid after resampling"):
        result.iid_jarzynski_estimate()


def test_two_state_one_step_ratio_and_final_weights_are_exact() -> None:
    path = FunctionalAnnealingPath(lambda x, beta: beta * (0.0 if int(x[0]) == 0 else np.log(3.0)))
    result = annealed_importance_sampling(
        np.array([[0.0], [1.0]]),
        path,
        AnnealingSchedule.linear(1),
        IdentityPopulationTransition(),
        np.random.default_rng(4),
    )

    assert result.normalizing_constant_ratio == pytest.approx(2.0)
    assert result.final_weighted_cloud.weights == pytest.approx([0.25, 0.75])
    assert result.final_cloud.expectation(lambda x: x[:, 0]) == pytest.approx(0.75)


def test_intermediate_cloud_reweighting_targets_later_path_law() -> None:
    cloud = ParticleCloud.uniform(np.array([[0.0], [1.0]]))
    path = FunctionalAnnealingPath(lambda x, beta: beta * (0.0 if int(x[0]) == 0 else np.log(3.0)))

    target = reweight_cloud(cloud, path, 0.0, 1.0)
    assert target.weights == pytest.approx([0.25, 0.75])


def test_jarzynski_estimator_uses_exp_minus_work_convention() -> None:
    estimate = jarzynski_estimate(np.array([0.0, np.log(2.0)]))

    assert estimate.ratio.value == pytest.approx(0.75)
    assert estimate.delta_free_energy == pytest.approx(-np.log(0.75))
    assert estimate.standard_error == pytest.approx(1.0 / 3.0)
    assert estimate.delta_method_bias == pytest.approx(1.0 / 18.0)


def test_jarzynski_log_estimate_remains_finite_when_ratio_overflows() -> None:
    estimate = jarzynski_estimate(np.array([-1000.0, -1001.0]))

    assert np.isfinite(estimate.ratio.log_value)
    assert estimate.ratio.value == float("inf")
    assert estimate.delta_free_energy == pytest.approx(-estimate.ratio.log_value)
