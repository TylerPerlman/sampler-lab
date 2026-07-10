from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from sampler_lab.adaptive import (
    DualAveragingStepSize,
    RobbinsMonroLogScale,
    RobbinsMonroSchedule,
    RunningMoments,
    diminishing_ratio,
    expanding_warmup_windows,
    is_nonincreasing,
    regularize_covariance,
)


def test_running_moments_matches_numpy_covariance() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=(200, 4))
    moments = RunningMoments(4)
    moments.update_batch(values)

    np.testing.assert_allclose(moments.mean, np.mean(values, axis=0), atol=1e-14)
    np.testing.assert_allclose(moments.covariance(), np.cov(values, rowvar=False), atol=1e-13)


def test_covariance_regularization_enforces_spectral_bounds() -> None:
    covariance = np.array([[4.0, 3.99], [3.99, 4.0]])
    result = regularize_covariance(
        covariance,
        shrinkage=0.1,
        target="isotropic",
        eigenvalue_floor=0.5,
        eigenvalue_ceiling=5.0,
    )
    eigenvalues = np.linalg.eigvalsh(result.matrix)
    assert np.min(eigenvalues) >= 0.5 - 1e-12
    assert np.max(eigenvalues) <= 5.0 + 1e-12
    assert result.correction_frobenius_norm > 0.0


def test_robbins_monro_schedule_is_diminishing() -> None:
    schedule = RobbinsMonroSchedule(initial_rate=1.0, exponent=0.6, offset=0.0)
    rates = schedule.rates(100)
    assert is_nonincreasing(rates)
    assert diminishing_ratio(rates) < 0.1
    assert np.sum(rates) > 1.0
    assert np.sum(rates * rates) < 5.0


def test_robbins_monro_log_scale_moves_in_expected_direction() -> None:
    adapter = RobbinsMonroLogScale(
        initial_scale=1.0,
        target_acceptance=0.5,
        schedule=RobbinsMonroSchedule(initial_rate=0.2, exponent=0.6, offset=0.0),
    )
    increased = adapter.update(1.0)
    assert increased > 1.0
    decreased = adapter.update(0.0)
    assert decreased < increased


def test_dual_averaging_first_update_matches_recursion() -> None:
    adapter = DualAveragingStepSize(
        initial_step_size=0.1,
        target_acceptance=0.8,
        gamma=0.05,
        t0=10.0,
        kappa=0.75,
    )
    observed = adapter.update(0.5)
    h_bar = (0.8 - 0.5) / 11.0
    expected = np.exp(np.log(1.0) - h_bar / 0.05)
    assert observed == pytest.approx(expected)
    assert adapter.averaged_step_size == pytest.approx(expected)


def test_warmup_windows_cover_each_step_once() -> None:
    windows = expanding_warmup_windows(200, initial_buffer=20, terminal_buffer=20, base_window=10)
    assert windows[0].start == 0
    assert windows[-1].stop == 200
    for first, second in pairwise(windows):
        assert first.stop == second.start
    assert sum(window.length for window in windows) == 200
    assert any(window.adapt_covariance for window in windows)


def test_frozen_policy_json_round_trip() -> None:
    from sampler_lab.adaptive import FrozenPolicy

    policy = FrozenPolicy(
        "test-policy",
        np.array([1.0, -2.5, 3.25]),
        {"warmup_steps": 100.0, "target_acceptance": 0.234},
    )
    restored = FrozenPolicy.from_json(policy.to_json())
    assert restored.name == policy.name
    np.testing.assert_array_equal(restored.parameters, policy.parameters)
    assert restored.metadata == policy.metadata
    assert restored.parameters.flags.writeable is False
