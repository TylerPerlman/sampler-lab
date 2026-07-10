from __future__ import annotations

import numpy as np
import pytest

from sampler_lab.rare_event_experiments import run_rare_event_experiment
from sampler_lab.rare_events import (
    GaussianHalfspaceRareEvent,
    GaussianShiftProposal,
    GaussianTwoSidedRareEvent,
    estimate_with_mixture,
    estimate_with_shift,
    symmetric_twist_mixture,
)

pytestmark = pytest.mark.statistical


def test_dominating_point_twist_resolves_event_crude_monte_carlo_barely_sees() -> None:
    problem = GaussianHalfspaceRareEvent([1.0], 1.0, [[1.0]])
    epsilon = 0.05
    n_samples = 100_000
    crude = estimate_with_shift(
        problem,
        GaussianShiftProposal([0.0], [[1.0]], epsilon),
        np.random.default_rng(1),
        n_samples,
    )
    twisted = estimate_with_shift(
        problem,
        GaussianShiftProposal(problem.dominant_point, [[1.0]], epsilon),
        np.random.default_rng(2),
        n_samples,
    )
    truth = problem.exact_probability(epsilon)
    assert crude.event_count <= 2
    assert twisted.event_count > 45_000
    assert abs(twisted.value - truth) < 4.0 * twisted.standard_error
    assert twisted.relative_standard_error < 0.01


def test_two_dominating_point_mixture_corrects_single_twist_typical_failure() -> None:
    problem = GaussianTwoSidedRareEvent([1.0], 1.0, [[1.0]])
    epsilon = 0.05
    n_samples = 100_000
    single = estimate_with_shift(
        problem,
        GaussianShiftProposal(problem.dominant_point, [[1.0]], epsilon),
        np.random.default_rng(2022),
        n_samples,
    )
    mixture = estimate_with_mixture(
        problem,
        symmetric_twist_mixture(problem, epsilon),
        np.random.default_rng(2022),
        n_samples,
    )
    truth = problem.exact_probability(epsilon)
    assert abs(single.value / truth - 0.5) < 0.03
    assert abs(mixture.value / truth - 1.0) < 0.03
    assert mixture.relative_standard_error < 0.01


def test_rare_event_experiment_smoke_and_expected_asymptotic_profile() -> None:
    rows = run_rare_event_experiment(
        epsilons=(0.25, 0.1),
        n_samples=20_000,
        dimension=3,
        seed=11,
    )
    assert len(rows) == 16
    smallest = [row for row in rows if row.epsilon == 0.1]
    one_twist = next(
        row
        for row in smallest
        if row.problem == "one-sided" and row.method == "dominant-point-twist"
    )
    two_single = next(
        row for row in smallest if row.problem == "two-sided" and row.method == "single-twist"
    )
    two_mixture = next(
        row
        for row in smallest
        if row.problem == "two-sided" and row.method == "two-dominating-point-mixture"
    )
    assert one_twist.absolute_relative_error < 0.06
    assert two_single.scaled_log_relative_variance > 4.0 * two_mixture.scaled_log_relative_variance
