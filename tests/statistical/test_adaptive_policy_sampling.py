from __future__ import annotations

import numpy as np
import pytest

from sampler_lab.diagnostics.time_series import empirical_effective_sample_size
from sampler_lab.learning.adaptive_mh import (
    evaluate_adaptive_random_walk,
    train_adaptive_random_walk,
)
from sampler_lab.learning.objectives import AcceptanceObjective, AcceptedSquaredJumpObjective
from sampler_lab.learning.optimizers import Adam
from sampler_lab.learning.policies import LinearSoftmaxPolicy
from sampler_lab.learning.trainer import evaluate_frozen_policy, train_kernel_selection_policy
from sampler_lab.mcmc.proposals import GaussianRandomWalkProposal
from sampler_lab.models.gaussian import GaussianTarget

pytestmark = pytest.mark.statistical


def test_adaptive_random_walk_freezes_and_recovers_gaussian() -> None:
    target = GaussianTarget(
        np.array([1.0, -2.0]),
        np.array([[1.0, 0.8], [0.8, 2.0]]),
    )
    training = train_adaptive_random_walk(
        target,
        np.array([8.0, 8.0]),
        np.random.default_rng(101),
        n_warmup=3000,
        initial_scale=0.1,
        covariance_start=50,
        covariance_update_interval=25,
    )
    evaluation = evaluate_adaptive_random_walk(
        training,
        training.training.states[-1],
        np.random.default_rng(102),
        n_steps=20_000,
    )
    samples = evaluation.trajectory.samples(discard=1000)
    np.testing.assert_allclose(np.mean(samples, axis=0), target.mean_vector, atol=0.11)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), target.covariance_matrix, atol=0.14)
    assert evaluation.training_steps_excluded == 3000
    assert training.training.states.flags.writeable is False


def test_acceptance_only_policy_prefers_tiny_moves_but_jump_reward_does_not() -> None:
    target = GaussianTarget(np.zeros(1), np.eye(1))
    proposals = [GaussianRandomWalkProposal(0.03), GaussianRandomWalkProposal(1.5)]

    acceptance_result = train_kernel_selection_policy(
        target,
        proposals,
        np.zeros(1),
        np.random.default_rng(201),
        policy=LinearSoftmaxPolicy(np.zeros((2, 1)), np.array([0.03, 1.5])),
        objective=AcceptanceObjective(use_probability=True),
        n_updates=160,
        rollout_length=12,
        optimizer=Adam(learning_rate=0.04),
    )
    jump_result = train_kernel_selection_policy(
        target,
        proposals,
        np.zeros(1),
        np.random.default_rng(202),
        policy=LinearSoftmaxPolicy(np.zeros((2, 1)), np.array([0.03, 1.5])),
        objective=AcceptedSquaredJumpObjective(),
        n_updates=160,
        rollout_length=12,
        optimizer=Adam(learning_rate=0.04),
    )
    assert acceptance_result.action_probabilities[0] > 0.75
    assert jump_result.action_probabilities[1] > 0.75

    acceptance_eval = evaluate_frozen_policy(
        acceptance_result,
        np.zeros(1),
        np.random.default_rng(203),
        n_steps=8000,
    ).trajectory.states[:, 0]
    jump_eval = evaluate_frozen_policy(
        jump_result,
        np.zeros(1),
        np.random.default_rng(204),
        n_steps=8000,
    ).trajectory.states[:, 0]
    assert empirical_effective_sample_size(jump_eval) > 3.0 * empirical_effective_sample_size(
        acceptance_eval
    )
