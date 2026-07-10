from __future__ import annotations

import numpy as np
import pytest

from sampler_lab.learning.baselines import LinearBaseline, RunningMeanBaseline
from sampler_lab.learning.gradients import (
    categorical_kl,
    linear_softmax_fisher,
    natural_gradient_direction,
    reinforce_gradient,
)
from sampler_lab.learning.objectives import (
    AcceptanceObjective,
    AcceptedSquaredJumpObjective,
    GeneralizedSpeedObjective,
    PolicyTransition,
)
from sampler_lab.learning.optimizers import Adam
from sampler_lab.learning.policies import LinearSoftmaxPolicy, SquashedGaussianPolicy
from sampler_lab.learning.trainer import train_kernel_selection_policy
from sampler_lab.mcmc.proposals import GaussianRandomWalkProposal
from sampler_lab.models.gaussian import GaussianTarget


def test_softmax_scores_have_zero_policy_expectation() -> None:
    policy = LinearSoftmaxPolicy(
        np.array([[0.2, -0.1], [0.3, 0.4], [-0.5, 0.2]]),
        action_values=np.array([0.1, 1.0, 3.0]),
    )
    features = np.array([1.0, -0.7])
    probabilities = policy.probabilities(features)
    expected_score = sum(
        probability * policy.score(features, action)
        for action, probability in enumerate(probabilities)
    )
    np.testing.assert_allclose(expected_score, 0.0, atol=1e-14)


def test_softmax_score_matches_finite_difference() -> None:
    policy = LinearSoftmaxPolicy(np.array([[0.2, -0.4], [0.1, 0.3]]))
    features = np.array([0.7, -1.2])
    action = 1
    analytic = policy.score(features, action)
    original = policy.parameters
    numerical = np.empty_like(original)
    epsilon = 1e-6
    for index in range(original.size):
        plus = original.copy()
        minus = original.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        policy.set_parameters(plus)
        plus_value = np.log(policy.probabilities(features)[action])
        policy.set_parameters(minus)
        minus_value = np.log(policy.probabilities(features)[action])
        numerical[index] = (plus_value - minus_value) / (2.0 * epsilon)
    policy.set_parameters(original)
    np.testing.assert_allclose(analytic, numerical, atol=1e-8)


def test_squashed_gaussian_score_matches_finite_difference() -> None:
    policy = SquashedGaussianPolicy(
        np.array([[0.2, -0.1], [-0.3, 0.4]]),
        np.log(np.array([0.7, 1.2])),
        lower=np.array([-2.0, 0.1]),
        upper=np.array([2.0, 3.0]),
    )
    features = np.array([1.0, 0.5])
    action = policy.act(np.random.default_rng(4), features).value
    _, analytic = policy.log_prob_and_score(action, features)
    original = policy.parameters
    numerical = np.empty_like(original)
    epsilon = 1e-6
    for index in range(original.size):
        plus = original.copy()
        minus = original.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        policy.set_parameters(plus)
        plus_value, _ = policy.log_prob_and_score(action, features)
        policy.set_parameters(minus)
        minus_value, _ = policy.log_prob_and_score(action, features)
        numerical[index] = (plus_value - minus_value) / (2.0 * epsilon)
    policy.set_parameters(original)
    np.testing.assert_allclose(analytic, numerical, atol=2e-7)


def test_linear_baseline_satisfies_normal_equations() -> None:
    rng = np.random.default_rng(8)
    features = rng.normal(size=(50, 3))
    returns = 1.5 + features @ np.array([0.4, -0.2, 0.7])
    baseline = LinearBaseline(ridge=0.0)
    baseline.fit(features, returns)
    np.testing.assert_allclose(baseline.predict_batch(features), returns, atol=1e-12)


def test_baseline_can_reduce_score_gradient_variance() -> None:
    rng = np.random.default_rng(9)
    scores = rng.normal(size=(1000, 4))
    nuisance = 20.0 + rng.normal(scale=0.2, size=1000)
    no_baseline = reinforce_gradient(scores, nuisance)
    running = RunningMeanBaseline(count=1, mean=20.0)
    with_baseline = reinforce_gradient(
        scores,
        nuisance,
        baseline_predictions=running.predict_batch(np.ones((1000, 1))),
    )
    assert with_baseline.centered_variance < no_baseline.raw_variance / 1000.0


def test_natural_gradient_respects_local_kl_bound() -> None:
    probabilities = np.array([0.2, 0.3, 0.5])
    features = np.array([1.0, -0.5])
    fisher = linear_softmax_fisher(probabilities, features)
    gradient = np.linspace(-1.0, 1.0, fisher.shape[0])
    result = natural_gradient_direction(gradient, fisher, damping=1e-3, max_kl=0.01)
    assert result.predicted_kl <= 0.01 + 1e-12
    assert result.scale <= 1.0


def test_categorical_kl_is_zero_only_for_equal_probabilities() -> None:
    probabilities = np.array([0.25, 0.75])
    assert categorical_kl(probabilities, probabilities) == pytest.approx(0.0)
    assert categorical_kl(probabilities, np.array([0.5, 0.5])) > 0.0


def _transition(*, accepted: bool, log_ratio: float, entropy: float = 0.0) -> PolicyTransition:
    current = np.array([0.0, 0.0])
    proposed = np.array([1.0, -2.0])
    return PolicyTransition(
        current_state=current,
        proposed_state=proposed,
        next_state=proposed if accepted else current,
        accepted=accepted,
        log_acceptance_ratio=log_ratio,
        proposal_entropy=entropy,
        current_features=current,
        next_features=proposed if accepted else current,
    )


def test_policy_objectives_make_blind_spots_explicit() -> None:
    rejected = _transition(accepted=False, log_ratio=-3.0, entropy=2.0)
    assert AcceptanceObjective().reward(rejected) == 0.0
    assert AcceptedSquaredJumpObjective().reward(rejected) == 0.0
    assert GeneralizedSpeedObjective(beta=0.5).reward(rejected) == pytest.approx(-2.0)


def test_kernel_selection_training_learns_larger_jump_when_acceptance_is_comparable() -> None:
    target = GaussianTarget(np.zeros(1), np.array([[1000.0]]))
    proposals = [GaussianRandomWalkProposal(0.05), GaussianRandomWalkProposal(1.0)]
    policy = LinearSoftmaxPolicy(np.zeros((2, 1)), action_values=np.array([0.05, 1.0]))
    result = train_kernel_selection_policy(
        target,
        proposals,
        np.zeros(1),
        np.random.default_rng(17),
        policy=policy,
        objective=AcceptedSquaredJumpObjective(),
        n_updates=80,
        rollout_length=8,
        optimizer=Adam(learning_rate=0.05),
    )
    assert result.action_probabilities[1] > 0.8
    assert result.frozen_policy.weights.flags.writeable is False


def test_kernel_selection_rejects_state_dependent_frozen_mixture() -> None:
    target = GaussianTarget(np.zeros(1), np.eye(1))
    policy = LinearSoftmaxPolicy(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="one constant feature"):
        train_kernel_selection_policy(
            target,
            [GaussianRandomWalkProposal(0.1), GaussianRandomWalkProposal(1.0)],
            np.zeros(1),
            np.random.default_rng(1),
            policy=policy,
            objective=AcceptanceObjective(),
            n_updates=1,
        )


def test_contrastive_divergence_lower_bound_matches_derived_terms() -> None:
    from sampler_lab.learning.objectives import ContrastiveDivergenceLowerBoundObjective

    transition = PolicyTransition(
        current_state=np.array([0.0]),
        proposed_state=np.array([2.0]),
        next_state=np.array([0.0]),
        accepted=False,
        log_acceptance_ratio=np.log(0.25),
        proposal_entropy=0.0,
        current_features=np.array([0.0]),
        next_features=np.array([0.0]),
        log_target_ratio=-1.0,
        forward_log_proposal=-2.0,
    )
    expected = 0.25 * (-1.0) - 0.25 * np.log(0.25) - 0.25 * (-2.0)
    assert ContrastiveDivergenceLowerBoundObjective().reward(transition) == pytest.approx(expected)


def test_contrastive_divergence_objective_requires_augmented_mh_information() -> None:
    from sampler_lab.learning.objectives import ContrastiveDivergenceLowerBoundObjective

    with pytest.raises(ValueError, match="requires target and proposal"):
        ContrastiveDivergenceLowerBoundObjective().reward(_transition(accepted=True, log_ratio=0.0))


def test_policy_transition_allows_negative_infinite_rejection_ratio() -> None:
    transition = PolicyTransition(
        current_state=np.array([0.0]),
        proposed_state=np.array([1.0]),
        next_state=np.array([0.0]),
        accepted=False,
        log_acceptance_ratio=float("-inf"),
        proposal_entropy=0.0,
        current_features=np.array([0.0]),
        next_features=np.array([0.0]),
        log_target_ratio=float("-inf"),
        forward_log_proposal=-1.0,
    )
    assert AcceptanceObjective(use_probability=True).reward(transition) == 0.0


def test_frozen_policy_evaluation_uses_fresh_operation_counter() -> None:
    from sampler_lab.core.counters import OperationCounter
    from sampler_lab.learning.trainer import evaluate_frozen_policy

    target = GaussianTarget(np.zeros(1), np.eye(1))
    proposals = [GaussianRandomWalkProposal(0.2), GaussianRandomWalkProposal(1.0)]
    training_counter = OperationCounter()
    policy = LinearSoftmaxPolicy(np.zeros((2, 1)))
    result = train_kernel_selection_policy(
        target,
        proposals,
        np.zeros(1),
        np.random.default_rng(31),
        policy=policy,
        objective=AcceptedSquaredJumpObjective(),
        n_updates=2,
        rollout_length=3,
        counter=training_counter,
    )
    training_snapshot = result.operation_counts
    evaluation_counter = OperationCounter()
    evaluate_frozen_policy(
        result,
        np.zeros(1),
        np.random.default_rng(32),
        n_steps=5,
        counter=evaluation_counter,
    )
    assert result.operation_counts == training_snapshot
    assert evaluation_counter.log_density_evaluations == 10
    assert evaluation_counter.policy_evaluations == 0
