from __future__ import annotations

import numpy as np

from sampler_lab.learning.generalized_speed import (
    diagonal_random_walk_generalized_speed_gradient,
    train_generalized_speed_random_walk,
)
from sampler_lab.learning.mixtures import PolicyMixtureProposal
from sampler_lab.learning.optimizers import Adam
from sampler_lab.learning.policies import LinearSoftmaxPolicy
from sampler_lab.mcmc.proposals import GaussianRandomWalkProposal
from sampler_lab.models.gaussian import GaussianTarget


def test_generalized_speed_gradient_matches_common_noise_finite_difference() -> None:
    target = GaussianTarget(np.array([0.5, -1.0]), np.array([[1.0, 0.2], [0.2, 2.0]]))
    current = np.array([1.3, -0.4])
    noise = np.array([1.2, -0.7])
    log_scale = np.log(np.array([0.8, 1.1]))
    beta = 0.4
    analytic = diagonal_random_walk_generalized_speed_gradient(
        target,
        current,
        noise,
        log_scale,
        beta=beta,
    ).gradient_log_scale
    numerical = np.empty_like(log_scale)
    epsilon = 1e-6
    for index in range(log_scale.size):
        plus = log_scale.copy()
        minus = log_scale.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_value = diagonal_random_walk_generalized_speed_gradient(
            target,
            current,
            noise,
            plus,
            beta=beta,
        ).objective
        minus_value = diagonal_random_walk_generalized_speed_gradient(
            target,
            current,
            noise,
            minus,
            beta=beta,
        ).objective
        numerical[index] = (plus_value - minus_value) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, numerical, atol=2e-8)


def test_generalized_speed_training_keeps_positive_noncollapsed_scale() -> None:
    target = GaussianTarget(np.zeros(2), np.diag([0.1, 10.0]))
    result = train_generalized_speed_random_walk(
        target,
        np.array([2.0, -3.0]),
        np.random.default_rng(21),
        n_warmup=600,
        initial_scale=np.array([0.01, 0.01]),
        optimizer=Adam(learning_rate=0.02),
    )
    assert np.all(result.final_scale > 0.03)
    assert result.final_scale[1] > result.final_scale[0]
    assert result.states.flags.writeable is False


def test_policy_mixture_density_uses_state_dependent_marginal_weights() -> None:
    mutable = LinearSoftmaxPolicy(
        np.array([[0.0, 1.0], [0.0, -1.0]]),
        action_values=np.array([0.2, 1.0]),
    )
    policy = mutable.freeze()
    proposals = (GaussianRandomWalkProposal(0.2), GaussianRandomWalkProposal(1.0))

    def features(state: np.ndarray) -> np.ndarray:
        return np.array([1.0, state[0]])

    mixture = PolicyMixtureProposal(policy, proposals, features)
    source = np.array([0.7])
    destination = np.array([-0.1])
    probabilities = policy.probabilities(features(source))
    component_densities = np.array(
        [np.exp(proposal.log_transition_density(destination, source)) for proposal in proposals]
    )
    expected = np.log(probabilities @ component_densities)
    np.testing.assert_allclose(
        mixture.log_transition_density(destination, source),
        expected,
        atol=1e-14,
    )
