"""REINFORCE training for mixtures of exact Metropolis--Hastings kernels."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.adaptive.warmup import EvaluationTrajectory
from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.protocols import LogDensity
from sampler_lab.core.results import Transition
from sampler_lab.learning.baselines import Baseline, RunningMeanBaseline
from sampler_lab.learning.gradients import discounted_returns, reinforce_gradient
from sampler_lab.learning.objectives import PolicyObjective, PolicyTransition
from sampler_lab.learning.optimizers import Adam, ParameterOptimizer
from sampler_lab.learning.policies import FrozenLinearSoftmaxPolicy, LinearSoftmaxPolicy
from sampler_lab.mcmc.chain import MCMCTrajectory, run_chain
from sampler_lab.mcmc.metropolis import MetropolisHastingsKernel, log_metropolis_hastings_ratio
from sampler_lab.mcmc.proposals import (
    Array,
    GaussianRandomWalkProposal,
    MultivariateGaussianRandomWalkProposal,
    Proposal,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
FeatureMap = Callable[[Array], FloatArray]


def constant_features(state: Array) -> FloatArray:
    """State-independent feature map used for exact frozen kernel mixtures."""

    del state
    return np.ones(1, dtype=np.float64)


def identity_features(state: Array) -> FloatArray:
    """Flatten the state into diagnostic features."""

    return np.asarray(state, dtype=np.float64).reshape(-1)


def _validate_feature(value: ArrayLike, *, expected: int | None = None) -> FloatArray:
    feature = np.asarray(value, dtype=np.float64)
    if feature.ndim != 1 or feature.size == 0 or not np.all(np.isfinite(feature)):
        raise ValueError("feature maps must return nonempty finite vectors")
    if expected is not None and feature.size != expected:
        raise ValueError("policy feature dimension changed")
    return feature


@dataclass(slots=True)
class FrozenKernelMixture:
    """State-independent mixture of individually invariant Markov kernels."""

    policy: FrozenLinearSoftmaxPolicy
    kernels: tuple[MetropolisHastingsKernel, ...]
    _features: FloatArray = field(default_factory=lambda: np.ones(1), repr=False)

    def __post_init__(self) -> None:
        if len(self.kernels) != self.policy.n_actions:
            raise ValueError("kernel count must match policy actions")
        if self.policy.n_features != 1:
            raise ValueError("exact component-kernel mixtures require a constant policy")
        self._features = np.ones(1, dtype=np.float64)

    @property
    def probabilities(self) -> FloatArray:
        return self.policy.probabilities(self._features)

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        action = int(rng.choice(len(self.kernels), p=self.probabilities))
        transition = self.kernels[action].step(state, rng)
        diagnostics = dict(transition.diagnostics)
        diagnostics["policy_action"] = float(action)
        diagnostics["policy_probability"] = float(self.probabilities[action])
        return Transition(
            state=transition.state,
            accepted=transition.accepted,
            log_acceptance_ratio=transition.log_acceptance_ratio,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True, slots=True, init=False)
class PolicyTrainingResult:
    """Warmup rollouts and a frozen exact kernel mixture."""

    states: FloatArray
    rewards: FloatArray
    actions: IntArray
    accepted: BoolArray
    parameter_history: FloatArray
    gradient_norms: FloatArray
    gradient_variance: FloatArray
    baseline_gradient_variance: FloatArray
    frozen_policy: FrozenLinearSoftmaxPolicy
    frozen_kernel: FrozenKernelMixture
    operation_counts: dict[str, int | dict[str, int]]

    def __init__(
        self,
        *,
        states: ArrayLike,
        rewards: ArrayLike,
        actions: ArrayLike,
        accepted: ArrayLike,
        parameter_history: ArrayLike,
        gradient_norms: ArrayLike,
        gradient_variance: ArrayLike,
        baseline_gradient_variance: ArrayLike,
        frozen_policy: FrozenLinearSoftmaxPolicy,
        frozen_kernel: FrozenKernelMixture,
        operation_counts: dict[str, int | dict[str, int]],
    ) -> None:
        state_values = np.asarray(states, dtype=np.float64)
        reward_values = np.asarray(rewards, dtype=np.float64)
        action_values = np.asarray(actions, dtype=np.int64)
        accepted_values = np.asarray(accepted, dtype=np.bool_)
        parameters = np.asarray(parameter_history, dtype=np.float64)
        gradient_norm_values = np.asarray(gradient_norms, dtype=np.float64)
        raw_variance = np.asarray(gradient_variance, dtype=np.float64)
        centered_variance = np.asarray(baseline_gradient_variance, dtype=np.float64)
        n_steps = reward_values.size
        if state_values.ndim != 2 or state_values.shape[0] != n_steps + 1:
            raise ValueError("states must contain one more row than rewards")
        if action_values.shape != (n_steps,) or accepted_values.shape != (n_steps,):
            raise ValueError("actions and accepted flags must match rewards")
        if parameters.ndim != 2 or parameters.shape[0] != gradient_norm_values.size + 1:
            raise ValueError("parameter history must have one more row than updates")
        if (
            raw_variance.shape != gradient_norm_values.shape
            or centered_variance.shape != raw_variance.shape
        ):
            raise ValueError("gradient diagnostics must have equal shapes")
        arrays = (
            state_values,
            reward_values,
            parameters,
            gradient_norm_values,
            raw_variance,
            centered_variance,
        )
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("policy training arrays must be finite")
        copied: list[np.ndarray] = []
        for array in (
            state_values,
            reward_values,
            action_values,
            accepted_values,
            parameters,
            gradient_norm_values,
            raw_variance,
            centered_variance,
        ):
            detached = np.array(array, copy=True)
            detached.setflags(write=False)
            copied.append(detached)
        object.__setattr__(self, "states", copied[0])
        object.__setattr__(self, "rewards", copied[1])
        object.__setattr__(self, "actions", copied[2])
        object.__setattr__(self, "accepted", copied[3])
        object.__setattr__(self, "parameter_history", copied[4])
        object.__setattr__(self, "gradient_norms", copied[5])
        object.__setattr__(self, "gradient_variance", copied[6])
        object.__setattr__(self, "baseline_gradient_variance", copied[7])
        object.__setattr__(self, "frozen_policy", frozen_policy)
        object.__setattr__(self, "frozen_kernel", frozen_kernel)
        object.__setattr__(self, "operation_counts", dict(operation_counts))

    @property
    def action_probabilities(self) -> FloatArray:
        return self.frozen_kernel.probabilities

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards))

    @property
    def acceptance_rate(self) -> float:
        return float(np.mean(self.accepted))


def _proposal_entropy(proposal: Proposal, dimension: int) -> float:
    covariance: np.ndarray | None = None
    if isinstance(proposal, MultivariateGaussianRandomWalkProposal):
        candidate = np.asarray(proposal._covariance, dtype=np.float64)
        if candidate.shape == (dimension, dimension):
            covariance = candidate
    elif isinstance(proposal, GaussianRandomWalkProposal):
        scale = np.broadcast_to(np.asarray(proposal.scale, dtype=np.float64), (dimension,))
        covariance = np.diag(scale * scale)
    if covariance is None:
        raise ValueError("proposal entropy is unavailable; pass explicit proposal_entropies")
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0.0:
        raise ValueError("proposal covariance must be positive definite")
    return float(0.5 * (dimension * (1.0 + np.log(2.0 * np.pi)) + log_determinant))


def train_kernel_selection_policy(
    target: LogDensity,
    proposals: Sequence[Proposal],
    initial_state: ArrayLike,
    rng: np.random.Generator,
    *,
    policy: LinearSoftmaxPolicy,
    objective: PolicyObjective,
    n_updates: int,
    rollout_length: int = 16,
    policy_feature_map: FeatureMap = constant_features,
    diagnostic_feature_map: FeatureMap = identity_features,
    baseline: Baseline | None = None,
    optimizer: ParameterOptimizer | None = None,
    discount: float = 1.0,
    normalize_advantages: bool = False,
    proposal_entropies: ArrayLike | None = None,
    counter: OperationCounter | None = None,
) -> PolicyTrainingResult:
    """Train a categorical policy over individually exact MH kernels.

    The ordinary exact evaluation API requires a state-independent policy. A
    state-dependent action policy would generally destroy invariance when mixing
    already-corrected kernels, so this function refuses to freeze one.
    """

    if not proposals:
        raise ValueError("at least one proposal is required")
    if len(proposals) != policy.n_actions:
        raise ValueError("proposal count must match policy actions")
    for name, value in (("n_updates", n_updates), ("rollout_length", rollout_length)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if policy.n_features != 1:
        raise ValueError("exact frozen component mixtures currently require one constant feature")
    initial = np.asarray(initial_state, dtype=np.float64)
    if initial.ndim != 1 or initial.size == 0 or not np.all(np.isfinite(initial)):
        raise ValueError("initial_state must be a nonempty finite vector")
    if not np.allclose(_validate_feature(policy_feature_map(initial), expected=1), 1.0):
        raise ValueError("exact frozen component mixtures require constant feature [1]")
    if proposal_entropies is None:
        entropy_values = np.asarray(
            [_proposal_entropy(proposal, initial.size) for proposal in proposals],
            dtype=np.float64,
        )
    else:
        entropy_values = np.asarray(proposal_entropies, dtype=np.float64)
        if entropy_values.shape != (len(proposals),) or not np.all(np.isfinite(entropy_values)):
            raise ValueError("proposal_entropies must have one finite value per proposal")

    resolved_baseline = baseline or RunningMeanBaseline()
    resolved_optimizer = optimizer or Adam(learning_rate=0.02, gradient_clip_norm=10.0)
    total_steps = n_updates * rollout_length
    states = np.empty((total_steps + 1, initial.size), dtype=np.float64)
    rewards = np.empty(total_steps, dtype=np.float64)
    actions = np.empty(total_steps, dtype=np.int64)
    accepted = np.empty(total_steps, dtype=np.bool_)
    parameter_history = np.empty((n_updates + 1, policy.parameters.size), dtype=np.float64)
    gradient_norms = np.empty(n_updates, dtype=np.float64)
    raw_variances = np.empty(n_updates, dtype=np.float64)
    centered_variances = np.empty(n_updates, dtype=np.float64)
    current = np.array(initial, copy=True)
    states[0] = current
    parameter_history[0] = policy.parameters
    global_step = 0
    total_counter = counter or OperationCounter()

    for update in range(n_updates):
        rollout_scores = np.empty((rollout_length, policy.parameters.size), dtype=np.float64)
        rollout_policy_features = np.empty((rollout_length, policy.n_features), dtype=np.float64)
        rollout_rewards = np.empty(rollout_length, dtype=np.float64)
        for local_step in range(rollout_length):
            policy_features = _validate_feature(
                policy_feature_map(current), expected=policy.n_features
            )
            diagnostic_current = _validate_feature(diagnostic_feature_map(current))
            action = policy.act(rng, policy_features)
            if action.index is None:
                raise TypeError("kernel-selection training requires a categorical policy")
            action_index = action.index
            total_counter.increment("policy_evaluations")
            proposal_kernel = proposals[action_index]
            proposed = np.asarray(
                proposal_kernel.sample(current, rng, counter=total_counter), dtype=np.float64
            )
            if proposed.shape != current.shape or not np.all(np.isfinite(proposed)):
                raise ValueError("proposal returned an invalid state")
            current_log_target = float(target.log_prob(current))
            proposed_log_target = float(target.log_prob(proposed))
            total_counter.log_density_evaluations += 2
            forward = proposal_kernel.log_transition_density(
                proposed, current, counter=total_counter
            )
            reverse = proposal_kernel.log_transition_density(
                current, proposed, counter=total_counter
            )
            log_ratio = log_metropolis_hastings_ratio(
                current_log_target=current_log_target,
                proposed_log_target=proposed_log_target,
                forward_log_proposal=forward,
                reverse_log_proposal=reverse,
            )
            total_counter.uniform_draws += 1
            is_accepted = bool(np.log(float(rng.random())) < min(0.0, log_ratio))
            next_state = proposed if is_accepted else current
            diagnostic_next = _validate_feature(
                diagnostic_feature_map(next_state), expected=diagnostic_current.size
            )
            transition = PolicyTransition(
                current_state=np.array(current, copy=True),
                proposed_state=np.array(proposed, copy=True),
                next_state=np.array(next_state, copy=True),
                accepted=is_accepted,
                log_acceptance_ratio=float(log_ratio),
                proposal_entropy=float(entropy_values[action_index]),
                current_features=diagnostic_current,
                next_features=diagnostic_next,
                log_target_ratio=float(proposed_log_target - current_log_target),
                forward_log_proposal=float(forward),
            )
            reward = float(objective.reward(transition, total_counter))
            if not np.isfinite(reward):
                raise ValueError("policy objective returned a nonfinite reward")
            rollout_scores[local_step] = action.score
            rollout_policy_features[local_step] = policy_features
            rollout_rewards[local_step] = reward
            states[global_step + 1] = next_state
            rewards[global_step] = reward
            actions[global_step] = action_index
            accepted[global_step] = is_accepted
            current = np.array(next_state, copy=True)
            global_step += 1

        returns = discounted_returns(rollout_rewards, discount=discount)
        predictions = resolved_baseline.predict_batch(rollout_policy_features)
        estimate = reinforce_gradient(
            rollout_scores,
            returns,
            baseline_predictions=predictions,
            normalize_advantages=normalize_advantages,
        )
        new_parameters = resolved_optimizer.step(policy.parameters, estimate.gradient)
        policy.set_parameters(new_parameters)
        resolved_baseline.fit(rollout_policy_features, returns)
        parameter_history[update + 1] = policy.parameters
        gradient_norms[update] = float(np.linalg.norm(estimate.gradient))
        raw_variances[update] = estimate.raw_variance
        centered_variances[update] = estimate.centered_variance
        total_counter.increment("training_objective_evaluations", rollout_length)

    frozen_policy = policy.freeze(name="kernel-selection-softmax")
    kernels = tuple(
        MetropolisHastingsKernel(target, proposal, total_counter) for proposal in proposals
    )
    frozen_kernel = FrozenKernelMixture(frozen_policy, kernels)
    return PolicyTrainingResult(
        states=states,
        rewards=rewards,
        actions=actions,
        accepted=accepted,
        parameter_history=parameter_history,
        gradient_norms=gradient_norms,
        gradient_variance=raw_variances,
        baseline_gradient_variance=centered_variances,
        frozen_policy=frozen_policy,
        frozen_kernel=frozen_kernel,
        operation_counts=total_counter.snapshot(),
    )


def evaluate_frozen_policy(
    result: PolicyTrainingResult,
    initial_state: ArrayLike,
    rng: np.random.Generator,
    *,
    n_steps: int,
    counter: OperationCounter | None = None,
) -> EvaluationTrajectory:
    """Run a fresh chain from a learned, frozen exact kernel mixture."""

    evaluation_kernels = tuple(
        MetropolisHastingsKernel(kernel.target, kernel.proposal, counter)
        for kernel in result.frozen_kernel.kernels
    )
    evaluation_kernel = FrozenKernelMixture(result.frozen_policy, evaluation_kernels)
    trajectory: MCMCTrajectory = run_chain(
        evaluation_kernel,
        initial_state,
        rng,
        n_steps=n_steps,
    )
    return EvaluationTrajectory(
        trajectory=trajectory,
        frozen_policy=result.frozen_policy.as_generic_frozen_policy(),
        training_steps_excluded=result.rewards.size,
    )
