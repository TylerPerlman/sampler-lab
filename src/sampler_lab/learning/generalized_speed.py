"""Maximum-entropy generalized-speed adaptation for Gaussian random walks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.adaptive.warmup import EvaluationTrajectory, FrozenPolicy
from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.protocols import DifferentiableLogDensity
from sampler_lab.learning.optimizers import Adam, ParameterOptimizer
from sampler_lab.mcmc.chain import run_chain
from sampler_lab.mcmc.metropolis import MetropolisHastingsKernel
from sampler_lab.mcmc.proposals import GaussianRandomWalkProposal

Array = NDArray[np.float64]


def _as_vector(value: ArrayLike, *, name: str) -> Array:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return vector


@dataclass(frozen=True, slots=True)
class GeneralizedSpeedGradient:
    """One reparameterized objective and gradient evaluation."""

    objective: float
    gradient_log_scale: Array
    log_acceptance_ratio: float
    acceptance_probability: float
    proposal: Array


def diagonal_random_walk_generalized_speed_gradient(
    target: DifferentiableLogDensity,
    current_state: ArrayLike,
    noise: ArrayLike,
    log_scale: ArrayLike,
    *,
    beta: float,
) -> GeneralizedSpeedGradient:
    r"""Evaluate the sampled lower bound and its pathwise gradient.

    For ``y = x + exp(log_scale) * noise``, the objective is

    ``min(0, log pi(y) - log pi(x)) + beta * sum(log_scale)``.
    """

    current = _as_vector(current_state, name="current_state")
    epsilon = _as_vector(noise, name="noise")
    log_scale_vector = _as_vector(log_scale, name="log_scale")
    if epsilon.shape != current.shape or log_scale_vector.shape != current.shape:
        raise ValueError("noise and log_scale must match current_state")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be nonnegative and finite")
    scales = np.exp(log_scale_vector)
    proposal = np.asarray(current + scales * epsilon, dtype=np.float64)
    current_log_prob = float(target.log_prob(current))
    proposed_log_prob = float(target.log_prob(proposal))
    if not np.isfinite(current_log_prob):
        raise ValueError("current_state must lie in finite target support")
    if np.isnan(proposed_log_prob) or proposed_log_prob == float("inf"):
        raise ValueError("target returned an invalid proposed log density")
    log_ratio = (
        float("-inf")
        if proposed_log_prob == float("-inf")
        else proposed_log_prob - current_log_prob
    )
    entropy_term = beta * float(np.sum(log_scale_vector))
    if log_ratio < 0.0 and np.isfinite(log_ratio):
        proposed_gradient = np.asarray(target.grad_log_prob(proposal), dtype=np.float64)
        if proposed_gradient.shape != current.shape or not np.all(np.isfinite(proposed_gradient)):
            raise ValueError("target gradient at proposal must match the finite state")
        gradient = proposed_gradient * scales * epsilon + beta
    elif log_ratio == float("-inf"):
        gradient = np.full(current.size, beta, dtype=np.float64)
    else:
        gradient = np.full(current.size, beta, dtype=np.float64)
    return GeneralizedSpeedGradient(
        objective=float(min(0.0, log_ratio) + entropy_term),
        gradient_log_scale=np.asarray(gradient, dtype=np.float64),
        log_acceptance_ratio=float(log_ratio),
        acceptance_probability=float(np.exp(min(0.0, log_ratio))),
        proposal=proposal,
    )


@dataclass(frozen=True, slots=True, init=False)
class GeneralizedSpeedTrainingResult:
    """Adaptive history and exact frozen Gaussian random-walk kernel."""

    states: Array
    log_scale_history: Array
    beta_history: Array
    objective_history: Array
    acceptance_probabilities: Array
    frozen_policy: FrozenPolicy
    frozen_kernel: MetropolisHastingsKernel
    operation_counts: dict[str, int | dict[str, int]]

    def __init__(
        self,
        *,
        states: ArrayLike,
        log_scale_history: ArrayLike,
        beta_history: ArrayLike,
        objective_history: ArrayLike,
        acceptance_probabilities: ArrayLike,
        frozen_policy: FrozenPolicy,
        frozen_kernel: MetropolisHastingsKernel,
        operation_counts: dict[str, int | dict[str, int]],
    ) -> None:
        state_values = np.asarray(states, dtype=np.float64)
        scales = np.asarray(log_scale_history, dtype=np.float64)
        betas = np.asarray(beta_history, dtype=np.float64)
        objectives = np.asarray(objective_history, dtype=np.float64)
        acceptance = np.asarray(acceptance_probabilities, dtype=np.float64)
        n_steps = objectives.size
        if state_values.ndim != 2 or state_values.shape[0] != n_steps + 1:
            raise ValueError("states must contain one more row than objectives")
        if scales.shape != (n_steps + 1, state_values.shape[1]):
            raise ValueError("log_scale_history has the wrong shape")
        if betas.shape != (n_steps + 1,) or acceptance.shape != (n_steps,):
            raise ValueError("beta and acceptance histories have the wrong shape")
        if not all(
            np.all(np.isfinite(array))
            for array in (state_values, scales, betas, objectives, acceptance)
        ):
            raise ValueError("training histories must be finite")
        copies: list[Array] = []
        for array in (state_values, scales, betas, objectives, acceptance):
            detached = np.array(array, dtype=np.float64, copy=True)
            detached.setflags(write=False)
            copies.append(detached)
        object.__setattr__(self, "states", copies[0])
        object.__setattr__(self, "log_scale_history", copies[1])
        object.__setattr__(self, "beta_history", copies[2])
        object.__setattr__(self, "objective_history", copies[3])
        object.__setattr__(self, "acceptance_probabilities", copies[4])
        object.__setattr__(self, "frozen_policy", frozen_policy)
        object.__setattr__(self, "frozen_kernel", frozen_kernel)
        object.__setattr__(self, "operation_counts", dict(operation_counts))

    @property
    def final_scale(self) -> Array:
        return np.asarray(np.exp(self.log_scale_history[-1]), dtype=np.float64)


def train_generalized_speed_random_walk(
    target: DifferentiableLogDensity,
    initial_state: ArrayLike,
    rng: np.random.Generator,
    *,
    n_warmup: int,
    initial_scale: ArrayLike = 1.0,
    initial_beta: float = 1.0,
    target_acceptance: float = 0.234,
    beta_learning_rate: float = 0.02,
    optimizer: ParameterOptimizer | None = None,
    minimum_log_scale: float = -12.0,
    maximum_log_scale: float = 8.0,
    counter: OperationCounter | None = None,
) -> GeneralizedSpeedTrainingResult:
    """Adapt diagonal RWM scales using the generalized-speed lower bound."""

    if isinstance(n_warmup, bool) or not isinstance(n_warmup, int):
        raise TypeError("n_warmup must be an integer")
    if n_warmup <= 0:
        raise ValueError("n_warmup must be positive")
    if not np.isfinite(initial_beta) or initial_beta <= 0.0:
        raise ValueError("initial_beta must be positive and finite")
    if not np.isfinite(target_acceptance) or not 0.0 < target_acceptance < 1.0:
        raise ValueError("target_acceptance must lie in (0, 1)")
    if not np.isfinite(beta_learning_rate) or beta_learning_rate <= 0.0:
        raise ValueError("beta_learning_rate must be positive and finite")
    if minimum_log_scale >= maximum_log_scale:
        raise ValueError("minimum_log_scale must be below maximum_log_scale")

    current = _as_vector(initial_state, name="initial_state")
    scales = np.broadcast_to(np.asarray(initial_scale, dtype=np.float64), current.shape)
    if np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
        raise ValueError("initial_scale must broadcast to positive finite values")
    log_scale = np.log(scales).astype(np.float64)
    beta = float(initial_beta)
    resolved_optimizer = optimizer or Adam(learning_rate=0.02, gradient_clip_norm=20.0)
    total_counter = counter or OperationCounter()

    states = np.empty((n_warmup + 1, current.size), dtype=np.float64)
    log_scale_history = np.empty((n_warmup + 1, current.size), dtype=np.float64)
    beta_history = np.empty(n_warmup + 1, dtype=np.float64)
    objectives = np.empty(n_warmup, dtype=np.float64)
    acceptance_probabilities = np.empty(n_warmup, dtype=np.float64)
    states[0] = current
    log_scale_history[0] = log_scale
    beta_history[0] = beta

    for step in range(n_warmup):
        noise = np.asarray(rng.normal(size=current.size), dtype=np.float64)
        evaluation = diagonal_random_walk_generalized_speed_gradient(
            target,
            current,
            noise,
            log_scale,
            beta=beta,
        )
        updated = resolved_optimizer.step(log_scale, evaluation.gradient_log_scale)
        log_scale = np.clip(updated, minimum_log_scale, maximum_log_scale).astype(np.float64)
        total_counter.normal_draws += current.size
        total_counter.log_density_evaluations += 2
        if evaluation.log_acceptance_ratio < 0.0 and np.isfinite(evaluation.log_acceptance_ratio):
            total_counter.gradient_evaluations += 1
        total_counter.uniform_draws += 1
        accepted = bool(float(rng.random()) < evaluation.acceptance_probability)
        if accepted:
            current = evaluation.proposal
        beta *= max(
            1e-6,
            1.0 + beta_learning_rate * (float(accepted) - target_acceptance),
        )
        beta = float(np.clip(beta, 1e-6, 1e6))
        states[step + 1] = current
        log_scale_history[step + 1] = log_scale
        beta_history[step + 1] = beta
        objectives[step] = evaluation.objective
        acceptance_probabilities[step] = evaluation.acceptance_probability
        total_counter.increment("training_objective_evaluations")

    final_scale = np.exp(log_scale).astype(np.float64)
    proposal = GaussianRandomWalkProposal(final_scale)
    frozen_kernel = MetropolisHastingsKernel(target, proposal, total_counter)
    frozen_policy = FrozenPolicy(
        "generalized-speed-diagonal-rwm",
        np.concatenate((log_scale, np.asarray([beta]))),
        {
            "dimension": float(current.size),
            "warmup_steps": float(n_warmup),
            "target_acceptance": float(target_acceptance),
            "mean_acceptance_probability": float(np.mean(acceptance_probabilities)),
        },
    )
    return GeneralizedSpeedTrainingResult(
        states=states,
        log_scale_history=log_scale_history,
        beta_history=beta_history,
        objective_history=objectives,
        acceptance_probabilities=acceptance_probabilities,
        frozen_policy=frozen_policy,
        frozen_kernel=frozen_kernel,
        operation_counts=total_counter.snapshot(),
    )


def evaluate_generalized_speed(
    result: GeneralizedSpeedTrainingResult,
    initial_state: ArrayLike,
    rng: np.random.Generator,
    *,
    n_steps: int,
    counter: OperationCounter | None = None,
) -> EvaluationTrajectory:
    """Run a fresh exact MH trajectory after generalized-speed warmup."""

    evaluation_kernel = MetropolisHastingsKernel(
        result.frozen_kernel.target,
        result.frozen_kernel.proposal,
        counter,
    )
    trajectory = run_chain(evaluation_kernel, initial_state, rng, n_steps=n_steps)
    return EvaluationTrajectory(
        trajectory=trajectory,
        frozen_policy=result.frozen_policy,
        training_steps_excluded=result.objective_history.size,
    )
