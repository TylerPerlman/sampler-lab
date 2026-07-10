"""Adaptive random-walk warmup and exact frozen evaluation kernels."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sampler_lab.adaptive.covariance import regularize_covariance
from sampler_lab.adaptive.dual_averaging import RobbinsMonroLogScale
from sampler_lab.adaptive.running_moments import RunningMoments
from sampler_lab.adaptive.schedules import RobbinsMonroSchedule
from sampler_lab.adaptive.warmup import (
    AdaptiveTrainingResult,
    EvaluationTrajectory,
    FrozenPolicy,
)
from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.protocols import LogDensity
from sampler_lab.core.results import Transition
from sampler_lab.mcmc.chain import run_chain
from sampler_lab.mcmc.metropolis import MetropolisHastingsKernel
from sampler_lab.mcmc.proposals import MultivariateGaussianRandomWalkProposal

Array = NDArray[np.float64]


def _as_vector(value: ArrayLike, *, name: str) -> Array:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return vector


@dataclass(slots=True)
class FrozenAdaptiveRandomWalkKernel:
    """Fixed Gaussian random-walk MH kernel produced after warmup."""

    target: LogDensity
    proposal_covariance: ArrayLike
    counter: OperationCounter | None = None
    _proposal: MultivariateGaussianRandomWalkProposal = field(init=False, repr=False)
    _kernel: MetropolisHastingsKernel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._proposal = MultivariateGaussianRandomWalkProposal(self.proposal_covariance)
        self._kernel = MetropolisHastingsKernel(self.target, self._proposal, self.counter)

    @property
    def covariance_matrix(self) -> Array:
        return np.asarray(self._proposal._covariance, dtype=np.float64).copy()

    def step(self, state: Array, rng: np.random.Generator) -> Transition:
        return self._kernel.step(state, rng)


@dataclass(frozen=True, slots=True)
class AdaptiveRandomWalkResult:
    """Adaptive warmup record and its frozen exact kernel."""

    training: AdaptiveTrainingResult
    frozen_kernel: FrozenAdaptiveRandomWalkKernel
    empirical_covariance: Array
    proposal_covariance: Array


def train_adaptive_random_walk(
    target: LogDensity,
    initial_state: ArrayLike,
    rng: np.random.Generator,
    *,
    n_warmup: int,
    initial_scale: float = 1.0,
    target_acceptance: float = 0.234,
    initial_covariance: ArrayLike | None = None,
    adapt_covariance: bool = True,
    covariance_start: int = 20,
    covariance_update_interval: int = 10,
    shrinkage: float = 0.05,
    eigenvalue_floor: float = 1e-6,
    eigenvalue_ceiling: float | None = None,
    schedule: RobbinsMonroSchedule | None = None,
    counter: OperationCounter | None = None,
) -> AdaptiveRandomWalkResult:
    """Warm up a Gaussian random walk, then freeze an ordinary MH kernel.

    Warmup states are returned only inside :class:`AdaptiveTrainingResult`; they are
    never merged with the fresh evaluation trajectory.
    """

    if isinstance(n_warmup, bool) or not isinstance(n_warmup, int):
        raise TypeError("n_warmup must be an integer")
    if n_warmup <= 0:
        raise ValueError("n_warmup must be positive")
    for name, value in (
        ("covariance_start", covariance_start),
        ("covariance_update_interval", covariance_update_interval),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    current = _as_vector(initial_state, name="initial_state")
    dimension = current.size
    if initial_covariance is None:
        base_covariance = np.eye(dimension, dtype=np.float64)
    else:
        base_covariance = regularize_covariance(
            initial_covariance,
            shrinkage=0.0,
            eigenvalue_floor=eigenvalue_floor,
            eigenvalue_ceiling=eigenvalue_ceiling,
        ).matrix
        if base_covariance.shape != (dimension, dimension):
            raise ValueError("initial_covariance must match the state dimension")

    scale_adaptation = RobbinsMonroLogScale(
        initial_scale,
        target_acceptance,
        schedule=schedule or RobbinsMonroSchedule(),
    )
    moments = RunningMoments(dimension)
    moments.update(current)
    states = np.empty((n_warmup + 1, dimension), dtype=np.float64)
    states[0] = current
    parameter_history = np.empty((n_warmup, 1 + dimension * dimension), dtype=np.float64)
    acceptance_probabilities = np.empty(n_warmup, dtype=np.float64)
    accepted_count = 0

    current_log_target = float(target.log_prob(current))
    if counter is not None:
        counter.log_density_evaluations += 1
    if not np.isfinite(current_log_target):
        raise ValueError("initial_state must lie in the target support")

    for step in range(n_warmup):
        if (
            adapt_covariance
            and moments.count > covariance_start
            and step % covariance_update_interval == 0
        ):
            base_covariance = regularize_covariance(
                moments.covariance(),
                shrinkage=shrinkage,
                eigenvalue_floor=eigenvalue_floor,
                eigenvalue_ceiling=eigenvalue_ceiling,
            ).matrix
        scale = scale_adaptation.scale
        proposal_covariance = scale * scale * base_covariance
        cholesky = np.asarray(np.linalg.cholesky(proposal_covariance), dtype=np.float64)
        proposal = np.asarray(current + cholesky @ rng.normal(size=dimension), dtype=np.float64)
        proposed_log_target = float(target.log_prob(proposal))
        if counter is not None:
            counter.normal_draws += dimension
            counter.log_density_evaluations += 1
            counter.matrix_factorizations += 1
            counter.uniform_draws += 1
        if np.isnan(proposed_log_target) or proposed_log_target == float("inf"):
            raise ValueError("target returned an invalid proposed log density")
        log_ratio = (
            float("-inf")
            if proposed_log_target == float("-inf")
            else proposed_log_target - current_log_target
        )
        acceptance_probability = float(np.exp(min(0.0, log_ratio)))
        accepted = bool(float(rng.random()) < acceptance_probability)
        if accepted:
            current = proposal
            current_log_target = proposed_log_target
            accepted_count += 1
        moments.update(current)
        scale_adaptation.update(acceptance_probability)
        states[step + 1] = current
        parameter_history[step, 0] = scale_adaptation.scale
        parameter_history[step, 1:] = base_covariance.reshape(-1)
        acceptance_probabilities[step] = acceptance_probability

    if adapt_covariance and moments.count > 1:
        empirical_covariance = moments.covariance()
        base_covariance = regularize_covariance(
            empirical_covariance,
            shrinkage=shrinkage,
            eigenvalue_floor=eigenvalue_floor,
            eigenvalue_ceiling=eigenvalue_ceiling,
        ).matrix
    else:
        empirical_covariance = base_covariance.copy()
    final_scale = scale_adaptation.scale
    proposal_covariance = np.asarray(final_scale * final_scale * base_covariance, dtype=np.float64)
    frozen_policy = FrozenPolicy(
        "adaptive-random-walk",
        np.concatenate(([final_scale], base_covariance.reshape(-1))),
        {
            "dimension": float(dimension),
            "warmup_steps": float(n_warmup),
            "target_acceptance": float(target_acceptance),
            "observed_acceptance": accepted_count / n_warmup,
        },
    )
    training = AdaptiveTrainingResult(
        states,
        parameter_history,
        acceptance_probabilities,
        frozen_policy,
        {
            "observed_acceptance": accepted_count / n_warmup,
            "mean_acceptance_probability": float(np.mean(acceptance_probabilities)),
            "final_scale": final_scale,
            "minimum_covariance_eigenvalue": float(np.min(np.linalg.eigvalsh(base_covariance))),
        },
    )
    frozen_kernel = FrozenAdaptiveRandomWalkKernel(target, proposal_covariance, counter)
    return AdaptiveRandomWalkResult(
        training=training,
        frozen_kernel=frozen_kernel,
        empirical_covariance=np.asarray(empirical_covariance, dtype=np.float64),
        proposal_covariance=proposal_covariance,
    )


def evaluate_adaptive_random_walk(
    result: AdaptiveRandomWalkResult,
    initial_state: ArrayLike,
    rng: np.random.Generator,
    *,
    n_steps: int,
    counter: OperationCounter | None = None,
) -> EvaluationTrajectory:
    """Run a fresh chain from the frozen post-warmup kernel."""

    evaluation_kernel = FrozenAdaptiveRandomWalkKernel(
        result.frozen_kernel.target,
        result.proposal_covariance,
        counter,
    )
    trajectory = run_chain(evaluation_kernel, initial_state, rng, n_steps=n_steps)
    return EvaluationTrajectory(
        trajectory=trajectory,
        frozen_policy=result.training.frozen_policy,
        training_steps_excluded=result.training.states.shape[0] - 1,
    )
