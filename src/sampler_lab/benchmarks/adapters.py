"""Capability-aware adapters for the common continuous benchmark suite."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sampler_lab.annealing.jarzynski import annealed_importance_sampling
from sampler_lab.annealing.paths import GeometricAnnealingPath
from sampler_lab.annealing.schedules import AnnealingSchedule
from sampler_lab.annealing.transitions import KernelPopulationTransition
from sampler_lab.benchmarks.capabilities import SamplerCapabilities
from sampler_lab.benchmarks.continuous_suite import (
    ContinuousTargetCase,
    ExactContinuousTarget,
    evaluate_samples,
    evaluate_weighted_samples,
)
from sampler_lab.benchmarks.metrics import BenchmarkResult
from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.rng import spawn_rngs
from sampler_lab.dynamics.hmc import HamiltonianMonteCarloKernel
from sampler_lab.dynamics.langevin import MetropolisAdjustedLangevinKernel
from sampler_lab.ensemble.state import EnsembleKernel, EnsembleState, run_ensemble_chain
from sampler_lab.ensemble.stretch import StretchMoveKernel
from sampler_lab.ensemble.walk import WalkMoveKernel
from sampler_lab.geometry.stochastic_newton import MetropolizedStochasticNewtonKernel
from sampler_lab.learning.adaptive_mh import (
    evaluate_adaptive_random_walk,
    train_adaptive_random_walk,
)
from sampler_lab.learning.objectives import ContrastiveDivergenceLowerBoundObjective
from sampler_lab.learning.optimizers import Adam
from sampler_lab.learning.policies import LinearSoftmaxPolicy
from sampler_lab.learning.stein import kernel_stein_discrepancy, run_svgd
from sampler_lab.learning.trainer import evaluate_frozen_policy, train_kernel_selection_policy
from sampler_lab.learning.variational import (
    DiagonalGaussianVariational,
    VariationalFitResult,
    fit_reverse_kl_diagonal_gaussian,
)
from sampler_lab.mcmc.chain import run_chain
from sampler_lab.mcmc.metropolis import MetropolisHastingsKernel
from sampler_lab.mcmc.proposals import MultivariateGaussianRandomWalkProposal
from sampler_lab.models.gaussian import GaussianTarget
from sampler_lab.particles.resampling import SystematicResampler

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Execution budget shared by continuous benchmark adapters."""

    n_samples: int = 2_000
    warmup_steps: int = 500
    reference_samples: int = 1_000
    n_walkers: int = 24
    variational_steps: int = 100
    policy_updates: int = 50
    policy_rollout_length: int = 8
    svgd_particles: int = 96
    svgd_steps: int = 30
    annealing_particles: int = 512
    annealing_steps: int = 16
    seed: int = 2022

    def __post_init__(self) -> None:
        values = (
            self.n_samples,
            self.warmup_steps,
            self.reference_samples,
            self.n_walkers,
            self.variational_steps,
            self.policy_updates,
            self.policy_rollout_length,
            self.svgd_particles,
            self.svgd_steps,
            self.annealing_particles,
            self.annealing_steps,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("benchmark lengths must be integers")
        if min(values) <= 0:
            raise ValueError("benchmark lengths must be positive")


@dataclass(frozen=True, slots=True)
class SamplerOutput:
    """Normalized output from one adapter before target-specific evaluation."""

    samples: Array
    exact_after_freeze: bool
    output_semantics: str
    log_weights: Array | None = None
    acceptance_rate: float | None = None
    evaluation_counter: OperationCounter = field(default_factory=OperationCounter)
    training_counter: OperationCounter = field(default_factory=OperationCounter)
    training_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    diagnostics: dict[str, float] = field(default_factory=dict)
    compute_mode_mixing: bool = True

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
            raise ValueError("adapter samples must be a nonempty matrix")
        if not np.all(np.isfinite(samples)):
            raise ValueError("adapter samples must be finite")
        copied_samples = np.array(samples, dtype=np.float64, copy=True)
        copied_samples.setflags(write=False)
        object.__setattr__(self, "samples", copied_samples)
        if self.log_weights is not None:
            weights = np.asarray(self.log_weights, dtype=np.float64)
            if weights.shape != (samples.shape[0],):
                raise ValueError("log_weights must contain one value per sample")
            if np.any(np.isnan(weights)) or np.any(np.isposinf(weights)):
                raise ValueError("log_weights may contain finite values or -inf only")
            copied_weights = np.array(weights, dtype=np.float64, copy=True)
            copied_weights.setflags(write=False)
            object.__setattr__(self, "log_weights", copied_weights)
        if self.training_seconds < 0.0 or self.evaluation_seconds < 0.0:
            raise ValueError("benchmark durations may not be negative")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


class ContinuousSamplerAdapter(Protocol):
    """Adapter from one repository method to the common benchmark output."""

    name: str
    description: str
    capabilities: SamplerCapabilities

    def run(
        self,
        case: ContinuousTargetCase,
        config: BenchmarkConfig,
        rng: np.random.Generator,
    ) -> SamplerOutput: ...


@dataclass(frozen=True, slots=True)
class FunctionalSamplerAdapter:
    """Small immutable adapter backed by a typed runner function."""

    name: str
    description: str
    capabilities: SamplerCapabilities
    runner: Callable[[ContinuousTargetCase, BenchmarkConfig, np.random.Generator], SamplerOutput]

    def run(
        self,
        case: ContinuousTargetCase,
        config: BenchmarkConfig,
        rng: np.random.Generator,
    ) -> SamplerOutput:
        return self.runner(case, config, rng)


def _target_scale(case: ContinuousTargetCase) -> float:
    dimension = case.target.mean_vector.size
    return float(np.sqrt(np.trace(case.target.covariance_matrix) / dimension))


def _initial_state(case: ContinuousTargetCase, rng: np.random.Generator) -> Array:
    candidates = case.reference_samples(rng, 64 if case.mode_labeler is not None else 1)
    if case.mode_labeler is None:
        return np.asarray(candidates[0], dtype=np.float64)
    labels = case.mode_labeler(candidates)
    matching = np.flatnonzero(labels == 0)
    index = int(matching[0]) if matching.size else 0
    return np.asarray(candidates[index], dtype=np.float64)


def evaluate_adapter_output(
    adapter: ContinuousSamplerAdapter,
    case: ContinuousTargetCase,
    output: SamplerOutput,
    reference: Array,
    replicate: int,
) -> BenchmarkResult:
    if output.log_weights is not None:
        return evaluate_weighted_samples(
            method=adapter.name,
            case=case,
            samples=output.samples,
            log_weights=output.log_weights,
            reference_samples=reference,
            exact_after_freeze=output.exact_after_freeze,
            operation_counter=output.evaluation_counter,
            training_operation_counter=output.training_counter,
            diagnostics=output.diagnostics,
            training_seconds=output.training_seconds,
            evaluation_seconds=output.evaluation_seconds,
            replicate=replicate,
            output_semantics=output.output_semantics,
        )
    return evaluate_samples(
        method=adapter.name,
        case=case,
        samples=output.samples,
        reference_samples=reference,
        exact_after_freeze=output.exact_after_freeze,
        acceptance_rate=output.acceptance_rate,
        operation_counter=output.evaluation_counter,
        training_operation_counter=output.training_counter,
        diagnostics=output.diagnostics,
        compute_mode_mixing=output.compute_mode_mixing,
        output_semantics=output.output_semantics,
        training_seconds=output.training_seconds,
        evaluation_seconds=output.evaluation_seconds,
        replicate=replicate,
    )


def run_adapter(
    adapter: ContinuousSamplerAdapter,
    case: ContinuousTargetCase,
    config: BenchmarkConfig,
    *,
    seed: int,
    replicate: int,
) -> BenchmarkResult:
    """Run and evaluate one compatible adapter/target pairing."""

    reference_rng, method_rng = spawn_rngs(seed, 2)
    reference = case.reference_samples(reference_rng, config.reference_samples)
    output = adapter.run(case, config, method_rng)
    return evaluate_adapter_output(adapter, case, output, reference, replicate)


def _direct_runner(
    case: ContinuousTargetCase,
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> SamplerOutput:
    counter = OperationCounter()
    start = perf_counter()
    samples = case.reference_samples(rng, config.n_samples)
    elapsed = perf_counter() - start
    counter.normal_draws += config.n_samples * samples.shape[1]
    return SamplerOutput(
        samples,
        True,
        "iid-samples",
        evaluation_counter=counter,
        evaluation_seconds=elapsed,
        compute_mode_mixing=False,
    )


def _importance_runner(
    case: ContinuousTargetCase,
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> SamplerOutput:
    proposal = GaussianTarget(case.target.mean_vector, case.target.covariance_matrix)
    counter = OperationCounter()
    start = perf_counter()
    samples = proposal.sample(rng, config.n_samples)
    log_weights = np.empty(config.n_samples, dtype=np.float64)
    for index, sample in enumerate(samples):
        log_weights[index] = case.target.log_prob(sample) - proposal.log_prob(sample)
    elapsed = perf_counter() - start
    counter.normal_draws += config.n_samples * samples.shape[1]
    counter.log_density_evaluations += config.n_samples
    counter.proposal_density_evaluations += config.n_samples
    return SamplerOutput(
        samples,
        True,
        "weighted-samples",
        log_weights=log_weights,
        evaluation_counter=counter,
        evaluation_seconds=elapsed,
        compute_mode_mixing=False,
    )


def _random_walk_runner(
    case: ContinuousTargetCase,
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> SamplerOutput:
    dimension = case.target.mean_vector.size
    covariance = case.target.covariance_matrix
    proposal_covariance = (1.2**2 / dimension) * covariance
    counter = OperationCounter()
    initial = _initial_state(case, rng)
    kernel = MetropolisHastingsKernel(
        case.target,
        MultivariateGaussianRandomWalkProposal(proposal_covariance),
        counter,
    )
    start = perf_counter()
    trajectory = run_chain(
        kernel,
        initial,
        rng,
        n_steps=config.warmup_steps + config.n_samples,
    )
    elapsed = perf_counter() - start
    samples = trajectory.samples(discard=config.warmup_steps + 1)[: config.n_samples]
    return SamplerOutput(
        samples,
        True,
        "markov-chain",
        acceptance_rate=trajectory.acceptance_rate,
        evaluation_counter=counter,
        evaluation_seconds=elapsed,
    )


def _adaptive_random_walk_runner(
    case: ContinuousTargetCase,
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> SamplerOutput:
    training_rng, evaluation_rng, initial_rng = spawn_rngs(int(rng.integers(0, 2**32)), 3)
    initial = _initial_state(case, initial_rng)
    training_counter = OperationCounter()
    training_start = perf_counter()
    trained = train_adaptive_random_walk(
        case.target,
        initial,
        training_rng,
        n_warmup=config.warmup_steps,
        initial_scale=max(0.05, 0.2 * _target_scale(case)),
        target_acceptance=0.3,
        counter=training_counter,
    )
    training_seconds = perf_counter() - training_start
    evaluation_counter = OperationCounter()
    evaluation_start = perf_counter()
    evaluated = evaluate_adaptive_random_walk(
        trained,
        initial,
        evaluation_rng,
        n_steps=config.n_samples,
        counter=evaluation_counter,
    )
    evaluation_seconds = perf_counter() - evaluation_start
    return SamplerOutput(
        np.asarray(evaluated.trajectory.states[1:], dtype=np.float64),
        True,
        "markov-chain",
        acceptance_rate=evaluated.trajectory.acceptance_rate,
        evaluation_counter=evaluation_counter,
        training_counter=training_counter,
        training_seconds=training_seconds,
        evaluation_seconds=evaluation_seconds,
        diagnostics={"frozen_scale": float(trained.training.diagnostics["final_scale"])},
    )


def _mala_runner(
    case: ContinuousTargetCase,
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> SamplerOutput:
    counter = OperationCounter()
    initial = _initial_state(case, rng)
    step_size = 0.08 / max(1.0, np.sqrt(_target_scale(case)))
    kernel = MetropolisAdjustedLangevinKernel(case.target, step_size, counter=counter)
    start = perf_counter()
    trajectory = run_chain(
        kernel,
        initial,
        rng,
        n_steps=config.warmup_steps + config.n_samples,
    )
    elapsed = perf_counter() - start
    samples = trajectory.samples(discard=config.warmup_steps + 1)[: config.n_samples]
    return SamplerOutput(
        samples,
        True,
        "markov-chain",
        acceptance_rate=trajectory.acceptance_rate,
        evaluation_counter=counter,
        evaluation_seconds=elapsed,
        diagnostics={"step_size": step_size},
    )


def _hmc_runner(
    case: ContinuousTargetCase,
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> SamplerOutput:
    counter = OperationCounter()
    initial = _initial_state(case, rng)
    dimension = initial.size
    step_size = 0.06 / max(1.0, np.sqrt(_target_scale(case)))
    kernel = HamiltonianMonteCarloKernel(
        case.target,
        step_size,
        max(4, min(12, dimension + 2)),
        counter=counter,
    )
    start = perf_counter()
    trajectory = run_chain(
        kernel,
        initial,
        rng,
        n_steps=config.warmup_steps + config.n_samples,
    )
    elapsed = perf_counter() - start
    samples = trajectory.samples(discard=config.warmup_steps + 1)[: config.n_samples]
    return SamplerOutput(
        samples,
        True,
        "markov-chain",
        acceptance_rate=trajectory.acceptance_rate,
        evaluation_counter=counter,
        evaluation_seconds=elapsed,
        diagnostics={
            "step_size": step_size,
            "leapfrog_steps": float(kernel.n_leapfrog_steps),
        },
    )


def _stochastic_newton_runner(
    case: ContinuousTargetCase,
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> SamplerOutput:
    counter = OperationCounter()
    initial = _initial_state(case, rng)
    kernel = MetropolizedStochasticNewtonKernel(
        case.target,
        step_size=0.35,
        minimum_eigenvalue=1e-4,
        counter=counter,
    )
    start = perf_counter()
    trajectory = run_chain(
        kernel,
        initial,
        rng,
        n_steps=config.warmup_steps + config.n_samples,
    )
    elapsed = perf_counter() - start
    samples = trajectory.samples(discard=config.warmup_steps + 1)[: config.n_samples]
    return SamplerOutput(
        samples,
        True,
        "markov-chain",
        acceptance_rate=trajectory.acceptance_rate,
        evaluation_counter=counter,
        evaluation_seconds=elapsed,
        diagnostics={"step_size": 0.35},
    )


def _initial_ensemble(
    case: ContinuousTargetCase,
    rng: np.random.Generator,
    n_walkers: int,
) -> EnsembleState:
    dimension = case.target.mean_vector.size
    count = max(n_walkers, 2 * (dimension + 1))
    center = _initial_state(case, rng)
    cholesky = np.linalg.cholesky(case.target.covariance_matrix)
    walkers = center + 0.08 * (rng.normal(size=(count, dimension)) @ cholesky.T)
    return EnsembleState.from_target(walkers, case.target)


def _ensemble_runner(
    case: ContinuousTargetCase,
    config: BenchmarkConfig,
    rng: np.random.Generator,
    *,
    method: str,
) -> SamplerOutput:
    counter = OperationCounter()
    initial = _initial_ensemble(case, rng, config.n_walkers)
    counter.normal_draws += initial.n_walkers * initial.dimension
    counter.log_density_evaluations += initial.n_walkers
    kernel: EnsembleKernel
    if method == "stretch-ensemble":
        kernel = StretchMoveKernel(case.target, schedule="split", counter=counter)
    else:
        kernel = WalkMoveKernel(case.target, schedule="split", counter=counter)
    warmup_sweeps = max(1, int(np.ceil(config.warmup_steps / initial.n_walkers)))
    sample_sweeps = max(1, int(np.ceil(config.n_samples / initial.n_walkers)))
    start = perf_counter()
    trajectory = run_ensemble_chain(
        kernel,
        initial,
        rng,
        n_steps=warmup_sweeps + sample_sweeps,
    )
    elapsed = perf_counter() - start
    samples = trajectory.samples(discard=warmup_sweeps + 1, flatten=True)[: config.n_samples]
    return SamplerOutput(
        samples,
        True,
        "ensemble-chain",
        acceptance_rate=trajectory.acceptance_rate,
        evaluation_counter=counter,
        evaluation_seconds=elapsed,
        diagnostics={"n_walkers": float(initial.n_walkers)},
        compute_mode_mixing=False,
    )


def _reverse_kl_fit(
    case: ContinuousTargetCase,
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> tuple[VariationalFitResult, OperationCounter, float]:
    dimension = case.target.mean_vector.size
    initial_scale = np.sqrt(np.maximum(np.diag(case.target.covariance_matrix), 1e-12))
    family = DiagonalGaussianVariational(case.target.mean_vector, np.log(initial_scale))
    counter = OperationCounter()
    start = perf_counter()
    fit = fit_reverse_kl_diagonal_gaussian(
        case.target,
        family,
        rng,
        n_steps=config.variational_steps,
        batch_size=min(128, max(32, config.n_samples // 10)),
        optimizer=Adam(learning_rate=0.025, gradient_clip_norm=20.0),
    )
    elapsed = perf_counter() - start
    batch_size = min(128, max(32, config.n_samples // 10))
    counter.normal_draws += config.variational_steps * batch_size * dimension
    counter.log_density_evaluations += config.variational_steps * batch_size
    counter.gradient_evaluations += config.variational_steps * batch_size
    return fit, counter, elapsed


def _reverse_kl_runner(
    case: ContinuousTargetCase,
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> SamplerOutput:
    training_rng, evaluation_rng = spawn_rngs(int(rng.integers(0, 2**32)), 2)
    fit, training_counter, training_seconds = _reverse_kl_fit(case, config, training_rng)
    evaluation_counter = OperationCounter()
    start = perf_counter()
    samples = fit.approximation.sample(evaluation_rng, config.n_samples)
    elapsed = perf_counter() - start
    evaluation_counter.normal_draws += config.n_samples * samples.shape[1]
    return SamplerOutput(
        samples,
        False,
        "approximate-iid-samples",
        evaluation_counter=evaluation_counter,
        training_counter=training_counter,
        training_seconds=training_seconds,
        evaluation_seconds=elapsed,
        diagnostics={"final_reverse_kl_objective": float(fit.objective_history[-1])},
        compute_mode_mixing=False,
    )


def _reverse_kl_corrected_runner(
    case: ContinuousTargetCase,
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> SamplerOutput:
    training_rng, evaluation_rng, initial_rng = spawn_rngs(int(rng.integers(0, 2**32)), 3)
    fit, training_counter, training_seconds = _reverse_kl_fit(case, config, training_rng)
    evaluation_counter = OperationCounter()
    initial = _initial_state(case, initial_rng)
    kernel = fit.approximation.corrected_kernel(case.target, counter=evaluation_counter)
    start = perf_counter()
    trajectory = run_chain(kernel, initial, evaluation_rng, n_steps=config.n_samples)
    elapsed = perf_counter() - start
    return SamplerOutput(
        np.asarray(trajectory.states[1:], dtype=np.float64),
        True,
        "markov-chain",
        acceptance_rate=trajectory.acceptance_rate,
        evaluation_counter=evaluation_counter,
        training_counter=training_counter,
        training_seconds=training_seconds,
        evaluation_seconds=elapsed,
        diagnostics={"final_reverse_kl_objective": float(fit.objective_history[-1])},
    )


def _svgd_runner(
    case: ContinuousTargetCase,
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> SamplerOutput:
    n_particles = min(config.n_samples, config.svgd_particles)
    dimension = case.target.mean_vector.size
    marginal_scale = np.clip(
        np.sqrt(np.maximum(np.diag(case.target.covariance_matrix), 1e-12)),
        0.25,
        2.0,
    )
    initial = case.target.mean_vector + rng.normal(size=(n_particles, dimension)) * marginal_scale
    initial_scores = np.asarray(
        [case.target.grad_log_prob(particle) for particle in initial],
        dtype=np.float64,
    )
    maximum_score_norm = float(np.max(np.linalg.norm(initial_scores, axis=1)))
    step_size = min(0.02, 0.02 / max(1.0, maximum_score_norm))
    counter = OperationCounter()
    start = perf_counter()
    result = run_svgd(
        initial,
        case.target,
        n_steps=config.svgd_steps,
        step_size=step_size,
        record_ksd=True,
    )
    elapsed = perf_counter() - start
    counter.normal_draws += n_particles * dimension
    counter.gradient_evaluations += 2 * config.svgd_steps * n_particles
    return SamplerOutput(
        result.particles,
        False,
        "approximate-particles",
        evaluation_counter=OperationCounter(),
        training_counter=counter,
        training_seconds=elapsed,
        diagnostics={
            "final_ksd": float(result.ksd_history[-1]),
            "reported_ksd": float(kernel_stein_discrepancy(result.particles, case.target)),
            "step_size": step_size,
        },
        compute_mode_mixing=False,
    )


def _policy_gradient_runner(
    case: ContinuousTargetCase,
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> SamplerOutput:
    training_rng, evaluation_rng, initial_rng = spawn_rngs(int(rng.integers(0, 2**32)), 3)
    initial = _initial_state(case, initial_rng)
    scale = _target_scale(case)
    action_scales = np.asarray([0.05, 0.25, 1.0, 3.0], dtype=np.float64) * max(scale, 0.1)
    proposals = tuple(
        MultivariateGaussianRandomWalkProposal(
            (action_scale**2 / initial.size) * np.eye(initial.size)
        )
        for action_scale in action_scales
    )
    policy = LinearSoftmaxPolicy(np.zeros((action_scales.size, 1)), action_values=action_scales)
    training_counter = OperationCounter()
    start = perf_counter()
    training = train_kernel_selection_policy(
        case.target,
        proposals,
        initial,
        training_rng,
        policy=policy,
        objective=ContrastiveDivergenceLowerBoundObjective(),
        n_updates=config.policy_updates,
        rollout_length=config.policy_rollout_length,
        optimizer=Adam(learning_rate=0.03, gradient_clip_norm=10.0),
        counter=training_counter,
    )
    training_seconds = perf_counter() - start
    evaluation_counter = OperationCounter()
    start = perf_counter()
    evaluation = evaluate_frozen_policy(
        training,
        initial,
        evaluation_rng,
        n_steps=config.n_samples,
        counter=evaluation_counter,
    )
    evaluation_seconds = perf_counter() - start
    return SamplerOutput(
        np.asarray(evaluation.trajectory.states[1:], dtype=np.float64),
        True,
        "markov-chain",
        acceptance_rate=evaluation.trajectory.acceptance_rate,
        evaluation_counter=evaluation_counter,
        training_counter=training_counter,
        training_seconds=training_seconds,
        evaluation_seconds=evaluation_seconds,
        diagnostics={
            "largest_scale_probability": float(training.action_probabilities[-1]),
            "mean_proposal_scale": float(training.action_probabilities @ action_scales),
        },
    )


@dataclass(frozen=True, slots=True)
class _GeometricPathTarget:
    base: ExactContinuousTarget
    target: ExactContinuousTarget
    beta: float

    def log_prob(self, x: Array) -> float:
        return float(
            (1.0 - self.beta) * self.base.log_prob(x) + self.beta * self.target.log_prob(x)
        )

    def grad_log_prob(self, x: Array) -> Array:
        return np.asarray(
            (1.0 - self.beta) * self.base.grad_log_prob(x)
            + self.beta * self.target.grad_log_prob(x),
            dtype=np.float64,
        )


def _annealed_smc_runner(
    case: ContinuousTargetCase,
    config: BenchmarkConfig,
    rng: np.random.Generator,
) -> SamplerOutput:
    base = GaussianTarget(case.target.mean_vector, case.target.covariance_matrix)
    counter = OperationCounter()
    initial = base.sample(rng, config.annealing_particles)
    counter.normal_draws += initial.size
    path = GeometricAnnealingPath(base, case.target)
    transition = KernelPopulationTransition(
        lambda beta: MetropolisAdjustedLangevinKernel(
            _GeometricPathTarget(base, case.target, beta),
            step_size=0.04 / max(1.0, np.sqrt(_target_scale(case))),
            counter=counter,
        ),
        n_steps=1,
    )
    schedule = AnnealingSchedule.power(config.annealing_steps, 1.5)
    start = perf_counter()
    result = annealed_importance_sampling(
        initial,
        path,
        schedule,
        transition,
        rng,
        resampler=SystematicResampler(),
        resample_ess_fraction=0.5,
        target_particle_count=config.annealing_particles,
    )
    elapsed = perf_counter() - start
    counter.extra["path_endpoint_log_density_evaluations"] = (
        2 * config.annealing_particles * config.annealing_steps
    )
    final = result.final_cloud
    return SamplerOutput(
        final.particles,
        False,
        "weighted-particles",
        log_weights=final.log_weights,
        evaluation_counter=counter,
        evaluation_seconds=elapsed,
        diagnostics={
            "minimum_stage_ess": float(np.min(result.ess_history)),
            "n_resampling_stages": float(np.count_nonzero(result.resampled)),
            "log_normalizing_constant_ratio": result.log_normalizing_constant_ratio,
        },
        compute_mode_mixing=False,
    )


def default_continuous_adapters() -> tuple[FunctionalSamplerAdapter, ...]:
    """Return the maintained adapter matrix for common continuous targets."""

    exact_chain = SamplerCapabilities()
    return (
        FunctionalSamplerAdapter(
            "direct-oracle",
            "exact IID reference draws",
            SamplerCapabilities(is_markov_chain=False),
            _direct_runner,
        ),
        FunctionalSamplerAdapter(
            "importance",
            "self-normalized Gaussian importance sampling",
            SamplerCapabilities(
                produces_weighted_samples=True,
                is_markov_chain=False,
                requires_normalized_density=True,
            ),
            _importance_runner,
        ),
        FunctionalSamplerAdapter(
            "annealed-smc",
            "geometric-path annealed SMC with MALA mutation",
            SamplerCapabilities(
                produces_weighted_samples=True,
                is_markov_chain=False,
                is_exact_after_freeze=False,
                requires_gradient=True,
            ),
            _annealed_smc_runner,
        ),
        FunctionalSamplerAdapter(
            "random-walk-mh", "covariance-scaled RWM", exact_chain, _random_walk_runner
        ),
        FunctionalSamplerAdapter(
            "adaptive-random-walk",
            "regularized adaptive RWM with frozen evaluation",
            exact_chain,
            _adaptive_random_walk_runner,
        ),
        FunctionalSamplerAdapter(
            "mala",
            "Metropolis-adjusted Langevin",
            SamplerCapabilities(requires_gradient=True),
            _mala_runner,
        ),
        FunctionalSamplerAdapter(
            "hmc",
            "fresh-momentum Hamiltonian Monte Carlo",
            SamplerCapabilities(requires_gradient=True),
            _hmc_runner,
        ),
        FunctionalSamplerAdapter(
            "stochastic-newton",
            "Metropolized local Hessian proposal",
            SamplerCapabilities(requires_gradient=True, requires_hessian=True),
            _stochastic_newton_runner,
        ),
        FunctionalSamplerAdapter(
            "stretch-ensemble",
            "Goodman-Weare stretch ensemble",
            exact_chain,
            lambda c, q, r: _ensemble_runner(c, q, r, method="stretch-ensemble"),
        ),
        FunctionalSamplerAdapter(
            "walk-ensemble",
            "symmetric affine walk ensemble",
            exact_chain,
            lambda c, q, r: _ensemble_runner(c, q, r, method="walk-ensemble"),
        ),
        FunctionalSamplerAdapter(
            "reverse-kl",
            "approximate diagonal-Gaussian reverse-KL fit",
            SamplerCapabilities(
                is_markov_chain=False,
                is_exact_after_freeze=False,
                requires_gradient=True,
                supports_multimodality=False,
            ),
            _reverse_kl_runner,
        ),
        FunctionalSamplerAdapter(
            "reverse-kl-independence-mh",
            "reverse-KL proposal with exact independence-MH correction",
            SamplerCapabilities(requires_gradient=True),
            _reverse_kl_corrected_runner,
        ),
        FunctionalSamplerAdapter(
            "svgd",
            "approximate Stein variational particle flow",
            SamplerCapabilities(
                is_markov_chain=False,
                is_exact_after_freeze=False,
                requires_gradient=True,
            ),
            _svgd_runner,
        ),
        FunctionalSamplerAdapter(
            "policy-gradient-mh",
            "contrastive-reward proposal-scale policy with frozen MH mixture",
            exact_chain,
            _policy_gradient_runner,
        ),
    )


def adapter_by_name(name: str) -> FunctionalSamplerAdapter:
    """Resolve one maintained adapter by its stable public name."""

    for adapter in default_continuous_adapters():
        if adapter.name == name:
            return adapter
    raise KeyError(f"unknown continuous benchmark adapter: {name!r}")
