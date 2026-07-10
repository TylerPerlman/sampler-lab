"""Reproducible adaptive, policy-gradient, variational, and Stein demonstrations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from sampler_lab.benchmarks import (
    BenchmarkResult,
    evaluate_samples,
    separated_gaussian_mixture_case,
)
from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.rng import spawn_rngs
from sampler_lab.diagnostics.time_series import empirical_effective_sample_size
from sampler_lab.learning.adaptive_mh import (
    evaluate_adaptive_random_walk,
    train_adaptive_random_walk,
)
from sampler_lab.learning.objectives import (
    AcceptanceObjective,
    AcceptedSquaredJumpObjective,
    ContrastiveDivergenceLowerBoundObjective,
    PolicyObjective,
)
from sampler_lab.learning.optimizers import Adam
from sampler_lab.learning.policies import LinearSoftmaxPolicy
from sampler_lab.learning.stein import run_svgd
from sampler_lab.learning.trainer import evaluate_frozen_policy, train_kernel_selection_policy
from sampler_lab.learning.variational import (
    DiagonalGaussianVariational,
    fit_reverse_kl_diagonal_gaussian,
)
from sampler_lab.mcmc.chain import run_chain
from sampler_lab.mcmc.metropolis import MetropolisHastingsKernel
from sampler_lab.mcmc.proposals import GaussianRandomWalkProposal
from sampler_lab.models.gaussian import GaussianTarget
from sampler_lab.models.gaussian_mixture import GaussianMixtureTarget


@dataclass(frozen=True, slots=True)
class ObjectiveStudyRow:
    """Learned proposal mixture and frozen-chain efficiency for one reward."""

    objective: str
    action_probabilities: tuple[float, ...]
    mean_scale: float
    acceptance_rate: float
    effective_sample_size: float
    training_steps: int


@dataclass(frozen=True, slots=True)
class LearningBenchmark:
    """Objective-gaming study and exact-reference mixture benchmark."""

    objective_rows: tuple[ObjectiveStudyRow, ...]
    benchmark_results: tuple[BenchmarkResult, ...]

    def to_json(self, *, indent: int | None = 2) -> str:
        payload = {
            "objective_rows": [
                {
                    "objective": row.objective,
                    "action_probabilities": row.action_probabilities,
                    "mean_scale": row.mean_scale,
                    "acceptance_rate": row.acceptance_rate,
                    "effective_sample_size": row.effective_sample_size,
                    "training_steps": row.training_steps,
                }
                for row in self.objective_rows
            ],
            "benchmark_results": [
                json.loads(result.to_json(indent=None)) for result in self.benchmark_results
            ],
        }
        return json.dumps(payload, indent=indent, sort_keys=True)


def _safe_ess(values: np.ndarray) -> float:
    if np.all(values == values[0]):
        return 1.0
    return empirical_effective_sample_size(values)


def run_objective_gaming_study(
    *,
    n_updates: int = 80,
    rollout_length: int = 12,
    n_evaluation_steps: int = 4_000,
    seed: int = 2022,
) -> tuple[ObjectiveStudyRow, ...]:
    """Compare acceptance, jump, and contrastive-divergence policy rewards."""

    if min(n_updates, rollout_length, n_evaluation_steps) <= 0:
        raise ValueError("study lengths must be positive")
    target = GaussianTarget(np.zeros(1), np.eye(1))
    scales = np.array([0.03, 0.3, 1.5, 4.0], dtype=np.float64)
    proposals = tuple(GaussianRandomWalkProposal(scale) for scale in scales)
    objectives: tuple[tuple[str, PolicyObjective], ...] = (
        ("acceptance", AcceptanceObjective()),
        ("accepted jump", AcceptedSquaredJumpObjective()),
        ("contrastive lower bound", ContrastiveDivergenceLowerBoundObjective()),
    )
    streams = iter(spawn_rngs(seed, 2 * len(objectives)))
    rows: list[ObjectiveStudyRow] = []
    for name, objective in objectives:
        policy = LinearSoftmaxPolicy(np.zeros((scales.size, 1)), action_values=scales)
        training = train_kernel_selection_policy(
            target,
            proposals,
            np.zeros(1),
            next(streams),
            policy=policy,
            objective=objective,
            n_updates=n_updates,
            rollout_length=rollout_length,
            optimizer=Adam(learning_rate=0.03, gradient_clip_norm=10.0),
        )
        evaluation_counter = OperationCounter()
        evaluation = evaluate_frozen_policy(
            training,
            np.zeros(1),
            next(streams),
            n_steps=n_evaluation_steps,
            counter=evaluation_counter,
        )
        samples = np.asarray(evaluation.trajectory.states[1:, 0], dtype=np.float64)
        probabilities = training.action_probabilities
        rows.append(
            ObjectiveStudyRow(
                objective=name,
                action_probabilities=tuple(float(value) for value in probabilities),
                mean_scale=float(probabilities @ scales),
                acceptance_rate=float(evaluation.trajectory.acceptance_rate or 0.0),
                effective_sample_size=_safe_ess(samples),
                training_steps=n_updates * rollout_length,
            )
        )
    return tuple(rows)


def run_learning_benchmark(
    *,
    n_samples: int = 3_000,
    n_warmup: int = 1_000,
    policy_updates: int = 80,
    variational_steps: int = 150,
    svgd_particles: int = 128,
    svgd_steps: int = 40,
    seed: int = 2023,
) -> tuple[BenchmarkResult, ...]:
    """Compare exact and approximate learned samplers on a separated mixture."""

    if min(n_samples, n_warmup, policy_updates, variational_steps, svgd_particles, svgd_steps) <= 0:
        raise ValueError("benchmark lengths must be positive")
    case = separated_gaussian_mixture_case(
        dimension=2,
        separation=8.0,
        condition_number=4.0,
        seed=19,
    )
    target = case.target
    if not isinstance(target, GaussianMixtureTarget):
        raise TypeError("separated mixture benchmark requires GaussianMixtureTarget")
    streams = iter(spawn_rngs(seed, 18))
    reference = case.reference_samples(next(streams), max(500, min(n_samples, 2_000)))
    initial = target.component_means[0]
    rows: list[BenchmarkResult] = []

    direct = case.reference_samples(next(streams), n_samples)
    rows.append(
        evaluate_samples(
            method="direct oracle",
            case=case,
            samples=direct,
            reference_samples=reference,
            exact_after_freeze=True,
            compute_mode_mixing=False,
        )
    )

    rwm_counter = OperationCounter()
    rwm = MetropolisHastingsKernel(target, GaussianRandomWalkProposal(0.8), rwm_counter)
    rwm_trajectory = run_chain(rwm, initial, next(streams), n_steps=n_samples)
    rows.append(
        evaluate_samples(
            method="isotropic RWM",
            case=case,
            samples=np.asarray(rwm_trajectory.states[1:], dtype=np.float64),
            reference_samples=reference,
            exact_after_freeze=True,
            acceptance_rate=rwm_trajectory.acceptance_rate,
            operation_counter=rwm_counter,
        )
    )

    adaptive_training_counter = OperationCounter()
    adaptive = train_adaptive_random_walk(
        target,
        initial,
        next(streams),
        n_warmup=n_warmup,
        initial_scale=0.5,
        target_acceptance=0.3,
        counter=adaptive_training_counter,
    )
    adaptive_evaluation_counter = OperationCounter()
    adaptive_evaluation = evaluate_adaptive_random_walk(
        adaptive,
        initial,
        next(streams),
        n_steps=n_samples,
        counter=adaptive_evaluation_counter,
    )
    rows.append(
        evaluate_samples(
            method="adaptive covariance RWM",
            case=case,
            samples=np.asarray(adaptive_evaluation.trajectory.states[1:], dtype=np.float64),
            reference_samples=reference,
            exact_after_freeze=True,
            acceptance_rate=adaptive_evaluation.trajectory.acceptance_rate,
            operation_counter=adaptive_evaluation_counter,
            diagnostics={
                "training_log_density_evaluations": float(
                    adaptive_training_counter.log_density_evaluations
                )
            },
        )
    )

    policy_scales = np.array([0.25, 1.2, 6.0], dtype=np.float64)
    policy = LinearSoftmaxPolicy(np.zeros((policy_scales.size, 1)), action_values=policy_scales)
    policy_training_counter = OperationCounter()
    policy_training = train_kernel_selection_policy(
        target,
        tuple(GaussianRandomWalkProposal(scale) for scale in policy_scales),
        initial,
        next(streams),
        policy=policy,
        objective=ContrastiveDivergenceLowerBoundObjective(),
        n_updates=policy_updates,
        rollout_length=8,
        optimizer=Adam(learning_rate=0.03, gradient_clip_norm=10.0),
        counter=policy_training_counter,
    )
    policy_evaluation_counter = OperationCounter()
    policy_evaluation = evaluate_frozen_policy(
        policy_training,
        initial,
        next(streams),
        n_steps=n_samples,
        counter=policy_evaluation_counter,
    )
    rows.append(
        evaluate_samples(
            method="policy-gradient MH",
            case=case,
            samples=np.asarray(policy_evaluation.trajectory.states[1:], dtype=np.float64),
            reference_samples=reference,
            exact_after_freeze=True,
            acceptance_rate=policy_evaluation.trajectory.acceptance_rate,
            operation_counter=policy_evaluation_counter,
            diagnostics={
                "training_policy_evaluations": float(policy_training_counter.policy_evaluations),
                "global_action_probability": float(policy_training.action_probabilities[-1]),
            },
        )
    )

    variational = DiagonalGaussianVariational(initial, np.zeros(target.dimension))
    fit = fit_reverse_kl_diagonal_gaussian(
        target,
        variational,
        next(streams),
        n_steps=variational_steps,
        batch_size=64,
        optimizer=Adam(learning_rate=0.03, gradient_clip_norm=20.0),
    )
    variational_samples = fit.approximation.sample(next(streams), n_samples)
    rows.append(
        evaluate_samples(
            method="reverse-KL Gaussian",
            case=case,
            samples=variational_samples,
            reference_samples=reference,
            exact_after_freeze=False,
            compute_mode_mixing=False,
            diagnostics={"final_reverse_kl_objective": float(fit.objective_history[-1])},
        )
    )

    corrected_counter = OperationCounter()
    corrected = run_chain(
        fit.approximation.corrected_kernel(target, counter=corrected_counter),
        initial,
        next(streams),
        n_steps=n_samples,
    )
    rows.append(
        evaluate_samples(
            method="reverse-KL independence MH",
            case=case,
            samples=np.asarray(corrected.states[1:], dtype=np.float64),
            reference_samples=reference,
            exact_after_freeze=True,
            acceptance_rate=corrected.acceptance_rate,
            operation_counter=corrected_counter,
        )
    )

    initial_particles = np.asarray(
        next(streams).normal(scale=4.0, size=(svgd_particles, target.dimension)),
        dtype=np.float64,
    )
    svgd = run_svgd(
        initial_particles,
        target,
        n_steps=svgd_steps,
        step_size=0.03,
        record_ksd=True,
    )
    rows.append(
        evaluate_samples(
            method="SVGD particles",
            case=case,
            samples=svgd.particles,
            reference_samples=reference,
            exact_after_freeze=False,
            compute_mode_mixing=False,
            diagnostics={"final_ksd": float(svgd.ksd_history[-1])},
        )
    )
    return tuple(rows)


def run_policy_learning_demo(
    *,
    n_samples: int = 3_000,
    n_warmup: int = 1_000,
    policy_updates: int = 80,
    seed: int = 2022,
) -> LearningBenchmark:
    """Run both Phase 10 demonstration laboratories."""

    objective_rows = run_objective_gaming_study(
        n_updates=policy_updates,
        rollout_length=10,
        n_evaluation_steps=n_samples,
        seed=seed,
    )
    benchmark_rows = run_learning_benchmark(
        n_samples=n_samples,
        n_warmup=n_warmup,
        policy_updates=policy_updates,
        variational_steps=max(20, policy_updates),
        svgd_particles=max(32, min(128, n_samples // 10)),
        svgd_steps=max(10, policy_updates // 2),
        seed=seed + 1,
    )
    return LearningBenchmark(objective_rows, benchmark_rows)


def policy_gradient_main() -> None:
    """Run adaptive-policy objective and separated-mixture comparisons."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--policy-updates", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    result = run_policy_learning_demo(
        n_samples=args.samples,
        n_warmup=args.warmup,
        policy_updates=args.policy_updates,
        seed=args.seed,
    )
    if args.as_json:
        print(result.to_json())
        return

    print("Policy-objective failure-mode study on N(0, 1)")
    print(
        f"{'objective':>26}  {'mean scale':>10}  {'accept':>8}  {'ESS':>10}  action probabilities"
    )
    for objective_row in result.objective_rows:
        probabilities = " ".join(f"{value:.2f}" for value in objective_row.action_probabilities)
        print(
            f"{objective_row.objective:>26}  {objective_row.mean_scale:10.3f}  "
            f"{objective_row.acceptance_rate:8.3f}  "
            f"{objective_row.effective_sample_size:10.1f}  {probabilities}"
        )

    print("\nSeparated anisotropic Gaussian-mixture benchmark")
    print(
        f"{'method':>30}  {'exact':>5}  {'accept':>8}  {'mean err':>10}  "
        f"{'cov err':>10}  {'mode L1':>9}  {'switches':>8}"
    )
    for benchmark_row in result.benchmark_results:
        acceptance = (
            "-" if benchmark_row.acceptance_rate is None else f"{benchmark_row.acceptance_rate:.3f}"
        )
        occupancy = benchmark_row.distribution.mode_occupancy_l1_error
        occupancy_text = "-" if occupancy is None else f"{occupancy:.3f}"
        switches = (
            "-" if benchmark_row.mode_mixing is None else str(benchmark_row.mode_mixing.n_switches)
        )
        print(
            f"{benchmark_row.method:>30}  "
            f"{benchmark_row.exact_after_freeze!s:>5}  {acceptance:>8}  "
            f"{benchmark_row.distribution.standardized_mean_error:10.3f}  "
            f"{benchmark_row.distribution.relative_covariance_error:10.3f}  "
            f"{occupancy_text:>9}  {switches:>8}"
        )


if __name__ == "__main__":
    policy_gradient_main()
