"""Reproducible rare-event and small-noise importance-sampling experiments."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal, cast

import numpy as np

from sampler_lab.core.counters import OperationCounter
from sampler_lab.core.rng import spawn_rngs
from sampler_lab.rare_events import (
    GaussianHalfspaceRareEvent,
    GaussianShiftProposal,
    GaussianTemperedProposal,
    GaussianTwoSidedRareEvent,
    estimate_with_mixture,
    estimate_with_shift,
    estimate_with_tempering,
    exact_relative_error,
    exact_shifted_log_second_moment,
    exact_symmetric_mixture_log_second_moment,
    exact_tempered_log_second_moment,
    fixed_scale_temperature,
    gaussian_linear_event_log_asymptotic,
    symmetric_twist_mixture,
)
from sampler_lab.rare_events.problems import RareGaussianProblem


@dataclass(frozen=True, slots=True)
class RareEventExperimentRow:
    """One method/problem/noise-level comparison."""

    problem: str
    method: str
    epsilon: float
    estimate: float
    log_estimate: float
    truth: float
    log_truth: float
    absolute_relative_error: float
    observed_relative_standard_error: float
    exact_relative_standard_error: float
    scaled_log_relative_variance: float
    event_count: int
    contribution_ess: float
    normal_draws: int


def _relative_error(estimate: float, truth: float) -> float:
    if truth == 0.0:
        return float("nan")
    return float(abs(estimate - truth) / truth)


def _make_problem(kind: Literal["one-sided", "two-sided"], dimension: int) -> RareGaussianProblem:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    direction = np.linspace(0.5, 1.5, dimension, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    eigenvalues = np.geomspace(0.5, 2.0, dimension)
    covariance = np.diag(eigenvalues)
    if kind == "one-sided":
        return GaussianHalfspaceRareEvent(direction, 1.0, covariance)
    return GaussianTwoSidedRareEvent(direction, 1.0, covariance)


def _row(
    *,
    problem_name: str,
    method: str,
    epsilon: float,
    problem: RareGaussianProblem,
    estimate: float,
    log_estimate: float,
    observed_rse: float,
    log_second_moment: float,
    event_count: int,
    contribution_ess: float,
    counter: OperationCounter,
    n_samples: int,
) -> RareEventExperimentRow:
    truth = problem.exact_probability(epsilon)
    log_truth = problem.exact_log_probability(epsilon)
    relative_variance, exact_rse, _log_relative_second = exact_relative_error(
        log_probability=log_truth,
        log_second_moment=log_second_moment,
        n_samples=n_samples,
    )
    log_relative_variance = (
        math.log(relative_variance) if relative_variance > 0.0 else float("-inf")
    )
    normal_draws = counter.normal_draws
    return RareEventExperimentRow(
        problem=problem_name,
        method=method,
        epsilon=epsilon,
        estimate=estimate,
        log_estimate=log_estimate,
        truth=truth,
        log_truth=log_truth,
        absolute_relative_error=_relative_error(estimate, truth),
        observed_relative_standard_error=observed_rse,
        exact_relative_standard_error=exact_rse,
        scaled_log_relative_variance=float(epsilon * log_relative_variance),
        event_count=event_count,
        contribution_ess=contribution_ess,
        normal_draws=normal_draws,
    )


def run_rare_event_experiment(
    *,
    epsilons: tuple[float, ...],
    n_samples: int,
    dimension: int,
    seed: int,
) -> list[RareEventExperimentRow]:
    """Compare crude, tempered, and twisted proposals on one- and two-minimizer events."""

    if not epsilons or any(not np.isfinite(value) or value <= 0.0 for value in epsilons):
        raise ValueError("epsilons must be positive and finite")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    one_sided = cast(GaussianHalfspaceRareEvent, _make_problem("one-sided", dimension))
    two_sided = cast(GaussianTwoSidedRareEvent, _make_problem("two-sided", dimension))
    method_count = 4 * len(epsilons) + 4 * len(epsilons)
    streams = iter(spawn_rngs(seed, method_count))
    rows: list[RareEventExperimentRow] = []

    for epsilon in epsilons:
        zero = np.zeros(dimension, dtype=np.float64)
        dominant = one_sided.dominant_point

        crude_counter = OperationCounter()
        crude = estimate_with_shift(
            one_sided,
            GaussianShiftProposal(zero, one_sided.covariance_matrix, epsilon),
            next(streams),
            n_samples,
            counter=crude_counter,
        )
        rows.append(
            _row(
                problem_name="one-sided",
                method="crude",
                epsilon=epsilon,
                problem=one_sided,
                estimate=crude.value,
                log_estimate=crude.log_value,
                observed_rse=crude.relative_standard_error,
                log_second_moment=one_sided.exact_log_probability(epsilon),
                event_count=crude.event_count,
                contribution_ess=crude.contribution_effective_sample_size,
                counter=crude_counter,
                n_samples=n_samples,
            )
        )

        fixed_counter = OperationCounter()
        fixed_temperature = 4.0
        fixed = estimate_with_tempering(
            one_sided,
            GaussianTemperedProposal(
                one_sided.covariance_matrix,
                epsilon,
                fixed_temperature,
            ),
            next(streams),
            n_samples,
            counter=fixed_counter,
        )
        rows.append(
            _row(
                problem_name="one-sided",
                method="temperature-4",
                epsilon=epsilon,
                problem=one_sided,
                estimate=fixed.value,
                log_estimate=fixed.log_value,
                observed_rse=fixed.relative_standard_error,
                log_second_moment=exact_tempered_log_second_moment(
                    one_sided, fixed_temperature, epsilon
                ),
                event_count=fixed.event_count,
                contribution_ess=fixed.contribution_effective_sample_size,
                counter=fixed_counter,
                n_samples=n_samples,
            )
        )

        scale_counter = OperationCounter()
        scale_temperature = fixed_scale_temperature(epsilon)
        scaled = estimate_with_tempering(
            one_sided,
            GaussianTemperedProposal(
                one_sided.covariance_matrix,
                epsilon,
                scale_temperature,
            ),
            next(streams),
            n_samples,
            counter=scale_counter,
        )
        rows.append(
            _row(
                problem_name="one-sided",
                method="fixed-covariance-tempering",
                epsilon=epsilon,
                problem=one_sided,
                estimate=scaled.value,
                log_estimate=scaled.log_value,
                observed_rse=scaled.relative_standard_error,
                log_second_moment=exact_tempered_log_second_moment(
                    one_sided, scale_temperature, epsilon
                ),
                event_count=scaled.event_count,
                contribution_ess=scaled.contribution_effective_sample_size,
                counter=scale_counter,
                n_samples=n_samples,
            )
        )

        twist_counter = OperationCounter()
        twist = estimate_with_shift(
            one_sided,
            GaussianShiftProposal(dominant, one_sided.covariance_matrix, epsilon),
            next(streams),
            n_samples,
            counter=twist_counter,
        )
        rows.append(
            _row(
                problem_name="one-sided",
                method="dominant-point-twist",
                epsilon=epsilon,
                problem=one_sided,
                estimate=twist.value,
                log_estimate=twist.log_value,
                observed_rse=twist.relative_standard_error,
                log_second_moment=exact_shifted_log_second_moment(one_sided, dominant, epsilon),
                event_count=twist.event_count,
                contribution_ess=twist.contribution_effective_sample_size,
                counter=twist_counter,
                n_samples=n_samples,
            )
        )

    for epsilon in epsilons:
        zero = np.zeros(dimension, dtype=np.float64)
        dominant = two_sided.dominant_point

        crude_counter = OperationCounter()
        crude = estimate_with_shift(
            two_sided,
            GaussianShiftProposal(zero, two_sided.covariance_matrix, epsilon),
            next(streams),
            n_samples,
            counter=crude_counter,
        )
        rows.append(
            _row(
                problem_name="two-sided",
                method="crude",
                epsilon=epsilon,
                problem=two_sided,
                estimate=crude.value,
                log_estimate=crude.log_value,
                observed_rse=crude.relative_standard_error,
                log_second_moment=two_sided.exact_log_probability(epsilon),
                event_count=crude.event_count,
                contribution_ess=crude.contribution_effective_sample_size,
                counter=crude_counter,
                n_samples=n_samples,
            )
        )

        temperature_counter = OperationCounter()
        temperature = fixed_scale_temperature(epsilon)
        tempered = estimate_with_tempering(
            two_sided,
            GaussianTemperedProposal(two_sided.covariance_matrix, epsilon, temperature),
            next(streams),
            n_samples,
            counter=temperature_counter,
        )
        rows.append(
            _row(
                problem_name="two-sided",
                method="fixed-covariance-tempering",
                epsilon=epsilon,
                problem=two_sided,
                estimate=tempered.value,
                log_estimate=tempered.log_value,
                observed_rse=tempered.relative_standard_error,
                log_second_moment=exact_tempered_log_second_moment(two_sided, temperature, epsilon),
                event_count=tempered.event_count,
                contribution_ess=tempered.contribution_effective_sample_size,
                counter=temperature_counter,
                n_samples=n_samples,
            )
        )

        single_counter = OperationCounter()
        single = estimate_with_shift(
            two_sided,
            GaussianShiftProposal(dominant, two_sided.covariance_matrix, epsilon),
            next(streams),
            n_samples,
            counter=single_counter,
        )
        rows.append(
            _row(
                problem_name="two-sided",
                method="single-twist",
                epsilon=epsilon,
                problem=two_sided,
                estimate=single.value,
                log_estimate=single.log_value,
                observed_rse=single.relative_standard_error,
                log_second_moment=exact_shifted_log_second_moment(two_sided, dominant, epsilon),
                event_count=single.event_count,
                contribution_ess=single.contribution_effective_sample_size,
                counter=single_counter,
                n_samples=n_samples,
            )
        )

        mixture_counter = OperationCounter()
        mixture = estimate_with_mixture(
            two_sided,
            symmetric_twist_mixture(two_sided, epsilon),
            next(streams),
            n_samples,
            counter=mixture_counter,
        )
        rows.append(
            _row(
                problem_name="two-sided",
                method="two-dominating-point-mixture",
                epsilon=epsilon,
                problem=two_sided,
                estimate=mixture.value,
                log_estimate=mixture.log_value,
                observed_rse=mixture.relative_standard_error,
                log_second_moment=exact_symmetric_mixture_log_second_moment(two_sided, epsilon),
                event_count=mixture.event_count,
                contribution_ess=mixture.contribution_effective_sample_size,
                counter=mixture_counter,
                n_samples=n_samples,
            )
        )

    return rows


def _print_rows(rows: list[RareEventExperimentRow]) -> None:
    print(
        f"{'problem':>11}  {'method':>30}  {'epsilon':>8}  {'estimate':>12}  "
        f"{'truth':>12}  {'rel.err':>9}  {'obs RSE':>9}  {'exact RSE':>10}  "
        f"{'eps log RV':>10}  {'events':>8}"
    )
    for row in rows:
        print(
            f"{row.problem:>11}  {row.method:>30}  {row.epsilon:8.4f}  "
            f"{row.estimate:12.5g}  {row.truth:12.5g}  "
            f"{row.absolute_relative_error:9.3g}  "
            f"{row.observed_relative_standard_error:9.3g}  "
            f"{row.exact_relative_standard_error:10.3g}  "
            f"{row.scaled_log_relative_variance:10.4f}  {row.event_count:8d}"
        )


def rare_event_main() -> None:
    """CLI entry point for the Phase 11 rare-event experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.5, 0.25, 0.125, 0.0625])
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = run_rare_event_experiment(
        epsilons=tuple(args.epsilons),
        n_samples=args.samples,
        dimension=args.dimension,
        seed=args.seed,
    )
    if args.json:
        payload = {
            "rows": [asdict(row) for row in rows],
            "laplace_log_approximations": {
                kind: [
                    gaussian_linear_event_log_asymptotic(
                        _make_problem(kind, args.dimension), epsilon
                    )
                    for epsilon in args.epsilons
                ]
                for kind in ("one-sided", "two-sided")
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_rows(rows)


__all__ = ["RareEventExperimentRow", "rare_event_main", "run_rare_event_experiment"]
