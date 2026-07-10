"""Replicated, capability-aware benchmark execution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

import numpy as np

from sampler_lab.benchmarks.adapters import (
    BenchmarkConfig,
    ContinuousSamplerAdapter,
    default_continuous_adapters,
    evaluate_adapter_output,
)
from sampler_lab.benchmarks.capabilities import check_compatibility
from sampler_lab.benchmarks.continuous_suite import (
    ContinuousTargetCase,
    default_continuous_cases,
)
from sampler_lab.benchmarks.metrics import BenchmarkResult


@dataclass(frozen=True, slots=True)
class BenchmarkExclusion:
    method: str
    target: str
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkFailure:
    method: str
    target: str
    replicate: int
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class BenchmarkAggregate:
    """Across-replicate summary for one method/target pair."""

    method: str
    target: str
    n_replicates: int
    exact_after_freeze: bool
    output_semantics: str
    standardized_mean_error_mean: float
    standardized_mean_error_se: float
    relative_covariance_error_mean: float
    relative_covariance_error_se: float
    imq_mmd_mean: float | None
    imq_mmd_se: float | None
    mode_occupancy_l1_error_mean: float | None
    mode_occupancy_l1_error_se: float | None
    acceptance_rate_mean: float | None
    acceptance_rate_se: float | None
    training_seconds_mean: float
    evaluation_seconds_mean: float

    @property
    def total_seconds_mean(self) -> float:
        """Mean training plus evaluation wall time for honest cost frontiers."""

        return self.training_seconds_mean + self.evaluation_seconds_mean


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteResult:
    """Complete replicated benchmark report, including exclusions and failures."""

    config: BenchmarkConfig
    n_replicates: int
    seed: int
    package_version: str
    results: tuple[BenchmarkResult, ...]
    exclusions: tuple[BenchmarkExclusion, ...]
    failures: tuple[BenchmarkFailure, ...]

    @property
    def aggregates(self) -> tuple[BenchmarkAggregate, ...]:
        return aggregate_results(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "n_replicates": self.n_replicates,
            "seed": self.seed,
            "package_version": self.package_version,
            "results": [asdict(result) for result in self.results],
            "aggregates": [asdict(row) for row in self.aggregates],
            "exclusions": [asdict(item) for item in self.exclusions],
            "failures": [asdict(item) for item in self.failures],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _package_version() -> str:
    try:
        return version("sampler-lab")
    except PackageNotFoundError:
        return "uninstalled"


def _seed_for(base: int, *indices: int) -> int:
    sequence = np.random.SeedSequence([base, *indices])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _select_cases(
    items: tuple[ContinuousTargetCase, ...],
    names: tuple[str, ...] | None,
) -> tuple[ContinuousTargetCase, ...]:
    if names is None:
        return items
    by_name = {item.name: item for item in items}
    missing = tuple(name for name in names if name not in by_name)
    if missing:
        raise KeyError(f"unknown benchmark targets: {missing!r}")
    return tuple(by_name[name] for name in names)


def _select_adapters(
    items: tuple[ContinuousSamplerAdapter, ...],
    names: tuple[str, ...] | None,
) -> tuple[ContinuousSamplerAdapter, ...]:
    if names is None:
        return items
    by_name = {item.name: item for item in items}
    missing = tuple(name for name in names if name not in by_name)
    if missing:
        raise KeyError(f"unknown benchmark methods: {missing!r}")
    return tuple(by_name[name] for name in names)


def run_benchmark_suite(
    *,
    config: BenchmarkConfig | None = None,
    n_replicates: int = 3,
    seed: int = 2022,
    cases: tuple[ContinuousTargetCase, ...] | None = None,
    adapters: tuple[ContinuousSamplerAdapter, ...] | None = None,
    target_names: tuple[str, ...] | None = None,
    method_names: tuple[str, ...] | None = None,
    fail_fast: bool = False,
) -> BenchmarkSuiteResult:
    """Run every selected compatible pair with shared references per replicate."""

    if isinstance(n_replicates, bool) or not isinstance(n_replicates, int):
        raise TypeError("n_replicates must be an integer")
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive")
    resolved_config = config or BenchmarkConfig(seed=seed)
    case_items = cases if cases is not None else default_continuous_cases()
    adapter_items = cast(
        tuple[ContinuousSamplerAdapter, ...],
        adapters if adapters is not None else default_continuous_adapters(),
    )
    resolved_cases = _select_cases(case_items, target_names)
    resolved_adapters = _select_adapters(adapter_items, method_names)
    results: list[BenchmarkResult] = []
    exclusions: list[BenchmarkExclusion] = []
    failures: list[BenchmarkFailure] = []

    for target_index, case in enumerate(resolved_cases):
        for method_index, adapter in enumerate(resolved_adapters):
            compatibility = check_compatibility(adapter.capabilities, case.capabilities)
            if not compatibility.compatible:
                exclusions.append(
                    BenchmarkExclusion(adapter.name, case.name, compatibility.reasons)
                )
                continue
            for replicate in range(n_replicates):
                reference_rng = np.random.default_rng(_seed_for(seed, target_index, replicate, 0))
                method_rng = np.random.default_rng(
                    _seed_for(seed, target_index, method_index, replicate, 1)
                )
                reference = case.reference_samples(
                    reference_rng,
                    resolved_config.reference_samples,
                )
                try:
                    output = adapter.run(case, resolved_config, method_rng)
                    results.append(
                        evaluate_adapter_output(
                            adapter,
                            case,
                            output,
                            reference,
                            replicate,
                        )
                    )
                except Exception as error:  # benchmark reports must preserve partial progress
                    if fail_fast:
                        raise
                    failures.append(
                        BenchmarkFailure(
                            adapter.name,
                            case.name,
                            replicate,
                            type(error).__name__,
                            str(error),
                        )
                    )
    return BenchmarkSuiteResult(
        resolved_config,
        n_replicates,
        seed,
        _package_version(),
        tuple(results),
        tuple(exclusions),
        tuple(failures),
    )


def _mean_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if array.size <= 1:
        return mean, 0.0
    return mean, float(np.std(array, ddof=1) / np.sqrt(array.size))


def _optional_mean_se(values: list[float | None]) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return None, None
    return _mean_se(finite)


def aggregate_results(results: tuple[BenchmarkResult, ...]) -> tuple[BenchmarkAggregate, ...]:
    """Aggregate common metrics without merging unlike method/target semantics."""

    groups: dict[tuple[str, str], list[BenchmarkResult]] = {}
    for result in results:
        groups.setdefault((result.method, result.target), []).append(result)
    rows: list[BenchmarkAggregate] = []
    for method, target in sorted(groups):
        group = groups[(method, target)]
        mean_error = _mean_se([row.distribution.standardized_mean_error for row in group])
        covariance_error = _mean_se([row.distribution.relative_covariance_error for row in group])
        mmd = _optional_mean_se([row.distribution.imq_mmd for row in group])
        occupancy = _optional_mean_se([row.distribution.mode_occupancy_l1_error for row in group])
        acceptance = _optional_mean_se([row.acceptance_rate for row in group])
        rows.append(
            BenchmarkAggregate(
                method=method,
                target=target,
                n_replicates=len(group),
                exact_after_freeze=all(row.exact_after_freeze for row in group),
                output_semantics=group[0].output_semantics,
                standardized_mean_error_mean=mean_error[0],
                standardized_mean_error_se=mean_error[1],
                relative_covariance_error_mean=covariance_error[0],
                relative_covariance_error_se=covariance_error[1],
                imq_mmd_mean=mmd[0],
                imq_mmd_se=mmd[1],
                mode_occupancy_l1_error_mean=occupancy[0],
                mode_occupancy_l1_error_se=occupancy[1],
                acceptance_rate_mean=acceptance[0],
                acceptance_rate_se=acceptance[1],
                training_seconds_mean=float(np.mean([row.training_seconds for row in group])),
                evaluation_seconds_mean=float(np.mean([row.evaluation_seconds for row in group])),
            )
        )
    return tuple(rows)


def pareto_frontier(
    aggregates: tuple[BenchmarkAggregate, ...],
    *,
    target: str,
    exact_after_freeze: bool | None = None,
    exclude_methods: tuple[str, ...] = (),
) -> tuple[BenchmarkAggregate, ...]:
    """Return methods nondominated in IMQ-MMD and total training-plus-evaluation time."""

    candidates = [
        row
        for row in aggregates
        if row.target == target
        and row.method not in exclude_methods
        and (exact_after_freeze is None or row.exact_after_freeze is exact_after_freeze)
        and row.imq_mmd_mean is not None
        and np.isfinite(row.imq_mmd_mean)
        and np.isfinite(row.evaluation_seconds_mean)
    ]
    frontier: list[BenchmarkAggregate] = []
    for row in candidates:
        row_mmd = row.imq_mmd_mean
        assert row_mmd is not None
        dominated = False
        for other in candidates:
            if other is row:
                continue
            other_mmd = other.imq_mmd_mean
            if other_mmd is None:
                continue
            no_worse = other_mmd <= row_mmd and other.total_seconds_mean <= row.total_seconds_mean
            strictly_better = (
                other_mmd < row_mmd or other.total_seconds_mean < row.total_seconds_mean
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return tuple(sorted(frontier, key=lambda item: item.total_seconds_mean))
