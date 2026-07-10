"""Optional reference figures generated from benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sampler_lab.benchmarks.continuous_suite import ContinuousTargetCase
from sampler_lab.benchmarks.runner import BenchmarkSuiteResult


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError("reference figures require the 'dev' matplotlib extra") from error
    return plt


def _principal_projection(samples: np.ndarray) -> np.ndarray:
    centered = samples - np.mean(samples, axis=0)
    if samples.shape[1] == 1:
        return np.column_stack((centered[:, 0], np.zeros(samples.shape[0])))
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    return np.asarray(centered @ right[:2].T, dtype=np.float64)


def plot_target_reference(
    case: ContinuousTargetCase,
    path: str | Path,
    *,
    seed: int = 2022,
    n_samples: int = 2_000,
) -> Path:
    """Plot a deterministic two-principal-component target reference cloud."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    samples = case.reference_samples(np.random.default_rng(seed), n_samples)
    projected = _principal_projection(samples)
    plt = _pyplot()
    figure = plt.figure(figsize=(7.0, 5.0))
    axis = figure.add_subplot(1, 1, 1)
    axis.scatter(projected[:, 0], projected[:, 1], s=5, alpha=0.35)
    axis.set_title(case.name)
    axis.set_xlabel("principal coordinate 1")
    axis.set_ylabel("principal coordinate 2")
    axis.grid(alpha=0.2)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def plot_target_metric_panel(
    suite: BenchmarkSuiteResult,
    target: str,
    path: str | Path,
) -> Path:
    """Plot IMQ-MMD against evaluation wall time for one target."""

    rows = [
        row for row in suite.aggregates if row.target == target and row.imq_mmd_mean is not None
    ]
    if not rows:
        raise ValueError(f"no MMD results are available for target {target!r}")
    plt = _pyplot()
    figure = plt.figure(figsize=(7.5, 5.5))
    axis = figure.add_subplot(1, 1, 1)
    for row in rows:
        axis.scatter(row.total_seconds_mean, row.imq_mmd_mean, s=45)
        axis.annotate(
            row.method,
            (row.total_seconds_mean, row.imq_mmd_mean),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("training + evaluation seconds")
    axis.set_ylabel("IMQ-MMD")
    axis.set_title(target)
    axis.grid(alpha=0.2)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def plot_target_diagnostic_panel(
    suite: BenchmarkSuiteResult,
    target: str,
    path: str | Path,
) -> Path:
    """Plot the most diagnostic target-specific absolute error by method."""

    target_results = [result for result in suite.results if result.target == target]
    if not target_results:
        raise ValueError(f"no results are available for target {target!r}")
    methods = sorted({result.method for result in target_results})
    mode_target = any(
        result.distribution.mode_occupancy_l1_error is not None for result in target_results
    )
    funnel_target = any("funnel_neck_abs_error" in result.diagnostics for result in target_results)
    if mode_target:
        label = "mode occupancy L1 error"
        values = [
            float(
                np.mean(
                    [
                        result.distribution.mode_occupancy_l1_error
                        for result in target_results
                        if result.method == method
                        and result.distribution.mode_occupancy_l1_error is not None
                    ]
                )
            )
            for method in methods
        ]
    elif funnel_target:
        label = "funnel neck probability error"
        values = [
            float(
                np.mean(
                    [
                        result.diagnostics["funnel_neck_abs_error"]
                        for result in target_results
                        if result.method == method and "funnel_neck_abs_error" in result.diagnostics
                    ]
                )
            )
            for method in methods
        ]
    else:
        label = "standardized mean error"
        values = [
            float(
                np.mean(
                    [
                        result.distribution.standardized_mean_error
                        for result in target_results
                        if result.method == method
                    ]
                )
            )
            for method in methods
        ]
    plt = _pyplot()
    figure = plt.figure(figsize=(9.0, 5.5))
    axis = figure.add_subplot(1, 1, 1)
    positions = np.arange(len(methods))
    axis.bar(positions, values)
    axis.set_xticks(positions, methods, rotation=60, ha="right")
    axis.set_ylabel(label)
    axis.set_title(target)
    axis.grid(axis="y", alpha=0.2)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def generate_reference_figures(
    suite: BenchmarkSuiteResult,
    cases: tuple[ContinuousTargetCase, ...],
    directory: str | Path,
    *,
    seed: int = 2022,
) -> tuple[Path, ...]:
    """Generate one target reference and one performance panel per represented target."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    represented = {row.target for row in suite.aggregates}
    paths: list[Path] = []
    for index, case in enumerate(cases):
        if case.name not in represented:
            continue
        slug = case.name.replace("-", "_")
        paths.append(
            plot_target_reference(
                case,
                root / f"{slug}_reference.png",
                seed=seed + index,
            )
        )
        paths.append(
            plot_target_metric_panel(
                suite,
                case.name,
                root / f"{slug}_accuracy_time.png",
            )
        )
        paths.append(
            plot_target_diagnostic_panel(
                suite,
                case.name,
                root / f"{slug}_diagnostic.png",
            )
        )
    return tuple(paths)
