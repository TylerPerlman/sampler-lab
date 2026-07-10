"""JSON, CSV, Markdown, and figure-ready benchmark reports."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sampler_lab.benchmarks.runner import BenchmarkSuiteResult, pareto_frontier


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), nested, output)
        return
    output[prefix] = value


def result_rows(suite: BenchmarkSuiteResult) -> tuple[dict[str, Any], ...]:
    """Flatten per-replicate results into stable CSV-compatible rows."""

    rows: list[dict[str, Any]] = []
    for result in suite.results:
        flattened: dict[str, Any] = {}
        _flatten("", asdict(result), flattened)
        rows.append(flattened)
    return tuple(rows)


def pairing_rows(suite: BenchmarkSuiteResult) -> tuple[dict[str, Any], ...]:
    """Return one status row per represented method/target pairing."""

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for result in suite.results:
        key = (result.method, result.target)
        row = rows.setdefault(
            key,
            {
                "method": result.method,
                "target": result.target,
                "status": "success",
                "exact_after_freeze": result.exact_after_freeze,
                "output_semantics": result.output_semantics,
                "reasons": "",
                "successful_replicates": 0,
            },
        )
        row["successful_replicates"] = int(row["successful_replicates"]) + 1
    for exclusion in suite.exclusions:
        rows[(exclusion.method, exclusion.target)] = {
            "method": exclusion.method,
            "target": exclusion.target,
            "status": "excluded",
            "exact_after_freeze": "",
            "output_semantics": "",
            "reasons": "; ".join(exclusion.reasons),
            "successful_replicates": 0,
        }
    for failure in suite.failures:
        key = (failure.method, failure.target)
        if key in rows and rows[key]["status"] == "success":
            rows[key]["status"] = "partial failure"
            rows[key]["reasons"] = f"{failure.error_type}: {failure.message}"
        else:
            rows[key] = {
                "method": failure.method,
                "target": failure.target,
                "status": "failure",
                "exact_after_freeze": "",
                "output_semantics": "",
                "reasons": f"{failure.error_type}: {failure.message}",
                "successful_replicates": 0,
            }
    return tuple(rows[key] for key in sorted(rows))


def write_pairing_report(suite: BenchmarkSuiteResult, path: str | Path) -> Path:
    """Write the explicit success/exclusion/failure matrix as CSV."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = pairing_rows(suite)
    fieldnames = (
        "method",
        "target",
        "status",
        "exact_after_freeze",
        "output_semantics",
        "successful_replicates",
        "reasons",
    )
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def write_json_report(suite: BenchmarkSuiteResult, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(suite.to_json() + "\n", encoding="utf-8")
    return destination


def write_csv_report(suite: BenchmarkSuiteResult, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = result_rows(suite)
    fieldnames = sorted({key for row in rows for key in row})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _format_optional(value: float | None) -> str:
    return "--" if value is None else f"{value:.4g}"


def markdown_summary(suite: BenchmarkSuiteResult) -> str:
    """Render a compact human-readable summary without hiding exclusions."""

    lines = [
        "# Sampler Lab Continuous Benchmark",
        "",
        f"- Package version: `{suite.package_version}`",
        f"- Replicates: {suite.n_replicates}",
        f"- Base seed: {suite.seed}",
        f"- Successful runs: {len(suite.results)}",
        f"- Excluded pairings: {len(suite.exclusions)}",
        f"- Failed runs: {len(suite.failures)}",
        "",
        "## Aggregate metrics",
        "",
        (
            "| Target | Method | Semantics | Exact after freeze | Mean error | "
            "Covariance error | IMQ-MMD | Mode L1 | Train seconds | Eval seconds | Total seconds |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in suite.aggregates:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.target,
                    row.method,
                    row.output_semantics,
                    str(row.exact_after_freeze),
                    f"{row.standardized_mean_error_mean:.4g}",
                    f"{row.relative_covariance_error_mean:.4g}",
                    _format_optional(row.imq_mmd_mean),
                    _format_optional(row.mode_occupancy_l1_error_mean),
                    f"{row.training_seconds_mean:.4g}",
                    f"{row.evaluation_seconds_mean:.4g}",
                    f"{row.total_seconds_mean:.4g}",
                )
            )
            + " |"
        )
    targets = sorted({row.target for row in suite.aggregates})
    lines.extend(("", "## Accuracy-total-time Pareto frontiers", ""))
    lines.append(
        "The direct oracle is a reference baseline and is excluded from algorithmic frontiers."
    )
    lines.append("")
    for target in targets:
        exact_frontier = pareto_frontier(
            suite.aggregates,
            target=target,
            exact_after_freeze=True,
            exclude_methods=("direct-oracle",),
        )
        approximate_frontier = pareto_frontier(
            suite.aggregates,
            target=target,
            exact_after_freeze=False,
        )
        exact_methods = ", ".join(row.method for row in exact_frontier) or "none"
        approximate_methods = ", ".join(row.method for row in approximate_frontier) or "none"
        lines.append(f"- **{target} — exact/corrected:** {exact_methods}")
        lines.append(f"- **{target} — approximate:** {approximate_methods}")
    if suite.exclusions:
        lines.extend(("", "## Explicit exclusions", ""))
        for exclusion in suite.exclusions:
            lines.append(
                f"- `{exclusion.method}` x `{exclusion.target}`: " + "; ".join(exclusion.reasons)
            )
    if suite.failures:
        lines.extend(("", "## Runtime failures", ""))
        for failure in suite.failures:
            lines.append(
                f"- `{failure.method}` x `{failure.target}` replicate {failure.replicate}: "
                f"{failure.error_type}: {failure.message}"
            )
    lines.append("")
    return "\n".join(lines)


def write_markdown_report(suite: BenchmarkSuiteResult, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown_summary(suite), encoding="utf-8")
    return destination


def write_manifest(
    suite: BenchmarkSuiteResult,
    directory: str | Path,
    paths: tuple[Path, ...],
) -> Path:
    """Write a manifest covering all report and optional figure artifacts."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "package_version": suite.package_version,
        "seed": suite.seed,
        "replicates": suite.n_replicates,
        "files": [str(path.relative_to(root)) for path in paths if path.name != "manifest.json"],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def write_report_bundle(suite: BenchmarkSuiteResult, directory: str | Path) -> tuple[Path, ...]:
    """Write the canonical machine- and human-readable report bundle."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    paths = (
        write_json_report(suite, root / "benchmark_results.json"),
        write_csv_report(suite, root / "benchmark_results.csv"),
        write_markdown_report(suite, root / "benchmark_summary.md"),
        write_pairing_report(suite, root / "benchmark_pairings.csv"),
    )
    return (*paths, write_manifest(suite, root, paths))
