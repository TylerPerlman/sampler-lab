"""Command-line entry point for replicated cross-method continuous benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from sampler_lab.benchmarks.adapters import (
    BenchmarkConfig,
    ContinuousSamplerAdapter,
    default_continuous_adapters,
)
from sampler_lab.benchmarks.continuous_suite import default_continuous_cases
from sampler_lab.benchmarks.figures import generate_reference_figures
from sampler_lab.benchmarks.reports import markdown_summary, write_manifest, write_report_bundle
from sampler_lab.benchmarks.runner import run_benchmark_suite


def _split_names(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise argparse.ArgumentTypeError("name lists may not be empty")
    return names


def cross_method_benchmark_main() -> None:
    """Run the Phase 12 capability-aware benchmark and write reproducible reports."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", help="comma-separated target names; default: all")
    parser.add_argument("--methods", help="comma-separated method names; default: all")
    parser.add_argument("--samples", type=int, default=2_000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--reference-samples", type=int, default=1_000)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_report"))
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--list", action="store_true", dest="list_names")
    args = parser.parse_args()

    cases = default_continuous_cases()
    adapters = default_continuous_adapters()
    if args.list_names:
        print("Targets:")
        for case in cases:
            print(f"  {case.name}")
        print("Methods:")
        for adapter in adapters:
            print(f"  {adapter.name}: {adapter.description}")
        return

    if args.quick:
        config = BenchmarkConfig(
            n_samples=min(args.samples, 300),
            warmup_steps=min(args.warmup, 80),
            reference_samples=min(args.reference_samples, 300),
            n_walkers=16,
            variational_steps=12,
            policy_updates=8,
            policy_rollout_length=4,
            svgd_particles=24,
            svgd_steps=4,
            annealing_particles=48,
            annealing_steps=5,
            seed=args.seed,
        )
    else:
        config = BenchmarkConfig(
            n_samples=args.samples,
            warmup_steps=args.warmup,
            reference_samples=args.reference_samples,
            seed=args.seed,
        )
    target_names = _split_names(args.targets)
    method_names = _split_names(args.methods)
    suite = run_benchmark_suite(
        config=config,
        n_replicates=args.replicates,
        seed=args.seed,
        cases=cases,
        adapters=cast(tuple[ContinuousSamplerAdapter, ...], adapters),
        target_names=target_names,
        method_names=method_names,
        fail_fast=args.fail_fast,
    )
    paths = write_report_bundle(suite, args.output_dir)
    if args.figures:
        paths = (
            *paths,
            *generate_reference_figures(suite, cases, args.output_dir / "figures", seed=args.seed),
        )
        write_manifest(suite, args.output_dir, paths)
    if args.as_json:
        print(suite.to_json())
    else:
        print(markdown_summary(suite))
        print("Report files:")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    cross_method_benchmark_main()
