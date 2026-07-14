# Capability-Aware Continuous Benchmarking

!!! tip "Visual tutorial"
    Work through [Cross-method benchmarking](notebooks/12_cross_method_benchmarking.ipynb) for capability
    checks, output semantics, replicated metrics, report bundles, and cost–accuracy Pareto frontiers.

The common benchmark compares continuous samplers without pretending that all outputs have the
same semantics. Independent draws, weighted particles, stationary chains, ensembles, and
variational particle clouds can share target-level accuracy metrics, but they do not share every
mixing diagnostic or exactness guarantee.

## Command line

List available targets and methods:

```bash
sampler-lab-benchmark --list
```

Run the complete replicated suite:

```bash
sampler-lab-benchmark \
  --samples 2000 \
  --warmup 500 \
  --reference-samples 1000 \
  --replicates 3 \
  --output-dir benchmark_report \
  --figures
```

Use `--targets` and `--methods` with comma-separated names to select a subset. `--quick` reduces
all internal budgets for a smoke run. `--json` prints the complete structured result, and
`--fail-fast` converts a recorded runtime failure into an immediate exception.

## Reference targets

The default suite forms a diagnostic ladder:

1. **Correlated Gaussian:** exact linear-algebra oracle for covariance and conditioning.
2. **Separated anisotropic Gaussian mixture:** exposes mode trapping and reverse-KL collapse.
3. **Rotated anisotropic funnel:** exposes position-dependent geometry and neck/tail failures.
4. **Bimodal anisotropic funnel:** combines global mode discovery with severe local curvature.

Every target has a normalized density, exact direct sampler, analytic gradient and Hessian, and
known first two moments. Mixture targets also expose exact mode labels and probabilities.

## Output semantics and exactness

Every adapter returns an explicit `output_semantics` label. Current labels include:

- `iid-samples`;
- `markov-chain`;
- `ensemble-chain`;
- `weighted-samples`;
- `weighted-particles`;
- `variational-samples`;
- `deterministic-particles`.

`exact_after_freeze` means that, after all warmup or learning has stopped, the evaluated method
preserves or exactly corrects to the requested target. It does **not** mean independent samples,
rapid convergence, or good finite-budget mode coverage. Reverse-KL and SVGD outputs are marked
approximate. A frozen variational proposal followed by full independence Metropolis correction is
marked exact.

## Capability checks

Sampler adapters declare density, gradient, Hessian, conditional, direct-sample, multimodality,
and state-space requirements. Targets declare the access they provide. Incompatible pairings are
written to the report with all exclusion reasons rather than omitted or assigned fabricated
scores.

Model-specific discrete methods, unit-disk samplers, and self-avoiding-walk estimators therefore
remain in their own exact laboratories. “Not applicable” is a result; forcing every method into one
leaderboard would be a category error with nicer typography.

## Metrics

All representations are compared against one exact reference sample per target and replicate.
Common distributional diagnostics include:

- standardized mean error;
- relative covariance error;
- inverse-multiquadric reference-sample MMD;
- mode-occupancy error when exact labels exist;
- funnel-neck, funnel-mouth, and latent-scale errors.

Stationary chains and ensemble trajectories additionally receive acceptance and mode-residence,
first-passage, switching, and round-trip diagnostics where meaningful. Weighted outputs receive
weight ESS and maximum normalized weight. Deterministic particle methods do not receive fictional
chain IATs.

The report deliberately has no overall winner score. The Pareto summary uses IMQ-MMD and total training-plus-evaluation
wall time as a compact view; the detailed rows retain target access, output semantics,
exactness, training cost, evaluation cost, and target-specific failures.

## Cost accounting

Training and frozen evaluation use separate `OperationCounter` objects and wall clocks. Reported
operations include log-density, proposal-density, gradient, Hessian, matrix-factorization, policy,
and random-draw costs when the implementation can count them directly. Counts are algorithmic
work proxies, not claims of equal hardware cost.

Warmup, variational fitting, and policy learning cannot quietly disappear into “free” samples.
Amortized comparisons can be derived from the JSON report for any declared number of downstream
evaluation draws.

## Reproducibility

A base seed is deterministically split by target, method, replicate, and purpose. Methods in one
replicate share the same exact reference sample but receive disjoint stochastic streams. Training
and frozen evaluation streams are also disjoint inside learned adapters.

The canonical report bundle contains:

- `benchmark_results.json`: complete nested results, aggregates, exclusions, and failures;
- `benchmark_results.csv`: flattened per-replicate rows;
- `benchmark_summary.md`: compact aggregate table and Pareto frontiers;
- `manifest.json`: package version, seed, replicate count, and filenames;
- optional reference figures.

The full benchmark is an opt-in experiment. CI checks deterministic identities and small fixed-seed
adapter runs; it does not require a stochastic total ordering between methods.

## Python API

```python
from sampler_lab.benchmarks import BenchmarkConfig, run_benchmark_suite, write_report_bundle

suite = run_benchmark_suite(
    config=BenchmarkConfig(n_samples=1000, warmup_steps=250, reference_samples=1000),
    n_replicates=3,
    seed=2022,
    target_names=("bimodal-anisotropic-funnel",),
)
write_report_bundle(suite, "benchmark_report")
```

Custom methods should implement the `ContinuousSamplerAdapter` protocol and return a validated
`SamplerOutput`. Custom targets should use `ContinuousTargetCase` and declare their capabilities.
