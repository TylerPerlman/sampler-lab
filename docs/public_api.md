# Public API and Stability Policy

Sampler Lab is a teaching and research codebase whose public surface is the package-level exports,
validated dataclasses, protocols, and command-line entry points documented here. Internal helper
functions beginning with `_` are not stable.

Version `0.12.0` is the first feature-complete beta. Backward-incompatible public changes before
`1.0` must be documented in `CHANGELOG.md`, accompanied by migration notes, and tested at the
package-export level.

## Core conventions

- Pass an explicit `numpy.random.Generator` to every stochastic operation.
- Represent target densities and importance weights in log space.
- Treat states, results, policies, and particle histories as immutable value objects where
  practical.
- Return diagnostics and raw samples rather than hiding rejection, resampling, adaptation, or
  correction steps.
- Keep training/warmup separate from frozen evaluation.
- Use `OperationCounter` for comparable algorithmic work.

## Stable package families

| Package | Stable responsibility |
|---|---|
| `sampler_lab.core` | RNG splitting, protocols, numerics, linear algebra, counters, common results |
| `sampler_lab.estimators` | IID and weighted estimators, normalization, empirical error metrics |
| `sampler_lab.exact` | inversion, Box--Muller, transformations, rejection sampling |
| `sampler_lab.importance` | log weights, IS/SNIS, ratio estimates, divergences, collapse diagnostics |
| `sampler_lab.particles` | particle clouds, resampling, SIS/SMC histories, ancestry |
| `sampler_lab.markov` | finite-state operators, invariant laws, Poisson equations, exact asymptotics |
| `sampler_lab.mcmc` | generic MH, proposals, chain runners, Gibbs schedules |
| `sampler_lab.annealing` | annealed paths, AIS/Jarzynski, annealed SMC, free energies |
| `sampler_lab.dynamics` | diffusion generators, ULA/MALA, Hamiltonian and underdamped methods |
| `sampler_lab.geometry` | affine maps, preconditioners, Hessian repair, stochastic Newton |
| `sampler_lab.ensemble` | complete-ensemble states, stretch and walk moves, ensemble diagnostics |
| `sampler_lab.adaptive` | running moments, schedules, dual averaging, warmup windows |
| `sampler_lab.learning` | proposal policies, REINFORCE, objectives, variational fitting, Stein methods |
| `sampler_lab.rare_events` | small-noise problems, Laplace asymptotics, twisting, tempering, mixtures |
| `sampler_lab.benchmarks` | capabilities, exact-reference targets, adapters, metrics, reports, figures |
| `sampler_lab.models` | transparent reference targets and model-specific laboratories |

Package `__init__.py` files define the intended import surface. Deep imports remain available for
study, but callers should prefer package exports when one exists.

## Protocols

The primary extensibility points are structural protocols rather than inheritance trees:

- log density, differentiable log density, and twice-differentiable log density;
- proposal and transition-kernel interfaces;
- population and ensemble transitions;
- continuous benchmark adapters.

A new method should implement the smallest relevant protocol, receive its RNG explicitly, and
return the repository result type rather than inventing an opaque wrapper.

## Exactness labels

The library distinguishes:

- exact direct or rejection sampling;
- exact invariant kernels after Metropolis correction or frozen adaptation;
- unbiased weighted estimators;
- approximate variational or deterministic-particle outputs.

“Exact” never implies well mixed. “Unbiased” never implies usable variance. Result objects and
benchmark reports preserve these distinctions.

## Command-line entry points

The supported educational CLIs are:

```text
sampler-lab-disk-benchmark
sampler-lab-importance-demo
sampler-lab-particle-demo
sampler-lab-markov-demo
sampler-lab-ising-demo
sampler-lab-annealing-demo
sampler-lab-langevin-demo
sampler-lab-hamiltonian-demo
sampler-lab-geometry-demo
sampler-lab-policy-demo
sampler-lab-rare-event-demo
sampler-lab-benchmark
```

All commands expose deterministic seeds and machine-readable output where the experiment is large
enough to justify downstream analysis.

## Deprecation policy

Before `1.0`, a renamed public symbol should normally remain as a documented compatibility alias
for at least one minor release. Behavioral changes to RNG consumption, weight normalization,
rejection-state handling, or exactness semantics require a regression test and changelog entry even
when the Python signature is unchanged.
