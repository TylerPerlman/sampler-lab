# Sampler Lab

Sampler Lab is a from-scratch Monte Carlo methods library and experiment suite. The project
emphasizes transparent algorithms, explicit random-number state, stable log-domain calculations,
and mathematical validation.

## What is implemented

- exact transforms and rejection sampling;
- importance sampling and rare-event twisting;
- sequential importance sampling, resampling, and particle ancestry;
- finite-state Markov theory and exact asymptotic variance;
- Metropolis--Hastings, Gibbs, Ising, Langevin, HMC, and underdamped methods;
- annealed importance sampling and annealed SMC;
- conditioning, stochastic Newton, and affine-invariant ensembles;
- adaptive MCMC and policy-gradient proposal learning;
- variational and Stein methods with explicit exact-versus-approximate labels;
- a capability-aware cross-method benchmark on Gaussian mixtures and funnel targets.

See the [development roadmap](roadmap.md), [public API policy](public_api.md), and
[public references](references.md).

## Design principles

1. Every stochastic API receives an explicit `numpy.random.Generator`.
2. Log densities and log weights are the default numerical representation.
3. Rejections, resampling, adaptation, and training costs remain visible in returned diagnostics.
4. Exact, corrected, weighted, and approximate outputs are never conflated.
5. Deterministic identities and invariance tests precede statistical regression tests.
6. Benchmarks report capabilities and exclusions rather than inventing a universal winner score.

## Install

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
mypy src/sampler_lab
pytest -m "not statistical"
mkdocs build --strict
```

## Start exploring

- [Finite-state Markov theory](methods/markov_theory.md)
- [Gibbs and Metropolis methods](methods/gibbs_metropolis.md)
- [Langevin dynamics](methods/langevin_dynamics.md)
- [Hamiltonian dynamics](methods/hamiltonian_dynamics.md)
- [Adaptive and policy-gradient sampling](methods/adaptive_policy_sampling.md)
- [Rare-event methods](methods/rare_events.md)
- [Cross-method benchmark methodology](benchmarking.md)
- [Reference benchmark summary](reference/continuous_benchmark/benchmark_summary.md)

## Repository health

GitHub Actions validates formatting, linting, strict typing, compilation, package installation,
console entry points, unit tests, isolated statistical tests, documentation, publication hygiene,
CodeQL analysis, and release builds. Dependabot monitors Python and GitHub Actions dependencies.
