# Sampler Lab

[![CI](https://github.com/TylerPerlman/sampler-lab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/TylerPerlman/sampler-lab/actions/workflows/ci.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Sampler Lab is a from-scratch Monte Carlo methods library and experiment suite for transparent,
reproducible sampling research.

The project favors explicit mathematical structure over convenience wrappers:

- NumPy is the only runtime dependency.
- Every stochastic API receives an explicit `numpy.random.Generator`.
- Log densities and log weights are the default numerical representation.
- Algorithms return samples and diagnostics rather than hiding decisions.
- Exact, corrected, weighted, and approximate outputs remain distinct.
- Analytical identities, invariance checks, and fixed-seed tests support numerical claims.

## Implemented methods

Sampler Lab currently includes:

- IID estimators, empirical error metrics, inverse transforms, Box–Muller sampling, rejection
  sampling, and change-of-variable constructions;
- standard and self-normalized importance sampling, normalization-ratio estimation, weighted
  diagnostics, rare-tail proposals, and high-dimensional weight-collapse studies;
- sequential importance sampling, multinomial/systematic/variable-population resampling,
  ancestry tracking, and self-avoiding-walk growth;
- finite-state operators, invariant laws, reversibility, time reversal, Poisson equations,
  asymptotic variance, integrated autocorrelation time, and spectral diagnostics;
- generic Metropolis–Hastings, random-walk and independence proposals, Gibbs kernels, and exact
  small-lattice Ising comparisons;
- annealed importance sampling, Jarzynski estimators, annealed SMC, path reweighting, and exact
  finite-model normalization checks;
- Euler–Maruyama simulation, ULA, MALA, position-dependent preconditioning, generator diagnostics,
  and exact Gaussian stability calculations;
- Hamiltonian dynamics, leapfrog integration, HMC, persistent momentum, involutive corrections,
  and underdamped Langevin splitting;
- affine transformations, Gaussian conditioning, Hessian repair, stochastic Newton proposals,
  affine-invariant ensembles, and Rosenbrock geometry benchmarks;
- adaptive warmup, policy-gradient proposal selection, variational proposals, Stein discrepancy,
  and SVGD with explicit exactness labels;
- small-noise Gaussian rare-event oracles, Laplace approximations, exponential twisting,
  tempering, and multiple-dominating-point mixtures;
- a capability-aware cross-method benchmark with exact-reference targets, explicit exclusions,
  replicated reports, operation counts, and optional figures.

## Install

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
python tools/publication_check.py --root .
ruff check .
ruff format --check .
mypy src/sampler_lab
pytest -m "not statistical"
mkdocs build --strict
```

## Quick start

```python
import numpy as np

from sampler_lab.models import sample_self_avoiding_walks

result = sample_self_avoiding_walks(
    np.random.default_rng(2022),
    n_steps=10,
    n_particles=20_000,
    resampling="systematic",
    resample_ess_fraction=0.8,
)

print(result.normalizing_constant_estimate)
print(result.ess_history)
print(result.ancestry.unique_ancestor_counts())
```

The result retains the weighted cloud before every resampling step, so uniform post-resampling
weights cannot erase the evidence that triggered resampling.

## Reproducible demonstrations

```bash
sampler-lab-disk-benchmark --samples 100000 --repeats 5
sampler-lab-importance-demo --threshold 4 --samples 100000
sampler-lab-particle-demo --steps 10 --particles 20000
sampler-lab-markov-demo --states 12 --samples 240 --replicates 2000
sampler-lab-ising-demo --sizes 6 --betas 0.3 0.44 0.6 --sweeps 4000 --burn-in 1000
sampler-lab-annealing-demo --size 2 --target-beta 0.6 --path-steps 2 4 8 16 32
sampler-lab-langevin-demo --condition-numbers 1 10 100 --samples 20000
sampler-lab-hamiltonian-demo --condition-numbers 1 10 100 --samples 5000
sampler-lab-geometry-demo --condition-numbers 1 10 100 --samples 3000 --walkers 24
sampler-lab-policy-demo --samples 3000 --warmup 1000 --policy-updates 80
sampler-lab-rare-event-demo --epsilons 0.5 0.25 0.125 0.0625 --samples 100000
sampler-lab-benchmark --quick --replicates 2 --figures --output-dir benchmark_report
```

The demonstrations report method-appropriate quantities such as acceptance, weighted ESS,
genealogical collapse, autocorrelation time, exact moment error, energy error, relative variance,
and evaluation cost. The benchmark does not invent a universal winner score across incompatible
output types.

## Documentation

- [Development roadmap](docs/roadmap.md)
- [Public references and provenance](docs/references.md)
- [Public API and extension policy](docs/public_api.md)
- [Benchmark semantics](docs/benchmarking.md)
- [Finite-state Markov theory](docs/methods/markov_theory.md)
- [Gibbs, Metropolis, and Ising](docs/methods/gibbs_metropolis.md)
- [Annealed paths](docs/methods/annealed_paths.md)
- [Langevin dynamics](docs/methods/langevin_dynamics.md)
- [Hamiltonian dynamics](docs/methods/hamiltonian_dynamics.md)
- [Conditioning and ensembles](docs/methods/conditioning_ensembles.md)
- [Adaptive and policy sampling](docs/methods/adaptive_policy_sampling.md)
- [Rare events](docs/methods/rare_events.md)
- [Reference benchmark bundle](docs/reference/continuous_benchmark/README.md)

## Design status

Version 0.12.0 is a feature-complete beta. Package families, result semantics, command-line entry
points, and benchmark outputs are documented and regression-tested. Names may still change before
1.0, but RNG behavior, rejection states, weight normalization, exactness labels, and output
semantics are compatibility-sensitive.
