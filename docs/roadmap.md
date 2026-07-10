# Development Roadmap

Sampler Lab is organized around reusable mathematical components rather than one-off scripts.
The current beta implements a broad set of exact, weighted, particle, Markov, adaptive, and
variational methods. Future work should strengthen reliability and interoperability before adding
large new algorithm families.

## Near-term priorities

1. Stabilize public protocols and result objects through downstream use.
2. Expand exact-reference targets and invariant-distribution tests.
3. Add moderate-budget benchmark bundles with reproducible manifests.
4. Improve documentation for numerical failure modes and cost accounting.
5. Add optional samplers only when their output semantics fit the existing benchmark model.

## Candidate extensions

- replica exchange and population tempering;
- dynamic Hamiltonian path-length selection;
- nonlinear dominating-point optimization for rare events;
- richer normalizing-flow proposals behind optional dependencies;
- parallel execution for replicated benchmarks.

## Definition of done for a new method

A new method should include a typed implementation, explicit random-number state, deterministic or
exact algebraic tests, statistical validation when necessary, operation accounting, documented
failure modes, and a benchmark adapter only when comparisons preserve output semantics.

## Release direction

The 0.x series remains free to refine names and protocols. A 1.0 review should occur after real
external use identifies which interfaces deserve long-term stability.
