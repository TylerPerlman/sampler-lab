# Performance Expectations

Sampler Lab optimizes first for inspectability, reproducibility, and mathematically explicit state.
Many Markov kernels therefore advance one state at a time in Python. That is deliberate: proposal
terms, rejected states, operation counts, adaptation boundaries, and diagnostics remain visible.
The tradeoff is lower throughput than vectorized, compiled, JAX, or GPU-oriented libraries.

## Illustrative throughput

The committed quick benchmark uses 300 evaluation outputs per method and two fixed-seed replicates.
On the environment that generated the reference artifact, the correlated-Gaussian case reported:

| Method | Mean evaluation time | Approximate outputs per second |
|---|---:|---:|
| Importance sampling | 0.001897 s | 158,000 |
| Random-walk Metropolis | 0.01806 s | 16,600 |
| MALA | 0.03010 s | 10,000 |
| HMC | 0.07698 s | 3,900 |
| Stochastic Newton | 0.08287 s | 3,600 |

These figures are orientation, not promises. The generating hardware was not recorded, targets have
different evaluation costs, and one output from one method is not equivalent to one output from
another. Prefer operation counts, ESS, error, and wall time together. Re-run
`sampler-lab-benchmark` on the hardware and target that matter to you.

## Intended use

Sampler Lab is well suited to:

- learning and auditing algorithm mechanics;
- validating identities and invariance properties;
- constructing reproducible small-to-medium experiments;
- comparing method semantics and failure modes;
- serving as a clear reference implementation before optimization.

It is not intended to beat specialized production samplers on raw throughput. Vectorized
`(n_chains, dimension)` kernels and compiled backends are reasonable future extensions, but they
must preserve explicit RNG ownership, rejection semantics, counters, and diagnostics rather than
silently changing the contract.
