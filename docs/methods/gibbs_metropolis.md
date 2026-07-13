# Gibbs and Metropolis--Hastings

!!! tip "Visual tutorial"
    Work through [Metropolis, Gibbs, and Ising](../notebooks/05_metropolis_gibbs_and_ising.ipynb)
    for proposal tuning, jump-chain failure, exact small-lattice checks, and metastability.

Phase 5 adds the first general simulation-oriented MCMC layer. The finite-state
operator package remains the place for exact transition matrices and Poisson
calculations; `sampler_lab.mcmc` is the place for actually advancing array-valued
states.

## Metropolis--Hastings convention

For a current state `x`, a proposal object draws

\[
Y \sim q(\cdot\mid x).
\]

The implementation accepts with probability

\[
\alpha(x,y)=1\wedge
\frac{\pi(y)q(x\mid y)}{\pi(x)q(y\mid x)}.
\]

All four logarithmic terms are evaluated explicitly. This is intentionally less
magical than APIs that silently assume symmetry: a state-dependent or independence
proposal is wrong if the reverse proposal term is omitted. A sampled proposal must
have finite forward density. A proposal outside target support, or one from which the
current state cannot be proposed in reverse, is rejected.

A rejection still creates the next Markov-chain state:

\[
X_{k+1}=X_k.
\]

`run_chain` therefore stores repeated states. Deleting rejected states would produce
the jump chain, which generally has a different invariant distribution.

Implemented proposal objects include:

- diagonal Gaussian random walks;
- diagonal Gaussian independence proposals;
- state-dependent diagonal Gaussian proposals;
- one-coordinate Gaussian random walks with explicit coordinate-mixture density.

## Gibbs schedules

A `ConditionalUpdate` represents an exact draw of one block conditional on its
complement. The same block abstraction supports scalar, vector, checkerboard, or
other problem-specific updates.

The package supplies:

- `BlockGibbsKernel` for one fixed block;
- `RandomScanGibbsKernel` for an independent random block choice;
- `DeterministicSweepGibbsKernel` for an ordered composition of blocks;
- `TransformedGibbsKernel` for conjugating a Gibbs kernel through invertible
  coordinates.

A deterministic sweep preserves the target when each component update does, but the
composed sweep is generally **not reversible**. The Ising exact-matrix tests make this
visible rather than treating “Gibbs” and “reversible” as synonyms.

## Empirical autocorrelation diagnostics

`sampler_lab.diagnostics.time_series` computes autocovariances by FFT and estimates
integrated autocorrelation time with Geyer's initial-positive paired sequence. For a
recorded scalar series of length `N`, the reported effective sample size is

\[
\operatorname{ESS}=\frac{N}{\widehat\tau_{\mathrm{int}}}.
\]

The estimator is useful, not omniscient. Near a phase transition, IAT estimates can
require trajectories far longer than the apparent relaxation time. The benchmark
therefore reports cost-normalized ESS but does not pretend a single finite run proves
convergence.

## Periodic Ising model

For spins \(s_i\in\{-1,+1\}\) on an \(L\times L\) periodic square lattice,

\[
\pi(s)\propto
\exp\left[
\beta\left(J\sum_{\langle i,j\rangle}s_i s_j+h\sum_i s_i\right)
\right].
\]

The conditional probability for one spin is

\[
\Pr(s_i=+1\mid s_{-i})
=
\frac{1}{1+\exp[-2\beta(J\sum_{j\sim i}s_j+h)]}.
\]

Flipping one spin has log target ratio

\[
\log\frac{\pi(s^{(i)})}{\pi(s)}
=-2\beta s_i\left(J\sum_{j\sim i}s_j+h\right).
\]

The Metropolis Ising kernel uses this local expression, so one update needs four
neighbor reads rather than an \(O(L^2)\) global density evaluation.

For small lattices the package enumerates all \(2^{L^2}\) configurations and builds
exact transition matrices for:

- random-scan single-site Gibbs;
- one lexicographic deterministic Gibbs sweep;
- random-scan single-spin Metropolis.

These matrices validate global balance, detailed balance where applicable, and the
simulation kernels' local formulas.

## Benchmark

```bash
sampler-lab-ising-demo --sizes 6 --betas 0.3 0.44 0.6 \
  --sweeps 4000 --burn-in 1000
```

The command compares random-scan Gibbs, deterministic-sweep Gibbs, and single-spin
Metropolis using one recorded state per lattice sweep. It reports acceptance,
magnetization, energy per site, magnetization IAT, ESS per spin update, and ESS per
second. For lattices with at most 16 sites it also reports the exactly enumerated
mean absolute magnetization.

## Failure modes kept explicit

- Proposal asymmetry must be accounted for in both directions.
- The initial MH state must lie in target support.
- Rejections must remain in the trajectory.
- A deterministic scan is a composition kernel, not the same kernel as random scan.
- Exact Ising enumeration is exponentially expensive and guarded by a site limit.
- A constant observed time series has undefined empirical autocorrelation; the
  benchmark reports zero ESS rather than manufacturing a reassuring number.
