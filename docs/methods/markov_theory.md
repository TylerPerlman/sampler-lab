# Finite-State Markov Theory

This phase treats a finite Markov chain as an exact linear operator rather than
starting with a simulated histogram. That gives stronger validation: invariance,
reversibility, periodicity, Poisson equations, and Monte Carlo error can all be
computed to numerical precision.

## Conventions

A transition matrix `P` is row stochastic. Observables are column vectors and
probability measures are row vectors:

\[
(Pf)_i = \sum_j P_{ij}f_j,
\qquad
(\mu P)_j = \sum_i \mu_iP_{ij}.
\]

The package checks the duality identity

\[
(\mu P)f = \mu(Pf).
\]

The package defines the discrete generator as

\[
L=P-I.
\]

This sign convention matters. Many references instead call \(I-P\) the
Poisson operator.

## Invariant measures and communicating classes

A probability vector \(\pi\) is invariant when

\[
\pi P=\pi.
\]

`FiniteStateMarkovChain.invariant_distributions()` finds one extreme invariant
law for each closed communicating class. Every invariant probability law is a
convex combination of these rows. Transient states receive zero invariant mass.
The singular method `invariant_distribution()` deliberately fails when that law
is not unique.

Irreducibility and periodicity are computed from the directed support graph of
`P`, not inferred from eigenvalues rounded near the unit circle. For a finite
chain, the repository uses "ergodic" in the standard irreducible-and-aperiodic
sense.

## Global balance, detailed balance, and time reversal

Global balance only requires total stationary flux into each state to equal the
flux out:

\[
\pi P=\pi.
\]

Detailed balance is the stronger pairwise condition

\[
\pi_iP_{ij}=\pi_jP_{ji}.
\]

The stationary time reversal is

\[
P^*_{ij}=\frac{\pi_jP_{ji}}{\pi_i}.
\]

A chain is reversible exactly when \(P=P^*\). The implementation requires full
stationary support before constructing `P*`; transitions out of zero-mass states
are not identified by stationarity.

## Poisson equation

For an observable \(f\), write

\[
g=f-\pi[f].
\]

The centered Poisson equation is

\[
Lu=g,
\qquad
\pi[u]=0.
\]

The additive constant is fixed by centering. Numerically, the package solves the
nonsingular fundamental system

\[
(I-P+\mathbf 1\pi)h=g,
\qquad
u=-h.
\]

The result reports both the equation residual and the centering residual.

## Martingale decomposition

Given a path \(X_0,\ldots,X_n\), define

\[
\Delta M_k=u(X_{k+1})-(Pu)(X_k).
\]

Then the centered additive functional satisfies the exact identity

\[
\sum_{k=0}^{n-1}g(X_k)
=u(X_n)-u(X_0)-\sum_{k=0}^{n-1}\Delta M_k.
\]

`poisson_martingale_decomposition` returns every term and a floating-point
residual. This is an algebraic test, not a statistical one.

## Exact autocovariance and Monte Carlo error

Under stationarity,

\[
\gamma_k
=\operatorname{Cov}_\pi(f(X_0),f(X_k))
=\langle g,P^kg\rangle_\pi.
\]

The exact variance of an average of \(N\) observations is

\[
\operatorname{Var}\!\left(\frac1N\sum_{k=0}^{N-1}f(X_k)\right)
=\frac{1}{N^2}\left[N\gamma_0
+2\sum_{k=1}^{N-1}(N-k)\gamma_k\right].
\]

The asymptotic variance follows from the Poisson solution:

\[
\sigma_f^2
=-2\langle g,L^{-1}g\rangle_\pi-\operatorname{Var}_\pi(f).
\]

The integrated autocorrelation time is

\[
\tau_f=\frac{\sigma_f^2}{\operatorname{Var}_\pi(f)}.
\]

A constant observable has zero asymptotic variance but no meaningful IAT, so the
IAT routine rejects it explicitly.

The Poisson formula also handles periodic chains. For example, deterministic
two-state alternation has autocorrelations \(1,-1,1,-1,\ldots\), an ordinary
infinite sum that does not converge, but its time averages have asymptotic
variance zero.

## Spectral diagnostics

The package keeps three different quantities separate:

- **Poincare gap** for reversible chains:
  \(\gamma=1-\lambda_2\), where \(\lambda_2\) is the largest nonconstant
  eigenvalue.
- **Absolute spectral gap**:
  \(1-\max_{\lambda\ne1}|\lambda|\).
- **Singular-value gap** in \(L^2(\pi)\): one minus the second singular value of
  \(D_\pi^{1/2}PD_\pi^{-1/2}\).

They answer different questions. A periodic deterministic cycle has Poincare
behavior that can make some averages excellent while its absolute and singular
value gaps are zero because the distribution does not converge.

For a reversible finite chain, the worst centered-observable IAT is

\[
\tau_{\max}=\frac{1+\lambda_2}{1-\lambda_2}
=\frac{2}{\gamma}-1.
\]

## Partial resampling

For a finite joint probability table, `conditional_resampling_transition`
constructs the exact Gibbs update of one coordinate block. More generally,
`conditional_component_transition` lifts any family of conditional-invariant
local kernels to the full state space and verifies every conditional law before
building the matrix.

Random scan is represented by a mixture of coordinate kernels. A deterministic
sweep is their matrix product in execution order. Both preserve the target;
random scan with reversible coordinate kernels is reversible, while a fixed
sweep generally is not.

## Reference experiment

```bash
sampler-lab-markov-demo --states 12 --samples 240 --replicates 2000
```

The experiment compares three chains with the same uniform invariant law:

1. a lazy reversible random walk;
2. a lazy directed walk;
3. a deterministic cycle.

It reports exact IAT, asymptotic variance, finite-sample standard error,
replicated empirical standard error, and the distinct spectral gaps. The directed
chain demonstrates that detailed balance is sufficient, not necessary, and can
be substantially less efficient for a chosen observable.

## Limitations

The exact matrix routines are intended for small educational state spaces. They
use dense linear algebra and graph closure, so memory grows as \(O(n^2)\) and
some structural calculations grow as \(O(n^3)\). Sparse large-state MCMC belongs
in later simulation-oriented layers rather than being disguised as a dense
finite-state problem.
