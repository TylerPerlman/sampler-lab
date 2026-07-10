# Annealed Paths, Jarzynski Weighting, and Free Energies

## Distribution path

Let \(\gamma_\beta(x)\) be an unnormalized density for
\(\beta\in[0,1]\), with

\[
Z_\beta = \int \gamma_\beta(x)\,dx,
\qquad
\pi_\beta(x)=\gamma_\beta(x)/Z_\beta.
\]

`AnnealingPath` stores only `log_unnormalized(x, beta)`. Normalizing constants
are deliberately absent because estimating their ratios is the point of the
algorithm.

The default geometric bridge is

\[
\gamma_\beta(x)
=\gamma_0(x)^{1-\beta}\gamma_1(x)^\beta,
\]

or, in log form,

\[
\log\gamma_\beta(x)
=(1-\beta)\log\gamma_0(x)+\beta\log\gamma_1(x).
\]

`AnnealingSchedule` contains a strictly increasing sequence

\[
0=\beta_0<\beta_1<\cdots<\beta_K=1.
\]

Linear and power schedules are built in; arbitrary validated values can be
supplied directly.

## Annealed importance sampling

Start with independent particles \(X_0^i\sim\pi_0\). At stage \(k\), before
moving the particles, compute

\[
\Delta \ell_k^i
=\log\gamma_{\beta_k}(X_{k-1}^i)
 -\log\gamma_{\beta_{k-1}}(X_{k-1}^i).
\]

Without resampling, the complete trajectory log weight is

\[
L^i=\sum_{k=1}^K \Delta\ell_k^i,
\]

and the normalization-ratio estimator is

\[
\widehat{Z_1/Z_0}
=\frac1N\sum_{i=1}^N e^{L^i}.
\]

After weighting, a transition kernel invariant for \(\pi_{\beta_k}\) moves each
particle. The transition need not produce an independent draw; invariance is
the required identity.

The implementation stores both:

- the weighted cloud before resampling and mutation; and
- the cloud after resampling and mutation.

This matters because post-resampling weights are uniform and therefore cannot
diagnose the collapse that triggered resampling.

## Reduced work and the Jarzynski convention

Sampler Lab defines reduced work by

\[
W^i=-L^i.
\]

Consequently,

\[
\mathbb E[e^{-W}] = Z_1/Z_0.
\]

The sign is fixed in the API: `jarzynski_estimate(work)` always averages
`exp(-work)`. The dimensionless free energy is

\[
F=-\log Z,
\qquad
\Delta F=F_1-F_0=-\log(Z_1/Z_0).
\]

For independent work trajectories, the elementary delta-method standard error
of \(\widehat{\Delta F}\) is the relative standard error of the ratio estimate.
The corresponding leading bias approximation is half its square. These IID
uncertainty formulas are intentionally disabled after resampling because the
resulting trajectories are dependent and duplicated.

## Annealed sequential Monte Carlo

At each stage, normalized incoming log weights \(\log w_{k-1}^i\) are updated by
\(\Delta\ell_k^i\). The incremental ratio estimator is

\[
\widehat{Z_{\beta_k}/Z_{\beta_{k-1}}}
=\sum_i w_{k-1}^i e^{\Delta\ell_k^i}.
\]

The complete ratio estimate is the product of these increments, accumulated in
log space:

\[
\log \widehat{Z_1/Z_0}
=\sum_{k=1}^K
 \log\left(\sum_i w_{k-1}^i e^{\Delta\ell_k^i}\right).
\]

Resampling can occur at every stage or when weighted ESS falls below a chosen
fraction of the population. The existing multinomial, systematic, and
floor-plus-Bernoulli resamplers are reused. Parent maps are retained, including
variable population sizes.

## Intermediate-state reweighting

A cloud approximating \(\pi_{\beta_a}\) can be reweighted toward a later path
law \(\pi_{\beta_b}\) by adding

\[
\log\gamma_{\beta_b}(x)-\log\gamma_{\beta_a}(x)
\]

to its current log weights. `reweight_cloud` performs this operation without
mutating the source cloud.

## Ising reference experiment

The demonstration starts from the exactly sampleable infinite-temperature Ising
law, \(\beta=0\), and anneals to a user-selected inverse temperature. For a
finite lattice,

\[
\log(Z_\beta/Z_0)
=\log Z_\beta - n_{\text{sites}}\log 2
\]

is available by exact enumeration on small systems. This gives simultaneous
checks of the normalization ratio, free-energy difference, and final-state
magnetization.

A vectorized deterministic Gibbs sweep updates each lattice site across the
entire particle population. This preserves the same lexicographic Gibbs
schedule as the scalar kernel while avoiding one Python call per particle and
site.

Run:

```bash
sampler-lab-annealing-demo \
  --size 2 \
  --target-beta 0.6 \
  --path-steps 2 4 8 16 32 \
  --particles 5000
```

The output compares ordinary Jarzynski/AIS with ESS-triggered systematic
resampling and reports log-ratio error, free energy, final absolute
magnetization, minimum pre-resampling ESS, resampling count, surviving initial
ancestors, and spin-update cost.
