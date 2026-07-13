# Sequential Importance Sampling and Particle Resampling

!!! tip "Visual tutorial"
    Work through [Particle methods](../notebooks/03_particle_methods.ipynb) for executed
    resampling, ancestry, and self-avoiding-walk experiments using the public API.

## Particle clouds

A `ParticleCloud` stores particles \(X^{(i)}\) and normalized log weights
\(\log W^{(i)}\), with

\[
\sum_i W^{(i)}=1.
\]

Inputs are copied and made read-only. This is deliberate: a cloud in a
sequential history is a snapshot, not a mutable bag whose past can change
underneath later diagnostics.

The sequential result retains two histories:

- `weighted_clouds`: after propagation and incremental weighting, before
  resampling;
- `clouds`: after the optional resampling decision.

The distinction matters. A resampled cloud has uniform weights by
construction, so examining only post-resampling ESS would hide the collapse
that triggered resampling.

## Sequential normalizing constants

Suppose the current normalized weights are \(W_{n-1}^{(i)}\), and propagation
produces incremental weights \(w_n^{(i)}\). The update is

\[
\widehat r_n=\sum_i W_{n-1}^{(i)}w_n^{(i)},
\qquad
W_n^{(i)}=
\frac{W_{n-1}^{(i)}w_n^{(i)}}{\widehat r_n}.
\]

The estimated normalizing constant telescopes:

\[
\widehat Z_n=\widehat Z_0\prod_{k=1}^n \widehat r_k.
\]

The implementation performs these calculations in the log domain. When a
resampling step occurs, the old weights are represented by offspring counts,
and the selected particles restart with uniform weights.

## Resampling rules

For normalized weights \(W_i\) and target population \(N\), unbiased offspring
counts satisfy

\[
\mathbb E[N_i\mid W]=NW_i.
\]

### Multinomial

Draw \(N\) parent indices independently from the categorical distribution
\(W\). Then

\[
\operatorname{Var}(N_i\mid W)=NW_i(1-W_i).
\]

This is simple but noisy.

### Bernoulli

The implemented offspring rule is

\[
N_i=\lfloor NW_i\rfloor
+\operatorname{Bernoulli}\bigl(NW_i-\lfloor NW_i\rfloor\bigr).
\]

Each marginal offspring variance is minimal among unbiased integer-valued
counts:

\[
\operatorname{Var}(N_i\mid W)=r_i(1-r_i),
\qquad r_i=NW_i-\lfloor NW_i\rfloor.
\]

The total population is random. It can, in principle, be zero; the library
raises `ParticleExtinctionError` rather than retrying and quietly changing the
law.

### Systematic

One uniform offset places \(N\) equally spaced points in the cumulative weight
interval. The total population is exactly \(N\), and each marginal count is
also either \(\lfloor NW_i\rfloor\) or \(\lceil NW_i\rceil\). Its joint
correlations differ from independent Bernoulli resampling. Systematic
resampling is often excellent in practice, but its ordering dependence should
not be mistaken for a universal convergence theorem.

## ESS-triggered resampling

`resample_ess_fraction=a` resamples when

\[
\operatorname{ESS}_w \le aN,
\qquad
\operatorname{ESS}_w=\frac{1}{\sum_i W_i^2}.
\]

The threshold is evaluated before resampling and saved in the weighted-cloud
history.

## Self-avoiding walks

A fixed-origin \(d\)-step square-lattice self-avoiding walk is a path with
\(d+1\) distinct vertices and nearest-neighbor increments. Under the
Rosenbluth proposal, a path is extended uniformly among its currently
available neighbors. If \(m_n\) neighbors are available at step \(n\), the
incremental importance weight is \(m_n\). Hence

\[
\widehat c_d
=rac1N\sum_{i=1}^N\prod_{n=1}^d m_n^{(i)}
\]

without resampling, while the particle version estimates the same count by a
product of average incremental weights. Dead paths receive zero weight.

Run the comparison with

```bash
sampler-lab-particle-demo --steps 10 --particles 20000
```

For small lengths the CLI compares against exact depth-first enumeration. The
exact routine is a validation tool, not an attempt to compete with specialized
self-avoiding-walk enumeration algorithms.
