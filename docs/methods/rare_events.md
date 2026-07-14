# Rare Events and Small-Noise Importance Sampling

!!! tip "Visual tutorial"
    Work through [Rare-event sampling](../notebooks/11_rare_event_sampling.ipynb) for log-tail stability,
    Laplace prefactors, tempering, dominant-point twisting, and multiple-minimizer mixtures.

## Target problem

Phase 11 estimates probabilities that vanish as a noise parameter tends to zero.  The exact
reference family is

\[
X_\varepsilon \sim N(0,\varepsilon C),
\]

with either

\[
A_+ = \{x:a^T x\ge b\}
\]

or the symmetric two-sided event

\[
A_\pm = \{x:|a^T x|\ge b\}.
\]

These problems are deliberately simple enough to admit exact probabilities and exact second
moments, but rich enough to expose the main asymptotic design issues.  The base covariance
may be anisotropic, the probability may be far below ordinary floating-point scale, and the
two-sided event has two equally important dominating points.

## Large-deviation geometry

The Gaussian rate function is

\[
I(x)=\frac12 x^T C^{-1}x.
\]

For the upper halfspace, the minimum of `I` over the event boundary is

\[
x_\star=\frac{bCa}{a^TCa},
\qquad
I_\star=\frac{b^2}{2a^TCa}.
\]

The exact probability is

\[
p_\varepsilon
=\bar\Phi\!\left(\frac{b}{\sqrt{\varepsilon a^TCa}}\right),
\]

and the boundary-Laplace approximation is

\[
p_\varepsilon
\sim
\frac{\sqrt{\varepsilon a^TCa}}{b\sqrt{2\pi}}
\exp\!\left(-\frac{I_\star}{\varepsilon}\right).
\]

The symmetric event has the same exponential rate and twice the leading prefactor because it
has two minimizers, `+x_star` and `-x_star`.

The package also exposes ordinary interior Laplace approximations.  For isolated
nondegenerate minimizers `x_j`,

\[
\int a(x)e^{-I(x)/\varepsilon}\,dx
\approx
\sum_j
 a(x_j)e^{-I(x_j)/\varepsilon}
 \frac{(2\pi\varepsilon)^{d/2}}
      {\sqrt{\det \nabla^2 I(x_j)}}.
\]

All contributions are summed in the log domain.  Multiple minimizers are not discarded just
because they share the same exponential rate.

## Crude Monte Carlo and relative error

For the indicator estimator

\[
\widehat p_N=\frac1N\sum_{i=1}^N 1_A(X_i),
\]

\[
\frac{\operatorname{Var}(1_A)}{p_\varepsilon^2}
=\frac{1-p_\varepsilon}{p_\varepsilon}.
\]

Consequently, the relative standard error is approximately

\[
\frac{1}{\sqrt{Np_\varepsilon}},
\]

which grows exponentially as `epsilon` shrinks.  Absolute error can look tiny while relative
error is catastrophic; Phase 11 therefore reports both.

For any nonnegative IID contribution `Y`, the implementation records

- the estimate and its logarithm;
- sample variance and standard error;
- relative variance `Var(Y) / E[Y]^2`;
- relative standard error;
- first and second moments in log form;
- contribution ESS `(sum Y)^2 / sum Y^2`;
- maximum normalized contribution;
- event count and operation counts.

The contribution ESS is estimator-specific.  It is not interchangeable with the ESS of raw
importance weights.

## Linear exponential twisting

A Gaussian shift proposal is

\[
q_m=N(m,\varepsilon C).
\]

Its exact likelihood ratio is

\[
\log\frac{p_\varepsilon(x)}{q_m(x)}
=-\frac{m^TC^{-1}x}{\varepsilon}
+\frac{m^TC^{-1}m}{2\varepsilon}.
\]

For the halfspace event, the exact second moment is

\[
E_{q_m}[1_Aw^2]
=
\exp\!\left(\frac{m^TC^{-1}m}{\varepsilon}\right)
\bar\Phi\!\left(
\frac{b+a^Tm}{\sqrt{\varepsilon a^TCa}}
\right).
\]

Choosing `m = x_star` is logarithmically efficient: the exponential growth rate of relative
variance vanishes, though polynomial growth remains.  This distinction matters.  A proposal
can be asymptotically efficient without having bounded relative error.

## Why one twist can fail

For `|a^T x| >= b`, choosing only `+x_star` makes the positive tail typical but pushes the
negative tail in the wrong direction.  The estimator remains unbiased because Gaussian
support is global, yet typical finite runs estimate only about half of the probability.  The
missing negative-tail contribution arrives through fantastically rare observations with huge
weights.

The exact second moment shows that

\[
\varepsilon\log\left(
\frac{E_q[1_Aw^2]}{p_\varepsilon^2}
\right)
\to 4I_\star.
\]

This is worse than crude Monte Carlo.  Unbiasedness has not been violated; usefulness has.

## Mixture twisting

The correct symmetric proposal is

\[
q=\tfrac12N(x_\star,\varepsilon C)
 +\tfrac12N(-x_\star,\varepsilon C).
\]

The implementation evaluates the mixture density with `logsumexp`.  For the exact reference
problem, orthogonal Gaussian coordinates integrate out and the second moment reduces to a
stable one-dimensional quadrature.  Its relative variance has zero exponential growth rate.

This is the central Phase 11 failure demonstration:

```text
single twist: exact but exponentially unstable
mixture twist: exact and logarithmically efficient
```

## Tempering

A centered tempered proposal uses

\[
q_\tau=N(0,\tau\varepsilon C),\qquad \tau\ge1.
\]

For a fixed temperature, broader tails reduce but generally do not remove exponential
relative-error growth.  A temperature that grows like `1 / epsilon` keeps the proposal
covariance at an `O(1)` scale and yields subexponential behavior on the Gaussian oracle, at
the cost of dimension-dependent prefactors and many samples far from the important boundary.

The exact second moment is available because

\[
\frac{p_\varepsilon^2}{q_\tau}
\]

is proportional to another centered Gaussian with precision multiplier
`2 - 1 / tau`.  A small grid selector minimizes this exact criterion.  It is intentionally a
purpose-built routine, not the seed of an accidental general optimization package.

## Stable normal tails

Ordinary `erfc` underflows in the far tail.  The implementation uses `erfc` centrally and a
Mills-ratio expansion for large positive thresholds, preserving exact log probabilities and
relative-error rates long after linear probabilities have become zero in `float64`.

## Exactness and required access

All estimators in this phase are ordinary importance-sampling estimators and are unbiased
when proposal support covers the event-supporting part of the target.  The Gaussian
implementations require

- the base covariance;
- a rare-event predicate;
- proposal means or temperatures;
- explicit random-number generators.

The exact second-moment diagnostics are oracle tools for this problem family.  General users
may still use the empirical log-contribution diagnostics when exact moments are unavailable.

## Cost

For `N` samples in dimension `d`:

- shifted and tempered Gaussian proposals consume `Nd` normal draws;
- a `K`-component mixture consumes `N` component uniforms, `Nd` normals, and `NK`
  proposal-density evaluations;
- likelihood ratios are evaluated in vectorized linear algebra;
- the exact symmetric-mixture second moment uses deterministic one-dimensional quadrature.

Operation counters accompany every sampled estimate.

## Reference experiment

```bash
sampler-lab-rare-event-demo \
  --epsilons 0.5 0.25 0.125 0.0625 \
  --samples 100000 \
  --dimension 4
```

The report compares

- crude Monte Carlo;
- fixed-temperature broadening;
- covariance-fixed asymptotic tempering;
- a dominant-point linear twist;
- a deliberately insufficient single twist for a two-minimizer event;
- the corresponding two-component mixture twist.

It reports observed and exact relative standard errors, event counts, contribution ESS, and
`epsilon * log(relative variance)`.  Add `--json` for machine-readable output.

## Known limitations

- The canonical exact problems are Gaussian and use linear rare sets.
- Generic nonlinear dominating-point optimization is not included; users should supply
  minimizers or use a dedicated optimizer externally.
- Splitting, subset simulation, nested sampling, and replica exchange remain outside the
  current implementation scope.
- The phase does not claim that one importance proposal is universally optimal.  Multiple
  minimizers, boundary curvature, and pre-exponential factors all matter.

## Validation scope

The implementations preserve repository-wide standards: typed APIs, explicit RNGs, stable log
arithmetic, operation counters, exact analytical tests, fixed-seed statistical checks, a
failure-mode experiment, and a runnable CLI.
