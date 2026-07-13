# Exact Sampling

!!! tip "Visual tutorial"
    Work through [Exact and IID sampling](../notebooks/01_exact_and_iid_sampling.ipynb) for
    executed examples, quantitative checks, and the matching unit-disk CLI configuration.

## Inverse transform

For a CDF `F`, sample `U ~ Uniform(0, 1)` and return the generalized inverse

\[
F^{-1}(u) = \inf\{x : F(x) \ge u\}.
\]

The implementation requires a vectorized inverse function and passes random-number
state explicitly. For finite discrete laws, cumulative probabilities and
`numpy.searchsorted` implement the generalized inverse directly.

## Box–Muller

Given independent uniforms `U1, U2`,

\[
R = \sqrt{-2\log U_1}, \qquad \Theta = 2\pi U_2,
\]

then `R cos(Theta)` and `R sin(Theta)` are independent standard normals. The sampler
records two uniforms for each generated pair and intentionally discards one variate
when an odd sample count is requested.

## Uniform unit disk

Uniform area measure has radial CDF `P(R <= r) = r^2`; therefore
`R = sqrt(U)` and a uniform angle produce exact samples. Sampling radius uniformly
would overpopulate the center—a classic tiny formula, large bug.

The rejection alternative proposes uniformly on `[-1, 1]^2` and keeps points with
squared radius at most one. Its theoretical acceptance probability is `pi / 4`, so it
uses `4 / pi` proposals on average for every accepted point.

## Generic rejection interface

The generic implementation works in log space. Callers provide proposal samples,
log target values, log proposal values, and `log(M)` satisfying

\[
\tilde\pi(x) \le M q(x).
\]

The target may be unnormalized so long as the same convention is used to choose `M`.
The implementation raises when an evaluated point violates the claimed envelope,
rather than quietly returning biased samples.
