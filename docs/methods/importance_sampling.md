# Importance Sampling

## Three estimators that should not be conflated

Let samples \(X_i\) be IID from a normalized proposal density \(q\), and let

\[
\ell_i = \log \widetilde p(X_i)-\log q(X_i).
\]

The package keeps three related calculations separate.

### Known target normalization

When \(p\) is normalized and \(w_i=p(X_i)/q(X_i)\),

\[
\widehat\mu_{\mathrm{IS}}
=\frac1N\sum_{i=1}^N w_i f(X_i)
\]

is unbiased under the usual integrability conditions. Its reported Monte Carlo
standard error is the sample standard deviation of the weighted contributions
divided by \(\sqrt N\).

```python
from sampler_lab.importance import standard_importance_estimate

estimate = standard_importance_estimate(values, log_weights)
```

### Unknown target normalization

When only \(\widetilde p\) is available,

\[
\widehat\mu_{\mathrm{SNIS}}
=\frac{\sum_i w_i f(X_i)}{\sum_i w_i}
=\sum_i \omega_i f(X_i),
\qquad
\omega_i=\frac{w_i}{\sum_j w_j}.
\]

The implementation normalizes in the log domain. The plug-in delta-method
standard error is

\[
\left[
\frac{N}{N-1}
\sum_i \omega_i^2
\left(f(X_i)-\widehat\mu_{\mathrm{SNIS}}\right)^2
\right]^{1/2}.
\]

A second-order ratio-estimator bias approximation is also returned. It is a
diagnostic, not a magic finite-sample correction.

```python
from sampler_lab.importance import self_normalized_importance_estimate

estimate = self_normalized_importance_estimate(values, log_unnormalized_weights)
```

### Normalizing-constant ratios

If \(X_i\) are sampled from \(q=\widetilde q/Z_q\), then

\[
\frac{Z_p}{Z_q}
=\mathbb E_q\left[\frac{\widetilde p(X)}{\widetilde q(X)}\right].
\]

`estimate_normalization_ratio` returns both the linear estimate and its stable
logarithm. The log result remains useful when the linear value overflows
`float64`.

## Weight diagnostics

For normalized weights \(\omega_i\), the package reports

\[
\operatorname{ESS}_w=\frac{1}{\sum_i\omega_i^2},
\qquad
\widehat{\chi^2}=N\sum_i\omega_i^2-1,
\]

along with maximum weight, entropy, normalized entropy, and perplexity.

Weight ESS measures concentration of the weights. It is not generally the ESS
of a particular observable. A proposal can have ugly global weights while
estimating one carefully chosen rare-event observable quite efficiently—and the
reverse can also happen.

## Gaussian tail experiment

For \(p=N(0,1)\), \(q=N(\mu,1)\), and the event \(X\ge a\),

\[
\log\frac{p(x)}{q(x)}=-\mu x+\frac{\mu^2}{2}.
\]

Run the fixed-seed comparison with

```bash
sampler-lab-importance-demo --threshold 4 --samples 100000
```

The default proposals include crude Monte Carlo (\(\mu=0\)) and shifted
proposals. At a four-sigma threshold, crude Monte Carlo commonly sees only a
handful of events, while a proposal centered near the threshold turns the event
into an ordinary occurrence.

## Product-space collapse

For centered isotropic Gaussians

\[
p=N(0,\sigma_p^2I_d),\qquad q=N(0,\sigma_q^2I_d),
\]

the exact divergence is

\[
1+\chi^2(p\Vert q)
=
\left(
\frac{\sigma_q^2}
{\sigma_p\sqrt{2\sigma_q^2-\sigma_p^2}}
\right)^d,
\]

provided \(2\sigma_q^2>\sigma_p^2\); otherwise it is infinite. Even a mild
one-dimensional mismatch is therefore amplified exponentially by product
dimension. The collapse experiment reports empirical and exact values side by
side.
