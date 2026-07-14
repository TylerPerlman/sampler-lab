# Generator Limits and Langevin Dynamics

!!! tip "Visual tutorial"
    Work through [Langevin sampling](../notebooks/07_langevin_sampling.ipynb) for executed
    diffusion-limit, ULA stability, preconditioning, MALA, and controlled-instability studies.

Phase 7 adds a common numerical layer for diffusion limits, Euler discretizations,
and gradient-based Markov chains.

## Generator convention

For a family of kernels `P_h`, the scaled discrete generator is

\[
L_h f(x)=\frac{P_h f(x)-f(x)}{h}
=\frac{\mathbb E_x[f(X_h')]-f(x)}{h}.
\]

`estimate_discrete_generator` estimates this conditional expectation by repeatedly
starting the kernel from the same state. It is not a time average from one chain.

For an Itô diffusion

\[
dX_t=b(X_t)\,dt+\sigma(X_t)\,dW_t,
\qquad a(x)=\sigma(x)\sigma(x)^\mathsf T,
\]

the continuous generator is

\[
Lf=b\cdot\nabla f+\frac12\operatorname{tr}(a\nabla^2 f).
\]

The local-moment utility separately reports

\[
\frac{\mathbb E[\Delta X]}{h},
\qquad
\frac{\mathbb E[\Delta X\Delta X^\mathsf T]}{h},
\]

as empirical drift and covariance-rate diagnostics.

## Euler--Maruyama

For a user-supplied drift and diffusion factor, the implemented step is

\[
X_{n+1}=X_n+h b(X_n)+\sqrt{h}\,\sigma(X_n)Z_n,
\qquad Z_n\sim N(0,I).
\]

The factor may be rectangular, so the Brownian-noise dimension need not equal the
state dimension.

## Preconditioned overdamped Langevin dynamics

For target density `pi` and positive-definite matrix field `M(x)`, the invariant
Itô diffusion is

\[
dX_t=
\left[M(X_t)\nabla\log\pi(X_t)+\operatorname{div}M(X_t)\right]dt
+\sqrt{2M(X_t)}\,dW_t.
\]

The row-wise divergence has components

\[
(\operatorname{div}M)_i=\sum_j \partial_j M_{ij}.
\]

The divergence term vanishes for constant preconditioners. Omitting it for a
position-dependent metric generally changes the invariant law. The API therefore
requires an explicit divergence function for a functional preconditioner and makes
suppression of the correction a visible option rather than an accidental default.

## ULA

The unadjusted Langevin algorithm uses the Euler step

\[
X_{n+1}=X_n+h\left[M(X_n)\nabla\log\pi(X_n)
+\operatorname{div}M(X_n)\right]
+\sqrt{2hM(X_n)}Z_n.
\]

ULA is inexpensive and often mixes well, but for fixed `h` its invariant law is
usually not the target law.

## MALA

The Metropolis-adjusted Langevin proposal is the same Gaussian Euler move,

\[
q(y\mid x)=N\!\left(
 x+h\left[M(x)\nabla\log\pi(x)+\operatorname{div}M(x)\right],
 2hM(x)
\right),
\]

followed by the full Hastings correction

\[
\alpha(x,y)=1\wedge
\frac{\pi(y)q(x\mid y)}{\pi(x)q(y\mid x)}.
\]

Both full-covariance proposal densities are evaluated explicitly. The implementation
therefore remains a valid MH kernel for state-dependent matrix fields, not only for
the constant identity case usually called MALA.

## Exact Gaussian ULA analysis

For a Gaussian target `N(m, Sigma)` and constant preconditioner `M`, centered ULA is

\[
Y_{n+1}=A Y_n+\xi_n,
\qquad
A=I-hM\Sigma^{-1},
\qquad
\xi_n\sim N(0,2hM).
\]

The chain is stable exactly when `rho(A) < 1`. Equivalently,

\[
0<h<\frac{2}{\lambda_{\max}(M\Sigma^{-1})}.
\]

When stable, its invariant covariance solves

\[
C=ACA^\mathsf T+2hM.
\]

The library solves this discrete Lyapunov equation with NumPy only and reports:

- the spectral radius;
- the largest stable step size;
- the exact stationary covariance;
- covariance bias relative to the target;
- the Gaussian KL divergence from the ULA invariant law to the target;
- exact IAT for any linear observable.

Choosing `M = Sigma` removes the target condition number from the stability bound:
`A = (1-h)I`. It does not remove Euler bias; for this case,

\[
C=\frac{\Sigma}{1-h/2}.
\]

MALA removes that fixed-step invariant-law bias through acceptance/rejection.

## Poisson-equation bias identity

If `g` solves

\[
Lg=f-\pi(f),
\]

and `pi_h` is invariant for a discrete kernel `P_h`, then

\[
\pi_h(f)-\pi(f)
=
\pi_h\left[
L g-\frac{P_h-I}{h}g
\right].
\]

`estimate_poisson_invariant_bias` estimates the right-hand side from samples of the
discrete invariant law. This gives an independent diagnostic of discretization bias,
separate from direct moment comparison.

## Small-step Metropolis limit

For one-dimensional random-walk Metropolis with proposal variance `2h`, the scaled
generator converges to overdamped Langevin:

\[
L f=f''+(\log\pi)'f'.
\]

The Phase 7 CLI estimates this limit numerically before comparing identity ULA,
covariance-preconditioned ULA, and covariance-preconditioned MALA on increasingly
ill-conditioned Gaussian targets.
