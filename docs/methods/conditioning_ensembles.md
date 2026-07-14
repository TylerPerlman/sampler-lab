# Conditioning, Affine Geometry, and Ensemble Methods

!!! tip "Visual tutorial"
    Work through [Geometry, conditioning, and ensembles](../notebooks/09_geometry_conditioning_and_ensembles.ipynb)
    for whitening, Gaussian conditioning, Hessian repair, stochastic Newton, and ensemble diagnostics.

This module keeps three related ideas separate:

1. changing coordinates to expose or remove conditioning;
2. using local curvature to build proposals;
3. learning useful directions from a population of walkers without estimating a
   covariance matrix explicitly.

The separation matters. A Hessian repair can improve a proposal while destroying exact
finite-step affine equivariance, and an affine-invariant ensemble can still mix badly if
its walkers begin in a degenerate affine subspace.

## Affine target transformations

Let

\[
y = A x + b,
\qquad A \in \mathbb R^{d\times d}\text{ invertible}.
\]

If `X` has density \(\pi_X\), the pushforward density is

\[
\log \pi_Y(y)
= \log \pi_X\!\left(A^{-1}(y-b)\right)-\log|\det A|.
\]

Its derivatives transform as

\[
\nabla_y \log\pi_Y=A^{-T}\nabla_x\log\pi_X,
\qquad
D_y^2\log\pi_Y=A^{-T}(D_x^2\log\pi_X)A^{-1}.
\]

`AffineMap` implements the map, inverse map, determinant, batch transforms, and
covariance pushforward. `AffineTransformedTarget` implements all three formulas above.
The tests compare the result against an explicitly transformed Gaussian, not merely
against a round trip through the same code.

For a Gaussian \(N(\mu,C)\), the whitening map is

\[
y=L^{-1}(x-\mu),\qquad C=LL^T,
\]

which produces a standard normal target and condition number one. Exact Gaussian
conditioning is also included. For observed block \(X_o=x_o\),

\[
\mu_{r\mid o}=\mu_r+C_{ro}C_{oo}^{-1}(x_o-\mu_o),
\]

\[
C_{r\mid o}=C_{rr}-C_{ro}C_{oo}^{-1}C_{or}.
\]

## Hessian metrics and repair

The local stochastic-Newton precision is

\[
G(x)=-D^2\log\pi(x).
\]

For a log-concave target, \(G(x)\) is positive definite and the local metric is
\(S(x)=G(x)^{-1}\). Non-log-concave targets such as Rosenbrock can have indefinite
Hessians. The library exposes the repair instead of silently hiding it:

- `raise`: reject a matrix that is not sufficiently positive definite;
- `clip`: replace each eigenvalue by \(\max(\lambda_i,\epsilon)\);
- `absolute`: replace each eigenvalue by \(\max(|\lambda_i|,\epsilon)\).

Every repair returns the original and repaired spectra and the Frobenius norm of the
correction. Clipping and absolute-value repair are useful, but they are generally not
affine equivariant under arbitrary nonorthogonal maps. That loss is a mathematical
property, not an implementation bug.

A centered finite-difference Hessian of a supplied gradient is available for targets
whose analytic Hessian is unavailable.

## Stochastic Newton and Metropolis correction

Using the package overdamped-Langevin scaling, the proposal is:

\[
Y=x+hS(x)\nabla\log\pi(x)+\sqrt{2hS(x)}\,\xi,
\qquad \xi\sim N(0,I).
\]

The divergence term of the position-dependent metric is omitted from the proposal.
This changes proposal efficiency but not the invariant law after the full asymmetric
Metropolis--Hastings correction:

\[
\log r
=\log\pi(y)-\log\pi(x)
 +\log q(x\mid y)-\log q(y\mid x).
\]

`MetropolizedStochasticNewtonKernel` evaluates the gradient, Hessian, repair, inverse,
and factorization once at each endpoint. The generic proposal interface remains
available, but the specialized kernel avoids evaluating the current geometry a second
time just to score the proposal it generated.

For a Gaussian target with no repair, the metric, proposal mean, and proposal covariance
transform exactly under affine maps. Tests verify those matrix identities directly.

## Ensemble state is the Markov state

For walkers \(x^{(1)},\ldots,x^{(L)}\), the target on the complete ensemble is

\[
\Pi(\mathbf x)=\prod_{j=1}^L\pi(x^{(j)}).
\]

The complete ensemble is Markov. A single walker generally is not, because its next
proposal depends on the other walkers. `EnsembleState`, `EnsembleTransition`, and
`EnsembleTrajectory` therefore store the whole product state, per-walker acceptance,
and partner information.

The affine-span rank is recorded explicitly. By default, the walk and stretch kernels
reject an ensemble whose centered walkers fail to span the target dimension. No
ensemble move can manufacture a missing affine direction from scalar combinations of
existing walkers.

## Stretch move

Choose another walker \(j\neq i\), sample \(Z\) from

\[
g(z)\propto z^{-1/2},\qquad z\in[1/a,a],
\]

and propose

\[
y=x^{(j)}+Z\bigl(x^{(i)}-x^{(j)}\bigr).
\]

The density satisfies \(g(1/z)=z g(z)\). The acceptance log ratio is

\[
(d-1)\log Z+\log\pi(y)-\log\pi(x^{(i)}).
\]

Both sequential and split-ensemble schedules are implemented. In a split update, all
walkers in one group use a frozen complementary group; the groups then swap roles.

## Walk move

The symmetric walk proposal selects \(s\) complementary walkers, centers them, and uses

\[
y=x^{(i)}+\frac{c}{\sqrt{s}}
  \sum_{j=1}^{s} z_j\left(x^{(j)}-\bar x\right),
\qquad z_j\sim N(0,1).
\]

Conditional on the complementary walkers, the increment distribution is symmetric.
The acceptance ratio therefore reduces to

\[
\log\pi(y)-\log\pi(x^{(i)}).
\]

Because every proposed displacement is an affine combination of walker differences,
the move is pathwise affine equivariant under coupled partner choices, Gaussian
coefficients, and uniforms. The tests run original and transformed ensembles with the
same random stream and compare entire updated product states.

## Ensemble diagnostics

For scalar walker values \(f(X_{k,j})\), the time series used for autocorrelation is the
ensemble average

\[
F_k=\frac1L\sum_{j=1}^{L}f(X_{k,j}).
\]

If equilibrium walkers are independent under the product law,
\(\operatorname{Var}(F)=\operatorname{Var}_\pi(f)/L\). The target-draw-equivalent ESS is
therefore reported as

\[
\operatorname{ESS}=\frac{LN}{\tau_F}.
\]

The package also reports per-walker acceptance and mean off-diagonal cross-walker
correlation. Flattening walkers is convenient for moment estimates, but pretending the
flattened array is IID would be statistically optimistic.

## Rosenbrock reference law

The benchmark density is

\[
\pi(x,y)\propto
\exp\left[-\frac{100(y-x^2)^2+(1-x)^2}{20}\right].
\]

It has the exact hierarchy

\[
X\sim N(1,10),
\qquad
Y\mid X\sim N(X^2,0.1).
\]

This gives exact independent samples and moments:

\[
E[X]=1,\quad E[Y]=11,
\qquad
\operatorname{Cov}(X,Y)=
\begin{pmatrix}
10&20\\
20&240.1
\end{pmatrix}.
\]

The geometry demonstration compares isotropic and fixed-covariance random walks,
Metropolized stochastic Newton, the stretch move, and the walk move. Exact starts are
used when measuring stationary IAT so burn-in error is not confused with transition
quality.
