# Hamiltonian and Underdamped Dynamics

## Phase-space convention

For a differentiable target density \(\pi(q)\) and a symmetric positive-definite
mass matrix \(M\), Sampler Lab uses

\[
H(q,p) = U(q) + K(p),
\qquad
U(q)=-\log\pi(q),
\qquad
K(p)=\frac12 p^\mathsf{T}M^{-1}p,
\]

with momentum

\[
p\sim N(0,M).
\]

The canonical equations are

\[
\dot q=M^{-1}p,
\qquad
\dot p=\nabla\log\pi(q).
\]

This mass convention is explicit because libraries disagree about whether an API
named `mass` stores \(M\) or \(M^{-1}\). For a Gaussian target with precision
\(\Lambda=\Sigma^{-1}\), choosing \(M=\Lambda\) equalizes the Hamiltonian normal-mode
frequencies. Choosing \(M=\Sigma\) does not.

`PhaseSpaceState` stores position and momentum separately; its array representation is
`[q, p]`. `MassMatrix` provides kinetic energy, velocity, Cholesky sampling, and the
inverse matrix without asking downstream algorithms to repeat factorizations.

## Conservative skew flows

The canonical symplectic matrix is only one member of a broader construction. For a
skew-symmetric matrix field \(A(x)\),

\[
b(x)=A(x)\nabla H(x)-\operatorname{div}A(x)
\]

preserves a density proportional to \(e^{-H(x)}\), subject to the usual boundary
conditions. The divergence is row-wise:

\[
(\operatorname{div}A)_i=\sum_j \partial_j A_{ij}.
\]

For constant canonical \(A\), the correction vanishes and
\(\nabla H\cdot b=0\), giving exact energy conservation. The package keeps this
continuous-time invariance statement separate from reversibility of a discrete Markov
chain; those are related words, not interchangeable claims.

## Leapfrog / velocity Verlet

For step size \(h\), one leapfrog step is

\[
\begin{aligned}
p_{n+1/2} &= p_n + \frac h2\nabla\log\pi(q_n),\\
q_{n+1} &= q_n + hM^{-1}p_{n+1/2},\\
p_{n+1} &= p_{n+1/2} + \frac h2\nabla\log\pi(q_{n+1}).
\end{aligned}
\]

The implementation reuses interior half steps, so a trajectory of \(L\) leapfrog steps
uses \(L+1\) target-gradient evaluations. It exposes:

- initial and final Hamiltonian;
- energy error \(\Delta H\);
- forward/backward reversibility residual;
- finite-difference Jacobians;
- volume error \(|\log|\det J||\);
- symplectic residual \(\|J^\mathsf{T}\Omega J-\Omega\|_F\).

Leapfrog is exactly volume preserving and time reversible in exact arithmetic, but it
is not exactly energy preserving. Over a fixed trajectory duration, its global energy
error is second order in \(h\).

A `position_map` may be applied after each drift step. The XY experiments use this to
wrap angles onto \([ -\pi,\pi )\), keeping the chain on the compact torus rather than
sampling an improper periodic lift in Euclidean space.

## HMC

A position-valued HMC transition:

1. draws \(p\sim N(0,M)\);
2. applies \(L\) leapfrog steps;
3. accepts with

\[
\alpha=\min\{1,\exp[-\Delta H]\};
\]

4. retains the current position after rejection.

`HamiltonianMonteCarloKernel` records energy error, trajectory length, squared jump
distance, acceptance, and operation counts. `UnadjustedHamiltonianKernel` deliberately
omits the Metropolis correction so discretization bias can be studied rather than
quietly hidden.

For Gaussian targets, the repository computes the exact Hamiltonian frequencies,
exact flow matrix, leapfrog matrix, and stability threshold

\[
h\,\omega_{\max}<2.
\]

The exact-flow position transition also gives an analytical linear-observable IAT,
which is used as a reference rather than pretending the Metropolized numerical chain
is itself a Gaussian AR(1).

## Involutive Metropolis correction

For a deterministic involution \(T(Tx)=x\), the generalized acceptance ratio is

\[
\log r(x)
=
\log\pi(Tx)-\log\pi(x)
+
\log|\det DT(x)|.
\]

`InvolutiveMetropolisKernel` includes the Jacobian term, so it also supports
non-volume-preserving involutions. Momentum flip is the basic phase-space example.
Because leapfrog satisfies

\[
\Phi^{-1}=R\Phi R,
\]

where \(R(q,p)=(q,-p)\), the composition \(T=R\Phi\) is an involution. Ordinary HMC
uses the equivalent energy correction because both leapfrog and momentum flip have
unit absolute Jacobian.

## Persistent momentum and transformed rejection

Partial refreshment is

\[
p' = \rho p + \sqrt{1-\rho^2}\,\xi,
\qquad
\xi\sim N(0,M),
\qquad 0\le\rho\le1.
\]

It preserves the Gaussian momentum law. `PersistentHamiltonianKernel` then applies a
leapfrog proposal. Accepted trajectories retain the forward momentum \(\Phi(q,p')\),
while rejected trajectories return \(R(q,p')\). The momentum-flipped rejection is not
cosmetic: it is the post-transformed form of Metropolis correction for the involution
\(R\Phi\).

## Underdamped Langevin and BAOAB

The continuous dynamics are

\[
\begin{aligned}
dq_t &= M^{-1}p_t\,dt,\\
dp_t &= \nabla\log\pi(q_t)\,dt
      -\gamma p_t\,dt
      +\sqrt{2\gamma M}\,dW_t.
\end{aligned}
\]

The generator is split into a skew Hamiltonian component and a symmetric
Ornstein--Uhlenbeck component. The exact OU substep is

\[
p(t+h)=e^{-\gamma h}p(t)
+\sqrt{1-e^{-2\gamma h}}\,\xi,
\qquad \xi\sim N(0,M).
\]

`UnderdampedLangevinKernel` uses the symmetric BAOAB splitting

\[
B_{h/2}A_{h/2}O_hA_{h/2}B_{h/2}.
\]

It is unadjusted and therefore may have discretization bias for nonlinear targets.
`MetropolizedUnderdampedLangevinKernel` combines the exact OU refresh with persistent
HMC and transformed rejection, restoring the exact phase-space target.

## XY model laboratory

For an \(L\times L\) periodic lattice,

\[
\log\pi(\theta)
=
\beta\left[
J\sum_{\langle i,j\rangle}\cos(\theta_i-\theta_j)
+h\sum_i\cos\theta_i
\right].
\]

`XYModel` provides the unnormalized log density, exact gradient, energy,
magnetization, toroidal wrapping, and periodic angle differences. The one-site field
case is a von Mises law with

\[
\mathbb E[\cos\theta]=\frac{I_1(\beta h)}{I_0(\beta h)},
\]

which supplies a continuous-state exact benchmark for HMC, BAOAB, and Metropolized
underdamped dynamics.

Run the combined conditioning and XY study with

```bash
sampler-lab-hamiltonian-demo \
  --condition-numbers 1 10 100 \
  --samples 5000
```

The Gaussian table reports exact-flow versus empirical IAT, HMC energy error,
acceptance, variance, and ESS per thousand gradient evaluations. The XY table reports
the exact circular response, empirical error, IAT, acceptance, and gradient-normalized
ESS.
