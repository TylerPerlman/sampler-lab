# Adaptive and Policy-Gradient Sampling

!!! tip "Visual tutorial"
    Work through [Adaptive and learned sampling](../notebooks/10_adaptive_and_learned_sampling.ipynb)
    for executed warmup schedules, covariance repair, objective gaming, variational correction, and SVGD comparisons.

This module develops adaptive and policy-based sampling as a set of transparent
adaptive, reinforcement-learning, variational, and Stein methods. The central rule is that
**training and sampling are different phases**. Warmup trajectories diagnose and optimize a
kernel; reported Monte Carlo output comes from a fresh chain after the learned parameters are
frozen.

## Exactness boundary

A learned proposal does not invalidate Metropolis--Hastings merely because it was learned.
After training, a fixed proposal kernel \(q_\theta(y\mid x)\) is corrected with

\[
\alpha_\theta(x,y)
=
1\wedge
\frac{\pi(y)q_\theta(x\mid y)}{\pi(x)q_\theta(y\mid x)}.
\]

The resulting frozen kernel is exact when the ordinary MH support conditions hold. Training
states are never returned as posterior samples. The repository records them in
`AdaptiveTrainingResult`, while `EvaluationTrajectory` contains a fresh post-freeze chain and
the number of excluded warmup transitions.

State-independent mixtures of valid MH kernels remain valid after their mixture probabilities
are frozen. A state-dependent policy requires the full marginal proposal density in both
directions; selecting a proposal and then ignoring the policy probability in the Hastings
ratio is incorrect.

`FrozenPolicy` stores detached numerical parameters and metadata and supports JSON
serialization. It deliberately serializes data rather than executable policy objects.

## Adaptive random-walk Metropolis

The adaptive random-walk implementation combines three elementary pieces.

### Running moments

For observations \(x_1,\ldots,x_n\), Welford updates maintain the mean and centered second
moment without storing the complete history. The sample covariance is regularized before it
becomes a proposal covariance.

### Covariance repair

An empirical covariance is symmetrized, shrunk toward a diagonal reference when requested,
and eigendecomposed. Eigenvalues are clipped to declared lower and upper bounds. The result
includes both the repaired matrix and spectral diagnostics.

### Diminishing scale adaptation

The scalar proposal scale is updated on the log scale with a Robbins--Monro schedule,

\[
\log s_{t+1}
=
\log s_t+\eta_t(\alpha_t-\alpha_\star),
\qquad
\eta_t=c(t+t_0)^{-\kappa}.
\]

The schedule validates the usual diminishing range \(1/2<\kappa\le 1\). A dual-averaging
implementation is also provided for algorithms that need a smoothed terminal step size.
Expanding warmup windows separate initial scale adaptation, covariance-learning windows, and
a final stabilization buffer.

`train_adaptive_random_walk` produces a frozen full-covariance random-walk kernel;
`evaluate_adaptive_random_walk` rebuilds that kernel with a fresh operation counter and random
stream.

## Policy families

### Linear softmax policy

For features \(\phi(x)\) and parameter matrix \(W\), the categorical policy is

\[
\Pr(A=a\mid x)
=
\operatorname{softmax}(W\phi(x))_a.
\]

The returned action includes its log probability and score
\(\nabla_W\log\pi_W(a\mid x)\). A frozen policy stores a read-only copy of the parameters.
The current exact kernel-selection trainer intentionally uses constant features, making the
post-training mixture state independent. `PolicyMixtureProposal` separately implements the
full forward and reverse density needed for exact state-dependent mixtures.

### Squashed Gaussian policy

A Gaussian latent action is transformed through a bounded hyperbolic tangent map. The class
exposes the transformed log density and score, making it suitable for continuous proposal
scales without permitting invalid negative or unbounded parameters.

## Policy-gradient estimators

For rewards-to-go \(G_t\), REINFORCE estimates

\[
\nabla_\theta J(\theta)
\approx
\frac1N\sum_t
\left(G_t-b_t\right)
\nabla_\theta\log\pi_\theta(A_t\mid X_t).
\]

The implementation includes zero, running-mean, and linear least-squares baselines. Baselines
change variance but not the score-function expectation when they do not depend on the sampled
action. Discounted and undiscounted returns are both available.

For categorical policies, the exact Fisher matrix is constructed from the action
probabilities and features. `natural_gradient_direction` solves a damped Fisher system and
scales the direction to a declared local KL trust region. The implementation returns the raw
gradient, natural direction, predicted quadratic KL, damping, and scale factor rather than
hiding the stabilization choices.

All gradient-bearing primitives are checked against finite differences in the test suite.

## Reward library

A reward receives the current state, proposal, retained next state, acceptance decision,
Metropolis log ratio, proposal entropy, diagnostic features, and optional operation costs.

### Acceptance-only baseline

\[
r_{\mathrm{acc}}=\mathbf 1\{\text{accepted}\}
\quad\text{or}\quad
\mathbb E[r\mid x,y]=\alpha(x,y).
\]

This is implemented as an intentional failure baseline. Shrinking a proposal toward zero can
make acceptance approach one while movement and effective sample size collapse.

### Accepted squared jump

\[
r_{\mathrm{jump}}
=
\mathbf 1\{\text{accepted}\}
(y-x)^\mathsf T M(y-x).
\]

The metric may standardize coordinates or emphasize declared directions. A feature-space
variant measures changes in user-supplied observables, which can include mode indicators or
funnel scale coordinates.

### Generalized speed

The sampled lower-bound objective is

\[
r_{\mathrm{GS}}
=
\min(0,\log r(x,y))+\beta H[q_\theta(\cdot\mid x)].
\]

Unlike accepted-jump rewards, it retains a gradient signal for rejected proposals through the
Metropolis ratio and proposal entropy. A specialized analytic gradient is implemented for
diagonal Gaussian random walks.

### Contrastive-divergence lower bound

Let \(a=\min(1,e^{\log r})\),
\(\Delta=\log\pi(y)-\log\pi(x)\), and let \(q(y\mid x)\) be the forward proposal density.
The implemented one-proposal reward is

\[
r_{\mathrm{CDLB}}
=
lambda_\pi a\Delta
-\lambda_H a\log a
-\lambda_H a\log q(y\mid x).
\]

The entropy terms form a lower bound on the transition-kernel entropy after dropping the
nonnegative entropy of the rejection atom. The reward balances movement toward higher target
density with transition diversity and does not require the proposal to be accepted in the
sampled rollout.

### Cost normalization

Any base objective can be divided by a weighted operation cost. The cost model has explicit
weights for log densities, proposal densities, gradients, Hessians, factorizations, policy
evaluations, and training-objective evaluations. Training and frozen evaluation use separate
`OperationCounter` instances.

## Variational proposal learning

`DiagonalGaussianVariational` parameterizes

\[
q_\lambda(x)=\mathcal N(x;\mu,\operatorname{diag}(e^{2\rho})).
\]

### Reverse KL

Using reparameterized samples \(x=\mu+e^\rho\odot\epsilon\), the optimizer minimizes

\[
\mathrm{KL}(q_\lambda\|\pi)
=
\mathbb E_q[\log q_\lambda(X)-\log\pi(X)] + \text{constant}.
\]

Reverse KL may select one component of a separated mixture. The benchmark treats that as a
measurable failure mode, not a bug in the optimizer.

### Forward KL from trusted samples

When independent reference samples are available, the same family may be fit by maximizing
its likelihood, equivalently minimizing \(\mathrm{KL}(\pi\|q_\lambda)\). This objective is
mode covering but depends on trusted target samples.

### Exact correction

A frozen variational approximation can become an independence proposal. The
`corrected_kernel` method includes both \(q(y)\) and \(q(x)\) in the Hastings ratio. The
corrected chain is exact even when the approximation is poor, although poor overlap can make
its finite-run mixing dreadful.

## Stein discrepancy and SVGD

For target score \(s_p(x)=\nabla\log p(x)\) and an inverse-multiquadric kernel, the empirical
kernel Stein discrepancy is computed from the standard Stein kernel

\[
u_p(x,y)
=
s_p(x)^\mathsf T k(x,y)s_p(y)
+s_p(x)^\mathsf T\nabla_y k(x,y)
+s_p(y)^\mathsf T\nabla_x k(x,y)
+\operatorname{tr}\nabla_{xy}k(x,y).
\]

SVGD updates every particle with the empirical velocity

\[
\phi^\star(x)
=
\frac1N\sum_j
\left[k(x_j,x)s_p(x_j)+\nabla_{x_j}k(x_j,x)\right].
\]

The output is a dependent deterministic particle approximation, not a stationary Markov
chain. It is therefore evaluated with discrepancy, moment, and mode-coverage metrics rather
than chain IAT. Exact downstream use requires a separate correction or weighting step.

## Benchmark targets

Phase 10 adds normalized, exactly sampleable targets with analytic derivatives:

- a rotated anisotropic Gaussian mixture;
- a rotated anisotropic Gaussian funnel;
- a bimodal mixture of anisotropic funnels.

The benchmark registry declares sampler and target capabilities before constructing a pair.
Unsupported combinations return an exclusion reason instead of a fabricated score.

The quick Phase 10 demonstration uses the separated mixture because it exposes several
failures cheaply: local-chain trapping, acceptance gaming, reverse-KL mode selection, and poor
independence-proposal overlap. The funnel and bimodal-funnel targets are available for the
larger cross-method benchmark planned as an opt-in experiment.

## Diagnostics and reporting

Every learned method reports training and evaluation separately. Depending on method type,
metrics include:

- acceptance, standardized jump distance, IAT, ESS, and mode switches;
- mean, covariance, and declared-expectation error against exact references;
- mode occupancy, first-passage, residence, and round-trip diagnostics;
- MMD or IMQ-KSD for approximate particle output;
- objective history, policy probabilities, covariance spectrum, and trust-region size;
- target, derivative, factorization, policy, and training-objective operation counts.

No single metric is treated as a universal proxy for convergence. In particular, coordinate
ESS cannot certify correct mode weights.

## Reproducible command

```bash
sampler-lab-policy-demo \
  --samples 3000 \
  --warmup 1000 \
  --policy-updates 80
```

Add `--json` for machine-readable output. The command runs an objective-gaming study on a
standard Gaussian and then compares exact, adaptive, policy-gradient, variational, corrected,
and Stein methods on a separated anisotropic mixture.
