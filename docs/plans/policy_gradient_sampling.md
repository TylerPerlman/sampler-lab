# Adaptive and Policy-Gradient Sampling Phase

## Status

Implemented in v0.10.0 after v0.9.0 and before the rare-event phase.

This design develops adaptive proposal learning from public literature while preserving the repository's
existing standards: NumPy-first implementations, explicit random-number generators,
mathematically explicit kernels, exactness checks, operation counters, typed results,
fixed-seed experiments, and derivation-oriented documentation.

The delivered transparent core includes adaptive covariance warmup, Robbins--Monro and dual
averaging, categorical and bounded Gaussian policies, REINFORCE baselines, natural-gradient
trust regions, generalized-speed and contrastive-divergence objectives, reverse/forward KL
Gaussian fitting, IMQ-KSD, SVGD, exact independence-MH correction, normalized mixture/funnel
targets, and capability-aware benchmark foundations. Neural transports and large-network
policies remain intentionally deferred: they would require an automatic-differentiation
runtime and obscure the mathematical mechanisms this repository is designed to expose.

## Why this is one phase

Policy-gradient MCMC is not safe to bolt directly onto a proposal object. It needs the
adaptive-MCMC layer that decides when parameters may change, how adaptation diminishes or
stops, and which chain is valid for inference. The phase therefore has four connected parts:

1. adaptive-MCMC foundations;
2. policy-gradient optimization of exact Metropolis--Hastings kernels;
3. variational and Stein objectives for learned proposals or transports;
4. a common continuous-target benchmark laboratory.

The default workflow is:

```text
training / warmup with adaptation
-> freeze the learned policy and kernel
-> start fresh evaluation chains
-> report only frozen-kernel samples as ordinary MCMC output
```

A theorem-backed diminishing-adaptation mode may be added, but it must be labeled and tested
separately. No API may silently mix warmup transitions into reported posterior samples.

## Conceptual split

### Exact learned MCMC

A policy chooses proposal parameters, a proposal family, or a transition schedule. The
resulting proposal is followed by an explicit Metropolis--Hastings correction. Once the
policy is frozen, the target remains invariant regardless of whether the learned proposal is
good. Learning changes efficiency, not the stationary law.

Examples:

- choose a random-walk scale or covariance action;
- select among local, independence, Newton, Langevin, or mode-jump kernels;
- condition proposal scale on state features;
- choose a trajectory length or integrator step size from a finite safe set;
- learn an independence proposal and correct it with MH.

### Approximate variational or Stein transport

A parameterized distribution or particle transport is optimized toward the target using a
distributional discrepancy. These outputs are approximate unless followed by an exact
correction such as independence MH, transport-preconditioned HMC, or importance weighting.
The result types and documentation must keep that distinction explicit.

Examples:

- reverse-KL affine or Gaussian variational proposals;
- mixture proposals trained by cross-entropy when target samples are available;
- Stein variational gradient descent;
- IMQ kernel Stein discrepancy minimization;
- a learned transport used only as a coordinate map or MH proposal.

## Initial policy families

The core implementation stays interpretable and NumPy-only.

### Kernel-selection policies

- softmax categorical policy over a finite set of existing exact kernels;
- state-feature-conditioned softmax policy;
- Gaussian policy over bounded log step sizes;
- policy over a finite grid of HMC trajectory lengths or safe step sizes;
- mixture policy combining local and global proposals.

### Differentiable proposal policies

- scalar and diagonal Gaussian random-walk policies;
- linear state-dependent Gaussian mean and log-scale policies;
- full-covariance Gaussian independence proposals;
- affine transport proposals;
- small Gaussian-mixture independence proposals.

Deep neural networks and general normalizing flows are optional extensions, not runtime
requirements. The first phase should prove the architecture with analytic gradients and
finite-difference gradient checks.

## Objective library

No single reward is accepted as a universal proxy for mixing. The implementation should make
objectives swappable and compare them under identical rollouts.

### Deliberately weak baselines

1. **Acceptance indicator or mean acceptance probability**

   Included to demonstrate the degenerate optimum: infinitesimal moves can approach unit
   acceptance while producing nearly zero effective sample size.

2. **Expected squared jump distance**

   Use both Euclidean and standardized Mahalanobis forms, always multiplied by the accepted
   move. This is more informative than acceptance alone but remains local and can reward
   movement within one mode while missing global failure.

### Mixing-oriented objectives

3. **Feature jump / Dirichlet objective**

   Maximize expected squared changes in a declared set of observables. The objective must
   record which observables were used; optimizing one feature is not a claim of global
   mixing.

4. **Lag-one decorrelation or short-horizon return**

   Reward reductions in empirical autocorrelation over short rollouts. Use held-out
   observables to detect objective overfitting.

5. **Maximum-entropy generalized speed objective**

   Include the proposal-learning objective from gradient-based adaptive MCMC as a bridge
   between pure acceptance tuning and global policy learning. Keep its entropy term and its
   proposal-family assumptions explicit.

6. **Contrastive-divergence reward**

   Implement the literature-defined finite-horizon RLMH reward from its derivation rather
   than guessing a formula from the name. Keep it separate from variational KL and from the
   older contrastive-divergence training algorithm used for energy-based models.

7. **Cost-normalized reward**

   Divide or constrain rewards by target, gradient, Hessian, factorization, and wall-clock
   cost. A proposal that moves twice as far for one hundred times the cost did not win.

### Distributional objectives

8. **Reverse KL**

   For tractable reparameterized proposals, minimize

   \[
   \mathrm{KL}(q_\theta\|\pi)
   = \mathbb E_{q_\theta}[\log q_\theta(X)-\log \gamma(X)] + \log Z.
   \]

   The unknown normalizer is irrelevant to gradients. Document the familiar mode-seeking
   failure on separated mixtures.

9. **Inclusive / forward KL or cross-entropy**

   Only available when exact target samples, trusted reference samples, or importance
   weights are available. This is useful for oracle experiments and amortized proposals but
   must not pretend to be black-box posterior inference.

10. **Kernel Stein discrepancy**

   Implement score-based KSD using a convergence-determining inverse-multiquadric kernel as
   the default. RBF KSD may be exposed for comparison but not treated as universally
   diagnostic.

11. **Stein variational gradient descent**

    Implement the particle functional-gradient update, bandwidth diagnostics, and the
    repulsive term explicitly. SVGD output is a dependent particle approximation, not an
    IID chain.

### Regularization

- policy entropy;
- KL trust region between consecutive policies;
- action bounds that preserve numerical stability;
- covariance eigenvalue floors and ceilings;
- gradient clipping with recorded clip counts;
- optional natural-gradient / Fisher preconditioning.

## Gradient estimators

### Score-function policy gradients

Implement REINFORCE for stochastic policies over kernel actions:

\[
\nabla_\theta J(\theta)
= \mathbb E\left[\sum_t \nabla_\theta \log \pi_\theta(A_t\mid S_t)
\left(G_t-b(S_t)\right)\right].
\]

Required baselines:

- no baseline;
- running-mean baseline;
- linear state-value baseline fitted by least squares.

Report gradient variance before and after baseline subtraction.

### Pathwise gradients

Use analytic reparameterization gradients for tractable Gaussian and affine variational
families. Do not silently differentiate through the discontinuous MH accept/reject decision.
Pathwise surrogates may train a proposal, but exactness comes from the subsequent frozen MH
correction.

### Natural gradients and trust regions

For softmax and Gaussian policies, implement small exact Fisher matrices where practical.
Use linear solves, damping, and a measured policy-KL constraint. This remains a focused
sampler-training utility, not a general reinforcement-learning framework.

### Gradient validation

Every analytic policy gradient receives:

- a deterministic algebraic test when available;
- central finite-difference checks on tiny examples;
- a common-random-number check for objective differences;
- a fixed-seed direction-of-improvement test.

## Adaptive-MCMC safeguards

Implement before online policy learning:

- running means and covariance estimates;
- covariance shrinkage and eigenvalue regularization;
- Robbins--Monro and dual-averaging scalar adaptation;
- explicit adaptation schedules;
- warmup windows;
- diminishing step-size diagnostics;
- containment proxies such as proposal-scale, acceptance, and local-drift bounds;
- frozen-policy serialization;
- fresh evaluation-chain construction.

The API must distinguish:

```text
AdaptiveTrainingResult
FrozenPolicy
FrozenMarkovKernel
EvaluationTrajectory
```

A training trajectory is not automatically a valid posterior trajectory.

## Architecture

Planned modules:

```text
src/sampler_lab/
├── adaptive/
│   ├── schedules.py
│   ├── running_moments.py
│   ├── covariance.py
│   ├── dual_averaging.py
│   └── warmup.py
├── learning/
│   ├── policies.py
│   ├── objectives.py
│   ├── gradients.py
│   ├── baselines.py
│   ├── optimizers.py
│   ├── trainer.py
│   ├── adaptive_mh.py
│   ├── variational.py
│   └── stein.py
├── benchmarks/
│   ├── capabilities.py
│   ├── registry.py
│   ├── metrics.py
│   └── continuous_suite.py
└── models/
    ├── gaussian_mixture.py
    ├── funnel.py
    └── bimodal_funnel.py
```

The optimizer module should contain only the small algorithms needed by the phase: SGD,
Adam, Robbins--Monro, and damped natural-gradient steps. It should not become a generic
optimization library.

## Core interfaces

```python
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class PolicyAction:
    value: Array
    log_prob: float
    score: Array
    diagnostics: dict[str, float]


class StochasticPolicy(Protocol):
    @property
    def parameters(self) -> Array:
        ...

    def act(
        self,
        rng: np.random.Generator,
        features: Array,
    ) -> PolicyAction:
        ...


class PolicyObjective(Protocol):
    def reward(
        self,
        rollout: "PolicyRollout",
        costs: "OperationCounts",
    ) -> float:
        ...
```

Policies should return scores at sampling time so training does not need to reconstruct
hidden random choices. Rollouts retain proposed states, accepted states, actions, rewards,
features, and operation counts.

## Testing plan

### Deterministic tests

- softmax probabilities and score sums;
- Gaussian policy log probabilities and score derivatives;
- baseline normal equations;
- natural-gradient Fisher identities;
- dual-averaging recursions;
- covariance regularization eigenvalue bounds;
- reverse-KL gradients on Gaussian targets;
- Stein kernel and score identities;
- SVGD repulsion symmetry;
- frozen learned MH acceptance ratios.

### Exact invariance tests

- learn a proposal on a finite-state target, freeze it, construct the full MH matrix, and
  verify \(\pi P=\pi\);
- verify detailed balance when the learned kernel is designed to be reversible;
- verify that variational training alone is labeled approximate;
- verify that adding an independence-MH correction restores exact invariance.

### Failure-mode tests

- acceptance-only training drives a random-walk scale toward zero on a Gaussian target;
- reverse KL collapses to one component of a widely separated symmetric mixture from an
  asymmetric initialization;
- Euclidean ESJD selects poor scaling on an anisotropic target while standardized ESJD does
  not;
- a learned proposal without MH correction shows measurable bias;
- continuing non-diminishing adaptation is rejected by the ordinary chain API.

### Statistical tests

- learned scalar random-walk scale improves held-out ESS over a deliberately bad initial
  scale on a Gaussian target;
- policy-selected local/global mixtures improve mode switching on a separated mixture;
- transport-preconditioned MCMC improves funnel-neck traversal without changing exact
  moments after MH correction;
- SVGD and reverse-KL proposals are evaluated by distributional discrepancy, not chain ESS;
- natural-gradient or trust-region steps respect the configured policy-KL bound.

Performance claims should use replicated confidence intervals. CI tests should check broad,
mathematically justified behavior, not encode a fragile total ordering of stochastic methods.

## Documentation required for every learned sampler

In addition to the repository-wide method template, each learned method page must state:

1. policy state, action, and parameterization;
2. training objective and its known blind spots;
3. gradient estimator and variance-reduction method;
4. whether target samples or the normalizing constant are required;
5. whether the output is exact, asymptotically exact, weighted, or variational;
6. adaptation schedule and stopping rule;
7. warmup/evaluation separation;
8. final frozen-kernel invariance argument;
9. training and evaluation costs separately;
10. failure cases and objective-gaming examples.

## Primary references guiding the design

- Wang, Chen, Kanagawa, and Oates, *Reinforcement Learning for Adaptive MCMC*.
- Wang, Fisher, Kanagawa, Chen, and Oates, *Harnessing the Power of Reinforcement Learning
  for Adaptive MCMC*.
- Titsias and Dellaportas, *Gradient-based Adaptive Markov Chain Monte Carlo*.
- Levy, Hoffman, and Sohl-Dickstein, *Generalizing Hamiltonian Monte Carlo with Neural
  Networks*.
- Liu and Wang, *Stein Variational Gradient Descent: A General Purpose Bayesian Inference
  Algorithm*.
- Gorham and Mackey, *Measuring Sample Quality with Kernels*.
- Hoffman et al., *NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo Using Neural
  Transport*.
- Brofos et al., *Adaptation of the Independent Metropolis-Hastings Sampler with Normalizing
  Flow Proposals*.

These references motivate the phase; the core implementation remains deliberately smaller,
transparent, and independently testable.
