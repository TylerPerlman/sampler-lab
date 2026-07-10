# Cross-Method Continuous Sampling Benchmark Plan

## Status

Implemented in v0.12.0. Phase 10 supplied the target hierarchy and learned-method foundation;
Phase 12 completed the adapter matrix, replicated execution, explicit exclusions and failures,
weighted and unweighted exact-reference metrics, target-specific funnel diagnostics, separate
training/evaluation costs, JSON/CSV/Markdown report bundles, Pareto summaries, figures, and the
`sampler-lab-benchmark` command. The expensive full suite remains opt-in rather than a stochastic
CI leaderboard.

## Purpose

The benchmark suite should reveal *why* methods differ, not compress every sampler into one
misleading leaderboard. It will compare every compatible continuous sampler under common
targets, initializations, operation counts, seeds, and evaluation metrics.

Not every repository method can act on the same problem. Self-avoiding-walk samplers and
Ising kernels are discrete and model-specific; exact disk samplers target a different support;
Jarzynski methods estimate normalization ratios rather than merely producing a stationary
chain. The benchmark registry must therefore declare capabilities and exclusions instead of
quietly comparing incompatible objects.

## Target hierarchy

### 1. Correlated Gaussian oracle

A normalized Gaussian with configurable dimension, rotation, and condition number.

Purpose:

- exact moments, scores, Hessians, and direct samples;
- exact or nearly exact behavior for several algorithms;
- deterministic checks of whitening and affine equivariance;
- cheap detection of implementation bugs before harder targets.

This is a sanity target, not the headline challenge.

### 2. Separated anisotropic Gaussian mixture

A balanced two-component mixture with:

- means separated by a configurable number of within-mode standard deviations;
- rotated, unequal anisotropic covariances;
- exact normalized log density and score;
- exact independent sampling;
- exact mode labels for reference samples.

Purpose:

- expose local-chain mode trapping;
- show that acceptance and local ESJD can look healthy while global mode mass is wrong;
- demonstrate reverse-KL mode collapse;
- test local/global mixture policies, independence proposals, annealing, and SMC.

Recommended default:

```text
dimension: 8
mode separation: 10 marginal standard deviations
condition number within each mode: 25
mixture weights: 0.5 / 0.5
```

Also provide a two-dimensional visualization preset.

### 3. Rotated anisotropic Gaussian funnel

Use a Neal-style hierarchy:

\[
V\sim N(0,\sigma_V^2),\qquad
Z_i\mid V=v\sim N\left(0, s_i^2 e^v\right),
\]

followed by a fixed orthogonal rotation. The scale factors \(s_i\) form a geometric sequence,
creating anisotropy in addition to funnel curvature.

Required capabilities:

- exact direct sampling from the hierarchy;
- normalized log density;
- analytic score;
- analytic or validated Hessian;
- centered and non-centered coordinate representations.

Purpose:

- expose the narrow-neck/wide-mouth geometry;
- distinguish affine conditioning from genuinely position-dependent geometry;
- compare RWM, MALA, HMC, stochastic Newton, ensembles, transport maps, and learned policies;
- test whether a method samples both the neck and tails rather than only matching means.

Recommended defaults:

```text
dimensions: 10 and 25
sigma_V: 3
anisotropy ratio: 20
rotation: fixed from a seeded QR factorization
```

### 4. Bimodal anisotropic funnel capstone

Construct an equally weighted mixture of two separated, rotated anisotropic funnels. Each
component remains exactly sampleable and normalized. This combines the two pathologies:

- long-distance mode discovery;
- severe within-mode curvature and scale variation.

This is the main common benchmark for compatible continuous methods. It is intentionally
hard enough that failure is informative. Component targets remain in the suite because a
single capstone score cannot diagnose whether a sampler failed from mode isolation, local
geometry, or both.

## Capability registry

Every benchmark adapter must declare:

```text
produces_samples
produces_weighted_samples
is_markov_chain
is_exact_after_freeze
requires_normalized_density
requires_log_density
requires_gradient
requires_hessian
requires_conditionals
requires_initial_reference_samples
supports_multimodality
supports_unbounded_continuous_space
```

Every target declares the corresponding access it provides. The registry either constructs a
valid pairing or returns an explicit exclusion reason.

The benchmark should include all compatible implementations from:

- exact independent sampling, as an oracle;
- rejection sampling where a valid practical envelope exists;
- standard and self-normalized importance sampling;
- sequential importance sampling, AIS, and annealed SMC;
- random-walk and independence MH;
- Gibbs only where a valid conditional decomposition exists;
- ULA and MALA;
- HMC and underdamped methods;
- stochastic Newton;
- stretch and walk ensembles;
- variational Gaussian/mixture proposals;
- SVGD;
- policy-gradient learned MH kernels.

Discrete and model-specific samplers remain in their own exact benchmarks and are listed as
not applicable rather than assigned nonsense scores.

## Initialization protocol

Report at least two regimes:

1. **Cold start**: all chains begin in one mode or in the funnel mouth.
2. **Dispersed start**: chains begin from an overdispersed reference distribution.

Oracle-initialized runs may be included only to measure stationary efficiency. They must not
replace cold-start convergence tests.

Training, warmup, and evaluation seeds are disjoint. Learned methods train on one set of
rollouts and are evaluated after freezing on fresh chains and fresh seeds.

## Metrics

### Distributional accuracy

- mean and covariance error against exact values;
- error in declared nonlinear expectations;
- component mass / mode-occupancy error;
- funnel-neck, central, and tail probability errors;
- energy distance or MMD against exact reference samples;
- inverse-multiquadric kernel Stein discrepancy when scores are available;
- bias and RMSE across independent replicates.

### Markov-chain mixing

- IAT and ESS for coordinates, log density, radius/scale, and mode indicator;
- first-passage time between modes;
- number of round trips;
- longest mode residence;
- split-chain between/within diagnostics;
- acceptance and standardized accepted squared jump distance.

ESS for a coordinate must never substitute for mode-occupancy diagnostics.

### Weighted and particle methods

- weight ESS, entropy, and maximum weight;
- normalizing-constant error when available;
- particle ancestry and unique ancestors;
- mode coverage before and after resampling;
- estimator-specific variance and relative error.

### Variational and deterministic particle methods

- reverse-KL estimate when tractable;
- KSD and reference-sample discrepancy;
- mode coverage;
- approximation bias;
- optimization iterations and target/gradient cost.

Do not report chain IAT for SVGD particles as though they were a stationary Markov chain.

### Cost

Normalize by:

- log-density evaluations;
- proposal-density evaluations;
- gradient evaluations;
- Hessian evaluations;
- matrix factorizations;
- policy evaluations;
- training objective evaluations;
- wall-clock time.

Report training and evaluation costs separately, plus an amortized total for several declared
numbers of downstream samples.

## Reporting

Produce:

- a machine-readable JSON result for every run;
- a long-form CSV table;
- Pareto plots of accuracy versus cost;
- target-specific diagnostic panels;
- a method-capability table with exclusions;
- replicated confidence intervals;
- no single overall winner score.

Methods should be grouped by required target access. Comparing RWM to HMC is useful, but the
report must display that HMC consumed gradients.

## Test strategy

The full benchmark is an opt-in experiment, not a fragile CI race.

### CI regression tests

- exact target normalization and direct-sampling moments;
- analytic score/Hessian versus finite differences;
- capability-registry inclusion/exclusion logic;
- deterministic metric identities;
- fixed-seed smoke runs for every compatible adapter;
- broad failure-mode checks, such as a deliberately tiny-step RWM showing high acceptance
  and poor mode switching.

### Statistical benchmark tests

- replicated runs with confidence intervals;
- broad thresholds derived from exact reference uncertainty;
- no required total ordering between close methods;
- explicit expected-failure profiles for pathological method/target pairs.

### Headline comparison

For the bimodal anisotropic funnel, compare at minimum:

- isotropic RWM;
- adapted covariance RWM;
- MALA;
- HMC;
- stochastic Newton;
- stretch ensemble;
- walk ensemble;
- AIS/annealed SMC;
- reverse-KL learned independence proposal with MH correction;
- KSD/SVGD transport followed by an exact correction where applicable;
- policy-gradient mixture of local and global MH proposals.

The benchmark should highlight that:

- high acceptance can coexist with no meaningful movement;
- local methods can have excellent within-mode ESS and wrong global weights;
- reverse KL can select one mode;
- affine methods can solve anisotropy but not funnel curvature;
- transport or local geometry can solve curvature but not necessarily mode isolation;
- annealing and learned global proposals can address mode transitions at additional cost.

These are hypotheses to test, not conclusions to hard-code.
