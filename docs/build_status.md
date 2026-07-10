# Build Status

## Implemented through v0.10.0

### Phase 0 - Infrastructure

- `src/` package layout and project metadata;
- explicit RNG construction and child-stream spawning;
- operation counters;
- stable `logsumexp` and log-weight normalization;
- target, independent-sampler, Markov-kernel, and resampler protocols;
- GitHub Actions configuration for Ruff, mypy, and pytest.

### Phase 1 - Foundations and exact sampling

- IID estimates and empirical error decomposition;
- inverse-CDF and finite generalized-inverse sampling;
- polar transforms and Box-Muller normals;
- generic log-domain rejection sampling with envelope validation;
- normalized Gaussian target with gradient and Hessian;
- direct and rejection unit-disk samplers;
- exact-sampling documentation and disk benchmark CLI.

### Phase 2 - Importance sampling

- explicit log-weight construction with proposal-support checks;
- standard and self-normalized importance estimates;
- delta-method standard errors and SNIS bias approximation;
- stable normalizing-constant ratio estimates;
- weighted moments and weight-quality diagnostics;
- scale-invariant chi-squared and order-two Renyi divergence estimates;
- Gaussian upper-tail proposal comparison;
- high-dimensional Gaussian weight-collapse experiment;
- `sampler-lab-importance-demo` CLI.

### Phase 3 - Particle methods

- immutable particle clouds with normalized log weights;
- separate pre-resampling and post-resampling histories;
- generic sequential proposal interface and SIS recursion;
- stable products of incremental normalizing-constant estimates;
- multinomial resampling;
- systematic resampling;
- floor-plus-Bernoulli resampling with variable population size;
- exact marginal offspring-variance formulas;
- offspring-count and unique-parent diagnostics;
- ESS-triggered resampling;
- variable-population ancestry maps, lineage tracing, and unique-ancestor histories;
- explicit particle-extinction errors;
- vectorized Rosenbluth self-avoiding-walk growth;
- exact depth-first enumeration for small walk lengths;
- `sampler-lab-particle-demo` comparison CLI.

### Phase 4 - Finite-state Markov theory

- row-stochastic transition operators acting on functions and measures;
- exact operator duality checks;
- communicating and closed-class decomposition;
- extreme invariant distributions and uniqueness checks;
- irreducibility, class periods, aperiodicity, and finite-state ergodicity;
- global-balance and detailed-balance residuals;
- stationary time reversal;
- discrete generators `L = P - I`;
- centered Poisson equations with residual diagnostics;
- exact finite-path martingale decompositions;
- exact lag autocovariances and autocorrelations;
- exact finite-sample variance of stationary time averages;
- exact asymptotic variance and integrated autocorrelation time;
- Poincare, absolute spectral, and singular-value gap diagnostics;
- finite-state Gibbs/partial-resampling kernels;
- random-scan mixtures, deterministic compositions, and coordinate relabeling;
- reversible, directed, and periodic ring-chain examples;
- `sampler-lab-markov-demo` CLI with replicated finite-sample validation.

### Phase 5 - Gibbs and Metropolis--Hastings

- generic Metropolis--Hastings with explicit forward and reverse proposal terms;
- retained repeated states after rejection;
- immutable chain trajectories with transition-level diagnostics;
- diagonal Gaussian random-walk and independence proposals;
- coordinate Gaussian and state-dependent Gaussian proposals;
- random-scan, deterministic-sweep, arbitrary-block, and transformed Gibbs kernels;
- FFT empirical autocovariances and initial-positive-sequence IAT estimates;
- periodic square-lattice Ising targets with exact local conditionals;
- O(1) local single-spin Metropolis ratios;
- exact small-lattice enumeration and partition functions;
- exact random-scan Gibbs, deterministic-sweep Gibbs, and Metropolis matrices;
- cost-normalized Ising benchmark with acceptance, magnetization, energy, IAT,
  ESS per spin update, and ESS per second;
- `sampler-lab-ising-demo` CLI.

### Phase 6 - Annealed weighting and free energy

- validated linear, power, and custom annealing schedules;
- geometric and functional paths of unnormalized distributions;
- optional batched path-density evaluation;
- stable incremental path log weights and cumulative reduced work;
- Jarzynski/AIS normalization-ratio estimates;
- dimensionless free-energy estimates with IID delta-method diagnostics;
- annealed SMC with every-stage or ESS-triggered resampling;
- separate pre-resampling and post-mutation particle histories;
- path-space ancestry and full genealogical work histories;
- intermediate-state reweighting toward later path laws;
- vectorized deterministic Gibbs population sweeps for Ising paths;
- exact small-lattice partition-ratio and magnetization validation;
- `sampler-lab-annealing-demo` path-length and resampling CLI.

### Phase 7 - Generator-based dynamics and Langevin methods

- repeated-start empirical estimates of scaled discrete generators;
- local drift, raw second-moment, and centered covariance-rate estimates;
- analytic Itô diffusion-generator evaluation;
- generic Euler--Maruyama kernels with rectangular diffusion factors;
- constant and position-dependent positive-definite preconditioners;
- explicit row-divergence corrections for state-dependent metrics;
- unadjusted Langevin and full-covariance MALA kernels;
- exact Gaussian ULA transition, stability boundary, and Lyapunov covariance;
- exact Gaussian invariant-law KL bias and linear-observable IAT;
- Poisson-equation invariant-bias estimation;
- small-step random-walk Metropolis generator experiment;
- conditioned-Gaussian ULA/MALA comparison CLI.

### Phase 8 - Hamiltonian and underdamped methods

- immutable position--momentum states and explicit Gaussian mass matrices;
- canonical Hamiltonian systems, vector fields, energies, and operation accounting;
- constant and position-dependent skew fields with row-divergence correction;
- exact Gaussian Hamiltonian frequencies and flow matrices;
- exact Gaussian leapfrog matrices and stability thresholds;
- velocity-Verlet / leapfrog trajectories with reused interior gradients;
- numerical reversibility, finite-difference volume, and symplectic diagnostics;
- unadjusted Hamiltonian trajectories and fresh-momentum HMC;
- partial Gaussian momentum refreshment and persistent generalized HMC;
- deterministic involution protocol with explicit log-Jacobian corrections;
- momentum-flip involutions and transformed rejection states;
- exact Ornstein--Uhlenbeck momentum substeps;
- skew/symmetric underdamped generator decomposition;
- BAOAB underdamped Langevin splitting;
- Metropolized underdamped dynamics with momentum-flipped rejections;
- periodic square-lattice XY target, exact gradient, toroidal wrapping, and magnetization;
- one-site von Mises exact-response validation;
- conditioned-Gaussian and XY `sampler-lab-hamiltonian-demo` CLI.

### Phase 9 - Conditioning and affine-invariant ensembles

- invertible affine maps with point, batch, inverse, composition, and covariance transforms;
- pushforward targets with exact log-density, gradient, and Hessian formulas;
- Gaussian whitening, spectral condition numbers, and exact block-conditionals;
- centered finite-difference Hessians from target gradients;
- positive-definite Hessian repair by rejection, clipping, or absolute eigenvalues;
- repair spectra and correction-norm diagnostics;
- stochastic-Newton local Gaussian proposals with explicit overdamped-Langevin scaling;
- specialized Metropolized stochastic Newton with reused endpoint geometry;
- full-covariance Gaussian random-walk proposals;
- immutable complete-ensemble states, transitions, and trajectories;
- explicit affine-span rank and degenerate-population rejection;
- sequential and split Goodman--Weare stretch moves;
- sequential and split symmetric walk moves from complementary walkers;
- per-walker acceptance, partner, cross-walker dependence, and ensemble ESS diagnostics;
- exact coupled affine-equivariance tests for both ensemble moves;
- parameterized Rosenbrock target with analytic derivatives, exact hierarchy, and exact moments;
- conditioned-Gaussian and Rosenbrock `sampler-lab-geometry-demo` CLI.

### Phase 10 - Adaptive and policy-gradient sampling

- Welford running moments and regularized covariance estimates;
- Robbins--Monro schedules, diminishing-adaptation diagnostics, and dual averaging;
- explicit expanding warmup windows and detached serializable frozen policies;
- adaptive full-covariance random-walk warmup with fresh post-freeze evaluation chains;
- categorical softmax and bounded Gaussian proposal policies with analytic scores;
- zero, running-mean, and linear policy-gradient baselines;
- REINFORCE returns, exact categorical Fisher matrices, damped natural gradients, and
  local policy-KL trust regions;
- acceptance-only failure baselines, accepted and feature-space jump rewards;
- generalized-speed and contrastive-divergence lower-bound objectives;
- operation-cost-normalized rewards with separate training and evaluation counters;
- exact frozen state-independent kernel mixtures and density-correct state-dependent proposal
  mixtures;
- reverse- and forward-KL diagonal Gaussian proposal fitting;
- exact independence-MH correction of frozen variational proposals;
- inverse-multiquadric KSD and Stein variational gradient descent;
- normalized, exactly sampleable anisotropic Gaussian-mixture, rotated funnel, and bimodal-
  funnel targets with analytic derivatives;
- capability-aware benchmark registration, exact reference sampling, mode diagnostics, MMD,
  and machine-readable benchmark results;
- `sampler-lab-policy-demo` objective-gaming and separated-mixture CLI.

### Phase 11 - Rare events and small-noise importance sampling

- exact anisotropic Gaussian halfspace and symmetric two-sided rare-event problems;
- exact probabilities, rate functions, and one or two dominating points;
- stable standard-normal log survival probabilities in extreme tails;
- ordinary multidimensional Laplace approximations with Hessian prefactors;
- Gaussian boundary-Laplace approximations with explicit small-noise prefactors;
- log-domain nonnegative-contribution estimates with absolute and relative uncertainty;
- contribution ESS, maximum-contribution, event-count, and operation diagnostics;
- exponential relative-variance rate fitting against `1 / epsilon`;
- shifted Gaussian linear exponential twisting with exact likelihood ratios;
- exact shifted-proposal second moments for one- and two-sided events;
- centered Gaussian temperature broadening with exact second moments;
- fixed-covariance asymptotic temperature schedules and exact grid selection;
- finite mixtures of Gaussian twists with stable `logsumexp` densities;
- deterministic one-dimensional quadrature for the symmetric-mixture second moment;
- deliberate single-twist failure on a two-dominating-point event;
- `sampler-lab-rare-event-demo` text and JSON CLI.

### Release 0.12.0 - Cross-method benchmark and release polish

- common validated adapter output across IID, weighted, particle, Markov, ensemble, variational,
  deterministic-particle, and learned methods;
- complete capability-aware adapter matrix for all compatible continuous samplers;
- exact-reference correlated Gaussian, anisotropic mixture, rotated funnel, and bimodal-funnel
  target hierarchy;
- weighted and unweighted mean, covariance, IMQ-MMD, mode, and funnel diagnostics;
- separate training and frozen-evaluation operation counters and wall clocks;
- deterministic seed derivation with shared per-replicate reference samples;
- replicated aggregates with standard errors and accuracy-time Pareto summaries;
- explicit incompatibility exclusions and preserved runtime failures;
- JSON, flattened CSV, Markdown, manifest, and optional figure report artifacts;
- `sampler-lab-benchmark` selection, quick-run, listing, JSON, and figure CLI;
- public API, benchmarking semantics, changelog, and capability, provenance, and reference docs.

## Validation at v0.12.0

- Ruff lint: clean;
- Ruff formatting: clean across 196 files;
- mypy strict mode: clean across 122 package source files;
- pytest: 279 passing tests (242 unit and 37 statistical);
- package compilation: clean;
- editable installation: successful;
- installed package reports version `0.12.0`;
- all twelve installed console entry points resolved successfully;
- the reference benchmark completed 108 method/target/replicate runs with two explicit
  incompatibility exclusions and zero runtime failures;
- JSON, CSV, Markdown, pairing matrix, manifest, and twelve PNG figures were generated.

Statistical files were run in isolated processes, and two historically troublesome aggregate files
were split by test node, because the environment still shows the documented pytest/plugin shutdown
stall. All 37 statistical assertions completed successfully.

## Maintenance target

The planned method families are implemented. Future work should be treated as maintenance or
explicitly scoped extension, with the same exactness, documentation, operation-accounting,
and failure-mode standards.
