# Tutorial notebooks

These executed notebooks are the visual pedagogy layer for Sampler Lab. They use only public package
namespaces, fixed named seeds, and method-appropriate diagnostics. The console demos remain the
NumPy-only, print-based reproduction layer; notebooks add derivations, plots, parameter studies, and
controlled failures.

Ten tutorials now cover the foundations, sequential methods, Markov methods, continuous dynamics,
and geometry-aware sampling. The learned, rare-event, and benchmark families remain assigned in
the machine-checked manifest.

## Start here

| Notebook | Use it to learn | Matching console surface |
|---|---|---|
| [Orientation and diagnostics](00_orientation_and_diagnostics.ipynb) | Distinguish IID, weighted, correlated, and approximate output; interpret RMSE, weight ESS, IAT, and chain ESS | Repository-wide diagnostic conventions |
| [Exact and IID sampling](01_exact_and_iid_sampling.ipynb) | Use inversion, Box–Muller, transformations, and rejection sampling; compare direct and rejected unit-disk draws | `sampler-lab-disk-benchmark --seed 2022` |
| [Importance sampling](02_importance_sampling.ipynb) | Compare standard and self-normalized estimators, diagnose weights, improve a Gaussian rare-event estimate, and expose product-space collapse | `sampler-lab-importance-demo --seed 2022 --threshold 4` |
| [Particle methods](03_particle_methods.ipynb) | Inspect weighted clouds, compare resampling variance, grow self-avoiding walks, and measure genealogical collapse | `sampler-lab-particle-demo --steps 10 --particles 20000` |
| [Finite-state Markov theory](04_finite_state_markov_theory.ipynb) | Compute invariance, reversibility, spectra, autocorrelation, Poisson solutions, and exact finite-sample variance | `sampler-lab-markov-demo --states 12 --samples 240 --replicates 2000` |
| [Metropolis, Gibbs, and Ising](05_metropolis_gibbs_and_ising.ipynb) | Tune random-walk MH, preserve rejected states, validate small Ising systems exactly, and expose metastability | `sampler-lab-ising-demo --sizes 6 --betas 0.3 0.44 0.6` |
| [Annealed paths and free energy](06_annealed_paths_and_free_energy.ipynb) | Estimate partition-function ratios with AIS and annealed SMC; inspect work tails, ESS, and schedule resolution | `sampler-lab-annealing-demo --size 2 --target-beta 0.6` |
| [Langevin sampling](07_langevin_sampling.ipynb) | Diagnose diffusion limits, ULA stability and bias, MALA correction, and preconditioning | `sampler-lab-langevin-demo --condition-numbers 1 10 100` |
| [Hamiltonian and underdamped dynamics](08_hamiltonian_and_underdamped_dynamics.ipynb) | Inspect phase trajectories, leapfrog errors, mass conditioning, HMC, BAOAB, and metropolized underdamped dynamics | `sampler-lab-hamiltonian-demo --condition-numbers 1 10 100` |
| [Geometry, conditioning, and ensembles](09_geometry_conditioning_and_ensembles.ipynb) | Whiten Gaussians, condition exactly, repair Hessians, and compare stochastic Newton with stretch and walk ensembles | `sampler-lab-geometry-demo --condition-numbers 1 10 100 --walkers 24` |

The remaining notebook families are assigned in [`manifest.toml`](manifest.toml). The manifest is a
machine-checked roadmap: every public sampling namespace is either covered by an implemented
notebook or assigned to a later notebook.

## Reproduce locally

```bash
python -m pip install -e '.[dev,notebooks]'
jupyter lab docs/notebooks
```

Publication outputs are committed. CI re-executes notebooks with
`SAMPLER_LAB_NOTEBOOK_MODE=quick`; quick mode changes sample counts, not the mathematical checks.
Use `python tools/normalize_notebooks.py` before committing an updated notebook.

## Reading conventions

Each tutorial labels whether its output is exact IID, exact-invariant Markov, weighted, or
approximate. Plots state the expected behavior before displaying results and interpret quantitative
checks afterward. A pretty cloud is evidence of successful plotting, not successful sampling.
