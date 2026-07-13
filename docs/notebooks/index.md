# Tutorial notebooks

These executed notebooks are the visual pedagogy layer for Sampler Lab. They use only public package
namespaces, fixed named seeds, and method-appropriate diagnostics. The console demos remain the
NumPy-only, print-based reproduction layer; notebooks add derivations, plots, parameter studies, and
controlled failures.

The current pilot establishes the notebook standard with three tutorials before the remaining method
families are implemented in later pull requests.

## Start here

| Notebook | Use it to learn | Matching console surface |
|---|---|---|
| [Orientation and diagnostics](00_orientation_and_diagnostics.ipynb) | Distinguish IID, weighted, correlated, and approximate output; interpret RMSE, weight ESS, IAT, and chain ESS | Repository-wide diagnostic conventions |
| [Exact and IID sampling](01_exact_and_iid_sampling.ipynb) | Use inversion, Box-Muller, transformations, and rejection sampling; compare direct and rejected unit-disk draws | `sampler-lab-disk-benchmark --seed 2022` |
| [Importance sampling](02_importance_sampling.ipynb) | Compare standard and self-normalized estimators, diagnose weights, improve a Gaussian rare-event estimate, and expose product-space collapse | `sampler-lab-importance-demo --seed 2022 --threshold 4` |

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
