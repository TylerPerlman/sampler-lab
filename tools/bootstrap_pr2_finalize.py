from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

NOTEBOOKS = (
    "03_particle_methods.ipynb",
    "04_finite_state_markov_theory.ipynb",
    "05_metropolis_gibbs_and_ising.ipynb",
    "06_annealed_paths_and_free_energy.ipynb",
)

SUPPORTING_SYMBOLS = {
    "docs/notebooks/03_particle_methods.ipynb": [
        "sampler_lab.spawn_rngs",
        "sampler_lab.models.count_self_avoiding_walks",
        "sampler_lab.models.sample_self_avoiding_walks",
    ],
    "docs/notebooks/04_finite_state_markov_theory.ipynb": [
        "sampler_lab.spawn_rngs",
        "sampler_lab.models.deterministic_cycle",
        "sampler_lab.models.ring_cosine_observable",
        "sampler_lab.models.ring_random_walk",
    ],
    "docs/notebooks/05_metropolis_gibbs_and_ising.ipynb": [
        "sampler_lab.OperationCounter",
        "sampler_lab.spawn_rngs",
        "sampler_lab.diagnostics.empirical_integrated_autocorrelation_time",
        "sampler_lab.models.GaussianTarget",
        "sampler_lab.models.IsingModel",
        "sampler_lab.models.RandomScanIsingMetropolisKernel",
        "sampler_lab.models.deterministic_sweep_ising_gibbs",
        "sampler_lab.models.exact_ising_distribution",
        "sampler_lab.models.random_scan_ising_gibbs",
    ],
    "docs/notebooks/06_annealed_paths_and_free_energy.ipynb": [
        "sampler_lab.spawn_rngs",
        "sampler_lab.models.IsingGibbsPopulationTransition",
        "sampler_lab.models.IsingModel",
        "sampler_lab.models.exact_ising_distribution",
        "sampler_lab.particles.ParticleCloud",
        "sampler_lab.particles.SystematicResampler",
    ],
}


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def copy_and_patch_notebooks(artifact_root: Path) -> None:
    source_dir = artifact_root / "docs" / "notebooks"
    target_dir = Path("docs/notebooks")
    target_dir.mkdir(parents=True, exist_ok=True)

    for name in NOTEBOOKS:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = target_dir / name
        shutil.copyfile(source, target)

        notebook = json.loads(target.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source_text = "".join(cell.get("source", []))
            source_text = source_text.replace("\x08eta", r"\beta")
            if name == "06_annealed_paths_and_free_energy.ipynb":
                source_text = source_text.replace(
                    "probe = np.ones((3, lattice_size, lattice_size))",
                    "probe = np.ones((lattice_size, lattice_size))",
                )
            cell["source"] = source_text.splitlines(keepends=True)
        write_text(target, json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")


def update_manifest() -> None:
    path = Path("docs/notebooks/manifest.toml")
    text = path.read_text(encoding="utf-8")

    for notebook_path, symbols in SUPPORTING_SYMBOLS.items():
        marker = f'path = "{notebook_path}"'
        start = text.index(marker)
        end = text.find("\n[[notebooks]]", start)
        if end == -1:
            end = len(text)
        block = text[start:end]
        block = block.replace('status = "planned"', 'status = "implemented"', 1)
        formatted = "supporting_symbols = [\n" + "".join(
            f'  "{symbol}",\n' for symbol in symbols
        ) + "]"
        block = block.replace("supporting_symbols = []", formatted, 1)
        text = text[:start] + block + text[end:]

    write_text(path, text)


def update_notebook_index() -> None:
    text = """# Tutorial notebooks

These executed notebooks are the visual pedagogy layer for Sampler Lab. They use only public package
namespaces, fixed named seeds, and method-appropriate diagnostics. The console demos remain the
NumPy-only, print-based reproduction layer; notebooks add derivations, plots, parameter studies, and
controlled failures.

Seven tutorials now cover the foundations, sequential methods, and discrete-state Markov methods.
The remaining continuous-dynamics, learned, rare-event, and benchmark families stay assigned in the
machine-checked manifest.

## Start here

| Notebook | Use it to learn | Matching console surface |
|---|---|---|
| [Orientation and diagnostics](00_orientation_and_diagnostics.ipynb) | Distinguish IID, weighted, correlated, and approximate output; interpret RMSE, weight ESS, IAT, and chain ESS | Repository-wide diagnostic conventions |
| [Exact and IID sampling](01_exact_and_iid_sampling.ipynb) | Use inversion, Box--Muller, transformations, and rejection sampling; compare direct and rejected unit-disk draws | `sampler-lab-disk-benchmark --seed 2022` |
| [Importance sampling](02_importance_sampling.ipynb) | Compare standard and self-normalized estimators, diagnose weights, improve a Gaussian rare-event estimate, and expose product-space collapse | `sampler-lab-importance-demo --seed 2022 --threshold 4` |
| [Particle methods](03_particle_methods.ipynb) | Inspect weighted clouds, compare resampling variance, grow self-avoiding walks, and measure genealogical collapse | `sampler-lab-particle-demo --steps 10 --particles 20000` |
| [Finite-state Markov theory](04_finite_state_markov_theory.ipynb) | Compute invariance, reversibility, spectra, autocorrelation, Poisson solutions, and exact finite-sample variance | `sampler-lab-markov-demo --states 12 --samples 240 --replicates 2000` |
| [Metropolis, Gibbs, and Ising](05_metropolis_gibbs_and_ising.ipynb) | Tune random-walk MH, preserve rejected states, validate small Ising systems exactly, and expose metastability | `sampler-lab-ising-demo --sizes 6 --betas 0.3 0.44 0.6` |
| [Annealed paths and free energy](06_annealed_paths_and_free_energy.ipynb) | Estimate partition-function ratios with AIS and annealed SMC; inspect work tails, ESS, and schedule resolution | `sampler-lab-annealing-demo --size 2 --target-beta 0.6` |

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
"""
    write_text(Path("docs/notebooks/index.md"), text)


def insert_after_heading(path: str, block: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if block.strip() in text:
        return
    first_break = text.index("\n") + 1
    text = text[:first_break] + "\n" + block.rstrip() + "\n" + text[first_break:]
    write_text(file_path, text)


def update_docs_links() -> None:
    mkdocs = Path("mkdocs.yml")
    text = mkdocs.read_text(encoding="utf-8")
    needle = "      - Importance sampling: notebooks/02_importance_sampling.ipynb\n"
    additions = """      - Particle methods: notebooks/03_particle_methods.ipynb
      - Finite-state Markov theory: notebooks/04_finite_state_markov_theory.ipynb
      - Metropolis, Gibbs, and Ising: notebooks/05_metropolis_gibbs_and_ising.ipynb
      - Annealed paths and free energy: notebooks/06_annealed_paths_and_free_energy.ipynb
"""
    if additions not in text:
        text = text.replace(needle, needle + additions, 1)
    write_text(mkdocs, text)

    insert_after_heading(
        "docs/methods/particle_methods.md",
        """!!! tip "Visual tutorial"
    Work through [Particle methods](../notebooks/03_particle_methods.ipynb) for executed
    resampling, ancestry, and self-avoiding-walk experiments using the public API.""",
    )
    insert_after_heading(
        "docs/methods/markov_theory.md",
        """!!! tip "Visual tutorial"
    Work through [Finite-state Markov theory](../notebooks/04_finite_state_markov_theory.ipynb)
    for exact operator calculations, simulated validation, and Poisson-error predictions.""",
    )
    insert_after_heading(
        "docs/methods/gibbs_metropolis.md",
        """!!! tip "Visual tutorial"
    Work through [Metropolis, Gibbs, and Ising](../notebooks/05_metropolis_gibbs_and_ising.ipynb)
    for proposal tuning, jump-chain failure, exact small-lattice checks, and metastability.""",
    )
    insert_after_heading(
        "docs/methods/annealed_paths.md",
        """!!! tip "Visual tutorial"
    Work through [Annealed paths and free energy](../notebooks/06_annealed_paths_and_free_energy.ipynb)
    for AIS, annealed SMC, Jarzynski work tails, schedule studies, and path reweighting.""",
    )


def update_readme() -> None:
    path = Path("README.md")
    text = path.read_text(encoding="utf-8")
    old = """The [executed tutorial notebooks](docs/notebooks/index.md) are the visual pedagogy layer. Start with
[orientation and diagnostics](docs/notebooks/00_orientation_and_diagnostics.ipynb), then continue to
[exact and IID sampling](docs/notebooks/01_exact_and_iid_sampling.ipynb) or
[importance sampling](docs/notebooks/02_importance_sampling.ipynb).
"""
    new = """The [executed tutorial notebooks](docs/notebooks/index.md) are the visual pedagogy layer. Start with
[orientation and diagnostics](docs/notebooks/00_orientation_and_diagnostics.ipynb), then continue
through exact and IID sampling, importance sampling, particles, finite-state Markov theory,
Metropolis/Gibbs/Ising, and annealed paths.
"""
    if old not in text:
        raise ValueError("README tutorial paragraph changed unexpectedly")
    write_text(path, text.replace(old, new, 1))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: bootstrap_pr2_finalize.py ARTIFACT_ROOT")
    artifact_root = Path(sys.argv[1])
    copy_and_patch_notebooks(artifact_root)
    update_manifest()
    update_notebook_index()
    update_docs_links()
    update_readme()


if __name__ == "__main__":
    main()
