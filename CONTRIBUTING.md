# Contributing to Sampler Lab

Sampler Lab favors transparent numerical implementations, explicit stochastic state, and tests that
validate mathematics rather than merely exercising code paths.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Python 3.11, 3.12, and 3.13 are supported.

## Required checks

Run the deterministic checks before opening a pull request:

```bash
python tools/publication_check.py --root .
ruff check .
ruff format --check .
mypy src/sampler_lab
python -m compileall -q src tests tools
pytest -m "not statistical"
mkdocs build --strict
```

Statistical tests use fixed seeds but are intentionally isolated because some pytest/plugin
combinations have stalled during aggregate shutdown:

```bash
for file in $(grep -rl '@pytest.mark.statistical' tests --include='test_*.py' | sort); do
  pytest "$file" -m statistical
done
```

GitHub Actions runs the same split on every pull request.

## Implementation standards

- Pass an explicit `numpy.random.Generator` to stochastic APIs.
- Prefer log densities and log weights for numerical stability.
- Return samples and diagnostics; do not hide rejection, adaptation, or resampling decisions.
- Separate training/warmup from frozen evaluation when exactness depends on freezing.
- Add deterministic algebraic or invariance tests before statistical regression tests.
- Record density, gradient, Hessian, factorization, policy, and random-draw costs when relevant.
- Keep exact, corrected, weighted, and approximate output semantics explicit.

## Pull requests

Keep changes focused and explain:

1. the mathematical method or engineering problem;
2. the convention used, including signs, scaling, and matrix orientation;
3. the exactness or approximation guarantee;
4. the tests used as an oracle;
5. any performance or numerical tradeoffs.

New public APIs should include type annotations, docstrings, tests, and an update to the relevant
method documentation. Benchmark additions must declare capabilities and explicit exclusion reasons
rather than forcing incompatible methods into a common score.

## Releasing

The package version in `pyproject.toml` must match the release tag. Pushing a tag such as `v0.13.0`
triggers the release workflow, which builds the wheel and source distribution and attaches them to
a GitHub Release.
