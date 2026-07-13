## Summary

Describe the method, fix, documentation change, or engineering improvement.

## Mathematical and numerical conventions

- State any sign, scaling, matrix-orientation, or normalization conventions.
- Mark the output as exact, corrected, weighted, or approximate where relevant.
- Explain how adaptation or training is separated from frozen evaluation.

## Validation

- [ ] `ruff check .`
- [ ] `ruff format --check .`
- [ ] `mypy src/sampler_lab`
- [ ] `python -m compileall -q src tests tools`
- [ ] `pytest -m "not statistical"`
- [ ] Relevant statistical tests run in isolation
- [ ] `mkdocs build --strict`

List the analytical identity, invariance property, exact enumeration, or fixed-seed regression used
as the primary oracle.

## Cost and API impact

Describe changes to operation counts, runtime, memory, public APIs, serialized formats, or CLI output.

## Documentation

- [ ] Public APIs and conventions are documented.
- [ ] The relevant method, benchmark, roadmap, or reference documentation is updated.
