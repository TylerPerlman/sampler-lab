# Testing and Reproducibility

Sampler Lab separates deterministic correctness checks from statistical evidence.

## Validation layers

- Algebraic identities, exact finite-state calculations, and invariance checks catch structural
  errors without relying on a lucky random stream.
- Unit tests cover validation, shapes, state retention, counters, and deterministic edge cases.
- Statistical tests use fixed seeds and run in isolated processes so one test file cannot consume
  another file's random stream.
- The README quick-start block is extracted and executed by pytest.
- CI measures unit-test coverage and publishes the current badge with the documentation site.
- A weekly scheduled CI run installs the current supported dependency versions and catches drift
  before an ordinary user encounters it.

## NumPy stream changes

A fixed seed is reproducible for a particular generator algorithm and dependency environment; it is
not a promise that every future NumPy feature release will emit the same stream. Statistical tests
therefore assert mathematical properties with tolerances rather than pinning long output arrays.

When an upstream RNG or numerical change breaks a statistical assertion:

1. Reproduce the failure on the old and new dependency sets.
2. Check deterministic identities and exact-reference comparisons first.
3. Estimate the assertion's false-positive rate across independent seeds.
4. Loosen or re-center a tolerance only when both implementations remain mathematically correct.
5. Record the rationale in the pull request and changelog when behavior visible to users changes.

A seed is evidence control, not a proof. Statistical thresholds should be wide enough for any
correct supported stream and tight enough to detect the intended failure mode.

## Coverage interpretation

The published percentage measures the non-statistical pytest suite. Statistical files are executed
separately because process isolation is part of their reliability contract. Coverage is a map of
exercised code, not a substitute for mathematical validation; a line can be covered while an
invariant is still wrong.
