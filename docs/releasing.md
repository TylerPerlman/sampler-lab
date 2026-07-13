# Releases and Installation

## Distribution policy

Sampler Lab is currently distributed from GitHub, not PyPI. Users can install the latest default
branch with:

```bash
python -m pip install "git+https://github.com/TylerPerlman/sampler-lab.git"
```

For reproducible research, prefer a release tag or commit SHA:

```bash
python -m pip install "git+https://github.com/TylerPerlman/sampler-lab.git@v0.12.0"
```

The project metadata exposes homepage, documentation, repository, changelog, and issue URLs to
package tooling. PyPI publishing may be added later with trusted publishing and OIDC; until that
workflow exists, documentation must not imply that `pip install sampler-lab` resolves from PyPI.

## Version source

`src/sampler_lab/_version.py` is the single source of truth. Setuptools reads that value dynamically,
`sampler_lab.__version__` re-exports it, CI compares it with installed distribution metadata, and the
release workflow verifies that a tag named `vX.Y.Z` matches it. Historical changelog and benchmark
artifacts retain the version they describe.

## Release checklist

1. Update `_version.py` and `CHANGELOG.md` in the same pull request.
2. Run the publication check, quality checks, tests, and strict docs build.
3. Merge to `main` with all required checks green.
4. Create and push the matching annotated tag, for example `v0.12.0`.
5. Confirm the release workflow attaches both the wheel and source distribution.
6. Optionally archive the release with Zenodo, then add the DOI to `CITATION.cff`.

The release workflow intentionally creates GitHub Releases only. That constraint is explicit rather
than leaving users to guess where packages are published.
