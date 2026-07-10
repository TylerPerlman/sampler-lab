# Security Policy

## Supported versions

Until Sampler Lab reaches 1.0, security fixes are applied to the latest release and the `main`
branch. Older tags remain reproducible historical snapshots but are not maintained independently.

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability.

Use GitHub's private **Report a vulnerability** flow when it is available in the repository's
Security tab. If private vulnerability reporting is not enabled, contact the repository owner
privately through GitHub before sharing exploit details.

Include:

- the affected version or commit;
- a minimal reproduction;
- the expected security impact;
- whether the issue affects local execution, generated artifacts, CI, or dependency handling;
- any suggested mitigation.

This project is a scientific and educational library, not a hardened network service. Reports about
unsafe deserialization, command execution, dependency compromise, workflow permissions, or release
artifact integrity are nevertheless taken seriously.

## Scope notes

Numerical instability, biased estimators, and incorrect convergence claims are scientific-correctness
bugs rather than conventional security vulnerabilities. They should normally be reported as issues,
unless they can be used to cross a trust boundary or execute unintended code.
