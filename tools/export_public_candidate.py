"""Create a clean working copy suitable for a new root commit."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

EXCLUDED = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "site",
}


def ignored(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in EXCLUDED or name.endswith((".pyc", ".pyo"))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--initialize-git", action="store_true")
    parser.add_argument("--author-name", default="Tyler Perlman")
    parser.add_argument(
        "--author-email",
        default="66510084+TylerPerlman@users.noreply.github.com",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise SystemExit(f"destination already exists: {destination}")

    shutil.copytree(source, destination, ignore=ignored)
    subprocess.run(
        [
            sys.executable,
            str(destination / "tools" / "publication_check.py"),
            "--root",
            str(destination),
        ],
        check=True,
    )
    if args.initialize_git:
        subprocess.run(["git", "-C", str(destination), "init", "-b", "main"], check=True)
        subprocess.run(["git", "-C", str(destination), "add", "--all"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(destination),
                "-c",
                f"user.name={args.author_name}",
                "-c",
                f"user.email={args.author_email}",
                "commit",
                "-m",
                "Initial public release candidate",
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(destination / "tools" / "publication_check.py"),
                "--root",
                str(destination),
                "--history",
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
