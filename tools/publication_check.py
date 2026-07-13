"""Validate that a release candidate contains only independently publishable material."""

from __future__ import annotations

import argparse
import codecs
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SKIP_PARTS = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "site",
}
SKIP_FILES = {"publication_check.py"}
TEXT_EXTENSIONS = {
    "",
    ".cfg",
    ".cff",
    ".csv",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
BLOCKED_BINARY_EXTENSIONS = {
    ".7z",
    ".bz2",
    ".doc",
    ".docx",
    ".gz",
    ".key",
    ".odp",
    ".odt",
    ".pdf",
    ".ppt",
    ".pptx",
    ".rar",
    ".tar",
    ".tgz",
    ".xz",
    ".zip",
}
ENCODED_TERMS = [
    "Wbanguna Jrner",
    "ALH",
    "Pbhenag",
    "Snyy 2022",
    "pbhefr abgrf",
    "thrfg yrpgher",
    "ubzrjbex",
    "nffvtazrag",
    "flyynohf",
]
WORD_PATTERNS = ["pbhefr", "yrpgher", "rkrepvfr"]
NUMBERED_WORDS = ["puncgre", "rkrepvfr"]
DATE_PATTERN = re.compile(r"\b(?:Sep|Oct|Nov)\.?\s+\d{1,2},?\s+2022\b", re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    path: str
    detail: str


def decoded(value: str) -> str:
    return codecs.decode(value, "rot_13")


def patterns() -> list[re.Pattern[str]]:
    values = [re.escape(decoded(value)) for value in ENCODED_TERMS]
    values.extend(rf"\b{re.escape(decoded(value))}\b" for value in WORD_PATTERNS)
    values.extend(rf"\b{re.escape(decoded(value))}\s+\d+\b" for value in NUMBERED_WORDS)
    compiled = [re.compile(value, re.IGNORECASE) for value in values]
    compiled.append(DATE_PATTERN)
    return compiled


PATTERNS = patterns()


def should_skip(path: Path) -> bool:
    return path.name in SKIP_FILES or any(part in SKIP_PARTS for part in path.parts)


def scan_text(name: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for pattern in PATTERNS:
            match = pattern.search(line)
            if match:
                findings.append(
                    Finding(name, f"line {line_number}: blocked phrase {match.group(0)!r}")
                )
    return findings


def notebook_source(raw_source: Any) -> str | None:
    if isinstance(raw_source, str):
        return raw_source
    if isinstance(raw_source, list) and all(isinstance(line, str) for line in raw_source):
        return "".join(raw_source)
    return None


def scan_notebook(name: str, text: str) -> list[Finding]:
    try:
        notebook = json.loads(text)
    except json.JSONDecodeError as exc:
        return [Finding(name, f"invalid notebook JSON: {exc}")]
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return [Finding(name, "notebook cells must be a list")]

    findings: list[Finding] = []
    for cell_number, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict) or cell.get("cell_type") not in {"code", "markdown"}:
            continue
        source = notebook_source(cell.get("source"))
        if source is None:
            findings.append(Finding(name, f"cell {cell_number}: source must be text or text lines"))
            continue
        findings.extend(scan_text(f"{name}:cell-{cell_number}", source))
    return findings


def read_utf8(path: Path, name: str) -> tuple[str | None, list[Finding]]:
    try:
        return path.read_text(encoding="utf-8"), []
    except UnicodeDecodeError:
        return None, [Finding(name, "expected text file is not valid UTF-8")]


def scan_tree(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or should_skip(path.relative_to(root)):
            continue
        relative = path.relative_to(root)
        name = str(relative)
        findings.extend(scan_text(name, name))
        suffix = path.suffix.lower()
        if suffix in BLOCKED_BINARY_EXTENSIONS:
            findings.append(Finding(name, "blocked binary document type"))
            continue
        if suffix == ".ipynb":
            text, read_findings = read_utf8(path, name)
            findings.extend(read_findings)
            if text is not None:
                findings.extend(scan_notebook(name, text))
            continue
        if suffix not in TEXT_EXTENSIONS:
            continue
        text, read_findings = read_utf8(path, name)
        findings.extend(read_findings)
        if text is not None:
            findings.extend(scan_text(name, text))
    return findings


def scan_archive(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory)
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                archive.extractall(destination)
        elif tarfile.is_tarfile(path):
            with tarfile.open(path) as archive:
                archive.extractall(destination, filter="data")
        else:
            return findings
        findings.extend(scan_tree(destination))
    return [Finding(f"{path.name}:{finding.path}", finding.detail) for finding in findings]


def scan_history(root: Path) -> list[Finding]:
    result = subprocess.run(
        ["git", "-C", str(root), "log", "--all", "--format=%H%n%B", "-p", "--no-ext-diff"],
        check=True,
        capture_output=True,
        text=True,
    )
    return scan_text("git-history", result.stdout)


def scan_artifact_text(path: Path) -> list[Finding]:
    name = str(path)
    suffix = path.suffix.lower()
    if suffix == ".ipynb":
        text, findings = read_utf8(path, name)
        if text is not None:
            findings.extend(scan_notebook(name, text))
        return findings
    if suffix not in TEXT_EXTENSIONS:
        return []
    text, findings = read_utf8(path, name)
    if text is not None:
        findings.extend(scan_text(name, text))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--artifacts", type=Path, nargs="*", default=[])
    args = parser.parse_args()

    root = args.root.resolve()
    findings = scan_tree(root)
    if args.history:
        findings.extend(scan_history(root))
    for artifact_path in args.artifacts:
        if not artifact_path.exists():
            continue
        paths = [artifact_path] if artifact_path.is_file() else artifact_path.rglob("*")
        for path in paths:
            if not path.is_file():
                continue
            archive_findings = scan_archive(path)
            if archive_findings:
                findings.extend(archive_findings)
                continue
            findings.extend(scan_artifact_text(path))

    if findings:
        for finding in findings:
            print(f"{finding.path}: {finding.detail}", file=sys.stderr)
        return 1
    print("Publication hygiene check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
