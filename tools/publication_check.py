"""Validate that a release candidate contains only independently publishable material."""

from __future__ import annotations

import argparse
import codecs
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

SKIP_PARTS = {
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


def scan_tree(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or should_skip(path.relative_to(root)):
            continue
        relative = path.relative_to(root)
        findings.extend(scan_text(str(relative), str(relative)))
        if path.suffix.lower() in BLOCKED_BINARY_EXTENSIONS:
            findings.append(Finding(str(relative), "blocked binary document type"))
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(Finding(str(relative), "expected text file is not valid UTF-8"))
            continue
        findings.extend(scan_text(str(relative), text))
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
            if path.suffix.lower() in TEXT_EXTENSIONS:
                try:
                    text = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                findings.extend(scan_text(str(path), text))

    if findings:
        for finding in findings:
            print(f"{finding.path}: {finding.detail}", file=sys.stderr)
        return 1
    print("Publication hygiene check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
