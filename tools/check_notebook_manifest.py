"""Validate notebook coverage, metadata, execution state, and repository size limits."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

ALLOWED_STATUSES = {"implemented", "planned"}
ALLOWED_MODES = {"quick-and-publication"}
ALLOWED_SEMANTICS = {
    "approximate",
    "diagnostic",
    "exact_iid",
    "exact_invariant",
    "mixed",
    "weighted",
}


def public_exports(namespace: str) -> tuple[ModuleType, set[str]]:
    module = importlib.import_module(namespace)
    raw_exports = getattr(module, "__all__", None)
    if not isinstance(raw_exports, (list, tuple)) or not all(
        isinstance(name, str) for name in raw_exports
    ):
        raise ValueError(f"{namespace} must define a string-only __all__")
    exports = set(raw_exports)
    missing = sorted(name for name in exports if not hasattr(module, name))
    if missing:
        raise ValueError(f"{namespace}.__all__ names missing attributes: {missing}")
    return module, exports


def expand_claim(claim: str, cache: dict[str, set[str]]) -> set[str]:
    if claim.endswith(".*"):
        namespace = claim[:-2]
        if namespace not in cache:
            _module, cache[namespace] = public_exports(namespace)
        return {f"{namespace}.{name}" for name in cache[namespace]}

    namespace, separator, symbol = claim.rpartition(".")
    if not separator or not namespace or not symbol:
        raise ValueError(f"invalid public-symbol claim: {claim!r}")
    if namespace not in cache:
        _module, cache[namespace] = public_exports(namespace)
    if symbol not in cache[namespace]:
        raise ValueError(f"{claim} is not exported by {namespace}.__all__")
    return {claim}


def validate_notebook(path: Path, size_limit: int) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"implemented notebook is missing: {path}"]
    size = path.stat().st_size
    if size > size_limit:
        errors.append(f"{path} is {size:,} bytes; limit is {size_limit:,}")

    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"cannot parse {path}: {exc}"]

    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        errors.append(f"{path} must be an nbformat 4 notebook")
        return errors
    metadata = notebook.get("metadata", {})
    if not isinstance(metadata, dict) or "kernelspec" not in metadata:
        errors.append(f"{path} must record a kernelspec")
    if isinstance(metadata, dict) and "widgets" in metadata:
        errors.append(f"{path} contains transient widget metadata")

    execution_counts: list[int] = []
    image_count = 0
    for cell_index, cell in enumerate(notebook["cells"], start=1):
        if not isinstance(cell, dict):
            errors.append(f"{path}: cell {cell_index} is not an object")
            continue
        if cell.get("cell_type") != "code":
            continue
        count = cell.get("execution_count")
        if not isinstance(count, int):
            errors.append(f"{path}: code cell {cell_index} is not executed")
        else:
            execution_counts.append(count)
        outputs = cell.get("outputs", [])
        if not isinstance(outputs, list):
            errors.append(f"{path}: code cell {cell_index} outputs are invalid")
            continue
        for output in outputs:
            if not isinstance(output, dict):
                errors.append(f"{path}: code cell {cell_index} has a non-object output")
                continue
            if output.get("output_type") == "error":
                errors.append(f"{path}: code cell {cell_index} has a committed error output")
            data = output.get("data", {})
            if isinstance(data, dict) and "image/png" in data:
                image_count += 1

    expected_counts = list(range(1, len(execution_counts) + 1))
    if execution_counts != expected_counts:
        errors.append(
            f"{path}: execution counts must be normalized to {expected_counts}, "
            f"found {execution_counts}"
        )
    if image_count == 0:
        errors.append(f"{path} has no committed PNG figure outputs")
    return errors


def validate_manifest(root: Path, manifest_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        return [f"cannot parse {manifest_path}: {exc}"]

    if manifest.get("version") != 1:
        errors.append("manifest version must be 1")
    size_limit = manifest.get("size_limit_bytes")
    if not isinstance(size_limit, int) or size_limit <= 0:
        errors.append("size_limit_bytes must be a positive integer")
        size_limit = 0

    namespaces = manifest.get("sampling_namespaces")
    if (
        not isinstance(namespaces, list)
        or not namespaces
        or not all(isinstance(namespace, str) for namespace in namespaces)
    ):
        return [*errors, "sampling_namespaces must be a nonempty string list"]
    if len(namespaces) != len(set(namespaces)):
        errors.append("sampling_namespaces contains duplicates")

    notebook_rows = manifest.get("notebooks")
    if not isinstance(notebook_rows, list) or not notebook_rows:
        return [*errors, "manifest must contain [[notebooks]] rows"]

    export_cache: dict[str, set[str]] = {}
    claimed_by: dict[str, list[str]] = defaultdict(list)
    paths_seen: set[str] = set()
    for row_index, raw_row in enumerate(notebook_rows, start=1):
        if not isinstance(raw_row, dict):
            errors.append(f"notebook row {row_index} is not a table")
            continue
        row: dict[str, Any] = raw_row
        path_value = row.get("path")
        family = row.get("family")
        status = row.get("status")
        mode = row.get("mode")
        semantics = row.get("output_semantics")
        coverage = row.get("coverage")
        supporting = row.get("supporting_symbols")

        label = path_value if isinstance(path_value, str) else f"row {row_index}"
        if not isinstance(path_value, str) or not path_value.endswith(".ipynb"):
            errors.append(f"{label}: path must be an .ipynb string")
            continue
        if path_value in paths_seen:
            errors.append(f"duplicate notebook path: {path_value}")
        paths_seen.add(path_value)
        if not isinstance(family, str) or not family.strip():
            errors.append(f"{label}: family must be a nonempty string")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{label}: status must be one of {sorted(ALLOWED_STATUSES)}")
        if mode not in ALLOWED_MODES:
            errors.append(f"{label}: mode must be one of {sorted(ALLOWED_MODES)}")
        if (
            not isinstance(semantics, list)
            or not semantics
            or not all(isinstance(value, str) for value in semantics)
        ):
            errors.append(f"{label}: output_semantics must be a nonempty string list")
        else:
            unknown_semantics = sorted(set(semantics) - ALLOWED_SEMANTICS)
            if unknown_semantics:
                errors.append(f"{label}: unknown output semantics {unknown_semantics}")
        if not isinstance(coverage, list) or not all(isinstance(value, str) for value in coverage):
            errors.append(f"{label}: coverage must be a string list")
            coverage = []
        if not isinstance(supporting, list) or not all(
            isinstance(value, str) for value in supporting
        ):
            errors.append(f"{label}: supporting_symbols must be a string list")
            supporting = []

        for claim in [*coverage, *supporting]:
            try:
                expanded = expand_claim(claim, export_cache)
            except (ImportError, ValueError) as exc:
                errors.append(f"{label}: {exc}")
                continue
            if claim in coverage:
                for symbol in expanded:
                    claimed_by[symbol].append(path_value)

        notebook_path = root / path_value
        if status == "implemented":
            errors.extend(validate_notebook(notebook_path, size_limit))
        elif notebook_path.exists():
            errors.append(f"{path_value}: status is planned but the notebook already exists")

    expected_symbols: set[str] = set()
    for namespace in namespaces:
        try:
            _module, exports = public_exports(namespace)
        except (ImportError, ValueError) as exc:
            errors.append(str(exc))
            continue
        export_cache[namespace] = exports
        expected_symbols.update(f"{namespace}.{name}" for name in exports)

    claimed_symbols = set(claimed_by)
    missing = sorted(expected_symbols - claimed_symbols)
    unknown = sorted(claimed_symbols - expected_symbols)
    duplicates = {symbol: paths for symbol, paths in claimed_by.items() if len(paths) > 1}
    if missing:
        errors.append(
            "public sampling symbols missing from notebook coverage: " + ", ".join(missing)
        )
    if unknown:
        errors.append("coverage contains non-sampling symbols: " + ", ".join(unknown))
    if duplicates:
        details = "; ".join(f"{symbol}: {paths}" for symbol, paths in sorted(duplicates.items()))
        errors.append("sampling symbols have multiple notebook homes: " + details)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/notebooks/manifest.toml"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    errors = validate_manifest(root, manifest_path)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("Notebook manifest check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
