"""Format notebooks and normalize execution counts plus transient metadata."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

TRANSIENT_NOTEBOOK_METADATA = {"signature", "widgets"}
TRANSIENT_CELL_METADATA = {"ExecuteTime", "collapsed", "execution", "scrolled"}
TRANSIENT_OUTPUT_METADATA = {"filenames", "isolated", "needs_background", "transient"}


def notebook_paths(root: Path, supplied: list[Path]) -> list[Path]:
    if supplied:
        paths = [path if path.is_absolute() else root / path for path in supplied]
    else:
        paths = sorted((root / "docs/notebooks").glob("*.ipynb"))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"notebook paths do not exist: {missing}")
    return paths


def remove_keys(mapping: dict[str, Any], keys: set[str]) -> None:
    for key in keys:
        mapping.pop(key, None)


def normalize(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    metadata = notebook.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: notebook metadata must be an object")
    remove_keys(metadata, TRANSIENT_NOTEBOOK_METADATA)
    language_info = metadata.get("language_info")
    if isinstance(language_info, dict):
        language_info.pop("version", None)

    execution_count = 1
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError(f"{path}: cells must be a list")
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError(f"{path}: every cell must be an object")
        cell_metadata = cell.setdefault("metadata", {})
        if not isinstance(cell_metadata, dict):
            raise ValueError(f"{path}: cell metadata must be an object")
        remove_keys(cell_metadata, TRANSIENT_CELL_METADATA)
        if cell.get("cell_type") != "code":
            continue
        cell["execution_count"] = execution_count
        outputs = cell.get("outputs", [])
        if not isinstance(outputs, list):
            raise ValueError(f"{path}: code-cell outputs must be a list")
        for output in outputs:
            if not isinstance(output, dict):
                raise ValueError(f"{path}: every output must be an object")
            output_metadata = output.get("metadata")
            if isinstance(output_metadata, dict):
                remove_keys(output_metadata, TRANSIENT_OUTPUT_METADATA)
            if output.get("output_type") == "execute_result":
                output["execution_count"] = execution_count
        execution_count += 1

    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    paths = notebook_paths(root, args.paths)
    relative_paths = [str(path.relative_to(root)) for path in paths]
    subprocess.run(["ruff", "format", *relative_paths], cwd=root, check=True)
    for path in paths:
        normalize(path)
    print(f"Normalized {len(paths)} notebook(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
