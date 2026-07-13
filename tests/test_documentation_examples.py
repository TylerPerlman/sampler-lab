"""Executable guards for user-facing documentation examples."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _python_block_after_heading(markdown: str, heading: str) -> str:
    heading_start = markdown.index(f"## {heading}")
    fence = "```python\n"
    code_start = markdown.index(fence, heading_start) + len(fence)
    code_end = markdown.index("\n```", code_start)
    return markdown[code_start:code_end]


def test_readme_quick_start_executes() -> None:
    readme = ROOT / "README.md"
    code = _python_block_after_heading(readme.read_text(encoding="utf-8"), "Quick start")
    namespace = {"__name__": "__sampler_lab_readme_example__"}
    exec(compile(code, str(readme), "exec"), namespace)
