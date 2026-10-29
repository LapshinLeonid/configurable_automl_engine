"""Structural limits of the package that keep the modules reviewable.

The large engine modules were split into cohesive modules; these tests stop
them from growing back:

- no module of the package is longer than ``MAX_MODULE_LINES`` lines
  (docstrings included), except the modules listed in
  ``MODULE_SIZE_EXCEPTIONS``;
- the classes listed in ``MAX_CLASS_LINES`` stay within their limit;
- ruff ``C901`` is enabled with ``max-complexity = 15`` and the package
  has no ``# noqa: C901`` exception (``ALLOWED_C901_NOQA`` is empty).
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

import configurable_automl_engine

PACKAGE_ROOT = Path(configurable_automl_engine.__file__).parent
PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

MAX_MODULE_LINES = 1500

#: Modules that are out of scope of the split (tracked separately).
MODULE_SIZE_EXCEPTIONS = {
    "training_engine/config_parser.py",
    "preprocessing.py",
}

#: Class (module path, class name) → maximum number of lines.
MAX_CLASS_LINES = {
    ("hpo/runner.py", "_OptimizeRunner"): 800,
    ("trainer.py", "ModelTrainer"): 1300,
}

#: Functions that may keep a ``# noqa: C901`` marker, as (module path,
#: function name). Empty: the complexity limit has no exception.
ALLOWED_C901_NOQA: frozenset[tuple[str, str]] = frozenset()


def _modules() -> list[Path]:
    return sorted(
        p for p in PACKAGE_ROOT.rglob("*.py") if p.name != "_version.py"
    )


def _rel(path: Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


@pytest.mark.parametrize("path", _modules(), ids=_rel)
def test_module_size_limit(path: Path) -> None:
    """Every module stays within the line limit (docstrings included)."""
    if _rel(path) in MODULE_SIZE_EXCEPTIONS:
        pytest.skip("out of scope of the module split (tracked separately)")
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    assert n_lines <= MAX_MODULE_LINES, (
        f"{_rel(path)} has {n_lines} lines (limit {MAX_MODULE_LINES}); "
        "split it into cohesive modules"
    )


@pytest.mark.parametrize(("module", "cls"), sorted(MAX_CLASS_LINES))
def test_class_size_limit(module: str, cls: str) -> None:
    """The large classes stay within their line limit."""
    tree = ast.parse((PACKAGE_ROOT / module).read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
    )
    n_lines = node.end_lineno - node.lineno + 1
    limit = MAX_CLASS_LINES[(module, cls)]
    assert n_lines <= limit, f"{cls} has {n_lines} lines (limit {limit})"


def test_ruff_complexity_limit_is_configured() -> None:
    """ruff C901 is enabled with max-complexity = 15."""
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    lint = config["tool"]["ruff"]["lint"]
    assert "C901" in lint.get("select", []) + lint.get("extend-select", [])
    assert lint["mccabe"]["max-complexity"] == 15


def test_no_c901_noqa_markers() -> None:
    """``src/`` has no ``# noqa: C901``: the complexity limit has no exception."""
    assert not ALLOWED_C901_NOQA
    found: list[str] = []
    for path in _modules():
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if re.search(r"#\s*noqa:.*\bC901\b", line):
                found.append(f"{_rel(path)}:{i + 1}")
    assert not found, f"noqa: C901 markers are not allowed: {found}"
