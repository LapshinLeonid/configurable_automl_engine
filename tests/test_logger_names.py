"""Static check: every module logs under its own name (``getLogger(__name__)``).

A logger name written as a string literal drifts when a module moves (the
records then carry the name of a module that no longer exists), so no
module of the package may call ``getLogger`` with a string literal (an
f-string included).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import configurable_automl_engine

PACKAGE_ROOT = Path(configurable_automl_engine.__file__).parent
MODULES = sorted(PACKAGE_ROOT.rglob("*.py"))


def _literal_logger_calls(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "getLogger":
            continue
        arg = node.args[0]
        if isinstance(arg, (ast.Constant, ast.JoinedStr)):
            lines.append(node.lineno)
    return lines


@pytest.mark.parametrize(
    "path", MODULES, ids=[str(p.relative_to(PACKAGE_ROOT)) for p in MODULES]
)
def test_no_literal_logger_names(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = _literal_logger_calls(tree)
    assert not lines, (
        f"{path.relative_to(PACKAGE_ROOT)}: getLogger with a string literal at "
        f"lines {lines}; use logging.getLogger(__name__)"
    )


def test_scanner_detects_literal_names() -> None:
    tree = ast.parse(
        "import logging\n"
        "a = logging.getLogger('x')\n"
        "b = logging.getLogger(f'{x}.y')\n"
        "c = logging.getLogger(__name__)\n"
    )
    assert _literal_logger_calls(tree) == [2, 3]
