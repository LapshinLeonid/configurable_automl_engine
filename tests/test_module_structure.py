"""Structural limits of the package that keep the modules reviewable.

The large engine modules were split into cohesive modules; these tests stop
them from growing back:

- no module of the package has more than ``MAX_MODULE_LINES`` code lines,
  with no exception (``MODULE_SIZE_EXCEPTIONS`` is empty). Blank lines,
  comments, and docstrings are not counted, so documentation does not
  inflate the metric;
- the classes listed in ``MAX_CLASS_LINES`` stay within their limit;
- ruff ``C901`` is enabled with ``max-complexity = 15`` and the package
  has no ``# noqa: C901`` exception (``ALLOWED_C901_NOQA`` is empty).
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
import tomllib
from pathlib import Path

import pytest

import configurable_automl_engine

PACKAGE_ROOT = Path(configurable_automl_engine.__file__).parent
PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

#: Code lines (comments and docstrings excluded) allowed per module.
#: Calibrated from the current maximum: ``experiment/reporting/markdown_reporter.py``
#: has 652 code lines, so the limit leaves a small headroom without forcing a
#: split.
MAX_MODULE_LINES = 700

#: Modules exempt from ``MAX_MODULE_LINES``. Empty: the limit has no
#: exception (``test_module_size_limit_has_no_exceptions``).
MODULE_SIZE_EXCEPTIONS: frozenset[str] = frozenset()

#: Class (module path, class name) → maximum number of lines.
MAX_CLASS_LINES = {
    ("hpo/runner.py", "_OptimizeRunner"): 800,
    ("trainer.py", "ModelTrainer"): 1300,
}

#: Functions that may keep a ``# noqa: C901`` marker, as (module path,
#: function name). Empty: the complexity limit has no exception.
ALLOWED_C901_NOQA: frozenset[tuple[str, str]] = frozenset()


def _modules() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if p.name != "_version.py")


def _rel(path: Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Return the line numbers occupied by module, class, and function docstrings."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


_NON_CODE_TOKENS = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
    }
)


def _code_line_numbers(path: Path) -> set[int]:
    """Return physical line numbers with code, ignoring blanks, comments, docstrings.

    Args:
        path: Module to measure.

    Returns:
        Line numbers that contain at least one code token.

    Raises:
        SyntaxError: The module cannot be parsed.
    """
    source = path.read_text(encoding="utf-8")
    docstrings = _docstring_lines(ast.parse(source))
    code_lines: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in _NON_CODE_TOKENS:
            continue
        end_line = token.end[0] if token.end[1] > 0 else token.end[0] - 1
        span = range(token.start[0], end_line + 1)
        if token.type == tokenize.STRING and all(line in docstrings for line in span):
            continue
        code_lines.update(span)
    return code_lines


def _count_code_lines(path: Path) -> int:
    """Count physical lines with code, ignoring blanks, comments, and docstrings.

    Args:
        path: Module to measure.

    Returns:
        Number of physical lines that contain at least one code token.

    Raises:
        SyntaxError: The module cannot be parsed.
    """
    return len(_code_line_numbers(path))


def test_module_size_limit_has_no_exceptions() -> None:
    """No module is exempt from the line limit."""
    assert not MODULE_SIZE_EXCEPTIONS


def test_count_code_lines_ignores_comments_and_docstrings(
    tmp_path: Path,
) -> None:
    """Comments and every kind of docstring are excluded."""
    module = tmp_path / "sample.py"
    module.write_text(
        '"""Module docstring.\n'
        "\n"
        "More text.\n"
        '"""\n'
        "\n"
        "# comment only\n"
        "VALUE = 1  # trailing comment\n"
        "\n"
        "\n"
        "class Example:\n"
        '    """Class docstring."""\n'
        "\n"
        "    def method(self):\n"
        '        """Method docstring."""\n'
        "        return VALUE\n"
        "\n"
        "    async def async_method(self):\n"
        '        """Coroutine docstring."""\n'
        "        return VALUE\n",
        encoding="utf-8",
    )
    assert _code_line_numbers(module) == {7, 10, 13, 15, 17, 19}


def test_count_code_lines_counts_non_docstring_string_literals(
    tmp_path: Path,
) -> None:
    """String literals that are data, not docstrings, remain code lines."""
    module = tmp_path / "sample.py"
    module.write_text(
        'TEXT = """\nfirst\nsecond\n"""\n',
        encoding="utf-8",
    )
    assert _code_line_numbers(module) == {1, 2, 3, 4}


@pytest.mark.parametrize("path", _modules(), ids=_rel)
def test_module_size_limit(path: Path) -> None:
    """Every module stays within the code-line limit."""
    n_lines = _count_code_lines(path)
    assert n_lines <= MAX_MODULE_LINES, (
        f"{_rel(path)} has {n_lines} code lines "
        f"(limit {MAX_MODULE_LINES}); "
        "split it into cohesive modules"
    )


@pytest.mark.parametrize(("module", "cls"), sorted(MAX_CLASS_LINES))
def test_class_size_limit(module: str, cls: str) -> None:
    """The large classes stay within their line limit."""
    tree = ast.parse((PACKAGE_ROOT / module).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
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
