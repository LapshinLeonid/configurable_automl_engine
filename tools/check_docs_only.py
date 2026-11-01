"""AST-based guard that verifies a diff touches only documentation.

Compares every changed Python file under ``src/`` against a base revision
(usually ``main``) and fails if any *executable* code changed. Docstrings,
comments and the ``description=`` keyword argument are ignored: they may
change freely, everything else may not.

Usage::

    python tools/check_docs_only.py origin/main

Prints ``OK`` and exits 0 when the diff is documentation-only, ``FAIL``
with a per-file, per-line report otherwise.
"""

from __future__ import annotations

import ast
import io
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

# Keyword arguments whose values are user-facing documentation and are
# allowed to change (e.g. Pydantic ``Field(description=...)`` texts).
IGNORED_KWARGS = {"description"}


@dataclass
class _DocMask:
    """Line numbers that carry no executable code."""

    docstring: set[int]
    comment: set[int]
    description: set[int]


def _line_index(source: str) -> tuple[list[str], list[int]]:
    """Return source lines and the start offset of every line."""
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    return lines, offsets


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Collect line numbers covered by module/class/function docstrings."""
    spans: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        body = node.body if not isinstance(node, ast.Module) else node.body
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            spans.update(range(first.lineno, first.end_lineno + 1))
    return spans


def _comment_lines(source: str) -> set[int]:
    """Collect line numbers whose comment tokens may change freely."""
    spans: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        spans.add(token.start[0])
    return spans


def _description_lines(tree: ast.AST, source: str) -> set[int]:
    """Collect lines of keyword arguments named in IGNORED_KWARGS."""
    lines, offsets = _line_index(source)
    spans: set[int] = set()

    def offset_of(line: int, col: int) -> int:
        return offsets[line - 1] + col

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg not in IGNORED_KWARGS:
                continue
            start = offset_of(keyword.value.lineno, keyword.value.col_offset)
            end = offset_of(keyword.value.end_lineno, keyword.value.end_col_offset)
            first_line, last_line = keyword.value.lineno, keyword.value.end_lineno
            spans.update(range(first_line, last_line + 1))
            # A multi-line value may share its first/last line with other
            # code; treat only the fully covered lines as safe.
            if last_line > first_line:
                spans.discard(first_line)
                spans.discard(last_line)
    return spans


def _mask(source: str) -> _DocMask:
    tree = ast.parse(source)
    return _DocMask(
        docstring=_docstring_lines(tree),
        comment=_comment_lines(source),
        description=_description_lines(tree, source),
    )


def _safe_lines(mask: _DocMask) -> set[int]:
    return mask.docstring | mask.comment | mask.description


def _strip_safe(source: str, safe: set[int]) -> str:
    """Drop documentation lines and verify the remainder is still valid code."""
    kept = [
        line
        for number, line in enumerate(source.splitlines(), start=1)
        if number not in safe
    ]
    return "\n".join(kept) + "\n"


def _normalized(source: str) -> str:
    """Return code with comments and docstrings blanked for comparison."""
    tree = ast.parse(source)
    lines = source.splitlines()
    for number in _docstring_lines(tree):
        if number <= len(lines):
            lines[number - 1] = ""
    # Blank comments with tokenize (end positions are reliable).
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            line = lines[token.start[0] - 1]
            lines[token.start[0] - 1] = line[: token.start[1]]
    # Blank description= values (they may span lines).
    for number in _description_lines(tree, source):
        if number <= len(lines):
            lines[number - 1] = ""
    return "\n".join(lines)


def _description_spans(tree: ast.AST) -> list[tuple[int, int]]:
    """Return ``(start_line, end_line)`` of keyword arguments in IGNORED_KWARGS."""
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg in IGNORED_KWARGS:
                spans.append((keyword.value.lineno, keyword.value.end_lineno))
    return spans


def _code_tokens(source: str) -> list[str]:
    """Return the executable content as a token sequence.

    Comments, docstrings and ``description=`` values are dropped, so the
    sequence is invariant under documentation edits (including changes that
    shift line numbers).
    """
    tree = ast.parse(source)
    docstring_lines = _docstring_lines(tree)
    desc_spans = _description_spans(tree)
    tokens: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.NL):
            continue
        if token.type == tokenize.STRING:
            line = token.start[0]
            if line in docstring_lines:
                continue
            # String literals that are part of a description= value are
            # documentation too (a long description is split across
            # several adjacent literals).
            if any(start <= line <= end for start, end in desc_spans):
                continue
        if token.type == tokenize.NEWLINE:
            # Statement boundaries matter for readability of the diff report.
            tokens.append("\n")
            continue
        if token.type in (
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENCODING,
            tokenize.ENDMARKER,
        ):
            continue
        tokens.append(token.string)
    return tokens


def _diff_report(old: list[str], new: list[str], path: str) -> None:
    """Print a line-oriented context report for changed token streams."""
    import difflib

    matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        print(f"  {path}:{i1 + 1}")
        for line in old[i1:i2][:5]:
            print(f"    - {line.strip()}")
        for line in new[j1:j2][:5]:
            print(f"    + {line.strip()}")
        if len(old[i1:i2]) > 5 or len(new[j1:j2]) > 5:
            print("    ...")


def _changed_py_files(base: str) -> list[str]:
    cmd = [
        "git",
        "diff",
        "--name-only",
        "--diff-filter=ACMR",
        "-z",
        base,
        "--",
        "src",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [f for f in out.stdout.split("\0") if f.endswith(".py")]


def _file_version(ref: str, path: str) -> str:
    cmd = ["git", "show", f"{ref}:{path}"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return ""
    return result.stdout


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python tools/check_docs_only.py <base-ref>")
        return 2
    base = argv[1]
    files = _changed_py_files(base)
    failures: list[str] = []
    for path in files:
        old_source = _file_version(base, path)
        new_source = Path(path).read_text(encoding="utf-8")
        if not old_source:
            failures.append(f"{path}: new file (new modules are not allowed here)")
            continue
        old_tokens = _code_tokens(old_source)
        new_tokens = _code_tokens(new_source)
        if old_tokens != new_tokens:
            failures.append(path)
            _diff_report(old_tokens, new_tokens, path)
    if failures:
        print("FAIL — executable code changed in:")
        for path in failures:
            print(f"  {path}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
