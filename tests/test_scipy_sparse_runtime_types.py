"""SciPy sparse matrix types must not be subscripted at runtime.

The engine supports SciPy releases whose sparse matrix classes do not
implement ``__class_getitem__``. A subscript such as
``sp.csr_matrix[np.float64]`` therefore raises ``TypeError`` when it is
evaluated. Such subscripts are allowed only inside annotations deferred by
``from __future__ import annotations``; any other occurrence, including the
first argument of ``typing.cast``, is executed at import or call time.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

SPARSE_MATRIX_NAMES = frozenset(
    {
        "spmatrix",
        "csr_matrix",
        "csc_matrix",
        "coo_matrix",
        "lil_matrix",
        "dok_matrix",
        "bsr_matrix",
        "dia_matrix",
        "csr_array",
        "csc_array",
        "coo_array",
        "lil_array",
        "dok_array",
        "bsr_array",
        "dia_array",
    }
)


def _defers_annotations(tree: ast.Module) -> bool:
    """Return whether the module enables postponed evaluation of annotations."""
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "__future__"
            and any(alias.name == "annotations" for alias in node.names)
        ):
            return True
    return False


def _annotation_nodes(tree: ast.Module) -> set[int]:
    """Return ids of every AST node inside an annotation."""
    roots: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.arg, ast.AnnAssign)) and node.annotation is not None:
            roots.append(node.annotation)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.returns is not None
        ):
            roots.append(node.returns)
    ids: set[int] = set()
    for root in roots:
        ids.update(id(child) for child in ast.walk(root))
    return ids


def _is_sparse_matrix_attribute(node: ast.expr) -> bool:
    """Return whether the expression names a SciPy sparse matrix class."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr in SPARSE_MATRIX_NAMES
        and isinstance(node.value, ast.Name)
        and node.value.id in {"sp", "sparse"}
    )


def _type_checking_nodes(tree: ast.Module) -> set[int]:
    """Return ids of every AST node guarded by ``if TYPE_CHECKING``."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        is_guard = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
            isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
        )
        if not is_guard:
            continue
        for child in node.body:
            ids.update(id(descendant) for descendant in ast.walk(child))
    return ids


def test_sparse_matrix_generics_are_never_evaluated() -> None:
    """No SciPy sparse matrix is subscripted outside deferred annotations."""
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        deferred = _defers_annotations(tree)
        annotation_nodes = _annotation_nodes(tree)
        guarded_nodes = _type_checking_nodes(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript):
                continue
            if not _is_sparse_matrix_attribute(node.value):
                continue
            if id(node) in guarded_nodes:
                continue
            if deferred and id(node) in annotation_nodes:
                continue
            offenders.append(f"{path.relative_to(SRC.parent)}:{node.lineno}")
    assert not offenders, "SciPy sparse generics evaluated at runtime: " + ", ".join(
        offenders
    )
