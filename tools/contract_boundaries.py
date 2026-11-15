"""Static guards for the contract/engine architecture boundary.

The module is shared by the pytest guards and the CI entry points.  It keeps
the checks independent from the engine runtime: contract imports are inspected
with :mod:`ast`, metadata is read from ``pyproject.toml``, and distribution
artifacts are inspected with :mod:`zipfile`/:mod:`tarfile`.
"""

from __future__ import annotations

import ast
import re
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = REPO_ROOT / "automl-contracts"
CONTRACT_SRC = CONTRACT_ROOT / "src" / "automl_contracts"
CONTRACT_PYPROJECT = CONTRACT_ROOT / "pyproject.toml"
ENGINE_SRC = REPO_ROOT / "src" / "configurable_automl_engine"
ENGINE_PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_CONFIG = REPO_ROOT / ".sourcecraft" / "ci.yaml"
GITHUB_CI_CONFIG = REPO_ROOT / ".github" / "workflows" / "CI.yaml"

ENGINE_MODULE = "configurable_automl_engine"
ALLOWED_CONTRACT_DEPENDENCIES = frozenset({"pydantic", "pydantic-core"})
ALLOWED_THIRD_PARTY_ROOTS = frozenset({"pydantic", "pydantic_core"})
FORBIDDEN_LIBRARY_ROOTS = frozenset(
    {
        "imblearn",
        "joblib",
        "matplotlib",
        "mlflow",
        "numpy",
        "optuna",
        "pandas",
        "pyarrow",
        "scipy",
        "seaborn",
        "shap",
        "sklearn",
        "tensorflow",
        "torch",
        "xgboost",
        "yaml",
    }
)
DYNAMIC_GETATTR_NAMES = frozenset(
    {
        "__import__",
        "eval",
        "exec",
        "find_spec",
        "import_module",
        "spec_from_file_location",
    }
)
DYNAMIC_IMPORT_CALLS = frozenset(
    {
        "__import__",
        "builtins.__import__",
        "builtins.eval",
        "builtins.exec",
        "eval",
        "exec",
        "import_module",
        "importlib.__import__",
        "importlib.import_module",
        "importlib.util.find_spec",
        "importlib.util.spec_from_file_location",
        "find_spec",
        "spec_from_file_location",
    }
)
RUNTIME_CALL_PREFIXES = (
    "builtins.open",
    "ctypes.",
    "io.open",
    "multiprocessing.",
    "os.access",
    "os.chdir",
    "os.chmod",
    "os.chown",
    "os.cpu_count",
    "os.getcwd",
    "os.get_exec_path",
    "os.getegid",
    "os.geteuid",
    "os.getgid",
    "os.getlogin",
    "os.getpid",
    "os.getppid",
    "os.getuid",
    "os.getenv",
    "os.getenvb",
    "os.listdir",
    "os.lstat",
    "os.readlink",
    "os.scandir",
    "os.makedirs",
    "os.mkdir",
    "os.open",
    "os.read",
    "os.write",
    "os.close",
    "os.fdopen",
    "os.pread",
    "os.pwrite",
    "os.dup",
    "os.dup2",
    "os.fsync",
    "os.ftruncate",
    "os.truncate",
    "os.fchmod",
    "os.fchown",
    "os.lseek",
    "os.sendfile",
    "os.splice",
    "os.copy_file_range",
    "os.system",
    "os.popen",
    "os.fork",
    "os.forkpty",
    "os.execv",
    "os.execve",
    "os.execvp",
    "os.execvpe",
    "os.execl",
    "os.execlp",
    "os.execlpe",
    "os.spawn",
    "os.posix_spawn",
    "os.kill",
    "os.killpg",
    "os.setuid",
    "os.setgid",
    "os.chroot",
    "os.ptrace",
    "os.putenv",
    "os.remove",
    "os.rename",
    "os.replace",
    "os.rmdir",
    "os.stat",
    "os.symlink",
    "os.uname",
    "os.walk",
    "os.path.abspath",
    "os.path.exists",
    "os.path.expanduser",
    "os.path.expandvars",
    "os.path.getatime",
    "os.path.getctime",
    "os.path.getmtime",
    "os.path.getsize",
    "os.path.isdir",
    "os.path.isfile",
    "os.path.islink",
    "os.path.ismount",
    "os.path.lexists",
    "os.path.realpath",
    "os.path.samefile",
    "asyncio.",
    "concurrent.",
    "ftplib.",
    "http.",
    "httplib.",
    "io.",
    "platform.",
    "psutil.",
    "requests.",
    "resource.",
    "signal.",
    "smtplib.",
    "socketserver.",
    "ssl.",
    "telnetlib.",
    "threading.",
    "urllib.",
    "urllib2.",
    "xmlrpc.",
    "shutil.copy",
    "shutil.copy2",
    "shutil.move",
    "shutil.rmtree",
    "shutil.which",
    "socket.",
    "subprocess.",
    "sqlite3.",
    "tempfile.",
)
TYPE_CONSTRUCTOR_NAMES = frozenset(
    {
        "Annotated",
        "AsyncIterator",
        "Callable",
        "ClassVar",
        "Coroutine",
        "Dict",
        "Final",
        "FrozenSet",
        "Generator",
        "Iterable",
        "Iterator",
        "List",
        "Literal",
        "Mapping",
        "Optional",
        "Sequence",
        "Set",
        "Tuple",
        "Type",
        "Union",
        "dict",
        "frozenset",
        "list",
        "set",
        "tuple",
        "type",
    }
)
PATH_PURE_METHODS = frozenset(
    {
        "__bytes__",
        "__eq__",
        "__fspath__",
        "__hash__",
        "__repr__",
        "__rtruediv__",
        "__str__",
        "__truediv__",
        "as_posix",
        "as_uri",
        "full_match",
        "is_absolute",
        "is_relative_to",
        "is_reserved",
        "joinpath",
        "match",
        "relative_to",
        "with_name",
        "with_segments",
        "with_stem",
        "with_suffix",
    }
)

PATH_CLASS_NAMES = frozenset({"Path", "pathlib.Path"})
PATH_RETURNING_METHODS = frozenset(
    {
        "absolute",
        "cwd",
        "expanduser",
        "home",
        "joinpath",
        "relative_to",
        "rename",
        "replace",
        "resolve",
        "with_name",
        "with_stem",
        "with_suffix",
    }
)
RUNTIME_ATTRIBUTES = frozenset(
    {
        "os.environ",
        "os.name",
        "sys.__stderr__",
        "sys.__stdin__",
        "sys.__stdout__",
        "sys.argv",
        "sys.base_prefix",
        "sys.exec_prefix",
        "sys.executable",
        "sys.getwindowsversion",
        "sys.implementation",
        "sys.meta_path",
        "sys.path",
        "sys.path_hooks",
        "sys.platform",
        "sys.prefix",
        "sys.stderr",
        "sys.stdin",
        "sys.stdout",
        "sys.version_info",
    }
)
SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


@dataclass(frozen=True)
class Violation:
    """One architecture-boundary violation."""

    rule: str
    path: Path
    line: int
    message: str

    def format(self, root: Path = REPO_ROOT) -> str:
        """Render the violation as a stable, repository-relative message."""
        try:
            relative = self.path.relative_to(root)
        except ValueError:
            relative = self.path
        return f"{relative}:{self.line}: [{self.rule}] {self.message}"


def python_files(root: Path) -> list[Path]:
    """Return all Python source files below ``root`` in deterministic order."""
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
        and ".mypy_cache" not in path.parts
        and ".ruff_cache" not in path.parts
    )


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _dotted_name(node: ast.AST) -> str | None:
    """Return the dotted spelling of a Name/Attribute expression."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _resolve_callable_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """Resolve a call target through local import aliases."""
    name = _dotted_name(node)
    if name is None:
        return None
    head, dot, tail = name.partition(".")
    resolved = aliases.get(head, head)
    return f"{resolved}.{tail}" if dot else resolved


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    """Map local import names to their fully qualified module paths."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _calls_in_expression(
    node: ast.AST, aliases: dict[str, str]
) -> Iterator[tuple[ast.Call, dict[str, str]]]:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            yield child, dict(aliases)


def _remove_target_aliases(targets: Sequence[ast.AST], aliases: dict[str, str]) -> None:
    for target in targets:
        if isinstance(target, ast.Name):
            aliases.pop(target.id, None)


def _assign_target_aliases(
    targets: Sequence[ast.AST], value: ast.AST, aliases: dict[str, str]
) -> None:
    resolved = None
    if isinstance(value, (ast.Name, ast.Attribute)):
        resolved = _resolve_callable_name(value, aliases)
    for target in targets:
        if not isinstance(target, ast.Name):
            continue
        if resolved is None:
            aliases.pop(target.id, None)
        else:
            aliases[target.id] = resolved


def _apply_scope_import(stmt: ast.stmt, aliases: dict[str, str]) -> None:
    if isinstance(stmt, ast.Import):
        for alias in stmt.names:
            local = alias.asname or alias.name.split(".")[0]
            aliases[local] = alias.name if alias.asname else alias.name.split(".")[0]
    elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module:
        for alias in stmt.names:
            aliases[alias.asname or alias.name] = f"{stmt.module}.{alias.name}"


def _iter_function_calls(
    stmt: ast.FunctionDef | ast.AsyncFunctionDef, aliases: dict[str, str]
) -> Iterator[tuple[ast.Call, dict[str, str]]]:
    for decorator in stmt.decorator_list:
        yield from _calls_in_expression(decorator, aliases)
    for default in stmt.args.defaults:
        yield from _calls_in_expression(default, aliases)
    for kw_default in stmt.args.kw_defaults:
        if kw_default is not None:
            yield from _calls_in_expression(kw_default, aliases)
    inner = dict(aliases)
    arguments = [
        *stmt.args.posonlyargs,
        *stmt.args.args,
        *stmt.args.kwonlyargs,
        stmt.args.vararg,
        stmt.args.kwarg,
    ]
    for argument in arguments:
        if argument is not None:
            inner.pop(argument.arg, None)
    yield from _iter_calls_with_aliases(stmt.body, inner)


def _iter_class_calls(
    stmt: ast.ClassDef, aliases: dict[str, str]
) -> Iterator[tuple[ast.Call, dict[str, str]]]:
    for decorator in stmt.decorator_list:
        yield from _calls_in_expression(decorator, aliases)
    for base in stmt.bases:
        yield from _calls_in_expression(base, aliases)
    for keyword in stmt.keywords:
        yield from _calls_in_expression(keyword.value, aliases)
    yield from _iter_calls_with_aliases(stmt.body, dict(aliases))


def _iter_assign_calls(
    stmt: ast.Assign, aliases: dict[str, str]
) -> Iterator[tuple[ast.Call, dict[str, str]]]:
    yield from _calls_in_expression(stmt.value, aliases)
    _assign_target_aliases(stmt.targets, stmt.value, aliases)


def _iter_ann_assign_calls(
    stmt: ast.AnnAssign, aliases: dict[str, str]
) -> Iterator[tuple[ast.Call, dict[str, str]]]:
    if stmt.value is not None:
        yield from _calls_in_expression(stmt.value, aliases)
        _assign_target_aliases([stmt.target], stmt.value, aliases)
    else:
        _remove_target_aliases([stmt.target], aliases)


def _iter_aug_assign_calls(
    stmt: ast.AugAssign, aliases: dict[str, str]
) -> Iterator[tuple[ast.Call, dict[str, str]]]:
    yield from _calls_in_expression(stmt.value, aliases)
    _remove_target_aliases([stmt.target], aliases)


def _iter_if_calls(
    stmt: ast.If, aliases: dict[str, str]
) -> Iterator[tuple[ast.Call, dict[str, str]]]:
    yield from _calls_in_expression(stmt.test, aliases)
    yield from _iter_calls_with_aliases(stmt.body, dict(aliases))
    yield from _iter_calls_with_aliases(stmt.orelse, dict(aliases))


def _iter_for_calls(
    stmt: ast.For | ast.AsyncFor, aliases: dict[str, str]
) -> Iterator[tuple[ast.Call, dict[str, str]]]:
    yield from _calls_in_expression(stmt.iter, aliases)
    inner = dict(aliases)
    _remove_target_aliases([stmt.target], inner)
    yield from _iter_calls_with_aliases(stmt.body, inner)
    yield from _iter_calls_with_aliases(stmt.orelse, dict(aliases))


def _iter_while_calls(
    stmt: ast.While, aliases: dict[str, str]
) -> Iterator[tuple[ast.Call, dict[str, str]]]:
    yield from _calls_in_expression(stmt.test, aliases)
    yield from _iter_calls_with_aliases(stmt.body, dict(aliases))
    yield from _iter_calls_with_aliases(stmt.orelse, dict(aliases))


def _iter_with_calls(
    stmt: ast.With | ast.AsyncWith, aliases: dict[str, str]
) -> Iterator[tuple[ast.Call, dict[str, str]]]:
    for item in stmt.items:
        yield from _calls_in_expression(item.context_expr, aliases)
    inner = dict(aliases)
    for item in stmt.items:
        if item.optional_vars is not None:
            _remove_target_aliases([item.optional_vars], inner)
    yield from _iter_calls_with_aliases(stmt.body, inner)


def _iter_try_calls(
    stmt: ast.Try, aliases: dict[str, str]
) -> Iterator[tuple[ast.Call, dict[str, str]]]:
    yield from _iter_calls_with_aliases(stmt.body, dict(aliases))
    for handler in stmt.handlers:
        yield from _iter_calls_with_aliases(handler.body, dict(aliases))
    yield from _iter_calls_with_aliases(stmt.orelse, dict(aliases))
    yield from _iter_calls_with_aliases(stmt.finalbody, dict(aliases))


def _iter_match_calls(
    stmt: ast.Match, aliases: dict[str, str]
) -> Iterator[tuple[ast.Call, dict[str, str]]]:
    yield from _calls_in_expression(stmt.subject, aliases)
    for case in stmt.cases:
        if case.guard is not None:
            yield from _calls_in_expression(case.guard, aliases)
        yield from _iter_calls_with_aliases(case.body, dict(aliases))


ScopeHandler = Callable[
    [Any, dict[str, str]], Iterator[tuple[ast.Call, dict[str, str]]]
]
_SCOPE_HANDLERS: dict[type[ast.stmt], ScopeHandler] = {
    ast.AsyncFunctionDef: _iter_function_calls,
    ast.FunctionDef: _iter_function_calls,
    ast.ClassDef: _iter_class_calls,
    ast.Assign: _iter_assign_calls,
    ast.AnnAssign: _iter_ann_assign_calls,
    ast.AugAssign: _iter_aug_assign_calls,
    ast.If: _iter_if_calls,
    ast.AsyncFor: _iter_for_calls,
    ast.For: _iter_for_calls,
    ast.While: _iter_while_calls,
    ast.AsyncWith: _iter_with_calls,
    ast.With: _iter_with_calls,
    ast.Try: _iter_try_calls,
    ast.Match: _iter_match_calls,
}


def _iter_calls_with_aliases(
    statements: list[ast.stmt], aliases: dict[str, str]
) -> Iterator[tuple[ast.Call, dict[str, str]]]:
    """Yield calls together with aliases visible at that point in the scope."""
    for stmt in statements:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            _apply_scope_import(stmt, aliases)
            continue
        handler = _SCOPE_HANDLERS.get(type(stmt))
        if handler is None:
            yield from _calls_in_expression(stmt, aliases)
        else:
            yield from handler(stmt, aliases)


def _absolute_imports(tree: ast.Module) -> Iterator[tuple[int, str]]:
    """Yield line numbers and modules of absolute imports."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module


def _stdlib_roots() -> frozenset[str]:
    return frozenset(getattr(sys, "stdlib_module_names", ())) | {"__future__"}


def _is_allowed_contract_root(module: str) -> bool:
    root = module.split(".")[0]
    if root in _stdlib_roots():
        return True
    if root in ALLOWED_THIRD_PARTY_ROOTS:
        return True
    return root == "automl_contracts"


def scan_contract_imports(contract_src: Path = CONTRACT_SRC) -> list[Violation]:
    """Reject engine and non-contract libraries imported by the contract."""
    violations: list[Violation] = []
    for path in python_files(contract_src):
        tree = _parse(path)
        for line, module in _absolute_imports(tree):
            root = module.split(".")[0]
            if root == ENGINE_MODULE:
                violations.append(
                    Violation(
                        "dependency-direction",
                        path,
                        line,
                        "the contract must not import "
                        f"{ENGINE_MODULE}: dependencies point from the engine "
                        "to the contract, never back",
                    )
                )
            elif not _is_allowed_contract_root(module):
                violations.append(
                    Violation(
                        "contract-dependency",
                        path,
                        line,
                        f"the contract may only use the standard library and "
                        f"{sorted(ALLOWED_CONTRACT_DEPENDENCIES)}, found "
                        f"{module!r}",
                    )
                )
    return violations


def _annotation_nodes(tree: ast.Module) -> Iterator[ast.AST]:
    for node in ast.walk(tree):
        annotation: ast.AST | None = None
        if isinstance(node, (ast.AnnAssign, ast.arg)):
            annotation = node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotation = node.returns
        if annotation is not None:
            yield annotation


def _type_expression_strings(
    node: ast.AST, aliases: dict[str, str]
) -> Iterator[ast.Constant]:
    """Yield string type references, skipping Literal/Annotated metadata."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            yield node
        return
    if isinstance(node, ast.Subscript):
        base = _resolve_callable_name(node.value, aliases)
        base_name = base.rsplit(".", 1)[-1] if base else ""
        if base_name == "Literal":
            return
        if base_name == "Annotated":
            if isinstance(node.slice, ast.Tuple) and node.slice.elts:
                yield from _type_expression_strings(node.slice.elts[0], aliases)
            else:
                yield from _type_expression_strings(node.slice, aliases)
            return
        yield from _type_expression_strings(node.slice, aliases)
        return
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for element in node.elts:
            yield from _type_expression_strings(element, aliases)
        return
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        yield from _type_expression_strings(node.left, aliases)
        yield from _type_expression_strings(node.right, aliases)


def _string_type_violation(
    path: Path, node: ast.Constant, rule: str
) -> Violation | None:
    value = str(node.value)
    if re.search(rf"\b{re.escape(ENGINE_MODULE)}\b", value):
        return Violation(
            rule,
            path,
            node.lineno,
            f"a string type annotation refers to {ENGINE_MODULE}; the "
            "contract cannot depend on the engine, including forward "
            "references",
        )
    for root in FORBIDDEN_LIBRARY_ROOTS:
        if re.search(rf"\b{re.escape(root)}\b", value):
            return Violation(
                rule,
                path,
                node.lineno,
                f"a string type annotation refers to forbidden library {root!r}",
            )
    return None


def _is_type_alias_annotation(annotation: ast.AST, aliases: dict[str, str]) -> bool:
    name = _resolve_callable_name(annotation, aliases)
    return name is not None and name.rsplit(".", 1)[-1] == "TypeAlias"


def _is_type_constructor(node: ast.AST, aliases: dict[str, str]) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    base = _resolve_callable_name(node.value, aliases)
    if base is None:
        return False
    last = base.rsplit(".", 1)[-1]
    return last in TYPE_CONSTRUCTOR_NAMES or base.startswith(
        ("typing.", "collections.abc.")
    )


def _is_type_expression(node: ast.AST, aliases: dict[str, str]) -> bool:
    if _is_type_constructor(node, aliases):
        return True
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.BitOr)
        and (
            _is_type_expression(node.left, aliases)
            or _is_type_expression(node.right, aliases)
        )
    )


def _looks_like_type_reference(value: str) -> bool:
    """Separate qualified type names from ordinary dotted string data."""
    if ENGINE_MODULE in value:
        return True
    if "." not in value:
        return False
    last = value.rsplit(".", 1)[-1]
    return bool(last) and (last[0].isupper() or last.startswith("_"))


def _assigned_type_strings(
    path: Path, tree: ast.Module, aliases: dict[str, str]
) -> list[Violation]:
    """Check type aliases, not ordinary string-valued data."""
    violations: list[Violation] = []
    type_alias_node = getattr(ast, "TypeAlias", ())
    for node in ast.walk(tree):
        constants: list[ast.Constant] = []
        if isinstance(node, ast.Assign):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                if _looks_like_type_reference(value.value):
                    constants = [value]
            elif isinstance(value, (ast.Subscript, ast.BinOp)) and _is_type_expression(
                value, aliases
            ):
                constants = list(_type_expression_strings(value, aliases))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if _is_type_alias_annotation(node.annotation, aliases):
                constants = list(_type_expression_strings(node.value, aliases))
        elif isinstance(node, type_alias_node) and isinstance(node.value, ast.AST):
            constants = list(_type_expression_strings(node.value, aliases))
        for constant in constants:
            violation = _string_type_violation(path, constant, "string-annotation")
            if violation is not None:
                violations.append(violation)
    return violations


def _call_type_strings(path: Path, tree: ast.Module) -> list[Violation]:
    violations: list[Violation] = []
    initial = _import_aliases(tree)
    for node, aliases in _iter_calls_with_aliases(tree.body, initial):
        name = _resolve_callable_name(node.func, aliases)
        if name is None:
            continue
        if name.rsplit(".", 1)[-1] not in {
            "ForwardRef",
            "NewType",
            "TypeVar",
            "cast",
        }:
            continue
        arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
        for argument in arguments:
            for constant in _type_expression_strings(argument, aliases):
                violation = _string_type_violation(path, constant, "string-annotation")
                if violation is not None:
                    violations.append(violation)
    return violations


def scan_contract_type_strings(contract_src: Path = CONTRACT_SRC) -> list[Violation]:
    """Reject engine/heavy references hidden in string annotations."""
    violations: list[Violation] = []
    for path in python_files(contract_src):
        tree = _parse(path)
        aliases = _import_aliases(tree)
        for annotation in _annotation_nodes(tree):
            for constant in _type_expression_strings(annotation, aliases):
                violation = _string_type_violation(path, constant, "string-annotation")
                if violation is not None:
                    violations.append(violation)
        violations.extend(_assigned_type_strings(path, tree, aliases))
        violations.extend(_call_type_strings(path, tree))
    return violations


def _literal_getattr_target(node: ast.Call) -> str | None:
    if len(node.args) < 2:
        return None
    argument = node.args[1]
    if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
        return None
    return argument.value if argument.value in DYNAMIC_GETATTR_NAMES else None


def scan_contract_dynamic_imports(contract_src: Path = CONTRACT_SRC) -> list[Violation]:
    """Reject dynamic import calls in the contract."""
    violations: list[Violation] = []
    for path in python_files(contract_src):
        tree = _parse(path)
        initial = _import_aliases(tree)
        for node, aliases in _iter_calls_with_aliases(tree.body, initial):
            name = _resolve_callable_name(node.func, aliases)
            if name in DYNAMIC_IMPORT_CALLS:
                message = (
                    f"dynamic execution through {name!r} is forbidden in the "
                    "contract; use a static import"
                )
                violations.append(
                    Violation("dynamic-import", path, node.lineno, message)
                )
            elif (
                name in {"getattr", "builtins.getattr"}
                and (target := _literal_getattr_target(node)) is not None
            ):
                violations.append(
                    Violation(
                        "dynamic-import",
                        path,
                        node.lineno,
                        f"dynamic access to {target!r} is forbidden in the contract",
                    )
                )
    return violations


def _runtime_violation(path: Path, line: int, name: str) -> Violation:
    return Violation(
        "runtime-access",
        path,
        line,
        f"the contract must not use {name!r}: runtime and environment "
        "access belong to the engine",
    )


def _annotation_contains_path(annotation: ast.AST, aliases: dict[str, str]) -> bool:
    name = _resolve_callable_name(annotation, aliases)
    if name is not None and name.rsplit(".", 1)[-1] == "Path":
        return True
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _annotation_contains_path(annotation.left, aliases) or (
            _annotation_contains_path(annotation.right, aliases)
        )
    if isinstance(annotation, ast.Subscript):
        base = _resolve_callable_name(annotation.value, aliases)
        base_name = base.rsplit(".", 1)[-1] if base else ""
        elements = (
            annotation.slice.elts
            if isinstance(annotation.slice, ast.Tuple)
            else [annotation.slice]
        )
        if base_name == "Annotated" and elements:
            return _annotation_contains_path(elements[0], aliases)
        if base_name in {"Optional", "Union"}:
            return any(
                _annotation_contains_path(element, aliases) for element in elements
            )
    return False


def _is_path_expression(
    node: ast.AST, aliases: dict[str, str], path_names: set[str]
) -> bool:
    """Recognise Path factories and Path-preserving method chains."""
    if isinstance(node, ast.Name):
        return (
            node.id in path_names or aliases.get(node.id, node.id) in PATH_CLASS_NAMES
        )
    if isinstance(node, ast.Attribute):
        if node.attr == "Path":
            name = _resolve_callable_name(node, aliases)
            if name in PATH_CLASS_NAMES:
                return True
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in {"self", "cls"}
            and node.attr in path_names
        ):
            return True
        return node.attr == "parent" and _is_path_expression(
            node.value, aliases, path_names
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _is_path_expression(node.left, aliases, path_names)
    if not isinstance(node, ast.Call):
        return False
    name = _resolve_callable_name(node.func, aliases)
    if name in PATH_CLASS_NAMES:
        return True
    if isinstance(node.func, ast.Attribute):
        return node.func.attr in PATH_RETURNING_METHODS and _is_path_expression(
            node.func.value, aliases, path_names
        )
    return False


def _add_path_names(targets: Sequence[ast.AST], names: set[str]) -> bool:
    changed = False
    for target in targets:
        if isinstance(target, ast.Name) and target.id not in names:
            names.add(target.id)
            changed = True
        elif (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id in {"self", "cls"}
            and target.attr not in names
        ):
            names.add(target.attr)
            changed = True
    return changed


def _path_names(tree: ast.Module, aliases: dict[str, str]) -> set[str]:
    """Collect variables and annotated attributes that hold a Path."""
    names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if _is_path_expression(node.value, aliases, names):
                    changed |= _add_path_names(node.targets, names)
            elif isinstance(node, ast.AnnAssign):
                if _annotation_contains_path(node.annotation, aliases):
                    changed |= _add_path_names([node.target], names)
                if node.value is not None and _is_path_expression(
                    node.value, aliases, names
                ):
                    changed |= _add_path_names([node.target], names)
            elif (
                isinstance(node, ast.arg)
                and node.annotation is not None
                and _annotation_contains_path(node.annotation, aliases)
                and node.arg not in names
            ):
                names.add(node.arg)
                changed = True
    return names


def scan_contract_runtime_access(
    contract_src: Path = CONTRACT_SRC,
) -> list[Violation]:
    """Reject environment, filesystem and process access in the contract."""
    violations: list[Violation] = []
    for path in python_files(contract_src):
        tree = _parse(path)
        initial = _import_aliases(tree)
        path_names = _path_names(tree, initial)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                name = initial.get(node.id, node.id)
                if name in RUNTIME_ATTRIBUTES:
                    violations.append(_runtime_violation(path, node.lineno, name))
                continue
            if isinstance(node, ast.Attribute):
                attribute_name = _resolve_callable_name(node, initial)
                if attribute_name in RUNTIME_ATTRIBUTES:
                    violations.append(
                        _runtime_violation(path, node.lineno, attribute_name)
                    )
        for node, aliases in _iter_calls_with_aliases(tree.body, initial):
            if isinstance(node.func, ast.Attribute) and _is_path_expression(
                node.func.value, aliases, path_names
            ):
                if node.func.attr not in PATH_PURE_METHODS:
                    violations.append(
                        _runtime_violation(path, node.lineno, f"Path.{node.func.attr}")
                    )
                continue
            call_name = _resolve_callable_name(node.func, aliases)
            if call_name is None:
                continue
            if call_name == "open" or call_name.startswith(RUNTIME_CALL_PREFIXES):
                violations.append(_runtime_violation(path, node.lineno, call_name))
    return violations


def _requirement_name(requirement: str) -> str:
    without_marker = requirement.split(";", 1)[0].strip()
    name = re.split(r"[<>=!~\[\s(]", without_marker, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def load_pyproject(path: Path) -> dict[str, object]:
    """Read a TOML project file."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _project_dependencies(project: dict[str, object]) -> list[str]:
    dependencies = project.get("dependencies", [])
    if not isinstance(dependencies, list):
        return []
    return [str(item) for item in dependencies]


def _optional_dependencies(project: dict[str, object]) -> list[str]:
    groups = project.get("optional-dependencies", {})
    if not isinstance(groups, dict):
        return []
    result: list[str] = []
    for values in groups.values():
        if isinstance(values, list):
            result.extend(str(item) for item in values)
    return result


def scan_contract_metadata(
    pyproject: Path = CONTRACT_PYPROJECT,
) -> list[Violation]:
    """Verify the declarative dependency and type-checking policy."""
    violations: list[Violation] = []
    document = load_pyproject(pyproject)
    project = document.get("project", {})
    if not isinstance(project, dict):
        return [Violation("contract-metadata", pyproject, 1, "missing [project]")]
    for requirement in _project_dependencies(project) + _optional_dependencies(project):
        name = _requirement_name(requirement)
        if name not in ALLOWED_CONTRACT_DEPENDENCIES:
            violations.append(
                Violation(
                    "contract-metadata",
                    pyproject,
                    1,
                    f"runtime dependency {name!r} is not allowed; only "
                    f"{sorted(ALLOWED_CONTRACT_DEPENDENCIES)} are permitted "
                    "without an ADR exception",
                )
            )
    version = project.get("version")
    if not isinstance(version, str) or not SEMVER_RE.match(version):
        violations.append(
            Violation(
                "contract-metadata",
                pyproject,
                1,
                f"contract version {version!r} is not a valid semantic version",
            )
        )
    tool = document.get("tool", {})
    if not isinstance(tool, dict):
        tool = {}
    mypy = tool.get("mypy", {})
    if not isinstance(mypy, dict) or mypy.get("strict") is not True:
        violations.append(
            Violation(
                "contract-metadata",
                pyproject,
                1,
                "the contract mypy section must enable strict mode",
            )
        )
    importlinter = tool.get("importlinter", {})
    if not isinstance(importlinter, dict):
        importlinter = {}
    if importlinter.get("exclude_type_checking_imports") is not False:
        violations.append(
            Violation(
                "contract-metadata",
                pyproject,
                1,
                "import-linter must inspect TYPE_CHECKING imports "
                "(exclude_type_checking_imports = false)",
            )
        )
    contracts = importlinter.get("contracts", [])
    forbidden_engine = False
    if isinstance(contracts, list):
        for contract in contracts:
            if not isinstance(contract, dict):
                continue
            if contract.get("type") != "forbidden":
                continue
            forbidden = contract.get("forbidden_modules", [])
            if isinstance(forbidden, list) and ENGINE_MODULE in forbidden:
                forbidden_engine = True
    if not forbidden_engine:
        violations.append(
            Violation(
                "contract-metadata",
                pyproject,
                1,
                "import-linter must declare a forbidden-modules contract for "
                f"{ENGINE_MODULE}",
            )
        )
    return violations


def _ruff_policy(pyproject: Path) -> dict[str, object]:
    document = load_pyproject(pyproject)
    tool = document.get("tool", {})
    if not isinstance(tool, dict):
        return {}
    ruff = tool.get("ruff", {})
    if not isinstance(ruff, dict):
        return {}
    lint = ruff.get("lint", {})
    if not isinstance(lint, dict):
        lint = {}
    mccabe = lint.get("mccabe", {})
    if not isinstance(mccabe, dict):
        mccabe = {}
    fmt = ruff.get("format", {})
    if not isinstance(fmt, dict):
        fmt = {}
    return {
        "line-length": ruff.get("line-length"),
        "target-version": ruff.get("target-version"),
        "extend-select": lint.get("extend-select"),
        "max-complexity": mccabe.get("max-complexity"),
        "quote-style": fmt.get("quote-style"),
        "indent-style": fmt.get("indent-style"),
        "line-ending": fmt.get("line-ending"),
    }


def scan_ruff_policy(
    engine_pyproject: Path = ENGINE_PYPROJECT,
    contract_pyproject: Path = CONTRACT_PYPROJECT,
) -> list[Violation]:
    """Verify that both distributions use the same Ruff rules."""
    engine = _ruff_policy(engine_pyproject)
    contract = _ruff_policy(contract_pyproject)
    violations: list[Violation] = []
    for key, value in engine.items():
        if contract.get(key) != value:
            violations.append(
                Violation(
                    "ruff-policy",
                    contract_pyproject,
                    1,
                    f"contract Ruff option {key!r} differs from the engine: "
                    f"expected {value!r}, found {contract.get(key)!r}",
                )
            )
    return violations


def _contract_names_from_file(path: Path) -> set[str]:
    tree = _parse(path)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            names.add(node.name)
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
            continue
        for element in node.value.elts:
            if (
                isinstance(element, ast.Constant)
                and isinstance(element.value, str)
                and element.value not in {"__all__", "__version__"}
            ):
                names.add(element.value)
    return names


def contract_public_names(contract_src: Path = CONTRACT_SRC) -> set[str]:
    """Return public contract classes, functions and constants."""
    names: set[str] = set()
    for path in python_files(contract_src):
        names.update(_contract_names_from_file(path))
    return names


def _is_contract_reexport(value: ast.AST, aliases: dict[str, str]) -> bool:
    name = _resolve_callable_name(value, aliases)
    return name == "automl_contracts" or bool(
        name and name.startswith("automl_contracts.")
    )


def scan_engine_duplicates(
    engine_src: Path = ENGINE_SRC,
    contract_src: Path = CONTRACT_SRC,
) -> list[Violation]:
    """Reject engine definitions that duplicate contract names."""
    contract_names = contract_public_names(contract_src)
    violations: list[Violation] = []
    for path in python_files(engine_src):
        tree = _parse(path)
        aliases = _import_aliases(tree)
        for node in tree.body:
            duplicate = False
            duplicate_targets: list[ast.expr] = []
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                duplicate = node.name in contract_names
            elif isinstance(node, ast.Assign):
                duplicate_targets = node.targets
                duplicate = any(
                    isinstance(target, ast.Name) and target.id in contract_names
                    for target in duplicate_targets
                ) and not _is_contract_reexport(node.value, aliases)
            elif isinstance(node, ast.AnnAssign):
                duplicate_targets = [node.target]
                duplicate = (
                    isinstance(node.target, ast.Name)
                    and node.target.id in contract_names
                    and node.value is not None
                    and not _is_contract_reexport(node.value, aliases)
                )
            if duplicate:
                if isinstance(
                    node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    name = node.name
                else:
                    name = next(
                        target.id
                        for target in duplicate_targets
                        if isinstance(target, ast.Name)
                    )
                violations.append(
                    Violation(
                        "engine-duplicate",
                        path,
                        node.lineno,
                        f"the engine defines {name!r}, which belongs to the "
                        "contract; import and re-export it from "
                        "automl_contracts instead",
                    )
                )
    return violations


def scan_engine_contract_usage(
    engine_src: Path = ENGINE_SRC,
    engine_pyproject: Path = ENGINE_PYPROJECT,
) -> list[Violation]:
    """Verify that the engine depends on and imports the contract."""
    violations: list[Violation] = []
    document = load_pyproject(engine_pyproject)
    project = document.get("project", {})
    dependencies = _project_dependencies(project) if isinstance(project, dict) else []
    dependency_names = {_requirement_name(item) for item in dependencies}
    if "automl-contracts" not in dependency_names:
        violations.append(
            Violation(
                "engine-contract-dependency",
                engine_pyproject,
                1,
                "the engine must declare automl-contracts as a runtime dependency",
            )
        )
    imports_contract = False
    for path in python_files(engine_src):
        tree = _parse(path)
        for _, module in _absolute_imports(tree):
            if module == "automl_contracts" or module.startswith("automl_contracts."):
                imports_contract = True
                break
        if imports_contract:
            break
    if not imports_contract:
        violations.append(
            Violation(
                "engine-contract-dependency",
                engine_src,
                1,
                "the engine does not import automl_contracts; shared "
                "definitions must come from the contract",
            )
        )
    return violations


def _package_data(document: dict[str, object]) -> dict[str, object]:
    tool = document.get("tool", {})
    if not isinstance(tool, dict):
        return {}
    setuptools = tool.get("setuptools", {})
    if not isinstance(setuptools, dict):
        return {}
    package_data = setuptools.get("package-data", {})
    return package_data if isinstance(package_data, dict) else {}


def scan_contract_package_data(
    contract_root: Path = CONTRACT_ROOT,
    pyproject: Path = CONTRACT_PYPROJECT,
) -> list[Violation]:
    """Verify that py.typed and the JSON Schema are packaged."""
    violations: list[Violation] = []
    document = load_pyproject(pyproject)
    package_data = _package_data(document)
    declared = package_data.get("automl_contracts", [])
    if not isinstance(declared, list):
        declared = []
    for required in ("py.typed", "schemas/*.json"):
        if required not in declared:
            violations.append(
                Violation(
                    "package-data",
                    pyproject,
                    1,
                    f"package data must include {required!r}",
                )
            )
    for required_path in (
        contract_root / "src" / "automl_contracts" / "py.typed",
        contract_root / "src" / "automl_contracts" / "schemas" / "config.schema.json",
    ):
        if not required_path.is_file():
            violations.append(
                Violation(
                    "package-data",
                    required_path,
                    1,
                    "required contract package data is missing",
                )
            )
    return violations


def _metadata_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line or line.startswith(" "):
            continue
        key, separator, value = line.partition(": ")
        if not separator:
            continue
        sections.setdefault(key, []).append(value)
    return sections


def _find_member(names: Sequence[str], suffix: str) -> str | None:
    for name in names:
        if name.endswith(suffix):
            return name
    return None


def _check_distribution_dependencies(
    path: Path, requirements: Iterable[str]
) -> list[Violation]:
    violations: list[Violation] = []
    for requirement in requirements:
        name = _requirement_name(requirement)
        if name not in ALLOWED_CONTRACT_DEPENDENCIES:
            violations.append(
                Violation(
                    "artifact-metadata",
                    path,
                    1,
                    f"contract metadata requires forbidden dependency {name!r}",
                )
            )
    return violations


def inspect_contract_wheel(
    wheel: Path, expected_version: str | None = None
) -> list[Violation]:
    """Inspect a built contract wheel for contents and metadata."""
    violations: list[Violation] = []
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    for name in names:
        if ENGINE_MODULE in name:
            violations.append(
                Violation(
                    "artifact-content",
                    wheel,
                    1,
                    f"contract wheel contains engine file {name!r}",
                )
            )
        if name.endswith(".py") and not name.startswith("automl_contracts/"):
            violations.append(
                Violation(
                    "artifact-content",
                    wheel,
                    1,
                    f"contract wheel contains non-contract Python module {name!r}",
                )
            )
    for required in (
        "automl_contracts/__init__.py",
        "automl_contracts/py.typed",
        "automl_contracts/schemas/config.schema.json",
    ):
        if required not in names:
            violations.append(
                Violation(
                    "artifact-content",
                    wheel,
                    1,
                    f"contract wheel is missing {required!r}",
                )
            )
    metadata_name = _find_member(names, ".dist-info/METADATA")
    if metadata_name is None:
        return violations + [
            Violation(
                "artifact-metadata",
                wheel,
                1,
                "contract wheel has no .dist-info/METADATA",
            )
        ]
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(metadata_name).decode("utf-8")
    sections = _metadata_sections(metadata)
    names_meta = sections.get("Name", [])
    versions = sections.get("Version", [])
    if names_meta != ["automl-contracts"]:
        violations.append(
            Violation(
                "artifact-metadata",
                wheel,
                1,
                f"contract wheel name must be 'automl-contracts', found {names_meta!r}",
            )
        )
    if not versions or not SEMVER_RE.match(versions[0]):
        violations.append(
            Violation(
                "artifact-metadata",
                wheel,
                1,
                f"contract wheel version {versions!r} is not semver",
            )
        )
    elif expected_version is not None and versions[0] != expected_version:
        violations.append(
            Violation(
                "artifact-metadata",
                wheel,
                1,
                f"contract wheel version {versions[0]!r} does not match "
                f"pyproject version {expected_version!r}",
            )
        )
    violations.extend(
        _check_distribution_dependencies(wheel, sections.get("Requires-Dist", []))
    )
    return violations


def inspect_contract_sdist(
    sdist: Path, expected_version: str | None = None
) -> list[Violation]:
    """Inspect a built contract sdist for contents and metadata."""
    violations: list[Violation] = []
    with tarfile.open(sdist, "r:*") as archive:
        names = archive.getnames()
    for name in names:
        if ENGINE_MODULE in name:
            violations.append(
                Violation(
                    "artifact-content",
                    sdist,
                    1,
                    f"contract sdist contains engine file {name!r}",
                )
            )
    for required in (
        "src/automl_contracts/__init__.py",
        "src/automl_contracts/py.typed",
        "src/automl_contracts/schemas/config.schema.json",
    ):
        if not any(name.endswith(required) for name in names):
            violations.append(
                Violation(
                    "artifact-content",
                    sdist,
                    1,
                    f"contract sdist is missing {required!r}",
                )
            )
    pyproject = next((name for name in names if name.endswith("/pyproject.toml")), None)
    if pyproject is None:
        return violations + [
            Violation(
                "artifact-metadata",
                sdist,
                1,
                "contract sdist has no pyproject.toml",
            )
        ]
    if expected_version is not None:
        with tarfile.open(sdist, "r:*") as archive:
            payload = archive.extractfile(pyproject)
            if payload is None:
                raise ValueError(f"cannot read {pyproject} from {sdist}")
            document = tomllib.loads(payload.read().decode("utf-8"))
        version = document.get("project", {}).get("version")
        if version != expected_version:
            violations.append(
                Violation(
                    "artifact-metadata",
                    sdist,
                    1,
                    f"contract sdist version {version!r} does not match "
                    f"pyproject version {expected_version!r}",
                )
            )
    return violations


def inspect_engine_artifacts(
    wheel: Path | None = None, sdist: Path | None = None
) -> list[Violation]:
    """Verify that engine artifacts do not contain contract sources."""
    violations: list[Violation] = []
    if wheel is not None:
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if "automl_contracts" in name:
                    violations.append(
                        Violation(
                            "artifact-separation",
                            wheel,
                            1,
                            f"engine wheel contains contract file {name!r}",
                        )
                    )
    if sdist is not None:
        with tarfile.open(sdist, "r:*") as archive:
            for name in archive.getnames():
                if "automl_contracts" in name:
                    violations.append(
                        Violation(
                            "artifact-separation",
                            sdist,
                            1,
                            f"engine sdist contains contract file {name!r}",
                        )
                    )
    return violations


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _block_after_marker(
    lines: list[str], marker: str, *, exact: bool = False
) -> tuple[list[str], int] | None:
    for index, line in enumerate(lines):
        stripped = line.strip()
        matches = stripped == marker if exact else stripped.startswith(marker)
        if matches:
            base = _line_indent(line)
            block = [line]
            for following in lines[index + 1 :]:
                if not following.strip():
                    block.append(following)
                    continue
                if _line_indent(following) <= base:
                    break
                block.append(following)
            return block, index + 1
    return None


def _check_ci_block(
    path: Path,
    block: list[str],
    required: dict[str, str],
    label: str,
) -> list[Violation]:
    text = "\n".join(line for line in block if not line.lstrip().startswith("#"))
    violations: list[Violation] = []
    for token, description in required.items():
        if token not in text:
            violations.append(
                Violation(
                    "ci-configuration",
                    path,
                    1,
                    f"{label} has no {description} ({token!r})",
                )
            )
    return violations


def scan_ci_configuration(
    ci_path: Path = CI_CONFIG,
    github_ci_path: Path | None = None,
) -> list[Violation]:
    """Verify that the dedicated CI jobs contain the contract gates."""
    violations: list[Violation] = []
    if not ci_path.is_file():
        violations.append(
            Violation("ci-configuration", ci_path, 1, "CI configuration is missing")
        )
    else:
        lines = ci_path.read_text(encoding="utf-8").splitlines()
        found = _block_after_marker(lines, "- name: contracts-guarantees")
        if found is None:
            violations.append(
                Violation(
                    "ci-configuration",
                    ci_path,
                    1,
                    "CI configuration has no contracts-guarantees task",
                )
            )
        else:
            block, _ = found
            violations.extend(
                _check_ci_block(
                    ci_path,
                    block,
                    {
                        "scripts/check_contract_boundaries.py": "static boundary check",
                        "scripts/verify_contract_distribution.py": "artifact verification",
                        "scripts/smoke_contract_install.py": "isolated smoke import",
                        "python3.11 -m build automl-contracts": "contract build",
                        "twine check": "twine metadata check",
                        "python3.11 -m venv": "isolated environment",
                        "pytest tests/contracts": "contract tests",
                        "mypy src": "contract mypy",
                        "ruff check src/ automl-contracts/src/": "ruff for both packages",
                        "ruff check tools/contract_boundaries.py": "ruff for boundary tooling",
                        "mypy tools/contract_boundaries.py": "mypy for boundary tooling",
                    },
                    "contracts-guarantees task",
                )
            )
    github_path = GITHUB_CI_CONFIG if github_ci_path is None else github_ci_path
    if github_path.is_file():
        lines = github_path.read_text(encoding="utf-8").splitlines()
        found = _block_after_marker(lines, "contracts:", exact=True)
        if found is None:
            violations.append(
                Violation(
                    "ci-configuration",
                    github_path,
                    1,
                    "GitHub CI has no contracts job",
                )
            )
        else:
            block, _ = found
            violations.extend(
                _check_ci_block(
                    github_path,
                    block,
                    {
                        "scripts/check_contract_boundaries.py": "static boundary check",
                        "scripts/verify_contract_distribution.py": "artifact verification",
                        "scripts/smoke_contract_install.py": "isolated smoke import",
                        "python -m build automl-contracts": "contract build",
                        "twine check": "twine metadata check",
                        "python -m venv": "isolated environment",
                        "python -m pytest tests/contracts": "contract tests",
                        "mypy src": "contract mypy",
                        "ruff check src/ automl-contracts/src/": "ruff for both packages",
                        "ruff check tools/contract_boundaries.py": "ruff for boundary tooling",
                        "mypy tools/contract_boundaries.py": "mypy for boundary tooling",
                    },
                    "GitHub contracts job",
                )
            )
    return violations


def run_static_checks(repo_root: Path = REPO_ROOT) -> list[Violation]:
    """Run every static guard against a repository checkout."""
    contract_root = repo_root / "automl-contracts"
    contract_src = contract_root / "src" / "automl_contracts"
    contract_pyproject = contract_root / "pyproject.toml"
    engine_src = repo_root / "src" / "configurable_automl_engine"
    engine_pyproject = repo_root / "pyproject.toml"
    ci_path = repo_root / ".sourcecraft" / "ci.yaml"
    github_ci_path = repo_root / ".github" / "workflows" / "CI.yaml"
    violations: list[Violation] = []
    violations.extend(scan_contract_imports(contract_src))
    violations.extend(scan_contract_type_strings(contract_src))
    violations.extend(scan_contract_dynamic_imports(contract_src))
    violations.extend(scan_contract_runtime_access(contract_src))
    violations.extend(scan_contract_metadata(contract_pyproject))
    violations.extend(scan_contract_package_data(contract_root, contract_pyproject))
    violations.extend(scan_ruff_policy(engine_pyproject, contract_pyproject))
    violations.extend(scan_engine_duplicates(engine_src, contract_src))
    violations.extend(scan_engine_contract_usage(engine_src, engine_pyproject))
    violations.extend(scan_ci_configuration(ci_path, github_ci_path))
    return violations
