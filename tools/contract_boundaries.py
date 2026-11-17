"""Static guards for the contract/engine architecture boundary.

The module is shared by the pytest guards and the CI entry points.  It keeps
the checks independent from the engine runtime: contract imports are inspected
with :mod:`ast`, metadata is read from ``pyproject.toml``, and distribution
artifacts are inspected with :mod:`zipfile`/:mod:`tarfile`.

The import and runtime scans follow simple value aliases through assignments,
tuple/list unpacking, walrus expressions, conditional expressions and literal
``getattr``/``__dict__`` indirection.  Path taint is flow-sensitive: every
scope tracks the names that may hold a ``Path`` at that point, merges branches
with a may-analysis and clears a name when a rebinding rules ``Path`` out.
"""

from __future__ import annotations

import ast
import re
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

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
WILDCARD_IMPORT_ALLOWLIST = frozenset({"__future__", "typing"})
IDENTITY_ALIAS_PREFIX = "@identity:"
CLASS_PATH_FIELD_PREFIX = "@pathfield:"
CALL_PARAMETER_PREFIX = "@param:"
INIT_PARAMETER_PREFIX = "@initparam:"


def _identity_alias_key(name: str) -> str:
    return f"{IDENTITY_ALIAS_PREFIX}{name}"


def _clear_identity_binding(name: str, aliases: dict[str, frozenset[str]]) -> None:
    aliases.pop(_identity_alias_key(name), None)


def _is_dynamic_import_name(name: str) -> bool:
    normalised = _normalise_builtins_name(name)
    if normalised in DYNAMIC_IMPORT_CALLS:
        return True
    if normalised.endswith(".__getattr__"):
        return normalised.split(".", 1)[0] in DYNAMIC_GETATTR_ROOTS
    if normalised.endswith((".load_module", ".exec_module")):
        return True
    if normalised.startswith("importlib.machinery."):
        tail = normalised[len("importlib.machinery.") :]
        return tail.endswith("Loader") or tail in {"load_module", "exec_module"}
    return False


DYNAMIC_GETATTR_ROOTS = frozenset(
    {"builtins", "importlib", "pkgutil", "runpy", "zipimport"}
)
DYNAMIC_GETATTR_NAMES = frozenset(
    {
        "__import__",
        "eval",
        "exec",
        "exec_module",
        "find_spec",
        "get_loader",
        "import_module",
        "load_module",
        "run_module",
        "run_path",
        "spec_from_file_location",
        "zipimporter",
        "ExtensionFileLoader",
        "FileLoader",
        "SourceFileLoader",
        "SourcelessFileLoader",
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
        "importlib.machinery.ExtensionFileLoader",
        "importlib.machinery.FileLoader",
        "importlib.machinery.SourceFileLoader",
        "importlib.machinery.SourcelessFileLoader",
        "importlib.util.find_spec",
        "importlib.util.spec_from_file_location",
        "find_spec",
        "pkgutil.get_loader",
        "runpy.run_module",
        "runpy.run_path",
        "spec_from_file_location",
        "zipimport.zipimporter",
    }
)
RUNTIME_CALL_PREFIXES = (
    "builtins.open",
    "ctypes.",
    "dbm.",
    "fileinput.",
    "glob.",
    "io.open",
    "mmap.",
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
    "os.fchdir",
    "os.fchmod",
    "os.fchown",
    "os.fstat",
    "os.fstatvfs",
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
    "os.lchmod",
    "os.lchown",
    "os.pathconf",
    "os.abort",
    "os._exit",
    "os.waitpid",
    "os.umask",
    "os.ptrace",
    "os.unsetenv",
    "os.putenv",
    "os.link",
    "os.mkfifo",
    "os.mknod",
    "os.remove",
    "os.rename",
    "os.replace",
    "os.unlink",
    "os.utime",
    "os.sync",
    "os.syncfs",
    "os.statvfs",
    "os.getxattr",
    "os.setxattr",
    "os.listxattr",
    "os.removexattr",
    "os.fwalk",
    "nt.",
    "posix.",
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
    "posixpath.exists",
    "posixpath.isfile",
    "posixpath.isdir",
    "posixpath.islink",
    "posixpath.ismount",
    "posixpath.lexists",
    "posixpath.getsize",
    "posixpath.getmtime",
    "posixpath.getctime",
    "posixpath.getatime",
    "posixpath.realpath",
    "posixpath.samefile",
    "ntpath.exists",
    "ntpath.isfile",
    "ntpath.isdir",
    "ntpath.islink",
    "ntpath.ismount",
    "ntpath.lexists",
    "ntpath.getsize",
    "ntpath.getmtime",
    "ntpath.getctime",
    "ntpath.getatime",
    "ntpath.realpath",
    "ntpath.samefile",
    "asyncio.",
    "concurrent.",
    "ftplib.",
    "http.",
    "httplib.",
    "io.BufferedRandom",
    "io.BufferedReader",
    "io.BufferedWriter",
    "io.FileIO",
    "io.TextIOWrapper",
    "codecs.open",
    "gzip.open",
    "bz2.open",
    "lzma.open",
    "tarfile.open",
    "zipfile.",
    # A logger itself is allowed; opening or reading a log file is not.
    "logging.FileHandler",
    "logging.config.",
    "logging.handlers.",
    "operator.",
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
    "shelve.",
    "shutil.chown",
    "shutil.copy",
    "shutil.copy2",
    "shutil.disk_usage",
    "shutil.make_archive",
    "shutil.move",
    "shutil.unpack_archive",
    "shutil.rmtree",
    "shutil.which",
    "socket.",
    "subprocess.",
    "sqlite3.",
    "tarfile.",
    "tempfile.",
    "webbrowser.",
    "zipfile.",
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

PATH_PROPERTIES = frozenset(
    {
        "anchor",
        "drive",
        "name",
        "parent",
        "parents",
        "parts",
        "root",
        "stem",
        "suffix",
        "suffixes",
    }
)
PATH_CLASS_NAMES = frozenset(
    {
        "Path",
        "PosixPath",
        "WindowsPath",
        "pathlib.Path",
        "pathlib.PosixPath",
        "pathlib.WindowsPath",
    }
)
RUNTIME_MODULE_ROOTS = frozenset(
    {
        "__builtins__",
        "builtins",
        "glob",
        "importlib",
        "io",
        "logging",
        "multiprocessing",
        "nt",
        "ntpath",
        "os",
        "pathlib",
        "platform",
        "posix",
        "posixpath",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "tempfile",
    }
)
FORBIDDEN_INDIRECTION_NAMES = frozenset(
    {
        "builtins.globals",
        "builtins.locals",
        "globals",
        "locals",
        "operator.attrgetter",
        "operator.methodcaller",
    }
)
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
        "os.environb",
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
        "sys.modules",
        "sys.path",
        "sys.path_hooks",
        "sys.platform",
        "sys.prefix",
        "sys.stderr",
        "sys.stdin",
        "sys.stdout",
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


def _subscript_alias_key(node: ast.AST) -> str | None:
    """Return the alias-map key for a subscript receiver expression."""
    if isinstance(node, (ast.Name, ast.Attribute)):
        return _dotted_name(node)
    if isinstance(node, ast.Subscript):
        base = _subscript_alias_key(node.value)
        key = _literal_subscript_key(node)
        if base is None or key is None:
            return None
        return f"{base}[{key}]"
    return None


def _resolve_callable_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """Resolve a call target through local import aliases."""
    name = _dotted_name(node)
    if name is None:
        return None
    head, dot, tail = name.partition(".")
    resolved = aliases.get(head, head)
    return f"{resolved}.{tail}" if dot else resolved


def _module_value_aliases(
    tree: ast.Module, static: dict[str, str]
) -> tuple[dict[str, str], set[str]]:
    aliases = dict(static)
    alias_names: set[str] = set()
    for node in _scope_nodes(tree.body):
        targets: Sequence[ast.AST] = ()
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            targets = [node.target]
            value = node.value
        if value is None:
            continue
        resolved: str | None = None
        if isinstance(value, (ast.Name, ast.Attribute)):
            resolved = _resolve_callable_name(value, aliases)
        elif isinstance(value, ast.Call):
            function = _resolve_callable_name(value.func, aliases)
            if function in {"vars", "builtins.vars"} and value.args:
                base = _resolve_callable_name(value.args[0], aliases)
                resolved = f"{base}.__dict__" if base else None
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            if resolved is None:
                aliases.pop(target.id, None)
                alias_names.discard(target.id)
            else:
                aliases[target.id] = resolved
                alias_names.add(target.id)
    return aliases, alias_names


def _clear_index_aliases(name: str, aliases: dict[str, str]) -> None:
    prefix = f"{name}["
    for key in [key for key in aliases if key.startswith(prefix)]:
        aliases.pop(key, None)


def _assigned_alias_value(value: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(value, ast.Name):
        if value.id not in aliases:
            return None
        return _resolve_callable_name(value, aliases)
    if isinstance(value, ast.Attribute):
        root = _root_name(value)
        if root is None or root not in aliases:
            return None
        return _resolve_callable_name(value, aliases)
    if isinstance(value, ast.Call):
        function = _resolve_callable_name(value.func, aliases)
        if function in PATH_CLASS_NAMES:
            return "Path"
        if function in {"vars", "builtins.vars"} and value.args:
            base = _resolve_callable_name(value.args[0], aliases)
            return f"{base}.__dict__" if base else None
    return None


def _bind_index_aliases(name: str, value: ast.AST, aliases: dict[str, str]) -> None:
    if isinstance(value, (ast.List, ast.Tuple)):
        for index, element in enumerate(value.elts):
            resolved = _assigned_alias_value(element, aliases)
            key = f"{name}[{index}]"
            if resolved is None:
                aliases.pop(key, None)
            else:
                aliases[key] = resolved
        return
    if isinstance(value, ast.Dict):
        for key_node, element in zip(value.keys, value.values):
            if not isinstance(key_node, ast.Constant):
                continue
            resolved = _assigned_alias_value(element, aliases)
            key = f"{name}[{key_node.value!r}]"
            if resolved is None:
                aliases.pop(key, None)
            else:
                aliases[key] = resolved
        return
    if isinstance(value, ast.Name):
        prefix = f"{value.id}["
        for key in [key for key in aliases if key.startswith(prefix)]:
            aliases[f"{name}{key[len(value.id) :]}"] = aliases.pop(key)


def _remove_alias_target(target: ast.AST, aliases: dict[str, str]) -> None:
    if isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            _remove_alias_target(element, aliases)
        return
    if isinstance(target, ast.Starred):
        _remove_alias_target(target.value, aliases)
        return
    if isinstance(target, ast.Name):
        aliases.pop(target.id, None)
        _clear_index_aliases(target.id, aliases)
        return
    dotted = _dotted_name(target)
    if dotted is not None:
        aliases.pop(dotted, None)


_FLAT_SHADOWED_RANGE = "<shadowed-range>"


def _bind_alias_target(
    target: ast.AST, value: ast.AST | None, aliases: dict[str, str]
) -> None:
    if isinstance(target, (ast.Tuple, ast.List)):
        if isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(
            value.elts
        ):
            for element_target, element_value in zip(target.elts, value.elts):
                _bind_alias_target(element_target, element_value, aliases)
        else:
            _remove_alias_target(target, aliases)
        return
    if isinstance(target, ast.Starred):
        _bind_alias_target(target.value, value, aliases)
        return
    if isinstance(target, ast.Name):
        name = target.id
        _clear_index_aliases(name, aliases)
        resolved = _assigned_alias_value(value, aliases) if value is not None else None
        if resolved is None:
            if name == "range":
                aliases[name] = _FLAT_SHADOWED_RANGE
            else:
                aliases.pop(name, None)
        else:
            aliases[name] = resolved
        if value is not None:
            _bind_index_aliases(name, value, aliases)
        return
    if isinstance(target, ast.Subscript):
        receiver = _subscript_alias_key(target.value)
        key = _literal_subscript_key(target)
        if receiver is not None and key is not None:
            alias_key = f"{receiver}[{key}]"
            _clear_index_aliases(alias_key, aliases)
            resolved = (
                _assigned_alias_value(value, aliases) if value is not None else None
            )
            if resolved is None:
                aliases.pop(alias_key, None)
            elif value is not None:
                aliases[alias_key] = resolved
                _bind_index_aliases(alias_key, value, aliases)
        return
    dotted = _dotted_name(target)
    if dotted is None:
        return
    resolved = _assigned_alias_value(value, aliases) if value is not None else None
    if resolved is None:
        aliases.pop(dotted, None)
    else:
        aliases[dotted] = resolved


def _apply_flat_import(stmt: ast.stmt, aliases: dict[str, str]) -> None:
    if isinstance(stmt, ast.Import):
        for alias in stmt.names:
            local = alias.asname or alias.name.split(".")[0]
            resolved = alias.name if alias.asname else alias.name.split(".")[0]
            aliases[local] = resolved
    elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module:
        for alias in stmt.names:
            if alias.name != "*":
                aliases[alias.asname or alias.name] = f"{stmt.module}.{alias.name}"


def _function_parameters(
    statement: ast.FunctionDef | ast.AsyncFunctionDef,
) -> Iterator[ast.arg]:
    parameters = [
        *statement.args.posonlyargs,
        *statement.args.args,
        *statement.args.kwonlyargs,
        statement.args.vararg,
        statement.args.kwarg,
    ]
    for parameter in parameters:
        if parameter is not None:
            yield parameter


def _walk_alias_scope(
    statement: ast.stmt,
    aliases: dict[str, str],
    walk: Callable[[Sequence[ast.stmt], dict[str, str]], bool | None],
    function_arguments: dict[str, dict[str, str]] | None = None,
) -> bool:
    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if statement.name == "range":
            aliases[statement.name] = _FLAT_SHADOWED_RANGE
        else:
            aliases.pop(statement.name, None)
        local = dict(aliases)
        for parameter in _function_parameters(statement):
            if parameter.arg == "range":
                local[parameter.arg] = _FLAT_SHADOWED_RANGE
            else:
                local.pop(parameter.arg, None)
            _clear_index_aliases(parameter.arg, local)
        for parameter_name, resolved in (
            (function_arguments or {}).get(statement.name, {}).items()
        ):
            local[parameter_name] = resolved
        walk(statement.body, local)
        declared = {
            name
            for node in ast.walk(statement)
            if isinstance(node, (ast.Global, ast.Nonlocal))
            for name in node.names
        }
        for name in declared:
            if name in local and local[name] is not None:
                aliases[name] = local[name]
        return True
    if isinstance(statement, ast.ClassDef):
        aliases.pop(statement.name, None)
        walk(statement.body, dict(aliases))
        if statement.name == "range":
            aliases[statement.name] = _FLAT_SHADOWED_RANGE
        return True
    return False


def _merge_flat_alias_branches(
    target: dict[str, str], branches: Sequence[dict[str, str]]
) -> None:
    keys: set[str] = set(target)
    for branch in branches:
        keys.update(branch)
    for key in keys:
        values = [branch[key] for branch in branches if key in branch]
        if not values:
            target.pop(key, None)
            continue
        preferred = _pick_candidate(values)
        if preferred is not None:
            target[key] = preferred
        else:
            target.pop(key, None)


FlatAliasWalk = Callable[[Sequence[ast.stmt], dict[str, str]], bool | None]


def _walk_if_aliases(
    statement: ast.If, aliases: dict[str, str], walk: FlatAliasWalk
) -> bool:
    body = dict(aliases)
    body_reachable = walk(statement.body, body)
    orelse = dict(aliases)
    else_reachable = walk(statement.orelse, orelse)
    branches: list[dict[str, str]] = []
    if body_reachable or _statements_escape_loop(statement.body):
        branches.append(body)
    if else_reachable or _statements_escape_loop(statement.orelse):
        branches.append(orelse)
    if not branches:
        _merge_flat_alias_branches(aliases, [body, orelse])
        return False
    _merge_flat_alias_branches(aliases, branches)
    return True


def _statement_escapes_loop(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.If):
        return _statements_escape_loop(statement.body) or _statements_escape_loop(
            statement.orelse
        )
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        return _statements_escape_loop(statement.body)
    if isinstance(statement, (ast.Try, ast.TryStar)):
        groups = [statement.body, statement.orelse, statement.finalbody]
        groups.extend(handler.body for handler in statement.handlers)
        return any(_statements_escape_loop(group) for group in groups)
    if isinstance(statement, ast.Match):
        return any(_statements_escape_loop(case.body) for case in statement.cases)
    return False


def _statements_escape_loop(statements: Sequence[ast.stmt]) -> bool:
    for statement in statements:
        if isinstance(statement, (ast.Break, ast.Continue)):
            return True
        if isinstance(statement, (ast.Return, ast.Raise)):
            return False
        if _statement_escapes_loop(statement):
            return True
    return False


def _statement_breaks_loop(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Break):
        return True
    if isinstance(statement, ast.If):
        return _statements_break_loop(statement.body) or _statements_break_loop(
            statement.orelse
        )
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        return _statements_break_loop(statement.body)
    if isinstance(statement, (ast.Try, ast.TryStar)):
        groups = [statement.body, statement.orelse, statement.finalbody]
        groups.extend(handler.body for handler in statement.handlers)
        return any(_statements_break_loop(group) for group in groups)
    if isinstance(statement, ast.Match):
        return any(_statements_break_loop(case.body) for case in statement.cases)
    return False


def _statements_break_loop(statements: Sequence[ast.stmt]) -> bool:
    for statement in statements:
        if isinstance(statement, (ast.Return, ast.Raise)):
            return False
        if _statement_breaks_loop(statement):
            return True
    return False


def _walk_loop_aliases(
    statement: ast.For | ast.AsyncFor | ast.While,
    aliases: dict[str, str],
    walk: FlatAliasWalk,
) -> bool:
    body = dict(aliases)
    if isinstance(statement, (ast.For, ast.AsyncFor)):
        for name in _iter_target_names(statement.target):
            if name.id == "range":
                body[name.id] = _FLAT_SHADOWED_RANGE
            else:
                body.pop(name.id, None)
            _clear_index_aliases(name.id, body)
    body_reachable = walk(statement.body, body)
    orelse = dict(aliases)
    walk(statement.orelse, orelse)
    escapes = _statements_escape_loop(statement.body)
    guaranteed_entry = isinstance(statement, ast.For) and (
        _iterable_guaranteed_nonempty(
            statement.iter, allow_range="range" not in aliases
        )
    )
    if guaranteed_entry and not statement.orelse:
        branches = [body]
    else:
        branches = [body, orelse] if body_reachable or escapes else [orelse]
    _merge_flat_alias_branches(aliases, branches)
    return True


def _walk_with_aliases(
    statement: ast.With | ast.AsyncWith,
    aliases: dict[str, str],
    walk: FlatAliasWalk,
) -> bool:
    inner = dict(aliases)
    for item in statement.items:
        if item.optional_vars is not None:
            _bind_alias_target(item.optional_vars, item.context_expr, inner)
    body_reachable = walk(statement.body, inner)
    _merge_flat_alias_branches(aliases, [inner])
    return bool(body_reachable)


def _walk_try_aliases(
    statement: ast.Try | ast.TryStar,
    aliases: dict[str, str],
    walk: FlatAliasWalk,
) -> bool:
    all_branches: list[dict[str, str]] = []
    reachable_flags: list[bool] = []
    body = dict(aliases)
    body_reachable = walk(statement.body, body)
    all_branches.append(body)
    reachable_flags.append(bool(body_reachable))
    for handler in statement.handlers:
        branch = dict(aliases)
        if handler.name is not None:
            if handler.name == "range":
                branch[handler.name] = _FLAT_SHADOWED_RANGE
            else:
                branch.pop(handler.name, None)
        reachable = bool(walk(handler.body, branch))
        all_branches.append(branch)
        reachable_flags.append(reachable)
    if body_reachable:
        orelse = dict(body)
        reachable = bool(walk(statement.orelse, orelse))
        all_branches.append(orelse)
        reachable_flags.append(reachable)
    final_input = dict(aliases)
    _merge_flat_alias_branches(final_input, all_branches)
    final_reachable = walk(statement.finalbody, final_input)
    _merge_flat_alias_branches(aliases, [final_input])
    return any(reachable_flags) and bool(final_reachable)


def _walk_match_aliases(
    statement: ast.Match, aliases: dict[str, str], walk: FlatAliasWalk
) -> bool:
    all_branches: list[dict[str, str]] = []
    reachable_branches: list[dict[str, str]] = []
    kept_branches: list[dict[str, str]] = []
    irrefutable = False
    for case in statement.cases:
        branch = dict(aliases)
        for name in _pattern_capture_names(case.pattern):
            if name == "range":
                branch[name] = _FLAT_SHADOWED_RANGE
            else:
                branch.pop(name, None)
        reachable = bool(walk(case.body, branch))
        all_branches.append(branch)
        if reachable:
            reachable_branches.append(branch)
        if reachable or _statements_escape_loop(case.body):
            kept_branches.append(branch)
        if _pattern_is_irrefutable(case.pattern) and case.guard is None:
            irrefutable = True
    if not irrefutable:
        default_branch = dict(aliases)
        all_branches.append(default_branch)
        reachable_branches.append(default_branch)
        kept_branches.append(default_branch)
    if kept_branches:
        _merge_flat_alias_branches(aliases, kept_branches)
    else:
        _merge_flat_alias_branches(aliases, all_branches)
    return bool(reachable_branches)


def _walk_alias_branches(
    statement: ast.stmt,
    aliases: dict[str, str],
    walk: FlatAliasWalk,
) -> bool | None:
    if isinstance(statement, ast.If):
        return _walk_if_aliases(statement, aliases, walk)
    if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
        return _walk_loop_aliases(statement, aliases, walk)
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        return _walk_with_aliases(statement, aliases, walk)
    if isinstance(statement, (ast.Try, ast.TryStar)):
        return _walk_try_aliases(statement, aliases, walk)
    if isinstance(statement, ast.Match):
        return _walk_match_aliases(statement, aliases, walk)
    return None


def _apply_alias_statement(statement: ast.stmt, aliases: dict[str, str]) -> None:
    if isinstance(statement, (ast.Import, ast.ImportFrom)):
        _apply_flat_import(statement, aliases)
    elif isinstance(statement, ast.Assign):
        for target in statement.targets:
            _bind_alias_target(target, statement.value, aliases)
    elif isinstance(statement, ast.AnnAssign):
        if statement.value is None:
            _remove_alias_target(statement.target, aliases)
        else:
            _bind_alias_target(statement.target, statement.value, aliases)
    elif isinstance(statement, ast.AugAssign):
        _remove_alias_target(statement.target, aliases)
    elif isinstance(statement, ast.Delete):
        for target in statement.targets:
            _remove_alias_target(target, aliases)
    else:
        for node in ast.walk(statement):
            if isinstance(node, ast.NamedExpr):
                _bind_alias_target(node.target, node.value, aliases)


class _Reachability:
    def __init__(self) -> None:
        self.result: set[int] = set()

    def mark_expression(self, node: ast.AST | None) -> None:
        if node is not None:
            self.result.update(id(child) for child in ast.walk(node))

    def visit(self, statements: Sequence[ast.stmt]) -> bool:
        reachable = True
        for statement in statements:
            if not reachable:
                break
            reachable = self._visit_statement(statement)
        return reachable

    def _visit_function(
        self, statement: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for decorator in statement.decorator_list:
            self.mark_expression(decorator)
        for default in [
            *statement.args.defaults,
            *[default for default in statement.args.kw_defaults if default is not None],
        ]:
            self.mark_expression(default)
        for argument in [
            *statement.args.posonlyargs,
            *statement.args.args,
            *statement.args.kwonlyargs,
            statement.args.vararg,
            statement.args.kwarg,
        ]:
            if argument is not None:
                self.mark_expression(argument.annotation)
        self.mark_expression(statement.returns)
        self.visit(statement.body)

    def _visit_class(self, statement: ast.ClassDef) -> None:
        for decorator in statement.decorator_list:
            self.mark_expression(decorator)
        for base in statement.bases:
            self.mark_expression(base)
        for keyword in statement.keywords:
            self.mark_expression(keyword.value)
        self.visit(statement.body)

    def _visit_if(self, statement: ast.If) -> bool:
        self.mark_expression(statement.test)
        body_reachable = self.visit(statement.body)
        else_reachable = self.visit(statement.orelse)
        return body_reachable or else_reachable

    def _visit_loop(self, statement: ast.For | ast.AsyncFor | ast.While) -> bool:
        if isinstance(statement, ast.While):
            self.mark_expression(statement.test)
        else:
            self.mark_expression(statement.iter)
            self.mark_expression(statement.target)
        self.visit(statement.body)
        self.visit(statement.orelse)
        return True

    def _visit_with(self, statement: ast.With | ast.AsyncWith) -> bool:
        for item in statement.items:
            self.mark_expression(item.context_expr)
            self.mark_expression(item.optional_vars)
        return self.visit(statement.body)

    def _visit_try(self, statement: ast.Try | ast.TryStar) -> bool:
        body_reachable = self.visit(statement.body)
        handler_reachable = False
        for handler in statement.handlers:
            self.mark_expression(handler.type)
            handler_reachable = self.visit(handler.body) or handler_reachable
        else_reachable = self.visit(statement.orelse) if body_reachable else False
        final_reachable = self.visit(statement.finalbody)
        return (
            body_reachable or handler_reachable or else_reachable
        ) and final_reachable

    def _visit_match(self, statement: ast.Match) -> bool:
        self.mark_expression(statement.subject)
        case_reachable = False
        irrefutable = False
        for case in statement.cases:
            self.mark_expression(case.guard)
            case_reachable = self.visit(case.body) or case_reachable
            if _pattern_is_irrefutable(case.pattern) and case.guard is None:
                irrefutable = True
        return case_reachable or not irrefutable

    def _visit_statement(self, statement: ast.stmt) -> bool:
        self.result.add(id(statement))
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._visit_function(statement)
            return True
        if isinstance(statement, ast.ClassDef):
            self._visit_class(statement)
            return True
        if isinstance(statement, ast.If):
            return self._visit_if(statement)
        if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            return self._visit_loop(statement)
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            return self._visit_with(statement)
        if isinstance(statement, (ast.Try, ast.TryStar)):
            return self._visit_try(statement)
        if isinstance(statement, ast.Match):
            return self._visit_match(statement)
        if isinstance(statement, ast.Return):
            self.mark_expression(statement.value)
            return False
        if isinstance(statement, ast.Raise):
            self.mark_expression(statement.exc)
            return False
        if isinstance(statement, (ast.Break, ast.Continue)):
            return False
        self.mark_expression(statement)
        return True


def _reachable_node_ids(tree: ast.Module) -> set[int]:
    reachability = _Reachability()
    reachability.visit(tree.body)
    return reachability.result


def _node_runtime_aliases(
    tree: ast.Module,
    imported: dict[str, str],
    function_arguments: dict[str, dict[str, str]] | None = None,
) -> dict[int, dict[str, str]]:
    """Map every node to the import aliases visible where it is evaluated.

    The replay is sequential and scope-aware.  Branch bodies are analysed with
    their own copies, so a conditional rebinding does not erase the alias seen
    by an earlier or unconditional access.
    """
    states: dict[int, dict[str, str]] = {}

    def record(statement: ast.AST, aliases: dict[str, str]) -> None:
        snapshot = dict(aliases)
        for node in ast.walk(statement):
            states[id(node)] = snapshot

    def walk(statements: Sequence[ast.stmt], aliases: dict[str, str]) -> bool:
        for statement in statements:
            record(statement, aliases)
            if _walk_alias_scope(statement, aliases, walk, function_arguments):
                continue
            branch_reachable = _walk_alias_branches(statement, aliases, walk)
            if branch_reachable is not None:
                if not branch_reachable:
                    return False
                continue
            _apply_alias_statement(statement, aliases)
            if isinstance(statement, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                return False
        return True

    walk(tree.body, dict(imported))
    reachable = _reachable_node_ids(tree)
    return {node: state for node, state in states.items() if node in reachable}


def _imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
    return names


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


def _relative_import_base(root: Path, path: Path, level: int) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    is_package = bool(parts and parts[-1] == "__init__")
    if is_package:
        parts.pop()
    absolute = [root.name, *parts]
    drop = level - 1 if is_package else level
    return ".".join(absolute[: max(0, len(absolute) - drop)])


def _file_import_aliases(tree: ast.Module, path: Path, root: Path) -> dict[str, str]:
    aliases = _import_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        base = _relative_import_base(root, path, node.level)
        if node.module:
            base = f"{base}.{node.module}" if base else node.module
        for alias in node.names:
            target = f"{base}.{alias.name}" if base else alias.name
            aliases[alias.asname or alias.name] = target
    return aliases


FlowAliases = dict[str, frozenset[str]]
CallContext = tuple[ast.Call, FlowAliases, set[str]]


def _calls_in_expression(
    node: ast.AST, aliases: FlowAliases, path_names: set[str]
) -> Iterator[CallContext]:
    static = _import_aliases_from_alias_sets(aliases)
    yield from _expression_calls(node, aliases, path_names, static)


def _expression_calls(
    node: ast.AST,
    aliases: FlowAliases,
    path_names: set[str],
    static: dict[str, str],
) -> Iterator[CallContext]:
    if isinstance(node, ast.NamedExpr):
        yield from _expression_calls(node.value, aliases, path_names, static)
        _assign_target_aliases([node.target], node.value, aliases, static, path_names)
        return
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        comprehension_paths = set(path_names)
        comprehension_aliases = dict(aliases)
        for generator in node.generators:
            yield from _expression_calls(
                generator.iter, comprehension_aliases, comprehension_paths, static
            )
            if _iterable_contains_path(generator.iter, static, comprehension_paths):
                _add_path_names([generator.target], comprehension_paths)
            else:
                _remove_path_names([generator.target], comprehension_paths)
            _assign_names(
                generator.target,
                _iterable_callable_names(
                    generator.iter, comprehension_aliases, static, comprehension_paths
                ),
                comprehension_aliases,
            )
            for condition in generator.ifs:
                yield from _expression_calls(
                    condition, comprehension_aliases, comprehension_paths, static
                )
        if isinstance(node, ast.DictComp):
            yield from _expression_calls(
                node.key, comprehension_aliases, comprehension_paths, static
            )
            yield from _expression_calls(
                node.value, comprehension_aliases, comprehension_paths, static
            )
        else:
            yield from _expression_calls(
                node.elt, comprehension_aliases, comprehension_paths, static
            )
        return
    if isinstance(node, ast.Call):
        call_aliases = dict(aliases)
        for child in ast.iter_child_nodes(node):
            yield from _expression_calls(child, aliases, path_names, static)
        yield node, call_aliases, set(path_names)
        return
    for child in ast.iter_child_nodes(node):
        yield from _expression_calls(child, aliases, path_names, static)


def _resolve_wrapper_names(
    node: ast.AST, aliases: FlowAliases
) -> frozenset[str] | None:
    if isinstance(node, ast.NamedExpr):
        return _resolve_possible_names(node.value, aliases)
    if isinstance(node, ast.IfExp):
        return _resolve_possible_names(node.body, aliases) | _resolve_possible_names(
            node.orelse, aliases
        )
    if isinstance(node, ast.BoolOp):
        result: set[str] = set()
        for value in node.values:
            result.update(_resolve_possible_names(value, aliases))
        return frozenset(result)
    return None


def _resolve_subscript_names(
    node: ast.AST, aliases: FlowAliases
) -> frozenset[str] | None:
    if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Constant):
        return None
    if isinstance(node.value, (ast.Tuple, ast.List)) and isinstance(
        node.slice.value, int
    ):
        elements = node.value.elts
        if -len(elements) <= node.slice.value < len(elements):
            return _resolve_possible_names(elements[node.slice.value], aliases)
        return frozenset()
    if isinstance(node.value, ast.Dict):
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and key.value == node.slice.value:
                return _resolve_possible_names(value, aliases)
        return frozenset()
    if isinstance(node.value, ast.Name):
        alias_key = f"{node.value.id}[{node.slice.value!r}]"
        if alias_key in aliases:
            return aliases[alias_key]
        if node.value.id in aliases:
            return aliases[node.value.id]
    indirect = _subscript_indirect_names(node, aliases)
    if indirect:
        return frozenset(_normalise_builtins_name(name) for name in indirect)
    return None


def _resolve_call_names(node: ast.Call, aliases: FlowAliases) -> frozenset[str]:
    call_results: set[str] = set()
    func_names = _resolve_possible_names(node.func, aliases)
    if not func_names and isinstance(node.func, ast.Name):
        func_names = frozenset({node.func.id})
    for func_name in func_names:
        call_results.update(aliases.get(f"{func_name}()", frozenset()))
    return frozenset(call_results)


def _resolve_possible_names(node: ast.AST, aliases: FlowAliases) -> frozenset[str]:
    wrapper = _resolve_wrapper_names(node, aliases)
    if wrapper is not None:
        return wrapper
    subscript = _resolve_subscript_names(node, aliases)
    if subscript is not None:
        return subscript
    if isinstance(node, ast.Call):
        indirect = _literal_indirect_names(node, aliases)
        if indirect:
            return frozenset(_normalise_builtins_name(name) for name in indirect)
        return _resolve_call_names(node, aliases)
    if isinstance(node, ast.Name):
        return aliases.get(node.id, frozenset({node.id}))
    if isinstance(node, ast.Attribute):
        dotted = _dotted_name(node)
        if dotted is not None and dotted in aliases:
            return aliases[dotted]
        bases = _resolve_possible_names(node.value, aliases)
        return frozenset(
            _normalise_builtins_name(f"{base}.{node.attr}") for base in bases
        )
    return frozenset()


def _iter_target_names(target: ast.AST) -> Iterator[ast.Name]:
    if isinstance(target, ast.Name):
        yield target
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _iter_target_names(element)
    elif isinstance(target, ast.Starred):
        yield from _iter_target_names(target.value)


def _clear_flow_index_aliases(name: str, aliases: FlowAliases) -> None:
    prefix = f"{name}["
    for key in [key for key in aliases if key.startswith(prefix)]:
        aliases.pop(key, None)


def _remove_target_aliases(targets: Sequence[ast.AST], aliases: FlowAliases) -> None:
    for target in targets:
        for name in _iter_target_names(target):
            aliases[name.id] = frozenset()
            _clear_identity_binding(name.id, aliases)
            _clear_flow_index_aliases(name.id, aliases)
        dotted = _dotted_name(target)
        if dotted is not None:
            aliases[dotted] = frozenset()


def _assign_names(
    target: ast.AST, values: frozenset[str], aliases: FlowAliases
) -> None:
    for name in _iter_target_names(target):
        _clear_identity_binding(name.id, aliases)
        _clear_flow_index_aliases(name.id, aliases)
        aliases[name.id] = values
    dotted = _dotted_name(target)
    if dotted is not None:
        aliases[dotted] = values


def _literal_sequence_element(value: ast.AST) -> ast.AST | None:
    if (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, (ast.Tuple, ast.List))
        and isinstance(value.slice, ast.Constant)
        and isinstance(value.slice.value, int)
    ):
        elements = value.value.elts
        if -len(elements) <= value.slice.value < len(elements):
            return elements[value.slice.value]
        return None
    if (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Dict)
        and isinstance(value.slice, ast.Constant)
    ):
        for key, element in zip(value.value.keys, value.value.values):
            if isinstance(key, ast.Constant) and key.value == value.slice.value:
                return element
    return None


def _compound_value_names(
    value: ast.AST,
    aliases: FlowAliases,
    path_aliases: dict[str, str] | None,
    path_names: set[str] | None,
) -> frozenset[str] | None:
    if isinstance(value, ast.NamedExpr):
        return _value_callable_names(value.value, aliases, path_aliases, path_names)
    if isinstance(value, ast.IfExp):
        return _value_callable_names(
            value.body, aliases, path_aliases, path_names
        ) | _value_callable_names(value.orelse, aliases, path_aliases, path_names)
    if isinstance(value, ast.BoolOp):
        bool_values: set[str] = set()
        for element in value.values:
            bool_values.update(
                _value_callable_names(element, aliases, path_aliases, path_names)
            )
        return frozenset(bool_values)
    if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
        result: set[str] = set()
        for element in value.elts:
            result.update(
                _value_callable_names(element, aliases, path_aliases, path_names)
            )
        return frozenset(result)
    if isinstance(value, ast.Dict):
        result = set()
        for key in value.keys:
            if key is not None:
                result.update(
                    _value_callable_names(key, aliases, path_aliases, path_names)
                )
        return frozenset(result)
    return None


def _comprehension_callable_names(
    value: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
    aliases: FlowAliases,
    path_aliases: dict[str, str] | None,
    path_names: set[str] | None,
) -> frozenset[str]:
    result: set[str] = set()
    for generator in value.generators:
        result.update(
            _iterable_callable_names(generator.iter, aliases, path_aliases, path_names)
        )
    if isinstance(value, ast.DictComp):
        result.update(
            _value_callable_names(value.key, aliases, path_aliases, path_names)
        )
        result.update(
            _value_callable_names(value.value, aliases, path_aliases, path_names)
        )
    else:
        result.update(
            _value_callable_names(value.elt, aliases, path_aliases, path_names)
        )
    return frozenset(result)


def _lambda_callable_names(
    value: ast.Lambda,
    aliases: FlowAliases,
    path_aliases: dict[str, str] | None,
    path_names: set[str] | None,
) -> frozenset[str]:
    result = set(_value_callable_names(value.body, aliases, path_aliases, path_names))
    defaults = [
        *value.args.defaults,
        *[default for default in value.args.kw_defaults if default is not None],
    ]
    for default in defaults:
        result.update(_value_callable_names(default, aliases, path_aliases, path_names))
    return frozenset(result)


def _mapping_values_callable_names(
    call: ast.Call,
    aliases: FlowAliases,
    path_aliases: dict[str, str] | None,
    path_names: set[str] | None,
) -> frozenset[str] | None:
    function = call.func
    if not isinstance(function, ast.Attribute):
        return None
    if function.attr not in {"values", "items"}:
        return None
    if call.args or call.keywords:
        return None
    receiver = function.value
    receiver_key = _subscript_alias_key(receiver)
    if receiver_key is not None:
        prefix = f"{receiver_key}["
        names: set[str] = set()
        for key, values in aliases.items():
            if key.startswith(prefix):
                names.update(values)
        return frozenset(names) if names else None
    if isinstance(receiver, ast.Dict):
        result: set[str] = set()
        for element in receiver.values:
            result.update(
                _value_callable_names(element, aliases, path_aliases, path_names)
            )
        return frozenset(result)
    return None


def _value_callable_names(
    value: ast.AST,
    aliases: FlowAliases,
    path_aliases: dict[str, str] | None,
    path_names: set[str] | None,
) -> frozenset[str]:
    indirect = _literal_indirect_names(value, aliases, path_names, path_aliases)
    if indirect:
        return indirect
    element = _literal_sequence_element(value)
    if element is not None:
        return _value_callable_names(element, aliases, path_aliases, path_names)
    compound = _compound_value_names(value, aliases, path_aliases, path_names)
    if compound is not None:
        return compound
    if isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
        return _comprehension_callable_names(value, aliases, path_aliases, path_names)
    if isinstance(value, ast.Lambda):
        return _lambda_callable_names(value, aliases, path_aliases, path_names)
    if isinstance(value, ast.Call):
        mapping_values = _mapping_values_callable_names(
            value, aliases, path_aliases, path_names
        )
        if mapping_values is not None:
            return mapping_values
        if (
            isinstance(value.func, ast.Name)
            and value.func.id in {"list", "tuple", "set", "iter", "sorted"}
            and value.args
        ):
            return _iterable_callable_names(
                value.args[0], aliases, path_aliases, path_names
            )
        return _resolve_possible_names(value, aliases)
    if isinstance(value, ast.Subscript):
        return _resolve_possible_names(value, aliases)
    if isinstance(value, ast.Attribute) and path_names is not None:
        static = path_aliases or {}
        if _is_path_expression(value.value, static, path_names):
            if value.attr in PATH_PROPERTIES or value.attr in PATH_PURE_METHODS:
                return frozenset()
            return frozenset({f"Path.{value.attr}"})
    if isinstance(value, (ast.Name, ast.Attribute)):
        return _resolve_possible_names(value, aliases)
    return frozenset()


def _value_references(
    node: ast.AST,
    aliases: FlowAliases,
    path_names: set[str] | None = None,
    path_aliases: dict[str, str] | None = None,
) -> Iterator[tuple[ast.Name | ast.Attribute | ast.Subscript, frozenset[str]]]:
    """Yield callable references used as values, not as call targets."""
    if isinstance(node, ast.Call):
        for argument in node.args:
            yield from _value_references(argument, aliases, path_names, path_aliases)
        for keyword in node.keywords:
            yield from _value_references(
                keyword.value, aliases, path_names, path_aliases
            )
        return
    if isinstance(node, ast.Subscript):
        names = _literal_indirect_names(node, aliases)
        if names:
            yield node, names
        for child in ast.iter_child_nodes(node):
            yield from _value_references(child, aliases, path_names, path_aliases)
        return
    if isinstance(node, (ast.Name, ast.Attribute)):
        names = _resolve_possible_names(node, aliases)
        if path_names is not None:
            names |= _value_callable_names(node, aliases, path_aliases, path_names)
        if names:
            yield node, names
        return
    for child in ast.iter_child_nodes(node):
        yield from _value_references(child, aliases, path_names, path_aliases)


def _lambda_reference_names(
    tree: ast.Module, aliases: dict[str, str]
) -> Iterator[tuple[ast.Lambda, frozenset[str]]]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Lambda):
            continue
        expressions = [
            node.body,
            *node.args.defaults,
            *[default for default in node.args.kw_defaults if default is not None],
        ]
        names: set[str] = set()
        for expression in expressions:
            for child in ast.walk(expression):
                if isinstance(child, ast.Call):
                    name = _resolve_callable_name(child.func, aliases)
                elif isinstance(child, (ast.Name, ast.Attribute)):
                    name = _resolve_callable_name(child, aliases)
                else:
                    continue
                if name is not None:
                    names.add(name)
        yield node, frozenset(names)


def _bind_flow_index_aliases(
    name: str,
    value: ast.AST,
    aliases: FlowAliases,
    path_aliases: dict[str, str] | None,
    path_names: set[str] | None,
) -> None:
    elements: list[tuple[str, ast.AST]] = []
    if isinstance(value, (ast.List, ast.Tuple)):
        elements = [(str(index), element) for index, element in enumerate(value.elts)]
    elif isinstance(value, ast.Dict):
        elements = [
            (repr(key.value), element)
            for key, element in zip(value.keys, value.values)
            if isinstance(key, ast.Constant)
        ]
    elif isinstance(value, ast.Name):
        prefix = f"{value.id}["
        for key in [key for key in aliases if key.startswith(prefix)]:
            aliases[f"{name}{key[len(value.id) :]}"] = aliases.pop(key)
    for suffix, element in elements:
        key = f"{name}[{suffix}]"
        aliases[key] = _value_callable_names(element, aliases, path_aliases, path_names)
        if isinstance(element, (ast.List, ast.Tuple, ast.Dict)):
            _bind_flow_index_aliases(key, element, aliases, path_aliases, path_names)


def _assign_target(
    target: ast.AST,
    value: ast.AST,
    aliases: FlowAliases,
    path_aliases: dict[str, str] | None,
    path_names: set[str] | None,
) -> None:
    if isinstance(target, (ast.Tuple, ast.List)):
        if isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(
            value.elts
        ):
            for element_target, element_value in zip(target.elts, value.elts):
                _assign_target(
                    element_target,
                    element_value,
                    aliases,
                    path_aliases,
                    path_names,
                )
        else:
            _assign_names(
                target,
                _value_callable_names(value, aliases, path_aliases, path_names),
                aliases,
            )
        return
    if isinstance(target, ast.Starred):
        _assign_target(target.value, value, aliases, path_aliases, path_names)
        return
    if isinstance(target, ast.Subscript):
        receiver = _subscript_alias_key(target.value)
        key = _literal_subscript_key(target)
        if receiver is not None and key is not None:
            alias_key = f"{receiver}[{key}]"
            _clear_flow_index_aliases(alias_key, aliases)
            aliases[alias_key] = _value_callable_names(
                value, aliases, path_aliases, path_names
            )
            _bind_flow_index_aliases(
                alias_key, value, aliases, path_aliases, path_names
            )
        return
    if isinstance(target, ast.Name):
        _clear_identity_binding(target.id, aliases)
        _clear_flow_index_aliases(target.id, aliases)
        if _identity_lambda_parameter(value) is not None:
            aliases[target.id] = frozenset({_identity_alias_key(target.id)})
        else:
            aliases[target.id] = _value_callable_names(
                value, aliases, path_aliases, path_names
            )
        _bind_flow_index_aliases(target.id, value, aliases, path_aliases, path_names)
        return
    dotted = _dotted_name(target)
    if dotted is not None:
        aliases[dotted] = _value_callable_names(
            value, aliases, path_aliases, path_names
        )


def _assign_target_aliases(
    targets: Sequence[ast.AST],
    value: ast.AST,
    aliases: FlowAliases,
    path_aliases: dict[str, str] | None = None,
    path_names: set[str] | None = None,
) -> None:
    if (
        len(targets) == 1
        and isinstance(targets[0], (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
    ):
        _assign_target(targets[0], value, aliases, path_aliases, path_names)
        return
    for target in targets:
        _assign_target(target, value, aliases, path_aliases, path_names)


def _apply_named_expressions(
    node: ast.AST,
    aliases: FlowAliases,
    path_aliases: dict[str, str] | None,
    path_names: set[str] | None,
) -> None:
    for child in ast.walk(node):
        if isinstance(child, ast.NamedExpr) and isinstance(child.target, ast.Name):
            _assign_target_aliases(
                [child.target], child.value, aliases, path_aliases, path_names
            )


def _iterable_callable_names(
    node: ast.AST,
    aliases: FlowAliases,
    path_aliases: dict[str, str] | None = None,
    path_names: set[str] | None = None,
) -> frozenset[str]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        result: set[str] = set()
        for element in node.elts:
            result.update(
                _value_callable_names(element, aliases, path_aliases, path_names)
            )
        return frozenset(result)
    if isinstance(node, ast.Dict):
        keys: set[str] = set()
        for key in node.keys:
            if key is not None:
                keys.update(
                    _value_callable_names(key, aliases, path_aliases, path_names)
                )
        return frozenset(keys)
    if isinstance(node, (ast.Name, ast.Attribute, ast.NamedExpr, ast.Call)):
        return _value_callable_names(node, aliases, path_aliases, path_names)
    return frozenset()


def _apply_scope_import(stmt: ast.stmt, aliases: FlowAliases) -> None:
    if isinstance(stmt, ast.Import):
        for alias in stmt.names:
            local = alias.asname or alias.name.split(".")[0]
            resolved = alias.name if alias.asname else alias.name.split(".")[0]
            _clear_identity_binding(local, aliases)
            aliases[local] = frozenset({resolved})
    elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and stmt.module:
        for alias in stmt.names:
            local = alias.asname or alias.name
            _clear_identity_binding(local, aliases)
            aliases[local] = frozenset({f"{stmt.module}.{alias.name}"})


def _merge_alias_sets(target: FlowAliases, *branches: FlowAliases) -> None:
    keys: set[str] = set(target)
    for branch in branches:
        keys.update(branch)
    for key in keys:
        values: set[str] = set()
        for branch in branches:
            if key in branch:
                values.update(branch[key])
            else:
                values.update(target.get(key, frozenset({key})))
        target[key] = frozenset(values)
    for key in [key for key in target if key.startswith(IDENTITY_ALIAS_PREFIX)]:
        if not any(key in branch for branch in branches):
            target.pop(key, None)


def _scope_nodes(body: Sequence[ast.stmt]) -> Iterator[ast.AST]:
    stack: list[ast.AST] = list(reversed(body))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _clear_index_path_names(name: str, names: set[str]) -> None:
    prefix = f"{name}["
    for key in [key for key in names if key.startswith(prefix)]:
        names.discard(key)


def _remove_path_names(targets: Sequence[ast.AST], names: set[str]) -> None:
    for target in targets:
        if isinstance(target, ast.Name):
            names.discard(target.id)
            _clear_index_path_names(target.id, names)
            prefix = f"{target.id}."
            for key in [key for key in names if key.startswith(prefix)]:
                names.discard(key)
        elif isinstance(target, ast.Attribute):
            dotted = _dotted_name(target)
            if dotted is not None:
                names.discard(dotted)
                prefix = f"{dotted}."
                for key in [key for key in names if key.startswith(prefix)]:
                    names.discard(key)
            if isinstance(target.value, ast.Name) and target.value.id in {
                "self",
                "cls",
            }:
                names.discard(f"{target.value.id}.{target.attr}")
        elif isinstance(target, (ast.Tuple, ast.List)):
            _remove_path_names(target.elts, names)


def _replace_path_names(target: set[str], values: set[str]) -> None:
    target.clear()
    target.update(values)


def _iterable_contains_path(
    node: ast.AST, aliases: dict[str, str], names: set[str]
) -> bool:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return any(
            _is_path_expression(element, aliases, names) for element in node.elts
        )
    if isinstance(node, ast.IfExp):
        return _iterable_contains_path(node.body, aliases, names) or (
            _iterable_contains_path(node.orelse, aliases, names)
        )
    return _is_path_expression(node, aliases, names)


def _apply_container_path_names(
    name: str,
    value: ast.AST,
    aliases: dict[str, str],
    names: set[str],
) -> None:
    _clear_index_path_names(name, names)
    elements: list[tuple[str, ast.AST]] = []
    if isinstance(value, (ast.List, ast.Tuple)):
        elements = [(str(index), element) for index, element in enumerate(value.elts)]
    elif isinstance(value, ast.Dict):
        elements = [
            (repr(key.value), element)
            for key, element in zip(value.keys, value.values)
            if isinstance(key, ast.Constant)
        ]
    elif isinstance(value, ast.Name):
        prefix = f"{value.id}["
        for key in [key for key in names if key.startswith(prefix)]:
            names.add(f"{name}{key[len(value.id) :]}")
    for suffix, element in elements:
        key = f"{name}[{suffix}]"
        if _is_path_expression(element, aliases, names):
            names.add(key)
        else:
            names.discard(key)


def _apply_binding_path(
    target: ast.AST,
    value: ast.AST | None,
    aliases: dict[str, str],
    names: set[str],
) -> None:
    if isinstance(target, (ast.Tuple, ast.List)):
        if isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(
            value.elts
        ):
            for element_target, element_value in zip(target.elts, value.elts):
                _apply_binding_path(element_target, element_value, aliases, names)
        else:
            _remove_path_names([target], names)
        return
    if isinstance(target, ast.Starred):
        _apply_binding_path(target.value, value, aliases, names)
        return
    is_path = value is not None and _is_path_expression(value, aliases, names)
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        is_path = any(
            _is_path_expression(element, aliases, names) for element in value.elts
        )
    if isinstance(value, ast.Lambda):
        is_path = _is_path_expression(value.body, aliases, names)
    if is_path:
        _add_path_names([target], names)
    else:
        _remove_path_names([target], names)
    if isinstance(target, ast.Name) and value is not None:
        _apply_container_path_names(target.id, value, aliases, names)


def _apply_assign_path(
    node: ast.Assign, aliases: dict[str, str], names: set[str]
) -> None:
    for target in node.targets:
        _apply_binding_path(target, node.value, aliases, names)


def _apply_ann_assign_path(
    node: ast.AnnAssign, aliases: dict[str, str], names: set[str]
) -> None:
    if _annotation_contains_path(node.annotation, aliases):
        _add_path_names([node.target], names)
    elif node.value is not None:
        _apply_binding_path(node.target, node.value, aliases, names)


def _apply_named_expr_path(
    node: ast.NamedExpr, aliases: dict[str, str], names: set[str]
) -> None:
    _apply_binding_path(node.target, node.value, aliases, names)


def _apply_for_path(
    node: ast.For | ast.AsyncFor, aliases: dict[str, str], names: set[str]
) -> None:
    if _iterable_contains_path(node.iter, aliases, names):
        _add_path_names([node.target], names)
    else:
        _remove_path_names([node.target], names)


def _apply_with_path(
    node: ast.With | ast.AsyncWith, aliases: dict[str, str], names: set[str]
) -> None:
    for item in node.items:
        if item.optional_vars is None:
            continue
        _apply_binding_path(item.optional_vars, item.context_expr, aliases, names)


def _apply_comprehension_path(
    node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
    aliases: dict[str, str],
    names: set[str],
) -> None:
    for generator in node.generators:
        if _iterable_contains_path(generator.iter, aliases, names):
            _add_path_names([generator.target], names)
        else:
            _remove_path_names([generator.target], names)


_PathHandler = Callable[[Any, dict[str, str], set[str]], None]
_PATH_HANDLERS: dict[type[ast.AST], _PathHandler] = {
    ast.Assign: _apply_assign_path,
    ast.AnnAssign: _apply_ann_assign_path,
    ast.NamedExpr: _apply_named_expr_path,
    ast.For: _apply_for_path,
    ast.AsyncFor: _apply_for_path,
    ast.With: _apply_with_path,
    ast.AsyncWith: _apply_with_path,
    ast.ListComp: _apply_comprehension_path,
    ast.SetComp: _apply_comprehension_path,
    ast.DictComp: _apply_comprehension_path,
    ast.GeneratorExp: _apply_comprehension_path,
}


def _apply_expression_bindings(
    node: ast.AST, aliases: dict[str, str], names: set[str]
) -> None:
    if isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    ):
        return
    if isinstance(node, ast.NamedExpr):
        _apply_named_expr_path(node, aliases, names)
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        _apply_comprehension_path(node, dict(aliases), set(names))
    for child in ast.iter_child_nodes(node):
        _apply_expression_bindings(child, aliases, names)


def _path_statements(
    statements: Sequence[ast.stmt], aliases: dict[str, str], names: set[str]
) -> set[str]:
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if statement.returns is not None and _annotation_contains_path(
                statement.returns, aliases
            ):
                names.add(statement.name)
            continue
        if isinstance(statement, ast.ClassDef):
            continue
        if isinstance(statement, ast.If):
            _apply_expression_bindings(statement.test, aliases, names)
            body = _path_statements(statement.body, aliases, set(names))
            orelse = _path_statements(statement.orelse, aliases, set(names))
            names = body | orelse
        elif isinstance(statement, (ast.Try, ast.TryStar)):
            branches = [
                _path_statements(statement.body, aliases, set(names)),
                *[
                    _path_statements(
                        handler.body,
                        aliases,
                        set(names) - ({handler.name} if handler.name else set()),
                    )
                    for handler in statement.handlers
                ],
                _path_statements(statement.orelse, aliases, set(names)),
            ]
            names = _path_statements(
                statement.finalbody, aliases, set().union(*branches)
            )
        elif isinstance(statement, (ast.For, ast.AsyncFor)):
            _apply_expression_bindings(statement.iter, aliases, names)
            _apply_for_path(statement, aliases, names)
            names |= _path_statements(statement.body, aliases, set(names))
            names = _path_statements(statement.orelse, aliases, names)
        elif isinstance(statement, ast.While):
            _apply_expression_bindings(statement.test, aliases, names)
            names |= _path_statements(statement.body, aliases, set(names))
            names = _path_statements(statement.orelse, aliases, names)
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            for item in statement.items:
                _apply_expression_bindings(item.context_expr, aliases, names)
            _apply_with_path(statement, aliases, names)
            names = _path_statements(statement.body, aliases, names)
        elif isinstance(statement, ast.Match):
            branch_names = [
                _path_statements(case.body, aliases, set(names))
                for case in statement.cases
            ]
            names = set().union(*branch_names) if branch_names else set(names)
        else:
            handler = _PATH_HANDLERS.get(type(statement))
            if handler is not None:
                handler(statement, aliases, names)
            _apply_expression_bindings(statement, aliases, names)
    return names


def _scope_path_names(
    body: Sequence[ast.stmt], aliases: dict[str, str], inherited: set[str]
) -> set[str]:
    return _path_statements(body, aliases, set(inherited))


def _function_returns_path(
    body: Sequence[ast.stmt], aliases: dict[str, str], path_names: set[str]
) -> bool:
    return any(
        isinstance(node, ast.Return)
        and node.value is not None
        and _is_path_expression(node.value, aliases, path_names)
        for node in _scope_nodes(body)
    )


def _iter_function_calls(
    stmt: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: FlowAliases,
    path_names: set[str],
    global_aliases: FlowAliases | None,
) -> Iterator[CallContext]:
    for decorator in stmt.decorator_list:
        yield from _calls_in_expression(decorator, aliases, path_names)
    for default in stmt.args.defaults:
        yield from _calls_in_expression(default, aliases, path_names)
    for kw_default in stmt.args.kw_defaults:
        if kw_default is not None:
            yield from _calls_in_expression(kw_default, aliases, path_names)
    for argument in [
        *stmt.args.posonlyargs,
        *stmt.args.args,
        *stmt.args.kwonlyargs,
        stmt.args.vararg,
        stmt.args.kwarg,
    ]:
        if argument is not None and argument.annotation is not None:
            yield from _calls_in_expression(argument.annotation, aliases, path_names)
    if stmt.returns is not None:
        yield from _calls_in_expression(stmt.returns, aliases, path_names)
    identity_key = _identity_alias_key(stmt.name)
    if _identity_function_parameter(stmt) is not None:
        aliases[identity_key] = frozenset({stmt.name})
        aliases[stmt.name] = frozenset({identity_key})
    else:
        aliases.pop(identity_key, None)
        aliases[stmt.name] = frozenset()
    static = _import_aliases_from_alias_sets(aliases)
    base_aliases = dict(global_aliases or {})
    base_aliases.update(aliases)
    bindings, path_parameters, path_attributes = _function_call_bindings(
        stmt.name, global_aliases
    )
    inner_paths = _function_argument_paths(
        stmt,
        aliases,
        static,
        path_names,
        base_aliases,
        bindings,
        path_parameters,
        path_attributes,
    )
    yield from _iter_calls_with_aliases(
        stmt.body, base_aliases, inner_paths, global_aliases
    )
    declared = {
        name
        for node in ast.walk(stmt)
        if isinstance(node, (ast.Global, ast.Nonlocal))
        for name in node.names
    }
    for name in declared:
        resolved = base_aliases.get(name)
        if resolved:
            aliases[name] = resolved
    returns_path = (
        stmt.returns is not None and _annotation_contains_path(stmt.returns, static)
    ) or _function_returns_path(stmt.body, static, inner_paths)
    if returns_path:
        path_names.add(stmt.name)
    else:
        path_names.discard(stmt.name)


def _function_argument_paths(
    stmt: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: FlowAliases,
    static: dict[str, str],
    path_names: set[str],
    base_aliases: FlowAliases,
    bindings: dict[str, frozenset[str]] | None = None,
    path_parameters: set[str] | None = None,
    path_attributes: dict[str, set[str]] | None = None,
) -> set[str]:
    """Bind defaults and ``Path`` annotations of function parameters."""
    inner_paths = set(path_names)
    positional = [*stmt.args.posonlyargs, *stmt.args.args]
    defaults: dict[str, ast.AST] = {}
    default_start = len(positional) - len(stmt.args.defaults)
    for argument, default in zip(positional[default_start:], stmt.args.defaults):
        defaults[argument.arg] = default
    for argument, kw_default in zip(stmt.args.kwonlyargs, stmt.args.kw_defaults):
        if kw_default is not None:
            defaults[argument.arg] = kw_default
    arguments = [
        *positional,
        *stmt.args.kwonlyargs,
        stmt.args.vararg,
        stmt.args.kwarg,
    ]
    for parameter in arguments:
        if parameter is None:
            continue
        default_value = defaults.get(parameter.arg)
        base_aliases[parameter.arg] = (
            _value_callable_names(default_value, aliases, static, path_names)
            if default_value is not None
            else frozenset()
        )
        bound = (bindings or {}).get(parameter.arg)
        if bound is not None:
            base_aliases[parameter.arg] = bound
        is_path = parameter.annotation is not None and _annotation_contains_path(
            parameter.annotation, static
        )
        if parameter.arg in (path_parameters or set()):
            is_path = True
        if default_value is not None and _is_path_expression(
            default_value, static, path_names
        ):
            is_path = True
        if is_path:
            inner_paths.add(parameter.arg)
        else:
            inner_paths.discard(parameter.arg)
        for field_name in (path_attributes or {}).get(parameter.arg, set()):
            inner_paths.add(f"{parameter.arg}.{field_name}")
        if parameter.annotation is None:
            continue
        class_fields = _class_path_fields_from_aliases(static)
        if not class_fields:
            continue
        for class_name in _annotation_instance_classes(
            parameter.annotation, static, set(class_fields)
        ):
            for field_name in class_fields.get(class_name, set()):
                inner_paths.add(f"{parameter.arg}.{field_name}")
    return inner_paths


def _record_argument_binding(
    function_name: str,
    parameter_name: str,
    argument: ast.AST,
    call_aliases: FlowAliases,
    static: dict[str, str],
    call_paths: set[str],
    class_fields: dict[str, set[str]],
    class_names: set[str],
    values: dict[str, dict[str, set[str]]],
    path_parameters: dict[str, set[str]],
    path_attributes: dict[str, dict[str, set[str]]],
) -> None:
    argument_names = _value_callable_names(argument, call_aliases, static, call_paths)
    values.setdefault(function_name, {}).setdefault(parameter_name, set()).update(
        argument_names
    )
    if _is_path_expression(argument, static, call_paths):
        path_parameters.setdefault(function_name, set()).add(parameter_name)
    for class_name in _value_instance_classes(argument, static, class_names):
        fields = class_fields.get(class_name, set())
        if fields:
            path_attributes.setdefault(function_name, {}).setdefault(
                parameter_name, set()
            ).update(fields)


def _call_binding_flow(
    values: dict[str, dict[str, set[str]]],
    path_parameters: dict[str, set[str]],
    path_attributes: dict[str, dict[str, set[str]]],
) -> tuple[FlowAliases, dict[str, dict[str, str]]]:
    flow: FlowAliases = {}
    plain: dict[str, dict[str, str]] = {}
    for name, bound_parameters in values.items():
        for parameter_name, parameter_values in bound_parameters.items():
            flow[f"{name}.__arg__{parameter_name}"] = frozenset(parameter_values)
            preferred = _pick_candidate(parameter_values)
            if preferred is not None:
                plain.setdefault(name, {})[parameter_name] = preferred
    for name, bound_path_parameters in path_parameters.items():
        for parameter_name in bound_path_parameters:
            flow[f"{name}.__patharg__{parameter_name}"] = frozenset({"1"})
    for name, bound_path_attributes in path_attributes.items():
        for parameter_name, fields in bound_path_attributes.items():
            flow[f"{name}.__pathattr__{parameter_name}"] = frozenset(fields)
    return flow, plain


def _call_argument_bindings(
    tree: ast.Module,
    initial: FlowAliases,
    final: FlowAliases,
) -> tuple[FlowAliases, dict[str, dict[str, str]]]:
    """Propagate direct call-site arguments into function parameters."""
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    values: dict[str, dict[str, set[str]]] = {}
    path_parameters: dict[str, set[str]] = {}
    path_attributes: dict[str, dict[str, set[str]]] = {}
    identity_aliases = _identity_static_aliases(tree)
    class_static = _import_aliases_from_alias_sets(initial)
    class_static.update(identity_aliases)
    class_fields = _class_instance_path_fields(tree, class_static)
    class_names = set(class_fields)
    for call, call_aliases, call_paths in _iter_calls_with_aliases(
        tree.body, dict(initial), set(), final
    ):
        func = call.func
        if not isinstance(func, ast.Name) or func.id not in functions:
            continue
        definition = functions[func.id]
        static = _import_aliases_from_alias_sets(call_aliases)
        static.update(identity_aliases)
        positional = [*definition.args.posonlyargs, *definition.args.args]
        for argument, definition_parameter in zip(call.args, positional):
            if isinstance(argument, ast.Starred):
                break
            _record_argument_binding(
                func.id,
                definition_parameter.arg,
                argument,
                call_aliases,
                static,
                call_paths,
                class_fields,
                class_names,
                values,
                path_parameters,
                path_attributes,
            )
        for keyword in call.keywords:
            if keyword.arg is None:
                continue
            _record_argument_binding(
                func.id,
                keyword.arg,
                keyword.value,
                call_aliases,
                static,
                call_paths,
                class_fields,
                class_names,
                values,
                path_parameters,
                path_attributes,
            )
    return _call_binding_flow(values, path_parameters, path_attributes)


def _function_call_bindings(
    function_name: str, global_aliases: FlowAliases | None
) -> tuple[dict[str, frozenset[str]], set[str], dict[str, set[str]]]:
    bindings: dict[str, frozenset[str]] = {}
    path_parameters: set[str] = set()
    path_attributes: dict[str, set[str]] = {}
    if not global_aliases:
        return bindings, path_parameters, path_attributes
    value_prefix = f"{function_name}.__arg__"
    path_prefix = f"{function_name}.__patharg__"
    attribute_prefix = f"{function_name}.__pathattr__"
    for key, value in global_aliases.items():
        if key.startswith(value_prefix):
            bindings[key[len(value_prefix) :]] = value
        elif key.startswith(path_prefix):
            path_parameters.add(key[len(path_prefix) :])
        elif key.startswith(attribute_prefix):
            path_attributes[key[len(attribute_prefix) :]] = set(value)
    return bindings, path_parameters, path_attributes


def _import_aliases_from_alias_sets(aliases: FlowAliases) -> dict[str, str]:
    return {
        name: next(iter(values)) for name, values in aliases.items() if len(values) == 1
    }


def _merge_flow_aliases(static: dict[str, str], aliases: FlowAliases) -> dict[str, str]:
    result = dict(static)
    for name, values in aliases.items():
        if len(values) == 1:
            result[name] = next(iter(values))
        elif not values:
            result.pop(name, None)
    return result


def _iter_class_calls(
    stmt: ast.ClassDef,
    aliases: FlowAliases,
    path_names: set[str],
    global_aliases: FlowAliases | None,
) -> Iterator[CallContext]:
    for decorator in stmt.decorator_list:
        yield from _calls_in_expression(decorator, aliases, path_names)
    for base in stmt.bases:
        yield from _calls_in_expression(base, aliases, path_names)
    for keyword in stmt.keywords:
        yield from _calls_in_expression(keyword.value, aliases, path_names)
    _clear_identity_binding(stmt.name, aliases)
    aliases[stmt.name] = frozenset()
    path_names.discard(stmt.name)
    static = _import_aliases_from_alias_sets(aliases)
    class_paths = set(path_names) | _class_self_path_attributes(stmt, static)
    yield from _iter_calls_with_aliases(
        stmt.body, dict(aliases), class_paths, global_aliases
    )
    methods = [
        statement
        for statement in stmt.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    returning: set[str] = set()
    changed = True
    while changed:
        changed = False
        for statement in methods:
            method_paths = _scope_path_names(statement.body, static, set(class_paths))
            returns_path = (
                statement.returns is not None
                and _annotation_contains_path(statement.returns, static)
            ) or _function_returns_path(statement.body, static, method_paths)
            if returns_path and statement.name not in returning:
                returning.add(statement.name)
                class_paths.add(statement.name)
                changed = True
    for method in returning:
        path_names.add(f"{stmt.name}.{method}")


def _iter_assign_calls(
    stmt: ast.Assign,
    aliases: FlowAliases,
    path_names: set[str],
    _global: FlowAliases | None,
) -> Iterator[CallContext]:
    yield from _calls_in_expression(stmt.value, aliases, path_names)
    static = _import_aliases_from_alias_sets(aliases)
    _apply_named_expressions(stmt.value, aliases, static, path_names)
    _assign_target_aliases(stmt.targets, stmt.value, aliases, static, path_names)
    _apply_assign_path(stmt, static, path_names)


def _iter_ann_assign_calls(
    stmt: ast.AnnAssign,
    aliases: FlowAliases,
    path_names: set[str],
    _global: FlowAliases | None,
) -> Iterator[CallContext]:
    yield from _calls_in_expression(stmt.annotation, aliases, path_names)
    static = _import_aliases_from_alias_sets(aliases)
    if stmt.value is not None:
        yield from _calls_in_expression(stmt.value, aliases, path_names)
        _apply_named_expressions(stmt.value, aliases, static, path_names)
        _assign_target_aliases([stmt.target], stmt.value, aliases, static, path_names)
    else:
        _remove_target_aliases([stmt.target], aliases)
    _apply_ann_assign_path(stmt, static, path_names)


def _iter_aug_assign_calls(
    stmt: ast.AugAssign,
    aliases: FlowAliases,
    path_names: set[str],
    _global: FlowAliases | None,
) -> Iterator[CallContext]:
    yield from _calls_in_expression(stmt.value, aliases, path_names)
    _apply_named_expressions(
        stmt.value, aliases, _import_aliases_from_alias_sets(aliases), path_names
    )
    _remove_target_aliases([stmt.target], aliases)
    if not (
        isinstance(stmt.op, ast.Div)
        and isinstance(stmt.target, ast.Name)
        and stmt.target.id in path_names
    ):
        _remove_path_names([stmt.target], path_names)


def _iter_if_calls(
    stmt: ast.If,
    aliases: FlowAliases,
    path_names: set[str],
    global_aliases: FlowAliases | None,
) -> Iterator[CallContext]:
    yield from _calls_in_expression(stmt.test, aliases, path_names)
    static = _import_aliases_from_alias_sets(aliases)
    _apply_named_expressions(stmt.test, aliases, static, path_names)
    _apply_expression_bindings(stmt.test, static, path_names)
    body_paths = set(path_names)
    body = dict(aliases)
    yield from _iter_calls_with_aliases(stmt.body, body, body_paths, global_aliases)
    orelse_paths = set(path_names)
    orelse = dict(aliases)
    yield from _iter_calls_with_aliases(
        stmt.orelse, orelse, orelse_paths, global_aliases
    )
    _merge_alias_sets(aliases, body, orelse)
    _replace_path_names(path_names, body_paths | orelse_paths)


def _is_items_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "items"
        and not node.args
        and not node.keywords
    )


def _assign_for_target(
    stmt: ast.For | ast.AsyncFor, values: frozenset[str], aliases: FlowAliases
) -> None:
    target = stmt.target
    if (
        _is_items_call(stmt.iter)
        and isinstance(target, (ast.Tuple, ast.List))
        and len(target.elts) == 2
    ):
        _assign_names(target.elts[0], frozenset(), aliases)
        _assign_names(target.elts[1], values, aliases)
        return
    _assign_names(target, values, aliases)


def _iter_for_calls(
    stmt: ast.For | ast.AsyncFor,
    aliases: FlowAliases,
    path_names: set[str],
    global_aliases: FlowAliases | None,
) -> Iterator[CallContext]:
    yield from _calls_in_expression(stmt.iter, aliases, path_names)
    static = _import_aliases_from_alias_sets(aliases)
    _apply_named_expressions(stmt.iter, aliases, static, path_names)
    _apply_expression_bindings(stmt.iter, static, path_names)
    inner_paths = set(path_names)
    _apply_for_path(stmt, static, inner_paths)
    inner = dict(aliases)
    iterable_values = _iterable_callable_names(stmt.iter, aliases, static, path_names)
    _assign_for_target(stmt, iterable_values, inner)
    yield from _iter_calls_with_aliases(stmt.body, inner, inner_paths, global_aliases)
    orelse_paths = set(path_names)
    orelse = dict(aliases)
    yield from _iter_calls_with_aliases(
        stmt.orelse, orelse, orelse_paths, global_aliases
    )
    shadowed_range = (
        isinstance(stmt.iter, ast.Call)
        and isinstance(stmt.iter.func, ast.Name)
        and stmt.iter.func.id == "range"
        and "range" in aliases
    )
    if (
        _iterable_guaranteed_nonempty(stmt.iter)
        and not shadowed_range
        and not stmt.orelse
    ):
        aliases.clear()
        aliases.update(inner)
        _replace_path_names(path_names, inner_paths)
        return
    _merge_alias_sets(aliases, inner, orelse)
    _replace_path_names(path_names, inner_paths | orelse_paths)


def _iter_while_calls(
    stmt: ast.While,
    aliases: FlowAliases,
    path_names: set[str],
    global_aliases: FlowAliases | None,
) -> Iterator[CallContext]:
    yield from _calls_in_expression(stmt.test, aliases, path_names)
    static = _import_aliases_from_alias_sets(aliases)
    _apply_named_expressions(stmt.test, aliases, static, path_names)
    _apply_expression_bindings(stmt.test, static, path_names)
    body_paths = set(path_names)
    body = dict(aliases)
    yield from _iter_calls_with_aliases(stmt.body, body, body_paths, global_aliases)
    orelse_paths = set(path_names)
    orelse = dict(aliases)
    yield from _iter_calls_with_aliases(
        stmt.orelse, orelse, orelse_paths, global_aliases
    )
    _merge_alias_sets(aliases, body, orelse)
    _replace_path_names(path_names, body_paths | orelse_paths)


def _iter_with_calls(
    stmt: ast.With | ast.AsyncWith,
    aliases: FlowAliases,
    path_names: set[str],
    global_aliases: FlowAliases | None,
) -> Iterator[CallContext]:
    for item in stmt.items:
        yield from _calls_in_expression(item.context_expr, aliases, path_names)
        static = _import_aliases_from_alias_sets(aliases)
        _apply_named_expressions(item.context_expr, aliases, static, path_names)
        _apply_expression_bindings(item.context_expr, static, path_names)
    inner = dict(aliases)
    inner_paths = set(path_names)
    static = _import_aliases_from_alias_sets(aliases)
    for item in stmt.items:
        if item.optional_vars is None:
            continue
        _remove_target_aliases([item.optional_vars], inner)
        _apply_binding_path(item.optional_vars, item.context_expr, static, inner_paths)
    yield from _iter_calls_with_aliases(stmt.body, inner, inner_paths, global_aliases)
    _merge_alias_sets(aliases, inner)
    _replace_path_names(path_names, inner_paths)


def _iter_try_calls(
    stmt: ast.Try,
    aliases: FlowAliases,
    path_names: set[str],
    global_aliases: FlowAliases | None,
) -> Iterator[CallContext]:
    branches: list[FlowAliases] = []
    path_branches: list[set[str]] = []
    body = dict(aliases)
    body_paths = set(path_names)
    branches.append(body)
    path_branches.append(body_paths)
    yield from _iter_calls_with_aliases(stmt.body, body, body_paths, global_aliases)
    for handler in stmt.handlers:
        if handler.type is not None:
            yield from _calls_in_expression(handler.type, aliases, path_names)
        branch = dict(aliases)
        handler_paths = set(path_names)
        if handler.name is not None:
            _clear_identity_binding(handler.name, branch)
            branch[handler.name] = frozenset()
            handler_paths.discard(handler.name)
        branches.append(branch)
        path_branches.append(handler_paths)
        yield from _iter_calls_with_aliases(
            handler.body, branch, handler_paths, global_aliases
        )
    orelse = dict(body)
    orelse_paths = set(body_paths)
    branches.append(orelse)
    path_branches.append(orelse_paths)
    yield from _iter_calls_with_aliases(
        stmt.orelse, orelse, orelse_paths, global_aliases
    )
    _merge_alias_sets(aliases, *branches)
    finalbody = dict(aliases)
    final_paths = set().union(*path_branches)
    yield from _iter_calls_with_aliases(
        stmt.finalbody, finalbody, final_paths, global_aliases
    )
    _merge_alias_sets(aliases, finalbody)
    _replace_path_names(path_names, final_paths)
    for handler in stmt.handlers:
        if handler.name is not None:
            aliases[handler.name] = frozenset()


def _iter_match_calls(
    stmt: ast.Match,
    aliases: FlowAliases,
    path_names: set[str],
    global_aliases: FlowAliases | None,
) -> Iterator[CallContext]:
    yield from _calls_in_expression(stmt.subject, aliases, path_names)
    static = _import_aliases_from_alias_sets(aliases)
    _apply_named_expressions(stmt.subject, aliases, static, path_names)
    _apply_expression_bindings(stmt.subject, static, path_names)
    branches: list[FlowAliases] = []
    path_branches: list[set[str]] = []
    for case in stmt.cases:
        if case.guard is not None:
            yield from _calls_in_expression(case.guard, aliases, path_names)
            static = _import_aliases_from_alias_sets(aliases)
            _apply_named_expressions(case.guard, aliases, static, path_names)
            _apply_expression_bindings(case.guard, static, path_names)
        branch = dict(aliases)
        branch_paths = set(path_names)
        branches.append(branch)
        path_branches.append(branch_paths)
        yield from _iter_calls_with_aliases(
            case.body, branch, branch_paths, global_aliases
        )
    _merge_alias_sets(aliases, *branches)
    _replace_path_names(path_names, set().union(*path_branches))


ScopeHandler = Callable[
    [Any, FlowAliases, set[str], FlowAliases | None], Iterator[CallContext]
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
    ast.TryStar: _iter_try_calls,
    ast.Match: _iter_match_calls,
}


def _clear_import_paths(stmt: ast.stmt, names: set[str]) -> None:
    if isinstance(stmt, ast.Import):
        for alias in stmt.names:
            names.discard(alias.asname or alias.name.split(".")[0])
    elif isinstance(stmt, ast.ImportFrom):
        for alias in stmt.names:
            names.discard(alias.asname or alias.name)


def _iter_calls_with_aliases(
    statements: list[ast.stmt],
    aliases: FlowAliases,
    path_names: set[str] | None = None,
    global_aliases: FlowAliases | None = None,
) -> Iterator[CallContext]:
    """Yield calls with aliases and flow-sensitive ``Path`` names.

    ``path_names`` is mutated in place, so callers observe the state after
    the last statement.  Branches merge their end states with a may-analysis:
    a name stays tainted when any reaching path binds a ``Path``.
    """
    current_paths = set() if path_names is None else path_names
    for stmt in statements:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            _apply_scope_import(stmt, aliases)
            _clear_import_paths(stmt, current_paths)
            continue
        if isinstance(stmt, ast.Delete):
            yield from _calls_in_expression(stmt, aliases, current_paths)
            for target in stmt.targets:
                _remove_path_names([target], current_paths)
            continue
        handler = _SCOPE_HANDLERS.get(type(stmt))
        if handler is None:
            yield from _calls_in_expression(stmt, aliases, current_paths)
            _apply_expression_bindings(
                stmt, _import_aliases_from_alias_sets(aliases), current_paths
            )
        else:
            yield from handler(stmt, aliases, current_paths, global_aliases)


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
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 0:
                continue
            if node.module is None or not any(
                alias.name == "*" for alias in node.names
            ):
                continue
            root = node.module.split(".")[0]
            if root not in WILDCARD_IMPORT_ALLOWLIST:
                violations.append(
                    Violation(
                        "wildcard-import",
                        path,
                        node.lineno,
                        f"the contract must not use 'from {node.module} import *': "
                        "wildcard imports bypass the name-based boundary checks",
                    )
                )
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
    if isinstance(node, ast.Dict):
        for value in node.values:
            if value is not None:
                yield from _type_expression_strings(value, aliases)
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
    """Separate qualified type names from ordinary dotted string data.

    Direct string assignments are ambiguous: ``"numpy.ndarray"`` can be a
    forward reference, while ``"numpy.float32"`` is commonly stored as data.
    The heuristic therefore requires an engine name, a spaced union, or an
    upper-case/underscore final dotted component. Lower-case qualified names
    such as ``"numpy.ndarray"`` remain a documented limitation of the direct
    assignment guard; explicit ``TypeAlias`` annotations are checked fully.
    """
    if ENGINE_MODULE in value:
        return True
    if " | " in value:
        return True
    if "." not in value:
        return False
    if "|" in value or "[" in value:
        return True
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


def _initial_flow_aliases(tree: ast.Module) -> FlowAliases:
    return {name: frozenset({target}) for name, target in _import_aliases(tree).items()}


def _with_function_returns(tree: ast.Module, aliases: FlowAliases) -> FlowAliases:
    result = dict(aliases)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        values: set[str] = set()
        for child in _scope_nodes(node.body):
            if isinstance(child, ast.Return) and child.value is not None:
                values.update(_value_callable_names(child.value, result, None, None))
        if values:
            result[f"{node.name}()"] = frozenset(values)
    return result


def _module_final_aliases(tree: ast.Module, initial: FlowAliases) -> FlowAliases:
    final = dict(initial)
    for _ in _iter_calls_with_aliases(tree.body, final):
        pass
    return final


def _module_path_names(tree: ast.Module, aliases: dict[str, str]) -> set[str]:
    return _scope_path_names(tree.body, aliases, set())


def _call_type_strings(path: Path, tree: ast.Module) -> list[Violation]:
    violations: list[Violation] = []
    static = _import_aliases(tree)
    initial = _initial_flow_aliases(tree)
    final = _module_final_aliases(tree, initial)
    paths = _module_path_names(tree, static)
    for node, aliases, _ in _iter_calls_with_aliases(
        tree.body, dict(initial), paths, final
    ):
        possible = _resolve_possible_names(node.func, aliases)
        type_constructor = {
            "ForwardRef",
            "NamedTuple",
            "NewType",
            "TypeVar",
            "TypedDict",
            "cast",
            "create_model",
            "make_dataclass",
        }
        if not any(name.rsplit(".", 1)[-1] in type_constructor for name in possible):
            continue
        arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
        for argument in arguments:
            for constant in _type_expression_strings(argument, static):
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


def _dynamic_lambda_violations(
    tree: ast.Module,
    imports: dict[str, str],
    path: Path,
    reachable: set[int],
) -> list[Violation]:
    violations: list[Violation] = []
    for lambda_node, lambda_names in _lambda_reference_names(tree, imports):
        if id(lambda_node) not in reachable:
            continue
        dynamic = frozenset(
            name for name in lambda_names if _is_dynamic_import_name(name)
        )
        if dynamic:
            violations.append(
                Violation(
                    "dynamic-import",
                    path,
                    lambda_node.lineno,
                    "dynamic execution through "
                    f"{sorted(dynamic)!r} is forbidden in the "
                    "contract; use a static import",
                )
            )
    return violations


def scan_contract_dynamic_imports(contract_src: Path = CONTRACT_SRC) -> list[Violation]:
    """Reject dynamic import calls in the contract."""
    violations: list[Violation] = []
    trees = {path: _parse(path) for path in python_files(contract_src)}
    (
        imported_attributes_registry,
        imported_methods_registry,
        imported_init_registry,
        imported_returns_registry,
        imported_identity_registry,
    ) = _runtime_fact_registry(contract_src, trees)
    for path, tree in trees.items():
        file_imports = _file_import_aliases(tree, path, contract_src)
        base_static = dict(file_imports)
        base_static.update(_identity_static_aliases(tree))
        static, _ = _module_value_aliases(tree, base_static)
        imported_runtime = _imported_runtime_facts(
            static,
            imported_attributes_registry,
            imported_methods_registry,
            imported_init_registry,
            imported_returns_registry,
            imported_identity_registry,
        )
        for name in imported_runtime.identity_functions:
            marker = _identity_alias_key(name)
            static[name] = marker
            static[marker] = name
        initial = _with_function_returns(tree, _initial_flow_aliases(tree))
        initial.update(_identity_flow_aliases(tree))
        for name in imported_runtime.identity_functions:
            marker = _identity_alias_key(name)
            initial[name] = frozenset({marker})
            initial[marker] = frozenset({name})
        final = _module_final_aliases(tree, initial)
        bindings, plain_bindings = _call_argument_bindings(tree, initial, final)
        initial.update(bindings)
        final.update(bindings)
        paths = _module_path_names(tree, static)
        states = _node_runtime_aliases(tree, file_imports, plain_bindings)
        reachable = set(states)

        violations.extend(
            _namespace_reference_violations(
                tree,
                states,
                _shadowed_names_by_node(tree),
                set(),
                static,
                _dynamic_reference_report(path),
                imported_attributes=imported_runtime.attributes,
                imported_methods=imported_runtime.methods,
                imported_init_methods=imported_runtime.init_methods,
                imported_returns=imported_runtime.returns,
                extra_identity_functions=imported_runtime.identity_functions,
            )
        )
        violations.extend(
            _dynamic_lambda_violations(tree, file_imports, path, reachable)
        )
        for node, aliases, call_paths in _iter_calls_with_aliases(
            tree.body, dict(initial), paths, final
        ):
            if id(node) not in reachable:
                continue
            possible = (
                _resolve_possible_names(node.func, aliases)
                | _literal_indirect_names(node, aliases, call_paths, static)
                | _literal_indirect_names(node.func, aliases, call_paths, static)
            )
            possible = frozenset(_normalise_builtins_name(name) for name in possible)
            dynamic = frozenset(
                name for name in possible if _is_dynamic_import_name(name)
            )
            if dynamic:
                message = (
                    f"dynamic execution through {sorted(dynamic)!r} is "
                    "forbidden in the contract; use a static import"
                )
                violations.append(
                    Violation("dynamic-import", path, node.lineno, message)
                )
                continue
            value_dynamic: frozenset[str] = frozenset()
            value_line = node.lineno
            for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
                for reference, reference_names in _value_references(argument, aliases):
                    value_dynamic = frozenset(
                        name
                        for name in reference_names
                        if _is_dynamic_import_name(name)
                    )
                    if value_dynamic:
                        value_line = reference.lineno
                        break
                if value_dynamic:
                    break
            if value_dynamic:
                violations.append(
                    Violation(
                        "dynamic-import",
                        path,
                        value_line,
                        "dynamic execution through "
                        f"{sorted(value_dynamic)!r} is forbidden in the "
                        "contract; use a static import",
                    )
                )
            elif (
                possible & {"getattr", "builtins.getattr"}
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
    return _dedupe_violations(violations)


def _dedupe_violations(violations: list[Violation]) -> list[Violation]:
    seen: set[tuple[str, str, int, str]] = set()
    result: list[Violation] = []
    for violation in violations:
        key = (
            violation.rule,
            str(violation.path),
            violation.line,
            violation.message,
        )
        if key not in seen:
            seen.add(key)
            result.append(violation)
    return result


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


def _attribute_is_path(
    node: ast.Attribute, aliases: dict[str, str], path_names: set[str]
) -> bool:
    if node.attr == "Path":
        name = _resolve_callable_name(node, aliases)
        if name in PATH_CLASS_NAMES:
            return True
    dotted = _dotted_name(node)
    if dotted is not None and dotted in path_names:
        return True
    if isinstance(node.value, ast.Call):
        class_name = _resolve_callable_name(node.value.func, aliases)
        if class_name is not None and f"{class_name}.{node.attr}" in path_names:
            return True
    if (
        isinstance(node.value, ast.Name)
        and node.value.id in {"self", "cls"}
        and f"{node.value.id}.{node.attr}" in path_names
    ):
        return True
    return node.attr in {"parent", "parents"} and _is_path_expression(
        node.value, aliases, path_names
    )


def _call_is_path(
    node: ast.Call, aliases: dict[str, str], path_names: set[str]
) -> bool:
    name = _resolve_callable_name(node.func, aliases)
    if name in PATH_CLASS_NAMES or name in path_names:
        return True
    if len(node.args) == 1 and not node.keywords:
        dotted = _dotted_name(node.func)
        target = aliases.get(dotted) if dotted is not None else None
        if target is not None and target.startswith(IDENTITY_ALIAS_PREFIX):
            return _is_path_expression(node.args[0], aliases, path_names)
    if not isinstance(node.func, ast.Attribute):
        return False
    if (
        isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"self", "cls"}
        and node.func.attr in path_names
    ):
        return True
    if isinstance(node.func.value, ast.Call):
        class_name = _resolve_callable_name(node.func.value.func, aliases)
        if class_name is not None and f"{class_name}.{node.func.attr}" in path_names:
            return True
    return node.func.attr in PATH_RETURNING_METHODS and _is_path_expression(
        node.func.value, aliases, path_names
    )


def _subscript_is_path(
    node: ast.AST, aliases: dict[str, str], path_names: set[str]
) -> bool | None:
    if not isinstance(node, ast.Subscript):
        return None
    if isinstance(node.value, ast.Attribute) and node.value.attr in {
        "parent",
        "parents",
    }:
        return _is_path_expression(node.value.value, aliases, path_names)
    if not isinstance(node.slice, ast.Constant):
        return None
    if isinstance(node.value, ast.Name):
        return f"{node.value.id}[{node.slice.value!r}]" in path_names
    if isinstance(node.value, (ast.List, ast.Tuple)) and isinstance(
        node.slice.value, int
    ):
        elements = node.value.elts
        if -len(elements) <= node.slice.value < len(elements):
            return _is_path_expression(elements[node.slice.value], aliases, path_names)
        return False
    if isinstance(node.value, ast.Dict):
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and key.value == node.slice.value:
                return _is_path_expression(value, aliases, path_names)
        return False
    if isinstance(node.value, ast.Attribute) and node.value.attr in {
        "parent",
        "parents",
    }:
        return _is_path_expression(node.value.value, aliases, path_names)
    return None


def _is_path_expression(
    node: ast.AST, aliases: dict[str, str], path_names: set[str]
) -> bool:
    """Recognise Path factories and Path-preserving method chains."""
    if isinstance(node, ast.NamedExpr):
        return _is_path_expression(node.value, aliases, path_names)
    if isinstance(node, ast.IfExp):
        return _is_path_expression(
            node.body, aliases, path_names
        ) or _is_path_expression(node.orelse, aliases, path_names)
    if isinstance(node, ast.BoolOp):
        return any(
            _is_path_expression(value, aliases, path_names) for value in node.values
        )
    if isinstance(node, ast.Name):
        return (
            node.id in path_names or aliases.get(node.id, node.id) in PATH_CLASS_NAMES
        )
    subscript = _subscript_is_path(node, aliases, path_names)
    if subscript is not None:
        return subscript
    if isinstance(node, ast.Attribute):
        return _attribute_is_path(node, aliases, path_names)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _is_path_expression(node.left, aliases, path_names)
    if isinstance(node, ast.Call):
        return _call_is_path(node, aliases, path_names)
    return False


def _add_path_names(targets: Sequence[ast.AST], names: set[str]) -> bool:
    changed = False
    for target in targets:
        if isinstance(target, ast.Name) and target.id not in names:
            names.add(target.id)
            changed = True
        elif isinstance(target, ast.Attribute):
            dotted = _dotted_name(target)
            if dotted is not None and dotted not in names:
                names.add(dotted)
                changed = True
        elif isinstance(target, (ast.Tuple, ast.List)):
            changed |= _add_path_names(target.elts, names)
        elif isinstance(target, ast.Starred):
            changed |= _add_path_names([target.value], names)
    return changed


Binding = tuple[int, str]


def _definition_binding_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.ExceptHandler):
        return node.name
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    return None


_COMPOUND_STATEMENTS = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.TryStar,
    ast.With,
    ast.AsyncWith,
    ast.Match,
)
_COMPOUND_BODY_FIELDS = frozenset({"body", "orelse", "finalbody", "handlers", "cases"})


def _binding_nodes(statements: Sequence[ast.stmt], nested: bool) -> Iterator[ast.AST]:
    """Yield binding nodes; optionally skip conditional compound bodies.

    Inside a function every assignment is local regardless of the branch, so
    function scopes walk nested bodies.  At module and class scope a binding
    inside a branch is only a maybe-binding and must not hide a later access.
    """
    stack: list[ast.AST] = list(reversed(list(statements)))
    while stack:
        node = stack.pop()
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            yield node
            continue
        if not nested and isinstance(node, _COMPOUND_STATEMENTS):
            for field, value in ast.iter_fields(node):
                if field in _COMPOUND_BODY_FIELDS:
                    continue
                if isinstance(value, list):
                    stack.extend(
                        reversed([item for item in value if isinstance(item, ast.AST)])
                    )
                elif isinstance(value, ast.AST):
                    stack.append(value)
            continue
        yield node
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _binding_entries(
    statements: Sequence[ast.stmt], *, nested: bool = True
) -> list[Binding]:
    entries: list[Binding] = []
    for node in _binding_nodes(statements, nested):
        targets: Sequence[ast.AST] = ()
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
            targets = [node.target]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            targets = [
                item.optional_vars
                for item in node.items
                if item.optional_vars is not None
            ]
        elif isinstance(node, ast.NamedExpr):
            targets = [node.target]
        elif (definition_name := _definition_binding_name(node)) is not None:
            entries.append((getattr(node, "lineno", 0), definition_name))
        for target in targets:
            entries.extend(
                (getattr(node, "lineno", 0), name.id)
                for name in _iter_target_names(target)
            )
    return entries


def _function_bindings(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[Binding]:
    arguments = [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        node.args.vararg,
        node.args.kwarg,
    ]
    entries = [(0, argument.arg) for argument in arguments if argument is not None]
    return entries + _binding_entries(node.body)


def _visit_shadow(
    node: ast.AST, bindings: Sequence[Binding], result: dict[int, set[str]]
) -> None:
    result[id(node)] = {
        name for line, name in bindings if line < getattr(node, "lineno", 0)
    }
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for decorator in node.decorator_list:
            _visit_shadow(decorator, bindings, result)
        for default in node.args.defaults:
            _visit_shadow(default, bindings, result)
        for kw_default in node.args.kw_defaults:
            if kw_default is not None:
                _visit_shadow(kw_default, bindings, result)
        local = [*bindings, *_function_bindings(node)]
        for statement in node.body:
            _visit_shadow(statement, local, result)
        return
    if isinstance(node, ast.ClassDef):
        for decorator in node.decorator_list:
            _visit_shadow(decorator, bindings, result)
        for base in node.bases:
            _visit_shadow(base, bindings, result)
        for keyword in node.keywords:
            _visit_shadow(keyword.value, bindings, result)
        local = [*bindings, *_binding_entries(node.body, nested=False)]
        for statement in node.body:
            _visit_shadow(statement, local, result)
        return
    for child in ast.iter_child_nodes(node):
        _visit_shadow(child, bindings, result)


def _shadowed_names_by_node(tree: ast.Module) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    _visit_shadow(tree, _binding_entries(tree.body, nested=False), result)
    return result


def _root_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _root_name(node.value)
    return None


def _getattr_bases(
    node: ast.Call,
    aliases: FlowAliases,
    path_names: set[str] | None,
    path_aliases: dict[str, str] | None,
) -> frozenset[str]:
    base_expression = node.args[0]
    bases = _resolve_possible_names(base_expression, aliases)
    if isinstance(base_expression, ast.Call):
        function_names = _resolve_possible_names(base_expression.func, aliases)
        if function_names & {"vars", "builtins.vars"} and base_expression.args:
            bases = _resolve_possible_names(base_expression.args[0], aliases)
    if (
        not bases
        and path_names is not None
        and _is_path_expression(base_expression, path_aliases or {}, path_names)
    ):
        bases = frozenset({"Path"})
    if not bases:
        base_candidates = (
            _resolve_possible_names(base_expression.func, aliases)
            if isinstance(base_expression, ast.Call)
            else _resolve_possible_names(base_expression, aliases)
        )
        if base_candidates & PATH_CLASS_NAMES:
            bases = frozenset({"Path"})
    return bases


def _call_indirect_names(
    node: ast.Call,
    aliases: FlowAliases,
    path_names: set[str] | None = None,
    path_aliases: dict[str, str] | None = None,
) -> frozenset[str]:
    names = _resolve_possible_names(node.func, aliases)
    if names & {"vars", "builtins.vars"} and node.args:
        bases = _resolve_possible_names(node.args[0], aliases)
        if bases:
            return frozenset(f"{base}.__dict__" for base in bases)
    if names & {"getattr", "builtins.getattr"} and len(node.args) >= 2:
        bases = _getattr_bases(node, aliases, path_names, path_aliases)
        attribute = node.args[1]
        if (
            bases
            and isinstance(attribute, ast.Constant)
            and isinstance(attribute.value, str)
        ):
            return frozenset(
                _normalise_builtins_name(f"{base}.{attribute.value}") for base in bases
            )
        path_base = path_names is not None and _is_path_expression(
            node.args[0], path_aliases or {}, path_names
        )
        if bases and (bases & RUNTIME_MODULE_ROOTS or path_base):
            return frozenset(f"{base}.__getattr__" for base in bases)
        return frozenset()
    if (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "__getattribute__"
        and len(node.args) >= 1
    ):
        bases = _resolve_possible_names(node.func.value, aliases)
        attribute = node.args[0]
        if (
            bases
            and isinstance(attribute, ast.Constant)
            and isinstance(attribute.value, str)
        ):
            return frozenset(f"{base}.{attribute.value}" for base in bases)
    if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
        return _module_dict_get_names(node, aliases)
    return frozenset()


def _module_dict_get_names(node: ast.Call, aliases: FlowAliases) -> frozenset[str]:
    function = node.func
    if not isinstance(function, ast.Attribute):
        return frozenset()
    bases = _module_dict_bases(function.value, aliases)
    attribute = node.args[0]
    if (
        bases
        and isinstance(attribute, ast.Constant)
        and isinstance(attribute.value, str)
    ):
        return frozenset(f"{base}.{attribute.value}" for base in bases)
    return frozenset()


def _module_dict_bases(node: ast.AST, aliases: FlowAliases) -> frozenset[str]:
    bases: set[str] = set()
    if isinstance(node, ast.Attribute) and node.attr == "__dict__":
        bases.update(_resolve_possible_names(node.value, aliases))
    elif isinstance(node, ast.Name):
        for alias in aliases.get(node.id, frozenset({node.id})):
            base = _dictionary_module_base(alias)
            if base is not None:
                bases.add(base)
    elif isinstance(node, ast.Call):
        function_names = _resolve_possible_names(node.func, aliases)
        if function_names & {"vars", "builtins.vars"} and node.args:
            bases.update(_resolve_possible_names(node.args[0], aliases))
        elif function_names & {"getattr", "builtins.getattr"} and len(node.args) >= 2:
            attribute = node.args[1]
            if isinstance(attribute, ast.Constant) and attribute.value == "__dict__":
                bases.update(_resolve_possible_names(node.args[0], aliases))
    return frozenset(bases)


def _dictionary_module_base(value: str) -> str | None:
    if value in {"__builtins__", "builtins", "builtins.__dict__"}:
        return "builtins"
    if value.endswith(".__dict__"):
        return value[: -len(".__dict__")]
    return None


def _subscript_indirect_names(
    node: ast.Subscript, aliases: FlowAliases
) -> frozenset[str]:
    if (
        isinstance(node.value, ast.Attribute)
        and node.value.attr == "modules"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        bases = _resolve_possible_names(node.value.value, aliases)
        if "sys" in bases:
            return frozenset({node.slice.value})
    subscript_bases = _module_dict_bases(node.value, aliases)
    attribute = node.slice
    if subscript_bases and not (
        isinstance(attribute, ast.Constant) and isinstance(attribute.value, str)
    ):
        return frozenset(f"{base}.__getattr__" for base in subscript_bases)
    if (
        subscript_bases
        and isinstance(attribute, ast.Constant)
        and isinstance(attribute.value, str)
    ):
        return frozenset(
            _normalise_builtins_name(f"{base}.{attribute.value}")
            for base in subscript_bases
        )
    return frozenset()


def _literal_indirect_names(
    node: ast.AST,
    aliases: FlowAliases,
    path_names: set[str] | None = None,
    path_aliases: dict[str, str] | None = None,
) -> frozenset[str]:
    if isinstance(node, ast.Call):
        return _call_indirect_names(node, aliases, path_names, path_aliases)
    if isinstance(node, ast.Subscript):
        return _subscript_indirect_names(node, aliases)
    return frozenset()


def _normalise_builtins_name(name: str) -> str:
    if name == "__builtins__":
        return "builtins"
    if name.startswith("__builtins__."):
        name = f"builtins.{name[len('__builtins__.') :]}"
    if name.startswith("builtins."):
        tail = name[len("builtins.") :]
        if tail.split(".", 1)[0] in RUNTIME_MODULE_ROOTS:
            return tail
    return name


def _normalise_path_name(name: str) -> str:
    """Map concrete ``pathlib`` classes to the ``Path`` spelling."""
    for path_class in sorted(PATH_CLASS_NAMES, key=len, reverse=True):
        if name == path_class:
            return "Path"
        if name.startswith(f"{path_class}."):
            return f"Path{name[len(path_class) :]}"
    return name


def _is_path_method_name(name: str) -> bool:
    normalised = _normalise_path_name(name)
    if not normalised.startswith("Path."):
        return False
    method = normalised.split(".", 1)[1]
    return method not in (PATH_PURE_METHODS | PATH_PROPERTIES)


def _is_forbidden_runtime_name(name: str) -> bool:
    name = _normalise_builtins_name(name)
    normalised = _normalise_path_name(name)
    return (
        name in {"open", "builtins.open", "__builtins__.open"}
        or name in RUNTIME_ATTRIBUTES
        or name in FORBIDDEN_INDIRECTION_NAMES
        or name.startswith(RUNTIME_CALL_PREFIXES)
        or name.endswith(".__getattr__")
        or (
            normalised.startswith("Path.")
            and normalised.split(".", 1)[1] not in (PATH_PURE_METHODS | PATH_PROPERTIES)
        )
    )


def _literal_subscript_violations(
    tree: ast.Module,
    states: dict[int, dict[str, str]],
    shadowed: dict[int, set[str]],
    value_alias_names: set[str],
    path: Path,
) -> list[Violation]:
    call_func_ids = {
        id(child.func) for child in ast.walk(tree) if isinstance(child, ast.Call)
    }
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if id(node) not in states:
            continue
        if not isinstance(node, ast.Subscript) or id(node) in call_func_ids:
            continue
        node_aliases: FlowAliases = {
            name: frozenset({target})
            for name, target in states.get(id(node), {}).items()
        }
        flat_aliases = {
            name: next(iter(values)) for name, values in node_aliases.items()
        }
        if _is_shadowed_runtime_name(
            _root_name(node.value),
            shadowed.get(id(node), set()),
            value_alias_names,
            flat_aliases,
        ):
            continue
        literal_names = _literal_indirect_names(node, node_aliases)
        forbidden = {name for name in literal_names if _is_forbidden_runtime_name(name)}
        if forbidden:
            violations.append(_runtime_violation(path, node.lineno, min(forbidden)))
    return violations


def _is_runtime_target(target: str | None) -> bool:
    if target is None:
        return False
    return (
        target in PATH_CLASS_NAMES
        or _is_forbidden_runtime_name(target)
        or target.split(".", 1)[0] in RUNTIME_MODULE_ROOTS
    )


def _is_shadowed_runtime_name(
    name: str | None,
    shadow: set[str],
    value_alias_names: set[str],
    aliases: dict[str, str] | None = None,
) -> bool:
    if name is None or name not in shadow:
        return False
    if aliases is not None and _is_runtime_target(aliases.get(name)):
        return False
    return name not in value_alias_names


def _path_returning_functions(tree: ast.Module, aliases: dict[str, str]) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        paths = _scope_path_names(node.body, aliases, set())
        if (
            node.returns is not None
            and _annotation_contains_path(node.returns, aliases)
        ) or _function_returns_path(node.body, aliases, paths):
            names.add(node.name)
    return names


def _class_contexts(tree: ast.Module) -> dict[int, str | None]:
    contexts: dict[int, str | None] = {}

    def visit(node: ast.AST, current: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, ast.ClassDef) else current
            contexts[id(child)] = name
            visit(child, name)

    visit(tree, None)
    return contexts


def _method_parameter_paths(
    method: ast.FunctionDef | ast.AsyncFunctionDef, aliases: dict[str, str]
) -> set[str]:
    names: set[str] = set()
    positional = [*method.args.posonlyargs, *method.args.args]
    defaults: dict[str, ast.AST] = {}
    default_start = len(positional) - len(method.args.defaults)
    for argument, default in zip(positional[default_start:], method.args.defaults):
        defaults[argument.arg] = default
    for argument, kw_default in zip(method.args.kwonlyargs, method.args.kw_defaults):
        if kw_default is not None:
            defaults[argument.arg] = kw_default
    parameters = [
        *positional,
        *method.args.kwonlyargs,
        method.args.vararg,
        method.args.kwarg,
    ]
    for parameter in parameters:
        if parameter is None:
            continue
        if parameter.annotation is not None and _annotation_contains_path(
            parameter.annotation, aliases
        ):
            names.add(parameter.arg)
            continue
        default_value = defaults.get(parameter.arg)
        if default_value is not None and _is_path_expression(
            default_value, aliases, set()
        ):
            names.add(parameter.arg)
    return names


def _iter_self_attributes(target: ast.AST) -> Iterator[str]:
    if isinstance(target, ast.Attribute):
        if isinstance(target.value, ast.Name) and target.value.id in {"self", "cls"}:
            yield f"{target.value.id}.{target.attr}"
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _iter_self_attributes(element)
    elif isinstance(target, ast.Starred):
        yield from _iter_self_attributes(target.value)


def _method_self_attribute_targets(
    statements: Sequence[ast.stmt],
) -> set[str]:
    targets: set[str] = set()
    for node in _scope_nodes(statements):
        if isinstance(node, ast.Assign):
            statement_targets: Sequence[ast.AST] = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
            statement_targets = [node.target]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            statement_targets = [
                item.optional_vars
                for item in node.items
                if item.optional_vars is not None
            ]
        else:
            continue
        for target in statement_targets:
            targets.update(_iter_self_attributes(target))
    return targets


def _class_returning_path_methods(
    class_node: ast.ClassDef, aliases: dict[str, str]
) -> set[str]:
    methods = [
        statement
        for statement in class_node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    returning: set[str] = set()
    changed = True
    while changed:
        changed = False
        for method in methods:
            start = _method_parameter_paths(method, aliases) | returning
            final = _path_statements(method.body, aliases, set(start))
            returns_path = (
                method.returns is not None
                and _annotation_contains_path(method.returns, aliases)
            ) or _function_returns_path(method.body, aliases, final)
            if returns_path and method.name not in returning:
                returning.add(method.name)
                changed = True
    return returning


def _class_level_path_names(
    class_node: ast.ClassDef, aliases: dict[str, str]
) -> set[str]:
    statements = [
        statement
        for statement in class_node.body
        if not isinstance(
            statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        )
    ]
    names = _path_statements(statements, aliases, set())
    for statement in statements:
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and _annotation_contains_path(statement.annotation, aliases)
        ):
            names.add(statement.target.id)
    return names


def _is_property(method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in method.decorator_list:
        name = _resolve_callable_name(decorator, {})
        if name is not None and name.rsplit(".", 1)[-1] == "property":
            return True
    return False


def _class_self_path_attributes(
    class_node: ast.ClassDef,
    aliases: dict[str, str],
    base_names: set[str] | None = None,
) -> set[str]:
    returning = _class_returning_path_methods(class_node, aliases)
    class_level = {
        f"self.{name}" for name in _class_level_path_names(class_node, aliases)
    }
    baseline = (base_names or set()) | returning | class_level
    attributes: set[str] = set(class_level)
    for statement in class_node.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if _is_property(statement) and statement.name in returning:
            attributes.add(f"self.{statement.name}")
        start = _method_parameter_paths(statement, aliases) | baseline
        final = _path_statements(statement.body, aliases, set(start))
        targets = _method_self_attribute_targets(statement.body)
        attributes |= {attribute for attribute in targets if attribute in final}
    return attributes


def _self_path_attributes_by_class(
    tree: ast.Module,
    aliases: dict[str, str],
    external_names: set[str] | None = None,
) -> dict[str, set[str]]:
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    module_names = _module_path_names(tree, aliases) | _path_returning_functions(
        tree, aliases
    )
    local_names = {node.name for node in classes}
    external = set(external_names or ())
    result = {
        class_node.name: _class_self_path_attributes(class_node, aliases, module_names)
        for class_node in classes
    }
    changed = True
    while changed:
        changed = False
        for class_node in classes:
            for base in class_node.bases:
                base_name = _resolved_base_names(base, aliases, local_names, external)
                if base_name is None or base_name not in result:
                    continue
                inherited = result[base_name] - result[class_node.name]
                if inherited:
                    result[class_node.name] |= inherited
                    changed = True
    return result


def _annotation_instance_classes(
    annotation: ast.AST, aliases: dict[str, str], class_names: set[str]
) -> set[str]:
    """Return classes named by an instance annotation."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        forward_name = annotation.value.rsplit(".", 1)[-1]
        return {forward_name} if forward_name in class_names else set()
    name = _resolve_callable_name(annotation, aliases)
    if name is not None:
        last = name.rsplit(".", 1)[-1]
        if last in class_names:
            return {last}
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _annotation_instance_classes(
            annotation.left, aliases, class_names
        ) | _annotation_instance_classes(annotation.right, aliases, class_names)
    if isinstance(annotation, ast.Subscript):
        base = _resolve_callable_name(annotation.value, aliases)
        base_name = base.rsplit(".", 1)[-1] if base else ""
        elements = (
            annotation.slice.elts
            if isinstance(annotation.slice, ast.Tuple)
            else [annotation.slice]
        )
        if base_name == "Annotated" and elements:
            return _annotation_instance_classes(elements[0], aliases, class_names)
        if base_name in {"Optional", "Union"}:
            result: set[str] = set()
            for element in elements:
                result |= _annotation_instance_classes(element, aliases, class_names)
            return result
    return set()


def _value_instance_classes(
    value: ast.AST, aliases: dict[str, str], class_names: set[str]
) -> set[str]:
    if isinstance(value, ast.Call):
        name = _resolve_callable_name(value.func, aliases)
        if name is not None and name.rsplit(".", 1)[-1] in class_names:
            return {name.rsplit(".", 1)[-1]}
        return set()
    return _annotation_instance_classes(value, aliases, class_names)


def _field_statement_parts(
    node: ast.AST,
) -> tuple[Sequence[ast.AST], ast.AST | None, ast.AST | None] | None:
    if isinstance(node, ast.Assign):
        return node.targets, node.value, None
    if isinstance(node, ast.AnnAssign):
        return [node.target], node.value, node.annotation
    return None


def _field_assignment_owners(
    node: ast.AST,
    aliases: dict[str, str],
    class_names: set[str],
    parameter_classes: dict[str, set[str]] | None = None,
) -> tuple[Sequence[ast.AST], set[str]] | None:
    parts = _field_statement_parts(node)
    if parts is None:
        return None
    targets, value, annotation = parts
    owners: set[str] = set()
    if annotation is not None:
        owners |= _annotation_instance_classes(annotation, aliases, class_names)
    if value is not None:
        owners |= _value_instance_classes(value, aliases, class_names)
        if isinstance(value, ast.Name) and parameter_classes is not None:
            owners |= parameter_classes.get(value.id, set())
    return targets, owners


def _method_parameter_classes(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: dict[str, str],
    class_names: set[str],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for parameter in _function_parameters(method):
        if parameter.annotation is None:
            continue
        result[parameter.arg] = _annotation_instance_classes(
            parameter.annotation, aliases, class_names
        )
    return result


def _function_parameter_types(
    tree: ast.Module, aliases: dict[str, str], class_names: set[str]
) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        parameters: dict[str, str] = {}
        for parameter in _function_parameters(node):
            if parameter.annotation is None:
                continue
            classes = _annotation_instance_classes(
                parameter.annotation, aliases, class_names
            )
            if classes:
                parameters[parameter.arg] = _preferred_name(classes)
        if parameters:
            result[id(node)] = parameters
    return result


def _class_field_owner_map(
    classes: Sequence[ast.ClassDef],
    aliases: dict[str, str],
    class_names: set[str],
) -> dict[str, dict[str, set[str]]]:
    result: dict[str, dict[str, set[str]]] = {name: {} for name in class_names}
    for class_node in classes:
        fields = result[class_node.name]
        for statement in class_node.body:
            assignment = _field_assignment_owners(statement, aliases, class_names)
            if assignment is None:
                continue
            targets, owners = assignment
            for target in targets:
                if isinstance(target, ast.Name):
                    fields.setdefault(target.id, set()).update(owners)
        for method in _class_methods(class_node):
            parameter_classes = _method_parameter_classes(method, aliases, class_names)
            for node in _scope_nodes(method.body):
                assignment = _field_assignment_owners(
                    node, aliases, class_names, parameter_classes
                )
                if assignment is None:
                    continue
                targets, owners = assignment
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in {"self", "cls"}
                    ):
                        fields.setdefault(target.attr, set()).update(owners)
    return result


def _inherit_field_types(
    classes: Sequence[ast.ClassDef],
    resolved: dict[str, dict[str, str]],
    bases: dict[str, list[str]],
) -> None:
    for _ in range(len(classes) + 1):
        changed = False
        for class_node in classes:
            for base in bases.get(class_node.name, []):
                for field_name, owner in resolved.get(base, {}).items():
                    if field_name not in resolved[class_node.name]:
                        resolved[class_node.name][field_name] = owner
                        changed = True
        if not changed:
            break


def _class_field_type_map(
    tree: ast.Module, aliases: dict[str, str], class_names: set[str]
) -> dict[str, dict[str, str]]:
    """Map instance field names to the class that owns their value."""
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    owner_map = _class_field_owner_map(classes, aliases, class_names)
    resolved: dict[str, dict[str, str]] = {
        name: {
            field: _preferred_name(owners) for field, owners in fields.items() if owners
        }
        for name, fields in owner_map.items()
    }
    _inherit_field_types(classes, resolved, _class_bases(classes, aliases, class_names))
    return resolved


def _class_instance_path_fields(
    tree: ast.Module,
    aliases: dict[str, str],
    external_fields: dict[str, set[str]] | None = None,
) -> dict[str, set[str]]:
    """Return dotted ``Path`` field chains for every class instance."""
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    external = external_fields or {}
    class_names = {node.name for node in classes} | set(external)
    direct = {
        name: {
            attribute.split(".", 1)[1]
            for attribute in attributes
            if attribute.startswith("self.")
        }
        for name, attributes in _self_path_attributes_by_class(tree, aliases).items()
    }
    field_types = _class_field_type_map(tree, aliases, class_names)
    result = {name: set(fields) for name, fields in direct.items()}
    for name, fields in external.items():
        result.setdefault(name, set()).update(fields)
    for _ in range(len(class_names) + 1):
        changed = False
        for name in class_names:
            expanded = set(result.get(name, set()))
            for field_name, owner in field_types.get(name, {}).items():
                for nested in result.get(owner, set()):
                    expanded.add(f"{field_name}.{nested}")
            if expanded != result.get(name, set()):
                result[name] = expanded
                changed = True
        if not changed:
            break
    return result


def _module_names_for_path(root: Path, path: Path) -> set[str]:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    relative_name = ".".join(parts)
    if not relative_name:
        return {root.name}
    return {relative_name, f"{root.name}.{relative_name}"}


def _class_field_registry(
    root: Path, trees: dict[Path, ast.Module]
) -> dict[str, set[str]]:
    statics: dict[Path, dict[str, str]] = {}
    modules: dict[Path, set[str]] = {}
    classes: dict[Path, list[str]] = {}
    registry: dict[str, set[str]] = {}
    for path, tree in trees.items():
        static, _ = _module_value_aliases(tree, _file_import_aliases(tree, path, root))
        static.update(_identity_static_aliases(tree))
        statics[path] = static
        modules[path] = _module_names_for_path(root, path)
        classes[path] = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]
        fields = _class_instance_path_fields(tree, static)
        for name in classes[path]:
            values = set(fields.get(name, set()))
            for module in modules[path]:
                registry[f"{module}.{name}"] = values
    for _ in range(len(trees) + 1):
        changed = False
        for path, tree in trees.items():
            imported = _imported_class_fields(statics[path], registry)
            fields = _class_instance_path_fields(tree, statics[path], imported)
            fields = _merge_class_fields(fields, imported)
            for name in classes[path]:
                values = set(fields.get(name, set()))
                for module in modules[path]:
                    key = f"{module}.{name}"
                    if registry.get(key) != values:
                        registry[key] = values
                        changed = True
        if not changed:
            break
    return registry


def _imported_class_fields(
    static: dict[str, str], registry: dict[str, set[str]]
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for alias, target in static.items():
        if alias.startswith((IDENTITY_ALIAS_PREFIX, CLASS_PATH_FIELD_PREFIX)):
            continue
        if target.startswith((IDENTITY_ALIAS_PREFIX, CLASS_PATH_FIELD_PREFIX)):
            continue
        fields = registry.get(target)
        if fields is not None:
            if alias.isidentifier():
                result.setdefault(alias, set()).update(fields)
            result.setdefault(target.rsplit(".", 1)[-1], set()).update(fields)
        prefix = f"{target}."
        for qualified, nested in registry.items():
            if not qualified.startswith(prefix):
                continue
            result.setdefault(qualified.rsplit(".", 1)[-1], set()).update(nested)
            if alias.isidentifier():
                result.setdefault(alias, set()).update(nested)
    return result


@dataclass
class _ImportedRuntimeFacts:
    attributes: dict[str, dict[str, str]] = field(default_factory=dict)
    methods: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = field(
        default_factory=dict
    )
    init_methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = field(
        default_factory=dict
    )
    returns: dict[str, dict[str, str]] = field(default_factory=dict)
    identity_functions: dict[str, str] = field(default_factory=dict)


def _update_identity_registry(
    tree: ast.Module,
    imported: _ImportedRuntimeFacts,
    modules: set[str],
    identities: dict[str, str],
) -> bool:
    changed = False
    for name, parameter in _module_identity_functions(
        tree, imported.identity_functions
    ).items():
        for module in modules:
            key = f"{module}.{name}"
            if identities.get(key) != parameter:
                identities[key] = parameter
                changed = True
    return changed


def _runtime_fact_registry(
    root: Path, trees: dict[Path, ast.Module]
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]],
    dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    dict[str, dict[str, str]],
    dict[str, str],
]:
    attributes: dict[str, dict[str, str]] = {}
    methods: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    init_methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    returns: dict[str, dict[str, str]] = {}
    identities: dict[str, str] = {}
    statics: dict[Path, dict[str, str]] = {}
    modules: dict[Path, set[str]] = {}
    classes: dict[Path, list[str]] = {}
    for path, tree in trees.items():
        static, _ = _module_value_aliases(tree, _file_import_aliases(tree, path, root))
        static.update(_identity_static_aliases(tree))
        statics[path] = static
        modules[path] = _module_names_for_path(root, path)
        classes[path] = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        ]
        for name, parameter in _module_identity_functions(tree).items():
            for module in modules[path]:
                identities[f"{module}.{name}"] = parameter
    for _ in range(len(trees) + 1):
        changed = False
        for path, tree in trees.items():
            static = statics[path]
            imported = _imported_runtime_facts(
                static,
                attributes,
                methods,
                init_methods,
                returns,
                identities,
            )
            class_attributes, class_returns = _class_namespace_facts(
                tree, static, imported.attributes, imported.returns
            )
            class_methods = _class_method_nodes(tree, imported.methods, static)
            class_init_methods = _class_init_methods(tree, imported.methods, static)
            if _update_identity_registry(tree, imported, modules[path], identities):
                changed = True
            for name in classes[path]:
                for module in modules[path]:
                    key = f"{module}.{name}"
                    values = class_attributes.get(name, {})
                    if attributes.get(key) != values:
                        attributes[key] = values
                        changed = True
                    class_method_values = class_methods.get(name, {})
                    if methods.get(key) != class_method_values:
                        methods[key] = class_method_values
                        changed = True
                    class_return_values = class_returns.get(name, {})
                    if returns.get(key) != class_return_values:
                        returns[key] = class_return_values
                        changed = True
                    init = class_init_methods.get(name)
                    if init is not None and init_methods.get(key) is not init:
                        init_methods[key] = init
                        changed = True
        if not changed:
            break
    return attributes, methods, init_methods, returns, identities


def _identity_registry_matches(target: str, qualified: str) -> str | None:
    if qualified == target:
        return ""
    if qualified.startswith(f"{target}."):
        return qualified[len(target) + 1 :]
    suffix = f".{target}."
    if suffix in qualified:
        return qualified.split(suffix, 1)[1]
    return None


def _imported_runtime_facts(
    static: dict[str, str],
    attributes: dict[str, dict[str, str]],
    methods: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]],
    init_methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    returns: dict[str, dict[str, str]],
    identities: dict[str, str],
) -> _ImportedRuntimeFacts:
    result = _ImportedRuntimeFacts()
    for alias, target in static.items():
        if alias.startswith((IDENTITY_ALIAS_PREFIX, CLASS_PATH_FIELD_PREFIX)):
            continue
        if target.startswith((IDENTITY_ALIAS_PREFIX, CLASS_PATH_FIELD_PREFIX)):
            continue
        keys = [target]
        prefix = f"{target}."
        keys.extend(
            qualified for qualified in attributes if qualified.startswith(prefix)
        )
        for key in keys:
            if key not in attributes:
                continue
            names = {key.rsplit(".", 1)[-1]}
            if alias.isidentifier():
                names.add(alias)
            for name in names:
                result.attributes.setdefault(name, dict(attributes[key]))
                result.methods.setdefault(name, dict(methods.get(key, {})))
                result.returns.setdefault(name, dict(returns.get(key, {})))
                if key in init_methods:
                    result.init_methods[name] = init_methods[key]
        identity = identities.get(target)
        if identity is not None:
            name = alias if alias.isidentifier() else target.rsplit(".", 1)[-1]
            result.identity_functions[name] = identity
            result.identity_functions[target.rsplit(".", 1)[-1]] = identity
        for key, qualified_identity in identities.items():
            remainder = _identity_registry_matches(target, key)
            if remainder is None or not remainder:
                continue
            result.identity_functions[f"{target}.{remainder}"] = qualified_identity
            if alias.isidentifier():
                result.identity_functions[f"{alias}.{remainder}"] = qualified_identity
    return result


def _merge_class_fields(
    local: dict[str, set[str]], imported: dict[str, set[str]]
) -> dict[str, set[str]]:
    result = {name: set(fields) for name, fields in local.items()}
    for name, fields in imported.items():
        result.setdefault(name, set()).update(fields)
    return result


def _class_path_field_static_aliases(
    fields: dict[str, set[str]],
) -> dict[str, str]:
    return {
        f"{CLASS_PATH_FIELD_PREFIX}{class_name}:{field}": "1"
        for class_name, class_fields in fields.items()
        for field in class_fields
    }


def _class_path_field_flow_aliases(
    fields: dict[str, set[str]],
) -> FlowAliases:
    return {key: frozenset({"1"}) for key in _class_path_field_static_aliases(fields)}


def _class_path_fields_from_aliases(
    aliases: dict[str, str],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for key in aliases:
        if not key.startswith(CLASS_PATH_FIELD_PREFIX):
            continue
        class_name, separator, field = key[len(CLASS_PATH_FIELD_PREFIX) :].partition(
            ":"
        )
        if separator and class_name and field:
            result.setdefault(class_name, set()).add(field)
    return result


def _preferred_name(names: frozenset[str] | set[str]) -> str:
    return min(names, key=lambda candidate: ("." not in candidate, candidate))


def _resolved_reference(value: ast.AST, aliases: dict[str, str]) -> str | None:
    flow = {name: frozenset({target}) for name, target in aliases.items()}
    if isinstance(value, (ast.Name, ast.Attribute)):
        names = _resolve_possible_names(value, flow)
        if names:
            return _preferred_name(names)
        return _resolve_callable_name(value, aliases)
    if isinstance(value, ast.Call):
        function = _resolve_callable_name(value.func, aliases)
        if function in PATH_CLASS_NAMES:
            return "Path"
    if isinstance(value, (ast.Call, ast.Subscript)):
        names = _literal_indirect_names(value, flow)
        if not names:
            names = _resolve_possible_names(value, flow)
        if names:
            return _preferred_name(names)
    return None


def _export_class_target(
    target: ast.AST, aliases: dict[str, str], class_attributes: dict[str, str]
) -> None:
    for name in _iter_target_names(target):
        resolved = aliases.get(name.id)
        if resolved is None:
            class_attributes.pop(name.id, None)
        else:
            class_attributes[name.id] = resolved


def _apply_class_named_expressions(
    value: ast.AST,
    aliases: dict[str, str],
    class_attributes: dict[str, str],
) -> None:
    for node in ast.walk(value):
        if isinstance(node, ast.NamedExpr):
            _bind_alias_target(node.target, node.value, aliases)
            _export_class_target(node.target, aliases, class_attributes)


def _literal_subscript_key(node: ast.Subscript) -> str | None:
    if isinstance(node.slice, ast.Constant):
        return repr(node.slice.value)
    return None


def _class_container_elements(
    value: ast.AST,
) -> list[tuple[str, ast.AST]]:
    if isinstance(value, (ast.List, ast.Tuple)):
        return [(str(index), element) for index, element in enumerate(value.elts)]
    if isinstance(value, ast.Dict):
        return [
            (repr(key.value), element)
            for key, element in zip(value.keys, value.values)
            if isinstance(key, ast.Constant)
        ]
    return []


def _class_container_candidates(
    class_attributes: dict[str, str], prefix: str, key: str | None
) -> set[str]:
    if key is not None:
        exact = class_attributes.get(f"{prefix}[{key}]")
        if exact is not None:
            return {exact}
        if key.lstrip("-").isdigit():
            numeric: dict[int, str] = {}
            for candidate, element in class_attributes.items():
                if not candidate.startswith(f"{prefix}["):
                    continue
                inner = candidate[len(prefix) + 1 : -1]
                if inner.lstrip("-").isdigit():
                    numeric[int(inner)] = element
            index = int(key)
            if index < 0:
                index += len(numeric)
            value = numeric.get(index)
            return {value} if value is not None else set()
    return {
        value
        for candidate, value in class_attributes.items()
        if candidate.startswith(f"{prefix}[")
    }


def _pick_candidate(candidates: Iterable[str]) -> str | None:
    values = [candidate for candidate in candidates if candidate]
    if not values:
        return None
    forbidden = [
        candidate for candidate in values if _is_forbidden_runtime_name(candidate)
    ]
    if forbidden:
        return _preferred_name(frozenset(forbidden))
    dynamic = [candidate for candidate in values if _is_dynamic_import_name(candidate)]
    if dynamic:
        return _preferred_name(frozenset(dynamic))
    module_candidates = [
        candidate
        for candidate in values
        if candidate.split(".", 1)[0] in RUNTIME_MODULE_ROOTS
    ]
    if module_candidates:
        return _preferred_name(frozenset(module_candidates))
    return _preferred_name(frozenset(values))


def _class_container_elements_map(
    class_attributes: dict[str, str], prefix: str
) -> dict[str, str]:
    marker = f"{prefix}["
    return {
        key[len(marker) : -1]: value
        for key, value in class_attributes.items()
        if key.startswith(marker)
    }


def _container_map_value(elements: dict[str, str], key: str | None) -> str | None:
    if key is not None:
        exact = elements.get(key)
        if exact is not None:
            return exact
        if key.lstrip("-").isdigit():
            numeric = {
                int(candidate): value
                for candidate, value in elements.items()
                if candidate.lstrip("-").isdigit()
            }
            index = int(key)
            if index < 0:
                index += len(numeric)
            return numeric.get(index)
        return None
    return _pick_candidate(elements.values())


def _container_map_candidates(elements: dict[str, str], key: str | None) -> set[str]:
    if key is None:
        return set(elements.values())
    value = _container_map_value(elements, key)
    return {value} if value is not None else set()


def _class_container_value(
    class_attributes: dict[str, str], prefix: str, key: str | None
) -> str | None:
    if key is not None:
        exact = class_attributes.get(f"{prefix}[{key}]")
        if exact is not None:
            return exact
        if key.lstrip("-").isdigit():
            numeric: dict[int, str] = {}
            for candidate, value in class_attributes.items():
                if not candidate.startswith(f"{prefix}["):
                    continue
                inner = candidate[len(prefix) + 1 : -1]
                if inner.lstrip("-").isdigit():
                    numeric[int(inner)] = value
            index = int(key)
            if index < 0:
                index += len(numeric)
            return numeric.get(index)
    candidates = [
        value
        for candidate, value in class_attributes.items()
        if candidate.startswith(f"{prefix}[")
    ]
    return _pick_candidate(candidates)


def _store_class_container(
    prefix: str,
    value: ast.AST,
    aliases: dict[str, str],
    class_attributes: dict[str, str],
) -> None:
    for key in [key for key in class_attributes if key.startswith(f"{prefix}[")]:
        class_attributes.pop(key, None)
    for key, element in _class_container_elements(value):
        resolved = _resolved_reference(element, aliases)
        if resolved is not None:
            class_attributes[f"{prefix}[{key}]"] = resolved


def _record_class_bindings(
    targets: Sequence[ast.AST],
    value: ast.AST | None,
    aliases: dict[str, str],
    class_attributes: dict[str, str],
) -> None:
    if value is not None:
        _apply_class_named_expressions(value, aliases, class_attributes)
    for target in targets:
        _bind_alias_target(target, value, aliases)
        if isinstance(target, ast.Name) and value is not None:
            _store_class_container(target.id, value, aliases, class_attributes)
            if isinstance(value, (ast.List, ast.Tuple)) and value.elts:
                resolved = _resolved_reference(value.elts[0], aliases)
            elif isinstance(value, ast.Dict):
                resolved = None
            else:
                resolved = _resolved_reference(value, aliases)
                if resolved is None:
                    flow = {
                        name: frozenset({target_value})
                        for name, target_value in aliases.items()
                    }
                    names = _value_callable_names(value, flow, None, None)
                    resolved = _preferred_name(names) if names else None
            if resolved is not None:
                aliases[target.id] = resolved
        _export_class_target(target, aliases, class_attributes)


def _bind_class_iterable(
    target: ast.AST,
    iterable: ast.AST,
    aliases: dict[str, str],
    class_attributes: dict[str, str],
) -> None:
    flow = {name: frozenset({value}) for name, value in aliases.items()}
    values = _iterable_callable_names(iterable, flow, None, None)
    for name in _iter_target_names(target):
        if values:
            aliases[name.id] = min(
                values, key=lambda candidate: ("." not in candidate, candidate)
            )
        else:
            aliases.pop(name.id, None)
    _export_class_target(target, aliases, class_attributes)


def _merge_class_maps(
    base: dict[str, str], branches: Sequence[dict[str, str]]
) -> dict[str, str]:
    merged = dict(base)
    for branch in branches:
        for key, value in branch.items():
            if value is not None:
                merged[key] = value
    return merged


def _bind_match_captures(
    pattern: ast.AST, aliases: dict[str, str], class_attributes: dict[str, str]
) -> None:
    for node in ast.walk(pattern):
        name: str | None = None
        if isinstance(node, (ast.MatchAs, ast.MatchStar)):
            name = node.name
        elif isinstance(node, ast.MatchMapping):
            name = node.rest
        if name is not None:
            aliases.pop(name, None)
            class_attributes.pop(name, None)


def _set_class_state(
    aliases: dict[str, str],
    class_attributes: dict[str, str],
    base_aliases: dict[str, str],
    base_attributes: dict[str, str],
    branch_aliases: Sequence[dict[str, str]],
    branch_attributes: Sequence[dict[str, str]],
) -> None:
    aliases.clear()
    aliases.update(_merge_class_maps(base_aliases, branch_aliases))
    class_attributes.clear()
    class_attributes.update(_merge_class_maps(base_attributes, branch_attributes))


def _apply_class_conditional(
    statement: ast.If | ast.While,
    aliases: dict[str, str],
    class_attributes: dict[str, str],
) -> None:
    base_aliases = dict(aliases)
    base_attributes = dict(class_attributes)
    body_aliases = dict(base_aliases)
    body_attributes = dict(base_attributes)
    _apply_class_body(statement.body, body_aliases, body_attributes)
    else_aliases = dict(base_aliases)
    else_attributes = dict(base_attributes)
    _apply_class_body(statement.orelse, else_aliases, else_attributes)
    _set_class_state(
        aliases,
        class_attributes,
        base_aliases,
        base_attributes,
        [body_aliases, else_aliases],
        [body_attributes, else_attributes],
    )


def _apply_class_loop(
    statement: ast.For | ast.AsyncFor | ast.While,
    aliases: dict[str, str],
    class_attributes: dict[str, str],
) -> None:
    base_aliases = dict(aliases)
    base_attributes = dict(class_attributes)
    body_aliases = dict(base_aliases)
    body_attributes = dict(base_attributes)
    if isinstance(statement, (ast.For, ast.AsyncFor)):
        _bind_class_iterable(
            statement.target, statement.iter, body_aliases, body_attributes
        )
    _apply_class_body(statement.body, body_aliases, body_attributes)
    else_aliases = dict(base_aliases)
    else_attributes = dict(base_attributes)
    _apply_class_body(statement.orelse, else_aliases, else_attributes)
    _set_class_state(
        aliases,
        class_attributes,
        base_aliases,
        base_attributes,
        [body_aliases, else_aliases],
        [body_attributes, else_attributes],
    )


def _apply_class_with(
    statement: ast.With | ast.AsyncWith,
    aliases: dict[str, str],
    class_attributes: dict[str, str],
) -> None:
    for item in statement.items:
        if item.optional_vars is not None:
            _bind_alias_target(item.optional_vars, item.context_expr, aliases)
            _export_class_target(item.optional_vars, aliases, class_attributes)
    _apply_class_body(statement.body, aliases, class_attributes)


def _apply_class_try(
    statement: ast.Try | ast.TryStar,
    aliases: dict[str, str],
    class_attributes: dict[str, str],
) -> None:
    base_aliases = dict(aliases)
    base_attributes = dict(class_attributes)
    branch_aliases: list[dict[str, str]] = []
    branch_attributes: list[dict[str, str]] = []
    body_aliases = dict(base_aliases)
    body_attributes = dict(base_attributes)
    _apply_class_body(statement.body, body_aliases, body_attributes)
    branch_aliases.append(body_aliases)
    branch_attributes.append(body_attributes)
    for handler in statement.handlers:
        handler_aliases = dict(base_aliases)
        handler_attributes = dict(base_attributes)
        if handler.name is not None:
            handler_aliases.pop(handler.name, None)
            handler_attributes.pop(handler.name, None)
        _apply_class_body(handler.body, handler_aliases, handler_attributes)
        branch_aliases.append(handler_aliases)
        branch_attributes.append(handler_attributes)
    else_aliases = dict(base_aliases)
    else_attributes = dict(base_attributes)
    _apply_class_body(statement.orelse, else_aliases, else_attributes)
    branch_aliases.append(else_aliases)
    branch_attributes.append(else_attributes)
    merged_aliases = _merge_class_maps(base_aliases, branch_aliases)
    merged_attributes = _merge_class_maps(base_attributes, branch_attributes)
    _apply_class_body(statement.finalbody, merged_aliases, merged_attributes)
    aliases.clear()
    aliases.update(merged_aliases)
    class_attributes.clear()
    class_attributes.update(merged_attributes)


def _apply_class_match(
    statement: ast.Match,
    aliases: dict[str, str],
    class_attributes: dict[str, str],
) -> None:
    base_aliases = dict(aliases)
    base_attributes = dict(class_attributes)
    branch_aliases: list[dict[str, str]] = []
    branch_attributes: list[dict[str, str]] = []
    for case in statement.cases:
        case_aliases = dict(base_aliases)
        case_attributes = dict(base_attributes)
        _bind_match_captures(case.pattern, case_aliases, case_attributes)
        _apply_class_body(case.body, case_aliases, case_attributes)
        branch_aliases.append(case_aliases)
        branch_attributes.append(case_attributes)
    _set_class_state(
        aliases,
        class_attributes,
        base_aliases,
        base_attributes,
        branch_aliases,
        branch_attributes,
    )


def _apply_class_body(
    statements: Sequence[ast.stmt],
    aliases: dict[str, str],
    class_attributes: dict[str, str],
) -> None:
    for statement in statements:
        if isinstance(statement, ast.Assign):
            _record_class_bindings(
                statement.targets, statement.value, aliases, class_attributes
            )
        elif isinstance(statement, ast.AnnAssign):
            _record_class_bindings(
                [statement.target], statement.value, aliases, class_attributes
            )
        elif isinstance(statement, ast.If):
            _apply_class_conditional(statement, aliases, class_attributes)
        elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            _apply_class_loop(statement, aliases, class_attributes)
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            _apply_class_with(statement, aliases, class_attributes)
        elif isinstance(statement, (ast.Try, ast.TryStar)):
            _apply_class_try(statement, aliases, class_attributes)
        elif isinstance(statement, ast.Match):
            _apply_class_match(statement, aliases, class_attributes)
        elif isinstance(
            statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            aliases.pop(statement.name, None)
            class_attributes.pop(statement.name, None)
        else:
            for node in ast.walk(statement):
                if isinstance(node, ast.NamedExpr):
                    _bind_alias_target(node.target, node.value, aliases)
                    _export_class_target(node.target, aliases, class_attributes)


def _class_body_facts(
    class_node: ast.ClassDef,
    aliases: dict[str, str],
    class_attributes: dict[str, str],
) -> None:
    class_aliases = dict(aliases)
    _apply_class_body(class_node.body, class_aliases, class_attributes)


def _class_container_element(
    value: ast.AST,
    class_attributes: dict[str, str],
    class_name: str,
) -> str | None:
    if not isinstance(value, ast.Subscript):
        return None
    base = value.value
    if not isinstance(base, ast.Attribute) or not isinstance(base.value, ast.Name):
        return None
    key = _literal_subscript_key(value)
    if base.value.id in {"self", "cls"}:
        resolved = _class_container_value(
            class_attributes, f"{base.value.id}.{base.attr}", key
        )
        if resolved is not None:
            return resolved
        return _class_container_value(class_attributes, base.attr, key)
    if base.value.id == class_name:
        return _class_container_value(class_attributes, base.attr, key)
    return None


def _class_receiver_name(
    name: str, class_name: str, receiver_aliases: set[str]
) -> str | None:
    if name in {"self", "cls"}:
        return name
    if name in receiver_aliases:
        return "self"
    if name == class_name:
        return class_name
    return None


def _class_attribute_chain(
    value: ast.AST,
    class_attributes: dict[str, str],
    class_name: str,
    receiver_aliases: set[str] | None = None,
) -> str | None:
    if not isinstance(value, ast.Attribute):
        return None
    base = value.value
    if not isinstance(base, ast.Attribute) or not isinstance(base.value, ast.Name):
        return None
    receiver = _class_receiver_name(
        base.value.id, class_name, receiver_aliases or set()
    )
    if receiver is None:
        return None
    if receiver in {"self", "cls"}:
        resolved = class_attributes.get(f"{receiver}.{base.attr}")
        if resolved is None:
            resolved = class_attributes.get(base.attr)
    else:
        resolved = class_attributes.get(base.attr)
    if resolved is None:
        return None
    return f"{resolved}.{value.attr}"


def _class_attribute_value(
    value: ast.AST,
    class_attributes: dict[str, str],
    class_name: str,
    receiver_aliases: set[str] | None = None,
) -> str | None:
    if not isinstance(value, ast.Attribute):
        return None
    base = value.value
    if not isinstance(base, ast.Name):
        return None
    receiver = _class_receiver_name(base.id, class_name, receiver_aliases or set())
    if receiver is None:
        return None
    if receiver in {"self", "cls"}:
        resolved = class_attributes.get(f"{receiver}.{value.attr}")
        if resolved is not None:
            return resolved
        return class_attributes.get(value.attr)
    return class_attributes.get(value.attr)


RETURN_SEPARATOR = "|"


def _return_candidates(value: str | None) -> list[str]:
    if not value:
        return []
    return [candidate for candidate in value.split(RETURN_SEPARATOR) if candidate]


def _merge_return_value(existing: str | None, candidate: str) -> str:
    values = set(_return_candidates(existing))
    values.add(candidate)
    return RETURN_SEPARATOR.join(sorted(values))


def _class_parameter_reference(value: ast.AST, parameters: set[str]) -> str | None:
    if isinstance(value, ast.Name):
        if value.id in parameters and value.id not in {"self", "cls"}:
            return f"{CALL_PARAMETER_PREFIX}{value.id}"
        return None
    if isinstance(value, ast.Attribute):
        dotted = _dotted_name(value)
        if dotted is not None:
            root = dotted.split(".", 1)[0]
            if root in parameters and root not in {"self", "cls"}:
                return f"{CALL_PARAMETER_PREFIX}{dotted}"
    return None


def _class_return_candidate(
    value: ast.AST,
    aliases: dict[str, str],
    class_attributes: dict[str, str],
    class_name: str,
    receiver_aliases: set[str],
    parameters: set[str],
    local_references: dict[str, str],
) -> str | None:
    if isinstance(value, ast.Name):
        resolved = local_references.get(value.id)
        if resolved is not None:
            return resolved
    resolved = _class_parameter_reference(value, parameters)
    if resolved is not None:
        return resolved
    resolved = _class_attribute_chain(
        value, class_attributes, class_name, receiver_aliases
    )
    if resolved is not None:
        return resolved
    resolved = _class_attribute_value(
        value, class_attributes, class_name, receiver_aliases
    )
    if resolved is not None:
        return resolved
    resolved = _class_container_element(value, class_attributes, class_name)
    if resolved is not None:
        return resolved
    return _resolved_reference(value, aliases)


def _clear_local_aliases(
    targets: Sequence[ast.AST],
    receiver_aliases: set[str],
    local_references: dict[str, str],
) -> None:
    for target in targets:
        for name in _iter_target_names(target):
            receiver_aliases.discard(name.id)
            local_references.pop(name.id, None)


def _track_method_aliases(
    targets: Sequence[ast.AST],
    value: ast.AST,
    parameters: set[str],
    receiver_aliases: set[str],
    local_references: dict[str, str],
) -> None:
    _clear_local_aliases(targets, receiver_aliases, local_references)
    for target in targets:
        if not isinstance(target, ast.Name):
            continue
        if isinstance(value, ast.Name) and value.id in {"self", "cls"}:
            local_references.pop(target.id, None)
            if target.id != value.id:
                receiver_aliases.add(target.id)
            continue
        receiver_aliases.discard(target.id)
        reference = local_references.get(
            value.id if isinstance(value, ast.Name) else ""
        )
        if reference is None:
            reference = _class_parameter_reference(value, parameters)
        if reference is None:
            local_references.pop(target.id, None)
        else:
            local_references[target.id] = reference


def _invalidate_rebound_aliases(
    node: ast.AST,
    receiver_aliases: set[str],
    local_references: dict[str, str],
) -> bool:
    if isinstance(node, (ast.For, ast.AsyncFor)):
        _clear_local_aliases([node.target], receiver_aliases, local_references)
        return True
    if isinstance(node, (ast.With, ast.AsyncWith)):
        _clear_local_aliases(
            [
                item.optional_vars
                for item in node.items
                if item.optional_vars is not None
            ],
            receiver_aliases,
            local_references,
        )
        return True
    if isinstance(node, ast.ExceptHandler):
        if node.name is not None:
            receiver_aliases.discard(node.name)
            local_references.pop(node.name, None)
        return True
    if isinstance(node, ast.AugAssign):
        _clear_local_aliases([node.target], receiver_aliases, local_references)
        return True
    if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
        receiver_aliases.discard(node.target.id)
        local_references.pop(node.target.id, None)
        return True
    return False


def _class_method_facts(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: dict[str, str],
    class_attributes: dict[str, str],
    class_returns: dict[str, str],
    class_name: str,
) -> None:
    parameters = _function_parameter_names(method)
    reference_parameters = parameters - _method_reassigned_parameters(method)
    receiver_aliases: set[str] = set()
    local_references: dict[str, str] = {}
    for inner in _scope_nodes(method.body):
        if isinstance(inner, ast.Return) and inner.value is not None:
            resolved = _class_return_candidate(
                inner.value,
                aliases,
                class_attributes,
                class_name,
                receiver_aliases,
                reference_parameters,
                local_references,
            )
            if resolved is not None:
                class_returns[method.name] = _merge_return_value(
                    class_returns.get(method.name), resolved
                )
            continue
        if _invalidate_rebound_aliases(inner, receiver_aliases, local_references):
            continue
        if isinstance(inner, ast.Assign):
            targets = inner.targets
            value = inner.value
        elif isinstance(inner, ast.AnnAssign) and inner.value is not None:
            targets = [inner.target]
            value = inner.value
        else:
            continue
        _track_method_aliases(
            targets, value, reference_parameters, receiver_aliases, local_references
        )
        resolved = _class_return_candidate(
            value,
            aliases,
            class_attributes,
            class_name,
            receiver_aliases,
            reference_parameters,
            local_references,
        )
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in {"self", "cls"}
            ):
                if resolved is not None and not resolved.startswith(
                    CALL_PARAMETER_PREFIX
                ):
                    class_attributes[target.attr] = resolved
                elif resolved is not None and method.name == "__init__":
                    class_attributes[target.attr] = (
                        f"{INIT_PARAMETER_PREFIX}"
                        f"{resolved[len(CALL_PARAMETER_PREFIX) :]}"
                    )
                _store_class_container(
                    f"{target.value.id}.{target.attr}",
                    value,
                    aliases,
                    class_attributes,
                )
                continue
            target_key = _subscript_alias_key(target)
            if target_key is not None and resolved is not None:
                class_attributes[target_key] = resolved


def _method_bound_names(method: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for node in _scope_nodes(method.body):
        if isinstance(node, ast.Import):
            names.update(
                alias.asname or alias.name.split(".", 1)[0]
                for alias in node.names
                if alias.name != "*"
            )
            continue
        if isinstance(node, ast.ImportFrom):
            names.update(
                alias.asname or alias.name for alias in node.names if alias.name != "*"
            )
            continue
        if isinstance(node, ast.ExceptHandler):
            if node.name is not None:
                names.add(node.name)
            continue
        if isinstance(node, ast.match_case):
            names.update(_pattern_capture_names(node.pattern))
            continue
        if isinstance(node, ast.Assign):
            targets: Sequence[ast.AST] = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
            targets = [node.target]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            targets = [
                item.optional_vars
                for item in node.items
                if item.optional_vars is not None
            ]
        elif isinstance(node, ast.NamedExpr):
            targets = [node.target]
        else:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            continue
        for target in targets:
            names.update(name.id for name in _iter_target_names(target))
    return names


def _method_shadowed_names(method: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    return set(_function_parameter_names(method)) | _method_bound_names(method)


def _method_reassigned_parameters(
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    parameters = set(_function_parameter_names(method))
    return (_method_bound_names(method) & parameters) - {"self", "cls"}


def _method_aliases(
    aliases: dict[str, str],
    class_attributes: dict[str, str],
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, str]:
    shadowed = _method_shadowed_names(method)
    resolved = dict(aliases)
    for key, value in class_attributes.items():
        if key.split("[", 1)[0] in shadowed:
            continue
        resolved[key] = value
    return resolved


def _resolved_base_names(
    base: ast.AST,
    aliases: dict[str, str],
    local_names: set[str],
    external_names: set[str],
) -> str | None:
    original = _dotted_name(base)
    if original is None:
        return None
    if original in local_names:
        return original
    if original in external_names:
        return original
    target = original
    for _ in range(len(aliases) + 1):
        next_target = aliases.get(target)
        if next_target is None or next_target == target:
            break
        target = next_target
    if target in local_names:
        return target
    if target in external_names:
        return target
    suffix = target.rsplit(".", 1)[-1]
    if suffix in local_names:
        return suffix
    if suffix in external_names:
        return suffix
    return None


def _class_bases(
    classes: Sequence[ast.ClassDef],
    aliases: dict[str, str],
    external_names: set[str] | None = None,
) -> dict[str, list[str]]:
    local_names = {node.name for node in classes}
    external = set(external_names or ())
    bases: dict[str, list[str]] = {}
    for node in classes:
        resolved: list[str] = []
        for base in node.bases:
            target = _resolved_base_names(base, aliases, local_names, external)
            if target is not None and target not in resolved:
                resolved.append(target)
        bases[node.name] = resolved
    return bases


def _class_methods(
    class_node: ast.ClassDef,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        statement
        for statement in class_node.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _class_defined_keys(class_node: ast.ClassDef) -> tuple[set[str], set[str]]:
    names: set[str] = set()
    for node in _scope_nodes(class_node.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue
        if isinstance(node, ast.Assign):
            targets: Sequence[ast.AST] = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.For, ast.AsyncFor)):
            targets = [node.target]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            targets = [
                item.optional_vars
                for item in node.items
                if item.optional_vars is not None
            ]
        elif isinstance(node, ast.NamedExpr):
            targets = [node.target]
        else:
            continue
        for target in targets:
            names.update(name.id for name in _iter_target_names(target))
    attributes: set[str] = set()
    for method in _class_methods(class_node):
        for node in _scope_nodes(method.body):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in {"self", "cls"}
                ):
                    attributes.add(target.attr)
                    attributes.add(f"{target.value.id}.{target.attr}")
    return names, attributes


def _shadowed_class_key(key: str, names: set[str], attributes: set[str]) -> bool:
    if key in attributes:
        return True
    if key.startswith(("self.", "cls.")):
        return key.split(".", 1)[1] in attributes
    return key.split("[", 1)[0] in names


def _merge_base_facts(
    own_attributes: dict[str, str],
    own_returns: dict[str, str],
    base_names: Sequence[str],
    merged_attributes: dict[str, dict[str, str]],
    merged_returns: dict[str, dict[str, str]],
    defined_names: set[str],
    defined_attributes: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    attributes = dict(own_attributes)
    returns = dict(own_returns)
    for base in base_names:
        base_attributes = {
            key: value
            for key, value in merged_attributes.get(base, {}).items()
            if not _shadowed_class_key(key, defined_names, defined_attributes)
        }
        base_returns = {
            key: value
            for key, value in merged_returns.get(base, {}).items()
            if key not in defined_names
        }
        attributes = {**base_attributes, **attributes}
        returns = {**base_returns, **returns}
    return attributes, returns


def _is_super_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "super"
    )


def _super_return_candidates(
    value: ast.AST,
    attributes: dict[str, str],
    returns: dict[str, str],
) -> set[str]:
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and _is_super_call(value.func.value)
    ):
        return set(_return_candidates(returns.get(value.func.attr)))
    if isinstance(value, ast.Attribute) and _is_super_call(value.value):
        resolved = attributes.get(value.attr) or attributes.get(f"self.{value.attr}")
        return set(_return_candidates(resolved))
    return set()


def _run_class_methods(
    class_node: ast.ClassDef,
    aliases: dict[str, str],
    merged_attributes: dict[str, str],
    merged_returns: dict[str, str],
    own_attributes: dict[str, str],
    own_returns: dict[str, str],
    base_attributes: dict[str, str],
    base_returns: dict[str, str],
) -> bool:
    changed = False
    attributes = dict(merged_attributes)
    returns = dict(merged_returns)
    for method in _class_methods(class_node):
        method_aliases = _method_aliases(aliases, attributes, method)
        _class_method_facts(
            method,
            method_aliases,
            attributes,
            returns,
            class_node.name,
        )
        super_candidates: set[str] = set()
        for inner in _scope_nodes(method.body):
            if isinstance(inner, ast.Return) and inner.value is not None:
                super_candidates |= _super_return_candidates(
                    inner.value, base_attributes, base_returns
                )
        if super_candidates:
            merged = set(_return_candidates(own_returns.get(method.name)))
            merged |= super_candidates
            resolved_value = RETURN_SEPARATOR.join(sorted(merged))
            if own_returns.get(method.name) != resolved_value:
                own_returns[method.name] = resolved_value
                changed = True
        for key, value in attributes.items():
            if merged_attributes.get(key) != value:
                own_attributes[key] = value
                changed = True
        for key, value in returns.items():
            if merged_returns.get(key) != value:
                own_returns[key] = value
                changed = True
    return changed


def _class_namespace_facts(
    tree: ast.Module,
    aliases: dict[str, str],
    imported_attributes: dict[str, dict[str, str]] | None = None,
    imported_returns: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    external_attributes = imported_attributes or {}
    external_returns = imported_returns or {}
    bases = _class_bases(
        classes, aliases, set(external_attributes) | set(external_returns)
    )
    defined_keys = {
        class_node.name: _class_defined_keys(class_node) for class_node in classes
    }
    own_attributes: dict[str, dict[str, str]] = {}
    own_returns: dict[str, dict[str, str]] = {}
    for class_node in classes:
        own_attributes.setdefault(class_node.name, {})
        own_returns.setdefault(class_node.name, {})
        _class_body_facts(class_node, aliases, own_attributes[class_node.name])
    attributes = {name: dict(values) for name, values in external_attributes.items()}
    returns = {name: dict(values) for name, values in external_returns.items()}
    for name, values in own_attributes.items():
        attributes[name] = dict(values)
    for name, values in own_returns.items():
        returns[name] = dict(values)
    for _ in range(len(classes) + 2):
        changed = False
        for class_node in classes:
            name = class_node.name
            defined_names, defined_attributes = defined_keys[name]
            base_attributes, base_returns = _merge_base_facts(
                {},
                {},
                bases.get(name, []),
                attributes,
                returns,
                set(),
                set(),
            )
            merged_attributes, merged_returns = _merge_base_facts(
                own_attributes[name],
                own_returns[name],
                bases.get(name, []),
                attributes,
                returns,
                defined_names,
                defined_attributes,
            )
            if merged_attributes != attributes[name]:
                attributes[name] = merged_attributes
                changed = True
            if merged_returns != returns[name]:
                returns[name] = merged_returns
                changed = True
            if _run_class_methods(
                class_node,
                aliases,
                merged_attributes,
                merged_returns,
                own_attributes[name],
                own_returns[name],
                base_attributes,
                base_returns,
            ):
                changed = True
        if not changed:
            break
    _propagate_class_alias_facts(tree, attributes, returns)
    _drop_rebound_class_facts(tree, attributes, returns)
    return attributes, returns


def _propagate_class_alias_facts(
    tree: ast.Module,
    attributes: dict[str, dict[str, str]],
    returns: dict[str, dict[str, str]],
) -> None:
    for alias, target in _class_alias_map(tree).items():
        if target in attributes:
            attributes[alias] = attributes[target]
        if target in returns:
            returns[alias] = returns[target]


def _drop_rebound_class_facts(
    tree: ast.Module,
    attributes: dict[str, dict[str, str]],
    returns: dict[str, dict[str, str]],
) -> None:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            assignment_value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            assignment_value = node.value
        else:
            continue
        keep_alias = (
            isinstance(assignment_value, ast.Name) and assignment_value.id in attributes
        )
        for rebind_target in targets:
            if isinstance(rebind_target, ast.Name) and not keep_alias:
                attributes.pop(rebind_target.id, None)
                returns.pop(rebind_target.id, None)


def _namespace_definition_references(
    node: ast.AST,
    aliases: dict[str, str],
    check: Callable[[ast.AST, str | None], None],
) -> None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        for decorator in node.decorator_list:
            check(decorator, _resolved_reference(decorator, aliases))
    if isinstance(node, ast.Return) and node.value is not None:
        check(node, _resolved_reference(node.value, aliases))
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in {"self", "cls"}
            ):
                check(node.value, _resolved_reference(node.value, aliases))
    elif (
        isinstance(node, ast.AnnAssign)
        and node.value is not None
        and isinstance(node.target, ast.Attribute)
        and isinstance(node.target.value, ast.Name)
        and node.target.value.id in {"self", "cls"}
    ):
        check(node.value, _resolved_reference(node.value, aliases))
    if isinstance(node, ast.ClassDef):
        for statement in node.body:
            if (
                isinstance(statement, (ast.Assign, ast.AnnAssign))
                and statement.value is not None
            ):
                check(statement.value, _resolved_reference(statement.value, aliases))


def _class_chain_reference(
    node: ast.Attribute,
    attributes: dict[str, dict[str, str]],
    class_alias_nodes: dict[int, str] | None = None,
    invalid_classes: frozenset[str] = frozenset(),
) -> str | None:
    aliases = class_alias_nodes or {}
    if isinstance(node.value, ast.Name):
        name = node.value.id
        if name in {"self", "cls"}:
            return None
        if name in invalid_classes:
            return None
        class_name = aliases.get(id(node.value), name)
        class_attributes = attributes.get(class_name)
        if class_attributes is not None and node.attr in class_attributes:
            return class_attributes[node.attr]
        return None
    if isinstance(node.value, ast.Attribute):
        base = _class_chain_reference(node.value, attributes, aliases, invalid_classes)
        if base is not None:
            return f"{base}.{node.attr}"
    value = node.value
    if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Attribute):
        subscript = value
        base_attr = value.value
        key = _literal_subscript_key(subscript)
        if isinstance(base_attr.value, ast.Name):
            root = base_attr.value
            if root.id in invalid_classes:
                return None
            class_name = aliases.get(id(root), root.id)
            class_attributes = attributes.get(class_name, {})
            resolved = _class_container_value(class_attributes, base_attr.attr, key)
            if resolved is not None:
                return f"{resolved}.{node.attr}"
        base = _class_chain_reference(base_attr, attributes, aliases, invalid_classes)
        if base is not None:
            return f"{base}.{node.attr}"
    return None


def _class_subscript_reference(
    node: ast.Subscript,
    attributes: dict[str, dict[str, str]],
    contexts: dict[int, str | None],
    container_nodes: dict[int, dict[str, str]] | None = None,
    class_alias_nodes: dict[int, str] | None = None,
    invalid_classes: frozenset[str] = frozenset(),
) -> str | None:
    candidates = _subscript_candidates(
        node,
        attributes,
        contexts,
        container_nodes or {},
        class_alias_nodes or {},
        invalid_classes,
        include_aliases=False,
    )
    if not candidates:
        return None
    return _pick_candidate(candidates)


def _class_container_alias_reference(
    func: ast.AST, container_nodes: dict[int, dict[str, str]]
) -> str | None:
    if not isinstance(func, ast.Subscript) or not isinstance(func.value, ast.Name):
        return None
    elements = container_nodes.get(id(func.value))
    if elements is None:
        return None
    return _container_map_value(elements, _literal_subscript_key(func))


def _leftmost_name(node: ast.AST) -> ast.Name | None:
    current = node
    while True:
        if isinstance(current, ast.Name):
            return current
        if isinstance(current, (ast.Attribute, ast.Subscript)):
            current = current.value
            continue
        return None


def _class_mapping_values_reference(
    node: ast.Call,
    attributes: dict[str, dict[str, str]],
    contexts: dict[int, str | None],
    class_alias_nodes: dict[int, str] | None = None,
    invalid_classes: frozenset[str] = frozenset(),
) -> str | None:
    function = node.func
    if not isinstance(function, ast.Attribute) or function.attr not in {
        "values",
        "items",
    }:
        return None
    if node.args or node.keywords:
        return None
    receiver = function.value
    receiver_key = _subscript_alias_key(receiver)
    if receiver_key is None:
        return None
    root = _leftmost_name(receiver)
    if root is None:
        return None
    if root.id in {"self", "cls"}:
        class_name = contexts.get(id(node))
        if class_name is None:
            return None
        class_attributes = attributes.get(class_name, {})
    else:
        if root.id in invalid_classes:
            return None
        class_name = (class_alias_nodes or {}).get(id(root), root.id)
        resolved_attributes = attributes.get(class_name)
        if resolved_attributes is None:
            return None
        class_attributes = resolved_attributes
    prefix = f"{receiver_key}["
    candidates = [
        value for key, value in class_attributes.items() if key.startswith(prefix)
    ]
    return _pick_candidate(candidates)


def _class_container_get_reference(
    node: ast.Call,
    attributes: dict[str, dict[str, str]],
    contexts: dict[int, str | None],
    container_nodes: dict[int, dict[str, str]] | None = None,
    class_alias_nodes: dict[int, str] | None = None,
    invalid_classes: frozenset[str] = frozenset(),
) -> str | None:
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "get":
        return None
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return None
    receiver = node.func.value
    key = repr(node.args[0].value)
    aliases = class_alias_nodes or {}
    if isinstance(receiver, ast.Name):
        if receiver.id in invalid_classes:
            return None
        elements = (container_nodes or {}).get(id(receiver))
        if elements is None:
            return None
        return _container_map_value(elements, key)
    if not isinstance(receiver, ast.Attribute) or not isinstance(
        receiver.value, ast.Name
    ):
        return None
    root = receiver.value
    if root.id in invalid_classes:
        return None
    if root.id in {"self", "cls"}:
        class_name = contexts.get(id(node))
        if class_name is not None:
            class_attributes = attributes.get(class_name, {})
            resolved = _class_container_value(
                class_attributes, f"{root.id}.{receiver.attr}", key
            )
            if resolved is not None:
                return resolved
            return _class_container_value(class_attributes, receiver.attr, key)
        return None
    class_name = aliases.get(id(root), root.id)
    resolved_attributes = attributes.get(class_name)
    if resolved_attributes is None:
        return None
    return _class_container_value(resolved_attributes, receiver.attr, key)


def _function_parameter_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    parameters = [
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
        function.args.vararg,
        function.args.kwarg,
    ]
    return {parameter.arg for parameter in parameters if parameter is not None}


@dataclass(frozen=True)
class _ConstructorBinding:
    """A constructor call together with its resolved init-parameter values."""

    constructor: ast.Call
    parameter_values: tuple[str | None, ...]
    owner: str | None = None


@dataclass
class _ElementScope:
    aliases: dict[str, str] = field(default_factory=dict)
    containers: dict[str, dict[str, str]] = field(default_factory=dict)
    class_aliases: dict[str, str] = field(default_factory=dict)
    invalid_classes: set[str] = field(default_factory=set)
    container_sources: dict[str, str] = field(default_factory=dict)
    container_columns: dict[str, dict[int, list[str]]] = field(default_factory=dict)
    instance_aliases: dict[str, frozenset[str]] = field(default_factory=dict)
    shadowed_identities: set[str] = field(default_factory=set)
    instance_constructors: dict[str, frozenset[_ConstructorBinding]] = field(
        default_factory=dict
    )
    identity_aliases: dict[str, str] = field(default_factory=dict)
    definitely_rebound: set[str] = field(default_factory=set)
    reachable: bool = True
    escapes_loop: bool = False
    escape_kind: str | None = None
    bound_names: set[str] = field(default_factory=set)
    class_bound_names: set[str] = field(default_factory=set)
    is_class_scope: bool = False

    def clone(self) -> _ElementScope:
        return _ElementScope(
            aliases=dict(self.aliases),
            containers={
                name: dict(elements) for name, elements in self.containers.items()
            },
            class_aliases=dict(self.class_aliases),
            invalid_classes=set(self.invalid_classes),
            container_sources=dict(self.container_sources),
            container_columns={
                name: {index: list(values) for index, values in columns.items()}
                for name, columns in self.container_columns.items()
            },
            instance_aliases=dict(self.instance_aliases),
            shadowed_identities=set(self.shadowed_identities),
            instance_constructors=dict(self.instance_constructors),
            identity_aliases=dict(self.identity_aliases),
            definitely_rebound=set(self.definitely_rebound),
            reachable=self.reachable,
            escapes_loop=self.escapes_loop,
            escape_kind=self.escape_kind,
            bound_names=set(self.bound_names),
            class_bound_names=set(self.class_bound_names),
            is_class_scope=self.is_class_scope,
        )


def _merge_scope_scalar_map(
    branches: Sequence[_ElementScope], attribute: str
) -> dict[str, str]:
    keys: set[str] = set()
    for branch in branches:
        keys.update(getattr(branch, attribute))
    result: dict[str, str] = {}
    for key in keys:
        values = [
            getattr(branch, attribute)[key]
            for branch in branches
            if key in getattr(branch, attribute)
        ]
        preferred = _pick_candidate(values)
        if preferred is not None:
            result[key] = preferred
    return result


def _merge_scope_name_sets(
    branches: Sequence[_ElementScope], attribute: str
) -> dict[str, frozenset[str]]:
    keys: set[str] = set()
    for branch in branches:
        keys.update(getattr(branch, attribute))
    result: dict[str, frozenset[str]] = {}
    for key in keys:
        names: set[str] = set()
        for branch in branches:
            names.update(getattr(branch, attribute).get(key, frozenset()))
        if names:
            result[key] = frozenset(names)
    return result


def _merge_scope_containers(
    branches: Sequence[_ElementScope],
) -> dict[str, dict[str, str]]:
    keys: set[str] = set()
    for branch in branches:
        keys.update(branch.containers)
    result: dict[str, dict[str, str]] = {}
    for key in keys:
        maps = [
            branch.containers[key] for branch in branches if key in branch.containers
        ]
        element_keys: set[str] = set()
        for elements in maps:
            element_keys.update(elements)
        merged_elements: dict[str, str] = {}
        for element_key in element_keys:
            values = [
                elements[element_key] for elements in maps if element_key in elements
            ]
            preferred = _pick_candidate(values)
            if preferred is not None:
                merged_elements[element_key] = preferred
        result[key] = merged_elements
    return result


def _merge_scope_sources(branches: Sequence[_ElementScope]) -> dict[str, str]:
    keys: set[str] = set()
    for branch in branches:
        keys.update(branch.container_sources)
    result: dict[str, str] = {}
    for key in keys:
        values = [
            branch.container_sources.get(key, "literal")
            for branch in branches
            if key in branch.container_sources
        ]
        result[key] = "class" if "class" in values else "literal"
    return result


def _merge_scope_columns(
    branches: Sequence[_ElementScope],
) -> dict[str, dict[int, list[str]]]:
    keys: set[str] = set()
    for branch in branches:
        keys.update(branch.container_columns)
    result: dict[str, dict[int, list[str]]] = {}
    for key in keys:
        merged: dict[int, list[str]] = {}
        for branch in branches:
            columns = branch.container_columns.get(key)
            if columns is None:
                continue
            for index, values in columns.items():
                merged.setdefault(index, []).extend(values)
        result[key] = merged
    return result


def _merge_scope_constructors(
    branches: Sequence[_ElementScope],
) -> dict[str, frozenset[_ConstructorBinding]]:
    keys: set[str] = set()
    for branch in branches:
        keys.update(branch.instance_constructors)
    result: dict[str, frozenset[_ConstructorBinding]] = {}
    for key in keys:
        bindings: set[_ConstructorBinding] = set()
        for branch in branches:
            bindings.update(branch.instance_constructors.get(key, frozenset()))
        if bindings:
            result[key] = frozenset(bindings)
    return result


def _copy_scope_state(target: _ElementScope, source: _ElementScope) -> None:
    target.aliases = source.aliases
    target.containers = source.containers
    target.class_aliases = source.class_aliases
    target.invalid_classes = source.invalid_classes
    target.container_sources = source.container_sources
    target.container_columns = source.container_columns
    target.instance_aliases = source.instance_aliases
    target.shadowed_identities = source.shadowed_identities
    target.instance_constructors = source.instance_constructors
    target.identity_aliases = source.identity_aliases
    target.definitely_rebound = source.definitely_rebound
    target.reachable = source.reachable
    target.escapes_loop = source.escapes_loop
    target.escape_kind = source.escape_kind
    target.bound_names = source.bound_names
    target.class_bound_names = source.class_bound_names
    target.is_class_scope = source.is_class_scope


def _merge_scope_shadowed(branches: Sequence[_ElementScope]) -> set[str]:
    shadowed = set(branches[0].shadowed_identities)
    for branch in branches[1:]:
        shadowed &= branch.shadowed_identities
    return shadowed


def _mark_definite_rebinding(target: ast.AST, scope: _ElementScope) -> None:
    for name in _iter_target_names(target):
        scope.definitely_rebound.add(name.id)
        scope.identity_aliases.pop(name.id, None)
        scope.shadowed_identities.add(name.id)


def _pattern_capture_names(pattern: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(pattern):
        if isinstance(node, (ast.MatchAs, ast.MatchStar)):
            if node.name is not None:
                names.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            names.add(node.rest)
    return names


def _pattern_is_irrefutable(pattern: ast.AST) -> bool:
    if isinstance(pattern, ast.MatchAs):
        return pattern.pattern is None
    if isinstance(pattern, ast.MatchOr):
        return any(_pattern_is_irrefutable(item) for item in pattern.patterns)
    return False


_MapKey = TypeVar("_MapKey")
_MapValue = TypeVar("_MapValue")
_SetValue = TypeVar("_SetValue")


def _iterable_guaranteed_nonempty(
    value: ast.AST,
    scope: _ElementScope | None = None,
    *,
    allow_range: bool = True,
) -> bool:
    """Return whether a literal iterable always yields at least one element."""
    if isinstance(value, (ast.List, ast.Set)):
        return any(not isinstance(element, ast.Starred) for element in value.elts)
    if isinstance(value, ast.Tuple):
        return any(not isinstance(element, ast.Starred) for element in value.elts)
    if isinstance(value, ast.Dict):
        return any(key is not None for key in value.keys)
    if isinstance(value, ast.Constant) and isinstance(value.value, (str, bytes)):
        return bool(value.value)
    if not (
        allow_range
        and isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "range"
        and (
            scope is None
            or not ("range" in scope.bound_names or "range" in scope.class_bound_names)
        )
        and not value.keywords
        and 1 <= len(value.args) <= 3
    ):
        return False
    constants: list[int] = []
    for argument in value.args:
        if not isinstance(argument, ast.Constant) or not isinstance(
            argument.value, int
        ):
            return False
        constants.append(int(argument.value))
    if len(constants) == 1:
        return constants[0] > 0
    step = constants[2] if len(constants) == 3 else 1
    if step == 0:
        return False
    if step > 0:
        return constants[0] < constants[1]
    return constants[0] > constants[1]


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _identity_function_parameter(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str | None:
    parameters = [*node.args.posonlyargs, *node.args.args]
    if len(parameters) != 1 or node.args.vararg or node.args.kwarg:
        return None
    body = [statement for statement in node.body if not _is_docstring(statement)]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None
    value = body[0].value
    if isinstance(value, ast.Name) and value.id == parameters[0].arg:
        return value.id
    return None


def _identity_lambda_parameter(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Lambda):
        return None
    parameters = [*node.args.posonlyargs, *node.args.args]
    if len(parameters) != 1 or node.args.vararg or node.args.kwarg:
        return None
    if isinstance(node.body, ast.Name) and node.body.id == parameters[0].arg:
        return node.body.id
    return None


def _identity_functions(tree: ast.Module) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        parameter = _identity_function_parameter(node)
        if parameter is not None:
            result[node.name] = parameter
    return result


def _identity_parameter_for_value(
    value: ast.AST | None, identities: dict[str, str]
) -> str | None:
    if value is None:
        return None
    parameter = _identity_lambda_parameter(value)
    if parameter is not None:
        return parameter
    if isinstance(value, (ast.Name, ast.Attribute)):
        dotted = _dotted_name(value)
        if dotted is not None:
            return identities.get(dotted)
    return None


def _bind_identity_names(
    targets: Sequence[ast.AST], parameter: str | None, identities: dict[str, str]
) -> None:
    for target in targets:
        for name in _iter_target_names(target):
            if parameter is None:
                identities.pop(name.id, None)
            else:
                identities[name.id] = parameter


def _module_assignment_targets(
    statement: ast.stmt,
) -> tuple[Sequence[ast.AST], ast.AST | None] | None:
    if isinstance(statement, ast.Assign):
        return statement.targets, statement.value
    if isinstance(statement, ast.AnnAssign):
        return [statement.target], statement.value
    return None


def _module_identity_functions(
    tree: ast.Module, imported: dict[str, str] | None = None
) -> dict[str, str]:
    """Return identity helpers and aliases bound at module level.

    The registry tracks ``def`` helpers, identity lambdas, imported helpers
    and transitive assignments in source order, so a later rebinding removes
    an earlier identity fact.
    """
    imported = imported or {}
    result: dict[str, str] = {}
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parameter = _identity_function_parameter(statement)
            if parameter is None:
                result.pop(statement.name, None)
            else:
                result[statement.name] = parameter
            continue
        if isinstance(statement, ast.ClassDef):
            result.pop(statement.name, None)
            continue
        if isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                if alias.name == "*":
                    continue
                local = alias.asname or alias.name
                parameter = imported.get(local)
                if parameter is None:
                    result.pop(local, None)
                else:
                    result[local] = parameter
            continue
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                result.pop(alias.asname or alias.name.split(".", 1)[0], None)
            continue
        if isinstance(statement, ast.Delete):
            _bind_identity_names(statement.targets, None, result)
            continue
        assignment = _module_assignment_targets(statement)
        if assignment is None:
            continue
        targets, value = assignment
        available = dict(imported)
        available.update(result)
        _bind_identity_names(
            targets, _identity_parameter_for_value(value, available), result
        )
    return result


def _identity_static_aliases(tree: ast.Module) -> dict[str, str]:
    """Expose identity helpers to the path scanner through scoped aliases."""
    result: dict[str, str] = {}
    for name in _identity_functions(tree):
        marker = _identity_alias_key(name)
        result[name] = marker
        result[marker] = name
    return result


def _identity_flow_aliases(tree: ast.Module) -> FlowAliases:
    return {
        key: frozenset({value}) for key, value in _identity_static_aliases(tree).items()
    }


class _ElementAnalysis:
    """Track element and class aliases per lexical scope.

    Aliases may come from plain assignments, ``for`` targets, unpacking,
    comprehension targets and literal containers.  Class names rebound to a
    non-class value are recorded as invalid for the rest of their scope, so
    module-level class facts are not reused for a local shadow.
    """

    def __init__(
        self,
        attributes: dict[str, dict[str, str]],
        contexts: dict[int, str | None],
        identity_functions: dict[str, str] | None = None,
        field_types: dict[str, dict[str, str]] | None = None,
        path_fields: dict[str, set[str]] | None = None,
        parameter_types: dict[int, dict[str, str]] | None = None,
        init_methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None = None,
    ) -> None:
        self.attributes = attributes
        self.contexts = contexts
        self.identity_functions = identity_functions or {}
        self.field_types = field_types or {}
        self.path_fields = path_fields or {}
        self.parameter_types = parameter_types or {}
        self.init_methods = init_methods or {}
        self.element_aliases: dict[int, str] = {}
        self.container_nodes: dict[int, dict[str, str]] = {}
        self.class_alias_nodes: dict[int, str] = {}
        self.instance_alias_nodes: dict[int, str] = {}
        self.instance_constructor_nodes: dict[int, frozenset[_ConstructorBinding]] = {}
        self.invalid_nodes: dict[int, frozenset[str]] = {}
        self._escape_sinks: list[list[_ElementScope]] = []
        self._record_enabled = True

    def run(self, tree: ast.Module) -> None:
        self.visit(tree.body, _ElementScope(), None)

    # -- resolution ---------------------------------------------------
    def _class_attributes(self, class_name: str | None) -> dict[str, str]:
        if class_name is None:
            return {}
        return self.attributes.get(class_name, {})

    def _identity_name(self, name: str, scope: _ElementScope) -> str | None:
        alias = scope.identity_aliases.get(name)
        if alias is not None:
            return alias
        if name in scope.shadowed_identities:
            return None
        head, dot, tail = name.partition(".")
        if dot:
            resolved_head = scope.aliases.get(head, head)
            qualified = f"{resolved_head}.{tail}"
            if qualified in self.identity_functions:
                return qualified
            if qualified in scope.identity_aliases:
                return scope.identity_aliases[qualified]
            return None
        if name in self.identity_functions:
            return name
        return None

    def _attribute_container(
        self, value: ast.Attribute, scope: _ElementScope, class_name: str | None
    ) -> dict[str, str] | None:
        base = value.value
        if not isinstance(base, ast.Name):
            return None
        if base.id in {"self", "cls"}:
            attrs = self._class_attributes(class_name)
            prefixes = [f"{base.id}.{value.attr}", value.attr]
        else:
            if base.id in scope.invalid_classes:
                return None
            owner = scope.class_aliases.get(base.id, base.id)
            attrs = self.attributes.get(owner, {})
            prefixes = [value.attr, f"{owner}.{value.attr}"]
        for prefix in prefixes:
            elements = _class_container_elements_map(attrs, prefix)
            if elements:
                return elements
        return None

    def _literal_elements(
        self,
        values: Sequence[ast.AST],
        scope: _ElementScope,
        class_name: str | None,
    ) -> dict[str, str]:
        elements: dict[str, str] = {}
        for index, element in enumerate(values):
            if isinstance(element, ast.Starred):
                nested = self.elements_of(element.value, scope, class_name)
                if nested:
                    elements.update(nested)
                continue
            resolved = self.value_of(element, scope, class_name)
            if resolved is not None:
                elements[str(index)] = resolved
        return elements

    def _attribute_value(
        self, value: ast.Attribute, scope: _ElementScope, class_name: str | None
    ) -> str | None:
        base = value.value
        if not isinstance(base, ast.Name):
            return None
        if base.id in {"self", "cls"}:
            attrs = self._class_attributes(class_name)
            resolved = attrs.get(f"{base.id}.{value.attr}")
            if resolved is not None:
                return resolved
            return attrs.get(value.attr)
        if base.id in scope.invalid_classes:
            return None
        owner = scope.class_aliases.get(base.id, base.id)
        owner_attributes = self.attributes.get(owner)
        if owner_attributes is None:
            return None
        return owner_attributes.get(value.attr)

    def _attribute_owners(
        self,
        value: ast.Attribute,
        scope: _ElementScope,
        class_name: str | None,
    ) -> frozenset[str]:
        dotted = _dotted_name(value)
        root = _leftmost_name(value)
        if dotted is None or root is None:
            return frozenset()
        prefix = f"{root.id}."
        if not dotted.startswith(prefix):
            return frozenset()
        owners = set(scope.instance_aliases.get(root.id, frozenset()))
        if not owners and root.id in {"self", "cls"} and class_name is not None:
            owners.add(class_name)
        resolved: set[str] = set()
        for owner in owners:
            current: str | None = owner
            for field_name in dotted[len(prefix) :].split("."):
                if current is None:
                    break
                current = self.field_types.get(current, {}).get(field_name)
            if current is not None:
                resolved.add(current)
        return frozenset(resolved)

    def _attribute_owner(
        self,
        value: ast.Attribute,
        scope: _ElementScope,
        class_name: str | None,
    ) -> str | None:
        return _pick_candidate(self._attribute_owners(value, scope, class_name))

    def _constructor_bindings(
        self,
        value: ast.AST,
        instance_classes: frozenset[str],
        scope: _ElementScope,
    ) -> frozenset[_ConstructorBinding]:
        if isinstance(value, ast.Call):
            bindings: set[_ConstructorBinding] = set()
            for instance_class in instance_classes:
                init_method = self.init_methods.get(instance_class)
                if init_method is None:
                    bindings.add(_ConstructorBinding(value, (), instance_class))
                    continue
                parameters = [
                    parameter.arg
                    for parameter in _function_parameters(init_method)
                    if parameter.arg not in {"self", "cls"}
                ]
                values = tuple(
                    _resolve_method_parameter(
                        value, init_method, parameter, scope.aliases
                    )
                    for parameter in parameters
                )
                bindings.add(_ConstructorBinding(value, values, instance_class))
            return frozenset(bindings)
        if isinstance(value, ast.Name):
            return scope.instance_constructors.get(value.id, frozenset())
        return frozenset()

    def _instance_attribute_value(
        self,
        value: ast.Attribute,
        scope: _ElementScope,
        class_name: str | None,
    ) -> str | None:
        if isinstance(value.value, ast.Name):
            owners = scope.instance_aliases.get(value.value.id, frozenset())
        elif isinstance(value.value, ast.Call):
            owners = self._instance_classes(value.value, scope, class_name)
        else:
            return None
        candidates: set[str] = set()
        for owner in owners:
            attributes = self.attributes.get(owner, {})
            resolved = (
                attributes.get(value.attr)
                or attributes.get(f"self.{value.attr}")
                or attributes.get(f"cls.{value.attr}")
            )
            if resolved is None:
                continue
            if not resolved.startswith(INIT_PARAMETER_PREFIX):
                candidates.add(resolved)
                continue
            init_method = self.init_methods.get(owner)
            bindings = self._constructor_bindings(
                value.value, frozenset({owner}), scope
            )
            if not bindings or init_method is None:
                continue
            parameters = [
                parameter.arg
                for parameter in _function_parameters(init_method)
                if parameter.arg not in {"self", "cls"}
            ]
            specification = resolved[len(INIT_PARAMETER_PREFIX) :]
            name, _, suffix = specification.partition(".")
            if name not in parameters:
                continue
            index = parameters.index(name)
            for binding in bindings:
                if binding.owner not in {None, owner}:
                    continue
                base_value = (
                    binding.parameter_values[index]
                    if index < len(binding.parameter_values)
                    else _resolve_method_parameter(
                        binding.constructor, init_method, name, scope.aliases
                    )
                )
                if base_value is not None:
                    candidates.add(f"{base_value}.{suffix}" if suffix else base_value)
        return _pick_candidate(candidates)

    def _attribute_is_path_field(
        self,
        value: ast.Attribute,
        scope: _ElementScope,
        class_name: str | None,
    ) -> bool:
        dotted = _dotted_name(value)
        root = _leftmost_name(value)
        if dotted is None or root is None:
            return False
        prefix = f"{root.id}."
        if not dotted.startswith(prefix):
            return False
        owners = set(scope.instance_aliases.get(root.id, frozenset()))
        if not owners and root.id in {"self", "cls"} and class_name is not None:
            owners.add(class_name)
        relative = dotted[len(prefix) :]
        return any(relative in self.path_fields.get(owner, set()) for owner in owners)

    def value_of(
        self, value: ast.AST, scope: _ElementScope, class_name: str | None
    ) -> str | None:
        if isinstance(value, ast.Name):
            resolved = scope.aliases.get(value.id)
            if resolved is not None:
                return resolved
            return _resolved_reference(value, scope.aliases)
        if isinstance(value, ast.Attribute):
            if self._attribute_is_path_field(value, scope, class_name):
                return "Path"
            resolved = self._instance_attribute_value(value, scope, class_name)
            if resolved is not None:
                return resolved
            resolved = self._attribute_value(value, scope, class_name)
            if resolved is not None:
                return resolved
        if isinstance(value, ast.Subscript):
            elements = self.elements_of(value.value, scope, class_name)
            if elements is not None:
                return _container_map_value(elements, _literal_subscript_key(value))
        if (
            isinstance(value, ast.Call)
            and (dotted := _dotted_name(value.func)) is not None
            and self._identity_name(dotted, scope) is not None
            and len(value.args) == 1
            and not value.keywords
        ):
            return self.value_of(value.args[0], scope, class_name)
        return _resolved_reference(value, scope.aliases)

    def elements_of(
        self, value: ast.AST, scope: _ElementScope, class_name: str | None
    ) -> dict[str, str] | None:
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            return self._literal_elements(value.elts, scope, class_name)
        if isinstance(value, ast.Dict):
            elements: dict[str, str] = {}
            for key, element in zip(value.keys, value.values):
                if not isinstance(key, ast.Constant):
                    continue
                resolved = self.value_of(element, scope, class_name)
                if resolved is not None:
                    elements[repr(key.value)] = resolved
            return elements
        if isinstance(value, ast.Name):
            stored = scope.containers.get(value.id)
            return dict(stored) if stored is not None else None
        if isinstance(value, ast.Attribute):
            return self._attribute_container(value, scope, class_name)
        if isinstance(value, ast.Call):
            function = value.func
            if isinstance(function, ast.Attribute) and function.attr in {
                "values",
                "items",
            }:
                return self.elements_of(function.value, scope, class_name)
            return None
        if isinstance(value, ast.IfExp):
            body = self.elements_of(value.body, scope, class_name)
            orelse = self.elements_of(value.orelse, scope, class_name)
            if body is None and orelse is None:
                return None
            merged = dict(body or {})
            merged.update(orelse or {})
            return merged
        return None

    def container_of(
        self, value: ast.AST, scope: _ElementScope, class_name: str | None
    ) -> tuple[dict[str, str] | None, str]:
        if isinstance(value, ast.Name):
            elements = scope.containers.get(value.id)
            if elements is None:
                return None, "literal"
            return dict(elements), scope.container_sources.get(value.id, "literal")
        if isinstance(value, ast.Attribute):
            return self._attribute_container(value, scope, class_name), "class"
        if isinstance(value, ast.Call):
            function = value.func
            if isinstance(function, ast.Attribute) and function.attr in {
                "values",
                "items",
            }:
                return self.container_of(function.value, scope, class_name)
        if isinstance(value, ast.IfExp):
            body, body_source = self.container_of(value.body, scope, class_name)
            orelse, else_source = self.container_of(value.orelse, scope, class_name)
            if body is None and orelse is None:
                return None, "literal"
            merged = dict(body or {})
            merged.update(orelse or {})
            source = "class" if "class" in {body_source, else_source} else "literal"
            return merged, source
        return self.elements_of(value, scope, class_name), "literal"

    def _literal_columns(
        self,
        value: ast.AST,
        scope: _ElementScope,
        class_name: str | None,
    ) -> dict[int, list[str]] | None:
        if not isinstance(value, (ast.List, ast.Tuple)) or not value.elts:
            return None
        if not all(
            isinstance(element, (ast.Tuple, ast.List)) for element in value.elts
        ):
            return None
        columns: dict[int, list[str]] = {}
        for sample in value.elts:
            if not isinstance(sample, (ast.Tuple, ast.List)):
                continue
            for index, element in enumerate(sample.elts):
                resolved = self.value_of(element, scope, class_name)
                if resolved is not None:
                    columns.setdefault(index, []).append(resolved)
        return columns

    def columns_of(
        self, value: ast.AST, scope: _ElementScope, class_name: str | None
    ) -> dict[int, list[str]] | None:
        if isinstance(value, ast.Name):
            columns = scope.container_columns.get(value.id)
            if columns is None:
                return None
            return {index: list(values) for index, values in columns.items()}
        if isinstance(value, ast.IfExp):
            body = self.columns_of(value.body, scope, class_name)
            orelse = self.columns_of(value.orelse, scope, class_name)
            if body is None and orelse is None:
                return None
            merged = {index: list(values) for index, values in (body or {}).items()}
            for index, values in (orelse or {}).items():
                merged.setdefault(index, []).extend(values)
            return merged
        return self._literal_columns(value, scope, class_name)

    # -- binding ------------------------------------------------------
    def store(
        self,
        name: str,
        element: str | None,
        container: dict[str, str] | None,
        scope: _ElementScope,
        source: str = "literal",
    ) -> None:
        was_class = name in self.attributes or name in scope.class_aliases
        self._bind_scope_name(scope, name)
        scope.definitely_rebound.discard(name)
        scope.class_aliases.pop(name, None)
        scope.instance_aliases.pop(name, None)
        scope.instance_constructors.pop(name, None)
        scope.identity_aliases.pop(name, None)
        if name in self.identity_functions:
            scope.shadowed_identities.add(name)
        if container is not None:
            scope.containers[name] = dict(container)
            scope.container_sources[name] = source
            scope.aliases.pop(name, None)
        else:
            scope.containers.pop(name, None)
            scope.container_sources.pop(name, None)
            if element is None:
                scope.aliases.pop(name, None)
            else:
                scope.aliases[name] = element
        if was_class:
            scope.invalid_classes.add(name)

    def store_class_alias(self, name: str, owner: str, scope: _ElementScope) -> None:
        self._bind_scope_name(scope, name)
        scope.aliases.pop(name, None)
        scope.containers.pop(name, None)
        scope.container_sources.pop(name, None)
        scope.instance_aliases.pop(name, None)
        scope.instance_constructors.pop(name, None)
        scope.identity_aliases.pop(name, None)
        if name in self.identity_functions:
            scope.shadowed_identities.add(name)
        scope.class_aliases[name] = owner
        scope.invalid_classes.discard(name)

    def _instance_classes(
        self,
        value: ast.AST,
        scope: _ElementScope,
        class_name: str | None = None,
    ) -> frozenset[str]:
        if isinstance(value, ast.Name):
            return scope.instance_aliases.get(value.id, frozenset())
        if isinstance(value, ast.Attribute):
            return self._attribute_owners(value, scope, class_name)
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            owner = scope.class_aliases.get(value.func.id)
            if owner is None and value.func.id in self.attributes:
                owner = value.func.id
            identity = self._identity_name(value.func.id, scope)
            if owner is None and identity is not None:
                return (
                    self._instance_classes(value.args[0], scope, class_name)
                    if value.args
                    else frozenset()
                )
            return frozenset({owner}) if owner is not None else frozenset()
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            dotted = _dotted_name(value.func)
            if dotted is not None and self._identity_name(dotted, scope) is not None:
                return (
                    self._instance_classes(value.args[0], scope, class_name)
                    if value.args
                    else frozenset()
                )
            attribute = value.func.attr
            if attribute in self.attributes:
                return frozenset({attribute})
        return frozenset()

    def _instance_class(
        self,
        value: ast.AST,
        scope: _ElementScope,
        class_name: str | None = None,
    ) -> str | None:
        return _pick_candidate(self._instance_classes(value, scope, class_name))

    def _bind_resolved_target(
        self, target: ast.AST, element: str | None, scope: _ElementScope
    ) -> None:
        for name in _iter_target_names(target):
            self.store(name.id, element, None, scope)

    def bind_iterated(
        self,
        target: ast.AST,
        elements: dict[str, str] | None,
        scope: _ElementScope,
    ) -> None:
        element = _pick_candidate(elements.values()) if elements else None
        self._bind_resolved_target(target, element, scope)

    def _bind_unpacking(
        self,
        target: ast.Tuple | ast.List,
        value: ast.AST,
        scope: _ElementScope,
        class_name: str | None,
    ) -> None:
        if isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(
            value.elts
        ):
            for element_target, element_value in zip(target.elts, value.elts):
                self.bind_target(element_target, element_value, scope, class_name)
            return
        elements, _ = self.container_of(value, scope, class_name)
        columns = self.columns_of(value, scope, class_name)
        for index, element_target in enumerate(target.elts):
            if isinstance(element_target, ast.Starred):
                self.bind_iterated(element_target.value, elements, scope)
                continue
            candidates = (columns or {}).get(index, [])
            if candidates:
                element = _pick_candidate(candidates)
            elif elements is not None:
                element = _container_map_value(elements, str(index))
            else:
                element = None
            self._bind_resolved_target(element_target, element, scope)

    def _binding_identity_name(
        self, value: ast.AST, target_name: str, scope: _ElementScope
    ) -> str | None:
        if _identity_lambda_parameter(value) is not None:
            return target_name
        if isinstance(value, (ast.Name, ast.Attribute)):
            dotted = _dotted_name(value)
            if dotted is not None:
                return self._identity_name(dotted, scope)
        return None

    def bind_target(
        self,
        target: ast.AST,
        value: ast.AST,
        scope: _ElementScope,
        class_name: str | None,
    ) -> None:
        if isinstance(target, ast.Starred):
            self.bind_target(target.value, value, scope, class_name)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            self._bind_unpacking(target, value, scope, class_name)
            return
        if not isinstance(target, ast.Name):
            return
        if isinstance(value, ast.Name):
            owner = scope.class_aliases.get(value.id)
            if owner is None and value.id in self.attributes:
                owner = value.id
            if owner is not None:
                self.store_class_alias(target.id, owner, scope)
                return
        identity_name = self._binding_identity_name(value, target.id, scope)
        instance_classes = self._instance_classes(value, scope, class_name)
        preserved_constructors = (
            scope.instance_constructors.get(value.id)
            if isinstance(value, ast.Name) and value.id == target.id
            else None
        )
        container, source = self.container_of(value, scope, class_name)
        if container is not None:
            self.store(target.id, None, container, scope, source)
            if instance_classes:
                scope.instance_aliases[target.id] = instance_classes
                constructors = (
                    preserved_constructors
                    if preserved_constructors is not None
                    else self._constructor_bindings(value, instance_classes, scope)
                )
                if constructors:
                    scope.instance_constructors[target.id] = constructors
            columns = self.columns_of(value, scope, class_name)
            if columns is not None:
                scope.container_columns[target.id] = {
                    index: list(values) for index, values in columns.items()
                }
            else:
                scope.container_columns.pop(target.id, None)
            return
        scope.container_columns.pop(target.id, None)
        element = (
            None
            if isinstance(value, ast.Name)
            else self.value_of(value, scope, class_name)
        )
        self.store(target.id, element, None, scope)
        if instance_classes:
            scope.instance_aliases[target.id] = instance_classes
            constructors = (
                preserved_constructors
                if preserved_constructors is not None
                else self._constructor_bindings(value, instance_classes, scope)
            )
            if constructors:
                scope.instance_constructors[target.id] = constructors
        if identity_name is not None:
            scope.identity_aliases[target.id] = identity_name

    def _bind_loop_literal_target(
        self,
        target: ast.AST,
        value: ast.AST,
        scope: _ElementScope,
        class_name: str | None,
    ) -> None:
        if isinstance(target, ast.Name):
            self._bind_literal_loop_name(target, value, scope, class_name)
            return
        if (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
        ):
            for element_target, element_value in zip(target.elts, value.elts):
                self._bind_loop_literal_target(
                    element_target, element_value, scope, class_name
                )
            return
        self.bind_target(target, value, scope, class_name)

    def _bind_unpacked_iterable(
        self,
        target: ast.AST,
        samples: Sequence[ast.AST] | None,
        scope: _ElementScope,
        class_name: str | None,
        *,
        final_iteration_only: bool = False,
    ) -> bool:
        if not isinstance(target, (ast.Tuple, ast.List)) or not samples:
            return False
        if not all(isinstance(sample, (ast.Tuple, ast.List)) for sample in samples):
            return False
        selected_samples = samples[-1:] if final_iteration_only else samples
        candidate_scopes: list[_ElementScope] = []
        for sample in selected_samples:
            if not isinstance(sample, (ast.Tuple, ast.List)):
                continue
            candidate = scope.clone()
            self._bind_loop_literal_target(target, sample, candidate, class_name)
            candidate_scopes.append(candidate)
        if not candidate_scopes:
            return False
        self._merge(scope, candidate_scopes, collect_exits=False)
        return True

    def _bind_literal_loop_name(
        self,
        target: ast.Name,
        value: ast.AST,
        scope: _ElementScope,
        class_name: str | None,
    ) -> None:
        self.bind_target(target, value, scope, class_name)
        if (
            isinstance(value, ast.Name)
            and target.id not in scope.identity_aliases
            and target.id not in scope.class_aliases
            and target.id not in scope.instance_aliases
            and target.id not in scope.containers
        ):
            resolved = self.value_of(value, scope, class_name)
            if resolved is not None:
                scope.aliases[target.id] = resolved

    def bind_loop(
        self,
        target: ast.AST,
        iterable: ast.AST,
        scope: _ElementScope,
        class_name: str | None,
        *,
        final_iteration_only: bool = False,
    ) -> None:
        if isinstance(target, ast.Name) and isinstance(
            iterable, (ast.List, ast.Tuple, ast.Set)
        ):
            literal_elements = iterable.elts
            if literal_elements and all(
                not isinstance(element, ast.Starred) for element in literal_elements
            ):
                ordered_literal = isinstance(iterable, (ast.List, ast.Tuple))
                selected_elements = (
                    literal_elements[-1:]
                    if final_iteration_only and ordered_literal
                    else literal_elements
                )
                candidate_scopes: list[_ElementScope] = []
                for element in selected_elements:
                    candidate = scope.clone()
                    self._bind_literal_loop_name(target, element, candidate, class_name)
                    candidate_scopes.append(candidate)
                self._merge(scope, candidate_scopes, collect_exits=False)
                return
        if (
            isinstance(iterable, ast.Call)
            and isinstance(iterable.func, ast.Attribute)
            and iterable.func.attr == "items"
            and isinstance(target, (ast.Tuple, ast.List))
            and len(target.elts) == 2
        ):
            self._bind_resolved_target(target.elts[0], None, scope)
            elements = self.elements_of(iterable, scope, class_name)
            self.bind_iterated(target.elts[1], elements, scope)
            return
        samples = (
            list(iterable.elts) if isinstance(iterable, (ast.List, ast.Tuple)) else None
        )
        if self._bind_unpacked_iterable(
            target,
            samples,
            scope,
            class_name,
            final_iteration_only=final_iteration_only,
        ):
            return
        columns = self.columns_of(iterable, scope, class_name)
        if isinstance(target, (ast.Tuple, ast.List)) and columns:
            for index, element_target in enumerate(target.elts):
                column_candidates = columns.get(index, [])
                resolved_element = (
                    _pick_candidate(column_candidates) if column_candidates else None
                )
                self._bind_resolved_target(element_target, resolved_element, scope)
            return
        self.bind_iterated(target, self.elements_of(iterable, scope, class_name), scope)

    # -- recording ----------------------------------------------------
    def _record_attribute_alias(
        self,
        node: ast.Attribute,
        scope: _ElementScope,
        class_name: str | None,
    ) -> None:
        if isinstance(node.value, ast.Name):
            instances = scope.instance_aliases.get(node.value.id, frozenset())
            instance = _pick_candidate(instances)
            if instance is not None:
                self.instance_alias_nodes[id(node.value)] = instance
            else:
                self.instance_alias_nodes.pop(id(node.value), None)
            constructors = scope.instance_constructors.get(node.value.id)
            if constructors:
                self.instance_constructor_nodes[id(node.value)] = constructors
            else:
                self.instance_constructor_nodes.pop(id(node.value), None)
            return
        if (
            isinstance(node.value, ast.Call)
            and (dotted := _dotted_name(node.value.func)) is not None
            and self._identity_name(dotted, scope) is not None
        ):
            element = self.value_of(node.value, scope, class_name)
            if element is not None:
                self.element_aliases[id(node.value)] = element
            else:
                self.element_aliases.pop(id(node.value), None)
            return
        if isinstance(node.value, ast.Call):
            self.element_aliases.pop(id(node.value), None)
            return
        if isinstance(node.value, ast.Attribute):
            element = None
            if (
                isinstance(node.value.value, ast.Name)
                and node.value.value.id in scope.instance_aliases
            ) or isinstance(node.value.value, ast.Call):
                element = self.value_of(node.value, scope, class_name)
            if element is not None:
                self.element_aliases[id(node.value)] = element
            else:
                self.element_aliases.pop(id(node.value), None)

    def _record_alias(
        self, node: ast.AST, scope: _ElementScope, class_name: str | None
    ) -> None:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            name = node.value.id
            element = scope.aliases.get(name)
            if element is not None:
                self.element_aliases[id(node.value)] = element
            else:
                self.element_aliases.pop(id(node.value), None)
            owner = scope.class_aliases.get(name)
            if owner is not None:
                self.class_alias_nodes[id(node.value)] = owner
            else:
                self.class_alias_nodes.pop(id(node.value), None)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            name = node.value.id
            elements = scope.containers.get(name)
            if elements is not None and scope.container_sources.get(name) == "class":
                self.container_nodes[id(node.value)] = dict(elements)
            else:
                self.container_nodes.pop(id(node.value), None)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            owner = scope.class_aliases.get(node.func.id)
            if owner is not None:
                self.class_alias_nodes[id(node.func)] = owner
            else:
                self.class_alias_nodes.pop(id(node.func), None)
        if isinstance(node, ast.Attribute):
            self._record_attribute_alias(node, scope, class_name)

    def _record_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
        scope: _ElementScope,
        class_name: str | None,
    ) -> None:
        current = scope.clone()
        for generator in node.generators:
            self.record(generator.iter, current, class_name)
            self.bind_loop(generator.target, generator.iter, current, class_name)
            for condition in generator.ifs:
                self.record(condition, current, class_name)
        if isinstance(node, ast.DictComp):
            self.record(node.key, current, class_name)
            self.record(node.value, current, class_name)
        else:
            self.record(node.elt, current, class_name)

    def record(
        self, node: ast.AST, scope: _ElementScope, class_name: str | None
    ) -> None:
        if not self._record_enabled:
            return
        self.invalid_nodes[id(node)] = frozenset(scope.invalid_classes)
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            return
        if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            self.bind_target(node.target, node.value, scope, class_name)
        if isinstance(
            node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
            self._record_comprehension(node, scope, class_name)
            return
        self._record_alias(node, scope, class_name)
        for child in ast.iter_child_nodes(node):
            self.record(child, scope, class_name)

    # -- scope walk ---------------------------------------------------
    def _apply_statement(
        self, statement: ast.stmt, scope: _ElementScope, class_name: str | None
    ) -> bool:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                self.bind_target(target, statement.value, scope, class_name)
            return True
        if isinstance(statement, ast.AnnAssign):
            if statement.value is None:
                self._bind_resolved_target(statement.target, None, scope)
            else:
                self.bind_target(statement.target, statement.value, scope, class_name)
            return True
        if isinstance(statement, (ast.AugAssign, ast.Delete)):
            targets = (
                [statement.target]
                if isinstance(statement, ast.AugAssign)
                else statement.targets
            )
            for target in targets:
                self._bind_resolved_target(target, None, scope)
            return True
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            for alias in statement.names:
                name = alias.asname or alias.name.split(".")[0]
                self._bind_scope_name(scope, name)
                scope.aliases.pop(name, None)
            return True
        return False

    @staticmethod
    def _bind_scope_name(scope: _ElementScope, name: str) -> None:
        if scope.is_class_scope:
            scope.class_bound_names.add(name)
        else:
            scope.bound_names.add(name)

    @staticmethod
    def _normalized_exit(
        scope: _ElementScope, kind: str | None = None
    ) -> _ElementScope:
        exit_scope = scope.clone()
        exit_scope.reachable = True
        exit_scope.escapes_loop = False
        exit_scope.escape_kind = kind if kind is not None else scope.escape_kind
        return exit_scope

    def _append_loop_exit(self, scope: _ElementScope) -> None:
        if not self._escape_sinks:
            return
        exit_scope = scope.clone()
        exit_scope.reachable = True
        exit_scope.escapes_loop = False
        self._escape_sinks[-1].append(exit_scope)

    def _copy_target_state(
        self, scope: _ElementScope, source: _ElementScope, target: ast.AST
    ) -> None:
        for name_node in _iter_target_names(target):
            name = name_node.id
            self._copy_name_scalar(scope.aliases, source.aliases, name)
            self._copy_name_scalar(scope.class_aliases, source.class_aliases, name)
            self._copy_name_map(scope.containers, source.containers, name)
            self._copy_name_scalar(
                scope.container_sources, source.container_sources, name
            )
            self._copy_name_map(scope.container_columns, source.container_columns, name)
            self._copy_name_set(scope.instance_aliases, source.instance_aliases, name)
            self._copy_name_set(
                scope.instance_constructors, source.instance_constructors, name
            )
            self._copy_name_scalar(
                scope.identity_aliases, source.identity_aliases, name
            )
            if name in source.invalid_classes:
                scope.invalid_classes.add(name)
            else:
                scope.invalid_classes.discard(name)
            if name in source.shadowed_identities:
                scope.shadowed_identities.add(name)
            else:
                scope.shadowed_identities.discard(name)
            if name in source.definitely_rebound:
                scope.definitely_rebound.add(name)
            else:
                scope.definitely_rebound.discard(name)

    @staticmethod
    def _copy_name_scalar(
        target: dict[str, str], source: dict[str, str], name: str
    ) -> None:
        value = source.get(name)
        if value is None:
            target.pop(name, None)
        else:
            target[name] = value

    @staticmethod
    def _copy_name_map(
        target: dict[str, dict[_MapKey, _MapValue]],
        source: dict[str, dict[_MapKey, _MapValue]],
        name: str,
    ) -> None:
        value = source.get(name)
        if value is None:
            target.pop(name, None)
        else:
            target[name] = dict(value)

    @staticmethod
    def _copy_name_set(
        target: dict[str, frozenset[_SetValue]],
        source: dict[str, frozenset[_SetValue]],
        name: str,
    ) -> None:
        value = source.get(name)
        if value is None:
            target.pop(name, None)
        else:
            target[name] = value

    def _copy_guaranteed_target_state(
        self,
        scope: _ElementScope,
        target: ast.AST,
        sources: Sequence[_ElementScope],
    ) -> None:
        if not sources:
            return
        target_names = {name.id for name in _iter_target_names(target)}
        clean_sources: list[_ElementScope] = []
        for source in sources:
            clean = source.clone()
            clean.definitely_rebound -= target_names
            clean_sources.append(clean)
        target_source = clean_sources[0].clone()
        self._merge(target_source, clean_sources, collect_exits=False)
        self._copy_target_state(scope, target_source, target)
        scope.definitely_rebound.update(target_names)

    def _collect_loop_exits(self, branches: Sequence[_ElementScope]) -> None:
        if not self._escape_sinks:
            return
        for branch in branches:
            if branch.escapes_loop:
                self._append_loop_exit(branch)

    def _merge(
        self,
        scope: _ElementScope,
        branches: Sequence[_ElementScope],
        *,
        collect_exits: bool = True,
    ) -> None:
        if not branches:
            scope.reachable = False
            return
        escaping = [branch for branch in branches if branch.escapes_loop]
        if collect_exits:
            self._collect_loop_exits(escaping)
        reachable = [
            branch
            for branch in branches
            if branch.reachable and not branch.escapes_loop
        ]
        if reachable:
            merged_branches = reachable
            scope.reachable = True
            scope.escapes_loop = False
            scope.escape_kind = None
        elif escaping:
            merged_branches = escaping
            scope.reachable = False
            scope.escapes_loop = True
            kinds = {branch.escape_kind for branch in escaping}
            scope.escape_kind = kinds.pop() if len(kinds) == 1 else None
        else:
            merged_branches = list(branches)
            scope.reachable = False
            scope.escapes_loop = False
            scope.escape_kind = None

        identity_aliases = _merge_scope_scalar_map(merged_branches, "identity_aliases")
        definitely_rebound = set(merged_branches[0].definitely_rebound)
        for branch in merged_branches[1:]:
            definitely_rebound &= branch.definitely_rebound
        shadowed = _merge_scope_shadowed(merged_branches) | definitely_rebound
        shadowed -= set(identity_aliases)
        invalid_classes: set[str] = set()
        bound_names: set[str] = set()
        class_bound_names: set[str] = set()
        for branch in merged_branches:
            invalid_classes |= branch.invalid_classes
            bound_names |= branch.bound_names
            class_bound_names |= branch.class_bound_names

        scope.aliases = _merge_scope_scalar_map(merged_branches, "aliases")
        scope.class_aliases = _merge_scope_scalar_map(merged_branches, "class_aliases")
        scope.containers = _merge_scope_containers(merged_branches)
        scope.container_sources = _merge_scope_sources(merged_branches)
        scope.container_columns = _merge_scope_columns(merged_branches)
        scope.instance_aliases = _merge_scope_name_sets(
            merged_branches, "instance_aliases"
        )
        scope.instance_constructors = _merge_scope_constructors(merged_branches)
        scope.identity_aliases = identity_aliases
        scope.shadowed_identities = shadowed
        scope.definitely_rebound = definitely_rebound
        scope.invalid_classes = invalid_classes
        scope.bound_names = bound_names
        scope.class_bound_names = class_bound_names

    def _visit_conditional(
        self,
        statement: ast.If,
        scope: _ElementScope,
        class_name: str | None,
    ) -> None:
        body_scope = scope.clone()
        self.visit(statement.body, body_scope, class_name)
        else_scope = scope.clone()
        self.visit(statement.orelse, else_scope, class_name)
        self._merge(scope, [body_scope, else_scope])

    def _loop_exit_branches(
        self,
        statement: ast.For | ast.AsyncFor | ast.While,
        scope: _ElementScope,
        guaranteed_entry: bool,
        normal_body: _ElementScope | None,
        break_exits: Sequence[_ElementScope],
        continue_exits: Sequence[_ElementScope],
        class_name: str | None,
    ) -> tuple[list[_ElementScope], _ElementScope | None]:
        while_forever = (
            isinstance(statement, ast.While)
            and isinstance(statement.test, ast.Constant)
            and statement.test.value is True
        )
        if while_forever:
            return list(break_exits), None
        if statement.orelse:
            else_inputs: list[_ElementScope] = []
            if not guaranteed_entry:
                else_inputs.append(scope.clone())
            if normal_body is not None:
                else_inputs.append(normal_body)
            else_inputs.extend(continue_exits)
            if not else_inputs:
                return list(break_exits), None
            else_scope = scope.clone()
            self._merge(else_scope, else_inputs, collect_exits=False)
            self.visit(statement.orelse, else_scope, class_name)
            return [*break_exits, else_scope], else_scope
        branches = [*break_exits, *continue_exits]
        if normal_body is not None:
            branches.append(normal_body)
        if not guaranteed_entry:
            branches.append(scope.clone())
        return branches, None

    def _replay_loop_body_without_recording(
        self,
        statement: ast.For | ast.AsyncFor | ast.While,
        scope: _ElementScope,
        class_name: str | None,
    ) -> list[_ElementScope]:
        previous = self._record_enabled
        sink: list[_ElementScope] = []
        self._record_enabled = False
        self._escape_sinks.append(sink)
        try:
            self.visit(statement.body, scope, class_name)
        finally:
            self._escape_sinks.pop()
            self._record_enabled = previous
        return sink

    def _visit_loop(
        self,
        statement: ast.For | ast.AsyncFor | ast.While,
        scope: _ElementScope,
        class_name: str | None,
    ) -> None:
        guaranteed_entry = isinstance(statement, ast.For) and (
            _iterable_guaranteed_nonempty(statement.iter, scope)
        )
        exits: list[_ElementScope] = []
        self._escape_sinks.append(exits)
        body_scope = scope.clone()
        if isinstance(statement, (ast.For, ast.AsyncFor)):
            self.bind_loop(
                statement.target,
                statement.iter,
                body_scope,
                class_name,
            )
            for name in _iter_target_names(statement.target):
                body_scope.definitely_rebound.add(name.id)
        try:
            self.visit(statement.body, body_scope, class_name)
        finally:
            self._escape_sinks.pop()

        normal_body = body_scope if body_scope.reachable else None
        if body_scope.escapes_loop:
            exits.append(self._normalized_exit(body_scope))

        break_exits: list[_ElementScope] = []
        continue_exits: list[_ElementScope] = []
        for exit_scope in exits:
            if exit_scope.escape_kind == "continue":
                continue_exits.append(exit_scope)
            else:
                break_exits.append(exit_scope)

        final_iteration_only = (
            guaranteed_entry
            and not break_exits
            and not _statements_break_loop(statement.body)
        )
        final_target_scope: _ElementScope | None = None
        post_body = normal_body
        if final_iteration_only and isinstance(statement, (ast.For, ast.AsyncFor)):
            final_target_scope = scope.clone()
            self.bind_loop(
                statement.target,
                statement.iter,
                final_target_scope,
                class_name,
                final_iteration_only=True,
            )
            replay_exits = self._replay_loop_body_without_recording(
                statement, final_target_scope, class_name
            )
            if final_target_scope.escapes_loop:
                final_target_scope.reachable = True
                final_target_scope.escapes_loop = False
                final_target_scope.escape_kind = None
            post_body = final_target_scope if final_target_scope.reachable else None
            continue_exits = [
                exit_scope
                for exit_scope in replay_exits
                if exit_scope.escape_kind == "continue"
            ]

        branches, else_scope = self._loop_exit_branches(
            statement,
            scope,
            guaranteed_entry,
            post_body,
            break_exits,
            continue_exits,
            class_name,
        )

        self._merge(scope, branches)
        scope.escapes_loop = False
        scope.escape_kind = None
        if guaranteed_entry and isinstance(statement, (ast.For, ast.AsyncFor)):
            target_sources: list[_ElementScope] = list(break_exits)
            if else_scope is not None:
                target_sources.append(else_scope)
            elif post_body is not None:
                target_sources.append(post_body)
                target_sources.extend(continue_exits)
            else:
                target_sources.extend(continue_exits)
            if not target_sources:
                target_sources.append(body_scope)
            self._copy_guaranteed_target_state(scope, statement.target, target_sources)

    def _visit_with(
        self,
        statement: ast.With | ast.AsyncWith,
        scope: _ElementScope,
        class_name: str | None,
    ) -> None:
        for item in statement.items:
            if item.optional_vars is not None:
                self.bind_target(
                    item.optional_vars, item.context_expr, scope, class_name
                )
                _mark_definite_rebinding(item.optional_vars, scope)
        self.visit(statement.body, scope, class_name)

    def _finalbody_escape_exits(
        self,
        statement: ast.Try | ast.TryStar,
        base_scope: _ElementScope,
        escape_states: Sequence[_ElementScope],
        class_name: str | None,
    ) -> list[_ElementScope]:
        grouped: dict[str, list[_ElementScope]] = {}
        for escape_state in escape_states:
            kind = escape_state.escape_kind or "break"
            grouped.setdefault(kind, []).append(escape_state)
        exits: list[_ElementScope] = []
        for kind, states in grouped.items():
            exit_scope = base_scope.clone()
            normalized = [self._normalized_exit(state, None) for state in states]
            self._merge(exit_scope, normalized, collect_exits=False)
            self.visit(statement.finalbody, exit_scope, class_name)
            if exit_scope.escapes_loop:
                if exit_scope.escape_kind is not None:
                    exits.append(self._normalized_exit(exit_scope))
            elif exit_scope.reachable:
                exits.append(self._normalized_exit(exit_scope, kind))
        return exits

    def _visit_try(
        self,
        statement: ast.Try | ast.TryStar,
        scope: _ElementScope,
        class_name: str | None,
    ) -> None:
        base_scope = scope.clone()
        sink: list[_ElementScope] = []
        self._escape_sinks.append(sink)
        pending_exits: list[_ElementScope] = []
        try:
            branches: list[_ElementScope] = []
            body_scope = scope.clone()
            self.visit(statement.body, body_scope, class_name)
            branches.append(body_scope)
            for handler in statement.handlers:
                handler_scope = scope.clone()
                if handler.name is not None:
                    self.store(handler.name, None, None, handler_scope)
                    handler_scope.definitely_rebound.add(handler.name)
                self.visit(handler.body, handler_scope, class_name)
                if handler.name is not None:
                    self.store(handler.name, None, None, handler_scope)
                branches.append(handler_scope)
            if body_scope.reachable:
                else_scope = body_scope.clone()
                self.visit(statement.orelse, else_scope, class_name)
                branches.append(else_scope)

            sink_snapshot = len(sink)
            normal_branches = [
                branch
                for branch in branches
                if branch.reachable
                and not branch.escapes_loop
                and (branch is not body_scope or not statement.orelse)
            ]
            escape_branches = [branch for branch in branches if branch.escapes_loop]
            pre_escape_states = list(sink[:sink_snapshot])
            for branch in escape_branches:
                if branch.escape_kind is not None:
                    pre_escape_states.append(self._normalized_exit(branch))
                elif not pre_escape_states:
                    pre_escape_states.append(self._normalized_exit(branch, "break"))

            if normal_branches:
                self._merge(scope, normal_branches, collect_exits=False)
            else:
                scope.reachable = False
                scope.escapes_loop = False
                scope.escape_kind = None
            continuing = bool(normal_branches)

            final_scope = base_scope.clone()
            final_inputs = [self._normalized_exit(branch, None) for branch in branches]
            final_inputs.extend(
                self._normalized_exit(state, None) for state in pre_escape_states
            )
            self._merge(final_scope, final_inputs, collect_exits=False)

            if continuing:
                continuation = base_scope.clone()
                self._merge(continuation, normal_branches, collect_exits=False)
                self.visit(statement.finalbody, continuation, class_name)
                _copy_scope_state(scope, continuation)
                if continuation.escapes_loop and continuation.escape_kind is not None:
                    pending_exits.append(self._normalized_exit(continuation))

            pending_exits.extend(
                self._finalbody_escape_exits(
                    statement, base_scope, pre_escape_states, class_name
                )
            )

            # Refresh finalbody facts from every possible entry state.
            self.visit(statement.finalbody, final_scope, class_name)
            pending_exits.extend(
                self._normalized_exit(state) for state in sink[sink_snapshot:]
            )
        finally:
            self._escape_sinks.pop()
        for escape_state in pending_exits:
            self._append_loop_exit(escape_state)

    def _visit_match(
        self, statement: ast.Match, scope: _ElementScope, class_name: str | None
    ) -> None:
        branches: list[_ElementScope] = []
        for case in statement.cases:
            case_scope = scope.clone()
            for name in _pattern_capture_names(case.pattern):
                self.store(name, None, None, case_scope)
                case_scope.definitely_rebound.add(name)
            self.visit(case.body, case_scope, class_name)
            branches.append(case_scope)
        if not any(
            _pattern_is_irrefutable(case.pattern) and case.guard is None
            for case in statement.cases
        ):
            branches.append(scope.clone())
        self._merge(scope, branches)

    def _class_scope(self, name: str) -> _ElementScope:
        attrs = self.attributes.get(name, {})
        containers: dict[str, dict[str, str]] = {}
        for key, value in attrs.items():
            if key.startswith(("self.", "cls.")) or "[" not in key:
                continue
            prefix, inner = key.split("[", 1)
            if inner.endswith("]"):
                containers.setdefault(prefix, {})[inner[:-1]] = value
        return _ElementScope(
            containers=containers,
            container_sources={name: "class" for name in containers},
            is_class_scope=True,
        )

    def _visit_branches(
        self, statement: ast.stmt, scope: _ElementScope, class_name: str | None
    ) -> None:
        if isinstance(statement, ast.If):
            self._visit_conditional(statement, scope, class_name)
        elif isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
            self._visit_loop(statement, scope, class_name)
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            self._visit_with(statement, scope, class_name)
        elif isinstance(statement, (ast.Try, ast.TryStar)):
            self._visit_try(statement, scope, class_name)
        elif isinstance(statement, ast.Match):
            self._visit_match(statement, scope, class_name)
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parameters = set(_function_parameter_names(statement))
            self.store(statement.name, None, None, scope)
            if _identity_function_parameter(statement) is not None:
                scope.shadowed_identities.discard(statement.name)
            inner = _ElementScope(
                aliases=dict(scope.aliases),
                class_aliases=dict(scope.class_aliases),
                invalid_classes=set(parameters),
                instance_aliases=dict(scope.instance_aliases),
                instance_constructors=dict(scope.instance_constructors),
                shadowed_identities=set(scope.shadowed_identities),
                identity_aliases=dict(scope.identity_aliases),
                bound_names=set(scope.bound_names),
            )
            for parameter in parameters:
                self.store(parameter, None, None, inner)
            for name, owner in self.parameter_types.get(id(statement), {}).items():
                inner.instance_aliases[name] = frozenset({owner})
            self.visit(statement.body, inner, class_name)
        elif isinstance(statement, ast.ClassDef):
            scope.aliases.pop(statement.name, None)
            scope.class_aliases.pop(statement.name, None)
            scope.instance_aliases.pop(statement.name, None)
            scope.instance_constructors.pop(statement.name, None)
            scope.identity_aliases.pop(statement.name, None)
            if statement.name in self.identity_functions:
                scope.shadowed_identities.add(statement.name)
            class_scope = self._class_scope(statement.name)
            class_scope.aliases = dict(scope.aliases)
            class_scope.class_aliases = dict(scope.class_aliases)
            class_scope.instance_aliases = dict(scope.instance_aliases)
            class_scope.instance_constructors = dict(scope.instance_constructors)
            class_scope.identity_aliases = dict(scope.identity_aliases)
            class_scope.shadowed_identities = set(scope.shadowed_identities)
            class_scope.bound_names = set(scope.bound_names)
            self.visit(statement.body, class_scope, statement.name)
            self._bind_scope_name(scope, statement.name)

    def visit(
        self,
        statements: Sequence[ast.stmt],
        scope: _ElementScope,
        class_name: str | None,
    ) -> None:
        for statement in statements:
            if not scope.reachable:
                break
            self.record(statement, scope, class_name)
            if self._apply_statement(statement, scope, class_name):
                if isinstance(statement, (ast.Return, ast.Raise)):
                    scope.reachable = False
                elif isinstance(statement, (ast.Break, ast.Continue)):
                    scope.reachable = False
                    scope.escapes_loop = True
                    scope.escape_kind = (
                        "break" if isinstance(statement, ast.Break) else "continue"
                    )
                continue
            self._visit_branches(statement, scope, class_name)
            if isinstance(statement, (ast.Return, ast.Raise)):
                scope.reachable = False
                scope.escape_kind = None
            elif isinstance(statement, (ast.Break, ast.Continue)):
                scope.reachable = False
                scope.escapes_loop = True
                scope.escape_kind = (
                    "break" if isinstance(statement, ast.Break) else "continue"
                )


def _element_alias_analysis(
    tree: ast.Module,
    attributes: dict[str, dict[str, str]],
    contexts: dict[int, str | None],
    field_types: dict[str, dict[str, str]] | None = None,
    path_fields: dict[str, set[str]] | None = None,
    parameter_types: dict[int, dict[str, str]] | None = None,
    init_methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None = None,
    extra_identity_functions: dict[str, str] | None = None,
) -> _ElementAnalysis:
    identities = dict(extra_identity_functions or {})
    identities.update(_identity_functions(tree))
    analysis = _ElementAnalysis(
        attributes,
        contexts,
        identities,
        field_types,
        path_fields,
        parameter_types,
        init_methods,
    )
    analysis.run(tree)
    return analysis


def _subscript_candidates(
    subscript: ast.Subscript,
    attributes: dict[str, dict[str, str]],
    contexts: dict[int, str | None],
    container_nodes: dict[int, dict[str, str]],
    class_alias_nodes: dict[int, str],
    invalid_classes: frozenset[str],
    include_aliases: bool = True,
) -> set[str] | None:
    base = subscript.value
    key = _literal_subscript_key(subscript)
    if include_aliases and isinstance(base, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
        elements: dict[str, str] = {}
        for element_key, element in _class_container_elements(base):
            resolved = _resolved_reference(element, {})
            if resolved is not None:
                elements[element_key] = resolved
        return _container_map_candidates(elements, key)
    if isinstance(base, ast.Name):
        if not include_aliases or base.id in invalid_classes:
            return None
        stored_elements = container_nodes.get(id(base))
        if stored_elements is None:
            return None
        return _container_map_candidates(stored_elements, key)
    if not isinstance(base, ast.Attribute) or not isinstance(base.value, ast.Name):
        return None
    root = base.value
    if root.id in {"self", "cls"}:
        class_name = contexts.get(id(subscript))
        if class_name is None:
            return None
        class_attributes = attributes.get(class_name, {})
        candidates = _class_container_candidates(
            class_attributes, f"{root.id}.{base.attr}", key
        )
        if not candidates:
            candidates = _class_container_candidates(class_attributes, base.attr, key)
        return candidates
    if root.id in invalid_classes:
        return None
    class_name = class_alias_nodes.get(id(root), root.id)
    resolved_attributes = attributes.get(class_name)
    if resolved_attributes is None:
        return None
    return _class_container_candidates(resolved_attributes, base.attr, key)


def _namespace_subscript_reference(
    node: ast.Attribute,
    attributes: dict[str, dict[str, str]],
    contexts: dict[int, str | None],
    container_nodes: dict[int, dict[str, str]],
    class_alias_nodes: dict[int, str],
    invalid_classes: frozenset[str],
    check: Callable[[ast.AST, str | None], None],
) -> bool:
    value = node.value
    if not isinstance(value, ast.Subscript):
        return False
    candidates = _subscript_candidates(
        value,
        attributes,
        contexts,
        container_nodes,
        class_alias_nodes,
        invalid_classes,
    )
    if not candidates:
        return False
    for candidate in candidates:
        if _is_forbidden_runtime_name(candidate) or _is_dynamic_import_name(candidate):
            check(node, candidate)
            return True
        combined = f"{candidate}.{node.attr}"
        if _is_forbidden_runtime_name(combined) or _is_dynamic_import_name(combined):
            check(node, combined)
            return True
    return True


def _function_parameters_for_nodes(tree: ast.Module) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}

    def visit(node: ast.AST, parameters: set[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                child_parameters = _function_parameter_names(child)
                for subnode in ast.walk(child):
                    result[id(subnode)] = child_parameters
                visit(child, child_parameters)
            else:
                visit(child, parameters)

    visit(tree, set())
    return result


def _class_alias_map(tree: ast.Module) -> dict[str, str]:
    classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value = node.value
            targets = [node.target]
        else:
            continue
        if not isinstance(value, ast.Name):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and (
                value.id in classes or value.id in aliases
            ):
                aliases[target.id] = value.id
    for _ in range(len(aliases)):
        changed = False
        for alias_name, alias_target in list(aliases.items()):
            resolved = aliases.get(alias_target)
            if (
                resolved is not None
                and resolved != alias_target
                and aliases[alias_name] != resolved
            ):
                aliases[alias_name] = resolved
                changed = True
        if not changed:
            break
    return {
        alias: target
        for alias, target in aliases.items()
        if target in classes and alias != target
    }


def _resolve_class_reference(
    base_node: ast.Name | None,
    class_name: str | None,
    class_aliases: dict[str, str],
    class_alias_nodes: dict[int, str],
) -> str | None:
    if base_node is not None:
        resolved = class_alias_nodes.get(id(base_node))
        if resolved is not None:
            return resolved
    if class_name is None:
        return None
    return class_aliases.get(class_name, class_name)


def _call_return_target(
    function: ast.AST,
    aliases: dict[str, str],
) -> tuple[str | None, str | None, ast.Name | None]:
    method: str | None = None
    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Call):
        constructor = function.value.func
        class_name = _resolve_callable_name(constructor, aliases)
        method = function.attr
        base_node = constructor if isinstance(constructor, ast.Name) else None
        return class_name, method, base_node
    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
        class_name = _resolve_callable_name(function.value, aliases)
        if class_name is None:
            class_name = function.value.id
        return class_name, function.attr, function.value
    if isinstance(function, ast.Name):
        return _resolve_callable_name(function, aliases), None, function
    return None, None, None


def _class_method_nodes(
    tree: ast.Module,
    external_methods: (
        dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] | None
    ) = None,
    aliases: dict[str, str] | None = None,
) -> dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    result: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {
        name: dict(methods) for name, methods in (external_methods or {}).items()
    }
    result.update(
        {
            class_node.name: {
                statement.name: statement
                for statement in class_node.body
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for class_node in classes
        }
    )
    bases = _class_bases(
        classes,
        aliases or _class_alias_map(tree),
        set(external_methods or ()),
    )
    for _ in range(len(classes) + 1):
        changed = False
        for class_node in classes:
            methods = result[class_node.name]
            for base in bases.get(class_node.name, []):
                for method_name, method in result.get(base, {}).items():
                    if method_name not in methods:
                        methods[method_name] = method
                        changed = True
        if not changed:
            break
    return result


def _class_init_methods(
    tree: ast.Module,
    external_methods: (
        dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] | None
    ) = None,
    aliases: dict[str, str] | None = None,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    result: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for class_name, methods in _class_method_nodes(
        tree, external_methods, aliases
    ).items():
        init = methods.get("__init__")
        if init is not None:
            result[class_name] = init
    return result


def _method_parameter(
    method: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> ast.arg | None:
    for parameter in _function_parameters(method):
        if parameter.arg == name:
            return parameter
    return None


def _resolve_argument_reference(
    expression: ast.AST, aliases: dict[str, str]
) -> str | None:
    if isinstance(expression, ast.Call):
        name = _resolve_callable_name(expression.func, aliases)
        if name in PATH_CLASS_NAMES:
            return "Path"
    return _resolved_reference(expression, aliases)


def _method_default_expression(
    method: ast.FunctionDef | ast.AsyncFunctionDef, name: str
) -> ast.AST | None:
    positional = [*method.args.posonlyargs, *method.args.args]
    defaults = method.args.defaults
    default_start = len(positional) - len(defaults)
    for index, parameter in enumerate(positional):
        if parameter.arg != name:
            continue
        default_index = index - default_start
        if 0 <= default_index < len(defaults):
            return defaults[default_index]
        return None
    for parameter, default in zip(method.args.kwonlyargs, method.args.kw_defaults):
        if parameter.arg == name and default is not None:
            return default
    return None


def _resolve_method_parameter(
    call: ast.Call,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
    aliases: dict[str, str],
) -> str | None:
    positional = [*method.args.posonlyargs, *method.args.args]
    offset = 0
    if positional and positional[0].arg in {"self", "cls"}:
        is_static = any(
            (
                (_resolve_callable_name(decorator, {}) or "").rsplit(".", 1)[-1]
                == "staticmethod"
            )
            for decorator in method.decorator_list
        )
        if not is_static:
            offset = 1
    expression: ast.AST | None = None
    for parameter_index, positional_parameter in enumerate(positional):
        if positional_parameter.arg != name:
            continue
        call_index = parameter_index - offset
        if 0 <= call_index < len(call.args):
            argument = call.args[call_index]
            if not isinstance(argument, ast.Starred):
                expression = argument
        break
    if expression is None:
        for keyword in call.keywords:
            if keyword.arg == name:
                expression = keyword.value
                break
    if expression is None:
        expression = _method_default_expression(method, name)
    if expression is not None:
        resolved = _resolve_argument_reference(expression, aliases)
        if resolved is not None:
            return resolved
    parameter_node = _method_parameter(method, name)
    if (
        parameter_node is not None
        and parameter_node.annotation is not None
        and _annotation_contains_path(parameter_node.annotation, aliases)
    ):
        return "Path"
    return None


def _receiver_constructor_call(call: ast.Call) -> ast.Call | None:
    function = call.func
    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Call):
        return function.value
    return None


def _resolve_return_candidates(
    candidate: str,
    call: ast.Call,
    method: ast.FunctionDef | ast.AsyncFunctionDef | None,
    aliases: dict[str, str],
    class_name: str | None = None,
    methods: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] | None = None,
    instance_constructors: (dict[int, frozenset[_ConstructorBinding]] | None) = None,
) -> frozenset[str]:
    if candidate.startswith(INIT_PARAMETER_PREFIX):
        init_method = (methods or {}).get(class_name or "", {}).get("__init__")
        if init_method is None:
            return frozenset()
        parameters = [
            parameter.arg
            for parameter in _function_parameters(init_method)
            if parameter.arg not in {"self", "cls"}
        ]
        specification = candidate[len(INIT_PARAMETER_PREFIX) :]
        name, _, suffix = specification.partition(".")
        if name not in parameters:
            return frozenset()
        index = parameters.index(name)
        values: set[str] = set()
        constructor = _receiver_constructor_call(call)
        if constructor is not None:
            resolved = _resolve_method_parameter(
                constructor, init_method, name, aliases
            )
            if resolved is not None:
                values.add(resolved)
        elif isinstance(call.func, ast.Attribute) and isinstance(
            call.func.value, ast.Name
        ):
            bindings = (instance_constructors or {}).get(
                id(call.func.value), frozenset()
            )
            for binding in bindings:
                if binding.owner not in {None, class_name}:
                    continue
                resolved_value = (
                    binding.parameter_values[index]
                    if index < len(binding.parameter_values)
                    else _resolve_method_parameter(
                        binding.constructor, init_method, name, aliases
                    )
                )
                if resolved_value is not None:
                    values.add(resolved_value)
        return frozenset(f"{value}.{suffix}" if suffix else value for value in values)
    if not candidate.startswith(CALL_PARAMETER_PREFIX):
        return frozenset({candidate})
    if method is None:
        return frozenset()
    specification = candidate[len(CALL_PARAMETER_PREFIX) :]
    name, _, suffix = specification.partition(".")
    resolved = _resolve_method_parameter(call, method, name, aliases)
    if resolved is None:
        return frozenset()
    return frozenset({f"{resolved}.{suffix}" if suffix else resolved})


def _return_class_names(
    class_name: str | None,
    base_node: ast.Name | None,
    instance_constructors: dict[int, frozenset[_ConstructorBinding]],
) -> set[str]:
    names: set[str] = set()
    if class_name is not None:
        names.add(class_name)
        names.add(class_name.rsplit(".", 1)[-1])
    if base_node is not None:
        for binding in instance_constructors.get(id(base_node), frozenset()):
            if binding.owner is not None:
                names.add(binding.owner)
    return names


def _namespace_call_return_reference(
    node: ast.Attribute,
    returns: dict[str, dict[str, str]],
    aliases: dict[str, str],
    class_aliases: dict[str, str],
    parameters: set[str],
    class_alias_nodes: dict[int, str],
    invalid_classes: frozenset[str],
    contexts: dict[int, str | None],
    methods: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]],
    instance_constructors: dict[int, frozenset[_ConstructorBinding]],
    check: Callable[[ast.AST, str | None], None],
) -> None:
    if not isinstance(node.value, ast.Call):
        return
    class_name, method, base_node = _call_return_target(node.value.func, aliases)
    if (
        base_node is not None
        and base_node.id not in {"self", "cls"}
        and base_node.id in invalid_classes
    ):
        return
    if class_name in {"self", "cls"}:
        class_name = contexts.get(id(node))
    class_name = _resolve_class_reference(
        base_node, class_name, class_aliases, class_alias_nodes
    )
    for possible_class in _return_class_names(
        class_name, base_node, instance_constructors
    ):
        if possible_class not in returns or possible_class in parameters:
            continue
        method_node = methods.get(possible_class, {}).get(method) if method else None
        base = returns[possible_class].get(method) if method is not None else None
        suffix = f".{node.attr}" if base is not None else ""
        if base is None:
            method_node = methods.get(possible_class, {}).get(node.attr)
            base = returns[possible_class].get(node.attr)
        for candidate in _return_candidates(base):
            for resolved in _resolve_return_candidates(
                candidate,
                node.value,
                method_node,
                aliases,
                possible_class,
                methods,
                instance_constructors,
            ):
                check(node, f"{resolved}{suffix}")


def _namespace_call_chain_reference(
    node: ast.Call,
    returns: dict[str, dict[str, str]],
    aliases: dict[str, str],
    class_aliases: dict[str, str],
    parameters: set[str],
    class_alias_nodes: dict[int, str],
    invalid_classes: frozenset[str],
    contexts: dict[int, str | None],
    methods: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]],
    instance_constructors: dict[int, frozenset[_ConstructorBinding]],
    check: Callable[[ast.AST, str | None], None],
) -> None:
    inner = node.func
    if not isinstance(inner, ast.Call):
        return
    class_name, method, base_node = _call_return_target(inner.func, aliases)
    if (
        base_node is not None
        and base_node.id not in {"self", "cls"}
        and base_node.id in invalid_classes
    ):
        return
    if class_name in {"self", "cls"}:
        class_name = contexts.get(id(node))
    class_name = _resolve_class_reference(
        base_node, class_name, class_aliases, class_alias_nodes
    )
    if method is None:
        return
    for possible_class in _return_class_names(
        class_name, base_node, instance_constructors
    ):
        if possible_class not in returns or possible_class in parameters:
            continue
        method_node = methods.get(possible_class, {}).get(method)
        for candidate in _return_candidates(returns[possible_class].get(method)):
            for resolved in _resolve_return_candidates(
                candidate,
                inner,
                method_node,
                aliases,
                possible_class,
                methods,
                instance_constructors,
            ):
                check(node, resolved)


def _namespace_attribute_reference(
    node: ast.Attribute,
    attributes: dict[str, dict[str, str]],
    returns: dict[str, dict[str, str]],
    contexts: dict[int, str | None],
    aliases: dict[str, str],
    element_aliases: dict[int, str],
    container_nodes: dict[int, dict[str, str]],
    class_aliases: dict[str, str],
    class_alias_nodes: dict[int, str],
    invalid_classes: frozenset[str],
    parameters: set[str],
    methods: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]],
    instance_constructors: dict[int, frozenset[_ConstructorBinding]],
    check: Callable[[ast.AST, str | None], None],
) -> None:
    element_alias = element_aliases.get(id(node.value))
    if element_alias is not None:
        check(node, f"{element_alias}.{node.attr}")
        return
    if _namespace_subscript_reference(
        node,
        attributes,
        contexts,
        container_nodes,
        class_alias_nodes,
        invalid_classes,
        check,
    ):
        return
    chain = _class_chain_reference(node, attributes, class_alias_nodes, invalid_classes)
    if chain is not None:
        check(node, chain)
        return
    if isinstance(node.value, ast.Name):
        name = node.value.id
        if name in {"self", "cls"}:
            class_name = contexts.get(id(node))
            if class_name is not None:
                check(node, attributes.get(class_name, {}).get(node.attr))
            return
        if name in invalid_classes:
            return
        class_name = class_alias_nodes.get(id(node.value), name)
        if class_name in attributes:
            check(node, attributes[class_name].get(node.attr))
        return
    _namespace_call_return_reference(
        node,
        returns,
        aliases,
        class_aliases,
        parameters,
        class_alias_nodes,
        invalid_classes,
        contexts,
        methods,
        instance_constructors,
        check,
    )


def _namespace_reference_violations(
    tree: ast.Module,
    states: dict[int, dict[str, str]],
    shadowed: dict[int, set[str]],
    value_alias_names: set[str],
    static: dict[str, str],
    report: Callable[[ast.AST, str], Violation | None],
    field_types: dict[str, dict[str, str]] | None = None,
    path_fields: dict[str, set[str]] | None = None,
    parameter_types: dict[int, dict[str, str]] | None = None,
    init_methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None = None,
    imported_attributes: dict[str, dict[str, str]] | None = None,
    imported_methods: (
        dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] | None
    ) = None,
    imported_init_methods: (
        dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None
    ) = None,
    imported_returns: dict[str, dict[str, str]] | None = None,
    extra_identity_functions: dict[str, str] | None = None,
) -> list[Violation]:
    attributes, returns = _class_namespace_facts(
        tree, static, imported_attributes, imported_returns
    )
    contexts = _class_contexts(tree)
    effective_init_methods = dict(imported_init_methods or {})
    effective_init_methods.update(_class_init_methods(tree, imported_methods, static))
    effective_init_methods.update(init_methods or {})
    analysis = _element_alias_analysis(
        tree,
        attributes,
        contexts,
        field_types,
        path_fields,
        parameter_types,
        effective_init_methods,
        extra_identity_functions,
    )
    receiver_aliases = dict(analysis.class_alias_nodes)
    receiver_aliases.update(analysis.instance_alias_nodes)
    class_aliases = _class_alias_map(tree)
    parameters = _function_parameters_for_nodes(tree)
    class_methods = _class_method_nodes(tree, imported_methods, static)
    violations: list[Violation] = []

    def check(node: ast.AST, resolved: str | None) -> None:
        if resolved is None:
            return
        normalised = _normalise_builtins_name(resolved)
        aliases = states.get(id(node), static)
        root = normalised.split(".", 1)[0]
        if _is_shadowed_runtime_name(
            root, shadowed.get(id(node), set()), value_alias_names, aliases
        ):
            return
        violation = report(node, normalised)
        if violation is not None:
            violations.append(violation)

    for node in ast.walk(tree):
        if id(node) not in states:
            continue
        aliases = states.get(id(node), static)
        invalid_classes = analysis.invalid_nodes.get(id(node), frozenset())
        _namespace_definition_references(node, aliases, check)
        if isinstance(node, ast.Attribute):
            _namespace_attribute_reference(
                node,
                attributes,
                returns,
                contexts,
                aliases,
                analysis.element_aliases,
                analysis.container_nodes,
                class_aliases,
                receiver_aliases,
                invalid_classes,
                parameters.get(id(node), set()),
                class_methods,
                analysis.instance_constructor_nodes,
                check,
            )
        elif isinstance(node, ast.Subscript):
            if isinstance(node.ctx, ast.Store):
                continue
            check(
                node,
                _class_subscript_reference(
                    node,
                    attributes,
                    contexts,
                    analysis.container_nodes,
                    receiver_aliases,
                    invalid_classes,
                ),
            )
        elif isinstance(node, ast.Call):
            check(
                node,
                _class_container_alias_reference(node.func, analysis.container_nodes),
            )
            check(
                node,
                _class_mapping_values_reference(
                    node,
                    attributes,
                    contexts,
                    receiver_aliases,
                    invalid_classes,
                ),
            )
            check(
                node,
                _class_container_get_reference(
                    node,
                    attributes,
                    contexts,
                    analysis.container_nodes,
                    receiver_aliases,
                    invalid_classes,
                ),
            )
            _namespace_call_chain_reference(
                node,
                returns,
                aliases,
                class_aliases,
                parameters.get(id(node), set()),
                receiver_aliases,
                invalid_classes,
                contexts,
                class_methods,
                analysis.instance_constructor_nodes,
                check,
            )
    return violations


def _skip_shadowed_path_attribute(
    node: ast.Attribute,
    shadow: set[str],
    value_alias_names: set[str],
    aliases: dict[str, str],
    instance_alias_nodes: dict[int, str] | None,
) -> bool:
    root_node = _leftmost_name(node)
    if root_node is not None and (instance_alias_nodes or {}).get(id(root_node)):
        return False
    root_name = _root_name(node)
    return root_name not in {"self", "cls"} and _is_shadowed_runtime_name(
        root_name, shadow, value_alias_names, aliases
    )


def _instance_path_attribute(
    node: ast.Attribute,
    instance_alias_nodes: dict[int, str] | None,
    class_contexts: dict[int, str | None],
    instance_path_fields: dict[str, set[str]] | None,
) -> str | None:
    if not isinstance(node.value, ast.Attribute):
        return None
    dotted = _dotted_name(node.value)
    root = _leftmost_name(node.value)
    if dotted is None or root is None:
        return None
    prefix = f"{root.id}."
    if not dotted.startswith(prefix):
        return None
    relative = dotted[len(prefix) :]
    value_class = (instance_alias_nodes or {}).get(id(root))
    if value_class is None and root.id in {"self", "cls"}:
        value_class = (
            class_contexts.get(id(node.value))
            or class_contexts.get(id(root))
            or class_contexts.get(id(node))
        )
    if value_class is None:
        return None
    if relative in (instance_path_fields or {}).get(value_class, set()):
        return f"Path.{node.attr}"
    return None


def _runtime_attribute_name(
    node: ast.Attribute,
    aliases: dict[str, str],
    path_functions: set[str],
    self_path_attributes: dict[str, set[str]],
    class_contexts: dict[int, str | None],
    instance_alias_nodes: dict[int, str] | None,
    instance_path_fields: dict[str, set[str]] | None = None,
) -> str | None:
    if isinstance(node.value, ast.NamedExpr):
        base_name = _resolve_callable_name(node.value.value, aliases)
        attribute_name = f"{base_name}.{node.attr}" if base_name else None
    else:
        attribute_name = _resolve_callable_name(node, aliases)
    if attribute_name is None and isinstance(node.value, ast.Call):
        base_name = _resolve_callable_name(node.value.func, aliases)
        if base_name in PATH_CLASS_NAMES or base_name in path_functions:
            attribute_name = f"Path.{node.attr}"
    class_name = class_contexts.get(id(node))
    if (
        isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id in {"self", "cls"}
        and class_name is not None
        and f"{node.value.value.id}.{node.value.attr}"
        in self_path_attributes.get(class_name, set())
    ):
        attribute_name = f"Path.{node.attr}"
    instance_attribute = _instance_path_attribute(
        node, instance_alias_nodes, class_contexts, instance_path_fields
    )
    if instance_attribute is not None:
        attribute_name = instance_attribute
    return attribute_name


def _runtime_attribute_violations(
    tree: ast.Module,
    states: dict[int, dict[str, str]],
    shadowed: dict[int, set[str]],
    value_alias_names: set[str],
    imported: set[str],
    path_functions: set[str],
    self_path_attributes: dict[str, set[str]],
    class_contexts: dict[int, str | None],
    path: Path,
    instance_alias_nodes: dict[int, str] | None = None,
    instance_path_fields: dict[str, set[str]] | None = None,
) -> list[Violation]:
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if id(node) not in states:
            continue
        aliases = states.get(id(node), {})
        shadow = shadowed.get(id(node), set())
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(
            node.ctx, ast.Store
        ):
            continue
        if isinstance(node, ast.Name):
            if node.id not in imported:
                continue
            if _is_shadowed_runtime_name(node.id, shadow, value_alias_names, aliases):
                continue
            name = aliases.get(node.id, node.id)
            if name in RUNTIME_ATTRIBUTES or name in FORBIDDEN_INDIRECTION_NAMES:
                violations.append(_runtime_violation(path, node.lineno, name))
            continue
        if isinstance(node, ast.Attribute):
            if _skip_shadowed_path_attribute(
                node, shadow, value_alias_names, aliases, instance_alias_nodes
            ):
                continue
            attribute_name = _runtime_attribute_name(
                node,
                aliases,
                path_functions,
                self_path_attributes,
                class_contexts,
                instance_alias_nodes,
                instance_path_fields,
            )
            if attribute_name is not None and _is_path_method_name(attribute_name):
                attribute_name = _normalise_path_name(attribute_name)
            if attribute_name in RUNTIME_ATTRIBUTES or (
                attribute_name is not None and _is_path_method_name(attribute_name)
            ):
                violations.append(_runtime_violation(path, node.lineno, attribute_name))
    return violations


def _runtime_reference_report(
    path: Path,
) -> Callable[[ast.AST, str], Violation | None]:
    def report(node: ast.AST, name: str) -> Violation | None:
        if not _is_forbidden_runtime_name(name):
            return None
        return _runtime_violation(path, getattr(node, "lineno", 1), name)

    return report


def _dynamic_reference_report(
    path: Path,
) -> Callable[[ast.AST, str], Violation | None]:
    def report(node: ast.AST, name: str) -> Violation | None:
        dynamic = frozenset({name}) if _is_dynamic_import_name(name) else frozenset()
        if not dynamic:
            return None
        return Violation(
            "dynamic-import",
            path,
            getattr(node, "lineno", 1),
            "dynamic execution through "
            f"{sorted(dynamic)!r} is forbidden in the contract; use a static import",
        )

    return report


def _logging_filename_violation(
    node: ast.Call, aliases: FlowAliases, path: Path
) -> Violation | None:
    if not isinstance(node.func, ast.Attribute):
        return None
    if not any(keyword.arg == "filename" for keyword in node.keywords):
        return None
    if "logging.basicConfig" not in _resolve_possible_names(node.func, aliases):
        return None
    return _runtime_violation(path, node.lineno, "logging.basicConfig")


def _runtime_call_access_violations(
    tree: ast.Module,
    initial: FlowAliases,
    final: FlowAliases,
    module_paths: set[str],
    static: dict[str, str],
    path: Path,
    reachable: set[int],
) -> list[Violation]:
    violations: list[Violation] = []
    for node, aliases, path_names in _iter_calls_with_aliases(
        tree.body, dict(initial), module_paths, final
    ):
        if id(node) not in reachable:
            continue
        call_static = _merge_flow_aliases(static, aliases)
        literal_targets = _literal_indirect_names(
            node, aliases, path_names, call_static
        ) | _literal_indirect_names(node.func, aliases, path_names, call_static)
        forbidden_literal = {
            name for name in literal_targets if _is_forbidden_runtime_name(name)
        }
        if forbidden_literal:
            violations.append(
                _runtime_violation(path, node.lineno, min(forbidden_literal))
            )
            continue
        if isinstance(node.func, ast.Attribute) and _is_path_expression(
            node.func.value, call_static, path_names
        ):
            if node.func.attr not in PATH_PURE_METHODS:
                violations.append(
                    _runtime_violation(path, node.lineno, f"Path.{node.func.attr}")
                )
            continue
        logging_violation = _logging_filename_violation(node, aliases, path)
        if logging_violation is not None:
            violations.append(logging_violation)
            continue
        value_forbidden: set[str] = set()
        value_line = node.lineno
        arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
        for argument in arguments:
            for reference, reference_names in _value_references(
                argument, aliases, path_names, call_static
            ):
                forbidden_values = {
                    name for name in reference_names if _is_forbidden_runtime_name(name)
                }
                if forbidden_values:
                    value_forbidden = forbidden_values
                    value_line = reference.lineno
                    break
            if value_forbidden:
                break
        if value_forbidden:
            violations.append(
                _runtime_violation(path, value_line, min(value_forbidden))
            )
        possible = _resolve_possible_names(node.func, aliases)
        forbidden = {name for name in possible if _is_forbidden_runtime_name(name)}
        if forbidden:
            violations.append(_runtime_violation(path, node.lineno, min(forbidden)))
    return violations


def scan_contract_runtime_access(
    contract_src: Path = CONTRACT_SRC,
) -> list[Violation]:
    """Reject environment, filesystem and process access in the contract."""
    violations: list[Violation] = []
    trees = {path: _parse(path) for path in python_files(contract_src)}
    registry = _class_field_registry(contract_src, trees)
    (
        imported_attributes_registry,
        imported_methods_registry,
        imported_init_registry,
        imported_returns_registry,
        imported_identity_registry,
    ) = _runtime_fact_registry(contract_src, trees)
    for path, tree in trees.items():
        file_imports = _file_import_aliases(tree, path, contract_src)
        base_static = dict(file_imports)
        base_static.update(_identity_static_aliases(tree))
        static, value_alias_names = _module_value_aliases(tree, base_static)
        imported_fields = _imported_class_fields(static, registry)
        imported_runtime = _imported_runtime_facts(
            static,
            imported_attributes_registry,
            imported_methods_registry,
            imported_init_registry,
            imported_returns_registry,
            imported_identity_registry,
        )
        for name in imported_runtime.identity_functions:
            marker = _identity_alias_key(name)
            static[name] = marker
            static[marker] = name
        class_fields = _class_instance_path_fields(tree, static, imported_fields)
        class_fields = _merge_class_fields(class_fields, imported_fields)
        static.update(_class_path_field_static_aliases(class_fields))
        initial = _with_function_returns(tree, _initial_flow_aliases(tree))
        initial.update(_identity_flow_aliases(tree))
        for name in imported_runtime.identity_functions:
            marker = _identity_alias_key(name)
            initial[name] = frozenset({marker})
            initial[marker] = frozenset({name})
        initial.update(_class_path_field_flow_aliases(class_fields))
        final = _module_final_aliases(tree, initial)
        bindings, plain_bindings = _call_argument_bindings(tree, initial, final)
        initial.update(bindings)
        final.update(bindings)
        module_paths: set[str] = set()
        shadowed = _shadowed_names_by_node(tree)
        states = _node_runtime_aliases(tree, file_imports, plain_bindings)
        contexts = _class_contexts(tree)
        attributes, _ = _class_namespace_facts(tree, static)
        for imported_name in imported_fields:
            attributes.setdefault(imported_name, {})
        for name, facts in imported_runtime.attributes.items():
            attributes.setdefault(name, facts)
        field_types = _class_field_type_map(tree, static, set(class_fields))
        parameter_types = _function_parameter_types(tree, static, set(class_fields))
        init_methods = _class_init_methods(tree)
        effective_init_methods = dict(imported_runtime.init_methods)
        effective_init_methods.update(init_methods)
        analysis = _element_alias_analysis(
            tree,
            attributes,
            contexts,
            field_types,
            class_fields,
            parameter_types,
            effective_init_methods,
            imported_runtime.identity_functions,
        )
        violations.extend(
            _literal_subscript_violations(
                tree, states, shadowed, value_alias_names, path
            )
        )
        violations.extend(
            _runtime_attribute_violations(
                tree,
                states,
                shadowed,
                value_alias_names,
                _imported_names(tree),
                _path_returning_functions(tree, static),
                _self_path_attributes_by_class(tree, static),
                contexts,
                path,
                analysis.instance_alias_nodes,
                class_fields,
            )
        )

        violations.extend(
            _namespace_reference_violations(
                tree,
                states,
                shadowed,
                value_alias_names,
                static,
                _runtime_reference_report(path),
                field_types,
                class_fields,
                parameter_types,
                init_methods,
                imported_runtime.attributes,
                imported_runtime.methods,
                imported_runtime.init_methods,
                imported_runtime.returns,
                imported_runtime.identity_functions,
            )
        )
        for lambda_node, lambda_names in _lambda_reference_names(tree, file_imports):
            if id(lambda_node) not in states:
                continue
            forbidden_lambda = {
                name for name in lambda_names if _is_forbidden_runtime_name(name)
            }
            if forbidden_lambda:
                violations.append(
                    _runtime_violation(path, lambda_node.lineno, min(forbidden_lambda))
                )
        violations.extend(
            _runtime_call_access_violations(
                tree, initial, final, module_paths, static, path, set(states)
            )
        )
    return _dedupe_violations(violations)


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


def _is_public_contract_constant(name: str) -> bool:
    """Recognise exported constant and type-alias spellings.

    Module-level ``logger``-style values are implementation details, so only
    UPPER_CASE constants and CamelCase aliases count as contract names.
    """
    return not name.startswith("_") and (name.isupper() or name[0].isupper())


def _contract_names_from_file(path: Path) -> set[str]:
    tree = _parse(path)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) and not node.name.startswith("_"):
            names.add(node.name)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_public_contract_constant(
                    target.id
                ):
                    names.add(target.id)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and _is_public_contract_constant(node.target.id)
        ):
            names.add(node.target.id)
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


_ECHO_COMMAND_RE = re.compile(r"^(?:echo|printf)\b")
_YAML_KEY_RE = re.compile(r"^-?\s*[A-Za-z_][\w-]*:\s")
_QUOTED_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'[^\']*\'')


def _python_c_imports_lint_imports(line: str) -> bool:
    match = re.search(r"-c\s+(['\"])(.*)\1", line)
    if match is None:
        return False
    try:
        payload = ast.parse(match.group(2))
    except SyntaxError:
        return False
    imported: set[str] = set()
    for node in ast.walk(payload):
        if isinstance(node, ast.ImportFrom) and node.module == "importlinter.cli":
            for alias in node.names:
                if alias.name == "lint_imports":
                    imported.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlinter.cli":
                    imported.add(alias.asname or "importlinter")
    if not imported:
        return False
    return any(
        isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in imported)
            or (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in imported
                and node.func.attr == "lint_imports"
            )
        )
        for node in ast.walk(payload)
    )


def _active_ci_commands(block: list[str]) -> list[str]:
    """Keep only executable command lines, dropping strings and YAML keys."""
    commands: list[str] = []
    for line in block:
        without_comment = line.split("#", 1)[0].rstrip()
        if not without_comment.strip():
            continue
        if _ECHO_COMMAND_RE.match(without_comment.strip()):
            continue
        if _YAML_KEY_RE.match(without_comment.strip()):
            continue
        if re.search(r"\bpython3(?:\.\d+)?\s+-c\b", without_comment):
            if _python_c_imports_lint_imports(without_comment):
                commands.append("lint_imports")
            continue
        unquoted = _QUOTED_RE.sub("", without_comment).strip()
        if unquoted:
            commands.append(unquoted)
    return commands


def _check_ci_block(
    path: Path,
    block: list[str],
    required: dict[str, str],
    label: str,
) -> list[Violation]:
    text = "\n".join(_active_ci_commands(block))
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
                        "ruff format --check src/ automl-contracts/src/": (
                            "ruff format check for both packages"
                        ),
                        "ruff format --check tools/contract_boundaries.py": (
                            "ruff format check for boundary tooling"
                        ),
                        "lint_imports": "import-linter",
                        "pip check": "pip check",
                        "--engine-wheel": "engine wheel verification",
                        "--engine-sdist": "engine sdist verification",
                    },
                    "contracts-guarantees task",
                )
            )
    github_path = GITHUB_CI_CONFIG if github_ci_path is None else github_ci_path
    if not github_path.is_file():
        violations.append(
            Violation(
                "ci-configuration",
                github_path,
                1,
                "GitHub CI configuration is missing",
            )
        )
        return violations
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
                    "ruff format --check src/ automl-contracts/src/": (
                        "ruff format check for both packages"
                    ),
                    "ruff format --check tools/contract_boundaries.py": (
                        "ruff format check for boundary tooling"
                    ),
                    "lint-imports": "import-linter",
                    "pip check": "pip check",
                    "--engine-wheel": "engine wheel verification",
                    "--engine-sdist": "engine sdist verification",
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
