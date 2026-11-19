"""Helpers for the contract API-reference and version checks.

The contract tests derive the documented public surface from
``API_REFERENCE.md`` and compare it with the package itself.  Nothing here
pins a snapshot of that surface, so a public-API change only has to keep the
reference documentation in sync.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from typing import TypeVar

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = REPO_ROOT / "automl-contracts"
CONTRACT_PYPROJECT = CONTRACT_ROOT / "pyproject.toml"
API_REFERENCE_DOC = REPO_ROOT / "API_REFERENCE.md"

#: Modules whose contents are part of the documented public import surface.
#: The root package is represented by ``__all__``.
PUBLIC_SUBMODULES = (
    "automl_contracts.definitions",
    "automl_contracts.errors",
    "automl_contracts.metrics",
    "automl_contracts.custom_metrics",
    "automl_contracts.protocols",
    "automl_contracts.search_space",
    "automl_contracts.feature_selection",
    "automl_contracts.preprocessing",
    "automl_contracts.result",
    "automl_contracts.config",
    "automl_contracts.config.general",
    "automl_contracts.config.phases",
    "automl_contracts.config.sanity_gate",
    "automl_contracts.config.complexity_guard",
    "automl_contracts.config.ensemble",
    "automl_contracts.config.algorithms",
)


def _api_reference_section(title: str) -> list[str]:
    """Return the lines of one top-level section of ``API_REFERENCE.md``."""
    lines = API_REFERENCE_DOC.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index(title)
    except ValueError as exc:
        raise AssertionError(f"API_REFERENCE.md has no {title!r} section") from exc
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("# "):
            end = index
            break
    return lines[start + 1 : end]


def parse_documented_public_modules() -> set[str]:
    """Return the public modules documented in ``API_REFERENCE.md``.

    The section "Public imports" contains one row per stable module. The root
    package is documented by the top-level import block and is always included.
    """
    modules = {"automl_contracts"}
    for line in _api_reference_section("## Public imports"):
        if not line.startswith("| `automl_contracts"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        modules.add(cells[0].strip("`"))
    return modules


def parse_documented_public_symbols() -> dict[str, str]:
    """Map documented public symbols to the module that documents them.

    Both the top-level import block and the stable-submodule table of
    ``API_REFERENCE.md`` are used. A symbol is therefore checked against the
    module the user is told to import it from.
    """
    text = API_REFERENCE_DOC.read_text(encoding="utf-8")
    start = text.index("from automl_contracts import (")
    end = text.index(")\n```", start) + 1
    import_node = ast.parse(text[start:end]).body[0]
    assert isinstance(import_node, ast.ImportFrom)
    result = {
        alias.asname or alias.name: "automl_contracts" for alias in import_node.names
    }
    for line in _api_reference_section("## Public imports"):
        if not line.startswith("| `automl_contracts"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        module = cells[0].strip("`")
        for symbol in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", cells[1]):
            result[symbol] = module
    return result


def parse_documented_engine_paths() -> set[str]:
    """Return the old engine import paths documented in ``API_REFERENCE.md``."""
    paths: set[str] = set()
    for line in _api_reference_section("## Migration from engine imports"):
        match = re.match(r"\|\s*`(configurable_automl_engine[^`]*)`", line)
        if match:
            paths.add(match.group(1))
    return paths


def _defined_public_names(module: object) -> set[str]:
    """Return the public names defined at the top level of ``module``."""
    source_path = getattr(module, "__file__", None)
    if source_path is None:
        return set()
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(
                target.id for target in targets if isinstance(target, ast.Name)
            )
    return {
        name
        for name in names
        if not name.startswith("_")
        and not isinstance(getattr(module, name, None), TypeVar)
    }


def module_public_names(module_name: str) -> set[str]:
    """Return the public names a documentation-listed submodule exposes.

    A module with ``__all__`` states its public surface explicitly. A module
    without it contributes the functions, classes and constants defined in
    that module; private helpers, imported objects and type variables are
    implementation details, not public symbols.
    """
    module = importlib.import_module(module_name)
    exported = getattr(module, "__all__", None)
    if exported is not None:
        return set(exported)
    return _defined_public_names(module)


def contract_public_names() -> set[str]:
    """Return every public name of the documented contract surface.

    The surface is the root ``__all__`` together with the exports of every
    module in ``PUBLIC_SUBMODULES``. ``__version__`` is installation metadata,
    not an API symbol.
    """
    from automl_contracts import __all__ as root_all

    names = set(root_all) - {"__version__"}
    for module_name in PUBLIC_SUBMODULES:
        names.update(module_public_names(module_name))
    return names
