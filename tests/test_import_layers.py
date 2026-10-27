"""The import layers of the package.

Imports go only downward: ``training_engine`` → ``hpo``/``tuner``/trainer →
building blocks → ``common``. The rules are the ``[tool.importlinter]``
contracts in ``pyproject.toml`` (run in CI by ``lint-imports``):

- the ``layers`` contract (top-level, function-level and ``TYPE_CHECKING``
  imports alike, no allowlist);
- ``hpo`` and the trainer modules are independent of each other;
- no import cycles between sibling modules.

These tests run the same contracts on the package and show that a
deliberately added upward import breaks them. A separate AST check keeps
package imports out of function bodies: the only one allowed is the lazy
``train_best_model`` proxy in ``training_engine/__init__.py``.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import configurable_automl_engine

PACKAGE_ROOT = Path(configurable_automl_engine.__file__).parent
PACKAGE = PACKAGE_ROOT.name
PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

#: The only function-level package imports allowed: (module path, imported
#: module).
ALLOWED_FUNCTION_IMPORTS = {
    ("training_engine/__init__.py", f"{PACKAGE}.training_engine.component"),
}


def _lint_imports(src_dir: Path) -> subprocess.CompletedProcess[str]:
    """Run ``lint-imports`` with the project contracts on ``src_dir``.

    ``src_dir`` is put first on ``PYTHONPATH``, so the package found there
    is the one analyzed.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(src_dir), env.get("PYTHONPATH", "")])
    code = (
        "import sys\n"
        "from importlinter.cli import lint_imports\n"
        f"sys.exit(lint_imports(config_filename={str(PYPROJECT)!r}, "
        "no_cache=True, no_logo=True))\n"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=src_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _package_copy(tmp_path: Path) -> Path:
    """Copy the package to ``tmp_path/src`` and return the ``src`` directory."""
    src = tmp_path / "src"
    shutil.copytree(
        PACKAGE_ROOT,
        src / PACKAGE,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return src


@pytest.fixture(scope="module")
def _importlinter() -> None:
    pytest.importorskip("importlinter")


@pytest.mark.usefixtures("_importlinter")
def test_layer_contracts_are_kept() -> None:
    """The package keeps every import contract."""
    result = _lint_imports(PACKAGE_ROOT.parent)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 broken" in result.stdout


@pytest.mark.usefixtures("_importlinter")
@pytest.mark.parametrize(
    ("module", "added_import", "broken_contract"),
    [
        # common → building block (top level).
        (
            "common/definitions.py",
            f"from {PACKAGE} import models\n",
            "Layers",
        ),
        # trainer → orchestrator (inside a function).
        (
            "trainer_state.py",
            "\n\ndef _upward():\n"
            f"    from {PACKAGE}.training_engine import config_parser\n",
            "Layers",
        ),
        # building block → engine (TYPE_CHECKING only).
        (
            "isotonic.py",
            "\nfrom typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n"
            f"    from {PACKAGE} import tuner\n",
            "Layers",
        ),
        # HPO → trainer: both are engines, but independent of each other.
        (
            "hpo/plans.py",
            f"\nfrom {PACKAGE} import trainer_data\n",
            "HPO does not import the trainer modules",
        ),
        # trainer → HPO.
        (
            "trainer_scoring.py",
            f"\nfrom {PACKAGE}.hpo import types\n",
            "The trainer modules do not import HPO",
        ),
    ],
)
def test_upward_import_breaks_contract(
    tmp_path: Path, module: str, added_import: str, broken_contract: str
) -> None:
    """A deliberately added forbidden import fails the check."""
    src = _package_copy(tmp_path)
    path = src / PACKAGE / module
    path.write_text(
        path.read_text(encoding="utf-8") + added_import + "  # noqa\n",
        encoding="utf-8",
    )
    result = _lint_imports(src)
    assert result.returncode != 0, result.stdout
    broken = [
        line for line in result.stdout.splitlines() if line.endswith(" BROKEN")
    ]
    assert any(line.startswith(broken_contract) for line in broken), result.stdout


def _function_level_imports(path: Path) -> list[tuple[int, str]]:
    """Package imports inside function bodies: ``(line, imported module)``."""
    rel_parts = path.relative_to(PACKAGE_ROOT).with_suffix("").parts
    module_parts = [PACKAGE, *rel_parts]
    if module_parts[-1] == "__init__":
        module_parts.pop()
        package_parts = module_parts
    else:
        package_parts = module_parts[:-1]

    found: list[tuple[int, str]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package_parts[: len(package_parts) - (node.level - 1)]
                    targets = [".".join([*base, *filter(None, [node.module])])]
                else:
                    targets = [node.module or ""]
            else:
                continue
            found.extend(
                (node.lineno, t)
                for t in targets
                if t == PACKAGE or t.startswith(f"{PACKAGE}.")
            )
    return found


def test_no_function_level_package_imports() -> None:
    """Package imports sit at module level, where the layer check sees them."""
    offenders = [
        f"{path.relative_to(PACKAGE_ROOT).as_posix()}:{line} imports {target}"
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        for line, target in _function_level_imports(path)
        if (path.relative_to(PACKAGE_ROOT).as_posix(), target)
        not in ALLOWED_FUNCTION_IMPORTS
    ]
    assert not offenders, offenders


def test_function_level_import_check_sees_the_proxy() -> None:
    """The AST check finds the one allowed lazy import (it is not blind)."""
    path = PACKAGE_ROOT / "training_engine" / "__init__.py"
    targets = {target for _, target in _function_level_imports(path)}
    assert f"{PACKAGE}.training_engine.component" in targets
