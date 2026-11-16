"""Smoke-check the engine installed against a built ``automl-contracts`` wheel.

Run this script from a directory outside the repository. It verifies that the
engine resolves the contract as a distribution dependency and that every
documented legacy public import path still works.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

DISTRIBUTION_NAME = "configurable-automl-engine"
CONTRACT_DISTRIBUTION = "automl-contracts"
IDENTITY_NAMES = ("AlgorithmRun", "Config", "CustomMetricSpec", "ExperimentResult")
IDENTITY_ERRORS = ("InvalidPredictionError", "TrainingError")


class EngineSmokeError(RuntimeError):
    """The installed engine does not use the installed contract correctly."""


def load_import_matrix(path: Path) -> dict[str, list[str]]:
    """Load the legacy public import matrix.

    Args:
        path: JSON file with a ``paths`` mapping.

    Returns:
        Mapping of module names to the public symbols they must expose.

    Raises:
        EngineSmokeError: The file is missing, malformed or empty.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EngineSmokeError(f"cannot read import matrix {path}: {error}") from error
    paths = document.get("paths") if isinstance(document, dict) else None
    if not isinstance(paths, dict) or not paths:
        raise EngineSmokeError(f"import matrix {path} has no paths")
    matrix: dict[str, list[str]] = {}
    for module_name, symbols in paths.items():
        if not isinstance(module_name, str) or not isinstance(symbols, list):
            raise EngineSmokeError(f"import matrix {path} has an invalid entry")
        matrix[module_name] = [str(symbol) for symbol in symbols]
    return matrix


def ensure_import_origin(package: Any, forbidden_root: Path | None, label: str) -> Path:
    """Fail when a package is imported from the repository checkout.

    Args:
        package: Imported package module.
        forbidden_root: Repository root that must not contain the package.
        label: Package name used in the error message.

    Returns:
        The resolved package location.

    Raises:
        EngineSmokeError: The package lives under ``forbidden_root``.
    """
    location = Path(package.__file__).resolve()
    if forbidden_root is not None:
        root = forbidden_root.resolve()
        if location.is_relative_to(root):
            raise EngineSmokeError(
                f"{label} was imported from {location}, not from the installed wheel"
            )
    return location


def check_import_matrix(matrix: dict[str, list[str]]) -> int:
    """Import every documented legacy path and check its symbols.

    Args:
        matrix: Mapping of module names to required symbols.

    Returns:
        The number of checked symbols.

    Raises:
        EngineSmokeError: A module cannot be imported or a symbol is missing.
    """
    checked = 0
    failures: list[str] = []
    for module_name, symbols in matrix.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as error:  # noqa: BLE001 - report every import failure
            failures.append(f"{module_name}: {type(error).__name__}: {error}")
            continue
        for symbol in symbols:
            if not hasattr(module, symbol):
                failures.append(f"{module_name}.{symbol} disappeared")
            checked += 1
    if failures:
        raise EngineSmokeError(
            "legacy public import paths are broken:\n" + "\n".join(failures)
        )
    return checked


def check_contract_dependency(expected_contract_version: str) -> dict[str, Any]:
    """Check the installed contract version and the declared engine range.

    Args:
        expected_contract_version: Contract version built into the wheel.

    Returns:
        Selected dependency metadata.

    Raises:
        EngineSmokeError: The engine does not require the contract or the
            installed version differs from the built wheel.
    """
    contracts = importlib.import_module("automl_contracts")
    requirements = importlib.metadata.requires(DISTRIBUTION_NAME) or []
    declared = [
        requirement
        for requirement in requirements
        if requirement.split(";", 1)[0].strip().startswith(CONTRACT_DISTRIBUTION)
    ]
    if not declared:
        raise EngineSmokeError(
            f"{DISTRIBUTION_NAME} does not declare a {CONTRACT_DISTRIBUTION} dependency"
        )
    installed = importlib.metadata.version(CONTRACT_DISTRIBUTION)
    if installed != expected_contract_version:
        raise EngineSmokeError(
            f"installed {CONTRACT_DISTRIBUTION} version {installed!r} does not match "
            f"the built version {expected_contract_version!r}"
        )
    if contracts.__version__ != installed:
        raise EngineSmokeError(
            f"automl_contracts.__version__ {contracts.__version__!r} differs "
            f"from distribution metadata {installed!r}"
        )
    return {"declared": declared, "installed": installed}


def check_contract_identity() -> int:
    """Check that the engine re-exports the contract classes themselves.

    Returns:
        The number of checked identities.

    Raises:
        EngineSmokeError: A re-export is missing or points to another class.
    """
    engine = importlib.import_module("configurable_automl_engine")
    contracts = importlib.import_module("automl_contracts")
    missing = [name for name in IDENTITY_NAMES if not hasattr(engine, name)]
    if missing:
        raise EngineSmokeError(f"engine root is missing contract re-exports: {missing}")
    for name in IDENTITY_NAMES:
        if getattr(engine, name) is not getattr(contracts, name):
            raise EngineSmokeError(f"engine re-export {name} is not the contract class")
    errors = importlib.import_module("configurable_automl_engine.common.errors")
    missing_errors = [name for name in IDENTITY_ERRORS if not hasattr(errors, name)]
    if missing_errors:
        raise EngineSmokeError(
            f"engine errors module is missing contract errors: {missing_errors}"
        )
    for name in IDENTITY_ERRORS:
        if getattr(errors, name) is not getattr(contracts, name):
            raise EngineSmokeError(f"engine error {name} is not the contract error")
    return len(IDENTITY_NAMES) + len(IDENTITY_ERRORS)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-contract-version", required=True)
    parser.add_argument("--import-matrix", type=Path, required=True)
    parser.add_argument("--forbid-source-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the installed-engine checks and print a JSON report.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv``.

    Returns:
        ``0`` on success, ``1`` when a check fails.
    """
    arguments = _build_parser().parse_args(argv)
    try:
        engine = importlib.import_module("configurable_automl_engine")
        contracts = importlib.import_module("automl_contracts")
        engine_location = ensure_import_origin(
            engine,
            arguments.forbid_source_root,
            "configurable_automl_engine",
        )
        contract_location = ensure_import_origin(
            contracts,
            arguments.forbid_source_root,
            "automl_contracts",
        )
        dependency = check_contract_dependency(arguments.expected_contract_version)
        matrix = load_import_matrix(arguments.import_matrix)
        symbols = check_import_matrix(matrix)
        identities = check_contract_identity()
    except EngineSmokeError as error:
        print(f"engine smoke failed: {error}", file=sys.stderr)
        return 1
    report = {
        "engine_smoke": "ok",
        "dependency": dependency,
        "import_matrix": {"modules": len(matrix), "symbols": symbols},
        "contract_identity": identities,
        "engine_location": str(engine_location),
        "contract_location": str(contract_location),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
