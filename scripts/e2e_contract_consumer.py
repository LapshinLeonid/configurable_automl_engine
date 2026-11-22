"""End-to-end consumer check for ``configurable-automl-contracts`` installed from a registry.

Run this script with the interpreter of a clean virtual environment, from a
directory outside the repository. It proves that the published contract is
self-sufficient: public models, protocols and the packaged JSON Schema validate
a reference configuration without the engine or its heavy dependencies.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import re
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

import automl_contracts
import jsonschema  # type: ignore[import-untyped]

DISTRIBUTION_NAME = "configurable-automl-contracts"
PUBLIC_MODULES = (
    "config",
    "custom_metrics",
    "definitions",
    "errors",
    "feature_selection",
    "metrics",
    "preprocessing",
    "protocols",
    "result",
    "search_space",
)
REQUIRED_NAMES = (
    "AlgorithmRun",
    "Config",
    "CustomMetricSpec",
    "ExperimentResult",
    "GeneralCfg",
    "TrainingError",
    "ValidationStrategy",
)
PROTOCOL_NAMES = ("MetricFunction", "ScorerCallable")
FORBIDDEN_INSTALLED = (
    "configurable_automl_engine",
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "optuna",
    "yaml",
    "joblib",
    "imblearn",
    "mlflow",
    "shap",
    "xgboost",
    "pyarrow",
)
SCHEMA_RESOURCE = "schemas/config.schema.json"


class ConsumerError(RuntimeError):
    """The installed contract cannot serve an external consumer."""


def ensure_modules_absent(names: tuple[str, ...]) -> None:
    """Fail when any forbidden module is importable.

    Args:
        names: Import names that must not be installed in the consumer
            environment.

    Raises:
        ConsumerError: At least one forbidden module is available.
    """
    present = [name for name in names if importlib.util.find_spec(name) is not None]
    if present:
        raise ConsumerError(
            "the consumer environment contains modules that the contract must "
            f"not need: {present}"
        )


def ensure_import_origin(forbidden_root: Path | None) -> Path:
    """Fail when the contract is imported from the repository checkout.

    Args:
        forbidden_root: Repository root that must not contain the imported
            package.

    Returns:
        The resolved path of the imported contract package.

    Raises:
        ConsumerError: The imported package lives under ``forbidden_root``.
    """
    location = Path(automl_contracts.__file__).resolve()
    if forbidden_root is not None:
        root = forbidden_root.resolve()
        if location.is_relative_to(root):
            raise ConsumerError(
                f"automl_contracts was imported from {location}, not from the "
                "installed distribution"
            )
    return location


def check_metadata(expected_version: str) -> dict[str, Any]:
    """Check distribution metadata and declared runtime requirements.

    Args:
        expected_version: Version encoded in the release tag.

    Returns:
        Selected metadata fields.

    Raises:
        ConsumerError: The installed distribution does not match the release or
            declares an unexpected requirement.
    """
    meta = importlib.metadata.metadata(DISTRIBUTION_NAME)
    name = meta["Name"]
    version = meta["Version"]
    if name != DISTRIBUTION_NAME:
        raise ConsumerError(f"unexpected distribution name {name!r}")
    if version != expected_version:
        raise ConsumerError(
            f"installed {DISTRIBUTION_NAME} version {version!r} does not match "
            f"the expected version {expected_version!r}"
        )
    if automl_contracts.__version__ != version:
        raise ConsumerError(
            f"automl_contracts.__version__ {automl_contracts.__version__!r} "
            f"differs from distribution metadata {version!r}"
        )
    requirements = importlib.metadata.requires(DISTRIBUTION_NAME) or []
    allowed = ("pydantic", "pydantic-core")
    for requirement in requirements:
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
        requirement_name = match.group(0).lower().replace("_", "-") if match else ""
        if requirement_name not in allowed:
            raise ConsumerError(
                f"{DISTRIBUTION_NAME} declares an unexpected requirement {requirement!r}"
            )
    return {"name": name, "version": version, "requirements": requirements}


def check_package_data() -> dict[str, Any]:
    """Check that ``py.typed`` and the canonical JSON Schema are installed.

    Returns:
        The checked package-data names and the schema title.

    Raises:
        ConsumerError: A required package-data file is missing.
    """
    package = files("automl_contracts")
    typed = package.joinpath("py.typed")
    schema = package.joinpath(SCHEMA_RESOURCE)
    if not typed.is_file():
        raise ConsumerError("installed contract is missing py.typed")
    if not schema.is_file():
        raise ConsumerError(f"installed contract is missing {SCHEMA_RESOURCE}")
    payload = json.loads(schema.read_text(encoding="utf-8"))
    return {
        "py_typed": True,
        "schema": SCHEMA_RESOURCE,
        "schema_title": payload.get("title"),
    }


def check_public_surface() -> dict[str, Any]:
    """Check that the public models, modules and protocols are available.

    Returns:
        Counts of the checked public names.

    Raises:
        ConsumerError: A documented public name or module is missing.
    """
    missing = [name for name in REQUIRED_NAMES if not hasattr(automl_contracts, name)]
    if missing:
        raise ConsumerError(f"contract public API is missing names: {missing}")
    for module_name in PUBLIC_MODULES:
        importlib.import_module(f"automl_contracts.{module_name}")
    protocols = importlib.import_module("automl_contracts.protocols")
    for name in PROTOCOL_NAMES:
        protocol = getattr(protocols, name, None)
        if protocol is None or not getattr(protocol, "_is_protocol", False):
            raise ConsumerError(f"protocol {name} is missing or is not a Protocol")
        if "__call__" not in protocol.__dict__:
            raise ConsumerError(f"protocol {name} declares no __call__")
    if not automl_contracts.__all__:
        raise ConsumerError("contract exports no public names")
    return {
        "required_names": len(REQUIRED_NAMES),
        "public_modules": len(PUBLIC_MODULES),
        "protocols": len(PROTOCOL_NAMES),
        "public_names": len(automl_contracts.__all__),
    }


def load_reference_config(path: Path) -> dict[str, Any]:
    """Load a reference configuration from JSON.

    Args:
        path: Reference configuration file.

    Returns:
        The parsed JSON object.

    Raises:
        ConsumerError: The file is unreadable, invalid or not an object.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConsumerError(
            f"cannot read reference configuration {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ConsumerError(f"reference configuration {path} is not a JSON object")
    return payload


def validate_reference_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a configuration with the contract models and packaged schema.

    Args:
        payload: Reference configuration data.

    Returns:
        The validated model dump and the schema identifier.

    Raises:
        ConsumerError: The contract rejects the reference configuration.
    """
    package = files("automl_contracts")
    schema = json.loads(package.joinpath(SCHEMA_RESOURCE).read_text(encoding="utf-8"))
    try:
        config = automl_contracts.Config.model_validate(payload)
    except ValueError as error:
        raise ConsumerError(
            f"the contract models reject the reference configuration: {error}"
        ) from error
    dumped = config.model_dump(mode="json", by_alias=True)
    try:
        jsonschema.validate(instance=dumped, schema=schema)
    except jsonschema.ValidationError as error:
        raise ConsumerError(
            "the packaged JSON Schema rejects the normalized reference "
            f"configuration: {error.message}"
        ) from error
    return {"schema": SCHEMA_RESOURCE, "model": type(config).__name__}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--forbid-source-root", type=Path)
    parser.add_argument(
        "--forbid-installed",
        nargs="*",
        default=list(FORBIDDEN_INSTALLED),
        help="Modules that must not be installed in the consumer environment.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the end-to-end consumer checks and print a JSON report.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv``.

    Returns:
        ``0`` on success, ``1`` when a consumer check fails.
    """
    arguments = _build_parser().parse_args(argv)
    try:
        ensure_modules_absent(tuple(arguments.forbid_installed))
        location = ensure_import_origin(arguments.forbid_source_root)
        metadata_report = check_metadata(arguments.expected_version)
        package_report = check_package_data()
        surface_report = check_public_surface()
        payload = load_reference_config(arguments.reference_config)
        config_report = validate_reference_config(payload)
    except ConsumerError as error:
        print(f"consumer e2e failed: {error}", file=sys.stderr)
        return 1
    report = {
        "consumer_e2e": "ok",
        "distribution": metadata_report,
        "package_data": package_report,
        "public_surface": surface_report,
        "reference_config": config_report,
        "package_location": str(location),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
