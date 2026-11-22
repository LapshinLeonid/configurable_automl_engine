"""Public API, dependency boundary and packaging data of the contract."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import automl_contracts

CONTRACT_ROOT = Path(__file__).resolve().parents[2] / "configurable-automl-contracts"
CONTRACT_SRC = CONTRACT_ROOT / "src" / "automl_contracts"

FORBIDDEN_TOP_LEVEL = {
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
}


def test_public_names_are_exported_and_importable() -> None:
    assert automl_contracts.__all__
    for name in automl_contracts.__all__:
        assert hasattr(automl_contracts, name), name


def test_stable_public_submodules() -> None:
    from automl_contracts import (
        config,
        custom_metrics,
        definitions,
        errors,
        feature_selection,
        metrics,
        preprocessing,
        protocols,
        result,
        search_space,
    )

    assert config.Config is automl_contracts.Config
    assert definitions.ValidationStrategy is automl_contracts.ValidationStrategy
    assert errors.TrainingError is automl_contracts.TrainingError
    assert metrics.AVAILABLE_METRICS == automl_contracts.AVAILABLE_METRICS
    assert custom_metrics.CustomMetricSpec is automl_contracts.CustomMetricSpec
    assert feature_selection.FeatureSelectionCfg is automl_contracts.FeatureSelectionCfg
    assert preprocessing.PreprocessingOverride is automl_contracts.PreprocessingOverride
    assert protocols.MetricFunction is automl_contracts.MetricFunction
    assert result.ExperimentResult is automl_contracts.ExperimentResult
    assert search_space.SearchSpaceEntry is automl_contracts.SearchSpaceEntry


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_contract_has_no_engine_or_heavy_imports() -> None:
    offenders: list[str] = []
    for path in CONTRACT_SRC.rglob("*.py"):
        forbidden = _imported_roots(path) & FORBIDDEN_TOP_LEVEL
        if forbidden:
            offenders.append(f"{path.relative_to(CONTRACT_SRC)}: {sorted(forbidden)}")
    assert offenders == []


def test_import_does_not_load_engine_or_heavy_modules() -> None:
    code = (
        "import json, sys\n"
        "import automl_contracts\n"
        "names = ['configurable_automl_engine', 'numpy', 'pandas', 'sklearn',\n"
        "         'scipy', 'optuna', 'yaml', 'joblib']\n"
        "print(json.dumps([n for n in names if n in sys.modules]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def test_py_typed_and_schema_are_package_data() -> None:
    package = files("automl_contracts")
    assert package.joinpath("py.typed").is_file()
    schema = package.joinpath("schemas", "config.schema.json")
    assert schema.is_file()
    payload = json.loads(schema.read_text(encoding="utf-8"))
    assert payload["$defs"]["AlgoCfg"]["additionalProperties"] is False


def test_distribution_metadata() -> None:
    from importlib.metadata import metadata, requires

    meta = metadata("configurable-automl-contracts")
    assert meta["Name"] == "configurable-automl-contracts"
    requirements = "\n".join(requires("configurable-automl-contracts") or [])
    for forbidden in ("scikit-learn", "pandas", "numpy", "optuna"):
        assert forbidden not in requirements
    assert "pydantic" in requirements
