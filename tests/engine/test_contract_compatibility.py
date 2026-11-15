"""Backward compatibility between the engine paths and the contract package."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import automl_contracts as contracts
import pytest

import configurable_automl_engine as engine
from tests.contracts._contract_snapshot import (
    contract_public_names,
    parse_engine_imports_from_source,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_SRC = REPO_ROOT / "src" / "configurable_automl_engine"
CONTRACT_SRC = REPO_ROOT / "automl-contracts" / "src"
IMPORT_MATRIX = Path(__file__).resolve().parent / "golden" / "import_matrix.json"


def load_import_matrix() -> dict[str, list[str]]:
    return json.loads(IMPORT_MATRIX.read_text(encoding="utf-8"))["paths"]


def contract_symbol(name: str) -> Any:
    if name == "__version__":
        raise KeyError(name)
    if hasattr(contracts, name):
        return getattr(contracts, name)
    from automl_contracts import preprocessing

    if hasattr(preprocessing, name):
        return getattr(preprocessing, name)
    raise KeyError(f"{name} is not a documented contract symbol")


def test_import_matrix_matches_the_documented_compatibility_matrix() -> None:
    documented: dict[str, list[str]] = {}
    for path, name in parse_engine_imports_from_source():
        documented.setdefault(path, [])
        if name not in documented[path]:
            documented[path].append(name)
    assert load_import_matrix() == documented


def test_golden_import_matrix_is_exact() -> None:
    golden = load_import_matrix()
    expected_paths = {
        "configurable_automl_engine",
        "configurable_automl_engine.common.definitions",
        "configurable_automl_engine.common.errors",
        "configurable_automl_engine.training_engine.config_parser",
        "configurable_automl_engine.training_engine.config.algorithms",
        "configurable_automl_engine.training_engine.config.general",
        "configurable_automl_engine.training_engine.config.phases",
        "configurable_automl_engine.training_engine.config.sanity_gate",
        "configurable_automl_engine.training_engine.config.complexity_guard",
        "configurable_automl_engine.training_engine.config.ensemble",
        "configurable_automl_engine.feature_selection",
        "configurable_automl_engine.preprocessing_presets",
        "configurable_automl_engine.outliers_presets",
        "configurable_automl_engine.hpo.default_spaces",
        "configurable_automl_engine.experiment.result",
        "configurable_automl_engine.experiment",
        "configurable_automl_engine.models",
        "configurable_automl_engine.common.metrics",
        "configurable_automl_engine.common.custom_metrics",
        "configurable_automl_engine.trainer",
        "configurable_automl_engine.tuner",
        "configurable_automl_engine.preprocessing",
        "configurable_automl_engine.training_engine",
    }
    assert set(golden) == expected_paths


@pytest.mark.parametrize(
    ("module_name", "symbol"),
    [
        (path, symbol)
        for path, symbols in load_import_matrix().items()
        for symbol in symbols
    ],
)
def test_old_public_import_path_works(module_name: str, symbol: str) -> None:
    module = importlib.import_module(module_name)
    assert hasattr(module, symbol), f"{module_name}.{symbol} disappeared"


def test_root_reexports_keep_class_identity() -> None:
    assert engine.Config is contracts.Config
    assert engine.CustomMetricSpec is contracts.CustomMetricSpec
    assert engine.ExperimentResult is contracts.ExperimentResult
    assert engine.AlgorithmRun is contracts.AlgorithmRun
    assert engine.AutoMLExperiment is not None
    assert engine.ModelTrainer is not None
    assert engine.train_best_model is not None
    assert engine.__version__


def test_contract_owned_symbols_keep_identity() -> None:
    public = contract_public_names()
    checked: set[str] = set()
    for path, symbols in load_import_matrix().items():
        module = importlib.import_module(path)
        for symbol in symbols:
            if symbol == "__version__" or symbol not in public:
                continue
            expected = contract_symbol(symbol)
            assert getattr(module, symbol) is expected, f"{path}.{symbol}"
            checked.add(symbol)
    assert {"Config", "ExperimentResult", "AlgorithmRun", "CustomMetricSpec"} <= checked
    assert {"TrainingError", "InvalidPredictionError"} <= checked
    assert {"ValidationStrategy", "MetricDirection", "RunStatus"} <= checked
    assert "ScalingType" in checked


def test_engine_does_not_redefine_contract_classes() -> None:
    contract_names = set(contracts.__all__) | {"ScalingType"}
    offenders: list[str] = []
    for path in ENGINE_SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in contract_names:
                offenders.append(f"{path.relative_to(ENGINE_SRC)}:{node.name}")
    assert offenders == []


def test_old_pickle_module_paths_still_resolve() -> None:
    preset = contracts.PreprocessingPreset()
    payload = pickle.dumps(preset, protocol=0)
    old_payload = payload.replace(
        b"automl_contracts.preprocessing",
        b"configurable_automl_engine.preprocessing_presets",
    )
    assert isinstance(pickle.loads(old_payload), contracts.PreprocessingPreset)

    spec = contracts.CustomMetricSpec(path="builtins:sum", greater_is_better=True)
    spec_payload = pickle.dumps(spec, protocol=0).replace(
        b"automl_contracts.custom_metrics",
        b"configurable_automl_engine.common.custom_metrics",
    )
    assert isinstance(pickle.loads(spec_payload), contracts.CustomMetricSpec)


def test_old_consumer_scenarios_still_work() -> None:
    from configurable_automl_engine.common.custom_metrics import CustomMetricSpec
    from configurable_automl_engine.common.metrics import (
        AVAILABLE_METRICS,
        to_sklearn_name,
    )
    from configurable_automl_engine.experiment.result import ExperimentResult
    from configurable_automl_engine.hpo.default_spaces import SearchSpaceEntry
    from configurable_automl_engine.preprocessing_presets import (
        PreprocessingOverride,
        merge_preprocessing_overrides,
    )
    from configurable_automl_engine.training_engine.config_parser import Config

    config = Config.model_validate(
        {
            "general": {"comparison_metric": "r2"},
            "algorithms": {"ridge": {}, "xgboosting": {}},
        }
    )
    assert config.general.comparison_metric == "r2"
    assert config.algorithms.xgboosting is not None

    spec = CustomMetricSpec(path="builtins:sum", greater_is_better=False)
    assert spec.kind == "metric"
    assert AVAILABLE_METRICS == contracts.AVAILABLE_METRICS
    assert to_sklearn_name("RMSE") == "neg_root_mean_squared_error"
    assert SearchSpaceEntry.model_validate([1, 10]).bounds == [1, 10, "int"]
    merged = merge_preprocessing_overrides(
        PreprocessingOverride(scaling="standard"), {"scaling": "robust"}
    )
    assert merged.scaling is not None and merged.scaling.type == "robust"  # type: ignore[union-attr]
    assert ExperimentResult is contracts.ExperimentResult


def test_read_config_keeps_yaml_loading_and_stable_contract_codes(
    tmp_path: Path,
) -> None:
    from pydantic import ValidationError

    from configurable_automl_engine.training_engine.config_parser import read_config

    valid = tmp_path / "valid.yaml"
    valid.write_text(
        """
general:
  comparison_metric: r2
  phases:
    - {name: search, n_trials: 1, action: all_algorithms}
    - {name: final_select, n_trials: 1, action: select, strategy: cv_best, output: winner}
algorithms:
  ridge: {}
""",
        encoding="utf-8",
    )
    config = read_config(valid)
    assert isinstance(config, contracts.Config)
    assert config.general.comparison_metric == "r2"

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        """
general:
  phases:
    - {name: bad, n_trials: 1, action: refine_winner}
algorithms:
  ridge: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError) as exc_info:
        read_config(invalid)
    assert exc_info.value.errors()[0]["type"] == contracts.ERR_REMOVED_REFINE_WINNER


def test_reference_config_round_trips_through_yaml(tmp_path: Path) -> None:
    """The JSON reference fixture also parses from the documented YAML path."""
    import yaml

    from configurable_automl_engine.training_engine.config_parser import read_config

    reference = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "contracts"
            / "data"
            / "reference_config.json"
        ).read_text(encoding="utf-8")
    )
    path = tmp_path / "reference.yaml"
    path.write_text(yaml.safe_dump(reference, sort_keys=False), encoding="utf-8")
    from_yaml = read_config(path)
    from_dict = contracts.Config.model_validate(reference)
    assert from_yaml.model_dump(mode="json") == from_dict.model_dump(mode="json")


def test_config_is_structural_and_runtime_checks_are_explicit() -> None:
    from configurable_automl_engine.training_engine.config_runtime import (
        validate_runtime_config,
    )

    data = {
        "general": {
            "parallel_mode": "threads",
            "phase_timeout": 10.0,
            "phases": [
                {"name": "search", "n_trials": 1},
                {
                    "name": "final",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
        },
        "algorithms": {"ridge": {"enable": True}},
    }
    cfg = contracts.Config.model_validate(data)
    with pytest.raises(ValueError, match="timeouts are not supported"):
        validate_runtime_config(cfg)


def test_custom_metric_spec_is_structural_but_entry_point_validates() -> None:
    from configurable_automl_engine.common.custom_metrics import (
        validate_custom_metric_spec,
    )

    spec = contracts.CustomMetricSpec(path="no_such_module_xyz:f")
    with pytest.raises(ValueError):
        validate_custom_metric_spec(spec)


def test_runtime_config_validation_is_idempotent() -> None:
    from configurable_automl_engine.training_engine.config_runtime import (
        validate_runtime_config,
    )

    cfg = contracts.Config.model_validate(
        {
            "general": {
                "validation_strategy": "k_fold",
                "n_folds": 2,
                "phases": [
                    {"name": "search", "n_trials": 1},
                    {
                        "name": "final",
                        "n_trials": 1,
                        "action": "select",
                        "strategy": "cv_best",
                        "output": "winner",
                    },
                ],
            },
            "algorithms": {"svr": {"enable": True, "hyperparameters": {"C": [1, 50]}}},
        }
    )
    validate_runtime_config(cfg)
    first = cfg.model_dump()
    assert validate_runtime_config(cfg) is cfg
    assert cfg.model_dump() == first


def test_engine_implementations_conform_to_contract_protocols(tmp_path: Path) -> None:
    """The public engine callables are checked by the static analyzer."""
    script = tmp_path / "protocol_conformance.py"
    script.write_text(
        """
import numpy

from automl_contracts import CustomMetricSpec
from automl_contracts.protocols import MetricFunction, ScorerCallable
from configurable_automl_engine.common.custom_metrics import custom_metric_function
from configurable_automl_engine.common.metrics import get_global_nrmse_scorer

metric: MetricFunction = custom_metric_function(
    CustomMetricSpec(path="tests.custom_metric_fixtures:scaled_r2", greater_is_better=True)
)
scorer: ScorerCallable = get_global_nrmse_scorer(numpy.array([1.0, 2.0]))
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["MYPYPATH"] = os.pathsep.join([str(ENGINE_SRC), str(CONTRACT_SRC)])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            str(script),
            "--strict",
            "--ignore-missing-imports",
            "--no-error-summary",
            "--cache-dir",
            str(tmp_path / ".mypy_cache"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_protocol_shaped_engine_adapters_behave_at_runtime() -> None:
    from configurable_automl_engine.common.custom_metrics import custom_metric_function
    from configurable_automl_engine.common.metrics import get_global_nrmse_scorer

    metric = custom_metric_function(
        contracts.CustomMetricSpec(
            path="tests.custom_metric_fixtures:scaled_r2",
            greater_is_better=True,
            kwargs={"scale": 2.0},
        )
    )
    value = metric([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert isinstance(value, float)
    assert value == pytest.approx(2.0)

    scorer = get_global_nrmse_scorer(__import__("numpy").array([1.0, 2.0, 3.0]))
    inspect.signature(scorer).bind(object(), [[1.0]], [1.0])


def test_schema_generator_writes_the_mirror_to_the_calling_directory(
    tmp_path: Path,
) -> None:
    """The compatibility generator keeps the historical cwd-relative CLI."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "automl-contracts" / "src"), env.get("PYTHONPATH", "")]
    )
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "schema_generator.py")],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )
    generated = (tmp_path / "config.schema.json").read_bytes()
    canonical = (
        REPO_ROOT
        / "automl-contracts"
        / "src"
        / "automl_contracts"
        / "schemas"
        / "config.schema.json"
    ).read_bytes()
    assert generated == canonical
    assert not generated.endswith(b"\n")
