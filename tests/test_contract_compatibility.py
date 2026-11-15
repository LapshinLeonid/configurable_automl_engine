"""Compatibility between the engine paths and the contract definitions."""

from __future__ import annotations

import ast
import os
import pickle
import subprocess
import sys
from pathlib import Path

import automl_contracts as contracts
import pytest

import configurable_automl_engine as engine


def test_root_reexports_keep_class_identity() -> None:
    assert engine.Config is contracts.Config
    assert engine.CustomMetricSpec is contracts.CustomMetricSpec
    assert engine.ExperimentResult is contracts.ExperimentResult
    assert engine.AlgorithmRun is contracts.AlgorithmRun


def test_old_submodule_paths_keep_class_identity() -> None:
    from configurable_automl_engine.common.custom_metrics import CustomMetricSpec
    from configurable_automl_engine.common.definitions import (
        WORST_SCORE_THRESHOLD,
        MetricDirection,
        ValidationStrategy,
    )
    from configurable_automl_engine.common.errors import (
        InvalidPredictionError,
        TrainingError,
    )
    from configurable_automl_engine.common.metrics import (
        AVAILABLE_METRICS,
        ERR_CUSTOM_NAME_INVALID,
        to_sklearn_name,
    )
    from configurable_automl_engine.experiment.result import (
        AlgorithmRun,
        ExperimentResult,
    )
    from configurable_automl_engine.feature_selection import (
        FeatureSelectionCfg,
        FeatureSelectionMethod,
        FeatureSelectionMode,
    )
    from configurable_automl_engine.hpo.default_spaces import (
        FloatSpace,
        SearchSpaceEntry,
    )
    from configurable_automl_engine.models import AVAILABLE_ALGORITHMS
    from configurable_automl_engine.outliers_presets import (
        IqrOutliers,
        NoOutliers,
        QuantileOutliers,
    )
    from configurable_automl_engine.preprocessing_presets import (
        PreprocessingOverride,
        PreprocessingPreset,
        RobustScaling,
        merge_preprocessing_overrides,
    )
    from configurable_automl_engine.training_engine.config.algorithms import AlgoCfg
    from configurable_automl_engine.training_engine.config.general import GeneralCfg
    from configurable_automl_engine.training_engine.config.phases import HPOPhaseCfg
    from configurable_automl_engine.training_engine.config.sanity_gate import (
        SanityGateCfg,
    )
    from configurable_automl_engine.training_engine.config_parser import Config

    assert Config is contracts.Config
    assert GeneralCfg is contracts.GeneralCfg
    assert AlgoCfg is contracts.AlgoCfg
    assert HPOPhaseCfg is contracts.HPOPhaseCfg
    assert SanityGateCfg is contracts.SanityGateCfg
    assert FeatureSelectionCfg is contracts.FeatureSelectionCfg
    assert FeatureSelectionMode is contracts.FeatureSelectionMode
    assert FeatureSelectionMethod is contracts.FeatureSelectionMethod
    assert PreprocessingOverride is contracts.PreprocessingOverride
    assert PreprocessingPreset is contracts.PreprocessingPreset
    assert RobustScaling is contracts.RobustScaling
    assert NoOutliers is contracts.NoOutliers
    assert QuantileOutliers is contracts.QuantileOutliers
    assert IqrOutliers is contracts.IqrOutliers
    assert merge_preprocessing_overrides is contracts.merge_preprocessing_overrides
    assert SearchSpaceEntry is contracts.SearchSpaceEntry
    assert FloatSpace is contracts.FloatSpace
    assert AlgorithmRun is contracts.AlgorithmRun
    assert ExperimentResult is contracts.ExperimentResult
    assert CustomMetricSpec is contracts.CustomMetricSpec
    assert ValidationStrategy is contracts.ValidationStrategy
    assert MetricDirection is contracts.MetricDirection
    assert TrainingError is contracts.TrainingError
    assert InvalidPredictionError is contracts.InvalidPredictionError
    assert AVAILABLE_METRICS == contracts.AVAILABLE_METRICS
    assert AVAILABLE_ALGORITHMS == contracts.AVAILABLE_ALGORITHMS
    assert isinstance(AVAILABLE_ALGORITHMS, list)
    assert isinstance(AVAILABLE_METRICS, list)
    assert WORST_SCORE_THRESHOLD == contracts.WORST_SCORE_THRESHOLD
    assert ERR_CUSTOM_NAME_INVALID == contracts.ERR_CUSTOM_NAME_INVALID
    assert to_sklearn_name("rmse") == "neg_root_mean_squared_error"


def test_engine_does_not_redefine_contract_classes() -> None:
    package = Path(engine.__file__).parent
    contract_names = set(contracts.__all__)
    offenders: list[str] = []
    for path in package.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in contract_names:
                offenders.append(f"{path.relative_to(package)}:{node.name}")
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
    """Repeated runtime validation does not change an already normalized config."""
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


def test_schema_generator_writes_the_mirror_to_the_calling_directory(
    tmp_path: Path,
) -> None:
    """The compatibility generator keeps the historical cwd-relative CLI."""
    repo_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root / "automl-contracts" / "src"), env.get("PYTHONPATH", "")]
    )
    subprocess.run(
        [sys.executable, str(repo_root / "schema_generator.py")],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
    )
    generated = (tmp_path / "config.schema.json").read_bytes()
    canonical = (
        repo_root
        / "automl-contracts"
        / "src"
        / "automl_contracts"
        / "schemas"
        / "config.schema.json"
    ).read_bytes()
    assert generated == canonical
    assert not generated.endswith(b"\n")
