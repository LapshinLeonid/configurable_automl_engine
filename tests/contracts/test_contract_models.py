"""Creation, defaults, constraints and serialization of the contract models."""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import automl_contracts as contracts
import pytest
from pydantic import ValidationError

DATA_DIR = Path(__file__).resolve().parent / "data"

BASE_CONFIG: dict[str, Any] = {
    "general": {
        "comparison_metric": "r2",
        "validation_strategy": "k_fold",
        "n_folds": 3,
        "phases": [
            {"name": "search", "n_trials": 2, "action": "all_algorithms"},
            {
                "name": "final_select",
                "n_trials": 1,
                "action": "select",
                "strategy": "cv_best",
                "output": "winner",
            },
        ],
    },
    "algorithms": {"ridge": {"enable": True}},
}


def load_json(name: str) -> Any:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def error_types(data: dict[str, Any]) -> set[str]:
    with pytest.raises(ValidationError) as exc_info:
        contracts.Config.model_validate(data)
    return {str(error["type"]) for error in exc_info.value.errors()}


def test_reference_config_parses_and_has_expected_values() -> None:
    config = contracts.Config.model_validate(load_json("reference_config.json"))
    assert config.general.comparison_metric == "r2"
    assert config.general.validation_strategy is contracts.ValidationStrategy.k_fold
    assert config.general.n_folds == 3
    assert config.general.n_repeats == 2
    assert config.general.split_aggregation == "median"
    assert (
        config.general.feature_selection.method
        is contracts.FeatureSelectionMethod.percentile
    )
    assert config.general.preprocessing is not None
    assert config.general.preprocessing.scaling.type == "robust"  # type: ignore[union-attr]
    assert config.general.preprocessing.outliers.method == "iqr"  # type: ignore[union-attr]
    assert config.general.target_transform is contracts.TargetTransform.yeo_johnson
    assert config.general.ensemble.enable is True
    assert config.general.complexity_guard.user_range_policy is (
        contracts.ComplexityGuardUserRangePolicy.engine_wins
    )
    assert config.oversampling.enable is True
    assert config.oversampling.algorithm is contracts.OversamplingAlgorithm.smote
    assert config.algorithms.ridge is not None
    assert config.algorithms.ridge.enable is True
    assert config.algorithms.random_forest is not None
    assert config.algorithms.random_forest.enable is False


def test_example_config_parses() -> None:
    config = contracts.Config.model_validate(load_json("example_config.json"))
    assert config.general.comparison_metric == "r2"
    assert config.general.path_to_model == Path("diabetes_model.pkl")
    assert [phase.name for phase in config.general.phases] == [
        "search",
        "final_select",
    ]
    assert config.algorithms.ridge is not None


def test_the_example_py_config_is_accepted() -> None:
    """The documented example.py config is a contract-valid configuration."""
    source = (Path(__file__).resolve().parents[2] / "example.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "config"
            for target in node.targets
        )
    )
    payload = ast.literal_eval(assignment.value)
    config = contracts.Config.model_validate(payload)
    assert config.general.comparison_metric == "r2"
    assert config.general.path_to_model == Path("diabetes_model.pkl")
    assert set(config.model_dump(mode="json")["algorithms"]) >= {
        "ridge",
        "random_forest",
    }


def test_config_round_trip_is_stable() -> None:
    config = contracts.Config.model_validate(load_json("reference_config.json"))
    dumped = config.model_dump(mode="json")
    restored = contracts.Config.model_validate(dumped)
    assert restored.model_dump(mode="json") == dumped
    assert restored.model_dump(mode="json") == config.model_dump(mode="json")
    assert json.dumps(dumped, sort_keys=True) == json.dumps(
        restored.model_dump(mode="json"), sort_keys=True
    )


def test_reference_config_serialization_matches_reference() -> None:
    config = contracts.Config.model_validate(load_json("reference_config.json"))
    actual = config.model_dump(mode="json")
    expected = json.loads(
        (DATA_DIR / "reference_config_serialized.json").read_text(encoding="utf-8")
    )
    assert actual == expected, (
        "The serialized reference config changed. Review the contract change, "
        "then regenerate the reference with "
        "`python -m tests.contracts.update_reference_serializations`."
    )


def test_default_values_of_the_config_sections() -> None:
    general = contracts.GeneralCfg()
    assert general.comparison_metric == "r2"
    assert general.validation_strategy is contracts.ValidationStrategy.auto
    assert general.n_folds == 5
    assert general.n_repeats == 1
    assert general.split_aggregation == "median"
    assert general.parallel_mode == "threads"
    assert [phase.name for phase in general.phases] == ["search", "final_select"]
    assert general.phases[-1].action == "select"
    assert general.phases[-1].output == "winner"
    assert general.pruning == contracts.PruningCfg()
    assert general.feature_selection.mode is contracts.FeatureSelectionMode.disabled
    assert general.sanity_gate.corridor_mode == "auto"
    assert general.complexity_guard.mode is contracts.ComplexityGuardMode.apply
    assert general.ensemble == contracts.EnsembleCfg()
    assert general.target_transform is contracts.TargetTransform.none
    assert general.inf_policy == "as_missing"
    assert general.datetime_policy == "expand"

    pruning = contracts.PruningCfg()
    assert pruning.enable is False
    assert pruning.strategy is contracts.PruningStrategy.median
    assert pruning.min_steps == 1
    assert pruning.reduction_factor == 3

    oversampling = contracts.OversamplingCfg()
    assert oversampling.enable is False
    assert oversampling.multiplier == 1.0
    assert oversampling.algorithm is contracts.OversamplingAlgorithm.random

    algo = contracts.AlgoCfg()
    assert algo.enable is True
    assert algo.limit_hyperparameters is False
    assert algo.hyperparameters is None


def test_config_requires_at_least_one_enabled_algorithm() -> None:
    data = deep_update(
        BASE_CONFIG,
        {"algorithms": {"ridge": {"enable": False}}},
    )
    assert "value_error" in error_types(data)


@pytest.mark.parametrize(
    ("update", "expected_code"),
    [
        ({"unknown_section": {}}, "extra_forbidden"),
        ({"algorithms": {"no_such_algo": {"enable": True}}}, "extra_forbidden"),
        (
            {
                "general": {
                    "phases": [
                        {"name": "bad", "n_trials": 1, "action": "bogus"},
                    ]
                }
            },
            "literal_error",
        ),
        (
            {
                "general": {
                    "phases": [
                        {"name": "bad", "n_trials": 1, "action": "refine_winner"},
                    ]
                }
            },
            contracts.ERR_REMOVED_REFINE_WINNER,
        ),
        (
            {
                "general": {
                    "phases": [
                        {"name": "select", "action": "select", "output": "winner"},
                    ]
                }
            },
            "value_error",
        ),
        (
            {
                "general": {
                    "phases": [
                        {
                            "name": "select",
                            "action": "select",
                            "strategy": "cv_best",
                            "output": "top_k",
                        },
                    ]
                }
            },
            "value_error",
        ),
        (
            {
                "general": {
                    "phases": [
                        {
                            "name": "select",
                            "action": "select",
                            "strategy": "cv_best",
                            "output": "winner",
                            "k": 2,
                        },
                    ]
                }
            },
            "value_error",
        ),
        (
            {"general": {"sanity_gate": {"mode": "active"}}},
            contracts.ERR_REMOVED_SANITY_GATE_MODE,
        ),
        (
            {"algorithms": {"ridge": {"enable": True, "tuner": "custom"}}},
            contracts.ERR_REMOVED_PLUGIN_KEY,
        ),
        ({"general": {"comparison_metric": "no_such_metric"}}, "value_error"),
        (
            {"general": {"custom_metrics": {"Bad-Name": {"path": "builtins:sum"}}}},
            contracts.ERR_CUSTOM_NAME_INVALID,
        ),
        ({"general": {"validation_strategy": "k_fold", "n_folds": 1}}, "value_error"),
        ({"general": {"target_transform": "sqrt"}}, "enum"),
        (
            {"general": {"feature_selection": {"percentile": 0}}},
            "greater_than",
        ),
        (
            {"general": {"complexity_guard": {"small_n_threshold": 300}}},
            "value_error",
        ),
        (
            {"general": {"sanity_gate": {"corridor_delta": 1.0}}},
            "less_than",
        ),
        (
            {
                "general": {
                    "preprocessing": {
                        "scaling": {"type": "robust", "quantile_range": [80, 20]}
                    }
                }
            },
            "value_error",
        ),
        (
            {"algorithms": {"ridge": {"hyperparameters": {"alpha": [100, 1]}}}},
            "value_error",
        ),
        (
            {"algorithms": {"ridge": {"limit_hyperparameters": True}}},
            "value_error",
        ),
        (
            {"oversampling": {"data_oversampling_multiplier": 0.5}},
            "greater_than_equal",
        ),
    ],
)
def test_invalid_configs_are_rejected_with_stable_codes(
    update: dict[str, Any], expected_code: str
) -> None:
    data = deep_update(BASE_CONFIG, update)
    assert expected_code in error_types(data)


def test_removed_sanity_gate_mode_has_migration_hint() -> None:
    data = deep_update(BASE_CONFIG, {"general": {"sanity_gate": {"mode": "active"}}})
    with pytest.raises(ValidationError) as exc_info:
        contracts.Config.model_validate(data)
    error = exc_info.value.errors()[0]
    assert error["type"] == contracts.ERR_REMOVED_SANITY_GATE_MODE
    assert "select" in str(error["msg"])


def test_removed_refine_winner_has_migration_hint() -> None:
    data = deep_update(
        BASE_CONFIG,
        {
            "general": {
                "phases": [{"name": "bad", "n_trials": 1, "action": "refine_winner"}]
            }
        },
    )
    with pytest.raises(ValidationError) as exc_info:
        contracts.Config.model_validate(data)
    error = exc_info.value.errors()[0]
    assert error["type"] == contracts.ERR_REMOVED_REFINE_WINNER
    assert "select" in str(error["msg"]) and "refine" in str(error["msg"])


def test_search_space_shorthands_and_constraints() -> None:
    categorical = contracts.SearchSpaceEntry.model_validate([["a", "b"], "categorical"])
    assert categorical.dist_type == "categorical"
    assert categorical.bounds == [["a", "b"], "categorical"]

    integer = contracts.SearchSpaceEntry.model_validate([1, 10])
    assert isinstance(integer.config, contracts.IntSpace)
    assert integer.bounds == [1, 10, "int"]

    real = contracts.SearchSpaceEntry.model_validate([0.1, 1.0])
    assert isinstance(real.config, contracts.FloatSpace)
    assert real.bounds == [0.1, 1.0, "float"]

    logarithmic = contracts.SearchSpaceEntry.model_validate([0.01, 100.0, "float_log"])
    assert logarithmic.dist_type == "float_log"

    with pytest.raises(ValidationError, match="must be <= high"):
        contracts.SearchSpaceEntry.model_validate([10, 1])
    with pytest.raises(ValidationError, match="low must be > 0"):
        contracts.SearchSpaceEntry.model_validate([0, 1, "float_log"])
    with pytest.raises(ValidationError, match="Step must be positive"):
        contracts.SearchSpaceEntry.model_validate([1, 10, "int", 0])


def test_preprocessing_models_validate_and_round_trip() -> None:
    preset = contracts.PreprocessingPreset(
        algorithm_class=contracts.AlgorithmClass.TREES,
        imputation_strategy="median",
        scaling={"type": "quantile", "output_distribution": "normal"},
        outliers={"method": "quantile", "lower_quantile": 0.05},
        description="trees",
    )
    assert preset.scaling.type == "quantile"  # type: ignore[union-attr]
    assert preset.scaling.output_distribution == "normal"  # type: ignore[union-attr]
    assert preset.outliers.method == "quantile"  # type: ignore[union-attr]
    assert (
        contracts.PreprocessingPreset.model_validate(preset.model_dump(mode="json"))
        == preset
    )

    merged = contracts.merge_preprocessing_overrides(
        {"scaling": "standard", "outliers": "none"},
        {"scaling": "robust"},
    )
    assert merged.scaling is not None and merged.scaling.type == "robust"  # type: ignore[union-attr]
    assert merged.outliers is not None and merged.outliers.method == "none"  # type: ignore[union-attr]

    with pytest.raises(ValidationError, match="0 <= low < high <= 100"):
        contracts.RobustScaling(quantile_range=(80, 20))
    with pytest.raises(ValidationError, match="subsample"):
        contracts.QuantileScaling(n_quantiles=100, subsample=10)
    with pytest.raises(ValidationError, match="strictly less"):
        contracts.QuantileOutliers(lower_quantile=0.9, upper_quantile=0.1)


def test_custom_metric_spec_is_structural_and_frozen() -> None:
    spec = contracts.CustomMetricSpec(
        path="tests.custom_metric_fixtures:scaled_r2",
        greater_is_better=True,
        kwargs={"scale": 2.0},
    )
    assert spec.kind == "metric"
    assert spec.cache_key == (
        "tests.custom_metric_fixtures:scaled_r2",
        "metric",
        True,
        False,
        '{"scale": 2.0}',
    )
    assert (
        contracts.CustomMetricSpec.model_validate(spec.model_dump(mode="json")) == spec
    )

    with pytest.raises(ValidationError):
        spec.path = "other:path"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        contracts.CustomMetricSpec(path="builtins:sum", unknown=True)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        contracts.CustomMetricSpec(path="")

    # Importing/validating the callable is runtime work of the engine: a
    # non-existent path is structurally valid.
    structural = contracts.CustomMetricSpec(path="no_such_module_xyz:f")
    assert structural.path == "no_such_module_xyz:f"


def test_reference_result_round_trip_matches_reference() -> None:
    payload = load_json("reference_result.json")
    result = contracts.ExperimentResult.model_validate(payload)
    assert result.status is contracts.RunStatus.completed
    assert result.metric_direction is contracts.MetricDirection.maximize
    assert result.runs[0].phase_action is contracts.RunPhaseAction.all_algorithms
    assert result.runs[0].status is contracts.RunStepStatus.completed
    restored = contracts.ExperimentResult.from_json(result.to_json())
    assert restored == result
    assert restored.to_json() == result.to_json()
    actual = json.loads(result.to_json())
    expected = json.loads(
        (DATA_DIR / "reference_result_serialized.json").read_text(encoding="utf-8")
    )
    assert actual == expected, (
        "The serialized reference result changed. Review the result-model "
        "change, then regenerate the reference with "
        "`python -m tests.contracts.update_reference_serializations`."
    )


def test_result_omits_unset_repeat_statistics() -> None:
    payload = load_json("reference_result.json")
    payload.pop("score_std")
    payload.pop("oof_score_std")
    payload.pop("n_repeats")
    result = contracts.ExperimentResult.model_validate(payload)
    dumped = json.loads(result.to_json())
    assert "score_std" not in dumped
    assert "oof_score_std" not in dumped
    assert "n_repeats" not in dumped
    assert result.score_std is None
    assert result.n_repeats is None


@pytest.mark.parametrize(
    "update",
    [
        {"metrics": None},
        {"run_id": "00000000-0000-0000-0000-000000000000"},
        {"started_at": "2026-11-15T10:00:00"},
        {"metric_user": "r2", "metric_direction": "minimize"},
        {"metric_user": "nrmse", "metric_direction": "maximize"},
        {"metric_sklearn": "neg_mean_absolute_error"},
        {"winner": None},
        {"winner_score": None},
        {"resolved_validation": "auto"},
        {"resolved_test_size": 1.5},
        {"resolved_test_size": 2.5},
        {"duration_seconds": -1},
        {"additional_metrics": {"mae": float("nan")}},
        {"winner_score": float("inf")},
    ],
)
def test_invalid_results_are_rejected(update: dict[str, Any]) -> None:
    payload = load_json("reference_result.json")
    if "metrics" in update and update["metrics"] is None:
        # An empty payload exercises every required field at once.
        payload = {}
    else:
        payload.update(update)
    with pytest.raises(ValidationError):
        contracts.ExperimentResult.model_validate(payload)


def test_result_accepts_plain_numbers_and_custom_metric_direction() -> None:
    payload = load_json("reference_result.json")
    payload.update(
        {
            "metric_user": "my_custom_metric",
            "metric_sklearn": "my_custom_metric",
            "metric_direction": "minimize",
            "winner_score": 1,
            "oof_score": 2,
        }
    )
    result = contracts.ExperimentResult.model_validate(payload)
    assert isinstance(result.winner_score, float)
    assert isinstance(result.oof_score, float)
