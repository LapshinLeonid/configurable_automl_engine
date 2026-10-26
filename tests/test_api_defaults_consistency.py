"""The direct Python API and the engine config share the same defaults.

``tuner.optimize`` and ``ModelTrainer`` expose the settings of the
``general`` config section as keyword arguments. A direct call without these
arguments must behave like a config run without the corresponding keys, so
their defaults are compared with the ``GeneralCfg`` field defaults here.
"""

from __future__ import annotations

import inspect

import pytest

from configurable_automl_engine.common.complexity_guard import ComplexityGuardConfig
from configurable_automl_engine.models import create_model
from configurable_automl_engine.trainer import ModelTrainer
from configurable_automl_engine.training_engine.config_parser import (
    ComplexityGuardCfg,
    GeneralCfg,
)
from configurable_automl_engine.tuner import _OptimizeRunner, optimize


def _general_default(field: str) -> object:
    """The default of a ``GeneralCfg`` field as a plain value."""
    value = GeneralCfg.model_fields[field].default
    return getattr(value, "value", value)


def _param_default(func: object, name: str) -> object:
    """The default of a function parameter as a plain value."""
    value = inspect.signature(func).parameters[name].default  # type: ignore[arg-type]
    return getattr(value, "value", value)


# (general field, optimize parameter, ModelTrainer parameter)
SHARED_SETTINGS = [
    ("validation_strategy", "val_method", "validation_strategy"),
    ("n_folds", "n_folds", "n_folds"),
    ("random_state", "random_state", "random_state"),
    ("categorical_encoding", None, "encoding_strategy"),
    (
        "high_cardinality_threshold",
        "high_cardinality_threshold",
        "high_cardinality_threshold",
    ),
    (
        "high_cardinality_encoding",
        "high_cardinality_encoding",
        "high_cardinality_encoding",
    ),
    ("hashing_n_components", "hashing_n_components", "hashing_n_components"),
    (
        "target_encoding_smoothing",
        "target_encoding_smoothing",
        "target_encoding_smoothing",
    ),
    (
        "target_encoding_fallback",
        "target_encoding_fallback",
        "target_encoding_fallback",
    ),
    ("inf_policy", "inf_policy", "inf_policy"),
    ("datetime_policy", "datetime_policy", "datetime_policy"),
    ("drop_id_like_columns", "drop_id_like_columns", "drop_id_like_columns"),
    ("serialization_format", None, "serialization_format"),
    ("comparison_metric", "metric", "metric"),
]


@pytest.mark.parametrize("field, opt_param, _trainer_param", SHARED_SETTINGS)
def test_optimize_defaults_match_config(
    field: str, opt_param: str | None, _trainer_param: str | None
) -> None:
    """``optimize()`` defaults equal the ``general`` config defaults."""
    if opt_param is None:
        pytest.skip(f"optimize() has no counterpart of general.{field}")
    assert _param_default(optimize, opt_param) == _general_default(field)


@pytest.mark.parametrize("field, _opt_param, trainer_param", SHARED_SETTINGS)
def test_model_trainer_defaults_match_config(
    field: str, _opt_param: str | None, trainer_param: str | None
) -> None:
    """``ModelTrainer`` defaults equal the ``general`` config defaults."""
    if trainer_param is None:
        pytest.skip(f"ModelTrainer has no counterpart of general.{field}")
    assert _param_default(ModelTrainer, trainer_param) == _general_default(field)


def test_optimize_complexity_guard_defaults_match_config() -> None:
    """``complexity_guard_cfg=None`` means the ``general.complexity_guard`` defaults."""
    runner = _OptimizeRunner(
        algo_name="random_forest",
        X=None,
        y=None,
        data_oversampling=False,
        data_oversampling_multiplier=1.0,
        data_oversampling_algorithm="random",
        metric="r2",
        val_method="auto",
        validation_strategy=None,
        n_folds=5,
        n_trials=1,
        random_state=42,
        sampler_seed=None,
        train_test_split_test_size=0.2,
        space_overrides=None,
        initial_params=None,
        preprocessor=None,
        categorical_features=None,
        numerical_features=None,
        encoding=None,
        preprocessing_override=None,
        pruning=None,
        high_cardinality_threshold=None,
        high_cardinality_encoding=None,
        hashing_n_components=16,
        target_encoding_smoothing=20.0,
        target_encoding_fallback=None,
        feature_selection_cfg=None,
        complexity_guard_cfg=None,
    )
    assert runner.complexity_guard_cfg == ComplexityGuardConfig.from_dict(
        ComplexityGuardCfg()
    )
    assert runner.complexity_guard_cfg.mode == "apply"


def test_create_model_rejects_unknown_hyperparameters() -> None:
    """An unknown name is rejected like in the config validation."""
    with pytest.raises(ValueError, match="unknown hyperparameters"):
        create_model("ridge", alpha=1.0, totally_fake_param=42)


def test_create_model_accepts_legacy_names_and_random_state() -> None:
    """Legacy names are remapped; ``random_state`` is dropped where unsupported."""
    model = create_model("ardregression", n_iter=300)
    assert model.get_params()["max_iter"] == 300
    knn = create_model("nearest_neighbors_regression", n_neighbors=3, random_state=1)
    assert "random_state" not in knn.get_params()
