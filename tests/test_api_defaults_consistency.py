"""The direct Python API and the engine config share the same defaults.

``tuner.optimize`` and ``ModelTrainer`` expose the settings of the
``general`` config section as keyword arguments. A direct call without these
arguments must behave like a config run without the corresponding keys, so
their defaults are compared with the ``GeneralCfg`` field defaults here.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from configurable_automl_engine.common.complexity_guard import ComplexityGuardConfig
from configurable_automl_engine.models import create_model
from configurable_automl_engine.trainer import ModelTrainer, train_model
from configurable_automl_engine.training_engine.config_parser import (
    ComplexityGuardCfg,
    GeneralCfg,
)
from configurable_automl_engine.tuner import optimize
from configurable_automl_engine.hpo.request import HpoRequest, as_complexity_guard_config


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
    request = HpoRequest(algo_name="random_forest")
    expected = ComplexityGuardConfig.from_dict(ComplexityGuardCfg())
    # optimize(complexity_guard_cfg=None) normalizes through the same helper.
    assert as_complexity_guard_config(None) == expected
    assert request.complexity_guard == expected
    assert request.complexity_guard.mode == "apply"



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


@pytest.mark.parametrize(
    ("algorithm", "params"),
    [
        ("ridge", {"alpha": 1.0, "use_feature_selection": True}),
        ("ridge", {"alpha": 1.0, "use_feature_selection": False}),
        ("isotonic_regression", {"feature_index": 0, "isotonic_feature": "x0"}),
    ],
)
def test_model_trainer_accepts_hpo_result_params(
    algorithm: str, params: dict[str, object]
) -> None:
    """The auxiliary keys of the HPO result ``params`` do not break training.

    ``use_feature_selection`` (feature_selection.mode='auto') and
    ``isotonic_feature`` (an isotonic winner) are not model hyperparameters;
    ``ModelTrainer`` and ``train_model`` drop them like the engine and the
    tuner instead of failing on the strict ``create_model`` name check.
    """
    rng = np.random.default_rng(0)
    X = rng.random((60, 3))
    y = 2 * X[:, 0] + 0.1 * rng.random(60)

    trainer = ModelTrainer(algorithm=algorithm, hyperparams=dict(params), metric="r2")
    trainer.fit(X, y)
    assert trainer.pipeline is not None
    assert trainer.hyperparams == params

    score = train_model(algorithm, "r2", dict(params), X, y)
    assert np.isfinite(score)
