"""Direct unit tests of the pure ``ModelTrainer.fit`` steps and helpers.

Every function is called with plain values — no ``ModelTrainer`` is
constructed or mocked (the pipeline steps come from a local ``make_steps``
factory). ``assert_pure`` checks that the inputs are unchanged after the call
(data frames are compared with a copy taken before the call) and that two
calls give equal results.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

from configurable_automl_engine.common.artifact_metadata import (
    _collect_artifact_metadata,
    check_artifact_format,
)
from configurable_automl_engine.common.definitions import SerializationFormat
from configurable_automl_engine.common.settings import (
    EncodingSettings,
    InputPolicies,
    OversamplingSettings,
)
from configurable_automl_engine.trainer_data import (
    _align_predict_columns,
    _detect_feature_types,
    _isotonic_n_numeric,
    _isotonic_n_total_estimate,
    _resolve_isotonic_feature_index,
    build_trainer_preprocessor,
    extract_feature_names,
    load_training_data,
    log_preprocessed_matrix,
    prepare_features,
)
from configurable_automl_engine.trainer_errors import TrainingError
from configurable_automl_engine.trainer_feature_selection import (
    _resolve_feature_selection_active,
    _run_feature_selection_auto_check,
)
from configurable_automl_engine.trainer_params import build_trainer_settings
from configurable_automl_engine.trainer_scoring import (
    _align_oof_vector,
    _finalize_oof,
    score_on_validation_splits,
)
from configurable_automl_engine.trainer_state import ARTIFACT_FORMAT_VERSION
from configurable_automl_engine.training_engine.config_parser import (
    FeatureSelectionCfg,
)
from tests.purity import assert_pure

ENCODING = EncodingSettings(encoding="one_hot")


def _frame(n: int = 30, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "city": rng.choice(["a", "b"], size=n),
            "const": np.zeros(n),
        }
    )
    y = pd.Series(2.0 * X["x1"] + rng.normal(scale=0.1, size=n), name="y")
    return X, y


def _features(X: pd.DataFrame) -> Any:
    return prepare_features(
        X,
        feature_names=list(X.columns),
        categorical_features=None,
        numerical_features=None,
        id_column=None,
        policies=InputPolicies(),
    )


def _make_steps(preprocessor: Any, model: Any) -> Any:
    """A minimal steps factory: fresh clones, an optional selector."""
    from configurable_automl_engine.feature_selection import FeatureSelector

    def make_steps(feature_selection_active: bool, X_for_estimate: Any) -> list:
        steps: list[tuple[str, Any]] = [("preprocessor", clone(preprocessor))]
        if feature_selection_active:
            steps.append(("feature_selector", FeatureSelector(random_state=0)))
        steps.append(("model", clone(model)))
        return steps

    return make_steps


def _preprocessor(features: Any) -> Any:
    return build_trainer_preprocessor(
        "ridge",
        features.categorical_features,
        features.numerical_features,
        feature_names=features.feature_names,
        encoding=ENCODING,
        override=None,
        random_state=0,
    ).preprocessor


# ──────────────────────────────────────────────────────────────────────────────
#  Input data
# ──────────────────────────────────────────────────────────────────────────────


def test_load_training_data() -> None:
    X, y = _frame()
    df = X.assign(target=y)

    data = assert_pure(load_training_data, df, "target")

    assert "target" not in data.X.columns and "target" in df.columns
    assert data.feature_names == ["x1", "x2", "city", "const"]
    pd.testing.assert_series_equal(data.y, df["target"])
    with pytest.raises(TrainingError, match="Insufficient records"):
        load_training_data(X.head(1), y.head(1))
    with pytest.raises(TrainingError, match="Unsupported data type"):
        load_training_data([1, 2], [1, 2])


def test_prepare_features_cleans_and_detects_in_the_result() -> None:
    X, _ = _frame()
    X["when"] = pd.Timestamp("2021-01-01") + pd.to_timedelta(np.arange(len(X)), "D")

    features = assert_pure(
        prepare_features,
        X,
        feature_names=list(X.columns),
        categorical_features=None,
        numerical_features=None,
        id_column=None,
        policies=InputPolicies(),
        warn_id_like=False,
    )

    assert "const" in X.columns and "when" in X.columns  # input untouched
    # const, and the components constant over 30 days (year, month).
    assert features.dropped_features[0] == "const"
    assert "when" in features.datetime_expansion
    assert "when" not in features.feature_names
    assert features.categorical_features == ["city"]
    assert {"x1", "x2"} <= set(features.numerical_features)
    assert list(features.X.columns) == features.feature_names


def test_prepare_features_reconciles_explicit_lists() -> None:
    X, _ = _frame()
    numerical = ["x1", "x2", "const"]

    features = assert_pure(
        prepare_features,
        X,
        feature_names=list(X.columns),
        categorical_features=["city"],
        numerical_features=numerical,
        id_column=None,
        policies=InputPolicies(),
    )

    assert features.numerical_features == ["x1", "x2"]
    assert numerical == ["x1", "x2", "const"]


def test_prepare_features_ndarray() -> None:
    X = np.arange(12, dtype=float).reshape(6, 2)
    features = assert_pure(
        prepare_features,
        X,
        feature_names=None,
        categorical_features=None,
        numerical_features=None,
        id_column=None,
        policies=InputPolicies(),
    )
    assert features.numerical_features == ["col_0", "col_1"]
    assert features.feature_names == ["col_0", "col_1"]
    assert features.categorical_features == []


def test_detect_feature_types_and_extract_names() -> None:
    X, _ = _frame()
    assert assert_pure(_detect_feature_types, X, None, None, "const") == (
        ["city"],
        ["x1", "x2"],
    )
    with pytest.raises(TrainingError, match="Specified columns not found"):
        _detect_feature_types(X, ["nope"], [], None)
    assert assert_pure(extract_feature_names, X) == list(X.columns)
    assert extract_feature_names(np.zeros((2, 3))) == ["col_0", "col_1", "col_2"]
    assert extract_feature_names(np.zeros(3)) is None


def test_build_trainer_preprocessor(caplog: pytest.LogCaptureFixture) -> None:
    X, _ = _frame()
    features = _features(X)

    with caplog.at_level(logging.INFO):
        setup = assert_pure(
            build_trainer_preprocessor,
            "ridge",
            features.categorical_features,
            features.numerical_features,
            feature_names=features.feature_names,
            encoding=ENCODING,
            override=None,
            random_state=0,
        )

    assert isinstance(setup.preprocessor, ColumnTransformer)
    assert setup.preset.scaling == "standard"
    assert "Resolved preprocessing preset for algorithm 'ridge'" in caplog.text


def test_isotonic_helpers() -> None:
    X, _ = _frame()
    assert assert_pure(_resolve_isotonic_feature_index, {"feature_index": 2}) == 2
    with pytest.raises(TrainingError, match="feature_index"):
        _resolve_isotonic_feature_index({"feature_index": -1})
    assert assert_pure(_isotonic_n_numeric, ["a", "b"]) == 2
    assert _isotonic_n_numeric(None) == 0
    n_total = assert_pure(
        _isotonic_n_total_estimate,
        X,
        categorical_features=["city"],
        numerical_features=["x1", "x2"],
        encoding=ENCODING,
    )
    assert n_total == 4  # one-hot city (2) + 2 numeric


def test_align_predict_columns() -> None:
    X, _ = _frame()
    aligned = assert_pure(_align_predict_columns, ["x2", "x1"], X)
    assert list(aligned.columns) == ["x2", "x1"]
    with pytest.raises(TrainingError, match="Missing columns"):
        _align_predict_columns(["nope"], X)


def test_log_preprocessed_matrix(caplog: pytest.LogCaptureFixture) -> None:
    X, y = _frame()
    features = _features(X)
    preprocessor = _preprocessor(features).fit(features.X, y)
    with caplog.at_level(logging.INFO):
        assert_pure(log_preprocessed_matrix, preprocessor, features.X)
    assert "preprocessed matrix: 30 × " in caplog.text


# ──────────────────────────────────────────────────────────────────────────────
#  Validation scoring and feature selection
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("strategy", "metric"),
    [("k_fold", "rmse"), ("k_fold", "mae"), ("train_test_split", "r2"), ("loo", "rmse")],
)
def test_score_on_validation_splits_is_pure(strategy: str, metric: str) -> None:
    X, y = _frame()
    features = _features(X)
    make_steps = _make_steps(_preprocessor(features), Ridge(alpha=0.5))

    scores = assert_pure(
        score_on_validation_splits,
        features.X,
        y,
        make_steps,
        feature_selection_active=False,
        metric=metric,
        additional_metrics=["mae"] if metric != "mae" else [],
        validation_strategy=strategy,
        n_folds=3,
        test_size=0.2,
        random_state=0,
        algorithm="ridge",
    )

    assert np.isfinite(scores.val_score)
    assert scores.oof_score is not None and scores.oof_score > 0
    assert 0 < scores.oof_coverage <= 1
    assert isinstance(scores.oof_predictions, pd.Series)
    if metric != "mae":
        assert set(scores.additional_scores) == {"mae"}


def test_score_on_validation_splits_all_folds_fail() -> None:
    X, y = _frame()

    def broken_steps(feature_selection_active: bool, X_for_estimate: Any) -> list:
        raise ValueError("cannot build")

    with pytest.raises(TrainingError, match="Validation scoring failed for all folds"):
        score_on_validation_splits(
            X,
            y,
            broken_steps,
            feature_selection_active=False,
            metric="rmse",
            additional_metrics=[],
            validation_strategy="k_fold",
            n_folds=3,
            test_size=0.2,
            random_state=0,
            algorithm="ridge",
        )


def test_finalize_and_align_oof() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=[10, 11, 12])
    pred = np.array([1.0, np.nan, 3.0])
    true = np.array([1.5, 2.0, 2.5])

    vector, score, coverage = assert_pure(
        _finalize_oof, pred, true, X, 3, method="k_fold"
    )

    assert list(vector.index) == [10, 11, 12]
    assert coverage == pytest.approx(2 / 3)
    assert score == pytest.approx(0.5)
    assert isinstance(assert_pure(_align_oof_vector, pred, X), pd.Series)


def test_resolve_feature_selection_active() -> None:
    X, y = _frame(n=40)
    features = _features(X)
    make_steps = _make_steps(_preprocessor(features), Ridge())
    common: dict[str, Any] = {"metric": "rmse", "random_state": 0}

    assert not assert_pure(
        _resolve_feature_selection_active,
        features.X,
        y,
        make_steps,
        algorithm="isotonic_regression",
        explicit_flag=True,
        fs_cfg=FeatureSelectionCfg(mode="always"),
        **common,
    )
    assert _resolve_feature_selection_active(
        features.X,
        y,
        make_steps,
        algorithm="ridge",
        explicit_flag=None,
        fs_cfg=FeatureSelectionCfg(mode="always"),
        **common,
    )
    assert not _resolve_feature_selection_active(
        features.X,
        y,
        make_steps,
        algorithm="ridge",
        explicit_flag=False,
        fs_cfg=FeatureSelectionCfg(mode="always"),
        **common,
    )
    auto = assert_pure(
        _resolve_feature_selection_active,
        features.X,
        y,
        make_steps,
        algorithm="ridge",
        explicit_flag=None,
        fs_cfg=FeatureSelectionCfg(mode="auto"),
        **common,
    )
    assert isinstance(auto, bool)


def test_feature_selection_auto_check_small_data(caplog) -> None:
    X, y = _frame(n=8)
    features = _features(X)
    with caplog.at_level(logging.WARNING):
        result = assert_pure(
            _run_feature_selection_auto_check,
            features.X,
            y,
            _make_steps(_preprocessor(features), Ridge()),
            metric="rmse",
            random_state=None,
        )
    assert result is False
    assert "auto-check skipped" in caplog.text


# ──────────────────────────────────────────────────────────────────────────────
#  Settings and artifact metadata
# ──────────────────────────────────────────────────────────────────────────────


def _settings(**overrides: Any) -> Any:
    values: dict[str, Any] = {
        "algorithm": "Ridge",
        "hyperparams": {"alpha": 0.5},
        "metric": "RMSE",
        "random_state": 3,
        "serialization_format": SerializationFormat.pickle,
        "categorical_features": None,
        "numerical_features": None,
        "id_column": None,
        "encoding_strategy": "one_hot",
        "high_cardinality_threshold": 50,
        "high_cardinality_encoding": "frequency",
        "hashing_n_components": 16,
        "target_encoding_smoothing": 20.0,
        "target_encoding_fallback": None,
        "additional_metrics": ["MAE"],
        "preprocessing_override": None,
        "feature_selection_cfg": None,
        "feature_selection_active": None,
        "validation_strategy": "K_FOLD",
        "n_folds": 3,
        "test_size": 0.2,
        "oversampling": OversamplingSettings(),
        "policies": InputPolicies(),
        "warn_id_like": True,
    }
    values.update(overrides)
    return build_trainer_settings(**values)


def test_build_trainer_settings() -> None:
    settings = assert_pure(_settings)
    assert settings.metric == "rmse"
    assert settings.validation_strategy == "k_fold"
    assert settings.additional_metrics == ["mae"]
    assert settings.feature_selection_cfg == FeatureSelectionCfg()
    with pytest.raises(TrainingError):
        _settings(test_size=2.5)


def test_collect_artifact_metadata_and_format_check() -> None:
    settings = _settings()
    first = _collect_artifact_metadata(settings)
    second = _collect_artifact_metadata(settings)

    assert first["artifact_format_version"] == ARTIFACT_FORMAT_VERSION
    assert first["config_hash"] == second["config_hash"]
    assert first["algorithm"] == "ridge" and first["params"] == {"alpha": 0.5}
    assert {k: v for k, v in first.items() if k != "created_at_utc"} == {
        k: v for k, v in second.items() if k != "created_at_utc"
    }
    assert settings == _settings()

    assert assert_pure(check_artifact_format, first) is None
    with pytest.raises(TrainingError, match="artifact format 1"):
        check_artifact_format({"engine_version": "0.4.1"})
    with pytest.raises(TrainingError, match="<= 0.4.x"):
        check_artifact_format(None)


# ──────────────────────────────────────────────────────────────────────────────
#  The trainer keeps a few grouped results
# ──────────────────────────────────────────────────────────────────────────────


def test_fitted_trainer_state_is_small_and_documented_attributes_readable() -> None:
    """A fitted ``ModelTrainer`` has at most 25 instance attributes, and the
    documented attributes are readable under their names."""
    from configurable_automl_engine.preprocessing_presets import PreprocessingPreset
    from configurable_automl_engine.trainer import ModelTrainer

    X, y = _frame()
    X["when"] = pd.Timestamp("2021-01-01") + pd.to_timedelta(
        np.arange(len(X)) * 40, "D"
    )
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="rmse",
        additional_metrics=["mae"],
        validation_strategy="k_fold",
        n_folds=3,
    ).fit(X, y)

    assert len(vars(trainer)) <= 25, sorted(vars(trainer))
    assert np.isfinite(trainer.val_score)
    assert set(trainer.additional_scores) == {"mae"}
    assert len(trainer.oof_predictions_) == len(X)
    assert "when" in trainer.datetime_expansion_
    assert isinstance(trainer.preprocessing_preset, PreprocessingPreset)
    assert trainer.artifact_metadata_ is None  # populated by save()
