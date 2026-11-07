"""Integration tests for ``ModelTrainer`` with ``target_transform``.

Coverage:
    1. ``none`` (default): the ``model`` step keeps the original estimator
       type (a ``TransformedTargetRegressor`` wrapper is absent); the whole
       existing behavior is unchanged.
    2. ``log1p`` on ``y = exp(X β + ε)``: predictions in the original scale;
       the linear model's MAE is noticeably better than without the
       transformation.
    3. Integration combinations: every method × {oversampling, feature
       selection, isotonic, target encoding} fits and predicts finite
       original-scale values (the wrapper sits after ``isotonic_selector``,
       the preprocessor/oversampler see the original ``y``).
    4. Domain errors: ``boxcox`` with ``y <= 0`` → ``ValueError`` from
       ``fit()`` before training; an unknown method → ``TrainingError`` at
       construction.
    5. Extrapolation: predictions far outside the train range are finite
       (the inverse-transform clip).
    6. save/load (pickle and joblib): identical predictions; the artifact
       metadata record the method.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor

from configurable_automl_engine.common.definitions import SerializationFormat
from configurable_automl_engine.common.errors import TrainingError
from configurable_automl_engine.trainer import ModelTrainer

METHODS = ["log1p", "boxcox", "yeo-johnson"]


def _log_normal_data(n: int = 150, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """``y = exp(X β + ε)`` with a multiplicative structure."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 4)
    beta = np.array([1.0, 0.8, -0.5, 0.0])
    y = np.exp(0.5 + X @ beta + 0.05 * rng.randn(n))
    return X, y


def _frame(X: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    return df, pd.Series(y, name="target")


# ──────────────────────────────────────────────────────────────────────────────
#  1. none: no wrapper, existing behavior unchanged
# ──────────────────────────────────────────────────────────────────────────────


def test_none_keeps_original_model_step() -> None:
    X, y = _log_normal_data()
    trainer = ModelTrainer(algorithm="ridge", random_state=42)
    assert trainer.target_transform.name == "none"
    trainer.fit(X, y)
    model_step = trainer.pipeline.named_steps["model"]
    assert not isinstance(model_step, TransformedTargetRegressor)
    assert type(model_step).__name__ == "Ridge"


def test_none_is_the_default_and_runs_the_standard_flow() -> None:
    X, y = _log_normal_data()
    trainer = ModelTrainer(algorithm="ridge", random_state=42)
    trainer.fit(X, y)
    assert trainer.val_score is not None
    assert np.all(np.isfinite(trainer.predict(X[:5])))


# ──────────────────────────────────────────────────────────────────────────────
#  2. log1p quality gain for a multiplicative target
# ──────────────────────────────────────────────────────────────────────────────


def test_log1p_improves_linear_mae_on_log_normal_target() -> None:
    X, y = _log_normal_data(n=200, seed=11)
    n_train = 160
    X_tr, y_tr, X_te, y_te = (
        X[:n_train],
        y[:n_train],
        X[n_train:],
        y[n_train:],
    )
    mae_plain = (
        ModelTrainer(algorithm="ridge", metric="mae", random_state=42)
        .fit(X_tr, y_tr)
        .predict(X_te)
    )
    mae_log = (
        ModelTrainer(
            algorithm="ridge",
            metric="mae",
            random_state=42,
            target_transform="log1p",
        )
        .fit(X_tr, y_tr)
        .predict(X_te)
    )
    mae_plain = float(np.mean(np.abs(mae_plain - y_te)))
    mae_log = float(np.mean(np.abs(mae_log - y_te)))
    assert mae_log < mae_plain


def test_log1p_predicts_original_scale_and_finite() -> None:
    X, y = _log_normal_data()
    trainer = ModelTrainer(algorithm="ridge", random_state=42, target_transform="log1p")
    trainer.fit(X, y)
    assert isinstance(trainer.pipeline.named_steps["model"], TransformedTargetRegressor)
    preds = trainer.predict(X[:10])
    assert np.all(np.isfinite(preds))
    # Original scale: the predictions track the magnitude of y, not log(y).
    assert preds.max() > 1.0


# ──────────────────────────────────────────────────────────────────────────────
#  3. every method × pipeline features
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", METHODS)
def test_with_oversampling(method: str) -> None:
    X, y = _log_normal_data()
    trainer = ModelTrainer(
        algorithm="ridge",
        random_state=42,
        target_transform=method,
        data_oversampling=True,
        data_oversampling_algorithm="random",
        data_oversampling_multiplier=1.5,
    )
    trainer.fit(X, y)
    assert np.all(np.isfinite(trainer.predict(X[:5])))


@pytest.mark.parametrize("method", METHODS)
def test_with_feature_selection(method: str) -> None:
    X, y = _log_normal_data()
    trainer = ModelTrainer(
        algorithm="ridge",
        random_state=42,
        target_transform=method,
        feature_selection_active=True,
    )
    trainer.fit(X, y)
    assert trainer.feature_selection_active_
    assert np.all(np.isfinite(trainer.predict(X[:5])))


@pytest.mark.parametrize("method", METHODS)
def test_with_isotonic(method: str) -> None:
    X, y = _log_normal_data(n=120)
    trainer = ModelTrainer(
        algorithm="isotonic_regression",
        random_state=42,
        target_transform=method,
        hyperparams={"feature_index": 0},
    )
    trainer.fit(X, y)
    preds = trainer.predict(X[:5])
    assert np.all(np.isfinite(preds))


@pytest.mark.parametrize("method", METHODS)
def test_with_target_encoding(method: str) -> None:
    rng = np.random.RandomState(42)
    n = 150
    cat = np.array(["a", "b", "c"])[rng.randint(0, 3, n)]
    num = rng.randn(n)
    y = np.exp(0.5 + 0.7 * num + 0.3 * (cat == "a") + 0.02 * rng.randn(n))
    df = pd.DataFrame({"cat": cat, "num": num})
    trainer = ModelTrainer(
        algorithm="ridge",
        random_state=42,
        target_transform=method,
        encoding_strategy="target",
    )
    trainer.fit(df, y)
    preds = trainer.predict(df.head())
    assert np.all(np.isfinite(preds))


# ──────────────────────────────────────────────────────────────────────────────
#  4. Domain errors and argument validation
# ──────────────────────────────────────────────────────────────────────────────


def test_boxcox_with_non_positive_y_raises_before_training() -> None:
    X, y = _log_normal_data()
    # Shift the target to include zero and negatives.
    y_bad = y - float(y.max())
    assert y_bad.min() <= 0
    trainer = ModelTrainer(
        algorithm="ridge", random_state=42, target_transform="boxcox"
    )
    with pytest.raises(ValueError, match="boxcox"):
        trainer.fit(X, y_bad)
    assert trainer.pipeline is None


def test_log1p_with_y_below_minus_one_raises() -> None:
    X, y = _log_normal_data()
    trainer = ModelTrainer(algorithm="ridge", random_state=42, target_transform="log1p")
    with pytest.raises(ValueError, match="log1p"):
        trainer.fit(X, y - float(y.max()) - 5.0)


def test_yeo_johnson_accepts_negative_y() -> None:
    X, y = _log_normal_data()
    y_signed = y - y.mean()
    trainer = ModelTrainer(
        algorithm="ridge", random_state=42, target_transform="yeo-johnson"
    )
    trainer.fit(X, y_signed)
    assert np.all(np.isfinite(trainer.predict(X[:5])))


def test_unknown_method_rejected_at_construction() -> None:
    with pytest.raises(TrainingError, match="target_transform"):
        ModelTrainer(algorithm="ridge", target_transform="sqrt")


def test_enum_member_and_string_accepted() -> None:
    from configurable_automl_engine.common.definitions import TargetTransform

    a = ModelTrainer(algorithm="ridge", target_transform=TargetTransform.log1p)
    b = ModelTrainer(algorithm="ridge", target_transform="LOG1P")
    assert a.settings.target_transform is TargetTransform.log1p
    assert b.settings.target_transform is TargetTransform.log1p


# ──────────────────────────────────────────────────────────────────────────────
#  5. Extrapolation stays finite
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", METHODS)
def test_far_extrapolation_predicts_finite(method: str) -> None:
    X, y = _log_normal_data()
    trainer = ModelTrainer(algorithm="ridge", random_state=42, target_transform=method)
    trainer.fit(X, y)
    far = X[:5] * 100.0
    preds = trainer.predict(far)
    assert np.all(np.isfinite(preds))


# ──────────────────────────────────────────────────────────────────────────────
#  6. save / load round trip
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", METHODS)
def test_save_load_pickle_same_predictions(method: str, tmp_path: Path) -> None:
    X, y = _log_normal_data()
    trainer = ModelTrainer(algorithm="ridge", random_state=42, target_transform=method)
    trainer.fit(X, y)
    path = tmp_path / "model.pkl"
    trainer.save(path)
    loaded = ModelTrainer.load(path)
    assert np.allclose(loaded.predict(X[:10]), trainer.predict(X[:10]))
    assert loaded.artifact_metadata_["target_transform"] == method


def test_save_load_joblib_same_predictions(tmp_path) -> None:
    X, y = _log_normal_data()
    trainer = ModelTrainer(
        algorithm="ridge",
        random_state=42,
        target_transform="log1p",
        serialization_format=SerializationFormat.joblib,
    )
    trainer.fit(X, y)
    path = tmp_path / "model.joblib"
    trainer.save(path)
    # Also verify raw joblib.load to prove the wrapper survives joblib.
    raw = joblib.load(path)
    assert np.allclose(raw.predict(X[:10]), trainer.predict(X[:10]))
    loaded = ModelTrainer.load(path)
    assert np.allclose(loaded.predict(X[:10]), trainer.predict(X[:10]))
    # The raw pickle module is compatible too: the trainer state pickles.
    assert pickle.loads(pickle.dumps(trainer)).settings.target_transform.value == (
        "log1p"
    )


def test_config_hash_depends_on_target_transform() -> None:
    from configurable_automl_engine.trainer_artifact import (
        _collect_artifact_metadata,
    )

    X, y = _log_normal_data(n=30, seed=5)
    a = ModelTrainer(algorithm="ridge", random_state=42)
    a.fit(X, y)
    b = ModelTrainer(algorithm="ridge", random_state=42, target_transform="log1p")
    b.fit(X, y)
    meta_a = _collect_artifact_metadata(a.settings)
    meta_b = _collect_artifact_metadata(b.settings)
    assert meta_a["config_hash"] != meta_b["config_hash"]
    assert meta_a["target_transform"] == "none"
    assert meta_b["target_transform"] == "log1p"
