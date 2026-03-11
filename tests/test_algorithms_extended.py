import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score

from configurable_automl_engine.trainer import ModelTrainer, TrainingError


ALGORITHMS = [
    "elasticnet",
    "dt",
    "sgd",
    "knn",
    "gpr",
    "svr",
    "isotonic",
    "ard",
    "rf",
    "glm",
    pytest.param(
        "xgboost",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("xgboost") is None,
            reason="xgboost не установлен",
        ),
    ),
]

X_full, y_full = load_diabetes(return_X_y=True, as_frame=True)
# Добавим NaN-ы в копию, чтобы проверить импьютер
X_nan = X_full.copy()
X_nan.iloc[0:20, 0] = np.nan


# ---------------------------------------------------------------------
# SUCCESSFUL TRAINING + NaN-handling
# ---------------------------------------------------------------------
@pytest.mark.parametrize("algo", ALGORITHMS)
def test_fit_and_score_with_nan(algo):
    X = X_nan.iloc[:, [0]] if algo == "isotonic" else X_nan
    trainer = ModelTrainer(algorithm=algo).fit(X, y_full)
    preds = trainer.predict(X)
    score = r2_score(y_full, preds)
    assert len(preds) == len(y_full)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------
# BAD HYPERPARAMETERS
# ---------------------------------------------------------------------
BAD_PARAMS = {
    "elasticnet": {"alpha": -1.0},
    "knn": {"n_neighbors": 0},
    "rf": {"n_estimators": 0},
}


@pytest.mark.parametrize("algo,bad_param", BAD_PARAMS.items())
def test_invalid_hyperparams(algo, bad_param):
    with pytest.raises(ValueError):
        ModelTrainer(algorithm=algo, hyperparams=bad_param).fit(X_full, y_full)


# ---------------------------------------------------------------------
# X / y MISMATCH
# ---------------------------------------------------------------------
@pytest.mark.parametrize("algo", ALGORITHMS)
def test_mismatch_dimensions_raises(algo):
    X2 = X_full.iloc[:10]
    y2 = y_full.iloc[:9]
    with pytest.raises(TrainingError):
        ModelTrainer(algorithm=algo).fit(X2, y2)


# ---------------------------------------------------------------------
# TOO FEW RECORDS
# ---------------------------------------------------------------------
@pytest.mark.parametrize("algo", ALGORITHMS)
def test_too_few_records(algo):
    X_small = X_full.iloc[:1]
    y_small = y_full.iloc[:1]
    with pytest.raises(TrainingError):
        ModelTrainer(algorithm=algo).fit(X_small, y_small)


# ---------------------------------------------------------------------
# SAVE / LOAD ROUND-TRIP
# ---------------------------------------------------------------------
@pytest.mark.parametrize("algo", ALGORITHMS)
def test_save_and_load(tmp_path: Path, algo):
    trainer = ModelTrainer(algorithm=algo).fit(
        X_nan.iloc[:, [0]] if algo == "isotonic" else X_nan, y_full
    )
    pkl = tmp_path / f"{algo}.pkl"
    trainer.save(pkl)
    restored = ModelTrainer.load(pkl)
    y_pred1 = trainer.predict(X_full)
    y_pred2 = restored.predict(X_full)
    assert np.allclose(y_pred1, y_pred2)
