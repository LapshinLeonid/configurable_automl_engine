"""
Параметризованные тесты для configurable_automl_engine.tuner.

• Smoke-проверяем, что optimize отрабатывает на каждом алгоритме,
  для которого задан search-space (включая CF-18: SGD, GPR, Isotonic,
  ARD, Poisson/Gamma/Tweedie).
• Валидируем пользовательский space-override.
• Проверяем ошибки: неизвестный алгоритм, плохие данные, n_trials ≤ 0.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from configurable_automl_engine import tuner as hyperopt

# ──────────────────────────────────────────────────────────────────────────────
# 0. Делаем пакет видимым и импортируем модуль
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# 1. Фикстура с игрушечными данными
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def toy_data() -> Tuple[pd.DataFrame, pd.Series]:
    X, y = make_regression(
        n_samples=120,
        n_features=10,
        noise=0.1,
        random_state=1,
    )
    return pd.DataFrame(X), pd.Series(y)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Вспомогательная адаптация X/y под «капризные» модели
# ──────────────────────────────────────────────────────────────────────────────
def _prepare_data(algo: str, X: pd.DataFrame, y: pd.Series):
    """Подгоняем форму и диапазон под требования конкретных алгоритмов."""
    # IsotonicRegression — ровно один признак
    if algo == "isotonicregression":
        X = X.iloc[:, [0]]
    # GLM-семейство требует y > 0
    if algo in {"poissonregressor", "gammaregressor", "tweedieregressor"}:
        y = np.abs(y) + 1.0
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# 3. Smoke-тест на каждый поддерживаемый алгоритм
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "algo",
    sorted(k for k, space_fn in hyperopt.ALGO_SPACES.items() if space_fn is not None),
)
def test_optimize_smoke(algo: str, toy_data):
    """optimize не должен падать и возвращает валидные объекты."""
    X, y = _prepare_data(algo, *toy_data)

    model, params, score = hyperopt.optimize(
        algo,
        X,
        y,
        n_trials=3,          # минимально для быстроты
        random_state=0,
    )

    assert isinstance(params, dict) and params, "пустой best_params"
    assert isinstance(score, float) and not np.isnan(score), "score некорректен"
    assert hasattr(model, "predict"), "у модели нет метода predict"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Пользовательский space-override
# ──────────────────────────────────────────────────────────────────────────────
def test_space_override(toy_data):
    X, y = toy_data

    def tiny_space(trial):
        return {"alpha": trial.suggest_float("alpha", 0.0, 0.001)}

    _, params, _ = hyperopt.optimize(
        "elasticnet",
        X,
        y,
        n_trials=2,
        space_overrides={"elasticnet": tiny_space},
    )
    assert params["alpha"] <= 0.001


# ──────────────────────────────────────────────────────────────────────────────
# 5. Негативные сценарии
# ──────────────────────────────────────────────────────────────────────────────
def test_invalid_algo(toy_data):
    X, y = toy_data
    with pytest.raises(hyperopt.InvalidAlgorithmError):
        hyperopt.optimize("does_not_exist", X, y, n_trials=2)


def test_bad_data():
    with pytest.raises(hyperopt.InvalidDataError):
        hyperopt.optimize("ridge", "not-an-array", [1, 2, 3], n_trials=2)


@pytest.mark.parametrize("bad_trials", [0, -3])
def test_non_positive_trials(bad_trials, toy_data):
    X, y = toy_data
    with pytest.raises(ValueError):
        hyperopt.optimize("ridge", X, y, n_trials=bad_trials)
