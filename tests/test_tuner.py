"""
Параметризованные тесты для configurable_automl_engine.tuner.

• Smoke-проверяем, что optimize отрабатывает на каждом алгоритме,
  для которого задан search-space (включая CF-18: SGD, GPR, Isotonic,
  ARD, Poisson/Gamma/Tweedie).
• Валидируем пользовательский space-override.
• Проверяем ошибки: неизвестный алгоритм, плохие данные, n_trials ≤ 0.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from configurable_automl_engine import tuner as hyperopt
from configurable_automl_engine.oversampling import DataOversampler
from imblearn.pipeline import Pipeline as ImbPipeline

from unittest.mock import MagicMock, patch
from configurable_automl_engine.tuner import (
    _apply_dynamic_space,
    _build_scorer, 
    HyperoptError
    )

import optuna
from optuna.trial import FixedTrial


from configurable_automl_engine.tuner import _can_stratify
from configurable_automl_engine.tuner import optimize


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

def test_apply_dynamic_space_types(toy_data):
    """
    Покрывает логику _apply_dynamic_space.
    Используем алгоритм 'knn', так как для него в коде есть фабрика пространств.
    """
    X, y = toy_data
    
    class MockEntry:
        def __init__(self, bounds):
            self.bounds = bounds
    # Имитируем структуру из YAML для KNN
    # Это покроет ветки int, float, float_log, categorical и константы
    dynamic_config = {
        "n_neighbors": MockEntry([5, 15, "int"]),
        "p": MockEntry([1, 2, "int"]),
        "weights": MockEntry([["uniform", "distance"], None, "categorical"]),
        "leaf_size": 30  # Константа (строка 144)
    }
    model, params, score = hyperopt.optimize(
        "knn",
        X, y,
        n_trials=2,
        space_overrides={"knn": dynamic_config}
    )
    
    assert isinstance(params["n_neighbors"], int)
    assert params["weights"] in ["uniform", "distance"]
    # Проверяем, что константа попала в модель
    assert model.leaf_size == 30


def test_validate_data_mismatch():
    """Проверка ошибки при несовпадении длин X и y (строка 159)."""
    X = np.zeros((10, 2))
    y = np.zeros(5)
    with pytest.raises(hyperopt.InvalidDataError, match="Размеры не совпадают"):
        hyperopt._validate_data(X, y)

def test_validate_data_invalid_y_type():
    """Проверка ошибки при недопустимом типе y (строка 155)."""
    X = np.zeros((5, 2))
    y = {1: 0, 2: 0} # dict не входит в ok_types
    with pytest.raises(hyperopt.InvalidDataError, match="y должен быть"):
        hyperopt._validate_data(X, y)

def test_optimize_with_oversampling(toy_data):
    """
    Покрывает логику включения оверсэмплинга в Pipeline.
    """
    X, y = toy_data
    y_bin = (y > y.mean()).astype(int) 
    model, params, score = hyperopt.optimize(
        "knn",
        X, y_bin,
        data_oversampling=True,
        data_oversampling_algorithm="random",
        data_oversampling_multiplier=1.2,
        n_trials=2
    )

    print(model.steps)
    assert isinstance(model, ImbPipeline)
    assert any(isinstance(step[1], DataOversampler) for step in model.steps)

def test_can_stratify_negative():
    """Проверка условий, когда стратификация невозможна (строки 182-183)."""
    # Много уникальных значений (регрессия)
    y_reg = np.linspace(0, 1, 100)
    assert hyperopt._can_stratify(y_reg) is False
    
    # Многомерный y
    y_multi = np.zeros((10, 2))
    assert hyperopt._can_stratify(y_multi) is False

def test_split_train_test_fallback():
    """Проверка fallback в train_test_split при ошибке стратификации (строки 194-196)."""
    # Создаем ситуацию, где стратификация невозможна из-за 1 экземпляра класса
    X = np.random.rand(10, 2)
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) 
    
    # Метод должен отработать без ошибок, поймав ValueError внутри
    res = hyperopt._split_train_test(X, y, test_size=0.5)
    assert len(res) == 4


def test_optimize_pruning(toy_data):
    """
    Покрывает блок обработки исключений в _objective.
    Исправлено: параметры теперь регистрируются через trial.suggest_int.
    """
    X, y = toy_data
    
    def pruning_space(trial):
        # Используем suggest_int, чтобы optuna зафиксировала параметры в trial.params
        if trial.number == 0:
            n_neighbors = trial.suggest_int("n_neighbors", 5, 5)
            return {"n_neighbors": n_neighbors, "weights": "uniform", "p": 2}
        
        # Вторая попытка вызовет ошибку (n_neighbors <= 0)
        n_neighbors = trial.suggest_int("n_neighbors", 0, 0)
        return {"n_neighbors": n_neighbors, "weights": "uniform", "p": 2}
    # Метод не должен упасть, Trial 1 просто будет помечен как Pruned
    model, params, score = hyperopt.optimize(
        "knn", X, y,
        n_trials=2,
        # Передаем как словарь для конкретного алгоритма
        space_overrides={"knn": pruning_space}
    )
    
    # Теперь params не будет пустым, так как Trial 0 успешно завершился
    assert "n_neighbors" in params
    assert params["n_neighbors"] == 5

def test_knn_space_dynamic_limit():
    """Проверка динамического ограничения k в KNN (строки 80-81)."""
    # Для 10 сэмплов 80% это 8. max_k должен быть 8.
    space_fn = hyperopt._make_knn_space(n_samples=10)
    

    # Эмулируем запрос n_neighbors
    trial = FixedTrial({"n_neighbors": 5, "weights": "uniform", "p": 1})
    params = space_fn(trial)
    assert params["n_neighbors"] <= 8

def test_apply_dynamic_space_floats():
    trial = MagicMock()
    
    # Имитируем структуру SearchSpaceEntry из Pydantic
    class MockEntry:
        def __init__(self, bounds):
            self.bounds = bounds
    space_dict = {
        "learning_rate": MockEntry([0.01, 0.1, "float"]),
        "gamma": MockEntry([1e-5, 1e-1, "float_log"]),
        "constant": 42
    }
    _apply_dynamic_space(trial, space_dict)
    # Проверяем вызовы Optuna
    trial.suggest_float.assert_any_call("learning_rate", 0.01, 0.1)
    trial.suggest_float.assert_any_call("gamma", 1e-5, 1e-1, log=True)

def test_build_scorer_error():
    with pytest.raises(HyperoptError, match="Неизвестная метрика"):
        _build_scorer("non_existent_metric_name_123")

def test_can_stratify_pandas():
    y_series = pd.Series([0, 1, 0, 1])
    assert _can_stratify(y_series) is True
    
    y_df = pd.DataFrame({"target": [0, 1, 0, 1]})
    assert _can_stratify(y_df) is False  # Т.к. ndim != 1 для DF с 1 колонкой (обычно)

def test_optimize_train_test_split_mode(toy_data):
    X, y = toy_data
    # Принудительно вызываем режим hold-out через передачу малого количества данных 
    # или явное указание стратегии
    model, params, score = optimize(
        "knn", X, y,
        validation_strategy="train_test_split",
        n_trials=1
    )
    assert score is not None

def test_optimize_raises_when_no_search_space(monkeypatch):
    """Проверяет выброс HyperoptError если для алгоритма нет search-space."""

    # --- 1. Подготавливаем фиктивные данные ---
    X = np.random.rand(20, 3)
    y = np.random.rand(20)

    # --- 2. Мокаем create_model → чтобы алгоритм считался валидным ---
    def fake_create_model(algo, **kwargs):
        class DummyModel:
            def fit(self, X, y):
                pass

            def predict(self, X):
                return np.zeros(len(X))

        return DummyModel()

    # --- делаем алгоритм валидным ---
    monkeypatch.setattr(
        hyperopt,
        "_get_estimator",
        lambda algo: True
    )

    # --- гарантируем отсутствие search-space ---
    monkeypatch.setitem(
        hyperopt.ALGO_SPACES,
        "fake_algo",
        None
    )

    with pytest.raises(HyperoptError, match="нет search-space"):
        optimize(
            "fake_algo",
            X,
            y,
            n_trials=1
        )


class TestTunerObjective:
    @pytest.fixture
    def dummy_data(self):
        return pd.DataFrame({'a': [1, 2, 3, 4, 5]}), pd.Series([1, 0, 1, 0, 1])
    @pytest.fixture
    def mock_space(self):
        return {"rf": lambda trial: {"n_estimators": 10}}
    def test_objective_trigger_fallback(self, dummy_data, mock_space):
        """
        Тест принудительно заставляет np.isfinite вернуть False,
        чтобы проверить возврат константы.
        """
        X, y = dummy_data
        EXPECTED_FALLBACK = -3.4028235e+38
        # 1. Патчим ВСЁ окружение, чтобы ни одна реальная функция не выполнилась
        with patch('configurable_automl_engine.tuner.model_selection.cross_val_score') as mock_cv, \
             patch('configurable_automl_engine.tuner.create_model') as mock_create, \
             patch('configurable_automl_engine.tuner._build_scorer') as mock_scorer_factory, \
             patch('configurable_automl_engine.tuner.make_cv') as mock_make_cv, \
             patch('configurable_automl_engine.tuner.np.isfinite') as mock_finite, \
             patch('configurable_automl_engine.tuner._validate_data'), \
             patch('configurable_automl_engine.tuner._get_estimator'):
            # ГАРАНТИРУЕМ:
            # 1. Мы в ветке k-fold
            mock_make_cv.return_value = ("k_fold", MagicMock())
            # 2. cross_val_score возвращает что-то
            mock_cv.return_value = np.array([0.5])
            # 3. КРИТИЧЕСКИЙ МОМЕНТ: Любое число НЕ конечное
            mock_finite.return_value = False 
            
            # Остальные заглушки
            mock_create.return_value = MagicMock()
            mock_scorer_factory.return_value = MagicMock()
            # Вызываем
            _, _, best_score = optimize(
                algo_name="rf",
                X=X,
                y=y,
                n_trials=1,
                space_overrides=mock_space
            )
            # Проверяем
            assert best_score == EXPECTED_FALLBACK
            assert mock_finite.called
    def test_objective_via_actual_nan(self, dummy_data, mock_space):
        """
        Тест через подмену np.mean (более естественный путь).
        Если в tuner.py: 'import numpy as np', патчим 'np.mean'.
        Если 'from numpy import mean', патчим 'mean'.
        """
        X, y = dummy_data
        
        # Попробуем запатчить mean в пространстве имен модуля tuner
        with patch('configurable_automl_engine.tuner.model_selection.cross_val_score') as mock_cv, \
             patch('configurable_automl_engine.tuner.create_model'), \
             patch('configurable_automl_engine.tuner._build_scorer'), \
             patch('configurable_automl_engine.tuner.make_cv') as mock_make_cv, \
             patch('configurable_automl_engine.tuner.np.mean', return_value=np.nan), \
             patch('configurable_automl_engine.tuner._validate_data'), \
             patch('configurable_automl_engine.tuner._get_estimator'):
            mock_make_cv.return_value = ("k_fold", MagicMock())
            mock_cv.return_value = np.array([1.0]) # значение не важно, так как mean вернет nan
            _, _, best_score = optimize(
                algo_name="rf",
                X=X,
                y=y,
                n_trials=1,
                space_overrides=mock_space
            )
            assert best_score == -3.4028235e+38