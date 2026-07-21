import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from sklearn.linear_model import Ridge
from configurable_automl_engine.common.hyperopt_defaults import FloatSpace

from configurable_automl_engine.common.hyperopt_defaults import (
    SearchSpaceEntry, 
    clip_search_space,
    IntSpace
)
from configurable_automl_engine.tuner import _make_knn_space, optimize

def test_clip_search_space_logic():
    """
    Проверяет базовую логику клиппинга:
    1. n_neighbors ограничивается n_samples - 1
    2. min_samples_leaf ограничивается n_samples
    3. Low подтягивается к High, если High стал меньше исходного Low
    """
    space = {
        "n_neighbors": SearchSpaceEntry.model_validate([5, 100, "int"]),
        "min_samples_leaf": SearchSpaceEntry.model_validate([10, 20, "int"])
    }
    
    # n_samples = 3
    # n_neighbors: limit = 2. New high = 2. New low = min(5, 2) = 2.
    # min_samples_leaf: limit = 3. New high = 3. New low = min(10, 3) = 3.
    clipped = clip_search_space(space, n_samples=3)
    
    assert clipped["n_neighbors"].high == 2
    assert clipped["n_neighbors"].low == 2
    assert isinstance(clipped["n_neighbors"].config, IntSpace)
    
    assert clipped["min_samples_leaf"].high == 3
    assert clipped["min_samples_leaf"].low == 3 # Исправлено: low теперь 3, а не 1

def test_knn_factory_clipping():
    """Проверяет, что фабрика KNN передает в Optuna правильные границы."""
    # Сценарий: 2 образца. Физический лимит n_neighbors = 1.
    space_fn = _make_knn_space(n_samples=2)
    
    trial = MagicMock()
    # Настраиваем мок, чтобы он возвращал значение в рамках границ
    trial.suggest_int.return_value = 1
    
    _ = space_fn(trial)
    
    # Проверяем, что suggest_int вызван с границами [1, 1]
    trial.suggest_int.assert_any_call("n_neighbors", 1, 1)

def test_optimize_integration_clipping(monkeypatch):
    """
    Проверяет, что optimize() применяет клиппинг к переданным параметрам.
    """
    # Создаем 20 строк, чтобы избежать проблем с делением при валидации
    X = pd.DataFrame(np.random.rand(20, 2), columns=['a', 'b'])
    y = pd.Series(np.random.rand(20))
    
    # Намеренно завышенные границы
    overrides = {
        "knn": {
            "n_neighbors": SearchSpaceEntry.model_validate([15, 100, "int"]),
        }
    }
    
    # 1. Мокаем метрику, чтобы она всегда возвращала 0.5 (избегаем NaN)
    monkeypatch.setattr(
        "configurable_automl_engine.tuner.get_scorer_object", 
        lambda name, global_y=None: lambda model, X, y: 0.5
    )

    # 2. Мокаем создание модели, чтобы проверить входящий n_neighbors
    def mock_create_model(algo, **params):
        if algo == "xgboosting": # Пропускаем инициализацию в начале optimize
            return Ridge()
        # Для KNN лимит n_samples-1 = 19
        if algo == "knn" and params.get("n_neighbors", 0) > 19:
            pytest.fail(f"n_neighbors {params['n_neighbors']} exceeds limit 19")
        return Ridge()

    monkeypatch.setattr("configurable_automl_engine.tuner.create_model", mock_create_model)
    
    # 3. Мокаем кросс-валидацию (на случай если сменится стратегия)
    monkeypatch.setattr(
        "configurable_automl_engine.tuner.model_selection.cross_val_score", 
        lambda *a, **kw: [0.5, 0.5]
    )

    # Запускаем оптимизацию
    _, best_params, _ = optimize(
        "knn", X, y, 
        n_trials=2, 
        space_overrides=overrides,
        validation_strategy="train_test_split",
        random_state=42
    )
    
    # При 20 образцах лимит n_neighbors = 19.
    # Исходный диапазон был [15, 100], должен стать [15, 19].
    assert 15 <= best_params["n_neighbors"] <= 19

def test_clipping_preserves_original_space():
    """Убеждаемся, что клиппинг не мутирует исходные объекты (Deep Copy)."""
    original_entry = SearchSpaceEntry.model_validate([10, 100, "int"])
    space = {"n_neighbors": original_entry}
    
    _ = clip_search_space(space, n_samples=5)
    
    # Исходный объект в словаре space не должен измениться
    assert original_entry.high == 100
    assert original_entry.low == 10

def test_clip_extreme_small_data():
    """Проверяет клиппинг при n_samples = 1."""
    space = {"n_neighbors": SearchSpaceEntry.model_validate([5, 10, "int"])}
    # n=1, limit = max(1, 1-1) = 1.
    clipped = clip_search_space(space, n_samples=1)
    assert clipped["n_neighbors"].low == 1
    assert clipped["n_neighbors"].high == 1

def test_clip_ignores_unrelated_params():
    """Проверяет, что клиппинг не трогает параметры не из списка ограничений."""
    space = {
        "max_depth": SearchSpaceEntry.model_validate([2, 32, "int"]),
        "learning_rate": SearchSpaceEntry.model_validate([0.01, 0.1, "float"])
    }
    clipped = clip_search_space(space, n_samples=5)
    assert clipped["max_depth"].high == 32
    assert clipped["learning_rate"].low == 0.01

def test_clip_preserves_types():
    """Проверяет, что FloatSpace остается FloatSpace после клиппинга."""
    space = {
        "min_samples_leaf": SearchSpaceEntry.model_validate([0.1, 0.5, "float"])
    }
    # При n=10, limit=10.0. Границы 0.1 и 0.5 не должны измениться, 
    # но важно, чтобы тип остался FloatSpace.
    clipped = clip_search_space(space, n_samples=10)
    assert isinstance(clipped["min_samples_leaf"].config, FloatSpace)

def test_clip_handles_categorical():
    """Проверяет, что категориальные параметры проходят через фильтр без изменений."""
    space = {
        "weights": SearchSpaceEntry.model_validate([["uniform", "distance"], "categorical"])
    }
    clipped = clip_search_space(space, n_samples=10)
    assert clipped["weights"].dist_type == "categorical"
    assert "uniform" in clipped["weights"].config.options