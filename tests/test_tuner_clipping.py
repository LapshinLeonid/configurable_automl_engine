import pytest
import pandas as pd
import numpy as np
import math
from unittest.mock import MagicMock, patch
from sklearn.linear_model import Ridge
from configurable_automl_engine.common.hyperopt_defaults import FloatSpace

from configurable_automl_engine.common.hyperopt_defaults import (
    SearchSpaceEntry, 
    clip_search_space,
    IntSpace
)
from configurable_automl_engine.tuner import _make_knn_space, optimize

from configurable_automl_engine.common.validation_utils import get_effective_train_size
from configurable_automl_engine.common.definitions import ValidationStrategy
from configurable_automl_engine.trainer import ModelTrainer, train_model, TrainingError


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

@pytest.mark.parametrize("n_total, strategy, n_folds, test_size, expected", [
    # K-Fold: floor(100 * (1 - 1/5)) = 80
    (100, ValidationStrategy.k_fold, 5, 0.2, 80),
    # K-Fold: floor(10 * (1 - 1/3)) = floor(6.66) = 6
    (10, "k_fold", 3, 0.2, 6),
    # LOO: 100 - 1 = 99
    (100, ValidationStrategy.loo, 5, 0.2, 99),
    # Train-Test Split: floor(100 * (1 - 0.25)) = 75
    (100, ValidationStrategy.train_test_split, 5, 0.25, 75),
    # Тест безопасности (clamping) test_size: 1.5 -> 0.99. floor(100 * 0.01) = 1
    (100, ValidationStrategy.train_test_split, 5, 1.5, 1),
    # Тест граничного случая: n=0
    (0, ValidationStrategy.k_fold, 5, 0.2, 0),
    # Минимум для Neff при n_total >= 2
    (2, ValidationStrategy.train_test_split, 5, 0.99, 1),
])
def test_get_effective_train_size_logic(n_total, strategy, n_folds, test_size, expected):
    result = get_effective_train_size(n_total, strategy, n_folds, test_size)
    assert result == expected


# ──────────────────────────────────────────────────────────────────────
# 2. Тесты для trainer.py (Refactoring ModelTrainer)
# ──────────────────────────────────────────────────────────────────────

def test_model_trainer_no_longer_splits_internally():
    """Проверяет, что ModelTrainer теперь обучается на всех переданных данных."""
    df = pd.DataFrame({
        "feature1": np.random.rand(10),
        "target": np.random.rand(10)
    })
    
    # Инициализация без test_size (теперь его нет в сигнатуре)
    trainer = ModelTrainer(algorithm="ridge")
    
    with patch.object(ModelTrainer, '_fit_internal', wraps=trainer._fit_internal) as mock_fit:
        trainer.fit(df, y="target")
        
        # Проверяем, что в _fit_internal передано 10 строк (весь df), а не часть
        args, _ = mock_fit.call_args
        # args[0] это X_train
        assert len(args[0]) == 10
        assert trainer.val_score is not None

def test_train_model_facade_compatibility():
    """Проверка, что функция-фасад train_model не падает без test_size."""
    X = pd.DataFrame(np.random.rand(10, 2))
    y = pd.Series(np.random.rand(10))
    
    # Вызов старого API (params_or_metric — это dict параметров)
    # test_size теперь просто игнорируется внутри или отсутствует
    score = train_model(
        cfg_or_algo="ridge",
        metric_or_testsize="r2",
        params_or_metric={"alpha": 1.0},
        X=X, y=y
    )
    assert isinstance(score, float)


# ──────────────────────────────────────────────────────────────────────
# 3. Тесты для tuner.py (KNN & Space Clipping)
# ──────────────────────────────────────────────────────────────────────

def test_knn_space_limit_uses_effective_size():
    """Проверяет, что верхний предел KNN учитывает Neff, а не N_total."""
    # Neff = 5
    trial = MagicMock()
    space_gen = _make_knn_space(n_samples=6)
    
    space_gen(trial)
    
    # physical_limit = max(1, 6 - 1) = 5.
    # suggest_int должен быть вызван с high=5 (min(30, 5))
    trial.suggest_int.assert_any_call("n_neighbors", 1, 5)

@patch("configurable_automl_engine.tuner.clip_search_space")
@patch("configurable_automl_engine.tuner.create_model")
@patch("optuna.create_study")
def test_optimize_calls_clipping_with_effective_size(mock_study, mock_create, mock_clip):
    """Проверяет, что при оптимизации клиппинг вызывается с эффективным размером."""
    X = pd.DataFrame(np.random.rand(20, 2))
    y = pd.Series(np.random.rand(20))
    
    # Имитируем внешний конфиг пространства поиска
    space_overrides = {"ridge": {"alpha": [0.1, 1.0]}}
    
    # Запускаем оптимизацию со стратегией k_fold (n_folds=2)
    # Neff = floor(20 * (1 - 1/2)) = 10
    try:
        optimize(
            algo_name="ridge",
            X=X, y=y,
            validation_strategy="k_fold",
            n_folds=2,
            n_trials=1,
            space_overrides=space_overrides
        )
    except Exception:
        pass # Нам важен только вызов клиппинга
    
    # Проверяем, что clip_search_space получил n_samples=10 (Neff)
    mock_clip.assert_called_once()
    assert mock_clip.call_args[0][1] == 10

def test_knn_space_limit_at_minimum():
    trial = MagicMock()
    # Если осталась всего 1 строка после сплита
    space_gen = _make_knn_space(n_samples=1)
    space_gen(trial)
    # Должен предложить 1 соседа (минимум для sklearn), а не 0
    trial.suggest_int.assert_any_call("n_neighbors", 1, 1)

def test_model_trainer_raises_error_on_empty_data():
    """Проверка генерации исключения TrainingError при передаче пустых данных."""
    trainer = ModelTrainer(algorithm="ridge")
    
    # Сценарий 1: Пустой DataFrame
    empty_df = pd.DataFrame()
    empty_y = pd.Series(dtype=float)
    
    with pytest.raises(TrainingError, match="Data is empty"):
        trainer.fit(empty_df, empty_y)
        
    # Сценарий 2: Пустой numpy массив
    empty_X_np = np.array([]).reshape(0, 2)
    empty_y_np = np.array([])
    
    with pytest.raises(TrainingError, match="Data is empty"):
        trainer.fit(empty_X_np, empty_y_np)

    # Сценарий 3: Данные с колонками, но без строк
    df_zero_rows = pd.DataFrame(columns=["a", "b"])
    y_zero_rows = pd.Series(dtype=float)
    
    with pytest.raises(TrainingError, match="Data is empty"):
        trainer.fit(df_zero_rows, y_zero_rows)

def test_get_effective_train_size_raises_on_invalid_string():
    """Проверка, что функция падает на неизвестной строке."""
    with pytest.raises(ValueError, match="Unknown validation strategy string"):
        get_effective_train_size(100, strategy="magic_split")

def test_get_effective_train_size_raises_on_invalid_type():
    """Проверка, что функция падает на некорректном типе (например, None)."""
    with pytest.raises(ValueError, match="Unsupported validation strategy type"):
        get_effective_train_size(100, strategy=None)
        
    with pytest.raises(ValueError, match="Unsupported validation strategy type"):
        get_effective_train_size(100, strategy=123.45)

def test_clip_search_space_raises_on_negative_samples():
    """
    Проверка, что clip_search_space выбрасывает ValueError при отрицательном n_samples.
    """
    # Создаем минимальное корректное пространство поиска для теста
    space = {
        "n_neighbors": SearchSpaceEntry.model_validate([1, 50, "int"])
    }
    
    # Сценарий 1: Отрицательное значение
    with pytest.raises(ValueError, match="must be positive"):
        clip_search_space(space, n_samples=-1)

    with pytest.raises(ValueError, match="Got -100"):
        clip_search_space(space, n_samples=-100)

def test_clip_search_space_raises_on_zero_samples():
    """Проверка, что 0 образцов теперь вызывает ошибку, а не возвращает space."""
    space = {"n_neighbors": SearchSpaceEntry.model_validate([1, 50, "int"])}
    
    with pytest.raises(ValueError, match="must be positive"):
        clip_search_space(space, n_samples=0)