import pytest
import numpy as np
from sklearn.model_selection import LeaveOneOut
import sys
import importlib
from unittest.mock import MagicMock

from configurable_automl_engine.validation import (
    iter_splits, 
    make_cv, 
    RANDOM_STATE, 
    InvalidDataError,
    ValidationStrategy
)

def _dummy_data(n=100):
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(n, 5))
    y = rng.integers(0, 2, size=n)
    return X, y

# --- Тесты для make_cv и norm_val_method (Target: 100% Coverage) ---

def test_norm_val_method_enum():
    """Покрытие строк 41-42: передача ValidationStrategy (Enum)."""
    # Тестируем через прямое обращение к стратегии
    method_name, _ = make_cv(100, val_method=ValidationStrategy.k_fold, n_folds=5, random_state=42)
    assert method_name == "k_fold"

def test_make_cv_loo_error():
    """Покрытие строки 110: InvalidDataError при n < 2 для LOO."""
    with pytest.raises(InvalidDataError, match="Для leave-one-out нужно ≥ 2 наблюдений"):
        make_cv(1, val_method="loo", n_folds=5, random_state=42)

def test_make_cv_kfold_fallback():
    """Покрытие веток fallback: когда данных слишком мало для KFold."""
    # Случай 1: n_samples < 4 (даже если n_folds маленькое)
    method_name, cv = make_cv(3, val_method="k_fold", n_folds=2, random_state=42)
    assert method_name == "train_test_split"
    assert cv is None

    # Случай 2: n_samples < 2 * n_folds (например, 5 < 2*3)
    method_name2, cv2 = make_cv(5, val_method="k_fold", n_folds=3, random_state=42)
    assert method_name2 == "train_test_split"
    assert cv2 is None

def test_make_cv_loo_success():
    """Покрытие успешного создания LOO."""
    method_name, cv = make_cv(10, val_method="loo", n_folds=5, random_state=42)
    assert method_name == "loo"
    assert isinstance(cv, LeaveOneOut)

# --- Тесты для iter_splits (Покрытие основной логики генератора) ---

def test_iter_splits_logic_all_branches():
    X, y = _dummy_data(n=10)
    
    # Покрываем ветку TTS
    assert len(list(iter_splits(X, y, method="train_test_split"))) == 1
    
    # Покрываем ветку KFold
    assert len(list(iter_splits(X, y, method="k_fold", n_folds=3))) == 3
    
    # Покрываем ветку LOO
    assert len(list(iter_splits(X, y, method="loo"))) == 10

def test_iter_splits_errors():
    X, y = _dummy_data(n=5)
    with pytest.raises(ValueError, match="Unknown validation method"):
        list(iter_splits(X, y, method="unknown_method"))
    
    with pytest.raises(ValueError, match="n_folds must be ≥2"):
        list(iter_splits(X, y, method="k_fold", n_folds=1))

def test_make_cv_invalid_string():
    """Покрытие ValueError в конце make_cv."""
    with pytest.raises(ValueError, match="val_method должен быть"):
        make_cv(100, val_method="not_a_strategy", n_folds=5, random_state=42)

def test_logging_edge_case_jupyter():
    """
    Покрывает строки 41-42: edge-case инициализации логирования.
    Симулирует отсутствие атрибута handlers у модуля logging.
    """
    # Сохраняем оригинал
    original_logging = sys.modules.get("logging")
    
    # Создаем фейковый модуль без handlers
    mock_logging = MagicMock()
    del mock_logging.handlers
    sys.modules["logging"] = mock_logging
    
    # Перезагружаем модуль валидации, чтобы сработал блок if
    import configurable_automl_engine.validation as validation
    importlib.reload(validation)
    
    # Проверяем, что алиас logging создался
    assert hasattr(validation, "logging")
    
    # Восстанавливаем систему
    if original_logging:
        sys.modules["logging"] = original_logging
    else:
        sys.modules.pop("logging", None)

def test_make_cv_train_test_split_branch():
    """
    Покрывает строку 92: возврат для 'train_test_split' в make_cv.
    """
    method_name, cv_obj = make_cv(
        n_samples=100,
        val_method="train_test_split",
        n_folds=5,
        random_state=42
    )
    
    assert method_name == "train_test_split"
    assert cv_obj is None