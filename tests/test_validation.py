import pytest
import numpy as np
from sklearn.model_selection import LeaveOneOut
import sys
import importlib
from unittest.mock import MagicMock, patch
import pandas as pd

from configurable_automl_engine.validation import (
    iter_splits, 
    make_cv, 
    RANDOM_STATE, 
    InvalidDataError,
    ValidationStrategy,
    ValidationError
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
    method_name, _ = make_cv(100, val_method=ValidationStrategy.k_fold, n_folds=5, random_state=42, test_size = 0.2)
    assert method_name == "k_fold"

def test_make_cv_loo_error():
    """
    Покрытие строки с проверкой n < 2 для LOO.
    Исправлено: match соответствует тексту 'at least 2 samples'.
    """
    with pytest.raises(InvalidDataError, match="at least 2 samples"):
        make_cv(1, val_method="loo", n_folds=5, random_state=42, test_size=0.2)

def test_make_cv_kfold_fallback():
    """Покрытие веток fallback: когда данных слишком мало для KFold."""
    # Случай 1: n_samples < 4 (даже если n_folds маленькое)
    method_name, cv = make_cv(3, val_method="k_fold", n_folds=2, random_state=42, test_size = 0.2)
    assert method_name == "train_test_split"
    assert cv is None

    # Случай 2: n_samples < 2 * n_folds (например, 5 < 2*3)
    method_name2, cv2 = make_cv(5, val_method="k_fold", n_folds=3, random_state=42, test_size = 0.2)
    assert method_name2 == "train_test_split"
    assert cv2 is None

def test_make_cv_loo_success():
    """Покрытие успешного создания LOO."""
    method_name, cv = make_cv(10, val_method="loo", n_folds=5, random_state=42, test_size = 0.2)
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
    """
    Проверка обработки ошибок в генераторе.
    """
    X, y = _dummy_data(n=5)
    
    # 1. Тест на неизвестный метод
    # match изменен на соответствующий функции make_cv, так как iter_splits вызывает её
    with pytest.raises(ValueError, match="Unknown validation method"):
        list(iter_splits(X, y, method="unknown_method"))
    # 2. Тест на пустые данные (X length == 0)
    with pytest.raises(InvalidDataError, match="Input array X is empty"):
        list(iter_splits(np.array([]), np.array([]), method="k_fold"))
    # 3. Тест на несоответствие длин X и y
    with pytest.raises(InvalidDataError, match="X and y length mismatch"):
        list(iter_splits(np.random.rand(10, 2), np.random.rand(5), method="k_fold"))

def test_make_cv_invalid_string():
    """
    Покрытие ValueError в конце make_cv.
    Исправлено: match соответствует тексту в коде реализации.
    """
    with pytest.raises(ValueError, match="Unknown validation method"):
        make_cv(100, val_method="not_a_strategy", n_folds=5, random_state=42, test_size=0.2)

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
        random_state=42, test_size = 0.2
    )
    
    assert method_name == "train_test_split"
    assert cv_obj is None

class TestValidationEngineCoverage:
    """Тесты для покрытия специфических веток кода в iter_splits."""
    def test_iter_splits_critical_error_cv_none(self):
            """
            Покрывает строку 178 (raise ValidationError).
            Используем перехват базового Exception, чтобы избежать проблем с путями импорта.
            """
            X = np.array([[1, 2], [3, 4]])
            y = np.array([0, 1])
            # Патчим make_cv именно по тому пути, который указан в логах ошибки
            with patch('configurable_automl_engine.validation.make_cv') as mocked_make_cv:
                mocked_make_cv.return_value = ("k_fold", None)
                
                gen = iter_splits(X, y, method="k_fold")
                
                # Перехватываем любое исключение и проверяем его имя и текст
                # Это обходит проблему, когда ValidationError в тесте != ValidationError в коде
                with pytest.raises(Exception) as exc_info:
                    next(gen)
                
                # Проверяем, что это именно наше исключение по имени класса
                assert exc_info.type.__name__ == "ValidationError"
                # Проверяем текст ошибки (склеенная строка из кода)
                assert "Critical error: CV object is not initialized" in str(exc_info.value)

    def test_iter_splits_pandas_iloc_coverage(self):
        """
        Покрывает:
            if hasattr(X, 'iloc'):
                x_tr, x_te = X.iloc[train_idx], X.iloc[test_idx]
            if hasattr(y, 'iloc'):
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        """
        # Создаем данные Pandas
        X_df = pd.DataFrame({'feat1': [1, 2, 3, 4, 5], 'feat2': [10, 20, 30, 40, 50]})
        y_ser = pd.Series([0, 1, 0, 1, 0], name='target')
        # Используем k_fold, чтобы сработал цикл с индексами
        gen = iter_splits(X_df, y_ser, method="k_fold", n_folds=2)
        
        # Получаем первую итерацию
        x_tr, x_te, y_tr, y_te = next(gen)
        # Проверки, что вернулись именно объекты Pandas (значит iloc сработал)
        assert isinstance(x_tr, pd.DataFrame)
        assert isinstance(x_te, pd.DataFrame)
        assert isinstance(y_tr, pd.Series)
        assert isinstance(y_te, pd.Series)
        
        # Проверка корректности данных
        assert len(x_tr) + len(x_te) == len(X_df)
        assert len(y_tr) + len(y_te) == len(y_ser)
    def test_iter_splits_pandas_X_only_iloc(self):
        """
        Дополнительный тест для случая, когда X - DataFrame, а y - None.
        Покрывает ветку индексации X без индексации y.
        """
        X_df = pd.DataFrame({'feat1': [1, 2, 3, 4, 5]})
        
        gen = iter_splits(X_df, y=None, method="k_fold", n_folds=2)
        x_tr, x_te, y_tr, y_te = next(gen)
        assert isinstance(x_tr, pd.DataFrame)
        assert y_tr is None
        assert y_te is None
    def test_iter_splits_numpy_fallback(self):
        """
        Для полноты: проверка, что если нет iloc (numpy), код идет по ветке else.
        """
        X_np = np.random.rand(10, 2)
        y_np = np.random.randint(0, 2, 10)
        gen = iter_splits(X_np, y_np, method="k_fold", n_folds=2)
        x_tr, x_te, y_tr, y_te = next(gen)
        assert isinstance(x_tr, np.ndarray)
        assert isinstance(y_tr, np.ndarray)