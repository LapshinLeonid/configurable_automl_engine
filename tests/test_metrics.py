import numpy as np
import pytest
from configurable_automl_engine.training_engine.metrics import (
    _rmse, _nrmse, get_metric, is_greater_better, 
    get_scorer_object, to_sklearn_name, NRMSEZeroRangeError
)

def test_nrmse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    nrmse = get_metric("nrmse")
    val = nrmse(y_true, y_pred)

    # ручная проверка
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    expected = rmse / (y_true.max() - y_true.min())

    assert np.isclose(val, expected, atol=1e-8)
    assert not is_greater_better("nrmse")

# Тестируем RMSE
def test_rmse_calculation():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    # MSE = (0.5^2 + 0.5^2 + 0 + 1^2) / 4 = 1.5 / 4 = 0.375
    # RMSE = sqrt(0.375) approx 0.61237
    result = _rmse(y_true, y_pred)
    assert isinstance(result, float)
    assert result == pytest.approx(np.sqrt(0.375))

def test_nrmse_normal_case():
    y_true = np.array([0, 10])
    y_pred = np.array([0, 5])
    # RMSE = sqrt((0^2 + 5^2)/2) = sqrt(12.5) approx 3.535
    # Denom = 10 - 0 = 10
    # Result = 3.535 / 10 = 0.3535
    assert _nrmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5) / 10)

def test_nrmse_zero_range_coverage():
    """Покрывает строки 35 (Error class) и 53-57 (denom < 1e-6)"""
    y_true = np.array([1, 1, 1])  # Константный таргет, range = 0
    y_pred = np.array([1, 2, 3])
    
    # 1. Проверяем логику обработки нулевого диапазона (строки 53-57)
    # Код возвращает float('inf'), а не выбрасывает исключение
    result = _nrmse(y_true, y_pred)
    assert result == float('inf')
    
    # 2. Покрываем объявление класса NRMSEZeroRangeError (строка 35)
    # Просто создаем экземпляр, чтобы анализатор зачел выполнение строки
    err = NRMSEZeroRangeError("Target range is too small")
    assert isinstance(err, ValueError)
    assert str(err) == "Target range is too small"

# Тестируем реестры и хелперы

def test_get_metric():
    # Проверка корректного получения
    assert get_metric("rmse") == _rmse
    assert get_metric("NRMSE") == _nrmse
    
    # Проверка исключения KeyError
    with pytest.raises(KeyError, match="Metric 'unknown' not implemented"):
        get_metric("unknown")

def test_is_greater_better():
    assert is_greater_better("r2") is True
    assert is_greater_better("neg_root_mean_squared_error") is True
    assert is_greater_better("rmse") is False
    assert is_greater_better("nrmse") is False

def test_get_scorer_object():
    # Проверка кастомных объектов (включая лямбды в _SCORER_OBJECTS)
    scorer = get_scorer_object("rmse")
    assert callable(scorer)
    
    # Проверка nrmse
    nrmse_scorer = get_scorer_object("nrmse")
    assert callable(nrmse_scorer)
    # Проверка стандартных метрик sklearn (вызов sklearn_get_scorer)
    std_scorer = get_scorer_object("explained_variance")
    assert hasattr(std_scorer, "_score_func")

# Тестируем покрытие строки 107 (to_sklearn_name)
def test_to_sklearn_name():
    # Случай из словаря
    assert to_sklearn_name("rmse") == "neg_root_mean_squared_error"
    # Случай default (строка 107: возврат как есть в lower-case)
    assert to_sklearn_name("R2") == "r2"
    assert to_sklearn_name("Unknown_Metric") == "unknown_metric"

# Тестируем лямбда-функции в реестрах (дополнительное покрытие)
def test_neg_rmse_lambda():
    # Покрываем лямбду в _METRICS["neg_root_mean_squared_error"]
    neg_rmse_func = get_metric("neg_root_mean_squared_error")
    y_true = np.array([0, 2])
    y_pred = np.array([0, 0])
    # Ожидаемое значение: -sqrt((0^2 + 2^2)/2) = -sqrt(2) ≈ -1.414
    assert neg_rmse_func(y_true, y_pred) == pytest.approx(-np.sqrt(2.0))
