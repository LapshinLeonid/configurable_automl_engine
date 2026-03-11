import numpy as np
import pytest
import logging
from unittest.mock import MagicMock
from configurable_automl_engine.training_engine.metrics import (
    _rmse, _nrmse, get_metric, is_greater_better, 
    get_scorer_object, to_sklearn_name, NRMSEZeroRangeError,
    _global_nrmse, get_global_nrmse_scorer
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

@pytest.mark.parametrize("metric_name, expected", [
    # 1. Тесты для явного списка _GREATER_IS_BETTER
    ("r2", True),
    ("R2", True),
    ("accuracy", True),
    
    # 2. Тесты для вхождения ключевых слов ошибок
    ("rmse", False),
    ("mean_squared_error", False),
    ("my_custom_mae_metric", False),
    ("NRMSE_score", False),
    
    # 3. КРИТИЧЕСКИЙ ТЕСТ: Покрытие строки if lname.startswith("neg_")
    # Эти метрики не входят в список _GREATER_IS_BETTER и не содержат "mse/mae/error"
    ("neg_log_loss", False),
    ("neg_mean_absolute_percentage_error", False),
    ("NEG_WHATEVER", False),
    
    # 4. Тест на значение по умолчанию (если не подошло ни одно условие)
    ("unknown_custom_metric", True),
])
def test_is_greater_better(metric_name, expected):
    assert is_greater_better(metric_name) == expected

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

def test_global_nrmse_coverage(caplog):
    """
    Тест покрывает:
    1. Исключение ValueError в get_scorer_object, если global_y не передан.
    2. Успешное создание скорера через get_global_nrmse_scorer.
    3. Ветку target_range < 1e-6 в _global_nrmse (защита от деления на ноль).
    4. Стандартный расчет _global_nrmse.
    """
    
    # 1. Проверка исключения: global_nrmse вызван без global_y
    with pytest.raises(ValueError, match="For 'global_nrmse', 'global_y' must be passed"):
        get_scorer_object("global_nrmse", global_y=None)
    # 2. Проверка успешного создания скорера
    y_full = np.array([10.0, 20.0, 30.0]) # range = 20.0
    scorer = get_scorer_object("global_nrmse", global_y=y_full)
    
    # Проверяем, что это объект-скорер (Callable)
    assert callable(scorer)
    # 3. Проверка ветки target_range < 1e-6 в _global_nrmse
    # Напрямую вызываем функцию с критически малым диапазоном
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.1, 1.9])
    
    with caplog.at_level(logging.WARNING):
        res_inf = _global_nrmse(y_true, y_pred, target_range=1e-7)
        assert res_inf == float('inf')
        assert "Global target_range is too small" in caplog.text
    # 4. Проверка стандартного расчета (основная ветка возврата)
    # RMSE для (1,2) и (1,2) = 0. Range = 10. Result = 0/10 = 0.0
    res_zero = _global_nrmse(np.array([1.0, 2.0]), np.array([1.0, 2.0]), target_range=10.0)
    assert res_zero == 0.0
    # Расчет с конкретными значениями:
    # y_true=[0, 2], y_pred=[0, 0] -> MSE = (0^2 + 2^2)/2 = 2 -> RMSE = sqrt(2) ≈ 1.4142
    # target_range = 2
    # Result = 1.4142 / 2 = 0.7071...
    y_t = np.array([0.0, 2.0])
    y_p = np.array([0.0, 0.0])
    expected_rmse = np.sqrt(2.0)
    expected_nrmse = expected_rmse / 2.0
    
    assert np.isclose(_global_nrmse(y_t, y_p, target_range=2.0), expected_nrmse)
