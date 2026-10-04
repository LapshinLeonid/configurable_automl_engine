import numpy as np
import pytest
import logging
from unittest.mock import MagicMock
from configurable_automl_engine.training_engine.metrics import (
    _rmse,
    _nrmse,
    get_metric,
    is_greater_better,
    get_scorer_object,
    to_sklearn_name,
    to_user_value,
    NRMSEZeroRangeError,
    _global_nrmse,
    get_global_nrmse_scorer,
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
    assert result == float("inf")

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


@pytest.mark.parametrize(
    "metric_name, expected",
    [
        # 1. Собственный реестр: направление хранится явно рядом со скорером
        ("r2", True),
        ("R2", True),
        ("rmse", False),
        ("mae", False),
        ("mse", False),
        ("nrmse", False),
        # neg_-метрика реестра: инвертированная ошибка → максимизация
        ("neg_root_mean_squared_error", True),
        # 2. sklearn-скореры: большее значение всегда лучше — метрики-ошибки
        # уже инвертированы в neg_-скореры, score-метрики возвращаются как есть
        ("accuracy", True),
        ("explained_variance", True),
        ("neg_log_loss", True),
        ("neg_mean_absolute_percentage_error", True),
        ("neg_mean_squared_error", True),
        ("neg_mean_absolute_error", True),
    ],
)
def test_is_greater_better(metric_name, expected):
    assert is_greater_better(metric_name) == expected


def test_is_greater_better_contract_from_issue():
    """Контракт из issue #26: направление определяется объектом-скорером."""
    assert is_greater_better("neg_root_mean_squared_error") is True
    assert is_greater_better("r2") is True
    assert is_greater_better("mae") is False


def test_is_greater_better_unknown_metric_raises():
    """Неизвестная метрика не имеет направления — понятный ValueError.

    Сообщение об ошибке объясняет, как исправить (реестр или sklearn),
    аналогично get_metric (ревью PR #18).
    """
    with pytest.raises(ValueError, match="not implemented"):
        is_greater_better("unknown_custom_metric")


def test_to_user_value_natural_semantics():
    """«Сырые» значения скореров приводятся к пользовательской семантике.

    RMSE/MAE/MSE/NRMSE и все neg_-метрики возвращаются положительными,
    score-метрики (R², accuracy) — как есть.
    """
    assert to_user_value("rmse", -0.123) == pytest.approx(0.123)
    assert to_user_value("neg_root_mean_squared_error", -0.123) == pytest.approx(0.123)
    assert to_user_value("mae", -0.05) == pytest.approx(0.05)
    assert to_user_value("mse", -1.5) == pytest.approx(1.5)
    assert to_user_value("nrmse", -0.3) == pytest.approx(0.3)
    assert to_user_value("neg_log_loss", -0.6) == pytest.approx(0.6)
    assert to_user_value("neg_mean_squared_error", -2.0) == pytest.approx(2.0)
    assert to_user_value("r2", 0.85) == pytest.approx(0.85)
    assert to_user_value("accuracy", 0.9) == pytest.approx(0.9)


def test_to_user_value_global_nrmse():
    """global_nrmse — динамическая ошибка: значение инвертируется."""
    assert to_user_value("global_nrmse", -0.25) == pytest.approx(0.25)


def test_is_greater_better_scorer_without_sign_fallback(monkeypatch):
    """Скорер без атрибута _sign: консервативный фолбэк True.

    Покрывает защитную ветку для гипотетических кастомных скореров,
    у которых направление не экспонировано.
    """
    import configurable_automl_engine.training_engine.metrics as metrics_mod

    class NoSignScorer:
        pass

    monkeypatch.setattr(
        metrics_mod, "sklearn_get_scorer", lambda name: NoSignScorer()
    )
    assert is_greater_better("some_custom_metric") is True


def test_is_greater_better_reflects_registry_mutation():
    """Мутация реестра _SCORER_OBJECTS видна сразу, без устаревшего кэша.

    Регрессия ревью PR #18: направление из собственного реестра читается
    без кэширования, поэтому изменение реестра в рантайме (регистрация
    кастомной метрики, смена направления) подхватывается немедленно.
    """
    from sklearn.metrics import make_scorer, mean_squared_error

    import configurable_automl_engine.training_engine.metrics as metrics_mod

    custom_name = "dyn_registry_metric"
    assert custom_name not in metrics_mod._SCORER_OBJECTS

    # До регистрации метрика неизвестна → ValueError.
    with pytest.raises(ValueError):
        is_greater_better(custom_name)

    # Регистрируем «ошибку» (меньше — лучше) и проверяем направление.
    metrics_mod._SCORER_OBJECTS[custom_name] = (
        make_scorer(mean_squared_error, greater_is_better=False),
        False,
    )
    try:
        assert is_greater_better(custom_name) is False
        # Меняем направление в реестре — результат обязан обновиться,
        # несмотря на предшествующие вызовы (кэш реестра отсутствует).
        scorer, _ = metrics_mod._SCORER_OBJECTS[custom_name]
        metrics_mod._SCORER_OBJECTS[custom_name] = (scorer, True)
        assert is_greater_better(custom_name) is True
    finally:
        del metrics_mod._SCORER_OBJECTS[custom_name]


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
    with pytest.raises(
        ValueError, match="For 'global_nrmse', 'global_y' must be passed"
    ):
        get_scorer_object("global_nrmse", global_y=None)
    # 2. Проверка успешного создания скорера
    y_full = np.array([10.0, 20.0, 30.0])  # range = 20.0
    scorer = get_scorer_object("global_nrmse", global_y=y_full)

    # Проверяем, что это объект-скорер (Callable)
    assert callable(scorer)
    # 3. Проверка ветки target_range < 1e-6 в _global_nrmse
    # Напрямую вызываем функцию с критически малым диапазоном
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.1, 1.9])

    with caplog.at_level(logging.WARNING):
        res_inf = _global_nrmse(y_true, y_pred, target_range=1e-7)
        assert res_inf == float("inf")
        assert "Global target_range is too small" in caplog.text
    # 4. Проверка стандартного расчета (основная ветка возврата)
    # RMSE для (1,2) и (1,2) = 0. Range = 10. Result = 0/10 = 0.0
    res_zero = _global_nrmse(
        np.array([1.0, 2.0]), np.array([1.0, 2.0]), target_range=10.0
    )
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
