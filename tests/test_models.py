import logging
from unittest.mock import patch, MagicMock

import pytest
from sklearn.linear_model import ARDRegression, ElasticNet
from sklearn.svm import SVR

from configurable_automl_engine.models import (
    LEGACY_PARAM_MAPPINGS,
    _get_constructor_param_info,
    _get_factory,
    clean_hyperparameters,
    create_model,
)


def test_get_constructor_param_info_ard_regression():
    """_get_constructor_param_info returns correct accepted params for ARDRegression."""
    accepted, accepts_kwargs = _get_constructor_param_info(ARDRegression)
    # max_iter is a known parameter
    assert "max_iter" in accepted
    # self should never be included
    assert "self" not in accepted
    # ~kwargs should not be present (ARDRegression does not accept **kwargs)
    assert not accepts_kwargs


def test_get_constructor_param_info_cache():
    """Calling with the same class returns the cached result."""
    result1 = _get_constructor_param_info(ElasticNet)
    result2 = _get_constructor_param_info(ElasticNet)
    assert result1 is result2  # same tuple object from lru_cache


def test_get_constructor_param_info_different_classes():
    """Different classes return different cached results."""
    r1 = _get_constructor_param_info(ElasticNet)
    r2 = _get_constructor_param_info(SVR)
    assert r1 != r2


def test_get_constructor_param_info_skips_self():
    """
    Cover the 'if name == "self": continue' branch (line 152).
    Use a class whose __init__ has only 'self' and **kwargs to also
    cover the VAR_KEYWORD branch (line 154).
    """
    class _ModelWithKwargs:
        def __init__(self, **kwargs):
            pass

    accepted, accepts_kwargs = _get_constructor_param_info(_ModelWithKwargs)
    # 'self' must never appear in accepted params
    assert "self" not in accepted
    # The class accepts **kwargs
    assert accepts_kwargs is True
    # No positional params beyond self
    assert accepted == frozenset()


def test_get_constructor_param_info_self_only():
    """
    Cover the 'if name == "self": continue' branch with a class
    that has ONLY 'self' (no params, no **kwargs).
    """
    class _ModelSelfOnly:
        def __init__(self):
            pass

    accepted, accepts_kwargs = _get_constructor_param_info(_ModelSelfOnly)
    assert "self" not in accepted
    assert accepts_kwargs is False
    assert accepted == frozenset()


def test_legacy_param_mappings_contains_ard():
    """LEGACY_PARAM_MAPPINGS has the expected entry for ardregression."""
    assert "ardregression" in LEGACY_PARAM_MAPPINGS
    assert LEGACY_PARAM_MAPPINGS["ardregression"]["n_iter"] == "max_iter"


def test_clean_hyperparameters_remapping():
    """clean_hyperparameters remaps n_iter -> max_iter for ARDRegression."""
    raw = {"n_iter": 500, "tol": 1e-4}
    cleaned = clean_hyperparameters("ardregression", ARDRegression, raw)
    assert "max_iter" in cleaned
    assert cleaned["max_iter"] == 500
    assert "n_iter" not in cleaned
    assert cleaned["tol"] == 1e-4


def test_clean_hyperparameters_drops_unknown_params():
    """clean_hyperparameters drops params not in the constructor signature."""
    raw = {"alpha": 1.0, "totally_fake_param": 42}
    cleaned = clean_hyperparameters("elasticnet", ElasticNet, raw)
    assert "alpha" in cleaned
    assert "totally_fake_param" not in cleaned


def test_clean_hyperparameters_remapped_key_kept_if_valid():
    """If the remapped key IS valid for the target class, it is kept."""
    raw = {"n_iter": 500}
    # ElasticNet DOES accept max_iter
    cleaned = clean_hyperparameters("ardregression", ElasticNet, raw)
    assert "max_iter" in cleaned
    assert cleaned["max_iter"] == 500


def test_clean_hyperparameters_empty_dict():
    """An empty hyperparams dict returns an empty dict."""
    cleaned = clean_hyperparameters("elasticnet", ElasticNet, {})
    assert cleaned == {}


def test_clean_hyperparameters_no_remapping_for_other_algo():
    """non-ard algo does not trigger remapping."""
    raw = {"n_iter": 500, "alpha": 1.0}
    cleaned = clean_hyperparameters("elasticnet", ElasticNet, raw)
    # n_iter is not a valid ElasticNet param -> dropped, not remapped
    assert "n_iter" not in cleaned
    assert "alpha" in cleaned


def test_create_model_ard_with_legacy_n_iter():
    """ARDRegression can be created with legacy 'n_iter' param."""
    model = create_model("ardregression", n_iter=300, tol=1e-4)
    assert isinstance(model, ARDRegression)
    # The clean function should have remapped n_iter -> max_iter
    assert model.max_iter == 300


def test_create_model_svr_default_max_iter():
    """SVR gets max_iter=10000 when not provided."""
    model = create_model("svr", C=1.0)
    assert model.max_iter == 10000


def test_create_model_svr_user_max_iter():
    """SVR respects user-provided max_iter."""
    model = create_model("svr", C=1.0, max_iter=5000)
    assert model.max_iter == 5000


def test_clean_hyperparameters_logs_remapped(caplog):
    """clean_hyperparameters logs at DEBUG level when remapping."""
    with caplog.at_level(logging.DEBUG, logger="configurable_automl_engine.models"):
        raw = {"n_iter": 500, "tol": 1e-4}
        clean_hyperparameters("ardregression", ARDRegression, raw)
    assert any("Remapped" in record.message for record in caplog.records)


def test_clean_hyperparameters_logs_dropped(caplog):
    """clean_hyperparameters logs at DEBUG level when dropping unknown param."""
    with caplog.at_level(logging.DEBUG, logger="configurable_automl_engine.models"):
        raw = {"alpha": 1.0, "fake_param": 99}
        clean_hyperparameters("elasticnet", ElasticNet, raw)
    assert any("Dropped" in record.message for record in caplog.records)


def test_create_model_invalid_type():
    """
    Проверка возбуждения TypeError, если аргумент algorithm не является строкой.
    """
    # Передаем список вместо строки
    with pytest.raises(TypeError, match="Алгоритм должен быть строкой"):
        create_model(algorithm=["elasticnet"])

    # Передаем число вместо строки
    with pytest.raises(TypeError, match="Алгоритм должен быть строкой"):
        create_model(algorithm=123)

def test_create_model_import_error_for_missing_package():
    """
    Тест покрытия строки 107: проверка возбуждения ImportError,
    если класс модели в фабрике равен None (имитация отсутствия XGBoost).
    """
    # Используем patch, чтобы временно подменить значение в _FACTORY на None
    # Это имитирует ситуацию, когда XGBRegressor не был импортирован
    with patch("configurable_automl_engine.models._FACTORY") as mocked_factory:
        # Настраиваем мок так, чтобы для 'xgboosting' возвращался None
        mocked_factory.__contains__.return_value = True
        mocked_factory.__getitem__.return_value = None
        
        # Пытаемся создать модель через алиас 'xgboost'
        with pytest.raises(ImportError, match="требует дополнительного пакета"):
            create_model("xgboost")

def test_create_model_unknown_algorithm():
    """
    Дополнительный тест для ветки неизвестного алгоритма (строка 102-103).
    """
    with pytest.raises(ValueError, match="Неизвестный алгоритм: 'unknown_model'"):
        create_model("unknown_model")

def test_create_model_gpr_default_kernel():
    """
    Проверка специфической логики для GaussianProcessRegressor (строки 114-115).
    """
    model = create_model("gpr")
    # Проверяем, что ядро RBF было установлено по умолчанию
    assert hasattr(model, "kernel")
    # В sklearn GPR после инициализации kernel сохраняется в параметрах
    assert "RBF" in str(model.get_params()["kernel"])
