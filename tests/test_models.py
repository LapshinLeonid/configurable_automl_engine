import pytest
from unittest.mock import patch
from configurable_automl_engine.models import create_model

def test_create_model_invalid_type():
    """
    Тест покрытия строки 96: проверка возбуждения ValueError, 
    если аргумент algorithm не является строкой.
    """
    # Передаем список вместо строки
    with pytest.raises(ValueError, match="Некорректный алгоритм:"):
        create_model(algorithm=["elasticnet"])
    
    # Передаем число вместо строки
    with pytest.raises(ValueError, match="Некорректный алгоритм:"):
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