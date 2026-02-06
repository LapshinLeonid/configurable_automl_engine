import pytest
from unittest.mock import patch
# Предположим, функция находится в файле utils.py
from configurable_automl_engine.common.dependency_utils import is_installed

def test_is_installed_empty_string():
    """Проверка случая с пустой строкой (ваша целевая строка)."""
    assert is_installed("") is False
def test_is_installed_none():
    """Проверка на случай передачи None (если типы не соблюдены)."""
    # Этот тест полезен, если не используется строгая типизация
    assert is_installed(None) is False
def test_is_installed_standard_library():
    """Проверка пакета из стандартной библиотеки (всегда есть)."""
    assert is_installed("os") is True
    assert is_installed("sys") is True
def test_is_installed_non_existent():
    """Проверка заведомо несуществующего пакета."""
    assert is_installed("this_package_definitely_does_not_exist_123") is False
def test_is_installed_mocked_success():
    """Тест с имитацией (mock) успешного нахождения спецификации."""
    with patch("importlib.util.find_spec") as mock_find:
        mock_find.return_value = True # find_spec возвращает объект, который не None
        assert is_installed("any_package") is True
        mock_find.assert_called_once_with("any_package")
def test_is_installed_mocked_failure():
    """Тест с имитацией отсутствия пакета."""
    with patch("importlib.util.find_spec") as mock_find:
        mock_find.return_value = None
        assert is_installed("missing_package") is False