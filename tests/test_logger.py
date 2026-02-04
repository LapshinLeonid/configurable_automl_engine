import logging
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from configurable_automl_engine.training_engine.logger import get_logger, setup_logging

@pytest.fixture
def clean_logger():
    """Фикстура для очистки обработчиков логгера перед и после теста."""
    logger = logging.getLogger("configurable_automl_engine")
    original_handlers = logger.handlers[:]
    logger.handlers = []
    yield logger
    logger.handlers = original_handlers

def test_get_logger():
    """Тестирует получение логгера по имени (строка 12)."""
    logger_name = "test_module"
    logger = get_logger(logger_name)
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == logger_name

def test_setup_logging_creates_directory(tmp_path, clean_logger):
    """
    Тестирует создание директории и файла лога, а также инициализацию (строки 19-33).
    """
    log_dir = tmp_path / "subdir"
    logfile = log_dir / "test.log"
    
    # Вызываем функцию
    setup_logging(logfile)
    
    # Проверяем создание директории и файла
    assert log_dir.exists()
    assert log_dir.is_dir()
    
    # Проверяем, что FileHandler добавлен
    assert len(clean_logger.handlers) == 1
    assert isinstance(clean_logger.handlers[0], logging.FileHandler)
    assert Path(clean_logger.handlers[0].baseFilename).resolve() == logfile.resolve()

def test_setup_logging_idempotency(tmp_path, clean_logger):
    """
    Проверяет, что повторный вызов setup_logging не добавляет дублирующие хендлеры (строка 31).
    """
    logfile = tmp_path / "first.log"
    second_logfile = tmp_path / "second.log"
    
    # Первый вызов
    setup_logging(logfile)
    assert len(clean_logger.handlers) == 1
    
    # Второй вызов (не должен добавить новый FileHandler)
    setup_logging(second_logfile)
    assert len(clean_logger.handlers) == 1
    
    # Проверяем, что остался первый путь
    assert Path(clean_logger.handlers[0].baseFilename).resolve() == logfile.resolve()

def test_setup_logging_format_and_level(tmp_path, clean_logger):
    """Проверяет корректность настройки уровня логирования и формата."""
    logfile = tmp_path / "format_test.log"
    
    setup_logging(logfile)
    handler = clean_logger.handlers[0]
    
    assert handler.level == logging.DEBUG
    # Проверка формата через доступ к приватному атрибуту или проверку вывода
    assert "%(levelname)-8s" in handler.formatter._fmt