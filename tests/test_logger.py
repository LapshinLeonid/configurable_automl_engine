import logging
import pytest
from pathlib import Path
import importlib
from unittest.mock import MagicMock, patch
import sys
import configurable_automl_engine.training_engine
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


def test_setup_logging_creates_directory(tmp_path, clean_logger):
    """
    Тестирует создание директории и файла лога.
    """
    log_dir = tmp_path / "subdir"
    logfile = log_dir / "test.log"
    
    # Вызываем функцию (по умолчанию log_to_console=True)
    configurable_automl_engine.training_engine.logger.setup_logging(logfile)
    
    # Проверяем файловую систему
    assert log_dir.exists()
    assert logfile.exists()
    
    # Ожидаем 2 хендлера: ConcurrentRotatingFileHandler + StreamHandler
    assert len(clean_logger.handlers) == 2

def test_setup_logging_idempotency(tmp_path, clean_logger):
    """
    Проверяет идемпотентность (отсутствие дубликатов при повторном вызове).
    """
    logfile = tmp_path / "first.log"
    
    # Первый вызов
    configurable_automl_engine.training_engine.logger.setup_logging(logfile)
    assert len(clean_logger.handlers) == 2
    
    # Второй вызов
    configurable_automl_engine.training_engine.logger.setup_logging(logfile)
    
    # ИСПРАВЛЕНИЕ: Количество не должно увеличиться, должно остаться 2.
    assert len(clean_logger.handlers) == 2
def test_setup_logging_import_fallback(tmp_path):
    """
    Покрывает ветку 'except ImportError'.
    """
    with patch.dict(sys.modules, {'concurrent_log_handler': None}):
        # Перезагружаем модуль
        importlib.reload(configurable_automl_engine.training_engine.logger)
        
        # Проверяем подмену класса
        handler_class = getattr(configurable_automl_engine.training_engine.logger, 'ConcurrentRotatingFileHandler')
        
        # ИСПРАВЛЕНИЕ: Используем полное имя из модуля logging.handlers для надежности
        assert handler_class is logging.handlers.RotatingFileHandler
    # Восстанавливаем состояние
    importlib.reload(configurable_automl_engine.training_engine.logger)
def test_setup_logging_stream_handler_logic_coverage(tmp_path, clean_logger):
    """
    Специфический тест для покрытия логики if не any(type(h) is StreamHandler).
    """
    logfile = tmp_path / "test.log"
    
    # 1. Вызываем без консоли
    configurable_automl_engine.training_engine.logger.setup_logging(logfile, log_to_console=False)
    assert len(clean_logger.handlers) == 1
    
    # 2. Вызываем с консолью (заходим в покрываемую ветку)
    configurable_automl_engine.training_engine.logger.setup_logging(logfile, log_to_console=True)
    assert len(clean_logger.handlers) == 2
    
    # 3. Вызываем снова с консолью (проверяем, что ветка пропускается)
    configurable_automl_engine.training_engine.logger.setup_logging(logfile, log_to_console=True)
    assert len(clean_logger.handlers) == 2