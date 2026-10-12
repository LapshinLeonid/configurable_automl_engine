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
    """Fixture that clears the logger handlers before and after the test."""
    logger = logging.getLogger("configurable_automl_engine")
    original_handlers = logger.handlers[:]
    logger.handlers = []
    yield logger
    logger.handlers = original_handlers


def test_get_logger():
    """Tests obtaining a logger by name (line 12)."""
    logger_name = "test_module"
    logger = get_logger(logger_name)

    assert isinstance(logger, logging.Logger)
    assert logger.name == logger_name


def test_setup_logging_creates_directory(tmp_path, clean_logger):
    """
    Tests the creation of the log directory and file, as well as the initialization (lines 19-33).
    """
    log_dir = tmp_path / "subdir"
    logfile = log_dir / "test.log"

    # Call the function
    setup_logging(logfile)

    # Check that the directory and file were created
    assert log_dir.exists()
    assert log_dir.is_dir()

    # Check that the FileHandler was added
    assert len(clean_logger.handlers) == 1
    assert isinstance(clean_logger.handlers[0], logging.FileHandler)
    assert Path(clean_logger.handlers[0].baseFilename).resolve() == logfile.resolve()


def test_setup_logging_idempotency(tmp_path, clean_logger):
    """
    Checks that calling setup_logging again does not add duplicate handlers (line 31).
    """
    logfile = tmp_path / "first.log"
    second_logfile = tmp_path / "second.log"

    # First call
    setup_logging(logfile)
    assert len(clean_logger.handlers) == 1

    # Second call (must not add a new FileHandler)
    setup_logging(second_logfile)
    assert len(clean_logger.handlers) == 1

    # Check that the first path is kept
    assert Path(clean_logger.handlers[0].baseFilename).resolve() == logfile.resolve()


def test_setup_logging_format_and_level(tmp_path, clean_logger):
    """Checks the correctness of the log level and format setup."""
    logfile = tmp_path / "format_test.log"

    setup_logging(logfile)
    handler = clean_logger.handlers[0]

    assert handler.level == logging.DEBUG
    # Format check via the private attribute or output inspection
    assert "%(levelname)-8s" in handler.formatter._fmt


def test_setup_logging_creates_directory(tmp_path, clean_logger):
    """
    Tests the creation of the log directory and file.
    """
    log_dir = tmp_path / "subdir"
    logfile = log_dir / "test.log"

    # Call the function (log_to_console=True by default)
    configurable_automl_engine.training_engine.logger.setup_logging(logfile)

    # Check the filesystem
    assert log_dir.exists()
    assert logfile.exists()

    # Expect 2 handlers: ConcurrentRotatingFileHandler + StreamHandler
    assert len(clean_logger.handlers) == 2


def test_setup_logging_idempotency(tmp_path, clean_logger):
    """
    Checks idempotency (no duplicates on repeated calls).
    """
    logfile = tmp_path / "first.log"

    # First call
    configurable_automl_engine.training_engine.logger.setup_logging(logfile)
    assert len(clean_logger.handlers) == 2

    # Second call
    configurable_automl_engine.training_engine.logger.setup_logging(logfile)

    # FIX: the count must not grow, it must stay 2.
    assert len(clean_logger.handlers) == 2


def test_setup_logging_import_fallback(tmp_path):
    """
    Covers the 'except ImportError' branch.
    Checks that the standard RotatingFileHandler is used when the library is missing.
    """
    # Simulate the missing library by blocking it in sys.modules
    with patch.dict(sys.modules, {"concurrent_log_handler": None}):
        # Reload the logger module so the try-except block is exercised
        importlib.reload(configurable_automl_engine.training_engine.logger)

        # Take HandlerClass (the new name from your code)
        # If you kept the name ConcurrentRotatingFileHandler, use it
        # But judging by the error, you named it differently or the import did not refresh
        handler_class = getattr(
            configurable_automl_engine.training_engine.logger, "HandlerClass"
        )

        # Check that it is now the standard class from logging.handlers
        assert handler_class is logging.handlers.RotatingFileHandler
    # IMPORTANT: restore everything after the test so other tests do not fail
    importlib.reload(configurable_automl_engine.training_engine.logger)


def test_setup_logging_stream_handler_logic_coverage(tmp_path, clean_logger):
    """
    Specific test covering the logic of `if not any(type(h) is StreamHandler)`.
    """
    logfile = tmp_path / "test.log"

    # 1. Call without console
    configurable_automl_engine.training_engine.logger.setup_logging(
        logfile, log_to_console=False
    )
    assert len(clean_logger.handlers) == 1

    # 2. Call with console (enter the covered branch)
    configurable_automl_engine.training_engine.logger.setup_logging(
        logfile, log_to_console=True
    )
    assert len(clean_logger.handlers) == 2

    # 3. Call with console again (check that the branch is skipped)
    configurable_automl_engine.training_engine.logger.setup_logging(
        logfile, log_to_console=True
    )
    assert len(clean_logger.handlers) == 2
