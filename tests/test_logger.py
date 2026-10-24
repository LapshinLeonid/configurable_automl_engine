import importlib
import logging
import logging.handlers
import sys
import weakref
from pathlib import Path
from unittest.mock import patch

import pytest

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
    assert logfile.exists()

    # Check the handlers: ConcurrentRotatingFileHandler + StreamHandler (log_to_console=True by default)
    assert len(clean_logger.handlers) == 2
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
    assert len(clean_logger.handlers) == 2

    # Second call (must not add a new FileHandler, even for a different path)
    setup_logging(second_logfile)
    assert len(clean_logger.handlers) == 2

    # Check that the first path is kept
    assert Path(clean_logger.handlers[0].baseFilename).resolve() == logfile.resolve()


def test_setup_logging_different_path_warns_first_path_wins(
    tmp_path, clean_logger, caplog
):
    """
    Calling setup_logging with a different path while a file handler is already
    attached logs a WARNING that the first path wins and does not switch the file.
    """
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"

    setup_logging(first)
    with caplog.at_level(
        logging.WARNING,
        logger="configurable_automl_engine.training_engine.logger",
    ):
        setup_logging(second)

    # Only one file handler remains, pointing at the FIRST path
    fhs = [h for h in clean_logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(fhs) == 1
    assert Path(fhs[0].baseFilename).resolve() == first.resolve()
    # The WARNING names both the existing and the ignored path
    assert "first path wins" in caplog.text
    assert str(first) in caplog.text
    assert str(second) in caplog.text


def test_setup_logging_same_path_no_warning(tmp_path, clean_logger, caplog):
    """
    Re-calling setup_logging with the SAME path is a silent no-op (no WARNING).
    """
    logfile = tmp_path / "same.log"
    setup_logging(logfile)
    with caplog.at_level(
        logging.WARNING,
        logger="configurable_automl_engine.training_engine.logger",
    ):
        setup_logging(logfile)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == []
    assert len(clean_logger.handlers) == 2


def test_setup_logging_twice_no_file_handler_leak(tmp_path):
    """
    Calling setup_logging twice (with different paths) must not construct a
    second, un-attached file handler: the file handler is created only when it
    is about to be attached.

    The regression is exercised through the standard ``RotatingFileHandler``
    fallback (which opens the file eagerly at construction). The old code
    built a new handler on every call even when one was already attached; the
    abandoned handler stayed pending until the interpreter finalizers ran
    (``Exception ignored ... ResourceWarning: unclosed file ...`` under
    ``-W error::ResourceWarning``). Here the abandoned handler is detected
    deterministically: a weakref to every constructed handler is registered
    via the ``HandlerClass`` constructor, and after the second call
    ``gc.collect()`` must collect nothing beyond the one attached handler.
    """
    import configurable_automl_engine.training_engine.logger as logger_module

    base = logging.getLogger("configurable_automl_engine")
    original_handlers = base.handlers[:]
    base.handlers = []
    constructed: list[weakref.ReferenceType[object]] = []

    original_handler_class = logger_module.HandlerClass

    class TrackingHandler(original_handler_class):
        """HandlerClass wrapper: records a weakref of every constructed instance."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            constructed.append(weakref.ref(self))

    # tmp_path (not a TemporaryDirectory) keeps the log files alive while the
    # attached handler still holds them open: Windows refuses to delete an
    # open file, which used to fail the test before its cleanup ran and leak
    # the handler into later tests.
    try:
        with patch.object(logger_module, "HandlerClass", TrackingHandler):
            logger_module.setup_logging(tmp_path / "first.log")
            first_handlers = len(base.handlers)
            logger_module.setup_logging(tmp_path / "second.log")

        import gc

        gc.collect()
        alive = [r for r in constructed if r() is not None]
        # Exactly ONE handler was ever constructed and it stays attached
        # (the second call must not construct another one).
        assert len(constructed) == 1
        assert len(alive) == 1
        # The single alive handler is the attached file handler
        assert alive[0]() is next(
            h for h in base.handlers if isinstance(h, logging.FileHandler)
        )
        assert first_handlers == 2  # file + console handler from the first call
    finally:
        for h in list(base.handlers):
            h.close()
        base.handlers = original_handlers


def test_setup_logging_format_and_level(tmp_path, clean_logger):
    """Checks the correctness of the log level and format setup."""
    logfile = tmp_path / "format_test.log"

    setup_logging(logfile)
    handler = clean_logger.handlers[0]

    assert handler.level == logging.DEBUG
    # Format check via the private attribute or output inspection
    assert "%(levelname)-8s" in handler.formatter._fmt


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
        handler_class = configurable_automl_engine.training_engine.logger.HandlerClass

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
