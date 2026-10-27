"""
Centralized Logging Engine: an intelligent event stream manager.
The module provides unified logging for the entire library with support for
safe multi-process writing and hierarchical filtering.
Key features:
    1. Process-Safe Rolling: `ConcurrentRotatingFileHandler` integration to
       prevent log corruption
       when writing concurrently from different processes.
    2. Seamless Fallback: Automatic rollback to the standard `RotatingFileHandler`
       when third-party dependencies are absent, ensuring system stability.
    3. Singleton-Like Hierarchy: Centralized configuration of the base logger
       via `setup_logging`, preventing message duplication in child modules.
    4. Atomic Initialization: existing-handler checks
       prevent re-initialization and "cluttering" of the output (Log Duplication).
"""

import logging
from pathlib import Path
from typing import Any

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Module-level logger within the package hierarchy: messages from
# ``setup_logging`` itself (e.g. the first-path-wins WARNING) must follow the
# same logging configuration as the rest of the library.
_LOGGER = logging.getLogger(__name__)

#: The package root logger: every module logs under ``__name__`` below it.
_PACKAGE_LOGGER_NAME = __name__.partition(".")[0]

# Use Any for the dynamic handler to avoid a type conflict in mypy
HandlerClass: Any


# Import the special handler
try:
    from concurrent_log_handler import ConcurrentRotatingFileHandler

    HandlerClass = ConcurrentRotatingFileHandler
except ImportError:
    # Fall back to the standard handler
    # if the library is not installed
    from logging.handlers import RotatingFileHandler

    HandlerClass = RotatingFileHandler


def get_logger(name: str) -> logging.Logger:
    """Get a named logger within the library hierarchy.
    Returns a logger object that automatically inherits its configuration
    from the base node 'configurable_automl_engine'.

    Args:
        name (str): Unique logger name (usually the module's __name__).
    Returns:
        logging.Logger: A configured logger instance.
    """
    return logging.getLogger(name)


def setup_logging(
    logfile: Path,
    log_to_console: bool = True,
    level: int = logging.DEBUG,
    console_level: int = logging.INFO,
    log_format: str = DEFAULT_FORMAT,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,  # Keep 5 old files
) -> None:
    """Global initialization of the logging system.
    Creates the infrastructure for writing events: builds the file layout,
    configures log rotation and prevents handler duplication.
    Uses a protected writing mechanism for multi-process environments.

    Idempotency contract: the file handler is attached only once per process
    (the first path wins). When ``setup_logging`` is called again with a
    different ``logfile`` while a file handler is already attached, a WARNING
    is logged and the new path is ignored — messages keep flowing into the
    first file to avoid duplicating the output. The file handler is created
    only when it is about to be attached, so repeated calls do not leak file
    descriptors.

    Args:
        logfile (Path): Path to the target log file.
        log_to_console (bool): Whether to enable output via a StreamHandler (STDOUT).
        level (int): Global filtering threshold and file output level.
        console_level (int): Individual console filtering threshold.
        log_format (str): Message string formatting template.
        max_bytes (int): Maximum size of a single file before rotation (in bytes).
        backup_count (int): Number of archived files kept on disk.
    """
    base_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)

    # Set the overall level on the base logger
    base_logger.setLevel(level)

    # Create the directory if it does not exist
    logfile.parent.mkdir(parents=True, exist_ok=True)

    # The file handler is created ONLY when it is about to be attached:
    # constructing it unconditionally on every call would leak an open file
    # descriptor each time the caller passes a path while a file handler is
    # already attached (see the idempotency contract below).
    existing_fh = next(
        (h for h in base_logger.handlers if isinstance(h, logging.FileHandler)),
        None,
    )
    if existing_fh is not None:
        # Documented behavior: the first path wins. A second path cannot be
        # attached without duplicating log output.
        if Path(existing_fh.baseFilename).resolve() != logfile.resolve():
            _LOGGER.warning(
                "A file handler for %s is already attached to "
                "'configurable_automl_engine'; the new path %s is ignored "
                "(the first path wins).",
                existing_fh.baseFilename,
                logfile,
            )
    else:
        fh = HandlerClass(
            filename=str(logfile),
            mode="a",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        formatter = logging.Formatter(log_format)
        fh.setFormatter(formatter)
        base_logger.addHandler(fh)
        base_logger.info(f"File logging initialized: {logfile}")

    # Check whether a console handler is already present
    if log_to_console and not any(
        type(h) is logging.StreamHandler for h in base_logger.handlers
    ):
        sh = logging.StreamHandler()
        sh.setLevel(console_level)
        sh.setFormatter(logging.Formatter(log_format))
        base_logger.addHandler(sh)
