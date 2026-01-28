import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str, logfile: Path | None = None) -> logging.Logger:
    """Создаёт цветной консольный + файловый логгер."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console ↦ stderr
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(sh)

        # File ↦ DEBUG-лог
        if logfile:
            logfile.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(logfile, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(_LOG_FORMAT))
            logger.addHandler(fh)

    # Не даём вспухать дублями
    logger.propagate = False
    return logger
