import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

def get_logger(name: str) -> logging.Logger:
    """
    Возвращает логгер по имени. 
    Настройка обработчиков (handlers) теперь должна производиться централизованно.
    """
    return logging.getLogger(name)

def setup_logging(logfile: Path) -> None:
    """
    Настраивает логирование в файл для всей библиотеки.
    Прикрепляет FileHandler к базовому логгеру 'configurable_automl_engine'.
    """
    base_logger = logging.getLogger("configurable_automl_engine")
    
    # Создаем директорию, если она не существует
    logfile.parent.mkdir(parents=True, exist_ok=True)
    
    # Настраиваем обработчик файла
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(_LOG_FORMAT)
    fh.setFormatter(formatter)
    
    # Добавляем обработчик, если такой еще не добавлен
    if not any(isinstance(h, logging.FileHandler) for h in base_logger.handlers):
        base_logger.addHandler(fh)
        base_logger.info(f"Логирование в файл инициализировано: {logfile}")