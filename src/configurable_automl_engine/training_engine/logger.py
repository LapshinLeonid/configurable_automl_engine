"""
    Centralized Logging Engine: Интеллектуальный менеджер потоков событий.
    Модуль обеспечивает унифицированное логирование для всей библиотеки с поддержкой 
    безопасной многопроцессной записи и иерархической фильтрации.
    Ключевые возможности:
        1. Process-Safe Rolling: Интеграция `ConcurrentRotatingFileHandler` для 
           предотвращения повреждения логов 
           при одновременной записи из разных процессов.
        2. Seamless Fallback: Автоматический откат на стандартный `RotatingFileHandler` 
           при отсутствии сторонних зависимостей, обеспечивающий стабильность системы.
        3. Singleton-Like Hierarchy: Централизованная настройка базового логгера 
           через `setup_logging`, исключающая дублирование сообщений в дочерних модулях.
        4. Atomic Initialization: Механизмы проверки существующих обработчиков 
           предотвращают повторную инициализацию и "засорение" вывода (Log Duplication).
"""

import logging
from pathlib import Path
from typing import Any, Type

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# Используем Any для динамического хендлера, чтобы избежать конфликта типов в mypy
HandlerClass: Any 


# Импортируем специальный обработчик
try:
    from concurrent_log_handler import ConcurrentRotatingFileHandler
    HandlerClass = ConcurrentRotatingFileHandler
except ImportError:
    # Фолбэк на стандартный обработчик, 
    # если библиотека не установлена
    from logging.handlers import RotatingFileHandler
    HandlerClass = RotatingFileHandler

def get_logger(name: str) -> logging.Logger:
    """Получить именованный логгер в рамках иерархии библиотеки.
    Обеспечивает получение объекта логгера, который автоматически наследует 
    конфигурацию от базового узла 'configurable_automl_engine'.
    
    Args:
        name (str): Уникальное имя логгера (обычно __name__ модуля).
    Returns:
        logging.Logger: Настроенный экземпляр логгера.
    """
    return logging.getLogger(name)

def setup_logging(logfile: Path,
                  log_to_console: bool = True,
                  level: int = logging.DEBUG,
                  console_level: int = logging.INFO,
                  log_format: str = DEFAULT_FORMAT,
                  max_bytes: int = 10 * 1024 * 1024,  # 10 MB
                  backup_count: int = 5,              # Хранить 5 старых файлов
                  ) -> None:
    """Глобальная инициализация системы логирования.
    Создает инфраструктуру для записи событий: формирует файловую структуру, 
    настраивает ротацию логов и предотвращает дублирование обработчиков. 
    Использует защищенный механизм записи для многопроцессных сред.
    Args:
        logfile (Path): Путь к целевому файлу лога.
        log_to_console (bool): Флаг активации вывода в StreamHandler (STDOUT).
        level (int): Общий порог фильтрации и уровень для файлового вывода.
        console_level (int): Индивидуальный порог фильтрации для консоли.
        log_format (str): Шаблон форматирования строки сообщения.
        max_bytes (int): Лимит размера одного файла перед ротацией (в байтах).
        backup_count (int): Количество архивных файлов, хранящихся на диске.
    """
    base_logger = logging.getLogger("configurable_automl_engine")

    # Устанавливаем общий уровень доступа "снизу"
    base_logger.setLevel(level)
    
    # Создаем директорию, если она не существует
    logfile.parent.mkdir(parents=True, exist_ok=True)
    
    # Настраиваем обработчик файла
    fh = HandlerClass(filename=str(logfile), 
                      mode="a", 
                      maxBytes=max_bytes, 
                      backupCount=backup_count,
                      encoding="utf-8")
    fh.setLevel(level)
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    
    # Добавляем обработчик, если такой еще не добавлен
    if not any(isinstance(h, logging.FileHandler) 
            for h in base_logger.handlers):
        base_logger.addHandler(fh)
        base_logger.info(f"Логирование в файл инициализировано: {logfile}")

    if log_to_console:
        # Проверяем, нет ли уже консольного обработчика
        if not any(type(h) is logging.StreamHandler for h in base_logger.handlers):
            sh = logging.StreamHandler()
            sh.setLevel(console_level)
            sh.setFormatter(logging.Formatter(log_format))
            base_logger.addHandler(sh)