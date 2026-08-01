"""Модуль для параллельного выполнения задач с управлением памятью.
Предоставляет инструменты для параллельного запуска функций (ThreadPool/ProcessPool)
с оптимизированной передачей тяжелых объектов pandas.DataFrame через механизмы
Shared Memory (разделяемая память) и Disk Persistence (дисковый кэш).
Основные компоненты:
    - SharedDataFrame: Контейнер для размещения данных в разделяемой памяти ОС.
    - DiskPersistenceManager: Менеджер временного хранения данных в Parquet.
    - run_parallel: Универсальный интерфейс для запуска вычислений.
Пример использования:
    results = run_parallel(
        func=my_heavy_function,
        args_seq=[(df1,), (df2,)],
        mode="processes",
        shared_args_indices=[0]
    )
"""

import concurrent.futures
import logging
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from multiprocessing import shared_memory
from typing import Any

import numpy as np
import pandas as pd

from configurable_automl_engine.tuner import InvalidAlgorithmError

logger = logging.getLogger(__name__)


class SharedDataFrame:
    """Обертка для размещения DataFrame в разделяемой памяти (Shared Memory).

    Позволяет передавать большие объемы данных между процессами без накладных
    расходов на сериализацию (pickle), используя механизм POSIX Shared Memory.

    Attributes:
        name (str | None): Уникальное имя сегмента разделяемой памяти в ОС.
        shm (shared_memory.SharedMemory): Объект управления сегментом памяти.
        shared_array (np.ndarray): NumPy-представление данных, указывающее на SHM буфер.
        (может быть помечено как read-only при восстановлении).
        shape (tuple): Размерность исходного массива данных.
        dtype (np.dtype): Тип данных элементов массива.
        columns (list[str]): Список имен столбцов для восстановления DataFrame.
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        name: str | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | Any = None,
        columns: list[str] | None = None,
    ) -> None:
        """Инициализировать объект разделяемой памяти для DataFrame.
        Логика инициализации:
        1. Создание: Если передан df, создается новый сегмент Shared Memory,
           с уникальным именем (shm_ + id + random), куда копируются данные
        2. Подключение: Если df не передан, выполняется подключение к существующему
           сегменту по имени (name) с использованием метаданных (shape, dtype, columns).
        3. Метаданные: Список имен столбцов сохраняется для последующего
           восстановления структуры DataFrame.
        Args:
            df (pd.DataFrame | None): Исходный DataFrame для размещения в SHM.
            name (str | None): Имя существующего сегмента памяти.
            shape (tuple | None): Размерность массива данных.
            dtype (np.dtype | Any): Тип данных элементов массива.
            columns (list[str] | None): Список имен столбцов.
        Returns:
            None
        """

        self.name: str | None = None
        self._owner = df is not None
        self.columns: list[str] | None = None

        if df is not None:
            self.name = f"shm_{id(df)}_{np.random.randint(1000)}"
            data = df.to_numpy()
            self.shm = shared_memory.SharedMemory(
                create=True, size=data.nbytes, name=self.name
            )
            self.shared_array = np.ndarray(
                data.shape, dtype=data.dtype, buffer=self.shm.buf
            )
            self.shared_array[:] = data[:]
            self.shape = data.shape
            self.dtype = data.dtype
            self.columns = df.columns.tolist()
        else:
            self.name = name
            self.shm = shared_memory.SharedMemory(name=name)
            actual_shape = shape if shape is not None else ()
            self.shared_array = np.ndarray(
                actual_shape, dtype=dtype, buffer=self.shm.buf
            )
            self.shape = self.shared_array.shape
            self.columns = columns

    @staticmethod
    def is_shared_array(X: Any) -> bool:
        """
        Проверяет, является ли объект массивом numpy, использующим SharedMemory.
        Это критично для исключения повторного копирования в ModelTrainer.
        """
        if not isinstance(X, np.ndarray):
            return False
        # Проверяем, указывает ли буфер массива на сегмент разделяемой памяти
        # (в Python 3.8+ SharedMemory.buf возвращает memoryview)
        return hasattr(X, "base") and isinstance(X.base, memoryview)

    @staticmethod
    def get_data_info(X: Any) -> tuple[int, list[str] | list[int]]:
        """
        Извлекает количество признаков и их имена (или индексы) без копирования.
        """
        if isinstance(X, pd.DataFrame):
            return X.shape[1], X.columns.tolist()
        elif isinstance(X, SharedDataFrame):
            shape = getattr(X, "shape", X.shared_array.shape)
            n_cols = shape[1] if len(shape) > 1 else 1
            cols = (
                X.columns if X.columns is not None else [str(i) for i in range(n_cols)]
            )
            return n_cols, cols
        elif isinstance(X, np.ndarray):
            n_cols = X.shape[1] if X.ndim > 1 else 1
            return n_cols, list(range(n_cols))
        return 0, []

    @staticmethod
    def is_compatible(df: Any) -> bool:
        """Проверяет, можно ли разместить DF в SHM
        (только простые типы и RangeIndex)."""
        """Проверяет, можно ли разместить DF в SHM 
        (уже разделяемый массив, либо DF с простыми типами и RangeIndex)."""
        if SharedDataFrame.is_shared_array(df):
            return True
        if not isinstance(df, pd.DataFrame):
            return False

        # Проверка типов данных (белый список: int, uint, float, bool)
        allowed_kinds = {"i", "u", "f", "b"}
        if not all(dt.kind in allowed_kinds for dt in df.dtypes):
            return False

        # Проверка индекса: SHM в текущей реализации не поддерживает сложные индексы
        # Если индекс не является RangeIndex, объект признается несовместимым с SHM
        # и будет автоматически перенаправлен в DiskPersistenceManager.
        return isinstance(df.index, pd.RangeIndex)

    def to_df(self) -> pd.DataFrame:
        """Восстановить pandas.DataFrame из разделяемой памяти.
        Логика восстановления:
        1. Проксирование: Создается объект np.ndarray, для которого принудительно
        устанавливается флаг writeable=False для предотвращения
        лишнего копирования в pandas.
        2. Реконструкция: На базе массива и сохраненного списка столбцов формируется
           новый объект DataFrame.
        3. Изоляция: Итоговый DataFrame является независимым объектом в памяти
           текущего процесса.
        Returns:
            pd.DataFrame: Восстановленный набор данных.
        """

        # Устанавливаем флаг writeable=False. Это критично: pandas часто делает
        # скрытую копию, если «опасается», что кто-то изменит общий буфер.
        self.shared_array.setflags(write=False)

        # copy=False в конструкторе и использование однородного numpy-массива
        # гарантирует создание DataFrame без выделения новой памяти под данные.
        return pd.DataFrame(self.shared_array, columns=self.columns, copy=False)

    def close(self) -> None:
        """Закрыть доступ к сегменту разделяемой памяти.
        Логика закрытия:
        1. Хендл: Закрывает дескриптор доступа к SHM в текущем процессе.
        2. Сохранность: Сами данные в ОС не уничтожаются, что позволяет другим
           процессам продолжать работу с сегментом.
        Returns:
            None
        """

        if hasattr(self, "shm"):
            self.shm.close()

    def unlink(self) -> None:
        """Уничтожить сегмент разделяемой памяти в операционной системе.
        Логика удаления:
        1. Владение: Операция выполняется только процессом-создателем (_owner=True).
        2. Освобождение: Помечает сегмент для удаления; память будет полностью
           освобождена ОС, когда все процессы закроют свои ссылки на него.
        Returns:
            None
        """

        if self._owner and hasattr(self, "shm"):
            try:
                self.shm.unlink()
            except (FileNotFoundError, OSError):
                pass  # Уже удалено

    def get_view(self, columns: list[str] | None = None) -> pd.DataFrame:
        """
        Возвращает представление (view) данных.
        Если переданы columns, возвращает view только для этих столбцов.
        """
        if columns is None:
            return self.to_df()

        # Важно: используем .loc для создания slice-view, а не копии
        return self.to_df().loc[:, columns]


class DiskPersistenceManager:
    """Утилита для временного сохранения DataFrame на диск в формате Parquet.

    Используется как альтернатива Shared Memory, когда данные слишком велики
    для оперативной памяти или требуется строгая типизация через дисковый кэш.

    Attributes:
        tmp_dir (str | None): Путь к временной директории (например, /dev/shm).
        created_files (list[str]): Список путей к созданным временным файлам.
    """

    def __init__(self, use_shm: bool = True):
        # Используем /dev/shm для Linux если доступно, иначе стандартный temp
        self.tmp_dir = "/dev/shm" if use_shm and os.path.exists("/dev/shm") else None
        self.created_files: list[str] = []

    def save_df(self, df: pd.DataFrame) -> str:
        """Сохранить DataFrame во временный файл Parquet.
        Логика сохранения:
        1. Локация: Файл создается в /dev/shm (RAM-диск) для ускорения операций
           в Linux или в системной временной папке.
        2. Формат: Используется Parquet с автоматическим выбором движка
        (fastparquet или pyarrow) в зависимости от доступности библиотек.
        3. Регистрация: Путь к файлу добавляется в список для последующей очистки.
        Args:
            df (pd.DataFrame): Набор данных для сохранения.
        Returns:
            str: Абсолютный путь к созданному временному файлу.
        """

        fd, path = tempfile.mkstemp(suffix=".parquet", dir=self.tmp_dir)
        os.close(fd)
        df.to_parquet(
            path, engine=("fastparquet" if "fastparquet" in globals() else "pyarrow")
        )
        self.created_files.append(path)
        return path

    def cleanup(self) -> None:
        """Удалить все созданные временные файлы.
        Логика очистки:
        1. Итерация: Проходит по списку путей, зарегистрированных в created_files.
        2. Безопасность: Подавляет ошибки отсутствия файла, если он был удален ранее.
        Returns:
            None
        """
        for path in self.created_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except (OSError, FileNotFoundError) as e:
                logger.warning(f"Failed to delete temp file {path}: {e}")


def _worker_proxy(
    func: Callable[..., Any],
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    disk_indices: list[int] | None,
    shm_info: dict[int, tuple],
) -> Any:
    """Десериализовать данные и выполнить целевую функцию внутри воркера.
    Логика выполнения:
    1. Shared Memory: Находит объекты SharedDataFrame по индексам и
       конвертирует их обратно в DataFrame.
    2. Disk: Читает Parquet-файлы по переданным путям и восстанавливает DataFrame.
    3. Вызов: Передает восстановленные данные в целевую функцию func.
    4. Очистка: Закрывает локальные хендлы Shared Memory и принудительно
        очищает список аргументов для ускорения работы Garbage Collector.
    Args:
        func (Callable): Целевая функция для выполнения.
        args (Sequence): Список аргументов (включая прокси-объекты).
        kwargs (Mapping): Именованные аргументы.
        disk_indices (list[int]): Индексы аргументов, сохраненных на диск.
        shm_info (dict[int, tuple]): Словарь, где ключ — индекс аргумента,
            а значение — кортеж с метаданными SHM (имя, shape, dtype, columns)
            для восстановления.
    Returns:
        Any: Результат выполнения функции func.
    """

    final_args = list(args)
    # Используем контекстный менеджер для автоматического закрытия дескрипторов SHM
    # Это освобождает системные ресурсы сразу после вызова функции.
    shm_wrappers: list[SharedDataFrame] = []
    try:
        # 1. Восстановление из Shared Memory
        # Восстановление SHM по метаданным (имя, shape, dtype, columns)
        if shm_info:
            for idx, meta in shm_info.items():
                wrapper = SharedDataFrame(
                    name=meta[0], shape=meta[1], dtype=meta[2], columns=meta[3]
                )
                final_args[idx] = wrapper.to_df()
                shm_wrappers.append(wrapper)
        # 2. Загрузка с диска
        if disk_indices:
            for idx in disk_indices:
                path = final_args[idx]
                if isinstance(path, str) and path.endswith(".parquet"):
                    final_args[idx] = pd.read_parquet(path)
        return func(*final_args, **kwargs)
    finally:
        # Важно: закрываем только локальные ссылки (дескрипторы) воркера.
        # Сами данные в SHM остаются живы,
        # пока их не удалит главный процесс через .unlink()
        for w in shm_wrappers:
            try:
                w.close()
            except Exception:  # noqa: S110, BLE001
                pass
        final_args.clear()  # Помогаем GC быстрее освободить ссылки


def _perform_cleanup(
    shm_refs: list[SharedDataFrame] | None,
    persistence_manager: DiskPersistenceManager | None,
) -> None:
    """Вспомогательная функция для безопасной очистки с защитой от системных ошибок."""
    if shm_refs:
        for ref in shm_refs:
            # Сначала закрываем дескриптор
            try:
                ref.close()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"SHM close error (expected during forced shutdown): {e}")

            # Затем пытаемся уничтожить сегмент в ОС
            try:
                ref.unlink()
            except (FileNotFoundError, OSError):
                # Игнорируем, если уже удалено или нет доступа
                pass
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Non-critical SHM unlink failure: {e}")

    if persistence_manager:
        try:
            persistence_manager.cleanup()
        except (PermissionError, OSError) as e:
            logger.error(f"Cleanup failed due to file locking/permissions: {e}")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Unexpected persistence cleanup error: {e}")


def _force_shutdown_processes(pool, shutdown_grace_period=5.0):
    """Принудительное завершение всех процессов в пуле.

    Используется watchdog-механизмом при обнаружении зависших задач
    (C-level crash воркера).

    Args:
        pool: ProcessPoolExecutor или ThreadPoolExecutor
        shutdown_grace_period: время ожидания graceful shutdown в секундах
    """
    workers = list(getattr(pool, "_processes", {}).values())
    pool.shutdown(wait=False, cancel_futures=True)

    stop_time = time.time() + shutdown_grace_period
    while time.time() < stop_time and any(w.is_alive() for w in workers):
        time.sleep(0.1)

    for w in workers:
        if w.is_alive():
            try:
                logger.error(f"CRITICAL: Worker {w.pid} hung. Forcing SIGTERM.")
                w.terminate()
            except Exception:  # noqa: S110, BLE001
                pass

    time.sleep(0.2)

    for w in workers:
        if w.is_alive():
            try:
                logger.critical(
                    f"HARD KILL: Worker {w.pid} resisted SIGTERM. Sending SIGKILL."
                )
                w.kill()
            except Exception:  # noqa: S110, BLE001
                pass


def run_parallel(
    func: Callable[..., Any],
    args_seq: Iterable[Sequence[Any]] | None = None,
    kwargs_seq: Iterable[Mapping[str, Any]] | None = None,
    max_workers: int | None = None,
    mode: str = "threads",
    timeout: float | None = 3600,
    shared_args_indices: list[int] | None = None,
    disk_args_indices: list[int] | None = None,
    shutdown_grace_period: float = 5.0,
    task_timeout: float | None = None,  # индивидуальный таймаут на задачу
) -> list[Any]:
    """Организовать параллельное выполнение функции
    с управлением памятью и жизненным циклом.

    Логика параллелизма:
    1. Режим: Поддерживает многопоточность (threads) и многопроцессорность (processes).
    2. Оптимизация данных: При работе с процессами переносит DataFrame в Shared Memory
       или на диск, исключая накладные расходы на Pickle-сериализацию.
    3. Управление жизненным циклом: Использует явную инициализацию и
       завершение Executor (shutdown с cancel_futures=True). Это позволяет
       немедленно прерывать выполнение при сбоях или таймаутах.
    4. Глобальный таймер: Параметр timeout ограничивает суммарное время выполнения
       всей последовательности задач. Если лимит превышен, сбор результатов
       прекращается, а оставшиеся в очереди задачи отменяются.
    5. Отказоустойчивость: При сбое инициализации или выполнении пула процессов
       выполняется попытка перезапуска всей последовательности в режиме "threads"
       с принудительной очисткой ресурсов.
    6. Ресурсный менеджмент: Гарантирует удаление сегментов памяти и временных
       файлов, а также корректную остановку всех рабочих процессов/потоков.

    Args:
        func (Callable): Функция для запуска.
        args_seq (Iterable): Последовательность кортежей аргументов.
        kwargs_seq (Iterable): Последовательность словарей именованных аргументов.
        max_workers (int): Лимит количества рабочих воркеров.
        mode (str): Режим параллелизма ("threads" или "processes").
        timeout (int | float): Глобальный лимит времени в секундах
        на выполнение всех задач. Если не задан — задачи выполняются без глобального ограничения.
        shared_args_indices (list[int]): Индексы DataFrame для Shared Memory.
        disk_args_indices (list[int]): Индексы DataFrame для дискового кэша.
        shutdown_grace_period (float): Время в секундах, отводимое на мягкое завершение
            воркеров перед принудительным отправлением SIGTERM/SIGKILL.
        task_timeout (float | None): Индивидуальный таймаут на одну задачу (алгоритм) в секундах.
            Если None — используется глобальный timeout. Deadline для каждой задачи считается
            от момента её отправки в пул (submit_time), а не от общего start_time.

    Returns:
        list[Any]: Список результатов. Задачи, не успевшие выполниться
            или вызвавшие стандартные исключения, заменяются на None.
            Критические ошибки (InvalidAlgorithmError) и прерывания пользователя
            (KeyboardInterrupt) пробрасываются вызывающему коду.
    """
    pool = None

    args_seq = list(args_seq or [()])
    kwargs_seq = list(kwargs_seq or [{}] * len(args_seq))

    if len(args_seq) != len(kwargs_seq):
        raise ValueError("args_seq and kwargs_seq must be of equal length")

    # Логика подготовки Shared Memory для процессов
    shm_refs = []
    persistence_manager = DiskPersistenceManager()

    if mode == "processes" and (shared_args_indices or disk_args_indices):
        task_payloads = []  # Список кортежей
        # (args, kwargs, actual_shm_idx, actual_disk_idx)
        target_shm_indices = shared_args_indices or []
        target_disk_indices = disk_args_indices or []

        for args, kwargs in zip(args_seq, kwargs_seq):
            new_args = list(args)
            curr_shm_info = {}  # Индекс -> (имя, shape, dtype, columns)
            for idx in set(target_shm_indices) | set(target_disk_indices):
                if idx < len(new_args) and isinstance(new_args[idx], pd.DataFrame):
                    if idx in target_shm_indices and SharedDataFrame.is_compatible(
                        new_args[idx]
                    ):
                        shm_wrapper = SharedDataFrame(new_args[idx])
                        shm_refs.append(shm_wrapper)
                        # Сохраняем метаданные для передачи в воркер
                        curr_shm_info[idx] = (
                            shm_wrapper.name,
                            shm_wrapper.shape,
                            shm_wrapper.dtype,
                            shm_wrapper.columns,
                        )
                        new_args[idx] = None  # Сам объект не передаем
                    else:
                        path = persistence_manager.save_df(new_args[idx])
                        new_args[idx] = path

            curr_disk = [
                i
                for i in (set(target_shm_indices) | set(target_disk_indices))
                if i < len(new_args)
                and isinstance(new_args[i], str)
                and new_args[i].endswith(".parquet")
            ]

            task_payloads.append((tuple(new_args), kwargs, curr_disk, curr_shm_info))

        # Переопределяем итерируемый объект для запуска
        execution_tasks = task_payloads
    else:
        # Для потоков или обычных процессов без SHM/Disk
        execution_tasks = [
            (tuple(a), kw, disk_args_indices or [], {})
            for a, kw in zip(args_seq, kwargs_seq)
        ]
    # Преаллокация списка для сохранения длины и порядка
    results: list[Any] = [None] * len(execution_tasks)

    # 1. Определяем класс исполнителя
    executor_cls: Callable[[int | None], Executor] = ThreadPoolExecutor
    if mode == "processes":
        try:
            executor_cls = ProcessPoolExecutor
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"Could not initialize ProcessPoolExecutor:"
                f" {e}. Falling back to threads."
            )
            executor_cls = ThreadPoolExecutor

    start_time = time.time()

    try:
        pool = executor_cls(max_workers)

        future_to_idx = {}
        submit_time_by_future: dict[Future, float] = {}
        for i, (a, kw, d_idx, s_idx) in enumerate(execution_tasks):
            if mode == "processes" and (shared_args_indices or disk_args_indices):
                fut = pool.submit(_worker_proxy, func, a, kw, d_idx, s_idx)
            else:
                fut = pool.submit(func, *a, **kw)
            future_to_idx[fut] = i
            submit_time_by_future[fut] = time.time()

        # Абсолютные дедлайны для каждой задачи (от времени submit)
        deadline_by_future: dict[Future, float] = {}
        for fut, idx in future_to_idx.items():
            task_t = task_timeout or float("inf")
            deadline_by_future[fut] = (
                submit_time_by_future.get(fut, start_time) + task_t
            )

        while future_to_idx:
            now = time.time()
            elapsed = now - start_time

            # Проверка: если глобальный таймаут истёк — выходим
            effective_timeout = timeout
            if effective_timeout is not None and elapsed >= effective_timeout:
                for fut, idx in future_to_idx.items():
                    logger.error(f"Task {idx} timed out")
                    results[idx] = None
                future_to_idx.clear()
                break

            # Проверка: если индивидуальный таймаут задачи (task_timeout) истёк — выходим
            expired_futures = []
            for fut, deadline in deadline_by_future.items():
                if fut in future_to_idx and now >= deadline:
                    idx = future_to_idx.pop(fut)
                    logger.error(f"Task {idx} timed out")
                    results[idx] = None
                    expired_futures.append(fut)
            if not future_to_idx:
                break

            # Вычисляем минимальный оставшийся таймаут среди оставшихся задач
            remaining_by_future = {}
            for fut, deadline in deadline_by_future.items():
                if fut in future_to_idx:
                    rem = max(0.1, deadline - now)
                    remaining_by_future[fut] = rem
            min_remaining = (
                min(remaining_by_future.values()) if remaining_by_future else None
            )

            # Если глобальный таймаут меньше — используем его
            if effective_timeout is not None:
                global_remaining = max(0.1, effective_timeout - elapsed)
                if min_remaining is not None:
                    min_remaining = min(min_remaining, global_remaining)
                else:
                    min_remaining = global_remaining  # pragma: no cover

            timeout_for_ac = (
                min_remaining
                if min_remaining is not None and min_remaining < float("inf")
                else None
            )

            try:
                if timeout_for_ac is not None:
                    for fut in as_completed(
                        future_to_idx.keys(), timeout=timeout_for_ac
                    ):
                        idx = future_to_idx.pop(fut)
                        try:
                            results[idx] = fut.result(timeout=0)
                        except (InvalidAlgorithmError, KeyboardInterrupt):
                            raise
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"Task {idx} failed: {e}")
                            results[idx] = None
                else:
                    for fut in as_completed(future_to_idx.keys()):
                        idx = future_to_idx.pop(fut)
                        try:
                            results[idx] = fut.result(timeout=0)
                        except (InvalidAlgorithmError, KeyboardInterrupt):
                            raise
                        except Exception as e:  # noqa: BLE001
                            logger.error(f"Task {idx} failed: {e}")
                            results[idx] = None

            except concurrent.futures.TimeoutError:
                # as_completed не дождался ни одной задачи за timeout_for_ac
                # Просто продолжаем ожидание — это не C-level падение, а медленные задачи
                continue

            except (InvalidAlgorithmError, KeyboardInterrupt) as e:
                if isinstance(e, KeyboardInterrupt):
                    logger.error("Interrupted by user")
                raise
            except Exception as e:  # noqa: BLE001
                logger.error(f"Error while waiting for tasks: {e}")
                future_to_idx.clear()
                break

    except Exception as e:
        if mode == "processes":
            logger.error("Error in process pool, falling back to threads: %s", e)

            _perform_cleanup(shm_refs, persistence_manager)
            return run_parallel(
                func,
                args_seq,
                kwargs_seq,
                max_workers,
                mode="threads",
                timeout=timeout,
                task_timeout=task_timeout,
            )
        raise
    finally:
        if pool is not None:
            is_proc_executor = type(pool).__name__ == "ProcessPoolExecutor" or hasattr(
                pool, "_processes"
            )
            if mode == "processes" and is_proc_executor:
                _force_shutdown_processes(pool, shutdown_grace_period)
                # Финальная очистка SHM/Disk (только в режиме процессов)
                _perform_cleanup(shm_refs, persistence_manager)
            else:
                # Для ThreadPoolExecutor или если режим не "processes"
                pool.shutdown(wait=True, cancel_futures=True)
    return results
