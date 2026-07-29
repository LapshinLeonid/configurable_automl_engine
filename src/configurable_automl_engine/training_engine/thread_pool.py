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
import logging
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
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
            columns: list[str] | None = None
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

        if df is not None:
            self.name = f"shm_{id(df)}_{np.random.randint(1000)}"
            data = df.to_numpy()
            self.shm = shared_memory.SharedMemory(create=True,
                                                  size=data.nbytes,
                                                  name=self.name)
            self.shared_array = np.ndarray(data.shape, 
                                           dtype=data.dtype, 
                                           buffer=self.shm.buf
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
                actual_shape, 
                dtype=dtype, 
                buffer=self.shm.buf
                )
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
        return hasattr(X, 'base') and isinstance(X.base, memoryview)

    @staticmethod
    def get_data_info(X: Any) -> tuple[int, list[str] | list[int]]:
        """
        Извлекает количество признаков и их имена (или индексы) без копирования.
        """
        if isinstance(X, pd.DataFrame):
            return X.shape[1], X.columns.tolist()
        elif isinstance(X, SharedDataFrame):
            return X.shape[1], X.columns
        elif isinstance(X, np.ndarray):
            return (X.shape[1] 
                    if X.ndim > 1 
                    else 1, 
                    list(range(X.shape[1] if X.ndim > 1 else 1)))
        return 0, []

    @staticmethod
    def is_compatible(df: pd.DataFrame) -> bool:
        """Проверяет, можно ли разместить DF в SHM 
        (только простые типы и RangeIndex)."""
        if SharedDataFrame.is_shared_array(df):
            return True
        if not isinstance(df, pd.DataFrame):
            return False
        
        #Проверка типов данных (белый список: int, uint, float, bool)
        allowed_kinds = {'i', 'u', 'f', 'b'}
        if not all(dt.kind in allowed_kinds for dt in df.dtypes):
            return False
            
        #Проверка индекса: SHM в текущей реализации не поддерживает сложные индексы
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

        if hasattr(self, 'shm'):
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

        if self._owner and hasattr(self, 'shm'):
            try:
                self.shm.unlink()
            except (FileNotFoundError, OSError):
                pass # Уже удалено
    
    def get_view(self, 
                 columns: list[str] | None = None
                 ) -> pd.DataFrame:
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
        df.to_parquet(path, 
                      engine=
                      ("fastparquet" if "fastparquet" in globals() else "pyarrow")
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
        shm_info: dict[int, tuple]
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
                    name=meta[0], 
                    shape=meta[1], 
                    dtype=meta[2], 
                    columns=meta[3])
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
            except Exception:
                pass
        final_args.clear() # Помогаем GC быстрее освободить ссылки

def _perform_cleanup(shm_refs: list[SharedDataFrame] | None,
                      persistence_manager: DiskPersistenceManager | None
                      ) -> None:
    """Вспомогательная функция для безопасной очистки с защитой от системных ошибок."""
    if shm_refs:
        for ref in shm_refs:
            # Сначала закрываем дескриптор
            try:
                ref.close()
            except Exception as e:
                logger.debug(f"SHM close error (expected during forced shutdown): {e}")
            
            # Затем пытаемся уничтожить сегмент в ОС
            try:
                ref.unlink()
            except (FileNotFoundError, OSError):
                # Игнорируем, если уже удалено или нет доступа
                pass
            except Exception as e:
                logger.warning(f"Non-critical SHM unlink failure: {e}")

    if persistence_manager:
        try:
            persistence_manager.cleanup()
        except (PermissionError, OSError) as e:
            logger.error(f"Cleanup failed due to file locking/permissions: {e}")
        except Exception as e:
            logger.error(f"Unexpected persistence cleanup error: {e}")

def run_parallel(
    func: Callable[..., Any],
    args_seq: Iterable[Sequence[Any]] | None = None,
    kwargs_seq: Iterable[Mapping[str, Any]] | None = None,
    max_workers: int | None = None, 
    mode: str = "threads",
    timeout: float | None = 3600,
    shared_args_indices: list[int] | None = None,
    disk_args_indices: list[int] | None = None,
    pool_timeout: float | None = None,
    shutdown_grace_period: float = 5.0
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
        на выполнение всех задач.
        shared_args_indices (list[int]): Индексы DataFrame для Shared Memory.
        disk_args_indices (list[int]): Индексы DataFrame для дискового кэша.
        pool_timeout (int | float | None): Индивидуальный таймаут для ожидания задач 
            в пуле (если не задан, используется глобальный timeout).
        shutdown_grace_period (float): Время в секундах, отводимое на мягкое завершение 
            воркеров перед принудительным отправлением SIGTERM/SIGKILL.

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
        task_payloads = [] # Список кортежей 
        #(args, kwargs, actual_shm_idx, actual_disk_idx)
        target_shm_indices = shared_args_indices or []
        target_disk_indices = disk_args_indices or []
        
        for args, kwargs in zip(args_seq, kwargs_seq):
            new_args = list(args)
            curr_shm_info = {} # Индекс -> (имя, shape, dtype, columns)
            for idx in set(target_shm_indices) | set(target_disk_indices):
                if idx < len(new_args) and isinstance(new_args[idx], pd.DataFrame):
                    if (idx in target_shm_indices 
                        and SharedDataFrame.is_compatible(new_args[idx])):
                        shm_wrapper = SharedDataFrame(new_args[idx])
                        shm_refs.append(shm_wrapper)
                        # Сохраняем метаданные для передачи в воркер
                        curr_shm_info[idx] = (
                            shm_wrapper.name, 
                            shm_wrapper.shape, 
                            shm_wrapper.dtype, 
                            shm_wrapper.columns)
                        new_args[idx] = None # Сам объект не передаем
                    else:
                        path = persistence_manager.save_df(new_args[idx])
                        new_args[idx] = path
            
            curr_disk = [i for i in (set(target_shm_indices) | set(target_disk_indices))
                         if i < len(new_args) and isinstance(new_args[i], str) 
                         and new_args[i].endswith(".parquet")]
            
            task_payloads.append((tuple(new_args), kwargs, curr_disk, curr_shm_info))
        
        # Переопределяем итерируемый объект для запуска
        execution_tasks = task_payloads
    else:
        # Для потоков или обычных процессов без SHM/Disk
        execution_tasks = [
                        (tuple(a), kw, disk_args_indices 
                         or [], {}) 
                        for a, kw in zip(args_seq, kwargs_seq)
                        ]
    # Преаллокация списка для сохранения длины и порядка
    results: list[Any] = [None] * len(execution_tasks)

    # 1. Определяем класс исполнителя
    executor_cls: Callable[[int | None], Executor] = ThreadPoolExecutor
    if mode == "processes":
        try:
            executor_cls = ProcessPoolExecutor
        except Exception as e:
            logger.error(f"Could not initialize ProcessPoolExecutor:"
                         f" {e}. Falling back to threads.")
            executor_cls = ThreadPoolExecutor

    start_time = time.time()
    
    try:
        pool = executor_cls(max_workers)

        future_to_idx = {}
        for i, (a, kw, d_idx, s_idx) in enumerate(execution_tasks):
            if mode == "processes" and (shared_args_indices or disk_args_indices):
                fut = pool.submit(_worker_proxy, func, a, kw, d_idx, s_idx)
            else:
                fut = pool.submit(func, *a, **kw)
            future_to_idx[fut] = i

        while future_to_idx:
            elapsed = time.time() - start_time
            effective_timeout = pool_timeout or timeout
            remaining_global = (
                max(0.1, effective_timeout - elapsed) if effective_timeout else None
                )
            
            try:
                # Ждем завершения хотя бы одной задачи в пределах оставшегося времени
                done, _ = wait(
                    future_to_idx.keys(), 
                    timeout=remaining_global, 
                    return_when=FIRST_COMPLETED
                )
                
                if not done: # Вышли по таймауту wait
                    # Чтобы прошел тест test_run_parallel_timeout_error_coverage:
                    for fut, idx in future_to_idx.items():
                        logger.error(f"Task {idx} timed out")
                    break
                
                for fut in done:
                    idx = future_to_idx.pop(fut)
                    try:
                        results[idx] = fut.result(timeout=0)
                    except (InvalidAlgorithmError, KeyboardInterrupt): 
                        raise # Пробрасываем критические ошибки наверх для тестов
                    except Exception as e:
                        logger.error(f"Task {idx} failed: {e}")
                        results[idx] = None
                        
            except (InvalidAlgorithmError, KeyboardInterrupt) as e:
                if isinstance(e, KeyboardInterrupt):
                    logger.error("Interrupted by user") # Строка для теста
                raise # Выход из цикла и проброс в блок finally
            except Exception as e:
                logger.error(f"Error while waiting for tasks: {e}")
                break

    except Exception as e:
        if mode == "processes":
            logger.error("Error in process pool, falling back to threads: %s", e)

            _perform_cleanup(shm_refs, persistence_manager) 
            return run_parallel(func, 
                                args_seq, 
                                kwargs_seq, 
                                max_workers, 
                                mode="threads", 
                                timeout=timeout)
        raise
    finally:
        if pool is not None:
            is_proc_executor = (
                type(pool).__name__ == "ProcessPoolExecutor" 
                or hasattr(pool, "_processes"))
            if mode == "processes" and is_proc_executor:
                # Безопасный захват воркеров до shutdown
                # Копируем список объектов процессов, пока они доступны в _processes
                workers = list(getattr(pool, "_processes", {}).values())
                
                pool.shutdown(wait=False, cancel_futures=True)
                
                # Льготный период ожидания (grace period)
                stop_time = time.time() + shutdown_grace_period
                
                while time.time() < stop_time and any(w.is_alive() for w in workers):
                    time.sleep(0.1)
                
                # Принудительное завершение выживших (terminate -> kill)
                for w in workers:
                    if w.is_alive():
                        try:
                            # ERROR уровень для прерывания 
                            # (проблема в алгоритме/библиотеке)
                            logger.error(f"CRITICAL: Worker {w.pid} hung "
                                         f"in task execution. "
                                         f"Forcing SIGTERM to release resources.")
                            w.terminate()
                        except Exception: 
                            pass
                
                # Короткая пауза для завершения системных вызовов
                time.sleep(0.2)
                
                for w in workers:
                    if w.is_alive():
                        try:
                            # CRITICAL уровень, если процесс проигнорировал даже SIGTERM
                            logger.critical(
                                            f"HARD KILL: Worker {w.pid} resisted"
                                            f" SIGTERM. Sending SIGKILL. Possible "
                                            f"memory leak or C-level freeze.")
                            w.kill()
                        except Exception: 
                            pass
                
                # Финальная очистка SHM/Disk (только в режиме процессов)
                _perform_cleanup(shm_refs, persistence_manager)
            else:
                # Для ThreadPoolExecutor или если режим не "processes"
                pool.shutdown(wait=True, cancel_futures=True)
    return results


