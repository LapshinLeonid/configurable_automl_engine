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
from concurrent.futures import (ThreadPoolExecutor,
                                ProcessPoolExecutor, 
                                as_completed, 
                                TimeoutError, 
                                Executor
                                )
from typing import Any, Callable, Iterable, Mapping, Sequence
from multiprocessing import shared_memory

import pandas as pd
import numpy as np
import os
import tempfile

from configurable_automl_engine.tuner import InvalidAlgorithmError

import logging
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
    def is_compatible(df: pd.DataFrame) -> bool:
        """Проверяет, можно ли разместить DF в SHM 
        (только простые типы и RangeIndex)."""
        # 1. Проверка типов данных (белый список: int, uint, float, bool)
        allowed_kinds = {'i', 'u', 'f', 'b'}
        if not all(dt.kind in allowed_kinds for dt in df.dtypes):
            return False
            
        # 2. Проверка индекса: SHM в текущей реализации не поддерживает сложные индексы
        # Если индекс не стандартный (0, 1, 2...), лучше отправить через Диск (Parquet)
        if not isinstance(df.index, pd.RangeIndex):
            return False
            
        return True

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
        shm_indices: list[int] | None
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
        shm_indices (list[int]): Индексы аргументов, содержащих объекты 
            SharedDataFrame или их прокси.
    Returns:
        Any: Результат выполнения функции func.
    """

    final_args = list(args)
    # Используем контекстный менеджер для автоматического закрытия дескрипторов SHM
    # Это освобождает системные ресурсы сразу после вызова функции.
    shm_wrappers: list[SharedDataFrame] = []
    try:
        # 1. Восстановление из Shared Memory
        if shm_indices:
            for idx in shm_indices:
                wrapper = final_args[idx]
                # Используем duck typing для совместимости 
                # (и с Mock, и с реальным классом)
                if hasattr(wrapper, 'to_df'):
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
    """Вспомогательная функция для безопасной очистки"""
    if shm_refs:
        for ref in shm_refs:
            try:
                # На Windows порядок close/unlink критичен
                ref.close() 
                ref.unlink()
            except Exception:
                pass
    if persistence_manager:
        try:
            persistence_manager.cleanup()
        except Exception:
            pass

def run_parallel(
    func: Callable[..., Any],
    args_seq: Iterable[Sequence[Any]] | None = None,
    kwargs_seq: Iterable[Mapping[str, Any]] | None = None,
    max_workers: int | None = None, 
    mode: str = "threads",
    timeout: int | float | None = 3600,
    shared_args_indices: list[int] | None = None,
    disk_args_indices: list[int] | None = None,
) -> list[Any]:
    """Организовать параллельное выполнение функции с управлением памятью.
    Логика параллелизма:
    1. Режим: Поддерживает многопоточность (threads) и многопроцессорность (processes).
    2. Оптимизация: При работе с процессами переносит DataFrame в Shared Memory 
       или на диск для исключения накладных расходов на Pickle-сериализацию.
    3. Исполнение: Использует Executor для запуска задач и собирает результаты с 
       контролем таймаута.
    4. Отказоустойчивость: При сбое инициализации или выполнении пула процессов 
        выполняется попытка перезапуска всей последовательности задач 
        в режиме "threads" с предварительной очисткой ресурсов.
    5. Ресурсный менеджмент: Гарантирует очистку сегментов памяти и временных 
       файлов после завершения всех задач.
    Args:
        func (Callable): Функция для запуска.
        args_seq (Iterable): Последовательность кортежей аргументов.
        kwargs_seq (Iterable): Последовательность словарей именованных аргументов.
        max_workers (int): Лимит количества рабочих воркеров.
        mode (str): Режим параллелизма ("threads" или "processes").
        timeout (int | float): Максимальное время ожидания для каждой задачи.
        shared_args_indices (list[int]): Индексы DataFrame для Shared Memory.
        disk_args_indices (list[int]): Индексы DataFrame для дискового кэша.
    Returns:
        list[Any]: Список результатов выполнения задач (None в случае ошибки/таймаута).
    """

    args_seq = list(args_seq or [()])
    kwargs_seq = list(kwargs_seq or [{}] * len(args_seq))


    if len(args_seq) != len(kwargs_seq):
        raise ValueError("args_seq and kwargs_seq must be of equal length")

    
    results: list[Any] = []

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
            for idx in set(target_shm_indices) | set(target_disk_indices):
                if idx < len(new_args) and isinstance(new_args[idx], pd.DataFrame):
                    # Проверка совместимости: если SHM нельзя, 
                    # принудительно на диск
                    if (target_shm_indices is not None and idx in target_shm_indices 
                        and SharedDataFrame.is_compatible(new_args[idx])):
                        shm_wrapper = SharedDataFrame(new_args[idx])
                        shm_refs.append(shm_wrapper)
                        new_args[idx] = shm_wrapper
                    else:
                        path = persistence_manager.save_df(new_args[idx])
                        new_args[idx] = path
            
            # Определяем индексы именно для этой задачи
            curr_shm = [i for i in target_shm_indices if i < len(new_args) 
                        and isinstance(new_args[i], SharedDataFrame)]
            curr_disk = [i for i in (set(target_shm_indices) | set(target_disk_indices))
                         if i < len(new_args) and isinstance(new_args[i], str) 
                         and new_args[i].endswith(".parquet")]
            
            task_payloads.append((tuple(new_args), kwargs, curr_disk, curr_shm))
        
        # Переопределяем итерируемый объект для запуска
        execution_tasks = task_payloads 
    else:
        # Для потоков или обычных процессов без SHM/Disk
        execution_tasks = [
                        (tuple(a), kw, disk_args_indices 
                         or [], shared_args_indices or []) 
                        for a, kw in zip(args_seq, kwargs_seq)
                        ]

    # 1. Определяем класс исполнителя
    executor_cls: Callable[[int | None], Executor] = ThreadPoolExecutor
    if mode == "processes":
        try:
            executor_cls = ProcessPoolExecutor
        except Exception as e:
            logger.error(
                f"Could not initialize ProcessPoolExecutor: {e}. "
                "Falling back to threads.")
            executor_cls = ThreadPoolExecutor
    


    # Флаг для отслеживания успешного завершения пула
    # 2. Используем выбранный класс (универсальный интерфейс)
    try:  
        with executor_cls(max_workers) as pool:
            futures = []
            for a, kw, d_idx, s_idx in execution_tasks:
                if mode == "processes" and (shared_args_indices or disk_args_indices):
                    # Вызываем прокси-функцию для десериализации данных 
                    # в контексте воркера
                    futures.append(pool.submit(_worker_proxy, 
                                               func, 
                                               a, 
                                               kw, 
                                               d_idx, 
                                               s_idx))
                else:
                    futures.append(pool.submit(func, *a, **kw))
            for fut in as_completed(futures):
                try:
                    # Здесь ловятся исключения из самих задач
                    results.append(fut.result(timeout=timeout))
                except TimeoutError:
                    logger.error(
                        "Task timed out after %s s, marking as failed", 
                        timeout
                    )
                    fut.cancel()
                    results.append(None)
                except KeyboardInterrupt:
                    logger.error("Interrupted by user (KeyboardInterrupt)")
                    # Пробрасываем прерывания немедленно
                    # Прерывание вызовет выход из блока 'with', что автоматически 
                    # запустит shutdown(wait=True) для очистки активных воркеров
                    raise 
                except InvalidAlgorithmError:
                    raise
                except Exception as e:
                    logger.error(
                        "Task failed with an unexpected error: %s", 
                        e, 
                        exc_info=True
                    )
                    results.append(None)
        # Если мы дошли сюда, пул закрылся штатно (wait=True отработал)

    except Exception as e:
        if mode == "processes":
            logger.error("Falling back to threads due to: %s", e)
            _perform_cleanup(shm_refs, persistence_manager) 
            return run_parallel(func, 
                                args_seq, 
                                kwargs_seq, 
                                max_workers, 
                                mode="threads", 
                                timeout=timeout
                                )
        raise
    finally:
        if mode == "processes":
            # Выполняем окончательное удаление сегментов SHM и временных файлов через 
            # вспомогательную функцию _perform_cleanup
            _perform_cleanup(shm_refs, persistence_manager)
    return results


