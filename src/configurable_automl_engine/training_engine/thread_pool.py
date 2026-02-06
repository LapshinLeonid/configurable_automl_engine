from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed, TimeoutError
from typing import Any, Callable, Iterable, Mapping, Sequence
from multiprocessing import shared_memory

import pandas as pd
import numpy as np
import os
import tempfile

import logging
logger = logging.getLogger(__name__)

from configurable_automl_engine.tuner import InvalidAlgorithmError

TIMEOUT_SECONDS = 3600

class SharedDataFrame:
    """Обертка для размещения DataFrame в разделяемой памяти (Shared Memory)."""
    def __init__(self, df: pd.DataFrame = None, name: str = None, shape=None, dtype=None, columns=None):
        if df is not None:
            self.name = f"shm_{id(df)}_{np.random.randint(1000)}"
            data = df.to_numpy()
            self.shm = shared_memory.SharedMemory(create=True, size=data.nbytes, name=self.name)
            self.shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=self.shm.buf)
            self.shared_array[:] = data[:]
            self.shape = data.shape
            self.dtype = data.dtype
            self.columns = df.columns.tolist()
        else:
            self.name = name
            self.shm = shared_memory.SharedMemory(name=name)
            self.shared_array = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf)
            self.columns = columns
    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.shared_array, columns=self.columns)
    def close(self):
        self.shm.close()
    def unlink(self):
        self.shm.unlink()

class DiskPersistenceManager:
    """Утилита для временного сохранения DataFrame на диск (Parquet)."""
    def __init__(self, use_shm: bool = True):
        # Используем /dev/shm для Linux если доступно, иначе стандартный temp
        self.tmp_dir = "/dev/shm" if use_shm and os.path.exists("/dev/shm") else None
        self.created_files = []
    def save_df(self, df: pd.DataFrame) -> str:
        fd, path = tempfile.mkstemp(suffix=".parquet", dir=self.tmp_dir)
        os.close(fd)
        df.to_parquet(path, engine="fastparquet" if "fastparquet" in globals() else "pyarrow")
        self.created_files.append(path)
        return path
    def cleanup(self):
        for path in self.created_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {path}: {e}")

def _worker_proxy(func, args, kwargs, disk_indices, shm_indices):
    """Функция-прокси для десериализации данных на стороне воркера."""
    final_args = list(args)
    
    # 1. Восстановление из Shared Memory
    if shm_indices:
        for idx in shm_indices:
            if isinstance(final_args[idx], SharedDataFrame):
                final_args[idx] = final_args[idx].to_df()
    # 2. Загрузка с диска (Parquet)
    if disk_indices:
        for idx in disk_indices:
            path = final_args[idx]
            if isinstance(path, str) and path.endswith(".parquet"):
                final_args[idx] = pd.read_parquet(path)
                
    return func(*final_args, **kwargs)

def run_parallel(
    func: Callable[..., Any],
    args_seq: Iterable[Sequence[Any]] | None = None,
    kwargs_seq: Iterable[Mapping[str, Any]] | None = None,
    max_workers: int | None = None, 
    mode: str = "threads",
    timeout=None,
    shared_args_indices: list[int] | None = None,
    disk_args_indices: list[int] | None = None
):
    """
    Запускает `func` параллельно для набора аргументов.

    *args_seq* — iterable из кортежей positional-аргументов  
    *kwargs_seq* — iterable из словарей keyword-аргументов

    Если оба не заданы, запустит `func()` ровно один раз.

    shared_args_indices: индексы в args_seq[i], которые содержат DataFrame 
    для перевода в Shared Memory (только для mode="processes").
    """
    args_seq = list(args_seq or [()])
    kwargs_seq = list(kwargs_seq or [{}] * len(args_seq))


    if len(args_seq) != len(kwargs_seq):
        raise ValueError("args_seq and kwargs_seq must be of equal length")

    if timeout is None:
        import configurable_automl_engine.training_engine.thread_pool as tp
        effective_timeout = tp.TIMEOUT_SECONDS
    else:
        effective_timeout = timeout
    results: list[Any] = []

    # Логика подготовки Shared Memory для процессов
    shm_refs = []
    persistence_manager = DiskPersistenceManager()
    if mode == "processes" and (shared_args_indices or disk_args_indices):
        new_args_seq = []
        for args in args_seq:
            new_args = list(args)
            # Обработка Shared Memory
            if shared_args_indices:
                for idx in shared_args_indices:
                    if isinstance(new_args[idx], pd.DataFrame):
                        shm_wrapper = SharedDataFrame(new_args[idx])
                        shm_refs.append(shm_wrapper)
                        new_args[idx] = shm_wrapper
            
            # Обработка дисковой персистентности
            if disk_args_indices:
                for idx in disk_args_indices:
                    if isinstance(new_args[idx], pd.DataFrame):
                        path = persistence_manager.save_df(new_args[idx])
                        new_args[idx] = path
            
            new_args_seq.append(tuple(new_args))
        args_seq = new_args_seq

    # 1. Определяем класс исполнителя
    executor_cls = ThreadPoolExecutor
    if mode == "processes":
        try:
            executor_cls = ProcessPoolExecutor
        except Exception as e:
            logger.error(f"Could not initialize ProcessPoolExecutor: {e}. Falling back to threads.")
            executor_cls = ThreadPoolExecutor
    
    # 2. Используем выбранный класс (универсальный интерфейс)
    try:  
        with executor_cls(max_workers=max_workers) as pool:
            futures = []
            for a, kw in zip(args_seq, kwargs_seq, strict=True):
                if mode == "processes" and (shared_args_indices or disk_args_indices):
                    # Используем прокси для восстановления данных
                    futures.append(pool.submit(_worker_proxy, func, a, kw, disk_args_indices, shared_args_indices))
                else:
                    futures.append(pool.submit(func, *a, **kw))
            for fut in as_completed(futures):
                try:
                    results.append(fut.result(timeout=effective_timeout))  # Здесь ловятся исключения из самих задач
                except TimeoutError:
                    logger.error("Task timed out after %s s, marking as failed", effective_timeout)
                    fut.cancel()
                    results.append(None)
                except InvalidAlgorithmError:
                    raise  # Пробрасываем ошибку конфига выше, чтобы тест её поймал
                except KeyboardInterrupt:
                    logger.error("Interrupted by user (KeyboardInterrupt)")
                    raise
                except Exception as e:
                    logger.error("Task failed: %s", e)
                    results.append(None)
        return results
    except Exception as e:
        if mode == "processes":
            logger.error("Falling back to threads due to: %s", e)
            return run_parallel(func, args_seq, kwargs_seq, max_workers, mode="threads", timeout=effective_timeout)
        raise
    finally:
        if mode == "processes" and 'shm_refs' in locals():
            for ref in shm_refs:
                try:
                    ref.unlink()
                except Exception:
                    pass
                try:
                    ref.close()
                except Exception:
                    pass
            if 'persistence_manager' in locals():
                persistence_manager.cleanup()
