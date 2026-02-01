from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed, TimeoutError
from typing import Any, Callable, Iterable, Mapping, Sequence

import logging
logger = logging.getLogger(__name__)

from configurable_automl_engine.tuner import InvalidAlgorithmError

TIMEOUT_SECONDS = 3600

def run_parallel(
    func: Callable[..., Any],
    args_seq: Iterable[Sequence[Any]] | None = None,
    kwargs_seq: Iterable[Mapping[str, Any]] | None = None,
    max_workers: int | None = None, 
    mode: str = "threads",
    timeout=None
):
    """
    Запускает `func` параллельно для набора аргументов.

    *args_seq* — iterable из кортежей positional-аргументов  
    *kwargs_seq* — iterable из словарей keyword-аргументов

    Если оба не заданы, запустит `func()` ровно один раз.
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
            futures = [
                pool.submit(func, *a, **kw) for a, kw in zip(args_seq, kwargs_seq, strict=True)
            ]
            for fut in as_completed(futures):
                try:
                    results.append(fut.result(timeout=effective_timeout))  # Здесь ловятся исключения из самих задач
                except TimeoutError:
                    logger.error("Task timed out after %s s, marking as failed", effective_timeout)
                    fut.cancel()
                    results.append(None)
                except InvalidAlgorithmError:
                    raise  # Пробрасываем ошибку конфига выше, чтобы тест её поймал
                except Exception as e:
                    logger.error("Task failed: %s", e)
                    results.append(None)
        return results
    except Exception as e:
        if mode == "processes":
            logger.error("Falling back to threads due to: %s", e)
            return run_parallel(func, args_seq, kwargs_seq, max_workers, mode="threads", timeout=effective_timeout)
        raise
