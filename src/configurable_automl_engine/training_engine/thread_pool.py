from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Mapping, Sequence


def run_parallel(
    func: Callable[..., Any],
    args_seq: Iterable[Sequence[Any]] | None = None,
    kwargs_seq: Iterable[Mapping[str, Any]] | None = None,
    max_workers: int | None = None,
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

    results: list[Any] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(func, *a, **kw) for a, kw in zip(args_seq, kwargs_seq, strict=True)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())  # propagate exceptions
    return results
