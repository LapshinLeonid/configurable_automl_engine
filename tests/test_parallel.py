import pytest
import time
import hashlib
from unittest.mock import patch, MagicMock
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError

from configurable_automl_engine.training_engine.thread_pool import run_parallel

# --- Тестовые функции ---
def cpu_bound_task(n: int) -> str:
    """Имитация задачи, нагружающей CPU (вычисление хеша)."""
    data = str(n).encode()
    for _ in range(100_000):
        data = hashlib.sha256(data).hexdigest().encode()
    return data.decode()
def simple_task(x: int, y: int = 0) -> int:
    """Простая задача для проверки аргументов."""
    return x + y
def slow_task(seconds: float):
    """Задача для проверки таймаута."""
    time.sleep(seconds)
    return "done"
def failing_task():
    """Задача для проверки обработки исключений."""
    raise ValueError("Intentional failure")
# --- Тесты ---
def test_run_parallel_threads_basic():
    """1. Проверка работы в режиме потоков (Threads)."""
    args = [(1, 2), (3, 4), (5, 6)]
    results = run_parallel(simple_task, args_seq=args, mode="threads")
    
    assert sorted(results) == [3, 7, 11]
def test_run_parallel_processes_cpu_bound():
    """2. Проверка работы в режиме процессов (Processes)."""
    # Используем небольшое количество задач для теста
    args = [(i,) for i in range(3)]
    results = run_parallel(cpu_bound_task, args_seq=args, mode="processes")
    
    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)
def test_run_parallel_fallback_mechanism():
    """3. Проверка механизма Fallback при сбое инициализации процессов."""
    args = [(1, 10), (2, 20)]
    
    # Имитируем ошибку при вызове ProcessPoolExecutor
    with patch("configurable_automl_engine.training_engine.thread_pool.ProcessPoolExecutor", side_effect=RuntimeError("OS Error")):
        with patch("configurable_automl_engine.training_engine.thread_pool.logger") as mock_logger:
            results = run_parallel(simple_task, args_seq=args, mode="processes")
            
            # Проверяем, что результаты все равно получены (через fallback)
            assert sorted(results) == [11, 22]
            # Проверяем, что ошибка была залогирована
            mock_logger.error.assert_called()
            assert "Falling back to threads" in mock_logger.error.call_args[0][0]

def test_run_parallel_error_propagation():
    """5. Проверка проброса ошибок и записи в лог."""
    args = [()] # Вызываем один раз без аргументов
    
    with patch("configurable_automl_engine.training_engine.thread_pool.logger") as mock_logger:
        results = run_parallel(failing_task, args_seq=args, mode="threads")
        
        assert results == [None]
        # Проверяем, что ошибка залогирована
        mock_logger.error.assert_called()
        assert "Task failed" in mock_logger.error.call_args[0][0]
def test_run_parallel_empty_args():
    """Дополнительно: проверка запуска без аргументов (ровно один раз)."""
    def get_one(): return 1
    results = run_parallel(get_one)
    assert results == [1]
def test_run_parallel_validation_error():
    """Дополнительно: проверка несовпадения длины аргументов."""
    with pytest.raises(ValueError, match="must be of equal length"):
        run_parallel(
            simple_task, 
            args_seq=[(1,)], 
            kwargs_seq=[{}, {}] # Разная длина
        )