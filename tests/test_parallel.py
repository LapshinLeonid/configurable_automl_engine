import pytest
import time
import hashlib
import logging

from unittest.mock import PropertyMock, MagicMock,patch
from configurable_automl_engine.tuner import InvalidAlgorithmError


from configurable_automl_engine.training_engine.thread_pool import run_parallel
from configurable_automl_engine.training_engine import thread_pool

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

def test_run_parallel_timeout_error_coverage(caplog):
    # Мокаем ThreadPoolExecutor, чтобы его футуры всегда кидали TimeoutError
    with patch("configurable_automl_engine.training_engine.thread_pool.ThreadPoolExecutor") as mock_executor:
        mock_pool = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_pool
        
        # Создаем фейковую футуру
        mock_future = MagicMock()
        # При вызове fut.result(timeout=...) кидаем TimeoutError из concurrent.futures
        from concurrent.futures import TimeoutError
        mock_future.result.side_effect = TimeoutError()
        
        mock_pool.submit.return_value = mock_future
        
        # as_completed должен вернуть список наших моков
        with patch("configurable_automl_engine.training_engine.thread_pool.as_completed", return_value=[mock_future]):
            with caplog.at_level(logging.ERROR):
                results = run_parallel(lambda: None, args_seq=[()])
    assert results == [None]
    assert any("Task timed out after" in rec.getMessage() for rec in caplog.records)

def raise_invalid():
    raise InvalidAlgorithmError("bad algo")

def test_run_parallel_invalid_algorithm_error_propagates():
    with pytest.raises(InvalidAlgorithmError):
        run_parallel(raise_invalid, args_seq=[()])

def raise_keyboard_interrupt():
    raise KeyboardInterrupt()
def test_run_parallel_keyboard_interrupt_propagates(caplog):
    with caplog.at_level(logging.ERROR):
        with pytest.raises(KeyboardInterrupt):
            run_parallel(raise_keyboard_interrupt, args_seq=[()])
    # Проверяем, что было залогировано сообщение
    assert any("Interrupted by user (KeyboardInterrupt)" in rec.getMessage() for rec in caplog.records)

def raise_value_error():
    raise ValueError("boom")
def test_run_parallel_generic_exception_logged_and_returns_none(caplog):
    with caplog.at_level(logging.ERROR):
        results = run_parallel(raise_value_error, args_seq=[()])
    assert results == [None]
    assert any("Task failed: boom" in rec.getMessage() for rec in caplog.records)

class FailingExecutor:
    def __init__(self, *args, **kwargs):
        # Важно: этот текст должен совпадать с тем, что мы ищем в логах
        raise RuntimeError("init failed")
    def __enter__(self): return self
    def __exit__(self, *args): pass
def test_run_parallel_fallback_from_processes_to_threads(monkeypatch, caplog):
    # Подменяем ProcessPoolExecutor
    monkeypatch.setattr(thread_pool, "ProcessPoolExecutor", FailingExecutor)
    
    def simple_func(x):
        return x + 10
    with caplog.at_level(logging.ERROR):
        results = thread_pool.run_parallel(
            simple_func, 
            args_seq=[(5,)], 
            mode="processes"
        )
    # 1. Результат должен быть успешным за счет рекурсивного вызова в threads
    assert results == [15]
    
    # 2. Проверяем лог. Текст ошибки берется из e (RuntimeError("init failed"))
    assert any("Falling back to threads due to: init failed" in rec.getMessage() 
               for rec in caplog.records)
    
def test_run_parallel_init_section_coverage(monkeypatch, caplog):
    """
    Тест для покрытия строк 43-45 (инициализация в начале функции).
    Мы удаляем ProcessPoolExecutor из модуля, чтобы попытка доступа к нему 
    вызвала ошибку в блоке try.
    """
    
    # 1. Удаляем атрибут из модуля thread_pool
    # Это заставит строку 'executor_cls = ProcessPoolExecutor' выбросить NameError или AttributeError
    monkeypatch.delattr(thread_pool, "ProcessPoolExecutor", raising=False)
    with caplog.at_level(logging.ERROR):
        # 2. Вызываем функцию. Теперь она споткнется прямо на входе в блок "processes"
        results = thread_pool.run_parallel(
            lambda x: x + 100,
            args_seq=[(1,)],
            mode="processes"
        )
    # ПРОВЕРКИ:
    # 1. Результат должен быть 101, так как сработал fallback на ThreadPool
    assert results == [101]
    
    # 2. Проверяем наличие лога именно из строки 44.
    # В этом блоке текст ошибки обычно: "Could not initialize ProcessPoolExecutor"
    log_messages = [rec.getMessage() for rec in caplog.records]
    
    # Ищем специфичное сообщение
    found = any("Could not initialize ProcessPoolExecutor" in msg for msg in log_messages)
    
    assert found, f"Ожидаемый лог 'Could not initialize...' не найден. В логах: {log_messages}"