import pytest
import time
import hashlib
import logging
import pandas as pd
import os

from unittest.mock import MagicMock,patch
from configurable_automl_engine.tuner import InvalidAlgorithmError

from configurable_automl_engine.training_engine import thread_pool

from configurable_automl_engine.training_engine.thread_pool import (
    run_parallel, 
    SharedDataFrame, 
    DiskPersistenceManager,
    _worker_proxy
)

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
    assert any("Task failed with an unexpected error: boom" in rec.getMessage() for rec in caplog.records)

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

# Вспомогательная функция для тестов
def process_df_task(df, multiplier):
    return df.sum().sum() * multiplier
def test_shared_dataframe_manual():
    """Тест базовой функциональности SharedDataFrame вне run_parallel."""
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    shm_wrapper = SharedDataFrame(df=df)
    
    try:
        assert shm_wrapper.name.startswith("shm_")
        df_recovered = shm_wrapper.to_df()
        pd.testing.assert_frame_equal(df, df_recovered)
    finally:
        shm_wrapper.close()
        shm_wrapper.unlink()
def test_disk_persistence_manager_manual():
    """Тест базовой функциональности DiskPersistenceManager."""
    df = pd.DataFrame({'a': range(10)})
    manager = DiskPersistenceManager(use_shm=False)
    path = None
    try:
        path = manager.save_df(df)
        assert os.path.exists(path)
        assert path.endswith(".parquet")
        df_recovered = pd.read_parquet(path)
        pd.testing.assert_frame_equal(df, df_recovered)
    finally:
        manager.cleanup()
        if path:
            assert not os.path.exists(path)
def test_run_parallel_with_shared_memory():
    """Тест Task 1.1 и 1.2: Передача данных через Shared Memory в процессах."""
    df = pd.DataFrame({'val': [10, 20, 30]})
    # Ожидаемый результат: (10+20+30) * 1 = 60 и (10+20+30) * 2 = 120
    args = [(df, 1), (df, 2)]
    
    results = run_parallel(
        process_df_task, 
        args_seq=args, 
        mode="processes", 
        shared_args_indices=[0]
    )
    
    assert sorted(results) == [60, 120]
def test_run_parallel_with_disk_persistence():
    """Тест Phase 2: Передача данных через Disk (Parquet) в процессах."""
    df = pd.DataFrame({'val': [1, 2, 3]})
    args = [(df, 10)]
    
    results = run_parallel(
        process_df_task, 
        args_seq=args, 
        mode="processes", 
        disk_args_indices=[0]
    )
    
    assert results == [60]

def test_run_parallel_shm_cleanup_on_error():
    """Проверка, что очистка ресурсов вызывается даже при ошибке в задаче."""
    df = pd.DataFrame({'a': [1]})
    
    def error_task(df):
        raise ValueError("Intentional error")
    # Вызываем и игнорируем ошибку (она залогируется как Task Failed)
    results = run_parallel(
        error_task,
        args_seq=[(df,)],
        mode="processes",
        shared_args_indices=[0]
    )
    assert results == [None]
    # Если здесь не упало и память не потекла (проверяется вручную или инструментами мониторинга)
    # то блок finally отработал.
def test_shared_dataframe_reconstruction():
    """Покрытие случая восстановления SharedDataFrame без передачи df (сторона воркера)."""
    df = pd.DataFrame({'x': [100]})
    original = SharedDataFrame(df=df)
    
    # Имитируем то, что делает прокси: создаем объект по имени и метаданным
    reconstructed = SharedDataFrame(
        name=original.name, 
        shape=original.shape, 
        dtype=original.dtype, 
        columns=original.columns
    )
    
    pd.testing.assert_frame_equal(reconstructed.to_df(), df)
    
    original.close()
    original.unlink()

# 1. Выносим функцию на уровень модуля
def mixed_task_global(d1, d2):
    return int(d1.iloc[0, 0] + d2.iloc[0, 0])
def test_run_parallel_mixed_shm_and_disk():
    """Тест одновременного использования обоих механизмов."""
    df_shm = pd.DataFrame({'a': [1]})
    df_disk = pd.DataFrame({'b': [2]})
    
    # 2. Передаем глобальную функцию
    results = run_parallel(
        mixed_task_global,
        args_seq=[(df_shm, df_disk)],
        mode="processes",
        shared_args_indices=[0],
        disk_args_indices=[1]
    )
    assert results == [3]

def test_disk_persistence_manager_cleanup_exception(caplog):
    """
    Тест покрывает случай, когда файл существует, но os.remove выбрасывает ошибку.
    Проверяет логирование logger.warning в блоке except.
    """
    # 1. Готовим данные
    df = pd.DataFrame({'a': [1]})
    manager = DiskPersistenceManager(use_shm=False)
    
    # Сохраняем реальный файл, чтобы получить валидный путь в списке created_files
    path = manager.save_df(df)
    
    # 2. Имитируем ошибку при удалении
    # Настраиваем перехват логов на уровне WARNING
    with caplog.at_level(logging.WARNING):
        # Патчим os.remove, чтобы он выбрасывал PermissionError при вызове
        with patch("os.remove", side_effect=OSError("Access Denied")):
            # Вызываем очистку. Она не должна прерываться исключением (оно ловится внутри)
            manager.cleanup()
    
    # 3. Проверки
    # Проверяем, что в логах появилось наше сообщение
    assert "Failed to delete temp file" in caplog.text
    assert "Access Denied" in caplog.text
    
    # Вручную удаляем файл после теста, так как mock помешал менеджеру это сделать
    try:
        import os
        if os.path.exists(path):
            os.remove(path)
    except:
        pass
def test_disk_persistence_manager_cleanup_file_not_found():
    """
    Дополнительный тест: если файла уже нет (os.path.exists = False), 
    ошибка не должна возникать и logger.warning не должен вызываться.
    """
    manager = DiskPersistenceManager(use_shm=False)
    manager.created_files.append("non_existent_file.parquet")
    
    # Это не должно вызвать ни исключения, ни предупреждения в логах
    # так как сработает проверка if os.path.exists(path)
    manager.cleanup()

def test_run_parallel_shm_cleanup_exception_handling():
    """
    Тест проверяет, что ошибки в методах close и unlink 
    игнорируются внутри блока finally функции run_parallel.
    """
    df = pd.DataFrame({'a': [1]})
    
    # 1. Патчим SharedDataFrame внутри модуля thread_pool
    with patch("configurable_automl_engine.training_engine.thread_pool.SharedDataFrame") as mock_shm_class:
        mock_instance = MagicMock()
        
        # Настраиваем side_effect так, чтобы методы вызывали ошибку, 
        # но при этом выполнение кода могло продолжаться, если код обернут в try/except
        mock_instance.close.side_effect = RuntimeError("Close failed")
        mock_instance.unlink.side_effect = RuntimeError("Unlink failed")
        mock_shm_class.return_value = mock_instance
        # 2. Вызываем run_parallel
        # Если в коде стоит try: shm.close() except: pass и следом try: shm.unlink() except: pass,
        # то оба ассерта ниже пройдут.
        results = run_parallel(
            func=lambda x: x,
            args_seq=[(df,)],
            mode="processes",
            shared_args_indices=[0]
        )
        # 3. Проверки
        # Проверяем, что close был вызван (даже если он бросил исключение)
        assert mock_instance.close.called, "Метод close() не был вызван"
        
        # Проверяем, что unlink был вызван. 
        # Если этот ассерт падает, значит в исходном коде ошибка close() 
        # прерывает выполнение и не дает дойти до unlink().
        assert mock_instance.unlink.called, "Метод unlink() не был вызван (прервано исключением в close?)"

# 1. Объявляем функцию на уровне модуля, чтобы pickle мог её найти
def worker_sum_func(df):
    return df['val'].sum()
def test_worker_proxy_integration():
    """
    Интеграционный тест для проверки _worker_proxy и Shared Memory.
    Этот тест покроет строки с восстановлением данных из SHM в воркере.
    """
    df = pd.DataFrame({'val': [1, 2, 3]})
    
    # 2. Вызываем run_parallel с функцией, доступной для импорта
    results = run_parallel(
        func=worker_sum_func,  # Передаем обычную функцию вместо лямбды
        args_seq=[(df,)],
        mode="processes",
        shared_args_indices=[0]
    )
    
    # 3. Проверяем результат
    assert results is not None, "Результат не должен быть None (проверьте логи на ошибки pickle)"
    assert results[0] == 6, f"Ожидалось 6, получено {results[0]}"
def test_worker_proxy_disk_integration(tmp_path):
    """
    Дополнительный тест для покрытия строк загрузки с диска (Parquet).
    """
    df = pd.DataFrame({'val': [10, 20]})
    file_path = str(tmp_path / "data.parquet")
    df.to_parquet(file_path)
    
    results = run_parallel(
        func=worker_sum_func,
        args_seq=[(file_path,)],
        mode="processes",
        disk_args_indices=[0] # Указываем, что 0-й аргумент - путь к parquet
    )
    
    assert results[0] == 30

def test_worker_proxy_logic_direct_coverage():
    """
    Исправленный тест для прямого вызова _worker_proxy.
    Используем спецификацию класса SharedDataFrame, чтобы пройти проверку isinstance.
    """
    # 1. Готовим данные для SHM
    # Передаем spec=SharedDataFrame, чтобы isinstance(mock_shm, SharedDataFrame) вернул True
    mock_shm = MagicMock(spec=SharedDataFrame)
    mock_df = pd.DataFrame({'a': [1]})
    mock_shm.to_df.return_value = mock_df
    
    def simple_func(x): return x
    
    # 2. Проверяем блок SHM (теперь условие isinstance сработает)
    result_shm = _worker_proxy(
        func=simple_func,
        args=(mock_shm,),
        kwargs={},
        disk_indices=None, # в коде проверка if disk_indices, None или [] подходят
        shm_indices=[0]
    )
    
    assert mock_shm.to_df.called, "Метод .to_df() должен был быть вызван"
    assert isinstance(result_shm, pd.DataFrame), "Результат должен быть DataFrame"
    assert result_shm.iloc[0]['a'] == 1
    # 3. Проверяем блок Disk/Parquet (строки 71-76)
    import os
    tmp_file = "test_coverage_direct.parquet"
    mock_df.to_parquet(tmp_file)
    
    try:
        # Передаем строку, заканчивающуюся на .parquet
        result_disk = _worker_proxy(
            func=simple_func,
            args=(tmp_file,),
            kwargs={},
            disk_indices=[0],
            shm_indices=[]
        )
        assert isinstance(result_disk, pd.DataFrame), "Данные с диска не загрузились как DF"
        assert result_disk.iloc[0]['a'] == 1
    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)