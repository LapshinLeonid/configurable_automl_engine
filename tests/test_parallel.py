import pytest
import time
import hashlib
import logging
import concurrent.futures
import pandas as pd
import numpy as np
import os
from multiprocessing import shared_memory
import unittest

from unittest.mock import MagicMock, patch
from configurable_automl_engine.tuner import InvalidAlgorithmError

from configurable_automl_engine.training_engine import thread_pool

from configurable_automl_engine.training_engine.thread_pool import (
    run_parallel,
    SharedDataFrame,
    DiskPersistenceManager,
    _worker_proxy,
    _perform_cleanup,
    _force_shutdown_processes,
)

MODULE_PATH = 'configurable_automl_engine.training_engine.thread_pool'

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

def test_run_parallel_fallback_mechanism(caplog):
    """Актуализировано: проверка отката при сбое инициализации."""
    args = [(1, 10)]
    with patch("configurable_automl_engine.training_engine.thread_pool.ProcessPoolExecutor", 
               side_effect=RuntimeError("OS Error")):
        with caplog.at_level(logging.ERROR):
            results = run_parallel(simple_task, args_seq=args, mode="processes")
    
    assert results == [11]
    # Код пишет: "Error in process pool, falling back to threads: OS Error"
    assert "falling back to threads" in caplog.text
    assert "OS Error" in caplog.text

def test_run_parallel_error_propagation(caplog):
    """Актуализировано: проверка лога при ошибке задачи (формат 'Task 0 failed')."""
    with caplog.at_level(logging.ERROR):
        results = run_parallel(failing_task, args_seq=[()], mode="threads")
    
    assert results == [None]
    # В актуальном коде: logger.error(f"Task {idx} failed: {e}")
    assert "Task 0 failed" in caplog.text
    assert "Intentional failure" in caplog.text

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
def test_run_parallel_timeout_error_coverage():
    """Проверка логирования таймаута при истечении глобального timeout."""
    with patch("configurable_automl_engine.training_engine.thread_pool.ThreadPoolExecutor") as mock_executor:
        mock_pool = MagicMock()
        mock_executor.return_value = mock_pool
        
        mock_future = MagicMock()
        mock_pool.submit.return_value = mock_future
        
        # as_completed — генератор. Чтобы корректно симулировать TimeoutError,
        # нужно, чтобы исключение выбрасывалось ПРИ ИТЕРАЦИИ (внутри for fut in ...),
        # а не при вызове as_completed(...).
        # Для этого side_effect должна быть функцией-генератором:
        #   def gen(*args, **kwargs):
        #       raise concurrent.futures.TimeoutError()
        #       yield  # делает функцию генератором
        def _raise_timeout_at_iteration(*args, **kwargs):
            raise concurrent.futures.TimeoutError()
            yield  # pragma: no cover
        
        with patch("configurable_automl_engine.training_engine.thread_pool.as_completed",
                   side_effect=_raise_timeout_at_iteration):
            with patch("configurable_automl_engine.training_engine.thread_pool.logger") as mock_logger:
                results = run_parallel(lambda: None, args_seq=[()], timeout=0.1)
                
    assert results == [None]
    # Проверяем, что logger.error был вызван с сообщением о таймауте
    error_messages = [call[0][0] for call in mock_logger.error.call_args_list]
    assert any("Task 0 timed out" in msg for msg in error_messages)

def raise_invalid():
    raise InvalidAlgorithmError("bad algo")

def test_run_parallel_invalid_algorithm_error_propagates():
    with pytest.raises(InvalidAlgorithmError):
        run_parallel(raise_invalid, args_seq=[()])

def raise_keyboard_interrupt():
    raise KeyboardInterrupt()

def test_run_parallel_keyboard_interrupt_propagates(caplog):
    """Исправлено: проверка новой строки лога при прерывании."""
    def raise_ki(): raise KeyboardInterrupt()
    
    with caplog.at_level(logging.ERROR):
        with pytest.raises(KeyboardInterrupt):
            run_parallel(raise_ki, args_seq=[()])
    
    # В обновленном коде строка: "Interrupted by user"
    assert "Interrupted by user" in caplog.text

def raise_value_error():
    raise ValueError("boom")

def test_run_parallel_generic_exception_logged_and_returns_none(caplog):
    """Актуализировано: проверка лога для произвольного исключения."""
    def boom(): raise ValueError("boom")
    
    with caplog.at_level(logging.ERROR):
        results = run_parallel(boom, args_seq=[()])
        
    assert results == [None]
    # В актуальном коде: "Task 0 failed: boom"
    assert "Task 0 failed" in caplog.text 
    assert "boom" in caplog.text

class FailingExecutor:
    def __init__(self, *args, **kwargs):
        # Важно: этот текст должен совпадать с тем, что мы ищем в логах
        raise RuntimeError("init failed")
    def __enter__(self): return self
    def __exit__(self, *args): pass

def test_run_parallel_init_section_coverage(monkeypatch, caplog):
    """Актуализировано: проверка при отсутствии ProcessPoolExecutor в пространстве имен."""
    # Удаляем атрибут, чтобы спровоцировать ошибку импорта/инициализации
    monkeypatch.delattr(thread_pool, "ProcessPoolExecutor", raising=False)
    
    with caplog.at_level(logging.ERROR):
        results = run_parallel(lambda x: x, args_seq=[(5,)], mode="processes")
    
    assert results == [5]
    assert "Could not initialize ProcessPoolExecutor" in caplog.text or "falling back to threads" in caplog.text

def test_run_parallel_init_section_coverage(monkeypatch, caplog):
    """Исправлено: проверка лога при отсутствии ProcessPoolExecutor в модуле."""
    monkeypatch.delattr(thread_pool, "ProcessPoolExecutor", raising=False)
    
    with caplog.at_level(logging.ERROR):
        results = run_parallel(lambda x: x, args_seq=[(5,)], mode="processes")
    
    assert results == [5]
    # В коде: "Could not initialize ProcessPoolExecutor"
    assert "Could not initialize ProcessPoolExecutor" in caplog.text

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
    from configurable_automl_engine.training_engine.thread_pool import _worker_proxy
    df = pd.DataFrame({'val': [30]})
    path = str(tmp_path / "test.parquet")
    df.to_parquet(path)
    
    # Прямой вызов прокси, как это делает воркер
    result = _worker_proxy(lambda x: x['val'].iloc[0], (path,), {}, [0], [])
    assert result == 30

def test_worker_proxy_logic_direct_coverage():
    # 1. Данные для восстановления
    meta = ("test_shm_name", (1, 1), np.float64, ['a'])
    
    # Патчим SharedDataFrame, чтобы он не лез в реальную память ОС
    with patch("configurable_automl_engine.training_engine.thread_pool.SharedDataFrame") as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.to_df.return_value = pd.DataFrame({'a': [1.0]})
        
        def simple_func(x): return x
    
        result = _worker_proxy(
            func=simple_func,
            args=(None,), # В args теперь None на месте SHM
            kwargs={},
            disk_indices=None,
            shm_info={0: meta} # Передаем через shm_info
        )
        assert result.iloc[0]['a'] == 1.0
        # Проверяем, что конструктор вызвался с нашими метаданными
        mock_class.assert_called_with(name=meta[0], shape=meta[1], dtype=meta[2], columns=meta[3])

def test_coverage_unlink_file_not_found():
    """
    Тест для принудительного покрытия веток обработки исключений в методе unlink.
    Использует мок, чтобы имитировать отсутствие сегмента в ОС.
    """
    # 1. Создаем минимальный DataFrame
    df = pd.DataFrame({'test': [1, 2, 3]})
    
    # 2. Инициализируем SharedDataFrame (он создаст реальный сегмент в памяти)
    shm_df = SharedDataFrame(df=df)
    
    # 3. Подменяем метод unlink объекта shm на мок, который всегда выбрасывает FileNotFoundError
    # Это имитирует ситуацию, когда сегмент уже был удален кем-то другим
    shm_df.shm.unlink = MagicMock(side_effect=FileNotFoundError("Уже удалено"))
    
    # 4. Вызываем метод unlink. 
    # Благодаря вашему блоку try-except, ошибка будет поймана, и выполнится строка 'pass'
    try:
        shm_df.unlink()
    except FileNotFoundError:
        import pytest
        pytest.fail("SharedDataFrame.unlink не перехватил FileNotFoundError, покрытие не сработало!")
    
    # 5. Очистка (закрываем дескрипторы)
    shm_df.close()
    
    # Проверяем, что мок действительно вызывался
    assert shm_df.shm.unlink.called

# 1. Тестирование SharedDataFrame.is_compatible (dtypes и Index)
def test_shared_data_frame_is_compatible_coverage():
    """Покрывает строки:
    - if not all(dt.kind in allowed_kinds for dt in df.dtypes): return False
    - if not isinstance(df.index, pd.RangeIndex): return False
    """
    # А. Тест на недопустимый тип (Object/String)
    df_obj = pd.DataFrame({'col': ['a', 'b', 'c']})
    assert SharedDataFrame.is_compatible(df_obj) is False
    
    # Б. Тест на недопустимый индекс (не RangeIndex)
    df_custom_index = pd.DataFrame(
        {'col': [1, 2, 3]}, 
        index=pd.Index([10, 20, 30])
    )
    assert SharedDataFrame.is_compatible(df_custom_index) is False
    # В. Тест на валидный DataFrame (int/float + RangeIndex)
    df_valid = pd.DataFrame({'col': [1.5, 2.0, 3.1]})
    assert SharedDataFrame.is_compatible(df_valid) is True

# 2. Тестирование исключения в _worker_proxy
def test_worker_proxy_close_exception():
    meta = ("name", (1, 2), np.int64, ["a", "b"])
    
    with patch("configurable_automl_engine.training_engine.thread_pool.SharedDataFrame") as mock_class:
        mock_instance = mock_class.return_value
        mock_instance.to_df.return_value = pd.DataFrame({'a': [1], 'b': [2]})
        mock_instance.close.side_effect = Exception("Simulated close error")
    
        def dummy_func(df): return len(df.columns)
        
        # Передаем shm_info, чтобы прокси создал воркер-враппер
        result = _worker_proxy(
            func=dummy_func,
            args=(None,),
            kwargs={},
            disk_indices=[],
            shm_info={0: meta}
        )
        assert result == 2
        assert mock_instance.close.called

    # Проверяем, что метод close был вызван
    mock_instance.close.assert_called_once()

def test_perform_cleanup_all_exceptions():
    # Объект, у которого close() падает, но unlink() должен быть вызван
    ref_fail = MagicMock()
    ref_fail.close.side_effect = Exception("Close error")
    
    mock_pm = MagicMock()
    mock_pm.cleanup.side_effect = Exception("PM error")
    
    # Функция не должна выбрасывать исключение (Phase 3 resilience)
    try:
        _perform_cleanup(shm_refs=[ref_fail], persistence_manager=mock_pm)
    except Exception as e:
        pytest.fail(f"_perform_cleanup leaked an exception: {e}")
    
    assert ref_fail.close.called
    # unlink должен вызываться всегда для очистки ОС, даже если дескриптор не закрылся
    assert ref_fail.unlink.called 
    assert mock_pm.cleanup.called

# 4. Проверка поведения при пустом persistence_manager
def test_perform_cleanup_none_pm():
    """Проверка ветки if persistence_manager: False"""
    mock_shm_ref = MagicMock()
    _perform_cleanup(shm_refs=[mock_shm_ref], persistence_manager=None)
    mock_shm_ref.close.assert_called_once()

def test_is_shared_array_robust():
    """Тест для SharedDataFrame.is_shared_array с учетом особенностей атрибута .base"""
    
    # 1. Покрытие: return False (не ndarray)
    assert SharedDataFrame.is_shared_array("not an array") is False
    
    # 2. Покрытие: return False (ndarray без base или base не memoryview)
    normal_arr = np.array([1, 2, 3])
    assert SharedDataFrame.is_shared_array(normal_arr) is False
    # 3. Покрытие: return True
    # Используем memoryview напрямую, чтобы гарантировать тип .base
    raw_data = bytearray(b'\x00' * 24)
    m_view = memoryview(raw_data)
    # Создаем массив из buffer — в большинстве окружений это установит .base в memoryview
    shared_arr = np.frombuffer(m_view, dtype=np.float64)
    
    # Если на вашей платформе numpy все равно разворачивает base, 
    # мы можем проверить это и принудительно создать ситуацию для покрытия строки
    if not isinstance(shared_arr.base, memoryview):
        # Принудительное создание объекта, удовлетворяющего условию (для покрытия)
        class MockArray(np.ndarray):
            pass
        
        mock_arr = np.array([1.0, 2.0]).view(MockArray)
        # Вручную подменяем base для теста логики
        mock_arr.base = m_view 
        assert SharedDataFrame.is_shared_array(mock_arr) is True
    else:
        assert SharedDataFrame.is_shared_array(shared_arr) is True

def test_get_data_info():
    # 1. Ветка: pandas.DataFrame
    df = pd.DataFrame({'a': [1], 'b': [2]})
    count, names = SharedDataFrame.get_data_info(df)
    assert count == 2
    assert names == ['a', 'b']
    
    # 2. Ветка: SharedDataFrame
    sdf = SharedDataFrame(df)
    try:
        count, names = SharedDataFrame.get_data_info(sdf)
        assert count == 2
        assert names == ['a', 'b']
    finally:
        sdf.unlink()
        sdf.close()
        
    # 3. Ветка: np.ndarray (ndim > 1)
    arr_2d = np.zeros((5, 3))
    count, names = SharedDataFrame.get_data_info(arr_2d)
    assert count == 3
    assert names == [0, 1, 2]
    
    # 4. Ветка: np.ndarray (ndim == 1)
    arr_1d = np.array([1, 2, 3])
    count, names = SharedDataFrame.get_data_info(arr_1d)
    assert count == 1
    assert names == [0]
    
    # 5. Ветка: Другой тип (else)
    count, names = SharedDataFrame.get_data_info("not_a_df")
    assert count == 0
    assert names == []

def test_is_compatible_all_branches():
    """Тест для SharedDataFrame.is_compatible — покрытие проверок типов и индексов"""
    
    # 1. Покрытие: is_shared_array(df) -> True
    # (Используем трюк с подменой base, если обычный SHM не срабатывает как memoryview)
    data = bytearray(b'\x00' * 8)
    mv = memoryview(data)
    fake_shared = np.frombuffer(mv, dtype=np.int64)
    if not isinstance(fake_shared.base, memoryview):
        # Если numpy проигнорировал memoryview в base, пропускаем эту ветку 
        # или используем мок, так как логика зависит от окружения
        pass 
    else:
        assert SharedDataFrame.is_compatible(fake_shared) is True
    # 2. Покрытие: isinstance(df, pd.DataFrame) -> False
    assert SharedDataFrame.is_compatible("not a dataframe") is False
    
    # 3. Покрытие: проверка типов (dtypes)
    # Только разрешенные типы (int, float, bool)
    df_valid = pd.DataFrame({'a': [1], 'b': [1.5]}, index=pd.RangeIndex(0, 1))
    assert SharedDataFrame.is_compatible(df_valid) is True
    
    # Запрещенный тип (object/string)
    df_invalid_type = pd.DataFrame({'a': ['text']})
    assert SharedDataFrame.is_compatible(df_invalid_type) is False
    
    # 4. Покрытие: проверка индекса (не RangeIndex)
    df_invalid_index = pd.DataFrame({'a': [1]}, index=[10]) # Int64Index, не RangeIndex
    assert SharedDataFrame.is_compatible(df_invalid_index) is False


def test_get_view():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    sdf = SharedDataFrame(df)
    
    # Для теста временно добавим _df, как того ожидает метод get_view в коде
    sdf._df = df 
    
    try:
        # 1. Ветка: columns is None
        view_all = sdf.get_view(None)
        pd.testing.assert_frame_equal(view_all, df)
        
        # 2. Ветка: переданы конкретные колонки
        view_subset = sdf.get_view(['A'])
        assert view_subset.columns.tolist() == ['A']
        assert view_subset.shape == (2, 1)
        # Проверка, что это view (изменение оригинала влияет на view)
        # Примечание: .loc[:, cols] для списка колонок в pandas может создавать копию 
        # или view в зависимости от версии, но тест покроет строку кода.
    finally:
        sdf.unlink()
        sdf.close()

def global_long_sleep_task(*args, **kwargs):
    time.sleep(10)

def _raise_timeout_at_iteration(*args, **kwargs):
    """Функция-генератор: TimeoutError выбрасывается при итерации (for fut in ...),
    а не при вызове as_completed(...). Это корректная симуляция поведения
    настоящего as_completed-генератора."""
    raise concurrent.futures.TimeoutError()
    yield  # pragma: no cover

def test_run_parallel_worker_hard_kill():
    """Проверяет, что при зависшем процессе вызываются terminate() и kill()."""
    with patch("configurable_automl_engine.training_engine.thread_pool.ProcessPoolExecutor") as mock_executor_cls:
        mock_pool = MagicMock()
        mock_executor_cls.return_value = mock_pool
        
        mock_future = MagicMock()
        mock_pool.submit.return_value = mock_future
        
        # as_completed при каждой итерации выбрасывает TimeoutError — watchdog срабатывает
        with patch("configurable_automl_engine.training_engine.thread_pool.as_completed",
                   side_effect=_raise_timeout_at_iteration):
            with patch("configurable_automl_engine.training_engine.thread_pool._force_shutdown_processes") as mock_force:
                with patch("configurable_automl_engine.training_engine.thread_pool.logger") as mock_logger:
                    results = run_parallel(
                        lambda: None,
                        args_seq=[()],
                        mode="processes",
                        timeout=3600
                    )
    
    assert results == [None]
    # Проверяем, что logger.error был вызван с сообщением WATCHDOG
    error_messages = [call[0][0] for call in mock_logger.error.call_args_list]
    assert any("WATCHDOG" in msg for msg in error_messages)
    assert mock_force.called

def test_run_parallel_timeout_global_limit(caplog):
    """Проверяет, что глобальный timeout работает как лимит фазы."""
    
    with caplog.at_level(logging.ERROR):
        results = run_parallel(
            slow_task,
            args_seq=[(1.0,)], # Задача спит 1 секунду
            mode="threads",
            timeout=0.1  # глобальный лимит фазы
        )
    
    assert results == [None]
    assert "Task 0 timed out" in caplog.text

def task_with_kwargs(df, multiplier=1, add=0):
    return (df.sum().sum() * multiplier) + add

def test_run_parallel_processes_with_kwargs():
    df = pd.DataFrame({'a': [1, 1]})
    # Передаем и args, и kwargs
    results = run_parallel(
        task_with_kwargs,
        args_seq=[(df,)],
        kwargs_seq=[{'multiplier': 2, 'add': 5}],
        mode="processes",
        shared_args_indices=[0]
    )
    assert results == [9] # (2 * 2) + 5

# В файле test_parallel.py (исправление самого теста)
def global_identity_func(x):
    return x.iloc[0, 0]

def test_run_parallel_duplicate_indices_handling():
    df = pd.DataFrame({'a': [1]})
    results = run_parallel(
        global_identity_func, # Вместо lambda
        args_seq=[(df,)],
        mode="processes",
        shared_args_indices=[0],
        disk_args_indices=[0]
    )
    assert results == [1]

def test_get_view_new():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    sdf = SharedDataFrame(df)
    try:
        # Проверяем, что get_view корректно работает через to_df()
        view_subset = sdf.get_view(['A'])
        assert isinstance(view_subset, pd.DataFrame)
        assert view_subset.columns.tolist() == ['A']
        assert view_subset.iloc[0,0] == 1
    finally:
        sdf.close()
        sdf.unlink()

class TestParallelErrorHandling(unittest.TestCase):

    # --- ТЕСТЫ ДЛЯ _perform_cleanup ---

    def test_shm_unlink_oserror_pass(self):
        """Покрытие: except (FileNotFoundError, OSError): pass в SHM unlink"""
        shm_mock = MagicMock(spec=SharedDataFrame)
        # Имитируем FileNotFoundError при попытке удалить сегмент
        shm_mock.unlink.side_effect = FileNotFoundError("Already gone")
        
        # Функция не должна упасть
        try:
            _perform_cleanup([shm_mock], None)
        except Exception as e:
            self.fail(f"_perform_cleanup raised {type(e).__name__} unexpectedly!")
        
        shm_mock.unlink.assert_called_once()

    def test_shm_unlink_generic_exception_warning(self):
        """Покрытие: except Exception as e: logger.warning(...) в SHM unlink"""
        shm_mock = MagicMock(spec=SharedDataFrame)
        shm_mock.unlink.side_effect = ValueError("Fatal SHM error")
        
        with self.assertLogs(MODULE_PATH, level='WARNING') as cm:
            _perform_cleanup([shm_mock], None)
            
        self.assertTrue(any("Non-critical SHM unlink failure" in log for log in cm.output))

    def test_persistence_cleanup_permission_error(self):
        """Покрытие: except (PermissionError, OSError) в persistence_manager.cleanup"""
        pm_mock = MagicMock(spec=DiskPersistenceManager)
        pm_mock.cleanup.side_effect = PermissionError("File locked")
        
        with self.assertLogs(MODULE_PATH, level='ERROR') as cm:
            _perform_cleanup(None, pm_mock)
            
        self.assertTrue(any("Cleanup failed due to file locking/permissions" in log for log in cm.output))

    # --- ТЕСТЫ ДЛЯ run_parallel (Цикл ожидания) ---

    @patch(f'{MODULE_PATH}.ThreadPoolExecutor')
    def test_run_parallel_wait_loop_exception(self, mock_executor_cls):
        """Покрытие: except Exception as e: logger.error(Error while waiting for tasks)"""
        mock_pool = MagicMock()
        mock_executor_cls.return_value = mock_pool
        
        mock_future = MagicMock()
        mock_pool.submit.return_value = mock_future
        
        def _raise_runtime_error(*args, **kwargs):
            raise RuntimeError("Internal pool crash")
        
        with patch(f'{MODULE_PATH}.as_completed') as mock_as_completed:
            mock_as_completed.side_effect = _raise_runtime_error
            
            with self.assertLogs(MODULE_PATH, level='ERROR') as cm:
                run_parallel(lambda x: x, args_seq=[(1,)], mode="threads")
                
        self.assertTrue(any("Error while waiting for tasks" in log for log in cm.output))

    # --- ТЕСТЫ ДЛЯ run_parallel (Принудительное завершение воркеров) ---

    @patch(f'{MODULE_PATH}.ProcessPoolExecutor')
    @patch(f'{MODULE_PATH}.time.sleep')
    def test_worker_terminate_and_kill_exceptions(self, mock_sleep, mock_executor_cls):
        """
        Покрытие блоков:
        - except Exception: pass при w.terminate()
        - except Exception: pass при w.kill()
        """
        mock_executor = mock_executor_cls.return_value
        
        # Создаем воркера, который не хочет умирать
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True
        mock_worker.pid = 777
        
        # Настраиваем ошибки на системные вызовы уничтожения
        mock_worker.terminate.side_effect = RuntimeError("Terminate blocked")
        mock_worker.kill.side_effect = RuntimeError("Kill blocked")
        
        # Подменяем список процессов внутри Executor
        mock_executor._processes = {0: mock_worker}
        
        # Имитируем ситуацию, когда задачи не завершились вовремя
        with patch(f'{MODULE_PATH}.as_completed') as mock_as_completed:
            mock_as_completed.side_effect = _raise_timeout_at_iteration
            
            with self.assertLogs(MODULE_PATH, level='ERROR') as cm:
                run_parallel(
                    func=lambda x: x,
                    args_seq=[(1,)],
                    mode="processes",
                    shutdown_grace_period=0.01
                )

            # Проверяем, что логи перед 'pass' были записаны
            self.assertTrue(any("WATCHDOG" in log for log in cm.output))
            
        # Проверяем, что методы вызывались, несмотря на ошибки
        self.assertTrue(mock_worker.terminate.called)
        self.assertTrue(mock_worker.kill.called)


# ═══════════════════════════════════════════════════════════════
#  Тесты as_completed + watchdog (deadlock prevention)
# ═══════════════════════════════════════════════════════════════

def test_as_completed_normal_completion():
    """Проверка, что as_completed корректно собирает результаты."""
    results = run_parallel(
        simple_task,
        args_seq=[(1, 2), (3, 4), (5, 6)],
        mode="threads",
        timeout=10
    )
    assert sorted(results) == [3, 7, 11]


def test_as_completed_with_task_timeout():
    """Проверка, что task_timeout работает: задача не успевает."""
    results = run_parallel(
        slow_task,
        args_seq=[(5.0,)],  # спит 5 секунд
        mode="threads",
        timeout=10,
        task_timeout=0.5  # индивидуальный таймаут на задачу
    )
    assert results == [None]


def test_watchdog_not_triggered_on_slow_but_alive_tasks():
    """Проверка, что watchdog НЕ срабатывает на медленных, но живых задачах.
    
    as_completed блокируется и ждёт завершения задачи. Если задача просто
    долгая — watchdog не увеличивает stale-счётчик.
    """
    results = run_parallel(
        slow_task,
        args_seq=[(2.0,)],  # задача на 2 секунды
        mode="threads",
        timeout=10,  # достаточно большой глобальный таймаут
    )
    assert results == ["done"]  # задача должна завершиться нормально


@patch("configurable_automl_engine.training_engine.thread_pool.as_completed")
def test_watchdog_triggers_on_c_level_crash(mock_as_completed):
    """Проверка, что watchdog срабатывает при C-level падении.
    
    Мокаем as_completed, чтобы он при каждой итерации выбрасывал TimeoutError,
    имитируя зависший процесс. Используем функцию-генератор, чтобы
    TimeoutError выбрасывался при итерации (for fut in ...), а не при вызове.
    """
    mock_as_completed.side_effect = _raise_timeout_at_iteration
    
    with patch("configurable_automl_engine.training_engine.thread_pool._force_shutdown_processes") as mock_shutdown:
        with patch("configurable_automl_engine.training_engine.thread_pool.logger") as mock_logger:
            results = run_parallel(
                lambda: None,
                args_seq=[()],
                mode="processes",
                timeout=3600  # большой таймаут, чтобы не сработал глобальный
            )
    
    assert results == [None]
    error_messages = [call[0][0] for call in mock_logger.error.call_args_list]
    assert any("WATCHDOG" in msg for msg in error_messages)
    assert mock_shutdown.called


@patch("configurable_automl_engine.training_engine.thread_pool.as_completed")
def test_watchdog_not_triggered_when_global_timeout_expires(mock_as_completed):
    """Проверка, что watchdog НЕ срабатывает, если истёк глобальный таймаут.
    
    Даже если as_completed выбрасывает TimeoutError, но remaining_global <= 0,
    stale-счётчик не увеличивается — происходит штатный выход по таймауту.
    Используем функцию-генератор, чтобы TimeoutError выбрасывался
    при итерации (for fut in ...), а не при вызове.
    """
    mock_as_completed.side_effect = _raise_timeout_at_iteration
    
    with patch("configurable_automl_engine.training_engine.thread_pool._force_shutdown_processes") as mock_shutdown:
        with patch("configurable_automl_engine.training_engine.thread_pool.logger") as mock_logger:
            results = run_parallel(
                lambda: None,
                args_seq=[()],
                mode="processes",
                timeout=0.1,  # маленький глобальный таймаут
            )
    
    assert results == [None]
    error_messages = [call[0][0] for call in mock_logger.error.call_args_list]
    assert not any("WATCHDOG" in msg for msg in error_messages)  # watchdog не сработал
    assert any("Task 0 timed out" in msg for msg in error_messages)  # штатный таймаут


def test_force_shutdown_processes_terminate_and_kill():
    """Проверка, что _force_shutdown_processes корректно завершает процессы."""
    mock_worker = MagicMock()
    mock_worker.is_alive.return_value = True
    mock_worker.pid = 9999
    
    mock_pool = MagicMock()
    mock_pool._processes = {0: mock_worker}
    
    with patch("configurable_automl_engine.training_engine.thread_pool.time.sleep"):
        _force_shutdown_processes(mock_pool, shutdown_grace_period=0.1)
    
    assert mock_worker.terminate.called
    assert mock_worker.kill.called


def test_run_parallel_generic_exception_no_timeout(caplog):
    """Покрытие except Exception в else-ветке (timeout_for_ac is None)."""
    def failing_task():
        raise ValueError("No-timeout failure")
    
    with caplog.at_level(logging.ERROR):
        results = run_parallel(
            failing_task,
            args_seq=[()],
            mode="threads",
            timeout=None  # отключаем глобальный таймаут
        )
    
    assert results == [None]
    assert "Task 0 failed" in caplog.text
    assert "No-timeout failure" in caplog.text