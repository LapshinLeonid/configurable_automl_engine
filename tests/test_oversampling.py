import sys
import os
import pytest
import pandas as pd
import numpy as np
import threading
import pickle
from pathlib import Path

from unittest.mock import patch

# Импорты согласно вашей структуре
from configurable_automl_engine.oversampling import DataOversampler, oversample as functional_oversample

class TestDataOversampler:
    """Полный набор тестов для DataOversampler (20 сценариев)"""

    @pytest.fixture
    def sample_data(self):
        """Стандартный набор: 3 примера '0', 2 примера '1'"""
        return pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'target': [0, 0, 0, 1, 1]
        })

    # --- 1. ТЕСТЫ АЛГОРИТМОВ (Базовые сценарии) ---

    def test_basic_oversampling_smote(self, sample_data):
        """Проверка SMOTE: multiplier=2 -> 10 строк"""
        sampler = DataOversampler(multiplier=2, algorithm='smote')
        result = sampler.oversample(sample_data, target='target')
        assert len(result) == 10
        assert 'target' in result.columns
        assert result.notna().all().all()

    def test_basic_oversampling_adasyn(self):
        # Создаем больше данных, чтобы у миноритарного класса были соседи из мажоритарного
        data = pd.DataFrame({
            'f1': np.arange(20),
            'target': [0]*15 + [1]*5
        })
        sampler = DataOversampler(multiplier=1.2, algorithm='adasyn')
        result = sampler.oversample(data, target='target')
        assert len(result) > 20
        assert result.notna().all().all()

    def test_random_without_noise(self, sample_data):
        """Проверка Random без шума"""
        sampler = DataOversampler(multiplier=3, algorithm='random', add_noise=False)
        result = sampler.oversample(sample_data)
        assert len(result) == 15
        # Проверяем, что это просто дубликаты
        assert result.duplicated().any()

    def test_random_with_noise_flag(self, sample_data):
        """Проверка флага add_noise в Random"""
        sampler = DataOversampler(multiplier=3, algorithm='random', add_noise=True)
        result = sampler.oversample(sample_data)
        # Значения должны быть уникальными из-за гауссовского шума
        assert not result.drop('target', axis=1).duplicated().any()

    def test_random_with_noise_algorithm_name(self, sample_data):
        """Проверка алгоритма 'random_with_noise' (синоним через имя)"""
        sampler = DataOversampler(multiplier=2, algorithm='random_with_noise')
        result = sampler.oversample(sample_data)
        assert len(result) == 10

    # --- 2. ТЕСТЫ ТИПОВ ДАННЫХ (Валидация sklearn) ---

    def test_dataframe_input_output(self, sample_data):
        """Вход DF -> Выход DF"""
        sampler = DataOversampler()
        result = sampler.oversample(sample_data)
        assert isinstance(result, pd.DataFrame)

    def test_array_input(self):
        """Вход numpy array через fit_resample (стандарт imblearn)"""
        X = np.random.rand(10, 2)
        y = np.array([0,0,0,0,0,1,1,1,1,1])
        sampler = DataOversampler(multiplier=2)
        X_res, y_res = sampler.fit_resample(X, y)
        assert isinstance(X_res, np.ndarray)
        assert len(X_res) == 20

    def test_list_input(self):
        """Вход list (должен перевариваться внутренним _check_X_y)"""
        X = [[1, 2], [3, 4], [5, 6], [7, 8]]
        y = [0, 0, 1, 1]
        sampler = DataOversampler(multiplier=2)
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) == 8

    # --- 3. ПАРАМЕТРЫ И ГРАНИЦЫ ---

    def test_multiplier_1(self, sample_data):
        """multiplier=1 не должен изменять размер данных"""
        sampler = DataOversampler(multiplier=1.0)
        result = sampler.oversample(sample_data)
        assert len(result) == len(sample_data)

    def test_fractional_multiplier(self, sample_data):
        """Проверка дробного множителя (ceil)"""
        sampler = DataOversampler(multiplier=1.5) # 3*1.5=4.5->5, 2*1.5=3. Итого 8.
        result = sampler.oversample(sample_data)
        assert len(result) == 8

    def test_invalid_multiplier(self, sample_data):
        sampler = DataOversampler(multiplier=-1.0)
        # Ошибка вылетит внутри fit_resample -> _strategy -> ValueError при создании словаря стратегии
        with pytest.raises(Exception): 
            sampler.oversample(sample_data)

    def test_unknown_algorithm(self, sample_data):
        """Неподдерживаемый алгоритм"""
        sampler = DataOversampler(algorithm='invalid_method')
        with pytest.raises(ValueError, match="Неподдерживаемый алгоритм"):
            sampler.oversample(sample_data)

    # --- 4. СПЕЦИФИЧЕСКАЯ ЛОГИКА (Новый код) ---

    def test_smote_single_minority_sample_error(self):
        """
        SMOTE математически не определён при 1 объекте миноритарного класса.
        Ожидаем ValueError согласно контракту imbalanced-learn.
        """
        data = pd.DataFrame({'f1': [1, 2, 3, 4], 'target': [0, 0, 0, 1]})

        sampler = DataOversampler(algorithm='smote', multiplier=2.0)

        with pytest.raises(ValueError):
            sampler.oversample(data)

    def test_single_class_error(self):
        """imblearn требует минимум 2 класса для работы"""
        data = pd.DataFrame({'f1': [1, 2, 3], 'target': [0, 0, 0]})
        sampler = DataOversampler()
        with pytest.raises(ValueError):
            sampler.oversample(data)

    def test_find_target_in_middle(self):
        """Поиск таргета, если он не последний"""
        df = pd.DataFrame({'f1': [1, 2, 3, 4], 'target': [0, 0, 1, 1], 'f2': [5, 6, 7, 8]})
        sampler = DataOversampler(multiplier=2)
        result = sampler.oversample(df, target='target')
        assert list(result.columns) == ['f1', 'f2', 'target'] # Сборка в oversample() переносит таргет в конец
        assert len(result) == 8

    # --- 5. СИСТЕМНЫЕ ТЕСТЫ ---

    def test_logging_creation(self, tmp_path, sample_data):
        """Проверка логов и их структуры"""
        log_dir = tmp_path / "logs"
        sampler = DataOversampler(log_dir=str(log_dir), algorithm='random')
        sampler.oversample(sample_data)
        
        log_file = log_dir / "oversampler.log"
        assert log_file.exists()
        content = log_file.read_text()
        assert "INFO" in content
        assert "Random resample" in content

    def test_logging_error(self, tmp_path):
        # Указываем путь к файлу внутри существующей временной папки
        log_dir = str(tmp_path)
        sampler = DataOversampler(log_dir=log_dir)
        with pytest.raises(Exception):
            # Передаем невалидный объект, чтобы вызвать ошибку в _fit_resample
            sampler.oversample(None)
        
        assert os.path.exists(sampler.log_path_)
        assert "ERROR" in Path(sampler.log_path_).read_text()

    def test_pickle_compatibility(self):
        """Pickle (важно для мультипроцессинга в GridSearchCV)"""
        sampler = DataOversampler(multiplier=2, algorithm='smote')

        dump = pickle.dumps(sampler)
        load = pickle.loads(dump)

        assert load.algorithm == 'smote'

        # lock восстановлен
        assert hasattr(load, "_lock")
        assert load._lock is not None

        # lock реентерабельный (ключевое свойство)
        acquired = load._lock.acquire(blocking=False)
        try:
            assert acquired is True
        finally:
            if acquired:
                load._lock.release()

    def test_multithreading_safety(self, sample_data, tmp_path):
        """Проверка lock при записи в лог из разных потоков"""
        sampler = DataOversampler(log_dir=str(tmp_path))
        def run():
            for _ in range(5): sampler.oversample(sample_data)
        
        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        
        assert os.path.exists(sampler.log_path_)

    def test_functional_interface(self, sample_data):
        """Проверка быстрой функции-обертки"""
        result = functional_oversample(sample_data, multiplier=2, algorithm='random')
        assert len(result) == 10


@pytest.fixture
def sample_data():
    """Создает минимальный DataFrame для тестов."""
    return pd.DataFrame({
        'feature1': np.random.rand(10),
        'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    })
# --- Тесты для строк 66-67 (Ошибка при получении фрейма стека) ---
def test_log_frame_error(sample_data, tmp_path):
    """Тестирует случай, когда sys._getframe вызывает ValueError/AttributeError (строки 66-67)."""
    log_dir = str(tmp_path / "logs")
    sampler = DataOversampler(log_dir=log_dir)
    
    # Патчим sys._getframe, чтобы он выбрасывал ValueError
    with patch('sys._getframe', side_effect=ValueError("Frame not found")):
        # Вызываем метод, который инициирует логирование
        sampler._log("Test message")
        
    # Проверяем, что в лог записалось 'unknown' вместо имени файла/функции
    with open(sampler.log_path_, 'r') as f:
        content = f.read()
        assert "unknown:unknown" in content
        assert "Test message" in content
# --- Тесты для строк 73-74 (Ошибка записи в файл лога) ---

def test_fit_resample_exception_logging(sample_data, tmp_path):
    log_dir = str(tmp_path / "logs_fit")
    # Передаем корректный multiplier, но ломаем алгоритм, чтобы вызвать исключение внутри блока try
    sampler = DataOversampler(multiplier=1.0, algorithm="invalid_algo", log_dir=log_dir)
    
    X = sample_data.drop('target', axis=1)
    y = sample_data['target']
    # Это вызовет ValueError("Неподдерживаемый алгоритм...") внутри блока try
    with pytest.raises(ValueError, match="Неподдерживаемый алгоритм"):
        sampler._fit_resample(X, y)
        
    # Теперь файл точно существует, так как _log был вызван в блоке except
    assert os.path.exists(sampler.log_path_)
    with open(sampler.log_path_, 'r', encoding='utf-8') as f:
        content = f.read()
        assert "ERROR: Ошибка в _fit_resample" in content
        assert "Неподдерживаемый алгоритм" in content

def test_log_write_exception_coverage(tmp_path, capsys):
    """
    Тест специально для покрытия строк 73-74:
    Имитация ошибки при записи в файл лога и проверка вывода в stderr.
    """
    # 1. Инициализируем оверсемплер
    log_dir = str(tmp_path / "crash_test_logs")
    sampler = DataOversampler(log_dir=log_dir)
    
    # Сообщение, которое мы попытаемся отправить в лог
    error_msg = "Critical disk failure simulation"
    
    # 2. Патчим 'builtins.open'. 
    # Мы выбрасываем RuntimeError (или любое Exception), чтобы попасть в блок 'except Exception as e'
    with patch("builtins.open", side_effect=RuntimeError("Simulated OS Error")):
        # Вызываем метод логирования напрямую
        sampler._log(error_msg)
    
    # 3. Проверяем, что сработал print(..., file=sys.stderr)
    captured = capsys.readouterr()
    
    # Строка 74: print(f"Не удалось записать лог: {e}", file=sys.stderr)
    assert "Не удалось записать лог: Simulated OS Error" in captured.err
    
    # Дополнительно убеждаемся, что программа не «упала», а просто вывела ошибку
    assert True