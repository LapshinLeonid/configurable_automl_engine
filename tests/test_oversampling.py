import pytest
import pandas as pd
import numpy as np
import threading
import pickle
import logging
from unittest.mock import MagicMock, patch

# Импорты согласно вашей структуре
from configurable_automl_engine.oversampling import (
    DataOversampler,
      oversample as functional_oversample
)

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
        # Создаем больше данных, чтобы у миноритарного класса 
        # были соседи из мажоритарного
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
        # Ошибка вылетит внутри fit_resample -> _strategy -> ValueError 
        # при создании словаря стратегии
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
        df = pd.DataFrame(
            {'f1': [1, 2, 3, 4], 
             'target': [0, 0, 1, 1], 
             'f2': [5, 6, 7, 8]
             }
             )
        sampler = DataOversampler(multiplier=2)
        result = sampler.oversample(df, target='target')
        # Сборка в oversample() переносит таргет в конец
        assert list(result.columns) == ['f1', 'f2', 'target'] 
        assert len(result) == 8

    # --- 5. СИСТЕМНЫЕ ТЕСТЫ ---

    def test_logging_creation(self, caplog, sample_data):
        """Проверка логов и их структуры"""
        with caplog.at_level(logging.INFO):
            sampler = DataOversampler(algorithm='random')
            sampler.oversample(sample_data)
        assert "Random resample" in caplog.text
        assert "INFO" in caplog.text

    def test_logging_error(self, caplog):
        sampler = DataOversampler()
        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception):
                sampler.oversample(None)
        assert "ERROR" in caplog.text
        assert "Ошибка в oversample" in caplog.text

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

    def test_multithreading_safety(self, sample_data):
        """Проверка lock при записи в лог из разных потоков"""
        sampler = DataOversampler()
        def run():
            for _ in range(5): 
                sampler.oversample(sample_data)
        threads = [threading.Thread(target=run) for _ in range(4)]
        for t in threads: 
            t.start()
        for t in threads: 
            t.join()

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


def test_critical_error_logging():
    # 1. Готовим данные
    X = pd.DataFrame(np.random.rand(10, 2), columns=['f1', 'f2'])
    y = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) # Сбалансированные данные
    
    # 2. Создаем экземпляр
    sampler = DataOversampler(algorithm="random")
    
    # 3. Патчим только логгер в модуле oversampling
    with patch("configurable_automl_engine.oversampling.logger") as mock_logger:
        
        # 4. Имитируем "непредвиденную" ошибку через метод _strategy
        # Этот метод вызывается внутри вашего _fit_resample СРАЗУ ПОСЛЕ _check_X_y
        error_text = "Unexpected Crash during Strategy"
        
        with patch.object(sampler, "_strategy", side_effect=AttributeError(error_text)):
            
            # Проверяем, что ошибка (AttributeError) пробрасывается наружу
            with pytest.raises(AttributeError, match=error_text):
                sampler.fit_resample(X, y)
        
        # 5. ПРОВЕРКА ЛОГОВ (те самые строки 136-139)
        
        # Проверяем logger.debug(..., exc_info=True)
        mock_logger.debug.assert_called_once_with(
            "Unexpected error trace in _fit_resample:",
            exc_info=True
        )
        
        # Проверяем logger.error(f"Critical error...")
        mock_logger.error.assert_called_once_with(
            f"Critical error during data oversampling: {error_text}"
        )

def test_catch_generic_exception_only():
    """
    Проверяет, что блок ловит именно Exception, 
    который не входит в список (ValueError, TypeError и т.д.)
    """
    with patch("configurable_automl_engine.oversampling.logger") as mock_log:
        sampler = DataOversampler()
        
        # Вызываем ошибку, которая НЕ должна логироваться этим блоком (ValueError)
        # так как она перехватывается выше и просто пробрасывается (re-raise)
        with patch.object(sampler, "_check_X_y", side_effect=ValueError("Standard Data Error")):
            with pytest.raises(ValueError):
                sampler.fit_resample(None, None)
        
        # Logger не должен был вызваться, так как ValueError ушел в первый except
        assert mock_log.error.call_count == 0