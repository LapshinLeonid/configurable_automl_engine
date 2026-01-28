import sys
import os
import pytest
import pandas as pd
import numpy as np

# Добавляем путь к корню проекта
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from configurable_automl_engine.oversampling import DataOversampler  

class TestDataOversampler:
    """Тест-кейсы для класса DataOversampler"""

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 0, 0, 1, 1]
        })

    # Базовые сценарии
    def test_basic_oversampling_smote(self, sample_data):
        oversampler = DataOversampler()
        result = oversampler.oversample(sample_data, multiplier=2, algorithm='smote')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10, "Неправильное количество образцов после увеличения"
        assert 'target' in result.columns, "Потеряна целевая переменная"
        assert result.notna().all().all(), "Наличие NaN в данных"

    def test_basic_oversampling_adasyn(self, sample_data):
        """Проверка метода ADASYN"""
        oversampler = DataOversampler()
        result = oversampler.oversample(sample_data, multiplier=2, algorithm='adasyn')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10, "Неправильное количество образцов после ADASYN"
        assert 'target' in result.columns, "Потеряна целевая переменная при ADASYN"
        assert result.notna().all().all(), "Наличие NaN после ADASYN"

    # Различные алгоритмы
    def test_random_without_and_with_noise(self, sample_data):
        oversampler = DataOversampler()
        res_no_noise = oversampler.oversample(sample_data, multiplier=3, algorithm='random', add_noise=False)
        res_with_noise = oversampler.oversample(sample_data, multiplier=3, algorithm='random', add_noise=True)
        assert isinstance(res_no_noise, pd.DataFrame)
        assert isinstance(res_with_noise, pd.DataFrame)
        assert len(res_no_noise) == len(res_with_noise) == 15
        assert not res_no_noise.iloc[:, :-1].equals(res_with_noise.iloc[:, :-1])

    # Тестирование типов данных: всегда возвращается DataFrame
    def test_dataframe_input(self, sample_data):
        oversampler = DataOversampler()
        result = oversampler.oversample(sample_data, multiplier=2)
        assert isinstance(result, pd.DataFrame), "Для DataFrame должен возвращаться DataFrame"

    def test_array_input(self, sample_data):
        oversampler = DataOversampler()
        arr_data = sample_data.values
        result = oversampler.oversample(arr_data, multiplier=2)
        assert isinstance(result, pd.DataFrame), "Для массива должен возвращаться DataFrame"

    def test_list_input(self, sample_data):
        oversampler = DataOversampler()
        list_data = sample_data.values.tolist()
        result = oversampler.oversample(list_data, multiplier=2)
        assert isinstance(result, pd.DataFrame), "Для списка должен возвращаться DataFrame"

    # Тестирование параметров
    def test_multiplier_1(self, sample_data):
        oversampler = DataOversampler()
        result = oversampler.oversample(sample_data, multiplier=1)
        assert len(result) == len(sample_data), "При множителе 1 данные не должны меняться"

    # Тестирование обработки ошибок
    def test_invalid_input_type(self):
        oversampler = DataOversampler()
        with pytest.raises(TypeError):
            oversampler.oversample("invalid_data", multiplier=2)

    def test_invalid_multiplier(self, sample_data):
        oversampler = DataOversampler()
        with pytest.raises(ValueError):
            oversampler.oversample(sample_data, multiplier=0)
        with pytest.raises(ValueError):
            oversampler.oversample(sample_data, multiplier=-5)

    def test_unknown_algorithm(self, sample_data):
        oversampler = DataOversampler()
        with pytest.raises(ValueError):
            oversampler.oversample(sample_data, multiplier=2, algorithm='invalid')

    # Граничные случаи
    def test_empty_data(self):
        oversampler = DataOversampler()
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            oversampler.oversample(empty_df, multiplier=2)

    def test_single_class_data(self):
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 0, 0]
        })
        oversampler = DataOversampler()
        result = oversampler.oversample(data, multiplier=3, algorithm='random')
        assert len(result) == 9, "Проблема с одним классом"
    
    def test_smote_adasyn_insufficient_samples(self):
        """Test SMOTE/ADASYN raise ValueError with < 2 samples in minority class (i.e. 1 sample)."""
        data_insufficient = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [10, 20, 30, 40],
            'target': [0, 0, 0, 1]  # Minority class '1' has only 1 sample
        })
        oversampler = DataOversampler()

        # imblearn's k_neighbors/n_neighbors becomes 2 due to k_neighbors+1 logic for estimator,
        # while n_samples for minority class is 1.
        expected_error_msg = r"Expected n_neighbors <= n_samples_fit, but n_neighbors = 2, n_samples_fit = 1, n_samples = 1"

        with pytest.raises(ValueError, match=expected_error_msg):
            oversampler.oversample(data_insufficient.copy(), multiplier=2, algorithm='smote')

        with pytest.raises(ValueError, match=expected_error_msg):
            oversampler.oversample(data_insufficient.copy(), multiplier=2, algorithm='adasyn')

    def test_min_count(self):
        df = pd.DataFrame({"f1": [1], "f2": [2], "target": [0]})
        oversampler = DataOversampler()
        result = oversampler.oversample(df, multiplier=2, algorithm='smote')

    # Многопоточность: разные n_jobs дают консистентный результат
    def test_multithreading_consistency(self, sample_data):
        oversampler = DataOversampler(n_jobs=1)
        result_single = oversampler.oversample(sample_data, multiplier=3)
        for n_jobs in [2, 4]:
            oversampler = DataOversampler(n_jobs=n_jobs)
            result_multi = oversampler.oversample(sample_data, multiplier=3)
            pd.testing.assert_frame_equal(
                result_single.sort_index(axis=1),
                result_multi.sort_index(axis=1),
                check_dtype=False
            )

    # Проверка балансировки классов для SMOTE
    def test_class_balancing(self, sample_data):
        multiplier = 4
        oversampler = DataOversampler()
        result = oversampler.oversample(sample_data, multiplier=multiplier, algorithm='smote')
        
        orig_counts = sample_data['target'].value_counts()
        new_counts = result['target'].value_counts()
        
        for cls, orig_cnt in orig_counts.items():
            assert new_counts[cls] == orig_cnt * multiplier, (f"Класс {cls}: ожидалось {orig_cnt*multiplier}, получено {new_counts[cls]}")

    # Проверка логирования
    def test_logging(self, tmp_path, sample_data):
        log_file = tmp_path / "oversampler.log"
        oversampler = DataOversampler(log_dir=str(tmp_path))
        oversampler.oversample(sample_data, multiplier=2)
        assert log_file.exists(), "Лог-файл не создан"
        content = log_file.read_text()
        assert "INFO" in content, "Нет информационных записей в логе"

    def test_error_logging(self, tmp_path):
        log_file = tmp_path / "oversampler.log"
        oversampler = DataOversampler(log_dir=str(tmp_path))
        with pytest.raises(TypeError):
            oversampler.oversample("invalid_data", multiplier=2)
        content = log_file.read_text()
        assert "ERROR" in content, "Нет записей об ошибках в логе"

    # Тестирование случая когда таргет не в конце
    def test_find_target(self):
        df = pd.DataFrame({
        'f1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'target': [0, 0, 1, 1, 1],
        'f2': [0.9, 0.8, 0.7, 0.6, 0.5],
        })

        oversampler = DataOversampler()
        result = oversampler.oversample(df, multiplier=2, algorithm='smote', target='target')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10, "Неправильное количество образцов после увеличения"
        assert 'target' in result.columns, "Потеряна целевая переменная"
        assert result.notna().all().all(), "Наличие NaN в данных"