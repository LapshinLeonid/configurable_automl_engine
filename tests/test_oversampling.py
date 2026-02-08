import pytest
import pandas as pd
import numpy as np
import threading
import pickle
import logging
from scipy import sparse
from unittest.mock import patch
from math import ceil

from pandas.api.types import is_numeric_dtype

# Импорты согласно вашей структуре
from configurable_automl_engine.oversampling import (
    DataOversampler,
      oversample as functional_oversample,
      oversample
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
        with pytest.raises(ValueError, match="Unsupported algorithm:"):
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

# 1. Покрытие строк 90-91: Шум в колонках типа object (содержащих числа)
def test_noise_on_object_numeric_cols():
    df = pd.DataFrame({
        'A': ['1.0', '2.0', '3.0', '4.0'], # Объект, но внутри числа
        'target': [0, 0, 1, 1]
    })
    # Используем множитель 1 и шум
    sampler = DataOversampler(multiplier=1.0, add_noise=True, noise_level=0.1)
    res = sampler.oversample(df, target='target')
    
    # Проверяем, что значения в 'A' изменились (зашумлены)
    assert not np.array_equal(res['A'].astype(float).values, df['A'].astype(float).values)
# 2. Покрытие строк 131-140: Сериализация (pickle)
def test_serialization_pickle():
    sampler = DataOversampler(multiplier=2.0, algorithm="random")
    # Проверка удаления и восстановления _lock
    dumped = pickle.dumps(sampler)
    loaded = pickle.loads(dumped)
    
    assert loaded.multiplier == 2.0
    assert hasattr(loaded, '_lock')
    assert isinstance(loaded._lock, type(sampler._lock))
# 3. Покрытие строк 180, 184, 189, 193: Восстановление типов (int/float)
@pytest.mark.parametrize("dtype, add_noise, expected_dtype", [
    (np.int64, True, np.float64),   # Строка 180
    (np.int32, True, np.float32),   # Строка 184 (через else)
    (np.float32, False, np.float32),# Строка 189
])
def test_dtype_restoration(dtype, add_noise, expected_dtype):
    df = pd.DataFrame({'A': np.array([1, 2], dtype=dtype), 'target': [0, 1]})
    sampler = DataOversampler(multiplier=1.0, add_noise=add_noise)
    res = sampler.oversample(df, target='target')
    assert res['A'].dtype == expected_dtype
# 4. Покрытие строки 211: Ошибка multiplier < 1
def test_invalid_multiplier():
    sampler = DataOversampler(multiplier=0.5)
    df = pd.DataFrame({'A': [1, 2], 'target': [0, 1]})
    with pytest.raises(ValueError, match="multiplier must be >= 1"):
        sampler.fit_resample(df[['A']], df['target'])
# 5. Покрытие строк 220-223: Шум + разреженная матрица
def test_sparse_matrix_noise_error():
    X_sparse = sparse.csr_matrix([[1, 0], [0, 1], [1, 1]])
    y = np.array([0, 1, 0])
    sampler = DataOversampler(add_noise=True)
    with pytest.raises(TypeError, match="does not support sparse matrices"):
        sampler.fit_resample(X_sparse, y)
# 6. Покрытие строки 259: Неподдерживаемый алгоритм
def test_unsupported_algorithm():
    # Обходим валидатор через прямой вызов или изменение атрибута
    sampler = DataOversampler(algorithm="unknown_algo")
    df = pd.DataFrame({'A': [1, 2, 3, 4], 'target': [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        sampler.fit_resample(df[['A']], df['target'])

def test_restoration_of_boolean_and_unusual_types():
    """Проверка восстановления типов для данных, не являющихся числами или категориями."""
    df = pd.DataFrame({
        'is_valid': [True, False, True, False],
        'target': [0, 0, 1, 1]
    })
    sampler = DataOversampler(algorithm="random", multiplier=1.2)
    result = sampler.oversample(df, target='target')
    # Проверяем, что тип bool сохранился после преобразований numpy
    assert result['is_valid'].dtype == bool
def test_type_restoration_error_handling(caplog):
    """Проверка устойчивости системы к ошибкам при восстановлении типов данных."""
    sampler = DataOversampler()
    df = pd.DataFrame({'feature': [1, 2, 3]})
    # Передаем некорректную серию типов для провокации исключения в блоке обработки
    with caplog.at_level(logging.WARNING):
        sampler._restore_dtypes(df, pd.Series({'feature': 'non_existent_type_object'}))
    assert "Не удалось восстановить тип" in caplog.text
def test_internal_resampling_mechanism_and_locks():
    """Тестирование защищенного метода ресемплирования и работы механизма блокировок."""
    sampler = DataOversampler(algorithm="random")
    features = np.array([[1.0], [2.0], [3.0], [4.0]])
    labels = np.array([0, 0, 1, 1])
    # Проверка работы внутреннего API напрямую
    res_features, res_labels = sampler._fit_resample(features, labels)
    assert isinstance(res_features, np.ndarray)
    assert len(res_features) == len(res_labels)
def test_synthetic_algorithm_validation_without_numeric_data():
    """Проверка защиты SMOTE от запуска на данных без числовых признаков."""
    df = pd.DataFrame({
        'category_a': ['high', 'low', 'high', 'low'],
        'category_b': ['red', 'blue', 'red', 'blue'],
        'target': [0, 0, 1, 1]
    })
    sampler = DataOversampler(algorithm="smote")
    # Ожидаем ошибку, так как SMOTE не может работать только с категориями
    with pytest.raises(TypeError, match="requires at least one numeric feature"):
        sampler.oversample(df, target='target')
def test_exception_propagation_for_invalid_parameters():
    """Проверка корректного проброса исключений при невалидных входных параметрах."""
    # Множитель меньше 1 недопустим по логике библиотеки
    sampler = DataOversampler(multiplier=0.1) 
    X = pd.DataFrame({'f1': [1, 2]})
    y = pd.Series([0, 1])
    with pytest.raises(ValueError, match="multiplier must be >= 1"):
        sampler._fit_resample(X, y)
def test_unsupported_algorithm_name_validation():
    """Проверка валидации названия алгоритма в публичном интерфейсе."""
    df = pd.DataFrame({'f1': [1, 2], 'target': [0, 1]})
    sampler = DataOversampler(algorithm="magic_boost") # Несуществующий алгоритм
    with pytest.raises(ValueError, match="Unsupported algorithm"):
        sampler.oversample(df, target='target')
def test_blind_oversampling_without_target_column():
    """Тестирование 'слепого' увеличения выборки без указания целевой переменной."""
    df = pd.DataFrame({
        'feature_1': [0.1, 0.2, 0.3, 0.4],
        'feature_2': ['A', 'B', 'A', 'B']
    })
    # Проверка работы в режиме простого дублирования строк (Random)
    result = oversample(df, multiplier=2.0, algorithm="random", target=None)
    assert len(result) == 8
    # Убеждаемся, что временные служебные колонки удалены из результата
    assert 'temp_target' not in result.columns

# Тест-кейс для проверки ветки balance=True
def test_strategy_balance_true_coverage(mocker):
    # Создаем мок-объект класса, где реализован _strategy
    # Имитируем self.balance = True
    mock_self = mocker.Mock()
    mock_self.balance = True
    
    # Входные данные: класс 0 (10 шт), класс 1 (100 шт)
    y = pd.Series([0] * 10 + [1] * 100)
    multiplier = 1.5
    
    # Вызываем метод (предположим, он определен в классе Sampler)
    # Если вы тестируете конкретный класс, замените Sampler на ваше имя
    
    # Вызов напрямую через класс, передавая наш mock_self
    result = DataOversampler._strategy(mock_self, y, multiplier)
    
    # Расчет: base_size = 100 (макс от 10 и 100)
    # Итог: ceil(100 * 1.5) = 150 для ВСЕХ классов
    expected = {0: 150, 1: 150}
    
    assert result == expected
    assert result[0] == result[1], "При balance=True размеры классов должны быть равны"


def test_restore_dtypes_none_handling():
    """
    Тест для покрытия веток:
    if original_dtypes is None:
        return df
    """
    # 1. Создаем экземпляр класса
    # Параметры инициализации (add_noise и др.) не важны для этого теста,
    # так как мы выходим из метода до их использования.
    sampler = DataOversampler()
    
    # 2. Подготавливаем тестовый DataFrame
    test_df = pd.DataFrame({
        'feature_1': [1.0, 2.5, 3.1],
        'feature_2': ['cat', 'dog', 'bird']
    })
    
    # 3. Вызываем целевой метод с original_dtypes=None
    # Это должно активировать ранний выход из функции
    result = sampler._restore_dtypes(df=test_df, original_dtypes=None)
    
    # 4. Проверки (Assertions)
    # Проверяем, что DataFrame вернулся без изменений
    pd.testing.assert_frame_equal(result, test_df)
    
    # Проверка на идентичность объекта (в данной реализации должен вернуться тот же df)
    assert result is test_df, "Метод должен возвращать исходный объект, если типы не переданы"

def test_restore_dtypes_missing_column_coverage():
    """
    Тест для покрытия строк:
    if col not in df.columns:
        continue
    """
    sampler = DataOversampler()
    
    # 1. Создаем DataFrame, в котором НЕТ колонки 'target'
    df = pd.DataFrame({
        'feature_1': [1, 2, 3]
    })
    
    # 2. Создаем Series с типами, где 'target' ПРИСУТСТВУЕТ
    # (имитируем ситуацию, когда исходные типы были замерены для большего числа колонок)
    original_dtypes = pd.Series({
        'feature_1': np.dtype('int64'),
        'target': np.dtype('int64')  # Этой колонки нет в df
    })
    
    # 3. Вызываем метод
    # Программа должна:
    # - обработать 'feature_1'
    # - увидеть, что 'target' нет в df.columns
    # - выполнить 'continue' (перейти к следующей итерации или выйти из цикла)
    result = sampler._restore_dtypes(df=df, original_dtypes=original_dtypes)
    
    # 4. Проверки
    assert 'feature_1' in result.columns
    assert 'target' not in result.columns
    assert len(result.columns) == 1
    # Убеждаемся, что метод завершился успешно и не упал с KeyError
    assert isinstance(result, pd.DataFrame)

def test_restore_dtypes_categorical_coverage():
    """
    Тест для покрытия строк:
    if isinstance(dtype, pd.CategoricalDtype):
        df[col] = df[col].astype(dtype)
    """
    sampler = DataOversampler()
    
    # 1. Определяем строгий категориальный тип с заданным порядком
    cat_type = pd.CategoricalDtype(categories=['small', 'medium', 'large'], ordered=True)
    
    # 2. Имитируем DataFrame после оверсэмплинга
    # (часто после библиотек генерации типы "слетают" в object или int)
    df = pd.DataFrame({
        'size': ['small', 'large', 'medium']  # Сейчас это тип object
    })
    
    # 3. Задаем исходные типы, где колонка 'size' была категориальной
    original_dtypes = pd.Series({
        'size': cat_type
    })
    
    # Проверяем начальное состояние (что это не категория)
    assert not isinstance(df['size'].dtype, pd.CategoricalDtype)
    
    # 4. Вызываем восстановление типов
    result = sampler._restore_dtypes(df=df, original_dtypes=original_dtypes)
    
    # 5. Проверки (Assertions)
    # Проверяем, что тип стал категориальным
    assert isinstance(result['size'].dtype, pd.CategoricalDtype)
    
    # Проверяем, что категории восстановились корректно
    assert list(result['size'].dtype.categories) == ['small', 'medium', 'large']
    assert result['size'].dtype.ordered is True
    
    # Убеждаемся, что данные не пострадали
    assert result['size'].iloc[1] == 'large'


def test_restore_dtypes_object_coverage():
    """
    Тест для покрытия строк:
    elif dtype == object:
        df[col] = df[col].astype(object)
    """
    sampler = DataOversampler()
    
    # 1. Создаем DataFrame, где колонка 'description' имеет тип, 
    # отличный от чистого object (например, имитируем результат генерации)
    # В Pandas 3.0 это может быть StringDtype, здесь просто создадим Series
    df = pd.DataFrame({
        'description': ['item1', 'item2', 'item3']
    })
    
    # 2. Указываем, что изначально тип был строго object
    original_dtypes = pd.Series({
        'description': np.dtype('O') # Это и есть тип object
    })
    
    # 3. Вызываем метод
    result = sampler._restore_dtypes(df=df, original_dtypes=original_dtypes)
    
    # 4. Проверки
    # Проверяем, что тип в итоге стал object
    assert result['description'].dtype == object
    
    # Убеждаемся, что значения сохранились
    assert result['description'].iloc[0] == 'item1'

def test_fit_resample_smotenc_via_smote_coverage():
    """
    Покрывает блок:
    if algo_local == "smote":
        if cat_features_idx:
            sampler = SMOTENC(...)
    """
    # 1. Инициализируем с алгоритмом SMOTE
    sampler = DataOversampler(algorithm='SMOTE', multiplier=1.5, random_state=42)
    
    # 2. Создаем данные с категориальной колонкой (object)
    X = pd.DataFrame({
        'numeric_feat': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'categorical_feat': ['A', 'A', 'B', 'B', 'A', 'B']  # Нечисловая колонка
    })
    y = pd.Series([0, 0, 0, 0, 1, 1])  # Дисбаланс: класс 1 всего 2 примера
    
    # 3. Вызываем ресемплирование
    # Внутри cat_features_idx станет [1], и вызовется SMOTENC
    X_res, y_res = sampler._fit_resample(X, y)
    
    # 4. Проверки
    assert X_res.shape[0] > X.shape[0]
    assert X_res.shape[1] == 2
    # Проверяем, что категориальная колонка осталась корректной (не превратилась в NaN)
    assert 'A' in X_res[:, 1] or 'B' in X_res[:, 1]

def test_fit_resample_adasyn_to_smotenc_fallback_coverage(caplog):
    """
    Тест для покрытия веток:
    elif algo_local == "adasyn":
        if cat_features_idx:
            logger.warning(...)
            sampler = SMOTENC(...)
    """
    # 1. Инициализируем оверсэмплер с алгоритмом ADASYN
    sampler = DataOversampler(
        algorithm='ADASYN', 
        multiplier=2.0, 
        random_state=42
    )
    
    # 2. Создаем данные, где есть хотя бы одна категориальная колонка (тип object/string)
    # Важно: ADASYN в оригинале упал бы на таких данных, 
    # но наш код должен переключиться на SMOTENC.
    X = pd.DataFrame({
        'feature_numeric': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6],
        'feature_cat': ['type_A', 'type_B', 'type_A', 'type_B', 'type_A', 'type_B']
    })
    
    # Создаем целевую переменную с дисбалансом (минимум 2 примера для миноритарного класса)
    y = np.array([0, 0, 0, 0, 1, 1])
    
    # 3. Вызываем метод. Используем caplog для проверки логов.
    with caplog.at_level("WARNING"):
        X_res, y_res = sampler._fit_resample(X, y)
    
    # 4. Проверки (Assertions)
    
    # Проверка покрытия лога: ищем текст сообщения в логах
    assert any("ADASYN does not support categorical columns" in record.message for record in caplog.records), \
        "Должно появиться предупреждение о переключении на SMOTENC"
    
    # Проверка результата: данные должны быть синтезированы
    assert len(y_res) > len(y), "Количество примеров должно увеличиться после оверсэмплинга"
    assert X_res.shape[1] == 2, "Количество признаков должно сохраниться"
    
    # Проверка целостности данных: категориальная колонка не должна быть пустой или сломанной
    # (проверяем наличие строк в результирующем numpy-массиве)
    unique_cats = np.unique(X_res[:, 1])
    assert 'type_A' in unique_cats or 'type_B' in unique_cats

def test_restore_dtypes_noise_numeric_pass():
    """
    Тест проверяет ветку 'pass', когда:
    1. add_noise = True
    2. Исходный тип был object
    3. Текущий тип в df является числовым
    """
    # 1. Инициализируем оверсэмплер с включенным шумом
    oversampler = DataOversampler(add_noise=True)
    # 2. Создаем исходные типы (когда-то колонка была строковой/объектом)
    original_dtypes = pd.Series({
        'feature_col': np.dtype('O')  # Тип object
    })
    # 3. Создаем DataFrame, где эта колонка СТАЛА числовой 
    # (например, в процессе генерации данных или добавления шума)
    df = pd.DataFrame({
        'feature_col': [1.1, 2.5, 3.8]  # Числовые значения (float64)
    })
    # Убеждаемся в исходном состоянии перед вызовом метода
    assert is_numeric_dtype(df['feature_col'])
    
    # 4. Вызываем метод
    result_df = oversampler._restore_dtypes(df, original_dtypes)
    # 5. Проверки:
    # Тип должен ОСТАТЬСЯ числовым (float64), так как сработал 'pass'
    # Если бы сработал 'else', тип стал бы 'object'
    assert is_numeric_dtype(result_df['feature_col'])
    assert result_df['feature_col'].dtype != object
    assert result_df['feature_col'].iloc[0] == 1.1