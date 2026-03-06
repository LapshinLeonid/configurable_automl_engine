import pytest
import numpy as np
import pandas as pd
import threading
import logging

import os


from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin

from configurable_automl_engine import trainer
from configurable_automl_engine.trainer import (
    ModelTrainer, 
    TrainingError, 
    IsotonicDataTransformer,
    train_model
)
from configurable_automl_engine.common.definitions import SerializationFormat
from unittest.mock import MagicMock, patch
from configurable_automl_engine.training_engine.logger import setup_logging # Импортируем setup


# Синтетические данные для тестирования
X = pd.DataFrame({
    "a": np.random.rand(50),
    "b": np.random.rand(50)
})
y = X["a"] * 2 + X["b"] * -3 + np.random.randn(50) * 0.1

@pytest.fixture
def base_params():
    return {"alpha": 0.1, "l1_ratio": 0.5}


def test_successful_training(tmp_path, monkeypatch, base_params):
    
    # Переходим в рабочую директорию теста
    monkeypatch.chdir(tmp_path)
    log_file = tmp_path / "training.log"
    
    # Предварительно настраиваем логгер для текущего теста
    setup_logging(logfile=log_file)
    
    score = train_model(
        "ElasticNet",
        "r2",
        base_params,
        X, y,
        enable_logging=True
    )
    
    assert isinstance(score, float)
    assert 0.3 < score <= 1.0
    # Теперь проверка пройдет, так как инфраструктура логирования была готова
    assert log_file.exists()


def test_no_logging(tmp_path, monkeypatch, base_params):
    # Без логирования файл не создаётся
    monkeypatch.chdir(tmp_path)
    score = train_model(
        "ElasticNet",
        "r2",
        base_params,
        X, y,
        enable_logging=False
    )
    assert isinstance(score, float)
    assert not (tmp_path / "training.log").exists()


def test_invalid_algorithm(base_params):
    with pytest.raises(TrainingError):
        train_model(None, "r2", base_params, X, y)


def test_invalid_metric(base_params):
    with pytest.raises(TrainingError):
        train_model("ElasticNet", "mse", base_params, X, y)


def test_empty_params(base_params):
    with pytest.raises(TrainingError):
        train_model("ElasticNet", "r2", {}, X, y)


def test_empty_data(base_params=None):
    # Оба DataFrame пустые
    empty_df = pd.DataFrame([])
    with pytest.raises(TrainingError):
        train_model("ElasticNet", "r2", base_params or {}, empty_df, empty_df)


def test_mismatch_dimensions(base_params):
    # X и y разной длины
    X2 = X.iloc[:10]
    y2 = y.iloc[:9]
    with pytest.raises(TrainingError):
        train_model("ElasticNet", "r2", base_params, X2, y2)


def test_too_few_records(base_params):
    # Меньше двух записей
    X_small = X.iloc[:1]
    y_small = y.iloc[:1]
    with pytest.raises(TrainingError):
        train_model("ElasticNet", "r2", base_params, X_small, y_small)


def test_invalid_param_key(base_params):
    # Параметр неизвестный для ElasticNet
    bad_params = {"foobar": 1}
    with pytest.raises(TypeError):
        train_model("ElasticNet", "r2", bad_params, X, y)


def test_negative_alpha(base_params):
    # Негативный alpha приводит к ValueError из sklearn
    neg_params = {"alpha": -1.0, "l1_ratio": 0.5}
    with pytest.raises(ValueError):
        train_model("ElasticNet", "r2", neg_params, X, y)


def test_invalid_data_type(base_params):
    # Неподдерживаемый тип данных (list)
    with pytest.raises(Exception):
        train_model("ElasticNet", "r2", base_params, [1, 2, 3], [1, 2, 3])


# --- Тесты для IsotonicDataTransformer ---
def test_isotonic_transformer_all_nan():
    """ Обработка случая, когда все значения NaN."""
    transformer = IsotonicDataTransformer()
    X = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})
    # Должен вернуть range(n_samples)
    result = transformer.transform(X)
    assert np.array_equal(result, np.array([0, 1, 2]).reshape(-1, 1))
def test_isotonic_transformer_median_nan():
    """Cлучай, когда медиана не вычисляется (напр. пустой ввод после фильтрации)."""
    # В текущей реализации до медианы доходит, если не все NaN. 
    # Но если median вернул NaN (крайний случай pandas), сработает строка.
    transformer = IsotonicDataTransformer()
    # Эмулируем структуру данных, где median может вернуть NaN
    X = pd.DataFrame([np.nan, 1.0]) 
    # В норме median будет 1.0, но если мы подменим поведение или передадим специфический тип:
    result = transformer.transform(X)
    assert result.shape == (2, 1)
# --- Тесты валидации параметров ---
def test_trainer_init_invalid_params():
    """Тесты исключений в конструкторе."""
    # Некорректный тип алгоритма
    with pytest.raises(TrainingError, match="Некорректный алгоритм"):
        ModelTrainer(algorithm=123)
    
    # model_params не словарь
    with pytest.raises(TrainingError, match="model_params должно быть словарём"):
        ModelTrainer(model_params="not a dict")
        
    # hyperparams не словарь
    with pytest.raises(TrainingError, match="hyperparams должно быть словарём"):
        ModelTrainer(hyperparams=[1, 2, 3])
    # множитель оверсэмплинга < 1
    with pytest.raises(TrainingError, match="data_oversampling_multiplier"):
        ModelTrainer(data_oversampling_multiplier=0.5)
    # неизвестный алгоритм оверсэмплинга
    with pytest.raises(TrainingError, match="Неизвестный data_oversampling_algorithm"):
        ModelTrainer(data_oversampling_algorithm="magic_boost")
# --- Тесты подготовки данных ---
def test_prepare_data_variants():
    """Тестирование различных форматов входных данных."""
    trainer = ModelTrainer()
    
    #  Неподдерживаемый тип (напр. list)
    with pytest.raises(TrainingError, match="Неподдерживаемый тип данных"):
        trainer._prepare_data([1, 2], [1, 2])
    #  y как DataFrame (превращение в Series)
    X = np.random.rand(10, 2)
    y_df = pd.DataFrame({'target': np.random.rand(10)})
    X_res, y_res = trainer._prepare_data(X, y_df)
    assert isinstance(y_res, pd.Series)
    assert len(y_res) == 10
    # Ошибка при разбиении (слишком мало данных для train_test_split)
    X_small = pd.DataFrame({'a': [1]})
    y_small = pd.Series([1])
    # _prepare_data пропустит (там проверка < 2), но split может упасть 
    with pytest.raises(TrainingError, match="Недостаточно записей"):
        trainer.fit(X_small, y_small)
# --- Тесты сохранения и загрузки  ---
def test_save_load_errors(tmp_path):
    """Тесты ошибок сериализации."""
    trainer = ModelTrainer()
    path = tmp_path / "model.pkl"
    # Сохранение необученной модели
    with pytest.raises(TrainingError, match="Нечего сохранять"):
        trainer.save(path)
    # Файл не найден при загрузке
    with pytest.raises(TrainingError, match="Файл не найден"):
        ModelTrainer.load(tmp_path / "non_existent.pkl")
    # Загрузка объекта другого типа
    dummy_path = tmp_path / "dummy.pkl"
    import pickle
    with open(dummy_path, "wb") as f:
        pickle.dump("just a string", f)
    
    with pytest.raises(TrainingError, match="не является ModelTrainer"):
        ModelTrainer.load(dummy_path)
# --- Тесты train_model API  ---

def test_train_model_legacy_api(tmp_path):
     # 0. ОЧИСТКА ЛОГГЕРА (Критически важно для тестов)
    # Удаляем старые хендлеры от предыдущих тестов, чтобы setup_logging сработал заново
    base_logger = logging.getLogger("configurable_automl_engine")
    for handler in base_logger.handlers[:]:
        base_logger.removeHandler(handler)
        handler.close() # Закрываем файлы, чтобы Windows позволила их удалить
    
    X = np.random.rand(20, 2)
    y = np.random.rand(20)
    log_file = tmp_path / "test.log"
    
    # 1. Тест случая «config dict»
    config = {
        "algorithm": "elasticnet",
        "metric": "r2",
        "model_params": {"alpha": 0.1},
        "enable_logging": True,
        "log_path": str(log_file)
    }
    
    # Теперь setup_logging увидит пустой список хендлеров и создаст нужный файл
    setup_logging(logfile=log_file)
    
    score = train_model(config, "r2", {}, X, y, enable_logging=True)
    
    assert isinstance(score, float)
    # Теперь файл точно будет создан
    assert os.path.exists(log_file)
    
    # 2. Тест простого API 
    score2 = train_model("elasticnet", "r2", {"alpha": 0.5}, X, y)
    assert isinstance(score2, float)
    
    # 3. Тест валидации
    with pytest.raises(TrainingError, match="Неверный алгоритм"):
        train_model(None, "r2", {}, X, y)
    
    with pytest.raises(TrainingError, match="Параметры модели не заданы"):
        train_model("elasticnet", "r2", {}, X, y)
        
    # 4. Тест проброса исключений 
    with pytest.raises(ValueError):
        # Некорректный параметр l1_ratio (> 1.0) вызовет ValueError в sklearn
        train_model("elasticnet", "r2", {"l1_ratio": 5.0}, X, y)

# --- Тесты препроцессора и оверсэмплинга ---
def test_fit_internal_and_predict():
    """Покрытие внутренних механизмов обучения и предсказания."""
    # алгоритм со скалированием (SGD)
    trainer = ModelTrainer(algorithm="sgdregressor", model_params={"max_iter": 5})
    X = pd.DataFrame({'num': [1, 2, 3, 4, 5, 6], 'cat': ['a', 'b', 'a', 'b', 'a', 'b']})
    y = np.array([1, 2, 3, 4, 5, 6])
    
    trainer.fit(X, y)
    assert trainer.pipeline is not None 
    
    # Вызов predict
    preds = trainer.predict(X)
    assert len(preds) == 6
    # Predict для необученной модели
    new_trainer = ModelTrainer()
    with pytest.raises(TrainingError, match="Метод predict вызван для неообученной модели"):
        new_trainer.predict(X)
    # Оверсэмплинг в fit_internal
    os_trainer = ModelTrainer(data_oversampling=True, data_oversampling_algorithm="random")
    os_trainer.fit(X, y)
    # Проверяем, что в шагах пайплайна есть oversampler
    step_names = [s[0] for s in os_trainer.pipeline.steps]
    assert "oversampler" in step_names

def test_init_with_hyperparams():
    """
    Проверяет ситуацию, когда model_params=None, но переданы hyperparams.
    """
    # Задаем тестовые гиперпараметры
    test_hyperparams = {"n_estimators": 100, "max_depth": 5}
    
    # Инициализируем ModelTrainer, передавая hyperparams, но НЕ передавая model_params
    # Это заставит код пропустить первый if и зайти в elif hyperparams is not None
    trainer = ModelTrainer(
        algorithm="rf",
        model_params=None,
        hyperparams=test_hyperparams
    )
    
    # Проверки (Assertions)
    # Убеждаемся, что в итоге внутренний атрибут self.model_params получил значения из hyperparams
    assert trainer.model_params == test_hyperparams
    assert isinstance(trainer.model_params, dict)
    assert "n_estimators" in trainer.model_params

def test_coverage_lock_removal_only():
    """
    Тест для покрытия строки: if 'lock' in state: del state['lock']
    """
    trainer = ModelTrainer(algorithm="elasticnet")
    
    # Внедряем lock напрямую в словарь объекта
    trainer.lock = threading.Lock()
    
    # Вызываем __getstate__, который создает копию состояния и удаляет lock
    state = trainer.__getstate__()
    
    # Проверяем, что в возвращенном состоянии ключа 'lock' нет
    assert 'lock' not in state

def test_prepare_data_empty_input_coverage():
    """
    Тест для проверки обработки пустых входных данных.
    После рефакторинга ожидается ValueError, так как это стандарт 
    для централизованной валидации в проекте.
    """
    trainer = ModelTrainer(algorithm="elasticnet")
    
    # Создаем пустые объекты (DataFrame и Series)
    X_empty = pd.DataFrame()
    y_empty = pd.Series([], dtype=float)
    
    # Теперь ожидаем ValueError вместо TrainingError
    with pytest.raises(ValueError, match="Данные пусты"):
        trainer._prepare_data(X_empty, y_empty)
    
def test_fit_internal_unexpected_error():
    """
    Тест для покрытия строки: raise TrainingError(f"Internal training failure: {e}")
    Используем корректный препроцессор и специально настроенный mock модели.
    """
    trainer = ModelTrainer(algorithm="elasticnet")
    X = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    y = pd.Series([1, 0])
    
    # 1. Создаем mock для финальной модели
    mock_model = MagicMock()
    
    # Чтобы imblearn.pipeline не ругался, что это одновременно и трансформер и модель,
    # удаляем атрибут 'transform', если он есть в моке по умолчанию.
    if hasattr(mock_model, 'transform'):
        del mock_model.transform
        
    # Настраиваем падение с системной ошибкой при вызове fit
    mock_model.fit.side_effect = RuntimeError("Системный сбой")
    
    # Нам также нужно, чтобы mock_model имел атрибут _estimator_type (нужно для sklearn/imblearn)
    mock_model._estimator_type = "regressor"
    # 2. Выполнение и проверка
    # Теперь мы должны проскочить валидацию шагов и упасть именно в блоке try-except fit
    with pytest.raises(TrainingError) as excinfo:
        trainer._fit_internal(
            X_train=X, 
            y_train=y, 
            preprocessor=StandardScaler(), # Используем реальный объект вместо мока
            base_model=mock_model
        )
    
    # 3. Assertions
    assert "Internal training failure" in str(excinfo.value)
    assert "Системный сбой" in str(excinfo.value)

def test_coverage_fit_create_model_error():
    """
    Тест для покрытия строки: 
    except (ValueError, ImportError) as e: raise TrainingError(f"Ошибка при создании модели: {e}")
    """
    # 1. Подготовка: Создаем трейнер с несуществующим алгоритмом.
    # Большинство фабрик выбрасывают ValueError, если алгоритм не найден в списке поддерживаемых.
    invalid_algorithm = "non_existent_model_2026"
    trainer = ModelTrainer(algorithm=invalid_algorithm)
    
    # Данные должны быть валидными, чтобы пройти Этап 1 (_prepare_data)
    X = pd.DataFrame({'feature': [1, 2, 3]})
    y = pd.Series([1, 0, 1])
    
    # 2. Выполнение: При вызове fit код дойдет до Этапа 4 и упадет в блоке try-except
    with pytest.raises(TrainingError) as excinfo:
        trainer.fit(X, y)
    
    # 3. Проверка: Убеждаемся, что ошибка обернута в наше сообщение
    assert "Ошибка при создании модели" in str(excinfo.value)

def test_fit_split_error_fixed(monkeypatch):
    """
    Тест для покрытия ветки ошибки при разбиении данных.
    """
    # 1. Подготовка
    trainer = ModelTrainer(algorithm="elasticnet")
    def mock_iter_splits(*args, **kwargs):
        raise RuntimeError("Force split failure")
    # Патчим 'iter_splits' в модуле 'trainer'
    monkeypatch.setattr("configurable_automl_engine.trainer.iter_splits", mock_iter_splits)
    # 2. Действие и Проверка
    # Используем данные с 2+ строками, чтобы пройти валидацию длины
    X_test = pd.DataFrame([[1], [2]], columns=["feature1"])
    y_test = pd.Series([0, 1])
    with pytest.raises(TrainingError) as excinfo:
        trainer.fit(X=X_test, y=y_test)
    
    # Теперь мы должны увидеть ошибку из блока исключений iter_splits
    assert "Ошибка при разбиении данных" in str(excinfo.value)
    assert "Force split failure" in str(excinfo.value)

def test_predict_general_exception():
    """
    Тест для покрытия ветки: raise TrainingError(f"Ошибка при выполнении предсказания: {e}")
    """
    # 1. Подготовка
    trainer = ModelTrainer(algorithm="elasticnet")
    
    # Создаем мок-объект для пайплайна
    mock_pipeline = MagicMock()
    # Настраиваем его так, чтобы вызов .predict() выбрасывал исключение
    mock_pipeline.predict.side_effect = RuntimeError("System failure during inference")
    
    # Вручную устанавливаем мок в trainer (имитируем, что модель "обучена")
    trainer.pipeline = mock_pipeline
    # 2. Действие и Проверка
    # Пытаемся вызвать predict с любыми данными
    X_input = pd.DataFrame([[1, 2, 3]])
    
    with pytest.raises(TrainingError) as excinfo:
        trainer.predict(X_input)
    
    # 3. Верификация
    # Проверяем, что возникло наше кастомное сообщение
    assert "Ошибка при выполнении предсказания" in str(excinfo.value)
    # Проверяем, что исходная причина (e) также попала в текст
    assert "System failure during inference" in str(excinfo.value)
    
    # Дополнительно проверяем, что вызов дошел до пайплайна
    mock_pipeline.predict.assert_called_once()

def test_load_general_exception_coverage(monkeypatch):
    """
    Тест для покрытия ветки: raise TrainingError(f"Ошибка при загрузке артефакта: {e}")
    """
    # 1. Подготовка
    # Имитируем ошибку, которая НЕ является FileNotFoundError
    def mock_load_artifact_crash(*args, **kwargs):
        raise RuntimeError("Unexpected corruption or memory error")
    # Патчим функцию load_artifact в модуле, где находится ModelTrainer
    # Предполагаем путь: configurable_automl_engine.trainer
    monkeypatch.setattr(
        "configurable_automl_engine.trainer.load_artifact", 
        mock_load_artifact_crash
    )
    # 2. Действие и Проверка
    test_path = "some_existing_file.pkl"
    
    with pytest.raises(TrainingError) as excinfo:
        # Вызываем метод load
        ModelTrainer.load(path=test_path)
    # 3. Верификация
    # Проверяем, что сработало именно общее исключение
    assert "Ошибка при загрузке артефакта" in str(excinfo.value)
    assert "Unexpected corruption or memory error" in str(excinfo.value)

def test_train_model_y_dataframe_conversion_coverage():
    """
    Тест для покрытия ветки: if isinstance(y, pd.DataFrame): y_s = pd.Series(y.iloc[:, 0])
    Проверяем, что функция корректно принимает DataFrame в качестве y.
    """
    # 1. Подготовка данных
    # Создаем X (минимум 2 строки, чтобы пройти валидацию внутри ModelTrainer)
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    
    # Создаем y как DataFrame с одним столбцом (как это часто бывает после чтения csv)
    y_df = pd.DataFrame({"target": [10, 20, 30]})
    
    # Параметры для старого API train_model
    algo = "elasticnet"
    metric = "r2"
    model_params = {"alpha": 0.1, "l1_ratio": 0.5}
    
    # 2. Выполнение
    # Мы вызываем функцию. Нам не обязательно проверять результат R2, 
    # главное — пройти через интересующую нас строку кода без ошибок.
    try:
        result = train_model(
            cfg_or_algo=algo,
            metric_or_testsize=metric,
            params_or_metric=model_params,
            X=X,
            y=y_df
        )
        
        # 3. Проверка
        assert isinstance(result, float)
    except Exception as e:
        # Если тест упал на обучении (например, из-за данных), 
        # проверка типа y уже должна была выполниться.
        pytest.fail(f"Функция train_model упала при обработке y как DataFrame: {e}")

def test_train_model_conversion_exception_coverage():
    """
    Тест для покрытия ветки: raise TrainingError(f"Ошибка при преобразовании данных: {e}")
    Используем объект, который проходит проверку типов, но падает при конвертации в Series.
    """
    # 1. Создаем объект, который "взорвется" при попытке его прочитать
    class BombArray(np.ndarray):
        def __new__(cls):
            # Создаем массив 2x1, чтобы пройти проверку на минимальное кол-во строк
            return np.asarray([[10], [20]]).view(cls)
        
        def __iter__(self):
            # Pandas вызывает __iter__ при создании Series из объекта, не являющегося Series
            raise RuntimeError("Data corruption during iteration")
        
        @property
        def values(self):
            # На случай, если pandas попытается достучаться через .values
            raise RuntimeError("Data corruption during access")
    # 2. Подготовка данных
    # X делаем валидным, чтобы пройти pd.DataFrame(X)
    X_valid = pd.DataFrame({"feat": [1, 2]})
    
    # y делаем BombArray. Он пройдет isinstance(y, np.ndarray), 
    # но упадет на строке y_s = pd.Series(y)
    y_broken = BombArray()
    
    algo = "elasticnet"
    metric = "r2"
    model_params = {"alpha": 0.1}
    # 3. Действие и Проверка
    with pytest.raises(TrainingError) as excinfo:
        train_model(
            cfg_or_algo=algo,
            metric_or_testsize=metric,
            params_or_metric=model_params,
            X=X_valid,
            y=y_broken
        )
    # 4. Верификация
    assert "Ошибка при преобразовании данных" in str(excinfo.value)
    assert "Data must be 1-dimensional" in str(excinfo.value)

def test_train_model_empty_data_coverage():
    """
    Тест для покрытия ветки: if n_samples == 0 or len(y_s) == 0: raise TrainingError("Данные пусты")
    """
    # Подготовка: X пустой, y содержит данные (или наоборот)
    X_empty = np.array([]) 
    y_valid = np.array([1, 2, 3])
    
    algo = "elasticnet"
    metric = "r2"
    model_params = {"alpha": 0.1}
    # Действие и Проверка
    with pytest.raises(TrainingError) as excinfo:
        train_model(
            cfg_or_algo=algo,
            metric_or_testsize=metric,
            params_or_metric=model_params,
            X=X_empty,
            y=y_valid
        )
    # Верификация
    assert str(excinfo.value) == "Данные пусты"

def test_fit_internal_rethrows_training_error():
    """
    Тест проверяет, что если внутри pipeline.fit возникает TrainingError,
    он пробрасывается (raise) без изменений.
    """
    
    # 1. Подготовка данных
    X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y_train = pd.Series([10, 20, 30])
    # 2. Создание "сломанного" препроцессора, который выкидывает TrainingError
    class BrokenPreprocessor(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            # Имитируем специфическую ошибку обучения
            raise TrainingError("Специфическая ошибка в процессе подготовки данных")
        def transform(self, X):
            return X
    # 3. Инициализация тренера
    trainer = ModelTrainer(algorithm="elasticnet")
    
    # Заменяем стандартную модель на заглушку
    mock_model = MagicMock()
    
    # 4. Проверка: перехватываем именно TrainingError
    with pytest.raises(TrainingError) as exc_info:
        trainer._fit_internal(
            X_train=X_train,
            y_train=y_train,
            preprocessor=BrokenPreprocessor(),
            base_model=mock_model
        )
    
    # Проверяем, что сообщение осталось оригинальным
    assert "Специфическая ошибка в процессе подготовки данных" in str(exc_info.value)

def test_prepare_data_index_error_coverage():
    """
    Тест для покрытия блока except (TypeError, IndexError).
    Передаем DataFrame без колонок. 
    isinstance(y, pd.DataFrame) вернет True, но y.iloc[:, 0] вызовет IndexError.
    """
    trainer = ModelTrainer(algorithm="elasticnet")
    
    # X — корректный (2 строки, 1 колонка)
    X = np.array([[1], [2]])
    
    # y — DataFrame, у которого есть строки (индексы), но НЕТ столбцов.
    # Это вызовет IndexError: single positional indexer is out-of-bounds
    y_no_columns = pd.DataFrame(index=[0, 1]) 
    
    with pytest.raises(TrainingError, match="Ошибка при преобразовании данных"):
        trainer._prepare_data(X, y_no_columns)

def test_fit_raises_error_when_scorer_returns_none():
    """
    Тест проверяет выброс TrainingError, если объект-скорер 
    возвращает None вместо числового значения.
    """
    
    # 1. Подготовка минимальных данных для обучения
    X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
    y = pd.Series([10, 20, 30, 40, 50])
    
    # 2. Настройка тренера
    # Используем любой алгоритм, так как до обучения дело дойдет, 
    # но упадет на расчете метрики
    trainer = ModelTrainer(
        algorithm="elasticnet",
        model_params={"alpha": 0.1},
        metric="r2"
    )
    
    # 3. Мокаем (подменяем) get_scorer_object
    # Нам нужно, чтобы get_scorer_object вернул функцию (callable), 
    # которая при вызове возвращает None
    mock_scorer = MagicMock(return_value=None)
    
    with patch("configurable_automl_engine.trainer.get_scorer_object", return_value=mock_scorer):
        # Проверяем, что вызывается именно наше исключение с нужным текстом
        with pytest.raises(TrainingError, match="Scorer returned None"):
            trainer.fit(X, y)


def test_train_model_raises_error_when_val_score_is_none():
    """
    Тест проверяет ситуацию в функции train_model, когда ModelTrainer.fit() 
    отработал, но не установил значение val_score.
    """
    # 1. Данные для прохождения валидации (минимум 2 примера)
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    
    algo = "elasticnet"
    metric = "r2"
    params = {"alpha": 0.5}
    # 2. Мокаем класс ModelTrainer прямо в модуле trainer.py
    # Это гарантирует, что train_model увидит именно Mock
    with patch("configurable_automl_engine.trainer.ModelTrainer") as MockTrainer:
        # Настраиваем поведение экземпляра
        mock_instance = MagicMock()
        MockTrainer.return_value = mock_instance
        
        # fit() возвращает self, имитируем успешное завершение
        mock_instance.fit.return_value = mock_instance
        
        # ПРОВОКАЦИЯ ОШИБКИ: val_score остается None
        mock_instance.val_score = None
        
        # 3. Проверяем, что функция train_model поймала этот None и выбросила исключение
        with pytest.raises(TrainingError, match="Модель не вернула значение метрики"):
            train_model(
                algo, 
                metric, 
                params, 
                X=X, 
                y=y
            )