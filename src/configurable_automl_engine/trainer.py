"""
trainer.py
----------
Обёртка над sklearn-регрессорами: берёт данные (X, y), строит пайплайн,
обучает модель и возвращает . Содержит:
  - класс ModelTrainer
  - исключение TrainingError
  - функцию train_model (старый API, необходимый тестам).
"""

from __future__ import annotations
import logging

from pathlib import Path
from typing import Any, Callable, Dict, cast, Optional, Union
import threading

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from configurable_automl_engine.oversampling import DataOversampler
from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin

from .models import create_model, _ALIASES

from configurable_automl_engine.validation import iter_splits
from configurable_automl_engine.common.definitions import SerializationFormat
from configurable_automl_engine.common.serialization_utils import (save_artifact,
                                                                    load_artifact)
from configurable_automl_engine.training_engine.metrics import get_scorer_object

__all__ = ["ModelTrainer", "TrainingError", "train_model"]


class TrainingError(RuntimeError):
    """Ошибки, связанные с некорректностью данных или параметров."""
    pass

class IsotonicDataTransformer(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """
    Трансформер для подготовки данных под IsotonicRegression.
    Выбирает первый столбец и обрабатывает пропуски (NaN).
    """ 
    def __init__(self, feature_index: int = 0):
        self.feature_index = feature_index
    def fit(self, X: Any, y: Any = None) -> IsotonicDataTransformer:
        return self
    def transform(self, X: Any) -> np.ndarray:
        X_df = pd.DataFrame(X)
        n_samples = len(X_df)
        if self.feature_index >= X_df.shape[1]:
            raise TrainingError(
                f"feature_index {self.feature_index} out of bounds. "
                f"Dataset has only {X_df.shape[1]} columns."
                )
        # Выбираем целевую колонку по индексу
        X_col = X_df.iloc[:, self.feature_index]
        
        # Логика обработки NaN (перенесена из ModelTrainer.fit)
        if X_col.isna().all():
            result_na = pd.Series(range(n_samples)).to_numpy().reshape(-1, 1)
            return np.asarray(result_na)
        
        median_val = float(X_col.median())

        X_imputed = X_col.fillna(median_val)
            
        result_imputed = X_imputed.to_numpy().reshape(-1, 1)
        return np.asarray(result_imputed)

class ModelTrainer:
    """
    Класс-обёртка над sklearn-регрессорами: берёт набор данных (X, y),
    строит пайплайн, обучает модель и сохраняет её (при необходимости).

    Параметры:
    -----------
    algorithm: str
        Строковый ключ алгоритма (регистр не важен). Поддерживаются алиасы.
    model_params: dict[str, Any] | None
        Словарь гиперпараметров для регрессора. Если не задан, {}.
    hyperparams: dict[str, Any] | None
        Альтернативный вариант того же самого (поддерживается тестами).
    test_size: float
        Доля hold-out (по умолчанию 0.3).
    metric: str
        Имя метрики.
    random_state: int | None
        random_state для train_test_split.
    """

    def __init__(
        self,
        algorithm: str = "elasticnet",
        model_params: dict[str, Any] | None = None,
        hyperparams: dict[str, Any] | None = None,
        test_size: float = 0.3,
        metric: str = "r2",
        random_state: int | None = 42,
        data_oversampling: bool = False,
        data_oversampling_multiplier: float = 1.0,
        data_oversampling_algorithm: str = "random",
        serialization_format: SerializationFormat = SerializationFormat.pickle,
    ):
        self.logger = logging.getLogger(__name__)

        self.lock = threading.RLock()

        # Проверка алгоритма
        if not isinstance(algorithm, str):
            raise TrainingError(f"Некорректный алгоритм: {algorithm!r}")
        self.algorithm = algorithm.lower()

        # model_params vs hyperparams (для совместимости с тестами)
        if model_params is not None:
            if not isinstance(model_params, dict):
                raise TrainingError("model_params должно быть словарём")
            self.model_params = model_params
        elif hyperparams is not None:
            if not isinstance(hyperparams, dict):
                raise TrainingError("hyperparams должно быть словарём")
            self.model_params = hyperparams
        else:
            self.model_params = {}

        # Остальные параметры
        self.test_size = test_size
        self.metric = metric.lower()
        self.random_state = random_state
        self.serialization_format = serialization_format

        # ---------- oversampling ----------
        self.os_enable = data_oversampling
        self.os_multiplier = data_oversampling_multiplier
        self.os_algorithm = data_oversampling_algorithm
        if self.os_multiplier < 1:
            raise TrainingError("data_oversampling_multiplier должно быть ≥ 1")
        if self.os_algorithm not in {"random", "random_with_noise", "smote", "adasyn"}:
            raise TrainingError("Неизвестный data_oversampling_algorithm")

        # Поля, которые заполняются после fit(...)
        self.pipeline: Pipeline | None = None
        self.base_model: Any = None 
        self.val_score: float | None = None
        self._last_train_y: pd.Series | None = None 
        self._last_val_y: pd.Series | None = None 

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Remove unpicklable entries
        if 'lock' in state:
            del state['lock']
        if 'logger' in state:
            del state['logger']
        return state
    def __setstate__(self, 
                     state: dict[str, Any]
                     ) -> None:
        self.__dict__.update(state)
        # Re-initialize the lock after unpickling
        self.lock = threading.RLock()
        # Re-initialize logger if necessary
        self.logger = logging.getLogger(__name__)
    
    def _build_preprocessor(self, X_df: pd.DataFrame, algo_key: str) -> Any:
        """
        Создает объект препроцессора (Transformer) в зависимости от алгоритма 
        и типов данных в X_df.
        """
        # 1. Специальный случай: Изотоническая регрессия
        if algo_key == "isotonic_regression":
            f_idx = self.model_params.get("feature_index", 0)
            return IsotonicDataTransformer(feature_index=f_idx)
        # 2. Определение типов колонок для стандартных алгоритмов
        num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_df.select_dtypes(exclude=["number"]).columns.tolist()
        # 3. Настройка обработки числовых признаков
        num_steps: list[tuple[str, Any]] = [
            ("impute", SimpleImputer(strategy="median"))
        ]
        
        # Список алгоритмов, чувствительных к масштабу признаков
        scaling_required = {
            "sgdregressor", 
            "elasticnet", 
            "ardregression", 
            "gaussian_process_regression"
        }
        
        if algo_key in scaling_required:
            num_steps.append(("scale", StandardScaler()))
            
        num_pipeline = Pipeline(steps=num_steps)
        # 4. Сборка трансформеров
        transformers: list[tuple[str, Pipeline, list[str]]] = [
            ("num", num_pipeline, num_cols)
        ]
        # 5. Настройка обработки категориальных признаков (если они есть)
        if cat_cols:
            cat_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            transformers.append(("cat", cat_pipeline, cat_cols))
        # Возвращаем ColumnTransformer, который объединяет все ветки обработки
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def _prepare_data(self, X: Any, y: Any) -> tuple[pd.DataFrame, pd.Series]:
        """
        Выполняет валидацию типов, приведение к pandas и проверку размеров.
        Возвращает кортеж (X_df, y_s).
        """
        try:
            valid_types = (pd.DataFrame, pd.Series, np.ndarray)
            if not all(isinstance(obj, valid_types) for obj in (X, y)):
                raise TrainingError("Неподдерживаемый тип данных для X или y")
            # Приведение X к DataFrame
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
            
            # Приведение y к Series (здесь могут возникнуть ошибки размерности 2D/3D)
            if isinstance(y, pd.DataFrame):
                # Если передан DataFrame с 1 колонкой, берем её как Series
                y_s = y.iloc[:, 0]
            else:
                y_s = pd.Series(y) if not isinstance(y, pd.Series) else y
            # 2) Проверка на пустоту 
            # (выбрасывает ValueError для соответствия тесту и утилитам)
            if X_df.empty or y_s.empty:
                raise ValueError("Данные пусты")
            # 3) Проверка соответствия размеров
            n_samples = len(X_df)
            if n_samples != len(y_s):
                raise TrainingError("Число строк X и y различается")
            if n_samples < 2:
                raise TrainingError("Недостаточно записей для обучения"
                                    " (нужно минимум 2)")
            return X_df, y_s
        except ValueError as e:
            # Если это наша ошибка про пустоту — пробрасываем как есть (для теста)
            if f"{e}" == "Данные пусты":
                raise
            # Если это техническая ошибка pandas 
            # (например, "Data must be 1-dimensional")
            # оборачиваем её в TrainingError для тестов на покрытие (exception line)
            raise TrainingError(f"Ошибка при преобразовании данных: {e}")
        except (TypeError, IndexError) as e:
            # Любые другие ошибки типизации или индексов также считаем ошибками обучения
            raise TrainingError(f"Ошибка при преобразовании данных: {e}")

    def _fit_internal(
        self, 
        X_train: pd.DataFrame | np.ndarray, 
        y_train: pd.Series | np.ndarray, 
        preprocessor: Any, 
        base_model: Any
    ) -> Pipeline:
        """
        Создает и обучает итоговый пайплайн (включая препроцессор, 
        возможный оверсэмплинг и модель).
        """
        # 1) Формируем шаги пайплайна
        steps = [("preprocessor", preprocessor)]
        # 2) Добавляем оверсэмплинг, если он активирован
        if self.os_enable:
            oversampler = DataOversampler(
                algorithm=self.os_algorithm,
                multiplier=self.os_multiplier,
            )
            steps.append(("oversampler", oversampler))
        # 3) Добавляем саму модель
        steps.append(("model", base_model))
        # 4) Инициализируем пайплайн
        # Используем Pipeline из imblearn, чтобы шаги ресэмплинга работали корректно
        self.pipeline = Pipeline(steps=steps)
        # 5) Выполняем обучение
        #ВАЖНО: Сначала сохраняем пайплайн в self, чтобы он был доступен
        # даже если обучение упадет (это позволит избежать NoneType ошибок)
        try:
            self.pipeline.fit(X_train, y_train)
        except (ValueError, TypeError) as e:
            # Пробрасываем валидационные ошибки напрямую, чтобы тесты могли их поймать
            # Это критично для тестов, проверяющих некорректные гиперпараметры
            raise e
        except TrainingError:
            raise
        except Exception as e:
            # Все остальные системные ошибки оборачиваем в TrainingError
            self.logger.debug("Unexpected pipeline fit failure trace:", exc_info=True)
            self.logger.error(f"Unexpected error in pipeline fit: {e}")
            raise TrainingError(f"Internal training failure: {e}")
        return self.pipeline

    def fit(self, X: Any, y: Any) -> ModelTrainer:
        """
        Оркестрация процесса обучения: подготовка, сборка и валидация.
        """
        with self.lock:
            # Этап 1: Валидация и подготовка данных
            X_df, y_s = self._prepare_data(X, y)

            # Этап 2: Определение ключа алгоритма 
            # (перенесено выше для использования в препроцессоре)
            algo_key = _ALIASES.get(self.algorithm, self.algorithm)

            # Этап 3: Создаём препроцессор через делегирование (Task 1.2)
            preprocessor = self._build_preprocessor(X_df, algo_key)

            # Этап 4: Создаём модель через фабрику
            try:
                model_kwargs = self.model_params.copy()
                model_kwargs.pop("feature_index", None)
                base_model = create_model(self.algorithm, **model_kwargs)
            except (ValueError, ImportError) as e:
                raise TrainingError(f"Ошибка при создании модели: {e}")
            
            # Этап 5: Разбиение данных (используем iter_splits)

            try:
                X_train, X_val, y_train, y_val = next(
                    iter_splits(X_df, 
                                y_s, 
                                method='train_test_split', 
                                random_state=self.random_state)
                )
            except Exception as e:
                raise TrainingError(f"Ошибка при разбиении данных: {e}")

            # Этап 6: Сборка и обучение пайплайна (Task 2.3)
            # Здесь автоматически применится оверсэмплинг, если он включен
            self.pipeline = self._fit_internal(X_train, y_train, preprocessor, base_model)

            # Этап 7: Валидация и расчет метрик (финальный шаг)
            try:
                # Получаем объект-скорер (наш кастомный или стандартный sklearn)
                scorer = cast(Callable[..., Any], get_scorer_object(self.metric))
                # Извлекаем значение и проверяем, что оно не None
                raw_score = scorer(self.pipeline, X_val, y_val)
                if raw_score is None:
                    raise TrainingError("Scorer returned None")
                # Вычисляем значение. 
                # Scorer сам внутри сделает predict и сравнит с y_val.
                # Мы переименовываем атрибут в универсальный val_score, 
                # чтобы он подходил для любой метрики (RMSE, MAE и т.д.)
                self.val_score = float(raw_score)
            except Exception as e:
                raise TrainingError(f"Ошибка при расчете метрик на валидации: {e}")
            
            return self


    def predict(self, X: Any) -> np.ndarray:
        """
        Предсказывает целевую переменную для новых данных X.
        
        Использует единый пайплайн, что исключает расхождения в подготовке 
        данных для разных типов моделей.
        """
        with self.lock:
            # 1) Проверка: обучена ли модель
            if self.pipeline is None:
                raise TrainingError(
                    "Метод predict вызван для неообученной модели. "
                    "Сначала необходимо выполнить fit()."
                )
            try:
                # 2) Приведение входных данных к формату pandas 
                # (совместимость с трансформерами)
                X_df = (
                    pd.DataFrame(X) if not isinstance(X, (pd.DataFrame, pd.Series)) 
                    else X)
                
                # 3) Выполнение предсказания через пайплайн
                # Все кастомные шаги (IsotonicDataTransformer, ColumnTransformer) 
                # выполнятся автоматически внутри вызова .predict()
                preds = self.pipeline.predict(X_df)
                return np.asarray(preds)
                
            except Exception as e:
                raise TrainingError(f"Ошибка при выполнении предсказания: {e}")


    def save(self, path: str | Path) -> None:
        """
        Сохраняет объект ModelTrainer (пайплайн + параметры) в указанный файл,
        используя заданный формат сериализации.
        """
        with self.lock:
            if self.pipeline is None and self.base_model is None:
                raise TrainingError("Нечего сохранять: модель не обучена")
            
            path_obj = Path(path)
            # Создаем директории, если они не существуют
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            # Вызываем новую утилиту вместо pickle.dump
            save_artifact(
                obj=self, 
                path=path_obj, 
                fmt=self.serialization_format
            )

    @classmethod
    def load(cls, 
             path: str | Path, 
             fmt: SerializationFormat = SerializationFormat.pickle
             ) -> ModelTrainer:
        """
        Загружает сохранённый объект ModelTrainer из файла, используя указанный формат.
        """
        path_obj = Path(path)
        
        # Попытка загрузки через утилиту
        try:
            obj = load_artifact(path=path_obj, fmt=fmt)
        except FileNotFoundError:
            raise TrainingError(f"Файл не найден: {path}")
        except Exception as e:
            raise TrainingError(f"Ошибка при загрузке артефакта: {e}")
        # Проверка типа загруженного объекта
        if not isinstance(obj, cls):
            raise TrainingError(f"Загруженный объект не является ModelTrainer: {path}")
            
        return obj


def train_model(
    cfg_or_algo: dict[str, Any] | str,
    metric_or_testsize: str | float,
    params_or_metric: dict[str, Any] | str,
    X: Any = None,
    y: Any = None,
    enable_logging: bool = False,
    random_state: int | None = 42,
    log_path: str | Path | None = None,
) -> float:
    algo: str = ""
    metric: str = ""
    model_params: dict[str, Any] = {}
    test_size: float = 0.3
    rs: int | None = random_state
    """
    Старый API (необходим тестам):
      1) train_model(config_dict) → float (R²)
      2) train_model
      (algo, metric, params_dict, X, y, enable_logging, random_state, log_path)

    Возвращает float R² на validation.

    При ошибках бросает TrainingError, ValueError или TypeError.
    """
    # Случай «config dict»
    if isinstance(cfg_or_algo, dict):
        cfg: dict = cfg_or_algo  # type: ignore
        algo = str(cfg.get("algorithm",""))
        metric = str(cfg.get("metric",""))
        model_params = cast(Dict[str, Any], cfg.get("model_params", {}))
        test_size = float(cfg.get("test_size", 0.3))
        rs = cast(Optional[int], cfg.get("random_state", 42))
        enable_logging = bool(cfg.get("enable_logging", False))
        log_path = cast(Optional[Union[str, Path]], cfg.get("log_path", None))
        data_os = bool(cfg.get("data_oversampling", False))
        data_os_mult = float(cfg.get("data_oversampling_multiplier", 1.0))
        data_os_alg = str(cfg.get("data_oversampling_algorithm", "random"))
    else:
        # Простой API
        # Если пришел None, мы НЕ превращаем его в строку "None" сразу,
        # чтобы сохранить логику оригинальной валидации для тестов.
        algo = cfg_or_algo if cfg_or_algo is not None else "" 
        metric = str(metric_or_testsize)  
        model_params = cast(Dict[str, Any], params_or_metric) 
        test_size = 0.3
        rs = random_state
        data_os = False
        data_os_mult = 1.0
        data_os_alg = "random"

    # Проверка алгоритма
    if not isinstance(cfg_or_algo, (str, dict)):
        raise TrainingError("Неверный алгоритм")
    algo_key = algo.lower()

    # Проверка model_params
    if not isinstance(model_params, dict) or len(model_params) == 0:
        raise TrainingError("Параметры модели не заданы")

    # Проверка типов X и y
    if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)) or not isinstance(
        y, (pd.Series, pd.DataFrame, np.ndarray)
    ):
        raise TrainingError("Неподдерживаемый тип данных для X или y")

    # Приведение к pandas
    try:
        X_df = (
            pd.DataFrame(X)
            if not isinstance(X, (pd.DataFrame, pd.Series))
            else X  
        )
        if isinstance(y, pd.DataFrame):
            y_s = pd.Series(y.iloc[:, 0])  
        else:
            y_s = pd.Series(y) if not isinstance(y, pd.Series) else y  
    except Exception as e:
        raise TrainingError(f"Ошибка при преобразовании данных: {e}")

    # Проверка размера выборки
    n_samples = len(X_df)
    if n_samples == 0 or len(y_s) == 0:
        raise TrainingError("Данные пусты")
    if n_samples != len(y_s):
        raise TrainingError("Число строк X и y различается")
    if n_samples < 2:
        raise TrainingError("Недостаточно записей для обучения")

    # Создаём и обучаем ModelTrainer
    try:
        trainer = ModelTrainer(
            algorithm=algo_key,
            model_params=model_params,
            test_size=test_size,
            metric=metric,
            random_state=rs,
            data_oversampling=data_os,
            data_oversampling_multiplier=data_os_mult,
            data_oversampling_algorithm=data_os_alg,
        )
        trainer.fit(X_df, y_s)
        val_score = trainer.val_score
        if val_score is None:
            raise TrainingError("Модель не вернула значение метрики")
    except TrainingError:
        raise
    except ValueError:
        raise
    except TypeError:
        raise

    # Логирование (если enable_logging=True)
    if enable_logging:
        # Используем централизованную настройку. 
        # Она должна уметь проверять, настроен ли уже логгер.
        
        # Теперь просто получаем логгер и пишем сообщение
        logger = logging.getLogger(__name__)
        logger.info(
            f"Training finished: Algorithm={algo_key}, "
            f"Metric={metric.upper()}, Score={val_score:.4f}"
        )

    return float(val_score)
