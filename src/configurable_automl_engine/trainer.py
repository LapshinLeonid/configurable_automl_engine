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

from configurable_automl_engine.training_engine.thread_pool import SharedDataFrame


from .models import create_model, _ALIASES

from configurable_automl_engine.validation import iter_splits
from configurable_automl_engine.common.definitions import SerializationFormat
from configurable_automl_engine.common.serialization_utils import (save_artifact,
                                                                    load_artifact)
from configurable_automl_engine.training_engine.metrics import (
    get_scorer_object, 
    is_greater_better
)

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
    
    def _get_dimensions(self, X):
        """Вспомогательный метод для получения строк и колонок."""
        if hasattr(X, 'shape'):
            return X.shape[0], (X.shape[1] if len(X.shape) > 1 else 1)
        n_rows = len(X)
        n_cols = len(X[0]) if n_rows > 0 and isinstance(X[0], (list, np.ndarray)) else 1
        return n_rows, n_cols
    def fit(self, X: Any, y: Any = None) -> IsotonicDataTransformer:
        
        _, n_cols = self._get_dimensions(X)
        idx = self.feature_index if self.feature_index < n_cols else 0
        
        if isinstance(X, pd.DataFrame):
            X_col = X.iloc[:, idx]
        elif isinstance(X, np.ndarray):
            X_col = X[:, idx] if len(X.shape) > 1 else X
        else:
            X_col = np.asarray(X)[:, idx]
            
        s_col = pd.Series(X_col)
        self.median_ = float(s_col.median()) if not s_col.isna().all() else 0.0
        return self
    
    def transform(self, X):
        try:
            # 1. Получаем метаданные без принудительного копирования в DataFrame
            # Используем логику из _extract_metadata, которую мы обсуждали ранее
            n_rows, n_cols = self._get_dimensions(X)
            
            # 2. Валидация индекса признака
            if self.feature_index >= n_cols:
                raise TrainingError(
                    f"Ошибка при преобразовании данных: feature_index {self.feature_index} "
                    f"out of bounds. Dataset has only {n_cols} columns."
                )
            # 3. Извлечение колонки (Zero-copy для numpy и pandas)
            # Если X - DataFrame, используем iloc. Если numpy - обычный слайсинг.
            if isinstance(X, pd.DataFrame):
                X_col = X.iloc[:, self.feature_index]
            elif isinstance(X, np.ndarray):
                X_col = X[:, self.feature_index]
            else:
                # Для списков или других типов приводим к numpy (минимальная копия)
                X_col = np.asarray(X)[:, self.feature_index]
            # 4. Логика обработки NaN
            # Проверяем, являются ли все значения в колонке NaN
            # Используем pd.Series для удобства вычисления медианы, если это еще не Series
            s_col = X_col if isinstance(X_col, pd.Series) else pd.Series(X_col)
            
            if s_col.isna().all():
                # Если все NaN, возвращаем индексы строк (согласно старой логике)
                return np.arange(n_rows).reshape(-1, 1).astype(float)
            # Вычисляем медиану и заполняем пропуски
            fill_value = getattr(self, "median_", 0.0)
            X_imputed = s_col.fillna(fill_value)
            # Возвращаем результат в виде 2D массива, как ожидает scikit-learn
            return X_imputed.to_numpy().reshape(-1, 1)
        except Exception as e:
            if isinstance(e, TrainingError):
                raise e
            # Унификация сообщения об ошибке согласно тестам (#A8)
            raise TrainingError(f"Ошибка при преобразовании данных: {e}")
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
        categorical_features: list[str] | None = None,  
        numerical_features: list[str] | None = None,  
        id_column: str | None = None
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
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.id_column = id_column

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

        for attr_name, features in [("categorical_features", categorical_features), 
                                ("numerical_features", numerical_features)]:
            if features is not None:
                if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
                    raise TrainingError(f"Параметр {attr_name} должен быть списком строк (имен колонок)")

    def _validate_features(self, X: pd.DataFrame) -> None:
        """Проверяет наличие всех указанных колонок в датафрейме."""
        specified_features = (self.categorical_features or []) + (self.numerical_features or [])
        missing = [col for col in specified_features if col not in X.columns]
        if missing:
            raise TrainingError(f"Указанные колонки не найдены в данных: {missing}")
    def _detect_feature_types(self, X: pd.DataFrame, target_column: str) -> None:
        """
        Автоматически разделяет признаки на типы, исключая целевую переменную и ID.
        Вызывается, если categorical_features или numerical_features не заданы.
        """
        with self.lock:
            # Если оба списка уже заполнены пользователем — просто валидируем
            if self.categorical_features is not None and self.numerical_features is not None:
                self._validate_features(X)
                return
            # Определяем список колонок для анализа (исключаем таргет и ID)
            exclude = {target_column}
            if self.id_column:
                exclude.add(self.id_column)
            
            df_to_analyze = X.drop(columns=list(exclude & set(X.columns)))
            # Авто-определение категориальных (object, category, bool)
            if self.categorical_features is None:
                self.categorical_features = df_to_analyze.select_dtypes(
                    include=['object', 'category', 'bool']
                ).columns.tolist()
            # Авто-определение числовых (все оставшиеся типы 'number')
            if self.numerical_features is None:
                self.numerical_features = df_to_analyze.select_dtypes(
                    include=['number']
                ).columns.tolist()
            self.logger.info(
                f"Auto-detected features: {len(self.categorical_features)} cat, "
                f"{len(self.numerical_features)} num."
            )

    def _extract_metadata(self, X):
        """Извлекает имена колонок без копирования данных."""
        if hasattr(X, 'get_data_info'):
            return X.get_data_info()['columns']
        if isinstance(X, pd.DataFrame):
            return X.columns.tolist()
        if hasattr(X, 'shape') and len(X.shape) > 1:
            return [f"col_{i}" for i in range(X.shape[1])]
        return None
    
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
    
    def _build_preprocessor(self, feature_names):
        # 1. Получаем индексы колонок на основе подготовленных списков
        # Используем feature_names для сопоставления имен с порядковыми номерами
        cat_indices = [
            feature_names.index(col) 
            for col in self.categorical_features 
            if col in feature_names
        ]
        num_indices = [
            feature_names.index(col) 
            for col in self.numerical_features 
            if col in feature_names
        ]

        if not cat_indices and not num_indices:
            self.logger.warning("No features matched for preprocessing. Defaulting to passthrough.")
            return ColumnTransformer(
                [('bypass', 'passthrough', slice(None))],
                remainder='drop'
            )
        
        # 2. Пайплайны трансформации
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        # 3. Сборка ColumnTransformer
        transformers = []
        if cat_indices:
            transformers.append(('cat', cat_transformer, cat_indices))
        if num_indices:
            transformers.append(('num', num_transformer, num_indices))
        return ColumnTransformer(
            transformers=transformers if transformers else [('pass', 'passthrough', slice(None))],
            remainder='drop' # Изменили с passthrough на drop для безопасности (ID и прочее)
        )

    def _prepare_data(self, X: Any, y: Any) -> tuple[Any, Any]:
        """
        Выполняет валидацию типов, приведение к pandas и проверку размеров.
        Возвращает кортеж (X_df, y_s).
        """
        try:
            self.feature_names = self._extract_metadata(X) 

            self._detected_dtypes = X.dtypes if isinstance(X, pd.DataFrame) else None
            # 1. Проверка типов
            valid_types = (pd.DataFrame, pd.Series, np.ndarray)
            if not isinstance(X, valid_types) and not isinstance(X, SharedDataFrame):
                raise TrainingError(f"Неподдерживаемый тип данных для X: {type(X)}")
            
            # 2. Zero-Copy Logic: Оставляем массивы как есть, если это возможно
            # Если это SharedDataFrame, извлекаем его внутренний массив
            if isinstance(X, SharedDataFrame):
                X_obj = X.shared_array
            else:
                X_obj = X
            # Приведение y к Series или 1D массиву
            if isinstance(y, pd.DataFrame):
                y_obj = y.iloc[:, 0]
            elif isinstance(y, (pd.Series, np.ndarray)):
                y_obj = y
            else:
                y_obj = np.asarray(y)
                if y_obj.ndim > 1: # Это вызовет ValueError для 3D данных
                    raise ValueError("y must be 1D")
            # 3. Проверка на пустоту (через форму, чтобы не триггерить загрузку данных)
            if hasattr(X_obj, "shape"):
                if X_obj.shape[0] == 0:
                    raise ValueError("Данные пусты")
                n_samples = X_obj.shape[0]
            else:
                n_samples = len(X_obj)
            if n_samples < 2:
                raise TrainingError("Недостаточно записей для обучения (нужно минимум 2)")
            if n_samples != len(y_obj):
                raise TrainingError("Число строк X и y различается")
            return X_obj, y_obj
        except (ValueError, TypeError, IndexError) as e:
            if str(e) == "Данные пусты":
                raise
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
            X_prepared, y_s = self._prepare_data(X, y)

            if isinstance(X_prepared, pd.DataFrame):
                self._detect_feature_types(X_prepared, target_column="") 
            else:
                # Если пришел numpy массив без имен, инициализируем списки индексами
                if self.categorical_features is None: self.categorical_features = []
                if self.numerical_features is None:
                    self.numerical_features = [f"col_{i}" for i in range(X_prepared.shape[1])]

            # Этап 2: Определение ключа алгоритма 
            # (перенесено выше для использования в препроцессоре)
            _ALIASES.get(self.algorithm, self.algorithm)

            # Этап 3: Создаём препроцессор через делегирование (Task 1.2)
            preprocessor = self._build_preprocessor(self.feature_names)

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
                    iter_splits(X_prepared, 
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
                # 1. Получаем объект-скорер
                scorer = cast(Callable[..., Any], get_scorer_object(self.metric))
                
                # 2. Вычисляем raw_score (для ошибок sklearn вернет отрицательное число)
                raw_score = scorer(self.pipeline, X_val, y_val)
                if raw_score is None:
                    raise TrainingError("Scorer returned None")
                
                # 3. Инвертируем знак обратно, если это метрика-ошибка (RMSE, MAE и т.д.)
                # Чтобы в val_score всегда лежало "честное" положительное значение ошибки
                if not is_greater_better(self.metric):
                    # Если sklearn вернул отрицательную ошибку (neg_rmse), берем модуль
                    # Если вдруг вернул положительную (custom scorer), оставляем как есть
                    self.val_score = float(abs(raw_score))
                else:
                    # Для R2 и прочих score-метрик оставляем как есть
                    self.val_score = float(raw_score)
                self.logger.debug(
                    f"Metric calculation: raw={raw_score:.4f}, final val_score={self.val_score:.4f} "
                    f"(greater_is_better={is_greater_better(self.metric)})"
                )
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
                # Избегаем конвертации, если X уже массив или DataFrame
                if isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)):
                    X_input = X
                elif isinstance(X, SharedDataFrame):
                    X_input = X.shared_array
                else:
                    X_input = np.asarray(X)
                
                preds = self.pipeline.predict(X_input)
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
        cast(Optional[Union[str, Path]], cfg.get("log_path", None))
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
        # Определяем тип метрики для понятного лога
        metric_type = "Score" if is_greater_better(metric) else "Error (Natural)"

        logger.info(
            f"Training finished: Algorithm={algo_key}, "
            f"Metric={metric.upper()} ({metric_type}), "
            f"Value={val_score:.4f}"
        )

    return float(val_score)
