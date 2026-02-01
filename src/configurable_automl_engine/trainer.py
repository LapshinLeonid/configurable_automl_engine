"""
trainer.py
----------
Обёртка над sklearn-регрессорами: берёт данные (X, y), строит пайплайн,
обучает модель и возвращает R². Содержит:
  - класс ModelTrainer
  - исключение TrainingError
  - функцию train_model (старый API, необходимый тестам).
"""

from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import Any
import threading

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from configurable_automl_engine.oversampling import DataOversampler
from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin

from .models import create_model, _ALIASES

from configurable_automl_engine.validation import make_cv, iter_splits
from configurable_automl_engine.common.definitions import SerializationFormat
from configurable_automl_engine.common.serialization_utils import save_artifact, load_artifact

__all__ = ["ModelTrainer", "TrainingError", "train_model"]


class TrainingError(RuntimeError):
    """Ошибки, связанные с некорректностью данных или параметров."""
    pass

class IsotonicDataTransformer(BaseEstimator, TransformerMixin):
    """
    Трансформер для подготовки данных под IsotonicRegression.
    Выбирает первый столбец и обрабатывает пропуски (NaN).
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        n_samples = len(X_df)
        
        # Выбираем первую колонку
        X_col = X_df.iloc[:, 0]
        
        # Логика обработки NaN (перенесена из ModelTrainer.fit)
        if X_col.isna().all():
            return pd.Series(range(n_samples)).to_numpy().reshape(-1, 1)
        
        median_val = X_col.median()
        if pd.isna(median_val):
            X_imputed = pd.Series(range(n_samples))
        else:
            X_imputed = X_col.fillna(median_val)
            
        return X_imputed.to_numpy().reshape(-1, 1)

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
        Имя метрики (поддерживается только "r2").
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
        self.val_r2_: float | None = None
        self._last_train_y: pd.Series | None = None 
        self._last_val_y: pd.Series | None = None 

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpicklable entries
        if 'lock' in state:
            del state['lock']
        if 'logger' in state:
            del state['logger']
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-initialize the lock after unpickling
        self.lock = threading.RLock()
        # Re-initialize logger if necessary
    
    def _build_preprocessor(self, X_df: pd.DataFrame, algo_key: str) -> Any:
        """
        Создает объект препроцессора в зависимости от алгоритма.
        """
        # Случай IsotonicRegression: используем наш новый кастомный трансформер
        if algo_key == "isotonic_regression":
            return IsotonicDataTransformer()

        # Стандартный случай (перенесено из оригинального fit)
        num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_df.select_dtypes(exclude=["number"]).columns.tolist()

        num_steps: list[tuple[str, Any]] = [
            ("impute", SimpleImputer(strategy="median"))
        ]
        
        # Список алгоритмов, требующих масштабирования
        scaling_required = {
            "sgdregressor", 
            "elasticnet", 
            "ardregression", 
            "gaussian_process_regression"
        }
        
        if algo_key in scaling_required:
            num_steps.append(("scale", StandardScaler()))
            
        num_pipeline = Pipeline(steps=num_steps)

        transformers: list[tuple[str, Pipeline, list[str]]] = [
            ("num", num_pipeline, num_cols)
        ]
        
        if cat_cols:
            cat_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            transformers.append(("cat", cat_pipeline, cat_cols))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit(self, X: Any, y: Any) -> ModelTrainer:
        """
        Обучает модель на данных (X, y).
        Вычисляет R² (self.val_r2_) и возвращает self.
        В случае ошибок бросает TrainingError, ValueError или TypeError.
        """
        # 1) Проверка типов X, y
        if not isinstance(X, (pd.DataFrame, pd.Series, np.ndarray)) or not isinstance(
            y, (pd.Series, pd.DataFrame, np.ndarray)
        ):
            raise TrainingError("Неподдерживаемый тип данных для X или y")

        # 2) Приведение к pandas
        try:
            X_df = (
                pd.DataFrame(X)
                if not isinstance(X, (pd.DataFrame, pd.Series))
                else X  # type: ignore
            )
            if isinstance(y, pd.DataFrame):
                y_s = pd.Series(y.iloc[:, 0])  # type: ignore
            else:
                y_s = (
                    pd.Series(y)
                    if not isinstance(y, pd.Series)
                    else y  # type: ignore
                )
        except Exception as e:
            raise TrainingError(f"Ошибка при преобразовании данных: {e}")

        # 3) Проверка пустоты и размеров
        n_samples = len(X_df)
        if n_samples == 0 or len(y_s) == 0:
            raise TrainingError("Данные пусты")
        if n_samples != len(y_s):
            raise TrainingError("Число строк X и y различается")
        if n_samples < 2:
            raise TrainingError("Недостаточно записей для обучения")

        # 4) Определение ключа алгоритма (перенесено выше для использования в препроцессоре)
        algo_key = _ALIASES.get(self.algorithm, self.algorithm)
        # 5) Создаём препроцессор через делегирование (Task 1.2)
        preprocessor = self._build_preprocessor(X_df, algo_key)
        # 6) Создаём модель через фабрику
        try:
            base_model = create_model(self.algorithm, **self.model_params)
        except (ValueError, ImportError) as e:
            raise TrainingError(f"Ошибка при создании модели: {e}")
        # 7) Собираем полный Pipeline (теперь preprocessor универсален)
        steps = [("preprocessor", preprocessor), ("model", base_model)]
        self.pipeline = Pipeline(steps=steps)

        # 8) Разбиваем выборку на train/validation
        try:
            # 1. Используем аргумент method вместо strategy
            # 2. Помним, что функция возвращает Generator кортежей (X_tr, X_te, y_tr, y_te)
            # 3. Переводим DataFrame в numpy (так как в validation.py стоит аннотация np.ndarray)
            #    и забираем первый результат через next()
            X_train, X_val, y_train, y_val = next(
                iter_splits(
                    X_df, 
                    y_s, 
                    method='train_test_split', 
                    random_state=self.random_state
                )
            )
            # сохраняем для тестов / дебага
            self._last_train_y = y_train
            self._last_val_y = y_val
        except ValueError:
            raise
        except Exception as e:
            raise TrainingError(f"Ошибка при разбиении данных: {e}")

        # 9) Обучаем пайплайн
        try:
            self.pipeline.fit(X_train, y_train)
        except Exception as e:
            raise TrainingError(f"Ошибка при обучении модели: {e}")

        # 10) Предсказание и R² на validation
        try:
            preds = self.pipeline.predict(X_val)
            self.val_r2_ = float(r2_score(y_val, preds))
        except ValueError:
            raise
        except Exception as e:
            raise TrainingError(f"Ошибка при валидации модели: {e}")

        return self

    def predict(self, X: Any) -> np.ndarray:
        """
        Предсказывает y для новых данных X, используя обученный пайплайн.
        """
        if self.pipeline is None:
            raise TrainingError("Модель не обучена. Сначала вызовите метод fit().")
            
        try:
            # Приводим к pandas DataFrame, если это еще не он
            X_df = pd.DataFrame(X) if not isinstance(X, (pd.DataFrame, pd.Series)) else X
            
            # Просто вызываем predict у пайплайна. 
            # Он сам прогонит данные через IsotonicDataTransformer (если это изотоническая регрессия)
            # или через ColumnTransformer (для всех остальных моделей).
            return self.pipeline.predict(X_df)
        except Exception as e:
            raise TrainingError(f"Ошибка при предсказании: {e}")


    def save(self, path: str | Path) -> None:
        """
        Сохраняет объект ModelTrainer (пайплайн + параметры) в указанный файл,
        используя заданный формат сериализации.
        """
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
    def load(cls, path: str | Path, fmt: SerializationFormat = SerializationFormat.pickle) -> ModelTrainer:
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
    cfg_or_algo: dict | str,
    metric_or_testsize: str | float,
    params_or_metric: dict | str,
    X: Any = None,
    y: Any = None,
    enable_logging: bool = False,
    random_state: int | None = 42,
    log_path: str | Path | None = None,
) -> float:
    """
    Старый API (необходим тестам):
      1) train_model(config_dict) → float (R²)
      2) train_model(algo, metric, params_dict, X, y, enable_logging, random_state, log_path)

    Возвращает float R² на validation.

    При ошибках бросает TrainingError, ValueError или TypeError.
    """
    # Случай «config dict»
    if isinstance(cfg_or_algo, dict):
        cfg: dict = cfg_or_algo  # type: ignore
        algo = cfg.get("algorithm")
        metric = cfg.get("metric")
        model_params = cfg.get("model_params", {})
        test_size = cfg.get("test_size", 0.3)
        rs = cfg.get("random_state", 42)
        enable_logging = cfg.get("enable_logging", False)
        log_path = cfg.get("log_path", None)
        data_os = cfg.get("data_oversampling", False)
        data_os_mult = cfg.get("data_oversampling_multiplier", 1.0)
        data_os_alg = cfg.get("data_oversampling_algorithm", "random")
    else:
        # Простой API
        algo = cfg_or_algo  # type: ignore
        metric = metric_or_testsize  # type: ignore
        model_params = params_or_metric  # type: ignore
        test_size = 0.3
        rs = random_state
        data_os = False
        data_os_mult = 1.0
        data_os_alg = "random"

    # Проверка алгоритма
    if not isinstance(algo, str):
        raise TrainingError("Неверный алгоритм")
    algo_key = algo.lower()

    # Проверка метрики (только "r2")
    if not isinstance(metric, str) or metric.lower() != "r2":
        raise TrainingError("Неподдерживаемая метрика")

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
            else X  # type: ignore
        )
        if isinstance(y, pd.DataFrame):
            y_s = pd.Series(y.iloc[:, 0])  # type: ignore
        else:
            y_s = pd.Series(y) if not isinstance(y, pd.Series) else y  # type: ignore
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
        val_score = trainer.val_r2_
    except TrainingError:
        raise
    except ValueError:
        raise
    except TypeError:
        raise

    # Логирование (если enable_logging=True)
    if enable_logging:
        log_file = log_path if log_path else "training.log"
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
        logger.debug(f"Алгоритм: {algo_key}; R2: {val_score:.4f}")

    return float(val_score)
