"""Preprocessing: единая точка построения препроцессора категориальных признаков.

Модуль инкапсулирует два низкоуровневых кирпича подготовки данных:

    1. :func:`detect_feature_types` — автоопределение категориальных и
       числовых колонок по ``pd.DataFrame``.
    2. :func:`build_preprocessor` — сборка ``sklearn.ColumnTransformer``
       с предобработкой по умолчанию **one-hot encoding** для категорий
       (импутация ``most_frequent`` + ``OneHotEncoder``) и
       ``StandardScaler`` для числовых признаков.

Единая точка построения препроцессора используется в ОБОИХ местах обучения —
фазе подбора гиперпараметров (``tuner.optimize``) и финальном обучении
(``trainer.ModelTrainer``), что исключает рассинхрон логики предобработки
между этапами.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def _to_string_array(X):
    """Привести категориальную матрицу к строковому объектному массиву.

    ``np.ndarray.astype(str)`` даёт fixed-width unicode (``<U``), а
    ``SimpleImputer`` принимает только ``object``-массивы, поэтому после
    преобразования в строки выполняется повторный каст в ``object`` dtype.
    """
    return np.asarray(X).astype(str).astype(object)


def detect_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Автоматически классифицировать колонки на числовые и категориальные.

    Категориальными считаются колонки с dtype ``object``, ``category`` и
    ``bool``; числовыми — колонки с dtype из семейства ``number``.

    Args:
        X: DataFrame признаков (без целевой переменной).

    Returns:
        Кортеж ``(categorical_features, numerical_features)`` — списки имён
        колонок каждого типа.

    Note:
        Функция ожидает именно ``pd.DataFrame``; для ``np.ndarray`` без имён
        колонок автоопределение невозможно (задокументированное ограничение).

    Note:
        Классификация основана на ``dtype`` колонок. Это осознанно отличается от
        value-based-детекции в ``oversampling._is_categorical_col``: здесь
        препроцессор обрабатывает данные до оверсэмплинга, поэтому ``bool`` и
        'числовые строки' (``'10','20'``) рассматриваются как категории (one-hot).
        Оверсэмплер же получает уже закодированные числовые признаки и применяет
        собственную value-based-логику для standalone-вызовов.
    """
    categorical = X.select_dtypes(
        include=["object", "str", "category", "bool"]
    ).columns.tolist()
    numerical = X.select_dtypes(include=["number"]).columns.tolist()
    return categorical, numerical


def build_preprocessor(
    feature_names: list[str],
    categorical_features: list[str],
    numerical_features: list[str],
) -> ColumnTransformer:
    """Сконструировать ColumnTransformer для раздельной обработки типов данных.

    По умолчанию применяется стратегия **one-hot encoding** для категориальных
    признаков (требование задачи) и скалирование для числовых.

    Args:
        feature_names: Полный список имён признаков в порядке следования колонок.
        categorical_features: Имена колонок, кодируемых one-hot.
        numerical_features: Имена колонок, подлежащих импутации и скалированию.

    Returns:
        ``ColumnTransformer``, преобразующий исходный DataFrame в числовую
        матрицу, готовую к подаче в модель. Если ни одна колонка не совпала,
        возвращается passthrough-трансформер (поведение сохранено из тренера).

    Note:
        Категориальные колонки приводятся к строковому объектному массиву перед
        импутацией и one-hot-кодированием, поэтому ``bool``-колонки обрабатываются
        корректно (без падения ``SimpleImputer`` на dtype bool).

    Raises:
        ValueError: Если имя колонки отсутствует в ``feature_names``.
    """
    # Сопоставляем имена колонок с порядковыми номерами
    cat_indices = [
        feature_names.index(col) for col in categorical_features if col in feature_names
    ]
    num_indices = [
        feature_names.index(col) for col in numerical_features if col in feature_names
    ]

    if not cat_indices and not num_indices:
        logger.warning(
            "No features matched for preprocessing. Defaulting to passthrough."
        )
        return ColumnTransformer(
            [("bypass", "passthrough", slice(None))], remainder="drop"
        )

    # Пайплайны трансформации
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            # Приводим категориальные колонки к строковому объектному массиву.
            # Это делает пайплайн устойчивым к bool-колонкам: SimpleImputer со
            # strategy="most_frequent" падает на numpy-bool ("SimpleImputer does
            # not support data with dtype bool"), поэтому bool-признаки приводятся
            # к строкам ("True"/"False") и кодируются one-hot как обычные категории.
            ("to_string", FunctionTransformer(_to_string_array)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if cat_indices:
        transformers.append(("cat", cat_transformer, cat_indices))
    if num_indices:
        transformers.append(("num", num_transformer, num_indices))

    return ColumnTransformer(
        transformers=(transformers if transformers else [("pass", "passthrough", [0])]),
        remainder="drop",
    )
