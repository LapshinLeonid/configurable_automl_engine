"""Preprocessing: единая точка построения препроцессора признаков.

Модуль инкапсулирует два низкоуровневых кирпича подготовки данных:

    1. :func:`detect_feature_types` — автоопределение категориальных и
       числовых колонок по ``pd.DataFrame``.
    2. :func:`build_preprocessor` — сборка ``sklearn.ColumnTransformer``
       с предобработкой по умолчанию **one-hot encoding** для категорий
       (импутация ``most_frequent`` + ``OneHotEncoder``) и скалированием
       для числовых признаков. Стратегия обработки числовых признаков
       (стратегия импутации и тип масштабирования) задаётся параметрами
       ``imputation_strategy``/``scaling`` и обычно берётся из адаптивного
       пресета предобработки (:mod:`preprocessing_presets`), который
       автоматически выбирается по регрессионному алгоритму. Поддерживается
       также альтернативная стратегия **ordinal encoding** (``OrdinalEncoder``)
       через аргумент ``encoding='ordinal'``.

Единая точка построения препроцессора используется в ОБОИХ местах обучения —
фазе подбора гиперпараметров (``tuner.optimize``) и финальном обучении
(``trainer.ModelTrainer``), что исключает рассинхрон логики предобработки
между этапами (FR-4).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from configurable_automl_engine.preprocessing_presets import (
    ImputationStrategy,
    ScalingType,
)

logger = logging.getLogger(__name__)

EncodingStrategy = Literal["one_hot", "ordinal"]


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
    encoding: EncodingStrategy = "one_hot",
    imputation_strategy: ImputationStrategy = "mean",
    scaling: ScalingType = "standard",
) -> ColumnTransformer:
    """Сконструировать ColumnTransformer для раздельной обработки типов данных.

    По умолчанию применяется стратегия **one-hot encoding** для категориальных
    признаков (требование задачи) и скалирование для числовых. При
    ``encoding='ordinal'`` категории кодируются ``OrdinalEncoder`` (1 выходной
    столбец на категориальную колонку) вместо one-hot.

    Стратегия предобработки числовых признаков задаётся параметрами
    ``imputation_strategy`` и ``scaling`` и обычно берётся из адаптивного
    пресета предобработки (:mod:`preprocessing_presets`): разные классы
    регрессионных алгоритмов получают подходящий именно им набор
    преобразований.

    Args:
        feature_names: Полный список имён признаков в порядке следования колонок.
        categorical_features: Имена колонок, кодируемых one-hot/ordinal.
        numerical_features: Имена колонок, подлежащих импутации и скалированию.
        encoding: Стратегия кодирования категорий: ``'one_hot'`` (по умолчанию)
            или ``'ordinal'``.
        imputation_strategy: Стратегия заполнения пропусков для числовых
            признаков: ``'mean'`` (по умолчанию) или ``'median'`` (устойчива
            к выбросам, учитывает скошенность распределений).
        scaling: Масштабирование числовых признаков: ``'standard'``
            (``StandardScaler``, по умолчанию), ``'robust'`` (``RobustScaler``,
            устойчив к выбросам) или ``'none'`` (масштабирование не
            применяется — данные передаются в модель без него).

    Returns:
        ``ColumnTransformer``, преобразующий исходный DataFrame в числовую
        матрицу, готовую к подаче в модель. Если ни одна колонка не совпала,
        возвращается passthrough-трансформер (поведение сохранено из тренера).

    Note:
        Категориальные колонки приводятся к строковому объектному массиву перед
        импутацией и кодированием, поэтому ``bool``-колонки обрабатываются
        корректно (без падения ``SimpleImputer`` на dtype bool).

    Note:
        При ``encoding='ordinal'`` категории кодируются целочисленными кодами,
        которые НЕ масштабируются (в отличие от числовых колонок, проходящих
        через скалер). Для линейных моделей (например, ``elasticnet``)
        несопоставимый масштаб кодов с категориями высокого порядка может
        доминировать над числовыми признаками.

    Raises:
        ValueError: Если имя колонки отсутствует в ``feature_names`` либо
            передано невалидное значение ``encoding``, ``imputation_strategy``
            или ``scaling``.
    """
    if encoding not in ("one_hot", "ordinal"):
        raise ValueError(
            f"Unknown encoding strategy: {encoding!r}. "
            "Expected one of ('one_hot', 'ordinal')."
        )
    if imputation_strategy not in ("mean", "median"):
        raise ValueError(
            f"Unknown imputation strategy: {imputation_strategy!r}. "
            "Expected one of ('mean', 'median')."
        )
    if scaling not in ("standard", "robust", "none"):
        raise ValueError(
            f"Unknown scaling type: {scaling!r}. "
            "Expected one of ('standard', 'robust', 'none')."
        )

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

    # Пайплайн трансформации числовых признаков: импутация + (опционально) скалер.
    # При scaling='none' шаг масштабирования отсутствует — данные передаются
    # в модель без него (AC-3 для деревьев и ансамблей).
    num_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy=imputation_strategy))
    ]
    if scaling != "none":
        scaler = RobustScaler() if scaling == "robust" else StandardScaler()
        num_steps.append(("scaler", scaler))
    num_transformer = Pipeline(steps=num_steps)

    if encoding == "ordinal":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        encoder_step_name = "ordinal"
    else:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder_step_name = "onehot"

    cat_transformer = Pipeline(
        steps=[
            # Приводим категориальные колонки к строковому объектному массиву.
            # Это делает пайплайн устойчивым к bool-колонкам: SimpleImputer со
            # strategy="most_frequent" падает на numpy-bool ("SimpleImputer does
            # not support data with dtype bool"), поэтому bool-признаки приводятся
            # к строкам ("True"/"False") и кодируются как обычные категории.
            ("to_string", FunctionTransformer(_to_string_array)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (encoder_step_name, encoder),
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
