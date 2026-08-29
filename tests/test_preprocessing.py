"""Unit tests for the shared preprocessing module (categorical feature handling)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from configurable_automl_engine.preprocessing import (
    build_preprocessor,
    detect_feature_types,
)


def test_detect_feature_types_mixed_dataframe():
    """detect_feature_types различает категориальные и числовые колонки."""
    df = pd.DataFrame(
        {
            "cat_color": ["red", "green", "blue"],
            "cat_size": pd.Categorical(["S", "M", "L"]),
            "flag": [True, False, True],
            "num_a": [1.0, 2.0, 3.0],
            "num_b": [10, 20, 30],
        }
    )
    cat, num = detect_feature_types(df)

    assert sorted(cat) == ["cat_color", "cat_size", "flag"]
    assert sorted(num) == ["num_a", "num_b"]


def test_detect_feature_types_all_numeric():
    """На чисто числовом DataFrame категориальных колонок нет."""
    df = pd.DataFrame({"a": [1, 2], "b": [0.5, 0.7]})
    cat, num = detect_feature_types(df)

    assert cat == []
    assert sorted(num) == ["a", "b"]


def test_build_preprocessor_onehot_for_categorical():
    """build_preprocessor возвращает ColumnTransformer с OneHotEncoder для категорий."""
    feature_names = ["cat", "num"]
    preprocessor = build_preprocessor(
        feature_names,
        categorical_features=["cat"],
        numerical_features=["num"],
    )

    assert isinstance(preprocessor, ColumnTransformer)

    names = [name for name, _, _ in preprocessor.transformers]
    assert "cat" in names
    assert "num" in names

    # Проверяем, что категориальный трансформер содержит OneHotEncoder
    cat_transformer = dict(
        (name, transformer) for name, transformer, _ in preprocessor.transformers
    )["cat"]
    encoder_names = [step[0] for step in cat_transformer.steps]
    assert "onehot" in encoder_names
    assert isinstance(cat_transformer.named_steps["onehot"], OneHotEncoder)


def test_build_preprocessor_end_to_end_encoding():
    """Проверка сквозного кодирования категорий через препроцессор."""
    df = pd.DataFrame(
        {
            "cat": ["red", "green", "red", "blue"],
            "num": [1.0, 2.0, 3.0, 4.0],
        }
    )
    preprocessor = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=["num"],
    )
    out = preprocessor.fit_transform(df)

    assert out.shape == (4, 3 + 1)  # 3 one-hot колонки + 1 числовая


def test_build_preprocessor_no_features_passthrough():
    """При отсутствии совпавших колонок возвращается passthrough-трансформер."""
    preprocessor = build_preprocessor(
        ["some_random_column"],
        categorical_features=[],
        numerical_features=[],
    )

    assert isinstance(preprocessor, ColumnTransformer)
    assert preprocessor.transformers[0][0] == "bypass"
    assert preprocessor.transformers[0][1] == "passthrough"


def test_build_preprocessor_numeric_strings_treated_as_categorical():
    """Колонки 'числовых строк' (ID '10','20') детектируются как категории по dtype."""
    df = pd.DataFrame({"id_code": ["10", "20", "30", "20"]})
    cat, _ = detect_feature_types(df)

    assert cat == ["id_code"]

    preprocessor = build_preprocessor(
        list(df.columns), categorical_features=cat, numerical_features=[]
    )
    out = preprocessor.fit_transform(df)
    assert out.shape == (4, 3)

    # Все значения в матрице являются числами (one-hot)
    assert np.isfinite(out).all()


def test_build_preprocessor_bool_column_not_crash():
    """Чисто bool-колонка не роняет препроцессор при fit_transform.

    Регрессия на критический баг: SimpleImputer(strategy="most_frequent")
    падал на numpy-bool ("SimpleImputer does not support data with dtype bool").
    """
    df = pd.DataFrame(
        {
            "flag": [True, False, True, False],
            "num": [1.0, 2.0, 3.0, 4.0],
        }
    )
    cat, num = detect_feature_types(df)

    assert cat == ["flag"]
    assert num == ["num"]

    preprocessor = build_preprocessor(list(df.columns), cat, num)
    out = preprocessor.fit_transform(df)

    # bool -> one-hot (2 категории) + 1 числовая
    assert out.shape == (4, 3)
    assert np.isfinite(out).all()


def test_build_preprocessor_mixed_object_and_bool():
    """Совместная обработка object- и bool-колонок через один категориальный
    пайплайн без падения на dtype bool."""
    df = pd.DataFrame(
        {
            "color": ["red", "green", "red", "blue"],
            "flag": [True, False, True, False],
            "num": [1.0, 2.0, 3.0, 4.0],
        }
    )
    cat, num = detect_feature_types(df)

    assert sorted(cat) == ["color", "flag"]
    assert num == ["num"]

    preprocessor = build_preprocessor(list(df.columns), cat, num)
    out = preprocessor.fit_transform(df)

    # color (3 категории) + flag (2 категории) + 1 числовая
    assert out.shape == (4, 6)
    assert np.isfinite(out).all()
