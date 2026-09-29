"""Unit-тесты новых стратегий кодирования категориальных признаков (issue #20).

Покрывают:
    * target encoding (сглаживание, fallback, отсутствие data leakage);
    * frequency encoding (частоты, детерминизм, unknown -> 0.0);
    * hashing encoding (фиксированная размерность, детерминизм, unknown);
    * автоматический high-cardinality режим (порог, threshold=0, крайние случаи);
    * валидацию параметров build_preprocessor;
    * краевые случаи: одна категория, пустая колонка, NaN;
    * сохранение типов категорий в target encoding (issue #1): int/float/bool
      standalone, int-fit/float-transform эквивалентность, консистентность с
      frequency encoding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from configurable_automl_engine.preprocessing import (
    FrequencyEncodingTransformer,
    HashingEncodingTransformer,
    SplitCategoricalEncoder,
    TargetEncodingTransformer,
    build_preprocessor,
)


# ───────────────────────── Target encoding ─────────────────────────


def _target_df(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Синтетический датасет с категориальной колонкой и числовым таргетом."""
    rng = np.random.default_rng(seed)
    cat = rng.choice(["a", "b", "c"], size=n)
    df = pd.DataFrame({"cat": cat, "num": rng.normal(size=n)})
    y = pd.Series(
        pd.Series(cat).map({"a": 1.0, "b": 2.0, "c": 3.0}).to_numpy()
        + rng.normal(0, 0.05, n)
    )
    return df, y


def test_target_encoding_smoothing_formula():
    """Значение закодированной категории равно формуле со сглаживанием."""
    X = np.array([["a"], ["a"], ["b"], ["b"], ["b"], ["c"]], dtype=object)
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    out = enc.fit_transform(X, y)
    # При m=0 — чистое среднее по категории
    assert out[0, 0] == pytest.approx(10.0)
    assert out[2, 0] == pytest.approx(20.0)
    assert out[5, 0] == pytest.approx(30.0)

    global_mean = float(np.mean(y))  # (10+10+20+20+20+30)/6 = 18.333...
    enc_smooth = TargetEncodingTransformer(smoothing=100.0)
    out_smooth = enc_smooth.fit_transform(X, y)
    # При очень большом m значение близко к глобальному среднему
    assert out_smooth[0, 0] == pytest.approx(global_mean, abs=0.5)


def test_target_encoding_fallback_global_mean():
    """Неизвестная категория -> глобальное среднее (fallback по умолчанию)."""
    X_train = np.array([["a"], ["a"], ["b"]], dtype=object)
    y_train = np.array([1.0, 1.0, 5.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    enc.fit(X_train, y_train)
    out = enc.transform(np.array([["unknown"], ["a"]], dtype=object))
    assert out[0, 0] == pytest.approx(float(np.mean(y_train)))
    assert out[1, 0] == pytest.approx(1.0)


def test_target_encoding_custom_fallback():
    """Настраиваемый fallback используется для неизвестных категорий."""
    X_train = np.array([["a"], ["a"], ["b"]], dtype=object)
    y_train = np.array([1.0, 1.0, 5.0])
    enc = TargetEncodingTransformer(smoothing=0.0, fallback=-99.0)
    enc.fit(X_train, y_train)
    out = enc.transform(np.array([["unknown"]], dtype=object))
    assert out[0, 0] == pytest.approx(-99.0)


def test_target_encoding_no_leakage():
    """Статистики считаются только по обучающей части: категория, встречающаяся
    только в валидации, получает fallback, а не своё валидационное среднее."""
    X_train = np.array([["a"], ["a"], ["b"]], dtype=object)
    y_train = np.array([1.0, 1.0, 5.0])
    X_val = np.array([["a"], ["only_val"]], dtype=object)
    y_val = np.array([999.0, 999.0])  # если бы утекало — получили бы 999

    enc = TargetEncodingTransformer(smoothing=0.0)
    enc.fit(X_train, y_train)
    out = enc.transform(X_val)
    # 'a' -> 1.0 (среднее по train), 'only_val' -> глобальное среднее train (не 999)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(float(np.mean(y_train)), abs=1e-9)


def test_target_encoding_requires_y():
    """fit() без y бросает ValueError."""
    enc = TargetEncodingTransformer()
    with pytest.raises(ValueError, match="requires y"):
        enc.fit(np.array([["a"], ["b"]], dtype=object))


def test_target_encoding_non_numeric_y():
    """Нечисловой таргет отклоняется."""
    enc = TargetEncodingTransformer()
    with pytest.raises(ValueError, match="numeric target"):
        enc.fit(np.array([["a"], ["b"]], dtype=object), np.array(["x", "y"]))


def test_target_encoding_end_to_end_shape():
    """Сквозное target-кодирование через препроцессор: 1 колонка на категорию."""
    df, y = _target_df()
    pre = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=["num"],
        encoding="target",
    )
    out = pre.fit_transform(df, y)
    assert out.shape == (len(df), 2)  # target-кодированная cat + num
    assert np.isfinite(out).all()


def test_target_encoding_nan_column():
    """NaN в категориальной колонке не роняет target encoding (импутация
    most_frequent заполняет пропуски до кодирования)."""
    df = pd.DataFrame(
        {"cat": ["a", "b", None, "a", None], "num": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    pre = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=["num"],
        encoding="target",
    )
    out = pre.fit_transform(df, y)
    assert out.shape == (5, 2)
    assert np.isfinite(out).all()


def test_target_encoding_single_unique_category():
    """Колонка с одной уникальной категорией кодируется без падения."""
    df = pd.DataFrame({"cat": ["only", "only", "only"], "num": [1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0, 3.0])
    pre = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=["num"],
        encoding="target",
    )
    out = pre.fit_transform(df, y)
    assert out.shape == (3, 2)
    assert np.isfinite(out).all()
    # Значение равно глобальному среднему (одна категория => чистое среднее)
    assert out[0, 0] == pytest.approx(2.0)


def test_target_encoding_all_missing_column():
    """Полностью пропущенная категориальная колонка после импутации кодируется
    (все значения становятся одной строкой-категорией)."""
    df = pd.DataFrame({"cat": [None, None, None], "num": [1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0, 3.0])
    pre = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=["num"],
        encoding="target",
    )
    out = pre.fit_transform(df, y)
    assert out.shape == (3, 2)
    assert np.isfinite(out).all()


def test_target_encoding_rare_category_smoothing():
    """Редкая категория с большим сглаживанием близка к глобальному среднему."""
    X = np.array([["common"]] * 100 + [["rare"]], dtype=object)
    y = np.array([0.0] * 100 + [100.0])
    enc = TargetEncodingTransformer(smoothing=20.0)
    enc.fit(X, y)
    out = enc.transform(np.array([["rare"]], dtype=object))
    # Формула: (mean_cat * n + global_mean * m) / (n + m)
    global_mean = float(np.mean(y))  # 100/101
    expected = (100.0 * 1 + global_mean * 20.0) / (1 + 20.0)
    assert out[0, 0] == pytest.approx(expected, rel=1e-6)


# ─── Target encoding: сохранение типов категорий (issue #1) ───


def test_target_encoding_int_categories_standalone():
    """int-категории standalone кодируются своими средними, а не fallback'ом.

    Регрессия issue #1: ранее ключи статистик приводились к строкам, и int-коды
    в transform() не находили соответствий (всё тихо падало в fallback).
    """
    X = np.array([[1], [1], [2], [2], [2], [3]], dtype=np.int64)
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    out = enc.fit_transform(X, y)
    assert out[:, 0].tolist() == pytest.approx(
        [10.0, 10.0, 20.0, 20.0, 20.0, 30.0], rel=1e-6
    )
    # Ключи статистик сохраняют исходный числовой тип (не нормализуются к строкам)
    assert not any(isinstance(k, str) for k in enc.statistics_[0].keys())
    assert all(isinstance(k, (int, np.integer)) for k in enc.statistics_[0].keys())


def test_target_encoding_float_categories_standalone():
    """float-категории standalone кодируются своими средними, а не fallback'ом."""
    X = np.array([[1.0], [1.0], [2.0], [2.0], [2.0], [3.0]])
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    out = enc.fit_transform(X, y)
    assert out[:, 0].tolist() == pytest.approx(
        [10.0, 10.0, 20.0, 20.0, 20.0, 30.0], rel=1e-6
    )


def test_target_encoding_bool_categories_standalone():
    """bool-категории standalone кодируются средними по True/False."""
    X = np.array([[True], [True], [False], [False], [False], [True]])
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    out = enc.fit_transform(X, y)
    true_mean = float(np.mean(y[[0, 1, 5]]))  # (10+10+30)/3 = 16.67
    false_mean = float(np.mean(y[[2, 3, 4]]))  # 20.0
    assert out[:, 0].tolist() == pytest.approx(
        [true_mean, true_mean, false_mean, false_mean, false_mean, true_mean],
        rel=1e-6,
    )


def test_target_encoding_bool_int_mixing_documented_semantics():
    """Смешивание bool и int 0/1 даёт задокументированное поведение (fallback).

    pandas подбирает соответствия по dtype: ``bool`` — отдельный dtype, поэтому
    bool-ключи статистик не находят int-значения (и наоборот), и все значения
    уходят в fallback. Это фиксирует предупреждение из docstring класса: категории
    одного признака должны сохранять один dtype в ``fit`` и ``transform``.
    """
    y = np.array([10.0, 10.0, 20.0, 20.0, 30.0])
    X_bool = np.array([[True], [True], [False], [False], [True]])

    enc = TargetEncodingTransformer(smoothing=0.0).fit(X_bool, y)
    out = enc.transform(np.array([[1], [0]], dtype=np.int64))
    assert out[:, 0].tolist() == pytest.approx([enc.fallback_, enc.fallback_])

    enc_int = TargetEncodingTransformer(smoothing=0.0).fit(
        np.array([[1], [1], [0], [0], [1]], dtype=np.int64), y
    )
    out_int = enc_int.transform(np.array([[True], [False]]))
    assert out_int[:, 0].tolist() == pytest.approx(
        [enc_int.fallback_, enc_int.fallback_]
    )


def test_target_encoding_int_fit_float_transform_equivalence():
    """Кодирование, полученное на int-кодах, применимо к float-значениям.

    pandas подбирает соответствия по dtype-совместимым ключам: int и float
    взаимозаменяемы.
    """
    X = np.array([[1], [1], [2], [2], [2], [3]], dtype=np.int64)
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    enc.fit(X, y)
    out = enc.transform(np.array([[1.0], [2.0], [3.0]]))
    assert out[:, 0].tolist() == pytest.approx([10.0, 20.0, 30.0], rel=1e-6)


def test_target_encoding_float_fit_int_transform_equivalence():
    """Обратная совместимость: fit на float, transform на int-кодах."""
    X = np.array([[1.0], [1.0], [2.0], [2.0], [2.0], [3.0]])
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    enc.fit(X, y)
    out = enc.transform(np.array([[1], [2], [3]], dtype=np.int64))
    assert out[:, 0].tolist() == pytest.approx([10.0, 20.0, 30.0], rel=1e-6)


def test_target_encoding_key_types_match_frequency_encoding():
    """Target- и Frequency-трансформеры одинаково сохраняют типы ключей.

    Согласованность соседних классов (issue #1): оба хранят ключи статистик в
    исходном типе категорий, поэтому int-колонка кодируется корректно в обоих.
    """
    X = np.array([[1], [1], [2], [2], [2], [3]], dtype=np.int64)
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    target = TargetEncodingTransformer(smoothing=0.0).fit(X, y)
    freq = FrequencyEncodingTransformer().fit(X)
    target_keys = set(target.statistics_[0].keys())
    freq_keys = set(freq.frequencies_[0].keys())
    assert target_keys == freq_keys == {1, 2, 3}
    assert not any(isinstance(k, str) for k in target_keys | freq_keys)
    # Оба корректно находят соответствия для int-кодов (не fallback)
    assert target.transform(np.array([[3]], dtype=np.int64))[0, 0] == pytest.approx(30.0)
    assert freq.transform(np.array([[3]], dtype=np.int64))[0, 0] == pytest.approx(1 / 6)


def test_target_encoding_numeric_unknown_falls_back():
    """Числовая категория, отсутствовавшая в обучении, -> fallback."""
    X = np.array([[1], [1], [2], [2], [2], [3]], dtype=np.int64)
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    enc.fit(X, y)
    out = enc.transform(np.array([[42], [1]], dtype=np.int64))
    assert out[0, 0] == pytest.approx(float(np.mean(y)))  # неизвестная -> fallback
    assert out[1, 0] == pytest.approx(10.0)  # известная кодируется как обычно


def test_target_encoding_nan_dropped_in_fit_and_fallback_in_transform():
    """NaN standalone: в fit группа NaN отбрасывается, в transform -> fallback."""
    X = np.array([[1.0], [1.0], [np.nan], [2.0], [2.0]])
    y = np.array([10.0, 10.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    enc.fit(X, y)
    assert set(enc.statistics_[0].keys()) == {1.0, 2.0}  # NaN-группа отброшена
    out = enc.transform(np.array([[np.nan], [1.0]]))
    assert out[0, 0] == pytest.approx(float(np.mean(y)))  # NaN -> fallback
    assert out[1, 0] == pytest.approx(10.0)


def test_target_encoding_single_int_category_standalone():
    """Одна int-категория standalone: все значения = глобальное среднее."""
    X = np.array([[7], [7], [7]], dtype=np.int64)
    y = np.array([1.0, 2.0, 3.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    out = enc.fit_transform(X, y)
    assert out[:, 0].tolist() == pytest.approx([2.0, 2.0, 2.0], rel=1e-6)


# ─────────────────────── Frequency encoding ────────────────────────


def test_frequency_encoding_values():
    """Частоты считаются как count / n_rows."""
    X = np.array([["a"], ["a"], ["b"], ["c"], ["c"], ["c"]], dtype=object)
    enc = FrequencyEncodingTransformer()
    out = enc.fit_transform(X)
    assert out[0, 0] == pytest.approx(2 / 6)
    assert out[2, 0] == pytest.approx(1 / 6)
    assert out[3, 0] == pytest.approx(3 / 6)
    # Сумма частот уникальных категорий = 1
    assert np.unique(out).sum() == pytest.approx(1.0)


def test_frequency_encoding_unknown_is_zero():
    """Неизвестная категория -> 0.0 (детерминированный fallback)."""
    X_train = np.array([["a"], ["a"], ["b"]], dtype=object)
    enc = FrequencyEncodingTransformer()
    enc.fit(X_train)
    out = enc.transform(np.array([["zzz"], ["a"]], dtype=object))
    assert out[0, 0] == 0.0
    assert out[1, 0] == pytest.approx(2 / 3)


def test_frequency_encoding_deterministic():
    """Frequency encoding детерминирован: одинаковые входы -> одинаковые выходы."""
    X = np.array([["a"], ["b"], ["a"]], dtype=object)
    enc1 = FrequencyEncodingTransformer().fit(X)
    enc2 = FrequencyEncodingTransformer().fit(X)
    out1 = enc1.transform(X)
    out2 = enc2.transform(X)
    np.testing.assert_array_equal(out1, out2)


def test_frequency_encoding_nan_handled():
    """NaN превращается в строку 'nan' и обрабатывается как обычная категория."""
    df = pd.DataFrame({"cat": ["a", "b", None, "a"], "num": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    pre = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=["num"],
        encoding="frequency",
    )
    out = pre.fit_transform(df, y)
    assert out.shape == (4, 2)
    assert np.isfinite(out).all()


# ───────────────────────── Hashing encoding ─────────────────────────


def test_hashing_encoding_dimension():
    """Выходная размерность фиксирована: n_components на входную колонку."""
    X = np.array([["a"], ["b"], ["c"]], dtype=object)
    enc = HashingEncodingTransformer(n_components=8)
    out = enc.fit_transform(X)
    assert out.shape == (3, 8)
    # Выход разреженный (csr), как у sklearn FeatureHasher, — плотная матрица
    # float64 размерности (n_rows, n_cols * n_components) вызвала бы OOM.
    assert sparse.issparse(out)
    assert out.format == "csr"
    # nnz = n_rows * n_cols, значения бинарные (1.0)
    assert out.nnz == 3
    assert set(np.unique(out.data)).issubset({1.0})


def test_hashing_encoding_multiple_columns():
    """Для нескольких колонок размерность умножается."""
    X = np.array([["a", "x"], ["b", "y"]], dtype=object)
    enc = HashingEncodingTransformer(n_components=4)
    out = enc.fit_transform(X)
    assert out.shape == (2, 8)
    assert sparse.issparse(out)
    assert out.nnz == 4


def test_hashing_encoding_deterministic():
    """Hashing детерминирован при фиксированных настройках и seed."""
    X = np.array([["alpha"], ["beta"], ["gamma"], ["delta"]], dtype=object)
    enc1 = HashingEncodingTransformer(n_components=32, random_state=42)
    enc2 = HashingEncodingTransformer(n_components=32, random_state=42)
    out1 = enc1.fit_transform(X)
    out2 = enc2.fit_transform(X)
    assert (out1 != out2).nnz == 0


def test_hashing_encoding_seed_changes_output():
    """Разные seed дают разные проекции (для типичного случая)."""
    X = np.array([[f"cat_{i}"] for i in range(64)], dtype=object)
    out1 = HashingEncodingTransformer(n_components=64, random_state=1).fit_transform(X)
    out2 = HashingEncodingTransformer(n_components=64, random_state=2).fit_transform(X)
    assert (out1 != out2).nnz > 0


def test_hashing_encoding_unknown_category():
    """Неизвестная категория хешируется детерминированно (без падений)."""
    X_train = np.array([["a"], ["b"], ["c"]], dtype=object)
    enc = HashingEncodingTransformer(n_components=16, random_state=42)
    enc.fit(X_train)
    out = enc.transform(np.array([["never_seen"], ["a"]], dtype=object))
    assert out.shape == (2, 16)
    assert sparse.issparse(out)
    # Повторный transform неизвестной категории даёт тот же результат
    out2 = enc.transform(np.array([["never_seen"]], dtype=object))
    assert (out[0] != out2[0]).nnz == 0


def test_hashing_encoding_binary_indicators():
    """Каждая строка имеет ровно одну активную колонку на входную колонку."""
    X = np.array([["a"], ["b"]], dtype=object)
    enc = HashingEncodingTransformer(n_components=16)
    out = enc.fit_transform(X)
    assert np.asarray(out.sum(axis=1)).ravel().tolist() == [1.0, 1.0]
    # Бинарность значений: любой элемент матрицы — 0 или 1
    dense = out.toarray()
    assert set(np.unique(dense)).issubset({0.0, 1.0})


# ───────────────────── High-cardinality режим ──────────────────────


def _hc_df(seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n = 200
    df = pd.DataFrame(
        {
            "low": rng.choice(["a", "b", "c"], size=n),
            "high": [f"x{i % 60}" for i in range(n)],
            "num": rng.normal(size=n),
        }
    )
    y = pd.Series(
        df["low"].map({"a": 1.0, "b": 2.0, "c": 3.0}).to_numpy() + rng.normal(0, 0.1, n)
    )
    return df, y


def test_hc_mode_splits_columns():
    """При пороге колонки разделяются: low -> default, high -> HC-стратегия."""
    df, y = _hc_df()
    pre = build_preprocessor(
        list(df.columns),
        categorical_features=["low", "high"],
        numerical_features=["num"],
        encoding="one_hot",
        high_cardinality_threshold=10,
        high_cardinality_encoding="target",
    )
    pre.fit(df, y)
    cat_transformer = pre.named_transformers_["cat"]
    encoder = cat_transformer.named_steps["encoder"]

    assert encoder.high_cardinality_threshold == 10
    assert encoder.cardinalities_ == [3, 60]
    assert encoder.default_columns_ == [0]  # low (3 <= 10)
    assert encoder.hc_columns_ == [1]  # high (60 > 10)

    out = pre.transform(df.head(5))
    # one_hot для low (3 колонки) + target для high (1 колонка) + num
    assert out.shape == (5, 3 + 1 + 1)
    assert np.isfinite(out).all()


def test_hc_mode_threshold_zero():
    """threshold=0: все непустые колонки считаются high-cardinality."""
    df, y = _hc_df()
    pre = build_preprocessor(
        list(df.columns),
        categorical_features=["low", "high"],
        numerical_features=["num"],
        encoding="one_hot",
        high_cardinality_threshold=0,
        high_cardinality_encoding="frequency",
    )
    pre.fit(df, y)
    cat_transformer = pre.named_transformers_["cat"]
    encoder = cat_transformer.named_steps["encoder"]
    assert encoder.hc_columns_ == [0, 1]
    assert encoder.default_columns_ == []
    out = pre.transform(df.head(5))
    assert out.shape == (5, 1 + 1 + 1)  # 2 frequency + num
    assert np.isfinite(out).all()


def test_hc_mode_threshold_above_all_cardinalities():
    """Порог больше всех кардинальностей: HC-стратегия не применяется."""
    df, y = _hc_df()
    pre = build_preprocessor(
        list(df.columns),
        categorical_features=["low", "high"],
        numerical_features=["num"],
        encoding="one_hot",
        high_cardinality_threshold=10_000,
        high_cardinality_encoding="hashing",
    )
    pre.fit(df, y)
    cat_transformer = pre.named_transformers_["cat"]
    encoder = cat_transformer.named_steps["encoder"]
    assert encoder.hc_columns_ == []
    assert encoder.default_columns_ == [0, 1]
    out = pre.transform(df.head(5))
    # обе колонки one-hot: 3 + 60 + num
    assert out.shape == (5, 63 + 1)


def test_hc_mode_hashing_fixed_dimension():
    """Hashing для HC-колонок не зависит от кардинальности (нет взрывного роста)."""
    df, y = _hc_df()
    pre = build_preprocessor(
        list(df.columns),
        categorical_features=["low", "high"],
        numerical_features=["num"],
        encoding="one_hot",
        high_cardinality_threshold=5,
        high_cardinality_encoding="hashing",
        hashing_n_components=8,
    )
    pre.fit(df, y)
    out = pre.transform(df.head(10))
    # low one-hot (3) + high hashing (8) + num (1)
    assert out.shape == (10, 12)


def test_hc_mode_with_oversampling_ordering():
    """Структурная гарантия: кодирование выполняется до оверсэмплинга
    (шаг preprocessor в пайплайне идёт раньше шага oversampler)."""
    from imblearn.pipeline import Pipeline as ImbPipeline

    df, y = _hc_df()
    pre = build_preprocessor(
        list(df.columns),
        categorical_features=["low", "high"],
        numerical_features=["num"],
        encoding="target",
        high_cardinality_threshold=5,
        high_cardinality_encoding="hashing",
    )
    from configurable_automl_engine.oversampling import DataOversampler

    pipe = ImbPipeline(
        steps=[
            ("preprocessor", pre),
            ("sampler", DataOversampler(algorithm="random", multiplier=1.5)),
            ("model", "passthrough"),
        ]
    )
    step_names = [s[0] for s in pipe.steps]
    assert step_names.index("preprocessor") < step_names.index("sampler")


def test_hc_consistency_invalid_pair():
    """Заданы только threshold или только HC-стратегия -> ValueError."""
    with pytest.raises(ValueError, match="must be set together"):
        build_preprocessor(
            ["cat"],
            categorical_features=["cat"],
            numerical_features=[],
            high_cardinality_threshold=5,
        )
    with pytest.raises(ValueError, match="must be set together"):
        build_preprocessor(
            ["cat"],
            categorical_features=["cat"],
            numerical_features=[],
            high_cardinality_encoding="target",
        )


# ─────────────────── Валидация параметров (FR) ─────────────────────


def test_build_preprocessor_invalid_hc_encoding():
    """Невалидная HC-стратегия -> ValueError."""
    with pytest.raises(ValueError, match="Unknown high_cardinality_encoding"):
        build_preprocessor(
            ["cat"],
            categorical_features=["cat"],
            numerical_features=[],
            high_cardinality_threshold=1,
            high_cardinality_encoding="binary",
        )


def test_build_preprocessor_invalid_hashing_dim():
    """hashing_n_components < 1 -> ValueError."""
    with pytest.raises(ValueError, match="hashing_n_components"):
        build_preprocessor(
            ["cat"],
            categorical_features=["cat"],
            numerical_features=[],
            encoding="hashing",
            hashing_n_components=0,
        )


def test_build_preprocessor_invalid_smoothing():
    """target_encoding_smoothing < 0 -> ValueError."""
    with pytest.raises(ValueError, match="target_encoding_smoothing"):
        build_preprocessor(
            ["cat"],
            categorical_features=["cat"],
            numerical_features=[],
            encoding="target",
            target_encoding_smoothing=-1.0,
        )


def test_build_preprocessor_invalid_threshold():
    """Отрицательный порог -> ValueError."""
    with pytest.raises(ValueError, match="high_cardinality_threshold"):
        build_preprocessor(
            ["cat"],
            categorical_features=["cat"],
            numerical_features=[],
            high_cardinality_threshold=-1,
            high_cardinality_encoding="target",
        )


@pytest.mark.parametrize(
    "enc", ["one_hot", "ordinal", "target", "frequency", "hashing"]
)
def test_build_preprocessor_all_strategies_end_to_end(enc):
    """Все стратегии дают конечную числовую матрицу через препроцессор.

    Для hashing выход разреженный (csr) — ColumnTransformer со стандартным
    sparse_threshold отдаёт csr при любой sparse-части; остальные стратегии
    дают плотный ndarray.
    """
    df, y = _target_df()
    pre = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=["num"],
        encoding=enc,
    )
    out = pre.fit_transform(df, y)
    assert isinstance(pre, ColumnTransformer)
    assert out.shape[0] == len(df)
    if sparse.issparse(out):
        assert out.format == "csr"
        assert np.isfinite(out.data).all()
    else:
        assert np.isfinite(out).all()


def test_split_encoder_feature_names():
    """get_feature_names_out возвращает согласованные имена."""
    df, y = _hc_df()
    pre = build_preprocessor(
        list(df.columns),
        categorical_features=["low", "high"],
        numerical_features=["num"],
        encoding="one_hot",
        high_cardinality_threshold=10,
        high_cardinality_encoding="target",
    )
    pre.fit(df, y)
    cat_transformer = pre.named_transformers_["cat"]
    encoder = cat_transformer.named_steps["encoder"]
    names = encoder.get_feature_names_out(["low", "high"])
    n_expected = encoder.transform(df[["low", "high"]].head(1)).shape[1]
    assert len(names) == n_expected


def test_split_encoder_hashing_feature_names():
    """Имена для hashing-кодирования содержат n_components на колонку."""
    X = np.array([["a", "x"], ["b", "y"]], dtype=object)
    enc = SplitCategoricalEncoder(
        encoding="hashing",
        hashing_n_components=4,
    )
    enc.fit(X)
    names = enc.get_feature_names_out(["c1", "c2"])
    assert len(names) == 8
    assert "c1__h0" in names


# ─────────────── Покрытие краевых веток (coverage) ──────────────────


def test_transformers_accept_1d_input():
    """Одномерный вход приводится к (n, 1) во всех трансформерах."""
    X_1d = np.array(["a", "b", "a"])
    y = np.array([1.0, 2.0, 1.0])

    out_target = TargetEncodingTransformer(smoothing=0.0).fit_transform(X_1d, y)
    assert out_target.shape == (3, 1)

    out_freq = FrequencyEncodingTransformer().fit_transform(X_1d)
    assert out_freq.shape == (3, 1)

    out_hash = HashingEncodingTransformer(n_components=4).fit_transform(X_1d)
    assert out_hash.shape == (3, 4)
    assert sparse.issparse(out_hash)


def test_target_encoding_non_finite_y_rejected():
    """Нефинитные значения в y отклоняются."""
    enc = TargetEncodingTransformer()
    with pytest.raises(ValueError, match="non-finite"):
        enc.fit(np.array([["a"], ["b"]]), np.array([1.0, np.nan]))


def test_frequency_encoding_feature_names():
    """get_feature_names_out frequency-кодирования возвращает имена колонок."""
    X = np.array([["a"], ["b"]], dtype=object)
    enc = FrequencyEncodingTransformer().fit(X)
    names = enc.get_feature_names_out(["col1"])
    assert list(names) == ["col1"]


def test_target_encoding_feature_names():
    """get_feature_names_out target-кодирования возвращает имена колонок."""
    X = np.array([["a"], ["b"]], dtype=object)
    enc = TargetEncodingTransformer().fit(X, np.array([1.0, 2.0]))
    names = enc.get_feature_names_out(["col1"])
    assert list(names) == ["col1"]


def test_feature_names_default_names():
    """Без input_features возвращаются имена-заглушки cat_0, cat_1, ..."""
    enc = FrequencyEncodingTransformer().fit(np.array([["a"], ["b"]]))
    names = enc.get_feature_names_out()
    assert list(names) == ["cat_0"]


def test_feature_names_wrong_length_rejected():
    """get_feature_names_out с неверным числом имён отклоняется."""
    enc = FrequencyEncodingTransformer().fit(np.array([["a"], ["b"]]))
    with pytest.raises(ValueError, match="expected"):
        enc.get_feature_names_out(["col_a", "col_b"])


def test_split_encoder_invalid_strategy_raises_at_fit():
    """Неизвестная стратегия отклоняется в fit().

    sklearn-конвенция: конструктор только сохраняет параметры (никаких
    внутренних энкодеров до обучения), поэтому валидация стратегии
    выполняется при обучении, а не в __init__.
    """
    enc = SplitCategoricalEncoder(encoding="binary")
    assert not hasattr(enc, "default_encoder_")
    with pytest.raises(ValueError, match="Unknown encoding strategy"):
        enc.fit(np.array([["a"], ["b"]], dtype=object))


def test_split_encoder_invalid_hc_strategy_raises_at_fit():
    """Невалидная HC-стратегия отклоняется в fit()."""
    enc = SplitCategoricalEncoder(
        encoding="one_hot",
        high_cardinality_threshold=1,
        high_cardinality_encoding="binary",
    )
    with pytest.raises(ValueError, match="Unknown high_cardinality_encoding"):
        enc.fit(np.array([["a"], ["b"]], dtype=object))


def test_split_encoder_missing_hc_encoding_raises():
    """Порог задан без HC-стратегии -> ValueError в fit.

    Валидация выполняется до разделения колонок по кардинальности,
    поэтому ошибка возникает независимо от фактических данных.
    """
    enc = SplitCategoricalEncoder(
        encoding="one_hot",
        high_cardinality_threshold=1,
        high_cardinality_encoding=None,
    )
    X = np.array([["a"], ["b"], ["c"], ["a"]], dtype=object)  # кардинальность 3 > 1
    with pytest.raises(ValueError, match="high_cardinality_encoding"):
        enc.fit(X)


def test_split_encoder_missing_hc_encoding_raises_without_hc_columns():
    """Порог без HC-стратегии отклоняется и при отсутствии HC-колонок в данных."""
    enc = SplitCategoricalEncoder(
        encoding="one_hot",
        high_cardinality_threshold=10,
        high_cardinality_encoding=None,
    )
    X = np.array([["a"], ["b"], ["c"], ["a"]], dtype=object)  # кардинальность 3 <= 10
    with pytest.raises(ValueError, match="high_cardinality_encoding"):
        enc.fit(X)


def test_split_encoder_empty_parts_transform():
    """transform без колонок возвращает пустую матрицу (n, 0)."""
    enc = SplitCategoricalEncoder(encoding="one_hot")
    X_empty = np.empty((3, 0), dtype=object)
    enc.fit(X_empty)
    out = enc.transform(X_empty)
    assert out.shape == (3, 0)


def test_split_encoder_sparse_aware_hstack():
    """SplitCategoricalEncoder склеивает sparse-части через sp.hstack:
    default one_hot (dense) + HC hashing (sparse) -> csr."""
    X = np.array([["a", "x"], ["b", "y"], ["c", "z"], ["d", "w"]], dtype=object)
    enc = SplitCategoricalEncoder(
        encoding="one_hot",
        high_cardinality_threshold=1,
        high_cardinality_encoding="hashing",
        hashing_n_components=4,
    )
    enc.fit(X)
    # 'a'..'d' -> кардинальность 4 > 1 => HC hashing; 'x'..'w' -> 4 > 1 => HC
    assert enc.hc_columns_ == [0, 1]
    out = enc.transform(X)
    assert sparse.issparse(out)
    assert out.format == "csr"
    assert out.shape == (4, 8)
    # nnz = n_rows * n_hc_cols (только hashing-части разреженные)
    assert out.nnz == 8


def test_split_encoder_pure_dense_stays_dense():
    """Без sparse-частей (one_hot/ordinal/target/frequency) выход плотный."""
    X = np.array([["a", "x"], ["b", "y"]], dtype=object)
    enc = SplitCategoricalEncoder(encoding="target")
    enc.fit(X, np.array([1.0, 2.0]))
    out = enc.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)


def test_split_encoder_set_params_changes_behavior():
    """set_params меняет поведение кодирования после повторного fit.

    Внутренние энкодеры создаются в fit(), поэтому базовая реализация
    set_params() корректно применяется при следующем обучении — кастомное
    переопределение больше не требуется.
    """
    X = np.array([["a"], ["b"], ["a"]], dtype=object)
    y = np.array([1.0, 2.0, 1.0])

    enc = SplitCategoricalEncoder(encoding="one_hot")
    # one_hot: 2 бинарные колонки на 2 категории
    out = enc.fit_transform(X, y)
    assert out.shape == (3, 2)

    # set_params(encoding='target') + повторный fit: 1 колонка на входную.
    enc.set_params(encoding="target")
    out = enc.fit_transform(X, y)
    assert out.shape == (3, 1)
    assert isinstance(enc.default_encoder_, TargetEncodingTransformer)
    assert not isinstance(enc.default_encoder_, OneHotEncoder)

    # HC-режим: кардинальность 2 > порога 1 => колонка идёт в hashing.
    enc.set_params(
        encoding="one_hot",
        high_cardinality_threshold=1,
        high_cardinality_encoding="hashing",
        hashing_n_components=4,
    )
    out = enc.fit_transform(X, y)
    assert out.shape == (3, 4)  # только hashing: n_components=4
    assert isinstance(enc.hc_encoder_, HashingEncodingTransformer)

    # Отключение HC-режима возвращает поведение one_hot.
    enc.set_params(high_cardinality_threshold=None, high_cardinality_encoding=None)
    out = enc.fit_transform(X, y)
    assert out.shape == (3, 2)
    assert enc.hc_encoder_ is None


def test_split_encoder_get_params_only_constructor_args():
    """get_params() возвращает только параметры конструктора (никакого состояния)."""
    enc = SplitCategoricalEncoder(encoding="ordinal", high_cardinality_threshold=5)
    params = enc.get_params()
    assert params["encoding"] == "ordinal"
    assert params["high_cardinality_threshold"] == 5
    assert params["high_cardinality_encoding"] is None
    # До fit() обученных атрибутов нет (sklearn-конвенция).
    assert not hasattr(enc, "default_encoder_")
    assert not hasattr(enc, "hc_encoder_")
    assert not hasattr(enc, "default_columns_")


def test_split_encoder_clone_is_independent():
    """clone() создаёт независимый экземпляр без обученного состояния."""
    enc = SplitCategoricalEncoder(
        encoding="one_hot",
        high_cardinality_threshold=2,
        high_cardinality_encoding="hashing",
        hashing_n_components=8,
        random_state=7,
    )
    cloned = clone(enc)
    assert cloned is not enc
    assert cloned.get_params() == enc.get_params()
    # Клон не несёт обученное состояние оригинала.
    assert not hasattr(cloned, "default_encoder_")
    assert not hasattr(cloned, "hc_encoder_")

    X = np.array([["a"], ["b"], ["a"]], dtype=object)
    y = np.array([1.0, 2.0, 1.0])
    cloned.fit(X, y)
    assert hasattr(cloned, "default_encoder_")
    # Обучение клона не влияет на оригинал.
    assert not hasattr(enc, "default_encoder_")


def test_split_encoder_pipeline_set_params_propagates():
    """set_params с pipeline-префиксом корректно меняет поведение энкодера."""
    X = np.array([["a"], ["b"], ["a"]], dtype=object)
    y = np.array([1.0, 2.0, 1.0])
    pipe = Pipeline(steps=[("encoder", SplitCategoricalEncoder(encoding="one_hot"))])

    pipe.set_params(encoder__encoding="target")
    out = pipe.fit_transform(X, y)
    assert out.shape == (3, 1)  # target: 1 колонка на входную
