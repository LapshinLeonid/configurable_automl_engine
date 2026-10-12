"""Unit tests for the new categorical feature encoding strategies.

Covers:
    * target encoding (smoothing, fallback, no data leakage);
    * frequency encoding (frequencies, determinism, unknown -> 0.0);
    * hashing encoding (fixed dimensionality, determinism, unknown);
    * automatic high-cardinality mode (threshold, threshold=0, edge cases);
    * build_preprocessor parameter validation;
    * edge cases: a single category, an empty column, NaN;
    * category type preservation in target encoding: int/float/bool
      standalone, int-fit/float-transform equivalence, consistency with
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
    """Synthetic dataset with a categorical column and a numeric target."""
    rng = np.random.default_rng(seed)
    cat = rng.choice(["a", "b", "c"], size=n)
    df = pd.DataFrame({"cat": cat, "num": rng.normal(size=n)})
    y = pd.Series(
        pd.Series(cat).map({"a": 1.0, "b": 2.0, "c": 3.0}).to_numpy()
        + rng.normal(0, 0.05, n)
    )
    return df, y


def test_target_encoding_smoothing_formula():
    """The encoded category value equals the smoothing formula.

    The formula is checked via fit + transform (full statistics); the
    fit_transform method performs cross-fitting and intentionally does NOT
    return category means that include the row's own y.
    """
    X = np.array([["a"], ["a"], ["b"], ["b"], ["b"], ["c"]], dtype=object)
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    out = enc.fit(X, y).transform(X)
    # With m=0 — plain category mean
    assert out[0, 0] == pytest.approx(10.0)
    assert out[2, 0] == pytest.approx(20.0)
    assert out[5, 0] == pytest.approx(30.0)

    global_mean = float(np.mean(y))  # (10+10+20+20+20+30)/6 = 18.333...
    enc_smooth = TargetEncodingTransformer(smoothing=100.0)
    out_smooth = enc_smooth.fit(X, y).transform(X)
    # With a very large m the value is close to the global mean
    assert out_smooth[0, 0] == pytest.approx(global_mean, abs=0.5)


def test_target_encoding_fallback_global_mean():
    """Unknown category -> global mean (default fallback)."""
    X_train = np.array([["a"], ["a"], ["b"]], dtype=object)
    y_train = np.array([1.0, 1.0, 5.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    enc.fit(X_train, y_train)
    out = enc.transform(np.array([["unknown"], ["a"]], dtype=object))
    assert out[0, 0] == pytest.approx(float(np.mean(y_train)))
    assert out[1, 0] == pytest.approx(1.0)


def test_target_encoding_custom_fallback():
    """A custom fallback is used for unknown categories."""
    X_train = np.array([["a"], ["a"], ["b"]], dtype=object)
    y_train = np.array([1.0, 1.0, 5.0])
    enc = TargetEncodingTransformer(smoothing=0.0, fallback=-99.0)
    enc.fit(X_train, y_train)
    out = enc.transform(np.array([["unknown"]], dtype=object))
    assert out[0, 0] == pytest.approx(-99.0)


def test_target_encoding_no_leakage():
    """Statistics are computed only on the training part: a category seen only in
    validation gets the fallback, not its validation mean."""
    X_train = np.array([["a"], ["a"], ["b"]], dtype=object)
    y_train = np.array([1.0, 1.0, 5.0])
    X_val = np.array([["a"], ["only_val"]], dtype=object)
    y_val = np.array([999.0, 999.0])  # if it leaked — we would get 999

    enc = TargetEncodingTransformer(smoothing=0.0)
    enc.fit(X_train, y_train)
    out = enc.transform(X_val)
    # 'a' -> 1.0 (train mean), 'only_val' -> global train mean (not 999)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(float(np.mean(y_train)), abs=1e-9)


def test_target_encoding_fit_transform_cross_fit_no_leakage():
    """Regression: fit_transform does not leak the row's own y.

    Scenario: the categorical column is pure noise (about 2 rows per
    category) and y is unrelated to the column. With leakage (fit ->
    transform on the same rows) the correlation of the encoded feature with
    y on training is ~0.6-0.7; with cross-fitting it should be close to zero.
    """
    rng = np.random.default_rng(42)
    n = 2000
    n_categories = n // 2  # ~2 rows per category
    X = np.array(
        [f"c{i % n_categories}" for i in range(n)], dtype=object
    ).reshape(-1, 1)
    y = rng.normal(size=n)  # noise, unrelated to the category

    enc = TargetEncodingTransformer(smoothing=20.0)
    encoded = enc.fit_transform(X, y)[:, 0]

    corr_cross_fit = float(np.corrcoef(encoded, y)[0, 1])
    assert abs(corr_cross_fit) < 0.1, corr_cross_fit

    # Control: naive encoding (fit + transform on the same rows)
    # intentionally correlates with y — this is exactly the leakage that
    # cross-fitting removes.
    naive = TargetEncodingTransformer(smoothing=20.0).fit(X, y).transform(X)[:, 0]
    corr_naive = float(np.corrcoef(naive, y)[0, 1])
    assert corr_naive > 0.4, corr_naive


def test_target_encoding_fit_transform_keeps_full_statistics_for_transform():
    """After fit_transform the encoder keeps full statistics: transform on the
    same rows is equivalent to fit + transform (new data is encoded without
    cross-fitting)."""
    X = np.array([["a"], ["a"], ["b"], ["b"], ["b"], ["c"]], dtype=object)
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    out_cross = enc.fit_transform(X, y)
    # 'c' appears once: in cross-fitting the row is encoded without its own y
    # (fallback of the training fold), not with the mean of 30.0.
    assert out_cross[5, 0] != pytest.approx(30.0)

    full = enc.transform(X)
    naive = TargetEncodingTransformer(smoothing=0.0).fit(X, y).transform(X)
    np.testing.assert_allclose(full, naive)
    assert full[5, 0] == pytest.approx(30.0)


def test_target_encoding_fit_transform_small_sample_leave_one_out():
    """Fewer rows than cv: fit_transform works (leave-one-out) and does not
    leak the row's own y."""
    X = np.array([["x"], ["y"], ["z"]], dtype=object)
    y = np.array([1.0, 2.0, 3.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    out = enc.fit_transform(X, y)
    # Each row is encoded with the mean of the other two (all categories are
    # different, so the encoding equals the LOO mean of the other y's).
    assert out[:, 0].tolist() == pytest.approx([2.5, 2.0, 1.5], rel=1e-6)


def test_target_encoding_fit_transform_single_row():
    """A single row is encoded with full statistics without failing."""
    enc = TargetEncodingTransformer()
    out = enc.fit_transform(np.array([["a"]]), np.array([5.0]))
    assert out.shape == (1, 1)
    assert np.isfinite(out).all()


def test_target_encoding_fit_transform_invalid_cv():
    """cv < 2 is rejected in fit_transform with an explicit error."""
    enc = TargetEncodingTransformer(cv=1)
    with pytest.raises(ValueError, match="cv must be >= 2"):
        enc.fit_transform(np.array([["a"], ["b"]]), np.array([1.0, 2.0]))


def test_target_encoding_fit_transform_deterministic():
    """Cross-fitting is deterministic with a fixed random_state."""
    rng = np.random.default_rng(3)
    n = 300
    X = np.array([f"k{i % 40}" for i in range(n)], dtype=object).reshape(-1, 1)
    y = rng.normal(size=n)
    out1 = TargetEncodingTransformer().fit_transform(X, y)
    out2 = TargetEncodingTransformer().fit_transform(X, y)
    np.testing.assert_array_equal(out1, out2)


def test_split_encoder_fit_transform_target_cross_fit_no_leakage():
    """SplitCategoricalEncoder.fit_transform delegates to the internal encoders'
    fit_transform: target encoding through the split encoder (including the
    HC branch) does not leak the row's own y (regression)."""
    rng = np.random.default_rng(7)
    n = 2000
    n_categories = n // 2  # ~2 rows per category
    df = pd.DataFrame(
        {
            "low": rng.choice(["a", "b", "c"], size=n),
            "high": [f"h{i % n_categories}" for i in range(n)],
        }
    )
    y = rng.normal(size=n)

    pre = build_preprocessor(
        list(df.columns),
        categorical_features=["low", "high"],
        numerical_features=[],
        encoding="target",
        high_cardinality_threshold=50,
        high_cardinality_encoding="target",
    )
    encoded = pre.fit_transform(df, y)
    # Column order: default ('low'), then HC ('high').
    high_encoded = np.asarray(encoded[:, 1]).ravel()
    corr = float(np.corrcoef(high_encoded, y)[0, 1])
    assert abs(corr) < 0.1, corr
    # transform (new data) uses full statistics, not OOF values.
    full = pre.transform(df)
    assert np.isfinite(full).all()


def test_target_encoding_requires_y():
    """fit() without y raises ValueError."""
    enc = TargetEncodingTransformer()
    with pytest.raises(ValueError, match="requires y"):
        enc.fit(np.array([["a"], ["b"]], dtype=object))


def test_target_encoding_non_numeric_y():
    """A non-numeric target is rejected."""
    enc = TargetEncodingTransformer()
    with pytest.raises(ValueError, match="numeric target"):
        enc.fit(np.array([["a"], ["b"]], dtype=object), np.array(["x", "y"]))


def test_target_encoding_end_to_end_shape():
    """End-to-end target encoding through the preprocessor: 1 column per category."""
    df, y = _target_df()
    pre = build_preprocessor(
        list(df.columns),
        categorical_features=["cat"],
        numerical_features=["num"],
        encoding="target",
    )
    out = pre.fit_transform(df, y)
    assert out.shape == (len(df), 2)  # target-encoded cat + num
    assert np.isfinite(out).all()


def test_target_encoding_nan_column():
    """NaN in a categorical column does not break target encoding (most_frequent
    imputation fills the gaps before encoding)."""
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
    """A column with a single unique category is encoded without failure."""
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
    # One category: transform (new data) gives the global mean — full
    # statistics are fixed on the encoder even after cross-fitting.
    assert pre.transform(df)[0, 0] == pytest.approx(2.0)


def test_target_encoding_all_missing_column():
    """A fully missing categorical column is encoded after imputation (all values
    become a single string category)."""
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
    """A rare category with strong smoothing is close to the global mean."""
    X = np.array([["common"]] * 100 + [["rare"]], dtype=object)
    y = np.array([0.0] * 100 + [100.0])
    enc = TargetEncodingTransformer(smoothing=20.0)
    enc.fit(X, y)
    out = enc.transform(np.array([["rare"]], dtype=object))
    # Formula: (mean_cat * n + global_mean * m) / (n + m)
    global_mean = float(np.mean(y))  # 100/101
    expected = (100.0 * 1 + global_mean * 20.0) / (1 + 20.0)
    assert out[0, 0] == pytest.approx(expected, rel=1e-6)


# ─── Target encoding: category type preservation ───


def test_target_encoding_int_categories_standalone():
    """int categories standalone are encoded with their own means, not the fallback.

    Regression: previously the statistic keys were converted to
    strings, so int codes in transform() found no matches (everything silently
    fell back).
    """
    X = np.array([[1], [1], [2], [2], [2], [3]], dtype=np.int64)
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    out = enc.fit(X, y).transform(X)
    assert out[:, 0].tolist() == pytest.approx(
        [10.0, 10.0, 20.0, 20.0, 20.0, 30.0], rel=1e-6
    )
    # Statistic keys keep their original numeric type (are not normalized to strings)
    assert not any(isinstance(k, str) for k in enc.statistics_[0].keys())
    assert all(isinstance(k, (int, np.integer)) for k in enc.statistics_[0].keys())


def test_target_encoding_float_categories_standalone():
    """float categories standalone are encoded with their own means, not the fallback."""
    X = np.array([[1.0], [1.0], [2.0], [2.0], [2.0], [3.0]])
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    out = enc.fit(X, y).transform(X)
    assert out[:, 0].tolist() == pytest.approx(
        [10.0, 10.0, 20.0, 20.0, 20.0, 30.0], rel=1e-6
    )


def test_target_encoding_bool_categories_standalone():
    """bool categories standalone are encoded with means per True/False."""
    X = np.array([[True], [True], [False], [False], [False], [True]])
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    out = enc.fit(X, y).transform(X)
    true_mean = float(np.mean(y[[0, 1, 5]]))  # (10+10+30)/3 = 16.67
    false_mean = float(np.mean(y[[2, 3, 4]]))  # 20.0
    assert out[:, 0].tolist() == pytest.approx(
        [true_mean, true_mean, false_mean, false_mean, false_mean, true_mean],
        rel=1e-6,
    )


def test_target_encoding_bool_int_mixing_documented_semantics():
    """Mixing bool and int 0/1 gives the documented behavior (fallback).

    pandas matches by dtype: ``bool`` is a separate dtype, so bool statistic
    keys do not match int values (and vice versa), and all values fall back.
    This pins down the warning from the class docstring: the categories of a
    single feature must keep one dtype between ``fit`` and ``transform``.
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
    """Encoding fitted on int codes is applicable to float values.

    pandas matches by dtype-compatible keys: int and float are interchangeable.
    """
    X = np.array([[1], [1], [2], [2], [2], [3]], dtype=np.int64)
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    enc.fit(X, y)
    out = enc.transform(np.array([[1.0], [2.0], [3.0]]))
    assert out[:, 0].tolist() == pytest.approx([10.0, 20.0, 30.0], rel=1e-6)


def test_target_encoding_float_fit_int_transform_equivalence():
    """Reverse compatibility: fit on float, transform on int codes."""
    X = np.array([[1.0], [1.0], [2.0], [2.0], [2.0], [3.0]])
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    enc.fit(X, y)
    out = enc.transform(np.array([[1], [2], [3]], dtype=np.int64))
    assert out[:, 0].tolist() == pytest.approx([10.0, 20.0, 30.0], rel=1e-6)


def test_target_encoding_key_types_match_frequency_encoding():
    """Target and Frequency transformers preserve key types the same way.

    Consistency between neighboring classes: both store statistic
    keys in the original category type, so an int column is encoded correctly
    in both.
    """
    X = np.array([[1], [1], [2], [2], [2], [3]], dtype=np.int64)
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    target = TargetEncodingTransformer(smoothing=0.0).fit(X, y)
    freq = FrequencyEncodingTransformer().fit(X)
    target_keys = set(target.statistics_[0].keys())
    freq_keys = set(freq.frequencies_[0].keys())
    assert target_keys == freq_keys == {1, 2, 3}
    assert not any(isinstance(k, str) for k in target_keys | freq_keys)
    # Both correctly match int codes (not the fallback)
    assert target.transform(np.array([[3]], dtype=np.int64))[0, 0] == pytest.approx(30.0)
    assert freq.transform(np.array([[3]], dtype=np.int64))[0, 0] == pytest.approx(1 / 6)


def test_target_encoding_numeric_unknown_falls_back():
    """A numeric category absent from training -> fallback."""
    X = np.array([[1], [1], [2], [2], [2], [3]], dtype=np.int64)
    y = np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    enc.fit(X, y)
    out = enc.transform(np.array([[42], [1]], dtype=np.int64))
    assert out[0, 0] == pytest.approx(float(np.mean(y)))  # unknown -> fallback
    assert out[1, 0] == pytest.approx(10.0)  # known one is encoded as usual


def test_target_encoding_nan_dropped_in_fit_and_fallback_in_transform():
    """NaN standalone: in fit the NaN group is dropped, in transform -> fallback."""
    X = np.array([[1.0], [1.0], [np.nan], [2.0], [2.0]])
    y = np.array([10.0, 10.0, 20.0, 20.0, 30.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    enc.fit(X, y)
    assert set(enc.statistics_[0].keys()) == {1.0, 2.0}  # NaN group dropped
    out = enc.transform(np.array([[np.nan], [1.0]]))
    assert out[0, 0] == pytest.approx(float(np.mean(y)))  # NaN -> fallback
    assert out[1, 0] == pytest.approx(10.0)


def test_target_encoding_single_int_category_standalone():
    """A single int category standalone: all values = global mean."""
    X = np.array([[7], [7], [7]], dtype=np.int64)
    y = np.array([1.0, 2.0, 3.0])
    enc = TargetEncodingTransformer(smoothing=0.0)
    out = enc.fit(X, y).transform(X)
    assert out[:, 0].tolist() == pytest.approx([2.0, 2.0, 2.0], rel=1e-6)


# ─────────────────────── Frequency encoding ────────────────────────


def test_frequency_encoding_values():
    """Frequencies are computed as count / n_rows."""
    X = np.array([["a"], ["a"], ["b"], ["c"], ["c"], ["c"]], dtype=object)
    enc = FrequencyEncodingTransformer()
    out = enc.fit_transform(X)
    assert out[0, 0] == pytest.approx(2 / 6)
    assert out[2, 0] == pytest.approx(1 / 6)
    assert out[3, 0] == pytest.approx(3 / 6)
    # Sum of frequencies of unique categories = 1
    assert np.unique(out).sum() == pytest.approx(1.0)


def test_frequency_encoding_unknown_is_zero():
    """Unknown category -> 0.0 (deterministic fallback)."""
    X_train = np.array([["a"], ["a"], ["b"]], dtype=object)
    enc = FrequencyEncodingTransformer()
    enc.fit(X_train)
    out = enc.transform(np.array([["zzz"], ["a"]], dtype=object))
    assert out[0, 0] == 0.0
    assert out[1, 0] == pytest.approx(2 / 3)


def test_frequency_encoding_deterministic():
    """Frequency encoding is deterministic: equal inputs -> equal outputs."""
    X = np.array([["a"], ["b"], ["a"]], dtype=object)
    enc1 = FrequencyEncodingTransformer().fit(X)
    enc2 = FrequencyEncodingTransformer().fit(X)
    out1 = enc1.transform(X)
    out2 = enc2.transform(X)
    np.testing.assert_array_equal(out1, out2)


def test_frequency_encoding_nan_handled():
    """NaN is converted to the string 'nan' and handled as a regular category."""
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
    """The output dimensionality is fixed: n_components per input column."""
    X = np.array([["a"], ["b"], ["c"]], dtype=object)
    enc = HashingEncodingTransformer(n_components=8)
    out = enc.fit_transform(X)
    assert out.shape == (3, 8)
    # The output is sparse (csr), like sklearn's FeatureHasher — a dense float64
    # matrix of size (n_rows, n_cols * n_components) would cause OOM.
    assert sparse.issparse(out)
    assert out.format == "csr"
    # nnz = n_rows * n_cols, values are binary (1.0)
    assert out.nnz == 3
    assert set(np.unique(out.data)).issubset({1.0})


def test_hashing_encoding_multiple_columns():
    """For multiple columns the dimensionality multiplies."""
    X = np.array([["a", "x"], ["b", "y"]], dtype=object)
    enc = HashingEncodingTransformer(n_components=4)
    out = enc.fit_transform(X)
    assert out.shape == (2, 8)
    assert sparse.issparse(out)
    assert out.nnz == 4


def test_hashing_encoding_deterministic():
    """Hashing is deterministic with fixed settings and seed."""
    X = np.array([["alpha"], ["beta"], ["gamma"], ["delta"]], dtype=object)
    enc1 = HashingEncodingTransformer(n_components=32, random_state=42)
    enc2 = HashingEncodingTransformer(n_components=32, random_state=42)
    out1 = enc1.fit_transform(X)
    out2 = enc2.fit_transform(X)
    assert (out1 != out2).nnz == 0


def test_hashing_encoding_seed_changes_output():
    """Different seeds give different projections (for the typical case)."""
    X = np.array([[f"cat_{i}"] for i in range(64)], dtype=object)
    out1 = HashingEncodingTransformer(n_components=64, random_state=1).fit_transform(X)
    out2 = HashingEncodingTransformer(n_components=64, random_state=2).fit_transform(X)
    assert (out1 != out2).nnz > 0


def test_hashing_encoding_unknown_category():
    """An unknown category is hashed deterministically (without failures)."""
    X_train = np.array([["a"], ["b"], ["c"]], dtype=object)
    enc = HashingEncodingTransformer(n_components=16, random_state=42)
    enc.fit(X_train)
    out = enc.transform(np.array([["never_seen"], ["a"]], dtype=object))
    assert out.shape == (2, 16)
    assert sparse.issparse(out)
    # Repeating transform on an unknown category gives the same result
    out2 = enc.transform(np.array([["never_seen"]], dtype=object))
    assert (out[0] != out2[0]).nnz == 0


def test_hashing_encoding_binary_indicators():
    """Each row has exactly one active column per input column."""
    X = np.array([["a"], ["b"]], dtype=object)
    enc = HashingEncodingTransformer(n_components=16)
    out = enc.fit_transform(X)
    assert np.asarray(out.sum(axis=1)).ravel().tolist() == [1.0, 1.0]
    # Binary values: every matrix element is 0 or 1
    dense = out.toarray()
    assert set(np.unique(dense)).issubset({0.0, 1.0})


# ───────────────────── High-cardinality mode ──────────────────────


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
    """At the threshold the columns split: low -> default, high -> HC strategy."""
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
    # one_hot for low (3 columns) + target for high (1 column) + num
    assert out.shape == (5, 3 + 1 + 1)
    assert np.isfinite(out).all()


def test_hc_mode_threshold_zero():
    """threshold=0: all non-empty columns are treated as high-cardinality."""
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
    """Threshold above all cardinalities: the HC strategy is not applied."""
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
    # both columns one-hot: 3 + 60 + num
    assert out.shape == (5, 63 + 1)


def test_hc_mode_hashing_fixed_dimension():
    """Hashing for HC columns does not depend on cardinality (no blow-up)."""
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
    """Structural guarantee: encoding runs before oversampling
    (the preprocessor step in the pipeline comes before the oversampler step)."""
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
    """Only threshold or only the HC strategy is set -> ValueError."""
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


# ─────────────────── Parameter validation (FR) ────────────────────


def test_build_preprocessor_invalid_hc_encoding():
    """Invalid HC strategy -> ValueError."""
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
    """Negative threshold -> ValueError."""
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
    """All strategies produce a finite numeric matrix through the preprocessor.

    For hashing the output is sparse (csr) — ColumnTransformer with the default
    sparse_threshold returns csr whenever any part is sparse; the other
    strategies produce a dense ndarray.
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
    """get_feature_names_out returns consistent names."""
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
    """Names for hashing encoding contain n_components per column."""
    X = np.array([["a", "x"], ["b", "y"]], dtype=object)
    enc = SplitCategoricalEncoder(
        encoding="hashing",
        hashing_n_components=4,
    )
    enc.fit(X)
    names = enc.get_feature_names_out(["c1", "c2"])
    assert len(names) == 8
    assert "c1__h0" in names


# ─────────────── Edge-case branch coverage ────────────────────────


def test_transformers_accept_1d_input():
    """A 1-D input is reshaped to (n, 1) in all transformers."""
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
    """Non-finite values in y are rejected."""
    enc = TargetEncodingTransformer()
    with pytest.raises(ValueError, match="non-finite"):
        enc.fit(np.array([["a"], ["b"]]), np.array([1.0, np.nan]))


def test_frequency_encoding_feature_names():
    """get_feature_names_out of frequency encoding returns column names."""
    X = np.array([["a"], ["b"]], dtype=object)
    enc = FrequencyEncodingTransformer().fit(X)
    names = enc.get_feature_names_out(["col1"])
    assert list(names) == ["col1"]


def test_target_encoding_feature_names():
    """get_feature_names_out of target encoding returns column names."""
    X = np.array([["a"], ["b"]], dtype=object)
    enc = TargetEncodingTransformer().fit(X, np.array([1.0, 2.0]))
    names = enc.get_feature_names_out(["col1"])
    assert list(names) == ["col1"]


def test_feature_names_default_names():
    """Without input_features, placeholder names cat_0, cat_1, ... are returned."""
    enc = FrequencyEncodingTransformer().fit(np.array([["a"], ["b"]]))
    names = enc.get_feature_names_out()
    assert list(names) == ["cat_0"]


def test_feature_names_wrong_length_rejected():
    """get_feature_names_out with a wrong number of names is rejected."""
    enc = FrequencyEncodingTransformer().fit(np.array([["a"], ["b"]]))
    with pytest.raises(ValueError, match="expected"):
        enc.get_feature_names_out(["col_a", "col_b"])


def test_split_encoder_invalid_strategy_raises_at_fit():
    """An unknown strategy is rejected in fit().

    sklearn convention: the constructor only stores parameters (no internal
    encoders before fitting), so strategy validation happens during fitting,
    not in __init__.
    """
    enc = SplitCategoricalEncoder(encoding="binary")
    assert not hasattr(enc, "default_encoder_")
    with pytest.raises(ValueError, match="Unknown encoding strategy"):
        enc.fit(np.array([["a"], ["b"]], dtype=object))


def test_split_encoder_invalid_hc_strategy_raises_at_fit():
    """An invalid HC strategy is rejected in fit()."""
    enc = SplitCategoricalEncoder(
        encoding="one_hot",
        high_cardinality_threshold=1,
        high_cardinality_encoding="binary",
    )
    with pytest.raises(ValueError, match="Unknown high_cardinality_encoding"):
        enc.fit(np.array([["a"], ["b"]], dtype=object))


def test_split_encoder_missing_hc_encoding_raises():
    """Threshold set without an HC strategy -> ValueError in fit.

    Validation runs before columns are split by cardinality, so the error
    occurs regardless of the actual data.
    """
    enc = SplitCategoricalEncoder(
        encoding="one_hot",
        high_cardinality_threshold=1,
        high_cardinality_encoding=None,
    )
    X = np.array([["a"], ["b"], ["c"], ["a"]], dtype=object)  # cardinality 3 > 1
    with pytest.raises(ValueError, match="high_cardinality_encoding"):
        enc.fit(X)


def test_split_encoder_missing_hc_encoding_raises_without_hc_columns():
    """Threshold without an HC strategy is rejected even when no HC columns are present in the data."""
    enc = SplitCategoricalEncoder(
        encoding="one_hot",
        high_cardinality_threshold=10,
        high_cardinality_encoding=None,
    )
    X = np.array([["a"], ["b"], ["c"], ["a"]], dtype=object)  # cardinality 3 <= 10
    with pytest.raises(ValueError, match="high_cardinality_encoding"):
        enc.fit(X)


def test_split_encoder_empty_parts_transform():
    """transform without columns returns an empty matrix (n, 0)."""
    enc = SplitCategoricalEncoder(encoding="one_hot")
    X_empty = np.empty((3, 0), dtype=object)
    enc.fit(X_empty)
    out = enc.transform(X_empty)
    assert out.shape == (3, 0)


def test_split_encoder_fit_transform_empty_columns():
    """fit_transform with zero columns returns an empty matrix, and the
    internal encoders are created even in that case (review note)."""
    X_empty = np.empty((3, 0), dtype=object)
    y = np.array([1.0, 2.0, 3.0])

    enc = SplitCategoricalEncoder(encoding="target")
    out = enc.fit_transform(X_empty, y)
    assert out.shape == (3, 0)
    # default_encoder_ is created in _prepare despite the absence of columns.
    assert isinstance(enc.default_encoder_, TargetEncodingTransformer)
    assert enc.transform(X_empty).shape == (3, 0)

    # HC mode: hc_encoder_ is also created with an empty column set.
    enc_hc = SplitCategoricalEncoder(
        encoding="one_hot",
        high_cardinality_threshold=2,
        high_cardinality_encoding="target",
    )
    out_hc = enc_hc.fit_transform(X_empty, y)
    assert out_hc.shape == (3, 0)
    assert isinstance(enc_hc.hc_encoder_, TargetEncodingTransformer)


def test_split_encoder_sparse_aware_hstack():
    """SplitCategoricalEncoder concatenates sparse parts via sp.hstack:
    default one_hot (dense) + HC hashing (sparse) -> csr."""
    X = np.array([["a", "x"], ["b", "y"], ["c", "z"], ["d", "w"]], dtype=object)
    enc = SplitCategoricalEncoder(
        encoding="one_hot",
        high_cardinality_threshold=1,
        high_cardinality_encoding="hashing",
        hashing_n_components=4,
    )
    enc.fit(X)
    # 'a'..'d' -> cardinality 4 > 1 => HC hashing; 'x'..'w' -> 4 > 1 => HC
    assert enc.hc_columns_ == [0, 1]
    out = enc.transform(X)
    assert sparse.issparse(out)
    assert out.format == "csr"
    assert out.shape == (4, 8)
    # nnz = n_rows * n_hc_cols (only the hashing parts are sparse)
    assert out.nnz == 8


def test_split_encoder_pure_dense_stays_dense():
    """Without sparse parts (one_hot/ordinal/target/frequency) the output is dense."""
    X = np.array([["a", "x"], ["b", "y"]], dtype=object)
    enc = SplitCategoricalEncoder(encoding="target")
    enc.fit(X, np.array([1.0, 2.0]))
    out = enc.transform(X)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)


def test_split_encoder_set_params_changes_behavior():
    """set_params changes the encoding behavior after a subsequent fit.

    Internal encoders are created in fit(), so the base set_params()
    implementation applies correctly at the next fit — no custom override
    is needed anymore.
    """
    X = np.array([["a"], ["b"], ["a"]], dtype=object)
    y = np.array([1.0, 2.0, 1.0])

    enc = SplitCategoricalEncoder(encoding="one_hot")
    # one_hot: 2 binary columns for 2 categories
    out = enc.fit_transform(X, y)
    assert out.shape == (3, 2)

    # set_params(encoding='target') + refit: 1 column per input.
    enc.set_params(encoding="target")
    out = enc.fit_transform(X, y)
    assert out.shape == (3, 1)
    assert isinstance(enc.default_encoder_, TargetEncodingTransformer)
    assert not isinstance(enc.default_encoder_, OneHotEncoder)

    # HC mode: cardinality 2 > threshold 1 => the column goes to hashing.
    enc.set_params(
        encoding="one_hot",
        high_cardinality_threshold=1,
        high_cardinality_encoding="hashing",
        hashing_n_components=4,
    )
    out = enc.fit_transform(X, y)
    assert out.shape == (3, 4)  # only hashing: n_components=4
    assert isinstance(enc.hc_encoder_, HashingEncodingTransformer)

    # Disabling HC mode restores the one_hot behavior.
    enc.set_params(high_cardinality_threshold=None, high_cardinality_encoding=None)
    out = enc.fit_transform(X, y)
    assert out.shape == (3, 2)
    assert enc.hc_encoder_ is None


def test_split_encoder_get_params_only_constructor_args():
    """get_params() returns only constructor parameters (no state)."""
    enc = SplitCategoricalEncoder(encoding="ordinal", high_cardinality_threshold=5)
    params = enc.get_params()
    assert params["encoding"] == "ordinal"
    assert params["high_cardinality_threshold"] == 5
    assert params["high_cardinality_encoding"] is None
    # Before fit() there are no fitted attributes (sklearn convention).
    assert not hasattr(enc, "default_encoder_")
    assert not hasattr(enc, "hc_encoder_")
    assert not hasattr(enc, "default_columns_")


def test_split_encoder_clone_is_independent():
    """clone() creates an independent instance without fitted state."""
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
    # The clone does not carry the original's fitted state.
    assert not hasattr(cloned, "default_encoder_")
    assert not hasattr(cloned, "hc_encoder_")

    X = np.array([["a"], ["b"], ["a"]], dtype=object)
    y = np.array([1.0, 2.0, 1.0])
    cloned.fit(X, y)
    assert hasattr(cloned, "default_encoder_")
    # Fitting the clone does not affect the original.
    assert not hasattr(enc, "default_encoder_")


def test_split_encoder_pipeline_set_params_propagates():
    """set_params with a pipeline prefix correctly changes the encoder behavior."""
    X = np.array([["a"], ["b"], ["a"]], dtype=object)
    y = np.array([1.0, 2.0, 1.0])
    pipe = Pipeline(steps=[("encoder", SplitCategoricalEncoder(encoding="one_hot"))])

    pipe.set_params(encoder__encoding="target")
    out = pipe.fit_transform(X, y)
    assert out.shape == (3, 1)  # target: 1 column per input
