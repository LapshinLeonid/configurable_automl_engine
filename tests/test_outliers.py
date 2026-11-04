"""Unit tests for the standalone Winsorizer transformer (preprocessing.outliers)."""

from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from configurable_automl_engine.preprocessing import Winsorizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantile_reference(
    column: np.ndarray, q_lo: float, q_hi: float
) -> tuple[float, float]:
    """NumPy reference bounds for the 'quantile' method."""
    return (
        float(np.nanquantile(column, q_lo)),
        float(np.nanquantile(column, q_hi)),
    )


def _iqr_reference(column: np.ndarray, multiplier: float) -> tuple[float, float]:
    """NumPy reference bounds for the 'iqr' method."""
    q1 = np.nanquantile(column, 0.25)
    q3 = np.nanquantile(column, 0.75)
    iqr = q3 - q1
    return float(q1 - multiplier * iqr), float(q3 + multiplier * iqr)


# ---------------------------------------------------------------------------
# Bounds match a NumPy reference
# ---------------------------------------------------------------------------


def test_bounds_match_numpy_reference_quantile():
    """The 'quantile' bounds equal a direct np.nanquantile computation."""
    rng = np.random.RandomState(42)
    X = rng.normal(size=(200, 3))
    X[5, 1] = np.nan
    X[10, 2] = np.nan
    X[11, 2] = np.nan

    wins = Winsorizer(method="quantile", lower_quantile=0.05, upper_quantile=0.95)
    wins.fit(X)

    for col in range(3):
        ref_lo, ref_hi = _quantile_reference(X[:, col], 0.05, 0.95)
        assert wins.lower_bounds_[col] == pytest.approx(ref_lo)
        assert wins.upper_bounds_[col] == pytest.approx(ref_hi)


def test_bounds_match_numpy_reference_iqr():
    """The 'iqr' bounds equal the Q1 - k*IQR / Q3 + k*IQR reference."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(150, 2)) * [1.0, 100.0]

    wins = Winsorizer(method="iqr", iqr_multiplier=1.5)
    wins.fit(X)

    for col in range(2):
        ref_lo, ref_hi = _iqr_reference(X[:, col], 1.5)
        assert wins.lower_bounds_[col] == pytest.approx(ref_lo)
        assert wins.upper_bounds_[col] == pytest.approx(ref_hi)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_transform_preserves_nan():
    """NaN values pass through transform unchanged."""
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
    wins = Winsorizer(method="quantile", lower_quantile=0.25, upper_quantile=0.75)
    wins.fit(X)

    X_nan = np.array([[np.nan, 1.0], [2.0, np.nan], [np.nan, np.nan]])
    out = wins.transform(X_nan)

    assert np.isnan(out).sum() == 4
    np.testing.assert_array_equal(np.isnan(out), np.isnan(X_nan))


def test_all_nan_column_passes_through_unchanged():
    """An all-NaN column gets (-inf, +inf) bounds and its NaN survives."""
    X = np.array(
        [[1.0, np.nan], [2.0, np.nan], [3.0, np.nan], [4.0, np.nan], [5.0, np.nan]]
    )
    wins = Winsorizer(method="quantile", lower_quantile=0.25, upper_quantile=0.75)
    wins.fit(X)

    assert wins.lower_bounds_[1] == -np.inf
    assert wins.upper_bounds_[1] == np.inf

    out = wins.transform(X)
    assert np.isnan(out[:, 1]).all()


# ---------------------------------------------------------------------------
# Collapsed columns
# ---------------------------------------------------------------------------


def test_collapsed_column_not_clipped_and_warning(caplog):
    """A constant column is widened to (-inf, +inf) and produces one WARNING."""
    X = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0], [4.0, 7.0], [100.0, 7.0]])
    wins = Winsorizer(method="quantile", lower_quantile=0.25, upper_quantile=0.75)

    with caplog.at_level(
        logging.WARNING, logger="configurable_automl_engine.preprocessing.outliers"
    ):
        wins.fit(X)

    assert wins.lower_bounds_[1] == -np.inf
    assert wins.upper_bounds_[1] == np.inf

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "column_1" in warnings[0].getMessage()


def test_collapsed_named_column_warning_uses_name(caplog):
    """With DataFrame input the WARNING names the collapsed column by name."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "flat": [5.0, 5.0, 5.0, 5.0]})
    wins = Winsorizer(method="quantile", lower_quantile=0.25, upper_quantile=0.75)

    with caplog.at_level(
        logging.WARNING, logger="configurable_automl_engine.preprocessing.outliers"
    ):
        wins.fit(df)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "flat" in warnings[0].getMessage()

    # The constant column is not clipped even for out-of-range values.
    out = wins.transform(pd.DataFrame({"a": [0.0, 9.0], "flat": [1e9, -1e9]}))
    assert np.allclose(out[:, 1], [1e9, -1e9])


# ---------------------------------------------------------------------------
# Clipping behaviour
# ---------------------------------------------------------------------------


def test_values_beyond_fitted_range_are_clipped():
    """transform clips values outside the interval learned in fit."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    wins = Winsorizer(method="quantile", lower_quantile=0.25, upper_quantile=0.75)
    wins.fit(X)
    lo, hi = wins.lower_bounds_[0], wins.upper_bounds_[0]

    out = wins.transform(np.array([[-10.0], [2.0], [100.0]]))
    assert out[0, 0] == lo
    assert out[1, 0] == 2.0
    assert out[2, 0] == hi


def test_transform_returns_float_ndarray_and_counts(caplog):
    """transform returns a float ndarray and logs a DEBUG summary."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    wins = Winsorizer(method="quantile", lower_quantile=0.25, upper_quantile=0.75)
    wins.fit(X)

    with caplog.at_level(
        logging.DEBUG, logger="configurable_automl_engine.preprocessing.outliers"
    ):
        out = wins.transform(np.array([[1.0], [2.0], [3.0]]))

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == (3, 1)

    debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug, "expected a DEBUG log record from transform"


def test_get_last_clip_counts():
    """Per-column clip counts are reported for a DataFrame and an ndarray."""
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]}
    )
    wins = Winsorizer(method="quantile", lower_quantile=0.25, upper_quantile=0.75)
    wins.fit(df)

    counts = wins.get_last_clip_counts(
        pd.DataFrame({"a": [0.0, 2.0], "b": [1e9, 20.0]})
    )
    assert counts == {"a": 1, "b": 1}

    X = np.arange(10, dtype=float).reshape(5, 2)
    wins2 = Winsorizer(method="quantile", lower_quantile=0.25, upper_quantile=0.75)
    wins2.fit(X)
    counts2 = wins2.get_last_clip_counts(np.array([[100.0, 100.0], [-100.0, 4.5]]))
    assert counts2 == {"column_0": 2, "column_1": 1}


# ---------------------------------------------------------------------------
# Input formats
# ---------------------------------------------------------------------------


def test_dataframe_and_ndarray_same_output():
    """DataFrame and ndarray inputs produce the same clipped output."""
    rng = np.random.RandomState(7)
    X = rng.normal(size=(100, 3))
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])

    wins_arr = Winsorizer().fit(X)
    wins_df = Winsorizer().fit(df)

    out_arr = wins_arr.transform(X)
    out_df = wins_df.transform(df)
    np.testing.assert_allclose(out_arr, out_df)


def test_feature_names_in_and_get_feature_names_out():
    """A DataFrame fit records feature_names_in_; one input maps to one output."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    wins = Winsorizer().fit(df)

    assert list(wins.feature_names_in_) == ["a", "b"]
    assert wins.n_features_in_ == 2
    np.testing.assert_array_equal(wins.get_feature_names_out(), ["a", "b"])
    np.testing.assert_array_equal(wins.get_feature_names_out(["a", "b"]), ["a", "b"])


def test_get_feature_names_out_ndarray_positional():
    """Without names, get_feature_names_out returns positional x<i> names."""
    X = np.arange(6, dtype=float).reshape(3, 2)
    wins = Winsorizer().fit(X)
    np.testing.assert_array_equal(wins.get_feature_names_out(), ["x0", "x1"])


def test_get_feature_names_out_length_mismatch():
    """A wrong-length input_features raises ValueError."""
    X = np.arange(6, dtype=float).reshape(3, 2)
    wins = Winsorizer().fit(X)
    with pytest.raises(ValueError, match="input_features"):
        wins.get_feature_names_out(["only_one"])


def test_transform_not_fitted_raises():
    """transform before fit raises NotFittedError."""
    wins = Winsorizer()
    with pytest.raises(NotFittedError):
        wins.transform(np.array([[1.0]]))


def test_transform_column_mismatch_raises():
    """transform rejects a column count different from fit."""
    X = np.arange(10, dtype=float).reshape(5, 2)
    wins = Winsorizer().fit(X)
    with pytest.raises(ValueError, match="features"):
        wins.transform(np.arange(5, dtype=float).reshape(5, 1))


# ---------------------------------------------------------------------------
# Pickle round-trip
# ---------------------------------------------------------------------------


def test_pickle_round_trip():
    """A fitted Winsorizer survives pickle round-trip with identical output."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 100.0],
            "b": [10.0, 20.0, 30.0, 40.0, np.nan],
        }
    )
    wins = Winsorizer(method="quantile", lower_quantile=0.1, upper_quantile=0.9)
    wins.fit(df)

    restored = pickle.loads(pickle.dumps(wins))
    np.testing.assert_allclose(restored.lower_bounds_, wins.lower_bounds_)
    np.testing.assert_allclose(restored.upper_bounds_, wins.upper_bounds_)
    np.testing.assert_array_equal(restored.feature_names_in_, wins.feature_names_in_)

    X_test = pd.DataFrame({"a": [0.0, 2.0], "b": [100.0, np.nan]})
    np.testing.assert_allclose(
        restored.transform(X_test), wins.transform(X_test), equal_nan=True
    )


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "median"},  # unknown method
        {"lower_quantile": 0.5, "upper_quantile": 0.5},  # lower >= upper
        {"lower_quantile": 0.9, "upper_quantile": 0.1},  # lower > upper
        {"lower_quantile": -0.1},  # out-of-range lower quantile
        {"upper_quantile": 1.5},  # out-of-range upper quantile
        {"iqr_multiplier": 0.0},  # non-positive multiplier
        {"iqr_multiplier": -1.0},  # negative multiplier
    ],
)
def test_invalid_parameters_raise_in_fit(kwargs):
    """Invalid parameters raise ValueError in fit (the constructor stays silent)."""
    wins = Winsorizer(**kwargs)
    # No error at construction time (sklearn convention).
    with pytest.raises(ValueError):
        wins.fit(np.arange(10, dtype=float).reshape(5, 2))


def test_validation_happens_in_fit_not_init():
    """The constructor performs no validation, following sklearn convention."""
    Winsorizer(method="nonsense", lower_quantile=5.0, iqr_multiplier=-1.0)


# ---------------------------------------------------------------------------
# sklearn compatibility
# ---------------------------------------------------------------------------


def test_check_estimator_passes():
    """The full sklearn estimator_checks suite passes with no exclusions.

    Note: when ``SCIPY_ARRAY_API`` is set in the environment, sklearn may
    skip the ``check_array_api_input`` family of checks (sklearn emits a
    ``SkipTestWarning`` for them). The outcome of this test therefore depends
    on the environment: it asserts that every check that actually runs passes,
    not that every known check was executed.
    """
    check_estimator(Winsorizer())


def test_exported_from_preprocessing_package():
    """Winsorizer is importable from the preprocessing package façade."""
    from configurable_automl_engine import preprocessing

    assert preprocessing.Winsorizer is Winsorizer
    assert Winsorizer.__module__ == (
        "configurable_automl_engine.preprocessing.outliers"
    )
    assert "Winsorizer" in preprocessing.__all__


def test_fit_returns_self():
    """fit returns self for chaining."""
    X = np.arange(10, dtype=float).reshape(5, 2)
    wins = Winsorizer()
    assert wins.fit(X) is wins
    check_is_fitted(wins)


def test_3d_input_raises():
    """An input with more than two dimensions raises ValueError."""
    X = np.zeros((2, 2, 2))
    with pytest.raises(ValueError, match="2D input"):
        Winsorizer().fit(X)
    wins = Winsorizer().fit(np.arange(4, dtype=float).reshape(2, 2))
    with pytest.raises(ValueError, match="2D input"):
        wins.transform(X)


def test_get_feature_names_out_mismatch_raises():
    """Names that disagree with the DataFrame fit raise ValueError."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 1.0]})
    wins = Winsorizer().fit(df)
    with pytest.raises(ValueError, match="do not match"):
        wins.get_feature_names_out(["a", "c"])


def test_get_last_clip_counts_column_mismatch_raises():
    """get_last_clip_counts rejects a column count different from fit."""
    X = np.arange(10, dtype=float).reshape(5, 2)
    wins = Winsorizer().fit(X)
    with pytest.raises(ValueError, match="features"):
        wins.get_last_clip_counts(np.arange(5, dtype=float).reshape(5, 1))


# ---------------------------------------------------------------------------
# Non-numeric / degenerate inputs
# ---------------------------------------------------------------------------


def test_string_dataframe_fit_raises_with_context():
    """String DataFrame input raises a contextual TypeError, not a raw astype error."""
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["1", "2", "3"]})
    wins = Winsorizer()
    with pytest.raises(TypeError, match="Winsorizer expects numeric data"):
        wins.fit(df)
    # The same check applies to a fitted transformer's transform.
    fitted = Winsorizer().fit(np.arange(6, dtype=float).reshape(3, 2))
    with pytest.raises(TypeError, match="Winsorizer expects numeric data"):
        fitted.transform(df)


def test_object_ndarray_fit_raises():
    """A non-convertible object ndarray is rejected with a contextual TypeError."""
    X = np.array([["a", 1], ["b", 2]], dtype=object)
    with pytest.raises(TypeError, match="could not convert string to float"):
        Winsorizer().fit(X)


def test_complex_input_raises():
    """Complex input raises ValueError in both fit and transform."""
    X = np.array([[1 + 2j, 3.0], [4 + 5j, 6.0]])
    wins = Winsorizer()
    with pytest.raises(ValueError, match="Complex data not supported"):
        wins.fit(X)
    fitted = Winsorizer().fit(np.arange(4, dtype=float).reshape(2, 2))
    with pytest.raises(ValueError, match="Complex data not supported"):
        fitted.transform(X)


def test_empty_input_raises():
    """Empty 2D inputs (zero samples or zero features) raise ValueError."""
    wins = Winsorizer().fit(np.arange(4, dtype=float).reshape(2, 2))
    with pytest.raises(ValueError, match="0 sample"):
        wins.transform(np.empty((0, 2)))
    with pytest.raises(ValueError, match="0 feature"):
        wins.transform(np.empty((3, 0)))
    with pytest.raises(ValueError, match="0 sample"):
        Winsorizer().fit(np.empty((0, 0)))


def test_bool_input_is_accepted():
    """A boolean ndarray is a valid numeric input and is cast to float."""
    X = np.array([[True, False], [False, True], [True, True], [False, False]])
    wins = Winsorizer(method="quantile", lower_quantile=0.25, upper_quantile=0.75)
    wins.fit(X)
    out = wins.transform(X)
    assert out.dtype == np.float64


def test_sparse_input_raises():
    """Sparse matrices are rejected with a dedicated TypeError."""
    sp_matrix = csr_matrix(np.arange(6, dtype=float).reshape(3, 2))
    wins = Winsorizer()
    with pytest.raises(TypeError, match="sparse"):
        wins.fit(sp_matrix)


# ---------------------------------------------------------------------------
# Non-numeric / NaN parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lower_quantile": "0.05"},
        {"upper_quantile": None},
        {"iqr_multiplier": "1.5"},
        {"lower_quantile": True},  # bool is not a valid real parameter
    ],
)
def test_non_numeric_parameters_raise_value_error(kwargs):
    """Non-numeric parameters raise ValueError (not TypeError) in fit."""
    X = np.arange(10, dtype=float).reshape(5, 2)
    wins = Winsorizer(**kwargs)
    with pytest.raises(ValueError, match="must be a real number"):
        wins.fit(X)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lower_quantile": float("nan")},
        {"upper_quantile": float("nan")},
        {"iqr_multiplier": float("nan")},
    ],
)
def test_nan_parameters_raise_value_error(kwargs):
    """NaN parameters raise ValueError instead of silently producing NaN bounds."""
    X = np.arange(10, dtype=float).reshape(5, 2)
    wins = Winsorizer(**kwargs)
    with pytest.raises(ValueError, match="must not be NaN"):
        wins.fit(X)
