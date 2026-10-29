import pytest
import numpy as np
from sklearn.model_selection import LeaveOneOut
import sys
import importlib
from unittest.mock import MagicMock, patch
import pandas as pd

from configurable_automl_engine.validation import (
    iter_splits,
    make_cv,
    RANDOM_STATE,
    InvalidDataError,
    ValidationStrategy,
    ValidationError,
)


def _dummy_data(n=100):
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(n, 5))
    y = rng.integers(0, 2, size=n)
    return X, y


# --- Tests for make_cv and norm_val_method (Target: 100% Coverage) ---


def test_norm_val_method_enum():
    """Lines 41-42 coverage: passing a ValidationStrategy (Enum)."""
    # Test via direct reference to the strategy
    method_name, _, _ = make_cv(
        100,
        val_method=ValidationStrategy.k_fold,
        n_folds=5,
        random_state=42,
        test_size=0.2,
    )
    assert method_name == "k_fold"


def test_make_cv_loo_error():
    """
    Coverage of the n < 2 check for LOO.
    Fixed: match corresponds to the text 'at least 2 samples'.
    """
    with pytest.raises(InvalidDataError, match="at least 2 samples"):
        make_cv(1, val_method="loo", n_folds=5, random_state=42, test_size=0.2)


def test_make_cv_kfold_fallback():
    """Coverage of the fallback branches: when there is too little data for KFold."""
    # Case 1: n_samples < 4 (even if n_folds is small)
    method_name, cv, _ = make_cv(
        3, val_method="k_fold", n_folds=2, random_state=42, test_size=0.2
    )
    assert method_name == "train_test_split"
    assert cv is None

    # Case 2: n_samples < 2 * n_folds (e.g. 5 < 2*3)
    method_name2, cv2, _ = make_cv(
        5, val_method="k_fold", n_folds=3, random_state=42, test_size=0.2
    )
    assert method_name2 == "train_test_split"
    assert cv2 is None


def test_make_cv_loo_success():
    """Coverage of successful LOO creation."""
    method_name, cv, _ = make_cv(
        10, val_method="loo", n_folds=5, random_state=42, test_size=0.2
    )
    assert method_name == "loo"
    assert isinstance(cv, LeaveOneOut)


# --- Tests for iter_splits (coverage of the generator's core logic) ---


def test_iter_splits_logic_all_branches():
    X, y = _dummy_data(n=10)

    # Cover the TTS branch
    assert len(list(iter_splits(X, y, method="train_test_split"))) == 1

    # Cover the KFold branch
    assert len(list(iter_splits(X, y, method="k_fold", n_folds=3))) == 3

    # Cover the LOO branch
    assert len(list(iter_splits(X, y, method="loo"))) == 10


def test_iter_splits_errors():
    """
    Check error handling in the generator.
    """
    X, y = _dummy_data(n=5)

    # 1. Test for an unknown method
    # match changed to correspond to the make_cv function, since iter_splits calls it
    with pytest.raises(ValueError, match="Unknown validation method"):
        list(iter_splits(X, y, method="unknown_method"))
    # 2. Test for empty data (X length == 0)
    with pytest.raises(InvalidDataError, match="Input array X is empty"):
        list(iter_splits(np.array([]), np.array([]), method="k_fold"))
    # 3. Test for an X/y length mismatch
    with pytest.raises(InvalidDataError, match="X and y length mismatch"):
        list(iter_splits(np.random.rand(10, 2), np.random.rand(5), method="k_fold"))


def test_make_cv_invalid_string():
    """
    Coverage of the ValueError at the end of make_cv.
    Fixed: match corresponds to the text in the implementation.
    """
    with pytest.raises(ValueError, match="Unknown validation method"):
        make_cv(
            100, val_method="not_a_strategy", n_folds=5, random_state=42, test_size=0.2
        )


def test_logging_edge_case_jupyter():
    """
    Covers lines 41-42: edge-case of logging initialization.
    Simulates the absence of the handlers attribute on the logging module.
    """
    # Save the original
    original_logging = sys.modules.get("logging")

    # Create a fake module without handlers
    mock_logging = MagicMock()
    del mock_logging.handlers
    sys.modules["logging"] = mock_logging

    # Reload the validation module so the if block runs
    import configurable_automl_engine.validation as validation

    importlib.reload(validation)

    # Check that the logging alias was created
    assert hasattr(validation, "logging")

    # Restore the system
    if original_logging:
        sys.modules["logging"] = original_logging
    else:
        sys.modules.pop("logging", None)


def test_make_cv_train_test_split_branch():
    """
    Covers line 92: the 'train_test_split' return in make_cv.
    """
    method_name, cv_obj, _ = make_cv(
        n_samples=100,
        val_method="train_test_split",
        n_folds=5,
        random_state=42,
        test_size=0.2,
    )

    assert method_name == "train_test_split"
    assert cv_obj is None


class TestValidationEngineCoverage:
    """Tests covering specific code branches in iter_splits."""

    def test_iter_splits_critical_error_cv_none(self):
        """
        Covers line 178 (raise ValidationError).
        We catch the base Exception to avoid import-path issues.
        """
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        # Patch make_cv exactly at the path reported in the error logs
        with patch("configurable_automl_engine.validation.make_cv") as mocked_make_cv:
            mocked_make_cv.return_value = ("k_fold", None, None)

            gen = iter_splits(X, y, method="k_fold")

            # Catch any exception and check its name and text
            # This avoids the problem where ValidationError in the test != ValidationError in the code
            with pytest.raises(Exception) as exc_info:
                next(gen)

            # Check that it is indeed our exception by class name
            assert exc_info.type.__name__ == "ValidationError"
            # Check the error text (the concatenated string from the code)
            assert "Critical error: CV object is not initialized" in str(exc_info.value)

    def test_iter_splits_unknown_final_method_yields_nothing(self):
        """A strategy make_cv does not produce yields no split (no error)."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        with patch("configurable_automl_engine.validation.make_cv") as mocked_make_cv:
            mocked_make_cv.return_value = ("time_series", None, None)
            assert list(iter_splits(X, y, method="k_fold")) == []

    def test_iter_splits_pandas_iloc_coverage(self):
        """
        Covers:
            if hasattr(X, 'iloc'):
                x_tr, x_te = X.iloc[train_idx], X.iloc[test_idx]
            if hasattr(y, 'iloc'):
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        """
        # Create Pandas data
        X_df = pd.DataFrame({"feat1": [1, 2, 3, 4, 5], "feat2": [10, 20, 30, 40, 50]})
        y_ser = pd.Series([0, 1, 0, 1, 0], name="target")
        # Use k_fold so the index loop runs
        gen = iter_splits(X_df, y_ser, method="k_fold", n_folds=2)

        # Get the first iteration
        x_tr, x_te, y_tr, y_te = next(gen)
        # Check that Pandas objects were returned (meaning iloc worked)
        assert isinstance(x_tr, pd.DataFrame)
        assert isinstance(x_te, pd.DataFrame)
        assert isinstance(y_tr, pd.Series)
        assert isinstance(y_te, pd.Series)

        # Check data correctness
        assert len(x_tr) + len(x_te) == len(X_df)
        assert len(y_tr) + len(y_te) == len(y_ser)

    def test_iter_splits_pandas_X_only_iloc(self):
        """
        Additional test for the case where X is a DataFrame and y is None.
        Covers indexing X without indexing y.
        """
        X_df = pd.DataFrame({"feat1": [1, 2, 3, 4, 5]})

        gen = iter_splits(X_df, y=None, method="k_fold", n_folds=2)
        x_tr, x_te, y_tr, y_te = next(gen)
        assert isinstance(x_tr, pd.DataFrame)
        assert y_tr is None
        assert y_te is None

    def test_iter_splits_numpy_fallback(self):
        """
        For completeness: check that if there is no iloc (numpy), the code takes the else branch.
        """
        X_np = np.random.rand(10, 2)
        y_np = np.random.randint(0, 2, 10)
        gen = iter_splits(X_np, y_np, method="k_fold", n_folds=2)
        x_tr, x_te, y_tr, y_te = next(gen)
        assert isinstance(x_tr, np.ndarray)
        assert isinstance(y_tr, np.ndarray)
