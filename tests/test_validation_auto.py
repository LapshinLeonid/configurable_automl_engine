"""Tests for the 'auto' validation strategy (#13)."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, LeaveOneOut

from configurable_automl_engine.common.definitions import ValidationStrategy
from configurable_automl_engine.common.validation_utils import (
    choose_validation_method,
    get_effective_train_size,
)
import configurable_automl_engine.validation as validation_module
from configurable_automl_engine.validation import iter_splits, make_cv


# ══════════════════════ choose_validation_method ══════════════════════


def test_choose_invalid_p_zero():
    """P <= 0 raises ValueError per spec."""
    with pytest.raises(ValueError, match=r"n_features \(P\) must be > 0"):
        choose_validation_method(100, 0)
    with pytest.raises(ValueError, match=r"n_features \(P\) must be > 0"):
        choose_validation_method(100, -5)


def test_choose_invalid_n_lt_2():
    """N < 2 returns the 'invalid' sentinel."""
    result = choose_validation_method(1, 5)
    assert result["method"] == "invalid"
    assert "reason" in result


def test_choose_loo_small_sample():
    """Branch 1: N <= 30 and N >= 2*P selects LOO."""
    result = choose_validation_method(10, 3)
    assert result["method"] == "LOO"
    # Boundary: N == 2*P still LOO.
    assert choose_validation_method(30, 15)["method"] == "LOO"


def test_choose_loo_small_sample_good_ratio():
    """Branch 2: 31 <= N <= 50 and N >= 10*P selects LOO."""
    assert choose_validation_method(40, 3)["method"] == "LOO"
    assert choose_validation_method(50, 5)["method"] == "LOO"


def test_choose_train_test_split_standard():
    """Branch 3: standard train-test split with computed sizes."""
    result = choose_validation_method(1000, 5)
    assert result["method"] == "train_test_split"
    assert result["regime"] == "standard"
    assert result["test_percent"] == 3
    assert result["test_size"] == 30
    assert result["train_size"] == 970
    assert result["test_size"] + result["train_size"] == 1000


def test_choose_train_test_split_high_dimensional():
    """Branch 4: high-dimensional train-test split with warning."""
    result = choose_validation_method(60000, 10000)
    assert result["method"] == "train_test_split"
    assert result["regime"] == "high_dimensional"
    assert result["test_percent"] == 34
    assert result["test_size"] == 20400
    assert result["train_size"] == 39600
    assert "Low confidence" in result["warning"]


def test_choose_kfold_continuous():
    """Branch 5: k-fold continuous heuristic."""
    result = choose_validation_method(100, 5)
    assert result["method"] == "kfold"
    assert result["k"] == 5
    assert result["average_test_size"] == 20.0
    assert "warning" not in result


def test_choose_kfold_micro_relabel_loo():
    """Micro-sample (N <= 5) protects against k > N and relabels k == N to LOO."""
    result = choose_validation_method(5, 3)
    assert result["method"] == "LOO"
    assert "k-fold with k=N" in result["note"]
    assert "warning" in result


def test_choose_kfold_low_confidence_warning():
    """k == 2 produces the low-confidence warning."""
    result = choose_validation_method(31, 16)
    assert result["method"] == "kfold"
    assert result["k"] == 2
    assert "Low confidence" in result["warning"]


def test_choose_kfold_high_variance_warning():
    """R_is_low (N < 2*P) produces the low-confidence warning."""
    result = choose_validation_method(100, 60)
    assert result["method"] == "kfold"
    assert "Low confidence" in result["warning"]


# ══════════════════════ make_cv with 'auto' ══════════════════════


def test_make_cv_auto_kfold():
    method, cv, _ = make_cv(
        100,
        val_method="auto",
        n_folds=5,
        random_state=42,
        test_size=0.2,
        n_features=5,
    )
    assert method == "k_fold"
    assert isinstance(cv, KFold)
    assert cv.get_n_splits() == 5


def test_make_cv_auto_enum():
    method, cv, _ = make_cv(
        100,
        val_method=ValidationStrategy.auto,
        n_folds=5,
        random_state=42,
        test_size=0.2,
        n_features=5,
    )
    assert method == "k_fold"
    assert isinstance(cv, KFold)


def test_make_cv_auto_loo():
    method, cv, _ = make_cv(
        10,
        val_method="auto",
        n_folds=5,
        random_state=42,
        test_size=0.2,
        n_features=3,
    )
    assert method == "loo"
    assert isinstance(cv, LeaveOneOut)


def test_make_cv_auto_relabel_loo():
    method, cv, _ = make_cv(
        5,
        val_method="auto",
        n_folds=5,
        random_state=42,
        test_size=0.2,
        n_features=3,
    )
    assert method == "loo"
    assert isinstance(cv, LeaveOneOut)


def test_make_cv_auto_train_test_split():
    method, cv, _ = make_cv(
        1000,
        val_method="auto",
        n_folds=5,
        random_state=42,
        test_size=0.2,
        n_features=5,
    )
    assert method == "train_test_split"
    assert cv is None


def test_make_cv_auto_kfold_with_warning():
    """auto resolving to k-fold with the low-confidence warning."""
    method, cv, _ = make_cv(
        31,
        val_method="auto",
        n_folds=5,
        random_state=42,
        test_size=0.2,
        n_features=16,
    )
    assert method == "k_fold"
    assert isinstance(cv, KFold)
    assert cv.get_n_splits() == 2


def test_make_cv_auto_high_dimensional_warning():
    """auto resolving to high-dimensional train-test split with warning."""
    method, cv, _ = make_cv(
        60000,
        val_method="auto",
        n_folds=5,
        random_state=42,
        test_size=0.2,
        n_features=10000,
    )
    assert method == "train_test_split"
    assert cv is None


def test_make_cv_auto_fallback_no_p():
    """auto without n_features safely falls back to k_fold."""
    method, cv, _ = make_cv(
        100,
        val_method="auto",
        n_folds=5,
        random_state=42,
        test_size=0.2,
    )
    assert method == "k_fold"
    assert isinstance(cv, KFold)


def test_make_cv_auto_fallback_no_p_small_sample():
    """auto fallback to k_fold then to train_test_split when N too small."""
    method, cv, _ = make_cv(
        3,
        val_method="auto",
        n_folds=5,
        random_state=42,
        test_size=0.2,
    )
    assert method == "train_test_split"
    assert cv is None


def test_make_cv_auto_invalid_n():
    """auto with N < 2 raises InvalidDataError."""
    # Resolve the class from the module at call time: test_validation.py reloads
    # the module, so class identity must not be relied upon.
    with pytest.raises(validation_module.InvalidDataError, match="N < 2"):
        make_cv(
            1,
            val_method="auto",
            n_folds=5,
            random_state=42,
            test_size=0.2,
            n_features=3,
        )


def test_make_cv_auto_invalid_p_fallback():
    """auto with non-positive P safely falls back to k_fold."""
    method, cv, _ = make_cv(
        100,
        val_method="auto",
        n_folds=5,
        random_state=42,
        test_size=0.2,
        n_features=0,
    )
    assert method == "k_fold"
    assert isinstance(cv, KFold)


# ══════════════════════ iter_splits with 'auto' ══════════════════════


def _dummy_data(n=100, p=5):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    return X, y


def test_iter_splits_auto_kfold():
    X, y = _dummy_data(n=100, p=5)
    splits = list(iter_splits(X, y, method="auto"))
    assert len(splits) == 5


def test_iter_splits_auto_loo():
    X, y = _dummy_data(n=10, p=3)
    splits = list(iter_splits(X, y, method="auto"))
    assert len(splits) == 10


def test_iter_splits_auto_train_test_split_dynamic_test_size():
    """auto -> train_test_split uses the computed (integer) test size, not 0.2."""
    X, y = _dummy_data(n=1000, p=5)
    (x_tr, x_te, y_tr, y_te) = next(iter_splits(X, y, method="auto"))
    assert len(x_te) == 30
    assert len(x_tr) == 970


def test_iter_splits_auto_pandas():
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.rand(100))
    splits = list(iter_splits(X, y, method="auto"))
    assert len(splits) == 5
    x_tr, x_te, y_tr, y_te = splits[0]
    assert isinstance(x_tr, pd.DataFrame)
    assert isinstance(y_tr, pd.Series)


# ══════════════════════ get_effective_train_size with 'auto' ══════════════════════


@pytest.mark.parametrize(
    "n_total, n_features, expected",
    [
        (100, 5, 80),  # kfold k=5 -> floor(100 * (1 - 1/5))
        (10, 3, 9),  # LOO -> N - 1
        (5, 3, 4),  # relabel LOO -> N - 1
        (1000, 5, 970),  # train_test_split -> train_size
    ],
)
def test_get_effective_train_size_auto(n_total, n_features, expected):
    result = get_effective_train_size(
        n_total,
        ValidationStrategy.auto,
        n_features=n_features,
    )
    assert result == expected


def test_get_effective_train_size_auto_fallback_no_p():
    """auto without P falls back to default k_fold behaviour."""
    result = get_effective_train_size(100, "auto", n_folds=5)
    assert result == math.floor(100 * (1 - 1 / 5))


def test_get_effective_train_size_auto_invalid_p():
    result = get_effective_train_size(100, "auto", n_folds=5, n_features=0)
    assert result == math.floor(100 * (1 - 1 / 5))


def test_get_effective_train_size_auto_fallback_small_sample_no_p():
    """auto without P and too few samples falls back to a train/test split."""
    # 3 samples < max(4, 2*5)=10 -> resolve_auto_no_features_fallback returns
    # train_test_split, matching make_cv's behaviour for the same inputs.
    result = get_effective_train_size(3, "auto", n_folds=5, test_size=0.2)
    assert result == math.floor(3 * (1 - 0.2))
    # make_cv resolves the same inputs to train_test_split (consistency).
    method, cv, decision = make_cv(
        3,
        val_method="auto",
        n_folds=5,
        random_state=42,
        test_size=0.2,
    )
    assert method == "train_test_split"
    assert cv is None
    assert decision is None


def test_get_effective_train_size_auto_n_lt_2():
    """auto with N < 2 (invalid decision) yields no reliable training rows."""
    result = get_effective_train_size(1, "auto", n_folds=5, n_features=3)
    assert result == 0


# ══════════════════════ tuner integration ══════════════════════


def test_tuner_passes_n_features_for_auto(monkeypatch):
    """optimize() must forward P=X.shape[1] to make_cv / get_effective_train_size."""
    from unittest.mock import MagicMock

    from configurable_automl_engine import tuner

    X = pd.DataFrame(np.random.rand(20, 4), columns=list("abcd"))
    y = pd.Series(np.random.rand(20))

    captured_cv = {}
    captured_eff = {}

    def fake_make_cv(n_samples, *, val_method, n_folds, random_state, test_size, n_features=None):
        captured_cv["n_features"] = n_features
        captured_cv["n_samples"] = n_samples
        return "k_fold", None, None

    def fake_get_effective_train_size(
        n_total, strategy, n_folds=5, test_size=0.2, n_features=None
    ):
        captured_eff["n_features"] = n_features
        return 16

    monkeypatch.setattr(tuner, "make_cv", fake_make_cv)
    monkeypatch.setattr(tuner, "get_effective_train_size", fake_get_effective_train_size)
    monkeypatch.setattr(tuner, "get_scorer_object", lambda *a, **k: lambda *x: 0.5)
    monkeypatch.setattr(tuner, "create_model", lambda *a, **k: MagicMock())
    monkeypatch.setattr(
        tuner.model_selection, "cross_val_score", lambda *a, **k: [0.5, 0.5]
    )

    tuner.optimize(
        "knn",
        X,
        y,
        validation_strategy="auto",
        n_folds=2,
        n_trials=1,
    )

    assert captured_cv["n_features"] == 4
    assert captured_cv["n_samples"] == 20
    assert captured_eff["n_features"] == 4


def test_tuner_auto_train_test_split_uses_resolved_test_size(monkeypatch):
    """optimize() routes auto->train_test_split with the resolved test size.

    Guards the threaded-decision path: when the auto strategy resolves to a
    train/test split, the objective must call iter_splits with
    ``method="train_test_split"`` and the dataset-derived (integer) ``test_size``
    from the already-resolved decision, instead of re-resolving 'auto' or using
    the fixed 0.2 fraction.
    """
    from unittest.mock import MagicMock

    from configurable_automl_engine import tuner

    X = pd.DataFrame(np.random.rand(1000, 5), columns=list("abcde"))
    y = pd.Series(np.random.rand(1000))

    captured = {}

    def fake_make_cv(n_samples, *, val_method, n_folds, random_state, test_size, n_features=None):
        # auto resolves to train_test_split; expose the resolved decision.
        return (
            "train_test_split",
            None,
            {"method": "train_test_split", "test_size": 30, "train_size": 970},
        )

    def fake_get_effective_train_size(
        n_total, strategy, n_folds=5, test_size=0.2, n_features=None
    ):
        return 970

    def fake_iter_splits(*args, **kwargs):
        captured["method"] = kwargs.get("method")
        captured["test_size"] = kwargs.get("test_size")
        return iter(
            [
                (
                    np.random.rand(970, 5),
                    np.random.rand(30, 5),
                    np.random.rand(970),
                    np.random.rand(30),
                )
            ]
        )

    monkeypatch.setattr(tuner, "make_cv", fake_make_cv)
    monkeypatch.setattr(tuner, "get_effective_train_size", fake_get_effective_train_size)
    monkeypatch.setattr(tuner, "iter_splits", fake_iter_splits)
    monkeypatch.setattr(tuner, "get_scorer_object", lambda *a, **k: lambda *x: 0.5)
    monkeypatch.setattr(tuner, "create_model", lambda *a, **k: MagicMock())

    tuner.optimize(
        "knn",
        X,
        y,
        validation_strategy="auto",
        n_folds=5,
        n_trials=1,
    )

    # The resolved test size (30) must be forwarded, and iter_splits must NOT be
    # asked to re-resolve 'auto' (method is the concrete "train_test_split").
    assert captured["method"] == "train_test_split"
    assert captured["test_size"] == 30
