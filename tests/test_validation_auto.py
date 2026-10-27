"""Tests for the 'auto' validation strategy."""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, LeaveOneOut

from configurable_automl_engine.hpo import evaluation as hpo_evaluation
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
    """Branch 1: N <= 15 and N >= 2*P selects LOO."""
    result = choose_validation_method(10, 3)
    assert result["method"] == "LOO"
    # Boundary: N == 2*P still LOO.
    assert choose_validation_method(15, 2)["method"] == "LOO"
    # N == 30/P == 15 is now beyond the LOO range: fixed k-fold takes over.
    assert choose_validation_method(30, 15)["method"] == "kfold"


def test_choose_fixed_kfold_small_dataset():
    """Branch 2: 16 <= N <= 50 uses a fixed k-fold (3-4 folds) instead of LOO."""
    assert choose_validation_method(40, 3)["method"] == "kfold"
    assert choose_validation_method(50, 5)["method"] == "kfold"


@pytest.mark.parametrize(
    "n_samples, n_features, expected_k, expected_avg_test_size",
    [
        (16, 2, 3, 5),
        (19, 2, 3, 6),
        (20, 2, 4, 5),
        (30, 3, 4, 7),
        (40, 3, 4, 10),
        (50, 5, 4, 12),
    ],
)
def test_choose_fixed_kfold_boundaries(
    n_samples, n_features, expected_k, expected_avg_test_size
):
    """Branch 2 fold-count boundaries: k=3 for N <= 19, k=4 for N >= 20."""
    result = choose_validation_method(n_samples, n_features)
    assert result["method"] == "kfold"
    assert result["k"] == expected_k
    assert result["average_test_size"] == expected_avg_test_size
    # average_test_size is an integer floor of N/k.
    assert result["average_test_size"] == n_samples // expected_k
    # Each fold tests the model on a slice of several distinct points.
    assert result["average_test_size"] >= 5


def test_choose_fixed_kfold_motivating_case():
    """N=24/P=2 -> k=4: the model is tested on slices of 6 points, not one."""
    result = choose_validation_method(24, 2)
    assert result["method"] == "kfold"
    assert result["k"] == 4
    assert result["average_test_size"] == 6
    assert "Small dataset: fixed k-fold CV (3-4 folds)" in result["warning"]


def test_choose_fixed_kfold_k_never_exceeds_n():
    """Defensive guard: k is clamped to [1, N] across the whole range."""
    for n_samples in range(16, 51):
        result = choose_validation_method(n_samples, 1)
        assert result["method"] == "kfold"
        assert 1 <= result["k"] <= n_samples


def test_choose_fixed_kfold_low_confidence_warning():
    """Fixed k-fold with N < 2*P attaches the uniform low-confidence warning."""
    result = choose_validation_method(16, 9)
    assert result["method"] == "kfold"
    assert result["k"] == 3
    assert "Low confidence" in result["warning"]


def test_choose_fixed_kfold_transition_15_to_16():
    """Transition 15 -> 16: LOO stops, fixed k-fold starts."""
    assert choose_validation_method(15, 2)["method"] == "LOO"
    result = choose_validation_method(16, 2)
    assert result["method"] == "kfold"
    assert result["k"] == 3


def test_choose_fixed_kfold_transition_19_to_20():
    """Transition 19 -> 20: k grows from 3 to 4."""
    assert choose_validation_method(19, 2)["k"] == 3
    assert choose_validation_method(20, 2)["k"] == 4


def test_choose_fixed_kfold_transition_50_to_51():
    """Transition 50 -> 51: fixed k-fold ends, continuous heuristic resumes."""
    result_50 = choose_validation_method(50, 5)
    assert result_50["method"] == "kfold"
    assert result_50["k"] == 4
    # Continuous branch must not crash and must not reuse the fixed k=4.
    result_51 = choose_validation_method(51, 5)
    assert result_51["method"] == "kfold"
    assert result_51["k"] != 4


@pytest.mark.parametrize(
    "n_samples, n_features, method, expected_k",
    [
        (10, 3, "LOO", None),  # Branch 1: LOO
        (15, 2, "LOO", None),  # Branch 1 boundary preserved
        (5, 3, "LOO", None),  # Branch 6: micro-sample relabel LOO
        (100, 5, "kfold", 5),  # Branch 6: continuous k-fold
        (150, 5, "kfold", 7),  # Branch 6: continuous k-fold (unchanged)
        (1000, 5, "kfold", 5),  # Branch 3: mid-size fixed 5-fold CV
        (50000, 5, "train_test_split", None),  # Branch 4: standard split, large N
        (60000, 10000, "train_test_split", None),  # Branch 5: high-dimensional
    ],
)
def test_choose_unchanged_branches(n_samples, n_features, method, expected_k):
    """Branches 1 (N <= 15), 4-6 keep their behaviour for these inputs."""
    result = choose_validation_method(n_samples, n_features)
    assert result["method"] == method
    if expected_k is not None:
        assert result["k"] == expected_k


def test_choose_mid_size_fixed_kfold():
    """Branch 3: mid-size samples with a good N/P ratio use fixed 5-fold CV.

    The old standard hold-out of max(30, 2*P) rows (30 rows for
    N=1000, 100 rows for N=10000) compared models on far too few points,
    making the winner nearly random and the Sanity Gate OOF vector cover only
    ~3% of the data. 5-fold CV scores every row exactly once.
    """
    for n_samples, n_features in [
        (200, 5),  # old: hold-out of 30 rows
        (500, 5),  # old: hold-out of 30 rows
        (999, 5),  # N is not a multiple of 5: folds of 200/200/200/200/199
        (1000, 5),  # old: hold-out of 30 rows
        (10000, 5),  # old: hold-out of 100 rows
        (300, 3),  # pipeline-size dataset (test_prepare_dataset case)
        (49999, 5),  # just below the large-N threshold
    ]:
        result = choose_validation_method(n_samples, n_features)
        assert result["method"] == "kfold"
        assert result["k"] == 5
        assert result["average_test_size"] == round(n_samples / 5, 1)


def test_choose_train_test_split_standard():
    """Branch 4: standard train-test split for large samples (N >= 50000)."""
    result = choose_validation_method(50000, 5)
    assert result["method"] == "train_test_split"
    assert result["regime"] == "standard"
    assert result["test_percent"] == 1
    assert result["test_size"] == 500
    assert result["train_size"] == 49500
    assert result["test_size"] + result["train_size"] == 50000


def test_choose_train_test_split_standard_high_dimension_p():
    """Branch 4 with a large P: the minimum test size grows with 2*P."""
    result = choose_validation_method(100000, 2500)
    assert result["method"] == "train_test_split"
    assert result["regime"] == "standard"
    # min_test = max(30, 2*2500) = 5000 -> test_percent = ceil(5000 / 100000) = 5
    assert result["test_percent"] == 5
    assert result["test_size"] == 5000
    assert result["train_size"] == 95000


def test_choose_train_test_split_high_dimensional():
    """Branch 5: high-dimensional train-test split with warning."""
    result = choose_validation_method(60000, 10000)
    assert result["method"] == "train_test_split"
    assert result["regime"] == "high_dimensional"
    assert result["test_percent"] == 34
    assert result["test_size"] == 20400
    assert result["train_size"] == 39600
    assert "Low confidence" in result["warning"]


def test_choose_kfold_continuous():
    """Branch 6: k-fold continuous heuristic."""
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
    """N < 2*P in the fixed-kfold range produces the low-confidence warning."""
    result = choose_validation_method(31, 16)
    assert result["method"] == "kfold"
    assert result["k"] == 4
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


def test_make_cv_auto_mid_size_kfold():
    """auto on mid-size data resolves to a fixed 5-fold KFold."""
    method, cv, decision = make_cv(
        1000,
        val_method="auto",
        n_folds=5,
        random_state=42,
        test_size=0.2,
        n_features=5,
    )
    assert method == "k_fold"
    assert isinstance(cv, KFold)
    assert cv.get_n_splits() == 5
    assert decision == {"method": "kfold", "k": 5, "average_test_size": 200.0}


def test_make_cv_auto_train_test_split():
    """auto on large samples (N >= 50000) resolves to a train/test split."""
    method, cv, _ = make_cv(
        50000,
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
    method, cv, decision = make_cv(
        31,
        val_method="auto",
        n_folds=5,
        random_state=42,
        test_size=0.2,
        n_features=16,
    )
    assert method == "k_fold"
    assert isinstance(cv, KFold)
    assert cv.get_n_splits() == 4
    assert decision["k"] == 4
    assert "Low confidence" in decision["warning"]


def test_make_cv_auto_fixed_kfold_motivating_case():
    """auto on 24x2 resolves to a 4-fold KFold (fixed k-fold branch)."""
    method, cv, decision = make_cv(
        24,
        val_method="auto",
        n_folds=5,
        random_state=42,
        test_size=0.2,
        n_features=2,
    )
    assert method == "k_fold"
    assert isinstance(cv, KFold)
    assert cv.get_n_splits() == 4
    assert decision == {
        "method": "kfold",
        "k": 4,
        "average_test_size": 6,
        "warning": (
            "Small dataset: fixed k-fold CV (3-4 folds) to prevent "
            "LOO overfitting."
        ),
    }


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


def test_iter_splits_auto_fixed_kfold():
    """auto on 24x2 resolves to a fixed 4-fold k-fold split."""
    X, y = _dummy_data(n=24, p=2)
    splits = list(iter_splits(X, y, method="auto"))
    assert len(splits) == 4
    # Each fold tests the model on exactly 6 distinct points (24 // 4).
    test_sizes = {len(x_te) for _, x_te, _, _ in splits}
    assert test_sizes == {6}


def test_iter_splits_auto_mid_size_kfold():
    """auto -> kfold: N=1000 is scored with 5-fold CV, not a 30-row
    hold-out, so every row is validated exactly once."""
    X, y = _dummy_data(n=1000, p=5)
    splits = list(iter_splits(X, y, method="auto"))
    assert len(splits) == 5
    test_sizes = {len(x_te) for _, x_te, _, _ in splits}
    assert test_sizes == {200}
    # OOF coverage: each row appears exactly once in a validation fold.
    assert sum(len(x_te) for _, x_te, _, _ in splits) == 1000


def test_iter_splits_auto_mid_size_kfold_uneven_n():
    """auto -> kfold for N not divisible by 5: KFold produces
    uneven folds (4x200 + 199 for N=999) and keeps 100% OOF coverage."""
    X, y = _dummy_data(n=999, p=5)
    splits = list(iter_splits(X, y, method="auto"))
    assert len(splits) == 5
    test_sizes = sorted(len(x_te) for _, x_te, _, _ in splits)
    assert test_sizes == [199, 200, 200, 200, 200]
    assert sum(test_sizes) == 999


def test_iter_splits_auto_train_test_split_dynamic_test_size(monkeypatch):
    """auto -> train_test_split (large N) uses the computed (integer) test size,
    not 0.2.

    The heuristic itself ("large N → split") is covered by the numeric tests
    (test_choose_train_test_split_standard / test_make_cv_auto_train_test_split);
    here make_cv is patched to avoid building a 50k-row array just to check
    that the integer test_size is forwarded to the actual split.
    """
    X, y = _dummy_data(n=1000, p=5)
    fake_decision = {
        "method": "train_test_split",
        "test_size": 500,
        "train_size": 49500,
    }

    def fake_make_cv(
        n_samples, *, val_method, n_folds, random_state, test_size, n_features=None
    ):
        return "train_test_split", None, fake_decision

    monkeypatch.setattr(validation_module, "make_cv", fake_make_cv)
    (x_tr, x_te, y_tr, y_te) = next(iter_splits(X, y, method="auto"))
    assert len(x_te) == 500
    assert len(x_tr) == 500


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
        (1000, 5, 800),  # mid-size: fixed 5-fold CV
        (50000, 5, 49500),  # large N: train_test_split -> train_size
        (24, 2, 18),  # fixed kfold k=4 -> floor(24 * (1 - 1/4))
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


def test_get_effective_train_size_integer_test_size():
    """Integer test_size ('auto' resolution → split):
    is treated as a number of rows, not a fraction (otherwise Neff would collapse to 1).
    """
    # Explicit train_test_split with an integer test_size
    result = get_effective_train_size(
        120, ValidationStrategy.train_test_split, test_size=15
    )
    assert result == 105
# auto → train_test_split: choose_validation_method returns an integer
    # number of rows; get_effective_train_size must produce the same train_size.
    decision = choose_validation_method(50000, 5)
    assert decision["method"] == "train_test_split"
    result_auto = get_effective_train_size(
        50000, ValidationStrategy.auto, n_folds=5, n_features=5
    )
    assert result_auto == int(decision["train_size"])


# ══════════════════════ tuner integration ══════════════════════


def test_tuner_passes_n_features_for_auto(monkeypatch):
    """optimize() must forward P=X.shape[1] to make_cv / get_effective_train_size."""
    from unittest.mock import MagicMock

    from configurable_automl_engine import tuner
    from configurable_automl_engine.hpo import evaluation as hpo_evaluation
    from configurable_automl_engine.hpo import runner as hpo_runner

    X = pd.DataFrame(np.random.rand(20, 4), columns=list("abcd"))
    y = pd.Series(np.random.rand(20))

    captured_cv = {}
    captured_eff = {}

    def fake_make_cv(n_samples, *, val_method, n_folds, random_state, test_size, n_features=None):
        captured_cv["n_features"] = n_features
        captured_cv["n_samples"] = n_samples
        # k_fold is always accompanied by a ready cv_obj (make_cv invariant),
        # from which the tuner takes the fold count for the pruning branch.
        return "k_fold", KFold(n_splits=2, shuffle=True, random_state=random_state), None

    def fake_get_effective_train_size(
        n_total, strategy, n_folds=5, test_size=0.2, n_features=None
    ):
        captured_eff["n_features"] = n_features
        return 16

    monkeypatch.setattr(hpo_runner, "make_cv", fake_make_cv)
    monkeypatch.setattr(hpo_runner, "get_effective_train_size", fake_get_effective_train_size)
    monkeypatch.setattr(hpo_evaluation, "get_scorer_object", lambda *a, **k: lambda *x: 0.5)
    monkeypatch.setattr(hpo_runner, "create_model", lambda *a, **k: MagicMock())
    monkeypatch.setattr(
        hpo_evaluation.model_selection, "cross_val_score", lambda *a, **k: [0.5, 0.5]
    )

    tuner.optimize(
        "nearest_neighbors_regression",
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
    train/test split (large N >= 50000), the objective must call
    iter_splits with ``method="train_test_split"`` and the dataset-derived
    (integer) ``test_size`` from the already-resolved decision, instead of
    re-resolving 'auto' or using the fixed 0.2 fraction.
    """
    from unittest.mock import MagicMock

    from configurable_automl_engine import tuner
    from configurable_automl_engine.hpo import evaluation as hpo_evaluation
    from configurable_automl_engine.hpo import runner as hpo_runner

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

    monkeypatch.setattr(hpo_runner, "make_cv", fake_make_cv)
    monkeypatch.setattr(hpo_runner, "get_effective_train_size", fake_get_effective_train_size)
    monkeypatch.setattr(hpo_evaluation, "iter_splits", fake_iter_splits)
    monkeypatch.setattr(hpo_evaluation, "get_scorer_object", lambda *a, **k: lambda *x: 0.5)
    model = MagicMock()
    # The hold-out scorer fits a clone of the pipeline (the trial estimator
    # stays unfitted): the mock model clones to itself.
    model.__sklearn_clone__ = lambda: model
    monkeypatch.setattr(hpo_runner, "create_model", lambda *a, **k: model)

    tuner.optimize(
        "nearest_neighbors_regression",
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
