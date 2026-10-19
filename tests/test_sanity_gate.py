"""Tests for ModelSanityGate — the model-degeneration audit module.

Coverage:
    1. Circuit A (Diversity Ratio): an honest model passes; a constant
       prediction fails; the 'ratio equals threshold' boundary passes;
       a constant target is skipped with a soft_warning; the adaptive
       mode relative to the constant model.
    2. Circuit B (Unique Ratio): rounding scale driven by std(y); the
       lower nunique bound; the 'nunique == min_unique_count' boundary
       passes.
    3. Circuit C (Dead Feature Check): Lasso with ~60% zeros is a signal,
       not a disqualification; the 'many zeros + low diversity' combination
       disqualifies; the permutation path for non-linear models with cost
       control and determinism; predict exceptions become reasons.
    4. Circuit D (Generalization Gap): overfitted models (depth-20 tree,
       1-NN) fail; RMSE_oof≈0 / RMSE_full≈0 — an infinite gap; division by
       zero; the gap == max boundary passes; the example (0.1 vs 0.19); the
       generalization_gap_mode regimes 'warn' (soft_warning, does not
       disqualify — including the infinite gap from division by zero) and
       'reject' (the strict old-default behavior, parity; division by zero
       → reasons).
    5. Modes and edge cases: warn_only; circuit toggling; NaN masking;
       severity; parameter validation; distinct reasons/soft_warnings.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from configurable_automl_engine.training_engine.sanity_gate import (
    ModelSanityGate,
    SanityCheckResult,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _dataset(seed: int = 7, n: int = 150, p: int = 6, noise: float = 1.0):
    """Clean dataset: X0/X1 are informative, the rest is noise."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * noise
    return X, y


def _honest_oof_full(
    factory: Callable[[], Any], X: np.ndarray, y: np.ndarray, folds: int = 4
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Honest OOF vector (the fold model has not seen the row) + full-fit.

    Returns (y_pred_oof, y_pred_full, full_model).
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        model = factory()
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    full_model = factory().fit(X, y)
    return oof, full_model.predict(X), full_model


def _with_rmse(base: np.ndarray, target_rmse: float, rng: np.random.RandomState) -> np.ndarray:
    """A vector with the given RMSE relative to ``base`` (see Circuit D)."""
    noise = rng.randn(len(base))
    noise = noise / np.sqrt(np.mean(noise**2)) * target_rmse
    return base + noise


class _CoefModel:
    """A sub-model with only ``coef_`` — for the linear path of Circuit C."""

    def __init__(self, coef: Any) -> None:
        self.coef_ = np.asarray(coef, dtype=float)


class _BrokenPredictModel:
    """A model that fails in ``predict`` — tests exception interception."""

    def predict(self, X: Any) -> Any:
        raise RuntimeError("boom")


# ──────────────────────────────────────────────────────────────────────────────
#  Result structure
# ──────────────────────────────────────────────────────────────────────────────


def test_sanity_check_result_defaults_and_fields():
    result = SanityCheckResult(is_valid=True)
    assert result.is_valid is True
    assert result.reasons == []
    assert result.soft_warnings == []
    assert np.isnan(result.diversity_ratio)
    assert np.isnan(result.unique_ratio)
    assert result.dead_features_count == 0
    assert np.isnan(result.generalization_gap)
    assert result.severity == 0


def test_reasons_and_soft_warnings_are_distinct_channels():
    gate = ModelSanityGate(check_dead_features=False)
    y = np.arange(50, dtype=float)
    result = gate.check(y=y, y_pred_oof=np.full(50, 1.0), y_pred_full=y)
    assert any("Circuit A" in r for r in result.reasons)
    assert any("Circuit B" in r for r in result.reasons)
    # Signals and disqualifications do not mix.
    assert not any("signal" in r for r in result.reasons)


# ──────────────────────────────────────────────────────────────────────────────
#  Circuit A — Diversity Ratio
# ──────────────────────────────────────────────────────────────────────────────


def test_diversity_honest_model_passes_circuit_a():
    X, y = _dataset()
    oof, full, model = _honest_oof_full(lambda: SVR(kernel="rbf", C=10.0), X, y)
    gate = ModelSanityGate(
        check_unique=False, check_dead_features=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=oof, y_pred_full=full, X=X, model=model)
    assert result.is_valid
    assert result.diversity_ratio > 0.15
    assert result.reasons == []


def test_diversity_constant_predictions_fail():
    y = np.arange(100, dtype=float)
    gate = ModelSanityGate(
        check_unique=False, check_dead_features=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=np.full(100, y.mean()))
    assert not result.is_valid
    assert any("Circuit A" in r for r in result.reasons)
    assert result.diversity_ratio == 0.0


def test_diversity_boundary_equality_passes():
    # y = arange → exact variance; pred = c*y → ratio equals c².
    y = np.arange(200, dtype=float)
    c = np.sqrt(0.15)
    gate = ModelSanityGate(
        min_prediction_diversity=0.15,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    result = gate.check(y=y, y_pred_oof=c * y)
    assert result.is_valid
    assert result.diversity_ratio == pytest.approx(0.15, abs=1e-9)

    below = ModelSanityGate(
        min_prediction_diversity=0.15,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    res_below = below.check(y=y, y_pred_oof=np.sqrt(0.149) * y)
    assert not res_below.is_valid
    assert any("Circuit A" in r for r in res_below.reasons)


def test_diversity_constant_target_skips_with_warning():
    y = np.full(50, 5.0)
    gate = ModelSanityGate(
        check_unique=False, check_dead_features=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=np.full(50, 5.0))
    assert result.is_valid  # Circuit A skipped; Circuit B is evaluated
    assert any("target variance is zero" in w for w in result.soft_warnings)
    assert not any("Circuit A" in r for r in result.reasons)
    # The ratio is undefined (division by zero) — NaN, not 0.0.
    assert np.isnan(result.diversity_ratio)


def test_diversity_adaptive_mode_relative_to_constant_model():
    y = np.arange(100, dtype=float)
    gate = ModelSanityGate(
        adaptive_diversity=True,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    # Any positive spread is better than a constant model.
    result = gate.check(y=y, y_pred_oof=0.01 * y)
    assert result.is_valid
    assert result.diversity_ratio == pytest.approx(0.0001, rel=1e-6)
    # Zero spread — exactly like a constant model → failure.
    const = gate.check(y=y, y_pred_oof=np.full(100, y.mean()))
    assert not const.is_valid
    assert any("Circuit A" in r for r in const.reasons)


# ──────────────────────────────────────────────────────────────────────────────
#  Circuit B — Unique Ratio
# ──────────────────────────────────────────────────────────────────────────────


def test_unique_honest_predictions_pass():
    y = np.arange(100, dtype=float)
    gate = ModelSanityGate(
        check_diversity=False, check_dead_features=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=np.linspace(0, 100, 100))
    assert result.is_valid
    assert result.unique_ratio == pytest.approx(1.0)
    assert result.reasons == []


def test_unique_constant_predictions_fail():
    y = np.arange(100, dtype=float)
    gate = ModelSanityGate(
        check_diversity=False, check_dead_features=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=np.full(100, 42.0))
    assert not result.is_valid
    assert any("Circuit B" in r for r in result.reasons)
    assert result.unique_ratio == pytest.approx(1 / 100)


def test_unique_boundary_equal_to_min_unique_count_passes():
    y = np.arange(100, dtype=float)
    pred = np.tile(np.arange(5, dtype=float), 20)  # exactly 5 unique values
    gate = ModelSanityGate(
        min_unique_count=5,
        check_diversity=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    result = gate.check(y=y, y_pred_oof=pred)
    assert result.is_valid

    gate_strict = ModelSanityGate(
        min_unique_count=6,
        check_diversity=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    res_strict = gate_strict.check(y=y, y_pred_oof=pred)
    assert not res_strict.is_valid


def test_unique_ratio_rounding_scaled_by_target_std():
    # Large-scale y: the same 0.001 'jitter' in predictions is
    # indistinguishable after rounding (granularity = 10) → failure.
    y_big = np.arange(100, dtype=float) * 100.0  # std ≈ 2887 → decimals ≤ 0
    pred_big = 50000.0 + np.linspace(0.0, 0.099, 100)  # step 0.001
    gate = ModelSanityGate(
        check_diversity=False, check_dead_features=False, check_generalization_gap=False
    )
    res_big = gate.check(y=y_big, y_pred_oof=pred_big)
    assert not res_big.is_valid
    assert any("Circuit B" in r for r in res_big.reasons)

    # Small-scale y (std ≈ 0.29, granularity = 0.001): the same jitter is
    # significant → all 100 values are distinct → pass.
    y_small = np.arange(100, dtype=float) * 0.01
    pred_small = 5.0 + np.linspace(0.0, 0.099, 100)  # step 0.001
    res_small = gate.check(y=y_small, y_pred_oof=pred_small)
    assert res_small.is_valid
    assert res_small.unique_ratio == pytest.approx(1.0)


def test_unique_min_unique_ratio_lower_bound_applies():
    # N=1000: required = max(5, 0.05*1000) = 50 → 20 unique values — failure.
    y = np.arange(1000, dtype=float)
    pred = np.tile(np.arange(20, dtype=float), 50)
    gate = ModelSanityGate(
        check_diversity=False, check_dead_features=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=pred)
    assert not result.is_valid
    assert "min_unique_ratio" in result.reasons[0]


def test_unique_threshold_capped_by_nunique_y_issue_125():
    """Issue #125: the unique threshold must not exceed nunique(y).

    The reproduction from the issue: N=18 rows with 13 distinct target
    values and ``min_unique_ratio=0.9`` demand 16.2 ≈ 17 distinct
    predictions — more than even a perfect model (13) can produce, so the
    old check disqualified it. The ratio bound is capped by the number of
    distinct target values: a model predicting y exactly always reaches the
    cap and passes.
    """
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                 dtype=float)
    assert len(y) == 18
    assert len(np.unique(y)) == 13

    gate = ModelSanityGate(
        min_unique_ratio=0.9,
        check_diversity=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    # A perfect model (predictions == y) must pass despite the 0.9 ratio.
    result = gate.check(y=y, y_pred_oof=y)
    assert result.is_valid
    assert result.reasons == []
    # The reached nunique (13) equals the cap; the ratio is reported as-is.
    assert result.unique_ratio == pytest.approx(13 / 18)
    # The reason, if ever emitted, names the cap.
    # A model with fewer distinct predictions than the cap still fails.
    below = gate.check(y=y, y_pred_oof=np.tile(y[:12], 2)[:18])
    assert not below.is_valid
    assert any("capped by nunique(y)" in r for r in below.reasons)


def test_unique_threshold_capped_by_rounded_nunique_y_issue_125():
    """The cap uses the same rounding granularity as the predictions.

    Distinct target values are counted at the same ``np.round(decimals)``
    granularity that is applied to the predictions: a model predicting y
    exactly reaches exactly ``nunique(round(y))`` distinct rounded
    predictions — the cap — and passes by construction. Here a large-scale
    target (granularity 1000) collapses pairs of close values: 4 raw
    distinct values become 2 rounded buckets; a perfect model (2 distinct
    rounded predictions == the cap) passes, a constant prediction (1) does
    not.
    """
    y = np.array([0.0, 0.001, 1_000_000.0, 1_000_000.001] * 5, dtype=float)
    gate = ModelSanityGate(
        min_unique_ratio=1.0,
        check_diversity=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    # Perfect model: rounded nunique == rounded nunique(y) == the cap (2).
    result = gate.check(y=y, y_pred_oof=y)
    assert result.is_valid
    assert result.reasons == []
    assert result.unique_ratio == pytest.approx(2 / 20)
    # A constant prediction (1 distinct) still fails: 1 < cap 2.
    below = gate.check(y=y, y_pred_oof=np.full(20, 500_000.0))
    assert not below.is_valid
    assert any("capped by nunique(y)=2" in r for r in below.reasons)


def test_unique_min_unique_count_also_capped_by_nunique_y_issue_125():
    """min_unique_count above nunique(y) cannot disqualify a perfect model.

    A static count higher than the number of distinct target values is
    unreachable for the most expressive model; the final threshold is capped
    by nunique(y) as well (not only the ratio term).
    """
    y = np.tile(np.arange(4, dtype=float), 10)  # nunique(y) = 4
    gate = ModelSanityGate(
        min_unique_count=10,
        check_diversity=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    # Perfect model: 4 distinct predictions == the cap → passes.
    result = gate.check(y=y, y_pred_oof=y)
    assert result.is_valid
    assert result.reasons == []
    # A degenerate model (fewer than the cap) still fails.
    const = gate.check(y=y, y_pred_oof=np.full(40, 1.0))
    assert not const.is_valid
    assert any("Circuit B" in r for r in const.reasons)


# ──────────────────────────────────────────────────────────────────────────────
#  Circuit C — Dead Feature Check
# ──────────────────────────────────────────────────────────────────────────────


def test_dead_features_lasso_many_zeros_is_signal_not_disqualification():
    rng = np.random.RandomState(11)
    X = rng.randn(200, 30)
    y = 2.0 * X[:, 0] + 1.0 * X[:, 1] + rng.randn(200)
    lasso = Lasso(alpha=0.06, random_state=0).fit(X, y)
    gate = ModelSanityGate()
    # Take the zero-coefficient threshold from the gate itself so the test
    # stays in sync with the implementation.
    tol = gate.zero_coef_tolerance
    zero_ratio = float((np.abs(lasso.coef_) <= tol).mean())
    assert zero_ratio > 0.5  # ~57–60% zero coefficients

    result = gate.check(
        y=y, y_pred_oof=lasso.predict(X), y_pred_full=lasso.predict(X), X=X, model=lasso
    )
    assert result.is_valid
    assert result.reasons == []
    assert any("zero-coefficient ratio" in w for w in result.soft_warnings)
    assert result.dead_features_count == int((np.abs(lasso.coef_) <= tol).sum())


def test_dead_features_linear_combo_with_low_diversity_disqualifies():
    rng = np.random.RandomState(5)
    n = 100
    y = rng.randn(n)
    model = _CoefModel(np.zeros(10))  # 100% zeros
    gate = ModelSanityGate(check_unique=False, check_generalization_gap=False)
    result = gate.check(
        y=y,
        y_pred_oof=np.full(n, y.mean()),  # diversity = 0 → Circuit A fails
        y_pred_full=np.full(n, y.mean()),
        X=rng.randn(n, 10),
        model=model,
    )
    # The 'many zeros AND low diversity' combination: Circuit C only
    # disqualifies because Circuit A already failed (diversity_failed=True).
    assert not result.is_valid
    assert any("combination" in r and "Circuit C" in r for r in result.reasons)
    assert any("Circuit A" in r for r in result.reasons)
    # The signal is present too.
    assert any("zero-coefficient ratio" in w for w in result.soft_warnings)


def test_dead_features_combo_disabled_when_require_low_diversity_false():
    rng = np.random.RandomState(5)
    n = 100
    y = rng.randn(n)
    model = _CoefModel(np.zeros(10))
    gate = ModelSanityGate(
        dead_features_require_low_diversity=False,
        check_unique=False,
        check_generalization_gap=False,
    )
    result = gate.check(
        y=y,
        y_pred_oof=np.full(n, y.mean()),
        y_pred_full=np.full(n, y.mean()),
        X=rng.randn(n, 10),
        model=model,
    )
    # Disqualification by Circuit A only; Circuit C is a signal.
    assert not result.is_valid
    assert all("Circuit C" not in r for r in result.reasons)
    assert any("zero-coefficient ratio" in w for w in result.soft_warnings)


def test_dead_features_combo_inactive_reported_when_check_diversity_off():
    # check_diversity=False: the C+A combination cannot be computed — the
    # gate explicitly reports this via a soft_warning instead of silently
    # disabling the rule.
    rng = np.random.RandomState(5)
    n = 100
    y = rng.randn(n)
    model = _CoefModel(np.zeros(10))
    gate = ModelSanityGate(
        check_diversity=False,
        check_unique=False,
        check_generalization_gap=False,
    )
    result = gate.check(
        y=y,
        y_pred_oof=np.full(n, y.mean()),
        y_pred_full=np.full(n, y.mean()),
        X=rng.randn(n, 10),
        model=model,
    )
    assert result.is_valid  # without Circuit A there is no disqualification
    assert any("combination rule inactive" in w for w in result.soft_warnings)


def test_dead_features_zero_ratio_boundary_equality_no_warning():
    model = _CoefModel([1.0, 0.0, 0.0, 0.0])  # exactly 0.75 zeros
    y = np.arange(40, dtype=float)
    gate = ModelSanityGate(
        max_dead_feature_ratio=0.75,
        check_diversity=False,
        check_unique=False,
        check_generalization_gap=False,
    )
    result = gate.check(y=y, y_pred_oof=np.arange(40, dtype=float), model=model)
    assert result.is_valid
    assert result.soft_warnings == []  # equality with the threshold is not an excess


def test_dead_features_nonlinear_permutation_positive_rf():
    rng = np.random.RandomState(3)
    n, p = 150, 6
    X = rng.randn(n, p)
    y = X.sum(axis=1) + rng.randn(n)  # all features are informative
    rf = RandomForestRegressor(max_depth=4, n_estimators=60, random_state=0)
    oof, full, model = _honest_oof_full(
        lambda: RandomForestRegressor(max_depth=4, n_estimators=60, random_state=0),
        X,
        y,
    )
    result = ModelSanityGate(check_diversity=False, check_generalization_gap=False).check(
        y=y, y_pred_oof=oof, y_pred_full=full, X=X, model=model
    )
    assert result.is_valid
    assert result.dead_features_count == 0
    assert result.soft_warnings == []


def test_dead_features_nonlinear_correlated_noise_soft_warning_only():
    # 'Falsely dead' features (correlated columns): the model uses only the
    # first column, the rest are almost exact copies (the information comes
    # from a neighbor). Permuting the copies does not change the
    # predictions → a high share of 'dead' features → soft_warning, NOT a
    # disqualification.
    rng = np.random.RandomState(3)
    n, p = 120, 10
    x0 = rng.randn(n)
    X = np.column_stack([x0] + [x0 + rng.randn(n) * 0.05 for _ in range(p - 1)])
    y = 3.0 * x0 + rng.randn(n)

    class _FirstColumnModel:
        def predict(self, X: Any) -> np.ndarray:
            return np.asarray(X)[:, 0] * 3.0

    model = _FirstColumnModel()
    result = ModelSanityGate(check_generalization_gap=False).check(
        y=y, y_pred_oof=model.predict(X), y_pred_full=model.predict(X), X=X, model=model
    )
    assert result.is_valid  # not a disqualification
    assert result.dead_features_count == p - 1
    assert any("dead-feature ratio" in w for w in result.soft_warnings)


def test_dead_features_permutation_disabled_when_sensitivity_off():
    # check_permutation_sensitivity=False disables the permutation path of
    # Circuit C for non-linear models; the linear path via coef_ remains.
    rng = np.random.RandomState(3)
    n, p = 120, 10
    x0 = rng.randn(n)
    X = np.column_stack([x0] + [x0 + rng.randn(n) * 0.05 for _ in range(p - 1)])
    y = 3.0 * x0 + rng.randn(n)

    class _FirstColumnModel:
        def predict(self, X: Any) -> np.ndarray:
            return np.asarray(X)[:, 0] * 3.0

    model = _FirstColumnModel()
    # With sensitivity disabled, the permutation audit is not performed:
    # no 'dead' features are counted and there are no soft warnings.
    gate = ModelSanityGate(
        check_permutation_sensitivity=False,
        check_generalization_gap=False,
    )
    result = gate.check(
        y=y, y_pred_oof=model.predict(X), y_pred_full=model.predict(X), X=X, model=model
    )
    assert result.is_valid
    assert result.dead_features_count == 0
    assert result.soft_warnings == []

    # Control: the same input with sensitivity enabled yields 'dead'
    # features (the permutation path actually works).
    gate_on = ModelSanityGate(check_generalization_gap=False)
    result_on = gate_on.check(
        y=y, y_pred_oof=model.predict(X), y_pred_full=model.predict(X), X=X, model=model
    )
    assert result_on.dead_features_count == p - 1


def test_dead_features_permutation_disabled_keeps_linear_path():
    # Disabling permutation sensitivity does NOT affect the linear path of
    # Circuit C (a model with coef_): the signal and the combination work
    # as before.
    rng = np.random.RandomState(5)
    n, p = 200, 12
    X = rng.randn(n, p)
    y = X[:, 0] + rng.randn(n) * 0.01
    lasso = Lasso(alpha=1e3).fit(X, y)
    assert np.mean(np.abs(lasso.coef_) <= 1e-12) > 0.8  # many zeros
    gate = ModelSanityGate(
        check_permutation_sensitivity=False,
        max_dead_feature_ratio=0.4,
        check_diversity=False,
        check_unique=False,
        check_generalization_gap=False,
    )
    result = gate.check(
        y=y,
        y_pred_oof=lasso.predict(X),
        y_pred_full=lasso.predict(X),
        X=X,
        model=lasso,
    )
    assert result.is_valid  # a signal, not a disqualification
    assert result.dead_features_count > 0
    assert any("zero-coefficient ratio" in w for w in result.soft_warnings)


def test_dead_features_permutation_cost_controls_rows_and_features():
    rng = np.random.RandomState(9)
    n, p = 120, 40
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + rng.randn(n)
    rf = RandomForestRegressor(max_depth=3, n_estimators=30, random_state=0).fit(X, y)
    gate = ModelSanityGate(
        permutation_max_rows=40,
        permutation_max_features=5,
        permutation_repeats=2,
        permutation_seed=1,
        check_diversity=False,
        check_unique=False,
        check_generalization_gap=False,
    )
    result = gate.check(y=y, y_pred_oof=rf.predict(X), y_pred_full=rf.predict(X), X=X, model=rf)
    # At most 5 columns were checked.
    assert 0 <= result.dead_features_count <= 5


def test_dead_features_permutation_deterministic_with_seed():
    rng = np.random.RandomState(9)
    n, p = 120, 12
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + rng.randn(n)
    rf = RandomForestRegressor(max_depth=3, n_estimators=30, random_state=0).fit(X, y)
    gate = ModelSanityGate(
        permutation_max_rows=60,
        permutation_max_features=8,
        permutation_repeats=3,
        permutation_seed=123,
        check_diversity=False,
        check_unique=False,
        check_generalization_gap=False,
    )
    r1 = gate.check(y=y, y_pred_oof=rf.predict(X), y_pred_full=rf.predict(X), X=X, model=rf)
    r2 = gate.check(y=y, y_pred_oof=rf.predict(X), y_pred_full=rf.predict(X), X=X, model=rf)
    # NaN fields do not participate in comparison (nan != nan); compare the essentials.
    assert r1.is_valid == r2.is_valid
    assert r1.reasons == r2.reasons
    assert r1.soft_warnings == r2.soft_warnings
    assert r1.dead_features_count == r2.dead_features_count
    assert r1.severity == r2.severity


def test_dead_features_permutation_works_with_dataframe_x():
    rng = np.random.RandomState(4)
    n, p = 100, 6
    X = pd.DataFrame(rng.randn(n, p), columns=[f"f{i}" for i in range(p)])
    y = X["f0"] * 2.0 + rng.randn(n)
    rf = RandomForestRegressor(max_depth=4, n_estimators=40, random_state=0).fit(X, y)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=rf.predict(X), y_pred_full=rf.predict(X), X=X, model=rf)
    assert result.is_valid


def test_dead_features_permutation_works_with_sparse_x():
    # Sparse X is a standard output of the repository's preprocessing
    # (HashingEncodingTransformer → csr_matrix). Permutations must not fail
    # on np.array(sparse) (a 0-D object array).
    from scipy import sparse

    rng = np.random.RandomState(0)
    n, p = 120, 10
    X = sparse.csr_matrix(rng.randn(n, p))
    y = np.asarray(X[:, 0].toarray()).ravel() * 2.0 + rng.randn(n)
    rf = RandomForestRegressor(max_depth=4, n_estimators=40, random_state=0).fit(X, y)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(
        y=y, y_pred_oof=rf.predict(X), y_pred_full=rf.predict(X), X=X, model=rf
    )
    assert result.is_valid
    assert 0 <= result.dead_features_count <= p


def test_dead_features_exception_in_predict_becomes_reason():
    rng = np.random.RandomState(0)
    X = rng.randn(60, 4)
    y = rng.randn(60)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=_BrokenPredictModel(), X=X)
    assert not result.is_valid
    assert any("prediction failed" in r for r in result.reasons)


def test_dead_features_nonlinear_without_x_is_reason():
    rng = np.random.RandomState(0)
    y = rng.randn(60)
    rf = RandomForestRegressor(max_depth=3, random_state=0)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=rf, X=None)
    assert not result.is_valid
    assert any("X is not provided" in r for r in result.reasons)


def test_dead_features_model_not_provided_is_reason():
    y = np.arange(50, dtype=float)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y)
    assert not result.is_valid
    assert any("model is not provided" in r for r in result.reasons)


def test_dead_features_pipeline_unwrapped_to_linear_estimator():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(6)
    n, p = 100, 8
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + rng.randn(n)
    pipe = make_pipeline(StandardScaler(), Lasso(alpha=0.3, random_state=0)).fit(X, y)
    gate = ModelSanityGate(check_diversity=False, check_generalization_gap=False)
    tol = gate.zero_coef_tolerance
    zero_ratio = float((np.abs(pipe.named_steps["lasso"].coef_) <= tol).mean())
    result = gate.check(
        y=y, y_pred_oof=pipe.predict(X), y_pred_full=pipe.predict(X), X=X, model=pipe
    )
    # The pipeline is unwrapped to the final estimator: count zeros of its coef_.
    assert result.dead_features_count == int(zero_ratio * p)
    # The zero share is noticeable → a signal; Circuit A is off, so the
    # C+A combination is not applied — there must be no Circuit C reasons.
    assert any("zero-coefficient ratio" in w for w in result.soft_warnings)
    assert not any("Circuit C" in r for r in result.reasons)
    assert "combination rule inactive" in " ".join(result.soft_warnings)


# ──────────────────────────────────────────────────────────────────────────────
#  Circuit D — Generalization Gap
# ──────────────────────────────────────────────────────────────────────────────


def _gap_only_gate(max_gap: float, mode: str = "reject") -> ModelSanityGate:
    # The default mode here is "reject": the existing circuit D tests fix the
    # STRICT behavior of the old default (parity). The new warn-mode tests
    # pass mode="warn" explicitly.
    return ModelSanityGate(
        max_generalization_gap=max_gap,
        generalization_gap_mode=mode,
        check_diversity=False,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=True,
    )


def test_gap_honest_model_passes():
    X, y = _dataset(n=500)
    oof, full, model = _honest_oof_full(lambda: SVR(kernel="rbf", C=3.0), X, y)
    result = ModelSanityGate(
        check_diversity=False, check_unique=False, check_dead_features=False
    ).check(y=y, y_pred_oof=oof, y_pred_full=full, X=X, model=model)
    assert result.is_valid
    assert 1.0 <= result.generalization_gap <= 1.5


def test_gap_issue_example_0_1_vs_0_19_fails_at_1_5_passes_at_2_0():
    # Example: RMSE_full=0.1, RMSE_oof=0.19 — clear overfitting.
    rng = np.random.RandomState(1)
    y = rng.randn(500) * 10.0
    full = _with_rmse(y, 0.1, rng)
    oof = _with_rmse(y, 0.19, rng)
    strict = _gap_only_gate(1.5).check(y=y, y_pred_oof=oof, y_pred_full=full)
    assert not strict.is_valid
    assert any("Circuit D" in r for r in strict.reasons)
    assert strict.generalization_gap == pytest.approx(1.9, rel=1e-6)

    soft = _gap_only_gate(2.0).check(y=y, y_pred_oof=oof, y_pred_full=full)
    assert soft.is_valid


def test_gap_deep_tree_rmse_full_zero_fails():
    rng = np.random.RandomState(2)
    X = rng.randn(120, 6)
    y = 2.0 * X[:, 0] + rng.randn(120)
    tree = DecisionTreeRegressor(max_depth=20, random_state=0).fit(X, y)
    full = tree.predict(X)
    # Scenario precondition: a depth-20 tree fits the train set perfectly
    # (RMSE_full ≈ 0 within float precision).
    assert np.sqrt(np.mean((y - full) ** 2)) < 1e-6
    oof = full + rng.randn(120) * 0.5
    result = _gap_only_gate(1.5).check(y=y, y_pred_oof=oof, y_pred_full=full)
    assert not result.is_valid
    assert any("division by zero" in r or "RMSE_full" in r for r in result.reasons)
    assert np.isinf(result.generalization_gap)


def test_gap_one_nn_fails():
    rng = np.random.RandomState(2)
    X = rng.randn(120, 6)
    y = 2.0 * X[:, 0] + rng.randn(120)
    knn = KNeighborsRegressor(n_neighbors=1).fit(X, y)
    full = knn.predict(X)
    # Precondition: 1-NN has zero error on the train set (RMSE_full ≈ 0).
    assert np.sqrt(np.mean((y - full) ** 2)) < 1e-6
    oof = full + rng.randn(120) * 0.5
    result = _gap_only_gate(1.5).check(y=y, y_pred_oof=oof, y_pred_full=full)
    assert not result.is_valid
    assert any("Circuit D" in r for r in result.reasons)


def test_gap_boundary_equality_passes():
    rng = np.random.RandomState(3)
    y = rng.randn(300) * 10.0
    full = _with_rmse(y, 1.0, rng)
    oof = _with_rmse(y, 1.0, rng)  # gap = 1.0 exactly
    result = _gap_only_gate(1.0).check(y=y, y_pred_oof=oof, y_pred_full=full)
    assert result.is_valid
    assert result.generalization_gap == pytest.approx(1.0, rel=1e-6)

    oof_over = _with_rmse(y, 2.0, rng)  # gap = 2.0 > 1.0 → failure
    over = _gap_only_gate(1.0).check(y=y, y_pred_oof=oof_over, y_pred_full=full)
    assert not over.is_valid


def test_gap_missing_y_pred_full_is_reason():
    y = np.arange(50, dtype=float)
    result = _gap_only_gate(1.5).check(y=y, y_pred_oof=y)
    assert not result.is_valid
    assert any("y_pred_full is not provided" in r for r in result.reasons)


def test_gap_rmse_oof_zero_is_division_by_zero_reason():
    y = np.arange(100, dtype=float)
    result = _gap_only_gate(1.5).check(y=y, y_pred_oof=y, y_pred_full=y)
    assert not result.is_valid
    assert any("RMSE_oof" in r and "division by zero" in r for r in result.reasons)
    assert np.isinf(result.generalization_gap)


# ──────────────────────────────────────────────────────────────────────────────
#  Circuit D — warn / reject modes
# ──────────────────────────────────────────────────────────────────────────────


def test_gap_warn_mode_finite_over_threshold_is_soft_warning_not_reason():
    """Warn mode (default): a finite gap above the threshold is a soft_warning, not a reason.

    ``is_valid`` stays True, severity == 1, ``generalization_gap`` is recorded.
    In reject mode the same input is a reason and ``is_valid == False``.
    """
    rng = np.random.RandomState(11)
    y = rng.randn(500) * 10.0
    full = _with_rmse(y, 1.0, rng)
    oof = _with_rmse(y, 1.9, rng)  # gap ≈ 1.9 > 1.5

    warn = _gap_only_gate(1.5, mode="warn").check(y=y, y_pred_oof=oof, y_pred_full=full)
    assert warn.is_valid is True
    assert warn.reasons == []
    assert any(
        "Circuit D" in w and "max_generalization_gap=1.500" in w
        for w in warn.soft_warnings
    )
    assert warn.severity == 1
    assert warn.generalization_gap == pytest.approx(1.9, rel=1e-6)

    strict = _gap_only_gate(1.5, mode="reject").check(
        y=y, y_pred_oof=oof, y_pred_full=full
    )
    assert strict.is_valid is False
    assert any("Circuit D" in r for r in strict.reasons)
    assert strict.soft_warnings == []
    assert strict.severity == 2


def test_gap_warn_mode_deep_tree_division_by_zero_is_soft_warning():
    """Warn mode: RMSE_full≈0 (deep tree) — soft_warning, not a reason.

    Zero training error is a normal property of a whole class of models and
    says nothing about quality on new data: in warn mode the model is valid
    (severity == 1); in reject mode it is a reason (parity with the old
    behavior).
    """
    rng = np.random.RandomState(12)
    X = rng.randn(120, 6)
    y = 2.0 * X[:, 0] + rng.randn(120)
    tree = DecisionTreeRegressor(max_depth=20, random_state=0).fit(X, y)
    full = tree.predict(X)
    assert np.sqrt(np.mean((y - full) ** 2)) < 1e-6
    oof = full + rng.randn(120) * 0.5

    warn = _gap_only_gate(1.5, mode="warn").check(
        y=y, y_pred_oof=oof, y_pred_full=full
    )
    assert warn.is_valid is True
    assert warn.reasons == []
    assert any("Circuit D" in w and "RMSE_full" in w for w in warn.soft_warnings)
    assert np.isinf(warn.generalization_gap)
    assert warn.severity == 1

    strict = _gap_only_gate(1.5, mode="reject").check(
        y=y, y_pred_oof=oof, y_pred_full=full
    )
    assert not strict.is_valid
    assert any(
        "Circuit D" in r and "division by zero" in r for r in strict.reasons
    )
    assert np.isinf(strict.generalization_gap)


def test_gap_warn_mode_one_nn_division_by_zero_is_soft_warning():
    """Warn mode: 1-NN (a perfect in-sample fit) — soft_warning."""
    rng = np.random.RandomState(12)
    X = rng.randn(120, 6)
    y = 2.0 * X[:, 0] + rng.randn(120)
    knn = KNeighborsRegressor(n_neighbors=1).fit(X, y)
    full = knn.predict(X)
    assert np.sqrt(np.mean((y - full) ** 2)) < 1e-6
    oof = full + rng.randn(120) * 0.5

    warn = _gap_only_gate(1.5, mode="warn").check(
        y=y, y_pred_oof=oof, y_pred_full=full
    )
    assert warn.is_valid is True
    assert warn.reasons == []
    assert any("Circuit D" in w and "RMSE_full" in w for w in warn.soft_warnings)
    assert np.isinf(warn.generalization_gap)
    assert warn.severity == 1

    strict = _gap_only_gate(1.5, mode="reject").check(
        y=y, y_pred_oof=oof, y_pred_full=full
    )
    assert not strict.is_valid
    assert any("Circuit D" in r for r in strict.reasons)
    assert np.isinf(strict.generalization_gap)


def test_gap_warn_mode_missing_y_pred_full_is_reason():
    """Warn mode: y_pred_full=None — a contract violation, a reason."""
    y = np.arange(50, dtype=float)
    for mode in ("warn", "reject"):
        result = _gap_only_gate(1.5, mode=mode).check(y=y, y_pred_oof=y)
        assert not result.is_valid, mode
        assert any("y_pred_full is not provided" in r for r in result.reasons)


def test_gap_warn_mode_rmse_oof_zero_is_soft_warning():
    """Warn mode: RMSE_oof≈0 — a soft_warning about target leakage.

    In warn mode the model is valid and the warning explicitly names the
    suspected target leakage; in reject mode it is a reason (parity with the
    old behavior).
    """
    y = np.arange(100, dtype=float)
    warn = _gap_only_gate(1.5, mode="warn").check(y=y, y_pred_oof=y, y_pred_full=y)
    assert warn.is_valid is True
    assert warn.reasons == []
    assert any("target leakage" in w for w in warn.soft_warnings)
    assert np.isinf(warn.generalization_gap)
    assert warn.severity == 1

    strict = _gap_only_gate(1.5, mode="reject").check(
        y=y, y_pred_oof=y, y_pred_full=y
    )
    assert not strict.is_valid
    assert any("RMSE_oof" in r and "division by zero" in r for r in strict.reasons)
    assert np.isinf(strict.generalization_gap)


def test_gap_boundary_equality_passes_in_both_modes():
    """Boundary: gap == max_generalization_gap exactly — not a failure in both modes.

    The ``_fails_above`` semantics: a strict excess; equality always passes.
    """
    rng = np.random.RandomState(13)
    y = rng.randn(300) * 10.0
    full = _with_rmse(y, 2.0, rng)
    oof = _with_rmse(y, 2.0, rng)  # gap = 1.0 exactly at a threshold of 1.0
    for mode in ("warn", "reject"):
        result = _gap_only_gate(1.0, mode=mode).check(
            y=y, y_pred_oof=oof, y_pred_full=full
        )
        assert result.is_valid, mode
        assert result.reasons == []
        assert result.soft_warnings == []
        assert result.generalization_gap == pytest.approx(1.0, rel=1e-6)


def test_gap_warn_mode_above_threshold_boundary():
    """Boundary: gap == 1.5 + ε — a warning in warn mode, a reason in reject mode."""
    rng = np.random.RandomState(14)
    y = rng.randn(300) * 10.0
    full = _with_rmse(y, 1.0, rng)
    oof = _with_rmse(y, 1.5 + 1e-3, rng)  # gap = 1.501 > 1.5
    warn = _gap_only_gate(1.5, mode="warn").check(y=y, y_pred_oof=oof, y_pred_full=full)
    assert warn.is_valid is True
    assert any("Circuit D" in w for w in warn.soft_warnings)
    strict = _gap_only_gate(1.5, mode="reject").check(
        y=y, y_pred_oof=oof, y_pred_full=full
    )
    assert strict.is_valid is False
    assert any("Circuit D" in r for r in strict.reasons)


def test_gap_inf_boundary_rmse_full_zero_soft_warning_in_warn_mode():
    """Boundary: RMSE_full==0 exactly with RMSE_oof>0 — soft_warning.

    In warn mode the model passes the audit (the infinite gap does not
    disqualify); in reject mode the same input is a reason with the
    RMSE_full message.
    """
    rng = np.random.RandomState(15)
    y = np.arange(100, dtype=float) * 1.0
    full = y.copy()  # RMSE_full == 0 exactly
    oof = y + rng.randn(100) * 1e-3  # RMSE_oof ≈ 1e-3 > _ZERO_RMSE_EPS
    warn = _gap_only_gate(1.5, mode="warn").check(
        y=y, y_pred_oof=oof, y_pred_full=full
    )
    assert warn.is_valid is True
    assert warn.reasons == []
    assert any(
        "RMSE_full" in w and "division by zero" in w for w in warn.soft_warnings
    )
    assert np.isinf(warn.generalization_gap)

    strict = _gap_only_gate(1.5, mode="reject").check(
        y=y, y_pred_oof=oof, y_pred_full=full
    )
    assert not strict.is_valid
    assert any(
        "RMSE_full" in r and "division by zero" in r for r in strict.reasons
    )
    assert np.isinf(strict.generalization_gap)


def test_gap_invalid_mode_raises_value_error():
    """An unknown generalization_gap_mode value is rejected."""
    with pytest.raises(ValueError, match="generalization_gap_mode"):
        ModelSanityGate(generalization_gap_mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="generalization_gap_mode"):
        ModelSanityGate(generalization_gap_mode="")  # type: ignore[arg-type]


def test_gap_reject_mode_parity_with_legacy_default():
    """Parity: reject mode = the old strict default behavior.

    An explicit ``generalization_gap_mode="reject"`` fully reproduces the old
    default: an overfit model is disqualified, an honest one passes, division
    by zero is a reason.
    """
    rng = np.random.RandomState(16)
    y = rng.randn(500) * 10.0
    full = _with_rmse(y, 0.1, rng)
    oof = _with_rmse(y, 0.19, rng)  # gap = 1.9 > 1.5
    strict = _gap_only_gate(1.5, mode="reject").check(
        y=y, y_pred_oof=oof, y_pred_full=full
    )
    assert not strict.is_valid
    assert any("Circuit D" in r for r in strict.reasons)
    assert strict.generalization_gap == pytest.approx(1.9, rel=1e-6)

    honest_y = rng.randn(200) * 10.0
    honest_full = _with_rmse(honest_y, 1.0, rng)
    honest_oof = _with_rmse(honest_y, 1.2, rng)  # gap = 1.2 <= 1.5
    ok = _gap_only_gate(1.5, mode="reject").check(
        y=honest_y, y_pred_oof=honest_oof, y_pred_full=honest_full
    )
    assert ok.is_valid
    assert ok.reasons == []
    assert ok.soft_warnings == []


# ──────────────────────────────────────────────────────────────────────────────
#  Circuit D — division by zero in warn mode
# ──────────────────────────────────────────────────────────────────────────────


# Models with zero training error (RMSE_full ≈ 0): a perfect in-sample fit
# is a normal property of a class of models and says nothing about quality
# on new data. In warn mode they are NOT disqualified.
_GAP_INF_MODEL_FACTORIES = [
    lambda: DecisionTreeRegressor(max_depth=20, random_state=0),
    lambda: KNeighborsRegressor(n_neighbors=1),
    lambda: ExtraTreesRegressor(min_samples_leaf=1, random_state=0),
    lambda: KNeighborsRegressor(n_neighbors=3, weights="distance"),
]


@pytest.mark.parametrize("factory", _GAP_INF_MODEL_FACTORIES)
def test_gap_warn_mode_division_by_zero_soft_warning_models(factory: Callable[[], Any]):
    """Positive (warn): models with RMSE_full≈0 pass the audit.

    is_valid is True, a Circuit D warning about RMSE_full≈0 in soft_warnings,
    generalization_gap == inf, severity == 1.
    """
    rng = np.random.RandomState(12)
    X = rng.randn(120, 6)
    y = 2.0 * X[:, 0] + rng.randn(120)
    oof, full, model = _honest_oof_full(factory, X, y, folds=5)
    assert np.sqrt(np.mean((y - full) ** 2)) < 1e-6  # precondition: RMSE_full≈0
    warn = _gap_only_gate(1.5, mode="warn").check(
        y=y, y_pred_oof=oof, y_pred_full=full, X=X, model=model
    )
    assert warn.is_valid is True
    assert warn.reasons == []
    assert any("Circuit D" in w and "RMSE_full" in w for w in warn.soft_warnings)
    assert np.isinf(warn.generalization_gap)
    assert warn.severity == 1


@pytest.mark.parametrize("factory", _GAP_INF_MODEL_FACTORIES)
def test_gap_reject_mode_division_by_zero_models_disqualified(
    factory: Callable[[], Any],
):
    """Negative (reject): the same models are disqualified.

    The reason texts match the current ones (parity with the old behavior).
    """
    rng = np.random.RandomState(12)
    X = rng.randn(120, 6)
    y = 2.0 * X[:, 0] + rng.randn(120)
    oof, full, model = _honest_oof_full(factory, X, y, folds=5)
    assert np.sqrt(np.mean((y - full) ** 2)) < 1e-6
    strict = _gap_only_gate(1.5, mode="reject").check(
        y=y, y_pred_oof=oof, y_pred_full=full, X=X, model=model
    )
    assert not strict.is_valid
    assert any(
        "Circuit D" in r and "RMSE_full" in r and "division by zero" in r
        for r in strict.reasons
    )
    assert np.isinf(strict.generalization_gap)


def test_gap_warn_mode_rmse_oof_zero_logs_warning(caplog):
    """Warn mode: RMSE_oof≈0 → a WARNING in the log (a deliberate exception).

    A silent disqualification would replace the winner and hide the target
    leakage; a noticeable WARNING-level signal is needed instead.
    """
    y = np.arange(100, dtype=float)
    with caplog.at_level("WARNING", logger="configurable_automl_engine"):
        warn = _gap_only_gate(1.5, mode="warn").check(
            y=y, y_pred_oof=y, y_pred_full=y
        )
    assert warn.is_valid is True
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("target leakage" in r.message for r in warning_records)

    # Contrast: RMSE_full≈0 (not leakage) in warn mode does NOT produce a WARNING.
    rng = np.random.RandomState(17)
    full = y.copy()
    oof = y + rng.randn(100) * 1e-3
    caplog.clear()
    with caplog.at_level("WARNING", logger="configurable_automl_engine"):
        _gap_only_gate(1.5, mode="warn").check(y=y, y_pred_oof=oof, y_pred_full=full)
    assert not any("target leakage" in r.message for r in caplog.records)


def test_gap_warn_boundary_rmse_full_zero_and_above_eps_both_pass():
    """The _ZERO_RMSE_EPS boundary: RMSE_full == 0 exactly and 2e-12 (just above eps).

    In warn mode both candidates pass the audit; only the gap value differs
    (inf or a finite large one).
    """
    rng = np.random.RandomState(18)
    y = np.arange(100, dtype=float) * 1.0
    oof = y + rng.randn(100) * 1e-3  # RMSE_oof ≈ 1e-3 > eps
    zero_full = _gap_only_gate(1.5, mode="warn").check(
        y=y, y_pred_oof=oof, y_pred_full=y.copy()
    )
    assert zero_full.is_valid is True
    assert np.isinf(zero_full.generalization_gap)

    above_full = y + rng.randn(100) * 2e-12  # RMSE_full ≈ 2e-12 > _ZERO_RMSE_EPS
    assert np.sqrt(np.mean((y - above_full) ** 2)) > 1e-12
    near = _gap_only_gate(1.5, mode="warn").check(
        y=y, y_pred_oof=oof, y_pred_full=above_full
    )
    assert near.is_valid is True
    assert np.isfinite(near.generalization_gap)
    assert near.generalization_gap > 1.5  # a finite large gap → warning


def test_gap_rmse_full_and_oof_both_zero_single_leakage_warning():
    """RMSE_full ≈ RMSE_oof ≈ 0: a single branch (RMSE_oof) — no duplicate RMSE_full message.

    In warn mode — a single leakage warning; in reject mode — a single
    reason (RMSE_oof).
    """
    y = np.arange(100, dtype=float)
    warn = _gap_only_gate(1.5, mode="warn").check(y=y, y_pred_oof=y, y_pred_full=y)
    assert warn.is_valid is True
    assert len(warn.soft_warnings) == 1
    assert "target leakage" in warn.soft_warnings[0]
    assert not any("RMSE_full" in w for w in warn.soft_warnings)

    strict = _gap_only_gate(1.5, mode="reject").check(
        y=y, y_pred_oof=y, y_pred_full=y
    )
    assert not strict.is_valid
    assert len(strict.reasons) == 1
    assert "RMSE_oof" in strict.reasons[0]


def test_gap_no_finite_pairs_is_reason_in_both_modes():
    """No finite pairs (y, oof, full) — a reason in both modes (uncomputable)."""
    y = np.arange(50, dtype=float)
    for mode in ("warn", "reject"):
        result = _gap_only_gate(1.5, mode=mode).check(
            y=y, y_pred_oof=y, y_pred_full=np.full(50, np.nan)
        )
        assert not result.is_valid, mode
        assert any("no finite pairs to compute RMSE" in r for r in result.reasons)
        assert result.soft_warnings == [], mode


# ──────────────────────────────────────────────────────────────────────────────
#  Complex scenarios (positive / negative)
# ──────────────────────────────────────────────────────────────────────────────


def test_positive_honest_nonlinear_passes_all_circuits_clean_data():
    rng = np.random.RandomState(7)
    n = 500
    X = rng.randn(n, 6)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(n)
    oof, full, model = _honest_oof_full(
        lambda: RandomForestRegressor(max_depth=5, n_estimators=100, random_state=0),
        X,
        y,
    )
    result = ModelSanityGate().check(y=y, y_pred_oof=oof, y_pred_full=full, X=X, model=model)
    assert result.is_valid
    assert result.reasons == []
    assert result.diversity_ratio > 0.15
    assert result.generalization_gap <= 1.5
    assert result.dead_features_count == 0


def test_positive_honest_model_passes_on_noisy_r2_about_0_3():
    # Noisy data with a theoretical R²≈0.3: an honest model is NOT
    # disqualified (the soft diversity threshold is a key v2 change).
    # Var(signal)=6 (the sum of 6 independent N(0,1)); we pick the noise so
    # that R² = Var(signal)/(Var(signal)+Var(noise)) = 0.3 exactly.
    rng = np.random.RandomState(3)
    n = 200
    X = rng.randn(n, 6)
    signal = X.sum(axis=1)
    noise_var = 6.0 * (1.0 / 0.3 - 1.0)
    y = signal + rng.randn(n) * np.sqrt(noise_var)
    oof, full, model = _honest_oof_full(lambda: SVR(kernel="rbf", C=2.0), X, y)
    result = ModelSanityGate().check(y=y, y_pred_oof=oof, y_pred_full=full, X=X, model=model)
    assert result.is_valid
    assert result.reasons == []
    assert 0.1 < result.diversity_ratio < 0.5  # ≈ R², well above the strict 0.15
    assert result.generalization_gap <= 1.5


def test_negative_elasticnet_zero_weights_and_zero_diversity():
    # A real ElasticNet with heavy regularization zeroes ALL weights (we check
    # the precondition); the gate itself receives a copy of those weights
    # via _CoefModel so the result does not depend on the sklearn version.
    rng = np.random.RandomState(5)
    X = rng.randn(120, 15)
    y = 2.0 * X[:, 0] + rng.randn(120)
    en = ElasticNet(alpha=500.0, random_state=0).fit(X, y)
    assert float((np.abs(en.coef_) <= 1e-12).mean()) == 1.0  # all weights are zero
    model = _CoefModel(en.coef_)
    flat = np.full(120, y.mean())
    result = ModelSanityGate(check_generalization_gap=False).check(
        y=y, y_pred_oof=flat, y_pred_full=en.predict(X), X=X, model=model
    )
    assert not result.is_valid
    assert any("Circuit A" in r for r in result.reasons)
    assert any("Circuit C" in r and "combination" in r for r in result.reasons)


def test_negative_svr_sigmoid_tanh_plateau():
    # Scenario: SVR with a sigmoid kernel, small gamma and coef0=0 hits the
    # tanh plateau and produces almost constant predictions. First we check
    # the precondition (prediction spread ≪ y spread), then that Circuit A
    # detects it.
    X, y = _dataset(n=150)
    sigmoid = SVR(kernel="sigmoid", gamma=0.001, coef0=0.0).fit(X, y)
    pred = sigmoid.predict(X)
    assert pred.std() < 0.1 * y.std()  # tanh plateau: predictions are almost constant
    result = ModelSanityGate(check_generalization_gap=False).check(
        y=y, y_pred_oof=pred, y_pred_full=pred, X=X, model=sigmoid
    )
    assert not result.is_valid
    assert any("Circuit A" in r for r in result.reasons)
    assert result.diversity_ratio < 0.15


def test_positive_deep_tree_rmse_full_zero_soft_warning_in_warn_mode():
    """Warn mode (default): a deep tree — Circuit D soft_warning."""
    X, y = _dataset(n=120)
    tree = DecisionTreeRegressor(max_depth=20, random_state=0).fit(X, y)
    result = ModelSanityGate(check_diversity=False, check_unique=False).check(
        y=y, y_pred_oof=tree.predict(X), y_pred_full=tree.predict(X), X=X, model=tree
    )
    assert result.is_valid
    assert any("Circuit D" in w for w in result.soft_warnings)


def test_positive_one_nn_soft_warning_in_warn_mode():
    """Warn mode (default): 1-NN — Circuit D soft_warning."""
    X, y = _dataset(n=120)
    knn = KNeighborsRegressor(n_neighbors=1).fit(X, y)
    result = ModelSanityGate(check_diversity=False, check_unique=False).check(
        y=y, y_pred_oof=knn.predict(X), y_pred_full=knn.predict(X), X=X, model=knn
    )
    assert result.is_valid
    assert any("Circuit D" in w for w in result.soft_warnings)


# ──────────────────────────────────────────────────────────────────────────────
#  Modes and edge cases
# ──────────────────────────────────────────────────────────────────────────────


def test_warn_only_returns_counts_without_disqualification():
    y = np.arange(100, dtype=float)
    gate = ModelSanityGate(warn_only=True, check_dead_features=False)
    result = gate.check(y=y, y_pred_oof=np.full(100, y.mean()), y_pred_full=y)
    assert result.is_valid is True  # no right to disqualify
    assert len(result.reasons) >= 2  # reasons are counted for the report
    assert result.severity == 1  # capped at the 'warning' level
    assert result.diversity_ratio == 0.0


def test_all_circuits_disabled_always_valid():
    gate = ModelSanityGate(
        check_diversity=False,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    result = gate.check(y=np.arange(50, dtype=float), y_pred_oof=np.full(50, 1.0))
    assert result.is_valid
    assert result.reasons == []
    assert result.soft_warnings == []
    assert np.isnan(result.diversity_ratio)
    assert np.isnan(result.unique_ratio)
    assert result.dead_features_count == 0
    assert np.isnan(result.generalization_gap)
    assert result.severity == 0


def test_nan_predictions_are_masked_like_oof_rmse():
    # Structured predictions: each row is unique after rounding, so
    # unique_ratio = 1.0 both with and without the mask — strict comparison.
    rng = np.random.RandomState(0)
    n = 100
    y = rng.randn(n) * 10.0
    pred = np.linspace(-5.0, 5.0, n)  # step ≈ 0.1 > rounding granularity
    pred_masked = pred.copy()
    pred_masked[::5] = np.nan  # 20% uncovered rows
    gate = ModelSanityGate(
        check_dead_features=False, check_generalization_gap=False
    )
    full_result = gate.check(y=y, y_pred_oof=pred, y_pred_full=y)
    masked_result = gate.check(y=y, y_pred_oof=pred_masked, y_pred_full=y)
    # Masking does not change the semantics: both pass/fail identically,
    # and the unique share among valid rows is the same.
    assert full_result.is_valid == masked_result.is_valid
    assert masked_result.unique_ratio == pytest.approx(1.0)
    assert masked_result.unique_ratio == pytest.approx(full_result.unique_ratio)


def test_all_nan_predictions_fail_with_reason():
    y = np.arange(50, dtype=float)
    gate = ModelSanityGate(check_dead_features=False, check_generalization_gap=False)
    result = gate.check(y=y, y_pred_oof=np.full(50, np.nan))
    assert not result.is_valid
    assert any("no finite" in r for r in result.reasons)


def test_nan_in_y_is_masked():
    rng = np.random.RandomState(0)
    y = rng.randn(60)
    y[::10] = np.nan
    gate = ModelSanityGate(
        check_diversity=True,
        check_unique=True,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    result = gate.check(y=y, y_pred_oof=rng.randn(60))
    assert result.is_valid  # computed over finite pairs, without raising


def test_severity_mapping():
    y = np.arange(100, dtype=float)
    one_reason = ModelSanityGate(
        check_diversity=True,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=False,
    ).check(y=y, y_pred_oof=np.full(100, y.mean()))
    assert one_reason.severity == 2

    two_reasons = ModelSanityGate(
        check_diversity=True,
        check_unique=True,
        check_dead_features=False,
        check_generalization_gap=False,
    ).check(y=y, y_pred_oof=np.full(100, y.mean()))
    assert two_reasons.severity == 3

    three_reasons = ModelSanityGate(
        check_diversity=True,
        check_unique=True,
        check_dead_features=True,
        check_generalization_gap=False,
    ).check(
        y=y,
        y_pred_oof=np.full(100, y.mean()),
        model=_CoefModel(np.zeros(4)),
    )
    assert three_reasons.severity == 4

    clean = ModelSanityGate(
        check_diversity=False,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=False,
    ).check(y=y, y_pred_oof=np.arange(100, dtype=float))
    assert clean.severity == 0


def test_constructor_validation():
    with pytest.raises(ValueError):
        ModelSanityGate(min_prediction_diversity=-0.1)
    with pytest.raises(ValueError):
        ModelSanityGate(min_prediction_diversity=0)
    with pytest.raises(ValueError):
        ModelSanityGate(min_prediction_diversity=1.5)
    with pytest.raises(ValueError):
        ModelSanityGate(min_unique_count=0)
    with pytest.raises(ValueError):
        ModelSanityGate(min_unique_ratio=0)
    with pytest.raises(ValueError):
        ModelSanityGate(min_unique_ratio=1.5)
    with pytest.raises(ValueError):
        ModelSanityGate(max_dead_feature_ratio=-0.1)
    with pytest.raises(ValueError):
        ModelSanityGate(max_generalization_gap=0.5)
    with pytest.raises(ValueError):
        ModelSanityGate(permutation_repeats=0)
    with pytest.raises(ValueError):
        ModelSanityGate(permutation_max_rows=0)
    with pytest.raises(ValueError):
        ModelSanityGate(permutation_max_features=-1)
    with pytest.raises(ValueError):
        ModelSanityGate(permutation_tolerance=-1e-6)


def test_input_contract_validation():
    gate = ModelSanityGate(check_diversity=False, check_unique=False)
    with pytest.raises(ValueError):
        gate.check(y=np.arange(10), y_pred_oof=np.arange(11))
    with pytest.raises(ValueError):
        gate.check(y=np.arange(10), y_pred_oof=np.arange(10), X=np.zeros((11, 2)))
    with pytest.raises(ValueError):
        gate.check(y=None, y_pred_oof=np.arange(10))
    with pytest.raises(ValueError):
        gate.check(y=np.arange(10).reshape(2, 5), y_pred_oof=np.arange(10))
    # A list-X has no .shape → ValueError, not an AttributeError.
    with pytest.raises(ValueError):
        gate.check(y=np.arange(10), y_pred_oof=np.arange(10), X=[[0.0] * 3] * 10)


def test_module_is_standalone_clean_import():
    # The module is independent from the rest of the engine: import it directly.
    import importlib

    mod = importlib.import_module(
        "configurable_automl_engine.training_engine.sanity_gate"
    )
    assert hasattr(mod, "ModelSanityGate")
    assert hasattr(mod, "SanityCheckResult")


# ──────────────────────────────────────────────────────────────────────────────
#  Additional edge-case coverage
# ──────────────────────────────────────────────────────────────────────────────


def test_input_2d_column_vector_is_accepted():
    gate = ModelSanityGate(
        check_diversity=False,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )
    y = np.arange(10).reshape(-1, 1)
    result = gate.check(y=y, y_pred_oof=np.arange(10).reshape(-1, 1))
    assert result.is_valid


def test_input_3d_vector_rejected():
    gate = ModelSanityGate(check_diversity=False, check_unique=False)
    with pytest.raises(ValueError):
        gate.check(y=np.zeros((2, 2, 2)), y_pred_oof=np.zeros((2, 2, 2)))


def test_gap_no_finite_pairs_for_rmse():
    y = np.arange(50, dtype=float)
    result = _gap_only_gate(1.5).check(
        y=y, y_pred_oof=y, y_pred_full=np.full(50, np.nan)
    )
    assert not result.is_valid
    assert any("no finite pairs to compute RMSE" in r for r in result.reasons)


def test_unique_constant_target_uses_fallback_precision():
    # Constant y: std(y)=0 → fallback rounding precision (6 decimals). The
    # threshold is capped by nunique(y)=1, so a perfect model (a constant
    # prediction matching the constant target) passes — the cap of issue
    # #125 guarantees that a model predicting y exactly is never
    # disqualified by circuit B.
    y = np.full(60, 7.0)
    gate = ModelSanityGate(
        check_diversity=False, check_dead_features=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=np.full(60, 7.0))
    assert result.is_valid
    assert result.reasons == []
    assert result.unique_ratio == pytest.approx(1 / 60)

    # The fallback granularity (1e-6) is in effect: predictions differing by
    # less than the granularity collapse to a single rounded value, and the
    # ratio is computed on the rounded values regardless of the cap.
    jitter = 7.0 + np.linspace(0.0, 1e-9, 60)  # below the granularity
    result_jitter = gate.check(y=y, y_pred_oof=jitter)
    assert result_jitter.is_valid
    assert result_jitter.unique_ratio == pytest.approx(1 / 60)


def test_dead_features_empty_coef_is_reason():
    y = np.arange(40, dtype=float)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=_CoefModel(np.array([])))
    assert not result.is_valid
    assert any("empty coef_" in r for r in result.reasons)


def test_dead_features_1d_x_is_reason():
    rng = np.random.RandomState(0)
    y = rng.randn(50)
    rf = RandomForestRegressor(max_depth=3, random_state=0)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=rf, X=rng.randn(50))
    assert not result.is_valid
    assert any("2-D matrix" in r for r in result.reasons)


def test_dead_features_nan_predictions_on_subsample_is_reason():
    class _NanPredictModel:
        def predict(self, X: Any) -> np.ndarray:
            return np.full(len(X), np.nan)

    rng = np.random.RandomState(0)
    X = rng.randn(50, 4)
    y = rng.randn(50)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=_NanPredictModel(), X=X)
    assert not result.is_valid
    assert any("no finite" in r for r in result.reasons)


def test_dead_features_predict_fails_only_on_permuted_x():
    class _FailsAfterFirstCall:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, X: Any) -> np.ndarray:
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("cannot predict permuted X")
            return np.full(len(X), 1.0)

    rng = np.random.RandomState(0)
    X = rng.randn(50, 4)
    y = rng.randn(50)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=_FailsAfterFirstCall(), X=X)
    assert not result.is_valid
    assert any("prediction failed on permuted X" in r for r in result.reasons)


def test_dead_features_predict_wrong_output_length_is_reason():
    class _WrongLengthModel:
        def predict(self, X: Any) -> np.ndarray:
            return np.ones(len(X) + 3)  # wrong output length

    rng = np.random.RandomState(0)
    X = rng.randn(50, 4)
    y = rng.randn(50)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=_WrongLengthModel(), X=X)
    assert not result.is_valid
    assert any("values for" in r and "Circuit C" in r for r in result.reasons)


def test_dead_features_permuted_column_nan_pairs_are_skipped():
    class _NanOnPermutedX:
        """Baseline prediction is fine; on permuted X it returns NaN (not a failure)."""

        def __init__(self) -> None:
            self.calls = 0

        def predict(self, X: Any) -> np.ndarray:
            self.calls += 1
            if self.calls > 1:
                return np.full(len(X), np.nan)
            return np.full(len(X), 1.0)

    rng = np.random.RandomState(0)
    X = rng.randn(50, 4)
    y = rng.randn(50)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=_NanOnPermutedX(), X=X)
    # All columns are skipped (no valid pairs) — no failure, dead_count=0.
    assert result.is_valid
    assert result.dead_features_count == 0


def test_dead_features_row_subsampling_failure_is_reason():
    class _UnsliceableX:
        """Has .shape but cannot be sliced by rows."""

        @property
        def shape(self) -> tuple[int, int]:
            return (50, 4)

        def __getitem__(self, idx: Any) -> Any:
            raise RuntimeError("cannot slice")

    rng = np.random.RandomState(0)
    y = rng.randn(50)
    rf = RandomForestRegressor(max_depth=3, random_state=0)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=rf, X=_UnsliceableX())
    assert not result.is_valid
    assert any("row subsampling" in r and "Circuit C" in r for r in result.reasons)


def test_dead_features_unexpected_permutation_error_is_reason():
    class _ListReturningX:
        """__getitem__ returns a list without .ndim → failure inside the implementation."""

        @property
        def shape(self) -> tuple[int, int]:
            return (50, 4)

        def __getitem__(self, idx: Any) -> list[list[float]]:
            return [[0.0] * 4 for _ in range(len(idx))]

    class _DummyModel:
        def predict(self, X: Any) -> np.ndarray:
            return np.ones(len(X))

    rng = np.random.RandomState(0)
    y = rng.randn(50)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(y=y, y_pred_oof=y, model=_DummyModel(), X=_ListReturningX())
    # A failure inside the permutation path becomes a reason, not a crash.
    assert not result.is_valid
    assert any("permutation check failed" in r for r in result.reasons)


def test_dead_features_column_permutation_failure_is_reason():
    class _BrokenSparseLike:
        """toarray() raises — simulates a broken sparse object."""

        @property
        def shape(self) -> tuple[int, int]:
            return (50, 4)

        @property
        def ndim(self) -> int:
            return 2

        @property
        def toarray(self) -> Any:
            raise RuntimeError("toarray boom")

        def __getitem__(self, idx: Any) -> "_BrokenSparseLike":
            return self

    class _DummyModel:
        def predict(self, X: Any) -> np.ndarray:
            return np.ones(X.shape[0])

    rng = np.random.RandomState(0)
    y = rng.randn(50)
    gate = ModelSanityGate(
        check_diversity=False, check_unique=False, check_generalization_gap=False
    )
    result = gate.check(
        y=y, y_pred_oof=y, model=_DummyModel(), X=_BrokenSparseLike()
    )
    assert not result.is_valid
    assert any("column permutation failed" in r for r in result.reasons)


def test_core_estimator_unwraps_nested_pipeline_objects():
    from configurable_automl_engine.training_engine.sanity_gate import _core_estimator

    class Step:
        def __init__(self, name: str, obj: Any) -> None:
            self.name = name
            self.obj = obj

    class FakePipeline:
        def __init__(self, steps: list[Step]) -> None:
            self._steps = {s.name: s.obj for s in steps}

        @property
        def named_steps(self) -> dict[str, Any]:
            return self._steps

    inner = FakePipeline([Step("model", _CoefModel([1.0, 0.0]))])
    outer = FakePipeline([Step("scale", object()), Step("inner", inner)])
    est = _core_estimator(outer)
    assert np.array_equal(est.coef_, [1.0, 0.0])

    # Pipelines without steps and cyclic objects do not loop forever.
    empty = FakePipeline([])
    assert _core_estimator(empty) is empty
    cyclic = FakePipeline([Step("self", None)])
    cyclic._steps["self"] = cyclic
    assert _core_estimator(cyclic) is cyclic

    class WeirdPipeline:
        @property
        def named_steps(self) -> Any:
            return object()  # no .values()

    weird = WeirdPipeline()
    assert _core_estimator(weird) is weird


def test_gap_y_pred_full_length_mismatch_rejected():
    y = np.arange(10, dtype=float)
    with pytest.raises(ValueError):
        _gap_only_gate(1.5).check(y=y, y_pred_oof=y, y_pred_full=np.arange(11))


def test_constructor_zero_coef_tolerance_validation():
    with pytest.raises(ValueError):
        ModelSanityGate(zero_coef_tolerance=-1e-6)

# ──────────────────────────────────────────────────────────────────────────────
#  Soft-warning log levels
# ──────────────────────────────────────────────────────────────────────────────


def test_soft_warnings_logged_at_debug_not_warning(caplog):
    """soft_warnings are logged at DEBUG/INFO, never at WARNING.

    Dead features in linear models (a 'signal') are logged at DEBUG/INFO:
    they do not disqualify on their own and must not be promoted to WARNING.
    """
    import logging

    rng = np.random.RandomState(11)
    X = rng.randn(200, 30)
    y = 2.0 * X[:, 0] + 1.0 * X[:, 1] + rng.randn(200)
    lasso = Lasso(alpha=0.06, random_state=0).fit(X, y)
    gate = ModelSanityGate()

    with caplog.at_level(
        logging.DEBUG, logger="configurable_automl_engine.training_engine.sanity_gate"
    ):
        result = gate.check(
            y=y,
            y_pred_oof=lasso.predict(X),
            y_pred_full=lasso.predict(X),
            X=X,
            model=lasso,
        )

    assert result.is_valid
    assert any("soft warning" in r.message for r in caplog.records)
    # Details of each soft_warning are logged at DEBUG.
    assert any("Sanity gate soft warning" in r.message for r in caplog.records)
    # No WARNING level or above is emitted.
    assert all(r.levelno < logging.WARNING for r in caplog.records)


def test_soft_warning_log_reports_final_status_and_candidate_name(caplog):
    """The 'N soft warning(s)' line reports the final status, not 'valid' by default.

    A model with BOTH disqualification reasons and soft warnings must be
    logged as 'disqualified', and the candidate name must prefix every audit
    log line so the warning is attributed to the right candidate.
    """
    rng = np.random.RandomState(5)
    n = 100
    y = rng.randn(n)
    # Diversity = 0 → Circuit A fails (a reason); 100% zero coefficients →
    # a Circuit C signal (a soft warning). Both channels are populated.
    model = _CoefModel(np.zeros(10))
    gate = ModelSanityGate(check_unique=False, check_generalization_gap=False)
    with caplog.at_level(
        logging.DEBUG, logger="configurable_automl_engine.training_engine.sanity_gate"
    ):
        result = gate.check(
            y=y,
            y_pred_oof=np.full(n, y.mean()),
            y_pred_full=np.full(n, y.mean()),
            X=rng.randn(n, 10),
            model=model,
            candidate_name="lasso_alpha_0.1",
        )
    assert not result.is_valid
    assert result.reasons
    assert result.soft_warnings
    # The final status is 'disqualified', not the unconditional 'valid'.
    assert any(
        "Sanity gate: disqualified with 1 soft warning(s)" in r.message
        for r in caplog.records
    )
    assert not any("Sanity gate: valid with" in r.message for r in caplog.records)


def test_soft_warning_log_uses_valid_status_for_passing_candidate(caplog):
    """A passing candidate with only soft warnings is logged as 'valid'."""
    rng = np.random.RandomState(11)
    X = rng.randn(200, 30)
    y = 2.0 * X[:, 0] + 1.0 * X[:, 1] + rng.randn(200)
    lasso = Lasso(alpha=0.06, random_state=0).fit(X, y)
    gate = ModelSanityGate()
    with caplog.at_level(
        logging.DEBUG, logger="configurable_automl_engine.training_engine.sanity_gate"
    ):
        result = gate.check(
            y=y,
            y_pred_oof=lasso.predict(X),
            y_pred_full=lasso.predict(X),
            X=X,
            model=lasso,
            candidate_name="lasso",
        )
    assert result.is_valid
    assert any(
        "Sanity gate: valid with" in r.message and "lasso: " in r.message
        for r in caplog.records
    )


def test_audit_log_lines_prefixed_with_candidate_name(caplog):
    """Every audit log line carries the candidate name prefix.

    Without the name it is unclear which candidate a soft warning about dead
    features belongs to; with it, each line is attributed.
    """
    rng = np.random.RandomState(5)
    n = 100
    y = rng.randn(n)
    model = _CoefModel(np.zeros(10))
    gate = ModelSanityGate(check_unique=False, check_generalization_gap=False)
    with caplog.at_level(
        logging.DEBUG, logger="configurable_automl_engine.training_engine.sanity_gate"
    ):
        gate.check(
            y=y,
            y_pred_oof=np.full(n, y.mean()),
            y_pred_full=np.full(n, y.mean()),
            X=rng.randn(n, 10),
            model=model,
            candidate_name="lasso_alpha_0.1",
        )
    # The reasons line, the status line, and the soft-warning details are all
    # attributed to the candidate.
    audit_records = [
        r
        for r in caplog.records
        if "lasso_alpha_0.1:" in r.message
    ]
    assert audit_records
    assert any("reason(s)" in r.message for r in audit_records)
    assert any("soft warning" in r.message for r in audit_records)
    # Without a candidate name no prefix is emitted (backward compatible).
    caplog.clear()
    with caplog.at_level(
        logging.DEBUG, logger="configurable_automl_engine.training_engine.sanity_gate"
    ):
        gate.check(
            y=y,
            y_pred_oof=np.full(n, y.mean()),
            y_pred_full=np.full(n, y.mean()),
            X=rng.randn(n, 10),
            model=model,
        )
    assert not any("None:" in r.message for r in caplog.records)


def test_target_leakage_warning_prefixed_with_candidate_name(caplog):
    """The deliberate WARNING (target leakage) is also attributed to the candidate."""
    y = np.arange(100, dtype=float)
    with caplog.at_level("WARNING", logger="configurable_automl_engine"):
        _gap_only_gate(1.5, mode="warn").check(
            y=y, y_pred_oof=y, y_pred_full=y, candidate_name="deep_tree"
        )
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "deep_tree: Sanity gate soft warning" in r.message
        and "target leakage" in r.message
        for r in warning_records
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Golden scenarios on synthetic data N=10–30
# ──────────────────────────────────────────────────────────────────────────────


def _small_dataset(n: int = 25, seed: int = 7, noise: float = 0.5, p: int = 4):
    """Small synthetic dataset: X0/X1 are informative, the rest is noise."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * noise
    return X, y


def test_golden_small_n_honest_gbr_passes_all_circuits():
    """Positive: an honest model (gradient boosting) on N=30.

    Small datasets (N=10–30) must not falsely disqualify honest models:
    diversity > 0.15, unique ratio sufficient, gap ≤ 1.5.
    """
    X, y = _small_dataset(n=30, seed=7)
    oof, full, model = _honest_oof_full(
        lambda: GradientBoostingRegressor(
            max_depth=1, n_estimators=20, learning_rate=0.05, random_state=0
        ),
        X,
        y,
        folds=5,
    )
    result = ModelSanityGate().check(
        y=y, y_pred_oof=oof, y_pred_full=full, X=X, model=model
    )
    assert result.is_valid
    assert result.reasons == []
    assert result.diversity_ratio > 0.15
    assert result.unique_ratio >= 0.05
    assert result.generalization_gap <= 1.5


@pytest.mark.parametrize(
    "name, factory, expected_circuit",
    [
        (
            "elasticnet_zero",
            lambda: ElasticNet(alpha=500.0, random_state=0),
            "Circuit A",
        ),
        (
            "svr_sigmoid",
            lambda: SVR(kernel="sigmoid", gamma=0.001, coef0=0.0),
            "Circuit A",
        ),
        (
            "deep_tree",
            lambda: DecisionTreeRegressor(max_depth=20, random_state=0),
            "Circuit D",
        ),
        (
            "knn_1",
            lambda: KNeighborsRegressor(n_neighbors=1),
            "Circuit D",
        ),
    ],
)
def test_golden_small_n_degenerate_models_fail(
    name: str, factory: Callable[[], Any], expected_circuit: str
):
    """Negative: degenerate models on N=25 are disqualified.

    (a) ElasticNet with zeroed coefficients ('shelf');
    (b) SVR-sigmoid stuck on the tanh plateau;
    (c) a depth-20 tree (RMSE_train≈0);
    (d) 1-NN.

    For the Circuit D cases (deep_tree, knn_1) the disqualification is
    achieved via an explicit ``generalization_gap_mode="reject"``: the
    default warn mode moves RMSE_full≈0 into soft_warnings and does not
    disqualify.
    """
    X, y = _small_dataset(n=25, seed=7)
    oof, full, model = _honest_oof_full(factory, X, y, folds=5)
    result = ModelSanityGate(generalization_gap_mode="reject").check(
        y=y, y_pred_oof=oof, y_pred_full=full, X=X, model=model
    )
    assert not result.is_valid
    assert any(expected_circuit in r for r in result.reasons)
    # A degenerate model must have at least one reason.
    assert result.reasons


def test_v2_wide_data_lasso_high_zero_ratio_soft_warning_only():
    """Wide data, Lasso with 60–90% zero coefficients.

    A high zero-coefficient share is a soft_warning, but NOT a
    disqualification: on wide noisy data Lasso legitimately zeroes features,
    and diversity stays high with a normal signal (the C+A combination does
    not trigger).
    """
    rng = np.random.RandomState(21)
    n, p = 100, 80
    X = rng.randn(n, p)
    y = 2.0 * X[:, 0] + 1.0 * X[:, 1] + rng.randn(n)
    lasso = Lasso(alpha=0.1, random_state=0).fit(X, y)
    gate = ModelSanityGate()
    tol = gate.zero_coef_tolerance
    zero_ratio = float((np.abs(lasso.coef_) <= tol).mean())
    assert 0.6 <= zero_ratio <= 0.95  # precondition: 60–90% zeros

    result = gate.check(
        y=y, y_pred_oof=lasso.predict(X), y_pred_full=lasso.predict(X), X=X, model=lasso
    )
    assert result.is_valid  # not a disqualification
    assert result.reasons == []
    assert any("zero-coefficient ratio" in w for w in result.soft_warnings)
    assert result.diversity_ratio > 0.15  # normal diversity


def test_v2_noisy_honest_model_not_disqualified_small_n():
    """Noisy data, honest model — the gate does NOT disqualify it.

    R²≈0.3 (Var(signal)=3 with 3 informative features, noise chosen so that
    R² = Var(signal)/(Var(signal)+Var(noise)) = 0.3) — the soft diversity
    threshold (0.15) lets the honest model through even on a small sample
    of N=30.
    """
    rng = np.random.RandomState(3)
    n = 30
    X = rng.randn(n, 3)
    signal = X.sum(axis=1)
    noise_var = 3.0 * (1.0 / 0.3 - 1.0)
    y = signal + rng.randn(n) * np.sqrt(noise_var)

    class _LinearHonest:
        """A linear honest model: recovers the signal with noise."""

        def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
            self._X = X
            self._y = y

        def fit(self, X: np.ndarray, y: np.ndarray) -> "_LinearHonest":
            # OLS (ordinary least squares) — no sklearn dependency.
            Xd = np.column_stack([np.ones(len(X)), X])
            coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
            self._coef = coef
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            Xd = np.column_stack([np.ones(len(X)), X])
            return Xd @ self._coef

    model = _LinearHonest(X, y).fit(X, y)
    pred = model.predict(X)
    # Under the test's OOF semantics, compute the honest OOF via KFold.
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        m = _LinearHonest(X, y).fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    result = ModelSanityGate(check_generalization_gap=False).check(
        y=y, y_pred_oof=oof, y_pred_full=pred, X=X, model=model
    )
    assert result.is_valid  # NOT disqualified (soft diversity threshold)
    assert result.reasons == []
    # Diversity ≈ R² of the noisy honest model: well above the strict 0.15
    # threshold but noticeably below the 'ideal' value (noise prevents R² ≈ 1).
    assert result.diversity_ratio > 0.15
    assert result.diversity_ratio < 0.9
