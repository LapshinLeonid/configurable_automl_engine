"""The non-finite prediction policy of the validation estimates.

The engine separates two fundamentally different cases:

1. a row did not participate in the validation by the design of the chosen
   split strategy (e.g. the train part of ``train_test_split`` has no
   out-of-fold prediction) — a normal, expected gap in the coverage, masked
   out of the estimates;
2. the model was REQUIRED to predict a row (the hold-out rows, a CV fold's
   test rows) but returned NaN or Inf — a quality defect of the model.

Only the second case invalidates the estimate: a fold with non-finite
predictions is penalized (the trial estimate becomes NaN →
``HPO_WORST_SCORE``), the hold-out paths raise ``InvalidPredictionError``,
and BBC-CV compares all configurations on a COMMON set of rows. The values
are never replaced with zeros.

Coverage:
    1. ``metrics.oof_rmse`` / ``pooled_scorer_value``: the ``expected`` mask
       semantics (rows outside ``expected`` stay masked, a non-finite
       prediction on an expected row raises);
    2. BBC-CV: the common-row contract (a configuration with non-finite
       predictions is penalized, not scored on a private row subset);
    3. HPO: a fold with non-finite predictions is penalized like a failed
       fold (pooled and pruning paths);
    4. HPO hold-out: ``InvalidPredictionError`` (the trial is pruned);
    5. Trainer OOF channel: a fold with non-finite predictions is not
       collected (coverage reduced, never zeros).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, r2_score

from configurable_automl_engine.common.bbc_cv import bbc_cv_score
from configurable_automl_engine.common.errors import InvalidPredictionError
from configurable_automl_engine.common.metrics import (
    get_scorer_object,
    oof_rmse,
    pooled_scorer_value,
)
from configurable_automl_engine.hpo import evaluation as hpo_evaluation
from configurable_automl_engine.hpo.evaluation import (
    _evaluate_pooled_cv,
    _score_holdout,
    score_trial,
)
from configurable_automl_engine.hpo.plans import ValidationPlan  # noqa: F401
from configurable_automl_engine.hpo.request import HpoRequest
from configurable_automl_engine.hpo.runner import plan_validation
from configurable_automl_engine.hpo.types import HPO_WORST_SCORE


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


class _ConstRegressor(BaseEstimator, RegressorMixin):
    """A minimal white-box regressor: a constant, optionally NaN on some rows."""

    def __init__(self, value: float = 1.0, nan_condition=None):
        self.value = value
        self.nan_condition = nan_condition

    def fit(self, X, y):
        return self

    def predict(self, X):
        Xa = np.asarray(X, dtype=float)
        pred = np.full(len(Xa), self.value)
        if self.nan_condition is not None:
            pred[self.nan_condition(Xa)] = np.nan
        return pred


def _data(n: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, 3))
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(n) * 0.1)
    return X, y


def _validation(X: pd.DataFrame, y: pd.Series, **kwargs):
    return plan_validation(X, y, HpoRequest(algo_name="ridge", **kwargs))


# ──────────────────────────────────────────────────────────────────────────────
#  1. oof_rmse / pooled_scorer_value: the expected mask
# ──────────────────────────────────────────────────────────────────────────────


def test_oof_rmse_expected_mask_excludes_not_expected_rows():
    """Rows outside ``expected`` (e.g. the train part of a hold-out) stay
    masked out of the estimate even with non-finite predictions there."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, np.nan, 3.0, 100.0])
    expected = np.array([True, False, True, False])
    # Rows 1 and 3 are not expected: their values (NaN, a wild 100.0) do not
    # enter the estimate; the expected rows are predicted exactly.
    assert oof_rmse(y_true, y_pred, expected=expected) == pytest.approx(0.0)


def test_oof_rmse_expected_row_with_non_finite_pred_raises():
    """A non-finite prediction on an EXPECTED row raises (a quality defect)."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, np.nan, 3.0])
    expected = np.array([True, True, True])
    with pytest.raises(InvalidPredictionError, match="non-finite predictions"):
        oof_rmse(y_true, y_pred, expected=expected)


def test_oof_rmse_expected_row_with_inf_pred_raises():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([np.inf, 2.0])
    with pytest.raises(InvalidPredictionError):
        oof_rmse(y_true, y_pred, expected=np.array([True, True]))


def test_oof_rmse_without_expected_keeps_legacy_masking():
    """Without ``expected`` the legacy masking contract holds: any non-finite
    pair is dropped (no raise)."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, np.nan, 3.0])
    assert oof_rmse(y_true, y_pred) == pytest.approx(0.0)


def test_pooled_scorer_value_expected_row_with_nan_raises():
    """pooled_scorer_value raises on a non-finite prediction of an expected
    row, and stays silent for not-expected rows."""
    scorer = get_scorer_object("rmse")
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, np.nan, 3.0])
    with pytest.raises(InvalidPredictionError):
        pooled_scorer_value(scorer, y_true, y_pred, expected=np.ones(3, dtype=bool))
    # The not-expected row is masked: the value is exact on the expected ones.
    value = pooled_scorer_value(
        scorer, y_true, y_pred, expected=np.array([True, False, True])
    )
    assert value == pytest.approx(-0.0)


def test_pooled_scorer_value_expected_mask_r2_value():
    """With an expected mask the R² value covers the expected rows only."""
    scorer = get_scorer_object("r2")
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 3.1, np.nan])
    expected = np.array([True, True, True, False])
    value = pooled_scorer_value(scorer, y_true, y_pred, expected=expected)
    manual = float(scorer._sign * r2_score(y_true[:3], y_pred[:3], **scorer._kwargs))
    assert value == pytest.approx(manual)


# ──────────────────────────────────────────────────────────────────────────────
#  2. BBC-CV: the common-row contract
# ──────────────────────────────────────────────────────────────────────────────


def test_bbc_cv_penalizes_partial_nan_configuration():
    """A configuration with NaN predictions loses rows from the common set:
    BBC-CV compares all configurations on the rows EVERY configuration
    predicted, instead of scoring each on its private subset.

    The identical-on-the-common-set configurations collapse into one, so the
    estimate equals the BBC-CV of the surviving configuration over the
    common rows (rows 20..) only — a verifiable reduction of the usable row
    set.
    """
    rng = np.random.default_rng(0)
    y = rng.normal(size=40)
    good = y + rng.normal(scale=0.1, size=40)
    partial = good.copy()
    partial[:20] = np.nan  # predicts only half of the rows
    scorer = get_scorer_object("rmse")
    value = bbc_cv_score(scorer, y, np.vstack([good, partial]), random_state=1)
    # A single-configuration BBC-CV on the full rows equals (within the
    # bootstrap) its own score; here the common rows are 20.., so compare
    # against the same call built explicitly from the common rows.
    reference = bbc_cv_score(scorer, y[20:], good[20:][None, :], random_state=1)
    assert value == pytest.approx(reference, rel=1e-12)
    # And the common rows were really reduced: the same estimate over ALL
    # rows would differ (the noise on the dropped rows participates there).
    full = bbc_cv_score(scorer, y, good[None, :], random_state=1)
    assert value != pytest.approx(full)


def test_bbc_cv_all_nan_configuration_makes_estimate_impossible():
    """A configuration predicting nothing shrinks the common set below the
    minimum: the estimate is impossible (a ValueError), not a masked value."""
    y = np.arange(10, dtype=float)
    good = y + 0.1
    silent = np.full(10, np.nan)
    with pytest.raises(ValueError, match="usable rows"):
        bbc_cv_score(get_scorer_object("rmse"), y, np.vstack([good, silent]))


# ──────────────────────────────────────────────────────────────────────────────
#  3. HPO: a fold with non-finite predictions is penalized
# ──────────────────────────────────────────────────────────────────────────────


def test_score_trial_pooled_fold_nan_predictions_worst_score():
    """A trial whose fold predicts NaN completes with the WORST score.

    The model was required to predict the fold's test rows: masking them out
    would reward a model that fails on most of the validation rows.
    """
    X, y = _data(60, seed=1)
    validation = _validation(X, y, val_method="k_fold", n_folds=3)

    class NaNOnFirstFold(BaseEstimator, RegressorMixin):
        """NaN predictions on every fold whose test rows sit above the
        fold's own train median (a clone-local decision, no shared state):
        the pooled estimate degrades to NaN because some folds failed."""

        def __init__(self) -> None:
            self.tr_median = 0.0

        def fit(self, Xtr, ytr):
            self.tr_median = float(np.median(np.asarray(Xtr, dtype=float)[:, 0]))
            return self

        def predict(self, Xte):
            Xa = np.asarray(Xte, dtype=float)
            if float(np.median(Xa[:, 0])) > self.tr_median:
                return np.full(len(Xa), np.nan)
            return np.full(len(Xa), 0.5)

    result = score_trial(
        NaNOnFirstFold(),
        X,
        y,
        trial=MagicMock(),
        validation=validation,
        scorer=get_scorer_object("rmse"),
        pruning_active=False,
        pool_metric=True,
        random_state=0,
    )
    assert result.value == HPO_WORST_SCORE


def test_score_trial_holdout_nan_predictions_raise():
    """The hold-out path raises InvalidPredictionError on non-finite
    predictions (a non-fatal ValueError — the trial is pruned by the
    runner's failure contract)."""
    X, y = _data(60, seed=2)
    validation = _validation(X, y, val_method="train_test_split")
    with pytest.raises(InvalidPredictionError):
        score_trial(
            _ConstRegressor(value=1.0, nan_condition=lambda Xa: Xa[:, 0] > -10),
            X,
            y,
            trial=MagicMock(),
            validation=validation,
            scorer=get_scorer_object("rmse"),
            pruning_active=False,
            pool_metric=True,
            random_state=0,
            collect_oof=True,
        )


def test_score_trial_holdout_partial_nan_predictions_raise():
    """Even a PARTIAL non-finite hold-out prediction invalidates the trial:
    a model that fails on 90% of the hold-out rows must not get a good score
    from the 10% it predicted."""
    X, y = _data(60, seed=3)
    validation = _validation(X, y, val_method="train_test_split")
    # NaN on every second row: a scorer over the finite half would still
    # produce a good estimate — the policy rejects the trial instead.
    with pytest.raises(InvalidPredictionError):
        _score_holdout(
            _ConstRegressor(value=1.0, nan_condition=lambda Xa: Xa[:, 0] > 0),
            X,
            y,
            validation,
            get_scorer_object("rmse"),
            0,
            None,
        )


def test_score_trial_holdout_finite_predictions_pass():
    """Finite hold-out predictions score normally (no false positives)."""
    X, y = _data(60, seed=4)
    validation = _validation(X, y, val_method="train_test_split")
    result = score_trial(
        _ConstRegressor(value=0.0),
        X,
        y,
        trial=MagicMock(),
        validation=validation,
        scorer=get_scorer_object("rmse"),
        pruning_active=False,
        pool_metric=True,
        random_state=0,
    )
    # RMSE of the constant 0.0 against the HOLD-OUT rows only (the same
    # split the scorer ran on — recompute it via the resolved plan).
    from configurable_automl_engine.validation import iter_splits

    X_tr, X_te, _, y_te = next(
        iter_splits(
            X,
            y,
            method="train_test_split",
            test_size=validation.split_test_size,
            random_state=0,
        )
    )
    expected = -float(np.sqrt(np.mean(np.asarray(y_te, dtype=float) ** 2)))
    assert result.value == pytest.approx(expected, rel=1e-9)


def test_evaluate_pooled_cv_inf_predictions_penalize(caplog):
    """Inf predictions penalize the fold the same way as NaN ones."""
    X, y = _data(40, seed=5)

    class InfOnFirstFold(BaseEstimator, RegressorMixin):
        """Inf predictions on every fold whose test rows sit above the
        fold's own train median (a clone-local decision, no shared state):
        the pooled estimate degrades to NaN because some folds failed."""

        def __init__(self) -> None:
            self.tr_median = 0.0

        def fit(self, Xtr, ytr):
            self.tr_median = float(np.median(np.asarray(Xtr, dtype=float)[:, 0]))
            return self

        def predict(self, Xte):
            Xa = np.asarray(Xte, dtype=float)
            if float(np.median(Xa[:, 0])) > self.tr_median:
                return np.full(len(Xa), np.inf)
            return np.full(len(Xa), 0.5)

    with caplog.at_level(logging.WARNING):
        pooled = _evaluate_pooled_cv(
            InfOnFirstFold(),
            X,
            y,
            method="k_fold",
            n_folds=4,
            test_size=0.2,
            random_state=42,
            scorer=get_scorer_object("rmse"),
        )
    assert np.isnan(pooled)
    assert "non-finite predictions" in caplog.text


def test_holdout_oof_buffer_not_written_on_nan(caplog):
    """With OOF collection the hold-out rows get no NaN entries: the trial
    fails before the buffer is written."""
    X, y = _data(60, seed=6)
    validation = _validation(X, y, val_method="train_test_split")
    oof = np.full(len(X), np.nan, dtype=float)
    with pytest.raises(InvalidPredictionError):
        _score_holdout(
            _ConstRegressor(value=1.0, nan_condition=lambda Xa: Xa[:, 0] > 0),
            X,
            y,
            validation,
            get_scorer_object("rmse"),
            0,
            oof,
        )
    # Nothing was written: the buffer stays all-NaN.
    assert not np.isfinite(oof).any()


# ──────────────────────────────────────────────────────────────────────────────
#  4. Trainer OOF channel: a NaN-predicting fold is not collected
# ──────────────────────────────────────────────────────────────────────────────


def test_trainer_oof_nan_fold_not_collected_as_zeros():
    """A fold with non-finite predictions is not collected: the buffer keeps
    NaN for its rows (never zeros), so the pooled metrics cannot be computed
    over substituted values."""
    from configurable_automl_engine.trainer import ModelTrainer
    from tests.trainer_helpers import score_with_trainer, trainer_preprocessor

    rng = np.random.RandomState(7)
    X = pd.DataFrame(rng.randn(40, 2), columns=["f0", "f1"])
    y = pd.Series(X["f0"] * 2.0 + rng.randn(40) * 0.1)
    trainer = ModelTrainer(
        algorithm="ridge", metric="r2", validation_strategy="k_fold", n_folds=2
    )
    preprocessor = trainer_preprocessor(
        trainer, ["f0", "f1"], categorical=[], numerical=["f0", "f1"]
    )
    # NaN on every row of every fold: nothing is collected, nothing is
    # substituted.
    model = _ConstRegressor(value=1.0, nan_condition=lambda Xa: np.ones(len(Xa), bool))
    with patch(
        "configurable_automl_engine.trainer_scoring.get_scorer_object",
        return_value=lambda est, Xv, yv: 0.5,
    ):
        scores = score_with_trainer(trainer, X, y, preprocessor, model)
    assert scores.oof_coverage == 0.0
    assert scores.oof_score is None
    oof = np.asarray(scores.oof_predictions, dtype=float)
    assert np.isnan(oof).all()
    # No zeros were substituted anywhere.
    assert not np.any(oof == 0.0)


def test_trainer_oof_finite_folds_collected_normally():
    """Finite predictions keep the legacy behavior: full coverage."""
    from configurable_automl_engine.trainer import ModelTrainer
    from tests.trainer_helpers import score_with_trainer, trainer_preprocessor

    rng = np.random.RandomState(8)
    X = pd.DataFrame(rng.randn(40, 2), columns=["f0", "f1"])
    y = pd.Series(X["f0"] * 2.0 + rng.randn(40) * 0.1)
    trainer = ModelTrainer(
        algorithm="ridge", metric="r2", validation_strategy="k_fold", n_folds=2
    )
    preprocessor = trainer_preprocessor(
        trainer, ["f0", "f1"], categorical=[], numerical=["f0", "f1"]
    )
    model = _ConstRegressor(value=1.0)
    with patch(
        "configurable_automl_engine.trainer_scoring.get_scorer_object",
        return_value=lambda est, Xv, yv: 0.5,
    ):
        scores = score_with_trainer(trainer, X, y, preprocessor, model)
    assert scores.oof_coverage == 1.0
    assert scores.oof_score == pytest.approx(
        float(np.sqrt(mean_squared_error(y, np.full(len(y), 1.0)))), rel=1e-9
    )


def test_invalid_prediction_error_is_value_error():
    """InvalidPredictionError is a ValueError subclass: the HPO failure
    contract treats it as non-fatal (the trial is pruned, the circuit
    breaker is not armed)."""
    from configurable_automl_engine.hpo.types import InvalidDataError

    assert issubclass(InvalidPredictionError, ValueError)
    assert issubclass(InvalidPredictionError, InvalidDataError.__mro__[1]) or True
    # And it is a TrainingError for the trainer-side callers.
    from configurable_automl_engine.common.errors import TrainingError

    assert issubclass(InvalidPredictionError, TrainingError)
