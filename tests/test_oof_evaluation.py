"""Tests for the OOF candidate-evaluation channel.

Coverage:
    1. ``iter_splits(include_indices=True)``: the positional indices of
       validation rows correspond to the X_val/y_val subsets (k_fold and
       train_test_split, including with random_state=None).
    2. ``metrics.oof_rmse``: the metric over the whole vector at once;
       NaN/None/inf pairs are masked; an error is raised when no valid
       pairs exist.
    3. Positive: RMSE_oof ≈ the fold average over equal folds; an exact
       pooled recomputation for unequal folds; each row is predicted by a
       model that did not see it (matches a manual per-fold recomputation —
       no leakage); 100% coverage for k_fold/loo; alignment by
       pandas index.
    4. Negative: with leakage (a model trained on a row), the OOF vector
       differs from the honest one — the test detects it (in-sample RMSE is
       noticeably lower than RMSE_oof).
    5. Boundary: N smaller than the fold count (fallback to train_test_split);
       train_test_split covers only the validation part; unequal fold sizes;
       NaN in predictions; a very small N; a duplicated index.
    6. Backward compatibility: OOF is a separate channel and the ``val_score``
       of HPO phases does not change; serialization keeps the OOF attributes;
       reset between fit() calls.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error

from configurable_automl_engine.models import create_model
from configurable_automl_engine.trainer import ModelTrainer
from configurable_automl_engine.training_engine.metrics import oof_rmse
from configurable_automl_engine.validation import iter_splits


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _noisy_regression(
    n: int = 60, p: int = 6, seed: int = 7, noise: float = 3.0
) -> tuple[pd.DataFrame, pd.Series]:
    """Noisy dataset: the first feature is informative, the rest is noise.

    On such data the models overfit, so the in-sample RMSE is noticeably
    below the honest OOF-RMSE (needed for the leakage negative test).
    """
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p))
    X.iloc[:, 0] = X.iloc[:, 0] * 3.0
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(n) * noise)
    return X, y


def _manual_oof(
    X: Any,
    y: Any,
    trainer: ModelTrainer,
    *,
    method: str,
    n_folds: int,
    seed: int,
    test_size: float = 0.2,
) -> np.ndarray:
    """Recompute the OOF vector by hand: the same splits and pipeline clones per fold.

    Repeats the logic of ``_score_on_validation_splits`` independently to
    verify the absence of leakage: each row is predicted by a pipeline
    trained without it (the test part of the fold).
    """
    preprocessor = trainer._build_preprocessor(trainer.feature_names or [])
    model = create_model(trainer.algorithm, **trainer.hyperparams)
    oof = np.full(len(X), np.nan, dtype=float)
    for X_tr, X_te, y_tr, y_te, test_idx in iter_splits(
        X,
        y,
        method=method,
        n_folds=n_folds,
        test_size=test_size,
        random_state=seed,
        include_indices=True,
    ):
        fold_pipe = ImbPipeline(
            trainer._assemble_steps(
                clone(preprocessor), clone(model), feature_selection_active=False
            )
        )
        fold_pipe.fit(X_tr, y_tr)
        oof[np.asarray(test_idx)] = np.asarray(fold_pipe.predict(X_te)).reshape(-1)
    return oof


class _ConstRegressor(BaseEstimator):
    """A minimal white-box regressor: a constant plus an optional NaN."""

    def __init__(self, value: float = 1.0, nan_condition=None):
        self.value = value
        self.nan_condition = nan_condition

    def fit(self, X, y):
        return self

    def predict(self, X):
        preds = np.full(len(X), self.value, dtype=float)
        if self.nan_condition is not None:
            mask = self.nan_condition(np.asarray(X))
            preds[mask] = np.nan
        return preds


# ──────────────────────────────────────────────────────────────────────────────
#  1. iter_splits(include_indices=True)
# ──────────────────────────────────────────────────────────────────────────────


def test_iter_splits_include_indices_kfold_match_positions():
    """k_fold: test_idx points to the same rows as X_val (numpy and pandas)."""
    rng = np.random.RandomState(1)
    X_np = rng.randn(30, 3)
    y_np = rng.randn(30)
    for X_tr, X_te, y_tr, y_te, test_idx in iter_splits(
        X_np, y_np, method="k_fold", n_folds=3, random_state=5, include_indices=True
    ):
        idx = np.asarray(test_idx)
        assert np.array_equal(X_te, X_np[idx])
        assert np.array_equal(np.asarray(y_te), y_np[idx])

    X_df = pd.DataFrame(X_np)
    y_s = pd.Series(y_np)
    for X_tr, X_te, y_tr, y_te, test_idx in iter_splits(
        X_df, y_s, method="k_fold", n_folds=3, random_state=5, include_indices=True
    ):
        idx = np.asarray(test_idx)
        assert np.array_equal(X_te.to_numpy(), X_df.to_numpy()[idx])
        assert np.array_equal(y_te.to_numpy(), y_s.to_numpy()[idx])


def test_iter_splits_include_indices_tts_match_positions():
    """train_test_split: test_idx is consistent with X_val, incl. random_state=None.

    Before, the split was built by a second train_test_split call, which
    with random_state=None would produce a different permutation and
    desynchronize the indices.
    """
    rng = np.random.RandomState(2)
    X = rng.randn(40, 3)
    y = rng.randn(40)
    for random_state in (7, None):
        for X_tr, X_te, y_tr, y_te, test_idx in iter_splits(
            X,
            y,
            method="train_test_split",
            test_size=0.25,
            random_state=random_state,
            include_indices=True,
        ):
            idx = np.asarray(test_idx)
            assert np.array_equal(X_te, X[idx])
            assert np.array_equal(np.asarray(y_te), y[idx])


def test_iter_splits_default_contract_unchanged():
    """Without include_indices the contract is unchanged: 4 elements, as before."""
    rng = np.random.RandomState(3)
    X = rng.randn(20, 2)
    y = rng.randn(20)
    for split in iter_splits(X, y, method="k_fold", n_folds=2):
        assert len(split) == 4


# ──────────────────────────────────────────────────────────────────────────────
#  2. metrics.oof_rmse
# ──────────────────────────────────────────────────────────────────────────────


def test_oof_rmse_over_vector():
    """RMSE over the whole vector at once (not averaged per fold)."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 4.0])
    expected = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    assert oof_rmse(y_true, y_pred) == pytest.approx(expected, rel=1e-12)


def test_oof_rmse_masks_non_finite_pairs():
    """NaN/None/inf in y_pred or y_true are dropped, not crash the metric."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    # Valid pairs: 0, 2, 4 — the errors are zero.
    assert oof_rmse(y_true, y_pred) == pytest.approx(0.0)

    y_true_nan = np.array([1.0, np.nan, 3.0])
    y_pred_ok = np.array([1.0, 2.0, 3.0])
    assert oof_rmse(y_true_nan, y_pred_ok) == pytest.approx(0.0)


def test_oof_rmse_no_valid_pairs_raises():
    """No valid pair → ValueError (the OOF estimate is impossible)."""
    with pytest.raises(ValueError, match="No valid"):
        oof_rmse(np.array([1.0, 2.0]), np.array([np.nan, np.nan]))
    with pytest.raises(ValueError, match="No valid"):
        oof_rmse(np.array([1.0, 2.0]), np.array([np.inf, -np.inf]))


# ──────────────────────────────────────────────────────────────────────────────
#  3. Positive: honesty and 100% coverage
# ──────────────────────────────────────────────────────────────────────────────


def test_oof_rmse_approx_mean_fold_rmse_equal_folds():
    """Equal folds: RMSE_oof ≈ the average fold RMSE.

    The exact relation is pooled: oof² = mean(MSE_fold) (equal fold sizes);
    approximately (within ~5%) RMSE_oof ≈ mean(RMSE_fold).
    """
    X, y = _noisy_regression(n=60, p=6, seed=7)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=3,
        random_state=7,
    ).fit(X, y)

    preprocessor = trainer._build_preprocessor(trainer.feature_names or [])
    model = create_model(trainer.algorithm, **trainer.hyperparams)
    fold_rmse: list[float] = []
    fold_mse: list[float] = []
    for X_tr, X_te, y_tr, y_te, _ in iter_splits(
        X, y, method="k_fold", n_folds=3, test_size=0.2, random_state=7, include_indices=True
    ):
        fold_pipe = ImbPipeline(
            trainer._assemble_steps(
                clone(preprocessor), clone(model), feature_selection_active=False
            )
        )
        fold_pipe.fit(X_tr, y_tr)
        preds = fold_pipe.predict(X_te)
        fold_rmse.append(float(np.sqrt(mean_squared_error(y_te, preds))))
        fold_mse.append(float(mean_squared_error(y_te, preds)))

    assert trainer.oof_score_ is not None
    # Exact pooled relation for equal folds.
    assert trainer.oof_score_ == pytest.approx(
        float(np.sqrt(np.mean(fold_mse))), rel=1e-9
    )
    # Approximate equality with the mean fold RMSE (Jensen: mean ≤ pooled).
    assert trainer.oof_score_ == pytest.approx(float(np.mean(fold_rmse)), rel=0.05)


def test_oof_no_leakage_matches_manual_reconstruction():
    """No leakage: each row is predicted by a model that never saw it.

    The manual per-fold recomputation (same splits/clones) yields exactly the
    same OOF vector the trainer collected — these are honest out-of-fold
    predictions by construction, not train predictions.
    """
    X, y = _noisy_regression(n=48, p=5, seed=11)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=4,
        random_state=7,
    ).fit(X, y)

    manual = _manual_oof(X, y, trainer, method="k_fold", n_folds=4, seed=7)
    assert trainer.oof_predictions_ is not None
    np.testing.assert_allclose(
        np.asarray(trainer.oof_predictions_), manual, rtol=1e-9, atol=1e-9
    )
    # Full coverage: no row is left without an honest prediction.
    assert np.isfinite(manual).all()
    assert trainer.oof_coverage_ == 1.0


def test_oof_predictions_aligned_with_pandas_index():
    """A non-standard pandas index is preserved in y_pred_oof."""
    X, y = _noisy_regression(n=40, p=4, seed=13)
    X = X.copy()
    X.index = np.arange(len(X)) * 10 + 5
    y = y.copy()
    y.index = X.index

    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=4,
        random_state=3,
    ).fit(X, y)

    assert isinstance(trainer.oof_predictions_, pd.Series)
    assert list(trainer.oof_predictions_.index) == list(X.index)
    assert trainer.oof_coverage_ == 1.0
    # Alignment: the value for the row labeled idx matches the manual OOF.
    manual = _manual_oof(X, y, trainer, method="k_fold", n_folds=4, seed=3)
    np.testing.assert_allclose(
        trainer.oof_predictions_.to_numpy(), manual, rtol=1e-9, atol=1e-9
    )


def test_oof_numpy_input_positional_alignment():
    """numpy input without an index: y_pred_oof is an np.ndarray, aligned by position."""
    X_np, y_np = _noisy_regression(n=50, p=4, seed=5)
    X_np = X_np.to_numpy()
    y_np = y_np.to_numpy()

    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=4,
        random_state=5,
    ).fit(X_np, y_np)

    assert isinstance(trainer.oof_predictions_, np.ndarray)
    assert trainer.oof_coverage_ == 1.0
    assert trainer.oof_predictions_.shape == (len(X_np),)
    # Positional alignment against the manual OOF on the same numpy arrays.
    manual = _manual_oof(X_np, y_np, trainer, method="k_fold", n_folds=4, seed=5)
    np.testing.assert_allclose(trainer.oof_predictions_, manual, rtol=1e-9, atol=1e-9)


def test_oof_loo_full_coverage():
    """LOO: each row is its own fold; coverage is 100%."""
    X, y = _noisy_regression(n=12, p=3, seed=17)
    trainer = ModelTrainer(
        algorithm="ridge", metric="r2", validation_strategy="loo"
    ).fit(X, y)

    assert trainer.oof_coverage_ == 1.0
    assert trainer.oof_score_ is not None
    assert trainer.oof_score_ >= 0
    assert trainer.oof_predictions_ is not None
    assert np.isfinite(np.asarray(trainer.oof_predictions_)).all()


# ──────────────────────────────────────────────────────────────────────────────
#  4. Negative: leakage detection
# ──────────────────────────────────────────────────────────────────────────────


def test_leaky_in_sample_predictions_detected():
    """Leakage is detected: the in-sample RMSE is noticeably below the honest RMSE_oof.

    The final pipeline is trained on 100% of the rows, so every row is
    'predicted' by a model that saw it. Such a (leaky) OOF vector
    systematically differs from the honest one and yields an optimistically
    low error on noisy data.
    """
    X, y = _noisy_regression(n=80, p=12, seed=0, noise=4.0)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=4,
        random_state=42,
    ).fit(X, y)

    leaky_preds = trainer.predict(X)  # the model saw all rows
    leaky_rmse = float(np.sqrt(mean_squared_error(y, leaky_preds)))

    assert trainer.oof_score_ is not None
    # The honest OOF vector does not match the leaky one (row by row).
    honest = np.asarray(trainer.oof_predictions_)
    assert not np.allclose(honest, leaky_preds, rtol=1e-6, atol=1e-6)
    # The in-sample error is systematically lower than the honest out-of-fold one.
    assert leaky_rmse < trainer.oof_score_
    assert trainer.oof_score_ - leaky_rmse > 0.5


# ──────────────────────────────────────────────────────────────────────────────
#  5. Boundary
# ──────────────────────────────────────────────────────────────────────────────


def test_oof_unequal_fold_sizes_pooled_rmse_exact():
    """Unequal folds: the concatenation stays aligned; pooled-RMSE is exact."""
    rng = np.random.RandomState(3)
    n = 11
    X = pd.DataFrame(rng.randn(n, 3))
    X.iloc[:, 0] *= 2.0
    y = pd.Series(X.iloc[:, 0] * 1.5 + rng.randn(n))

    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=3,
        random_state=3,
    ).fit(X, y)

    preprocessor = trainer._build_preprocessor(trainer.feature_names or [])
    model = create_model(trainer.algorithm, **trainer.hyperparams)
    fold_mse: list[float] = []
    fold_n: list[int] = []
    for X_tr, X_te, y_tr, y_te, _ in iter_splits(
        X, y, method="k_fold", n_folds=3, test_size=0.2, random_state=3, include_indices=True
    ):
        fold_pipe = ImbPipeline(
            trainer._assemble_steps(
                clone(preprocessor), clone(model), feature_selection_active=False
            )
        )
        fold_pipe.fit(X_tr, y_tr)
        preds = fold_pipe.predict(X_te)
        fold_mse.append(float(mean_squared_error(y_te, preds)))
        fold_n.append(len(y_te))

    assert trainer.oof_coverage_ == 1.0
    pooled = float(np.sqrt(np.sum(np.array(fold_n) * np.array(fold_mse)) / n))
    assert trainer.oof_score_ == pytest.approx(pooled, rel=1e-9)


def test_oof_train_test_split_partial_coverage_warning(caplog):
    """train_test_split: OOF covers only the validation part (a limitation).

    The limitation is explicitly documented in the log; RMSE_oof is computed
    over the covered subset and equals the RMSE on the hold-out part.
    """
    X, y = _noisy_regression(n=80, p=5, seed=21)
    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine.trainer"):
        trainer = ModelTrainer(
            algorithm="ridge",
            metric="r2",
            validation_strategy="train_test_split",
            test_size=0.25,
            random_state=9,
        ).fit(X, y)

    assert trainer.oof_coverage_ is not None
    assert 0.0 < trainer.oof_coverage_ < 1.0
    assert trainer.oof_score_ is not None
    assert "covers only the validation part" in caplog.text

    # Accuracy: RMSE_oof = RMSE over the covered (validation) hold-out part.
    manual = _manual_oof(
        X, y, trainer, method="train_test_split", n_folds=5, seed=9, test_size=0.25
    )
    valid = np.isfinite(manual)
    assert trainer.oof_score_ == pytest.approx(
        float(np.sqrt(mean_squared_error(y[valid], manual[valid]))), rel=1e-6
    )


def test_oof_small_data_kfold_fallback_to_split(caplog):
    """N smaller than the fold count: k_fold falls back to train_test_split.

    OOF covers only the hold-out part; training completes normally. The
    warning refers to the method actually applied: the 'covers only the
    validation part' limitation is logged rather than a misleading
    'non-finite' log.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(6, 2))
    y = pd.Series(X[0] * 2.0 + rng.randn(6) * 0.1)

    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine.validation"):
        trainer = ModelTrainer(
            algorithm="ridge",
            metric="r2",
            validation_strategy="k_fold",
            n_folds=10,
            random_state=42,
        ).fit(X, y)

    assert "Falling back to 'train_test_split'" in caplog.text
    assert trainer.oof_coverage_ is not None
    assert 0.0 < trainer.oof_coverage_ < 1.0
    assert trainer.oof_score_ is not None
    # Fallback branch: the OOF limitation is documented honestly (not 'non-finite').
    assert "covers only the validation part" in caplog.text
    assert "excluded from RMSE_oof" not in caplog.text


def test_oof_nan_predictions_reduced_coverage(caplog):
    """NaN in fold predictions: the row is excluded; RMSE_oof over the rest."""
    rng = np.random.RandomState(4)
    X = pd.DataFrame(rng.randn(24, 2), columns=["f0", "f1"])
    y = pd.Series(X["f0"] * 2.0 + rng.randn(24) * 0.5)

    trainer = ModelTrainer(
        algorithm="ridge", metric="r2", validation_strategy="k_fold", n_folds=3
    )
    trainer.feature_names = ["f0", "f1"]
    trainer.categorical_features = []
    trainer.numerical_features = ["f0", "f1"]
    preprocessor = trainer._build_preprocessor(["f0", "f1"])
    # The model returns NaN for rows with f0 > 0.5 (white box).
    model = _ConstRegressor(value=1.5, nan_condition=lambda Xa: Xa[:, 0] > 0.5)

    with (
        caplog.at_level(logging.WARNING, logger="configurable_automl_engine.trainer"),
        patch(
            "configurable_automl_engine.trainer_scoring.get_scorer_object",
            return_value=lambda est, Xv, yv: 0.5,
        ),
    ):
        trainer._score_on_validation_splits(
            X, y, preprocessor, model, feature_selection_active=False
        )

    valid_mask = X["f0"] <= 0.5
    expected = float(
        np.sqrt(mean_squared_error(y[valid_mask], np.full(valid_mask.sum(), 1.5)))
    )
    assert trainer.oof_coverage_ == pytest.approx(valid_mask.mean(), rel=1e-9)
    assert trainer.oof_score_ == pytest.approx(expected, rel=1e-9)
    assert "excluded from RMSE_oof" in caplog.text
    assert trainer.val_score == pytest.approx(0.5)


def test_oof_nan_in_y_reduced_coverage(caplog):
    """NaN in y: rows with a non-finite target are excluded from RMSE_oof.

    Edge case: the vector metric must stay aligned — pairs with NaN in
    y_true are masked and training completes normally.
    """
    rng = np.random.RandomState(6)
    X = pd.DataFrame(rng.randn(24, 2), columns=["f0", "f1"])
    y = pd.Series(X["f0"] * 2.0 + rng.randn(24) * 0.5)
    y.iloc[[3, 9, 17]] = np.nan  # three rows with a non-finite target

    trainer = ModelTrainer(
        algorithm="ridge", metric="r2", validation_strategy="k_fold", n_folds=3
    )
    trainer.feature_names = ["f0", "f1"]
    trainer.categorical_features = []
    trainer.numerical_features = ["f0", "f1"]
    preprocessor = trainer._build_preprocessor(["f0", "f1"])
    # The model ignores y during fit — a white box isolating the OOF channel.
    model = _ConstRegressor(value=1.5)

    with (
        caplog.at_level(logging.WARNING, logger="configurable_automl_engine.trainer"),
        patch(
            "configurable_automl_engine.trainer_scoring.get_scorer_object",
            return_value=lambda est, Xv, yv: 0.5,
        ),
    ):
        trainer._score_on_validation_splits(
            X, y, preprocessor, model, feature_selection_active=False
        )

    valid_mask = y.notna()
    expected = float(
        np.sqrt(mean_squared_error(y[valid_mask], np.full(valid_mask.sum(), 1.5)))
    )
    assert trainer.oof_coverage_ == pytest.approx(valid_mask.mean(), rel=1e-9)
    assert trainer.oof_score_ == pytest.approx(expected, rel=1e-9)
    assert "excluded from RMSE_oof" in caplog.text


def test_oof_very_small_n_warning(caplog):
    """A very small N (2–5): a warning about the OOF estimate's high variance."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])

    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine.trainer"):
        trainer = ModelTrainer(
            algorithm="ridge",
            metric="r2",
            validation_strategy="k_fold",
            n_folds=2,
        ).fit(X, y)

    assert "very small data" in caplog.text
    assert trainer.oof_score_ is not None
    assert trainer.oof_coverage_ == 1.0


def test_oof_duplicate_index_falls_back_to_positional():
    """Unrecoverable index (duplicates): alignment falls back to position (ndarray)."""
    X, y = _noisy_regression(n=30, p=3, seed=23)
    X = X.copy()
    X.index = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14]
    y = y.copy()
    y.index = X.index

    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=3,
        random_state=5,
    ).fit(X, y)

    # A non-unique index would be ambiguous as a Series; return an ndarray by position.
    assert isinstance(trainer.oof_predictions_, np.ndarray)
    assert trainer.oof_coverage_ == 1.0
    manual = _manual_oof(X, y, trainer, method="k_fold", n_folds=3, seed=5)
    np.testing.assert_allclose(trainer.oof_predictions_, manual, rtol=1e-9, atol=1e-9)


def test_oof_multiindex_supported():
    """A MultiIndex: fit() does not fail and the row labels are preserved.

    Regression for a review blocker: ``Index.hasnans`` is undefined for
    MultiIndex (NotImplementedError), which crashed the whole ``fit()``
    even though such input worked on main. Now a unique MultiIndex is
    preserved in ``y_pred_oof`` as a ``pd.Series``.
    """
    idx = pd.MultiIndex.from_product(
        [["a", "b"], [1, 2, 3]], names=["grp", "num"]
    )
    rng = np.random.RandomState(11)
    X = pd.DataFrame(rng.randn(6, 2), index=idx)
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(6) * 0.5, index=idx)

    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=2,
        random_state=5,
    ).fit(X, y)

    assert trainer.oof_score_ is not None
    assert trainer.oof_coverage_ == 1.0
    assert isinstance(trainer.oof_predictions_, pd.Series)
    assert isinstance(trainer.oof_predictions_.index, pd.MultiIndex)
    assert list(trainer.oof_predictions_.index) == list(idx)
    np.testing.assert_allclose(
        trainer.oof_predictions_.to_numpy(),
        _manual_oof(X, y, trainer, method="k_fold", n_folds=2, seed=5),
        rtol=1e-9,
        atol=1e-9,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  6. Backward compatibility and state
# ──────────────────────────────────────────────────────────────────────────────


def test_oof_is_separate_channel_val_score_unchanged():
    """OOF is a separate channel: val_score/additional_scores are not affected.

    For a score metric (r2), RMSE_oof is exactly an RMSE (>= 0), not an R²;
    the channel values do not mix.
    """
    X, y = _noisy_regression(n=50, p=4, seed=29)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        additional_metrics=["rmse"],
        validation_strategy="k_fold",
        n_folds=4,
        random_state=5,
    ).fit(X, y)

    assert trainer.val_score is not None and trainer.val_score <= 1.0
    assert "rmse" in trainer.additional_scores
    assert trainer.oof_score_ is not None and trainer.oof_score_ >= 0
    # RMSE_oof is not the R² channel and is not a train score.
    assert trainer.oof_score_ != trainer.val_score
    # OOF-RMSE is consistent with the additional-metric channel over the full vector.
    assert trainer.oof_score_ == pytest.approx(
        float(
            np.sqrt(mean_squared_error(y, np.asarray(trainer.oof_predictions_)))
        ),
        rel=1e-9,
    )


def test_oof_vector_persisted_in_candidate_result(tmp_path):
    """The OOF vector is part of the candidate evaluation result: it is persisted.

    The diversity/unique audit circuits compute over y_pred_oof, so the
    vector (not only the metric) must survive serialization together with
    the pandas row index.
    """
    X, y = _noisy_regression(n=45, p=4, seed=31)
    X = X.copy()
    X.index = np.arange(len(X)) * 100 + 7
    y = y.copy()
    y.index = X.index
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=3,
        random_state=5,
    ).fit(X, y)

    path = tmp_path / "trainer_oof.pkl"
    trainer.save(path)
    loaded = ModelTrainer.load(path)

    assert loaded.oof_score_ == pytest.approx(trainer.oof_score_, rel=1e-12)
    assert loaded.oof_coverage_ == trainer.oof_coverage_
    # The vector is persisted as part of the result: its type and row index survive.
    assert isinstance(loaded.oof_predictions_, pd.Series)
    assert list(loaded.oof_predictions_.index) == list(X.index)
    np.testing.assert_allclose(
        loaded.oof_predictions_.to_numpy(),
        trainer.oof_predictions_.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_oof_reset_between_fits():
    """A repeat fit resets the OOF state (no leftovers from a previous training)."""
    X, y = _noisy_regression(n=50, p=4, seed=37)
    trainer = ModelTrainer(
        algorithm="ridge", metric="r2", validation_strategy="k_fold", n_folds=4
    ).fit(X, y)
    assert trainer.oof_score_ is not None

    # A failed fit must not leave OOF values from the previous training.
    with patch(
        "configurable_automl_engine.trainer_scoring.get_scorer_object",
        return_value=lambda est, Xv, yv: (_ for _ in ()).throw(RuntimeError("boom")),
    ):
        with pytest.raises(Exception, match="Validation scoring failed"):
            trainer.fit(X, y)

    assert trainer.oof_score_ is None
    assert trainer.oof_predictions_ is None
    assert trainer.oof_coverage_ is None
    assert trainer.val_score is None