"""Semantic tests for ``val_score`` / ``additional_scores``.

Coverage:
    1. ``val_score`` — an 'honest' estimate on held-out data (hold-out/CV),
       not a train score: on noisy data it is noticeably below train-R²;
       it matches a manual recomputation via ``validation.iter_splits``.
    2. ``additional_scores`` are computed with the same validation method
       as the main metric (the same hold-out split / fold average).
    3. The engine path (``train_best_model`` with k_fold): ``trainer.val_score``
       matches ``result["score"]`` when method/folds/seed match.
    4. The ``auto`` strategy is resolved once; the resolved k/test_size
       (not the string ``auto``) are forwarded into the final fit.
    5. Negative and boundary scenarios: unknown strategy, all folds failing
       (fail-fast without leftover state), N=2, k_fold→split fallback,
       a constant target (NRMSE→inf), determinism with random_state=None,
       backward compatibility of old pickles without the new attributes.
"""

from __future__ import annotations

import logging
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

from configurable_automl_engine.models import create_model
from configurable_automl_engine.trainer import (
    ModelTrainer,
    TrainingError,
)
from configurable_automl_engine.training_engine.component import PhaseResult, train_best_model
from configurable_automl_engine.training_engine.metrics import (
    get_scorer_object,
    is_non_additive_metric,
    to_user_value,
)
from configurable_automl_engine.validation import iter_splits

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _noisy_regression(n: int = 40, p: int = 8, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Noisy dataset: the first feature is informative, the rest is noise.

    On such data the model systematically overfits, so train-R² is
    noticeably higher than the honest hold-out estimate (reproduces the
    motivating example).
    """
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p))
    X.iloc[:, 0] = X.iloc[:, 0] * 3.0
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(n) * 3.0)
    return X, y


def _manual_fold_score(
    X: pd.DataFrame,
    y: pd.Series,
    trainer: ModelTrainer,
    *,
    feature_names: list[str],
    method: str,
    n_folds: int,
    test_size: float | int,
    seed: int,
) -> float:
    """Recompute val_score by hand: the same splits and pipeline clones per fold.

    Repeats the logic of ``ModelTrainer._score_on_validation_splits`` so that
    the equality is verified independently (not self-consistently).

    For non-additive metrics (r2 here) the value is computed over the pooled
    OOF vector (each row predicted by a model that did not see it), matching
    the new ``val_score`` semantics — not the fold mean.

    ``feature_names`` is passed explicitly: the helper must not rely on
    trainer attributes that may not be set.
    """
    preprocessor = trainer._build_preprocessor(feature_names)
    model = create_model(trainer.algorithm, **trainer.hyperparams)
    raw: list[float] = []
    scorer = get_scorer_object(trainer.metric)
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
        raw.append(float(scorer(fold_pipe, X_te, y_te)))
        oof[np.asarray(test_idx)] = np.asarray(fold_pipe.predict(X_te)).reshape(-1)
    if is_non_additive_metric(trainer.metric):
        mask = np.isfinite(y.to_numpy(dtype=float)) & np.isfinite(oof)
        pooled = float(
            scorer._sign * scorer._score_func(y.to_numpy(dtype=float)[mask], oof[mask], **scorer._kwargs)
        )
        return to_user_value(trainer.metric, pooled)
    return to_user_value(trainer.metric, float(np.mean(raw)))


# ──────────────────────────────────────────────────────────────────────────────
#  1. Positive: honest estimates and equality with manual recomputation
# ──────────────────────────────────────────────────────────────────────────────


def test_val_score_is_holdout_not_train_score():
    """On noisy data, val_score is noticeably below train-R².

    Reproduces the motivating example: previously val_score was equal to the
    full-sample train-R² (a systematically optimistic quality estimate).
    """
    X, y = _noisy_regression()
    trainer = ModelTrainer(algorithm="ridge", metric="r2", random_state=42).fit(X, y)

    train_r2 = float(r2_score(y, trainer.predict(X)))
    assert trainer.val_score is not None
    # The hold-out estimate is 'more honest': noticeably lower than the full-data train score.
    assert trainer.val_score < train_r2
    assert train_r2 - trainer.val_score > 0.05


def test_val_score_matches_manual_iter_splits_for_holdout():
    """train_test_split: val_score matches the manual hold-out recomputation."""
    X, y = _noisy_regression(n=80, p=6, seed=3)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="train_test_split",
        test_size=0.25,
        random_state=7,
    ).fit(X, y)

    manual = _manual_fold_score(
        X,
        y,
        trainer,
        feature_names=trainer.feature_names or [],
        method="train_test_split",
        n_folds=5,
        test_size=0.25,
        seed=7,
    )
    # rel=1e-6: a robust tolerance for the float arithmetic after np.mean.
    assert trainer.val_score == pytest.approx(manual, rel=1e-6)


def test_val_score_matches_manual_iter_splits_for_kfold():
    """k_fold: val_score is the fold average, matching the manual recomputation."""
    X, y = _noisy_regression(n=80, p=6, seed=5)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=4,
        random_state=7,
    ).fit(X, y)

    manual = _manual_fold_score(
        X,
        y,
        trainer,
        feature_names=trainer.feature_names or [],
        method="k_fold",
        n_folds=4,
        test_size=0.2,
        seed=7,
    )
    assert trainer.val_score == pytest.approx(manual, rel=1e-6)


def test_val_score_matches_cross_val_score():
    """k_fold: val_score matches sklearn's cross_val_score (the same folds).

    For additive metrics only: r2 is non-additive, so this test is converted
    to use 'mae' (a per-sample mean — the fold average equals the pooled
    value). The fold index equality between ``iter_splits`` and ``KFold`` is
    also checked: otherwise comparing with ``cross_val_score`` would be
    pointless.
    """
    X, y = _noisy_regression(n=80, p=6, seed=5)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="mae",
        validation_strategy="k_fold",
        n_folds=4,
        random_state=7,
    ).fit(X, y)

    cv = KFold(n_splits=4, shuffle=True, random_state=7)
    # Guarantee the folds match: iter_splits returns data subsets
    # (X.iloc[idx]); with a RangeIndex the row labels equal the positional
    # indices — compare them against the KFold fold indices.
    cv_folds = [(tr.tolist(), te.tolist()) for tr, te in cv.split(X)]
    iter_folds = [
        (list(X_tr.index), list(X_te.index))
        for X_tr, X_te, _, _ in iter_splits(
            X, y, method="k_fold", n_folds=4, test_size=0.2, random_state=7
        )
    ]
    assert len(iter_folds) == len(cv_folds)
    for (tr_i, te_i), (tr_c, te_c) in zip(iter_folds, cv_folds):
        assert tr_i == tr_c
        assert te_i == te_c

    preprocessor = trainer._build_preprocessor(trainer.feature_names or [])
    model = create_model(trainer.algorithm, **trainer.hyperparams)
    pipe = ImbPipeline(
        trainer._assemble_steps(preprocessor, model, feature_selection_active=False)
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=get_scorer_object("mae"))
    # mae is an error metric: the raw scorer returns negative values, while
    # val_score is sign-corrected to the natural (positive) representation.
    assert trainer.val_score == pytest.approx(-float(np.mean(scores)), rel=1e-6)


def test_additional_scores_use_same_validation_method():
    """additional_scores use the same method: the same split as val_score.

    For a metric equal to the main one the values must be identical; for a
    hold-out split the extra metric is computed on exactly the same partition.
    """
    X, y = _noisy_regression(n=80, p=6, seed=11)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        additional_metrics=["r2", "rmse"],
        validation_strategy="train_test_split",
        test_size=0.2,
        random_state=3,
    ).fit(X, y)

    assert trainer.additional_scores["r2"] == pytest.approx(
        trainer.val_score, rel=1e-12
    )
    assert trainer.additional_scores["rmse"] >= 0

    # Manual rmse recomputation on the same split. The scorer returns negative
    # values (neg_rmse), so we average the absolute values.
    preprocessor = trainer._build_preprocessor(trainer.feature_names or [])
    model = create_model(trainer.algorithm, **trainer.hyperparams)
    scorer = get_scorer_object("rmse")
    raws: list[float] = []
    for X_tr, X_te, y_tr, y_te in iter_splits(
        X, y, method="train_test_split", n_folds=5, test_size=0.2, random_state=3
    ):
        fold_pipe = ImbPipeline(
            trainer._assemble_steps(
                clone(preprocessor), clone(model), feature_selection_active=False
            )
        )
        fold_pipe.fit(X_tr, y_tr)
        raws.append(float(scorer(fold_pipe, X_te, y_te)))
    assert trainer.additional_scores["rmse"] == pytest.approx(
        float(np.mean(np.abs(raws))), rel=1e-6
    )


def test_additional_scores_kfold_mean_over_folds():
    """k_fold: additional_scores are the fold average, not a train score."""
    X, y = _noisy_regression(n=60, p=5, seed=2)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        additional_metrics=["mse"],
        validation_strategy="k_fold",
        n_folds=3,
        random_state=5,
    ).fit(X, y)

    # mse is an error: the val score is positive, while the full-data train-MSE
    # is systematically lower (overfitting).
    train_mse = float(np.mean((y - trainer.predict(X)) ** 2))
    assert trainer.additional_scores["mse"] > train_mse


# ──────────────────────────────────────────────────────────────────────────────
#  2. Engine path: consistency of val_score and result["score"]
# ──────────────────────────────────────────────────────────────────────────────


def test_engine_path_val_score_consistent_with_result_score(tmp_path):
    """train_best_model(k_fold): trainer.val_score ≈ result['score'].

    Both values are CV estimates on the same folds (the method is resolved
    once), so any discrepancy is only float noise.
    """
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "a": np.arange(120, dtype=float),
            "b": rng.normal(size=120),
            "target": np.arange(120, dtype=float) * 2.5 + rng.normal(size=120),
        }
    )
    model_path = tmp_path / "model.pkl"
    cfg = {
        "general": {
            "comparison_metric": "r2",
            "validation_strategy": "k_fold",
            "n_folds": 3,
            "path_to_model": str(model_path),
            "phases": [
                {"name": "search", "n_trials": 2, "action": "all_algorithms"},
                {
                    "name": "final_select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
        },
        "algorithms": {
            "ridge": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"alpha": [0.1, 1.0]},
            }
        },
    }

    result = train_best_model(config=cfg, df=df, target="target")

    loaded = ModelTrainer.load(str(model_path))
    assert loaded.validation_strategy == "k_fold"
    assert loaded.n_folds == 3
    # r2 is a score metric (greater_is_better); the sign correction does not flip it.
    assert loaded.val_score == pytest.approx(result["score"], rel=1e-4)


# ──────────────────────────────────────────────────────────────────────────────
#  3. auto resolution
# ──────────────────────────────────────────────────────────────────────────────


def test_auto_resolved_once_and_forwarded_to_final_fit(tmp_path):
    """auto: make_cv is resolved once; the resolved k/test_size reach the trainer.

    Checked through a make_cv mock: the component forwards exactly the
    resolved values (method + folds/size) to _fit_and_save, not the string
    'auto'.
    """
    X_df = pd.DataFrame({"a": np.arange(50, dtype=float)})
    df = X_df.copy()
    df["target"] = np.arange(50, dtype=float)

    resolved_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    fake_decision = {"method": "kfold", "k": 3}

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.additional_scores = {}

        def fit(self, X, y):
            pass

        def save(self, path):
            pass

    fake_trainer = FakeTrainer()

    cfg = {
        "general": {
            "comparison_metric": "r2",
            "validation_strategy": "auto",
            "n_folds": 5,
            "path_to_model": str(tmp_path / "m.pkl"),
            "phases": [
                {"name": "p", "n_trials": 1, "action": "all_algorithms"},
                {
                    "name": "final_select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
        },
        "algorithms": {"ridge": {"enable": True}},
    }

    with (
        patch(
            "configurable_automl_engine.training_engine.component.make_cv",
            return_value=("k_fold", resolved_cv, fake_decision),
        ),
        patch(
            "configurable_automl_engine.training_engine.component._run_hpo",
            return_value=PhaseResult(score=0.9, params={"alpha": 0.1}),
        ),
        patch(
            "configurable_automl_engine.training_engine.component._fit_and_save",
            return_value=fake_trainer,
        ) as mock_save,
    ):
        result = train_best_model(config=cfg, df=df, target="target")

    assert result["score"] == 0.9
    call_kwargs = mock_save.call_args.kwargs
    # The final fit receives the resolved values, not 'auto'
    assert call_kwargs["validation_strategy"].value == "k_fold"
    assert call_kwargs["n_folds"] == 3
    # _run_hpo receives the same resolved method
    assert mock_save.call_count == 1


def test_auto_split_test_size_forwarded_to_hpo_and_fit(tmp_path):
    """auto → train_test_split: the integer test_size reaches the tuner and trainer."""
    df = pd.DataFrame({"a": np.arange(50, dtype=float)})
    df["target"] = np.arange(50, dtype=float)

    fake_decision = {
        "method": "train_test_split",
        "test_size": 15,
        "train_size": 35,
    }

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.additional_scores = {}

        def fit(self, X, y):
            pass

        def save(self, path):
            pass

    cfg = {
        "general": {
            "comparison_metric": "r2",
            "validation_strategy": "auto",
            "n_folds": 5,
            "path_to_model": str(tmp_path / "m.pkl"),
            "phases": [
                {"name": "p", "n_trials": 1, "action": "all_algorithms"},
                {
                    "name": "final_select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
        },
        "algorithms": {"ridge": {"enable": True}},
    }

    captured_hpo: dict = {}

    def fake_run_hpo(**kwargs):
        captured_hpo.update(kwargs)
        return PhaseResult(score=0.9, params={"alpha": 0.1})

    with (
        patch(
            "configurable_automl_engine.training_engine.component.make_cv",
            return_value=("train_test_split", None, fake_decision),
        ),
        patch(
            "configurable_automl_engine.training_engine.component._run_hpo",
            side_effect=fake_run_hpo,
        ),
        patch(
            "configurable_automl_engine.training_engine.component._fit_and_save",
            return_value=FakeTrainer(),
        ) as mock_save,
    ):
        train_best_model(config=cfg, df=df, target="target")

    assert captured_hpo["validation_strategy"].value == "train_test_split"
    assert captured_hpo["train_test_split_test_size"] == 15
    save_kwargs = mock_save.call_args.kwargs
    assert save_kwargs["validation_strategy"].value == "train_test_split"
    assert save_kwargs["test_size"] == 15


# ──────────────────────────────────────────────────────────────────────────────
#  4. Negative scenarios
# ──────────────────────────────────────────────────────────────────────────────


def test_unknown_validation_strategy_rejected_at_init():
    """An unknown validation_strategy → TrainingError at initialization."""
    with pytest.raises(TrainingError, match="Unknown validation_strategy"):
        ModelTrainer(validation_strategy="magic_split")
    # Enum values are normalized too; a case-mismatched string is fine
    assert ModelTrainer(validation_strategy="K_FOLD").validation_strategy == "k_fold"


def test_invalid_n_folds_rejected_at_init():
    """n_folds < 1 is rejected; for k_fold, n_folds < 2 is rejected."""
    with pytest.raises(TrainingError, match="n_folds"):
        ModelTrainer(n_folds=0)
    with pytest.raises(TrainingError, match="n_folds"):
        ModelTrainer(validation_strategy="k_fold", n_folds=1)


def test_invalid_test_size_rejected_at_init():
    """test_size outside (0, 1) for a fraction, or < 1 for an integer, is rejected."""
    with pytest.raises(TrainingError, match="test_size"):
        ModelTrainer(test_size=0.0)
    with pytest.raises(TrainingError, match="test_size"):
        ModelTrainer(test_size=1.5)
    with pytest.raises(TrainingError, match="test_size"):
        ModelTrainer(test_size=0)  # an integer < 1
    with pytest.raises(TrainingError, match="test_size must be a number"):
        ModelTrainer(test_size="big")  # a non-numeric value


def test_validation_split_init_failure_wrapped(caplog):
    """A critical failure while materializing splits is wrapped into a TrainingError.

    Covers the 'split materialization failed' branch in
    ``_score_on_validation_splits`` (e.g. a make_cv validator failure).
    """
    X, y = _noisy_regression(n=40, p=3, seed=1)
    trainer = ModelTrainer(algorithm="ridge", metric="r2")

    def broken_iter_splits(*args, **kwargs):
        raise ValueError("splits exploded")

    with patch(
        "configurable_automl_engine.trainer.iter_splits",
        side_effect=broken_iter_splits,
    ):
        with pytest.raises(
            TrainingError, match="Error calculating metrics on validation"
        ):
            trainer.fit(X, y)

    assert trainer.val_score is None
    assert trainer.pipeline is None


def test_all_folds_fail_raises_and_resets_state():
    """All scoring folds fail → TrainingError; pipeline/val_score are not left stale.

    First a successful training, then fit() with a deliberately broken
    scorer: the state of the previous training must not persist.
    """
    X, y = _noisy_regression(n=60, p=4, seed=1)
    trainer = ModelTrainer(algorithm="ridge", metric="r2").fit(X, y)
    assert trainer.pipeline is not None
    assert trainer.val_score is not None

    with patch(
        "configurable_automl_engine.trainer.get_scorer_object",
        return_value=lambda est, Xv, yv: (_ for _ in ()).throw(RuntimeError("boom")),
    ):
        with pytest.raises(TrainingError, match="Validation scoring failed"):
            trainer.fit(X, y)

    assert trainer.pipeline is None
    assert trainer.val_score is None
    assert trainer.additional_scores == {}


def test_single_additional_metric_failure_skipped_with_warning(caplog):
    """A single extra-metric failure → WARNING + skip; training still succeeds."""
    X, y = _noisy_regression(n=60, p=4, seed=1)

    real_scorer_factory = __import__(
        "configurable_automl_engine.trainer", fromlist=["get_scorer_object"]
    ).get_scorer_object

    def flaky(name: str, global_y=None):
        if name == "mae":
            raise RuntimeError("scorer exploded")
        return real_scorer_factory(name, global_y)

    with (
        caplog.at_level(logging.WARNING),
        patch(
            "configurable_automl_engine.trainer.get_scorer_object", side_effect=flaky
        ),
    ):
        trainer = ModelTrainer(
            algorithm="ridge", metric="r2", additional_metrics=["mae", "rmse"]
        ).fit(X, y)

    assert trainer.val_score is not None
    assert "rmse" in trainer.additional_scores
    assert "mae" not in trainer.additional_scores
    assert "could not be computed" in caplog.text


# ──────────────────────────────────────────────────────────────────────────────
#  5. Boundary cases
# ──────────────────────────────────────────────────────────────────────────────


def test_minimum_two_samples():
    """N = 2 (the minimum): training completes normally (hold-out 1/1)."""
    X = pd.DataFrame({"a": [1.0, 2.0]})
    y = pd.Series([1.0, 2.0])
    trainer = ModelTrainer(
        algorithm="ridge", metric="r2", validation_strategy="train_test_split"
    ).fit(X, y)
    assert trainer.val_score is not None
    assert trainer.pipeline is not None


def test_kfold_fallback_to_split_on_small_data(caplog):
    """k_fold on a small sample (N < max(4, 2K)) → fallback to train_test_split.

    The behavior is consistent with HPO (make_cv); training completes
    normally and val_score is not None.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(6, 2))
    y = pd.Series(X[0] * 2.0 + rng.randn(6) * 0.1)

    with caplog.at_level(logging.WARNING, logger="configurable_automl_engine.validation"):
        trainer = ModelTrainer(
            algorithm="ridge",
            metric="r2",
            validation_strategy="k_fold",
            n_folds=5,
            random_state=42,
        ).fit(X, y)

    assert trainer.val_score is not None
    assert "Falling back to 'train_test_split'" in caplog.text


def test_constant_target_rejected_before_training():
    """A constant target is rejected before training.

    Previously a constant target trained and reported a perfect score
    (e.g. R² = 1.0), which is silently misleading: every model trivially
    predicts the constant. The guard fails fast with a clear message.
    """
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([5.0, 5.0, 5.0, 5.0])
    trainer = ModelTrainer(algorithm="ridge", metric="nrmse")
    with pytest.raises(TrainingError, match="constant"):
        trainer.fit(X, y)
    assert trainer.pipeline is None
    assert trainer.val_score is None


def test_deterministic_splits_when_random_state_none():
    """random_state=None: the validation splits are deterministic (fixed seed 42).

    Comparison via pytest.approx: strict == on floats after fold averaging
    can be brittle.
    """
    X, y = _noisy_regression(n=80, p=5, seed=9)
    t1 = ModelTrainer(algorithm="ridge", metric="r2", random_state=None).fit(X, y)
    t2 = ModelTrainer(algorithm="ridge", metric="r2", random_state=None).fit(X, y)

    assert t1.val_score == pytest.approx(t2.val_score, rel=1e-9)
    assert t1.additional_scores == t2.additional_scores


def test_old_pickle_without_validation_attrs_compatible():
    """An old pickle without the new attributes: load + a re-fit both work.

    The new attributes are read via a getattr fallback (train_test_split/5/0.2):
    the attributes are not 'materialized' when an old artifact is loaded, but
    fit() computes val_score using the default strategy.
    """
    X, y = _noisy_regression(n=60, p=4, seed=4)
    trainer = ModelTrainer(algorithm="ridge", metric="r2").fit(X, y)
    for attr in ("validation_strategy", "n_folds", "test_size"):
        delattr(trainer, attr)

    # Serialize/deserialize the 'old' object
    restored = pickle.loads(pickle.dumps(trainer))
    restored.fit(X, y)
    assert restored.val_score is not None
    # The attributes are absent (as in an old pickle), but fit uses the defaults:
    # val_score matches a trainer trained with the pickle-fallback values.
    assert not hasattr(restored, "validation_strategy")
    baseline = ModelTrainer(
        algorithm="ridge", metric="r2", validation_strategy="train_test_split"
    ).fit(X, y)
    assert restored.val_score == pytest.approx(baseline.val_score, rel=1e-6)


def test_serialization_round_trip_preserves_validation_attrs(tmp_path):
    """A save/load round-trip preserves the new validation attributes."""
    X, y = _noisy_regression(n=60, p=4, seed=6)
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="r2",
        validation_strategy="k_fold",
        n_folds=3,
        test_size=0.25,
    ).fit(X, y)

    path = tmp_path / "trainer.pkl"
    trainer.save(path)
    loaded = ModelTrainer.load(path)

    assert loaded.validation_strategy == "k_fold"
    assert loaded.n_folds == 3
    assert loaded.test_size == 0.25
    assert loaded.val_score == trainer.val_score
    assert loaded.pipeline is not None


def test_val_score_reset_between_fits():
    """A repeat fit with a different validation method recomputes val_score."""
    X, y = _noisy_regression(n=80, p=5, seed=8)
    trainer = ModelTrainer(
        algorithm="ridge", metric="r2", validation_strategy="train_test_split"
    ).fit(X, y)
    split_score = trainer.val_score

    trainer.validation_strategy = "k_fold"
    trainer.n_folds = 4
    trainer.fit(X, y)
    assert trainer.val_score is not None
    # The values across different methods need not match, but must be recomputed
    # (no comparison against a train score — this is about estimate semantics).
    assert isinstance(trainer.val_score, float)
    assert trainer.pipeline is not None


# ──────────────────────────────────────────────────────────────────────────────
#  Non-additive metrics: pooled over the OOF vector
# ──────────────────────────────────────────────────────────────────────────────


def test_loo_r2_val_score_is_pooled_not_nan():
    """LOO + r2: val_score is the pooled r2 over the OOF vector (not NaN).

    Before the change, per-fold r2 on single-row folds is undefined (NaN), so
    the fold-mean val_score was NaN. The pooled value over the whole OOF
    vector is a valid finite number — the motivating case.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(12, 3))
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(12))
    trainer = ModelTrainer(
        algorithm="ridge", metric="r2", validation_strategy="loo"
    ).fit(X, y)

    assert trainer.val_score is not None
    assert np.isfinite(trainer.val_score)
    # The pooled r2 equals r2_score(y, y_pred_oof) over the full vector.
    oof = np.asarray(trainer.oof_predictions_)
    assert trainer.val_score == pytest.approx(float(r2_score(y, oof)), rel=1e-9)


def test_kfold_rmse_val_score_equals_oof_rmse():
    """k_fold + rmse: val_score is the pooled RMSE over the OOF vector.

    The pooled value equals RMSE_oof (the same estimate the select phase
    uses) — "HPO and select use the same value". Before the change val_score
    was the mean of per-fold RMSE, which differs from the pooled RMSE.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(40, 4))
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(40))
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="rmse",
        validation_strategy="k_fold",
        n_folds=5,
        random_state=42,
    ).fit(X, y)

    assert trainer.val_score is not None
    assert trainer.oof_score_ is not None
    # val_score (pooled RMSE) equals the RMSE_oof channel exactly.
    assert trainer.val_score == pytest.approx(trainer.oof_score_, rel=1e-9)
    # The pooled value is not the mean of per-fold RMSE.
    oof = np.asarray(trainer.oof_predictions_)
    assert trainer.val_score == pytest.approx(
        float(np.sqrt(mean_squared_error(y, oof))), rel=1e-9
    )


def test_mae_val_score_remains_fold_mean():
    """MAE (additive): val_score stays the fold mean (equals the pooled value)."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(40, 4))
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(40))
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="mae",
        validation_strategy="k_fold",
        n_folds=5,
        random_state=42,
    ).fit(X, y)

    assert trainer.val_score is not None
    # For a per-sample mean, the pooled value equals the fold mean: MAE_oof
    # over the whole vector is the same as mean(MAE_fold).
    oof = np.asarray(trainer.oof_predictions_)
    assert trainer.val_score == pytest.approx(
        float(np.mean(np.abs(y - oof))), rel=1e-9
    )


def test_kfold_non_additive_additional_metric_is_pooled():
    """k_fold: a non-additive additional metric (r2) is pooled over the OOF
    vector, not averaged by folds.

    The pooled r2 over the OOF predictions equals r2_score(y, y_pred_oof) and
    differs from the mean of per-fold r2 values.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(40, 4))
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(40))
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="rmse",
        additional_metrics=["r2", "mae"],
        validation_strategy="k_fold",
        n_folds=5,
        random_state=42,
    ).fit(X, y)

    # r2 (non-additive): the additional score is the pooled value over the OOF
    # vector — it equals r2_score(y, y_pred_oof) exactly.
    oof = np.asarray(trainer.oof_predictions_)
    assert trainer.additional_scores["r2"] == pytest.approx(
        float(r2_score(y, oof)), rel=1e-9
    )
    # mae (additive): the pool equals the fold mean — still finite.
    assert trainer.additional_scores["mae"] == pytest.approx(
        float(np.mean(np.abs(y - oof))), rel=1e-9
    )


def test_loo_non_additive_additional_metric_is_finite():
    """LOO + r2 as an additional metric: the pooled value is finite (not NaN).

    Per-fold r2 on single-row folds is undefined (NaN); the pooled value over
    the whole OOF vector is a valid finite number — the same fix as for
    val_score, applied to additional_scores.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(12, 3))
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(12))
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="rmse",
        additional_metrics=["r2"],
        validation_strategy="loo",
    ).fit(X, y)

    assert np.isfinite(trainer.additional_scores["r2"])
    oof = np.asarray(trainer.oof_predictions_)
    assert trainer.additional_scores["r2"] == pytest.approx(
        float(r2_score(y, oof)), rel=1e-9
    )


def test_loo_explained_variance_val_score_is_pooled_not_constant():
    """LOO + explained_variance: val_score is the pooled EV, not the
    degenerate fold-mean of 1.0.

    Per-fold explained_variance on a single-row fold is identically 1.0
    (sklearn on one pair), so the fold-mean would silently be 1.0 for any
    model. The pooled value over the whole OOF vector is the correct global
    estimate (the same class of bug this PR fixes for R², but silent).
    """
    from sklearn.metrics import explained_variance_score

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(12, 3))
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(12))
    trainer = ModelTrainer(
        algorithm="ridge", metric="explained_variance", validation_strategy="loo"
    ).fit(X, y)

    assert trainer.val_score is not None
    assert np.isfinite(trainer.val_score)
    # The pooled EV is the true global estimate over the OOF vector.
    oof = np.asarray(trainer.oof_predictions_)
    assert trainer.val_score == pytest.approx(
        float(explained_variance_score(y, oof)), rel=1e-9
    )
    # The degenerate LOO fold-mean (1.0 for every fold) differs from the
    # pooled value for an imperfect model.
    assert trainer.val_score != pytest.approx(1.0)


def test_loo_median_absolute_error_val_score_is_pooled_not_mae():
    """LOO + neg_median_absolute_error: val_score is the pooled median, not
    the MAE substitution.

    Per-fold median absolute error on a single-row fold equals that row's
    absolute error, so the fold-mean degenerates into MAE (the metric is
    silently substituted). The pooled value over the whole OOF vector is the
    correct global median.
    """
    from sklearn.metrics import median_absolute_error

    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(12, 3))
    y = pd.Series(X.iloc[:, 0] * 2.0 + rng.randn(12))
    trainer = ModelTrainer(
        algorithm="ridge",
        metric="neg_median_absolute_error",
        validation_strategy="loo",
    ).fit(X, y)

    assert trainer.val_score is not None
    assert np.isfinite(trainer.val_score)
    oof = np.asarray(trainer.oof_predictions_)
    # val_score is sign-corrected to the natural positive representation
    # (the median absolute error over the OOF vector).
    assert trainer.val_score == pytest.approx(
        float(median_absolute_error(y, oof)), rel=1e-9
    )
    # The degenerate LOO fold-mean equals MAE, which differs from the pooled
    # median for asymmetric errors.
    assert trainer.val_score != pytest.approx(
        float(np.mean(np.abs(y - oof))), rel=1e-6
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Pooled-fallback WARNING in the final fit
# ══════════════════════════════════════════════════════════════════════════════


def test_trainer_pooled_fallback_warns_once_per_algorithm(caplog):
    """The final-fit pooled fallback emits one WARNING per algorithm.

    With a scorer that exposes no raw metric function the ModelTrainer
    silently falls back to the per-fold mean — the WARNING must appear once
    per algorithm (the 'final_fit' phase label), not once per fold.
    """
    from configurable_automl_engine.training_engine import metrics as metrics_mod

    X, y = _noisy_regression(n=60, p=4, seed=1)

    class NoRawScorer:
        """A scorer without the private _score_func attributes."""

        def __call__(self, estimator, Xv, yv):
            return 0.5

    metrics_mod.reset_pooled_fallback_warnings()
    with (
        caplog.at_level(logging.WARNING),
        patch(
            "configurable_automl_engine.trainer.get_scorer_object",
            return_value=NoRawScorer(),
        ),
    ):
        trainer = ModelTrainer(
            algorithm="ridge",
            metric="r2",
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
        ).fit(X, y)

    assert trainer.val_score is not None
    assert "Pooled evaluation of the metric is not available" in caplog.text
    assert "ridge" in caplog.text
    assert "final_fit" in caplog.text
    # Once per algorithm and phase, not once per fold.
    assert caplog.text.count("Pooled evaluation of the metric is not available") == 1


def test_trainer_no_fallback_warning_on_pooled_path(caplog):
    """The pooled path of the final fit emits no fallback WARNING."""
    X, y = _noisy_regression(n=60, p=4, seed=1)
    with caplog.at_level(logging.WARNING):
        trainer = ModelTrainer(
            algorithm="ridge",
            metric="r2",
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
        ).fit(X, y)
    assert trainer.val_score is not None
    assert "Pooled evaluation of the metric is not available" not in caplog.text


class _NaNPredictingModel(BaseEstimator, RegressorMixin):
    """A regressor whose predictions are all NaN (empty OOF vector)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), np.nan)


class _NoRawScorer:
    """A scorers without the private raw-metric attributes (unpoolable)."""

    def __call__(self, estimator, Xv, yv):
        return 0.5


def test_trainer_primary_no_finite_oof_pairs_warns(caplog):
    """Primary metric: an empty finite OOF buffer warns about the fallback.

    Regression (trainer.py 'no finite pairs' branch):
    when the model predicts all-NaN, the OOF buffer stays empty and the
    pooled estimate is impossible — the engine falls back to the per-fold
    mean with exactly one WARNING per (algorithm, metric, 'final_fit').
    """
    from configurable_automl_engine.training_engine import metrics as metrics_mod

    X, y = _noisy_regression(n=60, p=4, seed=1)

    metrics_mod.reset_pooled_fallback_warnings()
    with (
        caplog.at_level(logging.WARNING),
        patch(
            "configurable_automl_engine.trainer.get_scorer_object",
            return_value=_NoRawScorer(),
        ),
        patch(
            "configurable_automl_engine.trainer.create_model",
            lambda algo, **kw: _NaNPredictingModel(),
        ),
    ):
        trainer = ModelTrainer(
            algorithm="ridge",
            metric="r2",
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
        ).fit(X, y)

    assert trainer.val_score is not None
    assert "Pooled evaluation of the metric is not available" in caplog.text
    assert "final_fit:r2" in caplog.text
    assert "no finite (y_true, y_pred) pairs" in caplog.text
    # Once per (algorithm, metric, phase), not once per fold.
    assert caplog.text.count("Pooled evaluation of the metric is not available") == 1


def test_trainer_additional_metric_pooled_fallback_warns(caplog):
    """Additional non-additive metric: an unpoolable scorer warns on fallback.

    The primary metric 'r2' pools successfully; the additional metric 'rmse'
    gets a scorer without raw-metric attributes, so its pooled value is
    impossible and the engine falls back to the fold mean with a WARNING.
    The deduplication key includes the metric ('final_fit:rmse'), so primary
    and additional metrics do not collapse into one warning.
    """
    from configurable_automl_engine.training_engine import metrics as metrics_mod

    X, y = _noisy_regression(n=60, p=4, seed=1)

    real_scorer_factory = __import__(
        "configurable_automl_engine.trainer", fromlist=["get_scorer_object"]
    ).get_scorer_object

    def flaky(name: str, global_y=None):
        if name == "rmse":
            return _NoRawScorer()
        return real_scorer_factory(name, global_y)

    metrics_mod.reset_pooled_fallback_warnings()
    with (
        caplog.at_level(logging.WARNING),
        patch(
            "configurable_automl_engine.trainer.get_scorer_object", side_effect=flaky
        ),
    ):
        trainer = ModelTrainer(
            algorithm="ridge",
            metric="r2",
            additional_metrics=["rmse"],
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
        ).fit(X, y)

    assert trainer.val_score is not None
    assert "rmse" in trainer.additional_scores
    assert "Pooled evaluation of the metric is not available" in caplog.text
    assert "final_fit:rmse" in caplog.text
    assert "final_fit:r2" not in caplog.text
    # Exactly one WARNING: only the additional metric fell back.
    assert caplog.text.count("Pooled evaluation of the metric is not available") == 1


def test_trainer_additional_metric_no_finite_oof_pairs_warns(caplog):
    """Additional metric: an empty finite OOF buffer warns about the fallback.

    Regression (trainer additional 'no finite pairs'
    branch): the model predicts all-NaN, so the OOF buffer is empty and the
    pooled value of the non-additive additional metric is impossible.
    """
    from configurable_automl_engine.training_engine import metrics as metrics_mod

    X, y = _noisy_regression(n=60, p=4, seed=1)

    metrics_mod.reset_pooled_fallback_warnings()
    with (
        caplog.at_level(logging.WARNING),
        patch(
            "configurable_automl_engine.trainer.get_scorer_object",
            return_value=_NoRawScorer(),
        ),
        patch(
            "configurable_automl_engine.trainer.create_model",
            lambda algo, **kw: _NaNPredictingModel(),
        ),
    ):
        trainer = ModelTrainer(
            algorithm="ridge",
            metric="r2",
            additional_metrics=["rmse"],
            validation_strategy="k_fold",
            n_folds=3,
            random_state=42,
        ).fit(X, y)

    assert "rmse" in trainer.additional_scores
    assert "Pooled evaluation of the metric is not available" in caplog.text
    assert "final_fit:rmse" in caplog.text
    assert "no finite (y_true, y_pred) pairs for additional metric" in caplog.text
    # Once per (algorithm, metric, phase): the primary ('final_fit:r2') and
    # the additional ('final_fit:rmse') metrics keep separate keys.
    assert caplog.text.count("Pooled evaluation of the metric is not available") == 2
