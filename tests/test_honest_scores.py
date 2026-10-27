"""Tests for the honest out-of-sample report keys.

Coverage:
    1. ``result["oof_score"]`` — the comparison metric computed on the final
       pipeline's pooled OOF predictions (the trainer's OOF channel), present
       for error (RMSE/MAE) and score (R²) metrics; equals a manual
       recomputation over ``trainer.oof_predictions_``. It is computed on the
       HPO folds, so without an outer hold-out it equals the best-trial
       ``score`` (it is NOT an independent estimate).
    2. ``result["n_trials_evaluated"]`` — the number of trials actually
       evaluated by the winner's HPO phase (own COMPLETED trials; transferred
       history never counts); 0 for custom tuners that do not report it.
    3. ``general.holdout_fraction`` — config validation (default 'auto',
       (0, 1) range, explicit None disables it), the 'auto' resolution by the
       number of rows, the outer split before HPO (rows never participate in
       model selection), ``result["holdout_score"]`` semantics (the winner
       trained on the TRAIN part only), and the final artifact still trained
       on ALL data.
    4. Backward compatibility: without the hold-out fraction the
       ``holdout_score`` key is absent; without the OOF channel the
       ``oof_score`` key is absent; custom tuners (legacy triple) report 0
       evaluated trials.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from configurable_automl_engine.trainer import ModelTrainer
from configurable_automl_engine.training_engine.component import (
    PhaseResult,
    prepare_dataset,
    train_best_model,
)
from configurable_automl_engine.training_engine.config_parser import (
    AUTO_HOLDOUT_FRACTION,
    AUTO_HOLDOUT_MIN_ROWS,
    Config,
)
from configurable_automl_engine.validation import RANDOM_STATE


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _zero_trainer(prepared):
    """A mock trainer without OOF predictions, shaped to the prepared data.

    The mock must be aligned with the data (so ``_oof_score_in_user_metric``
    can attempt the estimate) and expose the attributes the report assembly
    touches.
    """
    trainer = MagicMock()
    trainer.additional_scores = {}
    trainer.oof_predictions_ = None
    trainer.oof_score_ = None
    trainer.feature_names = list(prepared.X.columns)
    return trainer


def _noisy_regression(n: int = 120, p: int = 8, seed: int = 7) -> pd.DataFrame:
    """A noisy dataset with a target column (the first feature informative).

    On such data the best-trial score is mildly optimistic, which gives the
    honest out-of-sample keys something meaningful to report.
    """
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p))
    X.iloc[:, 0] = X.iloc[:, 0] * 3.0
    df = X.copy()
    df["target"] = X.iloc[:, 0] * 2.0 + rng.randn(n) * 3.0
    return df


def _cfg(
    *,
    metric: str = "r2",
    holdout_fraction: float | str | None = None,
    n_trials: int = 3,
    model_path: str = "model.pkl",
    select_strategy: str = "cv_best",
    trainer_module: str | None = None,
) -> dict:
    """A minimal valid config for a single-algorithm run."""
    algo: dict = {
        "enable": True,
        "limit_hyperparameters": True,
        "hyperparameters": {"alpha": [0.1, 1.0]},
    }
    if trainer_module is not None:
        algo["trainer_module"] = trainer_module
    return {
        "general": {
            "comparison_metric": metric,
            "validation_strategy": "k_fold",
            "n_folds": 4,
            "holdout_fraction": holdout_fraction,
            "path_to_model": model_path,
            "phases": [
                {"name": "search", "n_trials": n_trials, "action": "all_algorithms"},
                {
                    "name": "final_select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": select_strategy,
                    "output": "winner",
                },
            ],
        },
        "algorithms": {"ridge": algo},
    }


# ──────────────────────────────────────────────────────────────────────────────
#  1. oof_score
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("metric", "manual_metric"),
    [
        ("r2", lambda y, p: float(r2_score(y, p))),
        ("rmse", lambda y, p: float(np.sqrt(mean_squared_error(y, p)))),
        ("mae", lambda y, p: float(mean_absolute_error(y, p))),
    ],
)
def test_oof_score_matches_manual_oof_recomputation(tmp_path, metric, manual_metric):
    """oof_score is the comparison metric on the final pipeline's pooled OOF.

    The OOF vector is aligned with the rows the model has not been trained on
    (k_fold covers every row exactly once); the reported value equals a manual
    recomputation over ``trainer.oof_predictions_``.
    """
    df = _noisy_regression()
    cfg = _cfg(
        metric=metric, model_path=str(tmp_path / "model.pkl")
    )
    result = train_best_model(config=cfg, df=df, target="target")

    assert "oof_score" in result
    loaded = ModelTrainer.load(str(tmp_path / "model.pkl"))
    oof = np.asarray(loaded.oof_predictions_, dtype=float).reshape(-1)
    y = np.asarray(df["target"], dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(oof)
    assert np.any(mask)
    assert result["oof_score"] == pytest.approx(
        manual_metric(y[mask], oof[mask]), rel=1e-9
    )
    # The best-trial score is the value the HPO phase maximized; the OOF
    # estimate is computed on the final pipeline and is independent of the
    # score direction (both are present).
    assert "score" in result
    assert "n_trials_evaluated" in result


@pytest.mark.parametrize("metric", ["r2", "rmse"])
def test_oof_score_equals_best_trial_score_without_holdout(tmp_path, metric):
    """Without an outer hold-out oof_score is NOT independent of selection.

    The final pipeline's OOF vector is built on the same folds (same seed)
    on which the winner and its hyperparameters were selected, so for a
    pooled metric on k_fold it reproduces the best-trial ``score`` exactly:
    it carries the same best-of-N bias and must not be read as an honest
    estimate (the honest one is ``holdout_score``).
    """
    df = _noisy_regression(n=200, p=10, seed=3)
    cfg = _cfg(metric=metric, n_trials=10, model_path=str(tmp_path / "model.pkl"))
    result = train_best_model(config=cfg, df=df, target="target")

    loaded = ModelTrainer.load(str(tmp_path / "model.pkl"))
    oof = np.asarray(loaded.oof_predictions_, dtype=float).reshape(-1)
    y = np.asarray(df["target"], dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(oof)
    manual_fn = r2_score if metric == "r2" else (
        lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
    )
    assert result["oof_score"] == pytest.approx(manual_fn(y[mask], oof[mask]), rel=1e-9)
    assert result["oof_score"] == pytest.approx(result["score"], rel=1e-9)
    assert "holdout_score" not in result


def test_oof_score_absent_without_oof_channel(tmp_path):
    """A trainer without OOF predictions (e.g. a mocked final fit) → the key
    is absent (backward compatibility)."""
    cfg = _cfg(model_path=str(tmp_path / "model.pkl"))
    fake_trainer = MagicMock()
    fake_trainer.additional_scores = {}
    fake_trainer.oof_predictions_ = None
    fake_trainer.oof_score_ = None

    with (
        patch(
            "configurable_automl_engine.training_engine.hpo_phases._run_hpo",
            return_value=PhaseResult(score=0.9, params={"alpha": 0.1}),
        ),
        patch(
            "configurable_automl_engine.training_engine.artifact._fit_and_save",
            return_value=fake_trainer,
        ),
    ):
        result = train_best_model(config=cfg, df=_noisy_regression(n=40), target="target")

    assert "oof_score" not in result


# ──────────────────────────────────────────────────────────────────────────────
#  2. n_trials_evaluated
# ──────────────────────────────────────────────────────────────────────────────


def test_n_trials_evaluated_reports_completed_trials(tmp_path):
    """n_trials_evaluated counts the winner's own COMPLETED trials."""
    df = _noisy_regression(n=60, p=4, seed=5)
    cfg = _cfg(n_trials=3, model_path=str(tmp_path / "model.pkl"))
    result = train_best_model(config=cfg, df=df, target="target")
    assert result["n_trials_evaluated"] == 3


def test_n_trials_evaluated_zero_for_legacy_tuner(tmp_path):
    """A custom tuner returning the legacy triple reports an UNKNOWN count —
    the key is omitted (0 would be ambiguous for the consumer)."""
    cfg = _cfg(model_path=str(tmp_path / "model.pkl"))
    fake_trainer = MagicMock()
    fake_trainer.additional_scores = {}
    fake_trainer.oof_predictions_ = None
    fake_trainer.oof_score_ = None

    with (
        patch(
            "configurable_automl_engine.training_engine.hpo_phases._run_hpo",
            return_value=PhaseResult(score=0.9, params={"alpha": 0.1}),
        ),
        patch(
            "configurable_automl_engine.training_engine.artifact._fit_and_save",
            return_value=fake_trainer,
        ),
    ):
        result = train_best_model(config=cfg, df=_noisy_regression(n=40), target="target")

    assert "n_trials_evaluated" not in result


def test_n_trials_evaluated_reported_for_real_tuner():
    """The bundled tuner reports the count on OptimizeResult and the component
    carries it into the report."""
    from configurable_automl_engine.tuner import OptimizeResult

    result = OptimizeResult(
        model=None,
        params={"alpha": 0.5},
        score=0.5,
        trials=[],
        n_trials_evaluated=7,
    )
    assert result.n_trials_evaluated == 7
    # The default for code paths that do not set the field is 0.
    legacy = OptimizeResult(model=None, params={"a": 1}, score=0.1, trials=[])
    assert legacy.n_trials_evaluated == 0


# ──────────────────────────────────────────────────────────────────────────────
#  3. holdout_fraction
# ──────────────────────────────────────────────────────────────────────────────


def test_holdout_fraction_config_default_and_range():
    """holdout_fraction defaults to 'auto', accepts None and (0, 1) only."""
    from pydantic import ValidationError

    raw = _cfg()
    del raw["general"]["holdout_fraction"]
    assert Config.model_validate(raw).general.holdout_fraction == "auto"

    cfg = Config.model_validate(_cfg(holdout_fraction="auto"))
    assert cfg.general.holdout_fraction == "auto"

    cfg = Config.model_validate(_cfg(holdout_fraction=None))
    assert cfg.general.holdout_fraction is None

    cfg = Config.model_validate(_cfg(holdout_fraction=0.2))
    assert cfg.general.holdout_fraction == 0.2

    for bad in (0.0, 1.0, -0.1, 1.5, "always", ""):
        with pytest.raises(ValidationError):
            Config.model_validate(_cfg(holdout_fraction=bad))


@pytest.mark.parametrize(
    ("n_rows", "expected_holdout"),
    [
        (AUTO_HOLDOUT_MIN_ROWS - 1, None),
        (AUTO_HOLDOUT_MIN_ROWS, int(np.ceil(AUTO_HOLDOUT_FRACTION * AUTO_HOLDOUT_MIN_ROWS))),
        (250, int(np.ceil(AUTO_HOLDOUT_FRACTION * 250))),
    ],
)
def test_holdout_fraction_auto_resolution(n_rows, expected_holdout):
    """'auto' reserves AUTO_HOLDOUT_FRACTION from AUTO_HOLDOUT_MIN_ROWS rows on
    (the boundary is inclusive) and no hold-out below it."""
    df = _noisy_regression(n=n_rows, p=4, seed=5)
    prepared = prepare_dataset(
        Config.model_validate(_cfg(holdout_fraction="auto")), df, "target"
    )
    if expected_holdout is None:
        assert prepared.holdout_X is None
        assert prepared.holdout_y is None
        assert prepared.holdout_fraction is None
        assert len(prepared.X) == n_rows
    else:
        assert prepared.holdout_fraction == AUTO_HOLDOUT_FRACTION
        assert len(prepared.holdout_X) == expected_holdout
        assert len(prepared.X) == n_rows - expected_holdout


def test_holdout_fraction_auto_logs_on_small_data(caplog):
    """'auto' on small data says score/oof_score are optimistic and points
    to bbc_score."""
    df = _noisy_regression(n=40, p=4, seed=5)
    with caplog.at_level("INFO"):
        prepare_dataset(
            Config.model_validate(_cfg(holdout_fraction="auto")), df, "target"
        )
    assert any(
        r.levelname == "WARNING"
        and "no outer hold-out is reserved" in (m := r.getMessage())
        and "optimistic" in m
        and "bbc_score" in m
        for r in caplog.records
    )


def test_holdout_fraction_none_disables_without_log(caplog):
    """An explicit None disables the hold-out on any N without the 'auto'
    record."""
    df = _noisy_regression(n=250, p=4, seed=5)
    with caplog.at_level("INFO"):
        prepared = prepare_dataset(
            Config.model_validate(_cfg(holdout_fraction=None)), df, "target"
        )
    assert prepared.holdout_X is None
    assert len(prepared.X) == 250
    assert not any(
        "no outer hold-out is reserved" in r.getMessage() for r in caplog.records
    )


def test_holdout_score_reported_by_default(tmp_path):
    """With the default config (no holdout_fraction key) and enough rows the
    honest holdout_score is reported and the artifact is trained on all rows."""
    df = _noisy_regression(n=150, p=6, seed=9)
    raw = _cfg(model_path=str(tmp_path / "default.pkl"))
    del raw["general"]["holdout_fraction"]
    result = train_best_model(config=raw, df=df, target="target")
    assert "holdout_score" in result
    assert np.isfinite(result["holdout_score"])
    loaded = ModelTrainer.load(str(tmp_path / "default.pkl"))
    assert len(np.asarray(loaded.oof_predictions_).reshape(-1)) == len(df)


def test_holdout_rows_reserved_before_hpo(tmp_path):
    """prepare_dataset splits the outer hold-out OFF before HPO.

    The HPO train part (X/y) and the hold-out part are disjoint, the split is
    deterministic (RANDOM_STATE), and the target is absent from the features.
    """
    df = _noisy_regression(n=100, p=5, seed=11)
    cfg = Config.model_validate(_cfg(holdout_fraction=0.2))
    prepared = prepare_dataset(cfg, df, "target")

    assert len(prepared.X) == 80
    assert len(prepared.holdout_X) == 20
    assert len(prepared.y) == 80
    assert len(prepared.holdout_y) == 20
    assert prepared.holdout_fraction == 0.2
    assert "target" not in prepared.X.columns
    assert "target" not in prepared.holdout_X.columns
    # Deterministic: the same call reproduces the identical split.
    prepared2 = prepare_dataset(cfg, df, "target")
    assert list(prepared.X.index) == list(prepared2.X.index)
    assert list(prepared.holdout_X.index) == list(prepared2.holdout_X.index)
    # The rows are disjoint.
    assert set(prepared.X.index).isdisjoint(set(prepared.holdout_X.index))


def test_holdout_score_present_only_when_configured(tmp_path):
    """holdout_score is reported when general.holdout_fraction is set and
    absent otherwise."""
    df = _noisy_regression(n=120, p=6, seed=9)

    cfg_no = _cfg(model_path=str(tmp_path / "no.pkl"))
    result_no = train_best_model(config=cfg_no, df=df, target="target")
    assert "holdout_score" not in result_no

    cfg_yes = _cfg(
        holdout_fraction=0.2, model_path=str(tmp_path / "yes.pkl")
    )
    result_yes = train_best_model(config=cfg_yes, df=df, target="target")
    assert "holdout_score" in result_yes
    assert np.isfinite(result_yes["holdout_score"])


def test_holdout_score_is_winner_evaluated_on_reserved_rows(tmp_path):
    """holdout_score is the winner's comparison metric on the reserved rows.

    The winner is trained on the TRAIN part only (rows not used for HPO);
    the value matches a manual recomputation with the same preprocessing
    (ModelTrainer on the train part), so the estimate is honest rather than
    in-sample.
    """
    df = _noisy_regression(n=150, p=6, seed=13)
    cfg = _cfg(
        metric="r2",
        holdout_fraction=0.2,
        n_trials=4,
        model_path=str(tmp_path / "model.pkl"),
    )
    result = train_best_model(config=cfg, df=df, target="target")

    # Recreate the outer split exactly as prepare_dataset does.
    X = df.drop(columns="target")
    y = df["target"]
    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE
    )
    # The same pipeline the engine uses for the hold-out estimate: train on
    # the train part only, predict the reserved rows.
    ref = ModelTrainer(
        algorithm="ridge",
        hyperparams={"alpha": result["params"]["alpha"]},
        metric="r2",
        random_state=42,
    ).fit(X_tr, y_tr)
    manual = float(r2_score(y_ho, ref.predict(X_ho)))
    assert result["holdout_score"] == pytest.approx(manual, rel=1e-6)
    # The estimate is out-of-sample: a model that saw the hold-out rows in
    # training (the full-data artifact) does not systematically beat it; the
    # honest estimate is the holdout_score, not an in-sample value.
    loaded = ModelTrainer.load(str(tmp_path / "model.pkl"))
    in_sample_holdout_r2 = float(r2_score(y_ho, loaded.predict(X_ho)))
    assert result["holdout_score"] <= in_sample_holdout_r2 + 1e-9


def test_holdout_artifact_refit_on_all_data(tmp_path):
    """The saved artifact is refit on ALL data (train + hold-out).

    The OOF vector of the artifact is aligned to the full input length —
    k_fold covers every row exactly once, so with a reserved hold-out of 25%
    the artifact's ``oof_predictions_`` must have the FULL length (150), not
    the train-part length (112): the final model is still refit on all data.
    """
    df = _noisy_regression(n=150, p=6, seed=17)
    cfg = _cfg(
        metric="r2",
        holdout_fraction=0.25,
        n_trials=3,
        model_path=str(tmp_path / "model.pkl"),
    )
    result = train_best_model(config=cfg, df=df, target="target")

    loaded = ModelTrainer.load(str(tmp_path / "model.pkl"))
    oof = np.asarray(loaded.oof_predictions_, dtype=float).reshape(-1)
    # k_fold covers 100% of the fitted data: the artifact saw ALL rows, so
    # the OOF vector length equals the full input length, not the train part.
    assert len(oof) == len(df)
    # The OOF vector is positionally aligned with the concat order
    # (train part first, then the hold-out part) — the artifact was fit on
    # exactly that order (pd.concat in persist_artifact).
    X = df.drop(columns="target")
    y = df["target"]
    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=RANDOM_STATE
    )
    y_full = np.asarray(pd.concat([y_tr, y_ho]), dtype=float).reshape(-1)
    assert len(y_full) == len(oof)
    mask = np.isfinite(y_full) & np.isfinite(oof)
    assert int(mask.sum()) == len(df)
    # The reported honest OOF estimate is consistent with the artifact's
    # OOF channel over the full data.
    manual = float(r2_score(y_full[mask], oof[mask]))
    assert result["oof_score"] == pytest.approx(manual, rel=1e-9)
    # holdout_score remains the honest estimate on the reserved rows: the
    # winner trained on the train part only.
    assert "holdout_score" in result


# ──────────────────────────────────────────────────────────────────────────────
#  4. persist_artifact report keys
# ──────────────────────────────────────────────────────────────────────────────


def test_persist_artifact_adds_oof_and_trials_keys():
    """persist_artifact adds oof_score and n_trials_evaluated from the input."""
    from configurable_automl_engine.training_engine.component import persist_artifact

    df = _noisy_regression(n=60, p=4, seed=19)
    cfg = Config.model_validate(_cfg(model_path="model.pkl"))
    prepared = prepare_dataset(cfg, df, "target")

    trainer = ModelTrainer(algorithm="ridge", metric="r2", random_state=42)
    trainer.fit(prepared.X, prepared.y)
    oof_pred = np.asarray(trainer.oof_predictions_, dtype=float).reshape(-1)
    y_arr = np.asarray(prepared.y, dtype=float).reshape(-1)
    mask = np.isfinite(y_arr) & np.isfinite(oof_pred)
    expected_oof_r2 = float(r2_score(y_arr[mask], oof_pred[mask]))

    with patch(
        "configurable_automl_engine.training_engine.artifact._fit_and_save",
        return_value=trainer,
    ):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="ridge",
            final_score=0.9,
            final_params={},
            model_path_override=None,
            disqualified_algorithms={},
            n_trials_evaluated=12,
        )

    assert result["oof_score"] == pytest.approx(expected_oof_r2, rel=1e-9)
    assert result["n_trials_evaluated"] == 12


def test_persist_artifact_omits_trials_key_when_zero():
    """n_trials_evaluated=0 (unknown, legacy tuners) omits the key."""
    from configurable_automl_engine.training_engine.component import persist_artifact

    df = _noisy_regression(n=60, p=4, seed=53)
    cfg = Config.model_validate(_cfg(model_path="model.pkl"))
    prepared = prepare_dataset(cfg, df, "target")

    with patch(
        "configurable_automl_engine.training_engine.artifact._fit_and_save",
        return_value=_zero_trainer(prepared),
    ):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="ridge",
            final_score=0.9,
            final_params={},
            model_path_override=None,
            disqualified_algorithms={},
            n_trials_evaluated=0,
        )

    assert "n_trials_evaluated" not in result


def test_persist_artifact_omits_keys_when_unavailable():
    """Without OOF predictions and with an unknown trial count the keys are
    omitted (backward compatibility)."""
    from configurable_automl_engine.training_engine.component import persist_artifact

    df = _noisy_regression(n=60, p=4, seed=23)
    cfg = Config.model_validate(_cfg(model_path="model.pkl"))
    prepared = prepare_dataset(cfg, df, "target")

    fake_trainer = MagicMock()
    fake_trainer.additional_scores = {}
    fake_trainer.oof_predictions_ = None
    fake_trainer.oof_score_ = None

    with patch(
        "configurable_automl_engine.training_engine.artifact._fit_and_save",
        return_value=fake_trainer,
    ):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="ridge",
            final_score=0.9,
            final_params={},
            model_path_override=None,
            disqualified_algorithms={},
            n_trials_evaluated=None,
        )

    assert "oof_score" not in result
    assert "n_trials_evaluated" not in result
    assert "holdout_score" not in result


def test_persist_artifact_oof_shape_mismatch_omits_key():
    """An OOF vector whose length does not match the target → no oof_score."""
    from configurable_automl_engine.training_engine.component import persist_artifact

    df = _noisy_regression(n=60, p=4, seed=29)
    cfg = Config.model_validate(_cfg(model_path="model.pkl"))
    prepared = prepare_dataset(cfg, df, "target")

    fake_trainer = MagicMock()
    fake_trainer.additional_scores = {}
    fake_trainer.oof_predictions_ = np.zeros(len(prepared.y) + 5)  # misaligned
    fake_trainer.oof_score_ = 0.5

    with patch(
        "configurable_automl_engine.training_engine.artifact._fit_and_save",
        return_value=fake_trainer,
    ):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="ridge",
            final_score=0.9,
            final_params={},
            model_path_override=None,
            disqualified_algorithms={},
        )

    assert "oof_score" not in result


def test_persist_artifact_oof_fallback_to_rmse_oof():
    """When the pooled value is impossible, oof_score falls back to the
    trainer's RMSE_oof in the comparison metric semantics."""
    from configurable_automl_engine.training_engine.component import persist_artifact

    df = _noisy_regression(n=60, p=4, seed=31)
    cfg = Config.model_validate(_cfg(metric="rmse", model_path="model.pkl"))
    prepared = prepare_dataset(cfg, df, "target")

    class _NoRawScorer:
        """A scorer without the raw _score_func attributes (unpoolable)."""

        def __call__(self, estimator, Xv, yv):
            return 0.5

    fake_trainer = MagicMock()
    fake_trainer.additional_scores = {}
    fake_trainer.oof_predictions_ = np.zeros(len(prepared.y))
    fake_trainer.oof_score_ = 1.234

    with (
        patch(
            "configurable_automl_engine.training_engine.artifact._fit_and_save",
            return_value=fake_trainer,
        ),
        patch(
            "configurable_automl_engine.training_engine.artifact.get_scorer_object",
            return_value=_NoRawScorer(),
        ),
    ):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="ridge",
            final_score=0.9,
            final_params={},
            model_path_override=None,
            disqualified_algorithms={},
        )

    # rmse (error metric): the fallback value equals the natural RMSE_oof.
    assert result["oof_score"] == pytest.approx(1.234, rel=1e-9)


def test_persist_artifact_oof_score_metric_fallback_omitted():
    """For score metrics the RMSE_oof fallback is not an estimate of the
    comparison metric — the key is omitted."""
    from configurable_automl_engine.training_engine.component import persist_artifact

    df = _noisy_regression(n=60, p=4, seed=43)
    cfg = Config.model_validate(_cfg(metric="r2", model_path="model.pkl"))
    prepared = prepare_dataset(cfg, df, "target")

    class _NoRawScorer:
        def __call__(self, estimator, Xv, yv):
            return 0.5

    fake_trainer = MagicMock()
    fake_trainer.additional_scores = {}
    fake_trainer.oof_predictions_ = np.zeros(len(prepared.y))
    fake_trainer.oof_score_ = 1.234

    with (
        patch(
            "configurable_automl_engine.training_engine.artifact._fit_and_save",
            return_value=fake_trainer,
        ),
        patch(
            "configurable_automl_engine.training_engine.artifact.get_scorer_object",
            return_value=_NoRawScorer(),
        ),
    ):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="ridge",
            final_score=0.9,
            final_params={},
            model_path_override=None,
            disqualified_algorithms={},
        )

    assert "oof_score" not in result


def test_persist_artifact_oof_non_rmse_error_fallback_omitted():
    """For a non-RMSE error metric (MAE/MSE) the RMSE_oof fallback would
    silently substitute a DIFFERENT metric — the key is omitted instead."""
    from configurable_automl_engine.training_engine.component import persist_artifact

    df = _noisy_regression(n=60, p=4, seed=59)
    cfg = Config.model_validate(_cfg(metric="mae", model_path="model.pkl"))
    prepared = prepare_dataset(cfg, df, "target")

    class _NoRawScorer:
        def __call__(self, estimator, Xv, yv):
            return 0.5

    fake_trainer = MagicMock()
    fake_trainer.additional_scores = {}
    fake_trainer.oof_predictions_ = np.zeros(len(prepared.y))
    fake_trainer.oof_score_ = 1.234

    with (
        patch(
            "configurable_automl_engine.training_engine.artifact._fit_and_save",
            return_value=fake_trainer,
        ),
        patch(
            "configurable_automl_engine.training_engine.artifact.get_scorer_object",
            return_value=_NoRawScorer(),
        ),
    ):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="ridge",
            final_score=0.9,
            final_params={},
            model_path_override=None,
            disqualified_algorithms={},
        )

    assert "oof_score" not in result


def test_persist_artifact_oof_unknown_metric_warns(caplog):
    """An unresolvable comparison metric → a warning and no oof_score."""
    import logging

    from configurable_automl_engine.training_engine.component import persist_artifact

    df = _noisy_regression(n=60, p=4, seed=37)
    cfg = Config.model_validate(_cfg(model_path="model.pkl"))
    prepared = prepare_dataset(cfg, df, "target")

    fake_trainer = MagicMock()
    fake_trainer.additional_scores = {}
    fake_trainer.oof_predictions_ = np.zeros(len(prepared.y))
    fake_trainer.oof_score_ = 0.5

    with (
        patch(
            "configurable_automl_engine.training_engine.artifact._fit_and_save",
            return_value=fake_trainer,
        ),
        patch(
            "configurable_automl_engine.training_engine.artifact.get_scorer_object",
            side_effect=ValueError("unknown metric"),
        ),
        caplog.at_level(logging.WARNING, logger="training_engine"),
    ):
        result = persist_artifact(
            cfg=cfg,
            prepared=prepared,
            winner_algo="ridge",
            final_score=0.9,
            final_params={},
            model_path_override=None,
            disqualified_algorithms={},
        )

    assert "oof_score" not in result
    assert "OOF score could not be computed" in caplog.text


def test_holdout_fraction_zero_rows_nan(tmp_path):
    """An empty hold-out set → holdout_score is NaN without crashing.

    The edge case is guarded in prepare_dataset (the hold-out has at least
    one row for a valid fraction), so this covers the defensive NaN path of
    _compute_holdout_score via a hold-out with non-finite predictions.
    """
    from configurable_automl_engine.training_engine.artifact import (
        _compute_holdout_score,
    )

    df = _noisy_regression(n=60, p=4, seed=41)
    cfg = Config.model_validate(_cfg(holdout_fraction=0.2, model_path="model.pkl"))
    prepared = prepare_dataset(cfg, df, "target")

    # The trainer's predict on the hold-out returns all-NaN (a degenerate
    # model), so no finite pair remains → NaN without a crash.
    class _NaNModel:
        def predict(self, X):
            return np.full(len(X), np.nan)

    with patch(
        "configurable_automl_engine.trainer.ModelTrainer",
        return_value=_NaNModel(),
    ):
        score = _compute_holdout_score(
            cfg=cfg,
            winner_cfg=cfg.algorithms.ridge,
            prepared=prepared,
            winner_algo="ridge",
            final_params={},
        )
    assert np.isnan(score)


def test_holdout_score_exception_returns_nan(caplog):
    """A failure inside the hold-out estimation returns NaN without crashing."""
    import logging

    from configurable_automl_engine.training_engine.artifact import (
        _compute_holdout_score,
    )

    df = _noisy_regression(n=60, p=4, seed=47)
    cfg = Config.model_validate(_cfg(holdout_fraction=0.2, model_path="model.pkl"))
    prepared = prepare_dataset(cfg, df, "target")

    with (
        patch(
            "configurable_automl_engine.trainer.ModelTrainer",
            side_effect=RuntimeError("boom"),
        ),
        caplog.at_level(logging.WARNING, logger="training_engine"),
    ):
        score = _compute_holdout_score(
            cfg=cfg,
            winner_cfg=cfg.algorithms.ridge,
            prepared=prepared,
            winner_algo="ridge",
            final_params={},
        )
    assert np.isnan(score)
    assert "Hold-out score could not be computed" in caplog.text


# ──────────────────────────────────────────────────────────────────────────────
#  5. Combination scenarios (review follow-up)
# ──────────────────────────────────────────────────────────────────────────────


def test_sanity_gate_holdout_oof_e2e(tmp_path):
    """sanity_gate terminal select + holdout_fraction + oof_score work together.

    The sanity gate trains finalists on 100% of the data — the train part of
    the outer split; the terminal select hands the pre-trained trainer to
    ``persist_artifact``, which must refit the artifact on ALL data (train +
    hold-out) because a hold-out is configured. The report carries both
    ``oof_score`` (full-data OOF) and ``holdout_score``.
    """
    df = _noisy_regression(n=120, p=5, seed=23)
    cfg = _cfg(
        metric="r2",
        holdout_fraction=0.2,
        n_trials=3,
        select_strategy="sanity_gate",
        model_path=str(tmp_path / "model.pkl"),
    )
    result = train_best_model(config=cfg, df=df, target="target")

    assert "holdout_score" in result
    assert np.isfinite(result["holdout_score"])
    assert "oof_score" in result
    # Artifact refit on ALL data: the OOF vector covers the full length.
    loaded = ModelTrainer.load(str(tmp_path / "model.pkl"))
    oof = np.asarray(loaded.oof_predictions_, dtype=float).reshape(-1)
    assert len(oof) == len(df)
    X = df.drop(columns="target")
    y = df["target"]
    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE
    )
    y_full = np.asarray(pd.concat([y_tr, y_ho]), dtype=float).reshape(-1)
    mask = np.isfinite(y_full) & np.isfinite(oof)
    assert int(mask.sum()) == len(df)
    manual = float(r2_score(y_full[mask], oof[mask]))
    assert result["oof_score"] == pytest.approx(manual, rel=1e-9)


def test_holdout_score_uses_custom_trainer_module(tmp_path):
    """holdout_score honors the custom ``trainer_module`` of the winner.

    ``_compute_holdout_score`` assembles the hold-out trainer through
    ``_build_trainer`` (the same single point as the final fit), so a custom
    trainer module is used instead of the hardcoded built-in one.
    """
    from tests.dummy_trainer_module import ModelTrainer as CustomModelTrainer

    df = _noisy_regression(n=120, p=5, seed=29)
    cfg = _cfg(
        metric="r2",
        holdout_fraction=0.2,
        n_trials=3,
        trainer_module="tests.dummy_trainer_module",
        model_path=str(tmp_path / "model.pkl"),
    )

    result = train_best_model(config=cfg, df=df, target="target")

    # The hold-out estimate was computed with the CUSTOM trainer (an instance
    # was created through _build_trainer).
    assert CustomModelTrainer.instances
    assert "holdout_score" in result
    # The custom model predicts the train mean: its hold-out R² is the
    # fraction of the variance explained by the constant prediction.
    X = df.drop(columns="target")
    y = df["target"]
    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=RANDOM_STATE
    )
    manual = float(r2_score(y_ho, np.full(len(y_ho), float(np.mean(y_tr)))))
    assert result["holdout_score"] == pytest.approx(manual, rel=1e-6)


def test_holdout_small_fraction_warns(caplog):
    """A small hold-out (1 row) logs an explicit degradation WARNING.

    The estimate on a single reserved row has extremely high variance; the
    engine still reserves the row (the hold-out score remains the honest
    estimate of the winner), but the log warns so the number is not
    over-trusted.
    """
    import logging

    df = _noisy_regression(n=60, p=4, seed=31)
    # 60 rows with 0.01 → 1 row (train_test_split keeps at least 1 test row).
    cfg = Config.model_validate(_cfg(holdout_fraction=0.01, model_path="model.pkl"))

    with caplog.at_level(logging.WARNING, logger="training_engine"):
        prepared = prepare_dataset(cfg, df, "target")

    assert len(prepared.holdout_X) == 1
    assert "Outer hold-out is very small" in caplog.text


def test_holdout_extreme_fraction_rejected_with_clear_error():
    """An extreme fraction that empties the training part is rejected with a
    clear error instead of a raw sklearn ValueError."""
    df = _noisy_regression(n=100, p=4, seed=61)
    cfg = Config.model_validate(_cfg(holdout_fraction=0.999, model_path="model.pkl"))

    with pytest.raises(ValueError, match="EMPTY training part"):
        prepare_dataset(cfg, df, "target")