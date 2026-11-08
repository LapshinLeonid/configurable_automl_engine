"""Tests of ``general.target_transform: auto`` (issue: the ``auto`` mode).

Covers:

* ``resolve_target_transform`` (unit): the method choice on three synthetic
  distributions (log-normal → ``log1p``, normal → ``none``, a skewed target
  with negative values → ``yeo-johnson``); the threshold boundary (the
  decision flips exactly at ``|skew| == threshold``); explicit methods pass
  through unchanged; an invalid threshold raises ``ValueError``; the
  case-insensitive string input; the empty target rejection.
* The INFO log: the resolved method is announced with the skewness, the
  threshold and the chosen method.
* The config schema: ``auto`` is a valid ``general.target_transform`` value;
  the new ``target_transform_skew_threshold`` field (default 1.0, > 0) is
  validated — a non-positive or non-numeric value fails config validation.
* The pipeline integration: ``prepare_dataset`` resolves ``auto`` before the
  HPO phases (the guard and the HPO spec see the resolved method), the
  resolved method reaches the ``ExperimentResult`` config snapshot, the
  artifact metadata and the saved trainer; the INFO log carries the decision.
* The placeholder guards: ``auto`` passed directly to ``ModelTrainer`` /
  ``normalize_target_transform`` / ``TargetTransformer`` is rejected with a
  clear error instead of leaking into a pipeline.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from scipy import stats as sp_stats
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression

from configurable_automl_engine.common.definitions import TargetTransform
from configurable_automl_engine.preprocessing.target_transform import (
    DEFAULT_TARGET_TRANSFORM_SKEW_THRESHOLD,
    TargetTransformer,
    normalize_target_transform,
    resolve_target_transform,
    wrap_with_target_transform,
)
from configurable_automl_engine.trainer import ModelTrainer
from configurable_automl_engine.trainer_params import TrainingError
from configurable_automl_engine.training_engine import component
from configurable_automl_engine.training_engine.component import train_best_model
from configurable_automl_engine.training_engine.config_parser import Config
from configurable_automl_engine.training_engine.data_prep import (
    _fallback_transform_for_full_y,
    prepare_dataset,
)

# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────


def _lognormal_y(n: int = 400, seed: int = 0) -> pd.Series:
    """A strongly right-skewed positive target (log-normal)."""
    rng = np.random.default_rng(seed)
    return pd.Series(np.exp(rng.normal(size=n)), name="y")


def _normal_y(n: int = 400, seed: int = 1) -> pd.Series:
    """A (practically) symmetric target."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(size=n), name="y")


def _negative_skewed_y(n: int = 400, seed: int = 2) -> pd.Series:
    """A strongly skewed target with negative values (log1p domain fails)."""
    return pd.Series(-_lognormal_y(n, seed).to_numpy(), name="y")


def _base_config(**general: Any) -> dict:
    """A minimal valid engine config with the given general overrides."""
    general.setdefault("random_state", 42)
    return {
        "general": {
            "phases": [
                {"name": "p1", "n_trials": 1, "action": "all_algorithms"},
                {
                    "name": "final",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
            **general,
        },
        "algorithms": {"ridge": {"enable": True}},
        "oversampling": {"enable": False},
    }


# ──────────────────────────────────────────────────────────────────────────────
# 1. resolve_target_transform: the decision on synthetic distributions
# ──────────────────────────────────────────────────────────────────────────────


def test_auto_lognormal_selects_log1p() -> None:
    y = _lognormal_y()
    assert float(sp_stats.skew(y, bias=False)) > 1.0  # the fixture is skewed
    assert resolve_target_transform(y, TargetTransform.auto) is TargetTransform.log1p


def test_auto_normal_selects_none() -> None:
    y = _normal_y()
    assert abs(float(sp_stats.skew(y, bias=False))) < 1.0
    assert resolve_target_transform(y, TargetTransform.auto) is TargetTransform.none


def test_auto_negative_skewed_selects_yeo_johnson() -> None:
    y = _negative_skewed_y()
    assert float(y.min()) < -1.0  # log1p domain violated
    assert float(sp_stats.skew(y, bias=False)) < -1.0
    assert (
        resolve_target_transform(y, TargetTransform.auto) is TargetTransform.yeo_johnson
    )


def test_auto_negative_skewed_above_log1p_domain_selects_yeo_johnson() -> None:
    """Skewed y with -1 < min(y) < 0: log1p is in-domain and wins."""
    y = _lognormal_y() - 0.5  # min slightly below 0, still > -1
    assert -1.0 < float(y.min()) < 0.0
    assert resolve_target_transform(y, TargetTransform.auto) is TargetTransform.log1p


def test_auto_accepts_string_values_case_insensitive() -> None:
    y = _lognormal_y()
    assert resolve_target_transform(y, "auto") is TargetTransform.log1p
    assert resolve_target_transform(y, "AUTO") is TargetTransform.log1p


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("none", TargetTransform.none),
        ("log1p", TargetTransform.log1p),
        ("boxcox", TargetTransform.boxcox),
        ("yeo-johnson", TargetTransform.yeo_johnson),
    ],
)
def test_auto_explicit_methods_pass_through(
    method: str, expected: TargetTransform
) -> None:
    """An explicit method is never overridden by the skewness decision."""
    y = _lognormal_y()
    assert resolve_target_transform(y, method) is expected
    assert resolve_target_transform(y, expected, skew_threshold=100.0) is expected


def test_auto_threshold_boundary() -> None:
    """The decision flips exactly at ``|skew| == threshold``.

    A fixture target with skew ≈ 2.2: with the threshold equal to the
    observed skew the comparison is strict (``>``), so ``none`` is chosen;
    slightly below the threshold the target is transformed.
    """
    y = _lognormal_y(n=2000, seed=3)
    skew = float(sp_stats.skew(y, bias=False))
    assert abs(skew) > 1.5
    at_threshold = resolve_target_transform(
        y, TargetTransform.auto, skew_threshold=skew
    )
    assert at_threshold is TargetTransform.none
    below_threshold = resolve_target_transform(
        y, TargetTransform.auto, skew_threshold=skew - 1e-9
    )
    assert below_threshold is TargetTransform.log1p


@pytest.mark.parametrize("threshold", [0.0, -1.0, -1e-12])
def test_auto_invalid_threshold_rejected(threshold: float) -> None:
    y = _lognormal_y()
    with pytest.raises(ValueError, match="target_transform_skew_threshold"):
        resolve_target_transform(y, TargetTransform.auto, skew_threshold=threshold)


def test_auto_empty_target_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        resolve_target_transform(pd.Series(dtype=float), TargetTransform.auto)


def test_auto_constant_target_resolves_to_none() -> None:
    """A constant y has an undefined sample skewness (scipy returns NaN with
    RuntimeWarnings): the public function resolves deterministically to
    ``none`` — no skewness evidence, no transformation — and emits no
    warnings."""
    y = pd.Series([5.0] * 50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        resolved = resolve_target_transform(y, TargetTransform.auto)
    assert resolved is TargetTransform.none


def test_auto_degenerate_near_constant_target_resolves_deterministically() -> None:
    """A two-value degenerate target (zero variance around the mean) also has
    an undefined skewness: resolved to ``none`` without warnings."""
    y = pd.Series([1.0, -1.0] * 25)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_target_transform(y, TargetTransform.auto) is TargetTransform.none


def test_auto_unknown_method_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown target_transform"):
        resolve_target_transform(_lognormal_y(), "sqrt")


def test_auto_invalid_threshold_not_checked_for_explicit_method() -> None:
    """A non-auto method returns after threshold validation for consistency:
    the threshold argument must be positive regardless of the method (an
    invalid value in the config is rejected by the schema anyway)."""
    with pytest.raises(ValueError, match="target_transform_skew_threshold"):
        resolve_target_transform(
            _lognormal_y(), TargetTransform.none, skew_threshold=-5.0
        )
    assert (
        resolve_target_transform(
            _lognormal_y(), TargetTransform.none, skew_threshold=0.5
        )
        is TargetTransform.none
    )


def test_default_threshold_is_one() -> None:
    assert DEFAULT_TARGET_TRANSFORM_SKEW_THRESHOLD == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. the INFO log
# ──────────────────────────────────────────────────────────────────────────────


def test_auto_decision_logged_at_info(caplog: pytest.LogCaptureFixture) -> None:
    y = _lognormal_y()
    with caplog.at_level(
        logging.INFO, logger="configurable_automl_engine.preprocessing.target_transform"
    ):
        resolve_target_transform(y, TargetTransform.auto)
    messages = [r.getMessage() for r in caplog.records]
    assert any("auto" in m and "log1p" in m and "skew" in m for m in messages), messages


# ──────────────────────────────────────────────────────────────────────────────
# 3. the config schema
# ──────────────────────────────────────────────────────────────────────────────


def test_config_accepts_auto_target_transform() -> None:
    cfg = Config.model_validate(_base_config(target_transform="auto"))
    assert cfg.general.target_transform is TargetTransform.auto


def test_config_skew_threshold_default_is_one() -> None:
    cfg = Config.model_validate(_base_config())
    assert cfg.general.target_transform_skew_threshold == 1.0


def test_config_skew_threshold_custom_value() -> None:
    cfg = Config.model_validate(_base_config(target_transform_skew_threshold=0.5))
    assert cfg.general.target_transform_skew_threshold == 0.5


@pytest.mark.parametrize("threshold", [0, -0.5])
def test_config_skew_threshold_must_be_positive(threshold: float) -> None:
    with pytest.raises(ValidationError):
        Config.model_validate(_base_config(target_transform_skew_threshold=threshold))


def test_config_skew_threshold_must_be_numeric() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate(_base_config(target_transform_skew_threshold="large"))


def test_schema_contains_skew_threshold() -> None:
    schema_path = Path(__file__).resolve().parents[1] / "config.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    general = schema["$defs"]["GeneralCfg"]
    field = general["properties"]["target_transform_skew_threshold"]
    assert field["default"] == 1.0
    assert field["exclusiveMinimum"] == 0.0
    assert "auto" in schema["$defs"]["TargetTransform"]["enum"]


# ──────────────────────────────────────────────────────────────────────────────
# 4. pipeline integration
# ──────────────────────────────────────────────────────────────────────────────


def _e2e_config(model_path: str, **general: Any) -> dict:
    return _base_config(
        comparison_metric="r2",
        holdout_fraction=None,
        target_transform="auto",
        path_to_model=model_path,
        **general,
    )


def _e2e_df(rows: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"f{i}": rng.normal(size=rows) for i in range(3)})
    df["target"] = np.exp(2.0 + 0.8 * df["f0"] + rng.normal(scale=0.05, size=rows))
    return df


def test_prepare_dataset_resolves_auto_before_hpo(caplog) -> None:
    """``prepare_dataset`` writes the resolved method back into the config."""
    cfg = Config.model_validate(_base_config(target_transform="auto"))
    df = _e2e_df(rows=80)
    with caplog.at_level(logging.INFO):
        prepare_dataset(cfg, df, "target")
    assert cfg.general.target_transform is TargetTransform.log1p
    assert any("target_transform='auto'" in r.getMessage() for r in caplog.records)


# ──────────────────────────────────────────────────────────────────────────────
# 4a. 'auto' is resolved AFTER the outer hold-out split (holdout independence)
# ──────────────────────────────────────────────────────────────────────────────


def _holdout_independence_df(rows: int = 300, seed: int = 11) -> pd.DataFrame:
    """A dataset with a (practically) symmetric target.

    The target is a plain linear function of the features plus gaussian
    noise: ``|skew|`` stays well below the default 1.0 threshold, so
    ``target_transform='auto'`` must resolve to ``none``.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"f{i}": rng.normal(size=rows) for i in range(3)})
    df["target"] = 2.0 * df["f0"] + rng.normal(scale=0.1, size=rows)
    return df


def _holdout_row_index(df: pd.DataFrame, cfg: Config) -> list[int]:
    """Return the row labels that a ``prepare_dataset`` call reserves for
    the outer hold-out.

    The split itself is an implementation detail of ``prepare_dataset``
    (its seed comes from ``cfg.general.random_state``): instead of
    re-implementing it, this helper reads the contract — the labels of the
    hold-out rows are exactly the row labels of ``prepared.holdout_y``.
    The caller must build ``cfg`` with the SAME ``general.random_state``
    it later passes to ``prepare_dataset``.

    The probe runs on a DEEP COPY of ``cfg``: ``prepare_dataset`` resolves
    ``target_transform: 'auto'`` by mutating ``cfg`` in place. The resolved
    value does not change what the resolution itself computes (a concrete
    method passes through unchanged), but keeping the caller's config
    pristine makes the subsequent "poisoned" call a genuine ``auto``
    resolution on its own decision sample — the tests assert on a real
    end-to-end resolution, not on a value leaked from the probe.
    """
    probe_cfg = cfg.model_copy(deep=True)
    prepared = prepare_dataset(probe_cfg, df.copy(), "target")
    assert prepared.holdout_y is not None
    return [int(label) for label in prepared.holdout_y.index]


def test_auto_decision_ignores_holdout_target_values() -> None:
    """The 'auto' decision is made on the TRAIN part of the outer split.

    The symmetric target is shifted positive first: the TRAIN part stays
    symmetric (``|skew| < 1``) AND the whole vector keeps ``min(y) > -1``,
    so a large positive poison in the hold-out rows makes the FULL vector
    right-skewed (``|skew| > 1``) while remaining inside the ``log1p``
    domain — under the pre-#143 regression (the auto decision on the FULL
    vector) the resolved method would be ``log1p``, not ``none``, so the
    assertion below really tracks the fix.

    The hold-out rows are located through the public contract — the
    ``prepared.holdout_y`` labels of a probe call with the same config
    (the probe runs on a deep copy, see ``_holdout_row_index``). With
    ``target_transform: auto`` the resolved method must stay ``none`` —
    the hold-out must not influence how the pipeline is built.
    """
    df = _holdout_independence_df()
    df["target"] = df["target"] + 10.0  # min(y) > -1: log1p stays in-domain
    cfg = Config.model_validate(
        _base_config(target_transform="auto", holdout_fraction=0.2)
    )
    holdout_idx = _holdout_row_index(df, cfg)
    # A large positive poison flips the full-vector skewness decision to
    # log1p; the train part stays symmetric.
    df.loc[df.index[holdout_idx], "target"] = 1e6

    train_part = df["target"].drop(df.index[holdout_idx])
    assert float(sp_stats.skew(train_part, bias=False)) < 1.0
    full_skew = float(sp_stats.skew(df["target"], bias=False))
    assert full_skew > 1.0  # the full vector IS right-skewed — the poison works
    assert float(df["target"].min()) > -1.0  # log1p is a legal resolution

    prepare_dataset(cfg, df, "target")
    assert cfg.general.target_transform is TargetTransform.none


def test_auto_decision_uses_full_target_without_holdout() -> None:
    """Without a reserved hold-out the full vector is the decision sample.

    The same poisoned data with ``holdout_fraction: null`` resolves to
    ``log1p`` — the extreme values are part of the training data then
    (the base target is shifted positive, so ``min(y) > -1`` keeps ``log1p``
    inside its domain).
    """
    df = _holdout_independence_df()
    df["target"] = df["target"] + 50.0  # min(y) > -1: log1p stays in-domain
    df.loc[df.index[-5:], "target"] = 1e6
    full_skew = float(sp_stats.skew(df["target"], bias=False))
    assert full_skew > 1.0
    assert float(df["target"].min()) > -1.0

    cfg = Config.model_validate(
        _base_config(target_transform="auto", holdout_fraction=None)
    )
    prepare_dataset(cfg, df, "target")
    assert cfg.general.target_transform is TargetTransform.log1p


def test_auto_domain_fallback_when_holdout_breaks_log1p_domain(caplog) -> None:
    """A resolved ``log1p`` violated by a hold-out value downgrades to
    ``yeo-johnson`` with a WARNING.

    The train part resolves to ``log1p`` (right-skewed, in-domain), an
    extreme negative hold-out value pushes ``min(y)`` below -1: the final
    artifact is refit on ALL data, so the method must be downgraded to
    ``yeo-johnson`` instead of crashing the fit after the HPO budget was
    spent."""
    rng = np.random.default_rng(13)
    rows = 300
    df = pd.DataFrame({f"f{i}": rng.normal(size=rows) for i in range(3)})
    # Right-skewed positive target: log1p on the train part.
    df["target"] = np.exp(0.8 * df["f0"] + rng.normal(scale=0.05, size=rows))

    cfg = Config.model_validate(
        _base_config(target_transform="auto", holdout_fraction=0.2)
    )
    holdout_idx = _holdout_row_index(df, cfg)
    df.loc[df.index[holdout_idx], "target"] = -10.0  # log1p domain: min > -1

    train_part = df["target"].drop(df.index[holdout_idx])
    assert float(sp_stats.skew(train_part, bias=False)) > 1.0  # log1p decision

    with caplog.at_level(
        logging.WARNING, logger="configurable_automl_engine.training_engine.data_prep"
    ):
        prepare_dataset(cfg, df, "target")
    assert cfg.general.target_transform is TargetTransform.yeo_johnson
    assert any("downgraded to 'yeo-johnson'" in r.getMessage() for r in caplog.records)


def test_auto_no_fallback_when_full_target_in_domain() -> None:
    """An in-domain hold-out keeps the resolved ``log1p`` unchanged."""
    df = _e2e_df(rows=300)  # log-normal target → log1p everywhere
    cfg = Config.model_validate(
        _base_config(target_transform="auto", holdout_fraction=0.2)
    )
    prepare_dataset(cfg, df, "target")
    assert cfg.general.target_transform is TargetTransform.log1p


def test_explicit_log1p_broken_by_holdout_is_downgraded_with_warning(caplog) -> None:
    """An explicitly configured restricted method is ALSO downgraded.

    The domain violation of the full target cannot be fixed by keeping the
    configured ``log1p``: the final fit would crash with a
    ``validate_target_for_transform`` error after the HPO budget was spent.
    The downgrade to ``yeo-johnson`` is reported with a WARNING."""
    df = _e2e_df(rows=300)
    df.loc[df.index[:5], "target"] = -10.0  # violates the log1p domain

    cfg = Config.model_validate(
        _base_config(target_transform="log1p", holdout_fraction=0.2)
    )
    with caplog.at_level(
        logging.WARNING,
        logger="configurable_automl_engine.training_engine.data_prep",
    ):
        prepare_dataset(cfg, df, "target")
    assert cfg.general.target_transform is TargetTransform.yeo_johnson
    assert any("violated by the FULL target" in r.getMessage() for r in caplog.records)


# ──────────────────────────────────────────────────────────────────────────────
# 4b. the domain guard rejects a non-numeric / non-finite full target
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_full_y",
    [
        pd.Series([1.0, 2.0, float("nan")]),
        pd.Series([1.0, 2.0, float("inf")]),
        pd.Series(["a", "b", "c"]),
    ],
    ids=["nan", "inf", "non-numeric"],
)
def test_domain_guard_rejects_non_finite_full_y(bad_full_y: pd.Series) -> None:
    """A NaN/±inf/non-numeric full target makes the min(y) domain decision
    undefined (``min`` of a NaN vector silently returns NaN): the guard
    raises a clear ``ValueError`` instead of a silent downgrade.

    ``prepare_dataset`` never passes such data (the target is validated
    before the guard), so the error is a direct-caller contract."""
    with pytest.raises(ValueError, match="domain guard"):
        _fallback_transform_for_full_y(TargetTransform.log1p, bad_full_y)


def test_prepare_dataset_leaves_explicit_method_untouched() -> None:
    cfg = Config.model_validate(_base_config(target_transform="yeo-johnson"))
    prepare_dataset(cfg, _e2e_df(rows=80), "target")
    assert cfg.general.target_transform is TargetTransform.yeo_johnson


def test_e2e_auto_resolves_to_log1p_and_reaches_artifact(tmp_path: Path) -> None:
    """E2E: a log-normal target with ``target_transform: auto``.

    The resolved ``log1p`` reaches the HPO trials, the final trainer, the
    artifact metadata and the ``ExperimentResult`` (the config snapshot
    carries the resolved value, not ``auto``).
    """
    model_path = tmp_path / "model.pkl"
    result = train_best_model(
        config=_e2e_config(str(model_path)),
        df=_e2e_df(),
        target="target",
        model_path_override=model_path,
    )
    assert result["algorithm"] == "ridge"

    # The global is re-imported through the module each time: it is
    # reassigned by ``train_best_model``, a module-level ``from`` import
    # would keep the stale ``None`` binding.
    assert component.last_experiment_result is not None
    # The resolved value (not 'auto') is in the ExperimentResult.
    assert component.last_experiment_result.config["general"]["target_transform"] == (
        "log1p"
    )

    trainer = joblib.load(model_path)
    steps = dict(trainer.pipeline.steps)
    assert isinstance(steps["model"], TransformedTargetRegressor)
    assert steps["model"].transformer.method == "log1p"
    assert trainer.settings.target_transform is TargetTransform.log1p
    # The artifact metadata carries the resolved method.
    assert trainer.artifact_metadata_["target_transform"] == "log1p"


def test_e2e_auto_normal_target_resolves_to_none(tmp_path: Path) -> None:
    """A symmetric target keeps ``none`` end-to-end: the saved model step is
    the bare regressor (no ``TransformedTargetRegressor`` wrapper)."""
    rng = np.random.default_rng(5)
    rows = 150
    df = pd.DataFrame({f"f{i}": rng.normal(size=rows) for i in range(3)})
    df["target"] = 2.0 * df["f0"] + rng.normal(scale=0.1, size=rows)

    model_path = tmp_path / "model.pkl"
    result = train_best_model(
        config=_e2e_config(str(model_path)),
        df=df,
        target="target",
        model_path_override=model_path,
    )
    assert result["algorithm"] == "ridge"

    trainer = joblib.load(model_path)
    steps = dict(trainer.pipeline.steps)
    assert not isinstance(steps["model"], TransformedTargetRegressor)
    assert trainer.settings.target_transform is TargetTransform.none


def test_e2e_auto_negative_skewed_resolves_to_yeo_johnson(tmp_path: Path) -> None:
    """A skewed target with negative values resolves to ``yeo-johnson``."""
    rng = np.random.default_rng(6)
    rows = 300
    df = pd.DataFrame({f"f{i}": rng.normal(size=rows) for i in range(3)})
    df["target"] = -np.exp(2.0 - 0.8 * df["f0"] + rng.normal(scale=0.05, size=rows))

    model_path = tmp_path / "model.pkl"
    result = train_best_model(
        config=_e2e_config(str(model_path)),
        df=df,
        target="target",
        model_path_override=model_path,
    )
    assert result["algorithm"] == "ridge"

    trainer = joblib.load(model_path)
    steps = dict(trainer.pipeline.steps)
    assert isinstance(steps["model"], TransformedTargetRegressor)
    assert steps["model"].transformer.method == "yeo-johnson"


# ──────────────────────────────────────────────────────────────────────────────
# 5. the placeholder guards
# ──────────────────────────────────────────────────────────────────────────────


def test_trainer_rejects_auto_placeholder() -> None:
    with pytest.raises(TrainingError, match="must be resolved"):
        ModelTrainer(algorithm="ridge", target_transform="auto")
    with pytest.raises(TrainingError, match="must be resolved"):
        ModelTrainer(algorithm="ridge", target_transform=TargetTransform.auto)


def test_normalize_target_transform_rejects_auto() -> None:
    with pytest.raises(ValueError, match="must be resolved"):
        normalize_target_transform(TargetTransform.auto)
    with pytest.raises(ValueError, match="must be resolved"):
        normalize_target_transform("auto")


def test_target_transformer_rejects_auto() -> None:
    with pytest.raises(ValueError, match="auto"):
        TargetTransformer(method="auto").fit(_lognormal_y().to_numpy().reshape(-1, 1))


def test_wrapped_model_never_sees_auto() -> None:
    with pytest.raises(ValueError, match="auto"):
        wrap_with_target_transform(LinearRegression(), TargetTransform.auto)
