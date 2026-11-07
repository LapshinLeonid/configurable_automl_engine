"""Tests of ``target_transform`` plumbing through the HPO stack and the engine.

Covers:

* ``assemble_estimator`` (unit): the ``model`` step is wrapped with
  ``TransformedTargetRegressor`` for a non-``none`` method, both with and
  without a preprocessor (the ``len(steps) == 1`` bare-model branch);
* the config parsing: a valid value, an invalid value (pydantic
  ``ValidationError``) and the ``none`` default; the ``config.schema.json``
  enum;
* a short HPO run through ``tuner.optimize`` with ``log1p``: the trial
  estimator and the final model are wrapped;
* the target-domain guard over the TRANSFORMED target: ``yeo-johnson`` on a
  target with zeros passes where ``boxcox`` raises once before the first
  trial, ``gammaregressor`` is disqualified with the transformation
  mentioned in the reason;
* E2E: the full config run with ``general.target_transform: log1p`` in
  threads and processes modes — the HPO winner and the final trainer agree
  on the transformation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge

from configurable_automl_engine.common.definitions import TargetTransform
from configurable_automl_engine.common.settings import OversamplingSettings
from configurable_automl_engine.feature_selection import FeatureSelectionCfg
from configurable_automl_engine.hpo.plans import PreprocessingPlan
from configurable_automl_engine.hpo.request import HpoRequest
from configurable_automl_engine.hpo.trial_pipeline import (
    assemble_estimator,
    plan_preprocessing,
)
from configurable_automl_engine.training_engine.config_parser import AlgoCfg, Config

# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────


def _numeric_frame(n: int = 30, seed: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(3.0 * X["a"] + 4.0 + rng.normal(scale=0.1, size=n), name="y")
    return X, y


def _empty_plan() -> PreprocessingPlan:
    return PreprocessingPlan(
        X=None,
        dropped_features=[],
        categorical_features=None,
        numerical_features=None,
        preprocessor=None,
        n_features_eff=1,
        n_numeric_eff=2,
        n_features_size=1,
        isotonic_max_feature_index=1,
    )


def _positive_target_frame(rows: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"f{i}": rng.normal(size=rows) for i in range(3)})
    df["target"] = np.exp(1.0 + 0.5 * df["f0"] + rng.normal(scale=0.05, size=rows))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 1. assemble_estimator (unit)
# ──────────────────────────────────────────────────────────────────────────────


def test_assemble_estimator_wraps_model_with_preprocessor() -> None:
    X, _ = _numeric_frame()
    preprocessing = plan_preprocessing(X, HpoRequest(algo_name="ridge"))
    model = Ridge()

    pipeline = assemble_estimator(
        model,
        preprocessing,
        is_isotonic=False,
        fs_cfg=FeatureSelectionCfg(),
        oversampling=OversamplingSettings(),
        random_state=0,
        target_transform=TargetTransform.log1p,
    )

    wrapped = pipeline.named_steps["model"]
    assert isinstance(wrapped, TransformedTargetRegressor)
    # The transformer inside matches the requested method.
    assert wrapped.transformer.method == "log1p"
    # The base regressor is preserved inside the wrapper.
    assert wrapped.regressor is model


def test_assemble_estimator_bare_model_branch_wraps_without_preprocessor() -> None:
    """``len(steps) == 1``: the returned object is the WRAPPED model."""
    model = Ridge()
    bare = assemble_estimator(
        model,
        _empty_plan(),
        is_isotonic=False,
        fs_cfg=FeatureSelectionCfg(),
        oversampling=OversamplingSettings(),
        random_state=0,
        target_transform=TargetTransform.log1p,
    )
    assert isinstance(bare, TransformedTargetRegressor)
    assert bare.regressor is model
    assert bare.transformer.method == "log1p"


def test_assemble_estimator_none_keeps_bare_model() -> None:
    """Default behavior is preserved: ``none`` returns the model as is."""
    model = Ridge()
    bare = assemble_estimator(
        model,
        _empty_plan(),
        is_isotonic=False,
        fs_cfg=FeatureSelectionCfg(),
        oversampling=OversamplingSettings(),
        random_state=0,
    )
    assert bare is model


def test_assemble_estimator_wrapped_model_in_full_pipeline() -> None:
    """With preprocessor + selector + sampler the wrapped model is the last step."""
    X, _ = _numeric_frame()
    preprocessing = plan_preprocessing(X, HpoRequest(algo_name="ridge"))
    pipeline = assemble_estimator(
        Ridge(),
        preprocessing,
        is_isotonic=False,
        fs_cfg=FeatureSelectionCfg(),
        oversampling=OversamplingSettings(enable=True, multiplier=1.5),
        random_state=0,
        apply_fs=True,
        target_transform=TargetTransform.yeo_johnson,
    )
    assert [name for name, _ in pipeline.steps] == [
        "preprocessor",
        "feature_selector",
        "sampler",
        "model",
    ]
    wrapped = pipeline.named_steps["model"]
    assert isinstance(wrapped, TransformedTargetRegressor)
    assert wrapped.transformer.method == "yeo-johnson"


# ──────────────────────────────────────────────────────────────────────────────
# 2. config parsing
# ──────────────────────────────────────────────────────────────────────────────


def _config_with_target_transform(value: Any) -> dict:
    return {
        "general": {
            "target_transform": value,
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
        },
        "algorithms": {"ridge": {"enable": True}},
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("log1p", TargetTransform.log1p),
        ("boxcox", TargetTransform.boxcox),
        ("yeo-johnson", TargetTransform.yeo_johnson),
        ("none", TargetTransform.none),
    ],
)
def test_config_target_transform_valid(value, expected):
    cfg = Config.model_validate(_config_with_target_transform(value))
    assert cfg.general.target_transform is expected


def test_config_target_transform_default_is_none():
    cfg = Config.model_validate({"algorithms": {"ridge": {"enable": True}}})
    assert cfg.general.target_transform is TargetTransform.none


def test_config_target_transform_invalid_value_rejected():
    with pytest.raises(ValidationError):
        Config.model_validate(_config_with_target_transform("sqrt"))


def test_config_schema_contains_target_transform_enum():
    schema_path = Path(__file__).resolve().parents[1] / "config.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    ref = schema["properties"]["general"]["$ref"]
    assert ref.startswith("#/$defs/")
    general = schema["$defs"][ref.rsplit("/", 1)[-1]]
    field = general["properties"]["target_transform"]
    assert field["$ref"].startswith("#/$defs/")
    target = schema["$defs"][field["$ref"].rsplit("/", 1)[-1]]
    assert set(target["enum"]) == {"none", "log1p", "boxcox", "yeo-johnson"}


# ──────────────────────────────────────────────────────────────────────────────
# 3. short HPO through tuner.optimize with log1p
# ──────────────────────────────────────────────────────────────────────────────


def test_tuner_short_hpo_with_log1p_wraps_trials_and_final_model() -> None:
    from configurable_automl_engine.tuner import optimize

    df = _positive_target_frame()
    X = df.drop(columns=["target"])
    y = df["target"]
    captured: list[Any] = []

    real_assemble = assemble_estimator

    def capturing(model, preprocessing, **kwargs):
        estimator = real_assemble(model, preprocessing, **kwargs)
        captured.append(estimator)
        return estimator

    with patch(
        "configurable_automl_engine.hpo.runner.assemble_estimator",
        side_effect=capturing,
    ):
        result = optimize(
            "ridge",
            X,
            y,
            n_trials=2,
            random_state=42,
            target_transform="log1p",
        )

    assert result is not None
    # Every trial pipeline trained on the transformed target.
    assert captured
    for estimator in captured:
        wrapped = (
            estimator.named_steps["model"]
            if hasattr(estimator, "named_steps")
            else estimator
        )
        assert isinstance(wrapped, TransformedTargetRegressor)
        assert wrapped.transformer.method == "log1p"
    # The final best model (fit_best_model=True) is wrapped as well.
    final = result.model
    wrapped_final = (
        final.named_steps["model"] if hasattr(final, "named_steps") else final
    )
    assert isinstance(wrapped_final, TransformedTargetRegressor)
    assert wrapped_final.transformer.method == "log1p"


def test_tuner_log1p_domain_error_raised_once_before_first_trial() -> None:
    """A domain violation fails once before the trials, not in every fold."""
    from configurable_automl_engine.tuner import optimize

    X = _positive_target_frame().drop(columns=["target"])
    y = pd.Series(np.linspace(-5.0, -1.0, len(X)))  # min(y) < -1 → log1p invalid

    calls = {"n": 0}

    def counting(**kwargs):
        calls["n"] += 1
        return kwargs

    with (
        patch(
            "configurable_automl_engine.hpo.runner.plan_preprocessing",
            side_effect=counting,
        ),
        pytest.raises(ValueError, match="target_transform='log1p'"),
    ):
        optimize("ridge", X, y, n_trials=3, target_transform="log1p")
    # The failure happened BEFORE any preprocessing plan (i.e. before any trial).
    assert calls["n"] == 0


def test_tuner_yeo_johnson_accepts_target_with_zeros() -> None:
    from configurable_automl_engine.tuner import optimize

    X = _positive_target_frame().drop(columns=["target"])
    y = pd.Series(np.linspace(0.0, 10.0, len(X)))  # zeros → boxcox invalid
    result = optimize(
        "ridge", X, y, n_trials=2, random_state=42, target_transform="yeo-johnson"
    )
    assert result is not None
    assert result.model is not None


# ──────────────────────────────────────────────────────────────────────────────
# 4. GLM domain guard over the transformed target
# ──────────────────────────────────────────────────────────────────────────────


def _algorithms() -> dict:
    from configurable_automl_engine.training_engine.config_parser import AlgoCfg

    return {
        "gammaregressor": AlgoCfg(),
        "poissonregressor": AlgoCfg(),
        "glm": AlgoCfg(),
        "ridge": AlgoCfg(),
    }


def test_guard_yeo_johnson_checks_transformed_domain():
    """``yeo-johnson`` (standardize=True) makes the transformed target span
    negative values: the domain of every GLM is checked against y_eff, so
    ``gammaregressor`` (y > 0) and ``poissonregressor`` (y >= 0) are
    disqualified while ``glm`` (default power=0, any real y) stays."""
    from configurable_automl_engine.training_engine.hpo_phases import (
        _apply_target_domain_guard,
    )

    y = pd.Series(np.linspace(0.0, 10.0, 50))
    outcome = _apply_target_domain_guard(
        _algorithms(), y, target_transform=TargetTransform.yeo_johnson
    )
    assert set(outcome.disqualified) == {"gammaregressor", "poissonregressor"}
    assert "target_transform='yeo-johnson'" in outcome.disqualified["gammaregressor"]
    assert "glm" not in outcome.disqualified


def test_guard_log1p_passes_on_zero_target():
    """``log1p`` on a target with zeros: log(1+y) is fine for y >= 0, but the
    transformed minimum is 0, so ``gammaregressor`` (y > 0) is disqualified
    while ``poissonregressor`` (y >= 0) and ``glm`` (power=0) stay."""
    from configurable_automl_engine.training_engine.hpo_phases import (
        _apply_target_domain_guard,
    )

    y = pd.Series(np.linspace(0.0, 10.0, 50))
    outcome = _apply_target_domain_guard(
        _algorithms(), y, target_transform=TargetTransform.log1p
    )
    assert set(outcome.disqualified) == {"gammaregressor"}
    assert "target_transform='log1p'" in outcome.disqualified["gammaregressor"]


def test_guard_boxcox_raises_on_zero_target():
    """``boxcox`` requires min(y) > 0: the ValueError is raised before any
    trial (the guard runs once, before the first HPO phase)."""
    from configurable_automl_engine.training_engine.hpo_phases import (
        _apply_target_domain_guard,
    )

    y = pd.Series(np.linspace(0.0, 10.0, 50))
    with pytest.raises(ValueError, match="target_transform='boxcox'"):
        _apply_target_domain_guard(
            _algorithms(), y, target_transform=TargetTransform.boxcox
        )


def test_guard_gamma_disqualified_with_transform_note(caplog):
    """``yeo-johnson`` with standardize=True makes the transformed target
    negative somewhere → gammaregressor (y > 0) is disqualified BEFORE HPO,
    and the reason mentions the transformation."""
    from configurable_automl_engine.training_engine.hpo_phases import (
        _apply_target_domain_guard,
    )

    y = pd.Series(np.linspace(0.0, 10.0, 50))
    with caplog.at_level(logging.WARNING):
        outcome = _apply_target_domain_guard(
            _algorithms(), y, target_transform=TargetTransform.yeo_johnson
        )
    reason = outcome.disqualified.get("gammaregressor", "")
    assert "gammaregressor" in reason
    assert "y > 0" in reason
    assert "target_transform='yeo-johnson'" in reason
    # The Tweedie family with the default power=0 is valid for any real y.
    assert "glm" not in outcome.disqualified


def test_guard_none_keeps_original_behavior():
    from configurable_automl_engine.training_engine.hpo_phases import (
        _apply_target_domain_guard,
    )

    y = pd.Series([-2.0, 0.5, 1.0, 3.0])
    outcome = _apply_target_domain_guard(_algorithms(), y)
    assert set(outcome.disqualified) == {"gammaregressor", "poissonregressor"}
    # No transformation note when target_transform is none.
    assert "target_transform" not in outcome.disqualified["gammaregressor"]


def test_guard_tweedie_power_clipped_in_transformed_domain():
    """A partly-invalid Tweedie ``power`` range is CLIPPED to its valid part
    against the TRANSFORMED target: with ``log1p`` on ``y >= 0`` the
    transformed minimum is 0, so ``[1.0, 3.0]`` clips to ``[1.0, 2.0)`` and
    the algorithm stays a candidate (a WARNING is logged with the transform
    context)."""
    from configurable_automl_engine.training_engine.hpo_phases import (
        _apply_target_domain_guard,
    )

    y = pd.Series(np.linspace(0.0, 10.0, 50))
    algos = {"glm": AlgoCfg.model_validate({"hyperparameters": {"power": [1.0, 3.0]}})}
    outcome = _apply_target_domain_guard(
        algos, y, target_transform=TargetTransform.log1p
    )
    assert outcome.disqualified == {}
    power = outcome.algorithms["glm"].hyperparameters["power"].config
    assert power.low == pytest.approx(1.0)
    assert power.high < 2.0  # y_eff >= 0 forbids power >= 2 (gamma limit)
    assert power.high > 1.0  # the valid part was kept, not collapsed
    # The input config is not mutated.
    original = algos["glm"].hyperparameters["power"].config
    assert (original.low, original.high) == (1.0, 3.0)


def test_guard_tweedie_power_disqualified_in_transformed_domain():
    """A fully invalid Tweedie ``power`` range under the transformation is
    disqualified with the transform note: ``[2.5, 3.0]`` supports nothing
    when ``y_eff`` (log1p on ``y >= 0``) has min 0."""
    from configurable_automl_engine.training_engine.hpo_phases import (
        _apply_target_domain_guard,
    )

    y = pd.Series(np.linspace(0.0, 10.0, 50))
    algos = {"glm": AlgoCfg.model_validate({"hyperparameters": {"power": [2.5, 3.0]}})}
    outcome = _apply_target_domain_guard(
        algos, y, target_transform=TargetTransform.log1p
    )
    assert set(outcome.disqualified) == {"glm"}
    assert "target_transform='log1p'" in outcome.disqualified["glm"]


# ──────────────────────────────────────────────────────────────────────────────
# 5. E2E: full config with general.target_transform
# ──────────────────────────────────────────────────────────────────────────────


def _e2e_config(model_path: str, parallel_mode: str = "threads") -> dict:
    return {
        "general": {
            "comparison_metric": "r2",
            "holdout_fraction": None,
            "target_transform": "log1p",
            "random_state": 42,
            "parallel_mode": parallel_mode,
            "phases": [
                {"name": "coarse", "n_trials": 2, "action": "all_algorithms"},
                {
                    "name": "final",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "cv_best",
                    "output": "winner",
                },
            ],
            "path_to_model": model_path,
        },
        "algorithms": {
            "ridge": {"enable": True},
            "random_forest": {
                "enable": True,
                "hyperparameters": {"n_estimators": [5, 10]},
            },
        },
        "oversampling": {"enable": False},
    }


def _e2e_df(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = pd.DataFrame({f"f{i}": rng.normal(size=rows) for i in range(3)})
    df["target"] = np.exp(2.0 + 0.8 * df["f0"] + rng.normal(scale=0.05, size=rows))
    return df


def _assert_transformed_trainer(trainer: Any) -> None:
    steps = {name: step for name, step in trainer.pipeline.steps}
    assert isinstance(steps["model"], TransformedTargetRegressor)
    assert steps["model"].transformer.method == "log1p"


def test_e2e_target_transform_threads(tmp_path: Path) -> None:
    from configurable_automl_engine.training_engine.component import (
        train_best_model,
    )

    model_path = tmp_path / "model.pkl"
    result = train_best_model(
        config=_e2e_config(str(model_path)),
        df=_e2e_df(),
        target="target",
        model_path_override=model_path,
    )
    assert result["algorithm"] in ("ridge", "random_forest")
    # The saved artifact trains on the transformed target.
    import joblib

    trainer = joblib.load(model_path)
    _assert_transformed_trainer(trainer)


def test_e2e_target_transform_processes(tmp_path: Path) -> None:
    """``parallel_mode='processes'`` with a transformed target: everything
    (the request, the wrapped pipelines) is picklable end-to-end."""
    from configurable_automl_engine.training_engine.component import (
        train_best_model,
    )

    model_path = tmp_path / "model.pkl"
    result = train_best_model(
        config=_e2e_config(str(model_path), parallel_mode="processes"),
        df=_e2e_df(),
        target="target",
        model_path_override=model_path,
    )
    assert result["algorithm"] in ("ridge", "random_forest")
    import joblib

    trainer = joblib.load(model_path)
    _assert_transformed_trainer(trainer)


def test_e2e_hpo_and_final_model_consistent(tmp_path: Path) -> None:
    """HPO trials and the final trainer share the same target_transform."""
    from configurable_automl_engine.hpo.runner import run_optimization
    from configurable_automl_engine.training_engine.artifact import _build_trainer
    from configurable_automl_engine.training_engine.config_parser import AlgoCfg

    df = _e2e_df()
    y = df.pop("target")

    requests: list[HpoRequest] = []

    real_run = run_optimization

    def capturing(X, y, request):
        requests.append(request)
        return real_run(X, y, request)

    cfg = Config.model_validate(_e2e_config(str(tmp_path / "model.pkl")))
    prepared_y = y.reset_index(drop=True)
    algo_cfg = AlgoCfg()
    trainer = _build_trainer(
        "ridge",
        algo_cfg,
        {"alpha": 1.0},
        cfg,
        random_state=42,
        target_transform=cfg.general.target_transform,
    )
    assert isinstance(trainer.settings.target_transform, TargetTransform)
    assert trainer.settings.target_transform is TargetTransform.log1p

    with patch(
        "configurable_automl_engine.training_engine.hpo_phases.run_optimization",
        side_effect=capturing,
    ):
        from configurable_automl_engine.training_engine.data_prep import (
            prepare_dataset,
        )
        from configurable_automl_engine.training_engine.component import (
            execute_phases,
        )

        prepared = prepare_dataset(cfg, df.assign(target=prepared_y), "target")
        execute_phases(cfg, prepared)

    assert requests
    for request in requests:
        assert request.target_transform is TargetTransform.log1p


def _sanity_gate_config(model_path: str) -> dict:
    """Config with a sanity_gate select: the winner is trained by
    ``_train_finalists`` and reused as ``pre_trained_trainer``."""
    return {
        "general": {
            "comparison_metric": "r2",
            "holdout_fraction": None,
            "target_transform": "log1p",
            "random_state": 42,
            "phases": [
                {"name": "coarse", "n_trials": 2, "action": "all_algorithms"},
                {
                    "name": "gate",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "sanity_gate",
                    "output": "winner",
                },
            ],
            "path_to_model": model_path,
        },
        "algorithms": {
            "ridge": {"enable": True},
            "random_forest": {
                "enable": True,
                "hyperparameters": {"n_estimators": [5, 10]},
            },
        },
        "oversampling": {"enable": False},
    }


def test_e2e_sanity_gate_finalist_trains_with_target_transform(
    tmp_path: Path,
) -> None:
    """Blocker regression: the sanity_gate finalists (``_fit_finalist`` →
    ``_build_trainer``) receive ``general.target_transform``. With
    ``output: winner`` the finalist trainer IS the final artifact
    (``pre_trained_trainer``), so the saved model must be wrapped."""
    from configurable_automl_engine.training_engine.component import (
        train_best_model,
    )

    model_path = tmp_path / "model.pkl"
    result = train_best_model(
        config=_sanity_gate_config(str(model_path)),
        df=_e2e_df(),
        target="target",
        model_path_override=model_path,
    )
    assert result["algorithm"] in ("ridge", "random_forest")
    import joblib

    trainer = joblib.load(model_path)
    # The artifact came from the finalist path and IS transformed.
    _assert_transformed_trainer(trainer)
    assert trainer.settings.target_transform is TargetTransform.log1p


def test_e2e_sanity_gate_finalist_hpo_scale_consistent(tmp_path: Path) -> None:
    """The finalist's fitted transformer matches the HPO scale: the wrapped
    model's transformer, fitted on the full data, reproduces the train-range
    statistics (a proxy for "the hyperparameters were selected in the same
    scale the artifact predicts in")."""
    from configurable_automl_engine.training_engine.component import (
        train_best_model,
    )

    model_path = tmp_path / "model.pkl"
    df = _e2e_df()
    train_best_model(
        config=_sanity_gate_config(str(model_path)),
        df=df,
        target="target",
        model_path_override=model_path,
    )
    import joblib
    import numpy as np

    trainer = joblib.load(model_path)
    wrapped = dict(trainer.pipeline.steps)["model"]
    assert isinstance(wrapped, TransformedTargetRegressor)
    fitted = getattr(wrapped, "transformer_", wrapped.transformer)
    y = df["target"].to_numpy()
    expected_t = np.log1p(y)
    # The finalist transformer was fitted on the full dataset (the same rows
    # the HPO winner's hyperparameters were selected against).
    assert fitted.t_min_ == pytest.approx(expected_t.min())
    assert fitted.t_max_ == pytest.approx(expected_t.max())
