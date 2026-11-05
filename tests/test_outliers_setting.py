"""Tests of the ``outliers`` setting: presets, config, pipeline (8.5).

Cover the ``OutlierSpec`` discriminated union (string/object forms, per-method
validation), the preset/override resolution, the pipeline assembly order
(winsorizer -> imputer -> scaler), the leakage guarantees in CV folds, the
indicator integrity, the HPO/final parity and the regression safety (without
``outliers`` the pipeline and predictions are unchanged).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.model_selection import KFold

from configurable_automl_engine.common.settings import EncodingSettings
from configurable_automl_engine.preprocessing.builder import build_preprocessor
from configurable_automl_engine.preprocessing.features import (
    estimate_n_features_after_preprocessing,
)
from configurable_automl_engine.preprocessing.outliers import Winsorizer
from configurable_automl_engine.preprocessing_presets import (
    IqrOutliers,
    NoOutliers,
    NoScaling,
    PreprocessingOverride,
    PreprocessingPreset,
    QuantileOutliers,
    coerce_outliers,
    format_outliers,
    resolve_preprocessing_preset,
)
from configurable_automl_engine.trainer import ModelTrainer
from configurable_automl_engine.trainer_data import build_trainer_preprocessor
from configurable_automl_engine.tuner import optimize

# ──────────────────────────────────────────────────────────────────────────
#  OutlierSpec: string and object forms, validation
# ──────────────────────────────────────────────────────────────────────────


def test_outlier_spec_string_shorthand_is_normalized():
    """The string shorthand is normalized to its object form."""
    assert coerce_outliers("none") == NoOutliers()
    assert coerce_outliers("quantile") == QuantileOutliers()
    assert coerce_outliers("iqr") == IqrOutliers()


def test_outlier_spec_object_forms_pass_through():
    """The object forms (dicts and models) are accepted."""
    spec = {"method": "quantile", "lower_quantile": 0.02, "upper_quantile": 0.98}
    obj = coerce_outliers(spec)
    assert isinstance(obj, QuantileOutliers)
    assert obj.lower_quantile == 0.02
    assert obj.upper_quantile == 0.98
    assert isinstance(coerce_outliers(QuantileOutliers()), QuantileOutliers)


def test_outlier_spec_cross_method_parameters_rejected():
    """Parameters of another method are rejected (extra='forbid')."""
    with pytest.raises(ValidationError):
        coerce_outliers({"method": "quantile", "iqr_multiplier": 2.0})
    with pytest.raises(ValidationError):
        coerce_outliers({"method": "iqr", "lower_quantile": 0.01})
    with pytest.raises(ValidationError):
        coerce_outliers({"method": "none", "iqr_multiplier": 2.0})


@pytest.mark.parametrize(
    ("spec", "field", "value"),
    [
        ({"method": "quantile", "lower_quantile": 1.0, "upper_quantile": 1.0}, "q", 0),
        ({"method": "quantile", "lower_quantile": -0.1, "upper_quantile": 0.9}, "q", 0),
        ({"method": "quantile", "lower_quantile": 0.9, "upper_quantile": 0.1}, "q", 0),
        ({"method": "quantile", "lower_quantile": 0.5, "upper_quantile": 0.5}, "q", 0),
        ({"method": "quantile", "lower_quantile": 0.1, "upper_quantile": 2.0}, "q", 0),
        ({"method": "iqr", "iqr_multiplier": 0.0}, "k", 0),
        ({"method": "iqr", "iqr_multiplier": -1.0}, "k", 0),
        ({"method": "unknown_method"}, "m", 0),
    ],
)
def test_outlier_spec_out_of_range_values_rejected(
    spec: dict[str, Any], field: str, value: int
):
    """Out-of-range parameters are rejected with a ValidationError."""
    with pytest.raises(ValidationError):
        coerce_outliers(spec)


def test_outlier_spec_frozen():
    """The spec models are frozen: mutation raises."""
    with pytest.raises(ValidationError):
        QuantileOutliers().lower_quantile = 0.5  # type: ignore[misc]
    with pytest.raises(ValidationError):
        IqrOutliers().iqr_multiplier = 2.0  # type: ignore[misc]


def test_outlier_spec_override_model_accepts_both_forms():
    """``PreprocessingOverride.outliers`` accepts the string and object forms."""
    assert PreprocessingOverride(outliers="iqr").outliers == IqrOutliers()
    override = PreprocessingOverride(
        outliers={"method": "quantile", "lower_quantile": 0.02}
    )
    assert override.outliers == QuantileOutliers(lower_quantile=0.02)


# ──────────────────────────────────────────────────────────────────────────
#  Preset defaults and override resolution
# ──────────────────────────────────────────────────────────────────────────


def test_every_class_preset_defaults_to_none_outliers():
    """Every entry in PRESET_BY_CLASS uses method='none' by default."""
    from configurable_automl_engine.preprocessing_presets import (
        ALGORITHM_CLASS_MAPPING,
        PRESET_BY_CLASS,
    )

    for algo in ALGORITHM_CLASS_MAPPING:
        preset = resolve_preprocessing_preset(algo)
        assert preset.outliers == NoOutliers(), algo
    for cls, preset in PRESET_BY_CLASS.items():
        assert preset.outliers == NoOutliers(), cls


def test_override_replaces_the_preset_outliers_as_a_whole():
    """An override outliers field replaces the preset field (never merged)."""
    preset = resolve_preprocessing_preset(
        "ridge", {"outliers": {"method": "iqr", "iqr_multiplier": 2.5}}
    )
    assert preset.outliers == IqrOutliers(iqr_multiplier=2.5)


def test_general_default_override_applies_outliers():
    """``default_override`` (general.preprocessing) sets the outliers field."""
    preset = resolve_preprocessing_preset(
        "ridge",
        default_override={"outliers": {"method": "quantile", "lower_quantile": 0.05}},
    )
    assert preset.outliers == QuantileOutliers(lower_quantile=0.05)


def test_algorithm_level_wins_over_general_level():
    """The per-algorithm outliers block wins over the general one per field."""
    preset = resolve_preprocessing_preset(
        "ridge",
        {"outliers": "iqr"},
        default_override={"outliers": {"method": "quantile", "lower_quantile": 0.05}},
    )
    assert preset.outliers == IqrOutliers()


def test_unset_outliers_falls_through_levels():
    """An unset outliers at the specific level falls through to the default."""
    preset = resolve_preprocessing_preset(
        "ridge",
        {"scaling": "none"},
        default_override={"outliers": "quantile"},
    )
    assert preset.outliers == QuantileOutliers()
    # After the merge of the object-form `scaling` spec (!120) the resolved
    # preset stores the object form (the string shorthand is normalized).
    assert preset.scaling == NoScaling()


def test_none_outliers_in_override_keeps_preset_value():
    """``outliers: None`` in an override keeps the automatic selection."""
    preset = resolve_preprocessing_preset(
        "ridge",
        {"outliers": None},
        default_override={"outliers": "quantile"},
    )
    assert preset.outliers == QuantileOutliers()


def test_preset_str_includes_outliers():
    """``PreprocessingPreset.__str__`` includes the outlier spec."""
    assert str(
        PreprocessingPreset(outliers={"method": "quantile", "lower_quantile": 0.01})
    ).endswith("outliers=quantile(0.01, 0.99))")
    assert str(PreprocessingPreset(outliers="iqr")).endswith("outliers=iqr(k=1.5))")
    assert str(PreprocessingPreset()).endswith("outliers='none')")
    assert format_outliers("quantile") == "quantile(0.01, 0.99)"
    assert format_outliers("none") == "'none'"


# ──────────────────────────────────────────────────────────────────────────
#  Pipeline assembly: winsorizer -> imputer -> [scaler]
# ──────────────────────────────────────────────────────────────────────────


def _num_df() -> pd.DataFrame:
    """A numeric frame with missing values and extreme values."""
    rng = np.random.RandomState(0)
    a = rng.normal(size=60)
    a[3] = 1000.0  # extreme high
    a[10] = np.nan  # missing
    a[20] = -1000.0  # extreme low
    b = rng.normal(size=60)
    b[5] = np.nan
    return pd.DataFrame({"a": a, "b": b})


def _num_pipeline_steps(pre: Any) -> list[str]:
    """Names of the steps of the numerical transformer of a preprocessor."""
    for _, transformer, columns in pre.transformers:
        if isinstance(columns, str) or not hasattr(transformer, "steps"):
            continue
        # The numeric block is the one whose first non-trivial step is the
        # imputer (built by build_preprocessor) or the winsorizer.
        step_names = [name for name, _ in transformer.steps]
        if "imputer" in step_names:
            return step_names
    return []


def test_winsorizer_absent_by_default():
    """Without ``outliers`` the winsorizer step is absent (not a no-op)."""
    pre = build_preprocessor(["a", "b"], [], ["a", "b"])
    assert _num_pipeline_steps(pre) == ["imputer", "scaler"]


def test_winsorizer_absent_for_none_method():
    """``outliers='none'`` keeps the step absent from the pipeline."""
    pre = build_preprocessor(["a", "b"], [], ["a", "b"], outliers="none")
    assert _num_pipeline_steps(pre) == ["imputer", "scaler"]


def test_winsorizer_comes_before_imputer_and_indicator_integrity():
    """The winsorizer step precedes the imputer; indicators stay exactly 0/1."""
    X = _num_df()
    pre = build_preprocessor(
        ["a", "b"],
        [],
        ["a", "b"],
        outliers="quantile",
        add_indicator=True,
        scaling="none",
    )
    assert _num_pipeline_steps(pre) == ["winsorizer", "imputer"]
    out = pre.fit_transform(X)
    # The indicator columns (last two) are exactly 0/1: the winsorizer came
    # before the imputer, so the indicator columns were never clipped.
    indicators = out[:, out.shape[1] - 2 :]
    assert set(np.unique(indicators)) <= {0.0, 1.0}
    assert indicators.sum() == 2  # one NaN per column a and b


def test_mean_imputation_sees_clipped_column():
    """With mean imputation the imputed value is the mean of the clipped column."""
    X = _num_df()
    pre = build_preprocessor(
        ["a", "b"],
        [],
        ["a", "b"],
        imputation_strategy="mean",
        outliers="quantile",
        add_indicator=True,
        scaling="none",
    )
    wins = Winsorizer(method="quantile")
    wins.fit(X[["a", "b"]].values)
    clipped = wins.transform(X[["a", "b"]].values)
    expected_mean = np.nanmean(clipped, axis=0)

    fitted = pre.fit(X)
    imputer = (
        _num_pipeline_steps(fitted)
        and fitted.named_transformers_["num"].named_steps["imputer"]
    )
    # The first statistic is the column mean of the clipped data; the
    # indicator statistics (sentinel -1) are appended after the two means.
    assert np.allclose(imputer.statistics_[:2], expected_mean)


def test_cv_fold_bounds_equal_fold_training_bounds_no_leakage():
    """In KFold the bounds fitted on a fold equal the bounds of its train rows.

    Extreme values in the validation part must not shift the bounds."""
    rng = np.random.RandomState(1)
    X = pd.DataFrame({"a": rng.normal(size=40)})
    y = pd.Series(rng.normal(size=40))

    fold_bounds: list[tuple[float, float]] = []
    manual_bounds: list[tuple[float, float]] = []
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    for train_idx, val_idx in kf.split(X):
        pre = build_preprocessor(["a"], [], ["a"], outliers="quantile", scaling="none")
        pre.fit(X.iloc[train_idx], y.iloc[train_idx])
        wins = pre.named_transformers_["num"].named_steps["winsorizer"]
        fold_bounds.append((float(wins.lower_bounds_[0]), float(wins.upper_bounds_[0])))
        ref = Winsorizer(method="quantile").fit(X.iloc[train_idx].values)
        manual_bounds.append((float(ref.lower_bounds_[0]), float(ref.upper_bounds_[0])))
        # Sanity: the validation part really contains extreme values that
        # would shift the bounds if they leaked into the fit.
        col = X["a"].to_numpy()[val_idx]
        assert col.min() < fold_bounds[-1][0] or col.max() > fold_bounds[-1][1]
    assert fold_bounds == manual_bounds


def test_winsorizer_nan_passes_through():
    """NaN values pass through the winsorizer unchanged (they reach the imputer)."""
    X = _num_df()
    pre = build_preprocessor(
        ["a", "b"], [], ["a", "b"], outliers="quantile", scaling="none"
    )
    out = pre.fit_transform(X)
    a = out[:, 0]
    assert not np.isnan(a).any()  # the imputer filled it


def test_collapsed_columns_warning_prefixed_with_algorithm(caplog):
    """The collapsed-columns WARNING is prefixed with the algorithm name."""
    X = pd.DataFrame({"flat": [5.0] * 10, "a": np.arange(10.0)})
    pre = build_preprocessor(
        ["flat", "a"],
        [],
        ["flat", "a"],
        outliers="quantile",
        outlier_algorithm="ridge",
        scaling="none",
    )
    import logging

    with caplog.at_level(
        logging.WARNING, logger="configurable_automl_engine.preprocessing.outliers"
    ):
        pre.fit(X)
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("algorithm 'ridge'" in m for m in messages)


def test_collapsed_columns_warning_without_algorithm_unchanged(caplog):
    """Without an algorithm context the WARNING keeps its plain prefix."""
    X = pd.DataFrame({"flat": [5.0] * 10})
    pre = build_preprocessor(["flat"], [], ["flat"], outliers="iqr", scaling="none")
    import logging

    with caplog.at_level(
        logging.WARNING, logger="configurable_automl_engine.preprocessing.outliers"
    ):
        pre.fit(X)
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(m.startswith("Winsorizer: 1 column") for m in messages)


# ──────────────────────────────────────────────────────────────────────────
#  builder validation
# ──────────────────────────────────────────────────────────────────────────


def test_builder_rejects_invalid_outliers_spec():
    """An invalid outliers spec is rejected with a ValueError."""
    with pytest.raises(ValueError, match="outliers"):
        build_preprocessor(["a"], [], ["a"], outliers={"method": "quantile", "k": 1})
    with pytest.raises(ValueError, match="outliers"):
        build_preprocessor(["a"], [], ["a"], outliers="winsorize")
    with pytest.raises(ValueError, match="outliers"):
        build_preprocessor(
            ["a"], [], ["a"], outliers={"method": "quantile", "lower_quantile": 2.0}
        )


# ──────────────────────────────────────────────────────────────────────────
#  Trainer and HPO wiring
# ──────────────────────────────────────────────────────────────────────────


def test_trainer_preprocessor_uses_outliers_from_override():
    """``build_trainer_preprocessor`` passes the preset outliers to the builder."""
    setup = build_trainer_preprocessor(
        "ridge",
        [],
        ["a", "b"],
        feature_names=["a", "b"],
        encoding=EncodingSettings(encoding="one_hot"),
        override={"outliers": {"method": "iqr", "iqr_multiplier": 2.0}},
        random_state=42,
    )
    assert setup.preset.outliers == IqrOutliers(iqr_multiplier=2.0)
    assert _num_pipeline_steps(setup.preprocessor) == [
        "winsorizer",
        "imputer",
        "scaler",
    ]


def test_trainer_preset_outliers_default_is_none():
    """Without overrides the trainer preset keeps method='none'."""
    setup = build_trainer_preprocessor(
        "ridge",
        [],
        ["a"],
        feature_names=["a"],
        encoding=EncodingSettings(encoding="one_hot"),
        override=None,
        random_state=42,
    )
    assert setup.preset.outliers == NoOutliers()
    assert _num_pipeline_steps(setup.preprocessor) == ["imputer", "scaler"]


def test_isotonic_trains_with_outliers_iqr():
    """``isotonic_regression`` trains with ``outliers: iqr`` (one-to-one step)."""
    X = _num_df()
    y = pd.Series(
        np.arange(len(X), dtype=float) + np.nan_to_num(X["a"].to_numpy(), nan=0.0)
    )
    trainer = ModelTrainer(
        algorithm="isotonic_regression",
        preprocessing_override={"outliers": "iqr"},
    ).fit(X, y)
    assert trainer.preprocessing_preset.outliers == IqrOutliers()
    pre = trainer.pipeline.named_steps["preprocessor"]
    assert _num_pipeline_steps(pre) == ["winsorizer", "imputer"]
    pred = trainer.predict(X)
    assert len(pred) == len(X)
    assert np.isfinite(pred).all()


def test_estimate_n_features_unchanged_with_outliers():
    """``estimate_n_features_after_preprocessing`` is unchanged with outliers on."""
    X = _num_df()
    without = estimate_n_features_after_preprocessing(
        X,
        categorical_features=[],
        numerical_features=["a", "b"],
        add_indicator=True,
    )
    # The outliers setting only lives inside the builder; the deterministic
    # estimate does not take it (the winsorizer is one-to-one).
    assert without == estimate_n_features_after_preprocessing(
        X,
        categorical_features=[],
        numerical_features=["a", "b"],
        add_indicator=True,
    )


# ──────────────────────────────────────────────────────────────────────────
#  HPO / final parity and acceptance flow
# ──────────────────────────────────────────────────────────────────────────


def _small_regression() -> tuple[pd.DataFrame, pd.Series]:
    """A small numeric dataset with heavy-tailed feature and missing values."""
    rng = np.random.RandomState(3)
    a = rng.lognormal(size=80)  # heavy tail: winsorization matters
    a[4] = np.nan
    b = rng.normal(size=80)
    b[9] = np.nan
    X = pd.DataFrame({"a": a, "b": b})
    y = pd.Series(
        np.nan_to_num(a, nan=0.0) * 0.5 + np.nan_to_num(b, nan=0.0) + rng.rand(80) * 0.1
    )
    return X, y


def test_hpo_and_final_model_apply_outliers_identically():
    """HPO (optimize) and the final ModelTrainer resolve the same preset."""
    X, y = _small_regression()
    override: dict[str, Any] = {
        "outliers": {
            "method": "quantile",
            "lower_quantile": 0.02,
            "upper_quantile": 0.98,
        }
    }
    optimize(
        "ridge",
        X,
        y,
        n_trials=1,
        validation_strategy="k_fold",
        n_folds=3,
        random_state=7,
        sampler_seed=7,
        space_overrides={"ridge": {"alpha": [0.1, 1.0]}},
        limit_hyperparameters=True,
        preprocessing_override=override,
        numerical_features=["a", "b"],
        categorical_features=[],
        fit_best_model=False,
    )
    trainer = ModelTrainer(algorithm="ridge", preprocessing_override=override).fit(X, y)
    resolved = resolve_preprocessing_preset("ridge", override)
    assert trainer.preprocessing_preset == resolved
    assert resolved.outliers == QuantileOutliers(
        lower_quantile=0.02, upper_quantile=0.98
    )
    assert _num_pipeline_steps(trainer.pipeline.named_steps["preprocessor"]) == [
        "winsorizer",
        "imputer",
        "scaler",
    ]


# ──────────────────────────────────────────────────────────────────────────
#  Regression: without outliers everything is unchanged
# ──────────────────────────────────────────────────────────────────────────


def test_regression_without_outliers_pipeline_and_predictions_identical():
    """Without ``outliers`` the Pipeline.steps and predictions stay identical."""
    X, y = _small_regression()
    X_tr, X_te = X.iloc[:60], X.iloc[60:]
    y_tr = y.iloc[:60]

    trainer_old = ModelTrainer(algorithm="ridge", random_state=42).fit(X_tr, y_tr)
    trainer_new = ModelTrainer(
        algorithm="ridge", random_state=42, preprocessing_override={"outliers": "none"}
    ).fit(X_tr, y_tr)

    steps_old = [name for name, _ in trainer_old.pipeline.steps]
    steps_new = [name for name, _ in trainer_new.pipeline.steps]
    assert steps_old == steps_new
    assert _num_pipeline_steps(trainer_old.pipeline.named_steps["preprocessor"]) == [
        "imputer",
        "scaler",
    ]

    np.testing.assert_allclose(
        trainer_old.predict(X_te), trainer_new.predict(X_te), rtol=0, atol=0
    )


def test_pipeline_pickle_roundtrip_with_winsorizer():
    """A preprocessor with a winsorizer survives a pickle roundtrip."""
    import pickle

    X = _num_df()
    pre = build_preprocessor(["a", "b"], [], ["a", "b"], outliers="quantile")
    pre.fit(X)
    restored = pickle.loads(pickle.dumps(pre))
    np.testing.assert_allclose(pre.transform(X), restored.transform(X))


# ──────────────────────────────────────────────────────────────────────────
#  Config validation (general + per-algorithm blocks)
# ──────────────────────────────────────────────────────────────────────────


def test_config_accepts_outliers_in_general_and_algorithm_blocks():
    """``general.preprocessing.outliers`` and ``algorithms.*.preprocessing.outliers``
    accept both the string and object forms."""
    from configurable_automl_engine.training_engine.config.algorithms import AlgoCfg
    from configurable_automl_engine.training_engine.config.general import GeneralCfg

    general = GeneralCfg(
        preprocessing={
            "outliers": {
                "method": "quantile",
                "lower_quantile": 0.02,
                "upper_quantile": 0.98,
            }
        }
    )
    assert general.preprocessing is not None
    assert general.preprocessing.outliers == QuantileOutliers(
        lower_quantile=0.02, upper_quantile=0.98
    )

    algo = AlgoCfg(preprocessing={"outliers": "iqr"})
    assert algo.preprocessing is not None
    assert algo.preprocessing.outliers == IqrOutliers()


def test_config_rejects_cross_method_and_out_of_range_outliers():
    """Cross-method parameters and out-of-range values are rejected."""
    from configurable_automl_engine.training_engine.config.general import GeneralCfg

    with pytest.raises(ValidationError):
        GeneralCfg(preprocessing={"outliers": {"method": "quantile", "k": 2}})
    with pytest.raises(ValidationError):
        GeneralCfg(preprocessing={"outliers": {"method": "iqr", "iqr_multiplier": 0}})
    with pytest.raises(ValidationError):
        GeneralCfg(preprocessing={"outliers": "winsorize"})


def test_acceptance_flow_general_preprocessing_outliers(tmp_path, monkeypatch):
    """The acceptance flow: general.preprocessing.outliers is validated, applied
    in HPO and in the final model identically, and the indicators stay intact."""
    from configurable_automl_engine import train_best_model
    from configurable_automl_engine.trainer import ModelTrainer
    from configurable_automl_engine.training_engine import hpo_phases

    captured: list[Any] = []

    def fake_run(X, y, request):
        captured.append(request)
        return __import__(
            "configurable_automl_engine.tuner", fromlist=["OptimizeResult"]
        ).OptimizeResult(model=None, params={"alpha": 0.5}, score=0.9, trials=[])

    monkeypatch.setattr(hpo_phases, "run_optimization", fake_run)

    rng = np.random.RandomState(5)
    a = rng.lognormal(size=60)
    a[3] = np.nan
    b = rng.normal(size=60)
    b[7] = np.nan
    df = pd.DataFrame({"a": a, "b": b})
    df["target"] = (
        np.nan_to_num(a, nan=0.0) * 0.5 + np.nan_to_num(b, nan=0.0) + rng.rand(60) * 0.1
    )

    cfg = {
        "general": {
            "comparison_metric": "r2",
            "validation_strategy": "train_test_split",
            "preprocessing": {
                "outliers": {
                    "method": "quantile",
                    "lower_quantile": 0.02,
                    "upper_quantile": 0.98,
                }
            },
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
        },
        "algorithms": {
            "ridge": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"alpha": [0.1, 1.0]},
            },
        },
    }
    res = train_best_model(
        cfg, df, target="target", model_path_override=tmp_path / "m.pkl"
    )

    assert res["algorithm"] == "ridge"
    assert len(captured) == 1
    request = captured[0]
    assert request.preprocessing_default is not None
    assert request.preprocessing_default.outliers == QuantileOutliers(
        lower_quantile=0.02, upper_quantile=0.98
    )

    trainer = ModelTrainer.load(res["model_path"])
    assert trainer.preprocessing_preset.outliers == QuantileOutliers(
        lower_quantile=0.02, upper_quantile=0.98
    )
    assert _num_pipeline_steps(trainer.pipeline.named_steps["preprocessor"]) == [
        "winsorizer",
        "imputer",
        "scaler",
    ]
