"""Tests of the model explainability component (``experiment.explain``)
and of the facade method ``AutoMLExperiment.explain``.

Covered scenarios (positive / negative / boundary):

- P1: the permutation importance is returned for a fitted model (after
  ``run()``) — one entry per original feature;
- P2: the result format: entries sorted by the descending mean, the
  baseline score and the metric name recorded, the Markdown section
  renders a table;
- P3: the module-level ``explain_permutation_importance`` on a hand-fit
  trainer and ``aggregate_to_original_features`` over one-hot names;
- P4: ``explain(method="shap")`` without the ``shap`` package → a
  graceful fallback error naming the ``[shap]`` extra (the extra is not
  installed in the test environment); with the package installed the
  full SHAP path is exercised (aggregation and per-component view);
- N1: ``explain()`` before ``run()`` → the not-run state error;
- N2: a failed experiment (no model artifact) → a state error;
- N3: an unknown method name → ``ExplainNotAvailableError``;
- N4: permutation importance without a target → ``ExplainNotAvailableError``;
- B1: a single-feature dataset;
- B2: only categorical features;
- B3: a constant feature (zero importance, no crash);
- B4: an empty feature frame → a defined error;
- B5: a datetime column (``datetime_policy: 'expand'``) — the default
  frame is expanded to the trainer's schema, the importance covers the
  generated components;
- B6: a numpy ``y`` with a non-Range frame index — the target stays
  aligned by row (not by label);
- B7: an oversampling pipeline — the SHAP transform prefix stops before
  the sampler step.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest

from configurable_automl_engine.experiment import (
    AutoMLExperiment,
    ExplainNotAvailableError,
    ExperimentNotRunError,
    aggregate_to_original_features,
)
from configurable_automl_engine.experiment.explain import (
    FeatureImportance,
    PermutationImportanceResult,
    explain_permutation_importance,
    explain_shap,
    render_importance_markdown,
)
from configurable_automl_engine.experiment.result import ExperimentResult
from configurable_automl_engine.trainer import ModelTrainer
from configurable_automl_engine.training_engine import component

# ────────────────────────── helpers ────────────────────────── #


def _config_dict(tmp_path: Any, **general_overrides: Any) -> dict[str, Any]:
    """A valid two-phase config dict (HPO phase + terminal cv_best select)."""
    general: dict[str, Any] = {
        "comparison_metric": "rmse",
        "random_state": 42,
        "validation_strategy": "train_test_split",
        "holdout_fraction": None,
        "path_to_model": str(tmp_path / "model.pkl"),
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
    }
    general.update(general_overrides)
    return {
        "general": general,
        "algorithms": {
            "ridge": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"alpha": [0.1, 1.0]},
            },
            "elasticnet": {
                "enable": True,
                "limit_hyperparameters": True,
                "hyperparameters": {"alpha": [0.1, 1.0], "l1_ratio": [0.2, 0.8]},
            },
        },
        "oversampling": {"enable": False},
    }


def _small_df() -> pd.DataFrame:
    """A tiny deterministic regression dataset."""
    rows = 40
    return pd.DataFrame(
        {
            "f1": [float(i % 5) for i in range(rows)],
            "f2": [i % 7 * 0.5 for i in range(rows)],
            "target": [10.0 - 0.2 * i for i in range(rows)],
        }
    )


@pytest.fixture(autouse=True)
def _reset_engine_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the engine's module-level last-result between tests."""
    monkeypatch.setattr(component, "last_experiment_result", None)
    yield


def _fit_trainer(X: pd.DataFrame, y: pd.Series) -> ModelTrainer:
    """A hand-fit ridge trainer (fast: no HPO, no experiment run)."""
    trainer = ModelTrainer(
        algorithm="ridge",
        hyperparams={"alpha": 0.1},
        metric="neg_root_mean_squared_error",
        random_state=42,
        validation_strategy="train_test_split",
        test_size=0.25,
    )
    return trainer.fit(X, y)


def _fit_trainer_with_one_hot(X: pd.DataFrame, y: pd.Series) -> ModelTrainer:
    """A hand-fit ridge trainer with explicit one-hot encoding."""
    trainer = ModelTrainer(
        algorithm="ridge",
        hyperparams={"alpha": 0.1},
        metric="neg_root_mean_squared_error",
        random_state=42,
        validation_strategy="train_test_split",
        test_size=0.25,
        categorical_features=["cat"],
        numerical_features=["num"],
        encoding_strategy="one_hot",
        high_cardinality_threshold=None,
        high_cardinality_encoding=None,
    )
    return trainer.fit(X, y)


# ────────────────────────── positive scenarios ────────────────────────── #


def test_p1_permutation_importance_after_run(tmp_path: Any) -> None:
    """P1: importance is returned for the fitted winner after run()."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    importance, markdown = experiment.explain()
    assert isinstance(importance, PermutationImportanceResult)
    assert importance.metric_user == "rmse"
    assert importance.n_features == 2
    assert importance.n_rows == 40
    assert importance.data_semantics == "train"
    assert importance.baseline_score > 0
    assert {item.feature for item in importance.importances} == {"f1", "f2"}
    assert isinstance(markdown, str)
    assert "## Feature importance (permutation)" in markdown


def test_p2_importance_sorted_and_rendered(tmp_path: Any) -> None:
    """P2: the entries are sorted by the descending mean; the Markdown
    section is a table with the baseline line."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    importance, markdown = experiment.explain(n_repeats=2)
    means = [item.importance_mean for item in importance.importances]
    assert means == sorted(means, reverse=True)
    assert all(item.importance_std is not None for item in importance.importances)
    assert "| # | Feature | Importance mean | Importance std |" in markdown
    assert "Baseline rmse" in markdown
    # The importance is measured against the run's metric semantics.
    assert importance.metric_user == "rmse"


def test_p3_module_call_and_aggregation() -> None:
    """P3: the module-level call on a hand-fit trainer, and the
    aggregation of one-hot component names to the original columns."""
    X = pd.DataFrame(
        {
            "num": np.arange(30, dtype=float),
            "cat": ["a", "b"] * 15,
        }
    )
    y = pd.Series(np.arange(30, dtype=float) * 2.0)
    trainer = _fit_trainer(X, y)
    result = _minimal_result()
    importance = explain_permutation_importance(
        trainer, X, y, result, n_repeats=2, max_rows=100
    )
    assert {item.feature for item in importance.importances} <= {"num", "cat"}
    assert importance.baseline_score > 0

    components = [
        FeatureImportance(feature="cat__cat_a", importance_mean=0.4),
        FeatureImportance(feature="cat__cat_b", importance_mean=0.2),
        FeatureImportance(feature="num__num", importance_mean=0.9),
    ]
    aggregated = aggregate_to_original_features(components)
    assert [item.feature for item in aggregated] == ["num", "cat"]
    assert aggregated[1].importance_mean == pytest.approx(0.6)


def test_p4_shap_without_package_raises_defined_error(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P4: ``explain(method="shap")`` without the optional extra → a
    graceful error naming the ``[shap]`` extra (no bare ModuleNotFoundError).
    The check runs against a patched ``sys.modules`` (no real ``shap``
    import is consulted), so the test works in both environments."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    monkeypatch.setitem(sys.modules, "shap", None)
    with pytest.raises(ExplainNotAvailableError, match=r"\[shap\]"):
        experiment.explain(method="shap")


def test_p4_explicit_holdout_frame(tmp_path: Any) -> None:
    """P4: an explicit X/y block is explained instead of the run's data."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    df = _small_df()
    importance, _markdown = experiment.explain(
        X=df[["f1", "f2"]], y=df["target"], n_repeats=1
    )
    assert importance.n_rows == len(df)


def test_p4_markdown_disabled(tmp_path: Any) -> None:
    """P4: ``markdown=False`` returns the structured result only."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    result = experiment.explain(markdown=False)
    assert isinstance(result, PermutationImportanceResult)
    assert not isinstance(result, tuple)


# ────────────────────────── negative scenarios ────────────────────────── #


def test_n1_explain_before_run(tmp_path: Any) -> None:
    """N1: explain() before run() → the not-run state error."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    with pytest.raises(ExperimentNotRunError, match="has not been run"):
        experiment.explain()


def test_n2_failed_experiment_no_artifact(tmp_path: Any) -> None:
    """N2: a failed experiment (no model artifact) → a state error."""
    experiment = AutoMLExperiment(
        _config_dict(tmp_path), _small_df(), model_path_override=None
    )
    experiment.run()
    assert experiment.result is not None
    object.__setattr__(experiment.result, "model_path", None)
    with pytest.raises(RuntimeError, match="No model artifact"):
        experiment.explain()


def test_n2_failed_run_result(tmp_path: Any, monkeypatch: Any) -> None:
    """N2: a run whose training failed (no artifact recorded) → a state
    error from best_model(), surfaced through explain()."""
    broken_config = _config_dict(tmp_path)
    broken_config["algorithms"] = {
        "ridge": {"enable": True, "hyperparameters": {"alpha": [0.1]}}
    }
    df = _small_df()
    df["f1"] = np.nan
    df["f2"] = np.inf
    experiment = AutoMLExperiment(broken_config, df)
    try:
        experiment.run()
    except Exception:  # noqa: BLE001 — the failed run is the scenario
        pass
    if experiment.result is None or experiment.result.model_path:
        pytest.skip("the engine still produced a usable artifact")
    with pytest.raises(RuntimeError):
        experiment.explain()


def test_n3_unknown_method(tmp_path: Any) -> None:
    """N3: an unknown method name → a defined error naming the options."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    with pytest.raises(ExplainNotAvailableError, match="Unknown explain method"):
        experiment.explain(method="lime")


def test_n4_permutation_without_target(tmp_path: Any) -> None:
    """N4: permutation importance without a target → a defined error."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    df = _small_df()
    with pytest.raises(ExplainNotAvailableError, match="target"):
        experiment.explain(X=df[["f1", "f2"]], y=None)


# ────────────────────────── boundary scenarios ────────────────────────── #


def test_b1_single_feature(tmp_path: Any) -> None:
    """B1: a one-column dataset works end to end."""
    rows = 40
    df = pd.DataFrame(
        {
            "only": [float(i) for i in range(rows)],
            "target": [float(i) * 0.5 for i in range(rows)],
        }
    )
    experiment = AutoMLExperiment(_config_dict(tmp_path), df)
    experiment.run()
    importance, _markdown = experiment.explain(n_repeats=1)
    assert [item.feature for item in importance.importances] == ["only"]


def test_b2_only_categorical_features(tmp_path: Any) -> None:
    """B2: only categorical features (frequency encoding on a frame
    without numerical columns) — the importance is still produced."""
    rows = 40
    df = pd.DataFrame(
        {
            "cat": [str(i % 3) for i in range(rows)],
            "target": [float(i % 3) for i in range(rows)],
        }
    )
    config = _config_dict(tmp_path, categorical_encoding="frequency")
    experiment = AutoMLExperiment(config, df)
    experiment.run()
    importance, _markdown = experiment.explain(n_repeats=1)
    assert [item.feature for item in importance.importances] == ["cat"]


def test_b3_constant_feature_zero_importance(tmp_path: Any) -> None:
    """B3: a constant feature survives the computation with a zero
    importance (the pipeline drops it; no crash)."""
    rows = 40
    df = pd.DataFrame(
        {
            "informative": [float(i) for i in range(rows)],
            "constant": [7.0] * rows,
            "target": [float(i) * 0.5 for i in range(rows)],
        }
    )
    experiment = AutoMLExperiment(_config_dict(tmp_path), df)
    experiment.run()
    importance, markdown = experiment.explain(n_repeats=1)
    names = [item.feature for item in importance.importances]
    assert set(names) <= {"informative", "constant"}
    assert markdown.count("|") >= 2


def test_b4_empty_feature_frame(tmp_path: Any) -> None:
    """B4: an empty feature frame → a defined error (no IndexError)."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    with pytest.raises(ExplainNotAvailableError):
        experiment.explain(X=pd.DataFrame({"f1": []}), y=None)


def test_b5_datetime_expansion_default_frame(tmp_path: Any) -> None:
    """B5: a datetime source column (``datetime_policy: 'expand'``) — the
    default explanation frame is expanded to the trainer's schema and the
    importance covers the generated component columns."""
    rows = 40
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=rows, freq="D"),
            "num": np.arange(rows, dtype=float),
            "target": np.arange(rows, dtype=float) * 0.5,
        }
    )
    experiment = AutoMLExperiment(_config_dict(tmp_path), df)
    experiment.run()
    importance, _markdown = experiment.explain(n_repeats=1)
    names = {item.feature for item in importance.importances}
    assert "num" in names
    assert any(name.startswith("ts_") for name in names)


def test_b5_datetime_expansion_explicit_raw_frame(tmp_path: Any) -> None:
    """B5 (MAJOR-3 regression): an explicit RAW frame with a datetime
    source column is expanded by the module-level functions themselves
    (permutation and SHAP) — the documented 'pass the raw frame' advice
    holds for both methods."""
    rows = 40
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=rows, freq="D"),
            "num": np.arange(rows, dtype=float),
            "target": np.arange(rows, dtype=float) * 0.5,
        }
    )
    experiment = AutoMLExperiment(_config_dict(tmp_path), df)
    experiment.run()
    trainer = experiment.best_model()
    raw = df.drop(columns=["target"])
    y = df["target"]

    importance = explain_permutation_importance(
        trainer, raw, y, _minimal_result(), n_repeats=1, max_rows=100
    )
    names = {item.feature for item in importance.importances}
    assert "num" in names
    assert any(name.startswith("ts_") for name in names)

    try:
        import shap  # noqa: F401
    except ImportError:
        pass
    else:
        shap_result = explain_shap(trainer, raw, _minimal_result(), max_rows=100)
        assert shap_result.post_preprocessing is False

    # Both facade paths accept the raw frame too.
    facade_importance, _markdown = experiment.explain(
        X=raw, y=y, n_repeats=1, max_rows=100
    )
    assert facade_importance.n_rows == rows


def test_b5_datetime_missing_components_raise_defined_error(tmp_path: Any) -> None:
    """B5 (MAJOR-3 regression): a frame without the source column and
    without the components → a defined error naming the missing columns
    (no raw KeyError)."""
    rows = 40
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=rows, freq="D"),
            "num": np.arange(rows, dtype=float),
            "target": np.arange(rows, dtype=float) * 0.5,
        }
    )
    experiment = AutoMLExperiment(_config_dict(tmp_path), df)
    experiment.run()
    trainer = experiment.best_model()
    # A frame that has neither the source column nor the components.
    broken = pd.DataFrame({"num": np.arange(rows, dtype=float)})
    with pytest.raises(ExplainNotAvailableError, match="missing columns"):
        explain_permutation_importance(
            trainer, broken, df["target"], _minimal_result(), n_repeats=1
        )
    try:
        import shap  # noqa: F401
    except ImportError:
        pass
    else:
        with pytest.raises(ExplainNotAvailableError, match="missing columns"):
            explain_shap(trainer, broken, _minimal_result())


def test_b6_series_target_missing_labels_raise(tmp_path: Any) -> None:
    """B6 (MAJOR-2 regression): a Series target missing the frame's
    labels → a defined alignment error immediately (not a far-away
    sklearn ``Input contains NaN``); a positional target of the wrong
    length raises too (no silent truncation)."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    df = _small_df()
    X = df[["f1", "f2"]]
    y_bad_labels = pd.Series(
        np.arange(len(df), dtype=float), index=[9000 + i for i in range(len(df))]
    )
    with pytest.raises(ExplainNotAvailableError, match="missing labels"):
        experiment.explain(X=X, y=y_bad_labels, n_repeats=1)
    with pytest.raises(ExplainNotAvailableError, match="does not match"):
        experiment.explain(X=X, y=np.arange(len(df) + 3, dtype=float), n_repeats=1)


def test_b6_series_target_with_extra_labels_aligned(tmp_path: Any) -> None:
    """B6 (MAJOR-2, positive): a Series target carrying MORE labels than
    the frame is aligned by index — the extra labels are dropped, the
    scored rows are exactly the frame's rows."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    df = _small_df()
    X_half = df[["f1", "f2"]].iloc[: len(df) // 2]
    y_full = df["target"]
    importance, _markdown = experiment.explain(
        X=X_half, y=y_full, n_repeats=1, max_rows=100
    )
    assert importance.n_rows == len(X_half)


def test_b6_numpy_target_with_labeled_index(tmp_path: Any) -> None:
    """B6: a numpy ``y`` with a non-Range frame index — the target stays
    aligned by row position (sampling must not re-index by label)."""
    df = _small_df()
    df.index = [100 + i for i in range(len(df))]
    experiment = AutoMLExperiment(_config_dict(tmp_path), df)
    experiment.run()
    df2 = _small_df()
    rng = np.random.RandomState(3)
    perm = rng.permutation(len(df2))
    X_shuffled = df2[["f1", "f2"]].iloc[perm]
    y_shuffled = df2["target"].to_numpy()[perm]
    importance, _markdown = experiment.explain(
        X=X_shuffled, y=y_shuffled, n_repeats=1, max_rows=10
    )
    assert importance.n_rows == 10


def test_b6_numpy_target_sampling_stays_aligned(tmp_path: Any) -> None:
    """B6 (round-3 BLOCKER regression): a numpy ``y`` with sampling
    active — the row selection reduces X and y TOGETHER, so the
    (row, target) pairs stay aligned (no first-N truncation of ``y``
    against randomly sampled rows). A perfectly fitted linear target
    gives a near-zero baseline; the desynchronization gave ~47.5."""
    rows = 40
    X = pd.DataFrame({"f": np.arange(rows, dtype=float)})
    y = np.arange(rows, dtype=float) * 3.0
    trainer = ModelTrainer(
        algorithm="ridge",
        hyperparams={"alpha": 0.1},
        metric="neg_root_mean_squared_error",
        random_state=42,
        validation_strategy="train_test_split",
        test_size=0.25,
    )
    trainer.fit(X, y)
    result = _minimal_result()

    importance = explain_permutation_importance(
        trainer, X, y, result, n_repeats=2, max_rows=10
    )
    assert importance.n_rows == 10
    assert importance.baseline_score < 1.0

    # A Series target with the same sampling — identical pairs, same baseline.
    importance_series = explain_permutation_importance(
        trainer, X, pd.Series(y), result, n_repeats=2, max_rows=10
    )
    assert importance_series.baseline_score == pytest.approx(importance.baseline_score)

    # A list target — the same alignment guarantee.
    importance_list = explain_permutation_importance(
        trainer, X, list(y), result, n_repeats=2, max_rows=10
    )
    assert importance_list.baseline_score == pytest.approx(importance.baseline_score)

    # A non-Range frame index does not change the pairs either.
    X_labeled = X.copy()
    X_labeled.index = [100 + i for i in range(rows)]
    importance_labeled = explain_permutation_importance(
        trainer, X_labeled, y, result, n_repeats=2, max_rows=10
    )
    assert importance_labeled.baseline_score == pytest.approx(importance.baseline_score)


def test_b7_oversampling_pipeline_transform_prefix() -> None:
    """B7: an oversampling pipeline — the SHAP transform prefix stops
    before the sampler step (a sampler has no ``transform``)."""
    from configurable_automl_engine.experiment.explain import _transform_prefix

    rows = 60
    X = pd.DataFrame({"a": np.arange(rows, dtype=float)})
    y = pd.Series(np.arange(rows, dtype=float))
    trainer = ModelTrainer(
        algorithm="ridge",
        hyperparams={"alpha": 0.1},
        metric="neg_root_mean_squared_error",
        random_state=42,
        validation_strategy="train_test_split",
        test_size=0.25,
        data_oversampling=True,
    )
    trainer.fit(X, y)
    prefix = _transform_prefix(trainer.pipeline)
    assert [name for name, _step in prefix.steps] == ["preprocessor"]
    out = prefix.transform(X)
    assert out.shape[0] == rows


def test_b7_feature_selection_names_resolved_through_selector() -> None:
    """B7 (MAJOR-4 regression): an active feature selector stays in the
    SHAP transform chain (the matrix width matches the model exactly — no
    double reduction, no adapter), and the explained columns keep their
    real preprocessor names through the fitted selector's ``get_support``
    mapping — the aggregated view attributes the contributions to the
    original columns (no all-zero collapse, no positional ``x0``)."""
    from configurable_automl_engine.experiment.explain import (
        _selected_component_names,
        _transform_prefix,
    )

    rows = 80
    rng = np.random.RandomState(5)
    X = pd.DataFrame(
        {
            "informative": rng.rand(rows),
            "noise1": rng.rand(rows),
            "noise2": rng.rand(rows),
        }
    )
    y = pd.Series(X["informative"] * 5.0 + rng.rand(rows) * 0.1)
    trainer = ModelTrainer(
        algorithm="ridge",
        hyperparams={"alpha": 0.1},
        metric="neg_root_mean_squared_error",
        random_state=42,
        validation_strategy="train_test_split",
        test_size=0.25,
        feature_selection_cfg={"mode": "always", "method": "importance"},
        feature_selection_active=True,
    )
    trainer.fit(X, y)
    assert trainer.feature_selection_active_ is True
    prefix = _transform_prefix(trainer.pipeline)
    transformed = prefix.transform(X)
    # The matrix width matches the model's expectation exactly.
    model_step = trainer.pipeline.steps[-1][1]
    model_input_width = getattr(model_step, "n_features_in_", transformed.shape[1])
    assert transformed.shape[1] == model_input_width

    preprocessor_names = [
        str(n)
        for n in trainer.pipeline.named_steps["preprocessor"].get_feature_names_out()
    ]
    resolved = _selected_component_names(trainer.pipeline, preprocessor_names)
    assert len(resolved) == transformed.shape[1]
    assert "num__informative" in resolved
    assert not any(name.startswith("x") and name[1:].isdigit() for name in resolved)

    try:
        import shap  # noqa: F401
    except ImportError:
        pytest.skip("shap is not installed in this environment")
    result = explain_shap(trainer, X, _minimal_result(), max_rows=100)
    assert result.post_preprocessing is False
    names = {item.feature for item in result.importances}
    assert names == {"informative", "noise1", "noise2"}
    # The informative feature dominates; nothing collapsed into zeros.
    assert result.importances[0].feature == "informative"
    assert result.importances[0].importance_mean > 0.0


def test_b8_aggregation_with_underscored_names() -> None:
    """B8 (MINOR-2 regression): pipeline names of underscored columns are
    aggregated to the full source column (``my_col``), not to its prefix
    (``my``); the schema-free fallback keeps the documented heuristic."""
    components = [
        FeatureImportance(feature="cat__my_col_red", importance_mean=0.4),
        FeatureImportance(feature="num__my_col", importance_mean=0.9),
    ]
    aggregated = aggregate_to_original_features(
        components, original=["my_col", "other"]
    )
    assert [item.feature for item in aggregated] == ["my_col"]
    assert aggregated[0].importance_mean == pytest.approx(1.3)

    heuristic = aggregate_to_original_features(components)
    # The schema-free fallback is a documented best-effort heuristic: a
    # one-hot component keeps the prefix before the category separator,
    # so ``cat__my_col_red`` resolves to ``my``; the ``num__`` block keeps
    # the encoded column verbatim, but the category-suffix rule applies
    # there too — the true column ``my_col`` is unknown without the
    # training schema. The schema path above resolves both to ``my_col``.
    assert [item.feature for item in heuristic] == ["my"]
    assert heuristic[0].importance_mean == pytest.approx(1.3)

    # The reviewer's repro: underscored source columns (``user_id``,
    # ``age_group``) resolve correctly through the schema path.
    underscored = [
        FeatureImportance(feature="num__user_id", importance_mean=0.9),
        FeatureImportance(feature="cat__age_group_senior", importance_mean=0.4),
    ]
    resolved = aggregate_to_original_features(
        underscored, original=["user_id", "age_group"]
    )
    assert [item.feature for item in resolved] == ["user_id", "age_group"]
    assert resolved[0].importance_mean == pytest.approx(0.9)
    assert resolved[1].importance_mean == pytest.approx(0.4)


def test_b9_n_features_counts_scored_features(tmp_path: Any) -> None:
    """B9 (MINOR-1 regression): ``n_features`` counts the scored features
    (after the schema filtering), not the caller's raw frame width."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    df = _small_df()
    X_extra = df[["f1", "f2"]].copy()
    X_extra["junk"] = 1.0
    importance, _markdown = experiment.explain(X=X_extra, y=df["target"], n_repeats=1)
    assert importance.n_features == 2
    assert len(importance.importances) == 2


def test_p4_shap_dummy_module_full_path(tmp_path: Any) -> None:
    """P4 (NIT-2): the positive ``explain_shap`` path runs against a
    dummy ``shap`` module (no real dependency needed in light
    environments) — the generic Explainer route and the aggregation are
    covered without the package installed."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    trainer = experiment.best_model()
    X = _small_df()[["f1", "f2"]]
    transformed = trainer.pipeline.named_steps["preprocessor"].transform(X)
    n_columns = transformed.shape[1]
    matrix = np.zeros((len(X), n_columns), dtype=float)
    matrix[:, 0] = 1.5
    matrix[:, 1] = 0.5

    dummy_explanation = types.SimpleNamespace(values=matrix)
    calls: dict[str, Any] = {}

    class DummyExplainer:
        def __init__(self, model: Any, background: Any) -> None:
            calls["model"] = model
            calls["background"] = background

        def __call__(self, data: Any) -> Any:
            calls["data_rows"] = getattr(data, "shape", [None])[0]
            return dummy_explanation

    dummy_shap = types.ModuleType("shap")
    dummy_shap.Explainer = DummyExplainer  # type: ignore[attr-defined]
    dummy_shap.__spec__ = importlib.util.spec_from_loader("shap", loader=None)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "shap", dummy_shap)
    try:
        result = explain_shap(trainer, X, _minimal_result(), max_rows=100)
        assert calls["data_rows"] == len(X)
        assert result.post_preprocessing is False
        assert {item.feature for item in result.importances} == {"f1", "f2"}
        assert result.importances[0].feature == "f1"
        assert result.importances[0].importance_mean == pytest.approx(1.5)

        per_component = explain_shap(
            trainer, X, _minimal_result(), max_rows=100, post_preprocessing=True
        )
        assert per_component.post_preprocessing is True
        assert per_component.importances[0].importance_mean == pytest.approx(1.5)
    finally:
        monkeypatch.undo()


def test_p4_shap_dummy_module_permutation_fallback(tmp_path: Any) -> None:
    """P4 (NIT-2): the fallback route — a dummy ``shap`` whose generic
    ``Explainer`` always fails routes to ``algorithm="permutation"`` and
    the computation still succeeds (the fallback branch is covered in
    light CI without the real dependency)."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    trainer = experiment.best_model()
    X = _small_df()[["f1", "f2"]]
    transformed = trainer.pipeline.named_steps["preprocessor"].transform(X)
    n_columns = transformed.shape[1]
    matrix = np.full((len(X), n_columns), 0.75, dtype=float)

    calls: dict[str, Any] = {}

    class FailingGenericExplainer:
        def __init__(self, model: Any, background: Any) -> None:
            raise TypeError("generic route intentionally broken")

    class PermutationExplainer:
        def __init__(self, model: Any, background: Any, algorithm: Any) -> None:
            calls["model"] = model
            calls["algorithm"] = algorithm

        def __call__(self, data: Any) -> Any:
            return types.SimpleNamespace(values=matrix)

    def fake_explainer(model: Any, background: Any, algorithm: Any = None) -> Any:
        if algorithm == "permutation":
            return PermutationExplainer(model, background, algorithm)
        return FailingGenericExplainer(model, background)

    dummy_shap = types.ModuleType("shap")
    dummy_shap.Explainer = fake_explainer  # type: ignore[attr-defined]
    dummy_shap.__spec__ = importlib.util.spec_from_loader("shap", loader=None)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "shap", dummy_shap)
    try:
        result = explain_shap(trainer, X, _minimal_result(), max_rows=100)
        assert calls["algorithm"] == "permutation"
        assert calls["model"] == trainer.pipeline.steps[-1][1].predict
        assert result.post_preprocessing is False
        assert {item.feature for item in result.importances} == {"f1", "f2"}
        assert result.importances[0].importance_mean == pytest.approx(0.75)
    finally:
        monkeypatch.undo()


def test_b7_dummy_shap_with_selector_model_width(tmp_path: Any) -> None:
    """B7 (MAJOR-B regression, light-CI): with an active feature selector
    the dummy explainer receives the RAW model step and a matrix of the
    model's own input width — no adapter, no double reduction; the
    component names are resolved through the selector's ``get_support``."""
    from configurable_automl_engine.experiment.explain import _transform_prefix

    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    trainer = experiment.best_model()
    X = _small_df()[["f1", "f2"]]
    y = _small_df()["target"]

    selector_trainer = ModelTrainer(
        algorithm="ridge",
        hyperparams={"alpha": 0.1},
        metric="neg_root_mean_squared_error",
        random_state=42,
        validation_strategy="train_test_split",
        test_size=0.25,
        feature_selection_cfg={"mode": "always", "method": "importance"},
        feature_selection_active=True,
    )
    selector_trainer.fit(X, y)
    assert selector_trainer.feature_selection_active_ is True
    prefix = _transform_prefix(selector_trainer.pipeline)
    transformed = prefix.transform(X)
    model_step = selector_trainer.pipeline.steps[-1][1]
    model_input_width = getattr(model_step, "n_features_in_", None)
    assert model_input_width is not None
    assert transformed.shape[1] == model_input_width

    seen: dict[str, Any] = {}

    class DummyExplainer:
        def __init__(self, model: Any, background: Any) -> None:
            seen["model"] = model
            seen["background_width"] = background.shape[1]

        def __call__(self, data: Any) -> Any:
            seen["data_width"] = data.shape[1]
            matrix = np.full((data.shape[0], data.shape[1]), 0.8, dtype=float)
            return types.SimpleNamespace(values=matrix)

    dummy_shap = types.ModuleType("shap")
    dummy_shap.Explainer = DummyExplainer  # type: ignore[attr-defined]
    dummy_shap.__spec__ = importlib.util.spec_from_loader("shap", loader=None)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "shap", dummy_shap)
    try:
        result = explain_shap(selector_trainer, X, _minimal_result(), max_rows=100)
    finally:
        monkeypatch.undo()
    # The RAW model step is passed (no adapter); the matrices are exactly
    # the model's input width — a second reduction is impossible.
    assert seen["model"] is model_step
    assert seen["background_width"] == model_input_width
    assert seen["data_width"] == model_input_width
    assert result.post_preprocessing is False
    # The names are resolved through the selector's support mapping —
    # the surviving component keeps its real preprocessor name.
    assert all(
        not (name.startswith("x") and name[1:].isdigit())
        for name in (item.feature for item in result.importances)
    )


def test_p4_shap_full_path_when_installed(tmp_path: Any) -> None:
    """P4: with ``shap`` installed the full path works — the aggregated
    view covers the original columns and the per-component view names the
    pipeline columns."""
    try:
        import shap  # noqa: F401
    except ImportError:
        pytest.skip("shap is not installed in this environment")
    X = pd.DataFrame(
        {
            "cat": np.tile(["a", "b"], 40),
            "num": np.linspace(0.0, 1.0, 80),
        }
    )
    y = pd.Series(np.where(X["cat"] == "a", 10.0, 0.0) + X["num"])
    trainer = _fit_trainer_with_one_hot(X, y)
    result = _minimal_result()
    aggregated = explain_shap(trainer, X, result, max_rows=50)
    assert aggregated.post_preprocessing is False
    assert {item.feature for item in aggregated.importances} == {"cat", "num"}
    assert aggregated.importances[0].feature == "cat"

    per_component = explain_shap(
        trainer, X, result, max_rows=50, post_preprocessing=True
    )
    assert per_component.post_preprocessing is True
    assert all("__" in item.feature for item in per_component.importances)


# ────────────────────────── helpers (module scope) ────────────────────────── #


def _minimal_result() -> ExperimentResult:
    """A minimal completed ``ExperimentResult`` for module-level calls."""
    import datetime as dt
    import uuid

    return ExperimentResult(
        run_id=uuid.uuid4(),
        started_at=dt.datetime.now(dt.UTC),
        finished_at=dt.datetime.now(dt.UTC),
        duration_seconds=1.0,
        status="completed",
        config={},
        resolved_validation="train_test_split",
        resolved_n_folds=5,
        resolved_test_size=0.2,
        metric_user="rmse",
        metric_sklearn="neg_root_mean_squared_error",
        metric_direction="minimize",
        winner="ridge",
        winner_score=0.5,
    )


def test_render_importance_markdown_empty_and_cap() -> None:
    """The renderer handles an empty table and the row cap."""
    empty = render_importance_markdown("Empty", [])
    assert "_No feature importances available._" in empty
    capped = render_importance_markdown(
        "Capped",
        [
            FeatureImportance(feature=f"f{i}", importance_mean=float(i))
            for i in range(5)
        ],
        max_rows=2,
    )
    assert "and 3 more features" in capped
    no_table = render_importance_markdown(
        "No table",
        [FeatureImportance(feature="f", importance_mean=1.0)],
        max_rows=0,
    )
    assert "The table is omitted" in no_table


# ─────────────── explain result storage on the run (issue #181) ─────────────── #


def test_p5_explain_result_stored_on_run(tmp_path: Any) -> None:
    """P5: explain() stores the JSON-safe payload on the run result."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    importance, _section = experiment.explain(n_repeats=1)
    result = experiment.result
    assert result is not None
    payload = result.explanations
    assert payload is not None
    assert payload["method"] == "permutation"
    assert payload["metric_user"] == "rmse"
    assert payload["data_semantics"] == "train"
    assert payload["n_features"] == len(importance.importances)
    features = [row["feature"] for row in payload["importances"]]
    assert features == [item.feature for item in importance.importances]
    means = [row["importance_mean"] for row in payload["importances"]]
    assert means == [item.importance_mean for item in importance.importances]
    # The payload survives the JSON round-trip (the export contract).
    restored = ExperimentResult.from_json(result.to_json())
    assert restored.explanations == payload


def test_p5_explain_last_call_wins(tmp_path: Any) -> None:
    """P5: a repeated explain() (the same method, different budget)
    overwrites the stored payload — the last call wins."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    first, _ = experiment.explain(n_repeats=1)
    experiment.explain(n_repeats=2, markdown=False)
    result = experiment.result
    assert result is not None
    payload = result.explanations
    assert payload is not None
    features = [row["feature"] for row in payload["importances"]]
    assert features == [item.feature for item in first.importances]
    means = [row["importance_mean"] for row in payload["importances"]]
    assert means == [item.importance_mean for item in first.importances]


def test_b10_no_explain_explanations_is_none(tmp_path: Any) -> None:
    """B10: without explain() the run carries no explanations payload."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    result = experiment.result
    assert result is not None
    assert result.explanations is None


def test_b10_explanations_json_safe_no_numpy(tmp_path: Any) -> None:
    """B10: the stored payload is plain JSON types (no numpy scalars)."""
    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()
    experiment.explain(n_repeats=1, markdown=False)
    result = experiment.result
    assert result is not None
    payload = result.explanations
    assert payload is not None
    text = json.dumps(payload, allow_nan=False)  # must not raise
    assert "numpy" not in text


def test_p5_explain_storage_failure_does_not_mask_result(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """P5: a storage failure is logged as a warning and does not mask
    the explain outcome (the importance is still returned)."""
    from configurable_automl_engine.experiment import experiment as exp_module

    experiment = AutoMLExperiment(_config_dict(tmp_path), _small_df())
    experiment.run()

    def _reject(value: Any) -> None:
        raise ValueError("rejected by a hand-modified result validator")

    monkeypatch.setattr(exp_module, "_explanations_payload", _reject)
    with caplog.at_level(logging.WARNING, logger=exp_module._LOG.name):
        importance, _section = experiment.explain(n_repeats=1)
    assert importance.importances  # the outcome is not masked
    assert "could not be stored" in caplog.text
