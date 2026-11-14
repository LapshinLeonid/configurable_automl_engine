"""Multi-split (repeated) statistics in honest scores and the Markdown report.

Covered behavior:

* the new optional result fields serialize/deserialize and an old payload
  without them still restores;
* with ``n_repeats = 1`` the report keeps the classic scalar rendering and
  has no Sanity Gate table;
* with ``n_repeats > 1`` the winner and OOF estimates render as
  ``aggregate ± std`` together with the repeat count;
* an absent/non-finite spread renders as ``н/д`` instead of failing;
* the Sanity Gate section renders the RMSE spread and marks circuit values
  as aggregated only when the audit actually records per-repeat aggregates;
* additional metrics are aggregated by the same policy as the winner score.
"""

from __future__ import annotations

import datetime as dt
import uuid
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from optuna.trial import TrialState
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from configurable_automl_engine.common.definitions import (
    MetricDirection,
    RunStatus,
)
from configurable_automl_engine.common.metrics import (
    get_scorer_object,
    to_user_value,
)
from configurable_automl_engine.experiment import AutoMLExperiment, ExperimentResult
from configurable_automl_engine.experiment.reporting import MarkdownReporter
from configurable_automl_engine.trainer_scoring import score_on_validation_splits
from configurable_automl_engine.training_engine import artifact as artifact_module
from configurable_automl_engine.training_engine.artifact import _pooled_raw_oof_metric
from configurable_automl_engine.training_engine.component import _winner_repeat_scores
from configurable_automl_engine.training_engine.select_steps import _sanity_gate_section
from configurable_automl_engine.training_engine.state import (
    PhaseResult,
    RobustWinnerResult,
)
from configurable_automl_engine.validation import aggregate_split_scores

UTC = dt.UTC


def _ts(seconds: float = 0.0) -> dt.datetime:
    return dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC) + dt.timedelta(seconds=seconds)


def _result_kwargs(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "run_id": uuid.uuid4(),
        "label": "multi-split run",
        "started_at": _ts(),
        "finished_at": _ts(10.0),
        "duration_seconds": 10.0,
        "status": RunStatus.completed,
        "config": {"general": {"split_aggregation": "median"}},
        "resolved_validation": "k_fold",
        "resolved_n_folds": 4,
        "resolved_test_size": 0.2,
        "random_state": 42,
        "metric_user": "rmse",
        "metric_sklearn": "neg_root_mean_squared_error",
        "metric_direction": MetricDirection.minimize,
        "winner": "ridge",
        "winner_score": 0.0721,
        "model_path": "model.pkl",
        "oof_score": 0.0802,
    }
    values.update(overrides)
    return values


def _repeated_result(**overrides: Any) -> ExperimentResult:
    values = _result_kwargs(
        n_repeats=3,
        score_std=0.0067,
        oof_score_std=0.0102,
        additional_metrics={"mae": 0.05, "r2": 0.9},
    )
    values.update(overrides)
    return ExperimentResult(**values)


def test_single_split_report_keeps_the_classic_scalar_shape() -> None:
    result = ExperimentResult(**_result_kwargs())
    report = MarkdownReporter().report(result)

    assert "±" not in report
    assert "repeats" not in report
    assert "## Sanity Gate" not in report
    assert "- Winner score: 0.072 " in report
    assert "- OOF score: **0.080**" in report


def test_single_split_result_round_trip_has_optional_defaults() -> None:
    result = ExperimentResult(**_result_kwargs())
    payload = result.model_dump(mode="json")
    restored = ExperimentResult.from_json(result.to_json())

    assert restored == result
    assert restored.n_repeats is None
    assert restored.score_std is None
    assert restored.oof_score_std is None
    # The optional multi-split fields do not change the classic JSON shape.
    for field in ("score_std", "oof_score_std", "n_repeats"):
        assert field not in payload


def test_explicit_single_repeat_is_normalized_to_the_classic_shape() -> None:
    result = ExperimentResult(**_result_kwargs(n_repeats=1))
    payload = result.model_dump(mode="json")

    assert result.n_repeats is None
    assert "n_repeats" not in payload
    assert ExperimentResult.from_json(result.to_json()) == result


def test_repeated_report_shows_aggregate_spread_and_repeat_count() -> None:
    report = MarkdownReporter().report(_repeated_result())

    assert "- Winner score: 0.072 ± 0.007 " in report
    assert "median of 3 repeats" in report
    assert "### Honest score estimates (median of 3 repeats)" in report
    assert "- OOF score: **0.080 ± 0.010**" in report
    assert "- Reference — winner (selection) score: 0.072 ± 0.007" in report
    assert "mae: 0.050" in report
    assert "r2: 0.900" in report


def test_holdout_and_bbc_stay_scalar_in_a_repeated_report() -> None:
    report = MarkdownReporter().report(
        _repeated_result(holdout_score=0.09, bbc_score=0.085)
    )

    assert "- Hold-out score: **0.090**" in report
    assert "- BBC score: **0.085**" in report
    assert "0.090 ±" not in report
    assert "0.085 ±" not in report


def test_mean_aggregation_is_named_in_the_report() -> None:
    result = _repeated_result(
        config={"general": {"split_aggregation": "mean"}},
    )
    report = MarkdownReporter().report(result)

    assert "mean of 3 repeats" in report


def test_repeated_result_round_trip_preserves_new_fields() -> None:
    result = _repeated_result()
    payload = result.model_dump(mode="json")

    assert payload["n_repeats"] == 3
    assert payload["score_std"] == pytest.approx(0.0067)
    assert payload["oof_score_std"] == pytest.approx(0.0102)
    assert ExperimentResult.from_json(result.to_json()) == result


def test_winner_score_std_input_alias_is_supported() -> None:
    values = _result_kwargs()
    values["winner_score_std"] = 0.123
    result = ExperimentResult(**values)

    assert result.score_std == pytest.approx(0.123)
    assert result.winner_score_std == pytest.approx(0.123)
    assert result.model_dump(mode="json")["score_std"] == pytest.approx(0.123)


def test_missing_spread_renders_not_available_instead_of_failing() -> None:
    result = _repeated_result(score_std=None, oof_score_std=None)
    report = MarkdownReporter().report(result)

    assert "0.072 ± н/д" in report
    assert "0.080 ± н/д" in report


def test_large_spread_is_visible() -> None:
    report = MarkdownReporter().report(
        _repeated_result(score_std=12.345, oof_score_std=98.765)
    )

    assert "0.072 ± 12.345" in report
    assert "0.080 ± 98.765" in report


def _robust_result() -> RobustWinnerResult:
    return RobustWinnerResult(
        winner_algo="ridge",
        mode="active",
        pool=["ridge"],
        audit={},
        fallback_used=False,
        winner_rmse_oof=0.1,
    )


def test_gate_section_keeps_the_classic_keys() -> None:
    section = _sanity_gate_section(_robust_result(), "warn", 0.5)

    # M6 itself adds no repeat metadata (that belongs to M4), and a
    # single-split section keeps at least the classic keys. Using a subset
    # assertion keeps this test valid after the M4 branch merges its own
    # n_repeats/split_aggregation keys.
    assert {
        "strategy",
        "generalization_gap_mode",
        "pool",
        "audit",
        "disqualified",
        "fallback_used",
        "winner_rmse_oof",
        "overhead_seconds",
    } <= set(section)
    assert "rmse_oof_std" not in section
    assert "winner_rmse_oof_std" not in section


def test_winner_repeat_scores_ignores_pruned_intermediate_values() -> None:
    pruned = SimpleNamespace(
        state=TrialState.PRUNED,
        value=-0.01,
        user_attrs={},
    )
    complete = SimpleNamespace(
        state=TrialState.COMPLETE,
        value=-0.5,
        user_attrs={"repeat_scores": [-0.45, -0.5, -0.55]},
    )
    record = PhaseResult(
        score=-0.5,
        params={},
        trials=[pruned, complete],  # type: ignore[list-item]
    )

    assert _winner_repeat_scores(record) == (-0.45, -0.5, -0.55)


def test_markerless_gate_section_is_not_rendered_for_repeats() -> None:
    result = _repeated_result(
        sanity_gate={
            "gate": {
                "pool": ["ridge"],
                "audit": {
                    "ridge": {
                        "is_valid": True,
                        "reasons": [],
                        "soft_warnings": [],
                        "diversity_ratio": 0.9,
                        "unique_ratio": 0.95,
                        "dead_features_count": 0,
                        "generalization_gap": 2.5,
                        "generalization_gap_infinite": False,
                    }
                },
                "winner_rmse_oof": 0.08,
            }
        }
    )

    assert "## Sanity Gate" not in MarkdownReporter().report(result)


def test_m4_single_split_section_stays_out_of_the_report() -> None:
    # M4 (PR 170) always records n_repeats in the section and in every audit
    # element, including the single-split value 1. That must not turn on the
    # new Sanity Gate table for n_repeats = 1.
    result = ExperimentResult(
        **_result_kwargs(
            n_repeats=1,
            sanity_gate={
                "gate": {
                    "strategy": "sanity_gate",
                    "n_repeats": 1,
                    "split_aggregation": "median",
                    "pool": ["ridge"],
                    "audit": {
                        "ridge": {
                            "is_valid": True,
                            "reasons": [],
                            "soft_warnings": [],
                            "diversity_ratio": 0.9,
                            "unique_ratio": 0.95,
                            "dead_features_count": 0,
                            "generalization_gap": 2.5,
                            "generalization_gap_infinite": False,
                            "n_repeats": 1,
                        }
                    },
                    "disqualified": {},
                    "fallback_used": False,
                    "winner_rmse_oof": 0.0802,
                    "overhead_seconds": 0.5,
                }
            },
        )
    )
    report = MarkdownReporter().report(result)

    assert "## Sanity Gate" not in report
    assert "±" not in report


def test_sanity_gate_renders_aggregated_circuit_values() -> None:
    result = _repeated_result(
        sanity_gate={
            "gate": {
                "strategy": "sanity_gate",
                "pool": ["ridge"],
                "audit": {
                    "ridge": {
                        "is_valid": True,
                        "reasons": [],
                        "soft_warnings": ["Circuit D warning"],
                        "diversity_ratio": 0.9,
                        "diversity_ratio_std": 0.02,
                        "unique_ratio": 0.95,
                        "unique_ratio_std": None,
                        "dead_features_count": 0,
                        "generalization_gap": 2.5,
                        "generalization_gap_std": 0.4,
                    }
                },
                "disqualified": {},
                "fallback_used": False,
                "winner_rmse_oof": 0.0802,
                "winner_rmse_oof_std": 0.0102,
                "overhead_seconds": 0.5,
            }
        }
    )
    report = MarkdownReporter().report(result)

    assert "## Sanity Gate" in report
    assert "### Phase gate" in report
    assert "medians of the per-repeat values" in report
    assert "Winner RMSE_oof: **0.080 ± 0.010**" in report
    assert "| ridge | yes | 0.900 ± 0.020 | 0.950 |" in report
    assert "2.500 ± 0.400" in report
    assert "Circuit D warning" in report


def test_sanity_gate_non_finite_values_render_not_available() -> None:
    # The result boundary rejects NaN/±inf sections, but a legacy/persisted
    # section may still carry a raw non-finite value: the renderer must stay
    # total and show the explicit not-available marker.
    result = _repeated_result()
    result.sanity_gate = {
        "gate": {
            "audit": {
                "ridge": {
                    "is_valid": False,
                    "reasons": ["Circuit D"],
                    "diversity_ratio": float("nan"),
                    "unique_ratio": None,
                    "dead_features_count": 1,
                    "generalization_gap": float("inf"),
                    "generalization_gap_std": float("nan"),
                }
            },
            "winner_rmse_oof": None,
        }
    }
    report = MarkdownReporter().report(result)

    assert "н/д" in report
    assert "inf" not in report
    assert "nan" not in report


def test_section_repeats_without_audit_aggregation_uses_neutral_caption() -> None:
    result = _repeated_result(
        sanity_gate={
            "gate": {
                "n_repeats": 3,
                "split_aggregation": "median",
                "audit": {
                    "ridge": {
                        "is_valid": True,
                        "reasons": [],
                        "soft_warnings": [],
                        "diversity_ratio": 0.9,
                        "unique_ratio": 0.95,
                        "dead_features_count": 0,
                        "generalization_gap": 2.5,
                        "generalization_gap_infinite": False,
                    }
                },
            }
        }
    )
    report = MarkdownReporter().report(result)

    assert "## Sanity Gate" in report
    assert "not aggregated over repeats" in report
    assert "medians of the per-repeat values" not in report


def test_winner_rmse_spread_does_not_claim_aggregated_circuits() -> None:
    result = _repeated_result(
        sanity_gate={
            "gate": {
                "winner_rmse_oof": 0.0802,
                "winner_rmse_oof_std": 0.0102,
                "audit": {
                    "ridge": {
                        "is_valid": True,
                        "reasons": [],
                        "soft_warnings": [],
                        "diversity_ratio": 0.9,
                        "unique_ratio": 0.95,
                        "dead_features_count": 0,
                        "generalization_gap": 2.5,
                        "generalization_gap_infinite": False,
                    }
                },
            }
        }
    )
    report = MarkdownReporter().report(result)

    assert "not aggregated over repeats" in report
    assert "`RMSE_oof` is aggregated over the repeats" in report
    assert "medians of the per-repeat values" not in report


def test_sanity_gate_handles_non_numeric_overhead_and_string_reasons() -> None:
    result = _repeated_result(
        sanity_gate={
            "gate": {
                "n_repeats": 3,
                "split_aggregation": "median",
                "audit": {
                    "ridge": {
                        "is_valid": False,
                        "reasons": "Circuit D",
                        "soft_warnings": "soft warning",
                        "diversity_ratio": 0.9,
                        "unique_ratio": 0.95,
                        "dead_features_count": 1,
                        "generalization_gap": None,
                        "generalization_gap_infinite": True,
                    }
                },
                "overhead_seconds": "unavailable",
            }
        }
    )
    report = MarkdownReporter().report(result)

    assert "- Overhead: н/д" in report
    assert "∞ (infinite)" in report
    assert "Circuit D; soft warning" in report


def _make_steps(feature_selection_active: bool, X_for_estimate: Any) -> list[Any]:
    return [("scaler", StandardScaler()), ("model", Ridge(alpha=0.5))]


def _score(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int,
    n_repeats: int | None,
    split_aggregation: str,
) -> Any:
    return score_on_validation_splits(
        X,
        y,
        _make_steps,
        feature_selection_active=False,
        metric="rmse",
        additional_metrics=["mae", "r2"],
        validation_strategy="k_fold",
        n_folds=4,
        test_size=0.2,
        random_state=random_state,
        algorithm="ridge",
        n_repeats=n_repeats,
        split_aggregation=split_aggregation,
    )


@pytest.mark.parametrize("aggregation", ["median", "mean"])
def test_additional_metrics_follow_the_selection_aggregation(
    aggregation: str,
) -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    y = pd.Series(2.0 * X["a"] - 0.5 * X["b"] + rng.normal(size=40))

    combined = _score(
        X,
        y,
        random_state=42,
        n_repeats=3,
        split_aggregation=aggregation,
    )

    # The per-repeat seeds are random_state + r: reproducing them one by one
    # gives the reference values for the additional-metric aggregation.
    singles = [
        _score(
            X,
            y,
            random_state=42 + repeat,
            n_repeats=None,
            split_aggregation="median",
        )
        for repeat in range(3)
    ]
    for metric_name in ("mae", "r2"):
        expected = aggregate_split_scores(
            [single.additional_scores[metric_name] for single in singles],
            aggregation,
        )
        assert combined.additional_scores[metric_name] == pytest.approx(expected)


def test_repeated_run_propagates_statistics_into_the_result_and_report(
    tmp_path: Any,
) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["target"] = X[:, 0] * 2.0 + rng.normal(size=30) * 0.5
    config = {
        "general": {
            "comparison_metric": "rmse",
            "validation_strategy": "k_fold",
            "n_folds": 3,
            "n_repeats": 3,
            "holdout_fraction": None,
            "path_to_model": str(tmp_path / "model.pkl"),
            "additional_metrics": ["mae"],
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

    result = AutoMLExperiment(config, df, target="target").run()

    assert result.n_repeats == 3
    assert result.score_std is not None and result.score_std >= 0.0
    assert result.oof_score_std is not None and result.oof_score_std >= 0.0
    assert set(result.additional_metrics) == {"mae"}
    assert ExperimentResult.from_json(result.to_json()) == result
    report = MarkdownReporter().report(result)
    assert "median of 3 repeats" in report
    assert "±" in report


def test_score_std_pairs_with_the_hpo_winner_score_when_holdout_is_reserved(
    tmp_path: Any, monkeypatch: Any
) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["target"] = X[:, 0] * 2.0 + rng.normal(size=120) * 0.5
    config = {
        "general": {
            "comparison_metric": "rmse",
            "validation_strategy": "k_fold",
            "n_folds": 3,
            "n_repeats": 3,
            "holdout_fraction": 0.2,
            "split_aggregation": "median",
            "path_to_model": str(tmp_path / "model.pkl"),
            "phases": [
                {"name": "search", "n_trials": 3, "action": "all_algorithms"},
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

    captured: dict[str, Any] = {}
    original = artifact_module._add_repeat_statistics

    def _spy(
        result: dict[str, Any],
        trainer: Any,
        winner_repeat_scores: tuple[float, ...] | None = None,
        oof_score_std: float | None = None,
    ) -> None:
        captured["scores"] = winner_repeat_scores
        original(result, trainer, winner_repeat_scores, oof_score_std)

    monkeypatch.setattr(artifact_module, "_add_repeat_statistics", _spy)
    result = AutoMLExperiment(config, df, target="target").run()

    winner_repeat_scores = captured["scores"]
    assert winner_repeat_scores is not None
    assert len(winner_repeat_scores) == 3
    assert result.score_std == pytest.approx(np.std(winner_repeat_scores))
    # Optimizer semantics for the rmse metric are negated; std is unchanged.
    assert result.winner_score == pytest.approx(abs(np.median(winner_repeat_scores)))
    assert result.n_repeats == 3


def test_repeated_gate_report_does_not_claim_aggregated_circuits(
    tmp_path: Any,
) -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["target"] = X[:, 0] * 2.0 + rng.normal(size=40) * 0.5
    config = {
        "general": {
            "comparison_metric": "rmse",
            "validation_strategy": "k_fold",
            "n_folds": 3,
            "n_repeats": 2,
            "holdout_fraction": None,
            "split_aggregation": "median",
            "path_to_model": str(tmp_path / "model.pkl"),
            "phases": [
                {"name": "search", "n_trials": 2, "action": "all_algorithms"},
                {
                    "name": "gate_select",
                    "n_trials": 1,
                    "action": "select",
                    "strategy": "sanity_gate",
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

    result = AutoMLExperiment(config, df, target="target").run()

    section = (result.sanity_gate or {})["gate_select"]
    assert "winner_rmse_oof_std" in section or "rmse_oof_std" in section
    report = MarkdownReporter().report(result)
    assert "not aggregated over repeats" in report
    assert "`RMSE_oof` is aggregated over the repeats" in report
    assert "medians of the per-repeat values" not in report


def test_oof_score_std_uses_the_comparison_metric_not_rmse(
    tmp_path: Any, monkeypatch: Any
) -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["target"] = X[:, 0] * 2.0 + rng.normal(size=80) * 0.5
    config = {
        "general": {
            "comparison_metric": "mae",
            "validation_strategy": "k_fold",
            "n_folds": 3,
            "n_repeats": 3,
            "holdout_fraction": None,
            "split_aggregation": "median",
            "path_to_model": str(tmp_path / "model.pkl"),
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

    captured: dict[str, Any] = {}
    original = artifact_module._oof_score_with_std_in_user_metric

    def _capture_trainer(metric_user: str, trainer: Any, y: pd.Series) -> Any:
        captured["trainer"] = trainer
        return original(metric_user, trainer, y)

    monkeypatch.setattr(
        artifact_module, "_oof_score_with_std_in_user_metric", _capture_trainer
    )
    experiment = AutoMLExperiment(config, df, target="target")
    result = experiment.run()
    trainer = captured["trainer"]
    vectors = trainer.oof_predictions_by_repeat_
    assert isinstance(vectors, list) and len(vectors) == 3
    y = np.asarray(df["target"], dtype=float)
    scorer = get_scorer_object("mae", global_y=y)
    user_scores = []
    for vector in vectors:
        raw = _pooled_raw_oof_metric(scorer, y, vector)
        assert raw is not None
        user_scores.append(float(to_user_value("mae", raw)))

    assert result.oof_score == pytest.approx(float(np.median(user_scores)))
    assert result.oof_score_std == pytest.approx(float(np.std(user_scores)))
    assert result.oof_score_std != pytest.approx(trainer.oof_score_std_ or 0.0)
    report = MarkdownReporter().report(result)
    assert (
        f"OOF score: **{result.oof_score:.3f} ± {result.oof_score_std:.3f}**" in report
    )
