"""Multi-split aggregation in the Sanity Gate and winner ranking.

Coverage:

1. Circuits A/B evaluate every repeat and aggregate by the median: identical
   repeats keep the single-split verdict at any ``R``, while a majority of
   degenerate repeats still fails.
2. Circuit D reports the median per-repeat ``RMSE_oof / RMSE_full`` ratio,
   excludes non-computable repeats and keeps NaN for total non-computability.
3. Mixed per-repeat circuit D decisions are aggregated into one decision (the
   warn mode never disqualifies because of the gap magnitude).
4. The finalist ranking and ``n_repeats = 1`` keep the classic behavior.
5. ``_to_candidate`` forwards the per-repeat vectors and the report section
   states the aggregation that produced the metrics.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import configurable_automl_engine.training_engine.finalists as finalists_mod
import configurable_automl_engine.training_engine.select_steps as select_steps_mod
from configurable_automl_engine.common.metrics import oof_rmse
from configurable_automl_engine.training_engine.config_parser import (
    Config,
    HPOPhaseCfg,
    ValidationStrategy,
)
from configurable_automl_engine.training_engine.finalists import _to_candidate
from configurable_automl_engine.training_engine.robust_winner import (
    select_robust_winner,
)
from configurable_automl_engine.training_engine.sanity_circuits import (
    check_generalization_gap,
    check_generalization_gap_repeats,
    check_unique,
)
from configurable_automl_engine.training_engine.sanity_gate import (
    ModelSanityGate,
)
from configurable_automl_engine.training_engine.sanity_repeats import (
    check_diversity_repeats,
    check_unique_repeats,
)
from configurable_automl_engine.training_engine.select_steps import (
    _gate_disqualifications,
    _rank_sanity_candidates,
    _serialize_audit_result,
)
from configurable_automl_engine.training_engine.state import (
    FinalistCandidate,
    PhaseContext,
    PhaseResult,
    PreparedData,
)


def _with_rmse(base: np.ndarray, target_rmse: float, seed: int) -> np.ndarray:
    """A prediction vector with the requested RMSE relative to ``base``."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(base))
    noise = noise / np.sqrt(np.mean(noise**2)) * target_rmse
    return base + noise


def _candidate(
    name: str,
    rmse_oof: float,
    y_oof: np.ndarray,
    *,
    y_oof_by_repeat: list[np.ndarray] | None = None,
    y_full: np.ndarray | None = None,
    rmse_full: float | None = None,
) -> FinalistCandidate:
    return FinalistCandidate(
        algo_name=name,
        model=None,
        params={},
        cv_score=-rmse_oof,
        rmse_oof=rmse_oof,
        y_pred_oof=y_oof,
        y_pred_full=y_oof if y_full is None else y_full,
        rmse_full=rmse_full,
        y_pred_oof_by_repeat=y_oof_by_repeat,
        n_repeats=len(y_oof_by_repeat) if y_oof_by_repeat else 1,
    )


def _gate_without_other_circuits(**kwargs: Any) -> ModelSanityGate:
    return ModelSanityGate(
        check_dead_features=False,
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Median aggregation for circuits A/B
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("repeats", [2, 3, 5])
def test_identical_repeats_keep_the_single_split_diversity_and_verdict(
    repeats: int,
):
    y = np.linspace(-5.0, 5.0, 200)
    honest = y.copy()
    gate = _gate_without_other_circuits(
        adaptive_diversity=False,
        min_prediction_diversity=0.5,
        check_unique=False,
        check_generalization_gap=False,
    )

    single = gate.check(y=y, y_pred_oof=honest)
    multi = gate.check(
        y=y,
        y_pred_oof=honest,
        y_pred_oof_by_repeat=[honest] * repeats,
    )

    assert single.diversity_ratio == pytest.approx(1.0)
    assert multi.diversity_ratio == pytest.approx(single.diversity_ratio)
    assert multi.is_valid == single.is_valid
    assert multi.n_repeats == repeats


def test_diversity_median_rejects_a_majority_of_degenerate_repeats():
    y = np.linspace(-5.0, 5.0, 200)
    honest = y.copy()
    constant = np.full_like(y, float(np.mean(y)))
    gate = _gate_without_other_circuits(
        adaptive_diversity=False,
        min_prediction_diversity=0.15,
        check_unique=False,
        check_generalization_gap=False,
    )

    single = gate.check(y=y, y_pred_oof=honest)
    diluted = gate.check(
        y=y,
        y_pred_oof=honest,
        y_pred_oof_by_repeat=[honest, constant, constant, constant, constant],
    )

    assert single.is_valid is True
    assert diluted.is_valid is False
    assert diluted.diversity_ratio == pytest.approx(0.0)
    assert any("Circuit A" in reason for reason in diluted.reasons)


def test_diversity_median_boundary_passes_at_any_repeat_count():
    y = np.linspace(-5.0, 5.0, 200)
    honest = y.copy()
    constant = np.full_like(y, float(np.mean(y)))
    gate = _gate_without_other_circuits(
        adaptive_diversity=False,
        min_prediction_diversity=0.5,
        check_unique=False,
        check_generalization_gap=False,
    )

    result = gate.check(
        y=y,
        y_pred_oof=honest,
        y_pred_oof_by_repeat=[honest, constant],
    )

    assert result.diversity_ratio == pytest.approx(0.5)
    assert result.is_valid is True


@pytest.mark.parametrize("repeats", [2, 3, 5])
def test_identical_repeats_keep_the_single_split_unique_ratio(repeats: int):
    y = np.linspace(0.0, 100.0, 200)
    oof = np.round(y, decimals=1)
    # 190 distinct predictions: the single split passes the 90% bar.
    oof[-10:] = oof[0]
    valid = np.isfinite(y) & np.isfinite(oof)
    gate = _gate_without_other_circuits(
        check_diversity=False,
        min_unique_count=5,
        min_unique_ratio=0.9,
        check_generalization_gap=False,
    )

    single_ratio, single_reasons, _ = check_unique(
        y, oof, valid, min_unique_count=5, min_unique_ratio=0.9
    )
    multi = gate.check(
        y=y,
        y_pred_oof=oof,
        y_pred_oof_by_repeat=[oof] * repeats,
    )

    assert single_reasons == []
    assert single_ratio == pytest.approx(0.95)
    assert multi.unique_ratio == pytest.approx(single_ratio)
    assert multi.is_valid is True


def test_unique_median_rejects_a_majority_of_constant_repeats():
    y = np.linspace(0.0, 100.0, 200)
    honest = np.round(y, decimals=1)
    constant = np.full_like(y, float(np.mean(y)))
    gate = _gate_without_other_circuits(
        check_diversity=False,
        min_unique_count=5,
        min_unique_ratio=0.9,
        check_generalization_gap=False,
    )

    single = gate.check(y=y, y_pred_oof=honest)
    rejected = gate.check(
        y=y,
        y_pred_oof=honest,
        y_pred_oof_by_repeat=[honest, constant, constant],
    )

    assert single.is_valid is True
    assert rejected.is_valid is False
    assert rejected.unique_ratio < 0.05
    assert any("Circuit B" in reason for reason in rejected.reasons)


def test_gate_reports_the_number_of_repeat_vectors():
    y = np.linspace(-1.0, 1.0, 50)
    full = _with_rmse(y, 0.2, 1)
    repeats = [_with_rmse(y, 0.1, 2), _with_rmse(y, 0.3, 3)]
    gate = ModelSanityGate(
        check_diversity=False,
        check_unique=False,
        check_dead_features=False,
    )

    result = gate.check(
        y=y,
        y_pred_oof=repeats[0],
        y_pred_full=full,
        y_pred_oof_by_repeat=repeats,
    )

    assert result.n_repeats == 2


# ──────────────────────────────────────────────────────────────────────────────
#  Circuit D median ratio
# ──────────────────────────────────────────────────────────────────────────────


def test_generalization_gap_is_the_median_of_per_repeat_ratios():
    y = np.linspace(-1.0, 1.0, 120)
    full = _with_rmse(y, 0.2, 1)
    repeats = [
        _with_rmse(y, 0.1, 2),  # ratio 0.5
        _with_rmse(y, 0.3, 3),  # ratio 1.5
        _with_rmse(y, 0.5, 4),  # ratio 2.5
    ]

    gap, reasons, warnings = check_generalization_gap_repeats(
        y, repeats, full, max_generalization_gap=5.0
    )

    assert reasons == []
    assert warnings == []
    assert gap == pytest.approx(1.5, rel=1e-6)


def test_generalization_gap_excludes_non_computable_repeats():
    y = np.linspace(-1.0, 1.0, 80)
    full = _with_rmse(y, 0.2, 1)
    repeats = [
        np.full_like(y, np.nan),
        _with_rmse(y, 0.4, 2),  # ratio 2.0
        np.full_like(y, np.nan),
    ]

    gap, reasons, _ = check_generalization_gap_repeats(
        y, repeats, full, max_generalization_gap=5.0
    )

    assert reasons == []
    assert gap == pytest.approx(2.0, rel=1e-6)


def test_generalization_gap_is_nan_when_every_repeat_is_non_computable():
    y = np.linspace(-1.0, 1.0, 20)
    full = _with_rmse(y, 0.2, 1)
    repeats = [np.full_like(y, np.nan), np.full_like(y, np.nan)]

    gap, reasons, warnings = check_generalization_gap_repeats(
        y, repeats, full, max_generalization_gap=5.0
    )

    assert np.isnan(gap)
    assert warnings == []
    assert any("in any repeat" in reason for reason in reasons)


def test_single_repeat_preserves_the_classic_gap_and_messages():
    y = np.linspace(-1.0, 1.0, 80)
    full = _with_rmse(y, 0.2, 1)
    oof = _with_rmse(y, 0.4, 2)

    classic = check_generalization_gap(y, oof, full, max_generalization_gap=1.5)
    repeated = check_generalization_gap_repeats(
        y, [oof], full, max_generalization_gap=1.5
    )

    assert repeated == classic


def test_mixed_repeat_decisions_are_aggregated_once_in_warn_mode():
    y = np.linspace(-1.0, 1.0, 100)
    full = _with_rmse(y, 0.2, 1)
    # Ratios 1.0 and 3.0: the median (2.0) is above the threshold even though
    # one repeat alone would have passed.
    repeats = [_with_rmse(y, 0.2, 2), _with_rmse(y, 0.6, 3)]
    gate = _gate_without_other_circuits(
        check_diversity=False,
        check_unique=False,
        generalization_gap_mode="warn",
        max_generalization_gap=1.5,
    )

    result = gate.check(
        y=y,
        y_pred_oof=repeats[0],
        y_pred_full=full,
        y_pred_oof_by_repeat=repeats,
    )

    assert result.is_valid is True
    assert result.generalization_gap == pytest.approx(2.0, rel=1e-6)
    assert any("median gap=2.000" in warning for warning in result.soft_warnings)


def test_repeat_majority_below_threshold_keeps_the_gap_clean():
    y = np.linspace(-1.0, 1.0, 100)
    full = _with_rmse(y, 0.2, 1)
    repeats = [
        _with_rmse(y, 0.2, 2),  # ratio 1.0
        _with_rmse(y, 0.24, 3),  # ratio 1.2
        _with_rmse(y, 0.6, 4),  # ratio 3.0
    ]
    gate = _gate_without_other_circuits(
        check_diversity=False,
        check_unique=False,
        generalization_gap_mode="warn",
        max_generalization_gap=1.5,
    )

    result = gate.check(
        y=y,
        y_pred_oof=repeats[0],
        y_pred_full=full,
        y_pred_oof_by_repeat=repeats,
    )

    assert result.is_valid is True
    assert result.generalization_gap == pytest.approx(1.2, rel=1e-6)
    assert result.soft_warnings == []


def test_aggregated_gap_disqualifies_in_reject_mode_and_reaches_breaker():
    y = np.linspace(-1.0, 1.0, 100)
    full = _with_rmse(y, 0.2, 1)
    repeats = [_with_rmse(y, 0.4, 2), _with_rmse(y, 0.6, 3)]
    candidate = _candidate(
        "overfit",
        rmse_oof=0.5,
        y_oof=repeats[0],
        y_oof_by_repeat=repeats,
        y_full=full,
        rmse_full=float(oof_rmse(y, full)),
    )
    gate = _gate_without_other_circuits(
        check_diversity=False,
        check_unique=False,
        generalization_gap_mode="reject",
        max_generalization_gap=1.5,
    )

    result = select_robust_winner(
        {"overfit": candidate},
        pd.DataFrame(np.zeros((len(y), 2))),
        pd.Series(y),
        gate,
        mode="active",
    )

    assert "overfit" in result.disqualified
    assert any("median gap" in reason for reason in result.disqualified["overfit"])
    merged, disqualified_this_step = _gate_disqualifications({}, result)
    assert "overfit" in merged
    assert disqualified_this_step == {"overfit"}


# ──────────────────────────────────────────────────────────────────────────────
#  Ranking, candidate assembly and report
# ──────────────────────────────────────────────────────────────────────────────


def test_winner_is_ranked_by_the_aggregated_rmse_not_the_representative_split():
    y = np.linspace(-1.0, 1.0, 100)
    # The first candidate has the better aggregate but a worse representative
    # OOF vector; the ranking must follow ``rmse_oof`` (the aggregate).
    better_aggregate = _candidate(
        "better_aggregate",
        rmse_oof=0.10,
        y_oof=_with_rmse(y, 0.60, 1),
    )
    better_split = _candidate(
        "better_split",
        rmse_oof=0.20,
        y_oof=_with_rmse(y, 0.10, 2),
    )
    gate = _gate_without_other_circuits(
        check_diversity=False,
        check_unique=False,
        check_generalization_gap=False,
    )

    result = select_robust_winner(
        {"better_aggregate": better_aggregate, "better_split": better_split},
        pd.DataFrame(np.zeros((len(y), 2))),
        pd.Series(y),
        gate,
        mode="active",
    )

    assert result.winner_algo == "better_aggregate"
    assert result.winner_rmse_oof == pytest.approx(0.10)


def test_n_repeats_one_keeps_the_classic_gate_result():
    y = np.linspace(-1.0, 1.0, 80)
    oof = _with_rmse(y, 0.2, 1)
    full = _with_rmse(y, 0.1, 2)
    gate = ModelSanityGate(check_dead_features=False)

    classic = gate.check(y=y, y_pred_oof=oof, y_pred_full=full)
    explicit = gate.check(
        y=y,
        y_pred_oof=oof,
        y_pred_full=full,
        y_pred_oof_by_repeat=[oof],
    )

    assert explicit == classic
    assert explicit.n_repeats == 1


def test_to_candidate_forwards_repeat_vectors_and_scores():
    y = np.linspace(0.0, 1.0, 12)
    vectors = [np.zeros(len(y)), np.ones(len(y))]

    class _Trainer:
        pipeline = "fitted-pipeline"
        oof_score_ = 0.25
        oof_score_std_ = 0.05
        oof_predictions_ = np.full(len(y), 0.5)
        oof_predictions_by_repeat_ = vectors

        def predict(self, X: Any) -> np.ndarray:
            return np.zeros(len(X))

    candidate = _to_candidate(
        "ridge", _Trainer(), {"alpha": 0.5}, 0.1, pd.DataFrame({"x": y}), pd.Series(y)
    )

    assert candidate.y_pred_oof_by_repeat == vectors
    assert candidate.rmse_oof == pytest.approx(0.25)


def test_serialized_audit_states_the_number_of_repeats():
    gate = _gate_without_other_circuits(
        check_diversity=False,
        check_unique=False,
        check_generalization_gap=False,
    )
    y = np.linspace(-1.0, 1.0, 20)
    oof = _with_rmse(y, 0.2, 1)
    audit = gate.check(
        y=y,
        y_pred_oof=oof,
        y_pred_oof_by_repeat=[oof, oof],
    )
    serialized = _serialize_audit_result(audit, "warn")
    assert serialized["n_repeats"] == 2
    assert serialized["n_valid_repeats"] == 2


def test_multi_split_gate_is_deterministic():
    y = np.linspace(-1.0, 1.0, 60)
    full = _with_rmse(y, 0.2, 1)
    repeats = [_with_rmse(y, 0.1, 2), _with_rmse(y, 0.3, 3)]
    gate = _gate_without_other_circuits(
        check_diversity=False,
        check_unique=False,
    )

    first = gate.check(
        y=y,
        y_pred_oof=repeats[0],
        y_pred_full=full,
        y_pred_oof_by_repeat=repeats,
    )
    second = gate.check(
        y=y,
        y_pred_oof=repeats[0],
        y_pred_full=full,
        y_pred_oof_by_repeat=repeats,
    )

    assert first == second


# ──────────────────────────────────────────────────────────────────────────────
#  Invalidated repeats are excluded from the gate
# ──────────────────────────────────────────────────────────────────────────────


class _RepeatTrainer:
    """A trainer stub carrying the M3 per-repeat metadata."""

    def __init__(
        self,
        y: np.ndarray,
        vectors: list[np.ndarray],
        scores: list[float],
        *,
        full_rmse: float = 0.2,
    ) -> None:
        self.pipeline = self
        self.oof_predictions_ = vectors[0]
        self.oof_score_ = float(np.median(scores))
        self.oof_score_std_ = 0.1
        self.oof_predictions_by_repeat_ = vectors
        self.oof_repeat_scores_ = scores
        self._full_pred = _with_rmse(y, full_rmse, 99)

    def predict(self, X: Any) -> np.ndarray:
        return self._full_pred[: len(X)]

    def save(self, path: Any) -> None:
        Path(path).write_bytes(b"fake-model")


def test_to_candidate_filters_repeats_invalidated_by_scoring():
    y = np.linspace(-1.0, 1.0, 100)
    valid = _with_rmse(y, 0.2, 1)  # gap = 1.0 against RMSE_full = 0.2
    invalid = _with_rmse(y, 1.1, 2)  # gap = 5.5, but the repeat score is inf
    trainer = _RepeatTrainer(y, [valid, invalid], [0.2, float("inf")])

    candidate = _to_candidate(
        "ridge",
        trainer,
        {},
        0.2,
        pd.DataFrame({"x": y}),
        pd.Series(y),
    )
    gate = _gate_without_other_circuits(
        check_diversity=False,
        check_unique=False,
        generalization_gap_mode="reject",
        max_generalization_gap=1.5,
    )
    result = gate.check(
        y=y,
        y_pred_oof=candidate.y_pred_oof,
        y_pred_full=candidate.y_pred_full,
        y_pred_oof_by_repeat=candidate.y_pred_oof_by_repeat,
    )

    assert candidate.n_repeats == 2
    assert candidate.has_valid_oof is True
    assert candidate.y_pred_oof_by_repeat == [valid]
    assert result.n_repeats == 1
    assert result.n_valid_repeats == 1
    assert result.generalization_gap == pytest.approx(1.0, rel=1e-6)
    assert result.is_valid is True

    selected = select_robust_winner(
        {"ridge": candidate},
        pd.DataFrame({"x": y}),
        pd.Series(y),
        gate,
        mode="active",
    )
    selected_audit = selected.audit["ridge"]
    assert selected_audit.n_repeats == 2
    assert selected_audit.n_valid_repeats == 1
    serialized = _serialize_audit_result(selected_audit, "warn")
    assert serialized["n_repeats"] == 2
    assert serialized["n_valid_repeats"] == 1


# ──────────────────────────────────────────────────────────────────────────────
#  Circuit D decision is applied to the median only
# ──────────────────────────────────────────────────────────────────────────────


def test_circuit_d_finite_median_ignores_an_infinite_repeat_in_reject_mode():
    y = np.linspace(-1.0, 1.0, 100)
    full = _with_rmse(y, 0.2, 1)
    repeats = [
        y.copy(),  # RMSE_oof ≈ 0 -> inf ratio
        _with_rmse(y, 0.2, 2),  # ratio 1.0
        _with_rmse(y, 0.3, 3),  # ratio 1.5 (median, equality passes)
    ]
    gate = _gate_without_other_circuits(
        check_diversity=False,
        check_unique=False,
        generalization_gap_mode="reject",
        max_generalization_gap=1.5,
    )

    result = gate.check(
        y=y,
        y_pred_oof=repeats[1],
        y_pred_full=full,
        y_pred_oof_by_repeat=repeats,
    )

    assert result.generalization_gap == pytest.approx(1.5, rel=1e-6)
    assert result.is_valid is True
    assert result.reasons == []
    assert any("possible target leakage" in warning for warning in result.soft_warnings)


def test_circuit_d_infinite_median_rejects_in_reject_mode():
    y = np.linspace(-1.0, 1.0, 100)
    full = _with_rmse(y, 0.2, 1)
    repeats = [
        y.copy(),
        y.copy(),
        _with_rmse(y, 0.2, 2),
    ]
    gate = _gate_without_other_circuits(
        check_diversity=False,
        check_unique=False,
        generalization_gap_mode="reject",
        max_generalization_gap=1.5,
    )

    result = gate.check(
        y=y,
        y_pred_oof=repeats[2],
        y_pred_full=full,
        y_pred_oof_by_repeat=repeats,
    )

    assert np.isinf(result.generalization_gap)
    assert result.is_valid is False
    assert any("median gap=inf" in reason for reason in result.reasons)


# ──────────────────────────────────────────────────────────────────────────────
#  Adaptive diversity and empty/constant channels
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("sign", "expected_valid"),
    [(1.0, True), (-1.0, False)],
)
def test_adaptive_diversity_uses_median_r2(sign: float, expected_valid: bool):
    y = np.linspace(-1.0, 1.0, 100)
    repeats = [sign * y + _with_rmse(np.zeros_like(y), 0.05, i) for i in range(3)]
    gate = _gate_without_other_circuits(
        adaptive_diversity=True,
        check_unique=False,
        check_generalization_gap=False,
    )

    result = gate.check(
        y=y,
        y_pred_oof=repeats[0],
        y_pred_oof_by_repeat=repeats,
    )

    assert result.is_valid is expected_valid
    if not expected_valid:
        assert any("median R²_oof" in reason for reason in result.reasons)


def test_adaptive_diversity_median_can_mix_positive_and_negative_repeats():
    y = np.linspace(-1.0, 1.0, 100)
    positive = y + _with_rmse(np.zeros_like(y), 0.05, 1)
    negative = -y + _with_rmse(np.zeros_like(y), 0.05, 2)
    gate = _gate_without_other_circuits(
        adaptive_diversity=True,
        check_unique=False,
        check_generalization_gap=False,
    )

    result = gate.check(
        y=y,
        y_pred_oof=positive,
        y_pred_oof_by_repeat=[positive, positive, negative],
    )

    assert result.is_valid is True


def test_diversity_constant_target_is_a_soft_warning_in_repeat_path():
    y = np.zeros(50)
    repeats = [np.ones(50), np.arange(50, dtype=float)]
    gate = _gate_without_other_circuits(
        adaptive_diversity=True,
        check_unique=False,
        check_generalization_gap=False,
    )

    result = gate.check(
        y=y,
        y_pred_oof=repeats[0],
        y_pred_oof_by_repeat=repeats,
    )

    assert result.is_valid is True
    assert np.isnan(result.diversity_ratio)
    assert any("target variance is zero" in warning for warning in result.soft_warnings)


def test_diversity_without_finite_pairs_in_any_repeat_is_a_reason():
    y = np.linspace(-1.0, 1.0, 50)
    repeats = [np.full_like(y, np.nan), np.full_like(y, np.nan)]
    gate = _gate_without_other_circuits(
        adaptive_diversity=True,
        check_unique=False,
        check_generalization_gap=False,
    )

    result = gate.check(
        y=y,
        y_pred_oof=repeats[0],
        y_pred_oof_by_repeat=repeats,
    )

    assert result.is_valid is False
    assert any("in any repeat" in reason for reason in result.reasons)


def test_unique_without_finite_pairs_in_any_repeat_is_a_reason():
    y = np.linspace(-1.0, 1.0, 50)
    repeats = [np.full_like(y, np.nan), np.full_like(y, np.nan)]
    gate = _gate_without_other_circuits(
        check_diversity=False,
        check_unique=True,
        check_generalization_gap=False,
    )

    result = gate.check(
        y=y,
        y_pred_oof=repeats[0],
        y_pred_oof_by_repeat=repeats,
    )

    assert result.is_valid is False
    assert any("in any repeat" in reason for reason in result.reasons)


# ──────────────────────────────────────────────────────────────────────────────
#  End-to-end select step with repeated finalists
# ──────────────────────────────────────────────────────────────────────────────


def _gate_config(*, audit_budget: float = 0.0) -> Config:
    return Config.model_validate(
        {
            "general": {
                "comparison_metric": "mae",
                "validation_strategy": "train_test_split",
                "n_repeats": 3,
                "phases": [
                    {"name": "coarse", "n_trials": 1, "action": "all_algorithms"},
                    {
                        "name": "gate",
                        "n_trials": 1,
                        "action": "select",
                        "strategy": "sanity_gate",
                        "output": "winner",
                    },
                ],
                "path_to_model": "model.pkl",
                "sanity_gate": {"audit_time_budget_seconds": audit_budget},
            },
            "algorithms": {"ridge": {"enable": True}},
            "oversampling": {"data_oversampling": False},
        }
    )


def _prepared_for(y: np.ndarray) -> PreparedData:
    return PreparedData(
        X=pd.DataFrame({"x": np.linspace(-1.0, 1.0, len(y))}),
        y=pd.Series(y),
        metric_user="mae",
        metric_sklearn="mae",
        resolved_validation=ValidationStrategy.train_test_split,
        resolved_n_folds=5,
        resolved_test_size=0.2,
        categorical_features=[],
        numerical_features=["x"],
    )


def _gate_phase() -> HPOPhaseCfg:
    return HPOPhaseCfg(
        name="gate",
        n_trials=1,
        action="select",
        strategy="sanity_gate",
        output="winner",
    )


def test_select_step_reports_repeat_aggregation_end_to_end():
    y = np.linspace(-1.0, 1.0, 60)
    train_vectors = [
        _with_rmse(y, 0.15, 1),
        _with_rmse(y, 0.20, 2),
        _with_rmse(y, 0.25, 3),
    ]
    trainer = _RepeatTrainer(y, train_vectors, [0.15, 0.20, 0.25])
    prepared = _prepared_for(y)
    ctx = PhaseContext(working_set={"ridge": PhaseResult(score=-0.2, params={})})

    with (
        patch.object(select_steps_mod, "_build_finalists_pool", return_value=["ridge"]),
        patch.object(finalists_mod, "_fit_finalist", return_value=trainer),
    ):
        outcome = select_steps_mod._select_sanity_gate(
            _gate_phase(), _gate_config(), prepared, ctx
        )

    section = outcome.sanity_gate_section
    assert section is not None
    assert section["n_repeats"] == 3
    assert section["split_aggregation"] == "median"
    assert section["audit"]["ridge"]["n_repeats"] == 3
    assert section["audit"]["ridge"]["n_valid_repeats"] == 3


def test_select_step_reports_training_repeats_when_every_audit_is_skipped():
    y = np.linspace(-1.0, 1.0, 60)
    train_vectors = [
        _with_rmse(y, 0.15, 1),
        _with_rmse(y, 0.20, 2),
        _with_rmse(y, 0.25, 3),
    ]
    trainer = _RepeatTrainer(y, train_vectors, [0.15, 0.20, 0.25])
    prepared = _prepared_for(y)
    ctx = PhaseContext(working_set={"ridge": PhaseResult(score=-0.2, params={})})

    with (
        patch.object(select_steps_mod, "_build_finalists_pool", return_value=["ridge"]),
        patch.object(finalists_mod, "_fit_finalist", return_value=trainer),
    ):
        outcome = select_steps_mod._select_sanity_gate(
            _gate_phase(),
            _gate_config(audit_budget=1e-9),
            prepared,
            ctx,
        )

    section = outcome.sanity_gate_section
    assert section is not None
    assert section["n_repeats"] == 3
    assert section["audit"]["ridge"]["n_repeats"] == 3
    assert section["audit"]["ridge"]["n_valid_repeats"] == 0
    assert section["audit"]["ridge"]["is_valid"] is False


def test_all_invalid_repeats_are_rejected_not_audited_fall_back():
    y = np.linspace(-1.0, 1.0, 100)
    invalid = y.copy()
    invalid[::2] = np.nan
    invalid_trainer = _RepeatTrainer(
        y,
        [invalid, invalid.copy()],
        [float("inf"), float("inf")],
    )
    invalid_trainer.oof_score_ = None
    invalid_candidate = _to_candidate(
        "invalid",
        invalid_trainer,
        {},
        -1.0,
        pd.DataFrame({"x": y}),
        pd.Series(y),
    )
    good_trainer = _RepeatTrainer(
        y,
        [_with_rmse(y, 0.2, 1), _with_rmse(y, 0.25, 2)],
        [0.2, 0.25],
    )
    good_candidate = _to_candidate(
        "good",
        good_trainer,
        {},
        -0.2,
        pd.DataFrame({"x": y}),
        pd.Series(y),
    )
    gate = _gate_without_other_circuits(
        check_diversity=False,
        check_unique=False,
        check_generalization_gap=False,
    )

    result = select_robust_winner(
        {"invalid": invalid_candidate, "good": good_candidate},
        pd.DataFrame({"x": y}),
        pd.Series(y),
        gate,
        mode="active",
    )

    assert invalid_candidate.has_valid_oof is False
    assert invalid_candidate.y_pred_oof_by_repeat is None
    assert invalid_candidate.rmse_oof is None
    assert result.winner_algo == "good"
    assert "invalid" in result.disqualified
    assert any(
        "no valid OOF repeats" in reason for reason in result.disqualified["invalid"]
    )
    assert result.audit["invalid"].n_repeats == 2
    assert result.audit["invalid"].n_valid_repeats == 0


def test_successful_audit_reports_the_trainer_repeat_count():
    y = np.linspace(-1.0, 1.0, 100)
    candidate = _candidate(
        "mixed",
        rmse_oof=0.2,
        y_oof=_with_rmse(y, 0.2, 1),
        y_oof_by_repeat=[
            _with_rmse(y, 0.2, 1),
            _with_rmse(y, 0.3, 2),
            _with_rmse(y, 0.4, 3),
        ],
    )
    gate = _gate_without_other_circuits(
        check_diversity=False,
        check_unique=False,
        check_generalization_gap=False,
    )

    result = select_robust_winner(
        {"mixed": candidate},
        pd.DataFrame({"x": y}),
        pd.Series(y),
        gate,
        mode="active",
    )

    assert result.audit["mixed"].n_repeats == 3
    assert result.audit["mixed"].n_valid_repeats == 3


# ──────────────────────────────────────────────────────────────────────────────
#  Input-contract negative tests
# ──────────────────────────────────────────────────────────────────────────────


def test_check_rejects_repeat_vector_length_mismatch():
    y = np.linspace(-1.0, 1.0, 20)
    gate = _gate_without_other_circuits()

    with pytest.raises(ValueError, match="same length"):
        gate.check(
            y=y,
            y_pred_oof=y,
            y_pred_oof_by_repeat=[np.zeros(len(y) - 1)],
        )


def test_check_rejects_non_1d_repeat_vector():
    y = np.linspace(-1.0, 1.0, 20)
    gate = _gate_without_other_circuits()

    with pytest.raises(ValueError, match="1-D"):
        gate.check(
            y=y,
            y_pred_oof=y,
            y_pred_oof_by_repeat=[np.zeros((len(y), 2))],
        )


def test_check_empty_repeat_sequence_follows_the_single_split_path():
    y = np.linspace(-1.0, 1.0, 20)
    full = _with_rmse(y, 0.2, 1)
    gate = _gate_without_other_circuits()

    single = gate.check(y=y, y_pred_oof=y, y_pred_full=full)
    empty = gate.check(
        y=y,
        y_pred_oof=y,
        y_pred_full=full,
        y_pred_oof_by_repeat=[],
    )

    assert empty == single
    assert empty.n_repeats == 1


def test_gate_uses_the_only_valid_repeat_instead_of_invalid_representative():
    y = np.linspace(-1.0, 1.0, 100)
    valid = _with_rmse(y, 0.2, 1)
    invalid = np.full_like(y, float(np.mean(y)))
    invalid[::2] = np.nan
    full = _with_rmse(y, 0.2, 2)
    gate = ModelSanityGate(
        check_dead_features=False,
        adaptive_diversity=False,
        min_prediction_diversity=0.5,
        min_unique_count=5,
        min_unique_ratio=0.9,
        generalization_gap_mode="reject",
        max_generalization_gap=1.5,
    )

    result = gate.check(
        y=y,
        y_pred_oof=invalid,
        y_pred_full=full,
        y_pred_oof_by_repeat=[valid],
    )

    assert result.n_repeats == 1
    assert result.diversity_ratio > 0.5
    assert result.unique_ratio > 0.9
    assert result.generalization_gap == pytest.approx(1.0, rel=1e-6)
    assert result.is_valid is True


def test_diversity_repeats_enforces_vector_length():
    y = np.linspace(-1.0, 1.0, 20)

    with pytest.raises(ValueError, match="same length"):
        check_diversity_repeats(
            y,
            [np.zeros(len(y) - 1)],
            adaptive=True,
            min_prediction_diversity=0.5,
        )


def test_unique_repeats_enforces_vector_length():
    y = np.linspace(-1.0, 1.0, 20)

    with pytest.raises(ValueError, match="same length"):
        check_unique_repeats(
            y,
            [np.zeros(len(y) - 1)],
            min_unique_count=1,
            min_unique_ratio=0.5,
        )


def test_no_valid_oof_candidate_sorts_after_circuit_failure_in_fallback():
    y = np.linspace(-1.0, 1.0, 100)
    weak = _candidate(
        "weak",
        rmse_oof=0.5,
        y_oof=np.full_like(y, float(np.mean(y))),  # diversity = 0 -> fails A
    )
    bad = replace(
        _candidate(
            "bad",
            rmse_oof=0.0,
            y_oof=np.full_like(y, np.nan),
            y_oof_by_repeat=[np.full_like(y, np.nan)],
        ),
        rmse_oof=None,
        has_valid_oof=False,
        n_repeats=2,
    )
    candidates = {"bad": bad, "weak": weak}
    gate = _gate_without_other_circuits(
        adaptive_diversity=False,
        min_prediction_diversity=0.5,
        check_unique=False,
        check_generalization_gap=False,
    )

    result = select_robust_winner(
        candidates,
        pd.DataFrame({"x": y}),
        pd.Series(y),
        gate,
        mode="active",
    )

    assert result.fallback_used is True
    assert result.winner_algo == "weak"
    assert _rank_sanity_candidates(result, candidates) == ["weak", "bad"]


def test_mixed_no_pairs_and_constant_target_returns_soft_warning():
    y = np.full(100, 5.0)
    constant = np.full(100, 5.0)
    no_pairs = np.full(100, np.nan)
    gate = ModelSanityGate(
        check_diversity=True,
        adaptive_diversity=False,
        min_prediction_diversity=0.5,
        check_unique=False,
        check_dead_features=False,
        check_generalization_gap=False,
    )

    single = gate.check(y=y, y_pred_oof=constant)
    mixed = gate.check(
        y=y,
        y_pred_oof=constant,
        y_pred_oof_by_repeat=[no_pairs, constant],
    )

    assert single.is_valid is True
    assert mixed.is_valid is True
    assert mixed.reasons == []
    assert any("target variance is zero" in warning for warning in mixed.soft_warnings)
