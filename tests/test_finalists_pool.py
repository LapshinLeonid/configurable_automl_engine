"""
Unit tests for the Top-K finalists pool construction.

They cover ``select_finalists`` from ``training_engine/component.py``:

• the corridor δ in user semantics (direction from ``user_direction``):
  multiplicative for errors, additive for signed score metrics;
• the hard ``top_k_candidates`` cap and the explicit 'corridor ∩ top_k'
  combination;
• the optional soft family top-up (``enforce_family_diversity``);
• exclusion of disqualified algorithms (circuit breaker);
• determinism: ordering by 'raw' score, ties resolved by configuration order.

The ``results`` format matches the ``phase_results`` contract: 'algorithm →
(raw score, parameters)', where the raw score is in optimizer semantics
(for error metrics it is inverted, and ``to_user_value`` returns the
natural value).
"""

from __future__ import annotations

from typing import Any

import pytest

from configurable_automl_engine.training_engine.component import (
    FAMILY_DIVERSITY_MULTIPLIER_DEFAULT,
    PhaseResult,
    select_finalists,
)

R = dict[str, PhaseResult]

FAMILIES = {
    "elasticnet": "linear",
    "ridge": "linear",
    "lasso": "linear",
    "svr": "kernel",
    "gaussian_process_regression": "kernel",
    "random_forest": "ensemble",
    "gradient_boosting": "ensemble",
    "extra_trees": "ensemble",
}


# ──────────────────────────────────────────────────────────────────────────────
# Positive: example from the specification and basic corridors
# ──────────────────────────────────────────────────────────────────────────────
def test_positive_example_from_issue_statement():
    """ElasticNet (0.081), SVR (0.085), Random Forest (0.089) → pool at δ=0.15.

    The rmse metric (min-better) in auto mode → multiplicative:
    threshold = 0.081 × 1.15 = 0.09315; all three candidates fit inside
    the corridor.
    """
    results: R = {
        "elasticnet": PhaseResult(score=-0.081, params={"alpha": 0.1}),
        "svr": PhaseResult(score=-0.085, params={"C": 1.0}),
        "random_forest": PhaseResult(score=-0.089, params={"n_estimators": 100}),
    }
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.15, top_k_candidates=3
    )
    assert pool == ["elasticnet", "svr", "random_forest"]


def test_min_better_multiplicative_corridor_includes_within_delta():
    """min-better: candidates with error ≤ best×(1+δ) enter the pool."""
    results: R = {
        "ridge": PhaseResult(score=-0.05, params={}),
        "svr": PhaseResult(score=-0.055, params={}),
        "random_forest": PhaseResult(score=-0.06, params={}),
    }
    # Threshold = 0.05 × 1.2 = 0.06: SVR (0.055) and RF (0.06, exactly on the boundary) inside.
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.2, top_k_candidates=3
    )
    assert pool == ["ridge", "svr", "random_forest"]


def test_max_better_multiplicative_corridor_explicit_mode():
    """max-better: threshold = best×(1−δ); explicit multiplicative for r2."""
    results: R = {
        "ridge": PhaseResult(score=0.90, params={}),
        "svr": PhaseResult(score=0.85, params={}),
        "random_forest": PhaseResult(score=0.80, params={}),
    }
    # Threshold = 0.90 × 0.85 = 0.765: 0.85 and 0.80 inside, 0.70 outside.
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_delta=0.15,
        corridor_mode="multiplicative",
        top_k_candidates=3,
    )
    assert pool == ["ridge", "svr", "random_forest"]


def test_auto_mode_selects_multiplicative_for_errors():
    """auto for an error metric (rmse) → multiplicative: a candidate beyond the threshold is outside the pool."""
    results: R = {
        "elasticnet": PhaseResult(score=-0.081, params={}),
        "xgboosting": PhaseResult(score=-0.09, params={}),
    }
    # Threshold = 0.081 × 1.05 = 0.08505: 0.09 > threshold → xgboosting outside the pool.
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.05, corridor_mode="auto"
    )
    assert pool == ["elasticnet"]


def test_auto_mode_selects_additive_for_score_metric():
    """auto for r2 (a signed score metric) → additive, not multiplicative."""
    results: R = {
        "a": PhaseResult(score=-0.5, params={}),
        "b": PhaseResult(score=-0.55, params={}),
    }
    # additive: range = 0.05, threshold = 0.15×0.05 = 0.0075 → b outside the pool.
    pool = select_finalists(results, metric_user="r2", corridor_delta=0.15)
    assert pool == ["a"]


# ──────────────────────────────────────────────────────────────────────────────
# Negative: candidates outside the corridor and weak families
# ──────────────────────────────────────────────────────────────────────────────
def test_negative_model_outside_corridor_excluded():
    """A model whose CV score lies outside the corridor does not enter the pool (min-better)."""
    results: R = {
        "elasticnet": PhaseResult(score=-0.081, params={}),
        "svr": PhaseResult(score=-0.085, params={}),
        "random_forest": PhaseResult(score=-0.089, params={}),
        "xgboosting": PhaseResult(score=-0.20, params={}),  # 0.20 > 0.09315 → outside the corridor
    }
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.15, top_k_candidates=3
    )
    assert pool == ["elasticnet", "svr", "random_forest"]


def test_negative_model_outside_corridor_excluded_max_better():
    """max-better: a model whose r2 is below the best×(1−δ) threshold does not enter the pool."""
    results: R = {
        "ridge": PhaseResult(score=0.90, params={}),
        "svr": PhaseResult(score=0.85, params={}),
        "random_forest": PhaseResult(score=0.70, params={}),  # 0.70 < 0.765 → outside the corridor
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_delta=0.15,
        corridor_mode="multiplicative",
        top_k_candidates=3,
    )
    assert pool == ["ridge", "svr"]


def test_negative_family_diversity_disabled_no_topup():
    """enforce_family_diversity=false: a weak model from another family is not topped up.

    The base pool contains only linear models (a single family); the weak
    SVR from another family, being outside the base corridor, is not added.
    """
    results: R = {
        "elasticnet": PhaseResult(score=-0.081, params={}),
        "ridge": PhaseResult(score=-0.082, params={}),
        "svr": PhaseResult(score=-0.095, params={}),  # outside the base corridor (0.095 > 0.09315)
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=3,
        enforce_family_diversity=False,
        algorithm_families=FAMILIES,
    )
    assert pool == ["elasticnet", "ridge"]


# ──────────────────────────────────────────────────────────────────────────────
# Boundary: top_k, negative scores, a single candidate, disqualification
# ──────────────────────────────────────────────────────────────────────────────
def test_boundary_top_k_equals_one():
    """top_k=1: the pool is a single leader — equivalent to the current behavior."""
    results: R = {
        "ridge": PhaseResult(score=0.90, params={}),
        "svr": PhaseResult(score=0.85, params={}),
        "random_forest": PhaseResult(score=0.80, params={}),
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_mode="multiplicative",
        top_k_candidates=1,
    )
    assert pool == ["ridge"]


def test_boundary_pool_capped_at_top_k():
    """A pool larger than top_k: the hard cap truncates it to top_k_candidates."""
    results: R = {
        "ridge": PhaseResult(score=0.90, params={}),
        "svr": PhaseResult(score=0.85, params={}),
        "random_forest": PhaseResult(score=0.80, params={}),
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_delta=0.5,
        corridor_mode="multiplicative",
        top_k_candidates=2,
    )
    assert pool == ["ridge", "svr"]


def test_boundary_single_valid_algorithm():
    """A single valid algorithm — the pool has one candidate."""
    pool = select_finalists(
        {"ridge": PhaseResult(score=0.7, params={})}, metric_user="r2", top_k_candidates=3
    )
    assert pool == ["ridge"]


def test_boundary_corridor_gives_less_than_top_k():
    """The corridor yields fewer candidates than top_k: the pool keeps what exists.

    No top-up beyond the corridor happens (without the optional rule).
    """
    results: R = {
        "elasticnet": PhaseResult(score=-0.081, params={}),
        "xgboosting": PhaseResult(score=-0.20, params={}),  # outside the corridor
    }
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.15, top_k_candidates=3
    )
    assert pool == ["elasticnet"]


def test_boundary_negative_r2_additive_form():
    """Negative r2 scores: the additive form is correct and the boundary is included.

    range = 0.1, threshold = 0.5×0.1 = 0.05: |−0.55 − (−0.5)| = 0.05 is exactly
    on the boundary → included (documented tie-break); |−0.6+0.5| = 0.1 —
    outside.
    """
    results: R = {
        "a": PhaseResult(score=-0.5, params={}),
        "b": PhaseResult(score=-0.55, params={}),
        "c": PhaseResult(score=-0.6, params={}),
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_delta=0.5,
        corridor_mode="additive",
        top_k_candidates=3,
    )
    assert pool == ["a", "b"]


def test_boundary_multiplicative_empty_with_negative_best_falls_back_to_additive():
    """The corridor is empty for max-better with a negative leader → additive.

    Explicit multiplicative for r2: threshold = −0.5×0.85 = −0.425, the
    leader −0.5 does not pass (empty corridor) → the additive form is
    chosen; at δ=1.0 both candidates enter the pool.
    """
    results: R = {
        "a": PhaseResult(score=-0.5, params={}),
        "b": PhaseResult(score=-0.6, params={}),
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_delta=1.0,
        corridor_mode="multiplicative",
        top_k_candidates=3,
    )
    assert pool == ["a", "b"]


def test_boundary_mae_near_zero_additive_form():
    """MAE≈0: an explicit additive form (multiplicative is incorrect here).

    User scores 0.001/0.0015/0.002; range = 0.001, threshold = 0.5×0.001 = 0.0005.
    """
    results: R = {
        "a": PhaseResult(score=-0.001, params={}),
        "b": PhaseResult(score=-0.0015, params={}),
        "c": PhaseResult(score=-0.002, params={}),
    }
    pool = select_finalists(
        results,
        metric_user="mae",
        corridor_delta=0.5,
        corridor_mode="additive",
        top_k_candidates=3,
    )
    assert pool == ["a", "b"]


def test_boundary_disqualified_leader_excluded():
    """A disqualified algorithm is excluded from the pool (even the leader)."""
    results: R = {
        "svr": PhaseResult(score=0.95, params={}),
        "ridge": PhaseResult(score=0.90, params={}),
        "random_forest": PhaseResult(score=0.88, params={}),
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_mode="multiplicative",
        top_k_candidates=3,
        disqualified={"svr"},
    )
    assert pool == ["ridge", "random_forest"]


def test_boundary_tie_breaks_by_configuration_order():
    """Equal scores → configuration order (as in select_winner)."""
    results: R = {
        "random_forest": PhaseResult(score=0.5, params={}),
        "extra_trees": PhaseResult(score=0.5, params={}),
        "elasticnet": PhaseResult(score=0.5, params={}),
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_mode="multiplicative",
        top_k_candidates=3,
    )
    assert pool == ["random_forest", "extra_trees", "elasticnet"]


def test_boundary_equal_scores_on_corridor_edge_all_included():
    """Scores exactly on the corridor edge are all included (inclusive tie-break)."""
    results: R = {
        "ridge": PhaseResult(score=-0.05, params={}),
        "svr": PhaseResult(score=-0.06, params={}),
        "random_forest": PhaseResult(score=-0.06, params={}),
    }
    # Threshold = 0.05 × 1.2 = 0.06: both candidates are exactly on the boundary.
    pool = select_finalists(
        results, metric_user="rmse", corridor_delta=0.2, top_k_candidates=3
    )
    assert pool == ["ridge", "svr", "random_forest"]


# ──────────────────────────────────────────────────────────────────────────────
# Algorithm families: optional soft top-up
# ──────────────────────────────────────────────────────────────────────────────
def test_family_diversity_topup_from_extended_corridor():
    """The best representative of a missing family is topped up from the extended corridor.

    Base pool: [elasticnet, ridge] (linear). SVR (kernel, 0.095) is outside
    the base corridor (0.095 > 0.09315) but inside the extended one
    (0.081 × 1.225 = 0.0992) → added; the pool reaches top_k=3.
    """
    results: R = {
        "elasticnet": PhaseResult(score=-0.081, params={}),
        "ridge": PhaseResult(score=-0.082, params={}),
        "svr": PhaseResult(score=-0.095, params={}),
        "random_forest": PhaseResult(score=-0.11, params={}),  # outside even the extended corridor
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=3,
        enforce_family_diversity=True,
        algorithm_families=FAMILIES,
    )
    assert pool == ["elasticnet", "ridge", "svr"]


def test_family_diversity_respects_top_k_cap():
    """The family top-up does not exceed top_k_candidates."""
    results: R = {
        "elasticnet": PhaseResult(score=-0.081, params={}),
        "ridge": PhaseResult(score=-0.082, params={}),
        "svr": PhaseResult(score=-0.095, params={}),
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=2,
        enforce_family_diversity=True,
        algorithm_families=FAMILIES,
    )
    assert pool == ["elasticnet", "ridge"]


def test_family_diversity_allowed_families_restriction():
    """allowed_families restricts which families can be topped up."""
    results: R = {
        "elasticnet": PhaseResult(score=-0.081, params={}),
        "ridge": PhaseResult(score=-0.082, params={}),
        "svr": PhaseResult(score=-0.095, params={}),
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=3,
        enforce_family_diversity=True,
        algorithm_families=FAMILIES,
        allowed_families={"ensemble"},  # kernel is forbidden → no top-up
    )
    assert pool == ["elasticnet", "ridge"]


def test_family_diversity_no_topup_when_two_families_in_pool():
    """The pool already has two families — the top-up rule is not applied."""
    results: R = {
        "elasticnet": PhaseResult(score=-0.081, params={}),  # linear
        "svr": PhaseResult(score=-0.085, params={}),  # kernel
        "ridge": PhaseResult(score=-0.095, params={}),  # linear, outside the base corridor
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=3,
        enforce_family_diversity=True,
        algorithm_families=FAMILIES,
    )
    assert pool == ["elasticnet", "svr"]


def test_family_diversity_unmapped_algorithms_own_family():
    """Algorithms without a family mapping get their own family (determinism)."""
    results: R = {
        "elasticnet": PhaseResult(score=-0.081, params={}),  # linear
        "ridge": PhaseResult(score=-0.082, params={}),  # linear
        "custom_algo": PhaseResult(score=-0.095, params={}),  # its own family
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=3,
        enforce_family_diversity=True,
        algorithm_families={"elasticnet": "linear", "ridge": "linear"},
    )
    assert pool == ["elasticnet", "ridge", "custom_algo"]


def test_family_diversity_without_families_map():
    """Without a family map each algorithm is its own family.

    The base pool [a] holds a single 'family'; the top-up adds the best
    candidate from the extended corridor (b, 0.095 ≤ 0.081×1.225 = 0.0992).
    """
    results: R = {
        "a": PhaseResult(score=-0.081, params={}),
        "b": PhaseResult(score=-0.095, params={}),
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=2,
        enforce_family_diversity=True,
        algorithm_families=None,
    )
    assert pool == ["a", "b"]


def test_family_diversity_default_multiplier_constant():
    """The default extended-corridor multiplier is 1.5 (from the specification)."""
    assert FAMILY_DIVERSITY_MULTIPLIER_DEFAULT == 1.5


# ──────────────────────────────────────────────────────────────────────────────
# Edge cases and failure modes
# ──────────────────────────────────────────────────────────────────────────────
def test_edge_empty_results_raises_runtime_error():
    """An empty phase_results → RuntimeError (per the specification).

    Note: ``select_winner`` raises ``ValueError`` for an empty dict, but
    the specification requires ``RuntimeError`` for the finalists pool.
    """
    with pytest.raises(RuntimeError, match="empty results"):
        select_finalists({}, metric_user="r2")


def test_edge_all_disqualified_raises_runtime_error():
    """All algorithms are disqualified → RuntimeError."""
    results: R = {"svr": PhaseResult(score=0.95, params={}), "ridge": PhaseResult(score=0.90, params={})}
    with pytest.raises(RuntimeError, match="disqualified"):
        select_finalists(
            results,
            metric_user="r2",
            disqualified={"svr", "ridge"},
        )


def test_edge_invalid_top_k_candidates():
    with pytest.raises(ValueError, match="top_k_candidates"):
        select_finalists({"a": PhaseResult(score=0.5, params={})}, metric_user="r2", top_k_candidates=0)


def test_edge_invalid_corridor_delta():
    with pytest.raises(ValueError, match="corridor_delta"):
        select_finalists({"a": PhaseResult(score=0.5, params={})}, metric_user="r2", corridor_delta=-0.1)


def test_edge_invalid_corridor_mode():
    with pytest.raises(ValueError, match="corridor_mode"):
        select_finalists(
            {"a": PhaseResult(score=0.5, params={})},
            metric_user="r2",
            corridor_mode="weird",  # type: ignore[arg-type]
        )


def test_edge_invalid_family_diversity_multiplier():
    with pytest.raises(ValueError, match="family_diversity_multiplier"):
        select_finalists(
            {"a": PhaseResult(score=0.5, params={})}, metric_user="r2", family_diversity_multiplier=0.0
        )


def test_edge_unknown_metric_raises():
    with pytest.raises(ValueError, match="not implemented"):
        select_finalists({"a": PhaseResult(score=0.5, params={})}, metric_user="unknown_custom_metric")


def test_edge_determinism_repeated_calls():
    """Determinism: repeated calls return an identical pool and order."""
    results: R = {
        "elasticnet": PhaseResult(score=-0.081, params={}),
        "ridge": PhaseResult(score=-0.082, params={}),
        "svr": PhaseResult(score=-0.095, params={}),
        "random_forest": PhaseResult(score=-0.11, params={}),
    }
    kwargs = {
        "metric_user": "rmse",
        "corridor_delta": 0.15,
        "top_k_candidates": 3,
        "enforce_family_diversity": True,
        "algorithm_families": FAMILIES,
    }
    first = select_finalists(results, **kwargs)
    for _ in range(5):
        assert select_finalists(results, **kwargs) == first


def test_edge_zero_delta_only_leader():
    """δ=0: the corridor collapses to exact equality — only the leader."""
    results: R = {
        "ridge": PhaseResult(score=0.90, params={}),
        "svr": PhaseResult(score=0.90, params={}),  # exactly equal to the leader
        "random_forest": PhaseResult(score=0.89, params={}),
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_mode="multiplicative",
        corridor_delta=0.0,
        top_k_candidates=3,
    )
    # Leader + candidate with an equal score (boundary included); 0.89 — outside.
    assert pool == ["ridge", "svr"]


def test_edge_min_better_and_max_better_metrics_parametrized():
    """The direction comes from user_direction: min- and max-better metrics."""
    # min-better (rmse): leader with the smallest error
    min_results: R = {"good": PhaseResult(score=-0.05, params={}), "bad": PhaseResult(score=-0.10, params={})}
    # Threshold = 0.05 × 1.5 = 0.075: bad (0.10) outside the corridor.
    assert select_finalists(
        min_results, metric_user="rmse", corridor_delta=0.5, top_k_candidates=3
    ) == ["good"]
    # max-better (r2): leader with the highest score
    max_results: R = {"good": PhaseResult(score=0.90, params={}), "bad": PhaseResult(score=0.80, params={})}
    # Threshold = 0.90 × 0.5 = 0.45: bad (0.80) inside; the threshold is wide.
    assert select_finalists(
        max_results,
        metric_user="r2",
        corridor_delta=0.5,
        corridor_mode="multiplicative",
        top_k_candidates=3,
    ) == ["good", "bad"]


# ──────────────────────────────────────────────────────────────────────────────
# Code-review regressions
# ──────────────────────────────────────────────────────────────────────────────
def test_regression_disqualified_generator_not_consumed():
    """A generator in ``disqualified`` is not consumed by the dict-comprehension.

    Blocking review finding: previously ``set(disqualified)`` was computed
    inside the comprehension, the generator was consumed on the first
    iteration, and disqualified names 'leaked' into the pool
    (reproduced as ['c'] instead of ['a', 'b']). Now the iterable is
    materialized before the comprehension.
    """
    results: R = {
        "a": PhaseResult(score=0.90, params={}),
        "b": PhaseResult(score=0.85, params={}),
        "c": PhaseResult(score=0.95, params={}),  # best by CV, but disqualified
    }
    pool = select_finalists(
        results,
        metric_user="r2",
        corridor_mode="multiplicative",
        top_k_candidates=3,
        disqualified=(name for name in ("c",)),
    )
    assert pool == ["a", "b"]


def test_regression_pool_family_not_allowed_no_topup():
    """The only pool family is not allowed → the top-up is not performed.

    Blocking review finding: behavior aligned with the documentation — if
    the only pool family is absent from ``allowed_families``, the soft
    top-up is not applied (previously it returned ['elasticnet', 'svr']
    instead of ['elasticnet']).
    """
    results: R = {
        "elasticnet": PhaseResult(score=-0.081, params={}),  # linear
        "svr": PhaseResult(score=-0.095, params={}),  # kernel, inside the extended corridor
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=3,
        enforce_family_diversity=True,
        algorithm_families=FAMILIES,
        allowed_families={"ensemble"},  # the pool's linear family is forbidden
    )
    assert pool == ["elasticnet"]


def test_regression_pool_family_allowed_and_missing_allowed():
    """The pool family and the added family are allowed → the top-up runs.

    Positive control for ``test_regression_pool_family_not_allowed_no_topup``:
    the top-up happens when the only pool family is allowed by
    ``allowed_families`` and the missing family is allowed too.
    """
    results: R = {
        "elasticnet": PhaseResult(score=-0.081, params={}),  # linear
        "ridge": PhaseResult(score=-0.082, params={}),  # linear
        "svr": PhaseResult(score=-0.095, params={}),  # kernel, inside the extended corridor
        "random_forest": PhaseResult(score=-0.11, params={}),  # ensemble, outside the extended corridor
    }
    pool = select_finalists(
        results,
        metric_user="rmse",
        corridor_delta=0.15,
        top_k_candidates=3,
        enforce_family_diversity=True,
        algorithm_families=FAMILIES,
        allowed_families={"linear", "kernel"},
    )
    assert pool == ["elasticnet", "ridge", "svr"]


def test_regression_auto_mode_best_near_zero_switches_to_additive():
    """auto with best≈0 (MAE≈0) switches to additive.

    The multiplicative form with best=0 yields a 0 threshold and collapses
    the pool to the leader; the additive form includes candidates within
    δ×range.
    """
    results: R = {
        "a": PhaseResult(score=0.0, params={}),  # MAE ≈ 0 (raw -0.0 after inversion)
        "b": PhaseResult(score=-1e-4, params={}),
        "c": PhaseResult(score=-2e-4, params={}),
    }
    # additive: range = 2e-4, threshold = 0.5×2e-4 = 1e-4 → b, on the boundary, is included.
    pool = select_finalists(
        results, metric_user="mae", corridor_delta=0.5, top_k_candidates=3
    )
    assert pool == ["a", "b"]