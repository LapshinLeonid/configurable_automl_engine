"""Unit tests of the adaptive complexity guard for tree/boosting spaces.

Covers:
1. Size-mode resolution by (n, p) including edge cases (n = 2..5, p = 1).
2. Rule application per algorithm/family for the small and medium modes:
   min_samples_leaf, max_depth, max_features, n_estimators (time-only),
   min_child_weight, subsample, colsample_bytree, reg_lambda.
3. User-range conflicts in both policies (user_wins / engine_wins).
4. Range collapse to a single value (parameter fixed).
5. Preservation of the int/float type and the log scale.
6. off/warn modes and the "ordinary" mode (spaces unchanged).
"""

from __future__ import annotations

import math

import pytest

from configurable_automl_engine.common.complexity_guard import (
    ComplexityGuardConfig,
    apply_complexity_guard,
    resolve_size_mode,
)
from configurable_automl_engine.common.hyperopt_defaults import (
    DEFAULT_SPACES,
    FloatSpace,
    IntSpace,
    SearchSpaceEntry,
)


def _entry(*args):
    return SearchSpaceEntry.model_validate(list(args))


def _default_cfg(**overrides) -> ComplexityGuardConfig:
    return ComplexityGuardConfig(**overrides)


# ─────────────────────────────── size mode ─────────────────────────────────── #


@pytest.mark.parametrize(
    "n, p, expected",
    [
        # small: n < 50
        (15, 5, "small"),
        (49, 5, "small"),
        # small: n / p < 5
        (50, 20, "small"),
        (20, 4, "small"),
        # medium: n < 300
        (100, 5, "medium"),
        (299, 5, "medium"),
        # medium: n / p < 20
        (300, 30, "medium"),
        # ordinary: everything else (n >= 300 and n/p >= 20)
        (300, 5, "ordinary"),
        (1000, 10, "ordinary"),
        (500, 25, "ordinary"),
    ],
)
def test_resolve_size_mode(n, p, expected):
    assert resolve_size_mode(n, p, _default_cfg()) == expected


def test_resolve_size_mode_uses_original_columns():
    """The size-mode ratio uses the ORIGINAL column count, not p after one-hot.

    Regression (issue #134): p was counted after the one-hot expansion, so
    n=4000 rows with 250 one-hot columns (only 50 original columns) gave
    n/p = 16 < 20 and the "medium" mode — restricted spaces on large data.
    With the original column count the ratio is 4000/50 = 80 >= 20 and the
    mode is "ordinary".
    """
    cfg = _default_cfg()
    # 250 one-hot-expanded columns: the old code resolved "medium".
    assert resolve_size_mode(4000, 250, cfg) == "medium"
    # 50 ORIGINAL columns (250 = 50 * 5 one-hot categories): "ordinary".
    assert resolve_size_mode(4000, 50, cfg) == "ordinary"


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_resolve_size_mode_extreme_small_n(n):
    """Edge cases: tiny n is always 'small' regardless of p."""
    assert resolve_size_mode(n, 1, _default_cfg()) == "small"
    assert resolve_size_mode(n, 100, _default_cfg()) == "small"


def test_resolve_size_mode_zero_p():
    """p = 0: the ratio is undefined — falls back to the n threshold."""
    assert resolve_size_mode(15, 0, _default_cfg()) == "small"
    assert resolve_size_mode(1000, 0, _default_cfg()) == "ordinary"


# ─────────────────── physical minima on extreme small n ────────────────────── #


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_extreme_small_n_tree_ranges_never_zero(n):
    """Trees: on tiny n the ranges keep physical minima >= 1.

    Regression: sequential A→B rule application collapsed
    ``min_samples_leaf`` to [0, 0] at n=2..3 (sklearn
    ``InvalidParameterError``) and ``max_depth`` below 1.
    """
    space = {
        "min_samples_leaf": _entry(1, 10, "int"),
        "max_depth": _entry(2, 32, "int"),
    }
    adjusted, logs = apply_complexity_guard(
        space, "decision_tree", n=n, p=5, config=_default_cfg()
    )
    msl = adjusted["min_samples_leaf"].config
    assert msl.low >= 1 and msl.high >= 1 and msl.low <= msl.high
    md = adjusted["max_depth"].config
    assert md.low >= 1 and md.high >= 1 and md.low <= md.high
    assert logs  # the guard still restricted something


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_extreme_small_n_xgb_ranges_never_zero(n):
    """XGBoost: on tiny n max_depth never collapses to 0.

    Regression: XGBoost silently overrides max_depth=0 and builds a depth-2
    tree; min_child_weight may not go below 1 either.
    """
    space = {
        "min_child_weight": _entry(1, 10, "float"),
        "max_depth": _entry(3, 10, "int"),
        "n_estimators": _entry(100, 800, "int", 100),
    }
    adjusted, _ = apply_complexity_guard(
        space, "xgboosting", n=n, p=5, config=_default_cfg()
    )
    mcw = adjusted["min_child_weight"].config
    assert mcw.low >= 1.0 and mcw.high >= 1.0 and mcw.low <= mcw.high
    md = adjusted["max_depth"].config
    assert md.low >= 1 and md.high >= 1 and md.low <= md.high
    assert adjusted["n_estimators"].config.high == 300


# ──────────────── leaf regularization on n < 8 (issue #138) ────────────────── #


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_small_n_leaf_regularization_never_collapses_to_one(n):
    """Regression (issue #138): on n < 8 the level B range [2, floor(n/4)]
    is empty and used to collapse to min_samples_leaf = 1.

    Now the lower bound 2 is kept, capped by the physical limit
    floor(n / 2). On n=4..7 the leaf never collapses to 1; on n=2..3 the
    physical limit floor(n/2) < 2 legitimately keeps the leaf at 1 (a leaf
    of 2 rows is unreachable on 2-3 rows).
    """
    space = {"min_samples_leaf": _entry(1, 10, "int")}
    adjusted, logs = apply_complexity_guard(
        space, "decision_tree", n=n, p=5, config=_default_cfg()
    )
    entry = adjusted["min_samples_leaf"]
    assert isinstance(entry.config, IntSpace)
    physical = min(math.floor(n / 2), 10)
    expected = (min(2, physical), physical) if physical < 2 else (2, physical)
    assert (entry.config.low, entry.config.high) == expected
    if physical >= 2:
        levels = [log["level"] for log in logs]
        assert "B" in levels


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_small_n_leaf_regularization_boosting_same_rule(n):
    """GradientBoosting shares the tree min_samples_leaf rules (issue #138)."""
    space = {"min_samples_leaf": _entry(1, 10, "int")}
    adjusted, _ = apply_complexity_guard(
        space, "gradient_boosting", n=n, p=5, config=_default_cfg()
    )
    entry = adjusted["min_samples_leaf"]
    assert isinstance(entry.config, IntSpace)
    physical = min(math.floor(n / 2), 10)
    expected = (min(2, physical), physical) if physical < 2 else (2, physical)
    assert (entry.config.low, entry.config.high) == expected


def test_small_n_leaf_regularization_monotonic_issue_example():
    """The issue's example: n=5 gives leaf 2 (not 1), n=8 gives leaf 2.

    Before the fix the behavior was non-monotonic: n=5 collapsed to
    min_samples_leaf=1 while n=8 already kept [2, 2].
    """
    space = {"min_samples_leaf": _entry(1, 10, "int")}
    adjusted_5, _ = apply_complexity_guard(
        space, "decision_tree", n=5, p=5, config=_default_cfg()
    )
    adjusted_8, _ = apply_complexity_guard(
        space, "decision_tree", n=8, p=5, config=_default_cfg()
    )
    leaf_5 = adjusted_5["min_samples_leaf"].config
    leaf_8 = adjusted_8["min_samples_leaf"].config
    assert (leaf_5.low, leaf_5.high) == (2, 2)
    assert (leaf_8.low, leaf_8.high) == (2, 2)


def test_small_n_leaf_regularization_float_fraction_preserved():
    """A float min_samples_leaf fraction on n < 8 stays a fraction.

    The fallback (lower bound 2 rows) must not leak into fraction units:
    the row bounds are divided back by n, so on n=5 the guard range is
    [2/5, floor(5/2)/5] = [0.4, 0.4], never a row count.
    """
    space = {"min_samples_leaf": _entry(0.1, 1.0, "float")}
    adjusted, logs = apply_complexity_guard(
        space, "decision_tree", n=5, p=5, config=_default_cfg()
    )
    entry = adjusted["min_samples_leaf"]
    assert isinstance(entry.config, FloatSpace)
    assert entry.config.low == pytest.approx(2 / 5, abs=1e-6)
    assert entry.config.high == pytest.approx(2 / 5, abs=1e-6)
    levels = [log["level"] for log in logs]
    assert "B" in levels


# ─────────────────────────── small mode: trees ─────────────────────────────── #


def test_small_tree_min_samples_leaf():
    """min_samples_leaf: A -> floor(n/2) upper, B -> [2, floor(n/4)]."""
    space = {"min_samples_leaf": _entry(1, 10, "int")}
    adjusted, logs = apply_complexity_guard(
        space, "decision_tree", n=15, p=5, config=_default_cfg()
    )
    entry = adjusted["min_samples_leaf"]
    assert isinstance(entry.config, IntSpace)
    # floor(15/2) = 7, floor(15/4) = 3 -> [2, 3]
    assert (entry.config.low, entry.config.high) == (2, 3)
    levels = [log["level"] for log in logs]
    assert "A" in levels and "B" in levels


def test_small_tree_min_samples_leaf_float_fraction_user_wins():
    """A float min_samples_leaf in (0, 1) is a sample fraction, not a count.

    sklearn semantics: a float leaf size < 1 is a fraction of the sample
    size; the guard's rules are expressed in absolute rows and must be
    converted. The user range in (0, 1] must not be widened to a count
    (regression: [0.2, 1.0] used to silently widen the lower bound).
    """
    # n=15 (small). A: high -> floor(15/2)/15 = 7/15; B: [2/15, floor(15/4)/15]
    # = [0.1333, 0.2]; the intersection with [0.2, 1.0] narrows to [0.2, 0.2].
    space = {"min_samples_leaf": _entry(0.2, 1.0, "float")}
    adjusted, logs = apply_complexity_guard(
        space,
        "decision_tree",
        n=15,
        p=5,
        config=_default_cfg(user_range_policy="user_wins"),
        user_space_keys=["min_samples_leaf"],
    )
    entry = adjusted["min_samples_leaf"]
    assert isinstance(entry.config, FloatSpace)
    assert entry.config.low == pytest.approx(0.2)
    assert entry.config.high == pytest.approx(0.2, abs=1e-6)
    assert not any(log["conflict"] for log in logs)
    assert all(log["applied"] for log in logs)
    levels = [log["level"] for log in logs]
    assert "A" in levels and "B" in levels


def test_small_tree_min_samples_leaf_float_fraction_engine_wins():
    """engine_wins resolves float-fraction conflicts on the fraction domain.

    A user range that does NOT intersect the guard range stays in fraction
    units — not a row count (the old bug produced count values that could
    make the tree unfit).
    """
    # n=15, p=5. User [0.4, 0.9]; guard B target [2/15, floor(15/4)/15] =
    # [0.1333, 0.2] — empty intersection -> engine_wins keeps the guard range.
    space = {"min_samples_leaf": _entry(0.4, 0.9, "float")}
    adjusted, logs = apply_complexity_guard(
        space,
        "decision_tree",
        n=15,
        p=5,
        config=_default_cfg(user_range_policy="engine_wins"),
        user_space_keys=["min_samples_leaf"],
    )
    entry = adjusted["min_samples_leaf"]
    assert isinstance(entry.config, FloatSpace)
    assert entry.config.low == pytest.approx(2 / 15, abs=1e-6)
    assert entry.config.high == pytest.approx(0.2, abs=1e-6)
    assert any(log.get("conflict") for log in logs)


def test_small_tree_min_samples_leaf_float_count_is_not_fraction():
    """A float min_samples_leaf >= 1 is a row count (not a fraction)."""
    space = {"min_samples_leaf": _entry(2.0, 10.0, "float")}
    adjusted, logs = apply_complexity_guard(
        space, "decision_tree", n=15, p=5, config=_default_cfg()
    )
    entry = adjusted["min_samples_leaf"]
    assert isinstance(entry.config, FloatSpace)
    # A -> floor(15/2)=7 (upper 10 clamps to 7), B -> [2, floor(15/4)=3].
    assert (entry.config.low, entry.config.high) == (2.0, 3.0)
    levels = [log["level"] for log in logs]
    assert "A" in levels and "B" in levels


def test_small_gradient_boosting_min_samples_leaf_float_fraction():
    """GradientBoosting float min_samples_leaf fraction works too."""
    space = {"min_samples_leaf": _entry(0.2, 0.5, "float")}
    adjusted, logs = apply_complexity_guard(
        space,
        "gradient_boosting",
        n=15,
        p=5,
        config=_default_cfg(user_range_policy="user_wins"),
        user_space_keys=["min_samples_leaf"],
    )
    entry = adjusted["min_samples_leaf"]
    assert isinstance(entry.config, FloatSpace)
    assert entry.config.low == pytest.approx(0.2)
    # B on n=15: [2/15, floor(15/4)/15] = [0.1333, 0.2].
    assert entry.config.high == pytest.approx(0.2, abs=1e-6)


def test_small_tree_max_depth():
    """max_depth: A -> floor(n/leaf_min)-1, B -> ceil(log2(n/leaf_min))."""
    space = {
        "min_samples_leaf": _entry(1, 10, "int"),
        "max_depth": _entry(2, 32, "int"),
    }
    adjusted, logs = apply_complexity_guard(
        space, "random_forest", n=15, p=5, config=_default_cfg()
    )
    entry = adjusted["max_depth"]
    assert isinstance(entry.config, IntSpace)
    # leaf_min=2 (min_samples_leaf B lower bound) -> A: 15/2-1=6, B: ceil(log2(7.5))=3
    assert (entry.config.low, entry.config.high) == (2, 3)


def test_small_tree_max_features_float():
    """max_features (a fraction): A keeps at least one feature (1/p)."""
    space = {"max_features": _entry(0.1, 1.0, "float")}
    adjusted, logs = apply_complexity_guard(
        space, "decision_tree", n=15, p=5, config=_default_cfg()
    )
    entry = adjusted["max_features"]
    assert isinstance(entry.config, FloatSpace)
    assert entry.config.low == pytest.approx(0.2)  # 1/5
    assert entry.config.high == 1.0


def test_small_tree_max_features_int_kept_as_count():
    """max_features as an int is a feature count, not a fraction.

    Regression: ``_clamp_for_param`` treated ``max_features`` as a fraction
    unconditionally, so an int range [5, 10] silently collapsed to [1, 1]
    with ``applied=True`` and no conflict (violating the documented contract
    for user ranges). An int below 1 is clamped to the physical minimum of 1.
    """
    # User range [5, 10]: must stay untouched (no conflict, no collapse).
    space = {"max_features": _entry(5, 10, "int")}
    adjusted, logs = apply_complexity_guard(
        space,
        "decision_tree",
        n=100,
        p=20,
        config=_default_cfg(user_range_policy="user_wins"),
        user_space_keys=["max_features"],
    )
    entry = adjusted["max_features"]
    assert isinstance(entry.config, IntSpace)
    assert (entry.config.low, entry.config.high) == (5, 10)
    assert not any(log.get("conflict") for log in logs)
    assert not any(log.get("fixed") for log in logs)

    # Default int range [1, 20] stays a count too (A rule lower bound 1/p
    # is below 1, the physical minimum for an int).
    space2 = {"max_features": _entry(1, 20, "int")}
    adjusted2, logs2 = apply_complexity_guard(
        space2, "decision_tree", n=15, p=5, config=_default_cfg()
    )
    entry2 = adjusted2["max_features"]
    assert isinstance(entry2.config, IntSpace)
    assert (entry2.config.low, entry2.config.high) == (1, 20)


def test_small_tree_max_features_int_engine_wins():
    """engine_wins with an int max_features uses a count domain, not (0, 1].

    A user range that conflicts with the guard (empty intersection) is
    resolved in favor of the guard; ``_domain_for_param`` provides a count
    domain ``[1, p]`` for int ``max_features`` so the restriction is not
    computed against the fraction domain ``(0, 1]``.
    """
    # p=8, user [1,2]: the guard's A lower bound 1/p is below 1 for the int
    # count domain, so no conflict arises — the range is a valid count.
    space = {"max_features": _entry(1, 2, "int")}
    adjusted, logs = apply_complexity_guard(
        space,
        "decision_tree",
        n=100,
        p=8,
        config=_default_cfg(user_range_policy="engine_wins"),
        user_space_keys=["max_features"],
    )
    entry = adjusted["max_features"]
    assert isinstance(entry.config, IntSpace)
    # No empty intersection with the count domain [1, p]: unchanged.
    assert (entry.config.low, entry.config.high) == (1, 2)
    assert not any(log.get("conflict") for log in logs)

    # A user float range that genuinely conflicts (empty intersection with
    # the guard fraction domain) is resolved by engine_wins on the fraction
    # domain — max_features stays a fraction.
    space2 = {"max_features": _entry(0.05, 0.1, "float")}
    adjusted2, logs2 = apply_complexity_guard(
        space2,
        "decision_tree",
        n=100,
        p=8,
        config=_default_cfg(user_range_policy="engine_wins"),
        user_space_keys=["max_features"],
    )
    entry2 = adjusted2["max_features"]
    assert isinstance(entry2.config, FloatSpace)
    # Guard A lower bound 1/p = 0.125 on the fraction domain wins.
    assert entry2.config.low == pytest.approx(0.125)
    assert any(log.get("conflict") for log in logs2)


def test_small_tree_n_estimators_time_only():
    """n_estimators is capped for time (level 'time'), not regularization."""
    space = {"n_estimators": _entry(50, 800, "int", 50)}
    adjusted, logs = apply_complexity_guard(
        space, "random_forest", n=15, p=5, config=_default_cfg()
    )
    entry = adjusted["n_estimators"]
    assert (entry.config.low, entry.config.high) == (50, 300)
    assert all(log["level"] == "time" for log in logs)


# ─────────────────────────── small mode: boosting ──────────────────────────── #


def test_small_boosting_rules():
    """XGBoost rules on small data."""
    space = {
        "max_depth": _entry(3, 10, "int"),
        "subsample": _entry(0.5, 1.0, "float"),
        "colsample_bytree": _entry(0.5, 1.0, "float"),
        "reg_lambda": _entry(1e-3, 100.0, "float_log"),
        "n_estimators": _entry(100, 800, "int", 100),
    }
    adjusted, logs = apply_complexity_guard(
        space, "xgboosting", n=15, p=5, config=_default_cfg()
    )
    # max_depth: B target [1, 3]
    assert (adjusted["max_depth"].config.low, adjusted["max_depth"].config.high) == (1, 3)
    # subsample: B lower bound 0.7
    assert adjusted["subsample"].config.low == pytest.approx(0.7)
    # colsample_bytree: A lower bound 1/p = 0.2 (original 0.5 stays)
    assert adjusted["colsample_bytree"].config.low == pytest.approx(0.5)
    # reg_lambda: B [1, 100] preserving the log scale
    reg = adjusted["reg_lambda"]
    assert isinstance(reg.config, FloatSpace)
    assert reg.config.type == "float_log"
    assert reg.config.low == pytest.approx(1.0)
    assert reg.config.high == pytest.approx(100.0)
    # n_estimators: time cap
    assert adjusted["n_estimators"].config.high == 300


def test_small_boosting_min_child_weight_leaf_min():
    """min_child_weight is used as the leaf size for XGBoost max_depth."""
    space = {
        "min_child_weight": _entry(1, 10, "float"),
        "max_depth": _entry(3, 10, "int"),
    }
    adjusted, _ = apply_complexity_guard(
        space, "xgboosting", n=15, p=5, config=_default_cfg()
    )
    # min_child_weight B: lower bound 2 -> leaf_min=2; max_depth A: 15/2-1=6
    assert adjusted["min_child_weight"].config.low == pytest.approx(2.0)
    max_depth = adjusted["max_depth"].config
    assert max_depth.high <= 6


def test_small_gradient_boosting_min_samples_leaf():
    """GradientBoosting uses min_samples_leaf like the trees."""
    space = {"min_samples_leaf": _entry(1, 10, "int")}
    adjusted, _ = apply_complexity_guard(
        space, "gradient_boosting", n=15, p=5, config=_default_cfg()
    )
    assert (adjusted["min_samples_leaf"].config.low, adjusted["min_samples_leaf"].config.high) == (2, 3)


# ─────────────────────── default XGBoost space (issue #132) ───────────────── #


def test_default_xgboost_space_has_min_child_weight_and_reg_lambda():
    """The default XGBoost space exposes min_child_weight and reg_lambda.

    Regression for issue #132: the guard's ``min_child_weight`` and
    ``reg_lambda`` rules never fired on the defaults because the parameters
    were absent from ``DEFAULT_SPACES['xgboosting']`` — only max_depth,
    subsample and n_estimators were restricted.
    """
    space = DEFAULT_SPACES["xgboosting"]
    assert "min_child_weight" in space
    assert "reg_lambda" in space

    mcw = space["min_child_weight"].config
    assert isinstance(mcw, FloatSpace)
    assert mcw.type == "float"
    assert (mcw.low, mcw.high) == (1.0, 10.0)

    reg = space["reg_lambda"].config
    assert isinstance(reg, FloatSpace)
    assert reg.type == "float_log"
    assert reg.low == pytest.approx(1e-3)
    assert reg.high == pytest.approx(100.0)


def test_default_xgboost_space_guard_restricts_new_params():
    """On small data the guard restricts min_child_weight and reg_lambda
    from the DEFAULT_SPACES['xgboosting'].

    Regression for issue #132: previously the guard rules for these
    parameters could never fire on defaults. Now the default space carries
    them, so the level B rules tighten min_child_weight to [2, 10] and
    reg_lambda to [1, 100] (log scale).
    """
    space = dict(DEFAULT_SPACES["xgboosting"])
    adjusted, logs = apply_complexity_guard(
        space, "xgboosting", n=15, p=5, config=_default_cfg()
    )
    mcw = adjusted["min_child_weight"].config
    assert isinstance(mcw, FloatSpace)
    assert mcw.type == "float"
    assert mcw.low == pytest.approx(2.0)
    assert mcw.high == pytest.approx(10.0)
    reg = adjusted["reg_lambda"].config
    assert isinstance(reg, FloatSpace)
    assert reg.type == "float_log"
    assert reg.low == pytest.approx(1.0)
    assert reg.high == pytest.approx(100.0)
    restricted = {(log["algo"], log["param"]) for log in logs}
    assert ("xgboosting", "min_child_weight") in restricted
    assert ("xgboosting", "reg_lambda") in restricted


def test_default_xgboost_space_user_ranges_intersect_new_params():
    """A user range of min_child_weight/reg_lambda is intersected with the guard.

    The new default parameters behave like every other default: an explicit
    user range narrower than the guard target keeps its values when it does
    not conflict (user_wins).
    """
    space = dict(DEFAULT_SPACES["xgboosting"])
    space["min_child_weight"] = _entry(1, 3, "float")
    space["reg_lambda"] = _entry(1e-2, 5.0, "float_log")
    adjusted, logs = apply_complexity_guard(
        space,
        "xgboosting",
        n=15,
        p=5,
        config=_default_cfg(user_range_policy="user_wins"),
        user_space_keys=["min_child_weight", "reg_lambda"],
    )
    mcw = adjusted["min_child_weight"].config
    # Guard target [2, 10] ∩ user [1, 3] -> [2, 3] (no conflict).
    assert mcw.low == pytest.approx(2.0)
    assert mcw.high == pytest.approx(3.0)
    reg = adjusted["reg_lambda"].config
    # Guard target [1, 100] ∩ user [0.01, 5] -> [1, 5].
    assert reg.low == pytest.approx(1.0)
    assert reg.high == pytest.approx(5.0)
    assert not any(log.get("conflict") for log in logs)


def test_default_xgboost_space_medium_mode_new_params():
    """In the medium mode reg_lambda is capped and min_child_weight raised."""
    space = dict(DEFAULT_SPACES["xgboosting"])
    adjusted, logs = apply_complexity_guard(
        space, "xgboosting", n=100, p=5, config=_default_cfg()
    )
    mcw = adjusted["min_child_weight"].config
    assert mcw.low == pytest.approx(2.0)
    reg = adjusted["reg_lambda"].config
    assert isinstance(reg, FloatSpace)
    assert reg.type == "float_log"
    assert reg.low == pytest.approx(1.0)
    assert reg.high == pytest.approx(100.0)
    restricted = {(log["algo"], log["param"]) for log in logs}
    assert ("xgboosting", "min_child_weight") in restricted
    assert ("xgboosting", "reg_lambda") in restricted


# ─────────────────────────── medium mode ───────────────────────────────────── #


def test_medium_tree_rules():
    """Medium mode: min_samples_leaf >= 2, max_depth <= ceil(log2(n/leaf))+2."""
    space = {
        "min_samples_leaf": _entry(1, 10, "int"),
        "max_depth": _entry(2, 32, "int"),
    }
    adjusted, logs = apply_complexity_guard(
        space, "decision_tree", n=100, p=5, config=_default_cfg()
    )
    # medium: min_samples_leaf B lower bound 2, A upper bound floor(100/2)=50
    assert adjusted["min_samples_leaf"].config.low == 2
    # leaf_min=2 -> ceil(log2(50))+2 = 6+2 = 8
    assert adjusted["max_depth"].config.high == 8


def test_medium_xgb_max_depth():
    """Medium mode: XGBoost max_depth <= 5."""
    space = {
        "min_child_weight": _entry(1, 10, "float"),
        "max_depth": _entry(3, 10, "int"),
    }
    adjusted, _ = apply_complexity_guard(
        space, "xgboosting", n=100, p=5, config=_default_cfg()
    )
    assert adjusted["max_depth"].config.high == 5


# ─────────────────── ordinary mode / disabled / levels ────────────────────── #


def test_ordinary_mode_space_unchanged():
    """In the ordinary mode the spaces are identical."""
    space = {
        "max_depth": _entry(2, 32, "int"),
        "min_samples_leaf": _entry(1, 10, "int"),
    }
    adjusted, logs = apply_complexity_guard(
        space, "decision_tree", n=1000, p=10, config=_default_cfg()
    )
    assert logs == []
    assert adjusted["max_depth"].config.high == 32
    assert adjusted["min_samples_leaf"].config.high == 10


def test_ordinary_mode_with_one_hot_p_size():
    """Large-but-wide data (one-hot) stays 'ordinary' via p_size.

    Regression (issue #134): n=4000 rows, 250 one-hot columns (50 original)
    gave n/p = 16 < 20 -> "medium" and restricted spaces. With ``p_size``
    the size mode uses the ORIGINAL column count (50), so spaces do not
    change — the post-preprocessing ``p`` (250) is still used by the rules
    that depend on the feature count (``max_features``, ``colsample_bytree``).
    """
    tree_space = {
        "max_depth": _entry(2, 32, "int"),
        "min_samples_leaf": _entry(1, 10, "int"),
        "max_features": _entry(0.1, 1.0, "float"),
    }
    adjusted, logs = apply_complexity_guard(
        tree_space,
        "decision_tree",
        n=4000,
        p=250,
        p_size=50,
        config=_default_cfg(),
    )
    assert logs == []
    assert adjusted["max_depth"].config.high == 32
    assert adjusted["min_samples_leaf"].config.high == 10
    # max_features does not change either: the mode is ordinary, so even the
    # feature-count rules are not applied.
    assert adjusted["max_features"].config.low == pytest.approx(0.1)

    # The same data WITHOUT p_size falls back to p: the old behavior, which
    # resolved "medium" (restricted spaces).
    adjusted_old, logs_old = apply_complexity_guard(
        tree_space,
        "decision_tree",
        n=4000,
        p=250,
        config=_default_cfg(),
    )
    assert logs_old
    # medium: max_depth <= ceil(log2(4000/2)) + 2 = 13
    assert adjusted_old["max_depth"].config.high == 13
    assert adjusted_old["min_samples_leaf"].config.low == 2

    # Boosting: XGBoost max_depth <= 5 / subsample >= 0.7 would fire in the
    # medium mode; with p_size they stay unchanged.
    xgb_space = {
        "max_depth": _entry(3, 10, "int"),
        "subsample": _entry(0.5, 1.0, "float"),
        "colsample_bytree": _entry(0.5, 1.0, "float"),
    }
    adjusted_xgb, logs_xgb = apply_complexity_guard(
        xgb_space,
        "xgboosting",
        n=4000,
        p=250,
        p_size=50,
        config=_default_cfg(),
    )
    assert logs_xgb == []
    assert (adjusted_xgb["max_depth"].config.low, adjusted_xgb["max_depth"].config.high) == (3, 10)
    assert adjusted_xgb["subsample"].config.low == pytest.approx(0.5)


def test_off_mode_space_unchanged():
    """mode='off': no changes, no logs."""
    space = {"max_depth": _entry(2, 32, "int")}
    adjusted, logs = apply_complexity_guard(
        space,
        "decision_tree",
        n=15,
        p=5,
        config=_default_cfg(mode="off"),
    )
    assert logs == []
    assert adjusted["max_depth"].config.high == 32


def test_warn_mode_logs_but_does_not_apply():
    """mode='warn': restrictions are logged but NOT applied."""
    space = {"max_depth": _entry(2, 32, "int")}
    adjusted, logs = apply_complexity_guard(
        space,
        "decision_tree",
        n=15,
        p=5,
        config=_default_cfg(mode="warn"),
    )
    assert logs
    assert all(not log["applied"] for log in logs)
    assert adjusted["max_depth"].config.high == 32


def test_lossless_off_keeps_level_a_unchanged():
    """lossless=False disables level A only."""
    space = {
        "min_samples_leaf": _entry(1, 10, "int"),
        "max_depth": _entry(2, 32, "int"),
    }
    adjusted, logs = apply_complexity_guard(
        space,
        "decision_tree",
        n=15,
        p=5,
        config=_default_cfg(lossless=False),
    )
    # Level A disabled: min_samples_leaf only gets the B range [2, floor(15/4)=3]
    assert (adjusted["min_samples_leaf"].config.low, adjusted["min_samples_leaf"].config.high) == (2, 3)
    # max_depth: only B applies -> ceil(log2(15/2))=3
    assert adjusted["max_depth"].config.high == 3
    assert all(log["level"] != "A" for log in logs)


def test_regularize_off_disables_level_b():
    """regularize=False disables level B only."""
    space = {
        "min_samples_leaf": _entry(1, 10, "int"),
        "max_depth": _entry(2, 32, "int"),
    }
    adjusted, logs = apply_complexity_guard(
        space,
        "decision_tree",
        n=15,
        p=5,
        config=_default_cfg(regularize=False),
    )
    # Level B disabled: only A applies.
    # min_samples_leaf: A upper floor(15/2)=7, low stays 1 -> [1, 7].
    assert (adjusted["min_samples_leaf"].config.low, adjusted["min_samples_leaf"].config.high) == (1, 7)
    # max_depth A uses leaf_min = min_samples_leaf low = 1 -> floor(15/1)-1 = 14.
    assert adjusted["max_depth"].config.high == 14
    assert all(log["level"] == "A" for log in logs)


def test_unsupported_algorithm_unchanged():
    """Linear/other algorithms are not touched by the guard."""
    space = {"alpha": _entry(1e-4, 10.0, "float_log")}
    adjusted, logs = apply_complexity_guard(
        space, "ridge", n=15, p=5, config=_default_cfg()
    )
    assert logs == []
    assert adjusted["alpha"].config.low == pytest.approx(1e-4)


# ─────────────────────── user-range conflicts ──────────────────────────────── #


def test_user_range_conflict_user_wins():
    """user_wins: an empty intersection keeps the explicit user range."""
    space = {"min_samples_leaf": _entry(1, 1, "int")}
    adjusted, logs = apply_complexity_guard(
        space,
        "decision_tree",
        n=15,
        p=5,
        config=_default_cfg(user_range_policy="user_wins"),
        user_space_keys=["min_samples_leaf"],
    )
    # The user range [1, 1] stays despite the guard's B lower bound 2.
    assert (adjusted["min_samples_leaf"].config.low, adjusted["min_samples_leaf"].config.high) == (1, 1)
    conflicts = [log for log in logs if log.get("conflict") and not log["applied"]]
    assert conflicts
    # The conflict records carry before/after (the unchanged user range) and
    # the guard's target range — the logger reads these keys unconditionally.
    for log in conflicts:
        assert "before" in log and "after" in log
        assert log["before"] == log["after"]
        assert "rule_range" in log


def test_user_range_conflict_engine_wins():
    """engine_wins: on an empty intersection the guard restriction is applied."""
    space = {"min_samples_leaf": _entry(1, 1, "int")}
    adjusted, logs = apply_complexity_guard(
        space,
        "decision_tree",
        n=15,
        p=5,
        config=_default_cfg(user_range_policy="engine_wins"),
        user_space_keys=["min_samples_leaf"],
    )
    # The guard's B range [2, 3] wins.
    assert (adjusted["min_samples_leaf"].config.low, adjusted["min_samples_leaf"].config.high) == (2, 3)
    assert any(log.get("conflict") for log in logs)
    # Applied conflict records carry before/after too.
    for log in logs:
        assert "before" in log and "after" in log


def test_user_range_non_conflicting_intersection():
    """A user range that overlaps the guard range is narrowed by the intersection."""
    space = {"min_samples_leaf": _entry(1, 10, "int")}
    adjusted, logs = apply_complexity_guard(
        space,
        "decision_tree",
        n=15,
        p=5,
        config=_default_cfg(user_range_policy="user_wins"),
        user_space_keys=["min_samples_leaf"],
    )
    # Intersection of user [1, 10] with guard B [2, 3] -> [2, 3].
    assert (adjusted["min_samples_leaf"].config.low, adjusted["min_samples_leaf"].config.high) == (2, 3)
    assert not any(log.get("conflict") for log in logs)


# ─────────────────────── collapse to a single value ────────────────────────── #


def test_collapse_to_single_value():
    """A range collapsing to one value fixes the parameter (logged)."""
    space = {"colsample_bytree": _entry(0.5, 1.0, "float")}
    adjusted, logs = apply_complexity_guard(
        space, "xgboosting", n=15, p=1, config=_default_cfg()
    )
    # colsample A lower bound 1/p = 1.0 -> [1.0, 1.0] fixed.
    assert adjusted["colsample_bytree"].config.low == pytest.approx(1.0)
    assert adjusted["colsample_bytree"].config.high == pytest.approx(1.0)
    assert any(log["fixed"] for log in logs)


# ─────────────────────── type / scale preservation ─────────────────────────── #


def test_float_log_scale_preserved():
    """A float_log distribution stays float_log after the guard."""
    space = {"reg_lambda": _entry(1e-3, 100.0, "float_log")}
    adjusted, _ = apply_complexity_guard(
        space, "xgboosting", n=15, p=5, config=_default_cfg()
    )
    reg = adjusted["reg_lambda"]
    assert isinstance(reg.config, FloatSpace)
    assert reg.config.type == "float_log"
    assert reg.config.low == pytest.approx(1.0)


def test_int_type_preserved():
    """An int distribution stays int (integer bounds)."""
    space = {"max_depth": _entry(3, 10, "int")}
    adjusted, _ = apply_complexity_guard(
        space, "xgboosting", n=15, p=5, config=_default_cfg()
    )
    assert isinstance(adjusted["max_depth"].config, IntSpace)


def test_guard_does_not_mutate_input_space():
    """The original space entries are not mutated."""
    space = {"max_depth": _entry(2, 32, "int")}
    original_high = space["max_depth"].config.high
    apply_complexity_guard(space, "decision_tree", n=15, p=5, config=_default_cfg())
    assert space["max_depth"].config.high == original_high


# ─────────────────────── report records merging ────────────────────────────── #


def _rec(algo, param, level, before, after, **extra):
    rec = {
        "algo": algo,
        "param": param,
        "level": level,
        "before": before,
        "after": after,
        "applied": True,
        "fixed": False,
        "conflict": False,
        **extra,
    }
    return rec


def test_merge_complexity_guard_records_one_per_parameter():
    """A/B rules of the same parameter merge into one record with levels."""
    from configurable_automl_engine.training_engine.component import (
        _merge_complexity_guard_records,
    )

    records = [
        # Phase 1: level A then level B on the same parameter.
        _rec("decision_tree", "min_samples_leaf", "A", [1.0, 10.0], [1.0, 6.0]),
        _rec("decision_tree", "min_samples_leaf", "B", [1.0, 6.0], [2.0, 3.0]),
        # Phase 2 (refine): identical final state — must not duplicate.
        _rec("decision_tree", "min_samples_leaf", "B", [1.0, 6.0], [2.0, 3.0]),
        # Another parameter of the same algorithm.
        _rec("decision_tree", "max_depth", "A", [2.0, 32.0], [2.0, 14.0]),
        _rec("decision_tree", "max_depth", "B", [2.0, 14.0], [2.0, 3.0]),
        # A different algorithm.
        _rec("xgboosting", "max_depth", "B", [3.0, 10.0], [1.0, 3.0]),
    ]
    merged = _merge_complexity_guard_records(records)
    assert len(merged) == 3
    by_key = {(r["algo"], r["param"]): r for r in merged}
    assert by_key[("decision_tree", "min_samples_leaf")]["after"] == [2.0, 3.0]
    assert sorted(by_key[("decision_tree", "min_samples_leaf")]["levels"]) == ["A", "B"]
    assert by_key[("decision_tree", "max_depth")]["after"] == [2.0, 3.0]
    assert sorted(by_key[("decision_tree", "max_depth")]["levels"]) == ["A", "B"]
    assert by_key[("xgboosting", "max_depth")]["after"] == [1.0, 3.0]
    assert by_key[("xgboosting", "max_depth")]["levels"] == ["B"]


# ─────────────────── step-grid alignment (issue #139) ──────────────────────── #


def test_guard_n_estimators_aligns_high_to_step_grid():
    """A stepped n_estimators range is aligned to its step grid (issue #139).

    The time cap tightens [50, 800] (step 50) to [50, 300]: 300 lies on the
    step-50 grid, so the range stays. Regression scenario: with a step of 100
    the same cap produces [50, 300] whose width is not divisible by 100 —
    Optuna silently narrows the range to [50, 250]. The guard must store the
    aligned range [50, 250] and document it in the log record.
    """
    space = {"n_estimators": _entry(50, 800, "int", 100)}
    adjusted, logs = apply_complexity_guard(
        space, "random_forest", n=15, p=5, config=_default_cfg()
    )
    entry = adjusted["n_estimators"]
    assert isinstance(entry.config, IntSpace)
    # (300 - 50) % 100 != 0 -> aligned to low + 2*100 = 250.
    assert (entry.config.low, entry.config.high) == (50, 250)
    assert any(log["after"] == [50.0, 250.0] for log in logs)


def test_guard_already_divisible_step_unchanged():
    """A capped high that lies on the step grid stays unchanged."""
    space = {"n_estimators": _entry(50, 800, "int", 50)}
    adjusted, logs = apply_complexity_guard(
        space, "random_forest", n=15, p=5, config=_default_cfg()
    )
    entry = adjusted["n_estimators"]
    # (300 - 50) % 50 == 0 -> high stays 300.
    assert (entry.config.low, entry.config.high) == (50, 300)


def test_guard_empty_step_grid_fixes_value():
    """An empty step grid after a rule collapses to a single grid value.

    The user range [250, 800] with a step of 100 is capped to [250, 300] by
    the time rule. The step-100 grid anchored at 250 contains no value above
    250 within the range (the next grid value 350 exceeds the cap), so the
    parameter is fixed at 250 — the only value Optuna could sample.
    """
    space = {"n_estimators": _entry(250, 800, "int", 100)}
    adjusted, logs = apply_complexity_guard(
        space, "random_forest", n=15, p=5, config=_default_cfg()
    )
    entry = adjusted["n_estimators"]
    assert isinstance(entry.config, IntSpace)
    assert (entry.config.low, entry.config.high) == (250, 250)
    assert any(log.get("fixed") for log in logs)


def test_guard_step_grid_alignment_preserves_type_and_scale():
    """Alignment keeps the int type and does not touch log-scale floats."""
    space = {
        "n_estimators": _entry(100, 800, "int", 100),
        "reg_lambda": _entry(1e-3, 100.0, "float_log"),
    }
    adjusted, _ = apply_complexity_guard(
        space, "xgboosting", n=15, p=5, config=_default_cfg()
    )
    n_est = adjusted["n_estimators"]
    assert isinstance(n_est.config, IntSpace)
    # (300 - 100) % 100 == 0 -> stays 300.
    assert (n_est.config.low, n_est.config.high) == (100, 300)
    reg = adjusted["reg_lambda"]
    assert isinstance(reg.config, FloatSpace)
    assert reg.config.type == "float_log"
    assert reg.config.low == pytest.approx(1.0)
    assert reg.config.high == pytest.approx(100.0)


def test_guard_step_alignment_does_not_mutate_input():
    """Step alignment inside the guard never mutates the source entry."""
    space = {"n_estimators": _entry(50, 800, "int", 100)}
    original = space["n_estimators"]
    apply_complexity_guard(space, "random_forest", n=15, p=5, config=_default_cfg())
    assert (original.config.low, original.config.high) == (50, 800)
    assert original.config.step == 100
