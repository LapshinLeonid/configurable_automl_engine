"""The restriction rules of the adaptive complexity guard.

Holds :class:`Rule` (one declarative restriction of a hyperparameter range),
every ``_rule_*`` function with its constants and small helpers, the
algorithm families (``TREE_ALGORITHMS``, ``BOOSTING_ALGORITHMS``) and the rule
tables per (size mode, family). The rules are pure functions of the original
bounds and the data quantities ``(n, p, leaf_min)``. The module must not
import :mod:`.complexity_guard`, which applies the rules.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from configurable_automl_engine.hpo.default_spaces import (
    FloatSpace,
    IntSpace,
    SearchSpaceEntry,
)

# ─────────────────────────────── rule engine ───────────────────────────────── #


@dataclass(frozen=True)
class Rule:
    """A single declarative restriction for a hyperparameter.

    Attributes:
        level: ``"A"`` — lossless clipping, ``"B"`` — regularization,
            ``"time"`` — time-only restriction (bagging ensembles, for
            which more trees only reduce variance).
        reason: Human-readable reason for logs and reports.
        apply: Callable that computes the TARGET range ``(low, high)`` of the
            parameter from the original bounds ``(orig_low, orig_high)`` and
            the data quantities ``(n, p, leaf_min)``. Rules only tighten: the
            target's unconstrained bound equals the original one.
    """

    level: str
    reason: str
    apply: Callable[[float, float, int, int, float], tuple[float, float]]


# Lower bound for ``min_samples_leaf`` on level B (the "2–3" target range;
# the documented default is 2).
_MIN_SAMPLES_LEAF_B_LOW = 2.0
# Upper bound for ``min_samples_leaf`` on level B in the small mode.
# ``floor(n / 4)`` with the leaf bound is used as a base multiplier.
_MIN_SAMPLES_LEAF_B_DIVISOR = 4.0
# On ``n < 8`` the level B range ``[2, floor(n / 4)]`` is empty; it collapses
# to the lower bound 2 (regularization must not vanish), capped by the
# physical limit ``floor(n / 2)`` of level A.
_MIN_SAMPLES_LEAF_B_PHYSICAL_DIVISOR = 2.0

# Lower bound for XGBoost ``min_child_weight`` on level B (for a squared loss
# this is roughly the minimum number of rows in a leaf).
_MIN_CHILD_WEIGHT_B_LOW = 2.0

# Minimum ``subsample`` fraction on level B.
_SUBSAMPLE_B_LOW = 0.7

# Bounds of XGBoost ``reg_lambda`` on level B (log scale).
_REG_LAMBDA_B_LOW = 1.0
_REG_LAMBDA_B_HIGH = 100.0

# Target range of ``max_depth`` on level B for boosting in the small mode
# (the "1–3" target range).
_BOOST_MAX_DEPTH_B_LOW = 1.0
_BOOST_MAX_DEPTH_B_HIGH = 3.0

# Upper bound of XGBoost ``max_depth`` in the medium mode.
_XGB_MAX_DEPTH_MEDIUM_HIGH = 5.0

# Time-only upper bound of ``n_estimators`` in the small mode for BAGGING
# ensembles (RF, Extra trees): only the upper bound is capped — more trees
# reduce variance, not a regularization (the "100–300" target range).
_N_ESTIMATORS_TIME_HIGH = 300.0

# Level B budget of ``n_estimators`` for BOOSTING in the small mode.
# For a boosting ensemble the number of trees together with ``learning_rate``
# is the main complexity regulator, so the cap is a regularization rule
# (disabled together with ``regularize``), not a time-only restriction.
#
# A strict per-trial ``n_estimators * learning_rate`` budget cannot be
# enforced: the guard's declarative rules are per-parameter and Optuna samples
# the two ranges independently. Instead the UPPER BOUND of ``n_estimators``
# depends on the upper bound of ``learning_rate`` in the space:
# ``cap = budget / lr_high`` (see :func:`_boost_n_estimators_cap`). With the
# default ``learning_rate`` range [0.01, 0.3] of GradientBoosting/XGBoost the
# cap is ``200 = 60 / 0.3``; a higher lr upper bound tightens the cap, a lower
# one loosens it. Documented in API_REFERENCE.md.
_BOOST_N_ESTIMATORS_BUDGET = 60.0

# Default ``learning_rate`` upper bound assumed when the parameter is absent
# from the space: matches ``DEFAULT_SPACES`` of GradientBoosting/XGBoost
# ([0.01, 0.3]).
_BOOST_DEFAULT_LR_HIGH = 0.3

# ``+2`` slack added to ``ceil(log2(n / leaf_min))`` in the medium mode.
_MEDIUM_DEPTH_SLACK = 2


def _is_fraction_param(param: str, *, is_float: bool, low: float, high: float) -> bool:
    """Whether the range is a fraction of the sample/features in (0, 1].

    sklearn accepts ``min_samples_leaf`` both as an absolute row count (int
    >= 1) and as a fraction of the sample size (a float strictly in
    ``(0, 1)``; ``1.0`` counts as one row per sklearn validation). The
    guard's leaf rules are expressed in absolute rows, so a fractional range
    must be converted to rows before a rule applies and converted back
    afterwards. ``subsample``/``colsample_bytree`` and float
    ``max_features`` are always fractions.

    Args:
        param: The hyperparameter name.
        is_float: Whether the space entry is a float.
        low: Lower bound of the range.
        high: Upper bound of the range.

    Returns:
        bool: ``True`` when the range lives in the unit interval.
    """
    if param in {"subsample", "colsample_bytree"}:
        return True
    if param == "max_features":
        return is_float
    if param == "min_samples_leaf":
        # A float leaf size strictly below 1 is a fraction of the sample
        # size; ``1.0`` and above are row counts (sklearn semantics).
        return is_float and low < 1.0 and high <= 1.0
    return False


def _apply_rule(
    rule: Rule,
    low: float,
    high: float,
    n: int,
    p: int,
    leaf_min: float,
    param: str,
    is_fraction: bool,
) -> tuple[float, float]:
    """Apply a rule to a range, converting fraction ranges to rows.

    The leaf rules of ``min_samples_leaf`` are expressed in absolute rows
    (``floor(n / 2)``, ``[2, floor(n / 4)]`` with a fallback to 2 on an empty
    range). When the range is a fraction of the sample size (float in (0, 1)),
    the bounds are converted to rows (``value * n``) before the rule applies
    and divided back afterwards.

    Args:
        rule: The rule to apply.
        low: Current lower bound.
        high: Current upper bound.
        n: Number of rows.
        p: Number of features.
        leaf_min: Effective minimum leaf size, in rows.
        param: The hyperparameter name.
        is_fraction: Whether the range is a fraction.

    Returns:
        tuple[float, float]: The rule's target range in the input units.
    """
    if param == "min_samples_leaf" and is_fraction:
        low_r, high_r = low * n, high * n
        low_r, high_r = rule.apply(low_r, high_r, n, p, leaf_min)
        return low_r / n, high_r / n
    return rule.apply(low, high, n, p, leaf_min)


def _clamp_for_param(
    param: str,
    low: float,
    high: float,
    n: int,
    *,
    is_float: bool = False,
) -> tuple[float, float]:
    """Clamp a numeric range to the physical bounds of a parameter.

    Guarantees ``low <= high`` (a contradicting range collapses to a single
    value — the parameter is fixed), preserves the int/float type and the log
    scale. Integer-like parameters (leaf sizes, depths, tree counts) never go
    below their physical minimum of 1: on extremely small ``n`` a rule may
    compute an empty or inverted range whose bounds would collapse to 0 (e.g.
    on ``n=1`` a leaf rule yields at most 0 rows) — such a range is fixed at
    1, never at 0, because sklearn/XGBoost reject 0 (``InvalidParameterError``
    / silently overridden depth).

    Args:
        param: The hyperparameter name.
        low: The lower bound.
        high: The upper bound.
        n: Number of rows (physical upper bound for leaf-size parameters).
        is_float: Whether the parameter is a float in the search space.
            ``max_features`` and ``min_samples_leaf`` accept both a fraction
            (float in (0, 1]) and a plain count (int, or float >= 1).

    Returns:
        tuple[float, float]: A valid ``(low, high)`` with a physical minimum
            of 1 for integer-like parameters.
    """
    is_fraction = _is_fraction_param(param, is_float=is_float, low=low, high=high)
    if is_fraction:
        # Fractions stay within (0, 1].
        low = max(low, 1e-9)
        high = min(high, 1.0)
    else:
        # The leaf size cannot exceed the number of rows.
        if param in {"min_samples_leaf", "min_child_weight"}:
            high = min(high, float(n))
        # Physical minimum of integer-like parameters: never below 1.
        high = max(high, 1.0)
    if low > high:
        # An empty range collapses to a single value — the parameter is fixed.
        low = min(low, high)
    return low, high


def _rule_min_samples_leaf_a() -> Rule:
    """Level A: ``min_samples_leaf`` upper bound not above ``floor(n / 2)``."""

    def _apply(
        orig_low: float,
        orig_high: float,
        n: int,
        p: int,
        leaf_min: float,
    ) -> tuple[float, float]:
        return orig_low, min(orig_high, math.floor(n / 2))

    return Rule(
        level="A",
        reason=(
            "min_samples_leaf upper bound <= floor(n / 2): larger leaves "
            "cannot be reached on this sample size"
        ),
        apply=_apply,
    )


def _rule_min_samples_leaf_b() -> Rule:
    """Level B: ``min_samples_leaf`` in ``[2, floor(n / 4)]``.

    On ``n < 8`` the range ``[2, floor(n / 4)]`` is empty; instead of
    collapsing it to the top (``min_samples_leaf = 1``, which drops the leaf
    regularization entirely) the lower bound 2 is kept, capped by the
    physical limit ``floor(n / 2)`` of level A.
    """

    def _apply(
        orig_low: float,
        orig_high: float,
        n: int,
        p: int,
        leaf_min: float,
    ) -> tuple[float, float]:
        low = max(orig_low, _MIN_SAMPLES_LEAF_B_LOW)
        b_high = math.floor(n / _MIN_SAMPLES_LEAF_B_DIVISOR)
        if b_high < _MIN_SAMPLES_LEAF_B_LOW:
            # The regularization range [2, floor(n/4)] itself is empty
            # (n < 8): keep the lower bound 2, capped by the physical
            # limit floor(n / 2) — the leaf regularization must not
            # silently vanish on tiny data.
            high = min(orig_high, math.floor(n / _MIN_SAMPLES_LEAF_B_PHYSICAL_DIVISOR))
        else:
            high = min(orig_high, b_high)
        return low, high

    return Rule(
        level="B",
        reason=(
            f"min_samples_leaf >= {_MIN_SAMPLES_LEAF_B_LOW:g}: on n < 8 the "
            f"range [{_MIN_SAMPLES_LEAF_B_LOW:g}, "
            f"floor(n / {_MIN_SAMPLES_LEAF_B_DIVISOR:g})] is empty, so the "
            "lower bound is capped by the physical limit floor(n / 2) "
            "instead of collapsing the leaf regularization"
        ),
        apply=_apply,
    )


def _rule_max_depth_a() -> Rule:
    """Level A: ``max_depth`` not above ``floor(n / leaf_min) - 1``."""

    def _apply(
        orig_low: float,
        orig_high: float,
        n: int,
        p: int,
        leaf_min: float,
    ) -> tuple[float, float]:
        limit = math.floor(n / max(leaf_min, 1.0)) - 1
        return orig_low, min(orig_high, limit)

    return Rule(
        level="A",
        reason=(
            "max_depth <= floor(n / leaf_min) - 1: the maximum reachable "
            "depth even for an unbalanced tree"
        ),
        apply=_apply,
    )


def _rule_max_depth_b() -> Rule:
    """Level B (trees): ``max_depth`` not above ``ceil(log2(n / leaf_min))``."""

    def _apply(
        orig_low: float,
        orig_high: float,
        n: int,
        p: int,
        leaf_min: float,
    ) -> tuple[float, float]:
        limit = math.ceil(math.log2(max(n, 1) / max(leaf_min, 1.0)))
        return orig_low, min(orig_high, limit)

    return Rule(
        level="B",
        reason=("max_depth <= ceil(log2(n / leaf_min)): regularization on small data"),
        apply=_apply,
    )


def _rule_boost_max_depth_b() -> Rule:
    """Level B (boosting): ``max_depth`` in the target range ``[1, 3]``."""

    def _apply(
        orig_low: float,
        orig_high: float,
        n: int,
        p: int,
        leaf_min: float,
    ) -> tuple[float, float]:
        # The rule's target range is absolute: on small data shallower trees
        # are allowed (the default's lower bound may be loosened).
        return _BOOST_MAX_DEPTH_B_LOW, min(orig_high, _BOOST_MAX_DEPTH_B_HIGH)

    return Rule(
        level="B",
        reason=(
            f"max_depth in [{_BOOST_MAX_DEPTH_B_LOW:g}, "
            f"{_BOOST_MAX_DEPTH_B_HIGH:g}]: regularization of boosting on "
            "small data (stumps are allowed on tiny data)"
        ),
        apply=_apply,
    )


def _rule_medium_max_depth() -> Rule:
    """Level B (medium): ``max_depth`` not above ``ceil(log2(n / leaf_min)) + 2``."""

    def _apply(
        orig_low: float,
        orig_high: float,
        n: int,
        p: int,
        leaf_min: float,
    ) -> tuple[float, float]:
        limit = (
            math.ceil(math.log2(max(n, 1) / max(leaf_min, 1.0))) + _MEDIUM_DEPTH_SLACK
        )
        return orig_low, min(orig_high, limit)

    return Rule(
        level="B",
        reason=(
            f"max_depth <= ceil(log2(n / leaf_min)) + {_MEDIUM_DEPTH_SLACK}: "
            "mild regularization on medium data"
        ),
        apply=_apply,
    )


def _rule_medium_min_samples_leaf_b() -> Rule:
    """Level B (medium): ``min_samples_leaf`` lower bound is 2."""

    def _apply(
        orig_low: float,
        orig_high: float,
        n: int,
        p: int,
        leaf_min: float,
    ) -> tuple[float, float]:
        return max(orig_low, _MIN_SAMPLES_LEAF_B_LOW), orig_high

    return Rule(
        level="B",
        reason=(
            f"min_samples_leaf >= {_MIN_SAMPLES_LEAF_B_LOW:g}: mild "
            "regularization on medium data"
        ),
        apply=_apply,
    )


def _rule_max_features_a() -> Rule:
    """Level A: ``max_features`` lower bound not below ``1 / p``."""

    def _apply(
        orig_low: float,
        orig_high: float,
        n: int,
        p: int,
        leaf_min: float,
    ) -> tuple[float, float]:
        return max(orig_low, 1.0 / max(p, 1)), orig_high

    return Rule(
        level="A",
        reason="max_features >= 1 / p: at least one feature must be sampled",
        apply=_apply,
    )


def _rule_n_estimators_time() -> Rule:
    """Time-only (bagging, RF/Extra trees): ``n_estimators`` upper bound
    capped at 300.

    Bagging ensembles (RF, Extra trees) average independent trees: more trees
    only reduce the variance of the ensemble, so the cap is a time-only
    restriction, NOT a regularization. Boosting uses its own level B rule
    (:func:`_rule_boost_n_estimators_b`).
    """

    def _apply(
        orig_low: float,
        orig_high: float,
        n: int,
        p: int,
        leaf_min: float,
    ) -> tuple[float, float]:
        return orig_low, min(orig_high, _N_ESTIMATORS_TIME_HIGH)

    return Rule(
        level="time",
        reason=(
            f"n_estimators <= {_N_ESTIMATORS_TIME_HIGH:g}: capped for time "
            "only — bagging averages independent trees, more trees reduce "
            "variance, not a regularization"
        ),
        apply=_apply,
    )


def _boost_n_estimators_cap(lr_high: float | None) -> float:
    """Level B upper bound of ``n_estimators`` for boosting.

    ``cap = budget / lr_high`` where ``lr_high`` is the upper bound of
    ``learning_rate`` in the space: because every sampled ``learning_rate``
    is at most ``lr_high``, the product ``n_estimators * learning_rate``
    never exceeds the budget. A missing/non-positive ``lr_high`` falls back
    to :data:`_BOOST_DEFAULT_LR_HIGH`, which yields the default cap 200.

    Args:
        lr_high: The upper bound of ``learning_rate`` in the space, or
            ``None`` when the parameter is absent.

    Returns:
        float: The derived cap (>= 1).
    """
    effective_lr_high = (
        lr_high if lr_high is not None and lr_high > 0 else _BOOST_DEFAULT_LR_HIGH
    )
    return max(1.0, _BOOST_N_ESTIMATORS_BUDGET / effective_lr_high)


def _learning_rate_high(space: dict[str, Any]) -> float | None:
    """Upper bound of ``learning_rate`` in a space, or ``None``.

    ``learning_rate`` of GradientBoosting/XGBoost is a positive float range;
    anything else (absent, categorical, int-like) returns ``None`` and the
    default :data:`_BOOST_DEFAULT_LR_HIGH` is used by
    :func:`_boost_n_estimators_cap`.

    Args:
        space: The search space dict (parameter name -> entry).

    Returns:
        float | None: The upper bound, or ``None`` when it cannot be read.
    """
    entry = space.get("learning_rate")
    if not isinstance(entry, SearchSpaceEntry):
        return None
    config = entry.config
    if not isinstance(config, (IntSpace, FloatSpace)):
        return None
    return float(config.high)


def _rule_boost_n_estimators_b(lr_high: float | None = None) -> Rule:
    """Level B (boosting): ``n_estimators`` upper bound capped by the budget.

    For a boosting ensemble the number of trees together with
    ``learning_rate`` is the main complexity regulator: every tree is fitted
    on the residual of the previous ones, so more trees (and a higher
    learning rate) aggressively increase the model's capacity to overfit.
    On small data the budget ``n_estimators * learning_rate`` is limited:
    the rule caps ``n_estimators`` at ``budget / lr_high``, where
    ``lr_high`` is the UPPER bound of ``learning_rate`` in the space (see
    :func:`_boost_n_estimators_cap`). Because ``learning_rate <= lr_high``
    holds for every sampled value, the product ``n_estimators *
    learning_rate`` never exceeds the budget. With the default
    ``learning_rate`` range [0.01, 0.3] of GradientBoosting/XGBoost the cap
    is ``200 = 60 / 0.3``; a higher lr upper bound tightens the cap, a lower
    one loosens it. The rule is switched off together with ``regularize``
    (level B).

    A strict per-trial budget (sampling one parameter conditionally on the
    other) is not supported by the guard's per-parameter rules; the
    upper-bound derivation above is the closest the space mechanism allows
    and still bounds the product for ALL sampled pairs (documented in
    API_REFERENCE.md).

    Args:
        lr_high: The upper bound of ``learning_rate`` in the space. ``None``
            (or a non-positive value) — the default
            :data:`_BOOST_DEFAULT_LR_HIGH`.

    Returns:
        Rule: The level B ``n_estimators`` rule with the derived cap.
    """
    cap = _boost_n_estimators_cap(lr_high)
    effective_lr_high = (
        lr_high if lr_high is not None and lr_high > 0 else _BOOST_DEFAULT_LR_HIGH
    )

    def _apply(
        orig_low: float,
        orig_high: float,
        n: int,
        p: int,
        leaf_min: float,
    ) -> tuple[float, float]:
        return orig_low, min(orig_high, cap)

    return Rule(
        level="B",
        reason=(
            f"n_estimators <= {cap:g} (= budget "
            f"{_BOOST_N_ESTIMATORS_BUDGET:g} / learning_rate upper bound "
            f"{effective_lr_high:g}): boosting regularization on small data "
            "— tree count and learning_rate are the main complexity "
            "regulator of a boosting ensemble"
        ),
        apply=_apply,
    )


def _rule_min_child_weight_b() -> Rule:
    """Level B: XGBoost ``min_child_weight`` lower bound is 2."""

    def _apply(
        orig_low: float,
        orig_high: float,
        n: int,
        p: int,
        leaf_min: float,
    ) -> tuple[float, float]:
        return max(orig_low, _MIN_CHILD_WEIGHT_B_LOW), orig_high

    return Rule(
        level="B",
        reason=(
            f"min_child_weight >= {_MIN_CHILD_WEIGHT_B_LOW:g}: for a squared "
            "loss this is roughly the minimum number of rows in a leaf"
        ),
        apply=_apply,
    )


def _rule_subsample_b() -> Rule:
    """Level B: XGBoost ``subsample`` lower bound is 0.7."""

    def _apply(
        orig_low: float,
        orig_high: float,
        n: int,
        p: int,
        leaf_min: float,
    ) -> tuple[float, float]:
        return max(orig_low, _SUBSAMPLE_B_LOW), orig_high

    return Rule(
        level="B",
        reason=(
            f"subsample >= {_SUBSAMPLE_B_LOW:g}: with small n a lower "
            "subsample leaves too few rows per tree"
        ),
        apply=_apply,
    )


def _rule_colsample_a() -> Rule:
    """Level A: ``colsample_bytree`` lower bound not below ``1 / p``."""

    def _apply(
        orig_low: float,
        orig_high: float,
        n: int,
        p: int,
        leaf_min: float,
    ) -> tuple[float, float]:
        return max(orig_low, 1.0 / max(p, 1)), orig_high

    return Rule(
        level="A",
        reason="colsample_bytree >= 1 / p: at least one feature per tree",
        apply=_apply,
    )


def _rule_reg_lambda_b() -> Rule:
    """Level B: XGBoost ``reg_lambda`` in ``[1, 100]`` on a log scale."""

    def _apply(
        orig_low: float,
        orig_high: float,
        n: int,
        p: int,
        leaf_min: float,
    ) -> tuple[float, float]:
        low = max(orig_low, _REG_LAMBDA_B_LOW)
        high = min(orig_high, _REG_LAMBDA_B_HIGH)
        return low, high

    return Rule(
        level="B",
        reason=(
            f"reg_lambda in [{_REG_LAMBDA_B_LOW:g}, {_REG_LAMBDA_B_HIGH:g}] "
            "on a log scale: regularization of XGBoost on small data"
        ),
        apply=_apply,
    )


def _rule_xgb_max_depth_medium() -> Rule:
    """Level B (medium): XGBoost ``max_depth`` not above 5."""

    def _apply(
        orig_low: float,
        orig_high: float,
        n: int,
        p: int,
        leaf_min: float,
    ) -> tuple[float, float]:
        return orig_low, min(orig_high, _XGB_MAX_DEPTH_MEDIUM_HIGH)

    return Rule(
        level="B",
        reason=(
            f"max_depth <= {_XGB_MAX_DEPTH_MEDIUM_HIGH:g}: mild "
            "regularization of XGBoost on medium data"
        ),
        apply=_apply,
    )


#: Algorithm families covered by the guard. ``decision_tree``,
#: ``random_forest`` and ``extra_trees`` share the tree rules;
#: ``gradient_boosting`` and ``xgboosting`` the boosting rules.
TREE_ALGORITHMS = frozenset({"decision_tree", "random_forest", "extra_trees"})
BOOSTING_ALGORITHMS = frozenset({"gradient_boosting", "xgboosting"})

#: Declarative rule table: ``(size_mode, is_boosting) -> [(param, rule), ...]``.
#: Rules are applied in order; a parameter is restricted only when it is
#: actually present in the space (so the defaults adapt automatically and a
#: user range is intersected).
_SMALL_TREE_RULES: list[tuple[str, Rule]] = [
    ("min_samples_leaf", _rule_min_samples_leaf_a()),
    ("min_samples_leaf", _rule_min_samples_leaf_b()),
    ("max_depth", _rule_max_depth_a()),
    ("max_depth", _rule_max_depth_b()),
    ("max_features", _rule_max_features_a()),
    # Bagging (RF/Extra trees): the n_estimators cap is time-only — more
    # trees only reduce variance, not a regularization.
    ("n_estimators", _rule_n_estimators_time()),
]


def _small_boosting_rules(lr_high: float | None = None) -> list[tuple[str, Rule]]:
    """Rules of the small boosting mode.

    The ``n_estimators`` level B cap depends on the UPPER bound of
    ``learning_rate`` in the space (``budget / lr_high``, see
    :func:`_rule_boost_n_estimators_b`), so the rule table is built per run
    with the space's ``learning_rate``.

    Args:
        lr_high: The upper bound of ``learning_rate`` in the space (``None``
            — the default :data:`_BOOST_DEFAULT_LR_HIGH`).

    Returns:
        list[tuple[str, Rule]]: The small boosting rule table.
    """
    return [
        # min_samples_leaf (GradientBoosting only) — the same rules as for trees.
        ("min_samples_leaf", _rule_min_samples_leaf_a()),
        ("min_samples_leaf", _rule_min_samples_leaf_b()),
        ("max_depth", _rule_max_depth_a()),
        ("max_depth", _rule_boost_max_depth_b()),
        ("min_child_weight", _rule_min_child_weight_b()),
        ("subsample", _rule_subsample_b()),
        ("colsample_bytree", _rule_colsample_a()),
        ("reg_lambda", _rule_reg_lambda_b()),
        # Boosting: the n_estimators cap is a level B REGULARIZATION rule —
        # the tree count together with learning_rate is the main complexity
        # regulator of a boosting ensemble (disabled with ``regularize``).
        ("n_estimators", _rule_boost_n_estimators_b(lr_high)),
    ]


_SMALL_BOOSTING_RULES_DEFAULT = _small_boosting_rules()

_MEDIUM_TREE_RULES: list[tuple[str, Rule]] = [
    ("min_samples_leaf", _rule_min_samples_leaf_a()),
    ("min_samples_leaf", _rule_medium_min_samples_leaf_b()),
    ("max_depth", _rule_medium_max_depth()),
    ("max_features", _rule_max_features_a()),
]

_MEDIUM_BOOSTING_RULES: list[tuple[str, Rule]] = [
    ("min_samples_leaf", _rule_min_samples_leaf_a()),
    ("min_samples_leaf", _rule_medium_min_samples_leaf_b()),
    ("max_depth", _rule_medium_max_depth()),
    ("max_depth", _rule_xgb_max_depth_medium()),
    ("min_child_weight", _rule_min_child_weight_b()),
    ("subsample", _rule_subsample_b()),
    ("colsample_bytree", _rule_colsample_a()),
    ("reg_lambda", _rule_reg_lambda_b()),
]

_RULE_TABLE: dict[tuple[str, bool], list[tuple[str, Rule]]] = {
    ("small", False): _SMALL_TREE_RULES,
    ("small", True): _SMALL_BOOSTING_RULES_DEFAULT,
    ("medium", False): _MEDIUM_TREE_RULES,
    ("medium", True): _MEDIUM_BOOSTING_RULES,
}
