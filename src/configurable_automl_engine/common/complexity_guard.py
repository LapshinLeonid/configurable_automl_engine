"""Adaptive complexity guard for tree and boosting search spaces.

The module extends the existing data-dependent clipping layer
(:mod:`configurable_automl_engine.common.hyperopt_defaults`): on small data
the search ranges of tree/boosting hyperparameters are too wide, trials are
spent on values that do not change the model, the TPE surface has flat
regions and the search drifts toward overfit models. This guard tightens the
ranges depending on the number of rows (``n``) and the number of features
after preprocessing (``p``).

Two independent levels of restrictions:

* **Level A — lossless clipping.** Removes values that cannot change the
  model on data of the given size (for example, ``max_depth=25`` on 15 rows
  is equivalent to ``max_depth=6``). Safe, enabled by default. For
  ``min_samples_leaf`` the level A cap is a *physical* bound (larger leaves
  are unreachable on that sample size), not a claim about model quality:
  values below the cap still change the model.
* **Level B — regularization.** Deliberately shifts the space toward simpler
  models on small data. A heuristic, enabled by default and switched off
  independently.
* **Time-only restrictions** (``n_estimators`` caps) are marked explicitly as
  "for time" rather than "for regularization": more trees only reduce the
  variance of the ensemble.

The rules are declared declaratively (a rule table per algorithm and
parameter) so that future rules (kNN ``k=1`` on small data, an upper bound on
SVR ``C``) can be added without rewriting the mechanism.

Configuration is passed as a plain dataclass (:class:`ComplexityGuardConfig`)
so that the module stays decoupled from the Pydantic config schema
(:mod:`configurable_automl_engine.training_engine.config_parser`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from configurable_automl_engine.common.hyperopt_defaults import (
    FloatSpace,
    IntSpace,
    SearchSpaceEntry,
)

# ─────────────────────────────── configuration ─────────────────────────────── #


@dataclass
class ComplexityGuardConfig:
    """Runtime configuration of the complexity guard.

    Attributes:
        mode: ``"off"`` — the guard is disabled; ``"warn"`` — restrictions
            are computed and logged but NOT applied; ``"apply"`` — computed
            and applied.
        lossless: Enable level A (lossless clipping).
        regularize: Enable level B (regularization heuristic).
        user_range_policy: ``"user_wins"`` — on an empty intersection with a
            user range the user range stays (an explicit setting beats the
            heuristic); ``"engine_wins"`` — the guard restriction is applied.
        small_n_threshold: ``n < threshold`` marks the "small" mode.
        small_np_ratio_threshold: ``n / p < threshold`` marks the "small"
            mode.
        medium_n_threshold: ``n < threshold`` marks the "medium" mode.
        medium_np_ratio_threshold: ``n / p < threshold`` marks the "medium"
            mode.
        size_basis: ``"rows"`` — ``n`` counts rows; ``"groups"`` — reserved
            for group-based validation (repeats of one group carry less
            information than independent rows). Only ``"rows"`` is used
            today.
    """

    mode: str = "apply"
    lossless: bool = True
    regularize: bool = True
    user_range_policy: str = "user_wins"
    small_n_threshold: int = 50
    small_np_ratio_threshold: float = 5.0
    medium_n_threshold: int = 300
    medium_np_ratio_threshold: float = 20.0
    size_basis: str = "rows"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ComplexityGuardConfig:
        """Build a config from a plain dict (e.g. ``model_dump`` of the schema).

        Accepts either a mapping or a Pydantic model exposing ``model_dump``
        (normalized with ``mode='json'`` so that enum fields become their
        string values). Unknown keys are ignored so that forward/backward
        compatibility with the config schema is preserved.

        Args:
            data: A mapping of field names to values, a Pydantic model, or
                ``None``.

        Returns:
            ComplexityGuardConfig: The populated config (defaults for missing
                keys).
        """
        if not data:
            return cls()
        if hasattr(data, "model_dump"):
            data = data.model_dump(mode="json")
        allowed = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in allowed})


# ─────────────────────────────── size mode ─────────────────────────────────── #


def resolve_size_mode(n: int, p: int, cfg: ComplexityGuardConfig) -> str:
    """Resolve the data-size mode by ``n`` and the ratio ``n / p``.

    Rules:
    * ``"small"``: ``n < small_n_threshold`` or ``n / p < small_np_ratio_threshold``;
    * ``"medium"``: ``n < medium_n_threshold`` or ``n / p < medium_np_ratio_threshold``
      (and not small);
    * ``"ordinary"``: everything else.

    Args:
        n: Number of rows in the training fold (``n_samples_eff``).
        p: Number of features after preprocessing (what the model sees).
        cfg: The guard configuration (holds the thresholds).

    Returns:
        str: ``"small"``, ``"medium"`` or ``"ordinary"``.
    """
    if n < cfg.small_n_threshold or _ratio(n, p) < cfg.small_np_ratio_threshold:
        return "small"
    if n < cfg.medium_n_threshold or _ratio(n, p) < cfg.medium_np_ratio_threshold:
        return "medium"
    return "ordinary"


def _ratio(n: int, p: int) -> float:
    """Return ``n / p`` guarding against a zero feature count."""
    if p <= 0:
        return float("inf")
    return n / p


# ─────────────────────────────── rule engine ───────────────────────────────── #


@dataclass(frozen=True)
class Rule:
    """A single declarative restriction for a hyperparameter.

    Attributes:
        level: ``"A"`` — lossless clipping, ``"B"`` — regularization,
            ``"time"`` — time-only restriction.
        reason: Human-readable reason for logs and reports.
        apply: Callable that computes the TARGET range ``(low, high)`` of the
            parameter from the original bounds ``(orig_low, orig_high)`` and
            the data quantities ``(n, p, leaf_min)``. Rules only tighten: the
            target's unconstrained bound equals the original one.
    """

    level: str
    reason: str
    apply: Callable[[float, float, int, int, float], tuple[float, float]]


# Lower bound for ``min_samples_leaf`` on level B (the issue's "2–3"; the
# documented default is 2).
_MIN_SAMPLES_LEAF_B_LOW = 2.0
# Upper bound for ``min_samples_leaf`` on level B in the small mode.
# ``floor(n / 4)`` with the leaf bound is used as a base multiplier.
_MIN_SAMPLES_LEAF_B_DIVISOR = 4.0

# Lower bound for XGBoost ``min_child_weight`` on level B (for a squared loss
# this is roughly the minimum number of rows in a leaf).
_MIN_CHILD_WEIGHT_B_LOW = 2.0

# Minimum ``subsample`` fraction on level B.
_SUBSAMPLE_B_LOW = 0.7

# Bounds of XGBoost ``reg_lambda`` on level B (log scale).
_REG_LAMBDA_B_LOW = 1.0
_REG_LAMBDA_B_HIGH = 100.0

# Target range of ``max_depth`` on level B for boosting in the small mode
# (the issue's "1–3").
_BOOST_MAX_DEPTH_B_LOW = 1.0
_BOOST_MAX_DEPTH_B_HIGH = 3.0

# Upper bound of XGBoost ``max_depth`` in the medium mode.
_XGB_MAX_DEPTH_MEDIUM_HIGH = 5.0

# Time-only upper bound of ``n_estimators`` in the small mode (the issue's
# "100–300"; only the upper bound is capped — more trees reduce variance).
_N_ESTIMATORS_TIME_HIGH = 300.0

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
    (``floor(n / 2)``, ``[2, floor(n / 4)]``). When the range is a fraction
    of the sample size (float in (0, 1)), the bounds are converted to rows
    (``value * n``) before the rule applies and divided back afterwards.

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
    compute an empty range whose bounds would collapse to 0 (e.g.
    ``min_samples_leaf`` level B on ``n=2`` yields ``[2, 0]``) — such a range
    is fixed at 1, never at 0, because sklearn/XGBoost reject 0
    (``InvalidParameterError`` / silently overridden depth).

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
    """Level B: ``min_samples_leaf`` in ``[2, floor(n / 4)]``."""

    def _apply(
        orig_low: float,
        orig_high: float,
        n: int,
        p: int,
        leaf_min: float,
    ) -> tuple[float, float]:
        low = max(orig_low, _MIN_SAMPLES_LEAF_B_LOW)
        high = min(orig_high, math.floor(n / _MIN_SAMPLES_LEAF_B_DIVISOR))
        return low, high

    return Rule(
        level="B",
        reason=(
            f"min_samples_leaf in [{_MIN_SAMPLES_LEAF_B_LOW:g}, "
            f"floor(n / {_MIN_SAMPLES_LEAF_B_DIVISOR:g})]: regularization on "
            "small data"
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
    """Time-only: ``n_estimators`` upper bound capped at 300."""

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
            "only (more trees reduce variance, not a regularization)"
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
    ("n_estimators", _rule_n_estimators_time()),
]

_SMALL_BOOSTING_RULES: list[tuple[str, Rule]] = [
    # min_samples_leaf (GradientBoosting only) — the same rules as for trees.
    ("min_samples_leaf", _rule_min_samples_leaf_a()),
    ("min_samples_leaf", _rule_min_samples_leaf_b()),
    ("max_depth", _rule_max_depth_a()),
    ("max_depth", _rule_boost_max_depth_b()),
    ("min_child_weight", _rule_min_child_weight_b()),
    ("subsample", _rule_subsample_b()),
    ("colsample_bytree", _rule_colsample_a()),
    ("reg_lambda", _rule_reg_lambda_b()),
    ("n_estimators", _rule_n_estimators_time()),
]

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
    ("small", True): _SMALL_BOOSTING_RULES,
    ("medium", False): _MEDIUM_TREE_RULES,
    ("medium", True): _MEDIUM_BOOSTING_RULES,
}


# ─────────────────────────────── application ───────────────────────────────── #


def _leaf_size_param(space: dict[str, Any], is_boosting: bool) -> str:
    """Name of the parameter that defines the leaf size for ``max_depth``.

    For boosting families the leaf size comes from ``min_child_weight``
    (approximately the minimum rows in a leaf for a squared loss) when it is
    present; otherwise from ``min_samples_leaf``. Trees use
    ``min_samples_leaf``.

    Args:
        space: The search space dict.
        is_boosting: Whether the algorithm is a boosting family.

    Returns:
        str: The parameter name, or ``""`` when none is present.
    """
    if is_boosting and "min_child_weight" in space:
        return "min_child_weight"
    if "min_samples_leaf" in space:
        return "min_samples_leaf"
    return ""


def _effective_leaf_min(
    space: dict[str, Any],
    rules: list[tuple[str, Rule]],
    *,
    cfg: ComplexityGuardConfig,
    n: int,
    p: int,
    is_boosting: bool,
) -> float:
    """Compute the effective minimum leaf size used by ``max_depth`` rules.

    For trees and GradientBoosting the leaf size comes from
    ``min_samples_leaf`` (after the guard rules); for XGBoost from
    ``min_child_weight`` when present. When the parameter is absent, the
    sklearn/XGBoost default of 1 is used. A fractional ``min_samples_leaf``
    (float in (0, 1)) is converted to rows (``value * n``) before the rules
    apply, and the resulting absolute leaf size is returned.

    Args:
        space: The search space dict.
        rules: The rules of the current (size_mode, family).
        cfg: The guard configuration.
        n: Number of rows.
        p: Number of features.
        is_boosting: Whether the algorithm is a boosting family.

    Returns:
        float: The effective minimum leaf size (>= 1).
    """
    param = _leaf_size_param(space, is_boosting)
    if not param:
        return 1.0
    entry = space[param]
    config = entry.config
    if not isinstance(config, (IntSpace, FloatSpace)):
        return 1.0
    low, high = float(config.low), float(config.high)
    is_float = isinstance(config, FloatSpace)
    is_fraction = _is_fraction_param(param, is_float=is_float, low=low, high=high)
    for key, rule in rules:
        if key != param:
            continue
        if rule.level == "A" and not cfg.lossless:
            continue
        if rule.level == "B" and not cfg.regularize:
            continue
        low, high = _apply_rule(rule, low, high, n, p, low, param, is_fraction)
        low, high = _clamp_for_param(param, low, high, n, is_float=is_float)
    if is_fraction:
        # The effective absolute leaf size: the fraction (after the rules)
        # scaled to rows, at least one row.
        return max(1.0, low * n)
    return max(1.0, low)


def _build_entry(
    entry: SearchSpaceEntry,
    low: float,
    high: float,
) -> SearchSpaceEntry:
    """Build a deep-copied entry with the new bounds preserving type/log scale.

    Args:
        entry: The source entry.
        low: The new lower bound.
        high: The new upper bound.

    Returns:
        SearchSpaceEntry: A deep copy with updated bounds.
    """
    config = entry.config
    new_entry = entry.model_copy(deep=True)
    if isinstance(config, FloatSpace):
        new_entry.config = config.model_copy(update={"low": low, "high": high})
    else:
        low_i, high_i = round(low), round(high)
        low_i = min(low_i, high_i)
        new_entry.config = config.model_copy(update={"low": low_i, "high": high_i})
    return new_entry


def _domain_for_param(
    param: str, n: int, p: int, is_float: bool
) -> tuple[float, float]:
    """Full (unrestricted) domain of a parameter, for ``engine_wins``.

    Used when a user range conflicts with the guard: the guard restriction is
    computed from the parameter's full domain, not from the (possibly narrow)
    user range, so that e.g. ``min_samples_leaf`` level B yields ``[2, n/4]``
    regardless of the user's ``[1, 1]``.

    Args:
        param: The hyperparameter name.
        n: Number of rows.
        p: Number of features after preprocessing.
        is_float: Whether the parameter is a float in the search space (for
            ``max_features`` and ``min_samples_leaf`` the domain differs: a
            fraction vs a plain count).

    Returns:
        tuple[float, float]: The full domain ``(low, high)``.
    """
    if param == "min_samples_leaf" and is_float:
        # A float min_samples_leaf may be a fraction of the sample (0, 1).
        return 1e-9, 1.0
    if param in {"min_samples_leaf", "min_child_weight", "max_depth"}:
        return 1.0, float(max(n, 1))
    if param == "n_estimators":
        return 1.0, 10000.0
    if param in {"subsample", "colsample_bytree"}:
        return 1e-9, 1.0
    if param == "max_features":
        # float — a fraction of the features (0, 1]; int — a feature count.
        if is_float:
            return 1e-9, 1.0
        return 1.0, float(max(p, 1))
    if param == "reg_lambda":
        return 1e-9, 1e6
    return 0.0, 1e6


def _apply_rule_to_entry(
    entry: SearchSpaceEntry,
    rule: Rule,
    *,
    param: str,
    n: int,
    p: int,
    leaf_min: float,
    user_key: bool,
    cfg: ComplexityGuardConfig,
) -> tuple[SearchSpaceEntry, dict[str, Any] | None]:
    """Apply a single rule to one search-space entry.

    The rule computes a target range that tightens the parameter. For a
    default (non-user) parameter the target range replaces the default one
    (the default spaces adapt automatically); for a user parameter the target
    range is intersected with the user's range and, on an empty intersection,
    ``cfg.user_range_policy`` decides (by default the user range wins: an
    explicit setting beats the heuristic).

    In all cases the result is a valid non-empty range: ``low <= high``, the
    int/float type and the log scale are preserved, and a range collapsed to
    a single value fixes the parameter (logged).

    Args:
        entry: The search-space entry to restrict.
        rule: The rule to apply.
        param: The hyperparameter name.
        n: Number of rows.
        p: Number of features.
        leaf_min: Effective minimum leaf size (for ``max_depth`` rules).
        user_key: Whether the parameter was set explicitly by the user.
        cfg: The guard configuration.

    Returns:
        tuple[SearchSpaceEntry, dict | None]: The (possibly deep-copied)
            entry and a log record (``None`` when nothing changed).
    """
    config = entry.config
    if not isinstance(config, (IntSpace, FloatSpace)):
        return entry, None
    if rule.level == "A" and not cfg.lossless:
        return entry, None
    if rule.level == "B" and not cfg.regularize:
        return entry, None

    is_float = isinstance(config, FloatSpace)
    low, high = float(config.low), float(config.high)
    before = (low, high)
    is_fraction = _is_fraction_param(param, is_float=is_float, low=low, high=high)

    # The rule's tightened target range for this parameter. A fraction range
    # (e.g. float min_samples_leaf in (0, 1]) is converted to rows before
    # the rule applies and back afterwards.
    low_c, high_c = _apply_rule(rule, low, high, n, p, leaf_min, param, is_fraction)

    if user_key:
        # User range: the guard only narrows; the target range is intersected
        # BEFORE clamping so that a genuinely empty intersection is detected.
        new_low, new_high = max(low, low_c), min(high, high_c)
        conflict = new_low > new_high
        if conflict and cfg.user_range_policy == "user_wins":
            # The user range stays; the record carries ``before``/``after``
            # (both equal the user range) and the guard's target range in
            # ``rule_range`` so that the logger can report the conflict
            # without special-casing missing keys.
            return entry, {
                "conflict": True,
                "applied": False,
                "before": list(before),
                "after": list(before),
                "rule_range": [low_c, high_c],
            }
        if conflict:
            # engine_wins: the guard restriction is applied. The target is
            # recomputed from the full domain of the parameter so that the
            # restriction does not depend on the (narrow) user range.
            dom_low, dom_high = _domain_for_param(param, n, p, is_float)
            dom_fraction = _is_fraction_param(
                param, is_float=is_float, low=dom_low, high=dom_high
            )
            low_c, high_c = _apply_rule(
                rule, dom_low, dom_high, n, p, leaf_min, param, dom_fraction
            )
            low_c, high_c = _clamp_for_param(param, low_c, high_c, n, is_float=is_float)
            new_low, new_high = low_c, high_c
        else:
            low_c, high_c = _clamp_for_param(
                param, new_low, new_high, n, is_float=is_float
            )
            new_low, new_high = low_c, high_c
        after = (new_low, new_high)
        changed = before != after
        if not changed and not conflict:
            return entry, None
        new_entry = _build_entry(entry, *after)
        return new_entry, {
            "before": list(before),
            "after": list(after),
            "conflict": conflict,
            "applied": True,
            "fixed": after[0] == after[1],
        }

    # Default parameter: the target range replaces the default one. If the
    # target is a no-op (identical to the original), nothing changed.
    low_c, high_c = _clamp_for_param(param, low_c, high_c, n, is_float=is_float)
    after = (low_c, high_c)
    if before == after:
        return entry, None
    new_entry = _build_entry(entry, *after)
    return new_entry, {
        "before": list(before),
        "after": list(after),
        "conflict": False,
        "applied": True,
        "fixed": after[0] == after[1],
    }


def apply_complexity_guard(
    space: dict[str, Any],
    algo: str,
    *,
    n: int,
    p: int,
    config: ComplexityGuardConfig | None = None,
    user_space_keys: Iterable[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the adaptive complexity guard to a search space.

    Extends ``clip_search_space``: physical restrictions stay there, this
    guard tightens the tree/boosting ranges based on ``n`` and ``p``.

    Args:
        space: The search space dict (parameter name -> ``SearchSpaceEntry``).
        algo: The algorithm name (alias is resolved internally).
        n: Number of rows in the training fold (``n_samples_eff``).
        p: Number of features after preprocessing.
        config: The guard configuration; ``None`` — the defaults.
        user_space_keys: Names of parameters set explicitly by the user
            (``algo.hyperparameters``). Defaults to an empty set.

    Returns:
        tuple[dict[str, Any], list[dict[str, Any]]]: The (possibly adjusted)
            space and the list of log records. In ``warn`` mode the space is
            returned unchanged while the records describe the would-be
            changes; in ``off`` mode both are empty.
    """
    cfg = config or ComplexityGuardConfig()
    from configurable_automl_engine.models import resolve_algorithm_name

    algo = resolve_algorithm_name(algo)
    if cfg.mode == "off":
        return dict(space), []
    if algo not in TREE_ALGORITHMS and algo not in BOOSTING_ALGORITHMS:
        return dict(space), []

    is_boosting = algo in BOOSTING_ALGORITHMS
    size_mode = resolve_size_mode(n, p, cfg)
    if size_mode == "ordinary":
        return dict(space), []

    rules = _RULE_TABLE[(size_mode, is_boosting)]
    user_keys = set(user_space_keys or ())
    leaf_min = _effective_leaf_min(
        space, rules, cfg=cfg, n=n, p=p, is_boosting=is_boosting
    )

    result = dict(space)
    logs: list[dict[str, Any]] = []
    for param, rule in rules:
        entry = result.get(param)
        if not isinstance(entry, SearchSpaceEntry):
            continue
        applied, record = _apply_rule_to_entry(
            entry,
            rule,
            param=param,
            n=n,
            p=p,
            leaf_min=leaf_min,
            user_key=param in user_keys,
            cfg=cfg,
        )
        if record is None:
            continue
        record.update(
            {
                "algo": algo,
                "param": param,
                "level": rule.level,
                "size_mode": size_mode,
                "n": n,
                "p": p,
                "leaf_min": leaf_min,
                "reason": rule.reason,
                "user_range_policy": cfg.user_range_policy,
                "applied": record["applied"] and cfg.mode == "apply",
            }
        )
        logs.append(record)
        if cfg.mode == "apply" and record.get("applied"):
            result[param] = applied

    return result, logs
