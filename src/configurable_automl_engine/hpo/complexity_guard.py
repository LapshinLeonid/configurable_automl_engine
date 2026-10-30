"""Adaptive complexity guard for tree and boosting search spaces.

The module extends the existing data-dependent clipping layer
(:mod:`configurable_automl_engine.hpo.default_spaces`): on small data
the search ranges of tree/boosting hyperparameters are too wide, trials are
spent on values that do not change the model, the TPE surface has flat
regions and the search drifts toward overfit models. This guard tightens the
ranges depending on the number of rows (``n``) and the feature count.

Two feature counts are used deliberately:

* **Original columns** (a categorical column counts as 1) drive the
  size-mode ratio ``n / p``: the one-hot expansion must not push
  large-but-wide data into a smaller size mode;
* **Features after preprocessing** (what the model sees, including one-hot
  columns and missing-value indicators) feed the rules that depend on the
  feature count — ``max_features`` and ``colsample_bytree``.

Two independent levels of restrictions:

* **Level A — lossless clipping.** Removes values that cannot change the
  model on data of the given size (for example, ``max_depth=25`` on 15 rows
  is equivalent to ``max_depth=6``). Safe, enabled by default. For
  ``min_samples_leaf`` the level A cap is a *physical* bound (larger leaves
  are unreachable on that sample size), not a claim about model quality:
  values below the cap still change the model.
* **Level B — regularization.** Deliberately shifts the space toward simpler
  models on small data. A heuristic, enabled by default and switched off
  independently. For boosting this includes the ``n_estimators`` budget
  (``n_estimators`` x ``learning_rate`` is the main complexity knob of a
  boosting ensemble — see :func:`_rule_boost_n_estimators_b`).
* **Time-only restrictions** (``n_estimators`` caps of the BAGGING ensembles:
  RF and Extra trees) are marked explicitly as "for time" rather than "for
  regularization": bagging averages independent trees, so more trees only
  reduce the variance of the ensemble.

The rules are declared declaratively (a rule table per algorithm and
parameter) so that future rules (kNN ``k=1`` on small data, an upper bound on
SVR ``C``) can be added without rewriting the mechanism.

Configuration is passed as a plain dataclass (:class:`ComplexityGuardConfig`)
so that the module stays decoupled from the Pydantic config schema
(:mod:`configurable_automl_engine.training_engine.config_parser`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from configurable_automl_engine.hpo.complexity_rules import (
    _RULE_TABLE,
    BOOSTING_ALGORITHMS,
    TREE_ALGORITHMS,
    Rule,
    _apply_rule,
    _clamp_for_param,
    _is_fraction_param,
    _learning_rate_high,
    _small_boosting_rules,
)
from configurable_automl_engine.hpo.default_spaces import (
    FloatSpace,
    IntSpace,
    SearchSpaceEntry,
    align_entry_bounds,
)
from configurable_automl_engine.models import resolve_algorithm_name

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
            mode. The ratio uses the ORIGINAL column count (a categorical
            column counts as 1, before one-hot expansion).
        medium_n_threshold: ``n < threshold`` marks the "medium" mode.
        medium_np_ratio_threshold: ``n / p < threshold`` marks the "medium"
            mode. The ratio uses the ORIGINAL column count (a categorical
            column counts as 1, before one-hot expansion).
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


def resolve_size_mode(n: int, p_ratio: int, cfg: ComplexityGuardConfig) -> str:
    """Resolve the data-size mode by ``n`` and the ratio ``n / p``.

    The ratio uses the ORIGINAL column count (a categorical column counts
    as 1, i.e. before the one-hot expansion): the number of features the
    model sees after preprocessing is intentionally NOT used here, so that
    large-but-wide data (one-hot expanded) is not pushed into a smaller size
    mode. The post-preprocessing feature count still feeds the rules that
    depend on it (``max_features``, ``colsample_bytree``).

    Rules:
    * ``"small"``: ``n < small_n_threshold`` or ``n / p < small_np_ratio_threshold``;
    * ``"medium"``: ``n < medium_n_threshold`` or ``n / p < medium_np_ratio_threshold``
      (and not small);
    * ``"ordinary"``: everything else.

    Args:
        n: Number of rows in the training fold (``n_samples_eff``).
        p_ratio: Number of ORIGINAL columns for the ratio (a categorical
            column counts as 1).
        cfg: The guard configuration (holds the thresholds).

    Returns:
        str: ``"small"``, ``"medium"`` or ``"ordinary"``.
    """
    if n < cfg.small_n_threshold or _ratio(n, p_ratio) < cfg.small_np_ratio_threshold:
        return "small"
    if n < cfg.medium_n_threshold or _ratio(n, p_ratio) < cfg.medium_np_ratio_threshold:
        return "medium"
    return "ordinary"


def _ratio(n: int, p: int) -> float:
    """Return ``n / p`` guarding against a zero feature count."""
    if p <= 0:
        return float("inf")
    return n / p


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


def _effective_leaf_min(param: str, entry: SearchSpaceEntry | None, n: int) -> float:
    """Compute the effective minimum leaf size from a leaf-size range.

    For trees and GradientBoosting the leaf size comes from
    ``min_samples_leaf``; for XGBoost from ``min_child_weight`` when present.
    The entry is the FINAL range of the parameter (after the guard rules and
    the ``user_range_policy`` were applied), so the value reflects the
    resolved range — not the raw user range. When the parameter is absent,
    the sklearn/XGBoost default of 1 is used. A fractional
    ``min_samples_leaf`` (float in (0, 1)) is converted to rows
    (``value * n``) and the resulting absolute leaf size is returned.

    Args:
        param: The leaf-size parameter name (may be empty).
        entry: The final ``SearchSpaceEntry`` of the parameter.
        n: Number of rows.

    Returns:
        float: The effective minimum leaf size (>= 1).
    """
    if not param or entry is None:
        return 1.0
    config = entry.config
    if not isinstance(config, (IntSpace, FloatSpace)):
        return 1.0
    low, high = float(config.low), float(config.high)
    is_float = isinstance(config, FloatSpace)
    is_fraction = _is_fraction_param(param, is_float=is_float, low=low, high=high)
    if is_fraction:
        # The effective absolute leaf size: the fraction scaled to rows, at
        # least one row.
        return max(1.0, low * n)
    return max(1.0, low)


def _build_entry(
    entry: SearchSpaceEntry,
    low: float,
    high: float,
) -> SearchSpaceEntry:
    """Build a deep-copied entry with the new bounds preserving type/log scale.

    The upper bound is re-aligned to the step grid of a stepped range (e.g.
    an int step): a rule may compute ``high`` that does not lie on the grid,
    and Optuna would silently narrow such a range (``high`` is shifted down
    to the last grid point). When the aligned grid becomes empty, the range
    collapses to a single grid value — the parameter is fixed.

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
    return align_entry_bounds(new_entry)


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
        # The record documents the FINAL bounds of the parameter: for a
        # stepped range the upper bound may be re-aligned to the step grid
        # inside ``_build_entry`` (``after`` would otherwise claim a range
        # Optuna silently narrows).
        final_config = new_entry.config
        if isinstance(final_config, (IntSpace, FloatSpace)):
            final_after = (float(final_config.low), float(final_config.high))
        else:
            # Unreachable: ``_build_entry`` is only called on numeric ranges.
            final_after = after
        return new_entry, {
            "before": list(before),
            "after": list(final_after),
            "conflict": conflict,
            "applied": True,
            "fixed": final_after[0] == final_after[1],
        }

    # Default parameter: the target range replaces the default one. If the
    # target is a no-op (identical to the original), nothing changed.
    low_c, high_c = _clamp_for_param(param, low_c, high_c, n, is_float=is_float)
    after = (low_c, high_c)
    if before == after:
        return entry, None
    new_entry = _build_entry(entry, *after)
    # The record documents the FINAL bounds (the upper bound may be
    # re-aligned to the step grid inside ``_build_entry``).
    final_config = new_entry.config
    if isinstance(final_config, (IntSpace, FloatSpace)):
        final_after = (float(final_config.low), float(final_config.high))
    else:
        # Unreachable: ``_build_entry`` is only called on numeric ranges.
        final_after = after
    return new_entry, {
        "before": list(before),
        "after": list(final_after),
        "conflict": False,
        "applied": True,
        "fixed": final_after[0] == final_after[1],
    }


def apply_complexity_guard(
    space: dict[str, Any],
    algo: str,
    *,
    n: int,
    p: int,
    config: ComplexityGuardConfig | None = None,
    user_space_keys: Iterable[str] | None = None,
    p_size: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the adaptive complexity guard to a search space.

    Extends ``clip_search_space``: physical restrictions stay there, this
    guard tightens the tree/boosting ranges based on ``n`` and the feature
    count.

    The size mode (``small``/``medium``/``ordinary``) is resolved from the
    ratio ``n / p_size``, where ``p_size`` is the number of ORIGINAL columns
    (a categorical column counts as 1, before the one-hot expansion): the
    one-hot expansion must not push large-but-wide data into a smaller size
    mode. When ``p_size`` is not given, it falls back to ``p`` (the
    pre-fix behavior, kept for direct callers). The rules that depend on the
    feature count (``max_features``, ``colsample_bytree``) always use ``p`` —
    the number of features after preprocessing (what the model sees).

    Args:
        space: The search space dict (parameter name -> ``SearchSpaceEntry``).
        algo: The algorithm name (case-insensitive).
        n: Number of rows in the training fold (``n_samples_eff``).
        p: Number of features after preprocessing.
        config: The guard configuration; ``None`` — the defaults.
        user_space_keys: Names of parameters set explicitly by the user
            (``algo.hyperparameters``). Defaults to an empty set.
        p_size: Number of ORIGINAL columns (a categorical column counts as
            1) used only for the size-mode ratio ``n / p``. ``None`` — fall
            back to ``p``.

    Returns:
        tuple[dict[str, Any], list[dict[str, Any]]]: The (possibly adjusted)
            space and the list of log records. In ``warn`` mode the space is
            returned unchanged while the records describe the would-be
            changes; in ``off`` mode both are empty.
    """
    cfg = config or ComplexityGuardConfig()
    algo = resolve_algorithm_name(algo)
    if cfg.mode == "off":
        return dict(space), []
    if algo not in TREE_ALGORITHMS and algo not in BOOSTING_ALGORITHMS:
        return dict(space), []

    is_boosting = algo in BOOSTING_ALGORITHMS
    size_mode = resolve_size_mode(n, p_size if p_size is not None else p, cfg)
    if size_mode == "ordinary":
        return dict(space), []

    rules = _RULE_TABLE[(size_mode, is_boosting)]
    if is_boosting and size_mode == "small":
        # The boosting n_estimators level B cap depends on the space's
        # learning_rate upper bound — rebuild the small boosting table with
        # the actual lr range of this run.
        rules = _small_boosting_rules(_learning_rate_high(space))
    user_keys = set(user_space_keys or ())
    result = dict(space)
    logs: list[dict[str, Any]] = []
    # Would-be computed entries (even of an unapplied warn run) so that the
    # resolved leaf range is readable after the rules.
    computed_entries: dict[str, SearchSpaceEntry] = {}

    def _run_rules(
        params: Iterable[str],
        leaf_min: float,
    ) -> None:
        for param, rule in rules:
            if param not in params:
                continue
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
            computed_entries[param] = applied
            record.update(
                {
                    "algo": algo,
                    "param": param,
                    "level": rule.level,
                    "size_mode": size_mode,
                    "n": n,
                    "p": p,
                    "p_size": p_size if p_size is not None else p,
                    "leaf_min": leaf_min,
                    "reason": rule.reason,
                    "user_range_policy": cfg.user_range_policy,
                    "applied": record["applied"] and cfg.mode == "apply",
                }
            )
            logs.append(record)
            if cfg.mode == "apply" and record.get("applied"):
                result[param] = applied

    # The leaf-size parameter is processed FIRST: its final range (after the
    # guard rules and ``user_range_policy``) defines ``leaf_min``, which the
    # ``max_depth`` rules use. Resolving it from the final range — not from
    # the raw user range — keeps ``max_depth`` consistent with the resolved
    # ``min_samples_leaf``/``min_child_weight`` (e.g. with engine_wins a user
    # ``min_samples_leaf=[1, 1]`` resolves to ``[2, n/4]`` and ``max_depth``
    # must use the resolved leaf_min=2, not the user's 1).
    leaf_param = _leaf_size_param(result, is_boosting)
    if leaf_param:
        _run_rules({leaf_param}, leaf_min=1.0)
    leaf_entry = computed_entries.get(leaf_param, result.get(leaf_param))
    leaf_min = _effective_leaf_min(leaf_param, leaf_entry, n)
    # Invariant of the two-pass scheme: the second pass runs only with the
    # resolved leaf size (in rows, at least 1), never with a raw user value;
    # the first pass (the leaf-size rules) must not consume ``leaf_min``, so
    # the resolved value is always available before any consumer runs.
    assert leaf_min >= 1.0
    # The leaf-size rules themselves do not use ``leaf_min`` (only the
    # ``max_depth`` rules do); report the resolved value in their records too
    # so that all records of the run carry the same effective leaf size.
    for record in logs:
        if record["param"] == leaf_param:
            record["leaf_min"] = leaf_min
    _run_rules(
        {param for param, _ in rules} - {leaf_param},
        leaf_min=leaf_min,
    )

    return result, logs
