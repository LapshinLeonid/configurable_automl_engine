"""Search spaces of the HPO runner.

Builds the dict search space of an algorithm (``build_search_space`` — the
single merge of the user ranges over ``DEFAULT_SPACES``) and plans the search
space of a run (``plan_search_space``): the range-type normalization, the
data-dependent adjustments (the SVR adaptive epsilon, the isotonic feature
index), the physical clipping and the adaptive complexity guard. It also
logs the guard records, converts a dict space into Optuna suggestions
(``_apply_dynamic_space``), records the canonical distributions used to
validate the TPE history transfer, and warns when the best value lies at a
range boundary. Every function takes explicit inputs. It may import
``hpo.types``, ``hpo.plans`` and ``hpo.trial_pipeline``; it must not import
``hpo.runner`` or ``configurable_automl_engine.tuner``.
"""

from __future__ import annotations

import logging as _logging
import math
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import optuna
from optuna.trial import Trial

from configurable_automl_engine.hpo.complexity_guard import (
    ComplexityGuardConfig,
    apply_complexity_guard,
)
from configurable_automl_engine.hpo.default_spaces import (
    DEFAULT_SPACES,
    CategoricalSpace,
    FloatSpace,
    SearchSpaceEntry,
    clip_search_space,
    compute_svr_epsilon_bounds,
    normalize_hyperparameter_entry,
)
from configurable_automl_engine.models import (
    remap_legacy_param_names,
    resolve_algorithm_name,
)

from .plans import PreprocessingPlan, SearchSpacePlan
from .trial_pipeline import (
    _resolve_apply_fs,
)
from .types import (
    HyperoptError,
)

if TYPE_CHECKING:
    from configurable_automl_engine.feature_selection import (
        FeatureSelectionMode,
    )

log = _logging.getLogger(__name__)


# ═══════════════════════════════════ search spaces ═══════════════════════════


# ══════════ SVR space: epsilon depends on the spread of y ══════════
def _svr_adaptive_epsilon_entry(y: Any) -> SearchSpaceEntry:
    """Build a SearchSpaceEntry for SVR epsilon with bounds scaled by y.

    The bounds are computed from the spread of ``y`` (σ(y) by default) ONCE —
    before the Optuna loop (called from ``plan_search_space``, not from
    ``space_fn``): otherwise every trial would have its own bounds and the
    TPE sampler would lose determinism.

    Args:
        y (Any): Target variable vector.
    Returns:
        SearchSpaceEntry: float_log distribution of epsilon within adaptive bounds.
    """
    low, high = compute_svr_epsilon_bounds(y)
    log.info("SVR adaptive epsilon bounds: [%.6g, %.6g] (from y spread)", low, high)
    return SearchSpaceEntry.model_validate([low, high, "float_log"])


def build_search_space(
    algo_name: str,
    user_overrides: dict[str, Any] | None,
    limit: bool = False,
) -> dict[str, Any]:
    """Build the dict search space of an algorithm from the user ranges.

    The single merge rule of every HPO run (a direct ``optimize()`` call and
    the training engine, which passes the user ranges of the config), so a
    direct call and a config run with the same ranges search the same space.

    When ``limit`` is True the search is restricted to ONLY the parameters
    listed in ``user_overrides``: no ``DEFAULT_SPACES`` entries are added,
    and every other parameter stays at the estimator default. When ``limit``
    is False (the default) the user ranges override and are merged into
    ``DEFAULT_SPACES``.

    The SVR ``epsilon`` default (a blind ``[1e-3, 1.0]`` range) is dropped
    unless the user set ``epsilon``: its absence means "use the adaptive
    range" (bounds scaled by the spread of ``y``), which
    ``plan_search_space`` injects once the data is known.

    Legacy hyperparameter names (e.g. ``n_iter`` → ``max_iter`` for
    ARDRegression) are remapped ONCE here, before the Optuna loop — not at
    every model instantiation (which would log an identical "Remapped ..."
    line for ~100 trials).

    The merge is idempotent: merging an already-merged space again changes
    nothing (all ``DEFAULT_SPACES`` keys are already present, and a missing
    SVR ``epsilon`` again means "adaptive").

    Args:
        algo_name (str): The algorithm name used to look up defaults.
        user_overrides (dict[str, Any] | None): The user ranges
            (``SearchSpaceEntry`` objects, raw range lists or constants).
        limit (bool): Restrict the search to the user-listed parameters only
            (no ``DEFAULT_SPACES`` entries). Defaults to False.

    Returns:
        dict[str, Any]: The resulting search space (a new dict; the inputs
            are not mutated). Empty when ``limit`` is True and there are no
            user ranges, or when the algorithm has no default space and no
            user ranges.
    """
    if limit:
        if not user_overrides:
            return {}
        # Only the user-listed parameters are searched; everything else
        # stays at the estimator default. Legacy names are remapped once.
        return remap_legacy_param_names(algo_name, user_overrides)

    resolved = resolve_algorithm_name(algo_name)
    space = DEFAULT_SPACES.get(resolved, {}).copy()
    if user_overrides:
        # Remap legacy names in the user overrides BEFORE the merge so that
        # an explicit user value (e.g. ARDRegression n_iter → max_iter) wins
        # over the system default of the modern name.
        space.update(remap_legacy_param_names(algo_name, user_overrides))

    # Adaptive epsilon for SVR: the epsilon search bounds are scaled by the
    # y_train spread at runtime (plan_search_space), so the default blind
    # range [1e-3, 1.0] from DEFAULT_SPACES is dropped. If the user set
    # epsilon explicitly — it is already in space (space.update above), and
    # the user's priority is preserved: the adaptive logic does not
    # interfere.
    if resolved == "svr" and not (user_overrides and "epsilon" in user_overrides):
        space.pop("epsilon", None)

    # Remap legacy names once (e.g. ARDRegression n_iter → max_iter)
    # so that per-trial clean_hyperparameters has nothing to remap/log.
    return remap_legacy_param_names(algo_name, space)


# ═════════════════════════════ helper utilities ═════════════════════════════


def _apply_dynamic_space(trial: Trial, space_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a configuration dict into model parameters via Optuna methods.

    Args:
        trial (Trial): The current Optuna trial object.
        space_dict (dict[str, Any]): A dict of SearchSpaceEntry objects
            (with bounds and distribution types) or constant values.
    Returns:
        dict[str, Any]: The concrete hyperparameter values for this iteration.
    """
    params: dict[str, Any] = {}
    for key, value in space_dict.items():
        # If this is a SearchSpaceEntry
        # (use the low, high, dist_type, step properties)
        if hasattr(value, "dist_type"):
            low, high = value.low, value.high
            dist_type = value.dist_type
            step = value.step
            if dist_type == "int":
                low_val, high_val = int(cast(float, low)), int(cast(float, high))
                params[key] = trial.suggest_int(
                    key, low_val, high_val, step=int(step) if step is not None else 1
                )
            elif dist_type == "float":
                params[key] = trial.suggest_float(
                    key,
                    float(low),
                    float(high),
                    step=float(step) if step is not None else None,
                )
            elif dist_type == "float_log":
                FloatSpace.validate_log_low(float(low))
                params[key] = trial.suggest_float(
                    key, float(low), float(high), log=True
                )
            elif dist_type == "categorical":
                # Extract the options list from the options attribute or nested config
                options = getattr(value, "options", None)
                if options is None and hasattr(value, "config"):
                    options = getattr(value.config, "options", None)

                # If still None, fall back to low (for compatibility)
                if options is None:
                    options = low

                params[key] = trial.suggest_categorical(key, options)
        else:
            # If it is a plain value (constant), use it as is
            params[key] = value
    return params


class _DistributionRecorder:
    """Record the canonical distributions of the phase search space.

    A lightweight stand-in for an Optuna ``Trial`` that only collects the
    distributions of every ``suggest_*`` call (``suggest_int`` /
    ``suggest_float`` / ``suggest_categorical``). Running ``space_fn`` on the
    recorder yields the distributions the current phase would produce — they
    are compared against the distributions of ``previous_trials`` BEFORE
    ``study.add_trials`` so that an incompatible history fails fast with a
    clear message instead of a silent skip (or a confusing error during the
    first trial of the new phase).
    """

    def __init__(self) -> None:
        """Initialize an empty distribution recorder."""
        self.distributions: dict[str, Any] = {}
        # Trial-compatible attributes that user space functions may read
        # (e.g. ``trial.number`` to alternate between good/bad points).
        self.number = -1

    def suggest_int(self, name: str, low: int, high: int, step: int = 1) -> int:
        """Record an ``IntDistribution`` and return ``low`` (a valid value)."""
        self.distributions[name] = optuna.distributions.IntDistribution(
            low=low, high=high, step=step
        )
        return low

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float:
        """Record a ``FloatDistribution`` and return ``low`` (a valid value)."""
        self.distributions[name] = optuna.distributions.FloatDistribution(
            low=low, high=high, step=step, log=log
        )
        return low

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        """Record a ``CategoricalDistribution`` and return the first choice."""
        self.distributions[name] = optuna.distributions.CategoricalDistribution(
            choices=list(choices)
        )
        return choices[0]


def _enqueueable_initial_params(
    params: dict[str, Any], space_dict: dict[str, Any], algo: str
) -> dict[str, Any]:
    """Drop initial params whose value type does not fit the search space.

    A value carried over from an older space (e.g. SVR ``gamma="scale"``
    from the former categorical default, now a numeric ``float_log``
    range) would make Optuna fail the enqueued trial. Such a key is
    dropped with a warning and sampled from the space instead; numeric
    values outside the range are kept as is (Optuna handles them).

    Args:
        params: The initial params from the previous phase.
        space_dict: The resolved dict search space of the run.
        algo: The algorithm name (for the message).

    Returns:
        dict[str, Any]: The params that can be enqueued.
    """
    kept: dict[str, Any] = {}
    for name, value in params.items():
        entry = space_dict.get(name)
        if isinstance(entry, SearchSpaceEntry):
            config = entry.config
            if isinstance(config, CategoricalSpace):
                compatible = value in config.options
            else:
                compatible = isinstance(
                    value, (int, float, np.integer, np.floating)
                ) and not isinstance(value, bool)
            if not compatible:
                log.warning(
                    "Initial param %s=%r does not fit the search space of "
                    "'%s' (%s); it is sampled instead",
                    name,
                    value,
                    algo,
                    entry.dist_type,
                )
                continue
        kept[name] = value
    return kept


def _canonical_distributions(
    space_fn: Callable[[Trial], dict[str, Any]],
    *,
    fs_mode: FeatureSelectionMode,
    is_isotonic: bool,
    algo: str,
) -> dict[str, Any] | None:
    """Compute the canonical distributions of the current phase space.

    Runs ``space_fn`` (plus the feature selection suggestion) on a
    ``_DistributionRecorder`` — the distributions the phase WOULD produce
    for its trials. If the space function cannot be reproduced (a custom
    exotic ``space_fn``, e.g. one using legacy ``suggest_*`` methods the
    recorder does not implement), ``None`` is returned — the caller
    (``_transfer_previous_trials``) fails fast with ``HPOHistoryError``:
    the history transfer cannot be validated, and silently dropping it
    would degrade refine to a cold run.

    Args:
        space_fn: The search space function of the run.
        fs_mode: The feature selection mode.
        is_isotonic: Whether the algorithm is ``isotonic_regression``.
        algo: The algorithm name (for the message).

    Returns:
        dict[str, Any]: Parameter name → Optuna distribution; ``None`` —
            the space cannot be reproduced (the caller fails fast).
    """
    try:
        recorder = _DistributionRecorder()
        # The recorder is a Trial-compatible stand-in: it implements the
        # suggest_* methods (and the ``number`` attribute) that space_fn
        # and _resolve_apply_fs call.
        trial_like = cast(Trial, recorder)
        space_fn(trial_like)
        _resolve_apply_fs(trial_like, fs_mode, is_isotonic)
    except Exception as err:  # noqa: BLE001 — the pre-check is best-effort
        log.warning(
            "Cannot reproduce the search space for the history transfer "
            "check of algorithm '%s': %s. Skipping the distribution "
            "pre-check.",
            algo,
            err,
        )
        return None
    return recorder.distributions


def _resolve_complexity_guard(
    space: dict[str, Any],
    *,
    algo: str,
    config: ComplexityGuardConfig,
    user_keys: list[str] | None,
    n_samples_eff: int,
    n_features_eff: int,
    n_features_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the adaptive complexity guard to a dict search space.

    The guard depends on ``n`` (the effective training-fold rows,
    ``n_samples_eff``) and the feature counts: the size-mode ratio uses
    ``n_features_size`` — the number of ORIGINAL columns (a categorical
    column counts as 1, before one-hot), while the rules for
    ``max_features``/``colsample_bytree`` use ``n_features_eff`` — the
    number of features after preprocessing. Both are computed once per
    phase and are the same in every HPO phase (deterministic between
    phases) and in the subprocesses (identical inputs -> identical
    results), so the TPE history transfer between phases stays valid.

    When the guard is disabled (``mode='off'``) the space is returned
    unchanged and no records are produced.

    Args:
        space: The already physically-clipped search space.
        algo: The algorithm name.
        config: The complexity guard settings.
        user_keys: The parameters treated as user ranges.
        n_samples_eff: The number of rows a model sees in a training fold.
        n_features_eff: The number of features after preprocessing.
        n_features_size: The number of original columns.

    Returns:
        tuple[dict[str, Any], list[dict[str, Any]]]: The (possibly
            adjusted) space and the guard log records.
    """
    if config.mode == "off":
        return space, []
    adjusted, logs = apply_complexity_guard(
        space,
        algo,
        n=n_samples_eff,
        p=n_features_eff,
        config=config,
        user_space_keys=set(user_keys or ()),
        p_size=n_features_size,
    )
    return adjusted, logs


def log_complexity_guard(
    records: list[dict[str, Any]], config: ComplexityGuardConfig
) -> None:
    """Log the adaptive complexity guard records at INFO level.

    One record per (algorithm, phase, parameter): "was → became" with
    the reason, e.g. ``max_depth 3–20 → 2–3: n=15, p=5, mode=small,
    leaf_min=2``.

    Args:
        records: The guard records of the run (``SearchSpacePlan``).
        config: The complexity guard settings.
    """
    cfg_mode = config.mode
    cfg_policy = config.user_range_policy
    for record in records:
        before = record["before"]
        after = record["after"]

        def fmt(v: Any) -> str:
            if v is None:
                return "n/a"
            return str(v) if isinstance(v, int) else f"{v:g}"

        if record.get("applied"):
            message = (
                f"[complexity_guard] {record['algo']} {record['param']} "
                f"{fmt(before[0])}–{fmt(before[1])} → "
                f"{fmt(after[0])}–{fmt(after[1])}: n={record['n']}, "
                f"p={record['p']}, p_size={fmt(record['p_size'])}, "
                f"mode={record['size_mode']}, "
                f"leaf_min={fmt(record['leaf_min'])}, level={record['level']}"
            )
            if record.get("conflict"):
                message += " (user range conflict, engine wins)"
            if record.get("fixed"):
                message += " (parameter fixed)"
            log.info("%s: %s", message, record["reason"])
        elif record.get("conflict"):
            rule_range = record.get("rule_range") or [None, None]
            if cfg_mode == "warn":
                # In the warn mode nothing is applied at all — report the
                # conflict without claiming that the user range wins.
                log.warning(
                    "[complexity_guard] %s %s user range %s–%s conflicts "
                    "with the guard range %s–%s; computed but not applied "
                    "(mode='warn', user_range_policy='%s')",
                    record["algo"],
                    record["param"],
                    fmt(before[0]),
                    fmt(before[1]),
                    fmt(rule_range[0]),
                    fmt(rule_range[1]),
                    cfg_policy,
                )
            else:
                log.warning(
                    "[complexity_guard] %s %s user range %s–%s conflicts "
                    "with the guard range %s–%s; keeping the user range "
                    "(user_range_policy='%s')",
                    record["algo"],
                    record["param"],
                    fmt(before[0]),
                    fmt(before[1]),
                    fmt(rule_range[0]),
                    fmt(rule_range[1]),
                    cfg_policy,
                )
        else:
            # Computed but not applied (e.g. mode='warn'): report the
            # would-be change without claiming a conflict.
            log.info(
                "[complexity_guard] %s %s %s–%s → %s–%s: computed but "
                "not applied (mode=%s): %s",
                record["algo"],
                record["param"],
                fmt(before[0]),
                fmt(before[1]),
                fmt(after[0]),
                fmt(after[1]),
                cfg_mode,
                record["reason"],
            )


def _normalize_space_dict(algo: str, space_dict: dict[str, Any]) -> dict[str, Any]:
    """Re-resolve inferred range types in a dict search space.

    The same normalization the Config path applies (see
    ``Config._normalize_hyperparameter_types``): a range whose type was
    inferred from the boundary numbers (``[1, 50]`` in the list shorthand,
    two-element lists passed directly to ``space_overrides``) is re-typed
    against the parameter type (the default search space or the estimator
    signature). So ``space_overrides={"svr": {"C": [1, 50]}}`` searches
    ``C`` as a real range (``float_log``), not as an integer-only range.
    A parameter whose default space entry is
    ``float_log`` (e.g. ``alpha_1`` of an ARD regression) is likewise
    searched on a logarithmic scale without an explicit type. Raw lists
    (not yet parsed into ``SearchSpaceEntry`` objects) are parsed first;
    explicitly typed entries and constants pass through unchanged.

    Args:
        algo: The algorithm name.
        space_dict: The parameter name → value mapping (``SearchSpaceEntry``
            objects, raw lists, or constants).

    Returns:
        dict[str, Any]: A new dict with the same structure: every numeric
            key holds a ``SearchSpaceEntry`` (raw lists are parsed),
            inferred range types re-resolved against the parameter type.
    """
    normalized: dict[str, Any] = {}
    for param, value in space_dict.items():
        if isinstance(value, SearchSpaceEntry):
            entry = value
        elif isinstance(value, list):
            # A raw list from a direct call: parse it the same way the
            # Config path does. Constants (scalars) are never lists, so a
            # list here is always a search-space shorthand.
            entry = SearchSpaceEntry.model_validate(value)
        else:
            # A constant value — no normalization possible.
            normalized[param] = value
            continue
        normalized[param] = normalize_hyperparameter_entry(algo, param, entry)
    return normalized


def plan_search_space(
    algo: str,
    y: Any,
    *,
    space_override: Any,
    limit_hyperparameters: bool,
    user_hyperparameter_keys: list[str] | None,
    complexity_guard: ComplexityGuardConfig,
    n_samples_eff: int,
    preprocessing: PreprocessingPlan,
) -> SearchSpacePlan:
    """Plan the hyperparameter search space of a run.

    A callable override (a custom ``space_fn``) is used as is. Otherwise the
    dict search space is built by ``build_search_space`` from the dict
    override (or from no user ranges at all), so a direct ``optimize()``
    call and a config run with the same ranges search the same space. The
    data-dependent adjustments then apply to every dict space: the adaptive
    SVR epsilon (bounds scaled by the spread of ``y_train``), the isotonic
    ``feature_index`` over the numerical features, the physical clipping
    (e.g. KNN ``n_neighbors`` ≤ ``n_samples_eff - 1``) and the adaptive
    complexity guard. The guard records are returned in the plan (the
    caller logs them with ``log_complexity_guard``).

    When ``user_hyperparameter_keys`` is ``None``, the guard treats the keys
    of the dict override as the user ranges.

    Args:
        algo: The lower-cased algorithm name.
        y: The target vector (the SVR epsilon bounds).
        space_override: The override of the algorithm (a dict of ranges, a
            callable space function or ``None``).
        limit_hyperparameters: Search only the parameters of the override.
        user_hyperparameter_keys: The parameters the user set explicitly.
        complexity_guard: The complexity guard settings.
        n_samples_eff: The number of rows a model sees in a training fold.
        preprocessing: The preprocessing plan (the feature counts).

    Returns:
        SearchSpacePlan: The space function, the resolved dict space, the
            guard records and the user keys.

    Raises:
        HyperoptError: If the algorithm has neither a default search space
            nor a ``space_overrides`` entry.
    """
    # Priority 1: Direct overrides (functions) are used as is.
    if callable(space_override):
        return SearchSpacePlan(
            space_fn=space_override,
            space_dict={},
            guard_records=[],
            user_hyperparameter_keys=user_hyperparameter_keys,
        )

    # Priority 2: Dict ranges (SearchSpaceEntry objects or shorthand lists)
    # merged over DEFAULT_SPACES; no override — DEFAULT_SPACES only.
    user_ranges = space_override if isinstance(space_override, dict) else None
    # The limited mode searches exactly the listed parameters; without a
    # dict override there is nothing listed, so the defaults are searched.
    limit = bool(limit_hyperparameters) and user_ranges is not None
    # A direct call without the list: the user ranges are the keys the
    # caller listed.
    user_keys = (
        list(user_ranges or {})
        if user_hyperparameter_keys is None
        else list(user_hyperparameter_keys)
    )

    merged = build_search_space(algo, user_ranges, limit=limit)
    if user_ranges is None and not merged:
        raise HyperoptError(
            f"No search space for algorithm '{algo}': it has no "
            "default search space; pass "
            f"space_overrides={{'{algo}': {{...}}}} with the ranges "
            "to search (a dict of ranges or a callable space function)"
        )

    space_dict = _normalize_space_dict(algo, merged)
    # SVR: adaptive epsilon bounds from the y_train spread, if the
    # user did not set epsilon explicitly (build_search_space omits
    # the default epsilon from DEFAULT_SPACES, so its absence in the
    # dict means "not overridden"). The bounds are computed once
    # BEFORE the Optuna loop. In the limited mode
    # (``limit_hyperparameters: true``) the space is exactly what the
    # user listed: no parameter is added beyond it, so the adaptive
    # epsilon is NOT injected.
    if (
        resolve_algorithm_name(algo) == "svr"
        and "epsilon" not in space_dict
        and not limit
    ):
        space_dict = {**space_dict, "epsilon": _svr_adaptive_epsilon_entry(y)}
    # Isotonic: DEFAULT_SPACES holds only ``increasing``, and without an
    # explicit ``feature_index`` the model would silently use the default
    # index 0 (an arbitrary one-hot dummy column). Merge a data-driven
    # ``feature_index`` over the NUMERICAL features into the dict when
    # absent; an explicit user value is kept as is.
    if (
        resolve_algorithm_name(algo) == "isotonic_regression"
        and "feature_index" not in space_dict
    ):
        space_dict = {
            **space_dict,
            "feature_index": SearchSpaceEntry.model_validate(
                [0, preprocessing.isotonic_max_feature_index, "int"]
            ),
        }
    clipped = clip_search_space(space_dict, n_samples_eff)
    clipped, guard_records = _resolve_complexity_guard(
        clipped,
        algo=algo,
        config=complexity_guard,
        user_keys=user_keys,
        n_samples_eff=n_samples_eff,
        n_features_eff=preprocessing.n_features_eff,
        n_features_size=preprocessing.n_features_size,
    )
    return SearchSpacePlan(
        space_fn=partial(_apply_dynamic_space, space_dict=clipped),
        space_dict=clipped,
        guard_records=guard_records,
        user_hyperparameter_keys=user_keys,
    )


def _natural_domain_bounds(
    param: str,
    entry: SearchSpaceEntry,
    *,
    n_samples_eff: int,
    n_features_eff: int,
    isotonic_max_feature_index: int,
) -> tuple[float | None, float | None]:
    """Natural (physical) domain bounds of a parameter.

    Returns ``(low, high)``; a side is ``None`` when the domain is
    unbounded there. A boundary that coincides with a natural bound is not
    actionable: the value cannot be pushed beyond it even by widening the
    search range, so the "widen it" hint would be a false signal. Bounds
    that depend on the data (leaf sizes, neighbor counts) are derived from
    ``n_samples_eff`` / ``n_features_eff`` — these are the same physical
    limits that ``clip_search_space`` enforces.

    Args:
        param: The hyperparameter name.
        entry: The search-space entry (its distribution type decides the
            fraction-vs-count interpretation of ``min_samples_leaf`` and
            ``max_features``).
        n_samples_eff: The number of rows a model sees in a training fold.
        n_features_eff: The number of features after preprocessing.
        isotonic_max_feature_index: The upper bound of the isotonic
            ``feature_index`` range.

    Returns:
        tuple[float | None, float | None]: Natural ``(low, high)`` bounds.
    """
    low: float | None = None
    high: float | None = None
    dist_type = entry.dist_type
    if param in {"subsample", "colsample_bytree"}:
        # A row/feature sampling fraction: (0, 1].
        high = 1.0
    elif param == "min_samples_leaf":
        if dist_type == "float" and float(entry.low) < 1.0 and float(entry.high) <= 1.0:
            # A fraction of the sample size: (0, 1].
            high = 1.0
        else:
            # An absolute row count: at least 1 and at most n.
            low = 1.0
            high = float(max(1, int(n_samples_eff)))
    elif param == "min_samples_split":
        low = 2.0
        high = float(max(2, int(n_samples_eff)))
    elif param == "n_neighbors":
        low = 1.0
        high = float(max(1, int(n_samples_eff) - 1))
    elif param == "max_features":
        if dist_type == "float":
            # A fraction of the features: (0, 1].
            high = 1.0
        else:
            # An absolute feature count.
            low = 1.0
            high = float(max(1, int(n_features_eff)))
    elif param == "l1_ratio":
        low = 0.0
        high = 1.0
    elif param == "max_depth":
        low = 1.0
    elif param in {"min_child_weight", "gamma"}:
        low = 0.0
    elif param == "n_estimators":
        low = 1.0
    elif param == "p":
        # The Minkowski power of the KNN space.
        low = 1.0
    elif param == "feature_index":
        # The ordinal NUMERICAL column index consumed by
        # IsotonicDataTransformer: the natural domain spans
        # [0, n_numeric - 1] — only numeric features are meaningful
        # inputs for the strictly univariate model (one-hot dummies are
        # excluded), and neither side can be pushed beyond it even by
        # widening the search range.
        low = 0.0
        high = float(max(0, isotonic_max_feature_index))
    return low, high


def _is_guard_boundary(
    algo: str,
    guard_records: list[dict[str, Any]],
    param: str,
    value: float,
    tolerance: float,
) -> bool:
    """Whether ``value`` lies on a bound introduced by the complexity guard.

    The guard records the "before → after" bounds of every range it
    actually narrowed. A bound is guard-introduced when the guard moved
    that side of the range: the "after" bound differs from "before" (the
    other side may be the user's or the default bound and stays a normal
    "widen it" candidate). An optimum on a guard-moved bound is expected —
    the range was tightened deliberately (e.g. ``max_depth`` 2–3 on small
    data) — so the "widen it" warning would be a false signal.

    Args:
        algo: The algorithm name.
        guard_records: The complexity guard records of the run.
        param: The hyperparameter name.
        value: The boundary value of the best trial.
        tolerance: The same absolute tolerance the boundary check uses.

    Returns:
        bool: ``True`` when the value is a guard-introduced bound.
    """
    algo = resolve_algorithm_name(algo)
    for record in guard_records:
        if record.get("param") != param:
            continue
        if resolve_algorithm_name(str(record.get("algo", ""))) != algo:
            continue
        if not record.get("applied"):
            continue
        before = record.get("before")
        after = record.get("after")
        if not isinstance(before, list) or len(before) != 2:
            continue
        if not isinstance(after, list) or len(after) != 2:
            continue
        try:
            before = [float(before[0]), float(before[1])]
            after = [float(after[0]), float(after[1])]
        except (TypeError, ValueError):
            continue
        # A side introduced by the guard: the after bound differs from
        # the before one on that side.
        moved_low = not math.isclose(
            after[0], before[0], rel_tol=0.0, abs_tol=tolerance
        )
        moved_high = not math.isclose(
            after[1], before[1], rel_tol=0.0, abs_tol=tolerance
        )
        if (
            moved_low and math.isclose(value, after[0], rel_tol=0.0, abs_tol=tolerance)
        ) or (
            moved_high and math.isclose(value, after[1], rel_tol=0.0, abs_tol=tolerance)
        ):
            return True
    return False


def _warn_if_optimum_at_boundary(
    best_params: dict[str, Any],
    search: SearchSpacePlan,
    *,
    algo: str,
    n_samples_eff: int,
    preprocessing: PreprocessingPlan,
) -> None:
    """Warn when the best trial's value lies at a range boundary.

    An optimum that coincides with the lower or upper bound of the search
    range means the search was cut off: the true optimum may lie beyond
    the range, and the user should widen it. Only dict-based numeric
    search spaces (with ``SearchSpaceEntry`` bounds) are checked;
    categorical parameters and constant values are skipped.

    The warning is suppressed when the boundary is not actionable:
    * a fixed parameter (``low == high``) — e.g. a user
      ``min_samples_leaf [1, 1]`` or an XGBoost ``max_depth`` collapsed
      to 3–3 — the optimum sits at the only possible value;
    * a bound introduced by the adaptive complexity guard — the range
      was narrowed deliberately (e.g. ``max_depth`` 2–3) and the optimum
      is expected on it (an INFO line is logged instead);
    * a natural (physical) domain boundary of the parameter —
      ``subsample=1.0``, ``colsample_bytree=1.0``, ``min_samples_leaf=1``,
      ``l1_ratio=0/1`` — the value cannot be pushed beyond it even by
      widening the range.

    Args:
        best_params: The parameters of the best trial.
        search: The search space plan of the run.
        algo: The algorithm name.
        n_samples_eff: The number of rows a model sees in a training fold.
        preprocessing: The preprocessing plan (the feature counts).
    """
    if not search.space_dict:
        return
    for name, entry in search.space_dict.items():
        if not hasattr(entry, "dist_type"):
            continue
        dist_type = entry.dist_type
        if dist_type not in ("int", "float", "float_log"):
            continue
        if name not in best_params:
            continue
        value = best_params[name]
        try:
            low = float(entry.low)
            high = float(entry.high)
            num = float(value)
        except (TypeError, ValueError):
            continue
        if low == high:
            # A fixed parameter: the optimum is at the only possible
            # value, the "widen it" hint is meaningless.
            continue
        # Tolerance: 1e-6 of the range width (at least 1e-6 for tiny ranges).
        # Exact equality is not enough because Optuna may return 1.0 vs
        # a boundary 1.0000001. A relative tolerance is deliberately not
        # used: on very wide ranges (e.g. [0, 1e9]) a rel_tol of 1e-9
        # would already accept values ~1.0 away from the boundary — too
        # coarse for a "widen the range" hint.
        tolerance = 1e-6 * max(1.0, abs(high - low))
        at_low = math.isclose(num, low, rel_tol=0.0, abs_tol=tolerance)
        at_high = math.isclose(num, high, rel_tol=0.0, abs_tol=tolerance)
        if not (at_low or at_high):
            continue
        boundary = "low" if at_low else "high"
        bound_value = low if at_low else high

        # A guard-introduced bound: the range was narrowed deliberately,
        # the optimum on it is expected — INFO instead of a warning.
        if _is_guard_boundary(algo, search.guard_records, name, bound_value, tolerance):
            log.info(
                "Algorithm '%s': the optimum of '%s'=%.6g lies at the %s "
                "boundary of [%.6g, %.6g], set by the complexity_guard — "
                "expected, the range is narrowed deliberately, no 'widen "
                "it' warning",
                algo,
                name,
                num,
                boundary,
                low,
                high,
            )
            continue

        # A natural (physical) domain boundary of the parameter: the value
        # cannot be pushed beyond it even by widening the range.
        natural_low, natural_high = _natural_domain_bounds(
            name,
            entry,
            n_samples_eff=n_samples_eff,
            n_features_eff=preprocessing.n_features_eff,
            isotonic_max_feature_index=preprocessing.isotonic_max_feature_index,
        )
        natural_bound = natural_low if at_low else natural_high
        if natural_bound is not None and math.isclose(
            bound_value, natural_bound, rel_tol=0.0, abs_tol=tolerance
        ):
            log.info(
                "Algorithm '%s': the optimum of '%s'=%.6g lies at the %s "
                "boundary of [%.6g, %.6g] — the natural domain boundary "
                "of the parameter, no 'widen it' warning",
                algo,
                name,
                num,
                boundary,
                low,
                high,
            )
            continue

        log.warning(
            "Algorithm '%s': the optimum of '%s'=%.6g lies at the %s "
            "boundary of the search range [%.6g, %.6g] — the true "
            "optimum may lie beyond the range, widen it",
            algo,
            name,
            num,
            boundary,
            low,
            high,
        )
