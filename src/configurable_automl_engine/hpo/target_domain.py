"""Target-domain guard for GLM algorithms with a restricted target support.

Some generalized linear models are defined only on a part of the real line:
``GammaRegressor`` requires ``y > 0``, ``PoissonRegressor`` requires
``y >= 0``, and ``TweedieRegressor`` (also exposed as ``glm``) restricts the
target according to its ``power`` parameter (the same intervals as
``sklearn._loss.loss.HalfTweedieLoss``):

* ``power <= 0`` — any real ``y``;
* ``0 < power < 2`` — ``y >= 0``;
* ``power >= 2`` — ``y > 0``.

When such an algorithm meets a target outside its domain, every fold of every
trial fails with "Some value(s) of y are out of the valid range of the loss".
:func:`check_target_domain` detects this BEFORE HPO: it either reports a
disqualification reason, or — for a Tweedie ``power`` range that is only
partly invalid — returns the search space with the range clipped to its valid
part.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from configurable_automl_engine.hpo.default_spaces import (
    CategoricalSpace,
    FloatSpace,
    IntSpace,
    SearchSpaceEntry,
)

# Algorithms with a fixed target domain: name -> (lower bound, inclusive).
_FIXED_DOMAIN_ALGORITHMS: dict[str, tuple[str, bool]] = {
    "gammaregressor": ("y > 0", False),
    "poissonregressor": ("y >= 0", True),
}

# Algorithms backed by TweedieRegressor: the domain depends on ``power``.
_TWEEDIE_ALGORITHMS: frozenset[str] = frozenset({"tweedieregressor", "glm"})

# The default ``power`` of TweedieRegressor (normal distribution): valid for
# any real target, so a search space without ``power`` never conflicts.
_TWEEDIE_DEFAULT_POWER = 0.0

__all__ = ["TargetDomainCheck", "check_target_domain", "tweedie_power_is_valid"]


@dataclass(frozen=True)
class TargetDomainCheck:
    """Outcome of the pre-HPO target-domain check of one algorithm.

    Attributes:
        disqualify_reason (str | None): The reason to exclude the algorithm
            before HPO; ``None`` — the algorithm may run.
        clipped_entries (dict[str, SearchSpaceEntry]): Search space entries
            replaced by their valid part (currently only the Tweedie
            ``power``); empty — nothing was clipped.
        messages (tuple[str, ...]): Human-readable "before -> after" descriptions
            of the clipped entries (for logging).
    """

    disqualify_reason: str | None = None
    clipped_entries: dict[str, SearchSpaceEntry] = field(default_factory=dict)
    messages: tuple[str, ...] = ()


def tweedie_power_is_valid(power: float, y_min: float) -> bool:
    """Check whether a Tweedie ``power`` accepts a target with minimum ``y_min``.

    Args:
        power (float): The Tweedie power.
        y_min (float): The minimum of the target.

    Returns:
        bool: ``True`` if every target value lies in the loss domain.
    """
    if power <= 0:
        return True
    if power < 2:
        return y_min >= 0
    return y_min > 0


def _tweedie_valid_upper_bound(y_min: float) -> tuple[float, bool] | None:
    """Return the upper bound of the valid ``power`` set for ``y_min``.

    Args:
        y_min (float): The minimum of the target.

    Returns:
        tuple[float, bool] | None: ``(bound, inclusive)`` — valid powers are
            ``<= bound`` (inclusive) or ``< bound``; ``None`` — every power is
            valid (strictly positive target).
    """
    if y_min > 0:
        return None
    if y_min == 0:
        return 2.0, False
    return 0.0, True


def _clip_numeric(
    space: FloatSpace | IntSpace, bound: float, inclusive: bool
) -> FloatSpace | IntSpace | None:
    """Clip the upper end of a numeric range to ``bound``.

    Args:
        space (FloatSpace | IntSpace): The original range.
        bound (float): The upper bound of the valid set.
        inclusive (bool): Whether ``bound`` itself is valid.

    Returns:
        FloatSpace | IntSpace | None: The clipped range (the original object
            when nothing has to be clipped); ``None`` — the range has no valid
            values.
    """
    is_int = isinstance(space, IntSpace)
    max_valid: float
    if is_int:
        max_valid = math.floor(bound) if inclusive else math.ceil(bound) - 1
    else:
        max_valid = bound if inclusive else float(np.nextafter(bound, -np.inf))
    if space.high <= max_valid:
        return space
    if space.low > max_valid:
        return None
    step = space.step
    new_high: float = max_valid
    if step is not None:
        # Keep the grid low + k * step: the largest grid point not above the
        # valid bound (low itself is valid here).
        new_high = space.low + math.floor((max_valid - space.low) / step) * step
    if is_int:
        return space.model_copy(update={"high": int(new_high)})
    return space.model_copy(update={"high": float(new_high)})


def _clip_power_entry(entry: SearchSpaceEntry, y_min: float) -> SearchSpaceEntry | None:
    """Restrict a Tweedie ``power`` search entry to the target domain.

    Args:
        entry (SearchSpaceEntry): The ``power`` search entry.
        y_min (float): The minimum of the target.

    Returns:
        SearchSpaceEntry | None: The entry restricted to its valid part (the
            original object when it is fully valid); ``None`` — no value of
            the entry is valid.
    """
    upper = _tweedie_valid_upper_bound(y_min)
    if upper is None:
        return entry
    config = entry.config
    if isinstance(config, CategoricalSpace):
        options = [
            opt
            for opt in config.options
            if not isinstance(opt, (int, float))
            or isinstance(opt, bool)
            or tweedie_power_is_valid(float(opt), y_min)
        ]
        if not options:
            return None
        if len(options) == len(config.options):
            return entry
        return entry.model_copy(
            update={"config": config.model_copy(update={"options": options})}
        )
    clipped = _clip_numeric(config, *upper)
    if clipped is None:
        return None
    if clipped is config:
        return entry
    return entry.model_copy(update={"config": clipped})


def _describe_entry(entry: SearchSpaceEntry) -> str:
    """Return a compact text form of a search entry for log messages.

    Args:
        entry (SearchSpaceEntry): The search entry.

    Returns:
        str: ``[low, high]`` for ranges, the options list for categoricals.
    """
    config = entry.config
    if isinstance(config, CategoricalSpace):
        return str(list(config.options))
    return f"[{config.low}, {config.high}]"


def _target_min(y: Any) -> float | None:
    """Return the minimum of the finite target values.

    Args:
        y (Any): The target vector (array-like).

    Returns:
        float | None: The minimum, or ``None`` when the target has no finite
            numeric values (nothing to check).
    """
    try:
        arr = np.asarray(y, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.min())


def check_target_domain(
    algo_name: str,
    search_space: dict[str, Any],
    y: Any,
) -> TargetDomainCheck:
    """Check that an algorithm's loss domain fits the target before HPO.

    * ``gammaregressor`` needs ``y > 0`` and ``poissonregressor`` needs
      ``y >= 0`` — otherwise the algorithm is disqualified.
    * ``tweedieregressor`` / ``glm``: when ``power`` is in the search space,
      it is clipped to the values valid for the target; when no value is
      valid, the algorithm is disqualified. Without ``power`` in the search
      space the estimator default (``power=0``, any real target) is used, so
      nothing is checked.
    * Every other algorithm passes unchanged.

    Args:
        algo_name (str): The canonical algorithm name.
        search_space (dict[str, Any]): The prepared search space of the
            algorithm (``SearchSpaceEntry`` values or plain fixed values).
        y (Any): The target vector.

    Returns:
        TargetDomainCheck: The disqualification reason or the clipped entries.
    """
    if algo_name not in _FIXED_DOMAIN_ALGORITHMS and algo_name not in (
        _TWEEDIE_ALGORITHMS
    ):
        return TargetDomainCheck()
    y_min = _target_min(y)
    if y_min is None:
        return TargetDomainCheck()

    fixed = _FIXED_DOMAIN_ALGORITHMS.get(algo_name)
    if fixed is not None:
        requirement, inclusive = fixed
        valid = y_min >= 0 if inclusive else y_min > 0
        if valid:
            return TargetDomainCheck()
        return TargetDomainCheck(
            disqualify_reason=(
                f"Target domain mismatch: {algo_name} requires {requirement}, "
                f"but min(y) = {y_min:g}; excluded before HPO"
            )
        )

    power = search_space.get("power", _TWEEDIE_DEFAULT_POWER)
    if not isinstance(power, SearchSpaceEntry):
        # A fixed value (not searched): either valid or not.
        try:
            power_value = float(power)
        except (TypeError, ValueError):
            return TargetDomainCheck()
        if tweedie_power_is_valid(power_value, y_min):
            return TargetDomainCheck()
        return TargetDomainCheck(
            disqualify_reason=(
                f"Target domain mismatch: {algo_name} with power={power_value:g} "
                f"does not support min(y) = {y_min:g}; excluded before HPO"
            )
        )

    clipped = _clip_power_entry(power, y_min)
    if clipped is None:
        return TargetDomainCheck(
            disqualify_reason=(
                f"Target domain mismatch: no value of the {algo_name} power "
                f"search range {_describe_entry(power)} supports "
                f"min(y) = {y_min:g} (valid: power <= 0 for y < 0, "
                f"power < 2 for y >= 0); excluded before HPO"
            )
        )
    if clipped is power:
        return TargetDomainCheck()
    return TargetDomainCheck(
        clipped_entries={"power": clipped},
        messages=(
            (
                f"{algo_name}: power search range clipped "
                f"{_describe_entry(power)} -> {_describe_entry(clipped)} "
                f"to fit min(y) = {y_min:g}"
            ),
        ),
    )
