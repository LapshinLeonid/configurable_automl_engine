"""Runtime support for user-defined metrics and scorers.

The structural declaration (:class:`CustomMetricSpec`) and its kind live in
:mod:`automl_contracts.custom_metrics`. This module imports the callable by its
dotted path, validates its signature and direction, builds scorer objects and
keeps the process-local name registry used by the metric helpers.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import logging
import threading
from collections.abc import Callable
from typing import Any

from automl_contracts.custom_metrics import (
    ERR_CUSTOM_NAME_INVALID,
    CustomMetricKind,
    CustomMetricSpec,
)
from pydantic_core import PydanticCustomError
from sklearn.metrics import make_scorer

from configurable_automl_engine.common.sklearn_compat import (
    scorer_raw_parts,
    scorer_sign,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ERR_CUSTOM_NAME_INVALID",
    "CustomMetricKind",
    "CustomMetricSpec",
    "build_custom_scorer",
    "custom_metric_function",
    "lookup_custom_metric",
    "natural_greater_is_better",
    "register_custom_metric",
    "resolve_dotted_path",
    "validate_custom_metric_spec",
]

ERR_IMPORT = "custom_metric_import_error"
ERR_NOT_CALLABLE = "custom_metric_not_callable"
ERR_BAD_SIGNATURE = "custom_metric_bad_signature"
ERR_DIRECTION_REQUIRED = "custom_metric_direction_required"
ERR_DIRECTION_CONFLICT = "custom_metric_direction_conflict"
ERR_KWARGS_UNSUPPORTED = "custom_metric_kwargs_unsupported"


def resolve_dotted_path(path: str) -> Any:
    """Import the object named by a dotted path.

    Two forms are accepted: ``"pkg.module:attr"`` (the part after ``:`` may
    be a nested attribute chain, e.g. ``"pkg.module:Class.method"``) and
    ``"pkg.module.attr"`` (the last segment is the attribute).

    Args:
        path: The import path of the object.

    Returns:
        Any: The imported object.

    Raises:
        ImportError: If the path is malformed, the module cannot be imported
            (any exception raised while importing it is chained) or the
            attribute does not exist.
    """
    text = path.strip()
    if ":" in text:
        module_name, _, attr_chain = text.partition(":")
    else:
        module_name, _, attr_chain = text.rpartition(".")
    if not module_name or not attr_chain or ":" in attr_chain:
        raise ImportError(
            f"'{path}' is not a valid import path: expected "
            "'package.module:attribute' or 'package.module.attribute'"
        )
    try:
        obj: Any = importlib.import_module(module_name)
    except Exception as err:  # user code may raise anything on import
        raise ImportError(
            f"cannot import module '{module_name}': {type(err).__name__}: {err}"
        ) from err
    for attr in attr_chain.split("."):
        try:
            obj = getattr(obj, attr)
        except AttributeError as err:
            raise ImportError(
                f"module '{module_name}' has no attribute '{attr_chain}'"
            ) from err
    return obj


def _is_sklearn_scorer(obj: Any) -> bool:
    """Whether ``obj`` exposes the raw metric of a ``make_scorer`` object."""
    return scorer_raw_parts(obj)[0] is not None


def _check_positional_arity(fn: Any, n_args: int, kwargs: dict[str, Any]) -> str | None:
    """Return a reason when ``fn`` cannot take ``n_args`` positionals + kwargs.

    ``None`` means the call is compatible or the signature is unavailable
    (C extensions and some builtins expose none; the call is then trusted).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        logger.debug("No signature for %r; the call shape is not checked.", fn)
        return None
    try:
        sig.bind(*range(n_args), **kwargs)
    except TypeError as err:
        return str(err)
    return None


def validate_custom_metric_spec(spec: CustomMetricSpec) -> CustomMetricSpec:
    """Validate the callable referenced by ``spec`` and return the spec.

    The structural contract model deliberately does not import user code.
    The engine performs the import, callability, signature, direction and
    keyword-argument checks at every public entry point.

    Raises:
        PydanticCustomError: A stable ``ERR_*`` code describing the first
            failed check.
    """
    try:
        obj = resolve_dotted_path(spec.path)
    except ImportError as err:
        raise PydanticCustomError(
            ERR_IMPORT,
            "cannot load custom metric '{path}': {reason}",
            {"path": spec.path, "reason": str(err)},
        ) from err
    if not callable(obj):
        raise PydanticCustomError(
            ERR_NOT_CALLABLE,
            "custom metric '{path}' resolves to a non-callable {type}",
            {"path": spec.path, "type": type(obj).__name__},
        )
    if spec.kind == "metric":
        _validate_metric_function(spec, obj)
    else:
        _validate_scorer(spec, obj)
    return spec


def _validate_metric_function(spec: CustomMetricSpec, fn: Callable[..., Any]) -> None:
    """Check a ``kind="metric"`` function: two positionals and a direction.

    Args:
        spec: The contract declaration (path, kwargs and direction).
        fn: The imported callable ``(y_true, y_pred, **kwargs) -> float``.

    Raises:
        PydanticCustomError: ``custom_metric_direction_required`` when
            ``greater_is_better`` is unset; ``custom_metric_bad_signature``
            when the callable cannot accept the configured call shape.
    """
    if spec.greater_is_better is None:
        raise PydanticCustomError(
            ERR_DIRECTION_REQUIRED,
            "custom metric '{path}' (kind='metric') must set "
            "greater_is_better: true for scores, false for errors",
            {"path": spec.path},
        )
    reason = _check_positional_arity(fn, 2, spec.kwargs)
    if reason is not None:
        raise PydanticCustomError(
            ERR_BAD_SIGNATURE,
            "custom metric '{path}' must accept (y_true, y_pred) and the "
            "configured kwargs: {reason}",
            {"path": spec.path, "reason": reason},
        )


def _validate_scorer(spec: CustomMetricSpec, obj: Callable[..., Any]) -> None:
    """Check a ``kind="scorer"`` object.

    The function distinguishes a ``make_scorer`` object (direction is read
    from the scorer and compared with the declaration) from a plain callable
    scorer (three positional arguments, no kwargs, additivity warning).

    Args:
        spec: The contract declaration (path, kwargs, direction, additive).
        obj: The imported scorer object or plain callable.

    Raises:
        PydanticCustomError: ``custom_metric_direction_conflict`` when the
            declared direction disagrees with a ``make_scorer`` scorer;
            ``custom_metric_kwargs_unsupported`` for kwargs on a plain
            callable; ``custom_metric_bad_signature`` when the call shape is
            incompatible.
    """
    if _is_sklearn_scorer(obj):
        scorer_gib = bool((scorer_sign(obj) or 1) > 0)
        if spec.greater_is_better is not None and spec.greater_is_better != scorer_gib:
            raise PydanticCustomError(
                ERR_DIRECTION_CONFLICT,
                "custom scorer '{path}' has greater_is_better={scorer} "
                "but the config sets {config}",
                {
                    "path": spec.path,
                    "scorer": scorer_gib,
                    "config": spec.greater_is_better,
                },
            )
        return
    if spec.kwargs:
        raise PydanticCustomError(
            ERR_KWARGS_UNSUPPORTED,
            "custom scorer '{path}' is a plain callable: kwargs are only "
            "supported for kind='metric' and make_scorer scorers",
            {"path": spec.path},
        )
    reason = _check_positional_arity(obj, 3, {})
    if reason is not None:
        raise PydanticCustomError(
            ERR_BAD_SIGNATURE,
            "custom scorer '{path}' must accept (estimator, X, y): {reason}",
            {"path": spec.path, "reason": reason},
        )
    if not spec.additive:
        logger.warning(
            "Custom scorer '%s' is a plain callable: its value over the "
            "pooled out-of-fold vector (and BBC-CV) cannot be computed, "
            "the per-fold mean is used instead. Set additive: true to "
            "declare this explicitly, or use a make_scorer scorer / "
            "kind: metric.",
            spec.path,
        )


def natural_greater_is_better(spec: CustomMetricSpec) -> bool:
    """The natural direction of the metric value (``True`` — a score)."""
    if spec.greater_is_better is not None:
        return spec.greater_is_better
    obj = resolve_dotted_path(spec.path)
    if _is_sklearn_scorer(obj):
        return bool((scorer_sign(obj) or 1) > 0)
    return True


class _CallableScorer:
    """Adapter giving a plain callable scorer the sklearn scorer direction.

    The engine maximizes the scorer output and reads the direction flag
    (``common.sklearn_compat.scorer_sign``) to recover the natural value; a
    callable whose natural direction is "lower is better" is therefore
    negated here. No raw metric function is exposed: the value over the
    pooled out-of-fold vector cannot be computed for such a scorer.
    """

    def __init__(self, func: Callable[..., Any], sign: int) -> None:
        self._func = func
        self._sign = sign

    def __call__(self, estimator: Any, X: Any, y: Any, **kwargs: Any) -> float:
        return float(self._sign * self._func(estimator, X, y, **kwargs))


_cache_lock = threading.Lock()
_scorer_cache: dict[tuple[str, str, bool | None, bool, str], Any] = {}


def _build_scorer_uncached(spec: CustomMetricSpec) -> Any:
    """Import the callable of ``spec`` and wrap it into a scorer object."""
    obj = resolve_dotted_path(spec.path)
    if spec.kind == "metric":
        return make_scorer(
            obj, greater_is_better=bool(spec.greater_is_better), **spec.kwargs
        )
    if _is_sklearn_scorer(obj):
        if not spec.kwargs:
            return obj
        _, _, obj_kwargs = scorer_raw_parts(obj)
        merged = copy.copy(obj)
        merged._kwargs = {**obj_kwargs, **spec.kwargs}
        return merged
    return _CallableScorer(obj, 1 if natural_greater_is_better(spec) else -1)


def build_custom_scorer(spec: CustomMetricSpec) -> Any:
    """Return the scorer object of a custom metric, cached per process.

    A ``kind="metric"`` function is wrapped with ``make_scorer`` (so the
    pooled out-of-fold and BBC-CV evaluation can call the raw function); a
    ``make_scorer`` scorer is used as is (``kwargs`` merged into a copy); a
    plain callable scorer is wrapped into an adapter exposing the scorer
    direction flag (``common.sklearn_compat.scorer_sign``).

    Args:
        spec: The validated custom metric declaration.

    Returns:
        Any: A callable ``scorer(estimator, X, y) -> float`` (higher is
            better), exposing the direction flag (``_sign``).

    Raises:
        ImportError: If the callable cannot be imported in this process.
    """
    key = spec.cache_key
    with _cache_lock:
        cached = _scorer_cache.get(key)
        if cached is not None:
            return cached
    scorer = _build_scorer_uncached(spec)
    with _cache_lock:
        return _scorer_cache.setdefault(key, scorer)


def custom_metric_function(spec: CustomMetricSpec) -> Callable[..., Any]:
    """Return the raw ``(y_true, y_pred) -> float`` function of a custom metric.

    Raises:
        KeyError: If the spec is a scorer (it has no raw metric function).
    """
    if spec.kind != "metric":
        raise KeyError(f"custom scorer '{spec.path}' has no raw metric function")
    fn = resolve_dotted_path(spec.path)
    if not spec.kwargs:
        return fn  # type: ignore[no-any-return]
    kwargs = dict(spec.kwargs)
    return lambda y_true, y_pred: fn(y_true, y_pred, **kwargs)


_registry_lock = threading.Lock()
_registry: dict[str, CustomMetricSpec] = {}


def register_custom_metric(name: str, spec: CustomMetricSpec) -> None:
    """Register a custom metric under ``name`` in this process.

    Registering the same spec again is a no-op; a different spec replaces
    the registered one with a WARNING (the last declaration wins, so a
    re-run with an edited config uses the edited metric).
    """
    key = name.lower()
    with _registry_lock:
        current = _registry.get(key)
        if current == spec:
            return
        _registry[key] = spec
    if current is not None:
        logger.warning(
            "Custom metric '%s' is redefined: %s -> %s.", key, current.path, spec.path
        )


def lookup_custom_metric(name: str) -> CustomMetricSpec | None:
    """Return the custom metric registered under ``name`` in this process."""
    with _registry_lock:
        return _registry.get(name.lower().strip())
