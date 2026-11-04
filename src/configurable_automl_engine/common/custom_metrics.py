"""User-defined metrics and scorers loaded by a dotted import path.

``CustomMetricSpec`` is the declaration of one custom metric (the
``general.custom_metrics.<name>`` config entry): the import path of the
callable, its calling convention (``kind``), its natural direction and its
additivity. Validating a spec imports the callable and checks its signature,
so a broken declaration fails before training starts.

A spec holds data only: it is pickled into the worker processes and into the
saved artifact without the callable, and every process imports the callable
again from ``path`` on first use (``build_custom_scorer``, cached per
process). ``register_custom_metric`` / ``lookup_custom_metric`` keep the
process-local name → spec registry that ``common.metrics`` consults before
the built-in metrics; the engine entry points register the specs they
receive, so the registry is rebuilt in every worker process.

The module must not import any other module of the package.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import logging
import threading
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError
from sklearn.metrics import make_scorer

logger = logging.getLogger(__name__)

__all__ = [
    "CustomMetricKind",
    "CustomMetricSpec",
    "build_custom_scorer",
    "custom_metric_function",
    "lookup_custom_metric",
    "register_custom_metric",
    "resolve_dotted_path",
]

#: ``"metric"`` — a function ``(y_true, y_pred, **kwargs) -> float``;
#: ``"scorer"`` — an sklearn scorer ``(estimator, X, y) -> float``.
CustomMetricKind = Literal["metric", "scorer"]

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
    """Whether ``obj`` is a ``make_scorer`` object (it exposes the raw metric)."""
    return all(hasattr(obj, a) for a in ("_score_func", "_sign", "_kwargs"))


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


class CustomMetricSpec(BaseModel):
    """The declaration of one custom metric or scorer.

    Validation imports the callable named by ``path`` and checks that it is
    callable, that its signature accepts the calling convention of ``kind``
    (plus ``kwargs``) and that the direction is known.

    Attributes:
        path: Dotted import path of the callable (``"pkg.mod:attr"`` or
            ``"pkg.mod.attr"``).
        kind: ``"metric"`` — ``fn(y_true, y_pred, **kwargs) -> float``
            returning the natural metric value; ``"scorer"`` — an sklearn
            scorer ``scorer(estimator, X, y) -> float``.
        greater_is_better: The natural direction of the returned value.
            Required for ``kind="metric"``. For a ``make_scorer`` scorer the
            direction is taken from the scorer (an explicit value must agree
            with it); a plain callable scorer defaults to ``True``.
        additive: ``True`` — the CV estimate is the mean of the per-fold
            values (valid for per-sample means such as MAE); ``False`` — the
            value over the pooled out-of-fold vector (always correct).
        kwargs: Static keyword arguments of the metric function (for a
            ``make_scorer`` scorer they are merged into its kwargs).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(
        ...,
        min_length=1,
        description=(
            "Dotted import path of the callable: 'package.module:attribute' "
            "(preferred) or 'package.module.attribute'."
        ),
    )
    kind: CustomMetricKind = Field(
        default="metric",
        description=(
            "'metric' — fn(y_true, y_pred, **kwargs) -> float; "
            "'scorer' — sklearn scorer(estimator, X, y) -> float."
        ),
    )
    greater_is_better: bool | None = Field(
        default=None,
        description=(
            "Natural direction of the returned value. Required for "
            "kind='metric'; for a make_scorer scorer it is taken from the "
            "scorer; a plain callable scorer defaults to true."
        ),
    )
    additive: bool = Field(
        default=False,
        description=(
            "true — the CV estimate is the mean of the per-fold values (only "
            "valid for per-sample means such as MAE); false (default) — the "
            "value over the pooled out-of-fold vector."
        ),
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Static keyword arguments passed to the metric function.",
    )

    @model_validator(mode="after")
    def _validate_callable(self) -> CustomMetricSpec:
        """Import the callable and check its shape and direction."""
        try:
            obj = resolve_dotted_path(self.path)
        except ImportError as err:
            raise PydanticCustomError(
                ERR_IMPORT,
                "cannot load custom metric '{path}': {reason}",
                {"path": self.path, "reason": str(err)},
            ) from err
        if not callable(obj):
            raise PydanticCustomError(
                ERR_NOT_CALLABLE,
                "custom metric '{path}' resolves to a non-callable {type}",
                {"path": self.path, "type": type(obj).__name__},
            )
        if self.kind == "metric":
            self._validate_metric_function(obj)
        else:
            self._validate_scorer(obj)
        return self

    def _validate_metric_function(self, fn: Any) -> None:
        """Check a ``kind="metric"`` function: two positionals and a direction."""
        if self.greater_is_better is None:
            raise PydanticCustomError(
                ERR_DIRECTION_REQUIRED,
                "custom metric '{path}' (kind='metric') must set "
                "greater_is_better: true for scores, false for errors",
                {"path": self.path},
            )
        reason = _check_positional_arity(fn, 2, self.kwargs)
        if reason is not None:
            raise PydanticCustomError(
                ERR_BAD_SIGNATURE,
                "custom metric '{path}' must accept (y_true, y_pred) and the "
                "configured kwargs: {reason}",
                {"path": self.path, "reason": reason},
            )

    def _validate_scorer(self, obj: Any) -> None:
        """Check a ``kind="scorer"`` object: three positionals, direction, kwargs."""
        if _is_sklearn_scorer(obj):
            scorer_gib = obj._sign > 0
            if (
                self.greater_is_better is not None
                and self.greater_is_better != scorer_gib
            ):
                raise PydanticCustomError(
                    ERR_DIRECTION_CONFLICT,
                    "custom scorer '{path}' has greater_is_better={scorer} "
                    "but the config sets {config}",
                    {
                        "path": self.path,
                        "scorer": scorer_gib,
                        "config": self.greater_is_better,
                    },
                )
            return
        if self.kwargs:
            raise PydanticCustomError(
                ERR_KWARGS_UNSUPPORTED,
                "custom scorer '{path}' is a plain callable: kwargs are only "
                "supported for kind='metric' and make_scorer scorers",
                {"path": self.path},
            )
        reason = _check_positional_arity(obj, 3, {})
        if reason is not None:
            raise PydanticCustomError(
                ERR_BAD_SIGNATURE,
                "custom scorer '{path}' must accept (estimator, X, y): {reason}",
                {"path": self.path, "reason": reason},
            )
        if not self.additive:
            logger.warning(
                "Custom scorer '%s' is a plain callable: its value over the "
                "pooled out-of-fold vector (and BBC-CV) cannot be computed, "
                "the per-fold mean is used instead. Set additive: true to "
                "declare this explicitly, or use a make_scorer scorer / "
                "kind: metric.",
                self.path,
            )

    @property
    def natural_greater_is_better(self) -> bool:
        """The natural direction of the metric value (``True`` — a score)."""
        if self.greater_is_better is not None:
            return self.greater_is_better
        obj = resolve_dotted_path(self.path)
        if _is_sklearn_scorer(obj):
            return bool(obj._sign > 0)
        return True

    @property
    def cache_key(self) -> tuple[str, str, bool | None, bool, str]:
        """A hashable identity of the spec (the scorer cache key)."""
        return (
            self.path,
            self.kind,
            self.greater_is_better,
            self.additive,
            json.dumps(self.kwargs, sort_keys=True, default=repr),
        )


class _CallableScorer:
    """Adapter giving a plain callable scorer the ``_sign`` of a sklearn scorer.

    The engine maximizes the scorer output and reads ``_sign`` to recover the
    natural value; a callable whose natural direction is "lower is better"
    is therefore negated here. No ``_score_func`` is exposed: the value over
    the pooled out-of-fold vector cannot be computed for such a scorer.
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
        merged = copy.copy(obj)
        merged._kwargs = {**obj._kwargs, **spec.kwargs}
        return merged
    return _CallableScorer(obj, 1 if spec.natural_greater_is_better else -1)


def build_custom_scorer(spec: CustomMetricSpec) -> Any:
    """Return the scorer object of a custom metric, cached per process.

    A ``kind="metric"`` function is wrapped with ``make_scorer`` (so the
    pooled out-of-fold and BBC-CV evaluation can call the raw function); a
    ``make_scorer`` scorer is used as is (``kwargs`` merged into a copy); a
    plain callable scorer is wrapped into an adapter exposing ``_sign``.

    Args:
        spec: The validated custom metric declaration.

    Returns:
        Any: A callable ``scorer(estimator, X, y) -> float`` (higher is
            better), exposing ``_sign``.

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
