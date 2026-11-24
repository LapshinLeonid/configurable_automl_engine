"""Explicit hand-off of the instrumented run result between modules.

The training engine publishes every assembled ``ExperimentResult`` into this
process-wide registry; consumers read the result only through the functions
below, never through a module-level variable. The registry keeps the latest
result of a ``train_best_model`` call (completed or failed) together with the
private run token of its producer.

A consumer that owns a run token asks only for the result stamped with that
token, so a concurrent or sequential run of another experiment never leaks
into its state. The registry is thread-safe: publishing, reading and the
ownership check are serialized by a lock, and the result is stamped before it
becomes visible.
"""

from __future__ import annotations

import threading
import uuid

from configurable_automl_engine.experiment.result import ExperimentResult

from .run_result import mark_result_owner, result_owned_by

__all__ = [
    "clear_experiment_result",
    "get_experiment_result_owned_by",
    "get_last_experiment_result",
    "publish_experiment_result",
]


class _ResultRegistry:
    """The mutable state behind the explicit result hand-off API."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_result: ExperimentResult | None = None

    def publish(
        self,
        result: ExperimentResult,
        run_token: uuid.UUID | None,
    ) -> None:
        """Store ``result`` under the registry lock, stamping its token first."""
        with self._lock:
            if run_token is not None:
                mark_result_owner(result, run_token)
            self._last_result = result

    def last(self) -> ExperimentResult | None:
        """Return the latest result regardless of its owner."""
        with self._lock:
            return self._last_result

    def owned_by(self, run_token: uuid.UUID) -> ExperimentResult | None:
        """Return the latest result only when it carries ``run_token``."""
        with self._lock:
            result = self._last_result
            if result is None or not result_owned_by(result, run_token):
                return None
            return result

    def clear(self) -> None:
        """Drop the published result."""
        with self._lock:
            self._last_result = None


_REGISTRY = _ResultRegistry()


def publish_experiment_result(
    result: ExperimentResult,
    *,
    run_token: uuid.UUID | None = None,
) -> None:
    """Publish ``result`` as the latest instrumented run result.

    When ``run_token`` is given, the result is stamped with it before it
    becomes visible, so a consumer can recognize the result as its own.

    Args:
        result: The assembled result model to publish.
        run_token: The private token of the producer, if any.
    """
    _REGISTRY.publish(result, run_token)


def get_last_experiment_result() -> ExperimentResult | None:
    """Return the latest published result, regardless of its owner.

    Returns:
        ExperimentResult | None: The latest result, or ``None`` when no run
            has published a result in this process.
    """
    return _REGISTRY.last()


def get_experiment_result_owned_by(
    run_token: uuid.UUID,
) -> ExperimentResult | None:
    """Return the latest result only when ``run_token`` produced it.

    Args:
        run_token: The private token of the caller.

    Returns:
        ExperimentResult | None: The latest result stamped with ``run_token``;
            ``None`` when the latest result belongs to another producer or
            no result was published.
    """
    return _REGISTRY.owned_by(run_token)


def clear_experiment_result() -> None:
    """Drop the published result (test isolation and explicit reset)."""
    _REGISTRY.clear()
