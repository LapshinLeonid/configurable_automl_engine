"""MLflowReporter: the optional MLflow tracking adapter (the T8
component).

The reporter is the *only* place in the package that knows about MLflow:
the ``mlflow`` import lives inside :meth:`MLflowReporter._mlflow` (a
lazy, guarded import via
:func:`configurable_automl_engine.common.dependency_utils.is_installed`
+ :func:`importlib.import_module`) — the engine core, the other reporters
and the result model never touch MLflow, and the package imports fine
with MLflow absent.

Logging policy (documented contract):

* **run management** — the reporter never opens a run of its own by
  default: with ``nested=False`` everything is logged into the *active*
  run (``mlflow.active_run()``, as managed by the caller with
  ``mlflow.start_run()``); with ``nested=True`` every ``report`` call
  starts a *nested* run named after the experiment ``run_id`` under the
  active one (the experiment hierarchy: one outer run per engine call,
  one nested run per reported ``ExperimentResult``). Without an active
  run ``nested=True`` promotes the run to top level (MLflow semantics of
  ``nested=True`` without a parent); ``nested=False`` raises
  :class:`MLflowReporterError` instead of silently inventing a run —
  the caller decides where the experiment lands;
* **re-reporting** — with ``nested=True`` every ``report`` call opens a
  fresh run, so the same result may be reported repeatedly. With
  ``nested=False`` the data goes into the one active run: re-reporting
  the *same* result is safe (identical tag/param values are idempotent
  in MLflow, a metric re-log is appended as an extra step), but
  re-reporting a *different* result into the same run overwrites the
  ``automl.*`` param keys — prefer one active run per
  ``ExperimentResult`` (or ``nested=True``);
* **failure policy** — a tracking-server failure (connection error,
  timeout, any MLflow API exception) does not kill the experiment: the
  error is logged as a WARNING and :meth:`~Reporter.report` returns
  ``None`` (a "reported-failed" artifact, the separate status the epic
  calls for). Errors raised *before* the MLflow API is touched (a
  missing active run with ``nested=False``) still raise — the caller
  misused the API and must learn it immediately. A failure mid-report
  leaves a *partial* run: ``_log_run`` logs tags → metrics → params →
  artifacts, so a ``log_params`` failure keeps the tags and metrics but
  loses the params and artifacts, and ``report`` returns ``None`` — the
  caller cannot distinguish a fully-logged run from a partially-logged
  one beyond the WARNING naming the failure (the re-reporting policy
  below is the remedy: report again into a fresh run);
* **metric sanitization** — MLflow rejects NaN/±inf metric values: they
  are skipped with a WARNING (a skipped metric is better than a failed
  experiment). ``ExperimentResult`` validates its score fields, but
  ``additional_metrics`` may carry NaN from sklearn — sanitized here;
* **parameter deduplication** — per-phase best parameters and the
  winner parameters may repeat names across phases (HPO refines the
  same hyperparameters in several phases): a key that is already logged
  in this ``report`` call gets the ``<algorithm>__<phase>`` prefix.
  MLflow log_param is idempotent-tolerant only for equal values, and a
  re-logged *different* value fails the run — the prefix makes every
  key unique instead. The separator is ``__``, not ``:``: MLflow's
  parameter-name validation regex for Windows (``^[/\\w.\\- ]*$``)
  rejects a colon, which would make ``log_params`` raise
  ``INVALID_PARAMETER_VALUE`` on Windows and — per the failure policy —
  degrade the whole report to a warning. Both the phase segment and the
  flattened parameter sub-keys are additionally clamped to a head/tail
  budget with an ASCII ``...`` marker: MLflow rejects an entity key
  over 250 characters (``MAX_ENTITY_KEY_LENGTH``) — a user-configured
  300-character phase name would otherwise fail ``log_params`` the same
  way — and the marker itself must pass MLflow's name-validation regex
  (``^[/\\w.\\- :]*$``), which the unicode ellipsis ``…`` does not. As a
  final guard the *composed* key (prefix + algorithm + phase + sub-key)
  is clamped to the limit too — the segment budgets alone do not bound
  the sum of the parts;
* **artifact sizes** — the three artifacts (the leaderboard text
  table, the JSON snapshot, the Markdown report) are rendered in
  memory and logged as text artifacts; the JSON snapshot holds the
  whole config and the run history, so for very large experiments its
  size is the MLflow artifact-store limit's problem, not a metric/tag
  limit (metrics and params stay small: one value per key);
* **tags** — the metric name/direction, the status and the winner are
  logged as tags: string metadata, no metric-value limits, unicode-safe
  (MLflow stores tags as text).

Autolog interaction: the reporter logs only its own experiment-level
metrics/params/tags (``winner_score``, ``oof_score``, the per-run
aggregates, the metric name and the run id) — it does not log training
metrics, so an ``mlflow.sklearn.autolog()`` active in the caller's run
does not conflict: autolog's step-level metrics come from the model
fits, the reporter's ones are the experiment outcome, under distinct
key names. No autolog calls are made by the reporter.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
from types import ModuleType
from typing import Any

from configurable_automl_engine.common.dependency_utils import is_installed
from configurable_automl_engine.experiment.leaderboard import build_leaderboard
from configurable_automl_engine.experiment.reporting.base import (
    Reporter,
    ReporterArtifact,
)
from configurable_automl_engine.experiment.reporting.markdown_reporter import (
    MarkdownReporter,
)
from configurable_automl_engine.experiment.result import ExperimentResult

__all__ = ["MLflowReporter", "MLflowReporterError"]

logger = logging.getLogger(__name__)

#: The package MLflow is imported from, once, lazily.
_MLFLOW_PACKAGE = "mlflow"

#: The prefix of every metric/param/tag key the reporter logs.
_KEY_PREFIX = "automl"

#: The separator between the algorithm and the phase in a per-phase
#: parameter key (``automl.param.<algorithm>__<phase>.<key>``). Two
#: underscores, not a colon: MLflow's parameter-name validation regex
#: for Windows (``^[/\w.\- ]*$``) rejects ``:``, so a colon would make
#: ``log_params`` raise ``INVALID_PARAMETER_VALUE`` on Windows and — per
#: the failure policy — degrade the whole report to a warning. ``__``
#: is accepted on every platform and cannot collide with the dot-
#: flattened sub-keys (a single ``.`` is the nesting separator).
_PHASE_SEPARATOR = "__"

#: The maximum MLflow entity-key length (``mlflow.utils.validation.
#: MAX_ENTITY_KEY_LENGTH``, 250 in MLflow 2.x/3.x). A generated key
#: longer than the limit makes ``log_params`` raise ``MlflowException``
#: and — per the failure policy — degrade the whole report to a warning,
#: so every key the reporter builds is clamped to it.
_MAX_KEY_LENGTH = 250

#: The head/tail budget for one clamped key segment: the phase name and
#: the flattened parameter sub-keys are user-configured and unbounded,
#: the constant parts of a key (``automl.param.`` and the algorithm
#: name) take little, so 100 + 100 characters per segment keep every
#: key far below :data:`_MAX_KEY_LENGTH` while staying recognizable
#: (head + ellipsis + tail).
_SEGMENT_BUDGET = 100

#: The truncation marker of a clamped key/segment. ASCII-only, and not
#: an arbitrary choice: MLflow validates every entity name against
#: ``^[/\w.\- :]*$`` (``validate_param_and_metric_name``), so the
#: unicode ellipsis ``…`` (U+2026) would make every clamped key itself
#: invalid — ``log_metric``/``log_params`` would raise
#: ``INVALID_PARAMETER_VALUE`` and, per the failure policy, the whole
#: report would degrade to a warning. ``...`` passes the regex on every
#: platform (and a clamped key is length-clamped last, so the extra two
#: characters cannot push it over :data:`_MAX_KEY_LENGTH`).
_SEGMENT_ELLIPSIS = "..."


class MLflowReporterError(RuntimeError):
    """Raised when the reporter cannot log to the active run.

    Only the *usage* failures raise (no active run with ``nested=False``);
    a tracking-server failure is handled in-band: logged as a warning,
    ``report`` returns ``None`` and the experiment continues.
    """


def _sanitize_metric(name: str, value: float) -> float | None:
    """Return ``value`` if MLflow can store it, else ``None`` (warn).

    MLflow rejects NaN/±inf metric values. The experiment's own score
    fields never carry a non-finite value (the model validation rejects
    it), but ``additional_metrics`` may carry NaN from sklearn — those
    are skipped with a warning instead of failing the report.
    """
    if not math.isfinite(float(value)):
        logger.warning(
            "[%s] metric %r is non-finite (%r) — skipped (MLflow rejects "
            "NaN/±inf values)",
            _MLFLOW_PACKAGE,
            name,
            value,
        )
        return None
    return float(value)


def _sanitize_param(value: Any) -> str:
    """Render one parameter value as an MLflow-safe string.

    MLflow stores parameters as strings; the JSON rendering keeps
    unicode readable (``ensure_ascii=False``) and treats non-finite
    floats the way the result model never lets them through anyway.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)


def _clamp_segment(segment: str) -> str:
    """Clamp one user-configured key segment to the head/tail budget.

    The phase names and the parameter sub-keys of the result are
    unconstrained strings, while MLflow rejects a metric/param key over
    250 characters (``MAX_ENTITY_KEY_LENGTH``) — such a key would, per
    the failure policy, degrade the whole report to a warning. A too-
    long segment is cut to head + ``...`` + tail so the key stays unique
    per (algorithm, phase, sub-key) combination and recognizable in the
    MLflow UI; collisions of two over-long segments are possible but
    harmlessly merge two params of the same clamped identity.
    """
    if len(segment) <= 2 * _SEGMENT_BUDGET + 1:
        return segment
    head = segment[:_SEGMENT_BUDGET]
    tail = segment[-_SEGMENT_BUDGET:]
    return f"{head}{_SEGMENT_ELLIPSIS}{tail}"


def _clamp_key(key: str) -> str:
    """Clamp a *composed* key to MLflow's entity-key limit (final guard).

    :func:`_clamp_segment` bounds each user-configured segment
    separately, but a key is the *sum* of its parts (``automl.param.`` +
    algorithm + ``__`` + phase + ``.`` + sub-key): two individually
    unclamped segments of ~105 characters plus a long algorithm name
    still compose a key over :data:`_MAX_KEY_LENGTH` — MLflow would
    reject it and, per the failure policy, the whole report would
    degrade to a warning. The tail cut keeps the head of the key (the
    constant prefix and the algorithm), which is the identifying part;
    the tail of the key is the sub-key path, already clamped to be
    recognizable via its own head/tail.
    """
    if len(key) <= _MAX_KEY_LENGTH:
        return key
    return key[:_MAX_KEY_LENGTH]


def _flatten_params(params: dict[str, Any], prefix: str) -> dict[str, str]:
    """Flatten one parameter mapping into ``prefix.key`` string values.

    The result model stores the best parameters as a JSON-serializable
    (possibly nested) mapping; the nested structure is flattened with a
    dot so MLflow's flat key-value params can hold it. Each generated
    key is clamped twice: the user-configured sub-key by
    :func:`_clamp_segment`, the composed ``prefix + sub-key`` by the
    final :func:`_clamp_key` guard — so the key never exceeds MLflow's
    entity-key limit no matter how the segment lengths add up.
    """
    flat: dict[str, str] = {}

    def _walk(mapping: dict[str, Any], path: str) -> None:
        for key, value in mapping.items():
            key_path = f"{path}.{key}" if path else str(key)
            if isinstance(value, dict):
                _walk(value, key_path)
            else:
                flat[_clamp_key(f"{prefix}{_clamp_segment(key_path)}")] = (
                    _sanitize_param(value)
                )

    _walk(params, "")
    return flat


class MLflowReporter(Reporter):
    """Log the experiment result into the active MLflow run.

    The adapter is the single MLflow-aware component of the package:
    the ``mlflow`` package is imported lazily inside
    :meth:`MLflowReporter._mlflow`, so the package installs and imports
    without MLflow (the optional extra ``[mlflow]`` installs it).

    Args:
        nested: The run policy. ``False`` (default) — log into the
            caller's active run (``mlflow.start_run()``); no active run
            → :class:`MLflowReporterError`. ``True`` — start a *nested*
            run per ``report`` call, named after the experiment
            ``run_id`` (promoted to a top-level run when no parent run
            is active — MLflow's own semantics).
        log_artifacts: Also log the three text artifacts (the
            leaderboard table, the JSON snapshot, the Markdown report)
            into the run (default ``True``).
        artifact_prefix: The directory prefix of the artifacts inside
            the run (default ``"experiment"``).
    """

    def __init__(
        self,
        *,
        nested: bool = False,
        log_artifacts: bool = True,
        artifact_prefix: str = "experiment",
    ) -> None:
        self._nested = nested
        self._log_artifacts = log_artifacts
        self._artifact_prefix = artifact_prefix

    # ─────────────────────────── mlflow access ─────────────────────── #

    @staticmethod
    def _mlflow() -> ModuleType:
        """Import and return the ``mlflow`` module (lazy, guarded).

        The only MLflow import of the package: performed at report time,
        never at package import time. A missing installation raises a
        defined error naming the optional extra instead of a bare
        ``ModuleNotFoundError``.
        """
        if not is_installed(_MLFLOW_PACKAGE):
            raise MLflowReporterError(
                "MLflowReporter requires the optional dependency 'mlflow': "
                "install the package with the [mlflow] extra "
                "(pip install 'configurable-automl-engine[mlflow]')"
            )
        return importlib.import_module(_MLFLOW_PACKAGE)

    # ─────────────────────────── reporting ─────────────────────────── #

    def report(self, result: ExperimentResult) -> ReporterArtifact | None:
        """Log the experiment result into the MLflow run.

        Args:
            result: The finished experiment result to log.

        Returns:
            ReporterArtifact | None: The run id (``str``) of the MLflow
                run the data was logged into, or ``None`` when the
                tracking server failed — the failure is logged as a
                warning and the experiment continues (the documented
                failure policy).

        Raises:
            MLflowReporterError: MLflow is not installed (the optional
                extra is missing), or ``nested=False`` and no MLflow
                run is active (the caller must open the run).
        """
        mlflow = self._mlflow()
        if self._nested:
            try:
                # MLflow's own semantics: nested=True without an active
                # parent promotes the run to top level.
                with mlflow.start_run(run_name=str(result.run_id), nested=True) as run:
                    self._log_run(mlflow, result)
                    return str(run.info.run_id)
            except Exception as err:  # noqa: BLE001 — the failure policy
                return self._tracking_failure(result, err)
        if mlflow.active_run() is None:
            raise MLflowReporterError(
                "MLflowReporter: no active MLflow run — open one with "
                "mlflow.start_run() before reporting, or construct the "
                "reporter with nested=True"
            )
        try:
            self._log_run(mlflow, result)
            return str(mlflow.active_run().info.run_id)
        except Exception as err:  # noqa: BLE001 — the failure policy
            return self._tracking_failure(result, err)

    @staticmethod
    def _tracking_failure(result: ExperimentResult, err: Exception) -> None:
        """Handle a tracking-server failure per the failure policy.

        The error is logged as a WARNING (with the run id, so the
        failed tracking stays attributable) and ``report`` returns
        ``None`` — the experiment itself is never failed by the
        tracking sink.
        """
        logger.warning(
            "MLflowReporter: logging the result %s failed (%s: %s) — "
            "the experiment continues without MLflow tracking",
            result.run_id,
            type(err).__name__,
            err,
        )

    def _log_run(self, mlflow: ModuleType, result: ExperimentResult) -> None:
        """Log metrics, params, tags and artifacts into the active run."""
        mlflow.set_tag(f"{_KEY_PREFIX}.run_id", str(result.run_id))
        mlflow.set_tag(f"{_KEY_PREFIX}.status", result.status.value)
        mlflow.set_tag(f"{_KEY_PREFIX}.metric", result.metric_user)
        mlflow.set_tag(f"{_KEY_PREFIX}.metric_direction", result.metric_direction.value)
        mlflow.set_tag(f"{_KEY_PREFIX}.metric_sklearn", result.metric_sklearn)
        if result.label:
            mlflow.set_tag(f"{_KEY_PREFIX}.label", result.label)
        if result.engine_version:
            mlflow.set_tag(f"{_KEY_PREFIX}.engine_version", result.engine_version)
        if result.error:
            mlflow.set_tag(f"{_KEY_PREFIX}.error", result.error)

        for name, value in self._metrics(result).items():
            sanitized = _sanitize_metric(name, value)
            if sanitized is not None:
                mlflow.log_metric(name, sanitized)

        params: dict[str, str] = {
            f"{_KEY_PREFIX}.metric_user": result.metric_user,
            f"{_KEY_PREFIX}.resolved_validation": result.resolved_validation,
            f"{_KEY_PREFIX}.resolved_n_folds": str(result.resolved_n_folds),
            f"{_KEY_PREFIX}.random_state": str(result.random_state),
        }
        winner_params = _flatten_params(result.winner_params, f"{_KEY_PREFIX}.param.")
        phase_params: list[tuple[str, str]] = []
        for record in result.runs:
            prefix = (
                f"{_KEY_PREFIX}.param.{record.algorithm}"
                f"{_PHASE_SEPARATOR}{_clamp_segment(record.phase_name)}."
            )
            phase_params.extend(_flatten_params(record.params, prefix).items())
        # Re-clamp the composed phase keys: prefix + sub-key may exceed
        # the MLflow entity-key limit even with every segment clamped
        # (see _clamp_key).
        phase_params = [(_clamp_key(key), value) for key, value in phase_params]
        for param_key, param_value in winner_params.items():
            params[param_key] = param_value
        for param_key, param_value in phase_params:
            # The winner's parameters are already logged un-prefixed
            # (the winner is the experiment outcome); a per-phase
            # record of the same algorithm/phase keeps its unique
            # key, a duplicate name across phases never collides.
            params.setdefault(param_key, param_value)
        mlflow.log_params(params)

        if self._log_artifacts:
            self._log_artifacts_to_run(mlflow, result)

    def _metrics(self, result: ExperimentResult) -> dict[str, float]:
        """The experiment metrics to log (``key → value``).

        The winner score and the honest estimates use the metric name
        in the key, so a run metric is unambiguous in the MLflow UI;
        the per-algorithm scores carry the algorithm and the phase in
        the key.
        """
        metric = result.metric_user
        metrics: dict[str, float] = {}
        if result.winner_score is not None:
            metrics[f"{_KEY_PREFIX}.winner_score_{metric}"] = result.winner_score
        if result.holdout_score is not None:
            metrics[f"{_KEY_PREFIX}.holdout_score_{metric}"] = result.holdout_score
        if result.bbc_score is not None:
            metrics[f"{_KEY_PREFIX}.bbc_score_{metric}"] = result.bbc_score
        if result.oof_score is not None:
            metrics[f"{_KEY_PREFIX}.oof_score_{metric}"] = result.oof_score
        if result.duration_seconds is not None:
            metrics[f"{_KEY_PREFIX}.duration_seconds"] = result.duration_seconds
        for name, value in result.additional_metrics.items():
            metrics[_clamp_key(f"{_KEY_PREFIX}.additional.{_clamp_segment(name)}")] = (
                value
            )
        for record in result.runs:
            if record.score is None:
                continue
            metrics[
                _clamp_key(
                    f"{_KEY_PREFIX}.score.{record.algorithm}."
                    f"{_clamp_segment(record.phase_name)}"
                )
            ] = record.score
        return metrics

    def _log_artifacts_to_run(
        self, mlflow: ModuleType, result: ExperimentResult
    ) -> None:
        """Render and log the leaderboard, the JSON and the Markdown."""
        leaderboard = build_leaderboard(result)
        markdown = MarkdownReporter().report(result)
        json_payload = result.to_json(indent=2)
        artifacts = {
            f"{self._artifact_prefix}/leaderboard.txt": (
                leaderboard.to_table()
                if leaderboard.entries
                else leaderboard.to_summary()
            ),
            f"{self._artifact_prefix}/report.json": json_payload,
            f"{self._artifact_prefix}/report.md": markdown,
        }
        for path, text in artifacts.items():
            mlflow.log_text(text, path)
