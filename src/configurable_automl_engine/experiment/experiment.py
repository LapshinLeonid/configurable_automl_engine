"""``AutoMLExperiment`` — the object facade of the engine (the T5
component).

The facade is the central entity of the experiment epic: it wraps the
instrumented training loop (``training_engine.train_best_model``, T3),
registers the assembled :class:`~configurable_automl_engine.experiment.result.ExperimentResult`
in an :class:`~configurable_automl_engine.experiment.store.ExperimentStore`
(T4) and exposes the consumed artifacts:

    experiment = AutoMLExperiment(config, data, target="target")
    experiment.run()
    experiment.leaderboard()
    experiment.best_model()
    experiment.report()
    experiment.export(directory)
    experiment.explain()           # feature importance of the winner

Documented contract:

* the config argument is passed to ``load_config`` unchanged: a ``Config``
  object, a plain dict, a YAML file path (``str`` or ``pathlib.Path``) —
  every form the training engine accepts today keeps working;
* the training result is registered in the ``ExperimentStore`` of the
  experiment (an in-memory store by default; pass ``store`` or ``store_path``
  to persist the history as JSONL) and kept as the ``result`` attribute;
* the methods that need a finished run (``leaderboard``, ``best_model``,
  ``report``, ``export``, ``explain``, ``ensemble_model``) raise
  ``RuntimeError`` with a clear state message when called before
  ``run()``;
* a repeated ``run()`` is allowed: it trains again, produces a new
  ``run_id`` and replaces ``result`` (a ``RuntimeWarning`` is emitted so
  the repeated training is visible in the log);
* the data is snapshotted at ``run()`` time (a shallow copy of the
  DataFrame): mutations of the caller's frame after ``run()`` do not
  change what was trained. The snapshot is a copy of the frame object,
  not a deep copy of every block — mutating the column *values in place*
  (``df.loc[...] = ...``) still writes through to the training data, as
  with any pandas copy; assign a new frame instead;
* ``export(directory)`` bundles the consumed artifacts of the run
  into the destination directory (see
  :mod:`configurable_automl_engine.experiment.export`);
* ``best_model()`` returns the winner's ``ModelTrainer``: it is loaded
  from the artifact path recorded in the result (auto-detected format);
* the facade holds no global state: two experiments in different threads
  share nothing and can run concurrently (the engine itself supports
  ``general.parallel_mode: threads``); the engine's module-level
  ``last_experiment_result`` global is recognized as *this* run's
  result by a private owner stamp (``run_result.mark_result_owner``), so
  a concurrent or sequential stranger's run never leaks into the
  experiment's own state.

``explain()`` explains the winner: the permutation importance of the
original feature columns (always available) or the SHAP contributions
(the optional ``[shap]`` extra; see
:mod:`configurable_automl_engine.experiment.explain`).
"""

from __future__ import annotations

import copy
import logging
import uuid
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from configurable_automl_engine.common.datetime_features import (
    expand_datetime_columns,
)
from configurable_automl_engine.experiment.explain import (
    DEFAULT_N_REPEATS,
    DEFAULT_SAMPLE_ROWS,
    ExplainNotAvailableError,
    PermutationImportanceResult,
    ShapFeatureImportance,
    explain_permutation_importance,
    explain_shap,
    render_importance_markdown,
)
from configurable_automl_engine.experiment.leaderboard import (
    Leaderboard,
    build_leaderboard,
)
from configurable_automl_engine.experiment.reporting import (
    JSONReporter,
    MarkdownReporter,
    Reporter,
    ReporterArtifact,
)
from configurable_automl_engine.experiment.result import ExperimentResult
from configurable_automl_engine.experiment.store import ExperimentStore

if TYPE_CHECKING:
    from configurable_automl_engine.trainer import ModelTrainer

__all__ = [
    "AutoMLExperiment",
    "ExperimentNotRunError",
    "ExplainNotAvailableError",
]

#: The explain methods of ``AutoMLExperiment.explain``.
_EXPLAIN_METHODS = frozenset({"permutation", "shap"})

_LOG = logging.getLogger(__name__)

#: The report file written by the default ``report()`` reporter chain,
#: relative to the run working directory.
DEFAULT_REPORT_JSON_NAME = "experiment_report.json"


class ExperimentNotRunError(RuntimeError):
    """An experiment method was called before :meth:`AutoMLExperiment.run`.

    Raised by every result-consuming method of the facade (``leaderboard``,
    ``best_model``, ``report``, ``export``, ``explain``): there is no run
    result to serve yet. The message names the state and the fix (call
    ``run()`` first).
    """


class AutoMLExperiment:
    """Object facade over one training run of the engine.

    The experiment binds together the configuration, the data and the
    run history: ``run()`` executes the full training cycle, then every
    other method serves a facet of the finished run.

    Args:
        config: The training configuration: a ``Config`` object, a plain
            dict, or a path (``str``/``Path``) to a YAML file — the same
            forms ``training_engine.train_best_model`` accepts.
        data: The input dataset. The frame is validated and snapshotted
            at ``run()`` time; mutations of the caller's frame afterwards
            do not change what was trained (see the module docstring).
        target: The target column name (default ``"target"``).
        store: The run-history store to register the result in. ``None``
            creates a private in-memory store.
        store_path: Path of a JSONL history file; when given, the result
            of every ``run()`` is appended to the file and the history is
            restored from it at construction. Ignored when ``store`` is
            passed explicitly.
        label: Optional human-readable label of the run, attached to the
            ``ExperimentResult`` and visible in the reports.
        model_path_override: Alternative artifact save path (forwarded to
            the training engine; default ``None`` keeps the configured
            ``general.path_to_model``).
    """

    def __init__(
        self,
        config: str | Path | dict[str, Any] | Any,
        data: pd.DataFrame,
        *,
        target: str | None = None,
        store: ExperimentStore | None = None,
        store_path: str | Path | None = None,
        label: str | None = None,
        model_path_override: str | Path | None = None,
    ) -> None:
        """Create the experiment (no engine imports happen here).

        Args:
            config: The training configuration: a ``Config`` object, a
                plain dict, or a path (``str``/``Path``) to a YAML file —
                the same forms the training engine accepts.
            data: The input dataset (snapshotted at ``run()`` time).
            target: The target column name (default ``"target"``).
            store: The run-history store to register the result in;
                ``None`` creates a private in-memory store.
            store_path: Path of a JSONL history file; when given, the
                result of every ``run()`` is appended to the file and the
                history is restored from it at construction. Ignored when
                ``store`` is passed explicitly.
            label: Optional human-readable label of the run, attached to
                the ``ExperimentResult`` and visible in the reports.
            model_path_override: Alternative artifact save path (forwarded
                to the training engine; default ``None`` keeps the
                configured ``general.path_to_model``).
        """
        self._config = config
        self._data = data
        self._target = target
        self._label = label
        self._model_path_override = model_path_override
        self._store = store if store is not None else ExperimentStore(path=store_path)
        self._result: ExperimentResult | None = None
        # The private owner stamp passed into the engine's run: the engine
        # stamps its assembled ExperimentResult with this token, and the
        # facade accepts from the shared ``last_experiment_result`` global
        # only the result carrying its own token (concurrent or sequential
        # stranger's runs never leak into this experiment's state).
        self._run_token = uuid.uuid4()

    # ─────────────────────────── state views ────────────────────────── #

    @property
    def store(self) -> ExperimentStore:
        """The run-history store of the experiment."""
        return self._store

    @property
    def result(self) -> ExperimentResult | None:
        """The result of the last ``run()`` (``None`` before it)."""
        return self._result

    @property
    def run_id(self) -> Any:
        """The ``run_id`` of the last ``run()`` (``None`` before it)."""
        return self._result.run_id if self._result is not None else None

    # ──────────────────────────── execution ─────────────────────────── #

    def run(self) -> ExperimentResult:
        """Execute the full training cycle and register the result.

        The call delegates to the instrumented
        ``training_engine.train_best_model`` (the same entry point the
        functional API uses), takes the assembled ``ExperimentResult``,
        registers it in the experiment store and keeps it as ``result``.

        Re-running is allowed: a repeated call trains again and produces
        a new ``run_id`` (the result of the previous run stays in the
        store's history). A ``RuntimeWarning`` is emitted on the repeat so
        the duplicate training is not silent.

        Registration policy: the result of the new run (completed or
        failed) is registered in the store when the run produced one; the
        result is recognized by this experiment's private owner stamp, so
        a concurrent run of another experiment never leaks here. A store
        registration failure is logged as a warning and never masks the
        outcome of the run itself.

        Returns:
            ExperimentResult: The result of the new run.

        Raises:
            TypeError: A config of an unsupported type was passed (the
                engine's validation errors propagate unchanged).
            RuntimeError: The orchestration failed (a phase produced no
                valid scores, an empty working set, ...). A failed
                ``ExperimentResult`` is still recorded in the store.
            ValueError: The input data is empty or the target column is
                missing (the engine's data validation errors propagate
                unchanged).
        """
        if self._result is not None:
            warnings.warn(
                "AutoMLExperiment.run() called again: the previous result "
                f"(run_id={self._result.run_id}) stays in the store history "
                "and a new training run starts.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Snapshot the data at run() time (see the module docstring for
        # the exact copy semantics).
        data = copy.copy(self._data)
        # The engine layers are imported lazily inside run(): the facade
        # module must stay importable without the training engine (the
        # layered-imports contract and the pure result consumers).
        from configurable_automl_engine.training_engine import train_best_model

        try:
            train_best_model(
                self._config,
                data,
                target=self._target,
                model_path_override=self._model_path_override,
                run_token=self._run_token,
            )
        finally:
            # Both a completed and a failed run leave an ExperimentResult
            # behind (the T3 plumbing records the failed result too); it
            # is recognized by this experiment's private owner stamp —
            # the global is the result of the LAST engine call in the
            # process and may belong to a concurrent stranger's run.
            self._adopt_result()
        if self._result is None:
            # A distinct matchable prefix: this is NOT the "has not been
            # run yet" state — the engine ran but no result carrying this
            # experiment's owner stamp was recorded.
            raise ExperimentNotRunError(
                "No experiment result recorded: the training engine did "
                "not produce an ExperimentResult carrying this "
                "experiment's owner stamp; the run outcome is unknown. "
                "This points to an engine/facade version mismatch."
            )
        return self._result

    def _adopt_result(self) -> None:
        """Adopt the run result from the engine, if it is ours.

        Reads the engine's ``last_experiment_result`` global and accepts
        it only when it carries this experiment's owner stamp. Failures
        of the store registration are logged and do not mask the run
        outcome (the run's own error propagates from the engine call).
        """
        from configurable_automl_engine.training_engine.component import (
            last_experiment_result,
        )
        from configurable_automl_engine.training_engine.run_result import (
            result_owned_by,
        )

        if last_experiment_result is None or not result_owned_by(
            last_experiment_result, self._run_token
        ):
            return
        result = last_experiment_result
        self._attach_label(result)
        try:
            self._store.add(result)
        except Exception as err:  # noqa: BLE001 — registration must not mask the run outcome
            _LOG.warning(
                "Experiment run %s could not be registered in the store: %s",
                result.run_id,
                err,
            )
        _LOG.info(
            "Experiment run %s registered (status: %s)",
            result.run_id,
            result.status.value,
        )
        self._result = result

    def _attach_label(self, result: ExperimentResult) -> None:
        """Attach the configured run label to the result (if any)."""
        if self._label is None:
            return
        # The model is mutable (pydantic BaseModel without frozen config);
        # set the label before the store registration so it is persisted.
        result.label = self._label

    # ──────────────────────────── consumers ─────────────────────────── #

    def _require_result(self) -> ExperimentResult:
        """The last run result, or the not-run state error."""
        if self._result is None:
            raise ExperimentNotRunError(
                "Experiment has not been run yet: call run() before "
                f"{type(self).__name__} result accessors "
                "(leaderboard/best_model/report/export/explain)."
            )
        return self._result

    def leaderboard(
        self,
        *,
        print_table: bool = False,
        summary: bool = False,
    ) -> Leaderboard:
        """Return the algorithm ranking table of the last run.

        Args:
            print_table: Also print the rendered table to stdout.
            summary: Also print the UX summary block (the status line
                ``Experiment completed`` / ``Experiment failed``, the
                ranked list and the Winner line — the report layout of
                the epic #37 UX goal) to stdout. ``print_table`` and
                ``summary`` are independent: both may be printed.

        Returns:
            Leaderboard: The ranked leaderboard built from the run result
                (possibly empty for a failed run).

        Raises:
            ExperimentNotRunError: The experiment has not been run yet.
        """
        result = self._require_result()
        board = build_leaderboard(result)
        if print_table:
            print(board.to_table())
        if summary:
            print(board.to_summary(status=True))
        return board

    def best_model(self) -> ModelTrainer:
        """Return the trained model of the last run.

        The trainer is loaded from the artifact path recorded in the
        result (the serialization format is auto-detected from the file
        extension with a joblib-then-pickle fallback); the artifact path
        is the single source of truth, so the returned instance always
        matches the adopted result.

        Returns:
            ModelTrainer: The winner's trained model.

        Raises:
            ExperimentNotRunError: The experiment has not been run yet.
            RuntimeError: The run failed and no model artifact exists, or
                the recorded artifact cannot be loaded as a
                ``ModelTrainer``.
        """
        result = self._require_result()
        model_path = result.model_path
        if not model_path:
            raise RuntimeError(
                "No model artifact available: the run failed before the "
                "final model was saved (status: "
                f"{result.status.value}); best_model() cannot load a "
                "trainer."
            )
        from configurable_automl_engine.trainer import ModelTrainer

        return ModelTrainer.load(model_path)

    def ensemble_model(self) -> Any:
        """Return the experimental top-K ensemble of the last run.

        The ensemble is built by the engine when ``general.ensemble.enable``
        is true; the section (members, scores, weights, artifact path) is
        recorded in ``ExperimentResult.ensemble`` and the serialized model
        is loaded from the artifact path recorded there — the single
        source of truth, like ``best_model()``.

        Returns:
            The loaded ``EnsembleModel`` (a weighted-average predictor
            with a sklearn-style ``predict(X)``).

        Raises:
            ExperimentNotRunError: The experiment has not been run yet.
            RuntimeError: The run produced no ensemble (the feature is
                disabled, the run failed, or the ensemble step was
                skipped), or the recorded ensemble artifact cannot be
                loaded.
        """
        result = self._require_result()
        section = result.ensemble
        if not section or not section.get("artifact_path"):
            raise RuntimeError(
                "No ensemble available: the run did not produce an ensemble "
                "(enable general.ensemble in the configuration; the winner "
                "model is available via best_model())."
            )
        from configurable_automl_engine.experiment.ensemble import EnsembleModel

        return EnsembleModel.load(section["artifact_path"])

    def report(
        self,
        reporter: Reporter[Any] | None = None,
    ) -> ReporterArtifact | tuple[ReporterArtifact, ...]:
        """Run the reporter chain on the last run result.

        The default chain renders the Markdown report to stdout and
        writes the full JSON snapshot next to the model artifact (or into
        the current working directory when no artifact was saved).

        Args:
            reporter: A custom reporter (or a ``CompositeReporter``)
                replacing the default chain.

        Returns:
            The artifact(s) produced by the reporter chain: a ``str`` for
            the default chain (the rendered Markdown), a ``Path`` for a
            file reporter, a tuple for a composite.

        Raises:
            ExperimentNotRunError: The experiment has not been run yet.
        """
        result = self._require_result()
        if reporter is not None:
            return reporter.report(result)
        markdown_text, _json_path = _default_report_chain(result)
        print(markdown_text)
        return markdown_text

    def export(self, directory: str | Path) -> Path:
        """Export the artifact bundle of the last run into ``directory``.

        The export is delegated to
        :func:`configurable_automl_engine.experiment.export.export_experiment`
        (the T11 component): the model artifact (a byte-exact copy; the
        original file is never touched), the leaderboard (JSON and
        Markdown), the full Markdown report, the JSON result snapshot
        and the run metadata (identity, status, timings, engine
        version, the exported-file map) — every file prefixed with the
        ``run_id``. Missing destination directories are created
        recursively; an existing bundle of the same run is overwritten;
        the bundle is assembled atomically (a failure never leaves a
        partial export).

        Args:
            directory: Destination directory of the export bundle.

        Returns:
            Path: The destination directory containing the bundle.

        Raises:
            ExportError: The destination exists but is not a directory,
                or the recorded model artifact is missing on disk.
            OSError: The destination is not writable.
            ExperimentNotRunError: The experiment has not been run yet.
        """
        result = self._require_result()
        from configurable_automl_engine.experiment.export import export_experiment

        return export_experiment(result, directory)

    def explain(
        self,
        method: str = "permutation",
        X: pd.DataFrame | None = None,
        y: Any | None = None,
        *,
        n_repeats: int = DEFAULT_N_REPEATS,
        max_rows: int = DEFAULT_SAMPLE_ROWS,
        post_preprocessing: bool = False,
        markdown: bool = True,
    ) -> Any:
        """Explain the winner model of the last run.

        Two computations are available (one per call):

        * ``method="permutation"`` (default) — the permutation importance
          of every original input column: the mean change of the run's
          comparison metric when one column is shuffled. Works through
          the trainer's own ``predict`` (the raw pre-pipeline schema),
          so one importance per original column comes out regardless of
          the internal encoding/selection. The target ``y`` is required
          (the metric needs the true values); ``X``/``y`` default to the
          run's training frame and target.
        * ``method="shap"`` — mean absolute SHAP contributions (the
          optional ``[shap]`` extra; a missing installation raises
          ``ExplainNotAvailableError`` naming the extra). By default the
          per-component values are aggregated to the original columns;
          ``post_preprocessing=True`` reports them under their pipeline
          names instead. The target is not used.

        The importance rows are the run's training rows (in-sample for
        the saved artifact — the final model is fit on all data); pass
        ``X``/``y`` explicitly to score a different block (e.g. a
        held-out frame).

        Args:
            method: ``"permutation"`` or ``"shap"`` (case-insensitive).
            X: The feature frame to explain on (default — the training
                frame of the run, without the target column).
            y: The target vector aligned with ``X`` (default — the
                training target of the run; required for permutation).
            n_repeats: Shuffles per feature (permutation only).
            max_rows: The row budget of the computation (both methods;
                larger data is subsampled deterministically).
            post_preprocessing: SHAP only: report the per-component
                values under their pipeline names instead of aggregating
                to the original columns.
            markdown: Also render the importance table as a Markdown
                section (returned as the second element of the tuple).

        Returns:
            The structured importance of the requested method (a
            ``PermutationImportanceResult`` or a ``ShapFeatureImportance``);
            with ``markdown=True`` a ``(importance, str)`` tuple — the
            second element is the rendered Markdown section.

        Raises:
            ExperimentNotRunError: The experiment has not been run yet.
            RuntimeError: The run failed and no model artifact exists,
                or the recorded artifact cannot be loaded as a
                ``ModelTrainer`` (propagated from ``best_model``).
            ExplainNotAvailableError: The computation cannot be served:
                the model has no fitted pipeline, ``y`` is missing for
                the permutation importance, the data is empty or has no
                feature columns, an unknown ``method`` was passed, or
                SHAP was requested without the ``[shap]`` extra.
        """
        result = self._require_result()
        method_key = method.lower().strip()
        if method_key not in _EXPLAIN_METHODS:
            raise ExplainNotAvailableError(
                f"Unknown explain method: {method!r}. "
                f"Expected one of {sorted(_EXPLAIN_METHODS)}."
            )
        trainer = self.best_model()
        X_explain, y_explain = self._explain_data(X, y)
        importance: PermutationImportanceResult | ShapFeatureImportance
        if method_key == "shap":
            importance = explain_shap(
                trainer,
                X_explain,
                result,
                max_rows=max_rows,
                post_preprocessing=post_preprocessing,
            )
        else:
            if y_explain is None:
                raise ExplainNotAvailableError(
                    "Permutation importance requires the target values: "
                    "pass y explicitly (or leave the defaults, so the "
                    "training target of the run is used)."
                )
            importance = explain_permutation_importance(
                trainer,
                X_explain,
                y_explain,
                result,
                n_repeats=n_repeats,
                max_rows=max_rows,
            )
        if not markdown:
            return importance
        baseline: float | None
        metric_user: str | None
        if isinstance(importance, PermutationImportanceResult):
            baseline = importance.baseline_score
            metric_user = importance.metric_user
        else:
            baseline = None
            metric_user = None
        section = render_importance_markdown(
            f"Feature importance ({method_key})",
            importance.importances,
            baseline=baseline,
            metric_user=metric_user,
        )
        return importance, section

    def _explain_data(
        self,
        X: pd.DataFrame | None,
        y: Any | None,
    ) -> tuple[pd.DataFrame, Any | None]:
        """The frame/target the explanation is computed on.

        The defaults are the run's training data: the feature columns of
        the snapshotted input frame (the target column removed) and the
        target vector. The datetime expansion recorded on the trainer
        (``datetime_policy: 'expand'``) is applied to the default frame:
        the saved pipeline was fit on the generated component columns
        (``ts_month`` …), which do not exist in the raw input — without
        the expansion the alignment to ``trainer.feature_names`` would
        fail. An explicit ``X`` is used as-is (the caller owns the
        schema; the module-level explain functions apply the expansion to
        it as well); with an explicit ``X`` but no ``y``, ``y`` stays
        ``None`` (the permutation importance then raises a defined
        error).

        When the trainer cannot be loaded (a failed run without an
        artifact), the raw default frame is returned unchanged: the
        explain functions raise their own defined errors, and masking a
        genuine artifact-loading failure here would only move the
        confusing message further from its cause.
        """
        if X is not None:
            return X, y
        if not isinstance(self._data, pd.DataFrame):
            return self._data, y
        data = copy.copy(self._data)
        target = self._target or "target"
        if target in data.columns:
            features = data.drop(columns=[target])
            target_values: Any = data[target]
        else:
            features = data
            target_values = None
        trainer = self.best_model()
        expansion = trainer.datetime_expansion_
        if expansion:
            features = expand_datetime_columns(features, expansion)
        return features, target_values


# --------------------------------------------------------------------------- #
#  Module-level helper (kept out of the class: it touches the engine's        #
#  internal plumbing and must stay import-light for the facade consumers).    #
# --------------------------------------------------------------------------- #


def _default_report_chain(
    result: ExperimentResult,
) -> tuple[str, Path]:
    """Run the default report chain: Markdown to stdout + JSON to a file.

    The JSON snapshot is written next to the saved model artifact when
    the run produced one, otherwise into the current working directory.

    Returns:
        tuple[str, Path]: The rendered Markdown text and the written JSON
            path (the JSON reporter's artifact).
    """
    if result.model_path:
        json_dir = Path(result.model_path).parent
    else:
        json_dir = Path.cwd()
    json_path = json_dir / DEFAULT_REPORT_JSON_NAME
    markdown_text = str(MarkdownReporter().report(result))
    JSONReporter(json_path).report(result)
    return markdown_text, json_path
