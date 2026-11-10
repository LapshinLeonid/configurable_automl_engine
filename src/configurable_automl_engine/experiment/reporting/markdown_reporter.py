"""MarkdownReporter: the human-readable report of an experiment.

The report embeds the leaderboard (``build_leaderboard``) as a Markdown table, adds the run metadata, the winner block
and the per-run detail sections. Every value comes from the
:class:`~configurable_automl_engine.experiment.result.ExperimentResult`
model — the reporter never reaches into the engine and never imports
MLflow.

Semantics of the reported scores:

* the leaderboard and the winner block show the user-semantic scores
  (positive RMSE/MAE, plain R²) — the raw optimizer values, including
  the neg_-metric inversion, never reach the report;
* the honest out-of-sample estimates (``holdout_score``, ``bbc_score``,
  ``oof_score``) are shown in a dedicated block next to the optimistic
  winner score, each labeled with its semantics, so the reader can tell
  an independent estimate from the selection score.

Markdown safety: names, labels, parameters and error messages come from
arbitrary configuration and may contain table-breaking characters
(``|``, newlines), heading markers (``#``) or emphasis (``*``/``_``).
Every cell is escaped before it enters a table row.
"""

from __future__ import annotations

import json
from typing import Any

from configurable_automl_engine.common.definitions import RunStatus, RunStepStatus
from configurable_automl_engine.experiment.leaderboard import build_leaderboard
from configurable_automl_engine.experiment.reporting.base import (
    Reporter,
    ReporterArtifact,
)
from configurable_automl_engine.experiment.result import (
    AlgorithmRun,
    ExperimentResult,
)

__all__ = ["MarkdownReporter"]

#: The placeholder of the run label when the result has none.
_UNTITLED = "Untitled run"

#: The Markdown table cell rendering of an absent value.
_EMPTY_CELL = "—"

#: The placeholder of a leaderboard with no ranked entries.
_EMPTY_LEADERBOARD_ROW = (
    "_No ranked algorithms (no completed records with valid scores)._"
)


def _escape_cell(value: object) -> str:
    """Escape ``value`` for one Markdown table cell.

    Pipes are escaped with a backslash (they would split the table row),
    newlines and carriage returns become spaces (a cell must stay on one
    line), and the surrounding whitespace is stripped. Everything else
    (``*``, ``#``, ``_``, ``<`` …) is safe inside a plain table cell.
    """
    text = str(value)
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )


def _escape_inline(value: object) -> str:
    """Escape ``value`` for a plain Markdown line (headings, list items).

    Newlines become spaces (a heading or a list item must stay on one
    line); the surrounding whitespace is stripped. ``*``/``_``/``#`` in
    the middle of a text line do not start a new block in Markdown, so
    they are left readable.
    """
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def _escape_code_span(value: object) -> str:
    """Render ``value`` as the content of a ``code span``.

    A backtick run inside the value would close the span early, so the
    value is wrapped with a longer backtick fence when it contains
    backticks (the CommonMark rule: a fence must be longer than any run
    inside).
    """
    text = str(value)
    fence = "`"
    while fence in text:
        fence += "`"
    return f"{fence}{text}{fence}"


def _format_score(score: float | None) -> str:
    """Render one score cell (three decimal places, like the leaderboard)."""
    return _EMPTY_CELL if score is None else f"{score:.3f}"


def _format_seconds(seconds: float | None) -> str:
    """Render one duration cell in seconds (two decimal places)."""
    return _EMPTY_CELL if seconds is None else f"{seconds:.2f}s"


def _format_params(params: dict[str, Any]) -> str:
    """Render the best-parameters mapping as one inline-code cell."""
    if not params:
        return _EMPTY_CELL
    inner = ", ".join(f"{key}={value!r}" for key, value in params.items())
    return _escape_code_span(_escape_cell(inner))


class MarkdownReporter(Reporter[ReporterArtifact]):
    """Render the experiment report as a Markdown document.

    The report structure (in order): the title (the run label), the run
    metadata block, the outcome (status, winner, honest-score
    semantics), the leaderboard table, the stored explain() payload
    (when the run recorded one), the per-run detail sections and
    the failure message for a failed run.

    Args:
        include_config: Embed the full configuration snapshot as a
            collapsed ``<details>`` JSON block (default ``True``).
    """

    def __init__(self, *, include_config: bool = True) -> None:
        self._include_config = include_config

    def report(self, result: ExperimentResult) -> ReporterArtifact:
        """Render the experiment result as a Markdown document.

        Args:
            result: The finished experiment result to render.

        Returns:
            ReporterArtifact: The rendered Markdown document.
        """
        lines: list[str] = []
        self._render_header(lines, result)
        self._render_metadata(lines, result)
        self._render_outcome(lines, result)
        self._render_honest_scores(lines, result)
        self._render_leaderboard(lines, result)
        self._render_ensemble(lines, result)
        self._render_explanations(lines, result)
        self._render_run_details(lines, result)
        self._render_failure(lines, result)
        if self._include_config:
            self._render_config(lines, result)
        return "\n".join(lines).rstrip() + "\n"

    # ─────────────────────────── sections ──────────────────────────── #

    def _render_header(self, lines: list[str], result: ExperimentResult) -> None:
        title = result.label or _UNTITLED
        lines.append(f"# Experiment report: {_escape_inline(title)}")
        lines.append("")

    def _render_metadata(self, lines: list[str], result: ExperimentResult) -> None:
        lines.append("## Run metadata")
        lines.append("")
        lines.append(f"- Run ID: {_escape_code_span(str(result.run_id))}")
        if result.engine_version:
            lines.append(f"- Engine version: {_escape_inline(result.engine_version)}")
        lines.append(f"- Started: {_escape_inline(result.started_at.isoformat())}")
        lines.append(f"- Finished: {_escape_inline(result.finished_at.isoformat())}")
        lines.append(f"- Duration: {_format_seconds(result.duration_seconds)}")
        lines.append(f"- Status: **{result.status.value}**")
        lines.append(
            f"- Metric: {_escape_inline(result.metric_user)} "
            f"({_escape_inline(result.metric_direction.value)})"
        )
        lines.append(
            f"- Validation: {_escape_inline(result.resolved_validation)}"
            f" (folds: {result.resolved_n_folds})"
        )
        if result.resolved_validation == "train_test_split":
            test_size = result.resolved_test_size
            rendered = (
                f"{test_size} rows"
                if isinstance(test_size, int)
                else f"{test_size:.2f} of the data"
            )
            lines.append(f"- Hold-out size: {rendered}")
        if result.random_state is not None:
            lines.append(f"- Random state: {result.random_state}")
        lines.append("")

    def _render_outcome(self, lines: list[str], result: ExperimentResult) -> None:
        lines.append("## Outcome")
        lines.append("")
        if result.status is RunStatus.completed:
            lines.append(f"Winner: **{_escape_inline(result.winner)}**")
        elif result.winner:
            # A run may fail after a winner was already determined; the
            # winner is still reported, together with the failure reason.
            lines.append(
                f"Winner (determined before the failure): "
                f"**{_escape_inline(result.winner)}**"
            )
        else:
            lines.append("No winner (the experiment failed).")
        if result.winner_score is not None:
            lines.append(
                f"- Winner score: {_format_score(result.winner_score)} "
                f"({_escape_inline(result.metric_user)} in user semantics — "
                f"{_escape_inline(result.metric_direction.value)})"
            )
        if result.winner_params:
            lines.append(f"- Winner params: {_format_params(result.winner_params)}")
        if result.additional_metrics:
            rendered = ", ".join(
                f"{_escape_inline(name)}: {_format_score(value)}"
                for name, value in result.additional_metrics.items()
            )
            lines.append(f"- Additional metrics: {rendered}")
        lines.append("")

    def _render_honest_scores(self, lines: list[str], result: ExperimentResult) -> None:
        honest: list[tuple[str, float | None, str]] = [
            (
                "Hold-out score",
                result.holdout_score,
                "honest out-of-sample estimate on the reserved outer hold-out",
            ),
            (
                "BBC score",
                result.bbc_score,
                "bias-corrected (BBC-CV) estimate of the selection procedure",
            ),
            (
                "OOF score",
                result.oof_score,
                "pooled out-of-fold predictions (optimistic, not independent)",
            ),
        ]
        if all(score is None for _, score, _ in honest):
            return
        lines.append("### Honest score estimates")
        lines.append("")
        for name, score, semantics in honest:
            if score is None:
                continue
            lines.append(f"- {name}: **{_format_score(score)}** — {semantics}")
        if result.winner_score is not None:
            lines.append(
                "- Reference — winner (selection) score: "
                f"{_format_score(result.winner_score)} (optimistic)"
            )
        lines.append("")

    def _render_leaderboard(self, lines: list[str], result: ExperimentResult) -> None:
        lines.append("## Leaderboard")
        lines.append("")
        leaderboard = build_leaderboard(result)
        if not leaderboard:
            lines.append(_EMPTY_LEADERBOARD_ROW)
            lines.append("")
            return
        lines.append("| Rank | Algorithm | Score | Trials | Phase |")
        lines.append("|-----:|:----------|------:|-------:|:------|")
        for entry in leaderboard:
            lines.append(
                f"| {entry.rank} "
                f"| {_escape_cell(entry.algorithm)} "
                f"| {_format_score(entry.score)} "
                f"| {entry.trials} "
                f"| {_escape_cell(entry.phase)} |"
            )
        lines.append("")

    def _render_ensemble(self, lines: list[str], result: ExperimentResult) -> None:
        """Render the experimental ensemble section (when present).

        The section shows the members table (rank, algorithm, validation
        score, weight), the effective ensemble size next to the requested
        one (``min(k, n)`` with a warning note when they differ) and the
        ensemble artifact path; the section is skipped when the result
        carries no ensemble (the feature is off or the step failed).
        """
        section = result.ensemble
        if not section:
            return
        members = section.get("members", [])
        if not members:
            # A section without members (unreachable via the engine: the
            # builder always assembles a non-empty list, and the result
            # validator checks JSON safety) renders nothing — no dangling
            # heading.
            return
        lines.append("## Ensemble (experimental)")
        lines.append("")
        weight_method = section.get("weight_method")
        lines.append(
            f"Weighted average of the top-{section.get('k_used', len(members))} "
            f"leaderboard models "
            f"(strategy: {_escape_inline(str(section.get('strategy', 'weighted_average')))}"
            + (
                f", weights: {_escape_inline(str(weight_method))}"
                if weight_method
                else ""
            )
            + ")."
        )
        k_requested = section.get("k_requested")
        k_used = section.get("k_used")
        if (
            isinstance(k_requested, int)
            and isinstance(k_used, int)
            and k_requested != k_used
        ):
            lines.append(
                f"- Requested size k={k_requested} exceeds the available "
                f"candidates: min(k, n) = {k_used} member(s) used."
            )
        lines.append("")
        lines.append("| Rank | Algorithm | Score | Weight |")
        lines.append("|-----:|:----------|------:|-------:|")
        for member in members:
            if not isinstance(member, dict):
                continue
            rank = member.get("rank")
            lines.append(
                f"| {_escape_cell(rank)} "
                f"| {_escape_cell(member.get('algorithm'))} "
                f"| {_format_score(member.get('score'))} "
                f"| {_format_score(member.get('weight'))} |"
            )
        artifact_path = section.get("artifact_path")
        if artifact_path:
            lines.append("")
            lines.append(
                f"Ensemble artifact: {_escape_code_span(_escape_cell(artifact_path))}"
            )
        lines.append("")

    def _render_explanations(self, lines: list[str], result: ExperimentResult) -> None:
        """Render the stored explain() payload section (when present).

        The section renders the JSON-safe payload of the last
        ``explain()`` call (``ExperimentResult.explanations``): the
        method, the baseline (the metric score for the permutation
        importance, the mean prediction for SHAP) and the importance
        table (top rows, the rest summarized). It is skipped when the
        run never invoked explain — no dangling heading.
        """
        payload = result.explanations
        if not payload:
            return
        importances = payload.get("importances") or []
        if not importances:
            return
        method = payload.get("method", "unknown")
        lines.append(f"## Feature importance ({_escape_inline(str(method))})")
        lines.append("")
        baseline: str | None = None
        if payload.get("baseline_score") is not None:
            baseline = (
                f"{payload['baseline_score']:.3f} ({payload.get('metric_user', '')} "
                "in user semantics — positive RMSE/MAE, plain R²)"
            )
        elif payload.get("baseline_prediction") is not None:
            baseline = f"{payload['baseline_prediction']:.3f} (mean prediction)"
        if baseline is not None:
            lines.append(f"Baseline: **{_escape_inline(baseline)}**.")
            lines.append("")
        max_rows = 20
        lines.append("| # | Feature | Importance mean | Importance std |")
        lines.append("|--:|:--------|----------------:|---------------:|")
        for position, item in enumerate(importances[:max_rows], start=1):
            if not isinstance(item, dict):
                continue
            std = item.get("importance_std")
            std_cell = _EMPTY_CELL if std is None else f"{float(std):.6f}"
            lines.append(
                f"| {position} "
                f"| {_escape_cell(item.get('feature'))} "
                f"| {_format_score(item.get('importance_mean'))} "
                f"| {std_cell} |"
            )
        if len(importances) > max_rows:
            lines.append(f"_… and {len(importances) - max_rows} more features._")
        lines.append("")

    def _render_run_details(self, lines: list[str], result: ExperimentResult) -> None:
        lines.append("## Run details")
        lines.append("")
        if not result.runs:
            lines.append("_No algorithm records (an empty or failed run)._")
            lines.append("")
            return
        for record in result.runs:
            self._render_one_run(lines, record)

    def _render_one_run(self, lines: list[str], record: AlgorithmRun) -> None:
        title = _escape_inline(record.algorithm)
        heading = (
            f"### {title} — phase {_escape_inline(record.phase_name)} "
            f"({record.phase_action.value})"
        )
        lines.append(heading)
        lines.append("")
        lines.append(f"- Status: **{record.status.value}**")
        if record.status is RunStepStatus.completed:
            lines.append(f"- Score (user semantics): {_format_score(record.score)}")
            lines.append(f"- Best params: {_format_params(record.params)}")
            lines.append(
                f"- Trials: evaluated {record.n_trials_evaluated}"
                f" (completed {record.n_trials_completed},"
                f" pruned {record.n_trials_pruned},"
                f" failed {record.n_trials_failed})"
                + (
                    f" of {record.n_trials_requested} requested"
                    if record.n_trials_requested is not None
                    else ""
                )
            )
        else:
            lines.append(f"- Reason: {_escape_inline(record.error or _EMPTY_CELL)}")
        if record.duration_seconds is not None:
            lines.append(f"- Duration: {_format_seconds(record.duration_seconds)}")
        lines.append("")

    def _render_failure(self, lines: list[str], result: ExperimentResult) -> None:
        if result.error:
            lines.append("## Failure")
            lines.append("")
            lines.append(_escape_inline(result.error))
            lines.append("")

    def _render_config(self, lines: list[str], result: ExperimentResult) -> None:
        # The config snapshot is validated JSON-safe by the model; no
        # `default=str` fallback — a non-serializable value must surface
        # as an error, not be silently masked as a string.
        lines.append("<details>")
        lines.append("<summary>Configuration snapshot</summary>")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(result.config, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")
