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
import math
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

#: The placeholder of an absent or non-finite multi-split statistic.
_NA = "н/д"


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


def _format_number(value: Any) -> str:
    """Render a number, or ``н/д`` for a missing/non-finite value.

    Report sections are JSON-safe mappings assembled by the engine; a
    circuit that could not be computed is serialized as ``None`` and a
    legacy section may still carry a raw NaN. Neither may crash the report,
    so both render as the explicit not-available marker.
    """
    if value is None:
        return _NA
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _NA
    if not math.isfinite(numeric):
        return _NA
    return f"{numeric:.3f}"


def _repeat_count(result: ExperimentResult) -> int:
    """The effective number of repeats recorded in the result (>= 1)."""
    if result.n_repeats is None:
        return 1
    return max(int(result.n_repeats), 1)


def _aggregation_label(result: ExperimentResult) -> str:
    """The per-repeat aggregation policy from the config snapshot."""
    config = result.config if isinstance(result.config, dict) else {}
    general = config.get("general")
    if isinstance(general, dict):
        aggregation = general.get("split_aggregation")
        if aggregation in ("median", "mean"):
            return str(aggregation)
    return "median"


def _repeat_note(result: ExperimentResult) -> str | None:
    """Render the repeat count/aggregation note, or ``None`` for one split."""
    n_repeats = _repeat_count(result)
    if n_repeats <= 1:
        return None
    return f"{_aggregation_label(result)} of {n_repeats} repeats"


def _format_spread(value: Any, std: Any, result: ExperimentResult) -> str:
    """Render an aggregate with its spread for a repeated-validation run.

    A single split keeps the classic scalar rendering exactly. For several
    repeats the value is shown as ``aggregate ± std``; an absent/non-finite
    spread renders as ``н/д`` instead of failing the whole report.
    """
    if _repeat_count(result) <= 1:
        return _format_score(value if value is None else float(value))
    return f"{_format_number(value)} ± {_format_number(std)}"


def _format_value_with_std(value: Any, std: Any) -> str:
    """Render a section metric with an optional spread."""
    if std is None:
        return _format_number(value)
    return f"{_format_number(value)} ± {_format_number(std)}"


def _pick(mapping: dict[str, Any], *keys: str) -> Any:
    """Return the first present value among the alternative section keys."""
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _has_repeat_markers(value: Any) -> bool:
    """True when a report section carries multi-split aggregate markers.

    A section with a per-repeat spread (``*_std``), an explicit median key or
    a nested repeat count is a multi-split section even when the result-level
    ``n_repeats`` was not recorded (a legacy/serialized producer). Such a
    section is rendered instead of being silently dropped.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key in ("n_repeats", "n_valid_repeats"):
                try:
                    if int(item) > 1:
                        return True
                except (TypeError, ValueError, OverflowError):
                    pass
            elif (
                isinstance(key, str)
                and key.endswith(("_std", "_median", "_mean"))
                and item is not None
            ):
                return True
            if _has_repeat_markers(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_has_repeat_markers(item) for item in value)
    return False


def _has_aggregated_audit(value: Any) -> bool:
    """True when a gate section actually records aggregated audit values.

    The section-level ``n_repeats`` alone does not prove aggregation: M4
    records it on every section, including the single-split one. A candidate
    is aggregated when its ``n_valid_repeats`` is greater than one or it
    carries an explicit per-repeat spread/median key. ``n_repeats`` is the
    trainer total and may stay above one when only a single valid repeat was
    audited, so it must not be used as the aggregation marker. Only an
    actually aggregated audit lets the report state that the circuits use
    the concatenated OOF / median gap.
    """
    if isinstance(value, dict):
        audit = value.get("audit")
        if isinstance(audit, dict):
            for item in audit.values():
                if not isinstance(item, dict):
                    continue
                try:
                    if int(item.get("n_valid_repeats", 1)) > 1:
                        return True
                except (TypeError, ValueError, OverflowError):
                    pass
                for key, field_value in item.items():
                    if (
                        isinstance(key, str)
                        and key.endswith(("_std", "_median", "_mean"))
                        and field_value is not None
                    ):
                        return True
        return any(_has_aggregated_audit(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_aggregated_audit(item) for item in value)
    return False


def _has_rmse_spread(value: dict[str, Any]) -> bool:
    """True when a gate section records an ``RMSE_oof`` per-repeat spread."""
    if value.get("winner_rmse_oof_std") is not None:
        return True
    rmse_std = value.get("rmse_oof_std")
    return isinstance(rmse_std, dict) and any(
        spread is not None for spread in rmse_std.values()
    )


def _format_count(value: Any) -> str:
    """Render an integer count, or ``н/д`` for a missing value."""
    if value is None:
        return _NA
    try:
        return str(int(value))
    except (TypeError, ValueError, OverflowError):
        return _NA


def _text_items(value: Any) -> list[str]:
    """Normalize a report list-or-string value to a list of plain strings.

    A legacy/hand-written section may carry ``reasons`` as a single string
    instead of a list; iterating that string would split it into characters.
    Both shapes are rendered correctly.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return [str(value)]


def _format_gap(value: Any, std: Any, infinite: Any) -> str:
    """Render the generalization gap, honoring the explicit infinity flag.

    The engine serializes a non-finite gap as ``null`` plus
    ``generalization_gap_infinite: true``. The flag is authoritative: the
    report must show an infinite gap instead of the generic ``н/д`` marker.
    """
    if infinite:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = None
        if numeric is None or not math.isfinite(numeric):
            return "∞ (infinite)"
    return _format_value_with_std(value, std)


def _format_seconds(seconds: Any) -> str:
    """Render one duration cell in seconds (two decimal places)."""
    if seconds is None:
        return _EMPTY_CELL
    try:
        numeric = float(seconds)
    except (TypeError, ValueError):
        return _NA
    if not math.isfinite(numeric):
        return _NA
    return f"{numeric:.2f}s"


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
        self._render_sanity_gate(lines, result)
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
            repeat_note = _repeat_note(result)
            suffix = f"; {repeat_note}" if repeat_note else ""
            lines.append(
                f"- Winner score: "
                f"{_format_spread(result.winner_score, result.score_std, result)} "
                f"({_escape_inline(result.metric_user)} in user semantics — "
                f"{_escape_inline(result.metric_direction.value)}{suffix})"
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
        # The last flag tells whether the estimate itself is repeated. The
        # hold-out and BBC estimates stay single-split even in a repeated run,
        # so they keep the classic scalar rendering (no "± н/д").
        honest: list[tuple[str, float | None, float | None, str, bool]] = [
            (
                "Hold-out score",
                result.holdout_score,
                None,
                "honest out-of-sample estimate on the reserved outer hold-out",
                False,
            ),
            (
                "BBC score",
                result.bbc_score,
                None,
                "bias-corrected (BBC-CV) estimate of the selection procedure",
                False,
            ),
            (
                "OOF score",
                result.oof_score,
                result.oof_score_std,
                "pooled out-of-fold predictions (optimistic, not independent)",
                True,
            ),
        ]
        if all(score is None for _, score, _, _, _ in honest):
            return
        repeat_note = _repeat_note(result)
        heading = "### Honest score estimates"
        if repeat_note is not None:
            heading += f" ({_escape_inline(repeat_note)})"
        lines.append(heading)
        lines.append("")
        for name, score, std, semantics, repeated in honest:
            if score is None:
                continue
            rendered = (
                _format_spread(score, std, result) if repeated else _format_score(score)
            )
            lines.append(f"- {name}: **{rendered}** — {semantics}")
        if result.winner_score is not None:
            lines.append(
                "- Reference — winner (selection) score: "
                f"{_format_spread(result.winner_score, result.score_std, result)} "
                "(optimistic)"
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

    def _render_sanity_gate(self, lines: list[str], result: ExperimentResult) -> None:
        """Render the Sanity Gate report sections of a repeated run.

        The new table is rendered only when the section itself records
        multi-split data (a candidate ``n_repeats > 1``, an explicit spread or
        a median/mean key); a classic one-repeat section keeps the previous
        report unchanged, regardless of the result-level repeat count. When
        the audit carries aggregates, the caption says so and uses their
        medians; otherwise the circuits are explicitly labelled as values of
        one representative repeat. A section-level ``RMSE_oof`` spread is
        described separately because it does not aggregate the circuits. A
        missing/non-finite circuit statistic (a disabled or non-computable
        circuit) renders as ``н/д`` and never fails the report.

        The renderer is a pure function of the result: it only reads the
        JSON-safe ``sanity_gate`` mapping and never reaches into the engine.
        """
        sections = result.sanity_gate
        if not sections or not _has_repeat_markers(sections):
            return
        lines.append("## Sanity Gate")
        lines.append("")
        for phase_name, raw_section in sections.items():
            if not isinstance(raw_section, dict):
                continue
            if _has_aggregated_audit(raw_section):
                lines.append(
                    "Circuit values are aggregated over the repeated splits: "
                    "the diversity/unique ratios and the generalization gap "
                    "are medians of the per-repeat values. A spread is shown "
                    "when the audit recorded one."
                )
            else:
                lines.append(
                    "Circuit values are taken from one representative OOF "
                    "repeat and are not aggregated over repeats."
                )
            if _has_rmse_spread(raw_section):
                lines.append(
                    "The winner's `RMSE_oof` is aggregated over the repeats; "
                    "its spread is shown where recorded."
                )
            lines.append("")
            lines.append(f"### Phase {_escape_inline(phase_name)}")
            lines.append("")
            pool = raw_section.get("pool")
            if isinstance(pool, (list, tuple)):
                rendered_pool = ", ".join(_escape_inline(name) for name in pool)
                lines.append(f"- Pool: {rendered_pool}")
            winner_rmse = raw_section.get("winner_rmse_oof")
            winner_std = raw_section.get("winner_rmse_oof_std")
            if winner_rmse is not None or winner_std is not None:
                lines.append(
                    "- Winner RMSE_oof: "
                    f"**{_format_value_with_std(winner_rmse, winner_std)}**"
                )
            if "fallback_used" in raw_section:
                fallback = bool(raw_section.get("fallback_used"))
                lines.append(f"- Fallback used: {'yes' if fallback else 'no'}")
            overhead = raw_section.get("overhead_seconds")
            if overhead is not None:
                lines.append(f"- Overhead: {_format_seconds(overhead)}")
            section_aggregates = (
                (
                    "Diversity ratio",
                    _pick(
                        raw_section,
                        "diversity_ratio",
                        "diversity_ratio_median",
                        "diversity_ratio_mean",
                    ),
                    _pick(raw_section, "diversity_ratio_std", "diversity_std"),
                ),
                (
                    "Unique ratio",
                    _pick(
                        raw_section,
                        "unique_ratio",
                        "unique_ratio_median",
                        "unique_ratio_mean",
                    ),
                    _pick(raw_section, "unique_ratio_std", "unique_std"),
                ),
                (
                    "Generalization gap (median)",
                    _pick(
                        raw_section,
                        "generalization_gap",
                        "generalization_gap_median",
                        "gap_median",
                    ),
                    _pick(raw_section, "generalization_gap_std", "gap_std"),
                ),
            )
            for aggregate_name, aggregate_value, aggregate_std in section_aggregates:
                if aggregate_value is None and aggregate_std is None:
                    continue
                if aggregate_name.startswith("Generalization gap"):
                    rendered_aggregate = _format_gap(
                        aggregate_value,
                        aggregate_std,
                        raw_section.get("generalization_gap_infinite"),
                    )
                else:
                    rendered_aggregate = _format_value_with_std(
                        aggregate_value, aggregate_std
                    )
                lines.append(f"- {aggregate_name}: **{rendered_aggregate}**")
            audit = raw_section.get("audit")
            if isinstance(audit, dict) and audit:
                self._render_gate_audit(lines, audit)
            lines.append("")

    def _render_gate_audit(self, lines: list[str], audit: dict[str, Any]) -> None:
        """Render the per-candidate audit table of one gate section."""
        lines.append("")
        lines.append(
            "| Candidate | Valid | Diversity ratio | Unique ratio "
            "| Dead features | Generalization gap | Reasons |"
        )
        lines.append(
            "|:----------|:------|----------------:|------------:"
            "|--------------:|-------------------:|:--------|"
        )
        for candidate, raw_audit in audit.items():
            if not isinstance(raw_audit, dict):
                continue
            valid = raw_audit.get("is_valid")
            valid_cell = "yes" if valid else "no" if valid is False else _NA
            diversity = _pick(
                raw_audit,
                "diversity_ratio",
                "diversity_ratio_median",
                "diversity_ratio_mean",
            )
            diversity_std = _pick(
                raw_audit,
                "diversity_ratio_std",
                "diversity_std",
            )
            unique = _pick(
                raw_audit,
                "unique_ratio",
                "unique_ratio_median",
                "unique_ratio_mean",
            )
            unique_std = _pick(
                raw_audit,
                "unique_ratio_std",
                "unique_std",
            )
            gap = _pick(
                raw_audit,
                "generalization_gap",
                "generalization_gap_median",
                "gap_median",
            )
            gap_std = _pick(
                raw_audit,
                "generalization_gap_std",
                "gap_std",
            )
            gap_cell = _format_gap(
                gap,
                gap_std,
                raw_audit.get("generalization_gap_infinite"),
            )
            dead = _pick(raw_audit, "dead_features_count", "dead_features")
            reason_parts = _text_items(raw_audit.get("reasons")) + _text_items(
                raw_audit.get("soft_warnings")
            )
            reason_text = "; ".join(dict.fromkeys(reason_parts)) or _EMPTY_CELL
            lines.append(
                f"| {_escape_cell(candidate)} "
                f"| {valid_cell} "
                f"| {_format_value_with_std(diversity, diversity_std)} "
                f"| {_format_value_with_std(unique, unique_std)} "
                f"| {_format_count(dead)} "
                f"| {gap_cell} "
                f"| {_escape_cell(reason_text)} |"
            )

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
        comparison = section.get("comparison")
        if isinstance(comparison, dict):
            winner_rmse = comparison.get("winner_rmse")
            ensemble_rmse = comparison.get("ensemble_rmse")
            if isinstance(winner_rmse, (int, float)) and isinstance(
                ensemble_rmse, (int, float)
            ):
                better = comparison.get("better")
                verdict = (
                    "tie" if better == "tie" else f"{_escape_inline(better)} is better"
                )
                metric = _escape_inline(str(comparison.get("metric", "rmse")))
                n_repeats = comparison.get("n_repeats")
                n_repeats_used = comparison.get("n_repeats_used", n_repeats)
                policy = _escape_inline(
                    str(comparison.get("split_aggregation", "median"))
                )
                lines.append("")
                lines.append("**Winner vs ensemble (aggregated OOF).**")
                lines.append(
                    f"- {metric.upper()}: winner "
                    f"{_format_score(float(winner_rmse))}, ensemble "
                    f"{_format_score(float(ensemble_rmse))} — {verdict}."
                )
                lines.append(
                    f"- Repeats: {n_repeats_used}/{n_repeats}; aggregation: {policy}."
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
