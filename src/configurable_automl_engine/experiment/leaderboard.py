"""Leaderboard: the algorithm ranking table of an experiment (the T6
component).

``build_leaderboard`` ranks the algorithms of an
:class:`~configurable_automl_engine.experiment.result.ExperimentResult`
by their user-semantic scores (``AlgorithmRun.score``: positive RMSE/MAE,
plain R² — the raw optimizer values, including the neg_-metric inversion,
never leave the result model).

Documented contract:

* ranking direction follows ``ExperimentResult.metric_direction``: for
  ``minimize`` metrics (RMSE, MAE and every other error metric in user
  semantics) the smallest score takes rank 1; for ``maximize`` metrics
  (R² and similar) the largest score takes rank 1. The scores are sorted
  by their absolute order, so an epsilon-tolerant comparison is not
  needed and exact float equality is the tie condition;
* ties are resolved deterministically by the configuration order of the
  algorithms (the same tie-break as ``select_winner`` and the
  finalists-pool ordering): on equal scores the earlier-configured
  algorithm takes the better rank. The configuration order is read from
  the ``config`` snapshot (``config['general']['phases']`` → the
  ``algorithms`` section order, defensively: a malformed hand-built
  snapshot never crashes the ranking). When the order cannot be recovered
  from the snapshot, the fallback key sorts the tied algorithms by name
  (the names are unique keys of the algorithms configuration, so the
  fallback order is itself deterministic);
* the ``Phase`` column names the phase in which the algorithm showed its
  best result: the best record of the algorithm is selected with the same
  ranking direction and the same tie-break as the table itself;
* the ``Trials`` column counts the phase's own COMPLETED trials
  (``AlgorithmRun.n_trials_evaluated``, symmetric to
  ``ExperimentResult.n_trials_evaluated``): the trial history transferred
  between phases (``previous_trials``) is never counted — each phase
  rebuilds its trial budget. The value is the one of the record shown in
  the ``Phase`` column (the best record);
* records that may not appear in a leaderboard are filtered out before
  ranking: a ``completed`` record without a valid score (the score fields
  of the model already reject ``None``/non-finite/sentinel values, but a
  hand-built result may still carry ``score=None`` on a completed record),
  and every ``failed`` / ``disqualified`` record (their exclusion reason
  is preserved in the run history, not in the leaderboard);
* an empty leaderboard (a failed experiment, no valid records) is not an
  error: :func:`build_leaderboard` returns an empty leaderboard,
  :meth:`Leaderboard.to_table` renders a placeholder message and
  :meth:`Leaderboard.to_summary` reports the failure explicitly.

``to_summary`` contract: the statusless rendering (the default) is the
list plus the ``Winner: …`` line; with ``status=True`` the first line of
the block is the experiment status (``Experiment completed`` /
``Experiment failed``), matching the UX-цель example of the epic #37.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from configurable_automl_engine.common.definitions import (
    MetricDirection,
    RunPhaseAction,
    RunStatus,
    RunStepStatus,
)
from configurable_automl_engine.experiment.result import (
    AlgorithmRun,
    ExperimentResult,
)

__all__ = [
    "COMPLETED_SUMMARY_MESSAGE",
    "EMPTY_LEADERBOARD_MESSAGE",
    "FAILED_SUMMARY_MESSAGE",
    "SUMMARY_HEADER_INDENT",
    "Leaderboard",
    "LeaderboardEntry",
    "build_leaderboard",
    "visual_width",
]

#: The placeholder rendered by ``to_table`` when no entry survived the
#: filtering (a failed experiment, no completed records with valid scores).
EMPTY_LEADERBOARD_MESSAGE = "<no ranked algorithms>"

#: The status header of a completed experiment (the first line of the
#: UX summary, following the epic #37 example).
COMPLETED_SUMMARY_MESSAGE = "Experiment completed"

#: The summary line of a failed experiment (no winner to congratulate).
FAILED_SUMMARY_MESSAGE = "Experiment failed"

#: The indent of the ``Winner: …`` line: it hangs under the list so the
#: winner stands out without taking a rank number (the epic #37 layout).
SUMMARY_HEADER_INDENT = "   "

#: The columns of the ranking table, in the render order.
_COLUMNS: tuple[str, ...] = ("Rank", "Algorithm", "Score", "Trials", "Phase")

#: The default cell truncation limit: longer names and parameter strings
#: are cut with a single-character ellipsis ("…").
DEFAULT_MAX_CELL_WIDTH = 40

_ELLIPSIS = "…"

# How much of the string each side keeps when truncating: the head is
# slightly larger, so the informative prefix survives.
_HEAD_SHARE = 0.7


def _normalize(text: str) -> str:
    """Compose one cell string into its NFC form.

    A combining sequence (e.g. a base letter plus a combining accent)
    collapses into a single precomposed character, so combining marks do
    not desynchronize the visual column widths of a monospace grid.
    """
    return unicodedata.normalize("NFC", text)


def visual_width(text: str) -> int:
    """The terminal width of ``text`` in monospace cells.

    East Asian Wide and Fullwidth characters (CJK ideographs, katakana,
    fullwidth forms) occupy two cells; every other code point counts as
    one. Measured after NFC composition.
    """
    composed = _normalize(text)
    return sum(
        2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in composed
    )


def _truncate(text: str, max_width: int) -> str:
    """Cut ``text`` to at most ``max_width`` visual cells with an ellipsis.

    The head (≈70% of the budget) and the tail (≈30%) are kept so both
    the prefix and the suffix of a long value stay visible. A zero or
    negative budget collapses the cell to the ellipsis alone; a string
    that already fits is returned unchanged.
    """
    if max_width <= 0:
        return _ELLIPSIS
    if visual_width(text) <= max_width:
        return text
    if max_width == 1:
        return _ELLIPSIS
    budget = max_width - 1  # one cell for the ellipsis
    head_budget = max(budget * _HEAD_SHARE, 1)
    head: list[str] = []
    head_width = 0
    for ch in _normalize(text):
        ch_width = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if head_width + ch_width > head_budget:
            break
        head.append(ch)
        head_width += ch_width
    tail_budget = budget - head_width
    tail_chars: list[str] = []
    tail_width = 0
    for ch in reversed(_normalize(text)):
        ch_width = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if tail_width + ch_width > tail_budget:
            break
        tail_chars.append(ch)
        tail_width += ch_width
    return "".join(head) + _ELLIPSIS + "".join(reversed(tail_chars))


def _format_score(score: float) -> str:
    """Render one score cell: three decimal places, like the epic example."""
    return f"{score:.3f}"


def _register_algorithm_names(container: Any, order: dict[str, int]) -> None:
    """Register the algorithm names of one config container into ``order``.

    The ``algorithms`` section of a phase (or of the top-level config) may
    be a mapping (name → params) or a list of names / ``{name: …}`` dicts
    (defensive: the schema allows the mapping form only, but a hand-built
    snapshot must not crash the ranking).
    """
    if isinstance(container, dict):
        for name in container:
            order.setdefault(str(name), len(order))
    elif isinstance(container, list):
        for entry in container:
            if isinstance(entry, dict) and "name" in entry:
                order.setdefault(str(entry["name"]), len(order))
            elif isinstance(entry, str):
                order.setdefault(entry, len(order))


def _configuration_order(result: ExperimentResult) -> dict[str, int]:
    """Recover the configuration order of the algorithms from the result.

    The ``config`` snapshot holds the ``general.phases`` section; the
    ``algorithms`` mapping inside each phase lists the algorithms in their
    configuration order. When the phases section does not carry an
    ``algorithms`` mapping (a hand-built snapshot), the top-level
    ``config['algorithms']`` is used instead. Any algorithm missing from
    the recovered order is appended after the known ones by the ranking
    fallback, so a hand-built result never crashes the ranking.
    """
    order: dict[str, int] = {}
    config = result.config
    # Defensive: a hand-built snapshot may hold a non-mapping 'general'
    # section (the engine's model_dump always produces a mapping); the
    # declared contract is that such a snapshot never crashes the ranking.
    general = config.get("general")
    phases = general.get("phases") if isinstance(general, dict) else None
    if isinstance(phases, list):
        for phase in phases:
            if isinstance(phase, dict) and "algorithms" in phase:
                _register_algorithm_names(phase["algorithms"], order)
    if not order:
        _register_algorithm_names(config.get("algorithms"), order)
    return order


def _sort_key(
    entry_score: float,
    algorithm: str,
    *,
    direction: MetricDirection,
    config_order: dict[str, int],
) -> tuple[float, int, str]:
    """The total ranking key of one record.

    The score is negated for ``maximize`` metrics so that an ascending
    sort puts the best score first in both directions; the configuration
    index breaks the score ties deterministically (algorithms missing
    from the recovered configuration order share the fallback index, and
    the name sorts them alphabetically).
    """
    directional_score = (
        -entry_score if direction is MetricDirection.maximize else entry_score
    )
    order_index = config_order.get(algorithm, len(config_order))
    return (directional_score, order_index, algorithm)


def _records_in_configuration_order(
    result: ExperimentResult, config_order: dict[str, int]
) -> list[AlgorithmRun]:
    """The records of the result in their stable ranking order.

    Sorting before ranking implements the documented tie-break without
    relying on ``sorted`` stability across the per-algorithm grouping.
    """
    return sorted(
        result.runs,
        key=lambda r: (
            config_order.get(r.algorithm, len(config_order)),
            r.algorithm,
        ),
    )


@dataclass(frozen=True)
class LeaderboardEntry:
    """One row of the ranking table.

    Attributes:
        rank: The 1-based rank of the algorithm in the leaderboard.
        algorithm: The algorithm name (as configured in ``algorithms``).
        score: The score in user semantics (positive RMSE/MAE, plain R²).
        trials: The number of the phase's own COMPLETED trials
            (``AlgorithmRun.n_trials_evaluated``); transferred history
            (``previous_trials``) is never counted.
        phase: The name of the phase that produced the best result.
        phase_action: The action of that phase (``all_algorithms``,
            ``refine`` or ``select``).
        params: The best found hyperparameters of that phase.
        validation_strategy: The resolved validation strategy of the
            experiment (e.g. ``k_fold``).
        n_folds: The resolved number of folds (for ``k_fold``).
        training_time_seconds: The wall-clock duration of the best
            record, in seconds (``None`` when unknown).
        n_trials_failed: The number of failed trials of the best record.
    """

    rank: int
    algorithm: str
    score: float
    trials: int
    phase: str
    phase_action: RunPhaseAction
    params: dict[str, Any]
    validation_strategy: str
    n_folds: int
    training_time_seconds: float | None
    n_trials_failed: int


@dataclass(frozen=True)
class Leaderboard:
    """The ranked leaderboard of one experiment.

    A pure view over an :class:`ExperimentResult`: built by
    :func:`build_leaderboard`, consumed by the reporters (the structured
    access) and by the console output (:meth:`to_table`,
    :meth:`to_summary`).
    """

    metric_user: str
    metric_direction: MetricDirection
    entries: tuple[LeaderboardEntry, ...] = field(default_factory=tuple)
    #: The status of the experiment the leaderboard was built from
    #: (``None`` for a hand-built ``Leaderboard`` — the ``status=True``
    #: rendering then treats the run as not completed).
    result_status: RunStatus | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[LeaderboardEntry]:
        return iter(self.entries)

    @property
    def winner(self) -> LeaderboardEntry | None:
        """The rank-1 entry, or ``None`` for an empty leaderboard."""
        return self.entries[0] if self.entries else None

    def to_table(self, *, max_cell_width: int = DEFAULT_MAX_CELL_WIDTH) -> str:
        """Render the ranking table as aligned monospace text.

        Args:
            max_cell_width: The truncation budget of a text cell, in
                visual cells (long algorithm names and phase names are
                cut with an ellipsis).

        Returns:
            str: The ``Rank | Algorithm | Score | Trials | Phase`` grid;
                for an empty leaderboard a single placeholder line.
        """
        if not self.entries:
            return EMPTY_LEADERBOARD_MESSAGE
        rows: list[list[str]] = []
        for entry in self.entries:
            rows.append(
                [
                    str(entry.rank),
                    _truncate(entry.algorithm, max_cell_width),
                    _format_score(entry.score),
                    str(entry.trials),
                    _truncate(entry.phase, max_cell_width),
                ]
            )
        widths = [
            max(visual_width(header_col), *(visual_width(row[i]) for row in rows))
            for i, header_col in enumerate(_COLUMNS)
        ]
        lines = [
            " | ".join(
                header_col + " " * (widths[i] - visual_width(header_col))
                for i, header_col in enumerate(_COLUMNS)
            ),
            " | ".join("-" * width for width in widths),
        ]
        lines.extend(
            " | ".join(
                row[i] + " " * (widths[i] - visual_width(row[i]))
                for i in range(len(_COLUMNS))
            )
            for row in rows
        )
        return "\n".join(lines)

    def to_summary(self, *, status: bool = False) -> str:
        """Render the UX summary of the experiment outcome.

        The block lists every ranked entry as ``N. Name  Score`` (the
        names padded to a common width so the scores align) and closes
        with the ``Winner: …`` line; a failed experiment with no ranked
        entries reports the failure instead of a winner.

        Args:
            status: Prepend the experiment status line to the block —
                ``Experiment completed`` for a completed run or
                ``Experiment failed`` for a failed one (the failed
                status keeps the ranking below it: some algorithms may
                have valid records even when the run as a whole failed).
                The statusless rendering (the default) keeps the
                historical list + Winner layout unchanged; the status
                line matches the UX-цель block of the epic #37 example.
                A statusless summary of an empty leaderboard is the
                failure placeholder (unchanged); a failed status never
                prefixes an empty summary (the placeholder already says
                ``Experiment failed``).

        Returns:
            str: The rendered summary block; the exact shape depends on
                ``status`` and on whether entries exist:

                * ``status=False``, entries exist: the ``N. Name Score``
                  list plus the final ``   Winner: …`` line (no status
                  header — the documented deviation from the epic #37
                  example);
                * ``status=True``, entries exist: the status line
                  (``Experiment completed`` / ``Experiment failed``),
                  then the list, then the Winner line;
                * no entries: the ``Experiment failed`` placeholder —
                  with or without ``status=True`` (the placeholder is
                  never doubled).
        """
        if not self.entries:
            return FAILED_SUMMARY_MESSAGE
        name_width = max(visual_width(e.algorithm) for e in self.entries)
        lines = [
            f"{idx}. {entry.algorithm}{' ' * (name_width - visual_width(entry.algorithm))}  "
            f"{_format_score(entry.score)}"
            for idx, entry in enumerate(self.entries, start=1)
        ]
        lines.append(f"{SUMMARY_HEADER_INDENT}Winner: {self.entries[0].algorithm}")
        if not status:
            return "\n".join(lines)
        header = (
            COMPLETED_SUMMARY_MESSAGE
            if self.result_status is RunStatus.completed
            else FAILED_SUMMARY_MESSAGE
        )
        return "\n".join([header, *lines])


def build_leaderboard(result: ExperimentResult) -> Leaderboard:
    """Build the ranking leaderboard of one experiment result.

    Only the records eligible for a leaderboard are ranked: the
    ``completed`` records with a valid (non-``None``) user-semantic score.
    Failed and disqualified records are excluded (their reason stays in
    the run history). The ranking direction follows the metric semantics
    recorded in the result (``minimize`` for RMSE/MAE, ``maximize`` for
    R²); ties are broken by the configuration order.

    Args:
        result: The experiment result to rank.

    Returns:
        Leaderboard: The ranked leaderboard (possibly empty — a failed
            experiment or a result without valid records is not an error).
    """
    config_order = _configuration_order(result)
    # The best record per algorithm: the eligible record with the best
    # user-semantic score; the first record of the algorithm in the
    # deterministic record order wins the ties (the same tie-break as the
    # table ranking).
    best: dict[str, AlgorithmRun] = {}
    for record in _records_in_configuration_order(result, config_order):
        if record.status is not RunStepStatus.completed:
            continue
        if record.score is None:
            continue
        current = best.get(record.algorithm)
        if current is None or _record_beats(record, current, result.metric_direction):
            best[record.algorithm] = record
    ranked = sorted(
        best.values(),
        key=lambda r: _sort_key(
            float(r.score),  # type: ignore[arg-type]
            r.algorithm,
            direction=result.metric_direction,
            config_order=config_order,
        ),
    )
    entries = tuple(
        LeaderboardEntry(
            rank=idx,
            algorithm=record.algorithm,
            score=float(record.score),  # type: ignore[arg-type]
            trials=int(record.n_trials_evaluated or 0),
            phase=record.phase_name,
            phase_action=RunPhaseAction(record.phase_action),
            params=dict(record.params),
            validation_strategy=result.resolved_validation,
            n_folds=result.resolved_n_folds,
            training_time_seconds=record.duration_seconds,
            n_trials_failed=int(record.n_trials_failed or 0),
        )
        for idx, record in enumerate(ranked, start=1)
    )
    return Leaderboard(
        metric_user=result.metric_user,
        metric_direction=result.metric_direction,
        entries=entries,
        result_status=result.status,
    )


def _record_beats(
    candidate: AlgorithmRun,
    incumbent: AlgorithmRun,
    direction: MetricDirection,
) -> bool:
    """Whether ``candidate``'s score is better than ``incumbent``'s.

    A tie keeps the incumbent (the earlier record in the deterministic
    order), so the best-record choice follows the same tie-break as the
    table ranking.
    """
    candidate_score = float(candidate.score)  # type: ignore[arg-type]
    incumbent_score = float(incumbent.score)  # type: ignore[arg-type]
    if direction is MetricDirection.minimize:
        return candidate_score < incumbent_score
    return candidate_score > incumbent_score
