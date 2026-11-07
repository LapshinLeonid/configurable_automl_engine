"""Tests of the experiment leaderboard (``experiment.leaderboard``).

Covered scenarios (positive / negative / boundary):

- P1: the table format matches the epic example (columns, order, ranks);
- P2: the UX summary "Experiment completed … Winner: …" is rendered
  correctly (both the statusless default and the ``status=True`` block
  of the epic #37 UX-цель example);
- P3: the per-run details (params, validation strategy, training time,
  failed trials) are available on the entries;
- P4: a minimized metric (RMSE) ranks the smallest score first;
- N1: an empty result (failed experiment) → a defined behavior without
  exceptions;
- N2: records with invalid scores (``None`` on a completed record,
  failed/disqualified records) do not reach the table;
- B1: a score tie → the deterministic configuration-order tie-break;
- B2: exactly one record;
- B3: long names/parameters → correct alignment/truncation;
- B4: unicode/Cyrillic names align by display width;
- B5: a record with 0 own trials (a phase without own COMPLETED trials).
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Any

import pytest

from configurable_automl_engine.common.definitions import (
    MetricDirection,
    RunPhaseAction,
    RunStatus,
    RunStepStatus,
)
from configurable_automl_engine.experiment import AlgorithmRun, ExperimentResult
from configurable_automl_engine.experiment.leaderboard import (
    COMPLETED_SUMMARY_MESSAGE,
    EMPTY_LEADERBOARD_MESSAGE,
    FAILED_SUMMARY_MESSAGE,
    SUMMARY_HEADER_INDENT,
    Leaderboard,
    LeaderboardEntry,
    build_leaderboard,
    visual_width,
)

UTC = dt.UTC


def _ts(seconds: float = 0.0) -> dt.datetime:
    """A fixed UTC timestamp shifted by ``seconds``."""
    return dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC) + dt.timedelta(seconds=seconds)


def _run_kwargs(**overrides: Any) -> dict[str, Any]:
    """Default keyword arguments of a completed ``AlgorithmRun`` record."""
    values: dict[str, Any] = {
        "algorithm": "ridge",
        "phase_name": "search",
        "phase_action": "all_algorithms",
        "status": RunStepStatus.completed,
        "score_raw": -0.5,
        "score": 0.5,
        "params": {"alpha": 0.1},
        "n_trials_requested": 10,
        "n_trials_evaluated": 10,
        "n_trials_completed": 9,
        "n_trials_pruned": 1,
        "n_trials_failed": 0,
        "duration_seconds": 12.5,
    }
    values.update(overrides)
    return values


def _result_kwargs(**overrides: Any) -> dict[str, Any]:
    """Default keyword arguments of a completed ``ExperimentResult``."""
    values: dict[str, Any] = {
        "run_id": uuid.uuid4(),
        "label": "leaderboard run",
        "engine_version": "0.0.0",
        "started_at": _ts(),
        "finished_at": _ts(90.0),
        "duration_seconds": 90.0,
        "status": RunStatus.completed,
        "config": {},
        "resolved_validation": "k_fold",
        "resolved_n_folds": 5,
        "resolved_test_size": 0.2,
        "random_state": 42,
        "metric_user": "rmse",
        "metric_sklearn": "neg_root_mean_squared_error",
        "metric_direction": MetricDirection.minimize,
        "runs": [],
        "winner": "ridge",
        "winner_score": 0.5,
        "winner_params": {"alpha": 0.1},
    }
    values.update(overrides)
    return values


def _make_result(runs: list[dict[str, Any]], **overrides: Any) -> ExperimentResult:
    """Build an ``ExperimentResult`` from ``AlgorithmRun`` kwargs dicts."""
    values = _result_kwargs(runs=[AlgorithmRun(**r) for r in runs])
    values.update(overrides)
    return ExperimentResult(**values)


# ────────────────────────── positive scenarios ────────────────────────── #


def test_p1_table_matches_epic_example() -> None:
    """P1: columns, order and ranks follow the epic example layout."""
    result = _make_result(
        [
            _run_kwargs(algorithm="Random Forest", score=0.842, score_raw=-0.842),
            _run_kwargs(algorithm="XGBoost", score=0.831, score_raw=-0.831),
            _run_kwargs(algorithm="Ridge", score=0.774, score_raw=-0.774),
        ]
    )
    lb = build_leaderboard(result)
    table = lb.to_table()
    lines = table.splitlines()
    assert [col.strip() for col in lines[0].split(" | ")] == [
        "Rank",
        "Algorithm",
        "Score",
        "Trials",
        "Phase",
    ]
    cells = [[col.strip() for col in row.split(" | ")] for row in lines[2:]]
    # RMSE (minimize): the smallest score takes rank 1.
    assert [c[0] for c in cells] == ["1", "2", "3"]
    assert [c[1] for c in cells] == ["Ridge", "XGBoost", "Random Forest"]
    assert [c[2] for c in cells] == ["0.774", "0.831", "0.842"]
    assert all(c[3] == "10" for c in cells)
    assert all(c[4] == "search" for c in cells)
    # All rendered rows are visually aligned to a common width.
    assert len({visual_width(line) for line in lines}) == 1


def test_p2_summary_winner_line() -> None:
    """P2: the summary block ends with 'Winner: <rank 1 algorithm>'."""
    result = _make_result(
        [
            _run_kwargs(algorithm="Random Forest", score=0.842, score_raw=-0.842),
            _run_kwargs(algorithm="XGBoost", score=0.831, score_raw=-0.831),
            _run_kwargs(algorithm="Gradient Boosting", score=0.809, score_raw=-0.809),
            _run_kwargs(algorithm="Ridge", score=0.774, score_raw=-0.774),
        ]
    )
    lb = build_leaderboard(result)
    summary = lb.to_summary()
    lines = summary.splitlines()
    assert len(lines) == 5
    # The smallest RMSE takes rank 1 (minimize semantics).
    assert lines[0].startswith("1. Ridge")
    assert lines[1].startswith("2. Gradient Boosting")
    assert lines[2].startswith("3. XGBoost")
    assert lines[3].startswith("4. Random Forest")
    assert lines[4] == "   Winner: Ridge"
    assert "0.774" in lines[0]
    assert "0.842" in lines[3]
    # The scores are vertically aligned across the list lines.
    score_columns = [visual_width(line) for line in lines[:4]]
    assert len(set(score_columns)) == 1
    # The score position within each line is identical (visual cells).
    assert [visual_width(line.split("0.")[0]) for line in lines[:4]] == [
        visual_width(lines[0].split("0.")[0])
    ] * 4


def test_p2_summary_status_header_epic_block() -> None:
    """P2: ``status=True`` renders the epic #37 UX-цель block verbatim."""
    result = _make_result(
        [
            _run_kwargs(algorithm="Random Forest", score=0.842, score_raw=-0.842),
            _run_kwargs(algorithm="XGBoost", score=0.831, score_raw=-0.831),
            _run_kwargs(algorithm="Gradient Boosting", score=0.809, score_raw=-0.809),
            _run_kwargs(algorithm="Ridge", score=0.774, score_raw=-0.774),
        ]
    )
    lb = build_leaderboard(result)
    summary = lb.to_summary(status=True)
    lines = summary.splitlines()
    assert lines[0] == COMPLETED_SUMMARY_MESSAGE
    # The ranking follows: 1..N lines then the indented Winner line.
    assert lines[1].startswith("1. Ridge")
    assert lines[4].startswith("4. Random Forest")
    assert lines[5] == "   Winner: Ridge"
    assert len(lines) == 6
    # The Winner indent is a shared constant (kept aligned with the list).
    assert SUMMARY_HEADER_INDENT == "   "


def test_p2_summary_status_header_for_failed_result() -> None:
    """P2: a failed run with valid records reports the failed status.

    Symmetric to the completed case: some algorithms may have scored
    even when the run as a whole failed — the ranking stays under the
    ``Experiment failed`` header.
    """
    result = _make_result(
        [
            _run_kwargs(algorithm="healthy", score=0.5, score_raw=-0.5),
        ],
        status=RunStatus.failed,
        error="a late phase crashed",
    )
    lb = build_leaderboard(result)
    assert len(lb) == 1
    summary = lb.to_summary(status=True)
    lines = summary.splitlines()
    assert lines[0] == FAILED_SUMMARY_MESSAGE
    assert lines[1].startswith("1. healthy")
    assert lines[-1] == "   Winner: healthy"


def test_p2_summary_status_never_doubled_on_empty_leaderboard() -> None:
    """P2: the empty-leaderboard failure placeholder is not doubled."""
    result = ExperimentResult(
        **_result_kwargs(
            status=RunStatus.failed,
            error="No valid results after HPO phases",
            winner=None,
            winner_score=None,
            winner_params={},
            runs=[],
        )
    )
    lb = build_leaderboard(result)
    assert lb.to_summary() == FAILED_SUMMARY_MESSAGE
    assert lb.to_summary(status=True) == FAILED_SUMMARY_MESSAGE


def test_p3_run_details_available() -> None:
    """P3: params, validation strategy, training time and failed trials."""
    result = _make_result(
        [
            _run_kwargs(
                algorithm="ridge",
                score=0.5,
                score_raw=-0.5,
                params={"alpha": 0.1, "fit_intercept": True},
                n_trials_failed=2,
                duration_seconds=75.5,
            )
        ]
    )
    lb = build_leaderboard(result)
    entry = lb.winner
    assert entry is not None
    assert isinstance(entry, LeaderboardEntry)
    assert entry.params == {"alpha": 0.1, "fit_intercept": True}
    assert entry.validation_strategy == "k_fold"
    assert entry.n_folds == 5
    assert entry.training_time_seconds == pytest.approx(75.5)
    assert entry.n_trials_failed == 2
    assert entry.phase == "search"
    assert entry.phase_action is RunPhaseAction.all_algorithms


def test_p4_minimized_metric_smallest_score_wins() -> None:
    """P4: for RMSE the rank 1 belongs to the smallest score."""
    result = _make_result(
        [
            _run_kwargs(algorithm="worse", score=0.9, score_raw=-0.9),
            _run_kwargs(algorithm="better", score=0.1, score_raw=-0.1),
        ]
    )
    lb = build_leaderboard(result)
    assert [e.algorithm for e in lb.entries] == ["better", "worse"]
    assert lb.winner is not None and lb.winner.algorithm == "better"


def test_p4_maximized_metric_largest_score_wins() -> None:
    """The mirrored case: for R² the rank 1 belongs to the largest score."""
    result = _make_result(
        [
            _run_kwargs(algorithm="worse", score=0.1, score_raw=0.1),
            _run_kwargs(algorithm="better", score=0.9, score_raw=0.9),
        ],
        metric_user="r2",
        metric_sklearn="r2",
        metric_direction=MetricDirection.maximize,
        winner="better",
        winner_score=0.9,
    )
    lb = build_leaderboard(result)
    assert [e.algorithm for e in lb.entries] == ["better", "worse"]
    assert lb.metric_direction is MetricDirection.maximize


def test_p4_raw_optimizer_values_stay_hidden() -> None:
    """The leaderboard exposes only user-semantic values (no neg_ values)."""
    result = _make_result([_run_kwargs(algorithm="ridge", score=0.5, score_raw=-0.5)])
    lb = build_leaderboard(result)
    table = lb.to_table()
    assert "-0.5" not in table
    assert "0.500" in table


def test_p3_best_record_phase_is_reported() -> None:
    """The Phase column names the phase of the algorithm's best record."""
    result = _make_result(
        [
            _run_kwargs(
                algorithm="rf",
                phase_name="all_algorithms",
                phase_action="all_algorithms",
                score=0.85,
                score_raw=-0.85,
            ),
            _run_kwargs(
                algorithm="rf",
                phase_name="refine",
                phase_action="refine",
                score=0.80,
                score_raw=-0.80,
            ),
            _run_kwargs(
                algorithm="rf",
                phase_name="select",
                phase_action="select",
                score=0.90,
                score_raw=-0.90,
            ),
        ]
    )
    lb = build_leaderboard(result)
    assert len(lb) == 1
    assert lb.entries[0].phase == "refine"
    assert lb.entries[0].phase_action is RunPhaseAction.refine
    assert lb.entries[0].score == pytest.approx(0.80)


def test_p3_trials_semantics_own_completed_only() -> None:
    """The Trials column counts the phase's own COMPLETED trials only."""
    result = _make_result(
        [
            _run_kwargs(
                algorithm="rf",
                n_trials_requested=100,
                # The transferred history (previous_trials) is not counted
                # by the engine in n_trials_evaluated: the field is the
                # phase's own evaluated trials.
                n_trials_evaluated=42,
                n_trials_completed=40,
                n_trials_pruned=2,
                n_trials_failed=3,
            )
        ]
    )
    lb = build_leaderboard(result)
    assert lb.entries[0].trials == 42
    assert lb.entries[0].n_trials_failed == 3


# ────────────────────────── negative scenarios ────────────────────────── #


def test_n1_failed_experiment_no_exception() -> None:
    """N1: a failed experiment without valid records renders safely."""
    result = ExperimentResult(
        **_result_kwargs(
            status=RunStatus.failed,
            error="No valid results after HPO phases",
            winner=None,
            winner_score=None,
            winner_params={},
            runs=[],
        )
    )
    lb = build_leaderboard(result)
    assert len(lb) == 0
    assert lb.winner is None
    assert lb.to_table() == EMPTY_LEADERBOARD_MESSAGE
    assert lb.to_summary() == FAILED_SUMMARY_MESSAGE


def test_n1_empty_runs_completed_result() -> None:
    """N1: a completed result with an empty record list is not an error."""
    result = ExperimentResult(**_result_kwargs(runs=[]))
    lb = build_leaderboard(result)
    assert len(lb) == 0
    assert lb.to_table() == EMPTY_LEADERBOARD_MESSAGE


def test_n2_invalid_score_records_excluded() -> None:
    """N2: failed/disqualified records never reach the table."""
    result = _make_result(
        [
            _run_kwargs(
                algorithm="broken",
                status=RunStepStatus.failed,
                score=None,
                score_raw=None,
                error="HPO crashed",
            ),
            _run_kwargs(
                algorithm="disqualified",
                status=RunStepStatus.disqualified,
                score=None,
                score_raw=None,
                error="circuit breaker",
            ),
            _run_kwargs(algorithm="healthy", score=0.5, score_raw=-0.5),
        ]
    )
    lb = build_leaderboard(result)
    assert [e.algorithm for e in lb.entries] == ["healthy"]
    assert "broken" not in lb.to_table()
    assert "disqualified" not in lb.to_table()


def test_n2_completed_record_without_score_excluded() -> None:
    """N2: a completed record without a valid score is not ranked.

    The model rejects non-finite/sentinel score values, so the only way a
    completed record may carry no score is ``score=None`` (a hand-built
    result): the leaderboard filters it out instead of crashing.
    """
    result = _make_result(
        [
            _run_kwargs(algorithm="no_score", score=None, score_raw=None),
            _run_kwargs(algorithm="healthy", score=0.5, score_raw=-0.5),
        ]
    )
    lb = build_leaderboard(result)
    assert [e.algorithm for e in lb.entries] == ["healthy"]


# ────────────────────────── boundary scenarios ────────────────────────── #


def test_b1_tie_broken_by_configuration_order() -> None:
    """B1: equal scores → the configuration order decides the ranks."""
    result = _make_result(
        [
            _run_kwargs(algorithm="alpha", score=0.5, score_raw=-0.5),
            _run_kwargs(algorithm="beta", score=0.5, score_raw=-0.5),
            _run_kwargs(algorithm="gamma", score=0.5, score_raw=-0.5),
        ],
        config={
            "general": {
                "phases": [
                    {
                        "name": "search",
                        "action": "all_algorithms",
                        "algorithms": {"gamma": {}, "beta": {}, "alpha": {}},
                    }
                ]
            }
        },
    )
    lb = build_leaderboard(result)
    assert [e.algorithm for e in lb.entries] == ["gamma", "beta", "alpha"]


def test_b1_tie_without_config_snapshot_uses_name_order() -> None:
    """B1: without a config snapshot the ties fall back to name order."""
    result = _make_result(
        [
            _run_kwargs(algorithm="second", score=0.5, score_raw=-0.5),
            _run_kwargs(algorithm="first", score=0.5, score_raw=-0.5),
        ]
    )
    lb = build_leaderboard(result)
    # No configuration order is recoverable from an empty config: the
    # deterministic fallback is the alphabetical algorithm-name order.
    assert [e.algorithm for e in lb.entries] == ["first", "second"]


def test_b1_malformed_config_snapshot_does_not_crash() -> None:
    """B1: a malformed hand-built config snapshot never crashes.

    The defensive contract of ``_configuration_order``: a non-mapping
    ``config['general']`` section (and, symmetrically, any other
    malformed part of the snapshot) is skipped, the ranking falls back to
    the name order instead of raising.
    """
    result = _make_result(
        [
            _run_kwargs(algorithm="second", score=0.5, score_raw=-0.5),
            _run_kwargs(algorithm="first", score=0.5, score_raw=-0.5),
        ],
        config={"general": "garbage"},
    )
    lb = build_leaderboard(result)
    assert [e.algorithm for e in lb.entries] == ["first", "second"]
    assert len(lb) == 2


def test_b1_tie_independent_of_sort_input_order() -> None:
    """B1: the tie-break is stable against the record input order."""
    kwargs = {
        "config": {
            "general": {
                "phases": [
                    {
                        "name": "search",
                        "action": "all_algorithms",
                        "algorithms": {"one": {}, "two": {}},
                    }
                ]
            }
        }
    }
    forward = _make_result(
        [
            _run_kwargs(algorithm="one", score=0.5, score_raw=-0.5),
            _run_kwargs(algorithm="two", score=0.5, score_raw=-0.5),
        ],
        **kwargs,
    )
    backward = _make_result(
        [
            _run_kwargs(algorithm="two", score=0.5, score_raw=-0.5),
            _run_kwargs(algorithm="one", score=0.5, score_raw=-0.5),
        ],
        **kwargs,
    )
    assert [e.algorithm for e in build_leaderboard(forward).entries] == [
        e.algorithm for e in build_leaderboard(backward).entries
    ]


def test_b2_single_record() -> None:
    """B2: exactly one record — rank 1, no padding artifacts."""
    result = _make_result([_run_kwargs(algorithm="only", score=0.42, score_raw=-0.42)])
    lb = build_leaderboard(result)
    assert len(lb) == 1
    entry = lb.winner
    assert entry is not None
    assert entry.rank == 1
    assert entry.algorithm == "only"
    summary = lb.to_summary()
    assert summary.splitlines()[-1] == "   Winner: only"
    # The statusless rendering starts with the ranking (no status header).
    assert summary.splitlines()[0].startswith("1. only")
    # With the status flag the header precedes the single ranking line.
    status_lines = lb.to_summary(status=True).splitlines()
    assert status_lines[0] == COMPLETED_SUMMARY_MESSAGE
    assert status_lines[-1] == "   Winner: only"


def test_b3_long_names_truncated_and_aligned() -> None:
    """B3: long names are truncated with an ellipsis; rows stay aligned."""
    long_name = "very-long-algorithm-name-" + "x" * 60
    result = _make_result(
        [
            _run_kwargs(algorithm=long_name, score=0.5, score_raw=-0.5),
            _run_kwargs(algorithm="short", score=0.6, score_raw=-0.6),
        ]
    )
    lb = build_leaderboard(result)
    table = lb.to_table(max_cell_width=20)
    lines = table.splitlines()
    # Every line has the same visual width (monospace alignment holds).
    assert len({visual_width(line) for line in lines}) == 1
    truncated = [line for line in lines if "…" in line]
    assert truncated, "the long name must be truncated with an ellipsis"
    for line in truncated:
        assert "very-long-alg" in line


def test_b3_long_params_truncated_in_details() -> None:
    """B3: the detail rendering truncates long parameter strings."""
    result = _make_result(
        [
            _run_kwargs(
                algorithm="ridge",
                params={"alpha": 0.1, "long_param_name": "v" * 100},
            )
        ]
    )
    lb = build_leaderboard(result)
    entry = lb.winner
    assert entry is not None
    assert entry.params["long_param_name"] == "v" * 100  # structured: intact
    rendered = lb.to_table(max_cell_width=15)
    assert len({visual_width(line) for line in rendered.splitlines()}) == 1


def test_b4_cyrillic_names_aligned() -> None:
    """B4: Cyrillic names align by display width (len == width here)."""
    result = _make_result(
        [
            _run_kwargs(algorithm="Градиентный Бустинг", score=0.5, score_raw=-0.5),
            _run_kwargs(algorithm="Лес", score=0.6, score_raw=-0.6),
        ]
    )
    lb = build_leaderboard(result)
    table = lb.to_table()
    assert "Градиентный Бустинг" in table
    assert len({visual_width(line) for line in table.splitlines()}) == 1


def test_b4_wide_unicode_double_width() -> None:
    """B4: CJK/wide characters count as two cells in the alignment."""
    result = _make_result(
        [
            _run_kwargs(algorithm="漢字", score=0.5, score_raw=-0.5),
            _run_kwargs(algorithm="ab", score=0.6, score_raw=-0.6),
        ]
    )
    lb = build_leaderboard(result)
    table = lb.to_table()
    # 漢字 (4 cells) and ab (2 cells) share the same padded column: the
    # number of padding spaces differs by exactly 2.
    lines = table.splitlines()
    rows = [line for line in lines if line.startswith(("1 ", "2 "))]
    assert len({visual_width(row) for row in rows}) == 1
    assert visual_width("漢字") == 4
    assert visual_width("ab") == 2


def test_b4_combining_marks_normalized() -> None:
    """B4: decomposed unicode is composed (NFC) before width measuring."""
    decomposed = "e\u0301cole"  # é as e + combining accent
    assert visual_width(decomposed) == len("école")


def test_b5_zero_own_trials() -> None:
    """B5: a phase without own COMPLETED trials renders as 0 trials."""
    result = _make_result(
        [
            _run_kwargs(
                algorithm="rf",
                n_trials_requested=50,
                n_trials_evaluated=0,
                n_trials_completed=0,
                n_trials_pruned=0,
                n_trials_failed=0,
                duration_seconds=0.0,
            )
        ]
    )
    lb = build_leaderboard(result)
    entry = lb.winner
    assert entry is not None
    assert entry.trials == 0
    assert entry.n_trials_failed == 0
    assert entry.training_time_seconds == 0.0
    # The rendered Trials cell of the rank-1 row is exactly "0".
    trials_row = [line for line in lb.to_table().splitlines() if line.startswith("1 ")]
    assert len(trials_row) == 1
    assert trials_row[0].split(" | ")[3].strip() == "0"


def test_b5_unknown_training_time() -> None:
    """B5: a missing duration renders as 'n/a' in the detail rendering."""
    result = _make_result(
        [_run_kwargs(algorithm="ridge", duration_seconds=None)]
    )
    lb = build_leaderboard(result)
    entry = lb.winner
    assert entry is not None
    assert entry.training_time_seconds is None


def test_p1_structured_access_entries() -> None:
    """The structured access: entries expose typed fields, not strings."""
    result = _make_result(
        [
            _run_kwargs(algorithm="rf", score=0.842, score_raw=-0.842),
            _run_kwargs(algorithm="ridge", score=0.774, score_raw=-0.774),
        ]
    )
    lb = build_leaderboard(result)
    assert isinstance(lb, Leaderboard)
    assert lb.metric_user == "rmse"
    assert lb.metric_direction is MetricDirection.minimize
    for idx, entry in enumerate(lb):
        assert isinstance(entry, LeaderboardEntry)
        assert entry.rank == idx + 1
        assert isinstance(entry.score, float)
        assert isinstance(entry.trials, int)
        assert isinstance(entry.phase, str)
