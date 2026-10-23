"""Regression benchmark of the Sanity Gate on the target pipeline.

Comparison of pipeline variants on the same datasets:

* ``baseline`` — ``all_algorithms → select(cv_best, winner)`` (CV leader
  without audit, the modern counterpart of the old behavior);
* ``target`` — ``all_algorithms → select(sanity_gate, top_k) → refine →
  select(sanity_gate, winner)`` (the target DAG).

Coverage:

1. Positive: on reference datasets (noisy R²≈0.3, wide p=200, small N)
   the winner did not change AND the gate did not trigger (0 disqualifications
   in any sanity-select).
2. Negative: on degenerate datasets (shelf / tanh plateau / overfitting)
   the winner changed, the degenerate CV leader was disqualified, and the
   honest candidate wins in the target DAG.
3. Smoke: the table «dataset × variant × winner × gate» renders;
   the checks (check_acceptance) agree with the tests.

Runtime is bounded (CI): n_trials=1, N ≤ 300, 6 datasets.
"""

from __future__ import annotations

from benchmarks.sanity_gate_benchmark import (
    ALL_CASES,
    DEGENERATE_CASES,
    GATE_SELECT_PHASE,
    REFERENCE_CASES,
    check_acceptance,
    render_table,
    run_case,
)


# ──────────────────────────────────────────────────────────────────────────────
#  Positive: reference datasets — winner unchanged, gate not triggered
# ──────────────────────────────────────────────────────────────────────────────


def test_reference_winner_unchanged_and_gate_not_triggered():
    """Reference datasets: 0 winner changes (baseline vs target).

    The gate must NOT trigger on normal data: no candidate of the pool is
    disqualified and the target DAG winner matches the CV leader.
    """
    for case in REFERENCE_CASES:
        result = run_case(case)
        assert result.winner_target == result.winner_baseline, (
            f"{case.name}: {result.winner_baseline} -> {result.winner_target}"
        )
        assert result.changed is False
        assert result.gate_triggered is False, result.disqualified
        assert all(not names for names in result.disqualified.values())
        # The finalists pool of the first sanity-select is non-empty.
        assert result.pool


# ──────────────────────────────────────────────────────────────────────────────
#  Negative: degenerate datasets — the winner must change
# ──────────────────────────────────────────────────────────────────────────────


def test_degenerate_winner_changes():
    """Degenerate datasets: 100% winner changes.

    The degenerate candidate (baseline CV leader) is disqualified by the gate
    in the first sanity-select of the target DAG, and the honest candidate
    becomes the winner.
    """
    for case in DEGENERATE_CASES:
        result = run_case(case)
        # The winner must change (shelf / tanh plateau / overfitting).
        assert result.changed is True, (
            f"{case.name}: winners equal ({result.winner_baseline})"
        )
        assert result.winner_target != result.winner_baseline
        # The gate triggered: the degenerate CV leader was disqualified in the
        # first sanity-select; the honest winner passed the audit.
        assert result.gate_triggered is True
        assert result.winner_baseline in result.disqualified[GATE_SELECT_PHASE]
        assert result.audit_valid[result.winner_baseline] is False
        assert result.audit_valid[result.winner_target] is True


# ──────────────────────────────────────────────────────────────────────────────
#  Smoke: table and checks
# ──────────────────────────────────────────────────────────────────────────────


def test_benchmark_table_renders_with_expected_columns():
    """The «dataset × variant × winner × gate» table renders."""
    # Two fast cases are enough for the smoke check of the formatting.
    results = [run_case(case) for case in (REFERENCE_CASES[2], DEGENERATE_CASES[2])]
    table = render_table(results)
    assert "dataset" in table
    assert "variant" in table
    assert "winner" in table
    assert "gate" in table
    assert "baseline" in table and "target" in table
    # Each dataset has two rows (baseline + target).
    assert table.count(results[0].dataset) >= 1


def test_check_acceptance_matches_test_semantics():
    """check_acceptance agrees with the test semantics.

    Reference cases without changes and triggers pass; degenerate cases with
    a change and a CV-leader disqualification pass.
    """
    results = [run_case(case) for case in ALL_CASES]
    passed, errors = check_acceptance(results)
    assert passed, f"Acceptance criteria violated: {errors}"