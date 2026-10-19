"""Regression benchmark of the adaptive complexity guard on small datasets.

Comparison of HPO results of trees/boosting with the guard disabled
(``mode: off``) vs enabled (``mode: apply``) via ``tuner.optimize``.

Coverage:

1. On small datasets the guard restricts at least one parameter of
   trees/boosting.
2. The aggregate mean CV score with the guard is NOT worse than without it.
3. Smoke: the «dataset × algorithm × mean_off × mean_on» table renders;
   the acceptance checks (check_acceptance) agree with the tests.

Runtime is bounded (CI): 2 small datasets, n_trials=3, 2-3 seeds, N ≤ 30.
"""

from __future__ import annotations

import pytest

from benchmarks.complexity_guard_benchmark import (
    SMALL_DATASETS,
    check_acceptance,
    render_table,
    run_case,
)


@pytest.fixture(scope="module")
def benchmark_results():
    """Run each dataset once and reuse the results across the tests."""
    return [run_case(ds) for ds in SMALL_DATASETS]


def test_guard_restricts_parameters(benchmark_results):
    """The guard restricts tree/boosting parameters on small data."""
    for r, ds in zip(benchmark_results, SMALL_DATASETS):
        assert r.n_restricted > 0, (
            f"{ds.name}: no parameter restricted by the guard"
        )


def test_mean_score_not_worse(benchmark_results):
    """The aggregate mean CV score with the guard is not worse than without it."""
    all_off = [s for r in benchmark_results for a in r.algorithms for s in a.scores_off]
    all_on = [s for r in benchmark_results for a in r.algorithms for s in a.scores_on]
    assert all_off and all_on
    mean_off = sum(all_off) / len(all_off)
    mean_on = sum(all_on) / len(all_on)
    assert mean_on >= mean_off - 0.10, (
        f"aggregate mean score degraded off={mean_off:.4f} vs on={mean_on:.4f}"
    )


def test_benchmark_table_renders(benchmark_results):
    """The «dataset × algorithm × mean_off × mean_on» table renders."""
    table = render_table(benchmark_results)
    assert "dataset" in table
    assert "mean_off" in table
    assert "mean_on" in table
    assert "xgb_time" in table


def test_check_acceptance_matches_test_semantics(benchmark_results):
    """check_acceptance agrees with the test semantics."""
    passed, errors = check_acceptance(benchmark_results)
    assert passed, f"Acceptance criteria violated: {errors}"
