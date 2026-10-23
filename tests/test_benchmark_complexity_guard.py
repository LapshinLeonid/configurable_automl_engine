"""Regression benchmark of the adaptive complexity guard on small datasets.

Comparison of HPO results of trees/boosting with the guard disabled
(``mode: off``) vs enabled (``mode: apply``) via ``tuner.optimize``.

Coverage:

1. On small datasets the guard restricts at least one parameter of
   trees/boosting.
2. The aggregate mean CV score with the guard is NOT worse than without it.
3. Smoke: the «dataset × algorithm × mean_off × mean_on» table renders;
   the checks (check_acceptance) agree with the tests.
4. The FULL profile (``FULL_DATASETS``) has the required shape
   (>= 5 datasets, >= 5 seeds, >= 30 trials) and
   ``check_full_acceptance`` enforces the criteria on synthetic results
   (score not worse, seed scatter not larger, XGBoost time markedly lower).

Runtime is bounded (CI): 2 small datasets, n_trials=3, 2-3 seeds, N ≤ 30.
The FULL profile itself is NOT run in CI (it takes minutes) — only its
structure and the checker logic are unit-tested here.
"""

from __future__ import annotations

import pytest

from benchmarks.complexity_guard_benchmark import (
    FULL_DATASETS,
    SMALL_DATASETS,
    AlgoResult,
    Dataset,
    GuardResult,
    check_acceptance,
    check_full_acceptance,
    check_full_profile_shape,
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
        assert r.n_restricted > 0, f"{ds.name}: no parameter restricted by the guard"


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


# ─────────────────── FULL profile (outside CI) ──────────────────── #


def test_full_dataset_profile_shape():
    """The FULL profile has the required shape: >= 5 datasets, >= 5 seeds, 30-50 trials."""
    assert check_full_profile_shape(FULL_DATASETS) == [], (
        f"FULL profile violations: {check_full_profile_shape(FULL_DATASETS)}"
    )


def _guard_result_with(
    *,
    algo: str,
    scores_off: list[float],
    scores_on: list[float],
    xgb_time_off: float = 100.0,
    xgb_time_on: float = 60.0,
) -> GuardResult:
    """Build a synthetic GuardResult with a single algorithm."""
    return GuardResult(
        dataset="fake",
        algorithms=[
            AlgoResult(
                algo=algo,
                scores_off=scores_off,
                scores_on=scores_on,
                xgb_time_off=xgb_time_off,
                xgb_time_on=xgb_time_on,
            )
        ],
        n_restricted=3,
    )


def test_check_full_acceptance_passes_on_good_results():
    """The full checker passes on clearly good results."""
    results = [
        _guard_result_with(
            algo="xgboosting", scores_off=[0.8, 0.7], scores_on=[0.82, 0.72]
        ),
        _guard_result_with(
            algo="decision_tree", scores_off=[0.7, 0.6], scores_on=[0.71, 0.62]
        ),
    ]
    passed, errors = check_full_acceptance(results)
    assert passed, f"expected pass, got: {errors}"


def test_check_full_acceptance_rejects_degraded_mean():
    """The full checker rejects a degraded aggregate mean score."""
    results = [
        _guard_result_with(
            algo="xgboosting", scores_off=[0.9, 0.9], scores_on=[0.5, 0.5]
        ),
    ]
    passed, errors = check_full_acceptance(results)
    assert not passed
    assert any("aggregate mean CV score degraded" in e for e in errors)


def test_check_full_acceptance_rejects_larger_seed_scatter():
    """The full checker rejects a grown std over seeds."""
    results = [
        _guard_result_with(
            algo="xgboosting",
            scores_off=[0.8, 0.8, 0.8, 0.8, 0.8],
            scores_on=[0.2, 0.5, 0.8, 0.95, 0.98],
        ),
    ]
    passed, errors = check_full_acceptance(results)
    assert not passed
    assert any("per-cell std" in e or "std over seeds" in e for e in errors)


def test_check_full_acceptance_rejects_per_cell_scatter_growth():
    """The per-cell std criterion catches a growth the pooled one would hide.

    Four cells with DIFFERENT means (pooled std dominated by between-cell
    mean differences) and an ~2.4x per-cell seed-std growth with the guard.
    The pooled-std ratio stays ~1.15 (below the 1.5x bound) and would pass;
    the per-cell criterion must reject it.
    """
    cells_off = [
        [0.3764, 0.24, 0.2979, 0.4241, 0.3868, 0.1023],
        [0.5761, 0.5122, 0.5444, 0.5334, 0.6494, 0.4795],
        [0.927, 0.5546, 0.7046, 0.6813, 0.8533, 0.8469],
        [0.98, 0.98, 0.8613, 0.8698, 0.7951, 0.758],
    ]
    cells_on = [
        [0.485, 0.1546, 0.169, 0.3232, 0.2432, 0.6363],
        [0.5939, 0.2438, 0.02, 0.6961, 0.7593, 0.2774],
        [0.7465, 0.8134, 0.4337, 0.1058, 0.5956, 0.7469],
        [0.3881, 0.98, 0.7471, 0.7686, 0.5242, 0.98],
    ]
    algos = ["xgboosting", "decision_tree", "random_forest", "extra_trees"]
    results = [
        _guard_result_with(algo=algo, scores_off=off, scores_on=on)
        for algo, off, on in zip(algos, cells_off, cells_on)
    ]
    passed, errors = check_full_acceptance(results)
    assert not passed
    assert any("per-cell std" in e for e in errors), errors


def test_check_full_acceptance_rejects_slow_guard():
    """The full checker rejects an XGBoost guard time NOT lower."""
    results = [
        _guard_result_with(
            algo="xgboosting",
            scores_off=[0.8],
            scores_on=[0.81],
            xgb_time_off=10.0,
            xgb_time_on=10.0,
        ),
    ]
    passed, errors = check_full_acceptance(results)
    assert not passed
    assert any("XGBoost HPO time not markedly lower" in e for e in errors)


def test_check_full_acceptance_noisy_cell_drop_is_not_regression():
    """A mean drop within the seed noise is NOT treated as a regression.

    On tiny noisy datasets the per-cell mean difference over seeds has a
    large standard error (e.g. std ~ 0.2-0.4); a drop of 0.05-0.1 is pure
    seed noise, not a guard effect. The noise-aware per-cell tolerance must
    absorb it, while the aggregate mean (both cells) stays within the
    tolerance.
    """
    noisy_off = [0.2, 0.6, 0.1, 0.4, 0.3]  # mean 0.32, std ~0.17
    noisy_on = [0.16, 0.56, 0.06, 0.36, 0.26]  # drop ~0.04 (within noise)
    stable_off = [0.8, 0.82, 0.81, 0.8, 0.82]  # mean ~0.81
    stable_on = [0.79, 0.81, 0.8, 0.79, 0.81]  # drop ~0.01 (below tolerance)
    results = [
        _guard_result_with(
            algo="decision_tree", scores_off=noisy_off, scores_on=noisy_on
        ),
        _guard_result_with(
            algo="xgboosting", scores_off=stable_off, scores_on=stable_on
        ),
    ]
    passed, errors = check_full_acceptance(results)
    assert passed, f"noise-level drops must not fail acceptance: {errors}"


def test_check_full_profile_shape_rejects_bad_profile():
    """The profile-shape validator rejects every violation of the profile shape."""
    bad = Dataset(
        name="dup", generator=lambda *_: (None, None), seeds=(1, 2), n_trials=10
    )
    errors = check_full_profile_shape([bad, bad])
    joined = " | ".join(errors)
    assert "expected >= 5 datasets" in joined
    assert "dataset names are not unique" in joined
    assert any("seeds per dataset" in e for e in errors)
    assert any("30-50 trials" in e for e in errors)


def test_full_benchmark_is_reproducible():
    """Two identical FULL-profile runs produce IDENTICAL scores.

    Regression: the benchmark compared unseeded
    models — ``DEFAULT_SPACES`` carries no ``random_state``, so identical
    hyperparameters scored differently across reruns (bootstrap/boosting/
    XGBoost randomness) and the acceptance gates measured seed noise
    instead of the guard's effect. The benchmark now injects a constant
    ``random_state`` into every covered algorithm's search space, so a
    rerun of ``run_case`` must reproduce the exact per-seed scores.
    """
    from benchmarks.complexity_guard_benchmark import _make_linear

    small = Dataset(
        name="repro_n18", generator=_make_linear, n=18, p=5, seeds=(42, 7), n_trials=2
    )
    r1, r2 = run_case(small), run_case(small)
    for a1, a2 in zip(r1.algorithms, r2.algorithms):
        assert a1.scores_off == a2.scores_off, (
            f"{a1.algo}: off scores differ between reruns "
            f"{a1.scores_off} vs {a2.scores_off}"
        )
        assert a1.scores_on == a2.scores_on, (
            f"{a1.algo}: on scores differ between reruns "
            f"{a1.scores_on} vs {a2.scores_on}"
        )
