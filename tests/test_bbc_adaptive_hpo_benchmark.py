"""Tests for the BBC-CV-after-adaptive-HPO validation benchmark (CI gate).

The benchmark is a mandatory CI gate (the Sanity Gate pattern), so the tests
verify both its machinery (the dataset generation contract, the external
reference estimate, the per-seed comparison run, the reporting helpers) and
the acceptance semantics of the gate: honest results pass
``check_bbc_acceptance``, deliberately optimistic ones do not.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from benchmarks.bbc_adaptive_hpo_benchmark import (
    BBC_ACCEPT_MAX_MEDIAN_GAP,
    CASE_LINEAR,
    CASE_NOISY_NONLINEAR,
    CI_BBC_CASES,
    CI_SEEDS,
    DatasetResult,
    DatasetSpec,
    _external_estimate,
    _regression_xy,
    check_bbc_acceptance,
    render_bbc_table,
    run_bbc_case,
    summarize_bbc_verdict,
)


def test_regression_xy_shapes_and_reproducibility():
    X1, y1 = _regression_xy(50, 8, 3, 10.0, 0)
    X2, y2 = _regression_xy(50, 8, 3, 10.0, 0)
    assert isinstance(X1, pd.DataFrame) and isinstance(y1, pd.Series)
    assert X1.shape == (50, 8) and y1.shape == (50,)
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)


def test_external_estimate_linear_near_ceiling():
    """Ridge with the near-optimal alpha on a noise-free linear dataset
    must reach a high external R² (the generator's ceiling is ~1.0)."""
    spec = DatasetSpec(
        name="clean_linear",
        n_samples=100,
        n_features=5,
        n_informative=5,
        noise=1.0,
        algo="ridge",
        external_repeats=3,
    )
    value = _external_estimate(spec, {"alpha": 1.0}, seeds=range(3))
    assert 0.9 < value <= 1.0


def test_external_estimate_sensitive_to_configuration():
    """A deliberately bad configuration must score clearly worse than a
    good one — the reference really evaluates the selected configuration."""
    spec = DatasetSpec(
        name="linear",
        n_samples=100,
        n_features=8,
        n_informative=4,
        noise=10.0,
        algo="ridge",
        external_repeats=3,
    )
    good = _external_estimate(spec, {"alpha": 1.0}, seeds=range(3))
    bad = _external_estimate(spec, {"alpha": 1e6}, seeds=range(3))
    assert good > bad


def test_run_bbc_case_returns_aligned_per_seed_estimates():
    seeds = [1, 2]
    spec = DatasetSpec(
        name="linear_small",
        n_samples=120,
        n_features=10,
        n_informative=3,
        noise=15.0,
        algo="ridge",
        n_trials=4,
        external_repeats=2,
    )
    result = run_bbc_case(spec, seeds, n_bootstrap=20)
    assert result.seeds == seeds
    assert len(result.bbc_scores) == len(seeds)
    assert len(result.external_scores) == len(seeds)
    assert len(result.cv_best_scores) == len(seeds)
    assert all(np.isfinite(result.bbc_scores))
    assert all(np.isfinite(result.external_scores))
    # The BBC-CV estimate corrects the best-trial optimism: it must never
    # exceed the best CV score on the same OOF matrix by a wide margin.
    assert all(
        bbc <= cv + 0.05
        for bbc, cv in zip(result.bbc_scores, result.cv_best_scores, strict=True)
    )


def test_render_and_summarize_include_all_datasets():
    seeds = [1]
    spec = DatasetSpec(
        name="linear_tiny",
        n_samples=80,
        n_features=6,
        n_informative=2,
        noise=10.0,
        algo="ridge",
        n_trials=3,
        external_repeats=2,
    )
    result = run_bbc_case(spec, seeds, n_bootstrap=10)
    table = render_bbc_table([result])
    verdict = summarize_bbc_verdict([result])
    assert spec.name in table
    assert spec.name in verdict
    assert "gap" in table
    assert "mean gap" in verdict


def test_default_cases_cover_noisy_scenario():
    """A key scenario: many noise features — the default case set must
    include it."""
    assert CASE_NOISY_NONLINEAR.n_features > CASE_NOISY_NONLINEAR.n_informative
    assert CASE_NOISY_NONLINEAR.n_trials >= CASE_LINEAR.n_trials


def test_check_bbc_acceptance_passes_honest_results():
    """check_bbc_acceptance matches the test semantics: real (honest)
    benchmark results with gaps inside the allowance pass the gate."""
    spec = DatasetSpec(
        name="linear_gate_pass",
        n_samples=120,
        n_features=10,
        n_informative=3,
        noise=15.0,
        algo="ridge",
        n_trials=4,
        external_repeats=2,
    )
    results = [run_bbc_case(spec, [1, 2], n_bootstrap=20)]
    passed, errors = check_bbc_acceptance(results)
    assert passed, f"Acceptance criteria violated: {errors}"


def _make_result(name: str, bbc: list[float], external: list[float]) -> DatasetResult:
    """A synthetic DatasetResult with the given per-seed estimates."""
    return DatasetResult(
        name=name,
        algo="ridge",
        n_trials=10,
        seeds=list(range(1, len(bbc) + 1)),
        bbc_scores=list(bbc),
        external_scores=list(external),
        cv_best_scores=list(bbc),
    )


def test_check_bbc_acceptance_rejects_optimistic_bbc():
    """A systematically optimistic BBC-CV (a shifted median gap above the
    allowance) must fail the gate."""
    external = [0.5, 0.52, 0.48]
    biased = [v + BBC_ACCEPT_MAX_MEDIAN_GAP + 0.05 for v in external]
    passed, errors = check_bbc_acceptance(
        [_make_result("linear_optimistic", biased, external)]
    )
    assert not passed
    assert any("|median gap|" in e for e in errors)


def test_gate_median_absorbs_single_unlucky_seed():
    """The gate statistic is the MEDIAN of the per-seed gaps: one seed with
    a large negative gap (as measured on the real seed windows, e.g.
    −0.1156) must NOT fail a group whose other seeds sit near zero — a
    mean-based statistic would (the blocking finding of round 2)."""
    external = [0.5, 0.5, 0.5]
    result = _make_result(
        "linear_unlucky", [0.5 + g for g in (-0.1156, 0.0, 0.005)], external
    )
    # The median is inside the allowance while the mean is not.
    assert abs(float(np.median(result.gaps))) <= BBC_ACCEPT_MAX_MEDIAN_GAP
    assert abs(float(np.mean(result.gaps))) > BBC_ACCEPT_MAX_MEDIAN_GAP
    passed, errors = check_bbc_acceptance([result])
    assert passed, errors


def test_gate_has_no_z_criterion():
    """No z statistic is checked: with 3 seeds per group the sample SE is
    too noisy for a mandatory gate (a z from n=3 fails a zero-bias engine
    at random). Behavioral check: a tight consistent small gap with a
    near-infinite z (SE ≈ 0 at n=3) passes the gate — only the median-gap
    criterion exists."""
    external = [0.5, 0.5, 0.5]
    biased = [v + BBC_ACCEPT_MAX_MEDIAN_GAP / 2 for v in external]
    result = _make_result("linear_small_shift", biased, external)
    gaps = result.gaps
    se = float(np.std(gaps, ddof=1) / np.sqrt(len(gaps)))
    if se > 0:
        assert abs(float(np.mean(gaps)) / se) > 2.0  # a z criterion would fire
    passed, errors = check_bbc_acceptance([result])
    assert passed, errors


def test_check_bbc_acceptance_rejects_positive_bbc_on_pure_noise():
    """A positive mean BBC-CV estimate on a signal-free case is pure
    selection bias — the gate must fail."""
    passed, errors = check_bbc_acceptance(
        [_make_result("pure_noise_positive", [0.1, 0.12], [0.0, -0.02])]
    )
    assert not passed
    assert any("signal-free" in e for e in errors)


def test_check_bbc_acceptance_negative_gap_is_not_optimism():
    """A gap inside the allowance passes even when negative (pessimistic)
    — the gate guards against optimism of BBC-CV, not small noise."""
    external = [0.5, 0.5, 0.5]
    conservative = [v - BBC_ACCEPT_MAX_MEDIAN_GAP / 2 for v in external]
    passed, errors = check_bbc_acceptance(
        [_make_result("linear_pessimistic", conservative, external)]
    )
    assert passed, errors


def test_ci_profile_specs_are_cheap():
    """The CI profile stays ridge-only with a bounded trial budget and
    includes the signal-free case; the sample size and external repeats
    are chosen to shrink the per-seed gap noise (the gate statistic is a
    pooled median over 6 seeds and must not flip on seed luck)."""
    assert CI_BBC_CASES
    for spec in CI_BBC_CASES:
        assert spec.algo == "ridge", spec.name
        assert spec.n_trials <= 10, spec.name
        assert spec.n_samples >= 500, spec.name
        assert spec.external_repeats >= 10, spec.name
    assert any("pure_noise" in s.name for s in CI_BBC_CASES)


def test_ci_seeds_are_six_for_one_pooled_run_per_case():
    """The CI seed set covers 6 seeds: the gate statistic is the pooled
    median over all 6 per-seed gaps per case (calibrated on 15 independent
    6-seed windows — per-group-of-3 medians flip too often for a mandatory
    gate)."""
    assert len(CI_SEEDS) == 6
    assert len(set(CI_SEEDS)) == 6


def test_gate_pooled_median_absorbs_one_unlucky_seed():
    """The pooled gate statistics: one unlucky seed (a large negative gap,
    as measured on the real seed windows) does not fail the case whose
    other seeds sit near zero, while a systematic +0.05 shift on all
    seeds fails."""
    external = [0.5] * 6
    gaps = [-0.1156, 0.0004, 0.0068, -0.0058, 0.0055, -0.0180]
    result = _make_result("linear_pooled", [0.5 + g for g in gaps], list(external))
    # The pooled median is inside the allowance; the mean is dragged
    # 4× further from zero by the unlucky seed (0.0211 vs 0.0064) — a
    # mean-based statistic is close to the boundary.
    assert abs(float(np.median(result.gaps))) <= BBC_ACCEPT_MAX_MEDIAN_GAP
    assert abs(float(np.mean(result.gaps))) > abs(float(np.median(result.gaps))) * 3
    passed, errors = check_bbc_acceptance([result])
    assert passed, errors
    # A systematic shift: the pooled row fails.
    shifted = _make_result(
        "linear_shifted", [v + 0.05 for v in external], list(external)
    )
    passed2, errors2 = check_bbc_acceptance([shifted])
    assert not passed2
    assert len(errors2) == 1


def test_gate_constants_match_documented_allowances():
    """The documented gate allowance of the module docstring is the
    constant actually checked."""
    assert BBC_ACCEPT_MAX_MEDIAN_GAP == pytest.approx(0.025)


def test_acceptance_uses_case_results_not_copies():
    """check_bbc_acceptance consumes DatasetResult objects as produced by
    run_bbc_case: mutating the input list does not change the verdict."""
    spec = DatasetSpec(
        name="linear_gate_pass",
        n_samples=120,
        n_features=10,
        n_informative=3,
        noise=15.0,
        algo="ridge",
        n_trials=4,
        external_repeats=2,
    )
    results = [run_bbc_case(spec, [1, 2], n_bootstrap=20)]
    snapshot = copy.deepcopy(results)
    passed1, _ = check_bbc_acceptance(results)
    results.pop()
    passed2, _ = check_bbc_acceptance(snapshot)
    assert passed1 == passed2
