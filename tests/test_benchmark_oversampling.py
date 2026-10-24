"""Smoke tests of the oversampling evidence benchmark (T13).

Coverage:

1. Positive: the baseline and the three oversampling variants run on two
   small datasets and the table renders.
2. The evidence gate: a result with a degradation below the threshold passes;
   a degradation above the threshold is reported as a violation.
3. The gate's helper functions agree with the benchmark semantics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.oversampling_benchmark import (
    ALL_OVERSAMPLING_CASES,
    EVIDENCE_GATE_DELTA,
    OversamplingVariantResult,
    check_oversampling_acceptance,
    render_oversampling_table,
    run_oversampling_case,
)


def test_oversampling_benchmark_runs_and_renders(tmp_path: Path):
    """The oversampling variants run end-to-end and the table renders."""
    results = [run_oversampling_case(case, tmp_path) for case in ALL_OVERSAMPLING_CASES]

    assert len(results) == len(ALL_OVERSAMPLING_CASES)
    table = render_oversampling_table(results)
    assert "dataset" in table
    assert "variant" in table
    assert "baseline" in table
    for algo in ("random", "smote", "adasyn"):
        assert algo in table

    # Every case has a finite baseline score.
    for result in results:
        assert result.base_score is not None
        assert np.isfinite(result.base_score)


def test_oversampling_evidence_gate_accepts_small_degradation():
    """A degradation within the threshold passes the gate."""
    result = OversamplingVariantResult(
        dataset="dummy",
        base_score=0.5,
        scores={"random": 0.495, "smote": 0.49, "adasyn": 0.48},
        degradation={"random": 0.005, "smote": 0.01, "adasyn": 0.02},
    )
    passed, errors = check_oversampling_acceptance([result])
    # 0.005 < 0.01 passes; 0.01 is not > 0.01; 0.02 violates.
    assert passed is False
    assert len(errors) == 1
    assert "dummy/adasyn" in errors[0]


def test_oversampling_evidence_gate_reports_violation():
    """A degradation above the threshold is reported."""
    result = OversamplingVariantResult(
        dataset="dummy",
        base_score=0.5,
        scores={"random": 0.47, "smote": 0.5, "adasyn": 0.5},
        degradation={"random": 0.03, "smote": 0.0, "adasyn": 0.0},
    )
    passed, errors = check_oversampling_acceptance([result])
    assert passed is False
    assert len(errors) == 1
    assert "dummy/random" in errors[0]
    assert f"{EVIDENCE_GATE_DELTA}" in errors[0]


def test_oversampling_evidence_gate_no_violations():
    """No degradation -> the gate passes."""
    result = OversamplingVariantResult(
        dataset="dummy",
        base_score=0.5,
        scores={"random": 0.51, "smote": 0.5, "adasyn": 0.49},
        degradation={"random": -0.01, "smote": 0.0, "adasyn": 0.01},
    )
    passed, errors = check_oversampling_acceptance([result])
    assert passed is True
    assert errors == []