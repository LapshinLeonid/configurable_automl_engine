#!/usr/bin/env python3
"""CLI of the BBC-CV after adaptive HPO validation benchmark (CI gate).

Run from the repository root:

    python benchmarks/run_bbc_adaptive_hpo_benchmark.py [--quick] [--ci]

For every dataset case the benchmark runs a full adaptive HPO (Optuna TPE)
with OOF collection, computes the BBC-CV estimate, and compares it with a
repeated external hold-out estimate of the SELECTED configuration on fresh
data from the same generator — over multiple random seeds.

This is a MANDATORY CI gate (the Sanity Gate pattern): the acceptance
criteria of ``check_bbc_acceptance`` are

* |median gap (BBC-CV − external)| ≤ 0.025 per case — no systematic
  optimism/pessimism of BBC-CV after adaptive HPO (the median of the
  pooled per-seed gaps absorbs seed luck; no z criterion — an SE from a
  handful of seeds is too noisy for a mandatory gate);
* the signal-free case must not produce a positive mean BBC-CV estimate
  (a positive estimate on pure noise is pure selection bias).

Prints the table "dataset × cv_best × BBC-CV × external × gap", the
per-case verdict, and the gate result. Return code: 0 — the gate passed;
1 — ``ACCEPTANCE FAILED`` (the CI run fails).

Profiles:

* ``--ci`` — the CI gate profile (ridge cases plus the signal-free
  case, n=3000, 10 Optuna trials, 15 external repeats, one pooled run
  over 6 seeds per case, ~90 s);
* default and ``--quick`` — the heavier manual profiles for evidence
  collection (the gate is checked for them as well).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Repository root — so that imports of benchmarks.* work regardless of the
# current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bbc_adaptive_hpo_benchmark import (
    CI_BBC_CASES,
    CI_SEEDS,
    DatasetSpec,
    check_bbc_acceptance,
    render_bbc_table,
    run_all_bbc_cases,
    run_bbc_case,
    summarize_bbc_verdict,
)


def _quick_specs() -> list[DatasetSpec]:
    """The cheap manual profile (fewer seeds and trials)."""
    return [
        DatasetSpec(
            name="linear_20feat_5noise",
            n_samples=150,
            n_features=20,
            n_informative=5,
            noise=25.0,
            algo="ridge",
            n_trials=10,
        ),
        DatasetSpec(
            name="nonlinear_60feat_50noise",
            n_samples=150,
            n_features=60,
            n_informative=10,
            noise=25.0,
            algo="random_forest",
            n_trials=10,
        ),
    ]


def main() -> int:
    """Run the benchmark, print the table, verdict and gate, return the code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="cheap manual profile: fewer seeds and trials",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="the CI gate profile: ridge cases + the signal-free case, "
        "n=3000, 10 trials, 15 external repeats, one pooled run over "
        "6 seeds per case (bounded runtime)",
    )
    args = parser.parse_args()

    logging.getLogger("optuna").setLevel(logging.WARNING)

    if args.ci:
        # The gate statistic is the MEDIAN of the 6 pooled per-seed gaps
        # per case (one run over CI_SEEDS): a single unlucky seed cannot
        # fail the gate, while a systematic BBC-CV shift beyond the
        # allowance is caught. Calibrated on 15 independent 6-seed
        # windows — 0 false FAILs on the current engine.
        results = [run_bbc_case(spec, CI_SEEDS) for spec in CI_BBC_CASES]
    else:
        if args.quick:
            specs = _quick_specs()
        else:
            specs = None  # the default case set of run_all_bbc_cases
        results = run_all_bbc_cases(specs=specs)

    print("BBC-CV after adaptive HPO: validation benchmark (CI gate)")
    print("Estimates compared per seed: best-trial CV score (optimistic),")
    print("BBC-CV over the trial OOF matrix, and a repeated external")
    print("hold-out of the SELECTED configuration on fresh data (reference).")
    print("Gate: |median gap| <= 0.025 per case, and no positive BBC-CV")
    print("estimate on the signal-free case.")
    if args.ci:
        print("CI statistic: the median of the 6 pooled per-seed gaps of")
        print("each case (one run over CI_SEEDS) — one unlucky seed cannot")
        print("fail the gate, a systematic BBC-CV shift is caught.\n")

    print(render_bbc_table(results))
    print()
    print("Verdict per dataset (positive gap = BBC-CV more optimistic than")
    print("the external reference):")
    print(summarize_bbc_verdict(results))
    print()
    passed, errors = check_bbc_acceptance(results)
    if passed:
        print("ACCEPTANCE: passed — BBC-CV stays consistent with the external")
        print("reference after the adaptive HPO.")
        return 0
    print("ACCEPTANCE FAILED — BBC-CV is inconsistent with the external")
    print("reference after the adaptive HPO (possible selection-bias")
    print("regression in HPO/metrics):")
    for error in errors:
        print(f"  - {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
