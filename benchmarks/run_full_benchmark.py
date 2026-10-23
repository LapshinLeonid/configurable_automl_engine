#!/usr/bin/env python3
"""CLI of the FULL complexity guard acceptance benchmark (outside CI).

The acceptance profile of the task: 5 small datasets x 5+ seeds x 30 trials,
with the guard off and on — mean OOF and std over seeds for every algorithm,
plus the XGBoost HPO time. Run this script separately (NOT in CI) and post
its report to the pull request:

    python benchmarks/run_full_benchmark.py

Prints:

* the acceptance table "dataset x algorithm x mean_off x mean_on x std_off x
  std_on x xgb_time_off/xgb_time_on";
* the acceptance criteria verdict; return code 0 — met, 1 — violated.

The run is heavy (about 1-1.5 hours on one core) — do not add it to
CI. ``benchmarks/run_complexity_guard_benchmark.py`` remains the CI smoke
profile (2 datasets, n_trials=3, a few seeds).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import optuna

# Repository root — so that imports of benchmarks.* work regardless of the
# current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.complexity_guard_benchmark import (
    FULL_DATASETS,
    check_full_acceptance,
    check_full_profile_shape,
    render_table,
    run_case,
)


def main() -> int:
    """Run the full benchmark, print the report, and return the result code."""
    # Optuna INFO logs (every trial) clutter the table — keep only warnings
    # and errors. The engine may reset the optuna logger verbosity during
    # optimize(), so mute both the stdlib logger and optuna's own verbosity.
    logging.getLogger("optuna").setLevel(logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Fail fast on a malformed FULL profile BEFORE the ~1-2 h run: the
    # acceptance gates are only meaningful on the task's profile (>= 5
    # datasets, >= 5 seeds, 30-50 trials).
    profile_errors = check_full_profile_shape(FULL_DATASETS)
    if profile_errors:
        print("FULL profile validation FAILED:")
        for error in profile_errors:
            print(f"  - {error}")
        return 1

    print("Complexity Guard: FULL acceptance benchmark (outside CI)")
    print("Variants: baseline = guard off; target   = guard apply")
    print(
        f"Datasets: {len(FULL_DATASETS)}; seeds per dataset: "
        f"{len(FULL_DATASETS[0].seeds)} (>= 5); trials per algorithm: "
        f"{FULL_DATASETS[0].n_trials}\n"
    )

    started = time.monotonic()
    results = [run_case(ds) for ds in FULL_DATASETS]
    elapsed = time.monotonic() - started

    print(render_table(results))
    print()
    print(f"Total run time: {elapsed / 60:.1f} min\n")

    passed, errors = check_full_acceptance(results)
    if passed:
        print("ACCEPTANCE: all criteria met.")
    else:
        print("ACCEPTANCE FAILED:")
        for error in errors:
            print(f"  - {error}")

    report = {
        "profile": {
            "datasets": len(FULL_DATASETS),
            "seeds_per_dataset": len(FULL_DATASETS[0].seeds),
            "trials_per_algorithm": FULL_DATASETS[0].n_trials,
        },
        "elapsed_sec": elapsed,
        "datasets": [
            {
                "name": r.dataset,
                "n_restricted": r.n_restricted,
                "algorithms": [
                    {
                        "algo": a.algo,
                        "scores_off": a.scores_off,
                        "scores_on": a.scores_on,
                        "xgb_time_off": a.xgb_time_off,
                        "xgb_time_on": a.xgb_time_on,
                    }
                    for a in r.algorithms
                ],
            }
            for r in results
        ],
        "accepted": not errors,
        "errors": errors,
    }
    out = Path(__file__).resolve().parent / "full_benchmark_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport saved to {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
