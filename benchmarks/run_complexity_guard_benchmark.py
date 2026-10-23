#!/usr/bin/env python3
"""CLI of the complexity guard regression benchmark on small datasets.

Run from the repository root:

    python benchmarks/run_complexity_guard_benchmark.py

Prints a table "dataset × algorithm × mean_off × mean_on × std × xgb_time"
and checks the acceptance criteria:

* the mean CV score of trees/boosting with the guard is NOT worse than
  without it;
* the scatter between seeds is not dramatically larger with the guard;
* the guard restricts at least one parameter on every dataset.

Return code: 0 — criteria met; 1 — violation (CI blocks acceptance).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import optuna

# Repository root — so that imports of benchmarks.* work regardless of the
# current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.complexity_guard_benchmark import (
    SMALL_DATASETS,
    check_acceptance,
    render_table,
    run_case,
)


def main() -> int:
    """Run the benchmark, print the table, and return the result code."""
    # Optuna INFO logs (every trial) clutter the table — keep only warnings
    # and errors. The engine may reset the optuna logger verbosity during
    # optimize(), so mute both the stdlib logger and optuna's own verbosity.
    logging.getLogger("optuna").setLevel(logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("Complexity Guard: regression benchmark on small datasets")
    print("Variants: baseline = guard off; target   = guard apply\n")

    results = [run_case(ds) for ds in SMALL_DATASETS]
    print(render_table(results))
    print()
    passed, errors = check_acceptance(results)
    if passed:
        print("ACCEPTANCE: all criteria met.")
        return 0
    print("ACCEPTANCE FAILED:")
    for error in errors:
        print(f"  - {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
