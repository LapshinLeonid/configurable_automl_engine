#!/usr/bin/env python3
"""CLI of the oversampling evidence benchmark (T13, nightly profile).

Run from the repository root:

    python benchmarks/run_oversampling_benchmark.py

Prints the table "dataset × variant × median held-out R² × Δ vs baseline"
and checks the evidence gate: if any oversampling algorithm lowers the
median held-out R² by more than 0.01, the run reports the degradation (the
feature must then be marked experimental in README/API_REFERENCE).

Return code: 0 — gate passed; 1 — evidence of degradation found.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

# Repository root — so that imports of benchmarks.* work regardless of the
# current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.oversampling_benchmark import (  # noqa: E402
    EVIDENCE_GATE_DELTA,
    check_oversampling_acceptance,
    render_oversampling_table,
    run_all_oversampling,
)


def main() -> int:
    """Run the benchmark, print the table, and return the result code."""
    logging.getLogger("optuna").setLevel(logging.WARNING)

    print("Oversampling: nightly evidence benchmark (T13)")
    print(f"Variants: baseline (no oversampling) vs random/smote/adasyn "
          f"(multiplier 1.5)")
    print(f"Evidence gate: degradation of the median held-out R² by more "
          f"than {EVIDENCE_GATE_DELTA} marks the feature experimental.\n")

    with tempfile.TemporaryDirectory() as tmp:
        results, passed, errors = run_all_oversampling(Path(tmp))
        print(render_oversampling_table(results))
        print()
        if passed:
            print("EVIDENCE GATE: passed — no oversampling algorithm degraded "
                  "the median held-out R² beyond the threshold.")
            return 0
        print("EVIDENCE GATE: FAILED — mark the oversampling feature as "
              "experimental in README/API_REFERENCE:")
        for error in errors:
            print(f"  - {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())