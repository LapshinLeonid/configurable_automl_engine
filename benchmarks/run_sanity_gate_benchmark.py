#!/usr/bin/env python3
"""CLI of the Sanity Gate regression benchmark on the target pipeline.

Run from the repository root:

    python benchmarks/run_sanity_gate_benchmark.py

Prints a table "dataset × variant (baseline/target) × winner × whether the
gate triggered" and checks the acceptance criteria:

* reference datasets (noisy R²≈0.3, wide, small N) — the target DAG winner
  matches the CV leader, the gate does not trigger;
* degenerate datasets (flat shelf, tanh plateau, overfitting) — 100% of
  winner changes: the degenerate CV leader is disqualified by the gate.

Return code: 0 — criteria met; 1 — violation (CI blocks acceptance).

Datasets are fixed (seed, generator): the run is reproducible and sensitive
real data is not added to git.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Repository root — so that imports of tests.data_factory and benchmarks.* work
# regardless of the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.sanity_gate_benchmark import (  # noqa: E402
    check_acceptance,
    render_table,
    run_all,
)


def main() -> int:
    """Run the benchmark, print the table, and return the result code."""
    # Optuna INFO logs (every trial) clutter the table — keep only warnings
    # and errors. Engine logs at WARNING (gate disqualifications) are kept:
    # they show who was filtered out and why.
    logging.getLogger("optuna").setLevel(logging.WARNING)

    print("Sanity Gate: regression benchmark of pipeline variants")
    print("Variants: baseline = all_algorithms + select(cv_best, winner);")
    print("          target   = all_algorithms + select(sanity_gate, top_k)"
          " + refine + select(sanity_gate, winner)")
    print("Criterion: reference datasets — 0 winner changes; "
          "degenerate — 100% of changes.\n")
    results, passed, errors = run_all()
    print(render_table(results))
    print()
    if passed:
        print("ACCEPTANCE: all criteria met (reference datasets without "
              "winner changes, degenerate ones with a change).")
        return 0
    print("ACCEPTANCE FAILED:")
    for error in errors:
        print(f"  - {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
