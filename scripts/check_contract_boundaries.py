"""Run the static architecture guards for the contract boundary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.contract_boundaries import run_static_checks


def main(argv: list[str] | None = None) -> int:
    """Print violations and return a non-zero status when any exists."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="repository checkout to inspect",
    )
    arguments = parser.parse_args(argv)
    violations = run_static_checks(arguments.repo_root)
    if not violations:
        print("OK: contract architecture boundaries are intact")
        return 0
    for violation in violations:
        print(violation.format(arguments.repo_root), file=sys.stderr)
    print(
        f"FAIL: {len(violations)} contract architecture violation(s)",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
