"""Verify built contract and engine artifacts and their metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.contract_boundaries import (
    CONTRACT_PYPROJECT,
    inspect_contract_sdist,
    inspect_contract_wheel,
    inspect_engine_artifacts,
    load_pyproject,
)


def main(argv: list[str] | None = None) -> int:
    """Inspect the requested artifacts and report every violation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-wheel", type=Path)
    parser.add_argument("--contract-sdist", type=Path)
    parser.add_argument("--engine-wheel", type=Path)
    parser.add_argument("--engine-sdist", type=Path)
    parser.add_argument("--contract-pyproject", type=Path, default=CONTRACT_PYPROJECT)
    parser.add_argument("--expected-version")
    arguments = parser.parse_args(argv)

    artifacts = (
        arguments.contract_wheel,
        arguments.contract_sdist,
        arguments.engine_wheel,
        arguments.engine_sdist,
    )
    if all(artifact is None for artifact in artifacts):
        parser.error("at least one artifact path is required")

    expected_version = arguments.expected_version
    if expected_version is None:
        document = load_pyproject(arguments.contract_pyproject)
        project = document.get("project", {})
        expected_version = project.get("version") if isinstance(project, dict) else None

    violations = []
    if arguments.contract_wheel is not None:
        violations.extend(
            inspect_contract_wheel(arguments.contract_wheel, expected_version)
        )
    if arguments.contract_sdist is not None:
        violations.extend(
            inspect_contract_sdist(arguments.contract_sdist, expected_version)
        )
    violations.extend(
        inspect_engine_artifacts(arguments.engine_wheel, arguments.engine_sdist)
    )
    if not violations:
        print("OK: distribution artifacts satisfy the contract boundary")
        return 0
    for violation in violations:
        print(violation.format(REPO_ROOT), file=sys.stderr)
    print(
        f"FAIL: {len(violations)} artifact violation(s)",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
