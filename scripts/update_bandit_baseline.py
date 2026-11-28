#!/usr/bin/env python3
"""Regenerate the Bandit baseline used by the CI security gate.

The script runs Bandit over the shipped packages and stores the fields that
Bandit's baseline loader needs to rebuild each issue: the identity fields plus
the line context required by the loader. Keeping the file to those fields makes
the checked-in baseline reviewable and reproducible with the pinned Bandit
version.

Bandit matches an issue by ``(filename, test_id, text, severity, confidence,
cwe)`` and ignores line numbers, so a new occurrence of an already-baselined
check in the same file is not detected by the baseline gate.

Usage:
    python scripts/update_bandit_baseline.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import bandit

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / ".bandit-baseline.json"
TARGETS = ("src", "configurable-automl-contracts/src")
PINNED_BANDIT_VERSION = "1.9.4"
REQUIRED_FIELDS = (
    "filename",
    "test_name",
    "test_id",
    "issue_severity",
    "issue_cwe",
    "issue_confidence",
    "issue_text",
    "line_number",
    "line_range",
    "code",
)


def _bandit_version() -> str:
    """Return the installed Bandit version if it matches the pinned one.

    Returns:
        The installed Bandit version.

    Raises:
        RuntimeError: If the installed version differs from the pinned one.
    """
    version = bandit.__version__
    if version != PINNED_BANDIT_VERSION:
        raise RuntimeError(
            f"Bandit {PINNED_BANDIT_VERSION} is required to regenerate the "
            f"baseline, found {version}"
        )
    return version


def _scan() -> dict[str, Any]:
    """Run Bandit over the target packages and return its JSON report.

    Returns:
        The parsed Bandit JSON report.

    Raises:
        RuntimeError: If Bandit does not produce a report file.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        report_path = Path(handle.name)

    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "bandit",
                "-r",
                *TARGETS,
                "-ll",
                "-f",
                "json",
                "-o",
                str(report_path),
            ],
            cwd=ROOT,
            check=False,
        )
        if not report_path.exists():
            raise RuntimeError(
                f"Bandit did not produce a report (exit code {completed.returncode})"
            )
        return json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        report_path.unlink(missing_ok=True)


def main() -> int:
    """Write the compact baseline and return a process exit code.

    Returns:
        ``0`` after the baseline has been written.
    """
    version = _bandit_version()
    report = _scan()
    results = [
        {field: result[field] for field in REQUIRED_FIELDS}
        for result in report["results"]
    ]
    BASELINE.write_text(
        json.dumps({"results": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(results)} baseline entries to {BASELINE} (Bandit {version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
