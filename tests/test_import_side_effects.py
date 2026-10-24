"""Tests that importing the package has no side effects.

Importing ``configurable_automl_engine`` must not emit warnings (in
particular, no ``Pandas4Warning`` from touching the deprecated pandas
``mode.copy_on_write`` option, which is scheduled for removal in pandas 4)
and must not mutate the global pandas options.
"""

import copy
import subprocess
import sys

import pandas as pd


def test_import_with_warnings_as_errors() -> None:
    """Importing the package must not emit any warning."""

    code = "import configurable_automl_engine"
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert (
        result.returncode == 0
    ), f"Import failed with warnings-as-errors:\n{result.stderr}"
    assert result.stderr == "", f"Unexpected stderr on import:\n{result.stderr}"


def test_import_does_not_mutate_pandas_options() -> None:
    """Importing the package must leave the pandas options untouched."""

    before = copy.deepcopy(pd.options.d)
    # Import here, after the snapshot, so the comparison covers any
    # options mutation done at module import time.
    import configurable_automl_engine  # noqa: F401

    after = copy.deepcopy(pd.options.d)
    assert after == before
