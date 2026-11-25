"""Guards for coverage collection and the CI reporting pipeline."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_pyproject() -> dict[str, Any]:
    """Return the parsed root ``pyproject.toml``."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _load_workflow(path: Path) -> dict[str, Any]:
    """Return the parsed workflow YAML at ``path``."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _step_runs(steps: list[dict[str, Any]]) -> dict[str, str]:
    """Map every named step to its ``run`` script."""
    return {step["name"]: step.get("run", "") for step in steps}


def test_coverage_run_collects_branches_and_subprocesses() -> None:
    """The run configuration covers branches, threads and process workers."""
    run = _load_pyproject()["tool"]["coverage"]["run"]

    assert run["branch"] is True
    assert run["source"] == ["src"]
    assert run["parallel"] is True
    assert set(run["concurrency"]) == {"multiprocessing", "thread"}
    assert run["patch"] == ["subprocess"]


def test_coverage_threshold_is_ninety_percent() -> None:
    """The report gate fails below 90% coverage."""
    report = _load_pyproject()["tool"]["coverage"]["report"]

    assert report["fail_under"] == 90


def test_github_ci_combines_parallel_data_before_reporting() -> None:
    """GitHub CI merges worker data files and only then writes the report."""
    workflow = _load_workflow(REPO_ROOT / ".github" / "workflows" / "CI.yaml")
    steps = workflow["jobs"]["test"]["steps"]
    runs = _step_runs(steps)
    names = [step["name"] for step in steps]

    assert "coverage run --parallel-mode -m pytest" in runs["Run tests"]
    assert "coverage combine" in runs["Combine coverage data"]
    assert "coverage report" in runs["Report coverage"]
    assert "coverage xml" in runs["Write coverage XML"]
    assert names.index("Combine coverage data") < names.index("Report coverage")
    assert names.index("Report coverage") < names.index("Write coverage XML")


def test_sourcecraft_ci_combines_parallel_data_before_reporting() -> None:
    """SourceCraft test cubes defer reporting until the data is merged."""
    workflow = _load_workflow(REPO_ROOT / ".sourcecraft" / "ci.yaml")
    cubes = workflow["workflows"]["quality-checks"]["tasks"][0]["cubes"]
    scripts = {cube["name"]: "\n".join(cube["script"]) for cube in cubes}

    assert "coverage run --parallel-mode -m pytest" in scripts["run-tests"]
    assert "coverage run --parallel-mode -m pytest" in scripts["run-refactor-safety-net"]

    report = scripts["coverage-report"]
    assert report.index("coverage combine") < report.index("coverage report")
    assert report.index("coverage report") < report.index("coverage xml")
