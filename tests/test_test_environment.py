"""The ``test`` extra must describe a self-sufficient development environment.

Installing ``pip install -e ".[test]"`` into a clean environment has to be
enough to run the test suite, the linters and ``mypy src``: the extra must
therefore declare the test runner, the runtime type stubs, the lint, coverage
and import-linter tooling and the optional ``xgboost`` dependency used by the
benchmarks and the core package.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
PROJECT_NAME = "configurable-automl-engine"

REQUIRED_TEST_DEPENDENCIES = frozenset(
    {
        "pytest",
        "pytest-mock",
        "pytest-cov",
        "import-linter",
        "jsonschema",
        "mypy",
        "ruff",
        "types-pyyaml",
        "pandas-stubs",
        "scipy-stubs",
        "xgboost",
    }
)

REQUIREMENT_RE = re.compile(
    r"\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[(?P<extras>[^\]]*)\])?"
)


def _canonical_name(name: str) -> str:
    """Return the canonical PEP 503 form of a distribution or extra name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_parts(specifier: str) -> tuple[str, frozenset[str]]:
    """Split a PEP 508 requirement into a canonical name and extras.

    Args:
        specifier: Requirement string, possibly with extras, version
            specifiers and an environment marker.

    Returns:
        The canonical distribution name and the canonical names of the
        requested extras.

    Raises:
        AssertionError: If the requirement does not start with a name.
    """
    match = REQUIREMENT_RE.match(specifier)
    assert match is not None, f"unparsable requirement: {specifier!r}"
    extras = match.group("extras") or ""
    return _canonical_name(match.group("name")), frozenset(
        _canonical_name(extra.strip()) for extra in extras.split(",") if extra.strip()
    )


def _requirement_name(specifier: str) -> str:
    """Return the canonical distribution name of a PEP 508 requirement."""
    return _requirement_parts(specifier)[0]


def _expand_test_extra(
    test_extra: list[str], optional_dependencies: dict[str, list[str]]
) -> set[str]:
    """Return every distribution the ``test`` extra installs.

    Self-referential requirements such as
    ``configurable-automl-engine[xgboost]`` are expanded into the engine's own
    extras, so the result mirrors what a clean ``pip install -e ".[test]"``
    actually pulls in.

    Args:
        test_extra: Requirement strings declared by the ``test`` extra.
        optional_dependencies: All optional dependency groups of the project.

    Returns:
        Canonical names of the distributions provided by the ``test`` extra.

    Raises:
        AssertionError: If a self-reference points to an unknown extra.
    """
    by_canonical_extra = {
        _canonical_name(extra): requirements
        for extra, requirements in optional_dependencies.items()
    }
    provided: set[str] = set()
    seen: set[tuple[str, frozenset[str]]] = set()
    pending = list(test_extra)

    while pending:
        name, extras = _requirement_parts(pending.pop())
        if (name, extras) in seen:
            continue
        seen.add((name, extras))
        provided.add(name)
        if name == _canonical_name(PROJECT_NAME):
            unknown = extras - by_canonical_extra.keys()
            assert not unknown, (
                f"the `test` extra references unknown project extras: {sorted(unknown)}"
            )
            for extra in extras:
                pending.extend(by_canonical_extra[extra])

    return provided


def test_required_packages_are_declared_in_test_extra() -> None:
    """Every reproducibility dependency is part of the ``test`` extra."""
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    optional_dependencies = config["project"]["optional-dependencies"]

    provided = _expand_test_extra(optional_dependencies["test"], optional_dependencies)
    missing = REQUIRED_TEST_DEPENDENCIES - provided

    assert not missing, (
        "the `test` extra is missing required reproducibility dependencies: "
        f"{sorted(missing)}"
    )


@pytest.mark.parametrize(
    ("specifier", "expected"),
    [
        ("xgboost>=3.1.3", "xgboost"),
        ("Pandas_Stubs", "pandas-stubs"),
        ("types-pyYAML ; python_version>'3'", "types-pyyaml"),
        ("Foo_Bar.Baz>=1", "foo-bar-baz"),
    ],
)
def test_requirement_name_normalizes_names(specifier: str, expected: str) -> None:
    """Names are compared independently of case and separator style."""
    assert _requirement_name(specifier) == expected


def test_requirement_parts_extracts_extras() -> None:
    """Extras are canonicalized together with the distribution name."""
    name, extras = _requirement_parts("configurable-automl-engine[Xgboost, Logging]>=1")

    assert name == "configurable-automl-engine"
    assert extras == {"xgboost", "logging"}


def test_self_referential_requirement_expands_project_extras() -> None:
    """A self-reference contributes the dependencies of the referenced extra."""
    optional_dependencies = {
        "xgboost": ["xgboost>=3.1.3"],
        "test": ["configurable-automl-engine[xgboost]"],
    }

    provided = _expand_test_extra(optional_dependencies["test"], optional_dependencies)

    assert {"configurable-automl-engine", "xgboost"} <= provided


def test_unknown_self_reference_extra_is_rejected() -> None:
    """A typo in the referenced extra name is reported instead of ignored."""
    optional_dependencies = {"test": ["configurable-automl-engine[missing]"]}

    with pytest.raises(AssertionError, match="unknown project extras"):
        _expand_test_extra(optional_dependencies["test"], optional_dependencies)


def test_unparsable_requirement_is_rejected() -> None:
    """A requirement without a distribution name cannot be parsed."""
    with pytest.raises(AssertionError, match="unparsable requirement"):
        _requirement_name(">=1.0")
