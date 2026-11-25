"""Version matrix, SemVer synchronization and Pydantic support declarations."""

from __future__ import annotations

import tomllib

import automl_contracts
import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from tests.contracts._contract_snapshot import (
    API_REFERENCE_DOC,
    CONTRACT_PYPROJECT,
    REPO_ROOT,
)


def contract_project() -> dict:
    return tomllib.loads(CONTRACT_PYPROJECT.read_text(encoding="utf-8"))["project"]


def engine_project() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]


def contract_dependency_specifier() -> SpecifierSet:
    dependencies = engine_project()["dependencies"]
    for dependency in dependencies:
        if str(dependency).startswith("configurable-automl-contracts"):
            return SpecifierSet(
                str(dependency).split("configurable-automl-contracts", 1)[1]
            )
    raise AssertionError(
        "the engine must declare a configurable-automl-contracts dependency"
    )


def test_contract_version_is_synchronized() -> None:
    version = contract_project()["version"]
    assert automl_contracts.__version__ == version


def test_engine_declares_the_documented_contract_range() -> None:
    specifier = contract_dependency_specifier()
    version = Version(contract_project()["version"])
    assert specifier.contains(version), (str(specifier), str(version))
    assert specifier == SpecifierSet(">=0.2.0,<0.3.0")
    assert "configurable-automl-contracts>=0.2.0,<0.3.0" in API_REFERENCE_DOC.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("version", "supported"),
    [
        ("0.0.9", False),
        ("0.1.0", False),
        ("0.1.99", False),
        ("0.2.0", True),
        ("0.2.99", True),
        ("0.3.0", False),
        ("1.0.0", False),
    ],
)
def test_unsupported_contract_versions_are_rejected(
    version: str, supported: bool
) -> None:
    specifier = contract_dependency_specifier()
    assert specifier.contains(Version(version)) is supported


def test_the_next_0x_minor_is_a_new_supported_range() -> None:
    version = Version(contract_project()["version"])
    assert version.major == 0
    next_minor = f"0.{version.minor + 1}.0"
    assert not contract_dependency_specifier().contains(Version(next_minor))
    assert next_minor in API_REFERENCE_DOC.read_text(encoding="utf-8")


def test_pydantic_version_is_declared_and_currently_supported() -> None:
    import pydantic

    dependencies = contract_project()["dependencies"]
    pydantic_requires = next(
        dependency
        for dependency in dependencies
        if str(dependency).startswith("pydantic")
    )
    specifier = SpecifierSet(str(pydantic_requires).split("pydantic", 1)[1])
    assert Version("2.12.5") in specifier
    assert Version("3.0.0") not in specifier
    assert Version(pydantic.VERSION) in specifier, (pydantic.VERSION, str(specifier))
    assert contract_project()["requires-python"] == ">=3.11"


def test_semver_policy_is_documented() -> None:
    text = API_REFERENCE_DOC.read_text(encoding="utf-8")
    for requirement in (
        "## Versioning and compatibility",
        "During `0.x`, potentially incompatible changes get a MINOR version",
        "PATCH is reserved for",
        "strictly compatible fixes",
        "The engine declares `configurable-automl-contracts>=0.2.0,<0.3.0`",
    ):
        assert requirement in text
