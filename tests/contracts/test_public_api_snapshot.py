"""Golden snapshot of the public ``automl_contracts`` API.

The snapshot pins module exports, model fields, enum members, Protocol and
function signatures, constants and the canonical defining module of every
public symbol.  A change is a contract change: it is allowed only with an
explicit golden update, a ``CHANGELOG.md`` entry and a SemVer version bump.
"""

from __future__ import annotations

import enum
import itertools
import json
import types
from pathlib import Path
from typing import Annotated, Literal, Union, get_origin

import automl_contracts
import pytest
from pydantic import BaseModel

from tests.contracts._contract_snapshot import (
    ADR_DOC,
    PUBLIC_API_GOLDEN,
    RELEASE_HISTORY,
    build_public_api,
    build_snapshot,
    contract_public_names,
    load_golden,
    load_release_history,
    parse_adr_public_submodules,
    parse_contracts_matrix,
    read_changelog_latest_version,
    read_contract_version,
)

REGENERATE = (
    "Public API changed. Review the diff, then regenerate the snapshot with "
    "`python -m tests.contracts.update_golden`, add an entry to "
    "`automl-contracts/CHANGELOG.md` and bump the contract version in "
    "`automl-contracts/pyproject.toml` according to SemVer."
)


def _snapshot_without_metadata(snapshot: dict) -> dict:
    return {key: value for key, value in snapshot.items() if key != "api_digest"}


def test_public_api_matches_the_golden_file() -> None:
    actual = build_snapshot()
    golden = load_golden()
    assert actual == golden, REGENERATE


def test_snapshot_versions_are_synchronized() -> None:
    golden = load_golden()
    assert golden["contract_version"] == automl_contracts.__version__
    assert golden["pyproject_version"] == read_contract_version()
    assert golden["changelog_version"] == read_changelog_latest_version()
    assert (
        golden["contract_version"]
        == golden["pyproject_version"]
        == golden["changelog_version"]
    )


def test_api_digest_is_current() -> None:
    from tests.contracts._contract_snapshot import api_digest

    golden = load_golden()
    assert api_digest(build_public_api()) == golden["api_digest"], REGENERATE


def test_documented_submodules_match_the_adr() -> None:
    golden = load_golden()
    assert (
        set(golden["modules"]) == {"automl_contracts"} | parse_adr_public_submodules()
    )
    assert ADR_DOC.is_file()
    assert "### 7.2." in ADR_DOC.read_text(encoding="utf-8")


def test_documented_public_matrix_matches_the_contract() -> None:
    """Documented public matrix (public/deprecated rows) and the live API agree."""
    documented = parse_contracts_matrix()
    assert documented, "the CONTRACTS.md public matrix is empty"
    documented.setdefault("MetricFunction", "`Protocol`")
    documented.setdefault("ScorerCallable", "`Protocol`")
    actual = contract_public_names()
    missing = sorted(set(documented) - actual)
    unexpected = sorted(actual - set(documented))
    assert missing == [], f"documented but missing in the contract: {missing}"
    assert unexpected == [], f"public but undocumented in CONTRACTS.md: {unexpected}"


@pytest.mark.parametrize("name,type_text", sorted(parse_contracts_matrix().items()))
def test_documented_symbol_type_matches_the_matrix(name: str, type_text: str) -> None:
    module = __import__("automl_contracts")
    value = getattr(module, name, None)
    if value is None:
        value = getattr(automl_contracts.preprocessing, name)
    lowered = type_text.lower()
    if "pydantic-модель" in lowered:
        assert isinstance(value, type) and issubclass(value, BaseModel), name
    elif "enum" in lowered:
        assert isinstance(value, type) and issubclass(value, enum.Enum), name
    elif "exception" in lowered:
        assert isinstance(value, type) and issubclass(value, BaseException), name
    elif "literal" in lowered:
        assert get_origin(value) is Literal, name
    elif "typealias" in lowered:
        assert (
            get_origin(value) in (Annotated, Literal, Union, types.UnionType)
            or value is str
            or isinstance(value, type)
        ), name
    elif "protocol" in lowered:
        assert getattr(value, "_is_protocol", False), name
    elif "list[str]" in lowered:
        assert isinstance(value, list), name
    elif "dict" in lowered:
        assert isinstance(value, dict), name
    elif type_text.strip() == "`float`":
        assert isinstance(value, float), name
    elif type_text.strip() == "`str` (код ошибки)":
        assert isinstance(value, str), name


def test_protocols_are_documented_and_structural() -> None:
    from automl_contracts.protocols import MetricFunction, ScorerCallable

    assert MetricFunction._is_protocol
    assert ScorerCallable._is_protocol
    assert automl_contracts.MetricFunction is MetricFunction
    assert automl_contracts.ScorerCallable is ScorerCallable


def test_release_history_pins_the_current_api_digest() -> None:
    from packaging.version import Version

    golden = load_golden()
    history = load_release_history()
    assert history[golden["contract_version"]] == golden["api_digest"]
    versions = [Version(version) for version in history]
    assert versions == sorted(versions)
    for previous_version, current_version in itertools.pairwise(versions):
        previous_digest = history[str(previous_version)]
        current_digest = history[str(current_version)]
        if previous_digest == current_digest:
            continue
        assert current_version.major > previous_version.major or (
            current_version.major == previous_version.major
            and current_version.minor > previous_version.minor
        ), (
            "A changed public API must bump MINOR or MAJOR, never PATCH: "
            f"{previous_version} -> {current_version}"
        )


def test_snapshot_is_json_and_contains_every_public_symbol() -> None:
    golden = load_golden()
    json.dumps(golden)
    public = contract_public_names()
    assert public <= set(golden["symbols"])
    assert "__version__" not in golden["symbols"]
    assert PUBLIC_API_GOLDEN.read_text(encoding="utf-8").endswith("\n")
    assert Path(PUBLIC_API_GOLDEN).parent.name == "golden"
    assert RELEASE_HISTORY.is_file()
