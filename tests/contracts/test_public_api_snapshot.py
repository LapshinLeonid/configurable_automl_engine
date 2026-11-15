"""Golden snapshot of the public ``automl_contracts`` API.

The snapshot pins module exports, model fields, enum members, Protocol and
function signatures, constants and the canonical defining module of every
public symbol.  A change is a contract change: it is allowed only with an
explicit golden update, a ``CHANGELOG.md`` entry and a SemVer version bump.
"""

from __future__ import annotations

import importlib
import itertools
import json
from pathlib import Path

import automl_contracts
import pytest

from tests.contracts._contract_snapshot import (
    API_REFERENCE_DOC,
    PUBLIC_API_GOLDEN,
    RELEASE_HISTORY,
    build_public_api,
    build_snapshot,
    contract_public_names,
    load_golden,
    load_release_history,
    parse_documented_public_modules,
    parse_documented_public_symbols,
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


def test_documented_submodules_match_the_api_reference() -> None:
    golden = load_golden()
    assert set(golden["modules"]) == parse_documented_public_modules()
    assert API_REFERENCE_DOC.is_file()
    assert "## Public imports" in API_REFERENCE_DOC.read_text(encoding="utf-8")


def test_documented_public_symbols_match_the_contract() -> None:
    """Every public contract name is documented in the API reference."""
    documented = parse_documented_public_symbols()
    assert documented, "the API reference documents no public symbols"
    unexpected = sorted(contract_public_names() - set(documented))
    assert unexpected == [], (
        f"public but undocumented in API_REFERENCE.md: {unexpected}"
    )


@pytest.mark.parametrize(
    "name,module_name", sorted(parse_documented_public_symbols().items())
)
def test_documented_symbol_is_importable_from_its_module(
    name: str, module_name: str
) -> None:
    module = importlib.import_module(module_name)
    assert hasattr(module, name), f"{module_name}.{name} is documented but missing"


def test_protocols_are_documented_and_structural() -> None:
    from automl_contracts.protocols import MetricFunction, ScorerCallable

    assert MetricFunction._is_protocol
    assert ScorerCallable._is_protocol
    assert automl_contracts.MetricFunction is MetricFunction
    assert automl_contracts.ScorerCallable is ScorerCallable
    call = load_golden()["symbols"]["MetricFunction"]["call"]
    assert "**kwargs" in call, (
        "a keyword-only/positional change must alter the snapshot"
    )


def _semver_violation(history: dict[str, dict]) -> str | None:
    from packaging.version import Version

    versions = [Version(version) for version in history]
    if versions != sorted(versions):
        return "release history is not sorted"
    for previous_version, current_version in itertools.pairwise(versions):
        previous = history[str(previous_version)]
        current = history[str(current_version)]
        if previous["api_digest"] == current["api_digest"]:
            continue
        if previous["schema_version"] != current["schema_version"]:
            continue
        if current_version.major > previous_version.major:
            continue
        if (
            current_version.major == previous_version.major
            and current_version.minor > previous_version.minor
        ):
            continue
        return (
            "A changed public API must bump MINOR or MAJOR, never PATCH: "
            f"{previous_version} -> {current_version}"
        )
    return None


def test_release_history_pins_the_current_api_digest() -> None:
    golden = load_golden()
    history = load_release_history()
    current = history[golden["contract_version"]]
    assert current["schema_version"] == golden["schema_version"]
    assert current["api_digest"] == golden["api_digest"]
    assert _semver_violation(history) is None


@pytest.mark.parametrize(
    ("history", "violation"),
    [
        (
            {
                "0.1.0": {"schema_version": 2, "api_digest": "a"},
                "0.2.0": {"schema_version": 2, "api_digest": "b"},
            },
            False,
        ),
        (
            {
                "0.1.0": {"schema_version": 2, "api_digest": "a"},
                "0.1.1": {"schema_version": 2, "api_digest": "b"},
            },
            True,
        ),
        (
            {
                "1.0.0": {"schema_version": 2, "api_digest": "a"},
                "1.1.0": {"schema_version": 2, "api_digest": "b"},
            },
            False,
        ),
        (
            {
                "1.0.0": {"schema_version": 2, "api_digest": "a"},
                "1.0.1": {"schema_version": 2, "api_digest": "b"},
            },
            True,
        ),
        (
            {
                "0.1.0": {"schema_version": 1, "api_digest": "a"},
                "0.1.1": {"schema_version": 2, "api_digest": "b"},
            },
            False,
        ),
        (
            {
                "0.2.0": {"schema_version": 2, "api_digest": "a"},
                "0.1.0": {"schema_version": 2, "api_digest": "b"},
            },
            True,
        ),
    ],
)
def test_semver_rule_is_exercised_on_synthetic_history(
    history: dict[str, dict], violation: bool
) -> None:
    assert (_semver_violation(history) is not None) is violation


def test_snapshot_is_json_and_contains_every_public_symbol() -> None:
    golden = load_golden()
    json.dumps(golden)
    public = contract_public_names()
    assert public <= set(golden["symbols"])
    assert "__version__" not in golden["symbols"]
    assert PUBLIC_API_GOLDEN.read_text(encoding="utf-8").endswith("\n")
    assert Path(PUBLIC_API_GOLDEN).parent.name == "golden"
    assert RELEASE_HISTORY.is_file()
