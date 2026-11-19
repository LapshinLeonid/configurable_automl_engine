"""Checks that the documented public ``automl_contracts`` API stays importable."""

from __future__ import annotations

import importlib
import inspect
import types
from pathlib import Path
from typing import TypeVar

import automl_contracts
import pytest

from tests.contracts import _contract_snapshot
from tests.contracts._contract_snapshot import (
    API_REFERENCE_DOC,
    PUBLIC_SUBMODULES,
    contract_public_names,
    parse_documented_public_modules,
    parse_documented_public_symbols,
)


def test_documented_submodules_match_the_api_reference() -> None:
    assert set(PUBLIC_SUBMODULES) | {"automl_contracts"} == (
        parse_documented_public_modules()
    )
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


def test_new_metrics_export_must_be_documented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new ``metrics.__all__`` entry shows up as undocumented."""
    metrics = importlib.import_module("automl_contracts.metrics")
    monkeypatch.setattr(
        metrics, "__all__", [*metrics.__all__, "undocumented_metrics_probe"]
    )
    documented = set(parse_documented_public_symbols())
    assert "undocumented_metrics_probe" in contract_public_names() - documented


def test_defined_public_names_reads_the_module_source(tmp_path: Path) -> None:
    """The no-``__all__`` branch reads public definitions from the source AST."""
    source = tmp_path / "probe_module.py"
    source.write_text(
        "import os\n"
        "from typing import TypeVar\n"
        "\n"
        "T = TypeVar('T')\n"
        "_private = 1\n"
        "CONSTANT = 2\n"
        "\n"
        "class PublicClass:\n"
        "    pass\n"
        "\n"
        "def public_function() -> None:\n"
        "    pass\n",
        encoding="utf-8",
    )
    probe = types.SimpleNamespace(__file__=str(source), T=TypeVar("T"))
    assert _contract_snapshot._defined_public_names(probe) == {
        "CONSTANT",
        "PublicClass",
        "public_function",
    }


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
    parameters = inspect.signature(MetricFunction.__call__).parameters
    kwargs = parameters.get("kwargs")
    assert kwargs is not None
    assert kwargs.kind is inspect.Parameter.VAR_KEYWORD
