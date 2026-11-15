"""Golden JSON Schema, metaschema validity and package-data completeness."""

from __future__ import annotations

import importlib.util
import json
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import automl_contracts
import jsonschema
import pytest
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_SCHEMA = (
    REPO_ROOT
    / "automl-contracts"
    / "src"
    / "automl_contracts"
    / "schemas"
    / "config.schema.json"
)
ROOT_MIRROR = REPO_ROOT / "config.schema.json"
GENERATOR = REPO_ROOT / "automl-contracts" / "scripts" / "generate_schema.py"
DATA_DIR = Path(__file__).resolve().parent / "data"

REGENERATE = (
    "The saved JSON Schema is out of sync with the contract models. "
    "Regenerate it with `python automl-contracts/scripts/generate_schema.py`, "
    "copy the canonical file to the repository root `config.schema.json`, "
    "review the compatibility impact and update CHANGELOG.md/version."
)


def load_generator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "contract_schema_generator", GENERATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def serialized_schema() -> bytes:
    generator = load_generator()
    return json.dumps(generator.build_schema(), indent=2, ensure_ascii=False).encode(
        "utf-8"
    )


def walk_schema(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk_schema(value)
    elif isinstance(node, list):
        for item in node:
            yield from walk_schema(item)


def model_classes(annotation: Any) -> Iterator[type[BaseModel]]:
    from typing import Annotated, Union, get_args, get_origin

    origin = get_origin(annotation)
    if origin is Annotated:
        yield from model_classes(get_args(annotation)[0])
        return
    if origin in (Union, types.UnionType):
        for argument in get_args(annotation):
            yield from model_classes(argument)
        return
    if isinstance(origin, type) and issubclass(origin, BaseModel):
        yield origin
        return
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        metadata = getattr(annotation, "__pydantic_generic_metadata__", {})
        yield metadata.get("origin") or annotation
    for argument in get_args(annotation):
        yield from model_classes(argument)


def reachable_models(root: type[BaseModel]) -> dict[str, type[BaseModel]]:
    seen: dict[str, type[BaseModel]] = {}

    def visit(model: type[BaseModel]) -> None:
        if model.__name__ in seen:
            return
        seen[model.__name__] = model
        for field in model.model_fields.values():
            for nested in model_classes(field.annotation):
                visit(nested)

    visit(root)
    return seen


def test_schema_is_deterministic_and_address_free() -> None:
    first = serialized_schema()
    second = serialized_schema()
    assert first == second
    assert str(REPO_ROOT).encode() not in first
    assert b"/tmp/" not in first


def test_canonical_schema_matches_the_models() -> None:
    actual = serialized_schema()
    assert CANONICAL_SCHEMA.read_bytes() == actual, REGENERATE


def test_root_schema_mirror_matches_the_canonical_schema() -> None:
    canonical = CANONICAL_SCHEMA.read_bytes()
    assert ROOT_MIRROR.read_bytes() == canonical, REGENERATE
    assert not canonical.endswith(b"\n")


def test_schema_is_valid_against_its_metaschema() -> None:
    schema = json.loads(CANONICAL_SCHEMA.read_text(encoding="utf-8"))
    validator_class = jsonschema.validators.validator_for(schema)
    validator_class.check_schema(schema)
    assert validator_class is jsonschema.Draft202012Validator


def test_schema_definitions_and_references_are_complete() -> None:
    schema = json.loads(CANONICAL_SCHEMA.read_text(encoding="utf-8"))
    definitions = schema["$defs"]
    assert definitions

    references = {
        node["$ref"]
        for node in walk_schema(schema)
        if isinstance(node.get("$ref"), str)
    }
    assert references
    for reference in references:
        assert reference.startswith("#/$defs/"), reference
        assert reference.removeprefix("#/$defs/") in definitions, reference

    models = reachable_models(automl_contracts.Config)
    assert set(models) - {"Config"}
    for name in models:
        if name == "Config":
            continue
        assert any(
            definition == name or definition.startswith(f"{name}_")
            for definition in definitions
        ), f"{name} is reachable from Config but missing from $defs"

    # Abstract helpers that are intentionally not part of the Config schema
    # must not leak into it as standalone definitions.
    assert "BaseDistribution" not in definitions
    assert "NumericSpace" not in definitions


def test_schema_contains_the_public_config_enums() -> None:
    schema = json.loads(CANONICAL_SCHEMA.read_text(encoding="utf-8"))
    definitions = schema["$defs"]
    for name in (
        "ValidationStrategy",
        "SerializationFormat",
        "TargetTransform",
        "ParallelStrategy",
        "PruningStrategy",
        "ComplexityGuardMode",
        "ComplexityGuardUserRangePolicy",
        "ComplexityGuardSizeBasis",
        "FeatureSelectionMode",
        "FeatureSelectionMethod",
        "OversamplingAlgorithm",
    ):
        assert name in definitions, name


def test_schema_accepts_the_reference_configs() -> None:
    schema = json.loads(CANONICAL_SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    example = json.loads((DATA_DIR / "example_config.json").read_text(encoding="utf-8"))
    reference = json.loads(
        (DATA_DIR / "reference_config.json").read_text(encoding="utf-8")
    )
    # The shorthand search-space lists are a Pydantic input convenience; the
    # JSON Schema describes the canonical model form, so validate the dump.
    reference = automl_contracts.Config.model_validate(reference).model_dump(
        mode="json", by_alias=True
    )
    for payload in (example, reference):
        errors = sorted(
            validator.iter_errors(payload), key=lambda error: list(error.path)
        )
        assert errors == [], [error.message for error in errors]


def test_schema_rejects_an_unknown_section() -> None:
    schema = json.loads(CANONICAL_SCHEMA.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    payload = json.loads((DATA_DIR / "example_config.json").read_text(encoding="utf-8"))
    payload["unknown_section"] = {}
    assert list(validator.iter_errors(payload))


def test_schema_package_data_matches_the_repository_artifact() -> None:
    from importlib.resources import files

    package_schema = files("automl_contracts").joinpath("schemas/config.schema.json")
    assert package_schema.is_file()
    assert package_schema.read_bytes() == CANONICAL_SCHEMA.read_bytes()


def test_generator_import_has_no_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    load_generator()
    assert list(tmp_path.iterdir()) == []
