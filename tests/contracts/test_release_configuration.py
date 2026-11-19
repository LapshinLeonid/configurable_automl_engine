"""Guards for the CI publication and consumer-check configuration."""

from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_guard() -> ModuleType:
    path = REPO_ROOT / "scripts" / "check_release_configuration.py"
    spec = importlib.util.spec_from_file_location("release_configuration_guard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def guard() -> ModuleType:
    return load_guard()


@pytest.fixture(scope="module")
def sourcecraft_text() -> str:
    return (REPO_ROOT / ".sourcecraft" / "ci.yaml").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def github_publish_text() -> str:
    path = REPO_ROOT / ".github" / "workflows" / "publish-contracts.yaml"
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def github_ci_text() -> str:
    path = REPO_ROOT / ".github" / "workflows" / "CI.yaml"
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def engine_pyproject_text() -> str:
    return (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def messages(violations: list[object]) -> str:
    return "\n".join(violation.message for violation in violations)


def mutate_cubes(text: str, function: object) -> str:
    """Apply a mutation to the publish-contracts cubes and dump valid YAML."""
    document = yaml.safe_load(text)
    cubes = document["workflows"]["publish-contracts"]["tasks"][0]["cubes"]
    function(cubes)
    return yaml.safe_dump(document, sort_keys=False)


def inside(text: str, block: str, old: str, new: str, *, count: int = 1) -> str:
    """Replace ``old`` occurrences after the ``block`` marker."""
    start = text.index(block)
    head, tail = text[:start], text[start:]
    assert old in tail, (block, old)
    return head + tail.replace(old, new, count)


PUBLISH_WORKFLOW = "  publish-contracts:\n"
REGISTRY_ID_LINE = re.compile(
    r"^([ \t]*)SOURCECRAFT_PYPI_REGISTRY_ID:.*$", re.MULTILINE
)
WORKFLOW_HEADER = re.compile(r"^  [A-Za-z0-9_-]+:\s*$", re.MULTILINE)


def publish_workflow_section(text: str) -> tuple[str, str, str]:
    """Split ``text`` around the publish-contracts workflow section."""
    start = text.index(PUBLISH_WORKFLOW)
    match = WORKFLOW_HEADER.search(text, start + len(PUBLISH_WORKFLOW))
    end = match.start() if match is not None else len(text)
    return text[:start], text[start:end], text[end:]


def replace_publish_registry_id(text: str, value: str) -> str:
    """Set the registry identifier of the publish workflow to ``value``."""
    head, section, tail = publish_workflow_section(text)
    replaced, count = REGISTRY_ID_LINE.subn(
        lambda match: f'{match.group(1)}SOURCECRAFT_PYPI_REGISTRY_ID: "{value}"',
        section,
        count=1,
    )
    assert count == 1, "publish-contracts has no registry identifier"
    return head + replaced + tail


def append_to_publish_registry_env(text: str, entry: str) -> str:
    """Insert ``entry`` right after the publish workflow registry identifier."""
    head, section, tail = publish_workflow_section(text)
    replaced, count = REGISTRY_ID_LINE.subn(
        lambda match: f"{match.group(0)}\n{match.group(1)}{entry}",
        section,
        count=1,
    )
    assert count == 1, "publish-contracts has no registry identifier"
    return head + replaced + tail


def load_document(text: str) -> dict[str, object]:
    document = yaml.safe_load(text)
    assert isinstance(document, dict)
    return document


def dump_document(document: dict[str, object]) -> str:
    return yaml.safe_dump(document, sort_keys=False)


def publish_workflow(document: dict[str, object]) -> dict[str, object]:
    return document["workflows"]["publish-contracts"]  # type: ignore[index]


def find_cube(workflow: dict[str, object], name: str) -> dict[str, object]:
    for task in workflow["tasks"]:  # type: ignore[index]
        for cube in task.get("cubes", []):
            if cube.get("name") == name:
                return cube
    raise AssertionError(f"cube {name!r} is missing")


def cube_script(cube: dict[str, object]) -> str:
    script = cube["script"]
    if isinstance(script, list):
        return "\n".join(str(chunk) for chunk in script)
    return str(script)


def mutate_publish_workflow(text: str, function: object) -> str:
    document = load_document(text)
    function(publish_workflow(document))
    return dump_document(document)


def split_publish_tasks(workflow: dict[str, object], *, needs: object = None) -> None:
    cubes = workflow["tasks"][0]["cubes"]  # type: ignore[index]
    boundary = next(
        index for index, cube in enumerate(cubes) if cube["name"] == "publish"
    )
    gates = {"name": "gates", "cubes": cubes[:boundary]}
    upload = {"name": "upload", "cubes": cubes[boundary:]}
    if needs is not None:
        upload["needs"] = needs
    workflow["tasks"] = [gates, upload]


def test_repository_release_configuration_is_wired(guard: ModuleType) -> None:
    assert guard.run_checks(REPO_ROOT) == []


def test_check_scripts_rejects_a_modified_uploader(
    guard: ModuleType, tmp_path: Path
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in (
        "contract_release.py",
        "e2e_contract_consumer.py",
        "smoke_engine_install.py",
        "check_release_configuration.py",
    ):
        shutil.copy(REPO_ROOT / "scripts" / name, scripts / name)
    release = scripts / "contract_release.py"
    text = release.read_text(encoding="utf-8")
    release.write_text(
        text.replace(
            '"twine",\n        "upload",',
            '"twine",\n        "upload",\n        "--skip-existing",',
            1,
        ),
        encoding="utf-8",
    )
    violations = guard.check_scripts(tmp_path)
    assert any("skip it" in violation.message for violation in violations)


def test_sourcecraft_requires_the_contracts_tag_trigger(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace('tags: ["contracts-v*"]', 'tags: ["v*"]')
    assert "contracts-v*" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_requires_a_publish_tag_entry(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        '    - workflows: [publish-contracts]\n      filter:\n        tags: ["contracts-v*"]',
        '    - workflows: [publish-contracts]\n      filter:\n        branches: ["main"]',
        1,
    )
    assert "publish-contracts" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_requires_a_sync_tag_entry(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        '    - workflows: [sync-to-github]\n      filter:\n        tags: ["contracts-v*"]\n',
        "",
        1,
    )
    assert "sync-to-github" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_commented_out_gate(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        "python3.11 scripts/check_release_configuration.py",
        "# python3.11 scripts/check_release_configuration.py",
    )
    assert "check_release_configuration.py" in messages(
        guard.check_sourcecraft_ci(text)
    )


def test_sourcecraft_requires_build_before_publish(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def move_publish_first(cubes: list[dict[str, object]]) -> None:
        publish = next(cube for cube in cubes if cube["name"] == "publish")
        cubes.remove(publish)
        build = next(cube for cube in cubes if cube["name"] == "build-and-verify")
        cubes.insert(cubes.index(build), publish)

    text = mutate_cubes(sourcecraft_text, move_publish_first)
    assert "order is" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_requires_registry_verification_after_publish(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def move_verification_first(cubes: list[dict[str, object]]) -> None:
        verification = next(
            cube for cube in cubes if cube["name"] == "verify-from-registry"
        )
        cubes.remove(verification)
        publish = next(cube for cube in cubes if cube["name"] == "publish")
        cubes.insert(cubes.index(publish), verification)

    text = mutate_cubes(sourcecraft_text, move_verification_first)
    assert "order is" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_missing_quality_gates_cube(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def remove_gates(cubes: list[dict[str, object]]) -> None:
        cubes[:] = [cube for cube in cubes if cube["name"] != "quality-gates"]

    text = mutate_cubes(sourcecraft_text, remove_gates)
    assert "quality-gates" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_flow_style_branch_entry(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        "\n  pull_request:",
        "\n    - {workflows: [publish-contracts], filter: {branches: [main]}}\n\n  pull_request:",
        1,
    )
    assert "publish-contracts" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_bare_dash_branch_entry(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        "\n  pull_request:",
        "\n    -\n      workflows: [publish-contracts]\n      filter:\n        branches: [main]\n\n  pull_request:",
        1,
    )
    assert "publish-contracts" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_broad_tag_mask(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        'tags: ["contracts-v*"]',
        'tags: ["contracts-v*", "v*"]',
        1,
    )
    assert "publish-contracts" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_the_uploader_in_preflight(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def add_uploader(cubes: list[dict[str, object]]) -> None:
        preflight = next(cube for cube in cubes if cube["name"] == "preflight")
        script = preflight["script"]
        assert isinstance(script, list)
        script.append("python3.11 scripts/contract_release.py publish --dist dist")

    text = mutate_cubes(sourcecraft_text, add_uploader)
    assert "runs the uploader" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_second_branch_publish_entry(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    tag_entry = (
        "    - workflows: [publish-contracts]\n"
        "      filter:\n"
        '        tags: ["contracts-v*"]\n'
    )
    branch_entry = (
        "    - workflows: [publish-contracts]\n"
        "      filter:\n"
        '        branches: ["main"]\n'
    )
    assert tag_entry in sourcecraft_text
    text = sourcecraft_text.replace(tag_entry, tag_entry + branch_entry, 1)
    assert "publish-contracts" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_requires_pull_request_quality_workflow(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  pull_request:\n",
        "workflows: [quality-checks]",
        "workflows: [ci-static-analysis-only]",
    )
    assert "quality-checks" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_synchronizes_tags_to_github(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace("refs/tags/", "refs/other/")
    assert "tag ref" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_requires_pull_request_checks(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace("  pull_request:\n", "  manual_only:\n", 1)
    assert "pull requests" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_requires_the_publication_workflow(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        "  publish-contracts:\n", "  missing-workflow:\n", 1
    )
    assert "publish-contracts" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "token",
    [
        "scripts/contract_release.py check-tag",
        "scripts/check_contract_boundaries.py",
        "python3.11 scripts/check_release_configuration.py",
        "python3.11 -m ruff check src/ automl-contracts/src/",
        "python3.11 -m mypy scripts/contract_release.py",
        "importlinter.cli",
        "pytest tests/contracts",
        "python3.11 -m build automl-contracts",
        "twine check",
        "scripts/verify_contract_distribution.py",
        "scripts/contract_release.py publish",
        "scripts/contract_release.py install",
        '"$REPO_DIR/scripts/e2e_contract_consumer.py"',
        "SOURCECRAFT_PYPI_REGISTRY_ID",
        "PYPI_REGISTRY_PAT",
    ],
)
def test_sourcecraft_publication_requires_each_gate(
    guard: ModuleType, sourcecraft_text: str, token: str
) -> None:
    text = inside(sourcecraft_text, "  publish-contracts:\n", token, "removed-gate")
    assert token in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_accepts_a_configured_registry_id(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = replace_publish_registry_id(sourcecraft_text, "registry123")
    assert guard.check_sourcecraft_ci(text) == []


def test_registry_id_helpers_stay_inside_the_publish_workflow(
    sourcecraft_text: str,
) -> None:
    identifier = REGISTRY_ID_LINE.search(sourcecraft_text)
    assert identifier is not None
    text = sourcecraft_text + (
        '\n  another-workflow:\n    env:\n      SOURCECRAFT_PYPI_REGISTRY_ID: "other"\n'
    )
    without_publish_identifier = text.replace(identifier.group(0) + "\n", "", 1)
    with pytest.raises(AssertionError, match="publish-contracts"):
        replace_publish_registry_id(without_publish_identifier, "registry123")
    with pytest.raises(AssertionError, match="publish-contracts"):
        append_to_publish_registry_env(
            without_publish_identifier, "PYPI_REGISTRY_PAT: token"
        )


def test_sourcecraft_rejects_a_literal_pat(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        "PYPI_REGISTRY_PAT: ${{ secrets.PYPI_REGISTRY_PAT }}",
        'PYPI_REGISTRY_PAT: "ghp_literal_secret_value"',
    )
    assert "literal secret" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_installs_python_311_through_the_shared_script(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    assert "add-apt-repository" not in sourcecraft_text
    assert sourcecraft_text.count("bash scripts/setup_python311.sh") == 6
    assert guard.check_sourcecraft_ci(sourcecraft_text) == []


def test_sourcecraft_rejects_an_inline_python_installation(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        "bash scripts/setup_python311.sh",
        "apt-get update -qq\n"
        "                add-apt-repository -y ppa:deadsnakes/ppa\n"
        "                apt-get install -y -qq python3.11 python3.11-venv",
    )
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "installs Python 3.11 inline" in violations
    assert "setup_python311.sh" in violations


def test_sourcecraft_requires_the_shared_python_setup(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        "bash scripts/setup_python311.sh",
        "",
    )
    assert "installs Python 3.11 without" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_checks_python_setup_per_task(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    document = load_document(sourcecraft_text)
    workflow = document["workflows"]["quality-checks"]  # type: ignore[index]
    task = next(
        item
        for item in workflow["tasks"]  # type: ignore[index]
        if item["name"] == "contracts-guarantees"
    )
    for cube in task["cubes"]:
        cube["script"] = [
            line.replace("bash scripts/setup_python311.sh", "")
            for line in cube["script"]
        ]
    text = dump_document(document)
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "contracts-guarantees" in violations
    assert "installs Python 3.11 without" in violations


@pytest.mark.parametrize(
    "install_command",
    [
        "apt-get install -y python3.11 python3.11-venv",
        "apt-get -y install python3.11 python3.11-venv",
        "apt install python3.11 python3.11-venv",
    ],
)
def test_sourcecraft_rejects_inline_python_install_variants(
    guard: ModuleType, sourcecraft_text: str, install_command: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        "bash scripts/setup_python311.sh",
        install_command,
    )
    assert "installs Python 3.11 inline" in messages(guard.check_sourcecraft_ci(text))


def test_order_helper_reports_a_publish_before_the_gates(guard: ModuleType) -> None:
    order = ("preflight", "quality-gates", "build-and-verify", "publish")
    actual = ["preflight", "publish", "quality-gates", "build-and-verify"]
    assert guard._order_violation(Path("ci.yaml"), order, actual, "release")


def test_sourcecraft_rejects_skip_existing(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        "scripts/contract_release.py publish",
        "scripts/contract_release.py publish --skip-existing",
    )
    assert "skip it" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_literal_credential(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        "python3.11 -m twine check",
        "python3.11 -m twine upload https://iam:hardcoded-token@pkg.sourcecraft.tech/simple/",
    )
    assert "literal credential" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_password_argument(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        "python3.11 -m twine check",
        "python3.11 -m twine upload --password hardcoded-token",
    )
    assert "password literal" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_regular_checks_need_the_release_guard(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  quality-checks:\n",
        "python3.11 scripts/check_release_configuration.py",
        "removed-guard",
        count=-1,
    )
    assert "check_release_configuration.py" in messages(
        guard.check_sourcecraft_ci(text)
    )


def test_github_publication_is_limited_to_the_contract_tag(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace('"contracts-v*"', '"v*"')
    assert "contracts-v*" in messages(guard.check_github_publish(text))


def test_github_publication_requires_the_release_gates(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace("pytest tests/contracts", "removed-gate")
    assert "pytest tests/contracts" in messages(guard.check_github_publish(text))


def test_github_publication_rejects_a_branch_filter(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace(
        '      - "contracts-v*"',
        '      - "contracts-v*"\n    branches:\n      - "main"',
        1,
    )
    assert "limited to contracts-v*" in messages(guard.check_github_publish(text))


def test_github_publication_requires_tags_not_branches(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace(
        '    tags:\n      - "contracts-v*"',
        '    branches:\n      - "main"',
        1,
    )
    assert "limited to contracts-v*" in messages(guard.check_github_publish(text))


def test_github_publication_rejects_a_commented_out_gate(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace(
        "pytest tests/contracts", "# pytest tests/contracts", 1
    )
    assert "pytest tests/contracts" in messages(guard.check_github_publish(text))


def test_github_publication_rejects_skip_existing(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace(
        "pytest tests/contracts",
        "pytest tests/contracts\n          python scripts/contract_release.py publish --skip-existing",
        1,
    )
    assert "existing version" in messages(guard.check_github_publish(text))


def test_github_publication_requires_the_quality_step(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace("Run release quality gates", "removed-gates")
    assert "Run release quality gates" in messages(guard.check_github_publish(text))


def test_github_publication_rejects_a_literal_pat(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace(
        "jobs:",
        'env:\n  PYPI_REGISTRY_PAT: "ghp_literal_secret_value"\njobs:',
        1,
    )
    assert "literal" in messages(guard.check_github_publish(text))


def test_uploader_cannot_use_skip_existing_or_a_password(
    guard: ModuleType,
) -> None:
    path = REPO_ROOT / "scripts" / "contract_release.py"
    text = path.read_text(encoding="utf-8")
    assert guard._check_uploader(path, text) == []
    mutated = text.replace(
        '        "twine",\n        "upload",',
        '        "twine",\n        "upload",\n        "--skip-existing",\n        "--password",\n        "hardcoded-token",',
        1,
    )
    assert mutated != text
    messages = "\n".join(
        violation.message for violation in guard._check_uploader(path, mutated)
    )
    assert "skip it" in messages
    assert "password literal" in messages


def test_violation_paths_follow_the_repo_root(
    guard: ModuleType, sourcecraft_text: str, github_publish_text: str, tmp_path: Path
) -> None:
    ci_path = tmp_path / "ci.yaml"
    github_path = tmp_path / "publish.yaml"
    ci_violations = guard.check_sourcecraft_ci(
        sourcecraft_text.replace("  pull_request:\n", "  no_pr:\n", 1), ci_path
    )
    github_violations = guard.check_github_publish(
        github_publish_text.replace('"contracts-v*"', '"v*"'), github_path
    )
    assert ci_violations
    assert github_violations
    assert {violation.path for violation in ci_violations} == {ci_path}
    assert {violation.path for violation in github_violations} == {github_path}


def test_github_ci_requires_pull_request_checks(
    guard: ModuleType, github_ci_text: str
) -> None:
    text = github_ci_text.replace("  pull_request:\n", "", 1)
    assert "pull requests" in messages(guard.check_github_ci(text))


def test_github_ci_path_follows_the_repo_root(
    guard: ModuleType, github_ci_text: str, tmp_path: Path
) -> None:
    path = tmp_path / "CI.yaml"
    violations = guard.check_github_ci(
        github_ci_text.replace("scripts/smoke_engine_install.py", "removed"), path
    )
    assert violations
    assert {violation.path for violation in violations} == {path}


def test_github_ci_rejects_a_commented_out_guard(
    guard: ModuleType, github_ci_text: str
) -> None:
    text = github_ci_text.replace(
        "          python scripts/check_release_configuration.py",
        "          # python scripts/check_release_configuration.py",
        1,
    )
    assert "check_release_configuration.py" in messages(guard.check_github_ci(text))


def test_github_ci_requires_engine_integration(
    guard: ModuleType, github_ci_text: str
) -> None:
    text = github_ci_text.replace("scripts/smoke_engine_install.py", "removed")
    assert "smoke_engine_install.py" in messages(guard.check_github_ci(text))


def test_engine_versioning_ignores_contracts_tags(
    guard: ModuleType, engine_pyproject_text: str
) -> None:
    assert guard.check_engine_versioning(engine_pyproject_text) == []


def test_engine_versioning_requires_a_version_tag_regex(
    guard: ModuleType, engine_pyproject_text: str
) -> None:
    text = engine_pyproject_text.replace(
        'tag_regex = "^v(?P<version>\\\\d+\\\\.\\\\d+\\\\.\\\\d+.*)$"',
        'tag_regex = "^.*$"',
        1,
    )
    assert "tag_regex" in messages(guard.check_engine_versioning(text))


def test_engine_versioning_requires_the_tag_mask(
    guard: ModuleType, engine_pyproject_text: str
) -> None:
    text = engine_pyproject_text.replace(
        '"--match", "v[0-9]*"',
        '"--match", "*[0-9]*"',
        1,
    )
    assert "v[0-9]*" in messages(guard.check_engine_versioning(text))


def test_engine_versioning_rejects_multiple_tag_masks(
    guard: ModuleType, engine_pyproject_text: str
) -> None:
    text = engine_pyproject_text.replace(
        '"--match", "v[0-9]*"]',
        '"--match", "v[0-9]*", "--match", "*[0-9]*"]',
        1,
    )
    assert "exactly one" in messages(guard.check_engine_versioning(text))


def test_engine_versioning_accepts_a_string_describe_command(
    guard: ModuleType, engine_pyproject_text: str
) -> None:
    text = engine_pyproject_text.replace(
        'git_describe_command = ["git", "describe", "--dirty", "--tags", '
        '"--long", "--abbrev=40", "--match", "v[0-9]*"]',
        'git_describe_command = "git describe --dirty --tags --long '
        "--match 'v[0-9]*'\"",
        1,
    )
    assert guard.check_engine_versioning(text) == []


def test_engine_versioning_requires_the_scm_section(guard: ModuleType) -> None:
    assert "[tool.setuptools_scm]" in messages(
        guard.check_engine_versioning("[tool.setuptools]\n")
    )


def test_missing_repository_files_are_reported(
    guard: ModuleType, tmp_path: Path
) -> None:
    violations = guard.run_checks(tmp_path)
    text = messages(violations)
    assert "SourceCraft CI file is missing" in text
    assert "GitHub release workflow is missing" in text
    assert "GitHub CI file is missing" in text
    assert "engine pyproject.toml is missing" in text
    assert "release helper script is missing" in text


def test_guard_cli_reports_success(
    guard: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert guard.main(["--repo-root", str(REPO_ROOT)]) == 0
    assert "release configuration is wired" in capsys.readouterr().out


def test_guard_cli_reports_failures(
    guard: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert guard.main(["--repo-root", str(tmp_path)]) == 1
    assert "release configuration violation" in capsys.readouterr().err


PAT_SECRET = "${{ secrets.PYPI_REGISTRY_PAT }}"


@pytest.mark.parametrize(
    "replacement",
    [
        (
            "  pull_request:\n"
            "    - workflows: [quality-checks]\n"
            "      filter:\n"
            '        target_branches: ["**"]\n'
            "    - {workflows: [quality-checks, publish-contracts], "
            'filter: {target_branches: ["**"]}}'
        ),
        (
            "  pull_request:\n"
            "    - workflows: [quality-checks]\n"
            "      filter:\n"
            '        target_branches: ["**"]\n'
            "    - {workflows: [publish-contracts], "
            'filter: {target_branches: ["**"]}}'
        ),
        "  pull_request: [publish-contracts]",
        "  pull_request: publish-contracts",
    ],
)
def test_sourcecraft_rejects_release_workflows_in_pull_request(
    guard: ModuleType, sourcecraft_text: str, replacement: str
) -> None:
    original = (
        "  pull_request:\n"
        "    - workflows: [quality-checks]\n"
        "      filter:\n"
        '        target_branches: ["**"]'
    )
    text = sourcecraft_text.replace(original, replacement, 1)
    assert text != sourcecraft_text
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "pull_request entry" in violations
    assert "publish-contracts" in violations


def test_sourcecraft_rejects_a_bare_dash_release_pull_request_entry(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        "\n  pull_request:\n    - workflows: [quality-checks]",
        (
            "\n  pull_request:\n"
            "    - workflows: [quality-checks]\n"
            "      filter:\n"
            '        target_branches: ["**"]\n'
            "    -\n"
            "      workflows: [sync-to-github]\n"
            "      filter:\n"
            '        target_branches: ["**"]'
        ),
        1,
    )
    assert text != sourcecraft_text
    text = messages(guard.check_sourcecraft_ci(text))
    assert "pull_request entry" in text
    assert "sync-to-github" in text


def test_sourcecraft_rejects_a_needs_dash_that_allows_parallel_tasks(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def add_needs(workflow: dict[str, object]) -> None:
        workflow["tasks"][0]["needs"] = ["-"]  # type: ignore[index]

    text = mutate_publish_workflow(sourcecraft_text, add_needs)
    assert 'needs: ["-"]' in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_parallel_tasks_without_needs(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = mutate_publish_workflow(sourcecraft_text, split_publish_tasks)
    assert "order is invalid" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_accepts_a_chained_split_task_sequence(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def split(workflow: dict[str, object]) -> None:
        split_publish_tasks(workflow, needs=["gates"])
        upload = workflow["tasks"][1]  # type: ignore[index]
        for cube in upload["cubes"]:
            cube["script"][0] = f"{cube['script'][0]}\nbash scripts/setup_python311.sh"

    text = mutate_publish_workflow(sourcecraft_text, split)
    assert guard.check_sourcecraft_ci(text) == []


def test_sourcecraft_rejects_a_reordered_task_chain(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def split(workflow: dict[str, object]) -> None:
        cubes = workflow["tasks"][0]["cubes"]  # type: ignore[index]
        by_name = {cube["name"]: cube for cube in cubes}
        workflow["tasks"] = [
            {
                "name": "gates",
                "cubes": [
                    by_name["preflight"],
                    by_name["quality-gates"],
                    by_name["build-and-verify"],
                ],
            },
            {
                "name": "verify",
                "cubes": [by_name["verify-from-registry"]],
                "needs": ["gates"],
            },
            {"name": "upload", "cubes": [by_name["publish"]], "needs": ["verify"]},
        ]

    text = mutate_publish_workflow(sourcecraft_text, split)
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "verify-from-registry" in violations
    assert "publish" in violations


def test_sourcecraft_rejects_an_unknown_task_need(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def add_needs(workflow: dict[str, object]) -> None:
        workflow["tasks"][0]["needs"] = ["missing-task"]  # type: ignore[index]

    text = mutate_publish_workflow(sourcecraft_text, add_needs)
    assert "unknown task" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_cyclic_task_need(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def split(workflow: dict[str, object]) -> None:
        split_publish_tasks(workflow)
        workflow["tasks"][0]["needs"] = ["upload"]  # type: ignore[index]
        workflow["tasks"][1]["needs"] = ["gates"]  # type: ignore[index]

    text = mutate_publish_workflow(sourcecraft_text, split)
    assert "cyclic" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_an_unknown_cube_need(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def add_cube_needs(workflow: dict[str, object]) -> None:
        cube = find_cube(workflow, "publish")
        cube["needs"] = ["missing-cube"]

    text = mutate_publish_workflow(sourcecraft_text, add_cube_needs)
    assert "unknown cube" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("cube-allow-failure", "allow_failure"),
        ("cube-if", "conditional"),
        ("task-allow-failure", "allow_failure"),
        ("task-if", "conditional"),
    ],
)
def test_sourcecraft_rejects_neutralized_release_cubes(
    guard: ModuleType, sourcecraft_text: str, mutation: str, expected: str
) -> None:
    def apply(workflow: dict[str, object]) -> None:
        cube = find_cube(workflow, "quality-gates")
        if mutation == "cube-allow-failure":
            cube["allow_failure"] = True
        elif mutation == "cube-if":
            cube["if"] = "false"
        elif mutation == "task-allow-failure":
            workflow["tasks"][0]["allow_failure"] = True  # type: ignore[index]
        else:
            workflow["tasks"][0]["if"] = "false"  # type: ignore[index]

    text = mutate_publish_workflow(sourcecraft_text, apply)
    assert expected in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "python3.11 scripts/check_contract_boundaries.py",
            'echo "python3.11 scripts/check_contract_boundaries.py"',
        ),
        (
            "python3.11 scripts/check_contract_boundaries.py",
            'printf "%s\\n" "python3.11 scripts/check_contract_boundaries.py"',
        ),
        (
            "python3.11 scripts/check_contract_boundaries.py",
            "python3.11 scripts/check_contract_boundaries.py || true",
        ),
        (
            "python3.11 scripts/check_contract_boundaries.py",
            "python3.11 scripts/check_contract_boundaries.py; true",
        ),
        (
            "python3.11 scripts/check_contract_boundaries.py",
            "python3.11 scripts/check_contract_boundaries.py && true",
        ),
        (
            "python3.11 scripts/check_contract_boundaries.py",
            "! python3.11 scripts/check_contract_boundaries.py",
        ),
        (
            'set -e\n                echo "═══════════════════════════════════════════════════════════════"\n                echo "  RELEASE GATES',
            'set +e\n                echo "═══════════════════════════════════════════════════════════════"\n                echo "  RELEASE GATES',
        ),
        (
            "python3.11 scripts/check_contract_boundaries.py",
            "if false; then\n                  python3.11 scripts/check_contract_boundaries.py\n                fi",
        ),
        (
            "python3.11 scripts/check_contract_boundaries.py",
            "cat <<EOF\n                python3.11 scripts/check_contract_boundaries.py\n                EOF",
        ),
        (
            "python3.11 scripts/check_contract_boundaries.py",
            "python3.11 scripts/check_contract_boundaries.py || exit 0",
        ),
        (
            "python3.11 scripts/check_contract_boundaries.py",
            'python3.11 scripts/check_contract_boundaries.py || echo "skipped"',
        ),
        (
            "python3.11 scripts/check_contract_boundaries.py",
            "python3.11 scripts/check_contract_boundaries.py ||\n                  true",
        ),
        (
            "python3.11 scripts/check_contract_boundaries.py",
            "false && python3.11 scripts/check_contract_boundaries.py",
        ),
        (
            "python3.11 scripts/check_contract_boundaries.py",
            "true python3.11 scripts/check_contract_boundaries.py",
        ),
        (
            "python3.11 scripts/check_contract_boundaries.py",
            ": python3.11 scripts/check_contract_boundaries.py",
        ),
    ],
)
def test_sourcecraft_rejects_neutralized_gate_commands(
    guard: ModuleType, sourcecraft_text: str, old: str, new: str
) -> None:
    assert old in sourcecraft_text
    mutated = inside(sourcecraft_text, "  publish-contracts:\n", old, new)
    assert mutated != sourcecraft_text
    assert guard.check_sourcecraft_ci(mutated)


def test_sourcecraft_rejects_the_pat_at_workflow_level(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = append_to_publish_registry_env(
        sourcecraft_text, f"PYPI_REGISTRY_PAT: {PAT_SECRET}"
    )
    assert "workflow level" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "cube_name", ["setup", "quality-gates", "build-and-verify", "preflight"]
)
def test_sourcecraft_rejects_the_pat_outside_publication_cubes(
    guard: ModuleType, sourcecraft_text: str, cube_name: str
) -> None:
    def add_pat(workflow: dict[str, object]) -> None:
        cube = find_cube(workflow, cube_name)
        env = cube.setdefault("env", {})
        assert isinstance(env, dict)
        env["PYPI_REGISTRY_PAT"] = PAT_SECRET

    text = mutate_publish_workflow(sourcecraft_text, add_pat)
    assert "PYPI_REGISTRY_PAT" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_the_pat_at_task_level(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def add_pat(workflow: dict[str, object]) -> None:
        workflow["tasks"][0]["env"] = {"PYPI_REGISTRY_PAT": PAT_SECRET}  # type: ignore[index]

    text = mutate_publish_workflow(sourcecraft_text, add_pat)
    assert "task" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_requires_the_pat_on_registry_installation(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        "          - name: verify-from-registry\n"
        "            env:\n"
        f"              PYPI_REGISTRY_PAT: {PAT_SECRET}\n",
        "          - name: verify-from-registry\n",
        1,
    )
    assert text != sourcecraft_text
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "verify-from-registry" in violations
    assert "PYPI_REGISTRY_PAT" in violations
    assert PAT_SECRET not in violations


def test_sourcecraft_guard_accepts_the_pat_only_on_publishing_cubes(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    document = load_document(sourcecraft_text)
    workflow = publish_workflow(document)
    for cube_name in ("setup", "quality-gates", "build-and-verify", "preflight"):
        assert "PYPI_REGISTRY_PAT" not in find_cube(workflow, cube_name).get("env", {})
    for cube_name in ("publish", "verify-from-registry"):
        cube = find_cube(workflow, cube_name)
        assert cube["env"] == {"PYPI_REGISTRY_PAT": PAT_SECRET}


def test_sourcecraft_requires_the_full_tag_ref_in_preflight(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        '                  --ref "${SOURCECRAFT_COMMIT_REF}" \\\n',
        "",
    )
    assert '--ref "${SOURCECRAFT_COMMIT_REF}"' in messages(
        guard.check_sourcecraft_ci(text)
    )


def test_sourcecraft_requires_a_failing_tag_check_before_publish(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        """                python3.11 scripts/contract_release.py check-tag \\
                  --ref "${SOURCECRAFT_COMMIT_REF}" \\
                  --tag "${SOURCECRAFT_COMMIT_REF_NAME}" \\
                  --pyproject automl-contracts/pyproject.toml
""",
        """                if git tag --points-at HEAD | grep -q "^contracts-v"; then
                  echo "publishing"
                fi
""",
    )
    assert "advisory" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_requires_consumer_e2e_before_publish(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        '"$REPO_DIR/scripts/e2e_contract_consumer.py"',
        '"$REPO_DIR/scripts/removed-consumer.py"',
    )
    assert "e2e_contract_consumer.py" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_nested_list_pull_request_entry(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    original = (
        "  pull_request:\n"
        "    - workflows: [quality-checks]\n"
        "      filter:\n"
        '        target_branches: ["**"]'
    )
    text = sourcecraft_text.replace(
        original,
        (original + "\n    - [publish-contracts]"),
        1,
    )
    assert text != sourcecraft_text
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "nested workflow list" in violations
    assert "publish-contracts" in violations


def test_sourcecraft_rejects_a_pull_request_entry_without_workflows(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    original = (
        "  pull_request:\n"
        "    - workflows: [quality-checks]\n"
        "      filter:\n"
        '        target_branches: ["**"]'
    )
    text = sourcecraft_text.replace(
        original,
        (original + '\n    - filter:\n        target_branches: ["**"]'),
        1,
    )
    assert text != sourcecraft_text
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "pull_request entry" in violations
    assert "all workflows" in violations


def test_sourcecraft_rejects_release_workflows_on_schedule(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        "\n  pull_request:",
        (
            "\n  schedule:\n"
            "    - workflows: [publish-contracts]\n"
            "      interval: 1h\n"
            "\n  pull_request:"
        ),
        1,
    )
    assert text != sourcecraft_text
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "schedule" in violations
    assert "publish-contracts" in violations


def test_sourcecraft_rejects_release_workflows_on_an_event_trigger(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        "\n  pull_request:",
        (
            "\n  issue_comment.create:\n"
            "    - workflows: [sync-to-github]\n"
            "\n  pull_request:"
        ),
        1,
    )
    assert text != sourcecraft_text
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "issue_comment.create" in violations
    assert "sync-to-github" in violations


def test_sourcecraft_rejects_an_all_workflows_push_entry(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        "  push:\n",
        "  push:\n    - filter:\n        branches: [main]\n",
        1,
    )
    assert text != sourcecraft_text
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "push entry" in violations
    assert "all workflows" in violations


def test_sourcecraft_accepts_elif_in_a_gate_script(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    block = (
        'if [ -z "$X" ]; then\n'
        "                  echo empty\n"
        '                elif [ "$X" = 1 ]; then\n'
        "                  python3.11 scripts/check_contract_boundaries.py\n"
        "                fi"
    )
    text = inside(
        sourcecraft_text,
        "  publish-contracts:\n",
        "python3.11 scripts/check_contract_boundaries.py",
        block,
    )
    assert guard.check_sourcecraft_ci(text) == []


def test_sourcecraft_accepts_set_o_errexit_in_a_gate_script(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def replace_set_e(workflow: dict[str, object]) -> None:
        cube = find_cube(workflow, "quality-gates")
        cube["script"] = [cube_script(cube).replace("set -e", "set -o errexit", 1)]

    text = mutate_publish_workflow(sourcecraft_text, replace_set_e)
    assert guard.check_sourcecraft_ci(text) == []


def test_sourcecraft_rejects_the_pat_at_document_level(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    document = load_document(sourcecraft_text)
    document["env"] = {"PYPI_REGISTRY_PAT": PAT_SECRET}
    text = dump_document(document)
    assert "document level" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "script",
    [
        "gate() {\n  python3.11 scripts/check_contract_boundaries.py\n}",
        "gate() { python3.11 scripts/check_contract_boundaries.py; }",
        "function gate { python3.11 scripts/check_contract_boundaries.py; }",
    ],
)
def test_sourcecraft_rejects_a_gate_declared_only_in_a_function(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    def replace_gate(workflow: dict[str, object]) -> None:
        cube = find_cube(workflow, "quality-gates")
        cube["script"] = [script]

    text = mutate_publish_workflow(sourcecraft_text, replace_gate)
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "check_contract_boundaries.py" in violations


def test_sourcecraft_rejects_a_backgrounded_gate(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def background(workflow: dict[str, object]) -> None:
        cube = find_cube(workflow, "quality-gates")
        cube["script"] = [
            cube_script(cube).replace(
                "python3.11 scripts/check_contract_boundaries.py",
                "python3.11 scripts/check_contract_boundaries.py &",
            )
        ]

    text = mutate_publish_workflow(sourcecraft_text, background)
    assert "check_contract_boundaries.py" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "script",
    [
        "python3.11 scripts/check_contract_boundaries.py\nset -e",
        "set -e\nset +e\npython3.11 scripts/check_contract_boundaries.py",
        "set -e\nset +o errexit\npython3.11 scripts/check_contract_boundaries.py",
        "if false; then\n  set -e\nfi\npython3.11 scripts/check_contract_boundaries.py",
    ],
)
def test_sourcecraft_requires_effective_errexit_before_the_gate(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    def replace_gate(workflow: dict[str, object]) -> None:
        cube = find_cube(workflow, "quality-gates")
        cube["script"] = [script]

    text = mutate_publish_workflow(sourcecraft_text, replace_gate)
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "set -e" in violations or "set +e" in violations


def test_sourcecraft_requires_ref_and_check_tag_in_the_same_command(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    marker = '                  --ref "${SOURCECRAFT_COMMIT_REF}" \\\n'
    noop = '                true --ref "${SOURCECRAFT_COMMIT_REF}"\n'
    text = sourcecraft_text.replace(marker, "", 1)
    text = text.replace(
        "                python3.11 scripts/contract_release.py show-endpoints",
        noop + "                python3.11 scripts/contract_release.py show-endpoints",
        1,
    )
    assert text != sourcecraft_text
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "--ref" in violations
    assert "check-tag" in violations


def test_sourcecraft_inspects_a_string_task_reference(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    document = load_document(sourcecraft_text)
    workflow = publish_workflow(document)
    document["tasks"] = [
        {
            "name": "extra",
            "cubes": [
                {
                    "name": "publish-again",
                    "script": [
                        "set -e",
                        (
                            "python3.11 scripts/contract_release.py publish "
                            "--dist automl-contracts/dist "
                            "--expected-version 0.1.0"
                        ),
                    ],
                }
            ],
        }
    ]
    workflow["tasks"].append("extra")  # type: ignore[union-attr]
    text = dump_document(document)
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "publish-again" in violations
    assert "runs the uploader" in violations


def test_sourcecraft_accepts_a_valid_string_task_reference(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    document = load_document(sourcecraft_text)
    workflow = publish_workflow(document)
    inline = workflow["tasks"][0]  # type: ignore[index]
    document["tasks"] = [inline]
    workflow["tasks"] = ["contracts-release"]
    text = dump_document(document)
    assert guard.check_sourcecraft_ci(text) == []
