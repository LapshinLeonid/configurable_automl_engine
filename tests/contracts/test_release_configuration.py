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
def root_pyproject_text() -> str:
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


def replace_gate_script(sourcecraft_text: str, cube_name: str, script: str) -> str:
    """Replace the whole script of a publish-contracts cube."""

    def apply(workflow: dict[str, object]) -> None:
        find_cube(workflow, cube_name)["script"] = [script]

    return mutate_publish_workflow(sourcecraft_text, apply)


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


def test_engine_version_tags_ignore_contract_release_tags(
    guard: ModuleType, root_pyproject_text: str
) -> None:
    assert guard.check_engine_version_tags(root_pyproject_text) == []


def test_engine_version_tags_reject_a_contract_tag_collision(
    guard: ModuleType, root_pyproject_text: str
) -> None:
    text = root_pyproject_text.replace(
        '[tool.setuptools_scm.tag]\nprefix = "v"',
        '[tool.setuptools_scm.tag]\nprefix = ""',
    )
    assert "contracts-v0.1.0" in messages(guard.check_engine_version_tags(text))


def test_engine_version_tags_require_an_engine_regex(
    guard: ModuleType, root_pyproject_text: str
) -> None:
    text = root_pyproject_text.replace(
        'regex = "^(?P<version>\\\\d+\\\\.\\\\d+\\\\.\\\\d+.*)$"',
        'regex = "^(?P<version>contracts-.*)$"',
    )
    assert "v1.2.3" in messages(guard.check_engine_version_tags(text))


def test_engine_version_tags_require_valid_toml(guard: ModuleType) -> None:
    violations = guard.check_engine_version_tags("not [valid")
    assert "not valid TOML" in messages(violations)


def test_engine_version_tags_require_a_compatible_setuptools_scm(
    guard: ModuleType, root_pyproject_text: str
) -> None:
    text = root_pyproject_text.replace(
        '"setuptools-scm>=10.3.4"', '"setuptools-scm>=8"'
    )
    assert "setuptools-scm>=10.3.4" in messages(guard.check_engine_version_tags(text))


def test_engine_version_tags_require_the_setuptools_scm_requirement(
    guard: ModuleType, root_pyproject_text: str
) -> None:
    text = root_pyproject_text.replace('"setuptools-scm>=10.3.4", ', "")
    assert "do not declare setuptools-scm" in messages(
        guard.check_engine_version_tags(text)
    )


def pyproject_with_describe_command(text: str, value: str) -> str:
    """Insert a custom ``scm.git.describe_command`` into the engine config."""
    return text.replace(
        "[tool.setuptools_scm.tag]",
        f"[tool.setuptools_scm.scm.git]\ndescribe_command = {value}\n\n"
        "[tool.setuptools_scm.tag]",
        1,
    )


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git describe --tags --long --match v*", ["v*"]),
        (["git", "describe", "--tags", "--long", "--match", "v*"], ["v*"]),
        ("git describe --tags --long --match=v*", ["v*"]),
        (
            "git describe --tags --long --match v* --match *[0-9]*",
            ["v*", "*[0-9]*"],
        ),
        (
            ["git", "describe", "--match=v*", "--match", "release-*"],
            ["v*", "release-*"],
        ),
        ("git describe --tags --long", []),
        ("git describe --tags --long --match", None),
        ("git describe --tags --long --match=", None),
        ("git describe --tags --long --match 'v*", None),
        (42, None),
    ],
)
def test_describe_match_globs_parses_supported_forms(
    guard: ModuleType, command: object, expected: list[str] | None
) -> None:
    assert guard._describe_match_globs(command) == expected


@pytest.mark.parametrize(
    "value",
    [
        '"git describe --tags --long --match v* --match *[0-9]*"',
        (
            '["git", "describe", "--tags", "--long", "--match", "v*", '
            '"--match", "*[0-9]*"]'
        ),
    ],
)
def test_engine_version_tags_reject_unsafe_describe_commands(
    guard: ModuleType, root_pyproject_text: str, value: str
) -> None:
    text = pyproject_with_describe_command(root_pyproject_text, value)
    assert "contracts-v0.1.0" in messages(guard.check_engine_version_tags(text))


@pytest.mark.parametrize(
    "value",
    [
        '"git describe --tags --long --match=v*"',
        '["git", "describe", "--tags", "--long", "--match", "v*"]',
    ],
)
def test_engine_version_tags_accept_safe_describe_commands(
    guard: ModuleType, root_pyproject_text: str, value: str
) -> None:
    text = pyproject_with_describe_command(root_pyproject_text, value)
    assert guard.check_engine_version_tags(text) == []


def test_engine_version_tags_reject_a_describe_command_without_match(
    guard: ModuleType, root_pyproject_text: str
) -> None:
    text = pyproject_with_describe_command(
        root_pyproject_text, '"git describe --tags --long"'
    )
    assert "has no --match" in messages(guard.check_engine_version_tags(text))


def test_engine_version_tags_reject_an_unparsable_describe_command(
    guard: ModuleType, root_pyproject_text: str
) -> None:
    text = pyproject_with_describe_command(
        root_pyproject_text, '"git describe --match \'v*"'
    )
    assert "cannot be parsed" in messages(guard.check_engine_version_tags(text))


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
        "python3.11 -m ruff check src/ configurable-automl-contracts/src/",
        "python3.11 -m mypy scripts/contract_release.py",
        "importlinter.cli",
        "pytest tests/contracts",
        "python3.11 -m build configurable-automl-contracts",
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


def test_github_publication_requires_the_pypi_step(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace(
        """      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: configurable-automl-contracts/dist
""",
        "",
        1,
    )
    assert text != github_publish_text
    assert "Publish to PyPI" in messages(guard.check_github_publish(text))


def test_github_publication_requires_trusted_publishing(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace("      id-token: write\n", "", 1)
    assert text != github_publish_text
    assert "id-token" in messages(guard.check_github_publish(text))


def test_github_publication_publishes_the_built_contract(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace(
        "packages-dir: configurable-automl-contracts/dist", "packages-dir: dist", 1
    )
    assert "configurable-automl-contracts/dist" in messages(
        guard.check_github_publish(text)
    )


def test_github_publication_verifies_the_published_package(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace(
        "configurable-automl-contracts==${VERSION}",
        "configurable-automl-contracts",
        1,
    )
    assert "configurable-automl-contracts==${VERSION}" in messages(
        guard.check_github_publish(text)
    )


def test_github_publication_runs_pypi_after_the_release_gates(
    guard: ModuleType, github_publish_text: str
) -> None:
    document = load_document(github_publish_text)
    steps = document["jobs"]["publish"]["steps"]  # type: ignore[index]
    publish = next(step for step in steps if step.get("name") == "Publish to PyPI")
    steps.remove(publish)
    steps.insert(0, publish)
    text = dump_document(document)
    assert "in that order" in messages(guard.check_github_publish(text))


@pytest.mark.parametrize("job_permissions", [{"contents": "read"}, "read-all"])
def test_github_publication_rejects_a_job_permissions_override(
    guard: ModuleType, github_publish_text: str, job_permissions: object
) -> None:
    document = load_document(github_publish_text)
    document["permissions"] = {"contents": "read", "id-token": "write"}
    document["jobs"]["publish"]["permissions"] = job_permissions  # type: ignore[index]
    text = dump_document(document)
    assert "id-token" in messages(guard.check_github_publish(text))


@pytest.mark.parametrize("location", ["workflow", "job"])
def test_github_publication_accepts_the_write_all_shorthand(
    guard: ModuleType, github_publish_text: str, location: str
) -> None:
    document = load_document(github_publish_text)
    if location == "workflow":
        document["permissions"] = "write-all"
        del document["jobs"]["publish"]["permissions"]  # type: ignore[index]
    else:
        document["jobs"]["publish"]["permissions"] = "write-all"  # type: ignore[index]
    text = dump_document(document)
    assert guard.check_github_publish(text) == []


@pytest.mark.parametrize(
    "option",
    [
        "skip-existing: true",
        "repository-url: https://test.pypi.org/legacy/",
        "password: literal-token",
    ],
)
def test_github_publication_rejects_extra_publish_inputs(
    guard: ModuleType, github_publish_text: str, option: str
) -> None:
    anchor = "          packages-dir: configurable-automl-contracts/dist\n"
    text = github_publish_text.replace(anchor, anchor + f"          {option}\n", 1)
    assert "unsupported inputs" in messages(guard.check_github_publish(text))


@pytest.mark.parametrize(
    "action",
    [
        "pypa/gh-action-pypi-publish-fake@main",
        "pypa/gh-action-pypi-publish@main",
    ],
)
def test_github_publication_pins_the_pypi_action(
    guard: ModuleType, github_publish_text: str, action: str
) -> None:
    text = github_publish_text.replace(
        "pypa/gh-action-pypi-publish@release/v1", action, 1
    )
    assert "pypa/gh-action-pypi-publish@release/v1" in messages(
        guard.check_github_publish(text)
    )


def test_github_publication_verifies_in_isolation(
    guard: ModuleType, github_publish_text: str
) -> None:
    document = load_document(github_publish_text)
    steps = document["jobs"]["publish"]["steps"]  # type: ignore[index]
    step = next(
        step for step in steps if step.get("name") == "Verify installation from PyPI"
    )
    step["run"] = step["run"].replace("--isolated", "removed-isolation")
    text = dump_document(document)
    assert "--isolated" in messages(guard.check_github_publish(text))


def test_github_publication_verifies_the_sdist(
    guard: ModuleType, github_publish_text: str
) -> None:
    text = github_publish_text.replace(
        "--no-binary configurable-automl-contracts", "--no-binary :none:", 1
    )
    assert "--no-binary configurable-automl-contracts" in messages(
        guard.check_github_publish(text)
    )


def test_github_publication_binds_isolation_to_the_resolving_install(
    guard: ModuleType, github_publish_text: str
) -> None:
    document = load_document(github_publish_text)
    steps = document["jobs"]["publish"]["steps"]  # type: ignore[index]
    step = next(
        step for step in steps if step.get("name") == "Verify installation from PyPI"
    )
    old_command = """pip install \\
    --isolated \\
    --index-url"""
    new_command = """pip install \\
    --index-url"""
    assert old_command in step["run"]
    step["run"] = step["run"].replace(old_command, new_command, 1)
    text = dump_document(document)
    assert "--isolated" in messages(guard.check_github_publish(text))


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


@pytest.mark.parametrize(
    "secret",
    [
        PAT_SECRET,
        "${{secrets.PYPI_REGISTRY_PAT}}",
        "${{   secrets.PYPI_REGISTRY_PAT   }}",
    ],
)
def test_sourcecraft_rejects_an_alternate_pat_key_at_document_level(
    guard: ModuleType, sourcecraft_text: str, secret: str
) -> None:
    document = load_document(sourcecraft_text)
    document["env"] = {"REGISTRY_TOKEN": secret}
    text = dump_document(document)
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "document level" in violations
    assert "REGISTRY_TOKEN" in violations
    assert secret not in violations


def test_sourcecraft_rejects_an_alternate_pat_key_at_workflow_level(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = append_to_publish_registry_env(
        sourcecraft_text, f"REGISTRY_TOKEN: {PAT_SECRET}"
    )
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "workflow level" in violations
    assert "REGISTRY_TOKEN" in violations
    assert PAT_SECRET not in violations


def test_sourcecraft_rejects_an_alternate_pat_key_at_task_level(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def add_pat(workflow: dict[str, object]) -> None:
        workflow["tasks"][0]["env"] = {"LEAKED_TOKEN": PAT_SECRET}  # type: ignore[index]

    text = mutate_publish_workflow(sourcecraft_text, add_pat)
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "task" in violations
    assert "LEAKED_TOKEN" in violations
    assert PAT_SECRET not in violations


@pytest.mark.parametrize(
    "cube_name", ["setup", "quality-gates", "build-and-verify", "preflight"]
)
@pytest.mark.parametrize(
    "key",
    ["REGISTRY_TOKEN", "LEAKED_TOKEN", "PYPI_REGISTRY_PAT_BACKUP"],
)
def test_sourcecraft_rejects_an_alternate_pat_key_in_non_publishing_cubes(
    guard: ModuleType, sourcecraft_text: str, cube_name: str, key: str
) -> None:
    def add_pat(workflow: dict[str, object]) -> None:
        cube = find_cube(workflow, cube_name)
        cube["env"] = {key: PAT_SECRET}

    text = mutate_publish_workflow(sourcecraft_text, add_pat)
    violations = messages(guard.check_sourcecraft_ci(text))
    assert cube_name in violations
    assert key in violations
    assert PAT_SECRET not in violations


@pytest.mark.parametrize("cube_name", ["publish", "verify-from-registry"])
@pytest.mark.parametrize(
    "key",
    ["REGISTRY_TOKEN", "PYPI_REGISTRY_PAT_BACKUP", " PYPI_REGISTRY_PAT "],
)
def test_sourcecraft_rejects_an_alternate_pat_key_in_publishing_cubes(
    guard: ModuleType, sourcecraft_text: str, cube_name: str, key: str
) -> None:
    def add_pat(workflow: dict[str, object]) -> None:
        cube = find_cube(workflow, cube_name)
        cube["env"] = {"PYPI_REGISTRY_PAT": PAT_SECRET, key: PAT_SECRET}

    text = mutate_publish_workflow(sourcecraft_text, add_pat)
    violations = messages(guard.check_sourcecraft_ci(text))
    assert cube_name in violations
    assert key in violations
    assert PAT_SECRET not in violations


@pytest.mark.parametrize(
    "secret",
    [
        "${{secrets.PYPI_REGISTRY_PAT}}",
        "${{   secrets.PYPI_REGISTRY_PAT   }}",
    ],
)
def test_sourcecraft_accepts_pat_expressions_with_flexible_spacing(
    guard: ModuleType, sourcecraft_text: str, secret: str
) -> None:
    def replace(workflow: dict[str, object]) -> None:
        for cube_name in ("publish", "verify-from-registry"):
            find_cube(workflow, cube_name)["env"] = {"PYPI_REGISTRY_PAT": secret}

    text = mutate_publish_workflow(sourcecraft_text, replace)
    assert guard.check_sourcecraft_ci(text) == []


def test_sourcecraft_rejects_an_alternate_pat_key_in_list_env(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def add_pat(workflow: dict[str, object]) -> None:
        cube = find_cube(workflow, "quality-gates")
        cube["env"] = [f"REGISTRY_TOKEN={PAT_SECRET}"]

    text = mutate_publish_workflow(sourcecraft_text, add_pat)
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "quality-gates" in violations
    assert "REGISTRY_TOKEN" in violations
    assert "must be a mapping" in violations
    assert PAT_SECRET not in violations


@pytest.mark.parametrize("cube_name", ["publish", "verify-from-registry"])
def test_sourcecraft_rejects_a_non_mapping_publish_env(
    guard: ModuleType, sourcecraft_text: str, cube_name: str
) -> None:
    def add_pat(workflow: dict[str, object]) -> None:
        cube = find_cube(workflow, cube_name)
        cube["env"] = [f"PYPI_REGISTRY_PAT={PAT_SECRET}"]

    text = mutate_publish_workflow(sourcecraft_text, add_pat)
    violations = messages(guard.check_sourcecraft_ci(text))
    assert cube_name in violations
    assert "must be a mapping" in violations
    assert PAT_SECRET not in violations


def test_sourcecraft_rejects_a_non_mapping_document_env(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    document = load_document(sourcecraft_text)
    document["env"] = [f"PYPI_REGISTRY_PAT={PAT_SECRET}"]
    violations = messages(guard.check_sourcecraft_ci(dump_document(document)))
    assert "document env must be a mapping" in violations
    assert PAT_SECRET not in violations


def test_sourcecraft_rejects_an_alternate_pat_key_in_another_workflow(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    document = load_document(sourcecraft_text)
    quality_workflow = document["workflows"]["quality-checks"]  # type: ignore[index]
    find_cube(quality_workflow, "lint-and-typecheck")["env"] = {
        "LEAKED_TOKEN": PAT_SECRET
    }
    text = dump_document(document)
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "lint-and-typecheck" in violations
    assert "LEAKED_TOKEN" in violations
    assert PAT_SECRET not in violations


def test_sourcecraft_resolves_a_merged_pat_env(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    text = sourcecraft_text.replace(
        "\nworkflows:\n",
        (
            "\nx-shared-env: &shared-env\n"
            f"  REGISTRY_TOKEN: {PAT_SECRET}\n"
            "\nworkflows:\n"
        ),
        1,
    )
    text = text.replace(
        "          - name: quality-gates\n",
        "          - name: quality-gates\n"
        "            env:\n"
        "              <<: *shared-env\n",
        1,
    )
    violations = messages(guard.check_sourcecraft_ci(text))
    assert "quality-gates" in violations
    assert "REGISTRY_TOKEN" in violations
    assert PAT_SECRET not in violations


def test_sourcecraft_accepts_a_null_document_env(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    document = load_document(sourcecraft_text)
    document["env"] = None
    assert guard.check_sourcecraft_ci(dump_document(document)) == []


def test_sourcecraft_reports_each_pat_leak_without_the_secret(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    document = load_document(sourcecraft_text)
    document["env"] = {"REGISTRY_TOKEN": PAT_SECRET}
    find_cube(publish_workflow(document), "setup")["env"] = {"LEAKED_TOKEN": PAT_SECRET}
    text = dump_document(document)
    violations = messages(guard.check_sourcecraft_ci(text))
    assert violations.count("document level") == 1
    assert "setup" in violations
    assert PAT_SECRET not in violations


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
                  --pyproject configurable-automl-contracts/pyproject.toml
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
        "                  echo configured\n"
        "                fi\n"
        "                python3.11 scripts/check_contract_boundaries.py"
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
        (
            "set -e\ntest -f /nonexistent && "
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        "set -e\nfalse && python3.11 scripts/check_contract_boundaries.py",
        "set -e\ntrue && python3.11 scripts/check_contract_boundaries.py",
        "set -e\nsome_command && python3.11 scripts/check_contract_boundaries.py",
        (
            "set -e\ntest -f /nonexistent &&\n  "
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        "set -e\ntrue &&\n  python3.11 scripts/check_contract_boundaries.py",
        "set -e\ncd /tmp; python3.11 scripts/check_contract_boundaries.py",
        "set -e\n: && python3.11 scripts/check_contract_boundaries.py",
        "set -e\n(echo start && python3.11 scripts/check_contract_boundaries.py)",
        (
            "set -e\ntest -f /nonexistent &&\n\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\ntest -f /nonexistent &&\n# comment\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\ntest -f /nonexistent ||\n\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\ntest -f /nonexistent ||\n# comment\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\ntest -f /nonexistent |\n# comment\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
    ],
)
def test_sourcecraft_rejects_a_gate_chained_before_and(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "check_contract_boundaries.py" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "script",
    [
        "set -e\npython3.11 scripts/check_contract_boundaries.py",
        "set -e\n:\npython3.11 scripts/check_contract_boundaries.py",
        "set -eo pipefail\npython3.11 scripts/check_contract_boundaries.py",
        "set -e\npython3.11 scripts/check_contract_boundaries.py;",
        "set -e\nCHECK=$(python3.11 scripts/check_contract_boundaries.py)",
        "set -e\nnohup python3.11 scripts/check_contract_boundaries.py",
        "set -e\nenv python3.11 scripts/check_contract_boundaries.py",
        "set -e\n(\n  python3.11 scripts/check_contract_boundaries.py\n)",
        'set -e\neval "python3.11 scripts/check_contract_boundaries.py"',
        'set -e\nbash -c "python3.11 scripts/check_contract_boundaries.py"',
    ],
)
def test_sourcecraft_accepts_a_standalone_gate(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "check_contract_boundaries.py" not in messages(
        guard.check_sourcecraft_ci(text)
    )


@pytest.mark.parametrize(
    "script",
    [
        "set -e\nCHECK=$(echo python3.11 scripts/check_contract_boundaries.py)",
        "set -e\nCHECK=$(printf '%s' python3.11 scripts/check_contract_boundaries.py)",
        "set -e\nCHECK=$(true python3.11 scripts/check_contract_boundaries.py)",
        "set -e\nCHECK=$(alias python3.11 scripts/check_contract_boundaries.py)",
        "set -e\nexport CHECK=$(echo python3.11 scripts/check_contract_boundaries.py)",
    ],
)
def test_sourcecraft_rejects_a_gate_only_stored_in_a_variable(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "check_contract_boundaries.py" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "script",
    [
        (
            "set -e\ncommand declare CHECK='python3.11 "
            "scripts/check_contract_boundaries.py'"
        ),
        (
            "set -e\nbuiltin export CHECK='python3.11 "
            "scripts/check_contract_boundaries.py'"
        ),
        "set -e\nenv CHECK='python3.11 scripts/check_contract_boundaries.py'",
        "set -e\nenv -i CHECK='python3.11 scripts/check_contract_boundaries.py'",
        "set -e\nnohup echo python3.11 scripts/check_contract_boundaries.py",
        "set -e\ntime echo python3.11 scripts/check_contract_boundaries.py",
        "set -e\ntimeout 5 echo python3.11 scripts/check_contract_boundaries.py",
        (
            "set -e\ntimeout -s TERM 5 echo python3.11 "
            "scripts/check_contract_boundaries.py"
        ),
        "set -e\ntime -p echo python3.11 scripts/check_contract_boundaries.py",
        "set -e\nnice -n 5 echo python3.11 scripts/check_contract_boundaries.py",
        "set -e\nsudo -u root echo python3.11 scripts/check_contract_boundaries.py",
        "set -e\nstdbuf -oL echo python3.11 scripts/check_contract_boundaries.py",
        "set -e\ncommand -v python3.11 scripts/check_contract_boundaries.py",
    ],
)
def test_sourcecraft_rejects_a_gate_behind_a_command_prefix(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "check_contract_boundaries.py" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "script",
    [
        "set -e\ntest -f scripts/check_contract_boundaries.py",
        "set -e\ngrep -q import scripts/check_contract_boundaries.py",
        "set -e\nhash python3.11 scripts/check_contract_boundaries.py",
        'set -e\neval "echo python3.11 scripts/check_contract_boundaries.py"',
        'set -e\nbash -c "echo python3.11 scripts/check_contract_boundaries.py"',
        (
            "set -e\npython3.11 -c \"print('python3.11 "
            "scripts/check_contract_boundaries.py')\""
        ),
        (
            "set -e\nenv CHECK='python3.11 "
            'scripts/check_contract_boundaries.py\' python3.11 -c "print(1)"'
        ),
        "set -e\nls scripts/check_contract_boundaries.py",
        "set -e\npython3.11 -m py_compile scripts/check_contract_boundaries.py",
        "set -e\nruby -e 'puts \"scripts/check_contract_boundaries.py\"'",
        "set -e\nnode -e 'console.log(\"scripts/check_contract_boundaries.py\")'",
        ("set -e\nX='echo hi\npython3.11 scripts/check_contract_boundaries.py'"),
        ('set -e\nX="echo hi\npython3.11 scripts/check_contract_boundaries.py"'),
        "set -e\npython3.11 --version scripts/check_contract_boundaries.py",
        "set -e\nnode --version scripts/check_contract_boundaries.py",
        "set -e\npython3.11 scripts/check_contract_boundaries.py --help",
        "set -e\nenv --version python3.11 scripts/check_contract_boundaries.py",
        "set -e\npython3.11 scripts/check_release_configuration.py --help",
        "set -e\npython3.11 -m pytest tests/contracts --version",
        "set -e\npython3.11 -m pytest tests/contracts --collect-only",
        (
            "set -e\npython3.11 scripts/check_contract_boundaries.py "
            "--repo-root /tmp/other-tree"
        ),
        "set -e\nnice -n 5 --version python3.11 scripts/check_contract_boundaries.py",
        (
            "set -e\ntimeout --foreground --version 5 python3.11 "
            "scripts/check_contract_boundaries.py"
        ),
        "set -e\nstdbuf -o0 --version python3.11 scripts/check_contract_boundaries.py",
        (
            "set -e\nenv -S 'scripts/check_contract_boundaries.py "
            "--repo-root /tmp/other-tree'"
        ),
        (
            "set -e\nenv --split-string='scripts/check_contract_boundaries.py "
            "--repo-root /tmp/other-tree'"
        ),
        ("set -e\nenv -vS 'python3.11 scripts/check_release_configuration.py --help'"),
        ("set -e\nenv -iS 'python3.11 -m pytest tests/contracts --collect-only'"),
        (
            "set -e\nenv --split 'python3.11 -m ruff check src/ "
            "configurable-automl-contracts/src/ --help'"
        ),
        (
            "set -e\nenv --split-str 'python3.11 "
            "scripts/check_release_configuration.py --help'"
        ),
        ("set -e\nX=$'a\\'$(python3.11 scripts/check_contract_boundaries.py)'"),
        "set -e\ncommand -pv python3.11 scripts/check_contract_boundaries.py",
        "set -e\ncommand -p -v python3.11 scripts/check_contract_boundaries.py",
        "set -e\nsetsid -f python3.11 scripts/check_contract_boundaries.py",
        "set -e\nsetsid --fork python3.11 scripts/check_contract_boundaries.py",
        "set -e\nsudo -b python3.11 scripts/check_contract_boundaries.py",
        "set -e\nsudo --background python3.11 scripts/check_contract_boundaries.py",
        "set -e\nsetsid --for python3.11 scripts/check_contract_boundaries.py",
        "set -e\nsetsid --f python3.11 scripts/check_contract_boundaries.py",
        "set -e\nsudo --backg python3.11 scripts/check_contract_boundaries.py",
        "set -e\nsudo --back python3.11 scripts/check_contract_boundaries.py",
        "set -e\nsudo -bn python3.11 scripts/check_contract_boundaries.py",
        "set -e\nsudo -nb python3.11 scripts/check_contract_boundaries.py",
    ],
)
def test_sourcecraft_rejects_a_gate_only_in_an_argument(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "check_contract_boundaries.py" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "script",
    [
        'set -e\npython3.11 -c "import importlinter.cli"',
        'set -e\npython3.11 -c "import importlinter.cli; print(1)"',
        (
            'set -e\npython3.11 -c "from importlinter.cli import lint_imports; '
            'lint_imports()"'
        ),
        'set -e\npython3.11 -c "import importlinter.cli; print(importlinter.cli)"',
        (
            'set -e\npython3.11 -c "from importlinter.cli import lint_imports; '
            'print(lint_imports)"'
        ),
        (
            'set -e\npython3.11 -c "import sys; '
            "from importlinter.cli import lint_imports; "
            'sys.exit(0 * lint_imports())"'
        ),
        (
            'set -e\npython3.11 -c "from importlinter.cli import lint_imports; '
            'SystemExit(lint_imports())"'
        ),
        (
            'set -e\npython3.11 -c "import sys; import importlinter.cli; '
            'sys.exit(importlinter.cli.other())"'
        ),
        "set -e\npython3.11 -m importlinter.cli",
        "set -e\npython3.11 -m importlinter.cli ignored",
        (
            'set -e\npython3.11 -c "import sys; '
            "from importlinter.cli import lint_imports; "
            'sys.exit(lint_imports()) if False else None"'
        ),
        (
            'set -e\npython3.11 -c "import sys; '
            "from importlinter.cli import lint_imports; "
            'f = lambda: sys.exit(lint_imports())"'
        ),
        (
            'set -e\npython3.11 -c "import sys; '
            "from importlinter.cli import lint_imports; "
            'sys.exit = print; sys.exit(lint_imports())"'
        ),
        (
            'set -e\npython3.11 -c "import sys; '
            "from importlinter.cli import lint_imports; "
            'lint_imports = lambda: 0; sys.exit(lint_imports())"'
        ),
    ],
)
def test_sourcecraft_rejects_an_import_only_linter_marker(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "importlinter.cli" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    ("script", "marker"),
    [
        (
            (
                "set -e\npython3.11 -m ruff check src/ "
                "configurable-automl-contracts/src/ @/tmp/ruffopts"
            ),
            "python3.11 -m ruff check src/ configurable-automl-contracts/src/",
        ),
        (
            "set -e\npython3.11 -m mypy scripts/contract_release.py @/tmp/mypyopts",
            "python3.11 -m mypy scripts/contract_release.py",
        ),
    ],
)
def test_sourcecraft_rejects_tool_response_files(
    guard: ModuleType, sourcecraft_text: str, script: str, marker: str
) -> None:
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert f"has no executable '{marker}'" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "line",
    [
        "set $'\\x2be'",
        'set "$(echo +e)"',
        "FLAG=+e; set $FLAG",
        'O=errexit; set +o "$O"',
    ],
)
def test_sourcecraft_rejects_non_literal_errexit_options(
    guard: ModuleType, sourcecraft_text: str, line: str
) -> None:
    script = f"set -e\n{line}\npython3.11 scripts/check_contract_boundaries.py"
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "disables error handling" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        ("set '+e'", "disables error handling"),
        ("exit 0", "can exit before required gates"),
        ("source /tmp/stub.sh", "sources another script"),
        ("echo(){ exit 0; }", "shadows required command"),
    ],
)
def test_sourcecraft_rejects_backslash_continuation_inside_quotes(
    guard: ModuleType, sourcecraft_text: str, tail: str, expected: str
) -> None:
    script = (
        'set -e\necho "unterminated \\\n'
        f'" ; {tail}\n'
        "python3.11 scripts/check_contract_boundaries.py"
    )
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert expected in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize("connector", ["||", "&&", "|", "&"])
def test_sourcecraft_rejects_a_gate_in_a_short_circuited_subshell(
    guard: ModuleType, sourcecraft_text: str, connector: str
) -> None:
    script = (
        "set -e\n"
        f"true {connector} (\n"
        "python3.11 scripts/check_contract_boundaries.py\n"
        ")"
    )
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "check_contract_boundaries.py" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize("option", ["--collect-only", "--co"])
def test_sourcecraft_rejects_pytest_collection_only_in_quality_checks(
    guard: ModuleType, sourcecraft_text: str, option: str
) -> None:
    document = load_document(sourcecraft_text)
    for task in document["workflows"]["quality-checks"]["tasks"]:
        for cube in task["cubes"]:
            if cube["name"] == "contracts-static-checks":
                cube["script"] = [
                    cube_script(cube).replace(
                        "python3.11 -m pytest tests/contracts",
                        f"pytest tests/contracts {option}",
                    )
                ]
    text = dump_document(document)
    assert "no executable 'pytest tests/contracts'" in messages(
        guard.check_sourcecraft_ci(text)
    )


@pytest.mark.parametrize(
    "line",
    [
        "if false; then\n  echo hi ;# ; fi",
        "while false; do\n  echo hi ;# ; done",
        "case x in\n  x) echo hi ;# ;; esac",
        "true || (\n  echo hi ;# ; )",
    ],
)
def test_sourcecraft_rejects_a_comment_after_an_operator(
    guard: ModuleType, sourcecraft_text: str, line: str
) -> None:
    script = f"set -e\n{line}\npython3.11 scripts/check_contract_boundaries.py"
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "has no executable 'scripts/check_contract_boundaries.py'" in messages(
        guard.check_sourcecraft_ci(text)
    )


@pytest.mark.parametrize(
    "line",
    [
        "# note <<EOF\nexit 0\nEOF",
        "echo $(( 1 << y ))\nexit 0\ny",
    ],
)
def test_sourcecraft_rejects_fake_heredocs(
    guard: ModuleType, sourcecraft_text: str, line: str
) -> None:
    script = f"set -e\n{line}\npython3.11 scripts/check_contract_boundaries.py"
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "can exit before required gates" in messages(
        guard.check_sourcecraft_ci(text)
    )


@pytest.mark.parametrize(
    "line",
    [
        "shopt -u -o errexit",
        "builtin shopt -u -o errexit",
        "command shopt -u -o errexit",
        "shopt -su -o errexit",
        'O=errexit; shopt -u -o "$O"',
    ],
)
def test_sourcecraft_rejects_shopt_disabling_errexit(
    guard: ModuleType, sourcecraft_text: str, line: str
) -> None:
    script = f"set -e\n{line}\npython3.11 scripts/check_contract_boundaries.py"
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "disables error handling" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_a_cmdsub_close_popping_a_skipped_subshell(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    script = (
        "set -e\ntrue || (\nx=$(\n)\npython3.11 scripts/check_contract_boundaries.py\n)"
    )
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "has no executable 'scripts/check_contract_boundaries.py'" in messages(
        guard.check_sourcecraft_ci(text)
    )


def test_sourcecraft_requires_set_e_in_quality_checks_cubes(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    document = load_document(sourcecraft_text)
    for task in document["workflows"]["quality-checks"]["tasks"]:
        for cube in task["cubes"]:
            if cube["name"] == "contracts-static-checks":
                cube["script"] = [
                    cube_script(cube)
                    .replace("set -e\n", "", 1)
                    .replace(
                        "python3.11 -m pytest tests/contracts",
                        "python3.11 -m pytest tests/contracts\ntrue",
                        1,
                    )
                ]
    text = dump_document(document)
    assert "must enable 'set -e'" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_allows_exec_redirect_and_build_dot(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    def add_commands(workflow: dict[str, object]) -> None:
        cube = find_cube(workflow, "quality-gates")
        cube["script"] = [
            cube_script(cube).replace(
                "set -e\n",
                "set -e\nexec 3>&1\nshopt -s -o errexit\n"
                "python3.11 -m build . --outdir dist/engine\n",
                1,
            )
        ]

    text = mutate_publish_workflow(sourcecraft_text, add_commands)
    assert guard.check_sourcecraft_ci(text) == []


@pytest.mark.parametrize(
    "script",
    [
        "set -e\npython3.11 -m twine check --help",
        "set -e\ntwine check --help",
    ],
)
def test_sourcecraft_rejects_twine_help(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    text = replace_gate_script(sourcecraft_text, "build-and-verify", script)
    assert "twine check" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "script",
    [
        "set -e\npython3.11() { :; }\npython3.11 scripts/check_contract_boundaries.py",
        "set -e\nalias python3.11=:\npython3.11 scripts/check_contract_boundaries.py",
        (
            "set -e\nhash -p /bin/true python3.11\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        "set -e\npytest() { :; }\npython3.11 -m pytest tests/contracts",
        (
            'set -e\nhash -p /bin/true "python3.11"\n'
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            'set -e\nshopt -s expand_aliases\nalias "python3.11"=:\n'
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\nif true; then python3.11() { return 0; }; fi\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\npython3.11 () { return 0; }\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\npython3.11 ( ) { return 0; }\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\neval 'python3.11() { return 0; }'\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\nalias -- python3.11=:\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\nhash -p/bin/true python3.11\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        ("set -e\npython3.11(){ :; }\npython3.11 scripts/check_contract_boundaries.py"),
        (
            "set -e\nhash -p /bin/true -- python3.11\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\neval 'set(){ :; }; set -e'\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\necho(){ exit 0; }\necho banner\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
    ],
)
def test_sourcecraft_rejects_command_shadowing(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "shadows required command" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "script",
    [
        "set -e\nsource stub.sh\npython3.11 scripts/check_contract_boundaries.py",
        "set -e\n. stub.sh\npython3.11 scripts/check_contract_boundaries.py",
        (
            "set -e\nsource <(echo 'python3.11() { :; }')\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        ('set -e\nsource "stub.sh"\npython3.11 scripts/check_contract_boundaries.py'),
        ("set -e\n. 'stub.sh'\npython3.11 scripts/check_contract_boundaries.py"),
        (
            "set -e\neval 'source stub.sh'\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\ncommand source stub.sh\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        ("set -e\nX=1 source stub.sh\npython3.11 scripts/check_contract_boundaries.py"),
        ("set -e\nsour\\\nce stub.sh\npython3.11 scripts/check_contract_boundaries.py"),
        (
            "set -e\n{ source stub.sh; }\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\nif true; then source stub.sh; fi\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
    ],
)
def test_sourcecraft_rejects_sourced_scripts(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "sources another script" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "script",
    [
        (
            "set -e\nexport PATH=/tmp/stub:$PATH\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        "set -e\nPATH=/tmp/stub:$PATH\npython3.11 scripts/check_contract_boundaries.py",
        (
            "set -e\nexport PYTHONPATH=/tmp/stub\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        "set -e\nenv PATH=/tmp/stub python3.11 scripts/check_contract_boundaries.py",
        (
            "set -e\ndeclare -x PATH=/tmp/stub:$PATH\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\ntypeset -x PATH=/tmp/stub:$PATH\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\nreadonly PATH=/tmp/stub:$PATH\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\nbuiltin export PATH=/tmp/stub:$PATH\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\nprintf -v PATH '%s' /tmp/stub\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\neval 'export PATH=/tmp/stub:$PATH'\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            'set -e\nexport "PATH=/tmp/stub:$PATH"\n'
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\nexpor\\\nt PATH=/tmp/stub:$PATH\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\nexport PYTEST_ADDOPTS=--collect-only\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\nexport BASH_ENV=/tmp/stub\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\nexport PYTEST_ADDOPTS+=--collect-only\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\nexport PYTHONPATH+=/tmp/fake\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            'set -e\nexport PATH\nread -r PATH <<< "/tmp/fake:$PATH"\n'
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            'set -e\nK=PATH; export "$K=/tmp/stub"\n'
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            'set -e\nK=PATH; read -r "$K" <<< "/tmp/fake:$PATH"\n'
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            'set -e\nK=PYTHONPATH; printf -v "$K" "%s" /tmp/fake\n'
            "python3.11 scripts/check_contract_boundaries.py"
        ),
    ],
)
def test_sourcecraft_rejects_path_overrides(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "overrides PATH" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    ("scope", "key"),
    [
        ("document", "PATH"),
        ("workflow", "PYTHONPATH"),
        ("task", "BASH_ENV"),
        ("cube", "PYTEST_ADDOPTS"),
        ("cube", "PYTEST_PLUGINS"),
    ],
)
def test_sourcecraft_rejects_gate_env_keys(
    guard: ModuleType, sourcecraft_text: str, scope: str, key: str
) -> None:
    document = load_document(sourcecraft_text)
    if scope == "document":
        document["env"] = {key: "/tmp/stub"}
        text = dump_document(document)
    else:

        def add_env(workflow: dict[str, object]) -> None:
            if scope == "workflow":
                workflow["env"] = {key: "/tmp/stub"}
            elif scope == "task":
                workflow["tasks"][0]["env"] = {key: "/tmp/stub"}  # type: ignore[index]
            else:
                find_cube(workflow, "quality-gates")["env"] = {key: "/tmp/stub"}

        text = mutate_publish_workflow(sourcecraft_text, add_env)
    violations = messages(guard.check_sourcecraft_ci(text))
    assert key in violations
    assert "env must not set" in violations


@pytest.mark.parametrize(
    "script",
    [
        "set -e\ntrap 'exit 0' ERR\npython3.11 scripts/check_contract_boundaries.py",
        "set -e\ntrap 'exit 0' EXIT\npython3.11 scripts/check_contract_boundaries.py",
        (
            "set -e\nbuiltin trap 'exit 0' ERR\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\n{ trap 'exit 0' ERR; }\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "set -e\nif true; then trap 'exit 0' ERR; fi\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
    ],
)
def test_sourcecraft_rejects_traps(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "uses a trap" in messages(guard.check_sourcecraft_ci(text))


def test_sourcecraft_rejects_disabled_builtins(
    guard: ModuleType, sourcecraft_text: str
) -> None:
    script = "set -e\nenable -n set\npython3.11 scripts/check_contract_boundaries.py"
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "disables a shell builtin" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "script",
    [
        "set -e\nexec true\npython3.11 scripts/check_contract_boundaries.py",
        "set -e\n{ exit 0; }\npython3.11 scripts/check_contract_boundaries.py",
        (
            "set -e\nif true; then exit 0; fi\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        "set -e\neval 'exit 0'\npython3.11 scripts/check_contract_boundaries.py",
    ],
)
def test_sourcecraft_rejects_early_termination(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "can exit before required gates" in messages(
        guard.check_sourcecraft_ci(text)
    )


@pytest.mark.parametrize(
    "script",
    [
        "set -e\nE=exit; $E 0\npython3.11 scripts/check_contract_boundaries.py",
        "set -e\nE=eval; $E 'set +e'\npython3.11 scripts/check_contract_boundaries.py",
        "set -e\n$'\\x65xit' 0\npython3.11 scripts/check_contract_boundaries.py",
        (
            "set -e\nunused=1 $'\\x65xit' 0\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
    ],
)
def test_sourcecraft_rejects_dynamic_command_names(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "command name produced by" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "script",
    [
        (
            'set -e\neval "$(cat scripts/stub.sh)"\n'
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            'set -e\neval "`cat scripts/stub.sh`"\n'
            "python3.11 scripts/check_contract_boundaries.py"
        ),
    ],
)
def test_sourcecraft_rejects_dynamic_eval(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "command substitution with 'eval'" in messages(
        guard.check_sourcecraft_ci(text)
    )


@pytest.mark.parametrize(
    "block",
    [
        (
            'if [ -n "$PUBLISH_CONTRACTS" ]; then\n'
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "fi"
        ),
        (
            'if [ -n "$PUBLISH_CONTRACTS" ]; then\n'
            "  echo enabled\n"
            'elif [ "$PUBLISH_CONTRACTS" = 1 ]; then\n'
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "fi"
        ),
        (
            "if true; then\n"
            "  echo enabled\n"
            "else\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "fi"
        ),
        (
            'case "$PUBLISH_CONTRACTS" in\n'
            "  1)\n"
            "    python3.11 scripts/check_contract_boundaries.py\n"
            "    ;;\n"
            "esac"
        ),
        (
            'while [ -n "$PUBLISH_CONTRACTS" ]; do\n'
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "  break\n"
            "done"
        ),
        (
            'until [ -n "$PUBLISH_CONTRACTS" ]; do\n'
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "  break\n"
            "done"
        ),
        (
            "for value in one two; do\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "done"
        ),
        "if true; then\n  python3.11 scripts/check_contract_boundaries.py\nfi",
        "if [ 1 = 1 ]; then\n  python3.11 scripts/check_contract_boundaries.py\nfi",
        (
            'if [ -n "$X" ]; then\n'
            '  if [ -n "$Y" ]; then\n'
            "    python3.11 scripts/check_contract_boundaries.py\n"
            "  fi\n"
            "fi"
        ),
        "while true; do\n  python3.11 scripts/check_contract_boundaries.py\ndone",
        (
            "select value in one two; do\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "  break\n"
            "done"
        ),
        (
            "select value in one two; do\n  "
            "python3.11 scripts/check_contract_boundaries.py; break\ndone"
        ),
        (
            'X=$(if [ -n "$PUBLISH_CONTRACTS" ]; then\n'
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "fi)"
        ),
        (
            "X=$(for value in one two; do\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "done)"
        ),
        (
            'X=$(while [ -n "$PUBLISH_CONTRACTS" ]; do\n'
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "  break\n"
            "done)"
        ),
        (
            "X=$(select value in one two; do\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "  break\n"
            "done)"
        ),
        ("foo() (\n  python3.11 scripts/check_contract_boundaries.py\n)"),
        ("foo()\n(\n  python3.11 scripts/check_contract_boundaries.py\n)"),
        (
            "foo()\n"
            "# body follows\n"
            "(\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            ")"
        ),
        ("function foo()\n(\n  python3.11 scripts/check_contract_boundaries.py\n)"),
        ("function foo (\n  python3.11 scripts/check_contract_boundaries.py\n)"),
        ("foo(){\n  python3.11 scripts/check_contract_boundaries.py\n}"),
        (
            "foo() (\n"
            "  (\n"
            "    true\n"
            "  )\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            ")"
        ),
        (
            "echo start; if false; then\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "fi"
        ),
        (
            "true && if false; then\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "fi"
        ),
        (
            "true || for value in one two; do\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "done"
        ),
        (
            "echo start; case value in\n"
            "  1)\n"
            "    python3.11 scripts/check_contract_boundaries.py\n"
            "    ;;\n"
            "esac"
        ),
        (
            "X=$(echo hi; if false; then\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "fi)"
        ),
        (
            "X=$(echo hi && while true; do\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "  break\n"
            "done)"
        ),
        ("foo() { echo hi\n  python3.11 scripts/check_contract_boundaries.py\n}"),
        ("foo() ( echo hi\n  python3.11 scripts/check_contract_boundaries.py\n)"),
        ("foo()\n{ echo hi\n  python3.11 scripts/check_contract_boundaries.py\n}"),
        (
            "function foo { echo hi\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "}"
        ),
        (
            "echo start; foo() { echo hi\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "}"
        ),
        (
            "if true; then :; fi; if false; then\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "fi"
        ),
        ("{ if false; then\n  python3.11 scripts/check_contract_boundaries.py\nfi\n}"),
        (
            "if true; then :; fi; while false; do\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "done"
        ),
        ("X=`if false; then\n  python3.11 scripts/check_contract_boundaries.py\nfi`"),
        ("foo.bar() (\n  python3.11 scripts/check_contract_boundaries.py\n)"),
        ("1foo() {\n  python3.11 scripts/check_contract_boundaries.py\n}"),
        ("function foo:bar {\n  python3.11 scripts/check_contract_boundaries.py\n}"),
        (
            'echo "a\\"b" ; if false; then\n'
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "fi"
        ),
        (
            "echo a ; time if false; then\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "fi"
        ),
        (
            "time while false; do\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "done"
        ),
        (
            "time for value in one; do\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "done"
        ),
        (
            "time case value in\n"
            "  1)\n"
            "    python3.11 scripts/check_contract_boundaries.py\n"
            "    ;;\n"
            "esac"
        ),
        ("time foo() {\n  python3.11 scripts/check_contract_boundaries.py\n}"),
        ("time function foo {\n  python3.11 scripts/check_contract_boundaries.py\n}"),
        (
            "echo $'a\\'b' ; if false; then\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "fi"
        ),
        ("(\n  python3.11 scripts/check_contract_boundaries.py\n) &"),
        ("(\n  python3.11 scripts/check_contract_boundaries.py\n) | cat"),
        ("(\n  python3.11 scripts/check_contract_boundaries.py\n) || echo failed"),
        ("(\n  python3.11 scripts/check_contract_boundaries.py\n) && echo ok"),
        ("(\n  python3.11 scripts/check_contract_boundaries.py\n) >/dev/null &"),
        ("(\n  python3.11 scripts/check_contract_boundaries.py\n) 2>&1 | cat"),
        (
            "(\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            ") >/dev/null || echo failed"
        ),
        ("(\n  python3.11 scripts/check_contract_boundaries.py\n) 2>&1 && echo ok"),
        (
            "time ! if false; then\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "fi"
        ),
        (
            "time -p ! while false; do\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "done"
        ),
        (
            "time -p ! for value in one; do\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            "done"
        ),
        (
            "time -p ! case value in\n"
            "  1)\n"
            "    python3.11 scripts/check_contract_boundaries.py\n"
            "    ;;\n"
            "esac"
        ),
        ("time -p ! foo() {\n  python3.11 scripts/check_contract_boundaries.py\n}"),
        (
            "foo () (\n"
            "  arr=(\n"
            "    one\n"
            "    two\n"
            "  )\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            ")"
        ),
        (
            "foo () (\n"
            "  value=$(\n"
            "    echo hi\n"
            "  )\n"
            "  python3.11 scripts/check_contract_boundaries.py\n"
            ")"
        ),
    ],
)
def test_sourcecraft_rejects_a_gate_inside_runtime_conditions(
    guard: ModuleType, sourcecraft_text: str, block: str
) -> None:
    script = "set -e\n" + block
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "check_contract_boundaries.py" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "script",
    [
        "set -e\nalias boundaries='python3.11 scripts/check_contract_boundaries.py'",
        "set -e\nexport CHECK='python3.11 scripts/check_contract_boundaries.py'",
        "set -e\ndeclare CHECK='python3.11 scripts/check_contract_boundaries.py'",
        "set -e\ntypeset CHECK='python3.11 scripts/check_contract_boundaries.py'",
        "set -e\nlocal CHECK='python3.11 scripts/check_contract_boundaries.py'",
        "set -e\nreadonly CHECK='python3.11 scripts/check_contract_boundaries.py'",
        "set -e\nunset CHECK='python3.11 scripts/check_contract_boundaries.py'",
        "set -e\nset -- python3.11 scripts/check_contract_boundaries.py",
        "set -e\ntrap 'python3.11 scripts/check_contract_boundaries.py' EXIT",
        (
            "set -e\nalias boundaries='python3.11 "
            "scripts/check_contract_boundaries.py'\nboundaries"
        ),
        (
            "set -e\nalias boundaries='python3.11 \\\n"
            "scripts/check_contract_boundaries.py'\nboundaries"
        ),
    ],
)
def test_sourcecraft_rejects_a_gate_only_declared_or_aliased(
    guard: ModuleType, sourcecraft_text: str, script: str
) -> None:
    text = replace_gate_script(sourcecraft_text, "quality-gates", script)
    assert "check_contract_boundaries.py" in messages(guard.check_sourcecraft_ci(text))


@pytest.mark.parametrize(
    "script",
    [
        "python3.11 scripts/check_contract_boundaries.py\nset -e",
        "set -e\nset +e\npython3.11 scripts/check_contract_boundaries.py",
        "set -e\nset +o errexit\npython3.11 scripts/check_contract_boundaries.py",
        "set -e\ncommand set +e\npython3.11 scripts/check_contract_boundaries.py",
        "set -e\nbuiltin set +e\npython3.11 scripts/check_contract_boundaries.py",
        'set -e\neval "set +e"\npython3.11 scripts/check_contract_boundaries.py',
        (
            "set -e\ncommand set +o errexit\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        "env --version set -e\npython3.11 scripts/check_contract_boundaries.py",
        "nice set -e\npython3.11 scripts/check_contract_boundaries.py",
        "timeout 5 set -e\npython3.11 scripts/check_contract_boundaries.py",
        "env set -e\npython3.11 scripts/check_contract_boundaries.py",
        "nohup set -e\npython3.11 scripts/check_contract_boundaries.py",
        "stdbuf -o0 set -e\npython3.11 scripts/check_contract_boundaries.py",
        "sudo -u root set -e\npython3.11 scripts/check_contract_boundaries.py",
        "setsid set -e\npython3.11 scripts/check_contract_boundaries.py",
        "nice -n 5 --version set -e\npython3.11 scripts/check_contract_boundaries.py",
        "set -e | cat\npython3.11 scripts/check_contract_boundaries.py",
        "set -e &\npython3.11 scripts/check_contract_boundaries.py",
        "bash -c 'set -e'\npython3.11 scripts/check_contract_boundaries.py",
        "sh -c 'set -o errexit'\npython3.11 scripts/check_contract_boundaries.py",
        "set -e\nbash -c 'set +e'\npython3.11 scripts/check_contract_boundaries.py",
        "env -S 'set -e'\npython3.11 scripts/check_contract_boundaries.py",
        ("eval 'bash -c \"set -e\"'\npython3.11 scripts/check_contract_boundaries.py"),
        (
            "eval 'sh -c \"set -o errexit\"'\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        (
            "command eval 'bash -c \"set -e\"'\n"
            "python3.11 scripts/check_contract_boundaries.py"
        ),
        ("command -pv set -e\npython3.11 scripts/check_contract_boundaries.py"),
        ("eval 'set -e; set +e'\npython3.11 scripts/check_contract_boundaries.py"),
        ('eval "set -e; set +e"\npython3.11 scripts/check_contract_boundaries.py'),
        ("eval 'false && set -e'\npython3.11 scripts/check_contract_boundaries.py"),
        ("eval 'true || set -e'\npython3.11 scripts/check_contract_boundaries.py"),
        ("command --help set -e\npython3.11 scripts/check_contract_boundaries.py"),
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
                            "--dist configurable-automl-contracts/dist "
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
