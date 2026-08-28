"""Fail-closed coverage for sealed-board dependency contract v5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    PROJECT_DEPENDENCY_PROBE_SCHEMA,
    SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V5,
    preflight_validation_project_dependencies,
)

BOARD = "semantic-preserving-autonomous-remodularization-v1"
TASK_CID = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
TASK_ID = "SPAR-901"
TARGET = "test/api/semantic_refactoring/test_board_selected.py"
SOURCE_OUTPUT = "ipfs_accelerate_py/agent_supervisor/semantic_refactoring/board_selected.py"
OUTPUTS = [SOURCE_OUTPUT, TARGET]
COMMAND = f"python3 -m pytest -q {TARGET}"
PYTEST_REQUIREMENT = "pytest>=8.0.0"
TESTING_EXTRA = [PYTEST_REQUIREMENT, "anyio>=4.0.0"]
LEGACY_TARGET = "test/api/test_legacy_target.py"
LEGACY_COMMAND = f"python -m pytest -q {LEGACY_TARGET}"
LEGACY_CID = "baguqeerabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _content_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _board_block() -> str:
    return f"""
## {TASK_ID} Board-selected task

- Stable task ID: {TASK_ID}
- Board namespace: {BOARD}
- Validation: {COMMAND}
- Outputs: {", ".join(OUTPUTS)}
""".strip()


def _write_project(
    workspace: Path,
    *,
    taskboard_digest: str = "",
    duplicate_block: bool = False,
    policy_extra: str = "",
) -> dict[str, object]:
    requirements_payload = b"requests>=2.31.0\n"
    (workspace / "requirements.txt").write_bytes(requirements_payload)
    legacy_payload = b"def test_legacy():\n    assert True\n"
    legacy_path = workspace / LEGACY_TARGET
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_bytes(legacy_payload)
    board_payload = (_board_block() + "\n").encode("utf-8")
    if duplicate_block:
        board_payload += ("\n" + _board_block() + "\n").encode("utf-8")
    board_path = workspace / "docs/architecture/tasks.todo.md"
    board_path.parent.mkdir(parents=True, exist_ok=True)
    board_path.write_bytes(board_payload)
    policy_extra_line = f"\n{policy_extra}" if policy_extra else ""
    (workspace / "pyproject.toml").write_text(
        f"""
[project]
name = "board-scoped-project"
version = "1.0.0"
requires-python = ">=3.8"
dynamic = ["dependencies"]

[project.optional-dependencies]
testing = {json.dumps(TESTING_EXTRA)}

[tool.setuptools.dynamic]
dependencies = {{ file = ["requirements.txt"] }}

[tool.ipfs-accelerate-agent-supervisor.project-dependency-preflight]
schema = {json.dumps(SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V5)}
requires-python = ">=3.8"
authority = {{ file = "requirements.txt", sha256 = {json.dumps(hashlib.sha256(requirements_payload).hexdigest())}, extra = "testing", extra-requirements-sha256 = {json.dumps(_content_sha256(TESTING_EXTRA))} }}

[[tool.ipfs-accelerate-agent-supervisor.project-dependency-preflight.board-policies]]
board-namespace = {json.dumps(BOARD)}
taskboard-path = "docs/architecture/tasks.todo.md"
taskboard-sha256 = {json.dumps(taskboard_digest or hashlib.sha256(board_payload).hexdigest())}
task-prefix = "SPAR-"
requirements = [{json.dumps(PYTEST_REQUIREMENT)}]{policy_extra_line}

[[tool.ipfs-accelerate-agent-supervisor.project-dependency-preflight.targets]]
target = {json.dumps(LEGACY_TARGET)}
command-target = {json.dumps(LEGACY_TARGET)}
command-kind = "pytest"
validation-command-sha256 = {json.dumps(hashlib.sha256(LEGACY_COMMAND.encode()).hexdigest())}
requirements = [{json.dumps(PYTEST_REQUIREMENT)}]
task = {{ board-namespace = "legacy-board-v1", canonical-task-cid = {json.dumps(LEGACY_CID)}, declared-outputs = [{json.dumps(LEGACY_TARGET)}] }}
baseline = {{ state = "present", sha256 = {json.dumps(hashlib.sha256(legacy_payload).hexdigest())} }}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return {
        "board_sha256": hashlib.sha256(board_payload).hexdigest(),
        "requirements_sha256": hashlib.sha256(requirements_payload).hexdigest(),
        "legacy_sha256": hashlib.sha256(legacy_payload).hexdigest(),
    }


def _passing_probe(payloads: list[dict[str, object]]):
    def probe(payload, **_kwargs):
        payloads.append(payload)
        return {
            "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
            "passed": True,
            "reason": "project_dependencies_satisfied",
            "projects": [],
        }

    return probe


def _task_authority(
    *,
    board_namespace: str = BOARD,
    declared_outputs: list[str] | None = None,
) -> dict[str, object]:
    return {
        "board_namespace": board_namespace,
        "canonical_task_cid": TASK_CID,
        "declared_outputs": (
            list(OUTPUTS)
            if declared_outputs is None
            else declared_outputs
        ),
    }


def test_v5_selects_exact_sealed_board_task_without_reading_target(
    tmp_path: Path,
) -> None:
    expected = _write_project(tmp_path)
    payloads: list[dict[str, object]] = []
    assert not (tmp_path / TARGET).exists()

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [COMMAND],
        task_authority=_task_authority(),
        probe_runner=_passing_probe(payloads),
    )

    assert receipt["passed"] is True
    assert receipt["automatic_install_attempted"] is False
    project = receipt["projects"][0]
    assert project["dependency_contract_schema"] == (
        SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V5
    )
    assert project["scoped_validation_selection_kind"] == "board_policy"
    assert project["scoped_validation_target_materialization_state"] == (
        "not-read-board-policy"
    )
    assert {
        manifest["content_sha256"]
        for manifest in project["dependency_manifests"]
    } == {
        expected["requirements_sha256"],
        expected["board_sha256"],
    }
    assert payloads[0]["projects"][0]["requirements"] == [
        PYTEST_REQUIREMENT
    ]
    assert not (tmp_path / TARGET).exists()


def test_v5_preserves_exact_v4_legacy_target_semantics(
    tmp_path: Path,
) -> None:
    expected = _write_project(tmp_path)
    payloads: list[dict[str, object]] = []

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [LEGACY_COMMAND],
        task_authority={
            "board_namespace": "legacy-board-v1",
            "canonical_task_cid": LEGACY_CID,
            "declared_outputs": [LEGACY_TARGET],
        },
        probe_runner=_passing_probe(payloads),
    )

    assert receipt["passed"] is True
    project = receipt["projects"][0]
    assert "scoped_validation_selection_kind" not in project
    assert {
        manifest["content_sha256"]
        for manifest in project["dependency_manifests"]
    } == {
        expected["requirements_sha256"],
        expected["legacy_sha256"],
    }
    assert payloads[0]["projects"][0]["requirements"] == [
        PYTEST_REQUIREMENT
    ]


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        ("wrong-namespace", "v5_board_policy_not_declared"),
        ("wrong-hash", "v5_taskboard_digest_mismatch"),
        ("wrong-command", "v5_board_validation_command_invalid"),
        ("wrong-output", "v5_taskboard_task_not_exactly_matched"),
        ("duplicate-block", "v5_taskboard_duplicate_task_id"),
        ("unknown-field", "v5_board_policy_fields_invalid"),
    ],
)
def test_v5_board_policy_adversarial_inputs_fail_before_probe(
    tmp_path: Path,
    case: str,
    expected_reason: str,
) -> None:
    kwargs: dict[str, object] = {}
    command = COMMAND
    authority = _task_authority()
    if case == "wrong-hash":
        kwargs["taskboard_digest"] = "0" * 64
    elif case == "duplicate-block":
        kwargs["duplicate_block"] = True
    elif case == "unknown-field":
        kwargs["policy_extra"] = 'model = "forbidden"'
    elif case == "wrong-namespace":
        authority = _task_authority(board_namespace="other-board-v1")
    elif case == "wrong-command":
        command = COMMAND + " --maxfail=1"
    elif case == "wrong-output":
        authority = _task_authority(declared_outputs=[TARGET])
    _write_project(tmp_path, **kwargs)
    payloads: list[dict[str, object]] = []

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=authority,
        probe_runner=_passing_probe(payloads),
    )

    assert receipt["passed"] is False
    assert receipt["reason"] == (
        "project_dependency_contract_collection_failed"
    )
    assert receipt["projects"][0]["contract_error_reason"] == (
        expected_reason
    )
    assert receipt["automatic_install_attempted"] is False
    assert payloads == []


@pytest.mark.parametrize(
    "target",
    [
        "test/*.py",
        "test/?.py",
        "test/[a].py",
        "test/{a}.py",
        "test/$HOME.py",
        "test/`id`.py",
        "test/$(id).py",
        "test/a;b.py",
        "test/a&b.py",
        "test/a|b.py",
        "test/a<b.py",
        "test/a>b.py",
    ],
)
def test_v5_rejects_shell_dynamic_validation_targets_before_probe(
    tmp_path: Path,
    target: str,
) -> None:
    _write_project(tmp_path)
    payloads: list[dict[str, object]] = []
    command = f"python3 -m pytest -q {target}"

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=_task_authority(),
        probe_runner=_passing_probe(payloads),
    )

    assert receipt["passed"] is False
    if target == "test/$(id).py":
        assert receipt["projects"] == []
        assert receipt["invalid_commands"][0]["reason"] == (
            "validation_repository_root_is_unsafe"
        )
    else:
        assert receipt["projects"][0]["contract_error_reason"] == (
            "v5_board_validation_target_invalid"
        )
    assert payloads == []


def test_v5_rejects_quoted_validation_target_before_probe(
    tmp_path: Path,
) -> None:
    _write_project(tmp_path)
    payloads: list[dict[str, object]] = []

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [f'python3 -m pytest -q "{TARGET}"'],
        task_authority=_task_authority(),
        probe_runner=_passing_probe(payloads),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["contract_error_reason"] == (
        "v5_board_validation_command_invalid"
    )
    assert payloads == []


def test_checked_in_v5_contract_selects_spar_001_board_task() -> None:
    root = Path(__file__).resolve().parents[2]
    payloads: list[dict[str, object]] = []

    receipt = preflight_validation_project_dependencies(
        root,
        [
            "python3 -m pytest -q "
            "test/api/semantic_refactoring/test_capability_matrix.py"
        ],
        task_authority={
            "board_namespace": BOARD,
            "canonical_task_cid": TASK_CID,
            "declared_outputs": [
                "docs/architecture/semantic_preserving_autonomous_"
                "remodularization_inventory/verified_capability_matrix.json",
                "test/api/semantic_refactoring/test_capability_matrix.py",
            ],
        },
        probe_runner=_passing_probe(payloads),
    )

    assert receipt["passed"] is True
    assert receipt["projects"][0]["dependency_contract_schema"] == (
        SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V5
    )
    assert payloads[0]["projects"][0]["requirements"] == [
        PYTEST_REQUIREMENT
    ]
