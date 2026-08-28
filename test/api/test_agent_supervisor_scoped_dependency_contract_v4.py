"""Fail-closed coverage for file-backed scoped dependency contract v4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.project_dependency_preflight import (
    PROJECT_DEPENDENCY_PROBE_SCHEMA,
    SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V4,
    SCOPED_PROJECT_DEPENDENCY_PRIOR_SEED_SCHEMA,
    _require_safe_scoped_pytest_target,
    preflight_validation_project_dependencies,
)

BOARD = "logic-governed-compositional-verification-fabric-v1"
TASK_CID = "baguqeera22uu4o4ux6kzp4fgv5gxqupas3nhdjtbrtt73x2kc6mhtxkdbtwq"
TARGET = "test/api/test_agent_supervisor_program_repair_egraph.py"
COMMAND = f"python -m pytest -q {TARGET}"
PYTEST_REQUIREMENT = "pytest>=8.0.0"
TESTING_EXTRA = [PYTEST_REQUIREMENT, "anyio>=4.0.0"]


def _content_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_project(
    workspace: Path,
    *,
    schema: str = SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V4,
    authority_file: str = "requirements.txt",
    dynamic_file: str = "requirements.txt",
    authority_digest: str = "",
    extra_digest: str = "",
    selected_requirements: list[str] | None = None,
    target: str = TARGET,
    baseline_state: str = "present",
) -> dict[str, object]:
    requirements_payload = b"requests>=2.31.0\n"
    (workspace / "requirements.txt").write_bytes(requirements_payload)
    target_payload = b"import pytest\n"
    target_path = workspace / target
    if baseline_state == "present":
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(target_payload)
        baseline = (
            'state = "present", sha256 = '
            + json.dumps(hashlib.sha256(target_payload).hexdigest())
        )
    else:
        assert baseline_state == "declared-output-absent"
        baseline = 'state = "declared-output-absent"'
    selected = (
        [PYTEST_REQUIREMENT]
        if selected_requirements is None
        else selected_requirements
    )
    command = f"python -m pytest -q {target}"
    (workspace / "pyproject.toml").write_text(
        f"""
[project]
name = "file-backed-scoped-project"
version = "1.0.0"
requires-python = ">=3.8"
dynamic = ["dependencies"]

[project.optional-dependencies]
testing = {json.dumps(TESTING_EXTRA)}

[tool.setuptools.dynamic]
dependencies = {{ file = [{json.dumps(dynamic_file)}] }}

[tool.ipfs-accelerate-agent-supervisor.project-dependency-preflight]
schema = {json.dumps(schema)}
requires-python = ">=3.8"
authority = {{ file = {json.dumps(authority_file)}, sha256 = {json.dumps(authority_digest or hashlib.sha256(requirements_payload).hexdigest())}, extra = "testing", extra-requirements-sha256 = {json.dumps(extra_digest or _content_sha256(TESTING_EXTRA))} }}

[[tool.ipfs-accelerate-agent-supervisor.project-dependency-preflight.targets]]
target = {json.dumps(target)}
command-target = {json.dumps(target)}
command-kind = "pytest"
validation-command-sha256 = {json.dumps(hashlib.sha256(command.encode()).hexdigest())}
requirements = {json.dumps(selected)}
task = {{ board-namespace = {json.dumps(BOARD)}, canonical-task-cid = {json.dumps(TASK_CID)}, declared-outputs = [{json.dumps(target)}] }}
baseline = {{ {baseline} }}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return {
        "command": command,
        "target": target,
        "target_sha256": hashlib.sha256(target_payload).hexdigest(),
        "authority_sha256": hashlib.sha256(requirements_payload).hexdigest(),
        "task_authority": {
            "board_namespace": BOARD,
            "canonical_task_cid": TASK_CID,
            "declared_outputs": [target],
        },
    }


def _prior_seed_authority(
    baseline_receipt: dict[str, object],
    task_authority: dict[str, object],
    target: str,
    payload: bytes,
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": SCOPED_PROJECT_DEPENDENCY_PRIOR_SEED_SCHEMA,
        "board_namespace": task_authority["board_namespace"],
        "canonical_task_cid": task_authority["canonical_task_cid"],
        "baseline_receipt": baseline_receipt,
        "baseline_commit_id": "b" * 40,
        "repository_tree_id": "git-tree:" + "c" * 40,
        "proposal_repository_tree_id": "b" * 40,
        "source_proposal_id": "source-proposal-v4-seed",
        "source_proposal_receipt_id": "source-receipt-v4-seed",
        "proposal_id": "proposal-v4-seed",
        "proposal_receipt_id": "receipt-v4-seed",
        "changed_paths": [target],
        "authorized_paths": list(task_authority["declared_outputs"]),
        "seeded_outputs": [
            {
                "path": target,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "git_blob_id": "a" * 40,
            }
        ],
    }
    body["authority_sha256"] = _content_sha256(body)
    return body


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


def test_v4_selects_exact_file_backed_target_and_probe_payload(
    tmp_path: Path,
) -> None:
    expected = _write_project(tmp_path)
    payloads: list[dict[str, object]] = []

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [str(expected["command"])],
        task_authority=expected["task_authority"],
        probe_runner=_passing_probe(payloads),
    )

    assert receipt["passed"] is True
    project = receipt["projects"][0]
    assert project["dependency_contract_schema"] == (
        SCOPED_PROJECT_DEPENDENCY_CONTRACT_SCHEMA_V4
    )
    assert project["dependency_source"] == (
        "agent_supervisor_scoped_file_extra"
    )
    assert project["selected_validation_extras"] == ["testing"]
    assert project["requirement_count"] == 1
    assert {
        item["content_sha256"] for item in project["dependency_manifests"]
    } == {expected["authority_sha256"], expected["target_sha256"]}
    probed = payloads[0]["projects"][0]
    assert probed["requirements"] == [PYTEST_REQUIREMENT]
    assert probed["requirement_marker_extras"] == ["testing"]
    assert probed["scoped_validation_extra"] == "testing"


@pytest.mark.parametrize(
    "baseline_state",
    ["present", "declared-output-absent"],
)
def test_v4_authenticates_prior_seed_target_replay(
    tmp_path: Path,
    baseline_state: str,
) -> None:
    expected = _write_project(tmp_path, baseline_state=baseline_state)
    command = str(expected["command"])
    task_authority = expected["task_authority"]
    assert isinstance(task_authority, dict)
    baseline_receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
        probe_runner=_passing_probe([]),
    )
    assert baseline_receipt["passed"] is True

    payload = b"import pytest\n\ndef test_replayed():\n    assert True\n"
    target = tmp_path / str(expected["target"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    authority = _prior_seed_authority(
        baseline_receipt,
        task_authority,
        str(expected["target"]),
        payload,
    )

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
        prior_seed_authority=authority,
        probe_runner=_passing_probe([]),
    )

    assert receipt["passed"] is True
    project = receipt["projects"][0]
    assert project[
        "scoped_validation_target_materialization_state"
    ] == "authenticated-prior-seed"
    assert project[
        "scoped_validation_prior_seed_authority_sha256"
    ] == authority["authority_sha256"]


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("missing-authority", "v2_present_target_digest_mismatch"),
        ("content-drift", "v2_prior_seed_target_content_mismatch"),
    ],
)
def test_v4_present_target_replay_fails_closed(
    tmp_path: Path,
    case: str,
    reason: str,
) -> None:
    expected = _write_project(tmp_path)
    command = str(expected["command"])
    task_authority = expected["task_authority"]
    assert isinstance(task_authority, dict)
    baseline_receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
        probe_runner=_passing_probe([]),
    )
    payload = b"import pytest\n\ndef test_replayed():\n    assert True\n"
    target = tmp_path / str(expected["target"])
    target.write_bytes(payload)
    authority = _prior_seed_authority(
        baseline_receipt,
        task_authority,
        str(expected["target"]),
        payload,
    )
    if case == "missing-authority":
        authority = None
    else:
        target.write_bytes(b"# drifted after accepted seed\n")

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [command],
        task_authority=task_authority,
        prior_seed_authority=authority,
        probe_runner=lambda *_args, **_kwargs: pytest.fail(
            "unauthenticated v4 replay must fail before probing"
        ),
    )

    assert receipt["passed"] is False
    assert receipt["projects"][0]["contract_error_reason"] == reason


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("authority-file", "v4_dynamic_dependency_authority_mismatch"),
        (
            "authority-digest",
            "v4_dynamic_dependency_authority_digest_mismatch",
        ),
        (
            "extra-digest",
            "v4_optional_dependency_authority_digest_mismatch",
        ),
        ("unauthorized", "v2_target_requirements_not_authorized"),
    ],
)
def test_v4_rejects_authority_or_requirement_drift_before_probe(
    tmp_path: Path,
    case: str,
    reason: str,
) -> None:
    kwargs: dict[str, object] = {}
    if case == "authority-file":
        kwargs["authority_file"] = "other-requirements.txt"
    elif case == "authority-digest":
        kwargs["authority_digest"] = "0" * 64
    elif case == "extra-digest":
        kwargs["extra_digest"] = "0" * 64
    elif case == "unauthorized":
        kwargs["selected_requirements"] = ["pytest>=9.0.0"]
    expected = _write_project(tmp_path, **kwargs)
    payloads: list[dict[str, object]] = []

    receipt = preflight_validation_project_dependencies(
        tmp_path,
        [str(expected["command"])],
        task_authority=expected["task_authority"],
        probe_runner=_passing_probe(payloads),
    )

    assert receipt["passed"] is False
    assert receipt["reason"] == "project_dependency_contract_collection_failed"
    assert receipt["projects"][0]["contract_error_reason"] == reason
    assert payloads == []


def test_singular_test_root_is_v4_only() -> None:
    with pytest.raises(ValueError, match="v2_target_path_invalid"):
        _require_safe_scoped_pytest_target(TARGET)
    assert (
        _require_safe_scoped_pytest_target(
            TARGET,
            allow_singular_test_root=True,
        )
        == TARGET
    )


def test_checked_in_lgcvf_080_contract_selects_only_pytest() -> None:
    root = Path(__file__).resolve().parents[2]
    payloads: list[dict[str, object]] = []
    receipt = preflight_validation_project_dependencies(
        root,
        [COMMAND],
        task_authority={
            "board_namespace": BOARD,
            "canonical_task_cid": TASK_CID,
            "declared_outputs": [
                "ipfs_accelerate_py/agent_supervisor/planning/"
                "program_repair_synthesis.py",
                TARGET,
            ],
        },
        probe_runner=_passing_probe(payloads),
    )

    assert receipt["passed"] is True
    assert payloads[0]["projects"][0]["requirements"] == [
        PYTEST_REQUIREMENT
    ]
    assert receipt["projects"][0]["dependency_source"] == (
        "agent_supervisor_scoped_file_extra"
    )


@pytest.mark.parametrize(
    ("command", "task_cid", "declared_outputs", "requirements"),
    [
        (
            "python -m pytest -q "
            "test/api/test_agent_supervisor_program_repair_cegis.py",
            "baguqeeraxnljzua5u6p7tlwhor6pia72itrxgpw7kfc45qntwafgfhodrv4a",
            [
                "ipfs_accelerate_py/agent_supervisor/planning/"
                "repair_operator_registry.py",
                "ipfs_accelerate_py/agent_supervisor/planning/"
                "program_repair_synthesis.py",
                "test/api/test_agent_supervisor_program_repair_cegis.py",
            ],
            [PYTEST_REQUIREMENT],
        ),
        (
            "python scripts/qualify_logic_governed_compositional_"
            "verification_fabric.py --check",
            "baguqeerakwvsckoysv5edcru3makxvmcwjjm2alzam5umsyqbkp3efngyqpa",
            [
                "data/agent_supervisor/logic_governed_compositional_"
                "verification_fabric/independent_qualification_result.json"
            ],
            [
                PYTEST_REQUIREMENT,
                "z3-solver>=4.12.0,<5.0.0",
                "cvc5==1.3.3",
            ],
        ),
        (
            "python -m json.tool data/agent_supervisor/logic_governed_"
            "compositional_verification_fabric/"
            "external_qualification_receipt.json",
            "baguqeera6ivyotodkjj7v52oebdcxr4bwlw7yej2bkvk3z7ixiyutfvr2vta",
            [
                "data/agent_supervisor/logic_governed_compositional_"
                "verification_fabric/external_qualification_receipt.json"
            ],
            [],
        ),
        (
            "python scripts/validate_logic_governed_compositional_"
            "verification_fabric_closeout.py implementation --check",
            "baguqeera3y2hsc25gqc7wsa6nojhdz74fn4hkmgwexq5rgpajir3ahq6w4ma",
            [
                "docs/architecture/LOGIC_GOVERNED_COMPOSITIONAL_"
                "VERIFICATION_FABRIC_IMPLEMENTATION_REPORT.md",
                "data/agent_supervisor/logic_governed_compositional_"
                "verification_fabric/successor_tasks.json",
            ],
            [],
        ),
    ],
)
def test_checked_in_v4_contract_covers_later_command_grammars(
    command: str,
    task_cid: str,
    declared_outputs: list[str],
    requirements: list[str],
) -> None:
    root = Path(__file__).resolve().parents[2]
    payloads: list[dict[str, object]] = []

    receipt = preflight_validation_project_dependencies(
        root,
        [command],
        task_authority={
            "board_namespace": BOARD,
            "canonical_task_cid": task_cid,
            "declared_outputs": declared_outputs,
        },
        probe_runner=_passing_probe(payloads),
    )

    assert receipt["passed"] is True
    assert payloads[0]["projects"][0]["requirements"] == requirements
