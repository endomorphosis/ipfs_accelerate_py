from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.control.delegated_operator_completion import (
    DelegatedOperatorCompletionPolicy,
    complete_sealed_manual_task,
)
from ipfs_accelerate_py.agent_supervisor.control.manual_completion_seal import (
    DELEGATED_SUPERVISOR_OPERATOR,
    INTERACTIVE_OPERATOR,
    build_manual_completion_seal,
    verify_manual_completion_seal,
    write_manual_completion_seal,
)


def _git(root: Path, *args: str) -> str:
    import subprocess

    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _init_repo(root: Path) -> tuple[str, str]:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=root,
        check=True,
    )
    (root / "docs").mkdir()
    (root / "config").mkdir()
    policy = root / "docs" / "policy.json"
    policy.write_text('{"ok": true}\n', encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=root, check=True)
    return _git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", "HEAD^{tree}")


def test_delegated_operator_is_distinct_from_interactive() -> None:
    assert DELEGATED_SUPERVISOR_OPERATOR != INTERACTIVE_OPERATOR
    assert DELEGATED_SUPERVISOR_OPERATOR["automatic_controller"] is True
    assert INTERACTIVE_OPERATOR["automatic_controller"] is False


def test_build_and_verify_delegated_seal(tmp_path: Path) -> None:
    commit, tree = _init_repo(tmp_path)
    artifact_paths = {"policy": "docs/policy.json"}
    receipt = build_manual_completion_seal(
        repo_root=tmp_path,
        task_id="TEST-001",
        board_namespace="test-board-v1",
        schema="example.test.seal@1",
        interface="TestSeal@1",
        policy_revision="1",
        artifact_paths=artifact_paths,
        grant_type="policy_activation",
        grant_action="activate_policy_revision",
        operator=DELEGATED_SUPERVISOR_OPERATOR,
        commit=commit,
        tree=tree,
    )
    write_manual_completion_seal(
        "config/operator-seal.json",
        receipt,
        repo_root=tmp_path,
    )
    verified = verify_manual_completion_seal(
        "config/operator-seal.json",
        repo_root=tmp_path,
        task_id="TEST-001",
        board_namespace="test-board-v1",
        schema="example.test.seal@1",
        interface="TestSeal@1",
        policy_revision="1",
        expected_receipt_id=str(receipt["receipt_id"]),
        artifact_paths=artifact_paths,
        grant_type="policy_activation",
        grant_action="activate_policy_revision",
        allow_delegated_operator=True,
    )
    assert verified["receipt_id"] == receipt["receipt_id"]
    assert verified["operator"] == DELEGATED_SUPERVISOR_OPERATOR


def test_complete_sealed_manual_task_marks_board(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    todo = tmp_path / "docs" / "tasks.md"
    todo.write_text(
        """# Tasks

## TEST-001 Seal gated task

- Status: pending
- Completion: manual
- Outputs: docs/policy.json, config/operator-seal.json
- Validation: true

""",
        encoding="utf-8",
    )
    scheduler = tmp_path / "config" / "profile.json"
    seal_config = {
        "receipt_path": "config/operator-seal.json",
        "schema": "example.test.seal@1",
        "interface": "TestSeal@1",
        "policy_revision": "1",
        "expected_receipt_id": "sha256:" + ("0" * 64),
        "artifact_paths": {"policy": "docs/policy.json"},
        "grant_type": "policy_activation",
        "grant_action": "activate_policy_revision",
        "grant_claims": {},
        "reviewed_base_claims": {},
    }
    scheduler.write_text(
        json.dumps(
            {
                "manual_completion_seals": {"TEST-001": seal_config},
                "delegated_operator_completion": {
                    "enabled": True,
                    "allowed_task_ids": ["TEST-001"],
                    "require_validation": True,
                    "validation_timeout_seconds": 30,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    policy = DelegatedOperatorCompletionPolicy.from_mapping(
        {
            "enabled": True,
            "allowed_task_ids": ["TEST-001"],
            "require_validation": True,
            "validation_timeout_seconds": 30,
        }
    )
    result = complete_sealed_manual_task(
        repo_root=tmp_path,
        todo_path=todo,
        scheduler_path=scheduler,
        task_id="TEST-001",
        board_namespace="test-board-v1",
        seal_config=seal_config,
        validation_command="true",
        policy=policy,
    )
    assert result["completed"] is True
    assert "Status: completed" in todo.read_text(encoding="utf-8")
    assert (tmp_path / "config" / "operator-seal.json").is_file()
    pin = json.loads(scheduler.read_text(encoding="utf-8"))[
        "manual_completion_seals"
    ]["TEST-001"]["expected_receipt_id"]
    assert pin == result["receipt_id"]


def test_complete_sealed_manual_task_writes_seal_before_validation(
    tmp_path: Path,
) -> None:
    """Terminal validators that reload the seal as a task output need it present."""

    _init_repo(tmp_path)
    todo = tmp_path / "docs" / "tasks.md"
    todo.write_text(
        """# Tasks

## TEST-001 Seal gated task

- Status: pending
- Completion: manual
- Outputs: docs/policy.json, config/operator-seal.json
- Validation: test -f config/operator-seal.json

""",
        encoding="utf-8",
    )
    scheduler = tmp_path / "config" / "profile.json"
    seal_config = {
        "receipt_path": "config/operator-seal.json",
        "schema": "example.test.seal@1",
        "interface": "TestSeal@1",
        "policy_revision": "1",
        "expected_receipt_id": "sha256:" + ("0" * 64),
        "artifact_paths": {"policy": "docs/policy.json"},
        "grant_type": "policy_activation",
        "grant_action": "activate_policy_revision",
        "grant_claims": {},
        "reviewed_base_claims": {},
    }
    scheduler.write_text(
        json.dumps({"manual_completion_seals": {"TEST-001": seal_config}}),
        encoding="utf-8",
    )
    policy = DelegatedOperatorCompletionPolicy.from_mapping(
        {
            "enabled": True,
            "allowed_task_ids": ["TEST-001"],
            "require_validation": True,
            "validation_timeout_seconds": 30,
        }
    )
    result = complete_sealed_manual_task(
        repo_root=tmp_path,
        todo_path=todo,
        scheduler_path=scheduler,
        task_id="TEST-001",
        board_namespace="test-board-v1",
        seal_config=seal_config,
        validation_command="test -f config/operator-seal.json",
        policy=policy,
    )
    assert result["completed"] is True
    assert result["validation"]["ran"] is True
    assert result["validation"]["returncode"] == 0
    assert (tmp_path / "config" / "operator-seal.json").is_file()


def test_complete_sealed_manual_task_does_not_pin_on_validation_failure(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.control.delegated_operator_completion import (
        DelegatedOperatorCompletionError,
    )

    _init_repo(tmp_path)
    todo = tmp_path / "docs" / "tasks.md"
    todo.write_text(
        """# Tasks

## TEST-001 Seal gated task

- Status: pending
- Validation: false

""",
        encoding="utf-8",
    )
    scheduler = tmp_path / "config" / "profile.json"
    stale_pin = "sha256:" + ("a" * 64)
    seal_config = {
        "receipt_path": "config/operator-seal.json",
        "schema": "example.test.seal@1",
        "interface": "TestSeal@1",
        "policy_revision": "1",
        "expected_receipt_id": stale_pin,
        "artifact_paths": {"policy": "docs/policy.json"},
        "grant_type": "policy_activation",
        "grant_action": "activate_policy_revision",
        "grant_claims": {},
        "reviewed_base_claims": {},
    }
    scheduler.write_text(
        json.dumps(
            {
                "manual_completion_seals": {"TEST-001": dict(seal_config)},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    policy = DelegatedOperatorCompletionPolicy.from_mapping(
        {
            "enabled": True,
            "allowed_task_ids": ["TEST-001"],
            "require_validation": True,
            "validation_timeout_seconds": 30,
        }
    )
    try:
        complete_sealed_manual_task(
            repo_root=tmp_path,
            todo_path=todo,
            scheduler_path=scheduler,
            task_id="TEST-001",
            board_namespace="test-board-v1",
            seal_config=seal_config,
            validation_command="false",
            policy=policy,
        )
        raise AssertionError("expected validation failure")
    except DelegatedOperatorCompletionError as exc:
        assert "validation command failed" in str(exc)
    # Provisional seal may exist for diagnosis, but pin and board stay put.
    pin = json.loads(scheduler.read_text(encoding="utf-8"))[
        "manual_completion_seals"
    ]["TEST-001"]["expected_receipt_id"]
    assert pin == stale_pin
    assert "Status: pending" in todo.read_text(encoding="utf-8")
