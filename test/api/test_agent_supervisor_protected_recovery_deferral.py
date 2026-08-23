"""Focused regression tests for protected-recovery launch deferrals."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_lock_metadata,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    EXTERNAL_PROTECTED_RECOVERY_BACKOFF_SECONDS,
    PORTAL_RETRY_DEFERRAL_SCHEMA,
    TodoImplementationDaemon,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _daemon(tmp_path: Path) -> tuple[TodoImplementationDaemon, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    todo_path = repo / "todo.md"
    todo_path.write_text("# Generated board\n", encoding="utf-8")
    _git(repo, "add", "todo.md")
    _git(
        repo,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "seed board",
    )
    state_dir = tmp_path / "state"
    return (
        TodoImplementationDaemon(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            repo_root=repo,
            task_header_prefix="## AUTO-",
            implementation_protected_paths=("todo.md",),
        ),
        repo,
    )


def _foreign_recovery() -> dict[str, Any]:
    return {
        "required": True,
        "recovered": False,
        "adopted": False,
        "blocked": True,
        "reason": "external_protected_checkout_recovery_required",
        "protected_recovery_owner": "implementation_supervisor",
        "foreign_owner_liveness": "verified_live",
        "deferred": True,
        "attempt_consumed": False,
        "provider_dispatched": False,
        "backoff_seconds": EXTERNAL_PROTECTED_RECOVERY_BACKOFF_SECONDS,
        "lock_owner_pid": os.getpid(),
        "lock_path": "/repository/.git/agent-checkout-mutation.lock",
    }


def _assert_verified_live_deferral(implementation: dict[str, Any]) -> None:
    assert implementation["returncode"] == 1
    assert implementation["reason"] == (
        "external_protected_recovery_owner_active"
    )
    assert implementation["deferred"] is True
    assert implementation["attempt_consumed"] is False
    assert implementation["provider_dispatched"] is False
    assert implementation["backoff_seconds"] == (
        EXTERNAL_PROTECTED_RECOVERY_BACKOFF_SECONDS
    )
    assert implementation["deferral_schema"] == PORTAL_RETRY_DEFERRAL_SCHEMA
    assert implementation["retryable"] is True
    assert implementation["failure_kind"] == "lifecycle_setup"
    assert implementation["provider_call_allowed"] is False


def test_daemon_marks_only_live_compatible_foreign_owner_transient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, repo = _daemon(tmp_path)
    owner_script = Path(sys.argv[0]).name
    monkeypatch.setattr(
        implementation_daemon_module,
        "process_is_running",
        lambda pid: pid == os.getpid(),
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "process_command_line",
        lambda pid: f"python {owner_script}" if pid == os.getpid() else "",
    )
    metadata = checkout_lock_metadata(
        kind="merge",
        repo_root=repo,
        task_id="AUTO-001",
        owner_script=owner_script,
    )
    metadata.update(
        {
            "protected_recovery_required": True,
            "protected_recovery_owner": "implementation_supervisor",
        }
    )
    lock_path = daemon._repo_merge_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    original = lock_path.read_bytes()

    result = daemon._adopt_protected_checkout_recovery()

    assert result["required"] is True
    assert result["blocked"] is True
    assert result["reason"] == (
        "external_protected_checkout_recovery_required"
    )
    assert result["protected_recovery_owner"] == "implementation_supervisor"
    assert result["foreign_owner_liveness"] == "verified_live"
    assert result["deferred"] is True
    assert result["attempt_consumed"] is False
    assert result["provider_dispatched"] is False
    assert result["backoff_seconds"] == (
        EXTERNAL_PROTECTED_RECOVERY_BACKOFF_SECONDS
    )
    assert lock_path.read_bytes() == original
    assert daemon._current_checkout_mutation_lease() is None

    pass_result = daemon.run_once()
    assert pass_result["blocked"] is True
    _assert_verified_live_deferral(pass_result["implementation_result"])
    assert lock_path.read_bytes() == original


@pytest.mark.parametrize(
    "foreign_case",
    (
        "inactive",
        "foreign_repository",
        "unprovable_repository",
        "unknown_owner",
        "malformed_owner",
    ),
)
def test_daemon_keeps_unverified_foreign_owner_terminal(
    tmp_path: Path,
    foreign_case: str,
) -> None:
    daemon, repo = _daemon(tmp_path)
    metadata = checkout_lock_metadata(
        kind="merge",
        repo_root=repo,
        task_id="AUTO-001",
        owner_script=Path(sys.argv[0]).name,
    )
    metadata.update(
        {
            "protected_recovery_required": True,
            "protected_recovery_owner": "implementation_supervisor",
        }
    )
    if foreign_case == "inactive":
        metadata["pid"] = 2_147_483_647
    elif foreign_case == "foreign_repository":
        metadata["repository_id"] = "repository:foreign"
    elif foreign_case == "unprovable_repository":
        metadata["repository_id"] = ""
        metadata["worktree_root"] = ""
        metadata["repo_root"] = ""
    elif foreign_case == "unknown_owner":
        metadata["protected_recovery_owner"] = "unknown_component"
    else:
        metadata["owner_script"] = ""
    lock_path = daemon._repo_merge_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")

    result = daemon._adopt_protected_checkout_recovery()

    assert result["required"] is True
    assert result["blocked"] is True
    assert result["reason"] == (
        "external_protected_checkout_recovery_required"
    )
    assert result["foreign_owner_liveness"] == "not_verified"
    assert "deferred" not in result
    assert "attempt_consumed" not in result
    assert "provider_dispatched" not in result
    assert lock_path.exists()


def test_run_once_projects_verified_live_owner_as_typed_deferral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, _repo = _daemon(tmp_path)
    monkeypatch.setattr(
        daemon,
        "_recover_protected_checkout_mutation",
        _foreign_recovery,
    )

    result = daemon.run_once()

    assert result["blocked"] is True
    assert result["reason"] == (
        "external_protected_checkout_recovery_required"
    )
    _assert_verified_live_deferral(result["implementation_result"])


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("foreign_owner_liveness", "not_verified"),
        ("protected_recovery_owner", "unknown_component"),
        ("backoff_seconds", 0),
    ),
)
def test_run_once_does_not_project_unverified_foreign_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    daemon, _repo = _daemon(tmp_path)
    recovery = _foreign_recovery()
    recovery[field] = value
    monkeypatch.setattr(
        daemon,
        "_recover_protected_checkout_mutation",
        lambda: recovery,
    )

    result = daemon.run_once()

    assert result["blocked"] is True
    implementation = result["implementation_result"]
    assert implementation is not None
    assert implementation["reason"] != (
        "external_protected_recovery_owner_active"
    )
    assert implementation["deferral_schema"] == PORTAL_RETRY_DEFERRAL_SCHEMA
    assert implementation["deferred"] is True
