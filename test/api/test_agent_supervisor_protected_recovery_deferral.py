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
    checkout_mutation_lock_path,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    ProcessBirthIdentity,
    current_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    EXTERNAL_PROTECTED_RECOVERY_BACKOFF_SECONDS,
    INFLIGHT_IMPLEMENTATION_DEFERRAL_BACKOFF_SECONDS,
    PortalTask,
    PortalTaskState,
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


def _seed_inflight_state(
    daemon: TodoImplementationDaemon,
    *,
    worktree_path: Path,
    log_path: Path,
) -> tuple[PortalTaskState, dict[str, Any]]:
    task_cid = "sha256:inflight-task"
    branch = "implementation/auto-001-attempt-1"
    state = PortalTaskState(
        active_task_id="AUTO-001",
        active_task_cid=task_cid,
        active_attempt=1,
        active_phase="validating",
        active_log_path=str(log_path),
        active_worktree_path=str(worktree_path),
        active_branch=branch,
        implementation_in_progress=True,
        last_implementation_task_id="AUTO-001",
        last_implementation_task_cid=task_cid,
        last_implementation_worktree_path=str(worktree_path),
        last_implementation_branch=branch,
    )
    state.save(daemon.state_path)
    event = {
        "type": "implementation_started",
        "task_id": "AUTO-001",
        "canonical_task_cid": task_cid,
        "attempt": 1,
        "command": ["codex", "exec"],
        "log_path": str(log_path),
        "worktree_path": str(worktree_path),
        "branch": branch,
    }
    return state, event


def _seed_lifecycle(
    daemon: TodoImplementationDaemon,
    *,
    worktree_path: Path,
    owner: ProcessBirthIdentity,
) -> Any:
    return daemon.worktree_lifecycle.begin_preparing(
        task_id="AUTO-001",
        canonical_task_cid="sha256:inflight-task",
        attempt=1,
        lane_id="test-lane",
        workspace_path=worktree_path,
        branch="implementation/auto-001-attempt-1",
        merge_target="main",
        state_dir=str(daemon.state_path.parent.resolve()),
        owner=owner,
    )


def _inflight_task() -> PortalTask:
    return PortalTask(
        task_id="AUTO-001",
        title="Recover candidate",
        status="todo",
        completion="automatic",
        priority="P1",
        track="runtime",
        outputs=["feature.py"],
        validation=["python -m py_compile feature.py"],
        acceptance="Candidate validates.",
        canonical_task_cid="sha256:inflight-task",
    )


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
    lock_path = checkout_mutation_lock_path(repo)
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
    assert pass_result["implementation_result"] == {
        "returncode": 1,
        "reason": "external_protected_recovery_owner_active",
        "deferred": True,
        "attempt_consumed": False,
        "provider_dispatched": False,
        "backoff_seconds": EXTERNAL_PROTECTED_RECOVERY_BACKOFF_SECONDS,
    }
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
    lock_path = checkout_mutation_lock_path(repo)
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
    assert result["implementation_result"] == {
        "returncode": 1,
        "reason": "external_protected_recovery_owner_active",
        "deferred": True,
        "attempt_consumed": False,
        "provider_dispatched": False,
        "backoff_seconds": EXTERNAL_PROTECTED_RECOVERY_BACKOFF_SECONDS,
    }


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
    assert result["implementation_result"] is None


@pytest.mark.parametrize(
    ("owner_state", "expected_disposition", "expected_reason"),
    (
        (
            "live",
            "verified_live",
            "inflight_implementation_owner_active",
        ),
        (
            "controlled_restart",
            "controlled_restart_recovery",
            "inflight_implementation_controlled_restart_recovery",
        ),
    ),
)
def test_exact_inflight_owner_is_a_typed_pre_provider_deferral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_state: str,
    expected_disposition: str,
    expected_reason: str,
) -> None:
    daemon, repo = _daemon(tmp_path)
    worktree_path = repo / "worktrees" / "auto-001-attempt-1"
    worktree_path.mkdir(parents=True)
    log_path = tmp_path / "state" / "auto-001-attempt-1.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("validation started\n", encoding="utf-8")
    state, event = _seed_inflight_state(
        daemon,
        worktree_path=worktree_path,
        log_path=log_path,
    )
    lifecycle = _seed_lifecycle(
        daemon,
        worktree_path=worktree_path,
        owner=(
            current_process_birth(proc_root=daemon.worktree_lifecycle.proc_root)
            if owner_state == "live"
            else ProcessBirthIdentity(
                pid=2_147_483_647,
                start_time_ticks=1,
                boot_id="dead-owner",
            )
        ),
    )
    if owner_state == "controlled_restart":
        terminal = (
            daemon.worktree_lifecycle.reclaim_dead_owner_for_controlled_restart(
                worktree_path,
                expected_state_dir=daemon.state_path.parent.resolve(),
            )
        )
        assert terminal is not None
        assert terminal.record_id == lifecycle.record_id
        # Log age is diagnostic only.  Exact lifecycle/state/workspace binding,
        # not an mtime grace period, authorizes the recovery deferral.
        os.utime(log_path, (1, 1))

    monkeypatch.setattr(daemon, "_list_process_commands", lambda: [])
    monkeypatch.setattr(
        daemon,
        "_docker_isolation_active_for_worktree",
        lambda _path: False,
    )
    monkeypatch.setattr(
        daemon,
        "_inflight_implementation_events",
        lambda: [event],
    )

    inflight = daemon._find_live_inflight_implementation()
    assert inflight is not None
    assert inflight["_inflight_disposition"] == expected_disposition
    # The active projection must remain intact so a reconciliation pass can
    # adopt the exact unmerged candidate instead of dispatching a duplicate.
    assert PortalTaskState.load(daemon.state_path) == state

    monkeypatch.setattr(
        implementation_daemon_module,
        "completion_gap_edit_scope",
        lambda *_args, **_kwargs: ("feature.py",),
    )
    monkeypatch.setattr(
        daemon,
        "_active_provider_capacity_backoff_for_task",
        lambda _task: {},
    )
    monkeypatch.setattr(
        daemon,
        "_require_primary_provider_readiness",
        lambda _task: pytest.fail("inflight deferral reached provider setup"),
    )
    monkeypatch.setattr(
        daemon,
        "_run_implementation_in_ephemeral_worktree",
        lambda **_kwargs: pytest.fail("inflight deferral dispatched provider"),
    )

    result = daemon._run_implementation(_inflight_task(), state)

    assert result["reason"] == expected_reason
    assert result["deferred"] is True
    assert result["attempt_consumed"] is False
    assert result["provider_dispatched"] is False
    assert result["backoff_seconds"] == (
        INFLIGHT_IMPLEMENTATION_DEFERRAL_BACKOFF_SECONDS
    )


@pytest.mark.parametrize(
    "invalid_lifecycle",
    ("missing", "malformed", "foreign_schema", "foreign", "unknown"),
)
def test_unverifiable_inflight_lifecycle_remains_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_lifecycle: str,
) -> None:
    daemon, repo = _daemon(tmp_path)
    worktree_path = repo / "worktrees" / "auto-001-attempt-1"
    worktree_path.mkdir(parents=True)
    log_path = tmp_path / "state" / "auto-001-attempt-1.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("recent but not authority\n", encoding="utf-8")
    state, event = _seed_inflight_state(
        daemon,
        worktree_path=worktree_path,
        log_path=log_path,
    )
    lifecycle_path = daemon.worktree_lifecycle.workspace_path_for(
        worktree_path
    )
    if invalid_lifecycle == "missing":
        pass
    elif invalid_lifecycle == "malformed":
        lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
        lifecycle_path.write_text("{malformed", encoding="utf-8")
    else:
        lifecycle = _seed_lifecycle(
            daemon,
            worktree_path=worktree_path,
            owner=(
                current_process_birth(
                    proc_root=daemon.worktree_lifecycle.proc_root
                )
                if invalid_lifecycle in {"foreign", "foreign_schema"}
                else ProcessBirthIdentity(
                    pid=2_147_483_647,
                    start_time_ticks=1,
                    boot_id="unverifiable-owner",
                )
            ),
        )
        if invalid_lifecycle == "foreign":
            payload = lifecycle.to_dict()
            payload["repo_root"] = str(tmp_path / "foreign-repository")
            lifecycle_path.write_text(
                json.dumps(payload, sort_keys=True),
                encoding="utf-8",
            )
        elif invalid_lifecycle == "foreign_schema":
            payload = lifecycle.to_dict()
            payload["schema"] = "foreign/worktree-lifecycle@1"
            lifecycle_path.write_text(
                json.dumps(payload, sort_keys=True),
                encoding="utf-8",
            )
        else:
            daemon.worktree_lifecycle.proc_root = tmp_path / "missing-proc"

    monkeypatch.setattr(daemon, "_list_process_commands", lambda: [])
    monkeypatch.setattr(
        daemon,
        "_docker_isolation_active_for_worktree",
        lambda _path: False,
    )
    monkeypatch.setattr(
        daemon,
        "_inflight_implementation_events",
        lambda: [event],
    )
    inflight = daemon._find_live_inflight_implementation()
    assert inflight is not None
    assert inflight["_inflight_disposition"] == "unverifiable"

    monkeypatch.setattr(
        implementation_daemon_module,
        "completion_gap_edit_scope",
        lambda *_args, **_kwargs: ("feature.py",),
    )
    monkeypatch.setattr(
        daemon,
        "_active_provider_capacity_backoff_for_task",
        lambda _task: {},
    )
    result = daemon._run_implementation(_inflight_task(), state)

    assert result["reason"] == "inflight_process"
    assert "deferred" not in result
    assert "attempt_consumed" not in result
    assert "provider_dispatched" not in result
