"""Focused liveness and maintenance tests for database implementation lanes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    checkout_lock_metadata,
    checkout_mutation_lock_path,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    database_task_source as database_task_source_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as implementation_supervisor_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    DATABASE_DAEMON_PASS_HEARTBEAT_SCHEMA,
    PortalTaskState,
    current_process_birth,
    publish_database_daemon_pass_heartbeat,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    DATABASE_IDLE_DAEMON_STALL_REASON,
    SUPERVISOR_MAINTENANCE_RECEIPT_SCHEMA,
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
)


def _supervisor(
    tmp_path,
    *,
    lane_index: int = 1,
    task_prefix: str = "SAWM-",
) -> PortalImplementationSupervisor:
    repo = tmp_path / f"repo-{lane_index}"
    repo.mkdir()
    state_dir = repo / "state" / f"lane-{lane_index}"
    state_dir.mkdir(parents=True)
    state_path = state_dir / "task_state.json"
    PortalTaskState().save(state_path)
    program = SimpleNamespace(
        authority_mode="quack",
        task_source_kind="duckdb",
        quack_endpoint="quack:127.0.0.1:24068",
        store_id="control.duckdb",
        assert_quack_not_demoted=lambda **_kwargs: None,
    )
    return PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_path,
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            database_program=program,
            task_prefix=task_prefix,
            task_shard_count=4,
            task_shard_index=lane_index,
            strict_task_sharding=True,
            check_interval=1,
            daemon_interval=1,
        )
    )


def _ready_observation(*, same_shard: bool = True) -> dict[str, object]:
    return {
        "available": True,
        "reason": "authoritative_readiness_observed",
        "task_source_revision": 41,
        "ready_task_ids": ["SAWM-006"],
        "same_shard_ready_task_ids": ["SAWM-006"] if same_shard else [],
        "active_task_ids": [],
        "same_shard_active_task_ids": [],
    }


def _make_idle_recycle_safe(supervisor, monkeypatch) -> None:
    monkeypatch.setattr(supervisor, "_active_agent_worker_processes", lambda: [])
    monkeypatch.setattr(
        supervisor,
        "_active_validation_subprocess_exists",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_supervisor_module,
        "descendant_processes",
        lambda _pid: [],
    )


def _configure_ready_stale_idle_watchdog(supervisor, monkeypatch) -> None:
    monkeypatch.setattr(
        supervisor,
        "_authoritative_runnable_work_status",
        _ready_observation,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_pass_heartbeat_status",
        lambda _child, now_ts: {"stale": True, "reason": "heartbeat_stale"},
    )
    _make_idle_recycle_safe(supervisor, monkeypatch)


def _write_expired_legacy_cleanup_lease(
    supervisor: PortalImplementationSupervisor,
    *,
    operation: str = "cleanup_backlogged_worktrees",
    task_id: str = "",
    branch: str = "",
    owner_phase: str = "strategy_state_repair",
    status_retains_phase: bool = False,
    write_receipt: bool = True,
    receipt_overrides: dict[str, object] | None = None,
    extra: dict[str, object] | None = None,
):
    owner_state_dir = supervisor.config.repo_root / "legacy-owner-state"
    owner_state_dir.mkdir()
    owner_state_path = owner_state_dir / "legacy_lane_task_state.json"
    owner_state_path.write_text(
        json.dumps(
            {
                "active_task_id": "",
                "implementation_in_progress": False,
                "active_phase": "",
                "active_branch": "",
            }
        ),
        encoding="utf-8",
    )
    owner_status_path = owner_state_dir / "legacy_lane_supervisor_status.json"
    owner_status = {
        "schema": ("ipfs_accelerate_py.agent_supervisor.todo_implementation_supervisor.supervisor"),
        # Model the ordinary SupervisorLoop heartbeat which replaces the
        # richer maintenance status projection.
        "status": "running",
        "updated_at": datetime.now(UTC).isoformat(),
        "supervisor_pid": os.getpid(),
        "supervisor_pid_alive": True,
        "active_worker_count": 0,
        "active_worker_pids": [],
        "worker_descendant_count": 0,
    }
    if status_retains_phase:
        owner_status.update(
            {
                "status": "agentic_maintenance_started",
                "last_agentic_maintenance_phase": owner_phase,
            }
        )
    owner_status_path.write_text(json.dumps(owner_status), encoding="utf-8")
    receipt_now = datetime.now(UTC)
    if write_receipt:
        receipt = {
            "schema": SUPERVISOR_MAINTENANCE_RECEIPT_SCHEMA,
            "repo_root": str(supervisor.config.repo_root.resolve()),
            "state_dir": str(owner_state_dir.resolve()),
            "state_path": str(owner_state_path.resolve()),
            "state_prefix": "legacy_lane",
            "supervisor_pid": os.getpid(),
            "process_birth": current_process_birth().to_dict(),
            "phase": owner_phase,
            "status": "completed",
            "maintenance_started_at": (receipt_now - timedelta(minutes=5)).isoformat(),
            "updated_at": receipt_now.isoformat(),
            "completed_at": receipt_now.isoformat(),
        }
        receipt.update(dict(receipt_overrides or {}))
        (owner_state_dir / "legacy_lane_supervisor_maintenance_receipt.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
    metadata = checkout_lock_metadata(
        kind="merge",
        repo_root=supervisor.config.repo_root,
        task_id=task_id,
        branch=branch,
        owner_script="",
        extra={
            "operation": operation,
            "started_at": (datetime.now(UTC) - timedelta(days=2)).isoformat(),
            "state_dir": str(owner_state_dir.resolve()),
            "state_path": str(owner_state_path.resolve()),
            **dict(extra or {}),
        },
    )
    lock_path = checkout_mutation_lock_path(supervisor.config.repo_root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(metadata), encoding="utf-8")
    return lock_path


def _write_external_board_legacy_cleanup_lease(
    supervisor: PortalImplementationSupervisor,
    *,
    write_receipt: bool = True,
    receipt_status: str = "completed",
):
    repo = supervisor.config.repo_root
    supervisor.config.todo_path.write_text("# SAWM board\n", encoding="utf-8")

    def git(*args: str, cwd=repo) -> None:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    git("init", "-q")
    git("config", "user.email", "supervisor-tests@example.invalid")
    git("config", "user.name", "Supervisor Tests")
    (repo / "seed.txt").write_text("seed\n", encoding="utf-8")
    git("add", "seed.txt", supervisor.config.todo_path.name)
    git("commit", "-qm", "seed linked worktree")
    external_worktree = repo.parent / f"{repo.name}-external-board"
    git(
        "worktree",
        "add",
        "-q",
        "-b",
        f"external-board-{repo.name}",
        str(external_worktree),
    )
    external_todo = external_worktree / "docs" / "apmc.todo.md"
    external_todo.parent.mkdir(parents=True)
    external_todo.write_text("# APMC board\n", encoding="utf-8")
    state_dir = external_worktree / "state" / "lane-1"
    state_dir.mkdir(parents=True)
    state_path = state_dir / "apmc_lane_1_task_state.json"
    state_path.write_text(
        json.dumps(
            {
                "active_task_id": "",
                "implementation_in_progress": False,
                "active_phase": "",
                "active_branch": "",
            }
        ),
        encoding="utf-8",
    )
    status_path = state_dir / "apmc_lane_1_supervisor_status.json"
    status_path.write_text(
        json.dumps(
            {
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor.todo_implementation_supervisor.supervisor"
                ),
                "status": "running",
                "updated_at": datetime.now(UTC).isoformat(),
                "supervisor_pid": os.getpid(),
                "supervisor_pid_alive": True,
                "active_worker_count": 0,
                "active_worker_pids": [],
                "worker_descendant_count": 0,
                "repo_root": str(external_worktree.resolve()),
                "state_path": str(state_path.resolve()),
                "current_status_path": state_path.relative_to(external_worktree).as_posix(),
                "state_prefix": "apmc_lane_1",
                "todo_path": str(external_todo.resolve()),
                "task_prefix": "## APMC-",
            }
        ),
        encoding="utf-8",
    )
    if write_receipt:
        receipt_now = datetime.now(UTC)
        receipt_path = state_dir / "apmc_lane_1_supervisor_maintenance_receipt.json"
        receipt_path.write_text(
            json.dumps(
                {
                    "schema": SUPERVISOR_MAINTENANCE_RECEIPT_SCHEMA,
                    "repo_root": str(external_worktree.resolve()),
                    "state_dir": str(state_dir.resolve()),
                    "state_path": str(state_path.resolve()),
                    "state_prefix": "apmc_lane_1",
                    "supervisor_pid": os.getpid(),
                    "process_birth": current_process_birth().to_dict(),
                    "phase": "supervisor_check_event",
                    "status": receipt_status,
                    "maintenance_started_at": (receipt_now - timedelta(minutes=5)).isoformat(),
                    "updated_at": receipt_now.isoformat(),
                    "completed_at": (
                        receipt_now.isoformat() if receipt_status == "completed" else ""
                    ),
                }
            ),
            encoding="utf-8",
        )
    metadata = checkout_lock_metadata(
        kind="merge",
        repo_root=external_worktree,
        owner_script="",
        extra={
            "operation": "cleanup_backlogged_worktrees",
            "started_at": (datetime.now(UTC) - timedelta(days=2)).isoformat(),
            "state_dir": str(state_dir.resolve()),
            "state_path": str(state_path.resolve()),
        },
    )
    lock_path = checkout_mutation_lock_path(repo)
    lock_path.write_text(json.dumps(metadata), encoding="utf-8")
    return lock_path, metadata, status_path


def test_database_main_pass_heartbeat_is_current_process_bound(tmp_path) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    heartbeat = publish_database_daemon_pass_heartbeat(
        state_dir=supervisor.config.state_dir,
        state_prefix=supervisor.config.state_prefix,
        sequence=1,
        result={
            "unchanged": True,
            "write_count": 0,
            "active_task_id": "",
            "selection_idle_reason": "no_ready_tasks",
            "provider_result": {"secret": "must-not-be-projected"},
        },
        process_instance_id="process:test",
        owner_session_id="session:test",
        authority_mode="quack",
        task_source_kind="duckdb",
        task_shard_count=4,
        task_shard_index=1,
        strict_task_sharding=True,
    )
    child = SimpleNamespace(
        pid=os.getpid(),
        started_at=datetime.now(UTC).isoformat(),
        identity_process_birth=current_process_birth(),
    )

    status = supervisor._database_pass_heartbeat_status(child, now_ts=time.time())

    assert heartbeat["schema"] == DATABASE_DAEMON_PASS_HEARTBEAT_SCHEMA
    assert heartbeat["sequence"] == 1
    assert "provider_result" not in heartbeat
    assert status["available"] is True
    assert status["current_process"] is True
    assert status["stale"] is False


def test_supervisor_maintenance_receipt_survives_ordinary_status_overwrite(
    tmp_path,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    update_phase, finish = supervisor._begin_supervisor_maintenance_heartbeat("run_once")
    update_phase("supervisor_check_event")
    finish("completed")
    receipt_path = supervisor._supervisor_maintenance_receipt_path()
    receipt_before = json.loads(receipt_path.read_text(encoding="utf-8"))
    supervisor._supervisor_status_path().write_text(
        json.dumps(
            {
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor.todo_implementation_supervisor.supervisor"
                ),
                "status": "running",
                "updated_at": datetime.now(UTC).isoformat(),
                "supervisor_pid": os.getpid(),
                "active_worker_count": 0,
                "active_worker_pids": [],
                "worker_descendant_count": 0,
            }
        ),
        encoding="utf-8",
    )

    receipt_after = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt_after == receipt_before
    assert receipt_after["schema"] == SUPERVISOR_MAINTENANCE_RECEIPT_SCHEMA
    assert receipt_after["phase"] == "supervisor_check_event"
    assert receipt_after["status"] == "completed"
    assert receipt_after["completed_at"] == receipt_after["updated_at"]
    assert receipt_after["process_birth"] == current_process_birth().to_dict()


def test_database_watchdog_recycles_stale_idle_child_for_same_shard_ready_work(
    tmp_path,
    monkeypatch,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    monkeypatch.setattr(
        supervisor,
        "_authoritative_runnable_work_status",
        _ready_observation,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_pass_heartbeat_status",
        lambda _child, now_ts: {
            "schema": DATABASE_DAEMON_PASS_HEARTBEAT_SCHEMA,
            "available": True,
            "current_process": True,
            "stale": True,
            "reason": "heartbeat_stale",
            "age_seconds": 600.0,
        },
    )
    _make_idle_recycle_safe(supervisor, monkeypatch)
    maintenance_calls: list[bool] = []
    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance",
        lambda _update: maintenance_calls.append(True),
    )
    child = SimpleNamespace(
        pid=os.getpid(),
        started_at=(datetime.now(UTC) - timedelta(minutes=10)).isoformat(),
        identity_process_birth=current_process_birth(),
    )

    decision = supervisor._supervisor_loop_watchdog_decision(None, child, {})

    assert decision.action == "recycle"
    assert decision.reason == DATABASE_IDLE_DAEMON_STALL_REASON
    assert decision.detail["same_shard_ready_task_ids"] == ["SAWM-006"]
    assert decision.detail["attempt_budget_consumed"] is False
    assert decision.detail["provider_invocation_consumed"] is False
    assert maintenance_calls == []


def test_database_watchdog_preserves_child_when_readiness_is_unavailable(
    tmp_path,
    monkeypatch,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    monkeypatch.setattr(
        supervisor,
        "_authoritative_runnable_work_status",
        lambda: {
            "available": False,
            "reason": "authoritative_readiness_unavailable",
            "error_type": "QuackUnavailable",
            "task_source_revision": 0,
            "ready_task_ids": [],
            "same_shard_ready_task_ids": [],
            "active_task_ids": [],
            "same_shard_active_task_ids": [],
        },
    )
    monkeypatch.setattr(
        supervisor,
        "_database_pass_heartbeat_status",
        lambda *_args, **_kwargs: pytest.fail("heartbeat must not grant readiness"),
    )
    maintenance_calls: list[bool] = []
    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance",
        lambda _update: maintenance_calls.append(True),
    )

    decision = supervisor._supervisor_loop_watchdog_decision(
        None,
        SimpleNamespace(pid=os.getpid()),
        {},
    )

    assert decision.action == "continue"
    assert maintenance_calls == []


def test_database_watchdog_never_recycles_through_protected_checkout_lock(
    tmp_path,
    monkeypatch,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    monkeypatch.setattr(
        supervisor,
        "_authoritative_runnable_work_status",
        _ready_observation,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_pass_heartbeat_status",
        lambda _child, now_ts: {"stale": True, "reason": "heartbeat_stale"},
    )
    _make_idle_recycle_safe(supervisor, monkeypatch)
    lock_path = supervisor.config.repo_root / ".git" / "protected.lock"
    lock_path.parent.mkdir(exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "lease_id": "lease:protected",
                "protected_recovery_required": True,
                "pid": os.getpid(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor, "_repo_merge_lock_path", lambda: lock_path)

    decision = supervisor._supervisor_loop_watchdog_decision(
        None,
        SimpleNamespace(pid=os.getpid()),
        {},
    )

    assert decision.action == "continue"
    assert lock_path.exists()


def test_database_watchdog_preserves_external_board_legacy_cleanup_without_blocking(
    tmp_path,
    monkeypatch,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    _configure_ready_stale_idle_watchdog(supervisor, monkeypatch)
    lock_path, metadata, _status_path = _write_external_board_legacy_cleanup_lease(supervisor)
    original_bytes = lock_path.read_bytes()

    decision = supervisor._supervisor_loop_watchdog_decision(
        None,
        SimpleNamespace(pid=os.getpid()),
        {},
    )

    assert decision.action == "recycle"
    assert lock_path.read_bytes() == original_bytes
    assert json.loads(original_bytes)["lease_id"] == metadata["lease_id"]
    external = decision.detail["external_legacy_cleanup_lease"]
    assert external["external_board"] is True
    assert external["preserved"] is True
    assert external["reason"] == "external_board_legacy_cleanup_lease_preserved"
    assert "legacy_checkout_mutation_transaction_active" not in decision.detail["blockers"]


@pytest.mark.parametrize("replacement_mode", ("atomic", "same_inode_metadata"))
def test_database_watchdog_revalidates_external_lease_before_omitting_blocker(
    tmp_path,
    monkeypatch,
    replacement_mode,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    _configure_ready_stale_idle_watchdog(supervisor, monkeypatch)
    lock_path, external_metadata, _status_path = _write_external_board_legacy_cleanup_lease(
        supervisor
    )
    classify_external = supervisor._external_board_legacy_cleanup_lease_status

    protected_metadata = checkout_lock_metadata(
        kind="merge",
        repo_root=supervisor.config.repo_root,
        owner_script="",
        extra={
            "operation": "merge_branch_to_main",
            "protected_recovery_required": True,
        },
    )
    if replacement_mode == "same_inode_metadata":
        protected_metadata = {
            **external_metadata,
            "protected_recovery_required": True,
        }

    def classify_then_replace(metadata, *, now_ts):
        result = classify_external(metadata, now_ts=now_ts)
        assert result["external_board"] is True
        if replacement_mode == "atomic":
            replacement_path = lock_path.with_name(f".{lock_path.name}.protected")
            replacement_path.write_text(
                json.dumps(protected_metadata),
                encoding="utf-8",
            )
            os.replace(replacement_path, lock_path)
        else:
            lock_path.write_text(json.dumps(protected_metadata), encoding="utf-8")
        return result

    monkeypatch.setattr(
        supervisor,
        "_external_board_legacy_cleanup_lease_status",
        classify_then_replace,
    )

    guard = supervisor._idle_database_child_recycle_guard(
        state=PortalTaskState(),
        child=SimpleNamespace(pid=os.getpid()),
        readiness=_ready_observation(),
        heartbeat={"stale": True, "reason": "heartbeat_stale"},
    )

    assert guard["safe"] is False
    assert json.loads(lock_path.read_text(encoding="utf-8")) == protected_metadata
    assert "legacy_checkout_mutation_transaction_active" in guard["blockers"]
    external = guard["external_legacy_cleanup_lease"]
    assert external["external_board"] is False
    assert external["classified_external_board"] is True
    assert external["reason"] == "external_board_legacy_cleanup_lease_changed"
    assert external["lease_revalidation"]["reason"] == "checkout_lease_revalidation_replaced"


def test_database_watchdog_same_board_legacy_cleanup_without_receipt_blocks(
    tmp_path,
    monkeypatch,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    _configure_ready_stale_idle_watchdog(supervisor, monkeypatch)
    monkeypatch.setattr(
        supervisor,
        "_checkout_lock_owner_is_active",
        lambda _metadata: True,
    )
    lock_path = _write_expired_legacy_cleanup_lease(
        supervisor,
        write_receipt=False,
    )

    decision = supervisor._supervisor_loop_watchdog_decision(
        None,
        SimpleNamespace(pid=os.getpid()),
        {},
    )

    assert decision.action == "continue"
    assert lock_path.exists()


@pytest.mark.parametrize(
    "variant",
    (
        "ambiguous",
        "forged",
        "missing_receipt",
        "running_receipt",
        "active_cleanup",
        "malformed_workers",
        "symlink",
        "unverifiable",
    ),
)
def test_database_watchdog_unproved_external_legacy_cleanup_remains_blocking(
    tmp_path,
    monkeypatch,
    variant,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    _configure_ready_stale_idle_watchdog(supervisor, monkeypatch)
    lock_path, metadata, status_path = _write_external_board_legacy_cleanup_lease(
        supervisor,
        write_receipt=variant != "missing_receipt",
        receipt_status="running" if variant == "running_receipt" else "completed",
    )
    if variant == "ambiguous":
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["task_prefix"] = "## SAWM-"
        status_path.write_text(json.dumps(status), encoding="utf-8")
    elif variant == "forged":
        metadata["repository_id"] = "repository:forged"
        lock_path.write_text(json.dumps(metadata), encoding="utf-8")
    elif variant == "active_cleanup":
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status.update(
            {
                "status": "agentic_maintenance_started",
                "last_agentic_maintenance_phase": "worktree_cleanup",
            }
        )
        status_path.write_text(json.dumps(status), encoding="utf-8")
    elif variant == "malformed_workers":
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["active_worker_pids"] = 17
        status_path.write_text(json.dumps(status), encoding="utf-8")
    elif variant == "symlink":
        target = lock_path.parent / "external-cleanup-target.json"
        target.write_bytes(lock_path.read_bytes())
        lock_path.unlink()
        lock_path.symlink_to(target)
    elif variant == "unverifiable":
        lock_path.write_text("{not-json", encoding="utf-8")

    decision = supervisor._supervisor_loop_watchdog_decision(
        None,
        SimpleNamespace(pid=os.getpid()),
        {},
    )

    assert decision.action == "continue"
    assert lock_path.is_symlink() or lock_path.exists()


def test_external_cleanup_evidence_requires_exact_json_integer_authority_fields(
    tmp_path,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    lock_path, metadata, status_path = _write_external_board_legacy_cleanup_lease(supervisor)
    state_path = Path(str(metadata["state_path"]))
    receipt_path = status_path.with_name("apmc_lane_1_supervisor_maintenance_receipt.json")
    now_ts = time.time()

    baseline = supervisor._external_board_legacy_cleanup_lease_status(
        metadata,
        now_ts=now_ts,
    )
    assert baseline["external_board"] is True

    malformed_numbers = (0.9, "0", True, -1)
    for field_name in ("attempt", "pid"):
        for malformed in malformed_numbers:
            candidate = dict(metadata)
            candidate[field_name] = malformed
            result = supervisor._external_board_legacy_cleanup_lease_status(
                candidate,
                now_ts=now_ts,
            )
            assert result["external_board"] is False, (field_name, malformed)

    status = json.loads(status_path.read_text(encoding="utf-8"))
    for field_name in (
        "supervisor_pid",
        "active_worker_count",
        "worker_descendant_count",
    ):
        for malformed in malformed_numbers:
            candidate = dict(status)
            candidate[field_name] = malformed
            status_path.write_text(json.dumps(candidate), encoding="utf-8")
            result = supervisor._external_board_legacy_cleanup_lease_status(
                metadata,
                now_ts=now_ts,
            )
            assert result["external_board"] is False, (field_name, malformed)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    task_state = json.loads(state_path.read_text(encoding="utf-8"))
    for malformed in (False, 0, None, []):
        candidate = dict(task_state)
        candidate["active_task_id"] = malformed
        state_path.write_text(json.dumps(candidate), encoding="utf-8")
        result = supervisor._external_board_legacy_cleanup_lease_status(
            metadata,
            now_ts=now_ts,
        )
        assert result["external_board"] is False, ("active_task_id", malformed)
    state_path.write_text(json.dumps(task_state), encoding="utf-8")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for malformed in malformed_numbers:
        candidate = dict(receipt)
        candidate["supervisor_pid"] = malformed
        receipt_path.write_text(json.dumps(candidate), encoding="utf-8")
        result = supervisor._external_board_legacy_cleanup_lease_status(
            metadata,
            now_ts=now_ts,
        )
        assert result["external_board"] is False, ("receipt_pid", malformed)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert lock_path.exists()


def test_live_cleanup_successor_requires_exact_json_integer_authority_fields(
    tmp_path,
    monkeypatch,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    monkeypatch.setattr(
        supervisor,
        "_checkout_lock_owner_is_active",
        lambda _metadata: True,
    )
    lock_path = _write_expired_legacy_cleanup_lease(supervisor)
    metadata = json.loads(lock_path.read_text(encoding="utf-8"))
    state_path = Path(str(metadata["state_path"]))
    status_path = state_path.with_name("legacy_lane_supervisor_status.json")
    receipt_path = state_path.with_name("legacy_lane_supervisor_maintenance_receipt.json")
    now_ts = time.time()

    baseline = supervisor._expired_legacy_taskless_cleanup_lease_status(
        metadata,
        now_ts=now_ts,
    )
    assert baseline["eligible"] is True

    malformed_numbers = (0.9, "0", True, -1)
    for field_name in ("attempt", "pid"):
        for malformed in malformed_numbers:
            candidate = dict(metadata)
            candidate[field_name] = malformed
            result = supervisor._expired_legacy_taskless_cleanup_lease_status(
                candidate,
                now_ts=now_ts,
            )
            assert result["eligible"] is False, (field_name, malformed)

    status = json.loads(status_path.read_text(encoding="utf-8"))
    for field_name in (
        "supervisor_pid",
        "active_worker_count",
        "worker_descendant_count",
    ):
        for malformed in malformed_numbers:
            candidate = dict(status)
            candidate[field_name] = malformed
            status_path.write_text(json.dumps(candidate), encoding="utf-8")
            result = supervisor._expired_legacy_taskless_cleanup_lease_status(
                metadata,
                now_ts=now_ts,
            )
            assert result["eligible"] is False, (field_name, malformed)
    status_path.write_text(json.dumps(status), encoding="utf-8")

    task_state = json.loads(state_path.read_text(encoding="utf-8"))
    for malformed in (False, 0, None, []):
        candidate = dict(task_state)
        candidate["active_task_id"] = malformed
        state_path.write_text(json.dumps(candidate), encoding="utf-8")
        result = supervisor._expired_legacy_taskless_cleanup_lease_status(
            metadata,
            now_ts=now_ts,
        )
        assert result["eligible"] is False, ("active_task_id", malformed)
    state_path.write_text(json.dumps(task_state), encoding="utf-8")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for malformed in malformed_numbers:
        candidate = dict(receipt)
        candidate["supervisor_pid"] = malformed
        receipt_path.write_text(json.dumps(candidate), encoding="utf-8")
        result = supervisor._expired_legacy_taskless_cleanup_lease_status(
            metadata,
            now_ts=now_ts,
        )
        assert result["eligible"] is False, ("receipt_pid", malformed)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")


def test_cleanup_reclaim_replacement_requires_exact_json_integer_authority_fields(
    tmp_path,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    metadata = supervisor._legacy_cleanup_reclaim_metadata(
        reclaimed_from_lease_id="lease:expired-cleanup"
    )

    baseline = supervisor._legacy_cleanup_reclaim_replacement_status(metadata)
    assert baseline["resumable"] is True

    for field_name in ("attempt", "pid"):
        for malformed in (0.9, "0", True, -1):
            candidate = dict(metadata)
            candidate[field_name] = malformed
            result = supervisor._legacy_cleanup_reclaim_replacement_status(candidate)
            assert result["resumable"] is False, (field_name, malformed)


def test_database_watchdog_cas_reclaims_only_expired_legacy_taskless_cleanup(
    tmp_path,
    monkeypatch,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    monkeypatch.setattr(
        supervisor,
        "_authoritative_runnable_work_status",
        _ready_observation,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_pass_heartbeat_status",
        lambda _child, now_ts: {"stale": True, "reason": "heartbeat_stale"},
    )
    monkeypatch.setattr(
        supervisor,
        "_checkout_lock_owner_is_active",
        lambda _metadata: True,
    )
    _make_idle_recycle_safe(supervisor, monkeypatch)
    lock_path = _write_expired_legacy_cleanup_lease(supervisor)

    decision = supervisor._supervisor_loop_watchdog_decision(
        None,
        SimpleNamespace(pid=os.getpid()),
        {},
    )

    assert decision.action == "recycle"
    assert decision.reason == DATABASE_IDLE_DAEMON_STALL_REASON
    recovery = decision.detail["legacy_taskless_cleanup_lease_recovery"]
    assert recovery["attempted"] is True
    assert recovery["reclaimed"] is True
    assert recovery["replacement_released"] is True
    assert not lock_path.exists()


def test_database_watchdog_retries_reclaim_replacement_after_release_failure(
    tmp_path,
    monkeypatch,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    _configure_ready_stale_idle_watchdog(supervisor, monkeypatch)
    monkeypatch.setattr(
        supervisor,
        "_checkout_lock_owner_is_active",
        lambda _metadata: True,
    )
    lock_path = _write_expired_legacy_cleanup_lease(supervisor)
    real_release = implementation_supervisor_module.release_checkout_mutation_lease
    monkeypatch.setattr(
        implementation_supervisor_module,
        "release_checkout_mutation_lease",
        lambda _lease: False,
    )

    first = supervisor._supervisor_loop_watchdog_decision(
        None,
        SimpleNamespace(pid=os.getpid()),
        {},
    )

    assert first.action == "continue"
    replacement = json.loads(lock_path.read_text(encoding="utf-8"))
    assert replacement["operation"] == (
        implementation_supervisor_module.LEGACY_TASKLESS_CLEANUP_RECLAIM_OPERATION
    )
    assert replacement["reclaim_schema"] == (
        implementation_supervisor_module.LEGACY_TASKLESS_CLEANUP_RECLAIM_SCHEMA
    )
    assert replacement["reclaimed_from_lease_id"]
    assert replacement["reclaimer_process_birth"] == current_process_birth().to_dict()

    monkeypatch.setattr(
        implementation_supervisor_module,
        "release_checkout_mutation_lease",
        real_release,
    )
    second = supervisor._supervisor_loop_watchdog_decision(
        None,
        SimpleNamespace(pid=os.getpid()),
        {},
    )

    assert second.action == "recycle"
    assert (
        second.detail["legacy_taskless_cleanup_lease_recovery"]["reason"]
        == "legacy_cleanup_reclaim_replacement_released"
    )
    assert not lock_path.exists()


def test_database_watchdog_live_cleanup_owner_must_reach_post_cleanup_phase(
    tmp_path,
    monkeypatch,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    monkeypatch.setattr(
        supervisor,
        "_authoritative_runnable_work_status",
        _ready_observation,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_pass_heartbeat_status",
        lambda _child, now_ts: {"stale": True, "reason": "heartbeat_stale"},
    )
    monkeypatch.setattr(
        supervisor,
        "_checkout_lock_owner_is_active",
        lambda _metadata: True,
    )
    _make_idle_recycle_safe(supervisor, monkeypatch)
    lock_path = _write_expired_legacy_cleanup_lease(
        supervisor,
        owner_phase="worktree_reconciliation_replay",
    )

    decision = supervisor._supervisor_loop_watchdog_decision(
        None,
        SimpleNamespace(pid=os.getpid()),
        {},
    )

    assert decision.action == "continue"
    assert lock_path.exists()


def test_database_watchdog_reclaims_exact_cleanup_from_inactive_owner(
    tmp_path,
    monkeypatch,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    monkeypatch.setattr(
        supervisor,
        "_authoritative_runnable_work_status",
        _ready_observation,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_pass_heartbeat_status",
        lambda _child, now_ts: {"stale": True, "reason": "heartbeat_stale"},
    )
    monkeypatch.setattr(
        supervisor,
        "_checkout_lock_owner_is_active",
        lambda _metadata: False,
    )
    _make_idle_recycle_safe(supervisor, monkeypatch)
    lock_path = _write_expired_legacy_cleanup_lease(
        supervisor,
        owner_phase="worktree_reconciliation_replay",
        write_receipt=False,
    )

    decision = supervisor._supervisor_loop_watchdog_decision(
        None,
        SimpleNamespace(pid=os.getpid()),
        {},
    )

    assert decision.action == "recycle"
    assert not lock_path.exists()


def test_database_watchdog_accepts_reparented_stable_maintenance_birth(
    tmp_path,
    monkeypatch,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    _configure_ready_stale_idle_watchdog(supervisor, monkeypatch)
    monkeypatch.setattr(
        supervisor,
        "_checkout_lock_owner_is_active",
        lambda _metadata: True,
    )
    reparented_birth = current_process_birth().to_dict()
    reparented_birth["parent_pid"] += 1
    lock_path = _write_expired_legacy_cleanup_lease(
        supervisor,
        receipt_overrides={"process_birth": reparented_birth},
    )

    decision = supervisor._supervisor_loop_watchdog_decision(
        None,
        SimpleNamespace(pid=os.getpid()),
        {},
    )

    assert decision.action == "recycle"
    assert not lock_path.exists()


@pytest.mark.parametrize(
    "variant",
    (
        "missing",
        "forged",
        "binding",
        "stale",
        "pid",
        "birth",
        "incomplete",
        "malformed_workers",
    ),
)
def test_database_watchdog_rejects_invalid_live_owner_maintenance_receipt(
    tmp_path,
    monkeypatch,
    variant,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    monkeypatch.setattr(
        supervisor,
        "_authoritative_runnable_work_status",
        _ready_observation,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_pass_heartbeat_status",
        lambda _child, now_ts: {"stale": True, "reason": "heartbeat_stale"},
    )
    monkeypatch.setattr(
        supervisor,
        "_checkout_lock_owner_is_active",
        lambda _metadata: True,
    )
    _make_idle_recycle_safe(supervisor, monkeypatch)
    write_receipt = variant != "missing"
    receipt_overrides: dict[str, object] = {}
    if variant == "forged":
        receipt_overrides["schema"] = "forged/maintenance-receipt@1"
    elif variant == "binding":
        receipt_overrides["state_path"] = str(
            (supervisor.config.repo_root / "forged-state.json").resolve()
        )
    elif variant == "stale":
        stale_at = datetime.now(UTC) - timedelta(hours=2)
        receipt_overrides.update(
            {
                "maintenance_started_at": (stale_at - timedelta(minutes=5)).isoformat(),
                "updated_at": stale_at.isoformat(),
                "completed_at": stale_at.isoformat(),
            }
        )
    elif variant == "pid":
        receipt_overrides["supervisor_pid"] = os.getpid() + 100_000
    elif variant == "birth":
        forged_birth = current_process_birth().to_dict()
        forged_birth["start_time_ticks"] += 1
        receipt_overrides["process_birth"] = forged_birth
    elif variant == "incomplete":
        receipt_overrides.update({"status": "running", "completed_at": ""})
    lock_path = _write_expired_legacy_cleanup_lease(
        supervisor,
        write_receipt=write_receipt,
        receipt_overrides=receipt_overrides,
    )
    if variant == "malformed_workers":
        status_path = (
            supervisor.config.repo_root
            / "legacy-owner-state"
            / "legacy_lane_supervisor_status.json"
        )
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["active_worker_pids"] = 17
        status_path.write_text(json.dumps(status), encoding="utf-8")

    decision = supervisor._supervisor_loop_watchdog_decision(
        None,
        SimpleNamespace(pid=os.getpid()),
        {},
    )

    assert decision.action == "continue"
    assert lock_path.exists()


@pytest.mark.parametrize(
    ("operation", "task_id", "extra"),
    [
        ("cleanup_backlogged_worktrees", "SAWM-005", {}),
        ("merge_branch_to_main", "", {}),
        (
            "cleanup_backlogged_worktrees",
            "",
            {"protected_recovery_required": True},
        ),
    ],
)
def test_database_watchdog_never_expires_task_merge_or_protected_lease(
    tmp_path,
    monkeypatch,
    operation,
    task_id,
    extra,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    monkeypatch.setattr(
        supervisor,
        "_authoritative_runnable_work_status",
        _ready_observation,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_pass_heartbeat_status",
        lambda _child, now_ts: {"stale": True, "reason": "heartbeat_stale"},
    )
    monkeypatch.setattr(
        supervisor,
        "_checkout_lock_owner_is_active",
        lambda _metadata: True,
    )
    _make_idle_recycle_safe(supervisor, monkeypatch)
    lock_path = _write_expired_legacy_cleanup_lease(
        supervisor,
        operation=operation,
        task_id=task_id,
        extra=extra,
    )

    decision = supervisor._supervisor_loop_watchdog_decision(
        None,
        SimpleNamespace(pid=os.getpid()),
        {},
    )

    assert decision.action == "continue"
    assert lock_path.exists()


def test_database_watchdog_preserves_cleanup_lease_for_canonical_active_task(
    tmp_path,
    monkeypatch,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    active = _ready_observation()
    active["active_task_ids"] = ["SAWM-005"]
    active["same_shard_active_task_ids"] = ["SAWM-005"]
    monkeypatch.setattr(
        supervisor,
        "_authoritative_runnable_work_status",
        lambda: active,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_pass_heartbeat_status",
        lambda _child, now_ts: {"stale": True, "reason": "heartbeat_stale"},
    )
    monkeypatch.setattr(
        supervisor,
        "_checkout_lock_owner_is_active",
        lambda _metadata: True,
    )
    _make_idle_recycle_safe(supervisor, monkeypatch)
    lock_path = _write_expired_legacy_cleanup_lease(supervisor)

    decision = supervisor._supervisor_loop_watchdog_decision(
        None,
        SimpleNamespace(pid=os.getpid()),
        {},
    )

    assert decision.action == "continue"
    assert lock_path.exists()


def test_database_watchdog_preserves_cleanup_lease_for_child_descendant(
    tmp_path,
    monkeypatch,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=1)
    monkeypatch.setattr(
        supervisor,
        "_authoritative_runnable_work_status",
        _ready_observation,
    )
    monkeypatch.setattr(
        supervisor,
        "_database_pass_heartbeat_status",
        lambda _child, now_ts: {"stale": True, "reason": "heartbeat_stale"},
    )
    monkeypatch.setattr(
        supervisor,
        "_checkout_lock_owner_is_active",
        lambda _metadata: True,
    )
    monkeypatch.setattr(supervisor, "_active_agent_worker_processes", lambda: [])
    monkeypatch.setattr(
        supervisor,
        "_active_validation_subprocess_exists",
        lambda: False,
    )
    monkeypatch.setattr(
        implementation_supervisor_module,
        "descendant_processes",
        lambda _pid: [SimpleNamespace(pid=982451653)],
    )
    lock_path = _write_expired_legacy_cleanup_lease(supervisor)

    decision = supervisor._supervisor_loop_watchdog_decision(
        None,
        SimpleNamespace(pid=os.getpid()),
        {},
    )

    assert decision.action == "continue"
    assert lock_path.exists()


@pytest.mark.parametrize(
    ("lane_index", "observation"),
    [
        (0, _ready_observation(same_shard=False)),
        (
            2,
            {
                "available": True,
                "reason": "authoritative_readiness_observed",
                "task_source_revision": 43,
                "ready_task_ids": [],
                "same_shard_ready_task_ids": [],
                "active_task_ids": [],
                "same_shard_active_task_ids": [],
            },
        ),
    ],
)
def test_database_watchdog_suppresses_nonessential_lane_maintenance(
    tmp_path,
    monkeypatch,
    lane_index,
    observation,
) -> None:
    supervisor = _supervisor(tmp_path, lane_index=lane_index)
    monkeypatch.setattr(
        supervisor,
        "_authoritative_runnable_work_status",
        lambda: observation,
    )
    maintenance_calls: list[bool] = []
    monkeypatch.setattr(
        supervisor,
        "_run_once_with_maintenance",
        lambda _update: maintenance_calls.append(True),
    )

    decision = supervisor._supervisor_loop_watchdog_decision(
        None,
        SimpleNamespace(pid=os.getpid()),
        {},
    )

    assert decision.action == "continue"
    assert maintenance_calls == []


def test_authoritative_database_readiness_filters_manual_and_home_shard(
    tmp_path,
    monkeypatch,
) -> None:
    # Production scheduler configs carry the Markdown heading prefix while
    # canonical database aliases omit the heading marker.  The watchdog must
    # use the same normalization as DatabaseImplementationDaemon or it cannot
    # observe ready work and recycle a stalled idle lane.
    supervisor = _supervisor(
        tmp_path,
        lane_index=3,
        task_prefix="## SAWM-",
    )

    def task_id_for_lane(lane_index: int) -> str:
        for ordinal in range(1, 500):
            task_id = f"SAWM-{ordinal:03d}"
            digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
            if int(digest[:8], 16) % 4 == lane_index:
                return task_id
        raise AssertionError("could not construct deterministic shard fixture")

    home_id = task_id_for_lane(3)
    other_id = task_id_for_lane(0)
    tasks = (
        SimpleNamespace(task_alias=home_id, task_cid=f"task:{home_id}", body={}),
        SimpleNamespace(task_alias=other_id, task_cid=f"task:{other_id}", body={}),
        SimpleNamespace(
            task_alias="SAWM-999",
            task_cid="task:SAWM-999",
            body={"completion": "manual"},
        ),
    )
    observed: dict[str, object] = {}

    class FakeDatabaseTaskSource:
        def __init__(self, target, **kwargs):
            observed["target"] = target
            observed.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def ready_tasks(self, *, limit):
            observed["ready_limit"] = limit
            return SimpleNamespace(tasks=tasks, revision=51, next_cursor="")

        def list_tasks(self, *, status, limit):
            observed["active_status"] = status
            observed["active_limit"] = limit
            return SimpleNamespace(tasks=(), revision=52, next_cursor="")

    monkeypatch.setattr(
        database_task_source_module,
        "DatabaseTaskSource",
        FakeDatabaseTaskSource,
    )

    result = supervisor._authoritative_runnable_work_status()

    assert result["available"] is True
    assert result["task_source_revision"] == 52
    assert result["ready_task_ids"] == [home_id, other_id]
    assert result["same_shard_ready_task_ids"] == [home_id]
    assert observed["target"] == "quack:127.0.0.1:24068"
    assert observed["install_schema"] is False
    assert observed["active_status"] == ("claimed", "in_progress", "running")
