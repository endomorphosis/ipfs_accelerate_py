from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    DatabaseProgramConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as supervisor_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    supervisor_loop as supervisor_loop_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_runtime
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import (
    ManagedDaemonSpec,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    AdoptedManagedDaemonProcess,
    PortalImplementationSupervisor,
    TodoSupervisorConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import (
    SupervisorLoop,
    SupervisorLoopConfig,
    SupervisorLoopDecision,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    SUPERVISED_CHILD_IDENTITY_PATH_ENV,
    SUPERVISED_CHILD_OWNER_SCOPE_ENV,
    SupervisedChild,
    SupervisedChildIdentity,
    SupervisedChildSpec,
    adopt_or_launch_supervised_child,
    adopt_supervised_child,
    clear_child_pid_file,
    launch_supervised_child,
    terminate_supervised_child,
    wait_for_child_exit,
)
from ipfs_accelerate_py.agent_supervisor.worktree_lifecycle import (
    OwnerLiveness,
    ProcessBirthIdentity,
)


def _supervisor(
    tmp_path: Path,
    *,
    required_task_ids: tuple[str, ...] = (),
    database_managed: bool = False,
) -> PortalImplementationSupervisor:
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    return PortalImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task-state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            database_program=(
                DatabaseProgramConfig(
                    authority_mode="quack",
                    task_source_kind="duckdb",
                    endpoint_secret_handle="env://QUACK_TOKEN",
                    quack_endpoint="quack:127.0.0.1:45123",
                    store_id="control.duckdb",
                    store_generation="generation-1",
                    schema_revision="schema-v1",
                )
                if database_managed
                else None
            ),
            database_owner_session_id=(
                "test-database-owner" if database_managed else ""
            ),
            manual_completion_authority_task_ids=required_task_ids,
            manual_completion_authority_required_task_ids=required_task_ids,
        )
    )


def _write_identity(
    supervisor: PortalImplementationSupervisor,
    *,
    pid: int,
    command: tuple[str, ...],
    start_time_ticks: int = 1234,
    owner_scope: dict[str, str] | None = None,
) -> Path:
    identity = SupervisedChildIdentity(
        process_birth=ProcessBirthIdentity(
            pid=pid,
            start_time_ticks=start_time_ticks,
            boot_id="boot-test",
            parent_pid=17,
        ),
        command=command,
        owner_scope=(
            supervisor._managed_daemon_owner_scope()
            if owner_scope is None
            else owner_scope
        ),
        created_at="2026-08-03T00:00:00+00:00",
    )
    path = supervisor._managed_daemon_identity_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(identity.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_direct_managed_daemon_identity_requires_direct_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    command = tuple(supervisor._build_daemon_command())
    captured: dict[str, object] = {}

    def capture_identity(path: Path, **kwargs: object) -> None:
        captured.update({"path": path, **kwargs})

    monkeypatch.setattr(
        supervisor_module,
        "write_supervised_child_identity",
        capture_identity,
    )

    supervisor._write_managed_daemon_identity(pid=123, command=command)

    assert captured == {
        "path": supervisor._managed_daemon_identity_path(),
        "pid": 123,
        "command": command,
        "owner_scope": supervisor._managed_daemon_owner_scope(),
        "require_direct_child": True,
    }


def test_authority_arg_change_blocks_without_signalling_or_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path, required_task_ids=("TEST-001",))
    pid = 111
    pid_path = supervisor._managed_daemon_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{pid}\n", encoding="utf-8")
    desired = tuple(supervisor._build_daemon_command())
    old_command = tuple(
        part
        for index, part in enumerate(desired)
        if part not in {
            "--manual-completion-authority-task-id",
            "--manual-completion-authority-required-task-id",
        }
        and not (
            index > 0
            and desired[index - 1]
            in {
                "--manual-completion-authority-task-id",
                "--manual-completion-authority-required-task-id",
            }
        )
    )
    _write_identity(supervisor, pid=pid, command=old_command)
    monkeypatch.setattr(supervisor_module, "process_is_running", lambda value: int(value) == pid)
    monkeypatch.setattr(supervisor_module, "process_command_line", lambda _pid: " ".join(old_command))
    monkeypatch.setattr(supervisor_module, "read_process_command_argv", lambda _pid: old_command)
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.ALIVE,
    )
    monkeypatch.setattr(
        supervisor_module,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail(
            "authority-drifted daemon was signalled"
        ),
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is False
    assert result["blocked"] is True
    assert result["reason"] == "managed_daemon_authority_identity_mismatch"
    assert pid_path.read_text(encoding="utf-8").strip() == str(pid)
    assert supervisor._managed_daemon_identity_path().exists()


def test_pid_reuse_identity_mismatch_never_signals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path, required_task_ids=("TEST-001",))
    pid = 222
    pid_path = supervisor._managed_daemon_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{pid}\n", encoding="utf-8")
    old_command = tuple(supervisor._build_daemon_command()[:-2])
    _write_identity(supervisor, pid=pid, command=old_command)
    monkeypatch.setattr(supervisor_module, "process_is_running", lambda value: int(value) == pid)
    monkeypatch.setattr(supervisor_module, "process_command_line", lambda _pid: "unrelated process")
    monkeypatch.setattr(
        supervisor_module,
        "read_process_command_argv",
        lambda _pid: ("unrelated", "process"),
    )
    monkeypatch.setattr(supervisor, "_list_process_details", lambda: [])
    monkeypatch.setattr(
        supervisor,
        "_find_matching_managed_daemon_lane_process",
        lambda **_kwargs: None,
    )
    reused_birth = ProcessBirthIdentity(
        pid=pid,
        start_time_ticks=5678,
        boot_id="boot-test",
        parent_pid=23,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_birth",
        lambda _pid: reused_birth,
    )
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.DEAD,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_birth",
        lambda _pid: ProcessBirthIdentity(
            pid=pid,
            start_time_ticks=5678,
            boot_id="boot-test",
            parent_pid=23,
        ),
    )
    monkeypatch.setattr(
        supervisor_module,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail("reused PID was signalled"),
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is True, result
    assert result["reason"] == "managed_daemon_pid_reused"
    assert not pid_path.exists()
    assert not supervisor._managed_daemon_identity_path().exists()
    assert set(result["quarantined"]) == {"pid", "identity"}


def test_pid_reuse_with_matching_daemon_blocks_duplicate_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path, required_task_ids=("TEST-001",))
    pid = 223
    pid_path = supervisor._managed_daemon_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{pid}\n", encoding="utf-8")
    desired = tuple(supervisor._build_daemon_command())
    _write_identity(supervisor, pid=pid, command=desired)
    monkeypatch.setattr(
        supervisor_module,
        "process_is_running",
        lambda value: int(value) == pid,
    )
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.DEAD,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_birth",
        lambda _pid: ProcessBirthIdentity(
            pid=pid,
            start_time_ticks=5678,
            boot_id="boot-test",
            parent_pid=23,
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_find_matching_managed_daemon_pid",
        lambda **_kwargs: pid,
    )
    monkeypatch.setattr(
        supervisor_module,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail("unowned matching PID was signalled"),
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result == {
        "repaired": False,
        "blocked": True,
        "reason": "matching_managed_daemon_ownership_unproven",
        "path": str(pid_path),
        "identity_path": str(supervisor._managed_daemon_identity_path()),
        "pid": pid,
    }
    assert pid_path.exists()
    assert supervisor._managed_daemon_identity_path().exists()


def test_live_legacy_config_mismatch_blocks_without_unlink_or_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path, required_task_ids=("TEST-001",))
    pid = 333
    pid_path = supervisor._managed_daemon_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{pid}\n", encoding="utf-8")
    monkeypatch.setattr(supervisor_module, "process_is_running", lambda value: int(value) == pid)
    monkeypatch.setattr(supervisor_module, "process_command_line", lambda _pid: "legacy daemon config")
    monkeypatch.setattr(
        supervisor_module,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail("unowned legacy PID was signalled"),
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["blocked"] is True
    assert result["reason"] == "managed_daemon_ownership_unproven"
    assert pid_path.read_text(encoding="utf-8").strip() == str(pid)
    assert not supervisor._managed_daemon_identity_path().exists()


def test_exact_desired_legacy_child_is_migrated_to_identity_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path, required_task_ids=("TEST-001",))
    pid = 334
    pid_path = supervisor._managed_daemon_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{pid}\n", encoding="utf-8")
    desired = tuple(supervisor._build_daemon_command())
    monkeypatch.setattr(supervisor_module, "process_is_running", lambda value: int(value) == pid)
    monkeypatch.setattr(supervisor_module, "process_command_line", lambda _pid: " ".join(desired))
    monkeypatch.setattr(supervisor_module, "read_process_command_argv", lambda _pid: desired)
    monkeypatch.setattr(
        supervisor_runtime,
        "read_process_birth",
        lambda value: ProcessBirthIdentity(
            pid=int(value),
            start_time_ticks=99,
            boot_id="boot-test",
            parent_pid=17,
        ),
    )
    monkeypatch.setattr(
        supervisor_module,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail("matching legacy child was signalled"),
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is True
    assert result["reason"] == (
        "active_legacy_managed_daemon_identity_migrated"
    )
    identity = supervisor_runtime.load_supervised_child_identity(
        supervisor._managed_daemon_identity_path()
    )
    assert identity is not None
    assert identity.process_birth.pid == pid
    assert identity.command == desired
    assert dict(identity.owner_scope) == supervisor._managed_daemon_owner_scope()
    assert pid_path.read_text(encoding="utf-8").strip() == str(pid)


def test_orphaned_live_identity_reconstructs_pid_marker_without_duplicate_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path, required_task_ids=("TEST-001",))
    pid = 335
    desired = tuple(supervisor._build_daemon_command())
    _write_identity(supervisor, pid=pid, command=desired)
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.ALIVE,
    )
    monkeypatch.setattr(
        supervisor_module,
        "process_is_running",
        lambda value: int(value) == pid,
    )
    monkeypatch.setattr(
        supervisor_module,
        "process_command_line",
        lambda _pid: " ".join(desired),
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_command_argv",
        lambda _pid: desired,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_birth",
        lambda value: ProcessBirthIdentity(
            pid=int(value),
            start_time_ticks=1234,
            boot_id="boot-test",
            parent_pid=17,
        ),
    )
    monkeypatch.setattr(
        supervisor_module,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail("owned live daemon was signalled"),
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is True
    assert result["orphaned_identity_recovered"] is True
    assert result["reason"] == (
        "orphaned_live_managed_daemon_pid_reconstructed"
    )
    assert supervisor._managed_daemon_pid_path().read_text(
        encoding="utf-8"
    ).strip() == str(pid)
    assert supervisor._managed_daemon_identity_path().exists()


def test_dead_orphaned_database_identity_is_quarantined_for_fresh_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path, database_managed=True)
    pid = 340
    desired = tuple(supervisor._build_daemon_command())
    identity_path = _write_identity(supervisor, pid=pid, command=desired)
    original = identity_path.read_bytes()
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.DEAD,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_birth",
        lambda _pid: None,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_command_argv",
        lambda _pid: None,
    )
    monkeypatch.setattr(
        supervisor,
        "_find_matching_managed_daemon_pid",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        supervisor_module,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail("dead child was signalled"),
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is True
    assert result["blocked"] is False
    assert result["reason"] == (
        "orphaned_dead_managed_database_identity_quarantined"
    )
    assert set(result["quarantined"]) == {"identity"}
    backup = Path(result["quarantined"]["identity"])
    assert backup.name.startswith(
        supervisor._managed_daemon_identity_path().name
        + ".stale-child-identity-"
    )
    assert backup.read_bytes() == original
    assert not supervisor._managed_daemon_pid_path().exists()
    assert not identity_path.exists()
    assert supervisor.ensure_managed_daemon_pid_file()["reason"] == "missing"


def test_unknown_orphaned_database_identity_remains_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path, database_managed=True)
    pid = 341
    desired = tuple(supervisor._build_daemon_command())
    identity_path = _write_identity(supervisor, pid=pid, command=desired)
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.UNKNOWN,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_birth",
        lambda _pid: None,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_command_argv",
        lambda _pid: None,
    )
    monkeypatch.setattr(
        supervisor,
        "_find_matching_managed_daemon_pid",
        lambda **_kwargs: pytest.fail("unknown identity searched for a peer"),
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is False
    assert result["blocked"] is True
    assert result["reason"] == (
        "orphaned_managed_database_identity_liveness_unknown"
    )
    assert identity_path.exists()
    assert not supervisor._managed_daemon_pid_path().exists()


def test_dead_orphaned_database_identity_scope_mismatch_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path, database_managed=True)
    pid = 342
    desired = tuple(supervisor._build_daemon_command())
    identity_path = _write_identity(
        supervisor,
        pid=pid,
        command=desired,
        owner_scope={"repo_root": str(tmp_path / "other-repository")},
    )
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: pytest.fail("scope-mismatched identity was trusted"),
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is False
    assert result["blocked"] is True
    assert result["reason"] == (
        "orphaned_managed_database_identity_scope_mismatch"
    )
    assert identity_path.exists()
    assert not supervisor._managed_daemon_pid_path().exists()


def _predecessor_owner_identity(
    supervisor: PortalImplementationSupervisor,
    *,
    pid: int,
) -> tuple[list[str], dict[str, str]]:
    command = list(supervisor._build_daemon_command())
    owner_index = command.index("--owner-session-id")
    command[owner_index + 1] = "predecessor-owner-session"
    scope = dict(supervisor._managed_daemon_owner_scope())
    scope["database_owner_session_id"] = "predecessor-owner-session"
    return command, scope


def test_dead_predecessor_owner_session_identity_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path, database_managed=True)
    pid = 343
    command, scope = _predecessor_owner_identity(supervisor, pid=pid)
    identity_path = _write_identity(
        supervisor,
        pid=pid,
        command=tuple(command),
        owner_scope=scope,
    )
    original = identity_path.read_bytes()
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.DEAD,
    )
    monkeypatch.setattr(
        supervisor,
        "_find_matching_managed_daemon_pid",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        supervisor_module,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail("dead predecessor was signalled"),
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is True
    assert result["blocked"] is False
    assert result["reason"] == (
        "orphaned_dead_managed_database_identity_quarantined"
    )
    assert Path(result["quarantined"]["identity"]).read_bytes() == original
    assert not identity_path.exists()
    assert not supervisor._managed_daemon_pid_path().exists()


def test_dead_identity_does_not_quarantine_live_predecessor_lane_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path, database_managed=True)
    identity_pid = 348
    live_pid = 349
    command, scope = _predecessor_owner_identity(
        supervisor,
        pid=identity_pid,
    )
    identity_path = _write_identity(
        supervisor,
        pid=identity_pid,
        command=tuple(command),
        owner_scope=scope,
    )
    original = identity_path.read_bytes()
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.DEAD,
    )
    monkeypatch.setattr(
        supervisor,
        "_find_matching_managed_daemon_pid",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        supervisor,
        "_list_process_details",
        lambda: [(live_pid, " ".join(command))],
    )
    monkeypatch.setattr(
        supervisor_module,
        "process_is_running",
        lambda pid: int(pid) == live_pid,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_command_argv",
        lambda pid: tuple(command) if int(pid) == live_pid else None,
    )
    monkeypatch.setattr(
        supervisor_module,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail("lane process was signalled"),
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is False
    assert result["blocked"] is True
    assert result["reason"] == (
        "matching_managed_daemon_lane_ownership_unproven"
    )
    assert result["pid"] == live_pid
    assert identity_path.read_bytes() == original


def test_reused_identity_pid_still_scans_live_predecessor_lane_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path, database_managed=True)
    reused_pid = 350
    command, scope = _predecessor_owner_identity(
        supervisor,
        pid=reused_pid,
    )
    identity_path = _write_identity(
        supervisor,
        pid=reused_pid,
        command=tuple(command),
        owner_scope=scope,
    )
    supervisor._managed_daemon_pid_path().write_text(
        f"{reused_pid}\n",
        encoding="utf-8",
    )
    original = identity_path.read_bytes()
    # The persisted exact birth is dead, but the numeric PID has already
    # been reused by a live daemon in the same durable lane.  A broad scan
    # must not exclude that PID merely because it appears in the stale file.
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.DEAD,
    )
    monkeypatch.setattr(
        supervisor,
        "_find_matching_managed_daemon_pid",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        supervisor,
        "_list_process_details",
        lambda: [(reused_pid, " ".join(command))],
    )
    monkeypatch.setattr(
        supervisor_module,
        "process_is_running",
        lambda pid: int(pid) == reused_pid,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_command_argv",
        lambda pid: tuple(command) if int(pid) == reused_pid else None,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_birth",
        lambda _pid: ProcessBirthIdentity(
            pid=reused_pid,
            start_time_ticks=5678,
            boot_id="boot-test",
            parent_pid=23,
        ),
    )
    monkeypatch.setattr(
        supervisor_module,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail("reused lane PID was signalled"),
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is False
    assert result["blocked"] is True
    assert result["reason"] == (
        "matching_managed_daemon_lane_ownership_unproven"
    )
    assert result["pid"] == reused_pid
    assert identity_path.read_bytes() == original


def test_pid_reuse_birth_change_during_scan_preserves_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path, required_task_ids=("TEST-001",))
    pid = 351
    pid_path = supervisor._managed_daemon_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{pid}\n", encoding="utf-8")
    identity_path = _write_identity(
        supervisor,
        pid=pid,
        command=tuple(supervisor._build_daemon_command()[:-2]),
    )
    original_pid = pid_path.read_bytes()
    original_identity = identity_path.read_bytes()
    births = iter(
        (
            ProcessBirthIdentity(
                pid=pid,
                start_time_ticks=5678,
                boot_id="boot-test",
                parent_pid=23,
            ),
            ProcessBirthIdentity(
                pid=pid,
                start_time_ticks=6789,
                boot_id="boot-test",
                parent_pid=24,
            ),
        )
    )
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.DEAD,
    )
    monkeypatch.setattr(
        supervisor_module,
        "process_is_running",
        lambda value: int(value) == pid,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_birth",
        lambda _pid: next(births),
    )
    monkeypatch.setattr(
        supervisor,
        "_find_matching_managed_daemon_pid",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        supervisor,
        "_find_matching_managed_daemon_lane_process",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        supervisor_module,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail("reused PID was signalled"),
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is False
    assert result["blocked"] is True
    assert result["reason"] == "managed_daemon_pid_reuse_birth_advanced"
    assert pid_path.read_bytes() == original_pid
    assert identity_path.read_bytes() == original_identity


@pytest.mark.parametrize("unavailable_read", (1, 2, 3))
def test_pid_reuse_birth_unavailable_at_any_read_preserves_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unavailable_read: int,
) -> None:
    supervisor = _supervisor(tmp_path, required_task_ids=("TEST-001",))
    pid = 352
    pid_path = supervisor._managed_daemon_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{pid}\n", encoding="utf-8")
    identity_path = _write_identity(
        supervisor,
        pid=pid,
        command=tuple(supervisor._build_daemon_command()[:-2]),
    )
    original_pid = pid_path.read_bytes()
    original_identity = identity_path.read_bytes()
    reused_birth = ProcessBirthIdentity(
        pid=pid,
        start_time_ticks=5678,
        boot_id="boot-test",
        parent_pid=23,
    )
    reads = 0

    def read_birth(_pid: int) -> ProcessBirthIdentity:
        nonlocal reads
        reads += 1
        if reads == unavailable_read:
            raise OSError("process disappeared")
        return reused_birth

    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.DEAD,
    )
    monkeypatch.setattr(
        supervisor_module,
        "process_is_running",
        lambda value: int(value) == pid,
    )
    monkeypatch.setattr(supervisor_module, "read_process_birth", read_birth)
    monkeypatch.setattr(
        supervisor,
        "_find_matching_managed_daemon_pid",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        supervisor,
        "_find_matching_managed_daemon_lane_process",
        lambda **_kwargs: None,
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is False
    assert result["blocked"] is True
    assert result["reason"] in {
        "managed_daemon_pid_reuse_birth_unproven",
        "managed_daemon_pid_reuse_birth_advanced",
    }
    assert pid_path.read_bytes() == original_pid
    assert identity_path.read_bytes() == original_identity


@pytest.mark.parametrize("liveness", [OwnerLiveness.ALIVE, OwnerLiveness.UNKNOWN])
def test_non_dead_predecessor_owner_session_identity_stays_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    liveness: OwnerLiveness,
) -> None:
    supervisor = _supervisor(tmp_path, database_managed=True)
    pid = 344
    command, scope = _predecessor_owner_identity(supervisor, pid=pid)
    identity_path = _write_identity(
        supervisor,
        pid=pid,
        command=tuple(command),
        owner_scope=scope,
    )
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: liveness,
    )
    monkeypatch.setattr(
        supervisor_module,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail("predecessor was signalled"),
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is False
    assert result["blocked"] is True
    assert result["reason"] == (
        "orphaned_managed_database_identity_scope_mismatch"
    )
    assert identity_path.exists()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "argv_drift"])
def test_predecessor_lane_match_rejects_malformed_or_drifted_owner_argv(
    tmp_path: Path,
    mutation: str,
) -> None:
    supervisor = _supervisor(tmp_path, database_managed=True)
    pid = 345
    command, scope = _predecessor_owner_identity(supervisor, pid=pid)
    owner_index = command.index("--owner-session-id")
    if mutation == "missing":
        del command[owner_index : owner_index + 2]
    elif mutation == "duplicate":
        command.extend(("--owner-session-id", "second-owner"))
    else:
        command.append("--foreign-lane-argument")
    identity_path = _write_identity(
        supervisor,
        pid=pid,
        command=tuple(command),
        owner_scope=scope,
    )
    identity = supervisor_module.load_supervised_child_identity(identity_path)

    if mutation == "empty_owner":
        assert identity is None
        result = supervisor.ensure_managed_daemon_pid_file()
        assert result["blocked"] is True
        assert result["reason"] == (
            "orphaned_managed_database_identity_unproven"
        )
        return
    assert identity is not None
    assert not supervisor._managed_daemon_identity_matches_lane_scope(
        identity,
        pid=pid,
    )


@pytest.mark.parametrize(
    "mutation",
    ["missing_owner", "empty_owner", "extra_field", "durable_field_drift"],
)
def test_predecessor_lane_match_rejects_scope_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    supervisor = _supervisor(tmp_path, database_managed=True)
    pid = 346
    command, scope = _predecessor_owner_identity(supervisor, pid=pid)
    if mutation == "missing_owner":
        scope.pop("database_owner_session_id")
    elif mutation == "empty_owner":
        scope["database_owner_session_id"] = ""
    elif mutation == "extra_field":
        scope["foreign_lane"] = "other"
    else:
        scope["state_prefix"] = "foreign-state-prefix"
    identity_path = _write_identity(
        supervisor,
        pid=pid,
        command=tuple(command),
        owner_scope=scope,
    )
    identity = supervisor_module.load_supervised_child_identity(identity_path)

    if mutation == "empty_owner":
        assert identity is None
        result = supervisor.ensure_managed_daemon_pid_file()
        assert result["blocked"] is True
        assert result["reason"] == (
            "orphaned_managed_database_identity_unproven"
        )
        return
    assert identity is not None
    assert not supervisor._managed_daemon_identity_matches_lane_scope(
        identity,
        pid=pid,
    )


def test_current_owner_session_identity_uses_exact_scope_path(
    tmp_path: Path,
) -> None:
    supervisor = _supervisor(tmp_path, database_managed=True)
    pid = 347
    identity_path = _write_identity(
        supervisor,
        pid=pid,
        command=tuple(supervisor._build_daemon_command()),
    )
    identity = supervisor_module.load_supervised_child_identity(identity_path)

    assert identity is not None
    assert supervisor._managed_daemon_identity_matches_scope(identity, pid=pid)


def test_live_identity_repairs_wrong_raw_pid_without_duplicate_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path, required_task_ids=("TEST-001",))
    stale_pid = 336
    identity_pid = 337
    desired = tuple(supervisor._build_daemon_command())
    pid_path = supervisor._managed_daemon_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{stale_pid}\n", encoding="utf-8")
    _write_identity(supervisor, pid=identity_pid, command=desired)
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.ALIVE,
    )
    monkeypatch.setattr(
        supervisor_module,
        "process_is_running",
        lambda value: int(value) == identity_pid,
    )
    monkeypatch.setattr(
        supervisor_module,
        "process_command_line",
        lambda _pid: " ".join(desired),
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_command_argv",
        lambda value: desired if int(value) == identity_pid else None,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_birth",
        lambda value: ProcessBirthIdentity(
            pid=int(value),
            start_time_ticks=1234,
            boot_id="boot-test",
            parent_pid=17,
        ),
    )
    monkeypatch.setattr(
        supervisor_module,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail("owned live daemon was signalled"),
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is True
    assert result["recorded_pid_reconciled"] == stale_pid
    assert result["reason"] == (
        "managed_daemon_pid_reconciled_from_live_identity"
    )
    assert pid_path.read_text(encoding="utf-8").strip() == str(identity_pid)
    assert supervisor._managed_daemon_identity_path().exists()


def test_shutdown_recovers_orphan_identity_before_fencing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    pid = 338
    desired = tuple(supervisor._build_daemon_command())
    _write_identity(supervisor, pid=pid, command=desired)
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.ALIVE,
    )
    monkeypatch.setattr(
        supervisor_module,
        "process_is_running",
        lambda value: int(value) == pid,
    )
    monkeypatch.setattr(
        supervisor_module,
        "process_command_line",
        lambda _pid: " ".join(desired),
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_command_argv",
        lambda _pid: desired,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_birth",
        lambda value: ProcessBirthIdentity(
            pid=int(value),
            start_time_ticks=1234,
            boot_id="boot-test",
            parent_pid=17,
        ),
    )
    liveness = iter(
        (
            OwnerLiveness.ALIVE,
            OwnerLiveness.DEAD,
        )
    )
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: next(liveness),
    )
    monkeypatch.setattr(
        supervisor_module,
        "terminate_pid_tree",
        lambda value, **_kwargs: int(value) == pid,
    )
    monkeypatch.setattr(
        supervisor,
        "_find_matching_managed_daemon_pid",
        lambda **_kwargs: None,
    )

    result = supervisor._terminate_managed_daemon_tree()

    assert result["terminated"] is True
    assert result["quiesced"] is True
    assert result["pid"] == pid
    assert not supervisor._managed_daemon_pid_path().exists()
    assert not supervisor._managed_daemon_identity_path().exists()


def test_shutdown_refuses_managed_daemon_when_boot_identity_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    pid = 339
    desired = tuple(supervisor._build_daemon_command())
    _write_identity(supervisor, pid=pid, command=desired)
    monkeypatch.setattr(
        supervisor_module,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.ALIVE,
    )
    monkeypatch.setattr(
        supervisor_module,
        "read_process_birth",
        lambda value: ProcessBirthIdentity(
            pid=int(value),
            start_time_ticks=1234,
            boot_id="",
            parent_pid=17,
        ),
    )
    monkeypatch.setattr(
        supervisor_module,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail(
            "managed daemon with unknown boot identity was signalled"
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "_find_matching_managed_daemon_pid",
        lambda **_kwargs: None,
    )

    result = supervisor._terminate_managed_daemon_tree()

    assert result["terminated"] is False
    assert result["quiesced"] is False
    assert result["daemon_fence"]["safe_to_restart"] is False
    assert result["daemon_fence"]["reason"] == (
        "managed_daemon_ownership_liveness_unknown"
    )
    assert supervisor._managed_daemon_identity_path().exists()


def test_shared_launcher_refuses_unreconciled_identity_before_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    pid_path = repo / "state" / "child.pid"
    identity_path = repo / "state" / "child.identity.json"
    identity_path.parent.mkdir(parents=True)
    identity_path.write_text("unreconciled\n", encoding="utf-8")
    monkeypatch.setattr(
        supervisor_runtime,
        "launch_process_child",
        lambda *_args, **_kwargs: pytest.fail("duplicate child was launched"),
    )

    with pytest.raises(
        RuntimeError,
        match="identity marker was not reconciled",
    ):
        launch_supervised_child(
            SupervisedChildSpec(
                repo_root=repo,
                command=("python", "worker.py"),
                log_path=repo / "child.log",
                child_pid_path=pid_path,
                env={
                    SUPERVISED_CHILD_IDENTITY_PATH_ENV: str(identity_path),
                    SUPERVISED_CHILD_OWNER_SCOPE_ENV: json.dumps(
                        {"repo_root": str(repo)}
                    ),
                },
            )
        )
    assert identity_path.read_text(encoding="utf-8") == "unreconciled\n"
    assert not pid_path.exists()


def test_identity_protected_launcher_requires_dedicated_process_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    pid_path = repo / "state" / "child.pid"
    identity_path = repo / "state" / "child.identity.json"
    monkeypatch.setattr(
        supervisor_runtime,
        "launch_process_child",
        lambda *_args, **_kwargs: pytest.fail(
            "child was launched without a dedicated process session"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="requires a dedicated process session",
    ):
        launch_supervised_child(
            SupervisedChildSpec(
                repo_root=repo,
                command=("python", "worker.py"),
                log_path=repo / "child.log",
                child_pid_path=pid_path,
                start_new_session=False,
                env={
                    SUPERVISED_CHILD_IDENTITY_PATH_ENV: str(identity_path),
                    SUPERVISED_CHILD_OWNER_SCOPE_ENV: json.dumps(
                        {"repo_root": str(repo)}
                    ),
                },
            )
        )
    assert not pid_path.exists()


def test_shared_launcher_commits_identity_before_raw_pid_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    pid_path = repo / "state" / "child.pid"
    identity_path = repo / "state" / "child.identity.json"
    writes: list[Path] = []

    class FakeProcess:
        pid = 444

    monkeypatch.setattr(
        supervisor_runtime,
        "launch_process_child",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "read_process_birth",
        lambda pid: ProcessBirthIdentity(
            pid=int(pid),
            start_time_ticks=88,
            boot_id="boot-test",
            parent_pid=os.getpid(),
        ),
    )
    real_atomic_write = supervisor_runtime._write_bytes_atomic

    def capture_write(path: Path, content: bytes) -> None:
        writes.append(path)
        real_atomic_write(path, content)

    monkeypatch.setattr(supervisor_runtime, "_write_bytes_atomic", capture_write)

    child = launch_supervised_child(
        SupervisedChildSpec(
            repo_root=repo,
            command=("python", "worker.py"),
            log_path=repo / "child.log",
            child_pid_path=pid_path,
            env={
                SUPERVISED_CHILD_IDENTITY_PATH_ENV: str(identity_path),
                SUPERVISED_CHILD_OWNER_SCOPE_ENV: json.dumps(
                    {"repo_root": str(repo)}
                ),
            },
        )
    )

    assert child.pid == 444
    assert writes == [identity_path, pid_path]
    assert json.loads(identity_path.read_text(encoding="utf-8"))["process_birth"]["pid"] == 444
    assert pid_path.read_text(encoding="utf-8").strip() == "444"


def test_shared_launcher_reaps_direct_child_when_identity_capture_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    pid_path = repo / "state" / "child.pid"
    identity_path = repo / "state" / "child.identity.json"

    class FakeProcess:
        pid = 449

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.terminate_calls = 0

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminate_calls += 1
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is not None
            assert self.returncode is not None
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(
        supervisor_runtime,
        "launch_process_child",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "write_supervised_child_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("procfs identity unavailable")
        ),
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "read_process_birth",
        lambda _pid: pytest.fail("numeric PID fallback was unnecessary"),
    )

    with pytest.raises(RuntimeError, match="procfs identity unavailable"):
        launch_supervised_child(
            SupervisedChildSpec(
                repo_root=repo,
                command=("python", "worker.py"),
                log_path=repo / "child.log",
                child_pid_path=pid_path,
                env={
                    SUPERVISED_CHILD_IDENTITY_PATH_ENV: str(identity_path),
                    SUPERVISED_CHILD_OWNER_SCOPE_ENV: json.dumps(
                        {"repo_root": str(repo)}
                    ),
                },
            )
        )

    assert process.terminate_calls == 1
    assert process.returncode == -15


def test_shared_launcher_reaps_direct_child_on_system_exit_during_identity_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    pid_path = repo / "state" / "child.pid"
    identity_path = repo / "state" / "child.identity.json"

    class FakeProcess:
        pid = 450

        def __init__(self) -> None:
            self.returncode: int | None = None
            self.terminate_calls = 0

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminate_calls += 1
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is not None
            assert self.returncode is not None
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(
        supervisor_runtime,
        "launch_process_child",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "write_supervised_child_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SystemExit(143)),
    )

    with pytest.raises(SystemExit) as raised:
        launch_supervised_child(
            SupervisedChildSpec(
                repo_root=repo,
                command=("python", "worker.py"),
                log_path=repo / "child.log",
                child_pid_path=pid_path,
                env={
                    SUPERVISED_CHILD_IDENTITY_PATH_ENV: str(identity_path),
                    SUPERVISED_CHILD_OWNER_SCOPE_ENV: json.dumps(
                        {"repo_root": str(repo)}
                    ),
                },
            )
        )

    assert raised.value.code == 143
    assert process.terminate_calls == 1
    assert process.returncode == -15
    assert not pid_path.exists()
    assert not identity_path.exists()
    assert not pid_path.exists()
    assert not identity_path.exists()


def test_shared_adopt_or_launch_is_serialized_for_one_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    spec = SupervisedChildSpec(
        repo_root=repo,
        command=("python", "worker.py"),
        log_path=repo / "child.log",
        child_pid_path=repo / "state" / "child.pid",
    )
    child = SupervisedChild(
        pid=448,
        command=spec.command,
        log_path=spec.log_path,
        child_pid_path=spec.child_pid_path,
    )
    state: dict[str, SupervisedChild] = {}
    launch_count = 0

    def fake_adopt(_spec: SupervisedChildSpec) -> SupervisedChild | None:
        return state.get("child")

    def fake_launch(_spec: SupervisedChildSpec) -> SupervisedChild:
        nonlocal launch_count
        launch_count += 1
        time.sleep(0.05)
        state["child"] = child
        return child

    monkeypatch.setattr(
        supervisor_runtime,
        "adopt_supervised_child",
        fake_adopt,
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "launch_supervised_child",
        fake_launch,
    )
    lock_path = repo / "state" / "supervisor.lock"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                adopt_or_launch_supervised_child,
                spec,
                launch_lock_path=lock_path,
            )
            for _ in range(2)
        ]
        children = [future.result(timeout=2.0) for future in futures]

    assert [item.pid for item in children] == [448, 448]
    assert launch_count == 1


def test_shared_adoption_refuses_legacy_child_without_exact_proc_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    pid_path = repo / "state" / "child.pid"
    identity_path = repo / "state" / "child.identity.json"
    pid_path.parent.mkdir(parents=True)
    pid_path.write_text("445\n", encoding="utf-8")
    command = ("python", "worker.py", "--state-dir", "state")
    monkeypatch.setattr(supervisor_runtime, "pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        supervisor_runtime,
        "process_args",
        lambda _pid: "python worker.py --state-dir state --extra substring",
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "read_process_command_argv",
        lambda _pid: None,
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "write_supervised_child_identity",
        lambda *_args, **_kwargs: pytest.fail(
            "unproven legacy argv was migrated"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="legacy supervised child exact command is unproven",
    ):
        adopt_supervised_child(
            SupervisedChildSpec(
                repo_root=repo,
                command=command,
                log_path=repo / "child.log",
                child_pid_path=pid_path,
                env={
                    SUPERVISED_CHILD_IDENTITY_PATH_ENV: str(identity_path),
                    SUPERVISED_CHILD_OWNER_SCOPE_ENV: json.dumps(
                        {"repo_root": str(repo)}
                    ),
                },
            )
        )
    assert not identity_path.exists()


def test_shared_adoption_recovers_live_identity_without_pid_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    pid_path = repo / "state" / "child.pid"
    identity_path = repo / "state" / "child.identity.json"
    command = ("python", "worker.py", "--state-dir", "state")
    identity = SupervisedChildIdentity(
        process_birth=ProcessBirthIdentity(
            pid=445,
            start_time_ticks=102,
            boot_id="boot-test",
            parent_pid=17,
        ),
        command=command,
        owner_scope={"repo_root": str(repo)},
        created_at="2026-08-03T00:00:00+00:00",
    )
    identity_path.parent.mkdir(parents=True)
    identity_path.write_text(
        json.dumps(identity.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(supervisor_runtime, "pid_alive", lambda _pid: True)
    monkeypatch.setattr(
        supervisor_runtime,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.ALIVE,
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "read_process_command_argv",
        lambda _pid: command,
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "process_args",
        lambda _pid: " ".join(command),
    )

    child = adopt_supervised_child(
        SupervisedChildSpec(
            repo_root=repo,
            command=command,
            log_path=repo / "child.log",
            child_pid_path=pid_path,
            env={
                SUPERVISED_CHILD_IDENTITY_PATH_ENV: str(identity_path),
                SUPERVISED_CHILD_OWNER_SCOPE_ENV: json.dumps(
                    {"repo_root": str(repo)}
                ),
            },
        )
    )

    assert child is not None
    assert child.pid == 445
    assert pid_path.read_text(encoding="utf-8").strip() == "445"
    assert identity_path.exists()


def test_shared_adoption_preserves_dead_identity_from_another_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    pid_path = repo / "state" / "child.pid"
    identity_path = repo / "state" / "child.identity.json"
    command = ("python", "worker.py", "--state-dir", "state")
    identity = SupervisedChildIdentity(
        process_birth=ProcessBirthIdentity(
            pid=446,
            start_time_ticks=103,
            boot_id="boot-test",
            parent_pid=17,
        ),
        command=command,
        owner_scope={"repo_root": str(tmp_path / "other-repository")},
        created_at="2026-08-03T00:00:00+00:00",
    )
    identity_path.parent.mkdir(parents=True)
    identity_path.write_text(
        json.dumps(identity.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "supervised_child_identity_liveness",
        lambda _identity: pytest.fail("mismatched identity liveness was trusted"),
    )

    with pytest.raises(
        RuntimeError,
        match="orphaned supervised child ownership identity mismatch",
    ):
        adopt_supervised_child(
            SupervisedChildSpec(
                repo_root=repo,
                command=command,
                log_path=repo / "child.log",
                child_pid_path=pid_path,
                env={
                    SUPERVISED_CHILD_IDENTITY_PATH_ENV: str(identity_path),
                    SUPERVISED_CHILD_OWNER_SCOPE_ENV: json.dumps(
                        {"repo_root": str(repo)}
                    ),
                },
            )
        )

    assert identity_path.exists()
    assert not pid_path.exists()
    assert not tuple(identity_path.parent.glob("*.stale-child-identity-*"))


def test_shared_termination_does_not_signal_reused_identity_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_path = tmp_path / "child.pid"
    pid_path.write_text("446\n", encoding="utf-8")
    identity_path = supervisor_runtime.supervised_child_identity_path(pid_path)
    identity = SupervisedChildIdentity(
        process_birth=ProcessBirthIdentity(
            pid=446,
            start_time_ticks=101,
            boot_id="boot-test",
            parent_pid=17,
        ),
        command=("python", "worker.py"),
        owner_scope={"repo_root": str(tmp_path)},
        created_at="2026-08-03T00:00:00+00:00",
    )
    identity_path.write_text(
        json.dumps(identity.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.DEAD,
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail("reused child PID was signalled"),
    )

    stopped = terminate_supervised_child(
        SupervisedChild(
            pid=446,
            command=("python", "worker.py"),
            log_path=tmp_path / "child.log",
            child_pid_path=pid_path,
        )
    )

    assert stopped is False
    assert pid_path.exists()
    assert identity_path.exists()


def test_shared_termination_fails_closed_when_required_identity_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_path = tmp_path / "child.pid"
    pid_path.write_text("447\n", encoding="utf-8")
    identity_path = tmp_path / "custom-child-identity.json"
    monkeypatch.setattr(
        supervisor_runtime,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail(
            "identity-required child was signalled numerically"
        ),
    )

    stopped = terminate_supervised_child(
        SupervisedChild(
            pid=447,
            command=("python", "worker.py"),
            log_path=tmp_path / "child.log",
            child_pid_path=pid_path,
            identity_path=identity_path,
        )
    )

    assert stopped is False
    assert pid_path.exists()
    assert not identity_path.exists()


def test_supervisor_loop_preserves_markers_when_termination_is_unproven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    pid_path = state_dir / "child.pid"
    identity_path = state_dir / "child.identity.json"
    pid_path.parent.mkdir(parents=True)
    pid_path.write_text("450\n", encoding="utf-8")
    identity_path.write_text("unavailable\n", encoding="utf-8")
    spec = ManagedDaemonSpec(
        name="identity-required-daemon",
        schema="test.identity-required-daemon",
        repo_root=repo,
        daemon_dir=state_dir,
        runner=("python", "worker.py"),
        status_path=state_dir / "daemon-status.json",
        supervisor_status_path=state_dir / "supervisor-status.json",
        supervisor_pid_path=state_dir / "supervisor.pid",
        child_pid_path=pid_path,
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure-status.json",
        ensure_check_path=state_dir / "ensure-check.json",
        supervisor_lock_path=state_dir / "supervisor.lock",
    )
    child = SupervisedChild(
        pid=450,
        command=("python", "worker.py"),
        log_path=state_dir / "child.log",
        child_pid_path=pid_path,
        identity_path=identity_path,
    )
    monkeypatch.setattr(
        supervisor_loop_module,
        "adopt_or_launch_supervised_child",
        lambda _spec, **_kwargs: child,
    )
    monkeypatch.setattr(
        supervisor_loop_module,
        "_poll_child_exit",
        lambda _child: None,
    )
    monkeypatch.setattr(
        supervisor_loop_module,
        "terminate_supervised_child",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        supervisor_loop_module,
        "wait_for_child_exit",
        lambda _child: pytest.fail("unproven child exit was awaited"),
    )
    monkeypatch.setattr(
        supervisor_loop_module,
        "clear_child_pid_file",
        lambda _child: pytest.fail("ownership markers were cleared"),
    )
    loop = SupervisorLoop(
        SupervisorLoopConfig(
            spec=spec,
            command=child.command,
            log_prefix="child",
            heartbeat_seconds=0.01,
            poll_seconds=0.01,
            watchdog_startup_grace_seconds=0,
            max_restarts=1,
        ),
        watchdog_hook=lambda *_args: SupervisorLoopDecision.stop(
            "operator_stop"
        ),
        sleep=lambda _seconds: None,
    )

    result = loop.run()

    assert result.status == "termination_blocked"
    assert result.last_recycle_reason == (
        "supervised_child_termination_unproven"
    )
    assert pid_path.read_text(encoding="utf-8").strip() == "450"
    assert identity_path.read_text(encoding="utf-8") == "unavailable\n"


def test_adopted_child_exit_is_proven_before_identity_markers_are_cleared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_path = tmp_path / "child.pid"
    pid_path.write_text("451\n", encoding="utf-8")
    identity_path = tmp_path / "child.identity.json"
    identity = SupervisedChildIdentity(
        process_birth=ProcessBirthIdentity(
            pid=451,
            start_time_ticks=103,
            boot_id="boot-test",
            parent_pid=17,
        ),
        command=("python", "worker.py"),
        owner_scope={"repo_root": str(tmp_path)},
        created_at="2026-08-03T00:00:00+00:00",
    )
    identity_path.write_text(
        json.dumps(identity.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    persisted_identity = SupervisedChildIdentity.from_dict(identity.to_dict())
    assert persisted_identity is not None
    child = SupervisedChild(
        pid=451,
        command=identity.command,
        log_path=tmp_path / "child.log",
        child_pid_path=pid_path,
        identity_path=identity_path,
        identity_record_id=persisted_identity.record_id,
        identity_process_birth=persisted_identity.process_birth,
        owned_process_group_id=451,
    )
    monkeypatch.setattr(
        supervisor_runtime.os,
        "waitpid",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ChildProcessError()
        ),
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.DEAD,
    )

    assert wait_for_child_exit(child, poll_interval_seconds=0.01) == 0
    assert pid_path.exists()
    assert identity_path.exists()
    assert clear_child_pid_file(child) is True
    assert not pid_path.exists()
    assert not identity_path.exists()


def test_stale_child_handle_cannot_signal_replacement_identity_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid_path = tmp_path / "child.pid"
    pid_path.write_text("452\n", encoding="utf-8")
    identity_path = tmp_path / "child.identity.json"
    command = ("python", "worker.py")
    original = SupervisedChildIdentity.from_dict(
        SupervisedChildIdentity(
            process_birth=ProcessBirthIdentity(
                pid=452,
                start_time_ticks=104,
                boot_id="boot-test",
                parent_pid=17,
            ),
            command=command,
            owner_scope={"repo_root": str(tmp_path)},
            created_at="2026-08-03T00:00:00+00:00",
        ).to_dict()
    )
    replacement = SupervisedChildIdentity.from_dict(
        SupervisedChildIdentity(
            process_birth=ProcessBirthIdentity(
                pid=452,
                start_time_ticks=105,
                boot_id="boot-test",
                parent_pid=18,
            ),
            command=command,
            owner_scope={"repo_root": str(tmp_path)},
            created_at="2026-08-03T00:01:00+00:00",
        ).to_dict()
    )
    assert original is not None
    assert replacement is not None
    identity_path.write_text(
        json.dumps(replacement.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    child = SupervisedChild(
        pid=452,
        command=command,
        log_path=tmp_path / "child.log",
        child_pid_path=pid_path,
        identity_path=identity_path,
        identity_record_id=original.record_id,
        identity_process_birth=original.process_birth,
        owned_process_group_id=452,
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "supervised_child_identity_liveness",
        lambda _identity: OwnerLiveness.ALIVE,
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "read_process_command_argv",
        lambda _pid: command,
    )
    monkeypatch.setattr(
        supervisor_runtime,
        "terminate_pid_tree",
        lambda *_args, **_kwargs: pytest.fail(
            "replacement process generation was signalled"
        ),
    )

    assert terminate_supervised_child(child) is False
    assert clear_child_pid_file(child) is False
    assert pid_path.exists()
    assert identity_path.exists()


def test_run_forever_honors_unproven_managed_daemon_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    monkeypatch.setattr(supervisor, "ensure_event_log_file", lambda: {})
    monkeypatch.setattr(
        supervisor,
        "ensure_managed_daemon_pid_file",
        lambda: {
            "blocked": True,
            "reason": "managed_daemon_ownership_unproven",
        },
    )

    with pytest.raises(RuntimeError, match="managed_daemon_ownership_unproven"):
        supervisor._run_forever_loop()


def test_adopted_process_numeric_signal_methods_are_disabled() -> None:
    process = AdoptedManagedDaemonProcess(453)

    with pytest.raises(RuntimeError, match="ownership fence"):
        process.terminate()
    with pytest.raises(RuntimeError, match="ownership fence"):
        process.kill()
