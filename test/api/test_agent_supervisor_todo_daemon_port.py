from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.objective_daemon import (
    build_arg_parser,
    discovery_fingerprints,
    run_objective_daemon,
)
from ipfs_accelerate_py.agent_supervisor.bundle_supervisor import (
    build_arg_parser as build_bundle_arg_parser,
    plan_bundle_lanes,
    run_bundle_supervisor,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import parse_goal_heap
from ipfs_accelerate_py.agent_supervisor.objective_tracker import fibonacci_priority
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoTaskState,
    TodoImplementationDaemon,
    parse_args as parse_implementation_daemon_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    TodoImplementationSupervisor,
    TodoSupervisorConfig,
    parse_args as parse_implementation_supervisor_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import ManagedDaemonSpec, stop_daemon
from ipfs_accelerate_py.agent_supervisor.todo_daemon.runner import TodoDaemonRunner
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor import (
    SupervisorStatusContext,
    worktree_phase_worker_status,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import SupervisorLoop, SupervisorLoopConfig
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    RestartPolicy,
    SupervisedChildSpec,
    launch_supervised_child,
    terminate_supervised_child,
)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout.strip()


def _seed_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")

    objective_path = repo / "objective-heap.md"
    todo_path = repo / "todo.md"
    source = repo / "src" / "control_surface.py"
    source.parent.mkdir()
    source.write_text(
        """class VoiceCommandSurface:
    def route_click(self, event):
        return event
""",
        encoding="utf-8",
    )
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G010 Meta display control bridge

- Status: active
- Parent:
- Fib priority: 1
- Track: mobile
- Priority: P1
- Bundle: objective/mobile/meta-display
- Goal: Prove the glasses control bridge.
- Evidence: VoiceCommandSurface.route_click, missing_gesture_policy
- Outputs: src, tests
- Validation: test -f objective-heap.md
- Gap task: Add the missing gesture policy proof.
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    _git(repo, "add", "objective-heap.md", "todo.md", "src/control_surface.py")
    _git(repo, "commit", "-m", "seed objective heap")
    return repo, objective_path, todo_path


def _seed_parent_with_submodule(tmp_path: Path) -> tuple[Path, Path]:
    child_source = tmp_path / "child-source"
    child_source.mkdir()
    _git(child_source, "init")
    _git(child_source, "checkout", "-b", "main")
    _git(child_source, "config", "user.name", "Test User")
    _git(child_source, "config", "user.email", "test@example.invalid")
    (child_source / "child.txt").write_text("base\n", encoding="utf-8")
    _git(child_source, "add", "child.txt")
    _git(child_source, "commit", "-m", "child base")

    repo = tmp_path / "parent"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "-c", "protocol.file.allow=always", "submodule", "add", str(child_source), "libs/child")
    _git(repo, "add", ".gitmodules", "libs/child")
    _git(repo, "commit", "-m", "add child submodule")

    submodule = repo / "libs" / "child"
    _git(submodule, "checkout", "main")
    _git(submodule, "config", "user.name", "Test User")
    _git(submodule, "config", "user.email", "test@example.invalid")
    return repo, submodule


def test_todo_daemon_runtime_is_ported_to_accelerate_package():
    assert TodoDaemonRunner.__module__ == "ipfs_accelerate_py.agent_supervisor.todo_daemon.runner"
    assert TodoImplementationDaemon.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
    )
    assert TodoImplementationSupervisor.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor"
    )
    assert TodoSupervisorConfig.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor"
    )


def test_supervisor_runtime_repairs_marker_directories_before_launch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    state_dir.mkdir()
    log_path = state_dir / "child.log"
    pid_path = state_dir / "child.pid"
    latest_log_path = state_dir / "latest.log"
    for marker_path in (log_path, pid_path, latest_log_path):
        marker_path.mkdir()
        (marker_path / "fragment").write_text("old marker directory\n", encoding="utf-8")

    child = None
    try:
        child = launch_supervised_child(
            SupervisedChildSpec(
                repo_root=repo,
                command=(sys.executable, "-c", "import time; time.sleep(30)"),
                log_path=log_path,
                child_pid_path=pid_path,
                latest_log_path=latest_log_path,
            )
        )

        assert log_path.is_file()
        assert pid_path.read_text(encoding="utf-8").strip() == str(child.pid)
        assert latest_log_path.is_symlink()
        assert latest_log_path.readlink() == Path(log_path.name)
        for marker_name in ("child.log", "child.pid", "latest.log"):
            backups = list(state_dir.glob(f"{marker_name}.directory-backup-*"))
            assert len(backups) == 1
            assert (backups[0] / "fragment").exists()
    finally:
        if child is not None:
            terminate_supervised_child(child, grace_seconds=1)


def test_supervisor_status_write_repairs_directory_status_path(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    supervisor_status_path = state_dir / "supervisor_status.json"
    supervisor_status_path.mkdir(parents=True)
    (supervisor_status_path / "fragment").write_text("old status directory\n", encoding="utf-8")
    spec = ManagedDaemonSpec(
        name="test-daemon",
        schema="test.daemon",
        repo_root=repo,
        daemon_dir=state_dir,
        runner=(sys.executable, "-c", "pass"),
        status_path=state_dir / "daemon_status.json",
        supervisor_status_path=supervisor_status_path,
        supervisor_pid_path=state_dir / "supervisor.pid",
        child_pid_path=state_dir / "child.pid",
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure_status.json",
        ensure_check_path=state_dir / "ensure_check.json",
    )

    payload = SupervisorStatusContext(spec).write("running", run_id="run-1")

    assert supervisor_status_path.is_file()
    assert json.loads(supervisor_status_path.read_text(encoding="utf-8"))["status"] == "running"
    assert payload["run_id"] == "run-1"
    backups = list(state_dir.glob("supervisor_status.json.directory-backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "fragment").exists()


def test_stop_daemon_moves_directory_pid_markers(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    supervisor_pid_path = state_dir / "supervisor.pid"
    child_pid_path = state_dir / "child.pid"
    for marker_path in (supervisor_pid_path, child_pid_path):
        marker_path.mkdir(parents=True, exist_ok=True)
        (marker_path / "fragment").write_text("old pid directory\n", encoding="utf-8")
    spec = ManagedDaemonSpec(
        name="test-daemon",
        schema="test.daemon",
        repo_root=repo,
        daemon_dir=state_dir,
        runner=(sys.executable, "-c", "pass"),
        status_path=state_dir / "daemon_status.json",
        supervisor_status_path=state_dir / "supervisor_status.json",
        supervisor_pid_path=supervisor_pid_path,
        child_pid_path=child_pid_path,
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure_status.json",
        ensure_check_path=state_dir / "ensure_check.json",
    )

    result = stop_daemon(spec, cleanup_tmux=False)

    assert result.exit_code == 0
    assert result.payload["status"] == "not_running"
    for marker_name in ("supervisor.pid", "child.pid"):
        assert not (state_dir / marker_name).exists()
        backups = list(state_dir.glob(f"{marker_name}.directory-backup-*"))
        assert len(backups) == 1
        assert (backups[0] / "fragment").exists()


def test_supervisor_loop_retries_child_launch_failures(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    spec = ManagedDaemonSpec(
        name="test-daemon",
        schema="test.daemon",
        repo_root=repo,
        daemon_dir=state_dir,
        runner=("/definitely/missing/daemon",),
        status_path=state_dir / "daemon_status.json",
        supervisor_status_path=state_dir / "supervisor_status.json",
        supervisor_pid_path=state_dir / "supervisor.pid",
        child_pid_path=state_dir / "child.pid",
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure_status.json",
        ensure_check_path=state_dir / "ensure_check.json",
        latest_log_path=state_dir / "latest.log",
    )
    loop = SupervisorLoop(
        SupervisorLoopConfig(
            spec=spec,
            command=("/definitely/missing/daemon",),
            log_prefix="child",
            restart_policy=RestartPolicy(restart_backoff_seconds=0, fast_restart_backoff_seconds=0),
            heartbeat_seconds=0.01,
            poll_seconds=0.01,
            max_restarts=2,
        ),
        sleep=lambda _seconds: None,
    )

    result = loop.run()

    assert result.status == "max_restarts_reached"
    assert result.restart_count == 2
    assert result.last_exit_code == 127
    assert result.last_recycle_reason == "launch_failed"
    status = json.loads((state_dir / "supervisor_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "max_restarts_reached"
    assert status["last_recycle_reason"] == "launch_failed"
    assert status["last_exit_code"] == 127


def test_implementation_daemon_accepts_configured_submodule_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        llm_merge_resolver_command="true",
        llm_merge_resolver_timeout_seconds=5,
        worktree_submodule_paths=["packages/app,external/lib", "vendor/tools"],
    )

    assert daemon.worktree_submodule_paths == ("packages/app", "external/lib", "vendor/tools")
    assert daemon.llm_merge_resolver_command == "true"
    assert daemon.llm_merge_resolver_timeout_seconds == 5

    args = parse_implementation_daemon_args(
        [
            "--todo-path",
            str(repo / "todo.md"),
            "--llm-merge-resolver-command",
            "true",
            "--llm-merge-resolver-timeout-seconds",
            "5",
            "--worktree-submodule-path",
            "packages/app",
            "--worktree-submodule-path",
            "external/lib,vendor/tools",
        ]
    )
    assert args.worktree_submodule_path == ["packages/app", "external/lib,vendor/tools"]
    assert args.llm_merge_resolver_command == "true"
    assert args.llm_merge_resolver_timeout_seconds == 5


def test_implementation_daemon_runs_validation_non_interactively(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    log_path = repo / "validation.log"
    captured: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon.subprocess.run",
        fake_run,
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        repo_root=repo,
        implementation_timeout=1,
    )
    task = PortalTask(
        task_id="AUTO-001",
        title="Validate stdin handling",
        status="todo",
        completion="manual",
        priority="P1",
        track="ops",
        validation=["python3 -c 'import sys; sys.stdin.read()'"],
    )

    result = daemon._run_validation_commands(repo, task, log_path)

    assert result["passed"] is True
    assert captured["kwargs"]["stdin"] == subprocess.DEVNULL
    assert captured["kwargs"]["timeout"] == 1


def test_implementation_daemon_repairs_invalid_strategy_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Ready task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Ready task.
""",
        encoding="utf-8",
    )
    strategy_path = repo / "strategy.json"
    strategy_path.write_text("{not json", encoding="utf-8")
    events_path = repo / "events.jsonl"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=repo / "state.json",
        strategy_path=strategy_path,
        events_path=events_path,
        repo_root=repo,
        task_header_prefix="## AUTO-",
    )

    result = daemon.run_once()

    assert result["ready_count"] == 1
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == []
    assert strategy["last_strategy_repair_reason"] == "invalid_or_unreadable_strategy_file"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "strategy_file_repaired" for event in events)


def test_implementation_daemon_repairs_directory_strategy_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    state_dir = repo / "state"
    strategy_path = state_dir / "strategy.json"
    strategy_path.mkdir(parents=True)
    (strategy_path / "fragment").write_text("old strategy directory\n", encoding="utf-8")
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=strategy_path,
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
    )

    result = daemon.run_once()

    assert result["reason"] == "no_tasks_found"
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["last_strategy_repair_reason"] == "invalid_or_unreadable_strategy_file"
    backups = list(state_dir.glob("strategy.json.directory-backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "fragment").exists()
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "strategy_file_repaired" for event in events)


def test_implementation_daemon_repairs_malformed_state_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Ready task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Ready task.
""",
        encoding="utf-8",
    )
    state_path = repo / "state.json"
    state_path.write_text(json.dumps({"active_attempt": "not-an-int"}), encoding="utf-8")
    events_path = repo / "events.jsonl"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_path,
        strategy_path=repo / "strategy.json",
        events_path=events_path,
        repo_root=repo,
        task_header_prefix="## AUTO-",
    )

    result = daemon.run_once()

    assert result["state_file_repair"]["repaired"] is True
    assert result["state_file_repair"]["reason"] == "malformed_state_metadata"
    assert result["ready_count"] == 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["active_attempt"] == 0
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "state_file_repaired" for event in events)


def test_implementation_daemon_repairs_malformed_event_log(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Ready task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Ready task.
""",
        encoding="utf-8",
    )
    events_path = repo / "events.jsonl"
    events_path.write_text(
        json.dumps({"type": "seed", "task_id": "AUTO-000"}) + "\nnot json\n[]\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=events_path,
        repo_root=repo,
        task_header_prefix="## AUTO-",
    )

    result = daemon.run_once()

    assert result["event_log_repair"]["repaired"] is True
    assert result["event_log_repair"]["reason"] == "malformed_jsonl"
    assert result["event_log_repair"]["invalid_count"] == 2
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["type"] == "seed"
    assert any(event["type"] == "event_log_repaired" for event in events)
    quarantines = list(repo.glob("events.jsonl.invalid-jsonl-*"))
    assert len(quarantines) == 1


def test_implementation_daemon_moves_directory_lock_and_acquires_lock(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    lock_path = state_dir / "implementation.lock"
    lock_path.mkdir(parents=True)
    (lock_path / "fragment").write_text("not lock json\n", encoding="utf-8")
    events_path = state_dir / "events.jsonl"
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=events_path,
        repo_root=repo,
    )

    fd, reason, existing = daemon._try_acquire_lock(
        lock_path,
        lock_kind="implementation",
        owner_active=lambda _metadata: False,
    )
    if fd is not None:
        import os

        os.close(fd)

    assert reason == "acquired"
    assert existing is None
    assert lock_path.is_file()
    backups = list(state_dir.glob("implementation.lock.directory-backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "fragment").exists()
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    lock_events = [event for event in events if event["type"] == "implementation_lock_cleared"]
    assert len(lock_events) == 1
    assert lock_events[0]["moved_directory_path"] == str(backups[0])


def test_implementation_supervisor_repairs_invalid_strategy_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Ready task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Ready task.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    strategy_path = state_dir / "strategy.json"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text("{not json", encoding="utf-8")
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=strategy_path,
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
        )
    )

    result = supervisor.run_once()

    assert result["strategy_file_repair"]["repaired"] is True
    assert result["strategy_file_repair"]["reason"] == "invalid_or_unreadable_strategy_file"
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == []
    assert strategy["last_strategy_repair_reason"] == "invalid_or_unreadable_strategy_file"
    events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["type"] == "strategy_file_repaired" for event in events)


def test_implementation_supervisor_repairs_malformed_state_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Ready task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Ready task.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    state_path = state_dir / "task_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"active_attempt": "not-an-int"}), encoding="utf-8")
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_path,
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
        )
    )

    result = supervisor.run_once()

    assert result["state_file_repair"]["repaired"] is True
    assert result["state_file_repair"]["reason"] == "malformed_state_metadata"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["active_attempt"] == 0
    events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["type"] == "state_file_repaired" for event in events)


def test_implementation_supervisor_repairs_event_log_directory(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Ready task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Ready task.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    events_path = state_dir / "supervisor_events.jsonl"
    events_path.mkdir(parents=True)
    (events_path / "old-event-fragment").write_text("not json\n", encoding="utf-8")
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=events_path,
            state_dir=state_dir,
            repo_root=repo,
        )
    )

    result = supervisor.run_once()

    assert result["event_log_repair"]["repaired"] is True
    assert result["event_log_repair"]["reason"] == "event_path_was_directory"
    assert events_path.is_file()
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "event_log_repaired" for event in events)
    backup_path = Path(result["event_log_repair"]["backup_path"])
    assert backup_path.is_dir()
    assert (backup_path / "old-event-fragment").exists()


def test_implementation_supervisor_repairs_directory_managed_pid_path(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    pid_path = state_dir / "portal_managed_daemon.pid"
    pid_path.mkdir(parents=True)
    (pid_path / "fragment").write_text("not a pid\n", encoding="utf-8")
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
        )
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is True
    assert result["reason"] == "managed_pid_path_was_directory"
    assert not pid_path.exists()
    backup_path = Path(result["backup_path"])
    assert backup_path.is_dir()
    assert (backup_path / "fragment").exists()
    events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["type"] == "managed_daemon_pid_file_repaired"


def test_implementation_supervisor_repairs_stale_managed_pid_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    pid_path = state_dir / "portal_managed_daemon.pid"
    pid_path.parent.mkdir(parents=True)
    pid_path.write_text("999999999\n", encoding="utf-8")
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
        )
    )

    result = supervisor.ensure_managed_daemon_pid_file()

    assert result["repaired"] is True
    assert result["reason"] == "stale_managed_pid"
    assert not pid_path.exists()
    events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["type"] == "managed_daemon_pid_file_repaired"
    assert events[-1]["pid"] == 999999999


def test_implementation_supervisor_passes_configured_submodule_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon_script = repo / "custom_daemon.py"
    config = TodoSupervisorConfig(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        state_dir=repo / "state",
        implement=True,
        llm_merge_resolver_command="true",
        llm_merge_resolver_timeout_seconds=5,
        worktree_submodule_paths=("packages/app", "external/lib"),
        daemon_script_path=daemon_script,
    )
    supervisor = TodoImplementationSupervisor(config)

    command = supervisor._build_daemon_command()

    assert command[:2] == [sys.executable, str(daemon_script)]
    assert command.count("--worktree-submodule-path") == 2
    assert "packages/app" in command
    assert "external/lib" in command
    assert command[command.index("--llm-merge-resolver-command") + 1] == "true"
    assert command[command.index("--llm-merge-resolver-timeout-seconds") + 1] == "5"

    args = parse_implementation_supervisor_args(
        [
            "--implement",
            "--todo-path",
            str(repo / "todo.md"),
            "--llm-merge-resolver-command",
            "resolver-command",
            "--llm-merge-resolver-timeout-seconds",
            "7",
            "--daemon-script-path",
            str(repo / "daemon.py"),
            "--supervisor-script-path",
            str(repo / "supervisor.py"),
            "--worktree-submodule-path",
            "packages/app",
            "--worktree-submodule-path",
            "external/lib,vendor/tools",
            "--codebase-refill-scan",
            "--dependency-guardrail-discovery-dir",
            str(repo / "dependency-discovery"),
            "--dependency-guardrail-discovery-output-path",
            "dependency-discovery",
            "--dependency-guardrail-max-findings",
            "2",
            "--codebase-scan-min-open-tasks",
            "0",
            "--codebase-scan-depends-on",
            "AUTO-001,AUTO-002",
        ]
    )
    assert args.worktree_submodule_path == ["packages/app", "external/lib,vendor/tools"]
    assert args.llm_merge_resolver_command == "resolver-command"
    assert args.llm_merge_resolver_timeout_seconds == 7
    assert args.daemon_script_path == repo / "daemon.py"
    assert args.supervisor_script_path == repo / "supervisor.py"
    assert args.codebase_refill_scan is True
    assert args.codebase_scan_depends_on == ["AUTO-001,AUTO-002"]
    assert args.dependency_guardrail_discovery_dir == repo / "dependency-discovery"
    assert args.dependency_guardrail_discovery_output_path == "dependency-discovery"
    assert args.dependency_guardrail_max_findings == 2


def test_implementation_supervisor_does_not_recycle_active_merge_resolver(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = TodoSupervisorConfig(
        todo_path=repo / "todo.md",
        state_path=repo / "state.json",
        strategy_path=repo / "strategy.json",
        events_path=repo / "events.jsonl",
        state_dir=repo / "state",
        stale_seconds=60,
    )
    supervisor = TodoImplementationSupervisor(config)
    now = datetime.now(timezone.utc)
    state = TodoTaskState(
        active_task_id="AUTO-001",
        active_phase="merge_resolver",
        active_phase_detail="merge_conflict",
        heartbeat_at=now.isoformat(),
        last_progress_at="2000-01-01T00:00:00+00:00",
        ready_count=1,
    )

    stuck, reason = supervisor.is_stuck(state, now_ts=now.timestamp())

    assert stuck is False
    assert reason == ""


def test_implementation_supervisor_repairs_stale_merge_resolver_without_worker(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    state_dir.mkdir()
    state_path = state_dir / "task_state.json"
    now = datetime.now(timezone.utc)
    old = now.replace(year=2000)
    TodoTaskState(
        active_task_id="AUTO-001",
        active_task_title="Stale merge resolver",
        active_task_track="ops",
        active_task_started_at=old.isoformat(),
        active_attempt=1,
        active_phase="merge_resolver",
        active_phase_started_at=old.isoformat(),
        active_phase_detail="merge_conflict",
        implementation_in_progress=True,
        last_implementation_task_id="AUTO-001",
        last_implementation_started_at=old.isoformat(),
        heartbeat_at=now.isoformat(),
        last_progress_at=now.isoformat(),
        ready_count=1,
    ).save(state_path)
    config = TodoSupervisorConfig(
        todo_path=repo / "todo.md",
        state_path=state_path,
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        check_interval=1,
        stale_seconds=3600,
    )

    result = TodoImplementationSupervisor(config).run_once()

    assert result["stuck"] is True
    assert "merge_resolver stalled" in result["reason"]
    assert result["state_repair"]["repaired"] is True
    repaired_state = TodoTaskState.load(state_path)
    assert repaired_state.active_task_id == ""
    events = [json.loads(line) for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "merge_phase_without_worker" for event in events)


def test_supervisor_worker_watchdog_detects_active_merge_resolver_without_worker(tmp_path):
    now = datetime.now(timezone.utc)
    old = now.replace(year=2000)

    status = worktree_phase_worker_status(
        {
            "active_phase": "merge_resolver",
            "active_phase_started_at": old.isoformat(),
        },
        daemon_pid=999999999,
        threshold_seconds=60,
        now=now,
    )

    assert status["required"] is True
    assert status["phase"] == "merge_resolver"
    assert status["active_worker_count"] == 0
    assert status["stalled_without_active_worker"] is True


def test_implementation_supervisor_configures_worker_stall_watchdog(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    config = TodoSupervisorConfig(
        todo_path=repo / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        implementation_log_stall_seconds=42,
    )

    loop_config = TodoImplementationSupervisor(config).build_supervisor_loop_config()

    assert loop_config.status_static_fields["worktree_no_child_stall_seconds"] == 42


def test_implementation_supervisor_repairs_stale_active_state_after_rewrite(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    state_dir.mkdir()
    state_path = state_dir / "task_state.json"
    stale_timestamp = "2000-01-01T00:00:00+00:00"
    TodoTaskState(
        active_task_id="AUTO-001",
        active_task_title="Stale active task",
        active_task_track="ops",
        active_task_started_at=stale_timestamp,
        active_attempt=2,
        active_phase="implementing",
        active_phase_started_at=stale_timestamp,
        active_phase_detail="stale worker",
        active_log_path=str(state_dir / "implementation.log"),
        active_worktree_path=str(repo / "worktrees" / "auto-001"),
        active_branch="implementation/auto-001",
        implementation_in_progress=False,
        heartbeat_at=stale_timestamp,
        last_progress_at=stale_timestamp,
        ready_count=1,
    ).save(state_path)
    config = TodoSupervisorConfig(
        todo_path=repo / "todo.md",
        state_path=state_path,
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "supervisor_events.jsonl",
        state_dir=state_dir,
        stale_seconds=60,
    )

    result = TodoImplementationSupervisor(config).run_once()

    assert result["stuck"] is True
    assert result["state_repair"]["repaired"] is True
    assert result["state_repair"]["active_task_id"] == "AUTO-001"
    repaired_state = TodoTaskState.load(state_path)
    assert repaired_state.active_task_id == ""
    assert repaired_state.implementation_in_progress is False
    assert repaired_state.active_worktree_path == ""
    strategy = json.loads((state_dir / "strategy.json").read_text(encoding="utf-8"))
    assert strategy["deprioritized_tasks"] == ["AUTO-001"]
    events = [json.loads(line) for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "blocked_progress_state_repaired" for event in events)


def test_implementation_daemon_keeps_alive_on_empty_todo_board(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text("# Agent Todos\n", encoding="utf-8")
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
    )

    result = daemon.run_once()

    assert result["reason"] == "no_tasks_found"
    assert result["task_count"] == 0
    state = TodoTaskState.load(state_dir / "task_state.json")
    assert state.active_task_id == ""
    assert state.ready_count == 0
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "daemon_no_tasks"


def test_implementation_daemon_records_unreadable_todo_text(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_bytes(b"\xff\xfe\x00")
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
    )

    result = daemon.run_once()

    assert result["reason"] == "todo_read_failed"
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "daemon_no_tasks"
    assert events[-1]["reason"] == "todo_read_failed"


def test_implementation_daemon_records_non_ephemeral_setup_exception(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## ACCEL-001 Record command failure

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs:
- Validation:
- Acceptance: The daemon records setup failures as durable events.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implement=True,
        implementation_command="missing-implementation-command-for-test",
        use_ephemeral_worktree=False,
    )

    result = daemon.run_once()

    implementation = result["implementation_result"]
    assert implementation["returncode"] == 1
    assert implementation["exception_result"]["exception_type"] == "FileNotFoundError"
    assert implementation["exception_result"]["phase"] == "implementing"
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "implementation_exception" for event in events)
    assert events[-1]["type"] == "daemon_pass"


def test_implementation_supervisor_creates_missing_todo_before_refill(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    source = repo / "src" / "runtime_bridge.py"
    source.parent.mkdir()
    source.write_text(
        """class RuntimeBridge:
    def dispatch(self, request):
        return request
""",
        encoding="utf-8",
    )
    objective_path = repo / "objective-heap.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G001 Runtime bridge proof

- Status: active
- Parent:
- Fib priority: 1
- Track: runtime
- Priority: P1
- Goal: Prove that runtime bridge dispatch supports virtual AI OS execution.
- Evidence: RuntimeBridge.dispatch, missing_runtime_contract
- Outputs: src/runtime_bridge.py, tests
- Validation: test -f objective-heap.md
- Gap task: Add the missing runtime contract proof.
""",
        encoding="utf-8",
    )
    _git(repo, "add", "objective-heap.md", "src/runtime_bridge.py")
    _git(repo, "commit", "-m", "seed objective heap")
    todo_path = repo / "missing-todo.md"
    state_dir = repo / "state"

    result = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            task_prefix="## ACCEL-",
            objective_refill_enabled=True,
            objective_path=objective_path,
            objective_discovery_dir=repo / "discovery",
            objective_bundle_dir=repo / "bundles",
            objective_dataset_dir=repo / "datasets",
            objective_graph_path=repo / "objective-graph.json",
            objective_scan_min_open_tasks=0,
            objective_scan_max_findings=1,
            objective_scan_cooldown_seconds=21600,
            objective_persist_ast_dataset=False,
        )
    ).run_once()

    assert result["todo_board_repair"]["created"] is True
    assert result["objective_refill_count"] == 1
    assert todo_path.exists()
    assert "## ACCEL-001 Close objective gap" in todo_path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "todo_board_created" for event in events)


def test_implementation_supervisor_repairs_directory_todo_before_refill(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    objective_path = repo / "objective-heap.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G001 Directory todo repair

- Status: active
- Parent:
- Fib priority: 1
- Track: runtime
- Priority: P1
- Goal: Prove that todo board directory repair bootstraps refill.
- Evidence: missing_directory_todo_repair
- Outputs: src/runtime_bridge.py, tests
- Validation: test -f objective-heap.md
- Gap task: Add the directory todo repair proof.
""",
        encoding="utf-8",
    )
    (repo / "src").mkdir()
    (repo / "src" / "runtime_bridge.py").write_text("class RuntimeBridge: pass\n", encoding="utf-8")
    _git(repo, "add", "objective-heap.md", "src/runtime_bridge.py")
    _git(repo, "commit", "-m", "seed objective heap")
    todo_path = repo / "todo.md"
    todo_path.mkdir()
    (todo_path / "fragment").write_text("not a todo board\n", encoding="utf-8")
    state_dir = repo / "state"

    result = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            task_prefix="## ACCEL-",
            objective_refill_enabled=True,
            objective_path=objective_path,
            objective_discovery_dir=repo / "discovery",
            objective_bundle_dir=repo / "bundles",
            objective_dataset_dir=repo / "datasets",
            objective_graph_path=repo / "objective-graph.json",
            objective_scan_min_open_tasks=0,
            objective_scan_max_findings=1,
            objective_scan_cooldown_seconds=21600,
            objective_persist_ast_dataset=False,
        )
    ).run_once()

    assert result["todo_board_repair"]["repaired"] is True
    assert result["todo_board_repair"]["reason"] == "todo_path_was_directory"
    assert todo_path.is_file()
    assert "## ACCEL-001 Close objective gap" in todo_path.read_text(encoding="utf-8")
    backup_path = Path(result["todo_board_repair"]["backup_path"])
    assert backup_path.is_dir()
    assert (backup_path / "fragment").exists()
    events = [json.loads(line) for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "todo_board_repaired" for event in events)


def test_implementation_supervisor_records_dependency_guardrail(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Waiting on missing dependency

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: AUTO-999
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: This task should become ready when dependencies are valid.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            task_prefix="## AUTO-",
            dependency_guardrail_discovery_dir=repo / "dependency-discovery",
            dependency_guardrail_max_findings=1,
        )
    )

    result = supervisor.run_once()

    assert result["dependency_guardrail_count"] == 1
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## AUTO-002 Resolve dependency guardrail for AUTO-001" in todo_text
    strategy = json.loads((state_dir / "strategy.json").read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-001"]
    assert strategy["dependency_guardrail_findings"][0]["missing_dependencies"] == ["AUTO-999"]
    supervisor_events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["type"] == "dependency_guardrail" for event in supervisor_events)


def test_implementation_supervisor_releases_completed_guardrail_block(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Original blocked task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Original task should resume after repair.

## AUTO-002 Resolve implementation retry-budget failure for AUTO-001

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: discovery
- Validation: test -f todo.md
- Acceptance: Repair task completed.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    strategy_path = state_dir / "strategy.json"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        json.dumps(
            {
                "blocked_tasks": ["AUTO-001"],
                "retry_budget_findings": [
                    {
                        "source_task_id": "AUTO-001",
                        "follow_up_task_id": "AUTO-002",
                        "failure_kind": "implementation",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=strategy_path,
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            task_prefix="## AUTO-",
        )
    )

    result = supervisor.run_once()

    assert result["guardrail_unblock_count"] == 1
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == []
    assert strategy["guardrail_unblock_releases"][0]["source_task_id"] == "AUTO-001"
    supervisor_events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["type"] == "guardrail_blocks_released" for event in supervisor_events)


def test_implementation_supervisor_releases_stale_strategy_blocks(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Completed source

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Already complete.

## AUTO-002 Still blocked source

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Remains blocked.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    strategy_path = state_dir / "strategy.json"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        json.dumps({"blocked_tasks": ["AUTO-001", "AUTO-999", "AUTO-002", "AUTO-002"]}),
        encoding="utf-8",
    )
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=strategy_path,
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            task_prefix="## AUTO-",
        )
    )

    result = supervisor.run_once()

    assert result["guardrail_unblock_count"] == 3
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-002"]
    reasons = {release["reason"] for release in strategy["guardrail_unblock_releases"]}
    assert reasons == {"duplicate_strategy_block", "missing_task", "source_completed"}
    supervisor_events = [
        json.loads(line)
        for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["type"] == "guardrail_blocks_released" for event in supervisor_events)


def test_implementation_daemon_records_worktree_setup_exception(tmp_path):
    repo = tmp_path / "not-a-git-repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## ACCEL-001 Record setup failure

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs:
- Validation:
- Acceptance: The daemon records setup failures as durable events.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implement=True,
        implementation_command="true",
        use_ephemeral_worktree=True,
        worktree_root=repo / "worktrees",
    )

    result = daemon.run_once()

    implementation = result["implementation_result"]
    assert implementation["returncode"] == 1
    assert implementation["exception_result"]["exception_type"] == "RuntimeError"
    events = [json.loads(line) for line in (state_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "implementation_exception" for event in events)
    assert events[-1]["type"] == "daemon_pass"


def test_implementation_daemon_records_merge_reconcile_exception(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    event = {
        "task_id": "ACCEL-002",
        "attempt": 3,
        "branch": "implementation/accel-002",
        "implementation_commit": "abc123",
        "title": "Recover failed merge",
    }

    daemon._failed_merge_candidates = lambda skip_task_ids=None: [event]  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: False  # type: ignore[method-assign]
    daemon._git_ref_exists = lambda ref: True  # type: ignore[method-assign]

    def raise_merge_error(*args, **kwargs):
        raise RuntimeError("merge workspace unavailable")

    daemon._merge_branch_to_main = raise_merge_error  # type: ignore[method-assign]

    result = daemon._reconcile_failed_merges()

    assert result == [
        {
            "task_id": "ACCEL-002",
            "attempt": 3,
            "branch": "implementation/accel-002",
            "implementation_commit": "abc123",
            "resolved": False,
            "reason": "merge_reconcile_exception",
            "exception_type": "RuntimeError",
            "error": "merge workspace unavailable",
        }
    ]
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "merge_reconcile_exception"


def test_implementation_daemon_reconciles_merge_lock_deferrals(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    event = {
        "type": "implementation_finished",
        "task_id": "ACCEL-003",
        "attempt": 2,
        "branch": "implementation/accel-003",
        "implementation_commit": "def456",
        "validation_result": {"attempted": True, "passed": True},
        "merge_result": {
            "attempted": False,
            "merged": False,
            "reason": "lock_exists",
            "branch": "implementation/accel-003",
            "lock_owner_pid": 12345,
        },
    }

    daemon._iter_events = lambda: [event]  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: False  # type: ignore[method-assign]

    assert daemon._failed_merge_candidates() == [event]
    assert daemon._unresolved_merge_failures_by_task()["ACCEL-003"] == event


def test_implementation_daemon_retries_cleanup_failures_for_already_merged_branch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )
    event = {
        "type": "implementation_finished",
        "task_id": "ACCEL-004",
        "attempt": 1,
        "branch": "implementation/accel-004",
        "implementation_commit": "abc123",
        "worktree_path": str(repo / "worktrees" / "accel-004"),
        "validation_result": {"attempted": True, "passed": True},
        "merge_result": {
            "attempted": True,
            "merged": False,
            "reason": "cleanup_failed",
            "branch": "implementation/accel-004",
        },
    }

    daemon._failed_merge_candidates = lambda skip_task_ids=None: [event]  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: True  # type: ignore[method-assign]
    daemon._cleanup_merged_worktree = lambda worktree_path, branch: {  # type: ignore[method-assign]
        "cleaned": False,
        "reason": "worktree_remove_failed",
        "branch": branch,
        "worktree_path": str(worktree_path),
    }

    result = daemon._reconcile_failed_merges()

    assert result[0]["resolved"] is False
    assert result[0]["reason"] == "cleanup_retry_failed"
    assert result[0]["cleanup_result"]["reason"] == "worktree_remove_failed"
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "merge_reconciled"
    assert events[-1]["resolved"] is False


def test_implementation_supervisor_refills_drained_codebase_backlog(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    source = repo / "src" / "runtime.py"
    source.parent.mkdir()
    source.write_text(
        """def route_request(request):
    # TODO: inspect drained supervisor refill
    return request
""",
        encoding="utf-8",
    )
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Completed seed

- Status: completed
- Completion: manual
- Priority: P2
- Track: ops
- Depends on:
- Outputs: README.md
- Validation: test -f README.md
- Acceptance: Seed task.
""",
        encoding="utf-8",
    )
    _git(repo, "add", "todo.md", "src/runtime.py")
    _git(repo, "commit", "-m", "seed drained backlog")
    state_dir = repo / "state"
    config = TodoSupervisorConfig(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        task_prefix="## AUTO-",
        codebase_refill_enabled=True,
        codebase_scan_discovery_dir=repo / "discovery",
        codebase_scan_min_open_tasks=0,
        codebase_scan_max_findings=1,
        codebase_scan_cooldown_seconds=21600,
    )

    result = TodoImplementationSupervisor(config).run_once()

    assert result["codebase_refill_count"] == 1
    assert "## AUTO-002 Resolve code annotation in src/runtime.py:2" in todo_path.read_text(encoding="utf-8")
    strategy = json.loads((state_dir / "strategy.json").read_text(encoding="utf-8"))
    assert strategy["last_codebase_scan_mode"] == "drained_exhaustive"
    assert strategy["last_drained_codebase_scan_task_count"] == 1
    assert list((repo / "discovery").glob("*-auto-002-codebase-scan-*.md"))


def test_implementation_supervisor_records_codebase_refill_failures(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor import backlog_refinery

    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Completed seed

- Status: completed
- Completion: manual
- Priority: P2
- Track: ops
- Depends on:
- Outputs: README.md
- Validation: test -f README.md
- Acceptance: Seed task.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"

    def fail_refill(**_kwargs):
        raise FileNotFoundError("vanished worktree")

    monkeypatch.setattr(backlog_refinery, "record_codebase_scan_findings", fail_refill)
    config = TodoSupervisorConfig(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        task_prefix="## AUTO-",
        codebase_refill_enabled=True,
        codebase_scan_discovery_dir=repo / "discovery",
        codebase_scan_min_open_tasks=0,
        codebase_scan_max_findings=1,
        codebase_scan_cooldown_seconds=21600,
    )

    result = TodoImplementationSupervisor(config).run_once()

    assert result["codebase_refill_count"] == 0
    events = [json.loads(line) for line in (state_dir / "supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()]
    failure = [event for event in events if event["type"] == "codebase_refill_failed"][-1]
    assert failure["error_type"] == "FileNotFoundError"
    assert "vanished worktree" in failure["error"]


def test_implementation_supervisor_records_retry_budget_guardrail(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Fix implementation runtime

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f src/runtime.py
- Acceptance: Fix the repeated implementation blocker.
""",
        encoding="utf-8",
    )
    state_dir = repo / "state"
    events_path = state_dir / "auto_events.jsonl"
    events_path.parent.mkdir(parents=True)
    failure = {
        "type": "implementation_finished",
        "task_id": "AUTO-001",
        "attempt": 1,
        "returncode": 124,
        "validation_result": {"attempted": False, "passed": True},
        "merge_result": {"attempted": False, "merged": False, "reason": "not_attempted"},
        "log_path": "state/implementation_logs/auto-001-attempt-1.log",
    }
    events_path.write_text(json.dumps(failure) + "\n" + json.dumps({**failure, "attempt": 2}) + "\n", encoding="utf-8")
    config = TodoSupervisorConfig(
        todo_path=todo_path,
        state_path=state_dir / "auto_task_state.json",
        strategy_path=state_dir / "auto_strategy.json",
        events_path=state_dir / "auto_supervisor_events.jsonl",
        state_dir=state_dir,
        repo_root=repo,
        task_prefix="## AUTO-",
        state_prefix="auto",
        implementation_retry_budget=2,
        validation_retry_budget=0,
        merge_retry_budget=0,
        retry_budget_discovery_dir=repo / "discovery",
    )

    result = TodoImplementationSupervisor(config).run_once()

    assert result["retry_budget_count"] == 1
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## AUTO-002 Resolve implementation retry-budget failure for AUTO-001" in todo_text
    strategy = json.loads((state_dir / "auto_strategy.json").read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-001"]
    supervisor_events = [
        json.loads(line)
        for line in (state_dir / "auto_supervisor_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["type"] == "retry_budget_guardrail" for event in supervisor_events)


def test_implementation_supervisor_refines_objective_goals_before_generating_todos(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    source = repo / "src" / "runtime_bridge.py"
    source.parent.mkdir()
    source.write_text(
        """class RuntimeBridge:
    def dispatch(self, request):
        return request
""",
        encoding="utf-8",
    )
    objective_path = repo / "objective-heap.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G001 Runtime bridge proof

- Status: active
- Parent:
- Fib priority: 1
- Track: runtime
- Priority: P1
- Goal: Prove that runtime bridge dispatch supports virtual AI OS execution.
- Evidence: RuntimeBridge.dispatch, missing_runtime_contract
- Outputs: src/runtime_bridge.py, tests
- Validation: test -f objective-heap.md
- Gap task: Add the missing runtime contract proof.
""",
        encoding="utf-8",
    )
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## ACCEL-001 Completed seed

- Status: completed
- Completion: manual
- Priority: P2
- Track: ops
- Depends on:
- Outputs: README.md
- Validation: test -f objective-heap.md
- Acceptance: Seed task.
""",
        encoding="utf-8",
    )
    _git(repo, "add", "todo.md", "objective-heap.md", "src/runtime_bridge.py")
    _git(repo, "commit", "-m", "seed objective refill")
    state_dir = repo / "state"
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=todo_path,
            state_path=state_dir / "task_state.json",
            strategy_path=state_dir / "strategy.json",
            events_path=state_dir / "supervisor_events.jsonl",
            state_dir=state_dir,
            repo_root=repo,
            task_prefix="## ACCEL-",
            objective_refill_enabled=True,
            objective_path=objective_path,
            objective_discovery_dir=repo / "discovery",
            objective_bundle_dir=repo / "bundles",
            objective_dataset_dir=repo / "datasets",
            objective_graph_path=repo / "objective-graph.json",
            objective_scan_min_open_tasks=0,
            objective_scan_max_findings=1,
            objective_scan_cooldown_seconds=21600,
            objective_max_refinement_children=1,
            objective_max_refinement_depth=3,
            objective_persist_ast_dataset=False,
        )
    )

    result = supervisor.run_once()

    assert result["objective_refined_goal_count"] == 1
    assert result["objective_refill_count"] == 1
    objective_text = objective_path.read_text(encoding="utf-8")
    assert "## VAIOS-G002 Prove missing_runtime_contract for Runtime bridge proof" in objective_text
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## ACCEL-002 Close objective gap" in todo_text
    assert "Graph parents: VAIOS-G001" in todo_text
    assert (repo / "objective-graph.json").exists()
    assert (repo / "bundles" / "index.json").exists()
    strategy = json.loads((state_dir / "strategy.json").read_text(encoding="utf-8"))
    assert strategy["last_objective_goal_scan_mode"] == "drained_exhaustive"
    assert strategy["last_drained_objective_goal_scan_task_count"] == 1
    assert strategy["last_objective_refined_goal_ids"] == ["VAIOS-G002"]
    assert strategy["last_objective_generated_task_ids"] == ["ACCEL-002"]


def test_objective_daemon_generates_todos_bundles_and_dataset(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
    dataset_dir = repo / "data" / "agent_supervisor" / "objective_datasets"
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--discovery-dir",
            str(discovery_dir),
            "--bundle-dir",
            str(bundle_dir),
            "--dataset-dir",
            str(dataset_dir),
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "1",
        ]
    )

    payload = run_objective_daemon(args)

    assert payload["schema"] == "ipfs_accelerate_py.agent_supervisor.objective_daemon"
    assert payload["generated_count"] == 1
    assert payload["task_ids"] == ["ACCEL-001"]
    assert "## ACCEL-001 Close objective gap" in todo_path.read_text(encoding="utf-8")
    assert (bundle_dir / "objective-mobile-meta-display.todo.md").exists()
    assert (bundle_dir / "index.json").exists()
    manifest = dataset_dir / "accel-objective-ast.manifest.json"
    assert manifest.exists()
    assert json.loads(manifest.read_text(encoding="utf-8"))["row_count"] >= 2
    assert discovery_fingerprints(discovery_dir)


def test_objective_daemon_suppresses_existing_discovery_fingerprint(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
    args = argparse.Namespace(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        dataset_dir=repo / "data" / "agent_supervisor" / "objective_datasets",
        task_prefix="ACCEL-",
        depends_on=[],
        seen_fingerprint=[],
        repeat_existing=False,
        max_findings=1,
        no_persist_ast_dataset=True,
        submit_bundles=False,
        queue_path=None,
        queue_task_type="codex.todo_bundle",
        queue_model_name="codex",
        log_level="INFO",
    )

    first = run_objective_daemon(args)
    second = run_objective_daemon(args)

    assert first["generated_count"] == 1
    assert second["generated_count"] == 0
    assert todo_path.read_text(encoding="utf-8").count("## ACCEL-001 ") == 1


def test_objective_daemon_creates_tracking_document_and_graph(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    readme = repo / "README.md"
    readme.write_text("# Repo\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "seed")

    objective_path = repo / "docs" / "objective-heap.md"
    todo_path = repo / "docs" / "todo.md"
    graph_path = repo / "data" / "agent_supervisor" / "objective_graph.json"
    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--graph-path",
            str(graph_path),
            "--ensure-tracking-document",
            "--ultimate-goal",
            "Operate as a virtual AI OS for a Meta glasses remote display.",
            "--root-evidence",
            "missing_meta_display_bridge",
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "1",
            "--no-persist-ast-dataset",
        ]
    )

    payload = run_objective_daemon(args)

    assert payload["tracking_document_created"] is True
    assert payload["ensured_goal_ids"] == ["OBJ-G000"]
    assert payload["objective_goal_count"] == 1
    assert graph_path.exists()
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert graph["graph"]["roots"] == ["OBJ-G000"]
    assert "missing_meta_display_bridge" in objective_path.read_text(encoding="utf-8")
    assert "## ACCEL-001 Close objective gap" in todo_path.read_text(encoding="utf-8")


def test_objective_daemon_refines_missing_evidence_into_child_goals(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    args = build_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--refine-objective-heap",
            "--max-refinement-children",
            "1",
            "--task-prefix",
            "ACCEL-",
            "--max-findings",
            "2",
            "--no-persist-ast-dataset",
        ]
    )

    payload = run_objective_daemon(args)
    goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    refined = [goal for goal in goals if goal.goal_id in payload["refined_goal_ids"]]

    assert payload["refined_goal_ids"] == ["VAIOS-G011"]
    assert payload["generated_count"] == 1
    assert len(refined) == 1
    assert refined[0].parent_goal_ids == ["VAIOS-G010"]
    assert refined[0].required_evidence == ["missing_gesture_policy"]
    assert refined[0].fields["fib_priority"] == str(fibonacci_priority(1, 0))
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "Prove missing_gesture_policy for Meta display control bridge" in todo_text


def test_bundle_supervisor_plans_isolated_lanes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    index_path = repo / "data" / "agent_supervisor" / "objective_bundles" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(
        json.dumps(
            {
                "source_todo": "docs/main.todo.md",
                "bundles": {
                    "objective/runtime/kernel": {
                        "shard_path": "data/agent_supervisor/objective_bundles/runtime.todo.md",
                        "parallel_lane": "objective/runtime/kernel",
                        "conflict_policy": "bundle-local edits",
                        "tasks": [{"task_id": "ACCEL-001"}],
                    },
                    "objective/mobile/meta-display": {
                        "shard_path": "data/agent_supervisor/objective_bundles/mobile.todo.md",
                        "parallel_lane": "objective/mobile/meta-display",
                        "conflict_policy": "invoke resolver",
                        "tasks": [{"task_id": "ACCEL-002"}],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    lanes = plan_bundle_lanes(
        bundle_index_path=index_path,
        repo_root=repo,
        state_root=repo / "state",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        task_prefix="ACCEL-",
        implement=True,
        implementation_command="codex exec --full-auto",
        max_lanes=None,
    )

    assert [lane.bundle_key for lane in lanes] == [
        "objective/mobile/meta-display",
        "objective/runtime/kernel",
    ]
    assert lanes[0].todo_path == repo / "data/agent_supervisor/objective_bundles/mobile.todo.md"
    assert lanes[0].state_dir != lanes[1].state_dir
    assert lanes[0].worktree_root != lanes[1].worktree_root
    assert lanes[0].task_ids == ["ACCEL-002"]
    assert "--implement" in lanes[0].command
    assert "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor" in lanes[0].command
    assert "--implementation-command" in lanes[0].command


def test_bundle_supervisor_writes_manifest_without_starting_lanes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    index_path = repo / "objective_bundles" / "index.json"
    index_path.parent.mkdir()
    index_path.write_text(
        json.dumps(
            {
                "source_todo": "docs/main.todo.md",
                "bundles": {
                    "objective/ops/root": {
                        "shard_path": "objective_bundles/root.todo.md",
                        "parallel_lane": "objective/ops/root",
                        "conflict_policy": "use merge resolver",
                        "tasks": [{"task_id": "ACCEL-009"}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = repo / "manifest.json"
    args = build_bundle_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--bundle-index-path",
            str(index_path),
            "--manifest-path",
            str(manifest_path),
            "--task-prefix",
            "ACCEL-",
            "--no-implement",
        ]
    )

    payload = run_bundle_supervisor(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["planned_count"] == 1
    assert payload["started_count"] == 0
    assert manifest["lanes"][0]["bundle_key"] == "objective/ops/root"
    assert manifest["lanes"][0]["todo_path"] == "objective_bundles/root.todo.md"
    assert "--no-implement" in manifest["lanes"][0]["command"]


def test_implementation_daemon_invokes_configured_llm_merge_resolver(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    marker = repo / "README.md"
    marker.write_text("# Repo\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "seed")

    capture_path = tmp_path / "resolver-prompt.txt"
    resolver_script = tmp_path / "resolver.py"
    resolver_script.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        llm_merge_resolver_command=(
            f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))} {shlex.quote(str(capture_path))}"
        ),
        llm_merge_resolver_timeout_seconds=5,
    )
    result = daemon._invoke_llm_merge_resolver_for_failed_merge(
        workspace=repo,
        task=PortalTask(
            task_id="ACCEL-999",
            title="Resolve semantic merge",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        attempt=2,
        branch_name="implementation/accel-999",
        target_branch="main",
        merge_command=["git", "merge", "implementation/accel-999"],
        merge_stdout="",
        merge_stderr="CONFLICT (content): Merge conflict",
    )

    assert result["applied"] is True
    assert result["llm_returncode"] == 0
    prompt = capture_path.read_text(encoding="utf-8")
    assert "ACCEL-999" in prompt
    assert "implementation/accel-999" in prompt
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "llm_merge_resolver_invoked"
    assert events[-1]["prompt_chars"] == len(prompt)


def test_llm_merge_resolver_times_out_hung_command(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.merge_resolver import invoke_llm_resolver

    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")

    result = invoke_llm_resolver(
        {"found": True, "repo_root": str(tmp_path), "prompt": "resolve this merge"},
        command_template=f"{shlex.quote(sys.executable)} {shlex.quote(str(sleeper))}",
        timeout_seconds=0.01,
    )

    assert result["applied"] is False
    assert result["llm_timeout"] is True
    assert result["llm_returncode"] is None
    assert "timed out" in result["apply_error"]


def test_implementation_supervisor_aborts_interrupted_main_checkout_merge(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "conflict.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/conflict")
    target.write_text("feature\n", encoding="utf-8")
    _git(repo, "commit", "-am", "feature")
    _git(repo, "checkout", "main")
    target.write_text("main\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main")
    merge = subprocess.run(["git", "merge", "implementation/conflict"], cwd=repo, text=True, capture_output=True)
    assert merge.returncode != 0

    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=repo / "state" / "task_state.json",
            strategy_path=repo / "state" / "strategy.json",
            events_path=repo / "state" / "events.jsonl",
            state_dir=repo / "state",
            repo_root=repo,
        )
    )

    result = supervisor.repair_main_checkout_merge_state()

    assert result["repaired"] is True
    assert result["reason"] == "merge_aborted_without_resolver"
    assert result["abort_result"]["aborted"] is True
    assert supervisor._git_merge_head(repo) == ""
    assert supervisor._git_unmerged_paths(repo) == []
    assert target.read_text(encoding="utf-8") == "main\n"


def test_implementation_supervisor_invokes_llm_for_interrupted_main_checkout_merge(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "conflict.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/conflict")
    target.write_text("feature\n", encoding="utf-8")
    _git(repo, "commit", "-am", "feature")
    _git(repo, "checkout", "main")
    target.write_text("main\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main")
    merge = subprocess.run(["git", "merge", "implementation/conflict"], cwd=repo, text=True, capture_output=True)
    assert merge.returncode != 0

    capture_path = tmp_path / "supervisor-resolver-prompt.txt"
    resolver_script = tmp_path / "supervisor_resolver.py"
    resolver_script.write_text(
        "import pathlib, subprocess, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n"
        "pathlib.Path('conflict.txt').write_text('resolved by supervisor\\n', encoding='utf-8')\n"
        "subprocess.check_call(['git', 'add', 'conflict.txt'])\n",
        encoding="utf-8",
    )
    supervisor = TodoImplementationSupervisor(
        TodoSupervisorConfig(
            todo_path=repo / "todo.md",
            state_path=repo / "state" / "task_state.json",
            strategy_path=repo / "state" / "strategy.json",
            events_path=repo / "state" / "events.jsonl",
            state_dir=repo / "state",
            repo_root=repo,
            llm_merge_resolver_command=(
                f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))} "
                f"{shlex.quote(str(capture_path))}"
            ),
            llm_merge_resolver_timeout_seconds=5,
        )
    )

    result = supervisor.repair_main_checkout_merge_state()

    assert result["repaired"] is True
    assert result["reason"] == "llm_resolved_merge"
    assert result["llm_merge_resolver"]["applied"] is True
    assert result["commit_result"]["completed"] is True
    assert supervisor._git_merge_head(repo) == ""
    assert supervisor._git_unmerged_paths(repo) == []
    assert target.read_text(encoding="utf-8") == "resolved by supervisor\n"
    assert _git(repo, "merge-base", "--is-ancestor", "implementation/conflict", "HEAD") == ""
    prompt = capture_path.read_text(encoding="utf-8")
    assert "supervisor_main_checkout_merge_in_progress" in prompt
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "main_checkout_merge_state_repair"


def test_implementation_daemon_invokes_llm_resolver_for_dirty_checkout_blocker(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "blocked.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "blocked.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/auto-001")
    target.write_text("branch\n", encoding="utf-8")
    _git(repo, "commit", "-am", "branch")
    _git(repo, "checkout", "main")
    target.write_text("local dirty\n", encoding="utf-8")

    capture_path = tmp_path / "dirty-resolver-prompt.txt"
    resolver_script = tmp_path / "dirty_resolver.py"
    resolver_script.write_text(
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        llm_merge_resolver_command=(
            f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))} {shlex.quote(str(capture_path))}"
        ),
        llm_merge_resolver_timeout_seconds=5,
    )

    result = daemon._merge_branch_to_main(
        "implementation/auto-001",
        PortalTask(
            task_id="AUTO-001",
            title="Resolve dirty checkout blocker",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        1,
    )

    assert result["merged"] is False
    assert result["reason"] == "main_checkout_dirty_conflict"
    assert result["dirty_paths"] == ["blocked.txt"]
    assert result["llm_merge_resolver"]["applied"] is True
    prompt = capture_path.read_text(encoding="utf-8")
    assert "main_checkout_dirty_conflict" in prompt
    assert "Dirty paths: blocked.txt" in prompt


def test_implementation_daemon_repairs_dirty_managed_main_merge_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "blocked.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "blocked.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/auto-004")
    target.write_text("branch\n", encoding="utf-8")
    _git(repo, "commit", "-am", "branch")
    _git(repo, "checkout", "-b", "driver", "main")

    capture_path = tmp_path / "managed-main-resolver-prompt.txt"
    resolver_script = tmp_path / "managed_main_resolver.py"
    resolver_script.write_text(
        "import pathlib, subprocess, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n"
        "subprocess.check_call(['git', 'checkout', '--', 'blocked.txt'])\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        worktree_root=repo / "worktrees",
        llm_merge_resolver_command=(
            f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))} {shlex.quote(str(capture_path))}"
        ),
        llm_merge_resolver_timeout_seconds=5,
    )
    workspace_result = daemon._prepare_main_merge_workspace("main", "implementation/auto-004")
    workspace = Path(str(workspace_result["path"]))
    (workspace / "blocked.txt").write_text("dirty managed worktree\n", encoding="utf-8")

    result = daemon._merge_branch_to_main(
        "implementation/auto-004",
        PortalTask(
            task_id="AUTO-004",
            title="Repair dirty managed main merge worktree",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        1,
    )

    assert result["merged"] is True
    assert result["llm_workspace_resolver"]["applied"] is True
    assert _git(repo, "merge-base", "--is-ancestor", "implementation/auto-004", "main") == ""
    prompt = capture_path.read_text(encoding="utf-8")
    assert "main_merge_worktree_dirty" in prompt
    assert "Dirty paths: blocked.txt" in prompt
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "main_merge_workspace_blocker_resolved" for event in events)


def test_implementation_daemon_invokes_llm_resolver_for_dirty_submodule_checkout(tmp_path):
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    _git(submodule, "checkout", "-b", "implementation/auto-001-submodule-libs-child")
    (submodule / "child.txt").write_text("branch\n", encoding="utf-8")
    _git(submodule, "commit", "-am", "submodule branch")
    _git(submodule, "checkout", "main")
    (submodule / "child.txt").write_text("dirty\n", encoding="utf-8")

    capture_path = tmp_path / "submodule-dirty-prompt.txt"
    resolver_script = tmp_path / "submodule_dirty_resolver.py"
    resolver_script.write_text(
        "import pathlib, subprocess, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n"
        "subprocess.check_call(['git', 'checkout', '--', 'child.txt'])\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
        llm_merge_resolver_command=(
            f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))} {shlex.quote(str(capture_path))}"
        ),
        llm_merge_resolver_timeout_seconds=5,
    )

    results = daemon._merge_submodule_branches_to_main(
        "implementation/auto-001",
        task=PortalTask(
            task_id="AUTO-001",
            title="Resolve dirty submodule checkout",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        attempt=1,
    )

    assert results[0]["merged"] is True
    assert results[0]["path"] == "libs/child"
    assert _git(submodule, "rev-parse", "HEAD") == _git(
        submodule,
        "rev-parse",
        "implementation/auto-001-submodule-libs-child",
    )
    prompt = capture_path.read_text(encoding="utf-8")
    assert "submodule_checkout_dirty" in prompt
    assert "Dirty paths: child.txt" in prompt
    events = [json.loads(line) for line in (repo / "state" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "submodule_checkout_blocker_resolved" for event in events)


def test_implementation_daemon_commits_llm_resolved_merge(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "conflict.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "feature")
    target.write_text("feature\n", encoding="utf-8")
    _git(repo, "commit", "-am", "feature")
    _git(repo, "checkout", "main")
    target.write_text("main\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main")
    merge = subprocess.run(
        ["git", "merge", "feature"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert merge.returncode != 0
    target.write_text("resolved\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt")
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )

    result = daemon._commit_llm_resolved_merge(repo)

    assert result["completed"] is True
    assert target.read_text(encoding="utf-8") == "resolved\n"
    no_merge_head = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "MERGE_HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert no_merge_head.returncode != 0


def test_implementation_daemon_accepts_resolver_committed_merge(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    target = repo / "conflict.txt"
    target.write_text("base\n", encoding="utf-8")
    _git(repo, "add", "conflict.txt")
    _git(repo, "commit", "-m", "base")
    _git(repo, "checkout", "-b", "implementation/auto-merge")
    target.write_text("feature\n", encoding="utf-8")
    _git(repo, "commit", "-am", "feature")
    _git(repo, "checkout", "main")
    target.write_text("main\n", encoding="utf-8")
    _git(repo, "commit", "-am", "main")

    resolver_script = tmp_path / "resolver_commits.py"
    resolver_script.write_text(
        "import pathlib, subprocess\n"
        "pathlib.Path('conflict.txt').write_text('resolved by resolver\\n', encoding='utf-8')\n"
        "subprocess.check_call(['git', 'add', 'conflict.txt'])\n"
        "subprocess.check_call(['git', 'commit', '--no-edit'])\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        llm_merge_resolver_command=f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))}",
        llm_merge_resolver_timeout_seconds=5,
    )

    result = daemon._merge_branch_to_main(
        "implementation/auto-merge",
        PortalTask(
            task_id="AUTO-002",
            title="Accept resolver committed merge",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        1,
    )

    assert result["merged"] is True
    assert result["llm_merge_commit_result"]["reason"] == "resolver_committed_merge"
    assert target.read_text(encoding="utf-8") == "resolved by resolver\n"
    assert _git(repo, "merge-base", "--is-ancestor", "implementation/auto-merge", "HEAD") == ""


def test_implementation_daemon_invokes_llm_resolver_for_submodule_merge_conflict(tmp_path):
    repo, submodule = _seed_parent_with_submodule(tmp_path)
    _git(submodule, "checkout", "-b", "implementation/auto-003-submodule-libs-child")
    (submodule / "child.txt").write_text("branch\n", encoding="utf-8")
    _git(submodule, "commit", "-am", "submodule branch")
    _git(submodule, "checkout", "main")
    (submodule / "child.txt").write_text("main\n", encoding="utf-8")
    _git(submodule, "commit", "-am", "main change")

    capture_path = tmp_path / "submodule-conflict-prompt.txt"
    resolver_script = tmp_path / "submodule_conflict_resolver.py"
    resolver_script.write_text(
        "import pathlib, subprocess, sys\n"
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read(), encoding='utf-8')\n"
        "pathlib.Path('child.txt').write_text('resolved\\n', encoding='utf-8')\n"
        "subprocess.check_call(['git', 'add', 'child.txt'])\n",
        encoding="utf-8",
    )
    daemon = TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        worktree_submodule_paths=["libs/child"],
        llm_merge_resolver_command=(
            f"{shlex.quote(sys.executable)} {shlex.quote(str(resolver_script))} {shlex.quote(str(capture_path))}"
        ),
        llm_merge_resolver_timeout_seconds=5,
    )

    results = daemon._merge_submodule_branches_to_main(
        "implementation/auto-003",
        task=PortalTask(
            task_id="AUTO-003",
            title="Resolve submodule merge conflict",
            status="todo",
            completion="manual",
            priority="P1",
            track="ops",
        ),
        attempt=1,
    )

    assert results[0]["merged"] is True
    assert results[0]["ff_only_result"]["returncode"] != 0
    assert results[0]["llm_merge_resolver"]["applied"] is True
    assert results[0]["llm_merge_commit_result"]["completed"] is True
    assert (submodule / "child.txt").read_text(encoding="utf-8") == "resolved\n"
    assert _git(
        submodule,
        "merge-base",
        "--is-ancestor",
        "implementation/auto-003-submodule-libs-child",
        "HEAD",
    ) == ""
    prompt = capture_path.read_text(encoding="utf-8")
    assert "submodule_merge_conflict" in prompt
    assert "Unmerged paths: child.txt" in prompt
