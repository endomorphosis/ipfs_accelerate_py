"""Ordinary cleanup must retain custody until its native child is proven gone."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import pid_alive, write_json_atomic
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTaskState
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor, TodoSupervisorConfig,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import SupervisorLoop
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    adopt_supervised_child, clear_child_pid_file, launch_supervised_child,
    load_supervised_child_identity, read_process_birth, read_process_command_argv,
)


def _wait_until(predicate, *, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), "native fixture did not reach its bounded observation"


@contextmanager
def _native_owner(tmp_path: Path, *, identity_fault="", worker=False):
    repo = tmp_path / "repo"
    repo.mkdir()
    state_dir = repo / "state"
    state_dir.mkdir()
    board = repo / "tasks.todo.md"
    board.write_text(
        "# Tasks\n\n## ORD-001 Finished task\n- Status: completed\n"
        "- Completion: manual\n- Priority: P0\n- Outputs: answer.py\n"
        "- Validation: python -m py_compile answer.py\n", encoding="utf-8",
    )
    ready, trigger, worker_pid = (tmp_path / name for name in ("ready", "spawn-worker", "worker.pid"))
    script = repo / "authored_daemon.py"
    script.write_text(
        "import pathlib, subprocess, sys, time\n"
        f"ready=pathlib.Path({str(ready)!r})\n"
        f"trigger=pathlib.Path({str(trigger)!r})\n"
        f"worker_pid=pathlib.Path({str(worker_pid)!r})\n"
        "ready.write_text('ready')\n"
        "spawned=False\n"
        "while True:\n"
        "    if trigger.exists() and not spawned:\n"
        "        child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "        worker_pid.write_text(str(child.pid))\n"
        "        spawned=True\n"
        "    time.sleep(.01)\n", encoding="utf-8",
    )
    supervisor = PortalImplementationSupervisor(TodoSupervisorConfig(
        todo_path=board, state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json", events_path=state_dir / "events.jsonl",
        state_dir=state_dir, repo_root=repo, task_prefix="## ORD-", state_prefix="native-audit",
        daemon_script_path=script, implement=True,
    ))
    command = supervisor._build_daemon_command()
    if identity_fault == "weak_argv":
        command = [*command, "--unconfigured-test-argument"]
    process = subprocess.Popen(
        command, cwd=repo, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True,
    )
    captured_worker = None
    try:
        _wait_until(lambda: ready.exists() and read_process_command_argv(process.pid) == tuple(command))
        if identity_fault not in {"missing", "weak_argv"}:
            supervisor._write_managed_daemon_identity(pid=process.pid, command=command)
            identity_path = supervisor._managed_daemon_identity_path()
            identity = load_supervised_child_identity(identity_path)
            assert identity is not None
            if identity_fault == "foreign_scope":
                identity = replace(identity, record_id="", owner_scope={**identity.owner_scope, "repo_root": str(tmp_path / "foreign")})
            elif identity_fault == "stale_birth":
                identity = replace(identity, record_id="", process_birth=replace(
                    identity.process_birth, start_time_ticks=identity.process_birth.start_time_ticks + 1,
                ))
            elif identity_fault == "stale_boot":
                identity = replace(identity, record_id="", process_birth=replace(
                    identity.process_birth, boot_id="00000000-0000-0000-0000-000000000000",
                ))
            elif identity_fault == "changed_command":
                identity = replace(identity, record_id="", command=(*identity.command, "--foreign-sidecar-command"))
            if identity_fault == "corrupt":
                identity_path.write_text("invalid identity\n", encoding="utf-8")
            else:
                identity_path.write_text(json.dumps(identity.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
        pid_path = supervisor._managed_daemon_pid_path()
        pid_path.write_text(str(process.pid) + "\n", encoding="ascii")
        if worker:
            # Born after the durable daemon identity, so the fence needs a fresh census.
            trigger.touch()
            _wait_until(worker_pid.exists)
            captured_worker = read_process_birth(int(worker_pid.read_text()))
            assert captured_worker is not None and captured_worker.parent_pid == process.pid
        yield supervisor, process, captured_worker
    finally:
        # Only native fixture births created above can receive teardown signals.
        if captured_worker is not None:
            current_worker = read_process_birth(captured_worker.pid)
            if (current_worker is not None
                    and (current_worker.pid, current_worker.start_time_ticks, current_worker.boot_id)
                    == (captured_worker.pid, captured_worker.start_time_ticks, captured_worker.boot_id)
                    and pid_alive(captured_worker.pid)):
                os.kill(captured_worker.pid, signal.SIGKILL)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        else:
            process.wait(timeout=0)


def _markers(supervisor):
    return {
        str(path): path.read_bytes() if path.exists() else None
        for path in (supervisor._managed_daemon_pid_path(), supervisor._managed_daemon_identity_path())
    }


@pytest.mark.parametrize("fault", ["missing", "weak_argv", "foreign_scope", "stale_birth", "stale_boot", "changed_command", "corrupt"])
def test_native_cleanup_refuses_unproven_or_changed_custody_and_preserves_markers(tmp_path: Path, fault: str) -> None:
    with _native_owner(tmp_path, identity_fault=fault) as (supervisor, process, _):
        before = _markers(supervisor)
        result = supervisor._terminate_managed_daemon_tree(grace_seconds=0.1)
        assert result["terminated"] is False
        assert result["quiesced"] is False
        assert process.poll() is None
        assert _markers(supervisor) == before


@pytest.mark.parametrize("worker", [False, True])
def test_native_cleanup_fences_exact_birth_and_later_worker_then_clears_owned_markers(tmp_path: Path, worker: bool) -> None:
    with _native_owner(tmp_path, worker=worker) as (supervisor, process, captured_worker):
        result = supervisor._terminate_managed_daemon_tree(grace_seconds=0.1)
        assert result["terminated"] is True
        assert result["quiesced"] is True
        process.wait(timeout=3)
        if captured_worker is not None:
            assert not pid_alive(captured_worker.pid)
        assert all(value is None for value in _markers(supervisor).values())


def test_native_cleanup_clears_only_exact_owned_dead_birth(tmp_path: Path) -> None:
    with _native_owner(tmp_path) as (supervisor, process, _):
        process.terminate()
        process.wait(timeout=3)
        result = supervisor._terminate_managed_daemon_tree(grace_seconds=0.1)
        assert result["quiesced"] is True
        assert all(value is None for value in _markers(supervisor).values())


def test_native_cleanup_fences_later_worker_after_exact_owned_leader_exit(tmp_path: Path) -> None:
    with _native_owner(tmp_path, worker=True) as (supervisor, process, worker):
        assert worker is not None
        process.terminate()
        process.wait(timeout=3)
        assert pid_alive(worker.pid)
        result = supervisor._terminate_managed_daemon_tree(grace_seconds=0.1)
        assert result["quiesced"] is True
        assert not pid_alive(worker.pid)
        assert all(value is None for value in _markers(supervisor).values())


def test_native_adoption_retains_orphan_identity_while_later_owned_worker_is_live(tmp_path: Path) -> None:
    with _native_owner(tmp_path, worker=True) as (supervisor, process, worker):
        assert worker is not None
        spec = SupervisorLoop(supervisor.build_supervisor_loop_config())._child_spec("native-orphan-audit")
        supervisor._managed_daemon_pid_path().unlink()
        process.terminate()
        process.wait(timeout=3)
        assert pid_alive(worker.pid)
        before = _markers(supervisor)
        with pytest.raises(RuntimeError):
            adopt_supervised_child(spec)
        assert pid_alive(worker.pid)
        assert _markers(supervisor) == before


def test_native_adoption_retires_exact_dead_orphan_only_when_session_is_quiescent(tmp_path: Path) -> None:
    with _native_owner(tmp_path) as (supervisor, process, _):
        spec = SupervisorLoop(supervisor.build_supervisor_loop_config())._child_spec("native-dead-orphan-audit")
        identity_bytes = supervisor._managed_daemon_identity_path().read_bytes()
        supervisor._managed_daemon_pid_path().unlink()
        process.terminate()
        process.wait(timeout=3)
        assert adopt_supervised_child(spec) is None
        assert all(value is None for value in _markers(supervisor).values())
        retained = [path for path in supervisor.config.state_dir.iterdir() if path.is_file()]
        assert sum(path.read_bytes() == identity_bytes for path in retained) == 1


def test_native_marker_retirement_preserves_launched_handle_with_dead_leader_and_live_worker(tmp_path: Path) -> None:
    with _native_owner(tmp_path) as (supervisor, process, _):
        process.terminate()
        process.wait(timeout=3)
        for path in _markers(supervisor):
            Path(path).unlink(missing_ok=True)
        spec = SupervisorLoop(supervisor.build_supervisor_loop_config())._child_spec("native-launched-retirement-audit")
        child = launch_supervised_child(spec)
        worker = None
        try:
            _wait_until(lambda: read_process_command_argv(child.pid) == tuple(spec.command))
            (tmp_path / "spawn-worker").touch()
            worker_pid_path = tmp_path / "worker.pid"
            _wait_until(worker_pid_path.exists)
            worker = read_process_birth(int(worker_pid_path.read_text()))
            assert worker is not None and worker.parent_pid == child.pid
            os.kill(child.pid, signal.SIGTERM)
            assert os.waitpid(child.pid, 0)[0] == child.pid
            assert pid_alive(worker.pid)
            before = _markers(supervisor)
            assert clear_child_pid_file(child) is False
            assert pid_alive(worker.pid)
            assert _markers(supervisor) == before
        finally:
            if worker is not None:
                current = read_process_birth(worker.pid)
                if (current is not None
                        and (current.pid, current.start_time_ticks, current.boot_id)
                        == (worker.pid, worker.start_time_ticks, worker.boot_id)
                        and pid_alive(worker.pid)):
                    os.kill(worker.pid, signal.SIGKILL)
            current = read_process_birth(child.pid)
            if (current is not None and child.identity_process_birth is not None
                    and (current.pid, current.start_time_ticks, current.boot_id)
                    == (child.pid, child.identity_process_birth.start_time_ticks,
                        child.identity_process_birth.boot_id)
                    and pid_alive(child.pid)):
                os.kill(child.pid, signal.SIGKILL)
                os.waitpid(child.pid, 0)


def test_native_cleanup_preserves_conflicting_pid_marker_and_both_native_children(tmp_path: Path) -> None:
    with _native_owner(tmp_path, worker=True) as (supervisor, process, worker):
        assert worker is not None
        supervisor._managed_daemon_pid_path().write_text(str(worker.pid) + "\n", encoding="ascii")
        before = _markers(supervisor)
        result = supervisor._terminate_managed_daemon_tree(grace_seconds=0.1)
        assert result["terminated"] is False
        assert result["quiesced"] is False
        assert process.poll() is None and pid_alive(worker.pid)
        assert _markers(supervisor) == before


def test_native_cleanup_preserves_corrupt_pid_marker_and_exact_live_sidecar(tmp_path: Path) -> None:
    with _native_owner(tmp_path) as (supervisor, process, _):
        supervisor._managed_daemon_pid_path().write_text("unknown native pid\n", encoding="ascii")
        before = _markers(supervisor)
        result = supervisor._terminate_managed_daemon_tree(grace_seconds=0.1)
        assert result["terminated"] is False and result["quiesced"] is False
        assert process.poll() is None
        assert _markers(supervisor) == before


def test_native_cleanup_uses_exact_live_sidecar_when_pid_marker_is_absent(tmp_path: Path) -> None:
    with _native_owner(tmp_path) as (supervisor, process, _):
        supervisor._managed_daemon_pid_path().unlink()
        result = supervisor._terminate_managed_daemon_tree(grace_seconds=0.1)
        assert result["terminated"] is True and result["quiesced"] is True
        process.wait(timeout=3)
        assert all(value is None for value in _markers(supervisor).values())


def test_completed_leftover_preserves_active_state_when_native_cleanup_is_unproven(tmp_path: Path) -> None:
    with _native_owner(tmp_path, identity_fault="missing") as (supervisor, process, _):
        state = PortalTaskState(
            active_task_id="ORD-001", active_task_cid="task:still-owned", active_attempt=2,
            active_phase="implementation", implementation_in_progress=True,
            active_worktree_path=str(tmp_path / "candidate"), active_branch="implementation/ord-001",
        )
        state.save(supervisor.config.state_path)
        assert supervisor._board_task_is_completed("ORD-001")
        before_state = asdict(PortalTaskState.load(supervisor.config.state_path))
        assert before_state["active_task_cid"] == "task:still-owned"
        before_markers = _markers(supervisor)
        result = supervisor.release_completed_leftover_execution()
        assert result["released"] is False
        assert result["stop"]["quiesced"] is False
        assert asdict(PortalTaskState.load(supervisor.config.state_path)) == before_state
        assert _markers(supervisor) == before_markers
        assert process.poll() is None


def test_completed_leftover_releases_native_state_only_after_owned_session_fence(tmp_path: Path) -> None:
    with _native_owner(tmp_path, worker=True) as (supervisor, process, worker):
        state = PortalTaskState(
            active_task_id="ORD-001", active_task_cid="task:still-owned", active_attempt=2,
            active_phase="implementation", implementation_in_progress=True,
            active_worktree_path=str(tmp_path / "candidate"), active_branch="implementation/ord-001",
        )
        state.save(supervisor.config.state_path)
        result = supervisor.release_completed_leftover_execution()
        assert result["released"] is True and result["stop"]["quiesced"] is True
        process.wait(timeout=3)
        assert worker is not None and not pid_alive(worker.pid)
        persisted = PortalTaskState.load(supervisor.config.state_path)
        assert persisted.active_task_id == "ORD-001" and persisted.active_task_cid == "task:still-owned"
        assert persisted.last_implementation_task_id == "ORD-001"
        assert persisted.last_implementation_task_cid == "task:still-owned"
        assert persisted.implementation_attempts["ORD-001"] == 2
        assert persisted.implementation_attempts_by_cid["task:still-owned"] == 2
        assert persisted.active_attempt == 0 and persisted.implementation_in_progress is False
        assert persisted.active_worktree_path == "" and persisted.active_branch == ""
        assert all(value is None for value in _markers(supervisor).values())


@pytest.mark.parametrize("fault", ["missing", "foreign_scope"])
def test_signal_shutdown_projects_actual_blocked_native_cleanup(tmp_path: Path, fault: str) -> None:
    with _native_owner(tmp_path, identity_fault=fault) as (supervisor, process, _):
        before = _markers(supervisor)
        cleanup = supervisor._terminate_managed_daemon_tree(grace_seconds=0.1)
        supervisor._write_signal_shutdown_status(
            stop_signal=signal.SIGTERM, cleanup=cleanup,
            interrupted_reconciliation={"reconciled": False, "reason": "native_cleanup_unproven"},
        )
        status = json.loads(supervisor._supervisor_status_path().read_text())
        assert cleanup["quiesced"] is False
        assert status["status"] == "termination_blocked"
        assert status["managed_daemon_cleanup"] == cleanup
        assert status["daemon_pid"] == process.pid
        assert status["daemon_pid_alive"] is True
        assert process.poll() is None
        assert _markers(supervisor) == before


@pytest.mark.parametrize("identity_fault", ["", "foreign_scope"])
def test_signal_shutdown_keeps_worker_observations_until_native_quiescence(tmp_path: Path, identity_fault: str) -> None:
    with _native_owner(tmp_path, identity_fault=identity_fault, worker=True) as (supervisor, process, worker):
        assert worker is not None and pid_alive(worker.pid)
        write_json_atomic(supervisor._supervisor_status_path(), {
            "status": "running", "daemon_pid": process.pid, "daemon_pid_alive": True,
            "active_worker_count": 1, "active_worker_pids": [worker.pid],
            "worker_descendant_count": 1,
        })
        cleanup = supervisor._terminate_managed_daemon_tree(grace_seconds=0.1)
        supervisor._write_signal_shutdown_status(
            stop_signal=signal.SIGTERM, cleanup=cleanup,
            interrupted_reconciliation={"reconciled": False},
        )
        status = json.loads(supervisor._supervisor_status_path().read_text())
        if identity_fault:
            assert cleanup["quiesced"] is False
            assert status["status"] == "termination_blocked"
            assert status["daemon_pid"] == process.pid and status["daemon_pid_alive"] is True
            assert status["active_worker_pids"] == [worker.pid]
            assert status["active_worker_count"] == status["worker_descendant_count"] == 1
            assert process.poll() is None and pid_alive(worker.pid)
        else:
            assert cleanup["quiesced"] is True
            assert status["status"] == "stopped"
            assert status["daemon_pid"] is None and status["daemon_pid_alive"] is False
            assert status["active_worker_pids"] == []
            assert status["active_worker_count"] == status["worker_descendant_count"] == 0
            process.wait(timeout=3)
            assert not pid_alive(worker.pid)
