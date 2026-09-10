"""Merged census boundaries retain exact argv and attempt-scoped observations."""
from __future__ import annotations

import shlex
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor as workers

GROK_MODULE = "ipfs_accelerate_py.agent_supervisor.grok_cli_runner"
CLEANUP = "--internal-docker-cleanup-watchdog"


@pytest.mark.parametrize("caller", ["active", "phase"])
@pytest.mark.parametrize("argv,display,expected", [
    (("grok", "--prompt", ""), f"echo {CLEANUP}", True),
    (("python3", "-m", GROK_MODULE, "--prompt", CLEANUP), "", True),
    (("python3", "-m", GROK_MODULE, CLEANUP), "grok", False),
    (("python3", "-c", f"-m {GROK_MODULE}"), "grok", False),
])
def test_worker_census_uses_exact_arguments_before_display(monkeypatch, caller, argv, display, expected):
    item = {"pid": 41002, "argv": argv, "cmdline": display}
    monkeypatch.setattr(workers, "descendant_processes", lambda _pid: [item])
    if caller == "active":
        found = workers.active_codex_exec_workers(41001)
        assert found == ([item] if expected else [])
    else:
        found = workers.worktree_phase_worker_status(
            {"active_phase": "implementing"}, daemon_pid=41001,
        )
        assert found["active_worker_pids"] == ([41002] if expected else [])


@pytest.mark.parametrize("argv,cleanup", [
    (("python3", "-m", GROK_MODULE, CLEANUP), True),
    (("python3", "-uB", "-m", GROK_MODULE, CLEANUP), True),
    (("python3", "-W", "ignore", "/opt/grok_cli_runner.py", CLEANUP), True),
    (("python3", "--", "/opt/grok_cli_runner.py", CLEANUP), True),
    (("python3", "-m", GROK_MODULE, "--prompt", CLEANUP), False),
    (("python3", "-c", GROK_MODULE, CLEANUP), False),
    (("grok", "--prompt", CLEANUP), False),
    (("echo", CLEANUP), False),
])
def test_cleanup_exclusion_requires_actual_runner_dispatch(argv, cleanup):
    assert workers._is_internal_docker_cleanup_watchdog(shlex.join(argv)) is cleanup
    if cleanup:
        assert not workers._is_agent_worker_argv(argv)
        assert not workers._is_agent_worker_command(shlex.join(argv))


def _proc_record(root: Path, pid: int, parent: int, argv: bytes, ticks: int = 99) -> None:
    process = root / str(pid)
    process.mkdir(exist_ok=True)
    fields = ["0"] * 50
    fields[0] = "S"
    fields[1] = str(parent)
    fields[19] = str(ticks)
    (process / "stat").write_text(f"{pid} (worker) {' '.join(fields)}\n")
    (process / "cmdline").write_bytes(argv)


def test_strict_procfs_census_preserves_empty_downstream_arguments(tmp_path):
    _proc_record(tmp_path, 41001, 1, b"python3\0")
    _proc_record(tmp_path, 41002, 41001, b"grok\0--prompt\0\0")
    found = workers.procfs_descendant_processes(41001, proc_root=tmp_path)
    assert found == [{
        "pid": 41002, "parent_pid": 41001,
        "argv": ("grok", "--prompt", ""),
        "cmdline": "grok --prompt ''", "start_ticks": 99,
    }]


@pytest.mark.parametrize("raw", [b"", b"\0--prompt\0", b"grok", b"grok\0\xff\0"])
def test_strict_procfs_census_denies_invalid_argument_records(tmp_path, raw):
    _proc_record(tmp_path, 41001, 1, b"python3\0")
    _proc_record(tmp_path, 41002, 41001, raw)
    with pytest.raises(OSError):
        workers.procfs_descendant_processes(41001, proc_root=tmp_path)


@pytest.mark.parametrize("mutation", ["reuse", "reparent", "disappear"])
def test_strict_procfs_census_keeps_birth_and_ancestry_rechecks(tmp_path, monkeypatch, mutation):
    _proc_record(tmp_path, 41001, 1, b"python3\0")
    _proc_record(tmp_path, 41002, 41001, b"grok\0")
    read_bytes = Path.read_bytes
    def change_after_argv(path):
        value = read_bytes(path)
        if path == tmp_path / "41002" / "cmdline":
            if mutation == "disappear":
                (path.parent / "stat").unlink()
            else:
                _proc_record(tmp_path, 41002, 1 if mutation == "reparent" else 41001,
                             b"grok\0", ticks=100 if mutation == "reuse" else 99)
        return value
    monkeypatch.setattr(Path, "read_bytes", change_after_argv)
    with pytest.raises(OSError):
        workers.procfs_descendant_processes(41001, proc_root=tmp_path)


@pytest.mark.parametrize("phase,known,reason", [
    ("", True, "phase_not_guarded"),
    ("validating", True, "phase_not_guarded"),
    ("typo-phase", False, "phase_unknown"),
    ("implementing", True, "phase_started_at_unavailable"),
])
def test_injected_census_keeps_unavailable_stall_evidence(monkeypatch, phase, known, reason):
    monkeypatch.setattr(workers, "descendant_processes", lambda _pid: pytest.fail("duplicate census"))
    status = workers.worktree_phase_worker_status(
        {"active_phase": phase}, daemon_pid=41001,
        threshold_seconds=60, descendants=[],
    )
    assert status["phase_known"] is known
    assert status["stall_evidence_available"] is False
    assert status["stalled_without_active_worker"] is None
    assert status["stall_evidence_unavailable_reason"] == reason
    assert status["active_worker_count"] == 0


def test_pure_tracking_generation_binds_attempt_without_process_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(workers, "descendant_processes", lambda _pid: pytest.fail("unexpected census"))
    now = datetime.now(UTC)
    state = {
        "active_phase": "implementing", "active_phase_detail": "provider_launch_birth",
        "active_phase_started_at": (now - timedelta(seconds=1)).isoformat(),
        "active_task_id": "TASK-001", "active_attempt": 1, "active_task_cid": "revision-one",
        "active_worktree_path": str(tmp_path), "implementation_in_progress": True,
    }
    generation = workers.worktree_worker_tracking_generation(state)
    observed = workers.worktree_phase_worker_status(state, descendants=[], now=now)
    assert observed["tracking_generation"] == generation
    state.update({"heartbeat_at": now.isoformat(), "last_progress_at": now.isoformat(),
                  "active_provider_runner": {"observed_at": now.isoformat()}})
    assert workers.worktree_worker_tracking_generation(state) == generation
    state["active_attempt"] = 2
    assert workers.worktree_worker_tracking_generation(state) != generation
