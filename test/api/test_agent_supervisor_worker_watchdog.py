from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import ManagedDaemonSpec
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop import (
    SupervisorLoop,
    SupervisorLoopConfig,
)


@pytest.mark.parametrize(
    "cmdline",
    [
        (
            "/home/example/.local/bin/python -m "
            "ipfs_accelerate_py.agent_supervisor.grok_cli_runner "
            "--workspace /tmp/task --model grok-4.5"
        ),
        (
            "/usr/bin/python3 -P -m "
            "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner "
            "--workspace /tmp/task --model grok-4.5"
        ),
        (
            "/usr/bin/python3 /opt/ipfs/agent_supervisor/grok_cli_runner.py "
            "--workspace /tmp/task --model grok-4.5"
        ),
    ],
)
def test_watchdog_recognizes_packaged_grok_runner(
    monkeypatch: pytest.MonkeyPatch,
    cmdline: str,
) -> None:
    now = datetime.now(UTC)
    monkeypatch.setattr(
        supervisor,
        "descendant_processes",
        lambda _pid: [{"pid": 4322, "cmdline": cmdline}],
    )

    status = supervisor.worktree_phase_worker_status(
        {
            "active_phase": "implementing",
            "active_phase_started_at": (now - timedelta(minutes=10)).isoformat(),
        },
        daemon_pid=1234,
        threshold_seconds=60,
        now=now,
    )

    assert status["active_worker_count"] == 1
    assert status["active_worker_pids"] == [4322]
    assert status["stalled_without_active_worker"] is False


@pytest.mark.parametrize(
    "cmdline",
    [
        "/usr/bin/python3 -m pytest test/api -q",
        (
            "/usr/bin/python3 -m "
            "ipfs_accelerate_py.agent_supervisor.grok_cli_runner_helper"
        ),
        "/usr/bin/python3 /opt/ipfs/agent_supervisor/not_grok_cli_runner.py",
    ],
)
def test_watchdog_does_not_treat_arbitrary_python_as_agent_worker(
    monkeypatch: pytest.MonkeyPatch,
    cmdline: str,
) -> None:
    now = datetime.now(UTC)
    monkeypatch.setattr(
        supervisor,
        "descendant_processes",
        lambda _pid: [
            {
                "pid": 4323,
                "cmdline": cmdline,
            }
        ],
    )

    status = supervisor.worktree_phase_worker_status(
        {
            "active_phase": "implementing",
            "active_phase_started_at": (now - timedelta(minutes=10)).isoformat(),
        },
        daemon_pid=1234,
        threshold_seconds=60,
        now=now,
    )

    assert status["active_worker_count"] == 0
    assert status["stalled_without_active_worker"] is True


def test_supervisor_loop_graces_packaged_runner_disappearance(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    state_dir = repo / "state"
    state_dir.mkdir(parents=True)
    spec = ManagedDaemonSpec(
        name="test-daemon",
        schema="test.daemon",
        repo_root=repo,
        daemon_dir=state_dir,
        runner=(sys.executable, "-c", "pass"),
        status_path=state_dir / "daemon_status.json",
        supervisor_status_path=state_dir / "supervisor_status.json",
        supervisor_pid_path=state_dir / "supervisor.pid",
        child_pid_path=state_dir / "child.pid",
        supervisor_out_path=state_dir / "supervisor.out",
        ensure_status_path=state_dir / "ensure_status.json",
        ensure_check_path=state_dir / "ensure_check.json",
    )
    clock = [100.0]
    loop = SupervisorLoop(
        SupervisorLoopConfig(
            spec=spec,
            command=(sys.executable, "-c", "pass"),
            log_prefix="child",
            watchdog_stale_after_seconds=60,
        ),
        monotonic=lambda: clock[0],
    )
    runner_live = [True]
    monkeypatch.setattr(
        supervisor,
        "descendant_processes",
        lambda _pid: (
            [
                {
                    "pid": 4324,
                    "cmdline": (
                        "/usr/bin/python3.12 -m "
                        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner "
                        "--workspace /tmp/task --model grok-4.5"
                    ),
                }
            ]
            if runner_live[0]
            else []
        ),
    )
    now = datetime.now(UTC)
    status = {
        "heartbeat_at": now.isoformat(),
        "active_phase": "implementing",
        "active_phase_started_at": (now - timedelta(minutes=10)).isoformat(),
        "worktree_no_child_stall_seconds": 60,
    }
    child = SimpleNamespace(pid=1234)

    live = loop.default_watchdog(child, status)
    assert live.action == "continue"
    assert loop._last_worker_status["active_worker_count"] == 1
    assert loop._last_worker_status["worker_absence_age_seconds"] == 0.0

    runner_live[0] = False
    clock[0] = 110.0
    within_grace = loop.default_watchdog(child, status)
    assert within_grace.action == "continue"
    assert loop._last_worker_status["active_worker_count"] == 0
    assert loop._last_worker_status["worker_absence_age_seconds"] == 10.0

    clock[0] = 161.0
    expired = loop.default_watchdog(child, status)
    assert expired.action == "recycle"
    assert expired.reason == "worktree_phase_without_active_child"
    assert expired.detail["worker_absence_age_seconds"] == 61.0
