"""A stale heartbeat must not quiesce a daemon that still owns a worker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTaskState,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    PortalSupervisorConfig,
)


def _supervisor(tmp_path: Path) -> PortalImplementationSupervisor:
    return PortalImplementationSupervisor(
        PortalSupervisorConfig(
            todo_path=tmp_path / "todo.md",
            state_path=tmp_path / "state.json",
            strategy_path=tmp_path / "strategy.json",
            events_path=tmp_path / "events.jsonl",
            state_dir=tmp_path / "state",
            stale_seconds=60,
            implementation_timeout=3600,
            implementation_log_stall_seconds=60,
        )
    )


def _quiet_active_task() -> PortalTaskState:
    projected_at = datetime.now(UTC) - timedelta(hours=2)
    return PortalTaskState(
        heartbeat_at=projected_at.isoformat(),
        last_progress_at=projected_at.isoformat(),
        active_task_id="CIMP-UX-11",
        implementation_in_progress=False,
        ready_count=1,
    )


@pytest.mark.parametrize(
    "argv",
    [
        ("/usr/bin/grok", "--model", "grok-4.7"),
        ("python", "-m", "pytest", "-q", "tests/test_cimp_ux_11.py"),
    ],
)
def test_stale_heartbeat_does_not_quiesce_a_live_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: tuple[str, ...],
) -> None:
    outer = _supervisor(tmp_path)
    item = {"pid": 4242, "argv": argv}
    monkeypatch.setattr(outer, "_read_managed_daemon_pid", lambda: 3133285)
    monkeypatch.setattr(supervisor, "descendant_processes", lambda _pid: [item])
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor.descendant_processes",
        lambda _pid: [item],
    )

    stuck, reason = outer.is_stuck(
        _quiet_active_task(),
        now_ts=datetime.now(UTC).timestamp(),
    )

    assert stuck is False
    assert reason == ""


def test_stale_heartbeat_without_a_live_worker_stays_stuck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outer = _supervisor(tmp_path)
    monkeypatch.setattr(outer, "_read_managed_daemon_pid", lambda: 3133285)
    monkeypatch.setattr(supervisor, "descendant_processes", lambda _pid: [])
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor.descendant_processes",
        lambda _pid: [],
    )

    stuck, reason = outer.is_stuck(
        _quiet_active_task(),
        now_ts=datetime.now(UTC).timestamp(),
    )

    assert stuck is True
    assert reason == "heartbeat stale for active task CIMP-UX-11"


def test_projection_heartbeat_grace_stays_fail_closed_without_a_worker(
    tmp_path: Path,
) -> None:
    projected_at = datetime.now(UTC)
    state = PortalTaskState(
        heartbeat_at=projected_at.isoformat(),
        last_progress_at=projected_at.isoformat(),
        active_task_id="PCSM-061",
        ready_count=1,
    )
    outer = _supervisor(tmp_path)

    stuck, reason = outer.is_stuck(
        state,
        now_ts=(projected_at + timedelta(seconds=59)).timestamp(),
    )
    assert stuck is False
    assert reason == ""

    stuck, reason = outer.is_stuck(
        state,
        now_ts=(projected_at + timedelta(seconds=61)).timestamp(),
    )
    assert stuck is True
    assert reason == "heartbeat stale for active task PCSM-061"
