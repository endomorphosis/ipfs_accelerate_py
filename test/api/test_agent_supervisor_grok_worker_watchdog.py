from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor


@pytest.mark.parametrize("python_option", ["", "-P "])
def test_grok_cli_runner_module_is_an_active_worktree_worker(
    monkeypatch: pytest.MonkeyPatch,
    python_option: str,
) -> None:
    now = datetime.now(UTC)
    old = now - timedelta(minutes=10)
    command = (
        f"/usr/bin/python3 {python_option}-m "
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner "
        "--workspace /tmp/agent-worktree --model grok-4.6"
    )
    monkeypatch.setattr(
        supervisor,
        "descendant_processes",
        lambda _pid: [{"pid": 4321, "cmdline": command}],
    )

    status = supervisor.worktree_phase_worker_status(
        {
            "active_phase": "implementing",
            "active_phase_started_at": old.isoformat(),
        },
        daemon_pid=1234,
        threshold_seconds=60,
        now=now,
    )

    assert status["active_worker_count"] == 1
    assert status["active_worker_pids"] == [4321]
    assert status["stalled_without_active_worker"] is False


def test_unknown_python_module_does_not_extend_the_worker_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(UTC)
    old = now - timedelta(minutes=10)
    monkeypatch.setattr(
        supervisor,
        "descendant_processes",
        lambda _pid: [
            {"pid": 4322, "cmdline": "/usr/bin/python3 -m unrelated.worker"}
        ],
    )

    status = supervisor.worktree_phase_worker_status(
        {
            "active_phase": "implementing",
            "active_phase_started_at": old.isoformat(),
        },
        daemon_pid=1234,
        threshold_seconds=60,
        now=now,
    )

    assert status["active_worker_count"] == 0
    assert status["stalled_without_active_worker"] is True
