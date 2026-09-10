"""Retaining live callback custody must not spin the supervisor's status loop."""

from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_loop as module
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import ManagedDaemonSpec


def _loop(tmp_path, monkeypatch, decisions, *, heartbeat=30.0, poll=1.0):
    spec = ManagedDaemonSpec(
        name="poll-test", schema="poll-test@1", repo_root=tmp_path,
        daemon_dir=tmp_path / "state", runner=("test-python",),
        status_path=tmp_path / "state/status.json",
        supervisor_status_path=tmp_path / "state/supervisor.json",
        supervisor_pid_path=tmp_path / "state/supervisor.pid",
        child_pid_path=tmp_path / "state/child.pid",
        supervisor_out_path=tmp_path / "state/supervisor.out",
        ensure_status_path=tmp_path / "state/ensure.json",
        ensure_check_path=tmp_path / "state/check.json",
    )
    sleeps = []
    signals = []
    statuses = []
    loop = module.SupervisorLoop(
        module.SupervisorLoopConfig(
            spec=spec, command=("test-python",), log_prefix="test",
            heartbeat_seconds=heartbeat, poll_seconds=poll,
            watchdog_startup_grace_seconds=0,
        ), sleep=sleeps.append, monotonic=lambda: 1.0,
    )
    child = SimpleNamespace(pid=987654321, log_path=tmp_path / "child.log")
    monkeypatch.setattr(module, "adopt_or_launch_supervised_child", lambda *_a, **_k: child)
    monkeypatch.setattr(module, "_poll_child_exit", lambda _child: None)
    monkeypatch.setattr(module, "supervised_child_is_proven_dead", lambda _child: False)
    monkeypatch.setattr(module, "clear_child_pid_file", lambda _child: pytest.fail("custody cleared"))
    monkeypatch.setattr(module, "wait_for_child_exit", lambda _child: pytest.fail("unproved exit"))
    monkeypatch.setattr(
        module, "terminate_supervised_child",
        lambda _child, **_kwargs: signals.append(_child.pid) or False,
    )
    monkeypatch.setattr(loop, "_safe_write_status", lambda status, **_kwargs: statuses.append(status))
    decisions = iter(decisions)
    monkeypatch.setattr(loop, "watchdog_decision", lambda _child: next(decisions))
    return loop, sleeps, signals, statuses


@pytest.mark.parametrize("action", ["stop", "recycle"])
@pytest.mark.parametrize("heartbeat,poll,expected", [(30, 1, 1), (0.25, 2, 0.25), (0, 0, 0.01)])
def test_preserved_child_rechecks_are_paced(tmp_path, monkeypatch, action, heartbeat, poll, expected):
    retained = module.SupervisorLoopDecision(action=action, reason="control_plane_source_changed")
    blocked = module.SupervisorLoopDecision.stop("unrelated_termination_denied")
    loop, sleeps, signals, statuses = _loop(
        tmp_path, monkeypatch, [retained] * 3 + [blocked], heartbeat=heartbeat, poll=poll,
    )
    result = loop.run()
    assert sleeps == [expected] * 3
    assert statuses.count("running") == 4
    assert result.status == "termination_blocked"
    assert result.last_recycle_reason == "supervised_child_termination_unproven"
    assert result.restart_count == 0
    # Source-change stop preserves the live child before attempting a signal;
    # the recycle path keeps its existing failed-termination contract.
    assert len(signals) == (1 if action == "stop" else 4)


def test_ordinary_polling_and_unproven_termination_keep_existing_contract(tmp_path, monkeypatch):
    decisions = [module.SupervisorLoopDecision.keep_running()] * 2
    decisions.append(module.SupervisorLoopDecision.stop("explicit_stop"))
    loop, sleeps, signals, _statuses = _loop(tmp_path, monkeypatch, decisions)
    result = loop.run()
    assert sleeps == [1.0, 1.0]
    assert len(signals) == 1
    assert result.status == "termination_blocked"
    assert result.restart_count == 0
