"""Unresolved callbacks retain custody without polling at full speed."""

import copy
import argparse
import logging
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    bounded_daemon_wait_timeout,
    ImplementationDaemonRunContext,
    run_portal_implementation_daemon_loop,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon_runner as runner


def deferred(**updates):
    return {
        "deferred": True,
        "skipped": True,
        "implementation_result": None,
        "backoff_seconds": 5,
        "unchanged": None,
        "provider_dispatched": "unknown",
        "retry_authorized": False,
        **updates,
    }


def test_custody_deferral_honors_backoff_without_claiming_idle_or_retry():
    result = deferred()
    before = copy.deepcopy(result)
    assert bounded_daemon_wait_timeout(result, default_timeout=1) == 5
    assert result == before
    assert bounded_daemon_wait_timeout(result, default_timeout=10) == 10


@pytest.mark.parametrize("deadline", [0, 2, 20])
def test_stale_retry_deadline_cannot_cancel_backoff(deadline):
    assert bounded_daemon_wait_timeout(
        deferred(next_wake_after_seconds=deadline), default_timeout=1,
    ) == 5


@pytest.mark.parametrize("value", [None, True, "5", -1, 0, float("nan"), float("inf")])
def test_invalid_backoff_cannot_disable_polling(value):
    assert bounded_daemon_wait_timeout(deferred(backoff_seconds=value), default_timeout=1) == 1


@pytest.mark.parametrize("value", [60, 1000])
def test_excessive_backoff_is_bounded_for_heartbeats(value):
    assert bounded_daemon_wait_timeout(deferred(backoff_seconds=value), default_timeout=1) == 30


def test_pass_without_backoff_retains_scheduled_polling():
    assert bounded_daemon_wait_timeout({}, default_timeout=1) == 1
    assert bounded_daemon_wait_timeout({"next_wake_after_seconds": 0}, default_timeout=1) == 0


def test_database_wait_does_not_cap_custody_backoff_at_one_second(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon

    daemon = object.__new__(DatabaseImplementationDaemon)
    daemon._quack_attach_blocked_until = 0.0
    sleeps = []
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    daemon.wait_for_wake(bounded_daemon_wait_timeout(deferred(), default_timeout=1))
    daemon.wait_for_wake(300)
    daemon.wait_for_wake(0)
    assert sleeps == [5, 30]


@pytest.mark.parametrize("event_driven", [False, True])
def test_loop_waits_on_deferral_then_processes_wake_and_closes(tmp_path, monkeypatch, event_driven):
    calls = []

    def run_once():
        calls.append("pass")
        if calls.count("pass") == 2:
            raise KeyboardInterrupt
        return deferred()

    def wake(timeout):
        calls.append(("wait", timeout))
        # A native event can return before the timeout; no extra sleep follows.

    daemon = SimpleNamespace(
        run_once=run_once, close_event_runtime=lambda: calls.append("closed"),
    )
    if event_driven:
        daemon.wait_for_wake = wake
        monkeypatch.setattr(runner.time, "sleep", lambda _: pytest.fail("event wake followed by sleep"))
    else:
        monkeypatch.setattr(runner.time, "sleep", wake)
    context = ImplementationDaemonRunContext(
        parsed=argparse.Namespace(once=False, interval=1),
        state_path=tmp_path / "state", strategy_path=tmp_path / "strategy",
        events_path=tmp_path / "events",
    )
    with pytest.raises(KeyboardInterrupt):
        run_portal_implementation_daemon_loop(daemon, context, logger=logging.getLogger(__name__))
    assert calls == ["pass", ("wait", 5), "pass", "closed"]
