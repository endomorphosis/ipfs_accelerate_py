"""Owning regression for repeated legacy supervisor shutdown signals.

These cases exercise the original handler/finally and terminal status writer.
Signals and daemon cleanup are isolated fixture hooks, not native subprocess
custody or genuine cold-shutdown qualification. Product import is intentionally
separate from the lightweight regular-log-marker51 cohort.
"""
from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor as supervisor_module


@pytest.mark.parametrize("first_signal", [signal.SIGTERM, signal.SIGINT], ids=["first-term", "first-int"])
@pytest.mark.parametrize("repeat_signal", [signal.SIGTERM, signal.SIGINT], ids=["repeat-term", "repeat-int"])
def test_repeated_signal_finishes_original_shutdown_and_preserves_first_exit(tmp_path: Path, monkeypatch, first_signal, repeat_signal):
    status_path = tmp_path / "supervisor_status.json"
    status_path.write_text(json.dumps({
        "status": "running", "run_id": "fixture-run", "daemon_pid": 2468,
        "daemon_pid_alive": True, "preserved_fixture_field": "original",
    }), encoding="utf-8")
    supervisor = object.__new__(supervisor_module.PortalImplementationSupervisor)
    original_term, original_int = object(), object()
    handlers = {signal.SIGTERM: original_term, signal.SIGINT: original_int}
    registrations, events, order = [], [], []

    def capture_handler(signum, handler):
        previous = handlers[signum]
        handlers[signum] = handler
        registrations.append((signum, handler))
        return previous

    monkeypatch.setattr(supervisor_module.signal, "signal", capture_handler)

    def original_loop():
        order.append("first-signal")
        handlers[first_signal](first_signal, None)
        pytest.fail("the first signal did not leave the ordinary loop")

    def original_cleanup():
        order.append("cleanup-enter")
        # This is the observed failure boundary: the original finite owner
        # may send TERM again while the parent is fencing its managed child.
        assert handlers[repeat_signal](repeat_signal, None) is None
        order.append("cleanup-finished")
        return {"terminated": True, "quiesced": True, "blocked": False,
                "remaining_pid": None}

    def reconcile():
        order.append("reconcile")
        return {"reconciled": False, "reason": "fixture_has_no_task",
                "completion_authority": False}

    def record_event(kind, payload):
        order.append("event")
        events.append((kind, payload))

    supervisor._run_forever_loop = original_loop
    supervisor._terminate_managed_daemon_tree = original_cleanup
    supervisor._reconcile_interrupted_implementation_after_shutdown = reconcile
    supervisor._record_event = record_event
    supervisor._supervisor_status_path = lambda: status_path
    # Keep _write_signal_shutdown_status genuine: verify its retained file,
    # rather than replacing terminal publication with a successful fixture.
    with pytest.raises(SystemExit) as stopped:
        supervisor.run_forever()

    assert stopped.value.code == 128 + first_signal
    assert order == ["first-signal", "cleanup-enter", "cleanup-finished", "reconcile", "event"]
    assert len(events) == 1 and events[0][0] == "supervisor_signal_shutdown"
    assert events[0][1]["signal"] == first_signal
    assert events[0][1]["managed_daemon_cleanup"]["quiesced"] is True
    terminal = json.loads(status_path.read_text(encoding="utf-8"))
    assert terminal["status"] == "stopped"
    assert terminal["stop_signal"] == first_signal
    assert terminal["last_exit_code"] == 128 + first_signal
    assert terminal["last_recycle_reason"] == "supervisor_signal_shutdown"
    assert terminal["managed_daemon_cleanup"]["quiesced"] is True
    assert terminal["daemon_pid"] is None and terminal["daemon_pid_alive"] is False
    assert terminal["active_worker_count"] == 0 and terminal["active_worker_pids"] == []
    assert terminal["worker_descendant_count"] == 0
    assert terminal["interrupted_implementation_reconciliation"]["completion_authority"] is False
    assert terminal["run_id"] == "fixture-run"
    assert terminal["preserved_fixture_field"] == "original"
    assert handlers == {signal.SIGTERM: original_term, signal.SIGINT: original_int}
    assert len(registrations) == 4
