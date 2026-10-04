"""Strict maintenance facts never substitute for independently checked process scope."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_activity import (
    SupervisorMaintenanceWindow,
    observe_supervisor_activity,
)

NOW = datetime(2026, 9, 14, 5, 0, 0, tzinfo=timezone.utc)


def projection(*, phase="running", start=None, updated=None, timeout=300):
    start = start or NOW - timedelta(seconds=20)
    updated = updated or NOW
    window = SupervisorMaintenanceWindow.begin(started_at=start.isoformat(), timeout_seconds=timeout)
    return {
        "status": "agentic_maintenance_started" if phase == "running" else "agentic_maintenance_" + phase,
        "updated_at": updated.isoformat(),
        "supervisor_maintenance": window.event(phase=phase, observed_at=updated.isoformat()),
        "active_agentic_maintenance_has_daemon": True,
        "last_agentic_maintenance_status": phase,
        "active_agentic_maintenance_timeout_seconds": timeout,
        "active_agentic_maintenance_started_at": start.isoformat() if phase == "running" else "",
    }


def test_running_remains_dispatchable_despite_previous_maintenance_projection():
    value = projection(phase="failed")
    value.update(status="running", last_agentic_maintenance_error="previous failure")
    result = observe_supervisor_activity(value, now=NOW)
    assert result.phase_ready and result.permits_dispatch
    assert result.maintenance is None


@pytest.mark.parametrize("phase", ["running", "completed"])
def test_bounded_maintenance_is_supervised_but_not_dispatchable(phase):
    result = observe_supervisor_activity(projection(phase=phase), now=NOW)
    assert result.phase_ready and not result.permits_dispatch
    assert result.reason == "bounded_maintenance_" + phase
    assert result.to_dict()["maintenance"]["phase"] == phase


def test_completed_pass_requires_timely_finish_not_observation_before_deadline():
    value = projection(phase="completed", start=NOW - timedelta(seconds=310),
                       updated=NOW - timedelta(seconds=20))
    assert observe_supervisor_activity(value, now=NOW).phase_ready
    value["supervisor_maintenance"]["finished_at"] = NOW.isoformat()
    assert not observe_supervisor_activity(value, now=NOW).phase_ready


def test_fresh_heartbeat_cannot_renew_a_completed_maintenance_window():
    value = projection(phase="completed")
    value["updated_at"] = (NOW + timedelta(seconds=10)).isoformat()
    result = observe_supervisor_activity(value, now=NOW + timedelta(seconds=10))
    assert not result.phase_ready and not result.permits_dispatch
    assert result.reason == "maintenance_completion_not_current"


@pytest.mark.parametrize("timeout", [math.inf, -math.inf, math.nan, 0, -1, True, "300", 10 ** 400])
def test_producer_refuses_nonfinite_or_nonpositive_timeout(timeout):
    with pytest.raises((ValueError, OverflowError)):
        SupervisorMaintenanceWindow.begin(started_at=NOW.isoformat(), timeout_seconds=timeout)


@pytest.mark.parametrize("change", [
    "expired", "nonfinite", "nan", "bool-timeout", "future-start", "future-heartbeat",
    "stale-heartbeat", "naive-start", "naive-deadline", "inconsistent-deadline",
    "unbounded-deadline", "missing-window", "unknown-field", "different-schema",
    "different-phase", "timeout-disagrees", "active-start-disagrees", "active-finished",
    "without-daemon", "legacy-failure", "legacy-error", "failed",
    "completed-no-finish", "completed-future-finish", "completed-active-start",
])
def test_invalid_maintenance_facts_never_admit_health_or_dispatch(change):
    value = projection()
    window = value["supervisor_maintenance"]
    if change == "expired":
        value = projection(start=NOW - timedelta(seconds=301))
    elif change == "nonfinite": window["timeout_seconds"] = math.inf
    elif change == "nan": window["timeout_seconds"] = math.nan
    elif change == "bool-timeout": window["timeout_seconds"] = True
    elif change == "future-start": value = projection(start=NOW + timedelta(seconds=1))
    elif change == "future-heartbeat": value["updated_at"] = (NOW + timedelta(seconds=1)).isoformat()
    elif change == "stale-heartbeat": value["updated_at"] = (NOW - timedelta(seconds=61)).isoformat()
    elif change == "naive-start": window["started_at"] = "2026-09-14T04:59:40"
    elif change == "naive-deadline": window["deadline_at"] = "2026-09-14T05:04:40"
    elif change == "inconsistent-deadline": window["deadline_at"] = (NOW + timedelta(days=1)).isoformat()
    elif change == "unbounded-deadline": window["deadline_at"] = math.inf
    elif change == "missing-window": value.pop("supervisor_maintenance")
    elif change == "unknown-field": window["renewal_allowed"] = True
    elif change == "different-schema": window["schema"] += "-unknown"
    elif change == "different-phase": window["phase"] = "completed"
    elif change == "timeout-disagrees": value["active_agentic_maintenance_timeout_seconds"] = 301
    elif change == "active-start-disagrees": value["active_agentic_maintenance_started_at"] = NOW.isoformat()
    elif change == "active-finished": window["finished_at"] = NOW.isoformat()
    elif change == "without-daemon": value["active_agentic_maintenance_has_daemon"] = False
    elif change == "legacy-failure": value["last_agentic_maintenance_status"] = "failed"
    elif change == "legacy-error": value["last_agentic_maintenance_error"] = "retained failure"
    elif change == "failed": value = projection(phase="failed")
    else:
        value = projection(phase="completed")
        if change == "completed-no-finish": value["supervisor_maintenance"]["finished_at"] = None
        elif change == "completed-future-finish": value["supervisor_maintenance"]["finished_at"] = (NOW + timedelta(seconds=1)).isoformat()
        elif change == "completed-active-start": value["active_agentic_maintenance_started_at"] = NOW.isoformat()
    original = deepcopy(value)
    result = observe_supervisor_activity(value, now=NOW)
    assert not result.phase_ready and not result.permits_dispatch, result
    # The observation must not repair, renew or mutate its input.
    assert repr(value) == repr(original)


@pytest.mark.parametrize("clock", [NOW.replace(tzinfo=None), None, "2026-09-14T05:00:00Z"])
def test_unknown_observation_clock_refuses(clock):
    assert not observe_supervisor_activity(projection(), now=clock).phase_ready


@pytest.fixture
def writer(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_supervisor as module
    owner = object.__new__(module.TodoImplementationSupervisor)
    owner.config = SimpleNamespace(repo_root=tmp_path, state_path=tmp_path / "daemon.json",
        state_dir=tmp_path, state_prefix="fixture", task_prefix="FIX", stale_seconds=120,
        check_interval=10, strategy_path=tmp_path / "strategy.json", implementation_timeout=300,
        implementation_max_timeout=None, watchdog_startup_grace_seconds=30)
    owner._supervisor_status_path = lambda: tmp_path / "supervisor.json"
    owner._managed_daemon_pid_path = lambda: tmp_path / "daemon.pid"
    owner._proof_rollout_status_fields = lambda strategy=None: {}
    owner._autonomous_unstall_status = lambda: {}
    owner._control_plane_status_projection = lambda: {}
    monkeypatch.setattr(module, "utc_now", lambda: NOW.isoformat())
    return owner, module


def test_real_status_writer_binds_private_daemon_and_supervisor_birth(writer):
    owner, module = writer
    child = subprocess.Popen([sys.executable, "-B", "-c", "import sys; sys.stdin.buffer.read()"], stdin=subprocess.PIPE)
    try:
        birth = module.read_process_birth(child.pid)
        owner._write_supervisor_maintenance_status("fixture", status="running",
            started_at=(NOW - timedelta(seconds=20)).isoformat(), daemon_pid=child.pid,
            daemon_process_birth=birth)
        value = json.loads(owner._supervisor_status_path().read_text())
        assert value["daemon_pid_alive"] is True
        assert value["daemon_process_birth"] == birth.to_dict()
        assert value["supervisor_process_birth"] == module.read_process_birth(os.getpid()).to_dict()
        assert observe_supervisor_activity(value, now=NOW).phase_ready
        owner._write_supervisor_maintenance_status("fixture", status="failed", error="fixture failure",
            started_at=(NOW - timedelta(seconds=20)).isoformat(), daemon_pid=child.pid,
            daemon_process_birth=birth)
        failed = json.loads(owner._supervisor_status_path().read_text())
        assert failed["last_agentic_maintenance_error"] == "fixture failure"
        assert not observe_supervisor_activity(failed, now=NOW).phase_ready
    finally:
        child.stdin.close()
        child.wait(timeout=5)


def test_real_heartbeat_captures_window_once_before_config_or_phase_changes(writer, monkeypatch):
    owner, module = writer
    threads = []
    class Thread:
        def __init__(self, **kwargs): self.kwargs = kwargs; threads.append(self)
        def start(self): pass
        def join(self, **kwargs): pass
    monkeypatch.setattr(module.threading, "Thread", Thread)
    update, finish = owner._begin_supervisor_maintenance_heartbeat("first")
    first = json.loads(owner._supervisor_status_path().read_text())
    owner.config.implementation_timeout = 900
    monkeypatch.setattr(module, "utc_now", lambda: (NOW + timedelta(seconds=301)).isoformat())
    update("second")
    late = json.loads(owner._supervisor_status_path().read_text())
    assert late["supervisor_maintenance"] == first["supervisor_maintenance"]
    assert late["active_agentic_maintenance_timeout_seconds"] == 300
    assert late["updated_at"] != first["updated_at"]
    finish()
    done = json.loads(owner._supervisor_status_path().read_text())
    assert done["supervisor_maintenance"]["deadline_at"] == first["supervisor_maintenance"]["deadline_at"]
    assert done["supervisor_maintenance"]["phase"] == "completed"
    assert len(threads) == 1


@pytest.mark.parametrize("window", [False, {}, SupervisorMaintenanceWindow.begin(
    started_at=(NOW - timedelta(seconds=1)).isoformat(), timeout_seconds=300)])
def test_status_writer_refuses_untyped_or_different_original_window_without_overwrite(writer, window):
    owner, _ = writer
    path = owner._supervisor_status_path()
    path.write_text('{"retained":"original evidence"}\n')
    before = path.read_bytes()
    with pytest.raises(ValueError, match="original start"):
        owner._write_supervisor_maintenance_status("fixture", status="running",
            started_at=NOW.isoformat(), maintenance_window=window)
    assert path.read_bytes() == before
