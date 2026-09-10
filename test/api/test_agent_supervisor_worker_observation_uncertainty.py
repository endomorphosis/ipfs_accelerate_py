"""Unknown worker observations must not extend measured disappearance age."""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import current_process_birth
from ipfs_accelerate_py.agent_supervisor.todo_daemon import supervisor_loop as runtime
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import ManagedDaemonSpec


@pytest.mark.parametrize("failure", ["projection", "root_read", "root_absent", "census"])
def test_unknown_observation_breaks_disappearance_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    spec = ManagedDaemonSpec(
        name="observed-child", schema="test.observed-child", repo_root=tmp_path,
        daemon_dir=tmp_path, runner=("unused",), status_path=tmp_path / "child.json",
        supervisor_status_path=tmp_path / "supervisor.json",
        supervisor_pid_path=tmp_path / "supervisor.pid", child_pid_path=tmp_path / "child.pid",
        supervisor_out_path=tmp_path / "supervisor.out",
        ensure_status_path=tmp_path / "ensure.json", ensure_check_path=tmp_path / "check.json",
    )
    clock = [100.0]
    unavailable = [False]
    worker_count = [1]
    phase_timeout = [False]
    birth = current_process_birth()
    child = SimpleNamespace(pid=birth.pid, identity_process_birth=birth)
    status = {"heartbeat_at": datetime.now(timezone.utc).isoformat()}

    def projection(_child, _status):
        return None if unavailable[0] and failure == "projection" else {"active_phase": "implementing"}

    def read_birth(_pid):
        if unavailable[0] and failure == "root_read":
            raise OSError("injected read unavailable")
        return None if unavailable[0] and failure == "root_absent" else birth

    def descendants(_pid):
        if unavailable[0] and failure == "census":
            raise OSError("injected census unavailable")
        return []

    def measured(_status, _pid, _threshold, **_kwargs):
        # Stub only the phase census result. The loop's production projector,
        # birth sandwich, unavailable reporting and grace calculation run.
        return {
            "required": True, "phase": "implementing",
            "tracking_generation": "exact-attempt-generation",
            "active_worker_count": worker_count[0],
            "stalled_without_active_worker": phase_timeout[0],
        }

    monkeypatch.setattr(runtime, "read_process_birth", read_birth)
    monkeypatch.setattr(runtime, "procfs_descendant_processes", descendants)
    monkeypatch.setattr(runtime, "worktree_phase_worker_status", measured)
    loop = runtime.SupervisorLoop(
        runtime.SupervisorLoopConfig(
            spec=spec, command=("unused",), log_prefix="unused",
            worktree_status_projection=projection,
            status_static_fields={"worktree_no_child_stall_seconds": 30.0},
        ),
        monotonic=lambda: clock[0],
    )
    seen = loop._observe_worker_status(child, status)
    assert seen["worker_absence_age_seconds"] == 0.0
    assert loop._last_worker_seen_monotonic == 100.0

    clock[0] = 140.0
    unavailable[0] = True
    unknown = loop._observe_worker_status(child, status)
    assert unknown["worker_metrics_available"] is False
    assert unknown["stalled_without_active_worker"] is None
    assert loop.default_watchdog(child, status, worker_status=unknown).action == "continue"

    clock[0] = 150.0
    unavailable[0] = False
    worker_count[0] = 0
    absent = loop._observe_worker_status(child, status)
    assert loop.default_watchdog(child, status, worker_status=absent).action == "continue"
    assert absent["worker_metrics_available"] is True
    assert absent["worker_absence_age_seconds"] is None
    assert absent["stalled_without_active_worker"] is False
    assert loop._last_worker_seen_monotonic is None

    # An independently measured phase timeout remains actionable after the
    # unknown interval; only the unsupported disappearance interval is lost.
    phase_timeout[0] = True
    expired = loop._observe_worker_status(child, status)
    assert loop.default_watchdog(child, status, worker_status=expired).action == "recycle"

    phase_timeout[0] = False
    worker_count[0] = 1
    loop._observe_worker_status(child, status)
    worker_count[0] = 0
    clock[0] = 181.0
    fresh_disappearance = loop._observe_worker_status(child, status)
    assert fresh_disappearance["worker_absence_age_seconds"] == 31.0
    assert loop.default_watchdog(child, status, worker_status=fresh_disappearance).action == "recycle"
