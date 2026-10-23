from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_plane import LifecycleStatus
from ipfs_accelerate_py.agent_supervisor import supervisor_watchdog as watchdog_module
from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import (
    LIFECYCLE_STATUS_SCHEMA,
    SupervisorWatchdog,
    check_lane_heartbeat,
    lifecycle_status_projection,
    pid_alive,
    project_dynamic_manifest_lifecycle,
)


def _status(
    state: str,
    *,
    pid: int,
    heartbeat_at_ms: int | None = None,
) -> dict[str, Any]:
    now_ms = int(time.time() * 1000)
    return {
        "schema": LIFECYCLE_STATUS_SCHEMA,
        "target_id": "lane:test",
        "state": state,
        "phase": state,
        "heartbeat_at_ms": now_ms if heartbeat_at_ms is None else heartbeat_at_ms,
        "pid": pid,
        "active_leases": ["lease:b", "lease:a"],
        "refill_state": "idle",
        "backpressure": False,
        "backpressure_reasons": [],
        "terminal_reason": "clean shutdown" if state == "stopped" else "",
        "transition_id": "transition:1",
        "generation": 4,
        "fencing_epoch": 9,
        "updated_at_ms": now_ms,
    }


def _watchdog_fixture(
    tmp_path: Path,
    status: dict[str, Any],
    *,
    pid_file_value: int,
) -> tuple[SupervisorWatchdog, Path]:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    prefix = "agent_lane"
    pid_path = state_dir / f"{prefix}_bundle_supervisor.pid"
    pid_path.write_text(f"{pid_file_value}\n", encoding="utf-8")
    (state_dir / f"{prefix}_status.json").write_text(
        json.dumps(status),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "bundle_key": "lane:test",
                        "state_dir": str(state_dir),
                        "state_prefix": prefix,
                        "command": ["false"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return (
        SupervisorWatchdog(
            manifest_path=manifest_path,
            repo_root=tmp_path,
            lane_timeout=30,
        ),
        pid_path,
    )


def test_pid_probe_treats_permission_denial_as_alive_and_rejects_invalid_pids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied(_pid: int, _signal: int) -> None:
        raise PermissionError

    monkeypatch.setattr(os, "kill", denied)

    assert pid_alive(101)
    assert not pid_alive(0)
    assert not pid_alive(-1)


def test_proc_stat_state_ignores_parentheses_inside_the_command() -> None:
    assert watchdog_module._proc_stat_state("12 (python) S 1") == "S"
    assert watchdog_module._proc_stat_state("12 (name) with) paren) Z 1") == "Z"
    assert watchdog_module._proc_stat_state("no-paren") == ""


def test_pid_alive_is_false_for_a_zombie_child_and_reaps_it() -> None:
    if not Path("/proc").is_dir():
        pytest.skip("zombie detection needs /proc")
    child = subprocess.Popen([sys.executable, "-c", "import os; os._exit(0)"])
    deadline = time.time() + 2
    while time.time() < deadline:
        try:
            stat = Path(f"/proc/{child.pid}/stat").read_text(encoding="utf-8")
        except OSError:
            break
        if watchdog_module._proc_stat_state(stat) == "Z":
            break
        time.sleep(0.01)
    assert pid_alive(child.pid) is False
    assert not Path(f"/proc/{child.pid}").exists()
    child.wait(timeout=2)


def test_dynamic_manifest_lifecycle_projects_expired_and_dead_supervisors() -> None:
    manifest = {
        "schema": "ipfs_accelerate_py.agent_supervisor.dynamic_bundle_scheduler@1",
        "authoritative": True,
        "scheduler_state": "running",
        "supervisor_pid": 101,
        "heartbeat_at": "2026-01-01T00:00:00+00:00",
        "heartbeat_expires_at": "2026-01-01T00:01:00+00:00",
    }

    stale = project_dynamic_manifest_lifecycle(
        manifest,
        now_seconds=1_767_225_720.0,
        pid_probe=lambda _pid: True,
    )
    stopped = project_dynamic_manifest_lifecycle(
        manifest,
        now_seconds=1_767_225_620.0,
        pid_probe=lambda _pid: False,
    )

    assert stale["scheduler_state"] == "stale"
    assert stale["scheduler_state_declared"] == "running"
    assert stale["scheduler_state_reason"] == "heartbeat_expired"
    assert stale["supervisor_pid_alive"] is True
    assert stopped["scheduler_state"] == "stopped"
    assert stopped["scheduler_state_declared"] == "running"
    assert stopped["scheduler_state_reason"] == "supervisor_pid_not_running"
    assert stopped["supervisor_pid_alive"] is False
    assert manifest["scheduler_state"] == "running"


def test_dynamic_manifest_lifecycle_preserves_fresh_and_legacy_manifests() -> None:
    dynamic = {
        "schema": "ipfs_accelerate_py.agent_supervisor.dynamic_bundle_scheduler@1",
        "authoritative": True,
        "scheduler_state": "running",
        "supervisor_pid": 101,
        "heartbeat_expires_at": "2026-01-01T00:01:00+00:00",
    }
    legacy = {"scheduler_state": "running", "supervisor_pid": 101}

    fresh = project_dynamic_manifest_lifecycle(
        dynamic,
        now_seconds=1_767_225_620.0,
        pid_probe=lambda _pid: True,
    )

    assert fresh["scheduler_state"] == "running"
    assert fresh["supervisor_pid_alive"] is True
    assert "scheduler_state_declared" not in fresh
    assert project_dynamic_manifest_lifecycle(legacy) == legacy


def test_heartbeat_uses_canonical_timestamp_and_projects_exact_status_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    status_path = state_dir / "lane_status.json"
    status_path.write_text(
        json.dumps(_status("healthy", pid=101, heartbeat_at_ms=10_000)),
        encoding="utf-8",
    )
    # A recent mtime must not hide an old canonical heartbeat.
    monkeypatch.setattr(watchdog_module.time, "time", lambda: 100.0)

    heartbeat = check_lane_heartbeat(
        state_dir,
        "lane",
        timeout_seconds=30,
    )
    projection = lifecycle_status_projection(
        pid_check={"pid": 101, "alive": True},
        heartbeat_check=heartbeat,
        target_id="lane:test",
    )

    assert heartbeat["stale"] is True
    assert heartbeat["reason"] == "heartbeat_timeout"
    assert heartbeat["age_seconds"] == 90.0
    assert projection["schema"] == LIFECYCLE_STATUS_SCHEMA
    assert projection["state"] == "degraded"
    assert projection["active_leases"] == ["lease:a", "lease:b"]
    assert projection["active_lease_count"] == 2
    assert LifecycleStatus.from_dict(projection).to_dict() == projection


@pytest.mark.parametrize("state", ("blocked", "stopped", "failed"))
def test_canonical_non_running_states_are_never_raw_restarted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    status = _status(state, pid=111)
    if state == "failed":
        status["terminal_reason"] = "worker failed"
    supervisor, _pid_path = _watchdog_fixture(
        tmp_path,
        status,
        pid_file_value=111,
    )
    monkeypatch.setattr(watchdog_module, "pid_alive", lambda _pid: False)

    def unexpected_restart(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("canonical non-running state was raw restarted")

    monkeypatch.setattr(watchdog_module, "restart_lane", unexpected_restart)

    report = supervisor._check_cycle()
    lane = report["reports"][0]

    assert report["restarts"] == 0
    assert lane["action"] == "state_preserved"
    assert lane["status"]["state"] == state
    assert LifecycleStatus.from_dict(lane["status"]).to_dict() == lane["status"]


def test_progress_stall_is_quiet_live_work_without_completion_authority() -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import (
        classify_progress_stall,
    )

    stalled = classify_progress_stall(
        process_alive=True,
        heartbeat_stale=False,
        active_task_id="VDG-010",
        log_age_seconds=1_000,
        task_transition_age_seconds=1_000,
        quiet_seconds=900,
    )
    assert stalled["progress_stalled"] is True
    assert stalled["reason"] == "progress_stall_heartbeat_fresh"
    assert stalled["work_complete"] is False
    assert stalled["completion_authority"] is False
    assert classify_progress_stall(
        process_alive=True,
        heartbeat_stale=False,
        active_task_id="",
        log_age_seconds=1_000,
        task_transition_age_seconds=1_000,
        quiet_seconds=900,
    )["progress_stalled"] is False
    assert classify_progress_stall(
        process_alive=True,
        heartbeat_stale=False,
        active_task_id="VDG-010",
        log_age_seconds=10,
        task_transition_age_seconds=1_000,
        quiet_seconds=900,
    )["progress_stalled"] is False
    assert classify_progress_stall(
        process_alive=False,
        heartbeat_stale=False,
        active_task_id="VDG-010",
        log_age_seconds=1_000,
        task_transition_age_seconds=1_000,
        quiet_seconds=900,
    )["progress_stalled"] is False


def test_second_quarantine_writes_body_free_repair_and_stops_restarting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now_ms = int(time.time() * 1000)
    status = _status("running", pid=os.getpid())
    status["active_task_id"] = "VDG-010"
    status["last_task_transition_at_ms"] = now_ms - 2_000_000
    supervisor, _pid_path = _watchdog_fixture(tmp_path, status, pid_file_value=os.getpid())
    supervisor.progress_quiet_seconds = 900
    log_path = tmp_path / "lane.log"
    log_path.write_text("quiet\n", encoding="utf-8")
    os.utime(log_path, (time.time() - 2_000, time.time() - 2_000))
    manifest = json.loads(supervisor.manifest_path.read_text(encoding="utf-8"))
    manifest["lanes"][0]["log_path"] = str(log_path)
    manifest["unstall_repair_dir"] = str(tmp_path / "unstall-repair")
    manifest["report_path"] = str(tmp_path / "watchdog-report.json")
    supervisor.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def quarantined(**_kwargs: Any) -> dict[str, Any]:
        return {
            "recovered": False,
            "quarantined": True,
            "incident_cid": "incident-progress-1",
            "reason": "progress_stall_heartbeat_fresh",
            "missing_handler": "progress_stall_rearm",
            "work_complete": False,
            "completion_authority": False,
        }

    monkeypatch.setattr(supervisor, "_watchdog_unstall", quarantined)
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog.pid_alive",
        lambda _pid: True,
    )

    first = supervisor._check_cycle()["reports"][0]
    second = supervisor._check_cycle()["reports"][0]
    packet = json.loads(Path(second["repair_packet"]).read_text(encoding="utf-8"))
    written = json.loads((tmp_path / "watchdog-report.json").read_text(encoding="utf-8"))

    assert first["action"] == "autonomous_unstall_quarantined"
    assert first["quarantine_count"] == 1
    assert second["action"] == "repeated_quarantine_repair"
    assert second["quarantine_count"] == 2
    assert packet["body_free"] is True
    assert packet["implementable"] is False
    assert packet["work_complete"] is False
    assert packet["completion_authority"] is False
    assert packet["missing_handler"] == "progress_stall_rearm"
    assert "source" not in packet
    assert written["work_complete"] is False
    assert written["completion_authority"] is False


@pytest.mark.parametrize(
    ("state", "recovered_state"),
    (("paused", "failed"), ("draining", "failed"), ("stopping", "stopped")),
)
def test_dead_intentional_state_requests_control_recovery_without_raw_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    recovered_state: str,
) -> None:
    supervisor, _pid_path = _watchdog_fixture(
        tmp_path,
        _status(state, pid=111),
        pid_file_value=111,
    )
    monkeypatch.setattr(watchdog_module, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(
        watchdog_module,
        "restart_lane",
        lambda *_args, **_kwargs: pytest.fail("interrupted canonical state was raw restarted"),
    )

    report = supervisor._check_cycle()
    lane = report["reports"][0]

    assert report["restarts"] == 0
    assert lane["action"] == "control_recovery_required"
    assert lane["recovery"] == {
        "kind": "interrupted_transition",
        "previous_state": state,
        "recovered_state": recovered_state,
    }
    assert lane["status"]["state"] == recovered_state
    assert lane["status"]["terminal_reason"]


def test_live_canonical_status_repairs_stale_pid_file_without_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor, pid_path = _watchdog_fixture(
        tmp_path,
        _status("healthy", pid=222),
        pid_file_value=111,
    )
    monkeypatch.setattr(
        watchdog_module,
        "pid_alive",
        lambda pid: pid == 222,
    )

    def unexpected_restart(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("reconciled live PID was restarted")

    monkeypatch.setattr(watchdog_module, "restart_lane", unexpected_restart)

    report = supervisor._check_cycle()
    lane = report["reports"][0]

    assert report["restarts"] == 0
    assert pid_path.read_text(encoding="utf-8").strip() == "222"
    assert lane["action"] == "none"
    assert lane["recovery"] == {
        "kind": "stale_pid_file",
        "previous_pid": 111,
        "recovered_pid": 222,
    }
    assert lane["status"]["state"] == "healthy"


def test_live_stale_canonical_process_requires_fenced_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_ms = int((time.time() - 300) * 1000)
    supervisor, _pid_path = _watchdog_fixture(
        tmp_path,
        _status("healthy", pid=111, heartbeat_at_ms=stale_ms),
        pid_file_value=111,
    )
    monkeypatch.setattr(watchdog_module, "pid_alive", lambda pid: pid == 111)

    def unexpected_restart(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("live process was replaced without fencing")

    monkeypatch.setattr(watchdog_module, "restart_lane", unexpected_restart)

    report = supervisor._check_cycle()
    lane = report["reports"][0]

    assert report["restarts"] == 0
    assert lane["action"] == "fenced_stop_required"
    assert lane["status"]["state"] == "blocked"


@pytest.mark.parametrize("state", ("paused", "draining", "blocked", "stopping"))
def test_live_stale_intentional_state_still_requires_fenced_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    stale_ms = int((time.time() - 300) * 1000)
    supervisor, _pid_path = _watchdog_fixture(
        tmp_path,
        _status(state, pid=111, heartbeat_at_ms=stale_ms),
        pid_file_value=111,
    )
    monkeypatch.setattr(watchdog_module, "pid_alive", lambda pid: pid == 111)
    monkeypatch.setattr(
        watchdog_module,
        "restart_lane",
        lambda *_args, **_kwargs: pytest.fail("stale live process was replaced without fencing"),
    )

    report = supervisor._check_cycle()
    lane = report["reports"][0]

    assert report["restarts"] == 0
    assert lane["action"] == "fenced_stop_required"
    assert lane["status"]["state"] == "blocked"


@pytest.mark.parametrize("state", ("stopped", "failed"))
def test_live_process_in_terminal_state_requires_fenced_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    status = _status(state, pid=111)
    status["terminal_reason"] = f"{state} but process remains"
    supervisor, _pid_path = _watchdog_fixture(
        tmp_path,
        status,
        pid_file_value=111,
    )
    monkeypatch.setattr(watchdog_module, "pid_alive", lambda pid: pid == 111)
    monkeypatch.setattr(
        watchdog_module,
        "restart_lane",
        lambda *_args, **_kwargs: pytest.fail("terminal live process was replaced without fencing"),
    )

    report = supervisor._check_cycle()
    lane = report["reports"][0]

    assert report["restarts"] == 0
    assert lane["action"] == "fenced_stop_required"
    assert lane["reason"] == "terminal_state_pid_alive"
    assert lane["status"]["state"] == "blocked"


def test_dead_starting_process_recovers_interrupted_transition_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor, pid_path = _watchdog_fixture(
        tmp_path,
        _status("starting", pid=111),
        pid_file_value=111,
    )
    monkeypatch.setattr(watchdog_module, "pid_alive", lambda _pid: False)
    restarts: list[dict[str, Any]] = []

    def restart(
        lane_info: dict[str, Any],
        *,
        repo_root: Path,
    ) -> dict[str, Any]:
        restarts.append(dict(lane_info))
        assert repo_root == tmp_path
        return {"restarted": True, "new_pid": 333, "pid_persisted": True}

    monkeypatch.setattr(watchdog_module, "restart_lane", restart)

    report = supervisor._check_cycle()
    lane = report["reports"][0]

    assert report["restarts"] == 1
    assert len(restarts) == 1
    assert restarts[0]["pid_path"] == str(pid_path)
    assert lane["action"] == "restarted"
    assert lane["recovery"] == {
        "kind": "interrupted_transition",
        "previous_state": "starting",
        "restarted_pid": 333,
    }
    assert lane["status"]["state"] == "starting"
    assert report["status"]["schema"] == LIFECYCLE_STATUS_SCHEMA
    assert report["status"]["state"] == "starting"
    assert LifecycleStatus.from_dict(report["status"]).to_dict() == report["status"]


_PENDING_BOARD = """
## VDG-001 Seal
- Status: completed
- Depends on:

## VDG-010 Ledger
- Status: pending
- Depends on: VDG-001

## VDG-012 Join
- Status: pending
- Depends on: VDG-010

## VDG-051 Rearm
- Status: blocked
- Depends on: VDG-001
"""


def _bind_board(
    supervisor: SupervisorWatchdog,
    tmp_path: Path,
    *,
    todo_text: str | None,
    master_pid: int | None = None,
) -> None:
    manifest = json.loads(supervisor.manifest_path.read_text(encoding="utf-8"))
    manifest["unstall_repair_dir"] = str(tmp_path / "unstall-repair")
    manifest["report_path"] = str(tmp_path / "watchdog-report.json")
    if todo_text is None:
        manifest.pop("todo_path", None)
    else:
        todo_path = tmp_path / "board.md"
        todo_path.write_text(todo_text, encoding="utf-8")
        manifest["todo_path"] = str(todo_path)
        manifest["task_header_prefix"] = "## VDG-"
    if master_pid is None:
        manifest["master_pid_path"] = str(tmp_path / "missing-master.pid")
    else:
        master_path = tmp_path / "master.pid"
        master_path.write_text(f"{master_pid}\n", encoding="utf-8")
        manifest["master_pid_path"] = str(master_path)
    supervisor.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_selectable_pending_tasks_ignore_blocked_and_unmet_dependencies() -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog import (
        classify_stopped_pending_stall,
        selectable_pending_task_ids,
    )

    selected = selectable_pending_task_ids(
        _PENDING_BOARD,
        task_header_prefix="## VDG-",
    )
    assert selected == ["VDG-010"]
    assert selectable_pending_task_ids("", task_header_prefix="## VDG-") == []
    stalled = classify_stopped_pending_stall(
        process_alive=False,
        master_alive=False,
        heartbeat_stale=True,
        observed_state="stopped",
        selectable_pending_task_ids=selected,
    )
    assert stalled["stopped_pending_stalled"] is True
    assert stalled["reason"] == "stopped_with_pending_work"
    assert stalled["work_complete"] is False
    assert stalled["completion_authority"] is False
    assert classify_stopped_pending_stall(
        process_alive=False,
        master_alive=False,
        heartbeat_stale=False,
        observed_state="stopped",
        selectable_pending_task_ids=selected,
    )["stopped_pending_stalled"] is False
    assert classify_stopped_pending_stall(
        process_alive=True,
        master_alive=False,
        heartbeat_stale=True,
        observed_state="stopped",
        selectable_pending_task_ids=selected,
    )["stopped_pending_stalled"] is False
    assert classify_stopped_pending_stall(
        process_alive=False,
        master_alive=True,
        heartbeat_stale=True,
        observed_state="failed",
        selectable_pending_task_ids=selected,
    )["stopped_pending_stalled"] is False
    assert classify_stopped_pending_stall(
        process_alive=False,
        master_alive=False,
        heartbeat_stale=True,
        observed_state="blocked",
        selectable_pending_task_ids=selected,
    )["stopped_pending_stalled"] is False
    assert classify_stopped_pending_stall(
        process_alive=False,
        master_alive=False,
        heartbeat_stale=True,
        observed_state="stopped",
        selectable_pending_task_ids=[],
    )["stopped_pending_stalled"] is False
    assert classify_stopped_pending_stall(
        process_alive=False,
        master_alive=False,
        heartbeat_stale=True,
        observed_state="stopped",
        selectable_pending_task_ids=selected,
        terminal_reason="resume_stopped",
    )["stopped_pending_stalled"] is False
    assert classify_stopped_pending_stall(
        process_alive=False,
        master_alive=False,
        heartbeat_stale=True,
        observed_state="stopped",
        selectable_pending_task_ids=selected,
        terminal_reason="resume_duration_elapsed",
    )["stopped_pending_stalled"] is True
    assert classify_stopped_pending_stall(
        process_alive=False,
        master_alive=False,
        heartbeat_stale=False,
        observed_state="stopped",
        selectable_pending_task_ids=selected,
        terminal_reason="resume_duration_elapsed",
    )["stopped_pending_stalled"] is True
    assert classify_stopped_pending_stall(
        process_alive=True,
        master_alive=False,
        heartbeat_stale=False,
        observed_state="stopped",
        selectable_pending_task_ids=selected,
        terminal_reason="resume_duration_elapsed",
    )["stopped_pending_stalled"] is False
    assert classify_stopped_pending_stall(
        process_alive=False,
        master_alive=False,
        heartbeat_stale=True,
        observed_state="degraded",
        selectable_pending_task_ids=selected,
    )["stopped_pending_stalled"] is True
    assert classify_stopped_pending_stall(
        process_alive=True,
        master_alive=False,
        heartbeat_stale=True,
        observed_state="degraded",
        selectable_pending_task_ids=selected,
    )["stopped_pending_stalled"] is False


@pytest.mark.parametrize("state", ("stopped", "failed"))
def test_stale_stopped_lane_with_pending_work_resumes_once_then_writes_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    stale_ms = int((time.time() - 1_000) * 1000)
    status = _status(state, pid=111, heartbeat_at_ms=stale_ms)
    supervisor, _pid_path = _watchdog_fixture(tmp_path, status, pid_file_value=111)
    _bind_board(supervisor, tmp_path, todo_text=_PENDING_BOARD)
    calls: list[dict[str, Any]] = []

    def resume(lane_info: dict[str, Any], *, repo_root: Path, manifest: Any = None) -> dict[str, Any]:
        del manifest
        calls.append(dict(lane_info))
        assert repo_root == tmp_path
        return {
            "restarted": False,
            "reason": "resume_spawn_failed: test",
            "control_recovery_required": True,
            "work_complete": False,
            "completion_authority": False,
        }

    monkeypatch.setattr(watchdog_module, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(watchdog_module, "resume_stopped_lane", resume)

    first = supervisor._check_cycle()
    second = supervisor._check_cycle()
    third = supervisor._check_cycle()
    first_lane = first["reports"][0]
    second_lane = second["reports"][0]
    packet = json.loads(Path(second_lane["repair_packet"]).read_text(encoding="utf-8"))
    written = json.loads((tmp_path / "watchdog-report.json").read_text(encoding="utf-8"))

    assert len(calls) == 1
    assert calls[0]["unstall_class"] == "stopped_with_pending_work"
    assert calls[0]["exact_source_worktree"] is True
    assert first["restarts"] == 0
    assert first["work_complete"] is False
    assert first["completion_authority"] is False
    assert first_lane["action"] == "restart_failed"
    assert first_lane["reason"] == "resume_spawn_failed: test"
    assert first_lane["status"]["work_complete"] is False
    assert first_lane["status"]["completion_authority"] is False
    assert first_lane["heartbeat_check"]["stopped_pending_stall"]["selectable_pending_task_ids"] == [
        "VDG-010"
    ]
    assert second_lane["action"] == "repeated_stopped_pending_repair"
    assert second_lane["reason"] == "stopped_pending_resume_already_used"
    assert third["reports"][0]["action"] == "repeated_stopped_pending_repair"
    assert packet["body_free"] is True
    assert packet["implementable"] is False
    assert packet["missing_handler"] == "resume_stopped_lane"
    assert packet["work_complete"] is False
    assert packet["completion_authority"] is False
    assert "source" not in packet
    assert written["work_complete"] is False
    assert written["completion_authority"] is False


def test_attempt_backoff_exit_resumes_twice_then_packets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_ms = int((time.time() - 1_000) * 1000)
    status = _status("blocked", pid=111, heartbeat_at_ms=stale_ms)
    status["phase"] = "deterministic_resume_unavailable"
    status["terminal_reason"] = "deterministic_resume_unavailable"
    status["backpressure"] = True
    status["backpressure_reasons"] = ["VDG-010:attempts_exhausted"]
    supervisor, _pid_path = _watchdog_fixture(tmp_path, status, pid_file_value=111)
    _bind_board(supervisor, tmp_path, todo_text=_PENDING_BOARD)
    calls: list[int] = []

    def resume(_lane_info: dict[str, Any], *, repo_root: Path, manifest: Any = None) -> dict[str, Any]:
        del repo_root, manifest
        calls.append(1)
        return {
            "restarted": True,
            "new_pid": 4242,
            "reason": "resume_stopped_lane",
            "work_complete": False,
            "completion_authority": False,
        }

    monkeypatch.setattr(watchdog_module, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(watchdog_module, "resume_stopped_lane", resume)

    first = supervisor._check_cycle()["reports"][0]
    second = supervisor._check_cycle()["reports"][0]
    third = supervisor._check_cycle()["reports"][0]

    assert len(calls) == 2
    assert first["action"] == "restarted"
    assert second["action"] == "restarted"
    assert second["reason"] == "resume_stopped_lane"
    assert third["action"] == "repeated_stopped_pending_repair"
    assert third["reason"] == "stopped_pending_resume_already_used"
    assert third["status"]["work_complete"] is False
    assert third["status"]["completion_authority"] is False


def test_blank_command_block_still_packets_on_the_second_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_ms = int((time.time() - 1_000) * 1000)
    status = _status("blocked", pid=111, heartbeat_at_ms=stale_ms)
    status["phase"] = "deterministic_resume_unavailable"
    status["terminal_reason"] = "deterministic_resume_unavailable"
    status["backpressure"] = True
    status["backpressure_reasons"] = ["VDG-010:no_deterministic_command"]
    supervisor, _pid_path = _watchdog_fixture(tmp_path, status, pid_file_value=111)
    _bind_board(supervisor, tmp_path, todo_text=_PENDING_BOARD)
    calls: list[int] = []

    def resume(_lane_info: dict[str, Any], *, repo_root: Path, manifest: Any = None) -> dict[str, Any]:
        del repo_root, manifest
        calls.append(1)
        return {
            "restarted": False,
            "reason": "resume_spawn_failed: test",
            "work_complete": False,
            "completion_authority": False,
        }

    monkeypatch.setattr(watchdog_module, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(watchdog_module, "resume_stopped_lane", resume)

    first = supervisor._check_cycle()["reports"][0]
    second = supervisor._check_cycle()["reports"][0]

    assert len(calls) == 1
    assert first["action"] == "restart_failed"
    assert second["action"] == "repeated_stopped_pending_repair"
    assert second["status"]["work_complete"] is False
    assert second["status"]["completion_authority"] is False


def test_fresh_stopped_lane_with_pending_work_stays_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor, _pid_path = _watchdog_fixture(
        tmp_path,
        _status("stopped", pid=111),
        pid_file_value=111,
    )
    _bind_board(supervisor, tmp_path, todo_text=_PENDING_BOARD)
    monkeypatch.setattr(watchdog_module, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(
        watchdog_module,
        "resume_stopped_lane",
        lambda *_args, **_kwargs: pytest.fail("fresh canonical stop was resumed"),
    )

    lane = supervisor._check_cycle()["reports"][0]

    assert lane["action"] == "state_preserved"
    assert lane["reason"] == "canonical_stopped"
    assert lane["heartbeat_check"]["stopped_pending_stall"]["stopped_pending_stalled"] is False


def test_stale_stopped_lane_without_selectable_work_stays_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_ms = int((time.time() - 1_000) * 1000)
    supervisor, _pid_path = _watchdog_fixture(
        tmp_path,
        _status("stopped", pid=111, heartbeat_at_ms=stale_ms),
        pid_file_value=111,
    )
    _bind_board(
        supervisor,
        tmp_path,
        todo_text="## VDG-051 Rearm\n- Status: blocked\n- Depends on:\n",
    )
    monkeypatch.setattr(watchdog_module, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(
        watchdog_module,
        "resume_stopped_lane",
        lambda *_args, **_kwargs: pytest.fail("lane without selectable work was resumed"),
    )

    lane = supervisor._check_cycle()["reports"][0]

    assert lane["action"] == "state_preserved"
    assert lane["reason"] == "canonical_stopped"


def test_stale_stopped_lane_with_live_master_is_not_raw_resumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_ms = int((time.time() - 1_000) * 1000)
    supervisor, _pid_path = _watchdog_fixture(
        tmp_path,
        _status("stopped", pid=111, heartbeat_at_ms=stale_ms),
        pid_file_value=111,
    )
    _bind_board(supervisor, tmp_path, todo_text=_PENDING_BOARD, master_pid=os.getpid())
    monkeypatch.setattr(watchdog_module, "pid_alive", lambda pid: pid == os.getpid())
    monkeypatch.setattr(
        watchdog_module,
        "resume_stopped_lane",
        lambda *_args, **_kwargs: pytest.fail("live master lane was raw resumed"),
    )

    lane = supervisor._check_cycle()["reports"][0]

    assert lane["action"] == "state_preserved"
    assert lane["reason"] == "canonical_stopped"


def test_fenced_lifecycle_resume_does_not_repeat_or_grant_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_ms = int((time.time() - 1_000) * 1000)
    status = _status("stopped", pid=111, heartbeat_at_ms=stale_ms)
    supervisor, _pid_path = _watchdog_fixture(tmp_path, status, pid_file_value=111)
    _bind_board(supervisor, tmp_path, todo_text=_PENDING_BOARD)
    calls: list[Mapping[str, Any]] = []

    def lifecycle(lane_info: Mapping[str, Any]) -> dict[str, Any]:
        calls.append(dict(lane_info))
        return {"restarted": True, "new_pid": 5150}

    supervisor.lifecycle_restart = lifecycle
    monkeypatch.setattr(watchdog_module, "pid_alive", lambda _pid: False)

    first = supervisor._check_cycle()
    second = supervisor._check_cycle()
    first_lane = first["reports"][0]

    assert len(calls) == 1
    assert calls[0]["unstall_class"] == "stopped_with_pending_work"
    assert first["restarts"] == 1
    assert first["work_complete"] is False
    assert first["completion_authority"] is False
    assert first_lane["action"] == "restarted"
    assert first_lane["status"]["state"] == "starting"
    assert first_lane["status"]["work_complete"] is False
    assert second["reports"][0]["action"] == "repeated_stopped_pending_repair"
    assert second["restarts"] == 0
    assert second["work_complete"] is False
    assert second["completion_authority"] is False


def _duration_status(pid: int, *, terminal_reason: str, stale_ms: int) -> dict[str, Any]:
    body = _status("stopped", pid=pid, heartbeat_at_ms=stale_ms)
    body["phase"] = "lane_resume_exited"
    body["terminal_reason"] = terminal_reason
    body["backpressure_reasons"] = []
    return body


def test_duration_expiry_resumes_once_per_exited_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_ms = int((time.time() - 1_000) * 1000)
    status_path = tmp_path / "state" / "agent_lane_status.json"
    supervisor, _pid_path = _watchdog_fixture(
        tmp_path,
        _duration_status(111, terminal_reason="resume_duration_elapsed", stale_ms=stale_ms),
        pid_file_value=111,
    )
    _bind_board(supervisor, tmp_path, todo_text=_PENDING_BOARD)
    calls: list[dict[str, Any]] = []

    def resume(lane_info: dict[str, Any], *, repo_root: Path, manifest: Any = None) -> dict[str, Any]:
        del manifest
        calls.append(dict(lane_info))
        assert repo_root == tmp_path
        return {
            "restarted": False,
            "reason": "resume_spawn_failed: test",
            "control_recovery_required": True,
            "work_complete": False,
            "completion_authority": False,
        }

    monkeypatch.setattr(watchdog_module, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(watchdog_module, "resume_stopped_lane", resume)

    same_pid_again = supervisor._check_cycle()
    repeated = supervisor._check_cycle()
    status_path.write_text(
        json.dumps(
            _duration_status(222, terminal_reason="resume_duration_elapsed", stale_ms=stale_ms)
        ),
        encoding="utf-8",
    )
    next_budget = supervisor._check_cycle()
    next_budget_repeated = supervisor._check_cycle()

    assert len(calls) == 2
    assert same_pid_again["reports"][0]["action"] == "restart_failed"
    assert repeated["reports"][0]["action"] == "repeated_stopped_pending_repair"
    assert repeated["reports"][0]["reason"] == "stopped_pending_resume_already_used"
    assert next_budget["reports"][0]["action"] == "restart_failed"
    assert next_budget_repeated["reports"][0]["action"] == "repeated_stopped_pending_repair"
    assert same_pid_again["work_complete"] is False
    assert next_budget["completion_authority"] is False


def test_duration_expiry_resumes_while_heartbeat_is_inside_lane_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_ms = int((time.time() - 5) * 1000)
    supervisor, _pid_path = _watchdog_fixture(
        tmp_path,
        _duration_status(111, terminal_reason="resume_duration_elapsed", stale_ms=fresh_ms),
        pid_file_value=111,
    )
    _bind_board(supervisor, tmp_path, todo_text=_PENDING_BOARD)
    calls: list[int] = []

    def resume(lane_info: dict[str, Any], *, repo_root: Path, manifest: Any = None) -> dict[str, Any]:
        del lane_info, repo_root, manifest
        calls.append(1)
        return {
            "restarted": True,
            "new_pid": 4242,
            "reason": "resume_stopped_lane",
            "work_complete": False,
            "completion_authority": False,
        }

    monkeypatch.setattr(watchdog_module, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(watchdog_module, "resume_stopped_lane", resume)

    cycle = supervisor._check_cycle()
    lane = cycle["reports"][0]

    assert lane["heartbeat_check"]["stale"] is False
    assert lane["action"] == "restarted"
    assert len(calls) == 1
    assert cycle["work_complete"] is False
    assert cycle["completion_authority"] is False


def test_non_duration_stop_resumes_once_when_the_pid_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_ms = int((time.time() - 1_000) * 1000)
    status_path = tmp_path / "state" / "agent_lane_status.json"
    supervisor, _pid_path = _watchdog_fixture(
        tmp_path,
        _duration_status(111, terminal_reason="clean shutdown", stale_ms=stale_ms),
        pid_file_value=111,
    )
    _bind_board(supervisor, tmp_path, todo_text=_PENDING_BOARD)
    calls: list[int] = []

    def resume(lane_info: dict[str, Any], *, repo_root: Path, manifest: Any = None) -> dict[str, Any]:
        del lane_info, repo_root, manifest
        calls.append(1)
        return {
            "restarted": False,
            "reason": "resume_spawn_failed: test",
            "control_recovery_required": True,
            "work_complete": False,
            "completion_authority": False,
        }

    monkeypatch.setattr(watchdog_module, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(watchdog_module, "resume_stopped_lane", resume)

    supervisor._check_cycle()
    status_path.write_text(
        json.dumps(_duration_status(333, terminal_reason="clean shutdown", stale_ms=stale_ms)),
        encoding="utf-8",
    )
    changed_pid = supervisor._check_cycle()

    assert calls == [1]
    assert changed_pid["reports"][0]["action"] == "repeated_stopped_pending_repair"
    assert changed_pid["work_complete"] is False
    assert changed_pid["completion_authority"] is False


def test_operator_sigterm_with_pending_work_is_not_resumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_ms = int((time.time() - 1_000) * 1000)
    supervisor, _pid_path = _watchdog_fixture(
        tmp_path,
        _duration_status(111, terminal_reason="resume_stopped", stale_ms=stale_ms),
        pid_file_value=111,
    )
    _bind_board(supervisor, tmp_path, todo_text=_PENDING_BOARD)
    monkeypatch.setattr(watchdog_module, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(
        watchdog_module,
        "resume_stopped_lane",
        lambda *_args, **_kwargs: pytest.fail("operator SIGTERM was resumed"),
    )

    lane = supervisor._check_cycle()["reports"][0]

    assert lane["action"] == "state_preserved"
    assert lane["reason"] == "canonical_stopped"
    assert lane["heartbeat_check"]["stopped_pending_stall"]["stopped_pending_stalled"] is False


def test_unchanged_rendered_census_refills_one_live_dom_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume

    census = (
        tmp_path
        / "swissknife/test-results/virtual-desktop-ipfs-mcp-orb/vog/rendered-control-census.json"
    )
    census.parent.mkdir(parents=True)
    census.write_text(
        '{"schema":"swissknife.rendered-control-census.v1","controls":[]}\n',
        encoding="utf-8",
    )
    board = tmp_path / "board.md"
    board.write_text(
        "\n".join(
            [
                "## VOG-001 Seal",
                "- Status: completed",
                "- Depends on:",
                "",
                "## VOG-012 Run the rendered-control census",
                "- Status: pending",
                "- Depends on: VOG-001",
                "- Goal id: VOG-G011",
                "- Deterministic command: node swissknife/scripts/build-rendered-control-census.cjs",
                "- Validation: python3 swissknife/scripts/vog_rendered_census_requires_live_dom.py",
                "- Predicted files: swissknife/test-results/virtual-desktop-ipfs-mcp-orb/vog/rendered-control-census.json",
                "- Implementation timeout seconds: 7200",
                "- Resource class: cpu-browser",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "attempts.json").write_text(json.dumps({"VOG-012": 3}), encoding="utf-8")

    def fake(argv: list[str], **_kwargs: Any) -> int:
        joined = " ".join(argv)
        if "build-rendered-control-census" in joined:
            census.write_text(
                '{"schema":"swissknife.rendered-control-census.v1","controls":[]}\n',
                encoding="utf-8",
            )
            return 0
        if "run-rendered-census" in joined:
            rows = [
                {
                    "app_id": f"app-{index}",
                    "launch_id": f"app-{index}",
                    "selector": ".window-control.close",
                    "role": "button",
                    "accessible_name": "Close",
                }
                for index in range(45)
            ]
            census.write_text(
                json.dumps(
                    {
                        "schema": "swissknife.rendered-control-census.v1",
                        "source_template": False,
                        "clicked": False,
                        "rows": rows,
                    }
                ),
                encoding="utf-8",
            )
            return 0
        if "vog_rendered_census" in joined:
            payload = json.loads(census.read_text(encoding="utf-8"))
            return 0 if len(payload.get("rows") or []) >= 45 else 1
        return 1

    monkeypatch.setattr(lane_resume, "_run_argv", fake)
    kwargs = {
        "repo_root": tmp_path,
        "todo_path": board,
        "task_header_prefix": "## VOG-",
        "attempt_path": tmp_path / "attempts.json",
        "log_path": tmp_path / "lane.log",
        "heartbeat": lambda **_kwargs: None,
        "allow_bounded_implement": False,
    }
    first = lane_resume.run_resume_cycle(**kwargs)
    text = board.read_text(encoding="utf-8")
    assert first["phase"] == "gap_refilled"
    assert first["refill_state"] == "admitted"
    assert first["work_complete"] is False
    assert first["completion_authority"] is False
    assert text.count("## VOG-R012 ") == 1
    assert "VOG-G011-R" in text
    assert "node swissknife/scripts/run-rendered-census.mjs" in text
    assert "node swissknife/scripts/build-rendered-control-census.cjs" in text
    second = lane_resume.run_resume_cycle(**kwargs)
    final = board.read_text(encoding="utf-8")
    assert second["phase"] == "task_command_passed"
    assert second["advanced_task_id"] == "VOG-R012"
    assert second["work_complete"] is False
    assert final.count("## VOG-R012 ") == 1
    source = final.split("## VOG-R012 ", 1)[0]
    assert "- Status: completed" in source
    assert "node swissknife/scripts/build-rendered-control-census.cjs" in source


def test_completed_join_refills_when_gateway_bindings_stay_unbound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume

    matrix = (
        tmp_path
        / "swissknife/test-results/virtual-desktop-ipfs-mcp-orb/vog/orb-idl-control-matrix.json"
    )
    matrix.parent.mkdir(parents=True)
    matrix.write_text(
        json.dumps(
            {
                "schema": "swissknife.orb-idl-control-matrix.v1",
                "rows": [
                    {
                        "app_id": "image-viewer",
                        "selector": '[data-live-gateway-binding="image-viewer.ipfs_kit_py.retrieve_content"]',
                        "disposition": "unsupported",
                        "idl_method": "none",
                        "tool_id": "",
                        "reason": "no-tool-binding",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    board = tmp_path / "board.md"
    board.write_text(
        "\n".join(
            [
                "## VOG-014 Run the ORB/IDL join",
                "- Status: completed",
                "- Depends on: VOG-012",
                "- Goal id: VOG-G012",
                "- Deterministic command: node swissknife/scripts/build-orb-idl-control-matrix.cjs",
                "- Validation: python3 swissknife/scripts/vog_orb_idl_matrix_requires_join.py",
                "- Predicted files: swissknife/test-results/virtual-desktop-ipfs-mcp-orb/vog/orb-idl-control-matrix.json",
                "- Implementation timeout seconds: 1800",
                "- Resource class: cpu-small",
                "",
            ]
        ),
        encoding="utf-8",
    )

    def fail_if_called(argv: list[str], **_kwargs: Any) -> int:
        raise AssertionError(argv)

    monkeypatch.setattr(lane_resume, "_run_argv", fail_if_called)
    cycle = lane_resume.run_resume_cycle(
        repo_root=tmp_path,
        todo_path=board,
        task_header_prefix="## VOG-",
        attempt_path=tmp_path / "attempts.json",
        log_path=tmp_path / "lane.log",
        heartbeat=lambda **_kwargs: None,
    )
    text = board.read_text(encoding="utf-8")
    assert cycle["phase"] == "gap_refilled"
    assert cycle["refill_state"] == "admitted"
    assert cycle["active_task_id"] == "VOG-R014"
    assert cycle["work_complete"] is False
    assert cycle["completion_authority"] is False
    assert text.count("## VOG-R014 ") == 1
    assert "VOG-G012-R" in text
    assert "Satisfies: VOG-014" in text
    assert "node swissknife/scripts/build-orb-idl-control-matrix.cjs" in text
    source = text.split("## VOG-R014 ", 1)[0]
    assert "- Status: completed" in source
    assert "node swissknife/scripts/build-orb-idl-control-matrix.cjs" in source


def test_missing_module_repair_installs_declared_package_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume

    package = tmp_path / "swissknife"
    script_dir = package / "scripts"
    script_dir.mkdir(parents=True)
    (package / "package.json").write_text(
        json.dumps({"devDependencies": {"tsx": "^4.0.0"}}),
        encoding="utf-8",
    )
    (package / "package-lock.json").write_text("{}\n", encoding="utf-8")
    script = script_dir / "build.cjs"
    script.write_text("// ledger\n", encoding="utf-8")
    log_path = tmp_path / "lane.log"
    log_path.write_text("Error: Cannot find module 'tsx/cjs'\n", encoding="utf-8")
    calls: list[tuple[list[str], Path]] = []

    def run(argv: list[str], **kwargs: Any) -> int:
        calls.append((list(argv), Path(kwargs["cwd"]) if kwargs.get("cwd") else tmp_path))
        return 0

    monkeypatch.setattr(lane_resume, "_run_argv", run)
    assert lane_resume.repair_missing_module(
        ["node", str(script)],
        repo_root=tmp_path,
        log_path=log_path,
        attempt_path=tmp_path / "attempts.json",
        timeout=30,
        heartbeat=lambda **_kwargs: None,
    )
    assert calls == [(["npm", "ci", "--ignore-scripts"], package.resolve())]
    assert lane_resume.repair_missing_module(
        ["node", str(script)],
        repo_root=tmp_path,
        log_path=log_path,
        attempt_path=tmp_path / "attempts.json",
        timeout=30,
        heartbeat=lambda **_kwargs: None,
    ) is False
    assert len(calls) == 1
    (package / "package.json").write_text("{}\n", encoding="utf-8")
    other_log = tmp_path / "other.log"
    other_log.write_text("Error: Cannot find module 'left-pad'\n", encoding="utf-8")
    assert lane_resume.missing_module_repair_pending(
        ["node", str(script)],
        repo_root=tmp_path,
        log_path=other_log,
        attempt_path=tmp_path / "no-such-attempts.json",
    ) is False


def test_missing_named_export_is_restored_once_then_the_command_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume

    repo = tmp_path
    spec_rel = "swissknife/test/e2e/virtual-desktop-live-click.spec.ts"
    helper_rel = "swissknife/test/e2e/helpers/live-click-session.ts"
    spec = repo / spec_rel
    helper = repo / helper_rel
    spec.parent.mkdir(parents=True)
    helper.parent.mkdir(parents=True)
    spec.write_text(
        "import { runLiveClickTarget } from './helpers/live-click-session';\n",
        encoding="utf-8",
    )
    helper.write_text(
        "export async function runLiveClickSession(){ return null }\n",
        encoding="utf-8",
    )
    board = repo / "board.md"
    board.write_text(
        "\n".join(
            [
                "## VOG-021 Live click",
                "- Status: pending",
                "- Depends on:",
                f"- Deterministic command: node {spec_rel}",
                "- Validation: test -f ready.txt",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log_path = repo / "lane.log"
    log_path.write_text(
        "SyntaxError: The requested module './helpers/stale-session' "
        "does not provide an export named 'staleExport'\n",
        encoding="utf-8",
    )
    attempt_path = repo / "attempts.json"
    monkeypatch.setattr(
        lane_resume.shutil,
        "which",
        lambda name: "/usr/bin/grok" if name == "grok" else None,
    )
    calls: list[list[str]] = []
    real = lane_resume._run_argv
    node_runs = {"n": 0}

    def fake(argv: list[str], **kwargs: Any) -> int:
        calls.append(list(argv))
        if Path(argv[0]).name == "grok":
            allowed = [arg for arg in argv if arg.startswith("Edit(") and arg.endswith(")")]
            relative = allowed[0][len("Edit(") : -1]
            target = repo / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "export async function runLiveClickTarget(){ return null }\n",
                encoding="utf-8",
            )
            return 0
        if argv and argv[0] == "node":
            node_runs["n"] += 1
            if node_runs["n"] == 1:
                with Path(kwargs["log_path"]).open("a", encoding="utf-8") as handle:
                    handle.write(
                        "SyntaxError: The requested module './helpers/live-click-session' "
                        "does not provide an export named 'runLiveClickTarget'\n"
                    )
                return 1
            (repo / "ready.txt").write_text("ok\n", encoding="utf-8")
            return 0
        return real(argv, **kwargs)

    monkeypatch.setattr(lane_resume, "_run_argv", fake)
    command = ["node", spec_rel]
    assert (
        lane_resume.repair_missing_named_export(
            command,
            repo_root=repo,
            log_path=log_path,
            attempt_path=attempt_path,
            timeout=30,
            heartbeat=lambda **_kwargs: None,
            allow_bounded_implement=False,
            log_start=0,
        )
        is False
    )
    assert calls == []
    kwargs = {
        "repo_root": repo,
        "todo_path": board,
        "task_header_prefix": "## VOG-",
        "attempt_path": attempt_path,
        "log_path": log_path,
        "heartbeat": lambda **_kwargs: None,
        "allow_bounded_implement": True,
    }
    first = lane_resume.run_resume_cycle(**kwargs)
    grok_calls = [argv for argv in calls if Path(argv[0]).name == "grok"]
    assert first["phase"] == "task_command_passed"
    assert first["advanced_task_id"] == "VOG-021"
    assert first["work_complete"] is False
    assert first["completion_authority"] is False
    assert "- Status: completed" in board.read_text(encoding="utf-8")
    assert (repo / "ready.txt").is_file()
    assert len(grok_calls) == 1
    assert "--no-subagents" in grok_calls[0]
    assert grok_calls[0][grok_calls[0].index("--max-turns") + 1] == "16"
    assert grok_calls[0][grok_calls[0].index("--permission-mode") + 1] == "dontAsk"
    edit_allows = [arg for arg in grok_calls[0] if arg.startswith("Edit(")]
    assert edit_allows == [f"Edit({helper_rel})"]
    prompt = Path(grok_calls[0][grok_calls[0].index("--prompt-file") + 1]).read_text(
        encoding="utf-8"
    )
    assert "Add a runtime export of runLiveClickTarget used by the importer." in prompt
    assert "Keep every existing export." in prompt
    assert "Do not edit other files." in prompt
    assert "No shell." in prompt
    assert "run_terminal_command" in " ".join(grok_calls[0])
    assert "export async function runLiveClickTarget(){ return null }" in helper.read_text(
        encoding="utf-8"
    )
    assert (
        lane_resume.repair_missing_named_export(
            command,
            repo_root=repo,
            log_path=log_path,
            attempt_path=repo / "fresh-attempts.json",
            timeout=30,
            heartbeat=lambda **_kwargs: None,
            allow_bounded_implement=True,
            log_start=log_path.stat().st_size,
        )
        is False
    )
    assert (
        lane_resume.repair_missing_named_export(
            command,
            repo_root=repo,
            log_path=log_path,
            attempt_path=attempt_path,
            timeout=30,
            heartbeat=lambda **_kwargs: None,
            allow_bounded_implement=True,
            log_start=0,
        )
        is False
    )
    second = lane_resume.run_resume_cycle(**kwargs)
    assert second["work_complete"] is False
    assert second["completion_authority"] is False
    assert len([argv for argv in calls if Path(argv[0]).name == "grok"]) == 1


def test_dead_blocked_resume_with_pending_work_restarts_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _status("blocked", pid=111)
    status["phase"] = "deterministic_resume_unavailable"
    status["terminal_reason"] = "deterministic_resume_unavailable"
    supervisor, _pid_path = _watchdog_fixture(tmp_path, status, pid_file_value=111)
    _bind_board(supervisor, tmp_path, todo_text=_PENDING_BOARD)
    calls: list[dict[str, Any]] = []

    def resume(lane_info: dict[str, Any], *, repo_root: Path, manifest: Any = None) -> dict[str, Any]:
        del manifest
        calls.append(dict(lane_info))
        assert repo_root == tmp_path
        return {
            "restarted": True,
            "new_pid": 6161,
            "reason": "resume_stopped_lane",
            "work_complete": False,
            "completion_authority": False,
        }

    monkeypatch.setattr(watchdog_module, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(watchdog_module, "resume_stopped_lane", resume)

    first = supervisor._check_cycle()["reports"][0]
    second = supervisor._check_cycle()["reports"][0]

    assert len(calls) == 1
    assert first["action"] == "restarted"
    assert first["reason"] == "resume_stopped_lane"
    assert first["status"]["work_complete"] is False
    assert second["action"] == "repeated_stopped_pending_repair"
    assert second["status"]["completion_authority"] is False


def test_resume_stopped_lane_refuses_live_pid_and_spawns_one_fenced_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue.lane_resume import (
        mark_task_completed,
        parse_deterministic_command,
        resume_stopped_lane,
        run_resume_cycle,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    assert parse_deterministic_command("grok --yolo", repo_root=repo) is None
    assert parse_deterministic_command("node ok.js; rm -rf /", repo_root=repo) is None
    assert parse_deterministic_command("node swissknife/script.cjs", repo_root=repo) == [
        "node",
        "swissknife/script.cjs",
    ]
    board = repo / "board.md"
    board.write_text(
        "\n".join(
            [
                "## VDG-010 Ledger",
                "- Status: pending",
                "- Depends on:",
                "- Deterministic command: python3 -c 'open(\"ready.txt\",\"w\").write(\"ok\")'",
                "- Validation: test -f ready.txt",
                "",
                "## VDG-011 Census",
                "- Status: pending",
                "- Depends on:",
                "- Deterministic command: grok --yolo",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    def heartbeat(**_kwargs: Any) -> None:
        calls.append("hb")

    cycle = run_resume_cycle(
        repo_root=repo,
        todo_path=board,
        task_header_prefix="## VDG-",
        attempt_path=repo / "attempts.json",
        log_path=repo / "lane.log",
        heartbeat=heartbeat,
    )
    completed = board.read_text(encoding="utf-8")
    assert cycle["advanced_task_id"] == "VDG-010"
    assert cycle["work_complete"] is False
    assert cycle["completion_authority"] is False
    assert "- Status: completed" in completed
    assert "grok" not in (repo / "lane.log").read_text(encoding="utf-8")
    assert (repo / "ready.txt").read_text(encoding="utf-8") == "ok"
    second = run_resume_cycle(
        repo_root=repo,
        todo_path=board,
        task_header_prefix="## VDG-",
        attempt_path=repo / "attempts.json",
        log_path=repo / "lane.log",
        heartbeat=heartbeat,
    )
    assert second["state"] == "blocked"
    assert "VDG-011:provider_or_shell_refused" in second["backpressure_reasons"]
    assert "- Status: pending" in board.read_text(encoding="utf-8")

    pid_path = repo / "lane.pid"
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    refused = resume_stopped_lane(
        {
            "pid_path": str(pid_path),
            "todo_path": str(board),
            "task_header_prefix": "## VDG-",
            "state_dir": str(repo),
            "state_prefix": "lane",
            "bundle_key": "vdg-planning",
            "log_path": str(repo / "lane.log"),
        },
        repo_root=repo,
        manifest={"master_pid_path": str(repo / "missing-master.pid")},
    )
    assert refused["restarted"] is False
    assert refused["reason"] == "live_lane_pid"
    assert refused["completion_authority"] is False

    spawned: list[list[str]] = []

    class _Process:
        pid = 7171

    def popen(command: list[str], **_kwargs: Any) -> _Process:
        spawned.append(list(command))
        return _Process()

    pid_path.unlink()
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.rescue.lane_resume.subprocess.Popen",
        popen,
    )
    ready_pids = []

    def ready(process, _ready_fd, *, pid_path, **_kwargs):
        # Model the child's publication, not an unverified parent-side pid
        # write. Real inherited-fence/pipe behavior has lifecycle coverage.
        pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
        ready_pids.append(process.pid)
        return ""

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.rescue.lane_resume._await_resume_ready",
        ready,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.rescue.lane_resume._pid_alive",
        lambda _pid: False,
    )
    started = resume_stopped_lane(
        {
            "pid_path": str(pid_path),
            "todo_path": "board.md",
            "task_header_prefix": "## VDG-",
            "state_dir": str(repo),
            "state_prefix": "lane",
            "bundle_key": "vdg-planning",
            "log_path": str(repo / "lane.log"),
        },
        repo_root=repo,
        manifest={"master_pid_path": str(repo / "missing-master.pid")},
    )
    assert started["restarted"] is True
    assert started["new_pid"] == 7171
    assert started["readiness_verified"] is True
    assert ready_pids == [7171]
    assert started["work_complete"] is False
    assert spawned[0][1:3] == [
        "-m",
        "ipfs_accelerate_py.agent_supervisor.rescue.lane_resume",
    ]
    assert "implementation_supervisor" not in " ".join(spawned[0])
    assert pid_path.read_text(encoding="utf-8").strip() == "7171"
    marked = mark_task_completed(
        "## VDG-010 Work\n- Status: pending\n",
        "VDG-010",
        task_header_prefix="## VDG-",
        note="unit",
    )
    assert "- Status: completed" in marked
    assert "- Completion: unit" in marked
    flagged = resume_stopped_lane(
        {
            "pid_path": str(pid_path),
            "todo_path": "board.md",
            "task_header_prefix": "## VDG-",
            "state_dir": str(repo),
            "state_prefix": "lane",
            "bundle_key": "vdg-planning",
            "log_path": str(repo / "lane.log"),
        },
        repo_root=repo,
        manifest={
            "master_pid_path": str(repo / "missing-master.pid"),
            "allow_bounded_task_implement": True,
        },
    )
    assert flagged["restarted"] is True
    assert flagged["readiness_verified"] is True
    assert ready_pids == [7171, 7171]
    assert "--allow-bounded-implement" in spawned[-1]
    assert "implementation_supervisor" not in " ".join(spawned[-1])
    assert not any(part in {"grok", "codex"} for part in spawned[-1])


def test_changed_pending_signature_can_resume_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = _status("blocked", pid=111)
    status["phase"] = "deterministic_resume_unavailable"
    status["terminal_reason"] = "deterministic_resume_unavailable"
    supervisor, _pid_path = _watchdog_fixture(tmp_path, status, pid_file_value=111)
    _bind_board(supervisor, tmp_path, todo_text=_PENDING_BOARD)
    calls: list[int] = []

    def resume(lane_info: dict[str, Any], *, repo_root: Path, manifest: Any = None) -> dict[str, Any]:
        del lane_info, repo_root, manifest
        calls.append(1)
        return {
            "restarted": True,
            "new_pid": 6162,
            "reason": "resume_stopped_lane",
            "work_complete": False,
            "completion_authority": False,
        }

    monkeypatch.setattr(watchdog_module, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(watchdog_module, "resume_stopped_lane", resume)
    supervisor._check_cycle()
    board = tmp_path / "board.md"
    board.write_text(
        _PENDING_BOARD.replace("## VDG-010 Ledger", "## VDG-013 Harness"),
        encoding="utf-8",
    )
    supervisor._check_cycle()
    assert len(calls) == 2


def test_deterministic_repair_runs_before_one_bounded_implement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume

    repo = tmp_path
    (repo / "out").mkdir()
    board = repo / "board.md"
    board.write_text(
        "\n".join(
            [
                "## VDG-011 Census",
                "- Status: pending",
                "- Depends on:",
                "- Predicted files: out/census.spec.ts",
                "- Validation: test -f out/census.spec.ts",
                "- Acceptance: rows include app_id selector and disposition",
                "",
                "## VDG-051 Rearm",
                "- Status: pending",
                "- Depends on:",
                "- Deterministic repair: python3 -c 'open(\"rearmed.txt\",\"w\").write(\"ok\")'",
                "- Validation: test -f rearmed.txt",
                "",
                "## VDG-099 Refused",
                "- Status: pending",
                "- Depends on:",
                "- Deterministic command: grok --yolo",
                "- Predicted files: out/census.spec.ts",
                "- Validation: test -f out/census.spec.ts",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        lane_resume.shutil,
        "which",
        lambda name: "/usr/bin/grok" if name == "grok" else None,
    )
    calls: list[list[str]] = []
    real = lane_resume._run_argv

    def fake(argv: list[str], **kwargs: Any) -> int:
        calls.append(list(argv))
        if Path(argv[0]).name == "grok":
            target = repo / "out" / "census.spec.ts"
            target.write_text("app_id selector disposition " + ("x" * 128), encoding="utf-8")
            return 0
        return real(argv, **kwargs)

    monkeypatch.setattr(lane_resume, "_run_argv", fake)

    def heartbeat(**_kwargs: Any) -> None:
        return None

    kwargs = {
        "repo_root": repo,
        "todo_path": board,
        "task_header_prefix": "## VDG-",
        "attempt_path": repo / "attempts.json",
        "log_path": repo / "lane.log",
        "heartbeat": heartbeat,
        "allow_bounded_implement": True,
    }
    first = lane_resume.run_resume_cycle(**kwargs)
    assert first["advanced_task_id"] == "VDG-051"
    assert first["work_complete"] is False
    assert all(Path(argv[0]).name != "grok" for argv in calls)
    second = lane_resume.run_resume_cycle(**kwargs)
    assert second["advanced_task_id"] == "VDG-011"
    assert second["completion_authority"] is False
    grok_calls = [argv for argv in calls if Path(argv[0]).name == "grok"]
    assert len(grok_calls) == 1
    assert "--no-subagents" in grok_calls[0]
    assert "--permission-mode" in grok_calls[0]
    assert "dontAsk" in grok_calls[0]
    assert "implementation_supervisor" not in " ".join(grok_calls[0])
    assert (repo / "out" / "census.spec.ts").stat().st_size >= 128
    third = lane_resume.run_resume_cycle(**kwargs)
    assert third["state"] == "blocked"
    assert "VDG-099:provider_or_shell_refused" in third["backpressure_reasons"]
    assert len([argv for argv in calls if Path(argv[0]).name == "grok"]) == 1
    completed = board.read_text(encoding="utf-8")
    assert "- Status: completed" in completed.split("## VDG-099", 1)[0]
    assert "- Status: pending" in completed.split("## VDG-099", 1)[1]


def test_failed_bounded_implement_retries_once_with_more_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume

    repo = tmp_path
    board = repo / "board.md"
    board.write_text(
        "\n".join(
            [
                "## VDG-011 Census",
                "- Status: pending",
                "- Depends on:",
                "- Predicted files: out/census.spec.ts",
                "- Validation: test -f out/census.spec.ts",
                "- Acceptance: census rows",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(lane_resume.shutil, "which", lambda _name: "/usr/bin/grok")
    calls: list[list[str]] = []

    def fake(argv: list[str], **_kwargs: Any) -> int:
        calls.append(list(argv))
        return 1

    monkeypatch.setattr(lane_resume, "_run_argv", fake)
    kwargs = {
        "repo_root": repo,
        "todo_path": board,
        "task_header_prefix": "## VDG-",
        "attempt_path": repo / "attempts.json",
        "log_path": repo / "lane.log",
        "heartbeat": lambda **_kwargs: None,
        "allow_bounded_implement": True,
    }
    first = lane_resume.run_resume_cycle(**kwargs)
    second = lane_resume.run_resume_cycle(**kwargs)
    third = lane_resume.run_resume_cycle(**kwargs)
    assert first["phase"] == "bounded_implement_failed"
    assert second["phase"] == "bounded_implement_failed"
    assert first["work_complete"] is False
    assert [argv[argv.index("--max-turns") + 1] for argv in calls] == ["16", "32"]
    assert third["state"] == "blocked"
    assert "VDG-011:bounded_implement_already_used" in third["backpressure_reasons"]
    assert "- Status: pending" in board.read_text(encoding="utf-8")


def test_turn_exhausted_bounded_implement_escalates_once_then_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume

    repo = tmp_path
    board = repo / "board.md"
    board.write_text(
        "\n".join(
            [
                "## VOG-011 Census",
                "- Status: pending",
                "- Depends on:",
                "- Predicted files: out/census.cjs",
                "- Validation: test -f out/census.cjs",
                "- Acceptance: census rows",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(lane_resume.shutil, "which", lambda _name: "/usr/bin/grok")
    calls: list[list[str]] = []
    log_path = repo / "lane.log"

    def fake(argv: list[str], **kwargs: Any) -> int:
        calls.append(list(argv))
        target = Path(kwargs["log_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write("Error: max turns reached\n")
        return 1

    monkeypatch.setattr(lane_resume, "_run_argv", fake)
    kwargs = {
        "repo_root": repo,
        "todo_path": board,
        "task_header_prefix": "## VOG-",
        "attempt_path": repo / "attempts.json",
        "log_path": log_path,
        "heartbeat": lambda **_kwargs: None,
        "allow_bounded_implement": True,
    }
    phases = [lane_resume.run_resume_cycle(**kwargs)["phase"] for _ in range(4)]
    turns = [argv[argv.index("--max-turns") + 1] for argv in calls]
    assert phases == ["bounded_implement_failed"] * 4
    assert turns == ["16", "32", "64", "128"]
    blocked = lane_resume.run_resume_cycle(**kwargs)
    assert blocked["state"] == "blocked"
    assert blocked["phase"] == "deterministic_resume_unavailable"
    assert "VOG-011:bounded_implement_already_used" in blocked["backpressure_reasons"]
    marker = json.loads((repo / "bounded-implement-VOG-011.json").read_text(encoding="utf-8"))
    assert marker["escalation"] == 1
    assert marker["turn_exhausted"] is True
    assert marker["ok"] is False
    assert marker["work_complete"] is False
    assert "- Status: pending" in board.read_text(encoding="utf-8")


def test_legacy_turn_exhausted_marker_escalates_from_the_lane_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume

    repo = tmp_path
    board = repo / "board.md"
    board.write_text(
        "\n".join(
            [
                "## VOG-011 Census",
                "- Status: pending",
                "- Depends on:",
                "- Predicted files: out/census.cjs",
                "- Validation: test -f out/census.cjs",
                "- Acceptance: census rows",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (repo / "bounded-implement-VOG-011.json").write_text(
        json.dumps(
            {
                "task_id": "VOG-011",
                "attempts": 2,
                "ok": False,
                "work_complete": False,
                "completion_authority": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (repo / "lane.log").write_text(
        "bounded-implement-VOG-011.prompt.txt\nMax turns reached\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(lane_resume.shutil, "which", lambda _name: "/usr/bin/grok")
    calls: list[list[str]] = []

    def fake(argv: list[str], **_kwargs: Any) -> int:
        calls.append(list(argv))
        return 1

    monkeypatch.setattr(lane_resume, "_run_argv", fake)
    cycle = lane_resume.run_resume_cycle(
        repo_root=repo,
        todo_path=board,
        task_header_prefix="## VOG-",
        attempt_path=repo / "attempts.json",
        log_path=repo / "lane.log",
        heartbeat=lambda **_kwargs: None,
        allow_bounded_implement=True,
    )
    assert cycle["phase"] == "bounded_implement_failed"
    assert calls[0][calls[0].index("--max-turns") + 1] == "64"
    marker = json.loads((repo / "bounded-implement-VOG-011.json").read_text(encoding="utf-8"))
    assert marker["escalation"] == 1
    assert marker["attempts"] == 1


def test_withheld_gate_stays_pending_and_an_exhausted_gate_can_be_rechecked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume

    repo = tmp_path
    board = repo / "board.md"
    board.write_text(
        "\n".join(
            [
                "## VDG-060 Closeout",
                "- Status: pending",
                "- Depends on:",
                "- Deterministic command: python3 -c 'raise SystemExit(75)'",
                "- Validation: test -f coverage.json",
                "",
            ]
        ),
        encoding="utf-8",
    )
    attempts = repo / "attempts.json"
    attempts.write_text('{"VDG-060": 3}\n', encoding="utf-8")
    calls: list[int] = []

    def fake(_argv: list[str], **_kwargs: Any) -> int:
        calls.append(1)
        return 75

    monkeypatch.setattr(lane_resume, "_run_argv", fake)
    kwargs = {
        "repo_root": repo,
        "todo_path": board,
        "task_header_prefix": "## VDG-",
        "attempt_path": attempts,
        "log_path": repo / "lane.log",
        "heartbeat": lambda **_kwargs: None,
        "allow_bounded_implement": True,
    }
    cycle = lane_resume.run_resume_cycle(**kwargs)
    assert cycle["phase"] == "gate_withheld"
    assert cycle["state"] == "healthy"
    assert cycle["work_complete"] is False
    assert cycle["completion_authority"] is False
    assert "- Status: pending" in board.read_text(encoding="utf-8")
    assert "VDG-060" not in attempts.read_text(encoding="utf-8")
    again = lane_resume.run_resume_cycle(**kwargs)
    assert again["phase"] == "gate_withheld"
    assert len(calls) == 2


def test_exhausted_hard_failure_waits_before_the_next_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume

    repo = tmp_path
    board = repo / "board.md"
    board.write_text(
        "\n".join(
            [
                "## VDG-060 Closeout",
                "- Status: pending",
                "- Depends on:",
                "- Deterministic command: python3 -c 'raise SystemExit(1)'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    attempts = repo / "attempts.json"
    attempts.write_text('{"VDG-060": 3}\n', encoding="utf-8")
    calls: list[int] = []

    def fake(_argv: list[str], **_kwargs: Any) -> int:
        calls.append(1)
        return 1

    monkeypatch.setattr(lane_resume, "_run_argv", fake)
    kwargs = {
        "repo_root": repo,
        "todo_path": board,
        "task_header_prefix": "## VDG-",
        "attempt_path": attempts,
        "log_path": repo / "lane.log",
        "heartbeat": lambda **_kwargs: None,
        "allow_bounded_implement": False,
    }
    probed = lane_resume.run_resume_cycle(**kwargs)
    waiting = lane_resume.run_resume_cycle(**kwargs)
    assert probed["phase"] == "deterministic_command_failed"
    assert probed["work_complete"] is False
    assert waiting["state"] == "healthy"
    assert waiting["phase"] == "attempt_retry_wait"
    assert waiting["terminal_reason"] == ""
    assert waiting["work_complete"] is False
    assert waiting["completion_authority"] is False
    assert "VDG-060:attempts_exhausted" in waiting["backpressure_reasons"]
    assert len(calls) == 1
    assert "- Status: pending" in board.read_text(encoding="utf-8")


def test_retry_wait_does_not_complete_without_produced_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume

    repo = tmp_path
    board = repo / "board.md"
    board.write_text(
        "\n".join(
            [
                "## VOG-021 Click the image-viewer pilot",
                "- Status: pending",
                "- Depends on:",
                "- Deterministic command: python3 -c 'raise SystemExit(1)'",
                "- Validation: python3 -c 'raise SystemExit(0)'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    attempts = repo / "attempts.json"
    attempts.write_text(
        json.dumps(
            {
                "VOG-021": 3,
                "_retry_after_ms": {"VOG-021": int(time.time() * 1000) + 600_000},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake(argv: list[str], **_kwargs: Any) -> int:
        calls.append(list(argv))
        if "SystemExit(1)" in " ".join(argv):
            raise AssertionError("exhausted command ran after validation passed")
        return 0

    monkeypatch.setattr(lane_resume, "_run_argv", fake)
    cycle = lane_resume.run_resume_cycle(
        repo_root=repo,
        todo_path=board,
        task_header_prefix="## VOG-",
        attempt_path=attempts,
        log_path=repo / "lane.log",
        heartbeat=lambda **_kwargs: None,
        allow_bounded_implement=False,
    )
    text = board.read_text(encoding="utf-8")
    assert cycle["phase"] == "attempt_retry_wait"
    assert cycle["active_task_id"] == ""
    assert "advanced_task_id" not in cycle
    assert cycle["work_complete"] is False
    assert cycle["completion_authority"] is False
    assert "- Status: pending" in text
    assert calls == []


def test_retry_wait_stays_waiting_when_validation_still_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue import lane_resume

    repo = tmp_path
    board = repo / "board.md"
    board.write_text(
        "\n".join(
            [
                "## VOG-021 Click the image-viewer pilot",
                "- Status: pending",
                "- Depends on:",
                "- Deterministic command: python3 -c 'raise SystemExit(1)'",
                "- Validation: python3 -c 'raise SystemExit(2)'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    attempts = repo / "attempts.json"
    attempts.write_text(
        json.dumps(
            {
                "VOG-021": 3,
                "_retry_after_ms": {"VOG-021": int(time.time() * 1000) + 600_000},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake(argv: list[str], **_kwargs: Any) -> int:
        text = " ".join(argv)
        if "SystemExit(1)" in text:
            raise AssertionError("exhausted command ran while validation still fails")
        return 1

    monkeypatch.setattr(lane_resume, "_run_argv", fake)
    cycle = lane_resume.run_resume_cycle(
        repo_root=repo,
        todo_path=board,
        task_header_prefix="## VOG-",
        attempt_path=attempts,
        log_path=repo / "lane.log",
        heartbeat=lambda **_kwargs: None,
        allow_bounded_implement=False,
    )
    assert cycle["phase"] == "attempt_retry_wait"
    assert cycle["work_complete"] is False
    assert cycle["completion_authority"] is False
    assert "- Status: pending" in board.read_text(encoding="utf-8")
    assert "retry waiting VOG-021" in (repo / "lane.log").read_text(encoding="utf-8")


def test_attempt_backoff_stays_in_process_until_the_budget_ends(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue.lane_resume import serve

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "board.md").write_text(
        "\n".join(
            [
                "## VOG-021 Click the image-viewer pilot",
                "- Status: pending",
                "- Depends on:",
                "- Deterministic command: python3 -c 'raise SystemExit(1)'",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    status = repo / "status.json"
    (repo / "lane_resume_attempts.json").write_text(
        json.dumps(
            {
                "VOG-021": 3,
                "_retry_after_ms": {"VOG-021": int(time.time() * 1000) + 600_000},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    code = serve(
        [
            "--repo-root",
            str(repo),
            "--todo-path",
            "board.md",
            "--task-header-prefix",
            "## VOG-",
            "--status-path",
            str(status),
            "--pid-path",
            str(repo / "lane.pid"),
            "--log-path",
            str(repo / "lane.log"),
            "--bundle-key",
            "vog-planning",
            "--max-seconds",
            "2",
        ]
    )
    elapsed = time.monotonic() - started
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert code == 0
    assert elapsed >= 1.5
    assert payload["state"] == "stopped"
    assert payload["phase"] == "lane_resume_exited"
    assert payload["terminal_reason"] == "resume_duration_elapsed"
    assert payload["work_complete"] is False
    assert payload["completion_authority"] is False
    assert "- Status: pending" in (repo / "board.md").read_text(encoding="utf-8")
    assert "retry waiting VOG-021" in (repo / "lane.log").read_text(encoding="utf-8")


def test_live_blocked_resume_exits_and_keeps_the_blocked_status(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue.lane_resume import serve

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "board.md").write_text(
        "\n".join(
            [
                "## VOG-011 Census",
                "- Status: pending",
                "- Depends on:",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    status = repo / "status.json"
    started = time.monotonic()
    code = serve(
        [
            "--repo-root",
            str(repo),
            "--todo-path",
            "board.md",
            "--task-header-prefix",
            "## VOG-",
            "--status-path",
            str(status),
            "--pid-path",
            str(repo / "lane.pid"),
            "--log-path",
            str(repo / "lane.log"),
            "--bundle-key",
            "vog-planning",
            "--max-seconds",
            "30",
        ]
    )
    elapsed = time.monotonic() - started
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert code == 0
    assert elapsed < 5
    assert payload["state"] == "blocked"
    assert payload["phase"] == "deterministic_resume_unavailable"
    assert payload["terminal_reason"] == "deterministic_resume_unavailable"
    assert payload["work_complete"] is False
    assert payload["completion_authority"] is False


def test_attempt_retry_wait_stays_alive_until_the_duration_elapses(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue.lane_resume import serve

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "board.md").write_text(
        "\n".join(
            [
                "## VOG-021 Pilot",
                "- Status: pending",
                "- Depends on:",
                "- Deterministic command: python3 -c 'raise SystemExit(1)'",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    status = repo / "status.json"
    (repo / "lane_resume_attempts.json").write_text(
        json.dumps(
            {
                "VOG-021": 3,
                "_retry_after_ms": {"VOG-021": int(time.time() * 1000) + 900_000},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    code = serve(
        [
            "--repo-root",
            str(repo),
            "--todo-path",
            "board.md",
            "--task-header-prefix",
            "## VOG-",
            "--status-path",
            str(status),
            "--pid-path",
            str(repo / "lane.pid"),
            "--log-path",
            str(repo / "lane.log"),
            "--bundle-key",
            "vog-planning",
            "--max-seconds",
            "2",
        ]
    )
    elapsed = time.monotonic() - started
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert code == 0
    assert elapsed >= 1.5
    assert elapsed < 8
    assert payload["state"] == "stopped"
    assert payload["phase"] == "lane_resume_exited"
    assert payload["terminal_reason"] == "resume_duration_elapsed"
    assert payload["work_complete"] is False
    assert payload["completion_authority"] is False
