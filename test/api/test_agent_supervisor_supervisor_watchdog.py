from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

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
        lambda *_args, **_kwargs: pytest.fail(
            "interrupted canonical state was raw restarted"
        ),
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
        lambda *_args, **_kwargs: pytest.fail(
            "stale live process was replaced without fencing"
        ),
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
        lambda *_args, **_kwargs: pytest.fail(
            "terminal live process was replaced without fencing"
        ),
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
    assert (
        LifecycleStatus.from_dict(report["status"]).to_dict()
        == report["status"]
    )
