"""A failed lane must eventually retry without a tight restart loop."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import supervisor_watchdog as watchdog


@pytest.fixture
def failing_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "lanes": [{"bundle_key": "lane:test", "state_dir": "state", "state_prefix": "test"}],
    }))
    observations = {"alive": False, "stale": True}
    attempts = []

    def pid_check(*_args):
        return {"pid": 123, "alive": observations["alive"],
                "pid_path": str(tmp_path / "lane.pid")}

    def heartbeat_check(*_args, **_kwargs):
        return {key: value for key, value in observations.items() if key != "alive"}

    def restart(info):
        attempts.append(dict(info))
        return {"restarted": False, "reason": "temporary_launch_failure"}

    monkeypatch.setattr(watchdog, "check_lane_pid", pid_check)
    monkeypatch.setattr(watchdog, "check_lane_heartbeat", heartbeat_check)
    supervisor = watchdog.SupervisorWatchdog(
        manifest_path=manifest_path, repo_root=tmp_path,
        max_consecutive_restarts=2, lifecycle_restart=restart,
    )
    return SimpleNamespace(supervisor=supervisor, observations=observations,
                           attempts=attempts, manifest_path=manifest_path)


def cycle(lane):
    return lane.supervisor._check_cycle()["reports"][0]


def test_failed_restart_retries_after_each_finite_capped_backoff(failing_lane):
    lane = failing_lane
    assert cycle(lane)["action"] == "restart_failed"
    assert cycle(lane)["action"] == "restart_failed"
    for attempt_count, skip_cycles in enumerate([1, 2, 4, 8, 16, 32, 32], start=2):
        for remaining in range(skip_cycles, 0, -1):
            report = cycle(lane)
            assert report["action"] == "backoff"
            assert report["backoff_cycles"] == remaining
            assert len(lane.attempts) == attempt_count
        assert cycle(lane)["action"] == "restart_failed"
        assert len(lane.attempts) == attempt_count + 1


def test_healthy_observation_resets_failure_count_and_remaining_wait(failing_lane):
    lane = failing_lane
    for _ in range(2):
        assert cycle(lane)["action"] == "restart_failed"
    lane.observations.update(alive=True, stale=False)
    assert cycle(lane)["action"] == "none"
    lane.observations.update(alive=False, stale=True)
    assert cycle(lane)["action"] == "restart_failed"
    assert cycle(lane)["action"] == "restart_failed"
    assert len(lane.attempts) == 4
    assert cycle(lane)["backoff_cycles"] == 1


@pytest.mark.parametrize("state", ["paused", "stopped", "draining", "stopping"])
def test_intentional_state_suppresses_retries_during_backoff(failing_lane, state):
    lane = failing_lane
    for _ in range(2):
        assert cycle(lane)["action"] == "restart_failed"
    lane.observations.update(schema=watchdog.LIFECYCLE_STATUS_SCHEMA,
                             state=state, stale=False)
    for _ in range(4):
        assert cycle(lane)["action"] in {"state_preserved", "control_recovery_required"}
    assert len(lane.attempts) == 2


def test_dead_published_owner_prevents_retry_after_backoff(failing_lane, monkeypatch):
    lane = failing_lane
    for _ in range(2):
        assert cycle(lane)["action"] == "restart_failed"
    monkeypatch.setattr(watchdog, "published_owner_process_dead", lambda *_args, **_kwargs: True)
    for _ in range(4):
        assert cycle(lane)["action"] == "owner_process_dead"
    assert len(lane.attempts) == 2
    monkeypatch.setattr(watchdog, "published_owner_process_dead", lambda *_args, **_kwargs: False)
    assert cycle(lane)["action"] == "backoff"
    assert cycle(lane)["action"] == "restart_failed"
    assert len(lane.attempts) == 3


def test_dynamic_scheduler_keeps_restart_authority_during_backoff(failing_lane):
    lane = failing_lane
    for _ in range(2):
        assert cycle(lane)["action"] == "restart_failed"
    manifest = json.loads(lane.manifest_path.read_text())
    manifest.update(authoritative=True,
                    schema="ipfs_accelerate_py.agent_supervisor.dynamic_bundle_scheduler@1")
    lane.manifest_path.write_text(json.dumps(manifest))
    for _ in range(4):
        assert cycle(lane)["action"] == "scheduler_recovery_required"
    assert len(lane.attempts) == 2


def progress_stalled_lane(lane):
    now = time.time()
    log_path = lane.manifest_path.parent / "lane.log"
    log_path.write_text("last work before the stall\n")
    os.utime(log_path, (now - 2_000, now - 2_000))
    manifest = json.loads(lane.manifest_path.read_text())
    manifest["lanes"][0]["log_path"] = str(log_path)
    manifest["autonomous_unstall_policy"] = {"enabled": True, "cooldown_ms": 0}
    lane.manifest_path.write_text(json.dumps(manifest))
    lane.observations.update(
        alive=True, stale=False, schema=watchdog.LIFECYCLE_STATUS_SCHEMA,
        state="healthy", active_task_id="TASK-001",
        last_task_transition_at_ms=int((now - 2_000) * 1000),
    )
    return log_path


def test_fenced_restart_without_progress_is_not_reported_as_recovered(failing_lane):
    lane = failing_lane
    progress_stalled_lane(lane)
    manifest = json.loads(lane.manifest_path.read_text())
    assert lane.supervisor._classify_lane_progress(
        manifest["lanes"][0], process_alive=True, heartbeat_stale=False,
        heartbeat_check=lane.observations,
    )["progress_stalled"] is True

    def restart(info):
        lane.attempts.append(dict(info))
        return {"restarted": True, "new_pid": 123}

    lane.supervisor.lifecycle_restart = restart
    report = cycle(lane)
    result = report["autonomous_unstall"]
    assert result["recovered"] is False
    assert result["quarantined"] is True
    assert len(lane.attempts) == 1
    assert result["deterministic"]["attempts"][0]["outcome"] == "failed"
    assert result["work_complete"] is result["completion_authority"] is False


@pytest.mark.parametrize("advance", ["log", "transition", "idle"])
def test_recovery_probe_uses_current_progress_instead_of_original_stall(failing_lane, advance):
    lane = failing_lane
    log_path = progress_stalled_lane(lane)

    def restart(info):
        lane.attempts.append(dict(info))
        if advance == "log":
            os.utime(log_path, None)
        elif advance == "transition":
            lane.observations["last_task_transition_at_ms"] = int(time.time() * 1000)
        else:
            lane.observations["active_task_id"] = ""
        return {"restarted": True, "new_pid": 123}

    lane.supervisor.lifecycle_restart = restart
    report = cycle(lane)
    assert report["action"] == "autonomous_unstall_recovered"
    assert report["autonomous_unstall"]["recovered"] is True
    assert len(lane.attempts) == 1


def test_progress_stall_without_lifecycle_authority_never_restarts_live_owner(failing_lane):
    lane = failing_lane
    progress_stalled_lane(lane)
    lane.supervisor.lifecycle_restart = None
    report = cycle(lane)
    assert report["action"] == "autonomous_unstall_quarantined"
    assert report["autonomous_unstall"]["recovered"] is False
    assert report["autonomous_unstall"]["deterministic"]["attempts"] == []
    assert lane.attempts == []


def test_dead_lane_with_pending_work_remains_unhealthy_after_success_receipt(failing_lane):
    lane = failing_lane
    progress_stalled_lane(lane)
    lane.observations.update(alive=False, state="blocked",
                             phase="deterministic_resume_unavailable")
    manifest = json.loads(lane.manifest_path.read_text())
    board_path = lane.manifest_path.parent / "board.md"
    board_path.write_text("## TASK-001 Remaining work\n- Status: pending\n")
    manifest.update(todo_path=str(board_path), task_header_prefix="## TASK-",
                    master_pid_path="missing-master.pid")
    lane.manifest_path.write_text(json.dumps(manifest))

    def restart(info):
        lane.attempts.append(dict(info))
        return {"restarted": True, "new_pid": 123}

    lane.supervisor.lifecycle_restart = restart
    result = lane.supervisor._watchdog_unstall(
        manifest=manifest, lane=manifest["lanes"][0], lane_started={},
        bundle_key="lane:test", state_dir=lane.manifest_path.parent / "state",
        state_prefix="test", pid_check={"pid": 123, "alive": False},
        heartbeat_check=dict(lane.observations),
    )
    assert result["recovered"] is False
    assert result["quarantined"] is True
    assert len(lane.attempts) == 1


def test_fresh_gate_recovery_packet_does_not_count_as_recovered(failing_lane):
    lane = failing_lane
    progress_stalled_lane(lane)
    lane.observations.update(phase="gate_retry_wait", gate_recovery_needed=True,
                             gate_evidence_fingerprint="unchanged-gate",
                             next_retry_at_ms=int((time.time() + 300) * 1000))
    manifest = json.loads(lane.manifest_path.read_text())

    def restart(info):
        lane.attempts.append(dict(info))
        return {"restarted": True, "new_pid": 123}

    lane.supervisor.lifecycle_restart = restart
    result = lane.supervisor._watchdog_unstall(
        manifest=manifest, lane=manifest["lanes"][0], lane_started={},
        bundle_key="lane:test", state_dir=lane.manifest_path.parent / "state",
        state_prefix="test", pid_check={"pid": 123, "alive": True},
        heartbeat_check=dict(lane.observations),
    )
    assert result["recovered"] is False
    assert result["quarantined"] is True
    assert len(lane.attempts) == 1
