"""A declared gate repair wait is distinct from failed process progress."""

import json
import os
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import supervisor_watchdog as watchdog


@pytest.fixture
def gate(tmp_path):
    now = time.time()
    log = tmp_path / "lane.log"
    log.write_text("unchanged gate refused\n")
    os.utime(log, (now - 2000, now - 2000))
    supervisor = watchdog.SupervisorWatchdog(manifest_path=tmp_path / "manifest.json", repo_root=tmp_path)
    heartbeat = {
        "phase": "gate_retry_wait", "gate_recovery_needed": True,
        "gate_evidence_fingerprint": "no-go-evidence", "active_task_id": "TASK-060",
        "last_task_transition_at_ms": int((now - 2000) * 1000),
        "next_retry_at_ms": int((now + 300) * 1000),
    }
    return supervisor, {"log_path": str(log)}, heartbeat, now


def progress(gate, *, alive=True, stale=False):
    supervisor, lane, heartbeat, _now = gate
    return supervisor._classify_lane_progress(lane, process_alive=alive,
        heartbeat_stale=stale, heartbeat_check=heartbeat)


def test_bounded_pending_gate_retry_does_not_restart_a_live_lane(gate):
    result = progress(gate)
    assert result["progress_stalled"] is False
    assert result["reason"] == "gate_retry_not_due"
    assert result["active_task_id"] == "TASK-060"
    assert result["completion_authority"] is result["work_complete"] is False


@pytest.mark.parametrize("change", ["overdue", "unbounded", "missing_fingerprint", "not_repair", "malformed"])
def test_missing_expired_or_unbounded_wait_cannot_hide_a_stall(gate, change):
    heartbeat, now = gate[2:]
    if change == "overdue":
        heartbeat["next_retry_at_ms"] = int((now - 1) * 1000)
    elif change == "unbounded":
        heartbeat["next_retry_at_ms"] = int((now + 2000) * 1000)
    elif change == "missing_fingerprint":
        heartbeat.pop("gate_evidence_fingerprint")
    elif change == "not_repair":
        heartbeat["gate_recovery_needed"] = False
    else:
        heartbeat["next_retry_at_ms"] = "unknown"
    assert progress(gate)["progress_stalled"] is True


def test_gate_wait_does_not_mask_a_stale_heartbeat(gate):
    result = progress(gate, stale=True)
    assert result["reason"] != "gate_retry_not_due"


def test_heartbeat_projection_preserves_recovery_packet_and_deadline(tmp_path):
    status = {"schema": watchdog.LIFECYCLE_STATUS_SCHEMA, "state": "degraded",
              "pid": os.getpid(), "phase": "gate_retry_wait", "heartbeat_at_ms": time.time() * 1000,
              "gate_recovery_needed": True, "gate_evidence_fingerprint": "evidence",
              "repair_packet": "lane/repair.json", "next_retry_at_ms": 12345}
    (tmp_path / "gate_status.json").write_text(json.dumps(status))
    result = watchdog.check_lane_heartbeat(tmp_path, "gate", timeout_seconds=600)
    for name in ("gate_recovery_needed", "gate_evidence_fingerprint", "repair_packet", "next_retry_at_ms"):
        assert result[name] == status[name]
