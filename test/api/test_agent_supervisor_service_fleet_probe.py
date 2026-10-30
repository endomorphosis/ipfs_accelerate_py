"""Service report bridging preserves idle, scope and non-authority boundaries."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import service_fleet_probe as probe

NOW = 1_000.0
NAMESPACE = "legal-ir-autoencoder-v1"
UNITS = {f"lane-{i}": f"{NAMESPACE}{'-lane-' + str(i) if i else ''}.service"
         for i in range(4)}


def report():
    lanes = []
    for lane_id, unit in UNITS.items():
        evidence = {name: 0 for name in probe.COUNT_FIELDS}
        evidence.update(task_count=135, completed_count=135, active=False,
                        native_daemon_pid_alive=True, native_status_at_epoch=NOW - 1,
                        last_progress_at_epoch=100, last_work_event_at_epoch=100,
                        selection_idle_reason="no_shard_selectable_ready_tasks",
                        namespace_mismatch=False)
        lanes.append({"lane_id": lane_id, "health": {
            "schema": probe.LANE_SCHEMA, "service": unit, "board_namespace": NAMESPACE,
            "checked_at_epoch": NOW, "healthy": True, "status": "completed", "action": "none",
            "evidence": evidence}})
    return {"schema": probe.REPORT_SCHEMA, "checked_at_epoch": NOW, "healthy": True,
            "status": "healthy", "completion_authority": False, "lane_count": 4, "lanes": lanes}


def observe(tmp_path, value, **changes):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(value))
    arguments = dict(board_id="legal-ir", board_namespace=NAMESPACE,
                     expected_units=UNITS, max_age_seconds=180, now=NOW)
    arguments.update(changes)
    return probe.observe_service_fleet(path, **arguments)


def test_exhausted_legacy_population_is_healthy_idle_not_programme_complete(tmp_path):
    result = observe(tmp_path, report())
    assert result["health"] == "healthy"
    assert result["details"]["service_fleet"]["work_availability"] == "no_runnable_work"
    assert result["busy"] is result["complete"] is result["completion_candidate"] is False
    assert result["details"]["completion_authority"] is False
    lanes = result["details"]["service_fleet"]["lanes"]
    assert [lane["completed_count"] for lane in lanes] == [135] * 4
    assert "task_counts" not in result["details"]
    assert "recovery_action" not in result


def test_heartbeat_refresh_does_not_change_work_progress_token(tmp_path):
    value = report()
    first = observe(tmp_path, value)
    value["checked_at_epoch"] += 1
    for lane in value["lanes"]:
        lane["health"]["checked_at_epoch"] += 1
        lane["health"]["evidence"]["native_status_at_epoch"] += 1
    later = observe(tmp_path, value, now=NOW + 1)
    assert first["progress_token"] == later["progress_token"]
    value["lanes"][0]["health"]["evidence"]["last_work_event_at_epoch"] = NOW
    assert observe(tmp_path, value)["progress_token"] != first["progress_token"]


def test_ready_work_is_visible_without_pretending_a_worker_is_busy(tmp_path):
    value = report()
    evidence = value["lanes"][0]["health"]["evidence"]
    evidence.update(ready_count=1, selectable_ready_count=1, global_ready_count=1,
                    task_count=136)
    result = observe(tmp_path, value)
    assert result["health"] == "healthy" and result["busy"] is False
    assert result["details"]["service_fleet"]["work_availability"] == "runnable_work"


def test_fresh_live_active_lane_sets_busy_without_completion_authority(tmp_path):
    value = report()
    value["lanes"][0]["health"]["evidence"]["active"] = True
    result = observe(tmp_path, value)
    assert result["health"] == "healthy" and result["busy"] is True
    assert result["complete"] is result["completion_candidate"] is False


@pytest.mark.parametrize("field", ["blocked_count", "waiting_count"])
def test_external_dependency_wait_has_real_counts_and_no_ensure(tmp_path, field):
    value = report()
    value["lanes"][0]["health"]["evidence"][field] = 1
    result = observe(tmp_path, value)
    assert result["health"] == "blocked"
    assert result["reason_codes"] == ["external_dependency_wait"]
    assert result["details"]["service_fleet"]["work_availability"] == "dependency_wait"
    assert result["details"]["wait"]["kind"] == "dependency_or_review"
    assert result["details"]["service_fleet"]["lanes"][0][field] == 1
    assert "recovery_action" not in result


@pytest.mark.parametrize("change", ["top_schema", "lane_schema", "namespace", "unit", "missing_lane",
                                  "duplicate_lane", "count_bool", "count_negative", "missing_count",
                                  "completed_exceeds_population", "top_authority", "lane_authority",
                                  "namespace_mismatch", "stale_top", "stale_lane", "future_top",
                                  "future_lane", "stale_native", "future_native", "nonfinite_time",
                                  "boolean_active", "missing_native_live", "missing_lane_evidence"])
def test_invalid_or_mismatched_report_never_proves_idle_or_busy(tmp_path, change):
    value = report()
    lane = value["lanes"][0]["health"]
    evidence = lane["evidence"]
    if change == "top_schema": value["schema"] = "other"
    elif change == "lane_schema": lane["schema"] = "other"
    elif change == "namespace": lane["board_namespace"] = "other"
    elif change == "unit": lane["service"] = "other.service"
    elif change == "missing_lane": value["lanes"].pop()
    elif change == "duplicate_lane": value["lanes"][1] = copy.deepcopy(value["lanes"][0])
    elif change == "count_bool": evidence["ready_count"] = False
    elif change == "count_negative": evidence["ready_count"] = -1
    elif change == "missing_count": del evidence["ready_count"]
    elif change == "completed_exceeds_population": evidence["completed_count"] = 136
    elif change == "top_authority": value["completion_authority"] = True
    elif change == "lane_authority": lane["completion_authority"] = True
    elif change == "namespace_mismatch": evidence["namespace_mismatch"] = True
    elif change == "stale_top": value["checked_at_epoch"] = NOW - 181
    elif change == "stale_lane": lane["checked_at_epoch"] = NOW - 181
    elif change == "future_top": value["checked_at_epoch"] = NOW + 6
    elif change == "future_lane": lane["checked_at_epoch"] = NOW + 6
    elif change == "stale_native": evidence["native_status_at_epoch"] = NOW - 181
    elif change == "future_native": evidence["native_status_at_epoch"] = NOW + 6
    elif change == "nonfinite_time": value["checked_at_epoch"] = float("nan")
    elif change == "boolean_active": evidence["active"] = 1
    elif change == "missing_native_live": del evidence["native_daemon_pid_alive"]
    elif change == "missing_lane_evidence": del lane["evidence"]
    result = observe(tmp_path, value)
    assert result["health"] == "unknown"
    assert result["busy"] is result["complete"] is result["completion_candidate"] is False
    assert result["details"]["service_fleet"]["work_availability"] == "unknown"
    assert not result["progress_token"]
    assert "recovery_action" not in result


def test_monitor_unhealthy_and_dead_native_lane_are_diagnostics(tmp_path):
    value = report()
    value["healthy"] = False
    assert observe(tmp_path, value)["health"] == "degraded"
    value = report()
    value["lanes"][0]["health"]["evidence"]["native_daemon_pid_alive"] = False
    result = observe(tmp_path, value)
    assert result["health"] == "unknown" and result["busy"] is False
    assert result["details"]["service_fleet"]["work_availability"] == "unknown"


@pytest.mark.parametrize("kind", ["missing", "symlink", "fifo", "malformed", "nonobject", "oversized"])
def test_report_file_boundary_fails_closed_without_following_or_blocking(tmp_path, kind):
    path = tmp_path / "report.json"
    if kind == "symlink":
        target = tmp_path / "target.json"
        target.write_text(json.dumps(report()))
        path.symlink_to(target)
    elif kind == "fifo": os.mkfifo(path)
    elif kind == "malformed": path.write_text("{")
    elif kind == "nonobject": path.write_text("[]")
    elif kind == "oversized":
        with path.open("wb") as handle: handle.truncate(8 * 1024 * 1024 + 1)
    result = probe.observe_service_fleet(path, board_id="legal-ir", board_namespace=NAMESPACE,
                                       expected_units=UNITS, now=NOW)
    assert result["health"] == "unknown" and result["busy"] is False
    assert result["details"]["service_fleet"]["work_availability"] == "unknown"


def test_cli_has_same_exact_unit_binding_and_no_mutation(tmp_path, capsys, monkeypatch):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report()))
    before = path.read_bytes()
    monkeypatch.setattr(probe.time, "time", lambda: NOW)
    argv = ["--board", "legal-ir", "--report", str(path), "--namespace", NAMESPACE,
            "--max-age-seconds", "180"]
    for lane_id, unit in UNITS.items(): argv += ["--expected-unit", f"{lane_id}={unit}"]
    assert probe.main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["health"] == "healthy" and result["complete"] is False
    assert path.read_bytes() == before
    with pytest.raises(SystemExit): probe.main([*argv, "--expected-unit", "lane-0=other.service"])



def _backpressure_report():
    """Reconstruct the producer contract; this is not a retained live snapshot."""
    from ipfs_accelerate_py.agent_supervisor.rescue import board_service_monitor as monitor

    value = report()
    for lane in value["lanes"]:
        lane["health"]["evidence"].update(global_ready_count=2, task_count=137)
    health = value["lanes"][0]["health"]
    evidence = health["evidence"]
    evidence.update(ready_count=2, selectable_ready_count=2,
        selection_idle_reason="implementation_retry_deferred:all_eligible_ready_tasks_in_cooldown",
        native_progress_status="backpressure", last_heartbeat_at_epoch=NOW - 1, implement=True)
    produced = monitor.evaluate_health(
        {"startup_grace_seconds": 0, "heartbeat_timeout_seconds": 60, "progress_timeout_seconds": 600},
        {"LoadState": "loaded", "ActiveState": "active", "MainPID": "123",
         "ExecMainStartTimestampMonotonic": "1000000"}, evidence, {}, NOW, NOW)
    assert produced["status"] == "backpressure" and produced["restart_eligible"] is False
    health.update(produced)
    value.update(healthy=False, status="attention_required", issues=[{
        "code": "lane_unhealthy", "lane_id": value["lanes"][0]["lane_id"], "status": "backpressure"}])
    return value


def test_fresh_monitor_cooldown_is_a_non_authoritative_wait_with_positive_ready_counts(tmp_path):
    result = observe(tmp_path, _backpressure_report())
    assert result["health"] == "blocked"
    assert result["reason_codes"] == ["service_fleet_backpressure_wait"]
    assert result["details"]["service_fleet"]["work_availability"] == "backpressure_wait"
    paused = result["details"]["service_fleet"]["lanes"][0]
    assert paused["ready_count"] == paused["selectable_ready_count"] == 2
    assert paused["healthy"] is False and paused["restart_eligible"] is False
    assert result["progress_token"] and result["busy"] is False
    assert result["complete"] is result["completion_candidate"] is False
    assert result["details"]["completion_authority"] is False
    assert result["details"]["wait"]["completion_authority"] is False


@pytest.mark.parametrize("active", [False, True])
def test_paused_lane_cannot_hide_a_genuinely_runnable_or_active_peer(tmp_path, active):
    value = _backpressure_report()
    peer = value["lanes"][1]["health"]["evidence"]
    peer.update(active=active, ready_count=0 if active else 1, selectable_ready_count=0 if active else 1)
    result = observe(tmp_path, value)
    assert result["health"] == "healthy" and result["busy"] is active
    assert result["details"]["service_fleet"]["work_availability"] == "runnable_work"


@pytest.mark.parametrize("damage", ["restart_missing", "restart_true", "action_changed", "reason_missing", "empty_prefix",
                                  "dead_paused", "dead_peer", "active_paused", "stale_peer", "future_peer",
                                  "other_peer_unhealthy", "top_issue_missing", "top_issue_extra", "top_issue_duplicate",
                                  "top_healthy_contradiction", "native_status_invalid"])
def test_cooldown_exemption_requires_complete_fresh_monitor_provenance(tmp_path, damage):
    value = _backpressure_report()
    health = value["lanes"][0]["health"]
    evidence = health["evidence"]
    if damage == "restart_missing": health.pop("restart_eligible")
    elif damage == "restart_true": health["restart_eligible"] = True
    elif damage == "action_changed": health["action"] = "would_restart"
    elif damage == "reason_missing": evidence.update(selection_idle_reason="", native_progress_status=None)
    elif damage == "empty_prefix": evidence.update(selection_idle_reason="implementation_retry_deferred:", native_progress_status=None)
    elif damage == "dead_paused": evidence["native_daemon_pid_alive"] = False
    elif damage == "dead_peer": value["lanes"][1]["health"]["evidence"]["native_daemon_pid_alive"] = False
    elif damage == "active_paused": evidence["active"] = True
    elif damage == "stale_peer": value["lanes"][1]["health"]["checked_at_epoch"] = NOW - 181
    elif damage == "future_peer": value["lanes"][1]["health"]["checked_at_epoch"] = NOW + 6
    elif damage == "other_peer_unhealthy":
        peer = value["lanes"][1]["health"]
        peer.update(healthy=False, status="stale_worker", restart_eligible=True)
        value["issues"].append({"code": "lane_unhealthy", "lane_id": value["lanes"][1]["lane_id"], "status": "stale_worker"})
    elif damage == "top_issue_missing": value.pop("issues")
    elif damage == "top_issue_extra": value["issues"].append({"code": "repeated_lane_failures", "lane_id": "lane-0", "failures": 2})
    elif damage == "top_issue_duplicate": value["issues"].append(copy.deepcopy(value["issues"][0]))
    elif damage == "top_healthy_contradiction": value["healthy"] = True
    elif damage == "native_status_invalid": evidence["native_progress_status"] = ["backpressure"]
    result = observe(tmp_path, value)
    assert result["health"] in {"degraded", "unknown"}
    assert result["details"]["service_fleet"]["work_availability"] == "unknown"
    assert "service_fleet_backpressure_wait" not in result["reason_codes"]
    assert result["complete"] is result["completion_candidate"] is False


def test_cooldown_heartbeat_refresh_does_not_invent_work_progress(tmp_path):
    value = _backpressure_report()
    before = observe(tmp_path, value)
    value["checked_at_epoch"] += 1
    for lane in value["lanes"]:
        lane["health"]["checked_at_epoch"] += 1
        lane["health"]["evidence"]["native_status_at_epoch"] += 1
    after = observe(tmp_path, value, now=NOW + 1)
    assert before["progress_token"] == after["progress_token"]
