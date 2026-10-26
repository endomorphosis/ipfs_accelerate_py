"""External monitor failure injection without importing the supervisor package."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    # Declare the source dependency for the repository's AST test seal while
    # retaining the runtime test that this CLI needs only the standard library.
    import ipfs_accelerate_py.agent_supervisor.rescue.board_service_monitor


SOURCE = Path(__file__).resolve().parents[2] / "ipfs_accelerate_py/agent_supervisor/rescue/board_service_monitor.py"
SPEC = importlib.util.spec_from_file_location("board_service_monitor_standalone", SOURCE)
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)
NOW = 1_792_483_200.0
MONOTONIC = 10_000.0


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "service": "one-board.service", "board_namespace": "one-board",
        "state_dir": str(tmp_path / "state"), "state_prefix": "board",
        "startup_grace_seconds": 60, "heartbeat_timeout_seconds": 120,
        "progress_timeout_seconds": 600, "restart_cooldown_seconds": 180,
        "max_restarts_per_hour": 2,
    }))
    value = monitor.load_config(path)
    Path(value["state_dir"]).mkdir()
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(value))


def service(*, active="active", age=1000, pid=123):
    return {"ActiveState": active, "MainPID": str(pid), "LoadState": "loaded",
            "ExecMainStartTimestampMonotonic": str(int((MONOTONIC - age) * 1_000_000))}


def evidence(**kwargs):
    return {"state_available": True, "last_heartbeat_at_epoch": NOW - 5,
            "worker_heartbeat_at_epoch": NOW - 5, "last_progress_at_epoch": NOW - 10,
            "last_work_event_at_epoch": NOW - 10, "ready_count": 1, **kwargs}


def evaluate(config, *, observed=None, native=None, previous=None):
    return monitor.evaluate_health(config, observed or service(), native or evidence(),
                                   previous or {}, NOW, MONOTONIC)


def stub_runtime(monkeypatch, observed):
    monkeypatch.setattr(monitor.time, "time", lambda: NOW)
    monkeypatch.setattr(monitor.time, "monotonic", lambda: MONOTONIC)
    monkeypatch.setattr(monitor, "systemd_show", lambda _: observed)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(monitor.subprocess, "run", run)
    return calls


def test_dead_service_restart_is_exact_and_remains_unhealthy_until_verified(config, monkeypatch):
    calls = stub_runtime(monkeypatch, service(active="failed", pid=0))
    result = monitor.monitor(config, apply=True)
    assert calls == [["systemctl", "--user", "restart", "one-board.service"]]
    assert result["status"] == "service_dead"
    assert result["action"] == "restart_requested"
    assert result["healthy"] is False
    saved = json.loads(Path(config["monitor_state_path"]).read_text())
    assert saved["restart_attempts_at_epoch"] == [NOW]
    assert json.loads(Path(config["health_path"]).read_text()) == result
    assert [json.loads(line)["action"] for line in Path(config["actions_path"]).read_text().splitlines()] == [
        "restart_requested", "restart_requested"]


def test_dry_run_never_restarts_or_charges_budget(config, monkeypatch):
    calls = stub_runtime(monkeypatch, service(active="inactive", pid=0))
    result = monitor.monitor(config)
    assert result["action"] == "would_restart"
    assert calls == []
    assert json.loads(Path(config["monitor_state_path"]).read_text())["restart_attempts_at_epoch"] == []


def test_native_child_crash_is_visible_without_parent_restart_even_during_grace(config, monkeypatch):
    calls = stub_runtime(monkeypatch, service(age=10))
    write_json(config['native_state_path'], {'heartbeat_at': NOW - 1, 'ready_count': 50})
    write_json(config['status_path'], {'updated_at': NOW - 1, 'status': 'restarting',
        'daemon_pid_alive': False, 'last_exit_code': 1, 'restart_count': 6})
    result = monitor.monitor(config, apply=True)
    assert result['status'] == 'native_child_restarting'
    assert result['healthy'] is False and result['restart_eligible'] is False
    assert result['evidence']['native_restart_count'] == 6
    assert calls == []


def test_old_native_crash_status_cannot_override_new_parent_startup_grace(config):
    native = evidence(native_managed_status='restarting', native_status_at_epoch=NOW - 60,
        native_daemon_pid_alive=False, native_last_exit_code=1)
    assert evaluate(config, observed=service(age=5), native=native)['status'] == 'starting'


def test_running_native_child_clears_prior_failure_diagnostic(config):
    native = evidence(native_managed_status='running', native_status_at_epoch=NOW - 1,
        native_daemon_pid_alive=True, native_last_exit_code=1, native_restart_count=6)
    assert evaluate(config, native=native)['healthy'] is True


@pytest.mark.parametrize("attempts,action", [
    ([NOW - 10], "restart_suppressed_cooldown"),
    ([NOW - 200, NOW - 400], "restart_suppressed_budget"),
])
def test_restart_cooldown_and_hourly_budget_survive_monitor_invocations(config, monkeypatch, attempts, action):
    calls = stub_runtime(monkeypatch, service(active="failed", pid=0))
    write_json(config["monitor_state_path"], {"service": config["service"],
               "board_namespace": config["board_namespace"], "restart_attempts_at_epoch": attempts})
    result = monitor.monitor(config, apply=True)
    assert result["action"] == action
    assert calls == []


def test_corrupt_durable_budget_cannot_grant_restarts(config, monkeypatch):
    calls = stub_runtime(monkeypatch, service(active="failed", pid=0))
    Path(config["monitor_state_path"]).write_text("partial JSON")
    result = monitor.monitor(config, apply=True)
    assert result["action"] == "restart_suppressed_state_unreadable"
    assert Path(config["monitor_state_path"]).read_text() == "partial JSON"
    assert calls == []


def test_live_pid_without_fresh_evidence_is_stale(config):
    result = evaluate(config, native=evidence(last_heartbeat_at_epoch=NOW - 900,
                      last_progress_at_epoch=NOW - 900))
    assert result["status"] == "stale_heartbeat"
    assert result["restart_eligible"] and not result["healthy"]


def test_fresh_attempt_is_healthy(config):
    result = evaluate(config, native=evidence(active=True, active_task_id="T1",
                      active_started_at_epoch=NOW - 30))
    assert result["status"] == "working"
    assert result["healthy"] and not result["restart_eligible"]


def test_worker_log_cannot_extend_hard_task_deadline(config):
    result = evaluate(config, native=evidence(active=True, active_task_id="T1",
                      active_started_at_epoch=NOW - 601, active_log_at_epoch=NOW))
    assert result["status"] == "worker_deadline_exceeded"
    assert result["restart_eligible"]


def test_fresh_parent_heartbeat_does_not_hide_stale_worker(config):
    result = evaluate(config, native=evidence(active=True, active_task_id="T1",
                      active_started_at_epoch=NOW - 200, worker_heartbeat_at_epoch=NOW - 200,
                      last_progress_at_epoch=NOW - 200, last_heartbeat_at_epoch=NOW))
    assert result["status"] == "stale_worker"


def test_fresh_poll_with_ready_backlog_does_not_count_as_progress(config):
    result = evaluate(config, native=evidence(last_progress_at_epoch=NOW,
                      last_work_event_at_epoch=NOW - 800), previous={"ready_since_at_epoch": NOW - 900})
    assert result["status"] == "ready_backlog_stalled"
    assert result["restart_eligible"]


def test_intentional_report_only_is_visible_without_restart_storm(config):
    result = evaluate(config, native=evidence(implement=False))
    assert result["status"] == "report_only"
    assert "implement=false" in result["reason"]
    assert not result["healthy"] and not result["restart_eligible"]


def test_dependency_waiting_is_unhealthy_without_changing_contracts(config):
    result = evaluate(config, native=evidence(ready_count=0, blocked_count=1))
    assert result["status"] == "dependencies_waiting"
    assert not result["healthy"] and not result["restart_eligible"]


def test_native_backpressure_is_unhealthy_without_bypassing_retry_or_resource_fences(config):
    result = evaluate(config, native=evidence(native_progress_status="backpressure", native_status_at_epoch=NOW - 1))
    assert result["status"] == "backpressure"
    assert not result["healthy"] and not result["restart_eligible"]


def test_sharded_empty_lane_does_not_restart_for_another_lanes_global_backlog(config):
    config.update(task_shard_count=4, task_shard_index=2)
    write_json(config["native_state_path"], {"heartbeat_at": NOW - 5, "task_count": 12,
               "ready_count": 8, "selectable_ready_count": 0, "implementation_in_progress": False,
               "selection_idle_reason": "no_shard_selectable_ready_tasks"})
    observed = monitor.collect_evidence(config, NOW)
    assert observed["ready_count"] == 0
    assert observed["global_ready_count"] == 8
    assert observed["selectable_ready_count"] == 0
    result = evaluate(config, native=observed, previous={"ready_since_at_epoch": NOW - 900})
    assert result["status"] == "shard_idle"
    assert result["healthy"] and not result["restart_eligible"]


def test_latest_pass_selectable_count_overrides_stale_checkpoint_lists(config):
    config.update(task_shard_count=4, task_shard_index=2)
    write_json(config["checkpoint_path"], {"projection": {"task_state": {
        "heartbeat_at": NOW - 300, "ready_count": 8, "selectable_ready_count": 2,
        "selectable_ready_task_ids": ["T1", "T2"], "task_count": 12,
    }}})
    Path(config["event_path"]).write_text(json.dumps({"type": "daemon_pass", "timestamp": NOW - 1,
        "ready_count": 8, "selectable_ready_count": 0,
        "selection_idle_reason": "no_shard_selectable_ready_tasks"}))
    observed = monitor.collect_evidence(config, NOW)
    assert observed["ready_count"] == 0 and observed["global_ready_count"] == 8
    assert evaluate(config, native=observed)["status"] == "shard_idle"


def test_sharded_lane_does_not_infer_own_backlog_from_global_count(config):
    result = evaluate(config, native=evidence(sharded=True, selectable_ready_count=None, global_ready_count=8))
    assert result["status"] == "shard_backlog_unknown"
    assert not result["healthy"] and not result["restart_eligible"]


def test_startup_grace_allows_parent_to_initialize_child(config):
    result = evaluate(config, observed=service(age=5), native={"state_available": False})
    assert result["status"] == "starting"
    assert not result["restart_eligible"]


def test_native_checkpoint_and_live_pass_events_and_worker_log_are_observed(config):
    log_path = Path(config["state_dir"]) / "worker.log"
    log_path.write_text("working")
    os.utime(log_path, (NOW - 5, NOW - 5))
    write_json(config["checkpoint_path"], {"projection": {"task_state": {
        "ready_count": 4, "ready_task_ids": ["T1", "T2", "T3", "T4"],
        "task_count": 4, "heartbeat_at": NOW - 900,
    }}})
    write_json(config["status_path"], {"updated_at": NOW - 1, "implementation_enabled": True})
    events = [
        {"type": "implementation_started", "timestamp": NOW - 30,
         "board_namespace": "one-board", "task_id": "T1", "log_path": str(log_path)},
        {"type": "daemon_pass", "timestamp": NOW - 10, "ready_count": 3, "completed_count": 1,
         "task_count": 4, "active_task_id": "T2"},
        {"type": "task_selected", "timestamp": NOW - 1, "task_id": "T2"},
        {"type": "implementation_started", "timestamp": NOW, "board_namespace": "other-board", "task_id": "BAD"},
    ]
    Path(config["event_path"]).write_text("\n".join(json.dumps(value) for value in events) + '\n{"partial":')
    observed = monitor.collect_evidence(config, NOW)
    assert observed["state_source"] == config["checkpoint_path"]
    assert observed["ready_count"] == 3 and observed["completed_count"] == 1
    assert observed["active_task_id"] == "T1"
    assert observed["selected_task_id"] == "T2"
    assert observed["task_count"] == 4
    assert observed["active_log_at_epoch"] == NOW - 5
    assert observed["last_work_event_at_epoch"] == NOW - 30
    assert observed["implement"] is True


def test_completion_event_clears_checkpoint_active_worker(config):
    write_json(config["native_state_path"], {"active_task_id": "T1", "implementation_in_progress": True,
               "active_task_started_at": NOW - 100})
    Path(config["event_path"]).write_text("\n".join(json.dumps(value) for value in [
        {"type": "implementation_started", "task_id": "T1", "timestamp": NOW - 100},
        {"type": "implementation_finished", "task_id": "T1", "timestamp": NOW - 5},
    ]))
    observed = monitor.collect_evidence(config, NOW)
    assert observed["active"] is False


def test_repeated_selection_and_polling_does_not_renew_queue_progress(config):
    write_json(config["native_state_path"], {"task_count": 1, "ready_count": 1,
               "last_progress_at": NOW, "heartbeat_at": NOW, "active_task_id": "T1"})
    Path(config["event_path"]).write_text("\n".join(json.dumps(value) for value in [
        {"type": "task_selected", "task_id": "T1", "timestamp": NOW - 800},
        {"type": "task_selected", "task_id": "T1", "timestamp": NOW - 100},
        {"type": "implementation_skipped", "task_id": "T1", "timestamp": NOW - 50,
         "reason": "inflight_process"},
        {"type": "daemon_pass", "active_task_id": "T1", "ready_count": 1, "timestamp": NOW - 1},
    ]))
    observed = monitor.collect_evidence(config, NOW)
    assert observed["last_work_event_at_epoch"] is None
    assert observed["active"] is False
    result = evaluate(config, native=observed, previous={"ready_since_at_epoch": NOW - 900})
    assert result["status"] == "ready_backlog_stalled"


def test_fresh_poll_cannot_certify_historical_active_worker_with_stale_log(config):
    log_path = Path(config["state_dir"]) / "stale-worker.log"
    log_path.write_text("old output")
    os.utime(log_path, (NOW - 500, NOW - 500))
    write_json(config["checkpoint_path"], {"projection": {"task_state": {
        "active_task_id": "T1", "implementation_in_progress": True,
        "active_task_started_at": NOW - 500, "heartbeat_at": NOW - 500,
        "last_progress_at": NOW - 500, "active_log_path": str(log_path), "ready_count": 1,
    }}})
    write_json(config["status_path"], {"updated_at": NOW - 1, "implementation_enabled": True})
    Path(config["event_path"]).write_text("\n".join(json.dumps(value) for value in [
        {"type": "implementation_started", "task_id": "T1", "timestamp": NOW - 500,
         "log_path": str(log_path)},
        {"type": "implementation_skipped", "task_id": "T1", "timestamp": NOW - 2,
         "reason": "inflight_process"},
        {"type": "daemon_pass", "active_task_id": "T1", "ready_count": 1, "timestamp": NOW - 1},
    ]))
    observed = monitor.collect_evidence(config, NOW)
    assert observed["last_heartbeat_at_epoch"] == NOW - 1
    assert observed["worker_heartbeat_at_epoch"] == NOW - 500
    result = evaluate(config, native=observed)
    assert result["status"] == "stale_worker"
    assert result["restart_eligible"] and not result["healthy"]


def test_valid_json_with_invalid_restart_budget_is_preserved_and_suppresses_repair(config, monkeypatch):
    calls = stub_runtime(monkeypatch, service(active="failed", pid=0))
    original = {"service": config["service"], "board_namespace": config["board_namespace"],
                "restart_attempts_at_epoch": "lost records"}
    write_json(config["monitor_state_path"], original)
    result = monitor.monitor(config, apply=True)
    assert result["action"] == "restart_suppressed_state_unreadable"
    assert json.loads(Path(config["monitor_state_path"]).read_text()) == original
    assert calls == []


def test_native_lock_clear_event_terminalizes_old_attempt_without_claiming_completion(config):
    write_json(config["native_state_path"], {"active_task_id": "T1", "implementation_in_progress": True,
               "active_task_started_at": NOW - 100})
    Path(config["event_path"]).write_text("\n".join(json.dumps(value) for value in [
        {"type": "implementation_started", "task_id": "T1", "timestamp": NOW - 100},
        {"type": "implementation_lock_cleared", "task_id": "T1", "timestamp": NOW - 5},
    ]))
    observed = monitor.collect_evidence(config, NOW)
    assert observed["active"] is False
    assert observed["completed_count"] == 0


@pytest.mark.parametrize("bad_service", ["--all", "*.service", "a.service b.service", "a.timer"])
def test_unit_must_be_one_exact_service(config, tmp_path, bad_service):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({**config, "service": bad_service}))
    with pytest.raises(ValueError, match="exact systemd"):
        monitor.load_config(path)


@pytest.mark.parametrize("native_status", ["max_restarts_reached", "child_exited", "launch_failed"])
def test_native_unavailable_terminal_status_is_unhealthy_even_with_clean_exit(config, native_status):
    result = evaluate(config, observed=service(age=10), native=evidence(
        native_managed_status=native_status, native_status_at_epoch=NOW - 1,
        native_daemon_pid_alive=False, native_last_exit_code=0,
    ))
    assert result["status"] == "native_child_restarting"
    assert not result["healthy"] and not result["restart_eligible"]


@pytest.mark.parametrize("reason", [
    "implementation_retry_deferred:provider_task_contract_missing_validation_executables",
    "resource_claim_deferred:external/module", "provider_capacity_backoff",
    "all_selectable_ready_tasks_reached_max_task_attempts",
    "all_selectable_ready_tasks_deferred_by_resource_claim",
    "all_selectable_ready_tasks_deprioritized_as_off_mission",
    "no_eligible_ready_tasks_after_selection_filters", "virgin_transfer_rendezvous_pending",
])
def test_native_selection_fences_are_not_healthy_shard_idle(config, reason):
    result = evaluate(config, native=evidence(
        sharded=True, ready_count=0, selectable_ready_count=0, global_ready_count=8,
        selection_idle_reason=reason,
    ))
    assert result["status"] == "backpressure"
    assert not result["healthy"] and not result["restart_eligible"]


def test_zero_shard_backlog_without_positive_native_reason_is_unknown(config):
    result = evaluate(config, native=evidence(
        sharded=True, ready_count=0, selectable_ready_count=0, global_ready_count=8,
    ))
    assert result["status"] == "shard_backlog_unknown"
    assert not result["healthy"] and not result["restart_eligible"]


def _deferred_events(config, *, later=(), at=NOW - 400):
    config.update(task_shard_count=4, task_shard_index=2)
    write_json(config["native_state_path"], {
        "heartbeat_at": NOW - 10, "ready_count": 8, "selectable_ready_count": 0,
        "selection_idle_reason": "old checkpoint reason",
    })
    events = [{
        "type": "implementation_retry_deferred", "timestamp": at,
        "task_id": "T1", "canonical_task_cid": "cid-T1", "attempt_consumed": False,
        "reason": "provider_task_contract_missing_validation_executables",
        "backoff_seconds": 300, "contract_fingerprint": "sha256:contract",
        "provider_task_contract": {"missing_validation_executables": ["lake", "lean"]},
    }, {
        "type": "daemon_pass", "timestamp": NOW - 5, "ready_count": 8,
        "selectable_ready_count": 0, "selection_idle_reason": "no_shard_selectable_ready_tasks",
    }, *later]
    Path(config["event_path"]).write_text("\n".join(json.dumps(event) for event in events))


def test_expired_capability_cooldown_and_fresh_polls_do_not_prove_shard_idle(config, monkeypatch):
    calls = stub_runtime(monkeypatch, service())
    _deferred_events(config, later=[{
        "type": "task_selected", "timestamp": NOW - 1,
        "task_id": "T1", "canonical_task_cid": "cid-T1",
    }])
    result = monitor.monitor(config, apply=True)
    assert result["status"] == "backpressure"
    assert not result["healthy"] and not result["restart_eligible"]
    assert calls == []
    observed = result["evidence"]
    assert observed["selection_idle_reason"] == "no_shard_selectable_ready_tasks"
    pending = observed["unresolved_retry_deferrals"]
    assert len(pending) == 1
    assert pending[0]["missing_validation_executables"] == ["lake", "lean"]
    assert pending[0]["retry_after_epoch"] == NOW - 100
    assert pending[0]["contract_fingerprint"] == "sha256:contract"


def test_real_later_attempt_clears_own_capability_deferral(config, tmp_path):
    log = tmp_path / "worker.log"
    log.write_text("actual worker output")
    os.utime(log, (NOW - 1, NOW - 1))
    _deferred_events(config, later=[{
        "type": "implementation_started", "timestamp": NOW - 2,
        "task_id": "T1", "canonical_task_cid": "cid-T1", "log_path": str(log),
    }])
    observed = monitor.collect_evidence(config, NOW)
    assert observed["unresolved_retry_deferrals"] == []
    assert evaluate(config, native=observed)["status"] == "working"


def test_different_canonical_task_with_same_alias_cannot_clear_deferral(config):
    _deferred_events(config, later=[{
        "type": "implementation_started", "timestamp": NOW - 2,
        "task_id": "T1", "canonical_task_cid": "different-cid",
    }])
    observed = monitor.collect_evidence(config, NOW)
    assert observed["unresolved_retry_deferrals"][0]["canonical_task_cid"] == "cid-T1"


def test_native_completed_projection_retires_old_deferral(config):
    _deferred_events(config, later=[{
        "type": "daemon_pass", "timestamp": NOW - 1, "ready_count": 7,
        "selectable_ready_count": 0, "execution_slice_task_statuses": {"T1": "completed"},
        "selection_idle_reason": "no_shard_selectable_ready_tasks",
    }])
    observed = monitor.collect_evidence(config, NOW)
    assert observed["unresolved_retry_deferrals"] == []
    assert evaluate(config, native=observed)["status"] == "shard_idle"


def test_deferral_history_is_bounded_and_does_not_parse_provider_prose(config):
    events = [{"type": "implementation_retry_deferred", "timestamp": NOW - 200 + index,
        "task_id": f"T{index}", "attempt_consumed": False, "reason": "resource_wait"}
        for index in range(150)]
    events.append({"type": "thought", "timestamp": NOW - 1,
        "text": "implementation_retry_deferred: missing lake; provider quota"})
    Path(config["event_path"]).write_text("\n".join(json.dumps(event) for event in events))
    observed = monitor.collect_evidence(config, NOW)
    assert len(observed["unresolved_retry_deferrals"]) == 128
    assert {item["reason"] for item in observed["unresolved_retry_deferrals"]} == {"resource_wait"}


def test_typed_file_input_diagnostics_are_bounded_and_retire_only_on_real_attempt(config):
    _deferred_events(config)
    events = [json.loads(line) for line in Path(config["event_path"]).read_text().splitlines()]
    events[0]["reason"] = "provider_task_contract_required_validation_file_pins_unavailable"
    events[0]["provider_task_contract"] = {"unavailable_validation_file_pins": [
        {"path": "/missing/input-" + str(index), "reason": "missing"} for index in range(64)]}
    Path(config["event_path"]).write_text("\n".join(json.dumps(event) for event in events))
    observed = monitor.collect_evidence(config, NOW)
    item = observed["unresolved_retry_deferrals"][0]
    assert item["unavailable_validation_file_pin_count"] == 64
    assert item["unavailable_validation_file_pins"] == events[0]["provider_task_contract"]["unavailable_validation_file_pins"][:8]
    assert item["file_pin_evidence_truncated"] is True
    events.append({"type": "implementation_started", "timestamp": NOW - 1,
                   "task_id": "T1", "canonical_task_cid": "cid-T1"})
    Path(config["event_path"]).write_text("\n".join(json.dumps(event) for event in events))
    assert monitor.collect_evidence(config, NOW)["unresolved_retry_deferrals"] == []


@pytest.mark.parametrize("stale", [False, True])
def test_counted_ready_tasks_all_retry_fenced_never_restart(config, monkeypatch, stale):
    calls = stub_runtime(monkeypatch, service(age=2000))
    _deferred_events(config, at=NOW - 1500)
    # Match the actual strict SHA256 lane for this native identity.
    config["task_shard_index"] = int(monitor.hashlib.sha256(b"T1").hexdigest()[:8], 16) % 4
    events = [json.loads(line) for line in Path(config["event_path"]).read_text().splitlines()]
    events[-1].update(selectable_ready_count=1, selection_idle_reason="",
        execution_slice_task_statuses={"T1": "ready"}, execution_slice_task_cids_by_id={"T1": "cid-T1"})
    if not stale:
        events.insert(1, {"type": "implementation_finished", "timestamp": NOW - 10,
                         "task_id": "OTHER", "returncode": 0})
    Path(config["event_path"]).write_text("\n".join(json.dumps(event) for event in events))
    result = monitor.monitor(config, apply=True)
    assert result["status"] == "backpressure" and result["healthy"] is False
    assert result["restart_eligible"] is False and calls == []
    assert result["evidence"]["all_ready_tasks_retry_fenced"] is True


@pytest.mark.parametrize("mismatch", ["other_ready", "different_cid", "count_mismatch"])
def test_partial_or_stale_identity_fence_does_not_hide_real_ready_stall(config, mismatch):
    _deferred_events(config, at=NOW - 1500)
    config["task_shard_count"] = 1
    events = [json.loads(line) for line in Path(config["event_path"]).read_text().splitlines()]
    statuses = {"T1": "ready", **({"T2": "ready"} if mismatch == "other_ready" else {})}
    events[-1].update(ready_count=2 if mismatch in {"other_ready", "count_mismatch"} else 1,
        selectable_ready_count=len(statuses), selection_idle_reason="",
        execution_slice_task_statuses=statuses,
        execution_slice_task_cids_by_id={"T1": "other-cid" if mismatch == "different_cid" else "cid-T1"})
    Path(config["event_path"]).write_text("\n".join(json.dumps(event) for event in events))
    observed = monitor.collect_evidence(config, NOW)
    assert observed["all_ready_tasks_retry_fenced"] is False
    result = monitor.evaluate_health(config, service(age=2000), observed, {}, NOW, MONOTONIC)
    assert result["status"] == "ready_backlog_stalled" and result["restart_eligible"] is True


def test_deep_json_state_is_an_unreadable_observation(tmp_path):
    path = tmp_path / "deep.json"
    path.write_text('{"nested":' + '[' * 10000 + '0' + ']' * 10000 + '}')
    value, error = monitor._read_json(str(path))
    assert value == {} and error


def test_deep_json_event_preserves_other_records_and_reports_error(config):
    path = Path(config["event_path"])
    valid = [{"type": "daemon_pass", "timestamp": NOW - 2},
             {"type": "daemon_pass", "timestamp": NOW - 1}]
    deep = '{"nested":' + '[' * 10000 + '0' + ']' * 10000 + '}'
    path.write_text(json.dumps(valid[0]) + '\n' + deep + '\n' + json.dumps(valid[1]))
    events, error = monitor._event_tail(config, NOW)
    assert [event["timestamp"] for event in events] == [NOW - 2, NOW - 1]
    assert error == "event_json_nesting_limit"


def test_deep_restart_budget_never_grants_fresh_attempts_or_rewrites_state(config, monkeypatch):
    calls = stub_runtime(monkeypatch, service(active="failed", pid=0))
    path = Path(config["monitor_state_path"])
    original = '{"nested":' + '[' * 10000 + '0' + ']' * 10000 + '}'
    path.write_text(original)
    result = monitor.monitor(config, apply=True)
    assert result["action"] == "restart_suppressed_state_unreadable"
    assert result["healthy"] is False and calls == []
    assert path.read_text() == original
    assert json.loads(Path(config["health_path"]).read_text()) == result


def test_unwritable_observation_log_still_publishes_lane_health(config, monkeypatch):
    calls = stub_runtime(monkeypatch, service())
    monkeypatch.setattr(monitor, "collect_evidence", lambda *args: evidence())
    def fail(*args):
        raise OSError("observation log unavailable")
    monkeypatch.setattr(monitor, "_append_action", fail)
    result = monitor.monitor(config, apply=True)
    assert result["status"] == "monitor_error" and result["healthy"] is False
    assert result["observed_health_status"] == "ready"
    assert result["action"] == "none" and result["restart_eligible"] is False
    assert "observation log unavailable" in result["observation_log_error"]
    assert calls == [] and json.loads(Path(config["health_path"]).read_text()) == result


def test_restart_intent_log_failure_keeps_charged_budget_and_prevents_effect(config, monkeypatch):
    calls = stub_runtime(monkeypatch, service(active="failed", pid=0))
    def fail(*args):
        raise OSError("intent log unavailable")
    monkeypatch.setattr(monitor, "_append_action", fail)
    with pytest.raises(OSError, match="intent log unavailable"):
        monitor.monitor(config, apply=True)
    saved = json.loads(Path(config["monitor_state_path"]).read_text())
    assert saved["restart_attempts_at_epoch"] == [NOW] and calls == []
    result = monitor.monitor(config, apply=True)
    assert result["action"] == "restart_suppressed_cooldown"
    assert result["status"] == "monitor_error" and result["restart_eligible"] is False
    assert json.loads(Path(config["monitor_state_path"]).read_text())["restart_attempts_at_epoch"] == [NOW]
    assert calls == [] and json.loads(Path(config["health_path"]).read_text()) == result


def test_post_restart_observation_log_failure_preserves_actual_action(config, monkeypatch):
    calls = stub_runtime(monkeypatch, service(active="failed", pid=0))
    original = monitor._append_action
    def append(path, event):
        if "status" in event:
            raise OSError("final observation log unavailable")
        return original(path, event)
    monkeypatch.setattr(monitor, "_append_action", append)
    result = monitor.monitor(config, apply=True)
    assert calls == [["systemctl", "--user", "restart", "one-board.service"]]
    assert result["action"] == "restart_requested" and result["status"] == "monitor_error"
    assert result["restart_attempts_last_hour"] == 1 and result["restart_eligible"] is False
    assert json.loads(Path(config["monitor_state_path"]).read_text())["restart_attempts_at_epoch"] == [NOW]
    assert json.loads(Path(config["health_path"]).read_text()) == result


@pytest.mark.parametrize("status_age,service_age", [
    (900, 2000), (90, 70), (None, 2000),
], ids=["expired-status", "previous-service-generation", "undated-status"])
def test_stale_native_backpressure_cannot_hide_fresh_ready_stall(config, monkeypatch, status_age, service_age):
    calls = stub_runtime(monkeypatch, service(age=service_age))
    write_json(config["monitor_state_path"], {
        "service": config["service"], "board_namespace": config["board_namespace"],
        "first_seen_at_epoch": NOW - 2000, "ready_since_at_epoch": NOW - 1000,
        "restart_attempts_at_epoch": [],
    })
    write_json(config["native_state_path"], {
        "heartbeat_at": NOW - 900, "task_count": 1, "ready_count": 1,
        "task_statuses": {"T1": "ready"}, "last_progress_at": NOW - 900,
    })
    old_status = {"implementation_progress_status": "backpressure"}
    if status_age is not None:
        old_status["updated_at"] = NOW - status_age
    write_json(config["status_path"], old_status)
    Path(config["event_path"]).write_text(json.dumps({
        "type": "daemon_pass", "timestamp": NOW - 1, "ready_count": 1,
        "selectable_ready_count": 1, "task_count": 1, "selection_idle_reason": "",
        "execution_slice_task_statuses": {"T1": "ready"},
    }))
    result = monitor.monitor(config, apply=True)
    assert result["evidence"]["last_heartbeat_at_epoch"] == NOW - 1
    assert result["evidence"]["active"] is False
    assert result["evidence"]["unresolved_retry_deferrals"] == []
    assert result["evidence"]["all_ready_tasks_retry_fenced"] is False
    assert result["status"] == "ready_backlog_stalled"
    assert result["healthy"] is False and result["restart_eligible"] is True
    assert result["action"] == "restart_requested"
    assert calls == [["systemctl", "--user", "restart", config["service"]]]
    saved = json.loads(Path(config["monitor_state_path"]).read_text())
    assert saved["restart_attempts_at_epoch"] == [NOW]


def test_current_native_backpressure_still_prevents_idle_restart(config, monkeypatch):
    calls = stub_runtime(monkeypatch, service(age=2000))
    write_json(config["monitor_state_path"], {
        "service": config["service"], "board_namespace": config["board_namespace"],
        "first_seen_at_epoch": NOW - 2000, "ready_since_at_epoch": NOW - 1000,
        "restart_attempts_at_epoch": [],
    })
    write_json(config["native_state_path"], {
        "heartbeat_at": NOW - 1, "ready_count": 1, "task_count": 1,
    })
    write_json(config["status_path"], {
        "updated_at": NOW - 1, "implementation_progress_status": "backpressure",
    })
    result = monitor.monitor(config, apply=True)
    assert result["status"] == "backpressure"
    assert result["healthy"] is False and result["restart_eligible"] is False
    assert result["action"] == "none" and calls == []
    assert result["restart_attempts_last_hour"] == 0


@pytest.mark.parametrize("reason", [
    "implementation_retry_deferred:provider_task_contract_missing_validation_executables",
    "resource_claim_deferred:external/module",
], ids=["missing-capability-fence", "resource-claim-fence"])
def test_stale_native_label_does_not_remove_current_selection_fence(config, monkeypatch, reason):
    calls = stub_runtime(monkeypatch, service(age=2000))
    write_json(config["native_state_path"], {
        "heartbeat_at": NOW - 900, "ready_count": 1, "task_count": 1,
    })
    write_json(config["status_path"], {
        "updated_at": NOW - 900, "implementation_progress_status": "backpressure",
    })
    Path(config["event_path"]).write_text(json.dumps({
        "type": "daemon_pass", "timestamp": NOW - 1, "ready_count": 1,
        "selectable_ready_count": 1, "selection_idle_reason": reason,
    }))
    result = monitor.monitor(config, apply=True)
    assert result["evidence"]["selection_idle_reason"] == reason
    assert result["status"] == "backpressure" and result["restart_eligible"] is False
    assert result["action"] == "none" and calls == []


def test_stale_native_label_preserves_exact_ready_task_retry_fence(config, monkeypatch):
    calls = stub_runtime(monkeypatch, service(age=2000))
    _deferred_events(config, at=NOW - 1500)
    config["task_shard_index"] = int(monitor.hashlib.sha256(b"T1").hexdigest()[:8], 16) % 4
    events = [json.loads(line) for line in Path(config["event_path"]).read_text().splitlines()]
    events[-1].update(selectable_ready_count=1, selection_idle_reason="",
        execution_slice_task_statuses={"T1": "ready"},
        execution_slice_task_cids_by_id={"T1": "cid-T1"})
    Path(config["event_path"]).write_text("\n".join(json.dumps(event) for event in events))
    write_json(config["status_path"], {
        "updated_at": NOW - 900, "implementation_progress_status": "backpressure",
    })
    result = monitor.monitor(config, apply=True)
    assert result["evidence"]["all_ready_tasks_retry_fenced"] is True
    assert result["evidence"]["retry_fenced_ready_task_ids"] == ["T1"]
    assert result["status"] == "backpressure" and result["restart_eligible"] is False
    assert result["action"] == "none" and calls == []


@pytest.mark.parametrize("replacement", [
    service(age=1, pid=456),
    service(age=2000, pid=123),
], ids=["new-main-pid", "same-pid-different-birth"])
def test_restart_rechecks_service_birth_before_charging(config, monkeypatch, replacement):
    calls = stub_runtime(monkeypatch, service())
    observations = iter([service(), replacement])
    monkeypatch.setattr(monitor, "systemd_show", lambda _: next(observations))

    result = monitor.monitor(config, apply=True)

    assert result["action"] == "restart_suppressed_service_changed"
    assert result["restart_recheck"]["service_identity_unchanged"] is False
    assert calls == [] and result["restart_attempts_last_hour"] == 0
    assert json.loads(Path(config["monitor_state_path"]).read_text())["restart_attempts_at_epoch"] == []


@pytest.mark.parametrize("update,status", [
    ({"ready_count": 0, "completed_task_ids": ["T1"]}, "completed"),
    ({"implementation_in_progress": True, "active_task_id": "T1",
      "active_task_started_at": NOW - 1}, "working"),
    ({"selection_idle_reason": "resource_claim_deferred:repository"}, "backpressure"),
], ids=["completed-idle", "worker-started", "native-resource-fence"])
def test_restart_rechecks_useful_work_and_native_fences(config, monkeypatch, update, status):
    calls = stub_runtime(monkeypatch, service())
    write_json(config["monitor_state_path"], {
        "service": config["service"], "board_namespace": config["board_namespace"],
        "first_seen_at_epoch": NOW - 2000, "ready_since_at_epoch": NOW - 1000,
        "restart_attempts_at_epoch": [],
    })
    native = {"heartbeat_at": NOW - 1000, "ready_count": 1, "task_count": 1}
    write_json(config["native_state_path"], native)
    collect = monitor.collect_evidence
    observations = []

    def collect_after_native_progress(config, now):
        if observations:
            write_json(config["native_state_path"], {**native, "heartbeat_at": NOW - 1, **update})
        evidence = collect(config, now)
        observations.append(evidence)
        return evidence

    monkeypatch.setattr(monitor, "collect_evidence", collect_after_native_progress)
    result = monitor.monitor(config, apply=True)

    assert result["action"] == "restart_suppressed_health_changed"
    assert result["status"] == status and result["restart_eligible"] is False
    assert result["restart_recheck"]["previous_status"] == "stale_heartbeat"
    assert result["restart_recheck"]["service_identity_unchanged"] is True
    assert len(observations) == 2 and calls == []
    saved = json.loads(Path(config["monitor_state_path"]).read_text())
    assert saved["restart_attempts_at_epoch"] == []
    if status in {"completed", "working"}:
        assert "ready_since_at_epoch" not in saved


@pytest.mark.parametrize("replacement", [
    {"error": "systemd unavailable"},
    {**service(), "ExecMainStartTimestampMonotonic": "0"},
    {**service(), "ExecMainStartTimestampMonotonic": "invalid"},
], ids=["observation-failed", "missing-running-birth", "invalid-birth"])
def test_restart_recheck_without_verified_identity_fails_closed(config, monkeypatch, replacement):
    calls = stub_runtime(monkeypatch, service())
    observations = iter([service(), replacement])
    monkeypatch.setattr(monitor, "systemd_show", lambda _: next(observations))

    result = monitor.monitor(config, apply=True)

    assert result["action"] == "restart_suppressed_identity_unavailable"
    assert calls == [] and result["restart_attempts_last_hour"] == 0
    assert json.loads(Path(config["monitor_state_path"]).read_text())["restart_attempts_at_epoch"] == []



def test_replacement_service_does_not_inherit_expired_ready_deadline(config, monkeypatch):
    calls = stub_runtime(monkeypatch, service())
    observations = iter([service(), service(age=1, pid=456)])
    monkeypatch.setattr(monitor, "systemd_show", lambda _: next(observations))
    write_json(config["monitor_state_path"], {
        "service": config["service"], "board_namespace": config["board_namespace"],
        "first_seen_at_epoch": NOW - 2000, "ready_since_at_epoch": NOW - 1000,
        "restart_attempts_at_epoch": [],
    })
    write_json(config["native_state_path"], {"heartbeat_at": NOW - 1, "ready_count": 1})

    result = monitor.monitor(config, apply=True)

    assert result["action"] == "restart_suppressed_service_changed" and calls == []
    saved = json.loads(Path(config["monitor_state_path"]).read_text())
    assert saved["ready_since_at_epoch"] == NOW and saved["restart_attempts_at_epoch"] == []


# Execute the actual staged native emission helper without importing its model/runtime package.
import ast
import types
from datetime import datetime, timezone
_NATIVE_SOURCE = SOURCE.parents[1] / "todo_daemon/implementation_supervisor.py"
_tree = ast.parse(_NATIVE_SOURCE.read_text())
_native_node = next(n for n in ast.walk(_tree) if isinstance(n, ast.FunctionDef) and n.name == "_unavailable_child_recovery_status")
_native_unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), _native_node], type_ignores=[])
import sys
import importlib.util
import copy
_base = SOURCE.parents[1]
for _name, _directory in [("_watchdog_owners", _base), ("_watchdog_owners.proof", _base/"proof"),
        ("_watchdog_owners.control", _base/"control"), ("_watchdog_owners.task_sources", _base/"task_sources"),
        ("_watchdog_owners.todo_daemon", _base/"todo_daemon")]:
    _package = types.ModuleType(_name);_package.__path__=[str(_directory)];sys.modules[_name]=_package
for _name, _relative in [("proof.formal_verification_contracts", "proof/formal_verification_contracts.py"),
        ("control.control_contracts", "control/control_contracts.py"),
        ("task_sources.taskboard_store", "task_sources/taskboard_store.py"),
        ("todo_daemon.supervisor_activity", "todo_daemon/supervisor_activity.py")]:
    _spec=importlib.util.spec_from_file_location("_watchdog_owners."+_name,_base/_relative)
    _module=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_module;_spec.loader.exec_module(_module)
_Store=sys.modules["_watchdog_owners.task_sources.taskboard_store"].ProjectionDeltaCheckpointStore
_Window=sys.modules["_watchdog_owners.todo_daemon.supervisor_activity"].SupervisorMaintenanceWindow
_native_globals = dict(Mapping=__import__("collections.abc", fromlist=["Mapping"]).Mapping,
    load_json_dict=lambda p: json.loads(Path(p).read_text()) if Path(p).is_file() else None,
    checkout_mutation_lock_path=lambda p: Path(p)/"global-checkout.lock", ProjectionDeltaCheckpointStore=_Store,
    SupervisorMaintenanceWindow=_Window, os=types.SimpleNamespace(getpid=lambda:123),
    read_process_birth=lambda pid:types.SimpleNamespace(to_dict=lambda:dict(pid=123,start_time_ticks=99,boot_id="test-boot")),
    utc_now=lambda:datetime.fromtimestamp(NOW-1,timezone.utc).isoformat(), PROOF_ROLLOUT_PROJECTION_FIELDS=(),
    write_json_atomic=lambda p,v:write_json(p,v))
exec(compile(ast.fix_missing_locations(_native_unit), str(_NATIVE_SOURCE), "exec"), _native_globals)
_native_emit = _native_globals["_unavailable_child_recovery_status"]
_writer_node=next(n for n in ast.walk(_tree) if isinstance(n,ast.FunctionDef) and n.name=="_write_supervisor_maintenance_status")
_writer_unit=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),_writer_node],type_ignores=[])
exec(compile(ast.fix_missing_locations(_writer_unit),str(_NATIVE_SOURCE),"exec"),_native_globals)
_native_write=_native_globals["_write_supervisor_maintenance_status"]
for _function in ("_publish_unavailable_child_recovery", "_run_forever_loop"):
    _node=next(n for n in ast.walk(_tree) if isinstance(n,ast.FunctionDef) and n.name==_function)
    _unit=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),_node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(_unit),str(_NATIVE_SOURCE),"exec"),_native_globals)
_native_publish=_native_globals["_publish_unavailable_child_recovery"]
_native_parent_loop=_native_globals["_run_forever_loop"]


def native_owner(config, **changes):
    state = Path(config["native_state_path"])
    write_json(state, dict(active_task_id="", implementation_in_progress=False, active_phase=""))
    owner = types.SimpleNamespace(config=types.SimpleNamespace(state_path=state, repo_root=Path(config["state_dir"]),
        events_path=Path(config["state_dir"])/(config["state_prefix"]+"_supervisor_events.jsonl"),strategy_path=Path(config["state_dir"])/"strategy.json",board_namespace=config["board_namespace"]),
        _database_authority_enabled=lambda: False, _active_agent_worker_processes=lambda: [],
        _active_validation_subprocess_exists=lambda: False, _current_supervisor_checkout_lease=lambda: None,
        _repo_merge_lock_path=lambda: Path(config["state_dir"])/"board-checkout.lock")
    owner._supervisor_status_path=lambda:Path(config["status_path"])
    owner._supervisor_maintenance_timeout_seconds=lambda:900
    owner._watchdog_startup_grace_seconds=lambda:60
    owner.config.stale_seconds=120;owner.config.check_interval=20
    owner.config.implement=True;owner.config.reconciliation_only=False
    owner.config.state_dir=Path(config["state_dir"]);owner.config.state_prefix=config["state_prefix"]
    owner.config.task_prefix="## TEST-";owner.config.task_shard_index=1
    owner._managed_daemon_pid_path=lambda:Path(config["state_dir"])/"managed.pid"
    owner._proof_rollout_status_fields=lambda _:{}
    owner._autonomous_unstall_status=lambda:{}
    owner._control_plane_status_projection=lambda:{}
    owner._persist_own_supervisor_pid=lambda:None
    owner._write_supervisor_maintenance_receipt=lambda *args,**kwargs:None
    owner._unavailable_child_recovery_status=types.MethodType(_native_emit,owner)
    owner.__dict__.update(changes)
    return owner


def process_snapshot(birth=99, children=None):
    return dict(pid=123, birth_ticks=birth, boot_id="test-boot", children=children or [])


def recovery_evidence(config, **changes):
    proof = _native_emit(native_owner(config), daemon_pid=None,
        observed_at=datetime.fromtimestamp(NOW-1, timezone.utc).isoformat(),
        supervisor_birth=dict(pid=123, start_time_ticks=99, boot_id="test-boot"))
    value = evidence(active=False, implement=True, native_managed_status="max_restarts_reached",
        native_daemon_pid_alive=False, native_status_at_epoch=NOW-1, native_supervisor_pid=123,
        native_run_id="attempt-1", native_restart_count=8, native_active_worker_count=0,
        native_active_worker_pids=[], native_worker_descendant_count=0, native_worker_phase="",
        native_recycle_safe=False, native_unavailable_child_recovery=proof)
    value.update(changes)
    return value


def aged_recovery(config, monkeypatch, native):
    monkeypatch.setattr(monitor, "_process_snapshot", lambda _: process_snapshot())
    prior = monitor._native_recovery_observation(config, service(), native, {}, NOW, MONOTONIC)
    prior["since_at_epoch"] = NOW-601
    return {"native_recovery": prior}


def test_actual_native_dead_child_guard_is_reachable_without_idle_recycle_flag(config, monkeypatch):
    value = recovery_evidence(config)
    assert value["native_recycle_safe"] is False and value["native_unavailable_child_recovery"]["safe"] is True
    prior = aged_recovery(config, monkeypatch, value)
    observed = monitor._native_recovery_observation(config, service(), value, prior, NOW, MONOTONIC)
    assert observed["expired"] is True and observed["quiescent"] is True


@pytest.mark.parametrize("blocker", ["daemon", "database", "task", "worker", "validation", "lease", "board_lock", "global_lock", "error"])
def test_actual_native_emitter_preserves_every_native_work_and_checkout_fence(config, blocker):
    owner = native_owner(config)
    daemon = None
    if blocker == "daemon":daemon=456
    if blocker == "database":owner._database_authority_enabled=lambda: True
    if blocker == "task":write_json(owner.config.state_path, dict(active_task_id="active", implementation_in_progress=True, active_phase="validation"))
    if blocker == "worker":owner._active_agent_worker_processes=lambda: [456]
    if blocker == "validation":owner._active_validation_subprocess_exists=lambda: True
    if blocker == "lease":owner._current_supervisor_checkout_lease=lambda: object()
    if blocker == "board_lock":owner._repo_merge_lock_path().write_text("retained")
    if blocker == "global_lock":(owner.config.repo_root/"global-checkout.lock").write_text("retained")
    if blocker == "error":owner._current_supervisor_checkout_lease=lambda: (_ for _ in ()).throw(OSError("unreadable"))
    proof = _native_emit(owner, daemon_pid=daemon, observed_at="now", supervisor_birth={})
    assert proof["safe"] is False and proof["blockers"] and proof["lease_or_task_mutation_authority"] is False


def test_failed_run_and_restart_counter_changes_do_not_renew_unavailable_deadline(config, monkeypatch):
    native = recovery_evidence(config)
    prior = aged_recovery(config, monkeypatch, native)
    native.update(native_run_id="failed-next-attempt", native_restart_count=9)
    observed = monitor._native_recovery_observation(config, service(), native, prior, NOW, MONOTONIC)
    assert observed["expired"] is True and observed["age_seconds"] == 601


@pytest.mark.parametrize("changed", ["birth", "healthy", "work"])
def test_new_parent_successful_child_or_actual_work_clears_recovery_grace(config, monkeypatch, changed):
    native = recovery_evidence(config);prior=aged_recovery(config, monkeypatch, native)
    if changed == "birth":monkeypatch.setattr(monitor, "_process_snapshot", lambda _: process_snapshot(birth=100))
    if changed == "healthy":native["native_daemon_pid_alive"]=True
    if changed == "work":native["last_work_event_at_epoch"]=NOW-1
    observed=monitor._native_recovery_observation(config, service(), native, prior, NOW, MONOTONIC)
    assert observed["expired"] is False


def test_first_native_maintenance_deadline_is_respected_but_not_renewed_by_failed_attempts(config, monkeypatch):
    native=recovery_evidence(config, native_managed_status="agentic_maintenance_started")
    native["native_maintenance_window"]=dict(schema="ipfs_accelerate_py/agent-supervisor/supervisor-maintenance@1",
        phase="running", started_at=datetime.fromtimestamp(NOW-1000, timezone.utc).isoformat(),
        deadline_at=datetime.fromtimestamp(NOW+100, timezone.utc).isoformat(), timeout_seconds=1100, finished_at=None)
    prior=aged_recovery(config, monkeypatch, native)
    prior["native_recovery"]["maintenance_deadline_at_epoch"]=NOW-1
    native["native_maintenance_window"]["deadline_at"]=datetime.fromtimestamp(NOW+500, timezone.utc).isoformat()
    native["native_maintenance_window"]["timeout_seconds"]=1500
    observed=monitor._native_recovery_observation(config, service(), native, prior, NOW, MONOTONIC)
    assert observed["expired"] is True and observed["maintenance_deadline_at_epoch"] == NOW-1
    fresh=monitor._native_recovery_observation(config, service(), native, {}, NOW, MONOTONIC)
    assert fresh["expired"] is False and fresh["maintenance_deadline_at_epoch"] == NOW+500


def test_native_recovery_grace_budget_and_before_effect_recheck(config, monkeypatch):
    calls=stub_runtime(monkeypatch, service())
    monkeypatch.setattr(monitor, "_process_snapshot", lambda _: process_snapshot())
    monkeypatch.setattr(monitor, "collect_evidence", lambda *_: recovery_evidence(config))
    monitor.monitor(config, apply=True)
    state=json.loads(Path(config["monitor_state_path"]).read_text());state["native_recovery"]["since_at_epoch"]=NOW-601
    write_json(config["monitor_state_path"],state)
    processes=iter([process_snapshot(),process_snapshot(children=[456])]);monkeypatch.setattr(monitor,"_process_snapshot",lambda _:next(processes))
    assert monitor.monitor(config,apply=True)["action"] == "restart_suppressed_recovery_changed" and calls == []
    monkeypatch.setattr(monitor,"_process_snapshot",lambda _:process_snapshot())
    assert monitor.monitor(config,apply=True)["action"] == "restart_requested"
    assert monitor.monitor(config,apply=True)["action"] == "restart_suppressed_cooldown"
    assert calls == [["systemctl","--user","restart",config["service"]]]


def checkpoint_owner(config, *, foreign=False):
    owner=native_owner(config);owner.config.state_path.unlink()
    # Exact independently captured native checkpoint, retained from 2026-10-21.
    # SHA256 54b6b15e1b84b3f95e6985b2d1af81e5e0f62ffe919225bca99e3a0d4f98609a.
    # Fixture bytes stay intact; the owner rebinds only the three paths below.
    original=json.loads((Path(__file__).with_name("fixtures")/"legal_ir_deployed_event_driven_checkpoint.json").read_text())
    projection=copy.deepcopy(original["projection"])
    projection["result"].update(state_path=str(owner.config.state_path),events_path=str(owner.config.state_dir/(owner.config.state_prefix+"_events.jsonl")),strategy_path=str(owner.config.strategy_path))
    if foreign:projection["result"]["state_path"]="/foreign/board.json"
    path=owner.config.state_path.with_name(owner.config.state_path.stem+".event-driven-checkpoint.json")
    _Store(path).materialize(projection,original["cursor"])
    return owner,path


def test_real_maintenance_writer_emits_reachable_safe_checkpoint_payload_and_monitor_uses_it(config,monkeypatch):
    owner,checkpoint=checkpoint_owner(config)
    before=checkpoint.read_bytes();write_json(config["status_path"],dict(active_worker_count=0,active_worker_pids=[],worker_descendant_count=0,worker_phase="",run_id="failed-1",restart_count=8,implementation_enabled=True))
    _native_write(owner,"recovery",status="running",started_at=datetime.fromtimestamp(NOW-2000,timezone.utc).isoformat())
    status=json.loads(Path(config["status_path"]).read_text());assert not owner.config.state_path.exists() and checkpoint.read_bytes()==before
    proof=status["unavailable_child_recovery"];assert proof["safe"] is True and proof["state_source"]==str(checkpoint)
    assert status["supervisor_maintenance"]["schema"]=="ipfs_accelerate_py/agent-supervisor/supervisor-maintenance@1"
    native=monitor.collect_evidence(config,NOW);assert native["native_managed_status"]=="agentic_maintenance_started"
    prior=aged_recovery(config,monkeypatch,native)
    observed=monitor._native_recovery_observation(config,service(),native,prior,NOW,MONOTONIC)
    assert observed["expired"] is True and observed["quiescent"] is True


@pytest.mark.parametrize("kind",["foreign","malformed","missing","tampered"])
def test_checkpoint_failure_never_defaults_to_empty_quiescence(config,kind):
    owner,checkpoint=checkpoint_owner(config,foreign=kind=="foreign")
    if kind=="malformed":checkpoint.write_text("[]")
    if kind=="missing":checkpoint.unlink()
    if kind=="tampered":
        value=json.loads(checkpoint.read_text());value["projection"]["task_state"]["active_task_id"]="resealed-not-provided";write_json(checkpoint,value)
    proof=_native_emit(owner,daemon_pid=None,observed_at="now",supervisor_birth={})
    assert proof["safe"] is False and proof["blockers"] and not owner.config.state_path.exists()


@pytest.mark.parametrize("leader",[True,False])
def test_actual_parent_recovery_publishes_fresh_guard_for_leader_and_nonleader(config,monkeypatch,leader):
    owner,checkpoint=checkpoint_owner(config)
    write_json(config["status_path"],dict(active_worker_count=9,active_worker_pids=[456],worker_descendant_count=9,worker_phase="old-finished-work"))
    owner.ensure_event_log_file=lambda:None;owner.ensure_managed_daemon_pid_file=lambda:None
    owner._is_board_maintenance_leader=lambda:leader;owner.repair_main_checkout_merge_state=lambda:None
    owner.run_once=lambda **kwargs:{};owner._record_event=lambda *args:None
    result=types.SimpleNamespace(status="max_restarts_reached",restart_count=8,last_exit_code=1,last_recycle_reason="launch_failed",last_run_id="failed-8",last_log_path="none")
    owner.shared_supervisor_loop_class=lambda *args,**kwargs:types.SimpleNamespace(run=lambda:result)
    owner.build_supervisor_loop_config=lambda:None;owner._supervisor_loop_watchdog_decision=lambda *args:None
    owner._publish_unavailable_child_recovery=types.MethodType(_native_publish,owner)
    owner._supervisor_loop_recovery_delay_seconds=lambda:20
    monkeypatch.setitem(_native_globals,"RECOVERABLE_SUPERVISOR_LOOP_STATUSES",{"max_restarts_reached"})
    monkeypatch.setitem(_native_globals,"CONTROL_PLANE_RELOAD_STATUS","other")
    monkeypatch.setitem(_native_globals,"time",types.SimpleNamespace(monotonic=lambda:MONOTONIC,sleep=lambda _:(_ for _ in ()).throw(RuntimeError("one-cycle-complete"))))
    with pytest.raises(RuntimeError,match="one-cycle-complete"):_native_parent_loop(owner)
    status=json.loads(Path(config["status_path"]).read_text())
    assert status["status"]=="native_child_recovery" and status["unavailable_child_recovery"]["safe"] is True
    assert status["active_worker_count"]==status["worker_descendant_count"]==0 and status["active_worker_pids"]==[] and status["worker_phase"]==""
    monkeypatch.setattr(monitor,"_process_snapshot",lambda _:process_snapshot())
    native=monitor.collect_evidence(config,NOW)
    native["native_recovery"]=monitor._native_recovery_observation(config,service(),native,{},NOW,MONOTONIC)
    health=evaluate(config,native=native)
    assert native["native_recovery"]["tracked"] is True and native["native_recovery"]["expired"] is False
    assert health["status"]=="native_child_restarting" and health["healthy"] is False and health["restart_eligible"] is False


@pytest.mark.parametrize("phase",["agentic_maintenance_completed","agentic_maintenance_failed","native_child_recovery"])
def test_no_child_terminal_or_completed_maintenance_does_not_clear_recovery_interval(config,monkeypatch,phase):
    native=recovery_evidence(config);prior=aged_recovery(config,monkeypatch,native)
    native["native_managed_status"]=phase
    observed=monitor._native_recovery_observation(config,service(),native,prior,NOW,MONOTONIC)
    assert observed["expired"] is True and observed["since_at_epoch"]==NOW-601


def test_native_checkpoint_uses_child_events_and_refuses_resealed_parent_supervisor_stream(config):
    owner,path=checkpoint_owner(config)
    child_path=owner.config.state_dir/(owner.config.state_prefix+"_events.jsonl")
    assert owner.config.events_path != child_path
    proof=_native_emit(owner,daemon_pid=None,observed_at="now",supervisor_birth={})
    assert proof["safe"] is True
    checkpoint=json.loads(path.read_text());projection=checkpoint["projection"]
    projection["result"]["events_path"]=str(owner.config.events_path)
    _Store(path).materialize(projection,checkpoint["cursor"])
    refused=_native_emit(owner,daemon_pid=None,observed_at="now",supervisor_birth={})
    assert refused["safe"] is False and "native_checkpoint_scope_unproven" in refused["blockers"]
