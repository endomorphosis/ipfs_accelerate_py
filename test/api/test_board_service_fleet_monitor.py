"""Independent lane recovery and observational cross-lane diagnostics."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import ipfs_accelerate_py.agent_supervisor.rescue.board_service_fleet_monitor

SOURCE = Path(__file__).resolve().parents[2] / "ipfs_accelerate_py/agent_supervisor/rescue/board_service_fleet_monitor.py"
SPEC = importlib.util.spec_from_file_location("board_service_fleet_monitor_standalone", SOURCE)
fleet = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fleet)
NOW = 1_792_483_200.0


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_events(lane, events):
    path = Path(lane["monitor_config"]["event_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(event) for event in events))


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet.time, "time", lambda: NOW)
    lanes = []
    for index in range(2):
        lane_path = tmp_path / f"lane-{index}.json"
        write_json(lane_path, {"service": f"board-lane-{index}.service", "state_prefix": f"lane_{index}",
            "state_dir": str(tmp_path / f"state-{index}"), "board_namespace": "shared-board",
            "task_shard_count": 2, "task_shard_index": index})
        lanes.append({"lane_id": f"lane-{index}", "config": lane_path.name})
    path = tmp_path / "fleet.json"
    write_json(path, {"lanes": lanes, "health_path": str(tmp_path / "fleet-health.json"),
                     "error_window_seconds": 300, "repeated_error_threshold": 3, "merge_stale_seconds": 600})
    result = fleet.load_config(path)
    for index, lane in enumerate(result["lanes"]):
        write_events(lane, [{"type": "implementation_started", "timestamp": NOW - 30,
            "task_id": f"T{index}", "canonical_task_cid": f"cid-{index}", "board_namespace": "shared-board"}])
    return result


def healthy(lane, task_id=None):
    index = lane["monitor_config"]["task_shard_index"]
    return {"healthy": True, "status": "working", "action": "none",
        "service": lane["monitor_config"]["service"], "board_namespace": "shared-board",
        "service_observation": {"ActiveState": "active"},
        "evidence": {"active": True, "active_task_id": task_id or f"T{index}",
                     "completed_count": 7, "task_count": 14, "global_ready_count": 5}}


def stub_health(config, monkeypatch, summaries=None):
    values = summaries or {lane["monitor_config"]["service"]: healthy(lane) for lane in config["lanes"]}
    calls = []

    def run(per_lane, *, apply):
        calls.append((per_lane["service"], apply))
        return values[per_lane["service"]]

    monkeypatch.setattr(fleet.lane_monitor, "monitor", run)
    return calls


def test_independent_lanes_are_reported_without_summing_global_task_counts(config, monkeypatch):
    calls = stub_health(config, monkeypatch)
    result = fleet.monitor_fleet(config)
    assert result["healthy"] is True
    assert result["lane_count"] == result["healthy_lane_count"] == 2
    assert result["task_counts_aggregated"] is False
    assert "completed_count" not in result
    assert {lane["health"]["evidence"]["completed_count"] for lane in result["lanes"]} == {7}
    assert sorted(calls) == [("board-lane-0.service", False), ("board-lane-1.service", False)]
    assert json.loads(Path(config["health_path"]).read_text()) == result


def test_duplicate_working_canonical_claims_are_diagnostic_only(config, monkeypatch):
    for index, lane in enumerate(config["lanes"]):
        write_events(lane, [{"type": "implementation_started", "timestamp": NOW - 30,
            "task_id": f"alias-{index}", "canonical_task_cid": "same-cid", "board_namespace": "shared-board"}])
    values = {lane["monitor_config"]["service"]: healthy(lane, f"alias-{index}")
              for index, lane in enumerate(config["lanes"])}
    calls = stub_health(config, monkeypatch, values)
    result = fleet.monitor_fleet(config)
    assert not result["healthy"]
    assert result["duplicate_active_tasks"][0]["task_identity"] == "same-cid"
    assert result["duplicate_active_tasks"][0]["lane_ids"] == ["lane-0", "lane-1"]
    assert all(not apply for _, apply in calls)


def test_identical_display_ids_with_different_canonical_identity_are_not_duplicates(config, monkeypatch):
    for index, lane in enumerate(config["lanes"]):
        write_events(lane, [{"type": "implementation_started", "timestamp": NOW - 30,
            "task_id": "T", "canonical_task_cid": f"cid-{index}"}])
    stub_health(config, monkeypatch, {lane["monitor_config"]["service"]: healthy(lane, "T") for lane in config["lanes"]})
    result = fleet.monitor_fleet(config)
    assert result["healthy"]
    assert result["duplicate_active_tasks"] == []


def test_stale_historical_active_state_is_not_a_duplicate_claim(config, monkeypatch):
    for lane in config["lanes"]:
        write_events(lane, [{"type": "implementation_started", "timestamp": NOW - 500,
            "task_id": "T", "canonical_task_cid": "same-cid"}])
    values = {lane["monitor_config"]["service"]: healthy(lane, "T") for lane in config["lanes"]}
    values["board-lane-1.service"].update(healthy=False, status="stale_worker")
    stub_health(config, monkeypatch, values)
    result = fleet.monitor_fleet(config)
    assert not result["healthy"]
    assert result["duplicate_active_tasks"] == []


def test_repeated_structured_failure_fingerprints_ignore_old_and_unrelated_events(config, monkeypatch):
    lane = config["lanes"][0]
    events = [{"type": "implementation_finished", "timestamp": NOW - offset,
               "task_id": "T0", "returncode": 1, "reason_code": "validation_failed"} for offset in (60, 40, 20)]
    events += [{"type": "implementation_finished", "timestamp": NOW - 900, "task_id": "T0", "returncode": 1},
               {"type": "worker_output", "timestamp": NOW - 1, "text": "Error: example discussed by model"}]
    write_events(lane, events)
    stub_health(config, monkeypatch)
    result = fleet.monitor_fleet(config)
    assert not result["healthy"]
    details = result["lanes"][0]["diagnostics"]
    assert details["recent_failure_count"] == 3
    assert details["repeated_failures"] == [{"event_type": "implementation_finished", "task_id": "T0",
                                               "reason": "validation_failed", "count": 3}]


def test_stale_shared_merge_receipts_are_observed_without_opening_database(config, monkeypatch, tmp_path):
    queue = tmp_path / "queue"
    config["merge_queue_dir"] = str(queue)
    database = queue / "merge_queue.duckdb"
    database.parent.mkdir()
    database.write_text("must never be opened or modified")
    write_json(queue / "processing/request-a.json", {"request_id": "request-a", "task_id": "T0",
               "lane_id": "lane-0", "enqueued_at": NOW - 900, "claimed_at": NOW - 700})
    write_json(queue / "pending/request-b.json", {"request_id": "request-b", "task_id": "T1",
               "lane_id": "lane-1", "enqueued_at": NOW - 900, "retry_not_before": NOW + 100})
    stub_health(config, monkeypatch)
    result = fleet.monitor_fleet(config)
    assert not result["healthy"]
    assert result["merge_queue"]["database_opened"] is False
    assert [item["request_id"] for item in result["merge_queue"]["stale_requests"]] == ["request-a"]
    assert database.read_text() == "must never be opened or modified"
    assert (queue / "processing/request-a.json").exists()


def test_terminal_receipt_supersedes_transient_old_stage_file(config, monkeypatch, tmp_path):
    queue = tmp_path / "queue"
    config["merge_queue_dir"] = str(queue)
    write_json(queue / "pending/request-a.json", {"request_id": "request-a", "enqueued_at": NOW - 900})
    write_json(queue / "completed/request-a.json", {"request_id": "request-a", "finished_at": NOW - 1})
    stub_health(config, monkeypatch)
    result = fleet.monitor_fleet(config)
    assert result["healthy"]
    assert result["merge_queue"]["stale_requests"] == []
    assert result["merge_queue"]["counts"]["completed"] == 1


def test_failed_merge_stage_is_reported_without_repairing_or_deleting_receipt(config, monkeypatch, tmp_path):
    queue = tmp_path / "queue"
    config["merge_queue_dir"] = str(queue)
    path = queue / "failed/request-a.json"
    write_json(path, {"request_id": "request-a", "task_id": "T0", "lane_id": "lane-0"})
    stub_health(config, monkeypatch)
    result = fleet.monitor_fleet(config)
    assert not result["healthy"]
    assert any(issue["code"] == "merge_failed" for issue in result["issues"])
    assert path.exists()


def test_recent_integration_failure_stays_visible_during_retry_without_restarting_workers(config, monkeypatch, tmp_path):
    queue = tmp_path / "queue"
    config["merge_queue_dir"] = str(queue)
    path = queue / "pending/request-a.json"
    write_json(path, {"request_id": "request-a", "task_id": "T0", "lane_id": "lane-0",
        "enqueued_at": NOW - 5, "failure_count": 1, "failure_reason": "merge_callback_exception",
        "retry_not_before": NOW + 300})
    stub_health(config, monkeypatch)
    result = fleet.monitor_fleet(config, apply=True)
    assert not result["healthy"]
    assert all(lane["health"]["healthy"] for lane in result["lanes"])
    assert result["issues"] == [{"code": "merge_retrying_failure", "request_id": "request-a",
        "task_id": "T0", "lane_id": "lane-0", "failure_count": 1, "reason": "merge_callback_exception"}]
    assert result["merge_queue"]["stale_requests"] == []
    assert path.exists()
    write_json(queue / "completed/request-a.json", {"request_id": "request-a"})
    assert fleet.monitor_fleet(config)["healthy"]


def test_lane_failure_does_not_prevent_other_lane_observation(config, monkeypatch):
    calls = []

    def observe(per_lane, *, apply):
        calls.append(per_lane["service"])
        if per_lane["service"] == "board-lane-0.service":
            raise OSError("locked by another monitor")
        return healthy(config["lanes"][1])

    monkeypatch.setattr(fleet.lane_monitor, "monitor", observe)
    result = fleet.monitor_fleet(config)
    assert sorted(calls) == ["board-lane-0.service", "board-lane-1.service"]
    assert result["healthy_lane_count"] == 1
    assert result["lanes"][0]["health"]["status"] == "monitor_error"


def test_lane_observations_run_concurrently_with_separate_scopes(config, monkeypatch):
    barrier = threading.Barrier(2, timeout=3)
    values = {lane["monitor_config"]["service"]: healthy(lane) for lane in config["lanes"]}

    def observe(per_lane, *, apply):
        barrier.wait()
        return values[per_lane["service"]]

    monkeypatch.setattr(fleet.lane_monitor, "monitor", observe)
    assert fleet.monitor_fleet(config)["healthy"]


def test_apply_restarts_only_failed_lane_exact_service(config, monkeypatch):
    monkeypatch.setattr(fleet.lane_monitor.time, "monotonic", lambda: 10_000)
    for lane in config["lanes"]:
        write_json(lane["monitor_config"]["native_state_path"], {
            "heartbeat_at": NOW - 5, "ready_count": 2, "selectable_ready_count": 1})

    def show(service):
        return {"ActiveState": "failed" if service == "board-lane-1.service" else "active",
                "MainPID": "100", "LoadState": "loaded", "ExecMainStartTimestampMonotonic": "8000000000"}

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(fleet.lane_monitor, "systemd_show", show)
    monkeypatch.setattr(fleet.lane_monitor.subprocess, "run", run)
    result = fleet.monitor_fleet(config, apply=True)
    assert calls == [["systemctl", "--user", "restart", "board-lane-1.service"]]
    assert result["lanes"][0]["health"]["action"] == "none"
    assert result["lanes"][1]["health"]["action"] == "restart_requested"


def test_duplicate_service_configuration_is_rejected_before_any_action(config, tmp_path):
    path = tmp_path / "bad-fleet.json"
    write_json(path, {"lanes": [
        {"lane_id": "first", "config": config["lanes"][0]["config_path"]},
        {"lane_id": "second", "config": config["lanes"][0]["config_path"]},
    ], "health_path": str(tmp_path / "health.json")})
    with pytest.raises(ValueError, match="distinct"):
        fleet.load_config(path)


def test_diagnostic_exception_publishes_sibling_and_recovers_next_pass(config, monkeypatch):
    calls = stub_health(config, monkeypatch)
    original = fleet.lane_diagnostics

    def broken(lane, *args):
        if lane["lane_id"] == "lane-0":
            raise RuntimeError("isolated diagnostic failure")
        return original(lane, *args)

    monkeypatch.setattr(fleet, "lane_diagnostics", broken)
    result = fleet.monitor_fleet(config, apply=True)
    assert len(result["lanes"]) == 2 and result["healthy_lane_count"] == 1
    assert not result["healthy"] and result["status"] == "attention_required"
    failed, sibling = result["lanes"]
    assert failed["health"]["status"] == "monitor_error"
    assert failed["health"]["restart_eligible"] is False
    assert failed["health"]["action"] == "none"
    assert failed["diagnostics"]["active_task_id"] == failed["diagnostics"]["active_task_cid"] == ""
    assert failed["diagnostics"]["active_task_key"] == ""
    assert "isolated diagnostic failure" in failed["diagnostics"]["diagnostics_error"]
    assert sibling["health"]["healthy"] is True
    assert json.loads(Path(config["health_path"]).read_text()) == result
    assert len(calls) == 2
    monkeypatch.setattr(fleet, "lane_diagnostics", original)
    assert fleet.monitor_fleet(config)["healthy"] is True


def test_diagnostic_error_preserves_already_requested_lane_action(config, monkeypatch):
    values = {lane["monitor_config"]["service"]: healthy(lane) for lane in config["lanes"]}
    values["board-lane-0.service"].update(action="restart_requested", healthy=False, status="service_dead")
    stub_health(config, monkeypatch, values)
    original = fleet.lane_diagnostics

    def broken(lane, *args):
        if lane["lane_id"] == "lane-0":
            raise OSError("diagnostic file unavailable")
        return original(lane, *args)

    monkeypatch.setattr(fleet, "lane_diagnostics", broken)
    result = fleet.monitor_fleet(config, apply=True)
    assert result["lanes"][0]["health"]["action"] == "restart_requested"
    assert result["lanes"][0]["health"]["restart_eligible"] is False
    assert result["lanes"][1]["health"]["action"] == "none"


def test_deep_json_event_cannot_suppress_aggregate_observation(config, monkeypatch):
    stub_health(config, monkeypatch)
    lane = config["lanes"][0]
    path = Path(lane["monitor_config"]["event_path"])
    with path.open("a") as stream:
        stream.write('\n{"nested":' + '[' * 10000 + '0' + ']' * 10000 + '}\n')
    result = fleet.monitor_fleet(config)
    assert result["healthy"] is False and result["healthy_lane_count"] == 1
    assert result["lanes"][0]["health"]["status"] == "monitor_error"
    assert "event_json_nesting_limit" in result["lanes"][0]["diagnostics"]["event_read_error"]
    assert result["lanes"][1]["health"]["healthy"] is True
    assert json.loads(Path(config["health_path"]).read_text()) == result


def test_deep_merge_receipt_is_reported_without_hiding_lanes(config, monkeypatch, tmp_path):
    stub_health(config, monkeypatch)
    queue = tmp_path / "queue"
    config["merge_queue_dir"] = str(queue)
    path = queue / "pending/deep.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"nested":' + '[' * 10000 + '0' + ']' * 10000 + '}')
    result = fleet.monitor_fleet(config)
    assert result["healthy"] is False and result["healthy_lane_count"] == 2
    assert result["merge_queue"]["issues"] == [
        {"code": "merge_receipt_unreadable", "stage": "pending", "path": str(path)}]
    assert json.loads(Path(config["health_path"]).read_text()) == result
    assert path.exists()


_IO_PROBE = '''
import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location("isolated_io_probe",sys.argv[1])
fleet=importlib.util.module_from_spec(spec);spec.loader.exec_module(fleet)
lane=fleet.lane_monitor
path,kind=sys.argv[2:4]
try:
    if kind=="json":
        value,error=lane._read_json(path)
        result={"error":error,"value":value}
    elif kind=="events":
        value,error=lane._event_tail({"event_path":path,"board_namespace":"fixture"},1700000000)
        result={"error":error,"value":value}
    elif kind=="append":
        lane._append_action(path,{"fixture":True});result={"error":None}
    elif kind=="lane_lock":
        lane.monitor({"monitor_lock_path":path});result={"error":None}
    elif kind=="fleet_lock":
        fleet.monitor_fleet({"lock_path":path});result={"error":None}
    elif kind=="aggregate":
        config=json.loads(Path(path).read_text())
        lane.monitor=lambda c,apply=False:{"healthy":True,"status":"shard_idle","action":"none",
            "service":c["service"],"board_namespace":c["board_namespace"],"evidence":{}}
        value=fleet.monitor_fleet(config)
        result={"status":value["status"],"lane_count":len(value["lanes"]),
                "healthy_lane_count":value["healthy_lane_count"],"issues":value["issues"]}
    print(json.dumps(result))
except (OSError,ValueError) as exc:
    print(json.dumps({"error":str(exc)}))
'''


@pytest.mark.parametrize("kind", ["json", "events", "append", "lane_lock", "fleet_lock"])
def test_fifo_paths_reject_without_waiting_for_peer(tmp_path, kind):
    path = tmp_path / "fixture.fifo"
    os.mkfifo(path, 0o600)
    process = subprocess.run([sys.executable, "-c", _IO_PROBE, str(SOURCE), str(path), kind],
                             capture_output=True, text=True, timeout=2, check=True)
    assert json.loads(process.stdout)["error"]
    assert path.is_fifo()


@pytest.mark.parametrize("name", ["board_service_monitor.py", "board_service_fleet_monitor.py"])
def test_fifo_configuration_cli_returns_monitor_error_without_waiting(tmp_path, name):
    path = tmp_path / "config.fifo"
    os.mkfifo(path, 0o600)
    process = subprocess.run([sys.executable, str(SOURCE.with_name(name)), "--config", str(path)],
                             capture_output=True, text=True, timeout=2)
    assert process.returncode == 2
    report = json.loads(process.stdout)
    assert report["status"] == "monitor_error" and report["healthy"] is False
    assert path.is_fifo()


def test_fifo_action_path_with_reader_never_receives_monitor_bytes(tmp_path):
    path = tmp_path / "actions.fifo"
    os.mkfifo(path, 0o600)
    reader = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        process = subprocess.run([sys.executable, "-c", _IO_PROBE, str(SOURCE), str(path), "append"],
                                 capture_output=True, text=True, timeout=2, check=True)
        assert json.loads(process.stdout)["error"]
        assert os.read(reader, 4096) == b""
    finally:
        os.close(reader)


@pytest.mark.parametrize("kind", ["json", "append", "lane_lock", "fleet_lock"])
def test_symlink_io_paths_leave_target_untouched(tmp_path, kind):
    target = tmp_path / "foreign.json"
    original = b'{"foreign":true}\n'
    target.write_bytes(original)
    path = tmp_path / "alias"
    path.symlink_to(target)
    process = subprocess.run([sys.executable, "-c", _IO_PROBE, str(SOURCE), str(path), kind],
                             capture_output=True, text=True, timeout=2, check=True)
    assert json.loads(process.stdout)["error"]
    assert target.read_bytes() == original and path.is_symlink()


@pytest.mark.parametrize("kind,healthy_count", [("event", 1), ("merge", 2)])
def test_fifo_diagnostic_still_publishes_sibling_observations(config, tmp_path, kind, healthy_count):
    if kind == "event":
        path = Path(config["lanes"][0]["monitor_config"]["event_path"])
        path.unlink()
    else:
        queue = tmp_path / "queue"
        path = queue / "pending/fixture.json"
        path.parent.mkdir(parents=True)
        config["merge_queue_dir"] = str(queue)
    os.mkfifo(path, 0o600)
    config_path = tmp_path / "isolated-config.json"
    config_path.write_text(json.dumps(config))
    process = subprocess.run([sys.executable, "-c", _IO_PROBE, str(SOURCE), str(config_path), "aggregate"],
                             capture_output=True, text=True, timeout=2, check=True)
    report = json.loads(process.stdout)
    assert report["status"] == "attention_required" and report["lane_count"] == 2
    assert report["healthy_lane_count"] == healthy_count
    assert json.loads(Path(config["health_path"]).read_text())["status"] == "attention_required"
    assert path.is_fifo()


def test_fifo_fleet_observation_log_still_publishes_fresh_health(config, tmp_path):
    path = Path(config["actions_path"])
    os.mkfifo(path, 0o600)
    write_json(config["health_path"], {"checked_at_epoch": 1, "healthy": True})
    config_path = tmp_path / "isolated-config.json"
    config_path.write_text(json.dumps(config))
    process = subprocess.run([sys.executable, "-c", _IO_PROBE, str(SOURCE), str(config_path), "aggregate"],
                             capture_output=True, text=True, timeout=2, check=True)
    report = json.loads(process.stdout)
    assert report["status"] == "attention_required" and report["healthy_lane_count"] == 2
    assert report["issues"][-1]["code"] == "fleet_observation_log_unavailable"
    saved = json.loads(Path(config["health_path"]).read_text())
    assert saved["checked_at_epoch"] > 1 and saved["healthy"] is False
    assert path.is_fifo()


# These inert descriptions exercise the monitor's file/authority boundary.
# The separate dataset-owner cohort qualifies its actual diagnostic content.
import hashlib


def _requirement_bundle():
    diagnostics = []
    for task_id, dimension in (("LIRAE-WIDTH-08", 8), ("LIRAE-WIDTH-4096", 4096)):
        diagnostics.append({
            "schema": "legal-ir-width-replay-prerequisites/v1", "task_id": task_id,
            "dimension": dimension, "reviewed_local_dimensions": [384, 768],
            "diagnostic_scope": "declared_requirements_only", "input_availability": "not_inspected",
            "input_files_checked": False, "executable_presence_checked": False,
            "reviewed_route_available": False, "executable_presence_sufficient": False,
            "source_contract": {}, "required_inputs": {}, "task_constraints": {}, "next_actions": [],
            "completion_allowed": False, "resource_admission_granted": False,
            "encoder_executed": False, "training_executed": False,
            "admitted": False, "formalized": False, "proof_authority": False,
        })
    return {"schema": "legal-ir-width-replay-prerequisite-bundle/v1",
            "diagnostic_scope": "declared_requirements_only", "diagnostics": diagnostics,
            "completion_allowed": False}


def _requirement_file_pin(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _requirement_attachment(tmp_path, *, report_bytes=None):
    report = tmp_path / "prerequisites.json"
    owner = tmp_path / "declarative-owner.py"
    # Reading this pin must never import or execute the described owner.
    owner.write_text('raise AssertionError("the monitor executed the declared owner")\n')
    report.write_bytes(report_bytes if report_bytes is not None else json.dumps(_requirement_bundle()).encode())
    return {**_requirement_file_pin(report), "owner": _requirement_file_pin(owner)}


def test_pinned_prerequisites_preserve_lane_health_and_actions_without_owner_execution(config, monkeypatch, tmp_path):
    calls = stub_health(config, monkeypatch)
    baseline = fleet.monitor_fleet(config, apply=True)
    assert "prerequisite_diagnostics" not in baseline
    reference = _requirement_attachment(tmp_path)
    config["prerequisite_diagnostics"] = reference
    before = {reference["path"]: Path(reference["path"]).read_bytes(),
              reference["owner"]["path"]: Path(reference["owner"]["path"]).read_bytes()}

    def forbid_execution(*args, **kwargs):
        raise AssertionError("requirement attachment executed an owner or command")

    monkeypatch.setattr(fleet.importlib.util, "spec_from_file_location", forbid_execution)
    monkeypatch.setattr(fleet.lane_monitor.subprocess, "run", forbid_execution)
    result = fleet.monitor_fleet(config, apply=True)
    attachment = result.pop("prerequisite_diagnostics")
    assert result == baseline
    assert attachment["available"] is True
    assert attachment["authority"] == "declared_requirements_only"
    assert attachment["completion_authority"] is attachment["action_authority"] is False
    assert attachment["report"] == _requirement_bundle()
    assert attachment["consistency_scope"] == "sequential_pinned_file_observations"
    assert attachment["payload_validation_scope"] == "pinned_description_envelope_and_non_authority_flags_only"
    assert attachment["report_pin"] == {key: reference[key] for key in ("path", "bytes", "sha256")}
    assert attachment["owner_pin"] == reference["owner"]
    assert len(calls) == 4 and all(apply is True for _, apply in calls)
    assert all(Path(path).read_bytes() == original for path, original in before.items())


@pytest.mark.parametrize("change", [
    "report_hash", "owner_hash", "report_missing", "owner_missing",
    "report_size_boolean", "owner_size_boolean", "report_oversize", "owner_oversize",
    "report_extra_pin", "owner_extra_pin",
])
def test_bad_prerequisite_pins_remain_diagnostic_and_preserve_all_lane_observations(config, monkeypatch, tmp_path, change):
    reference = _requirement_attachment(tmp_path)
    target = reference["owner"] if change.startswith("owner") else reference
    if change.endswith("_hash"):
        target["sha256"] = "0" * 64
    elif change.endswith("_missing"):
        Path(target["path"]).unlink()
    elif change.endswith("_size_boolean"):
        target["bytes"] = True
    elif change.endswith("_oversize"):
        maximum = 256 * 1024 if change.startswith("owner") else 64 * 1024
        Path(target["path"]).write_bytes(b"x" * (maximum + 1))
        target.update(_requirement_file_pin(Path(target["path"])))
    else:
        target["unreviewed_extra"] = True
    config["prerequisite_diagnostics"] = reference
    calls = stub_health(config, monkeypatch)
    result = fleet.monitor_fleet(config, apply=True)
    assert result["healthy"] is False and result["status"] == "attention_required"
    assert result["healthy_lane_count"] == result["lane_count"] == 2
    assert all(lane["health"]["action"] == "none" for lane in result["lanes"])
    assert len(calls) == 2 and all(apply for _, apply in calls)
    attachment = result["prerequisite_diagnostics"]
    assert attachment["available"] is False and attachment["error"]
    assert attachment["completion_authority"] is attachment["action_authority"] is False
    assert result["issues"] == [{"code": "prerequisite_diagnostics_unavailable", "error": attachment["error"]}]
    assert json.loads(Path(config["health_path"]).read_text()) == result


@pytest.mark.parametrize("flag", [
    "input_files_checked", "executable_presence_checked", "reviewed_route_available",
    "executable_presence_sufficient", "completion_allowed", "resource_admission_granted",
    "encoder_executed", "training_executed", "admitted", "formalized", "proof_authority",
])
def test_prerequisite_attachment_rejects_asserted_input_or_execution_authority(tmp_path, flag):
    report = _requirement_bundle()
    report["diagnostics"][0][flag] = True
    reference = _requirement_attachment(tmp_path, report_bytes=json.dumps(report).encode())
    result = fleet.prerequisite_diagnostics({"prerequisite_diagnostics": reference})
    assert result["available"] is False and result["error"]


@pytest.mark.parametrize("change", [
    "bundle_completion", "bundle_extra", "bundle_scope", "item_scope", "item_availability",
    "duplicate_task", "missing_false_flag", "numeric_false_flag", "nan", "overflow_float",
    "duplicate_json_key", "duplicate_nested_json_key", "item_extra", "wrong_dimension",
    "boolean_dimension", "wrong_order", "wrong_reviewed_dimensions", "boolean_reviewed_dimension",
])
def test_prerequisite_attachment_rejects_malformed_or_nonfinite_descriptions(tmp_path, change):
    report = _requirement_bundle()
    item = report["diagnostics"][0]
    if change == "bundle_completion":
        report["completion_allowed"] = True
    elif change == "bundle_extra":
        report["action"] = "restart"
    elif change == "bundle_scope":
        report["diagnostic_scope"] = "inputs_verified"
    elif change == "item_scope":
        item["diagnostic_scope"] = "inputs_verified"
    elif change == "item_availability":
        item["input_availability"] = "available"
    elif change == "duplicate_task":
        report["diagnostics"][1]["task_id"] = item["task_id"]
    elif change == "missing_false_flag":
        del item["encoder_executed"]
    elif change == "numeric_false_flag":
        item["encoder_executed"] = 0
    elif change == "nan":
        item["source_contract"]["untrusted_measurement"] = float("nan")
    elif change == "overflow_float":
        item["source_contract"]["untrusted_measurement"] = "EXPONENT_PLACEHOLDER"
    elif change == "item_extra":
        item["unreviewed_extra"] = False
    elif change == "wrong_dimension":
        item["dimension"] = 384
    elif change == "boolean_dimension":
        item["dimension"] = True
    elif change == "wrong_order":
        report["diagnostics"].reverse()
    elif change == "wrong_reviewed_dimensions":
        item["reviewed_local_dimensions"] = [8, 4096]
    elif change == "boolean_reviewed_dimension":
        item["reviewed_local_dimensions"] = [True, 768]
    payload = json.dumps(report).encode()
    if change == "overflow_float":
        payload = payload.replace(b'"EXPONENT_PLACEHOLDER"', b'1e999')
    elif change == "duplicate_json_key":
        payload = b'{"schema":"duplicate",' + payload[1:]
    elif change == "duplicate_nested_json_key":
        payload = payload.replace(b'"input_availability": "not_inspected"',
            b'"input_availability": "not_inspected", "input_availability": "not_inspected"', 1)
    reference = _requirement_attachment(tmp_path, report_bytes=payload)
    result = fleet.prerequisite_diagnostics({"prerequisite_diagnostics": reference})
    assert result["available"] is False and result["error"]


@pytest.mark.parametrize("which,change", [
    ("owner", "content"), ("report", "content"), ("owner", "replace"), ("report", "replace"),
])
def test_prerequisite_file_change_or_replacement_during_read_is_rejected(tmp_path, monkeypatch, which, change):
    reference = _requirement_attachment(tmp_path)
    target = Path(reference["owner"]["path"] if which == "owner" else reference["path"])
    original = target.read_bytes()
    read = fleet.os.read
    changed = []

    def mutate(descriptor, amount):
        piece = read(descriptor, amount)
        if piece and not changed and Path(os.readlink(f"/proc/self/fd/{descriptor}")) == target:
            changed.append(True)
            if change == "content":
                target.write_bytes(b"changed source while the descriptor was open")
            else:
                replacement = target.with_name(target.name + ".replacement")
                replacement.write_bytes(original)
                os.replace(replacement, target)
        return piece

    monkeypatch.setattr(fleet.os, "read", mutate)
    result = fleet.prerequisite_diagnostics({"prerequisite_diagnostics": reference})
    assert changed and result["available"] is False and result["error"]


@pytest.mark.parametrize("which", ["owner", "report", "entire_reference"])
def test_prerequisite_original_references_cannot_change_between_opening_and_closing(tmp_path, monkeypatch, which):
    reference = _requirement_attachment(tmp_path)
    config = {"prerequisite_diagnostics": reference}
    original_pin = fleet._prerequisite_pin
    calls = []

    def mutate(pin, maximum):
        result = original_pin(pin, maximum)
        calls.append(str(pin["path"]))
        if len(calls) == 1:
            if which == "owner":
                replacement = tmp_path / "different-owner.py"
                replacement.write_text("raise AssertionError('different owner must not run')\n")
                reference["owner"] = _requirement_file_pin(replacement)
            elif which == "report":
                replacement = tmp_path / "different-report.json"
                replacement.write_bytes(json.dumps(_requirement_bundle()).encode())
                reference.update(_requirement_file_pin(replacement))
            else:
                config["prerequisite_diagnostics"] = {**reference, "sha256": "0" * 64}
        return result

    monkeypatch.setattr(fleet, "_prerequisite_pin", mutate)
    result = fleet.prerequisite_diagnostics(config)
    assert calls and result["available"] is False and result["error"]


@pytest.mark.parametrize("which", ["owner", "report"])
def test_prerequisite_symlink_never_follows_foreign_file(tmp_path, which):
    originals = tmp_path / "originals"
    originals.mkdir()
    reference = _requirement_attachment(originals)
    target = reference["owner"] if which == "owner" else reference
    foreign = Path(target["path"])
    original = foreign.read_bytes()
    path = tmp_path / foreign.name
    special_pin = {key: target[key] for key in ("path", "bytes", "sha256")}
    special_pin["path"] = str(path)
    if which == "owner":
        reference["owner"] = special_pin
    else:
        reference.update(special_pin)
    path.symlink_to(foreign)
    try:
        result = fleet.prerequisite_diagnostics({"prerequisite_diagnostics": reference})
        assert result["available"] is False and result["error"]
        assert foreign.read_bytes() == original and path.is_symlink()
    finally:
        if path.is_symlink():
            path.unlink()


@pytest.mark.parametrize("which", ["owner", "report"])
def test_prerequisite_fifo_rejects_without_waiting_for_a_peer(tmp_path, monkeypatch, which):
    originals = tmp_path / "originals"
    originals.mkdir()
    reference = _requirement_attachment(originals)
    target = reference["owner"] if which == "owner" else reference
    path = tmp_path / Path(target["path"]).name
    special_pin = {key: target[key] for key in ("path", "bytes", "sha256")}
    special_pin["path"] = str(path)
    if which == "owner":
        reference["owner"] = special_pin
    else:
        reference.update(special_pin)
    os.mkfifo(path, 0o600)
    original_open = fleet.os.open
    observed = []

    def nonblocking_open(name, flags, *args, **kwargs):
        if str(name) == str(path):
            # Assert before opening, so a regression cannot block this test.
            assert flags & os.O_NONBLOCK and flags & os.O_NOFOLLOW
            observed.append(str(name))
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(fleet.os, "open", nonblocking_open)
    try:
        result = fleet.prerequisite_diagnostics({"prerequisite_diagnostics": reference})
        assert observed and result["available"] is False and result["error"]
        assert path.is_fifo()
    finally:
        if path.is_fifo():
            path.unlink()


def test_prerequisite_relative_config_paths_preserve_pins_and_resolve_without_execution(config, tmp_path, monkeypatch):
    reference = _requirement_attachment(tmp_path)
    path = tmp_path / "fleet.json"
    original = json.loads(path.read_text())
    original["prerequisite_diagnostics"] = {
        **reference, "path": Path(reference["path"]).name,
        "owner": {**reference["owner"], "path": Path(reference["owner"]["path"]).name},
    }
    write_json(path, original)
    loaded = fleet.load_config(path)
    assert loaded["prerequisite_diagnostics"] == reference
    calls = stub_health(loaded, monkeypatch)
    result = fleet.monitor_fleet(loaded)
    assert result["healthy"] is True and len(calls) == 2
    assert result["prerequisite_diagnostics"]["available"] is True


def test_prerequisite_returned_pins_do_not_alias_configuration(tmp_path):
    reference = _requirement_attachment(tmp_path)
    expected = json.dumps(reference, sort_keys=True)
    result = fleet.prerequisite_diagnostics({"prerequisite_diagnostics": reference})
    assert result["available"] is True
    result["owner_pin"]["sha256"] = "caller-change"
    result["report_pin"]["path"] = "/caller-change"
    assert json.dumps(reference, sort_keys=True) == expected


# The optional watchdog attachment is diagnostic only. These fixtures never
# query a real unit, process, provider, or native task store.
_WATCHDOG_MONOTONIC = 10_000.0
_WATCHDOG_BOOT_ID = "b82a31a5-7008-41e2-b518-d01e719bfb42"


@pytest.fixture
def watchdog_liveness(config, tmp_path, monkeypatch):
    path = tmp_path / "dedicated-watchdog-status.json"
    binding = {
        "schema": "agent-supervisor/watchdog-liveness-binding@1",
        "service": "fixture-dedicated-watchdog.service",
        "status_path": str(path),
        "deployment_id": "fixture-deployment-v1",
        "source_sha256": "a" * 64,
        "stale_after_seconds": 60,
        "startup_grace_seconds": 30,
    }
    process = {"pid": 4242, "birth_ticks": 12345, "boot_id": _WATCHDOG_BOOT_ID,
               "children": []}
    report = {
        "schema": "agent-supervisor/fleet-watchdog@1",
        "observed_at": NOW - 10,
        "apply": True,
        "boards": {"fixture-engineering": {"health": "healthy"}},
        "liveness": {
            "schema": "agent-supervisor/watchdog-liveness@1",
            "available": True,
            "process": {key: process[key] for key in ("pid", "birth_ticks", "boot_id")},
            "profile": {key: binding[key] for key in ("deployment_id", "source_sha256")},
            "source_binding": "declared_profile_only",
            "action_authority": False,
        },
    }
    unit = {"LoadState": "loaded", "ActiveState": "active", "SubState": "running",
            "MainPID": "4242", "ExecMainStartTimestampMonotonic": "9000000000"}
    systemd_calls, process_calls = [], []

    def systemd_show(service):
        systemd_calls.append(service)
        return dict(unit)

    def process_snapshot(pid):
        process_calls.append(pid)
        return dict(process)

    monkeypatch.setattr(fleet.time, "monotonic", lambda: _WATCHDOG_MONOTONIC)
    monkeypatch.setattr(fleet.lane_monitor, "systemd_show", systemd_show)
    monkeypatch.setattr(fleet.lane_monitor, "_process_snapshot", process_snapshot)
    config["watchdog_liveness"] = binding
    write_json(path, report)
    return {"config": config, "binding": binding, "path": path, "report": report,
            "unit": unit, "process": process, "systemd_calls": systemd_calls,
            "process_calls": process_calls}


def _assert_watchdog_observation_only(result):
    assert result["action_authority"] is False
    assert result["restart_eligible"] is False


def _watchdog_observe(fixture, now=NOW):
    return fleet.watchdog_liveness_diagnostics(fixture["config"], now)


def test_watchdog_liveness_absent_preserves_legacy_without_extra_io(config, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("an absent optional binding must perform no watchdog I/O")

    monkeypatch.setattr(fleet.lane_monitor, "systemd_show", forbidden)
    monkeypatch.setattr(fleet.lane_monitor, "_process_snapshot", forbidden)
    monkeypatch.setattr(fleet, "_read_watchdog_liveness_report", forbidden)
    calls = stub_health(config, monkeypatch)
    result = fleet.monitor_fleet(config, apply=True)
    assert "watchdog_liveness" not in result
    assert result["healthy"] is True and result["issues"] == []
    assert sorted(calls) == [("board-lane-0.service", True), ("board-lane-1.service", True)]


def test_watchdog_liveness_healthy_report_is_bound_twice_and_declared_only(watchdog_liveness):
    result = _watchdog_observe(watchdog_liveness)
    assert result["configured"] is True and result["available"] is True
    assert result["status"] == "healthy" and result["attention_required"] is False
    assert result["source_binding"] == "declared_profile_only"
    _assert_watchdog_observation_only(result)
    assert watchdog_liveness["systemd_calls"] == [watchdog_liveness["binding"]["service"]] * 2
    assert watchdog_liveness["process_calls"] == [4242, 4242]


@pytest.mark.parametrize("age,status", [(60, "healthy"), (60.001, "stale")])
def test_watchdog_liveness_age_boundary_does_not_grant_restart(watchdog_liveness, age, status):
    watchdog_liveness["report"]["observed_at"] = NOW - age
    write_json(watchdog_liveness["path"], watchdog_liveness["report"])
    result = _watchdog_observe(watchdog_liveness)
    assert result["available"] is True and result["status"] == status
    assert result["attention_required"] is (status == "stale")
    _assert_watchdog_observation_only(result)


@pytest.mark.parametrize("apply", [False, True])
def test_stale_watchdog_attachment_preserves_healthy_idle_lane_authority(watchdog_liveness, monkeypatch, apply):
    config = watchdog_liveness["config"]
    watchdog_liveness["report"]["observed_at"] = NOW - 100
    write_json(watchdog_liveness["path"], watchdog_liveness["report"])
    values = {}
    for lane in config["lanes"]:
        value = healthy(lane)
        value.update(status="shard_idle")
        value["evidence"].update(active=False, active_task_id="", global_ready_count=0)
        values[lane["monitor_config"]["service"]] = value
    calls = stub_health(config, monkeypatch, values)
    result = fleet.monitor_fleet(config, apply=apply)
    assert result["watchdog_liveness"]["status"] == "stale"
    _assert_watchdog_observation_only(result["watchdog_liveness"])
    assert result["healthy"] is True and result["status"] == "healthy"
    assert result["issues"] == [] and result["healthy_lane_count"] == 2
    assert result["completion_authority"] is False and result["task_counts_aggregated"] is False
    assert all(lane["health"]["action"] == "none" for lane in result["lanes"])
    assert sorted(calls) == [("board-lane-0.service", apply), ("board-lane-1.service", apply)]


@pytest.mark.parametrize("field,value", [
    ("pid", 4243), ("birth_ticks", 12346),
    ("boot_id", "c82a31a5-7008-41e2-b518-d01e719bfb42"),
])
def test_watchdog_liveness_rejects_report_from_another_process_birth(watchdog_liveness, field, value):
    watchdog_liveness["report"]["liveness"]["process"][field] = value
    write_json(watchdog_liveness["path"], watchdog_liveness["report"])
    result = _watchdog_observe(watchdog_liveness)
    assert result["status"] == "unavailable" and result["available"] is False
    assert result["reason_codes"]
    _assert_watchdog_observation_only(result)


@pytest.mark.parametrize("field,value", [("deployment_id", "previous-deployment"), ("source_sha256", "b" * 64)])
def test_watchdog_liveness_rejects_different_declared_profile(watchdog_liveness, field, value):
    watchdog_liveness["report"]["liveness"]["profile"][field] = value
    write_json(watchdog_liveness["path"], watchdog_liveness["report"])
    result = _watchdog_observe(watchdog_liveness)
    assert result["available"] is False and result["status"] == "unavailable"
    _assert_watchdog_observation_only(result)


@pytest.mark.parametrize("stamp", [NOW + 0.001, NOW - 1001, True, float("nan"), float("inf"), "recent"])
def test_watchdog_liveness_rejects_future_prestart_and_invalid_report_clocks(watchdog_liveness, stamp):
    watchdog_liveness["report"]["observed_at"] = stamp
    write_json(watchdog_liveness["path"], watchdog_liveness["report"])
    result = _watchdog_observe(watchdog_liveness)
    assert result["available"] is False and result["status"] == "unavailable"
    _assert_watchdog_observation_only(result)


def test_legacy_watchdog_report_cannot_mint_process_or_source_binding(watchdog_liveness):
    del watchdog_liveness["report"]["liveness"]
    write_json(watchdog_liveness["path"], watchdog_liveness["report"])
    result = _watchdog_observe(watchdog_liveness)
    assert result["available"] is False and result["status"] == "unavailable"
    _assert_watchdog_observation_only(result)


@pytest.mark.parametrize("which", ["unit_pid", "unit_birth", "process_birth", "process_boot"])
def test_watchdog_liveness_rejects_identity_change_during_observation(watchdog_liveness, monkeypatch, which):
    counts = {"unit": 0, "process": 0}

    def show(service):
        counts["unit"] += 1
        value = dict(watchdog_liveness["unit"])
        if counts["unit"] > 1 and which.startswith("unit_"):
            value["MainPID" if which == "unit_pid" else "ExecMainStartTimestampMonotonic"] = "4243" if which == "unit_pid" else "9000000001"
        return value

    def snapshot(pid):
        counts["process"] += 1
        value = dict(watchdog_liveness["process"])
        if counts["process"] > 1 and which.startswith("process_"):
            value["birth_ticks" if which == "process_birth" else "boot_id"] = 12346 if which == "process_birth" else "other-boot"
        return value

    monkeypatch.setattr(fleet.lane_monitor, "systemd_show", show)
    monkeypatch.setattr(fleet.lane_monitor, "_process_snapshot", snapshot)
    result = _watchdog_observe(watchdog_liveness)
    assert result["available"] is False and result["status"] == "unavailable"
    _assert_watchdog_observation_only(result)


def test_watchdog_startup_grace_uses_verified_monotonic_service_birth(watchdog_liveness):
    watchdog_liveness["path"].unlink()
    watchdog_liveness["unit"]["ExecMainStartTimestampMonotonic"] = "9990000000"
    result = _watchdog_observe(watchdog_liveness)
    assert result["status"] == "starting" and result["attention_required"] is False
    _assert_watchdog_observation_only(result)
    watchdog_liveness["unit"]["ExecMainStartTimestampMonotonic"] = "9970000000"
    result = _watchdog_observe(watchdog_liveness)
    assert result["status"] == "unavailable" and result["available"] is False
    _assert_watchdog_observation_only(result)


@pytest.mark.parametrize("held", [False, True])
def test_inactive_or_held_watchdog_never_becomes_restart_eligible(watchdog_liveness, held):
    if held:
        watchdog_liveness["binding"]["operator_hold"] = True
    else:
        watchdog_liveness["unit"].update(ActiveState="inactive", SubState="dead", MainPID="0", ExecMainStartTimestampMonotonic="0")
    result = _watchdog_observe(watchdog_liveness)
    assert result["status"] == ("held" if held else "inactive")
    assert result["attention_required"] is (not held)
    assert (result["incident_id"] is None) is held
    _assert_watchdog_observation_only(result)


@pytest.mark.parametrize("field,value", [
    ("service", "fixture.service;start-other.service"), ("deployment_id", ""),
    ("deployment_id", 17),
    ("source_sha256", "A" * 64), ("stale_after_seconds", True),
    ("stale_after_seconds", float("inf")), ("startup_grace_seconds", -1),
    ("operator_hold", "false"), ("status_path", 17),
])
def test_malformed_watchdog_binding_is_unavailable_without_actions(watchdog_liveness, field, value):
    watchdog_liveness["binding"][field] = value
    result = _watchdog_observe(watchdog_liveness)
    assert result["status"] == "unavailable" and result["available"] is False
    _assert_watchdog_observation_only(result)


@pytest.mark.parametrize("kind", ["missing", "malformed", "symlink", "fifo"])
def test_watchdog_report_io_failure_is_bounded_and_does_not_hide_lanes(watchdog_liveness, monkeypatch, tmp_path, kind):
    path = watchdog_liveness["path"]
    foreign = tmp_path / "foreign-watchdog.json"
    original = path.read_bytes()
    special = kind in {"symlink", "fifo"}
    if special:
        # Keep ordinary fixture evidence. This leaf exists only as its declared
        # transient kind, then the owning test removes that special fixture.
        path = tmp_path / "nonregular-watchdog-status.json"
        watchdog_liveness["binding"]["status_path"] = str(path)
        if kind == "symlink":
            foreign.write_bytes(original)
            path.symlink_to(foreign)
        else:
            os.mkfifo(path, 0o600)
            real_open = fleet.os.open

            def nonblocking_open(name, flags, *args, **kwargs):
                if str(name) == str(path):
                    assert flags & os.O_NONBLOCK and flags & os.O_NOFOLLOW
                return real_open(name, flags, *args, **kwargs)

            monkeypatch.setattr(fleet.os, "open", nonblocking_open)
    elif kind == "malformed":
        path.write_text('{"schema":')
    else:
        path.unlink()
    try:
        stub_health(watchdog_liveness["config"], monkeypatch)
        result = fleet.monitor_fleet(watchdog_liveness["config"], apply=True)
        assert result["watchdog_liveness"]["status"] == "unavailable"
        _assert_watchdog_observation_only(result["watchdog_liveness"])
        assert result["healthy"] is True and result["issues"] == []
        assert result["healthy_lane_count"] == 2
        assert json.loads(Path(watchdog_liveness["config"]["health_path"]).read_text())["healthy"] is True
        if kind == "symlink":
            assert foreign.read_bytes() == original and path.is_symlink()
        elif kind == "fifo":
            assert path.is_fifo()
    finally:
        if special:
            path.unlink()


def test_watchdog_incident_is_stable_across_age_and_deduplicated_until_transition(watchdog_liveness, monkeypatch):
    config = watchdog_liveness["config"]
    stub_health(config, monkeypatch)
    watchdog_liveness["report"]["observed_at"] = NOW - 100
    write_json(watchdog_liveness["path"], watchdog_liveness["report"])
    first = fleet.monitor_fleet(config, apply=True)["watchdog_liveness"]
    monkeypatch.setattr(fleet.time, "time", lambda: NOW + 10)
    monkeypatch.setattr(fleet.time, "monotonic", lambda: _WATCHDOG_MONOTONIC + 10)
    second = fleet.monitor_fleet(config, apply=True)["watchdog_liveness"]
    assert first["status"] == second["status"] == "stale"
    assert first["incident_id"] and first["incident_id"] == second["incident_id"]
    actions = [json.loads(line) for line in Path(config["actions_path"]).read_text().splitlines()]
    assert len([event for event in actions if event["action"] == "fleet_observed"]) == 2
    assert len([event for event in actions if event["action"] != "fleet_observed"]) == 1
    watchdog_liveness["report"]["observed_at"] = NOW + 10
    write_json(watchdog_liveness["path"], watchdog_liveness["report"])
    assert fleet.monitor_fleet(config, apply=True)["watchdog_liveness"]["status"] == "healthy"
    actions = [json.loads(line) for line in Path(config["actions_path"]).read_text().splitlines()]
    assert len([event for event in actions if event["action"] != "fleet_observed"]) == 2


def test_watchdog_incident_identity_separates_dedicated_fleets(watchdog_liveness):
    watchdog_liveness["report"]["observed_at"] = NOW - 100
    write_json(watchdog_liveness["path"], watchdog_liveness["report"])
    first = _watchdog_observe(watchdog_liveness)
    watchdog_liveness["binding"]["service"] = "another-dedicated-watchdog.service"
    second = _watchdog_observe(watchdog_liveness)
    assert first["status"] == second["status"] == "stale"
    assert first["incident_id"] != second["incident_id"]
    _assert_watchdog_observation_only(first)
    _assert_watchdog_observation_only(second)


@pytest.mark.parametrize("case", ["replaced", "closing_read_error"])
def test_watchdog_liveness_rejects_report_change_between_opening_and_closing_reads(watchdog_liveness, monkeypatch, case):
    original_read = fleet.lane_monitor._read_json
    calls = []

    def read(path):
        value, error = original_read(path)
        if str(path) != str(watchdog_liveness["path"]):
            return value, error
        calls.append(str(path))
        if len(calls) == 1 and case == "replaced":
            replacement = watchdog_liveness["path"].with_name("replacement-status.json")
            changed = json.loads(json.dumps(watchdog_liveness["report"]))
            changed["observed_at"] = NOW - 5
            write_json(replacement, changed)
            replacement.replace(watchdog_liveness["path"])
        elif len(calls) == 2 and case == "closing_read_error":
            return {}, "fixture closing report unavailable"
        return value, error

    monkeypatch.setattr(fleet, "_read_watchdog_liveness_report", read)
    result = _watchdog_observe(watchdog_liveness)
    assert len(calls) == 2
    assert result["available"] is False and result["status"] == "unavailable"
    if case == "replaced":
        assert "report_changed" in result["reason_codes"]
    _assert_watchdog_observation_only(result)


@pytest.mark.parametrize("field,value", [
    ("schema", "agent-supervisor/watchdog-liveness@999"),
    ("available", False), ("source_binding", "verified_imported_source"),
    ("action_authority", True),
])
def test_watchdog_liveness_rejects_unavailable_or_authoritative_envelopes(watchdog_liveness, field, value):
    watchdog_liveness["report"]["liveness"][field] = value
    write_json(watchdog_liveness["path"], watchdog_liveness["report"])
    result = _watchdog_observe(watchdog_liveness)
    assert result["available"] is False and result["status"] == "unavailable"
    _assert_watchdog_observation_only(result)


def test_watchdog_incident_identity_tracks_kernel_boot_with_unchanged_unit_generation(watchdog_liveness):
    watchdog_liveness["report"]["observed_at"] = NOW - 100
    write_json(watchdog_liveness["path"], watchdog_liveness["report"])
    original_unit = dict(watchdog_liveness["unit"])
    original_profile = dict(watchdog_liveness["report"]["liveness"]["profile"])
    first = _watchdog_observe(watchdog_liveness)
    next_boot = "c82a31a5-7008-41e2-b518-d01e719bfb42"
    watchdog_liveness["process"]["boot_id"] = next_boot
    watchdog_liveness["report"]["liveness"]["process"]["boot_id"] = next_boot
    write_json(watchdog_liveness["path"], watchdog_liveness["report"])
    second = _watchdog_observe(watchdog_liveness)
    assert first["available"] is True and second["available"] is True
    assert first["status"] == second["status"] == "stale"
    assert first["incident_id"] and first["incident_id"] != second["incident_id"]
    assert watchdog_liveness["unit"] == original_unit
    assert watchdog_liveness["report"]["liveness"]["profile"] == original_profile
    _assert_watchdog_observation_only(first)
    _assert_watchdog_observation_only(second)


def test_watchdog_liveness_uses_closing_clock_after_lane_work(watchdog_liveness, monkeypatch):
    config = watchdog_liveness["config"]
    clock = {"wall": NOW, "monotonic": _WATCHDOG_MONOTONIC}
    monkeypatch.setattr(fleet.time, "time", lambda: clock["wall"])
    monkeypatch.setattr(fleet.time, "monotonic", lambda: clock["monotonic"])
    watchdog_liveness["report"]["observed_at"] = NOW + 15
    write_json(watchdog_liveness["path"], watchdog_liveness["report"])
    values = {lane["monitor_config"]["service"]: healthy(lane) for lane in config["lanes"]}

    def completed_lane_work(per_lane, *, apply):
        clock.update(wall=NOW + 20, monotonic=_WATCHDOG_MONOTONIC + 20)
        return values[per_lane["service"]]

    monkeypatch.setattr(fleet.lane_monitor, "monitor", completed_lane_work)
    result = fleet.monitor_fleet(config)
    attachment = result["watchdog_liveness"]
    assert result["checked_at_epoch"] == NOW
    assert attachment["checked_at_epoch"] == NOW + 20
    assert attachment["available"] is True and attachment["status"] == "healthy"
    assert attachment["report_age_seconds"] == 5
    assert attachment["attention_required"] is False
    assert result["healthy"] is True and result["issues"] == []
    _assert_watchdog_observation_only(attachment)


@pytest.mark.parametrize("which", ["wall", "monotonic"])
def test_watchdog_liveness_rejects_backward_closing_clock(watchdog_liveness, monkeypatch, which):
    if which == "wall":
        monkeypatch.setattr(fleet.time, "time", lambda: NOW - 1)
    else:
        samples = iter([_WATCHDOG_MONOTONIC, _WATCHDOG_MONOTONIC - 1])
        monkeypatch.setattr(fleet.time, "monotonic", lambda: next(samples))
    result = _watchdog_observe(watchdog_liveness)
    assert result["available"] is False and result["status"] == "unavailable"
    assert "clock_changed" in result["reason_codes"]
    _assert_watchdog_observation_only(result)


def test_watchdog_liveness_failed_user_expansion_preserves_lane_observations(watchdog_liveness, monkeypatch, tmp_path):
    bad_path = "~fixture_missing_user/status.json"
    original_expanduser = Path.expanduser
    expansions = []

    def expanduser(path):
        if str(path) == bad_path:
            expansions.append(str(path))
            raise RuntimeError("fixture missing user; no system lookup performed")
        return original_expanduser(path)

    config_path = tmp_path / "fleet.json"
    raw_config = json.loads(config_path.read_text())
    raw_config["watchdog_liveness"] = {**watchdog_liveness["binding"], "status_path": bad_path}
    write_json(config_path, raw_config)
    monkeypatch.setattr(Path, "expanduser", expanduser)
    loaded = fleet.load_config(config_path)
    assert expansions == [bad_path]
    assert loaded["watchdog_liveness"]["status_path"] == bad_path
    calls = stub_health(loaded, monkeypatch)
    result = fleet.monitor_fleet(loaded, apply=True)
    assert result["watchdog_liveness"]["available"] is False
    assert result["watchdog_liveness"]["status"] == "unavailable"
    _assert_watchdog_observation_only(result["watchdog_liveness"])
    assert result["healthy"] is True and result["issues"] == []
    assert result["healthy_lane_count"] == 2
    assert all(lane["health"]["action"] == "none" for lane in result["lanes"])
    assert sorted(calls) == [("board-lane-0.service", True), ("board-lane-1.service", True)]


@pytest.mark.parametrize("state", ["inactive", "failed"])
def test_unheld_stopped_watchdog_has_stable_attention_without_action(watchdog_liveness, state):
    watchdog_liveness["unit"].update(ActiveState=state, SubState="dead", MainPID="0",
                                       ExecMainStartTimestampMonotonic="0")
    first = _watchdog_observe(watchdog_liveness)
    second = _watchdog_observe(watchdog_liveness)
    assert first["available"] is True and first["status"] == "inactive"
    assert first["attention_required"] is True
    assert first["reason_codes"] == ["service_not_active"]
    assert first["incident_id"].startswith("watchdog-liveness:")
    assert first["incident_id"] == second["incident_id"]
    _assert_watchdog_observation_only(first)


@pytest.mark.parametrize("field", ["schema", "observed_at", "process_pid"])
def test_duplicate_watchdog_report_fields_are_not_healthy(watchdog_liveness, field):
    report = watchdog_liveness["report"]
    raw = json.dumps(report)
    if field == "process_pid":
        pid = report["liveness"]["process"]["pid"]
        spelling = '"pid": ' + str(pid)
        assert raw.count(spelling) == 1
        raw = raw.replace(spelling, '"pid": 0, ' + spelling, 1)
    else:
        raw = '{"' + field + '": null, ' + raw[1:]
    watchdog_liveness["path"].write_text(raw)
    result = _watchdog_observe(watchdog_liveness)
    assert result["status"] == "unavailable" and result["available"] is False
    assert result["attention_required"] is True
    assert result["reason_codes"] == ["report_unavailable"]
    assert result["incident_id"] is not None
    _assert_watchdog_observation_only(result)


def test_fresh_dependency_wait_report_is_watchdog_progress(watchdog_liveness, monkeypatch):
    watchdog_liveness["report"]["boards"] = {
        "scientific": {"health": "blocked", "planned_action": "",
                       "observation": {"reason_codes": ["native_catalogue_activation_missing"]}}
    }
    write_json(watchdog_liveness["path"], watchdog_liveness["report"])
    stub_health(watchdog_liveness["config"], monkeypatch)
    result = fleet.monitor_fleet(watchdog_liveness["config"], apply=True)
    attachment = result["watchdog_liveness"]
    assert attachment["status"] == "healthy" and attachment["attention_required"] is False
    assert attachment["incident_id"] is None
    assert result["healthy"] is True and result["issues"] == []
    _assert_watchdog_observation_only(attachment)


@pytest.mark.parametrize("lane_seconds", [45.001, 75, 100])
def test_slow_lanes_skip_optional_watchdog_queries_and_publish(watchdog_liveness, monkeypatch, lane_seconds):
    config = watchdog_liveness["config"]
    clock = {"monotonic": _WATCHDOG_MONOTONIC}
    monkeypatch.setattr(fleet.time, "monotonic", lambda: clock["monotonic"])
    values = {lane["monitor_config"]["service"]: healthy(lane) for lane in config["lanes"]}

    def slow_lane(per_lane, *, apply):
        clock["monotonic"] = _WATCHDOG_MONOTONIC + lane_seconds
        return values[per_lane["service"]]

    monkeypatch.setattr(fleet.lane_monitor, "monitor", slow_lane)
    result = fleet.monitor_fleet(config, apply=True)
    attachment = result["watchdog_liveness"]
    assert attachment["status"] == "unavailable"
    assert attachment["reason_codes"] == ["observation_budget_exhausted"]
    assert watchdog_liveness["systemd_calls"] == []
    assert watchdog_liveness["process_calls"] == []
    assert result["healthy"] is True and result["issues"] == []
    assert json.loads(Path(config["health_path"]).read_text())["healthy"] is True
    _assert_watchdog_observation_only(attachment)


@pytest.mark.parametrize("lane_seconds", [44.999, 45])
def test_watchdog_queries_fit_exact_cycle_reserve_boundary(watchdog_liveness, monkeypatch, lane_seconds):
    config = watchdog_liveness["config"]
    clock = {"monotonic": _WATCHDOG_MONOTONIC}
    monkeypatch.setattr(fleet.time, "monotonic", lambda: clock["monotonic"])
    values = {lane["monitor_config"]["service"]: healthy(lane) for lane in config["lanes"]}

    def completed_lane(per_lane, *, apply):
        clock["monotonic"] = _WATCHDOG_MONOTONIC + lane_seconds
        return values[per_lane["service"]]

    monkeypatch.setattr(fleet.lane_monitor, "monitor", completed_lane)
    result = fleet.monitor_fleet(config)
    attachment = result["watchdog_liveness"]
    assert attachment["status"] == "healthy" and attachment["available"] is True
    assert len(watchdog_liveness["systemd_calls"]) == 2
    assert result["healthy"] is True and result["issues"] == []
    _assert_watchdog_observation_only(attachment)


def test_watchdog_closing_query_is_skipped_when_local_observation_consumes_reserve(watchdog_liveness, monkeypatch):
    config = watchdog_liveness["config"]
    clock = {"monotonic": _WATCHDOG_MONOTONIC}
    monkeypatch.setattr(fleet.time, "monotonic", lambda: clock["monotonic"])
    stub_health(config, monkeypatch)
    original_reader = fleet._read_watchdog_liveness_report

    def slow_read(path):
        clock["monotonic"] += 31
        return original_reader(path)

    monkeypatch.setattr(fleet, "_read_watchdog_liveness_report", slow_read)
    result = fleet.monitor_fleet(config, apply=True)
    attachment = result["watchdog_liveness"]
    assert attachment["status"] == "unavailable"
    assert attachment["reason_codes"] == ["observation_budget_exhausted"]
    assert len(watchdog_liveness["systemd_calls"]) == 1
    assert result["healthy"] is True and result["issues"] == []
    assert json.loads(Path(config["health_path"]).read_text())["healthy"] is True
    _assert_watchdog_observation_only(attachment)
