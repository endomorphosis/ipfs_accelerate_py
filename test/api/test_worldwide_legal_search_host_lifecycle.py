from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters import worldwide_legal_search_host_lifecycle as host
from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_watchdog import WorldwideWatchdogBridge, native_identity, read_owned_json


@pytest.fixture
def registered(tmp_path, monkeypatch):
    repository, state, home = tmp_path / "repo", tmp_path / "state", tmp_path / "home"
    repository.mkdir()
    state.mkdir()
    home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    config_path = repository / "config/worldwide_supervisor_host.json"
    config_path.parent.mkdir()
    config = {"schema": "worldwide-legal-search.host@1", "site_root": str(repository), "state_root": str(state),
              "runtime_argv": ["python", "-m", host.MODULE, "run", "--config", str(config_path)]}
    config_path.write_text(json.dumps(config))
    fragment = home / ".config/systemd/user" / host.UNIT
    fragment.parent.mkdir(parents=True)
    fragment.write_text("[Service]\nWorkingDirectory=" + str(repository) + "\nExecStart=python -m " + host.MODULE + " run --config " + str(config_path) + "\nKillMode=control-group\n")
    clock = [10000.0]
    native = {"pid": 101, "ready": False, "wrong_identity": False}
    group = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + host.UNIT
    calls = []

    def process(pid):
        return {"pid": pid, "alive": pid > 0 and pid == native["pid"], "start_ticks": pid * 10, "uid": os.getuid(),
                "argv": ["python", "-m", "other.module" if native["wrong_identity"] else host.MODULE, "run", "--config", str(config_path)], "cgroups": [group]}

    monkeypatch.setattr(host, "process_identity", process)

    def command(argv, timeout):
        calls.append(argv)
        assert argv[:2] == ["systemctl", "--user"]
        assert host.UNIT in argv
        if argv[2] == "show":
            return "\n".join(["Id=" + host.UNIT, "MainPID=" + str(native["pid"]), "ControlGroup=" + (group if native["pid"] else ""), "FragmentPath=" + str(fragment), "ActiveState=" + ("active" if native["pid"] else "inactive"), "SubState=" + ("running" if native["pid"] else "dead"), "KillMode=control-group"])
        if argv[2] == "stop":
            native["pid"] = 0
        elif argv[2] == "start":
            native.update(pid=202, ready=True)
            projection_path = state / "campaign-status.json"
            projection = json.loads(projection_path.read_text()) if projection_path.exists() else {}
            projection.update(run_id="wwl-test", owner_pid=202, observed_at=clock[0])
            projection_path.write_text(json.dumps(projection))
        else:
            pytest.fail("unexpected command")
        return ""

    lifecycle = host.RegisteredSystemdLifecycle(repository=repository, state_root=state, run_id="wwl-test", command=command,
                clock=lambda: clock[0], monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(lifecycle, "_cgroup_empty", lambda value: True)
    return lifecycle, native, calls, config_path, clock


def test_registered_host_recovery_fences_then_starts_only_exact_unit(registered):
    lifecycle, native, calls, config_path, clock = registered
    action = {"action_id": "a" * 64, "run_id": "wwl-test", "policy_id": "wwl-policy", "kind": "recover"}
    result = lifecycle.recover(action)
    assert result["recovered"] is True
    assert result["old_owner_fenced"] is True
    assert result["receipt"]
    mutations = [item for item in calls if item[2] != "show"]
    assert mutations == [["systemctl", "--user", "stop", host.UNIT], ["systemctl", "--user", "start", host.UNIT]]
    assert native["pid"] == 202
    # A repeated action consumes the native durable receipt, not another stop.
    calls.clear()
    repeated = lifecycle.recover(action)
    assert repeated["deduplicated"] is True
    assert not [item for item in calls if item[2] != "show"]


def test_lost_owner_can_be_recovered_without_owner_rpc(registered):
    lifecycle, native, calls, config_path, clock = registered
    native["pid"] = 0
    result = lifecycle.recover({"action_id": "b" * 64, "run_id": "wwl-test", "policy_id": "wwl-policy", "kind": "recover"})
    assert result["recovered"] is True
    assert native["pid"] == 202


def test_wrong_runtime_owner_is_never_stopped(registered):
    lifecycle, native, calls, config_path, clock = registered
    native["wrong_identity"] = True
    with pytest.raises(host.HostLifecycleError, match="MainPID"):
        lifecycle.recover({"action_id": "c" * 64, "run_id": "wwl-test", "policy_id": "wwl-policy", "kind": "recover"})
    assert not [item for item in calls if item[2] != "show"]


def test_other_service_registration_is_rejected_before_control(registered):
    lifecycle, native, calls, config_path, clock = registered
    config = json.loads(config_path.read_text())
    config["supervisor_unit"] = "another-important-service.service"
    config_path.write_text(json.dumps(config))
    calls.clear()
    with pytest.raises(host.HostLifecycleError, match="exact registered"):
        host.RegisteredSystemdLifecycle(repository=lifecycle.repository, state_root=lifecycle.state_root, run_id="wwl-test")
    assert calls == []


def test_other_campaign_action_is_rejected_before_control(registered):
    lifecycle, native, calls, config_path, clock = registered
    with pytest.raises(host.HostLifecycleError, match="registered campaign"):
        lifecycle.recover({"action_id": "d" * 64, "run_id": "different-campaign", "policy_id": "wwl-policy", "kind": "recover"})
    assert not [item for item in calls if item[2] != "show"]


def test_adapter_dispatch_recovers_a_dead_owner_and_preserves_single_authority(registered):
    lifecycle, native, calls, config_path, clock = registered
    state = lifecycle.state_root
    native["pid"] = 0
    (state / "campaign-readiness.json").write_text(json.dumps({"schema": "wwl.watchdog.ready@1", "run_id": "wwl-test", "repository": str(lifecycle.repository), "policy_id": "wwl-policy", "native_revision": "accepted", "admitted_plan_revision": "plan-1", "required_task_ids": ["WWL-000"], "bindings": {"reconcile": True, "lifecycle_restart": False, "repair_dispatch": False}}))
    (state / "campaign-status.json").write_text(json.dumps({"schema": "wwl.watchdog.snapshot@1", "run_id": "wwl-test", "owner_pid": 101, "observed_at": 9700, "heartbeat_at": 9700,
        "progress_token": "last-real-phase", "authority_revision": "source-1", "accepted_tree": "tree-1", "hold": False}))
    bridge = WorldwideWatchdogBridge(repository=lifecycle.repository, state_root=state, run_id="wwl-test", clock=lambda: clock[0], host_lifecycle=lifecycle)
    request = {"schema": "wwl.watchdog.action@1", "action_id": "e" * 64, "run_id": "wwl-test", "policy_id": "wwl-policy", "kind": "recover",
        "issued_at": clock[0], "authority_revision": "source-1", "accepted_tree": "tree-1"}
    result = bridge.dispatch(request, timeout_seconds=0)
    assert result["status"] == "accepted"
    assert not (state / "control/requests").exists()
    observed = bridge.observe({"run_id": "wwl-test", "policy_id": "wwl-policy", "action_ids": [request["action_id"]]})
    assert observed["action_observations"][0]["status"] == "succeeded"
    assert observed["action_observations"][0]["effects_verified"] is True
    calls.clear()
    assert bridge.dispatch(request, timeout_seconds=0) == result
    assert not [item for item in calls if item[2] != "show"]


def test_stale_task_projection_keeps_old_heartbeat_but_gets_real_host_observation(registered):
    lifecycle, native, calls, config_path, clock = registered
    state = lifecycle.state_root
    (state / "campaign-readiness.json").write_text(json.dumps({"schema": "wwl.watchdog.ready@1", "run_id": "wwl-test", "repository": str(lifecycle.repository), "policy_id": "wwl-policy", "native_revision": "accepted", "admitted_plan_revision": "plan-1", "required_task_ids": ["WWL-000"], "bindings": {"lifecycle_restart": True}}))
    snapshot = {"schema": "wwl.watchdog.snapshot@1", "run_id": "wwl-test", "owner_pid": 101, "observed_at": 9700, "heartbeat_at": 9700,
                "progress_token": "last-real-phase", "authority_revision": "source-1", "completion_verified": True, "completion_receipt": "old-root-proof"}
    (state / "campaign-status.json").write_text(json.dumps(snapshot))
    native["pid"] = 0
    bridge = WorldwideWatchdogBridge(repository=lifecycle.repository, state_root=state, run_id="wwl-test", clock=lambda: clock[0], host_lifecycle=lifecycle)
    observed = bridge.observe({"run_id": "wwl-test", "policy_id": "wwl-policy"})
    assert observed["observed_at"] == 10000
    assert observed["projection_observed_at"] == 9700
    assert observed["heartbeat_at"] == 9700
    assert observed["progress_token"] == "last-real-phase"
    assert observed["owner_alive"] is False
    assert observed["source_projection_stale"] is True
    assert observed["completion_verified"] is False
    assert not observed["completion_receipt"]


@pytest.fixture
def promoted(registered):
    lifecycle, native, calls, config_path, clock = registered
    state = lifecycle.state_root
    native["pid"] = 0

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    failure = {"schema": "wwl.native-failure@1", "run_id": "wwl-test", "observed_at": 9000, "task_id": "WWL-000", "phase": "preselection", "effects_unknown": False}
    failure["receipt_id"] = native_identity(failure)
    write(state / "failures/original.json", failure)
    repair_id = "a" * 64
    original = {"schema": "wwl.watchdog.action@1", "run_id": "wwl-test", "policy_id": "wwl-policy", "kind": "repair", "action_id": repair_id, "issued_at": 9800,
                "reproducer_receipt": failure["receipt_id"]}
    observation = {"schema": "wwl.watchdog.action-observation@1", "action_id": repair_id, "status": "running", "receipt": "native:promotion",
        "candidate_promotion_verified": True, "recovery_verified": False, "requires_fenced_reload": True, "effects_verified": True,
        "expected_framework_revision": "2" * 40, "original_failure_receipt": failure["receipt_id"], "deadline_at": 10120}
    write(state / "control/action-ledger.json", {repair_id: {"request": original, "result": {"action_id": repair_id, "status": "accepted", "deadline_at": 10120}, "observation": observation}})
    write(state / "repairs" / repair_id / "native-repair-result.json", {"schema": "wwl.native-repair-result@1", "action_id": repair_id, "validated": True,
        "native_completion_receipt": "native:completion", "native_projection": "native:projection", "action_request_identity": native_identity(original),
        "framework_head": "2" * 40, "original_failure_receipt": failure["receipt_id"]})
    write(state / "campaign-readiness.json", {"schema": "wwl.watchdog.ready@1", "run_id": "wwl-test", "repository": str(lifecycle.repository), "policy_id": "wwl-policy", "native_revision": "2" * 40, "admitted_plan_revision": "plan-1", "required_task_ids": ["WWL-000"], "bindings": {"reconcile": True, "lifecycle_restart": False, "repair_dispatch": True}})
    snapshot = {"schema": "wwl.watchdog.snapshot@1", "run_id": "wwl-test", "owner_pid": 101, "observed_at": 9990, "heartbeat_at": 9990,
        "progress_token": "last-real-phase", "authority_revision": "source-1", "accepted_tree": "tree-1", "hold": False, "native_error": failure,
        "action_observations": [observation], "loaded_framework_revision": "1" * 40}
    write(state / "campaign-status.json", snapshot)
    bridge = WorldwideWatchdogBridge(repository=lifecycle.repository, state_root=state, run_id="wwl-test", clock=lambda: clock[0], host_lifecycle=lifecycle)
    request = {"schema": "wwl.watchdog.action@1", "action_id": "b" * 64, "run_id": "wwl-test", "policy_id": "wwl-policy", "kind": "recover",
        "issued_at": clock[0], "authority_revision": "source-1", "accepted_tree": "tree-1", "reload_after_action_id": repair_id,
        "expected_framework_revision": "2" * 40, "original_failure_receipt": failure["receipt_id"]}
    return bridge, lifecycle, native, calls, clock, request, repair_id, snapshot


def publish_recovered_pass(promoted, **changes):
    bridge, lifecycle, native, calls, clock, request, repair_id, snapshot = promoted
    result = {"active_task_id": "WWL-000", "implementation_result": {"status": "completed"}}
    proof = {"owner_pid": native["pid"], "framework_revision": "2" * 40, "completed_at": 10001, "result": result,
        "authority_revision": "source-2", "progress_token": "actual-new-native-pass", "receipt_id": native_identity(result)}
    after = {**snapshot, "owner_pid": native["pid"], "observed_at": clock[0], "heartbeat_at": clock[0], "loaded_framework_revision": "2" * 40,
        "native_error": {}, "effects_unknown": False, "native_passes": 1, "last_successful_native_pass": proof,
        "verified_task_ids": ["WWL-000"], "task_statuses": {"WWL-000": "completed"},
        "native_completion_receipts_by_alias": {"WWL-000": "native:real-task-completion"}, "native_completion_projection": "native:canonical-proof", **changes}
    (bridge.state_root / "campaign-status.json").write_text(json.dumps(after))
    return after


def test_promoted_repair_requires_fence_exact_source_and_actual_native_pass(promoted):
    bridge, lifecycle, native, calls, clock, request, repair_id, snapshot = promoted
    assert bridge.dispatch(request, timeout_seconds=0)["status"] == "accepted"
    query = {"run_id": "wwl-test", "policy_id": "wwl-policy", "action_ids": [repair_id, request["action_id"]]}
    initial = bridge.observe(query)
    assert next(item for item in initial["action_observations"] if item["action_id"] == repair_id)["status"] == "running"
    assert not (bridge.state_root / "control/repair-terminal-observations").exists()
    clock[0] = 10002
    publish_recovered_pass(promoted)
    calls.clear()
    observed = bridge.observe(query)
    terminal = next(item for item in observed["action_observations"] if item["action_id"] == repair_id)
    assert terminal["status"] == "succeeded"
    assert terminal["recovery_verified"] is True
    receipt = read_owned_json(bridge.state_root / "control/repair-settlement-receipts" / (repair_id + ".json"))
    assert receipt["native_pass"]["owner_pid"] == 202
    assert receipt["original_fault_cleared"] is True
    assert not [item for item in calls if item[2] != "show"]
    assert bridge.observe(query)["action_observations"] == observed["action_observations"]
    later = bridge.observe({"run_id": "wwl-test", "policy_id": "wwl-policy"})
    assert next(item for item in later["action_observations"] if item["action_id"] == repair_id)["status"] == "succeeded"


@pytest.mark.parametrize("change", ["wrong_revision", "wrong_failure", "unvalidated_result", "missing_ledger"])
def test_reload_without_exact_native_promotion_never_controls_unit(promoted, change):
    bridge, lifecycle, native, calls, clock, request, repair_id, snapshot = promoted
    if change == "wrong_revision":
        request["expected_framework_revision"] = "3" * 40
    elif change == "wrong_failure":
        request["original_failure_receipt"] = "native:unrelated"
    elif change == "missing_ledger":
        (bridge.state_root / "control/action-ledger.json").unlink()
    else:
        path = bridge.state_root / "repairs" / repair_id / "native-repair-result.json"
        value = json.loads(path.read_text()); value["validated"] = False; path.write_text(json.dumps(value))
    assert bridge.dispatch(request, timeout_seconds=0)["status"] == "blocked"
    assert not [item for item in calls if item[2] != "show"]
    assert not (bridge.state_root / "control/host-requests").exists()


@pytest.mark.parametrize("change", ["wrong_revision", "wrong_owner", "old_pass", "same_fault", "wrong_task", "cleared_bool_only", "unknown_effects", "blocked_native_pass", "missing_completion_receipt"])
def test_restart_or_cleared_boolean_alone_cannot_settle_repair(promoted, change):
    bridge, lifecycle, native, calls, clock, request, repair_id, snapshot = promoted
    assert bridge.dispatch(request, timeout_seconds=0)["status"] == "accepted"
    clock[0] = 10002
    after = publish_recovered_pass(promoted)
    proof = after["last_successful_native_pass"]
    if change == "wrong_revision":
        after["loaded_framework_revision"] = "3" * 40
    elif change == "wrong_owner":
        proof["owner_pid"] = 101
    elif change == "old_pass":
        proof["completed_at"] = 9999
    elif change == "same_fault":
        after["native_error"] = snapshot["native_error"]
    elif change == "wrong_task":
        proof["result"]["active_task_id"] = "WWL-999"
        proof["receipt_id"] = native_identity(proof["result"])
    elif change == "unknown_effects":
        after["effects_unknown"] = True
    elif change == "blocked_native_pass":
        proof["result"]["implementation_result"] = {"status": "blocked", "reason": "implementation_protected_path_mutated"}
        proof["receipt_id"] = native_identity(proof["result"])
        after["verified_task_ids"] = []
        after["task_statuses"] = {"WWL-000": "blocked"}
        after["native_completion_receipts_by_alias"] = {}
    elif change == "missing_completion_receipt":
        after["native_completion_receipts_by_alias"] = {}
    else:
        after.pop("last_successful_native_pass")
    (bridge.state_root / "campaign-status.json").write_text(json.dumps(after))
    observed = bridge.observe({"run_id": "wwl-test", "policy_id": "wwl-policy", "action_ids": [repair_id]})
    assert observed["action_observations"][0]["status"] == "running"
    assert not (bridge.state_root / "control/repair-terminal-observations").exists()


def test_proven_recovery_after_original_deadline_does_not_refund_repair_budget(promoted):
    bridge, lifecycle, native, calls, clock, request, repair_id, snapshot = promoted
    assert bridge.dispatch(request, timeout_seconds=0)["status"] == "accepted"
    clock[0] = 10122
    after = publish_recovered_pass(promoted)
    after["last_successful_native_pass"]["completed_at"] = 10121
    (bridge.state_root / "campaign-status.json").write_text(json.dumps(after))
    observed = bridge.observe({"run_id": "wwl-test", "policy_id": "wwl-policy", "action_ids": [repair_id]})
    terminal = observed["action_observations"][0]
    assert terminal["status"] == "failed"
    assert terminal["effects_verified"] is True
    assert terminal["recovery_verified"] is True
