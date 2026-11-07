"""Exercise the WWL bridge against owner projections without a live database."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_watchdog import (
    WatchdogBridgeError, WorldwideWatchdogBridge, read_owned_json,
)


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def bridge(tmp_path):
    repository, state = tmp_path / "repo", tmp_path / "state"
    repository.mkdir()
    state.mkdir()
    ready = {
        "schema": "wwl.watchdog.ready@1", "run_id": "wwl-test", "repository": str(repository),
        "policy_id": "wwl-policy", "native_revision": "accepted-native-revision", "admitted_plan_revision": "plan-1",
        "required_task_ids": ["WWL-000"], "readiness_receipt": "native:actual-test-evidence",
        "bindings": {"task_source": True, "meaningful_progress": True, "reconcile": True, "lifecycle_restart": True, "repair_dispatch": False},
    }
    snapshot = {
        "schema": "wwl.watchdog.snapshot@1", "run_id": "wwl-test", "observed_at": 10000,
        "authority_revision": "task-source-1", "accepted_tree": "accepted-tree-1", "hold": False,
        "admitted_plan_revision": "plan-1", "heartbeat_at": 10000, "progress_token": "validated-phase-1",
    }
    write(state / "campaign-readiness.json", ready)
    write(state / "campaign-status.json", snapshot)
    return WorldwideWatchdogBridge(repository=repository, state_root=state, run_id="wwl-test", clock=lambda: 10000), ready, snapshot


def action(kind="reconcile"):
    return {
        "schema": "wwl.watchdog.action@1", "action_id": "a" * 64, "run_id": "wwl-test", "policy_id": "wwl-policy",
        "kind": kind, "authority_revision": "task-source-1", "accepted_tree": "accepted-tree-1", "issued_at": 10000,
        "incident_id": "incident-1", "reason": "reproduced-failure", "task_id": "WWL-000", "stage": "validation",
    }


def test_ready_preserves_truthful_unbound_repair_capability(bridge):
    adapter, ready, snapshot = bridge
    result = adapter.ready({"policy_id": "wwl-policy", "required_task_ids": ["WWL-000"]})
    assert result == ready
    assert result["bindings"]["repair_dispatch"] is False


def test_observation_never_refreshes_owner_heartbeat_or_progress(bridge):
    adapter, ready, snapshot = bridge
    result = adapter.observe({"policy_id": "wwl-policy", "run_id": "wwl-test"})
    assert all(result[key] == value for key, value in snapshot.items())
    assert result["action_observations"] == []
    adapter.clock = lambda: 10200
    with pytest.raises(WatchdogBridgeError, match="stale"):
        adapter.observe({"policy_id": "wwl-policy", "run_id": "wwl-test"})


def test_execute_mode_recovers_a_crashed_maintenance_pause_without_task_writes(bridge):
    adapter, _, _ = bridge
    adapter.state_root.chmod(0o700)
    marker = adapter.state_root / "HOLD"
    code = ("from pathlib import Path; import sys; "
            "from ipfs_accelerate_py.agent_supervisor.rescue.maintenance_hold import create_hold; "
            "create_hold(Path(sys.argv[1]), reason='bounded maintenance', ttl_seconds=1, now=9990)")
    subprocess.run([sys.executable, "-c", code, str(marker)], check=True, timeout=20)
    from ipfs_accelerate_py.agent_supervisor.rescue.maintenance_hold import hold_binding
    native = read_owned_json(adapter.state_root / "campaign-status.json")
    write(adapter.state_root / "campaign-status.json", {**native, "hold": True,
          "maintenance_hold_bindings": [hold_binding(marker, now=10000)]})
    request = {"policy_id": "wwl-policy", "run_id": "wwl-test"}
    assert adapter.observe(request)["hold"] is True
    assert marker.exists()
    before = (adapter.state_root / "campaign-status.json").read_bytes()
    result = adapter.observe({**request, "execution_requested": True})
    assert result["hold"] is False and result["hold_reviews"][0]["status"] == "released"
    assert adapter.observe(request)["hold"] is False
    assert not marker.exists()
    assert (adapter.state_root / "campaign-status.json").read_bytes() == before


def test_execute_mode_keeps_manual_maintenance_holds(bridge):
    adapter, _, _ = bridge
    marker = adapter.state_root / "HOLD"
    marker.write_text("operator maintenance")
    result = adapter.observe({"policy_id": "wwl-policy", "run_id": "wwl-test", "execution_requested": True})
    assert result["hold"] is True and marker.read_text() == "operator maintenance"


def test_repair_without_binding_is_a_durable_no_effect_refusal(bridge):
    adapter, _, _ = bridge
    result = adapter.dispatch(action("repair"), timeout_seconds=0)
    assert result["status"] == "blocked"
    assert not (adapter.state_root / "control/requests").exists()
    refusal = read_owned_json(adapter.state_root / "control/refusals" / ("a" * 64 + ".json"))
    assert refusal["effects_applied"] is False


def test_dispatch_without_ack_is_unknown_and_idempotent(bridge):
    adapter, _, _ = bridge
    request = action()
    first = adapter.dispatch(request, timeout_seconds=0)
    second = adapter.dispatch(request, timeout_seconds=0)
    assert first == second
    assert first["status"] == "unknown"
    requests = list((adapter.state_root / "control/requests").glob("*.json"))
    assert len(requests) == 1
    assert read_owned_json(requests[0]) == request


def test_admitted_native_result_is_not_reclassified_after_scope_advances(bridge):
    adapter, _, snapshot = bridge
    request = action()
    assert adapter.dispatch(request, timeout_seconds=0)["status"] == "unknown"
    result = {"schema": "wwl.watchdog.action-result@1", "action_id": request["action_id"], "status": "accepted", "deadline_at": 10120, "receipt": "native:admitted-operation"}
    write(adapter.state_root / "control/results" / (request["action_id"] + ".json"), result)
    write(adapter.state_root / "campaign-status.json", {**snapshot, "authority_revision": "task-source-2"})
    assert adapter.dispatch(request, timeout_seconds=0) == result


def test_existing_uncertain_submission_cannot_become_no_effect_refusal(bridge):
    adapter, _, snapshot = bridge
    request = action()
    assert adapter.dispatch(request, timeout_seconds=0)["status"] == "unknown"
    write(adapter.state_root / "campaign-status.json", {**snapshot, "hold": True, "authority_revision": "task-source-2"})
    assert adapter.dispatch(request, timeout_seconds=0)["status"] == "unknown"
    assert not (adapter.state_root / "control/refusals").exists()


def test_action_identity_reuse_with_different_body_is_rejected(bridge):
    adapter, _, _ = bridge
    original = action()
    adapter.dispatch(original, timeout_seconds=0)
    with pytest.raises(WatchdogBridgeError, match="reused"):
        adapter.dispatch({**original, "reason": "different-effect"}, timeout_seconds=0)


def test_stale_revision_cannot_enqueue_a_new_action(bridge):
    adapter, _, snapshot = bridge
    write(adapter.state_root / "campaign-status.json", {**snapshot, "authority_revision": "task-source-2"})
    assert adapter.dispatch(action(), timeout_seconds=0)["status"] == "blocked"
    assert not (adapter.state_root / "control/requests").exists()


def test_native_hold_cannot_enqueue_a_new_action(bridge):
    adapter, _, snapshot = bridge
    write(adapter.state_root / "campaign-status.json", {**snapshot, "hold": True})
    assert adapter.dispatch(action(), timeout_seconds=0)["status"] == "blocked"


@pytest.mark.parametrize("name", ["HOLD", "STOP", "OPERATOR_STOP"])
def test_current_operator_hold_cannot_be_hidden_by_old_projection(bridge, name):
    adapter, _, snapshot = bridge
    assert snapshot["hold"] is False
    (adapter.state_root / name).touch()
    assert adapter.dispatch(action(), timeout_seconds=0)["status"] == "blocked"
    assert not (adapter.state_root / "control/requests").exists()


@pytest.mark.parametrize("issued_at", [None, "10000", float("nan"), float("inf"), 10031])
def test_malformed_action_time_cannot_be_published(bridge, issued_at):
    adapter, _, _ = bridge
    with pytest.raises(WatchdogBridgeError, match="issue time"):
        adapter.dispatch({**action(), "issued_at": issued_at}, timeout_seconds=0)
    assert not (adapter.state_root / "control").exists()


def test_projection_and_action_path_escape_are_refused(bridge):
    adapter, _, _ = bridge
    bad = action()
    bad["action_id"] = "../../another-campaign"
    with pytest.raises(WatchdogBridgeError, match="malformed"):
        adapter.dispatch(bad, timeout_seconds=0)
    with pytest.raises(WatchdogBridgeError, match="identity"):
        adapter.observe({"policy_id": "wwl-policy", "run_id": "other-run"})


def test_live_database_is_not_opened_by_projection_operations(bridge):
    adapter, _, _ = bridge
    sentinel = adapter.state_root / "live.duckdb"
    sentinel.write_bytes(b"one-owner-only")
    adapter.ready({"policy_id": "wwl-policy", "required_task_ids": ["WWL-000"]})
    adapter.observe({"policy_id": "wwl-policy", "run_id": "wwl-test"})
    adapter.dispatch(action(), timeout_seconds=0)
    assert sentinel.read_bytes() == b"one-owner-only"


def test_diagnosis_binding_is_truthful_and_dispatch_does_not_open_task_database(bridge):
    adapter, ready, snapshot = bridge
    assert adapter.observe({"policy_id": "wwl-policy", "run_id": "wwl-test"})["native_diagnostic_available"] is False
    request = {**action("diagnose"), "task_id": "WWL-000", "active_attempt_id": "attempt:original"}
    assert adapter.dispatch(request, timeout_seconds=0)["status"] == "blocked"
    assert not (adapter.state_root / "control/requests").exists()


def test_diagnosis_requires_exact_native_attempt_before_publish(bridge):
    adapter, ready, snapshot = bridge
    ready["bindings"]["diagnose"] = True
    write(adapter.state_root / "campaign-readiness.json", ready)
    snapshot.update(task_id="WWL-000", active_attempt_id="attempt:original")
    write(adapter.state_root / "campaign-status.json", snapshot)
    assert adapter.observe({"policy_id": "wwl-policy", "run_id": "wwl-test"})["native_diagnostic_available"] is True
    request = {**action("diagnose"), "task_id": "WWL-000", "active_attempt_id": "attempt:other"}
    assert adapter.dispatch(request, timeout_seconds=0)["status"] == "blocked"
    assert not (adapter.state_root / "control/requests").exists()
    admitted = {**request, "action_id": "b" * 64, "active_attempt_id": "attempt:original"}
    result = adapter.dispatch(admitted, timeout_seconds=0)
    assert result["status"] == "unknown"
    assert read_owned_json(adapter.state_root / "control/requests" / ("b" * 64 + ".json")) == admitted
