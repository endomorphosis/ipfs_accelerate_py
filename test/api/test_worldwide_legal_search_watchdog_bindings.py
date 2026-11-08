"""WWL-004 bindings: observe, dispatch and verify over native authorities."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_watchdog import (
    WatchdogBridgeError, WorldwideWatchdogBridge, main, native_identity, read_owned_json,
)
from ipfs_accelerate_py.agent_supervisor.rescue.rescue_planner import RescuePlanningRequest


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def completion_proof(run_id: str, tree: str, tasks: list[str]) -> dict:
    body = {
        "schema": "wwl.native-current-tree-completion@1", "run_id": run_id, "accepted_tree": tree,
        "required_task_ids": tasks, "verified_task_ids": list(tasks), "runnable_count": 0, "task_id": "",
        "effects_unknown": False, "hold": False,
    }
    return {**body, "completion_receipt": native_identity(body)}


class Planner:
    def __init__(self, result=None, error: BaseException | None = None):
        self.calls = 0
        self.requests = []
        self.result = {"disposition": "no_plan", "provider_invoked": False, "reason_code": "policy_denied"} if result is None else result
        self.error = error

    def plan(self, request):
        self.calls += 1
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.result


class Orchestrator:
    def __init__(self, result=None, error: BaseException | None = None):
        self.calls = 0
        self.requests = []
        self.result = {"disposition": "stopped", "recovered": False, "completion_authority": False} if result is None else result
        self.error = error

    def execute(self, request):
        self.calls += 1
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def campaign(tmp_path):
    repository, state = tmp_path / "repo", tmp_path / "state"
    repository.mkdir()
    state.mkdir()
    ready = {
        "schema": "wwl.watchdog.ready@1", "run_id": "wwl-test", "repository": str(repository),
        "policy_id": "wwl-policy", "native_revision": "accepted-native-revision", "admitted_plan_revision": "plan-1",
        "required_task_ids": ["WWL-000", "WWL-001"], "readiness_receipt": "native:readiness",
        "bindings": {"task_source": True, "meaningful_progress": True, "reconcile": True, "lifecycle_restart": False,
                     "repair_dispatch": False, "incident_deduplication": False, "rescue_planner": False, "rescue_orchestrator": False},
    }
    snapshot = {
        "schema": "wwl.watchdog.snapshot@1", "run_id": "wwl-test", "observed_at": 10000,
        "authority_revision": "task-source-1", "accepted_tree": "tree-1", "hold": False,
        "admitted_plan_revision": "plan-1", "heartbeat_at": 10000, "progress_token": "validated-phase-1",
        "required_task_ids": ["WWL-000", "WWL-001"], "verified_task_ids": [], "task_id": "WWL-000",
        "runnable_count": 1, "effects_unknown": False, "stage": "implementation",
    }
    write(state / "campaign-readiness.json", ready)
    write(state / "campaign-status.json", snapshot)
    return repository, state, ready, snapshot


def bridge(campaign, **authorities):
    repository, state, ready, snapshot = campaign
    return WorldwideWatchdogBridge(repository=repository, state_root=state, run_id="wwl-test", clock=lambda: 10000, **authorities), ready, snapshot


def action(kind="reconcile", action_id="a" * 64, incident_id="incident-1"):
    body = {
        "schema": "wwl.watchdog.action@1", "action_id": action_id, "run_id": "wwl-test", "policy_id": "wwl-policy",
        "kind": kind, "authority_revision": "task-source-1", "accepted_tree": "tree-1", "issued_at": 10000,
        "reason": "reproduced-failure", "task_id": "WWL-000", "stage": "validation",
    }
    if incident_id is not None:
        body["incident_id"] = incident_id
    return body


def diagnosis():
    incident = SimpleNamespace(incident_cid="cid-incident", evidence_cids=("sha256:evidence",))
    return SimpleNamespace(incident=incident, kind=SimpleNamespace(value="stall"), reason_codes=("meaningful_progress_stalled",), health={"healthy": False}, lease_id="lease-1", fencing_epoch=2)


def roots():
    return {"repository_root_cid": "repo-root", "run_cid": "run-root", "policy_root": "policy-root"}


def test_unbound_authorities_do_not_invent_mutation_readiness(campaign):
    adapter, ready, _ = bridge(campaign)
    result = adapter.ready({"policy_id": "wwl-policy", "required_task_ids": ["WWL-000"]})
    # The durable incident ledger is installed with the adapter. Lifecycle, rescue
    # and repair flags stay false until those native handlers are actually bound.
    assert result["bindings"]["incident_deduplication"] is True
    assert result["bindings"]["lifecycle_restart"] is False
    assert result["bindings"]["rescue_planner"] is False
    assert result["bindings"]["rescue_orchestrator"] is False
    assert result["bindings"]["repair_dispatch"] is False
    assert result == {**ready, "bindings": {**ready["bindings"], "incident_deduplication": True}}
    assert adapter.rescue_request_factory is not None
    assert adapter.rescue_execution_request_factory is not None
    assert adapter.rescue_planner is None
    assert adapter.rescue_orchestrator is None
    with pytest.raises(WatchdogBridgeError, match="rescue planner is not bound"):
        adapter.plan_rescue(diagnosis(), object(), roots(), incident_id="incident-1")
    with pytest.raises(WatchdogBridgeError, match="rescue orchestrator is not bound"):
        adapter.execute_rescue(diagnosis(), object(), {"plan_cid": "plan-1"}, roots(), incident_id="incident-1")


def test_ready_wires_lifecycle_rescue_and_incident_dedup_only_when_bound(campaign):
    planner, orchestrator = Planner(), Orchestrator()

    def restart(info):
        return {"restarted": True, "new_pid": 40, "start_ticks": 9}

    adapter, _, _ = bridge(campaign, lifecycle_restart=restart, rescue_planner=planner, rescue_orchestrator=orchestrator, process_identity=lambda pid: {"pid": pid, "alive": True, "start_ticks": 9})
    result = adapter.ready({"policy_id": "wwl-policy", "required_task_ids": ["WWL-000"]})
    assert result["bindings"]["lifecycle_restart"] is True
    assert result["bindings"]["rescue_planner"] is True
    assert result["bindings"]["rescue_orchestrator"] is True
    assert result["bindings"]["incident_deduplication"] is True
    assert result["bindings"]["repair_dispatch"] is False
    watchdog = adapter.bind_supervisor_watchdog(adapter.repository / "manifest.json")
    assert watchdog.lifecycle_restart is restart
    assert watchdog.rescue_planner is planner
    assert watchdog.rescue_orchestrator is orchestrator
    assert watchdog.rescue_request_factory is adapter.rescue_request_factory
    assert watchdog.rescue_execution_request_factory is adapter.rescue_execution_request_factory


def test_rescue_factories_are_used_once_and_replay_is_durable(campaign):
    planner, orchestrator = Planner(), Orchestrator()
    adapter, _, _ = bridge(campaign, rescue_planner=planner, rescue_orchestrator=orchestrator)
    planned = adapter.plan_rescue(diagnosis(), {"exhaustion": "native"}, roots(), incident_id="incident-1")
    assert planner.calls == 1
    assert isinstance(planner.requests[0], RescuePlanningRequest)
    assert planner.requests[0].current_repository_root_cid == "repo-root"
    assert planner.requests[0].evidence_redacted is True
    assert planned["completion_authority"] is False
    assert adapter.plan_rescue(diagnosis(), {"exhaustion": "native"}, roots(), incident_id="incident-1") == planned
    assert planner.calls == 1
    executed = adapter.execute_rescue(diagnosis(), {"exhaustion": "native"}, {"plan_cid": "plan-1"}, roots(), incident_id="incident-1")
    assert orchestrator.calls == 1
    request = orchestrator.requests[0]
    assert request["schema"] == "wwl.watchdog.rescue-execution-request@1"
    assert request["completion_authority"] is False
    assert request["plan"] == {"plan_cid": "plan-1"}
    assert executed["completion_authority"] is False
    reloaded, _, _ = bridge(campaign, rescue_planner=planner, rescue_orchestrator=orchestrator)
    assert reloaded.execute_rescue(diagnosis(), {"exhaustion": "native"}, {"plan_cid": "plan-1"}, roots(), incident_id="incident-1") == executed
    assert orchestrator.calls == 1


def test_uncertain_rescue_effect_is_not_replayed(campaign):
    planner = Planner(error=RuntimeError("provider already received the prompt"))
    orchestrator = Orchestrator(error=RuntimeError("execution crossed an effect boundary"))
    adapter, _, _ = bridge(campaign, rescue_planner=planner, rescue_orchestrator=orchestrator)
    with pytest.raises(RuntimeError, match="prompt"):
        adapter.plan_rescue(diagnosis(), {"exhaustion": "native"}, roots(), incident_id="incident-9")
    unknown = adapter.plan_rescue(diagnosis(), {"exhaustion": "native"}, roots(), incident_id="incident-9")
    assert unknown["status"] == "unknown" and unknown["effects_unknown"] is True and unknown["completion_authority"] is False
    assert planner.calls == 1
    with pytest.raises(RuntimeError, match="effect boundary"):
        adapter.execute_rescue(diagnosis(), {"exhaustion": "native"}, {"plan_cid": "plan-1"}, roots(), incident_id="incident-8")
    assert adapter.execute_rescue(diagnosis(), {"exhaustion": "native"}, {"plan_cid": "plan-1"}, roots(), incident_id="incident-8")["effects_unknown"] is True
    assert orchestrator.calls == 1


def test_rescue_authority_cannot_grant_completion(campaign):
    adapter, _, _ = bridge(campaign, rescue_planner=Planner(result={"completion_authority": True}), rescue_orchestrator=Orchestrator())
    with pytest.raises(WatchdogBridgeError, match="completion"):
        adapter.plan_rescue(diagnosis(), {"exhaustion": "native"}, roots(), incident_id="incident-1")


def test_durable_incident_claim_blocks_a_second_action_id(campaign):
    adapter, _, _ = bridge(campaign, lifecycle_restart=lambda info: {"restarted": False})
    ready = read_owned_json(adapter.state_root / "campaign-readiness.json")
    ready["bindings"]["lifecycle_restart"] = True
    ready["bindings"]["reconcile"] = True
    write(adapter.state_root / "campaign-readiness.json", ready)
    first = action()
    assert adapter.dispatch(first, timeout_seconds=0)["status"] == "unknown"
    second = action(action_id="b" * 64)
    refused = adapter.dispatch(second, timeout_seconds=0)
    assert refused["status"] == "blocked"
    refusal = read_owned_json(adapter.state_root / "control/refusals" / ("b" * 64 + ".json"))
    assert refusal["reason"] == "durable_incident_already_claimed"
    assert refusal["effects_applied"] is False
    assert [path.name for path in (adapter.state_root / "control/requests").glob("*.json")] == ["a" * 64 + ".json"]
    reloaded, _, _ = bridge(campaign, lifecycle_restart=lambda info: {"restarted": False})
    assert reloaded.dispatch(second, timeout_seconds=0)["status"] == "blocked"
    assert reloaded.dispatch(first, timeout_seconds=0)["status"] == "unknown"


def test_process_fence_rejects_the_same_birth_and_dedups_a_new_birth(campaign):
    births = {5: {"pid": 5, "alive": True, "start_ticks": 10}, 6: {"pid": 6, "alive": True, "start_ticks": 99}}
    calls = {"n": 0}

    def restart(info):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"restarted": True, "succeeded": True, "new_pid": 5, "start_ticks": 10, "receipt_id": "native:same-birth"}
        return {"restarted": True, "succeeded": True, "new_pid": 6, "start_ticks": 99, "receipt_id": "native:fenced"}

    adapter, _, _ = bridge(campaign, lifecycle_restart=restart, process_identity=lambda pid: births[pid])
    lane = {"pid": 5, "incident_id": "incident-fence", "action_id": "c" * 64, "policy_id": "wwl-policy", "bundle_key": "wwl"}
    refused = adapter.fenced_lifecycle_restart(lane)
    assert refused["restarted"] is False
    assert refused["reason"] == "previous_process_not_fenced"
    assert calls["n"] == 1
    assert not (adapter.state_root / "control/process-fences").exists()
    fenced = adapter.fenced_lifecycle_restart(lane)
    assert fenced["schema"] == "wwl.watchdog.process-fence@1"
    assert fenced["result"]["new_pid"] == 6
    assert fenced["result"]["old_owner_fenced"] is True
    assert fenced["completion_authority"] is False
    assert calls["n"] == 2
    assert adapter.fenced_lifecycle_restart(lane) == fenced
    assert calls["n"] == 2


def test_current_tree_verification_rejects_counts_and_accepts_bound_proof(campaign):
    adapter, _, snapshot = bridge(campaign)
    claimed = {**snapshot, "completion_verified": True, "completion_receipt": "native:counts-only", "task_id": "",
               "runnable_count": 0, "verified_task_ids": ["WWL-000", "WWL-001"], "effects_unknown": False, "hold": False}
    write(adapter.state_root / "campaign-status.json", claimed)
    observed = adapter.observe({"run_id": "wwl-test", "policy_id": "wwl-policy"})
    assert observed["completion_verified"] is False
    assert observed["completion_receipt"] == ""
    assert observed["current_tree_completion"]["reason"] == "current_tree_completion_proof_missing"
    assert observed["current_tree_completion"]["completion_authority"] is False
    proof = completion_proof("wwl-test", "tree-1", ["WWL-000", "WWL-001"])
    write(adapter.state_root / "control/current-tree-completion.json", proof)
    verified_snapshot = {**claimed, "completion_receipt": proof["completion_receipt"], "completion_verified": False}
    write(adapter.state_root / "campaign-status.json", verified_snapshot)
    verified = adapter.verify({"run_id": "wwl-test", "policy_id": "wwl-policy"})
    assert verified["completion_verified"] is True
    assert verified["completion_receipt"] == proof["completion_receipt"]
    assert verified["completion_authority"] is False
    assert verified["task_retry_authority"] is False
    mismatched = {**proof, "accepted_tree": "tree-2"}
    mismatched["completion_receipt"] = native_identity({key: mismatched[key] for key in mismatched if key != "completion_receipt"})
    write(adapter.state_root / "control/current-tree-completion.json", mismatched)
    rejected = adapter.verify({"run_id": "wwl-test", "policy_id": "wwl-policy"})
    assert rejected["completion_verified"] is False
    assert rejected["reason"] == "current_tree_mismatch"
    active = {**verified_snapshot, "task_id": "WWL-004", "completion_verified": True}
    write(adapter.state_root / "campaign-status.json", active)
    write(adapter.state_root / "control/current-tree-completion.json", proof)
    assert adapter.observe({"run_id": "wwl-test", "policy_id": "wwl-policy"})["completion_verified"] is False


def test_verifier_cannot_mint_completion_authority(campaign):
    def evil(snapshot):
        return {"completion_verified": True, "completion_receipt": "native:forged", "accepted_tree": snapshot.get("accepted_tree"), "completion_authority": True}

    adapter, _, snapshot = bridge(campaign, completion_verifier=evil)
    write(adapter.state_root / "campaign-status.json", {**snapshot, "completion_verified": True, "completion_receipt": "native:forged", "task_id": "", "runnable_count": 0})
    observed = adapter.observe({"run_id": "wwl-test", "policy_id": "wwl-policy"})
    assert observed["completion_verified"] is False
    assert observed["current_tree_completion"]["reason"] == "completion_verifier_refused_authority"


def test_verify_operation_checks_a_terminal_action_receipt(campaign):
    adapter, _, _ = bridge(campaign)
    action_id = "d" * 64
    observation = {"schema": "wwl.watchdog.action-observation@1", "action_id": action_id, "status": "failed", "effects_verified": True, "receipt": "native:fenced-failure"}
    write(adapter.state_root / "control/host-terminal-observations" / (action_id + ".json"), observation)
    verified = adapter.verify({"run_id": "wwl-test", "policy_id": "wwl-policy", "action_id": action_id})
    assert verified["schema"] == "wwl.watchdog.verified-action@1"
    assert verified["status"] == "failed"
    assert verified["effects_verified"] is True
    assert verified["completion_authority"] is False
    assert verified["task_retry_authority"] is False
    write(adapter.state_root / "control/host-terminal-observations" / (action_id + ".json"), {**observation, "effects_verified": False})
    with pytest.raises(WatchdogBridgeError, match="not terminally verified"):
        adapter.verify({"action_id": action_id})


def test_verify_cli_prints_one_completion_object(campaign, monkeypatch, capsys):
    repository, state, _, snapshot = campaign
    # main() uses the wall clock. A fixture epoch of 10000 is stale to that clock.
    now = time.time()
    proof = completion_proof("wwl-test", "tree-1", ["WWL-000", "WWL-001"])
    frontier = {**snapshot, "observed_at": now, "heartbeat_at": now, "task_id": "", "runnable_count": 0, "verified_task_ids": ["WWL-000", "WWL-001"], "effects_unknown": False, "hold": False, "completion_receipt": proof["completion_receipt"]}
    write(state / "campaign-status.json", frontier)
    write(state / "control/current-tree-completion.json", proof)
    request = {"schema": "wwl.watchdog.request@1", "operation": "verify", "run_id": "wwl-test", "policy_id": "wwl-policy"}
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(json.dumps(request).encode())))
    assert main(["--repository", str(repository), "--state-root", str(state), "--run-id", "wwl-test", "verify"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "wwl.watchdog.completion-verification@1"
    assert payload["completion_verified"] is True
    assert payload["completion_authority"] is False
