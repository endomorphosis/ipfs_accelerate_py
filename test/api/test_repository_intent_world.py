"""Real intent/source owners feed bounded evidence without creating authority."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_intent_world as bridge
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import NativeRepositoryIndex
from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
from test.api.test_agent_supervisor_local_planning_admission import scenario
from test.api.test_intent_requirement_observation import _admit, _start
from test.api.test_repository_source_index import scheduler


@pytest.fixture
def intent_case(scenario, tmp_path, scheduler):
    admission, tasks = _admit(scenario)
    with RepositorySourceSession(scenario["repository"], tmp_path / "source-state",
        index_factory=lambda path: NativeRepositoryIndex(path, scheduler=scheduler)) as session:
        yield dict(case=scenario, admission=admission, task_id=tasks["LOCAL-TASK"],
                   session=session, scheduler=scheduler)
    assert scheduler.snapshot()["active_lease_count"] == 0


def capture(case, **changes):
    args = dict(session=case["session"], intent=case["case"]["intent"],
        admission=case["admission"], task_id=case["task_id"], generation=1)
    args.update(changes)
    return bridge.capture_repository_intent_world(**args)


def validate(observation, case, **changes):
    args = dict(session=case["session"], generation=1, task_id=case["task_id"])
    args.update(changes)
    return observation.validate_current(**args)


def test_native_task_world_is_complete_bounded_read_only_and_namespaced(intent_case):
    case = intent_case["case"]
    before = case["intent"].event_watermark()
    original = deepcopy(intent_case["admission"])
    observed = capture(intent_case)
    value = json.loads(observed.reference.summary)
    receipt = validate(observed, intent_case)
    assert observed.reference.required and observed.reference.kind == "repository-intent-world"
    assert receipt["summary_bytes"] == len(observed.reference.summary.encode())
    assert receipt["reference_bytes"] == len(observed.reference.canonical_bytes()) <= 32768
    assert value["intent_binding"]["repository_id"] != value["source_binding"]["datasets_head"]["repository_id"]
    assert receipt["source_repository_id"] == observed.reference.repository_id
    assert receipt["intent_repository_id"] == value["intent_binding"]["repository_id"]
    spec = value["task_spec"]
    assert spec["task_cid"] == intent_case["task_id"]
    assert spec["scope_paths"] and spec["objective"] and spec["validations"]
    assert spec["acceptance"] and spec["outputs"]
    assert "dependency_task_cids" in spec
    assert value["observed_task"]["validations"][0]["status"] == "unobserved"
    assert value["requirements"][0]["measurement_status"] == "unobserved"
    assert value["intent_world"]["component_status"]["repository_tree"] == "unavailable"
    assert value["intent_world"]["component_status"]["task_population"] == "current"
    assert value["intent_world"]["schedulable"] is False
    assert value["intent_world"]["repository_world_projection"] == "unavailable"
    for authority in bridge._AUTHORITY:
        assert value[authority] is False and receipt[authority] is False
    encoded = observed.reference.summary
    for private in (str(case["profile"]), str(case["lifecycle"]), str(case["intent"].database_path)):
        assert private not in encoded
    assert "local_task_contract" not in encoded and "local_observed_validation" not in encoded
    assert case["intent"].event_watermark() == before
    assert intent_case["admission"] == original
    receipt["task_id"] = "mutated-copy"
    assert observed.receipt()["task_id"] == intent_case["task_id"]


def test_actual_signed_failed_check_is_preserved_and_old_owner_state_refused(intent_case):
    first = capture(intent_case)
    case = intent_case["case"]
    _start(case, intent_case["task_id"])
    result = local.run_local_task_validations(intent=case["intent"],
        task_cid=intent_case["task_id"], attempt_id="world-observed-failure")
    assert result["passed"] is False
    with pytest.raises(bridge.RepositoryIntentWorldError, match="observation_changed"):
        validate(first, intent_case)
    current = capture(intent_case)
    value = json.loads(current.reference.summary)
    check = value["observed_task"]["validations"][0]
    assert check["status"] == "failed"
    assert check["evidence_digest"] == result["results"][0]["evidence_digest"]
    assert check["run_id"] and check["result_id"] and check["event_sequence"]
    assert value["requirements"][0]["measurement_status"] == "failed"
    assert first.receipt()["source_tree_sha256"] == current.receipt()["source_tree_sha256"]
    assert current.receipt()["native_event_watermark"] > first.receipt()["native_event_watermark"]


def test_generic_pass_is_not_promoted_to_verified_public_check(intent_case):
    case = intent_case["case"]
    case["intent"].record_validation_result(task_cid=intent_case["task_id"], outcome="passed",
        evidence_digest="generic-pass", argv=["unrelated-command"], attempt_id="generic",
        body={"private_source_dump": "private-test-marker"})
    observed = capture(intent_case)
    value = json.loads(observed.reference.summary)
    assert value["observed_task"]["validations"][0]["status"] == "unobserved"
    assert value["requirements"][0]["public_checks_passed"] is False
    assert "private-test-marker" not in observed.reference.summary


def test_corrupt_signed_result_cannot_reappear_as_older_pass(intent_case):
    case = intent_case["case"]
    _start(case, intent_case["task_id"])
    local.run_local_task_validations(intent=case["intent"], task_cid=intent_case["task_id"],
                                    attempt_id="world-corruption")
    with case["intent"]._connection(write=True) as connection:
        connection.execute("UPDATE validation_results SET outcome='passed' WHERE task_cid=?",
                           [intent_case["task_id"]])
    with pytest.raises(ValueError, match="native evidence links"):
        capture(intent_case)


def test_source_edit_and_refresh_require_a_new_observation(intent_case):
    observed = capture(intent_case)
    (intent_case["case"]["repository"] / "answer.py").write_text("def answer():\n    return 2\n")
    with pytest.raises(bridge.RepositoryIntentWorldError, match="source_not_current"):
        validate(observed, intent_case)
    intent_case["session"].refresh(generation=1)
    with pytest.raises(bridge.RepositoryIntentWorldError):
        validate(observed, intent_case, generation=2)
    successor = capture(intent_case, generation=2)
    assert successor.receipt()["source_tree_sha256"] != observed.receipt()["source_tree_sha256"]


@pytest.mark.parametrize("mutation", ["admission", "owner", "replacement", "consumer"])
def test_issued_observation_refuses_changed_owners_and_consumer(intent_case, mutation):
    observed = capture(intent_case)
    if mutation == "admission":
        intent_case["admission"]["receipt"]["payload"]["graph_cid"] = "changed"
    elif mutation == "owner":
        intent_case["case"]["intent"].owner_id = "another-owner"
    elif mutation == "replacement":
        observed = replace(observed)
    else:
        with pytest.raises(bridge.RepositoryIntentWorldError, match="consumer_binding"):
            validate(observed, intent_case, task_id="LOCAL-TASK")
        return
    with pytest.raises(bridge.RepositoryIntentWorldError):
        validate(observed, intent_case)


def test_alias_and_foreign_workspace_cannot_select_native_task(intent_case, tmp_path):
    with pytest.raises(bridge.RepositoryIntentWorldError, match="exact_native_task"):
        capture(intent_case, task_id="LOCAL-TASK")
    altered = deepcopy(intent_case["admission"])
    altered["manifest"]["payload"]["repository"] = str(tmp_path)
    with pytest.raises(bridge.RepositoryIntentWorldError, match="workspace_mismatch"):
        capture(intent_case, admission=altered)


def test_overflow_refuses_entire_required_reference(intent_case):
    with pytest.raises(bridge.RepositoryIntentWorldError, match="reference_byte_limit"):
        capture(intent_case, max_bytes=1024)
    assert capture(intent_case, max_bytes=65536).reference.required


@pytest.mark.parametrize("maximum", [True, 0, 65537])
def test_invalid_bounds_refused_before_owners(maximum):
    with pytest.raises(bridge.RepositoryIntentWorldError, match="invalid_intent_world_limit"):
        bridge.capture_repository_intent_world(session=None, intent=None, admission={},
            task_id="task", generation=1, max_bytes=maximum)


def test_native_change_between_captures_refuses_mixed_snapshot(intent_case, monkeypatch):
    original = bridge.capture_intent_world_snapshot
    def changed(*args, **kwargs):
        value = original(*args, **kwargs)
        _start(intent_case["case"], intent_case["task_id"])
        return value
    monkeypatch.setattr(bridge, "capture_intent_world_snapshot", changed)
    with pytest.raises(bridge.RepositoryIntentWorldError, match="capture_changed"):
        capture(intent_case)
    assert intent_case["scheduler"].snapshot()["active_lease_count"] == 0


def test_caller_admission_mutation_during_read_cannot_change_checked_projection(intent_case, monkeypatch):
    original = bridge.observe_local_intent_requirements
    caller_admission = intent_case["admission"]
    captured = []
    def changed(*, admission, intent):
        assert admission is not caller_admission
        captured.append(deepcopy(admission))
        result = original(admission=admission, intent=intent)
        caller_admission["graph"]["tasks"][0]["objective"] = "Unchecked replacement objective"
        return result
    monkeypatch.setattr(bridge, "observe_local_intent_requirements", changed)
    with pytest.raises(bridge.RepositoryIntentWorldError, match="owner_or_admission_changed"):
        capture(intent_case)
    assert len(captured) == 2 and captured[0] == captured[1]
    assert captured[0]["graph"]["tasks"][0]["objective"] != "Unchecked replacement objective"
    assert intent_case["scheduler"].snapshot()["active_lease_count"] == 0


@pytest.mark.parametrize("mode", ["denied", "cancelled", "deadline"])
def test_resource_gate_refuses_before_native_capture(intent_case, monkeypatch, mode):
    calls = []
    def should_not_capture(**kwargs):
        calls.append(kwargs)
        raise AssertionError("native capture was not resource-admitted")
    monkeypatch.setattr(bridge, "observe_local_intent_requirements", should_not_capture)
    monkeypatch.setattr(bridge, "capture_intent_world_snapshot", should_not_capture)
    @contextmanager
    def resources(*args, **kwargs):
        if mode == "denied":
            raise RuntimeError("resource reservation denied")
        yield SimpleNamespace(cancelled=True)
    if mode == "deadline":
        readings = iter((0.0, 31.0))
        monkeypatch.setattr(bridge.time, "monotonic", lambda: next(readings))
    else:
        monkeypatch.setattr(bridge, "_resources", resources)
    with pytest.raises((bridge.RepositoryIntentWorldError, RuntimeError)):
        capture(intent_case)
    assert not calls
    # Restore monotonic before consulting the scheduler's own resource clock.
    monkeypatch.undo()
    assert intent_case["scheduler"].snapshot()["active_lease_count"] == 0


def test_cancellation_during_world_capture_releases_reservation(intent_case, monkeypatch):
    live = []
    reservations = []
    @contextmanager
    def resources(*args, **kwargs):
        lease = SimpleNamespace(cancelled=False)
        reservations.append(lease)
        live.append(lease)
        try:
            yield lease
        finally:
            live.remove(lease)
    original = bridge.capture_intent_world_snapshot
    def cancelled(*args, **kwargs):
        value = original(*args, **kwargs)
        live[-1].cancelled = True
        return value
    monkeypatch.setattr(bridge, "_resources", resources)
    monkeypatch.setattr(bridge, "capture_intent_world_snapshot", cancelled)
    with pytest.raises(bridge.RepositoryIntentWorldError, match="capture_cancelled"):
        capture(intent_case)
    assert len(reservations) == 2 and not live
    assert intent_case["scheduler"].snapshot()["active_lease_count"] == 0
