"""Disjoint companion-only wiring controls with explicit native boundaries doubled.

The existing worker fixture supplies genuine persisted metadata, typed planning
records and deterministic previews. Source observations, leases and companion
signing remain explicit protocol doubles; no model, native owner, scheduler,
repository check or worker runs. Separate genuine companion verification is
outside this preparation API and outside this control module's qualification.
"""
from copy import deepcopy
from dataclasses import replace
import inspect
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import preview_successor_native_profile as adapter
from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import freeze_plan_create_input_snapshot
from ipfs_accelerate_py.agent_supervisor.runtime import (
    codebase_preview_successor_admission as companion,
    codebase_preview_successor_contract as contract,
    codebase_preview_successor_worker_admission as worker,
    local_planning_admission as local,
)
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    LeaseCancelledError, LeaseTimeoutError,
)
from test.api.test_preview_successor_native_profile import ControlledLease, _rewrite, owner_case


@pytest.fixture
def companion_case(owner_case, monkeypatch):
    """Refuse every worker route and any implicit companion receiving operation."""
    forbidden = []
    for owner, name in (
            (companion, "verify_current_preview_successor_bridge"),
            *((worker, name) for name in (
                "author_current_preview_successor_worker_manifest",
                "admit_current_preview_successor_worker_plan",
                "verify_current_preview_successor_worker_admission",
                "materialize_current_preview_successor_worker_plan",
                "prepare_current_preview_successor_worker_context",
                "reserve_current_preview_successor_execution"))):
        def refuse(*args, _name=name, **kwargs):
            forbidden.append(_name)
            raise AssertionError("companion preparation crossed " + _name)

        monkeypatch.setattr(owner, name, refuse)
    owner_case["forbidden"] = forbidden
    return owner_case


def _prepare(case, **changes):
    return adapter.prepare_native_preview_successor_companion(**{
        "parent_admission": case["parent"], "graph": case["graph"], "index": case["index"],
        "registry": object(), "selection": case["selection"], "root": case["root"],
        "completion": case["completion"], "repository": case["repository"],
        "owner_state_dir": case["directory"], "scheduler": case["scheduler"],
        "timeout_seconds": 90, "admission_timeout_seconds": 30, "memory_mb": 1024,
        **changes})


def test_companion_entrypoint_preserves_the_existing_worker_keyword_signature():
    expected = ("(*, parent_admission, graph, index, registry, selection, root, completion, "
        "repository, owner_state_dir, scheduler=None, parent_lease=None, cancel_event=None, "
        "timeout_seconds=30.0, admission_timeout_seconds=30.0, memory_mb=1024)")
    assert str(inspect.signature(adapter.prepare_native_preview_successor)) == expected
    assert inspect.signature(adapter.prepare_native_preview_successor_companion) == inspect.signature(
        adapter.prepare_native_preview_successor)
    assert adapter.SCHEMA == "native-preview-successor-owner-profile@1"


def test_companion_preparation_authors_only_the_companion_without_worker_credit(companion_case):
    result = _prepare(companion_case)
    assert [name for name, _ in companion_case["state"]["dispatch"]] == ["bridge"]
    assert not companion_case["forbidden"]
    assert set(result) == {"bridge_inputs", "request", "materials", "preview", "bridge",
                           "checker_policy_cid", "records"}
    records = result["records"]
    assert records["schema"] == adapter.COMPANION_SCHEMA == "native-preview-successor-companion-owner-profile@1"
    assert records["worker_profile_integrated"] is False
    assert {"worker_manifest_cid", "worker_admission_cid", "codebase_preview_successor_context_cid"}.isdisjoint(records)
    assert records["proof_authority"] is records["production_admitted"] is records["worker_launched_here"] is False
    assert records["model_calls"] == records["training_steps"] == 0
    assert records["orchestration_lease_released"] is True
    assert companion_case["scheduler"].leases[0].release_count == 1


def test_companion_retains_the_complete_parent_and_explicit_task_acceptance_mappings(companion_case):
    parent = deepcopy(companion_case["parent"])
    result = _prepare(companion_case)
    arguments = companion_case["state"]["dispatch"][0][1]
    tasks = {task.task_key: task for task in companion_case["graph"].tasks}
    assert arguments["admission"] == parent == companion_case["parent"]
    assert [row["task_cid"] for row in arguments["candidate_bindings"]] == [
        tasks[name].task_cid for name in ("SUCCESSOR-TYPE", "SUCCESSOR-FORMAT")]
    for row in arguments["requirement_bindings"]:
        assert len(row["task_cids"]) == 1
        task = next(task for task in tasks.values() if task.task_cid == row["task_cids"][0])
        assert row["acceptance_cids"] == sorted(item.content_id for item in task.acceptance)
    assert result["records"]["administrator_task_cids"] == sorted(task.task_cid for task in tasks.values())
    assert result["records"]["population"] == {"administrator_tasks": 2}


def test_companion_keeps_original_typed_private_inputs_for_separate_receiving(companion_case):
    result = _prepare(companion_case)
    arguments = companion_case["state"]["dispatch"][0][1]
    inputs = result["bridge_inputs"]
    assert type(inputs) is worker.PreviewSuccessorBridgeInputs
    for name in ("request", "materials", "preview"):
        assert getattr(inputs, name) is result[name] is arguments[name]
    for name in ("roots_observer", "checker_observer"):
        assert getattr(inputs, name) is arguments[name]
    assert result["materials"].current_roots is None
    assert not result["materials"].current_facts and result["materials"].model_provider is None
    assert result["records"]["execution_queue_root_observed"] is False
    assert not companion_case["forbidden"]


@pytest.mark.parametrize("entries,expected", ((64, 256), (512, 512)))
def test_companion_capture_budget_binds_source_preview_and_signing(companion_case, entries, expected):
    companion_case["captured"].max_entries = entries
    result = _prepare(companion_case)
    records = result["records"]
    assert records["capture_limits"] == {"max_entries": entries, "max_file_bytes": 65536}
    assert records["preview_memory_mb"] == records["preview_owner_memory_mb"] == expected
    assert records["preview_lease"]["memory_mb"] == expected
    assert records["signed_memory_mb"] == companion_case["state"]["dispatch"][0][1]["memory_mb"] == 1024
    assert all(row["controls"]["memory_mb"] == expected for row in companion_case["state"]["calls"])
    assert all(row["memory_mb"] == expected for row in records["owner_observations"] if row["kind"] == "roots")
    assert companion_case["scheduler"].leases[0].released


@pytest.mark.parametrize("memory", (True, 1024.0, 512, 4097))
def test_companion_public_resource_validation_precedes_index_or_owner_work(companion_case, memory):
    def unexpected_load(*unused):
        raise AssertionError("invalid companion envelope reached index loading")
    companion_case["index"].load = unexpected_load
    with pytest.raises(ValueError, match="bounded native preview successor resources"):
        _prepare(companion_case, memory_mb=memory)
    assert not companion_case["directory"].exists() and not companion_case["scheduler"].leases
    assert not companion_case["state"]["calls"] and not companion_case["state"]["dispatch"]


@pytest.mark.parametrize("entries,memory", ((2048, 1024), (4097, 4096)))
def test_companion_insufficient_capture_reservation_refuses_before_side_effects(companion_case, entries, memory):
    companion_case["captured"].max_entries = entries
    with pytest.raises(ValueError, match="captured scan exceeds"):
        _prepare(companion_case, memory_mb=memory)
    assert not companion_case["directory"].exists() and not companion_case["scheduler"].leases
    assert not companion_case["state"]["calls"] and not companion_case["state"]["dispatch"]


def test_companion_preview_child_releases_without_releasing_its_supplied_parent(companion_case):
    companion_case["captured"].max_entries = 512
    parent = ControlledLease(companion_case["scheduler"].leases, {"memory_mb": 1024})
    result = _prepare(companion_case, scheduler=None, parent_lease=parent)
    child = companion_case["scheduler"].leases[0]
    assert child.arguments["memory_mb"] == result["records"]["preview_memory_mb"] == 512
    assert child.released and child.release_count == 1
    assert not parent.released and parent.release_count == 0
    parents = [row["controls"]["parent_lease"] for row in companion_case["state"]["calls"]]
    assert parent in parents and child in parents
    assert companion_case["state"]["dispatch"][0][1]["parent_lease"] is parent


def test_companion_precancellation_opens_no_owner_or_resource(companion_case):
    with pytest.raises(LeaseCancelledError, match="native preview successor cancelled"):
        _prepare(companion_case, cancel_event=SimpleNamespace(is_set=lambda: True))
    assert not companion_case["directory"].exists() and not companion_case["scheduler"].leases
    assert not companion_case["state"]["calls"] and not companion_case["state"]["dispatch"]


def test_companion_return_checks_cancellation_after_authoring(companion_case, monkeypatch):
    cancelled = {"value": False}
    author = companion.author_current_preview_successor_bridge

    def cancel_after_author(**arguments):
        bridge = author(**arguments)
        cancelled["value"] = True
        return bridge

    monkeypatch.setattr(companion, "author_current_preview_successor_bridge", cancel_after_author)
    with pytest.raises(LeaseCancelledError, match="native preview successor cancelled"):
        _prepare(companion_case, cancel_event=SimpleNamespace(is_set=lambda: cancelled["value"]))
    assert [name for name, _ in companion_case["state"]["dispatch"]] == ["bridge"]
    assert companion_case["scheduler"].leases[0].released
    assert not companion_case["forbidden"]


def test_companion_return_checks_the_original_deadline_after_authoring(companion_case, monkeypatch):
    author = companion.author_current_preview_successor_bridge

    def expire_after_author(**arguments):
        bridge = author(**arguments)
        companion_case["state"]["clock"] = 101.0
        return bridge

    monkeypatch.setattr(companion, "author_current_preview_successor_bridge", expire_after_author)
    with pytest.raises(LeaseTimeoutError, match="preparation deadline exceeded"):
        _prepare(companion_case)
    assert [name for name, _ in companion_case["state"]["dispatch"]] == ["bridge"]
    assert companion_case["scheduler"].leases[0].release_count == 1
    assert not companion_case["forbidden"]


@pytest.mark.parametrize("late_failure", ("cancel", "deadline"))
def test_companion_final_return_guard_covers_parent_serialization(companion_case, monkeypatch, late_failure):
    cancelled = {"value": False}
    plain = local._plain
    parent_reads = []

    def parent_guard_changes_budget(value):
        if value is companion_case["parent"]:
            parent_reads.append(value)
            if len(parent_reads) == 2:
                if late_failure == "cancel":
                    cancelled["value"] = True
                else:
                    companion_case["state"]["clock"] = 101.0
        return plain(value)

    monkeypatch.setattr(local, "_plain", parent_guard_changes_budget)
    expected = LeaseCancelledError if late_failure == "cancel" else LeaseTimeoutError
    with pytest.raises(expected):
        _prepare(companion_case, cancel_event=SimpleNamespace(is_set=lambda: cancelled["value"]))
    assert len(parent_reads) == 2
    assert [name for name, _ in companion_case["state"]["dispatch"]] == ["bridge"]
    assert companion_case["scheduler"].leases[0].release_count == 1
    assert not companion_case["forbidden"]


def test_companion_cancelled_inside_preview_releases_derived_lease(companion_case):
    companion_case["captured"].max_entries = 512
    cancelled = {"value": False}

    def cancel_after_acquisition():
        if companion_case["scheduler"].leases:
            cancelled["value"] = True

    companion_case["state"]["source_enter_callback"] = cancel_after_acquisition
    with pytest.raises(LeaseCancelledError):
        _prepare(companion_case, cancel_event=SimpleNamespace(is_set=lambda: cancelled["value"]))
    child = companion_case["scheduler"].leases[0]
    assert child.arguments["memory_mb"] == 512 and child.released and child.release_count == 1
    assert not companion_case["state"]["dispatch"]


@pytest.mark.parametrize("field", ("preview_error", "dispatch_error"))
def test_companion_preview_and_author_failures_release_the_preview_lease(companion_case, field):
    companion_case["state"][field] = RuntimeError("explicit companion operation failure")
    with pytest.raises(RuntimeError, match="companion operation failure"):
        _prepare(companion_case)
    assert len(companion_case["scheduler"].leases) == 1
    assert companion_case["scheduler"].leases[0].release_count == 1
    assert companion_case["scheduler"].leases[0].released and not companion_case["forbidden"]


def test_companion_return_rejects_mutation_of_the_original_parent(companion_case, monkeypatch):
    author = companion.author_current_preview_successor_bridge

    def mutate_original_parent(**arguments):
        bridge = author(**arguments)
        companion_case["parent"]["late_owner_mutation"] = True
        return bridge

    monkeypatch.setattr(companion, "author_current_preview_successor_bridge", mutate_original_parent)
    with pytest.raises(ValueError, match="original @6 parent admission changed"):
        _prepare(companion_case)
    assert companion_case["scheduler"].leases[0].release_count == 1
    assert not companion_case["forbidden"]


@pytest.mark.parametrize("kind", ("roots", "checker"))
def test_companion_observers_refuse_another_request_before_side_effects(companion_case, kind):
    result = _prepare(companion_case)
    changed = replace(result["request"], caller="principal:another-companion-owner")
    before = (len(companion_case["state"]["calls"]), len(result["records"]["owner_observations"]))
    with pytest.raises(ValueError, match="observer received another request"):
        getattr(result["bridge_inputs"], kind + "_observer")(changed)
    assert before == (len(companion_case["state"]["calls"]), len(result["records"]["owner_observations"]))


def test_companion_later_roots_use_fresh_reservations_after_preparation_expires(companion_case):
    companion_case["captured"].max_entries = 512
    result = _prepare(companion_case)
    released = companion_case["scheduler"].leases[0]
    companion_case["state"]["clock"] = 1000.0
    before = len(companion_case["state"]["calls"])
    assert result["bridge_inputs"].roots_observer(result["request"]) == result["request"].roots
    controls = companion_case["state"]["calls"][before]["controls"]
    assert controls["scheduler"] is companion_case["scheduler"] and controls["parent_lease"] is None
    assert controls["timeout_seconds"] == 90 and controls["admission_timeout_seconds"] == 30
    assert controls["memory_mb"] == 512 and released.release_count == 1


@pytest.mark.parametrize("name", ("check_type.py", "check_offset.py"))
def test_companion_later_checker_observation_refuses_public_script_drift(companion_case, name):
    result = _prepare(companion_case)
    (companion_case["repository"] / name).write_bytes(b"# changed, never executed\n")
    with pytest.raises(ValueError, match="validation policy, scripts or checker implementation changed"):
        result["bridge_inputs"].checker_observer(result["request"])
    assert result["records"]["worker_profile_integrated"] is False


@pytest.mark.parametrize("phase", ("source_enter_callback", "source_exit_callback"))
def test_companion_roots_refuse_owner_metadata_drift_during_native_observation(companion_case, phase):
    result = _prepare(companion_case)
    companion_case["state"][phase] = lambda: _rewrite(companion_case, "usage-policy.json",
        lambda value: value.__setitem__("training_steps", 1))
    with pytest.raises(ValueError, match="owner roots changed during native source"):
        result["bridge_inputs"].roots_observer(result["request"])
    assert not companion_case["forbidden"]


def test_companion_private_material_changes_alter_the_complete_snapshot(companion_case):
    result = _prepare(companion_case)
    materials = result["materials"]
    bound = contract.bind_repository_preview_materials(materials, companion_case["state"]["structural"])
    assert freeze_plan_create_input_snapshot(result["request"], materials=bound).snapshot_cid == result["preview"].input_snapshot_cid
    materials.extra["late_companion_material_change"] = True
    changed = contract.bind_repository_preview_materials(materials, companion_case["state"]["structural"])
    assert freeze_plan_create_input_snapshot(result["request"], materials=changed).snapshot_cid != result["preview"].input_snapshot_cid
    assert result["records"]["worker_profile_integrated"] is False
    assert not companion_case["forbidden"]
