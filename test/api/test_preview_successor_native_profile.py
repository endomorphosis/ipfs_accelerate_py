"""Offline owner-adapter controls; every native/resource boundary is doubled.

Persisted metadata, typed requests/materials, deterministic proposal stages and
source/checker digests are genuine. The index, structural observations, leases,
signature verification and bridge/admission APIs are explicit protocol doubles.
These controls open no native owners, run no repository checks, fit no model,
execute no inference, launch no workers and qualify no native head or freshness.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import (
    preview_successor_native_profile as adapter,
    qualify_codebase_signed_successor as benchmark,
)
from ipfs_accelerate_py.agent_supervisor.planning import (
    repository_plan_preview as native_preview,
    structural_codebase_context as structural_module,
)
from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
    PlanCreateService, freeze_plan_create_input_snapshot,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import (
    codebase_preview_successor_admission as companion,
    codebase_preview_successor_contract as contract,
    codebase_preview_successor_worker_admission as worker,
    local_planning_admission as local,
)
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError


def _cid(label):
    return content_identity({"offline-native-profile-control": label})


class ControlledLease:
    """Accounting assertions only; this is not a scheduler reservation."""

    def __init__(self, records, arguments):
        self.records = records
        self.arguments = arguments
        self.released = False
        self.release_count = 0

    def __enter__(self):
        assert not self.released
        return self

    def __exit__(self, *unused):
        self.release_count += 1
        self.released = True

    def combined_cancellation_signal(self, value):
        return value

    def acquire_child(self, **arguments):
        assert not self.released
        child = ControlledLease(self.records, arguments)
        self.records.append(child)
        return child

    def to_dict(self):
        return {"lease_id": "explicit-offline-profile-lease", "lane": self.arguments["lane"],
            "cpu_slots": self.arguments["cpu_slots"], "memory_mb": self.arguments["memory_mb"],
            "child_process_slots": self.arguments["child_process_slots"],
            "released": self.released, "cancelled": False}


class ControlledScheduler:
    def __init__(self):
        self.leases = []

    def acquire(self, **arguments):
        lease = ControlledLease(self.leases, arguments)
        self.leases.append(lease)
        return lease


@pytest.fixture
def owner_case(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    for name in ("calc.py", "check_type.py", "check_offset.py"):
        (repository / name).write_bytes(b"# offline adapter control, never executed\n")
    (repository / "README.md").write_text("Offline source/checker owner controls\n")
    head = CodebaseHead(repository_id="repository:offline-native-profile", generation=2,
        manifest_cid=_cid("manifest"), snapshot_cid=_cid("snapshot"),
        ast_revision_id=f"rev:repository:offline-native-profile:snapshot:{_cid('snapshot')}",
        receipt_cid=_cid("publication"))
    structural = structural_module.StructuralCodebaseContext(head, _cid("semantic-state"), {
        "inventory_entries": 4, "captured_entries": 4, "ast_ok": 0, "ast_partial": 0,
        "ast_failed": 0, "opaque_entries": 0, "unindexed_entries": 4,
        "semantic_symbols": 0, "formalized_properties": 0, "checked_properties": 0})
    captured = SimpleNamespace(max_entries=64, max_file_bytes=65536)
    manifest = SimpleNamespace(snapshot=captured,
        semantic_state=SimpleNamespace(state_cid=structural.semantic_state_cid))
    index = SimpleNamespace(current=lambda identity: head, load=lambda cid: manifest)
    completion = SimpleNamespace(artifact_cid=_cid("completion"))
    graph, _, _ = benchmark._native_graph(index, head, completion)
    selection = SimpleNamespace(artifact_cid=_cid("selection"), to_dict=lambda: {"current_head": head.to_dict()})
    root = SimpleNamespace(artifact_cid=_cid("scan-root"))
    parent = {"schema": "explicit-offline-parent-double@1", "population": graph.to_dict()}
    state = {"structural": structural, "calls": [], "preview_calls": [], "dispatch": [],
        "source_enter_callback": None, "source_exit_callback": None,
        "preview_error": None, "dispatch_error": None, "clock": 10.0}
    scheduler = ControlledScheduler()
    monkeypatch.setattr(adapter, "time", SimpleNamespace(monotonic=lambda: state["clock"]))
    monkeypatch.setattr(local, "verify_local_benchmark_admission", lambda value: {
        "manifest": {"schema": local.SUCCESSOR_MANIFEST_SCHEMA}, "graph": graph})

    @contextmanager
    def structural_double(received_index, received_repository, **controls):
        assert received_index is index and received_repository == repository
        assert controls["repository_id"] == head.repository_id and controls["expected_head"] == head
        if controls["cancel_event"] is not None and controls["cancel_event"].is_set():
            raise LeaseCancelledError("explicit structural observation double cancelled")
        parent_lease = controls["parent_lease"]
        assert parent_lease is None or not parent_lease.released
        # The observation remains doubled; exercise the real scan reservation
        # validator that the native source owner applies to its loaded capture.
        CodebaseScanLimits(captured.max_entries, captured.max_file_bytes).validate_reservation(
            controls["memory_mb"])
        state["calls"].append({"controls": controls, "phase": "enter"})
        if state["source_enter_callback"] is not None:
            state["source_enter_callback"]()
        try:
            yield state["structural"]
        finally:
            state["calls"].append({"controls": controls, "phase": "exit"})
            if state["source_exit_callback"] is not None:
                state["source_exit_callback"]()

    def preview_double(*, owner, request, materials, policy_observer):
        state["preview_calls"].append((owner, request, materials))
        assert owner.parent_lease is scheduler.leases[0] and not owner.parent_lease.released
        policy_observer(request)
        if state["preview_error"] is not None:
            raise state["preview_error"]
        bound = contract.bind_repository_preview_materials(materials, state["structural"])
        receipt = PlanCreateService(root_observer=policy_observer,
            require_live_root_observation=True).preview_create(request, materials=bound)
        return {"preview": receipt.to_dict(), "structural_context_cid": state["structural"].cid,
            "observed_facts_supplied": 0, "model_calls": 0, "proof_authority": False,
            "native_observation_doubled": True}

    def bridge_double(**arguments):
        state["dispatch"].append(("bridge", arguments))
        arguments["roots_observer"](arguments["request"])
        arguments["checker_observer"](arguments["request"])
        if state["dispatch_error"] is not None:
            raise state["dispatch_error"]
        return {"schema": "explicit-offline-bridge-double@1", "payload": {
            "bridge_cid": _cid("controlled-bridge"), "population": {"administrator_tasks": 2}}}

    def manifest_double(**arguments):
        state["dispatch"].append(("manifest", arguments))
        return {"schema": "explicit-offline-manifest-double@1", "payload": {"parent": parent}}

    def admission_double(**arguments):
        state["dispatch"].append(("admission", arguments))
        return {"schema": "explicit-offline-admission-double@1", "population": graph.to_dict()}

    monkeypatch.setattr(structural_module, "structural_codebase_context", structural_double)
    monkeypatch.setattr(native_preview, "RepositoryPlanPreviewOwner", lambda **values: SimpleNamespace(**values))
    monkeypatch.setattr(native_preview, "preview_repository_plan", preview_double)
    monkeypatch.setattr(companion, "author_current_preview_successor_bridge", bridge_double)
    monkeypatch.setattr(worker, "author_current_preview_successor_worker_manifest", manifest_double)
    monkeypatch.setattr(worker, "admit_current_preview_successor_worker_plan", admission_double)
    return {"repository": repository, "head": head, "index": index, "graph": graph,
        "captured": captured, "native_manifest": manifest,
        "selection": selection, "root": root, "completion": completion, "parent": parent,
        "state": state, "scheduler": scheduler, "directory": tmp_path / "private-owner"}


def _prepare(case, **changes):
    return adapter.prepare_native_preview_successor(**{
        "parent_admission": case["parent"], "graph": case["graph"], "index": case["index"],
        "registry": object(), "selection": case["selection"], "root": case["root"],
        "completion": case["completion"], "repository": case["repository"],
        "owner_state_dir": case["directory"], "scheduler": case["scheduler"],
        "timeout_seconds": 90, "admission_timeout_seconds": 30, "memory_mb": 1024,
        **changes})


def _rewrite(case, name, mutate):
    path = case["directory"] / name
    value = json.loads(path.read_bytes())
    mutate(value)
    path.chmod(0o600)
    path.write_bytes(adapter._wire(value))
    path.chmod(0o444)


def test_adapter_keeps_full_native_graph_and_every_explicit_mapping(owner_case):
    result = _prepare(owner_case)
    state = owner_case["state"]
    assert [name for name, _ in state["dispatch"]] == ["bridge", "manifest", "admission"]
    arguments = state["dispatch"][0][1]
    tasks = {task.task_key: task for task in owner_case["graph"].tasks}
    assert [row["task_cid"] for row in arguments["candidate_bindings"]] == [
        tasks[name].task_cid for name in ("SUCCESSOR-TYPE", "SUCCESSOR-FORMAT")]
    assert sorted(row["task_cids"][0] for row in arguments["requirement_bindings"]) == sorted(
        task.task_cid for task in tasks.values())
    for row in arguments["requirement_bindings"]:
        task = next(task for task in tasks.values() if task.task_cid == row["task_cids"][0])
        assert row["acceptance_cids"] == sorted(item.content_id for item in task.acceptance)
    assert state["dispatch"][2][1]["graph"] is owner_case["graph"]
    assert state["dispatch"][1][1]["parent_admission"] == owner_case["parent"]
    assert result["records"]["administrator_task_cids"] == sorted(task.task_cid for task in tasks.values())
    assert result["records"]["orchestration_lease_released"] is True
    assert owner_case["scheduler"].leases[0].release_count == 1


@pytest.mark.parametrize("entries,file_bytes,expected", (
    (64, 65536, 256), (512, 65536, 512), (256, 65537, 257), (4096, 65536, 4096)))
def test_memory_derivation_obeys_captured_limits_and_integer_ceiling(entries, file_bytes, expected):
    captured = SimpleNamespace(max_entries=entries, max_file_bytes=file_bytes)
    assert adapter.native_preview_memory_mb(captured, memory_mb=4096) == expected
    CodebaseScanLimits(entries, file_bytes).validate_reservation(expected)


@pytest.mark.parametrize("field,value", (
    ("max_entries", True), ("max_entries", 0), ("max_entries", -1),
    ("max_entries", 512.0), ("max_entries", "512"),
    ("max_file_bytes", False), ("max_file_bytes", 0), ("max_file_bytes", -1),
    ("max_file_bytes", 65536.0), ("max_file_bytes", None)))
def test_malformed_capture_limits_refuse_before_owner_artifacts_or_resources(owner_case, field, value):
    setattr(owner_case["captured"], field, value)
    with pytest.raises(ValueError, match=field + " must be a positive exact integer"):
        _prepare(owner_case)
    assert not owner_case["directory"].exists()
    assert not owner_case["state"]["calls"] and not owner_case["state"]["preview_calls"]
    assert not owner_case["state"]["dispatch"] and not owner_case["scheduler"].leases


@pytest.mark.parametrize("captured", (None, SimpleNamespace(max_entries=64), {}))
def test_memory_derivation_refuses_missing_capture_limits(captured):
    with pytest.raises(ValueError, match="captured native scan limits required"):
        adapter.native_preview_memory_mb(captured, memory_mb=1024)


@pytest.mark.parametrize("memory", (True, 256.0, 255, 4097))
def test_memory_derivation_requires_a_bounded_exact_caller_envelope(memory):
    with pytest.raises(ValueError, match="bounded exact native preview memory envelope"):
        adapter.native_preview_memory_mb(SimpleNamespace(max_entries=64, max_file_bytes=65536),
                                        memory_mb=memory)


def test_memory_derivation_refuses_one_mib_below_the_required_capture():
    with pytest.raises(ValueError, match="captured scan exceeds"):
        adapter.native_preview_memory_mb(SimpleNamespace(max_entries=512, max_file_bytes=65536),
                                        memory_mb=511)


@pytest.mark.parametrize("entries,memory", ((2048, 1024), (4097, 4096)))
def test_insufficient_capture_budget_refuses_before_native_preview_operations(owner_case, entries, memory):
    owner_case["captured"].max_entries = entries
    with pytest.raises(ValueError, match="captured scan exceeds"):
        _prepare(owner_case, memory_mb=memory)
    assert not owner_case["directory"].exists()
    assert not owner_case["state"]["calls"] and not owner_case["state"]["preview_calls"]
    assert not owner_case["state"]["dispatch"] and not owner_case["scheduler"].leases


@pytest.mark.parametrize("memory", (True, 1024.0, 512, 4097))
def test_public_caller_validation_remains_before_index_loading(owner_case, memory):
    def unexpected_load(*unused):
        raise AssertionError("invalid caller envelope reached index loading")
    owner_case["index"].load = unexpected_load
    with pytest.raises(ValueError, match="bounded native preview successor resources"):
        _prepare(owner_case, memory_mb=memory)
    assert not owner_case["directory"].exists() and not owner_case["scheduler"].leases


@pytest.mark.parametrize("entries,expected", ((64, 256), (512, 512)))
def test_capture_reservation_binds_every_source_preview_and_signed_boundary(owner_case, entries, expected):
    owner_case["captured"].max_entries = entries
    result = _prepare(owner_case)
    records = result["records"]
    assert records["capture_limits"] == {"max_entries": entries, "max_file_bytes": 65536}
    assert records["preview_memory_mb"] == records["preview_owner_memory_mb"] == expected
    assert records["signed_memory_mb"] == 1024
    assert records["preview_lease"]["memory_mb"] == expected
    assert records["preview_lease"]["released"] is records["preview_lease"]["cancelled"] is False
    assert owner_case["scheduler"].leases[0].arguments["memory_mb"] == expected
    assert all(owner.memory_mb == expected for owner, _, _ in owner_case["state"]["preview_calls"])
    assert all(row["controls"]["memory_mb"] == expected for row in owner_case["state"]["calls"])
    assert all(row["memory_mb"] == expected for row in records["owner_observations"] if row["kind"] == "roots")
    assert all(arguments["memory_mb"] == 1024 for _, arguments in owner_case["state"]["dispatch"])
    count = len(owner_case["state"]["calls"])
    result["bridge_inputs"].roots_observer(result["request"])
    assert owner_case["state"]["calls"][count]["controls"]["memory_mb"] == expected
    assert records["orchestration_lease_released"] and owner_case["scheduler"].leases[0].release_count == 1


def test_later_capture_growth_is_refused_by_native_scan_reservation_validation(owner_case):
    owner_case["captured"].max_entries = 512
    result = _prepare(owner_case)
    owner_case["captured"].max_entries = 1024
    before = len(owner_case["state"]["calls"])
    with pytest.raises(ValueError, match="source bounds exceed the preparation memory envelope"):
        result["bridge_inputs"].roots_observer(result["request"])
    assert result["records"]["capture_limits"]["max_entries"] == 512
    assert result["records"]["preview_memory_mb"] == 512
    assert len(owner_case["state"]["calls"]) == before
    assert owner_case["scheduler"].leases[0].release_count == 1


def test_derived_preview_child_releases_without_releasing_its_supplied_parent(owner_case):
    owner_case["captured"].max_entries = 512
    parent = ControlledLease(owner_case["scheduler"].leases, {"memory_mb": 1024})
    result = _prepare(owner_case, scheduler=None, parent_lease=parent)
    child = owner_case["scheduler"].leases[0]
    assert child.arguments["memory_mb"] == result["records"]["preview_memory_mb"] == 512
    assert child.released and child.release_count == 1
    assert not parent.released and parent.release_count == 0
    source_parents = [row["controls"]["parent_lease"] for row in owner_case["state"]["calls"]]
    assert parent in source_parents and child in source_parents


def test_cancelled_large_capture_refuses_before_owner_metadata_or_resource_calls(owner_case):
    owner_case["captured"].max_entries = 512
    cancellation = SimpleNamespace(is_set=lambda: True)
    with pytest.raises(LeaseCancelledError, match="native preview successor cancelled"):
        _prepare(owner_case, cancel_event=cancellation)
    assert not owner_case["directory"].exists()
    assert not owner_case["scheduler"].leases and not owner_case["state"]["calls"]


def test_cancelled_preview_releases_derived_large_capture_lease_exactly_once(owner_case):
    owner_case["captured"].max_entries = 512
    cancelled = {"value": False}
    cancellation = SimpleNamespace(is_set=lambda: cancelled["value"])

    def cancel_after_preview_lease_enters():
        if owner_case["scheduler"].leases:
            cancelled["value"] = True

    owner_case["state"]["source_enter_callback"] = cancel_after_preview_lease_enters
    with pytest.raises(LeaseCancelledError):
        _prepare(owner_case, cancel_event=cancellation)
    lease = owner_case["scheduler"].leases[0]
    assert lease.arguments["memory_mb"] == 512 and lease.released and lease.release_count == 1
    assert all(row["controls"]["memory_mb"] == 512 for row in owner_case["state"]["calls"])
    assert not owner_case["state"]["dispatch"]


@pytest.mark.parametrize("name,field", (("planning-policy.json", "checker_policy_cid"),
    ("capability-catalog.json", "checker_policy_cid"), ("configuration.json", "root_cid")))
def test_late_canonical_persisted_owner_changes_produce_fresh_distinct_roots(owner_case, name, field):
    result = _prepare(owner_case)
    inputs = result["bridge_inputs"]
    before = inputs.roots_observer(inputs.request)
    _rewrite(owner_case, name, lambda value: value.__setitem__(field, _cid("late-change")))
    after = inputs.roots_observer(inputs.request)
    assert before != after
    assert after != inputs.request.roots
    assert after.repository_root_cid == owner_case["head"].snapshot_cid


@pytest.mark.parametrize("mutation", ("unknown_field", "schema", "duplicate", "noncanonical", "nonfinite"))
def test_late_malformed_persisted_metadata_refuses_before_native_source_observation(owner_case, mutation):
    result = _prepare(owner_case)
    path = owner_case["directory"] / "usage-policy.json"
    raw = path.read_bytes()
    value = json.loads(raw)
    if mutation == "unknown_field":
        value["authority"] = True
        raw = adapter._wire(value)
    elif mutation == "schema":
        value["schema"] = "unknown@2"
        raw = adapter._wire(value)
    elif mutation == "duplicate":
        raw = raw[:-1] + b',"model_calls":0}'
    elif mutation == "noncanonical":
        raw = json.dumps(value, indent=2).encode()
    else:
        raw = raw.replace(b'"model_calls":0', b'"model_calls":NaN')
    path.chmod(0o600)
    path.write_bytes(raw)
    count = len(owner_case["state"]["calls"])
    with pytest.raises(ValueError):
        result["bridge_inputs"].roots_observer(result["request"])
    assert len(owner_case["state"]["calls"]) == count


def test_late_authored_markdown_change_requires_its_matching_current_metadata(owner_case):
    result = _prepare(owner_case)
    path = owner_case["directory"] / "TASKS.md"
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b"Additional authored requirement\n")
    with pytest.raises(ValueError, match="planning source differs"):
        result["bridge_inputs"].roots_observer(result["request"])
    _rewrite(owner_case, "task-source.json", lambda value: value.__setitem__(
        "source_sha256", hashlib.sha256(path.read_bytes()).hexdigest()))
    roots = result["bridge_inputs"].roots_observer(result["request"])
    assert roots.task_source_revision != result["request"].roots.task_source_revision


@pytest.mark.parametrize("name", ("check_type.py", "check_offset.py"))
def test_late_public_check_script_drift_refuses_current_checker_identity(owner_case, name):
    result = _prepare(owner_case)
    (owner_case["repository"] / name).write_bytes(b"# changed check identity, never executed\n")
    with pytest.raises(ValueError, match="validation policy, scripts or checker implementation changed"):
        result["bridge_inputs"].checker_observer(result["request"])


def test_late_actual_validation_policy_drift_refuses_persisted_checker_identity(owner_case, monkeypatch):
    result = _prepare(owner_case)
    policy = deepcopy(local._plain(local.LOCAL_POLICY))
    policy["offline_policy_drift_control"] = True
    monkeypatch.setattr(local, "LOCAL_POLICY", policy)
    with pytest.raises(ValueError, match="validation policy, scripts or checker implementation changed"):
        result["bridge_inputs"].checker_observer(result["request"])


def test_rewriting_checker_metadata_cannot_authorize_changed_check_script(owner_case):
    result = _prepare(owner_case)
    path = owner_case["directory"] / "checker-policy.json"
    record = json.loads(path.read_bytes())
    record["checker_executed"] = True
    path.chmod(0o600)
    path.write_bytes(adapter._wire(record))
    with pytest.raises(ValueError, match="validation policy, scripts or checker implementation changed"):
        result["bridge_inputs"].checker_observer(result["request"])


@pytest.mark.parametrize("phase", ("source_enter_callback", "source_exit_callback"))
def test_owner_metadata_drift_inside_source_operation_invalidates_derived_roots(owner_case, phase):
    result = _prepare(owner_case)
    owner_case["state"][phase] = lambda: _rewrite(owner_case, "usage-policy.json",
        lambda value: value.__setitem__("training_steps", 1))
    with pytest.raises(ValueError, match="owner roots changed during native source"):
        result["bridge_inputs"].roots_observer(result["request"])


def test_later_roots_observer_uses_fresh_source_operation_after_preview_lease_and_deadline(owner_case):
    result = _prepare(owner_case)
    released = owner_case["scheduler"].leases[0]
    assert released.released
    owner_case["state"]["clock"] = 1000.0
    count = len(owner_case["state"]["calls"])
    roots = result["bridge_inputs"].roots_observer(result["request"])
    assert roots == result["request"].roots
    controls = owner_case["state"]["calls"][count]["controls"]
    assert controls["scheduler"] is owner_case["scheduler"] and controls["parent_lease"] is None
    assert controls["timeout_seconds"] == 90 and controls["admission_timeout_seconds"] == 30
    assert controls["memory_mb"] == 256
    assert released.release_count == 1


def test_preview_source_observer_borrows_only_its_live_owned_lease(owner_case):
    _prepare(owner_case)
    lease = owner_case["scheduler"].leases[0]
    borrowed = [row["controls"] for row in owner_case["state"]["calls"]
        if row["phase"] == "enter" and row["controls"]["parent_lease"] is not None]
    assert borrowed and all(value["parent_lease"] is lease and value["scheduler"] is None for value in borrowed)
    assert lease.released and lease.release_count == 1
    assert lease.arguments["cpu_slots"] == 1 and lease.arguments["memory_mb"] == 256
    assert lease.arguments["child_process_slots"] == 1


@pytest.mark.parametrize("field", ("preview_error", "dispatch_error"))
def test_adapter_errors_release_owned_preview_lease_exactly_once(owner_case, field):
    owner_case["state"][field] = RuntimeError("explicit controlled adapter failure")
    with pytest.raises(RuntimeError, match="controlled adapter failure"):
        _prepare(owner_case)
    leases = owner_case["scheduler"].leases
    assert len(leases) == 1 and leases[0].released and leases[0].release_count == 1


@pytest.mark.parametrize("kind", ("roots", "checker"))
def test_observers_refuse_another_request_before_owner_or_resource_side_effects(owner_case, kind):
    result = _prepare(owner_case)
    changed = replace(result["request"], caller="principal:another-owner")
    before = (len(owner_case["state"]["calls"]), len(result["records"]["owner_observations"]),
        len(owner_case["scheduler"].leases))
    with pytest.raises(ValueError, match="observer received another request"):
        getattr(result["bridge_inputs"], kind + "_observer")(changed)
    assert before == (len(owner_case["state"]["calls"]), len(result["records"]["owner_observations"]),
        len(owner_case["scheduler"].leases))


def test_typed_material_identity_reaches_every_dispatch_boundary_without_static_roots(owner_case):
    result = _prepare(owner_case)
    materials = result["materials"]
    assert materials.current_roots is None and not materials.current_facts and materials.model_provider is None
    assert materials.extra["root_observation_profile"] == native_preview.PROFILE
    assert owner_case["state"]["dispatch"][0][1]["materials"] is materials
    assert owner_case["state"]["dispatch"][1][1]["bridge_inputs"] is result["bridge_inputs"]
    assert owner_case["state"]["dispatch"][2][1]["bridge_inputs"] is result["bridge_inputs"]
    bound = contract.bind_repository_preview_materials(materials, owner_case["state"]["structural"])
    before = freeze_plan_create_input_snapshot(result["request"], materials=bound).snapshot_cid
    assert before == result["preview"].input_snapshot_cid
    materials.extra["late_controlled_material_change"] = True
    changed = contract.bind_repository_preview_materials(materials, owner_case["state"]["structural"])
    assert freeze_plan_create_input_snapshot(result["request"], materials=changed).snapshot_cid != before


def test_current_root_observations_are_append_only_and_never_claim_checker_execution(owner_case):
    result = _prepare(owner_case)
    observations = result["records"]["owner_observations"]
    before = deepcopy(observations)
    result["bridge_inputs"].roots_observer(result["request"])
    result["bridge_inputs"].checker_observer(result["request"])
    assert observations[:len(before)] == before
    assert [row["ordinal"] for row in observations] == list(range(len(observations)))
    assert [row["kind"] for row in observations[-2:]] == ["roots", "checker"]
    assert observations[-1]["checker_executed"] is observations[-1]["outcomes_recorded"] is False
    assert result["records"]["proof_authority"] is result["records"]["production_admitted"] is False


@pytest.mark.parametrize("kind", ("symlink", "directory", "empty", "oversize"))
def test_metadata_reader_refuses_nonordinary_or_unbounded_files(tmp_path, kind):
    path = tmp_path / "metadata"
    if kind == "symlink":
        target = tmp_path / "target"
        target.write_bytes(b"ordinary")
        path.symlink_to(target)
    elif kind == "directory":
        path.mkdir()
    else:
        path.write_bytes(b"" if kind == "empty" else b"x" * (adapter.MAX_FILE_BYTES + 1))
    with pytest.raises((ValueError, OSError)):
        adapter._read_file(path)


def test_exact_native_owner_rejects_an_index_protocol_double(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    head = CodebaseHead(repository_id="repository:owner-refusal", generation=1,
        manifest_cid=_cid("refusal-manifest"), snapshot_cid=_cid("refusal-snapshot"),
        ast_revision_id=f"rev:repository:owner-refusal:snapshot:{_cid('refusal-snapshot')}",
        receipt_cid=_cid("refusal-receipt"))
    with pytest.raises(native_preview.RepositoryPlanPreviewError, match="exact native codebase owner"):
        native_preview.RepositoryPlanPreviewOwner(index=SimpleNamespace(current=lambda _: head),
            repository=repository, expected_head=head, timeout_seconds=30, memory_mb=256)
