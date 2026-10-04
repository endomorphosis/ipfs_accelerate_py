"""Tiny native source successors through actual model-off planning stages.

Each source fixture uses native DuckDB/CAS publication and real scheduler
leases. Planner callbacks are fault controls, never source-property proofs or
worker execution. Deadline advancement below is injected clock telemetry.
"""
from copy import deepcopy
from dataclasses import replace
import json
import threading
import time
from types import SimpleNamespace

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.planning import codebase_source_delta_context as adapter
from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import FrozenPlanningGoal
from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    ProducerRule, TaskCandidate, TypedIntent, TypedPredicate, obligation_id_for_producer,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator import EvidenceAwarePlanPolicy
from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
    PlanCreateInputSnapshot, PlanCreateMaterials, PlanCreatePreviewReceipt,
    PlanCreateService, freeze_plan_create_input_snapshot,
)
from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog, CodebaseCatalogLimits
from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as scan
from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor as delta
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex, CodebaseScanLimits
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, LeaseCancelledError, LeaseTimeoutError, ResourceSchedulerConfig,
)
from test.api.test_plan_create_semantic_input_identity import _request


@pytest.fixture(autouse=True)
def forbid_model_work(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        pytest.fail("source-only planning opened a model owner or executed training/inference")

    monkeypatch.setattr(AutoencoderRegistry, "__init__", forbidden)
    monkeypatch.setattr(scan.features, "train_projection_features", forbidden)
    monkeypatch.setattr(scan.features, "infer_projection_features", forbidden)
    monkeypatch.setattr(scan.training, "train_current_codebase_features", forbidden)
    monkeypatch.setattr(scan, "_worker", forbidden)
    yield calls
    assert calls == []


@pytest.fixture
def native(tmp_path):
    repository = tmp_path / "private-source-repository"
    repository.mkdir()
    for name in ("keep", "change", "remove", "rename"):
        (repository / (name + ".py")).write_text(
            "def private_" + name + "(n: int) -> int:\n    return n + 1\n")
    (repository / "note.txt").write_text("complete unindexed source member\n")
    (repository / "opaque.py").symlink_to("missing-target")
    connection = duckdb.connect(str(tmp_path / "source.duckdb"),
                                config={"threads": 1, "memory_limit": "128MB"})
    store = DuckDBASTStore(connection=connection)
    artifacts = ImmutableCAS(tmp_path / "artifacts")
    index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=artifacts,
        catalog=CodebaseCatalog(store, artifacts, limits=CodebaseCatalogLimits(max_entries=32)))
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "scheduler.json", proof_resource_sampler=lambda: ProofHostResources(8, 8192, 8192),
        lane_reservations={}, auto_renew_leases=False, proof_backoff_seconds=0,
        poll_interval_seconds=0.005))
    capture = CodebaseScanLimits(max_entries=16, max_file_bytes=4096)
    old = index.prepare_current(repository, repository_id="worktree:source-successor", operation_id="old",
        expected_head=None, limits=capture, scheduler=scheduler, memory_mb=1024)
    (repository / "change.py").write_text("def private_change(n: int) -> int:\n    return n + 2\n")
    (repository / "remove.py").unlink()
    (repository / "rename.py").rename(repository / "renamed.py")
    (repository / "added.py").write_text("def private_added(n: int) -> int:\n    return n + 3\n")
    current = index.prepare_current(repository, repository_id=old.repository_id, operation_id="current",
        expected_head=old.head, limits=capture, scheduler=scheduler, memory_mb=1024)
    record = delta.build_current_codebase_source_delta(index, repository, previous_head=old.head,
        expected_head=current.head, scheduler=scheduler)
    value = SimpleNamespace(repository=repository, index=index, connection=connection,
        scheduler=scheduler, capture=capture, old=old, current=current, record=record)
    try:
        yield value
    finally:
        state = scheduler.snapshot()
        assert state["active_lease_count"] == state["waiting_request_count"] == 0
        connection.close()


def planning_case(native):
    head = native.current.head
    manifest = native.index.load(head.manifest_cid)
    baseline = _request()
    request = replace(baseline, repository_id=head.repository_id,
        repository_root=str(native.repository.resolve()), scope_paths=("change.py",), observe_roots=True,
        budget=replace(baseline.budget, max_model_calls=0),
        roots=replace(baseline.roots, repository_id=head.repository_id,
            repository_root_cid=head.snapshot_cid, dirty_worktree_root=head.snapshot_cid,
            program_root=manifest.semantic_state.state_cid))
    goals = tuple(TypedPredicate("goal:runtime:" + str(number), "reviewed_runtime_requirement", "change.py",
                                object_ref="requirement:" + str(number)) for number in range(2))
    intent = TypedIntent("intent:authored-source-successor", goals, ("source:authored-runtime",),
                         current_root_id=head.snapshot_cid)
    producers = tuple(ProducerRule("producer:runtime:" + str(number), (goal.predicate_id,))
                      for number, goal in enumerate(goals))
    tasks = tuple(TaskCandidate("task:runtime:" + str(number),
        (obligation_id_for_producer(producer.producer_id, goal.predicate_id),), producer_id=producer.producer_id)
        for number, (producer, goal) in enumerate(zip(producers, goals)))
    materials = PlanCreateMaterials(intent=intent, producers=producers, task_candidates=tasks,
        frozen_goal=FrozenPlanningGoal("goal:runtime", cid_for_structured({"authored_goal": "runtime"}),
            head.snapshot_cid, EvidenceAwarePlanPolicy(acceptance_criteria=intent.goal_predicate_ids,
                evidence_terms=intent.source_refs, allowed_scopes=("scope:change.py",),
                available_resource_classes=("cpu",), require_validation=True, require_proof=False)),
        candidate_context={"repository_paths": ["change.py"], "task_metadata": {
            item.candidate_id: {"predicted_files": ["change.py"], "scope_ids": ["scope:change.py"],
                               "resource_classes": ["cpu"]} for item in tasks}},
        extra={"authored_review": "review:complete-runtime-requirements"})
    return request, materials


def run(native, **changes):
    request, materials = planning_case(native)
    arguments = {"materials": materials, "request": request,
        "policy_observer": lambda value: value.roots, "scheduler": native.scheduler}
    arguments.update(changes)
    return adapter.preview_current_source_delta_plan(native.record, native.index, native.repository, **arguments)


def test_native_successor_references_bind_actual_planner_and_keep_all_runtime_tasks(native):
    request, materials = planning_case(native)
    authored = deepcopy(materials.to_binding_dict())
    original = freeze_plan_create_input_snapshot(request, materials=materials)
    observed = []

    def observe(value):
        observed.append(value.request_cid)
        return value.roots

    result = run(native, request=request, materials=materials, policy_observer=observe)
    assert result["schema"] == adapter.SCHEMA
    refs = result[adapter.MATERIAL_KEY]
    assert refs["schema"] == adapter.REFS_SCHEMA
    body = native.record.to_dict()
    assert refs["artifact_cid"] == native.record.artifact_cid
    assert refs["previous_head"] == native.old.head.to_dict()
    assert refs["current_head"] == native.current.head.to_dict()
    assert refs["coverage"] == body["coverage"]
    assert refs["coverage"]["classifications"] == {"retained": 3, "changed": 1, "added": 2, "removed": 2}
    assert [row["source_key"] for row in refs["ledger"]] == [row["source_key"] for row in body["ledger"]]
    for row, source in zip(refs["ledger"], body["ledger"]):
        for side in ("previous", "current"):
            assert row[side] == (None if source[side] is None else source[side]["member"])
    kept = next(row for row in refs["ledger"] if row["source_key"] == "raw:" + b"keep.py".hex())
    assert kept["source_bytes_comparison"] == "equal" and kept["ast_identity_comparison"] == "different"
    assert refs["numerical_reuse"] is refs["model_advanced"] is refs["physical_absence_verified"] is False
    assert refs["removal_scope"] == "absent_from_current_complete_capture"
    assert all(flag is False for flag in refs["authority"].values())
    assert result["declared_requirement_ids"] == list(materials.intent.goal_predicate_ids)
    assert result["declared_task_ids"] == [item.candidate_id for item in materials.task_candidates]
    assert [row["predicate_id"] for row in result["residual_requirements"]] == result["declared_requirement_ids"]
    assert all(row["status"] == "runtime_behavior_unresolved" for row in result["residual_requirements"])
    assert result["current_facts"] == result["removed_task_ids"] == []
    assert result["training_steps"] == result["inference_calls"] == 0
    assert all(flag is False for flag in result["authority"].values())
    receipt = PlanCreatePreviewReceipt.from_dict(result["repository_preview"]["preview"])
    snapshot = PlanCreateInputSnapshot.from_dict(result["repository_preview"]["input_snapshot"])
    assert receipt.input_snapshot_cid == snapshot.snapshot_cid
    stages = {stage.stage.value: stage for stage in receipt.stage_results}
    assert stages["obligation"].passed is True and stages["candidate"].passed is True
    assert receipt.read_only is True and receipt.wrote_effects == ()
    assert snapshot.snapshot_cid != original.snapshot_cid and snapshot.material_binding["reuse_supported"] is True
    assert len(observed) >= 2
    assert materials.to_binding_dict() == authored and adapter.MATERIAL_KEY not in materials.extra
    assert cid_for_structured({key: value for key, value in result.items() if key != "result_cid"}) == result["result_cid"]
    encoded = json.dumps(result, sort_keys=True)
    assert "return n +" not in encoded and "private_change" not in encoded
    assert '"model_weights"' not in encoded and '"source_text"' not in encoded


def test_receiving_uses_published_sources_without_reprepare_or_parse(native, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("planning reparsed or republished the source checkout")

    for name in ("prepare", "prepare_current"):
        monkeypatch.setattr(native.index, name, forbidden)
    monkeypatch.setattr(native.index.ingestor, "ingest_snapshot", forbidden)
    monkeypatch.setattr(native.index.ingestor, "_parse_or_reuse", forbidden)
    assert run(native)["current_facts"] == []


@pytest.mark.parametrize("part", ["checkout", "old_source", "current_ast", "delta_record", "native_sql"])
def test_corrupt_native_source_or_delta_refuses_before_native_planning(native, monkeypatch, part):
    entered = []
    original_stage = PlanCreateService._stage_candidate

    def stage(*args, **kwargs):
        entered.append(1)
        return original_stage(*args, **kwargs)

    monkeypatch.setattr(PlanCreateService, "_stage_candidate", stage)
    if part == "checkout":
        (native.repository / "keep.py").write_text("def private_keep(n):\n    return n + 91\n")
    elif part == "native_sql":
        native.connection.execute("UPDATE symbols SET name='corrupted' WHERE name='private_keep'")
    else:
        manifest = native.index.load(native.old.manifest_cid if part == "old_source" else native.current.manifest_cid)
        entry = next(entry for entry in manifest.snapshot.entries if entry.path == "change.py")
        unit = next(unit for unit in manifest.units if unit.source_key == entry.source_key)
        artifact = native.record.artifact_cid if part == "delta_record" else entry.source_cid if part == "old_source" else unit.ast_cid
        path = native.index.artifacts.path_for(artifact, source=part == "old_source")
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        run(native)
    assert entered == []


@pytest.mark.parametrize("part", ["checkout", "source_cas", "authored_materials", "policy_roots"])
def test_late_policy_callback_cannot_hide_source_or_material_drift(native, part):
    request, materials = planning_case(native)
    observed = []

    def observe(value):
        observed.append(1)
        if part == "checkout":
            (native.repository / "keep.py").write_text("def private_keep(n):\n    return n + 92\n")
        elif part == "source_cas":
            manifest = native.index.load(native.current.manifest_cid)
            entry = next(entry for entry in manifest.snapshot.entries if entry.path == "keep.py")
            path = native.index.artifacts.path_for(entry.source_cid, source=True)
            path.write_bytes(path.read_bytes() + b" ")
        elif part == "authored_materials":
            materials.extra["authored_review"] = "changed-during-policy"
        else:
            return replace(value.roots, policy_root=cid_for_structured({"changed_policy": True}))
        return value.roots

    with pytest.raises((ValueError, RuntimeError)):
        run(native, request=request, materials=materials, policy_observer=observe)
    assert observed


@pytest.mark.parametrize("part", ["checkout", "delta_record", "authored_materials"])
def test_mutation_after_actual_candidate_stage_withholds_preview(native, monkeypatch, part):
    request, materials = planning_case(native)
    completed, persisted = [], []
    original_stage, original_persist = PlanCreateService._stage_candidate, PlanCreateService._persist

    def stage(*args, **kwargs):
        result = original_stage(*args, **kwargs)
        completed.append(1)
        if part == "checkout":
            (native.repository / "keep.py").write_text("def private_keep(n):\n    return n + 93\n")
        elif part == "delta_record":
            path = native.index.artifacts.path_for(native.record.artifact_cid)
            path.write_bytes(path.read_bytes() + b" ")
        else:
            materials.candidate_context["repository_paths"].append("unreviewed.py")
        return result

    def persist(*args, **kwargs):
        persisted.append(1)
        return original_persist(*args, **kwargs)

    monkeypatch.setattr(PlanCreateService, "_stage_candidate", stage)
    monkeypatch.setattr(PlanCreateService, "_persist", persist)
    with pytest.raises((ValueError, RuntimeError)):
        run(native, request=request, materials=materials)
    assert completed == [1]
    if part != "delta_record":
        assert persisted == []


def test_final_consumer_pin_callback_source_edit_is_caught_by_last_native_receiver(native, monkeypatch):
    completed, changed = [], []
    native_preview, producer_pin = adapter.preview_repository_plan, adapter._producer_pin

    def preview(*args, **kwargs):
        result = native_preview(*args, **kwargs)
        completed.append(1)
        return result

    def pin():
        result = producer_pin()
        if completed and not changed:
            changed.append(1)
            (native.repository / "keep.py").write_text("def private_keep(n):\n    return n + 94\n")
        return result

    monkeypatch.setattr(adapter, "preview_repository_plan", preview)
    monkeypatch.setattr(adapter, "_producer_pin", pin)
    with pytest.raises(ValueError):
        run(native)
    assert completed == changed == [1]


@pytest.mark.parametrize("part", ["bound_references", "authored_materials"])
def test_final_native_receiving_cas_callback_cannot_mutate_already_planned_materials(native, monkeypatch, part):
    request, materials = planning_case(native)
    bound_materials, completed, reads, receiving = [], [], [], []
    native_preview = adapter.preview_repository_plan
    native_receive = adapter.validate_current_codebase_source_delta
    native_get = native.index.artifacts.get

    def preview(*args, **kwargs):
        bound_materials.append(kwargs["materials"])
        result = native_preview(*args, **kwargs)
        completed.append(1)
        return result

    def get(*args, **kwargs):
        value = native_get(*args, **kwargs)
        if completed and len(receiving) == 2 and not reads:
            reads.append(1)
            if part == "bound_references":
                bound_materials[0].extra[adapter.MATERIAL_KEY]["coverage"]["classifications"]["retained"] += 1
            else:
                materials.extra["authored_review"] = "changed-during-final-native-CAS-read"
        return value

    def receive(*args, **kwargs):
        receiving.append(1)
        return native_receive(*args, **kwargs)

    monkeypatch.setattr(adapter, "preview_repository_plan", preview)
    monkeypatch.setattr(adapter, "validate_current_codebase_source_delta", receive)
    monkeypatch.setattr(native.index.artifacts, "get", get)
    with pytest.raises(ValueError):
        run(native, request=request, materials=materials)
    assert completed == reads == [1] and receiving == [1, 1]
    # This fault changes neither source bytes nor the exact current source head.
    assert native.index.current(native.current.repository_id) == native.current.head
    assert (native.repository / "keep.py").read_text().endswith("return n + 1\n")


@pytest.mark.parametrize("mutation", ["tag", "count_alias", "authority_alias", "material_commitment"])
def test_native_preview_response_tampering_cannot_return_source_successor_context(native, monkeypatch, mutation):
    native_preview, completed = adapter.preview_repository_plan, []

    def preview(*args, **kwargs):
        result = native_preview(*args, **kwargs)
        completed.append(1)
        if mutation == "tag":
            result["schema"] = "supervisor-repository-plan-preview@2"
        elif mutation == "count_alias":
            result["model_calls"] = False
        elif mutation == "authority_alias":
            result["worker_launched"] = 0
        else:
            result["input_snapshot"]["material_binding"]["field_digests"]["extra"] = "sha256:" + "0" * 64
        return result

    monkeypatch.setattr(adapter, "preview_repository_plan", preview)
    with pytest.raises(ValueError):
        run(native)
    assert completed == [1]


def test_same_snapshot_new_publication_generation_does_not_refresh_old_delta(native, monkeypatch):
    newer = native.index.prepare_current(native.repository, repository_id=native.current.repository_id,
        operation_id="same-snapshot-republication", expected_head=native.current.head,
        limits=native.capture, scheduler=native.scheduler, memory_mb=1024)
    assert newer.head.snapshot_cid == native.current.head.snapshot_cid
    assert newer.head.generation == native.current.head.generation + 1
    entered = []
    monkeypatch.setattr(PlanCreateService, "_stage_candidate", lambda *args, **kwargs: entered.append(1))
    with pytest.raises(ValueError):
        run(native)
    assert entered == []


@pytest.mark.parametrize("container", ["extra", "candidate_context"])
def test_reserved_material_collision_refuses_before_resource_admission(native, container):
    request, materials = planning_case(native)
    getattr(materials, container)[adapter.MATERIAL_KEY] = {"schema": "caller-forgery"}
    before = native.scheduler.snapshot()["counters"]["acquisitions_total"]
    with pytest.raises(ValueError):
        run(native, request=request, materials=materials)
    assert native.scheduler.snapshot()["counters"]["acquisitions_total"] == before


@pytest.mark.parametrize("controls", [
    {"timeout_seconds": True}, {"timeout_seconds": 0}, {"timeout_seconds": float("nan")},
    {"admission_timeout_seconds": True}, {"memory_mb": True}, {"memory_mb": 1023}, {"cancel_event": object()},
])
def test_control_type_aliases_refuse_before_resource_admission(native, controls):
    before = native.scheduler.snapshot()["counters"]["acquisitions_total"]
    with pytest.raises((TypeError, ValueError)):
        run(native, **controls)
    assert native.scheduler.snapshot()["counters"]["acquisitions_total"] == before


def test_precancelled_request_enters_no_native_planning(native, monkeypatch):
    cancel = threading.Event()
    cancel.set()
    entered = []
    monkeypatch.setattr(PlanCreateService, "_stage_candidate", lambda *args, **kwargs: entered.append(1))
    with pytest.raises(LeaseCancelledError):
        run(native, cancel_event=cancel)
    assert entered == []


def test_cancellation_after_native_candidate_keeps_result_withheld(native, monkeypatch):
    cancel, completed = threading.Event(), []
    original = PlanCreateService._stage_candidate

    def stage(*args, **kwargs):
        result = original(*args, **kwargs)
        completed.append(1)
        cancel.set()
        return result

    monkeypatch.setattr(PlanCreateService, "_stage_candidate", stage)
    with pytest.raises(LeaseCancelledError):
        run(native, cancel_event=cancel)
    assert completed == [1]


def test_expired_shared_deadline_after_policy_is_not_integrity_failure(native, monkeypatch):
    advanced, observed = [], []
    clock = time.monotonic
    monkeypatch.setattr(adapter, "time", SimpleNamespace(monotonic=lambda: clock() + (601 if advanced else 0)))

    def observe(value):
        observed.append(1)
        advanced.append(1)
        return value.roots

    with pytest.raises(LeaseTimeoutError):
        run(native, policy_observer=observe)
    assert observed == [1]


def test_exact_native_record_type_is_required(native):
    alias = SimpleNamespace(artifact_cid=native.record.artifact_cid, _payload=native.record._payload,
                            to_dict=native.record.to_dict)
    request, materials = planning_case(native)
    before = native.scheduler.snapshot()["counters"]["acquisitions_total"]
    with pytest.raises(ValueError):
        adapter.preview_current_source_delta_plan(alias, native.index, native.repository,
            request=request, materials=materials, policy_observer=lambda value: value.roots,
            scheduler=native.scheduler)
    assert native.scheduler.snapshot()["counters"]["acquisitions_total"] == before
