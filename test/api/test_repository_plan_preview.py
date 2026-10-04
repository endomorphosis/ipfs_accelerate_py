"""Native source fences around the opt-in, model-off repository preview."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.planning import repository_plan_preview as adapter
from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import FrozenPlanningGoal
from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    ProducerRule, TaskCandidate, TypedIntent, TypedPredicate,
    obligation_id_for_producer,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator import EvidenceAwarePlanPolicy
from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
    PlanCreateInputSnapshot, PlanCreateMaterials, PlanCreatePreviewReceipt,
    PlanCreateService, freeze_plan_create_input_snapshot,
)
from ipfs_datasets_py.logic.software_contracts.codebase_ir import (
    CodebaseScanLimits, StaleCodebaseError,
)
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    LeaseCancelledError, LeaseTimeoutError,
)
from test.api.test_plan_create_semantic_input_identity import _cid, _request
from test.api.test_structural_codebase_context import native


def preview_case(native, *, count=1, head=None):
    selected = native["head"] if head is None else head
    manifest = native["index"].load(selected.manifest_cid)
    roots = replace(
        _request().roots, repository_id=selected.repository_id,
        repository_root_cid=selected.snapshot_cid,
        dirty_worktree_root=selected.snapshot_cid,
        program_root=manifest.semantic_state.state_cid,
    )
    request = replace(
        _request(), repository_id=selected.repository_id,
        repository_root=str(native["repository"].resolve()), scope_paths=(native["source"].name,),
        roots=roots, observe_roots=True,
        budget=replace(_request().budget, max_model_calls=0),
    )
    goals = tuple(TypedPredicate(f"goal:repository:{index}", "reviewed_goal", "fixture:target",
                                object_ref=f"requirement:{index}")
                  for index in range(count))
    intent = TypedIntent("intent:repository-preview", goals, ("source:authored",),
                         current_root_id=selected.snapshot_cid)
    producers = tuple(ProducerRule(f"producer:repository:{index}", (goal.predicate_id,))
                      for index, goal in enumerate(goals))
    tasks = tuple(TaskCandidate(
        f"task:repository:{index}",
        (obligation_id_for_producer(producer.producer_id, goal.predicate_id),),
        producer_id=producer.producer_id,
    ) for index, (producer, goal) in enumerate(zip(producers, goals)))
    policy = EvidenceAwarePlanPolicy(
        acceptance_criteria=intent.goal_predicate_ids,
        evidence_terms=intent.source_refs, allowed_scopes=("scope:repository",),
        available_resource_classes=("cpu",), require_validation=True, require_proof=False,
    )
    materials = PlanCreateMaterials(
        intent=intent, producers=producers, task_candidates=tasks,
        frozen_goal=FrozenPlanningGoal("goal:repository-preview", _cid("authored-goal"),
                                       selected.snapshot_cid, policy),
        candidate_context={"domain": "repository-preview-fixture", "repository_paths": [native["source"].name],
                           "task_metadata": {task.candidate_id: {
                               "predicted_files": [native["source"].name], "scope_ids": ["scope:repository"],
                               "resource_classes": ["cpu"],
                           } for task in tasks}},
        extra={"authored_metadata": "fixture"},
    )
    owner = adapter.RepositoryPlanPreviewOwner(
        index=native["index"], repository=native["repository"].resolve(),
        expected_head=selected, scheduler=native["scheduler"], timeout_seconds=30,
        memory_mb=64,
    )
    return owner, request, materials


def run_preview(native, *, policy_observer=None, **changes):
    owner, request, materials = preview_case(native)
    values = {"owner": owner, "request": request, "materials": materials,
              "policy_observer": policy_observer or (lambda value: value.roots)}
    values.update(changes)
    return adapter.preview_repository_plan(**values)


def assert_released(native):
    state = native["scheduler"].snapshot()
    assert state["active_lease_count"] == state["waiting_request_count"] == 0


@pytest.mark.parametrize("goal_count", [1, 2])
def test_native_preview_compiles_declared_tasks_and_binds_current_structural_input(native, goal_count):
    owner, request, materials = preview_case(native, count=goal_count)
    original = deepcopy(materials.to_binding_dict())
    observed = []

    def observe(value):
        observed.append(value.request_cid)
        return value.roots

    result = adapter.preview_repository_plan(
        owner=owner, request=request, materials=materials, policy_observer=observe,
    )
    receipt = PlanCreatePreviewReceipt.from_dict(result["preview"])
    snapshot = PlanCreateInputSnapshot.from_dict(result["input_snapshot"])
    assert receipt.input_snapshot_cid == snapshot.snapshot_cid
    assert receipt.obligation_graph_cid and receipt.candidate_portfolio_cid and receipt.critique_cid
    stages = {stage.stage.value: stage for stage in receipt.stage_results}
    assert stages["obligation"].passed is True and stages["candidate"].passed is True
    assert snapshot.roots.repository_root_cid == native["head"].snapshot_cid
    assert snapshot.roots.dirty_worktree_root == native["head"].snapshot_cid
    assert snapshot.material_binding["reuse_supported"] is True
    assert result["structural_context"]["head"] == {
        name: value for name, value in native["head"].to_dict().items() if name != "schema"
    }
    assert result["structural_context"]["coverage"]["inventory_entries"] == 1
    assert result["structural_context"]["coverage"]["checked_properties"] == 0
    assert result["structural_context_cid"]
    assert result["observed_facts_supplied"] == result["model_calls"] == 0
    for name in ("source_semantics_verified", "production_admitted", "execution_authority", "completion_authority"):
        assert result[name] is False
    assert receipt.read_only is True and receipt.wrote_effects == ()
    assert len(observed) >= 2
    assert materials.to_binding_dict() == original
    assert materials.current_facts == () and materials.current_roots is None
    # The source stays in its owner; neither raw code nor a model body enters a receipt.
    encoded = json.dumps(result, sort_keys=True)
    assert "return n + 1" not in encoded and "private_source_symbol" not in encoded
    for forbidden in ("source_text", "source_body", "prompt_body", "model_weights"):
        assert '"' + forbidden + '"' not in encoded
    assert freeze_plan_create_input_snapshot(request, materials=materials).snapshot_cid != snapshot.snapshot_cid
    assert_released(native)


@pytest.mark.parametrize("name", ["repository_root_cid", "dirty_worktree_root", "program_root"])
def test_wrong_selected_source_roots_reject_before_planning(native, name):
    owner, request, materials = preview_case(native)
    request = replace(request, roots=replace(request.roots, **{name: _cid("wrong-root")}))
    calls = []
    with pytest.raises((ValueError, RuntimeError)):
        adapter.preview_repository_plan(owner=owner, request=request, materials=materials,
                                        policy_observer=lambda value: value.roots,
                                        service_factory=lambda **kwargs: calls.append(kwargs))
    assert calls == []
    assert_released(native)


@pytest.mark.parametrize("name", ["current_roots", "current_facts", "obligation_graph",
                                  "evidence_bundle", "admission_materials", "model_provider",
                                  "scan", "evidence_adapters", "evidence_queries",
                                  "parallel_request", "parallel_tasks"])
def test_caller_authority_and_model_inputs_cannot_enter_repository_route(native, name):
    owner, request, materials = preview_case(native)
    bad = request.roots if name == "current_roots" else ({"claimed": True},) if name == "current_facts" else {"claimed": True}
    if name == "model_provider":
        def bad(*args, **kwargs):
            pytest.fail("rejected provider was invoked")
    setattr(materials, name, bad)
    calls = []
    with pytest.raises((ValueError, RuntimeError)):
        adapter.preview_repository_plan(owner=owner, request=request, materials=materials,
                                        policy_observer=lambda value: value.roots,
                                        service_factory=lambda **kwargs: calls.append(kwargs))
    assert calls == []
    assert_released(native)


@pytest.mark.parametrize("field,key", [
    ("candidate_context", "request_cid"),
    ("candidate_context", "scope_paths"),
    ("candidate_context", "structural_codebase_context_cid"),
    ("extra", "structural_codebase"),
])
def test_caller_cannot_shadow_frozen_request_or_owner_context(native, field, key):
    owner, request, materials = preview_case(native)
    setattr(materials, field, {**getattr(materials, field), key: "caller-shadow"})
    with pytest.raises(adapter.RepositoryPlanPreviewError):
        adapter.preview_repository_plan(owner=owner, request=request, materials=materials,
                                        policy_observer=lambda value: value.roots)
    assert_released(native)


def test_preview_observation_never_reprepares_or_parses_repository(native, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("preview attempted source preparation, extraction or model work")

    monkeypatch.setattr(native["index"], "prepare", forbidden)
    monkeypatch.setattr(native["index"], "prepare_current", forbidden)
    monkeypatch.setattr(native["index"].ingestor, "ingest_snapshot", forbidden)
    monkeypatch.setattr(native["index"].ingestor, "_parse_or_reuse", forbidden)
    assert run_preview(native)["model_calls"] == 0
    assert_released(native)


@pytest.mark.parametrize("field", ["intent", "frozen_goal"])
def test_authored_goal_must_reference_selected_snapshot(native, field):
    owner, request, materials = preview_case(native)
    if field == "intent":
        materials.intent = replace(materials.intent, current_root_id=_cid("other-snapshot"))
    else:
        materials.frozen_goal = replace(materials.frozen_goal, repository_tree_id=_cid("other-snapshot"))
    with pytest.raises((ValueError, RuntimeError)):
        adapter.preview_repository_plan(owner=owner, request=request, materials=materials,
                                        policy_observer=lambda value: value.roots)
    assert_released(native)


def test_changed_checkout_rejects_even_when_policy_observer_returns_static_expected_roots(native):
    owner, request, materials = preview_case(native)
    native["source"].write_text("def private_source_symbol(n):\n    return n + 2\n")
    calls = []
    with pytest.raises(StaleCodebaseError):
        adapter.preview_repository_plan(owner=owner, request=request, materials=materials,
                                        policy_observer=lambda value: value.roots,
                                        service_factory=lambda **kwargs: calls.append(kwargs))
    assert calls == []
    assert_released(native)


def test_policy_observer_cannot_hide_source_edit_during_observation(native):
    def observe(value):
        native["source"].write_text("def private_source_symbol(n):\n    return n + 3\n")
        return value.roots

    with pytest.raises(StaleCodebaseError):
        run_preview(native, policy_observer=observe)
    assert_released(native)


def test_source_edit_after_real_candidate_work_cannot_persist_or_cache(native):
    services, persisted = [], []

    class EditingService(PlanCreateService):
        def _stage_candidate(self, *args, **kwargs):
            result = super()._stage_candidate(*args, **kwargs)
            native["source"].write_text("def private_source_symbol(n):\n    return n + 4\n")
            return result

        def _persist(self, receipt):
            persisted.append(receipt.receipt_cid)
            return super()._persist(receipt)

    def factory(**kwargs):
        service = EditingService(**kwargs)
        services.append(service)
        return service

    with pytest.raises((StaleCodebaseError, ValueError, RuntimeError)):
        run_preview(native, service_factory=factory)
    assert len(services) == 1
    assert persisted == [] and services[0]._preview_by_key == {}
    assert_released(native)


def test_policy_change_during_real_candidate_work_prevents_persistence(native):
    services, changed, persisted = [], [], []

    class PolicyChangingService(PlanCreateService):
        def _stage_candidate(self, *args, **kwargs):
            result = super()._stage_candidate(*args, **kwargs)
            changed.append(True)
            return result

        def _persist(self, receipt):
            persisted.append(receipt.receipt_cid)
            return super()._persist(receipt)

    def observe(value):
        return replace(value.roots, policy_root=_cid("changed-policy")) if changed else value.roots

    def factory(**kwargs):
        service = PolicyChangingService(**kwargs)
        services.append(service)
        return service

    with pytest.raises((ValueError, RuntimeError)):
        run_preview(native, policy_observer=observe, service_factory=factory)
    assert changed and len(services) == 1
    assert persisted == [] and services[0]._preview_by_key == {}
    assert_released(native)


def test_stale_selected_generation_is_not_replaced_by_current_head(native):
    owner, request, materials = preview_case(native)
    native["source"].write_text("def private_source_symbol(n):\n    return n + 5\n")
    successor = native["index"].prepare_current(
        native["repository"], repository_id=native["head"].repository_id,
        operation_id="repository-preview-successor", expected_head=native["head"],
        scheduler=native["scheduler"], memory_mb=64,
        limits=CodebaseScanLimits(max_entries=16, max_file_bytes=1024),
    ).head
    assert successor.generation > owner.expected_head.generation
    with pytest.raises(StaleCodebaseError):
        adapter.preview_repository_plan(owner=owner, request=request, materials=materials,
                                        policy_observer=lambda value: value.roots)
    assert owner.expected_head == native["head"]
    assert_released(native)


def test_native_successor_changes_frozen_metadata_and_planner_identity(native):
    first = run_preview(native)
    native["source"].write_text("def private_source_symbol(n):\n    return n + 6\n")
    successor = native["index"].prepare_current(
        native["repository"], repository_id=native["head"].repository_id,
        operation_id="repository-preview-new-source", expected_head=native["head"],
        scheduler=native["scheduler"], memory_mb=64,
        limits=CodebaseScanLimits(max_entries=16, max_file_bytes=1024),
    ).head
    owner, request, materials = preview_case(native, head=successor)
    second = adapter.preview_repository_plan(owner=owner, request=request, materials=materials,
                                             policy_observer=lambda value: value.roots)
    assert first["structural_context_cid"] != second["structural_context_cid"]
    assert first["input_snapshot"]["snapshot_cid"] != second["input_snapshot"]["snapshot_cid"]
    before = first["input_snapshot"]["material_binding"]["field_digests"]
    after = second["input_snapshot"]["material_binding"]["field_digests"]
    assert before["candidate_context"] != after["candidate_context"]
    assert before["extra"] != after["extra"]
    assert first["preview"]["obligation_graph_cid"] != second["preview"]["obligation_graph_cid"]
    assert first["preview"]["candidate_portfolio_cid"] != second["preview"]["candidate_portfolio_cid"]
    assert_released(native)


@pytest.mark.parametrize("when", ["before_entry", "during_policy_observation"])
def test_cancelled_repository_preview_returns_no_result_or_cache(native, when):
    owner, request, materials = preview_case(native)
    cancelled = threading.Event()
    owner = replace(owner, cancel_event=cancelled)
    if when == "before_entry":
        cancelled.set()

    def observe(value):
        cancelled.set()
        return value.roots

    with pytest.raises(LeaseCancelledError):
        adapter.preview_repository_plan(owner=owner, request=request, materials=materials,
                                        policy_observer=observe)
    assert_released(native)


def test_expired_overall_deadline_prevents_service_work(native):
    owner, request, materials = preview_case(native)
    owner = replace(owner, timeout_seconds=1e-9)
    calls = []
    with pytest.raises(LeaseTimeoutError):
        adapter.preview_repository_plan(owner=owner, request=request, materials=materials,
                                        policy_observer=lambda value: value.roots,
                                        service_factory=lambda **kwargs: calls.append(kwargs))
    assert calls == []
    assert_released(native)


def test_missing_live_policy_observer_and_wrong_repository_locator_reject(native):
    owner, request, materials = preview_case(native)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        adapter.preview_repository_plan(owner=owner, request=request, materials=materials,
                                        policy_observer=None)
    request = replace(request, repository_root=str(native["repository"].parent.resolve()))
    with pytest.raises((ValueError, RuntimeError)):
        adapter.preview_repository_plan(owner=owner, request=request, materials=materials,
                                        policy_observer=lambda value: value.roots)
    assert_released(native)


def test_owner_controls_are_frozen_and_native_owner_is_required(native):
    owner, request, materials = preview_case(native)
    with pytest.raises(FrozenInstanceError):
        owner.expected_head = None
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        fake = replace(owner, index=object())
        adapter.preview_repository_plan(owner=fake, request=request, materials=materials,
                                        policy_observer=lambda value: value.roots)
    assert_released(native)
