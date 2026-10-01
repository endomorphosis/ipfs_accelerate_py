"""Full semantic inputs qualify preview identity and cache reuse (TIP-007)."""

from copy import deepcopy
from dataclasses import dataclass, fields, replace

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    FactAuthority, FactTruth, ObservedFact, PredicatePolarity, ProducerRule,
    TaskCandidate, TypedIntent, TypedPredicate, compile_obligation_graph,
    obligation_id_for_producer,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
    DirtyTreePolicy, PlanAuthorityRoots, PlanCreateRequest, PlanRequestBudget,
    TaskSourceKind, plan_revision_cid,
)
from ipfs_accelerate_py.agent_supervisor.prompt import plan_create_service as module
from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
    PLAN_CREATE_INPUT_SNAPSHOT_SCHEMA, PLAN_CREATE_MATERIAL_INPUT_SNAPSHOT_SCHEMA,
    PlanCreateBodyError, PlanCreateInputSnapshot, PlanCreateMaterials,
    PlanCreatePreviewReceipt, PlanCreateService, PlanCreateServiceError,
    freeze_plan_create_input_snapshot,
)


def _cid(label):
    return plan_revision_cid({"fixture": label})


def _request():
    roots = PlanAuthorityRoots(
        repository_id="repository:semantic-inputs", task_source_id="source:semantic-inputs",
        **{name: _cid(name) for name in (
            "repository_root_cid", "dirty_worktree_root", "task_source_revision",
            "policy_root", "intent_ir_root", "legal_ir_root", "security_ir_root",
            "program_root", "capability_catalog_root", "provider_catalog_root",
            "usage_policy_root", "configuration_root",
        )},
    )
    return PlanCreateRequest(
        prompt_source_cid=_cid("prompt"), repository_id=roots.repository_id,
        repository_root="/workspace/semantic-inputs", scope_paths=("src",),
        dirty_tree_policy=DirtyTreePolicy.OBSERVE_AND_BIND,
        task_source_kind=TaskSourceKind.BOTH, board_namespace="semantic-inputs",
        alias_prefix="TIP", roots=roots, budget=PlanRequestBudget(),
        required_analysis_operations=(), optional_analysis_operations=(),
        required_logic_families=(), optional_logic_families=(),
    )


def _materials():
    goal = TypedPredicate("goal:answer", "behavior_state", "answer")
    prerequisite = TypedPredicate("goal:ready", "behavior_state", "ready")
    producer = ProducerRule("producer:answer", (goal.predicate_id,),
                            required_predicate_ids=(prerequisite.predicate_id,))
    task = TaskCandidate("task:answer", (
        obligation_id_for_producer(producer.producer_id, goal.predicate_id),
    ), producer_id=producer.producer_id)
    intent = TypedIntent("intent:answer", (goal,), ("source:answer",),
                         current_root_id=_request().roots.dirty_worktree_root)
    fact = ObservedFact("fact:ready", prerequisite, FactTruth.TRUE,
                        FactAuthority.CURRENT_ROOT_FACT, ("evidence:ready",),
                        current_root_id=_request().roots.dirty_worktree_root)
    materials = PlanCreateMaterials(
        scan={"scan_cid": _cid("scan"), "scope_paths": ["src"]},
        intent=intent, current_facts=(fact,), producers=(producer,),
        task_candidates=(task,), predicates=(prerequisite,),
        candidate_context={"assumptions": ["assumption:ready"]},
    )
    # Deliberately preserve one graph while changing independently supplied inputs.
    materials.obligation_graph = compile_obligation_graph(
        intent, current_facts=materials.current_facts, producers=materials.producers,
        task_candidates=materials.task_candidates, predicates=materials.predicates,
        current_root_id=_request().roots.dirty_worktree_root,
    )
    return materials


def test_request_only_identity_and_empty_material_callers_remain_compatible():
    request = _request()
    legacy = freeze_plan_create_input_snapshot(request)
    assert legacy.to_dict()["schema"] == PLAN_CREATE_INPUT_SNAPSHOT_SCHEMA
    assert "material_binding" not in legacy.to_dict()
    assert freeze_plan_create_input_snapshot(request, materials={}) == legacy
    assert freeze_plan_create_input_snapshot(request, materials=PlanCreateMaterials()) == legacy
    assert PlanCreateInputSnapshot.from_dict(legacy.to_dict()) == legacy


@pytest.mark.parametrize("field_name", [
    "scan", "query_plan", "evidence_bundle", "evidence_adapters",
    "obligation_graph", "intent", "frozen_goal", "parallel_tasks",
    "parallel_request", "admission_materials", "workflow_request", "current_roots",
])
def test_explicit_empty_value_differs_from_absent_optional_material(field_name):
    legacy = freeze_plan_create_input_snapshot(_request())
    material = freeze_plan_create_input_snapshot(_request(), materials={field_name: {}})
    assert material.to_dict()["schema"] == PLAN_CREATE_MATERIAL_INPUT_SNAPSHOT_SCHEMA
    assert material.snapshot_cid != legacy.snapshot_cid


def test_empty_supplied_scan_cannot_reuse_default_scan_preview():
    service, request = PlanCreateService(), _request()
    first = service.preview_create(request)
    second = service.preview_create(request, materials={"scan": {}})
    assert first is not second
    assert first.input_snapshot_cid != second.input_snapshot_cid
    assert first.scan_cid != second.scan_cid


def test_material_snapshot_roundtrip_freezes_complete_field_support():
    snapshot = freeze_plan_create_input_snapshot(_request(), materials=_materials())
    assert snapshot.to_dict()["schema"] == PLAN_CREATE_MATERIAL_INPUT_SNAPSHOT_SCHEMA
    binding = snapshot.material_binding
    assert binding["reuse_supported"] is True
    assert binding["unsupported_fields"] == {}
    assert binding["opaque_material_nonce"] == ""
    assert set(binding["field_digests"]) == {field.name for field in fields(PlanCreateMaterials)}
    assert PlanCreateInputSnapshot.from_dict(snapshot.to_dict()) == snapshot
    with pytest.raises(TypeError):
        binding["field_digests"]["intent"] = _cid("forged")
    payload = snapshot.to_dict()
    payload["material_binding"]["field_digests"]["intent"] = "sha256:" + "0" * 64
    with pytest.raises(PlanCreateServiceError, match="identity"):
        PlanCreateInputSnapshot.from_dict(payload)


@pytest.mark.parametrize("field_name", [field.name for field in fields(PlanCreateMaterials)])
def test_every_material_field_value_is_bound_even_when_claimed_cid_is_constant(field_name):
    # Isolate each declared field. A claimed ID cannot suppress value changes.
    before = PlanCreateMaterials(**{field_name: {"cid": _cid("same"), "claim": "before"}})
    after = PlanCreateMaterials(**{field_name: {"cid": _cid("same"), "claim": "after"}})
    first = freeze_plan_create_input_snapshot(_request(), materials=before)
    second = freeze_plan_create_input_snapshot(_request(), materials=after)
    assert first.snapshot_cid != second.snapshot_cid
    assert first.bounds_digest == second.bounds_digest
    assert first.material_binding["field_digests"][field_name] != second.material_binding["field_digests"][field_name]


def _change_semantics(materials, aspect):
    if aspect == "goal_polarity":
        goal = replace(materials.intent.desired_predicates[0], polarity=PredicatePolarity.NEGATIVE)
        materials.intent = replace(materials.intent, desired_predicates=(goal,))
    elif aspect == "intent_assumption":
        goal = replace(materials.intent.desired_predicates[0], assumption_refs=("assumption:new",))
        materials.intent = replace(materials.intent, desired_predicates=(goal,))
    elif aspect == "fact_truth":
        materials.current_facts = (replace(materials.current_facts[0], truth=FactTruth.FALSE),)
    elif aspect == "fact_provenance":
        materials.current_facts = (replace(materials.current_facts[0], provenance_refs=("evidence:new",)),)
    elif aspect == "producer_prerequisite":
        materials.producers = (replace(materials.producers[0], required_predicate_ids=("goal:new",)),)
    elif aspect == "producer_effect":
        materials.producers = (replace(materials.producers[0], effect_predicate_ids=("goal:new",)),)
    elif aspect == "task_dependency":
        materials.task_candidates = (replace(materials.task_candidates[0], depends_on_candidate_ids=("task:new",)),)
    elif aspect == "predicate_validation":
        materials.predicates = (replace(materials.predicates[0], validation_requirement_refs=("validation:new",)),)
    elif aspect == "candidate_policy":
        materials.candidate_context = {"assumptions": ["assumption:changed"], "policy": {"require_proof": True}}
    elif aspect == "scan_contents":
        materials.scan = {**materials.scan, "scope_paths": ["new"]}
    else:
        materials.extra = {"policy": {"allowed_operations": ["modify"]}}


@pytest.mark.parametrize("aspect", [
    "goal_polarity", "intent_assumption", "fact_truth", "fact_provenance",
    "producer_prerequisite", "producer_effect", "task_dependency",
    "predicate_validation", "candidate_policy", "scan_contents", "extra_policy",
])
def test_same_goal_graph_cannot_reuse_preview_after_semantic_inputs_change(aspect):
    service = PlanCreateService()
    request, materials = _request(), _materials()
    first = service.preview_create(request, materials=materials)
    assert service.preview_create(request, materials=materials) is first
    _change_semantics(materials, aspect)
    second = service.preview_create(request, materials=materials)
    assert second is not first
    assert first.request_cid == second.request_cid
    assert first.obligation_graph_cid == second.obligation_graph_cid
    assert first.input_snapshot_cid != second.input_snapshot_cid
    assert first.receipt_cid != second.receipt_cid
    assert PlanCreatePreviewReceipt.from_dict(second.to_dict()).receipt_cid == second.receipt_cid


def test_mapping_order_is_irrelevant_but_full_nested_values_and_finite_numbers_are_bound():
    request = _request()
    first = freeze_plan_create_input_snapshot(request, materials={"extra": {"policy": {"score": 0.25, "rank": 2}}})
    reordered = freeze_plan_create_input_snapshot(request, materials={"extra": {"policy": {"rank": 2, "score": 0.25}}})
    changed = freeze_plan_create_input_snapshot(request, materials={"extra": {"policy": {"score": 0.75, "rank": 2}}})
    assert first == reordered
    assert changed.snapshot_cid != first.snapshot_cid
    with pytest.raises(PlanCreateBodyError, match="floating"):
        module._digest("core-proof-identity", {"score": 0.25})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), b"body", {1: "non-string"}, {"api_key": "value"}])
def test_invalid_materials_fail_before_any_existing_cache_reuse(bad):
    materials, service = _materials(), PlanCreateService()
    service.preview_create(_request(), materials=materials)
    materials.extra = {"value": bad}
    with pytest.raises(PlanCreateBodyError):
        service.preview_create(_request(), materials=materials)


def test_bounded_cycles_depth_and_bytes_fail_closed(monkeypatch):
    cycle = []
    cycle.append(cycle)
    with pytest.raises(PlanCreateBodyError, match="structure"):
        freeze_plan_create_input_snapshot(_request(), materials={"extra": {"items": cycle}})
    monkeypatch.setattr(module, "MAX_SEMANTIC_MATERIAL_BYTES", 256)
    with pytest.raises(PlanCreateBodyError, match="byte bound"):
        freeze_plan_create_input_snapshot(_request(), materials={"extra": {"value": "x" * 512}})


def test_node_bound_applies_to_combined_materials_not_only_individual_fields(monkeypatch):
    monkeypatch.setattr(module, "MAX_SEMANTIC_MATERIAL_NODES", 30)
    materials = {"scan": {"items": [None] * 10}, "extra": {"items": [None] * 10}}
    with pytest.raises(PlanCreateBodyError, match="structure"):
        freeze_plan_create_input_snapshot(_request(), materials=materials)


def test_opaque_provider_disables_reuse_without_invoking_provider():
    def provider(*args, **kwargs):
        pytest.fail("deterministic mode must not call a supplied provider")
    materials, service = _materials(), PlanCreateService()
    materials.model_provider = provider
    snapshot = freeze_plan_create_input_snapshot(_request(), materials=materials)
    assert snapshot.material_binding["reuse_supported"] is False
    assert "model_provider" in snapshot.material_binding["unsupported_fields"]
    first = service.preview_create(_request(), materials=materials)
    second = service.preview_create(_request(), materials=materials)
    assert first is not second
    assert first.input_snapshot_cid != second.input_snapshot_cid
    assert not service._preview_by_key


def test_partial_to_dict_live_object_is_explicitly_nonreusable():
    class PartialRecord:
        ready = True
        def to_dict(self):
            return {"cid": _cid("claimed")}
    materials = PlanCreateMaterials(evidence_bundle=PartialRecord())
    first = freeze_plan_create_input_snapshot(_request(), materials=materials)
    materials.evidence_bundle.ready = False
    second = freeze_plan_create_input_snapshot(_request(), materials=materials)
    assert first.material_binding["reuse_supported"] is False
    assert "evidence_bundle" in first.material_binding["unsupported_fields"]
    assert first.snapshot_cid != second.snapshot_cid


def test_dataclass_full_fields_precede_partial_to_dict_and_bind_runtime_type():
    @dataclass
    class PartialRecord:
        ready: bool = True
        def to_dict(self):
            return {"cid": _cid("claimed")}
    first = freeze_plan_create_input_snapshot(_request(), materials={"evidence_bundle": PartialRecord()})
    changed = freeze_plan_create_input_snapshot(_request(), materials={"evidence_bundle": PartialRecord(False)})
    mapping = freeze_plan_create_input_snapshot(_request(), materials={"evidence_bundle": PartialRecord().to_dict()})
    assert first.material_binding["reuse_supported"] is True
    assert first.snapshot_cid != changed.snapshot_cid
    assert first.snapshot_cid != mapping.snapshot_cid


def test_plain_mapping_cannot_impersonate_the_internal_typed_record_encoding():
    @dataclass
    class Record:
        ready: bool = True
    item = Record()
    spoof = {"$record": f"{type(item).__module__}.{type(item).__qualname__}",
             "fields": {"$mapping": {"ready": True}}}
    first = freeze_plan_create_input_snapshot(_request(), materials={"extra": {"item": item}})
    mapping = freeze_plan_create_input_snapshot(_request(), materials={"extra": {"item": spoof}})
    assert first.snapshot_cid != mapping.snapshot_cid


def test_typed_query_plan_and_equal_public_mapping_cannot_share_cache_identity():
    service, request, materials = PlanCreateService(), _request(), _materials()
    plan = service.query_planner.compile(request)
    materials.query_plan = plan
    typed = service.preview_create(request, materials=materials)
    materials.query_plan = plan.to_dict()
    mapped = service.preview_create(request, materials=materials)
    assert mapped is not typed
    assert mapped.input_snapshot_cid != typed.input_snapshot_cid


def test_live_dataclass_evidence_adapter_remains_nonreusable():
    @dataclass
    class Adapter:
        ready: bool = True
        def to_dict(self):
            return {"ready": self.ready}
    snapshot = freeze_plan_create_input_snapshot(_request(), materials={"evidence_adapters": {"slot": Adapter()}})
    assert snapshot.material_binding["reuse_supported"] is False
    assert "evidence_adapters" in snapshot.material_binding["unsupported_fields"]


def test_mutation_during_preview_cannot_persist_or_cache_stale_identity():
    materials, store = _materials(), {}
    def observe(request):
        materials.extra = {"assumption": "changed-during-preview"}
        return request.roots
    service = PlanCreateService(root_observer=observe, receipt_store=store)
    with pytest.raises(PlanCreateServiceError, match="materials changed"):
        service.preview_create(_request(), materials=materials)
    assert store == {}
    assert service._preview_by_key == {}


@pytest.mark.parametrize("mutation", ["drop-field", "overclaim-reuse", "remove-binding", "downgrade-schema", "unknown-field", "unknown-snapshot-field"])
def test_material_snapshot_rejects_forged_or_incomplete_support(mutation):
    snapshot = freeze_plan_create_input_snapshot(_request(), materials=_materials())
    payload = deepcopy(snapshot.to_dict())
    if mutation == "drop-field":
        payload["material_binding"]["field_digests"].pop("intent")
    elif mutation == "overclaim-reuse":
        payload["material_binding"]["reuse_supported"] = False
    elif mutation == "remove-binding":
        payload.pop("material_binding")
    elif mutation == "downgrade-schema":
        payload["schema"] = PLAN_CREATE_INPUT_SNAPSHOT_SCHEMA
    elif mutation == "unknown-field":
        payload["material_binding"]["unknown"] = True
    else:
        payload["unbound_semantic_input"] = {"claim": True}
    with pytest.raises(PlanCreateServiceError):
        PlanCreateInputSnapshot.from_dict(payload)
