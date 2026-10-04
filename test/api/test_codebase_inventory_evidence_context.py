"""Pure advisory metadata, native receipt, and live-fence protocol controls.

The scope/validator doubles below test orchestration only. They supply no
native source, checkpoint, normalized-catalog or host qualification evidence.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import sys
from types import ModuleType, SimpleNamespace
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.planning import codebase_inventory_evidence_context as adapter
from ipfs_accelerate_py.agent_supervisor.planning.adaptive_planner import FrozenPlanningGoal
from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    ProducerRule, TaskCandidate, TypedIntent, TypedPredicate, obligation_id_for_producer,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_evaluator import EvidenceAwarePlanPolicy
from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
    PlanCreateMaterials, PlanCreateMode, PlanCreateService, freeze_plan_create_input_snapshot,
)
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError
from test.api.test_plan_create_semantic_input_identity import _request


def cid(label):
    return cid_for_structured({"fixture": label})


def reference_case():
    head = CodebaseHead(repository_id="repository:inventory-fixture", generation=1,
        manifest_cid=cid("manifest"), snapshot_cid=cid("snapshot"),
        ast_revision_id="rev:repository:inventory-fixture:snapshot:" + cid("snapshot"),
        receipt_cid=cid("publication"))
    ledger = [
        {"source_key": "source:unit", "entry_cid": cid("source-entry"),
         "evidence_disposition": "matched_complete", "evidence_entry_ids": [cid("proof-entry")]},
        {"source_key": "source:opaque", "entry_cid": cid("opaque-entry"),
         "evidence_disposition": "no_exact_indexed_conditional_evidence", "evidence_entry_ids": []},
    ]
    raw_id = cid_for_bytes(b"protocol-only retained fixture")
    refs = {"schema": "codebase-inventory-evidence-advisory-refs@1", "artifact_cid": raw_id,
        "head": head.to_dict(), "head_cid": cid_for_structured(head.to_dict()),
        "scan_artifact_cid": cid_for_bytes(b"protocol-only scan"),
        "membership_cid": cid_for_structured([{key: row[key] for key in ("source_key", "entry_cid")} for row in ledger]),
        "model": {"version_id": "fixture-model", "variant_id": "fixture-variant", "artifact_cid": cid_for_bytes(b"fixture model"),
            "contract_sha256": "1" * 64, "state_sha256": "2" * 64, "feature_space_sha256": "3" * 64},
        "coverage": {"inventory_entries": 2, "inferred_rows": 1, "evidence_entries": 1,
            "evidence_matched_members": 1, "evidence_complete_absent_members": 1, "evidence_unknown_members": 0},
        "query": {"selector_cid": cid("selector"), "inventory_cid": cid("inventory"), "epoch": 1,
            "complete": True, "next_cursor": None, "page_cids": [cid("page")]},
        "entry_evidence": ledger, "authority": {name: False for name in adapter._AUTHORITY_NAMES}}
    return head, raw_id, refs


def planning_case(head):
    request = _request()
    request = replace(request, repository_id=head.repository_id, repository_root="/workspace/inventory-fixture",
        scope_paths=("unit.py",), budget=replace(request.budget, max_model_calls=0),
        roots=replace(request.roots, repository_id=head.repository_id,
            repository_root_cid=head.snapshot_cid, dirty_worktree_root=head.snapshot_cid,
            program_root=cid("semantic")))
    goals = tuple(TypedPredicate(f"goal:runtime:{number}", "reviewed_runtime_requirement", "unit.py",
                                object_ref=f"requirement:{number}") for number in range(2))
    intent = TypedIntent("intent:authored-runtime", goals, ("source:reviewed-runtime",), current_root_id=head.snapshot_cid)
    producers = tuple(ProducerRule(f"producer:runtime:{number}", (goal.predicate_id,))
                      for number, goal in enumerate(goals))
    tasks = tuple(TaskCandidate(f"task:runtime:{number}",
        (obligation_id_for_producer(producer.producer_id, goal.predicate_id),), producer_id=producer.producer_id)
        for number, (producer, goal) in enumerate(zip(producers, goals)))
    materials = PlanCreateMaterials(intent=intent, producers=producers, task_candidates=tasks,
        frozen_goal=FrozenPlanningGoal("goal:runtime", cid("reviewed-goal"), head.snapshot_cid,
            EvidenceAwarePlanPolicy(acceptance_criteria=intent.goal_predicate_ids, evidence_terms=intent.source_refs,
                allowed_scopes=("scope:unit.py",), available_resource_classes=("cpu",),
                require_validation=True, require_proof=False)),
        candidate_context={"repository_paths": ["unit.py"], "task_metadata": {
            item.candidate_id: {"predicted_files": ["unit.py"], "scope_ids": ["scope:unit.py"], "resource_classes": ["cpu"]}
            for item in tasks}}, extra={"caller_review": "review:fixture"})
    return request, materials


def native_memory_preview(*, owner, request, materials, policy_observer):
    """Use actual planner/receipt classes; substitute source observation only."""
    policy_observer(request)
    context = {"schema": "supervisor-structural-codebase-context@1",
        "head": {key: value for key, value in owner.expected_head.to_dict().items() if key != "schema"},
        "semantic_state_cid": request.roots.program_root,
        "coverage": {"inventory_entries": 2, "captured_entries": 1, "ast_ok": 1, "ast_partial": 0,
            "ast_failed": 0, "opaque_entries": 1, "unindexed_entries": 0, "semantic_symbols": 1,
            "formalized_properties": 0, "checked_properties": 0},
        "authority": "structural_only", "source_semantics_verified": False, "proof_authority": False,
        "execution_authority": False, "completion_authority": False}
    context_cid = cid_for_structured(context)
    bound = replace(materials, scan={"scan_cid": context_cid, "structural_codebase": context},
        candidate_context={**materials.candidate_context, "structural_codebase_context_cid": context_cid,
                           "structural_codebase": context},
        extra={**materials.extra, "structural_codebase_context_cid": context_cid, "structural_codebase": context,
               "root_observation_profile": "repository-live-root-observation@1"})
    snapshot = freeze_plan_create_input_snapshot(request, materials=bound)
    receipt = PlanCreateService(root_observer=lambda value: value.roots,
        require_live_root_observation=True).preview_create(request, mode=PlanCreateMode.DETERMINISTIC, materials=bound)
    return {"schema": "supervisor-repository-plan-preview@1", "preview": receipt.to_dict(),
        "input_snapshot": snapshot.to_dict(), "structural_context": context, "structural_context_cid": context_cid,
        "observed_facts_supplied": 0, "model_calls": 0, "source_semantics_verified": False,
        "proof_authority": False, "production_admitted": False, "worker_launched": False,
        "execution_authority": False, "completion_authority": False}


@pytest.fixture
def protocol(monkeypatch):
    """Explicit orchestration doubles; never an alternative production owner."""
    head, artifact, refs = reference_case()
    request, materials = planning_case(head)
    events = []

    class FixtureRecord:
        artifact_cid = artifact
        _payload = b"protocol-only retained fixture"

        def advisory_refs(self):
            return deepcopy(refs)

    def validate(record, index, repository, **controls):
        events.append(("validate", controls))
        return record

    module_name = "ipfs_datasets_py.logic.software_contracts.codebase_inventory_evidence"
    module = ModuleType(module_name)
    module.CodebaseInventoryEvidenceRecord = FixtureRecord
    module.validate_current_inventory_evidence = validate
    monkeypatch.setitem(sys.modules, module_name, module)
    from ipfs_datasets_py.logic.software_contracts import codebase_resources

    class Lease:
        def combined_cancellation_signal(self, signal):
            return signal or threading.Event()

    lease = Lease()

    @contextmanager
    def acquire(**controls):
        events.append(("admit", controls))
        try:
            yield lease
        finally:
            events.append(("release", {}))

    monkeypatch.setattr(codebase_resources, "acquire_codebase_resources", acquire)
    monkeypatch.setattr(adapter, "RepositoryPlanPreviewOwner", lambda **controls: SimpleNamespace(**controls))
    monkeypatch.setattr(adapter, "preview_repository_plan", native_memory_preview)
    record = FixtureRecord()
    arguments = dict(verification_catalog=object(), registry=object(), materials=materials,
                     request=request, policy_observer=lambda value: value.roots)
    return SimpleNamespace(record=record, head=head, refs=refs, module=module, events=events,
        arguments=arguments, lease=lease, request=request, materials=materials,
        run=lambda **changes: adapter.preview_current_inventory_evidence_plan(
            record, object(), "/workspace/inventory-fixture", **{**arguments, **changes}))


def test_reference_ledger_detaches_and_preserves_complete_partial_coverage():
    head, artifact, refs = reference_case()
    checked = adapter._validate_refs(refs, artifact_cid=artifact, head=head)
    checked["coverage"]["inventory_entries"] = 99
    assert refs["coverage"]["inventory_entries"] == 2
    assert refs["entry_evidence"][1]["evidence_disposition"] == "no_exact_indexed_conditional_evidence"


def test_zero_consumed_page_byte_budget_retains_all_members_as_unknown():
    head, artifact, refs = reference_case()
    refs["query"].update(complete=False, next_cursor=None, page_cids=[])
    refs["coverage"].update(evidence_entries=0, evidence_matched_members=0,
                            evidence_complete_absent_members=0, evidence_unknown_members=2)
    for row in refs["entry_evidence"]:
        row.update(evidence_disposition="unknown_budget", evidence_entry_ids=[])
    checked = adapter._validate_refs(refs, artifact_cid=artifact, head=head)
    assert checked["query"]["complete"] is False and checked["query"]["page_cids"] == []
    assert checked["coverage"]["evidence_complete_absent_members"] == 0


def test_partial_page_inventory_preserves_matched_and_unknown_members_without_absence():
    from ipfs_datasets_py.duckdb_control.codebase_verification_queries import CodebaseVerificationQueryCursor
    head, artifact, refs = reference_case()
    cursor = CodebaseVerificationQueryCursor(refs["head_cid"], refs["query"]["inventory_cid"],
        refs["query"]["epoch"], refs["query"]["selector_cid"], cid("proof-entry"))
    refs["query"].update(complete=False, next_cursor=cursor.to_dict())
    refs["coverage"].update(evidence_complete_absent_members=0, evidence_unknown_members=1)
    refs["entry_evidence"][0]["evidence_disposition"] = "matched_partial"
    refs["entry_evidence"][1]["evidence_disposition"] = "unknown_budget"
    assert adapter._validate_refs(refs, artifact_cid=artifact, head=head) == refs


@pytest.mark.parametrize("flag", sorted(adapter._AUTHORITY_NAMES))
@pytest.mark.parametrize("bad", [0, True])
def test_references_reject_elevated_authority_and_boolean_aliases(flag, bad):
    head, artifact, refs = reference_case()
    refs["authority"][flag] = bad
    with pytest.raises(adapter.CodebaseInventoryEvidenceContextError, match="false authority"):
        adapter._validate_refs(refs, artifact_cid=artifact, head=head)


@pytest.mark.parametrize("mutation", [
    lambda value: value["coverage"].update(inventory_entries=True),
    lambda value: value["coverage"].update(inferred_rows=3),
    lambda value: value["coverage"].update(evidence_unknown_members=1),
    lambda value: value["entry_evidence"].pop(),
    lambda value: value["entry_evidence"].reverse(),
    lambda value: value["entry_evidence"].append(deepcopy(value["entry_evidence"][0])),
    lambda value: value["head"].update(generation=2),
    lambda value: value["model"].update(state_sha256="Z" * 64),
    lambda value: value["model"].update(state_sha256=0.5),
    lambda value: value["query"].update(epoch=True),
    lambda value: value["query"].update(complete=False),
    lambda value: value["query"].update(page_cids=[]),
    lambda value: value["query"].update(extra="undeclared"),
])
def test_reference_identity_coverage_and_finite_scope_are_closed(mutation):
    head, artifact, refs = reference_case()
    mutation(refs)
    with pytest.raises(adapter.CodebaseInventoryEvidenceContextError):
        adapter._validate_refs(refs, artifact_cid=artifact, head=head)


def test_wrapper_binds_advisory_metadata_and_preserves_every_runtime_requirement_and_task(protocol):
    original = freeze_plan_create_input_snapshot(protocol.request, materials=protocol.materials)
    result = protocol.run()
    assert result["declared_requirement_ids"] == list(protocol.materials.intent.goal_predicate_ids)
    assert result["declared_task_ids"] == [item.candidate_id for item in protocol.materials.task_candidates]
    assert [row["predicate_id"] for row in result["residual_requirements"]] == result["declared_requirement_ids"]
    assert all(row["status"] == "runtime_behavior_unresolved" for row in result["residual_requirements"])
    assert result["current_facts"] == result["removed_task_ids"] == []
    assert result["training_steps"] == result["inference_calls"] == 0
    assert all(flag is False for flag in result["authority"].values())
    assert result[adapter.MATERIAL_KEY] == protocol.refs
    assert result["codebase_inventory_evidence_cid"] == cid_for_structured(protocol.refs)
    assert cid_for_structured({key: value for key, value in result.items() if key != "result_cid"}) == result["result_cid"]
    assert freeze_plan_create_input_snapshot(protocol.request, materials=protocol.materials) == original
    assert adapter.MATERIAL_KEY not in protocol.materials.extra
    assert [event[0] for event in protocol.events] == ["admit", "validate", "validate", "release"]
    controls = [event[1] for event in protocol.events if event[0] == "validate"]
    assert all(item["parent_lease"] is protocol.lease and "scheduler" not in item for item in controls)
    assert controls[1]["timeout_seconds"] <= controls[0]["timeout_seconds"]


def test_refs_change_semantic_material_identity_without_intent_or_task_change(protocol):
    baseline = protocol.run()
    protocol.refs["query"]["epoch"] += 1
    updated = protocol.run()
    assert baseline["repository_preview"]["input_snapshot"]["snapshot_cid"] != updated["repository_preview"]["input_snapshot"]["snapshot_cid"]
    assert baseline["declared_requirement_ids"] == updated["declared_requirement_ids"]
    assert baseline["declared_task_ids"] == updated["declared_task_ids"]


def test_zero_page_partial_refs_are_safe_planning_metadata_and_leave_every_task_residual(protocol):
    full = protocol.run()
    protocol.refs["query"].update(complete=False, next_cursor=None, page_cids=[])
    protocol.refs["coverage"].update(evidence_entries=0, evidence_matched_members=0,
                                     evidence_complete_absent_members=0, evidence_unknown_members=2)
    for row in protocol.refs["entry_evidence"]:
        row.update(evidence_disposition="unknown_budget", evidence_entry_ids=[])
    partial = protocol.run()
    assert partial["declared_task_ids"] == full["declared_task_ids"]
    assert partial["residual_requirements"] == full["residual_requirements"]
    assert partial["repository_preview"]["input_snapshot"]["snapshot_cid"] != full["repository_preview"]["input_snapshot"]["snapshot_cid"]
    assert partial["current_facts"] == partial["removed_task_ids"] == []


@pytest.mark.parametrize("container", ["extra", "candidate_context"])
def test_caller_reserved_key_collision_rejects_before_admission(protocol, container):
    getattr(protocol.materials, container)[adapter.MATERIAL_KEY] = protocol.refs
    with pytest.raises(adapter.CodebaseInventoryEvidenceContextError, match="collision"):
        protocol.run()
    assert protocol.events == []


def test_policy_callback_cannot_mutate_independently_authored_materials(protocol):
    def policy(value):
        protocol.materials.extra["caller_review"] = "mutated"
        return value.roots
    with pytest.raises(adapter.CodebaseInventoryEvidenceContextError, match="authored planning inputs changed"):
        protocol.run(policy_observer=policy)
    assert protocol.events[-1][0] == "release"


def test_closing_current_validator_withholds_preview_after_same_head_epoch_change(protocol):
    calls = []
    def validate(record, index, repository, **controls):
        calls.append(controls)
        if len(calls) == 2:
            raise ValueError("sealed evidence inventory epoch changed")
        return record
    protocol.module.validate_current_inventory_evidence = validate
    with pytest.raises(ValueError, match="epoch changed"):
        protocol.run()
    assert len(calls) == 2 and protocol.events[-1][0] == "release"


def test_cancelled_call_never_enters_admission(protocol):
    signal = threading.Event()
    signal.set()
    with pytest.raises(LeaseCancelledError):
        protocol.run(cancel_event=signal)
    assert protocol.events == []


@pytest.mark.parametrize("tamper", ["receipt", "snapshot", "material", "authority", "bool_alias"])
def test_wrong_preview_receipt_or_consumed_material_cannot_escape(protocol, monkeypatch, tamper):
    def wrong_preview(**arguments):
        value = native_memory_preview(**arguments)
        if tamper == "receipt":
            value["preview"]["receipt_cid"] = cid("wrong-receipt")
        elif tamper == "snapshot":
            value["input_snapshot"]["snapshot_cid"] = cid("wrong-snapshot")
        elif tamper == "material":
            arguments["materials"].extra[adapter.MATERIAL_KEY]["query"]["epoch"] += 1
        elif tamper == "authority":
            value["proof_authority"] = True
        else:
            value["proof_authority"] = 0
        return value
    monkeypatch.setattr(adapter, "preview_repository_plan", wrong_preview)
    with pytest.raises(adapter.CodebaseInventoryEvidenceContextError):
        protocol.run()
    assert protocol.events[-1][0] == "release"
