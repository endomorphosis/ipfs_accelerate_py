"""Administrative symbolic coverage keeps source truth and execution pending."""
from copy import deepcopy
import hashlib
import json

import pytest

from test.api.test_intent_plan_coverage import _ledger
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.planning.intent_requirement_adapter import (
    INTERPRETATION_SCOPE, IntentRequirementAdapterError,
    build_intent_planning_materials, validate_symbolic_operations,
)
from ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler import (
    SemanticSupport, compile_obligation_graph,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_critic import PlanCritic
from ipfs_accelerate_py.agent_supervisor.planning.symbolic_candidate_planner import (
    SymbolicCandidateBounds, SymbolicCandidatePlanner,
)
from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import build_intent_requirement_contract


def fixture(*, two=False):
    """Caller-verified manifest declarations; no signature or observation claim."""
    source, ledger = _ledger(("required", "required") if two else ("required",),
                             ("answer", "result") if two else ("answer",))
    groundings, operations, specs = [], [], []
    for index, requirement in enumerate(ledger["requirements"]):
        obj = "answer" if index == 0 else "result"
        output = {"path": obj + ".py", "effect": "modify", "media_type": "text/x-python"}
        validation_key = "validation:" + obj
        groundings.append({"requirement_id": requirement["requirement_id"], "outputs": [deepcopy(output)],
            "validation_keys": [validation_key], "dependency_requirement_ids": []})
        native = ledger["source_report"]["candidates"][index]["candidate_intent_ir"]["statements"][0]
        operations.append({"operation_id": "operation:" + obj, "task_key": "task:" + obj,
            "matchers": [{"requirement_id": requirement["requirement_id"],
                "native_document_sha256": requirement["native_document_sha256"],
                "statement_id": native["statement_id"], "predicate": native["predicate"],
                "arguments": list(native["arguments"]), "modality": native["modality"]}],
            "outputs": [deepcopy(output)], "validation_keys": [validation_key], "dependency_operation_ids": []})
        specs.append({"task_key": "task:" + obj, "scope_paths": [obj + ".py"],
            "outputs": [deepcopy(output)], "dependencies": [], "validations": [{"validation_key": validation_key,
                "argv": ["python", "test_" + obj + ".py"], "cwd": ".", "expected_exit_codes": [0],
                "policy_cid": cid_for_dag_json({"policy": "test-local"})}],
            "acceptance": [{"criterion_key": "criterion:" + obj,
                "criterion": "Reviewed public check passes", "validation_keys": [validation_key], "evidence_cids": []}]})
    symbolic = {"schema": "intent-symbolic-operation-contract@1", "ledger_sha256": ledger["ledger_sha256"],
        "review_ref": "reviewed:adapter-fixture@1", "interpretation_scope": INTERPRETATION_SCOPE,
        "operations": operations, "semantic_alignment_verified": False, "proof_authority": False,
        "execution_authority": False, "completion_authority": False}
    return source, ledger, groundings, symbolic, specs


def prepared(*, two=False, ordered=False):
    source, ledger, groundings, symbolic, specs = fixture(two=two)
    if ordered:
        groundings[1]["dependency_requirement_ids"] = [groundings[0]["requirement_id"]]
        symbolic["operations"][1]["dependency_operation_ids"] = [symbolic["operations"][0]["operation_id"]]
        specs[1]["dependencies"] = [specs[0]["task_key"]]
    contract = build_intent_requirement_contract(source_path="instruction.txt", ledger=ledger,
        requirements=groundings, source_text=source, symbolic_operations=symbolic)
    payload = {"schema": "supervisor-local-benchmark-manifest@4", "tasks": specs,
        "sources": {"instruction.txt": {"sha256": ledger["source"]["sha256"],
            "bytes": len(source.encode()), "executable": False}},
        "intent_requirements": {"schema": "supervisor-local-intent-requirement-artifact@1",
            "contract_json": json.dumps(contract, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False, allow_nan=False),
            "contract_cid": cid_for_dag_json(contract)}}
    return contract, {"payload": payload, "signature": "caller-verifies-signature"}


def test_exact_reviewed_native_operation_generates_real_open_symbolic_obligations():
    contract, manifest = prepared()
    materials = build_intent_planning_materials(contract, manifest=manifest)
    assert materials.current_facts == ()
    assert all(item.support is SemanticSupport.REVIEWED for item in materials.predicates)
    assert all(item.predicate_type == INTERPRETATION_SCOPE for item in materials.predicates)
    assert materials.predicates[0].subject_ref == cid_for_dag_json(contract)
    assert materials.predicates[0].object_ref == contract["requirements"][0]["requirement_id"]
    graph = compile_obligation_graph(materials.intent, current_facts=materials.current_facts,
        producers=materials.producers, task_candidates=materials.task_candidates, predicates=materials.predicates)
    assert not graph.planning_blocked
    assert not graph.review_required
    assert not any(item.status.value == "discharged" for item in graph.nodes)
    portfolio = SymbolicCandidatePlanner(bounds=SymbolicCandidateBounds(
        candidate_count=1, max_model_candidates=0)).plan(
            graph, materials.frozen_goal, materials.candidate_context, allow_model=False)
    assert portfolio.selected is not None
    assert portfolio.provider_usage.attempted is False
    assert set(portfolio.selected.symbolic_candidate.task_candidate_ids) == set(materials.candidate_task_keys)
    critique = PlanCritic().critique(portfolio.to_dict(), obligation_graph=graph)
    assert critique.accepted, critique.to_dict()
    receipt = materials.to_dict()
    assert receipt["materials_cid"]
    assert receipt["current_facts"] == []
    assert receipt["source_semantics_verified"] is False
    assert receipt["observed_source_facts"] is False
    for flag in ("semantic_alignment_verified", "proof_authority", "execution_authority", "completion_authority"):
        assert receipt[flag] is False


def test_grounded_dependency_becomes_and_premise_and_concrete_task_dependency():
    contract, manifest = prepared(two=True, ordered=True)
    materials = build_intent_planning_materials(contract, manifest=manifest)
    producer_by_id = {item.producer_id: item for item in materials.producers}
    first_id = materials.operation_candidate_ids["operation:answer"]
    second_id = materials.operation_candidate_ids["operation:result"]
    candidates = {item.candidate_id: item for item in materials.task_candidates}
    assert candidates[second_id].depends_on_candidate_ids == (first_id,)
    second_producer = producer_by_id[candidates[second_id].producer_id]
    first_producer = producer_by_id[candidates[first_id].producer_id]
    assert second_producer.required_predicate_ids == first_producer.effect_predicate_ids
    graph = compile_obligation_graph(materials.intent, producers=materials.producers,
        task_candidates=materials.task_candidates, predicates=materials.predicates)
    portfolio = SymbolicCandidatePlanner(bounds=SymbolicCandidateBounds(
        candidate_count=1, max_model_candidates=0)).plan(
            graph, materials.frozen_goal, materials.candidate_context, allow_model=False)
    assert portfolio.selected is not None
    assert portfolio.selected.symbolic_candidate.schedule.waves == ((first_id,), (second_id,))
    assert PlanCritic().critique(portfolio.to_dict(), obligation_graph=graph).accepted


@pytest.mark.parametrize("key,value", [
    ("predicate", "delete"), ("arguments", ["agent", "secret"]), ("modality", "permitted"),
    ("native_document_sha256", "0" * 64), ("statement_id", "unknown"),
])
def test_primitive_native_matcher_erasure_is_rejected(key, value):
    _, ledger, groundings, symbolic, _ = fixture()
    symbolic["operations"][0]["matchers"][0][key] = value
    with pytest.raises(IntentRequirementAdapterError, match="exact native atom"):
        validate_symbolic_operations(symbolic, ledger=ledger, requirements=groundings)


def test_same_source_with_a_distinct_reviewed_predicate_requires_a_new_exact_matcher():
    from ipfs_datasets_py.logic.intent_ir.formalize.requirements import build_intent_requirement_ledger
    source, old_ledger, groundings, symbolic, _ = fixture()
    report = deepcopy(old_ledger["source_report"])
    report["candidates"][0]["candidate_intent_ir"]["statements"][0]["predicate"] = "inspect"
    report.pop("report_sha256")
    report["report_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True,
        separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    ledger = build_intent_requirement_ledger(source, source_report=report,
        source_identity=old_ledger["source"]["source_identity"])
    assert ledger["source"] == old_ledger["source"]
    assert ledger["requirements"][0]["native_document_sha256"] != old_ledger["requirements"][0]["native_document_sha256"]
    assert ledger["semantic_alignment_verified"] is False
    symbolic["ledger_sha256"] = ledger["ledger_sha256"]
    matcher = symbolic["operations"][0]["matchers"][0]
    matcher.update(requirement_id=ledger["requirements"][0]["requirement_id"],
                   native_document_sha256=ledger["requirements"][0]["native_document_sha256"])
    groundings[0]["requirement_id"] = matcher["requirement_id"]
    with pytest.raises(IntentRequirementAdapterError, match="exact native atom"):
        validate_symbolic_operations(symbolic, ledger=ledger, requirements=groundings)
    matcher["predicate"] = "inspect"
    assert validate_symbolic_operations(symbolic, ledger=ledger, requirements=groundings)["operations"][0]["matchers"][0]["predicate"] == "inspect"


@pytest.mark.parametrize("mode", ["prohibited", "permitted", "guard", "conditional"])
def test_symbolic_atomic_profile_rejects_norms_and_context_instead_of_flattening(mode):
    _, old_ledger, groundings, symbolic, _ = fixture()
    _, ledger = _ledger((mode,) if mode in {"prohibited", "permitted"} else ("required",),
        statement_kind="guard" if mode == "guard" else "goal", native_context=mode == "conditional")
    symbolic["ledger_sha256"] = ledger["ledger_sha256"]
    with pytest.raises(IntentRequirementAdapterError, match="atomic mandatory goal"):
        validate_symbolic_operations(symbolic, ledger=ledger, requirements=groundings)


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(execution_authority=True),
    lambda s: s.update(observed_facts=[]),
    lambda s: s.update(ledger_sha256="0" * 64),
    lambda s: s.update(review_ref=""),
    lambda s: s.update(interpretation_scope="verified_code_effects"),
    lambda s: s["operations"][0]["outputs"][0].update(effect="delete"),
    lambda s: s["operations"][0].update(validation_keys=[]),
    lambda s: s["operations"][0]["matchers"].append(deepcopy(s["operations"][0]["matchers"][0])),
    lambda s: s["operations"].append(deepcopy(s["operations"][0])),
    lambda s: s["operations"][0].update(dependency_operation_ids=["missing"]),
])
def test_malformed_or_escalated_operation_contract_is_rejected(mutate):
    _, ledger, groundings, symbolic, _ = fixture()
    mutate(symbolic)
    with pytest.raises(IntentRequirementAdapterError):
        validate_symbolic_operations(symbolic, ledger=ledger, requirements=groundings)


def test_grounded_ordering_cannot_be_dropped_or_turned_into_cycle():
    _, ledger, groundings, symbolic, _ = fixture(two=True)
    groundings[1]["dependency_requirement_ids"] = [groundings[0]["requirement_id"]]
    with pytest.raises(IntentRequirementAdapterError, match="ordering"):
        validate_symbolic_operations(symbolic, ledger=ledger, requirements=groundings)
    symbolic["operations"][1]["dependency_operation_ids"] = ["operation:answer"]
    symbolic["operations"][0]["dependency_operation_ids"] = ["operation:result"]
    with pytest.raises(IntentRequirementAdapterError, match="cycle"):
        validate_symbolic_operations(symbolic, ledger=ledger, requirements=groundings)


def test_requirement_cannot_be_assigned_to_multiple_independent_operations():
    _, ledger, groundings, symbolic, _ = fixture()
    duplicate = deepcopy(symbolic["operations"][0])
    duplicate.update(operation_id="operation:second", task_key="task:second")
    symbolic["operations"].append(duplicate)
    with pytest.raises(IntentRequirementAdapterError, match="one unique operation per requirement"):
        validate_symbolic_operations(symbolic, ledger=ledger, requirements=groundings)


def test_multiple_exact_requirements_can_share_one_independently_grounded_task():
    source, ledger, groundings, symbolic, specs = fixture(two=True)
    first, second = symbolic["operations"]
    first["matchers"].extend(second["matchers"])
    first["outputs"].extend(second["outputs"])
    first["validation_keys"].extend(second["validation_keys"])
    symbolic["operations"] = [first]
    specs[0]["scope_paths"].extend(specs[1]["scope_paths"])
    for field in ("outputs", "validations", "acceptance"):
        specs[0][field].extend(specs[1][field])
    contract = build_intent_requirement_contract(source_path="instruction.txt", ledger=ledger,
        requirements=groundings, source_text=source, symbolic_operations=symbolic)
    manifest = {"schema": "supervisor-local-benchmark-manifest@4", "tasks": [specs[0]],
        "sources": {"instruction.txt": {"sha256": ledger["source"]["sha256"],
            "bytes": len(source.encode()), "executable": False}},
        "intent_requirements": {"schema": "supervisor-local-intent-requirement-artifact@1",
            "contract_json": json.dumps(contract, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False, allow_nan=False),
            "contract_cid": cid_for_dag_json(contract)}}
    materials = build_intent_planning_materials(contract, manifest=manifest)
    assert len(materials.producers) == len(materials.task_candidates) == 1
    assert len(materials.producers[0].effect_predicate_ids) == len(materials.predicates) == 2
    assert len(next(iter(materials.candidate_requirement_ids.values()))) == 2
    assert len(materials.requirements) == 2


@pytest.mark.parametrize("change", ["population", "effect", "validation", "acceptance", "dependency", "scope", "contract"])
def test_signed_manifest_task_declarations_remain_independent_constraints(change):
    contract, manifest = prepared()
    spec = manifest["payload"]["tasks"][0]
    if change == "population":
        spec["task_key"] = "foreign-task"
    elif change == "effect":
        spec["outputs"][0]["effect"] = "delete"
    elif change == "validation":
        spec["validations"][0]["validation_key"] = "foreign-validation"
    elif change == "acceptance":
        spec["acceptance"][0]["validation_keys"] = []
    elif change == "dependency":
        spec["dependencies"] = ["foreign-task"]
    elif change == "scope":
        spec["scope_paths"] = ["unrelated.py"]
    else:
        manifest["payload"]["intent_requirements"]["contract_cid"] = cid_for_dag_json({"foreign": True})
    with pytest.raises(ValueError):
        build_intent_planning_materials(contract, manifest=manifest)


def test_pure_material_replay_does_not_observe_repository_or_create_source_facts():
    contract, manifest = prepared()
    first = build_intent_planning_materials(contract, manifest=manifest).to_dict()
    # A post-execution caller can supply an already verified publication
    # transition. The adapter uses declared immutable baseline sources.
    manifest["payload"]["repository"] = "/this/path/does/not/exist"
    second = build_intent_planning_materials(contract, manifest=manifest).to_dict()
    assert first["current_root_id"] == second["current_root_id"]
    assert first["manifest_cid"] != second["manifest_cid"]
    assert first["current_facts"] == second["current_facts"] == []


def test_candidate_context_is_frozen_and_proof_is_explicitly_not_required():
    contract, manifest = prepared()
    materials = build_intent_planning_materials(contract, manifest=manifest)
    assert materials.frozen_goal.policy.require_proof is False
    assert materials.frozen_goal.policy.trusted_assumptions == ()
    assert not any(predicate.proof_requirement_refs for predicate in materials.predicates)
    with pytest.raises(TypeError):
        materials.candidate_context["task_metadata"]["forged"] = {}
