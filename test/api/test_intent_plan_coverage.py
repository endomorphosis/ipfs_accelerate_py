"""Negative controls for explicit source requirements and graph bindings."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from ipfs_datasets_py.logic.intent_ir.formalize.requirements import build_intent_requirement_ledger
from ipfs_datasets_py.logic.intent_ir.formalize.roundtrip import frame_to_intent_ir
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import (
    INTENT_PLAN_PROPOSAL_SCHEMA,
    IntentPlanCoverageError,
    build_intent_plan_provider_request,
    build_intent_requirement_contract,
    check_intent_plan_coverage,
    parse_intent_plan_proposal,
    validate_intent_requirement_contract,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    PromptAcceptanceRecord, PromptGoalGraph, PromptGoalRecord, PromptOutputRecord,
    PromptTaskRecord, PromptValidationRecord, prompt_workflow_cid,
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _ledger(modalities=("required",), objects=("answer",), *, statement_kind="goal", native_context=False):
    clauses = [f"agent {dict(required='must', intended='intends to', prohibited='must not', permitted='may')[modality]} modify {obj}."
               for modality, obj in zip(modalities, objects)]
    text = "\n".join(clauses)
    units, candidates, left = [], [], 0
    for index, (clause, modality, obj) in enumerate(zip(clauses, modalities, objects)):
        key = f"source:{index}"
        units.append({"unit_id": key, "start_char": left, "end_char": left + len(clause),
                      "start_byte": left, "end_byte": left + len(clause),
                      "text": clause, "sha256": _sha(clause.encode()),
                      "disposition": "interpreted_candidate", "reason": "reviewed fixture"})
        native = frame_to_intent_ir({"actor": "agent", "action": "modify", "object": obj,
                                    "modality": modality}, instruction=clause)
        document = native.to_dict()
        if statement_kind != "goal":
            document["statements"].append({**document["statements"][0],
                                           "statement_id": "role:" + statement_kind,
                                           "kind": statement_kind})
        if native_context:
            document["statements"].append({**document["statements"][0],
                                           "statement_id": "guard", "kind": "guard",
                                           "modality": "asserted"})
            document["actions"].append({**document["actions"][0], "action_id": "conditional-action"})
            document["control_edges"] = [{
                "edge_id": "conditional-edge", "source_action_id": "action",
                "target_action_id": "conditional-action", "kind": "conditional",
                "guard_statement_id": "guard", "source_ref_ids": ["source"],
                "grounding": "inferred",
            }]
            document["terminal_action_ids"] = ["conditional-action"]
        candidates.append({"unit_id": key, "candidate_intent_ir": document})
        left += len(clause) + 1
    report = {"schema": "intent-reviewed-source-report@1", "source_sha256": _sha(text.encode()),
              "source_bytes": len(text.encode()), "source_characters": len(text),
              "producer": {"name": "explicit-test-fixture", "revision": "1"},
              "interpretation_status": "reviewed_candidate", "units": units, "candidates": candidates,
              "proof_authority": False, "execution_authority": False,
              "completion_authority": False, "source_semantics_verified": False}
    report["report_sha256"] = _sha(_wire(report))
    return text, build_intent_requirement_ledger(text, source_report=report,
                                                source_identity="fixture:instruction")


def _spec(requirement_id, path="answer.py", *, validation_keys=("validate:answer",), effect="modify"):
    return {"requirement_id": requirement_id,
            "outputs": [{"path": path, "effect": effect, "media_type": "text/x-python"}],
            "validation_keys": list(validation_keys), "dependency_requirement_ids": []}


def _contract(modalities=("required",), objects=("answer",)):
    text, ledger = _ledger(modalities, objects)
    specs = []
    for req, obj in zip(ledger["requirements"], objects):
        if req["modality"] == "permitted":
            continue
        specs.append(_spec(req["requirement_id"], obj + ".py",
                           validation_keys=() if req["modality"] == "prohibited" else ("validate:" + obj,)))
    return build_intent_requirement_contract(source_path="instruction.txt", ledger=ledger,
                                            requirements=specs, source_text=text)


def _graph(*, effect="modify", path="answer.py", media_type="text/x-python", linked=True):
    cid = prompt_workflow_cid({"fixture": "coverage"})
    acceptance = PromptAcceptanceRecord(criterion_key="criterion:answer", criterion="Answer check passes.",
                                       validation_keys=("validate:answer",) if linked else ())
    goal = PromptGoalRecord(goal_key="goal:answer", parent_goal_cid="", dependency_goal_cids=(),
                           title="Modify answer", objective="Modify answer", rationale="Explicit requirement",
                           scope_paths=("answer.py",), acceptance=(acceptance,))
    task = PromptTaskRecord(task_key="task:answer", goal_cid=goal.goal_cid, dependency_task_cids=(),
                            objective="Modify answer", rationale="Explicit requirement", scope_paths=("answer.py",),
                            outputs=(PromptOutputRecord(path=path, effect=effect, media_type=media_type),),
                            validations=(PromptValidationRecord(validation_key="validate:answer",
                                                                 argv=("python", "-m", "pytest")),),
                            acceptance=(acceptance,), evidence_cids=(), policy_roots=(cid,))
    return PromptGoalGraph(request_cid=cid, scan_cid=cid, program_root=cid, policy_roots=(cid,),
                           goals=(goal,), tasks=(task,), evidence=())


def _bindings(contract):
    return [{"requirement_id": contract["ledger"]["requirements"][0]["requirement_id"],
             "task_keys": ["task:answer"], "validation_keys": ["validate:answer"]}]


def test_valid_coverage_binds_task_and_contract_identities_without_authority():
    contract, graph = _contract(), _graph()
    receipt = check_intent_plan_coverage(contract, graph=graph, bindings=_bindings(contract))
    assert receipt["accepted"] is True
    assert receipt["contract_cid"] == cid_for_dag_json(contract)
    assert receipt["graph_cid"] == graph.plan_root_cid
    assert receipt["ledger_sha256"] == contract["ledger"]["ledger_sha256"]
    assert receipt["bindings"][0]["task_cids"] == [graph.tasks[0].task_cid]
    assert receipt["semantic_support_complete"] is False
    for key in ("semantic_alignment_verified", "proof_authority", "execution_authority", "completion_authority"):
        assert receipt[key] is False


def test_missing_requirement_binding_retains_omission_and_orphan_evidence():
    contract = _contract()
    receipt = check_intent_plan_coverage(contract, graph=_graph(), bindings=[])
    assert receipt["accepted"] is False
    assert receipt["uncovered_requirement_ids"] == [contract["ledger"]["requirements"][0]["requirement_id"]]
    assert "orphan_task" in {item["code"] for item in receipt["errors"]}


@pytest.mark.parametrize("changes", [{"effect": "delete"}, {"path": "wrong.py"}, {"media_type": "text/plain"}, {"linked": False}])
def test_requirement_name_alone_does_not_cover_grounded_obligations(changes):
    contract = _contract()
    receipt = check_intent_plan_coverage(contract, graph=_graph(**changes), bindings=_bindings(contract))
    assert receipt["accepted"] is False
    assert receipt["uncovered_requirement_ids"]


@pytest.mark.parametrize("change", ["unknown_requirement", "unknown_task", "unknown_validation", "duplicate_requirement", "duplicate_task"])
def test_malformed_or_unknown_bindings_are_rejected(change):
    contract = _contract()
    bindings = _bindings(contract)
    if change == "unknown_requirement":
        bindings[0]["requirement_id"] = "unknown"
    elif change == "unknown_task":
        bindings[0]["task_keys"] = ["unknown"]
    elif change == "unknown_validation":
        bindings[0]["validation_keys"] = ["unknown"]
    elif change == "duplicate_requirement":
        bindings *= 2
    elif change == "duplicate_task":
        bindings[0]["task_keys"] *= 2
    with pytest.raises(IntentPlanCoverageError):
        check_intent_plan_coverage(contract, graph=_graph(), bindings=bindings)


def test_permitted_statement_does_not_create_execution_obligation():
    contract = _contract(("required", "permitted"), ("answer", "secret"))
    receipt = check_intent_plan_coverage(contract, graph=_graph(), bindings=_bindings(contract))
    assert receipt["accepted"] is True
    permitted = contract["ledger"]["requirements"][1]["requirement_id"]
    bindings = _bindings(contract) + [{"requirement_id": permitted, "task_keys": ["task:answer"], "validation_keys": []}]
    with pytest.raises(IntentPlanCoverageError, match="nonmandatory"):
        check_intent_plan_coverage(contract, graph=_graph(), bindings=bindings)


def test_prohibited_effect_is_checked_with_reviewed_grounding():
    contract = _contract(("required", "prohibited"), ("answer", "answer"))
    receipt = check_intent_plan_coverage(contract, graph=_graph(), bindings=_bindings(contract))
    assert receipt["accepted"] is False
    assert receipt["prohibited_output_violations"][0]["path"] == "answer.py"


def test_ungrounded_prohibition_is_visible_and_rejected():
    contract = _contract(("required", "prohibited"), ("answer", "secret"))
    contract["requirements"] = [item for item in contract["requirements"]
                                if item["validation_keys"]]
    receipt = check_intent_plan_coverage(contract, graph=_graph(), bindings=_bindings(contract))
    assert receipt["accepted"] is False
    assert receipt["unsupported_requirement_ids"]


def test_forged_normative_polarity_and_source_are_rejected():
    contract = _contract()
    forged = json.loads(json.dumps(contract))
    forged["ledger"]["requirements"][0]["modality"] = "permitted"
    with pytest.raises(IntentPlanCoverageError, match="ledger"):
        validate_intent_requirement_contract(forged)
    with pytest.raises(IntentPlanCoverageError, match="ledger"):
        validate_intent_requirement_contract(contract, source_text="agent may modify answer.")


def test_contract_must_ground_every_mandatory_requirement():
    contract = _contract()
    contract["requirements"] = []
    with pytest.raises(IntentPlanCoverageError, match="lacks independent grounding"):
        validate_intent_requirement_contract(contract)


def test_dependency_ordering_is_checked_against_task_cids():
    contract = _contract(("required", "required"), ("answer", "report"))
    answer_id, report_id = [item["requirement_id"] for item in contract["ledger"]["requirements"]]
    for spec in contract["requirements"]:
        if spec["requirement_id"] == report_id:
            spec["dependency_requirement_ids"] = [answer_id]
    graph = _graph()
    task = graph.tasks[0]
    report_check = PromptAcceptanceRecord(criterion_key="criterion:report", criterion="Report check passes.",
                                         validation_keys=("validate:report",))
    report_task = replace(task, task_key="task:report", scope_paths=("report.py",),
                          outputs=(PromptOutputRecord(path="report.py", effect="modify", media_type="text/x-python"),),
                          validations=(PromptValidationRecord(validation_key="validate:report", argv=("python", "-m", "pytest")),),
                          acceptance=(report_check,))
    bindings = _bindings(contract) + [{"requirement_id": report_id, "task_keys": ["task:report"],
                                      "validation_keys": ["validate:report"]}]
    receipt = check_intent_plan_coverage(contract, graph=replace(graph, tasks=(task, report_task)), bindings=bindings)
    assert receipt["accepted"] is False
    assert "missing_requirement_ordering" in {item["code"] for item in receipt["errors"]}
    report_task = replace(report_task, dependency_task_cids=(task.task_cid,))
    assert check_intent_plan_coverage(contract, graph=replace(graph, tasks=(task, report_task)), bindings=bindings)["accepted"] is True


def test_proposal_unwrap_preserves_inner_graph_and_rejects_stale_contract():
    contract = _contract()
    inner = {"schema": "opaque-inner-graph", "nested": {"value": 7}}
    envelope = {"schema": INTENT_PLAN_PROPOSAL_SCHEMA, "contract_cid": cid_for_dag_json(contract),
                "graph_proposal": inner, "requirement_bindings": _bindings(contract)}
    graph_text, bindings = parse_intent_plan_proposal(json.dumps(envelope), contract=contract)
    assert json.loads(graph_text) == inner
    assert bindings == _bindings(contract)
    envelope["contract_cid"] = prompt_workflow_cid({"fixture": "stale"})
    with pytest.raises(IntentPlanCoverageError, match="stale or mismatched"):
        parse_intent_plan_proposal(json.dumps(envelope), contract=contract)


def test_provider_wrapper_retains_original_request_and_explicit_groundings():
    contract = _contract()
    original = {"schema": "graph-request", "context": {"existing": True}}
    wrapped = json.loads(build_intent_plan_provider_request(json.dumps(original), contract))
    assert wrapped["graph_request"] == original
    assert wrapped["intent_contract"] == contract
    assert wrapped["contract_cid"] == cid_for_dag_json(contract)
    assert wrapped["execution_authority"] is False


def test_proposal_rejects_malformed_contract_identity_even_without_expected_contract():
    envelope = {"schema": INTENT_PLAN_PROPOSAL_SCHEMA, "contract_cid": "invented",
                "graph_proposal": {}, "requirement_bindings": []}
    with pytest.raises(IntentPlanCoverageError, match="canonical contract CID"):
        parse_intent_plan_proposal(json.dumps(envelope))


@pytest.mark.parametrize("text", ['{"schema":1,"schema":2}', '{"value":NaN}', '```json\n{}\n```'])
def test_provider_wrapper_requires_strict_json(text):
    with pytest.raises(IntentPlanCoverageError):
        build_intent_plan_provider_request(text, _contract())


def test_rich_compound_remains_unsupported_in_execution_coverage():
    from unittest.mock import patch
    from ipfs_datasets_py.logic.intent_ir.formalize import rich_document
    from ipfs_datasets_py.logic.intent_ir.formalize.rich_grammar import parse_instruction

    text = "agent must modify answer and agent must modify report."
    ast = parse_instruction(text)
    inference = {"status": "semantic_candidate_advice",
                 "rich_ir": {"schema": "intent-rich-ir/v1", "source_sha256": _sha(text.encode()), "ast": ast},
                 "logic": {"native_intent_ir": None, "projections": []},
                 "counts": {"encoder_executions": 1, "decoder_executions": 1},
                 "report_sha256": "b" * 64}
    # Supply an explicit candidate through the real report producer without
    # any model inference, download or external call.
    with patch.object(rich_document, "prepare_rich_intent_instruction", return_value=inference):
        report = rich_document.prepare_rich_intent_document(
            text, {"schema": "intent-rich-copy-checkpoint/v1"}, requested_families=["first_order"])
    ledger = build_intent_requirement_ledger(text, source_report=report, source_identity="fixture:compound")
    contract = build_intent_requirement_contract(source_path="instruction.txt", ledger=ledger, requirements=[])
    receipt = check_intent_plan_coverage(contract, graph=_graph(), bindings=[])
    assert receipt["accepted"] is False
    assert receipt["unsupported_requirement_ids"] == [ledger["requirements"][0]["requirement_id"]]
    assert ledger["requirements"][0]["compound"]["ast"] == ast


@pytest.mark.parametrize("kind", ["precondition", "guard", "effect", "verification", "postcondition", "invariant", "assumption", "failure"])
def test_non_goal_statement_role_cannot_become_unconditional_execution_coverage(kind):
    text, ledger = _ledger(statement_kind=kind)
    goal = next(requirement for requirement in ledger["requirements"] if requirement["kind"] == "goal")
    non_goal = next(requirement for requirement in ledger["requirements"] if requirement["kind"] == kind)
    contract = build_intent_requirement_contract(source_path="instruction.txt", ledger=ledger,
                                                requirements=[_spec(goal["requirement_id"])])
    receipt = check_intent_plan_coverage(contract, graph=_graph(), bindings=[{
        "requirement_id": goal["requirement_id"], "task_keys": ["task:answer"],
        "validation_keys": ["validate:answer"],
    }])
    assert receipt["accepted"] is False
    assert receipt["unsupported_requirement_ids"] == [non_goal["requirement_id"]]
    with pytest.raises(IntentPlanCoverageError, match="unsupported"):
        build_intent_requirement_contract(source_path="instruction.txt", ledger=ledger,
                                         requirements=[_spec(goal["requirement_id"]), _spec(non_goal["requirement_id"])],
                                         source_text=text)


def test_native_conditional_graph_retains_scope_and_cannot_bind_one_unconditional_task():
    text, ledger = _ledger(native_context=True)
    requirement = ledger["requirements"][0]
    assert requirement["compound"]["schema"] == "intent-native-requirement-scope@1"
    assert requirement["compound"]["control_edges"][0]["kind"] == "conditional"
    contract = build_intent_requirement_contract(source_path="instruction.txt", ledger=ledger, requirements=[])
    receipt = check_intent_plan_coverage(contract, graph=_graph(), bindings=[])
    assert receipt["accepted"] is False
    assert receipt["unsupported_requirement_ids"] == sorted(item["requirement_id"] for item in ledger["requirements"])
    with pytest.raises(IntentPlanCoverageError, match="unsupported"):
        build_intent_requirement_contract(source_path="instruction.txt", ledger=ledger,
                                         requirements=[_spec(requirement["requirement_id"])], source_text=text)
