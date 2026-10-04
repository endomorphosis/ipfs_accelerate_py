"""Actual symbolic selection is replayed through native local admission.

Reviewed fixture semantics remain candidate interpretations. These tests check
exact scheduling, formal output projection, owner replay and public acceptance.
"""
from copy import deepcopy

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_intent_requirement_admission import _author, _reviewed_contract
from test.api.test_local_planning_declared_create import _with_creation
from ipfs_accelerate_py.agent_supervisor.planning import intent_symbolic_planning as symbolic
from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import check_intent_plan_coverage
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentCompletionError


def _symbolic_contract(case, *, create=False, confidence=0):
    contract = _reviewed_contract(case, confidence=confidence)
    requirement = contract["ledger"]["requirements"][0]
    native = contract["ledger"]["source_report"]["candidates"][0]["candidate_intent_ir"]["statements"][0]
    if create:
        contract["requirements"][0]["outputs"].append(
            {"path": "report.jsonl", "effect": "create", "media_type": "application/json"})
    contract["schema"] = "intent-plan-requirement-contract@2"
    contract["symbolic_operations"] = {
        "schema": "intent-symbolic-operation-contract@1",
        "ledger_sha256": contract["ledger"]["ledger_sha256"],
        "review_ref": "review:public-answer@1",
        "interpretation_scope": "administrative_requirement_task_coverage",
        "operations": [{"operation_id": "operation:answer", "task_key": "LOCAL-TASK",
            "matchers": [{"requirement_id": requirement["requirement_id"],
                "native_document_sha256": requirement["native_document_sha256"],
                "statement_id": native["statement_id"], "predicate": native["predicate"],
                "arguments": list(native["arguments"]), "modality": native["modality"]}],
            "outputs": deepcopy(contract["requirements"][0]["outputs"]),
            "validation_keys": ["public-answer"], "dependency_operation_ids": []}],
        "semantic_alignment_verified": False, "proof_authority": False,
        "execution_authority": False, "completion_authority": False,
    }
    return contract


def _prepare(case, *, create=False, confidence=0):
    contract = _symbolic_contract(case, create=create, confidence=confidence)
    original = _with_creation(case)[1]["payload"] if create else None
    manifest = _author(case, contract, original=original)
    proposed = symbolic.build_intent_symbolic_plan(contract, manifest=manifest)
    return contract, manifest, proposed


@pytest.mark.parametrize("create", [False, True])
@pytest.mark.parametrize("confidence", [0, 0.75])
def test_real_symbolic_selection_projects_exact_signed_effects_and_replays(scenario, create, confidence):
    contract, manifest, proposed = _prepare(scenario, create=create, confidence=confidence)
    replay = symbolic.build_intent_symbolic_plan(contract, manifest=manifest)
    assert replay["graph"].to_dict() == proposed["graph"].to_dict()
    assert replay["receipt"] == proposed["receipt"]
    receipt = proposed["receipt"]
    assert receipt["provider_calls"] == 0
    assert receipt["observed_facts_supplied"] == 0
    assert receipt["critic_decision"] == "accepted"
    assert receipt["formal_plan_id"]
    assert all(receipt[key] is False for key in ("semantic_alignment_verified",
        "proof_authority", "execution_authority", "completion_authority", "source_semantics_verified"))
    outputs = [{key: getattr(output, key) for key in ("path", "effect", "media_type")}
               for output in proposed["graph"].tasks[0].outputs]
    assert sorted(outputs, key=lambda row: row["path"]) == sorted(manifest["payload"]["tasks"][0]["outputs"], key=lambda row: row["path"])
    admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=manifest,
        requirement_bindings=proposed["requirement_bindings"])
    assert admission["receipt"]["payload"]["intent_symbolic_planning"] == receipt
    materialized = local.materialize_local_benchmark_plan(admission=admission, intent=scenario["intent"])
    task = scenario["intent"].get_task(materialized["task_cids"][0])
    pending, _, _, _ = local._contract(task["body"], task["task_cid"])
    assert pending["intent_plan"]["schema"] == "supervisor-local-intent-plan@2"
    assert pending["intent_plan"]["symbolic_planning"] == receipt
    reference = scenario["intent"].get_plan(materialized["plan_id"])["body"]["local_planning_receipt_ref"]
    assert local.load_local_planning_receipt(reference, manifest=manifest) == admission["receipt"]


def test_coverage_correct_direct_graph_cannot_bypass_symbolic_selection(scenario):
    contract, manifest, proposed = _prepare(scenario)
    coverage = check_intent_plan_coverage(contract, graph=scenario["graph"],
        bindings=proposed["requirement_bindings"])
    assert coverage["accepted"]
    with pytest.raises(local.LocalPlanningError, match="deterministic symbolic selection"):
        local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=manifest,
            requirement_bindings=proposed["requirement_bindings"])


@pytest.mark.parametrize("field", ["drop", "provider_calls", "selected_operation_ids", "formal_effects_cid", "materials_cid"])
def test_owner_resigning_changed_selection_receipt_fails_replay(scenario, field):
    _, manifest, proposed = _prepare(scenario)
    admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=manifest,
        requirement_bindings=proposed["requirement_bindings"])
    forged = deepcopy(admission)
    payload = forged["receipt"]["payload"]
    if field == "drop":
        del payload["intent_symbolic_planning"]
    else:
        payload["intent_symbolic_planning"][field] = 1 if field == "provider_calls" else [] if field == "selected_operation_ids" else "forged"
    forged["receipt"] = local._signed(payload, manifest["payload"])
    with pytest.raises(local.LocalPlanningError, match="recomputed contract"):
        local.verify_local_benchmark_admission(forged)
    with pytest.raises(local.LocalPlanningError, match="symbolic selection"):
        local._store_local_planning_receipt(forged["receipt"], manifest=manifest)
    with pytest.raises(local.LocalPlanningError):
        local.materialize_local_benchmark_plan(admission=forged, intent=scenario["intent"])
    assert scenario["intent"].get_task(proposed["graph"].tasks[0].task_cid) is None


@pytest.mark.parametrize("mutation", ["drop", "coverage_schema", "alter_receipt", "alter_graph"])
def test_owner_resigned_native_contract_cannot_omit_or_forge_symbolic_replay(scenario, mutation):
    _, manifest, proposed = _prepare(scenario)
    admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=manifest,
        requirement_bindings=proposed["requirement_bindings"])
    materialized = local.materialize_local_benchmark_plan(admission=admission, intent=scenario["intent"])
    task = scenario["intent"].get_task(materialized["task_cids"][0])
    body = deepcopy(task["body"])
    payload = body[local.CONTRACT_KEY]["payload"]
    if mutation == "drop":
        del payload["intent_plan"]["symbolic_planning"]
    elif mutation == "coverage_schema":
        payload["intent_plan"]["schema"] = "supervisor-local-intent-plan@1"
        del payload["intent_plan"]["symbolic_planning"]
    elif mutation == "alter_receipt":
        payload["intent_plan"]["symbolic_planning"]["observed_facts_supplied"] = 1
    else:
        payload["intent_plan"]["graph"]["tasks"][0]["objective"] = "Changed proposal"
    body[local.CONTRACT_KEY] = local._signed(payload, manifest["payload"])
    with pytest.raises(ValueError):
        local._contract(body, task["task_cid"])


def test_symbolic_selection_still_requires_real_passing_public_check_and_created_output(scenario):
    _, manifest, proposed = _prepare(scenario, create=True)
    admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=manifest,
        requirement_bindings=proposed["requirement_bindings"])
    intent = scenario["intent"]
    cid = local.materialize_local_benchmark_plan(admission=admission, intent=intent)["task_cids"][0]
    intent.cas_task_status(task_cid=cid, expected_revision=intent.get_task(cid)["revision"], new_status="in_progress")
    failed = local.run_local_task_validations(intent=intent, task_cid=cid, attempt_id="symbolic:failed")
    assert not failed["passed"]
    with pytest.raises(IntentCompletionError):
        intent.cas_task_status(task_cid=cid, expected_revision=intent.get_task(cid)["revision"], new_status="completed")
    (scenario["repository"] / "answer.py").write_text("def answer():\n    return 2\n")
    assert local.run_local_task_validations(intent=intent, task_cid=cid, attempt_id="symbolic:missing-output")["passed"]
    with pytest.raises(IntentCompletionError):
        intent.cas_task_status(task_cid=cid, expected_revision=intent.get_task(cid)["revision"], new_status="completed")
    (scenario["repository"] / "report.jsonl").write_text('{"answer":2}\n')
    assert local.run_local_task_validations(intent=intent, task_cid=cid, attempt_id="symbolic:passed")["passed"]
    intent.cas_task_status(task_cid=cid, expected_revision=intent.get_task(cid)["revision"], new_status="completed")
    assert intent.get_task(cid)["status"] == "completed"


def test_failed_critic_never_returns_a_graph_or_admission(scenario, monkeypatch):
    contract = _symbolic_contract(scenario)
    manifest = _author(scenario, contract)
    class RejectingCritic:
        def critique(self, *args, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(accepted=False, decision=SimpleNamespace(value="rejected"), findings=())
    monkeypatch.setattr(symbolic, "PlanCritic", RejectingCritic)
    with pytest.raises(symbolic.IntentSymbolicPlanningError, match="independent critique"):
        symbolic.build_intent_symbolic_plan(contract, manifest=manifest)


def test_no_selected_candidate_never_uses_a_baseline_fallback(scenario, monkeypatch):
    contract = _symbolic_contract(scenario)
    manifest = _author(scenario, contract)
    class EmptyPlanner:
        def __init__(self, **kwargs):
            pass
        def plan(self, *args, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(selected=None)
    monkeypatch.setattr(symbolic, "SymbolicCandidatePlanner", EmptyPlanner)
    with pytest.raises(symbolic.IntentSymbolicPlanningError, match="no symbolic candidate"):
        symbolic.build_intent_symbolic_plan(contract, manifest=manifest)


def _native_case(scenario):
    contract, manifest, proposed = _prepare(scenario)
    admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=manifest,
        requirement_bindings=proposed["requirement_bindings"])
    return {**scenario, "requirements": contract, "intent_manifest": manifest,
        "graph": proposed["graph"], "bindings": proposed["requirement_bindings"],
        "task_cid": proposed["graph"].tasks[0].task_cid, "admission": admission}


def test_real_native_publication_replays_symbolic_selection_against_original_baseline(scenario, tmp_path):
    # Reuse the qualified real claim/Git publication/Portal observation exercise
    # with the new graph and symbolic pending contract instead of a model graph.
    from test.api.test_intent_requirement_admission import (
        test_v4_publication_requires_real_observation_then_allows_typed_completion as publication_exercise,
    )
    case = _native_case(scenario)
    before = case["admission"]["receipt"]["payload"]["intent_symbolic_planning"]
    publication_exercise(case, tmp_path)
    assert case["intent"].get_task(case["task_cid"])["status"] == "completed"
    # Owner validation above supplies the authorized publication transition.
    # Pure symbolic replay continues to use the signed original source tree.
    after = symbolic.build_intent_symbolic_plan(case["requirements"], manifest=case["intent_manifest"])
    assert after["receipt"] == before


def test_launch_refuses_owner_resigned_symbolic_receipt_in_original_native_row(scenario, tmp_path):
    from test.api.test_intent_requirement_admission import _native_intent_owner, _wire
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime

    case = _native_case(scenario)
    with _native_intent_owner(case, tmp_path) as owner:
        runtime = AdmittedBenchmarkRuntime()
        runtime.source, runtime.server, runtime.admission = owner.source, owner.server, case["admission"]
        verified = local.verify_local_benchmark_admission(case["admission"])
        runtime._verify_tasks(verified)
        original = owner.source.get_task(case["task_cid"])
        body = deepcopy(dict(original.body))
        payload = body[local.CONTRACT_KEY]["payload"]
        payload["intent_plan"]["symbolic_planning"]["selected_candidate_id"] = "forged-selection"
        body[local.CONTRACT_KEY] = local._signed(payload, case["intent_manifest"]["payload"])
        with owner.server._lock:
            owner.server._connection.execute("UPDATE tasks SET body_json = ? WHERE task_cid = ?",
                [_wire(body).decode("utf-8"), original.task_cid])
        with pytest.raises(ValueError):
            runtime._verify_tasks(verified)
