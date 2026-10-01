"""Public planning replay uses the owner's declaration rules without its keys."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_intent_requirement_admission import _author, _reviewed_contract
from test.api.test_local_planning_declared_create import _with_creation
from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import check_intent_plan_coverage
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction as instruction


def _case(scenario, *, create=False):
    contract = _reviewed_contract(scenario)
    original = None
    graph = scenario["graph"]
    if create:
        graph, created = _with_creation(scenario)
        original = created["payload"]
        contract["requirements"][0]["outputs"].append(
            {"path": "report.jsonl", "effect": "create", "media_type": "application/json"})
    manifest = _author(scenario, contract, original=original)
    bindings = [{"requirement_id": contract["requirements"][0]["requirement_id"],
                 "task_keys": ["LOCAL-TASK"], "validation_keys": ["public-answer"]}]
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=manifest,
                                                requirement_bindings=bindings)
    verified = local.verify_local_benchmark_admission(admission)
    payload = {"schema": instruction.INTENT_SCHEMA, "manifest": manifest,
        "owner_identity": verified["profile"].identity_did,
        "owner_profile_id": verified["profile"].profile_id,
        "task_id": graph.tasks[0].task_key, "task_cid": graph.tasks[0].task_cid,
        "intent_plan_admission": {"graph": graph.to_dict(), "receipt": admission["receipt"],
                                  "requirement_bindings": bindings}}
    return {"graph": graph, "manifest": manifest, "bindings": bindings, "payload": payload,
            "source_text": (scenario["repository"] / "test_answer.py").read_text()}


def _mutate(payload, mutation):
    task = payload["tasks"][0]
    if mutation == "exit_code":
        task["validations"][0]["expected_exit_codes"] = [1]
    elif mutation == "validation_policy":
        task["validations"][0]["policy_cid"] = content_identity({"foreign_policy": True})
    elif mutation == "validation_cwd":
        task["validations"][0]["cwd"] = "../escape"
    elif mutation == "validation_argv":
        task["validations"][0]["argv"] = ["python\nunauthorized"]
    elif mutation == "empty_argv":
        task["validations"][0]["argv"] = []
    elif mutation == "output_effect":
        task["outputs"][0]["effect"] = "delete"
    elif mutation == "created_population":
        payload["created_outputs"] = ["foreign.py"]
    elif mutation == "created_type":
        payload["created_outputs"] = {"foreign.py": True}
    elif mutation == "foreign_scope":
        task["scope_paths"].append("foreign.py")
    elif mutation == "output_scope":
        task["scope_paths"].remove("answer.py")
    elif mutation == "dependency":
        task["dependencies"] = ["FOREIGN-TASK"]
    elif mutation == "manifest_extra":
        payload["proofs"] = []
    elif mutation == "task_extra":
        task["phase"] = "completed"
    elif mutation == "validation_extra":
        task["validations"][0]["runner"] = "foreign"
    elif mutation == "acceptance_validation":
        task["acceptance"][0]["validation_keys"] = []
    elif mutation == "acceptance_evidence":
        task["acceptance"][0]["evidence_cids"] = [content_identity({"unobserved": True})]
    elif mutation == "source_extra":
        payload["sources"]["answer.py"]["owner"] = "foreign"
    elif mutation == "source_hash":
        payload["sources"]["answer.py"]["sha256"] = "not-a-sha256"
    elif mutation == "source_mode":
        payload["sources"]["answer.py"]["executable"] = 1
    elif mutation == "source_path":
        payload["sources"]["../outside.py"] = payload["sources"].pop("answer.py")
    else:
        raise AssertionError(mutation)


@pytest.mark.parametrize("mutation", ["exit_code", "validation_policy", "validation_cwd",
    "validation_argv", "empty_argv", "output_effect", "created_population", "created_type",
    "foreign_scope", "output_scope", "dependency", "manifest_extra", "task_extra",
    "validation_extra", "acceptance_validation", "acceptance_evidence", "source_extra",
    "source_hash", "source_mode", "source_path"])
def test_owner_resigned_invalid_declaration_fails_owner_and_public_replay(scenario, mutation):
    case = _case(scenario)
    altered = deepcopy(case["manifest"]["payload"])
    _mutate(altered, mutation)
    resigned = local._signed(altered, altered)
    public = deepcopy(case["payload"])
    public["manifest"] = resigned
    with pytest.raises(local.LocalPlanningError):
        local.admit_local_benchmark_plan(graph=case["graph"], manifest=resigned,
                                        requirement_bindings=case["bindings"])
    with pytest.raises(local.LocalPlanningError):
        local._validate_local_manifest_declarations(altered)
    with pytest.raises(local.LocalPlanningError):
        local._planning_payload(case["graph"], resigned, altered,
            SimpleNamespace(profile_id=public["owner_profile_id"]), altered["sources"],
            case["bindings"], requirement_source_text=case["source_text"])
    with pytest.raises(ValueError):
        instruction._intent_context(public, altered, case["source_text"])


def test_matching_owner_signed_nonzero_exit_receipt_cannot_bypass_declaration_rules(scenario):
    case = _case(scenario)
    graph = case["graph"]
    task = graph.tasks[0]
    task = replace(task, validations=(replace(task.validations[0], expected_exit_codes=(1,)),))
    graph = replace(graph, tasks=(task,))
    altered = deepcopy(case["manifest"]["payload"])
    altered["tasks"][0]["validations"][0]["expected_exit_codes"] = [1]
    resigned = local._signed(altered, altered)
    # Compilation and requirement coverage alone can accept this proposal.
    # The shared declaration check must still reject it at either boundary.
    compiled, validation, pending = local._graph_contract(graph, altered, local._tree(altered["sources"]))
    coverage = check_intent_plan_coverage(local.decode_intent_requirement_contract(altered),
                                          graph=graph, bindings=case["bindings"])
    assert coverage["accepted"] is True
    receipt = deepcopy(case["payload"]["intent_plan_admission"]["receipt"]["payload"])
    receipt.update(manifest_cid=content_identity(resigned), graph_cid=graph.content_id,
        plan_id=compiled.plan.plan_id, pending_requirements=pending,
        pending_cid=content_identity(pending), plan_evidence=validation,
        declared_input_closure=local._declared_closure(graph, altered, local._tree(altered["sources"])),
        requirement_coverage=coverage)
    public = deepcopy(case["payload"])
    public.update(manifest=resigned, task_cid=task.task_cid)
    public["intent_plan_admission"].update(graph=graph.to_dict(),
                                           receipt=local._signed(receipt, altered))
    with pytest.raises(local.LocalPlanningError, match="unsupported or duplicate validation"):
        local.admit_local_benchmark_plan(graph=graph, manifest=resigned,
                                        requirement_bindings=case["bindings"])
    with pytest.raises(local.LocalPlanningError, match="unsupported or duplicate validation"):
        instruction._intent_context(public, altered, case["source_text"])


@pytest.mark.parametrize("create", [False, True])
def test_public_replay_uses_observed_baseline_text_without_private_or_current_source_access(
        scenario, monkeypatch, create):
    case = _case(scenario, create=create)
    before = instruction._intent_context(case["payload"], case["manifest"]["payload"],
                                         case["source_text"])
    (scenario["repository"] / "test_answer.py").write_text("subsequent instruction revision\n")

    def inaccessible(*args, **kwargs):
        raise AssertionError("public replay attempted private or current source access")

    for name in ("load_local_profile", "_verify_intent_requirements", "_repository", "_sources"):
        monkeypatch.setattr(local, name, inaccessible)
    assert local._validate_local_manifest_declarations(case["manifest"]["payload"]) == (
        {"answer.py", "report.jsonl"} if create else {"answer.py"})
    after = instruction._intent_context(case["payload"], case["manifest"]["payload"],
                                        case["source_text"])
    assert before == after
    assert after["source_semantics_verified"] is False
    assert after["native_persistence_verified_here"] is False
    assert after["requirements"]


def test_shared_declarations_preserve_legacy_manifest_versions_without_source_reads(scenario, monkeypatch):
    legacy = scenario["manifest"]["payload"]
    created = _with_creation(scenario)[1]["payload"]

    def inaccessible(*args, **kwargs):
        raise AssertionError("declaration validation observed private or source state")

    for name in ("load_local_profile", "_repository", "_sources", "_git"):
        monkeypatch.setattr(local, name, inaccessible)
    assert local._validate_local_manifest_declarations(legacy) == {"answer.py"}
    assert local._validate_local_manifest_declarations(created) == {"answer.py", "report.jsonl"}
