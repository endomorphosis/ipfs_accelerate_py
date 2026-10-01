"""Two independent signed outputs retain symbolic and native task ordering.

The fixture supplies reviewed candidate interpretations explicitly. Source
accounting and task dependency checks do not assert source semantic fidelity.
"""
from copy import deepcopy
import sys

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_intent_requirement_admission import _author, _reviewed_contract, _sha, _wire
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.planning import intent_symbolic_planning as symbolic
from ipfs_accelerate_py.agent_supervisor.planning.intent_requirement_adapter import build_intent_planning_materials
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentCompletionError


def _ordered_case(case):
    from ipfs_datasets_py.logic.intent_ir.formalize.requirements import build_intent_requirement_ledger

    original = _reviewed_contract(case)
    report = deepcopy(original["ledger"]["source_report"])
    document = report["candidates"][0]["candidate_intent_ir"]
    report_statement = deepcopy(document["statements"][0])
    report_statement.update(
        statement_id="required-report", predicate="create", arguments=["agent", "report.jsonl"],
        normalized_text="Create the reviewed answer report after repairing the answer.",
    )
    document["statements"].append(report_statement)
    report["report_sha256"] = _sha(_wire({key: value for key, value in report.items()
                                         if key != "report_sha256"}))
    source = (case["repository"] / "test_answer.py").read_text()
    ledger = build_intent_requirement_ledger(
        source, source_report=report, source_identity=original["ledger"]["source"]["source_identity"],
    )
    requirements = {row["statement_ids"][0]: row for row in ledger["requirements"]}
    answer_requirement = requirements["required-answer"]
    report_requirement = requirements["required-report"]
    report_output = {"path": "report.jsonl", "effect": "create", "media_type": "application/json"}
    groundings = [
        {"requirement_id": answer_requirement["requirement_id"],
         "outputs": deepcopy(original["requirements"][0]["outputs"]),
         "validation_keys": ["public-answer"], "dependency_requirement_ids": []},
        {"requirement_id": report_requirement["requirement_id"], "outputs": [report_output],
         "validation_keys": ["public-report"],
         "dependency_requirement_ids": [answer_requirement["requirement_id"]]},
    ]
    operations = []
    for statement, requirement, grounding, operation_id, task_key, dependencies in (
        (document["statements"][0], answer_requirement, groundings[0],
         "operation:answer", "LOCAL-TASK", []),
        (document["statements"][1], report_requirement, groundings[1],
         "operation:report", "REPORT-TASK", ["operation:answer"]),
    ):
        operations.append({
            "operation_id": operation_id, "task_key": task_key,
            "matchers": [{"requirement_id": requirement["requirement_id"],
                "native_document_sha256": requirement["native_document_sha256"],
                "statement_id": statement["statement_id"], "predicate": statement["predicate"],
                "arguments": statement["arguments"], "modality": statement["modality"]}],
            "outputs": deepcopy(grounding["outputs"]),
            "validation_keys": list(grounding["validation_keys"]),
            "dependency_operation_ids": dependencies,
        })
    contract = {
        "schema": "intent-plan-requirement-contract@2", "source_path": "test_answer.py",
        "ledger": ledger, "requirements": groundings,
        "symbolic_operations": {"schema": "intent-symbolic-operation-contract@1",
            "ledger_sha256": ledger["ledger_sha256"], "review_ref": "review:ordered-public-requirements@1",
            "interpretation_scope": "administrative_requirement_task_coverage", "operations": operations,
            "semantic_alignment_verified": False, "proof_authority": False,
            "execution_authority": False, "completion_authority": False},
    }
    baseline = deepcopy(case["manifest"]["payload"])
    baseline["tasks"].append({
        "task_key": "REPORT-TASK", "scope_paths": ["report.jsonl"], "dependencies": ["LOCAL-TASK"],
        "outputs": [deepcopy(report_output)],
        "validations": [{"validation_key": "public-report", "argv": [sys.executable, "-c",
            "import json; from pathlib import Path; assert json.loads(Path('report.jsonl').read_text()) == {'answer': 2}"],
            "cwd": ".", "expected_exit_codes": [0], "policy_cid": baseline["tasks"][0]["validations"][0]["policy_cid"]}],
        "acceptance": [{"criterion_key": "report-has-answer", "criterion": "The public report check passes",
            "validation_keys": ["public-report"], "evidence_cids": []}],
    })
    manifest = _author(case, contract, original=baseline)
    proposed = symbolic.build_intent_symbolic_plan(contract, manifest=manifest)
    return contract, manifest, proposed


def _admit(case, manifest, proposed):
    admission = local.admit_local_benchmark_plan(
        graph=proposed["graph"], manifest=manifest, requirement_bindings=proposed["requirement_bindings"],
    )
    materialized = local.materialize_local_benchmark_plan(admission=admission, intent=case["intent"])
    return admission, materialized


def test_ordered_signed_operations_project_exact_native_dependencies(scenario, tmp_path):
    from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import verify_owner_local_benchmark_observation
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import probe_quack_capabilities
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE

    contract, manifest, proposed = _ordered_case(scenario)
    materials = build_intent_planning_materials(contract, manifest=manifest)
    answer_candidate = materials.operation_candidate_ids["operation:answer"]
    report_candidate = materials.operation_candidate_ids["operation:report"]
    schedule = proposed["receipt"]["schedule"]
    assert schedule["waves"] == [[answer_candidate], [report_candidate]]
    assert schedule["dependency_edges"] == [[answer_candidate, report_candidate]]
    assert materials.current_facts == ()
    assert materials.frozen_goal.policy.satisfied_dependencies == ()
    assert proposed["receipt"]["provider_calls"] == proposed["receipt"]["observed_facts_supplied"] == 0
    tasks = {task.task_key: task for task in proposed["graph"].tasks}
    assert tasks["LOCAL-TASK"].dependency_task_cids == ()
    assert tasks["REPORT-TASK"].dependency_task_cids == (tasks["LOCAL-TASK"].task_cid,)
    for spec in manifest["payload"]["tasks"]:
        task = tasks[spec["task_key"]]
        assert [{key: getattr(output, key) for key in ("path", "effect", "media_type")}
                for output in task.outputs] == spec["outputs"]
        assert [validation.validation_key for validation in task.validations] == [
            row["validation_key"] for row in spec["validations"]]
    expected_effects = sorted(
        [spec["task_key"], output["path"], output["effect"], output["media_type"]]
        for spec in manifest["payload"]["tasks"] for output in spec["outputs"]
    )
    assert proposed["receipt"]["formal_effects_cid"] == cid_for_dag_json(expected_effects)
    admission, materialized = _admit(scenario, manifest, proposed)
    assert set(materialized["task_cids"]) == {task.task_cid for task in tasks.values()}
    persisted = scenario["intent"].get_task(tasks["REPORT-TASK"].task_cid)
    assert tuple(persisted["dependencies"]) == (tasks["LOCAL-TASK"].task_cid,)
    pending, _, _, _ = local._contract(persisted["body"], persisted["task_cid"])
    assert pending["dependencies"] == list(persisted["dependencies"])
    assert pending["intent_plan"]["symbolic_planning"] == proposed["receipt"]
    capabilities = probe_quack_capabilities()
    if not capabilities.passes_health_check:
        pytest.skip(f"installed native Quack unavailable: {capabilities.reason_code}")
    routes = {key: GROK_CODEX_EXECUTION_MODE for key in tasks}
    with open_existing_native_owner(
        database=scenario["intent"].database_path, checkout=scenario["repository"],
        state_dir=tmp_path / "ordered-native-owner", repository_id=manifest["payload"]["repository_cid"],
        execution_routes=routes,
    ) as owner:
        observed = verify_owner_local_benchmark_observation(server=owner.server, admission=admission)
        assert observed["receipt"]["intent_symbolic_planning"] == proposed["receipt"]
        runtime = AdmittedBenchmarkRuntime()
        runtime.source, runtime.server, runtime.admission = owner.source, owner.server, admission
        runtime._verify_tasks(observed)
        with owner.server._lock:
            native_edges = owner.server._connection.execute(
                "SELECT task_cid, dependency_task_cid, kind FROM task_dependencies ORDER BY task_cid"
            ).fetchall()
        assert [tuple(row[index] for index in range(3)) for row in native_edges] == [
            (tasks["REPORT-TASK"].task_cid, tasks["LOCAL-TASK"].task_cid, "depends_on")]


def test_ordered_outputs_require_both_real_public_checks(scenario):
    _, manifest, proposed = _ordered_case(scenario)
    _admit(scenario, manifest, proposed)
    intent = scenario["intent"]
    tasks = {task.task_key: task for task in proposed["graph"].tasks}
    answer_cid, report_cid = tasks["LOCAL-TASK"].task_cid, tasks["REPORT-TASK"].task_cid
    intent.cas_task_status(task_cid=answer_cid, expected_revision=intent.get_task(answer_cid)["revision"],
                           new_status="in_progress")
    assert not local.run_local_task_validations(intent=intent, task_cid=answer_cid, attempt_id="ordered:answer-fails")["passed"]
    with pytest.raises(IntentCompletionError):
        intent.cas_task_status(task_cid=answer_cid, expected_revision=intent.get_task(answer_cid)["revision"], new_status="completed")
    (scenario["repository"] / "answer.py").write_text("def answer():\n    return 2\n")
    assert local.run_local_task_validations(intent=intent, task_cid=answer_cid, attempt_id="ordered:answer-passes")["passed"]
    intent.cas_task_status(task_cid=answer_cid, expected_revision=intent.get_task(answer_cid)["revision"], new_status="completed")
    intent.cas_task_status(task_cid=report_cid, expected_revision=intent.get_task(report_cid)["revision"], new_status="in_progress")
    assert not local.run_local_task_validations(intent=intent, task_cid=report_cid, attempt_id="ordered:report-absent")["passed"]
    (scenario["repository"] / "report.jsonl").write_text('{"answer":2}\n')
    assert local.run_local_task_validations(intent=intent, task_cid=report_cid, attempt_id="ordered:report-passes")["passed"]
    intent.cas_task_status(task_cid=report_cid, expected_revision=intent.get_task(report_cid)["revision"], new_status="completed")
    assert all(intent.get_task(task.task_cid)["status"] == "completed" for task in tasks.values())


@pytest.mark.parametrize("mutation", ["drop", "reverse"])
def test_owner_signed_dependency_erasure_or_reversal_cannot_enter_native_storage(scenario, mutation):
    _, manifest, proposed = _ordered_case(scenario)
    payload = deepcopy(manifest["payload"])
    specs = {row["task_key"]: row for row in payload["tasks"]}
    specs["REPORT-TASK"]["dependencies"] = []
    if mutation == "reverse":
        specs["LOCAL-TASK"]["dependencies"] = ["REPORT-TASK"]
    forged_manifest = local._signed(payload, payload)
    profile = local._manifest(manifest)[1]
    assert local._verify_signature(forged_manifest, profile) == payload
    with pytest.raises(ValueError):
        symbolic.build_intent_symbolic_plan(local.decode_intent_requirement_contract(payload), manifest=forged_manifest)
    with pytest.raises(ValueError):
        local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=forged_manifest,
                                         requirement_bindings=proposed["requirement_bindings"])
    assert all(scenario["intent"].get_task(task.task_cid) is None for task in proposed["graph"].tasks)
