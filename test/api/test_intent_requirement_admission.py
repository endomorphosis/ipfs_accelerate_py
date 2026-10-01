"""Source-bound candidate requirements survive signed admission and native tasks.

The reviewed fixture supplies its interpretation explicitly. These tests check
source accounting, plan coverage and immutable authority bindings; they make no
claim that a translator recovered the meaning of an arbitrary instruction.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_local_planning_declared_create import _with_creation
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction as instruction


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _reviewed_contract(scenario, *, confidence=0):
    from ipfs_datasets_py.logic.intent_ir.formalize.requirements import build_intent_requirement_ledger
    from ipfs_datasets_py.logic.intent_ir.schema import (
        IntentIRDocument, IntentKind, IntentModality, IntentStatement,
        NodeGrounding, ReviewStatus, SourceRef, SourceSpan, StatementKind,
    )

    source = (scenario["repository"] / "test_answer.py").read_text()
    raw = source.encode("utf-8")
    digest = _sha(raw)
    source_ref = SourceRef(
        ref_id="public-check", source_uri="fixture:public-check", source_id=digest,
        source_revision=digest, content_sha256=digest, span=SourceSpan(0, len(source)),
        review_status=ReviewStatus.MACHINE_EXTRACTED,
    )
    statement = IntentStatement(
        statement_id="required-answer", kind=StatementKind.GOAL,
        modality=IntentModality.REQUIRED, normalized_text="The public answer check must pass.",
        source_ref_ids=(source_ref.ref_id,), predicate="modify",
        arguments=("agent", "answer.py"), confidence=confidence, grounding=NodeGrounding.INFERRED,
        review_status=ReviewStatus.MACHINE_EXTRACTED,
    )
    document = IntentIRDocument(
        document_id="reviewed-answer:" + digest, title="Explicitly reviewed public check",
        intent_kind=IntentKind.DECLARATIVE, sources=(source_ref,), statements=(statement,),
    )
    document.validate()
    report = {
        "schema": "intent-reviewed-source-report@1", "source_sha256": digest,
        "source_bytes": len(raw), "source_characters": len(source),
        "producer": {"name": "reviewed-admission-fixture", "revision": "1"},
        "interpretation_status": "reviewed_candidate",
        "units": [{"unit_id": "unit:public-answer", "start_char": 0,
                   "end_char": len(source), "start_byte": 0, "end_byte": len(raw),
                   "text": source, "sha256": digest,
                   "disposition": "interpreted_candidate", "reason": "explicit_reviewed_fixture"}],
        "candidates": [{"unit_id": "unit:public-answer", "candidate_intent_ir": document.to_dict()}],
        "proof_authority": False, "execution_authority": False,
        "completion_authority": False, "source_semantics_verified": False,
    }
    report["report_sha256"] = _sha(_wire(report))
    ledger = build_intent_requirement_ledger(
        source, source_report=report,
        source_identity={"path": "test_answer.py", "revision": scenario["manifest"]["payload"]["baseline_commit"]},
    )
    requirement_id = ledger["requirements"][0]["requirement_id"]
    return {
        "schema": "intent-plan-requirement-contract@1", "source_path": "test_answer.py",
        "ledger": ledger,
        "requirements": [{"requirement_id": requirement_id,
                          "outputs": [{"path": "answer.py", "effect": "modify", "media_type": "text/x-python"}],
                          "validation_keys": ["public-answer"], "dependency_requirement_ids": []}],
    }


def _author(scenario, contract, *, original=None):
    baseline = original or scenario["manifest"]["payload"]
    return local.author_local_benchmark_manifest(
        repository=scenario["repository"], profile_dir=baseline["profile_dir"],
        lifecycle_dir=baseline["lifecycle_dir"], task_specs=baseline["tasks"],
        planning_roots=baseline["planning_roots"],
        planning_inputs=baseline.get("planning_inputs"), intent_requirements=contract,
    )


@pytest.fixture
def intent_admitted(scenario):
    contract = _reviewed_contract(scenario)
    manifest = _author(scenario, contract)
    bindings = [{"requirement_id": contract["requirements"][0]["requirement_id"],
                 "task_keys": ["LOCAL-TASK"], "validation_keys": ["public-answer"]}]
    admission = local.admit_local_benchmark_plan(
        graph=scenario["graph"], manifest=manifest, requirement_bindings=bindings,
    )
    return {**scenario, "requirements": contract, "bindings": bindings,
            "intent_manifest": manifest, "admission": admission}


def test_native_materialization_retains_recomputed_intent_plan(intent_admitted):
    case = intent_admitted
    manifest, admission = case["intent_manifest"], case["admission"]
    assert manifest["payload"]["schema"] == local.INTENT_MANIFEST_SCHEMA
    assert manifest["payload"]["created_outputs"] == []
    verified = local.verify_local_benchmark_admission(admission)
    coverage = verified["receipt"]["requirement_coverage"]
    assert coverage["accepted"] is True
    assert coverage["semantic_alignment_verified"] is False
    assert coverage["completion_authority"] is False
    result = local.materialize_local_benchmark_plan(admission=admission, intent=case["intent"])
    task = case["intent"].get_task(result["task_cids"][0])
    contract, _, _, _ = local._contract(task["body"], task["task_cid"])
    assert contract["schema"] == local.INTENT_CONTRACT_SCHEMA
    assert contract["intent_plan"] == {
        "schema": "supervisor-local-intent-plan@1", "graph": admission["graph"],
        "requirement_bindings": admission["requirement_bindings"], "coverage": coverage,
    }
    assert contract["planning_receipt_cid"] == content_identity(admission["receipt"])
    assert local.decode_intent_requirement_contract(contract["manifest"]["payload"]) == case["requirements"]
    reference = case["intent"].get_plan(result["plan_id"])["body"]["local_planning_receipt_ref"]
    assert reference["requirement_contract_cid"] == coverage["contract_cid"]
    assert reference["requirement_coverage_cid"] == content_identity(coverage)
    assert local.load_local_planning_receipt(reference, manifest=manifest) == admission["receipt"]


def test_fractional_candidate_confidence_survives_signed_native_materialization(scenario):
    requirements = _reviewed_contract(scenario, confidence=0.75)
    manifest = _author(scenario, requirements)
    artifact = manifest["payload"]["intent_requirements"]
    assert artifact["schema"] == "supervisor-local-intent-requirement-artifact@1"
    decoded = local.decode_intent_requirement_contract(manifest["payload"])
    assert decoded == requirements
    assert decoded["ledger"]["source_report"]["candidates"][0]["candidate_intent_ir"]["statements"][0]["confidence"] == 0.75
    # The formal manifest identity is still subject to the existing serializer's
    # numeric policy; candidate numeric metadata lives in its inert JSON binding.
    assert content_identity(manifest)
    bindings = [{"requirement_id": requirements["requirements"][0]["requirement_id"],
                 "task_keys": ["LOCAL-TASK"], "validation_keys": ["public-answer"]}]
    admission = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=manifest,
                                               requirement_bindings=bindings)
    result = local.materialize_local_benchmark_plan(admission=admission, intent=scenario["intent"])
    task = scenario["intent"].get_task(result["task_cids"][0])
    contract, _, _, _ = local._contract(task["body"], task["task_cid"])
    assert local.decode_intent_requirement_contract(contract["manifest"]["payload"]) == requirements


@pytest.mark.parametrize("mutation", ["requirement", "source_span"])
def test_owner_resigned_ledger_must_match_its_source_report(intent_admitted, mutation):
    case = intent_admitted
    requirements = deepcopy(case["requirements"])
    ledger = requirements["ledger"]
    if mutation == "requirement":
        ledger["requirements"][0]["modality"] = "intended"
    else:
        report = ledger["source_report"]
        report["units"][0]["text"] = "A replacement requirement"
        report["report_sha256"] = _sha(_wire({k: v for k, v in report.items() if k != "report_sha256"}))
        ledger["source_report_sha256"] = _sha(_wire(report))
    ledger["ledger_sha256"] = _sha(_wire({k: v for k, v in ledger.items() if k != "ledger_sha256"}))
    with pytest.raises(ValueError):
        _author(case, requirements)


@pytest.mark.parametrize("mutation", ["omitted", "no_task", "no_validation", "foreign_requirement"])
def test_requirement_omission_cannot_receive_admission(intent_admitted, mutation):
    case = intent_admitted
    bindings = deepcopy(case["bindings"])
    if mutation == "omitted":
        bindings = []
    elif mutation == "no_task":
        bindings[0]["task_keys"] = []
    elif mutation == "no_validation":
        bindings[0]["validation_keys"] = []
    else:
        bindings[0]["requirement_id"] = "intent-requirement:" + "0" * 64
    with pytest.raises(ValueError):
        local.admit_local_benchmark_plan(
            graph=case["graph"], manifest=case["intent_manifest"], requirement_bindings=bindings,
        )


@pytest.mark.parametrize("mutation", ["drop", "empty", "foreign_task"])
def test_admission_bindings_cannot_be_deleted_or_replaced(intent_admitted, mutation):
    case = intent_admitted
    admission = deepcopy(case["admission"])
    if mutation == "drop":
        del admission["requirement_bindings"]
    elif mutation == "empty":
        admission["requirement_bindings"] = []
    else:
        admission["requirement_bindings"][0]["task_keys"] = ["UNAUTHORIZED-TASK"]
    with pytest.raises(ValueError):
        local.verify_local_benchmark_admission(admission)
    with pytest.raises(ValueError):
        local.materialize_local_benchmark_plan(admission=admission, intent=case["intent"])
    assert case["intent"].get_task(case["task_cid"]) is None


def test_owner_resigning_forged_coverage_cannot_bypass_recomputation(intent_admitted):
    case = intent_admitted
    admission = deepcopy(case["admission"])
    payload = admission["receipt"]["payload"]
    payload["requirement_coverage"]["graph_cid"] = "foreign-graph"
    admission["receipt"] = local._signed(payload, admission["manifest"]["payload"])
    with pytest.raises(ValueError, match="recomputed|receipt"):
        local.verify_local_benchmark_admission(admission)


@pytest.mark.parametrize("mutation", ["coverage", "bindings", "graph"])
def test_owner_resigned_pending_contract_must_recompute_intent_plan(intent_admitted, mutation):
    case = intent_admitted
    result = local.materialize_local_benchmark_plan(admission=case["admission"], intent=case["intent"])
    cid = result["task_cids"][0]
    body = deepcopy(case["intent"].get_task(cid)["body"])
    payload = body[local.CONTRACT_KEY]["payload"]
    if mutation == "coverage":
        payload["intent_plan"]["coverage"]["graph_cid"] = "foreign-graph"
    elif mutation == "bindings":
        payload["intent_plan"]["requirement_bindings"] = []
    else:
        payload["intent_plan"]["graph"]["tasks"][0]["objective"] = "An unauthorized objective"
    body[local.CONTRACT_KEY] = local._signed(payload, case["intent_manifest"]["payload"])
    with pytest.raises(ValueError):
        local._contract(body, cid)


def test_changed_immutable_requirement_source_refuses_plan_and_execution(intent_admitted):
    case = intent_admitted
    result = local.materialize_local_benchmark_plan(admission=case["admission"], intent=case["intent"])
    cid = result["task_cids"][0]
    (case["repository"] / "test_answer.py").write_text("pass\n")
    with pytest.raises(ValueError):
        local.verify_local_benchmark_admission(case["admission"])
    with pytest.raises(ValueError):
        local._contract(case["intent"].get_task(cid)["body"], cid)


def test_requirement_source_cannot_be_declared_as_mutable_output(intent_admitted):
    case = intent_admitted
    original = deepcopy(case["intent_manifest"]["payload"])
    original["tasks"][0]["outputs"].append(
        {"path": "test_answer.py", "effect": "modify", "media_type": "text/x-python"}
    )
    with pytest.raises(ValueError, match="immutable"):
        _author(case, case["requirements"], original=original)


def test_intent_manifest_retains_declared_create_inventory(intent_admitted):
    case = intent_admitted
    graph, original = _with_creation(case)
    requirements = deepcopy(case["requirements"])
    requirements["requirements"][0]["outputs"].append(
        {"path": "report.jsonl", "effect": "create", "media_type": "application/json"}
    )
    manifest = _author(case, requirements, original=original["payload"])
    assert manifest["payload"]["schema"] == local.INTENT_MANIFEST_SCHEMA
    assert manifest["payload"]["created_outputs"] == ["report.jsonl"]
    admission = local.admit_local_benchmark_plan(
        graph=graph, manifest=manifest, requirement_bindings=case["bindings"],
    )
    verified = local.verify_local_benchmark_admission(admission)
    assert verified["receipt"]["requirement_coverage"]["accepted"] is True
    (case["repository"] / "report.jsonl").write_text('{"result":"candidate"}\n')
    observed = local.observe_local_manifest_sources(case["repository"], manifest["payload"])
    assert "report.jsonl" in observed
    assert "report.jsonl" not in manifest["payload"]["sources"]


def test_public_instruction_reader_accepts_signed_intent_manifest(intent_admitted, tmp_path):
    case = intent_admitted
    raw = (case["repository"] / "test_answer.py").read_bytes()
    context = instruction.prepare_public_instruction_context(
        repository=case["repository"], admission=case["admission"], task_cid=case["task_cid"],
        source_path="test_answer.py", expected_source_sha256=_sha(raw),
    )
    workspace = tmp_path / "allocated-worker"
    subprocess.run(["git", "-C", str(case["repository"]), "worktree", "add", "--detach",
                    "-q", str(workspace)], check=True)
    prompt, receipt = instruction.load_public_instruction(
        artifact=Path(context["artifact"]), expected_sha256=context["sha256"],
        task_cid=case["task_cid"], prompt=json.dumps({"objective_id": "LOCAL-TASK"}),
        workspace=workspace,
    )
    assert raw.decode("utf-8") in prompt
    assert receipt["manifest_signature_verified"] is True
    assert receipt["source_freshness_verified"] is True
    assert receipt["completion_authority"] is False


@contextmanager
def _native_intent_owner(case, tmp_path):
    from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
    from ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities import probe_quack_capabilities
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE

    capabilities = probe_quack_capabilities()
    if not capabilities.passes_health_check:
        pytest.skip(f"installed native Quack unavailable: {capabilities.reason_code}")
    local.materialize_local_benchmark_plan(admission=case["admission"], intent=case["intent"])
    with open_existing_native_owner(
        database=case["intent"].database_path, checkout=case["repository"],
        state_dir=tmp_path / "native-owner", repository_id=case["intent_manifest"]["payload"]["repository_cid"],
        execution_routes={"LOCAL-TASK": GROK_CODEX_EXECUTION_MODE},
    ) as owner:
        yield owner


def test_actual_native_owner_observes_signed_intent_admission(intent_admitted, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import verify_owner_local_benchmark_observation

    case = intent_admitted
    with _native_intent_owner(case, tmp_path) as owner:
        observed = verify_owner_local_benchmark_observation(server=owner.server, admission=case["admission"])
        assert observed["receipt"]["requirement_coverage"]["accepted"] is True
        assert observed["receipt"]["completion_authority"] is False
        assert observed["manifest"]["schema"] == local.INTENT_MANIFEST_SCHEMA
        for mutation in ("drop", "empty"):
            altered = deepcopy(case["admission"])
            if mutation == "drop":
                del altered["requirement_bindings"]
            else:
                altered["requirement_bindings"] = []
            with pytest.raises(ValueError):
                verify_owner_local_benchmark_observation(server=owner.server, admission=altered)


def test_v4_publication_requires_real_observation_then_allows_typed_completion(intent_admitted, tmp_path):
    from test.api.test_local_completion_bridge import native_published_transition, complete
    from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import (
        run_owner_local_task_validations, verify_owner_local_benchmark_observation,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import TransactionError
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_source import TaskSourceConflictError
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon

    case = intent_admitted
    with _native_intent_owner(case, tmp_path) as owner:
        daemon = DatabaseImplementationDaemon(
            database_path=owner.database, coordination_path=tmp_path / "coordination.duckdb",
            execution_path=tmp_path / "execution.duckdb", authority_mode="quack",
            task_source_kind="duckdb", owner_session_id="session:intent-publication-test",
            process_instance_id=owner.identity.process_birth_id,
            quack_uri=owner.identity.listen_uri, task_source=owner.source,
            close_task_source=False, state_owner_bootstrap_credentials=owner.credentials,
            strict_task_sharding=True, max_task_attempts=2, lease_ms=60_000,
            require_real_execution=True,
        ).open()
        try:
            attempt = daemon.claim_next()
            assert attempt is not None
            task = owner.source.get_task(attempt.task_cid)
            transition = native_published_transition(case, tmp_path, owner, attempt)
            with pytest.raises(local.LocalPlanningError, match="no exact native owner observation"):
                verify_owner_local_benchmark_observation(server=owner.server, admission=case["admission"])
            with pytest.raises((TaskSourceConflictError, TransactionError)):
                complete(owner, attempt, local.content_identity(transition))
            passed = run_owner_local_task_validations(
                server=owner.server, task_cid=task.task_cid, attempt_id=attempt.attempt_id,
                expected_revision=task.revision, source_transition=transition,
            )
            assert passed["passed"] is True
            observed = verify_owner_local_benchmark_observation(server=owner.server, admission=case["admission"])
            assert observed["current_source_tree_id"] == passed["source_tree_id"]
            assert observed["receipt"]["requirement_coverage"]["accepted"] is True
            altered = deepcopy(case["admission"])
            altered["requirement_bindings"] = []
            with pytest.raises(local.LocalPlanningError):
                verify_owner_local_benchmark_observation(server=owner.server, admission=altered)
            complete(owner, attempt, passed["results"][0]["evidence_digest"])
            assert owner.source.get_task(task.task_cid).status == "completed"
            after = verify_owner_local_benchmark_observation(server=owner.server, admission=case["admission"])
            assert after["current_source_tree_id"] == passed["source_tree_id"]
        finally:
            daemon.close()


@pytest.mark.parametrize("field", ["task_spec", "dependencies", "pending_requirements", "manifest",
                                   "intent_owner_id", "planning_receipt_cid"])
@pytest.mark.parametrize("mutation", ["delete", "corrupt"])
def test_runtime_rejects_owner_signed_corruption_of_entire_pending_contract(intent_admitted, tmp_path, field, mutation):
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime

    case = intent_admitted
    with _native_intent_owner(case, tmp_path) as owner:
        runtime = AdmittedBenchmarkRuntime()
        runtime.source, runtime.server, runtime.admission = owner.source, owner.server, case["admission"]
        verified = local.verify_local_benchmark_admission(case["admission"])
        runtime._verify_tasks(verified)
        original = owner.source.get_task(case["task_cid"])
        body = deepcopy(dict(original.body))
        payload = body[local.CONTRACT_KEY]["payload"]
        if mutation == "delete":
            del payload[field]
        elif field == "task_spec":
            payload[field]["acceptance"][0]["criterion"] = "A replaced acceptance requirement"
        elif field == "dependencies":
            payload[field] = [content_identity({"foreign_dependency": True})]
        elif field == "pending_requirements":
            payload[field] = []
        elif field == "manifest":
            nested = deepcopy(payload[field]["payload"])
            nested["tasks"][0]["acceptance"][0]["criterion"] = "A replaced manifest acceptance"
            payload[field] = local._signed(nested, nested)
        elif field == "intent_owner_id":
            payload[field] = "intent-repository:foreign"
        else:
            payload[field] = content_identity({"foreign_planning_receipt": True})
        body[local.CONTRACT_KEY] = local._signed(payload, case["intent_manifest"]["payload"])
        # This directly injects a signed malformed persisted row into a
        # disposable owner. Ordinary task-update guards refuse the change.
        with owner.server._lock:
            owner.server._connection.execute(
                "UPDATE tasks SET body_json = ? WHERE task_cid = ?",
                [_wire(body).decode("utf-8"), original.task_cid],
            )
        assert owner.source.get_task(original.task_cid).body[local.CONTRACT_KEY] == body[local.CONTRACT_KEY]
        with pytest.raises(ValueError):
            runtime._verify_tasks(verified)


@pytest.mark.parametrize("relation", ["task_alias", "dependencies", "goal_cid", "plan_cid"])
def test_runtime_rejects_forged_native_relations_with_valid_contract(intent_admitted, tmp_path, relation):
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime

    case = intent_admitted
    with _native_intent_owner(case, tmp_path) as owner:
        runtime = AdmittedBenchmarkRuntime()
        runtime.source, runtime.server, runtime.admission = owner.source, owner.server, case["admission"]
        verified = local.verify_local_benchmark_admission(case["admission"])
        runtime._verify_tasks(verified)
        original = owner.source.get_task(case["task_cid"])
        envelope = deepcopy(original.body[local.CONTRACT_KEY])
        foreign = content_identity({"forged_native_relation": relation})
        with owner.server._lock:
            if relation == "dependencies":
                owner.server._connection.execute(
                    "INSERT INTO task_dependencies (task_cid, dependency_task_cid, kind) VALUES (?, ?, ?)",
                    [original.task_cid, foreign, "depends_on"],
                )
            else:
                # relation is an explicit parameter drawn from the three
                # schema columns above, never a source-controlled identifier.
                owner.server._connection.execute(
                    f"UPDATE tasks SET {relation} = ? WHERE task_cid = ?",
                    ["FORGED-ALIAS" if relation == "task_alias" else foreign, original.task_cid],
                )
            retained = json.loads(owner.server._connection.execute(
                "SELECT body_json FROM tasks WHERE task_cid = ?", [original.task_cid],
            ).fetchone()[0])
        assert retained[local.CONTRACT_KEY] == envelope
        assert local._verify_signature(envelope, verified["profile"])["task_cid"] == original.task_cid
        with pytest.raises(ValueError):
            runtime._verify_tasks(verified)
