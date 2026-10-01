"""Exact reviewed candidate requirements reach the worker without owner secrets.

The provider alone is replaced. Signed public replay, allocated Git worktrees,
native context compilation and optional semantic transport execute normally.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_intent_requirement_admission import _author, _reviewed_contract, _sha, _wire
from test.api.test_intent_symbolic_planning import _prepare
from test.api.test_intent_symbolic_planning_ordered import _ordered_case
from test.api.test_semantic_router_integration import provider  # noqa: F401
from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    ContextCompiler, build_text_context_references, render_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import ContextBudget
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction as instruction
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import prepare_semantic_context
from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit

BEGIN = "--- BEGIN TASK INTENT REQUIREMENTS ---\n"
END = "\n--- END TASK INTENT REQUIREMENTS ---"
AUTHORITY = ("source_semantics_verified", "semantic_alignment_verified", "proof_authority",
             "execution_authority", "completion_authority", "publication_authority", "scope_expansion_authority")


def _view(text):
    assert text.count(BEGIN) == text.count(END) == 1
    return json.loads(text.split(BEGIN, 1)[1].split(END, 1)[0])


def _context(case, admission, task, tmp_path):
    root = case["repository"]
    source = (root / "test_answer.py").read_bytes()
    context = instruction.prepare_public_instruction_context(
        repository=root, admission=admission, task_cid=task.task_cid,
        source_path="test_answer.py", expected_source_sha256=_sha(source),
    )
    workspace = tmp_path / "allocated-intent-worker"
    subprocess.run(["git", "-C", str(root), "worktree", "add", "--detach", "-q", str(workspace)], check=True)
    request = dict(artifact=Path(context["artifact"]), expected_sha256=context["sha256"],
        task_cid=task.task_cid, prompt=json.dumps({"objective_id": task.task_key}), workspace=workspace)
    return {**case, "admission": admission, "task": task, "source": source,
            "context": context, "request": request, "workspace": workspace}


@pytest.fixture
def worker_case(scenario, tmp_path):
    contract, manifest, proposed = _ordered_case(scenario)
    admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=manifest,
        requirement_bindings=proposed["requirement_bindings"])
    tasks = {task.task_key: task for task in proposed["graph"].tasks}
    return {**_context(scenario, admission, tasks["REPORT-TASK"], tmp_path),
            "contract": contract, "manifest": manifest, "proposed": proposed, "tasks": tasks}


def _capsule(case, semantic):
    root = case["repository"]
    refs = ()
    if semantic:
        output = root / ".runtime/intent-semantic"
        prepare_semantic_context(repository=root, paths=["answer.py"], required_raw_paths=["answer.py"],
            objective="Create the public report", task_id="REPORT-TASK", output=output)
        artifact = output / "worker-context.json"
        refs = build_text_context_references(artifact.read_text(), reference_prefix="semantic-context",
            kind="semantic-context", path=artifact.relative_to(root).as_posix(), repository_id="repo:test",
            tree_id="tree:test", required=True, chunk_bytes=1201)
    compiled = ContextCompiler(ContextBudget(max_input_tokens=32768, max_items=128,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:test", tree_id="tree:test", objective_id="REPORT-TASK",
        objective_revision="sha256:task", policy_id="policy:test", policy_revision="sha256:policy",
        caller="supervisor:test", stage="implementation", goal={"id": "REPORT-TASK"},
        authority={"mode": "candidate_only", "completion_authority": False},
        scope={"allowed_paths": list(case["task"].scope_paths)},
        acceptance={"criteria": ["public validation pending"]}, evidence=refs)
    return render_context_capsule(compiled.capsule)


def _invoke(case, **extra):
    request = case["request"]
    return runner.run(prompt=request["prompt"], provider="codex_cli", model="pinned", timeout=1,
        max_output_tokens=128, public_instruction_artifact=request["artifact"],
        public_instruction_sha256=request["expected_sha256"],
        public_instruction_task_cid=request["task_cid"], **extra)


@pytest.mark.parametrize("semantic", [False, True])
def test_real_router_delivers_exact_task_requirements_dependencies_and_source(worker_case, provider, monkeypatch, semantic):
    case = worker_case
    case["request"]["prompt"] = _capsule(case, semantic)
    assert case["source"].decode() not in case["request"]["prompt"]
    monkeypatch.chdir(case["workspace"])
    monkeypatch.setenv("CODEX_HOME", str(case["repository"].parent / "empty-codex-home"))
    text, receipt = _invoke(case, semantic_repository=case["repository"] if semantic else None)
    assert text == "literal summary" and len(provider[1]) == 1
    actual = provider[1][0][0]
    view = _view(actual)
    source = case["source"].decode()
    assert actual.count(source) == 1
    assert view["task_cid"] == case["task"].task_cid
    assert view["task_id"] == "REPORT-TASK"
    assert view["source_path"] == "test_answer.py"
    assert "test_answer.py" not in view["task_contract"]["scope_paths"]
    assert view["task_contract"]["scope_paths"] == ["report.jsonl"]
    assert len(view["requirements"]) == len(view["dependency_requirements"]) == 1
    requirement, dependency = view["requirements"][0], view["dependency_requirements"][0]
    assert requirement["native_atom"]["predicate"] == "create"
    assert requirement["native_atom"]["arguments"] == ["agent", "report.jsonl"]
    assert dependency["native_atom"]["predicate"] == "modify"
    assert dependency["native_atom"]["arguments"] == ["agent", "answer.py"]
    assert requirement["dependency_requirement_ids"] == [dependency["requirement_id"]]
    assert requirement["binding"]["task_cids"] == [case["task"].task_cid]
    assert dependency["binding"]["task_cids"] == [case["tasks"]["LOCAL-TASK"].task_cid]
    for row in (requirement, dependency):
        unit = row["source_unit"]
        assert unit["text"] == source and unit["sha256"] == _sha(case["source"])
        assert [unit[key] for key in ("start_char", "end_char", "start_byte", "end_byte")] == [
            0, len(source), 0, len(case["source"])]
        assert all(row[key] is False for key in AUTHORITY)
    assert view["operations"] == [row for row in case["contract"]["symbolic_operations"]["operations"]
                                  if row["task_key"] == "REPORT-TASK"]
    assert view["task_contract"]["dependency_task_cids"] == [case["tasks"]["LOCAL-TASK"].task_cid]
    assert view["task_contract"]["outputs"] == [row.to_dict() for row in case["task"].outputs]
    assert view["task_contract"]["validations"] == [row.to_dict() for row in case["task"].validations]
    assert view["task_contract"]["acceptance"] == [row.to_dict() for row in case["task"].acceptance]
    assert all(view[key] is False for key in AUTHORITY)
    assert view["native_persistence_verified_here"] is False and view["official_reward"] is None
    inclusion = receipt["public_instruction"]
    assert inclusion["schema"] == "supervisor-public-instruction-inclusion@2"
    summary = inclusion["intent_requirements"]
    assert summary["requirements_context_cid"] == view["requirements_context_cid"]
    assert summary["requirement_ids"] == [requirement["requirement_id"]]
    assert summary["dependency_requirement_ids"] == [dependency["requirement_id"]]
    assert summary["prohibition_requirement_ids"] == []
    assert summary["planning_receipt_cid"] == content_identity(case["admission"]["receipt"])
    assert summary["graph_cid"] == case["proposed"]["graph"].content_id
    assert summary["coverage_cid"] == content_identity(case["admission"]["receipt"]["payload"]["requirement_coverage"])
    assert summary["admission_graph_verified"] is True
    assert summary["native_persistence_verified_here"] is False
    assert receipt["native_prompt_sha256"] == _sha(case["request"]["prompt"].encode())
    assert receipt["model_prompt_sha256"] == _sha(actual.encode())
    parsed = audit._receipt({**receipt, "phase": "coding"}, case["workspace"].parent)
    (case["repository"] / "answer.py").write_text("def answer():\n    return 2\n")
    subprocess.run(["git", "-C", str(case["repository"]), "worktree", "remove", "--force",
                    str(case["workspace"])], check=True)
    monkeypatch.chdir(case["repository"].parent)
    replayed, _, checks = audit._model_projection(rendered=case["request"]["prompt"],
        receipt=parsed, repository=case["repository"])
    assert replayed == actual and all(checks.values())
    assert checks["public_instruction_intent_requirements"] is True


def test_worker_public_replay_never_opens_owner_keys_or_materializes_tasks(worker_case, monkeypatch):
    def owner_access(*args, **kwargs):
        raise AssertionError("worker attempted owner-only access")
    monkeypatch.setattr(local, "load_local_profile", owner_access)
    monkeypatch.setattr(local, "_signed", owner_access)
    monkeypatch.setattr(local, "materialize_local_benchmark_plan", owner_access)
    block, receipt = instruction.load_public_instruction(**worker_case["request"])
    assert _view(block)["task_cid"] == worker_case["task"].task_cid
    assert receipt["intent_requirements"]["admission_graph_verified"] is True
    assert receipt["intent_requirements"]["native_persistence_verified_here"] is False
    assert all(worker_case["intent"].get_task(task.task_cid) is None
               for task in worker_case["proposed"]["graph"].tasks)


def _repin(request, payload):
    payload["context_cid"] = content_identity({key: value for key, value in payload.items() if key != "context_cid"})
    raw = _wire(payload)
    request["artifact"].chmod(0o644)
    request["artifact"].write_bytes(raw)
    request["artifact"].chmod(0o444)
    request["expected_sha256"] = _sha(raw)


@pytest.mark.parametrize("mutation", ["drop_admission", "bindings", "graph", "coverage", "receipt_authority",
    "task_cid", "task_id", "fake_native_claim", "source"])
def test_repinned_public_artifact_cannot_forge_requirement_context(worker_case, provider, monkeypatch, mutation):
    case = worker_case
    payload = json.loads(case["request"]["artifact"].read_text())
    plan = payload["intent_plan_admission"]
    if mutation == "drop_admission":
        del payload["intent_plan_admission"]
    elif mutation == "bindings":
        plan["requirement_bindings"] = []
    elif mutation == "graph":
        plan["graph"]["tasks"][0]["objective"] = "Forged public objective"
    elif mutation == "coverage":
        signed = plan["receipt"]["payload"]
        signed["requirement_coverage"]["bindings"][0]["task_cids"] = ["foreign-task"]
        plan["receipt"] = local._signed(signed, case["manifest"]["payload"])
    elif mutation == "receipt_authority":
        signed = plan["receipt"]["payload"]
        signed["completion_authority"] = True
        plan["receipt"] = local._signed(signed, case["manifest"]["payload"])
    elif mutation == "task_cid":
        payload["task_cid"] = case["tasks"]["LOCAL-TASK"].task_cid
        case["request"]["task_cid"] = payload["task_cid"]
    elif mutation == "task_id":
        payload["task_id"] = "LOCAL-TASK"
        case["request"]["prompt"] = json.dumps({"objective_id": "LOCAL-TASK"})
    elif mutation == "fake_native_claim":
        plan["native_persistence_verified_here"] = True
    else:
        payload["source_path"] = "answer.py"
        source = (case["repository"] / "answer.py").read_bytes()
        payload.update(source_sha256=_sha(source), source_bytes=len(source))
    _repin(case["request"], payload)
    monkeypatch.chdir(case["workspace"])
    monkeypatch.setenv("CODEX_HOME", str(case["repository"].parent / "empty-codex-home"))
    with pytest.raises(ValueError):
        _invoke(case)
    assert provider[1] == []


@pytest.mark.parametrize("where", ["canonical", "workspace"])
def test_requirement_source_drift_refuses_dispatch_before_provider(worker_case, provider, monkeypatch, where):
    case = worker_case
    root = case["repository"] if where == "canonical" else case["workspace"]
    (root / "test_answer.py").write_text("altered immutable task requirements\n")
    monkeypatch.chdir(case["workspace"])
    monkeypatch.setenv("CODEX_HOME", str(case["repository"].parent / "empty-codex-home"))
    with pytest.raises(ValueError, match="stale"):
        _invoke(case)
    assert provider[1] == []


def test_historical_replay_preserves_exact_context_without_fresh_dispatch(worker_case, provider, monkeypatch):
    case = worker_case
    original, _ = instruction.load_public_instruction(**case["request"])
    (case["repository"] / "answer.py").write_text("def answer():\n    return 3\n")
    request = {**case["request"], "workspace": None, "require_current_source": False}
    replayed, receipt = instruction.load_public_instruction(**request)
    assert replayed == original and _view(replayed) == _view(original)
    assert receipt["historical_replay"] is True and receipt["source_freshness_verified"] is False
    assert receipt["intent_requirements"]["native_persistence_verified_here"] is False
    monkeypatch.chdir(case["workspace"])
    with pytest.raises(ValueError, match="stale"):
        _invoke(case)
    assert provider[1] == []


def test_fractional_candidate_metadata_stays_bound_without_semantic_authority(scenario, tmp_path):
    contract, manifest, proposed = _prepare(scenario, confidence=0.75)
    admission = local.admit_local_benchmark_plan(graph=proposed["graph"], manifest=manifest,
        requirement_bindings=proposed["requirement_bindings"])
    case = _context(scenario, admission, proposed["graph"].tasks[0], tmp_path)
    block, _ = instruction.load_public_instruction(**case["request"])
    view = _view(block)
    assert view["contract_cid"] == cid_for_dag_json(contract)
    assert view["requirements"][0]["source_semantics_verified"] is False
    assert contract["ledger"]["source_report"]["candidates"][0]["candidate_intent_ir"]["statements"][0]["confidence"] == 0.75


def test_global_prohibitions_are_delivered_without_becoming_tasks_or_bindings(scenario, tmp_path):
    from ipfs_datasets_py.logic.intent_ir.formalize.requirements import build_intent_requirement_ledger

    contract = _reviewed_contract(scenario)
    report = deepcopy(contract["ledger"]["source_report"])
    document = report["candidates"][0]["candidate_intent_ir"]
    prohibited = deepcopy(document["statements"][0])
    prohibited.update(statement_id="private-unchanged", modality="prohibited",
                      normalized_text="The private file must remain unchanged.", arguments=["agent", "private.txt"])
    document["statements"].append(prohibited)
    report["report_sha256"] = _sha(_wire({key: value for key, value in report.items() if key != "report_sha256"}))
    source = (scenario["repository"] / "test_answer.py").read_text()
    ledger = build_intent_requirement_ledger(source, source_report=report,
        source_identity=contract["ledger"]["source"]["source_identity"])
    by_statement = {row["statement_ids"][0]: row for row in ledger["requirements"]}
    mandatory_id = by_statement["required-answer"]["requirement_id"]
    prohibition_id = by_statement["private-unchanged"]["requirement_id"]
    contract["ledger"] = ledger
    contract["requirements"][0]["requirement_id"] = mandatory_id
    contract["requirements"].append({"requirement_id": prohibition_id,
        "outputs": [{"path": "private.txt", "effect": "modify", "media_type": "text/plain"}],
        "validation_keys": [], "dependency_requirement_ids": []})
    manifest = _author(scenario, contract)
    admission = local.admit_local_benchmark_plan(graph=scenario["graph"], manifest=manifest,
        requirement_bindings=[{"requirement_id": mandatory_id,
            "task_keys": ["LOCAL-TASK"], "validation_keys": ["public-answer"]}])
    case = _context(scenario, admission, scenario["graph"].tasks[0], tmp_path)
    block, receipt = instruction.load_public_instruction(**case["request"])
    view = _view(block)
    assert view["requirement_ids"] == [mandatory_id]
    assert [row["requirement_id"] for row in view["global_prohibitions"]] == [prohibition_id]
    assert view["global_prohibitions"][0]["native_atom"]["arguments"] == ["agent", "private.txt"]
    assert view["global_prohibitions"][0]["binding"] is None
    assert receipt["intent_requirements"]["prohibition_requirement_ids"] == [prohibition_id]
    assert len(admission["graph"]["tasks"]) == 1 and view["operations"] == []


def _audit_receipt(case, monkeypatch):
    case["request"]["prompt"] = _capsule(case, False)
    monkeypatch.chdir(case["workspace"])
    monkeypatch.setenv("CODEX_HOME", str(case["repository"].parent / "empty-codex-home"))
    _, receipt = _invoke(case)
    return {**receipt, "phase": "coding"}


@pytest.mark.parametrize("field", ["admission_graph_verified", "native_persistence_verified_here",
    "source_semantics_verified", "semantic_alignment_verified", "proof_authority", "execution_authority",
    "completion_authority", "requirements_context_cid", "graph_cid", "ledger_sha256", "requirement_ids"])
def test_terminal_audit_rejects_invalid_intent_inclusion_authority_or_identity(worker_case, provider, monkeypatch, field):
    case = worker_case
    receipt = _audit_receipt(case, monkeypatch)
    summary = receipt["public_instruction"]["intent_requirements"]
    if field == "admission_graph_verified":
        summary[field] = False
    elif field in {"requirements_context_cid", "graph_cid"}:
        summary[field] = "unbounded identity with spaces"
    elif field == "ledger_sha256":
        summary[field] = "not-a-digest"
    elif field == "requirement_ids":
        summary[field] *= 2
    else:
        summary[field] = True
    with pytest.raises(ValueError):
        audit._receipt(receipt, case["workspace"].parent)
    assert len(provider[1]) == 1


@pytest.mark.parametrize("field", ["graph_cid", "planning_receipt_cid", "requirement_ids",
                                   "dependency_requirement_ids", "prohibition_requirement_ids"])
def test_terminal_historical_projection_compares_complete_requirement_summary(worker_case, provider, monkeypatch, field):
    case = worker_case
    receipt = _audit_receipt(case, monkeypatch)
    summary = receipt["public_instruction"]["intent_requirements"]
    foreign_cid = content_identity({"independently_different_requirement_identity": True})
    summary[field] = [foreign_cid] if field.endswith("_ids") else foreign_cid
    parsed = audit._receipt(receipt, case["workspace"].parent)
    replayed, _, checks = audit._model_projection(rendered=case["request"]["prompt"],
        receipt=parsed, repository=case["repository"])
    assert replayed == provider[1][0][0]
    assert checks["model_prompt_sha256"] is True
    assert checks["public_instruction_intent_requirements"] is False
    assert not all(checks.values())
