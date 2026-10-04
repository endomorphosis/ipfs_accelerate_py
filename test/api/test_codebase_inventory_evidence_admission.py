"""Real local signatures/native pending tasks around typed authored history.

These controls qualify the additive join and worker replay. The completed scan
producer and current native source/model validation have their own tests and
integration qualification; no training, inference or solver is run here.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    PromptAcceptanceRecord, PromptGoalGraph, PromptGoalRecord, PromptOutputRecord,
    PromptTaskRecord, PromptValidationRecord,
)
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction as instruction
from ipfs_accelerate_py.agent_supervisor.runtime import codebase_inventory_evidence_admission as joined
from ipfs_accelerate_py.agent_supervisor.runtime import codebase_inventory_evidence_worker_context as worker
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository


def _git(root, *argv):
    return subprocess.check_output(["git", "-C", str(root), *argv], text=True).strip()


@pytest.fixture
def inventory_case(tmp_path, monkeypatch):
    """Typed historical records; local signer/graph/DB/Git are real."""
    root = tmp_path / "repository"
    root.mkdir()
    (root / "answer.py").write_text("def answer():\n    return 1\n")
    (root / "test_answer.py").write_text("from answer import answer\nassert answer() == 2\n")
    for number in range(298):
        (root / f"member_{number:03d}.py").write_text(f"VALUE = {number}\n")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=Inventory controls", "-c", "user.email=controls@example.invalid",
         "commit", "-qm", "independent fixture")
    profile, lifecycle = tmp_path / "profile", tmp_path / "lifecycle"
    Supervisor.init_local(repository=root, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
    policy = content_identity(local.LOCAL_POLICY)
    check = PromptValidationRecord(validation_key="public-answer", argv=(sys.executable, "test_answer.py"),
                                   policy_cid=policy)
    acceptance = PromptAcceptanceRecord(criterion_key="answer-is-two", criterion="The public check must pass",
                                        validation_keys=(check.validation_key,))
    goal = PromptGoalRecord(goal_key="ALL-TASKS", parent_goal_cid="", dependency_goal_cids=(),
        title="Repair answer", objective="Retain both tasks",
        rationale="Independent administrator data", scope_paths=("answer.py", "test_answer.py"),
        acceptance=(acceptance,))
    tasks = []
    for key in ("INVENTORY-FIRST", "INVENTORY-SECOND"):
        tasks.append(PromptTaskRecord(task_key=key, goal_cid=goal.goal_cid,
            dependency_task_cids=() if not tasks else (tasks[0].task_cid,), objective="Repair public answer",
            rationale="Independent declared work", scope_paths=("answer.py", "test_answer.py"),
            outputs=(PromptOutputRecord(path="answer.py", effect="modify", media_type="text/x-python"),),
            validations=(check,), acceptance=(acceptance,), evidence_cids=(),
            policy_roots=(policy,), predicted_files=("answer.py",)))
    roots = {name: content_identity({"independent": name})
             for name in ("request_cid", "scan_cid", "program_root")}
    graph = PromptGoalGraph(**roots, policy_roots=(policy,), goals=(goal,), tasks=tuple(tasks), evidence=())
    specs = [{"task_key": task.task_key, "scope_paths": list(task.scope_paths),
        "outputs": [{key: getattr(output, key) for key in ("path", "effect", "media_type")}
                    for output in task.outputs],
        "validations": [{key: local._plain(getattr(check, key)) for key in
            ("validation_key", "argv", "cwd", "expected_exit_codes", "policy_cid")}],
        "acceptance": [{key: local._plain(getattr(acceptance, key)) for key in
            ("criterion_key", "criterion", "evidence_cids", "validation_keys")}],
        "dependencies": [] if position == 0 else [tasks[0].task_key]} for position, task in enumerate(tasks)]
    # These typed source/head/page identities are explicitly authored history.
    # Current native CAS/model/page reconciliation belongs to the receiver.
    from test.api.test_codebase_inventory_evidence_worker_context import native_refs
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
    names = sorted(filter(None, _git(root, "ls-files", "-z").split("\0")))
    context = native_refs.__wrapped__()
    body = context["scan"]["root_record"]
    body["members"] = [{"path": name, "raw_path_hex": name.encode().hex(),
        "source_key": "raw:" + name.encode().hex(), "entry_cid": cid_for_structured({"authored-entry": name}),
        "source_cid": cid_for_bytes((root / name).read_bytes()), "ast_cid": cid_for_structured({"authored-AST": name}),
        "parse_status": "ok", "source_size_bytes": (root / name).stat().st_size, "opaque_reason": None} for name in names]
    body["membership_cid"] = cid_for_structured(body["members"])
    body["limits"] = resume.CodebaseScanResumeLimits().to_dict()
    typed_root = resume.CodebaseScanResumeRoot.from_dict(cid_for_structured(body), body)
    complete = context["scan"]["completion_record"]
    complete.update(root_cid=typed_root.artifact_cid, membership_cid=body["membership_cid"], pages=[])
    for start in range(0, len(names), 16):
        end = min(start + 16, len(names))
        complete["pages"].append({"page_cid": cid_for_bytes(f"authored-page:{start}".encode()),
            "start": start, "end": end, "membership_cid": cid_for_structured(body["members"][start:end]),
            "inferred_rows": 0, "dispositions": {"deferred_budget": end - start}})
    complete["coverage"] = {"inventory_entries": 300, "inferred_rows": 0, "pages": len(complete["pages"]),
        "dispositions": {"deferred_budget": 300}}
    typed_complete = resume.CodebaseScanResumeCompletion.from_dict(cid_for_structured(complete), complete)
    context["scan"] = typed_complete.advisory_refs(typed_root)
    manifest = local.author_local_benchmark_manifest(repository=root, profile_dir=profile,
        lifecycle_dir=lifecycle, task_specs=specs, planning_roots=roots, codebase_inventory_context=context)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=manifest)
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        yield {"repository": root, "profile": profile, "lifecycle": lifecycle, "context": context,
            "manifest": manifest, "graph": graph, "admission": admission, "intent": intent,
            "tasks": tasks, "specs": specs, "roots": roots}


def test_explicit_large_profile_preserves_old_caps_and_all_pending_tasks(inventory_case):
    case = inventory_case
    assert len(case["manifest"]["payload"]["sources"]) == 300
    assert case["manifest"]["payload"]["schema"] == local.INVENTORY_MANIFEST_SCHEMA
    receipt = local.verify_local_benchmark_admission(case["admission"])["receipt"]
    assert receipt["schema"] == local.INVENTORY_PLANNING_RECEIPT_SCHEMA
    assert receipt["administrator_task_cids"] == sorted(task.task_cid for task in case["tasks"])
    assert receipt["current_facts"] == receipt["removed_task_cids"] == []
    assert receipt["runtime_requirements_preserved"] is True
    assert all(row["required"] is True and row["phase"] == "post_execution" for row in receipt["pending_requirements"])
    for schema in (local.MANIFEST_SCHEMA, local.CREATE_MANIFEST_SCHEMA, local.INTENT_MANIFEST_SCHEMA):
        payload = deepcopy(case["manifest"]["payload"])
        payload["schema"] = schema
        payload.pop("codebase_inventory_context")
        if schema == local.MANIFEST_SCHEMA:
            payload.pop("created_outputs")
        elif schema == local.INTENT_MANIFEST_SCHEMA:
            payload["intent_requirements"] = {}
        with pytest.raises(ValueError, match="inventory bound"):
            local._validate_local_manifest_declarations(payload)


def test_large_profile_member_cap_and_exact_coverage_are_separate(inventory_case):
    payload = deepcopy(inventory_case["manifest"]["payload"])
    for number in range(300, 1024):
        name = f"extra_{number}.py"
        payload["sources"][name] = {"sha256": "a" * 64, "executable": False}
        payload["codebase_inventory_context"]["scan"]["member_paths"].append(name)
    assert len(payload["sources"]) == local.source_inventory_limit(payload) == 1024
    # Count dispatch is independent of the genuine completion partition; a
    # mismatched complete scan cannot be accepted just because it is ≤1024.
    with pytest.raises(ValueError, match="complete unique signed member paths"):
        local._validate_local_manifest_declarations(payload)
    payload["sources"]["too_many.py"] = {"sha256": "a" * 64, "executable": False}
    with pytest.raises(ValueError, match="inventory bound"):
        local._validate_local_manifest_declarations(payload)
    payload["sources"].pop("too_many.py")
    payload = deepcopy(inventory_case["manifest"]["payload"])
    payload["sources"].pop("member_000.py")
    with pytest.raises(ValueError, match="complete scan member population"):
        local._validate_local_manifest_declarations(payload)


@pytest.mark.parametrize("change", ["omitted_task", "validation", "scope", "context", "numeric_receipt", "numeric_authority"])
def test_resigned_receipt_or_declaration_cannot_change_full_contract(inventory_case, change):
    case = inventory_case
    admission = deepcopy(case["admission"])
    if change == "omitted_task":
        admission["graph"]["tasks"].pop()
    elif change == "validation":
        admission["manifest"]["payload"]["tasks"][0]["validations"][0]["argv"] = ["true"]
    elif change == "scope":
        admission["manifest"]["payload"]["tasks"][0]["scope_paths"].append("member_000.py")
    elif change in {"context", "numeric_receipt"}:
        if change == "context":
            admission["receipt"]["payload"]["codebase_inventory_context_cid"] = content_identity({"other": 1})
        else:
            admission["receipt"]["payload"]["runtime_requirements_preserved"] = 1
        admission["receipt"] = local._signed(admission["receipt"]["payload"], admission["manifest"]["payload"])
    else:
        admission["manifest"]["payload"]["codebase_inventory_context"]["authority"]["proof_authority"] = 0
    if change not in {"context", "numeric_receipt"}:
        admission["manifest"] = local._signed(admission["manifest"]["payload"], admission["manifest"]["payload"])
    with pytest.raises(ValueError):
        local.verify_local_benchmark_admission(admission)


def test_native_full_population_and_receipt_reference_bind_inventory(inventory_case):
    case = inventory_case
    result = local.materialize_local_benchmark_plan(admission=case["admission"], intent=case["intent"])
    assert sorted(result["task_cids"]) == sorted(task.task_cid for task in case["tasks"])
    reference = case["intent"].get_plan(result["plan_id"])["body"]["local_planning_receipt_ref"]
    assert reference["schema"] == local.INVENTORY_PLANNING_RECEIPT_REFERENCE_SCHEMA
    assert reference["codebase_inventory_context_cid"] == content_identity(case["context"])
    assert local.load_local_planning_receipt(reference, manifest=case["manifest"]) == case["admission"]["receipt"]
    for task in case["tasks"]:
        native = case["intent"].get_task(task.task_cid)
        contract, manifest, _, _ = local._contract(native["body"], task.task_cid)
        assert contract["schema"] == local.CONTRACT_SCHEMA
        assert manifest["codebase_inventory_context"] == worker.inventory_declaration(case["context"])
        assert len(json.dumps(native["body"], separators=(",", ":")).encode()) < 262_144
        assert contract["task_spec"] == next(spec for spec in case["specs"] if spec["task_key"] == task.task_key)


def test_closing_receiver_fault_rolls_back_native_population(inventory_case, monkeypatch):
    case = inventory_case
    @contextmanager
    def controlled_scope(**_):
        def closing():
            raise ValueError("closing native receiving fault")
        yield worker.inventory_declaration(case["context"]), closing
    monkeypatch.setattr(joined, "_current_scope", controlled_scope)
    with pytest.raises(ValueError, match="closing native receiving fault"):
        joined.materialize_current_inventory_plan(root=object(), completion=object(), index=object(),
            repository=case["repository"], registry=object(), admission=case["admission"], intent=case["intent"])
    assert all(case["intent"].get_task(task.task_cid) is None for task in case["tasks"])
    plan_id = case["admission"]["receipt"]["payload"]["plan_id"]
    assert case["intent"].get_plan(plan_id) is None


def test_public_worker_replays_without_private_keys_or_native_owner(inventory_case, monkeypatch, tmp_path):
    case = inventory_case
    task = case["tasks"][0]
    source = (case["repository"] / "test_answer.py").read_bytes()
    selected = instruction.prepare_public_instruction_context(repository=case["repository"],
        admission=case["admission"], task_cid=task.task_cid, source_path="test_answer.py",
        expected_source_sha256=hashlib.sha256(source).hexdigest(), codebase_inventory_context=case["context"])
    workspace = tmp_path / "worker"
    _git(case["repository"], "worktree", "add", "--detach", "-q", str(workspace))
    def forbidden(*_, **__):
        raise AssertionError("worker attempted owner private state")
    monkeypatch.setattr(local, "load_local_profile", forbidden)
    monkeypatch.setattr(local, "_signed", forbidden)
    block, receipt = instruction.load_public_instruction(artifact=Path(selected["artifact"]),
        expected_sha256=selected["sha256"], task_cid=task.task_cid,
        prompt=json.dumps({"objective_id": task.task_key}), workspace=workspace)
    assert source.decode() in block and "CODEBASE INVENTORY ADVISORY" in block
    assert receipt["schema"] == "supervisor-public-instruction-inclusion@3"
    refs = receipt["codebase_inventory"]
    assert refs["administrator_task_cids"] == sorted(task.task_cid for task in case["tasks"])
    assert refs["pending_cid"] == case["admission"]["receipt"]["payload"]["pending_cid"]
    assert refs["native_inventory_current_verified_here"] is False
    assert refs["native_persistence_verified_here"] is False
    assert refs["current_facts"] == refs["removed_task_cids"] == []
    assert all(flag is False for flag in refs["authority"].values())
    (workspace / "undeclared.py").write_text("VALUE = 1\n")
    with pytest.raises(ValueError, match="undeclared new source"):
        instruction.load_public_instruction(artifact=Path(selected["artifact"]),
            expected_sha256=selected["sha256"], task_cid=task.task_cid,
            prompt=json.dumps({"objective_id": task.task_key}), workspace=workspace)


@pytest.mark.parametrize("key", ["proof_authority", "scan_execution_attested", "completion_authority"])
def test_context_false_flags_reject_numeric_aliases(inventory_case, key):
    value = deepcopy(inventory_case["context"])
    value["authority"][key] = 0
    with pytest.raises(ValueError, match="cannot acquire authority"):
        worker.validate_inventory_context(value)


def test_inventory_artifact_reader_retains_explicit_large_bound(tmp_path):
    artifact = tmp_path / "retained.json"
    raw = b"x" * 2_000_001
    artifact.write_bytes(raw)
    artifact.chmod(0o444)
    descriptor = instruction._directory(tmp_path)
    try:
        actual, info = instruction._read_public_artifact(descriptor, artifact.name)
        assert actual == raw and info.st_size == len(raw)
    finally:
        import os
        os.close(descriptor)
    artifact.chmod(0o644)
    with artifact.open("r+b") as stream:
        stream.truncate(instruction.MAX_INVENTORY_BYTES + 1)
    descriptor = instruction._directory(tmp_path)
    try:
        with pytest.raises(ValueError, match="bounded single-link"):
            instruction._read_public_artifact(descriptor, artifact.name)
    finally:
        os.close(descriptor)
