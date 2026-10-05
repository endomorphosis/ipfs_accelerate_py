"""Local administrative qualification; no coding provider or numerical proof.

Reads a separately retained public benchmark checkout. Full instruction and
candidate contracts stay in a new private artifact directory, never in a
training corpus. The authored atoms are interpretation candidates, not human
reviewed or semantically verified translations. Run with the native datasets
checkout on PYTHONPATH and DuckDB 1.5.5; the normal planner is not mocked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from ipfs_accelerate_py.agent_supervisor.planning.intent_symbolic_planning import build_intent_symbolic_plan
from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import build_intent_requirement_contract
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_datasets_py.logic.intent_ir.formalize.requirements import build_intent_requirement_ledger
from ipfs_datasets_py.logic.intent_ir.schema import (
    IntentIRDocument, IntentKind, IntentModality, IntentStatement, NodeGrounding,
    ReviewStatus, SourceRef, SourceSpan, StatementKind,
)

EXPECTED_INSTRUCTION = "6fa938389f99648508e5a9772e0d5e95d1102738c68d1ff61bd65469d76c52ab"
# One or more independent declarative atoms per complete public source line.
# Permission and suggested tooling are accounted for without becoming duties.
ATOMS = (
    (("implement_entrypoint", ("eigen.py", "find_dominant_eigenvalue_and_eigenvector"),
      "Complete the specified Python eigenpair entrypoint."),),
    (("dominant_modulus", ("eigenvalue", "largest_magnitude"),
      "Return an eigenvalue having maximal absolute value."),),
    (("input_domain", ("square_2d", "real_float64", "size_at_most_10"),
      "Support the stated square real float64 array domain up to size 10."),
     ("general_complex_eigenpair", ("nonsymmetric", "complex_output_allowed"),
      "Support nonsymmetric real matrices whose eigenpair may be complex.")),
    (("faster_than_public_reference", ("eval.py", "consistently_faster"),
      "Optimize for consistent speed improvement relative to the public reference."),
     ("public_residual", ("A_times_v", "lambda_times_v", "numpy_allclose"),
      "Satisfy the stated approximate eigenpair residual predicate.")),
    (("median_call_timing", ("multiple_tests", "median_per_call"),
      "Account for the stated repeated median per-call timing criterion."),),
    (("python_entrypoint_retained", ("eigen.py", "python_function"),
      "Retain the Python entrypoint despite optional packages or other languages."),),
    (),
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_binding(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": digest(raw)}


def author_contract(text):
    if digest(text.encode()) != EXPECTED_INSTRUCTION:
        raise ValueError("This authored interpretation only binds the pinned public instruction")
    lines = text.splitlines(keepends=True)
    if len(lines) != len(ATOMS):
        raise ValueError("Public source line inventory changed")
    report = {
        "schema": "intent-reviewed-source-report@1", "source_sha256": EXPECTED_INSTRUCTION,
        "source_bytes": len(text.encode()), "source_characters": len(text),
        "producer": {"name": "agent-authored-administrative-eigenvalue-candidate", "revision": "1"},
        "interpretation_status": "reviewed_candidate", "units": [], "candidates": [],
        "proof_authority": False, "execution_authority": False,
        "completion_authority": False, "source_semantics_verified": False,
    }
    documents = {}
    offset = 0
    for index, (line, atoms) in enumerate(zip(lines, ATOMS), 1):
        raw = line.encode()
        unit_id = f"unit:public-line-{index}"
        report["units"].append({
            "unit_id": unit_id, "start_char": offset, "end_char": offset + len(line),
            "start_byte": len(text[:offset].encode()), "end_byte": len(text[:offset + len(line)].encode()),
            "text": line, "sha256": digest(raw),
            "disposition": "interpreted_candidate" if atoms else "non_requirement",
            "reason": "agent_authored_candidate_atoms" if atoms else "optional_evaluator_suggestion",
        })
        offset += len(line)
        if not atoms:
            continue
        source = SourceRef("source", "instruction:public-candidate", digest(raw), digest(raw),
                           digest(raw), review_status=ReviewStatus.MACHINE_EXTRACTED,
                           span=SourceSpan(0, len(line)))
        statements = tuple(IntentStatement(
            statement_id=predicate, kind=StatementKind.GOAL, modality=IntentModality.REQUIRED,
            normalized_text=description, source_ref_ids=("source",), predicate=predicate,
            arguments=arguments, confidence=0.0, review_status=ReviewStatus.MACHINE_EXTRACTED,
            grounding=NodeGrounding.INFERRED,
        ) for predicate, arguments, description in atoms)
        document = IntentIRDocument(f"document:public-line-{index}", "Authored eigenvalue requirements",
            IntentKind.DECLARATIVE, (source,), statements,
            tags=("authored-administrative-candidate", "held-out-benchmark"))
        document.validate()
        documents[unit_id] = document.to_dict()
        report["candidates"].append({"unit_id": unit_id, "candidate_intent_ir": document.to_dict()})
    report["report_sha256"] = digest(canonical(report))
    ledger = build_intent_requirement_ledger(text, source_report=report,
        source_identity={"path": prep.INSTRUCTION, "revision": EXPECTED_INSTRUCTION})
    output = [{"path": "eigen.py", "effect": "modify", "media_type": "text/x-python"}]
    grounds, matchers = [], []
    for requirement in ledger["requirements"]:
        grounds.append({"requirement_id": requirement["requirement_id"], "outputs": output,
                       "validation_keys": ["public-structural-smoke"], "dependency_requirement_ids": []})
        document = documents[requirement["source_unit_id"]]
        for statement_id in requirement["statement_ids"]:
            statement = next(row for row in document["statements"] if row["statement_id"] == statement_id)
            matchers.append({"requirement_id": requirement["requirement_id"],
                "native_document_sha256": requirement["native_document_sha256"],
                "statement_id": statement_id, "predicate": statement["predicate"],
                "arguments": statement["arguments"], "modality": statement["modality"]})
    operations = {
        "schema": "intent-symbolic-operation-contract@1", "ledger_sha256": ledger["ledger_sha256"],
        "review_ref": "reviewed:agent-authored-administrative-candidate@1",
        "interpretation_scope": "administrative_requirement_task_coverage",
        "operations": [{"operation_id": "operation:implement-eigenpair", "task_key": "TB-CODE-TASK",
            "matchers": matchers, "outputs": output, "validation_keys": ["public-structural-smoke"],
            "dependency_operation_ids": []}],
        "semantic_alignment_verified": False, "proof_authority": False,
        "execution_authority": False, "completion_authority": False,
    }
    return build_intent_requirement_contract(source_path=prep.INSTRUCTION, ledger=ledger,
        requirements=grounds, source_text=text, symbolic_operations=operations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    artifact.mkdir(parents=True, exist_ok=False)
    app = artifact / "app"
    app.mkdir()
    text = (args.task / "instruction.md").read_text()
    instruction = artifact / "instruction.md"
    instruction.write_text(text)
    for name in ("eigen.py", "eval.py"):
        shutil.copyfile(args.task / "environment/src" / name, app / name)
    inputs = {name: file_binding(app / name) for name in ("eigen.py", "eval.py")}
    for argv in (("init", "-q"), ("add", "."), ("-c", "user.name=Qualification",
            "-c", "user.email=qualification@example.invalid", "commit", "-qm", "Public source inputs")):
        subprocess.run(["git", "-C", str(app), *argv], check=True, capture_output=True)
    contract = author_contract(text)
    contract_path = artifact / "private-contract.json"
    write(contract_path, contract)
    profile = {"schema": "terminal-public-task-profile@1", "instruction_sha256": EXPECTED_INSTRUCTION,
        "input_paths": ["eigen.py", "eval.py"], "outputs": [{"effect": "modify",
        "media_type": "text/x-python", "path": "eigen.py"}]}
    started = time.monotonic()
    prepared = prep.prepare(repository=app, instruction=instruction, state=artifact / "state",
        intent_requirement_contract=contract_path, task_profile=profile,
        resource_profile="source384-5cpu-16gib-extended@1", disable_intent_autoencoder=True)
    prepare_seconds = time.monotonic() - started
    result = prep.plan(artifact / "state")
    write(artifact / "private-planning-result.json", result)
    if not result.get("qualified"):
        raise RuntimeError(str(result.get("failure", result)))
    assert result["provider_calls"] == 0
    assert result["planning_strategy"] == "intent_symbolic"
    assert prepared["request"]["planning_policy"]["allow_model"] is False
    assert result["requirement_coverage"]["accepted"] is True
    replay = build_intent_symbolic_plan(contract, manifest=prepared["manifest"])
    assert replay["receipt"] == result["symbolic_planning"]
    admission = json.loads((artifact / "state/admission.json").read_text())
    local.verify_local_benchmark_admission(admission)
    with IntentRepository(artifact / "state/intent.duckdb") as intent:
        stored = [dict(intent.get_task(cid)) for cid in result["task_cids"]]
        for task in stored:
            pending, manifest, _, _ = local._contract(task["body"], task["task_cid"])
            assert local.decode_intent_requirement_contract(manifest) == contract
            assert pending["intent_plan"]["requirement_bindings"] == admission["requirement_bindings"]
    assert inputs == {name: file_binding(app / name) for name in inputs}
    assert not (artifact / "state/provider-request.json").exists()
    bound = {
        "schema": "largest-eigenval-symbolic-planning-qualification@1", "status": "qualified",
        "scope": "native administrative planning, replay, admission and task storage only",
        "instruction_sha256": EXPECTED_INSTRUCTION, "input_files": inputs,
        "contract": file_binding(contract_path), "contract_cid": result["requirement_coverage"]["contract_cid"],
        "authoring": "agent-authored interpretation candidate; no human semantic review",
        "training_exclusion": True, "benchmark_success": None, "coding_executed": False,
        "provider_calls": result["provider_calls"], "planning_strategy": result["planning_strategy"],
        "prepare_seconds": prepare_seconds, "planning_seconds": result["elapsed_seconds"],
        "requirements": len(contract["requirements"]), "goals": result["goals"], "tasks": result["tasks"],
        "requirement_atoms": [row[0] for group in ATOMS for row in group],
        "requirement_coverage": result["requirement_coverage"], "symbolic_planning": result["symbolic_planning"],
        "native_task_cids": result["task_cids"], "native_storage_verified": True,
        "deterministic_replay_verified": True, "public_program_bytes_unchanged": True,
        "admission": file_binding(artifact / "state/admission.json"),
        "stored_tasks_sha256": digest(canonical(stored)), "producer": file_binding(Path(__file__)),
        "source_semantics_verified": False, "semantic_alignment_verified": False,
        "proof_authority": False, "execution_authority": False, "completion_authority": False,
        "limitations": ["The structural smoke does not verify eigenpair behavior or speed.",
            "Requirements are authored candidates, not learned translations or numerical certificates.",
            "No official symbolic-planning benchmark trial or coding token saving measured.",
            "Optional autoencoder preprocessing explicitly disabled in this local qualification."],
    }
    write(artifact / "qualification.json", bound)
    print(json.dumps({key: bound[key] for key in ("status", "provider_calls", "requirements", "tasks",
                                                  "planning_strategy", "planning_seconds")}))


if __name__ == "__main__":
    main()
