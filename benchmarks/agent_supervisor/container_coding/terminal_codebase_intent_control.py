"""Keep the public request unresolved beside a separate authored planning control.

No prompt frontend, learned decoder or proof index supplies an interpretation
here. The native atomic IntentIR is explicitly authored development input. Its
operation means administrative task coverage, never satisfied software behavior.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath

SCHEMA = "terminal-codebase-intent-control@1"
PUBLIC_INSTRUCTION_SHA256 = "89b1abdf0af19399f720233b5cdf6b2e8e0ff7b6e3fe3e2a3af0d4d298a8641a"
AUTHORED_CONTROL_TEXT = "agent must repair bottle."
AUTHORED_CONTROL_PATH = ".supervisor-authored-intent-control.md"
VALIDATION_KEY = "public-structural-smoke"
TASK_KEY = "TB-CODE-TASK"
_AUTHORITY = {"semantic_alignment_verified": False, "source_semantics_verified": False,
    "proof_authority": False, "execution_authority": False, "completion_authority": False,
    "mutation_authority": False, "official_reward_measured": False}


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _source_report(text, *, identity, document=None):
    raw = text.encode()
    report = {"schema": "intent-reviewed-source-report@1", "source_sha256": _sha(raw),
        "source_bytes": len(raw), "source_characters": len(text),
        "producer": {"name": "terminal-explicit-authored-control" if document else
                         "terminal-public-unresolved-source-accounting", "revision": "1"},
        "interpretation_status": "reviewed_candidate",
        "units": [{"unit_id": identity, "start_char": 0, "end_char": len(text),
            "start_byte": 0, "end_byte": len(raw), "text": text, "sha256": _sha(raw),
            "disposition": "interpreted_candidate" if document else "unsupported",
            "reason": "explicitly_authored_atomic_development_control" if document else
                "public_request_not_formalized; no admitted IntentIR frontend executed"}],
        "candidates": [{"unit_id": identity, "candidate_intent_ir": document}] if document else [],
        "proof_authority": False, "execution_authority": False,
        "completion_authority": False, "source_semantics_verified": False}
    report["report_sha256"] = _sha(_wire(report))
    return report


def _authored_document():
    from ipfs_datasets_py.logic.intent_ir.schema import (
        IntentIRDocument, IntentKind, IntentStatement, StatementKind, IntentModality,
        IntentAction, SourceRef, SourceSpan, NodeGrounding, ReviewStatus,
    )
    from ipfs_datasets_py.logic.intent_ir.decoder import decode_intent_ir
    digest = _sha(AUTHORED_CONTROL_TEXT.encode())
    source = SourceRef(ref_id="authored-control-source",
        source_uri="authored-development-control:" + AUTHORED_CONTROL_PATH,
        source_id=digest, source_revision=digest, content_sha256=digest,
        review_status=ReviewStatus.TRUSTED_FIXTURE,
        span=SourceSpan(0, len(AUTHORED_CONTROL_TEXT)))
    statement = IntentStatement(statement_id="authored-repair-goal", kind=StatementKind.GOAL,
        modality=IntentModality.REQUIRED, normalized_text=AUTHORED_CONTROL_TEXT,
        source_ref_ids=(source.ref_id,), predicate="repair", arguments=("agent", "bottle"),
        confidence=0.0, review_status=ReviewStatus.TRUSTED_FIXTURE, grounding=NodeGrounding.GROUNDED)
    action = IntentAction(action_id="authored-repair-action", actor="agent", verb="repair",
        object_refs=("bottle",), source_ref_ids=(source.ref_id,), grounding=NodeGrounding.GROUNDED)
    document = IntentIRDocument(document_id="terminal-authored-control:" + digest,
        title="Explicitly authored atomic development control", intent_kind=IntentKind.DECLARATIVE,
        sources=(source,), statements=(statement,), actions=(action,),
        entry_action_ids=(action.action_id,), terminal_action_ids=(action.action_id,),
        tags=("authored-development-control", "administrative-task-coverage-only"))
    return decode_intent_ir(document.to_dict()).to_dict()


def build_terminal_intent_control(*, public_instruction_bytes,
                                 public_instruction_path=".supervisor-instruction.md"):
    """Build native ledgers and a v2 authored operation without filesystem writes."""
    from ipfs_datasets_py.logic.intent_ir.formalize.requirements import build_intent_requirement_ledger
    from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import build_intent_requirement_contract
    if (type(public_instruction_bytes) is not bytes or not 0 < len(public_instruction_bytes) <= 32_768
            or _sha(public_instruction_bytes) != PUBLIC_INSTRUCTION_SHA256):
        raise ValueError("exact independently captured public Bottle instruction required")
    if (type(public_instruction_path) is not str or not public_instruction_path
            or len(public_instruction_path) > 4096 or "\\" in public_instruction_path
            or PurePosixPath(public_instruction_path).is_absolute()
            or ".." in PurePosixPath(public_instruction_path).parts
            or str(PurePosixPath(public_instruction_path)) != public_instruction_path
            or any(char in public_instruction_path for char in "\r\n\0")
            or ".git" in PurePosixPath(public_instruction_path).parts
            or public_instruction_path == AUTHORED_CONTROL_PATH):
        raise ValueError("distinct canonical source-relative public instruction path required")
    text = public_instruction_bytes.decode("utf-8")
    public_report = _source_report(text, identity="unit:exact-public-instruction")
    public_ledger = build_intent_requirement_ledger(text, source_report=public_report,
        source_identity={"path": public_instruction_path, "revision": PUBLIC_INSTRUCTION_SHA256,
                         "origin": "exact_public_terminal_bench_instruction"})
    public_contract = build_intent_requirement_contract(source_path=public_instruction_path,
        ledger=public_ledger, requirements=[], source_text=text)
    document = _authored_document()
    authored_report = _source_report(AUTHORED_CONTROL_TEXT, identity="unit:authored-atomic-control",
                                     document=document)
    authored_ledger = build_intent_requirement_ledger(AUTHORED_CONTROL_TEXT,
        source_report=authored_report, source_identity={"path": AUTHORED_CONTROL_PATH,
            "revision": _sha(AUTHORED_CONTROL_TEXT.encode()), "origin": "explicitly_authored_development_control"})
    requirement = authored_ledger["requirements"][0]
    statement = document["statements"][0]
    outputs = [{"path": "bottle.py", "effect": "modify", "media_type": "text/x-python"},
               {"path": "report.jsonl", "effect": "create", "media_type": "text/plain"}]
    grounding = {"requirement_id": requirement["requirement_id"], "outputs": outputs,
                 "validation_keys": [VALIDATION_KEY], "dependency_requirement_ids": []}
    operations = {"schema": "intent-symbolic-operation-contract@1",
        "ledger_sha256": authored_ledger["ledger_sha256"],
        "review_ref": "authored-development-control:terminal-bottle-operation@1",
        "interpretation_scope": "administrative_requirement_task_coverage",
        "operations": [{"operation_id": "operation:authored-repair-bottle", "task_key": TASK_KEY,
            "matchers": [{"requirement_id": requirement["requirement_id"],
                "native_document_sha256": requirement["native_document_sha256"],
                "statement_id": statement["statement_id"], "predicate": statement["predicate"],
                "arguments": statement["arguments"], "modality": statement["modality"]}],
            "outputs": outputs, "validation_keys": [VALIDATION_KEY], "dependency_operation_ids": []}],
        "semantic_alignment_verified": False, "proof_authority": False,
        "execution_authority": False, "completion_authority": False}
    authored_contract = build_intent_requirement_contract(source_path=AUTHORED_CONTROL_PATH,
        ledger=authored_ledger, requirements=[grounding], source_text=AUTHORED_CONTROL_TEXT,
        symbolic_operations=operations)
    value = {"schema": SCHEMA,
        "public_request": {"source_path": public_instruction_path, "text": text,
            "source_sha256": PUBLIC_INSTRUCTION_SHA256, "source_bytes": len(public_instruction_bytes),
            "status": "unresolved", "ledger": public_ledger, "contract": public_contract,
            "interpretation_origin": "explicit accounting only; no IntentIR interpretation supplied"},
        "authored_control": {"source_path": AUTHORED_CONTROL_PATH, "text": AUTHORED_CONTROL_TEXT,
            "source_sha256": _sha(AUTHORED_CONTROL_TEXT.encode()), "source_bytes": len(AUTHORED_CONTROL_TEXT.encode()),
            "native_document": document, "ledger": authored_ledger, "contract": authored_contract,
            "interpretation_origin": "independently authored atomic development control",
            "covers_complete_public_request": False, "semantic_alignment_to_public_request_verified": False},
        "current_behavioral_facts": [], "behavioral_satisfied_requirements": [],
        "public_request_fully_interpreted": False, "canonical_state_mutated": False,
        "provider_calls": 0, "training_steps": 0, "solver_calls": 0, "network_calls": 0,
        "official_reward": None, **_AUTHORITY}
    value["control_sha256"] = _sha(_wire(value))
    return value


__all__ = ["build_terminal_intent_control", "SCHEMA", "PUBLIC_INSTRUCTION_SHA256",
           "AUTHORED_CONTROL_TEXT", "AUTHORED_CONTROL_PATH", "VALIDATION_KEY", "TASK_KEY"]
