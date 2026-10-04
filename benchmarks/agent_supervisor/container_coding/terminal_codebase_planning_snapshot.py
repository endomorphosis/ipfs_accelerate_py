"""Freeze conditional repository evidence beside the native administrative plan.

This is experimental planning context. No repository evidence is converted to
a satisfied behavioral predicate, and the existing admission profile is not
extended to accept repository proof facts by this adapter.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

SCHEMA = "repository-proof-planning-snapshot@1"
AUTHORITY = {"proof_authority": False, "execution_authority": False,
             "completion_authority": False, "source_semantics_verified": False,
             "semantic_alignment_verified": False}


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _digest(value):
    return "sha256:" + hashlib.sha256(_wire(value)).hexdigest()


def build_repository_proof_planning_snapshot(*, manifest, workflow_request,
        control, proof_index_manifest, match_result, code_source_path="bottle.py"):
    """Verify current signed leaves and freeze full proposal-only material values."""
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.planning.intent_requirement_adapter import build_intent_planning_materials
    from ipfs_accelerate_py.agent_supervisor.planning.intent_symbolic_planning import build_intent_symbolic_plan
    from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
        PlanCreateMaterials, freeze_plan_create_input_snapshot, plan_create_request_from_workflow,
    )
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptWorkflowRequest
    from .terminal_codebase_intent_control import build_terminal_intent_control
    from .terminal_codebase_proof_index import validate_terminal_codebase_model_evidence_lookup
    from ipfs_accelerate_py.agent_supervisor.planning.intent_codebase_matching import match_intent_codebase

    # A canonical copy prevents mutable aliases from changing the frozen inputs.
    control, proof_index_manifest, match_result = (
        json.loads(_wire(value)) for value in (control, proof_index_manifest, match_result))
    expected_control = build_terminal_intent_control(
        public_instruction_bytes=control["public_request"]["text"].encode("utf-8"),
        public_instruction_path=control["public_request"]["source_path"])
    if control != expected_control:
        raise ValueError("intent control differs from native source reconstruction")
    if any(match_result.get(key) is not False for key in AUTHORITY):
        raise ValueError("repository match grants no behavioral or execution authority")
    if any(proof_index_manifest.get(key) is not False for key in (
            "proof_authority", "execution_authority", "completion_authority", "source_semantics_verified")):
        raise ValueError("conditional proof index grants no behavioral or execution authority")
    if proof_index_manifest.get("semantic_alignment_verified", False) is not False:
        raise ValueError("conditional proof index cannot claim prompt alignment")
    index_body = dict(proof_index_manifest)
    index_id = index_body.pop("manifest_id", None)
    index_path = Path(proof_index_manifest["output"]) / "manifest.json"
    if (index_id != _digest(index_body) or not index_path.is_absolute() or not index_path.is_file()
            or index_path.is_symlink() or index_path.resolve(strict=True) != index_path.absolute()):
        raise ValueError("canonical immutable proof-index manifest required")
    before = index_path.stat()
    if before.st_size > 32 * 1024 * 1024:
        raise ValueError("bounded complete proof-index manifest required")
    index_raw = index_path.read_bytes()
    after = index_path.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or len(index_raw) > 32 * 1024 * 1024 or json.loads(index_raw) != proof_index_manifest):
        raise ValueError("proof-index artifact differs from the supplied complete body")
    index_binding = {"schema": "repository-proof-index-artifact-reference@1",
        "manifest_id": index_id, "path": str(index_path), "bytes": len(index_raw),
        "sha256": hashlib.sha256(index_raw).hexdigest(),
        "canonical_body_sha256": _digest(proof_index_manifest),
        "source_snapshot": proof_index_manifest["source_snapshot"],
        "scope": "complete_external_body_retained_and_byte_checked; conditional_models_only"}
    if match_result.get("behavioral_satisfied_requirements") or match_result.get("current_behavioral_facts"):
        raise ValueError("conditional repository models cannot supply behavioral facts")
    declared, _, observed = local._manifest(manifest, initial=True)
    authored = control["authored_control"]
    contract = authored["contract"]
    if (local.decode_intent_requirement_contract(declared) != contract
            or observed[authored["source_path"]]["sha256"] != authored["source_sha256"]
            or observed[control["public_request"]["source_path"]]["sha256"]
                != control["public_request"]["source_sha256"]):
        raise ValueError("authored control or complete public request differs from signed sources")
    if type(code_source_path) is not str or code_source_path not in observed:
        raise ValueError("current code source must be a signed exact source leaf")
    code_leaf = {"source_path": code_source_path, "source_sha256": observed[code_source_path]["sha256"]}
    indexed_source = proof_index_manifest.get("source_snapshot", {})
    if (any(indexed_source.get(key) != value for key, value in code_leaf.items())
            or match_result.get("current_source_snapshot") != indexed_source):
        raise ValueError("proof index and intent match differ from the current signed code leaf")
    entries = {entry["entry_id"]: entry for entry in proof_index_manifest.get("entries", [])}
    if len(entries) != len(proof_index_manifest.get("entries", [])):
        raise ValueError("ambiguous complete proof-index entries")
    for lookup in match_result.get("evidence_rows", []):
        if lookup.get("status") == "hit":
            expected_entry = entries.get(lookup.get("entry_id"))
            if expected_entry is None:
                raise ValueError("matched evidence is outside the complete pinned proof index")
            validate_terminal_codebase_model_evidence_lookup(lookup=lookup,
                expected_entry=expected_entry, expected_environment=proof_index_manifest["environment"])
    native = authored["native_document"]
    source_ref = native["sources"][0]
    source_identity = {key: source_ref[key] for key in (
        "ref_id", "source_uri", "source_id", "source_revision", "content_sha256")}
    rebuilt_match = match_intent_codebase(intent_document=native, source_text=authored["text"],
        source_identity=source_identity, query=match_result["query"],
        evidence_rows=match_result["evidence_rows"], current_source_snapshot=indexed_source)
    if rebuilt_match != match_result:
        raise ValueError("repository match differs from exact native intent/evidence/residual reconstruction")
    materials = build_intent_planning_materials(contract, manifest=manifest)
    if materials.current_facts:
        raise ValueError("administrative control must retain zero current behavioral facts")
    proposed = build_intent_symbolic_plan(contract, manifest=manifest)
    if (proposed["receipt"]["observed_facts_supplied"] != 0
            or not proposed["coverage"]["accepted"]
            or sorted(task.task_key for task in proposed["graph"].tasks)
                != sorted(item["task_key"] for item in declared["tasks"])):
        raise ValueError("native symbolic proposal changed the complete declared task population")
    request = (PromptWorkflowRequest.from_dict(workflow_request)
               if type(workflow_request) is dict else workflow_request)
    if (request.to_dict() != declared.get("planning_inputs", {}).get("request")
            or request.program_root != declared["planning_roots"]["program_root"]
            or request.planning_policy.allow_model is not False):
        raise ValueError("frozen workflow request differs from signed native planning inputs")
    model_off = {"schema": "repository-symbolic-model-off@1", "allow_model": False,
        "max_model_candidates": 0, "provider_calls": 0,
        "route": "existing_intent_symbolic_planning",
        "semantic_decoder_mode": "frozen_inference_only; no_fit_in_planning"}
    extra = {"schema": SCHEMA, "intent_control": control,
        "conditional_proof_index": index_binding, "intent_codebase_match": match_result,
        "current_code_leaf": code_leaf, "model_off": model_off,
        "operation_contract": _plain(materials.operation_contract),
        "current_behavioral_facts": [], "behavioral_satisfied_requirements": [], **AUTHORITY}
    bound = PlanCreateMaterials(intent=materials.intent, producers=materials.producers,
        task_candidates=materials.task_candidates, predicates=materials.predicates,
        current_facts=(), frozen_goal=materials.frozen_goal,
        candidate_context=materials.candidate_context, extra=extra)
    create_request = plan_create_request_from_workflow(request,
        dirty_worktree_root=materials.intent.current_root_id,
        scope_paths=tuple(sorted({path for spec in declared["tasks"] for path in spec["scope_paths"]})))
    if create_request.budget.max_model_calls != 0:
        raise ValueError("native repository planning budget must disable model calls")
    native_snapshot = freeze_plan_create_input_snapshot(create_request, materials=bound)
    if not native_snapshot.material_binding.get("reuse_supported", False):
        raise ValueError("complete repository planning materials are not exactly reusable")
    snapshot = {"schema": SCHEMA, "scope": "authored_administrative_control_with_conditional_model_context",
        "signed_manifest_sha256": _digest(_plain(manifest)),
        "current_root_id": materials.intent.current_root_id,
        "current_source_inventory": observed, "current_code_leaf": code_leaf,
        "intent_control_sha256": _digest(control), "proof_index_manifest_sha256": _digest(proof_index_manifest),
        "match_result_sha256": _digest(match_result), "model_off_identity": _digest(model_off),
        "full_materials_sha256": _digest(extra), "full_materials": extra,
        "native_input_snapshot": native_snapshot.to_dict(),
        "native_symbolic_receipt": proposed["receipt"],
        "native_graph_cid": proposed["graph"].content_id,
        "source_correspondence_scope": "exact_common_code_leaf; distinct_source_forests_are_not_aliased",
        "current_behavioral_facts": [], "behavioral_satisfied_requirements": [],
        "complete_declared_task_population_preserved": True,
        "public_request_fully_interpreted": False, "public_request_planned": False,
        "repository_evidence_admitted": False, "worker_launched": False,
        "provider_calls": 0, "training_steps": 0, **AUTHORITY}
    snapshot["snapshot_id"] = _digest(snapshot)
    return {"snapshot": snapshot, "symbolic_plan": proposed}


def replay_repository_proof_planning_snapshot(*, expected, **inputs):
    observed = build_repository_proof_planning_snapshot(**inputs)
    if _wire(observed["snapshot"]) != _wire(expected):
        raise ValueError("repository planning snapshot differs from exact current material replay")
    return observed
