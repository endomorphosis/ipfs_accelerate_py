"""Bounded observations of the admitted Terminal-Bench symbolic route.

The caller must first run ``verify_local_benchmark_admission`` and pass the
original signed manifest envelope and its exact task specification. This projection neither verifies that
signature again nor upgrades an index, model nomination, or tool installation
to proof. It never reads source bodies or executes a prover.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity

SCHEMA = "terminal-symbolic-capabilities@1"
MAX_REPORT_BYTES = 16384
_GENERIC_WORKFLOWS = {
    "closed-local-keyword-rename@1": "closed_local_keyword_rename",
    "closed-imported-alias-call@1": "closed_imported_alias_call",
}
_DIGEST = re.compile(r"[0-9a-f]{64}")
_REASONS = frozenset({
    "unsupported_or_incomplete_source_inventory", "unsupported_output_effect_or_language",
    "unsupported_module_or_signature_shape", "ambiguous_supported_repairs",
    "no_supported_keyword_mismatch", "required_local_prover_unavailable",
    "unsupported_binding_scope", "doctor_analysis_secret_screen_refused",
    "native_doctor_planning_not_admitted", "native_doctor_gate_abstained",
    "native_doctor_transaction_not_admitted", "native_doctor_transaction_rejected",
    "ambiguous_header_candidates", "no_supported_header_candidate",
    "local_operator_does_not_cover_declared_outputs",
    "operator_proof_bounds_exceeded",
})
_SUPPORT = {
    ".supervisor-instruction.md": "instruction",
    ".supervisor-task-profile.json": "task_profile",
    ".supervisor-public-smoke.py": "structural_smoke",
}


def _require(condition):
    if not condition:
        raise ValueError("symbolic capability observation has inconsistent or unbounded inputs")


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _bounded_mapping(value, maximum):
    _require(isinstance(value, Mapping) and len(value) <= maximum)
    return value


def _identity(value):
    _require(type(value) is str and 1 <= len(value) <= 256 and value.isprintable())
    return value


def _path(value):
    _require(type(value) is str and 1 <= len(value) <= 1024 and value.isprintable()
             and not PurePosixPath(value).is_absolute()
             and str(PurePosixPath(value)) == value
             and not any(part in {".", "..", ".git"} for part in value.split("/")))
    return value


def _source_hashes(sources):
    _bounded_mapping(sources, 1024)
    hashes = {}
    for name, row in sources.items():
        _path(name)
        _bounded_mapping(row, 8)
        digest = row.get("sha256")
        _require(type(digest) is str and _DIGEST.fullmatch(digest) is not None)
        hashes[name] = digest
    return hashes


def _partition(result, *, hashes, task_cid, manifest_cid):
    partition = result.get("source_partition")
    if partition is None:
        return None
    keys = {"schema", "manifest_cid", "task_cid", "profile_sha256", "program_paths",
            "harness_support", "partition_cid", "execution_authority", "proof_authority",
            "completion_authority"}
    _require(type(partition) is dict and set(partition) == keys
             and partition["schema"] == "doctor-terminal-source-partition@1"
             and partition["task_cid"] == task_cid and partition["manifest_cid"] == manifest_cid
             and all(partition[key] is False for key in (
                 "execution_authority", "proof_authority", "completion_authority")))
    paths, support = partition["program_paths"], partition["harness_support"]
    _require(type(paths) is list and len(paths) <= 1024
             and type(support) is list and len(support) == len(_SUPPORT))
    _require(len(set(map(_path, paths))) == len(paths))
    seen = set()
    for row in support:
        _require(type(row) is dict and set(row) == {"path", "role", "sha256"})
        name = _path(row["path"])
        _require(name in _SUPPORT and name not in seen and row["role"] == _SUPPORT[name]
                 and name in hashes and row["sha256"] == hashes[name])
        seen.add(name)
    _require(not seen.intersection(paths) and seen | set(paths) == set(hashes)
             and partition["profile_sha256"] == hashes[".supervisor-task-profile.json"]
             and partition["partition_cid"] == content_identity({
                 key: value for key, value in partition.items() if key != "partition_cid"}))
    return {"program_input_count": len(paths), "harness_support_count": len(support),
            "partition_cid": partition["partition_cid"]}


def _prover(path):
    """Presence only: no subprocess, version request, download, or binary read."""
    if path is None:
        return "not_selected"
    _require(isinstance(path, (str, Path)) and len(str(path)) <= 4096)
    try:
        selected = Path(path)
        info = selected.stat()
        return "executable_present" if stat.S_ISREG(info.st_mode) and os.access(selected, os.X_OK) else "not_executable"
    except (OSError, ValueError):
        return "unavailable"


def _proof_observation(result, *, header):
    stages = _bounded_mapping(result.get("stages", {}), 32)
    proof = result.get("proof") if header else stages.get("proof")
    if proof is None:
        return {"native_stage_reported": False, "receipt_reported": False,
                "local_contract_proof_reported": False}
    _bounded_mapping(proof, 64)
    receipt = proof.get("proof_receipt_id" if header else "receipt_id")
    if receipt is not None:
        _identity(receipt)
    proved = (proof.get("status") == "proved_local_contract" if header else
              proof.get("status") == "executed" and proof.get("mutation_capable") is True)
    _require(not proved or receipt is not None)
    return {"native_stage_reported": True, "receipt_reported": receipt is not None,
            "local_contract_proof_reported": proved}


def assess_terminal_symbolic_capabilities(*, manifest: Mapping, task_cid: str,
        task_spec: Mapping, doctor_result: Mapping, prover_paths: Mapping,
        contract_profile: str | None = None) -> dict:
    """Project an already admitted task and its actual native Doctor result.

    This is observational, not an admission or proof API. Callers must not feed
    model-proposed manifests, task specifications, results, or tool paths.
    Unknown result reason text is counted, never copied into the report.
    """
    _bounded_mapping(manifest, 2)
    _require(set(manifest) == {"payload", "binding"})
    envelope = manifest
    _bounded_mapping(envelope["binding"], 3)
    _require(set(envelope["binding"]) == {"identity", "signature", "profile_id"}
             and all(type(value) is str and 1 <= len(value) <= 4096
                     for value in envelope["binding"].values()))
    manifest = _bounded_mapping(envelope["payload"], 32)
    _bounded_mapping(task_spec, 16)
    _bounded_mapping(doctor_result, 64)
    _require(manifest.get("schema") in {
        f"supervisor-local-benchmark-manifest@{version}" for version in range(1, 7)})
    task_cid = _identity(task_cid)
    tasks = manifest.get("tasks")
    _require(type(tasks) is list and 1 <= len(tasks) <= 64
             and sum(row == task_spec for row in tasks) == 1)
    hashes = _source_hashes(manifest.get("sources"))
    # Native planning receipts bind the complete signed envelope, including
    # signer and signature. Hashing only the verified payload loses that identity.
    manifest_cid = content_identity(dict(envelope))
    _require(contract_profile in {None, "wsgi-header-controls@1"}
             and doctor_result.get("task_cid") == task_cid)
    header = doctor_result.get("schema") == "supervisor-header-contract-workflow@1"
    _require(header or doctor_result.get("schema") == "supervisor-doctor-task-preparation@1")
    analysis = doctor_result.get("analysis", {}) if header else doctor_result
    _bounded_mapping(analysis, 64)
    _require(analysis.get("manifest_cid") == manifest_cid)
    if header:
        _require(contract_profile == "wsgi-header-controls@1")
    status = doctor_result.get("status")
    _require(status in {"residual", "prepared", "candidate_ready"})
    reasons = doctor_result.get("reason_codes", [])
    _require(type(reasons) in {list, tuple} and len(reasons) <= 128
             and all(type(code) is str and len(code) <= 256 for code in reasons))
    partition = _partition(doctor_result, hashes=hashes, task_cid=task_cid, manifest_cid=manifest_cid)
    observed_hashes = analysis.get("source_hashes")
    binding = "not_reported"
    if observed_hashes is not None:
        _bounded_mapping(observed_hashes, 1024)
        _require(all(name in hashes and digest == hashes[name] for name, digest in observed_hashes.items()))
        binding = "complete_hash_match" if dict(observed_hashes) == hashes else "scoped_hash_match"
    outputs = task_spec.get("outputs")
    validations = task_spec.get("validations")
    _require(type(outputs) is list and 1 <= len(outputs) <= 64
             and type(validations) is list and 1 <= len(validations) <= 64)
    effects = Counter()
    nonpython = 0
    for row in outputs:
        _bounded_mapping(row, 8)
        name = _path(row.get("path"))
        effect = row.get("effect")
        _require(effect in {"create", "modify", "write", "delete"})
        effects[effect] += 1
        nonpython += not name.endswith(".py")
    _require(type(prover_paths) is dict and set(prover_paths) == {"lean", "z3"})
    provers = {key: _prover(prover_paths[key]) for key in sorted(prover_paths)}
    proof = _proof_observation(doctor_result, header=header)
    operator = doctor_result.get("operator")
    selected_workflow = ("reviewed_header_guard" if header else
        "not_reported" if operator is None else
        _GENERIC_WORKFLOWS.get(operator, "unrecognized") if type(operator) is str else "unrecognized")
    named_structural = all(type(row) is dict and row.get("validation_key") in {
        "public-structural-smoke", "public-smoke"} for row in validations)
    gaps = []
    if partition is None:
        gaps.append("program_and_support_partition_not_observed")
    elif partition["program_input_count"] == 0:
        gaps.append("empty_program_context_required")
    if "unsupported_or_incomplete_source_inventory" in reasons:
        gaps.append("semantic_source_coverage_incomplete")
    if not header:
        gaps.append("task_behavior_contract_not_selected")
    if not header and (effects.get("create", 0) or nonpython or set(effects) != {"modify"}):
        gaps.append("generic_operator_output_coverage_missing")
    if len(tasks) == 1:
        gaps.append("single_declared_task_no_parallel_decomposition")
    if any(value != "executable_present" for value in provers.values()):
        gaps.append("local_prover_unavailable")
    if not proof["local_contract_proof_reported"]:
        gaps.append("local_contract_proof_not_reported")
    result = {
        "schema": SCHEMA, "task_cid": task_cid, "manifest_cid": manifest_cid,
        "manifest_payload_cid": content_identity(dict(manifest)),
        "task_spec_cid": content_identity(dict(task_spec)),
        "inventory": {"signed_input_count": len(hashes), "doctor_hash_binding": binding,
            "partition": partition, "analysis_status": (
                "available" if analysis.get("analysis_status", analysis.get("status")) == "available"
                else "unavailable" if analysis.get("analysis_status", analysis.get("status")) == "unavailable"
                else "not_reported"),
            "semantic_coverage": "incomplete" if "unsupported_or_incomplete_source_inventory" in reasons
                else "not_established_by_this_observation"},
        "contracts": {"requested_profile": contract_profile,
            "selected_profile": "wsgi-header-controls@1" if header else None,
            "named_structural_check": named_structural, "validation_semantics_verified": False,
            "validation_scope": "named_structural_check_not_verified" if named_structural else "not_classified",
            "validation_count": len(validations), "checks_executed_by_assessment": False,
            "task_behavior_contract": "reviewed_local_header_contract" if header
                else "not_selected", "whole_task_behavior_verified": False},
        "operators": {"selected_workflow": selected_workflow,
            "supported_generic_workflows": sorted(_GENERIC_WORKFLOWS.values()),
            "generic_operator_effects": ["modify"], "generic_operator_languages": ["python"],
            "declared_output_effect_counts": dict(sorted(effects.items())),
            "non_python_output_count": nonpython, "doctor_status": status,
            "candidate_ready_reported": status == "candidate_ready",
            "known_residual_reasons": sorted(set(reasons) & _REASONS),
            "other_reason_count": sum(code not in _REASONS for code in reasons)},
        "provers": {"presence": provers, "executed_by_assessment": False,
            "presence_establishes_proof": False},
        "proof": {**proof, "independently_reverified_by_assessment": False,
            "whole_program_verified": False, "scope": "reviewed_local_operator_only"},
        "planning": {"declared_task_count": len(tasks),
            "intent_requirement_artifact_declared": "intent_requirements" in manifest,
            "symbolic_selection_execution": "not_observed_here",
            "decomposition": "single_declared_task" if len(tasks) == 1 else "multiple_declared_tasks",
            "parallel_execution": "not_observed_here"},
        "gap_codes": sorted(gaps), "assessment_provider_calls": 0,
        "benchmark_solvability": "not_established", "proof_authority": False,
        "execution_authority": False, "publication_authority": False, "completion_authority": False,
    }
    result["observation_sha256"] = hashlib.sha256(_wire(result)).hexdigest()
    _require(len(_wire(result)) <= MAX_REPORT_BYTES)
    return result
