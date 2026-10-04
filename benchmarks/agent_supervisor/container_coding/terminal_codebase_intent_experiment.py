"""Join exact conditional-model lookup to a separately authored symbolic control.

The complete public request remains unresolved. This local experiment neither
fits a model nor promotes conditional model checks into software satisfaction.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

SCHEMA = "terminal-codebase-intent-experiment@1"
AUTHORITY = {"proof_authority": False, "execution_authority": False,
    "completion_authority": False, "source_semantics_verified": False,
    "semantic_alignment_verified": False, "asymptotic_optimizer_convergence_proved": False}


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_wire(value) + b"\n")


def _artifact(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    return {"path": str(path), "sha256": _sha(raw), "bytes": len(raw)}


def _progress(stage, **details):
    print(json.dumps({"stage": stage, **details}, sort_keys=True), flush=True)


def _decoder_capture(path):
    from .terminal_codebase_decoder_experiment import _captured_source
    root = Path(path).resolve(strict=True)
    result = json.loads((root / "result.json").read_bytes())
    manifest = json.loads((root / "codebase-ir-manifest.json").read_bytes())
    identifier = manifest.pop("manifest_id")
    if (result["status"] != "completed" or result["manifest_id"] != identifier
            or "sha256:" + _sha(_wire(manifest)) != identifier):
        raise ValueError("completed immutable decoder capture required")
    raw, parent = _captured_source(result["parent_capture"]["prepared_experiment"])
    if parent != result["parent_capture"]:
        raise ValueError("original public supervisor capture differs")
    for descriptor in result["execution_sources"]:
        if _artifact(descriptor["path"]) != descriptor:
            raise ValueError("qualified decoder implementation changed")
    instruction = Path(parent["source_snapshot"]["repository"]) / ".supervisor-instruction.md"
    binding = {"schema": "terminal-codebase-intent-parent@1", "decoder_experiment": str(root),
        "decoder_manifest_id": identifier, "decoder_manifest": _artifact(root / "codebase-ir-manifest.json"),
        "decoder_result": _artifact(root / "result.json"), "captured_public_source": parent["source"],
        "captured_public_instruction": _artifact(instruction),
        "role": "exact_frozen_decoder_and_public_source; no behavioral satisfaction inherited"}
    return result, raw, instruction.read_bytes(), binding


def prepare_authored_symbolic_capture(*, source_bytes, control, output):
    """Run native public preparation, then separately sign the authored control."""
    from . import terminal_indexed_preparation as prep
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptWorkflowRequest, PromptSource
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_directory_scanner import (
        RepositoryAllowlist, scan_prompt_directory_detailed,
    )
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import _select_evidence
    from ipfs_accelerate_py.agent_supervisor.planning.intent_symbolic_planning import build_intent_symbolic_plan
    from ipfs_accelerate_py.agent_supervisor.prompt.intent_plan_coverage import check_intent_plan_coverage

    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("fresh canonical authored control capture required")
    output.mkdir()
    repository = output / "repository"
    repository.mkdir()
    authored = control["authored_control"]
    (repository / "bottle.py").write_bytes(source_bytes)
    (repository / authored["source_path"]).write_text(authored["text"], encoding="utf-8")
    for arguments in (["init", "-q"], ["add", "--", "bottle.py", authored["source_path"]],
            ["-c", "user.name=Codebase intent qualification", "-c",
             "user.email=qualification@example.invalid", "commit", "-qm", "Exact public source and authored control"]):
        subprocess.run(["git", "-C", str(repository), *arguments], check=True,
                       capture_output=True, timeout=30)
    instruction = output / "original-public-instruction.md"
    instruction.write_text(control["public_request"]["text"], encoding="utf-8")
    prepared = prep.prepare(repository=repository, instruction=instruction,
        state=output / "preparation", disable_intent_autoencoder=True)
    payload = prepared["manifest"]["payload"]
    spec = deepcopy(prepared["spec"])
    spec["scope_paths"] = sorted({*spec["scope_paths"], authored["source_path"]})
    domains = local.local_planning_domain_declarations(repository=repository,
        profile_dir=Path(payload["profile_dir"]), lifecycle_dir=Path(payload["lifecycle_dir"]), task_specs=[spec])
    request = PromptWorkflowRequest.from_dict(prepared["request"])
    request = replace(request,
        prompt_source=PromptSource.inline(authored["text"], redacted_metadata={
            "summary": "Separately authored atomic development control; full public request remains unresolved."}),
        planning_policy=replace(request.planning_policy,
            policy_id="terminal-repository-proof-authored-symbolic-control", allow_model=False),
        scan_policy=replace(request.scan_policy, include_patterns=(
            "bottle.py", prep.INSTRUCTION, prep.SMOKE, authored["source_path"])),
        program_root=local._tree(payload["sources"]),
        intent_ir_root=local.content_identity(domains["intent"]),
        legal_ir_root=local.content_identity(domains["legal"]),
        security_ir_root=local.content_identity(domains["security"]))
    allowlist = RepositoryAllowlist.from_roots([repository])
    details = scan_prompt_directory_detailed(request, repository_allowlist=allowlist)
    request = replace(request, program_root=details.receipt.program_root)
    details = scan_prompt_directory_detailed(request, repository_allowlist=allowlist, previous=details)
    evidence = _select_evidence(request, details.receipt, prep._config(repository))
    spec["acceptance"][0]["evidence_cids"] = [evidence[0].evidence_cid]
    inputs = {"request": request.to_dict(), "scan": details.receipt.to_dict(),
        "domain_declarations": domains, "selected_evidence": [row.to_dict() for row in evidence]}
    roots = {"request_cid": request.request_cid, "scan_cid": details.receipt.scan_cid,
             "program_root": request.program_root}
    manifest = local.author_local_benchmark_manifest(repository=repository,
        profile_dir=Path(payload["profile_dir"]), lifecycle_dir=Path(payload["lifecycle_dir"]),
        task_specs=[spec], planning_roots=roots, planning_inputs=inputs,
        intent_requirements=authored["contract"])
    public_manifest = local.author_local_benchmark_manifest(repository=repository,
        profile_dir=Path(payload["profile_dir"]), lifecycle_dir=Path(payload["lifecycle_dir"]),
        task_specs=[spec], planning_roots=roots, planning_inputs=inputs,
        intent_requirements=control["public_request"]["contract"])
    local._manifest(manifest, initial=True)
    local._manifest(public_manifest, initial=True)
    proposed = build_intent_symbolic_plan(authored["contract"], manifest=manifest)
    coverage = check_intent_plan_coverage(control["public_request"]["contract"],
        graph=proposed["graph"], bindings=[])
    try:
        build_intent_symbolic_plan(control["public_request"]["contract"], manifest=public_manifest)
    except ValueError as exc:
        refusal = {"schema": "terminal-public-intent-planning-refusal@1", "status": "refused",
            "reason": str(exc), "full_source_accounting": True, "native_requirement_count": 0,
            "unsupported_source_unit_count": sum(row["disposition"] == "unsupported"
                for row in control["public_request"]["ledger"]["source_units"]),
            "coverage": coverage, "provider_fallback": False, "public_request_planned": False, **AUTHORITY}
    else:
        raise ValueError("unresolved public request unexpectedly produced a symbolic plan")
    if coverage["accepted"]:
        raise ValueError("authored atomic plan cannot cover the unresolved complete public request")
    _write(output / "authored-control-manifest.json", manifest)
    _write(output / "public-unresolved-manifest.json", public_manifest)
    _write(output / "public-planning-refusal.json", refusal)
    _write(output / "authored-control-workflow-request.json", request.to_dict())
    return {"repository": str(repository), "manifest": manifest, "public_manifest": public_manifest,
        "workflow_request": request.to_dict(), "public_refusal": refusal,
        "source_snapshot": local._manifest(manifest, initial=True)[2],
        "original_public_preparation_manifest": prepared["manifest"],
        "scope": "separate_authored_development_control; full_public_request_retained_unresolved"}


def run_intent_experiment(*, decoder_experiment, output):
    from .terminal_codebase_intent_control import build_terminal_intent_control
    from .terminal_codebase_proof_index import (
        persist_terminal_codebase_proof_index, validate_terminal_codebase_proof_index,
        build_terminal_codebase_proof_key_relationship, lookup_terminal_codebase_model_evidence,
    )
    from .terminal_codebase_planning_snapshot import (
        build_repository_proof_planning_snapshot, replay_repository_proof_planning_snapshot,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.intent_codebase_matching import (
        match_intent_codebase, reviewed_header_matching_domain,
    )
    from .terminal_codebase_supervisor_fixture import (
        bound_terminal_codebase_metadata_records, reconstruct_terminal_codebase_metadata_records,
    )
    from .codebase_ir_metadata import hydrate_codebase_ir_metadata, validate_codebase_ir_metadata

    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("fresh canonical intent experiment namespace required")
    parent, source, instruction, parent_binding = _decoder_capture(decoder_experiment)
    output.mkdir(parents=True)
    started = time.monotonic()
    modules = (sys.modules[build_terminal_intent_control.__module__],
        sys.modules[persist_terminal_codebase_proof_index.__module__],
        sys.modules[match_intent_codebase.__module__], sys.modules[build_repository_proof_planning_snapshot.__module__])
    execution_sources = [_artifact(path) for path in sorted({Path(__file__).resolve(),
        *(Path(module.__file__).resolve() for module in modules)})]
    _write(output / "experiment-policy.json", {"schema": SCHEMA, "parent": parent_binding,
        "execution_sources": execution_sources, "provider_calls": 0, "training_steps": 0,
        "proof_index_profile": "seven_exact_conditional_header_model_entries",
        "public_request_policy": "complete_unresolved_ledger; refuse_unsupported_planning",
        "planning_control": "separate_explicitly_authored_atomic_intent; zero_behavioral_facts",
        "current_behavioral_facts": [], "source_execution": False, "worker_launch": False,
        "existing_admission_profile_extended": False, **AUTHORITY})
    control = build_terminal_intent_control(public_instruction_bytes=instruction)
    _write(output / "intent-control.json", control)
    phases = {}
    _progress("exact_model_evidence_production_start")
    phase_started = time.monotonic()
    index = persist_terminal_codebase_proof_index(source_bytes=source, source_path="bottle.py",
        decoder_experiment=parent, qualification=parent["logic"], output=output / "proof-index")
    replay = validate_terminal_codebase_proof_index(output=output / "proof-index", expected=index,
        source_bytes=source, source_path="bottle.py", decoder_experiment=parent,
        qualification=parent["logic"], fresh_process=True)
    phases["proof_index_native_production_and_fresh_replay"] = time.monotonic() - phase_started
    _write(output / "proof-index-result.json", index)
    _write(output / "proof-index-replay.json", replay)
    _progress("exact_model_evidence_complete", entries=len(index["entries"]))
    phase_started = time.monotonic()
    capture = prepare_authored_symbolic_capture(source_bytes=source, control=control, output=output / "supervisor")
    phases["native_supervisor_preparation_and_authored_control"] = time.monotonic() - phase_started
    _write(output / "control-capture.json", capture)
    authored = control["authored_control"]
    native = authored["native_document"]
    source_ref = native["sources"][0]
    source_identity = {key: source_ref[key] for key in (
        "ref_id", "source_uri", "source_id", "source_revision", "content_sha256")}
    statement = native["statements"][0]
    query = {"schema": "intent-codebase-query@1", "review_ref": "authored-header-focus-development-control@1",
        "statement": {key: statement[key] for key in ("statement_id", "predicate", "arguments")},
        "source_path": "bottle.py", "symbols": ["_hkey", "_hval"],
        "property": "header_delimiter_rejection", "polarity": "positive",
        "domain": reviewed_header_matching_domain(), "semantic_alignment_verified": False}
    matching_inputs = {"intent_document": native, "source_text": authored["text"],
        "source_identity": source_identity, "current_source_snapshot": index["source_snapshot"]}
    absent_dimensions = deepcopy(index["entries"][0]["key_relationship"]["dimensions"])
    absent_dimensions["policy"] = {**absent_dimensions["policy"],
        "exact_lookup_control": "different_policy_not_in_this_index"}
    absent_key = build_terminal_codebase_proof_key_relationship(dimensions=absent_dimensions)
    absent_lookup = lookup_terminal_codebase_model_evidence(output=output / "proof-index",
        expected=index, expected_key=absent_key, expected_environment=index["environment"])
    if absent_lookup["status"] != "miss" or absent_lookup["evidence"] is not None:
        raise ValueError("different complete proof key unexpectedly reused model evidence")
    _write(output / "exact-key-miss.json", absent_lookup)
    # Match only the complete exact lookup envelopes, not status-only rows.
    matches = {"conditional_model_context": match_intent_codebase(**matching_inputs,
        query=query, evidence_rows=replay["entries"]),
        "evidence_off": match_intent_codebase(**matching_inputs, query=query, evidence_rows=[])}
    matches["exact_key_absence"] = match_intent_codebase(**matching_inputs,
        query=query, evidence_rows=[absent_lookup])
    prohibited = deepcopy(query)
    prohibited["polarity"] = "prohibition"
    matches["prohibition_without_evidence"] = match_intent_codebase(**matching_inputs,
        query=prohibited, evidence_rows=[])
    _write(output / "intent-codebase-matches.json", matches)
    phase_started = time.monotonic()
    snapshot_inputs = {"manifest": capture["manifest"], "workflow_request": capture["workflow_request"],
        "control": control, "proof_index_manifest": index,
        "match_result": matches["conditional_model_context"]}
    planned = build_repository_proof_planning_snapshot(**snapshot_inputs)
    replay_repository_proof_planning_snapshot(expected=planned["snapshot"], **snapshot_inputs)
    phases["native_symbolic_planning_and_snapshot_replay"] = time.monotonic() - phase_started
    graph = planned["symbolic_plan"]["graph"].to_dict()
    _write(output / "repository-planning-snapshot.json", planned["snapshot"])
    _write(output / "symbolic-graph.json", graph)
    _write(output / "symbolic-receipt.json", planned["symbolic_plan"]["receipt"])
    _progress("symbolic_control_complete", observed_behavioral_facts=0, public_request_planned=False)
    records = {"proof_index_manifest": [index], "proof_index_entries": index["entries"],
        "proof_index_replay": [replay], "proof_index_negative_lookup": [absent_lookup], "intent_control": [control],
        "intent_codebase_matches": list(matches.values()),
        "repository_planning_snapshot": [planned["snapshot"]],
        "repository_symbolic_graph": [graph],
        "repository_symbolic_receipt": [planned["symbolic_plan"]["receipt"]],
        "repository_symbolic_bindings": [{"bindings": planned["symbolic_plan"]["requirement_bindings"]}],
        "public_planning_refusal": [capture["public_refusal"]],
        "authored_control_signed_manifest": [capture["manifest"]],
        "public_unresolved_signed_manifest": [capture["public_manifest"]],
        "current_source_correspondence": [{"sources": capture["source_snapshot"],
            "current_root": planned["snapshot"]["current_root_id"],
            "proof_source_snapshot": index["source_snapshot"],
            "correspondence": "exact Bottle file leaf; distinct complete source forests"}],
        "parent_decoder_capture": [parent_binding], "execution_sources": execution_sources,
        "ast": [], "kg": [], "vectors": [], "contracts": []}
    records = json.loads(_wire(records))
    complete_sha = _sha(_wire(records))
    bounded = bound_terminal_codebase_metadata_records(records)
    if _wire(reconstruct_terminal_codebase_metadata_records(bounded)) != _wire(records):
        raise ValueError("bounded planning metadata changed complete producer records")
    source_snapshot = {"schema": "terminal-codebase-intent-source-snapshot@1",
        "parent_decoder_manifest_id": parent_binding["decoder_manifest_id"],
        "public_source_sha256": _sha(source), "public_instruction_sha256": _sha(instruction),
        "proof_index_manifest_id": index["manifest_id"],
        "repository_planning_snapshot_id": planned["snapshot"]["snapshot_id"],
        "complete_metadata_sha256": complete_sha, "public_request_planned": False,
        "authored_control_only": True, "current_behavioral_facts": []}
    phase_started = time.monotonic()
    hydration = hydrate_codebase_ir_metadata(records=bounded, output=output / "metadata", source_snapshot=source_snapshot)
    metadata_replay = validate_codebase_ir_metadata(output=output / "metadata", expected=hydration, fresh_process=True)
    restored = {family: [json.loads(line)["payload"] for line in
        (output / "metadata" / descriptor["relative_path"]).read_text().splitlines()]
        for family, descriptor in hydration["exports"].items()}
    recovered = reconstruct_terminal_codebase_metadata_records(restored)
    if _sha(_wire(recovered)) != complete_sha:
        raise ValueError("native complete planning metadata reconstruction differs")
    phases["native_metadata_hydration_and_fresh_replay"] = time.monotonic() - phase_started
    reconstruction = {"schema": "terminal-codebase-intent-metadata-reconstruction@1", "status": "exact",
        "original_records_sha256": complete_sha, "recovered_records_sha256": _sha(_wire(recovered)),
        "zero_truncation": True, "original_family_counts": {family: len(rows) for family, rows in records.items()},
        "bounded_family_counts": hydration["family_counts"]}
    _write(output / "metadata-result.json", hydration)
    _write(output / "metadata-replay.json", metadata_replay)
    _write(output / "metadata-reconstruction.json", reconstruction)
    if (_decoder_capture(decoder_experiment)[1:3] != (source, instruction)
            or any(_artifact(row["path"]) != row for row in execution_sources)):
        raise ValueError("captured source or experiment implementations changed during execution")
    replay_repository_proof_planning_snapshot(expected=planned["snapshot"], **snapshot_inputs)
    result = {"schema": SCHEMA, "status": "completed", "output": str(output),
        "status_meaning": "bounded experimental model-evidence/control join; public planning remains unresolved",
        "parent": parent_binding, "source_snapshot": source_snapshot,
        "proof_index": index, "proof_index_fresh_process_replay": replay,
        "intent_control": control, "intent_codebase_matches": matches,
        "public_planning_refusal": capture["public_refusal"],
        "repository_planning_snapshot": planned["snapshot"],
        "symbolic_receipt": planned["symbolic_plan"]["receipt"],
        "metadata": hydration, "metadata_reconstruction": reconstruction,
        "qualification_outcomes": {"exact_conditional_model_proof_key_lookup": "passed",
            "native_current_source_and_frozen_model_replay": "passed",
            "unresolved_public_instruction_refusal": "passed",
            "authored_control_native_symbolic_selection_and_material_replay": "passed",
            "native_duckdb_ducklake_complete_restart": "passed",
            "public_instruction_to_native_intent_semantics": "unqualified",
            "behavioral_requirement_satisfaction": "unproved",
            "repository_evidence_admission_or_execution": "not_exercised"},
        "phase_seconds": phases, "elapsed_seconds": time.monotonic() - started,
        "execution_sources": execution_sources, "provider_calls": 0, "training_steps": 0,
        "official_verifier_executed": False, "official_reward": None,
        "current_behavioral_facts": [], "public_request_planned": False,
        "repository_evidence_admitted": False, **AUTHORITY}
    _write(output / "result.json", result)
    _progress("intent_codebase_experiment_complete", result=str(output / "result.json"), seconds=result["elapsed_seconds"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        run_intent_experiment(decoder_experiment=args.decoder_experiment, output=args.output)
    except Exception as exc:
        _progress("intent_codebase_experiment_failed", error_type=type(exc).__name__, message=str(exc))
        if not existed and (args.output / "experiment-policy.json").is_file():
            _write(args.output / "failure.json", {"schema": SCHEMA, "status": "failed",
                "error_type": type(exc).__name__, "message": str(exc), "successful_completion_claimed": False})
        raise


if __name__ == "__main__":
    main()
