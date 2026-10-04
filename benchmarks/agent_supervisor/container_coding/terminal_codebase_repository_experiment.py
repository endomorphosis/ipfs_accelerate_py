"""Qualify actual current-source metadata, native retrieval and edit invalidation.

Reuses the completed conditional-evidence experiment without modifying it. No
model is fitted, no source is executed, and no behavioral fact is admitted.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .terminal_codebase_planning_snapshot import AUTHORITY
from .terminal_codebase_supervisor_fixture import _read

SCHEMA = "terminal-codebase-repository-experiment@1"


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _artifact(path):
    path = Path(path).absolute()
    raw = _read(path, 32 * 1024 * 1024)
    return {"path": str(path), "bytes": len(raw), "sha256": _sha(raw)}


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_wire(value) + b"\n")
    return _artifact(path)


def _progress(stage, **details):
    print(json.dumps({"stage": stage, **details}, sort_keys=True), flush=True)


def load_completed_intent_experiment(path):
    """Check complete stored producer exports, implementation and source pins."""
    from .terminal_codebase_intent_experiment import _decoder_capture
    from .terminal_codebase_indexed_planning import current_signed_sources
    from .codebase_ir_metadata import validate_codebase_ir_metadata
    from .terminal_codebase_supervisor_fixture import reconstruct_terminal_codebase_metadata_records

    root = Path(path).resolve(strict=True)
    result = json.loads(_read(root / "result.json", 32 * 1024 * 1024))
    if result["schema"] != "terminal-codebase-intent-experiment@1" or result["status"] != "completed":
        raise ValueError("completed conditional intent experiment required")
    if (root / "failure.json").exists():
        raise ValueError("failed intent experiment is not an eligible parent")
    for row in result["execution_sources"]:
        if _artifact(row["path"]) != row:
            raise ValueError("qualified parent implementation changed")
    decoder, source, instruction, binding = _decoder_capture(result["parent"]["decoder_experiment"])
    if binding != result["parent"]:
        raise ValueError("complete parent decoder binding differs")
    capture = json.loads(_read(root / "control-capture.json", 32 * 1024 * 1024))
    declared, observed, sources = current_signed_sources(capture["manifest"])
    if (observed != capture["source_snapshot"] or sources["bottle.py"] != source
            or sources[result["intent_control"]["public_request"]["source_path"]] != instruction):
        raise ValueError("parent source forest differs from original public capture")
    report = validate_codebase_ir_metadata(output=root / "metadata", expected=result["metadata"])
    bounded = {family: [json.loads(line)["payload"] for line in
        _read(root / "metadata" / descriptor["relative_path"], 32 * 1024 * 1024).splitlines()]
        for family, descriptor in report["exports"].items()}
    records = reconstruct_terminal_codebase_metadata_records(bounded)
    if _sha(_wire(records)) != result["metadata_reconstruction"]["original_records_sha256"]:
        raise ValueError("complete parent metadata producer root differs")
    return {"root": root, "result": result, "capture": capture,
        "declared": declared, "sources": sources, "decoder": decoder,
        "records": records, "binding": {"schema": "terminal-codebase-repository-parent@1",
            "intent_result": _artifact(root / "result.json"),
            "intent_metadata_manifest": _artifact(root / "metadata/manifest.json"),
            "intent_planning_snapshot": _artifact(root / "repository-planning-snapshot.json"),
            "intent_signed_capture": _artifact(root / "control-capture.json"),
            "intent_producer_sha256": _sha(_wire(records)),
            "scope": "exact_prior_conditional_context; no_behavioral_fact_inherited"}}


def _rejection(function):
    try:
        function()
    except ValueError as exc:
        return {"status": "refused", "reason": str(exc), "exception": type(exc).__name__}
    raise ValueError("stale source generation unexpectedly remained eligible")


def _edit_control(*, output, parent, index, query):
    from .terminal_codebase_repository_index import (
        build_current_repository_metadata, persist_repository_index, validate_repository_index, query_repository_index,
    )
    from .terminal_codebase_catalog_capture import capture_frozen_codebase_learning

    output.mkdir()
    repository = output / "repository"
    repository.mkdir()
    for path, raw in parent["sources"].items():
        target = repository / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    def git(*args):
        return subprocess.run(["git", "-C", str(repository), *args], check=True,
                              capture_output=True, text=True, timeout=30).stdout.strip()
    git("init", "-q")
    git("add", "--", *sorted(parent["sources"]))
    git("-c", "user.name=Repository source generation qualification", "-c",
        "user.email=qualification@example.invalid", "commit", "-qm", "Exact four-source copy")
    head = git("rev-parse", "HEAD")
    bottle = repository / "bottle.py"
    bottle.write_bytes(bottle.read_bytes() + b"\n# Isolated dirty-overlay qualification; original source unchanged.\n")
    changed = {path: _read(repository / path, 1_048_576) for path in parent["sources"]}
    if git("rev-parse", "HEAD") != head or not git("diff", "--name-only"):
        raise ValueError("dirty-overlay control must change source at unchanged HEAD")
    inputs = {"source_bytes": changed, "repository_id": parent["declared"]["repository_cid"],
        "task_spec": parent["declared"]["tasks"][0]}
    stale_index = _rejection(lambda: validate_repository_index(**inputs,
        proof_index_manifest=parent["result"]["proof_index"], expected=index, output=Path(index["output"])))
    stale_learning = _rejection(lambda: capture_frozen_codebase_learning(
        intent_experiment=parent["root"], current_source_bytes=changed))
    dirty = build_current_repository_metadata(**inputs)
    stale_proof = _rejection(lambda: persist_repository_index(current=dirty,
        proof_index_manifest=parent["result"]["proof_index"], output=output / "stale-proof-must-refuse"))
    # An edited source gets a new generation and zero inherited proof/learning.
    changed_index = persist_repository_index(current=dirty, output=output / "changed-index")
    changed_replay = validate_repository_index(**inputs, expected=changed_index,
        output=output / "changed-index", fresh_process=True)
    changed_query = query_repository_index(**inputs, expected=changed_index,
        output=output / "changed-index", query=query)
    cold = build_current_repository_metadata(**inputs)
    if _wire(dirty) != _wire(cold):
        raise ValueError("edited incremental source generation differs from independent cold rebuild")
    cold_index = persist_repository_index(current=cold, output=output / "cold-index")
    cold_query = query_repository_index(**inputs, expected=cold_index, output=output / "cold-index", query=query)
    # Only storage-local identities can differ; complete query content must agree.
    local_keys = {"manifest_id", "index_manifest_id", "query_id", "output"}
    semantic = lambda value: {key: item for key, item in value.items() if key not in local_keys}
    if _wire(semantic(changed_query)) != _wire(semantic(cold_query)):
        raise ValueError("edited query differs from independent cold native rebuild")
    _write(output / "changed-generation.json", dirty["source_snapshot"])
    _write(output / "changed-query.json", changed_query)
    _write(output / "cold-query.json", cold_query)
    return {"schema": "terminal-codebase-repository-edit-control@1",
        "scope": "private_copied_source; unsigned_successor_context_only",
        "repository": str(repository), "head_before": head, "head_after": git("rev-parse", "HEAD"),
        "dirty_paths": git("diff", "--name-only").splitlines(),
        "changed_sources": {path: {"sha256": _sha(raw), "bytes": len(raw)} for path, raw in changed.items()},
        "stale_repository_index": stale_index, "stale_frozen_learning": stale_learning,
        "stale_conditional_proof": stale_proof, "changed_index": changed_index,
        "changed_index_fresh_process_replay": changed_replay,
        "changed_query": changed_query, "cold_query": cold_query,
        "complete_source_producer_cold_equivalence": True, "semantic_query_cold_equivalence": True,
        "invalidated_entire_prior_generation": True, "active_inherited_proof_entries": 0,
        "active_inherited_learned_rows": 0, "successor_training": "deferred; no_fit",
        "selective_dependency_reproof_qualified": False, "signed_successor_admission": False,
        "current_behavioral_facts": [], "training_steps": 0, **AUTHORITY}


def run_repository_experiment(*, intent_experiment, output):
    from .terminal_codebase_repository_index import (
        build_current_repository_metadata, persist_repository_index, validate_repository_index, query_repository_index,
    )
    from .terminal_codebase_catalog_capture import capture_frozen_codebase_learning
    from .terminal_codebase_proof_index import validate_terminal_codebase_proof_index
    from .terminal_codebase_indexed_planning import (
        build_indexed_repository_planning_snapshot, replay_indexed_repository_planning_snapshot,
    )
    from .terminal_codebase_supervisor_fixture import (
        bound_terminal_codebase_metadata_records, reconstruct_terminal_codebase_metadata_records,
    )
    from .codebase_ir_metadata import hydrate_codebase_ir_metadata, validate_codebase_ir_metadata

    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("fresh canonical repository qualification namespace required")
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    modules = (sys.modules[build_current_repository_metadata.__module__],
        sys.modules[capture_frozen_codebase_learning.__module__],
        sys.modules[build_indexed_repository_planning_snapshot.__module__])
    execution_sources = [_artifact(path) for path in sorted({Path(__file__).resolve(),
        *(Path(module.__file__).resolve() for module in modules)})]
    parent = load_completed_intent_experiment(intent_experiment)
    parent_seconds = time.monotonic() - started
    if any(_artifact(row["path"]) != row for row in execution_sources):
        raise ValueError("repository implementations changed during parent preflight")
    output.mkdir(parents=True)
    _write(output / "experiment-policy.json", {"schema": SCHEMA, "parent": parent["binding"], "started_at": started_at,
        "execution_sources": execution_sources, "source_scope": "four_exact_signed_sources",
        "index_scope": "native_AST_KG_lexical_vectors_and_declared_contracts",
        "learning_scope": "exact_prior_checkpoint_and_source_replay; no_fit",
        "edit_control_scope": "private_copy; conservative_generation_invalidation",
        "provider_calls": 0, "training_steps": 0, "source_execution": False,
        "worker_launched": False, "repository_evidence_admitted": False, **AUTHORITY})
    phases = {"complete_parent_capture_and_metadata_replay": parent_seconds}
    phase = time.monotonic()
    _progress("current_source_and_frozen_packages_start")
    proof = parent["result"]["proof_index"]
    proof_replay = validate_terminal_codebase_proof_index(output=Path(proof["output"]), expected=proof,
        source_bytes=parent["sources"]["bottle.py"], source_path="bottle.py",
        decoder_experiment=parent["decoder"], qualification=parent["decoder"]["logic"], fresh_process=True)
    learning = capture_frozen_codebase_learning(intent_experiment=parent["root"],
        current_source_bytes=parent["sources"], fresh_process=True)
    learning_artifact = _write(output / "frozen-learning.json", learning)
    _write(output / "proof-index-replay.json", proof_replay)
    phases["current_source_frozen_inference_and_package_replay"] = time.monotonic() - phase
    inputs = {"source_bytes": parent["sources"], "repository_id": parent["declared"]["repository_cid"],
        "task_spec": parent["declared"]["tasks"][0], "proof_index_manifest": proof}
    phase = time.monotonic()
    current = build_current_repository_metadata(**{key: value for key, value in inputs.items()
                                                 if key != "proof_index_manifest"})
    index = persist_repository_index(current=current, proof_index_manifest=proof, output=output / "repository-index")
    replay = validate_repository_index(**inputs, expected=index, output=output / "repository-index", fresh_process=True)
    focus = parent["result"]["intent_codebase_matches"]["conditional_model_context"]["query"]
    query = {"source_path": focus["source_path"], "symbols": focus["symbols"],
             "property": focus["property"], "limit": 12}
    queried = query_repository_index(**inputs, expected=index, output=output / "repository-index", query=query)
    _write(output / "repository-generation.json", current["source_snapshot"])
    _write(output / "repository-index-result.json", index)
    _write(output / "repository-index-replay.json", replay)
    _write(output / "repository-query.json", queried)
    phases["native_repository_production_typed_index_and_query"] = time.monotonic() - phase
    phase = time.monotonic()
    snapshot_inputs = {"intent_experiment": parent["root"], "manifest": parent["capture"]["manifest"],
        "workflow_request": parent["capture"]["workflow_request"], "control": parent["result"]["intent_control"],
        "proof_index_manifest": proof,
        "match_result": parent["result"]["intent_codebase_matches"]["conditional_model_context"],
        "repository_index_manifest": index, "repository_query": queried,
        "frozen_learning": learning, "learning_artifact": learning_artifact}
    planned = build_indexed_repository_planning_snapshot(**snapshot_inputs)
    _write(output / "indexed-planning-snapshot.json", planned["snapshot"])
    _write(output / "symbolic-graph.json", planned["symbolic_plan"]["graph"].to_dict())
    _write(output / "symbolic-receipt.json", planned["symbolic_plan"]["receipt"])
    phases["native_indexed_planning_material_freeze"] = time.monotonic() - phase
    phase = time.monotonic()
    edited = _edit_control(output=output / "edit-control", parent=parent, index=index, query=query)
    _write(output / "edit-control-result.json", edited)
    phases["dirty_source_invalidation_and_cold_rebuild"] = time.monotonic() - phase
    _progress("source_edit_controls_complete", stale_index="refused", inherited_proof_entries=0)
    records = deepcopy(parent["records"])
    for family, rows in current["records"].items():
        records[family] = rows
    for family, rows in learning["records"].items():
        records[family] = rows
    records["repository_index"] = [index, replay, queried, edited]
    records["repository_planning_snapshot"].append(planned["snapshot"])
    records["repository_symbolic_receipt"].append(planned["symbolic_plan"]["receipt"])
    records["execution_sources"].extend(execution_sources)
    complete_sha = _sha(_wire(records))
    bounded = bound_terminal_codebase_metadata_records(records)
    if _wire(reconstruct_terminal_codebase_metadata_records(bounded)) != _wire(records):
        raise ValueError("complete current repository producer changed during lossless packaging")
    source_snapshot = {"schema": "terminal-codebase-indexed-repository-source@1",
        "parent": parent["binding"], "current_repository_source": current["source_snapshot"],
        "repository_index_manifest_id": index["manifest_id"],
        "indexed_planning_snapshot_id": planned["snapshot"]["snapshot_id"],
        "frozen_learning_sha256": _sha(_wire(learning)), "complete_producer_sha256": complete_sha,
        "current_behavioral_facts": [], "public_request_planned": False}
    phase = time.monotonic()
    metadata = hydrate_codebase_ir_metadata(records=bounded, source_snapshot=source_snapshot, output=output / "metadata")
    native_replay = validate_codebase_ir_metadata(output=output / "metadata", expected=metadata, fresh_process=True)
    restored = {family: [json.loads(line)["payload"] for line in
        _read(output / "metadata" / descriptor["relative_path"], 32 * 1024 * 1024).splitlines()]
        for family, descriptor in native_replay["exports"].items()}
    recovered = reconstruct_terminal_codebase_metadata_records(restored)
    if _sha(_wire(recovered)) != complete_sha:
        raise ValueError("native current repository complete producer reconstruction differs")
    reconstruction = {"schema": "terminal-codebase-indexed-repository-reconstruction@1",
        "original_records_sha256": complete_sha, "recovered_records_sha256": _sha(_wire(recovered)),
        "original_family_counts": {name: len(rows) for name, rows in records.items()},
        "bounded_family_counts": metadata["family_counts"], "zero_truncation": True}
    _write(output / "metadata-result.json", metadata)
    _write(output / "metadata-replay.json", native_replay)
    _write(output / "metadata-reconstruction.json", reconstruction)
    phases["native_full_metadata_hydration_and_fresh_replay"] = time.monotonic() - phase
    phase = time.monotonic()
    replay_indexed_repository_planning_snapshot(expected=planned["snapshot"], **snapshot_inputs)
    if (load_completed_intent_experiment(parent["root"])["binding"] != parent["binding"]
            or any(_artifact(row["path"]) != row for row in execution_sources)):
        raise ValueError("parent source or repository experiment implementation changed during execution")
    phases["final_exact_planning_and_source_replay"] = time.monotonic() - phase
    result = {"schema": SCHEMA, "status": "completed", "output": str(output),
        "parent": parent["binding"], "execution_sources": execution_sources,
        "repository_source_snapshot": current["source_snapshot"], "repository_index": index,
        "repository_index_replay": replay, "repository_query": queried,
        "frozen_learning": learning_artifact,
        "learning_family_counts": {name: len(rows) for name, rows in learning["records"].items()},
        "indexed_planning_snapshot": planned["snapshot"], "edit_control": edited,
        "metadata": metadata, "metadata_reconstruction": reconstruction,
        "phase_seconds": phases, "elapsed_seconds": time.monotonic() - started,
        "started_at": started_at, "completed_at": datetime.now(timezone.utc).isoformat(),
        "current_behavioral_facts": [], "public_request_planned": False,
        "repository_evidence_admitted": False, "successor_admission": False,
        "provider_calls": 0, "training_steps": 0, "source_execution": False,
        "official_verifier_executed": False, "official_reward": None, **AUTHORITY}
    _write(output / "result.json", result)
    _progress("repository_experiment_complete", result=str(output / "result.json"), seconds=result["elapsed_seconds"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent-experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        run_repository_experiment(intent_experiment=args.intent_experiment, output=args.output)
    except Exception as exc:
        _progress("repository_experiment_failed", error_type=type(exc).__name__, message=str(exc))
        if not existed and (args.output / "experiment-policy.json").is_file():
            _write(args.output / "failure.json", {"schema": SCHEMA, "status": "failed",
                "error_type": type(exc).__name__, "message": str(exc), "completed_join_claimed": False})
        raise


if __name__ == "__main__":
    main()
