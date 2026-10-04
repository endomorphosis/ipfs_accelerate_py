"""Read-only carry of exact, frozen codebase learning into a current catalog.

The original feature/vector/training inventories remain model candidates. This
helper validates native storage, the qualified fixture and actual frozen-weight
inference. It never fits a model, executes source, repairs code or publishes
authoritative facts. New or changed Python inputs cannot reuse this generation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from .terminal_codebase_supervisor_fixture import (
    MAX_ARTIFACT_BYTES, _read, _json_bytes,
    extract_terminal_codebase_metadata_records,
    reconstruct_terminal_codebase_metadata_records,
)

SCHEMA = "terminal-codebase-frozen-learning-capture@1"
FAMILIES = ("features", "vectors", "feature_frontiers", "training")
AUTHORITY = {"proof_authority": False, "execution_authority": False,
    "completion_authority": False, "mutation_authority": False,
    "source_semantics_verified": False, "formalization_authority": False,
    "omission_authority": False, "behavioral_satisfaction": False}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _pin(path):
    path = Path(path).absolute()
    raw = _read(path, MAX_ARTIFACT_BYTES)
    return {"path": str(path), "sha256": _sha(raw), "bytes": len(raw)}


def _json(path):
    return json.loads(_read(Path(path).absolute(), MAX_ARTIFACT_BYTES))


def _manifest(path, expected_id):
    manifest = _json(path)
    body = dict(manifest)
    identifier = body.pop("manifest_id", None)
    if identifier != expected_id or identifier != "sha256:" + _sha(_json_bytes(body)):
        raise ValueError("frozen ancestor manifest identity differs")
    return manifest


def _current_sources(values):
    if type(values) is not dict or not 1 <= len(values) <= 256:
        raise ValueError("bounded complete current source byte dictionary required")
    result, total = {}, 0
    if any(type(name) is not str for name in values):
        raise ValueError("exact canonical current source bytes required")
    for name, raw in sorted(values.items()):
        if (not name or name == "." or PurePosixPath(name).is_absolute()
                or str(PurePosixPath(name)) != name or "\\" in name
                or any(part in {"..", ".git", ".runtime"} for part in PurePosixPath(name).parts)
                or type(raw) is not bytes or len(raw) > 2_000_000):
            raise ValueError("exact canonical current source bytes required")
        total += len(raw)
        if total > 4_000_000:
            raise ValueError("current source byte budget exceeded")
        result[name] = {"sha256": _sha(raw), "bytes": len(raw)}
    return result


def _lineage(intent_experiment):
    root = Path(intent_experiment).absolute()
    if root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError("canonical completed intent experiment required")
    intent = _json(root / "result.json")
    if (intent.get("schema") != "terminal-codebase-intent-experiment@1"
            or intent.get("status") != "completed" or intent.get("output") != str(root)
            or intent.get("training_steps") != 0 or intent.get("current_behavioral_facts") != []):
        raise ValueError("completed model-only intent experiment required")
    intent_metadata = intent["metadata"]
    if (intent_metadata["output"] != str(root / "metadata")
            or "sha256:" + _pin(root / "metadata/manifest.json")["sha256"]
                != intent_metadata["manifest_sha256"]):
        raise ValueError("exact intent native metadata manifest required")
    intent_index = intent["proof_index"]
    if (intent_index["output"] != str(root / "proof-index")
            or _manifest(root / "proof-index/manifest.json", intent_index["manifest_id"]) != intent_index):
        raise ValueError("complete intent proof index manifest differs")
    parent = intent["parent"]
    decoder_root = Path(parent["decoder_experiment"])
    for name in ("decoder_result", "decoder_manifest", "captured_public_source", "captured_public_instruction"):
        if _pin(parent[name]["path"]) != parent[name]:
            raise ValueError("intent ancestor artifact drift: " + name)
    if (parent["decoder_result"]["path"] != str(decoder_root / "result.json")
            or parent["decoder_manifest"]["path"] != str(decoder_root / "codebase-ir-manifest.json")):
        raise ValueError("exact decoder ancestor namespace required")
    decoder = _json(decoder_root / "result.json")
    decoder_manifest = _manifest(decoder_root / "codebase-ir-manifest.json", parent["decoder_manifest_id"])
    if (decoder.get("schema") != "terminal-codebase-decoder-experiment@1"
            or decoder.get("status") != "completed" or decoder.get("output") != str(decoder_root)
            or decoder["manifest_id"] != decoder_manifest["manifest_id"]):
        raise ValueError("completed exact decoder ancestor required")
    capture = decoder["parent_capture"]
    original = Path(capture["prepared_experiment"])
    for name in ("parent_manifest", "parent_metadata_manifest", "source"):
        if _pin(capture[name]["path"]) != capture[name]:
            raise ValueError("decoder ancestor artifact drift: " + name)
    if (capture["parent_manifest"]["path"] != str(original / "codebase-ir-manifest.json")
            or capture["parent_metadata_manifest"]["path"] != str(original / "metadata/manifest.json")):
        raise ValueError("exact original qualification namespace required")
    qualification = _json(original / "result.json")
    manifest = _manifest(original / "codebase-ir-manifest.json", capture["parent_ir_manifest_id"])
    if (qualification.get("schema") != "terminal-codebase-ir-experiment@1"
            or qualification.get("status") != "completed" or qualification.get("output") != str(original)
            or qualification["codebase_ir_manifest_id"] != manifest["manifest_id"]
            or manifest["source_snapshot"] != capture["source_snapshot"]
            or qualification["metadata"] != manifest["metadata"]
            or qualification["metadata_reconstruction"] != manifest["metadata_reconstruction"]
            or qualification["source_snapshot"] != capture["source_snapshot"]):
        raise ValueError("complete original qualification/source snapshot differs")
    fixture_path = original / "supervisor/fixture.json"
    fixture = _json(fixture_path)
    fixture["fixture_artifact"] = _pin(fixture_path)
    if (fixture["schema"] != "terminal-codebase-supervisor-fixture@1"
            or fixture["output"] != str(original / "supervisor")
            or fixture["repository"] != capture["source_snapshot"]["repository"]
            or fixture["learner"]["checkpoint_sha256"] != manifest["training"]["checkpoint_sha256"]
            or fixture["learner"]["receipt_sha256"] != manifest["training"]["learner_receipt_sha256"]):
        raise ValueError("qualified original learner fixture differs")
    # No old helper may create a missing sidecar during this read-only carry.
    _pin(original / "supervisor/metadata-records.json")
    return root, intent, decoder_root, decoder, original, qualification, manifest, fixture


def _export_records(metadata, manifest):
    """Read every pinned family export; never select a truncated inventory."""
    records, pins, total = {}, [], 0
    for family, descriptor in sorted(manifest["families"].items()):
        relative = descriptor["export"]["relative_path"]
        if relative != "exports/" + family + ".jsonl":
            raise ValueError("exact native family export path required")
        path = metadata / relative
        raw = _read(path, MAX_ARTIFACT_BYTES)
        total += len(raw)
        if total > 64 * 1024 * 1024:
            raise ValueError("complete export byte budget exceeded")
        if (descriptor["export"]["sha256"] != "sha256:" + _sha(raw)
                or descriptor["export"]["bytes"] != len(raw)):
            raise ValueError("native producer export artifact changed")
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != descriptor["count"]:
            raise ValueError("complete producer export population differs")
        for ordinal, row in enumerate(rows):
            if (row["family"] != family or row["row_ordinal"] != ordinal
                    or row["source_snapshot_sha256"] != manifest["source_snapshot_sha256"]):
                raise ValueError("native producer export source/order differs")
        records[family] = [row["payload"] for row in rows]
        pins.append(_pin(path))
    return reconstruct_terminal_codebase_metadata_records(records), pins


def _native_learning(*, fixture, records):
    from ipfs_datasets_py.logic.formalization.autoencoder.security.codebase_autoencoder import validate_codebase_autoencoder

    learner = fixture["learner"]
    verified = validate_codebase_autoencoder(repository=Path(fixture["repository"]), expected_receipt=learner)
    output = Path(learner["output"])
    features, vectors = _json(output / "features.json"), _json(output / "index.json")
    checkpoint, receipt = _json(output / "checkpoint.json"), _json(output / "receipt.json")
    expected_vectors = [{**row, "checkpoint_sha256": learner["checkpoint_sha256"],
        "authority": "unverified_candidate_only", "proof_authority": False,
        "execution_authority": False, "completion_authority": False, "official_reward": None}
        for row in vectors["ranks"]]
    if (records["features"] != features["rows"] or records["feature_frontiers"] != features["unsupported"]
            or records["vectors"] != expected_vectors or len(records["training"]) != 1
            or records["training"][0]["learner"] != learner
            or records["training"][0]["receipt"] != receipt
            or records["training"][0]["checkpoint"] != checkpoint
            or verified["ranks"] != vectors["ranks"]
            or verified["sample_count"] != len(features["rows"])):
        raise ValueError("complete original learning inventories differ from native checkpoint/inference")
    if verified["status"] != "verified" or any(verified[key] is not False for key in (
            "proof_authority", "formalization_authority", "omission_authority")):
        raise ValueError("candidate-only native learning validation required")
    return verified, [_pin(output / (name + ".json")) for name in ("receipt", "checkpoint", "features", "index")]


def capture_frozen_codebase_learning(*, intent_experiment: Path,
        current_source_bytes: dict[str, bytes], fresh_process: bool = False) -> dict:
    """Bind current Python bytes to complete original source/model inventories.

    Additional non-Python signed source leaves are retained as context, never
    used as model labels. ``fresh_process`` affects validation effort only; the
    returned artifact is deterministic and identical for both settings.
    """
    from .codebase_ir_metadata import validate_codebase_ir_metadata

    if type(fresh_process) is not bool:
        raise ValueError("explicit fresh-process validation Boolean required")
    current = _current_sources(current_source_bytes)
    lineage = _lineage(intent_experiment)
    root, intent, decoder_root, decoder, original, qualification, manifest, fixture = lineage
    training_sources = fixture["learner"]["source_hashes"]
    if (set(training_sources) != {"bottle.py", ".supervisor-public-smoke.py"}
            or {name: pin["sha256"] for name, pin in current.items() if name.endswith(".py")} != training_sources
            or manifest["source_snapshot"]["training_sources"] != training_sources):
        raise ValueError("current signed Python source scope differs from the frozen training generation")
    metadata = original / "metadata"
    native = validate_codebase_ir_metadata(output=metadata, expected=qualification["metadata"],
        fresh_process=fresh_process)
    native.pop("fresh_process_readback", None)
    native_manifest = _json(metadata / "manifest.json")
    recovered, exports = _export_records(metadata, native_manifest)
    recovered_sha = _sha(_json_bytes(recovered))
    reconstruction = qualification["metadata_reconstruction"]
    if (reconstruction["status"] != "exact" or reconstruction["all_fields_preserved"] is not True
            or reconstruction["truncated_records"] != 0
            or recovered_sha != reconstruction["original_records_sha256"]
            or recovered_sha != reconstruction["recovered_records_sha256"]):
        raise ValueError("complete original producer reconstruction differs")
    # Reuses the qualified existing AST/semantic/retrieval/fixture checks and
    # only compares an already-existing exact sidecar; it cannot create outputs.
    replayed = reconstruct_terminal_codebase_metadata_records(extract_terminal_codebase_metadata_records(fixture))
    if any(recovered.get(family) != rows for family, rows in replayed.items()):
        raise ValueError("native original producer catalog differs from qualified fixture replay")
    if any(family not in recovered for family in FAMILIES):
        raise ValueError("complete four-family learning inventories required")
    verified, model_artifacts = _native_learning(fixture=fixture, records=recovered)
    provenance = {"schema": "terminal-codebase-frozen-learning-provenance@1",
        "intent_result": _pin(root / "result.json"),
        "intent_metadata_manifest": _pin(root / "metadata/manifest.json"),
        "intent_proof_index_manifest": _pin(root / "proof-index/manifest.json"),
        "decoder_result": _pin(decoder_root / "result.json"),
        "decoder_manifest": _pin(decoder_root / "codebase-ir-manifest.json"),
        "qualification_result": _pin(original / "result.json"),
        "qualification_manifest": _pin(original / "codebase-ir-manifest.json"),
        "fixture": fixture["fixture_artifact"], "metadata_manifest": _pin(metadata / "manifest.json"),
        "metadata_exports": exports, "producer_export": _pin(original / "supervisor/metadata-records.json"),
        "model_artifacts": model_artifacts, "original_source_snapshot": manifest["source_snapshot"],
        "parent_manifest_ids": {"decoder": decoder["manifest_id"], "qualification": manifest["manifest_id"]},
        "complete_producer_sha256": recovered_sha, "complete_original_family_counts": {
            family: len(rows) for family, rows in sorted(recovered.items())},
        "scope": "same_current_Python_bytes; exact_frozen_model_and_native_inference; candidates_only"}
    # Freeze mutable inputs and fence referenced artifacts across storage and
    # model replay; original sources are also rechecked by the native validators.
    after = _lineage(intent_experiment)
    normalized = lambda value: [str(item) if isinstance(item, Path) else item for item in value]
    if (_json_bytes(normalized(after)) != _json_bytes(normalized(lineage))
            or _current_sources(current_source_bytes) != current):
        raise ValueError("ancestor or current source changed during frozen catalog capture")
    for pin in [*exports, *model_artifacts, provenance["producer_export"], provenance["fixture"]]:
        if _pin(pin["path"]) != pin:
            raise ValueError("bound producer/model artifact changed during capture")
    result = {"schema": SCHEMA, "status": "captured_current_candidate_learning",
        "records": {family: recovered[family] for family in FAMILIES},
        "provenance": provenance, "current_sources": current, "training_sources": training_sources,
        "learning_validation": verified, "metadata_validation": native,
        "training_steps": 0, "provider_calls": 0, "source_executed": False,
        "canonical_state_mutated": False, "official_reward": None, **AUTHORITY}
    result["capture_sha256"] = _sha(_json_bytes(result))
    return json.loads(_json_bytes(result))


__all__ = ["capture_frozen_codebase_learning", "SCHEMA"]
