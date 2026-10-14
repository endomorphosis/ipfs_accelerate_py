#!/usr/bin/env python3
"""Lossless, bounded LAC-002 evidence envelope for native proposal file limits.

The unchanged bootstrap collector and checker retain authority over the full
inventory. This helper changes its storage representation only: no run, input,
numeric evidence, gap, or provenance record is selected out of the payload.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zlib


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "legal-autoformalization-training-inventory-envelope/v1"
TASK = "LAC-002"
HELPER_PATH = "scripts/legal_autoformalization_training_inventory.py"
MAX_RAW_BYTES = 16 * 1024 * 1024
MAX_COMPRESSED_BYTES = 600 * 1024
MAX_ENCODED_BYTES = 800 * 1024
MAX_FILE_BYTES = 900 * 1024  # Strictly below the unchanged native 1 MiB gate.

_spec = importlib.util.spec_from_file_location(
    "_legal_autoformalization_training_inventory_core",
    ROOT / "scripts/legal_autoformalization_bootstrap.py")
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def summary(full):
    observations = full["observations"]
    runs = observations["runs"]
    core.need(isinstance(runs, list), "Full run population must be a list")
    lineages = Counter(row["lineage"] for row in runs)
    return {"source_record_count": observations["source_record_count"],
            "deduplicated_run_count": observations["deduplicated_run_count"],
            "retained_run_count": len(runs),
            "input_artifact_count": len(full["input_artifacts"]),
            "gap_count": len(full["gaps"]),
            "runs_by_lineage": dict(sorted(lineages.items()))}


def pack_result(full):
    raw = core.canonical(full)
    core.need(len(raw) <= MAX_RAW_BYTES, "Full training inventory exceeds raw payload cap")
    compressed = gzip.compress(raw, mtime=0)
    core.need(len(compressed) <= MAX_COMPRESSED_BYTES, "Compressed training inventory exceeds cap")
    encoded = base64.b64encode(compressed).decode("ascii")
    core.need(len(encoded) <= MAX_ENCODED_BYTES, "Encoded training inventory exceeds cap")
    envelope = {"schema": SCHEMA, "campaign_id": full["campaign_id"], "task_id": TASK,
        "scope": full["scope"], "original_task_acceptance": "partial",
        "legal_semantic_acceptance": False, "summary": summary(full),
        "writer": {"repository_path": HELPER_PATH,
                   "captured_path": str(Path(__file__).resolve()),
                   "sha256": core.file_identity(__file__)["sha256"]},
        "payload": {"encoding": "gzip+base64", "raw_bytes": len(raw),
                    "raw_sha256": sha256(raw), "compressed_bytes": len(compressed),
                    "compressed_sha256": sha256(compressed),
                    "encoded_bytes": len(encoded), "data": encoded}}
    core.need(len(core.canonical(envelope)) + 1 <= MAX_FILE_BYTES,
              "Training inventory envelope exceeds native file cap")
    return envelope


def unpack_result(envelope):
    core.need(isinstance(envelope, dict), "Training inventory envelope must be an object")
    core.need(len(core.canonical(envelope)) + 1 <= MAX_FILE_BYTES, "Training envelope exceeds file cap")
    core.need(envelope.get("schema") == SCHEMA and envelope.get("task_id") == TASK,
              "Wrong training envelope identity")
    core.need(envelope.get("original_task_acceptance") == "partial"
              and envelope.get("legal_semantic_acceptance") is False,
              "Training envelope cannot confer original or legal acceptance")
    writer = envelope.get("writer")
    core.need(isinstance(writer, dict) and writer.get("repository_path") == HELPER_PATH,
              "Wrong training envelope writer")
    core.need(writer.get("sha256") == core.file_identity(__file__)["sha256"],
              "Training envelope writer code moved")
    payload = envelope.get("payload")
    core.need(isinstance(payload, dict) and payload.get("encoding") == "gzip+base64",
              "Unsupported training payload encoding")
    for field, maximum in (("raw_bytes", MAX_RAW_BYTES),
                           ("compressed_bytes", MAX_COMPRESSED_BYTES),
                           ("encoded_bytes", MAX_ENCODED_BYTES)):
        core.need(type(payload.get(field)) is int and 0 < payload[field] <= maximum,
                  f"Training payload {field} exceeds cap or has invalid type")
    encoded = payload.get("data")
    core.need(isinstance(encoded, str) and len(encoded) == payload["encoded_bytes"]
              and len(encoded) <= MAX_ENCODED_BYTES, "Encoded training payload size differs")
    compressed = base64.b64decode(encoded, validate=True)
    core.need(len(compressed) == payload["compressed_bytes"]
              and sha256(compressed) == payload.get("compressed_sha256"),
              "Compressed training payload identity differs")
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    # Limit before allocation. A dishonest size or compressed expansion bomb
    # cannot force an unbounded decompress/flush operation.
    raw = decoder.decompress(compressed, payload["raw_bytes"] + 1)
    core.need(len(raw) == payload["raw_bytes"] and decoder.eof
              and not decoder.unconsumed_tail and not decoder.unused_data,
              "Training payload expansion bound, length, or gzip stream differs")
    core.need(sha256(raw) == payload.get("raw_sha256"), "Raw training payload identity differs")
    full = json.loads(raw)
    core.need(isinstance(full, dict), "Full training inventory must be an object")
    for field in ("campaign_id", "task_id", "scope", "original_task_acceptance", "legal_semantic_acceptance"):
        core.need(core.canonical(envelope.get(field)) == core.canonical(full.get(field)),
                  f"Training envelope/full {field} differs")
    core.need(core.canonical(envelope.get("summary")) == core.canonical(summary(full)),
              "Training envelope summary differs from full population")
    return full


def validate_envelope(envelope, sources_path, sources):
    full = unpack_result(envelope)
    checked = core.validate(full, TASK, sources_path, sources)
    return {**checked, "envelope_schema": SCHEMA, "summary": summary(full),
            "payload_raw_bytes": envelope["payload"]["raw_bytes"],
            "payload_raw_sha256": envelope["payload"]["raw_sha256"]}


def load_envelope(path):
    core.need(path.stat().st_size <= MAX_FILE_BYTES, "Training envelope file exceeds cap")
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    core.need(len(raw) <= MAX_FILE_BYTES, "Training envelope file exceeds cap")
    return json.loads(raw)


def write_envelope_exclusive(path, envelope):
    """Keep compact file bytes within the cap and never replace evidence."""
    raw = core.canonical(envelope) + b"\n"
    core.need(len(raw) <= MAX_FILE_BYTES, "Training envelope file exceeds cap")
    # Reuse the qualified exclusive atomic publisher, checking its exact
    # pretty-JSON representation before publication as well as compact bytes.
    pretty = json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2,
                        allow_nan=False).encode("utf-8") + b"\n"
    core.need(len(pretty) <= MAX_FILE_BYTES, "Training envelope serialized file exceeds cap")
    core.atomic_write(path, envelope, replace_existing=False)


def materialize(sources_path, sources, path):
    core.need(path == core.output_path(TASK, str(path)), "Training result escapes task output scope")

    def existing():
        before = core.file_identity(path)
        checked = validate_envelope(load_envelope(path), sources_path, sources)
        after = core.file_identity(path)
        core.need(before["sha256"] == after["sha256"] and before["bytes"] == after["bytes"],
                  "Training result changed during validation")
        return {**checked, "materialized": False, "output": str(path),
                "sha256": after["sha256"], "bytes": after["bytes"]}

    if path.exists():
        return existing()
    full = core.common_result(TASK, sources_path, sources)
    core.collect_runs(full, sources)
    core.validate(full, TASK, sources_path, sources)
    envelope = pack_result(full)
    checked = validate_envelope(envelope, sources_path, sources)
    try:
        write_envelope_exclusive(path, envelope)
    except FileExistsError:
        return existing()
    identity = core.file_identity(path)
    return {**checked, "materialized": True, "output": str(path),
            "sha256": identity["sha256"], "bytes": identity["bytes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("materialize", "validate"))
    parser.add_argument("--task", choices=(TASK,), required=True)
    parser.add_argument("--sources", default="config/legal_autoformalization_bootstrap_sources.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    sources_path = Path(args.sources)
    if not sources_path.is_absolute():
        sources_path = ROOT / sources_path
    sources = core.load_json(sources_path)
    path = core.output_path(TASK, args.output)
    if args.command == "materialize":
        checked = materialize(sources_path, sources, path)
    else:
        checked = validate_envelope(load_envelope(path), sources_path, sources)
    print(json.dumps(checked, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError, zlib.error) as error:
        print(json.dumps({"valid": False, "status": "refused", "error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
