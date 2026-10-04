"""Lossless experimental assembly of bounded, independently native metadata parts.

Every part uses the unchanged DuckDB/DuckLake metadata owner and its limits.
The global manifest and exports describe an assembly, not one native transaction
or a new proof, publication, completion, or execution grant. Failed creation
retains its partial namespace; this profile does not claim crash-safe commit.
"""
from __future__ import annotations

import argparse
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from . import codebase_ir_metadata as native
from .terminal_codebase_supervisor_fixture import (
    ARTIFACT_SCHEMA, CHUNK_BYTES, CHUNK_FAMILY, CHUNK_SCHEMA, MAX_ARTIFACT_BYTES,
    reconstruct_terminal_codebase_metadata_records,
)

MODULE = "benchmarks.agent_supervisor.container_coding.finite_repository_sharded_metadata"
SCHEMA = "experimental-codebase-ir-sharded-metadata-manifest@1"
REPORT_SCHEMA = "experimental-codebase-ir-sharded-metadata-readback@1"
PART_SOURCE_SCHEMA = "experimental-codebase-ir-sharded-part-source@1"
LIMITS = {"part_payload_bytes": 32 * 1024**2, "part_rows": 32768, "parts": 16,
          "total_payload_bytes": 512 * 1024**2, "rows": 524288, "families": 32,
          "manifest_bytes": 16 * 1024**2, "export_bytes": 256 * 1024**2,
          "restart_seconds": 300}
CHUNK_POLICY = {"schema": CHUNK_SCHEMA, "bytes": CHUNK_BYTES, "family": CHUNK_FAMILY,
                "producer_artifact_bytes": MAX_ARTIFACT_BYTES}
QUALIFICATION = {"experimental": True, "production_activated": False, "admitted": False,
                 "proof_authority": False, "execution_authority": False,
                 "completion_authority": False, "whole_population_native_transaction": False}


class ShardedMetadataError(ValueError):
    """An incomplete, altered, conflicting or oversized shard assembly."""


def _need(condition, message):
    if not condition:
        raise ShardedMetadataError(message)


def _same(left, right):
    return native._wire(left) == native._wire(right)


def _closed_errors(function):
    """Normalize malformed or inaccessible artifact refusals at the public API."""
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except ShardedMetadataError:
            raise
        except Exception as error:
            raise ShardedMetadataError("sharded metadata operation refused: " + str(error)) from error
    return checked


def _records_digest(records):
    """Canonical mapping digest without constructing one aggregate JSON blob."""
    digest = hashlib.sha256()
    digest.update(b"{")
    for number, family in enumerate(sorted(records)):
        if number:
            digest.update(b",")
        digest.update(native._wire(family) + b":[")
        for ordinal, payload in enumerate(records[family]):
            if ordinal:
                digest.update(b",")
            digest.update(native._wire(payload))
        digest.update(b"]")
    digest.update(b"}")
    return "sha256:" + digest.hexdigest()


def _family_digest(payloads):
    digest = hashlib.sha256(b"[")
    for ordinal, payload in enumerate(payloads):
        if ordinal:
            digest.update(b",")
        digest.update(native._wire(payload))
    digest.update(b"]")
    return "sha256:" + digest.hexdigest()


def _global_row_root(records, source_digest):
    digest = hashlib.sha256(b"{")
    for number, family in enumerate(sorted(records)):
        if number:
            digest.update(b",")
        digest.update(native._wire(family) + b":[")
        for ordinal, payload in enumerate(records[family]):
            if ordinal:
                digest.update(b",")
            digest.update(native._wire(native._row(family, payload, ordinal, source_digest)))
        digest.update(b"]")
    digest.update(b"}")
    return "sha256:" + digest.hexdigest()


def partition_codebase_ir_metadata_records(*, records, max_payload_bytes=None, max_rows=None):
    """Deterministic pure partition; test budgets may only reduce native part caps.

    Original row bytes and chunk descriptors retain global ordinals. Defaults
    are represented as empty families, while ``input_families`` preserves the
    original mapping's exact keys for producer reconstruction.
    """
    byte_cap = LIMITS["part_payload_bytes"] if max_payload_bytes is None else max_payload_bytes
    row_cap = LIMITS["part_rows"] if max_rows is None else max_rows
    _need(type(byte_cap) is int and 0 < byte_cap <= LIMITS["part_payload_bytes"]
          and type(row_cap) is int and 0 < row_cap <= LIMITS["part_rows"], "bounded partition budgets required")
    _need(type(records) is dict and all(type(name) is str and native._FAMILY.fullmatch(name) for name in records),
          "named metadata family mapping required")
    names = sorted(set(records) | set(native.DEFAULT_FAMILIES))
    _need(len(names) <= LIMITS["families"], "global family bound exceeded")
    normalized, total, payload_bytes = {}, 0, 0
    for family in names:
        values = records.get(family, [])
        _need(type(values) is list, "metadata family arrays required")
        normalized[family], identities, declared = [], set(), set()
        for ordinal, payload in enumerate(values):
            _need(type(payload) is dict, "ordinary metadata object row required")
            if payload.get("schema") == ARTIFACT_SCHEMA:
                _need(type(payload.get("original_ordinal")) is int and payload["original_ordinal"] == ordinal
                      and payload.get("family") == family
                      and type(payload.get("payload_bytes")) is int
                      and 1 <= payload["payload_bytes"] <= MAX_ARTIFACT_BYTES
                      and type(payload.get("chunk_count")) is int
                      and payload["chunk_count"] == (payload["payload_bytes"] + CHUNK_BYTES - 1) // CHUNK_BYTES,
                      "bounded producer artifact descriptor with exact global ordinal required")
            raw = native._json(payload)
            _need(len(raw) <= native.LIMITS["row_bytes"] and len(raw) <= byte_cap,
                  "one payload exceeds unchanged row or part bound")
            identity = native._digest(raw)
            _need(identity not in identities, "duplicate global metadata row identity")
            identities.add(identity)
            if "row_id" in payload:
                declared_id = native._json(payload["row_id"])
                _need(declared_id not in declared, "conflicting global declared row identity")
                declared.add(declared_id)
            normalized[family].append(json.loads(raw))
            total += 1
            payload_bytes += len(raw)
            _need(total <= LIMITS["rows"] and payload_bytes <= LIMITS["total_payload_bytes"],
                  "global metadata population bound exceeded; no truncation permitted")
    original = {family: normalized[family] for family in sorted(records)}
    producers = reconstruct_terminal_codebase_metadata_records(original)
    parts, positions = [], {family: 0 for family in names}
    pending, starts, pending_bytes, pending_rows = {family: [] for family in names}, dict(positions), 0, 0

    def flush():
        nonlocal pending, starts, pending_bytes, pending_rows
        _need(len(parts) < LIMITS["parts"], "global part count bound exceeded")
        parts.append({"ordinal": len(parts), "records": pending, "row_count": pending_rows,
                      "payload_bytes": pending_bytes, "global_family_ranges": {
                          family: {"start": starts[family], "stop": positions[family],
                                   "count": positions[family] - starts[family]} for family in names}})
        pending, starts, pending_bytes, pending_rows = {family: [] for family in names}, dict(positions), 0, 0

    for family in names:
        for payload in normalized[family]:
            size = len(native._json(payload))
            if pending_rows and (pending_rows == row_cap or pending_bytes + size > byte_cap):
                flush()
            pending[family].append(payload)
            positions[family] += 1
            pending_rows += 1
            pending_bytes += size
    if pending_rows or not parts:
        flush()
    return {"input_families": sorted(records), "records": normalized,
            "family_counts": {family: len(values) for family, values in normalized.items()},
            "row_count": total, "payload_bytes": payload_bytes,
            "packaged_sha256": _records_digest(original), "producer_sha256": _records_digest(producers),
            "parts": parts}


def _part_source(manifest, part):
    return {"schema": PART_SOURCE_SCHEMA, "original_source_snapshot": manifest["source_snapshot"],
            "global_source_snapshot_sha256": manifest["source_snapshot_sha256"],
            "global_packaged_sha256": manifest["packaged_sha256"],
            "global_producer_sha256": manifest["producer_sha256"], "part_ordinal": part["ordinal"],
            "global_family_ranges": part["global_family_ranges"]}


def _stable_native_report(report):
    _need(type(report) is dict and report.get("schema") == native.REPORT_SCHEMA,
          "actual native metadata report required")
    return {key: value for key, value in report.items() if key != "fresh_process_readback"}


def _descriptor(path, relative, maximum):
    raw = native._read(path, maximum)
    return {"relative_path": relative, "sha256": native._digest(raw), "bytes": len(raw)}


def _check_descriptor(output, descriptor, maximum, relative=None):
    _need(type(descriptor) is dict and set(descriptor) == {"relative_path", "sha256", "bytes"}
          and type(descriptor["relative_path"]) is str
          and not Path(descriptor["relative_path"]).is_absolute()
          and ".." not in Path(descriptor["relative_path"]).parts
          and type(descriptor["bytes"]) is int and 0 <= descriptor["bytes"] <= maximum
          and (relative is None or descriptor["relative_path"] == relative), "closed artifact descriptor required")
    raw = native._read(output / descriptor["relative_path"], maximum)
    _need(len(raw) == descriptor["bytes"] and native._digest(raw) == descriptor["sha256"],
          "sharded artifact bytes or digest differ")
    return raw


def _part_payloads(output, report, names, source_digest):
    _need(type(report) is dict and all(type(report.get(key)) is dict
          for key in ("exports", "family_counts", "family_digests"))
          and set(report["exports"]) == set(report["family_counts"]) == set(report["family_digests"]) == set(names),
          "native part family population differs")
    recovered = {}
    for family in names:
        descriptor = report["exports"][family]
        raw = _check_descriptor(output, descriptor, 2 * native.LIMITS["total_payload_bytes"],
                                "exports/" + family + ".jsonl")
        rows, payloads = [], []
        for ordinal, line in enumerate(raw.splitlines()):
            row = native._loads(line)
            _need(type(row) is dict and "payload" in row, "complete native export row required")
            wanted = native._row(family, row["payload"], ordinal, source_digest)
            _need(native._wire(wanted) == line, "native part export order or row identity differs")
            rows.append(wanted)
            payloads.append(wanted["payload"])
        _need(len(payloads) == report["family_counts"][family]
              and native._digest(native._row_array(rows)) == report["family_digests"][family],
              "native part export count or full digest differs")
        recovered[family] = payloads
    return recovered


def _write_global_exports(output, records, source_digest):
    exports = {}
    for family in sorted(records):
        relative = "exports/" + family + ".jsonl"
        path = output / relative
        count, digest = 0, hashlib.sha256()
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            for ordinal, payload in enumerate(records[family]):
                raw = native._wire(native._row(family, payload, ordinal, source_digest)) + b"\n"
                count += len(raw)
                _need(count <= LIMITS["export_bytes"], "global family export bound exceeded")
                digest.update(raw)
                stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        exports[family] = {"relative_path": relative, "sha256": "sha256:" + digest.hexdigest(), "bytes": count}
        _check_descriptor(output, exports[family], LIMITS["export_bytes"], relative)
    return exports


def _report(manifest, manifest_digest):
    keys = ("output", "source_snapshot_sha256", "input_families", "packaged_sha256", "producer_sha256",
            "row_count", "payload_bytes", "row_root_sha256", "family_counts", "family_digests",
            "exports", "parts", "limits", "native_limits", "chunk_policy", "qualification")
    return {"schema": REPORT_SCHEMA, "manifest_sha256": manifest_digest,
            **{key: manifest[key] for key in keys}}


@_closed_errors
def hydrate_sharded_codebase_ir_metadata(*, records, output, source_snapshot):
    """Hydrate every bounded part, with native per-part fresh readback unchanged."""
    output = native._path(output)
    _need(not output.exists() and output.parent.is_dir(), "fresh sharded namespace required")
    _need(type(source_snapshot) is dict and bool(source_snapshot), "source snapshot object required")
    source_raw = native._json(source_snapshot)
    _need(len(source_raw) <= native.LIMITS["source_bytes"], "unchanged source snapshot bound exceeded")
    plan = partition_codebase_ir_metadata_records(records=records)
    _need("producer_sha256" not in source_snapshot
          or source_snapshot["producer_sha256"] == plan["producer_sha256"],
          "source snapshot producer attribution differs from complete records")
    manifest = {"schema": SCHEMA, "output": str(output), "source_snapshot": json.loads(source_raw),
                "source_snapshot_sha256": native._digest(source_raw), "input_families": plan["input_families"],
                "limits": dict(LIMITS), "native_limits": dict(native.LIMITS), "chunk_policy": dict(CHUNK_POLICY),
                "packaged_sha256": plan["packaged_sha256"], "producer_sha256": plan["producer_sha256"],
                "row_count": plan["row_count"], "payload_bytes": plan["payload_bytes"],
                "family_counts": plan["family_counts"], "family_digests": {
                    name: _family_digest(rows) for name, rows in plan["records"].items()},
                "row_root_sha256": _global_row_root(plan["records"], native._digest(source_raw)),
                "qualification": dict(QUALIFICATION)}
    # Refuse all known assembly bounds before creating a namespace or native job.
    for part in plan["parts"]:
        _need(len(native._json(_part_source(manifest, part))) <= native.LIMITS["source_bytes"],
              "part source context exceeds unchanged native source bound")
    for family, rows in plan["records"].items():
        size = sum(len(native._wire(native._row(family, row, ordinal, manifest["source_snapshot_sha256"]))) + 1
                   for ordinal, row in enumerate(rows))
        _need(size <= LIMITS["export_bytes"], "global family export bound exceeded")
    output.mkdir(mode=0o700)
    with native._owner(output, create=True):
        (output / "parts").mkdir(mode=0o700)
        (output / "exports").mkdir(mode=0o700)
        native._write(output / "source-snapshot.json", source_raw)
        manifest["source_snapshot_artifact"] = _descriptor(output / "source-snapshot.json", "source-snapshot.json",
                                                          native.LIMITS["source_bytes"])
        parts, fresh, recovered = [], [], {name: [] for name in plan["records"]}
        for part in plan["parts"]:
            relative = "parts/" + f"{part['ordinal']:06d}"
            source = _part_source(manifest, part)
            result = native.hydrate_codebase_ir_metadata(records=part["records"], output=output / relative,
                                                        source_snapshot=source)
            _need(result.get("fresh_process_readback", {}).get("verified") is True,
                  "native part fresh process did not verify")
            stable = _stable_native_report(result)
            _need(stable["source_snapshot_sha256"] == native._digest(native._json(source)),
                  "native part source context differs")
            payloads = _part_payloads(output / relative, stable, sorted(plan["records"]), stable["source_snapshot_sha256"])
            _need(_same(payloads, part["records"]), "native hydration changed a packaged row")
            for family in recovered:
                recovered[family].extend(payloads[family])
            descriptor = {key: part[key] for key in ("ordinal", "row_count", "payload_bytes", "global_family_ranges")}
            descriptor.update(relative_path=relative, source_snapshot_sha256=stable["source_snapshot_sha256"],
                native_manifest_sha256=stable["manifest_sha256"], native_report=stable,
                native_manifest=_descriptor(output / relative / "manifest.json", relative + "/manifest.json",
                                            native.LIMITS["manifest_bytes"]))
            _need(descriptor["native_manifest"]["sha256"] == stable["manifest_sha256"], "native manifest binding differs")
            parts.append(descriptor)
            fresh.append({"ordinal": part["ordinal"], "readback": result["fresh_process_readback"]})
        _need(_same(recovered, plan["records"]), "native part assembly lost or reordered rows")
        manifest["parts"] = parts
        manifest["exports"] = _write_global_exports(output, recovered, manifest["source_snapshot_sha256"])
        raw = native._wire(manifest)
        _need(len(raw) <= LIMITS["manifest_bytes"], "global manifest byte bound exceeded")
        native._write(output / "manifest.json", raw)
        _artifact_fence(output, manifest, raw)
    result = _report(manifest, native._digest(raw))
    result["fresh_process_readback"] = {"verified": True, "method": "each_native_part_new_python_process_readback",
                                       "parts": fresh, "global_new_process_verified": False}
    return result


def _artifact_fence(output, manifest, original_raw):
    _need(native._read(output / "manifest.json", LIMITS["manifest_bytes"]) == original_raw,
          "global manifest changed during verification")
    source = _check_descriptor(output, manifest["source_snapshot_artifact"], native.LIMITS["source_bytes"],
                               "source-snapshot.json")
    _need(source == native._json(manifest["source_snapshot"]), "physical global source snapshot differs")
    for part in manifest["parts"]:
        _check_descriptor(output, part["native_manifest"], native.LIMITS["manifest_bytes"],
                          part["relative_path"] + "/manifest.json")
        for family, descriptor in part["native_report"]["exports"].items():
            _check_descriptor(output / part["relative_path"], descriptor,
                              2 * native.LIMITS["total_payload_bytes"], "exports/" + family + ".jsonl")
    for family, descriptor in manifest["exports"].items():
        _check_descriptor(output, descriptor, LIMITS["export_bytes"], "exports/" + family + ".jsonl")


def _checked(output, fresh_process):
    raw = native._read(output / "manifest.json", LIMITS["manifest_bytes"])
    manifest = native._loads(raw)
    fields = {"schema", "output", "source_snapshot", "source_snapshot_sha256", "source_snapshot_artifact",
              "input_families", "limits", "native_limits", "chunk_policy", "packaged_sha256", "producer_sha256",
              "row_count", "payload_bytes", "family_counts", "family_digests", "row_root_sha256", "qualification",
              "parts", "exports"}
    _need(type(manifest) is dict and set(manifest) == fields and manifest["schema"] == SCHEMA
          and native._wire(manifest) == raw and manifest["output"] == str(output)
          and _same(manifest["limits"], LIMITS) and _same(manifest["native_limits"], native.LIMITS)
          and _same(manifest["chunk_policy"], CHUNK_POLICY) and _same(manifest["qualification"], QUALIFICATION),
          "closed sharded manifest or namespace differs")
    _need(type(manifest["input_families"]) is list
          and all(type(name) is str and native._FAMILY.fullmatch(name) for name in manifest["input_families"])
          and manifest["input_families"] == sorted(set(manifest["input_families"])),
          "original family keys differ")
    names = sorted(set(manifest["input_families"]) | set(native.DEFAULT_FAMILIES))
    _need(len(names) <= LIMITS["families"]
          and all(type(manifest[key]) is dict for key in ("family_counts", "family_digests", "exports"))
          and set(manifest["family_counts"]) == set(names)
          and set(manifest["family_digests"]) == set(names) and set(manifest["exports"]) == set(names),
          "global family population differs")
    _need(type(manifest["source_snapshot"]) is dict and bool(manifest["source_snapshot"])
          and native._digest(native._json(manifest["source_snapshot"])) == manifest["source_snapshot_sha256"],
          "global source snapshot identity differs")
    _need(len(native._json(manifest["source_snapshot"])) <= native.LIMITS["source_bytes"],
          "unchanged source snapshot bound exceeded")
    _need(type(manifest["parts"]) is list and 1 <= len(manifest["parts"]) <= LIMITS["parts"], "global part population differs")
    directories = sorted(path.name for path in (output / "parts").iterdir())
    _need(directories == [f"{ordinal:06d}" for ordinal in range(len(manifest["parts"]))],
          "missing, duplicate or unaccounted part namespace")
    recovered, positions, fresh = {name: [] for name in names}, {name: 0 for name in names}, []
    part_fields = {"ordinal", "relative_path", "row_count", "payload_bytes", "global_family_ranges",
                   "source_snapshot_sha256", "native_manifest_sha256", "native_manifest", "native_report"}
    for ordinal, part in enumerate(manifest["parts"]):
        _need(type(part) is dict and set(part) == part_fields and type(part["ordinal"]) is int
              and part["ordinal"] == ordinal and part["relative_path"] == "parts/" + f"{ordinal:06d}"
              and type(part["row_count"]) is int and 0 <= part["row_count"] <= LIMITS["part_rows"]
              and type(part["payload_bytes"]) is int and 0 <= part["payload_bytes"] <= LIMITS["part_payload_bytes"]
              and type(part["native_report"]) is dict and "manifest_sha256" in part["native_report"]
              and type(part["global_family_ranges"]) is dict and set(part["global_family_ranges"]) == set(names),
              "closed bounded ordered part descriptor required")
        for family in names:
            value = part["global_family_ranges"][family]
            _need(type(value) is dict and set(value) == {"start", "stop", "count"}
                  and all(type(value[key]) is int for key in value)
                  and value["start"] == positions[family] and value["stop"] >= value["start"]
                  and value["count"] == value["stop"] - value["start"], "global family ordinal span differs")
        part_raw = _check_descriptor(output, part["native_manifest"], native.LIMITS["manifest_bytes"],
                                     part["relative_path"] + "/manifest.json")
        _need(native._digest(part_raw) == part["native_manifest_sha256"] == part["native_report"]["manifest_sha256"],
              "native part manifest identity differs")
        source = _part_source(manifest, part)
        part_manifest = native._loads(part_raw)
        _need(type(part_manifest) is dict and "source_snapshot" in part_manifest
              and _same(part_manifest["source_snapshot"], source)
              and native._digest(native._json(source)) == part["source_snapshot_sha256"], "native part full source context differs")
        report = native.validate_codebase_ir_metadata(output=output / part["relative_path"],
                    expected=part["native_report"], fresh_process=fresh_process)
        _need(_same(_stable_native_report(report), part["native_report"])
              and report["row_count"] == part["row_count"], "native part report differs")
        if fresh_process:
            _need(report.get("fresh_process_readback", {}).get("verified") is True, "native part fresh readback failed")
            fresh.append({"ordinal": ordinal, "readback": report["fresh_process_readback"]})
        payloads = _part_payloads(output / part["relative_path"], report, names, part["source_snapshot_sha256"])
        _need(sum(map(len, payloads.values())) == part["row_count"]
              and sum(len(native._json(row)) for values in payloads.values() for row in values) == part["payload_bytes"],
              "part payload population or byte count differs")
        for family in names:
            _need(len(payloads[family]) == part["global_family_ranges"][family]["count"], "native part span count differs")
            recovered[family].extend(payloads[family])
            positions[family] += len(payloads[family])
    original = {name: recovered[name] for name in manifest["input_families"]}
    plan = partition_codebase_ir_metadata_records(records=original)
    _need("producer_sha256" not in manifest["source_snapshot"]
          or manifest["source_snapshot"]["producer_sha256"] == plan["producer_sha256"],
          "source snapshot producer attribution differs from complete records")
    _need(_same(plan["family_counts"], manifest["family_counts"])
          and plan["row_count"] == manifest["row_count"] and type(manifest["row_count"]) is int
          and plan["payload_bytes"] == manifest["payload_bytes"] and type(manifest["payload_bytes"]) is int
          and plan["packaged_sha256"] == manifest["packaged_sha256"]
          and plan["producer_sha256"] == manifest["producer_sha256"]
          and len(plan["parts"]) == len(manifest["parts"]), "complete global packaged or producer identity differs")
    for planned, observed in zip(plan["parts"], manifest["parts"]):
        _need(_same(planned["global_family_ranges"], observed["global_family_ranges"])
              and planned["row_count"] == observed["row_count"] and planned["payload_bytes"] == observed["payload_bytes"],
              "part boundaries differ from deterministic partition")
    for family in names:
        _need(_family_digest(recovered[family]) == manifest["family_digests"][family], "global full family digest differs")
        export = _check_descriptor(output, manifest["exports"][family], LIMITS["export_bytes"], "exports/" + family + ".jsonl")
        expected = b"".join(native._wire(native._row(family, row, ordinal, manifest["source_snapshot_sha256"])) + b"\n"
                            for ordinal, row in enumerate(recovered[family]))
        _need(export == expected, "global assembly export omits, reorders or changes a payload")
    _need(_global_row_root(recovered, manifest["source_snapshot_sha256"]) == manifest["row_root_sha256"],
          "global row root differs")
    _artifact_fence(output, manifest, raw)
    return _report(manifest, native._digest(raw)), original, manifest, raw, fresh


def _fresh(output, expected):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(str(Path(value or os.getcwd()).resolve()) for value in sys.path)
    process = subprocess.run([sys.executable, "-B", "-m", MODULE, "--validate", str(output),
                             "--manifest-sha256", expected["manifest_sha256"]], env=environment,
                            capture_output=True, text=True, timeout=LIMITS["restart_seconds"])
    _need(process.returncode == 0, "fresh global native-part validation failed: " + process.stderr[-2000:])
    try:
        observed = native._loads(process.stdout)
    except (ValueError, TypeError) as error:
        raise ShardedMetadataError("fresh global process returned no complete report") from error
    _need(_same(observed, expected), "fresh global process report differs")
    return {"verified": True, "method": "new_python_process_native_part_validation_and_global_reassembly",
            "manifest_sha256": expected["manifest_sha256"], "packaged_sha256": expected["packaged_sha256"],
            "producer_sha256": expected["producer_sha256"], "row_count": expected["row_count"],
            "global_new_process_verified": True}


@_closed_errors
def validate_sharded_codebase_ir_metadata(*, output, expected=None, fresh_process=True):
    """Validate every original part, global span, export and producer reconstruction."""
    output = native._path(output)
    _need(type(fresh_process) is bool, "fresh_process must be a boolean")
    with native._owner(output):
        report, records, manifest, raw, fresh = _checked(output, fresh_process)
    if expected is not None:
        _need(type(expected) is dict and "manifest_sha256" in expected, "expected global manifest identity required")
        for key, value in expected.items():
            if key != "fresh_process_readback":
                _need(key in report and _same(report[key], value), "expected global metadata binding differs: " + key)
    if fresh_process:
        global_fresh = _fresh(output, report)
        with native._owner(output):
            _artifact_fence(output, manifest, raw)
        report["fresh_process_readback"] = {**global_fresh, "parts": fresh}
    return report


@_closed_errors
def reconstruct_sharded_codebase_ir_metadata_records(*, output, expected=None):
    """Return exact original packaged families after actual native validation."""
    output = native._path(output)
    with native._owner(output):
        report, records, manifest, raw, _ = _checked(output, False)
    if expected is not None:
        _need(type(expected) is dict and "manifest_sha256" in expected, "expected reconstruction manifest identity required")
        for key, value in expected.items():
            if key != "fresh_process_readback":
                _need(key in report and _same(report[key], value), "expected reconstruction binding differs: " + key)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    arguments = parser.parse_args()
    print(json.dumps(validate_sharded_codebase_ir_metadata(output=arguments.validate,
        expected={"manifest_sha256": arguments.manifest_sha256}, fresh_process=False), sort_keys=True))
