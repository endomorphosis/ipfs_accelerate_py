"""Bounded, independent read-only joins for a closed native source-delta run.

Only standard-library files/JSON/hashes are used. No product module, native
owner, SQL connection, model, solver, Git command, or Docker daemon is opened.
A removed member is absent from a complete admitted capture; it need not be
physically absent. Native execution/resource reports are retained observations,
not process-origin, current deployment, or semantic correctness attestations.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time

SCHEMA = "codebase-source-delta-independent-audit@1"
DELTA_SCHEMA = "codebase-inventory-source-delta@1"
MIB = 1024 ** 2
MAX_FILES = 65_536
MAX_ARCHIVE_BYTES = 4 * 1024 ** 3
MAX_FILE_BYTES = 1024 ** 3
MAX_TOTAL_READ_BYTES = 512 * MIB
AUTHORITY = frozenset(("source_semantics_verified", "runtime_behavior_verified", "proof_authority",
    "execution_authority", "completion_authority", "mutation_authority", "admission_authority",
    "authoritative_cache_eligible", "behavioral_satisfaction", "training_executed",
    "decoded_formulas_generated", "repository_code_executed", "source_execution_attested",
    "scan_execution_attested"))
CLASSES = ("retained", "changed", "added", "removed")
COMPARISONS = ("equal", "different", "unavailable")
ENTRY_FIELDS = {"schema", "path", "raw_path_hex", "kind", "size_bytes", "source_cid", "opaque_reason",
    "git_blob_oid", "acquisition", "disposition", "head_blob_oid", "index_blob_oids", "entry_cid"}
HEAD_FIELDS = {"schema", "repository_id", "generation", "manifest_cid", "snapshot_cid", "ast_revision_id", "receipt_cid"}
TABLE_NAMES = {"source_revisions", "source_files", "ast_blobs", "ast_nodes", "scopes", "symbols",
    "imports", "references", "calls", "effects", "interfaces", "diagnostics", "invalidations"}


class SourceDeltaAuditError(ValueError):
    """A retained artifact, complete join, or closed audit condition differs."""


def need(condition, message):
    if not condition:
        raise SourceDeltaAuditError(message)


def closed(value, fields, name):
    need(type(value) is dict and set(value) == set(fields), "closed " + name + " fields required")


def integer(value, ceiling=2**63 - 1, minimum=0):
    need(type(value) is int and minimum <= value <= ceiling, "exact bounded integer required")
    return value


def text(value, maximum=8192):
    need(type(value) is str and 0 < len(value.encode("utf-8")) <= maximum, "bounded exact text required")
    return value


def strict(value, depth=0):
    need(depth <= 96, "structured nesting exceeds bound")
    if type(value) is dict:
        need(all(type(key) is str for key in value), "structured keys must be exact strings")
        for item in value.values():
            strict(item, depth + 1)
    elif type(value) is list:
        for item in value:
            strict(item, depth + 1)
    else:
        need(type(value) in {str, int, bool, type(None)}, "structured identity rejects unreviewed scalar")


def wire(value):
    strict(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def cid(raw, *, source=False):
    need(type(raw) is bytes, "exact CID bytes required")
    prefix = b"\x01\x55" if source else b"\x01\xa9\x02"
    return "b" + base64.b32encode(prefix + b"\x12\x20" + hashlib.sha256(raw).digest()).decode().lower().rstrip("=")


def structured(value):
    return cid(wire(value))


def check_cid(value, *, source=False):
    text(value, 80)
    try:
        decoded = base64.b32decode(value[1:].upper() + "=" * (-len(value[1:]) % 8))
    except Exception as error:
        raise SourceDeltaAuditError("canonical CID required") from error
    prefix = b"\x01\x55\x12\x20" if source else b"\x01\xa9\x02\x12\x20"
    need(value.startswith("b") and len(decoded) == len(prefix) + 32 and decoded.startswith(prefix)
         and "b" + base64.b32encode(decoded).decode().lower().rstrip("=") == value, "exact canonical CID profile required")
    return value


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON member")
            result[key] = value
        return result
    def constant(value):
        raise SourceDeltaAuditError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def same(left, right):
    return wire(left) == wire(right)


def observation_same(left, right):
    """Compare native telemetry JSON; it is outside CID identity encoding."""
    def checked(value, depth=0):
        need(depth <= 96, "observation nesting exceeds bound")
        if type(value) is dict:
            need(all(type(key) is str for key in value), "observation keys must be strings")
            for item in value.values():
                checked(item, depth + 1)
        elif type(value) is list:
            for item in value:
                checked(item, depth + 1)
        elif type(value) is float:
            need(math.isfinite(value), "observation float must be finite")
        else:
            need(type(value) in {str, int, bool, type(None)}, "unreviewed observation scalar")
    checked(left); checked(right)
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() == json.dumps(right, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def false_authority(value):
    need(type(value) is dict and set(value) == AUTHORITY and all(flag is False for flag in value.values()),
         "authority must remain exactly false")


def raw_display(raw):
    try:
        value = raw.decode("utf-8", "strict")
        need(value and value == value.strip() and not value.startswith("/")
             and not any(part in {"", ".", ".."} for part in value.split("/"))
             and not any(ord(character) < 32 for character in value), "unsafe raw path")
        return value
    except (UnicodeError, SourceDeltaAuditError):
        return "@malformed-path/" + raw.hex()


def entry_shape(entry):
    closed(entry, ENTRY_FIELDS, "snapshot entry")
    need(entry["schema"] == "ipfs-datasets.software-contracts.semantic-snapshot-entry@3", "entry schema differs")
    raw = entry["raw_path_hex"]
    need(type(raw) is str and 0 < len(raw) <= 8192 and re.fullmatch(r"(?:[0-9a-f]{2})+", raw), "exact raw path identity required")
    need(entry["path"] == raw_display(bytes.fromhex(raw)), "raw path display differs")
    text(entry["kind"])
    if entry["size_bytes"] is not None:
        integer(entry["size_bytes"])
    if entry["source_cid"] is not None:
        check_cid(entry["source_cid"], source=True)
    opaque = entry["opaque_reason"] is not None
    if opaque:
        text(entry["opaque_reason"], 512)
        need(entry["kind"] == "opaque", "opaque source kind differs")
    else:
        need(entry["source_cid"] is not None and entry["size_bytes"] is not None, "captured source identity required")
    for field in ("git_blob_oid", "head_blob_oid"):
        need(entry[field] is None or (type(entry[field]) is str and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", entry[field])), "Git blob locator differs")
    need(type(entry["index_blob_oids"]) is dict and set(entry["index_blob_oids"]) <= {"0", "1", "2", "3"}
         and all(type(value) is str and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) for value in entry["index_blob_oids"].values()), "index blob locators differ")
    need(entry["acquisition"] in {"captured", "git-object", "working-captured", "opaque"}, "source acquisition differs")
    need(entry["disposition"] in {"clean", "working", "tracked_modified", "staged_added", "staged_modified", "staged_deleted", "unstaged_deleted", "untracked", "conflicted", "unborn", "filesystem", "opaque"}, "source disposition differs")
    check_cid(entry["entry_cid"])
    need(entry["entry_cid"] == structured({key: value for key, value in entry.items() if key != "entry_cid"}), "snapshot entry CID differs")


def head_shape(head):
    closed(head, HEAD_FIELDS, "native head")
    need(head["schema"] == "codebase-head@1", "head schema differs")
    text(head["repository_id"], 512); integer(head["generation"], minimum=1)
    for field in ("manifest_cid", "snapshot_cid", "receipt_cid"):
        check_cid(head[field])
    need(head["ast_revision_id"] == "rev:" + head["repository_id"] + ":snapshot:" + head["snapshot_cid"], "head AST revision differs")


def receipt_head(receipt):
    closed(receipt, {"schema", "operation_id", "request_cid", "previous_head", "repository_id", "generation", "manifest_cid", "snapshot_cid", "ast_revision_id"}, "native publication receipt")
    need(receipt["schema"] == "codebase-publication-receipt@1" and len(wire(receipt)) <= 64 * 1024, "bounded publication receipt required")
    text(receipt["operation_id"], 256); text(receipt["repository_id"], 512)
    previous = receipt["previous_head"]
    if previous is not None:
        head_shape(previous)
        need(previous["repository_id"] == receipt["repository_id"], "publication repository changed")
    integer(receipt["generation"], minimum=1)
    need(receipt["generation"] == (1 if previous is None else previous["generation"] + 1), "publication generation differs")
    need(receipt["request_cid"] == structured({"schema": "codebase-publication-request@1", "repository_id": receipt["repository_id"], "manifest_cid": receipt["manifest_cid"], "expected_head": previous}), "publication request identity differs")
    head = {key: receipt[key] for key in HEAD_FIELDS - {"schema", "receipt_cid"}}
    head.update(schema="codebase-head@1", receipt_cid=structured(receipt))
    head_shape(head)
    return head


def manifest_members(manifest, head, *, get_artifact):
    """Join every complete native entry/unit and admitted source/AST CAS."""
    closed(manifest, {"schema", "authority", "snapshot", "semantic_state", "ast_revision_id", "units", "coverage"}, "native structural manifest")
    need(manifest["schema"] == "codebase-ir-structural-manifest@1" and manifest["authority"] == "structural_only"
         and len(wire(manifest)) <= 4 * MIB and structured(manifest) == head["manifest_cid"], "complete head/manifest binding differs")
    snapshot = manifest["snapshot"]
    closed(snapshot, {"schema", "repository_id", "entries", "mode", "max_file_bytes", "max_entries", "git_tree", "git_commit", "exclusions", "snapshot_cid"}, "repository snapshot")
    need(snapshot["schema"] == "ipfs-datasets.software-contracts.semantic-repository-snapshot@4"
         and snapshot["repository_id"] == head["repository_id"] and snapshot["snapshot_cid"] == head["snapshot_cid"]
         and structured({key: value for key, value in snapshot.items() if key != "snapshot_cid"}) == head["snapshot_cid"]
         and manifest["ast_revision_id"] == head["ast_revision_id"], "complete snapshot/head identity differs")
    integer(snapshot["max_entries"], 1024, 1); integer(snapshot["max_file_bytes"], 65536, 1)
    need(snapshot["mode"] in {"git-clean", "git-working", "git-unborn", "filesystem"}, "snapshot capture mode differs")
    for field in ("git_tree", "git_commit"):
        need(snapshot[field] is None or (type(snapshot[field]) is str and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", snapshot[field])), "snapshot Git identity differs")
    if snapshot["mode"] in {"git-clean", "git-working"}:
        need(snapshot["git_tree"] is not None and snapshot["git_commit"] is not None, "committed capture identity absent")
    exclusions = snapshot["exclusions"]
    need(type(exclusions) is list and len(exclusions) <= 1024 and all(type(value) is str and 0 < len(value.encode()) <= 8192 for value in exclusions)
         and exclusions == sorted(set(exclusions)), "capture exclusions differ")
    entries, units = snapshot["entries"], manifest["units"]
    need(type(entries) is list and len(entries) <= snapshot["max_entries"] and type(units) is list and len(units) == len(entries), "complete entry/unit population required")
    for entry in entries:
        entry_shape(entry)
    keys = ["raw:" + entry["raw_path_hex"] for entry in entries]
    need(keys == sorted(set(keys)), "complete inventory raw paths duplicated/reordered")
    by_key = dict(zip(keys, entries))
    unit_keys, members, source_bytes, ast_bytes = [], [], 0, 0
    for unit in units:
        closed(unit, {"schema", "source_key", "entry_cid", "ast_cid", "parse_status"}, "native unit")
        need(unit["schema"] == "codebase-ir-structural-unit@1" and unit["source_key"] in by_key, "native unit source differs")
        unit_keys.append(unit["source_key"])
        entry = by_key[unit["source_key"]]
        need(unit["entry_cid"] == entry["entry_cid"] and unit["parse_status"] in {"ok", "partial", "failed", "opaque", "unindexed"}, "native unit/entry/status differs")
        opaque = entry["opaque_reason"] is not None
        need((unit["parse_status"] == "opaque") == opaque and (unit["ast_cid"] is None) == (unit["parse_status"] in {"opaque", "unindexed"}), "native AST opacity differs")
        if not opaque:
            raw = get_artifact(entry["source_cid"], source=True)
            need(type(raw) is bytes and len(raw) == entry["size_bytes"] <= snapshot["max_file_bytes"] and cid(raw, source=True) == entry["source_cid"], "captured raw source CAS differs")
            source_bytes += len(raw)
            need(source_bytes <= 64 * MIB, "aggregate source payload exceeds bound")
        if unit["ast_cid"] is not None:
            check_cid(unit["ast_cid"])
            raw = get_artifact(unit["ast_cid"], source=False)
            need(type(raw) is bytes and len(raw) <= 4 * MIB and cid(raw) == unit["ast_cid"], "native AST CAS differs")
            ast_bytes += len(raw)
            need(ast_bytes <= 128 * MIB, "aggregate AST payload exceeds bound")
            ast = parse(raw)
            need(raw == wire(ast), "AST canonical bytes differ")
            verify_ast_binding(ast, entry, unit, head)
        members.append({"source_key": unit["source_key"], "path": entry["path"], "raw_path_hex": entry["raw_path_hex"],
            "entry_cid": entry["entry_cid"], "source_cid": entry["source_cid"], "ast_cid": unit["ast_cid"],
            "parse_status": unit["parse_status"], "source_size_bytes": entry["size_bytes"], "opaque_reason": entry["opaque_reason"]})
    need(unit_keys == keys, "native units omitted/duplicated/reordered")
    state = manifest["semantic_state"]
    closed(state, {"schema", "repository_id", "symbols", "artifacts", "edges", "extractor_name", "extractor_version", "state_cid"}, "semantic repository state")
    need(state["schema"] == "ipfs-datasets.software-contracts.semantic-index@2" and state["repository_id"] == head["repository_id"], "semantic repository state differs")
    for field in ("symbols", "artifacts", "edges"):
        need(type(state[field]) is list and len(state[field]) <= 65536, "bounded semantic collections required")
    state_identity = {key: value for key, value in state.items() if key not in {"schema", "state_cid"}}
    state_identity.update(schema="ipfs-datasets.software-contracts.semantic-repository-state@1", semantic_index_schema=state["schema"])
    need(state["state_cid"] == structured(state_identity), "semantic repository state CID differs")
    evidence = [row for row in state["artifacts"] if type(row) is dict and row.get("artifact_id") == "artifact:snapshot-evidence"]
    need(len(evidence) == 1 and evidence[0].get("source_cid") == head["snapshot_cid"] and same(evidence[0].get("metadata", {}).get("snapshot"), snapshot), "semantic snapshot evidence differs")
    for symbol in state["symbols"]:
        need(type(symbol) is dict and symbol.get("repository_id") == head["repository_id"], "semantic symbol repository differs")
        matching = [entry for entry in entries if entry["path"] == symbol.get("module_path")]
        need(len(matching) == 1 and matching[0]["opaque_reason"] is None and symbol.get("source_cid") == matching[0]["source_cid"], "semantic symbol/captured source differs")
    coverage = {"inventory_entries": len(entries), "captured_entries": sum(entry["opaque_reason"] is None for entry in entries),
        "ast_ok": sum(unit["parse_status"] == "ok" for unit in units), "ast_partial": sum(unit["parse_status"] == "partial" for unit in units),
        "ast_failed": sum(unit["parse_status"] == "failed" for unit in units), "opaque_entries": sum(unit["parse_status"] == "opaque" for unit in units),
        "unindexed_entries": sum(unit["parse_status"] == "unindexed" for unit in units), "semantic_symbols": len(state["symbols"]),
        "formalized_properties": 0, "checked_properties": 0}
    need(same(manifest["coverage"], coverage), "complete manifest coverage differs")
    return {"members": members, "entries": by_key, "source_bytes": source_bytes, "ast_bytes": ast_bytes,
            "capture_policy": {field: snapshot[field] for field in ("max_entries", "max_file_bytes", "exclusions")}}


def verify_ast_binding(ast, entry, unit, head):
    if ast.get("kind") == "parse_failure":
        closed(ast, {"schema", "kind", "source_cid", "path", "repository_id", "revision", "code", "message", "language"}, "parse failure")
        need(ast["schema"] == "duckdb-ast-store/v1" and unit["parse_status"] == "failed", "parse failure status differs")
        wanted = {"source_cid": entry["source_cid"], "path": entry["path"], "repository_id": head["repository_id"], "revision": "snapshot:" + head["snapshot_cid"]}
        need(all(ast[field] == value for field, value in wanted.items()), "parse failure source provenance differs")
        for field in ("code", "message", "language"):
            text(ast[field])
        return
    closed(ast, {"schema", "provenance", "frontend", "module", "scopes", "symbols", "imports", "references", "calls", "effects", "diagnostics", "unsupported"}, "native AST record")
    need(ast["schema"] == "ipfs-datasets.software-contracts.ast-ir@1.0.0", "AST schema differs")
    wanted = {"source_cid": entry["source_cid"], "path": entry["path"], "repository_id": head["repository_id"],
        "revision": "snapshot:" + head["snapshot_cid"], "repository_tree_cid": head["snapshot_cid"]}
    need(same(ast["provenance"], wanted), "native AST/source/snapshot provenance differs")
    for field in ("scopes", "symbols", "imports", "references", "calls", "effects", "diagnostics", "unsupported"):
        need(type(ast[field]) is list and len(ast[field]) <= 65536, "bounded native AST collections required")
    diagnostics = ast["diagnostics"]
    for row in diagnostics:
        need(type(row) is dict and type(row.get("code")) is str and row.get("severity") in {"info", "warning", "error", "fatal"}, "AST diagnostic classification differs")
    failed = any(row["severity"] == "fatal" or row["code"] == "ast.parse_failure" or row["code"].endswith((".parse_error", ".invalid_encoding", ".resource_limit")) for row in diagnostics)
    status = "failed" if failed else "partial" if any(row["severity"] == "error" for row in diagnostics) else "ok"
    need(status == unit["parse_status"], "native AST parse classification differs")


def delta_ledger(previous, current):
    old = {member["source_key"]: {"entry": previous["entries"][member["source_key"]], "member": member} for member in previous["members"]}
    new = {member["source_key"]: {"entry": current["entries"][member["source_key"]], "member": member} for member in current["members"]}
    rows = []
    for key in sorted(set(old) | set(new)):
        left, right = old.get(key), new.get(key)
        classification = "added" if left is None else "removed" if right is None else "retained" if left["entry"]["entry_cid"] == right["entry"]["entry_cid"] else "changed"
        source_comparison = ast_comparison = "unavailable"
        if left is not None and right is not None:
            if left["entry"]["opaque_reason"] is None and right["entry"]["opaque_reason"] is None:
                source_comparison = "equal" if (left["entry"]["source_cid"], left["entry"]["size_bytes"]) == (right["entry"]["source_cid"], right["entry"]["size_bytes"]) else "different"
            if left["member"]["ast_cid"] is not None and right["member"]["ast_cid"] is not None:
                ast_comparison = "equal" if left["member"]["ast_cid"] == right["member"]["ast_cid"] else "different"
        rows.append({"source_key": key, "classification": classification, "previous": left, "current": right,
                     "source_bytes_comparison": source_comparison, "ast_identity_comparison": ast_comparison})
    return rows


def delta_coverage(ledger):
    return {"previous_entries": sum(row["previous"] is not None for row in ledger),
        "current_entries": sum(row["current"] is not None for row in ledger), "union_entries": len(ledger),
        "classifications": {key: sum(row["classification"] == key for row in ledger) for key in CLASSES},
        "source_bytes_comparisons": {key: sum(row["source_bytes_comparison"] == key for row in ledger) for key in COMPARISONS},
        "ast_identity_comparisons": {key: sum(row["ast_identity_comparison"] == key for row in ledger) for key in COMPARISONS}}


def verify_source_delta(envelope, previous_manifest, current_manifest, previous_receipt, current_receipt, *, get_artifact):
    closed(envelope, {"artifact_cid", "value"}, "delta export")
    value = envelope["value"]
    need(type(value) is dict and len(wire(value)) <= 8 * MIB and envelope["artifact_cid"] == structured(value), "delta export CID/byte bound differs")
    closed(value, {"schema", "previous_head", "current_head", "previous_publication_receipt", "current_publication_receipt", "previous_membership_cid", "current_membership_cid", "capture_policy", "ledger", "coverage", "limits", "optimized", "implementation", "authority", "numerical_reuse", "model_advanced", "removal_scope", "physical_absence_verified"}, "source delta")
    need(value["schema"] == DELTA_SCHEMA and type(value["optimized"]) is bool, "delta schema/opt-out differs")
    need(value["numerical_reuse"] is False and value["model_advanced"] is False and value["physical_absence_verified"] is False
         and value["removal_scope"] == "absent_from_current_complete_capture", "delta cannot claim numerical or physical absence authority")
    false_authority(value["authority"])
    limits = value["limits"]
    closed(limits, {"max_inventory_entries", "max_union_entries", "max_file_bytes", "max_manifest_bytes", "max_delta_bytes"}, "delta limits")
    for field in limits:
        integer(limits[field], {"max_inventory_entries": 1024, "max_union_entries": 2048, "max_file_bytes": 65536, "max_manifest_bytes": 4 * MIB, "max_delta_bytes": 8 * MIB}[field], 1)
    need(len(wire(value)) <= limits["max_delta_bytes"], "declared delta byte bound exceeded")
    previous_head, current_head = receipt_head(previous_receipt), receipt_head(current_receipt)
    need(same(current_receipt["previous_head"], previous_head) and current_head["repository_id"] == previous_head["repository_id"], "immediate same-repository publication required")
    for key, expected in (("previous_head", previous_head), ("current_head", current_head), ("previous_publication_receipt", previous_receipt), ("current_publication_receipt", current_receipt)):
        need(same(value[key], expected), "delta publication receipt/head differs: " + key)
    previous = manifest_members(previous_manifest, previous_head, get_artifact=get_artifact)
    current = manifest_members(current_manifest, current_head, get_artifact=get_artifact)
    need(same(previous["capture_policy"], current["capture_policy"]) and same(value["capture_policy"], current["capture_policy"]), "complete captures have different policies")
    need(current["capture_policy"]["max_entries"] <= limits["max_inventory_entries"] and current["capture_policy"]["max_file_bytes"] <= limits["max_file_bytes"]
         and len(wire(previous_manifest)) <= limits["max_manifest_bytes"] and len(wire(current_manifest)) <= limits["max_manifest_bytes"], "capture exceeds declared delta limits")
    ledger = delta_ledger(previous, current)
    need(len(ledger) <= limits["max_union_entries"] and same(value["ledger"], ledger), "complete ordered source union/classifications differ")
    coverage = delta_coverage(ledger)
    need(same(value["coverage"], coverage), "complete delta coverage differs")
    need(value["previous_membership_cid"] == structured(previous["members"]) and value["current_membership_cid"] == structured(current["members"]), "complete native membership CID differs")
    implementation = value["implementation"]
    closed(implementation, {"files", "sha256", "scope"}, "delta implementation")
    need(type(implementation["files"]) is dict and 1 <= len(implementation["files"]) <= 32
         and implementation["scope"] == "listed_local_files_only_not_execution_attestation"
         and implementation["sha256"] == sha(json.dumps(implementation["files"], ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()), "selected implementation digest differs")
    for name, digest in implementation["files"].items():
        text(name, 256)
        need(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest), "producer SHA differs")
    return {"delta_cid": envelope["artifact_cid"], "previous_head": previous_head, "current_head": current_head,
        "previous_membership_cid": value["previous_membership_cid"], "current_membership_cid": value["current_membership_cid"],
        "coverage": coverage, "source_bytes_joined": previous["source_bytes"] + current["source_bytes"],
        "ast_bytes_joined": previous["ast_bytes"] + current["ast_bytes"], "physical_absence_verified": False,
        "implementation": implementation,
        "artifact_closure": {"manifest_cids": sorted({previous_head["manifest_cid"], current_head["manifest_cid"]}),
            "source_cids": sorted({member["source_cid"] for side in (previous, current) for member in side["members"] if member["opaque_reason"] is None}),
            "ast_cids": sorted({member["ast_cid"] for side in (previous, current) for member in side["members"] if member["ast_cid"] is not None})},
        "model_owner_opened": False, "numerical_reuse": False, "source_execution_attested": False,
        "native_ast_schema_validation_here": False, "authority": {key: False for key in sorted(AUTHORITY)}}


def identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, stat.S_IMODE(info.st_mode), info.st_nlink)


class Reader:
    def __init__(self, namespace, seconds=60):
        need(type(seconds) in {int, float} and math.isfinite(seconds) and 0 < seconds <= 600, "bounded audit deadline required")
        self.root = Path(namespace).absolute()
        need(self.root.resolve(strict=True) == self.root and self.root.is_dir(), "canonical archive root required")
        self.deadline = time.monotonic() + seconds
        self.total_read_bytes, self.observed = 0, {}

    def tick(self):
        need(time.monotonic() < self.deadline, "read-only audit deadline exceeded")

    def path(self, locator):
        need(type(locator) is str and locator and "\0" not in locator, "bounded retained locator required")
        path = PurePosixPath(locator)
        need(not path.is_absolute() and ".." not in path.parts and path.as_posix() == locator, "relative canonical locator required")
        result = self.root / path
        for parent in (result, *result.parents):
            need(not parent.is_symlink(), "retained locator traverses a symlink")
            if parent == self.root:
                break
        return result

    def raw(self, locator, maximum=8 * MIB):
        self.tick()
        path = self.path(locator)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= maximum, "bounded single-link regular artifact required")
            raw = stream.read(maximum + 1)
            after = os.fstat(stream.fileno())
        need(len(raw) == before.st_size <= maximum and identity(before) == identity(after) == identity(path.lstat()), "retained bytes changed during read")
        self.total_read_bytes += len(raw)
        need(self.total_read_bytes <= MAX_TOTAL_READ_BYTES, "aggregate guarded read bound exceeded")
        record = {"bytes": len(raw), "sha256": sha(raw)}
        need(locator not in self.observed or same(self.observed[locator], record), "retained artifact changed between observations")
        self.observed[locator] = record
        return raw

    def json(self, locator, maximum=8 * MIB):
        return parse(self.raw(locator, maximum))

    def cas(self, wanted, *, source=False):
        check_cid(wanted, source=source)
        locator = "private/source-artifacts/" + ("source" if source else "structured") + "/" + wanted[:4] + "/" + wanted
        raw = self.raw(locator, 65536 if source else 8 * MIB)
        need(cid(raw, source=source) == wanted, "retained CAS CID differs")
        if not source:
            need(raw == wire(parse(raw)), "retained structured CAS canonical bytes differ")
        return raw

    def whole_archive(self):
        root_info = self.root.lstat()
        records, total, traversed = [{"path": ".", "kind": "directory", "mode": stat.S_IMODE(root_info.st_mode), "mtime_ns": root_info.st_mtime_ns}], 0, 0
        for directory, names, files in os.walk(self.root, followlinks=False):
            self.tick(); names.sort(); files.sort()
            for name in list(names):
                path = Path(directory) / name; info = path.lstat(); traversed += 1
                need(traversed <= MAX_FILES * 2, "archive traversal bound exceeded")
                relative = path.relative_to(self.root).as_posix()
                if stat.S_ISLNK(info.st_mode):
                    names.remove(name)
                    records.append({"path": relative, "kind": "symlink", "target": os.readlink(path), "mode": stat.S_IMODE(info.st_mode), "mtime_ns": info.st_mtime_ns})
                else:
                    need(stat.S_ISDIR(info.st_mode), "archive directory is not inert")
                    records.append({"path": relative, "kind": "directory", "mode": stat.S_IMODE(info.st_mode), "mtime_ns": info.st_mtime_ns})
            for name in files:
                self.tick(); path = Path(directory) / name; before = path.lstat(); traversed += 1
                need(traversed <= MAX_FILES * 2 and len(records) < MAX_FILES, "archive entry bound exceeded")
                relative = path.relative_to(self.root).as_posix()
                if stat.S_ISLNK(before.st_mode):
                    records.append({"path": relative, "kind": "symlink", "target": os.readlink(path), "mode": stat.S_IMODE(before.st_mode), "mtime_ns": before.st_mtime_ns})
                    continue
                need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= MAX_FILE_BYTES, "archive has nonregular/oversized member")
                total += before.st_size; need(total <= MAX_ARCHIVE_BYTES, "archive byte bound exceeded")
                digest = hashlib.sha256()
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, "rb") as stream:
                    need(identity(os.fstat(stream.fileno())) == identity(before), "archive changed before stream")
                    for block in iter(lambda: stream.read(MIB), b""):
                        self.tick(); digest.update(block)
                    need(identity(os.fstat(stream.fileno())) == identity(before), "archive changed during stream")
                need(identity(path.lstat()) == identity(before), "archive changed after stream")
                records.append({"path": relative, "kind": "file", "bytes": before.st_size, "sha256": digest.hexdigest(), "mode": stat.S_IMODE(before.st_mode), "nlink": before.st_nlink, "mtime_ns": before.st_mtime_ns})
        records.sort(key=lambda row: row["path"])
        return {"files": records, "regular_files": sum(row["kind"] == "file" for row in records), "regular_bytes": total, "inventory_cid": structured(records)}


def native_resources(value):
    closed(value, {"active_lease_count", "waiting_request_count"}, "native ending resources")
    need(type(value["active_lease_count"]) is int and type(value["waiting_request_count"]) is int
         and value["active_lease_count"] == value["waiting_request_count"] == 0,
         "actual native resource counts must be zero")


def audit_producers(reader, envelopes, archive, *, authored_fixture=True):
    selected = reader.json("generation-inputs.json")
    closed(selected, {"schema", "files", "execution_attestation", "scope"}, "selected producer exports")
    need(selected["schema"] == "codebase-source-delta-selected-producers@1"
         and selected["execution_attestation"] is False and selected["scope"] == "listed_local_files_only",
         "selected producer scope differs")
    rows = selected["files"]
    need(type(rows) is list and 1 <= len(rows) <= 40, "bounded selected producer list required")
    names = []
    files = {row["path"]: row for row in archive["files"] if row["kind"] == "file"}
    observed = {}
    for row in rows:
        closed(row, {"name", "path", "copy", "bytes", "sha256"}, "selected producer")
        text(row["name"], 256); text(row["path"]); integer(row["bytes"], 4 * MIB, 1)
        names.append(row["name"])
        raw = reader.raw(row["copy"], 4 * MIB)
        need(len(raw) == row["bytes"] and sha(raw) == row["sha256"], "retained producer copy SHA differs")
        need(row["copy"] in files and files[row["copy"]]["sha256"] == row["sha256"]
             and files[row["copy"]]["bytes"] == row["bytes"], "producer copy differs from full archive pin")
        observed[row["name"]] = row["sha256"]
    need(names == sorted(set(names)), "selected producer names duplicated/reordered")
    implementations = [value["value"]["implementation"]["files"] for value in envelopes]
    need(all(same(implementations[0], item) for item in implementations[1:]), "default/reference implementation generation differs")
    extras = {"qualification_harness", "authored_repository_fixture"} if authored_fixture else {"qualification_harness"}
    need(set(observed) == set(implementations[0]) | extras
         and all(observed.get(name) == digest for name, digest in implementations[0].items()),
         "selected copies do not bind every delta producer and both helpers")
    return {"selected_files": len(rows), "implementation_files": len(implementations[0]),
        "selected_producer_cid": structured(rows), "selected_copies_verified": True,
        "mutable_working_checkout_qualified": False, "transitive_dependency_closure_attested": False}


def verify_ignored_present_observation(result, envelope, previous_manifest, current_manifest, *, raw_physical):
    """Verify a removed capture member is still a bounded physical file."""
    need(type(result) is dict and result.get("schema") == "codebase-source-delta-ignore-native-qualification@1"
         and result.get("qualified") is True and result.get("scope") == "two_member_ignored_but_physically_present_capture_removal",
         "native ignored-file control did not complete")
    integer(result.get("pid"), minimum=1)
    value = envelope["value"]
    need(value["optimized"] is True and result.get("optimized_cid") == envelope["artifact_cid"]
         and same(result.get("previous_head"), value["previous_head"])
         and same(result.get("current_head"), value["current_head"]), "ignored-file control head/delta join differs")
    wanted_coverage = {"previous_entries": 2, "current_entries": 1, "union_entries": 2,
        "classifications": {"retained": 1, "changed": 0, "added": 0, "removed": 1}}
    need(all(same(value["coverage"].get(key), expected) for key, expected in wanted_coverage.items())
         and same(result.get("coverage"), value["coverage"]), "ignored-file complete capture coverage differs")
    previous, current = result.get("physical_presence_before"), result.get("physical_presence_after")
    for observation in (previous, current):
        closed(observation, {"path", "bytes", "sha256", "source_cid", "regular"}, "physical ignored-source observation")
        need(observation["path"] == "ignored.py" and observation["regular"] is True, "physical ignored-source path/type differs")
        integer(observation["bytes"], 65536)
        need(type(raw_physical) is bytes and len(raw_physical) == observation["bytes"]
             and sha(raw_physical) == observation["sha256"] and cid(raw_physical, source=True) == observation["source_cid"],
             "physical ignored-source bytes/CID differ")
    need(same(previous, current), "physical ignored source changed between native observations")
    ignored_key, normal_key = "raw:" + b"ignored.py".hex(), "raw:" + b"normal.py".hex()
    rows = {row["source_key"]: row for row in value["ledger"]}
    need(set(rows) == {ignored_key, normal_key}, "ignored-file raw capture membership differs")
    removed = rows[ignored_key]
    need(removed["classification"] == "removed" and removed["current"] is None
         and removed["previous"]["entry"]["source_cid"] == previous["source_cid"]
         and removed["previous"]["entry"]["opaque_reason"] is None
         and removed["previous"]["entry"]["disposition"] == "untracked"
         and removed["source_bytes_comparison"] == removed["ast_identity_comparison"] == "unavailable",
         "ignored capture removal was promoted to physical absence or equality")
    need(rows[normal_key]["classification"] == "retained", "normal entry retention must preserve complete entry identity")
    old_keys = {"raw:" + row["raw_path_hex"] for row in previous_manifest["snapshot"]["entries"]}
    new_keys = {"raw:" + row["raw_path_hex"] for row in current_manifest["snapshot"]["entries"]}
    need(ignored_key in old_keys and ignored_key not in new_keys and normal_key in old_keys & new_keys,
         "ignored source is not removed from complete captures")
    for field in ("physical_absence_verified", "numerical_reuse", "model_advanced", "repository_code_executed",
                  "worker_launched", "proof_authority", "admission_authority"):
        need(result.get(field) is False, "ignored control authority differs: " + field)
    need(result.get("removal_scope") == "absent_from_current_complete_capture", "ignored-file removal scope differs")
    false_authority(result.get("authority"))
    for field in ("model_registry_open_attempts", "training_attempts", "inference_attempts", "new_fitting_epochs", "source_owner_reopens"):
        need(type(result.get(field)) is int and result[field] == 0, "ignored-file source-only guard differs: " + field)
    need(result.get("receiving_verified") is True and result.get("physical_bytes_unchanged") is True
         and result.get("source_connection_closed") is True, "ignored-file receiving/owner closure missing")
    closed(result.get("current_owner_counts"), {"source", "model"}, "ignored-file current owners")
    need(all(type(count) is int and count == 0 for count in result["current_owner_counts"].values()), "ignored-file native owners remain open")
    return {"ignored_source_key": ignored_key, "physical_presence": previous,
        "removed_from_current_complete_capture": True, "physical_absence_verified": False,
        "observed_native_pid": result["pid"], "model_owners_opened_here": False,
        "new_fitting_epochs": 0, "numerical_reuse": False, "process_origin_attested": False}


def audit_ignored_archive(namespace, *, seconds=60):
    began = time.monotonic()
    reader = Reader(namespace, seconds)
    before = reader.whole_archive()
    report = {"schema": "codebase-source-delta-ignore-independent-audit@1", "namespace": str(reader.root),
        "qualified": False, "preserved": False, "errors": [], "native_owners_opened": False,
        "git_executed": False, "primary_writes": False, "physical_absence_verified": False}
    try:
        previous_manifest, current_manifest = reader.json("previous-manifest.json", 4 * MIB), reader.json("current-manifest.json", 4 * MIB)
        previous_receipt, current_receipt = reader.json("previous-publication-receipt.json", 64 * 1024), reader.json("current-publication-receipt.json", 64 * 1024)
        envelope = reader.json("optimized-source-delta.json")
        need(reader.cas(envelope["artifact_cid"]) == wire(envelope["value"]), "ignored-file export differs from native durable CAS")
        report["source_delta"] = verify_source_delta(envelope, previous_manifest, current_manifest, previous_receipt, current_receipt, get_artifact=reader.cas)
        result = reader.json("result.json")
        report["ignored_present_observation"] = verify_ignored_present_observation(result, envelope, previous_manifest, current_manifest,
                                                                                 raw_physical=reader.raw("repository/ignored.py", 65536))
        need(b"\nignored.py\n" in reader.raw("repository/.git/info/exclude", 65536), "retained native local ignore rule missing")
        report["producers"] = audit_producers(reader, [envelope], before, authored_fixture=False)
        phases = result.get("phases")
        need(type(phases) is list and [row.get("name") for row in phases] == ["git_init", "git_config", "git_config", "git_add", "git_commit",
            "publish_previous_source", "publish_current_source", "build_default_source_delta", "receive_current_source_delta"]
             and all(row.get("status") == "completed" for row in phases), "ignored native phase population/status differs")
        guards = result.get("named_execution_guards")
        need(type(guards) is list and len(guards) == 7 and len({row.get("name") for row in guards}) == 7
             and Counter(row.get("counter") for row in guards) == {"model_registry_open_attempts": 1, "training_attempts": 3, "inference_attempts": 3},
             "ignored native named execution guards differ")
        final = result.get("final_resources")
        need(type(final) is dict and all(type(final.get(field)) is int and final[field] == 0 for field in ("active_lease_count", "active_child_lease_count", "waiting_request_count"))
             and type(final.get("allocated")) is dict and all(type(amount) is int and amount == 0 for amount in final["allocated"].values()),
             "ignored native resources were not drained")
        state = reader.json("resource-admission.json")
        need(type(state) is dict and state.get("leases") == {} and state.get("waiters") == {}
             and observation_same(state.get("config"), result.get("scheduler_configuration")), "ignored persisted resource state differs")
        elapsed = result.get("recorded_seconds")
        need(type(elapsed) in {int, float} and math.isfinite(elapsed) and elapsed >= 0, "ignored actual native cost invalid")
        report.update(qualified=True, recorded_native_seconds=elapsed, final_resources=final)
    except Exception as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        after = reader.whole_archive()
        report["preserved"] = same(before, after)
        if not report["preserved"]:
            report["qualified"] = False
            report["errors"].append({"type": "ArchiveChanged", "message": "ignored sibling bytes/membership/modes/mtimes changed"})
        report["archive"] = after
        report["guarded_read_bytes"] = reader.total_read_bytes
        report["recorded_audit_seconds"] = time.monotonic() - began
    return report


def audit_native_observations(reader, result, summary, archive):
    need(type(result) is dict and result.get("schema") == "codebase-source-delta-native-qualification@1"
         and result.get("qualified") is True and result.get("scope") == "300_member_structural_source_successor",
         "native qualification did not complete")
    integer(result.get("pid"), minimum=1)
    for field in ("model_registry_open_attempts", "training_attempts", "inference_attempts", "new_fitting_epochs"):
        need(type(result.get(field)) is int and result[field] == 0, "native source-only work guard differs: " + field)
    for field in ("repository_code_executed", "proof_authority", "admission_authority", "production_default_activated",
                  "cuda_qualified", "384d_qualified", "physical_absence_verified"):
        need(result.get(field) is False, "native scope cannot grant authority: " + field)
    for field in ("default_reference_equal", "native_owner_unchanged", "original_source_artifacts_preserved",
                  "cold_receiving_verified", "selected_producers_unchanged"):
        need(result.get(field) is True, "native closing condition missing: " + field)
    need(same(result.get("coverage"), summary["coverage"]) and same(result.get("previous_head"), summary["previous_head"])
         and same(result.get("current_head"), summary["current_head"]), "native result full source/coverage join differs")
    need(summary["coverage"]["previous_entries"] == summary["coverage"]["current_entries"] == 300
         and summary["coverage"]["union_entries"] == 302
         and same(summary["coverage"]["classifications"], {"retained": 297, "changed": 1, "added": 2, "removed": 2}),
         "actual 300-member source fixture coverage differs")
    phases = result.get("phases")
    wanted = ["author_300_member_repository", "publish_previous_source", "publish_current_source",
        "build_default_source_delta", "build_reference_source_delta", "receive_current_source_delta",
        "refuse_same_head_source_drift", "refuse_precancelled_receiving", "refuse_native_sql_drift",
        "cold_receive_current_source_delta"]
    need(type(phases) is list and [row.get("name") for row in phases] == wanted
         and all(row.get("status") == "completed" for row in phases), "native phase population/status differs")
    for row in phases:
        need(type(row.get("elapsed_seconds")) in {int, float} and math.isfinite(row["elapsed_seconds"])
             and row["elapsed_seconds"] >= 0, "native operation elapsed observation invalid")
    controls = result.get("controls")
    need(type(controls) is list and [row.get("name") for row in controls] == ["same_head_source_drift", "precancelled_receiving", "native_sql_drift"], "native refusal control population differs")
    for row in controls:
        need(row.get("refused") is True, "native control was accepted")
        if row["name"] == "precancelled_receiving":
            need(row.get("error_type") == "LeaseCancelledError", "pre-cancelled control must report cancellation")
        else:
            need(row.get("error_type") in {"StaleCodebaseError", "CodebaseSourceDeltaError", "DuckDBASTStoreIntegrityError", "CodebaseProjectionReplayError"}, "resource pressure cannot count as integrity refusal")
    before, after = reader.json("native-source-owner-before.json"), reader.json("native-source-owner-after.json")
    need(same(before, after), "native source owner rows/head changed")
    closed(before, {"current_head", "tables"}, "native source owner export")
    need(same(before["current_head"], summary["current_head"]) and type(before["tables"]) is dict and set(before["tables"]) == TABLE_NAMES, "exact current native 13-table export required")
    for table in before["tables"].values():
        closed(table, {"rows", "sha256"}, "native table digest")
        integer(table["rows"], 100_000)
        need(type(table["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", table["sha256"]), "native table SHA differs")
    need(type(result.get("source_owner_reopens")) is int and result["source_owner_reopens"] == 1, "one actual cold source owner reopen required")
    reopen = result.get("cold_owner_reopen")
    closed(reopen, {"before", "after", "old_connection_closed", "new_source_owner", "same_process", "new_process_claimed"}, "cold owner reopen")
    need(reopen["old_connection_closed"] is True and reopen["new_source_owner"] is True and reopen["same_process"] is True and reopen["new_process_claimed"] is False, "cold receiving does not establish a new process")
    for side in (reopen["before"], reopen["after"]):
        closed(side, {"pid", "index_id", "connection_id", "current_head"}, "cold owner identity")
        for field in ("pid", "index_id", "connection_id"):
            integer(side[field], minimum=1)
        need(side["pid"] == result["pid"] and same(side["current_head"], summary["current_head"]), "cold owner PID/head differs")
    need(reopen["before"]["index_id"] != reopen["after"]["index_id"]
         and reopen["before"]["connection_id"] != reopen["after"]["connection_id"], "cold native owner was not recreated")
    original = reader.json("source-artifacts-before.json")
    need(type(original) is list and len(original) <= 8192, "bounded original source artifact membership required")
    files = {row["path"]: row for row in archive["files"] if row["kind"] == "file"}
    paths = []
    for row in original:
        closed(row, {"path", "bytes", "sha256"}, "original source artifact")
        path = "private/source-artifacts/" + row["path"]
        reader.path(path); paths.append(row["path"])
        need(path in files and files[path]["bytes"] == row["bytes"] and files[path]["sha256"] == row["sha256"], "original source artifact bytes were not preserved")
    need(paths == sorted(set(paths)), "original source artifact population reordered/duplicated")
    for head in (summary["previous_head"], summary["current_head"]):
        wanted_path = "structured/" + head["manifest_cid"][:4] + "/" + head["manifest_cid"]
        need(wanted_path in paths, "original captured manifest is missing from preserved subset")
    for kind, wanted in summary["artifact_closure"].items():
        prefix = "source" if kind == "source_cids" else "structured"
        need(all(prefix + "/" + value[:4] + "/" + value in paths for value in wanted),
             "complete captured source/AST closure is missing from original preserved subset")
    native_resources(result.get("final_resources"))
    state = reader.json("resource-admission.json")
    need(type(state) is dict and state.get("leases") == {} and state.get("waiters") == {}
         and observation_same(state.get("config"), result.get("scheduler_configuration")), "persisted native resources/config differ")
    elapsed = result.get("recorded_seconds")
    need(type(elapsed) in {int, float} and math.isfinite(elapsed) and elapsed >= 0, "actual native cost observation invalid")
    return {"qualified_native_observation": True, "pid": result["pid"], "recorded_seconds": elapsed,
        "source_owner_reopens": 1, "same_process_cold_reopen": True, "new_process_claimed": False,
        "original_artifacts_preserved": len(original), "native_13_table_exports_unchanged": True,
        "sql_execution_attested_here": False, "new_fitting_epochs": 0, "model_registry_open_attempts": 0,
        "training_attempts": 0, "inference_attempts": 0, "final_resources": result["final_resources"],
        "controls": controls}


def audit_closed_archive(namespace, *, seconds=60, ignore_namespace=None):
    began = time.monotonic()
    reader = Reader(namespace, seconds)
    before = reader.whole_archive()
    report = {"schema": SCHEMA, "namespace": str(reader.root), "qualified": False, "preserved": False,
        "authority": {key: False for key in sorted(AUTHORITY)}, "native_owners_opened": False,
        "sql_opened": False, "model_owner_opened": False, "git_executed": False,
        "repository_source_executed": False, "primary_writes": False, "errors": [],
        "qualification_profile": "native_300_capture_plus_ignored_present_control",
        "main_observation_qualified": False, "ignored_control_qualified": False}
    try:
        previous_manifest, current_manifest = reader.json("previous-manifest.json", 4 * MIB), reader.json("current-manifest.json", 4 * MIB)
        previous_receipt, current_receipt = reader.json("previous-publication-receipt.json", 64 * 1024), reader.json("current-publication-receipt.json", 64 * 1024)
        envelopes = [reader.json(name + "-source-delta.json") for name in ("optimized", "reference")]
        summaries = []
        for envelope in envelopes:
            need(reader.cas(envelope["artifact_cid"]) == wire(envelope["value"]), "delta export differs from durable CAS")
            summaries.append(verify_source_delta(envelope, previous_manifest, current_manifest, previous_receipt, current_receipt, get_artifact=reader.cas))
        need(envelopes[0]["value"]["optimized"] is True and envelopes[1]["value"]["optimized"] is False, "native default/reference opt-out missing")
        left = {key: value for key, value in envelopes[0]["value"].items() if key != "optimized"}
        right = {key: value for key, value in envelopes[1]["value"].items() if key != "optimized"}
        need(same(left, right), "default/reference differ beyond opt-out flag")
        report["source_delta"] = summaries[0]
        report["reference_delta_cid"] = envelopes[1]["artifact_cid"]
        report["default_reference_equal"] = True
        report["producers"] = audit_producers(reader, envelopes, before)
        result = reader.json("result.json")
        need(result.get("optimized_cid") == summaries[0]["delta_cid"] and result.get("reference_cid") == summaries[1]["delta_cid"], "native result/delta CID join differs")
        report["native_observations"] = audit_native_observations(reader, result, summaries[0], before)
        report["main_observation_qualified"] = True
        need(ignore_namespace is not None, "explicit closed ignored-file sibling required for complete qualification")
        ignored_root = Path(ignore_namespace).resolve(strict=True)
        need(ignored_root != reader.root and not ignored_root.is_relative_to(reader.root) and not reader.root.is_relative_to(ignored_root), "ignored control must be a separate closed namespace")
        report["ignored_control"] = audit_ignored_archive(ignored_root, seconds=min(seconds, 60))
        need(report["ignored_control"]["qualified"] is True and report["ignored_control"]["preserved"] is True, "ignored-but-physically-present native control did not independently qualify")
        need(same(summaries[0]["implementation"], report["ignored_control"]["source_delta"]["implementation"]),
             "main and ignored control selected product generations differ")
        report["ignored_control_qualified"] = True
        report["common_product_generation_equal"] = True
        report["qualified"] = True
    except Exception as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        after = reader.whole_archive()
        report["preserved"] = same(before, after)
        if not report["preserved"]:
            report["qualified"] = False
            report["errors"].append({"type": "ArchiveChanged", "message": "closed archive bytes/membership/modes/mtimes changed"})
        report["archive"] = after
        report["guarded_read_bytes"] = reader.total_read_bytes
        report["guarded_artifacts"] = reader.observed
        report["recorded_seconds"] = time.monotonic() - began
        report["reader"] = {"bytes": Path(__file__).stat().st_size, "sha256": sha(Path(__file__).read_bytes())}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("namespace", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ignore-namespace", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    namespace = args.namespace.resolve(strict=True)
    output = args.output.absolute()
    need(not output.is_relative_to(namespace) and not output.exists() and not output.is_symlink(), "audit output must be fresh and outside the primary archive")
    need(output.parent.resolve(strict=True) == output.parent, "canonical existing audit parent required")
    source = Path(__file__).read_bytes()
    retained_reader = output.with_suffix(".reader.py")
    with retained_reader.open("xb") as stream:
        stream.write(source)
    retained_reader.chmod(0o444)
    ignored_root = args.ignore_namespace.resolve(strict=True)
    need(not output.is_relative_to(ignored_root), "audit output cannot alter ignored sibling")
    report = audit_closed_archive(namespace, seconds=args.seconds, ignore_namespace=ignored_root)
    need(sha(Path(__file__).read_bytes()) == sha(source), "audit reader changed during qualification")
    with output.open("xb") as stream:
        stream.write(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
    output.chmod(0o444)
    print(json.dumps({"qualified": report["qualified"], "preserved": report["preserved"], "errors": report["errors"], "sha256": sha(output.read_bytes())}))
    return 0 if report["qualified"] and report["preserved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
