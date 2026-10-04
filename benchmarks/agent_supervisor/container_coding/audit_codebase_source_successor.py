"""Independent stdlib receiving for a closed current source successor run.

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
import unicodedata

SCHEMA = "codebase-source-successor-independent-audit@1"
DELTA_SCHEMA = "codebase-inventory-source-delta@1"
MIB = 1024 ** 2
MAX_FILES = 65_536
MAX_ARCHIVE_BYTES = 4 * 1024 ** 3
MAX_FILE_BYTES = 1024 ** 3
MAX_TOTAL_READ_BYTES = 1024 * MIB
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


class SourceSuccessorAuditError(ValueError):
    """A retained artifact, complete join, or closed audit condition differs."""


def need(condition, message):
    if not condition:
        raise SourceSuccessorAuditError(message)


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
        raise SourceSuccessorAuditError("canonical CID required") from error
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
        raise SourceSuccessorAuditError("nonfinite JSON constant: " + value)
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
    except (UnicodeError, SourceSuccessorAuditError):
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


def numerical_wire(value):
    """Finite numerical/preimage JSON lives outside structured proof identity."""
    observation_same(value, value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def named_digest(namespace, value):
    strict(value)
    return "sha256:" + sha(namespace.encode() + b"\n" + numerical_wire(value))


def modality_digest(value):
    strict(value)
    def normalize(item):
        if type(item) is str:
            return unicodedata.normalize("NFC", item)
        if type(item) is list:
            return [normalize(child) for child in item]
        if type(item) is dict:
            result = {unicodedata.normalize("NFC", key): normalize(child) for key, child in item.items()}
            need(len(result) == len(item), "modality normalization collision")
            return result
        return item
    return sha(wire({"canonicalization": "ir-canonical-json-v1", "collection_semantics": {},
        "domain": "autoencoder.modality", "identity_profile": "ir-canonical-identity-v1",
        "payload": normalize(value), "schema_version": "autoencoder-modality-contract/v1"}))


def export_value(reader, locator, *, raw=False):
    envelope = reader.json(locator)
    closed(envelope, {"artifact_cid", "value"}, "native artifact export")
    expected = cid(numerical_wire(envelope["value"]), source=True) if raw else structured(envelope["value"])
    need(envelope["artifact_cid"] == expected, "native export CID differs: " + locator)
    if raw:
        wanted = check_cid(expected, source=True)
        locator = "private/source-artifacts/source/" + wanted[:4] + "/" + wanted
        retained = reader.raw(locator, 16 * MIB)
        need(cid(retained, source=True) == expected and retained == numerical_wire(envelope["value"]),
             "raw numerical native CAS bytes differ")
    else:
        need(reader.cas(expected) == wire(envelope["value"]), "structured native export/CAS bytes differ")
    return envelope


def implementation_shape(value, producers=None):
    closed(value, {"files", "sha256", "scope"}, "selected implementation")
    files = value["files"]
    need(type(files) is dict and 1 <= len(files) <= 32
         and value["sha256"] == sha(numerical_wire(files))
         and value["scope"] == "listed_local_files_only_not_execution_attestation",
         "selected implementation generation differs")
    for name, digest in files.items():
        text(name, 256)
        need(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest), "implementation SHA differs")
        if producers is not None:
            need(producers.get(name) == digest, "retained producer does not match implementation: " + name)


def verify_model(model, raw, head):
    closed(model, {"version_id", "variant_id", "artifact", "artifact_cid", "contract_sha256",
        "state_sha256", "feature_space_sha256", "latent_width", "feature_columns", "projection_ids",
        "projection_widths", "ancestry"}, "selected model")
    text(model["version_id"], 512); text(model["variant_id"], 512)
    integer(model["latent_width"], 8, 8); integer(model["feature_columns"], 1024, 1)
    closed(model["artifact"], {"sha256", "bytes"}, "model artifact")
    integer(model["artifact"]["bytes"], 16 * MIB, 1)
    need(type(raw) is bytes and len(raw) == model["artifact"]["bytes"]
         and sha(raw) == model["artifact"]["sha256"] and cid(raw, source=True) == model["artifact_cid"],
         "full selected model byte identity differs")
    saved = parse(raw)
    need(raw == numerical_wire(saved) and type(saved) is dict
         and set(saved) == {"contract", "state", "feature_space", "report"}
         and type(saved["report"].get("codebase_provenance")) is dict,
         "full canonical model checkpoint required")
    need(modality_digest(saved["contract"]) == model["contract_sha256"]
         and sha(numerical_wire(saved["state"])) == model["state_sha256"]
         and sha(numerical_wire(saved["feature_space"])) == model["feature_space_sha256"],
         "complete numerical model fingerprints differ")
    state, space, provenance = saved["state"], saved["feature_space"], saved["report"]["codebase_provenance"]
    need(state.get("contract_sha256") == model["contract_sha256"]
         and state.get("latent_width") == model["latent_width"] == 8
         and len(space["columns"]) == model["feature_columns"]
         and same(provenance.get("head"), head)
         and provenance.get("schema") == "codebase-source-feature-lineage@1",
         "source head/state/feature-space provenance differs")
    ids, widths = model["projection_ids"], model["projection_widths"]
    need(type(ids) is list and ids == sorted(set(ids)) and 1 <= len(ids) <= 2,
         "complete ordered model projection IDs required")
    closed(widths, ids, "model projection widths")
    for amount in widths.values():
        integer(amount, 1024, 1)
    need(sum(widths.values()) == model["feature_columns"], "complete model projection widths differ")
    ancestry = model["ancestry"]
    need(type(ancestry) is list and 1 <= len(ancestry) <= 8
         and ancestry[0] == {"version_id": model["version_id"], "artifact": model["artifact"]},
         "model ancestry head differs")
    for row in ancestry:
        closed(row, {"version_id", "artifact"}, "model ancestor")
        text(row["version_id"], 512)
        closed(row["artifact"], {"sha256", "bytes"}, "ancestor artifact")
        integer(row["artifact"]["bytes"], 16 * MIB, 1)
        need(type(row["artifact"]["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["artifact"]["sha256"]),
             "ancestor artifact SHA differs")
    need(len({row["version_id"] for row in ancestry}) == len(ancestry), "duplicate model ancestor")
    training_flags = {"qualified", "admitted", "formalized", "promotion_performed", "proof_authority",
        "source_runtime_semantics_verified", "behavioral_satisfaction", "admission_authority", "completion_authority"}
    closed(provenance, {"schema", "head", "selections", "parent_version_id", "continuation", "training_targets", "tuning_targets",
        "canary_targets", "replay_targets", "ancestral_training", "current_evaluation_bindings", "implementation",
        "contract_sha256", "feature_space_sha256", "canary_monitoring", "replay_monitoring", *training_flags}, "stored source feature lineage")
    need(all(provenance.get(field) is False for field in training_flags), "source model lineage acquired authority")
    basic_flags = {"qualified", "admitted", "formalized", "promotion_performed"}
    closed(state, {"schema", "contract_sha256", "feature_space_sha256", "latent_width", "parameters", "adam", "completed_epochs",
        "optimizer_config", "tuning_targets_sha256", *basic_flags}, "full native numerical state")
    closed(space, {"schema", "domain_id", "projection_ids", "projections", "columns", "training_sources", "excluded_projection_ids",
        "training_targets_sha256", "normalization", *basic_flags}, "full native feature basis")
    need(state["schema"] == "native-projection-feature-state/v1" and space["schema"] == "native-projection-feature-space/v1"
        and saved["report"]["schema"] == "native-projection-feature-training/v1"
        and all(owner[field] is False for owner in (state, space, saved["report"]) for field in basic_flags)
        and state["feature_space_sha256"] == provenance["feature_space_sha256"] == model["feature_space_sha256"]
        and provenance["contract_sha256"] == model["contract_sha256"]
        and same(space["projection_ids"], ids) and set(space["projections"]) == set(ids)
        and space["normalization"] == "log1p_then_l2_per_projection", "native numerical schema/basis/authority differs")
    columns = space["columns"]
    need(type(columns) is list and all(type(row) is list and len(row) == 2 and row[0] in ids and type(row[1]) is str for row in columns)
        and columns == sorted(columns) and len({tuple(row) for row in columns}) == len(columns)
        and same(dict(Counter(row[0] for row in columns)), widths), "canonical complete numerical feature columns differ")
    integer(state["completed_epochs"])
    config = state["optimizer_config"]
    closed(config, {"name", "learning_rate", "betas", "eps"}, "stored Adam configuration")
    need(config["name"] == "Adam" and observation_same(config["betas"], [0.9, 0.999]) and config["eps"] == 1e-8
        and type(config["learning_rate"]) in {int, float} and math.isfinite(config["learning_rate"])
        and 0 < config["learning_rate"] <= 0.1, "stored Adam configuration differs")
    shapes = ((model["feature_columns"], 8), (8,), (8, model["feature_columns"]), (model["feature_columns"],))
    def tensor(values, shape, *, nonnegative=False):
        need(type(values) is list and len(values) == shape[0], "full numerical tensor shape differs")
        for value in values:
            if len(shape) > 1:
                tensor(value, shape[1:], nonnegative=nonnegative)
            else:
                need(type(value) in {int, float} and math.isfinite(value) and (not nonnegative or value >= 0),
                    "full numerical tensor scalar differs")
    need(type(state["parameters"]) is list and type(state["adam"]) is list and len(state["parameters"]) == len(state["adam"]) == 4,
        "complete four parameter/Adam blocks required")
    for parameter, moment, shape in zip(state["parameters"], state["adam"], shapes):
        tensor(parameter, shape)
        closed(moment, {"step", "exp_avg", "exp_avg_sq"}, "full Adam block")
        need(type(moment["step"]) is int and moment["step"] == state["completed_epochs"], "Adam step/epoch binding differs")
        tensor(moment["exp_avg"], shape); tensor(moment["exp_avg_sq"], shape, nonnegative=True)
    return {"saved": saved, "checkpoint_state": {"state_sha256": model["state_sha256"],
        "completed_epochs": integer(state["completed_epochs"]),
        "adam_steps": [integer(item["step"]) for item in state["adam"]],
        "latent_width": 8, "feature_columns": model["feature_columns"], "artifact": model["artifact"],
        "report_sha256": sha(numerical_wire(saved["report"]))}}


def verify_training_record(envelope, model, saved, head, parent_version):
    value = envelope["value"]
    need(envelope["artifact_cid"] == structured(value), "training record structured CID differs")
    closed(value, {"schema", "version_id", "variant_id", "parent_version_id", "head", "registry_artifact",
        "checkpoint_raw_cid", "contract_sha256", "state_sha256", "feature_space_sha256", "report_json",
        "authority", "source_model_generation", "training_performed_during_load", "model_head_selected"}, "source training record")
    need(value["schema"] == "codebase-source-feature-training@1"
         and value["version_id"] == model["version_id"] and value["variant_id"] == model["variant_id"]
         and value["parent_version_id"] == parent_version and same(value["head"], head)
         and same(value["registry_artifact"], model["artifact"])
         and value["checkpoint_raw_cid"] == model["artifact_cid"]
         and all(value[field] == model[field] for field in ("contract_sha256", "state_sha256", "feature_space_sha256"))
         and value["report_json"].encode() == numerical_wire(saved["report"])
         and value["source_model_generation"] == "historical_recorded_source_and_private_model_candidate"
         and value["training_performed_during_load"] is False and value["model_head_selected"] is False,
         "source training record full source/model/parent/report binding differs")
    flags = {"qualified", "admitted", "formalized", "promotion_performed", "proof_authority",
        "source_runtime_semantics_verified", "behavioral_satisfaction", "admission_authority", "completion_authority"}
    need(type(value["authority"]) is dict and set(value["authority"]) == flags
         and all(flag is False for flag in value["authority"].values()), "training record authority differs")


def verify_selection(envelope, delta, root, previous_model):
    value = envelope["value"]
    need(envelope["artifact_cid"] == structured(value), "successor selection CID differs")
    closed(value, {"schema", "source_delta_cid", "previous_head", "current_head", "previous_membership_cid",
        "current_membership_cid", "previous_training_record_cid", "training_record_cid", "previous_model", "model",
        "root_cid", "scan_limits", "optimized", "implementation", "authority", "training_performed_here",
        "inference_performed_here", "numerical_reuse", "model_head_promoted"}, "successor selection")
    need(value["schema"] == "codebase-inventory-successor-scan@1" and type(value["optimized"]) is bool,
         "successor selection schema/optimization differs")
    false_authority(value["authority"])
    need(all(value[field] is False for field in ("training_performed_here", "inference_performed_here", "numerical_reuse", "model_head_promoted")),
         "successor coordinator cannot acquire numerical or promotion authority")
    source = delta["value"]
    need(value["source_delta_cid"] == delta["artifact_cid"]
         and all(same(value[field], source[field]) for field in ("previous_head", "current_head", "previous_membership_cid", "current_membership_cid"))
         and same(value["previous_model"], previous_model), "successor source/delta/parent binding differs")
    r = root["value"]
    need(value["root_cid"] == root["artifact_cid"] and same(value["model"], r["model"])
         and same(value["scan_limits"], r["limits"]) and value["optimized"] is r["optimized"],
         "successor ordinary root/model/profile binding differs")
    child, parent = value["model"], value["previous_model"]
    need(child["version_id"] != parent["version_id"] and len(child["ancestry"]) == 2
         and same(child["ancestry"][1:], parent["ancestry"]), "exact registered direct child required")
    for field in ("variant_id", "contract_sha256", "feature_space_sha256", "latent_width", "feature_columns", "projection_ids", "projection_widths"):
        need(same(child[field], parent[field]), "parent/child frozen numerical basis differs: " + field)
    implementation_shape(value["implementation"])


def verify_root(envelope, delta, model):
    value = envelope["value"]
    need(envelope["artifact_cid"] == structured(value), "ordinary scan root CID differs")
    closed(value, {"schema", "head", "head_cid", "members", "membership_cid", "model", "limits",
        "optimized", "implementation", "authority"}, "ordinary scan root")
    source = delta["value"]
    members = [row["current"]["member"] for row in source["ledger"] if row["current"] is not None]
    need(value["schema"] == "codebase-inventory-resume-root@1" and type(value["optimized"]) is bool
         and same(value["head"], source["current_head"]) and value["head_cid"] == structured(value["head"])
         and same(value["members"], members) and value["membership_cid"] == structured(members) == source["current_membership_cid"]
         and same(value["model"], model), "fresh ordinary root source/model/complete membership differs")
    false_authority(value["authority"])
    implementation_shape(value["implementation"])
    limits = value["limits"]
    closed(limits, {"max_file_bytes", "max_inferred_rows", "max_input_bytes", "max_inventory_entries",
        "max_manifest_bytes", "max_output_bytes", "max_pages", "max_target_bytes", "page_entries"}, "ordinary scan limits")
    need(same(limits, {"max_file_bytes": 65536, "max_inferred_rows": 1024, "max_input_bytes": 32 * MIB,
        "max_inventory_entries": 512, "max_manifest_bytes": 4 * MIB, "max_output_bytes": 16 * MIB,
        "max_pages": 1024, "max_target_bytes": 4 * MIB, "page_entries": 32})
        and len(members) == 300, "explicit complete300/page32 scan profile differs")
    return members


def verify_prefix_page(envelope, root):
    value, r = envelope["value"], root["value"]
    need(envelope["artifact_cid"] == cid(numerical_wire(value), source=True), "numerical prefix raw CID differs")
    closed(value, {"schema", "root_cid", "head_cid", "membership_cid", "model_artifact_cid", "start", "end",
        "total_entries", "page_membership_cid", "previous_page_cid", "entries", "inference", "worker_receipt", "coverage", "authority"}, "prefix page")
    members = r["members"][:32]
    need(value["schema"] == "codebase-inventory-resume-page@1" and value["root_cid"] == root["artifact_cid"]
         and value["head_cid"] == r["head_cid"] and value["membership_cid"] == r["membership_cid"]
         and value["model_artifact_cid"] == r["model"]["artifact_cid"]
         and type(value["start"]) is int and value["start"] == 0 and type(value["end"]) is int and value["end"] == 32
         and type(value["total_entries"]) is int and value["total_entries"] == 300
         and value["page_membership_cid"] == structured(members) and value["previous_page_cid"] is None,
         "fresh prefix complete ordered membership binding differs")
    false_authority(value["authority"])
    entries = value["entries"]
    need(type(entries) is list and len(entries) == 32, "complete32 prefix entry ledger required")
    for ordinal, (entry, member) in enumerate(zip(entries, members)):
        closed(entry, {"member_index", "source_key", "entry_cid", "disposition", "reason", "target_sha256",
            "source_digest", "coverage", "inference_index"}, "prefix entry")
        need(type(entry["member_index"]) is int and entry["member_index"] == ordinal
             and entry["source_key"] == member["source_key"] and entry["entry_cid"] == member["entry_cid"]
             and entry["disposition"] in {"inferred", "opaque", "unindexed", "parse_failed", "parse_partial", "unsupported_target", "feature_incompatible", "deferred_budget"},
             "prefix member/disposition differs")
        need(entry["reason"] is None or (type(entry["reason"]) is str and 0 < len(entry["reason"]) <= 256),
            "bounded prefix disposition reason required")
        for field in ("target_sha256", "source_digest"):
            need(entry[field] is None or (type(entry[field]) is str and re.fullmatch(r"[0-9a-f]{64}", entry[field])),
                "prefix target/source SHA differs")
        need((entry["target_sha256"] is None) == (entry["source_digest"] is None), "prefix target binding incomplete")
        need(type(entry["coverage"]) is list and len(entry["coverage"]) <= 2, "bounded prefix feature coverage required")
        coverage_ids = []
        for item in entry["coverage"]:
            closed(item, {"projection_id", "known_atoms", "unknown_atoms"}, "prefix projection coverage")
            text(item["projection_id"], 512); integer(item["known_atoms"]); integer(item["unknown_atoms"])
            coverage_ids.append(item["projection_id"])
        need(coverage_ids == sorted(set(coverage_ids)), "prefix feature coverage duplicated or reordered")
        is_inferred = entry["disposition"] == "inferred"
        need(is_inferred == (entry["inference_index"] is not None), "prefix inference index/disposition differs")
        if is_inferred:
            need(entry["target_sha256"] is not None and entry["reason"] is None
                and len(entry["coverage"]) > 0 and all(item["known_atoms"] > 0 for item in entry["coverage"])
                and coverage_ids == r["model"]["projection_ids"], "positive selected projection coverage differs")
    counts = dict(sorted(Counter(row["disposition"] for row in entries).items()))
    inferred = [row for row in entries if row["disposition"] == "inferred"]
    need(len(inferred) > 0 and same(value["coverage"], {"inventory_entries": 32, "inferred_rows": len(inferred), "dispositions": counts}),
         "positive prefix coverage conservation differs")
    model, inference, receipt = r["model"], value["inference"], value["worker_receipt"]
    closed(inference, {"schema", "contract_sha256", "state_sha256", "feature_space_sha256", "rows", "coverage",
        "training_executed", "decoded_formulas_generated", "representation", "qualified", "admitted", "formalized", "promotion_performed"}, "native prefix inference")
    need(all(inference[field] is False for field in ("qualified", "admitted", "formalized", "promotion_performed")),
        "native numerical inference gained authority")
    need(type(inference) is dict and inference.get("schema") == "native-projection-feature-inference/v1"
         and inference.get("training_executed") is False and inference.get("decoded_formulas_generated") is False
         and inference.get("representation") == "native_compiler_structural_features_not_semantic_text_embeddings"
         and all(inference.get(field) == model[field] for field in ("contract_sha256", "state_sha256", "feature_space_sha256"))
         and same(inference.get("coverage"), [item for entry in inferred for item in entry["coverage"]]),
         "prefix inference selected model/coverage/authority differs")
    rows = inference["rows"]
    need(type(rows) is list and len(rows) == len(inferred), "positive numerical row population differs")
    for ordinal, (entry, row) in enumerate(zip(inferred, rows)):
        closed(row, {"source_digest", "latent", "reconstructed_projection_features"}, "numerical row")
        need(type(entry["inference_index"]) is int and entry["inference_index"] == ordinal
             and entry["source_digest"] == row["source_digest"] and len(row["latent"]) == 8
             and set(row["reconstructed_projection_features"]) == set(model["projection_ids"]), "numerical row source/layout differs")
        numbers = list(row["latent"])
        for projection, block in row["reconstructed_projection_features"].items():
            need(type(block) is list and len(block) == model["projection_widths"][projection], "numerical projection width differs")
            numbers.extend(block)
        need(all(type(number) in {int, float} and math.isfinite(number) for number in numbers), "nonfinite or aliased numerical value")
    closed(receipt, {"executable_sha256", "worker_sha256", "input_sha256", "output_sha256", "input_bytes", "output_bytes", "elapsed_ms",
        "returncode", "workspace_cleaned", "limits", "memory_enforcement", "source_execution_attested"}, "native prefix worker receipt")
    need(type(receipt["returncode"]) is int and receipt["returncode"] == 0 and receipt["workspace_cleaned"] is True
         and receipt["source_execution_attested"] is False
         and receipt["memory_enforcement"] == "sampled_process_tree_rss_with_possible_overshoot", "native prefix worker observation differs")
    for field in ("executable_sha256", "worker_sha256", "input_sha256", "output_sha256"):
        need(type(receipt[field]) is str and re.fullmatch(r"[0-9a-f]{64}", receipt[field]), "worker receipt SHA differs")
    integer(receipt["elapsed_ms"])
    closed(receipt["limits"], {"resident_memory_bytes", "max_input_bytes", "max_output_bytes"}, "native worker limits")
    integer(receipt["limits"]["resident_memory_bytes"], 1024 * MIB, 1024 * MIB)
    for field in ("input_bytes", "output_bytes"):
        integer(receipt[field], 32 * MIB, 1)
        limit = "max_" + field
        need(receipt[field] <= receipt["limits"][limit] == r["limits"][limit], "worker byte limits differ")
    return {"coverage": value["coverage"], "worker_receipt": receipt,
            "numerical_execution_independently_reperformed": False, "complete_scan_qualified": False}


MATERIAL_FIELDS = {"scan", "query_plan", "evidence_bundle", "evidence_adapters", "evidence_queries",
    "obligation_graph", "intent", "current_facts", "producers", "task_candidates", "predicates", "frozen_goal",
    "candidate_context", "model_provider", "parallel_tasks", "parallel_request", "admission_materials",
    "current_roots", "workflow_request", "extra"}



PLANNING_PREFIX = "ipfs_accelerate_py.agent_supervisor.planning."
RECORD_FIELDS = {
    PLANNING_PREFIX + "obligation_graph_compiler.TypedIntent": {"intent_id", "desired_predicates", "source_refs", "current_root_id", "metadata"},
    PLANNING_PREFIX + "obligation_graph_compiler.TypedPredicate": {"predicate_id", "predicate_type", "subject_ref", "object_ref", "property_id",
        "polarity", "support", "provenance_refs", "assumption_refs", "validation_requirement_refs", "proof_requirement_refs", "invalidation_selectors"},
    PLANNING_PREFIX + "obligation_graph_compiler.ProducerRule": {"producer_id", "effect_predicate_ids", "required_predicate_ids", "task_candidate_ids",
        "provenance_refs", "assumption_refs", "validation_requirement_refs", "proof_requirement_refs", "invalidation_selectors", "executable"},
    PLANNING_PREFIX + "obligation_graph_compiler.TaskCandidate": {"candidate_id", "closes_obligation_ids", "producer_id", "depends_on_candidate_ids", "provenance_refs"},
    PLANNING_PREFIX + "adaptive_planner.FrozenPlanningGoal": {"goal_id", "goal_content_id", "repository_tree_id", "policy"},
    PLANNING_PREFIX + "plan_evaluator.EvidenceAwarePlanPolicy": {"acceptance_criteria", "evidence_terms", "trusted_assumptions", "supported_semantics",
        "satisfied_dependencies", "allowed_scopes", "available_resource_classes", "max_estimated_resource_cost", "max_estimated_tokens",
        "max_estimated_runtime_seconds", "min_novelty", "require_validation", "require_proof"},
}
ENUM_VALUES = {PLANNING_PREFIX + "obligation_graph_compiler.PredicatePolarity": {"positive", "negative"},
               PLANNING_PREFIX + "obligation_graph_compiler.SemanticSupport": {"reviewed", "unsupported", "unknown"}}


def qualification_authored_preimages(snapshot_cid):
    """Reconstruct this harness's full authored meanings using stdlib data only."""
    def project(value):
        if type(value) in {tuple, list}:
            return {"$sequence": "tuple" if type(value) is tuple else "list", "items": [project(item) for item in value]}
        if type(value) is dict:
            if "$record" in value or "$enum" in value:
                return value
            return {"$mapping": {key: project(item) for key, item in value.items()}}
        return value
    def record(name, fields):
        return {"$record": PLANNING_PREFIX + name, "fields": project(fields)}
    def enum(name, value):
        return {"$enum": PLANNING_PREFIX + "obligation_graph_compiler." + name, "value": value}
    goals, producers, tasks = [], [], []
    for number in range(2):
        goal_id, producer_id = "goal:runtime:" + str(number), "producer:runtime:" + str(number)
        goals.append(record("obligation_graph_compiler.TypedPredicate", {"predicate_id": goal_id,
            "predicate_type": "reviewed_runtime_requirement", "subject_ref": "calc.py", "object_ref": "requirement:" + str(number),
            "property_id": "", "polarity": enum("PredicatePolarity", "positive"), "support": enum("SemanticSupport", "reviewed"),
            **{key: () for key in ("provenance_refs", "assumption_refs", "validation_requirement_refs", "proof_requirement_refs", "invalidation_selectors")}}))
        producers.append(record("obligation_graph_compiler.ProducerRule", {"producer_id": producer_id, "effect_predicate_ids": (goal_id,),
            **{key: () for key in ("required_predicate_ids", "task_candidate_ids", "provenance_refs", "assumption_refs",
                                  "validation_requirement_refs", "proof_requirement_refs", "invalidation_selectors")}, "executable": True}))
        tasks.append(record("obligation_graph_compiler.TaskCandidate", {"candidate_id": "task:runtime:" + str(number),
            "closes_obligation_ids": ("obligation:producer:" + producer_id + ":for:" + goal_id,), "producer_id": producer_id,
            "depends_on_candidate_ids": (), "provenance_refs": ()}))
    policy = {"acceptance_criteria": ("goal:runtime:0", "goal:runtime:1"), "evidence_terms": ("source:authored-runtime",),
        "trusted_assumptions": (), "supported_semantics": (), "satisfied_dependencies": (), "allowed_scopes": ("scope:calc.py",),
        "available_resource_classes": ("cpu",), "max_estimated_resource_cost": 1000000.0, "max_estimated_tokens": 1000000000,
        "max_estimated_runtime_seconds": 1000000.0, "min_novelty": 0.0, "require_validation": True, "require_proof": False}
    materials = {field: None for field in MATERIAL_FIELDS}
    materials.update(evidence_queries=(), current_facts=(), predicates=(), producers=tuple(producers), task_candidates=tuple(tasks),
        intent=record("obligation_graph_compiler.TypedIntent", {"intent_id": "intent:authored-source-successor", "desired_predicates": tuple(goals),
            "source_refs": ("source:authored-runtime",), "current_root_id": snapshot_cid, "metadata": {}}),
        frozen_goal=record("adaptive_planner.FrozenPlanningGoal", {"goal_id": "goal:runtime",
            "goal_content_id": structured({"authored_goal": "runtime"}), "repository_tree_id": snapshot_cid,
            "policy": record("plan_evaluator.EvidenceAwarePlanPolicy", policy)}),
        candidate_context={"repository_paths": ["calc.py"], "task_metadata": {"task:runtime:" + str(number):
            {"predicted_files": ["calc.py"], "scope_ids": ["scope:calc.py"], "resource_classes": ["cpu"]} for number in range(2)}},
        extra={"authored_review": "review:complete-runtime-requirements"})
    return {field: project(value) for field, value in materials.items()}


def qualification_frozen_goal_metadata(snapshot_cid):
    policy = {"acceptance_criteria": ["goal:runtime:0", "goal:runtime:1"], "evidence_terms": ["source:authored-runtime"],
        "trusted_assumptions": [], "supported_semantics": [], "satisfied_dependencies": [], "allowed_scopes": ["scope:calc.py"],
        "available_resource_classes": ["cpu"], "max_estimated_tokens": 1000000000,
        "max_estimated_runtime_milliseconds": 1000000000, "min_novelty_millionths": 0,
        "max_estimated_resource_cost_millionths": 1000000000000, "require_validation": True, "require_proof": False}
    return {"goal_id": "goal:runtime", "goal_content_id": structured({"authored_goal": "runtime"}),
            "repository_tree_id": snapshot_cid, "policy_digest": structured(policy), "policy": policy}


def decode_preimage(value, depth=0):
    need(depth <= 48, "semantic preimage nesting bound exceeded")
    if type(value) is not dict:
        need(type(value) in {str, int, bool, float, type(None)}, "unreviewed semantic scalar")
        if type(value) is float:
            need(math.isfinite(value), "nonfinite semantic preimage")
        return value
    if set(value) == {"$mapping"}:
        mapping = value["$mapping"]
        need(type(mapping) is dict and all(type(key) is str for key in mapping), "exact semantic mapping required")
        return {key: decode_preimage(item, depth + 1) for key, item in mapping.items()}
    if set(value) == {"$sequence", "items"}:
        need(value["$sequence"] in {"list", "tuple"} and type(value["items"]) is list, "exact semantic sequence required")
        return [decode_preimage(item, depth + 1) for item in value["items"]]
    if set(value) == {"$record", "fields"}:
        text(value["$record"], 512)
        need(value["$record"] in RECORD_FIELDS, "unreviewed semantic native record class")
        closed(value["fields"], {"$mapping"}, "semantic native record field tag")
        closed(value["fields"]["$mapping"], RECORD_FIELDS[value["$record"]], "semantic native record")
        decoded = decode_preimage(value["fields"], depth + 1)
        need(type(decoded) is dict, "semantic record fields must be a mapping")
        return {"native_type": value["$record"], "fields": decoded}
    if set(value) == {"$enum", "value"}:
        text(value["$enum"], 512)
        need(value["$enum"] in ENUM_VALUES and type(value["value"]) is str
            and value["value"] in ENUM_VALUES[value["$enum"]], "unreviewed semantic native enum class/value")
        return {"enum_type": value["$enum"], "value": value["value"]}
    raise SourceSuccessorAuditError("unsupported closed semantic preimage tag")


def verify_preimages(export):
    closed(export, {"schema", "entries", "capture", "execution_attestation"}, "semantic preimage export")
    need(export["schema"] == "source-successor-planning-semantic-preimages@1"
         and export["capture"] == "explicit_reconstruction_checked_against_native_consumed_material_binding"
         and export["execution_attestation"] is False and type(export["entries"]) is list and len(export["entries"]) == 3,
         "three explicitly reconstructed planning bindings required")
    results, identities = [], []
    for entry in export["entries"]:
        closed(entry, {"binding", "preimages"}, "semantic preimage entry")
        binding, preimages = entry["binding"], entry["preimages"]
        closed(binding, {"schema", "field_digests", "unsupported_fields", "reuse_supported", "opaque_material_nonce"}, "semantic material binding")
        need(binding["schema"] == "ipfs_accelerate_py/agent-supervisor/plan-create-semantic-material-binding@1"
             and binding["unsupported_fields"] == {} and binding["reuse_supported"] is True
             and binding["opaque_material_nonce"] == "", "complete supported semantic binding required")
        closed(preimages, MATERIAL_FIELDS, "complete semantic material field preimages")
        closed(binding["field_digests"], MATERIAL_FIELDS, "semantic field digests")
        decoded, total = {}, 0
        for name, preimage in preimages.items():
            raw = numerical_wire(preimage); total += len(raw)
            need(binding["field_digests"][name] == "sha256:" + sha(b"plan-create-semantic-input\n" + name.encode() + b"\n" + raw),
                 "semantic field preimage digest differs: " + name)
            decoded[name] = decode_preimage(preimage)
        need(total <= 4 * MIB, "combined native semantic field byte bound exceeded")
        identities.append(sha(numerical_wire(binding)))
        results.append({"binding": binding, "preimages": preimages, "decoded": decoded})
    need(identities == sorted(set(identities)), "semantic variants duplicated or reordered")
    return results


def expected_source_refs(delta):
    d = delta["value"]
    result = {"schema": "codebase-source-delta-advisory-refs@1", "artifact_cid": delta["artifact_cid"],
        **{field: d[field] for field in ("previous_head", "current_head", "previous_membership_cid",
            "current_membership_cid", "capture_policy", "coverage", "authority", "numerical_reuse", "model_advanced",
            "removal_scope", "physical_absence_verified")}, "ledger": []}
    for row in d["ledger"]:
        result["ledger"].append({**{field: row[field] for field in ("source_key", "classification",
            "source_bytes_comparison", "ast_identity_comparison")},
            **{side: None if row[side] is None else row[side]["member"] for side in ("previous", "current")}})
    return result


def verify_planning(result, authored, preimage_export, delta, current_manifest):
    closed(result, {"schema", "codebase_source_delta_advisory", "source_delta_advisory_cid", "repository_preview",
        "declared_requirement_ids", "declared_task_ids", "residual_requirements", "residual_task_ids", "current_facts",
        "removed_task_ids", "training_steps", "inference_calls", "source_property_solver_calls", "numerical_reuse",
        "model_advanced", "physical_absence_verified", "production_admitted", "worker_launched", "authority", "producer", "result_cid"}, "source delta planning result")
    need(result["schema"] == "supervisor-codebase-source-delta-plan-preview@1"
         and result["result_cid"] == structured({key: value for key, value in result.items() if key != "result_cid"}),
         "planning result CID differs")
    refs = expected_source_refs(delta)
    need(same(result["codebase_source_delta_advisory"], refs)
         and result["source_delta_advisory_cid"] == structured(refs), "complete body-free source delta planning refs differ")
    false_authority(result["authority"])
    for field in ("numerical_reuse", "model_advanced", "physical_absence_verified", "production_admitted", "worker_launched"):
        need(result[field] is False, "planning gained authority: " + field)
    for field in ("training_steps", "inference_calls", "source_property_solver_calls"):
        need(type(result[field]) is int and result[field] == 0, "planning non-numerical scope differs")
    need(result["current_facts"] == result["removed_task_ids"] == [], "planning produced facts or removed tasks")
    closed(authored, {"request", "materials"}, "authored planning export")
    request = authored["request"]
    closed(request, {"schema", "contract_version", "prompt_source_cid", "repository_id", "repository_root",
        "scope_paths", "dirty_tree_policy", "task_source_kind", "board_namespace", "alias_prefix", "roots", "budget",
        "required_analysis_operations", "optional_analysis_operations", "required_logic_families", "optional_logic_families",
        "fallback_policy", "supervisor_profile", "observe_roots", "redacted_source_metadata", "caller", "idempotency_key"},
        "authored native planning request")
    need(request["schema"] == "ipfs_accelerate_py/agent-supervisor/plan-create-request@1"
        and type(request["contract_version"]) is int and request["contract_version"] == 1
        and request["observe_roots"] is False, "authored native request schema differs")
    closed(request["budget"], {"schema", "contract_version", "max_goals", "max_tasks", "max_graph_depth", "max_output_paths",
        "max_ready_width", "max_repair_rounds", "max_scan_bytes", "max_analysis_operations", "max_evidence_items",
        "max_logic_families", "max_model_calls", "max_latency_ms", "max_provider_tokens", "max_cost_micros"}, "native planning budget")
    need(request["budget"]["schema"] == "ipfs_accelerate_py/agent-supervisor/plan-request-budget@1", "native budget schema differs")
    for field, value in request["budget"].items():
        if field != "schema":
            integer(value)
    closed(request["roots"], {"schema", "contract_version", "repository_id", "repository_root_cid", "dirty_worktree_root",
        "task_source_id", "task_source_revision", "policy_root", "intent_ir_root", "legal_ir_root", "security_ir_root",
        "program_root", "capability_catalog_root", "provider_catalog_root", "usage_policy_root", "configuration_root"},
        "native planning roots")
    need(request["roots"]["schema"] == "ipfs_accelerate_py/agent-supervisor/plan-authority-roots@1"
        and type(request["roots"]["contract_version"]) is int and request["roots"]["contract_version"] == 1
        and request["repository_id"] == request["roots"]["repository_id"] == delta["value"]["current_head"]["repository_id"],
        "authored native policy/source repository differs")
    closed(authored["materials"], {"schema", "scan_cid", "query_plan_cid", "evidence_bundle_cid", "obligation_graph_cid",
        "frozen_goal_cid", "has_model_provider", "has_admission_materials", "has_workflow_request", "evidence_adapter_slots",
        "current_roots", "extra_keys"}, "legacy authored material metadata")
    variants = verify_preimages(preimage_export)
    initial = [row for row in variants if row["decoded"]["scan"] is None and "codebase_source_delta_advisory" not in row["decoded"]["extra"]]
    injected = [row for row in variants if row["decoded"]["scan"] is None and "codebase_source_delta_advisory" in row["decoded"]["extra"]]
    final = [row for row in variants if row["decoded"]["scan"] is not None]
    need(len(initial) == len(injected) == len(final) == 1, "authored/source/full material variants differ")
    a, s, f = initial[0], injected[0], final[0]
    for field in MATERIAL_FIELDS - {"scan", "candidate_context", "extra"}:
        need(observation_same(a["preimages"][field], s["preimages"][field])
             and observation_same(a["preimages"][field], f["preimages"][field]), "authored planning meaning changed: " + field)
    need(observation_same(a["preimages"]["candidate_context"], s["preimages"]["candidate_context"])
         and same(s["decoded"]["extra"], {**a["decoded"]["extra"], "codebase_source_delta_advisory": refs}),
         "source-bound material changed unrelated authored values")
    need(observation_same(a["preimages"], qualification_authored_preimages(delta["value"]["current_head"]["snapshot_cid"])),
        "complete independently authored qualification meanings differ")
    metadata = authored["materials"]
    need(metadata["schema"] == "ipfs_accelerate_py/agent-supervisor/plan-create-materials@1"
        and all(metadata[field] == "" for field in ("scan_cid", "query_plan_cid", "evidence_bundle_cid", "obligation_graph_cid"))
        and all(metadata[field] is False for field in ("has_model_provider", "has_admission_materials", "has_workflow_request"))
        and metadata["evidence_adapter_slots"] == [] and metadata["current_roots"] == {}
        and metadata["extra_keys"] == sorted(a["decoded"]["extra"])
        and metadata["frozen_goal_cid"] == named_digest("frozen-goal",
            qualification_frozen_goal_metadata(delta["value"]["current_head"]["snapshot_cid"])), "legacy authored material metadata differs")
    intent = a["decoded"]["intent"]
    need(intent["native_type"] == "ipfs_accelerate_py.agent_supervisor.planning.obligation_graph_compiler.TypedIntent",
         "independently authored typed intent required")
    intent_fields = intent["fields"]
    goals = intent_fields["desired_predicates"]
    requirements = [goal["fields"]["predicate_id"] for goal in goals]
    tasks = [task["fields"]["candidate_id"] for task in a["decoded"]["task_candidates"]]
    need(len(requirements) == len(tasks) == 2 and len(set(requirements)) == len(set(tasks)) == 2
         and intent_fields["current_root_id"] == delta["value"]["current_head"]["snapshot_cid"]
         and same(result["declared_requirement_ids"], requirements) and same(result["declared_task_ids"], tasks)
         and same(result["residual_task_ids"], tasks)
         and same(result["residual_requirements"], [{"predicate_id": identity, "status": "runtime_behavior_unresolved"} for identity in requirements]),
         "complete authored requirement/task meanings and residual population differ")
    native = result["repository_preview"]
    closed(native, {"schema", "preview", "input_snapshot", "structural_context", "structural_context_cid", "observed_facts_supplied", "model_calls",
        "source_semantics_verified", "proof_authority", "production_admitted", "worker_launched", "execution_authority", "completion_authority"}, "native repository preview")
    need(native["schema"] == "supervisor-repository-plan-preview@1" and type(native["observed_facts_supplied"]) is int
         and native["observed_facts_supplied"] == 0 and type(native["model_calls"]) is int and native["model_calls"] == 0,
         "native model-off preview scope differs")
    need(all(native[field] is False for field in ("source_semantics_verified", "proof_authority", "production_admitted", "worker_launched", "execution_authority", "completion_authority")),
         "native repository preview gained authority")
    context = native["structural_context"]
    expected_context = {"schema": "supervisor-structural-codebase-context@1",
        "head": {key: value for key, value in delta["value"]["current_head"].items() if key != "schema"},
        "semantic_state_cid": current_manifest["semantic_state"]["state_cid"], "coverage": current_manifest["coverage"],
        "authority": "structural_only", "source_semantics_verified": False, "proof_authority": False,
        "execution_authority": False, "completion_authority": False}
    need(same(context, expected_context) and native["structural_context_cid"] == structured(context), "native current structural context differs")
    context_cid = native["structural_context_cid"]
    need(same(f["decoded"]["scan"], {"scan_cid": context_cid, "structural_codebase": context})
         and same(f["decoded"]["candidate_context"], {**s["decoded"]["candidate_context"], "structural_codebase_context_cid": context_cid, "structural_codebase": context})
         and same(f["decoded"]["extra"], {**s["decoded"]["extra"], "structural_codebase_context_cid": context_cid, "structural_codebase": context, "root_observation_profile": "repository-live-root-observation@1"}),
         "complete consumed structural/source material injection differs")
    snapshot = native["input_snapshot"]
    closed(snapshot, {"schema", "request_cid", "repository_id", "repository_root", "scope_paths", "roots", "budget",
        "required_analysis_operations", "optional_analysis_operations", "required_logic_families", "optional_logic_families",
        "fallback_policy", "supervisor_profile", "board_namespace", "alias_prefix", "task_source_kind", "dirty_tree_policy",
        "bounds_digest", "snapshot_cid", "material_binding"}, "native consumed planning snapshot")
    need(snapshot.get("schema") == "ipfs_accelerate_py/agent-supervisor/plan-create-input-snapshot@2"
         and snapshot["snapshot_cid"] == named_digest("plan-create-input-snapshot", {key: value for key, value in snapshot.items() if key != "snapshot_cid"})
         and same(snapshot["material_binding"], f["binding"]) and snapshot["request_cid"] == structured(request),
         "native consumed full material snapshot identity differs")
    for field in ("repository_id", "repository_root", "scope_paths", "roots", "budget", "required_analysis_operations", "optional_analysis_operations",
        "required_logic_families", "optional_logic_families", "fallback_policy", "supervisor_profile", "board_namespace", "alias_prefix", "task_source_kind", "dirty_tree_policy"):
        need(same(snapshot[field], request[field]), "native snapshot/request binding differs: " + field)
    need(request["budget"]["max_model_calls"] == 0 and request["roots"]["repository_root_cid"] == delta["value"]["current_head"]["snapshot_cid"]
         and request["roots"]["dirty_worktree_root"] == delta["value"]["current_head"]["snapshot_cid"]
         and request["roots"]["program_root"] == context["semantic_state_cid"], "authored current source/policy roots differ")
    bounds = {field: request[field] for field in ("budget", "required_analysis_operations", "optional_analysis_operations",
        "required_logic_families", "optional_logic_families", "fallback_policy", "scope_paths")}
    need(snapshot["bounds_digest"] == named_digest("plan-create-bounds", bounds), "native planner bounds digest differs")
    receipt = native["preview"]
    closed(receipt, {"schema", "interface", "request_cid", "input_snapshot_cid", "mode", "verdict", "roots", "stage_results",
        "scan_cid", "query_plan_cid", "evidence_bundle_cid", "obligation_graph_cid", "candidate_portfolio_cid", "critique_cid",
        "admission_receipt_cid", "execution_plan_cid", "plan_root_cid", "rejection_reasons", "artifact_refs", "read_only",
        "wrote_effects", "compatibility_alias", "receipt_cid"}, "native plan preview receipt")
    need(receipt.get("schema") == "ipfs_accelerate_py/agent-supervisor/plan-create-preview-receipt@1"
         and receipt["interface"] == "PlanCreateService@1" and receipt["mode"] == "deterministic"
         and receipt["verdict"] in {"review_only", "blocked", "rejected"}
         and receipt["read_only"] is True and receipt["wrote_effects"] == []
         and receipt["request_cid"] == snapshot["request_cid"] and receipt["input_snapshot_cid"] == snapshot["snapshot_cid"]
         and same(receipt["roots"], request["roots"])
         and receipt["receipt_cid"] == named_digest("plan-create-preview-receipt", {key: value for key, value in receipt.items() if key != "receipt_cid"}),
         "native read-only preview receipt identity differs")
    stages = receipt["stage_results"]
    need([row["stage"] for row in stages] == ["scan", "query", "evidence", "obligation", "candidate", "critique", "admission", "parallel_plan"], "native planning stage order differs")
    for row in stages:
        closed(row, {"schema", "stage", "artifact_cid", "passed", "blockers", "detail_ids", "message", "result_cid"},
            "native planning stage result")
        need(row["schema"] == "ipfs_accelerate_py/agent-supervisor/plan-create-stage-result@1", "native stage schema differs")
        need(type(row["passed"]) is bool and row["result_cid"] == named_digest("plan-create-stage-result", {key: value for key, value in row.items() if key != "result_cid"}),
             "native stage result identity differs")
    need(all(row["passed"] is True for row in stages if row["stage"] in {"obligation", "candidate"}), "native declared task compilation did not pass")
    return {"result_cid": result["result_cid"], "input_snapshot_cid": snapshot["snapshot_cid"],
        "receipt_cid": receipt["receipt_cid"], "requirements": requirements, "tasks": tasks,
        "complete_semantic_preimages_verified": True, "full_authored_fixture_meanings_verified": True,
        "native_record_and_enum_whitelist_verified": True, "legacy_frozen_goal_identity_verified": True,
        "preimage_capture_scope": "explicit reconstruction bound to recorded native consumed snapshot; not execution attestation",
        "observed_facts": 0, "removed_tasks": 0, "proof_authority": False}


def audit_producers(reader, archive):
    selected = reader.json("generation-inputs.json")
    closed(selected, {"schema", "files", "scope", "execution_attestation"}, "selected producer exports")
    need(selected["schema"] == "codebase-source-successor-selected-producers@1"
         and selected["scope"] == "listed_local_files_only" and selected["execution_attestation"] is False,
         "selected producer scope differs")
    rows = selected["files"]
    need(type(rows) is list and 23 <= len(rows) <= 64, "bounded complete selected producer population required")
    names, observed = [], {}
    files = {row["path"]: row for row in archive["files"] if row["kind"] == "file"}
    for row in rows:
        closed(row, {"name", "path", "copy", "bytes", "sha256"}, "selected producer")
        text(row["name"], 512); text(row["path"]); integer(row["bytes"], 4 * MIB, 1)
        raw = reader.raw(row["copy"], 4 * MIB)
        need(len(raw) == row["bytes"] and sha(raw) == row["sha256"]
             and files[row["copy"]]["sha256"] == row["sha256"], "frozen selected producer bytes differ")
        names.append(row["name"]); observed[row["name"]] = row["sha256"]
    need(names == sorted(set(names)), "selected producers duplicated or reordered")
    return observed



def stored_target_binding(target):
    """Join recorded source bytes/AST/manifests without redoing native lowering."""
    closed(target, {"schema_version", "domain_id", "source_digest", "projections", "unsupported", "validation", "ready_for_training",
        "qualification_gaps", "qualified", "admitted", "formalized"}, "stored structural feature target")
    need(target["schema_version"] == "autoencoder-domain-targets/v1" and target["domain_id"] == "codebase_ir"
        and target["ready_for_training"] is True and target["unsupported"] == []
        and all(target[field] is False for field in ("qualified", "admitted", "formalized")), "stored feature target schema/authority differs")
    need(type(target["validation"]) is list and 1 <= len(target["validation"]) <= 32, "bounded recorded target validators required")
    rows = [row for row in target["validation"] if type(row) is dict and row.get("validator_id") == "codebase_ir.exact_native_target_replay@1"]
    need(len(rows) == 1 and rows[0].get("status") == "passed", "one recorded native target validator required")
    details = rows[0]["details"]
    binding = details["source_binding"]
    closed(binding, {"schema", "head", "repository_id", "path", "source_key", "source_revision", "source_cid", "content_sha256",
        "entry", "unit", "ast_cid"}, "stored feature source binding")
    need(binding["schema"] == "codebase-ir-feature-source-binding@1", "stored source binding schema differs")
    head_shape(binding["head"]); entry_shape(binding["entry"])
    closed(binding["unit"], {"schema", "source_key", "entry_cid", "ast_cid", "parse_status"}, "stored feature source unit")
    need(binding["repository_id"] == binding["head"]["repository_id"] and binding["path"] == binding["entry"]["path"]
        and binding["source_key"] == "raw:" + binding["entry"]["raw_path_hex"] == binding["unit"]["source_key"]
        and binding["source_revision"] == "snapshot:" + binding["head"]["snapshot_cid"]
        and binding["source_cid"] == binding["entry"]["source_cid"] and binding["ast_cid"] == binding["unit"]["ast_cid"]
        and binding["unit"]["schema"] == "codebase-ir-structural-unit@1" and binding["unit"]["parse_status"] == "ok"
        and binding["unit"]["entry_cid"] == binding["entry"]["entry_cid"], "stored feature source membership differs")
    source_hex = details["source_bytes_hex"]
    need(type(source_hex) is str and len(source_hex) <= 2 * 65536 and re.fullmatch(r"(?:[0-9a-f]{2})*", source_hex),
        "canonical bounded recorded target source bytes required")
    raw = bytes.fromhex(source_hex)
    need(len(raw) == binding["entry"]["size_bytes"] and sha(raw) == binding["content_sha256"]
        and cid(raw, source=True) == binding["source_cid"], "stored target full source bytes differ")
    need(structured(details["captured_ast"]) == binding["ast_cid"]
        and structured(details["manifest"]) == binding["head"]["manifest_cid"]
        and structured(details["publication_receipt"]) == binding["head"]["receipt_cid"]
        and same(receipt_head(details["publication_receipt"]), binding["head"]), "stored target AST/manifest/receipt identity differs")
    verify_ast_binding(details["captured_ast"], binding["entry"], binding["unit"], binding["head"])
    need(target["source_digest"] == sha(numerical_wire({"target_schema": details["target_schema"],
        "source_binding": binding, "authored_contracts": details["authored_contracts"]})), "stored target source digest differs")
    manifest = details["manifest"]
    need([row for row in manifest["snapshot"]["entries"] if row["entry_cid"] == binding["entry"]["entry_cid"]] == [binding["entry"]]
        and [row for row in manifest["units"] if row["source_key"] == binding["source_key"]] == [binding["unit"]],
        "stored feature target omitted from captured source manifest")
    for field in ("source_semantics_verified", "proof_authority", "execution_authority", "completion_authority", "semantic_formula_decoder"):
        need(details[field] is False, "stored feature target gained authority")
    return binding


def training_request(saved):
    provenance, configuration = saved["report"]["codebase_provenance"], saved["report"]["configuration"]
    return {"schema": "codebase-source-feature-training@1", "head": provenance["head"], "selections": provenance["selections"],
        "parent_version_id": provenance["parent_version_id"], "contract_sha256": provenance["contract_sha256"],
        **{field + "_sha256": sha(numerical_wire(provenance[field])) for field in ("training_targets", "tuning_targets", "canary_targets")},
        "configuration": {field: configuration[field] for field in ("epochs", "learning_rate", "seed")},
        "implementation": provenance["implementation"]}


def verify_model_lineage(root_checked, child_checked):
    """Intrinsic stored continuation/cohort closure; no optimizer execution."""
    root, child = root_checked["saved"], child_checked["saved"]
    rp, cp = (item["report"]["codebase_provenance"] for item in (root, child))
    for saved, provenance in ((root, rp), (child, cp)):
        for field in ("training_targets", "tuning_targets", "canary_targets", "replay_targets"):
            values = provenance[field]
            need(type(values) is list and 1 <= len(values) <= 32, "bounded complete stored training/evaluation population required")
            for target in values:
                binding = stored_target_binding(target)
                need(binding["head"] in [rp["head"], cp["head"]], "stored target is outside selected parent/child source heads")
        need(saved["report"]["training_targets_sha256"] == sha(numerical_wire(provenance["training_targets"]))
            and saved["report"]["tuning_targets_sha256"] == saved["state"]["tuning_targets_sha256"] == sha(numerical_wire(provenance["tuning_targets"]))
            and saved["state"]["optimizer_config"]["learning_rate"] == saved["report"]["configuration"]["learning_rate"]
            and saved["report"]["codebase_request_sha256"] == sha(numerical_wire(training_request(saved))),
            "stored complete training/evaluation request/report/optimizer binding differs")
    need(observation_same(root["contract"], child["contract"]) and observation_same(root["feature_space"], child["feature_space"])
        and child["report"]["base_state_sha256"] == sha(numerical_wire(root["state"]))
        and cp["continuation"] == "exact_frozen_basis_adam_resume", "complete parent state/frozen numerical basis continuation differs")
    for field in ("selections", "tuning_targets", "canary_targets", "replay_targets"):
        need(observation_same(rp[field], cp[field]), "frozen parent/child source cohort differs: " + field)
    need(rp["continuation"] == "fresh_feature_basis" and rp["ancestral_training"] == []
        and root["report"]["base_state_sha256"] is None and rp["canary_monitoring"]["before"] is None
        and observation_same(rp["training_targets"], rp["replay_targets"]), "fresh root inherited training/evaluation state")
    identities = []
    for target in rp["training_targets"]:
        binding = stored_target_binding(target)
        identities.append({"repository_id": binding["head"]["repository_id"], "path": binding["path"], "source_digest": binding["content_sha256"]})
    expected_history = sorted({sha(numerical_wire(row)): row for row in [*rp["ancestral_training"], *identities]}.values(),
        key=lambda row: (row["repository_id"], row["path"], row["source_digest"]))
    need(same(cp["ancestral_training"], expected_history), "complete parent source-byte training history differs")
    need(same(rp["selections"], [{"contracts": [], "path": "calc.py", "role": "train"},
        {"contracts": [], "path": "canary.py", "role": "canary"}, {"contracts": [], "path": "tune.py", "role": "tune"}]),
        "explicit complete three-split stored fixture selection differs")
    need(len(rp["training_targets"]) == 1 and len(cp["training_targets"]) == 2
        and all(len(provenance[field]) == 1 for provenance in (rp, cp) for field in ("tuning_targets", "canary_targets", "replay_targets")),
        "complete stored root/current-plus-replay/evaluation fixture populations differ")
    need(stored_target_binding(cp["training_targets"][0])["head"] == cp["head"]
        and observation_same(cp["training_targets"][1:], rp["replay_targets"]), "current training and immutable replay union differs")
    return {"parent_state_and_frozen_basis_verified": True, "complete_stored_cohorts_verified": True,
        "ancestral_source_byte_history_verified": True, "optimizer_independently_replayed": False,
        "source_target_semantics_independently_relowered": False}


def verify_registry_lineage(registry, selection, models):
    """Reconstruct retained version/completed-run identities; no SQL or owner."""
    schema = "ipfs_datasets_py/autoencoder-control@1"
    chosen = selection["value"]
    versions = registry["versions"]
    need(type(versions) is list and len(versions) == 2 and all(type(row) is list and len(row) == 5 for row in versions),
        "two complete retained native version rows required")
    by_id = {row[0]: row for row in versions}
    need(len(by_id) == 2, "duplicate native selected model version")
    child_metadata = None
    for label, model, parent in (("root", chosen["previous_model"], None), ("child", chosen["model"], chosen["previous_model"]["version_id"])):
        row = by_id[model["version_id"]]
        artifact, metadata = parse(row[3]), parse(row[4])
        expected_id = "sha256:" + sha(wire({"schema": schema, "variant_id": row[1], "artifact": artifact,
            "metadata": metadata, "parent_version_id": row[2]}))
        need(row[0] == expected_id and row[1] == model["variant_id"] == "modality-" + model["contract_sha256"]
            and row[2] == parent and same(artifact, model["artifact"]), "native selected version content identity differs")
        flags = {"qualified": False, "admitted": False, "formalized": False, "promotion_performed": False}
        if label == "root":
            need(same(metadata, {"contract_sha256": model["contract_sha256"], "training_purpose": "feature_pretraining", **flags}),
                "native bootstrap model metadata differs")
        else:
            closed(metadata, {"producer_run", "attempt", "result"}, "native child metadata")
            expected_result = {"contract_sha256": model["contract_sha256"], "feature_space_sha256": model["feature_space_sha256"],
                "state_sha256": model["state_sha256"], "report_sha256": models["child"]["checkpoint_state"]["report_sha256"],
                "training_purpose": "feature_pretraining", **flags}
            need(same(metadata["result"], expected_result), "native completed-run full model result differs")
            child_metadata = metadata
    runs = registry["runs"]
    need(type(runs) is list and len(runs) == 1 and type(runs[0]) is list and len(runs[0]) == 9,
        "one complete child native run required")
    run = runs[0]
    spec, lease, result = parse(run[3]), parse(run[7]), parse(run[8])
    need(run[0] == child_metadata["producer_run"] == "codebase-source:child"
        and run[1] == chosen["model"]["variant_id"] and run[2] == chosen["previous_model"]["version_id"]
        and run[4] == "completed" and observation_same(spec, training_request(models["child"]["saved"]))
        and same(result, child_metadata["result"]), "retained native completed child request/result differs")
    integer(run[5], minimum=1); integer(run[6], minimum=1); integer(child_metadata["attempt"], minimum=1)
    closed(lease, {"run_id", "attempt", "worker_id", "owner_generation", "fence", "expires_at"}, "historical completed native lease")
    need(lease["run_id"] == run[0] and type(lease["attempt"]) is int and lease["attempt"] == run[5] == child_metadata["attempt"]
        and type(lease["fence"]) is int and lease["fence"] == run[6]
        and type(lease["expires_at"]) in {int, float} and math.isfinite(lease["expires_at"]), "native historical attempt/fence/expiry differs")
    integer(lease["owner_generation"], minimum=1); text(lease["worker_id"], 256)
    completions = []
    for row in registry["operations"]:
        need(type(row) is list and len(row) == 3, "closed retained native operation row required")
        receipt = parse(row[2])
        if receipt.get("command") == "CompleteRun" and receipt.get("run_id") == run[0]:
            completions.append((row, receipt))
    need(len(completions) == 1, "one exact durable child completion operation required")
    row, receipt = completions[0]
    closed(receipt, {"schema", "operation_id", "command", "admitted", "run_id", "version_id", "event_id", "status", "promoted"},
        "native completed-run receipt")
    need(receipt["schema"] == schema and receipt["operation_id"] == row[0] and receipt["command"] == "CompleteRun"
        and receipt["run_id"] == run[0] and receipt["version_id"] == chosen["model"]["version_id"]
        and receipt["status"] == "completed" and receipt["admitted"] is False and receipt["promoted"] is False,
        "native durable child completion receipt differs")
    payload = {"command": "CompleteRun", "payload": {"lease": lease, "artifact": chosen["model"]["artifact"], "result": result}}
    observation_same(payload, payload)
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    need(row[1] == sha(payload_bytes), "native durable completion command digest differs")
    event_id = "sha256:" + sha(wire({"operation_id": row[0], "kind": "candidate_durable", "consumer": "ducklake"}))
    events = [event for event in registry["events"] if type(event) is list and len(event) == 4 and event[0] == event_id]
    need(receipt["event_id"] == event_id and len(events) == 1 and events[0][1] == "candidate_durable"
        and same(parse(events[0][2]), {"run_id": run[0], "version_id": chosen["model"]["version_id"]}),
        "native durable completed child event identity differs")
    return {"complete_version_identities_verified": True, "registered_result_metadata_verified": True,
        "child_completed_request_lease_command_event_verified": True,
        "historical_lease_compared_to_current_owner_generation": False, "registry_owner_opened": False}


def audit_models(reader, selection, previous_record, current_record, checkpoint_states):
    selected = selection["value"]
    models = {}
    for label, model, head, record, parent in (
        ("root", selected["previous_model"], selected["previous_head"], previous_record, None),
        ("child", selected["model"], selected["current_head"], current_record, selected["previous_model"]["version_id"]),
    ):
        digest = model["artifact"]["sha256"]
        raw = reader.raw("private/model-artifacts/" + digest[:2] + "/" + digest, 16 * MIB)
        checked = verify_model(model, raw, head)
        verify_training_record(record, model, checked["saved"], head, parent)
        need(checked["saved"]["report"]["codebase_provenance"]["parent_version_id"] == parent
             and same(checked["checkpoint_state"], checkpoint_states[label]), "full lineage state/parent differs")
        implementation_shape(checked["saved"]["report"]["codebase_provenance"]["implementation"])
        models[label] = checked
    need(selected["previous_training_record_cid"] == previous_record["artifact_cid"]
         and selected["training_record_cid"] == current_record["artifact_cid"], "successor selected training-record identities differ")
    need(models["root"]["checkpoint_state"]["completed_epochs"] == 1
         and models["child"]["checkpoint_state"]["completed_epochs"] == 2, "explicit parent/child setup epochs differ")
    verify_model_lineage(models["root"], models["child"])
    return models


def resources_zero(reader, result):
    final = result["final_resources"]
    closed(final, {"active_lease_count", "waiting_request_count"}, "final native resources")
    need(all(type(amount) is int and amount == 0 for amount in final.values()), "native resources did not drain")
    state = reader.json("resource-admission.json", 4 * MIB)
    need(state.get("leases") == {} and state.get("waiters") == {}
         and observation_same(state.get("config"), result["scheduler_configuration"]), "closed scheduler state or configuration differs")
    return {"final_resources": final, "leases": 0, "waiters": 0,
            "kernel_memory_enforcement_claimed": False}


def verify_costs(result):
    elapsed = result["recorded_seconds"]
    need(type(elapsed) in {int, float} and math.isfinite(elapsed) and elapsed >= 0, "finite actual native cost required")
    phases = result["phases"]
    need(type(phases) is list and len(phases) <= 64, "bounded native phase ledger required")
    for row in phases:
        need(type(row) is dict and row.get("status") in {"completed", "failed"}
             and type(row.get("elapsed_seconds")) in {int, float} and math.isfinite(row["elapsed_seconds"])
             and row["elapsed_seconds"] >= 0, "closed actual native phase cost required")
    attempts = result["setup_training_attempts"]
    need(type(attempts) is list and len(attempts) == 2 and [row.get("name") for row in attempts] == ["root", "child"], "explicit two setup training attempts required")
    for row in attempts:
        need(type(row.get("requested_epochs")) is int and row["requested_epochs"] == 1
             and type(row.get("actual_completed_epochs")) is int and row["actual_completed_epochs"] == 1
             and row["unknown_actual_epochs_on_failure"] is False, "explicit actual setup epoch accounting differs")
    need(type(result["new_fitting_epochs"]) is int and result["new_fitting_epochs"] == 2
         and type(result["training_attempts_after_setup"]) is int and result["training_attempts_after_setup"] == 0,
         "coordinator/receiving post-setup fitting guard differs")
    for field in ("inference_attempts_during_selection_or_cold_receiving", "source_property_solver_calls"):
        need(type(result[field]) is int and result[field] == 0, "native non-numerical receiving guard differs: " + field)
    deadlines = result.get("operation_deadline_seconds")
    if deadlines is not None:
        closed(deadlines, {"default_selection", "reference_selection", "inference_page", "planning_adapter",
            "existing_repository_preview", "reference_override_is_qualification_only"}, "qualification operation deadlines")
        need(all(type(deadlines[field]) is int and deadlines[field] == amount for field, amount in
            (("default_selection", 120), ("inference_page", 600), ("planning_adapter", 120), ("existing_repository_preview", 90)))
            and type(deadlines["reference_selection"]) in {int, float} and deadlines["reference_selection"] == 600
            and deadlines["reference_override_is_qualification_only"] is True,
            "qualification-only reference deadline declaration differs")
    else:
        need(result["qualified"] is False, "qualified native run must declare operation deadlines")
    return {"recorded_native_seconds": elapsed, "phases": phases, "explicit_setup_training": attempts,
        "new_fitting_epochs": 2, "training_attempts_after_setup": 0,
        "inference_attempts_during_selection_or_cold_receiving": 0, "source_property_solver_calls": 0,
        "operation_deadline_seconds": deadlines,
        "phases_are_cooperative_not_kernel_deadlines": True}


def compare_owners(reader, result, selection, models):
    before, warm, cold = (reader.json(name) for name in ("owners-before.json", "owners-after-warm.json", "owners-after-cold.json"))
    need(observation_same(before, warm), "warm source/model native owners changed")
    expected = parse(numerical_wire(before))
    meta = expected["registry"]["meta"]
    need(type(meta) is list and len(meta) == 1 and len(meta[0]) == 6
         and type(meta[0][5]) is int and meta[0][5] >= 1, "native registry generation baseline differs")
    meta[0][5] += 1
    need(observation_same(cold, expected), "cold changes exceed one explicit owner reopen")
    for owner in (before, warm, cold):
        closed(owner, {"source", "registry", "model_artifacts"}, "native source/model owner projection")
        need(same(owner["source"]["current_head"], result["current_head"])
             and set(owner["source"]["tables"]) == TABLE_NAMES, "native full13 source table/head projection differs")
        for table in owner["source"]["tables"].values():
            closed(table, {"rows", "sha256"}, "native source table fingerprint")
            integer(table["rows"])
            need(type(table["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", table["sha256"]), "native table fingerprint SHA differs")
        need(set(owner["registry"]) == {"meta", "operations", "variants", "versions", "heads", "runs", "events", "outbox"}
             and owner["registry"]["heads"] == [] and len(owner["registry"]["versions"]) == 2,
             "private two-version registry or non-promotion binding differs")
        versions = {row[0]: row for row in owner["registry"]["versions"]}
        for model, parent in ((selection["value"]["previous_model"], None), (selection["value"]["model"], selection["value"]["previous_model"]["version_id"])):
            row = versions[model["version_id"]]
            need(len(row) == 5 and row[1] == model["variant_id"] and row[2] == parent
                 and same(parse(row[3]), model["artifact"]), "native registered direct-child row differs")
    for row in before["model_artifacts"]:
        raw = reader.raw("private/model-artifacts/" + row["path"], 16 * MIB)
        need(len(raw) == row["bytes"] and sha(raw) == row["sha256"], "original numerical artifact changed")
    after = reader.json("checkpoint-states-after.json")
    need(same(after, result["checkpoint_states"]), "cold full checkpoint/Adam projection differs")
    for row in reader.json("source-artifacts-before.json"):
        raw = reader.raw("private/source-artifacts/" + row["path"], 16 * MIB)
        need(len(raw) == row["bytes"] and sha(raw) == row["sha256"], "original source/AST/training CAS bytes changed")
    registry_closure = verify_registry_lineage(before["registry"], selection, models)
    return {"warm_logical_owners_unchanged": True, "cold_registry_generation_delta": 1,
        "intrinsic_registry_closure": registry_closure,
        "model_artifacts_unchanged": True, "checkpoint_states_unchanged": True,
        "reopen_scope": "new source/model connections in same native process; not fresh-process qualification"}


def audit_closed_archive(namespace, *, seconds=120):
    started = time.monotonic()
    reader = Reader(namespace, seconds)
    before = reader.whole_archive()
    report = {"schema": SCHEMA, "qualified": False, "preserved": False, "namespace": str(reader.root), "errors": [],
        "native_owners_opened": False, "sql_executed": False, "git_executed": False, "primary_writes": False,
        "proof_authority": False, "numerical_execution_independently_reperformed": False,
        "process_origin_attested": False, "cuda_qualified": False, "384d_qualified": False,
        "complete_scan_qualified": False, "worker_admission_qualified": False,
        "mutable_working_checkout_qualified": False}
    try:
        native = reader.json("result.json")
        need(native.get("schema") == "codebase-source-successor-native-qualification@1"
             and native.get("scope") == "fresh_300_member_successor_and_two_32_member_cpu8d_prefix_pages"
             and type(native.get("qualified")) is bool, "closed native successor result schema differs")
        integer(native["pid"], minimum=1)
        native_raw = reader.raw("result.json")
        report["audited_result"] = {"path": "result.json", "bytes": len(native_raw), "sha256": sha(native_raw)}
        report["costs"] = verify_costs(native)
        report["resources"] = resources_zero(reader, native)
        producers = audit_producers(reader, before)
        report["selected_producers"] = producers
        previous = export_value(reader, "previous-manifest.json")
        current = export_value(reader, "current-manifest.json")
        d = export_value(reader, "source-delta.json")
        report["source_delta"] = verify_source_delta(d, previous["value"], current["value"],
            reader.json("previous-publication-receipt.json"), reader.json("current-publication-receipt.json"), get_artifact=reader.cas)
        coverage = report["source_delta"]["coverage"]
        need(coverage["previous_entries"] == coverage["current_entries"] == 300 and coverage["union_entries"] == 302
             and same(coverage["classifications"], {"retained": 297, "changed": 1, "added": 2, "removed": 2})
             and same(native["coverage"], coverage) and same(native["previous_head"], d["value"]["previous_head"])
             and same(native["current_head"], d["value"]["current_head"]), "actual complete300 source transition differs")
        implementation_shape(d["value"]["implementation"], producers)
        selection = export_value(reader, "successor-selection.json")
        root = export_value(reader, "scan-root.json")
        verify_selection(selection, d, root, selection["value"]["previous_model"])
        verify_root(root, d, selection["value"]["model"])
        previous_record, current_record = export_value(reader, "root-training-record.json"), export_value(reader, "child-training-record.json")
        models = audit_models(reader, selection, previous_record, current_record, native["checkpoint_states"])
        for item in (selection["value"]["implementation"], root["value"]["implementation"],
            models["root"]["saved"]["report"]["codebase_provenance"]["implementation"], models["child"]["saved"]["report"]["codebase_provenance"]["implementation"]):
            implementation_shape(item, producers)
        report["selected_models"] = {label: item["checkpoint_state"] for label, item in models.items()}
        report["selected_model_identities"] = {"root": selection["value"]["previous_model"], "child": selection["value"]["model"]}
        report["stored_model_lineage"] = verify_model_lineage(models["root"], models["child"])
        report["intrinsic_registry_closure"] = verify_registry_lineage(reader.json("owners-before.json")["registry"], selection, models)
        need(native["parent_version_id"] == selection["value"]["previous_model"]["version_id"]
            and native["child_version_id"] == selection["value"]["model"]["version_id"]
            and [row["version_id"] for row in native["setup_training_attempts"]] == [native["parent_version_id"], native["child_version_id"]],
            "explicit setup attempts are not the selected registered model lineage")
        report["default_selection_integrity"] = {"selection_cid": selection["artifact_cid"], "root_cid": root["artifact_cid"],
            "complete_members": 300, "direct_child_joined": True, "current_eligibility_attested": False}
        # A failed execution retains valid inert setup/default closures without
        # promoting them to a successful overall qualification.
        need(native["qualified"] is True, "native qualification failed: " + str(native.get("error_type")) + ": " + str(native.get("error")))
        for field in ("complete_scan_qualified", "worker_admission_qualified", "cuda_qualified", "384d_qualified", "proof_authority",
            "production_default_activated", "repository_code_executed", "numerical_reuse", "model_head_promoted"):
            need(native[field] is False, "native successor acquired unsupported scope: " + field)
        for field in ("cold_receiving_verified", "native_owner_unchanged_except_explicit_reopen", "original_source_artifacts_preserved", "selected_producers_unchanged"):
            need(native[field] is True, "native ending condition missing: " + field)
        for field, expected in (("source_owner_reopens", 1), ("model_owner_reopens", 1), ("new_scan_pages_created", 2), ("inherited_scan_pages", 0), ("inherited_setup_epochs", 0)):
            need(type(native[field]) is int and native[field] == expected, "native actual work/reopen accounting differs: " + field)
        reference = export_value(reader, "reference-successor-selection.json")
        reference_root = export_value(reader, "reference-scan-root.json")
        verify_selection(reference, d, reference_root, selection["value"]["previous_model"])
        verify_root(reference_root, d, selection["value"]["model"])
        need(root["value"]["optimized"] is True and reference_root["value"]["optimized"] is False
             and same({key: value for key, value in root["value"].items() if key != "optimized"},
                      {key: value for key, value in reference_root["value"].items() if key != "optimized"}), "default/opt-out root outcomes differ")
        optimized_page, reference_page = export_value(reader, "prefix-page.json", raw=True), export_value(reader, "reference-prefix-page.json", raw=True)
        report["fresh_prefix_pages"] = [verify_prefix_page(optimized_page, root), verify_prefix_page(reference_page, reference_root)]
        for field in ("entries", "coverage", "inference"):
            need(observation_same(optimized_page["value"][field], reference_page["value"][field]), "default/opt-out numerical prefix outcomes differ: " + field)
        need(same(native["prefix_coverage"], optimized_page["value"]["coverage"]), "native prefix coverage differs")
        worker_name = "ipfs_datasets_py.optimizers.logic_theorem_optimizer.codebase_inventory_resume_worker"
        need(all(page["worker_receipt"]["worker_sha256"] == producers[worker_name] for page in report["fresh_prefix_pages"]), "fresh page worker is not selected-source pinned")
        planning = reader.json("planning-preview.json")
        preimages = reader.json("planning-semantic-preimages.json", 16 * MIB)
        report["planning"] = verify_planning(planning, reader.json("authored-planning-inputs.json"), preimages, d, current["value"])
        cold_planning, cold_preimages = reader.json("cold-planning-preview.json"), reader.json("cold-planning-semantic-preimages.json", 16 * MIB)
        need(same(cold_planning, planning) and observation_same(cold_preimages, preimages), "cold planning differs from retained identical preimages/result")
        verify_planning(cold_planning, reader.json("authored-planning-inputs.json"), cold_preimages, d, current["value"])
        report["planning"]["cold_preimage_scope"] = "same warm reconstructed preimages reused against exactly identical cold consumed binding"
        producer = planning["producer"]
        need(producer["module"] == "ipfs_accelerate_py.agent_supervisor.planning.codebase_source_delta_context"
             and producer["sha256"] == producers[producer["module"]]
             and producer["scope"] == "listed_local_file_only_not_execution_attestation", "planning consumer producer pin differs")
        report["owners"] = compare_owners(reader, native, selection, models)
        controls = native["controls"]
        need(type(controls) is list and [row["name"] for row in controls] == ["old_model_for_new_head", "precancelled_selection_receiving"]
             and all(row["refused"] is True for row in controls)
             and controls[1]["error_type"] == "LeaseCancelledError", "separate native source-model integrity/cancellation controls differ")
        phases = [row["name"] for row in native["phases"]]
        need(phases == ["author_fresh_300_member_repository", "publish_previous_source", "explicit_previous_head_root_training",
            "publish_current_source", "build_default_source_delta", "explicit_current_head_child_training", "select_default_fresh_successor_root",
            "select_opt_out_successor_root", "infer_default_fresh_32_member_prefix", "infer_opt_out_fresh_32_member_prefix",
            "actual_current_source_delta_plan_preview", "refuse_old_model_for_new_head", "refuse_precancelled_selection_receiving",
            "cold_receive_successor_selection", "cold_receive_fresh_prefix_page", "cold_current_source_delta_plan_preview"]
             and all(row["status"] == "completed" for row in native["phases"]), "complete actual native phase population differs")
        need(native["source_delta_cid"] == d["artifact_cid"] and native["successor_selection_cid"] == selection["artifact_cid"]
             and native["root_cid"] == root["artifact_cid"] and native["prefix_page_cid"] == optimized_page["artifact_cid"]
             and native["planning_result_cid"] == planning["result_cid"], "native result closed artifact identities differ")
        report["native_artifact_cids"] = {"previous_manifest": previous["artifact_cid"], "current_manifest": current["artifact_cid"],
            "source_delta": d["artifact_cid"], "root_training_record": previous_record["artifact_cid"],
            "child_training_record": current_record["artifact_cid"], "default_selection": selection["artifact_cid"],
            "reference_selection": reference["artifact_cid"], "default_root": root["artifact_cid"],
            "reference_root": reference_root["artifact_cid"], "default_prefix_page": optimized_page["artifact_cid"],
            "reference_prefix_page": reference_page["artifact_cid"], "warm_planning_result": planning["result_cid"],
            "cold_planning_result": cold_planning["result_cid"]}
        report.update(qualified=True, selected_producer_files=len(producers), native_pid=native["pid"],
            scope=native["scope"], default_reference_prefix_equal=True)
    except Exception as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    finally:
        after = reader.whole_archive()
        report["preserved"] = same(before, after)
        if not report["preserved"]:
            report["qualified"] = False
            report["errors"].append({"type": "ArchiveChanged", "message": "primary archive bytes/membership/modes/mtimes changed"})
        report.update(archive=after, guarded_read_bytes=reader.total_read_bytes,
                      recorded_audit_seconds=time.monotonic() - started)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("namespace", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--seconds", type=float, default=120)
    args = parser.parse_args()
    namespace, destination = args.namespace.resolve(strict=True), args.report.absolute()
    need(destination.resolve(strict=False) == destination and namespace not in destination.parents,
         "new report must be outside primary archive")
    result = audit_closed_archive(namespace, seconds=args.seconds)
    reader_raw = Path(__file__).read_bytes()
    result["reader"] = {"bytes": len(reader_raw), "sha256": sha(reader_raw), "imports": "standard_library_only"}
    with destination.open("xb") as stream:
        stream.write(numerical_wire(result) + b"\n")
    destination.chmod(0o444)
    copy = Path(str(destination) + ".reader.py")
    with copy.open("xb") as stream:
        stream.write(reader_raw)
    copy.chmod(0o444)
    print(json.dumps({"qualified": result["qualified"], "preserved": result["preserved"],
                      "errors": result["errors"], "report": str(destination)}))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
