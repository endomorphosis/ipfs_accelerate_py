"""Opt-in native admission of owner-signed, independently checked metadata packets.

The intent owner remains the authority. Contracts live in its existing evidence
nodes; observed checks live in validation_runs/results. This module creates no
registry, does not execute a command, and grants no model or legal authority.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from itertools import chain
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from .control_plane_contracts import content_identity

POLICY_SCHEMA = "ipfs_accelerate_py/agent-supervisor/execution-packet-policy@1"
CONTRACT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/execution-packet-contract@1"
VALIDATION_SCHEMA = "ipfs_accelerate_py/agent-supervisor/execution-packet-observed-validation@1"
EVIDENCE_KIND = "execution_packet_contract"
PURPOSES = frozenset({"control_metadata_analysis", "metadata_validation"})
CLAIM_STATUSES = frozenset({"proposed", "admitted", "pending", "ready", "todo", "queued", "retrying"})
DISPATCH_STATUSES = CLAIM_STATUSES | frozenset({"claimed", "in_progress", "running"})
COMPLETED = frozenset({"completed", "complete", "done", "skipped"})
MAX_FILES = 128
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES = 64 * 1024 * 1024
PROFILE_FIELDS = frozenset({"profile_dir", "lifecycle_dir", "repository_cid", "profile_id", "identity_did", "content_id"})
PACKET_FIELDS = frozenset({"purpose", "cwd", "argv", "validation_argv", "source_descriptors", "inputs", "outputs", "code_allowlist", "test_allowlist", "exclusive_keys", "budgets", "retry_key", "invalidation_key"})
BUDGET_FIELDS = frozenset({"cpu_threads", "memory_mib", "storage_bytes", "max_children", "max_tokens", "deadline_seconds", "max_attempts"})


class ExecutionPacketAdmissionError(ValueError):
    """An opted-in task has no current authoritative packet acceptance."""


def _need(value, code):
    if not value:
        raise ExecutionPacketAdmissionError(code)


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _decode(value):
    def pairs(items):
        result = {}
        for k, v in items:
            _need(k not in result, "packet_duplicate_json_key")
            result[k] = v
        return result
    _need(isinstance(value, str) and len(value.encode()) <= MAX_JSON_BYTES, "packet_json_bound")
    return json.loads(value, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ExecutionPacketAdmissionError("packet_nonfinite_json")))


def _values(row, names):
    return [row[name] for name in names] if isinstance(row, Mapping) else list(row)


def execution_packet_required(body):
    if not isinstance(body, Mapping):
        return False
    required = False
    for value in (body, body.get("authored_task", {})):
        if isinstance(value, Mapping) and "execution_packet_required" in value:
            flag = value["execution_packet_required"]
            _need(type(flag) is bool, "packet_requirement_malformed")
            required = required or flag
    return required


def execution_packet_required_metadata(metadata):
    """Read the existing PortalTask string grammar without accepting truthy text."""
    required = False
    if isinstance(metadata, Mapping):
        for key in ("execution packet required", "execution_packet_required"):
            if key in metadata:
                value = metadata[key]
                if type(value) is str:
                    _need(value.strip().lower() in {"true", "false"}, "packet_metadata_requirement_malformed")
                    required = required or value.strip().lower() == "true"
                else:
                    _need(type(value) is bool, "packet_metadata_requirement_malformed")
                    required = required or value
    return required


def execution_packet_required_bundle(bundle):
    """Inspect only declared bundle/task flags and their legacy metadata."""
    _need(isinstance(bundle, Mapping), "packet_bundle_shape")
    tasks = bundle.get("tasks", [])
    _need(isinstance(tasks, (list, tuple)), "packet_bundle_task_shape")
    required = False
    for item in chain((bundle,), tasks):
        _need(isinstance(item, Mapping), "packet_bundle_task_shape")
        required = execution_packet_required(item) or required
        required = execution_packet_required(item.get("body", {})) or required
        required = execution_packet_required_metadata(item.get("metadata", {})) or required
    return required


def preserve_execution_packet_requirement(previous, body):
    if execution_packet_required(previous):
        _need(execution_packet_required(body), "packet_requirement_cannot_be_removed")
        for old, new in ((previous, body), (previous.get("authored_task", {}), body.get("authored_task", {}))):
            if isinstance(old, Mapping) and old.get("execution_packet_required") is True:
                _need(isinstance(new, Mapping) and new.get("execution_packet_required") is True,
                      "packet_requirement_cannot_be_removed")


def task_record_on(connection, task_cid):
    names = ("task_cid", "task_alias", "goal_cid", "plan_cid", "objective_id", "ordinal", "status", "revision", "priority", "identity_json", "body_json")
    rows = connection.execute("SELECT " + ", ".join(names) + " FROM tasks WHERE task_cid = ?", [task_cid]).fetchall()
    _need(len(rows) == 1, "packet_task_missing_or_ambiguous")
    record = dict(zip(names, _values(rows[0], names)))
    record["identity"] = _decode(record.pop("identity_json"))
    record["body"] = _decode(record.pop("body_json"))
    record["dependencies"] = [str(_values(r, ("dependency_task_cid",))[0]) for r in connection.execute(
        "SELECT dependency_task_cid FROM task_dependencies WHERE task_cid=? ORDER BY dependency_task_cid", [task_cid]).fetchall()]
    for key, table, columns, json_columns in (
        ("outputs", "task_outputs", ("ordinal", "path", "effect_json"), {"effect_json": "effect"}),
        ("acceptance", "task_acceptance", ("ordinal", "criterion", "evidence_policy_json"), {"evidence_policy_json": "evidence_policy"}),
        ("validations", "task_validations", ("ordinal", "argv_json", "policy_json"), {"argv_json": "argv", "policy_json": "policy"}),
    ):
        items = []
        for row in connection.execute("SELECT " + ", ".join(columns) + " FROM " + table + " WHERE task_cid=? ORDER BY ordinal", [task_cid]).fetchall():
            item = dict(zip(columns, _values(row, columns)))
            for old, new in json_columns.items():
                item[new] = _decode(item.pop(old))
            items.append(item)
        record[key] = items
    return record


def authority_spec_cid(record):
    from .intent_repository import task_authority_spec_cid
    return task_authority_spec_cid(record)


def latest_spec_event_on(connection, task_cid):
    rows = connection.execute(
        "SELECT global_sequence, event_type, body_json FROM domain_events WHERE task_cid=? "
        "AND event_type IN ('intent.task_upserted','intent.task_dependencies_set') ORDER BY global_sequence DESC LIMIT 1025", [task_cid]).fetchall()
    for row in rows:
        sequence, kind, raw = _values(row, ("global_sequence", "event_type", "body_json"))
        event = _decode(raw)
        body = event.get("body", {})
        same_spec = (body.get("execution_packet_previous_spec_cid") == body.get("execution_packet_spec_cid")
                     and bool(body.get("execution_packet_spec_cid")))
        # A dependency-only event cannot change the plan; upserts can. Older
        # upserts without explicit plan lineage are conservatively invalidating.
        same_plan = ("execution_packet_previous_plan_cid" in body
                     and type(body.get("execution_packet_plan_cid")) is str
                     and body["execution_packet_previous_plan_cid"] == body["execution_packet_plan_cid"])
        if same_spec and (kind == "intent.task_dependencies_set" or same_plan):
            continue
        return int(sequence)
    _need(len(rows) < 1025, "packet_spec_event_population_bound")
    return 0


def dependencies_on(connection, record):
    result = []
    _need(len(record["dependencies"]) <= MAX_FILES, "packet_dependency_population_bound")
    for cid in record["dependencies"]:
        task = task_record_on(connection, cid)
        receipts = connection.execute(
            "SELECT receipt_cid, body_json FROM completion_receipts WHERE task_cid=? ORDER BY completed_at DESC, receipt_cid", [cid]).fetchall()
        receipt = None
        for row in receipts:
            rid, raw = _values(row, ("receipt_cid", "body_json"))
            if _decode(raw).get("revision") == task["revision"]:
                _need(receipt is None, "packet_dependency_completion_ambiguous")
                receipt = str(rid)
        result.append({"task_cid": cid, "revision": task["revision"], "status": task["status"],
                       "authority_spec_cid": authority_spec_cid(task), "completion_receipt_cid": receipt})
    return result


def _text(value):
    return type(value) is str and 0 < len(value) <= 4096 and not any(ord(c) < 32 for c in value)


def _argv(value):
    return type(value) is list and 0 < len(value) <= 128 and all(_text(v) for v in value)


def _file_current(descriptor, cwd):
    _need(type(descriptor) is dict and set(descriptor) == {"path", "bytes", "sha256"}, "packet_descriptor_shape")
    name, size, wanted = (descriptor[k] for k in ("path", "bytes", "sha256"))
    _need(_text(name) and Path(name).is_absolute() and type(size) is int and 0 <= size <= MAX_FILE_BYTES
          and type(wanted) is str and len(wanted) == 64 and all(c in "0123456789abcdef" for c in wanted), "packet_descriptor_bound")
    path = Path(name)
    _need(str(path) == os.path.abspath(name), "packet_input_not_normalized")
    _need(path == cwd or cwd in path.parents, "packet_input_outside_workspace")
    _need(not any(p.is_symlink() for p in (path, *path.parents)), "packet_input_symlink")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size == size, "packet_input_not_exact_regular")
        sha = hashlib.sha256()
        count = 0
        while True:
            block = os.read(fd, 65536)
            if not block:
                break
            count += len(block)
            _need(count <= MAX_FILE_BYTES, "packet_input_grew")
            sha.update(block)
        after = os.fstat(fd)
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_nlink, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        _need(count == size and sha.hexdigest() == wanted and stamp(before) == stamp(after) == stamp(path.lstat()), "packet_input_changed")
    finally:
        os.close(fd)


def _profile(reference, cwd):
    from ..control.profile_authority import load_local_profile
    _need(type(reference) is dict and set(reference) == PROFILE_FIELDS, "packet_profile_reference_shape")
    _need(all(_text(v) for v in reference.values()), "packet_profile_reference_bound")
    for key in ("profile_dir", "lifecycle_dir"):
        path = Path(reference[key])
        _need(path.is_absolute() and path != cwd and cwd not in path.parents, "packet_owner_keys_inside_workspace")
    profile = load_local_profile(repository_cid=reference["repository_cid"], profile_dir=Path(reference["profile_dir"]), lifecycle_dir=Path(reference["lifecycle_dir"]), mirror=False)
    _need(profile.profile_id == reference["profile_id"] and profile.identity_did == reference["identity_did"] and profile.content_id == reference["content_id"], "packet_profile_changed")
    _need(profile.allows("read") and profile.allows("test"), "packet_profile_lacks_metadata_effects")
    # Reuse only the existing repository observation, never benchmark admission.
    from ..runtime.local_planning_admission import _repository
    _root, head, repository_cid = _repository(cwd)
    _need(profile.repository_cid == repository_cid and profile.baseline_commit == head, "packet_profile_repository_changed")
    return profile


def _policy(record, cwd):
    policy = record["body"].get("execution_packet_policy")
    _need(type(policy) is dict and set(policy) == {"schema", "purpose", "owner", "verifier", "max_age_seconds"}
          and policy["schema"] == POLICY_SCHEMA and policy["purpose"] in PURPOSES, "packet_purpose_authority_not_selected")
    _need(type(policy["max_age_seconds"]) is int and 0 < policy["max_age_seconds"] <= 86400, "packet_acceptance_age_bound")
    owner, verifier = (_profile(policy[k], cwd) for k in ("owner", "verifier"))
    _need(owner.identity_did != verifier.identity_did and owner.repository_cid == verifier.repository_cid, "packet_independent_profile_required")
    return policy, owner, verifier


def _packet_current(packet, record):
    _need(type(packet) is dict and set(packet) == PACKET_FIELDS and packet["purpose"] in PURPOSES, "packet_contract_shape_or_purpose")
    cwd = Path(packet["cwd"]) if _text(packet.get("cwd")) else Path(".")
    _need(cwd.is_absolute() and str(cwd) == os.path.abspath(cwd) and cwd.is_dir() and not any(p.is_symlink() for p in (cwd, *cwd.parents)), "packet_workspace_not_exact")
    _need(_argv(packet["argv"]) and _argv(packet["validation_argv"]), "packet_exact_commands_required")
    _need(type(packet["source_descriptors"]) is list and 0 < len(packet["source_descriptors"]) <= MAX_FILES, "packet_source_population_bound")
    paths = []
    total_bytes = 0
    for descriptor in packet["source_descriptors"]:
        _need(type(descriptor) is dict and type(descriptor.get("bytes")) is int, "packet_descriptor_shape")
        total_bytes += descriptor["bytes"]
        _need(0 <= total_bytes <= MAX_TOTAL_INPUT_BYTES, "packet_total_input_bound")
        _file_current(descriptor, cwd)
        paths.append(descriptor["path"])
    _need(len(paths) == len(set(paths)), "packet_duplicate_source")
    _need(any(p in packet["argv"] for p in paths) and any(p in packet["validation_argv"] for p in paths), "packet_commands_not_source_bound")
    inputs = packet["inputs"]
    _need(type(inputs) is list and 0 < len(inputs) <= MAX_FILES, "packet_input_population_bound")
    for item in inputs:
        _need(type(item) is dict and set(item) == {"role", "use", "descriptor"} and item["role"] in {"source_metadata", "prior_evidence"} and item["use"] in {"verification", "provenance"}, "packet_metadata_input_role")
        descriptor = item["descriptor"]
        _need(type(descriptor) is dict and type(descriptor.get("bytes")) is int, "packet_descriptor_shape")
        total_bytes += descriptor["bytes"]
        _need(0 <= total_bytes <= MAX_TOTAL_INPUT_BYTES, "packet_total_input_bound")
        _file_current(descriptor, cwd)
    _need(packet["outputs"] == [r["path"] for r in record["outputs"]] and bool(packet["outputs"]), "packet_native_output_scope_changed")
    _need(packet["exclusive_keys"] == record["body"].get("exclusive_keys") and type(packet["exclusive_keys"]) is list and bool(packet["exclusive_keys"]) and all(_text(k) for k in packet["exclusive_keys"]), "packet_exclusive_scope_changed")
    _need(packet["code_allowlist"] == record["body"].get("code_edit_allowlist", []) == [], "packet_metadata_code_edits_forbidden")
    _need(type(packet["test_allowlist"]) is list and len(packet["test_allowlist"]) <= MAX_FILES and all(_text(p) and p in paths for p in packet["test_allowlist"]), "packet_test_scope_required")
    limits = packet["budgets"]
    _need(type(limits) is dict and set(limits) == BUDGET_FIELDS and type(limits["max_tokens"]) is int and limits["max_tokens"] == 0
          and all(type(v) is int and 0 < v <= 2**63 - 1 for k, v in limits.items() if k != "max_tokens"), "packet_finite_zero_model_budget_required")
    _need(_text(packet["retry_key"]) and _text(packet["invalidation_key"]), "packet_retry_invalidation_required")
    policy, owner, verifier = _policy(record, cwd)
    _need(packet["purpose"] == policy["purpose"], "packet_purpose_policy_mismatch")
    return policy, owner, verifier


def contract_payload_on(connection, task_cid, packet):
    """Build unsigned review material; only native stored signed evidence can admit."""
    record = task_record_on(connection, task_cid)
    _need(execution_packet_required(record["body"]), "packet_task_not_opted_in")
    _packet_current(packet, record)
    return {"schema": CONTRACT_SCHEMA, "task_cid": record["task_cid"], "plan_cid": record["plan_cid"],
            "reviewed_revision": record["revision"], "authority_spec_cid": authority_spec_cid(record),
            "spec_event_cursor": latest_spec_event_on(connection, task_cid),
            "dependencies": dependencies_on(connection, record), "packet": _plain(packet)}


def observed_validation_payload(contract, *, outcome="passed"):
    """Unsigned observed-result material, requiring separate reviewer signature/native recording."""
    payload = contract["payload"]
    return {"schema": VALIDATION_SCHEMA, "task_cid": payload["task_cid"], "contract_digest": content_identity(contract),
            "authority_spec_cid": payload["authority_spec_cid"], "spec_event_cursor": payload["spec_event_cursor"],
            "plan_cid": payload["plan_cid"], "dependencies": payload["dependencies"],
            "packet_cid": content_identity(payload["packet"]), "argv": payload["packet"]["validation_argv"], "outcome": outcome}


def _verify_envelope(envelope, profile):
    from ..control.profile_authority import verify_did_key_signature
    _need(type(envelope) is dict and set(envelope) == {"payload", "binding"}, "packet_signed_envelope_required")
    binding = envelope["binding"]
    _need(type(binding) is dict and set(binding) == {"identity", "profile_id", "signature"}
          and binding["identity"] == profile.identity_did and binding["profile_id"] == profile.profile_id, "packet_wrong_signing_owner")
    verify_did_key_signature(identity_did=profile.identity_did, payload=envelope["payload"], signature=binding["signature"], mirror=False)
    return envelope["payload"]


def _require_execution_packet_on(connection, task_cid, *, phase="dispatch", record=None):
    """Check native evidence on the caller's transaction before any dispatch effect."""
    task = task_record_on(connection, task_cid) if record is None else record
    if not execution_packet_required(task["body"]):
        return None
    if phase == "claim":
        _need(task["status"] in CLAIM_STATUSES, "packet_canonical_task_not_ready")
    rows = connection.execute("SELECT evidence_id, parent_evidence_id, digest, body_json FROM evidence_nodes WHERE task_cid=? AND evidence_kind=? ORDER BY created_at, evidence_id LIMIT 65", [task_cid, EVIDENCE_KIND]).fetchall()
    _need(0 < len(rows) <= 64, "packet_current_native_contract_required")
    nodes = {}
    parents = set()
    parent_by_id = {}
    for row in rows:
        eid, parent, digest, raw = _values(row, ("evidence_id", "parent_evidence_id", "digest", "body_json"))
        _need(str(eid) not in nodes, "packet_duplicate_evidence")
        nodes[str(eid)] = (str(digest), _decode(raw))
        parent_by_id[str(eid)] = str(parent) if parent else None
        if parent:
            parents.add(str(parent))
    heads = set(nodes) - parents
    _need(len(heads) == 1 and parents <= set(nodes), "packet_evidence_head_ambiguous")
    head = next(iter(heads))
    seen = set()
    cursor = head
    while cursor is not None:
        _need(cursor not in seen, "packet_evidence_cycle")
        seen.add(cursor)
        cursor = parent_by_id[cursor]
    _need(seen == set(nodes), "packet_evidence_disconnected_or_branch")
    digest, body = nodes[head]
    _need(type(body) is dict and set(body) == {"execution_packet_contract"}, "packet_native_contract_body")
    envelope = body["execution_packet_contract"]
    _need(type(envelope) is dict, "packet_signed_envelope_required")
    _need(content_identity(envelope) == digest, "packet_native_contract_digest_mismatch")
    payload = envelope.get("payload", {})
    expected_fields = {"schema", "task_cid", "plan_cid", "reviewed_revision", "authority_spec_cid", "spec_event_cursor", "dependencies", "packet"}
    _need(type(payload) is dict and set(payload) == expected_fields and payload["schema"] == CONTRACT_SCHEMA, "packet_contract_payload_shape")
    policy, owner, verifier = _packet_current(payload["packet"], task)
    _verify_envelope(envelope, owner)
    _need(payload["task_cid"] == task_cid and payload["plan_cid"] == task["plan_cid"]
          and type(payload["reviewed_revision"]) is int and 0 < payload["reviewed_revision"] <= task["revision"]
          and type(payload["spec_event_cursor"]) is int and payload["spec_event_cursor"] >= 0
          and payload["authority_spec_cid"] == authority_spec_cid(task)
          and payload["spec_event_cursor"] == latest_spec_event_on(connection, task_cid), "packet_native_spec_changed")
    dependencies = dependencies_on(connection, task)
    _need(payload["dependencies"] == dependencies, "packet_dependency_changed")
    _need(all(d["status"] in COMPLETED and d["completion_receipt_cid"] for d in dependencies), "packet_dependency_not_accepted")
    found = False
    expected = observed_validation_payload(envelope)
    validations = connection.execute(
        "SELECT r.outcome, r.evidence_digest, r.body_json, v.status, v.command_digest, v.finished_at "
        "FROM validation_results r JOIN validation_runs v ON v.run_id=r.run_id "
        "WHERE r.task_cid=? ORDER BY v.finished_at DESC, r.result_id DESC LIMIT 64", [task_cid]).fetchall()
    for row in validations:
        outcome, result_digest, raw, run_status, command_digest, finished = _values(row, ("outcome", "evidence_digest", "body_json", "status", "command_digest", "finished_at"))
        result_body = _decode(raw)
        _need(type(result_body) is dict, "packet_native_validation_body")
        result = result_body.get("execution_packet_validation")
        if not isinstance(result, dict) or not isinstance(result.get("payload"), dict) or result["payload"].get("contract_digest") != digest:
            continue
        _need(outcome == run_status == "passed" and result_digest == content_identity(result), "packet_native_validation_failed_or_digest_mismatch")
        observed = _verify_envelope(result, verifier)
        _need(observed == expected and command_digest == content_identity({"argv": payload["packet"]["validation_argv"]}), "packet_native_validation_scope_mismatch")
        timestamp = datetime.fromisoformat(str(finished).replace("Z", "+00:00")).timestamp()
        age = time.time() - timestamp
        _need(-60 <= age <= policy["max_age_seconds"], "packet_native_validation_expired")
        found = True
        break
    _need(found, "packet_independent_native_validation_required")
    if phase == "provider":
        raise ExecutionPacketAdmissionError("packet_metadata_has_no_provider_authority")
    return {"contract_digest": digest, "authority_spec_cid": payload["authority_spec_cid"], "purpose": payload["packet"]["purpose"]}


def require_execution_packet_on(connection, task_cid, *, phase="dispatch", record=None):
    try:
        return _require_execution_packet_on(connection, task_cid, phase=phase, record=record)
    except ExecutionPacketAdmissionError:
        raise
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise ExecutionPacketAdmissionError("packet_acceptance_invalid:" + str(exc)) from exc


class NativeExecutionPacketAuthority:
    """A live existing intent owner, never a caller-supplied admission boolean."""
    def __init__(self, intent):
        from .intent_repository import IntentRepository
        _need(type(intent) is IntentRepository, "packet_original_intent_owner_required")
        self._intent = intent

    def require(self, task_cid, *, phase="dispatch"):
        with self._intent._connection(write=False) as connection:
            return require_execution_packet_on(connection, task_cid, phase=phase)


def require_coordination_packet(connection, task_cid, authority=None):
    """A serialized coordination projection cannot invent native acceptance."""
    row = connection.execute("SELECT body_json FROM coordination_tasks WHERE task_cid=?", [task_cid]).fetchone()
    if row is None:
        return
    body = _decode(_values(row, ("body_json",))[0])
    if not execution_packet_required(body):
        return
    _need(type(authority) is NativeExecutionPacketAuthority, "packet_live_native_authority_required")
    _need(authority.require(task_cid, phase="claim") is not None, "packet_canonical_requirement_missing")
