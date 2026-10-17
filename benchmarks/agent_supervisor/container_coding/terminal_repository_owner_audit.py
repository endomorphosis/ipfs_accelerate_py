"""Pure historical join of a closed owner session and retained operation bytes.

The caller must securely acquire both retained files and reject ambiguous
session-to-invocation matches across its cohort. This function does no I/O,
live descriptor validation, resource admission or provider invocation. Hashes
bind supplied artifacts; they do not independently authenticate their history.
Native operations describe prepared payloads. Transport records describe byte
reservations and completed server sends, never provider receipt or consumption.
Version 2 adds explicit operation references; a separate projection can join
preserved results to completed sends without changing consumption evidence.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp as mcp
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_binding as bindings
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol


SCHEMA = "terminal-repository-owner-session-observation@1"
OWNER_AUDIT_SCHEMAS = frozenset({"repository-source-owner-session@1", "repository-source-owner-session@2"})
OPERATION_CORRELATION_SCHEMA = "terminal-repository-owner-operation-correlation@1"
OPERATION_COUNTERS = ("total_operations", "completed_socket_send_operations",
    "completed_socket_send_tool_results", "completed_socket_send_successful_tool_results",
    "completed_socket_send_successful_reads", "completed_socket_send_failed_tool_results",
    "completed_socket_send_native_payload_bytes", "unconfirmed_operations")
MAX_AUDIT_BYTES = protocol.MAX_FRAME_BYTES
MAX_OPERATIONS_BYTES = mcp.MAX_AUDIT_BYTES
_AUDIT_FIELDS = frozenset({"schema", "session", "descriptor_sha256", "requests",
    "accepted_requests", "inflight_request", "response_bytes", "service_stopped",
    "cleanup_complete", "cleanup_pending", "accounting_complete", "native_child",
    "native_child_acquired", "native_child_acquisition_uncertain", "native_child_released",
    "service_error", "cleanup_error", "construction_error", "thread_launch_attempted",
    "join_error", "prepared_mcp_usage", "retained_mcp_audit", "canonical_source_read_only",
    "cross_uid_deployment_qualified", "automatic_terminal_wiring", "provider_calls",
    "token_savings_measured", "proof_authority", "execution_authority", "completion_authority"})
_RECORD_FIELDS = frozenset({"ordinal", "status", "request_sha256", "prepared_response_sha256",
    "prepared_response_bytes", "response_sha256", "delivered", "response_bytes", "elapsed_ns"})
_CAPTURE_FIELDS = frozenset({"status", "operations"})
_OPERATION_REFERENCE_FIELDS = frozenset({"sequence", "sha256", "bytes"})
_CHILD_FIELDS = frozenset({"lease_id", "lane", "cpu_slots", "memory_mb", "gpu_memory_mb",
    "unified_memory_mb", "child_process_slots", "requires_gpu", "parent_lease_id",
    "owner_pid", "acquired_at", "wait_seconds", "released", "cancelled"})
_CLEANUP_FIELDS = frozenset({"store_cleanup_complete", "artifacts_retained_after_invocation",
                            "store_cleanup_error"})
_FALSE_FIELDS = ("cross_uid_deployment_qualified", "automatic_terminal_wiring",
                 "token_savings_measured", "proof_authority", "execution_authority",
                 "completion_authority")


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def _text(value, limit=256):
    return (type(value) is str and 0 < len(value) <= limit
            and len(value.encode("utf-8")) <= limit
            and not any(ord(char) < 32 or ord(char) == 127 for char in value))


def _error(value):
    return value is None or type(value) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", value) is not None


def _child(value, descriptor):
    _need(type(value) is dict and set(value) == _CHILD_FIELDS,
          "closed native child record required")
    _need(_text(value["lease_id"]) and _text(value["parent_lease_id"])
          and value["lease_id"] != value["parent_lease_id"]
          and value["lane"] == "validation"
          and type(value["owner_pid"]) is int and value["owner_pid"] == descriptor["owner_pid"]
          and value["released"] is True and type(value["cancelled"]) is bool
          and value["requires_gpu"] is False,
          "native child identity or release differs")
    for name, expected in {"cpu_slots": 1, "memory_mb": 512, "child_process_slots": 1,
                           "gpu_memory_mb": 0, "unified_memory_mb": 0}.items():
        _need(type(value[name]) is int and value[name] == expected,
              "native child demand differs")
    for name in ("acquired_at", "wait_seconds"):
        _need(type(value[name]) in (int, float) and 0 <= value[name] <= 10**12
              and math.isfinite(value[name]), "bounded native child timing required")


def _transport(audit, descriptor):
    records = audit["requests"]
    maximum = descriptor["max_total_response_bytes"]
    _need(type(records) is list and len(records) <= descriptor["max_requests"]
          and _integer(audit["accepted_requests"], 0, descriptor["max_requests"])
          and audit["accepted_requests"] == len(records) and audit["inflight_request"] is None
          and _integer(audit["response_bytes"], 0, maximum),
          "complete bounded request accounting required")
    counts = {name: 0 for name in sorted(protocol.REASON_CODES | {"ok"})}
    result = dict(accepted_requests=len(records), status_counts=counts,
        prepared_responses=0, prepared_response_bytes=0, reserved_responses=0,
        reserved_response_bytes=0, completed_frame_sends=0, completed_frame_send_bytes=0,
        prepared_final_changes=0,
        scope="reported_canonical_envelope_reservations_and_completed_server_sends")
    validated_requests = 0
    record_fields = _RECORD_FIELDS | ({"operation_capture"}
        if audit["schema"] == "repository-source-owner-session@2" else set())
    for ordinal, record in enumerate(records, 1):
        _need(type(record) is dict and set(record) == record_fields
              and type(record["ordinal"]) is int and record["ordinal"] == ordinal
              and type(record["status"]) is str and record["status"] in counts
              and type(record["delivered"]) is bool
              and _integer(record["elapsed_ns"], 0, (1 << 63) - 1)
              and _integer(record["response_bytes"], 0, maximum),
              "closed bounded request record required")
        request, prepared, final = (record[name] for name in
            ("request_sha256", "prepared_response_sha256", "response_sha256"))
        _need(all(value is None or _digest(value) for value in (request, prepared, final)),
              "request or response digest malformed")
        _need((prepared is None and record["prepared_response_bytes"] is None)
              or prepared is not None and _integer(record["prepared_response_bytes"],
                  1, protocol.MAX_RESPONSE_FRAME_BYTES), "prepared response fields disagree")
        _need((final is None) == (record["response_bytes"] == 0)
              and (prepared is None or request is not None)
              and (final is None or prepared is not None)
              and (not record["delivered"] or final is not None)
              and (record["status"] != "ok" or final is not None)
              and (not record["delivered"] or record["status"] in {"ok", "budget_exhausted", "invalid_request"}),
              "request response null pattern differs")
        if prepared is not None and final is not None:
            _need(prepared != final or record["prepared_response_bytes"] == record["response_bytes"],
                  "identical response digest has different byte count")
            if prepared != final:
                # The sole source-side response replacement is the budget refusal.
                _need(record["prepared_response_bytes"] > maximum - result["reserved_response_bytes"]
                      and record["response_bytes"] < record["prepared_response_bytes"]
                      and record["status"] != "ok", "response replacement lacks budget boundary")
                result["prepared_final_changes"] += 1
        counts[record["status"]] += 1
        validated_requests += request is not None
        if prepared is not None:
            result["prepared_responses"] += 1
            result["prepared_response_bytes"] += record["prepared_response_bytes"]
        if final is not None:
            result["reserved_responses"] += 1
            result["reserved_response_bytes"] += record["response_bytes"]
        if record["delivered"]:
            result["completed_frame_sends"] += 1
            result["completed_frame_send_bytes"] += record["response_bytes"]
    _need(result["reserved_response_bytes"] == audit["response_bytes"] <= maximum,
          "aggregate reserved response bytes differ")
    return result, validated_requests


def operation_correlation_unavailable(reason="legacy_owner_audit"):
    """Separate null accounting; unknown references cannot become zero sends."""
    if type(reason) is not str or reason not in {"legacy_owner_audit", "invalid_operation_capture",
            "incomplete_operation_capture", "operation_reference_mismatch"}:
        reason = "invalid_operation_capture"
    return dict(schema=OPERATION_CORRELATION_SCHEMA, status="unavailable", reason=reason, counts=None,
        scope="owner_recorded_operation_results_in_completed_socket_sends_only",
        reference_sha256_scope="exact_native_audit_jsonl_record_including_lf",
        worker_receipt_verified=False, provider_ingestion_verified=False,
        response_payloads_independently_verified=False, token_savings_measured=False,
        proof_authority=False, execution_authority=False, completion_authority=False)


def _operation_correlation(audit, operations_raw, operation_rows):
    """Join explicit references after full session/operation validation.

    Exact persisted lines, including LF, are hashed; reserializing parsed rows
    would change the original insertion order. Individual operation rows are
    never filtered/reindexed through the native summarizer because their read
    and retrieval chains belong to the complete retained population.
    """
    if audit["schema"] == "repository-source-owner-session@1":
        return operation_correlation_unavailable()
    references = []
    incomplete = False
    try:
        for record in audit["requests"]:
            capture = record["operation_capture"]
            _need(type(capture) is dict and set(capture) == _CAPTURE_FIELDS
                  and type(capture["status"]) is str
                  and capture["status"] in {"not_entered", "complete", "incomplete"}
                  and type(capture["operations"]) is list and len(capture["operations"]) <= 1,
                  "invalid operation capture")
            if capture["status"] == "not_entered":
                _need(not capture["operations"], "unentered operation capture must be empty")
            else:
                _need(_digest(record["request_sha256"]), "entered capture requires validated request")
            incomplete |= capture["status"] == "incomplete"
            for reference in capture["operations"]:
                _need(type(reference) is dict and set(reference) == _OPERATION_REFERENCE_FIELDS
                      and _integer(reference["sequence"], 1, mcp.MAX_REQUESTS)
                      and _digest(reference["sha256"])
                      and _integer(reference["bytes"], 1, MAX_OPERATIONS_BYTES),
                      "invalid operation reference")
                references.append((reference, record))
    except (ValueError, KeyError, TypeError, UnicodeError, OverflowError, RecursionError):
        return operation_correlation_unavailable("invalid_operation_capture")
    if incomplete:
        return operation_correlation_unavailable("incomplete_operation_capture")
    lines = operations_raw.splitlines(keepends=True)
    if len(references) != len(operation_rows) or len(lines) != len(operation_rows):
        return operation_correlation_unavailable("operation_reference_mismatch")
    counts = dict.fromkeys(OPERATION_COUNTERS, 0)
    for index, ((reference, record), line, operation) in enumerate(zip(references, lines, operation_rows), 1):
        if (not line.endswith(b"\n") or reference["sequence"] != operation["sequence"]
                or reference["sequence"] != index or reference["sha256"] != _sha(line)
                or reference["bytes"] != len(line)):
            return operation_correlation_unavailable("operation_reference_mismatch")
        counts["total_operations"] += 1
        confirmed = (record["delivered"] is True and record["status"] == "ok"
            and record["prepared_response_sha256"] is not None
            and record["prepared_response_sha256"] == record["response_sha256"]
            and record["prepared_response_bytes"] == record["response_bytes"])
        if not confirmed:
            counts["unconfirmed_operations"] += 1
            continue
        counts["completed_socket_send_operations"] += 1
        if operation["event"] == "tool":
            counts["completed_socket_send_tool_results"] += 1
            counts["completed_socket_send_native_payload_bytes"] += operation["returned_bytes"]
            if operation["success"]:
                counts["completed_socket_send_successful_tool_results"] += 1
                counts["completed_socket_send_successful_reads"] += operation["tool"] == "repository_read"
            else:
                counts["completed_socket_send_failed_tool_results"] += 1
    result = operation_correlation_unavailable()
    result.update(status="observed", reason=None, counts=counts)
    return result


def _summarize(*, audit_raw, operations_raw, row):
    _need(type(audit_raw) is bytes and 0 < len(audit_raw) <= MAX_AUDIT_BYTES
          and type(operations_raw) is bytes and len(operations_raw) <= MAX_OPERATIONS_BYTES,
          "bounded retained audit bytes required")
    _need(type(row) is dict and row.get("schema") == "router-implementation-invocation@1"
          and row.get("purpose") == row.get("phase") == "coding"
          and row.get("provider") == "codex_cli"
          and type(row.get("invocation_id")) is str
          and re.fullmatch(r"[0-9a-f]{32}", row["invocation_id"]) is not None
          and row.get("status") in ("provider_returned", "failed")
          and row.get("completion_authority") is False
          and type(row.get("workspace")) is str,
          "bounded coding invocation receipt required")
    binding = row.get("repository_source_policy")
    _need(type(binding) is dict and len(protocol.canonical_json(binding)) <= MAX_AUDIT_BYTES,
          "bounded owner binding required")
    bindings.validate_binding(binding, workspace=Path(row["workspace"]), materialized=False)
    audit = protocol.strict_json_loads(audit_raw, max_bytes=MAX_AUDIT_BYTES)
    _need(type(audit) is dict and set(audit) == _AUDIT_FIELDS
          and audit["schema"] in OWNER_AUDIT_SCHEMAS,
          "closed owner session audit required")
    descriptor = protocol.validate_descriptor(audit["session"])
    descriptor_raw = protocol.canonical_json(descriptor)
    _need(audit["descriptor_sha256"] == binding["descriptor_sha256"] == _sha(descriptor_raw)
          and descriptor_raw == protocol.canonical_json(binding["descriptor"]),
          "owner session descriptor differs from invocation")
    boundary = row.get("external_container_boundary")
    if boundary is not None:
        _need(type(boundary) is dict
              and set(boundary) == {"schema", "container_id", "image_id", "namespaces", "owner_uid",
                                   "worker_uid", "purpose", "manifest_sha256", "completion_authority"}
              and boundary["schema"] == "supervisor-container-worker-boundary@1"
              and boundary["purpose"] == "coding" and boundary["completion_authority"] is False
              and boundary["manifest_sha256"] == descriptor["boundary_manifest_sha256"]
              and all(type(boundary[key]) is int and boundary[key] == descriptor[key]
                      for key in ("owner_uid", "worker_uid"))
              and _text(boundary["container_id"]) and _text(boundary["image_id"])
              and type(boundary["namespaces"]) is dict and set(boundary["namespaces"]) == {"pid", "mnt", "net"}
              and all(_text(value) for value in boundary["namespaces"].values()),
              "retained container boundary differs")
    usage = row.get("usage")
    _need(usage is None or type(usage) is dict, "router usage malformed")
    if type(usage) is dict:
        if "codex_repository_source_configured" in usage:
            _need(type(usage["codex_repository_source_configured"]) is bool,
                  "router configuration observation malformed")
        if usage.get("codex_repository_source_config_sha256") is not None:
            _need(usage["codex_repository_source_config_sha256"] == binding["config_sha256"],
                  "worker launch configuration digest differs")
        if usage.get("codex_repository_source_configured") is True:
            _need(usage.get("codex_repository_source_config_sha256") == binding["config_sha256"],
                  "configured worker launch lacks matching digest")
    _need(all(audit[key] is True for key in ("service_stopped", "cleanup_complete", "accounting_complete",
              "native_child_acquired", "native_child_released", "thread_launch_attempted", "canonical_source_read_only"))
          and audit["cleanup_pending"] is False and audit["native_child_acquisition_uncertain"] is False
          and audit["cleanup_error"] is None and audit["join_error"] is None
          and all(audit[key] is False for key in _FALSE_FIELDS)
          and type(audit["provider_calls"]) is int and audit["provider_calls"] == 0
          and _error(audit["service_error"]) and _error(audit["construction_error"]),
          "only fully stopped and cleaned owner sessions are supported")
    _child(audit["native_child"], descriptor)
    transport, validated_requests = _transport(audit, descriptor)
    retained = audit["retained_mcp_audit"]
    _need(type(retained) is dict and set(retained) == {"name", "sha256", "bytes", "content_free"}
          and retained["name"] == "repository-mcp-" + descriptor["session_id"] + "-audit-operations.jsonl"
          and retained["sha256"] == _sha(operations_raw)
          and type(retained["bytes"]) is int and retained["bytes"] == len(operations_raw)
          and retained["content_free"] is True, "retained operation binding differs")
    prepared = mcp.summarize_repository_audit_bytes(operations_raw)
    _need(prepared.get("status") in {"observed", "not_observed"},
          "retained operations are malformed")
    operation_rows = [protocol.strict_json_loads(line, max_bytes=MAX_OPERATIONS_BYTES)
                      for line in operations_raw.splitlines()]
    _need(len(operation_rows) <= validated_requests
          and all(record["tool"] is None or record["tool"] in protocol.READ_ONLY_TOOL_NAMES
                  for record in operation_rows)
          and prepared["canonical_edits"] == prepared["uncertain_edits"] == 0,
          "operations exceed read-only validated request population")
    reported = audit["prepared_mcp_usage"]
    _need(type(reported) is dict and reported.get("store_cleanup_complete") is True
          and reported.get("artifacts_retained_after_invocation") is False
          and "store_cleanup_error" not in reported
          and protocol.canonical_json({key: value for key, value in reported.items()
                                       if key not in _CLEANUP_FIELDS}) == protocol.canonical_json(prepared),
          "prepared operation recount differs from owner report")
    operation_correlation = _operation_correlation(audit, operations_raw, operation_rows)
    return dict(schema=SCHEMA, status="joined_complete", audit_sha256=_sha(audit_raw),
        audit_bytes=len(audit_raw), operations_sha256=_sha(operations_raw), operations_bytes=len(operations_raw),
        descriptor_sha256=audit["descriptor_sha256"], invocation_sha256=_sha(row["invocation_id"].encode()),
        owner_config_sha256=binding["owner_config_sha256"], worker_config_sha256=binding["config_sha256"],
        boundary_sha256=descriptor["boundary_manifest_sha256"],
        container_boundary_identity_joined=boundary is not None,
        worker_configuration_observed=(usage or {}).get("codex_repository_source_configured"),
        transport=transport, prepared_mcp_usage=prepared,
        operation_correlation=operation_correlation,
        cleanup=dict(scope="reported_session_child_and_private_store_only", service_stopped=True,
            accounting_complete=True, cleanup_complete=True, native_child_acquired=True,
            native_child_released=True, store_cleanup_complete=True,
            artifacts_retained_after_invocation=False, service_error_reported=audit["service_error"] is not None,
            construction_error_reported=audit["construction_error"] is not None,
            native_parent_released=None, native_parent_release_reason="no_retained_lifecycle_receipt"),
        consumption=dict(delivered_tool_calls=None, delivered_reads=None,
            delivery_reason=("missing_worker_receipt_evidence" if operation_correlation["status"] == "observed"
                else "missing_per_operation_correlation"), provider_ingestion_verified=None,
            provider_consumption_reason="no_provider_consumption_evidence", provider_tokens=None,
            token_savings=None, token_reason="not_measured"),
        request_response_payloads_recomputed=False, session_uniqueness_verified=False,
        live_authority_verified=False, proof_authority=False, execution_authority=False,
        completion_authority=False)


def summarize_owner_session(*, audit_raw: bytes, operations_raw: bytes, row: dict) -> dict:
    """Join supplied evidence or raise ValueError, without reading live state.

    Only the completed, acquired-and-released child branch is admitted. Missing
    operations must never be replaced with ``b''``: empty bytes establish zero
    prepared operations only after the caller has actually retained that file.
    The input row is a historical coding receipt, not a synthetic invocation.
    """
    try:
        return _summarize(audit_raw=audit_raw, operations_raw=operations_raw, row=row)
    except (KeyError, TypeError, UnicodeError, OverflowError, RecursionError) as error:
        raise ValueError("invalid owner session evidence") from error
