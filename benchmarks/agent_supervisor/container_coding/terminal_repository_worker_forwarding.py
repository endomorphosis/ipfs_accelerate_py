"""Join optional worker-reported forwarding to retained owner evidence.

These copied receipts report validated owner responses and completed writes
into a stdio pipe. Hash matches do not authenticate worker-generated receipts,
prove the pipe reader consumed bytes, or establish provider/model ingestion.
Optional invocation-row captures join the existing bounded worker outbox
handoff. A complete scan covers retained slots, not every worker invocation:
an unpublished or zero-request bridge cannot be identified from these records.
"""
from __future__ import annotations

import hashlib
import re

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_protocol as protocol
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_worker_receipts as capture_api
from . import terminal_repository_owner_audit as owner_audit


SCHEMA = "terminal-repository-worker-forwarding-observation@1"
WORKER_SCHEMA = "repository-source-worker-forwarding@1"
MAX_RECEIPTS = 16
MAX_RECEIPT_BYTES = 65536
MAX_EVENTS = 64
COUNTERS = ("reported_received_operations", "reported_received_tool_results",
    "reported_received_successful_reads", "reported_completed_stdout_operations",
    "reported_completed_stdout_tool_results", "reported_completed_stdout_successful_reads",
    "unconfirmed_stdout_operations")
_FIELDS = frozenset({"schema", "bridge_id", "status", "complete", "reason", "session_id",
    "descriptor_sha256", "returncode", "owner_exchange_attempts", "owner_exchange_unconfirmed",
    "observed_responses", "events", "worker_reported", "independently_verified",
    "provider_ingestion_verified", "token_savings_measured", "proof_authority", "execution_authority",
    "completion_authority", "raw_content_exported", "observer_provider_calls"})
_EVENT_FIELDS = frozenset({"exchange_ordinal", "request_sha256", "owner_response_sha256",
    "owner_response_bytes", "owner_ok", "owner_reason_code", "output_kind", "stdout_sha256",
    "stdout_bytes", "forwarding_status"})
_REASONS = frozenset({"owner_evidence_unavailable", "owner_correlation_unavailable",
    "worker_evidence_unavailable", "invalid_worker_receipt", "incomplete_worker_receipt",
    "worker_identity_mismatch", "ambiguous_worker_population", "owner_response_mismatch",
    "incomplete_owner_population", "worker_capture_unavailable", "invalid_worker_capture",
    "worker_capture_identity_mismatch", "incomplete_worker_forwarding_population"})


def unavailable(reason="worker_evidence_unavailable"):
    if type(reason) is not str or reason not in _REASONS:
        reason = "invalid_worker_receipt"
    return dict(schema=SCHEMA, status="unavailable", reason=reason, counts=None,
        audit_sha256=None, operations_sha256=None, descriptor_sha256=None, invocation_sha256=None,
        forwarding_receipt_sha256s=[], forwarding_receipt_count=None,
        scope="worker_reported_validated_owner_response_events_and_pipe_write_completion_only",
        worker_reported_only=True, independent_worker_receipt_verified=False,
        provider_ingestion_verified=False, output_payloads_independently_verified=False,
        token_savings_measured=False, proof_authority=False, execution_authority=False,
        completion_authority=False, production_route_qualified=False)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _need(condition, reason="invalid_worker_receipt"):
    if not condition:
        raise _Invalid(reason)


class _Invalid(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def _integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def _event_key(event):
    return (event["request_sha256"], event["owner_response_sha256"], event["owner_response_bytes"])


def _owner_key(record):
    return record["request_sha256"], record["response_sha256"], record["response_bytes"]


def _receipt(raw, descriptor_sha256, session_id):
    _need(type(raw) is bytes and 0 < len(raw) <= MAX_RECEIPT_BYTES)
    value = protocol.strict_json_loads(raw, max_bytes=MAX_RECEIPT_BYTES)
    _need(type(value) is dict and set(value) == _FIELDS and value["schema"] == WORKER_SCHEMA)
    _need(type(value["bridge_id"]) is str and re.fullmatch("[0-9a-f]{32}", value["bridge_id"]) is not None)
    _need(value["descriptor_sha256"] == descriptor_sha256 and value["session_id"] == session_id,
          "worker_identity_mismatch")
    _need(value["status"] == "observed" and value["complete"] is True and value["reason"] is None
          and type(value["returncode"]) is int and value["returncode"] in (0, 2),
          "incomplete_worker_receipt")
    _need(value["worker_reported"] is True
          and all(value[key] is False for key in ("independently_verified", "provider_ingestion_verified",
              "token_savings_measured", "proof_authority", "execution_authority", "completion_authority",
              "raw_content_exported"))
          and type(value["observer_provider_calls"]) is int and value["observer_provider_calls"] == 0)
    _need(type(value["events"]) is list and len(value["events"]) <= MAX_EVENTS
          and _integer(value["owner_exchange_attempts"], 0, MAX_EVENTS)
          and _integer(value["owner_exchange_unconfirmed"], 0, MAX_EVENTS)
          and _integer(value["observed_responses"], 0, MAX_EVENTS))
    _need(value["owner_exchange_unconfirmed"] == 0
          and value["owner_exchange_attempts"] == value["observed_responses"] == len(value["events"]),
          "incomplete_worker_receipt")
    for ordinal, event in enumerate(value["events"], 1):
        _need(type(event) is dict and set(event) == _EVENT_FIELDS
              and type(event["exchange_ordinal"]) is int and event["exchange_ordinal"] == ordinal
              and owner_audit._digest(event["request_sha256"])
              and owner_audit._digest(event["owner_response_sha256"])
              and _integer(event["owner_response_bytes"], 1, protocol.MAX_RESPONSE_FRAME_BYTES)
              and type(event["owner_ok"]) is bool
              and type(event["output_kind"]) is str
              and event["output_kind"] in {"owner_message", "local_error", "no_output"}
              and type(event["forwarding_status"]) is str
              and event["forwarding_status"] in {"write_completed", "no_output", "budget_refused", "unconfirmed"})
        _need(event["owner_reason_code"] is None if event["owner_ok"] else
              type(event["owner_reason_code"]) is str and event["owner_reason_code"] in protocol.REASON_CODES)
        if event["output_kind"] == "no_output":
            _need(event["stdout_sha256"] is None and type(event["stdout_bytes"]) is int
                  and event["stdout_bytes"] == 0
                  and event["forwarding_status"] in {"no_output", "budget_refused"})
        else:
            _need(owner_audit._digest(event["stdout_sha256"])
                  and _integer(event["stdout_bytes"], 1, protocol.MAX_MESSAGE_RESPONSE_BYTES + 1)
                  and event["forwarding_status"] != "no_output")
            _need(event["owner_ok"] is (event["output_kind"] == "owner_message"))
            if event["output_kind"] == "owner_message":
                _need(event["stdout_bytes"] <= event["owner_response_bytes"])
        stopped = event["owner_ok"] is False or event["forwarding_status"] in {"budget_refused", "unconfirmed"}
        _need(not stopped or ordinal == len(value["events"]) and value["returncode"] == 2)
    return value


def summarize_worker_forwarding(*, audit_raw, operations_raw, row, forwarding_receipts):
    """Recompute a complete reported population; retain nulls for unknowns.

    No file, live descriptor, scheduler, provider or control database is opened.
    Matching worker claims remain worker claims even if every hash matches.
    The population covers validated owner-response events, not local parse
    errors or every byte written to stdout by an adapter or another process.
    Unconfirmed stdout operations are all prepared owner operations lacking a
    reported preserved-result pipe write, including budget-replaced results.
    """
    try:
        owner = owner_audit.summarize_owner_session(audit_raw=audit_raw,
                                                  operations_raw=operations_raw, row=row)
    except Exception:
        return unavailable("owner_evidence_unavailable")
    base = unavailable()
    base.update(**{key: owner[key] for key in ("audit_sha256", "operations_sha256",
                                              "descriptor_sha256", "invocation_sha256")})
    def unknown(reason):
        result = dict(base)
        result["reason"] = reason
        return result
    if owner["operation_correlation"]["status"] != "observed":
        return unknown("owner_correlation_unavailable")
    if type(forwarding_receipts) is not list or not 1 <= len(forwarding_receipts) <= MAX_RECEIPTS:
        return unknown("worker_evidence_unavailable")
    try:
        audit = protocol.strict_json_loads(audit_raw)
        descriptor = audit["session"]
        receipts = [_receipt(raw, owner["descriptor_sha256"], descriptor["session_id"])
                    for raw in forwarding_receipts]
        _need(len({receipt["bridge_id"] for receipt in receipts}) == len(receipts),
              "ambiguous_worker_population")
        events = [event for receipt in receipts for event in receipt["events"]]
        _need(len(events) <= descriptor["max_requests"], "ambiguous_worker_population")
        lookup = {}
        for event in events:
            key = _event_key(event)
            _need(key not in lookup, "ambiguous_worker_population")
            lookup[key] = event
        owner_requests = audit["requests"]
        _need(len(owner_requests) == len(lookup)
              and all(owner_audit._digest(record["request_sha256"]) for record in owner_requests)
              and len({_owner_key(record) for record in owner_requests}) == len(owner_requests),
              "incomplete_owner_population")
        _need(set(lookup) == {_owner_key(record) for record in owner_requests}, "owner_response_mismatch")
        ordinal_by_key = {_owner_key(record): record["ordinal"] for record in owner_requests}
        for receipt in receipts:
            ordinals = [ordinal_by_key[_event_key(event)] for event in receipt["events"]]
            _need(all(before < after for before, after in zip(ordinals, ordinals[1:])),
                  "ambiguous_worker_population")
        counts = dict.fromkeys(COUNTERS, 0)
        operations = [protocol.strict_json_loads(line, max_bytes=owner_audit.MAX_OPERATIONS_BYTES)
                      for line in operations_raw.splitlines()]
        for record in owner_requests:
            event = lookup[_owner_key(record)]
            if record["delivered"]:
                _need(event["owner_ok"] is (record["status"] == "ok"), "owner_response_mismatch")
                if record["status"] != "ok":
                    _need(event["owner_reason_code"] == record["status"], "owner_response_mismatch")
            _need(not record["operation_capture"]["operations"] or event["output_kind"] != "no_output",
                  "owner_response_mismatch")
            for reference in record["operation_capture"]["operations"]:
                operation = operations[reference["sequence"] - 1]
                preserved = (event["owner_ok"] is True and record["prepared_response_sha256"] is not None
                    and record["prepared_response_sha256"] == record["response_sha256"]
                    and record["prepared_response_bytes"] == record["response_bytes"])
                # The client may receive a complete reply before a final owner
                # send checkpoint fails. Never require owner.delivered=True.
                if preserved:
                    counts["reported_received_operations"] += 1
                    if operation["event"] == "tool":
                        counts["reported_received_tool_results"] += 1
                        counts["reported_received_successful_reads"] += (
                            operation["tool"] == "repository_read" and operation["success"])
                written = (preserved and event["output_kind"] == "owner_message"
                           and event["forwarding_status"] == "write_completed")
                if written:
                    counts["reported_completed_stdout_operations"] += 1
                    if operation["event"] == "tool":
                        counts["reported_completed_stdout_tool_results"] += 1
                        counts["reported_completed_stdout_successful_reads"] += (
                            operation["tool"] == "repository_read" and operation["success"])
                else:
                    counts["unconfirmed_stdout_operations"] += 1
        result = dict(base)
        result.update(status="observed", reason=None, counts=counts,
            forwarding_receipt_sha256s=[_sha(raw) for raw in forwarding_receipts],
            forwarding_receipt_count=len(receipts))
        return result
    except _Invalid as error:
        return unknown(error.reason)
    except (ValueError, KeyError, TypeError, UnicodeError, OverflowError, RecursionError):
        return unknown("invalid_worker_receipt")


def summarize_captured_worker_forwarding(*, audit_raw, operations_raw, row):
    """Join copied invocation-row receipts without reopening a worker outbox.

    Canonical capture and target hashes identify retained metadata, not an
    authenticated worker observation. Complete coverage refers to the owner
    request population; it does not establish a complete helper inventory.
    """
    result = summarize_worker_forwarding(audit_raw=audit_raw, operations_raw=operations_raw,
        row=row, forwarding_receipts=[])
    result.update(capture_sha256=None, target_sha256=None,
                  worker_invocation_population_verified=False)
    if result["reason"] in {"owner_evidence_unavailable", "owner_correlation_unavailable"}:
        return result
    capture = row.get("repository_source_worker_forwarding")
    target = row["repository_source_policy"].get("worker_forwarding")
    if capture is None or target is None:
        result["reason"] = "worker_capture_unavailable"
        return result
    try:
        binding = row["repository_source_policy"]
        capture_api.validate_target(target, workspace=row["workspace"],
            session_id=binding["session_id"], descriptor_sha256=binding["descriptor_sha256"],
            invocation_id=row["invocation_id"])
        _need(target["uid"] == binding["descriptor"]["worker_uid"],
              "worker_capture_identity_mismatch")
        receipts = capture_api.captured_receipt_bytes(capture, target)
        capture_sha256 = _sha(protocol.canonical_json(capture))
        target_sha256 = _sha(protocol.canonical_json(target))
        result = summarize_worker_forwarding(audit_raw=audit_raw, operations_raw=operations_raw,
            row=row, forwarding_receipts=receipts)
        result.update(capture_sha256=capture_sha256, target_sha256=target_sha256,
                      worker_invocation_population_verified=False)
        return result
    except _Invalid as error:
        result["reason"] = error.reason
    except (ValueError, KeyError, TypeError, UnicodeError, OverflowError, RecursionError):
        result["reason"] = ("worker_capture_unavailable" if type(capture) is dict
            and capture.get("status") == "unavailable" else "invalid_worker_capture")
    return result


def aggregate(observations):
    """Sum only complete selected-session reported forwarding populations."""
    known = sum(item["status"] == "observed" for item in observations)
    complete = bool(observations) and known == len(observations)
    result = unavailable("incomplete_worker_forwarding_population")
    result.update(schema="terminal-repository-worker-forwarding-population@1",
        status="observed" if complete else "unavailable", reason=None if complete else result["reason"],
        sessions=observations, session_count=len(observations), known_session_count=known,
        unknown_session_count=len(observations) - known,
        worker_invocation_population_verified=False,
        counts={key: sum(item["counts"][key] for item in observations) for key in COUNTERS}
            if complete else None)
    return result
