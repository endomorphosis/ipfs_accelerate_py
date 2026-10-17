"""Project retained owner summaries without reopening their original artifacts.

Counts and hashes below remain reported summary evidence. Structural joins and
sum checks do not repeat source-audit verification or authenticate worker reports.
This optional projection never supplies provider usage or tool-adoption claims.
"""
from __future__ import annotations

import hashlib
import re

from .terminal_repository_owner_audit import OPERATION_COUNTERS
from .terminal_repository_worker_forwarding import COUNTERS as WORKER_COUNTERS

SCHEMA = "terminal-benchmark-repository-owner-observation@1"
PREPARED_COUNTERS = ("initializations", "tool_calls", "successful_reads", "compact_reads",
    "canonical_edits", "uncertain_edits", "refreshes", "errors", "returned_bytes")
_AUTHORITY = ("proof_authority", "execution_authority", "completion_authority")
_OWNER_FALSE = (*_AUTHORITY, "provider_ingestion_verified", "token_savings_measured",
    "raw_content_exported", "native_parent_release_verified", "worker_cleanup_independently_verified")
_CORRELATION_FALSE = (*_AUTHORITY, "worker_receipt_verified", "provider_ingestion_verified",
    "response_payloads_independently_verified", "token_savings_measured")
_WORKER_FALSE = (*_AUTHORITY, "independent_worker_receipt_verified", "provider_ingestion_verified",
    "output_payloads_independently_verified", "token_savings_measured", "production_route_qualified")
_LIFECYCLE_FALSE = (*_AUTHORITY, "independent_scheduler_capacity_verified", "worker_reaping_verified",
    "worker_cleanup_independently_verified", "provider_ingestion_verified", "token_savings_measured")


def _need(value):
    if not value:
        raise ValueError("invalid_retained_owner_summary")


def _digest(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _count(value):
    return type(value) is int and 0 <= value <= 2**63 - 1


def _false(value, fields):
    _need(type(value) is dict and all(value.get(key) is False for key in fields))


def _counts(value, fields):
    _need(type(value) is dict and set(value) == set(fields) and all(_count(value[key]) for key in fields))
    return {key: value[key] for key in fields}


def _unknown(reason="missing_retained_summary"):
    return dict(status="unavailable", reason=reason, counts=None,
        known_session_count=None, unknown_session_count=None)


def _population(value, *, schema, counts, known, total):
    _need(type(value) is dict and value.get("schema") == schema
          and type(value.get("session_count")) is int and value["session_count"] == total
          and type(value.get("known_session_count")) is int and value["known_session_count"] == known
          and type(value.get("unknown_session_count")) is int and value["unknown_session_count"] == total - known)
    complete = known == total and total > 0
    _need(value.get("status") == ("observed" if complete else "unavailable"))
    if complete:
        _need(value.get("reason") is None and value.get("counts") == counts)
        _counts(value["counts"], counts)
    else:
        _need(value.get("counts") is None)
    return dict(status="observed" if complete else "unavailable",
        reason=None if complete else "incomplete_retained_population", counts=counts if complete else None,
        known_session_count=known, unknown_session_count=total - known)


def _operation_counts(session):
    value = session.get("operation_correlation")
    _false(value, _CORRELATION_FALSE)
    _need(value.get("schema") == "terminal-repository-owner-operation-correlation@1")
    if value.get("status") == "unavailable":
        _need(value.get("counts") is None)
        return None
    _need(value.get("status") == "observed" and value.get("reason") is None)
    row = _counts(value.get("counts"), OPERATION_COUNTERS)
    prepared = session["prepared_mcp_usage"]
    _need(row["total_operations"] == prepared["initializations"] + prepared["tool_calls"]
                and row["completed_socket_send_operations"] + row["unconfirmed_operations"] == row["total_operations"]
                and row["completed_socket_send_successful_reads"] <= prepared["successful_reads"]
                and row["completed_socket_send_tool_results"] <= prepared["tool_calls"]
                and row["completed_socket_send_successful_tool_results"] + row["completed_socket_send_failed_tool_results"]
                    == row["completed_socket_send_tool_results"]
                and row["completed_socket_send_successful_reads"] <= row["completed_socket_send_successful_tool_results"]
                    <= row["completed_socket_send_tool_results"] <= row["completed_socket_send_operations"]
                and row["completed_socket_send_native_payload_bytes"] <= prepared["returned_bytes"])
    return row


def _operation_population(raw, sessions):
    try:
        _false(raw, _CORRELATION_FALSE)
        known, counts = 0, dict.fromkeys(OPERATION_COUNTERS, 0)
        for session in sessions:
            row = _operation_counts(session)
            if row is None:
                continue
            known += 1
            for key in counts:
                counts[key] += row[key]
        return _population(raw, schema="terminal-repository-owner-operation-correlation-population@1",
                           counts=counts, known=known, total=len(sessions))
    except (ValueError, TypeError, KeyError):
        return _unknown("invalid_retained_operation_summary")


def _worker_population(raw, sessions):
    try:
        _false(raw, _WORKER_FALSE)
        _need(raw.get("worker_reported_only") is True
              and raw.get("worker_invocation_population_verified", False) is False)
        rows = raw.get("sessions")
        _need(type(rows) is list and len(rows) == len(sessions))
        by_invocation = {session["invocation_sha256"]: session for session in sessions}
        seen, known, counts = set(), 0, dict.fromkeys(WORKER_COUNTERS, 0)
        for value in rows:
            _false(value, _WORKER_FALSE)
            identity = value.get("invocation_sha256")
            _need(type(identity) is str and identity in by_invocation and identity not in seen
                  and value.get("schema") == "terminal-repository-worker-forwarding-observation@1"
                  and value.get("worker_reported_only") is True
                  and value.get("worker_invocation_population_verified", False) is False)
            seen.add(identity)
            if value.get("status") == "unavailable":
                _need(value.get("counts") is None)
                continue
            session = by_invocation[identity]
            _need(_operation_counts(session) is not None)
            _need(value.get("status") == "observed" and value.get("reason") is None
                  and all(value.get(key) == session[key] for key in
                          ("audit_sha256", "operations_sha256", "descriptor_sha256")))
            hashes = value.get("forwarding_receipt_sha256s")
            _need(type(hashes) is list and 1 <= len(hashes) <= 16 and all(_digest(item) for item in hashes)
                  and len(set(hashes)) == len(hashes) and type(value.get("forwarding_receipt_count")) is int
                  and value["forwarding_receipt_count"] == len(hashes))
            row = _counts(value.get("counts"), WORKER_COUNTERS)
            prepared = session["prepared_mcp_usage"]
            _need(row["reported_completed_stdout_operations"] <= row["reported_received_operations"]
                  <= prepared["initializations"] + prepared["tool_calls"]
                and row["reported_completed_stdout_operations"] + row["unconfirmed_stdout_operations"]
                    == prepared["initializations"] + prepared["tool_calls"]
                and row["reported_completed_stdout_tool_results"] <= row["reported_received_tool_results"]
                    <= prepared["tool_calls"]
                and row["reported_completed_stdout_successful_reads"] <= row["reported_received_successful_reads"]
                    <= prepared["successful_reads"]
                and row["reported_received_successful_reads"] <= row["reported_received_tool_results"]
                    <= row["reported_received_operations"]
                and row["reported_completed_stdout_successful_reads"] <= row["reported_completed_stdout_tool_results"]
                    <= row["reported_completed_stdout_operations"])
            known += 1
            for key in counts:
                counts[key] += row[key]
        result = _population(raw, schema="terminal-repository-worker-forwarding-population@1",
                             counts=counts, known=known, total=len(sessions))
        result.update(worker_reported_only=True, independent_worker_receipt_verified=False,
            worker_invocation_population_verified=False, provider_ingestion_verified=False)
        return result
    except (ValueError, TypeError, KeyError):
        return {**_unknown("invalid_retained_worker_summary"), "worker_reported_only": True,
            "independent_worker_receipt_verified": False, "worker_invocation_population_verified": False,
            "provider_ingestion_verified": False}


def _lifecycle_population(raw, sessions):
    result = dict(status="unavailable", reason="invalid_retained_lifecycle_summary",
        own_service_reservation_release_reported=None, known_session_count=None, unknown_session_count=None,
        independent_scheduler_capacity_verified=False, worker_cleanup_independently_verified=False)
    try:
        _false(raw, _LIFECYCLE_FALSE)
        _need(raw.get("borrowed_caller_release") is None)
        rows = raw.get("sessions")
        _need(type(rows) is list and len(rows) == len(sessions))
        lookup = {session["invocation_sha256"]: session for session in sessions}
        seen, known = set(), 0
        for value in rows:
            _false(value, _LIFECYCLE_FALSE)
            identity = value.get("invocation_sha256")
            _need(type(identity) is str and identity in lookup and identity not in seen
                  and value.get("schema") == "terminal-repository-owner-lifecycle-session-observation@1"
                  and value.get("borrowed_caller_release") is None)
            seen.add(identity)
            if value.get("status") == "unavailable":
                _need(value.get("own_service_reservation_release_reported") is None)
                continue
            _need(value.get("status") == "joined_complete" and value.get("reason") is None
                and value.get("own_service_reservation_release_reported") is True
                and value.get("session_audit_sha256") == lookup[identity]["audit_sha256"]
                and value.get("descriptor_sha256") == lookup[identity]["descriptor_sha256"]
                and _digest(value.get("retirement_sha256")))
            known += 1
        complete = known == len(sessions)
        _need(raw.get("schema") == "terminal-repository-owner-lifecycle-observation@1"
              and type(raw.get("session_count")) is int and raw["session_count"] == len(sessions)
              and type(raw.get("known_session_count")) is int and raw["known_session_count"] == known
              and type(raw.get("unknown_session_count")) is int and raw["unknown_session_count"] == len(sessions) - known
              and raw.get("status") == ("observed" if complete else "unavailable")
              and raw.get("own_service_reservation_release_reported") is (True if complete else None))
        result.update(status="observed" if complete else "unavailable",
            reason=None if complete else "incomplete_retained_population", known_session_count=known,
            unknown_session_count=len(sessions) - known,
            own_service_reservation_release_reported=True if complete else None)
    except (ValueError, TypeError, KeyError):
        pass
    return result


def project(report):
    """Select bounded summary facts; missing or invalid optional evidence is null."""
    result = dict(schema=SCHEMA, status="unavailable", reason="missing_retained_owner_summary",
        session_count=None, prepared_mcp_counts=None, sessions=[],
        operation_correlation=_unknown(), worker_forwarding_observation=_unknown(),
        lifecycle_observation=dict(status="unavailable", reason="missing_retained_summary",
                                  own_service_reservation_release_reported=None),
        retained_summary_only=True, source_audits_replayed_here=False,
        provider_usage_modified=False, token_savings_measured=False,
        provider_ingestion_verified=False, worker_receipts_independently_verified=False,
        tool_adoption_established=False, proof_authority=False, execution_authority=False,
        completion_authority=False, raw_content_exported=False)
    raw = report.get("repository_source_owner_observation")
    if raw is None:
        return result
    result["reason"] = "invalid_retained_owner_summary"
    try:
        _false(raw, _OWNER_FALSE)
        _need(raw.get("schema") == "terminal-repository-owner-observation@1"
              and raw.get("observation_only") is True
              and all(raw.get(key) is None for key in ("delivered_tool_calls", "delivered_reads", "model_consumed_tokens")))
        if raw.get("status") in {"unavailable", "not_selected"}:
            _need(raw.get("session_count") is None and raw.get("prepared_mcp_counts") is None)
            result.update(status=raw["status"], reason="retained_owner_summary_unavailable")
            return result
        sessions, rows = raw.get("sessions"), report.get("provider_invocations")
        _need(raw.get("status") == "observed" and raw.get("reason") is None
              and type(sessions) is list and 1 <= len(sessions) <= 16
              and type(raw.get("session_count")) is int and raw["session_count"] == len(sessions)
              and type(rows) is list and len(rows) <= 128 and all(type(row) is dict for row in rows)
              and report.get("unreceipted_provider_attempt") is None
              and type(report.get("worker_cleanup_returncode")) is int and report["worker_cleanup_returncode"] == 0)
        selected = [row for row in rows if type(row.get("repository_source_policy")) is dict
            and row["repository_source_policy"].get("schema") == "supervisor-repository-source-owner-binding@1"]
        _need(len(selected) == len(sessions) and all(row in selected for row in rows if row.get("phase") == "coding"))
        bindings, session_ids, configurations = {}, set(), set()
        for row in selected:
            identity = row.get("invocation_id")
            _need(type(identity) is str and re.fullmatch("[0-9a-f]{32}", identity)
                  and row.get("phase") == row.get("purpose") == "coding" and row.get("provider") == "codex_cli")
            key = hashlib.sha256(identity.encode()).hexdigest()
            binding = row["repository_source_policy"]
            session_id, config = binding.get("session_id"), binding.get("config_sha256")
            _need(key not in bindings and sum(item.get("invocation_id") == identity for item in rows) == 1
                  and type(session_id) is str and re.fullmatch("[0-9a-f]{32}", session_id)
                  and session_id not in session_ids and _digest(config) and config not in configurations
                  and binding.get("requested_policy") in {"repository-source-raw@1", "repository-source-compact@1"}
                  and binding["requested_policy"] == report.get("repository_source_policy"))
            session_ids.add(session_id)
            configurations.add(config)
            bindings[key] = binding
        totals, projected, seen = dict.fromkeys(PREPARED_COUNTERS, 0), [], set()
        for session in sessions:
            _false(session, _AUTHORITY)
            identity = session.get("invocation_sha256")
            _need(type(identity) is str and identity in bindings and identity not in seen
                  and session.get("schema") == "terminal-repository-owner-session-observation@1"
                  and session.get("status") == "joined_complete"
                  and session.get("live_authority_verified") is False
                  and all(_digest(session.get(key)) for key in
                          ("audit_sha256", "operations_sha256", "descriptor_sha256"))
                  and session["descriptor_sha256"] == bindings[identity].get("descriptor_sha256")
                  and session.get("worker_config_sha256") == bindings[identity].get("config_sha256"))
            seen.add(identity)
            prepared = session.get("prepared_mcp_usage")
            _need(type(prepared) is dict)
            counts = _counts({key: prepared.get(key) for key in PREPARED_COUNTERS}, PREPARED_COUNTERS)
            for key in totals:
                totals[key] += counts[key]
            projected.append({key: session[key] for key in
                ("invocation_sha256", "audit_sha256", "operations_sha256", "descriptor_sha256")})
        _need(_counts(raw.get("prepared_mcp_counts"), PREPARED_COUNTERS) == totals)
        result.update(status="observed", reason=None, session_count=len(sessions), prepared_mcp_counts=totals,
            sessions=projected, operation_correlation=_operation_population(raw.get("operation_correlation"), sessions),
            worker_forwarding_observation=_worker_population(raw.get("worker_forwarding_observation"), sessions),
            lifecycle_observation=_lifecycle_population(raw.get("lifecycle_observation"), sessions))
    except (ValueError, TypeError, KeyError):
        pass
    return result
