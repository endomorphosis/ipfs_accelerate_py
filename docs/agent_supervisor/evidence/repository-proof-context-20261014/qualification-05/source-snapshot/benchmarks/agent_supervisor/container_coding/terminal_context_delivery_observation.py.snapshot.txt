"""Compose retained context and native-usage observations without private I/O.

Source-reader sessions have already been replayed by their existing collector.
This adapter binds their bounded count projections to router invocation IDs;
it does not reread owner audits or establish provider ingestion. Repository MCP
counters describe prepared replies, which have no delivery acknowledgement.
"""
from __future__ import annotations

import json
import re


SCHEMA = "terminal-context-delivery-observation@1"
MAX_INVOCATIONS = 128
MAX_SOURCE_SESSIONS = 16
MAX_OBSERVATION_BYTES = 4 * 1024 * 1024
MAX_COUNTER = (1 << 63) - 1
SOURCE_COUNTS = (
    "accepted_requests", "delivered_responses", "prepared_pages", "delivered_pages",
    "prepared_source_bytes", "delivered_source_bytes", "response_bytes",
    "delivered_response_bytes", "request_hash_count", "response_hash_count",
)
PAGE_BYTE_COUNTS = ("prepared_pages", "delivered_pages", "prepared_bytes", "delivered_bytes")
PAGE_ROW_COUNTS = ("prepared_pages", "delivered_pages", "prepared_rows", "delivered_rows")
MCP_COUNTS = (
    "initializations", "tool_calls", "successful_reads", "compact_reads",
    "canonical_edits", "uncertain_edits", "refreshes", "errors", "returned_bytes",
)
_SESSION_HASHES = (
    "session_id_sha256", "invocation_id_sha256", "audit_sha256", "descriptor_sha256",
    "semantic_sha256", "source_manifest_sha256",
)
_SOURCE_AUTHORITY = ("proof_authority", "execution_authority", "completion_authority",
                     "settlement_authority", "retry_authority")
_MCP_AUTHORITY = ("native_tool_interception", "helper_adoption_guaranteed",
                  "token_savings_measured", "proof_authority", "completion_authority")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _counter(value):
    return type(value) is int and 0 <= value <= MAX_COUNTER


def _counts(value, fields):
    _need(type(value) is dict and set(value) == set(fields)
          and all(_counter(value[key]) for key in fields), "invalid source-reader counters")
    return {key: value[key] for key in fields}


def _page_counts(value, fields):
    if value is None:
        return None
    result = _counts(value, fields)
    _need(result["delivered_pages"] <= result["prepared_pages"], "invalid delivered page count")
    detail = "bytes" if fields == PAGE_BYTE_COUNTS else "rows"
    _need(result["delivered_" + detail] <= result["prepared_" + detail],
          "invalid delivered page detail count")
    return result


def _bounded_totals(values, fields):
    totals = dict.fromkeys(fields, 0)
    for value in values:
        for key in fields:
            _need(value[key] <= MAX_COUNTER - totals[key], "source_counter_aggregate_overflow")
            totals[key] += value[key]
    return totals


def _session(value):
    _need(type(value) is dict and all(_sha(value.get(key)) for key in _SESSION_HASHES),
          "invalid source-reader session identity")
    profile = value.get("profile")
    _need(profile in {"identity@1", "python-comments@1", "python-whitespace@1"},
          "unsupported source-reader profile")
    counts = _counts(value.get("counts"), SOURCE_COUNTS)
    _need(counts["delivered_responses"] <= counts["accepted_requests"]
          and counts["prepared_pages"] <= counts["accepted_requests"]
          and counts["delivered_pages"] <= counts["prepared_pages"]
          and counts["delivered_pages"] <= counts["delivered_responses"]
          and counts["prepared_source_bytes"] <= counts["response_bytes"]
          and counts["delivered_source_bytes"] <= counts["prepared_source_bytes"]
          and counts["delivered_source_bytes"] <= counts["delivered_response_bytes"]
          and counts["delivered_response_bytes"] <= counts["response_bytes"],
          "inconsistent source-reader counters")
    result = {key: value[key] for key in _SESSION_HASHES}
    result.update(profile=profile, counts=counts,
        candidate_context_counts=_page_counts(value.get("candidate_context_counts"), PAGE_BYTE_COUNTS),
        discovery_counts=_page_counts(value.get("discovery_counts"), PAGE_ROW_COUNTS))
    return result


def _source_sessions(report, invocations, raw_invocations):
    expected = {row["invocation_id_sha256"] for row, raw in zip(invocations, raw_invocations)
                if "source_view_retrieval" in raw}
    by_invocation = {}
    for row in invocations:
        by_invocation.setdefault(row["invocation_id_sha256"], []).append(row)
    source = report.get("source_view_retrieval_observation")
    result = {
        "status": "not_observed", "reason": "retained_owner_observation_missing",
        "declared_session_count": None, "matched_session_count": 0,
        "selected_invocation_count": len(expected), "missing_selected_invocation_count": len(expected),
        "sessions": [], "counts": None, "candidate_context_counts": None, "discovery_counts": None,
        "owner_audit_replayed_here": False,
        "response_byte_scope": "canonical JSON response bodies; excludes frame prefixes and CLI newlines",
        "source_byte_scope": "source slices including repeated reads; excludes candidate and discovery artifacts",
        "delivery_scope": "successful owner sends; no receiver or model receipt acknowledgement; ingestion and displayed text unknown",
        "provider_ingestion_observed": None, "tool_token_attribution": None,
    }
    if source is None:
        return result
    if (type(source) is not dict or source.get("schema") != "terminal-source-view-observation@1"
            or source.get("observation_only") is not True
            or any(source.get(key) is not False for key in _SOURCE_AUTHORITY)):
        return dict(result, status="unavailable", reason="invalid_retained_owner_observation")
    state = source.get("status")
    if type(state) is not str:
        return dict(result, status="unavailable", reason="invalid_retained_owner_status")
    if state != "observed":
        if state in {"unavailable", "not_selected", "not_applicable"}:
            return dict(result, status=state, reason="retained_owner_observation_" + state)
        return dict(result, status="unavailable", reason="invalid_retained_owner_status")
    sessions = source.get("sessions")
    if type(sessions) is not list or len(sessions) > MAX_SOURCE_SESSIONS:
        return dict(result, status="unavailable", reason="invalid_owner_session_population")
    result["declared_session_count"] = len(sessions)
    invocation_counts, session_counts = {}, {}
    for value in sessions:
        if type(value) is dict:
            for key, target in (("invocation_id_sha256", invocation_counts),
                                ("session_id_sha256", session_counts)):
                if _sha(value.get(key)):
                    target[value[key]] = target.get(value[key], 0) + 1
    matched = set()
    for ordinal, value in enumerate(sessions):
        item = {"session_ordinal": ordinal, "status": "invalid", "reason": "invalid_session_projection",
                "invocation_id_sha256": value.get("invocation_id_sha256")
                    if type(value) is dict and _sha(value.get("invocation_id_sha256")) else None,
                "session": None}
        try:
            projected = _session(value)
            key = projected["invocation_id_sha256"]
            if invocation_counts[key] != 1 or session_counts[projected["session_id_sha256"]] != 1:
                item.update(status="ambiguous", reason="duplicate_owner_session_binding")
            elif len(by_invocation.get(key, [])) != 1:
                item.update(status="unmatched", reason="invocation_identity_missing_or_ambiguous")
            elif key not in expected or by_invocation[key][0]["purpose"] != "coding":
                item.update(status="unmatched", reason="source_session_not_selected_for_coding_invocation")
            else:
                item.update(status="matched", reason="retained_owner_session_joined", session=projected)
                matched.add(key)
        except (ValueError, TypeError, KeyError):
            pass
        result["sessions"].append(item)
    result["matched_session_count"] = len(matched)
    result["missing_selected_invocation_count"] = len(expected - matched)
    complete = bool(sessions) and len(matched) == len(sessions) and not (expected - matched)
    result.update(status="observed" if complete else "partial",
                  reason="complete_retained_session_join" if complete else "incomplete_retained_session_join")
    if complete:
        values = [item["session"] for item in result["sessions"]]
        try:
            counts = _bounded_totals([value["counts"] for value in values], SOURCE_COUNTS)
            optional = {}
            for name, fields in (("candidate_context_counts", PAGE_BYTE_COUNTS), ("discovery_counts", PAGE_ROW_COUNTS)):
                if all(value[name] is not None for value in values):
                    optional[name] = _bounded_totals([value[name] for value in values], fields)
            result.update(counts=counts, **optional)
        except ValueError:
            result.update(status="partial", reason="source_counter_aggregate_overflow",
                          counts=None, candidate_context_counts=None, discovery_counts=None)
    return result


def _repository_mcp(invocations, raw_invocations):
    from .terminal_repository_source_policy import _usage_observation

    rows = []
    for invocation, raw in zip(invocations, raw_invocations):
        if "repository_source_policy" not in raw:
            continue
        binding = raw["repository_source_policy"]
        valid = (type(binding) is dict and binding.get("schema") == "supervisor-repository-source-policy@1"
            and type(binding.get("requested_policy")) is str
            and binding.get("requested_policy") in {"repository-source-raw@1", "repository-source-compact@1"}
            and all(binding.get(key) is False for key in _MCP_AUTHORITY)
            and invocation["provider"] == "codex_cli" and invocation["purpose"] == "coding")
        usage = binding.get("usage") if type(binding) is dict else None
        fields = {"status", *MCP_COUNTS, "store_cleanup_complete", "artifacts_retained_after_invocation",
                  "store_cleanup_error"}
        usage = {key: usage[key] for key in fields if key in usage} if type(usage) is dict else None
        observed = _usage_observation(usage)
        valid = valid and observed["server_audit_valid"]
        rows.append({"invocation_id_sha256": invocation["invocation_id_sha256"],
            "status": observed["server_audit_status"] if valid else "unavailable",
            "requested_policy": binding["requested_policy"] if valid else None,
            "prepared_counts": {key: observed[key] if valid else None for key in MCP_COUNTS},
            "store_cleanup_complete": observed["store_cleanup_complete"] if valid else None})
    return {
        "status": "not_selected" if not rows else "observed" if all(row["status"] == "observed" for row in rows) else "partial",
        "observations": rows, "selected_invocation_count": len(rows),
        "count_scope": "prepared repository MCP replies; returned_bytes includes metadata and repeated replies",
        "delivery_acknowledgements": None, "displayed_source_bytes": None,
        "provider_ingestion_observed": None, "tool_token_attribution": None,
        "audit_reread_here": False,
    }


def collect(*, report: dict) -> dict:
    """Compose complete supplied receipt populations; never infer whole-run usage."""
    from ipfs_accelerate_py.agent_supervisor.runtime import context_delivery_accounting as core

    _need(type(report) is dict, "terminal report must be an object")
    raw = report.get("provider_invocations", [])
    _need(type(raw) is list and len(raw) <= MAX_INVOCATIONS and all(type(row) is dict for row in raw),
          "bounded complete provider invocation population required")
    observations = []
    for row in raw:
        projection = {key: row[key] for key in core.OBSERVE_INPUT_FIELDS if key in row}
        if "purpose" not in projection and "phase" in projection:
            projection["purpose"] = projection["phase"]
        observations.append(core.observe_invocation(projection))
    marker = report.get("unreceipted_provider_attempt")
    normalized = True if type(marker) is dict else marker if marker is None or type(marker) is bool else None
    summary = core.summarize_invocations(observations, unreceipted_provider_attempt=normalized)
    sources = _source_sessions(report, observations, raw)
    repository = _repository_mcp(observations, raw)
    result = {
        "schema": SCHEMA, "status": "observed", "native_invocations": summary,
        "supplied_invocation_count": len(raw),
        "unreceipted_attempt_marker": normalized,
        "unreceipted_attempt_marker_scope": "mapping_present" if type(marker) is dict else "boolean" if type(marker) is bool else "unknown",
        "source_view_retrieval": sources,
        "repository_source_mcp": repository,
        "join_coverage": {
            "supplied_router_invocations": len(raw),
            "source_owner_sessions_declared": sources["declared_session_count"],
            "source_owner_sessions_bound": sources["matched_session_count"],
            "selected_source_invocations_missing_session": sources["missing_selected_invocation_count"],
            "repository_mcp_invocations_with_counter_projection": repository["selected_invocation_count"],
            "source_owner_response_to_native_output_digest_matches": None,
            "ordinary_source_identity": None, "provider_ingestion_observed": None,
            "per_tool_provider_tokens": None, "source_owner_audit_replayed_here": False,
            "exact_source_response_digest_join_available": False,
            "ordinary_source_identity_available": False, "per_tool_provider_token_attribution_available": False,
        },
        "population_scope": "all provider_invocations supplied by this retained terminal report; no whole-campaign completeness claim",
        "context_scope": "rendered user prompts, native CLI message observations and retained owner-tool counters remain separate",
        "provider_ingestion_observed": None, "tool_token_causality": None,
        "observation_only": True, "private_artifacts_reread": False,
        "provider_usage_modified": False, "billing_total_verified": False,
        "raw_content_exported": False, "provider_ingestion_verified": False,
        "total_token_savings_measured": False, "token_savings_established": False,
        "proof_authority": False, "execution_authority": False,
        "settlement_authority": False, "completion_authority": False,
    }
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    _need(len(encoded) <= MAX_OBSERVATION_BYTES, "terminal context observation exceeds output bound")
    return result


__all__ = ["SCHEMA", "collect"]
