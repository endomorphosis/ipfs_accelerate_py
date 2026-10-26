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


def _inline_source_overlap_base(invocation):
    return {
        "schema": "terminal-inline-source-read-overlap@1",
        "status": "not_observed", "reason": "initial_inline_sources_not_recorded",
        "scope": "declared_initial_source_carrier_vs_prepared_original_file_reads",
        "initial_source_carrier_scope": None, "initial_source_observation_schema": None,
        "logical_byte_scope": "decoded_utf8_source_text",
        "invocation_id_sha256": invocation["invocation_id_sha256"],
        "model_prompt_sha256": invocation["model_prompt"]["sha256"],
        "model_prompt_bytes": invocation["model_prompt"]["bytes"],
        "known_matched_reads": None, "known_matched_logical_bytes": None,
        "matched_reads": None, "matched_logical_bytes": None,
        "comparable_reads": None, "unclassified_reads": None, "comparison_complete": None,
        "inline_overlap_bytes": None, "provider_tokens": None, "token_savings": None,
        "provider_ingestion_observed": None, "full_prompt_source_coverage": None,
        "retained_input_audit_joined": False, "audit_reread_here": False,
        "journal_closure_verified": False, "native_usage_modified": False,
        "raw_content_exported": False, "proof_authority": False, "completion_authority": False,
    }


def _inline_source_overlap_checked(report, invocations, invocation, repository_row):
    """Join declared initial source slots to prepared reads, never token usage."""
    result = _inline_source_overlap_base(invocation)

    def unavailable(reason):
        return {**result, "status": "unavailable", "reason": reason}

    if sum(row["invocation_id"] == invocation["invocation_id"] for row in invocations) != 1:
        return unavailable("ambiguous_invocation_population")
    if repository_row["status"] == "unavailable" or repository_row.get("read_observation_valid") is False:
        return unavailable("repository_read_observation_unavailable")
    read_summary = repository_row.get("read_observation")
    if read_summary is None:
        return {**result, "reason": "repository_read_observation_not_recorded"}
    audit = report.get("context_input_audit")
    if audit is None:
        return result
    if (type(audit) is not dict or audit.get("schema") != "terminal-final-context-audit@1"
            or type(audit.get("status")) is not str
            or audit.get("status") not in {"verified_all_observed_model_inputs", "verified_some_observed_model_inputs"}
            or audit.get("any_model_input_verified") is not True
            or any(audit.get(key) is not False for key in ("proof_authority", "execution_authority",
                "completion_authority", "raw_prompts_exported"))
            or type(audit.get("provider_calls")) is not int or audit["provider_calls"] != 0):
        return unavailable("invalid_context_input_audit")
    census = audit.get("rendered_input_census")
    if census is None:
        return result
    if (type(census) is not dict or census.get("schema") != "terminal-rendered-input-census@1"
            or type(census.get("status")) is not str
            or census.get("status") not in {"complete_for_verified_inputs", "partial_for_verified_inputs"}
            or census.get("raw_content_exported") is not False
            or census.get("token_savings_established") is not False
            or type(census.get("provider_calls")) is not int or census["provider_calls"] != 0):
        return unavailable("invalid_rendered_input_census")
    populations = (audit.get("invocations"), audit.get("matches"), census.get("observations"))
    if any(type(rows) is not list or len(rows) > limit or any(type(row) is not dict for row in rows)
           for rows, limit in zip(populations, (16, 512, 16))):
        return unavailable("invalid_input_join_population")
    audited, matches, observations = populations
    absent = census.get("unavailable")
    if (type(absent) is not list or len(absent) > 16
            or any(type(row) is not dict or not _sha(row.get("invocation_id_sha256")) for row in absent)):
        return unavailable("invalid_rendered_input_census")
    if (type(census.get("observed_invocation_count")) is not int
            or census["observed_invocation_count"] != len(observations)
            or type(census.get("verified_invocation_count")) is not int
            or not len(observations) <= census["verified_invocation_count"] <= 16
            or (census["status"] == "complete_for_verified_inputs"
                and len(observations) != census["verified_invocation_count"])):
        return unavailable("invalid_rendered_input_census")
    audited = [row for row in audited if row.get("invocation_id") == invocation["invocation_id"]]
    matches = [row for row in matches if row.get("invocation_id") == invocation["invocation_id"]]
    observations = [row for row in observations if row.get("invocation_id_sha256") == invocation["invocation_id_sha256"]]
    if observations and any(row["invocation_id_sha256"] == invocation["invocation_id_sha256"] for row in absent):
        return unavailable("contradictory_census_observation")
    if any(len(rows) > 1 for rows in (audited, matches, observations)):
        return unavailable("ambiguous_exact_input_join")
    if any(len(rows) != 1 for rows in (audited, matches, observations)):
        return unavailable("missing_exact_input_join")
    audited, match, observation = audited[0], matches[0], observations[0]
    if "inline_sources" not in observation:
        return result
    checks = match.get("model_input_checks")
    if (audited.get("status") != "verified_exact_model_input"
            or audited.get("model_input_verified") is not True
            or audited.get("native_input_verified") is not True
            or match.get("model_input_verified") is not True
            or type(checks) is not dict or not 2 <= len(checks) <= 256
            or checks.get("model_prompt_sha256") is not True
            or checks.get("model_prompt_bytes") is not True
            or any(value is not True for value in checks.values())
            or observation.get("schema") != "supervisor-rendered-input-census@1"
            or observation.get("raw_content_exported") is not False):
        return unavailable("unverified_exact_input_join")
    prompt = invocation["model_prompt"]
    if (not _sha(prompt["sha256"]) or type(prompt["bytes"]) is not int
            or not 0 < prompt["bytes"] <= 256_000
            or any(row.get("model_prompt_sha256") != prompt["sha256"]
                   or type(row.get("model_prompt_bytes")) is not int
                   or row["model_prompt_bytes"] != prompt["bytes"] for row in (match, observation))):
        return unavailable("final_prompt_binding_mismatch")
    from ipfs_accelerate_py.agent_supervisor.runtime.inline_source_observation import validate_inline_sources
    try:
        inline = validate_inline_sources(observation["inline_sources"],
            model_prompt_sha256=prompt["sha256"], model_prompt_bytes=prompt["bytes"],
            invocation_id_sha256=invocation["invocation_id_sha256"])
    except (ValueError, TypeError, KeyError):
        return unavailable("invalid_inline_sources")
    if inline["status"] != "observed":
        return unavailable("inline_sources_" + inline["reason"])
    entries = {(entry["source_identity_sha256"], entry["content_sha256"], entry["content_bytes"])
               for entry in inline["entries"]}
    comparable = matched = logical_bytes = 0
    for group in read_summary["groups"]:
        if (group["scope"], group["representation"], group["encoding"]) != ("file", "original", "utf-8"):
            continue
        comparable += group["read_count"]
        if (group["source_identity_sha256"], group["content_sha256"], group["content_bytes"]) in entries:
            matched += group["read_count"]
            logical_bytes += group["read_count"] * group["content_bytes"]
    unclassified = read_summary["successful_reads"] - comparable
    complete = read_summary["complete"] is True and unclassified == 0
    return {**result, "status": "observed" if complete else "partial",
        "reason": "all_prepared_original_file_reads_classified" if complete else "partially_classified_prepared_reads",
        "known_matched_reads": matched, "known_matched_logical_bytes": logical_bytes,
        "matched_reads": matched if complete else None, "matched_logical_bytes": logical_bytes if complete else None,
        "comparable_reads": comparable, "unclassified_reads": unclassified, "comparison_complete": complete,
        "initial_source_carrier_scope": inline["scope"], "initial_source_observation_schema": inline["schema"],
        "retained_input_audit_joined": True}


def _inline_source_overlap(report, invocations, invocation, repository_row):
    try:
        return _inline_source_overlap_checked(report, invocations, invocation, repository_row)
    except Exception:
        # Optional diagnostics cannot invalidate independently valid counts or
        # disclose exception text. Native usage is composed separately.
        return {**_inline_source_overlap_base(invocation), "status": "unavailable",
                "reason": "inline_source_diagnostic_failure"}


def _repository_mcp(invocations, raw_invocations, report):
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
                  "store_cleanup_error", "read_observation"}
        usage = {key: usage[key] for key in fields if key in usage} if type(usage) is dict else None
        observed = _usage_observation(usage)
        valid = valid and observed["server_audit_valid"]
        row = {"invocation_id_sha256": invocation["invocation_id_sha256"],
            "status": observed["server_audit_status"] if valid else "unavailable",
            "requested_policy": binding["requested_policy"] if valid else None,
            "prepared_counts": {key: observed[key] if valid else None for key in MCP_COUNTS},
            "store_cleanup_complete": observed["store_cleanup_complete"] if valid else None,
            **({"read_observation_valid": bool(valid and observed["read_observation_valid"]),
                "read_observation": observed["read_observation"] if valid and observed["read_observation_valid"] else None}
               if "read_observation_valid" in observed else {})}
        row["inline_source_overlap"] = _inline_source_overlap(report, invocations, invocation, row)
        rows.append(row)
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
    repository = _repository_mcp(observations, raw, report)
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
    if len(encoded) > MAX_OBSERVATION_BYTES:
        # Optional source-overlap enrichment yields the existing output budget
        # before independently valid base context and native measurements.
        for row in repository["observations"]:
            row.pop("inline_source_overlap", None)
        encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    _need(len(encoded) <= MAX_OBSERVATION_BYTES, "terminal context observation exceeds output bound")
    return result


__all__ = ["SCHEMA", "collect"]
