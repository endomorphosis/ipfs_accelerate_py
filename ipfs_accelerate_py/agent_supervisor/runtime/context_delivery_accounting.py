"""Bounded context observations joined to supplied native usage receipts.

This module opens no files or databases and calls no providers or tokenizers.
Prompt bytes, native recorded tool payloads and owner responses have different
measurement scopes. None is an additional token charge. A complete supplied
native population does not prove that every campaign call was supplied or that
the provider ingested particular source text.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

INVOCATION_SCHEMA = "supervisor-context-delivery-invocation@1"
SUMMARY_SCHEMA = "supervisor-context-delivery-summary@1"
OBSERVE_INPUT_FIELDS = frozenset({
    "schema", "invocation_id", "provider", "model", "purpose", "phase", "status",
    "error_type", "model_prompt_sha256", "model_prompt_bytes", "workspace",
    "native_rollout_usage", "native_context_delivery",
})
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_PROMPT_BYTES = 256_000
MAX_OBSERVATION_BYTES = 192 * 1024
MAX_AGGREGATE_OBSERVATION_BYTES = 8 * 1024 * 1024
MAX_INVOCATIONS = 512
MAX_NODES = 32_768
MAX_DEPTH = 32
MAX_COUNTER = (1 << 63) - 1
MAX_RECORDS = 100_000
COUNTERS = (
    "input_tokens", "cached_input_tokens", "cache_write_input_tokens",
    "output_tokens", "reasoning_output_tokens", "total_tokens",
)
NORMALIZED_COUNTERS = (
    "input_tokens_including_cache", "fresh_input_tokens", "cached_input_tokens",
    "cache_write_input_tokens", "output_tokens", "reasoning_output_tokens", "total_tokens",
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,255}\Z")
_AUTHORITY = dict(observation_only=True, billing_total_verified=False,
    provider_delivery_verified=False, provider_ingestion_verified=False,
    raw_content_exported=False, total_token_savings_measured=False,
    token_savings_established=False, completion_authority=False)
_DELIVERY_COUNTS = ("observed_records", "observed_tool_call_events", "observed_tool_output_events",
    "observed_tool_events", "retained_tool_events", "omitted_tool_events",
    "payloads_without_fingerprint", "measured_payload_bytes")
_DELIVERY_COVERAGE = ("ledger_truncated", "observation_census_complete",
    "tool_observations_complete", "fingerprints_complete")
_DELIVERY_FIELDS = frozenset({"status", "ledger_schema", "ledger_sha256", "source_ledger_sha256",
    "interaction_telemetry_sha256", "telemetry_binding_verified", "counts", "payload_measurements", "coverage"})
_NATIVE_FIELDS = frozenset({"schema", "status", "scope", "identity_kind",
    "session_id_sha256", "artifact_sha256", "observed_records", "complete",
    "counts", "normalized_counts", "reasons"})
_ROW_FIELDS = frozenset({"schema", "invocation_id", "invocation_id_sha256", "provider",
    "model", "purpose", "phase", "status", "error_type", "receipt_sha256",
    "workspace_binding_sha256", "model_prompt", "native_usage", "native_context_delivery",
    "observation_sha256", *_AUTHORITY})


class ContextDeliveryAccountingError(ValueError):
    """An unsupported input shape, bound or inconsistent sealed observation."""


def _need(condition, reason):
    if not condition:
        raise ContextDeliveryAccountingError(reason)


def _wire(value, maximum):
    """Bound traversal and string expansion before canonical serialization."""
    pending = [(value, 0)]
    nodes = characters = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _need(nodes <= MAX_NODES and depth <= MAX_DEPTH, "input_structure_bound")
        if type(item) is str:
            characters += len(item)
        elif type(item) is dict:
            _need(len(item) + len(pending) <= MAX_NODES and all(type(k) is str for k in item),
                  "input_structure_bound")
            characters += sum(len(k) for k in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            _need(len(item) + len(pending) <= MAX_NODES, "input_structure_bound")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is int:
            _need(item.bit_length() <= 64, "input_integer_bound")
        elif type(item) is float:
            _need(math.isfinite(item), "nonfinite_input")
        else:
            _need(item is None or type(item) is bool, "non_json_input")
        _need(characters <= maximum, "input_byte_bound")
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise ContextDeliveryAccountingError("non_json_input") from error
    _need(len(raw) <= maximum, "input_byte_bound")
    return raw


def _digest(value, maximum=MAX_OBSERVATION_BYTES):
    return hashlib.sha256(_wire(value, maximum)).hexdigest()


def _identity(value, label, *, optional=False):
    if optional and value is None:
        return None
    _need(type(value) is str and _IDENTITY.fullmatch(value) is not None, "invalid_" + label)
    return value


def _sha(value, label, *, optional=False):
    if optional and value is None:
        return None
    _need(type(value) is str and _SHA.fullmatch(value) is not None, "invalid_" + label)
    return value


def _count(value, label, *, optional=False, maximum=MAX_COUNTER):
    if optional and value is None:
        return None
    _need(type(value) is int and 0 <= value <= maximum, "invalid_" + label)
    return value


def _boolean(value, label, *, optional=False):
    if optional and value is None:
        return None
    _need(type(value) is bool, "invalid_" + label)
    return value


def invocation_id_sha256(value):
    """Hash an exact declared invocation ID; do not normalize or alias it."""
    return hashlib.sha256(_identity(value, "invocation_id").encode("utf-8")).hexdigest()


def _normalized(schema, counts):
    result = dict.fromkeys(NORMALIZED_COUNTERS)
    if schema is not None and _coherence(schema, counts) is False:
        return result
    for key in ("cached_input_tokens", "cache_write_input_tokens", "output_tokens",
                "reasoning_output_tokens", "total_tokens"):
        result[key] = counts[key]
    if schema == "native-codex-cumulative-usage@1":
        result["input_tokens_including_cache"] = counts["input_tokens"]
        # Native Codex establishes cached input as a subset, but this module
        # does not assert that its optional cache-write counter is disjoint.
        # Fresh input therefore remains unknown rather than inventing a split.
    elif schema == "native-grok-final-usage@1":
        result["fresh_input_tokens"] = counts["input_tokens"]
        parts = [counts[k] for k in ("input_tokens", "cached_input_tokens", "cache_write_input_tokens")]
        if all(v is not None for v in parts):
            result["input_tokens_including_cache"] = _sum(parts)
    return result


def _sum(values):
    total = sum(values)
    _need(total <= MAX_COUNTER, "aggregate_counter_bound")
    return total


def _coherence(schema, counts):
    """Return False for contradictions, None for incomplete native categories."""
    inputs, outputs, total = (counts[k] for k in ("input_tokens", "output_tokens", "total_tokens"))
    reason = counts["reasoning_output_tokens"]
    if reason is not None and outputs is not None and reason > outputs:
        return False
    if schema == "native-codex-cumulative-usage@1":
        cached = counts["cached_input_tokens"]
        if inputs is not None and cached is not None and cached > inputs:
            return False
        main = [inputs, outputs, total]
        return None if any(v is None for v in main) else inputs + outputs == total
    parts = [counts[k] for k in ("input_tokens", "cached_input_tokens", "cache_write_input_tokens", "output_tokens")]
    return None if total is None or any(v is None for v in parts) else sum(parts) == total


def _native_usage(native, *, provider):
    empty = dict(schema=None, status="unavailable", scope=None, identity_kind=None,
        session_id_sha256=None, artifact_sha256=None, observed_records=None, complete=None,
        counts=dict.fromkeys(COUNTERS), normalized_counts=dict.fromkeys(NORMALIZED_COUNTERS),
        reasons=["native_usage_unavailable"])
    if native is None:
        return empty
    _need(type(native) is dict, "invalid_native_usage")
    schema = native.get("schema")
    _need(schema is None or type(schema) is str, "invalid_native_schema")
    if schema not in {"native-codex-cumulative-usage@1", "native-grok-final-usage@1"}:
        return {**empty, "status": "unsupported", "reasons": ["native_usage_schema_unsupported"]}
    codex = schema == "native-codex-cumulative-usage@1"
    expected_provider = "codex_cli" if codex else "grok_cli"
    reasons = []
    if provider != expected_provider:
        reasons.append("native_provider_mismatch")
    cache = _boolean(native.get("cache_included_in_input"), "cache_semantics", optional=True)
    if cache is not codex:
        reasons.append("native_cache_semantics_unavailable_or_mismatched")
    if "billing_total_verified" in native:
        _need(native["billing_total_verified"] is False, "native_billing_authority_not_supported")
    available = _boolean(native.get("usage_available"), "usage_available", optional=True)
    task_complete = _boolean(native.get("task_complete_observed"), "task_complete", optional=True)
    usage = native.get("usage", {})
    _need(type(usage) is dict and set(usage).issubset(COUNTERS), "invalid_native_counter_fields")
    counts = {key: _count(usage.get(key), key, optional=True) for key in COUNTERS}
    if available is not True:
        reasons.append("native_usage_availability_unconfirmed")
    if available is False and any(v is not None for v in counts.values()):
        reasons.append("native_availability_counter_conflict")
    coherent = _coherence(schema, counts)
    if coherent is False:
        reasons.append("native_counter_incoherence")
    elif coherent is None:
        reasons.append("native_counter_population_incomplete")
    if codex:
        identity = native.get("thread_id")
        _need(identity is None or type(identity) is str and re.fullmatch(r"[A-Za-z0-9-]{16,80}", identity),
              "invalid_native_session_id")
        session = hashlib.sha256(identity.encode()).hexdigest() if identity is not None else None
        artifact = _sha(native.get("rollout_sha256"), "rollout_sha256", optional=True)
        records = _count(native.get("observed_token_count_records"), "observed_records", optional=True,
                         maximum=MAX_RECORDS)
        scope = "native_session_cumulative"
        if session is None or artifact is None or records in (None, 0):
            reasons.append("native_session_custody_incomplete")
        latest = _boolean(native.get("latest_usage_snapshot_valid"), "latest_snapshot", optional=True)
        regression = _boolean(native.get("counter_regression_observed"), "counter_regression", optional=True)
        if latest is False or regression is True:
            reasons.append("native_snapshot_invalid_or_regressed")
        usage_complete = task_complete
    else:
        session = None
        artifact = _sha(native.get("usage_observation_sha256"), "usage_sha256", optional=True)
        records = _count(native.get("observed_final_usage_records"), "observed_records", optional=True, maximum=1)
        scope = "native_final_envelope"
        if native.get("totals_scope") != scope or artifact is None or records != 1:
            reasons.append("native_final_envelope_custody_incomplete")
        supplied_complete = _boolean(native.get("usage_complete_observed"), "usage_complete", optional=True)
        usage_complete = (False if task_complete is False or supplied_complete is False else
                          True if task_complete is True and supplied_complete is True else None)
    invalid = any(r in reasons for r in ("native_provider_mismatch", "native_cache_semantics_unavailable_or_mismatched",
        "native_availability_counter_conflict", "native_counter_incoherence", "native_snapshot_invalid_or_regressed"))
    complete = False if invalid else (usage_complete if not reasons else False if usage_complete is False else None)
    if not invalid and usage_complete is None:
        reasons.append("native_usage_completeness_unknown")
    elif not invalid and usage_complete is False:
        reasons.append("native_usage_incomplete")
    status = ("invalid" if invalid else "complete" if complete is True else "incomplete" if complete is False
              else "partial" if coherent is None else "unknown_complete_usage")
    return dict(schema=schema, status=status, scope=scope,
        identity_kind="native_session" if codex and session is not None else "invocation_final_envelope" if not codex else None,
        session_id_sha256=session, artifact_sha256=artifact, observed_records=records, complete=complete,
        counts=counts, normalized_counts=_normalized(schema, counts), reasons=sorted(set(reasons)))


def _delivery(receipt, native):
    top = receipt.get("native_context_delivery")
    nested = native.get("native_context_delivery") if type(native) is dict else None
    if top is not None and nested is not None:
        _need(_wire(top, MAX_OBSERVATION_BYTES) == _wire(nested, MAX_OBSERVATION_BYTES), "delivery_ledger_alias_conflict")
    ledger = top if top is not None else nested
    if ledger is None:
        return _unavailable_delivery()
    from .native_context_delivery import validate_native_context_delivery
    validated = validate_native_context_delivery(ledger)
    telemetry = native.get("interaction_telemetry") if type(native) is dict else None
    matched = (type(telemetry) is dict and _digest(telemetry, MAX_INPUT_BYTES)
               == validated["interaction_telemetry_sha256"])
    if validated.get("status") == "unavailable":
        return {**_unavailable_delivery(), "ledger_schema": validated["schema"],
            "ledger_sha256": _digest(validated),
            "interaction_telemetry_sha256": validated["interaction_telemetry_sha256"],
            "telemetry_binding_verified": matched}
    return dict(status="observed", ledger_schema=validated["schema"],
        ledger_sha256=_digest(validated), source_ledger_sha256=validated["ledger_sha256"],
        interaction_telemetry_sha256=validated["interaction_telemetry_sha256"],
        telemetry_binding_verified=matched,
        counts={key: validated[key] for key in _DELIVERY_COUNTS},
        payload_measurements=json.loads(_wire(validated["payload_measurements"], MAX_OBSERVATION_BYTES)),
        coverage={key: validated[key] for key in _DELIVERY_COVERAGE})


def _unavailable_delivery():
    return dict(status="unavailable", ledger_schema=None, ledger_sha256=None,
        source_ledger_sha256=None, interaction_telemetry_sha256=None,
        telemetry_binding_verified=False, counts=None, payload_measurements=None, coverage=None)


def observe_invocation(receipt):
    """Project supported fields only; the input pin covers that projection.

    Unknown receipt fields are not copied, hashed, published or interpreted.
    Generic router ``usage`` and local tokenizer counts cannot replace native
    usage. Existing provider status is preserved independently of completeness.
    """
    _need(type(receipt) is dict, "invalid_invocation_receipt")
    selected = {k: v for k, v in receipt.items() if k in OBSERVE_INPUT_FIELDS}
    pin = _digest(selected, MAX_INPUT_BYTES)
    _identity(receipt.get("schema"), "receipt_schema", optional=True)
    invocation = _identity(receipt.get("invocation_id"), "invocation_id")
    provider = _identity(receipt.get("provider"), "provider")
    model = _identity(receipt.get("model"), "model")
    purpose = _identity(receipt.get("purpose", receipt.get("phase")), "purpose", optional=True)
    phase = _identity(receipt.get("phase"), "phase", optional=True)
    status = _identity(receipt.get("status"), "status", optional=True)
    error = _identity(receipt.get("error_type"), "error_type", optional=True)
    prompt_sha = _sha(receipt.get("model_prompt_sha256"), "model_prompt_sha256", optional=True)
    prompt_bytes = _count(receipt.get("model_prompt_bytes"), "model_prompt_bytes", optional=True, maximum=MAX_PROMPT_BYTES)
    _need((prompt_sha is None) == (prompt_bytes is None), "incomplete_model_prompt_binding")
    workspace = receipt.get("workspace")
    _need(workspace is None or type(workspace) is str and 0 < len(workspace) <= 4096,
          "invalid_workspace_binding")
    native = receipt.get("native_rollout_usage")
    result = dict(schema=INVOCATION_SCHEMA, invocation_id=invocation,
        invocation_id_sha256=invocation_id_sha256(invocation), provider=provider, model=model,
        purpose=purpose, phase=phase, status=status, error_type=error, receipt_sha256=pin,
        workspace_binding_sha256=hashlib.sha256(workspace.encode()).hexdigest() if workspace is not None else None,
        model_prompt=dict(sha256=prompt_sha, bytes=prompt_bytes, count_scope="rendered_user_prompt_only"),
        native_usage=_native_usage(native, provider=provider), native_context_delivery=_delivery(receipt, native),
        **_AUTHORITY)
    result["observation_sha256"] = _digest(result)
    return result


def _validate_observation(row):
    _need(type(row) is dict and set(row) == _ROW_FIELDS and row["schema"] == INVOCATION_SCHEMA,
          "invalid_invocation_observation_fields")
    _wire(row, MAX_OBSERVATION_BYTES)
    for key, value in _AUTHORITY.items():
        _need(type(row[key]) is bool and row[key] is value, "observation_authority_not_supported")
    _need(row["invocation_id_sha256"] == invocation_id_sha256(row["invocation_id"]), "invocation_hash_mismatch")
    for key in ("provider", "model"):
        _identity(row[key], key)
    for key in ("purpose", "phase", "status", "error_type"):
        _identity(row[key], key, optional=True)
    for key in ("receipt_sha256", "observation_sha256"):
        _sha(row[key], key)
    _sha(row["workspace_binding_sha256"], "workspace_sha256", optional=True)
    prompt = row["model_prompt"]
    _need(type(prompt) is dict and set(prompt) == {"sha256", "bytes", "count_scope"}
          and prompt["count_scope"] == "rendered_user_prompt_only", "invalid_prompt_observation")
    _sha(prompt["sha256"], "model_prompt_sha256", optional=True)
    _count(prompt["bytes"], "model_prompt_bytes", optional=True, maximum=MAX_PROMPT_BYTES)
    _need((prompt["sha256"] is None) == (prompt["bytes"] is None), "incomplete_model_prompt_binding")
    native = row["native_usage"]
    _need(type(native) is dict and set(native) == _NATIVE_FIELDS, "invalid_native_projection_fields")
    _need((native["schema"] is None or type(native["schema"]) is str)
          and native["schema"] in {None, "native-codex-cumulative-usage@1", "native-grok-final-usage@1"}
          and type(native["status"]) is str
          and native["status"] in {"unavailable", "unsupported", "invalid", "complete", "incomplete", "partial", "unknown_complete_usage"},
          "invalid_native_projection_kind")
    _need(type(native["counts"]) is dict and set(native["counts"]) == set(COUNTERS)
          and type(native["normalized_counts"]) is dict and set(native["normalized_counts"]) == set(NORMALIZED_COUNTERS),
          "invalid_native_projection_counters")
    for key, value in native["counts"].items():
        _count(value, key, optional=True)
    for key, value in native["normalized_counts"].items():
        _count(value, key, optional=True)
    _need(native["normalized_counts"] == _normalized(native["schema"], native["counts"]), "normalized_counter_mismatch")
    _boolean(native["complete"], "native_complete", optional=True)
    _sha(native["session_id_sha256"], "native_session_hash", optional=True)
    _sha(native["artifact_sha256"], "native_artifact_hash", optional=True)
    _count(native["observed_records"], "observed_records", optional=True, maximum=MAX_RECORDS)
    _need(type(native["reasons"]) is list and len(native["reasons"]) <= 16
          and all(type(v) is str and re.fullmatch(r"[a-z_]{1,80}", v) for v in native["reasons"]), "invalid_native_reasons")
    _need(native["reasons"] == sorted(set(native["reasons"])), "invalid_native_reasons")
    codex = native["schema"] == "native-codex-cumulative-usage@1"
    grok = native["schema"] == "native-grok-final-usage@1"
    expected_scope = "native_session_cumulative" if codex else "native_final_envelope" if grok else None
    expected_kind = "native_session" if codex and native["session_id_sha256"] is not None else "invocation_final_envelope" if grok else None
    _need(native["scope"] == expected_scope and native["identity_kind"] == expected_kind,
          "invalid_native_identity_scope")
    if native["complete"] is True:
        _need(native["status"] == "complete" and not native["reasons"]
              and _coherence(native["schema"], native["counts"]) is True
              and native["artifact_sha256"] is not None and native["observed_records"] not in (None, 0)
              and row["provider"] == ("codex_cli" if codex else "grok_cli")
              and (not codex or native["session_id_sha256"] is not None), "invalid_complete_native_projection")
    else:
        _need(native["status"] != "complete", "invalid_complete_native_projection")
    if not codex and not grok:
        _need(native["complete"] is None and native["scope"] is None
              and all(v is None for v in native["counts"].values()), "unsupported_native_counts")
    delivery = row["native_context_delivery"]
    _need(type(delivery) is dict and set(delivery) == _DELIVERY_FIELDS, "invalid_delivery_projection")
    if delivery["status"] == "unavailable":
        if delivery["ledger_schema"] is None:
            _need(delivery == _unavailable_delivery(), "invalid_unavailable_delivery")
        else:
            _need(delivery["ledger_schema"] == "native-codex-context-delivery-unavailable@1"
                  and delivery["source_ledger_sha256"] is None and delivery["counts"] is None
                  and delivery["payload_measurements"] is None and delivery["coverage"] is None,
                  "invalid_unavailable_delivery")
            _sha(delivery["ledger_sha256"], "unavailable_ledger_hash")
            _sha(delivery["interaction_telemetry_sha256"], "unavailable_telemetry_hash", optional=True)
            _boolean(delivery["telemetry_binding_verified"], "telemetry_binding_verified")
    else:
        _need(delivery["status"] == "observed" and delivery["ledger_schema"] == "native-codex-context-delivery@1",
              "invalid_delivery_status")
        for key in ("ledger_sha256", "source_ledger_sha256", "interaction_telemetry_sha256"):
            _sha(delivery[key], key)
        _boolean(delivery["telemetry_binding_verified"], "telemetry_binding_verified")
        counts, coverage, measurements = (delivery[k] for k in ("counts", "coverage", "payload_measurements"))
        _need(type(counts) is dict and set(counts) == set(_DELIVERY_COUNTS)
              and type(coverage) is dict and set(coverage) == set(_DELIVERY_COVERAGE), "invalid_delivery_counts")
        for key, value in counts.items():
            _count(value, key, maximum=32_000_000 if key == "measured_payload_bytes" else 200_000)
        for key, value in coverage.items():
            _boolean(value, key)
        _need(counts["observed_tool_events"] == counts["observed_tool_call_events"] + counts["observed_tool_output_events"]
              and counts["retained_tool_events"] + counts["omitted_tool_events"] == counts["observed_tool_events"]
              and counts["retained_tool_events"] <= 256
              and counts["payloads_without_fingerprint"] <= counts["observed_tool_events"]
              and coverage["ledger_truncated"] is (counts["omitted_tool_events"] > 0)
              and (not coverage["observation_census_complete"] or not coverage["ledger_truncated"])
              and coverage["fingerprints_complete"] is (coverage["observation_census_complete"]
                                                        and counts["payloads_without_fingerprint"] == 0),
              "delivery_counter_population_mismatch")
        _need(type(measurements) is dict and set(measurements) == {"decoded_utf8_string", "canonical_json_utf8", "unavailable"},
              "invalid_delivery_measurements")
        for scope, value in measurements.items():
            _need(type(value) is dict and set(value) == {"call_events", "output_events", "bytes"}, "invalid_delivery_measurements")
            for key in ("call_events", "output_events"):
                _count(value[key], key)
            _count(value["bytes"], "payload_bytes", optional=scope == "unavailable", maximum=32_000_000)
            _need(scope != "unavailable" or value["bytes"] is None, "invalid_unknown_payload_bytes")
        _need(sum(v["call_events"] for v in measurements.values()) == counts["observed_tool_call_events"]
              and sum(v["output_events"] for v in measurements.values()) == counts["observed_tool_output_events"]
              and sum(v["bytes"] for v in measurements.values() if v["bytes"] is not None) == counts["measured_payload_bytes"],
              "delivery_measurement_population_mismatch")
    body = {k: v for k, v in row.items() if k != "observation_sha256"}
    _need(row["observation_sha256"] == _digest(body), "observation_hash_mismatch")


def summarize_invocations(rows, unreceipted_provider_attempt=None):
    """Reconcile the ordered supplied population without claiming all calls.

    Codex snapshots require an exact provider/session identity; monotonic record
    counts/counters select the latest observation, never the sum of snapshots.
    Equal record counts with differing artifact pins are conservatively a
    conflict. Grok final envelopes have invocation identity, not an invented
    native session ID. Unknown missing-attempt markers withhold complete totals.
    """
    _need(type(rows) is list and len(rows) <= MAX_INVOCATIONS, "invocation_population_bound")
    _boolean(unreceipted_provider_attempt, "unreceipted_provider_attempt", optional=True)
    total_bytes = 0
    for row in rows:
        raw = _wire(row, MAX_OBSERVATION_BYTES)
        total_bytes += len(raw)
        _need(total_bytes <= MAX_AGGREGATE_OBSERVATION_BYTES, "aggregate_observation_bound")
        _validate_observation(row)
    # Serialization above precedes the detached result snapshot.
    observations = [json.loads(_wire(row, MAX_OBSERVATION_BYTES)) for row in rows]
    invocation_rows = {}
    for index, row in enumerate(observations):
        invocation_rows.setdefault(row["invocation_id"], []).append(index)
    conflicting_invocations = {identity for identity, indices in invocation_rows.items()
        if len({observations[i]["observation_sha256"] for i in indices}) > 1}
    groups = {}
    for index, row in enumerate(observations):
        native = row["native_usage"]
        identity = native["session_id_sha256"] if native["identity_kind"] == "native_session" else row["invocation_id_sha256"]
        kind = "native_session" if native["identity_kind"] == "native_session" else "invocation"
        groups.setdefault((row["provider"], kind, identity), []).append(index)
    sessions = []
    for (provider, kind, identity), indices in groups.items():
        unique = []
        seen = set()
        for index in indices:
            digest = observations[index]["observation_sha256"]
            if digest not in seen:
                unique.append(index)
                seen.add(digest)
        reasons = []
        selected = unique[0]
        first = observations[selected]
        if any(observations[i]["invocation_id"] in conflicting_invocations for i in indices):
            reasons.append("conflicting_invocation_observations")
        if len({observations[i]["model"] for i in unique}) != 1:
            reasons.append("native_session_model_conflict")
        workspaces = {observations[i]["workspace_binding_sha256"] for i in unique}
        if len(workspaces) > 1:
            reasons.append("native_session_workspace_conflict")
        if kind == "native_session" and len(unique) > 1:
            previous = first["native_usage"]
            for index in unique[1:]:
                current = observations[index]["native_usage"]
                prior_count, current_count = previous["observed_records"], current["observed_records"]
                if prior_count is None or current_count is None:
                    reasons.append("native_snapshot_order_unavailable")
                elif current_count < prior_count:
                    reasons.append("native_snapshot_record_regression")
                elif current_count == prior_count and (current["counts"] != previous["counts"]
                        or current["artifact_sha256"] != previous["artifact_sha256"]):
                    reasons.append("conflicting_equal_record_snapshots")
                elif current_count > prior_count and current["artifact_sha256"] == previous["artifact_sha256"]:
                    reasons.append("native_artifact_counter_conflict")
                for key in COUNTERS:
                    a, b = previous["counts"][key], current["counts"][key]
                    if a is not None and b is not None and b < a:
                        reasons.append("native_snapshot_counter_regression")
                previous = current
                selected = index
        chosen = observations[selected]["native_usage"]
        if any(observations[i]["native_usage"]["status"] == "invalid" for i in unique):
            reasons.append("native_usage_observation_invalid")
        if chosen["status"] == "invalid":
            reasons.append("selected_native_usage_invalid")
        conflict = bool(reasons)
        counts = dict.fromkeys(NORMALIZED_COUNTERS) if conflict else chosen["normalized_counts"]
        coherent = _coherence(chosen["schema"], chosen["counts"]) if chosen["schema"] else None
        if coherent is not True:
            counts = dict.fromkeys(NORMALIZED_COUNTERS)
        complete = False if conflict else chosen["complete"]
        sessions.append(dict(provider=provider, model=first["model"], identity_kind=kind,
            identity_sha256=identity, observation_indices=indices,
            selected_observation_index=None if conflict else selected,
            status="conflict" if conflict else chosen["status"], reasons=sorted(set(reasons)),
            complete=complete, reported_counts=counts,
            complete_counts=counts if complete is True else dict.fromkeys(NORMALIZED_COUNTERS)))
    subtotals = {}
    for key in NORMALIZED_COUNTERS:
        present = [s["reported_counts"][key] for s in sessions if s["reported_counts"][key] is not None]
        subtotals[key] = _sum(present) if present else None
    native_complete = bool(sessions) and all(s["complete"] is True for s in sessions)
    complete = native_complete and unreceipted_provider_attempt is False
    complete_counts = {key: _sum([s["complete_counts"][key] for s in sessions])
        if complete and all(s["complete_counts"][key] is not None for s in sessions) else None
        for key in NORMALIZED_COUNTERS}
    result = dict(schema=SUMMARY_SCHEMA, observations=observations, sessions=sessions,
        supplied_observation_count=len(rows), supplied_invocation_count=len(invocation_rows),
        duplicate_observation_count=len(rows) - len({r["observation_sha256"] for r in rows}),
        conflicting_invocation_count=len(conflicting_invocations),
        supplied_native_sessions_complete=native_complete,
        unreceipted_provider_attempt=unreceipted_provider_attempt,
        complete_supplied_native_usage=complete, campaign_call_population_complete=None,
        input_observation_authenticity_verified=False,
        known_native_counter_subtotals=subtotals,
        reported_native_token_subtotal=subtotals["total_tokens"],
        complete_native_counts=complete_counts, complete_native_total_tokens=complete_counts["total_tokens"],
        native_session_count=sum(s["identity_kind"] == "native_session" for s in sessions),
        invocation_envelope_count=sum(s["identity_kind"] == "invocation" for s in sessions),
        failed_invocation_count=sum(any(observations[i]["status"] == "failed" for i in indices)
                                    for indices in invocation_rows.values()),
        measurement_scope="supplied native sessions or invocation envelopes; each selected once",
        delivery_bytes_added_to_native_tokens=False, cumulative_snapshots_are_additive=False,
        local_tokenizer_counts_used_as_native_usage=False, **_AUTHORITY)
    # Fixed output bounds include both the complete observation population and indexes.
    result["summary_sha256"] = _digest(result, MAX_AGGREGATE_OBSERVATION_BYTES + 1024 * 1024)
    return result
