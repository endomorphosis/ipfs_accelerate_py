"""Bounded fingerprints of already recorded Codex tool events.

The existing interaction telemetry owns pairing, ambiguity and completion.
This overlay observes retained rollout content, not transport delivery, model
consumption, a source identity, a provider request or token attribution.
"""
from __future__ import annotations

import hashlib
import json

from . import codex_interaction_telemetry as interaction

SCHEMA = "native-codex-context-delivery@1"
UNAVAILABLE_SCHEMA = "native-codex-context-delivery-unavailable@1"
MAX_LEDGER_EVENTS = 256
MAX_LEDGER_BYTES = 128 * 1024
MAX_MEASURED_PAYLOAD_BYTES = 32_000_000
MAX_TELEMETRY_BYTES = 4_000_000
MAX_TELEMETRY_NODES = 40_000
_KINDS = {"function_call": ("call", "function"), "custom_tool_call": ("call", "custom"),
          "function_call_output": ("output", "function"), "custom_tool_call_output": ("output", "custom")}
_PAIR_STATUSES = frozenset({"ambiguous_call_identity", "output_without_observed_call",
    "call_without_observed_output", "multiple_output_chunks_observed", "output_precedes_call",
    "single_call_and_output_observed"})
_SCOPES = frozenset({"decoded_utf8_string", "canonical_json_utf8", None})
_REASONS = frozenset({None, "payload_limit", "invalid_unicode", "unsupported_payload", "structure_limit",
    "unsupported_structure", "nonfinite_number", "aggregate_payload_limit"})
_ID_REASONS = frozenset({None, "missing_or_invalid_identity", "invalid_unicode"})
_NAME_REASONS = frozenset({None, "missing_or_invalid_name", "name_limit", "invalid_unicode", "missing_call"})
_FALSE_FIELDS = frozenset({"raw_content_exported", "owner_response_match_verified", "source_identity_verified",
    "provider_ingestion_verified", "model_consumption_verified", "tool_execution_verified",
    "provider_request_count_inferred", "causal_tool_token_attribution", "native_aggregate_usage_modified",
    "usage_completeness_established", "proof_authority", "execution_authority", "completion_authority"})
_ROW_FIELDS = frozenset({"record_ordinal", "tool_event_ordinal", "event_kind", "tool_kind", "call_id_sha256",
    "call_id_unknown_reason", "tool_name", "tool_name_sha256", "tool_name_unknown_reason",
    "interaction_observation_ordinal", "pairing_status", "preceding_token_observation_ordinal",
    "payload_sha256", "payload_bytes", "payload_byte_scope", "payload_unknown_reason"})
_COUNTER_FIELDS = frozenset({"observed_records", "observed_tool_call_events", "observed_tool_output_events",
    "observed_tool_events", "retained_tool_events", "omitted_tool_events", "payloads_without_fingerprint",
    "measured_payload_bytes", "retained_rows_canonical_bytes"})
_FIELDS = frozenset({"schema", "interaction_telemetry_sha256", "records", "limits", "ledger_truncated",
    "observation_census_complete", "tool_observations_complete", "fingerprints_complete", "payload_measurements",
    "malformed_json_records", "record_limit_exceeded", "byte_scope", "token_position_scope",
    "completeness_scope", "ledger_sha256", *_COUNTER_FIELDS, *_FALSE_FIELDS})
_BYTE_SCOPE = "Recorded decoded UTF-8 strings or canonical structured JSON; neither transport frames nor provider bytes/tokens."
_TOKEN_SCOPE = "Preceding retained interaction checkpoint ordinal only; not a provider request or per-tool token assignment."
_COMPLETE_SCOPE = "Bounded already-selected rollout observation census only; hashes and seals do not authenticate source or delivery."


def _need(ok, reason):
    if not ok:
        raise ValueError(reason)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _integer(value, maximum=interaction.MAX_RECORDS):
    return type(value) is int and 0 <= value <= maximum


def _identity(value):
    if type(value) is not str or not value or len(value) > 256:
        return None, "missing_or_invalid_identity"
    try:
        return _sha(value.encode("utf-8")), None
    except UnicodeError:
        return None, "invalid_unicode"


def _name(value):
    if type(value) is not str or not value:
        return None, None, "missing_or_invalid_name"
    classification = value if value in interaction._KNOWN_TOOLS else "other"
    if len(value) > 256:
        return classification, None, "name_limit"
    try:
        raw = value.encode("utf-8")
    except UnicodeError:
        return classification, None, "invalid_unicode"
    return classification, _sha(raw), None


def _telemetry(telemetry, records):
    _need(type(telemetry) is dict and telemetry.get("schema") == interaction.SCHEMA,
          "Existing interaction telemetry required")
    # A full telemetry packet contains up to 128 pairs AND 256 checkpoints;
    # the individual tool-payload 10,000-node budget is not its population cap.
    pending = [(telemetry, 0)]; nodes = characters = 0
    while pending:
        value, depth = pending.pop(); nodes += 1
        _need(nodes <= MAX_TELEMETRY_NODES and depth <= 16, "Interaction telemetry exceeds bounded structure")
        if type(value) is str:
            characters += len(value)
        elif type(value) is dict:
            _need(len(value) + len(pending) <= MAX_TELEMETRY_NODES and all(type(k) is str for k in value),
                  "Interaction telemetry exceeds bounded structure")
            characters += sum(len(k) for k in value)
            pending.extend((child, depth + 1) for child in value.values())
        elif type(value) is list:
            _need(len(value) + len(pending) <= MAX_TELEMETRY_NODES, "Interaction telemetry exceeds bounded structure")
            pending.extend((child, depth + 1) for child in value)
        else:
            _need(value is None or type(value) in (bool, int, float), "Invalid interaction telemetry structure")
            _need(type(value) is not int or value.bit_length() <= 64, "Invalid interaction telemetry integer")
        _need(characters <= MAX_TELEMETRY_BYTES, "Interaction telemetry exceeds byte bound")
    _need(len(_wire(telemetry)) <= MAX_TELEMETRY_BYTES, "Interaction telemetry exceeds byte bound")
    _need(_integer(telemetry.get("observed_records")) and telemetry["observed_records"] == min(len(records), interaction.MAX_RECORDS),
          "Telemetry record census differs from supplied records")
    _need(type(telemetry.get("record_limit_exceeded")) is bool
          and telemetry["record_limit_exceeded"] is (len(records) > interaction.MAX_RECORDS),
          "Telemetry record limit differs")
    _need(type(telemetry.get("tool_observations_complete")) is bool
          and type(telemetry.get("task_complete_observed")) is bool
          and _integer(telemetry.get("invalid_record_shapes"))
          and _integer(telemetry.get("malformed_json_records"), interaction.MAX_COUNTER), "Invalid telemetry completeness metadata")
    pairs, tokens = telemetry.get("tool_pairs"), telemetry.get("token_observations")
    _need(type(pairs) is list and len(pairs) <= interaction.MAX_TOOL_PAIRS
          and type(tokens) is list and len(tokens) <= interaction.MAX_TOKEN_OBSERVATIONS,
          "Bounded existing telemetry populations required")
    lookup = {}
    for index, row in enumerate(pairs, 1):
        _need(type(row) is dict and row.get("observation_ordinal") == index
              and type(row.get("observation_ordinal")) is int
              and type(row.get("pairing_status")) is str and row["pairing_status"] in _PAIR_STATUSES
              and _integer(row.get("session_scope_ordinal")) and row["session_scope_ordinal"] > 0
              and _integer(row.get("first_record_ordinal")) and 0 < row["first_record_ordinal"] <= len(records)
              and row.get("tool_name") in (None, "other", *interaction._KNOWN_TOOLS),
              "Invalid existing tool-pair observation")
        first = records[row["first_record_ordinal"] - 1]
        payload = first.get("payload") if type(first) is dict else None
        _need(type(payload) is dict and first.get("type") == "response_item"
              and type(payload.get("type")) is str and payload["type"] in _KINDS,
              "Existing pair ordinal differs from recorded tool event")
        call_hash, _ = _identity(payload.get("call_id"))
        if call_hash is not None:
            key = (row["session_scope_ordinal"], call_hash)
            _need(key not in lookup, "Duplicate existing pair identity")
            lookup[key] = row
    previous = 0
    for index, row in enumerate(tokens, 1):
        _need(type(row) is dict and type(row.get("observation_ordinal")) is int and row["observation_ordinal"] == index
              and _integer(row.get("record_ordinal")) and previous < row["record_ordinal"] <= len(records),
              "Invalid retained token observation position")
        previous = row["record_ordinal"]
    return lookup, tokens


def build_native_context_delivery(records, *, telemetry):
    """Add exact fingerprints while preserving the existing pairing decisions.

    Records and telemetry must refer to the same independently selected rollout.
    No raw content, paths, call IDs or unrecognized tool names leave this API.
    """
    _need(type(records) in (list, tuple), "Already-read bounded record sequence required")
    lookup, tokens = _telemetry(telemetry, records)
    rows = []; retained_bytes = 2; events = calls = outputs = missing = measured = 0
    measurements = {scope: {"call_events": 0, "output_events": 0, "bytes": 0}
                    for scope in ("decoded_utf8_string", "canonical_json_utf8")}
    measurements["unavailable"] = {"call_events": 0, "output_events": 0, "bytes": None}
    token_index = 0; token_position = None; scope = 1; seen_meta = False; hash_budget_exhausted = False
    for ordinal, record in enumerate(records, 1):
        if ordinal > interaction.MAX_RECORDS:
            break
        while token_index < len(tokens) and tokens[token_index]["record_ordinal"] < ordinal:
            token_position = tokens[token_index]["observation_ordinal"]; token_index += 1
        if type(record) is not dict:
            continue
        if record.get("type") == "session_meta":
            if seen_meta or ordinal > 1:
                scope += 1
            seen_meta = True
            continue
        payload = record.get("payload")
        if (record.get("type") != "response_item" or type(payload) is not dict
                or type(payload.get("type")) is not str or payload["type"] not in _KINDS):
            continue
        event, tool_kind = _KINDS[payload["type"]]
        events += 1; calls += event == "call"; outputs += event == "output"
        call_hash, id_reason = _identity(payload.get("call_id"))
        pair = lookup.get((scope, call_hash)) if call_hash is not None else None
        if pair is not None:
            _need(pair["first_record_ordinal"] <= ordinal, "Pair identity appears before its existing observation")
        value = payload.get("output" if event == "output" else "arguments" if tool_kind == "function" else "input")
        if hash_budget_exhausted:
            size = byte_scope = None; unknown = "aggregate_payload_limit"
        else:
            size, byte_scope, unknown = interaction._payload_size(value)
            if size is not None and measured + size > MAX_MEASURED_PAYLOAD_BYTES:
                size = None; unknown = "aggregate_payload_limit"; hash_budget_exhausted = True
        payload_hash = None
        if size is not None:
            encoded = value.encode("utf-8") if byte_scope == "decoded_utf8_string" else _wire(value)
            _need(len(encoded) == size, "Existing payload measurement differs from fingerprint scope")
            payload_hash = _sha(encoded); measured += size
        else:
            missing += 1
        count_key = event + "_events"
        measurements[byte_scope if size is not None else "unavailable"][count_key] += 1
        if size is not None:
            measurements[byte_scope]["bytes"] += size
        if event == "call":
            name, name_hash, name_reason = _name(payload.get("name"))
        elif pair is not None and pair.get("call_record_ordinal") is not None:
            call_ordinal = pair["call_record_ordinal"]
            _need(_integer(call_ordinal) and 0 < call_ordinal <= len(records), "Existing call ordinal unavailable")
            call_record = records[call_ordinal - 1]
            named = call_record.get("payload") if type(call_record) is dict else None
            _need(type(named) is dict, "Existing call ordinal differs from recorded call")
            name, name_hash, name_reason = _name(named.get("name"))
        else:
            name = name_hash = None; name_reason = "missing_call"
        row = {"record_ordinal": ordinal, "tool_event_ordinal": events, "event_kind": payload["type"],
            "tool_kind": tool_kind, "call_id_sha256": call_hash, "call_id_unknown_reason": id_reason,
            "tool_name": name, "tool_name_sha256": name_hash, "tool_name_unknown_reason": name_reason,
            "interaction_observation_ordinal": None if pair is None else pair["observation_ordinal"],
            "pairing_status": None if pair is None else pair["pairing_status"],
            "preceding_token_observation_ordinal": token_position, "payload_sha256": payload_hash,
            "payload_bytes": size, "payload_byte_scope": byte_scope, "payload_unknown_reason": unknown}
        size = len(_wire(row)) + bool(rows)
        if events == len(rows) + 1 and len(rows) < MAX_LEDGER_EVENTS and retained_bytes + size <= MAX_LEDGER_BYTES:
            retained_bytes += size; rows.append(row)
    _need((calls, outputs) == (telemetry.get("observed_tool_call_events"), telemetry.get("observed_tool_output_events")),
          "Existing tool-event census differs from supplied records")
    truncated = events != len(rows)
    census = (telemetry["task_complete_observed"] and telemetry["malformed_json_records"] == 0
              and telemetry["invalid_record_shapes"] == 0 and not telemetry["record_limit_exceeded"] and not truncated)
    result = {"schema": SCHEMA, "interaction_telemetry_sha256": _sha(_wire(telemetry)),
        "observed_records": telemetry["observed_records"], "observed_tool_call_events": calls,
        "observed_tool_output_events": outputs, "observed_tool_events": events, "retained_tool_events": len(rows),
        "omitted_tool_events": events - len(rows), "payloads_without_fingerprint": missing,
        "measured_payload_bytes": measured, "retained_rows_canonical_bytes": retained_bytes,
        "records": rows, "payload_measurements": measurements, "limits": {"ledger_events": MAX_LEDGER_EVENTS,
            "ledger_bytes": MAX_LEDGER_BYTES, "measured_payload_bytes": MAX_MEASURED_PAYLOAD_BYTES},
        "ledger_truncated": truncated, "observation_census_complete": census,
        "tool_observations_complete": telemetry["tool_observations_complete"],
        "fingerprints_complete": census and missing == 0,
        "malformed_json_records": telemetry["malformed_json_records"], "record_limit_exceeded": telemetry["record_limit_exceeded"],
        "byte_scope": _BYTE_SCOPE, "token_position_scope": _TOKEN_SCOPE, "completeness_scope": _COMPLETE_SCOPE,
        **dict.fromkeys(_FALSE_FIELDS, False)}
    result["ledger_sha256"] = _sha(_wire(result))
    return validate_native_context_delivery(result)


def unavailable_native_context_delivery(telemetry=None):
    """An optional observer failure cannot replace the existing usage result."""
    try:
        telemetry_hash = _sha(_wire(telemetry)) if type(telemetry) is dict else None
    except Exception:
        telemetry_hash = None
    return {"schema": UNAVAILABLE_SCHEMA, "status": "unavailable", "reason": "optional_observer_failed",
            "interaction_telemetry_sha256": telemetry_hash, **dict.fromkeys(_FALSE_FIELDS, False)}


def validate_native_context_delivery(receipt):
    """Validate bounded closed observations; a seal grants no provenance authority."""
    if type(receipt) is dict and receipt.get("schema") == UNAVAILABLE_SCHEMA:
        _need(set(receipt) == {"schema", "status", "reason", "interaction_telemetry_sha256", *_FALSE_FIELDS}
              and receipt["status"] == "unavailable" and receipt["reason"] == "optional_observer_failed"
              and (receipt["interaction_telemetry_sha256"] is None or _digest(receipt["interaction_telemetry_sha256"]))
              and all(receipt[k] is False for k in _FALSE_FIELDS), "Closed unavailable native observation required")
        return dict(receipt)
    _need(type(receipt) is dict and set(receipt) == _FIELDS and receipt["schema"] == SCHEMA,
          "Closed native context-delivery receipt required")
    rows, limits = receipt["records"], receipt["limits"]
    _need(type(rows) is list and len(rows) <= MAX_LEDGER_EVENTS and type(limits) is dict
          and set(limits) == {"ledger_events", "ledger_bytes", "measured_payload_bytes"}
          and all(type(limits[k]) is int and 0 < limits[k] <= cap for k, cap in (
              ("ledger_events", MAX_LEDGER_EVENTS), ("ledger_bytes", MAX_LEDGER_BYTES),
              ("measured_payload_bytes", MAX_MEASURED_PAYLOAD_BYTES)))
          and len(rows) <= limits["ledger_events"], "Native ledger bounds differ")
    counter_bounds = {"measured_payload_bytes": MAX_MEASURED_PAYLOAD_BYTES,
                      "retained_rows_canonical_bytes": MAX_LEDGER_BYTES}
    _need(all(_integer(receipt[k], maximum=counter_bounds.get(k, interaction.MAX_RECORDS * 2)) for k in _COUNTER_FIELDS)
        and _integer(receipt["malformed_json_records"], interaction.MAX_COUNTER), "Invalid native ledger counters")
    for flag in ("ledger_truncated", "observation_census_complete", "tool_observations_complete", "fingerprints_complete", "record_limit_exceeded"):
        _need(type(receipt[flag]) is bool, "Exact native ledger boolean required")
    _need(all(receipt[k] is False for k in _FALSE_FIELDS)
          and receipt["byte_scope"] == _BYTE_SCOPE and receipt["token_position_scope"] == _TOKEN_SCOPE
          and receipt["completeness_scope"] == _COMPLETE_SCOPE, "Native ledger authority/scope differs")
    calls = outputs = missing = measured = 0; prior = 0
    retained_measurements = {scope: {"call_events": 0, "output_events": 0, "bytes": 0}
                             for scope in ("decoded_utf8_string", "canonical_json_utf8", "unavailable")}
    for index, row in enumerate(rows, 1):
        _need(type(row) is dict and set(row) == _ROW_FIELDS and type(row["event_kind"]) is str and row["event_kind"] in _KINDS
              and row["tool_kind"] == _KINDS[row["event_kind"]][1]
              and type(row["tool_event_ordinal"]) is int and row["tool_event_ordinal"] == index
              and _integer(row["record_ordinal"]) and prior < row["record_ordinal"] <= receipt["observed_records"],
              "Closed ordered native ledger event required")
        prior = row["record_ordinal"]
        for key in ("call_id_sha256", "tool_name_sha256", "payload_sha256"):
            _need(row[key] is None or _digest(row[key]), "Invalid native fingerprint")
        _need((row["call_id_unknown_reason"] is None or type(row["call_id_unknown_reason"]) is str and row["call_id_unknown_reason"] in _ID_REASONS)
              and (row["tool_name_unknown_reason"] is None or type(row["tool_name_unknown_reason"]) is str and row["tool_name_unknown_reason"] in _NAME_REASONS)
              and (row["payload_unknown_reason"] is None or type(row["payload_unknown_reason"]) is str and row["payload_unknown_reason"] in _REASONS)
              and row["tool_name"] in (None, "other", *interaction._KNOWN_TOOLS)
              and (row["payload_byte_scope"] is None or type(row["payload_byte_scope"]) is str and row["payload_byte_scope"] in _SCOPES),
              "Unsupported native observation tag")
        _need((row["call_id_sha256"] is None) is (row["call_id_unknown_reason"] is not None)
              and (row["tool_name_sha256"] is None) is (row["tool_name_unknown_reason"] is not None),
              "Native identity unknown status differs")
        for key, cap in (("interaction_observation_ordinal", interaction.MAX_TOOL_PAIRS),
                         ("preceding_token_observation_ordinal", interaction.MAX_TOKEN_OBSERVATIONS)):
            _need(row[key] is None or _integer(row[key], cap) and row[key] > 0, "Invalid native positional reference")
        _need((row["interaction_observation_ordinal"] is None) is (row["pairing_status"] is None)
              and (row["pairing_status"] is None or type(row["pairing_status"]) is str and row["pairing_status"] in _PAIR_STATUSES),
              "Native pairing reference differs")
        known = row["payload_bytes"] is not None
        _need((known and _integer(row["payload_bytes"], interaction.MAX_PAYLOAD_BYTES) and row["payload_sha256"] is not None
               and row["payload_unknown_reason"] is None and row["payload_byte_scope"] is not None)
              or (not known and row["payload_sha256"] is None and row["payload_unknown_reason"] is not None),
              "Native payload scope/fingerprint differs")
        calls += _KINDS[row["event_kind"]][0] == "call"; outputs += _KINDS[row["event_kind"]][0] == "output"
        missing += not known; measured += row["payload_bytes"] if known else 0
        counts = retained_measurements[row["payload_byte_scope"] if known else "unavailable"]
        counts[_KINDS[row["event_kind"]][0] + "_events"] += 1
        counts["bytes"] += row["payload_bytes"] if known else 0
        if row["tool_name"] not in (None, "other"):
            _need(row["tool_name_sha256"] == _sha(row["tool_name"].encode()), "Known tool-name fingerprint differs")
    _need(receipt["retained_tool_events"] == len(rows)
          and receipt["observed_tool_events"] == receipt["observed_tool_call_events"] + receipt["observed_tool_output_events"]
          and receipt["omitted_tool_events"] == receipt["observed_tool_events"] - len(rows)
          and receipt["ledger_truncated"] is (receipt["omitted_tool_events"] > 0)
          and calls <= receipt["observed_tool_call_events"] and outputs <= receipt["observed_tool_output_events"]
          and missing <= receipt["payloads_without_fingerprint"] <= receipt["observed_tool_events"]
          and measured <= receipt["measured_payload_bytes"] <= limits["measured_payload_bytes"], "Native ledger denominator differs")
    measurements = receipt["payload_measurements"]
    _need(type(measurements) is dict and set(measurements) == {"decoded_utf8_string", "canonical_json_utf8", "unavailable"},
          "Closed native byte-scope counters required")
    for scope, counts in measurements.items():
        _need(type(counts) is dict and set(counts) == {"call_events", "output_events", "bytes"}
              and _integer(counts["call_events"]) and _integer(counts["output_events"])
              and (counts["bytes"] is None if scope == "unavailable" else _integer(counts["bytes"], MAX_MEASURED_PAYLOAD_BYTES)),
              "Invalid native byte-scope counters")
        retained = retained_measurements[scope]
        _need(all(retained[k] <= counts[k] for k in ("call_events", "output_events"))
              and (scope == "unavailable" or retained["bytes"] <= counts["bytes"]), "Retained native scope counters differ")
        if not receipt["ledger_truncated"]:
            _need(all(retained[k] == counts[k] for k in ("call_events", "output_events"))
                  and (scope == "unavailable" or retained["bytes"] == counts["bytes"]), "Complete native scope counters differ")
    _need(sum(c["call_events"] for c in measurements.values()) == receipt["observed_tool_call_events"]
          and sum(c["output_events"] for c in measurements.values()) == receipt["observed_tool_output_events"]
          and sum(c["bytes"] for s, c in measurements.items() if s != "unavailable") == receipt["measured_payload_bytes"]
          and sum(measurements["unavailable"][k] for k in ("call_events", "output_events")) == receipt["payloads_without_fingerprint"],
          "Native byte-scope population differs")
    if not receipt["ledger_truncated"]:
        _need((calls, outputs, missing, measured) == (receipt["observed_tool_call_events"], receipt["observed_tool_output_events"],
            receipt["payloads_without_fingerprint"], receipt["measured_payload_bytes"]), "Complete native event ledger differs")
    _need(not receipt["observation_census_complete"] or (not receipt["ledger_truncated"] and not receipt["record_limit_exceeded"]
          and receipt["malformed_json_records"] == 0), "Incomplete native census promoted")
    _need(receipt["fingerprints_complete"] is (receipt["observation_census_complete"] and receipt["payloads_without_fingerprint"] == 0),
          "Native fingerprint completeness differs")
    _need(receipt["retained_rows_canonical_bytes"] == len(_wire(rows)) <= limits["ledger_bytes"], "Native retained row bytes differ")
    _need(_digest(receipt["interaction_telemetry_sha256"]) and _digest(receipt["ledger_sha256"])
          and receipt["ledger_sha256"] == _sha(_wire({k: v for k, v in receipt.items() if k != "ledger_sha256"})),
          "Native ledger seal differs")
    return json.loads(_wire(receipt))


__all__ = ["SCHEMA", "UNAVAILABLE_SCHEMA", "build_native_context_delivery", "validate_native_context_delivery",
           "unavailable_native_context_delivery"]
