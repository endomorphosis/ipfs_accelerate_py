"""Bounded, content-free observations from an already selected Codex rollout.

These observations supplement native aggregate usage; they never replace it.
Checkpoints are cumulative observations, not provider requests or model turns.
"""
from __future__ import annotations

from datetime import datetime
import json
import math


SCHEMA = "native-codex-interaction-telemetry@1"
MAX_TOOL_PAIRS = 128
MAX_TOKEN_OBSERVATIONS = 256
MAX_RECORDS = 100_000
MAX_PAYLOAD_BYTES = 4_000_000
MAX_COUNTER = (1 << 63) - 1
MAX_ELAPSED_SECONDS = 86_400
_KNOWN_TOOLS = frozenset({
    "exec_command", "write_stdin", "apply_patch", "view_image",
    "functions.exec", "functions.wait", "functions.exec_command",
    "functions.write_stdin", "functions.apply_patch", "functions.view_image",
    "shell", "shell_command", "update_plan", "request_user_input",
})
_COUNTERS = (
    "input_tokens", "output_tokens", "total_tokens", "cached_input_tokens",
    "cache_write_input_tokens", "reasoning_output_tokens",
)
_REQUIRED_COUNTERS = frozenset({"input_tokens", "output_tokens", "total_tokens"})
_CALLS = {"function_call": "function", "custom_tool_call": "custom"}
_OUTPUTS = {"function_call_output": "function", "custom_tool_call_output": "custom"}


def _integer(value, *, maximum=MAX_COUNTER):
    return type(value) is int and 0 <= value <= maximum


def _payload_size(value):
    """Count decoded strings or bounded canonical JSON; never return content."""
    if type(value) is str:
        if len(value) > MAX_PAYLOAD_BYTES:
            return None, "decoded_utf8_string", "payload_limit"
        try:
            size = len(value.encode("utf-8"))
        except UnicodeError:
            return None, "decoded_utf8_string", "invalid_unicode"
        return (size if size <= MAX_PAYLOAD_BYTES else None,
                "decoded_utf8_string", None if size <= MAX_PAYLOAD_BYTES else "payload_limit")
    if type(value) not in (dict, list):
        return None, None, "unsupported_payload"
    # Bound traversal before serializing. The input is expected to be decoded
    # JSON, but rejecting cycles/non-JSON values makes direct use safe as well.
    pending = [(value, 0)]
    nodes = chars = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > 10_000 or depth > 32:
            return None, "canonical_json_utf8", "structure_limit"
        if type(item) is str:
            chars += len(item)
        elif type(item) is dict:
            if len(item) + len(pending) > 10_000 or any(type(key) is not str for key in item):
                return None, "canonical_json_utf8", "unsupported_structure"
            chars += sum(len(key) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            if len(item) + len(pending) > 10_000:
                return None, "canonical_json_utf8", "structure_limit"
            pending.extend((child, depth + 1) for child in item)
        elif item is not None and type(item) not in (bool, int, float):
            return None, "canonical_json_utf8", "unsupported_structure"
        elif type(item) is float and not math.isfinite(item):
            return None, "canonical_json_utf8", "nonfinite_number"
        elif type(item) is int and item.bit_length() > 4096:
            return None, "canonical_json_utf8", "structure_limit"
        if chars > MAX_PAYLOAD_BYTES:
            return None, "canonical_json_utf8", "payload_limit"
    try:
        size = len(json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False).encode("utf-8"))
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return None, "canonical_json_utf8", "unsupported_structure"
    return (size if size <= MAX_PAYLOAD_BYTES else None,
            "canonical_json_utf8", None if size <= MAX_PAYLOAD_BYTES else "payload_limit")


def _timestamp(record):
    value = record.get("timestamp")
    if type(value) is not str or len(value) > 64:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.utcoffset() is not None else None
    except (ValueError, OverflowError, OSError):
        return None


def _structured_status(value):
    """Only explicit structured fields have status semantics, never prose."""
    if type(value) is not dict:
        return None, None
    code = value.get("exit_code")
    if type(code) is not int or not -(1 << 31) <= code < (1 << 31):
        code = None
    flags = [value[key] for key in ("is_error", "isError", "error")
             if key in value and type(value[key]) is bool]
    if flags and any(flag != flags[0] for flag in flags):
        return None, "conflicting_structured_status"
    error = flags[0] if flags else None
    if code == 0 and error is True:
        return None, "conflicting_structured_status"
    if code is not None:
        return code, "observed_zero_exit" if code == 0 else "observed_nonzero_exit"
    return None, "observed_error_flag" if error is True else None


def validated_cumulative_usage(payload):
    """Return bounded consistent native cumulative counters, or an unknown reason."""
    if type(payload) is not dict:
        return None, "missing_cumulative_counters"
    info = payload.get("info")
    value = info.get("total_token_usage") if type(info) is dict else None
    if type(value) is not dict or not _REQUIRED_COUNTERS.issubset(value):
        return None, "missing_cumulative_counters"
    selected = {key: value[key] for key in _COUNTERS if key in value}
    if any(not _integer(item) for item in selected.values()):
        return None, "invalid_counter"
    if selected["total_tokens"] != selected["input_tokens"] + selected["output_tokens"]:
        return None, "inconsistent_total"
    for key in ("cached_input_tokens", "cache_write_input_tokens"):
        if key in selected and selected[key] > selected["input_tokens"]:
            return None, "inconsistent_input_subset"
    if selected.get("reasoning_output_tokens", 0) > selected["output_tokens"]:
        return None, "inconsistent_output_subset"
    return selected, None


def build_interaction_telemetry(records, *, completed, malformed, session_start_known=False):
    """Observe one selected rollout without exporting text or call identifiers.

    ``session_start_known=True`` explicitly asserts a zero counter origin. A
    session_meta record alone never establishes this (sessions can be resumed).
    Callers must pass records from one independently selected native rollout;
    observed session_meta boundaries reset pairing and counter continuity.
    """
    malformed_known = _integer(malformed)
    malformed_count = malformed if malformed_known else None
    states = {}
    rows = []
    token_rows = []
    calls = outputs = bad_records = invalid_ids = omitted_events = 0
    invalid_snapshots = invalid_intervals = resets = repeated = token_count = omitted_tokens = 0
    non_usage_token_events = 0
    previous = None
    first_checkpoint = True
    baseline_allowed = session_start_known is True
    interval_calls = interval_outputs = 0
    scope = 1
    seen_meta = False
    boundary = None
    boundary_count = 0
    record_count = 0
    record_limit = False
    timestamps_monotonic = True
    valid_timestamps = 0
    prior_timestamp = None
    for ordinal, record in enumerate(records, 1):
        if ordinal > MAX_RECORDS:
            record_limit = True
            break
        record_count = ordinal
        if type(record) is not dict:
            bad_records += 1
            previous = None
            baseline_allowed = False
            continue
        kind = record.get("type")
        if type(kind) is not str:
            bad_records += 1
            previous = None
            baseline_allowed = False
            continue
        stamp = _timestamp(record)
        if stamp is not None:
            valid_timestamps += 1
            if prior_timestamp is not None and stamp < prior_timestamp:
                timestamps_monotonic = False
            prior_timestamp = stamp
        payload = record.get("payload")
        if kind == "session_meta":
            if seen_meta or ordinal > 1:
                scope += 1
                previous = None
                baseline_allowed = False
                first_checkpoint = False
                interval_calls = interval_outputs = 0
                boundary = None
            seen_meta = True
            continue
        if type(payload) is not dict:
            if kind in ("response_item", "event_msg"):
                bad_records += 1
                previous = None
                baseline_allowed = False
            continue
        item_kind = payload.get("type")
        if kind in ("response_item", "event_msg") and type(item_kind) is not str:
            bad_records += 1
            previous = None
            baseline_allowed = False
            continue
        if kind == "event_msg" and item_kind in ("task_started", "turn_started"):
            boundary_count += 1
            boundary = boundary_count
        if kind == "event_msg" and item_kind == "token_count":
            info = payload.get("info")
            if info is None or (type(info) is dict and "total_token_usage" not in info):
                # Native quota/rate-limit updates use token_count without a
                # cumulative snapshot. They do not delimit usage intervals.
                non_usage_token_events += 1
                continue
            token_count += 1
            counters, reason = validated_cumulative_usage(payload)
            delta = None
            interval_status = "unknown_baseline"
            if counters is None:
                invalid_snapshots += 1
                interval_status = "invalid_snapshot"
            elif previous is not None:
                common = set(counters).intersection(previous)
                if any(counters[key] < previous[key] for key in common):
                    resets += 1
                    interval_status = "counter_decrease"
                    reason = "counter_decrease"
                else:
                    delta = {key: counters[key] - previous[key] for key in common}
                    interval_status = "adjacent_observations"
                    if counters == previous:
                        repeated += 1
                        interval_status = "repeated_snapshot"
                    # Subsets can be absent on either endpoint. Keep absent
                    # categories unknown, and reject inconsistent intervals.
                    if (delta.get("cached_input_tokens", 0) > delta["input_tokens"]
                            or delta.get("cache_write_input_tokens", 0) > delta["input_tokens"]
                            or delta.get("reasoning_output_tokens", 0) > delta["output_tokens"]):
                        delta = None
                        interval_status = "invalid_interval"
                        reason = "inconsistent_delta_subset"
            elif first_checkpoint and baseline_allowed and malformed_count == 0 and bad_records == 0:
                delta = dict(counters)
                interval_status = "asserted_zero_origin"
            if interval_status == "invalid_interval":
                invalid_intervals += 1
            if malformed_count != 0 and counters is not None:
                # A total malformed-line count gives no positions. It cannot
                # establish which adjacent checkpoint intervals were intact.
                delta = None
                interval_status = "unknown_malformed_record_positions"
                reason = "malformed_record_positions_unavailable"
            row = {
                "observation_ordinal": token_count, "record_ordinal": ordinal,
                "session_scope_ordinal": scope, "explicit_boundary_event_ordinal": boundary,
                "cumulative_counters": counters, "adjacent_counter_delta": delta,
                "interval_status": interval_status, "unknown_reason": reason,
                "observed_tool_call_events_since_previous_checkpoint": interval_calls,
                "observed_tool_output_events_since_previous_checkpoint": interval_outputs,
            }
            if len(token_rows) < MAX_TOKEN_OBSERVATIONS:
                token_rows.append(row)
            else:
                omitted_tokens += 1
            # A rejected interval cannot establish a new trusted endpoint:
            # the next valid snapshot must first re-establish a baseline.
            previous = counters if interval_status not in (
                "invalid_snapshot", "counter_decrease", "invalid_interval") else None
            first_checkpoint = False
            baseline_allowed = False
            interval_calls = interval_outputs = 0
        if kind != "response_item" or type(item_kind) is not str:
            continue
        is_call = item_kind in _CALLS
        if not is_call and item_kind not in _OUTPUTS:
            continue
        if is_call:
            calls += 1
            interval_calls += 1
        else:
            outputs += 1
            interval_outputs += 1
        identity = payload.get("call_id")
        if type(identity) is not str or not identity or len(identity) > 256:
            invalid_ids += 1
            continue
        key = (scope, identity)
        state = states.get(key)
        if state is None:
            if len(rows) >= MAX_TOOL_PAIRS:
                omitted_events += 1
                continue
            state = {
                "observation_ordinal": len(rows) + 1, "first_record_ordinal": ordinal,
                "session_scope_ordinal": scope, "explicit_boundary_event_ordinal": boundary,
                "tool_name": None, "tool_kind": None,
                "call_event_count": 0, "output_event_count": 0,
                "call_record_ordinal": None, "output_record_ordinal": None,
                "input_bytes": None, "input_byte_scope": None,
                "observed_output_bytes": 0, "output_byte_scope": None,
                "input_size_unknown_reason": None, "output_size_unknown_reason": None,
                "structured_exit_code": None, "structured_output_status": None,
                "recorded_observation_elapsed_seconds": None,
                "_call_time": None, "_output_time": None, "_conflict": False,
            }
            states[key] = state
            rows.append(state)
        expected_kind = (_CALLS if is_call else _OUTPUTS)[item_kind]
        if state["tool_kind"] is not None and state["tool_kind"] != expected_kind:
            state["_conflict"] = True
        state["tool_kind"] = expected_kind
        if is_call:
            state["call_event_count"] += 1
            if state["call_event_count"] == 1:
                state["call_record_ordinal"] = ordinal
                name = payload.get("name")
                state["tool_name"] = name if type(name) is str and name in _KNOWN_TOOLS else "other"
                value = payload.get("arguments" if item_kind == "function_call" else "input")
                size, byte_scope, unknown_reason = _payload_size(value)
                state["input_bytes"], state["input_byte_scope"] = size, byte_scope
                state["input_size_unknown_reason"] = unknown_reason
                state["_call_time"] = stamp
        else:
            state["output_event_count"] += 1
            size, byte_scope, unknown_reason = _payload_size(payload.get("output"))
            if size is None or state["observed_output_bytes"] is None:
                state["observed_output_bytes"] = None
            else:
                state["observed_output_bytes"] += size
            if state["output_event_count"] == 1:
                state["output_record_ordinal"] = ordinal
                state["output_byte_scope"] = byte_scope
                state["_output_time"] = stamp
                code, status = _structured_status(payload.get("output"))
                state["structured_exit_code"], state["structured_output_status"] = code, status
            elif state["output_byte_scope"] != byte_scope:
                state["output_byte_scope"] = "mixed_decoded_string_and_canonical_json"
            if unknown_reason is not None:
                state["output_size_unknown_reason"] = unknown_reason
    paired = 0
    for state in rows:
        call_count, output_count = state["call_event_count"], state["output_event_count"]
        unique = call_count == output_count == 1 and not state["_conflict"]
        ordered = (unique and state["call_record_ordinal"] < state["output_record_ordinal"])
        if state["_conflict"] or call_count > 1:
            status = "ambiguous_call_identity"
        elif call_count == 0:
            status = "output_without_observed_call"
        elif output_count == 0:
            status = "call_without_observed_output"
        elif output_count > 1:
            status = "multiple_output_chunks_observed"
        elif not ordered:
            status = "output_precedes_call"
        else:
            status = "single_call_and_output_observed"
            paired += 1
        state["pairing_status"] = status
        start, stop = state.pop("_call_time"), state.pop("_output_time")
        state.pop("_conflict")
        if ordered and timestamps_monotonic and start is not None and stop is not None:
            elapsed = stop - start
            if math.isfinite(elapsed) and 0 <= elapsed <= MAX_ELAPSED_SECONDS:
                state["recorded_observation_elapsed_seconds"] = elapsed
        if call_count != 1:
            state["input_bytes"] = None
            state["input_size_unknown_reason"] = "nonunique_call" if call_count else "missing_call"
        if output_count == 0:
            state["observed_output_bytes"] = None
            state["output_size_unknown_reason"] = "missing_output"
        if not ordered:
            state["structured_exit_code"] = state["structured_output_status"] = None
    all_records_known = (completed is True and malformed_count == 0 and bad_records == 0
                         and not record_limit)
    tool_complete = (all_records_known and invalid_ids == omitted_events == 0
                     and paired == len(rows)
                     and all(row["input_bytes"] is not None and row["observed_output_bytes"] is not None
                             for row in rows))
    return {
        "schema": SCHEMA, "task_complete_observed": completed is True,
        "observed_records": record_count, "malformed_json_records": malformed_count,
        "invalid_record_shapes": bad_records, "record_limit_exceeded": record_limit,
        "limits": {"records": MAX_RECORDS, "tool_pairs": MAX_TOOL_PAIRS,
                   "token_observations": MAX_TOKEN_OBSERVATIONS, "payload_bytes": MAX_PAYLOAD_BYTES},
        "observed_tool_call_events": calls, "observed_tool_output_events": outputs,
        "invalid_tool_identity_events": invalid_ids, "omitted_tool_events": omitted_events,
        "tool_pairs": rows, "tool_observations_complete": tool_complete,
        "observed_token_checkpoints": token_count, "omitted_token_observations": omitted_tokens,
        "non_usage_token_events": non_usage_token_events,
        "invalid_token_snapshots": invalid_snapshots, "counter_decrease_intervals": resets,
        "invalid_counter_intervals": invalid_intervals,
        "repeated_cumulative_snapshots": repeated, "token_observations": token_rows,
        "explicit_boundary_events_observed": boundary_count,
        "token_observations_complete": (all_records_known and invalid_snapshots == 0
                                        and invalid_intervals == 0 and resets == 0 and omitted_tokens == 0),
        "timestamp_order_monotonic": timestamps_monotonic,
        "valid_timestamp_events": valid_timestamps,
        "raw_content_exported": False, "provider_request_count_inferred": False,
        "provider_request_count_known": False, "tool_token_attribution_available": False,
        "causal_tool_token_attribution": False, "native_aggregate_usage_modified": False,
        "usage_completeness_established": False,
        "byte_scope": "decoded UTF-8 strings or canonical structured JSON; not provider payload bytes or tokens",
        "interval_scope": "adjacent cumulative observations; tool counts are co-observations, not causal attribution",
        "completeness_scope": "bounded observation census only; never proof of complete billed usage or tool execution",
        "elapsed_scope": "recorded observation elapsed time; not tool execution latency",
    }
