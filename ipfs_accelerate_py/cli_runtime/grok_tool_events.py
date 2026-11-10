"""Bounded tool-event byte observations; never provider request accounting.

The CLI's stdout is an unauthenticated event stream. A repeated output may be
a replacement, delta, or duplicate notification. These observations establish
none of those interpretations and never establish history resends or tokens.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

SCHEMA = "native-grok-tool-events@1"
COVERAGE_SCHEMA = "native-grok-tool-events@2"
MAX_TOOL_EVENTS = 64
MAX_FRAME_BYTES = 256 * 1024
MAX_PAYLOAD_BYTES = MAX_FRAME_BYTES
MAX_CALL_ID_BYTES = 1024
MAX_PAYLOAD_NODES = 4096
# Each non-root JSON node requires a value and separator. This v2 bound
# permits dense payloads inside the unchanged byte cap; it does not identify
# which v1 bound caused any previously observed undifferentiated overflow.
COVERAGE_MAX_PAYLOAD_NODES = MAX_PAYLOAD_BYTES // 2
MAX_PAYLOAD_DEPTH = 32
MAX_INTEGER_BITS = 4096
MAX_COUNT = (1 << 63) - 1
TOOLS = ("read_file", "search_replace", "grep", "list_dir", "todo_write", "run_terminal_cmd", "unknown")
STATUSES = ("pending", "in_progress", "completed", "failed", "unknown")
_EVENTS = ("tool_call", "tool_call_update", "unknown")
_RELATIONS = ("new_call", "repeated_call", "lifecycle_update", "orphan_update", "unattributed")
_REPETITIONS = ("not_measured", "first_observed", "same_call", "cross_call", "same_and_cross_call", "unattributed")
_STATES = ("missing", "null", "empty", "valid", "invalid", "oversized")
_SHAPES = ("missing", "null", "boolean", "integer", "number", "string", "object", "array", "invalid")
_KNOWN_FIELDS = frozenset({"type", "toolCallId", "toolName", "status", "rawInput", "rawOutput",
                         "title", "kind", "content", "locations"})
PAYLOAD_LANES = ("input", "output", "content")
_CONSTANTS = {
    "schema": SCHEMA, "row_limit": MAX_TOOL_EVENTS, "call_identity_limit": MAX_TOOL_EVENTS,
    "call_id_byte_limit": MAX_CALL_ID_BYTES, "payload_byte_limit": MAX_PAYLOAD_BYTES,
    "payload_node_limit": MAX_PAYLOAD_NODES, "payload_depth_limit": MAX_PAYLOAD_DEPTH,
    "payload_integer_bit_limit": MAX_INTEGER_BITS, "selection": "first_tool_events",
    "frame_measurement_scope": "exact-json-line-bytes-excluding-lf",
    "payload_measurement_scope": "separate-canonical-raw-input-raw-output-content-json",
    "payload_lanes_may_overlap": True, "unique_output_bytes_measured": False,
    "repetition_scope": "retained-tool-events-only", "repetition_categories_overlap": True,
    "replacement_or_delta_semantics": "unverified", "observed_only": True,
    "provider_request_bytes": None, "history_resend_bytes": None,
    "request_history_complete": False, "tool_result_coverage_complete": False,
    "event_origin_authenticated": False, "provider_tokens": None,
    "token_savings_claimed": False, "observation_only": True,
    "proof_authority": False, "execution_authority": False, "completion_authority": False,
    "omission_authority": False, "raw_provider_data_exported": False,
}
_AGGREGATE_FIELDS = ("measured_fields", "canonical_json_bytes", "repeated_events", "repeated_bytes",
    "same_call_repeated_events", "same_call_repeated_bytes", "cross_call_repeated_events", "cross_call_repeated_bytes")
_REASONS = ("stream_incomplete", "malformed_frames", "oversized_frames", "non_object_frames",
    "unknown_events", "post_final_events", "row_limit_exceeded", "counter_overflow",
    "unknown_event_shape", "invalid_payload", "payload_bound_exceeded", "unattributed_call",
    "call_id_bound_exceeded", "stream_tainted", "text_truncated", "timed_out")
_ROW_FIELDS = frozenset({"event_ordinal", "frame_ordinal", "elapsed_seconds", "after_final", "event_type",
    "tool", "tool_name_present", "status", "frame_bytes", "call_id_status", "call_id_sha256", "call_relation",
    "unknown_field_count", "input", "output", "content"})
_PAYLOAD_FIELDS = frozenset({"state", "shape", "canonical_json_bytes", "sha256", "repetition"})
_FIELDS = frozenset({*_CONSTANTS, "observed_events", "retained_events", "omitted_events", "rows",
    "retained_frame_bytes", "call_identities", "call_relation_counts", "payloads", "incompleteness", "observation_complete"})
_BOUND_REASONS = ("bytes", "nodes", "depth", "integer_bits")
_RESOLUTION_BASES = ("exact", "documented_alias", "unknown")
_TOOL_ALIASES = {"run_terminal_command": "run_terminal_cmd"}
_COVERAGE_CONSTANTS = {**_CONSTANTS, "schema": COVERAGE_SCHEMA,
    "payload_node_limit": COVERAGE_MAX_PAYLOAD_NODES, "tool_name_byte_limit": MAX_CALL_ID_BYTES,
    "tool_classification_scope": "retained-tool-events",
    "tool_alias_profile": "grok-1.0.46-public-tool-names@1",
    "post_final_coverage_scope": "retained-tool-events-with-conservative-overflow"}
_COVERAGE_FIELDS = _FIELDS | set(_COVERAGE_CONSTANTS) | {"classification_counts"}
_COVERAGE_ROW_FIELDS = _ROW_FIELDS | {"tool_name_shape", "tool_name_status", "tool_name_sha256",
    "status_present", "status_shape", "resolved_tool", "tool_resolution_basis"}
_MISSING = object()


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _count(value):
    return type(value) is int and 0 <= value <= MAX_COUNT


def _time(value):
    # Bound integers before math.isfinite, which otherwise can overflow.
    return type(value) in (int, float) and 0 <= value <= MAX_COUNT and math.isfinite(value)


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bucket(value, allowed):
    return value if type(value) is str and value in allowed else "unknown"


def _shape(value):
    if value is _MISSING:
        return "missing"
    return {type(None): "null", bool: "boolean", int: "integer", float: "number",
            str: "string", dict: "object", list: "array"}.get(type(value), "invalid")


def _resolve_tool(digest):
    if digest is not None:
        for name in TOOLS[:-1]:
            if digest == _sha(name.encode("utf-8")):
                return name, "exact"
        for name, resolved in _TOOL_ALIASES.items():
            if digest == _sha(name.encode("utf-8")):
                return resolved, "documented_alias"
    return "unknown", "unknown"


class _Bound(Exception):
    pass


def _payload(raw, *, coverage=False):
    result = dict(state="missing", shape="missing", canonical_json_bytes=None, sha256=None, repetition="not_measured")
    if coverage:
        result["bound_reason"] = None
    if raw is _MISSING:
        return result
    shapes = {type(None): "null", bool: "boolean", int: "integer", float: "number",
              str: "string", dict: "object", list: "array"}
    shape = shapes.get(type(raw), "invalid")
    nodes = 0
    string_bytes = 0
    def inspect(value, depth):
        nonlocal nodes, string_bytes
        nodes += 1
        if nodes > (COVERAGE_MAX_PAYLOAD_NODES if coverage else MAX_PAYLOAD_NODES):
            raise _Bound("nodes")
        if depth > MAX_PAYLOAD_DEPTH:
            raise _Bound("depth")
        kind = type(value)
        if kind is str:
            if len(value) > MAX_PAYLOAD_BYTES:
                raise _Bound("bytes")
            string_bytes += len(value.encode("utf-8"))
            if string_bytes > MAX_PAYLOAD_BYTES:
                raise _Bound("bytes")
        elif kind is int:
            if value.bit_length() > MAX_INTEGER_BITS:
                raise _Bound("integer_bits")
        elif kind is float:
            if not math.isfinite(value):
                raise ValueError
        elif kind is dict:
            for key, child in value.items():
                if type(key) is not str:
                    raise ValueError
                inspect(key, depth+1); inspect(child, depth+1)
        elif kind is list:
            for child in value:
                inspect(child, depth+1)
        elif kind not in (type(None), bool):
            raise ValueError
    try:
        inspect(raw, 0)
        serialized = json.dumps(raw, ensure_ascii=False, allow_nan=False, sort_keys=True,
                                separators=(",", ":")).encode("utf-8")
        if len(serialized) > MAX_PAYLOAD_BYTES:
            raise _Bound("bytes")
    except _Bound as error:
        return {**result, "state": "oversized", "shape": shape,
                **({"bound_reason": error.args[0]} if coverage else {})}
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        return {**result, "state": "invalid", "shape": "invalid"}
    state = "null" if raw is None else "empty" if type(raw) in (str, dict, list) and len(raw) == 0 else "valid"
    return dict(state=state, shape=shape, canonical_json_bytes=len(serialized), sha256=_sha(serialized),
                repetition="first_observed", **({"bound_reason": None} if coverage else {}))


def _identity(value):
    if value is _MISSING:
        return "missing", None
    if value is None:
        return "null", None
    if type(value) is not str:
        return "invalid", None
    if not value:
        return "empty", None
    if len(value) > MAX_CALL_ID_BYTES:
        return "oversized", None
    try:
        raw = value.encode("utf-8")
    except UnicodeError:
        return "invalid", None
    if len(raw) > MAX_CALL_ID_BYTES:
        return "oversized", None
    return "valid", _sha(raw)


def _context(state):
    _need(type(state) is dict, "Grok tool stream context required")
    integers = ("frames", "stdout_bytes", "malformed_frames", "oversized_frames", "non_object_frames",
                "unknown_events", "events_after_final")
    booleans = ("stream_complete", "stream_tainted", "text_truncated", "timed_out")
    _need(all(_count(state.get(key)) for key in integers)
        and all(type(state.get(key)) is bool for key in booleans), "invalid Grok tool stream counters")
    duration, final = state.get("duration_seconds"), state.get("final_event_seconds")
    _need((duration is None or _time(duration)) and (final is None or _time(final))
        and (final is None or duration is not None and final <= duration), "invalid Grok tool stream times")
    counts = state.get("event_counts")
    _need(type(counts) is dict and all(_count(counts.get(key)) for key in _EVENTS[:2]),
        "invalid Grok tool event parent counts")
    for key, names in (("tool_name_counts", TOOLS), ("tool_status_counts", STATUSES)):
        _need(type(state.get(key)) is dict and set(state[key]) == set(names)
            and all(_count(n) for n in state[key].values()), "invalid Grok tool parent buckets")
    total = counts["tool_call"] + counts["tool_call_update"]
    _need(min(total, MAX_COUNT) <= state["frames"] and state["events_after_final"] <= state["frames"],
        "inconsistent Grok tool parent frame counts")
    return min(total, MAX_COUNT)


def _classification_counts(rows):
    result = dict(resolved_tools=dict.fromkeys(TOOLS, 0), resolution_bases=dict.fromkeys(_RESOLUTION_BASES, 0),
        tool_name_shapes=dict.fromkeys(_SHAPES, 0), tool_name_statuses=dict.fromkeys(_STATES, 0),
        status_shapes=dict.fromkeys(_SHAPES, 0), status_presence=dict(present=0, missing=0),
        payload_bounds={lane: dict.fromkeys(_BOUND_REASONS, 0) for lane in PAYLOAD_LANES})
    for row in rows:
        for aggregate, key in (("resolved_tools", "resolved_tool"), ("resolution_bases", "tool_resolution_basis"),
                ("tool_name_shapes", "tool_name_shape"), ("tool_name_statuses", "tool_name_status"),
                ("status_shapes", "status_shape")):
            result[aggregate][row[key]] += 1
        result["status_presence"]["present" if row["status_present"] else "missing"] += 1
        for lane in PAYLOAD_LANES:
            if row[lane]["bound_reason"] is not None:
                result["payload_bounds"][lane][row[lane]["bound_reason"]] += 1
    return result


def _derive(rows, observed, state, *, coverage=False):
    identities = set()
    payload_seen = {name: {} for name in PAYLOAD_LANES}
    aggregates = {name: dict.fromkeys(_AGGREGATE_FIELDS, 0) for name in payload_seen}
    relations = dict.fromkeys(_RELATIONS, 0)
    copied = []
    for original in rows:
        row = {**original, **{name: dict(original[name]) for name in PAYLOAD_LANES}}
        identity = row["call_id_sha256"]
        if identity is None:
            relation = "unattributed"
        elif row["event_type"] == "tool_call":
            relation = "repeated_call" if identity in identities else "new_call"
        else:
            relation = "lifecycle_update" if identity in identities else "orphan_update"
        if identity is not None:
            identities.add(identity)
        row["call_relation"] = relation
        relations[relation] += 1
        for name in payload_seen:
            item = row[name]
            digest = item["sha256"]
            if digest is None:
                item["repetition"] = "not_measured"
                continue
            seen = payload_seen[name].setdefault(digest, set())
            same = identity is not None and identity in seen
            cross = identity is not None and bool(seen - {identity, None})
            repetition = ("first_observed" if not seen else
                "same_and_cross_call" if same and cross else "same_call" if same else
                "cross_call" if cross else "unattributed")
            item["repetition"] = repetition
            size = item["canonical_json_bytes"]
            stats = aggregates[name]
            stats["measured_fields"] += 1
            stats["canonical_json_bytes"] += size
            if seen:
                stats["repeated_events"] += 1
                stats["repeated_bytes"] += size
            for label, match in (("same_call", same), ("cross_call", cross)):
                if match:
                    stats[label+"_repeated_events"] += 1
                    stats[label+"_repeated_bytes"] += size
            seen.add(identity)
        copied.append(row)
    reasons = dict.fromkeys(_REASONS, False)
    reasons.update(stream_incomplete=not state["stream_complete"],
        row_limit_exceeded=observed > MAX_TOOL_EVENTS,
        counter_overflow=observed == MAX_COUNT or any(state[key] == MAX_COUNT for key in
            ("frames", "stdout_bytes", "malformed_frames", "oversized_frames", "non_object_frames", "unknown_events", "events_after_final")),
        post_final_events=state["events_after_final"] > 0)
    for key in ("malformed_frames", "oversized_frames", "non_object_frames", "unknown_events"):
        reasons[key] = state[key] > 0
    for key in ("stream_tainted", "text_truncated", "timed_out"):
        reasons[key] = state[key]
    for row in copied:
        reasons["unknown_event_shape"] |= (row["unknown_field_count"] > 0 or row["event_type"] == "unknown"
            or row["status"] == "unknown"
            or (row["event_type"] == "tool_call" or row["tool_name_present"]) and row["tool"] == "unknown"
            or row["call_id_status"] == "invalid")
        reasons["unattributed_call"] |= row["call_id_sha256"] is None
        reasons["call_id_bound_exceeded"] |= row["call_id_status"] == "oversized"
        for name in PAYLOAD_LANES:
            reasons["invalid_payload"] |= row[name]["state"] == "invalid"
            reasons["payload_bound_exceeded"] |= row[name]["state"] == "oversized"
    if coverage:
        # The parent parser alone admits a terminal trailer. Its post-final
        # count is not evidence of an additional tool event. Omitted rows and
        # tainted/unknown frames remain conservatively incomplete.
        reasons["post_final_events"] = any(row["after_final"] for row in copied) or (
            observed > len(copied) and state["events_after_final"] > 0)
        reasons["unknown_event_shape"] = any(row["unknown_field_count"] > 0
            or row["event_type"] == "unknown" or row["call_id_status"] == "invalid"
            or (row["event_type"] == "tool_call" or row["tool_name_present"]) and row["resolved_tool"] == "unknown"
            or (row["event_type"] == "tool_call" or row["status_present"]) and row["status"] == "unknown"
            for row in copied)
    return {**(_COVERAGE_CONSTANTS if coverage else _CONSTANTS),
        "observed_events": observed, "retained_events": len(copied),
        "omitted_events": observed-len(copied), "rows": copied,
        "retained_frame_bytes": sum(row["frame_bytes"] for row in copied),
        "call_identities": len(identities), "call_relation_counts": relations,
        "payloads": aggregates, "incompleteness": reasons, "observation_complete": not any(reasons.values()),
        **({"classification_counts": _classification_counts(copied)} if coverage else {})}


def _validate_payload(value, *, coverage=False):
    fields = _PAYLOAD_FIELDS | {"bound_reason"} if coverage else _PAYLOAD_FIELDS
    _need(type(value) is dict and set(value) == fields, "invalid Grok tool payload fields")
    _need(type(value["state"]) is str and value["state"] in _STATES
        and type(value["shape"]) is str and value["shape"] in _SHAPES
        and type(value["repetition"]) is str and value["repetition"] in _REPETITIONS,
        "invalid Grok tool payload labels")
    status, shape, size, digest = (value[key] for key in ("state", "shape", "canonical_json_bytes", "sha256"))
    if coverage:
        reason = value["bound_reason"]
        _need(type(reason) is str and reason in _BOUND_REASONS if status == "oversized" else reason is None,
            "invalid Grok tool payload bound reason")
        if reason is not None:
            allowed = {"bytes": {"string", "object", "array"}, "nodes": {"object", "array"},
                "depth": {"object", "array"}, "integer_bits": {"integer", "object", "array"}}
            _need(shape in allowed[reason], "inconsistent Grok tool payload bound shape")
    if status in {"missing", "invalid", "oversized"}:
        _need(size is None and digest is None and value["repetition"] == "not_measured"
            and (shape == status if status in {"missing", "invalid"} else shape not in {"missing", "null", "invalid"}),
            "invalid Grok tool unmeasured payload")
        return
    _need(_count(size) and 1 <= size <= MAX_PAYLOAD_BYTES and _digest(digest), "invalid Grok tool payload measure")
    if status == "null":
        _need(shape == "null" and size == 4 and digest == _sha(b"null"), "invalid Grok tool null payload")
    elif status == "empty":
        empty = {"string": b'""', "object": b"{}", "array": b"[]"}
        _need(shape in empty and size == 2 and digest == _sha(empty[shape]), "invalid Grok tool empty payload")
    else:
        _need(shape not in {"missing", "null", "invalid"}, "invalid Grok tool payload shape")


def _same_types_and_values(value, expected):
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(_same_types_and_values(value[key], item) for key, item in expected.items())
    if type(expected) is list:
        return len(value) == len(expected) and all(_same_types_and_values(left, right) for left, right in zip(value, expected))
    return value == expected


def _validate_coverage_labels(row):
    for key, allowed in (("tool_name_shape", _SHAPES), ("status_shape", _SHAPES),
            ("tool_name_status", _STATES), ("resolved_tool", TOOLS), ("tool_resolution_basis", _RESOLUTION_BASES)):
        _need(type(row[key]) is str and row[key] in allowed, "invalid Grok tool coverage label")
    _need(type(row["status_present"]) is bool, "invalid Grok tool status presence")
    shape, status, digest = row["tool_name_shape"], row["tool_name_status"], row["tool_name_sha256"]
    _need((shape == "missing") == (not row["tool_name_present"])
        and (status == "missing") == (not row["tool_name_present"]), "inconsistent Grok tool name presence")
    if status == "valid":
        _need(shape == "string" and _digest(digest), "invalid Grok tool name digest")
    else:
        _need(digest is None, "invalid Grok unmeasured tool name digest")
        if status == "null": _need(shape == "null", "invalid Grok null tool name")
        if status in {"empty", "oversized"}: _need(shape == "string", "invalid Grok tool name string shape")
        if status == "invalid": _need(shape not in {"missing", "null"}, "invalid Grok tool name shape")
    resolved, basis = _resolve_tool(digest)
    _need(row["resolved_tool"] == resolved and row["tool_resolution_basis"] == basis
        and row["tool"] == (resolved if basis == "exact" else "unknown"), "inconsistent Grok tool classification")
    _need((row["status_shape"] == "missing") == (not row["status_present"])
        and (row["status"] == "unknown" or row["status_shape"] == "string"),
        "inconsistent Grok tool status shape")


def validate_grok_tool_events(value: object, *, stream_state: dict) -> dict:
    """Validate closed rows and recompute counts/repetition against parent state."""
    observed = _context(stream_state)
    _need(type(value) is dict and type(value.get("schema")) is str
        and value["schema"] in {SCHEMA, COVERAGE_SCHEMA}, "invalid Grok tool census schema")
    coverage = value["schema"] == COVERAGE_SCHEMA
    _need(set(value) == (_COVERAGE_FIELDS if coverage else _FIELDS), "invalid Grok tool census fields")
    for key, expected in (_COVERAGE_CONSTANTS if coverage else _CONSTANTS).items():
        _need(type(value[key]) is type(expected) and value[key] == expected, "invalid Grok tool census constants")
    _need(all(_count(value[key]) for key in ("observed_events", "retained_events", "omitted_events",
        "retained_frame_bytes", "call_identities")) and value["observed_events"] == observed
        and value["retained_events"] == min(observed, MAX_TOOL_EVENTS)
        and value["omitted_events"] == observed-value["retained_events"], "invalid Grok tool census counts")
    rows = value["rows"]
    _need(type(rows) is list and len(rows) == value["retained_events"], "invalid Grok tool census rows")
    last_frame, last_elapsed = 0, 0
    kinds = dict.fromkeys(_EVENTS, 0); names = dict.fromkeys(TOOLS, 0); statuses = dict.fromkeys(STATUSES, 0)
    for ordinal, row in enumerate(rows, 1):
        _need(type(row) is dict and set(row) == (_COVERAGE_ROW_FIELDS if coverage else _ROW_FIELDS),
            "invalid Grok tool event fields")
        _need(_count(row["event_ordinal"]) and row["event_ordinal"] == ordinal
            and _count(row["frame_ordinal"]) and last_frame < row["frame_ordinal"] <= stream_state["frames"]
            and _count(row["frame_bytes"]) and 1 <= row["frame_bytes"] <= MAX_FRAME_BYTES
            and _count(row["unknown_field_count"]), "invalid Grok tool event ordinal or bounds")
        elapsed = row["elapsed_seconds"]; final = stream_state["final_event_seconds"]
        _need(_time(elapsed) and stream_state["duration_seconds"] is not None
            and last_elapsed <= elapsed <= stream_state["duration_seconds"], "invalid Grok tool event time")
        _need(type(row["after_final"]) is bool and type(row["tool_name_present"]) is bool
            and (row["tool_name_present"] or row["tool"] == "unknown"),
            "invalid Grok tool event flags")
        if row["after_final"]:
            _need(final is not None and elapsed >= final and
                row["frame_ordinal"] > stream_state["frames"]-stream_state["events_after_final"],
                "invalid Grok tool final boundary")
        else:
            _need((final is None or elapsed <= final) and
                row["frame_ordinal"] <= stream_state["frames"]-stream_state["events_after_final"],
                "invalid Grok tool prefinal boundary")
        for key, labels in (("event_type", _EVENTS), ("tool", TOOLS), ("status", STATUSES),
                            ("call_relation", _RELATIONS), ("call_id_status", _STATES)):
            _need(type(row[key]) is str and row[key] in labels, "invalid Grok tool event vocabulary")
        _need(_digest(row["call_id_sha256"]) if row["call_id_status"] == "valid"
            else row["call_id_sha256"] is None, "invalid Grok tool identity binding")
        if coverage:
            _validate_coverage_labels(row)
        for name in PAYLOAD_LANES:
            _validate_payload(row[name], coverage=coverage)
        kinds[row["event_type"]] += 1
        statuses[row["status"]] += 1
        if row["event_type"] == "tool_call": names[row["tool"]] += 1
        last_frame, last_elapsed = row["frame_ordinal"], elapsed
    _need(kinds["unknown"] == 0 and all(kinds[key] <= stream_state["event_counts"][key] for key in _EVENTS[:2])
        and all(names[key] <= stream_state["tool_name_counts"][key] for key in TOOLS)
        and all(statuses[key] <= stream_state["tool_status_counts"][key] for key in STATUSES),
        "inconsistent Grok tool parent event buckets")
    if value["omitted_events"] == 0:
        _need(all(kinds[key] == stream_state["event_counts"][key] for key in _EVENTS[:2])
            and names == stream_state["tool_name_counts"] and statuses == stream_state["tool_status_counts"],
            "incomplete Grok tool parent event buckets")
    _need(sum(row["frame_bytes"] for row in rows) <= stream_state["stdout_bytes"],
        "Grok tool frame bytes exceed stream bytes")
    _need(type(value["incompleteness"]) is dict and set(value["incompleteness"]) == set(_REASONS)
        and all(type(flag) is bool for flag in value["incompleteness"].values())
        and type(value["observation_complete"]) is bool, "invalid Grok tool coverage fields")
    _need(type(value["call_relation_counts"]) is dict and set(value["call_relation_counts"]) == set(_RELATIONS)
        and all(_count(n) for n in value["call_relation_counts"].values()), "invalid Grok tool relation counts")
    _need(type(value["payloads"]) is dict and set(value["payloads"]) == set(PAYLOAD_LANES),
        "invalid Grok tool payload totals")
    for item in value["payloads"].values():
        _need(type(item) is dict and set(item) == set(_AGGREGATE_FIELDS) and all(_count(n) for n in item.values()),
            "invalid Grok tool payload counters")
    expected = _derive(rows, observed, stream_state, coverage=coverage)
    _need(_same_types_and_values(value, expected) if coverage else value == expected,
        "inconsistent Grok tool census replay")
    return expected


class GrokToolEventObserver:
    """Keep at most 64 hashed rows; no IO, raw payload retention, or authority."""

    def __init__(self, *, schema=SCHEMA):
        _need(type(schema) is str and schema in {SCHEMA, COVERAGE_SCHEMA}, "unsupported Grok tool census schema")
        self.schema = schema
        self.observed_events = 0
        self.rows = []

    def observe(self, event: dict, *, frame_bytes: int, frame_ordinal: int,
                elapsed_seconds: float, after_final: bool):
        self.observed_events = min(MAX_COUNT, self.observed_events+1)
        if len(self.rows) >= MAX_TOOL_EVENTS:
            return
        event = event if type(event) is dict else {}
        identity_status, identity = _identity(event.get("toolCallId", _MISSING))
        coverage = self.schema == COVERAGE_SCHEMA
        labels = {}
        if coverage:
            tool_name = event.get("toolName", _MISSING)
            name_status, name_digest = _identity(tool_name)
            resolved, basis = _resolve_tool(name_digest)
            labels = dict(tool_name_shape=_shape(tool_name), tool_name_status=name_status, tool_name_sha256=name_digest,
                status_present="status" in event, status_shape=_shape(event.get("status", _MISSING)),
                resolved_tool=resolved, tool_resolution_basis=basis)
        self.rows.append(dict(event_ordinal=self.observed_events, frame_ordinal=frame_ordinal,
            elapsed_seconds=elapsed_seconds, after_final=after_final, frame_bytes=frame_bytes,
            event_type=_bucket(event.get("type"), _EVENTS), tool=_bucket(event.get("toolName"), TOOLS),
            tool_name_present="toolName" in event,
            status=_bucket(event.get("status"), STATUSES), call_id_status=identity_status,
            call_id_sha256=identity, call_relation="unattributed",
            unknown_field_count=sum(key not in _KNOWN_FIELDS for key in event),
            input=_payload(event.get("rawInput", _MISSING), coverage=coverage),
            output=_payload(event.get("rawOutput", _MISSING), coverage=coverage),
            content=_payload(event.get("content", _MISSING), coverage=coverage), **labels))

    def snapshot(self, *, stream_state: dict) -> dict:
        """Use final parser coverage flags; callbacks need not retain event rows."""
        _need(self.observed_events == _context(stream_state), "Grok tool observer count differs from parser")
        return validate_grok_tool_events(_derive(self.rows, self.observed_events, stream_state,
            coverage=self.schema == COVERAGE_SCHEMA), stream_state=stream_state)


__all__ = ["GrokToolEventObserver", "validate_grok_tool_events", "SCHEMA", "COVERAGE_SCHEMA", "MAX_TOOL_EVENTS"]
