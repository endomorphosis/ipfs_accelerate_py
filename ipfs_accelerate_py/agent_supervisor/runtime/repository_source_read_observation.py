"""Content-free observations of prepared repository reads, not consumption.

Identities bind the selected file/symbol and canonical source version. Repeated
reads are classified within that identity, representation, encoding and returned
content. A generation change alone does not hide unchanged source; a changed
canonical source hash remains a distinct identity even if compact text matches.
The event-chain digest commits to ordered read observations, not independent
provenance. Group validation can reconstruct counts but not that event order.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re


EVENT_SCHEMA = "repository-source-read-event@1"
SUMMARY_SCHEMA = "repository-source-read-observation@1"
MAX_READS = 512
MAX_CONTENT_BYTES = 1024 * 1024
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024
MAX_SUMMARY_BYTES = 512 * 1024
_GROUP_KEYS = ("generation", "source_identity_sha256", "scope", "representation",
               "encoding", "content_sha256", "content_bytes")
_EVENT_FIELDS = frozenset({"schema", *_GROUP_KEYS})
_GROUP_FIELDS = frozenset({*_GROUP_KEYS, "read_count"})
_AUDIT_FIELDS = frozenset({"schema", "sequence", "event", "tool", "success", "returned_bytes",
                          "compact", "canonical_changed", "canonical_uncertain", "refreshed"})
_TOOLS = frozenset({"repository_status", "repository_list", "repository_symbols", "repository_read",
    "repository_replace_symbol", "repository_refresh", "repository_cache_inventory",
    "repository_context_query", "repository_capsule_expand"})
_TOTALS = ("content_bytes", "first_identity_reads", "same_generation_repeat_reads",
           "cross_generation_repeat_reads", "first_identity_bytes", "same_generation_repeat_bytes",
           "cross_generation_repeat_bytes")
_POPULATION = ("successful_reads", "identified_reads", "legacy_successful_reads",
               "unbound_successful_reads", "read_call_count", "read_error_count")
_PAYLOAD_TOTALS = ("successful_read_payload_bytes", "failed_read_payload_bytes", "other_operation_payload_bytes")
_UNKNOWN = ("provider_ingestion_verified", "provider_tokens", "token_savings", "inline_overlap_bytes")
_FALSE = ("proof_authority", "completion_authority", "journal_closure_verified")
_SUMMARY_FIELDS = frozenset({"schema", "scope", "population_scope", "payload_byte_scope", "complete", "event_chain_sha256",
    "groups", *_POPULATION, *_PAYLOAD_TOTALS, *_TOTALS, *("known_" + name for name in _TOTALS), *_UNKNOWN, *_FALSE})


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def _digest(value):
    return type(value) is str and len(value) == 64 and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _enum(value, choices):
    return type(value) is str and value in choices


def _text(value, limit):
    if type(value) is not str or not 0 < len(value) <= limit:
        return False
    try:
        return len(value.encode("utf-8")) <= limit and not any(ord(char) < 32 or ord(char) == 127 for char in value)
    except UnicodeError:
        return False


def validate_read_event(event):
    """Validate and detach a bounded observation without authenticating source."""
    if (type(event) is not dict or len(event) != len(_EVENT_FIELDS) or set(event) != _EVENT_FIELDS
            or event["schema"] != EVENT_SCHEMA
            or not _integer(event["generation"], 1, 8)
            or not _enum(event["scope"], {"file", "symbol"})
            or not _enum(event["representation"], {"original", "compact"})
            or not _enum(event["encoding"], {"utf-8", "base64"})
            or not _digest(event["source_identity_sha256"]) or not _digest(event["content_sha256"])
            or not _integer(event["content_bytes"], 0, MAX_CONTENT_BYTES)
            or (event["content_bytes"] == 0) != (event["content_sha256"] == _sha(b""))):
        raise ValueError("invalid_read_event")
    return dict(event)


def make_read_event(arguments, payload, success):
    """Observe a bound successful *delivered-payload shape*, or return None.

    The caller still owns transport delivery. File payloads omit scope; current
    Python symbol payloads declare ``python_definition``. Content hashes/lengths
    cover the UTF-8 returned string, including base64 text, not decoded binaries.
    """
    if success is not True or type(arguments) is not dict or type(payload) is not dict:
        return None
    generation, path = arguments.get("generation"), arguments.get("path")
    if (not _integer(generation, 1, 8) or type(payload.get("generation")) is not int
            or payload["generation"] != generation or not _text(path, 1024)
            or path != payload.get("path") or "\\" in path
            or len(path.split("/")) > 64 or any(part in {"", ".", "..", ".git"} for part in path.split("/"))
            or not _digest(payload.get("source_sha256")) or payload.get("proof_authority") is not False):
        return None
    representation, encoding = payload.get("representation"), payload.get("encoding")
    if (not _enum(representation, {"original", "compact"}) or not _enum(encoding, {"utf-8", "base64"})
            or "representation" in arguments and not _enum(arguments["representation"], {"original", "compact"})
            or arguments.get("representation") == "original" and representation != "original"):
        return None
    symbol_id = arguments.get("symbol_id")
    if "symbol_id" in arguments:
        if (not _text(symbol_id, 256) or payload.get("symbol_id") != symbol_id
                or payload.get("scope") != "python_definition" or encoding != "utf-8"):
            return None
        scope = "symbol"
    else:
        if "symbol_id" in payload or payload.get("scope", "file") != "file":
            return None
        scope = "file"
    content = payload.get("content")
    if type(content) is not str or len(content) > MAX_CONTENT_BYTES:
        return None
    try:
        raw = content.encode("utf-8")
        if len(raw) > MAX_CONTENT_BYTES:
            return None
        if encoding == "base64" and base64.b64encode(base64.b64decode(raw, validate=True)) != raw:
            return None
    except (UnicodeError, ValueError, binascii.Error):
        return None
    identity = dict(path=path, source_sha256=payload["source_sha256"], scope=scope, symbol_id=symbol_id)
    return validate_read_event(dict(schema=EVENT_SCHEMA, generation=generation, scope=scope,
        representation=representation, encoding=encoding, source_identity_sha256=_sha(_json(identity)),
        content_sha256=_sha(raw), content_bytes=len(raw)))


def _groups_totals(groups):
    if type(groups) is not list or len(groups) > MAX_READS:
        raise ValueError("invalid_read_groups")
    keys, identities, content_sizes, source_scopes, source_content, copies = set(), set(), {}, {}, {}, []
    result = dict.fromkeys(_TOTALS, 0)
    identified = compact = 0
    previous = None
    for group in groups:
        if (type(group) is not dict or len(group) != len(_GROUP_FIELDS) or set(group) != _GROUP_FIELDS
                or not _integer(group["read_count"], 1, MAX_READS)):
            raise ValueError("invalid_read_groups")
        event = validate_read_event({"schema": EVENT_SCHEMA, **{name: group[name] for name in _GROUP_KEYS}})
        key = tuple(event[name] for name in _GROUP_KEYS)
        if key in keys or previous is not None and key <= previous:
            raise ValueError("invalid_read_groups")
        keys.add(key)
        previous = key
        count, size, digest = group["read_count"], event["content_bytes"], event["content_sha256"]
        if (digest in content_sizes and content_sizes[digest] != size
                or event["source_identity_sha256"] in source_scopes
                    and source_scopes[event["source_identity_sha256"]] != event["scope"]):
            raise ValueError("inconsistent_read_identity")
        content_sizes[digest] = size
        source_scopes[event["source_identity_sha256"]] = event["scope"]
        source_key = (event["source_identity_sha256"], event["scope"], event["representation"], event["encoding"])
        if source_key in source_content and source_content[source_key] != (digest, size):
            raise ValueError("inconsistent_read_identity")
        source_content[source_key] = digest, size
        identified += count
        compact += count if event["representation"] == "compact" else 0
        if identified > MAX_READS:
            raise ValueError("invalid_read_groups")
        result["content_bytes"] += size * count
        result["same_generation_repeat_reads"] += count - 1
        result["same_generation_repeat_bytes"] += size * (count - 1)
        # A new generation group for the same source/content identity is one
        # cross-generation repeat. This classification needs no invented order.
        identity = key[1:]
        prefix = "cross_generation_repeat" if identity in identities else "first_identity"
        result[prefix + "_reads"] += 1
        result[prefix + "_bytes"] += size
        identities.add(identity)
        copies.append({name: group[name] for name in (*_GROUP_KEYS, "read_count")})
    return result, identified, compact, copies


def _audit_row(row, sequence):
    if type(row) is not dict or len(row) not in {len(_AUDIT_FIELDS), len(_AUDIT_FIELDS) + 1}:
        raise ValueError("invalid_read_audit_row")
    schema = row.get("schema")
    if not _enum(schema, {"repository-source-mcp-audit@1", "repository-source-mcp-audit@2", "repository-source-mcp-audit@3"}):
        raise ValueError("invalid_read_audit_row")
    extra = {"retrieval"} if schema.endswith("@2") else {"read_observation"} if schema.endswith("@3") else set()
    if (set(row) != _AUDIT_FIELDS | extra or type(row["sequence"]) is not int or row["sequence"] != sequence
            or not _enum(row["event"], {"initialize", "tool"})
            or row["event"] == "initialize" and row["tool"] is not None
            or row["event"] == "tool" and not _enum(row["tool"], _TOOLS)
            or any(type(row[name]) is not bool for name in (
                "success", "compact", "canonical_changed", "canonical_uncertain", "refreshed"))
            or not _integer(row["returned_bytes"], 0, MAX_PAYLOAD_BYTES)
            or schema.endswith("@2") and (row["tool"] not in {"repository_context_query", "repository_capsule_expand"}
                or type(row["retrieval"]) is not dict)
            or schema.endswith("@3") and (row["event"] != "tool" or row["tool"] != "repository_read")):
        raise ValueError("invalid_read_audit_row")


def summarize_read_observation(rows):
    """Summarize the bounded audit population; legacy-only input returns None.

    Every modern read call is represented, including failed or unbound calls
    with a null observation. Only unidentified *successful* reads make content
    totals incomplete. Operation byte totals cover inner JSON payloads for the
    entire audit population, not complete wire frames or decoded source bytes.
    """
    if type(rows) is not list or len(rows) > MAX_READS + 1:
        raise ValueError("invalid_read_audit_population")
    counts = dict.fromkeys((*_POPULATION, *_PAYLOAD_TOTALS), 0)
    groups = {}
    chain = _sha(b"")
    modern_seen = False
    tool_calls = initializations = compact_reads = returned_bytes = errors = 0
    for sequence, row in enumerate(rows, 1):
        _audit_row(row, sequence)
        tool_calls += row["event"] == "tool"
        initializations += row["event"] == "initialize"
        returned_bytes += row["returned_bytes"]
        errors += not row["success"]
        if tool_calls > MAX_READS or initializations > 1:
            raise ValueError("invalid_read_audit_population")
        if row["tool"] != "repository_read":
            counts["other_operation_payload_bytes"] += row["returned_bytes"]
            continue
        modern = row["schema"] == "repository-source-mcp-audit@3"
        modern_seen |= modern
        event = row["read_observation"] if modern else None
        if event is not None:
            event = validate_read_event(event)
            if (row["success"] is not True or (event["representation"] == "compact") is not row["compact"]
                    or event["content_bytes"] > row["returned_bytes"]):
                raise ValueError("inconsistent_read_event")
        counts["read_call_count"] += 1
        if row["success"]:
            counts["successful_reads"] += 1
            compact_reads += row["compact"]
            counts["successful_read_payload_bytes"] += row["returned_bytes"]
            if event is None:
                counts["unbound_successful_reads" if modern else "legacy_successful_reads"] += 1
            else:
                counts["identified_reads"] += 1
                key = tuple(event[name] for name in _GROUP_KEYS)
                groups[key] = groups.get(key, 0) + 1
        else:
            counts["read_error_count"] += 1
            counts["failed_read_payload_bytes"] += row["returned_bytes"]
        # Commit all read rows, including missing identities and errors. No
        # source/query bodies or raw private fields enter the commitment.
        chain = _sha(bytes.fromhex(chain) + _json({"sequence": sequence, "modern": modern,
            "success": row["success"], "compact": row["compact"], "returned_bytes": row["returned_bytes"],
            "read_observation": event}))
    if not modern_seen:
        return None
    grouped = [{**dict(zip(_GROUP_KEYS, key)), "read_count": count} for key, count in sorted(groups.items())]
    totals, _, _, _ = _groups_totals(grouped)
    complete = counts["legacy_successful_reads"] == counts["unbound_successful_reads"] == 0
    result = dict(schema=SUMMARY_SCHEMA, scope="prepared_repository_read_content_utf8",
        population_scope="supplied_validated_audit_rows",
        payload_byte_scope="inner_payload_json_utf8", complete=complete, event_chain_sha256=chain,
        groups=grouped, **counts, **{"known_" + key: value for key, value in totals.items()},
        **{key: value if complete else None for key, value in totals.items()},
        **dict.fromkeys(_UNKNOWN), **dict.fromkeys(_FALSE, False))
    return validate_read_observation(result, successful_reads=counts["successful_reads"],
        compact_reads=compact_reads, returned_bytes=returned_bytes, tool_calls=tool_calls, errors=errors)


def validate_read_observation(value, *, successful_reads, compact_reads, returned_bytes, tool_calls, errors):
    """Recompute grouped arithmetic and join parent counters; return a deep copy.

    This validates reported observations, not independent source provenance or
    the event-chain preimage. Known partial diagnostics never qualify full
    population totals when any successful read has no retained identity.
    """
    if (not _integer(tool_calls, 0, MAX_READS) or not _integer(successful_reads, 0, tool_calls)
            or not _integer(errors, 0, tool_calls + 1)
            or not _integer(compact_reads, 0, successful_reads)
            or not _integer(returned_bytes, 0, (MAX_READS + 1) * MAX_PAYLOAD_BYTES)
            or type(value) is not dict or len(value) != len(_SUMMARY_FIELDS) or set(value) != _SUMMARY_FIELDS
            or value["schema"] != SUMMARY_SCHEMA or value["scope"] != "prepared_repository_read_content_utf8"
            or value["population_scope"] != "supplied_validated_audit_rows"
            or value["payload_byte_scope"] != "inner_payload_json_utf8"
            or type(value["complete"]) is not bool or not _digest(value["event_chain_sha256"])
            or any(value[key] is not None for key in _UNKNOWN)
            or any(value[key] is not False for key in _FALSE)
            or any(not _integer(value[key], 0, MAX_READS) for key in _POPULATION)
            or any(not _integer(value[key], 0, (MAX_READS + 1) * MAX_PAYLOAD_BYTES) for key in _PAYLOAD_TOTALS)):
        raise ValueError("invalid_read_observation")
    totals, identified, compact, groups = _groups_totals(value["groups"])
    complete = value["legacy_successful_reads"] == value["unbound_successful_reads"] == 0
    if (value["successful_reads"] != successful_reads or value["identified_reads"] != identified
            or identified + value["legacy_successful_reads"] + value["unbound_successful_reads"] != successful_reads
            or value["read_call_count"] != successful_reads + value["read_error_count"]
            or value["read_error_count"] > errors
            or not 1 <= value["read_call_count"] <= tool_calls
            or compact > compact_reads or identified - compact > successful_reads - compact_reads
            or value["complete"] is not complete
            or sum(value[key] for key in _PAYLOAD_TOTALS) != returned_bytes
            or value["successful_read_payload_bytes"] > successful_reads * MAX_PAYLOAD_BYTES
            or value["failed_read_payload_bytes"] > value["read_error_count"] * MAX_PAYLOAD_BYTES
            or value["other_operation_payload_bytes"] > (tool_calls - value["read_call_count"] + 1) * MAX_PAYLOAD_BYTES
            or totals["content_bytes"] > value["successful_read_payload_bytes"]):
        raise ValueError("inconsistent_read_observation")
    for name, expected in totals.items():
        known = value["known_" + name]
        if type(known) is not int or known != expected:
            raise ValueError("inconsistent_read_observation")
        if complete:
            if type(value[name]) is not int or value[name] != expected:
                raise ValueError("inconsistent_read_observation")
        elif value[name] is not None:
            raise ValueError("incomplete_read_observation")
    if len(_json(value)) > MAX_SUMMARY_BYTES:
        raise ValueError("read_observation_limit")
    return {**value, "groups": groups}
