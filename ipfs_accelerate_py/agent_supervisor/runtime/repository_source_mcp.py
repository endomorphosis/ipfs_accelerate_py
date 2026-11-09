"""Bounded stdio MCP transport for source-current repository sessions.

The server has no provider or network client. One private, hash-bound config
selects the canonical checkout. Requests cannot select a different root or run
code. Source output appears only in tool replies; the audit contains counters.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_REQUESTS = 512
MAX_AUDIT_BYTES = 512 * 1024
PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
TOOL_NAMES = (
    "repository_status", "repository_list", "repository_symbols",
    "repository_read", "repository_replace_symbol", "repository_refresh",
    "repository_cache_inventory",
    "repository_context_query", "repository_capsule_expand",
)


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError("duplicate_field")
        value[key] = item
    return value


def _loads(raw):
    def invalid(_):
        raise ValueError("nonfinite_json")
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)
    pending = [(value, 0)]
    nodes = 0
    while pending:
        node, depth = pending.pop()
        nodes += 1
        if depth > 16 or nodes > 8192:
            raise ValueError("json_limit")
        if isinstance(node, dict):
            pending.extend((item, depth + 1) for item in node.values())
        elif isinstance(node, list):
            pending.extend((item, depth + 1) for item in node)
    return value


def _private_directory(path):
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("invalid_private_directory")
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError("invalid_private_directory")


def load_config(path: Path, expected_sha256: str) -> dict:
    """Read one bounded regular config without following a final symlink."""
    _private_directory(path.parent)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_size > 16384):
            raise ValueError("invalid_config_file")
        raw = stream.read(16385)
    if (len(raw) > 16384 or type(expected_sha256) is not str
            or re.fullmatch(r"[a-f0-9]{64}", expected_sha256) is None
            or hashlib.sha256(raw).hexdigest() != expected_sha256):
        raise ValueError("config_hash_mismatch")
    config = _loads(raw)
    fields = {"schema", "workspace", "state_dir", "audit_store", "representation"}
    if (type(config) is not dict or not (
            set(config) == fields and config.get("schema") == "repository-source-mcp-config@1"
            or set(config) == fields | {"tokenizer"} and config.get("schema") == "repository-source-mcp-config@2")
            or type(config["representation"]) is not str
            or config["representation"] not in {"original", "compact"}):
        raise ValueError("invalid_config")
    if "tokenizer" in config:
        from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_token_budget import validate_tokenizer_binding
        validate_tokenizer_binding(config["tokenizer"])
    paths = {}
    for key in ("workspace", "state_dir", "audit_store"):
        value = config[key]
        if (type(value) is not str or not 1 <= len(value.encode()) <= 4096
                or any(ord(c) < 32 for c in value)):
            raise ValueError("invalid_config_path")
        current = Path(value)
        if not current.is_absolute() or ".." in current.parts or str(current) != value:
            raise ValueError("invalid_config_path")
        paths[key] = current
    workspace, state_dir, audit = (paths[key] for key in ("workspace", "state_dir", "audit_store"))
    if (workspace.resolve(strict=True) != workspace or not workspace.is_dir()
            or state_dir.parent != path.parent or audit.parent != path.parent
            or audit.name != "operations.jsonl" or state_dir.name != "state"
            or path.parent.is_relative_to(workspace) or workspace.is_relative_to(path.parent)):
        raise ValueError("invalid_config_scope")
    _private_directory(state_dir)
    return config


def _tool(name, description, properties, required, *, read_only=True):
    return {"name": name, "description": description,
            "inputSchema": {"type": "object", "properties": properties,
                            "required": required, "additionalProperties": False},
            "annotations": {"readOnlyHint": read_only, "destructiveHint": not read_only,
                            "openWorldHint": False}}


GENERATION = {"type": "integer", "minimum": 1}
PATH = {"type": "string", "minLength": 1, "maxLength": 1024}
LIMIT = {"type": "integer", "minimum": 1, "maximum": 100}
CONTEXT_BYTES = {"type": "integer", "minimum": 1024, "maximum": 65536}
CID = {"type": "string", "minLength": 1, "maxLength": 256}
TOOLS = [
    _tool("repository_status", "Get the current generation, index health and cache counts/digest. No source content or full cache inventory.", {}, []),
    _tool("repository_cache_inventory", "Read a page of retained opaque cache metadata using the current status digest; caches are not natively indexed.",
          {"generation": GENERATION, "expected_inventory_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
           "offset": {"type": "integer", "minimum": 0}, "limit": LIMIT}, ["generation", "expected_inventory_sha256"]),
    _tool("repository_list", "List a bounded page of files in a source-current generation.",
          {"generation": GENERATION, "offset": {"type": "integer", "minimum": 0}, "limit": LIMIT}, ["generation"]),
    _tool("repository_symbols", "Find native datasets symbols in the current generation; no proof authority.",
          {"generation": GENERATION, "path": PATH, "query": {"type": "string", "maxLength": 256}, "limit": LIMIT}, ["generation"]),
    _tool("repository_context_query", "Find structural capsule references by symbol/path terms in the current generation. Results are advisory; no match does not mean an intent is satisfied. Expand selected references for details.",
          {"generation": GENERATION, "query": {"type": "string", "minLength": 1, "maxLength": 4096},
           "limit": {"type": "integer", "minimum": 1, "maximum": 32}, "max_bytes": CONTEXT_BYTES},
          ["generation", "query"]),
    _tool("repository_capsule_expand", "Expand selected native capsules from a query's semantic root. Returns structural facts, confidence and limitations. Read current source before editing; capsules do not prove behavior.",
          {"generation": GENERATION, "semantic_root_cid": CID,
           "stable_symbol_ids": {"type": "array", "items": CID, "minItems": 1, "maxItems": 32, "uniqueItems": True},
           "max_bytes": CONTEXT_BYTES}, ["generation", "semantic_root_cid", "stable_symbol_ids"]),
    _tool("repository_read", "Read current source with its canonical file SHA256. Optional symbol_id selects a supported Python definition and its verified local edit name. For any UTF-8 file, use start_line and end_line together with representation original for an exact inclusive physical-line range (at most 4096 lines/64 KiB); ranges omit other context and provide no symbol edit handle. Do not combine a range with symbol_id. Omit both selectors for the full file. Definitions are dedented and omit enclosing/import context. Compact omits comments/formatting; a bound local tokenizer selects it only for a smaller complete reply. Request original source for missing context.",
          {"generation": GENERATION, "path": PATH, "symbol_id": CID,
           "start_line": {"type": "integer", "minimum": 1, "maximum": 2**31 - 1},
           "end_line": {"type": "integer", "minimum": 1, "maximum": 2**31 - 1},
           "representation": {"type": "string", "enum": ["original", "compact"]}}, ["generation", "path"]),
    _tool("repository_replace_symbol", "Replace one Python definition using the read's source_sha256, verified local symbol and returned representation. Legacy local names such as Class.method are accepted; native CIDs and module-prefixed index names are not edit names. Refreshes canonical checkout/index; check refresh_required and run task tests. Comment-only changes require a direct original-source edit.",
          {"generation": GENERATION, "path": PATH, "symbol": {"type": "string", "minLength": 1, "maxLength": 1024},
           "expected_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
           "replacement": {"type": "string", "maxLength": 262144},
           "representation": {"type": "string", "enum": ["original", "compact"]}},
          ["generation", "path", "symbol", "expected_sha256", "replacement"], read_only=False),
    _tool("repository_refresh", "Capture external checkout edits and rebuild the native index. Invalidates older generations; bounded to eight captures per invocation.",
          {"generation": GENERATION}, ["generation"], read_only=False),
]


def _arguments(name, value):
    schema = next(tool["inputSchema"] for tool in TOOLS if tool["name"] == name)
    if (type(value) is not dict or set(value) - set(schema["properties"])
            or set(schema["required"]) - set(value)):
        raise ValueError("invalid_arguments")
    for key, item in value.items():
        rule = schema["properties"][key]
        if rule["type"] == "integer":
            if type(item) is not int or not rule.get("minimum", 0) <= item <= rule.get("maximum", 2**31 - 1):
                raise ValueError("invalid_arguments")
        elif rule["type"] == "array":
            if (type(item) is not list or not rule["minItems"] <= len(item) <= rule["maxItems"]
                    or any(type(member) is not str or not rule["items"]["minLength"] <= len(member.encode()) <= rule["items"]["maxLength"]
                           for member in item)
                    or len(set(item)) != len(item)):
                raise ValueError("invalid_arguments")
        elif type(item) is not str or not rule.get("minLength", 0) <= len(item.encode()) <= rule.get("maxLength", 4096):
            raise ValueError("invalid_arguments")
        if "enum" in rule and item not in rule["enum"]:
            raise ValueError("invalid_arguments")
        if "pattern" in rule and re.fullmatch(rule["pattern"], item) is None:
            raise ValueError("invalid_arguments")
    return value


def _delivery_payload(name, value):
    """Keep full binding receipts in status, not repeated beside every read.

    The transport is already scoped to one session. Generation and canonical
    source hashes retain the edit/freshness contract without sending SQL heads,
    native CIDs and duplicate representation hashes for every returned file.
    """
    if name == "repository_read":
        result = {"generation": value["generation"], "path": value["path"],
                "source_sha256": value["original_sha256"],
                **{key: value[key] for key in ("representation", "encoding", "content", "proof_authority")}}
        if "symbol_id" in value:
            result.update({key: value[key] for key in ("symbol_id", "symbol", "edit_supported", "scope")})
        if value.get("scope") == "line_range":
            result.update({key: value[key] for key in (
                "scope", "start_line", "end_line", "total_lines", "omitted_lines")})
        return result
    if name == "repository_list":
        entries = []
        for row in value["entries"]:
            item = {"path": row["path"], "kind": row["kind"]}
            if row["kind"] == "file":
                item.update(original_bytes=row["original_bytes"], compact_bytes=row["compact_bytes"])
            entries.append(item)
        return {"generation": value["generation"], "entries": entries,
                "total": value["total"], "next_offset": value["next_offset"]}
    if name == "repository_symbols":
        return {"generation": value["generation"], "symbols": [
            {key: row[key] for key in ("symbol_id", "path", "name", "kind", "edit_handle_available")}
            for row in value["symbols"]], "proof_authority": False}
    if name == "repository_refresh":
        return {key: value[key] for key in ("generation", "source_current", "refresh_required",
                                           "source_tree_sha256", "proof_authority")}
    if name == "repository_replace_symbol":
        result = {key: value[key] for key in ("generation", "canonical_changed", "refresh_required", "proof_authority")}
        if "error" in value:
            result["error"] = value["error"]
        if "edit" in value:
            result.update(path=value["edit"]["path"], source_sha256=value["edit"]["result_sha256"])
        return result
    return value


AUDIT_FIELDS = {"schema", "sequence", "event", "tool", "success", "returned_bytes",
                "compact", "canonical_changed", "canonical_uncertain", "refreshed"}
RETRIEVAL_TOOLS = {"repository_context_query", "repository_capsule_expand"}
RETRIEVAL_COUNTERS = ("query_calls", "successful_queries", "empty_queries", "nonempty_queries",
    "candidates_returned", "expansion_calls", "successful_expansions", "empty_expansions",
    "nonempty_expansions", "capsules_returned", "errors", "stale_generation_errors",
    "stale_source_errors", "root_mismatch_errors", "legacy_retrieval_calls", "unbound_successes",
    "generation_bindings_observed")
RETRIEVAL_FALSE = ("initial_context_link_verified", "omitted_capsule_recovery_verified",
    "model_attention_verified", "token_savings_measured", "proof_authority", "completion_authority")
_RETRIEVAL_ERRORS = {None, "stale_generation", "stale_source", "root_mismatch", "other"}
READ_COUNTERS = ("read_calls", "successful_reads", "failed_reads", "full_file_reads",
    "definition_reads", "line_range_reads", "original_reads", "compact_reads",
    "exposed_content_utf8_bytes", "line_range_selected_lines", "line_range_total_lines",
    "line_range_omitted_lines", "generation_bindings_observed", "unbound_successes",
    "unbound_compact_reads", "legacy_read_calls", "legacy_successful_reads", "legacy_compact_reads")
READ_FALSE = ("source_authority", "proof_authority", "execution_authority", "completion_authority",
    "model_attention_verified", "source_displacement_verified", "provider_usage_measured", "token_savings_measured")
_READ_EVENT_FIELDS = {"requested_generation", "binding_sha256", "scope", "representation",
    "content_utf8_bytes", "start_line", "end_line", "total_lines", "omitted_lines"}
_READ_RANGE_LINES = 4096
_READ_RANGE_BYTES = 65536
_READ_LINE_NUMBER = 2**31 - 1


def _digest(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _retrieval_event(tool, arguments, payload, success):
    """Observe prepared response identities without retaining query or source text."""
    generation = arguments.get("generation") if type(arguments) is dict else None
    if type(generation) is not int or not 1 <= generation <= 8:
        generation = None
    event = dict(requested_generation=generation, binding_sha256=None, returned_items=None, error=None)
    if not success:
        error = payload.get("error")
        event["error"] = (error if error == "stale_generation" else "stale_source"
            if error in {"stale_source", "refresh_required"} else "root_mismatch"
            if error == "capsule_root_differs" else "other")
        return event
    items = payload.get("candidates" if tool == "repository_context_query" else "capsules")
    fields = ("generation", "semantic_root_cid", "semantic_state_cid", "source_tree_sha256")
    if (generation is not None and type(payload.get("generation")) is int
            and payload["generation"] == generation and type(items) is list and len(items) <= 32
            and _digest(payload.get("source_tree_sha256"))
            and all(type(payload.get(key)) is str and 0 < len(payload[key]) <= 4096
                    for key in ("semantic_root_cid", "semantic_state_cid"))):
        event.update(returned_items=len(items), binding_sha256=hashlib.sha256(
            _json({key: payload[key] for key in fields})).hexdigest())
    return event


def _validate_retrieval_event(event, success):
    if (type(event) is not dict or set(event) != {
            "requested_generation", "binding_sha256", "returned_items", "error"}
            or not (event["requested_generation"] is None or type(event["requested_generation"]) is int
                    and 1 <= event["requested_generation"] <= 8)
            or event["error"] not in _RETRIEVAL_ERRORS):
        raise ValueError("invalid_retrieval_audit")
    bound = event["binding_sha256"] is not None
    if (bound and (not _digest(event["binding_sha256"]) or event["requested_generation"] is None
            or type(event["returned_items"]) is not int or not 0 <= event["returned_items"] <= 32)
            or not bound and event["returned_items"] is not None
            or success and event["error"] is not None
            or not success and (bound or event["error"] is None)):
        raise ValueError("invalid_retrieval_audit")


def validate_retrieval_observation(value, *, tool_calls, errors):
    """Validate bounded arithmetic, not independent provenance or model attention."""
    fields = {"schema", "complete", "event_chain_sha256", *RETRIEVAL_COUNTERS, *RETRIEVAL_FALSE}
    if (type(value) is not dict or set(value) != fields
            or value["schema"] != "repository-source-retrieval-observation@1"
            or type(value["complete"]) is not bool or not _digest(value["event_chain_sha256"])
            or any(value[key] is not False for key in RETRIEVAL_FALSE)
            or any(type(value[key]) is not int or not 0 <= value[key] <= MAX_REQUESTS * 32
                   for key in RETRIEVAL_COUNTERS)
            or type(tool_calls) is not int or not 0 <= tool_calls <= MAX_REQUESTS
            or type(errors) is not int or not 0 <= errors <= tool_calls + 1):
        raise ValueError("invalid_retrieval_observation")
    q, e = value["query_calls"], value["expansion_calls"]
    successes = value["successful_queries"] + value["successful_expansions"]
    if (q + e + value["legacy_retrieval_calls"] > tool_calls or value["errors"] > errors
            or successes > tool_calls - max(0, errors - 1)
            or successes + value["errors"] != q + e
            or value["successful_queries"] > q or value["successful_expansions"] > e
            or value["empty_queries"] + value["nonempty_queries"] > value["successful_queries"]
            or value["empty_expansions"] + value["nonempty_expansions"] > value["successful_expansions"]
            or sum(value[key] for key in ("empty_queries", "nonempty_queries", "empty_expansions",
                "nonempty_expansions", "unbound_successes")) != successes
            or value["generation_bindings_observed"] + value["unbound_successes"] != successes
            or not value["nonempty_queries"] <= value["candidates_returned"] <= 32 * value["nonempty_queries"]
            or not value["nonempty_expansions"] <= value["capsules_returned"] <= 32 * value["nonempty_expansions"]
            or sum(value[key] for key in ("stale_generation_errors", "stale_source_errors", "root_mismatch_errors")) > value["errors"]
            or value["complete"] is not (value["legacy_retrieval_calls"] == value["unbound_successes"] == 0)):
        raise ValueError("invalid_retrieval_observation")
    return dict(value)


def _read_integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def _read_event(arguments, payload, success):
    """Observe one prepared source slice without retaining its text or path.

    Counts describe the UTF-8 content string inside the reply, including base64
    text for binary whole-file reads. They exclude JSON escaping, envelopes,
    requests, schemas and guidance, and do not establish provider ingestion.
    """
    generation = arguments.get("generation") if type(arguments) is dict else None
    if not _read_integer(generation, 1, 8):
        generation = None
    event = dict.fromkeys(_READ_EVENT_FIELDS)
    event["requested_generation"] = generation
    if not success or generation is None or type(payload) is not dict:
        return event
    path, content = payload.get("path"), payload.get("content")
    representation = payload.get("representation")
    if (not _read_integer(payload.get("generation"), 1, 8) or payload["generation"] != generation
            or type(path) is not str or not 0 < len(path.encode("utf-8")) <= 1024
            or path != arguments.get("path") or not _digest(payload.get("source_sha256"))
            or type(representation) is not str or representation not in {"original", "compact"}
            or type(payload.get("encoding")) is not str or payload["encoding"] not in {"utf-8", "base64"}
            or type(content) is not str or payload.get("proof_authority") is not False):
        return event
    content_bytes = len(content.encode("utf-8"))
    if content_bytes > MAX_RESPONSE_BYTES:
        return event
    scope = payload.get("scope")
    line_fields = {}
    identity = {key: payload[key] for key in ("generation", "path", "source_sha256", "representation", "encoding")}
    if scope == "line_range":
        line_fields = {key: payload.get(key) for key in ("start_line", "end_line", "total_lines", "omitted_lines")}
        start, end, total, omitted = (line_fields[key] for key in ("start_line", "end_line", "total_lines", "omitted_lines"))
        if (representation != "original" or payload["encoding"] != "utf-8"
                or any(not _read_integer(value, 1, _READ_LINE_NUMBER) for value in (start, end, total))
                or not start <= end <= total or end - start + 1 > _READ_RANGE_LINES
                or not _read_integer(omitted, 0, _READ_LINE_NUMBER) or omitted != total - (end - start + 1)
                or not end - start + 1 <= content_bytes <= _READ_RANGE_BYTES
                or content.count("\n") + int(not content.endswith("\n")) != end - start + 1
                or end < total and not content.endswith("\n")
                or not _read_integer(arguments.get("start_line"), 1, _READ_LINE_NUMBER)
                or not _read_integer(arguments.get("end_line"), 1, _READ_LINE_NUMBER)
                or arguments.get("start_line") != start or arguments.get("end_line") != end
                or "symbol_id" in arguments or "symbol_id" in payload or "symbol" in payload):
            return event
        identity.update(line_fields)
    elif scope == "python_definition":
        symbol_id = payload.get("symbol_id")
        if (type(symbol_id) is not str or not 0 < len(symbol_id.encode("utf-8")) <= 256
                or symbol_id != arguments.get("symbol_id") or type(payload.get("symbol")) is not str
                or not 0 < len(payload["symbol"].encode("utf-8")) <= 1024
                or type(payload.get("edit_supported")) is not bool or payload["encoding"] != "utf-8"
                or content_bytes == 0 or any(key in arguments for key in ("start_line", "end_line"))):
            return event
        scope = "definition"
        identity["symbol_id"] = symbol_id
    elif scope is None and not any(key in arguments for key in ("symbol_id", "start_line", "end_line")):
        scope = "full_file"
    else:
        return event
    identity["scope"] = scope
    event.update(binding_sha256=hashlib.sha256(_json(identity)).hexdigest(), scope=scope,
        representation=representation, content_utf8_bytes=content_bytes, **line_fields)
    return event


def _validate_read_event(event, success, compact):
    if (type(event) is not dict or set(event) != _READ_EVENT_FIELDS
            or not (event["requested_generation"] is None or _read_integer(event["requested_generation"], 1, 8))):
        raise ValueError("invalid_read_audit")
    bound = event["binding_sha256"] is not None
    if not bound:
        if any(event[key] is not None for key in _READ_EVENT_FIELDS - {"requested_generation"}):
            raise ValueError("invalid_read_audit")
        if not success and compact:
            raise ValueError("invalid_read_audit")
        return
    if (not success or not _digest(event["binding_sha256"]) or event["requested_generation"] is None
            or type(event["scope"]) is not str or event["scope"] not in {"full_file", "definition", "line_range"}
            or type(event["representation"]) is not str or event["representation"] not in {"original", "compact"}
            or compact is not (event["representation"] == "compact")
            or not _read_integer(event["content_utf8_bytes"], 0, MAX_RESPONSE_BYTES)):
        raise ValueError("invalid_read_audit")
    if event["scope"] == "line_range":
        start, end, total, omitted = (event[key] for key in ("start_line", "end_line", "total_lines", "omitted_lines"))
        if (any(not _read_integer(value, 1, _READ_LINE_NUMBER) for value in (start, end, total))
                or not start <= end <= total or end - start + 1 > _READ_RANGE_LINES
                or not _read_integer(omitted, 0, _READ_LINE_NUMBER) or omitted != total - (end - start + 1)
                or event["representation"] != "original"
                or not end - start + 1 <= event["content_utf8_bytes"] <= _READ_RANGE_BYTES):
            raise ValueError("invalid_read_audit")
    elif (any(event[key] is not None for key in ("start_line", "end_line", "total_lines", "omitted_lines"))
            or event["scope"] == "definition" and event["content_utf8_bytes"] == 0):
        raise ValueError("invalid_read_audit")


def validate_read_observation(value, *, tool_calls, errors, successful_reads, compact_reads):
    """Validate disjoint bounded read populations, never delivery or savings."""
    fields = {"schema", "complete", "event_chain_sha256", *READ_COUNTERS, *READ_FALSE}
    if (type(value) is not dict or set(value) != fields
            or value["schema"] != "repository-source-read-observation@1"
            or type(value["complete"]) is not bool or not _digest(value["event_chain_sha256"])
            or any(value[key] is not False for key in READ_FALSE)
            or not _read_integer(tool_calls, 0, MAX_REQUESTS)
            or not _read_integer(errors, 0, tool_calls + 1)
            or not _read_integer(successful_reads, 0, tool_calls)
            or not _read_integer(compact_reads, 0, successful_reads)):
        raise ValueError("invalid_read_observation")
    bounds = {"exposed_content_utf8_bytes": MAX_REQUESTS * MAX_RESPONSE_BYTES,
        "line_range_selected_lines": MAX_REQUESTS * _READ_RANGE_LINES,
        "line_range_total_lines": MAX_REQUESTS * _READ_LINE_NUMBER,
        "line_range_omitted_lines": MAX_REQUESTS * _READ_LINE_NUMBER}
    if any(not _read_integer(value[key], 0, bounds.get(key, MAX_REQUESTS)) for key in READ_COUNTERS):
        raise ValueError("invalid_read_observation")
    bound = value["generation_bindings_observed"]
    ranges = value["line_range_reads"]
    if (not 1 <= value["read_calls"] == value["successful_reads"] + value["failed_reads"]
            or value["read_calls"] + value["legacy_read_calls"] > tool_calls
            or value["successful_reads"] + value["legacy_successful_reads"] != successful_reads
            or value["legacy_successful_reads"] > value["legacy_read_calls"]
            or value["legacy_compact_reads"] > value["legacy_successful_reads"]
            or value["failed_reads"] + value["legacy_read_calls"] - value["legacy_successful_reads"] > errors
            or successful_reads > tool_calls - max(0, errors - 1)
            or bound + value["unbound_successes"] != value["successful_reads"]
            or value["unbound_compact_reads"] > value["unbound_successes"]
            or value["compact_reads"] + value["unbound_compact_reads"] + value["legacy_compact_reads"] != compact_reads
            or value["full_file_reads"] + value["definition_reads"] + ranges != bound
            or value["original_reads"] + value["compact_reads"] != bound
            or ranges > value["original_reads"]
            or not value["definition_reads"] + value["line_range_selected_lines"] <= value["exposed_content_utf8_bytes"] <= bound * MAX_RESPONSE_BYTES
            or not ranges <= value["line_range_selected_lines"] <= ranges * _READ_RANGE_LINES
            or not value["line_range_selected_lines"] <= value["line_range_total_lines"] <= ranges * _READ_LINE_NUMBER
            or value["line_range_omitted_lines"] != value["line_range_total_lines"] - value["line_range_selected_lines"]
            or value["complete"] is not (value["legacy_read_calls"] == value["unbound_successes"] == 0)):
        raise ValueError("invalid_read_observation")
    return dict(value)


class _Audit:
    def __init__(self, path):
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.sequence = 0

    def record(self, event, *, tool=None, success=True, returned_bytes=0,
               compact=False, canonical_changed=False, canonical_uncertain=False, refreshed=False,
               retrieval=None, read=None):
        self.sequence += 1
        row = {"schema": "repository-source-mcp-audit@1", "sequence": self.sequence,
               "event": event, "tool": tool, "success": success, "returned_bytes": returned_bytes,
               "compact": compact, "canonical_changed": canonical_changed,
               "canonical_uncertain": canonical_uncertain, "refreshed": refreshed}
        if retrieval is not None:
            _validate_retrieval_event(retrieval, success)
            row.update(schema="repository-source-mcp-audit@2", retrieval=retrieval)
        if read is not None:
            if retrieval is not None or event != "tool" or tool != "repository_read":
                raise ValueError("invalid_read_audit")
            _validate_read_event(read, success, compact)
            if read["content_utf8_bytes"] is not None and read["content_utf8_bytes"] > returned_bytes:
                raise ValueError("invalid_read_audit")
            row.update(schema="repository-source-mcp-audit@3", read=read)
        raw = _json(row) + b"\n"
        if os.fstat(self.fd).st_size + len(raw) > MAX_AUDIT_BYTES:
            raise ValueError("audit_limit")
        view = memoryview(raw)
        while view:
            size = os.write(self.fd, view)
            if size <= 0:
                raise OSError("audit_write_failed")
            view = view[size:]

    def close(self):
        os.close(self.fd)


def summarize_repository_audit(path: str | Path) -> dict:
    result = {"status": "not_observed", "initializations": 0, "tool_calls": 0,
              "successful_reads": 0, "compact_reads": 0, "canonical_edits": 0,
              "uncertain_edits": 0, "refreshes": 0, "errors": 0, "returned_bytes": 0}
    try:
        path = Path(path)
        _private_directory(path.parent)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600
                    or info.st_size > MAX_AUDIT_BYTES):
                raise ValueError("invalid_audit")
            raw = stream.read(MAX_AUDIT_BYTES + 1)
        if len(raw) > MAX_AUDIT_BYTES or (raw and not raw.endswith(b"\n")):
            raise ValueError("invalid_audit")
        rows = raw.splitlines()
        if len(rows) > MAX_REQUESTS + 1:
            raise ValueError("invalid_audit")
        retrieval = dict(schema="repository-source-retrieval-observation@1", complete=True,
            event_chain_sha256=hashlib.sha256(b"").hexdigest(),
            **dict.fromkeys(RETRIEVAL_COUNTERS, 0), **dict.fromkeys(RETRIEVAL_FALSE, False))
        retrieval_seen = False
        reads = dict(schema="repository-source-read-observation@1", complete=True,
            event_chain_sha256=hashlib.sha256(b"").hexdigest(),
            **dict.fromkeys(READ_COUNTERS, 0), **dict.fromkeys(READ_FALSE, False))
        read_seen = False
        for sequence, line in enumerate(rows, 1):
            row = _loads(line)
            modern = type(row) is dict and row.get("schema") == "repository-source-mcp-audit@2"
            modern_read = type(row) is dict and row.get("schema") == "repository-source-mcp-audit@3"
            extra = {"retrieval"} if modern else {"read"} if modern_read else set()
            if (type(row) is not dict or set(row) != AUDIT_FIELDS | extra
                    or row["schema"] not in {"repository-source-mcp-audit@1", "repository-source-mcp-audit@2", "repository-source-mcp-audit@3"}
                    or type(row["sequence"]) is not int or row["sequence"] != sequence
                    or row["event"] not in {"initialize", "tool"}
                    or (row["event"] == "initialize" and row["tool"] is not None)
                    or (row["event"] == "tool" and row["tool"] not in TOOL_NAMES)
                    or any(type(row[key]) is not bool for key in ("success", "compact", "canonical_changed", "canonical_uncertain", "refreshed"))
                    or type(row["returned_bytes"]) is not int
                    or not 0 <= row["returned_bytes"] <= MAX_RESPONSE_BYTES):
                raise ValueError("invalid_audit")
            if modern:
                if row["event"] != "tool" or row["tool"] not in RETRIEVAL_TOOLS:
                    raise ValueError("invalid_retrieval_audit")
                event = row["retrieval"]
                _validate_retrieval_event(event, row["success"])
                retrieval_seen = True
                kind = "queries" if row["tool"] == "repository_context_query" else "expansions"
                retrieval["query_calls" if kind == "queries" else "expansion_calls"] += 1
                retrieval["event_chain_sha256"] = hashlib.sha256(bytes.fromhex(
                    retrieval["event_chain_sha256"]) + _json(row)).hexdigest()
                if row["success"]:
                    retrieval["successful_" + kind] += 1
                    if event["returned_items"] is None:
                        retrieval["unbound_successes"] += 1
                    else:
                        retrieval["generation_bindings_observed"] += 1
                        retrieval[("nonempty_" if event["returned_items"] else "empty_") + kind] += 1
                        retrieval["candidates_returned" if kind == "queries" else "capsules_returned"] += event["returned_items"]
                else:
                    retrieval["errors"] += 1
                    counter = {"stale_generation": "stale_generation_errors", "stale_source": "stale_source_errors",
                               "root_mismatch": "root_mismatch_errors"}.get(event["error"])
                    if counter:
                        retrieval[counter] += 1
            elif row["tool"] in RETRIEVAL_TOOLS:
                retrieval["legacy_retrieval_calls"] += 1
            if modern_read:
                if row["event"] != "tool" or row["tool"] != "repository_read":
                    raise ValueError("invalid_read_audit")
                event = row["read"]
                _validate_read_event(event, row["success"], row["compact"])
                if event["content_utf8_bytes"] is not None and event["content_utf8_bytes"] > row["returned_bytes"]:
                    raise ValueError("invalid_read_audit")
                read_seen = True
                reads["read_calls"] += 1
                reads["event_chain_sha256"] = hashlib.sha256(bytes.fromhex(
                    reads["event_chain_sha256"]) + _json(row)).hexdigest()
                if row["success"]:
                    reads["successful_reads"] += 1
                    if event["binding_sha256"] is None:
                        reads["unbound_successes"] += 1
                        reads["unbound_compact_reads"] += row["compact"]
                    else:
                        reads["generation_bindings_observed"] += 1
                        reads[event["scope"] + "_reads"] += 1
                        reads[event["representation"] + "_reads"] += 1
                        reads["exposed_content_utf8_bytes"] += event["content_utf8_bytes"]
                        if event["scope"] == "line_range":
                            reads["line_range_selected_lines"] += event["end_line"] - event["start_line"] + 1
                            reads["line_range_total_lines"] += event["total_lines"]
                            reads["line_range_omitted_lines"] += event["omitted_lines"]
                else:
                    reads["failed_reads"] += 1
            elif row["tool"] == "repository_read":
                reads["legacy_read_calls"] += 1
                reads["legacy_successful_reads"] += row["success"]
                reads["legacy_compact_reads"] += row["success"] and row["compact"]
            result["initializations"] += row["event"] == "initialize" and row["success"]
            result["tool_calls"] += row["event"] == "tool"
            result["successful_reads"] += row["tool"] == "repository_read" and row["success"]
            result["compact_reads"] += row["tool"] == "repository_read" and row["success"] and row["compact"]
            result["canonical_edits"] += row["canonical_changed"]
            result["uncertain_edits"] += row["canonical_uncertain"]
            result["refreshes"] += row["refreshed"]
            result["errors"] += not row["success"]
            result["returned_bytes"] += row["returned_bytes"]
        if rows:
            result["status"] = "observed"
        if retrieval_seen:
            retrieval["complete"] = retrieval["legacy_retrieval_calls"] == retrieval["unbound_successes"] == 0
            result["retrieval_observation"] = validate_retrieval_observation(
                retrieval, tool_calls=result["tool_calls"], errors=result["errors"])
        if read_seen:
            reads["complete"] = reads["legacy_read_calls"] == reads["unbound_successes"] == 0
            result["read_observation"] = validate_read_observation(reads,
                tool_calls=result["tool_calls"], errors=result["errors"],
                successful_reads=result["successful_reads"], compact_reads=result["compact_reads"])
        return result
    except FileNotFoundError:
        return result
    except (ValueError, OSError, RecursionError, UnicodeError, TypeError):
        return {"status": "unavailable", **{key: None for key in result if key != "status"}}


class RepositoryMCPServer:
    def __init__(self, config, *, session_factory=None):
        self.config = config
        self.session_factory = session_factory
        self.session = None
        self.ready = False
        self.initialization_attempted = False
        self.tokenizer = None
        self.selection_audit = None
        self.audit = _Audit(config["audit_store"])
        if "tokenizer" in config:
            try:
                from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_token_budget import SelectionAudit
                self.selection_audit = SelectionAudit(Path(config["audit_store"]).with_name("token-selections.jsonl"))
            except BaseException:
                self.audit.close()
                raise

    def close(self):
        try:
            if self.session is not None:
                self.session.close()
        finally:
            try:
                if self.selection_audit is not None:
                    self.selection_audit.close()
            finally:
                self.audit.close()

    def handle(self, request):
        """Serve the tools-only MCP subset. Never echo exception text or input."""
        identity = request.get("id") if isinstance(request, dict) else None
        def error(code, message):
            return {"jsonrpc": "2.0", "id": identity, "error": {"code": code, "message": message}}
        try:
            valid_identity = type(identity) is not str or len(identity.encode()) <= 256
        except UnicodeError:
            valid_identity = False
        if (type(request) is not dict or set(request) - {"jsonrpc", "id", "method", "params"}
                or request.get("jsonrpc") != "2.0" or type(request.get("method")) is not str
                or ("id" in request and type(identity) not in {int, str})
                or not valid_identity):
            identity = None
            return error(-32600, "invalid_request")
        method = request["method"]
        params = request.get("params", {})
        if "id" not in request:
            if method == "notifications/initialized" and self.session is not None:
                self.ready = True
            return None
        if type(params) is not dict:
            return error(-32602, "invalid_params")
        if method == "ping":
            result = {}
        elif method == "initialize":
            if self.initialization_attempted:
                return error(-32600, "already_initialized")
            if (type(params.get("protocolVersion")) is not str
                    or type(params.get("capabilities")) is not dict
                    or type(params.get("clientInfo")) is not dict):
                return error(-32602, "invalid_initialization")
            self.initialization_attempted = True
            try:
                if "tokenizer" in self.config:
                    from ipfs_accelerate_py.agent_supervisor.runtime.local_prompt_tokenizer import load_owner_prompt_tokenizer
                    self.tokenizer = load_owner_prompt_tokenizer(**self.config["tokenizer"])
                factory = self.session_factory
                if factory is None:
                    from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSession
                    factory = RepositorySourceSession
                self.session = factory(Path(self.config["workspace"]), Path(self.config["state_dir"]),
                                       representation=self.config["representation"])
            except Exception:
                self.audit.record("initialize", success=False)
                return error(-32603, "repository_session_unavailable")
            self.audit.record("initialize")
            requested = params["protocolVersion"]
            result = {"protocolVersion": requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[-1],
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": "repository_source", "version": "1.0.0"}}
        elif not self.ready:
            return error(-32002, "not_initialized")
        elif method == "tools/list":
            if set(params) - {"_meta"} or ("_meta" in params and type(params["_meta"]) is not dict):
                return error(-32602, "invalid_params")
            result = {"tools": TOOLS}
        elif method == "tools/call":
            name = params.get("name")
            if type(name) is not str or name not in TOOL_NAMES:
                return error(-32602, "unknown_tool")
            success = True
            selection = None
            try:
                if set(params) - {"name", "arguments", "_meta"}:
                    raise ValueError("invalid_params")
                arguments = _arguments(name, params.get("arguments", {}))
                target = {"repository_status": "status_summary", "repository_list": "list_files",
                          "repository_replace_symbol": "replace_symbol"}.get(name, name.removeprefix("repository_"))
                if name == "repository_read" and self.selection_audit is not None:
                    payload, selection = self._selected_read(identity, arguments)
                else:
                    payload = getattr(self.session, target)(**arguments)
                success = not payload.get("refresh_required", False) if name == "repository_replace_symbol" else True
                payload = _delivery_payload(name, payload)
                raw = _json(payload)
                if len(raw) > MAX_RESPONSE_BYTES // 2:
                    raise ValueError("response_limit")
                if selection is not None:
                    frame = _json({"jsonrpc": "2.0", "id": identity, "result": {
                        "content": [{"type": "text", "text": raw.decode()}], "isError": False}}) + b"\n"
                    if (len(frame) > MAX_RESPONSE_BYTES or len(frame) != selection["selected_reply_bytes"]
                            or hashlib.sha256(frame).hexdigest() != selection["selected_reply_sha256"]):
                        raise ValueError("selected_reply_differs")
                    self.selection_audit.record(selection)
            except Exception as exc:
                success = False
                reason = getattr(exc, "code", "tool_failed")
                if type(reason) is not str or re.fullmatch(r"[a-z_]{1,80}", reason) is None:
                    reason = "tool_failed"
                payload = {"error": reason}
                raw = _json(payload)
            self.audit.record("tool", tool=name, success=success, returned_bytes=len(raw),
                compact=name == "repository_read" and payload.get("representation") == "compact",
                canonical_changed=name == "repository_replace_symbol" and payload.get("canonical_changed") is True,
                canonical_uncertain=name == "repository_replace_symbol" and "canonical_changed" in payload and payload["canonical_changed"] is None,
                refreshed=name in {"repository_refresh", "repository_replace_symbol"} and success and not payload.get("refresh_required", False),
                retrieval=_retrieval_event(name, params.get("arguments"), payload, success)
                    if name in RETRIEVAL_TOOLS else None,
                read=_read_event(params.get("arguments"), payload, success)
                    if name == "repository_read" else None)
            result = {"content": [{"type": "text", "text": raw.decode()}], "isError": not success}
        else:
            return error(-32601, "method_not_found")
        return {"jsonrpc": "2.0", "id": identity, "result": result}

    def _selected_read(self, identity, arguments):
        from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_token_budget import select_repository_reply
        requested = arguments.get("representation", self.config["representation"])
        if (any(key in arguments for key in ("start_line", "end_line"))
                and requested != "original"):
            from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_session import RepositorySourceSessionError
            raise RepositorySourceSessionError("line_range_requires_original")
        original = self.session.read(**{**arguments, "representation": "original"})
        compact = self.session.read(**{**arguments, "representation": "compact"}) if requested == "compact" else None
        if compact is not None and any(original.get(key) != compact.get(key) for key in (
                "generation", "path", "original_sha256", "symbol_id", "symbol", "scope", "edit_supported")):
            raise ValueError("source_changed_during_selection")
        def reply(value):
            payload = _json(_delivery_payload("repository_read", value)).decode()
            return (_json({"jsonrpc": "2.0", "id": identity,
                "result": {"content": [{"type": "text", "text": payload}], "isError": False}}) + b"\n").decode()
        selection = select_repository_reply(original_reply=reply(original),
            compact_reply=reply(compact) if compact is not None else None,
            requested_representation=requested, tokenizer=self.tokenizer)
        # Tokenization may take time. Re-read the chosen representation so a
        # concurrent source edit cannot escape the normal freshness fences.
        chosen = original if selection["selected_representation"] == "original" else compact
        current = self.session.read(**{**arguments, "representation": selection["selected_representation"]})
        if _delivery_payload("repository_read", current) != _delivery_payload("repository_read", chosen):
            raise ValueError("source_changed_during_selection")
        return current, selection


def serve(server, incoming, outgoing):
    """Sequential bounded framing also prevents overlapping canonical edits."""
    for _ in range(MAX_REQUESTS):
        raw = incoming.readline(MAX_REQUEST_BYTES + 1)
        if not raw:
            return 0
        if len(raw) > MAX_REQUEST_BYTES or not raw.endswith(b"\n"):
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "request_limit"}}
            outgoing.write(_json(response) + b"\n")
            outgoing.flush()
            return 2
        try:
            request = _loads(raw)
        except (ValueError, RecursionError, UnicodeError):
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "invalid_json"}}
        else:
            # Imported dependencies must never corrupt JSON-RPC stdout.
            with redirect_stdout(sys.stderr):
                response = server.handle(request)
        if response is not None:
            encoded = _json(response)
            if len(encoded) > MAX_RESPONSE_BYTES:
                raise ValueError("response_limit")
            outgoing.write(encoded + b"\n")
            outgoing.flush()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args(argv)
    server = None
    try:
        config = load_config(args.config, args.config_sha256)
        server = RepositoryMCPServer(config)
        return serve(server, sys.stdin.buffer, sys.stdout.buffer)
    except (OSError, ValueError, RecursionError, UnicodeError):
        print("repository_source_mcp_unavailable", file=sys.stderr)
        return 2
    finally:
        if server is not None:
            with redirect_stdout(sys.stderr):
                server.close()


if __name__ == "__main__":
    # Direct invocation pins the implementation checkout ahead of the task cwd.
    if __package__ in {None, ""}:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    raise SystemExit(main())
