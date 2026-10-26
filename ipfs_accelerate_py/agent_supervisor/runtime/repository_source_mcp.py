"""Bounded stdio MCP transport for source-current repository sessions.

The server has no provider or network client. One private, hash-bound config
selects the canonical checkout. Requests cannot select a different root or run
code. Source output appears only in tool replies; the audit contains counters
and content-free commitments for prepared reads, not provider ingestion.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import re
import select
import stat
import sys
import threading

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_read_observation as read_observations

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
    _tool("repository_read", "Read current source with its canonical file SHA256. Optional symbol_id from a query or symbols result selects a supported Python definition and returns its verified local edit name; omit it for the full file. Definitions are dedented and omit enclosing/import context. Compact omits comments/formatting; a bound local tokenizer selects it only for a smaller complete reply. Request original source for missing context.",
          {"generation": GENERATION, "path": PATH, "symbol_id": CID,
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
_NO_READ_OBSERVATION = object()
RETRIEVAL_TOOLS = {"repository_context_query", "repository_capsule_expand"}
RETRIEVAL_COUNTERS = ("query_calls", "successful_queries", "empty_queries", "nonempty_queries",
    "candidates_returned", "expansion_calls", "successful_expansions", "empty_expansions",
    "nonempty_expansions", "capsules_returned", "errors", "stale_generation_errors",
    "stale_source_errors", "root_mismatch_errors", "legacy_retrieval_calls", "unbound_successes",
    "generation_bindings_observed")
RETRIEVAL_FALSE = ("initial_context_link_verified", "omitted_capsule_recovery_verified",
    "model_attention_verified", "token_savings_measured", "proof_authority", "completion_authority")
_RETRIEVAL_ERRORS = {None, "stale_generation", "stale_source", "root_mismatch", "other"}


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


class _Audit:
    def __init__(self, path):
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.sequence = 0
        self._owner_thread = threading.get_ident()
        self._capture = None

    @contextmanager
    def capture(self, witness=None):
        """Witness exact committed lines for one synchronous MCP handle.

        A complete scope can contain zero operations. The current handle emits
        at most one audit line; overflow or ordinary witness publication failure
        makes the observation incomplete without changing its MCP outcome.
        Interrupted handlers and audit append failures remain incomplete.
        """
        if threading.get_ident() != self._owner_thread:
            raise ValueError("audit_capture_thread_differs")
        if self._capture is not None:
            raise ValueError("audit_capture_already_active")
        if witness is None:
            witness = {"status": "incomplete", "operations": []}
        if (type(witness) is not dict or set(witness) != {"status", "operations"}
                or witness["status"] != "incomplete" or type(witness["operations"]) is not list
                or witness["operations"]):
            raise ValueError("invalid_audit_capture")
        capture = {"witness": witness, "failed": False}
        self._capture = capture
        try:
            yield witness
        except BaseException:
            witness["status"] = "incomplete"
            raise
        else:
            witness["status"] = "incomplete" if capture["failed"] else "complete"
        finally:
            if self._capture is not capture:
                witness["status"] = "incomplete"
            self._capture = None

    def _publish_capture(self, capture, *, sequence, raw):
        operations = capture["witness"]["operations"]
        if len(operations) >= 1:
            raise ValueError("audit_capture_operation_limit")
        operations.append({"sequence": sequence, "sha256": hashlib.sha256(raw).hexdigest(),
                           "bytes": len(raw)})

    def record(self, event, *, tool=None, success=True, returned_bytes=0,
               compact=False, canonical_changed=False, canonical_uncertain=False, refreshed=False,
               retrieval=None, read_observation=_NO_READ_OBSERVATION):
        capture = self._capture
        try:
            return self._record(event, tool=tool, success=success, returned_bytes=returned_bytes,
                compact=compact, canonical_changed=canonical_changed, canonical_uncertain=canonical_uncertain,
                refreshed=refreshed, retrieval=retrieval, read_observation=read_observation)
        except BaseException:
            if capture is not None:
                capture["failed"] = True
                capture["witness"]["status"] = "incomplete"
            raise

    def _record(self, event, *, tool=None, success=True, returned_bytes=0,
                compact=False, canonical_changed=False, canonical_uncertain=False, refreshed=False,
                retrieval=None, read_observation=_NO_READ_OBSERVATION):
        capture = self._capture
        if capture is not None and threading.get_ident() != self._owner_thread:
            raise ValueError("audit_capture_thread_differs")
        self.sequence += 1
        row = {"schema": "repository-source-mcp-audit@1", "sequence": self.sequence,
               "event": event, "tool": tool, "success": success, "returned_bytes": returned_bytes,
               "compact": compact, "canonical_changed": canonical_changed,
               "canonical_uncertain": canonical_uncertain, "refreshed": refreshed}
        if retrieval is not None:
            _validate_retrieval_event(retrieval, success)
            row.update(schema="repository-source-mcp-audit@2", retrieval=retrieval)
        if read_observation is not _NO_READ_OBSERVATION:
            if event != "tool" or tool != "repository_read" or retrieval is not None:
                raise ValueError("invalid_read_audit")
            if read_observation is not None:
                if not success:
                    raise ValueError("invalid_read_audit")
                read_observation = read_observations.validate_read_event(read_observation)
            row.update(schema="repository-source-mcp-audit@3", read_observation=read_observation)
        raw = _json(row) + b"\n"
        if os.fstat(self.fd).st_size + len(raw) > MAX_AUDIT_BYTES:
            raise ValueError("audit_limit")
        view = memoryview(raw)
        while view:
            size = os.write(self.fd, view)
            if size <= 0:
                raise OSError("audit_write_failed")
            view = view[size:]
        if capture is not None:
            try:
                self._publish_capture(capture, sequence=row["sequence"], raw=raw)
            except Exception:
                # Ordinary evidence failures must not turn a successful tool
                # into an error. Control-flow interruptions still propagate
                # through the scope and cannot become complete observations.
                capture["failed"] = True
                capture["witness"]["status"] = "incomplete"

    def close(self):
        os.close(self.fd)


def _empty_repository_audit_summary():
    return {"status": "not_observed", "initializations": 0, "tool_calls": 0,
            "successful_reads": 0, "compact_reads": 0, "canonical_edits": 0,
            "uncertain_edits": 0, "refreshes": 0, "errors": 0, "returned_bytes": 0}


def summarize_repository_audit_bytes(raw: bytes) -> dict:
    """Summarize bounded retained bytes without I/O or provenance claims.

    The caller owns secure acquisition and authentication of these bytes. Empty
    input is not observed; malformed input has unavailable, null counters.
    Prepared operation/read observations do not establish response delivery.
    """
    result = _empty_repository_audit_summary()
    try:
        if (type(raw) is not bytes or len(raw) > MAX_AUDIT_BYTES
                or raw and not raw.endswith(b"\n")):
            raise ValueError("invalid_audit")
        rows = raw.splitlines()
        if len(rows) > MAX_REQUESTS + 1:
            raise ValueError("invalid_audit")
        retrieval = dict(schema="repository-source-retrieval-observation@1", complete=True,
            event_chain_sha256=hashlib.sha256(b"").hexdigest(),
            **dict.fromkeys(RETRIEVAL_COUNTERS, 0), **dict.fromkeys(RETRIEVAL_FALSE, False))
        retrieval_seen = False
        audit_rows = []
        for sequence, line in enumerate(rows, 1):
            row = _loads(line)
            modern = type(row) is dict and row.get("schema") == "repository-source-mcp-audit@2"
            read_row = type(row) is dict and row.get("schema") == "repository-source-mcp-audit@3"
            extras = {"retrieval"} if modern else {"read_observation"} if read_row else set()
            if (type(row) is not dict or set(row) != AUDIT_FIELDS | extras
                    or row["schema"] not in {"repository-source-mcp-audit@1", "repository-source-mcp-audit@2", "repository-source-mcp-audit@3"}
                    or type(row["sequence"]) is not int or row["sequence"] != sequence
                    or row["event"] not in {"initialize", "tool"}
                    or (row["event"] == "initialize" and row["tool"] is not None)
                    or (row["event"] == "tool" and row["tool"] not in TOOL_NAMES)
                    or any(type(row[key]) is not bool for key in ("success", "compact", "canonical_changed", "canonical_uncertain", "refreshed"))
                    or type(row["returned_bytes"]) is not int
                    or not 0 <= row["returned_bytes"] <= MAX_RESPONSE_BYTES):
                raise ValueError("invalid_audit")
            if read_row:
                if row["event"] != "tool" or row["tool"] != "repository_read":
                    raise ValueError("invalid_read_audit")
                event = row["read_observation"]
                if event is not None:
                    if not row["success"]:
                        raise ValueError("invalid_read_audit")
                    read_observations.validate_read_event(event)
                    if row["compact"] is not (event["representation"] == "compact"):
                        raise ValueError("invalid_read_audit")
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
            result["initializations"] += row["event"] == "initialize" and row["success"]
            result["tool_calls"] += row["event"] == "tool"
            result["successful_reads"] += row["tool"] == "repository_read" and row["success"]
            result["compact_reads"] += row["tool"] == "repository_read" and row["success"] and row["compact"]
            result["canonical_edits"] += row["canonical_changed"]
            result["uncertain_edits"] += row["canonical_uncertain"]
            result["refreshes"] += row["refreshed"]
            result["errors"] += not row["success"]
            result["returned_bytes"] += row["returned_bytes"]
            audit_rows.append(row)
        if rows:
            result["status"] = "observed"
        if retrieval_seen:
            retrieval["complete"] = retrieval["legacy_retrieval_calls"] == retrieval["unbound_successes"] == 0
            result["retrieval_observation"] = validate_retrieval_observation(
                retrieval, tool_calls=result["tool_calls"], errors=result["errors"])
        read_summary = read_observations.summarize_read_observation(audit_rows)
        if read_summary is not None:
            result["read_observation"] = read_observations.validate_read_observation(
                read_summary, successful_reads=result["successful_reads"], compact_reads=result["compact_reads"],
                returned_bytes=result["returned_bytes"], tool_calls=result["tool_calls"], errors=result["errors"])
        return result
    except (ValueError, OSError, RecursionError, UnicodeError, TypeError):
        return {"status": "unavailable", **{key: None for key in result if key != "status"}}


def summarize_repository_audit(path: str | Path) -> dict:
    """Read one private audit without relaxing its file/owner checks."""
    result = _empty_repository_audit_summary()
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
        return summarize_repository_audit_bytes(raw)
    except FileNotFoundError:
        return result
    except (ValueError, OSError, RecursionError, UnicodeError, TypeError):
        return {"status": "unavailable", **{key: None for key in result if key != "status"}}


class RepositoryMCPServer:
    def __init__(self, config, *, session_factory=None, scheduler=None, parent_lease=None,
                 cancel_event=None, deadline_monotonic=None):
        from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import (
            _resource_controls, _validate_resource_deadline,
        )

        self._deadline_monotonic = _validate_resource_deadline(deadline_monotonic)
        # Custom factories retain the existing trusted opaque-dependency seam.
        # Native factories must validate borrowed owner controls before files
        # are created; explicit deadlines apply to either factory.
        self._resource_signal = (_resource_controls(scheduler=scheduler, parent_lease=parent_lease,
            cancel_event=cancel_event) if session_factory is None
            else cancel_event if callable(getattr(cancel_event, "is_set", None)) else None)
        self._interruptible_stdio = self._resource_signal is not None or deadline_monotonic is not None
        self.checkpoint()
        # Trusted in-process dependencies only. Never serialize native lease
        # capabilities into the closed model-visible MCP configuration.
        self._resource_options = {key: value for key, value in (
            ("scheduler", scheduler), ("parent_lease", parent_lease), ("cancel_event", cancel_event),
            ("deadline_monotonic", deadline_monotonic))
            if value is not None}
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

    def checkpoint(self):
        """Cooperative stop fence; cleanup remains available after cancellation."""
        from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_index import _check_resource_cancellation
        _check_resource_cancellation(self._resource_signal, self._deadline_monotonic)

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
        self.checkpoint()
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
                                       representation=self.config["representation"], **self._resource_options)
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
                read_observation=read_observations.make_read_event(params.get("arguments"), payload, success)
                    if name == "repository_read" else _NO_READ_OBSERVATION)
            result = {"content": [{"type": "text", "text": raw.decode()}], "isError": not success}
        else:
            return error(-32601, "method_not_found")
        return {"jsonrpc": "2.0", "id": identity, "result": result}

    def capture_audit_operations(self, witness=None):
        """Scope content-free append witnesses to one owner-dispatched handle."""
        return self.audit.capture(witness)

    def _selected_read(self, identity, arguments):
        from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_token_budget import select_repository_reply
        requested = arguments.get("representation", self.config["representation"])
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


class _InterruptibleStdio:
    """Bounded FD framing for owned stdio, including partial and blocked pipes.

    No worker thread can outlive this scope. Synchronous handlers cooperate at
    their own checkpoints; this transport does not claim hard preemption.
    """

    def __init__(self, server, incoming, outgoing):
        self.server, self.pending = server, bytearray()
        self.input_fd, self.output_fd = incoming.fileno(), outgoing.fileno()
        if (type(self.input_fd) is not int or type(self.output_fd) is not int
                or min(self.input_fd, self.output_fd) < 0 or self.input_fd == self.output_fd):
            raise ValueError("interruptible_stdio_descriptors_required")
        self._blocking = {}

    def __enter__(self):
        try:
            for descriptor in (self.input_fd, self.output_fd):
                self._blocking[descriptor] = os.get_blocking(descriptor)
                os.set_blocking(descriptor, False)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for descriptor, blocking in self._blocking.items():
            try:
                os.set_blocking(descriptor, blocking)
            except OSError:
                pass
        self._blocking.clear()

    def _ready(self, descriptor, *, writing=False):
        while True:
            self.server.checkpoint()
            try:
                readable, writable, _ = select.select(
                    [] if writing else [descriptor], [descriptor] if writing else [], [], .1)
            except InterruptedError:
                continue
            self.server.checkpoint()
            if (writable if writing else readable):
                return

    def readline(self, limit):
        while True:
            self.server.checkpoint()
            newline = self.pending.find(b"\n", 0, limit)
            size = newline + 1 if newline >= 0 else limit if len(self.pending) >= limit else None
            if size is not None:
                raw = bytes(self.pending[:size])
                del self.pending[:size]
                return raw
            self._ready(self.input_fd)
            try:
                chunk = os.read(self.input_fd, min(65536, limit - len(self.pending)))
            except (BlockingIOError, InterruptedError):
                continue
            if not chunk:
                raw = bytes(self.pending)
                self.pending.clear()
                return raw
            self.pending.extend(chunk)

    def write(self, raw):
        if type(raw) is not bytes or len(raw) > MAX_RESPONSE_BYTES + 1:
            raise ValueError("response_limit")
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            self._ready(self.output_fd, writing=True)
            try:
                count = os.write(self.output_fd, view[offset:offset + 65536])
            except (BlockingIOError, InterruptedError):
                continue
            if count <= 0:
                raise OSError("stdio_write_failed")
            offset += count

    def flush(self):
        # Writes go directly to the owned FD; there is no Python output buffer.
        pass


def _serve_frames(server, incoming, outgoing):
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


def serve(server, incoming, outgoing):
    """Sequential bounded requests; explicit resource controls poll owned FDs."""
    if getattr(server, "_interruptible_stdio", False):
        with _InterruptibleStdio(server, incoming, outgoing) as streams:
            return _serve_frames(server, streams, streams)
    return _serve_frames(server, incoming, outgoing)


def main(argv=None):
    protocol_input, protocol_output = sys.stdin.buffer, sys.stdout.buffer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--resource-grant", type=Path)
    parser.add_argument("--resource-grant-sha256")
    args = parser.parse_args(argv)
    if (args.resource_grant is None) != (args.resource_grant_sha256 is None):
        parser.error("resource grant path and digest must be supplied together")

    def run(config, options):
        with redirect_stdout(sys.stderr):
            server = RepositoryMCPServer(config, **options)
        try:
            return serve(server, protocol_input, protocol_output)
        finally:
            # Close synchronous session/index handles while delegated capacity
            # is still held. A cancellation observer never releases this scope
            # in parallel with a running handler.
            with redirect_stdout(sys.stderr):
                server.close()

    failures = (OSError, ValueError, RecursionError, UnicodeError, ImportError)
    try:
        config = load_config(args.config, args.config_sha256)
        if args.resource_grant is None:
            return run(config, {})
        with redirect_stdout(sys.stderr):
            # Ordinary help, ping and EOF do not need datasets. Only explicit
            # delegation loads the native admission API at CLI startup.
            from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceSchedulerError
            failures += (ResourceSchedulerError,)
            from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_resource_handoff import (
                delegated_repository_source_resources,
            )
            with delegated_repository_source_resources(artifact=args.resource_grant,
                    expected_sha256=args.resource_grant_sha256, config_sha256=args.config_sha256,
                    workspace=Path(config["workspace"])) as phase:
                return run(config, phase.native_options())
    except failures:
        print("repository_source_mcp_unavailable", file=sys.stderr)
        return 2


if __name__ == "__main__":
    # Direct invocation pins the implementation checkout ahead of the task cwd.
    if __package__ in {None, ""}:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    raise SystemExit(main())
