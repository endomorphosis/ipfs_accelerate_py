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
TOOLS = [
    _tool("repository_status", "Get the current generation, index health and cache counts/digest. No source content or full cache inventory.", {}, []),
    _tool("repository_cache_inventory", "Read a page of retained opaque cache metadata using the current status digest; caches are not natively indexed.",
          {"generation": GENERATION, "expected_inventory_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
           "offset": {"type": "integer", "minimum": 0}, "limit": LIMIT}, ["generation", "expected_inventory_sha256"]),
    _tool("repository_list", "List a bounded page of files in a source-current generation.",
          {"generation": GENERATION, "offset": {"type": "integer", "minimum": 0}, "limit": LIMIT}, ["generation"]),
    _tool("repository_symbols", "Find native datasets symbols in the current generation; no proof authority.",
          {"generation": GENERATION, "path": PATH, "query": {"type": "string", "maxLength": 256}, "limit": LIMIT}, ["generation"]),
    _tool("repository_read", "Read a source-current original or compact file with its canonical SHA256. Compact Python omits comments and formatting. When a local tokenizer is bound, compact requests return original source unless the complete compact reply uses fewer tokens.",
          {"generation": GENERATION, "path": PATH, "representation": {"type": "string", "enum": ["original", "compact"]}}, ["generation", "path"]),
    _tool("repository_replace_symbol", "Replace one Python definition using the read's source_sha256. symbol is its local name, such as Class.method, not a native CID or module-prefixed index name. Refreshes the canonical checkout and index; check refresh_required and run task tests.",
          {"generation": GENERATION, "path": PATH, "symbol": {"type": "string", "minLength": 1, "maxLength": 1024},
           "expected_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
           "replacement": {"type": "string", "maxLength": 262144}},
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
        return {"generation": value["generation"], "path": value["path"],
                "source_sha256": value["original_sha256"],
                **{key: value[key] for key in ("representation", "encoding", "content", "proof_authority")}}
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


class _Audit:
    def __init__(self, path):
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.sequence = 0

    def record(self, event, *, tool=None, success=True, returned_bytes=0,
               compact=False, canonical_changed=False, canonical_uncertain=False, refreshed=False):
        self.sequence += 1
        row = {"schema": "repository-source-mcp-audit@1", "sequence": self.sequence,
               "event": event, "tool": tool, "success": success, "returned_bytes": returned_bytes,
               "compact": compact, "canonical_changed": canonical_changed,
               "canonical_uncertain": canonical_uncertain, "refreshed": refreshed}
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
        for sequence, line in enumerate(rows, 1):
            row = _loads(line)
            if (type(row) is not dict or set(row) != AUDIT_FIELDS
                    or row["schema"] != "repository-source-mcp-audit@1"
                    or type(row["sequence"]) is not int or row["sequence"] != sequence
                    or row["event"] not in {"initialize", "tool"}
                    or (row["event"] == "initialize" and row["tool"] is not None)
                    or (row["event"] == "tool" and row["tool"] not in TOOL_NAMES)
                    or any(type(row[key]) is not bool for key in ("success", "compact", "canonical_changed", "canonical_uncertain", "refreshed"))
                    or type(row["returned_bytes"]) is not int
                    or not 0 <= row["returned_bytes"] <= MAX_RESPONSE_BYTES):
                raise ValueError("invalid_audit")
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
                refreshed=name in {"repository_refresh", "repository_replace_symbol"} and success and not payload.get("refresh_required", False))
            result = {"content": [{"type": "text", "text": raw.decode()}], "isError": not success}
        else:
            return error(-32601, "method_not_found")
        return {"jsonrpc": "2.0", "id": identity, "result": result}

    def _selected_read(self, identity, arguments):
        from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_token_budget import select_repository_reply
        requested = arguments.get("representation", self.config["representation"])
        original = self.session.read(**{**arguments, "representation": "original"})
        compact = self.session.read(**{**arguments, "representation": "compact"}) if requested == "compact" else None
        if compact is not None and any(original[key] != compact[key] for key in ("generation", "path", "original_sha256")):
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
