"""Bounded wire protocol for a separately owned, source-read-only MCP service.

This protocol does not allocate resources or grant proof/task authority. The
owner retains its native lease and private index; workers receive only a public
identity descriptor and read/query/refresh access. Refresh changes private
index state, never canonical source. Each frame has an absolute deadline and
cooperative cancellation; a peer cannot extend the deadline with partial I/O.

Only the standard library and the neighbouring numerical client are imported,
so loading this file by its path under Python ``-I`` does not import the package.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import importlib.util
import math
import os
from pathlib import Path
import select
import socket
import stat
import time
import uuid

if __package__:
    from . import numerical_obligation_client as _base
else:  # Path loading under -I must not depend on an ambient PYTHONPATH.
    _spec = importlib.util.spec_from_file_location(
        "_repository_owner_numerical_client", Path(__file__).with_name("numerical_obligation_client.py"))
    _base = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_base)

DESCRIPTOR_FLAG = "--repository-source-owner-descriptor"
DIGEST_FLAG = "--repository-source-owner-descriptor-sha256"
DESCRIPTOR_SCHEMA = "repository-owner-mcp-descriptor@1"
REQUEST_SCHEMA = "repository-owner-mcp-request@1"
RESPONSE_SCHEMA = "repository-owner-mcp-response@1"
MAX_MESSAGE_REQUEST_BYTES = 1024 * 1024
MAX_MESSAGE_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_WRAPPER_BYTES = 4096
MAX_REQUEST_FRAME_BYTES = MAX_MESSAGE_REQUEST_BYTES + MAX_WRAPPER_BYTES
MAX_RESPONSE_FRAME_BYTES = MAX_MESSAGE_RESPONSE_BYTES + MAX_WRAPPER_BYTES
MAX_FRAME_BYTES = MAX_RESPONSE_FRAME_BYTES
MAX_FRAME_SECONDS = 2.0
MAX_RPC_SECONDS = 30.0
POLL_SECONDS = 0.1
MAX_JSON_DEPTH = 20
MAX_JSON_NODES = 32768
READ_ONLY_TOOL_NAMES = frozenset({"repository_status", "repository_list", "repository_symbols",
    "repository_read", "repository_refresh", "repository_cache_inventory",
    "repository_context_query", "repository_capsule_expand"})
REASON_CODES = frozenset({"invalid_request", "peer_refused", "stale_context", "budget_exhausted",
    "session_closed", "retrieval_refused", "deadline_exceeded", "cancelled",
    "transport_error", "invalid_response"})
_DESCRIPTOR_FIELDS = frozenset({"schema", "session_id", "owner_uid", "worker_uid", "owner_pid",
    "owner_start_time_ticks", "workspace", "boundary_manifest_sha256", "socket_path",
    "socket_device", "socket_inode", "issued_monotonic_ns", "expires_monotonic_ns",
    "max_requests", "max_total_response_bytes", "proof_authority", "execution_authority",
    "completion_authority", "representation", "config_sha256", "helper_sha256", "read_only"})


class RepositorySourceOwnerProtocolError(ValueError):
    """Closed refusal; no source bytes or private capabilities appear in errors."""
    def __init__(self, reason_code="invalid_request"):
        self.reason_code = reason_code if reason_code in REASON_CODES else "invalid_request"
        super().__init__(self.reason_code)


def _need(condition, reason_code="invalid_request"):
    if not condition:
        raise RepositorySourceOwnerProtocolError(reason_code)


def _tree(value):
    pending, nodes = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _need(depth <= MAX_JSON_DEPTH and nodes <= MAX_JSON_NODES)
        if type(item) is dict:
            _need(all(type(key) is str for key in item))
            pending.extend((member, depth + 1) for member in item.values())
        elif type(item) is list:
            pending.extend((member, depth + 1) for member in item)
        else:
            _need(item is None or type(item) in (str, bool, int, float))
            if type(item) is float:
                _need(math.isfinite(item))


def canonical_json(value):
    _tree(value)
    try:
        return _base.canonical_json(value)
    except (ValueError, TypeError, OverflowError, UnicodeError, RecursionError) as exc:
        raise RepositorySourceOwnerProtocolError() from exc


def _bound(max_bytes):
    _need(type(max_bytes) is int and 2 <= max_bytes <= MAX_FRAME_BYTES)
    return max_bytes


def strict_json_loads(raw, *, max_bytes=MAX_FRAME_BYTES):
    """Reject duplicates, nonfinite numbers, invalid UTF-8 and oversized trees."""
    import json
    _bound(max_bytes)
    def pairs(items):
        value = {}
        for key, item in items:
            _need(key not in value)
            value[key] = item
        return value
    def finite_float(text):
        value = float(text)
        _need(math.isfinite(value))
        return value
    def invalid_constant(_):
        raise RepositorySourceOwnerProtocolError()
    try:
        if type(raw) is bytes:
            _need(len(raw) <= max_bytes)
            raw = raw.decode("utf-8")
        _need(type(raw) is str and len(raw.encode("utf-8")) <= max_bytes)
        result = json.loads(raw, object_pairs_hook=pairs, parse_float=finite_float,
                            parse_constant=invalid_constant)
        _tree(result)
        # JSON permits escaped lone surrogates syntactically, but our wire is UTF-8.
        canonical_json(result)
        return result
    except (ValueError, TypeError, OverflowError, UnicodeError, RecursionError) as exc:
        raise RepositorySourceOwnerProtocolError() from exc


def process_identity(pid):
    try:
        return _base.process_identity(pid)
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        raise RepositorySourceOwnerProtocolError("peer_refused") from exc


def validate_descriptor(value):
    try:
        _base._validate_descriptor(value, schema=DESCRIPTOR_SCHEMA, fields=_DESCRIPTOR_FIELDS,
            digest_fields=("boundary_manifest_sha256", "config_sha256", "helper_sha256"),
            identity_fields=())
        _need(type(value["representation"]) is str and value["representation"] in ("original", "compact")
              and value["read_only"] is True)
        return value
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise RepositorySourceOwnerProtocolError("peer_refused") from exc


def load_descriptor(path, digest, *, workspace=None):
    """Reuse existing kernel/ownership checks, with this service's closed schema."""
    try:
        return _base.load_descriptor(path, digest, workspace=workspace, validator=validate_descriptor)
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        raise RepositorySourceOwnerProtocolError("peer_refused") from exc


def _id(value):
    return (_base._integer(value, -(1 << 53) + 1, (1 << 53) - 1)
            or _base._text(value, 256))


def _meta(params, allowed, required=()):
    _need(type(params) is dict and not set(params) - set(allowed) - {"_meta"}
          and set(required) <= set(params)
          and ("_meta" not in params or type(params["_meta"]) is dict))


def _arguments(name, arguments):
    # The native MCP handler applies its own full validation as well. Keeping a
    # closed field vocabulary here prevents the worker from selecting an owner
    # root, command, configuration, private index or a write operation.
    fields = {
        "repository_status": (set(), set()),
        "repository_list": ({"generation", "offset", "limit"}, {"generation"}),
        "repository_symbols": ({"generation", "path", "query", "limit"}, {"generation"}),
        "repository_read": ({"generation", "path", "symbol_id", "representation"}, {"generation", "path"}),
        "repository_refresh": ({"generation"}, {"generation"}),
        "repository_cache_inventory": ({"generation", "expected_inventory_sha256", "offset", "limit"},
                                        {"generation", "expected_inventory_sha256"}),
        "repository_context_query": ({"generation", "query", "limit", "max_bytes"}, {"generation", "query"}),
        "repository_capsule_expand": ({"generation", "semantic_root_cid", "stable_symbol_ids", "max_bytes"},
                                      {"generation", "semantic_root_cid", "stable_symbol_ids"}),
    }
    allowed, required = fields[name]
    _need(type(arguments) is dict and set(arguments) <= allowed and required <= set(arguments))
    for key, value in arguments.items():
        if key in {"generation", "offset", "limit", "max_bytes"}:
            minimum, maximum = (0 if key == "offset" else 1), (1 << 31) - 1
            if key == "limit":
                maximum = 32 if name == "repository_context_query" else 100
            elif key == "max_bytes":
                minimum, maximum = 1024, 65536
            _need(_base._integer(value, minimum, maximum))
        elif key == "representation":
            _need(type(value) is str and value in ("original", "compact"))
        elif key == "expected_inventory_sha256":
            _need(_base._hex(value))
        elif key == "stable_symbol_ids":
            _need(type(value) is list and 1 <= len(value) <= 32
                  and all(_base._text(item, 256) for item in value)
                  and len(set(value)) == len(value))
        else:
            maximum = 4096 if key == "query" and name == "repository_context_query" else 256
            if key == "path":
                maximum = 1024
            # repository_symbols permits an empty query; no other string does.
            _need(value == "" and key == "query" and name == "repository_symbols"
                  or _base._text(value, maximum))
            if key == "path":
                _need(not Path(value).is_absolute() and str(Path(value)) == value
                      and value != "." and ".." not in Path(value).parts)


def validate_message(message):
    _need(type(message) is dict and set(message) <= {"jsonrpc", "id", "method", "params"}
          and message.get("jsonrpc") == "2.0" and type(message.get("method")) is str)
    method = message["method"]
    _need(method in {"initialize", "ping", "notifications/initialized", "tools/list", "tools/call"})
    params = message.get("params", {})
    if method == "notifications/initialized":
        _need("id" not in message and type(params) is dict and not params)
    else:
        _need("id" in message and _id(message["id"]))
        if method == "initialize":
            _meta(params, {"protocolVersion", "capabilities", "clientInfo"},
                  {"protocolVersion", "capabilities", "clientInfo"})
            _need(_base._text(params["protocolVersion"], 64)
                  and type(params["capabilities"]) is dict and type(params["clientInfo"]) is dict)
            client = params["clientInfo"]
            _need(set(client) <= {"name", "version", "title"} and {"name", "version"} <= set(client)
                  and all(_base._text(item, 512) for item in client.values()))
        elif method == "ping":
            _need(type(params) is dict and not params)
        elif method == "tools/list":
            _meta(params, set())
        else:
            _meta(params, {"name", "arguments"}, {"name"})
            _need(type(params["name"]) is str and params["name"] in READ_ONLY_TOOL_NAMES)
            _arguments(params["name"], params.get("arguments", {}))
    _need(len(canonical_json(message)) <= MAX_MESSAGE_REQUEST_BYTES)
    return message


def validate_request(value):
    _need(type(value) is dict and set(value) == {"schema", "session_id", "request_id", "message"}
          and value["schema"] == REQUEST_SCHEMA and _base._hex(value["session_id"], 32)
          and _base._hex(value["request_id"], 32))
    validate_message(value["message"])
    _need(len(canonical_json(value)) <= MAX_REQUEST_FRAME_BYTES)
    return value


def validate_response(value, *, request, descriptor=None, max_bytes=MAX_RESPONSE_FRAME_BYTES):
    validate_request(request)
    _bound(max_bytes)
    _need(type(value) is dict and set(value) == {"schema", "session_id", "request_id", "ok", "message", "reason_code"}
          and value["schema"] == RESPONSE_SCHEMA and type(value["ok"]) is bool
          and value["session_id"] == request["session_id"]
          and value["request_id"] == request["request_id"], "invalid_response")
    if descriptor is not None:
        validate_descriptor(descriptor)
        _need(value["session_id"] == descriptor["session_id"], "invalid_response")
        max_bytes = min(max_bytes, descriptor["max_total_response_bytes"])
    _need(len(canonical_json(value)) <= max_bytes, "invalid_response")
    if not value["ok"]:
        _need(value["message"] is None and type(value["reason_code"]) is str
              and value["reason_code"] in REASON_CODES, "invalid_response")
        return value
    message = value["message"]
    _need(value["reason_code"] is None, "invalid_response")
    if "id" not in request["message"]:
        _need(message is None, "invalid_response")
        return value
    _need(type(message) is dict and set(message) in ({"jsonrpc", "id", "result"}, {"jsonrpc", "id", "error"})
          and message["jsonrpc"] == "2.0"
          and type(message["id"]) is type(request["message"]["id"])
          and message["id"] == request["message"]["id"], "invalid_response")
    if "error" in message:
        error = message["error"]
        _need(type(error) is dict and set(error) in ({"code", "message"}, {"code", "message", "data"})
              and _base._integer(error["code"], -(1 << 31), (1 << 31) - 1)
              and _base._text(error["message"], 4096), "invalid_response")
    else:
        _need(type(message["result"]) is dict, "invalid_response")
        if request["message"]["method"] == "tools/list":
            result = message["result"]
            _need(set(result) == {"tools"} and type(result["tools"]) is list
                  and all(type(tool) is dict and type(tool.get("name")) is str
                          and tool["name"] in READ_ONLY_TOOL_NAMES for tool in result["tools"])
                  and len({tool["name"] for tool in result["tools"]}) == len(result["tools"]), "invalid_response")
    _need(len(canonical_json(message)) <= MAX_MESSAGE_RESPONSE_BYTES, "invalid_response")
    return value


def _deadline(value, *, max_seconds=MAX_FRAME_SECONDS):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    _need(valid, "deadline_exceeded")
    return min(value, time.monotonic() + max_seconds)


def _check(deadline, cancel_event):
    if cancel_event is not None:
        _need(callable(getattr(cancel_event, "is_set", None)))
        if cancel_event.is_set():
            raise RepositorySourceOwnerProtocolError("cancelled")
    remaining = deadline - time.monotonic()
    _need(remaining > 0, "deadline_exceeded")
    return remaining


@contextmanager
def _nonblocking(channel):
    try:
        descriptor = channel if type(channel) is int else channel.fileno()
        _need(type(descriptor) is int and descriptor >= 0, "transport_error")
        mode = os.fstat(descriptor).st_mode
        _need(stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode), "transport_error")
        blocking = os.get_blocking(descriptor)
        os.set_blocking(descriptor, False)
        try:
            yield descriptor
        finally:
            os.set_blocking(descriptor, blocking)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        if isinstance(exc, RepositorySourceOwnerProtocolError):
            raise
        raise RepositorySourceOwnerProtocolError("transport_error") from exc


def _ready(descriptor, *, writing, deadline, cancel_event):
    while True:
        remaining = _check(deadline, cancel_event)
        try:
            ready = select.select([] if writing else [descriptor], [descriptor] if writing else [],
                                  [], min(POLL_SECONDS, remaining))
        except InterruptedError:
            continue
        _check(deadline, cancel_event)
        if ready[1 if writing else 0]:
            return


def _exact(descriptor, size, *, deadline, cancel_event):
    result = bytearray()
    while len(result) < size:
        _ready(descriptor, writing=False, deadline=deadline, cancel_event=cancel_event)
        try:
            part = os.read(descriptor, min(65536, size - len(result)))
        except (BlockingIOError, InterruptedError):
            continue
        _need(bool(part), "transport_error")
        result.extend(part)
    _check(deadline, cancel_event)
    return bytes(result)


def send_frame(channel, value, *, deadline_monotonic, cancel_event=None, max_bytes=MAX_FRAME_BYTES):
    """Write one complete length-prefixed canonical frame, never a truncation."""
    _bound(max_bytes)
    deadline = _deadline(deadline_monotonic, max_seconds=MAX_FRAME_SECONDS)
    _check(deadline, cancel_event)
    raw = canonical_json(value)
    _need(type(value) is dict and 2 <= len(raw) <= max_bytes)
    _check(deadline, cancel_event)
    payload, offset = len(raw).to_bytes(4, "big") + raw, 0
    with _nonblocking(channel) as descriptor:
        while offset < len(payload):
            _ready(descriptor, writing=True, deadline=deadline, cancel_event=cancel_event)
            try:
                count = os.write(descriptor, payload[offset:offset + 65536])
            except (BlockingIOError, InterruptedError):
                continue
            _need(count > 0, "transport_error")
            offset += count
    _check(deadline, cancel_event)
    return len(raw)


def receive_frame(channel, *, deadline_monotonic, cancel_event=None, max_bytes=MAX_FRAME_BYTES):
    """Read one frame with one absolute deadline across both header and body."""
    _bound(max_bytes)
    deadline = _deadline(deadline_monotonic, max_seconds=MAX_FRAME_SECONDS)
    _check(deadline, cancel_event)
    with _nonblocking(channel) as descriptor:
        size = int.from_bytes(_exact(descriptor, 4, deadline=deadline, cancel_event=cancel_event), "big")
        _need(2 <= size <= max_bytes)
        raw = _exact(descriptor, size, deadline=deadline, cancel_event=cancel_event)
    value = strict_json_loads(raw, max_bytes=max_bytes)
    _need(type(value) is dict and canonical_json(value) == raw)
    _check(deadline, cancel_event)
    return value


def make_request(session_id, message, *, request_id=None):
    return validate_request({"schema": REQUEST_SCHEMA, "session_id": session_id,
        "request_id": uuid.uuid4().hex if request_id is None else request_id, "message": message})


def _wait_for_response_start(channel, *, deadline_monotonic, cancel_event=None):
    """Wait without consuming bytes; the subsequent frame retains its own cap."""
    deadline = _deadline(deadline_monotonic, max_seconds=MAX_RPC_SECONDS)
    _check(deadline, cancel_event)
    with _nonblocking(channel) as descriptor:
        _ready(descriptor, writing=False, deadline=deadline, cancel_event=cancel_event)


def request_message(descriptor_path, descriptor_sha256, *, message, workspace=None,
                    deadline_monotonic=None, cancel_event=None):
    """Perform one exchange; session-wide admission/accounting is the owner's.

    Connection and response-start waiting share a maximum 30-second absolute
    deadline, shortened by the descriptor and optional caller deadline. Each
    active frame transfer remains bounded to two seconds. A slower owner job
    may complete without a delivered reply; this helper neither retries nor
    interprets a transport timeout as proof that no private refresh happened.
    """
    descriptor = load_descriptor(descriptor_path, descriptor_sha256, workspace=workspace)
    request = make_request(descriptor["session_id"], message)
    expires = descriptor["expires_monotonic_ns"] / 1_000_000_000
    deadline = _deadline(expires, max_seconds=MAX_RPC_SECONDS)
    if deadline_monotonic is not None:
        deadline = min(deadline, _deadline(deadline_monotonic, max_seconds=MAX_RPC_SECONDS))
    _check(deadline, cancel_event)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.setblocking(False)
            result = channel.connect_ex(descriptor["socket_path"])
            _need(result in (0, errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY), "transport_error")
            if result:
                _ready(channel.fileno(), writing=True, deadline=deadline, cancel_event=cancel_event)
                _need(channel.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR) == 0, "transport_error")
            _base._verify_peer(channel, descriptor)
            path = Path(descriptor_path)
            _base._socket_identity(descriptor, path.parent, path.parent.stat().st_gid)
            send_frame(channel, request, deadline_monotonic=deadline, cancel_event=cancel_event,
                       max_bytes=MAX_REQUEST_FRAME_BYTES)
            _wait_for_response_start(channel, deadline_monotonic=deadline, cancel_event=cancel_event)
            response = receive_frame(channel, deadline_monotonic=deadline, cancel_event=cancel_event,
                                     max_bytes=descriptor["max_total_response_bytes"])
            _base._verify_peer(channel, descriptor)
            _check(deadline, cancel_event)
        return validate_response(response, request=request, descriptor=descriptor)
    except RepositorySourceOwnerProtocolError:
        raise
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        raise RepositorySourceOwnerProtocolError("transport_error") from exc
