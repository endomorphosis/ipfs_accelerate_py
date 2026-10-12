"""Bounded, read-only numerical obligation client; usable with Python ``-I``.

This module deliberately imports only the standard library. The owner selects
and validates its private context; a worker can nominate only a bounded query.
Socket routing and the public descriptor do not grant proof or task authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import struct
import sys
import time
import uuid

DESCRIPTOR_FLAG = "--numerical-obligation-descriptor"
DIGEST_FLAG = "--numerical-obligation-descriptor-sha256"
DESCRIPTOR_ENV = "IPFS_SUPERVISOR_NUMERICAL_OBLIGATION_DESCRIPTOR"
DIGEST_ENV = "IPFS_SUPERVISOR_NUMERICAL_OBLIGATION_DESCRIPTOR_SHA256"
DESCRIPTOR_SCHEMA = "supervisor-numerical-obligation-descriptor@1"
REQUEST_SCHEMA = "supervisor-numerical-obligation-request@1"
RESPONSE_SCHEMA = "supervisor-numerical-obligation-response@1"
MAX_FRAME_BYTES = 8192
MIN_PAGE_BYTES = 1024
MAX_SESSION_SECONDS = 1800
REASON_CODES = frozenset({"invalid_request", "peer_refused", "stale_context",
                         "budget_exhausted", "session_closed", "retrieval_refused"})
_DESCRIPTOR_FIELDS = frozenset({"schema", "session_id", "owner_uid", "worker_uid",
    "owner_pid", "owner_start_time_ticks", "workspace", "task_id", "task_cid",
    "context_bundle_sha256", "context_receipt_sha256", "source_tree_id",
    "boundary_manifest_sha256", "socket_path", "socket_device", "socket_inode",
    "issued_monotonic_ns", "expires_monotonic_ns", "max_requests",
    "max_total_response_bytes", "proof_authority", "execution_authority", "completion_authority"})


class NumericalObligationClientError(ValueError):
    """A public protocol or current peer binding was refused."""


def _need(condition, message):
    if not condition:
        raise NumericalObligationClientError(message)


def canonical_json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise NumericalObligationClientError("invalid JSON value") from exc


def strict_json_loads(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            _need(key not in value, "duplicate JSON field")
            value[key] = item
        return value
    def finite_float(text):
        value = float(text)
        _need(math.isfinite(value), "nonfinite JSON number")
        return value
    try:
        if type(raw) is bytes:
            raw = raw.decode("utf-8")
        _need(type(raw) is str and len(raw.encode("utf-8")) <= MAX_FRAME_BYTES,
              "bounded JSON input required")
        return json.loads(raw, object_pairs_hook=pairs, parse_float=finite_float,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise NumericalObligationClientError("invalid strict JSON") from exc


def send_frame(channel, value, *, max_bytes=MAX_FRAME_BYTES):
    _need(type(max_bytes) is int and 2 <= max_bytes <= MAX_FRAME_BYTES, "invalid frame bound")
    raw = canonical_json(value)
    _need(2 <= len(raw) <= max_bytes, "frame exceeds byte bound")
    channel.sendall(len(raw).to_bytes(4, "big") + raw)


def _exact(channel, size):
    chunks = []
    while size:
        part = channel.recv(size)
        _need(bool(part), "incomplete frame")
        chunks.append(part)
        size -= len(part)
    return b"".join(chunks)


def receive_frame(channel, *, max_bytes=MAX_FRAME_BYTES):
    _need(type(max_bytes) is int and 2 <= max_bytes <= MAX_FRAME_BYTES, "invalid frame bound")
    size = int.from_bytes(_exact(channel, 4), "big")
    _need(2 <= size <= max_bytes, "frame exceeds byte bound")
    raw = _exact(channel, size)
    result = strict_json_loads(raw)
    _need(type(result) is dict and canonical_json(result) == raw, "canonical object frame required")
    return result


def _integer(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def _text(value, maximum=4096):
    return (type(value) is str and 0 < len(value.encode("utf-8")) <= maximum
            and not any(char in value for char in ("\x00", "\n", "\r")))


def _hex(value, length=64):
    return type(value) is str and re.fullmatch("[0-9a-f]{%d}" % length, value) is not None


def process_identity(pid):
    """Observe kernel PID birth and effective UID, rejecting dead processes."""
    _need(_integer(pid, 1, (1 << 31) - 1), "invalid process identity")
    root = Path("/proc") / str(pid)
    def facts():
        with (root / "stat").open("rb") as stream:
            raw = stream.read(16385)
        _need(len(raw) <= 16384, "process stat exceeds bound")
        end = raw.rfind(b")")
        fields = raw[end + 2:].split()
        _need(end > 0 and raw.split(b" ", 1)[0] == str(pid).encode()
              and len(fields) >= 20 and fields[0] not in {b"Z", b"X", b"x"},
              "live process identity required")
        return int(fields[1]), int(fields[19])
    parent, ticks = facts()
    with (root / "status").open("rb") as stream:
        status = stream.read(131073)
    _need(len(status) <= 131072, "process status exceeds bound")
    uids = [line.split()[1:] for line in status.splitlines() if line.startswith(b"Uid:")]
    _need(len(uids) == 1 and len(uids[0]) == 4 and facts() == (parent, ticks)
          and ticks > 0, "process identity changed")
    return {"pid": pid, "parent_pid": parent, "uid": int(uids[0][1]), "start_time_ticks": ticks}


def validate_descriptor(value):
    """Validate the public descriptor shape; filesystem checks are separate."""
    _need(type(value) is dict and set(value) == _DESCRIPTOR_FIELDS
          and value["schema"] == DESCRIPTOR_SCHEMA, "invalid descriptor schema")
    _need(_hex(value["session_id"], 32), "invalid session identity")
    for name in ("context_bundle_sha256", "context_receipt_sha256", "boundary_manifest_sha256"):
        _need(_hex(value[name]), "invalid descriptor digest")
    for name in ("task_id", "task_cid", "source_tree_id"):
        _need(_text(value[name], 512), "invalid descriptor identity")
    for name in ("workspace", "socket_path"):
        _need(_text(value[name]) and Path(value[name]).is_absolute()
              and str(Path(value[name])) == value[name] and ".." not in Path(value[name]).parts,
              "canonical descriptor path required")
    for name in ("owner_uid", "worker_uid", "owner_pid"):
        _need(_integer(value[name], 1, (1 << 31) - 1), "invalid descriptor process")
    _need(value["owner_uid"] != value["worker_uid"], "separate worker identity required")
    for name in ("owner_start_time_ticks", "socket_inode"):
        _need(_integer(value[name], 1, (1 << 64) - 1), "invalid descriptor kernel identity")
    _need(_integer(value["socket_device"], 0, (1 << 64) - 1), "invalid socket device")
    issued, expires = value["issued_monotonic_ns"], value["expires_monotonic_ns"]
    _need(_integer(issued, 1, (1 << 63) - 1) and _integer(expires, 1, (1 << 63) - 1)
          and 0 < expires-issued <= MAX_SESSION_SECONDS * 1_000_000_000,
          "invalid session lifetime")
    _need(_integer(value["max_requests"], 1, 64)
          and _integer(value["max_total_response_bytes"], MIN_PAGE_BYTES, 524288),
          "invalid session budget")
    _need(all(value[name] is False for name in
          ("proof_authority", "execution_authority", "completion_authority")),
          "descriptor grants no authority")
    return value


def _protected_parent(parent, owner):
    _need(parent.resolve(strict=True) == parent, "canonical descriptor parent required")
    for path in (parent, *parent.parents):
        info = path.lstat()
        _need(stat.S_ISDIR(info.st_mode) and info.st_uid in (0, owner)
              and not info.st_mode & 0o022 and not os.access(path, os.W_OK),
              "descriptor ancestor is not owner protected")
    info = parent.lstat()
    _need(info.st_uid == owner and stat.S_IMODE(info.st_mode) == 0o750,
          "exact descriptor directory permissions required")
    return info


def _socket_identity(value, parent, group):
    path = Path(value["socket_path"])
    _need(path.parent == parent and path.resolve(strict=True) == path,
          "socket must share the protected descriptor parent")
    info = path.lstat()
    _need(stat.S_ISSOCK(info.st_mode) and info.st_uid == value["owner_uid"]
          and info.st_gid == group and stat.S_IMODE(info.st_mode) == 0o660
          and (info.st_dev, info.st_ino) == (value["socket_device"], value["socket_inode"]),
          "socket identity differs")


def load_descriptor(path, digest, *, workspace=None):
    _need(_hex(digest), "descriptor digest required")
    path = Path(path)
    _need(path.is_absolute() and path.resolve(strict=True) == path, "canonical descriptor path required")
    initial = path.lstat()
    _need(stat.S_ISREG(initial.st_mode) and initial.st_uid != os.geteuid(),
          "separate owner regular descriptor required")
    parent = _protected_parent(path.parent, initial.st_uid)
    def snapshot(info):
        return tuple(getattr(info, name) for name in ("st_dev", "st_ino", "st_mode",
            "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"))
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and 2 <= before.st_size <= MAX_FRAME_BYTES, "bounded regular descriptor required")
        raw = os.read(fd, MAX_FRAME_BYTES + 1)
        after = os.fstat(fd)
        _need(snapshot(initial) == snapshot(before) == snapshot(after) == snapshot(path.lstat())
              and hashlib.sha256(raw).hexdigest() == digest, "descriptor bytes changed")
    finally:
        os.close(fd)
    value = validate_descriptor(strict_json_loads(raw))
    _need(raw == canonical_json(value), "canonical descriptor bytes required")
    _need(os.getuid() == os.geteuid() == value["worker_uid"], "selected worker UID required")
    _need(parent.st_uid == value["owner_uid"], "descriptor parent owner differs")
    _need(before.st_uid == value["owner_uid"] and before.st_gid == parent.st_gid
          and stat.S_IMODE(before.st_mode) == 0o440 and not os.access(path, os.W_OK),
          "descriptor ownership or permissions differ")
    current = Path.cwd()
    allocated = Path(workspace) if workspace is not None else current
    _need(allocated.is_absolute() and allocated.resolve(strict=True) == allocated
          and current == allocated and str(allocated) == value["workspace"],
          "descriptor belongs to another allocated workspace")
    _need(value["issued_monotonic_ns"] <= time.monotonic_ns() <= value["expires_monotonic_ns"],
          "session expired or not yet issued")
    identity = process_identity(value["owner_pid"])
    _need(identity["uid"] == value["owner_uid"]
          and identity["start_time_ticks"] == value["owner_start_time_ticks"], "owner process changed")
    _socket_identity(value, path.parent, parent.st_gid)
    return value


def validate_request(value):
    _need(type(value) is dict and set(value) == {"schema", "session_id", "request_id", "query"}
          and value["schema"] == REQUEST_SCHEMA and _hex(value["session_id"], 32)
          and _hex(value["request_id"], 32), "invalid request envelope")
    query = value["query"]
    _need(type(query) is dict and set(query) == {"path", "function_id", "kind", "cursor", "limit", "max_bytes"},
          "invalid query schema")
    path = query["path"]
    _need(_text(path) and not Path(path).is_absolute() and str(Path(path)) == path
          and path != "." and ".." not in Path(path).parts, "canonical selected source path required")
    _need(all(query[key] is None or _text(query[key], 512) for key in ("function_id", "kind"))
          and (query["cursor"] is None or type(query["cursor"]) is dict), "invalid query filters")
    if query["cursor"] is not None:
        cursor = query["cursor"]
        _need(set(cursor) == {"context_receipt_sha256", "cursor"}
              and _hex(cursor["context_receipt_sha256"]) and type(cursor["cursor"]) is dict,
              "invalid owner cursor")
        inner = cursor["cursor"]
        _need(set(inner) == {"schema", "request_sha256", "offset"}
              and inner["schema"] == "numerical-obligation-cursor@1"
              and _hex(inner["request_sha256"]) and _integer(inner["offset"], 1, (1 << 31) - 1),
              "invalid numerical cursor")
    _need(_integer(query["limit"], 1, 8) and _integer(query["max_bytes"], MIN_PAGE_BYTES, MAX_FRAME_BYTES),
          "invalid query budget")
    _need(len(canonical_json(value)) <= MAX_FRAME_BYTES, "request exceeds bound")
    return value


def validate_response(value, *, request, descriptor=None):
    validate_request(request)
    _need(type(value) is dict and set(value) == {"schema", "session_id", "request_id", "ok", "page", "reason_code"}
          and value["schema"] == RESPONSE_SCHEMA and type(value["ok"]) is bool
          and value["session_id"] == request["session_id"] and value["request_id"] == request["request_id"],
          "response differs from request")
    _need(len(canonical_json(value)) <= request["query"]["max_bytes"], "response exceeds query byte bound")
    if not value["ok"]:
        _need(value["page"] is None and type(value["reason_code"]) is str
              and value["reason_code"] in REASON_CODES, "invalid refusal response")
        return value
    page = value["page"]
    _need(type(page) is dict and value["reason_code"] is None
          and page.get("schema") == "supervisor-numerical-obligation-page@1",
          "invalid page response")
    _need(all(page.get(key) is False for key in ("proof_authority", "execution_authority", "completion_authority",
              "formalization_authority", "source_semantics_verified", "whole_program_semantics_verified", "source_executed"))
          and type(page.get("source_obligations_discharged")) is int
          and page["source_obligations_discharged"] == 0, "retrieval grants no proof authority")
    _need(type(page.get("binding")) is dict and page["binding"].get("path") == request["query"]["path"],
          "page source differs from query")
    query = page.get("query")
    _need(type(query) is dict and set(query) == {"function_id", "kind", "limit", "max_bytes"}
          and all(query[key] == request["query"][key] for key in ("function_id", "kind"))
          and type(query["limit"]) is int and query["limit"] == request["query"]["limit"]
          and _integer(query["max_bytes"], 1, request["query"]["max_bytes"]),
          "page selection differs from query")
    _need(type(page.get("rows")) is list and len(page["rows"]) <= query["limit"]
          and type(page.get("returned_count")) is int and page["returned_count"] == len(page["rows"]),
          "page row accounting differs")
    if descriptor is not None:
        _need(page.get("context_receipt_sha256") == descriptor["context_receipt_sha256"]
              and page.get("source_tree_id") == descriptor["source_tree_id"], "page owner context differs")
    return value


def _verify_peer(channel, descriptor):
    pid, uid, _gid = struct.unpack("3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
    _need(pid == descriptor["owner_pid"] and uid == descriptor["owner_uid"], "server peer differs")
    identity = process_identity(pid)
    _need(identity["uid"] == uid and identity["start_time_ticks"] == descriptor["owner_start_time_ticks"],
          "server process birth differs")


def request_page(descriptor_path, descriptor_sha256, *, path, function_id=None, kind=None,
                 cursor=None, limit=8, max_bytes=8192, workspace=None):
    descriptor = load_descriptor(descriptor_path, descriptor_sha256, workspace=workspace)
    request = validate_request(dict(schema=REQUEST_SCHEMA, session_id=descriptor["session_id"],
        request_id=uuid.uuid4().hex, query=dict(path=path, function_id=function_id, kind=kind,
                                             cursor=cursor, limit=limit, max_bytes=max_bytes)))
    remaining = (descriptor["expires_monotonic_ns"] - time.monotonic_ns()) / 1_000_000_000
    _need(remaining > 0, "session expired")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(min(30.0, remaining))
        channel.connect(descriptor["socket_path"])
        _verify_peer(channel, descriptor)
        _socket_identity(descriptor, Path(descriptor_path).parent, Path(descriptor_path).parent.stat().st_gid)
        send_frame(channel, request)
        response = receive_frame(channel, max_bytes=max_bytes)
        _verify_peer(channel, descriptor)
        _need(time.monotonic_ns() <= descriptor["expires_monotonic_ns"], "session expired")
    return validate_response(response, request=request, descriptor=descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(DESCRIPTOR_FLAG, default=os.environ.get(DESCRIPTOR_ENV))
    parser.add_argument(DIGEST_FLAG, default=os.environ.get(DIGEST_ENV))
    parser.add_argument("--path", required=True)
    parser.add_argument("--function-id")
    parser.add_argument("--kind")
    parser.add_argument("--cursor")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-bytes", type=int, default=8192)
    args = parser.parse_args(argv)
    try:
        _need(args.numerical_obligation_descriptor is not None
              and args.numerical_obligation_descriptor_sha256 is not None, "descriptor required")
        result = request_page(args.numerical_obligation_descriptor,
            args.numerical_obligation_descriptor_sha256, path=args.path,
            function_id=args.function_id, kind=args.kind,
            cursor=strict_json_loads(args.cursor) if args.cursor is not None else None,
            limit=args.limit, max_bytes=args.max_bytes)
        print(canonical_json(result).decode("utf-8"))
        return 0 if result["ok"] else 2
    except (OSError, ValueError, TypeError, RuntimeError):
        print('{"reason_code":"transport_error"}', file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
