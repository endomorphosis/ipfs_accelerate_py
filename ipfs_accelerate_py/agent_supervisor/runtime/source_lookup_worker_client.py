"""Bounded exact-source client for an explicitly admitted coding worker.

The public descriptor routes a read-only request to the original owner; its
private native grant never enters this process. The deployed ``bin/source-lookup``
helper supplies protected source imports with isolated runtime Python. A direct
isolated invocation of this file is supported only in that protected deployment.
Ordinary Terminal tasks do not receive this channel automatically. Required
facts remain inline; retrieval grants no proof, edit, omission or completion
authority and does not establish a token saving.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import sys
import time
import uuid

if not __package__:
    # An isolated absolute-file invocation ignores PYTHONPATH. Its bootstrap
    # derives only the root-owned deployed trees, never a payload/env locator.
    try:
        _file = Path(__file__).absolute()
        _source = _file.parents[3]
        _datasets = _source.parent / "datasets"
        for _path in (_file, _source, _datasets):
            if _path.resolve(strict=True) != _path:
                raise ValueError("exact protected source required")
            for _parent in (_path, *_path.parents):
                _info = _parent.lstat()
                if _info.st_uid != 0 or _info.st_mode & 0o022:
                    raise ValueError("deployer protection required")
        if _source.name != "source" or not _source.is_dir() or not _datasets.is_dir():
            raise ValueError("deployed source roots required")
        sys.path[:0] = [str(_source), str(_datasets)]
        from ipfs_accelerate_py.agent_supervisor.runtime import numerical_obligation_client as _numeric
    except (OSError, ValueError, ImportError):
        raise SystemExit("source_worker_bootstrap_refused") from None
else:
    from . import numerical_obligation_client as _numeric

DESCRIPTOR_FLAG = "--source-lookup-descriptor"
DIGEST_FLAG = "--source-lookup-descriptor-sha256"
DESCRIPTOR_ENV = "IPFS_SUPERVISOR_SOURCE_LOOKUP_DESCRIPTOR"
DIGEST_ENV = "IPFS_SUPERVISOR_SOURCE_LOOKUP_DESCRIPTOR_SHA256"
DESCRIPTOR_SCHEMA = "supervisor-source-worker-descriptor@1"
REQUEST_SCHEMA = "supervisor-source-worker-request@1"
RESPONSE_SCHEMA = "supervisor-source-worker-response@1"
RESULT_SCHEMA = "supervisor-source-worker-result@1"
SOURCE_REQUEST_SCHEMA = "supervisor-current-source-request@1"
MAX_REQUEST_BYTES = 8192
MAX_FRAME_BYTES = 65536
MAX_RESPONSE_BYTES = 32768
MAX_TOTAL_RESPONSE_BYTES = 524288
MAX_REQUESTS = 64
MAX_SESSION_SECONDS = _numeric.MAX_SESSION_SECONDS
REASON_CODES = _numeric.REASON_CODES
canonical_json = _numeric.canonical_json
process_identity = _numeric.process_identity
_protected_parent = _numeric._protected_parent
_socket_identity = _numeric._socket_identity
_verify_peer = _numeric._verify_peer
_hex = _numeric._hex
_integer = _numeric._integer
_text = _numeric._text
_DESCRIPTOR_FIELDS = frozenset({"schema", "session_id", "owner_uid", "worker_uid",
    "owner_pid", "owner_start_time_ticks", "workspace", "task_cid", "task_revision",
    "repository_id", "context_cid", "source_binding_cid", "prompt_sha256",
    "boundary_manifest_sha256", "socket_path", "socket_device", "socket_inode",
    "issued_monotonic_ns", "expires_monotonic_ns", "max_requests",
    "max_total_response_bytes", "max_response_bytes", "max_slice_bytes", "native_grant_id", "source_refs",
    "proof_authority", "execution_authority", "completion_authority"})
_PAYLOAD_FIELDS = frozenset({"schema", "binding_cid", "raw_path_hex", "source_cid",
                           "offset", "max_bytes", "continuation"})
_RESPONSE_FIELDS = frozenset({"schema", "session_id", "request_id", "ok", "result",
                            "reason_code", "response_bytes", "response_wire_bytes"})
_RESULT_FIELDS = frozenset({"schema", "request_cid", "raw_path_hex", "source_cid",
    "slice_cid", "encoding", "content", "offset", "end_offset", "source_bytes",
    "source_complete", "next_continuation", "unresolved_frontier",
    "working_source_verified", "source_check_profile", "repository_complete",
    "read_only", "completion_authority", "proof_authority", "edit_authority",
    "omission_authority", "source_binding_cid", "backend_response_sha256",
    "backend_response_bytes", "backend_response_wire_bytes"})


class SourceLookupWorkerClientError(ValueError):
    """A public protocol or current worker binding was refused."""


def _need(condition, message):
    if not condition:
        raise SourceLookupWorkerClientError(message)


def _cid(value, codec="dag-json"):
    from ipfs_datasets_py.logic.software_contracts.content import validate_cid
    _need(type(value) is str and len(value) <= 512, "bounded native CID required")
    return validate_cid(value, codecs=(codec,))


def _raw_path(raw):
    _need(type(raw) is str and 0 < len(raw) <= MAX_REQUEST_BYTES and len(raw) % 2 == 0
          and re.fullmatch("[0-9a-f]+", raw) is not None, "invalid raw source path")
    decoded = bytes.fromhex(raw)
    _need(b"\0" not in decoded and all(part and part not in (b".", b"..", b".git")
          for part in decoded.split(b"/")), "unsafe raw source path")
    return decoded


def _source_ref(descriptor, payload):
    _need(payload["max_bytes"] <= descriptor["max_slice_bytes"],
          "source byte bound exceeds admitted slice limit")
    rows = [row for row in descriptor["source_refs"] if row["raw_path_hex"] == payload["raw_path_hex"]]
    _need(len(rows) == 1 and rows[0]["source_cid"] == payload["source_cid"],
          "source request outside declared scope")
    return rows[0]


def strict_json_loads(raw, *, max_bytes=MAX_FRAME_BYTES):
    """Source-specific strict JSON; the numerical client's 8 KiB cap stays intact."""
    _need(_integer(max_bytes, 2, MAX_FRAME_BYTES), "invalid JSON byte bound")
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    def number(text):
        value = float(text)
        _need(math.isfinite(value), "nonfinite JSON number")
        return value
    try:
        if type(raw) is bytes:
            raw = raw.decode("utf-8")
        _need(type(raw) is str and len(raw.encode("utf-8")) <= max_bytes,
              "bounded JSON input required")
        return json.loads(raw, object_pairs_hook=pairs, parse_float=number,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise SourceLookupWorkerClientError("invalid strict JSON") from exc


def send_frame(channel, value, *, max_bytes=MAX_FRAME_BYTES):
    _need(_integer(max_bytes, 2, MAX_FRAME_BYTES), "invalid frame bound")
    raw = canonical_json(value)
    _need(2 <= len(raw) <= max_bytes, "frame exceeds byte bound")
    channel.sendall(len(raw).to_bytes(4, "big") + raw)


def receive_frame(channel, *, max_bytes=MAX_FRAME_BYTES):
    _need(_integer(max_bytes, 2, MAX_FRAME_BYTES), "invalid frame bound")
    size = int.from_bytes(_numeric._exact(channel, 4), "big")
    _need(2 <= size <= max_bytes, "frame exceeds byte bound")
    raw = _numeric._exact(channel, size)
    result = strict_json_loads(raw, max_bytes=max_bytes)
    _need(type(result) is dict and canonical_json(result) == raw, "canonical object frame required")
    return result


def validate_descriptor(value):
    """Validate public metadata only; load_descriptor also checks the deployed peer."""
    _need(type(value) is dict and set(value) == _DESCRIPTOR_FIELDS
          and value["schema"] == DESCRIPTOR_SCHEMA, "invalid descriptor schema")
    _need(len(canonical_json(value)) <= MAX_REQUEST_BYTES, "descriptor exceeds byte bound")
    _need(_hex(value["session_id"], 32), "invalid session identity")
    for name in ("prompt_sha256", "boundary_manifest_sha256"):
        _need(_hex(value[name]), "invalid descriptor digest")
    for name in ("task_cid", "repository_id", "native_grant_id"):
        _need(_text(value[name], 4096), "invalid descriptor identity")
    for name in ("context_cid", "source_binding_cid"):
        _cid(value[name])
    refs = value["source_refs"]
    _need(type(refs) is list and 1 <= len(refs) <= 32, "bounded complete source refs required")
    paths = []
    for row in refs:
        _need(type(row) is dict and set(row) == {"raw_path_hex", "source_cid", "size_bytes", "git_mode"},
              "invalid source reference fields")
        _raw_path(row["raw_path_hex"])
        _cid(row["source_cid"], "raw")
        _need(_integer(row["size_bytes"], 0, 128 * 1024**2)
              and type(row["git_mode"]) is str and row["git_mode"] in ("100644", "100755"),
              "source ref must be a bounded regular file")
        paths.append(row["raw_path_hex"])
    _need(paths == sorted(set(paths)), "source refs must be unique and sorted")
    _need(_integer(value["task_revision"], 1, (1 << 63) - 1), "invalid task revision")
    for name in ("workspace", "socket_path"):
        path = Path(value[name]) if type(value[name]) is str else None
        _need(_text(value[name]) and path.is_absolute() and str(path) == value[name]
              and ".." not in path.parts, "canonical descriptor path required")
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
    _need(type(value["max_requests"]) is int and value["max_requests"] == MAX_REQUESTS
          and type(value["max_total_response_bytes"]) is int
          and value["max_total_response_bytes"] == MAX_TOTAL_RESPONSE_BYTES
          and type(value["max_response_bytes"]) is int
          and value["max_response_bytes"] == MAX_RESPONSE_BYTES, "invalid session budget")
    _need(_integer(value["max_slice_bytes"], 1, 16384), "invalid admitted source slice bound")
    _need(all(value[name] is False for name in
              ("proof_authority", "execution_authority", "completion_authority")),
          "descriptor grants no authority")
    return strict_json_loads(canonical_json(value))


def load_descriptor(path, expected_sha256, *, workspace=None):
    _need(_hex(expected_sha256), "descriptor digest required")
    path = Path(path)
    _need(path.is_absolute() and path.resolve(strict=True) == path,
          "canonical descriptor path required")
    initial = path.lstat()
    _need(stat.S_ISREG(initial.st_mode) and initial.st_uid != os.geteuid()
          and path.parent.name == "worker-observations", "protected owner descriptor required")
    parent = _protected_parent(path.parent, initial.st_uid)
    def snapshot(info):
        return tuple(getattr(info, name) for name in ("st_dev", "st_ino", "st_mode",
            "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"))
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and 2 <= before.st_size <= MAX_REQUEST_BYTES, "bounded regular descriptor required")
        raw = os.read(fd, MAX_REQUEST_BYTES + 1)
        _need(snapshot(initial) == snapshot(before) == snapshot(os.fstat(fd))
              == snapshot(path.lstat()) and hashlib.sha256(raw).hexdigest() == expected_sha256,
              "descriptor bytes changed")
    finally:
        os.close(fd)
    value = validate_descriptor(strict_json_loads(raw, max_bytes=MAX_REQUEST_BYTES))
    _need(raw == canonical_json(value), "canonical descriptor bytes required")
    _need(os.getuid() == os.geteuid() == value["worker_uid"], "selected worker UID required")
    _need(parent.st_uid == before.st_uid == value["owner_uid"]
          and before.st_gid == parent.st_gid and stat.S_IMODE(before.st_mode) == 0o440
          and not os.access(path, os.W_OK), "descriptor ownership or permissions differ")
    current = Path.cwd()
    allocated = Path(workspace) if workspace is not None else current
    _need(allocated.is_absolute() and allocated.resolve(strict=True) == allocated
          and current == allocated and str(allocated) == value["workspace"],
          "descriptor belongs to another allocated workspace")
    from ipfs_accelerate_py.agent_supervisor.runtime.container_worker_boundary import verify_container_worker_boundary
    boundary = verify_container_worker_boundary(artifact=path.parent.parent / "container-boundary.json",
        expected_sha256=value["boundary_manifest_sha256"], workspace=allocated)
    _need(boundary["owner_uid"] == value["owner_uid"]
          and boundary["worker_uid"] == value["worker_uid"], "descriptor boundary identity differs")
    _need(value["issued_monotonic_ns"] <= time.monotonic_ns() <= value["expires_monotonic_ns"],
          "session expired or not yet issued")
    identity = process_identity(value["owner_pid"])
    _need(identity["uid"] == value["owner_uid"]
          and identity["start_time_ticks"] == value["owner_start_time_ticks"], "owner process changed")
    _socket_identity(value, path.parent, parent.st_gid)
    return value


def validate_request(payload):
    _need(type(payload) is dict and set(payload) == _PAYLOAD_FIELDS
          and payload["schema"] == SOURCE_REQUEST_SCHEMA, "invalid source request schema")
    _need(len(canonical_json(payload)) <= MAX_REQUEST_BYTES, "source request exceeds byte bound")
    _cid(payload["binding_cid"])
    _cid(payload["source_cid"], "raw")
    _raw_path(payload["raw_path_hex"])
    _need(_integer(payload["offset"], 0, 128 * 1024**2)
          and _integer(payload["max_bytes"], 1, 16384), "invalid source byte range")
    if payload["continuation"] is not None:
        _cid(payload["continuation"])
    _need((payload["offset"] == 0) == (payload["continuation"] is None),
          "continuation required exactly for later source ranges")
    return strict_json_loads(canonical_json(payload), max_bytes=MAX_REQUEST_BYTES)


def validate_worker_request(value):
    _need(type(value) is dict and set(value) == {"schema", "session_id", "request_id", "payload"}
          and value["schema"] == REQUEST_SCHEMA and _hex(value["session_id"], 32)
          and _hex(value["request_id"], 32), "invalid worker request envelope")
    payload = validate_request(value["payload"])
    _need(len(canonical_json(value)) <= MAX_REQUEST_BYTES, "complete worker request exceeds byte bound")
    return dict(schema=REQUEST_SCHEMA, session_id=value["session_id"],
                request_id=value["request_id"], payload=payload)


def validate_response(value, *, descriptor, payload, request_id):
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes, cid_for_structured
    descriptor, payload = validate_descriptor(descriptor), validate_request(payload)
    _need(_hex(request_id, 32) and payload["binding_cid"] == descriptor["source_binding_cid"],
          "request source admission differs")
    source_ref = _source_ref(descriptor, payload)
    _need(type(value) is dict and set(value) == _RESPONSE_FIELDS
          and value["schema"] == RESPONSE_SCHEMA and type(value["ok"]) is bool
          and value["session_id"] == descriptor["session_id"]
          and value["request_id"] == request_id, "response differs from request")
    size = len(canonical_json(value)) + 1
    _need(type(value["response_bytes"]) is int and value["response_bytes"] == size
          and type(value["response_wire_bytes"]) is int and value["response_wire_bytes"] == size + 3
          and size <= descriptor["max_response_bytes"], "complete worker response accounting differs")
    if not value["ok"]:
        _need(value["result"] is None and type(value["reason_code"]) is str
              and value["reason_code"] in REASON_CODES, "invalid refusal response")
        return strict_json_loads(canonical_json(value))
    result = value["result"]
    _need(type(result) is dict and set(result) == _RESULT_FIELDS
          and result["schema"] == RESULT_SCHEMA and value["reason_code"] is None,
          "invalid source result fields")
    _need(result["source_binding_cid"] == payload["binding_cid"]
          and result["request_cid"] == cid_for_structured(payload)
          and result["raw_path_hex"] == payload["raw_path_hex"]
          and result["source_cid"] == payload["source_cid"], "source response selection differs")
    _need(type(result["content"]) is str, "source content must be encoded text")
    if result["encoding"] == "utf-8":
        data = result["content"].encode("utf-8")
    elif result["encoding"] == "base64":
        data = base64.b64decode(result["content"], validate=True)
        _need(base64.b64encode(data).decode("ascii") == result["content"], "noncanonical source encoding")
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            raise SourceLookupWorkerClientError("unnecessary source encoding")
    else:
        raise SourceLookupWorkerClientError("invalid source encoding")
    for name in ("offset", "end_offset", "source_bytes"):
        _need(_integer(result[name], 0, 128 * 1024**2), "invalid source count")
    _need(result["offset"] == payload["offset"]
          and result["end_offset"] == result["offset"] + len(data)
          and result["end_offset"] <= result["source_bytes"] == source_ref["size_bytes"]
          and len(data) <= payload["max_bytes"] and (data or result["source_bytes"] == 0)
          and result["slice_cid"] == cid_for_bytes(data), "source byte range differs")
    complete = result["end_offset"] == result["source_bytes"]
    if complete and result["offset"] == 0:
        _need(cid_for_bytes(data) == payload["source_cid"], "complete source CID differs")
    continuation = None if complete else cid_for_structured(dict(schema="supervisor-source-continuation@1",
        binding_cid=payload["binding_cid"], caller_grant_id=descriptor["native_grant_id"],
        raw_path_hex=payload["raw_path_hex"], source_cid=payload["source_cid"],
        offset=result["end_offset"], max_bytes=payload["max_bytes"]))
    frontier = [] if complete else [dict(kind="remaining-source-bytes", offset=result["end_offset"],
        end_offset=result["source_bytes"], source_cid=payload["source_cid"])]
    _need(result["source_complete"] is complete and result["next_continuation"] == continuation
          and canonical_json(result["unresolved_frontier"]) == canonical_json(frontier),
          "source response frontier differs")
    _need(all(result[name] is False for name in ("repository_complete", "completion_authority",
          "proof_authority", "edit_authority", "omission_authority"))
          and result["working_source_verified"] is True and result["read_only"] is True
          and result["source_check_profile"] == "clean-committed-regular@1", "invalid source authority flags")
    _need(_hex(result["backend_response_sha256"])
          and _integer(result["backend_response_bytes"], 2, MAX_FRAME_BYTES)
          and type(result["backend_response_wire_bytes"]) is int
          and result["backend_response_wire_bytes"] == result["backend_response_bytes"] + 3,
          "invalid native backend accounting")
    return strict_json_loads(canonical_json(value))


def request_source(*, path, expected_sha256, payload, workspace=None):
    descriptor = load_descriptor(path, expected_sha256, workspace=workspace)
    payload = validate_request(payload)
    _need(payload["binding_cid"] == descriptor["source_binding_cid"], "foreign source binding")
    source_ref = _source_ref(descriptor, payload)
    _need(payload["offset"] <= source_ref["size_bytes"], "source offset exceeds declared file")
    request_id = uuid.uuid4().hex
    request = validate_worker_request(dict(schema=REQUEST_SCHEMA, session_id=descriptor["session_id"],
                   request_id=request_id, payload=payload))
    remaining = (descriptor["expires_monotonic_ns"] - time.monotonic_ns()) / 1_000_000_000
    _need(remaining > 0, "session expired")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(min(30.0, remaining))
        channel.connect(descriptor["socket_path"])
        _verify_peer(channel, descriptor)
        _socket_identity(descriptor, Path(path).parent, Path(path).parent.stat().st_gid)
        send_frame(channel, request, max_bytes=MAX_REQUEST_BYTES)
        response = receive_frame(channel, max_bytes=descriptor["max_response_bytes"])
        _verify_peer(channel, descriptor)
    _need(load_descriptor(path, expected_sha256, workspace=workspace) == descriptor,
          "source descriptor changed during request")
    return validate_response(response, descriptor=descriptor, payload=payload, request_id=request_id)


def main(argv=None):
    class PublicParser(argparse.ArgumentParser):
        def error(self, message):
            raise SourceLookupWorkerClientError("invalid CLI arguments")
    try:
        parser = PublicParser(description=__doc__)
        parser.add_argument(DESCRIPTOR_FLAG, default=os.environ.get(DESCRIPTOR_ENV))
        parser.add_argument(DIGEST_FLAG, default=os.environ.get(DIGEST_ENV))
        args = parser.parse_args(argv)
        _need(args.source_lookup_descriptor is not None and args.source_lookup_descriptor_sha256 is not None,
              "descriptor required")
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        payload = strict_json_loads(raw, max_bytes=MAX_REQUEST_BYTES)
        result = request_source(path=args.source_lookup_descriptor,
            expected_sha256=args.source_lookup_descriptor_sha256, payload=payload)
        sys.stdout.buffer.write(canonical_json(result) + b"\n")
        sys.stdout.buffer.flush()
        return 0 if result["ok"] else 2
    except (OSError, ValueError, TypeError, RuntimeError, ImportError):
        print('{"reason_code":"transport_error"}', file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
