"""Bounded, peer-authenticated assignment metadata for one launched worker.

This owner-held library neither launches the child nor delegates database or
source authority. Its fixed client observes only the existing native handoff.
Kernel identity is observed independently of request JSON. A response is not a
world packet, an execution permit, a task result, or historical activation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
import uuid

from .numerical_obligation_client import process_identity

DESCRIPTOR_SCHEMA = "repository-world-worker-observer-descriptor@1"
REQUEST_SCHEMA = "repository-world-worker-observer-request@1"
RESPONSE_SCHEMA = "repository-world-worker-observer-response@1"
AUDIT_SCHEMA = "repository-world-worker-observer-audit@1"
SCOPE = "authenticated_worker_assignment_metadata_only"
MAX_REQUESTS = 16
MAX_FRAME_BYTES = 8192
MAX_REQUEST_BYTES = 1024
MAX_TOTAL_RESPONSE_BYTES = MAX_REQUESTS * MAX_FRAME_BYTES
MAX_SESSION_SECONDS = 180.0
MAX_COMMAND_BYTES = 8192
FALSE_FIELDS = frozenset({"proof_authority", "execution_authority", "completion_authority",
    "world_dispatch_ready", "compiled_context_verified", "provider_launch", "task_completed",
    "raw_source_exported", "database_access_granted", "worker_workspace_verified"})
REASONS = frozenset({"invalid_request", "replayed_request", "session_expired",
    "worker_not_registered", "peer_refused", "observation_refused", "response_budget_exhausted"})
_NAME = re.compile(r"repository-world-worker-[0-9a-f]{32}")
_HEX = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[0-9a-f]{32}")
_DESCRIPTOR_FIELDS = frozenset({"schema", "scope", "session_id", "endpoint_name",
    "handoff_id", "original_assignment_cid", "prompt_sha256", "prompt_bytes",
    "owner_pid", "owner_uid", "owner_gid", "owner_start_time_ticks", "boot_id",
    "worker_uid", "worker_gid", "worker_command_sha256", "worker_command_bytes",
    "launcher_command_sha256", "launcher_command_bytes",
    "issued_monotonic_ns", "expires_monotonic_ns", "max_requests", "max_frame_bytes",
    "max_total_response_bytes", *FALSE_FIELDS})
_REQUEST_FIELDS = frozenset({"schema", "operation", "session_id", "request_id", "descriptor_sha256"})
_RESPONSE_FIELDS = frozenset({"schema", "scope", "session_id", "request_id", "ok", "reason_code",
    "assignment", "peer", "worker_peer_authenticated", *FALSE_FIELDS})
_PEER_FIELDS = frozenset({"pid", "uid", "gid", "start_time_ticks", "root_pid",
    "root_start_time_ticks", "groups_sha256", "group_count"})


class RepositoryWorldWorkerObserverError(ValueError):
    """Closed observer refusal, without command/source/credential content."""


def _need(condition, code):
    if not condition:
        raise RepositoryWorldWorkerObserverError(code)


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _seconds(value, maximum):
    return type(value) in (int, float) and 0 < value <= maximum


def _wire(value, *, maximum=MAX_FRAME_BYTES):
    pending, nodes, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _need(nodes <= 2048 and depth <= 16, "observer_structure_bound")
        if type(item) is str:
            _need(len(item) <= maximum, "observer_text_bound")
            text_bytes += len(item.encode("utf-8", "strict"))
            _need(text_bytes <= maximum, "observer_text_bound")
        elif type(item) is dict:
            _need(len(item) <= 256 and all(type(k) is str for k in item), "observer_structure_bound")
            pending.extend((v, depth + 1) for pair in item.items() for v in pair)
        elif type(item) in (list, tuple):
            _need(len(item) <= 256, "observer_structure_bound")
            pending.extend((v, depth + 1) for v in item)
        else:
            _need(item is None or type(item) is bool or _integer(item, -(1 << 63), (1 << 63) - 1)
                  or type(item) is float and math.isfinite(item) and abs(item) <= (1 << 63) - 1,
                  "observer_value_invalid")
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8")
    _need(len(raw) <= maximum, "observer_frame_bound")
    return raw


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        _need(key not in value, "observer_duplicate_key")
        value[key] = item
    return value


def _decode(raw, *, maximum):
    _need(type(raw) is bytes and 0 < len(raw) <= maximum, "observer_frame_bound")
    try:
        value = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_unique,
            parse_constant=lambda _value: (_ for _ in ()).throw(RepositoryWorldWorkerObserverError("observer_nonfinite")))
        _need(type(value) is dict and _wire(value, maximum=maximum) == raw,
              "observer_canonical_object_required")
        return value
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise RepositoryWorldWorkerObserverError("observer_invalid_json") from error


def _recv(channel, *, maximum):
    def exact(count):
        parts, remaining = [], count
        while remaining:
            raw = channel.recv(remaining)
            _need(bool(raw), "observer_short_frame")
            parts.append(raw)
            remaining -= len(raw)
        return b"".join(parts)
    count = struct.unpack("!I", exact(4))[0]
    _need(0 < count <= maximum, "observer_frame_bound")
    return exact(count)


def _send(channel, raw):
    _need(type(raw) is bytes and 0 < len(raw) <= MAX_FRAME_BYTES, "observer_frame_bound")
    channel.sendall(struct.pack("!I", len(raw)) + raw)


def _boot_id():
    with Path("/proc/sys/kernel/random/boot_id").open("rb") as stream:
        raw = stream.read(129)
    _need(len(raw) <= 128, "observer_boot_identity_invalid")
    value = raw.decode("ascii", "strict").strip()
    _need(re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value) is not None,
          "observer_boot_identity_invalid")
    return value


def _command(argv):
    _need(type(argv) is tuple and 1 <= len(argv) <= 32
          and all(type(v) is str and v and "\0" not in v and len(v) <= 4096 for v in argv),
          "observer_literal_command_required")
    raw = _wire(list(argv), maximum=MAX_COMMAND_BYTES)
    cmdline = b"".join(v.encode("utf-8", "strict") + b"\0" for v in argv)
    _need(len(cmdline) <= MAX_COMMAND_BYTES, "observer_command_bound")
    return raw, cmdline


def _cmdline(pid):
    with (Path("/proc") / str(pid) / "cmdline").open("rb") as stream:
        raw = stream.read(MAX_COMMAND_BYTES + 1)
    _need(0 < len(raw) <= MAX_COMMAND_BYTES and raw.endswith(b"\0"), "observer_command_unavailable")
    return raw


def _groups(pid):
    before = process_identity(pid)
    with (Path("/proc") / str(pid) / "status").open("rb") as stream:
        raw = stream.read(131073)
    _need(len(raw) <= 131072, "observer_peer_status_bound")
    def values(label, count=None):
        rows = [line.split()[1:] for line in raw.splitlines() if line.startswith(label)]
        _need(len(rows) == 1 and len(rows[0]) <= 64 and (count is None or len(rows[0]) == count)
              and all(v.isdigit() for v in rows[0]), "observer_peer_status_invalid")
        return tuple(int(v) for v in rows[0])
    uids, gids, groups = values(b"Uid:", 4), values(b"Gid:", 4), values(b"Groups:")
    _need(process_identity(pid) == before and len(groups) == len(set(groups)), "observer_peer_birth_changed")
    return before, uids, gids, tuple(sorted(groups))


def validate_worker_observer_descriptor(value, *, expected_sha256):
    _need(type(value) is dict and set(value) == _DESCRIPTOR_FIELDS and value["schema"] == DESCRIPTOR_SCHEMA
          and value["scope"] == SCOPE and all(value[k] is False for k in FALSE_FIELDS),
          "observer_descriptor_invalid")
    _need(type(expected_sha256) is str and _HEX.fullmatch(expected_sha256)
          and _sha(_wire(value)) == expected_sha256, "observer_descriptor_pin_mismatch")
    _need(type(value["session_id"]) is str and _ID.fullmatch(value["session_id"])
          and type(value["endpoint_name"]) is str and _NAME.fullmatch(value["endpoint_name"]),
          "observer_descriptor_identity_invalid")
    for key in ("handoff_id", "original_assignment_cid", "boot_id"):
        _need(type(value[key]) is str and 0 < len(value[key].encode()) <= 512 and "\0" not in value[key],
              "observer_descriptor_identity_invalid")
    for key in ("prompt_sha256", "worker_command_sha256", "launcher_command_sha256"):
        _need(type(value[key]) is str and _HEX.fullmatch(value[key]), "observer_descriptor_digest_invalid")
    for key in ("owner_pid", "owner_uid", "owner_gid", "worker_uid", "worker_gid", "owner_start_time_ticks"):
        _need(_integer(value[key], 1, (1 << 63) - 1), "observer_descriptor_principal_invalid")
    _need(value["owner_uid"] != value["worker_uid"]
          and _integer(value["prompt_bytes"], 1, 2 * 1024 * 1024)
          and _integer(value["worker_command_bytes"], 1, MAX_COMMAND_BYTES)
          and _integer(value["launcher_command_bytes"], 1, MAX_COMMAND_BYTES), "observer_descriptor_principal_invalid")
    issued, expires = value["issued_monotonic_ns"], value["expires_monotonic_ns"]
    _need(_integer(issued, 1, (1 << 63) - 1) and _integer(expires, issued + 1, (1 << 63) - 1)
          and expires - issued <= int(MAX_SESSION_SECONDS * 1e9)
          and value["max_requests"] == MAX_REQUESTS and type(value["max_requests"]) is int
          and value["max_frame_bytes"] == MAX_FRAME_BYTES and type(value["max_frame_bytes"]) is int
          and value["max_total_response_bytes"] == MAX_TOTAL_RESPONSE_BYTES
          and type(value["max_total_response_bytes"]) is int, "observer_descriptor_budget_invalid")
    return _decode(_wire(value), maximum=MAX_FRAME_BYTES)


class RepositoryWorldWorkerObserver:
    """Owner-held socket for one real worker birth; no launch or credentials."""

    def __init__(self, *, handoff, prompt, expected_worker_argv, expected_launcher_argv=None,
                 timeout_seconds=60.0):
        from .repository_world_worker_handoff import NativeRepositoryWorldWorkerHandoff
        _need(type(handoff) is NativeRepositoryWorldWorkerHandoff and _seconds(timeout_seconds, MAX_SESSION_SECONDS),
              "observer_original_handoff_required")
        command, self._worker_command_line = _command(expected_worker_argv)
        expected_launcher_argv = expected_worker_argv if expected_launcher_argv is None else expected_launcher_argv
        launcher, self._command_line = _command(expected_launcher_argv)
        self._handoff = self._original_handoff = handoff
        self._prompt = prompt
        self._expected_argv = expected_launcher_argv
        self._owner = process_identity(os.getpid())
        _need(os.getuid() == os.geteuid() == self._owner["uid"] > 0 and os.getgid() > 0,
              "observer_nonroot_owner_required")
        self._boot = _boot_id()
        initial = NativeRepositoryWorldWorkerHandoff.require_current(handoff)
        self._worker_uid, self._worker_gid = initial.get("worker_uid"), initial.get("worker_gid")
        self._identity = {key: initial.get(key) for key in ("handoff_id", "original_assignment_cid",
            "prompt_sha256", "prompt_bytes", "worker_uid", "worker_gid", "owner_pid", "owner_uid",
            "owner_start_time_ticks", "boot_id")}
        initial = self._handoff_current()
        self._lock = threading.RLock()
        self._process = self._root_child = self._peer_binding = None
        self._seen = set()
        self._accepted = self._responses = self._response_bytes = self._response_wire_bytes = self._refused = 0
        self._stop = threading.Event()
        self._listener = self._thread = None
        self._closed = False
        session = uuid.uuid4().hex
        now = time.monotonic_ns()
        self._descriptor = dict(schema=DESCRIPTOR_SCHEMA, scope=SCOPE, session_id=session,
            endpoint_name="repository-world-worker-" + session, **self._identity,
            owner_gid=os.getgid(), worker_command_sha256=_sha(command), worker_command_bytes=len(command),
            launcher_command_sha256=_sha(launcher), launcher_command_bytes=len(launcher),
            issued_monotonic_ns=now, expires_monotonic_ns=now + int(timeout_seconds * 1e9),
            max_requests=MAX_REQUESTS, max_frame_bytes=MAX_FRAME_BYTES,
            max_total_response_bytes=MAX_TOTAL_RESPONSE_BYTES, **dict.fromkeys(FALSE_FIELDS, False))
        self._descriptor_bytes = _wire(self._descriptor)
        self.descriptor_sha256 = _sha(self._descriptor_bytes)
        validate_worker_observer_descriptor(self._descriptor, expected_sha256=self.descriptor_sha256)
        try:
            self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._listener.bind("\0" + self._descriptor["endpoint_name"])
            self._listener.listen(2)
            self._listener.settimeout(.1)
            self._thread = threading.Thread(target=self._serve, name="world-worker-metadata", daemon=True)
            self._thread.start()
        except BaseException:
            if self._listener is not None:
                self._listener.close()
            raise

    @property
    def descriptor(self):
        return _decode(self._descriptor_bytes, maximum=MAX_FRAME_BYTES)

    def _handoff_current(self):
        from .repository_world_worker_handoff import NativeRepositoryWorldWorkerHandoff
        _need(self._handoff is self._original_handoff and process_identity(os.getpid()) == self._owner
              and os.getuid() == os.geteuid() == self._owner["uid"] and _boot_id() == self._boot,
              "observer_original_owner_changed")
        observation = NativeRepositoryWorldWorkerHandoff.require_observer_assignment(self._handoff,
            prompt=self._prompt, worker_uid=self._worker_uid, worker_gid=self._worker_gid)
        _need(type(observation) is dict and all(observation.get(k) == v for k, v in self._identity.items())
              and observation.get("state") == "TRANSITIONED"
              and observation.get("source_continuity_verified") is True
              and observation.get("native_assignment_continuity_verified") is True
              and observation.get("read_only_transition_verified") is True
              and observation.get("worker_process_verified") is False
              and all(observation.get(k) is False for k in ("world_dispatch_ready", "compiled_context_verified",
                  "proof_authority", "execution_authority", "completion_authority", "provider_launch", "omission_authority")),
              "observer_handoff_not_current")
        _need(self._identity["owner_pid"] == self._owner["pid"]
              and self._identity["owner_uid"] == self._owner["uid"]
              and self._identity["owner_start_time_ticks"] == self._owner["start_time_ticks"]
              and self._identity["boot_id"] == self._boot, "observer_handoff_owner_mismatch")
        _wire(observation)
        return observation

    def _live(self):
        return not self._closed and not self._stop.is_set() and time.monotonic_ns() < self._descriptor["expires_monotonic_ns"]

    def note_spawned(self, process):
        _need(type(process) is subprocess.Popen, "observer_genuine_popen_required")
        with self._lock:
            _need(self._live() and self._process is None and subprocess.Popen.poll(process) is None,
                  "observer_one_live_child_required")
            _need(type(process.args) in (list, tuple) and tuple(process.args) == self._expected_argv,
                  "observer_literal_child_command_mismatch")
            birth = process_identity(process.pid)
            _need(birth["parent_pid"] == self._owner["pid"] and _cmdline(process.pid) == self._command_line
                  and process_identity(process.pid) == birth, "observer_direct_child_command_or_birth_mismatch")
            self._handoff_current()
            self._process, self._root_child = process, birth

    def _registered(self):
        _need(self._live(), "session_expired")
        _need(self._process is not None and self._root_child is not None, "worker_not_registered")
        _need(subprocess.Popen.poll(self._process) is None
              and process_identity(self._process.pid) == self._root_child
              and self._root_child["parent_pid"] == self._owner["pid"]
              and _cmdline(self._process.pid) == self._command_line, "peer_refused")

    def _peer(self, channel):
        self._registered()
        pid, uid, gid = struct.unpack("3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        _need(uid == self._worker_uid and gid == self._worker_gid, "peer_refused")
        first, uids, gids, groups = _groups(pid)
        _need(first["uid"] == uid and all(v == uid for v in uids) and all(v == gid for v in gids)
              and gid in groups and 0 not in groups and _cmdline(pid) == self._worker_command_line
              and process_identity(pid) == first, "peer_refused")
        current, seen = first, set()
        for _ in range(64):
            _need(current["pid"] not in seen, "peer_refused")
            seen.add(current["pid"])
            if current["pid"] == self._root_child["pid"]:
                _need(current == self._root_child and process_identity(pid) == first, "peer_refused")
                break
            _need(current["parent_pid"] > 1 and current["parent_pid"] != self._owner["pid"], "peer_refused")
            current = process_identity(current["parent_pid"])
        else:
            raise RepositoryWorldWorkerObserverError("peer_refused")
        peer = dict(pid=pid, uid=uid, gid=gid, start_time_ticks=first["start_time_ticks"],
            root_pid=self._root_child["pid"], root_start_time_ticks=self._root_child["start_time_ticks"],
            groups_sha256=_sha(_wire(list(groups))), group_count=len(groups))
        _need(self._peer_binding is None or peer == self._peer_binding, "peer_refused")
        return peer

    def _response(self, request_id, *, assignment=None, peer=None, reason=None):
        _need(reason is None or reason in REASONS, "observer_reason_invalid")
        return dict(schema=RESPONSE_SCHEMA, scope=SCOPE, session_id=self._descriptor["session_id"],
            request_id=request_id, ok=reason is None, reason_code=reason, assignment=assignment, peer=peer,
            worker_peer_authenticated=reason is None, **dict.fromkeys(FALSE_FIELDS, False))

    def _answer(self, request, channel):
        request_id = request.get("request_id") if type(request) is dict else None
        if type(request_id) is not str or not _ID.fullmatch(request_id):
            request_id = None
        if not self._live():
            return self._response(request_id, reason="session_expired")
        if (type(request) is not dict or set(request) != _REQUEST_FIELDS or request_id is None
                or request["schema"] != REQUEST_SCHEMA or request["operation"] != "observe_assignment"
                or request["session_id"] != self._descriptor["session_id"]
                or request["descriptor_sha256"] != self.descriptor_sha256):
            return self._response(request_id, reason="invalid_request")
        if request_id in self._seen:
            return self._response(request_id, reason="replayed_request")
        self._seen.add(request_id)
        try:
            peer = self._peer(channel)
        except (OSError, ValueError, RuntimeError):
            return self._response(request_id, reason="worker_not_registered" if self._process is None else "peer_refused")
        try:
            assignment = self._handoff_current()
            response = self._response(request_id, assignment=assignment, peer=peer)
            # Prepare/count before the final native freshness fence. Native/CAS
            # work stays outside SERVER; the owner encapsulates its lock order.
            _wire(response)
            closing = self._handoff_current()
            _need(closing == assignment and self._peer(channel) == peer, "observer_closing_identity_changed")
            self._peer_binding = peer
            return response
        except (OSError, ValueError, RuntimeError):
            return self._response(request_id, reason="observation_refused")

    def _serve(self):
        while self._live() and self._accepted < MAX_REQUESTS:
            try:
                channel, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with channel:
                channel.settimeout(1.0)
                with self._lock:
                    self._accepted += 1
                    try:
                        request = _decode(_recv(channel, maximum=MAX_REQUEST_BYTES), maximum=MAX_REQUEST_BYTES)
                        response = self._answer(request, channel)
                    except (OSError, ValueError, RuntimeError):
                        response = self._response(None, reason="invalid_request")
                    raw = _wire(response)
                    if self._response_wire_bytes + len(raw) + 4 > MAX_TOTAL_RESPONSE_BYTES:
                        response = self._response(response["request_id"], reason="response_budget_exhausted")
                        raw = _wire(response)
                    if self._response_wire_bytes + len(raw) + 4 > MAX_TOTAL_RESPONSE_BYTES:
                        self._stop.set()
                        continue
                    try:
                        # Final expiry/child check immediately before send.
                        if response["ok"]:
                            self._registered()
                        _send(channel, raw)
                        self._responses += 1
                        self._response_bytes += len(raw)
                        self._response_wire_bytes += len(raw) + 4
                        self._refused += int(not response["ok"])
                    except (OSError, ValueError, RuntimeError):
                        pass

    def close(self):
        self._closed = True
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        return dict(schema=AUDIT_SCHEMA, scope=SCOPE, service_stopped=self._thread is None or not self._thread.is_alive(),
            accepted_connections=self._accepted, sent_responses=self._responses, refused_responses=self._refused,
            sent_response_bytes=self._response_bytes, sent_response_wire_bytes=self._response_wire_bytes,
            response_byte_scope="canonical_json_body_and_separate_four_byte_frame_counter",
            peer_birth_bound=self._peer_binding is not None,
            raw_command_exported=False, **dict.fromkeys(FALSE_FIELDS, False))


def start_repository_world_worker_observer(**kwargs):
    return RepositoryWorldWorkerObserver(**kwargs)


def observe_repository_world_worker_assignment(*, descriptor, expected_descriptor_sha256,
                                               request_id=None, timeout_seconds=10.0):
    """Fixed trusted client: assignment metadata only, never an operation router."""
    checked = validate_worker_observer_descriptor(descriptor, expected_sha256=expected_descriptor_sha256)
    _need(_seconds(timeout_seconds, 30.0) and _boot_id() == checked["boot_id"]
          and time.monotonic_ns() < checked["expires_monotonic_ns"], "observer_client_expired")
    request_id = uuid.uuid4().hex if request_id is None else request_id
    _need(type(request_id) is str and _ID.fullmatch(request_id), "observer_request_id_invalid")
    request = dict(schema=REQUEST_SCHEMA, operation="observe_assignment", session_id=checked["session_id"],
        request_id=request_id, descriptor_sha256=expected_descriptor_sha256)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(timeout_seconds)
        channel.connect("\0" + checked["endpoint_name"])
        pid, uid, gid = struct.unpack("3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        owner = process_identity(pid)
        _need((pid, uid, gid) == (checked["owner_pid"], checked["owner_uid"], checked["owner_gid"])
              and owner["start_time_ticks"] == checked["owner_start_time_ticks"], "observer_client_owner_mismatch")
        _send(channel, _wire(request, maximum=MAX_REQUEST_BYTES))
        response = _decode(_recv(channel, maximum=MAX_FRAME_BYTES), maximum=MAX_FRAME_BYTES)
        _need(process_identity(pid) == owner and _boot_id() == checked["boot_id"]
              and time.monotonic_ns() < checked["expires_monotonic_ns"], "observer_client_owner_changed")
    _need(set(response) == _RESPONSE_FIELDS and response["schema"] == RESPONSE_SCHEMA
          and response["scope"] == SCOPE and response["session_id"] == checked["session_id"]
          and response["request_id"] == request_id and type(response["ok"]) is bool
          and all(response[k] is False for k in FALSE_FIELDS), "observer_client_response_invalid")
    if response["ok"]:
        peer = response["peer"]
        _need(response["reason_code"] is None and response["worker_peer_authenticated"] is True
              and type(response["assignment"]) is dict and type(peer) is dict and set(peer) == _PEER_FIELDS
              and peer["pid"] == os.getpid() and peer["uid"] == os.geteuid() == checked["worker_uid"]
              and peer["gid"] == os.getegid() == checked["worker_gid"]
              and peer["start_time_ticks"] == process_identity(os.getpid())["start_time_ticks"]
              and response["assignment"].get("handoff_id") == checked["handoff_id"],
              "observer_client_peer_mismatch")
    else:
        _need(response["reason_code"] in REASONS and response["assignment"] is None and response["peer"] is None
              and response["worker_peer_authenticated"] is False, "observer_client_response_invalid")
    return response


def _worker_observer_client_main():
    """Fixed root-installed helper: one descriptor, at most two observe lines."""
    raw = sys.stdin.buffer.readline(MAX_FRAME_BYTES + 2)
    _need(raw.endswith(b"\n") and len(raw) <= MAX_FRAME_BYTES + 1, "observer_client_initial_frame_invalid")
    initial = _decode(raw[:-1], maximum=MAX_FRAME_BYTES)
    _need(set(initial) == {"descriptor", "expected_descriptor_sha256"}, "observer_client_initial_frame_invalid")
    for _index in range(2):
        command = sys.stdin.buffer.readline(16)
        if not command:
            break
        _need(command == b"observe\n", "observer_client_operation_invalid")
        try:
            response = observe_repository_world_worker_assignment(**initial)
        except (OSError, ValueError, RuntimeError):
            response = dict(schema="repository-world-worker-observer-client-error@1", scope=SCOPE,
                ok=False, reason_code="client_refused", worker_peer_authenticated=False,
                **dict.fromkeys(FALSE_FIELDS, False))
        sys.stdout.buffer.write(_wire(response) + b"\n")
        sys.stdout.buffer.flush()
    return 0


def worker_observer_client_main():
    """Report client refusal without exporting exception text or traceback."""
    try:
        return _worker_observer_client_main()
    except (OSError, ValueError, RuntimeError):
        response = dict(schema="repository-world-worker-observer-client-error@1", scope=SCOPE,
            ok=False, reason_code="client_refused", worker_peer_authenticated=False,
            **dict.fromkeys(FALSE_FIELDS, False))
        sys.stdout.buffer.write(_wire(response) + b"\n")
        sys.stdout.buffer.flush()
        return 1
