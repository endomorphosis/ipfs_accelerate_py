"""Explicit, bounded compiled-packet delivery to one authenticated worker.

The existing assignment observer remains metadata-only. This separate service
uses its original native handoff and registered process, plus a live compiled
packet binding. A worker acknowledgement records byte receipt, not provider
ingestion, task execution, intent approval, or a source/semantic proof.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid

from . import repository_world_worker_observer as metadata

DESCRIPTOR_SCHEMA = "repository-world-worker-packet-descriptor@1"
REQUEST_SCHEMA = "repository-world-worker-packet-request@1"
RESPONSE_SCHEMA = "repository-world-worker-packet-response@1"
RECEIPT_SCHEMA = "repository-world-worker-packet-receipt@1"
AUDIT_SCHEMA = "repository-world-worker-packet-audit@1"
SCOPE = "authenticated_worker_compiled_packet_bytes_only"
MAX_PACKET_BYTES = 262144
MAX_FRAME_BYTES = 2 * MAX_PACKET_BYTES + 16384
MAX_METADATA_BYTES = 8192
MAX_REQUEST_BYTES = 2048
MAX_REQUESTS = 8
MAX_TOTAL_RESPONSE_WIRE_BYTES = MAX_FRAME_BYTES + MAX_REQUESTS * (MAX_METADATA_BYTES + 4)
MAX_SESSION_SECONDS = 180
FALSE_FIELDS = frozenset({"proof_authority", "execution_authority", "completion_authority",
    "world_dispatch_ready", "provider_launch", "provider_ingestion_verified", "task_completed",
    "database_access_granted", "worker_workspace_verified", "intent_approval_verified",
    "semantic_correctness_verified", "total_token_savings_measured"})
REASONS = frozenset({"invalid_request", "replayed_request", "session_expired", "peer_refused",
    "worker_not_registered", "delivery_refused", "packet_already_delivered", "acknowledgement_refused",
    "response_budget_exhausted"})
_HEX = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[0-9a-f]{32}")
_NAME = re.compile(r"repository-world-packet-[0-9a-f]{32}")
_PEER_FIELDS = metadata._PEER_FIELDS
_DESCRIPTOR_FIELDS = frozenset({"schema", "scope", "session_id", "endpoint_name", "metadata_descriptor_sha256", "binding_metadata_sha256",
    "handoff_id", "original_assignment_cid", "binding_id", "packet_id", "context_cid",
    "prompt_sha256", "prompt_bytes", "original_prompt_sha256", "original_prompt_bytes",
    "owner_pid", "owner_uid", "owner_gid", "owner_start_time_ticks", "boot_id", "worker_uid", "worker_gid",
    "worker_command_sha256", "worker_command_bytes", "launcher_command_sha256", "launcher_command_bytes",
    "issued_monotonic_ns", "expires_monotonic_ns", "max_requests", "max_packet_bytes", "max_frame_bytes",
    "max_total_response_wire_bytes", *FALSE_FIELDS})
_GET_FIELDS = frozenset({"schema", "operation", "session_id", "request_id", "descriptor_sha256"})
_ACK_FIELDS = _GET_FIELDS | {"get_request_id", "prompt_sha256", "prompt_bytes", "packet_id", "context_cid"}
_RESPONSE_FIELDS = frozenset({"schema", "scope", "session_id", "request_id", "operation", "ok", "reason_code",
    "binding_id", "packet_id", "context_cid", "prompt_sha256", "prompt_bytes", "binding", "packet_text", "peer",
    "worker_peer_authenticated", "compiled_packet_membership_verified", "acknowledgement_recorded", *FALSE_FIELDS})


class RepositoryWorldWorkerPacketError(ValueError):
    """Closed refusal that exports no packet, command, or exception body."""


def _need(condition, code):
    if not condition:
        raise RepositoryWorldWorkerPacketError(code)


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _wire(value, *, maximum=MAX_METADATA_BYTES):
    pending, nodes, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _need(nodes <= 4096 and depth <= 16, "packet_structure_bound")
        if type(item) is str:
            _need(len(item) <= maximum, "packet_text_bound")
            text_bytes += len(item.encode("utf-8", "strict"))
            _need(text_bytes <= maximum, "packet_text_bound")
        elif type(item) is dict:
            _need(len(item) <= 256 and all(type(k) is str for k in item), "packet_structure_bound")
            pending.extend((v, depth + 1) for pair in item.items() for v in pair)
        elif type(item) in (list, tuple):
            _need(len(item) <= 256, "packet_structure_bound")
            pending.extend((v, depth + 1) for v in item)
        else:
            _need(item is None or type(item) is bool or _integer(item, -(1 << 63), (1 << 63) - 1),
                  "packet_value_invalid")
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8", "strict")
    except (UnicodeError, ValueError) as error:
        raise RepositoryWorldWorkerPacketError("packet_value_invalid") from error
    _need(len(raw) <= maximum, "packet_frame_bound")
    return raw


def _unique(pairs):
    value = {}
    for key, item in pairs:
        _need(key not in value, "packet_duplicate_key")
        value[key] = item
    return value


def _decode(raw, *, maximum):
    _need(type(raw) is bytes and 0 < len(raw) <= maximum, "packet_frame_bound")
    try:
        text = raw.decode("utf-8", "strict")
        # Bound container depth and prospective node cardinality before JSON
        # allocation. The complete canonical/object grammar is checked below.
        depth, nodes, quoted, escaped = 0, 1, False, False
        for character in text:
            if quoted:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
            elif character == '"':
                quoted = True
            elif character in "[{":
                depth += 1
                nodes += 1
                _need(depth <= 16, "packet_structure_bound")
            elif character in "]}":
                depth -= 1
                _need(depth >= 0, "packet_structure_bound")
            elif character in ",:":
                nodes += 1
                _need(nodes <= 4096, "packet_structure_bound")
        _need(not quoted and depth == 0, "packet_invalid_json")
        value = json.loads(text, object_pairs_hook=_unique,
            parse_constant=lambda _value: (_ for _ in ()).throw(RepositoryWorldWorkerPacketError("packet_nonfinite")))
        _need(type(value) is dict and _wire(value, maximum=maximum) == raw, "packet_canonical_object_required")
        return value
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise RepositoryWorldWorkerPacketError("packet_invalid_json") from error


def _recv(channel, *, maximum):
    def exact(count):
        result = bytearray()
        while len(result) < count:
            block = channel.recv(count - len(result))
            _need(bool(block), "packet_short_frame")
            result.extend(block)
        return bytes(result)
    count = struct.unpack("!I", exact(4))[0]
    _need(0 < count <= maximum, "packet_frame_bound")
    return exact(count)


def _send(channel, raw, *, maximum):
    _need(type(raw) is bytes and 0 < len(raw) <= maximum, "packet_frame_bound")
    channel.sendall(struct.pack("!I", len(raw)) + raw)


def validate_worker_packet_descriptor(value, *, expected_sha256):
    _need(type(value) is dict and set(value) == _DESCRIPTOR_FIELDS and value["schema"] == DESCRIPTOR_SCHEMA
          and value["scope"] == SCOPE and all(value[k] is False for k in FALSE_FIELDS), "packet_descriptor_invalid")
    _need(type(expected_sha256) is str and _HEX.fullmatch(expected_sha256)
          and _sha(_wire(value)) == expected_sha256, "packet_descriptor_pin_mismatch")
    _need(type(value["session_id"]) is str and _ID.fullmatch(value["session_id"])
          and type(value["endpoint_name"]) is str and _NAME.fullmatch(value["endpoint_name"]), "packet_descriptor_identity_invalid")
    for key in ("handoff_id", "original_assignment_cid", "binding_id", "packet_id", "context_cid", "boot_id"):
        _need(type(value[key]) is str and 0 < len(value[key].encode("utf-8")) <= 512 and "\0" not in value[key],
              "packet_descriptor_identity_invalid")
    for key in ("metadata_descriptor_sha256", "binding_metadata_sha256", "prompt_sha256", "original_prompt_sha256",
                "worker_command_sha256", "launcher_command_sha256"):
        _need(type(value[key]) is str and _HEX.fullmatch(value[key]), "packet_descriptor_digest_invalid")
    for key in ("owner_pid", "owner_uid", "owner_gid", "worker_uid", "worker_gid", "owner_start_time_ticks"):
        _need(_integer(value[key], 1, (1 << 63) - 1), "packet_descriptor_principal_invalid")
    _need(value["owner_uid"] != value["worker_uid"] and _integer(value["prompt_bytes"], 1, MAX_PACKET_BYTES)
          and _integer(value["original_prompt_bytes"], 1, 2 * 1024 * 1024)
          and all(_integer(value[k], 1, metadata.MAX_COMMAND_BYTES) for k in ("worker_command_bytes", "launcher_command_bytes")),
          "packet_descriptor_principal_invalid")
    issued, expires = value["issued_monotonic_ns"], value["expires_monotonic_ns"]
    _need(_integer(issued, 1, (1 << 63) - 1) and _integer(expires, issued + 1, (1 << 63) - 1)
          and expires - issued <= MAX_SESSION_SECONDS * 1_000_000_000, "packet_descriptor_lifetime_invalid")
    for key, expected in (("max_requests", MAX_REQUESTS), ("max_packet_bytes", MAX_PACKET_BYTES),
                          ("max_frame_bytes", MAX_FRAME_BYTES), ("max_total_response_wire_bytes", MAX_TOTAL_RESPONSE_WIRE_BYTES)):
        _need(type(value[key]) is int and value[key] == expected, "packet_descriptor_budget_invalid")
    return _decode(_wire(value), maximum=MAX_METADATA_BYTES)


class RepositoryWorldWorkerPacketDelivery:
    """A separate content endpoint sharing one original metadata observer.

    Start before child registration and register through ``note_spawned``.
    Closing this service does not close the caller-owned observer or handoff.
    Only one complete packet response and one byte acknowledgement are allowed.
    """

    __slots__ = ("_observer", "_original_observer", "_binding", "_original_binding", "_metadata_bytes", "_metadata_sha",
        "_handoff", "_original_prompt", "_owner", "_boot", "_expected_argv", "_launcher_command", "_worker_command",
        "_process", "_root_child", "_peer_binding", "_initial_binding", "_initial_text_sha", "_descriptor",
        "_descriptor_bytes", "descriptor_sha256", "_lock", "_stop", "_closed", "_seen", "_pending_get", "_acknowledged",
        "_accepted", "_responses", "_refused", "_body_bytes", "_wire_bytes", "_send_failures", "_packet_responses",
        "_packet_bytes", "_ack_responses", "_listener", "_thread")

    def __init__(self, *, observer, binding, timeout_seconds=60):
        from .repository_world_worker_packet import NativeRepositoryWorldCompiledPacketBinding
        _need(type(observer) is metadata.RepositoryWorldWorkerObserver
              and type(binding) is NativeRepositoryWorldCompiledPacketBinding
              and type(timeout_seconds) in (int, float) and 0 < timeout_seconds <= MAX_SESSION_SECONDS,
              "packet_original_owners_required")
        _need(all(hasattr(observer, key) for key in ("_descriptor_bytes", "descriptor_sha256", "_handoff", "_prompt",
            "_expected_argv", "_command_line", "_worker_command_line", "_process", "_root_child", "_peer_binding",
            "_listener", "_thread")), "packet_original_observer_required")
        self._observer = self._original_observer = observer
        self._binding = self._original_binding = binding
        self._metadata_bytes = metadata._wire(observer.descriptor)
        self._metadata_sha = observer.descriptor_sha256
        original = metadata.validate_worker_observer_descriptor(observer.descriptor, expected_sha256=self._metadata_sha)
        self._handoff = observer._handoff
        self._original_prompt = observer._prompt
        self._owner = metadata.process_identity(os.getpid())
        self._boot = metadata._boot_id()
        self._expected_argv = observer._expected_argv
        self._launcher_command = observer._command_line
        self._worker_command = observer._worker_command_line
        self._process = self._root_child = self._peer_binding = None
        _need(observer._process is None and observer._root_child is None and observer._peer_binding is None
              and observer._listener is not None and observer._thread is not None and observer._thread.is_alive(),
              "packet_original_unregistered_observer_required")
        self._initial_binding = None
        information, text = self._material()
        self._initial_binding = _wire(information)
        self._initial_text_sha = _sha(text.encode("utf-8"))
        now, session = time.monotonic_ns(), uuid.uuid4().hex
        expires = min(original["expires_monotonic_ns"], now + int(timeout_seconds * 1_000_000_000))
        _need(expires > now, "packet_session_expired")
        self._descriptor = dict(schema=DESCRIPTOR_SCHEMA, scope=SCOPE, session_id=session,
            endpoint_name="repository-world-packet-" + session, metadata_descriptor_sha256=self._metadata_sha,
            binding_metadata_sha256=_sha(self._initial_binding),
            **{k: information[k] for k in ("handoff_id", "original_assignment_cid", "binding_id", "packet_id",
                "context_cid", "prompt_sha256", "prompt_bytes")},
            original_prompt_sha256=original["prompt_sha256"], original_prompt_bytes=original["prompt_bytes"],
            **{k: original[k] for k in ("owner_pid", "owner_uid", "owner_gid", "owner_start_time_ticks", "boot_id",
                "worker_uid", "worker_gid", "worker_command_sha256", "worker_command_bytes",
                "launcher_command_sha256", "launcher_command_bytes")},
            issued_monotonic_ns=now, expires_monotonic_ns=expires, max_requests=MAX_REQUESTS,
            max_packet_bytes=MAX_PACKET_BYTES, max_frame_bytes=MAX_FRAME_BYTES,
            max_total_response_wire_bytes=MAX_TOTAL_RESPONSE_WIRE_BYTES, **dict.fromkeys(FALSE_FIELDS, False))
        self._descriptor_bytes = _wire(self._descriptor)
        self.descriptor_sha256 = _sha(self._descriptor_bytes)
        validate_worker_packet_descriptor(self._descriptor, expected_sha256=self.descriptor_sha256)
        self._lock, self._stop = threading.RLock(), threading.Event()
        self._closed = False
        self._seen = set()
        self._pending_get = None
        self._acknowledged = False
        self._accepted = self._responses = self._refused = self._body_bytes = self._wire_bytes = self._send_failures = 0
        self._packet_responses = self._packet_bytes = self._ack_responses = 0
        self._listener = self._thread = None
        try:
            self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._listener.bind("\0" + self._descriptor["endpoint_name"])
            self._listener.listen(2)
            self._listener.settimeout(.1)
            self._thread = threading.Thread(target=self._serve, name="world-worker-packet", daemon=True)
            self._thread.start()
        except BaseException:
            if self._listener is not None:
                self._listener.close()
            raise

    @property
    def descriptor(self):
        return _decode(self._descriptor_bytes, maximum=MAX_METADATA_BYTES)

    def _observer_current(self):
        value = self._observer
        _need(value is self._original_observer and value._handoff is self._handoff
              and value._original_handoff is self._handoff and value._prompt == self._original_prompt
              and value._expected_argv == self._expected_argv and value._command_line == self._launcher_command
              and value._worker_command_line == self._worker_command
              and value.descriptor_sha256 == self._metadata_sha
              and metadata._wire(value.descriptor) == self._metadata_bytes
              and metadata.process_identity(os.getpid()) == self._owner
              and metadata._boot_id() == self._boot, "packet_original_observer_changed")
        _need(metadata.RepositoryWorldWorkerObserver._live(value), "packet_session_expired")
        observation = metadata.RepositoryWorldWorkerObserver._handoff_current(value)
        _need(observation["handoff_id"] == value.descriptor["handoff_id"], "packet_original_observer_changed")
        if self._process is not None:
            _need(value._process is self._process and value._root_child == self._root_child,
                  "packet_registered_process_changed")
        return observation

    def _material(self):
        from .repository_world_worker_packet import NativeRepositoryWorldCompiledPacketBinding
        self._observer_current()
        _need(self._binding is self._original_binding, "packet_original_binding_changed")
        information, text = NativeRepositoryWorldCompiledPacketBinding.delivery_material(self._binding,
            handoff=self._handoff, original_prompt=self._original_prompt)
        _need(type(information) is dict and type(text) is str and len(text) <= MAX_PACKET_BYTES,
              "packet_binding_material_invalid")
        raw = text.encode("utf-8", "strict")
        _need(0 < len(raw) <= MAX_PACKET_BYTES and information.get("prompt_sha256") == _sha(raw)
              and information.get("schema") == "repository-world-compiled-worker-packet-binding@1"
              and information.get("profile") == "native_structural_readonly_packet@1"
              and type(information.get("prompt_bytes")) is int and information["prompt_bytes"] == len(raw)
              and information.get("handoff_id") == self._observer.descriptor["handoff_id"]
              and information.get("original_assignment_cid") == self._observer.descriptor["original_assignment_cid"]
              and all(information.get(k) is True for k in ("structural_compiler_context_current",
                  "native_task_projection_current", "native_assignment_continuity_verified"))
              and all(information.get(k) is False for k in ("compiler_signature_verified", "semantic_alignment_verified",
                  "world_dispatch_ready", "provider_launch", "provider_ingestion_verified", "proof_authority",
                  "execution_authority", "completion_authority", "omission_authority", "worker_edit_authority")),
              "packet_binding_material_invalid")
        wire = _wire(information)
        _need(self._initial_binding is None or wire == self._initial_binding, "packet_binding_changed")
        self._observer_current()
        return information, text

    def _live(self):
        return (not self._closed and not self._stop.is_set()
            and _wire(self._descriptor) == self._descriptor_bytes and _sha(self._descriptor_bytes) == self.descriptor_sha256
            and time.monotonic_ns() < self._descriptor["expires_monotonic_ns"])

    def note_spawned(self, process):
        _need(type(process) is subprocess.Popen, "packet_genuine_popen_required")
        with self._lock:
            _need(self._live() and self._process is None, "packet_one_live_child_required")
            self._material()
            metadata.RepositoryWorldWorkerObserver.note_spawned(self._observer, process)
            self._process, self._root_child = process, dict(self._observer._root_child)
            self._registered()

    def _registered(self):
        _need(self._live(), "packet_session_expired")
        _need(self._observer is self._original_observer and self._process is not None and self._root_child is not None
              and self._observer._process is self._process and self._observer._root_child == self._root_child,
              "packet_worker_not_registered")
        metadata.RepositoryWorldWorkerObserver._registered(self._observer)

    def _peer(self, channel):
        self._registered()
        peer = metadata.RepositoryWorldWorkerObserver._peer(self._observer, channel)
        _need(self._peer_binding is None or peer == self._peer_binding, "packet_peer_refused")
        return peer

    def _response(self, request_id, operation=None, *, information=None, text=None, peer=None, reason=None, acknowledged=False):
        _need(reason is None or reason in REASONS, "packet_reason_invalid")
        return dict(schema=RESPONSE_SCHEMA, scope=SCOPE, session_id=self._descriptor["session_id"],
            request_id=request_id, operation=operation, ok=reason is None, reason_code=reason,
            binding_id=None if information is None else information["binding_id"],
            packet_id=None if information is None else information["packet_id"],
            context_cid=None if information is None else information["context_cid"],
            prompt_sha256=None if information is None else information["prompt_sha256"],
            prompt_bytes=None if information is None else information["prompt_bytes"], binding=information,
            packet_text=text, peer=peer, worker_peer_authenticated=reason is None,
            compiled_packet_membership_verified=reason is None, acknowledgement_recorded=acknowledged,
            **dict.fromkeys(FALSE_FIELDS, False))

    def _answer(self, request, channel):
        request_id = request.get("request_id") if type(request) is dict else None
        if type(request_id) is not str or not _ID.fullmatch(request_id):
            request_id = None
        operation = request.get("operation") if type(request) is dict else None
        operation = operation if operation in ("get_packet", "ack_packet") else None
        if not self._live():
            return self._response(request_id, operation, reason="session_expired")
        fields = _ACK_FIELDS if operation == "ack_packet" else _GET_FIELDS
        if (type(request) is not dict or set(request) != fields or operation is None or request_id is None
                or request["schema"] != REQUEST_SCHEMA or request["session_id"] != self._descriptor["session_id"]
                or request["descriptor_sha256"] != self.descriptor_sha256):
            return self._response(request_id, operation, reason="invalid_request")
        if request_id in self._seen:
            return self._response(request_id, operation, reason="replayed_request")
        self._seen.add(request_id)
        try:
            peer = self._peer(channel)
        except (OSError, ValueError, RuntimeError):
            return self._response(request_id, operation,
                reason="worker_not_registered" if self._process is None else "peer_refused")
        try:
            information, text = self._material()
            if operation == "get_packet":
                if self._pending_get is not None:
                    return self._response(request_id, operation, reason="packet_already_delivered")
            elif (self._pending_get is None or self._acknowledged
                    or request["get_request_id"] != self._pending_get["request_id"]
                    or request["prompt_sha256"] != information["prompt_sha256"]
                    or type(request["prompt_bytes"]) is not int or request["prompt_bytes"] != information["prompt_bytes"]
                    or request["packet_id"] != information["packet_id"] or request["context_cid"] != information["context_cid"]
                    or self._pending_get["peer"] != peer):
                return self._response(request_id, operation, reason="acknowledgement_refused")
            response = self._response(request_id, operation, information=information,
                text=text if operation == "get_packet" else None, peer=peer, acknowledged=operation == "ack_packet")
            _wire(response, maximum=MAX_FRAME_BYTES)
            closing, closing_text = self._material()
            _need(closing == information and closing_text == text and self._peer(channel) == peer,
                  "packet_closing_identity_changed")
            return response
        except (OSError, ValueError, RuntimeError):
            return self._response(request_id, operation, reason="delivery_refused")

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
                    raw = _wire(response, maximum=MAX_FRAME_BYTES)
                    if self._wire_bytes + len(raw) + 4 > MAX_TOTAL_RESPONSE_WIRE_BYTES:
                        response = self._response(response["request_id"], response["operation"], reason="response_budget_exhausted")
                        raw = _wire(response)
                    if self._wire_bytes + len(raw) + 4 > MAX_TOTAL_RESPONSE_WIRE_BYTES:
                        self._stop.set()
                        continue
                    if response["ok"]:
                        try:
                            # Serialize first, then native/CAS fences, then peer/expiry
                            # immediately before send. No SERVER lock is held here.
                            closing, text = self._material()
                            _need(closing == response["binding"]
                                  and (response["packet_text"] is None or text == response["packet_text"])
                                  and self._peer(channel) == response["peer"], "packet_closing_identity_changed")
                            self._registered()
                        except (OSError, ValueError, RuntimeError):
                            response = self._response(response["request_id"], response["operation"], reason="delivery_refused")
                            raw = _wire(response)
                    try:
                        # Share the observer's process/peer lock for the final
                        # identity check and send, preventing a parallel metadata
                        # request from binding a different sibling in this gap.
                        with self._original_observer._lock:
                            if response["ok"]:
                                closing, text = self._material()
                                _need(closing == response["binding"]
                                      and (response["packet_text"] is None or text == response["packet_text"]),
                                      "packet_closing_binding_changed")
                                _need(self._peer(channel) == response["peer"], "packet_closing_peer_changed")
                            _send(channel, raw, maximum=MAX_FRAME_BYTES)
                            self._responses += 1
                            self._body_bytes += len(raw)
                            self._wire_bytes += len(raw) + 4
                            self._refused += int(not response["ok"])
                            if response["ok"]:
                                self._peer_binding = response["peer"]
                                self._observer._peer_binding = dict(response["peer"])
                                if response["operation"] == "get_packet":
                                    self._pending_get = dict(request_id=response["request_id"], peer=dict(response["peer"]))
                                    self._packet_responses += 1
                                    self._packet_bytes += response["prompt_bytes"]
                                else:
                                    self._acknowledged = True
                                    self._ack_responses += 1
                    except (OSError, ValueError, RuntimeError):
                        self._send_failures += 1

    def close(self):
        self._closed = True
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
        return dict(schema=AUDIT_SCHEMA, scope=SCOPE, service_stopped=self._thread is None or not self._thread.is_alive(),
            accepted_connections=self._accepted, sent_responses=self._responses, refused_responses=self._refused,
            sent_response_bytes=self._body_bytes, sent_response_wire_bytes=self._wire_bytes,
            sent_packet_responses=self._packet_responses, sent_packet_bytes=self._packet_bytes,
            acknowledged_packets=self._ack_responses, send_failures=self._send_failures,
            peer_birth_bound=self._peer_binding is not None, response_byte_scope="canonical_json_body_plus_four_byte_frame",
            packet_byte_scope="exact_compiler_prompt_utf8", raw_packet_exported=False, raw_command_exported=False,
            raw_packet_export_scope="receipt_and_stdout_only",
            **dict.fromkeys(FALSE_FIELDS, False))


def start_repository_world_worker_packet_delivery(**kwargs):
    return RepositoryWorldWorkerPacketDelivery(**kwargs)


def _client_exchange(descriptor, pin, request, timeout_seconds):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(timeout_seconds)
        channel.connect("\0" + descriptor["endpoint_name"])
        pid, uid, gid = struct.unpack("3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        owner = metadata.process_identity(pid)
        _need((pid, uid, gid) == (descriptor["owner_pid"], descriptor["owner_uid"], descriptor["owner_gid"])
              and owner["start_time_ticks"] == descriptor["owner_start_time_ticks"], "packet_client_owner_mismatch")
        _send(channel, _wire(request, maximum=MAX_REQUEST_BYTES), maximum=MAX_REQUEST_BYTES)
        result = _decode(_recv(channel, maximum=MAX_FRAME_BYTES), maximum=MAX_FRAME_BYTES)
        _need(metadata.process_identity(pid) == owner and metadata._boot_id() == descriptor["boot_id"]
              and time.monotonic_ns() < descriptor["expires_monotonic_ns"], "packet_client_owner_changed")
    _need(set(result) == _RESPONSE_FIELDS and result["schema"] == RESPONSE_SCHEMA and result["scope"] == SCOPE
          and result["session_id"] == descriptor["session_id"] and result["request_id"] == request["request_id"]
          and result["operation"] == request["operation"] and type(result["ok"]) is bool
          and all(result[k] is False for k in FALSE_FIELDS), "packet_client_response_invalid")
    if not result["ok"]:
        _need(type(result["reason_code"]) is str and result["reason_code"] in REASONS and result["worker_peer_authenticated"] is False
              and result["compiled_packet_membership_verified"] is False and result["acknowledgement_recorded"] is False
              and all(result[k] is None for k in ("binding_id", "packet_id", "context_cid", "prompt_sha256",
                  "prompt_bytes", "binding", "packet_text", "peer")), "packet_client_response_invalid")
        return result
    peer = result["peer"]
    _need(result["reason_code"] is None and result["worker_peer_authenticated"] is True
          and result["compiled_packet_membership_verified"] is True
          and type(result["binding"]) is dict and _sha(_wire(result["binding"])) == descriptor["binding_metadata_sha256"]
          and type(peer) is dict and set(peer) == _PEER_FIELDS
          and all(_integer(peer[k], 1, (1 << 63) - 1) for k in ("pid", "uid", "gid", "start_time_ticks", "root_pid", "root_start_time_ticks"))
          and _integer(peer["group_count"], 1, 64) and type(peer["groups_sha256"]) is str and _HEX.fullmatch(peer["groups_sha256"])
          and peer["pid"] == os.getpid() and peer["uid"] == os.getuid() == os.geteuid() == descriptor["worker_uid"]
          and peer["gid"] == os.getgid() == os.getegid() == descriptor["worker_gid"]
          and peer["start_time_ticks"] == metadata.process_identity(os.getpid())["start_time_ticks"],
          "packet_client_peer_or_binding_mismatch")
    _need(all(result[k] == descriptor[k] for k in ("binding_id", "packet_id", "context_cid", "prompt_sha256", "prompt_bytes"))
          and type(result["prompt_bytes"]) is int, "packet_client_packet_identity_mismatch")
    if request["operation"] == "get_packet":
        text = result["packet_text"]
        _need(type(text) is str and len(text) <= MAX_PACKET_BYTES and result["acknowledgement_recorded"] is False,
              "packet_client_payload_invalid")
        raw = text.encode("utf-8", "strict")
        _need(len(raw) == descriptor["prompt_bytes"] and _sha(raw) == descriptor["prompt_sha256"],
              "packet_client_payload_pin_mismatch")
    else:
        _need(result["packet_text"] is None and result["acknowledgement_recorded"] is True,
              "packet_client_acknowledgement_invalid")
    return result


def fetch_repository_world_worker_packet(*, descriptor, expected_descriptor_sha256, timeout_seconds=10):
    """Receive exact bytes and acknowledge their digest on the same worker birth.

    ``packet_text`` is returned privately only following a successful current
    acknowledgement. The receipt itself contains no task/source/packet text.
    """
    checked = validate_worker_packet_descriptor(descriptor, expected_sha256=expected_descriptor_sha256)
    _need(type(timeout_seconds) in (int, float) and 0 < timeout_seconds <= 30
          and metadata._boot_id() == checked["boot_id"]
          and time.monotonic_ns() < checked["expires_monotonic_ns"], "packet_client_expired")
    get_id, ack_id = uuid.uuid4().hex, uuid.uuid4().hex
    common = dict(schema=REQUEST_SCHEMA, session_id=checked["session_id"], descriptor_sha256=expected_descriptor_sha256)
    response = _client_exchange(checked, expected_descriptor_sha256,
        dict(**common, operation="get_packet", request_id=get_id), timeout_seconds)
    receipt = dict(schema=RECEIPT_SCHEMA, scope=SCOPE, status="refused", reason_code=response["reason_code"],
        descriptor_sha256=expected_descriptor_sha256, binding_metadata_sha256=checked["binding_metadata_sha256"],
        get_request_id=get_id, ack_request_id=None, binding_id=checked["binding_id"], packet_id=checked["packet_id"],
        context_cid=checked["context_cid"], prompt_sha256=checked["prompt_sha256"], prompt_bytes=checked["prompt_bytes"],
        worker_peer_authenticated=response["ok"], packet_byte_identity_verified=False, acknowledgement_recorded=False,
        peer=response["peer"], raw_packet_exported=False, raw_packet_export_scope="receipt_and_stdout_only",
        **dict.fromkeys(FALSE_FIELDS, False))
    if not response["ok"]:
        return dict(packet_text=None, receipt=receipt)
    # These values come from the actual received bytes, not just the descriptor.
    received = response["packet_text"].encode("utf-8", "strict")
    acknowledgement = dict(**common, operation="ack_packet", request_id=ack_id, get_request_id=get_id,
        prompt_sha256=_sha(received), prompt_bytes=len(received), packet_id=response["packet_id"], context_cid=response["context_cid"])
    receipt.update(status="received_unacknowledged", reason_code="acknowledgement_refused",
        ack_request_id=ack_id, packet_byte_identity_verified=True)
    try:
        ack = _client_exchange(checked, expected_descriptor_sha256, acknowledgement, timeout_seconds)
    except (OSError, ValueError, RuntimeError):
        return dict(packet_text=None, receipt=receipt)
    if not ack["ok"]:
        receipt["reason_code"] = ack["reason_code"]
        return dict(packet_text=None, receipt=receipt)
    _need(ack["peer"] == response["peer"], "packet_client_acknowledgement_peer_changed")
    receipt.update(status="received_and_acknowledged", reason_code=None, acknowledgement_recorded=True)
    return dict(packet_text=response["packet_text"], receipt=receipt)


def worker_packet_client_main():
    """Fixed helper: at most two fetches; stdout contains receipts, never bytes."""
    def error():
        return dict(schema="repository-world-worker-packet-client-error@1", scope=SCOPE,
            status="unavailable", reason_code="client_refused", raw_packet_exported=False,
            raw_packet_export_scope="receipt_and_stdout_only",
            **dict.fromkeys(FALSE_FIELDS, False))
    try:
        raw = sys.stdin.buffer.readline(MAX_METADATA_BYTES + 2)
        _need(raw.endswith(b"\n") and len(raw) <= MAX_METADATA_BYTES + 1, "packet_client_input_invalid")
        inputs = _decode(raw[:-1], maximum=MAX_METADATA_BYTES)
        _need(set(inputs) == {"descriptor", "expected_descriptor_sha256"}, "packet_client_input_invalid")
        checked = validate_worker_packet_descriptor(inputs["descriptor"], expected_sha256=inputs["expected_descriptor_sha256"])
        for _ in range(2):
            command = sys.stdin.buffer.readline(32)
            if not command:
                return 0
            _need(command == b"get_packet\n", "packet_client_operation_invalid")
            try:
                value = fetch_repository_world_worker_packet(descriptor=checked,
                    expected_descriptor_sha256=inputs["expected_descriptor_sha256"])["receipt"]
            except (OSError, ValueError, RuntimeError):
                value = error()
            sys.stdout.buffer.write(_wire(value) + b"\n")
            sys.stdout.buffer.flush()
        return 0
    except (OSError, ValueError, RuntimeError):
        sys.stdout.buffer.write(_wire(error()) + b"\n")
        sys.stdout.buffer.flush()
        return 1
