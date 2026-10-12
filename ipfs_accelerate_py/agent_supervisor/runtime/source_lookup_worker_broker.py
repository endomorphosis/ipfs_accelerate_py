"""Source-only descendant reads through the original native task owner.

The public worker endpoint carries no credentials. One owner-process proxy
grant survives shell subprocesses, so native continuations retain their caller
binding. The source profile still requires clean committed source in both
checkouts. Sequential checks do not provide a filesystem writer lease.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import socket
import stat
import struct
import threading
import time
import uuid

from ..task_sources import typed_state_owner as typed
from ..task_sources.control_plane_contracts import canonical_json_bytes
from . import current_source_lookup as native
from . import source_lookup_worker_client as protocol
from . import source_formalization_context as guard
from .numerical_obligation_broker import _write_new, _unlink_same
from .quack_state_server import QuackStateServer, ServerLifecycle

MAX_REQUESTS = 64
MAX_TOTAL_RESPONSE_BYTES = 524288
MAX_SESSION_SECONDS = 1800
MAX_RESPONSE_BYTES = 32768
AUDIT_SCHEMA = "supervisor-source-worker-session@1"
_PUBLIC_FIELDS = ("raw_path_hex", "source_cid", "slice_cid", "encoding", "content",
    "offset", "end_offset", "source_bytes", "source_complete", "next_continuation",
    "unresolved_frontier", "working_source_verified", "source_check_profile",
    "repository_complete", "read_only", "completion_authority", "proof_authority",
    "edit_authority", "omission_authority")


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _require_descendant(pid, *, uid, root_child, owner_pid):
    """Return a genuine live peer beneath the retained direct child birth."""
    first = protocol.process_identity(pid)
    _need(first["uid"] == uid, "source worker UID differs")
    current, seen = first, set()
    for _ in range(64):
        _need(current["pid"] not in seen, "source worker ancestry repeats")
        seen.add(current["pid"])
        if current["pid"] == root_child["pid"]:
            actual = protocol.process_identity(current["pid"])
            _need(actual["start_time_ticks"] == root_child["start_time_ticks"]
                  and actual["parent_pid"] == owner_pid
                  and protocol.process_identity(pid) == first,
                  "source worker root or peer birth changed")
            return first
        _need(current["parent_pid"] > 1 and current["parent_pid"] != owner_pid,
              "source worker is outside the launched child")
        current = protocol.process_identity(current["parent_pid"])
    raise ValueError("source worker ancestry exceeds bound")


def _count_response(response):
    response["response_bytes"] = response["response_wire_bytes"] = 0
    for _ in range(12):
        size = len(protocol.canonical_json(response)) + 1
        if (size, size + 3) == (response["response_bytes"], response["response_wire_bytes"]):
            return size
        response["response_bytes"], response["response_wire_bytes"] = size, size + 3
    raise ValueError("source worker byte accounting failed")


class SourceLookupWorkerBroker:
    """Trusted launcher opt-in; no database, owner or CAS is constructed."""

    def __init__(self, *, server, canonical_repository, workspace, boundary, prompt,
                 timeout_seconds=90):
        _need(type(self) is SourceLookupWorkerBroker and type(server) is QuackStateServer,
              "original native state owner required")
        _need(type(timeout_seconds) is int and 1 <= timeout_seconds <= MAX_SESSION_SECONDS,
              "bounded source worker session required")
        _need(type(prompt) in (str, bytes), "exact source worker prompt required")
        prompt_bytes = prompt.encode("utf-8") if type(prompt) is str else prompt
        prompt_bytes.decode("utf-8", "strict")
        _need(0 < len(prompt_bytes) <= 2 * 1024 * 1024, "source worker prompt exceeds bound")
        self.repository, self.workspace = guard._workspace(canonical_repository, workspace)
        self.boundary = Path(boundary).absolute()
        boundary_value, boundary_sha = guard._owner_boundary(self.boundary)
        self.owner_uid, self.worker_uid = boundary_value["owner_uid"], boundary_value["worker_uid"]
        _need(self.owner_uid == os.getuid() == os.geteuid() == os.getgid()
              and self.owner_uid > 0 and self.worker_uid > 0 and self.worker_uid != self.owner_uid,
              "separate nonroot source owner and worker required")
        self._pinned_owner_uid, self._pinned_worker_uid = self.owner_uid, self.worker_uid
        self._server, self._owner_lock = server, server._lock
        self._client = self._grant = None
        self._grant_revoked = False
        self._listener = self._thread = None
        self._socket_identity = self._descriptor_identity = None
        self._stop, self._registered = threading.Event(), threading.Event()
        self._child = self._process = None
        self._closed = False
        self._records, self._seen = [], set()
        self._accepted = self._response_bytes = self._native_response_bytes = self._native_wire_bytes = 0
        self._inflight = None
        self._accounting_lock = threading.Lock()
        self.parent = self.boundary.parent / "worker-observations"
        guard._protected_directory(self.parent, owner=self.owner_uid, group=self.owner_uid)
        parent = self.parent.lstat()
        self._parent_identity = parent.st_dev, parent.st_ino, parent.st_uid, parent.st_gid, parent.st_mode
        session = uuid.uuid4().hex
        self.socket_path = self.parent / ("source-" + session + ".sock")
        self.descriptor_path = self.parent / ("source-" + session + ".json")
        self.audit_path = self.parent / ("source-" + session + "-audit.json")
        self._paths = self.repository, self.workspace, self.boundary, self.parent, self.socket_path, self.descriptor_path, self.audit_path
        try:
            with self._owner_lock:
                _need(server.lifecycle is ServerLifecycle.READY, "source owner is not ready")
                server.ready()
                self._gateway, self._connection, self._identity = server._command_gateway, server._connection, server.identity
                gateway = self._gateway
                _need(type(gateway) is typed.TypedStateOwnerGateway
                      and gateway._connection is self._connection
                      and gateway._transaction_lock is self._owner_lock, "original source gateway required")
                self._service = gateway._current_source_lookup
                _need(type(self._service) is native.CurrentSourceLookup
                      and self._service._root == self.workspace, "native source binding must name the allocation")
                gateway._assert_current_source_binding(self._service)
                self._binding = self._service.binding(gateway.identity)
                self._binding_bytes = canonical_json_bytes(self._binding)
                self._owner_bytes = canonical_json_bytes(dict(gateway.identity))
                _need(self._binding["max_response_bytes"] <= MAX_RESPONSE_BYTES - 2048,
                      "native source response profile exceeds worker envelope reserve")
                source = self._service
                self._cas = source._store.cas
                self._canonical = native.CurrentSourceLookup(self.repository, cas=self._cas,
                    expected_inventory_cid=source._inventory_cid, request=source._request, limits=source._limits,
                    task_cid=self._binding["task_cid"], task_revision=self._binding["task_revision"],
                    expected_task_body_cid=self._binding["expected_task_body_cid"],
                    context_cid=self._binding["context_cid"], allowed_paths=source._paths,
                    max_response_bytes=self._binding["max_response_bytes"], max_slice_bytes=self._binding["max_slice_bytes"])
                self._canonical.check_native(self._connection, gateway.identity)
                _need(1 <= len(source._paths) <= 32, "source worker admitted path population exceeds profile")
                retained = self._canonical._load()
                by_path = {item.entry.raw_path_hex: item for item in retained.entries}
                refs = []
                for name in source._paths:
                    item = by_path.get(name)
                    _need(item is not None and item.data is not None
                          and item.entry.git_mode in ("100644", "100755"),
                          "source worker scope contains a nonregular or unavailable file")
                    refs.append(dict(raw_path_hex=name, source_cid=item.source_cid,
                        size_bytes=len(item.data), git_mode=item.entry.git_mode))
                self._source_refs_bytes = protocol.canonical_json(refs)
                self._source_pairs = tuple((ref["raw_path_hex"], ref["source_cid"]) for ref in refs)
                gateway._assert_current_source_binding(source)
                source.check_native(self._connection, gateway.identity)
                self._canonical.check_native(self._connection, gateway.identity)
                source._check_source()
                self._canonical._check_source()
                self._canonical_binding_bytes = canonical_json_bytes(self._canonical.binding(gateway.identity))
                self._original_refs = (server, gateway, source, self._connection, self._owner_lock, self._canonical, self._cas)
                token, self._grant = server.issue_typed_client_grant_record(
                    client_id="source-worker-proxy:" + session, process_birth_id=typed.kernel_process_birth_id(),
                    peer_pid=os.getpid(), allowed_operations=("source.get",),
                    entity_scopes={"task_cid": self._binding["task_cid"], "repository_id": self._binding["repository_id"],
                                   "tree_id": self._binding["expected_tree"]}, ttl_seconds=timeout_seconds)
                self._original_grant_id = self._grant.grant_id
            # The gateway's serving thread needs the owner lock during open/RPC.
            self._client = typed.TypedStateOwnerConnection(socket_path=server.typed_command_socket_path(),
                token=token, client_id=self._grant.client_id, process_birth_id=self._grant.process_birth_id,
                store_id=gateway.store_id, timeout_seconds=min(timeout_seconds, 30))
            self._original_client = self._client
            self._client_identity = canonical_json_bytes(dict(self._client.identity))
            self._client_grant = canonical_json_bytes(dict(self._client.grant))
            with self._owner_lock:
                self._current_owner()
            owner, now = protocol.process_identity(os.getpid()), time.monotonic_ns()
            self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o660)
            endpoint = self.socket_path.lstat()
            self._socket_identity = endpoint.st_dev, endpoint.st_ino
            self.descriptor = dict(schema=protocol.DESCRIPTOR_SCHEMA, session_id=session,
                owner_uid=self.owner_uid, worker_uid=self.worker_uid, owner_pid=owner["pid"],
                owner_start_time_ticks=owner["start_time_ticks"], workspace=str(self.workspace),
                task_cid=self._binding["task_cid"], task_revision=self._binding["task_revision"],
                repository_id=self._binding["repository_id"], context_cid=self._binding["context_cid"],
                source_binding_cid=self._binding["binding_cid"], prompt_sha256=_sha(prompt_bytes),
                source_refs=refs,
                boundary_manifest_sha256=boundary_sha, socket_path=str(self.socket_path),
                socket_device=endpoint.st_dev, socket_inode=endpoint.st_ino,
                issued_monotonic_ns=now, expires_monotonic_ns=now + timeout_seconds * 1_000_000_000,
                max_requests=MAX_REQUESTS, max_total_response_bytes=MAX_TOTAL_RESPONSE_BYTES,
                max_response_bytes=MAX_RESPONSE_BYTES, max_slice_bytes=self._binding["max_slice_bytes"],
                native_grant_id=self._grant.grant_id,
                proof_authority=False, execution_authority=False, completion_authority=False)
            protocol.validate_descriptor(self.descriptor)
            self._descriptor_bytes = protocol.canonical_json(self.descriptor)
            _need(len(self._descriptor_bytes) <= 8192, "complete source worker descriptor exceeds metadata bound")
            self._session_id = session
            self._expires_monotonic_ns = self.descriptor["expires_monotonic_ns"]
            self.descriptor_sha256, self._descriptor_identity = _write_new(self.descriptor_path, self.descriptor, 0o440)
            self._listener.listen(4)
            self._listener.settimeout(0.2)
            self._thread = threading.Thread(target=self._serve, name="source-worker-reader", daemon=True)
            self._thread.start()
        except BaseException:
            self._dispose()
            raise

    @property
    def extra_argv(self):
        return [protocol.DESCRIPTOR_FLAG, str(self.descriptor_path), protocol.DIGEST_FLAG, self.descriptor_sha256]

    def _current_owner(self):
        current = (self._server, self._gateway, self._service, self._connection,
                   self._owner_lock, self._canonical, self._cas)
        _need(all(left is right for left, right in zip(current, self._original_refs))
              and self._server.lifecycle is ServerLifecycle.READY
              and self._server._lock is self._owner_lock and self._server.identity is self._identity
              and self._server._command_gateway is self._gateway and self._server._connection is self._connection,
              "source worker original owner changed")
        if hasattr(self, "_descriptor_bytes"):
            _need(protocol.canonical_json(self.descriptor) == self._descriptor_bytes,
                  "source worker original descriptor changed")
        gateway = self._gateway
        gateway._assert_current_source_binding(self._service)
        _need(canonical_json_bytes(dict(gateway.identity)) == self._owner_bytes
              and canonical_json_bytes(self._service.binding(gateway.identity)) == self._binding_bytes
              and self._service._store.cas is self._cas and self._service._root == self.workspace
              and self._canonical._store.cas is self._cas and self._canonical._root == self.repository
              and canonical_json_bytes(self._canonical.binding(gateway.identity)) == self._canonical_binding_bytes
              and canonical_json_bytes(self._binding) == self._binding_bytes
              and (self.repository, self.workspace, self.boundary, self.parent, self.socket_path, self.descriptor_path, self.audit_path) == self._paths,
              "source worker custody or routing changed")
        self._service.check_native(self._connection, gateway.identity)
        self._canonical.check_native(self._connection, gateway.identity)
        active = gateway._require_active_grant(self._grant, peer_identity=(
            self._grant.peer_pid, self._grant.peer_uid, self._grant.peer_start_time_ticks))
        _need(active.grant_id == self._original_grant_id, "source proxy grant changed")
        gateway._authorize_source_grant(active, self._service)
        if self._client is not None:
            _need(self._client is self._original_client
                  and canonical_json_bytes(dict(self._client.identity)) == self._client_identity
                  and canonical_json_bytes(dict(self._client.grant)) == self._client_grant,
                  "source worker original proxy changed")

    def _context_current(self):
        _need(self._live() and protocol.canonical_json(self.descriptor) == self._descriptor_bytes,
              "source worker session changed or expired")
        _need(self.owner_uid == self._pinned_owner_uid == self.descriptor["owner_uid"]
              == os.getuid() == os.geteuid() == os.getgid()
              and self.worker_uid == self._pinned_worker_uid == self.descriptor["worker_uid"],
              "source worker owner or selected UID changed")
        _need(guard._owner_boundary(self.boundary)[1] == self.descriptor["boundary_manifest_sha256"],
              "source worker boundary changed")
        self._parent_current()
        for path, identity, mode, kind in ((self.socket_path, self._socket_identity, 0o660, stat.S_ISSOCK),
                (self.descriptor_path, self._descriptor_identity, 0o440, stat.S_ISREG)):
            info = path.lstat()
            _need(kind(info.st_mode) and (info.st_dev, info.st_ino) == identity
                  and info.st_uid == self.owner_uid and info.st_gid == self.owner_uid
                  and stat.S_IMODE(info.st_mode) == mode, "source worker published endpoint changed")
        _need(guard._read(self.descriptor_path, protocol.MAX_FRAME_BYTES) == self._descriptor_bytes,
              "source worker published descriptor bytes changed")

    def _parent_current(self):
        parent_path = self._paths[3]
        guard._protected_directory(parent_path, owner=self.owner_uid, group=self.owner_uid)
        parent = parent_path.lstat()
        _need((parent.st_dev, parent.st_ino, parent.st_uid, parent.st_gid, parent.st_mode) == self._parent_identity,
              "source worker protected directory changed")

    def _live(self):
        return not (self._closed or self._stop.is_set()
            or time.monotonic_ns() >= self._expires_monotonic_ns
            or self._process is not None and self._process.poll() is not None)

    def note_spawned(self, process):
        _need(not self._closed and not self._registered.is_set() and process.poll() is None,
              "source worker requires one live spawned child")
        child = protocol.process_identity(process.pid)
        _need(child["parent_pid"] == os.getpid(), "source worker child belongs to another owner")
        self._child, self._process = child, process
        self._registered.set()

    def _peer(self, channel):
        _need(self._registered.wait(timeout=2) and self._live(), "source worker is inactive")
        pid, uid, _ = struct.unpack("3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        _need(uid == self.worker_uid, "source worker socket UID differs")
        return _require_descendant(pid, uid=uid, root_child=self._child, owner_pid=os.getpid())

    def _selected(self, payload):
        self._canonical._check_source()
        retained = self._canonical._load()
        selected = next(e for e in retained.entries if e.entry.raw_path_hex == payload["raw_path_hex"])
        _need(selected.data is not None and selected.entry.git_mode in ("100644", "100755")
              and selected.source_cid == payload["source_cid"], "canonical source differs")
        self._canonical._working_file(selected.entry, selected.data)
        return selected

    def _read_source(self, payload):
        """Owner-side proxy only; worker admission belongs to _answer."""
        payload = native._payload(payload)
        _need(payload["binding_cid"] == self._binding["binding_cid"]
              and payload["raw_path_hex"] in self._service._paths
              and (payload["raw_path_hex"], payload["source_cid"]) in self._source_pairs
              and payload["max_bytes"] <= self._binding["max_slice_bytes"], "source query is outside session scope")
        self._context_current()
        with self._owner_lock:
            self._current_owner()
            self._selected(payload)
        response = self._client.source_get(payload)
        response = native.validate_source_response(response, payload=payload,
            owner_identity=self._client.identity, grant=self._client.grant)
        _need(canonical_json_bytes(response["result"]["binding"]) == self._binding_bytes,
              "native source response binding changed")
        with self._accounting_lock:
            self._native_response_bytes += response["result"]["response_bytes"]
            self._native_wire_bytes += response["result"]["response_wire_bytes"]
        with self._owner_lock:
            self._current_owner()
            selected = self._selected(payload)
            # Close native task/grant custody after the canonical CAS replay.
            self._current_owner()
            self._canonical._check_source()
            self._service._check_source()
            self._canonical._working_file(selected.entry, selected.data)
            self._service._working_file(selected.entry, selected.data)
        self._context_current()
        result = {key: response["result"][key] for key in _PUBLIC_FIELDS}
        # Hash the complete native JSON body; its LF/wire counters are distinct.
        result.update(schema=protocol.RESULT_SCHEMA, request_cid=response["result"]["request_cid"],
            source_binding_cid=self._binding["binding_cid"],
            backend_response_sha256=_sha(canonical_json_bytes(response)),
            backend_response_bytes=response["result"]["response_bytes"],
            backend_response_wire_bytes=response["result"]["response_wire_bytes"])
        return result

    def _response(self, request, *, result=None, reason=None):
        response = dict(schema=protocol.RESPONSE_SCHEMA, session_id=self._session_id,
            request_id=request["request_id"], ok=reason is None, result=result, reason_code=reason)
        _count_response(response)
        return response

    def _answer(self, request, channel):
        if not self._live():
            return self._response(request, reason="session_closed")
        if request["session_id"] != self._session_id or request["request_id"] in self._seen:
            return self._response(request, reason="invalid_request")
        self._seen.add(request["request_id"])
        if self._response_bytes + MAX_RESPONSE_BYTES > MAX_TOTAL_RESPONSE_BYTES:
            return self._response(request, reason="budget_exhausted")
        try:
            self._peer(channel)
        except (OSError, ValueError):
            return self._response(request, reason="peer_refused")
        try:
            result = self._read_source(request["payload"])
            self._peer(channel)
        except (OSError, ValueError, RuntimeError):
            return self._response(request, reason="retrieval_refused")
        response = self._response(request, result=result)
        if response["response_bytes"] > MAX_RESPONSE_BYTES:
            return self._response(request, reason="retrieval_refused")
        return response

    def _serve(self):
        while self._live() and self._accepted < MAX_REQUESTS:
            try:
                channel, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self._accounting_lock:
                self._accepted += 1
                self._inflight = self._accepted
            started = time.monotonic_ns()
            record = dict(ordinal=self._accepted, status="invalid_request", request_sha256=None,
                response_sha256=None, response_bytes=0, response_wire_bytes=0, delivered=False)
            with channel:
                channel.settimeout(2)
                try:
                    request = protocol.validate_worker_request(protocol.receive_frame(
                        channel, max_bytes=protocol.MAX_REQUEST_BYTES))
                    record["request_sha256"] = _sha(protocol.canonical_json(request))
                    response = self._answer(request, channel)
                    if not self._live():
                        response = self._response(request, reason="session_closed")
                    if self._response_bytes + response["response_bytes"] > MAX_TOTAL_RESPONSE_BYTES:
                        record["status"] = "budget_exhausted"
                        continue
                    raw = protocol.canonical_json(response)
                    record.update(status="ok" if response["ok"] else response["reason_code"],
                        response_sha256=_sha(raw), response_bytes=response["response_bytes"],
                        response_wire_bytes=response["response_wire_bytes"])
                    with self._accounting_lock:
                        self._response_bytes += response["response_bytes"]
                    protocol.send_frame(channel, response)
                    record["delivered"] = True
                except (OSError, ValueError, RuntimeError, RecursionError):
                    pass
                finally:
                    record["elapsed_ns"] = time.monotonic_ns() - started
                    with self._accounting_lock:
                        self._records.append(record)
                        self._inflight = None
        self._listener.close()

    def _dispose(self):
        self._closed = True
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
        errors = []
        if self._grant is not None:
            try:
                self._original_refs[1].revoke_grant(self._original_grant_id)
                self._grant_revoked = True
            except Exception as error:
                errors.append(error)
        if self._client is not None:
            # Do not wait on an in-flight request lock during bounded shutdown.
            client = getattr(self, "_original_client", self._client)
            try:
                client._poison_transport()
            except Exception as error:
                errors.append(error)
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=3)
        for path, identity in ((self._paths[4], self._socket_identity), (self._paths[5], self._descriptor_identity)):
            try:
                _unlink_same(path, identity)
            except Exception as error:
                errors.append(error)
        return errors

    def close(self):
        if self._closed:
            return
        errors = self._dispose()
        with self._accounting_lock:
            stopped = self._thread is None or not self._thread.is_alive()
            original = protocol.strict_json_loads(self._descriptor_bytes, max_bytes=8192)
            audit = dict(schema=AUDIT_SCHEMA, session=original, descriptor_sha256=_sha(self._descriptor_bytes),
                requests=list(self._records), accepted_requests=self._accepted, inflight_request=self._inflight,
                response_bytes=self._response_bytes, backend_response_bytes=self._native_response_bytes,
                backend_response_wire_bytes=self._native_wire_bytes, service_stopped=stopped,
                backend_accounting_scope="validated-source-success-envelopes",
                accounting_complete=stopped and self._accepted == len(self._records),
                native_proxy_grant_revoked=self._grant_revoked, cleanup_complete=stopped and not errors,
                cleanup_error_types=[type(error).__name__ for error in errors], retrieval_provider_calls=0,
                proof_authority=False, execution_authority=False, completion_authority=False)
        self._parent_current()
        _write_new(self._paths[6], audit, 0o400)
        if errors:
            raise ValueError("source worker cleanup refused a changed resource") from errors[0]


def start_source_lookup_worker_broker(*, server, canonical_repository, workspace, boundary,
                                    prompt, timeout_seconds=90):
    return SourceLookupWorkerBroker(server=server, canonical_repository=canonical_repository,
        workspace=workspace, boundary=boundary, prompt=prompt, timeout_seconds=timeout_seconds)
