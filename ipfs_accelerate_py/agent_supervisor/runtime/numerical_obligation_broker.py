"""Task-bound read-only numerical pages for one isolated coding invocation.

The descriptor is public routing information. Kernel peer credentials and the
actual launched child's ancestry admit readers; neither the descriptor nor a
query conveys access to owner databases or any proof/completion authority.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import os
from pathlib import Path
import socket
import struct
import threading
import time
import uuid

from . import numerical_obligation_client as protocol

MAX_REQUESTS = 64
MAX_TOTAL_RESPONSE_BYTES = 524288
MAX_SESSION_SECONDS = 1800
AUDIT_SCHEMA = "supervisor-numerical-obligation-session@1"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _argument(argv, name, default=None):
    values = []
    for index, arg in enumerate(argv):
        if arg == name:
            if index + 1 == len(argv) or argv[index + 1].startswith("--"):
                raise ValueError("numerical broker argument lacks a value")
            values.append(argv[index + 1])
        elif arg.startswith(name + "="):
            values.append(arg.split("=", 1)[1])
    if len(values) > 1:
        raise ValueError("numerical broker argument is ambiguous")
    return values[0] if values else default


def _write_new(path, value, mode):
    raw = protocol.canonical_json(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    identity = info.st_dev, info.st_ino
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
    except BaseException:
        _unlink_same(path, identity)
        raise
    return _sha(raw), identity


def _unlink_same(path, identity):
    if identity is None:
        return
    try:
        actual = path.lstat()
        if (actual.st_dev, actual.st_ino) != identity:
            raise ValueError("numerical broker path identity changed")
        path.unlink()
    except FileNotFoundError:
        pass


class NumericalObligationBroker:
    """Owner-created invocation service; use the sealed-prompt factory below."""

    protocol = protocol
    audit_schema = AUDIT_SCHEMA
    file_prefix = "numerical"

    def __init__(self, *, repository, workspace, boundary, payload, evidence_text,
                 selection, timeout_seconds):
        from . import source_formalization_context as guard

        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= MAX_SESSION_SECONDS:
            raise ValueError("bounded numerical broker session required")
        self.repository, self.workspace = guard._workspace(repository, workspace)
        self.boundary = Path(boundary).absolute()
        boundary_value, boundary_sha = guard._owner_boundary(self.boundary)
        self.owner_uid, self.worker_uid = boundary_value["owner_uid"], boundary_value["worker_uid"]
        if not (self.owner_uid > 0 and self.worker_uid > 0 and os.getgid() == self.owner_uid):
            raise ValueError("numerical broker requires the isolated nonroot owner group")
        if payload.get("repository") != str(self.repository) or guard._json(selection.get("worker_evidence")) != evidence_text:
            raise ValueError("numerical broker differs from selected task evidence")
        # Keep an independent canonical copy. Callers cannot mutate this selection
        # after it has been admitted, and request data never supplies a locator.
        self.payload = deepcopy(payload)
        self.selection = deepcopy({k: v for k, v in selection.items() if k != "worker_evidence"})
        self.receipt = self.selection["numerical_formalization"]
        self.sources = {}
        for key in ("source384_context", "numerical_formalization", "numerical_selection_384"):
            for name, digest in self.selection.get(key, {}).get("source_hashes", {}).items():
                if name in self.sources and self.sources[name] != digest:
                    raise ValueError("numerical broker selected source populations disagree")
                self.sources[name] = digest
        if len(self.sources) > 768:
            raise ValueError("numerical broker source population exceeds bound")
        identity = dict(task_id=payload["task_id"], task_cid=payload["task_cid"],
            context_bundle_sha256=payload["context_bundle"]["sha256"],
            context_receipt_sha256=self.receipt["receipt_sha256"],
            source_tree_id=self.receipt["record"]["source_tree_id"])
        self._start_session(identity=identity, boundary_sha=boundary_sha, timeout_seconds=timeout_seconds)

    def _start_session(self, *, identity, boundary_sha, timeout_seconds, descriptor_schema=None):
        """Shared bounded socket lifecycle; callers independently admit source semantics."""
        from . import source_formalization_context as guard

        self.parent = self.boundary.parent / "worker-observations"
        guard._protected_directory(self.parent, owner=self.owner_uid, group=self.owner_uid)
        session = uuid.uuid4().hex
        self.socket_path = self.parent / (self.file_prefix + "-" + session + ".sock")
        self.descriptor_path = self.parent / (self.file_prefix + "-" + session + ".json")
        self.audit_path = self.parent / (self.file_prefix + "-" + session + "-audit.json")
        self._socket_identity = self._descriptor_identity = None
        self._stop = threading.Event()
        self._registered = threading.Event()
        self._child = None
        self._process = None
        self._thread = None
        self._closed = False
        self._records = []
        self._accepted = 0
        self._inflight = None
        self._response_bytes = 0
        self._seen = set()
        self._accounting_lock = threading.Lock()
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.settimeout(0.2)
        now = time.monotonic_ns()
        owner = self.protocol.process_identity(os.getpid())
        try:
            self._listener.bind(str(self.socket_path))
            info = self.socket_path.lstat()
            self._socket_identity = info.st_dev, info.st_ino
            os.chmod(self.socket_path, 0o660)
            self.descriptor = dict(schema=(self.protocol.DESCRIPTOR_SCHEMA if descriptor_schema is None
                                           else descriptor_schema), session_id=session,
                owner_uid=self.owner_uid, worker_uid=self.worker_uid, owner_pid=owner["pid"],
                owner_start_time_ticks=owner["start_time_ticks"], workspace=str(self.workspace),
                **identity,
                boundary_manifest_sha256=boundary_sha,
                socket_path=str(self.socket_path), socket_device=info.st_dev, socket_inode=info.st_ino,
                issued_monotonic_ns=now, expires_monotonic_ns=now+timeout_seconds*1_000_000_000,
                max_requests=MAX_REQUESTS, max_total_response_bytes=MAX_TOTAL_RESPONSE_BYTES,
                proof_authority=False, execution_authority=False, completion_authority=False)
            self.protocol.validate_descriptor(self.descriptor)
            self._current_context()
            self.descriptor_sha256, self._descriptor_identity = _write_new(self.descriptor_path, self.descriptor, 0o440)
            self.extra_argv = [self.protocol.DESCRIPTOR_FLAG, str(self.descriptor_path),
                               self.protocol.DIGEST_FLAG, self.descriptor_sha256]
            self._listener.listen(4)
            self._thread = threading.Thread(target=self._serve, name=self.file_prefix + "-reader", daemon=True)
            self._thread.start()
        except BaseException:
            self._listener.close()
            _unlink_same(self.socket_path, self._socket_identity)
            _unlink_same(self.descriptor_path, self._descriptor_identity)
            raise

    def note_spawned(self, process):
        if self._closed or self._registered.is_set() or process.poll() is not None:
            raise ValueError("numerical broker requires one live spawned child")
        child = self.protocol.process_identity(process.pid)
        if child["parent_pid"] != os.getpid():
            raise ValueError("numerical broker child belongs to another owner")
        self._child, self._process = child, process
        self._registered.set()

    def _live(self):
        if (self._stop.is_set() or time.monotonic_ns() >= self.descriptor["expires_monotonic_ns"]
                or self._process is not None and self._process.poll() is not None):
            return False
        return True

    def _peer(self, channel):
        if not self._registered.wait(timeout=2.) or not self._live():
            raise ValueError("inactive numerical retrieval session")
        pid, uid, _ = struct.unpack("3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if uid != self.worker_uid:
            raise ValueError("numerical retrieval peer uid differs")
        first = self.protocol.process_identity(pid)
        if first["uid"] != uid:
            raise ValueError("numerical retrieval peer changed uid")
        current, seen = first, set()
        for _ in range(64):
            if current["pid"] in seen:
                break
            seen.add(current["pid"])
            if current["pid"] == self._child["pid"]:
                actual = self.protocol.process_identity(current["pid"])
                if (actual["start_time_ticks"] != self._child["start_time_ticks"]
                        or actual["parent_pid"] != os.getpid()
                        or self.protocol.process_identity(pid) != first):
                    break
                return first
            if current["parent_pid"] <= 1 or current["parent_pid"] == os.getpid():
                break
            current = self.protocol.process_identity(current["parent_pid"])
        raise ValueError("numerical retrieval peer is outside the launched child")

    def _current_context(self):
        from . import source_formalization_context as guard
        from .task_context_bundle import read_task_context_historical_selection

        if not self._live():
            raise ValueError("numerical retrieval session expired")
        if guard._owner_boundary(self.boundary)[1] != self.descriptor["boundary_manifest_sha256"]:
            raise ValueError("numerical retrieval boundary changed")
        root, allocated = guard._workspace(self.repository, self.workspace)
        binding = self.payload["context_bundle"]
        # This checks envelope and receipt identity only. Each successful page
        # separately replays its live numerical owner via the native pager.
        current = read_task_context_historical_selection(repository=root, artifact=binding["artifact"],
            expected_sha256=binding["sha256"], task_cid=self.payload["task_cid"], task_id=self.payload["task_id"])
        if self.protocol.canonical_json(current) != self.protocol.canonical_json(self.selection):
            raise ValueError("numerical retrieval task selection changed")
        guard._current_sources(root, self.sources)
        guard._current_sources(allocated, self.sources)

    def _response(self, request, *, page=None, reason=None):
        return dict(schema=self.protocol.RESPONSE_SCHEMA, session_id=self.descriptor["session_id"],
            request_id=request["request_id"], ok=reason is None, page=page, reason_code=reason)

    def _answer(self, request, channel):
        from .numerical_formalization_context import retrieve_numerical_obligations

        if not self._live():
            return self._response(request, reason="session_closed")
        if request["session_id"] != self.descriptor["session_id"] or request["request_id"] in self._seen:
            return self._response(request, reason="invalid_request")
        self._seen.add(request["request_id"])
        if self._response_bytes + request["query"]["max_bytes"] > MAX_TOTAL_RESPONSE_BYTES:
            return self._response(request, reason="budget_exhausted")
        try:
            self._peer(channel)
        except (OSError, ValueError):
            return self._response(request, reason="peer_refused")
        try:
            self._current_context()
        except (OSError, ValueError, RuntimeError):
            return self._response(request, reason="stale_context")
        # Reserve the exact transport overhead before invoking the native page
        # builder. Fixed-width request/session IDs keep continuation budgets stable.
        overhead = len(self.protocol.canonical_json(self._response(request, page={}))) - 2
        query = dict(request["query"], max_bytes=request["query"]["max_bytes"] - overhead)
        try:
            page = retrieve_numerical_obligations(repository=self.repository, expected_receipt=self.receipt, **query)
        except (OSError, ValueError, RuntimeError):
            return self._response(request, reason="retrieval_refused")
        try:
            self._current_context()
            self._peer(channel)
        except (OSError, ValueError, RuntimeError):
            return self._response(request, reason="stale_context")
        response = self._response(request, page=page)
        if len(self.protocol.canonical_json(response)) > request["query"]["max_bytes"]:
            return self._response(request, reason="retrieval_refused")
        return response

    def _record_details(self, response=None):
        return dict(returned_rows=response["page"]["returned_count"]
            if response is not None and response["ok"] else 0)

    def _serve(self):
        accepted = 0
        while self._live() and accepted < MAX_REQUESTS:
            try:
                channel, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            accepted += 1
            with self._accounting_lock:
                self._accepted = accepted
                self._inflight = accepted
            started = time.monotonic_ns()
            record = dict(ordinal=accepted, status="invalid_request", request_sha256=None,
                response_sha256=None, response_bytes=0, delivered=False, **self._record_details())
            with channel:
                channel.settimeout(2.)
                try:
                    request = self.protocol.validate_request(self.protocol.receive_frame(channel))
                    record["request_sha256"] = _sha(self.protocol.canonical_json(request))
                    response = self._answer(request, channel)
                    if self._stop.is_set():
                        raise ValueError("numerical retrieval stopped before response")
                    if not self._live():
                        response = self._response(request, reason="session_closed")
                    raw = self.protocol.canonical_json(response)
                    record.update(status="ok" if response["ok"] else response["reason_code"],
                        response_sha256=_sha(raw), response_bytes=len(raw),
                        **self._record_details(response))
                    # Account bytes prepared even if the requesting process exits.
                    with self._accounting_lock:
                        self._response_bytes += len(raw)
                    self.protocol.send_frame(channel, response)
                    record["delivered"] = True
                except (OSError, ValueError, RuntimeError, RecursionError):
                    pass
                finally:
                    record["elapsed_ns"] = time.monotonic_ns() - started
                    with self._accounting_lock:
                        self._records.append(record)
                        self._inflight = None
        self._listener.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._listener.close()
        try:
            _unlink_same(self.socket_path, self._socket_identity)
        finally:
            _unlink_same(self.descriptor_path, self._descriptor_identity)
        if self._thread is not None:
            self._thread.join(timeout=3.)
        with self._accounting_lock:
            records = list(self._records)
            total = self._response_bytes
            accepted, inflight = self._accepted, self._inflight
        stopped = self._thread is None or not self._thread.is_alive()
        _write_new(self.audit_path, dict(schema=self.audit_schema, session=self.descriptor,
            descriptor_sha256=self.descriptor_sha256, requests=records,
            accepted_requests=accepted, inflight_request=inflight, response_bytes=total,
            service_stopped=stopped, accounting_complete=stopped and accepted == len(records),
            retrieval_provider_calls=0, source_obligations_discharged=0,
            proof_authority=False, execution_authority=False, completion_authority=False), 0o400)


def prepare_numerical_obligation_broker(*, argv, boundary, workspace, input_fd=0):
    """Derive the optional service exclusively from the owner's sealed prompt."""
    import fcntl
    from . import source_formalization_context as guard

    if (type(argv) is not list or any(type(arg) is not str for arg in argv)
            or any(arg == flag or arg.startswith(flag + "=") for arg in argv
                   for flag in (protocol.DESCRIPTOR_FLAG, protocol.DIGEST_FLAG))):
        raise ValueError("owner creates numerical retrieval descriptors independently")
    required = sum(getattr(fcntl, name, value) for name, value in (
        ("F_SEAL_SEAL", 1), ("F_SEAL_SHRINK", 2), ("F_SEAL_GROW", 4), ("F_SEAL_WRITE", 8)))
    if fcntl.fcntl(input_fd, getattr(fcntl, "F_GET_SEALS", 1034)) & required != required:
        raise ValueError("numerical broker requires the sealed owner prompt")
    size = os.fstat(input_fd).st_size
    if size > 2 * 1024 * 1024:
        raise ValueError("numerical broker prompt exceeds bound")
    parsed = guard._parts(os.pread(input_fd, size + 1, 0).decode("utf-8"))
    if parsed is None:
        return None
    payload, text, _ = parsed
    if not any(row["kind"] == "numerical-formalization-context" for row in payload["contexts"]):
        return None
    if _argument(argv, "--purpose", "coding") != "coding":
        raise ValueError("numerical retrieval requires an isolated coding invocation")
    repository = _argument(argv, "--semantic-repository")
    if repository is None:
        raise ValueError("numerical retrieval requires the selected repository")
    root, allocated = guard._workspace(repository, workspace)
    binding = payload["context_bundle"]
    selection = guard.load_task_context_selection(repository=root, artifact=binding["artifact"],
        expected_sha256=binding["sha256"], task_cid=payload["task_cid"], task_id=payload["task_id"],
        source384_timeout_seconds=90.)
    return NumericalObligationBroker(repository=root, workspace=allocated, boundary=boundary,
        payload=payload, evidence_text=text, selection=selection,
        timeout_seconds=int(_argument(argv, "--timeout", "90")))
