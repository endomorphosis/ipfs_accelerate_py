"""Explicit native-parent repository reader behind the existing worker boundary.

This opt-in service is not wired into Terminal Bench production launches. The
owner retains native scheduler and index authority. The worker receives only a
public descriptor and bounded JSON-RPC replies. ``read_only`` describes source:
capture and refresh write an owner-private index. No canonical edit is exposed.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
from pathlib import Path
import socket
import threading
import time

from . import repository_source_owner_protocol as protocol
from . import repository_source_policy as policy
from . import repository_source_mcp as mcp
from . import source_formalization_context as guard
from .numerical_obligation_broker import NumericalObligationBroker, _unlink_same, _write_new
from .repository_source_resource_handoff import DEMAND, validate_owner_resources

MAX_REQUESTS = 64
MAX_TOTAL_RESPONSE_BYTES = 524288
AUDIT_SCHEMA = "repository-source-owner-session@2"


class RepositoryOwnerConstructionError(RuntimeError):
    """A partially started service remains reachable for stopped cleanup."""
    def __init__(self, owner):
        super().__init__("repository_owner_construction_interrupted")
        self.owner = owner


def _need(value, reason="invalid_repository_owner_service"):
    if not value:
        raise ValueError(reason)


class _Stop:
    def __init__(self, owner):
        self.owner = owner

    def is_set(self):
        owner = self.owner
        descriptor = getattr(owner, "descriptor", None)
        return (owner._stop.is_set() or time.monotonic() >= owner.deadline_monotonic
                or descriptor is not None and time.monotonic_ns() >= descriptor["expires_monotonic_ns"]
                or owner.parent_lease.released or owner.parent_lease.cancelled
                or owner._native is not None and (owner._native.released or owner._native.cancelled)
                or owner.external_cancel is not None and owner.external_cancel.is_set())


class OwnerRepositoryMCPBroker(NumericalObligationBroker):
    """One authenticated worker session; the caller retains its actual parent.

    Only the service thread constructs, handles and closes the native MCP
    server and releases its child. ``close`` may report cleanup_pending; it
    never closes an active handler or releases its resources concurrently.
    Call close again after the cooperating handler returns. A failed native
    close retains custody and reports incomplete cleanup rather than guessing.
    """

    protocol = protocol
    audit_schema = AUDIT_SCHEMA
    file_prefix = "repository-mcp"

    def __init__(self, *, repository, workspace, boundary, parent_lease,
                 expected_policy_sha256, deadline_monotonic,
                 representation="original", cancel_event=None):
        # A trusted lifecycle may preallocate this object to preserve failed
        # construction custody. Failures before any store/thread creation have
        # a known empty cleanup scope, rather than an unreachable partial owner.
        self._resource_creation_started = False
        validate_owner_resources(parent_lease=parent_lease,
            expected_policy_sha256=expected_policy_sha256, deadline_monotonic=deadline_monotonic)
        _need(1 <= deadline_monotonic - time.monotonic() <= 1800)
        _need(type(representation) is str and representation in {"original", "compact"})
        _need(cancel_event is None or callable(getattr(cancel_event, "is_set", None)))
        _need(cancel_event is None or not cancel_event.is_set())
        self.repository, self.workspace = guard._workspace(repository, workspace)
        self.boundary = Path(boundary).absolute()
        value, boundary_sha = guard._owner_boundary(self.boundary)
        self.owner_uid, self.worker_uid = value["owner_uid"], value["worker_uid"]
        _need(os.getgid() == self.owner_uid and self.worker_uid != self.owner_uid)
        roots = value.get("allowed_worktree_roots")
        _need(type(roots) is list and 1 <= len(roots) <= 8
              and any(type(root) is str and self.workspace != Path(root)
                      and self.workspace.is_relative_to(Path(root)) for root in roots))
        self._workspace_identity = self._workspace_stat()
        self._boundary_sha256 = boundary_sha
        self.parent_lease = parent_lease
        self.deadline_monotonic = deadline_monotonic
        self.external_cancel = cancel_event
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler
        self._scheduler = open_existing_scheduler(parent_lease.token.state_path, expected_policy_sha256)
        self._native = self._server = None
        self._native_acquisition_uncertain = False
        self._usage = None
        self._service_error = self._cleanup_error = None
        self._child_record = None
        self._retained_mcp_audit = None
        self._done = threading.Event()
        self._ready = threading.Event()
        self._close_lock = threading.Lock()
        self._retire_lock = threading.Lock()
        self._audit_written = False
        self._budget_exhausted = False
        self._store_identity = None
        self._thread = None
        self._thread_launch_attempted = False
        self._stop = threading.Event()
        self._closed = False
        self._construction_error = None
        self._prelaunch_cleanup_complete = False
        self._binding = policy.create_binding(workspace=self.workspace,
            requested_policy=policy.POLICIES[1 if representation == "original" else 2])
        self._signal = _Stop(self)
        self._resource_creation_started = True
        try:
            self._store_identity = policy.create_runtime_store(self._binding)
            remaining = deadline_monotonic - time.monotonic()
            _need(remaining >= 1, "repository_owner_deadline_expired")
            self._start_session(identity=dict(representation=representation,
                config_sha256=self._binding["config_sha256"],
                helper_sha256=self._binding["helper_sha256"], read_only=True),
                boundary_sha=boundary_sha, timeout_seconds=max(1, math.floor(remaining) - 1),
                descriptor_schema=protocol.DESCRIPTOR_SCHEMA)
        except BaseException as error:
            self._construction_error = type(error).__name__
            # Thread.start can be interrupted before ident becomes visible.
            # Only a known prelaunch failure may clean the store on this
            # thread. An uncertain launch keeps its original cleanup owner.
            try:
                result = self.close(timeout_seconds=0)
            except BaseException:
                result = None
            if result is not None and result["cleanup_complete"]:
                raise
            raise RepositoryOwnerConstructionError(self) from error

    def _start_service_thread(self):
        self._thread_launch_attempted = True
        self._thread.start()

    def _workspace_stat(self):
        _need(self.workspace.resolve(strict=True) == self.workspace and self.workspace.is_dir())
        info = self.workspace.lstat()
        return info.st_dev, info.st_ino

    def _current_context(self):
        _need(self.descriptor["expires_monotonic_ns"] <= int(self.deadline_monotonic * 1_000_000_000),
              "repository_owner_descriptor_deadline_extended")
        _need(guard._owner_boundary(self.boundary)[1] == self._boundary_sha256,
              "repository_owner_boundary_changed")
        _need(self._workspace_stat() == self._workspace_identity,
              "repository_owner_workspace_changed")
        _need(not self._signal.is_set(), "repository_owner_stopped")

    def _live(self):
        return super()._live() and not self._signal.is_set()

    def _retire(self):
        with self._retire_lock:
            listener = getattr(self, "_listener", None)
            if listener is not None:
                listener.close()
            try:
                if getattr(self, "_socket_identity", None) is not None:
                    _unlink_same(self.socket_path, self._socket_identity)
            finally:
                if getattr(self, "_descriptor_identity", None) is not None:
                    _unlink_same(self.descriptor_path, self._descriptor_identity)

    def _response(self, request, *, message=None, reason=None):
        return dict(schema=protocol.RESPONSE_SCHEMA, session_id=self.descriptor["session_id"],
            request_id=request["request_id"], ok=reason is None, message=message, reason_code=reason)

    def _authenticate_peer(self, channel):
        try:
            self._peer(channel)
        except (ValueError, OSError) as error:
            raise protocol.RepositorySourceOwnerProtocolError("peer_refused") from error

    def _answer(self, request, channel, *, operation_capture):
        if request["session_id"] != self.descriptor["session_id"] or request["request_id"] in self._seen:
            return self._response(request, reason="invalid_request")
        self._seen.add(request["request_id"])
        self._authenticate_peer(channel)
        self._current_context()
        exhausted = self._response(request, reason="budget_exhausted")
        if (self._budget_exhausted or MAX_TOTAL_RESPONSE_BYTES - self._response_bytes
                <= len(protocol.canonical_json(exhausted))):
            self._budget_exhausted = True
            return exhausted
        message = request["message"]
        if message.get("method") == "tools/call" and message.get("params", {}).get("name") == "repository_replace_symbol":
            return self._response(request, reason="invalid_request")
        operation_capture["status"] = "incomplete"
        with self._server.capture_audit_operations(operation_capture):
            reply = self._server.handle(message)
        if message.get("method") == "tools/list" and type(reply) is dict and "result" in reply:
            reply = copy.deepcopy(reply)
            reply["result"]["tools"] = [tool for tool in reply["result"]["tools"]
                if tool["name"] != "repository_replace_symbol"]
        self._current_context()
        self._authenticate_peer(channel)
        return self._response(request, message=reply)

    def _retain_mcp_audit(self):
        """Retain exact closed content-free operation records before cleanup."""
        source = Path(self._binding["audit_store"])
        if not source.exists():
            return
        summary = mcp.summarize_repository_audit(source)
        _need(summary.get("status") in {"observed", "not_observed"}, "invalid_mcp_audit")
        raw = policy._read_file(source, mcp.MAX_AUDIT_BYTES)
        target = self.audit_path.with_name(self.audit_path.stem + "-operations.jsonl")
        descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(descriptor)
            os.fchmod(descriptor, 0o400)
        finally:
            os.close(descriptor)
        self._retained_mcp_audit = dict(name=target.name, sha256=hashlib.sha256(raw).hexdigest(),
                                       bytes=len(raw), content_free=True)

    def _serve_requests(self):
        for ordinal in range(1, MAX_REQUESTS + 1):
            while self._live():
                try:
                    channel, _ = self._listener.accept()
                    break
                except socket.timeout:
                    continue
                except OSError:
                    return
            else:
                return
            with self._accounting_lock:
                self._accepted, self._inflight = ordinal, ordinal
            record = dict(ordinal=ordinal, status="invalid_request", request_sha256=None,
                prepared_response_sha256=None, prepared_response_bytes=None,
                response_sha256=None, delivered=False, response_bytes=0,
                operation_capture={"status": "not_entered", "operations": []})
            started = time.monotonic_ns()
            request = response = None
            with channel:
                try:
                    self._authenticate_peer(channel)
                    request = protocol.validate_request(protocol.receive_frame(channel,
                        deadline_monotonic=self.deadline_monotonic, cancel_event=self._signal,
                        max_bytes=protocol.MAX_REQUEST_FRAME_BYTES))
                    record["request_sha256"] = hashlib.sha256(protocol.canonical_json(request)).hexdigest()
                    response = self._answer(request, channel, operation_capture=record["operation_capture"])
                    raw = protocol.canonical_json(response)
                    record.update(prepared_response_sha256=hashlib.sha256(raw).hexdigest(),
                                  prepared_response_bytes=len(raw))
                    if len(raw) > MAX_TOTAL_RESPONSE_BYTES - self._response_bytes:
                        self._budget_exhausted = True
                        response = self._response(request, reason="budget_exhausted")
                        raw = protocol.canonical_json(response)
                    if len(raw) > MAX_TOTAL_RESPONSE_BYTES - self._response_bytes:
                        record["status"] = "budget_exhausted"
                        self._stop.set()
                        continue
                    protocol.validate_response(response, request=request, descriptor=self.descriptor)
                    self._current_context()
                    self._authenticate_peer(channel)
                    record.update(status="ok" if response["ok"] else response["reason_code"],
                                  response_bytes=len(raw), response_sha256=hashlib.sha256(raw).hexdigest())
                    with self._accounting_lock:
                        self._response_bytes += len(raw)
                    protocol.send_frame(channel, response,
                        deadline_monotonic=self.deadline_monotonic, cancel_event=self._signal)
                    record["delivered"] = True
                except Exception as error:
                    record["status"] = getattr(error, "reason_code", "retrieval_refused")
                    if type(record["status"]) is not str or record["status"] not in protocol.REASON_CODES:
                        record["status"] = "retrieval_refused"
                finally:
                    record["elapsed_ns"] = time.monotonic_ns() - started
                    with self._accounting_lock:
                        self._records.append(record)
                        self._inflight = None

    def _serve(self):
        server_closed = False
        try:
            from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
                LeaseCancelledError, LeaseTimeoutError, ResourceLane,
            )
            # Persistence may precede the returned lease handle. An unknown
            # interruption in that window cannot prove that no child exists.
            self._native_acquisition_uncertain = True
            try:
                self._native = self._scheduler.acquire(lane=ResourceLane.VALIDATION,
                    parent_lease=self.parent_lease.token, **DEMAND,
                    timeout=min(10., max(0., self.deadline_monotonic - time.monotonic())),
                    cancel_event=self._signal, request_id="repository-owner:" + self.descriptor["session_id"])
            except (LeaseCancelledError, LeaseTimeoutError):
                self._native_acquisition_uncertain = False
                raise
            self._native_acquisition_uncertain = False
            self._child_record = self._native.to_dict()
            self._current_context()
            self._server = mcp.RepositoryMCPServer(self._binding["config"], parent_lease=self._native,
                cancel_event=self._signal, deadline_monotonic=self.deadline_monotonic)
            self._ready.set()
            self._serve_requests()
        except BaseException as error:
            self._service_error = type(error).__name__
            if self._native is not None:
                self._native_acquisition_uncertain = False
        finally:
            try:
                if self._server is not None:
                    self._server.close()
                server_closed = True
            except BaseException as error:
                self._cleanup_error = type(error).__name__
            if server_closed:
                try:
                    self._retain_mcp_audit()
                    self._usage = policy.finish_runtime_store(self._binding,
                        expected_identity=self._store_identity)
                    if self._native is not None:
                        self._native.release()
                        self._child_record = self._native.to_dict()
                except BaseException as error:
                    self._cleanup_error = type(error).__name__
            try:
                self._retire()
            except BaseException as error:
                self._cleanup_error = type(error).__name__
            self._ready.set()
            self._done.set()

    def close(self, *, timeout_seconds=3.):
        _need(type(timeout_seconds) in (int, float) and 0 <= timeout_seconds <= 3)
        if getattr(self, "_resource_creation_started", None) is False:
            return dict(schema=AUDIT_SCHEMA, session=None, descriptor_sha256=None,
                service_stopped=True, cleanup_complete=True, cleanup_pending=False,
                native_child=None, native_child_acquired=False, native_child_released=None,
                native_child_acquisition_uncertain=False, prepared_mcp_usage=None,
                retained_mcp_audit=None, thread_launch_attempted=False,
                resource_creation_started=False, construction_error=None,
                service_error=None, cleanup_error=None, join_error=None,
                cross_uid_deployment_qualified=False, automatic_terminal_wiring=False,
                provider_calls=0, token_savings_measured=False,
                proof_authority=False, execution_authority=False, completion_authority=False)
        with self._close_lock:
            self._closed = True
            self._stop.set()
            join_error = None
            try:
                self._retire()
            except Exception as error:
                self._cleanup_error = type(error).__name__
            if not self._thread_launch_attempted:
                stopped = True
                if not self._prelaunch_cleanup_complete:
                    try:
                        if self._store_identity is not None:
                            self._usage = policy.finish_runtime_store(self._binding,
                                expected_identity=self._store_identity)
                            self._prelaunch_cleanup_complete = self._usage.get("store_cleanup_complete") is True
                        else:
                            # No store was returned. A constructor may have
                            # been interrupted during creation; absence of a
                            # returned identity is not cleanup evidence.
                            self._prelaunch_cleanup_complete = False
                    except BaseException as error:
                        self._cleanup_error = type(error).__name__
                complete = self._prelaunch_cleanup_complete and self._cleanup_error is None
            else:
                # A start attempt with no published ident may still launch.
                # join() is illegal before start; never infer stopped here.
                if self._thread is not None and self._thread.ident is not None:
                    try:
                        self._thread.join(timeout_seconds)
                        stopped = not self._thread.is_alive() and self._done.is_set()
                    except Exception as error:
                        join_error = type(error).__name__
                        stopped = False
                else:
                    stopped = False
                complete = (stopped and (self._native is None or self._native.released)
                    and self._cleanup_error is None and self._usage is not None
                    and self._usage.get("store_cleanup_complete") is True
                    and not self._native_acquisition_uncertain)
            released = self._native is None or self._native.released
            with getattr(self, "_accounting_lock", self._retire_lock):
                records = getattr(self, "_records", [])
                accepted = getattr(self, "_accepted", 0)
                inflight = getattr(self, "_inflight", None)
                value = dict(schema=AUDIT_SCHEMA, session=getattr(self, "descriptor", None),
                    descriptor_sha256=getattr(self, "descriptor_sha256", None),
                    requests=copy.deepcopy(records), accepted_requests=accepted,
                    inflight_request=inflight, response_bytes=getattr(self, "_response_bytes", 0),
                    service_stopped=stopped, cleanup_complete=complete,
                    cleanup_pending=not complete, accounting_complete=stopped and inflight is None
                        and accepted == len(records),
                    native_child=copy.deepcopy(self._child_record),
                    native_child_acquired=(None if self._native_acquisition_uncertain else self._native is not None),
                    native_child_acquisition_uncertain=self._native_acquisition_uncertain,
                    native_child_released=released if self._native is not None else None,
                    service_error=self._service_error, cleanup_error=self._cleanup_error,
                    construction_error=self._construction_error,
                    thread_launch_attempted=self._thread_launch_attempted,
                    join_error=join_error,
                    prepared_mcp_usage=copy.deepcopy(self._usage),
                    retained_mcp_audit=copy.deepcopy(self._retained_mcp_audit),
                    canonical_source_read_only=True, cross_uid_deployment_qualified=False,
                    automatic_terminal_wiring=False, provider_calls=0,
                    token_savings_measured=False, proof_authority=False,
                    execution_authority=False, completion_authority=False)
            if stopped and not self._audit_written and getattr(self, "audit_path", None) is not None:
                _write_new(self.audit_path, value, 0o400)
                self._audit_written = True
            return value
