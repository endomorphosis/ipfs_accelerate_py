"""Explicit native custody for one owner-hosted repository source service.

Configuration alone admits nothing. ``start`` opens an exact existing native
scheduler policy and acquires the fixed source-service reservation, optionally
beneath a genuine caller lease. The existing broker owns its child and handler.
This reservation does not admit a provider, worker execution, or a benchmark.
The established caller monitor still owns worker reaping and isolated cleanup.

No context manager, automatic retry, private allocator, or cleanup thread is
provided. An incomplete close retains the actual lease and broker on this
object. Start exceptions carry that same object in ``error.owner``. Callers
must retain it and explicitly retry close when an active handler has drained;
a failed broker cleanup can require separate intervention. Native auto-renew
is required as an operational facade setting, not a persisted policy claim.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid

from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    LeaseCancelledError, LeaseTimeoutError, ResourceLane, ResourceLease,
)

from .repository_source_owner_broker import OwnerRepositoryMCPBroker, RepositoryOwnerConstructionError
from .repository_source_resource_handoff import DEMAND

SCHEMA = "repository-source-owner-lifecycle@1"
MAX_ADMISSION_SECONDS = 30.0
MAX_LIFETIME_SECONDS = 1800.0
READY_POLL_SECONDS = 0.05


class RepositorySourceOwnerLifecycleError(RuntimeError):
    """A closed failure code with the original reachable cleanup owner."""
    def __init__(self, owner, reason_code):
        self.owner, self.reason_code = owner, reason_code
        super().__init__(reason_code)


def _need(condition, reason):
    if not condition:
        raise ValueError(reason)


def _path(value):
    _need(isinstance(value, (str, Path)), "canonical_absolute_path_required")
    text = str(value)
    try:
        valid = (0 < len(text.encode("utf-8")) <= 4096
                 and not any(ord(character) < 32 for character in text))
    except UnicodeError:
        valid = False
    path = Path(text)
    _need(valid and path.is_absolute() and str(path) == text and ".." not in path.parts,
          "canonical_absolute_path_required")
    return path


def _number(value, *, minimum, maximum, reason):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum
    except OverflowError:
        valid = False
    _need(valid, reason)
    return value


class _Signal:
    def __init__(self, owner):
        self.owner = owner

    def is_set(self):
        owner = self.owner
        return (owner._stop.is_set() or time.monotonic() >= owner.deadline_monotonic
                or owner.cancel_event is not None and owner.cancel_event.is_set()
                or owner._caller_parent is not None
                    and (owner._caller_parent.released or owner._caller_parent.cancelled)
                or owner._lease is not None and (owner._lease.released or owner._lease.cancelled))


class RepositorySourceOwnerLifecycle:
    """Own only the source-service reservation until verified broker cleanup.

    ``start()`` returns this object once, or raises an error carrying it. The
    public descriptor arguments become available only after the broker is
    live. ``close()`` returns a content-free receipt, including incomplete
    cleanup, and never releases a borrowed caller parent. It never kills or
    reaps the worker registered for authentication.
    """
    def __init__(self, *, repository, workspace, boundary, scheduler_state_path,
                 expected_policy_sha256, deadline_monotonic,
                 admission_timeout_seconds=10.0, representation="original",
                 cancel_event=None, parent_lease=None):
        self.repository, self.workspace = _path(repository), _path(workspace)
        self.boundary, self.scheduler_state_path = _path(boundary), _path(scheduler_state_path)
        _need(type(expected_policy_sha256) is str
              and re.fullmatch("[0-9a-f]{64}", expected_policy_sha256) is not None,
              "exact_scheduler_policy_digest_required")
        now = time.monotonic()
        self.deadline_monotonic = _number(deadline_monotonic, minimum=now,
            maximum=now + MAX_LIFETIME_SECONDS, reason="invalid_owner_deadline")
        _need(deadline_monotonic > now, "invalid_owner_deadline")
        self.admission_timeout_seconds = _number(admission_timeout_seconds, minimum=0.0,
            maximum=MAX_ADMISSION_SECONDS, reason="invalid_admission_timeout")
        _need(admission_timeout_seconds > 0, "invalid_admission_timeout")
        _need(type(representation) is str and representation in {"original", "compact"},
              "invalid_source_representation")
        _need(cancel_event is None or callable(getattr(cancel_event, "is_set", None)),
              "invalid_cancellation_signal")
        self.expected_policy_sha256, self.representation = expected_policy_sha256, representation
        self.cancel_event, self._caller_parent = cancel_event, parent_lease
        self._owner_pid = os.getpid()
        self._validate_caller_parent()
        self._lease = self._broker = self._scheduler = None
        self._stop = threading.Event()
        self._signal = _Signal(self)
        self._lock = threading.Lock()
        self._state = "new"
        self._start_attempted = self._ready = self._spawn_registered = False
        self._admission_entered = self._admission_uncertain = self._admission_inflight = False
        self._admission_deadline = None
        self._broker_constructed = False
        self._broker_audit = None
        self._cleanup_complete = False
        self._renewal_unproven = False
        self._error = self._cleanup_error = None
        self._request_id = "repository-source-lifecycle:" + uuid.uuid4().hex

    def _validate_caller_parent(self):
        parent = self._caller_parent
        if parent is None:
            return
        _need(type(parent) is ResourceLease and parent.owner_pid == os.getpid()
              and not parent.released, "genuine_live_caller_parent_required")
        _need(_path(parent.token.state_path) == self.scheduler_state_path,
              "caller_parent_scheduler_path_differs")
        _need(not parent.cancelled, "genuine_live_caller_parent_required")
        _need(all(type(getattr(parent, name, None)) is int and getattr(parent, name) >= value
                  for name, value in DEMAND.items()), "caller_parent_capacity_insufficient")

    def _checkpoint(self):
        _need(os.getpid() == self._owner_pid, "lifecycle_belongs_to_another_process")
        if self._signal.is_set():
            raise RepositorySourceOwnerLifecycleError(self, "owner_stopped")

    def _admission_checkpoint(self):
        self._checkpoint()
        if time.monotonic() >= self._admission_deadline:
            raise LeaseTimeoutError("repository_owner_admission_deadline")

    def _live_broker(self):
        self._checkpoint()
        broker = self._broker
        _need(self._ready and self._broker_constructed and broker is not None
              and broker._ready.is_set() and not broker._done.is_set()
              and broker._service_error is None and broker._native is not None
              and broker._server is not None and broker._thread is not None
              and broker._thread.is_alive(), "repository_owner_not_ready")
        broker._current_context()

    def start(self):
        """Perform one bounded admission and readiness handshake, without retry."""
        with self._lock:
            if self._start_attempted or self._state != "new":
                raise RepositorySourceOwnerLifecycleError(self, "owner_already_started_or_closed")
            self._start_attempted, self._state = True, "starting"
            self._admission_deadline = min(self.deadline_monotonic,
                time.monotonic() + self.admission_timeout_seconds)
            stage = "native_policy_unavailable"
            try:
                self._checkpoint()
                self._validate_caller_parent()
                self._scheduler = open_existing_scheduler(self.scheduler_state_path,
                                                          self.expected_policy_sha256)
                _need(_path(self._scheduler.state_path) == self.scheduler_state_path,
                      "selected_scheduler_path_differs")
                _need(self._scheduler.config.proof_safety_enabled is True,
                      "native_proof_safety_required")
                _need(self._scheduler.config.auto_renew_leases is True,
                      "native_auto_renew_required")
                self._admission_checkpoint()
                _need(self.deadline_monotonic - time.monotonic() >= 1.0,
                      "insufficient_owner_start_lifetime")
                stage = "native_admission_failed"
                self._admission_entered = self._admission_inflight = True
                try:
                    self._lease = self._scheduler.acquire(lane=ResourceLane.VALIDATION,
                        parent_lease=self._caller_parent, **DEMAND,
                        timeout=max(0.0, self._admission_deadline - time.monotonic()),
                        cancel_event=self._signal, request_id=self._request_id)
                except (LeaseTimeoutError, LeaseCancelledError):
                    # Native documented refusal paths remove their waiter.
                    raise
                except BaseException:
                    # An asynchronous interruption can occur after persistence
                    # but before a lease handle returns. Do not invent recovery
                    # authority or assert that no native reservation exists.
                    self._admission_uncertain = self._lease is None
                    raise
                finally:
                    self._admission_inflight = False
                _need(type(self._lease) is ResourceLease and self._lease.owner_pid == self._owner_pid
                      and _path(self._lease.token.state_path) == self.scheduler_state_path,
                      "native_admission_returned_invalid_lease")
                self._admission_checkpoint()
                _need(self.deadline_monotonic - time.monotonic() >= 1.0,
                      "insufficient_owner_start_lifetime")
                stage = "repository_owner_construction_failed"
                # Keep the exact partially initialized broker even if an
                # exception occurs before its constructor can wrap custody.
                self._broker = OwnerRepositoryMCPBroker.__new__(OwnerRepositoryMCPBroker)
                try:
                    OwnerRepositoryMCPBroker.__init__(self._broker,
                        repository=self.repository, workspace=self.workspace, boundary=self.boundary,
                        parent_lease=self._lease, expected_policy_sha256=self.expected_policy_sha256,
                        deadline_monotonic=self.deadline_monotonic,
                        representation=self.representation, cancel_event=self._signal)
                except RepositoryOwnerConstructionError as error:
                    _need(error.owner is self._broker, "repository_constructor_owner_changed")
                    raise
                self._broker_constructed = True
                stage = "repository_owner_start_failed"
                while not self._broker._ready.is_set():
                    self._admission_checkpoint()
                    if self._broker._done.is_set() or not self._broker._thread.is_alive():
                        raise ValueError("repository_owner_stopped_during_start")
                    self._broker._ready.wait(min(READY_POLL_SECONDS,
                        max(0.0, self._admission_deadline - time.monotonic())))
                # The existing broker sets _ready in its failure-finally path
                # too; the event alone is not successful native startup.
                self._admission_checkpoint()
                self._ready = True
                self._live_broker()
                self._state = "active"
                return self
            except BaseException as error:
                self._ready, self._state = False, "start_failed"
                self._stop.set()
                self._error = stage
                if self._lease is not None:
                    self._admission_uncertain = False
                raise RepositorySourceOwnerLifecycleError(self, stage) from error

    @property
    def extra_argv(self):
        """Return public descriptor flags only while the existing broker is live."""
        with self._lock:
            try:
                self._live_broker()
                return list(self._broker.extra_argv)
            except BaseException as error:
                if isinstance(error, RepositorySourceOwnerLifecycleError):
                    raise
                raise RepositorySourceOwnerLifecycleError(self, "repository_owner_not_live") from error

    def note_spawned(self, process):
        """Authenticate the caller's existing live child; acquire no execution authority."""
        with self._lock:
            try:
                self._live_broker()
                self._broker.note_spawned(process)
                self._spawn_registered = True
            except BaseException as error:
                self._stop.set()
                self._error = "worker_registration_failed"
                raise RepositorySourceOwnerLifecycleError(self, "worker_registration_failed") from error

    def _receipt(self, *, observation_pending=False):
        lease = self._lease
        unknown_admission = (self._admission_uncertain or self._admission_inflight
            or observation_pending and self._start_attempted and not self._cleanup_complete)
        acquired = None if unknown_admission else lease is not None
        released = None if lease is None or observation_pending else lease.released
        broker_complete = None if self._broker_audit is None else self._broker_audit.get("cleanup_complete")
        # ResourceLease.to_dict includes a cancellation query that can itself
        # fail during scheduler recovery. Stable public identity is enough here.
        identity = None if lease is None else {
            "lease_id": lease.lease_id, "owner_pid": lease.owner_pid,
            "parent_lease_id": lease.parent_lease_id,
            **{name: getattr(lease, name) for name in DEMAND},
        }
        return dict(schema=SCHEMA, state="cleanup_pending" if observation_pending else self._state,
            ready=self._ready and not self._stop.is_set() and not observation_pending,
            started=self._start_attempted, worker_spawn_registered=self._spawn_registered,
            cleanup_complete=self._cleanup_complete and not observation_pending,
            cleanup_pending=not self._cleanup_complete or observation_pending,
            native_parent_acquired=acquired, native_parent_released=released,
            native_parent=identity, admission_uncertain=unknown_admission,
            observation_pending=observation_pending,
            admission_request_id=self._request_id,
            broker_cleanup_complete=broker_complete, broker_audit=copy.deepcopy(self._broker_audit),
            renewal_unproven=self._renewal_unproven,
            retained_custody=(lease is not None and not lease.released) or unknown_admission
                or self._broker is not None and not self._cleanup_complete,
            error=self._error, cleanup_error=self._cleanup_error,
            borrowed_parent_present=self._caller_parent is not None,
            resource_scope="repository_source_service_only",
            worker_reaping_performed=False, worker_cleanup_verified=False,
            automatic_terminal_wiring=False, cross_uid_deployment_qualified=False,
            provider_calls=0, token_savings_measured=False,
            proof_authority=False, execution_authority=False, completion_authority=False)

    def close(self, *, timeout_seconds=3.0):
        """Retire the source service and release only after complete broker cleanup.

        A release exception retains the native handle. Native ``release`` stops
        its heartbeat before attempting persistence, so the receipt explicitly
        marks renewal unproven; it does not claim continued heartbeat coverage.
        A later explicit close can retry release after cleanup is established.
        """
        _number(timeout_seconds, minimum=0.0, maximum=3.0, reason="invalid_owner_close_timeout")
        _need(os.getpid() == self._owner_pid, "lifecycle_belongs_to_another_process")
        self._stop.set()
        started = time.monotonic()
        if not self._lock.acquire(timeout=timeout_seconds):
            # Another synchronous operation still owns startup/cleanup. The
            # stop signal is already visible; no release is attempted here.
            return self._receipt(observation_pending=True)
        try:
            self._ready = False
            if self._cleanup_complete:
                return self._receipt()
            self._state = "cleanup_pending"
            if self._broker is not None:
                try:
                    audit = self._broker.close(timeout_seconds=max(0.0,
                        timeout_seconds - (time.monotonic() - started)))
                    _need(type(audit) is dict and type(audit.get("cleanup_complete")) is bool,
                          "invalid_repository_owner_cleanup_receipt")
                    self._broker_audit = copy.deepcopy(audit)
                    child_released = audit.get("native_child_released") is True
                    no_child = (audit.get("native_child_acquired") is False
                        and audit.get("native_child_released") is None
                        and audit.get("native_child_acquisition_uncertain") is False
                        and getattr(self._broker, "_native", None) is None)
                    if (audit["cleanup_complete"] is not True
                            or audit.get("service_stopped") is not True
                            or not (child_released or no_child)):
                        self._cleanup_error = "repository_owner_cleanup_unproven"
                        return self._receipt()
                except BaseException:
                    self._cleanup_error = "repository_owner_cleanup_failed"
                    return self._receipt()
            if self._lease is None:
                if self._admission_uncertain:
                    self._cleanup_error = "native_admission_outcome_unknown"
                    return self._receipt()
            elif not self._lease.released:
                try:
                    self._lease.release()
                    _need(self._lease.released is True, "native_parent_release_not_acknowledged")
                except BaseException:
                    self._renewal_unproven = True
                    self._cleanup_error = "native_parent_release_failed"
                    return self._receipt()
            self._cleanup_complete, self._state = True, "closed"
            self._cleanup_error = None
            return self._receipt()
        finally:
            self._lock.release()
