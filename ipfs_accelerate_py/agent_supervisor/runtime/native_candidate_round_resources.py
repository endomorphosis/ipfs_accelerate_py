"""Retain one genuine native sublease through candidate-child retirement.

This host-owner handle reserves capacity below an existing native parent. It
does not create a scheduler, a root reservation, a task claim, or kernel limits.
Only live owner issuance is usable; detached metadata never restores capacity.
An uncertain birth, cleanup, or release keeps its owner reachable for explicit
recovery. There is deliberately no context-manager or garbage-collection release.
Pending retention applies to this live owner only: native dead-owner recovery
does not establish that an independently surviving external child has stopped.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import threading
import time
import uuid
import weakref

from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceLane, ResourceLease, ResourceSchedulerConfig,
)

from . import owner_worker_attempt as attempts

SCHEMA = "supervisor-native-candidate-round-resources@1"
RETIREMENT_SCHEMA = "owner-native-candidate-resource-retirement@1"
MAX_RECORD_BYTES = 16384
_SEAL = object()
_LOCK = threading.RLock()
_ISSUED = weakref.WeakKeyDictionary()
# Strong custody is intentional: an exception/forgotten caller is not cleanup.
_PENDING = {}
_ATTEMPTS = {}
_LEASE_FIELDS = ("lease_id", "lease_key", "lane", "cpu_slots", "memory_mb",
    "gpu_memory_mb", "unified_memory_mb", "child_process_slots", "requires_gpu",
    "parent_lease_id", "owner_pid", "acquired_at")
_ROW_FIELDS = tuple(name for name in _LEASE_FIELDS if name != "lease_key") + (
    "owner_birth_marker", "owner_boot_id", "request_id")
_SCHEDULER_METHODS = ("_locked_state", "_validate_state_configuration",
    "_recover_stale_locked", "_parent_authority", "_normalise_lane", "_can_grant", "_is_fair_turn",
    "_descendants", "_prune_released_locked", "_validate_recovery_state", "_lane_metrics")


def _need(value, message):
    if not value:
        raise ValueError(message)


def _wire(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")
    _need(len(raw) <= MAX_RECORD_BYTES, "bounded native candidate resource record required")
    return raw


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex(value):
    return type(value) is str and len(value) == 32 and all(c in "0123456789abcdef" for c in value)


def _positive(value, maximum, name):
    _need(type(value) is int and 0 < value <= maximum, "bounded exact " + name + " required")


def _function(value):
    return getattr(value, "__self__", None), getattr(value, "__func__", value)


def _scheduler_material(scheduler):
    _need(isinstance(scheduler, GlobalResourceScheduler)
        and type(scheduler.config) is ResourceSchedulerConfig,
        "original native scheduler and configuration required")
    path = scheduler.state_path
    _need(isinstance(path, Path) and path.is_absolute() and path.resolve(strict=True) == path
        and scheduler.lock_path == path.with_name(path.name + ".lock"),
        "original canonical native scheduler paths required")
    policy = _wire(ResourceSchedulerConfig.persisted_dict(scheduler.config))
    runtime = _wire({name: getattr(scheduler.config, name) for name in (
        "poll_interval_seconds", "lease_ttl_seconds", "auto_renew_leases")})
    references = (type(scheduler), scheduler.config, scheduler._thread_lock,
        scheduler.config.proof_resource_sampler, scheduler.config.resource_pressure_sampler,
        *(_function(getattr(scheduler, name)) for name in _SCHEDULER_METHODS),
        _function(scheduler.config.reservations), _function(scheduler.config.persisted_dict))
    return policy, runtime, (str(path), str(scheduler.lock_path)), references


def _lease_material(lease, scheduler):
    _need(type(lease) is ResourceLease and lease._scheduler is scheduler
        and _hex(lease.lease_id) and _hex(lease.lease_key),
        "original native lease with a nonempty exact capability required")
    value = {name: getattr(lease, name) for name in _LEASE_FIELDS}
    for name, maximum in (("cpu_slots", 1024), ("memory_mb", 2**24),
            ("child_process_slots", 1024), ("owner_pid", (1 << 31) - 1)):
        _positive(value[name], maximum, "native " + name)
    _need(type(value["requires_gpu"]) is bool and value["requires_gpu"] is False
        and type(value["gpu_memory_mb"]) is int and value["gpu_memory_mb"] == 0
        and type(value["unified_memory_mb"]) is int and value["unified_memory_mb"] == 0,
        "candidate capacity supports CPU-only native leases")
    _need(type(value["lane"]) is str and value["lane"]
        and (value["parent_lease_id"] is None or _hex(value["parent_lease_id"]))
        and type(value["acquired_at"]) in (int, float)
        and math.isfinite(value["acquired_at"]) and value["acquired_at"] > 0,
        "original native lease identity required")
    return _wire(value)


def _row_material(row):
    _need(type(row) is dict and all(name in row for name in _ROW_FIELDS),
        "complete original native resource row required")
    return _wire({name: row[name] for name in _ROW_FIELDS})


def _chain(rows, identifier):
    result, seen = [], set()
    for _ in range(64):
        _need(identifier not in seen and identifier in rows, "native resource ancestry is unavailable")
        seen.add(identifier)
        row = rows[identifier]
        _need(_hex(identifier) and row.get("lease_id") == identifier
            and row.get("cancelled") is False and not row.get("release_requested")
            and type(row.get("expires_at")) in (int, float)
            and math.isfinite(row["expires_at"]) and row["expires_at"] > time.time(),
            "native resource ancestry is cancelled, released or expired")
        _row_material(row)
        if result:
            child = result[-1]
            _need(all(type(row.get(name)) is int and type(child.get(name)) is int
                and 0 <= child[name] <= row[name] for name in
                ("cpu_slots", "memory_mb", "child_process_slots", "gpu_memory_mb", "unified_memory_mb")),
                "native child capacity exceeds its original ancestor")
        result.append(row)
        identifier = row["parent_lease_id"]
        if identifier is None:
            return result
    raise ValueError("native resource ancestry exceeds its bound")


def _native_rows(scheduler, leases, *, current):
    # Native storage authenticates capabilities here. Redacted lease IDs alone
    # are insufficient: the native legacy API also accepts ID-only delegation.
    with GlobalResourceScheduler._locked_state(scheduler, persist=False) as state:
        rows = state["leases"]
        for lease, material in leases:
            _need(_lease_material(lease, scheduler) == material, "original native lease was changed")
            expected = json.loads(material)
            row = rows.get(expected["lease_id"])
            _need(type(row) is dict and row.get("lease_key") == expected["lease_key"]
                and all(row.get(name) == expected[name] for name in _LEASE_FIELDS),
                "native resource capability or admitted capacity differs")
        identifier = json.loads(leases[-1][1])["lease_id"]
        if current:
            chain = _chain(rows, identifier)
        else:
            chain = []
        return tuple(_row_material(row) for row in chain)


def _events(attempt, *, retirement=False):
    _need(type(attempt) is attempts.OwnerWorkerAttempt and attempt._token is attempts._TOKEN
        and type(attempt._state) is attempts._State, "exact owner-issued candidate attempt required")
    common = attempts.OwnerWorkerAttempt._current(attempt, deadline=False)
    state = attempt._state
    _need(type(state.prepared) is attempts._Event and state.prepared.raw == attempt._common
        and set(common) == attempts.COMMON_FIELDS and common["event"] == "prepared",
        "original prepared candidate census required")
    events = [state.prepared]
    if state.spawned is not None:
        _need(type(state.spawned) is attempts._Event and type(state.process) is subprocess.Popen
            and state.process._child_created is True and type(state.child) is tuple
            and len(state.child) == 3 and all(type(item) is int for item in state.child)
            and state.child[0] == state.process.pid and state.child[1] == common["owner_pid"]
            and state.child[2] > 0 and state.spent is True,
            "original actual candidate birth required")
        expected = dict(common, event="spawned", prepared_sha256=_sha(state.prepared.raw),
            child_pid=state.child[0], child_start_ticks=state.child[2])
        _need(state.spawned.raw == attempts._wire(expected), "candidate spawned census was changed")
        events.append(state.spawned)
    if state.retired_event is not None:
        _need(type(state.retired_event) is attempts._Event and state.retired is True,
            "original retired candidate event required")
        value = json.loads(state.retired_event.raw)
        _need(set(value) == attempts.RETIRED_FIELDS
            and all(value[name] == item for name, item in common.items() if name != "event")
            and value["event"] == "retired" and value["prepared_sha256"] == _sha(state.prepared.raw)
            and value["spawned_sha256"] == (None if state.spawned is None else _sha(state.spawned.raw))
            and all(type(value[name]) is bool for name in ("waited", "cleanup_succeeded", "brokers_closed")),
            "candidate retirement census bindings differ")
        if state.spawned is not None:
            _need(value["child_pid"] == state.child[0] and value["child_start_ticks"] == state.child[2]
                and (value["child_returncode"] is None if not value["waited"] else
                    type(value["child_returncode"]) is int and value["child_returncode"] == state.process.returncode),
                "candidate retirement child outcome differs")
        events.append(state.retired_event)
    _need(not retirement or state.spawned is not None, "published candidate birth is required for retirement")
    descriptor = os.open(attempt._parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        _need(attempts._directory_identity(os.fstat(descriptor)) == attempt._parent_identity,
            "candidate census directory was replaced")
        for event in events:
            attempts.OwnerWorkerAttempt._check_event(attempt, descriptor, event)
    finally:
        os.close(descriptor)
    attempts.OwnerWorkerAttempt._current(attempt, deadline=False)
    return common, tuple(events)


@dataclass
class _State:
    references: tuple
    literals: tuple
    lock: object = field(default_factory=threading.RLock)
    admission: object = None
    admission_binding: bytes | None = None
    snapshot: object = None
    broker: object = None
    broker_references: tuple | None = None
    broker_literals: tuple | None = None
    spawn_attempted: bool = False
    process: subprocess.Popen | None = None
    child: tuple | None = None
    released: bool = False
    retirement_event: object = None
    retirement_mode: str | None = None
    retiring: bool = False
    retirement_reentry: bool = False
    publication_attempts: int = 0


@contextmanager
def _retirement(state):
    with state.lock:
        if state.retiring:
            state.retirement_reentry = True
            raise ValueError("native candidate retirement cannot reenter an owner callback")
        _need(not state.released, "candidate resource retirement already completed")
        state.retiring, state.retirement_reentry = True, False
        try:
            yield
            _need(not state.retirement_reentry, "recursive retirement invalidated owner cleanup")
        finally:
            state.retiring = False


class NativeCandidateRoundResources:
    __slots__ = ("__weakref__", "_seal", "_token", "_parent", "_native", "_scheduler",
        "_worker_attempt", "_repository_id", "_metadata", "_parent_material", "_native_material",
        "_ancestry", "_scheduler_snapshot", "_attempt_common", "_attempt_refs")

    def __init__(self, seal, *, parent, native, worker_attempt, repository_id, metadata,
            parent_material, native_material, ancestry, scheduler_snapshot):
        _need(seal is _SEAL and type(self) is NativeCandidateRoundResources,
            "native candidate resources require their live owner factory")
        for name, value in dict(_seal=seal, _token=object(), _parent=parent, _native=native,
                _scheduler=parent._scheduler, _worker_attempt=worker_attempt, _repository_id=repository_id,
                _metadata=_wire(metadata), _parent_material=parent_material, _native_material=native_material,
                _ancestry=ancestry, _scheduler_snapshot=scheduler_snapshot, _attempt_common=worker_attempt._common,
                _attempt_refs=(worker_attempt._state, worker_attempt._state.prepared,
                    worker_attempt._boundary, worker_attempt._workspace, worker_attempt._parent,
                    worker_attempt._workspace_identity, worker_attempt._parent_identity)).items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("native candidate resource handle is immutable")

    def _references(self):
        return (self._token, self._parent, self._native, self._scheduler, self._worker_attempt,
            *self._attempt_refs[:5])

    def _literals(self):
        return (self._repository_id, self._metadata, self._parent_material, self._native_material,
            self._ancestry, self._scheduler_snapshot, self._attempt_common, self._attempt_refs[5:])

    def _issued(self):
        _need(type(self) is NativeCandidateRoundResources and self._seal is _SEAL,
            "exact issued native candidate resources required")
        with _LOCK:
            state = _ISSUED.get(self)
            _need(type(state) is _State and all(a is b for a, b in zip(
                NativeCandidateRoundResources._references(self), state.references))
                and NativeCandidateRoundResources._literals(self) == state.literals,
                "native candidate resources lost original owner custody")
            return state

    def _retained(self):
        state = NativeCandidateRoundResources._issued(self)
        snapshot = _scheduler_material(self._scheduler)
        original = self._scheduler_snapshot
        _need(snapshot[:3] == original[:3]
            and all(a is b if not isinstance(a, tuple) else a == b for a, b in zip(snapshot[3], original[3]))
            and _lease_material(self._parent, self._scheduler) == self._parent_material
            and _lease_material(self._native, self._scheduler) == self._native_material
            and self._worker_attempt._common == self._attempt_common
            and all(a is b for a, b in zip((self._worker_attempt._state,
                self._worker_attempt._state.prepared, self._worker_attempt._boundary,
                self._worker_attempt._workspace, self._worker_attempt._parent), self._attempt_refs[:5]))
            and (self._worker_attempt._workspace_identity, self._worker_attempt._parent_identity) == self._attempt_refs[5:],
            "native resource scheduler, capability, policy or attempt changed")
        if state.admission is not None:
            from .native_candidate_round_admission import NativeCandidateRoundAdmission
            gate = state.admission
            NativeCandidateRoundAdmission._issued(gate)
            _need(gate._resources is self and gate._worker_attempt is self._worker_attempt
                and gate._snapshot is state.snapshot and gate._binding == state.admission_binding
                and gate._allocation._server.identity.repository_id == self._repository_id
                and gate._snapshot._source_broker.workspace == self._worker_attempt._workspace,
                "candidate resources differ from their original native admission")
        if state.broker is not None:
            broker = state.broker
            _need(all(a is b for a, b in zip((broker._snapshot, broker._thread, broker._stop,
                    broker._registered), state.broker_references))
                and (str(broker.descriptor_path), str(broker.socket_path), broker.descriptor_sha256,
                    _wire(broker.descriptor)) == state.broker_literals,
                "candidate resource broker custody changed")
        return state

    @property
    def metadata_bytes(self):
        NativeCandidateRoundResources._issued(self)
        return self._metadata

    @property
    def native_lease(self):
        NativeCandidateRoundResources._issued(self)
        return self._native

    @property
    def worker_attempt(self):
        NativeCandidateRoundResources._issued(self)
        return self._worker_attempt

    @property
    def repository_id(self):
        NativeCandidateRoundResources._issued(self)
        return self._repository_id

    @property
    def workspace(self):
        NativeCandidateRoundResources._issued(self)
        return self._worker_attempt._workspace

    @property
    def absolute_deadline_ns(self):
        NativeCandidateRoundResources._issued(self)
        return json.loads(self._attempt_common)["absolute_deadline_ns"]

    def _observation(self):
        state = NativeCandidateRoundResources._issued(self)
        value = json.loads(self._metadata)
        value.update(admission_bound=state.admission is not None, broker_bound=state.broker is not None,
            spawn_attempted=state.spawn_attempted, spawned=state.process is not None,
            released=state.released, retained_pending_cleanup=not state.released,
            retention_scope="live_owner_custody_not_proof_of_current_native_charge",
            retirement_mode=state.retirement_mode,
            retirement_event_sha256=None if state.retirement_event is None else _sha(state.retirement_event.raw))
        return value

    def remaining(self):
        state = NativeCandidateRoundResources._retained(self)
        _need(not state.released, "candidate resource reservation already released")
        left = (json.loads(self._attempt_common)["absolute_deadline_ns"] - time.monotonic_ns()) / 1_000_000_000
        if left <= 0:
            raise TimeoutError("original candidate resource deadline exhausted")
        return left

    def revalidate(self):
        state = NativeCandidateRoundResources._retained(self)
        with state.lock:
            NativeCandidateRoundResources.remaining(self)
            _need(not self._parent.released and not self._native.released,
                "native candidate resource reservation was released")
            chain = _native_rows(self._scheduler, ((self._parent, self._parent_material),
                (self._native, self._native_material)), current=True)
            _need(chain == self._ancestry and not GlobalResourceScheduler.is_cancelled(
                self._scheduler, self._native.lease_id, missing_is_cancelled=True),
                "native candidate resource ancestry is no longer current")
            NativeCandidateRoundResources._retained(self)
            return NativeCandidateRoundResources._observation(self)

    def poll(self):
        """Lightweight resource-only refusal polling; no candidate source replay."""
        try:
            NativeCandidateRoundResources.revalidate(self)
            return False
        except Exception:
            return True

    def thread_environment(self):
        NativeCandidateRoundResources._issued(self)
        count = str(json.loads(self._metadata)["budget"]["threads_per_process"])
        return {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": count, "OPENBLAS_NUM_THREADS": count,
            "MKL_NUM_THREADS": count, "NUMEXPR_NUM_THREADS": count, "VECLIB_MAXIMUM_THREADS": count,
            "RAYON_NUM_THREADS": count, "TOKENIZERS_PARALLELISM": "false"}

    def bind_admission(self, gate):
        from .native_candidate_round_admission import NativeCandidateRoundAdmission
        state = NativeCandidateRoundResources._retained(self)
        with state.lock:
            _need(state.admission is None and not state.spawn_attempted and not state.released
                and type(gate) is NativeCandidateRoundAdmission and gate._resources is self,
                "one original native admission must bind candidate resources")
            NativeCandidateRoundAdmission._issued(gate)
            _need(gate._worker_attempt is self._worker_attempt
                and gate._allocation._server.identity.repository_id == self._repository_id
                and gate._snapshot._source_broker.workspace == self._worker_attempt._workspace
                and not gate._state.consumed and not gate._state.spawn_started and gate._state.process is None,
                "candidate resource admission does not match its fresh owner attempt")
            state.admission, state.admission_binding, state.snapshot = gate, gate._binding, gate._snapshot
            NativeCandidateRoundResources.revalidate(self)

    def bind_broker(self, broker):
        from .worker_candidate_round_broker import WorkerCandidateRoundBroker
        from .candidate_dispatch_broker import CandidateDispatchBroker
        state = NativeCandidateRoundResources._retained(self)
        with state.lock:
            _need(state.admission is not None and state.broker is None and not state.spawn_attempted
                and not state.released and type(broker) in (WorkerCandidateRoundBroker, CandidateDispatchBroker)
                and broker._snapshot is state.snapshot and broker.workspace == self._worker_attempt._workspace
                and broker._process is None and not broker._registered.is_set() and not broker._closed,
                "one original unused candidate broker required")
            if type(broker) is CandidateDispatchBroker:
                _need(broker._admission is state.admission, "candidate dispatch belongs to another admission")
            state.broker = broker
            state.broker_references = (broker._snapshot, broker._thread, broker._stop, broker._registered)
            state.broker_literals = (str(broker.descriptor_path), str(broker.socket_path),
                broker.descriptor_sha256, _wire(broker.descriptor))
            NativeCandidateRoundResources.revalidate(self)

    def note_spawn_attempt(self):
        state = NativeCandidateRoundResources._retained(self)
        with state.lock:
            _need(state.admission is not None and state.broker is not None and not state.spawn_attempted
                and state.process is None and state.admission._state.executing is True
                and state.admission._state.consumed is True,
                "candidate native gate must enter one actual spawn attempt")
            NativeCandidateRoundResources.revalidate(self)
            state.spawn_attempted = True

    def note_spawned(self, process):
        state = NativeCandidateRoundResources._issued(self)
        with state.lock:
            _need(state.spawn_attempted and state.process is None and type(process) is subprocess.Popen
                and process._child_created is True, "one actual resource-bound Popen child required")
            # Retain recognition before any kernel/census observation can fail.
            state.process = process
            NativeCandidateRoundResources._retained(self)
            common, _ = _events(self._worker_attempt, retirement=True)
            child = self._worker_attempt._state.child
            _need(self._worker_attempt._state.process is process
                and state.admission._state.process is process and child[0] == process.pid
                and child[1] == common["owner_pid"] == os.getpid(),
                "resource child differs from the actual admitted owner birth")
            state.child = child
            NativeCandidateRoundResources.revalidate(self)

    def _closed_broker(self):
        state = NativeCandidateRoundResources._retained(self)
        broker = state.broker
        _need(broker is not None and broker._process is state.process and broker._closed
            and broker._stop.is_set() and not broker._thread.is_alive()
            and all(not path.exists() and not path.is_symlink() for path in
                (broker.descriptor_path, broker.socket_path)),
            "candidate resource broker has not actually retired")

    def _quiescent(self, *, recovered):
        state = NativeCandidateRoundResources._retained(self)
        _need(state.spawn_attempted and type(state.process) is subprocess.Popen
            and state.process._child_created is True and state.child is not None
            and state.process.pid == state.child[0], "known original resource child required for retirement")
        common, events = _events(self._worker_attempt, retirement=True)
        attempt_state = self._worker_attempt._state
        _need(attempt_state.process is state.process and attempt_state.child == state.child
            and state.admission._state.process is state.process
            and type(subprocess.Popen.poll(state.process)) is int,
            "actual resource child must be reaped before release")
        if not recovered:
            _need(type(attempt_state.retired_event) is attempts._Event,
                "published original candidate retirement required")
            retired = json.loads(attempt_state.retired_event.raw)
            _need(retired["cleanup_succeeded"] is True and retired["brokers_closed"] is True,
                "original candidate cleanup/retirement is not complete")
        NativeCandidateRoundResources._closed_broker(self)
        return common, events

    def _publish_retirement(self, *, mode, events):
        state = NativeCandidateRoundResources._retained(self)
        _need(not state.retirement_reentry, "recursive retirement invalidated owner cleanup")
        attempt = self._worker_attempt
        if state.retirement_event is not None:
            descriptor = os.open(attempt._parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                _need(attempts._directory_identity(os.fstat(descriptor)) == attempt._parent_identity,
                    "resource retirement directory changed")
                attempts.OwnerWorkerAttempt._check_event(attempt, descriptor, state.retirement_event)
            finally:
                os.close(descriptor)
            return
        original_retired = attempt._state.retired_event
        _need(state.publication_attempts < 16, "resource retirement publication attempts exhausted")
        state.publication_attempts += 1
        publication_id = uuid.uuid4().hex
        value = dict(schema=RETIREMENT_SCHEMA, resource_id=json.loads(self._metadata)["resource_id"],
            publication_id=publication_id, publication_attempt=state.publication_attempts,
            worker_attempt_id=json.loads(self._attempt_common)["attempt_id"],
            native_lease_id=self._native.lease_id, parent_lease_id=self._parent.lease_id,
            resource_metadata_sha256=_sha(self._metadata), retirement_mode=mode,
            prepared_event_sha256=_sha(attempt._state.prepared.raw),
            spawned_event_sha256=_sha(attempt._state.spawned.raw),
            retired_event_sha256=None if original_retired is None else _sha(original_retired.raw),
            original_census_flags_unchanged=True, child_pid=state.child[0], child_start_ticks=state.child[2],
            child_returncode=state.process.returncode, child_reaped=True, descendant_cleanup_observed=True,
            broker_retirement_verified=True, observation_only=True, execution_authority=False,
            proof_authority=False, completion_authority=False, retry_authority=False)
        raw = _wire(value)
        # Preserve any uncertain prior publication. A recovery publishes a new
        # exclusive observation instead of adopting/overwriting partial bytes.
        name = "worker-resource-" + value["resource_id"] + "-retired-" + publication_id + ".json"
        attempts.OwnerWorkerAttempt._current(attempt, deadline=False)
        descriptor = os.open(attempt._parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            _need(attempts._directory_identity(os.fstat(descriptor)) == attempt._parent_identity,
                "resource retirement directory was replaced")
            for event in events:
                attempts.OwnerWorkerAttempt._check_event(attempt, descriptor, event)
            output = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o400, dir_fd=descriptor)
            try:
                remaining = memoryview(raw)
                while remaining:
                    written = os.write(output, remaining)
                    _need(written > 0, "resource retirement publication made no progress")
                    remaining = remaining[written:]
                os.fchmod(output, 0o400)
                os.fsync(output)
                info = os.fstat(output)
                _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o400
                    and info.st_nlink == 1 and info.st_uid == os.getuid() and info.st_gid == os.getgid()
                    and info.st_size == len(raw), "immutable resource retirement publication required")
                event = attempts._Event(name, raw, attempts._file_identity(info))
            finally:
                os.close(output)
            os.fsync(descriptor)
            for previous in (*events, event):
                attempts.OwnerWorkerAttempt._check_event(attempt, descriptor, previous)
            attempts.OwnerWorkerAttempt._current(attempt, deadline=False)
            _need(attempts._directory_identity(os.fstat(descriptor)) == attempt._parent_identity,
                "resource retirement directory changed during publication")
            state.retirement_event, state.retirement_mode = event, mode
        finally:
            os.close(descriptor)

    def _release(self):
        state = NativeCandidateRoundResources._retained(self)
        _need(not state.retirement_reentry, "recursive retirement cannot release candidate capacity")
        identifier = json.loads(self._native_material)["lease_id"]
        with GlobalResourceScheduler._locked_state(self._scheduler, persist=False) as storage:
            rows = storage["leases"]
            row = rows.get(identifier)
            if row is not None:
                expected = json.loads(self._native_material)
                _need(all(row.get(name) == expected[name] for name in _LEASE_FIELDS),
                    "original resource release capability changed")
                _need(not any(value.get("parent_lease_id") == identifier for value in rows.values()),
                    "live native consumers retain candidate resource capacity")
        if row is not None:
            # A previous acknowledged-but-deferred release may have set the
            # local flag. Retry through the original keyed native owner only.
            if self._native.released:
                GlobalResourceScheduler.release(self._scheduler, identifier, self._native.lease_key)
            else:
                ResourceLease.release(self._native)
        with GlobalResourceScheduler._locked_state(self._scheduler, persist=False) as storage:
            _need(identifier not in storage["leases"], "resource release remains deferred or uncertain")
        state.released = True
        with _LOCK:
            _PENDING.pop(json.loads(self._metadata)["resource_id"], None)
        receipt = NativeCandidateRoundResources._observation(self)
        # Issuance values must not retain their weak keys through a completed
        # gate/broker -> resource cycle. Pending scopes keep these references;
        # only positively retired scopes detach them after their final receipt.
        state.admission = state.broker = None
        return receipt

    def finish_retirement(self):
        """Release after original census and actual broker/child quiescence.

        Cancellation or expiry cannot prevent safe retirement. Uncertain
        publication/release leaves this handle pending; no census is rewritten.
        """
        state = NativeCandidateRoundResources._retained(self)
        with _retirement(state):
            _common, events = NativeCandidateRoundResources._quiescent(self, recovered=False)
            NativeCandidateRoundResources._publish_retirement(self, mode="census-confirmed", events=events)
            NativeCandidateRoundResources._quiescent(self, recovered=False)
            return NativeCandidateRoundResources._release(self)

    def recover_retirement(self, *, cleanup):
        """Explicitly reap a known child and retain the old census unchanged."""
        state = NativeCandidateRoundResources._retained(self)
        with _retirement(state):
            _need(callable(cleanup) and state.process is not None
                and state.child is not None and state.process.pid == state.child[0],
                "known original child and owner cleanup operation required")
            process = state.process
            if subprocess.Popen.poll(process) is None:
                try:
                    subprocess.Popen.terminate(process)
                except ProcessLookupError:
                    pass
                try:
                    subprocess.Popen.wait(process, timeout=5)
                except subprocess.TimeoutExpired:
                    subprocess.Popen.kill(process)
                    subprocess.Popen.wait(process, timeout=5)
            _need(type(subprocess.Popen.poll(process)) is int, "actual child cleanup did not reap")
            result = cleanup(process)
            _need(type(result) is bool and result is True, "owner descendant cleanup did not confirm quiescence")
            _need(not state.retirement_reentry, "recursive retirement invalidated owner cleanup")
            _need(state.broker is not None, "original candidate broker required for recovery")
            type(state.broker).close(state.broker)
            _common, events = NativeCandidateRoundResources._quiescent(self, recovered=True)
            NativeCandidateRoundResources._publish_retirement(self, mode="owner-recovered", events=events)
            NativeCandidateRoundResources._quiescent(self, recovered=True)
            return NativeCandidateRoundResources._release(self)

    def abort_unstarted(self):
        """Release a scope proven never to have attempted a child launch."""
        state = NativeCandidateRoundResources._retained(self)
        gate_lock = nullcontext() if state.admission is None else state.admission._state.lock
        # Match the admission's gate -> resource lock order, including an abort
        # racing a launch. Holding the resource lock while waiting on the gate
        # would otherwise deadlock the gate's final resource checks.
        with gate_lock, _retirement(state):
            _need(not state.spawn_attempted and state.process is None
                and self._worker_attempt._state.process is None and self._worker_attempt._state.spent is False,
                "attempted or uncertain candidate birth cannot be aborted as unstarted")
            if state.admission is not None:
                gate = state.admission
                _need(not gate._state.executing and not gate._state.spawn_started
                    and gate._state.process is None, "entered candidate spawn cannot be aborted")
                gate._state.consumed = True
            if state.broker is not None:
                _need(state.broker._process is None, "candidate broker already owns a child")
                type(state.broker).close(state.broker)
                _need(state.broker._closed and state.broker._stop.is_set()
                    and not state.broker._thread.is_alive()
                    and all(not path.exists() and not path.is_symlink() for path in
                        (state.broker.descriptor_path, state.broker.socket_path)),
                    "unstarted candidate broker did not retire")
            state.retirement_mode = "unstarted-abort"
            return NativeCandidateRoundResources._release(self)


def reserve_native_candidate_round_resources(*, parent_lease, worker_attempt, repository_id,
        cpu_slots, memory_mb, child_process_slots, threads_per_process=1):
    """Acquire one bounded child under the actual existing native parent."""
    _need(type(parent_lease) is ResourceLease and isinstance(parent_lease._scheduler, GlobalResourceScheduler),
        "actual existing native parent lease required")
    _need(type(repository_id) is str and 0 < len(repository_id.encode()) <= 256
        and repository_id == repository_id.strip() and not any(c in repository_id for c in "\0\r\n"),
        "bounded original repository identity required")
    for name, value, maximum in (("cpu_slots", cpu_slots, 1024), ("memory_mb", memory_mb, 2**24),
            ("child_process_slots", child_process_slots, 1024), ("threads_per_process", threads_per_process, 1024)):
        _positive(value, maximum, name)
    _need(cpu_slots >= threads_per_process, "CPU admission must cover one active child's declared threads")
    common, events = _events(worker_attempt)
    _need(worker_attempt._state.spent is False and worker_attempt._state.retired is False
        and worker_attempt._state.process is None and worker_attempt._state.spawned is None
        and worker_attempt._state.retired_event is None, "fresh unstarted owner worker attempt required")
    attempts._remaining(common["absolute_deadline_ns"])
    scheduler = parent_lease._scheduler
    snapshot = _scheduler_material(scheduler)
    parent_material = _lease_material(parent_lease, scheduler)
    parent = json.loads(parent_material)
    _need(not parent_lease.released and cpu_slots <= parent["cpu_slots"]
        and memory_mb <= parent["memory_mb"] and child_process_slots <= parent["child_process_slots"],
        "candidate resource demand exceeds its actual native parent")
    _native_rows(scheduler, ((parent_lease, parent_material),), current=True)
    _need(not GlobalResourceScheduler.is_cancelled(scheduler, parent_lease.lease_id, missing_is_cancelled=True)
        and GlobalResourceScheduler.renew(scheduler, parent_lease.lease_id, parent_lease.lease_key),
        "original native parent capability is unavailable")
    token = object()
    attempt_id = common["attempt_id"]
    with _LOCK:
        prior = _ATTEMPTS.get(attempt_id)
        _need(prior is None or isinstance(prior, weakref.ReferenceType) and prior() is None,
            "candidate owner attempt already reserved resource capacity")
        _ATTEMPTS[attempt_id] = token
    native = None
    result = None
    try:
        remaining = (common["absolute_deadline_ns"] - time.monotonic_ns()) / 1_000_000_000
        if remaining <= 0:
            raise TimeoutError("original candidate deadline exhausted before resource admission")
        resource_id = uuid.uuid4().hex
        native = GlobalResourceScheduler.acquire(scheduler, lane=ResourceLane.ORCHESTRATION,
            cpu_slots=cpu_slots, memory_mb=memory_mb, child_process_slots=child_process_slots,
            parent_lease=parent_lease, timeout=remaining, cancel_event=parent_lease.cancellation_signal,
            request_id="native-candidate-round:" + resource_id)
        native_material = _lease_material(native, scheduler)
        ancestry = _native_rows(scheduler, ((parent_lease, parent_material), (native, native_material)), current=True)
        _need(json.loads(native_material)["parent_lease_id"] == parent_lease.lease_id
            and native.owner_pid == os.getpid() and native.lane == ResourceLane.ORCHESTRATION.value,
            "actual candidate child must remain below its original native parent")
        root = json.loads(ancestry[-1])["lease_id"]
        metadata = dict(schema=SCHEMA, resource_id=resource_id, repository_id=repository_id,
            worker_attempt_id=common["attempt_id"], worker_prepared_event_sha256=_sha(events[0].raw),
            workspace_identity_sha256=common["workspace_identity_sha256"],
            boundary_manifest_sha256=common["boundary_manifest_sha256"], native_prompt_sha256=common["native_prompt_sha256"],
            absolute_deadline_ns=common["absolute_deadline_ns"], native_lease_id=native.lease_id,
            parent_lease_id=parent_lease.lease_id, root_lease_id=root,
            scheduler_state_path_sha256=_sha(str(scheduler.state_path).encode()),
            scheduler_policy_sha256=_sha(snapshot[0]), ancestry_sha256=_sha(_wire([json.loads(row) for row in ancestry])),
            budget=dict(cpu_slots=cpu_slots, memory_mb=memory_mb, child_process_slots=child_process_slots,
                threads_per_process=threads_per_process), quantitative_resource_reservation_verified=True,
            resource_capacity_only=True, native_sublease_only=True, hard_enforcement=False,
            execution_authority=False, proof_authority=False, completion_authority=False, retry_authority=False,
            production_container_dispatch_qualified=False)
        result = NativeCandidateRoundResources(_SEAL, parent=parent_lease, native=native,
            worker_attempt=worker_attempt, repository_id=repository_id, metadata=metadata,
            parent_material=parent_material, native_material=native_material,
            ancestry=ancestry, scheduler_snapshot=snapshot)
        with _LOCK:
            _ISSUED[result] = _State(NativeCandidateRoundResources._references(result),
                NativeCandidateRoundResources._literals(result))
            _PENDING[resource_id] = result
            def discard(reference):
                with _LOCK:
                    if _ATTEMPTS.get(attempt_id) is reference:
                        _ATTEMPTS.pop(attempt_id, None)
            _ATTEMPTS[attempt_id] = weakref.ref(result, discard)
        NativeCandidateRoundResources.revalidate(result)
        return result
    except BaseException as error:
        if result is not None:
            # Factory-local issuance has never returned or bound a gate. Its
            # unstarted abort is safe even when final currentness was lost.
            # A failed abort retains custody and gives only the trusted owner
            # the live typed handle; preserve the original refusal unchanged.
            try:
                NativeCandidateRoundResources.abort_unstarted(result)
            except BaseException:
                try:
                    error.candidate_resources = result
                except BaseException:
                    pass
        elif native is not None:
            # No child launch was admitted. This exact native acquisition is
            # safe to release; failed release retains its original lease handle.
            try:
                ResourceLease.release(native)
            except BaseException:
                with _LOCK:
                    _PENDING["unissued:" + native.lease_id] = native
        if result is None:
            with _LOCK:
                if _ATTEMPTS.get(attempt_id) is token:
                    _ATTEMPTS.pop(attempt_id, None)
        raise


__all__ = ["SCHEMA", "RETIREMENT_SCHEMA", "NativeCandidateRoundResources",
    "reserve_native_candidate_round_resources"]
