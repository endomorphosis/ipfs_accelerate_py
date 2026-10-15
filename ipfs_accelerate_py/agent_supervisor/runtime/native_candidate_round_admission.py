"""One host-owner candidate birth under existing native claim and run leases.

Only an issued live handle can invoke the trusted owner's spawn callback. Its
detached observations grant nothing. This joins existing advisory candidate
preparation and child census; it installs no RPC, container route, provider, or
quantitative resource reservation. The caller retains all cleanup duties and
must withhold provider input until the complete native fence has returned.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from types import MappingProxyType
import uuid
import weakref

from ..merge.database_coordination import (
    DatabaseCoordinator, ProcessSerializedDatabaseCoordinator, FencedLease,
    LeaseKind, LeaseMode, LeaseState,
)
from .current_source_task_admission import _owned_file
from .current_source_worktree_allocation import NativeSourceWorktreeAllocation
from .owner_candidate_round_context import OwnerCandidateRoundContext
from .owner_candidate_round_prompt import OwnerCandidateRoundPrompt
from . import owner_worker_attempt as attempts
from .worker_source_broker import WorkerSourceBroker

SCHEMA = "native-candidate-round-admission@1"
MINIMUM_REMAINING_MS = 1000
_SEAL = object()
_REGISTRY_LOCK = threading.RLock()
_PRIOR_ADMISSIONS = weakref.WeakKeyDictionary()
_CAPTURING = weakref.WeakKeyDictionary()
_ISSUED = weakref.WeakKeyDictionary()


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _wire(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    _need(len(raw) <= 262144, "candidate admission binding exceeds its byte bound")
    return raw


def _lease_material(lease):
    _need(type(lease) is FencedLease and type(lease.lease_kind) is LeaseKind
        and type(lease.mode) is LeaseMode and type(lease.state) is LeaseState,
        "exact native run lease required")
    value = {}
    for item in fields(FencedLease):
        raw = getattr(lease, item.name)
        if item.name in {"lease_kind", "mode", "state"}:
            value[item.name] = raw.value
        elif item.name == "body":
            _need(type(raw) is MappingProxyType, "original native lease body required")
            value[item.name] = dict(raw)
        elif item.name in {"fencing_token", "fence_epoch", "acquired_at_ms", "expires_at_ms",
                "revision", "attempt_number"}:
            _need(type(raw) is int and 0 <= raw < (1 << 63), "exact bounded native lease integer required")
            value[item.name] = raw
        else:
            _need(type(raw) is str and len(raw.encode("utf-8")) <= 4096
                and not any(c in raw for c in "\0\r\n"), "bounded native lease text required")
            value[item.name] = raw
    _need(lease.lease_kind is LeaseKind.RESOURCE and lease.resource_kind == "supervisor_run"
        and lease.state is LeaseState.ACCEPTED
        and all(value[name] for name in ("lease_id", "scope_key", "scope", "owner_session_id",
            "resource_id", "repository_id"))
        and lease.fencing_token > 0 and lease.fence_epoch > 0 and lease.revision > 0
        and lease.expires_at_ms > lease.acquired_at_ms,
        "live supervisor-run resource lease required")
    _wire(value)
    return value


def _lease_identity(value):
    return {name: item for name, item in value.items() if name not in {"expires_at_ms", "revision"}}


def _coordinator_refs(coordinator):
    _need(type(coordinator) in (DatabaseCoordinator, ProcessSerializedDatabaseCoordinator),
        "exact native run coordinator required")
    if type(coordinator) is DatabaseCoordinator:
        _need(coordinator._connection is not None and not coordinator._closed
            and coordinator._quack_transport is False,
            "open local run coordinator required")
        _need(getattr(coordinator._connection, "in_transaction", None) is False,
            "candidate admission cannot borrow a run transaction")
        references = (coordinator._connection, coordinator._lock, coordinator._clock_ms)
        configuration = (coordinator._path, coordinator._open_target,
            coordinator._default_lease_ms, coordinator._require_existing_authority)
    else:
        _need(not coordinator._closed, "open serialized run coordinator required")
        references = (coordinator._state_lock, coordinator._clock_ms)
        configuration = (coordinator._path, coordinator._serialization_lock_path,
            coordinator._default_lease_ms, coordinator._lock_timeout_seconds,
            coordinator._require_existing_authority)
    return references, configuration, _owned_file(coordinator.database_path)


def _file_pin(identity):
    # The original helper includes path and its private parent's inode/mode.
    return _wire([_sha(str(identity[0]).encode("utf-8")), *identity[1:]])


def _expiry(lease, *, minimum_ms):
    _need(type(lease) is FencedLease and type(lease.expires_at_ms) is int,
        "actual protected native lease required")
    if lease.expires_at_ms <= time.time_ns() // 1_000_000 + minimum_ms:
        raise TimeoutError("native lease lacks remaining candidate birth budget")


def _attempt(attempt, *, phase, process=None):
    """Read only original retained event witnesses; never adopt a named file."""
    _need(type(attempt) is attempts.OwnerWorkerAttempt and attempt._token is attempts._TOKEN
        and type(attempt._state) is attempts._State, "exact owner-issued worker attempt required")
    common = attempts.OwnerWorkerAttempt._current(attempt, deadline=False)
    state = attempt._state
    _need(type(state.prepared) is attempts._Event and state.prepared.raw == attempt._common
        and set(common) == attempts.COMMON_FIELDS and common["event"] == "prepared",
        "original prepared worker event required")
    events = [state.prepared]
    if phase == "prepared":
        _need(state.spent is False and state.retired is False and state.retired_event is None
            and state.process is None and state.child is None and state.spawned is None,
            "fresh unstarted candidate worker attempt required")
    else:
        _need(state.spent is True and type(state.process) is subprocess.Popen
            and state.process._child_created is True and type(state.spawned) is attempts._Event
            and type(state.child) is tuple and len(state.child) == 3
            and all(type(item) is int for item in state.child)
            and state.child[0] == state.process.pid and state.child[1] == common["owner_pid"]
            and state.child[2] > 0,
            "original actual worker birth required")
        expected = dict(common, event="spawned", prepared_sha256=_sha(state.prepared.raw),
            child_pid=state.child[0], child_start_ticks=state.child[2])
        _need(state.spawned.raw == _wire(expected), "original spawned worker event differs")
        events.append(state.spawned)
        if phase == "retired":
            _need(state.retired is True and type(state.retired_event) is attempts._Event
                and type(state.process.returncode) is int and 1 <= state.process.returncode <= 255
                and subprocess.Popen.poll(state.process) == state.process.returncode,
                "successfully retired failed worker required")
            expected = dict(common, event="retired", prepared_sha256=_sha(state.prepared.raw),
                spawned_sha256=_sha(state.spawned.raw), child_pid=state.child[0],
                child_start_ticks=state.child[2], waited=True, child_returncode=state.process.returncode,
                cleanup_succeeded=True, brokers_closed=True)
            _need(state.retired_event.raw == _wire(expected),
                "truthful waited cleanup and broker retirement required")
            events.append(state.retired_event)
        else:
            _need(phase == "spawned" and state.process is process and state.retired is False
                and state.retired_event is None, "candidate birth census does not match actual child")
    descriptor = os.open(attempt._parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        _need(attempts._directory_identity(os.fstat(descriptor)) == attempt._parent_identity,
            "worker event directory changed")
        for event in events:
            attempts.OwnerWorkerAttempt._check_event(attempt, descriptor, event)
    finally:
        os.close(descriptor)
    attempts.OwnerWorkerAttempt._current(attempt, deadline=False)
    return common, events


@dataclass
class _State:
    lock: object = field(default_factory=threading.RLock)
    consumed: bool = False
    executing: bool = False
    spawn_started: bool = False
    spawn_completed: bool = False
    process: subprocess.Popen | None = None
    child: tuple | None = None


class NativeCandidateRoundAdmission:
    """Exact issued one-use host handle; no deserializer or dispatch route."""
    __slots__ = ("__weakref__", "_seal", "_token", "_allocation", "_run_coordinator", "_run_lease",
        "_candidate_prompt", "_prior_worker_attempt", "_worker_attempt", "_snapshot", "_source_broker", "_resources",
        "_original", "_literal", "_run_refs", "_run_input", "_binding", "_state", "_original_state")

    def __init__(self, seal, *, token, allocation, run_coordinator, run_lease, candidate_prompt,
            prior_worker_attempt, worker_attempt, resources=None):
        _need(seal is _SEAL and type(self) is NativeCandidateRoundAdmission,
            "candidate admission must be captured by its host owner")
        values = dict(_seal=seal, _token=token, _allocation=allocation, _run_coordinator=run_coordinator,
            _run_lease=run_lease, _candidate_prompt=candidate_prompt,
            _prior_worker_attempt=prior_worker_attempt, _worker_attempt=worker_attempt,
            _resources=resources,
            _snapshot=candidate_prompt._snapshot, _source_broker=candidate_prompt._snapshot._source_broker,
            _state=_State(), _binding=None)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_original_state", self._state)
        object.__setattr__(self, "_original", self._references())
        object.__setattr__(self, "_literal", self._literals())
        object.__setattr__(self, "_run_refs", _coordinator_refs(run_coordinator))
        object.__setattr__(self, "_run_input", _wire(_lease_material(run_lease)))

    def __setattr__(self, name, value):
        raise AttributeError("native candidate admission is immutable")

    def _references(self):
        return (self._allocation, self._allocation._task_admission, self._run_coordinator, self._run_lease,
            self._candidate_prompt, self._prior_worker_attempt, self._worker_attempt, self._snapshot,
            self._source_broker, self._state, self._prior_worker_attempt._state,
            self._worker_attempt._state, self._prior_worker_attempt._state.prepared,
            self._prior_worker_attempt._state.spawned, self._prior_worker_attempt._state.retired_event,
            self._worker_attempt._state.prepared, self._resources)

    def _literals(self):
        return (self._allocation._material, self._allocation._task_bytes,
            self._candidate_prompt.provider_prompt, self._candidate_prompt.metadata_bytes,
            self._snapshot.metadata_bytes, self._snapshot.native_prompt, self._snapshot._baseline_bytes,
            self._snapshot._absolute_deadline, self._prior_worker_attempt._common, self._worker_attempt._common,
            None if self._resources is None else self._resources.metadata_bytes)

    def _issued(self):
        _need(type(self) is NativeCandidateRoundAdmission and self._seal is _SEAL
            and self._state is self._original_state, "exact live candidate admission required")
        with _REGISTRY_LOCK:
            original = _ISSUED.get(self)
            _need(type(original) is tuple and original[0] is self._token
                and original[1] == self._binding
                and _PRIOR_ADMISSIONS.get(self._source_broker) is self._token,
                "candidate admission is not its original owner issuance")

    def _retained(self):
        _need(type(self) is NativeCandidateRoundAdmission and self._seal is _SEAL
            and self._state is self._original_state
            and all(a is b for a, b in zip(self._references(), self._original))
            and self._literals() == self._literal
            and self._candidate_prompt._snapshot is self._snapshot
            and self._snapshot._source_broker is self._source_broker,
            "candidate admission original inputs changed")
        with _REGISTRY_LOCK:
            original = _ISSUED.get(self)
            _need(original is None or (original[0] is self._token and original[1] == self._binding),
                "candidate admission original observation binding changed")
        references, configuration, identity = _coordinator_refs(self._run_coordinator)
        _need(all(a is b for a, b in zip(references, self._run_refs[0]))
            and configuration == self._run_refs[1] and identity == self._run_refs[2]
            and _wire(_lease_material(self._run_lease)) == self._run_input,
            "candidate admission original run authority changed")

    @property
    def process(self):
        NativeCandidateRoundAdmission._issued(self)
        return self._state.process

    @property
    def allocation(self):
        NativeCandidateRoundAdmission._issued(self)
        return self._original[0]

    @property
    def candidate_prompt(self):
        NativeCandidateRoundAdmission._issued(self)
        return self._original[4]

    @property
    def worker_attempt(self):
        NativeCandidateRoundAdmission._issued(self)
        return self._original[6]

    @property
    def resources(self):
        NativeCandidateRoundAdmission._issued(self)
        return self._resources

    def _resources_current(self):
        if self._resources is not None:
            from .native_candidate_round_resources import NativeCandidateRoundResources
            NativeCandidateRoundResources.revalidate(self._resources)

    def _resources_final(self):
        if self._resources is not None:
            from .native_candidate_round_resources import NativeCandidateRoundResources
            if NativeCandidateRoundResources.poll(self._resources):
                raise RuntimeError("candidate resource reservation is no longer current")
            NativeCandidateRoundResources.remaining(self._resources)

    def _run_current(self):
        NativeCandidateRoundAdmission._retained(self)
        coordinator, lease = self._run_coordinator, self._run_lease
        arguments = dict(expected_fencing_token=lease.fencing_token, expected_fence_epoch=lease.fence_epoch)
        if type(coordinator) is DatabaseCoordinator:
            current = DatabaseCoordinator.protect_write(coordinator, lease, **arguments)
        else:
            current = ProcessSerializedDatabaseCoordinator._invoke(coordinator, "protect_write", lease, **arguments)
        material = _lease_material(current)
        _need(_lease_identity(material) == _lease_identity(json.loads(self._run_input))
            and current.revision >= lease.revision,
            "protected run lease differs from its exact original identity")
        _expiry(current, minimum_ms=MINIMUM_REMAINING_MS)
        NativeCandidateRoundAdmission._retained(self)
        return current

    def _before_spawn(self):
        NativeCandidateRoundAdmission._retained(self)
        NativeCandidateRoundAdmission._resources_current(self)
        protected = NativeCandidateRoundAdmission._run_current(self)
        prior, prior_events = _attempt(self._prior_worker_attempt, phase="retired")
        current, current_events = _attempt(self._worker_attempt, phase="prepared")
        prompt = OwnerCandidateRoundPrompt.revalidate(self._candidate_prompt)
        snapshot = OwnerCandidateRoundContext.revalidate(self._snapshot)
        _need(type(snapshot["round_number"]) is int and snapshot["round_number"] == 2,
            "native candidate admission requires exactly round two")
        material = json.loads(self._allocation._material)
        source = self._source_broker
        _need(source.repository == self._allocation._canonical_repository == Path(material["canonical_repository"])
            and source.workspace == self._allocation._workspace == Path(material["workspace"])
            and snapshot["baseline_binding"]["task_id"] == prompt["task_id"] == material["task_alias"],
            "candidate source or task alias differs from native allocation")
        _need(self._prior_worker_attempt._state.process is source._process
            and source._process.returncode == snapshot["prior_returncode"]
            and self._prior_worker_attempt._state.child[0] == snapshot["prior_child"]["pid"]
            and self._prior_worker_attempt._state.child[2] == snapshot["prior_child"]["start_time_ticks"],
            "candidate snapshot does not retain the original retired worker")
        deadline = attempts._deadline_ns(self._snapshot._absolute_deadline)
        _need(prior["absolute_deadline_ns"] == current["absolute_deadline_ns"] == deadline
            and prior["attempt_id"] != current["attempt_id"]
            and prior["native_prompt_sha256"] == _sha(self._snapshot.native_prompt)
            and current["native_prompt_sha256"] == _sha(self._candidate_prompt.provider_prompt),
            "candidate attempt changed its original deadline or complete prompt")
        for attempt, row in ((self._prior_worker_attempt, prior), (self._worker_attempt, current)):
            _need(attempt._workspace == source.workspace and attempt._boundary == source.boundary
                and row["workspace_identity_sha256"] == _sha(str(source.workspace).encode("utf-8"))
                and row["boundary_manifest_sha256"] == source.descriptor["boundary_manifest_sha256"],
                "candidate worker attempt workspace or boundary differs")
        NativeCandidateRoundAdmission._retained(self)
        # Native controls close AFTER source/producer and owner-file observers.
        # This does not enter another claim transaction or read source bytes.
        NativeSourceWorktreeAllocation._observe_callback_boundary(self._allocation)
        NativeCandidateRoundAdmission._resources_final(self)
        _expiry(protected, minimum_ms=MINIMUM_REMAINING_MS)
        if time.monotonic_ns() + MINIMUM_REMAINING_MS * 1_000_000 >= deadline:
            raise TimeoutError("original candidate deadline lacks a full birth budget")
        return protected, prior, prior_events, current, current_events

    def _observation(self):
        _need(type(self._binding) is bytes, "candidate admission binding is not prepared")
        value = json.loads(self._binding)
        state = self._state
        value.update(consumed=state.consumed, spawn_started=state.spawn_started,
            spawn_completed=state.spawn_completed, child_pid=None if state.child is None else state.child[0],
            child_start_ticks=None if state.child is None else state.child[2])
        return value

    def revalidate(self):
        """Check current inputs; a successful birth never restores this permit.

        After birth this still checks the same advisory source generation, for
        the caller's final pre-input fence. Provider edits may make it stale;
        this method neither refreshes it nor admits another child.

        Source and detached native closing observations are bounded checks,
        not a filesystem transaction or distributed atomicity guarantee. Any
        late refusal retains the caller's cleanup and possible-effects duties.
        """
        NativeCandidateRoundAdmission._issued(self)
        with self._state.lock:
            _need(not self._state.executing, "candidate admission reentry refused")
            protected_leases = {}
            def checked(claim_lease):
                if self._state.process is None:
                    protected, _prior, _prior_events, _current, _current_events = (
                        NativeCandidateRoundAdmission._before_spawn(self))
                else:
                    NativeCandidateRoundAdmission._retained(self)
                    NativeCandidateRoundAdmission._resources_current(self)
                    protected = NativeCandidateRoundAdmission._run_current(self)
                    _attempt(self._prior_worker_attempt, phase="retired")
                    _attempt(self._worker_attempt, phase="spawned", process=self._state.process)
                    OwnerCandidateRoundPrompt.revalidate(self._candidate_prompt)
                    OwnerCandidateRoundContext.revalidate(self._snapshot)
                    NativeCandidateRoundAdmission._retained(self)
                    NativeSourceWorktreeAllocation._observe_callback_boundary(self._allocation)
                protected_leases.update(task=claim_lease, run=protected)
            NativeSourceWorktreeAllocation.execute_with_current_claim(self._allocation,
                checked, minimum_remaining_ms=MINIMUM_REMAINING_MS)
            # Observe one current run-lease result after closing native clocks,
            # including an authentic same-fence shortening. This authority's
            # clock also precedes the final source verification below.
            NativeCandidateRoundAdmission._resources_current(self)
            protected_leases["run"] = NativeCandidateRoundAdmission._run_current(self)
            # The native coordinator's closing clock may invoke owner code.
            # Recheck source after it has finished; no coordinator/observer
            # clock runs again after this final source verification.
            OwnerCandidateRoundPrompt.revalidate(self._candidate_prompt)
            OwnerCandidateRoundContext.revalidate(self._snapshot)
            NativeCandidateRoundAdmission._retained(self)
            with self._allocation._server._lock:
                NativeSourceWorktreeAllocation._observe_callback_boundary(self._allocation)
            NativeCandidateRoundAdmission._resources_final(self)
            _expiry(protected_leases["task"], minimum_ms=MINIMUM_REMAINING_MS)
            _expiry(protected_leases["run"], minimum_ms=MINIMUM_REMAINING_MS)
            attempts._remaining(json.loads(self._worker_attempt._common)["absolute_deadline_ns"])
            return NativeCandidateRoundAdmission._observation(self)

    def execute_spawn(self, spawn, register):
        """Invoke one trusted spawn and retain all possible effects on failure.

        ``spawn()`` must return its actual direct Popen child. The gate records
        that birth in the owner census; ``register(process)`` attaches external
        brokers. It never kills, waits, releases prompt input or calls a model.
        """
        NativeCandidateRoundAdmission._issued(self)
        _need(callable(spawn) and callable(register), "trusted spawn and registration callbacks required")
        with self._state.lock:
            state = self._state
            _need(not state.consumed and not state.executing, "candidate admission is already consumed")
            state.consumed = state.executing = True
            try:
                protected_leases = {}
                def checked(claim_lease):
                    protected_leases["task"] = claim_lease
                    protected, _prior, _prior_events, _current, _current_events = (
                        NativeCandidateRoundAdmission._before_spawn(self))
                    _expiry(claim_lease, minimum_ms=MINIMUM_REMAINING_MS)
                    _expiry(protected, minimum_ms=MINIMUM_REMAINING_MS)
                    if self._resources is not None:
                        from .native_candidate_round_resources import NativeCandidateRoundResources
                        NativeCandidateRoundResources.note_spawn_attempt(self._resources)
                    state.spawn_started = True
                    process = spawn()
                    if type(process) is subprocess.Popen:
                        state.process = process
                    _need(type(process) is subprocess.Popen, "spawn callback must return its actual Popen child")
                    attempts.OwnerWorkerAttempt.note_spawned(self._worker_attempt, process)
                    if self._resources is not None:
                        NativeCandidateRoundResources.note_spawned(self._resources, process)
                    child = attempts._birth(process.pid)
                    state.child = child
                    _need(child[1] == os.getpid()
                        and os.readlink("/proc/" + str(process.pid) + "/cwd") == str(self._source_broker.workspace),
                        "candidate child direct parent or allocated cwd differs")
                    register(process)
                    # Never replay candidate sources after a real child birth.
                    NativeCandidateRoundAdmission._retained(self)
                    _attempt(self._worker_attempt, phase="spawned", process=process)
                    NativeCandidateRoundAdmission._run_current(self)
                    NativeCandidateRoundAdmission._retained(self)
                    return process
                result = NativeSourceWorktreeAllocation.execute_with_current_claim(self._allocation,
                    checked, minimum_remaining_ms=MINIMUM_REMAINING_MS)
                # Native coordinator closing callbacks have now finished.
                NativeCandidateRoundAdmission._resources_current(self)
                protected_leases["run"] = NativeCandidateRoundAdmission._run_current(self)
                NativeCandidateRoundAdmission._retained(self)
                # The run authority has its own native clock. Close any of its
                # observer effects with detached task/allocation checks, without
                # replaying candidate source after a child has been born.
                with self._allocation._server._lock:
                    NativeSourceWorktreeAllocation._observe_callback_boundary(self._allocation)
                NativeCandidateRoundAdmission._resources_final(self)
                _expiry(protected_leases["task"], minimum_ms=MINIMUM_REMAINING_MS)
                _expiry(protected_leases["run"], minimum_ms=MINIMUM_REMAINING_MS)
                attempts._remaining(json.loads(self._worker_attempt._common)["absolute_deadline_ns"])
                state.spawn_completed = True
                return result
            finally:
                state.executing = False


def capture_native_candidate_round_admission(*, allocation, run_coordinator, run_lease,
        candidate_prompt, prior_worker_attempt, worker_attempt, resources=None):
    """Capture one real host-native candidate admission per original source worker.

    This deliberately retains the existing allocation's protected-directory
    contract. It does not qualify the deployed writable container workspace.
    """
    _need(type(allocation) is NativeSourceWorktreeAllocation
        and type(candidate_prompt) is OwnerCandidateRoundPrompt
        and type(candidate_prompt._snapshot) is OwnerCandidateRoundContext
        and type(candidate_prompt._snapshot._source_broker) is WorkerSourceBroker
        and type(prior_worker_attempt) is attempts.OwnerWorkerAttempt
        and type(worker_attempt) is attempts.OwnerWorkerAttempt
        and prior_worker_attempt is not worker_attempt,
        "exact native allocation, candidate prompt and distinct owner worker attempts required")
    _need(type(run_coordinator) in (DatabaseCoordinator, ProcessSerializedDatabaseCoordinator)
        and type(run_lease) is FencedLease, "exact native run coordinator and lease required")
    task_coordinator = allocation._task_admission._coordinator
    run_identity = _coordinator_refs(run_coordinator)[2]
    task_identity = _owned_file(task_coordinator.database_path)
    _need(run_coordinator is not task_coordinator
        and run_coordinator.database_path != task_coordinator.database_path
        and run_identity[1:3] != task_identity[1:3],
        "run and task coordination must be independent native authorities")
    material = _lease_material(run_lease)
    repository_id = allocation._server.identity.repository_id
    _need(type(repository_id) is str and repository_id and run_lease.repository_id == repository_id,
        "run lease repository differs from the native task owner")
    if resources is not None:
        from .native_candidate_round_resources import NativeCandidateRoundResources
        _need(type(resources) is NativeCandidateRoundResources
            and resources.worker_attempt is worker_attempt
            and resources.repository_id == repository_id
            and resources.workspace == allocation._workspace
            and resources.absolute_deadline_ns == json.loads(worker_attempt._common)["absolute_deadline_ns"],
            "candidate resources differ from exact native owner, attempt, workspace or deadline")
        NativeCandidateRoundResources.revalidate(resources)
    source = candidate_prompt._snapshot._source_broker
    token = object()
    with _REGISTRY_LOCK:
        _need(source not in _PRIOR_ADMISSIONS and source not in _CAPTURING,
            "original failed worker already has a candidate admission")
        _CAPTURING[source] = token
    try:
        result = NativeCandidateRoundAdmission(_SEAL, token=token, allocation=allocation,
            run_coordinator=run_coordinator, run_lease=run_lease, candidate_prompt=candidate_prompt,
            prior_worker_attempt=prior_worker_attempt, worker_attempt=worker_attempt, resources=resources)
        def checked(_lease):
            _protected, prior, prior_events, current, current_events = (
                NativeCandidateRoundAdmission._before_spawn(result))
            return dict(schema=SCHEMA, admission_id=uuid.uuid4().hex, round_number=2,
                allocation_sha256=_sha(allocation._material), allocation_cid=json.loads(allocation._material)["allocation_cid"],
                native_task_binding_sha256=_sha(allocation._task_bytes),
                run_lease_identity_sha256=_sha(_wire(_lease_identity(material))),
                run_coordinator_identity_sha256=_sha(_file_pin(run_identity)),
                candidate_prompt_sha256=_sha(candidate_prompt.provider_prompt),
                candidate_context_sha256=_sha(candidate_prompt._snapshot.metadata_bytes),
                original_native_prompt_sha256=prior["native_prompt_sha256"],
                workspace_identity_sha256=current["workspace_identity_sha256"],
                boundary_manifest_sha256=current["boundary_manifest_sha256"],
                prior_worker_attempt_id=prior["attempt_id"], worker_attempt_id=current["attempt_id"],
                prior_retired_event_sha256=_sha(prior_events[-1].raw),
                worker_prepared_event_sha256=_sha(current_events[0].raw),
                absolute_deadline_ns=current["absolute_deadline_ns"], host_native_admission_only=True,
                serialized_observation_grants_authority=False, native_run_lease_checked=True,
                quantitative_resource_reservation_verified=resources is not None,
                resource_reservation_sha256=None if resources is None else _sha(resources.metadata_bytes),
                candidate_artifacts_advisory=True,
                proof_authority=False, completion_authority=False, retry_authority=False,
                production_container_dispatch_qualified=False)
        binding = NativeSourceWorktreeAllocation.execute_with_current_claim(allocation,
            checked, minimum_remaining_ms=MINIMUM_REMAINING_MS)
        NativeCandidateRoundAdmission._run_current(result)
        NativeCandidateRoundAdmission._retained(result)
        NativeCandidateRoundAdmission._resources_final(result)
        object.__setattr__(result, "_binding", _wire(binding))
        with _REGISTRY_LOCK:
            _need(_CAPTURING.get(source) is token and source not in _PRIOR_ADMISSIONS,
                "original candidate admission issuance changed")
            _PRIOR_ADMISSIONS[source] = token
            _ISSUED[result] = (token, result._binding)
        if resources is not None:
            NativeCandidateRoundResources.bind_admission(resources, result)
        return result
    finally:
        with _REGISTRY_LOCK:
            if _CAPTURING.get(source) is token:
                del _CAPTURING[source]


__all__ = ["SCHEMA", "MINIMUM_REMAINING_MS", "NativeCandidateRoundAdmission",
    "capture_native_candidate_round_admission"]
