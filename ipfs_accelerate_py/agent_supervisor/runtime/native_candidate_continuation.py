"""Compose one host-owner candidate continuation from existing live handles.

Preparation follows an actually failed and fully retired first child. The
caller retains ownership of its allocation, run coordinator/lease and native
resource parent. No serialized checkpoint can reconstruct this session, and
neither this composer nor its observation enables the deployed wrapper route.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import os
import threading
import uuid
import weakref

from . import native_candidate_round_admission as admissions
from . import native_candidate_round_resources as reservations
from . import native_candidate_round_worker as workers
from . import owner_candidate_round_context as contexts
from . import owner_candidate_round_prompt as prompts
from . import owner_worker_attempt as attempts
from .candidate_dispatch_broker import CandidateDispatchBroker
from .candidate_dispatch_client import validate_policy
from .current_source_worktree_allocation import NativeSourceWorktreeAllocation
from .worker_source_broker import WorkerSourceBroker

SCHEMA = "native-candidate-continuation@1"
_SEAL = object()
_LOCK = threading.RLock()
_PREPARATIONS = weakref.WeakKeyDictionary()
_PENDING = {}
_RESERVED_FLAGS = (
    "--owner-worker-attempt-id", "--owner-provider-deadline-ns",
    "--candidate-dispatch-descriptor", "--candidate-dispatch-descriptor-sha256",
    "--candidate-round-descriptor", "--candidate-round-descriptor-sha256",
    "--source-view-descriptor", "--source-view-descriptor-sha256",
    "--source-lookup-descriptor", "--source-lookup-descriptor-sha256",
)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


@dataclass(slots=True)
class _State:
    inputs: tuple
    token: object = field(default_factory=object)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    lock: object = field(default_factory=threading.RLock)
    snapshot: object = None
    prompt: object = None
    attempt: object = None
    resources: object = None
    admission: object = None
    broker: object = None
    prepared: bool = False
    run_spent: bool = False
    operating: bool = False


class NativeCandidateContinuation:
    """One nonserializable owner session; the existing native gate admits birth."""
    __slots__ = ("__weakref__", "_seal", "_inputs", "_state")

    def __init__(self, seal, inputs):
        _need(seal is _SEAL and type(self) is NativeCandidateContinuation,
            "live owner continuation factory required")
        object.__setattr__(self, "_seal", seal)
        object.__setattr__(self, "_inputs", inputs)
        object.__setattr__(self, "_state", _State(inputs))

    def __setattr__(self, name, value):
        raise AttributeError("native candidate continuation is immutable")

    def _retained(self):
        _need(type(self) is NativeCandidateContinuation and self._seal is _SEAL
            and type(self._state) is _State and self._state.inputs is self._inputs,
            "exact live owner continuation required")
        with _LOCK:
            _need(_PREPARATIONS.get(self._inputs[4]) is self._state.token,
                "continuation lost its original preparation issuance")
        return self._state

    @property
    def source_broker(self):
        NativeCandidateContinuation._retained(self)
        return self._inputs[4]

    @property
    def prior_worker_attempt(self):
        NativeCandidateContinuation._retained(self)
        return self._inputs[5]

    @property
    def candidate_prompt(self):
        return NativeCandidateContinuation._retained(self).prompt

    @property
    def worker_attempt(self):
        return NativeCandidateContinuation._retained(self).attempt

    @property
    def resources(self):
        return NativeCandidateContinuation._retained(self).resources

    @property
    def admission(self):
        return NativeCandidateContinuation._retained(self).admission

    @property
    def broker(self):
        return NativeCandidateContinuation._retained(self).broker

    @property
    def extra_argv(self):
        state = NativeCandidateContinuation._retained(self)
        _need(state.prepared and state.broker is not None and state.attempt is not None,
            "completed owner candidate preparation required")
        return [*state.broker.extra_argv, "--owner-worker-attempt-id", state.attempt.attempt_id]

    @property
    def metadata(self):
        state = NativeCandidateContinuation._retained(self)
        prior = json.loads(self._inputs[5]._common)
        resource = state.resources
        process = None if state.admission is None else state.admission.process
        attempted = state.admission is not None and state.admission._state.spawn_started
        return dict(schema=SCHEMA, session_id=state.session_id, max_worker_children=2,
            max_candidate_children=1, round_number=2, prepared=state.prepared,
            run_consumed=state.run_spent, prior_worker_attempt_id=prior["attempt_id"],
            worker_attempt_id=None if state.attempt is None else state.attempt.attempt_id,
            absolute_deadline_ns=prior["absolute_deadline_ns"],
            original_requirements_sha256=_sha(self._inputs[6].metadata_bytes),
            candidate_spawn_attempted=attempted,
            observed_worker_children=1 + int(process is not None),
            worker_child_count_complete=not attempted or process is not None,
            quantitative_resource_reservation_verified=resource is not None,
            resource_capacity_held=resource is not None and not
                reservations.NativeCandidateRoundResources._observation(resource)["released"],
            host_owner_continuation_only=True, production_container_dispatch_qualified=False,
            serialized_observation_grants_authority=False, proof_authority=False,
            completion_authority=False, retry_authority=False)

    def _keep_pending(self):
        state = NativeCandidateContinuation._retained(self)
        released = state.resources is None
        if state.resources is not None:
            try:
                released = reservations.NativeCandidateRoundResources._observation(state.resources)["released"]
            except BaseException:
                # Lost observation cannot establish actual native row absence.
                released = False
        with _LOCK:
            if not released:
                _PENDING[state.session_id] = self
            else:
                _PENDING.pop(state.session_id, None)

    def _close_unstarted_broker(self):
        state = NativeCandidateContinuation._retained(self)
        broker = state.broker
        if broker is None:
            return
        if not hasattr(broker, "_listener"):
            _need(not any(hasattr(broker, name) for name in
                ("_thread", "socket_path", "descriptor_path")),
                "incomplete candidate reader construction remains uncertain")
            return
        CandidateDispatchBroker.close(broker)
        _need(broker._closed and broker._stop.is_set()
            and (broker._thread is None or not broker._thread.is_alive())
            and all(not path.exists() and not path.is_symlink() for path in
                (broker.socket_path, broker.descriptor_path)),
            "candidate reader has not fully retired")

    def abort_unstarted(self):
        """Retire only a session that never attempted candidate process creation."""
        state = NativeCandidateContinuation._retained(self)
        with state.lock:
            _need(not state.operating, "continuation operation is already active")
            state.operating = True
            state.run_spent = True
            try:
                _need(state.admission is None or not state.admission._state.spawn_started,
                    "attempted candidate birth requires explicit owner recovery")
                NativeCandidateContinuation._close_unstarted_broker(self)
                return (None if state.resources is None else
                    reservations.NativeCandidateRoundResources.abort_unstarted(state.resources))
            except BaseException as error:
                _attach(error, self)
                raise
            finally:
                state.operating = False
                NativeCandidateContinuation._keep_pending(self)

    def recover_retirement(self, *, cleanup):
        """Use the existing native resource recovery without rewriting census."""
        state = NativeCandidateContinuation._retained(self)
        with state.lock:
            _need(not state.operating and state.resources is not None,
                "inactive continuation with retained native resources required")
            state.operating = True
            state.run_spent = True
            try:
                return reservations.NativeCandidateRoundResources.recover_retirement(
                    state.resources, cleanup=cleanup)
            except BaseException as error:
                _attach(error, self)
                raise
            finally:
                state.operating = False
                NativeCandidateContinuation._keep_pending(self)

    def run(self, *, command, environment, cleanup, stdout=None, stderr=None):
        """Spend once and run the existing worker with owner-generated bindings."""
        state = NativeCandidateContinuation._retained(self)
        with state.lock:
            _need(not state.run_spent and not state.operating,
                "candidate continuation run is already consumed")
            state.run_spent = True
            state.operating = True
        try:
            _need(state.prepared, "completed owner candidate preparation required")
            literal = workers._command(command)
            _need(not any(part == flag or part.startswith(flag + "=")
                for part in literal for flag in _RESERVED_FLAGS),
                "candidate command contains owner-generated binding flags")
            return workers.run_native_candidate_round_worker(admission=state.admission,
                source_broker=state.broker, command=(*literal, *self.extra_argv),
                environment=environment, cleanup=cleanup, stdout=stdout, stderr=stderr)
        except BaseException as error:
            _attach(error, self)
            raise
        finally:
            with state.lock:
                state.operating = False
            NativeCandidateContinuation._keep_pending(self)


def _attach(error, session):
    # Retain exact typed custody without replacing the original error outcome.
    try:
        error.candidate_continuation = session
    except BaseException:
        pass


def _sealed_prompt(raw):
    from ..validation.validation_runtime import _validation_memfd_create
    required = sum(getattr(fcntl, name, fallback) for name, fallback in (
        ("F_SEAL_SEAL", 1), ("F_SEAL_SHRINK", 2), ("F_SEAL_GROW", 4), ("F_SEAL_WRITE", 8)))
    fd = _validation_memfd_create("supervisor-candidate-continuation-prompt",
        getattr(os, "MFD_CLOEXEC", 1) | getattr(os, "MFD_ALLOW_SEALING", 2))
    try:
        remaining = memoryview(raw)
        while remaining:
            count = os.write(fd, remaining)
            _need(count > 0, "candidate prompt write made no progress")
            remaining = remaining[count:]
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, getattr(fcntl, "F_ADD_SEALS", 1033), required)
        _need(os.pread(fd, len(raw) + 1, 0) == raw and os.fstat(fd).st_size == len(raw)
            and fcntl.fcntl(fd, getattr(fcntl, "F_GET_SEALS", 1034)) & required == required,
            "sealed candidate prompt differs from its exact owner bytes")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _preparation_authority(*, allocation, run_coordinator, run_lease, parent_lease,
        cpu_slots, memory_mb, child_process_slots, threads_per_process):
    """Reuse the native kernels before paying for fresh candidate generation."""
    run_refs = admissions._coordinator_refs(run_coordinator)
    run_identity = run_refs[2]
    task_coordinator = allocation._task_admission._coordinator
    task_identity = admissions._owned_file(task_coordinator.database_path)
    _need(run_coordinator is not task_coordinator
        and run_coordinator.database_path != task_coordinator.database_path
        and run_identity[1:3] != task_identity[1:3],
        "run and task coordination require independent native authorities")
    original = admissions._lease_material(run_lease)
    _need(run_lease.repository_id == allocation._server.identity.repository_id,
        "run lease repository differs from the native owner")
    arguments = dict(expected_fencing_token=run_lease.fencing_token,
        expected_fence_epoch=run_lease.fence_epoch)
    if type(run_coordinator) is admissions.DatabaseCoordinator:
        current = admissions.DatabaseCoordinator.protect_write(run_coordinator, run_lease, **arguments)
    else:
        current = admissions.ProcessSerializedDatabaseCoordinator._invoke(
            run_coordinator, "protect_write", run_lease, **arguments)
    _need(admissions._lease_identity(admissions._lease_material(current))
        == admissions._lease_identity(original) and current.revision >= run_lease.revision,
        "original supervisor run lease changed during candidate preparation")
    admissions._expiry(current, minimum_ms=admissions.MINIMUM_REMAINING_MS)
    _need(type(parent_lease) is reservations.ResourceLease
        and isinstance(parent_lease._scheduler, reservations.GlobalResourceScheduler),
        "actual existing native resource parent required")
    scheduler = parent_lease._scheduler
    scheduler_material = reservations._scheduler_material(scheduler)
    material = reservations._lease_material(parent_lease, scheduler)
    reservations._native_rows(scheduler, ((parent_lease, material),), current=True)
    parent = json.loads(material)
    for name, value, maximum in (("cpu_slots", cpu_slots, 1024), ("memory_mb", memory_mb, 2**24),
            ("child_process_slots", child_process_slots, 1024),
            ("threads_per_process", threads_per_process, 1024)):
        reservations._positive(value, maximum, name)
    _need(not parent_lease.released and cpu_slots <= parent["cpu_slots"]
        and memory_mb <= parent["memory_mb"] and child_process_slots <= parent["child_process_slots"]
        and threads_per_process <= cpu_slots, "candidate demand exceeds its actual native parent")
    return material, scheduler_material, admissions._wire(original), run_refs


def _unchanged_authority(original, current):
    """Compare existing kernel snapshots without adopting later capabilities."""
    old_parent, old_scheduler, old_run, old_coordinator = original
    parent, scheduler, run, coordinator = current
    _need(parent == old_parent and run == old_run
        and scheduler[:3] == old_scheduler[:3]
        and all(a is b if not isinstance(a, tuple) else a == b
            for a, b in zip(scheduler[3], old_scheduler[3]))
        and all(a is b for a, b in zip(coordinator[0], old_coordinator[0]))
        and coordinator[1:] == old_coordinator[1:],
        "original resource parent, scheduler policy or run authority changed during preparation")


def prepare_native_candidate_continuation(*, allocation, run_coordinator, run_lease,
        parent_lease, source_broker, prior_worker_attempt, requirements, absolute_deadline,
        max_worker_children, provider, model, timeout, max_output_tokens, reasoning_effort,
        cpu_slots, memory_mb, child_process_slots, threads_per_process=1):
    """Compose round two once, after the first actual failed child is quiescent.

    Failure consumes preparation. The original exception exposes this exact
    session when available; uncertain native capacity stays retained until the
    owner explicitly aborts an unstarted session or recovers its known child.
    """
    _need(type(source_broker) is WorkerSourceBroker, "exact first owner source broker required")
    inputs = (allocation, run_coordinator, run_lease, parent_lease, source_broker,
        prior_worker_attempt, requirements, absolute_deadline, max_worker_children,
        (provider, model, timeout, max_output_tokens, reasoning_effort),
        (cpu_slots, memory_mb, child_process_slots, threads_per_process))
    session = NativeCandidateContinuation(_SEAL, inputs)
    state = session._state
    with _LOCK:
        _need(source_broker not in _PREPARATIONS,
            "original source worker already consumed candidate preparation")
        _PREPARATIONS[source_broker] = state.token
    try:
        _need(type(max_worker_children) is int and max_worker_children == 2,
            "explicit two-worker-child continuation policy required")
        _need(type(absolute_deadline) is float, "exact original float worker deadline required")
        prior, _events = admissions._attempt(prior_worker_attempt, phase="retired")
        _need(attempts._deadline_ns(absolute_deadline) == prior["absolute_deadline_ns"],
            "candidate continuation changed the original worker deadline")
        _need(type(requirements) is prompts.OriginalRequirements
            and requirements._source_broker is source_broker
            and source_broker.original_requirements is requirements,
            "exact original requirements captured before the first worker required")
        _need(type(allocation) is NativeSourceWorktreeAllocation
            and source_broker.workspace == allocation._workspace
            and source_broker.repository == allocation._canonical_repository
            and prior_worker_attempt._state.process is source_broker._process
            and prior_worker_attempt._boundary == source_broker.boundary
            and prior_worker_attempt._workspace == source_broker.workspace,
            "first worker and source broker differ from their native allocation")
        baseline = contexts._closed_baseline(source_broker, prior_worker_attempt._state.process.returncode)
        _need(baseline["prior_child"]["pid"] == prior_worker_attempt._state.child[0]
            and baseline["prior_child"]["start_time_ticks"] == prior_worker_attempt._state.child[2]
            and _sha(source_broker.prompt.encode("utf-8")) == prior["native_prompt_sha256"],
            "first worker census and closed source birth differ")
        NativeSourceWorktreeAllocation.require_current(allocation)
        authority_arguments = dict(allocation=allocation, run_coordinator=run_coordinator,
            run_lease=run_lease, parent_lease=parent_lease, cpu_slots=cpu_slots,
            memory_mb=memory_mb, child_process_slots=child_process_slots,
            threads_per_process=threads_per_process)
        authority = _preparation_authority(**authority_arguments)
        validate_policy(provider=provider, model=model, timeout=timeout,
            max_output_tokens=max_output_tokens, reasoning_effort=reasoning_effort)
        state.snapshot = contexts.prepare_candidate_round_context(source_broker=source_broker,
            prior_returncode=prior_worker_attempt._state.process.returncode, round_number=2,
            absolute_deadline=absolute_deadline)
        state.prompt = prompts.prepare_candidate_round_prompt(snapshot=state.snapshot, requirements=requirements)
        fd = _sealed_prompt(state.prompt.provider_prompt)
        try:
            state.attempt = attempts.prepare_owner_worker_attempt(boundary=source_broker.boundary,
                workspace=source_broker.workspace, input_fd=fd, absolute_deadline=absolute_deadline)
        finally:
            os.close(fd)
        _unchanged_authority(authority, _preparation_authority(**authority_arguments))
        try:
            state.resources = reservations.reserve_native_candidate_round_resources(
                parent_lease=parent_lease, worker_attempt=state.attempt,
                repository_id=allocation._server.identity.repository_id, cpu_slots=cpu_slots,
                memory_mb=memory_mb, child_process_slots=child_process_slots,
                threads_per_process=threads_per_process)
        except BaseException as error:
            retained = getattr(error, "candidate_resources", None)
            if type(retained) is reservations.NativeCandidateRoundResources:
                state.resources = retained
            raise
        finally:
            NativeCandidateContinuation._keep_pending(session)
        _unchanged_authority(authority, _preparation_authority(**authority_arguments))
        state.admission = admissions.capture_native_candidate_round_admission(allocation=allocation,
            run_coordinator=run_coordinator, run_lease=run_lease, candidate_prompt=state.prompt,
            prior_worker_attempt=prior_worker_attempt, worker_attempt=state.attempt, resources=state.resources)
        # Keep the actual object before initialization can create any service.
        state.broker = object.__new__(CandidateDispatchBroker)
        CandidateDispatchBroker.__init__(state.broker, admission=state.admission,
            provider=provider, model=model, timeout=timeout,
            max_output_tokens=max_output_tokens, reasoning_effort=reasoning_effort)
        state.prepared = True
        return session
    except BaseException as error:
        _attach(error, session)
        NativeCandidateContinuation._keep_pending(session)
        raise


__all__ = ["SCHEMA", "NativeCandidateContinuation", "prepare_native_candidate_continuation"]
