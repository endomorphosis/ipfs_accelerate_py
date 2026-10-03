"""Opt-in payload and durable disk bounds under one repository host envelope.

The native scheduler still owns host admission. This adapter partitions that
already admitted parent, keeps immutable queued payloads within explicit bounds,
and composes the existing daemon disk/RSS ledger. It is a cooperative CPU profile,
not a cgroup, device allocator, or filesystem quota.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from .repository_resource_bridge import (
    MIB, RepositoryResourceBridge, RepositoryResourceBudget,
    RepositoryPhaseDemand, RepositoryResourceError,
)
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_daemon_resources import (
    DaemonResourceReservation, DaemonResourceError,
)
from ipfs_datasets_py.logic.backends.codebase_process import (
    BoundedToolRunner, SubprocessExecutor, ToolRunLimits, _terminate_process_tree,
)

SCHEMA = "repository-pipeline-resources@1"
PROFILE = "cpu-protected-payload-durable-disk@1"
PROTECTED = frozenset({"validation", "cleanup"})
DIMENSIONS = ("cpu_slots", "memory_mb", "process_slots", "disk_bytes")
# Unsafe scopes must not release generator-owned leases through garbage
# collection. Entries exist only for admitted host roots and disappear after
# explicit reaping. Native lease expiry does not bound Python handle lifetime.
_PENDING_REAP = {}
_PENDING_REAP_LOCK = threading.RLock()
# Process-local Python lifecycle admission only: no CPU/RAM/disk authority.
# Waiting for host admission already occupies a slot. One cleanup-only slot
# remains available when ordinary lifecycles have exhausted their compartment.
MAX_PIPELINE_LIFECYCLES = 32
RECOVERY_PIPELINE_LIFECYCLES = 1
_PIPELINE_LIFECYCLES = {}


def _claim_lifecycle(recovery_only):
    _require(type(recovery_only) is bool, "exact recovery_only Boolean required")
    with _PENDING_REAP_LOCK:
        limit = RECOVERY_PIPELINE_LIFECYCLES if recovery_only else MAX_PIPELINE_LIFECYCLES - RECOVERY_PIPELINE_LIFECYCLES
        count = sum(row["recovery_only"] == recovery_only for row in _PIPELINE_LIFECYCLES.values())
        _require(count < limit and len(_PIPELINE_LIFECYCLES) < MAX_PIPELINE_LIFECYCLES,
                 "process-local pipeline lifecycle capacity exhausted; explicitly reap retained owners")
        token = uuid.uuid4().hex
        _PIPELINE_LIFECYCLES[token] = dict(recovery_only=recovery_only, owner=None)
        return token


def pipeline_lifecycle_inventory():
    """Bounded counts only; no lease capabilities or resource authority."""
    with _PENDING_REAP_LOCK:
        return dict(schema="repository-pipeline-lifecycle-inventory@1", scope="process_local_python_handles_only",
            maximum=MAX_PIPELINE_LIFECYCLES, recovery_only_maximum=RECOVERY_PIPELINE_LIFECYCLES,
            reserved=len(_PIPELINE_LIFECYCLES),
            awaiting_host_admission=sum(row["owner"] is None for row in _PIPELINE_LIFECYCLES.values()),
            ordinary=sum(not row["recovery_only"] for row in _PIPELINE_LIFECYCLES.values()),
            recovery_only=sum(row["recovery_only"] for row in _PIPELINE_LIFECYCLES.values()),
            pending_reap=len(_PENDING_REAP), host_resource_authority=False)

LIMITATIONS = (
    "CPU admission and process-group RSS sampling are not aggregate kernel enforcement",
    "disk authority covers the existing ledger's named roots, not all host filesystems",
    "payload bounds cover references retained by this adapter, not caller-owned copies or model tensors",
    "consumer APIs must honor native lease/cancellation/thread options and reap their own children",
    "validation and cleanup share a protected compartment and can queue behind each other",
    "GPU, unknown devices, hard enforcement and gradual host recovery are not qualified",
)


def _require(condition, message):
    if not condition:
        raise RepositoryResourceError(message)


def _positive(value, name, maximum):
    _require(type(value) is int and 0 < value <= maximum,
             name + " must be a bounded exact positive integer")


@dataclass(frozen=True)
class PipelineResourcePolicy:
    protected_cpu_slots: int = 1
    protected_memory_mb: int = 128
    protected_process_slots: int = 1
    protected_disk_bytes: int = 8 * MIB
    maximum_queued_payload_bytes: int = 4 * MIB
    protected_queued_payload_bytes: int = 64 * 1024
    maximum_retained_payload_bytes: int = 16 * MIB
    protected_retained_payload_bytes: int = MIB
    protected_queue_slots: int = 1
    protected_phase_slots: int = 8

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            _positive(getattr(self, name), name, 2**40)

    def validate(self, budget):
        _require(type(budget) is RepositoryResourceBudget, "exact repository budget required")
        for dimension in DIMENSIONS:
            _require(getattr(self, "protected_" + dimension) < getattr(budget, dimension),
                     "parent must leave ordinary and protected " + dimension)
        _require(self.protected_cpu_slots >= self.protected_process_slots,
                 "protected CPU slots must cover processes")
        _require(self.protected_queue_slots < budget.maximum_queued_phases,
                 "parent queue must retain ordinary and protected slots")
        _require(self.protected_phase_slots < budget.maximum_phases,
                 "parent history must retain ordinary and protected slots")
        _require(self.protected_queued_payload_bytes < self.maximum_queued_payload_bytes
                 <= self.maximum_retained_payload_bytes < budget.memory_mb * MIB,
                 "queued and retained payload bounds must fit parent RAM")
        _require(self.protected_queued_payload_bytes <= self.protected_retained_payload_bytes
                 < self.maximum_retained_payload_bytes,
                 "protected payload bounds must fit retained capacity")
        _require(self.protected_retained_payload_bytes < self.protected_memory_mb * MIB
                 and self.maximum_retained_payload_bytes - self.protected_retained_payload_bytes
                 < (budget.memory_mb - self.protected_memory_mb) * MIB,
                 "each retained payload compartment must fit its RAM partition")


class _UsageCancellation:
    def __init__(self, phase):
        self.phase, self.error = phase, None
        self.next_sample = 0.0

    def is_set(self):
        if self.phase.pipeline.parent.cancellation.is_set():
            return True
        now = time.monotonic()
        if now >= self.next_sample:
            self.next_sample = now + .1
            try:
                self.phase.check_usage()
            except BaseException as error:
                self.error = error
                return True
        return False


class PipelinePhase:
    """One synchronous consumer scope; successful exit requires finalization."""

    def __init__(self, pipeline, native, daemon, demand, payload, attempt):
        self.pipeline, self.native, self.daemon = pipeline, native, daemon
        self.demand, self.payload, self.attempt = demand, payload, attempt
        self._closed = False
        self._run_lock = threading.Lock()
        self._completed = False
        self._owner_thread = threading.get_ident()
        self.last_process_result = None

    def _live(self):
        _require(not self._closed and not self._completed, "phase is closed or finalized")
        _require(threading.get_ident() == self._owner_thread, "phase consumers must remain synchronous on their owner thread")
        self.pipeline.parent.remaining()

    def native_options(self):
        self._live()
        _require(not self.pipeline.recovery_only, "recovery-only lifecycle cannot delegate consumer capacity")
        return dict(parent_lease=self.daemon.native_lease,
                    cancel_event=self.pipeline.parent.cancellation,
                    timeout_seconds=self.pipeline.parent.remaining(),
                    memory_mb=self.demand.memory_mb)

    def thread_environment(self):
        self._live()
        return self.native.thread_environment()

    def check_usage(self):
        self._live()
        usage = self.daemon.check_usage(self.attempt)
        if usage["process_slot_estimate_exceeded"]:
            raise DaemonResourceError("sampled child process count exceeds phase capacity")
        return usage

    def charge_external(self, path, byte_count):
        """Precharge an output in these named roots before its owner writes it."""
        self._live()
        path = Path(path).absolute()
        _require(not any(part.is_symlink() for part in (path, *path.parents)),
                 "external output cannot traverse symlinks")
        _require(any(path == root or root in path.parents for root in self.daemon.roots),
                 "external output is outside named disk roots")
        return self.daemon.account_external_bytes(
            "pipeline-" + hashlib.sha256(str(path).encode()).hexdigest(), byte_count)

    def recover_retained(self, reservation_id, *, artifacts_durable=False, owner=None):
        self._live()
        _require(self.demand.phase == "cleanup", "disk recovery requires protected cleanup phase")
        owner = self.pipeline if owner is None else owner
        _require(type(owner) is RepositoryPipelineReservation,
                 "exact pipeline owner required for recovery")
        _require(owner.parent.shared.state_path == self.pipeline.parent.shared.state_path
                 and owner.roots == self.pipeline.roots
                 and Path(owner.ledger_path).absolute() == Path(self.pipeline.ledger_path).absolute(),
                 "recovery must share native authority and exact named disk scope")
        return owner._recover(reservation_id, artifacts_durable=artifacts_durable)

    def run(self, argv, *, timeout_seconds=30.0, max_output_bytes=65536):
        """Run an actual bounded process; admitted payload is its only stdin.

        Output remains bounded by the explicit reserved capture allowance. The
        existing runner owns termination/reaping; daemon sampling additionally
        observes the attempt directory, process group and durable disk claims.
        """
        self._live()
        _require(not self.pipeline.recovery_only, "recovery-only lifecycle cannot run workload processes")
        _positive(max_output_bytes, "max_output_bytes", MIB)
        _require(2 * max_output_bytes <= self.pipeline.output_allowance,
                 "capture exceeds reserved payload allowance")
        _require(type(argv) in (list, tuple) and 0 < len(argv) <= 256
                 and all(type(arg) is str and "\x00" not in arg for arg in argv)
                 and sum(len(arg.encode()) for arg in argv) <= 65536,
                 "bounded explicit argv required")
        _require(type(timeout_seconds) in (int, float) and 0 < timeout_seconds
                 <= self.pipeline.parent.remaining(), "bounded remaining process deadline required")
        _require(self._run_lock.acquire(blocking=False), "phase already runs a process")
        signal = _UsageCancellation(self)

        def launch(*args, **kwargs):
            self._live()
            process = subprocess.Popen(*args, **kwargs)
            try:
                self.daemon.check_usage(self.attempt, child_pid=process.pid)
            except BaseException:
                _terminate_process_tree(process, grace_seconds=.25)
                process.wait(timeout=2)
                raise
            return process

        try:
            limits = ToolRunLimits(timeout_seconds=timeout_seconds,
                resident_memory_bytes=self.demand.memory_mb * MIB,
                max_input_bytes=max(1, len(self.payload)),
                max_output_bytes=max_output_bytes,
                max_workspace_bytes=self.demand.disk_bytes)
            runner = BoundedToolRunner(executor=SubprocessExecutor(popen=launch),
                                       workspace_root=self.attempt)
            result = runner.run(argv, stdin=self.payload, limits=limits,
                cancellation=signal, environment=self.thread_environment())
            self.last_process_result = result
            if signal.error is not None:
                raise signal.error
            self.check_usage()
            return result
        finally:
            self._run_lock.release()

    def finalize(self, *, artifacts_durable=False):
        self._live()
        _require(not self._run_lock.locked(), "process must finish before finalization")
        _require(not self.pipeline._consumer_leases(self.daemon),
                 "native consumers must finish and release before finalization")
        self.check_usage()
        result = self.daemon.finalize(self.attempt, artifacts_durable=artifacts_durable)
        self._completed = True
        return result


class RepositoryPipelineReservation:
    def __init__(self, parent, policy, ledger_path, roots, *, lifecycle_token, recovery_only):
        self.parent, self.policy = parent, policy
        self._lifecycle_token, self.recovery_only = lifecycle_token, recovery_only
        self._outer_exited = False
        self.ledger_path, self.roots = ledger_path, roots
        self._condition = threading.Condition(threading.RLock())
        self._queued, self._active, self._disk, self._retained = {}, {}, {}, {}
        self._retained_contexts = {}
        self._history = {False: 0, True: 0}
        self._events = []
        self._attempts = set()
        self._closed = False
        # Each phase reserves room for two bounded output streams, in addition
        # to its immutable input. Consumer tensors are governed by phase RAM.
        self.output_allowance = 128 * 1024
        with _PENDING_REAP_LOCK:
            row = _PIPELINE_LIFECYCLES.get(lifecycle_token)
            _require(row is not None and row["owner"] is None and row["recovery_only"] is recovery_only,
                     "one reserved process-local lifecycle slot required")
            row["owner"] = self

    def _release_lifecycle_if_idle(self):
        # Callers hold _condition. Never use native lease expiry as evidence
        # that Python contexts, native consumers or disk claims were cleaned.
        if not self._outer_exited or self._queued or self._active or self._retained_contexts or not self.parent._closed:
            return False
        with _PENDING_REAP_LOCK:
            _PENDING_REAP.pop(self.parent.native.lease_id, None)
            _PIPELINE_LIFECYCLES.pop(self._lifecycle_token, None)
        return True

    def reap_lifecycle(self):
        """Finish an idle failed outer close; preserve every durable disk claim.

        Live/retained consumers must first be reaped with the existing protected
        cleanup API. This only retries the canonical parent's normal close.
        """
        with self._condition:
            _require(self._outer_exited and not self._queued and not self._active and not self._retained_contexts,
                     "pipeline contexts must finish before lifecycle reaping")
            self.parent.close()
            _require(self._release_lifecycle_if_idle(), "pipeline lifecycle remains unsafe")
            return pipeline_lifecycle_inventory()

    def _capacity(self, protected, dimension):
        reserve = getattr(self.policy, "protected_" + dimension)
        return reserve if protected else getattr(self.parent.budget, dimension) - reserve

    def _limit(self, protected, total, reserve):
        return reserve if protected else total - reserve

    def _fits(self, row):
        protected, demand = row["protected"], row["demand"]
        for dimension in DIMENSIONS:
            inventory = self._disk.values() if dimension == "disk_bytes" else self._active.values()
            used = sum(item[dimension] if dimension == "disk_bytes" else getattr(item["demand"], dimension)
                       for item in inventory if item["protected"] == protected)
            if used + getattr(demand, dimension) > self._capacity(protected, dimension):
                return False
        return True

    def _consumer_leases(self, daemon):
        record = daemon.to_dict()
        lease = record["resource_lease"]
        if lease is None:
            return []
        # Every live descendant has a live direct ancestor. Canonical recovery
        # runs during this read; no private lease capability is serialized.
        return [row for row in self.parent.shared.active_leases()
                if row.get("parent_lease_id") == lease["lease_id"]]

    @contextmanager
    def phase(self, demand, *, payload=b"", attempt_directory):
        _require(type(demand) is RepositoryPhaseDemand, "exact phase demand required")
        _require(type(payload) is bytes, "queued payload must be immutable bytes")
        _require(not self._closed, "pipeline is closed")
        _require(not self.recovery_only or demand.phase == "cleanup", "recovery-only lifecycle admits cleanup phases only")
        protected = demand.phase in PROTECTED
        for dimension in DIMENSIONS:
            _require(getattr(demand, dimension) <= self._capacity(protected, dimension),
                     "phase exceeds its protected or ordinary " + dimension)
        attempt = Path(attempt_directory).absolute()
        _require(attempt.is_dir() and not any(part.is_symlink() for part in (attempt, *attempt.parents)),
                 "existing nonsymlink attempt directory required")
        _require(any(attempt == root or root in attempt.parents for root in self.roots),
                 "attempt outside named disk roots")
        request_id = uuid.uuid4().hex
        retained_bytes = len(payload) + self.output_allowance
        row = dict(protected=protected, demand=demand, payload_bytes=len(payload),
                   retained_bytes=retained_bytes)
        started, admitted = time.monotonic(), False
        daemon, phase = None, None
        native_context, native_entered, daemon_entered = None, False, False
        retained_context = False
        status = "failed"
        with self._condition:
            _require(not self._closed, "pipeline is closed")
            self.parent.remaining()
            _require(not any(attempt == old or attempt in old.parents or old in attempt.parents
                             for old in self._attempts), "phase attempts must be disjoint and never reused")
            same = [item for item in self._queued.values() if item["protected"] == protected]
            _require(len(same) < self._limit(protected, self.parent.budget.maximum_queued_phases,
                                          self.policy.protected_queue_slots), "payload queue slots exhausted")
            _require(sum(item["payload_bytes"] for item in same) + len(payload) <= self._limit(protected,
                self.policy.maximum_queued_payload_bytes, self.policy.protected_queued_payload_bytes),
                "queued payload byte bound exceeded")
            retained = [item for item in (*self._queued.values(), *self._active.values())
                        if item["protected"] == protected]
            _require(sum(item["retained_bytes"] for item in retained) + retained_bytes <= self._limit(protected,
                self.policy.maximum_retained_payload_bytes, self.policy.protected_retained_payload_bytes),
                "retained payload byte bound exceeded")
            _require(self._history[protected] < self._limit(protected, self.parent.budget.maximum_phases,
                self.policy.protected_phase_slots), "protected or ordinary phase history exhausted")
            self._history[protected] += 1
            self._attempts.add(attempt)
            self._queued[request_id] = row
        try:
            with self._condition:
                while True:
                    remaining = self.parent.remaining()
                    first = next(key for key, item in self._queued.items() if item["protected"] == protected)
                    if first == request_id and self._fits(row):
                        self._queued.pop(request_id)
                        self._active[request_id] = row
                        self._disk[request_id] = dict(protected=protected, disk_bytes=demand.disk_bytes)
                        admitted = True
                        break
                    self._condition.wait(min(.05, remaining))
            native_context = self.parent.phase(demand)
            native = native_context.__enter__()
            native_entered = True
            daemon = DaemonResourceReservation(self.ledger_path, roots=self.roots,
                storage_bytes=demand.disk_bytes, memory_mb=demand.memory_mb,
                cpu_slots=demand.cpu_slots, child_process_slots=demand.process_slots,
                timeout_seconds=self.parent.remaining(), ledger_lock_timeout_seconds=min(5, self.parent.remaining()),
                parent_lease=native.native)
            daemon.__enter__()
            daemon_entered = True
            daemon.check_usage(attempt)
            phase = PipelinePhase(self, native, daemon, demand, payload, attempt)
            yield phase
            _require(phase._completed, "phase requires explicit durable finalization")
            status = "completed"
        finally:
            exit_info = sys.exc_info()
            cleanup_error = None
            if phase is not None:
                phase._closed = True
                phase.payload = b""
                phase.last_process_result = None
            if native_entered:
                try:
                    if daemon_entered and self._consumer_leases(daemon):
                        retained_context = True
                    else:
                        if daemon_entered:
                            daemon.__exit__(*exit_info)
                            retained_context = not daemon.to_dict()["resource_lease"]["released"]
                        if not retained_context:
                            native_context.__exit__(*exit_info)
                except BaseException as error:
                    # A failed census/cleanup cannot authorize releasing a live
                    # bridge phase. Keep its context reachable for explicit reap.
                    retained_context = not native.native.released
                    cleanup_error = error
                finally:
                    if retained_context:
                        self._retained_contexts[request_id] = native_context
                        with _PENDING_REAP_LOCK:
                            _PENDING_REAP[self.parent.native.lease_id] = self
            with self._condition:
                self._queued.pop(request_id, None)
                if retained_context:
                    row["payload_bytes"] = row["retained_bytes"] = 0
                else:
                    self._active.pop(request_id, None)
                record = None if daemon is None else daemon.to_dict()
                if admitted:
                    if record is None or record["status"] == "not_entered":
                        self._disk.pop(request_id, None)
                    elif record["status"] == "released":
                        self._disk[request_id]["disk_bytes"] = record["record"]["final_total_charged_bytes"]
                    else:
                        self._retained[daemon.reservation_id] = (request_id, daemon)
                self._events.append(dict(request_id=request_id, phase=demand.phase, protected=protected,
                    status=status, elapsed_ms=int((time.monotonic()-started)*1000),
                    payload_bytes=len(payload), disk=record))
                self._release_lifecycle_if_idle()
                self._condition.notify_all()
            if cleanup_error is not None:
                raise cleanup_error
            if retained_context and exit_info[0] is None:
                raise RepositoryResourceError("live consumer retained; host and disk reservations require explicit reap")

    def _recover(self, reservation_id, *, artifacts_durable):
        with self._condition:
            _require(reservation_id in self._retained, "unknown retained disk reservation")
            request_id, daemon = self._retained[reservation_id]
            _require(not self._consumer_leases(daemon), "native consumers must release before disk recovery")
            result = daemon.release(artifacts_durable=artifacts_durable)
            context = self._retained_contexts.get(request_id)
            if context is not None:
                failure = RepositoryResourceError("unsafe consumer scope explicitly reaped")
                context.__exit__(type(failure), failure, None)
                self._retained_contexts.pop(request_id, None)
                self._active.pop(request_id, None)
            self._disk[request_id]["disk_bytes"] = result["record"]["final_total_charged_bytes"]
            self._retained.pop(reservation_id)
            if not self._retained_contexts:
                if self._closed:
                    self.parent.close()
                if not self._outer_exited:
                    with _PENDING_REAP_LOCK:
                        _PENDING_REAP.pop(self.parent.native.lease_id, None)
                self._release_lifecycle_if_idle()
            self._condition.notify_all()
            return result

    def receipt(self):
        with self._condition:
            return json.loads(json.dumps(dict(schema=SCHEMA, profile=PROFILE, policy=asdict(self.policy),
                lifecycle=pipeline_lifecycle_inventory(), recovery_only=self.recovery_only,
                root=self.parent.receipt(), active_phases=len(self._active), queued_phases=len(self._queued),
                queued_payload_bytes=sum(row["payload_bytes"] for row in self._queued.values()),
                retained_payload_bytes=sum(row["retained_bytes"] for row in (*self._queued.values(), *self._active.values())),
                retained_disk_reservations=sorted(self._retained),
                retained_host_phase_count=len(self._retained_contexts),
                owned_disk_bytes=sum(row["disk_bytes"] for row in self._disk.values()),
                events=self._events, closed=self._closed, limitations=list(LIMITATIONS),
                hard_enforcement=False, task_execution_authority=False, completion_authority=False)))


class RepositoryPipelineResources:
    def __init__(self, supervisor):
        self.bridge = RepositoryResourceBridge(supervisor)

    @contextmanager
    def reserve(self, *, repository_id, workspace, budget, policy, ledger_path, roots, cancel_event=None, recovery_only=False):
        _require(type(policy) is PipelineResourcePolicy, "exact pipeline policy required")
        policy.validate(budget)
        roots = tuple(Path(root).absolute() for root in roots)
        # Canonical constructor validates roots/ledger without reserving capacity.
        DaemonResourceReservation(ledger_path, roots=roots, storage_bytes=budget.disk_bytes,
            memory_mb=budget.memory_mb, cpu_slots=budget.cpu_slots, child_process_slots=budget.process_slots)
        token = _claim_lifecycle(recovery_only)
        pipeline = None
        try:
            with self.bridge.reserve(repository_id=repository_id, workspace=workspace,
                                     budget=budget, cancel_event=cancel_event) as parent:
                pipeline = RepositoryPipelineReservation(parent, policy, ledger_path, roots,
                    lifecycle_token=token, recovery_only=recovery_only)
                try:
                    yield pipeline
                finally:
                    with pipeline._condition:
                        pipeline._closed = True
                    # The enclosing canonical bridge drains and closes its parent.
        finally:
            if pipeline is None:
                with _PENDING_REAP_LOCK:
                    _PIPELINE_LIFECYCLES.pop(token, None)
            else:
                with pipeline._condition:
                    pipeline._outer_exited = True
                    if not pipeline._release_lifecycle_if_idle():
                        with _PENDING_REAP_LOCK:
                            _PENDING_REAP[pipeline.parent.native.lease_id] = pipeline


def pending_pipeline_recoveries():
    """Live local owner handles requiring reap; no model work or disk mutation.

    A subsequent admitted cleanup phase with the same ledger/roots can call
    ``recover_retained(..., owner=handle)``. Cross-process crash recovery remains
    with the existing scheduler and durable daemon ledger owners.
    """
    with _PENDING_REAP_LOCK:
        return tuple(_PENDING_REAP.values())


__all__ = ["PipelineResourcePolicy", "RepositoryPipelineResources", "PipelinePhase",
           "RepositoryPipelineReservation", "pending_pipeline_recoveries", "pipeline_lifecycle_inventory",
           "MAX_PIPELINE_LIFECYCLES", "RECOVERY_PIPELINE_LIFECYCLES", "SCHEMA", "PROFILE"]
