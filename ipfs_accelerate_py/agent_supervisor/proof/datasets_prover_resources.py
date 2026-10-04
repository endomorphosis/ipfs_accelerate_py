"""Supervisor subdivisions of the datasets process-safe proof resource owner.

Open this explicit bridge to share one host reservation with datasets solvers.
Native leases stay charged through asynchronous callable cleanup. This is
admission accounting, not a thread kill or a physical RSS limit: adapters must
bound native processes and join their work before returning. A permanently
uncooperative callable retains capacity indefinitely.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import replace
from typing import Any

from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceLease, ResourceLane, ResourceSchedulerError,
    LeaseCancelledError, get_global_resource_scheduler,
)
from .multi_prover_resources import (
    ChildResourceLease, MultiProverResourceBudget, MultiProverResourceLease,
    ProverResourceRequest, ResourceAdmission,
)

_MIB = 1024 * 1024


def _seconds(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


class _Cancellation:
    """Event-compatible cancellation combining the native owner and callers."""
    def __init__(self, native: ResourceLease, external: Any = None) -> None:
        self._event = threading.Event()
        self._native = native.combined_cancellation_signal(external)

    def set(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        if self._event.is_set():
            return True
        try:
            return self._native.is_set()
        except ResourceSchedulerError:
            # An unreadable authority cannot grant continued execution.
            self.set()
            return True

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0, timeout)
        while not self.is_set():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            self._event.wait(.02 if remaining is None else min(.02, remaining))
        return True


class _AdmissionCancellation:
    def __init__(self, *signals):
        self.signals = tuple(signal for signal in signals if signal is not None)

    def is_set(self):
        return any(signal.is_set() for signal in self.signals)


class DatasetsChildResourceLease(ChildResourceLease):
    """A supervisor child with the actual native parent for nested adapters."""
    def __init__(self, owner, lease_id, request):
        super().__init__(owner, lease_id, request)
        self._native: ResourceLease | None = None
        self._cancel_propagated = False
        # Admission is live work: close must not reclaim its local reservation
        # between local admission and assignment of the native child.
        self._execution_holds = 1

    @property
    def datasets_parent_lease(self) -> ResourceLease:
        if self._native is None:
            raise RuntimeError("native resource admission has not completed")
        return self._native

    @property
    def datasets_scheduler(self) -> GlobalResourceScheduler:
        return self.owner.datasets_scheduler

    @property
    def cancelled(self) -> bool:
        return self.owner.cancellation.is_set() or (self._native is not None and self._native.cancelled)

    def cancel(self) -> None:
        self._cancel_propagated = True
        native = self._native
        if native is not None:
            native.cancel()
        super().cancel()


class DatasetsProverResourceLease(MultiProverResourceLease):
    """One native envelope with supervisor-local subdivisions and live backoff.

    Use :func:`open_datasets_prover_lease` to acquire the envelope. Closing
    requests cancellation immediately; the envelope is released after every
    admitted child and its execution holds have drained.
    """
    def __init__(self, budget, *, native_lease, scheduler, cancel_event,
                 admission_timeout_seconds, deadline, default_child_memory_mb):
        super().__init__(budget)
        self.datasets_parent_lease = native_lease
        self.datasets_scheduler = scheduler
        self._admission_timeout_seconds = admission_timeout_seconds
        self._default_child_memory_mb = default_child_memory_mb
        self._deadline_monotonic = deadline
        self.cancellation = _Cancellation(native_lease, cancel_event)
        self._native_released = False
        self._watch_stop = threading.Event()
        self._watcher = threading.Thread(target=self._watch_cancellation,
            name="datasets-supervisor-cancellation", daemon=True)
        try:
            self._watcher.start()
        except BaseException:
            native_lease.release()
            raise

    def _watch_cancellation(self):
        while not self._watch_stop.wait(.02):
            if time.monotonic() >= self._deadline_monotonic:
                self.close()
                return
            if self.cancellation.is_set():
                self.cancel()
                return
            for child in self.active_children:
                if not child._cancel_propagated and child.cancelled:
                    child.cancel()

    def _new_child(self, child_id, request):
        return DatasetsChildResourceLease(self, child_id, request)

    def try_acquire(self, request):
        return self.acquire_for_execution(request, cancellation=None)

    def normalized_request(self, request):
        if not isinstance(request, ProverResourceRequest):
            raise TypeError("request must be ProverResourceRequest")
        memory = request.memory_bytes or self._default_child_memory_mb * _MIB
        return replace(request, cpu_slots=max(1, request.cpu_slots, request.thread_slots),
                       memory_bytes=((memory + _MIB - 1) // _MIB) * _MIB)

    def acquire_for_execution(self, request, *, cancellation, deadline_monotonic=None):
        request = self.normalized_request(request)
        signal = _AdmissionCancellation(self.cancellation, cancellation)
        if signal.is_set():
            return ResourceAdmission(False, request.task_id, ("datasets_admission_cancelled",), self.lease_id), None
        deadline = min(self._deadline_monotonic, time.monotonic() + self._admission_timeout_seconds)
        if deadline_monotonic is not None:
            deadline = min(deadline, deadline_monotonic)
        # Every native admission requires a CPU and finite positive RAM.
        admission, child = super().try_acquire(request)
        if child is None:
            return admission, None
        try:
            remaining = max(0, deadline - time.monotonic())
            native = self.datasets_parent_lease.acquire_child(
                lane=ResourceLane.HAMMER, cpu_slots=request.cpu_slots,
                memory_mb=request.memory_bytes // _MIB,
                child_process_slots=request.process_slots,
                timeout=remaining,
                cancel_event=signal, request_id=request.task_id,
            )
            child._native = native
            if signal.is_set() or self.closed:
                child.release()
                return ResourceAdmission(False, request.task_id,
                    ("datasets_admission_cancelled",), self.lease_id), None
            if time.monotonic() >= deadline:
                child.release()
                return ResourceAdmission(False, request.task_id,
                    ("task_deadline" if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic
                     else "datasets_admission_timeout",), self.lease_id), None
            return admission, child
        except ResourceSchedulerError as exc:
            child.release()
            reason = ("datasets_admission_cancelled" if isinstance(exc, LeaseCancelledError)
                      else "task_deadline" if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic
                      else f"datasets_admission_{type(exc).__name__}")
            return ResourceAdmission(False, request.task_id, (reason,), self.lease_id), None
        except BaseException:
            child.release()
            raise
        finally:
            child._finish_execution()

    acquire = try_acquire

    def _release(self, child):
        # Called only once all supervisor execution holds have stopped.
        if child._native is not None:
            child._native.release()
        result = super()._release(child)
        self._release_envelope_if_drained()
        return result

    def _release_envelope_if_drained(self):
        with self._condition:
            if self._closed and not self._children and not self._native_released:
                self.datasets_parent_lease.release()
                self._native_released = True
                self._watch_stop.set()

    def cancel(self):
        self.cancellation.set()
        self.datasets_parent_lease.cancel()
        super().cancel()

    def close(self):
        super().close()
        self._release_envelope_if_drained()


def open_datasets_prover_lease(
    *, budget: MultiProverResourceBudget | None = None,
    scheduler: GlobalResourceScheduler | None = None,
    parent_lease: ResourceLease | None = None,
    cancel_event: Any = None,
    admission_timeout_seconds: float = 30,
    timeout_seconds: float = 120,
    default_child_memory_mb: int = 512,
) -> DatasetsProverResourceLease:
    """Acquire a shared root or one child of an existing native authority.

    Omitted budgets choose at most four workers, each with positive memory.
    Explicit budgets require positive memory and CPU-backed thread capacity;
    memory is rounded up to MiB. The total deadline includes root admission.
    A supplied scheduler must be the actual parent's owner object. Native
    authority tokens and local source paths are never serialized by this bridge.
    """
    admission = _seconds(admission_timeout_seconds, "admission_timeout_seconds")
    timeout = _seconds(timeout_seconds, "timeout_seconds")
    if type(default_child_memory_mb) is not int or default_child_memory_mb <= 0:
        raise ValueError("default_child_memory_mb must be a positive integer")
    if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
        raise TypeError("cancel_event must support is_set")
    if parent_lease is not None and not isinstance(parent_lease, ResourceLease):
        raise TypeError("parent_lease must be an actual datasets ResourceLease")
    if scheduler is not None and not isinstance(scheduler, GlobalResourceScheduler):
        raise TypeError("scheduler must be a datasets GlobalResourceScheduler")
    if parent_lease is not None:
        if scheduler is not None and scheduler is not parent_lease._scheduler:
            raise ValueError("scheduler must be the parent's actual owner")
        scheduler = parent_lease._scheduler
    if scheduler is None:
        scheduler = get_global_resource_scheduler()
    if budget is not None and not isinstance(budget, MultiProverResourceBudget):
        raise TypeError("budget must be MultiProverResourceBudget")
    if budget is None:
        if parent_lease is not None:
            cpu, memory, processes = parent_lease.cpu_slots, parent_lease.memory_mb, parent_lease.child_process_slots
        else:
            config = scheduler.config
            other = [value for lane, value in config.reservations().items()
                     if lane != ResourceLane.ORCHESTRATION.value]
            cpu = config.total_cpu_slots - sum(value.cpu_slots for value in other)
            memory = config.total_memory_mb - config.reserved_memory_mb - sum(value.memory_mb for value in other)
            processes = config.total_child_process_slots
        width = min(4, cpu, processes, memory // default_child_memory_mb)
        if width < 1:
            raise ValueError("native authority cannot fit the default proof worker")
        budget = MultiProverResourceBudget(cpu_slots=width, process_slots=width,
            thread_slots=width, memory_bytes=width * default_child_memory_mb * _MIB,
            max_portfolio_width=width)
    if budget.memory_bytes <= 0 or budget.thread_slots > budget.cpu_slots:
        raise ValueError("bridge budget requires positive RAM and threads no greater than CPU slots")
    memory_mb = (budget.memory_bytes + _MIB - 1) // _MIB
    budget = replace(budget, memory_bytes=memory_mb * _MIB)
    if budget.wall_time_ms:
        timeout = min(timeout, budget.wall_time_ms / 1000)
    deadline = time.monotonic() + timeout
    native = scheduler.acquire(ResourceLane.ORCHESTRATION,
        cpu_slots=budget.cpu_slots, memory_mb=memory_mb,
        child_process_slots=budget.process_slots, parent_lease=parent_lease,
        timeout=min(admission, timeout), cancel_event=cancel_event)
    try:
        if time.monotonic() >= deadline:
            raise TimeoutError("supervisor bridge deadline elapsed during admission")
        return DatasetsProverResourceLease(budget, native_lease=native, scheduler=scheduler,
            cancel_event=cancel_event, admission_timeout_seconds=admission,
            deadline=deadline, default_child_memory_mb=default_child_memory_mb)
    except BaseException:
        native.release()
        raise


__all__ = ["DatasetsChildResourceLease", "DatasetsProverResourceLease", "open_datasets_prover_lease"]
