"""Task-sized native bundles with datasets admission enabled by default.

The plan describes declared costs; only a live datasets lease grants execution.
Native adapters still enforce their own process limits and cleanup contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import time

from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceLease, ResourceLane, get_global_resource_scheduler,
)
from .datasets_prover_resources import _seconds, open_datasets_prover_lease
from .multi_prover_resources import (
    BundleExecutionReceipt, BundleProverSupervisor, MultiProverResourceBudget,
    ProverResourceRequest, ProverTask,
)

_MIB = 1024 * 1024
SCHEMA = "datasets-native-bundle-plan@1"


@dataclass(frozen=True)
class DatasetsNativeBundlePlan:
    task_ids: tuple[str, ...]
    normalized_requests: tuple[ProverResourceRequest, ...]
    budget: MultiProverResourceBudget
    requested_max_workers: int
    cpu_capacity: int
    memory_capacity_mb: int
    process_capacity: int
    default_child_memory_mb: int
    proof_safety_enabled: bool

    def to_dict(self):
        return {"schema": SCHEMA, "task_ids": list(self.task_ids),
                "normalized_requests": [request.to_dict() for request in self.normalized_requests],
                "budget": self.budget.to_dict(), "requested_max_workers": self.requested_max_workers,
                "planned_max_workers": self.budget.max_portfolio_width,
                "cpu_capacity": self.cpu_capacity, "memory_capacity_mb": self.memory_capacity_mb,
                "process_capacity": self.process_capacity, "default_child_memory_mb": self.default_child_memory_mb,
                "proof_safety_enabled": self.proof_safety_enabled, "resource_authority": False}


@dataclass(frozen=True)
class DatasetsNativeBundleExecution:
    plan: DatasetsNativeBundlePlan
    receipt: BundleExecutionReceipt

    def to_dict(self):
        return {"schema": "datasets-native-bundle-execution@1", "plan": self.plan.to_dict(),
                "receipt": self.receipt.to_dict(), "proof_composition_authority": False}


def _owner(scheduler, parent_lease):
    if parent_lease is not None and not isinstance(parent_lease, ResourceLease):
        raise TypeError("parent_lease must be an actual datasets ResourceLease")
    if scheduler is not None and not isinstance(scheduler, GlobalResourceScheduler):
        raise TypeError("scheduler must be a datasets GlobalResourceScheduler")
    if parent_lease is not None:
        if scheduler is not None and scheduler is not parent_lease._scheduler:
            raise ValueError("scheduler must be the parent's actual owner")
        return parent_lease._scheduler
    return scheduler if scheduler is not None else get_global_resource_scheduler()


def _tasks(tasks):
    if type(tasks) not in (list, tuple) or not 1 <= len(tasks) <= 64:
        raise ValueError("tasks must be a nonempty list or tuple of at most 64 tasks")
    result = tuple(tasks)
    if any(not isinstance(task, ProverTask) for task in result):
        raise TypeError("tasks must contain ProverTask objects")
    if len({task.task_id for task in result}) != len(result):
        raise ValueError("bundle task ids must be unique")
    return result


def plan_datasets_native_bundle(
    tasks, *, max_workers: int = 4, scheduler: GlobalResourceScheduler | None = None,
    parent_lease: ResourceLease | None = None, default_child_memory_mb: int = 512,
) -> DatasetsNativeBundlePlan:
    """Bound one envelope using declared costs and configured host/parent limits.

    For each dimension, reserve the sum of the largest ``width`` requests.
    Reduce width until every possible ready subset of that size fits. This is
    conservative for dependency-constrained populations, not an optimal packing
    promise. Live pressure and other owners are checked by actual admission.
    Provider/model work needs explicitly configured accounting and is rejected.
    """
    tasks = _tasks(tasks)
    if type(max_workers) is not int or not 1 <= max_workers <= 32:
        raise ValueError("max_workers must be an exact integer in [1, 32]")
    if type(default_child_memory_mb) is not int or default_child_memory_mb <= 0:
        raise ValueError("default_child_memory_mb must be a positive integer")
    owner = _owner(scheduler, parent_lease)
    if parent_lease is not None:
        cpu, memory, processes = parent_lease.cpu_slots, parent_lease.memory_mb, parent_lease.child_process_slots
    else:
        config = owner.config
        protected = [value for lane, value in config.reservations().items()
                     if lane != ResourceLane.ORCHESTRATION.value]
        cpu = config.total_cpu_slots - sum(value.cpu_slots for value in protected)
        memory = config.total_memory_mb - config.reserved_memory_mb - sum(value.memory_mb for value in protected)
        processes = config.total_child_process_slots
    requests = []
    for task in tasks:
        request = task.resources
        if request.model_slots or request.provider_id or request.provider_quota:
            raise ValueError(f"{task.task_id}: provider/model resources require explicit owner configuration")
        # Exactly match the existing datasets bridge's normalization.
        ram = request.memory_bytes or default_child_memory_mb * _MIB
        request = replace(request, cpu_slots=max(1, request.cpu_slots, request.thread_slots),
                          memory_bytes=((ram + _MIB - 1) // _MIB) * _MIB)
        if request.cpu_slots > cpu or request.memory_bytes > memory * _MIB or request.process_slots > processes:
            raise ValueError(f"{task.task_id}: declared native task cannot fit the enclosing capacity")
        requests.append(request)
    if min(cpu, memory, processes) <= 0:
        raise ValueError("native bundle requires positive enclosing CPU, memory and process capacity")

    def largest(name, width):
        return sum(sorted((getattr(request, name) for request in requests), reverse=True)[:width])

    for width in range(min(max_workers, len(tasks)), 0, -1):
        cpu_need = largest("cpu_slots", width)
        memory_need = largest("memory_bytes", width)
        process_need = max(1, largest("process_slots", width))
        if cpu_need > cpu or memory_need > memory * _MIB or process_need > processes:
            continue
        budget = MultiProverResourceBudget(cpu_slots=cpu_need, memory_bytes=memory_need,
            process_slots=process_need, thread_slots=max(width, largest("thread_slots", width)),
            disk_bytes=largest("disk_bytes", width), artifact_concurrency=max(1, largest("artifact_slots", width)),
            max_portfolio_width=width)
        return DatasetsNativeBundlePlan(tuple(task.task_id for task in tasks), tuple(requests), budget,
            max_workers, cpu, memory, processes, default_child_memory_mb, owner.config.proof_safety_enabled)
    raise ValueError("no bounded native bundle envelope fits")  # defensive: individual costs were checked above


def execute_datasets_native_bundle(
    tasks, *, max_workers: int = 4, scheduler: GlobalResourceScheduler | None = None,
    parent_lease: ResourceLease | None = None, cancel_event=None,
    admission_timeout_seconds: float = 30, timeout_seconds: float = 120,
    default_child_memory_mb: int = 512,
) -> DatasetsNativeBundleExecution:
    """Plan, admit and execute native tasks under one safe owner by default.

    The overall deadline includes planning and admission. Result order follows
    input order. No supplied plan or completed-task assertion can bypass fresh
    admission; result authority remains the responsibility of each adapter.
    An injected scheduler retains its policy, as in the lower-level bridge.
    """
    timeout = _seconds(timeout_seconds, "timeout_seconds")
    admission = _seconds(admission_timeout_seconds, "admission_timeout_seconds")
    started = time.monotonic()
    tasks = _tasks(tasks)
    owner = _owner(scheduler, parent_lease)
    plan = plan_datasets_native_bundle(tasks, max_workers=max_workers, scheduler=owner,
        parent_lease=parent_lease, default_child_memory_mb=default_child_memory_mb)
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("native bundle deadline elapsed during planning")
    with open_datasets_prover_lease(budget=plan.budget, scheduler=owner, parent_lease=parent_lease,
            cancel_event=cancel_event, timeout_seconds=remaining, admission_timeout_seconds=admission,
            default_child_memory_mb=default_child_memory_mb) as lease:
        receipt = BundleProverSupervisor(lease).execute(tasks, cancellation=cancel_event)
    return DatasetsNativeBundleExecution(plan, receipt)


__all__ = ["DatasetsNativeBundlePlan", "DatasetsNativeBundleExecution",
           "plan_datasets_native_bundle", "execute_datasets_native_bundle"]
