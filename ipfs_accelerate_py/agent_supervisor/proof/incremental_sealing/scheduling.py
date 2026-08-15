"""Proof-work scheduling and resource admission (IPS-033).

Adapts accelerate's existing resource-admission vocabulary to incremental
proof work.  This is not a second scheduler runtime: it builds a deterministic
admission schedule and asks the current resource policy whether each item may
run.  Oversubscribed work waits or returns typed unavailable.  There is no
mock-hardware success path.

Interfaces: ``ProofWorkScheduler``, ``ProofWorkItem``, ``ProofResourcePolicy``,
``build_proof_schedule``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

EVIDENCE_SUBSET: Final[str] = "ips/proof-schedule@1"
RESOURCE_EVIDENCE_SUBSET: Final[str] = "ips/resource-admission@1"

PRIORITY_ORDER: Final[tuple[str, ...]] = (
    "invalidation_check",
    "cache_verification",
    "small_independent",
    "critical_path",
    "expensive_direct",
    "full_fallback",
)
_PRIORITY_INDEX: Final[dict[str, int]] = {
    name: index for index, name in enumerate(PRIORITY_ORDER)
}


class SchedulingError(ValueError):
    """Fail-closed proof-work scheduling contract violation."""


class WorkClass(str, Enum):
    INVALIDATION_CHECK = "invalidation_check"
    CACHE_VERIFICATION = "cache_verification"
    SMALL_INDEPENDENT = "small_independent"
    CRITICAL_PATH = "critical_path"
    EXPENSIVE_DIRECT = "expensive_direct"
    FULL_FALLBACK = "full_fallback"


class AdmissionVerdict(str, Enum):
    ADMITTED = "admitted"
    WAIT = "wait"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ProofResourcePolicy:
    """Closed resource envelope.  GPU is actual capacity, never simulated."""

    max_cpu: int = 4
    max_memory_mb: int = 4096
    max_gpu: int = 0
    max_parallel: int = 4
    max_fan_in: int = 8
    reject_simulated_gpu: bool = True

    def __post_init__(self) -> None:
        for name in ("max_cpu", "max_memory_mb", "max_gpu", "max_parallel", "max_fan_in"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise SchedulingError(f"{name} must be a non-negative int")
        if self.max_parallel < 1:
            raise SchedulingError("max_parallel must be >= 1")
        if self.max_fan_in < 2:
            raise SchedulingError("max_fan_in must be >= 2")


@dataclass(frozen=True, slots=True)
class ProofWorkItem:
    """One schedulable proof-work unit with explicit resource identity."""

    work_id: str
    unit_id: str
    work_class: WorkClass
    cpu: int = 1
    memory_mb: int = 64
    gpu: int = 0
    depends_on: tuple[str, ...] = ()
    publication_order: int = 0
    simulated_gpu: bool = False

    def __post_init__(self) -> None:
        if not self.work_id or not self.unit_id:
            raise SchedulingError("work_id and unit_id are required")
        if type(self.cpu) is not int or self.cpu < 0:
            raise SchedulingError("cpu must be a non-negative int")
        if type(self.memory_mb) is not int or self.memory_mb < 0:
            raise SchedulingError("memory_mb must be a non-negative int")
        if type(self.gpu) is not int or self.gpu < 0:
            raise SchedulingError("gpu must be a non-negative int")
        object.__setattr__(self, "depends_on", tuple(self.depends_on))


@dataclass(frozen=True, slots=True)
class ScheduledWork:
    """Deterministic schedule slot."""

    item: ProofWorkItem
    wave: int
    priority: int
    verdict: AdmissionVerdict = AdmissionVerdict.WAIT

    def to_canonical(self) -> dict[str, Any]:
        return {
            "work_id": self.item.work_id,
            "unit_id": self.item.unit_id,
            "work_class": self.item.work_class.value,
            "wave": self.wave,
            "priority": self.priority,
            "verdict": self.verdict.value,
            "cpu": self.item.cpu,
            "memory_mb": self.item.memory_mb,
            "gpu": self.item.gpu,
        }


def build_proof_schedule(
    items: Sequence[ProofWorkItem],
    policy: ProofResourcePolicy | None = None,
) -> tuple[ScheduledWork, ...]:
    """Return a deterministic priority/dependency schedule (no execution)."""

    del policy
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise SchedulingError("items must be a sequence of ProofWorkItem")
    known = {item.work_id for item in items}
    if len(known) != len(items):
        raise SchedulingError("work_id values must be unique")
    remaining = {item.work_id: item for item in items}
    scheduled: list[ScheduledWork] = []
    satisfied: set[str] = set()
    wave = 0
    while remaining:
        ready = [
            item
            for item in remaining.values()
            if all(dep in satisfied for dep in item.depends_on)
        ]
        if not ready:
            raise SchedulingError("dependency cycle or missing predecessor")
        ready.sort(
            key=lambda item: (
                _PRIORITY_INDEX[item.work_class.value],
                item.publication_order,
                item.work_id,
            )
        )
        for item in ready:
            scheduled.append(
                ScheduledWork(
                    item=item,
                    wave=wave,
                    priority=_PRIORITY_INDEX[item.work_class.value],
                )
            )
            satisfied.add(item.work_id)
            del remaining[item.work_id]
        wave += 1
    return tuple(scheduled)


class ProofWorkScheduler:
    """Admit scheduled proof work under a closed resource policy.

    Independent ready items in the same wave may run up to ``max_parallel``.
    Publication order and explicit dependencies are preserved.  GPU requests
    require actual GPU capacity; simulated GPU is rejected.
    """

    def __init__(self, policy: ProofResourcePolicy | None = None) -> None:
        self.policy = policy or ProofResourcePolicy()
        self._used_cpu = 0
        self._used_memory = 0
        self._used_gpu = 0
        self._in_flight: dict[str, ProofWorkItem] = {}

    def schedule(self, items: Sequence[ProofWorkItem]) -> tuple[ScheduledWork, ...]:
        return build_proof_schedule(items, self.policy)

    def admit(self, item: ProofWorkItem) -> AdmissionVerdict:
        if item.simulated_gpu and self.policy.reject_simulated_gpu:
            return AdmissionVerdict.UNAVAILABLE
        if item.gpu > 0 and self.policy.max_gpu <= 0:
            return AdmissionVerdict.UNAVAILABLE
        if item.cpu > self.policy.max_cpu or item.memory_mb > self.policy.max_memory_mb:
            return AdmissionVerdict.UNAVAILABLE
        if item.gpu > self.policy.max_gpu:
            return AdmissionVerdict.UNAVAILABLE
        if len(self._in_flight) >= self.policy.max_parallel:
            return AdmissionVerdict.WAIT
        if (
            self._used_cpu + item.cpu > self.policy.max_cpu
            or self._used_memory + item.memory_mb > self.policy.max_memory_mb
            or self._used_gpu + item.gpu > self.policy.max_gpu
        ):
            return AdmissionVerdict.WAIT
        self._in_flight[item.work_id] = item
        self._used_cpu += item.cpu
        self._used_memory += item.memory_mb
        self._used_gpu += item.gpu
        return AdmissionVerdict.ADMITTED

    def release(self, work_id: str) -> None:
        item = self._in_flight.pop(work_id, None)
        if item is None:
            return
        self._used_cpu -= item.cpu
        self._used_memory -= item.memory_mb
        self._used_gpu -= item.gpu

    def admit_schedule(
        self, items: Sequence[ProofWorkItem]
    ) -> tuple[ScheduledWork, ...]:
        slots = []
        for slot in self.schedule(items):
            verdict = self.admit(slot.item)
            slots.append(
                ScheduledWork(
                    item=slot.item,
                    wave=slot.wave,
                    priority=slot.priority,
                    verdict=verdict,
                )
            )
        return tuple(slots)

    @property
    def in_flight(self) -> tuple[str, ...]:
        return tuple(sorted(self._in_flight))


__all__ = (
    "EVIDENCE_SUBSET",
    "PRIORITY_ORDER",
    "RESOURCE_EVIDENCE_SUBSET",
    "AdmissionVerdict",
    "ProofResourcePolicy",
    "ProofWorkItem",
    "ProofWorkScheduler",
    "ScheduledWork",
    "SchedulingError",
    "WorkClass",
    "build_proof_schedule",
)
