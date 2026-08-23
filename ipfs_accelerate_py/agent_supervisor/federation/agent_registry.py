"""Logical subagent registry facade and bounded async execution pool.

Registered identities, admitted concurrent operations, worker threads/processes,
and provider calls are deliberately separate populations.  Registering 256
logical agents therefore never creates 256 operating-system processes.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import (
    FederationAuthorityError,
    FederationBoundsError,
    SubagentInstance,
    SubagentOutcome,
)

DEFAULT_REGISTERED_TARGET = 256
DEFAULT_CONCURRENT_TARGET = 64


class LogicalSubagentStore(Protocol):
    def register_subagent(self, instance: SubagentInstance) -> SubagentInstance: ...

    def get_subagent(
        self,
        subagent_id: str,
        *,
        tenant_id: str,
        federation_id: str,
        supervisor_id: str,
    ) -> SubagentInstance | None: ...

    def record_subagent_outcome(self, outcome: SubagentOutcome) -> None: ...

    def reserve_subagent_slot(
        self,
        *,
        subagent_id: str,
        tenant_id: str,
        federation_id: str,
        supervisor_id: str,
        expected_revision: int,
        expected_fencing_epoch: int,
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...

    def release_subagent_slot(
        self,
        *,
        subagent_id: str,
        tenant_id: str,
        federation_id: str,
        supervisor_id: str,
        expected_revision: int,
        expected_fencing_epoch: int,
        idempotency_key: str,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PoolCapacity:
    registered: int
    registered_ceiling: int
    active: int
    active_ceiling: int
    available_slots: int


class BoundedSubagentPool:
    """Async work slots over durable logical identities.

    The caller decides whether a slot invokes local async code, a bounded
    executor, or a provider queue.  This class owns only admission, per-agent
    cancellation, and outcome recording.
    """

    def __init__(
        self,
        store: LogicalSubagentStore,
        *,
        maximum_registered: int = DEFAULT_REGISTERED_TARGET,
        maximum_active: int = DEFAULT_CONCURRENT_TARGET,
    ) -> None:
        if maximum_registered < 1:
            raise FederationBoundsError("maximum_registered must be positive")
        if maximum_active < 1 or maximum_active > maximum_registered:
            raise FederationBoundsError(
                "maximum_active must be positive and no greater than maximum_registered"
            )
        self._store = store
        self._maximum_registered = int(maximum_registered)
        self._maximum_active = int(maximum_active)
        self._registered: set[str] = set()
        self._instances: dict[str, SubagentInstance] = {}
        self._active: set[str] = set()
        self._active_tasks: dict[str, asyncio.Task[Any]] = {}
        self._cancelled: set[str] = set()
        self._semaphore = asyncio.Semaphore(self._maximum_active)
        self._lock = asyncio.Lock()
        self._maximum_observed_active = 0
        self._execution_epochs: dict[str, int] = {}

    @property
    def maximum_observed_active(self) -> int:
        return self._maximum_observed_active

    def register(self, instance: SubagentInstance) -> SubagentInstance:
        if not isinstance(instance, SubagentInstance):
            raise FederationAuthorityError("logical registration requires SubagentInstance")
        if (
            instance.record_id not in self._registered
            and len(self._registered) >= self._maximum_registered
        ):
            raise FederationBoundsError("logical subagent registration ceiling reached")
        admitted = self._store.register_subagent(instance)
        self._registered.add(admitted.record_id)
        self._instances[admitted.record_id] = admitted
        return admitted

    def cancel(self, subagent_id: str) -> None:
        if subagent_id not in self._registered:
            raise FederationAuthorityError("unknown logical subagent")
        self._cancelled.add(subagent_id)
        active = self._active_tasks.get(subagent_id)
        if active is not None and not active.done():
            loop = active.get_loop()
            loop.call_soon_threadsafe(active.cancel)

    def clear_cancel(self, subagent_id: str) -> None:
        self._cancelled.discard(subagent_id)

    def capacity(self) -> PoolCapacity:
        active = len(self._active)
        return PoolCapacity(
            registered=len(self._registered),
            registered_ceiling=self._maximum_registered,
            active=active,
            active_ceiling=self._maximum_active,
            available_slots=max(0, self._maximum_active - active),
        )

    async def execute(
        self,
        *,
        subagent_id: str,
        worker: Callable[[], Any | Awaitable[Any]],
        outcome_factory: Callable[[Any, BaseException | None], SubagentOutcome],
    ) -> SubagentOutcome:
        if subagent_id not in self._registered:
            raise FederationAuthorityError("subagent must be registered before execution")
        if subagent_id in self._cancelled:
            raise asyncio.CancelledError(f"subagent {subagent_id} is cancelled")
        async with self._semaphore:
            active_instance: SubagentInstance
            execution_epoch: int
            async with self._lock:
                if subagent_id in self._active:
                    raise FederationAuthorityError(
                        "one logical subagent cannot own two active execution slots"
                    )
                if subagent_id in self._cancelled:
                    raise asyncio.CancelledError(f"subagent {subagent_id} is cancelled")
                registered = self._instances[subagent_id]
                execution_epoch = self._execution_epochs.get(subagent_id, 0) + 1
                reservation = self._store.reserve_subagent_slot(
                    subagent_id=subagent_id,
                    tenant_id=registered.binding.tenant_id,
                    federation_id=registered.federation_id,
                    supervisor_id=registered.supervisor_id,
                    expected_revision=registered.revision,
                    expected_fencing_epoch=registered.fencing_epoch,
                    idempotency_key=(
                        f"pool-slot-reserve:{subagent_id}:{execution_epoch}"
                    ),
                )
                encoded_instance = reservation.get("subagent")
                if not isinstance(encoded_instance, Mapping):
                    raise FederationAuthorityError(
                        "slot reservation omitted its typed subagent authority"
                    )
                active_instance = SubagentInstance.from_dict(encoded_instance)
                if (
                    active_instance.record_id != registered.record_id
                    or active_instance.binding != registered.binding
                    or active_instance.federation_id != registered.federation_id
                    or active_instance.supervisor_id != registered.supervisor_id
                    or active_instance.task_id != registered.task_id
                    or active_instance.lease_id != registered.lease_id
                    or active_instance.fencing_epoch != registered.fencing_epoch
                    or active_instance.revision != registered.revision + 1
                    or active_instance.state != "ACTIVE"
                ):
                    raise FederationAuthorityError(
                        "slot reservation returned escalated or stale subagent authority"
                    )
                self._execution_epochs[subagent_id] = execution_epoch
                self._instances[subagent_id] = active_instance
                self._active.add(subagent_id)
                current_task = asyncio.current_task()
                if current_task is None:
                    raise FederationAuthorityError("subagent execution requires an asyncio task")
                self._active_tasks[subagent_id] = current_task
                self._maximum_observed_active = max(
                    self._maximum_observed_active, len(self._active)
                )
            try:
                result: Any = None
                error: BaseException | None = None
                try:
                    value = worker()
                    result = await value if inspect.isawaitable(value) else value
                except BaseException as exc:
                    error = exc
                outcome = outcome_factory(result, error)
                if type(outcome) is not SubagentOutcome:
                    raise FederationAuthorityError(
                        "outcome_factory must return an immutable SubagentOutcome"
                    )
                # Re-run the closed wire decoder at the trust boundary.  Frozen
                # dataclasses can still be tampered with via low-level Python APIs,
                # and a subclass could otherwise override authority-bearing
                # behavior after construction.
                decoded_outcome = SubagentOutcome.from_dict(outcome.to_dict())
                if decoded_outcome != outcome:
                    raise FederationAuthorityError(
                        "subagent outcome differs from its closed canonical encoding"
                    )
                outcome = decoded_outcome
                instance = self._store.get_subagent(
                    subagent_id,
                    tenant_id=active_instance.binding.tenant_id,
                    federation_id=active_instance.federation_id,
                    supervisor_id=active_instance.supervisor_id,
                )
                if instance is None:
                    raise FederationAuthorityError(
                        "subagent identity disappeared before outcome"
                    )
                if instance != active_instance or instance.state != "ACTIVE":
                    raise FederationAuthorityError(
                        "subagent execution authority changed before outcome"
                    )
                if (
                    outcome.subagent_id != subagent_id
                    or outcome.federation_id != instance.federation_id
                    or outcome.supervisor_id != instance.supervisor_id
                    or outcome.task_id != instance.task_id
                    or outcome.fencing_epoch != instance.fencing_epoch
                    or outcome.binding != instance.binding
                ):
                    raise FederationAuthorityError(
                        "subagent outcome does not bind the live authority, task scope, and fence"
                    )
                expected_outcome = (
                    "cancelled"
                    if isinstance(error, asyncio.CancelledError)
                    else "failed"
                    if error is not None
                    else "succeeded"
                )
                if outcome.outcome != expected_outcome:
                    raise FederationAuthorityError(
                        "subagent outcome disposition differs from the observed execution result"
                    )
                self._store.record_subagent_outcome(outcome)
                if error is not None:
                    raise error
                return outcome
            finally:
                try:
                    released = self._store.release_subagent_slot(
                        subagent_id=subagent_id,
                        tenant_id=active_instance.binding.tenant_id,
                        federation_id=active_instance.federation_id,
                        supervisor_id=active_instance.supervisor_id,
                        expected_revision=active_instance.revision,
                        expected_fencing_epoch=active_instance.fencing_epoch,
                        idempotency_key=(
                            f"pool-slot-release:{subagent_id}:{execution_epoch}"
                        ),
                    )
                    encoded_released = released.get("subagent")
                    if not isinstance(encoded_released, Mapping):
                        raise FederationAuthorityError(
                            "slot release omitted its typed subagent authority"
                        )
                    admitted = SubagentInstance.from_dict(encoded_released)
                    if (
                        admitted.record_id != active_instance.record_id
                        or admitted.binding != active_instance.binding
                        or admitted.federation_id != active_instance.federation_id
                        or admitted.supervisor_id != active_instance.supervisor_id
                        or admitted.task_id != active_instance.task_id
                        or admitted.lease_id != active_instance.lease_id
                        or admitted.fencing_epoch != active_instance.fencing_epoch
                        or admitted.revision != active_instance.revision + 1
                        or admitted.state != "ADMITTED"
                    ):
                        raise FederationAuthorityError(
                            "slot release returned escalated or stale subagent authority"
                        )
                    self._instances[subagent_id] = admitted
                finally:
                    async with self._lock:
                        self._active.discard(subagent_id)
                        self._active_tasks.pop(subagent_id, None)


def effective_concurrency(
    *,
    policy_ceiling: int,
    host_capacity: int | None,
    provider_capacity: int | None,
    proof_capacity: int | None,
    merge_capacity: int | None,
    storage_capacity: int | None,
) -> int:
    """Return the minimum fresh capacity; missing telemetry grants no slots."""

    observed = (
        host_capacity,
        provider_capacity,
        proof_capacity,
        merge_capacity,
        storage_capacity,
    )
    if any(value is None for value in observed):
        return 0
    values = [int(policy_ceiling), *(int(value) for value in observed if value is not None)]
    if any(value < 0 for value in values):
        raise FederationBoundsError("capacity values must be non-negative")
    return min(values)


__all__ = [
    "BoundedSubagentPool",
    "DEFAULT_CONCURRENT_TARGET",
    "DEFAULT_REGISTERED_TARGET",
    "LogicalSubagentStore",
    "PoolCapacity",
    "effective_concurrency",
]
