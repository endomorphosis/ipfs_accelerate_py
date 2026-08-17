"""Neutral non-authoritative provider-capacity observation contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

NON_AUTHORITATIVE_CAPACITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/non-authoritative-provider-capacity@1"
)


@dataclass(frozen=True)
class NonAuthoritativeProviderCapacityObservation:
    """Bounded capacity observation that never authorizes dispatch alone."""

    schema: str
    observed_at_ms: int
    available_worker_capacity: int
    worker_limit: int
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema != NON_AUTHORITATIVE_CAPACITY_SCHEMA:
            raise ValueError("unsupported capacity observation schema")
        if self.available_worker_capacity < 0 or self.worker_limit < 0:
            raise ValueError("capacity counts must be non-negative")
        object.__setattr__(self, "details", dict(self.details))


@runtime_checkable
class ProviderAttemptStoreService(Protocol):
    """Port for CAS reservation effects owned by control.provider_attempt_store."""

    def reserve(self, *args: Any, **kwargs: Any) -> Any: ...

    def adopt(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class PlanExecutionStore(Protocol):
    """Port for plan-bound store effects owned by control.plan_execution_store."""

    def load_execution_lease(self, *args: Any, **kwargs: Any) -> Any: ...
