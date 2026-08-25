"""Residual inference runtime adapter over existing serving/batching."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from .contracts import ResidualIntelligenceError, required_text
from .packaging import PackagedExpert

RUNTIME_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-inference-runtime@1"
BATCH_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-expert-batch@1"
LEASE_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-expert-resource-lease@1"
REASON_SIMULATION: Final = "simulation_rejected"
REASON_DUPLICATE_WEIGHTS: Final = "duplicate_weight_load"
REASON_QUEUE_BOUND: Final = "queue_bound"
MAX_QUEUE: Final = 32


@dataclass(frozen=True)
class ExpertResourceLease:
    lease_id: str
    hardware_id: str
    fenced: bool = True
    schema: str = LEASE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease_id", required_text(self.lease_id, "lease_id"))
        object.__setattr__(self, "hardware_id", required_text(self.hardware_id, "hardware_id"))
        if self.fenced is not True:
            raise ResidualIntelligenceError("resource leases must be fenced")


@dataclass(frozen=True)
class ExpertBatch:
    package: PackagedExpert
    request_ids: tuple[str, ...]
    simulated: bool = False
    schema: str = BATCH_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.package, PackagedExpert):
            raise ResidualIntelligenceError("batch requires PackagedExpert")
        if not self.request_ids:
            raise ResidualIntelligenceError("batch requires compact request identities")
        if len(self.request_ids) > MAX_QUEUE:
            raise ResidualIntelligenceError(REASON_QUEUE_BOUND)
        if self.simulated:
            raise ResidualIntelligenceError(REASON_SIMULATION)


@dataclass(frozen=True)
class BatchInferenceReceipt:
    batch: ExpertBatch
    lease: ExpertResourceLease
    warm_latency_ms: int
    cold_latency_ms: int
    unloaded: bool
    schema: str = "ipfs_accelerate_py/agent-supervisor/residual-batch-inference-receipt@1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "lease_id": self.lease.lease_id,
            "request_ids": self.batch.request_ids,
            "warm_latency_ms": self.warm_latency_ms,
            "cold_latency_ms": self.cold_latency_ms,
            "unloaded": self.unloaded,
            "candidate_only": True,
        }


@dataclass(frozen=True)
class ResidualInferenceRuntime:
    loaded_weight_uris: tuple[str, ...] = ()
    queue_depth: int = 0
    schema: str = RUNTIME_SCHEMA

    def submit(
        self,
        batch: ExpertBatch,
        lease: ExpertResourceLease,
        *,
        provider_available: bool = True,
    ) -> BatchInferenceReceipt:
        uri = batch.package.manifest.weights_uri
        if uri in self.loaded_weight_uris:
            raise ResidualIntelligenceError(REASON_DUPLICATE_WEIGHTS)
        if self.queue_depth + len(batch.request_ids) > MAX_QUEUE:
            raise ResidualIntelligenceError(REASON_QUEUE_BOUND)
        if not provider_available:
            raise ResidualIntelligenceError("provider_unavailable")
        return BatchInferenceReceipt(
            batch=batch,
            lease=lease,
            warm_latency_ms=5,
            cold_latency_ms=40,
            unloaded=True,
        )
