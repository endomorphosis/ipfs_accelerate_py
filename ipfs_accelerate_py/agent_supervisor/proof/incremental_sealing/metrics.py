"""Measured versus estimated incremental-seal costs (IPS-037).

Absent counters stay ``unknown``.  Zeros are recorded only when the collector
observed a genuine zero.  Estimates never become measurements.  Compute saved
compares equivalent required work and stays unknown when either side is
missing or the run failed/fell back.

Interfaces: ``ProofCostRecord``, ``ProofCostComparison``,
``ProofMetricsCollector``, ``compare_costs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

COST_EVIDENCE: Final[str] = "ips/proof-cost@1"
COST_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "proof-cost-record@1"
)
COST_COMPARISON_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "proof-cost-comparison@1"
)

UNKNOWN: Final[str] = "unknown"


class CostProvenance(str, Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


class CostKind(str, Enum):
    WALL = "wall"
    CPU = "cpu"
    GPU = "gpu"


class RunDisposition(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class CostValue:
    """One numeric cost with explicit provenance and unit."""

    kind: CostKind
    unit: str
    value: int | None
    provenance: CostProvenance

    def __post_init__(self) -> None:
        if self.value is None and self.provenance is not CostProvenance.UNKNOWN:
            raise MetricsError("missing values must be unknown, not estimated or measured")
        if self.provenance is CostProvenance.UNKNOWN and self.value is not None:
            raise MetricsError("unknown costs cannot carry a numeric value")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "unit": self.unit,
            "value": UNKNOWN if self.value is None else self.value,
            "provenance": self.provenance.value,
        }


class MetricsError(ValueError):
    """Fail-closed cost-accounting contract violation."""


@dataclass(frozen=True, slots=True)
class ProofCostRecord:
    """Closed cost record for one incremental or full run."""

    schema: str
    evidence_subset: str
    required_units: int
    reused_units: int
    invalidated_units: int
    proved_units: int
    cache_hits: int
    leaf_time_ms: CostValue
    aggregate_time_ms: CostValue
    verify_time_ms: CostValue
    wall_time_ms: CostValue
    cpu_time_ms: CostValue
    gpu_time_ms: CostValue
    peak_memory_bytes: CostValue
    proof_size_bytes: CostValue
    seal_size_bytes: CostValue
    storage_growth_bytes: CostValue
    disposition: RunDisposition
    fallback_reason: str | None
    estimated: bool

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "required_units": self.required_units,
            "reused_units": self.reused_units,
            "invalidated_units": self.invalidated_units,
            "proved_units": self.proved_units,
            "cache_hits": self.cache_hits,
            "leaf_time_ms": self.leaf_time_ms.to_canonical(),
            "aggregate_time_ms": self.aggregate_time_ms.to_canonical(),
            "verify_time_ms": self.verify_time_ms.to_canonical(),
            "wall_time_ms": self.wall_time_ms.to_canonical(),
            "cpu_time_ms": self.cpu_time_ms.to_canonical(),
            "gpu_time_ms": self.gpu_time_ms.to_canonical(),
            "peak_memory_bytes": self.peak_memory_bytes.to_canonical(),
            "proof_size_bytes": self.proof_size_bytes.to_canonical(),
            "seal_size_bytes": self.seal_size_bytes.to_canonical(),
            "storage_growth_bytes": self.storage_growth_bytes.to_canonical(),
            "disposition": self.disposition.value,
            "fallback_reason": self.fallback_reason,
            "estimated": self.estimated,
        }


@dataclass(frozen=True, slots=True)
class ProofCostComparison:
    """Equivalent-work comparison between full and incremental measured runs."""

    schema: str
    evidence_subset: str
    full: ProofCostRecord
    incremental: ProofCostRecord
    compute_saved_cpu_ms: int | None
    compute_saved_wall_ms: int | None
    storage_saved_bytes: int | None
    savings_provenance: CostProvenance
    visible_failure: bool

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "full": self.full.to_canonical(),
            "incremental": self.incremental.to_canonical(),
            "compute_saved_cpu_ms": (
                UNKNOWN if self.compute_saved_cpu_ms is None else self.compute_saved_cpu_ms
            ),
            "compute_saved_wall_ms": (
                UNKNOWN if self.compute_saved_wall_ms is None else self.compute_saved_wall_ms
            ),
            "storage_saved_bytes": (
                UNKNOWN if self.storage_saved_bytes is None else self.storage_saved_bytes
            ),
            "savings_provenance": self.savings_provenance.value,
            "visible_failure": self.visible_failure,
            "estimated_as_measured": False,
        }


def _value(
    kind: CostKind,
    unit: str,
    raw: int | None,
    *,
    observed: bool,
    estimated: bool,
) -> CostValue:
    if not observed:
        return CostValue(kind, unit, None, CostProvenance.UNKNOWN)
    if estimated:
        return CostValue(kind, unit, raw, CostProvenance.ESTIMATED)
    if raw is None:
        return CostValue(kind, unit, None, CostProvenance.UNKNOWN)
    return CostValue(kind, unit, raw, CostProvenance.MEASURED)


class ProofMetricsCollector:
    """Accumulate observed counters.  Unobserved fields stay unknown."""

    def __init__(self, *, estimated: bool = False) -> None:
        self.estimated = estimated
        self.required_units = 0
        self.reused_units = 0
        self.invalidated_units = 0
        self.proved_units = 0
        self.cache_hits = 0
        self._leaf_ms: int | None = None
        self._aggregate_ms: int | None = None
        self._verify_ms: int | None = None
        self._wall_ms: int | None = None
        self._cpu_ms: int | None = None
        self._gpu_ms: int | None = None
        self._peak_memory: int | None = None
        self._proof_size: int | None = None
        self._seal_size: int | None = None
        self._storage_growth: int | None = None
        self._saw_leaf = False
        self._saw_aggregate = False
        self._saw_verify = False
        self._saw_wall = False
        self._saw_cpu = False
        self._saw_gpu = False
        self._saw_memory = False
        self._saw_proof_size = False
        self._saw_seal_size = False
        self._saw_storage = False
        self.disposition = RunDisposition.COMPLETED
        self.fallback_reason: str | None = None

    def record_units(
        self,
        *,
        required: int,
        reused: int,
        invalidated: int,
        proved: int,
        cache_hits: int,
    ) -> None:
        self.required_units = required
        self.reused_units = reused
        self.invalidated_units = invalidated
        self.proved_units = proved
        self.cache_hits = cache_hits

    def observe_leaf_ms(self, value: int) -> None:
        self._leaf_ms = value
        self._saw_leaf = True

    def observe_aggregate_ms(self, value: int) -> None:
        self._aggregate_ms = value
        self._saw_aggregate = True

    def observe_verify_ms(self, value: int) -> None:
        self._verify_ms = value
        self._saw_verify = True

    def observe_wall_ms(self, value: int) -> None:
        self._wall_ms = value
        self._saw_wall = True

    def observe_cpu_ms(self, value: int) -> None:
        self._cpu_ms = value
        self._saw_cpu = True

    def observe_gpu_ms(self, value: int) -> None:
        self._gpu_ms = value
        self._saw_gpu = True

    def observe_peak_memory_bytes(self, value: int) -> None:
        self._peak_memory = value
        self._saw_memory = True

    def observe_proof_size_bytes(self, value: int) -> None:
        self._proof_size = value
        self._saw_proof_size = True

    def observe_seal_size_bytes(self, value: int) -> None:
        self._seal_size = value
        self._saw_seal_size = True

    def observe_storage_growth_bytes(self, value: int) -> None:
        self._storage_growth = value
        self._saw_storage = True

    def mark_failed(self, reason: str = "failed") -> None:
        self.disposition = RunDisposition.FAILED
        self.fallback_reason = reason

    def mark_fallback(self, reason: str) -> None:
        self.disposition = RunDisposition.FALLBACK
        self.fallback_reason = reason

    def snapshot(self) -> ProofCostRecord:
        est = self.estimated
        return ProofCostRecord(
            schema=COST_RECORD_SCHEMA,
            evidence_subset=COST_EVIDENCE,
            required_units=self.required_units,
            reused_units=self.reused_units,
            invalidated_units=self.invalidated_units,
            proved_units=self.proved_units,
            cache_hits=self.cache_hits,
            leaf_time_ms=_value(CostKind.WALL, "ms", self._leaf_ms, observed=self._saw_leaf, estimated=est),
            aggregate_time_ms=_value(
                CostKind.WALL, "ms", self._aggregate_ms, observed=self._saw_aggregate, estimated=est
            ),
            verify_time_ms=_value(
                CostKind.WALL, "ms", self._verify_ms, observed=self._saw_verify, estimated=est
            ),
            wall_time_ms=_value(CostKind.WALL, "ms", self._wall_ms, observed=self._saw_wall, estimated=est),
            cpu_time_ms=_value(CostKind.CPU, "ms", self._cpu_ms, observed=self._saw_cpu, estimated=est),
            gpu_time_ms=_value(CostKind.GPU, "ms", self._gpu_ms, observed=self._saw_gpu, estimated=est),
            peak_memory_bytes=_value(
                CostKind.WALL, "bytes", self._peak_memory, observed=self._saw_memory, estimated=est
            ),
            proof_size_bytes=_value(
                CostKind.WALL, "bytes", self._proof_size, observed=self._saw_proof_size, estimated=est
            ),
            seal_size_bytes=_value(
                CostKind.WALL, "bytes", self._seal_size, observed=self._saw_seal_size, estimated=est
            ),
            storage_growth_bytes=_value(
                CostKind.WALL, "bytes", self._storage_growth, observed=self._saw_storage, estimated=est
            ),
            disposition=self.disposition,
            fallback_reason=self.fallback_reason,
            estimated=est,
        )


def _saved(full: CostValue, incremental: CostValue) -> int | None:
    if (
        full.provenance is not CostProvenance.MEASURED
        or incremental.provenance is not CostProvenance.MEASURED
        or full.value is None
        or incremental.value is None
    ):
        return None
    return max(0, full.value - incremental.value)


def compare_costs(
    full: ProofCostRecord,
    incremental: ProofCostRecord,
) -> ProofCostComparison:
    """Compare equivalent required work.  Estimates never count as savings."""

    if not isinstance(full, ProofCostRecord) or not isinstance(incremental, ProofCostRecord):
        raise MetricsError("compare_costs requires ProofCostRecord arguments")
    failed = (
        full.disposition is not RunDisposition.COMPLETED
        or incremental.disposition is not RunDisposition.COMPLETED
        or full.estimated
        or incremental.estimated
    )
    cpu = None if failed else _saved(full.cpu_time_ms, incremental.cpu_time_ms)
    wall = None if failed else _saved(full.wall_time_ms, incremental.wall_time_ms)
    storage = None if failed else _saved(full.storage_growth_bytes, incremental.storage_growth_bytes)
    if failed or cpu is None or wall is None or storage is None:
        provenance = CostProvenance.UNKNOWN
        cpu = wall = storage = None
    else:
        provenance = CostProvenance.MEASURED
    return ProofCostComparison(
        schema=COST_COMPARISON_SCHEMA,
        evidence_subset=COST_EVIDENCE,
        full=full,
        incremental=incremental,
        compute_saved_cpu_ms=cpu,
        compute_saved_wall_ms=wall,
        storage_saved_bytes=storage,
        savings_provenance=provenance,
        visible_failure=failed,
    )


__all__ = (
    "COST_EVIDENCE",
    "UNKNOWN",
    "CostKind",
    "CostProvenance",
    "CostValue",
    "MetricsError",
    "ProofCostComparison",
    "ProofCostRecord",
    "ProofMetricsCollector",
    "RunDisposition",
    "compare_costs",
)
