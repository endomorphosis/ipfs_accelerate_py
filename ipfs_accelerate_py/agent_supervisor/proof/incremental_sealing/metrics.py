"""Measure proving, aggregation, verification, storage, and savings cost (IPS-037).

Accelerate owns the metrics surface for incremental sealing.  Every numeric
field carries an explicit ``measured`` / ``estimated`` / ``unavailable``
provenance.  Estimates never masquerade as measurements, absent counters stay
unknown (never fabricated zero savings), and failed or full-fallback runs
remain inspectable.

Compute-saved compares equivalent required work: the full-path cost of the
same required unit set against the incremental path actually executed.
Simulated units are excluded from production proving-compute claims.

Interfaces: ``ProofCostRecord``, ``ProofCostComparison``,
``ProofMetricsCollector``, ``compare_costs``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

EVIDENCE_SUBSET: Final[str] = "ips/proof-cost@1"
COST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "proof-cost-record@1"
)
COMPARISON_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "proof-cost-comparison@1"
)

# Integer percentage scale: 10_000 basis points == 100%.
BASIS_POINTS: Final[int] = 10_000
MAX_NON_NEGATIVE: Final[int] = 10**15

# Default deterministic cost model (ms / bytes per unit) used only for
# estimated comparisons when measured counters are unavailable.
DEFAULT_LEAF_CPU_MS: Final[int] = 1_000
DEFAULT_AGGREGATE_CPU_MS: Final[int] = 50
DEFAULT_VERIFY_CPU_MS: Final[int] = 20
DEFAULT_LEAF_STORAGE_BYTES: Final[int] = 4_096
DEFAULT_REUSE_STORAGE_BYTES: Final[int] = 256


class MetricsError(ValueError):
    """Fail-closed proof-cost metrics contract violation."""


class MetricProvenance(str, Enum):
    """Source of a numeric observation.  Never coerce estimated to measured."""

    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


class RunVisibility(str, Enum):
    """Every run remains visible; failures and fallbacks are first-class."""

    SUCCESS = "success"
    FAILED = "failed"
    FALLBACK_FULL = "fallback_full"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ProvenancedValue:
    """One scalar with mandatory provenance.  Unavailable values are None."""

    value: int | None
    provenance: MetricProvenance
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, MetricProvenance):
            raise MetricsError("provenance must be a MetricProvenance")
        if not self.unit or not isinstance(self.unit, str):
            raise MetricsError("unit must be a non-empty string")
        if self.provenance is MetricProvenance.UNAVAILABLE:
            if self.value is not None:
                raise MetricsError(
                    "unavailable metrics must not carry a numeric value"
                )
            return
        if self.value is None:
            raise MetricsError(
                f"{self.provenance.value} metrics require a numeric value"
            )
        if type(self.value) is not int:
            raise MetricsError("metric values must be int (deterministic units)")
        if self.value < 0 or self.value > MAX_NON_NEGATIVE:
            raise MetricsError("metric value out of closed non-negative range")

    @property
    def is_measured(self) -> bool:
        return self.provenance is MetricProvenance.MEASURED

    @property
    def is_estimated(self) -> bool:
        return self.provenance is MetricProvenance.ESTIMATED

    @property
    def is_available(self) -> bool:
        return self.provenance is not MetricProvenance.UNAVAILABLE

    def to_canonical(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "provenance": self.provenance.value,
            "unit": self.unit,
        }


def measured(value: int, unit: str) -> ProvenancedValue:
    return ProvenancedValue(value, MetricProvenance.MEASURED, unit)


def estimated(value: int, unit: str) -> ProvenancedValue:
    return ProvenancedValue(value, MetricProvenance.ESTIMATED, unit)


def unavailable(unit: str) -> ProvenancedValue:
    return ProvenancedValue(None, MetricProvenance.UNAVAILABLE, unit)


def _require_non_negative_int(name: str, value: int) -> int:
    if type(value) is not int:
        raise MetricsError(f"{name} must be an int")
    if value < 0 or value > MAX_NON_NEGATIVE:
        raise MetricsError(f"{name} out of closed non-negative range")
    return value


def _basis_points(numerator: int, denominator: int) -> int:
    """Floor-divide percentage into basis points; 0 when denominator is 0."""
    if denominator <= 0:
        return 0
    return (numerator * BASIS_POINTS) // denominator


@dataclass(frozen=True, slots=True)
class UnitCounts:
    """Closed unit arithmetic for one transition.

    Invariants:
    * ``required == reused + newly_proved``
    * ``newly_proved == invalidated + added``
    * ``removed`` is independent of the required set
    """

    required: int
    reused: int
    invalidated: int
    added: int
    removed: int
    newly_proved: int
    simulated: int = 0

    def __post_init__(self) -> None:
        for name in (
            "required",
            "reused",
            "invalidated",
            "added",
            "removed",
            "newly_proved",
            "simulated",
        ):
            _require_non_negative_int(name, getattr(self, name))
        if self.required != self.reused + self.newly_proved:
            raise MetricsError(
                "required must equal reused + newly_proved "
                f"({self.required} != {self.reused} + {self.newly_proved})"
            )
        if self.newly_proved != self.invalidated + self.added:
            raise MetricsError(
                "newly_proved must equal invalidated + added "
                f"({self.newly_proved} != {self.invalidated} + {self.added})"
            )
        if self.simulated > self.required:
            raise MetricsError("simulated units cannot exceed required units")

    @property
    def production_required(self) -> int:
        """Required units that may contribute to production compute claims."""
        return self.required - self.simulated

    @property
    def production_reused(self) -> int:
        # Simulated units never authorize production reuse claims.
        return max(0, self.reused - min(self.reused, self.simulated))

    @property
    def cache_hit_basis_points(self) -> int:
        return _basis_points(self.reused, self.required)

    @property
    def production_cache_hit_basis_points(self) -> int:
        return _basis_points(self.production_reused, self.production_required)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "reused": self.reused,
            "invalidated": self.invalidated,
            "added": self.added,
            "removed": self.removed,
            "newly_proved": self.newly_proved,
            "simulated": self.simulated,
            "cache_hit_basis_points": self.cache_hit_basis_points,
            "production_cache_hit_basis_points": (
                self.production_cache_hit_basis_points
            ),
        }


def unit_counts_from_sets(
    *,
    required: Sequence[str],
    reused: Sequence[str] = (),
    invalidated: Sequence[str] = (),
    added: Sequence[str] = (),
    removed: Sequence[str] = (),
    newly_proved: Sequence[str] | None = None,
    simulated: Sequence[str] = (),
) -> UnitCounts:
    """Build :class:`UnitCounts` from identity sets with fail-closed checks."""
    required_ids = frozenset(required)
    reused_ids = frozenset(reused)
    invalidated_ids = frozenset(invalidated)
    added_ids = frozenset(added)
    removed_ids = frozenset(removed)
    simulated_ids = frozenset(simulated)
    if newly_proved is None:
        proved_ids = (invalidated_ids | added_ids) - reused_ids
    else:
        proved_ids = frozenset(newly_proved)
    if reused_ids - required_ids:
        raise MetricsError("reused units must be a subset of required")
    if proved_ids - required_ids:
        raise MetricsError("newly_proved units must be a subset of required")
    if reused_ids & proved_ids:
        raise MetricsError("a unit cannot be both reused and newly_proved")
    if required_ids != reused_ids | proved_ids:
        raise MetricsError("required must partition into reused and newly_proved")
    if simulated_ids - required_ids:
        raise MetricsError("simulated units must be a subset of required")
    return UnitCounts(
        required=len(required_ids),
        reused=len(reused_ids),
        invalidated=len(invalidated_ids),
        added=len(added_ids),
        removed=len(removed_ids),
        newly_proved=len(proved_ids),
        simulated=len(simulated_ids),
    )


@dataclass(frozen=True, slots=True)
class ResourceTimings:
    """Wall / CPU / GPU / memory / size observations for one path."""

    leaf_proving_ms: ProvenancedValue = field(
        default_factory=lambda: unavailable("ms")
    )
    aggregation_ms: ProvenancedValue = field(
        default_factory=lambda: unavailable("ms")
    )
    verification_ms: ProvenancedValue = field(
        default_factory=lambda: unavailable("ms")
    )
    wall_ms: ProvenancedValue = field(default_factory=lambda: unavailable("ms"))
    cpu_ms: ProvenancedValue = field(default_factory=lambda: unavailable("ms"))
    gpu_ms: ProvenancedValue = field(default_factory=lambda: unavailable("ms"))
    peak_memory_bytes: ProvenancedValue = field(
        default_factory=lambda: unavailable("bytes")
    )
    proof_size_bytes: ProvenancedValue = field(
        default_factory=lambda: unavailable("bytes")
    )
    seal_size_bytes: ProvenancedValue = field(
        default_factory=lambda: unavailable("bytes")
    )
    storage_growth_bytes: ProvenancedValue = field(
        default_factory=lambda: unavailable("bytes")
    )

    def __post_init__(self) -> None:
        for name in (
            "leaf_proving_ms",
            "aggregation_ms",
            "verification_ms",
            "wall_ms",
            "cpu_ms",
            "gpu_ms",
            "peak_memory_bytes",
            "proof_size_bytes",
            "seal_size_bytes",
            "storage_growth_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, ProvenancedValue):
                raise MetricsError(f"{name} must be a ProvenancedValue")
        # Wall/CPU/GPU must keep distinct units even when unavailable.
        for name in (
            "leaf_proving_ms",
            "aggregation_ms",
            "verification_ms",
            "wall_ms",
            "cpu_ms",
            "gpu_ms",
        ):
            if getattr(self, name).unit != "ms":
                raise MetricsError(f"{name} unit must be 'ms'")
        for name in (
            "peak_memory_bytes",
            "proof_size_bytes",
            "seal_size_bytes",
            "storage_growth_bytes",
        ):
            if getattr(self, name).unit != "bytes":
                raise MetricsError(f"{name} unit must be 'bytes'")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "leaf_proving_ms": self.leaf_proving_ms.to_canonical(),
            "aggregation_ms": self.aggregation_ms.to_canonical(),
            "verification_ms": self.verification_ms.to_canonical(),
            "wall_ms": self.wall_ms.to_canonical(),
            "cpu_ms": self.cpu_ms.to_canonical(),
            "gpu_ms": self.gpu_ms.to_canonical(),
            "peak_memory_bytes": self.peak_memory_bytes.to_canonical(),
            "proof_size_bytes": self.proof_size_bytes.to_canonical(),
            "seal_size_bytes": self.seal_size_bytes.to_canonical(),
            "storage_growth_bytes": self.storage_growth_bytes.to_canonical(),
        }


def estimate_path_cost(
    counts: UnitCounts,
    *,
    full: bool,
    aggregate_nodes: int = 1,
    leaf_cpu_ms: int = DEFAULT_LEAF_CPU_MS,
    aggregate_cpu_ms: int = DEFAULT_AGGREGATE_CPU_MS,
    verify_cpu_ms: int = DEFAULT_VERIFY_CPU_MS,
    leaf_storage_bytes: int = DEFAULT_LEAF_STORAGE_BYTES,
    reuse_storage_bytes: int = DEFAULT_REUSE_STORAGE_BYTES,
) -> ResourceTimings:
    """Deterministic cost-model estimate for full or incremental work."""
    _require_non_negative_int("aggregate_nodes", aggregate_nodes)
    for name, value in (
        ("leaf_cpu_ms", leaf_cpu_ms),
        ("aggregate_cpu_ms", aggregate_cpu_ms),
        ("verify_cpu_ms", verify_cpu_ms),
        ("leaf_storage_bytes", leaf_storage_bytes),
        ("reuse_storage_bytes", reuse_storage_bytes),
    ):
        _require_non_negative_int(name, value)

    # Equivalent required work: full always proves the production required set.
    production = counts.production_required
    if full:
        prove = production
        reuse = 0
        nodes = max(aggregate_nodes, 1)
    else:
        # Incremental proves newly_proved (excluding simulated) and reuses rest.
        prove = max(0, counts.newly_proved - min(counts.newly_proved, counts.simulated))
        reuse = max(0, production - prove)
        nodes = max(aggregate_nodes, 1 if prove or reuse else 0)

    leaf_ms = prove * leaf_cpu_ms
    agg_ms = nodes * aggregate_cpu_ms
    verify_ms = (prove + reuse) * verify_cpu_ms
    cpu_ms = leaf_ms + agg_ms + verify_ms
    storage = prove * leaf_storage_bytes + reuse * reuse_storage_bytes
    return ResourceTimings(
        leaf_proving_ms=estimated(leaf_ms, "ms"),
        aggregation_ms=estimated(agg_ms, "ms"),
        verification_ms=estimated(verify_ms, "ms"),
        wall_ms=estimated(cpu_ms, "ms"),
        cpu_ms=estimated(cpu_ms, "ms"),
        # Cost model does not invent GPU or peak-memory figures.
        gpu_ms=unavailable("ms"),
        peak_memory_bytes=unavailable("bytes"),
        proof_size_bytes=estimated(prove * leaf_storage_bytes, "bytes"),
        seal_size_bytes=estimated(storage, "bytes"),
        storage_growth_bytes=estimated(storage, "bytes"),
    )


def _combine_cost_ms(timings: ResourceTimings) -> ProvenancedValue:
    """Prefer measured CPU; otherwise measured wall; otherwise estimated CPU."""
    if timings.cpu_ms.is_measured:
        return timings.cpu_ms
    if timings.wall_ms.is_measured:
        return timings.wall_ms
    if timings.cpu_ms.is_estimated:
        return timings.cpu_ms
    if timings.wall_ms.is_estimated:
        return timings.wall_ms
    return unavailable("ms")


def _savings(
    full: ProvenancedValue, incremental: ProvenancedValue
) -> tuple[ProvenancedValue, ProvenancedValue]:
    """Absolute and basis-point savings for equivalent required work.

    Savings provenance is the weaker of the two inputs: both measured =>
    measured; any estimated => estimated; any unavailable => unavailable.
    Negative savings (incremental worse) clamp absolute savings to 0 but keep
    the comparison visible via full/incremental values.
    """
    if not full.is_available or not incremental.is_available:
        return unavailable("ms"), unavailable("basis_points")
    assert full.value is not None and incremental.value is not None
    absolute = max(0, full.value - incremental.value)
    ratio = _basis_points(absolute, full.value) if full.value > 0 else 0
    if full.is_measured and incremental.is_measured:
        provenance = MetricProvenance.MEASURED
    else:
        # Mixed measured/estimated or pure estimated never upgrades to measured.
        provenance = MetricProvenance.ESTIMATED
    return (
        ProvenancedValue(absolute, provenance, "ms"),
        ProvenancedValue(ratio, provenance, "basis_points"),
    )


@dataclass(frozen=True, slots=True)
class ProofCostRecord:
    """One path's measured or estimated cost envelope."""

    schema: str
    evidence_subset: str
    path: str  # "full" | "incremental"
    counts: UnitCounts
    timings: ResourceTimings
    cost_ms: ProvenancedValue
    visibility: RunVisibility
    fallback_reason: str = ""
    chain_depth: int = 0
    aggregate_nodes: int = 0
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.path not in {"full", "incremental"}:
            raise MetricsError("path must be 'full' or 'incremental'")
        if not isinstance(self.counts, UnitCounts):
            raise MetricsError("counts must be UnitCounts")
        if not isinstance(self.timings, ResourceTimings):
            raise MetricsError("timings must be ResourceTimings")
        if not isinstance(self.cost_ms, ProvenancedValue):
            raise MetricsError("cost_ms must be ProvenancedValue")
        if self.cost_ms.unit != "ms":
            raise MetricsError("cost_ms unit must be 'ms'")
        if not isinstance(self.visibility, RunVisibility):
            raise MetricsError("visibility must be RunVisibility")
        _require_non_negative_int("chain_depth", self.chain_depth)
        _require_non_negative_int("aggregate_nodes", self.aggregate_nodes)
        # Estimates must never be reported as measurements.
        if self.cost_ms.provenance is MetricProvenance.MEASURED and not (
            self.timings.cpu_ms.is_measured or self.timings.wall_ms.is_measured
        ):
            raise MetricsError(
                "cost_ms cannot be measured without measured cpu_ms or wall_ms"
            )
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.visibility is RunVisibility.FALLBACK_FULL and not self.fallback_reason:
            raise MetricsError("fallback_full visibility requires fallback_reason")

    @property
    def is_visible_failure(self) -> bool:
        return self.visibility in {
            RunVisibility.FAILED,
            RunVisibility.FALLBACK_FULL,
            RunVisibility.CANCELLED,
            RunVisibility.TIMEOUT,
            RunVisibility.UNAVAILABLE,
        }

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "path": self.path,
            "counts": self.counts.to_canonical(),
            "timings": self.timings.to_canonical(),
            "cost_ms": self.cost_ms.to_canonical(),
            "visibility": self.visibility.value,
            "fallback_reason": self.fallback_reason,
            "chain_depth": self.chain_depth,
            "aggregate_nodes": self.aggregate_nodes,
            "notes": list(self.notes),
        }

    def record_cid(self) -> str:
        payload = json.dumps(
            self.to_canonical(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ProofCostComparison:
    """Equivalent-work full versus incremental comparison with savings."""

    schema: str
    evidence_subset: str
    counts: UnitCounts
    full: ProofCostRecord
    incremental: ProofCostRecord
    compute_saved_ms: ProvenancedValue
    compute_saved_basis_points: ProvenancedValue
    storage_saved_bytes: ProvenancedValue
    visibility: RunVisibility
    fallback_reason: str = ""
    production_claims_excluded: bool = False

    def __post_init__(self) -> None:
        if self.full.path != "full":
            raise MetricsError("full record path must be 'full'")
        if self.incremental.path != "incremental":
            raise MetricsError("incremental record path must be 'incremental'")
        if self.full.counts != self.counts or self.incremental.counts != self.counts:
            raise MetricsError(
                "comparison counts must match both path records "
                "(equivalent required work)"
            )
        for field_name in ("compute_saved_ms", "compute_saved_basis_points", "storage_saved_bytes"):
            value = getattr(self, field_name)
            if not isinstance(value, ProvenancedValue):
                raise MetricsError(f"{field_name} must be ProvenancedValue")
        if self.compute_saved_ms.unit != "ms":
            raise MetricsError("compute_saved_ms unit must be 'ms'")
        if self.compute_saved_basis_points.unit != "basis_points":
            raise MetricsError(
                "compute_saved_basis_points unit must be 'basis_points'"
            )
        if self.storage_saved_bytes.unit != "bytes":
            raise MetricsError("storage_saved_bytes unit must be 'bytes'")
        # Never promote estimated savings to measured.
        if self.compute_saved_ms.is_measured:
            if not (
                self.full.cost_ms.is_measured and self.incremental.cost_ms.is_measured
            ):
                raise MetricsError(
                    "measured compute savings require measured full and "
                    "incremental costs"
                )
        if self.production_claims_excluded and self.compute_saved_ms.is_measured:
            raise MetricsError(
                "production-excluded comparisons cannot report measured savings"
            )

    @property
    def is_visible_failure(self) -> bool:
        return self.visibility in {
            RunVisibility.FAILED,
            RunVisibility.FALLBACK_FULL,
            RunVisibility.CANCELLED,
            RunVisibility.TIMEOUT,
            RunVisibility.UNAVAILABLE,
        } or self.full.is_visible_failure or self.incremental.is_visible_failure

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "counts": self.counts.to_canonical(),
            "full": self.full.to_canonical(),
            "incremental": self.incremental.to_canonical(),
            "compute_saved_ms": self.compute_saved_ms.to_canonical(),
            "compute_saved_basis_points": (
                self.compute_saved_basis_points.to_canonical()
            ),
            "storage_saved_bytes": self.storage_saved_bytes.to_canonical(),
            "visibility": self.visibility.value,
            "fallback_reason": self.fallback_reason,
            "production_claims_excluded": self.production_claims_excluded,
        }

    def comparison_cid(self) -> str:
        payload = json.dumps(
            self.to_canonical(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_record(
    *,
    path: str,
    counts: UnitCounts,
    timings: ResourceTimings,
    visibility: RunVisibility,
    fallback_reason: str = "",
    chain_depth: int = 0,
    aggregate_nodes: int = 0,
    notes: Sequence[str] = (),
) -> ProofCostRecord:
    return ProofCostRecord(
        schema=COST_SCHEMA,
        evidence_subset=EVIDENCE_SUBSET,
        path=path,
        counts=counts,
        timings=timings,
        cost_ms=_combine_cost_ms(timings),
        visibility=visibility,
        fallback_reason=fallback_reason,
        chain_depth=chain_depth,
        aggregate_nodes=aggregate_nodes,
        notes=tuple(notes),
    )


def _storage_savings(
    full: ResourceTimings, incremental: ResourceTimings
) -> ProvenancedValue:
    full_s = full.storage_growth_bytes
    inc_s = incremental.storage_growth_bytes
    if not full_s.is_available or not inc_s.is_available:
        return unavailable("bytes")
    assert full_s.value is not None and inc_s.value is not None
    absolute = max(0, full_s.value - inc_s.value)
    if full_s.is_measured and inc_s.is_measured:
        provenance = MetricProvenance.MEASURED
    else:
        provenance = MetricProvenance.ESTIMATED
    return ProvenancedValue(absolute, provenance, "bytes")


def compare_costs(
    counts: UnitCounts,
    *,
    full_timings: ResourceTimings | None = None,
    incremental_timings: ResourceTimings | None = None,
    aggregate_nodes: int = 1,
    visibility: RunVisibility = RunVisibility.SUCCESS,
    fallback_reason: str = "",
    chain_depth: int = 0,
    exclude_simulated_from_production: bool = True,
) -> ProofCostComparison:
    """Compare full versus incremental cost for the same required unit set.

    When measured timings are omitted, a deterministic cost model supplies
    estimated values.  Simulated units are excluded from production compute
    claims when ``exclude_simulated_from_production`` is true.
    """
    if not isinstance(counts, UnitCounts):
        raise MetricsError("counts must be UnitCounts")
    _require_non_negative_int("aggregate_nodes", aggregate_nodes)
    if not isinstance(visibility, RunVisibility):
        raise MetricsError("visibility must be RunVisibility")
    if visibility is RunVisibility.FALLBACK_FULL and not fallback_reason:
        raise MetricsError("fallback_full requires fallback_reason")

    production_excluded = bool(
        exclude_simulated_from_production and counts.simulated > 0
    )
    notes: list[str] = []
    if production_excluded:
        notes.append("simulated_units_excluded_from_production_compute_claims")

    full = full_timings or estimate_path_cost(
        counts, full=True, aggregate_nodes=aggregate_nodes
    )
    incremental = incremental_timings or estimate_path_cost(
        counts, full=False, aggregate_nodes=aggregate_nodes
    )
    if not isinstance(full, ResourceTimings) or not isinstance(
        incremental, ResourceTimings
    ):
        raise MetricsError("timings must be ResourceTimings")

    # Derive savings before building records so production-exclusion notes are
    # complete and attached to both path records.
    full_cost = _combine_cost_ms(full)
    incremental_cost = _combine_cost_ms(incremental)
    saved_ms, saved_bp = _savings(full_cost, incremental_cost)
    storage_saved = _storage_savings(full, incremental)

    if production_excluded:
        # Keep absolute comparison inspectable but strip measured production claim.
        if saved_ms.is_measured:
            assert saved_ms.value is not None and saved_bp.value is not None
            saved_ms = estimated(saved_ms.value, "ms")
            saved_bp = estimated(saved_bp.value, "basis_points")
            notes.append("production_compute_claims_downgraded_to_estimated")
        if storage_saved.is_measured:
            assert storage_saved.value is not None
            storage_saved = estimated(storage_saved.value, "bytes")

    full_record = _build_record(
        path="full",
        counts=counts,
        timings=full,
        visibility=visibility,
        fallback_reason=fallback_reason,
        chain_depth=chain_depth,
        aggregate_nodes=aggregate_nodes,
        notes=notes,
    )
    incremental_record = _build_record(
        path="incremental",
        counts=counts,
        timings=incremental,
        visibility=visibility,
        fallback_reason=fallback_reason,
        chain_depth=chain_depth,
        aggregate_nodes=aggregate_nodes,
        notes=notes,
    )

    return ProofCostComparison(
        schema=COMPARISON_SCHEMA,
        evidence_subset=EVIDENCE_SUBSET,
        counts=counts,
        full=full_record,
        incremental=incremental_record,
        compute_saved_ms=saved_ms,
        compute_saved_basis_points=saved_bp,
        storage_saved_bytes=storage_saved,
        visibility=visibility,
        fallback_reason=fallback_reason,
        production_claims_excluded=production_excluded,
    )


def compare_full_and_incremental(
    counts: UnitCounts | Mapping[str, int],
    *,
    full_timings: ResourceTimings | Mapping[str, Any] | None = None,
    incremental_timings: ResourceTimings | Mapping[str, Any] | None = None,
    aggregate_nodes: int = 1,
    visibility: RunVisibility | str = RunVisibility.SUCCESS,
    fallback_reason: str = "",
    chain_depth: int = 0,
) -> ProofCostComparison:
    """Public facade matching the plan's compare_full_and_incremental API."""
    resolved_counts = (
        counts
        if isinstance(counts, UnitCounts)
        else UnitCounts(
            required=int(counts["required"]),
            reused=int(counts.get("reused", 0)),
            invalidated=int(counts.get("invalidated", 0)),
            added=int(counts.get("added", 0)),
            removed=int(counts.get("removed", 0)),
            newly_proved=int(
                counts.get(
                    "newly_proved",
                    int(counts.get("invalidated", 0)) + int(counts.get("added", 0)),
                )
            ),
            simulated=int(counts.get("simulated", 0)),
        )
    )
    vis = (
        visibility
        if isinstance(visibility, RunVisibility)
        else RunVisibility(visibility)
    )
    return compare_costs(
        resolved_counts,
        full_timings=_coerce_timings(full_timings),
        incremental_timings=_coerce_timings(incremental_timings),
        aggregate_nodes=aggregate_nodes,
        visibility=vis,
        fallback_reason=fallback_reason,
        chain_depth=chain_depth,
    )


def _coerce_timings(
    value: ResourceTimings | Mapping[str, Any] | None,
) -> ResourceTimings | None:
    if value is None:
        return None
    if isinstance(value, ResourceTimings):
        return value
    if not isinstance(value, Mapping):
        raise MetricsError("timings must be ResourceTimings, mapping, or None")
    fields: dict[str, ProvenancedValue] = {}
    for name in (
        "leaf_proving_ms",
        "aggregation_ms",
        "verification_ms",
        "wall_ms",
        "cpu_ms",
        "gpu_ms",
        "peak_memory_bytes",
        "proof_size_bytes",
        "seal_size_bytes",
        "storage_growth_bytes",
    ):
        if name not in value:
            continue
        item = value[name]
        if isinstance(item, ProvenancedValue):
            fields[name] = item
            continue
        if not isinstance(item, Mapping):
            raise MetricsError(f"{name} must be ProvenancedValue or mapping")
        prov = MetricProvenance(str(item.get("provenance", "unavailable")))
        unit = str(item.get("unit", "ms" if name.endswith("_ms") else "bytes"))
        raw = item.get("value")
        if prov is MetricProvenance.UNAVAILABLE:
            fields[name] = unavailable(unit)
        else:
            fields[name] = ProvenancedValue(
                None if raw is None else int(raw), prov, unit
            )
    return ResourceTimings(**fields)


class ProofMetricsCollector:
    """Accumulate lifecycle samples and emit deterministic cost records.

    Failed, timeout, cancelled, unavailable, and full-fallback runs are retained
    in ``history`` and never silently dropped.
    """

    def __init__(self) -> None:
        self._history: list[ProofCostComparison] = []

    @property
    def history(self) -> tuple[ProofCostComparison, ...]:
        return tuple(self._history)

    def record(
        self,
        counts: UnitCounts,
        *,
        full_timings: ResourceTimings | None = None,
        incremental_timings: ResourceTimings | None = None,
        aggregate_nodes: int = 1,
        visibility: RunVisibility = RunVisibility.SUCCESS,
        fallback_reason: str = "",
        chain_depth: int = 0,
    ) -> ProofCostComparison:
        comparison = compare_costs(
            counts,
            full_timings=full_timings,
            incremental_timings=incremental_timings,
            aggregate_nodes=aggregate_nodes,
            visibility=visibility,
            fallback_reason=fallback_reason,
            chain_depth=chain_depth,
        )
        self._history.append(comparison)
        return comparison

    def record_measured_incremental(
        self,
        counts: UnitCounts,
        *,
        leaf_proving_ms: int,
        aggregation_ms: int,
        verification_ms: int,
        wall_ms: int,
        cpu_ms: int,
        gpu_ms: int | None = None,
        peak_memory_bytes: int | None = None,
        proof_size_bytes: int | None = None,
        seal_size_bytes: int | None = None,
        storage_growth_bytes: int | None = None,
        aggregate_nodes: int = 1,
        visibility: RunVisibility = RunVisibility.SUCCESS,
        fallback_reason: str = "",
        chain_depth: int = 0,
        include_estimated_full: bool = True,
    ) -> ProofCostComparison:
        """Record a measured incremental path and optional estimated full path."""
        incremental = ResourceTimings(
            leaf_proving_ms=measured(leaf_proving_ms, "ms"),
            aggregation_ms=measured(aggregation_ms, "ms"),
            verification_ms=measured(verification_ms, "ms"),
            wall_ms=measured(wall_ms, "ms"),
            cpu_ms=measured(cpu_ms, "ms"),
            gpu_ms=(
                measured(gpu_ms, "ms") if gpu_ms is not None else unavailable("ms")
            ),
            peak_memory_bytes=(
                measured(peak_memory_bytes, "bytes")
                if peak_memory_bytes is not None
                else unavailable("bytes")
            ),
            proof_size_bytes=(
                measured(proof_size_bytes, "bytes")
                if proof_size_bytes is not None
                else unavailable("bytes")
            ),
            seal_size_bytes=(
                measured(seal_size_bytes, "bytes")
                if seal_size_bytes is not None
                else unavailable("bytes")
            ),
            storage_growth_bytes=(
                measured(storage_growth_bytes, "bytes")
                if storage_growth_bytes is not None
                else unavailable("bytes")
            ),
        )
        full = (
            estimate_path_cost(counts, full=True, aggregate_nodes=aggregate_nodes)
            if include_estimated_full
            else None
        )
        return self.record(
            counts,
            full_timings=full,
            incremental_timings=incremental,
            aggregate_nodes=aggregate_nodes,
            visibility=visibility,
            fallback_reason=fallback_reason,
            chain_depth=chain_depth,
        )

    def visible_failures(self) -> tuple[ProofCostComparison, ...]:
        return tuple(item for item in self._history if item.is_visible_failure)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": COMPARISON_SCHEMA,
            "evidence_subset": EVIDENCE_SUBSET,
            "comparisons": [item.to_canonical() for item in self._history],
            "visible_failure_count": len(self.visible_failures()),
        }


__all__ = (
    "BASIS_POINTS",
    "COMPARISON_SCHEMA",
    "COST_SCHEMA",
    "DEFAULT_AGGREGATE_CPU_MS",
    "DEFAULT_LEAF_CPU_MS",
    "DEFAULT_LEAF_STORAGE_BYTES",
    "DEFAULT_REUSE_STORAGE_BYTES",
    "DEFAULT_VERIFY_CPU_MS",
    "EVIDENCE_SUBSET",
    "MetricProvenance",
    "MetricsError",
    "ProofCostComparison",
    "ProofCostRecord",
    "ProofMetricsCollector",
    "ProvenancedValue",
    "ResourceTimings",
    "RunVisibility",
    "UnitCounts",
    "compare_costs",
    "compare_full_and_incremental",
    "estimate_path_cost",
    "estimated",
    "measured",
    "unavailable",
    "unit_counts_from_sets",
)
