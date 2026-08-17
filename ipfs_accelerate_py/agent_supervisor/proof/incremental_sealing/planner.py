"""Full-versus-incremental planning and reuse explanations (IPS-032).

Accelerate consumes datasets invalidation semantics and kit candidate hints
without copying either authority.  A kit cache candidate never authorizes
reuse; only a complete cache key plus an accelerate-issued admission record
may preserve a unit.

Interfaces: ``IncrementalProofPlan``, ``AggregationPlan``, ``ResourceEstimate``,
``FinalAcceptancePolicy``, ``create_incremental_plan``, ``plan_incremental_proof``,
``explain_reuse``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

EVIDENCE_SUBSET: Final[str] = "ips/incremental-plan@1"
REUSE_EVIDENCE_SUBSET: Final[str] = "ips/reuse-explanation@1"
PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "incremental-proof-plan@1"
)
AGGREGATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "aggregation-plan@1"
)

FULL_FALLBACK_REASONS: Final[tuple[str, ...]] = (
    "first_state",
    "missing_parent",
    "trust_policy_change",
    "schema_change",
    "canonicalization_change",
    "environment_change",
    "circuit_or_key_change",
    "incomplete_cache_key",
    "full_fallback_required",
)


class PlannerError(ValueError):
    """Fail-closed incremental planning contract violation."""


class PlanMode(str, Enum):
    INCREMENTAL = "incremental"
    FULL = "full"


class UnitPlanKind(str, Enum):
    REUSE = "reuse"
    REPROVE = "reprove"
    PROVE_NEW = "prove_new"
    REMOVE = "remove"
    REJECT_REUSE = "reject_reuse"


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    """Deterministic expected resource envelope for a plan."""

    prove_units: int
    reuse_units: int
    aggregate_nodes: int
    expected_cpu_ms: int
    expected_storage_bytes: int
    expected_full_cpu_ms: int
    expected_full_storage_bytes: int

    @property
    def savings_cpu_ms(self) -> int:
        return max(0, self.expected_full_cpu_ms - self.expected_cpu_ms)

    @property
    def savings_storage_bytes(self) -> int:
        return max(0, self.expected_full_storage_bytes - self.expected_storage_bytes)

    def to_canonical(self) -> dict[str, Any]:
        return {
            "prove_units": self.prove_units,
            "reuse_units": self.reuse_units,
            "aggregate_nodes": self.aggregate_nodes,
            "expected_cpu_ms": self.expected_cpu_ms,
            "expected_storage_bytes": self.expected_storage_bytes,
            "expected_full_cpu_ms": self.expected_full_cpu_ms,
            "expected_full_storage_bytes": self.expected_full_storage_bytes,
            "savings_cpu_ms": self.savings_cpu_ms,
            "savings_storage_bytes": self.savings_storage_bytes,
        }


@dataclass(frozen=True, slots=True)
class AggregationPlan:
    """Bounded fan-in aggregation over affected units only."""

    schema: str = AGGREGATION_SCHEMA
    strategy: str = "manifest_aggregation"
    fan_in: int = 8
    affected_unit_ids: tuple[str, ...] = ()
    rebuild_aggregates: tuple[str, ...] = ()
    recursive_verification: bool = False

    def __post_init__(self) -> None:
        if self.fan_in < 2:
            raise PlannerError("aggregation fan_in must be >= 2")
        object.__setattr__(
            self,
            "affected_unit_ids",
            tuple(sorted(set(self.affected_unit_ids))),
        )
        object.__setattr__(
            self,
            "rebuild_aggregates",
            tuple(sorted(set(self.rebuild_aggregates))),
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "strategy": self.strategy,
            "fan_in": self.fan_in,
            "affected_unit_ids": list(self.affected_unit_ids),
            "rebuild_aggregates": list(self.rebuild_aggregates),
            "recursive_verification": self.recursive_verification,
        }


@dataclass(frozen=True, slots=True)
class FinalAcceptancePolicy:
    """Gates that must hold before an incremental seal may be published."""

    require_complete_manifest: bool = True
    require_parent_match: bool = True
    forbid_candidate_only_reuse: bool = True
    forbid_simulated_reuse: bool = True
    require_admission_for_reuse: bool = True

    def to_canonical(self) -> dict[str, Any]:
        return {
            "require_complete_manifest": self.require_complete_manifest,
            "require_parent_match": self.require_parent_match,
            "forbid_candidate_only_reuse": self.forbid_candidate_only_reuse,
            "forbid_simulated_reuse": self.forbid_simulated_reuse,
            "require_admission_for_reuse": self.require_admission_for_reuse,
        }


@dataclass(frozen=True, slots=True)
class PlannedUnit:
    """One unit's incremental disposition."""

    unit_id: str
    kind: UnitPlanKind
    cache_key_complete: bool
    admitted: bool
    candidate_present: bool
    reason: str

    def to_canonical(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "kind": self.kind.value,
            "cache_key_complete": self.cache_key_complete,
            "admitted": self.admitted,
            "candidate_present": self.candidate_present,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class IncrementalProofPlan:
    """Deterministic incremental-or-full plan for one repository step."""

    schema: str
    evidence_subset: str
    mode: PlanMode
    parent_seal_cid: str
    old_repository_state_cid: str
    new_repository_state_cid: str
    reusable_unit_ids: tuple[str, ...]
    invalidated_unit_ids: tuple[str, ...]
    added_unit_ids: tuple[str, ...]
    removed_unit_ids: tuple[str, ...]
    changed_root_cids: tuple[str, ...]
    fallback_reasons: tuple[str, ...]
    units: tuple[PlannedUnit, ...]
    aggregation: AggregationPlan
    resources: ResourceEstimate
    acceptance: FinalAcceptancePolicy
    complete: bool

    def plan_cid(self) -> str:
        payload = json.dumps(
            self.to_canonical(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "mode": self.mode.value,
            "parent_seal_cid": self.parent_seal_cid,
            "old_repository_state_cid": self.old_repository_state_cid,
            "new_repository_state_cid": self.new_repository_state_cid,
            "reusable_unit_ids": list(self.reusable_unit_ids),
            "invalidated_unit_ids": list(self.invalidated_unit_ids),
            "added_unit_ids": list(self.added_unit_ids),
            "removed_unit_ids": list(self.removed_unit_ids),
            "changed_root_cids": list(self.changed_root_cids),
            "fallback_reasons": list(self.fallback_reasons),
            "units": [item.to_canonical() for item in self.units],
            "aggregation": self.aggregation.to_canonical(),
            "resources": self.resources.to_canonical(),
            "acceptance": self.acceptance.to_canonical(),
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class ProofReuseExplanation:
    """Why one unit was reused or rejected for reuse."""

    evidence_subset: str
    unit_id: str
    reused: bool
    reason: str
    cache_key_complete: bool
    admitted: bool
    candidate_present: bool
    parent_seal_cid: str

    def to_canonical(self) -> dict[str, Any]:
        return {
            "evidence_subset": self.evidence_subset,
            "unit_id": self.unit_id,
            "reused": self.reused,
            "reason": self.reason,
            "cache_key_complete": self.cache_key_complete,
            "admitted": self.admitted,
            "candidate_present": self.candidate_present,
            "parent_seal_cid": self.parent_seal_cid,
        }


@dataclass(frozen=True, slots=True)
class UnitPlanningInput:
    """Planner-facing view of one required unit.  Kit/datasets stay outside."""

    unit_id: str
    preserved: bool = False
    invalidated: bool = False
    added: bool = False
    removed: bool = False
    cache_key_complete: bool = False
    admitted: bool = False
    candidate_present: bool = False
    simulated: bool = False
    aggregate: bool = False
    source_root_cid: str = ""


@dataclass(frozen=True, slots=True)
class ParentSealContext:
    """Declared parent seal bindings the planner verifies, not trusts blindly."""

    seal_cid: str
    repository_state_cid: str
    source_root_cid: str = ""
    schema_version: str = "1"
    canonicalization_version: str = "1"
    environment_cid: str = ""
    policy_cid: str = ""
    first_state: bool = False


@dataclass(frozen=True, slots=True)
class PlanningRequest:
    """Closed inputs for incremental planning."""

    parent: ParentSealContext | None
    old_repository_state_cid: str
    new_repository_state_cid: str
    units: tuple[UnitPlanningInput, ...]
    changed_root_cids: tuple[str, ...] = ()
    trust_policy_changed: bool = False
    schema_changed: bool = False
    canonicalization_changed: bool = False
    environment_changed: bool = False
    circuit_or_key_changed: bool = False
    full_fallback_required: bool = False
    new_source_root_cid: str = ""
    acceptance: FinalAcceptancePolicy = field(default_factory=FinalAcceptancePolicy)


def _sorted_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _estimate(prove: int, reuse: int, aggregates: int) -> ResourceEstimate:
    incremental_cpu = prove * 1000 + aggregates * 50
    incremental_storage = prove * 4096 + reuse * 256
    full_cpu = (prove + reuse) * 1000 + max(aggregates, 1) * 50
    full_storage = (prove + reuse) * 4096
    return ResourceEstimate(
        prove_units=prove,
        reuse_units=reuse,
        aggregate_nodes=aggregates,
        expected_cpu_ms=incremental_cpu,
        expected_storage_bytes=incremental_storage,
        expected_full_cpu_ms=full_cpu,
        expected_full_storage_bytes=full_storage,
    )


def _fallback_reasons(request: PlanningRequest) -> tuple[str, ...]:
    reasons: list[str] = []
    if request.parent is None or request.parent.first_state or not request.parent.seal_cid:
        reasons.append("first_state")
        if request.parent is None or not request.parent.seal_cid:
            reasons.append("missing_parent")
    if request.trust_policy_changed:
        reasons.append("trust_policy_change")
    if request.schema_changed:
        reasons.append("schema_change")
    if request.canonicalization_changed:
        reasons.append("canonicalization_change")
    if request.environment_changed:
        reasons.append("environment_change")
    if request.circuit_or_key_changed:
        reasons.append("circuit_or_key_change")
    if request.full_fallback_required:
        reasons.append("full_fallback_required")
    return tuple(reasons)


def _plan_unit(unit: UnitPlanningInput, *, force_full: bool) -> PlannedUnit:
    if unit.removed:
        return PlannedUnit(
            unit.unit_id,
            UnitPlanKind.REMOVE,
            unit.cache_key_complete,
            unit.admitted,
            unit.candidate_present,
            "removed",
        )
    if unit.added:
        return PlannedUnit(
            unit.unit_id,
            UnitPlanKind.PROVE_NEW,
            unit.cache_key_complete,
            unit.admitted,
            unit.candidate_present,
            "added_selected_unit",
        )
    if force_full or unit.invalidated:
        return PlannedUnit(
            unit.unit_id,
            UnitPlanKind.REPROVE,
            unit.cache_key_complete,
            unit.admitted,
            unit.candidate_present,
            "full_fallback" if force_full else "invalidated",
        )
    # Preserved units still need a complete key + accelerate admission.
    if unit.candidate_present and not unit.admitted:
        return PlannedUnit(
            unit.unit_id,
            UnitPlanKind.REJECT_REUSE,
            unit.cache_key_complete,
            False,
            True,
            "candidate_presence_is_not_reuse_authority",
        )
    if unit.simulated:
        return PlannedUnit(
            unit.unit_id,
            UnitPlanKind.REJECT_REUSE,
            unit.cache_key_complete,
            unit.admitted,
            unit.candidate_present,
            "simulated_evidence_cannot_be_reused",
        )
    if not unit.cache_key_complete:
        return PlannedUnit(
            unit.unit_id,
            UnitPlanKind.REPROVE,
            False,
            unit.admitted,
            unit.candidate_present,
            "incomplete_cache_key",
        )
    if not unit.admitted:
        return PlannedUnit(
            unit.unit_id,
            UnitPlanKind.REJECT_REUSE,
            True,
            False,
            unit.candidate_present,
            "missing_accelerate_admission",
        )
    if not unit.preserved:
        return PlannedUnit(
            unit.unit_id,
            UnitPlanKind.REPROVE,
            unit.cache_key_complete,
            unit.admitted,
            unit.candidate_present,
            "not_preserved_by_invalidation",
        )
    return PlannedUnit(
        unit.unit_id,
        UnitPlanKind.REUSE,
        True,
        True,
        unit.candidate_present,
        "admitted_complete_key",
    )


def plan_incremental_proof(request: PlanningRequest) -> IncrementalProofPlan:
    """Build a deterministic incremental or full plan from a closed request."""

    if not isinstance(request, PlanningRequest):
        raise PlannerError("request must be a PlanningRequest")
    fallbacks = _fallback_reasons(request)
    force_full = bool(fallbacks)
    planned = tuple(
        _plan_unit(unit, force_full=force_full)
        for unit in sorted(request.units, key=lambda item: item.unit_id)
    )
    reusable = tuple(
        item.unit_id for item in planned if item.kind is UnitPlanKind.REUSE
    )
    invalidated = tuple(
        item.unit_id
        for item in planned
        if item.kind is UnitPlanKind.REPROVE
    )
    added = tuple(
        item.unit_id for item in planned if item.kind is UnitPlanKind.PROVE_NEW
    )
    removed = tuple(
        item.unit_id for item in planned if item.kind is UnitPlanKind.REMOVE
    )
    if force_full:
        # Full fallback proves every remaining required unit.
        invalidated = tuple(
            item.unit_id
            for item in planned
            if item.kind is not UnitPlanKind.REMOVE
        )
        reusable = ()
        added = tuple(
            item.unit_id for item in planned if item.kind is UnitPlanKind.PROVE_NEW
        )
    aggregates = tuple(
        unit.unit_id for unit in request.units if unit.aggregate
    )
    prove_count = len(invalidated) + len(
        [
            item
            for item in planned
            if item.kind is UnitPlanKind.PROVE_NEW and not force_full
        ]
    )
    if force_full:
        prove_count = len(invalidated)
    resources = _estimate(prove_count, len(reusable), max(1, len(aggregates)))
    aggregation = AggregationPlan(
        affected_unit_ids=invalidated + added,
        rebuild_aggregates=aggregates,
        recursive_verification=False,
    )
    parent_cid = request.parent.seal_cid if request.parent is not None else ""
    mode = PlanMode.FULL if force_full else PlanMode.INCREMENTAL
    return IncrementalProofPlan(
        schema=PLAN_SCHEMA,
        evidence_subset=EVIDENCE_SUBSET,
        mode=mode,
        parent_seal_cid=parent_cid,
        old_repository_state_cid=request.old_repository_state_cid,
        new_repository_state_cid=request.new_repository_state_cid,
        reusable_unit_ids=reusable,
        invalidated_unit_ids=_sorted_unique(invalidated),
        added_unit_ids=added,
        removed_unit_ids=removed,
        changed_root_cids=_sorted_unique(request.changed_root_cids),
        fallback_reasons=fallbacks,
        units=planned,
        aggregation=aggregation,
        resources=resources,
        acceptance=request.acceptance,
        complete=True,
    )


def create_incremental_plan(
    parent_seal: ParentSealContext | Mapping[str, Any] | None,
    old_repository_state: str | Mapping[str, Any],
    new_repository_state: str | Mapping[str, Any],
    verification_policy: Mapping[str, Any] | None = None,
    *,
    units: Sequence[UnitPlanningInput | Mapping[str, Any]] = (),
    **flags: Any,
) -> IncrementalProofPlan:
    """Public facade matching the plan document's ``create_incremental_plan``."""

    parent: ParentSealContext | None
    if parent_seal is None:
        parent = None
    elif isinstance(parent_seal, ParentSealContext):
        parent = parent_seal
    elif isinstance(parent_seal, Mapping):
        parent = ParentSealContext(
            seal_cid=str(parent_seal.get("seal_cid", "")),
            repository_state_cid=str(parent_seal.get("repository_state_cid", "")),
            source_root_cid=str(parent_seal.get("source_root_cid", "")),
            schema_version=str(parent_seal.get("schema_version", "1")),
            canonicalization_version=str(
                parent_seal.get("canonicalization_version", "1")
            ),
            environment_cid=str(parent_seal.get("environment_cid", "")),
            policy_cid=str(parent_seal.get("policy_cid", "")),
            first_state=bool(parent_seal.get("first_state", False)),
        )
    else:
        raise PlannerError("parent_seal must be ParentSealContext, mapping, or None")

    def _state_cid(value: str | Mapping[str, Any], label: str) -> str:
        if isinstance(value, str) and value:
            return value
        if isinstance(value, Mapping):
            for key in ("identity_cid", "repository_state_cid", "cid"):
                item = value.get(key)
                if isinstance(item, str) and item:
                    return item
        raise PlannerError(f"{label} must provide a repository state CID")

    parsed_units: list[UnitPlanningInput] = []
    for raw in units:
        if isinstance(raw, UnitPlanningInput):
            parsed_units.append(raw)
            continue
        if not isinstance(raw, Mapping):
            raise PlannerError("units entries must be UnitPlanningInput or mapping")
        parsed_units.append(
            UnitPlanningInput(
                unit_id=str(raw["unit_id"]),
                preserved=bool(raw.get("preserved", False)),
                invalidated=bool(raw.get("invalidated", False)),
                added=bool(raw.get("added", False)),
                removed=bool(raw.get("removed", False)),
                cache_key_complete=bool(raw.get("cache_key_complete", False)),
                admitted=bool(raw.get("admitted", False)),
                candidate_present=bool(raw.get("candidate_present", False)),
                simulated=bool(raw.get("simulated", False)),
                aggregate=bool(raw.get("aggregate", False)),
                source_root_cid=str(raw.get("source_root_cid", "")),
            )
        )
    policy = verification_policy or {}
    request = PlanningRequest(
        parent=parent,
        old_repository_state_cid=_state_cid(old_repository_state, "old_repository_state"),
        new_repository_state_cid=_state_cid(new_repository_state, "new_repository_state"),
        units=tuple(parsed_units),
        changed_root_cids=tuple(flags.get("changed_root_cids", ())),
        trust_policy_changed=bool(
            flags.get("trust_policy_changed", policy.get("trust_policy_changed", False))
        ),
        schema_changed=bool(flags.get("schema_changed", policy.get("schema_changed", False))),
        canonicalization_changed=bool(
            flags.get(
                "canonicalization_changed",
                policy.get("canonicalization_changed", False),
            )
        ),
        environment_changed=bool(
            flags.get("environment_changed", policy.get("environment_changed", False))
        ),
        circuit_or_key_changed=bool(
            flags.get(
                "circuit_or_key_changed",
                policy.get("circuit_or_key_changed", False),
            )
        ),
        full_fallback_required=bool(
            flags.get(
                "full_fallback_required",
                policy.get("full_fallback_required", False),
            )
        ),
    )
    return plan_incremental_proof(request)


def explain_reuse(
    plan: IncrementalProofPlan,
    proof_unit_id: str,
) -> ProofReuseExplanation:
    """Explain why ``proof_unit_id`` was reused or rejected."""

    if not isinstance(plan, IncrementalProofPlan):
        raise PlannerError("plan must be an IncrementalProofPlan")
    if not isinstance(proof_unit_id, str) or not proof_unit_id:
        raise PlannerError("proof_unit_id must be a non-empty string")
    for item in plan.units:
        if item.unit_id == proof_unit_id:
            return ProofReuseExplanation(
                evidence_subset=REUSE_EVIDENCE_SUBSET,
                unit_id=item.unit_id,
                reused=item.kind is UnitPlanKind.REUSE,
                reason=item.reason,
                cache_key_complete=item.cache_key_complete,
                admitted=item.admitted,
                candidate_present=item.candidate_present,
                parent_seal_cid=plan.parent_seal_cid,
            )
    raise PlannerError(f"unit {proof_unit_id!r} is not in the plan")


__all__ = (
    "AGGREGATION_SCHEMA",
    "EVIDENCE_SUBSET",
    "FULL_FALLBACK_REASONS",
    "PLAN_SCHEMA",
    "REUSE_EVIDENCE_SUBSET",
    "AggregationPlan",
    "FinalAcceptancePolicy",
    "IncrementalProofPlan",
    "ParentSealContext",
    "PlanMode",
    "PlannedUnit",
    "PlannerError",
    "PlanningRequest",
    "ProofReuseExplanation",
    "ResourceEstimate",
    "UnitPlanKind",
    "UnitPlanningInput",
    "create_incremental_plan",
    "explain_reuse",
    "plan_incremental_proof",
)
