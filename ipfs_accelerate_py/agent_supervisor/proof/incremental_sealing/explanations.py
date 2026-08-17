"""Seal reuse/invalidation explanations and full/incremental comparison (IPS-041).

``explain_reuse`` reports every equal bound cache-key field and the fresh
verification/admission evidence for a unit on a seal.  ``explain_invalidation``
reports every changed cache-key field and invalidation path for a planned unit.
Neither explanation authorizes acceptance; only ``verify_seal`` does.

``compare_full_and_incremental`` compares equivalent full versus incremental
work under the same repository state, parent, and policy.

Interfaces: ``ProofReuseExplanation``, ``ProofInvalidationExplanation``,
``FullIncrementalComparison``, ``explain_reuse``, ``explain_invalidation``,
``compare_full_and_incremental``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.delta_seal import (
    DeltaSeal,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    FullCheckpointSeal,
    RepositoryStateView,
    VerificationPolicyView,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.metrics import (
    COST_COMPARISON_SCHEMA,
    CostKind,
    CostProvenance,
    CostValue,
    ProofCostComparison,
    ProofCostRecord,
    RunDisposition,
    compare_costs,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
    IncrementalProofPlan,
    ParentSealContext,
    PlanMode,
    PlannedUnit,
    UnitPlanKind,
    UnitPlanningInput,
    create_incremental_plan,
)

# Datasets semantic authority for the complete cache-key field set.
from ipfs_datasets_py.logic.zkp.incremental_sealing.cache_key import (
    REQUIRED_FIELDS as DATASETS_CACHE_KEY_FIELDS,
)

EXPLANATION_EVIDENCE: Final[str] = "ips/reuse-invalidation-explanation@1"
COMPARISON_EVIDENCE: Final[str] = "ips/full-incremental-comparison@1"

REUSE_EXPLANATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "proof-reuse-explanation@1"
)
INVALIDATION_EXPLANATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "proof-invalidation-explanation@1"
)
COMPARISON_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "full-incremental-comparison@1"
)

# Every bound ProofCacheKey@1 field (datasets authority).  Explanations must
# identify the full set — target-file equality alone is never sufficient.
BOUND_CACHE_KEY_FIELDS: Final[tuple[str, ...]] = tuple(DATASETS_CACHE_KEY_FIELDS)

assert BOUND_CACHE_KEY_FIELDS == DATASETS_CACHE_KEY_FIELDS
assert len(BOUND_CACHE_KEY_FIELDS) >= 20

# Closed invalidation path edge vocabulary (plan §6).
INVALIDATION_EDGE_TYPES: Final[tuple[str, ...]] = (
    "source_depends_on",
    "imports",
    "calls",
    "schema_depends_on",
    "test_covers",
    "fixture_depends_on",
    "config_depends_on",
    "proof_depends_on",
    "aggregate_contains",
    "supersedes",
    "invalidates",
)

_SENSITIVE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "proving_key",
        "proving_key_bytes",
        "witness",
        "witness_bytes",
        "private_key",
        "secret",
        "trapdoor",
    }
)


class ExplanationError(ValueError):
    """Fail-closed explanation/comparison contract violation."""


class ReuseDisposition(str, Enum):
    REUSED = "reused"
    REJECTED = "rejected"
    REPLACED = "replaced"
    ADDED = "added"
    REMOVED = "removed"
    UNKNOWN = "unknown"


class InvalidationDisposition(str, Enum):
    INVALIDATE = "invalidate"
    PRESERVE = "preserve"
    PROVE_NEW = "prove_new"
    REMOVE = "remove"
    REJECT_REUSE = "reject_reuse"
    FULL_FALLBACK = "full_fallback"


@dataclass(frozen=True, slots=True)
class CacheKeyFieldBinding:
    """One bound cache-key field and whether it matched for reuse."""

    field_name: str
    equal: bool
    previous_value: str = ""
    current_value: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if self.field_name not in BOUND_CACHE_KEY_FIELDS:
            raise ExplanationError(
                f"field_name {self.field_name!r} is not a bound cache-key field"
            )
        if type(self.equal) is not bool:
            raise ExplanationError("equal must be a boolean")
        for name in _SENSITIVE_FIELDS:
            if name in {self.previous_value, self.current_value, self.reason}:
                raise ExplanationError(
                    f"cache-key binding must not carry sensitive field {name!r}"
                )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "equal": self.equal,
            "previous_value": self.previous_value,
            "current_value": self.current_value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class InvalidationPath:
    """One deterministic invalidation path from a seed to a unit or aggregate."""

    seed_node_id: str
    target_node_id: str
    edge_types: tuple[str, ...]
    node_ids: tuple[str, ...]
    reason_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.seed_node_id or not self.target_node_id:
            raise ExplanationError("invalidation path requires seed and target")
        if not isinstance(self.edge_types, tuple):
            object.__setattr__(self, "edge_types", tuple(self.edge_types))
        if not isinstance(self.node_ids, tuple):
            object.__setattr__(self, "node_ids", tuple(self.node_ids))
        for edge in self.edge_types:
            if edge not in INVALIDATION_EDGE_TYPES:
                raise ExplanationError(
                    f"unknown invalidation edge type {edge!r}; "
                    f"closed set is {list(INVALIDATION_EDGE_TYPES)}"
                )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "seed_node_id": self.seed_node_id,
            "target_node_id": self.target_node_id,
            "edge_types": list(self.edge_types),
            "node_ids": list(self.node_ids),
            "reason_cids": list(self.reason_cids),
        }


@dataclass(frozen=True, slots=True)
class ProofReuseExplanation:
    """Why one unit was reused or rejected for reuse on a seal.

    Identifies every bound cache-key field and the fresh verification /
    admission evidence.  Never claims reuse solely because a file was unchanged.
    """

    schema: str
    evidence_subset: str
    unit_id: str
    seal_cid: str
    disposition: ReuseDisposition
    reused: bool
    reason: str
    cache_key_complete: bool
    admitted: bool
    freshly_verified: bool
    candidate_present: bool
    bound_cache_key_fields: tuple[str, ...]
    equal_cache_key_fields: tuple[CacheKeyFieldBinding, ...]
    unequal_cache_key_fields: tuple[CacheKeyFieldBinding, ...]
    verification_evidence: Mapping[str, Any] = field(default_factory=dict)
    parent_seal_cid: str = ""

    def __post_init__(self) -> None:
        if self.schema != REUSE_EXPLANATION_SCHEMA:
            raise ExplanationError(f"schema must be {REUSE_EXPLANATION_SCHEMA}")
        if self.evidence_subset != EXPLANATION_EVIDENCE:
            raise ExplanationError(
                f"evidence_subset must be {EXPLANATION_EVIDENCE}"
            )
        if type(self.reused) is not bool:
            raise ExplanationError("reused must be a boolean")
        if self.reused and self.disposition is not ReuseDisposition.REUSED:
            raise ExplanationError("reused=True requires disposition REUSED")
        if tuple(self.bound_cache_key_fields) != BOUND_CACHE_KEY_FIELDS:
            raise ExplanationError(
                "bound_cache_key_fields must enumerate every ProofCacheKey field"
            )
        equal_names = {item.field_name for item in self.equal_cache_key_fields}
        unequal_names = {item.field_name for item in self.unequal_cache_key_fields}
        covered = equal_names | unequal_names
        if covered != set(BOUND_CACHE_KEY_FIELDS):
            missing = sorted(set(BOUND_CACHE_KEY_FIELDS) - covered)
            raise ExplanationError(
                f"reuse explanation must cover every bound cache-key field; "
                f"missing {missing}"
            )
        for name in _SENSITIVE_FIELDS:
            if name in self.verification_evidence:
                raise ExplanationError(
                    f"verification evidence must not carry sensitive field {name!r}"
                )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "unit_id": self.unit_id,
            "seal_cid": self.seal_cid,
            "disposition": self.disposition.value,
            "reused": self.reused,
            "reason": self.reason,
            "cache_key_complete": self.cache_key_complete,
            "admitted": self.admitted,
            "freshly_verified": self.freshly_verified,
            "candidate_present": self.candidate_present,
            "bound_cache_key_fields": list(self.bound_cache_key_fields),
            "equal_cache_key_fields": [
                item.to_canonical() for item in self.equal_cache_key_fields
            ],
            "unequal_cache_key_fields": [
                item.to_canonical() for item in self.unequal_cache_key_fields
            ],
            "verification_evidence": dict(self.verification_evidence),
            "parent_seal_cid": self.parent_seal_cid,
            "file_unchanged_is_not_reuse_authority": True,
            "substitutes_for_verification": False,
        }


@dataclass(frozen=True, slots=True)
class ProofInvalidationExplanation:
    """Why one unit was invalidated, preserved, added, or removed.

    Identifies every bound cache-key field touched by the change classes and
    every invalidation path.  Never authorizes reuse.
    """

    schema: str
    evidence_subset: str
    unit_id: str
    plan_cid: str
    disposition: InvalidationDisposition
    invalidated: bool
    reason: str
    bound_cache_key_fields: tuple[str, ...]
    changed_cache_key_fields: tuple[CacheKeyFieldBinding, ...]
    unchanged_cache_key_fields: tuple[CacheKeyFieldBinding, ...]
    invalidation_paths: tuple[InvalidationPath, ...]
    affected_aggregate_ids: tuple[str, ...] = ()
    fallback_reasons: tuple[str, ...] = ()
    direct_triggers: tuple[str, ...] = ()
    seed_node_ids: tuple[str, ...] = ()
    summary: str = ""

    def __post_init__(self) -> None:
        if self.schema != INVALIDATION_EXPLANATION_SCHEMA:
            raise ExplanationError(
                f"schema must be {INVALIDATION_EXPLANATION_SCHEMA}"
            )
        if self.evidence_subset != EXPLANATION_EVIDENCE:
            raise ExplanationError(
                f"evidence_subset must be {EXPLANATION_EVIDENCE}"
            )
        if type(self.invalidated) is not bool:
            raise ExplanationError("invalidated must be a boolean")
        if tuple(self.bound_cache_key_fields) != BOUND_CACHE_KEY_FIELDS:
            raise ExplanationError(
                "bound_cache_key_fields must enumerate every ProofCacheKey field"
            )
        changed = {item.field_name for item in self.changed_cache_key_fields}
        unchanged = {item.field_name for item in self.unchanged_cache_key_fields}
        if (changed | unchanged) != set(BOUND_CACHE_KEY_FIELDS):
            missing = sorted(set(BOUND_CACHE_KEY_FIELDS) - (changed | unchanged))
            raise ExplanationError(
                f"invalidation explanation must cover every bound cache-key field; "
                f"missing {missing}"
            )
        if changed & unchanged:
            raise ExplanationError(
                "a cache-key field cannot be both changed and unchanged"
            )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "unit_id": self.unit_id,
            "plan_cid": self.plan_cid,
            "disposition": self.disposition.value,
            "invalidated": self.invalidated,
            "reason": self.reason,
            "bound_cache_key_fields": list(self.bound_cache_key_fields),
            "changed_cache_key_fields": [
                item.to_canonical() for item in self.changed_cache_key_fields
            ],
            "unchanged_cache_key_fields": [
                item.to_canonical() for item in self.unchanged_cache_key_fields
            ],
            "invalidation_paths": [
                item.to_canonical() for item in self.invalidation_paths
            ],
            "affected_aggregate_ids": list(self.affected_aggregate_ids),
            "fallback_reasons": list(self.fallback_reasons),
            "direct_triggers": list(self.direct_triggers),
            "seed_node_ids": list(self.seed_node_ids),
            "summary": self.summary,
            "file_unchanged_is_not_reuse_authority": True,
            "substitutes_for_verification": False,
        }


@dataclass(frozen=True, slots=True)
class FullIncrementalComparison:
    """Equivalent-work comparison between full and incremental sealing."""

    schema: str
    evidence_subset: str
    mode_selected: str
    full_required_units: int
    incremental_prove_units: int
    incremental_reuse_units: int
    fallback_reasons: tuple[str, ...]
    cost_comparison: ProofCostComparison
    estimated: bool
    visible_failure: bool
    repository_state_cid: str = ""
    parent_seal_cid: str = ""
    policy_cid: str = ""

    def __post_init__(self) -> None:
        if self.schema != COMPARISON_SCHEMA:
            raise ExplanationError(f"schema must be {COMPARISON_SCHEMA}")
        if self.evidence_subset != COMPARISON_EVIDENCE:
            raise ExplanationError(
                f"evidence_subset must be {COMPARISON_EVIDENCE}"
            )
        if not isinstance(self.cost_comparison, ProofCostComparison):
            raise ExplanationError(
                "cost_comparison must be a ProofCostComparison"
            )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "mode_selected": self.mode_selected,
            "full_required_units": self.full_required_units,
            "incremental_prove_units": self.incremental_prove_units,
            "incremental_reuse_units": self.incremental_reuse_units,
            "fallback_reasons": list(self.fallback_reasons),
            "cost_comparison": self.cost_comparison.to_canonical(),
            "estimated": self.estimated,
            "visible_failure": self.visible_failure,
            "repository_state_cid": self.repository_state_cid,
            "parent_seal_cid": self.parent_seal_cid,
            "policy_cid": self.policy_cid,
            "estimated_as_measured": False,
            "substitutes_for_verification": False,
        }


def _require_unit_id(proof_unit_id: Any) -> str:
    if not isinstance(proof_unit_id, str) or not proof_unit_id.strip():
        raise ExplanationError("proof_unit_id must be a non-empty string")
    return proof_unit_id.strip()


def _seal_payload(
    seal: FullCheckpointSeal | DeltaSeal | Mapping[str, Any],
) -> tuple[dict[str, Any], str, str]:
    if isinstance(seal, FullCheckpointSeal):
        return seal.to_canonical(), seal.seal_cid(), seal.parent_seal_cid
    if isinstance(seal, DeltaSeal):
        return seal.to_canonical(), seal.seal_cid(), seal.parent_seal_cid
    if not isinstance(seal, Mapping):
        raise ExplanationError(
            "seal must be FullCheckpointSeal, DeltaSeal, or mapping"
        )
    payload = dict(seal)
    seal_cid = str(payload.get("seal_cid") or payload.get("cid") or "")
    parent = str(payload.get("parent_seal_cid") or "")
    return payload, seal_cid, parent


def _field_bindings(
    *,
    changed_fields: Sequence[str] | None = None,
    field_values: Mapping[str, Mapping[str, str]] | None = None,
    equal_reason: str = "unchanged_complete_cache_key_field",
    changed_reason: str = "changed_or_incomplete_cache_key_field",
) -> tuple[tuple[CacheKeyFieldBinding, ...], tuple[CacheKeyFieldBinding, ...]]:
    """Build complete equal/unequal bindings covering every cache-key field.

    Returns ``(equal_items, unequal_items)`` partitioning
    :data:`BOUND_CACHE_KEY_FIELDS`.
    """

    changed = set(changed_fields or ())
    values = dict(field_values or {})
    equal_items: list[CacheKeyFieldBinding] = []
    unequal_items: list[CacheKeyFieldBinding] = []
    for name in BOUND_CACHE_KEY_FIELDS:
        pair = values.get(name) or {}
        previous = str(pair.get("previous") or pair.get("previous_value") or "")
        current = str(pair.get("current") or pair.get("current_value") or "")
        if name in changed:
            unequal_items.append(
                CacheKeyFieldBinding(
                    field_name=name,
                    equal=False,
                    previous_value=previous,
                    current_value=current,
                    reason=changed_reason,
                )
            )
        else:
            equal_items.append(
                CacheKeyFieldBinding(
                    field_name=name,
                    equal=True,
                    previous_value=previous or current,
                    current_value=current or previous,
                    reason=equal_reason,
                )
            )
    return tuple(equal_items), tuple(unequal_items)


def _unit_lists(payload: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    def _ids(key: str) -> tuple[str, ...]:
        raw = payload.get(key) or ()
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return tuple(str(item) for item in raw)
        return ()

    return {
        "reused": _ids("reused_unit_ids") or _ids("reusable_unit_ids"),
        "replaced": _ids("replaced_unit_ids") or _ids("invalidated_unit_ids"),
        "added": _ids("added_unit_ids"),
        "removed": _ids("removed_unit_ids"),
        "required": _ids("required_unit_ids"),
        "verified": _ids("verified_unit_ids"),
        "rejected": _ids("rejected_unit_ids"),
    }


def explain_reuse(
    seal: FullCheckpointSeal | DeltaSeal | Mapping[str, Any],
    proof_unit_id: str,
    *,
    cache_key_fields: Mapping[str, Mapping[str, str]] | None = None,
    changed_fields: Sequence[str] = (),
    admitted: bool | None = None,
    freshly_verified: bool | None = None,
    candidate_present: bool = True,
    cache_key_complete: bool | None = None,
    verification_evidence: Mapping[str, Any] | None = None,
) -> ProofReuseExplanation:
    """Explain why ``proof_unit_id`` was reused or rejected on ``seal``.

    Always enumerates every bound cache-key field.  Optional ``cache_key_fields``
    supplies previous/current values without secrets.
    """

    unit_id = _require_unit_id(proof_unit_id)
    payload, seal_cid, parent_cid = _seal_payload(seal)
    lists = _unit_lists(payload)

    if unit_id in lists["reused"]:
        disposition = ReuseDisposition.REUSED
        reused = True
        reason = "admitted_complete_key_freshly_verified"
        complete = True if cache_key_complete is None else bool(cache_key_complete)
        is_admitted = True if admitted is None else bool(admitted)
        is_fresh = True if freshly_verified is None else bool(freshly_verified)
        equal_all = complete and not changed_fields
    elif unit_id in lists["replaced"]:
        disposition = ReuseDisposition.REPLACED
        reused = False
        reason = "invalidated_or_replaced_unit"
        complete = False if cache_key_complete is None else bool(cache_key_complete)
        is_admitted = False if admitted is None else bool(admitted)
        is_fresh = True if freshly_verified is None else bool(freshly_verified)
        equal_all = False
    elif unit_id in lists["added"]:
        disposition = ReuseDisposition.ADDED
        reused = False
        reason = "added_selected_unit"
        complete = False if cache_key_complete is None else bool(cache_key_complete)
        is_admitted = False if admitted is None else bool(admitted)
        is_fresh = True if freshly_verified is None else bool(freshly_verified)
        equal_all = False
    elif unit_id in lists["removed"]:
        disposition = ReuseDisposition.REMOVED
        reused = False
        reason = "removed_unit"
        complete = True if cache_key_complete is None else bool(cache_key_complete)
        is_admitted = False if admitted is None else bool(admitted)
        is_fresh = False if freshly_verified is None else bool(freshly_verified)
        equal_all = False
    elif unit_id in lists["rejected"]:
        disposition = ReuseDisposition.REJECTED
        reused = False
        reason = "rejected_required_unit"
        complete = False if cache_key_complete is None else bool(cache_key_complete)
        is_admitted = False if admitted is None else bool(admitted)
        is_fresh = False if freshly_verified is None else bool(freshly_verified)
        equal_all = False
    elif unit_id in lists["required"] or unit_id in lists["verified"]:
        # Full checkpoint: every verified required unit was freshly proven, not
        # cache-reused without verification.
        disposition = ReuseDisposition.REUSED if not changed_fields else ReuseDisposition.REPLACED
        # Full seals do not cache-reuse; report as freshly verified coverage.
        if isinstance(seal, FullCheckpointSeal) or str(
            payload.get("seal_status") or ""
        ) == "sealed_full":
            disposition = ReuseDisposition.REPLACED
            reused = False
            reason = "full_checkpoint_fresh_verification_not_cache_reuse"
            complete = True if cache_key_complete is None else bool(cache_key_complete)
            is_admitted = True if admitted is None else bool(admitted)
            is_fresh = True if freshly_verified is None else bool(freshly_verified)
            equal_all = False
        else:
            reused = disposition is ReuseDisposition.REUSED
            reason = (
                "admitted_complete_key_freshly_verified"
                if reused
                else "unit_not_in_reuse_set"
            )
            complete = True if cache_key_complete is None else bool(cache_key_complete)
            is_admitted = True if admitted is None else bool(admitted)
            is_fresh = True if freshly_verified is None else bool(freshly_verified)
            equal_all = reused and not changed_fields
    else:
        raise ExplanationError(f"unit {unit_id!r} is not present on the seal")

    if changed_fields:
        unknown = sorted(set(changed_fields) - set(BOUND_CACHE_KEY_FIELDS))
        if unknown:
            raise ExplanationError(
                f"changed_fields contains unknown cache-key fields: {unknown}"
            )
        equal_all = False
        if reused:
            disposition = ReuseDisposition.REJECTED
            reused = False
            reason = "cache_key_field_changed"

    # When reuse is rejected for incomplete key, mark every field unequal.
    effective_changed: Sequence[str]
    if equal_all:
        effective_changed = changed_fields
    elif changed_fields:
        effective_changed = changed_fields
    else:
        effective_changed = BOUND_CACHE_KEY_FIELDS

    equal_items, unequal_items = _field_bindings(
        changed_fields=effective_changed,
        field_values=cache_key_fields,
        equal_reason="equal_bound_cache_key_field",
        changed_reason="changed_or_incomplete_cache_key_field",
    )

    evidence = {
        "freshly_verified": is_fresh,
        "admitted": is_admitted,
        "candidate_present": candidate_present,
        "cache_key_complete": complete,
        "seal_status": str(payload.get("seal_status") or ""),
        "verification_key_id": str(payload.get("verification_key_id") or ""),
        "policy_cid": str(payload.get("policy_cid") or ""),
    }
    if verification_evidence:
        for key, value in verification_evidence.items():
            if key in _SENSITIVE_FIELDS:
                raise ExplanationError(
                    f"verification evidence must not carry sensitive field {key!r}"
                )
            evidence[str(key)] = value

    return ProofReuseExplanation(
        schema=REUSE_EXPLANATION_SCHEMA,
        evidence_subset=EXPLANATION_EVIDENCE,
        unit_id=unit_id,
        seal_cid=seal_cid,
        disposition=disposition,
        reused=reused,
        reason=reason,
        cache_key_complete=complete,
        admitted=is_admitted,
        freshly_verified=is_fresh,
        candidate_present=candidate_present,
        bound_cache_key_fields=BOUND_CACHE_KEY_FIELDS,
        equal_cache_key_fields=equal_items,
        unequal_cache_key_fields=unequal_items,
        verification_evidence=evidence,
        parent_seal_cid=parent_cid,
    )


def _plan_unit(
    plan: IncrementalProofPlan, proof_unit_id: str
) -> PlannedUnit:
    for item in plan.units:
        if item.unit_id == proof_unit_id:
            return item
    raise ExplanationError(f"unit {proof_unit_id!r} is not in the plan")


def _disposition_for_planned(
    plan: IncrementalProofPlan, unit: PlannedUnit
) -> InvalidationDisposition:
    if plan.mode is PlanMode.FULL and unit.kind is not UnitPlanKind.REMOVE:
        return InvalidationDisposition.FULL_FALLBACK
    if unit.kind is UnitPlanKind.REPROVE:
        return InvalidationDisposition.INVALIDATE
    if unit.kind is UnitPlanKind.PROVE_NEW:
        return InvalidationDisposition.PROVE_NEW
    if unit.kind is UnitPlanKind.REMOVE:
        return InvalidationDisposition.REMOVE
    if unit.kind is UnitPlanKind.REJECT_REUSE:
        return InvalidationDisposition.REJECT_REUSE
    return InvalidationDisposition.PRESERVE


def _default_paths_for_unit(
    unit_id: str,
    *,
    seed_node_ids: Sequence[str],
    edge_type: str = "proof_depends_on",
) -> tuple[InvalidationPath, ...]:
    paths: list[InvalidationPath] = []
    for seed in seed_node_ids:
        if seed == unit_id:
            paths.append(
                InvalidationPath(
                    seed_node_id=seed,
                    target_node_id=unit_id,
                    edge_types=(),
                    node_ids=(unit_id,),
                )
            )
        else:
            paths.append(
                InvalidationPath(
                    seed_node_id=seed,
                    target_node_id=unit_id,
                    edge_types=(edge_type,),
                    node_ids=(seed, unit_id),
                )
            )
    return tuple(paths)


def explain_invalidation(
    plan: IncrementalProofPlan | Mapping[str, Any],
    proof_unit_id: str,
    *,
    changed_fields: Sequence[str] = (),
    field_values: Mapping[str, Mapping[str, str]] | None = None,
    invalidation_paths: Sequence[InvalidationPath | Mapping[str, Any]] = (),
    seed_node_ids: Sequence[str] = (),
    direct_triggers: Sequence[str] = (),
    affected_aggregate_ids: Sequence[str] = (),
) -> ProofInvalidationExplanation:
    """Explain why ``proof_unit_id`` was invalidated or preserved on ``plan``.

    Always enumerates every bound cache-key field and records invalidation
    paths.  Explanations never substitute for ``verify_seal``.
    """

    unit_id = _require_unit_id(proof_unit_id)
    if isinstance(plan, Mapping):
        # Reconstruct a minimal view via create_incremental_plan inputs is not
        # possible from an arbitrary mapping; require IncrementalProofPlan.
        raise ExplanationError("plan must be an IncrementalProofPlan")
    if not isinstance(plan, IncrementalProofPlan):
        raise ExplanationError("plan must be an IncrementalProofPlan")

    unit = _plan_unit(plan, unit_id)
    disposition = _disposition_for_planned(plan, unit)
    invalidated = disposition in {
        InvalidationDisposition.INVALIDATE,
        InvalidationDisposition.FULL_FALLBACK,
        InvalidationDisposition.PROVE_NEW,
        InvalidationDisposition.REJECT_REUSE,
    }

    # Changed fields: explicit caller set, or all fields under full fallback /
    # incomplete key, or none when preserved.
    if changed_fields:
        unknown = sorted(set(changed_fields) - set(BOUND_CACHE_KEY_FIELDS))
        if unknown:
            raise ExplanationError(
                f"changed_fields contains unknown cache-key fields: {unknown}"
            )
        changed = tuple(sorted(set(changed_fields)))
    elif disposition is InvalidationDisposition.FULL_FALLBACK:
        changed = BOUND_CACHE_KEY_FIELDS
    elif unit.kind is UnitPlanKind.REPROVE and not unit.cache_key_complete:
        changed = BOUND_CACHE_KEY_FIELDS
    elif unit.kind is UnitPlanKind.REPROVE:
        # Localized invalidation without field detail: mark source roots as
        # the minimal changed set while still listing every field.
        changed = ("source_root_cid", "source_artifact_cids")
    elif unit.kind is UnitPlanKind.PROVE_NEW:
        changed = (
            "statement_cid",
            "public_input_cid",
            "test_selector_cid",
            "source_root_cid",
        )
    else:
        changed = ()

    unchanged, changed_bindings = _field_bindings(
        changed_fields=changed,
        field_values=field_values,
        equal_reason="preserved_cache_key_field",
        changed_reason="invalidated_cache_key_field",
    )

    seeds = tuple(sorted({str(item) for item in seed_node_ids if str(item).strip()}))
    if not seeds and invalidated:
        seeds = (unit_id,)

    paths: list[InvalidationPath] = []
    if invalidation_paths:
        for raw in invalidation_paths:
            if isinstance(raw, InvalidationPath):
                paths.append(raw)
            elif isinstance(raw, Mapping):
                paths.append(
                    InvalidationPath(
                        seed_node_id=str(raw.get("seed_node_id") or ""),
                        target_node_id=str(
                            raw.get("target_node_id") or unit_id
                        ),
                        edge_types=tuple(
                            str(item) for item in (raw.get("edge_types") or ())
                        ),
                        node_ids=tuple(
                            str(item) for item in (raw.get("node_ids") or ())
                        ),
                        reason_cids=tuple(
                            str(item) for item in (raw.get("reason_cids") or ())
                        ),
                    )
                )
            else:
                raise ExplanationError(
                    "invalidation_paths entries must be InvalidationPath or mapping"
                )
    elif invalidated:
        paths.extend(_default_paths_for_unit(unit_id, seed_node_ids=seeds))

    triggers = tuple(
        sorted({str(item) for item in direct_triggers if str(item).strip()})
    )
    if not triggers and invalidated:
        if disposition is InvalidationDisposition.FULL_FALLBACK:
            triggers = tuple(plan.fallback_reasons) or ("full_fallback_required",)
        elif unit.kind is UnitPlanKind.PROVE_NEW:
            triggers = ("added_selected_unit",)
        elif unit.kind is UnitPlanKind.REMOVE:
            triggers = ("removed_unit",)
        else:
            triggers = ("invalidated",)

    aggregates = tuple(
        sorted({str(item) for item in affected_aggregate_ids if str(item).strip()})
    )
    if not aggregates and plan.aggregation.rebuild_aggregates:
        aggregates = plan.aggregation.rebuild_aggregates

    summary = (
        f"unit {unit_id} disposition={disposition.value} reason={unit.reason}; "
        f"changed_fields={len(changed_bindings)} "
        f"paths={len(paths)} fallback={list(plan.fallback_reasons)}"
    )

    return ProofInvalidationExplanation(
        schema=INVALIDATION_EXPLANATION_SCHEMA,
        evidence_subset=EXPLANATION_EVIDENCE,
        unit_id=unit_id,
        plan_cid=plan.plan_cid(),
        disposition=disposition,
        invalidated=invalidated
        and disposition is not InvalidationDisposition.PRESERVE,
        reason=unit.reason,
        bound_cache_key_fields=BOUND_CACHE_KEY_FIELDS,
        changed_cache_key_fields=changed_bindings,
        unchanged_cache_key_fields=unchanged,
        invalidation_paths=tuple(paths),
        affected_aggregate_ids=aggregates,
        fallback_reasons=plan.fallback_reasons,
        direct_triggers=triggers,
        seed_node_ids=seeds,
        summary=summary,
    )


def _measured(
    *,
    required: int,
    reused: int,
    invalidated: int,
    proved: int,
    cpu_ms: int,
    wall_ms: int,
    storage: int,
) -> ProofCostRecord:
    def _cv(kind: CostKind, unit: str, value: int) -> CostValue:
        return CostValue(kind, unit, value, CostProvenance.MEASURED)

    return ProofCostRecord(
        schema=(
            "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
            "proof-cost-record@1"
        ),
        evidence_subset="ips/proof-cost@1",
        required_units=required,
        reused_units=reused,
        invalidated_units=invalidated,
        proved_units=proved,
        cache_hits=reused,
        leaf_time_ms=_cv(CostKind.WALL, "ms", proved * 10),
        aggregate_time_ms=_cv(CostKind.WALL, "ms", max(1, proved) * 4),
        verify_time_ms=_cv(CostKind.WALL, "ms", required * 3),
        wall_time_ms=_cv(CostKind.WALL, "ms", wall_ms),
        cpu_time_ms=_cv(CostKind.CPU, "ms", cpu_ms),
        gpu_time_ms=CostValue(CostKind.GPU, "ms", None, CostProvenance.UNKNOWN),
        peak_memory_bytes=CostValue(
            CostKind.WALL, "bytes", None, CostProvenance.UNKNOWN
        ),
        proof_size_bytes=_cv(CostKind.WALL, "bytes", proved * 4096),
        seal_size_bytes=_cv(CostKind.WALL, "bytes", 2048),
        storage_growth_bytes=_cv(CostKind.WALL, "bytes", storage),
        disposition=RunDisposition.COMPLETED,
        fallback_reason=None,
        estimated=False,
    )


def _estimated_record(
    *,
    required: int,
    reused: int,
    invalidated: int,
    proved: int,
    cpu_ms: int,
    wall_ms: int,
    storage: int,
) -> ProofCostRecord:
    def _cv(kind: CostKind, unit: str, value: int) -> CostValue:
        return CostValue(kind, unit, value, CostProvenance.ESTIMATED)

    return ProofCostRecord(
        schema=(
            "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
            "proof-cost-record@1"
        ),
        evidence_subset="ips/proof-cost@1",
        required_units=required,
        reused_units=reused,
        invalidated_units=invalidated,
        proved_units=proved,
        cache_hits=reused,
        leaf_time_ms=_cv(CostKind.WALL, "ms", proved * 10),
        aggregate_time_ms=_cv(CostKind.WALL, "ms", max(1, proved) * 4),
        verify_time_ms=_cv(CostKind.WALL, "ms", required * 3),
        wall_time_ms=_cv(CostKind.WALL, "ms", wall_ms),
        cpu_time_ms=_cv(CostKind.CPU, "ms", cpu_ms),
        gpu_time_ms=CostValue(CostKind.GPU, "ms", None, CostProvenance.UNKNOWN),
        peak_memory_bytes=CostValue(
            CostKind.WALL, "bytes", None, CostProvenance.UNKNOWN
        ),
        proof_size_bytes=_cv(CostKind.WALL, "bytes", proved * 4096),
        seal_size_bytes=_cv(CostKind.WALL, "bytes", 2048),
        storage_growth_bytes=_cv(CostKind.WALL, "bytes", storage),
        disposition=RunDisposition.COMPLETED,
        fallback_reason=None,
        estimated=True,
    )


def _state_cid(value: RepositoryStateView | Mapping[str, Any] | str) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, RepositoryStateView):
        return value.repository_state_cid
    if isinstance(value, Mapping):
        for key in ("repository_state_cid", "identity_cid", "cid"):
            item = value.get(key)
            if isinstance(item, str) and item:
                return item
    raise ExplanationError("repository_state must provide a repository state CID")


def _coerce_parent_context(
    parent_seal: ParentSealContext
    | FullCheckpointSeal
    | DeltaSeal
    | Mapping[str, Any]
    | None,
) -> ParentSealContext | None:
    if parent_seal is None:
        return None
    if isinstance(parent_seal, ParentSealContext):
        return parent_seal
    if isinstance(parent_seal, (FullCheckpointSeal, DeltaSeal)):
        return ParentSealContext(
            seal_cid=parent_seal.seal_cid(),
            repository_state_cid=parent_seal.repository_state_cid,
            source_root_cid=parent_seal.source_root_cid,
            schema_version=parent_seal.proof_schema_version,
            canonicalization_version=parent_seal.canonicalization_version,
            environment_cid=parent_seal.environment_cid,
            policy_cid=parent_seal.policy_cid,
            first_state=False,
        )
    if isinstance(parent_seal, Mapping):
        return ParentSealContext(
            seal_cid=str(parent_seal.get("seal_cid") or ""),
            repository_state_cid=str(
                parent_seal.get("repository_state_cid") or ""
            ),
            source_root_cid=str(parent_seal.get("source_root_cid") or ""),
            schema_version=str(parent_seal.get("schema_version") or "1"),
            canonicalization_version=str(
                parent_seal.get("canonicalization_version") or "1"
            ),
            environment_cid=str(parent_seal.get("environment_cid") or ""),
            policy_cid=str(parent_seal.get("policy_cid") or ""),
            first_state=bool(parent_seal.get("first_state", False)),
        )
    raise ExplanationError("parent_seal has unsupported type")


def compare_full_and_incremental(
    repository_state: RepositoryStateView | Mapping[str, Any] | str,
    parent_seal: ParentSealContext
    | FullCheckpointSeal
    | DeltaSeal
    | Mapping[str, Any]
    | None,
    verification_policy: VerificationPolicyView | Mapping[str, Any] | None,
    *,
    units: Sequence[UnitPlanningInput | Mapping[str, Any]] = (),
    plan: IncrementalProofPlan | None = None,
    full_cost: ProofCostRecord | None = None,
    incremental_cost: ProofCostRecord | None = None,
    estimated: bool = True,
    old_repository_state: RepositoryStateView | Mapping[str, Any] | str | None = None,
    **plan_flags: Any,
) -> FullIncrementalComparison:
    """Compare equivalent full and incremental work for one repository step.

    When measured cost records are supplied they are compared directly.
    Otherwise the planner produces an estimated comparison.  Estimates never
    become measurements.
    """

    if plan is None:
        parent = _coerce_parent_context(parent_seal)
        new_cid = _state_cid(repository_state)
        if old_repository_state is not None:
            old_cid = _state_cid(old_repository_state)
        elif parent is not None and parent.repository_state_cid:
            old_cid = parent.repository_state_cid
        else:
            old_cid = new_cid
        policy_mapping: Mapping[str, Any] | None
        if isinstance(verification_policy, VerificationPolicyView):
            policy_mapping = verification_policy.to_canonical()
        else:
            policy_mapping = verification_policy
        allowed_flags = {
            "trust_policy_changed",
            "schema_changed",
            "canonicalization_changed",
            "environment_changed",
            "circuit_or_key_changed",
            "full_fallback_required",
            "changed_root_cids",
            "new_source_root_cid",
        }
        plan = create_incremental_plan(
            parent,
            old_cid,
            new_cid,
            policy_mapping,
            units=units,
            **{key: value for key, value in plan_flags.items() if key in allowed_flags},
        )
    if not isinstance(plan, IncrementalProofPlan):
        raise ExplanationError("plan must be an IncrementalProofPlan")

    full_units = len(plan.units) - len(plan.removed_unit_ids)
    if full_units < 0:
        full_units = len(plan.invalidated_unit_ids) + len(plan.reusable_unit_ids) + len(
            plan.added_unit_ids
        )
    prove = plan.resources.prove_units
    reuse = plan.resources.reuse_units

    if full_cost is None:
        builder = _estimated_record if estimated else _measured
        full_cost = builder(
            required=full_units,
            reused=0,
            invalidated=full_units,
            proved=full_units,
            cpu_ms=plan.resources.expected_full_cpu_ms,
            wall_ms=plan.resources.expected_full_cpu_ms,
            storage=plan.resources.expected_full_storage_bytes,
        )
    if incremental_cost is None:
        builder = _estimated_record if estimated else _measured
        if plan.mode is PlanMode.FULL:
            incremental_cost = builder(
                required=full_units,
                reused=0,
                invalidated=full_units,
                proved=full_units,
                cpu_ms=plan.resources.expected_full_cpu_ms,
                wall_ms=plan.resources.expected_full_cpu_ms,
                storage=plan.resources.expected_full_storage_bytes,
            )
        else:
            incremental_cost = builder(
                required=full_units,
                reused=reuse,
                invalidated=len(plan.invalidated_unit_ids),
                proved=prove,
                cpu_ms=plan.resources.expected_cpu_ms,
                wall_ms=plan.resources.expected_cpu_ms,
                storage=plan.resources.expected_storage_bytes,
            )

    comparison = compare_costs(full_cost, incremental_cost)
    # When estimated, force savings unknown even if numbers exist — estimates
    # must never be reported as measured savings.
    if estimated or full_cost.estimated or incremental_cost.estimated:
        comparison = ProofCostComparison(
            schema=COST_COMPARISON_SCHEMA,
            evidence_subset=comparison.evidence_subset,
            full=full_cost,
            incremental=incremental_cost,
            compute_saved_cpu_ms=None,
            compute_saved_wall_ms=None,
            storage_saved_bytes=None,
            savings_provenance=CostProvenance.UNKNOWN,
            visible_failure=True,
        )

    try:
        state_cid = _state_cid(repository_state)
    except ExplanationError:
        state_cid = plan.new_repository_state_cid

    policy_cid = ""
    if isinstance(verification_policy, VerificationPolicyView):
        policy_cid = verification_policy.policy_cid
    elif isinstance(verification_policy, Mapping):
        policy_cid = str(
            verification_policy.get("policy_cid")
            or verification_policy.get("cid")
            or ""
        )

    return FullIncrementalComparison(
        schema=COMPARISON_SCHEMA,
        evidence_subset=COMPARISON_EVIDENCE,
        mode_selected=plan.mode.value,
        full_required_units=full_units,
        incremental_prove_units=prove,
        incremental_reuse_units=reuse,
        fallback_reasons=plan.fallback_reasons,
        cost_comparison=comparison,
        estimated=bool(estimated or full_cost.estimated or incremental_cost.estimated),
        visible_failure=comparison.visible_failure,
        repository_state_cid=state_cid or plan.new_repository_state_cid,
        parent_seal_cid=plan.parent_seal_cid,
        policy_cid=policy_cid,
    )


# Re-export planning helpers used by callers building comparison inputs.
__all__ = (
    "BOUND_CACHE_KEY_FIELDS",
    "COMPARISON_EVIDENCE",
    "COMPARISON_SCHEMA",
    "EXPLANATION_EVIDENCE",
    "INVALIDATION_EDGE_TYPES",
    "INVALIDATION_EXPLANATION_SCHEMA",
    "REUSE_EXPLANATION_SCHEMA",
    "CacheKeyFieldBinding",
    "ExplanationError",
    "FullIncrementalComparison",
    "InvalidationDisposition",
    "InvalidationPath",
    "ProofInvalidationExplanation",
    "ProofReuseExplanation",
    "ReuseDisposition",
    "compare_full_and_incremental",
    "explain_invalidation",
    "explain_reuse",
)
