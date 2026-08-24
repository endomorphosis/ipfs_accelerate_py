"""Fail-closed qualification gates for CASF promotion, rollback, and quarantine.

This module is deliberately a pure, non-authoritative boundary.  It turns a
bounded population of independently-produced receipts into a deterministic
*recommendation* for the typed state owner.  In particular, it never opens a
database, starts a process, writes an outbox event, or changes a federation
state.  A caller must reverify a permitted decision at the authoritative
state-owner boundary before it can promote, roll back, or quarantine anything.

Interface: ``FederationPromotionGate@1``
Evidence: ``casf/promotion-decision@1``
"""

# Python 3.8 remains supported by the package.
# ruff: noqa: UP042

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..task_sources.control_plane_contracts import content_identity
from .contracts import (
    FederationAuthorityError,
    FederationBoundsError,
    FederationContractError,
    FederationSecretError,
    UnknownNormativeFieldError,
)

FEDERATION_PROMOTION_GATE_INTERFACE: Final[str] = "FederationPromotionGate@1"
QUALIFICATION_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/qualification-identity@1"
)
GATE_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation/gate-evidence@1"
)
PROMOTION_DECISION_SCHEMA: Final[str] = "casf/promotion-decision@1"
ROLLBACK_DECISION_SCHEMA: Final[str] = "casf/rollback-decision@1"
QUARANTINE_DECISION_SCHEMA: Final[str] = "casf/quarantine-decision@1"

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+\-=]{0,511}")
_OID = re.compile(r"[0-9a-f]{40}")
_CONTENT_REF = re.compile(r"(?:sha256:[0-9a-f]{64}|b[a-z2-7]{20,})")
_SECRET = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\bBearer\s+[A-Za-z0-9._~+/=-]{8,}|\bsk-[A-Za-z0-9_-]{12,})",
    re.IGNORECASE,
)
MAX_GATE_EVIDENCE: Final[int] = 128
MAX_BLOCKERS: Final[int] = 128


class PromotionGateError(FederationContractError):
    """Promotion, rollback, or quarantine evidence is malformed or unsafe."""


class StaleQualificationEvidenceError(PromotionGateError):
    """Evidence does not bind the exact current authoritative identity."""


class MissingQualificationCapabilityError(PromotionGateError):
    """A capability needed by the selected profile is absent or unqualified."""


class GateProfile(str, Enum):
    """Independent promotion profiles; DuckLake never gates the core profile."""

    DUCKDB_QUACK = "duckdb_quack"
    DUCKLAKE = "ducklake"


class GateStatus(str, Enum):
    PASSED = "passed"
    BLOCKED = "blocked"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class EvidenceOrigin(str, Enum):
    """Closed independent evidence producers; models cannot certify a gate."""

    STATE_OWNER = "state_owner"
    VALIDATION_RUNNER = "validation_runner"
    PROOF_RUNNER = "proof_runner"
    BENCHMARK_RUNNER = "benchmark_runner"
    DRIFT_MONITOR = "drift_monitor"
    CHAOS_SUITE = "chaos_suite"
    DUCKLAKE_PROJECTION = "ducklake_projection"


class PromotionGate(str, Enum):
    STATE_OWNER = "state_owner"
    NO_DIRECT_MULTIPROCESS_WRITES = "no_direct_multiprocess_writes"
    NO_STORE_AMBIGUITY = "no_store_ambiguity"
    NO_EVENT_LOSS = "no_event_loss"
    NO_DUPLICATE_EFFECTS = "no_duplicate_effects"
    NO_STALE_FENCE_COMPLETION = "no_stale_fence_completion"
    NO_UNAUTHORIZED_CREATION = "no_unauthorized_creation"
    NO_TENANT_LEAKAGE = "no_tenant_leakage"
    NO_AGENT_SQL = "no_agent_sql"
    NO_SECRET_LEAK = "no_secret_leak"
    EXACT_DESCENDANT_NOTIFICATION = "exact_descendant_notification"
    NO_NOMINATION_OR_STALE_MAP_AUTHORITY = "no_nomination_or_stale_map_authority"
    NO_CYCLE_OR_SHARD_CORRUPTION = "no_cycle_or_shard_corruption"
    IDLE_QUIESCENCE = "idle_quiescence"
    REPLAY_IDEMPOTENCY = "replay_idempotency"
    BENCHMARK_CAPACITY = "benchmark_capacity"
    OWNERSHIP_EFFECT_MERGE_INTEGRITY = "ownership_effect_merge_integrity"
    TOKEN_EFFICIENCY = "token_efficiency"
    DRIFT_FREE = "drift_free"
    FORMAL_PROOFS = "formal_proofs"
    CHAOS_CONTAINMENT = "chaos_containment"
    POST_MERGE_VALIDATION = "post_merge_validation"
    CAPABILITIES = "capabilities"
    DUCKLAKE_HTTPFS_CURRENT = "ducklake_httpfs_current"
    DUCKLAKE_TYPED_CATALOG = "ducklake_typed_catalog"
    DUCKLAKE_RECOVERABLE_PROJECTION = "ducklake_recoverable_projection"
    DUCKLAKE_EVENT_RANGE_BINDING = "ducklake_event_range_binding"
    DUCKLAKE_RECEIPT = "ducklake_receipt"


class DecisionKind(str, Enum):
    PROMOTION = "promotion"
    ROLLBACK = "rollback"
    QUARANTINE = "quarantine"


class DecisionStatus(str, Enum):
    PERMITTED = "permitted"
    BLOCKED = "blocked"


def _decision_prefix(kind: DecisionKind) -> str:
    """Return the content-addressed namespace for one closed decision type."""

    return {
        DecisionKind.PROMOTION: "promotion-decision",
        DecisionKind.ROLLBACK: "rollback-decision",
        DecisionKind.QUARANTINE: "quarantine-decision",
    }[kind]


_CORE_GATES: Final[frozenset[PromotionGate]] = frozenset(
    {
        PromotionGate.STATE_OWNER,
        PromotionGate.NO_DIRECT_MULTIPROCESS_WRITES,
        PromotionGate.NO_STORE_AMBIGUITY,
        PromotionGate.NO_EVENT_LOSS,
        PromotionGate.NO_DUPLICATE_EFFECTS,
        PromotionGate.NO_STALE_FENCE_COMPLETION,
        PromotionGate.NO_UNAUTHORIZED_CREATION,
        PromotionGate.NO_TENANT_LEAKAGE,
        PromotionGate.NO_AGENT_SQL,
        PromotionGate.NO_SECRET_LEAK,
        PromotionGate.EXACT_DESCENDANT_NOTIFICATION,
        PromotionGate.NO_NOMINATION_OR_STALE_MAP_AUTHORITY,
        PromotionGate.NO_CYCLE_OR_SHARD_CORRUPTION,
        PromotionGate.IDLE_QUIESCENCE,
        PromotionGate.REPLAY_IDEMPOTENCY,
        PromotionGate.BENCHMARK_CAPACITY,
        PromotionGate.OWNERSHIP_EFFECT_MERGE_INTEGRITY,
        PromotionGate.TOKEN_EFFICIENCY,
        PromotionGate.DRIFT_FREE,
        PromotionGate.FORMAL_PROOFS,
        PromotionGate.CHAOS_CONTAINMENT,
        PromotionGate.POST_MERGE_VALIDATION,
        PromotionGate.CAPABILITIES,
    }
)
_DUCKLAKE_GATES: Final[frozenset[PromotionGate]] = frozenset(
    {
        PromotionGate.DUCKLAKE_HTTPFS_CURRENT,
        PromotionGate.DUCKLAKE_TYPED_CATALOG,
        PromotionGate.DUCKLAKE_RECOVERABLE_PROJECTION,
        PromotionGate.DUCKLAKE_EVENT_RANGE_BINDING,
        PromotionGate.DUCKLAKE_RECEIPT,
    }
)


def _token(value: Any, name: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise PromotionGateError(f"{name} must be nonempty exact text")
    if _SECRET.search(value):
        raise FederationSecretError(f"{name} contains credential-shaped material")
    if _TOKEN.fullmatch(value) is None:
        raise PromotionGateError(f"{name} is not a compact identity")
    return value


def _oid(value: Any, name: str) -> str:
    value = _token(value, name)
    if _OID.fullmatch(value) is None:
        raise PromotionGateError(f"{name} must be a lowercase 40-hex Git object id")
    return value


def _content_ref(value: Any, name: str) -> str:
    value = _token(value, name)
    if _CONTENT_REF.fullmatch(value) is None:
        raise PromotionGateError(f"{name} must be a CID or sha256 content reference")
    return value


def _closed_mapping(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromotionGateError(f"{label} must be an object")
    unknown = set(value) - fields
    missing = fields - set(value)
    if unknown:
        raise UnknownNormativeFieldError(
            f"{label} has unknown fields: {sorted(str(item) for item in unknown)!r}"
        )
    if missing:
        raise PromotionGateError(
            f"{label} is missing fields: {sorted(str(item) for item in missing)!r}"
        )
    return value


def _canonical_tokens(value: Any, name: str, *, maximum: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PromotionGateError(f"{name} must be an array")
    result = tuple(_token(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(result) > maximum:
        raise FederationBoundsError(f"{name} exceeds its bound")
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise PromotionGateError(f"{name} must be sorted and unique")
    return result


@dataclass(frozen=True)
class QualificationIdentity:
    """All identities which a gate must bind before it can be considered."""

    tenant_id: str
    federation_id: str
    repository_id: str
    revision: str
    tree_id: str
    schema_id: str
    generation_id: str
    policy_id: str
    policy_revision: str
    capability_ids: tuple[str, ...]
    task_id: str
    attempt_id: str
    fence_id: str
    schema: str = QUALIFICATION_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != QUALIFICATION_IDENTITY_SCHEMA:
            raise PromotionGateError("unsupported qualification identity schema")
        for name in (
            "tenant_id", "federation_id", "repository_id", "schema_id",
            "generation_id", "policy_id", "policy_revision", "task_id",
            "attempt_id", "fence_id",
        ):
            object.__setattr__(self, name, _token(getattr(self, name), name))
        object.__setattr__(self, "revision", _oid(self.revision, "revision"))
        object.__setattr__(self, "tree_id", _oid(self.tree_id, "tree_id"))
        capabilities = _canonical_tokens(
            self.capability_ids, "capability_ids", maximum=MAX_GATE_EVIDENCE
        )
        if not capabilities:
            raise PromotionGateError("capability_ids must not be empty")
        object.__setattr__(self, "capability_ids", capabilities)

    @property
    def identity_id(self) -> str:
        return "qualification:" + content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "tenant_id": self.tenant_id,
            "federation_id": self.federation_id,
            "repository_id": self.repository_id,
            "revision": self.revision,
            "tree_id": self.tree_id,
            "schema_id": self.schema_id,
            "generation_id": self.generation_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "capability_ids": list(self.capability_ids),
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "fence_id": self.fence_id,
        }
        if include_identity:
            value["identity_id"] = self.identity_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QualificationIdentity:
        fields = frozenset(
            {
                "schema", "tenant_id", "federation_id", "repository_id",
                "revision", "tree_id", "schema_id", "generation_id", "policy_id",
                "policy_revision", "capability_ids", "task_id", "attempt_id",
                "fence_id", "identity_id",
            }
        )
        value = _closed_mapping(value, fields, "qualification identity")
        try:
            result = cls(
                tenant_id=value["tenant_id"], federation_id=value["federation_id"],
                repository_id=value["repository_id"], revision=value["revision"],
                tree_id=value["tree_id"], schema_id=value["schema_id"],
                generation_id=value["generation_id"], policy_id=value["policy_id"],
                policy_revision=value["policy_revision"],
                capability_ids=tuple(value["capability_ids"]), task_id=value["task_id"],
                attempt_id=value["attempt_id"], fence_id=value["fence_id"], schema=value["schema"],
            )
        except (KeyError, TypeError, PromotionGateError) as exc:
            raise PromotionGateError("qualification identity is malformed") from exc
        if value["identity_id"] != result.identity_id:
            raise PromotionGateError("qualification identity identity mismatches")
        return result


@dataclass(frozen=True)
class GateEvidence:
    """One bounded, independently produced pass or blocker for an exact tree."""

    identity_id: str
    gate: PromotionGate
    status: GateStatus
    receipt_id: str
    origin: EvidenceOrigin
    observed_effects: bool
    model_authored: bool = False
    authority_created: bool = False
    completion_created: bool = False
    schema: str = GATE_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != GATE_EVIDENCE_SCHEMA:
            raise PromotionGateError("unsupported gate evidence schema")
        object.__setattr__(self, "identity_id", _token(self.identity_id, "identity_id"))
        if type(self.gate) is not PromotionGate or type(self.status) is not GateStatus:
            raise PromotionGateError("gate and status must be closed exact enum values")
        if type(self.origin) is not EvidenceOrigin:
            raise PromotionGateError("origin must be a closed exact enum value")
        object.__setattr__(self, "receipt_id", _content_ref(self.receipt_id, "receipt_id"))
        if type(self.observed_effects) is not bool:
            raise PromotionGateError("observed_effects must be boolean")
        if type(self.model_authored) is not bool:
            raise PromotionGateError("model_authored must be boolean")
        if type(self.authority_created) is not bool or type(self.completion_created) is not bool:
            raise PromotionGateError("authority and completion flags must be boolean")
        if self.model_authored or self.authority_created or self.completion_created:
            raise FederationAuthorityError("gate evidence may not manufacture authority")
        if self.status is GateStatus.PASSED and not self.observed_effects:
            raise PromotionGateError("passed gate evidence requires effect observation")

    @property
    def evidence_id(self) -> str:
        return "gate-evidence:" + content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": self.schema, "identity_id": self.identity_id,
            "gate": self.gate.value, "status": self.status.value,
            "receipt_id": self.receipt_id, "origin": self.origin.value,
            "observed_effects": self.observed_effects,
            "model_authored": False, "authority_created": False,
            "completion_created": False,
        }
        if include_identity:
            value["evidence_id"] = self.evidence_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GateEvidence:
        fields = frozenset(
            {
                "schema", "identity_id", "gate", "status", "receipt_id", "origin",
                "observed_effects", "model_authored", "authority_created",
                "completion_created", "evidence_id",
            }
        )
        value = _closed_mapping(value, fields, "gate evidence")
        try:
            result = cls(
                identity_id=value["identity_id"], gate=PromotionGate(value["gate"]),
                status=GateStatus(value["status"]), receipt_id=value["receipt_id"],
                origin=EvidenceOrigin(value["origin"]), observed_effects=value["observed_effects"],
                model_authored=value["model_authored"], authority_created=value["authority_created"],
                completion_created=value["completion_created"], schema=value["schema"],
            )
        except (KeyError, TypeError, ValueError, PromotionGateError) as exc:
            raise PromotionGateError("gate evidence is malformed") from exc
        if value["evidence_id"] != result.evidence_id:
            raise PromotionGateError("gate evidence identity mismatches")
        return result


def required_gates(profile: GateProfile) -> frozenset[PromotionGate]:
    """Return the complete closed conjunction required for one profile."""

    if type(profile) is not GateProfile:
        raise PromotionGateError("profile must be a closed exact enum value")
    return _CORE_GATES | (_DUCKLAKE_GATES if profile is GateProfile.DUCKLAKE else frozenset())


def _evaluate_evidence(
    identity: QualificationIdentity, profile: GateProfile, evidence: tuple[GateEvidence, ...]
) -> tuple[tuple[GateEvidence, ...], tuple[str, ...]]:
    if not isinstance(evidence, tuple) or any(type(item) is not GateEvidence for item in evidence):
        raise PromotionGateError("evidence must be an immutable tuple of exact gate records")
    if len(evidence) > MAX_GATE_EVIDENCE:
        raise FederationBoundsError("evidence exceeds its bound")
    if len({item.gate for item in evidence}) != len(evidence):
        raise PromotionGateError("duplicate evidence for one gate is forbidden")
    if tuple(item.gate.value for item in evidence) != tuple(sorted(item.gate.value for item in evidence)):
        raise PromotionGateError("evidence must be sorted by gate")
    blockers: list[str] = []
    by_gate = {item.gate: item for item in evidence}
    for gate in sorted(required_gates(profile), key=lambda item: item.value):
        item = by_gate.get(gate)
        if item is None:
            blockers.append("missing:" + gate.value)
        elif item.identity_id != identity.identity_id:
            blockers.append("stale_identity:" + gate.value)
        elif item.status is not GateStatus.PASSED:
            blockers.append(item.status.value + ":" + gate.value)
    # Evidence for a foreign profile is a safe diagnostic but cannot make the
    # selected profile pass.  Its status is nevertheless retained in receipts.
    return evidence, tuple(blockers)


@dataclass(frozen=True)
class GateDecision:
    """A non-authoritative result to be reverified by the state owner."""

    kind: DecisionKind
    identity: QualificationIdentity
    profile: GateProfile
    status: DecisionStatus
    evidence: tuple[GateEvidence, ...]
    blockers: tuple[str, ...]
    rollback_target: QualificationIdentity | None = None
    schema: str = PROMOTION_DECISION_SCHEMA

    def __post_init__(self) -> None:
        expected_schema = {
            DecisionKind.PROMOTION: PROMOTION_DECISION_SCHEMA,
            DecisionKind.ROLLBACK: ROLLBACK_DECISION_SCHEMA,
            DecisionKind.QUARANTINE: QUARANTINE_DECISION_SCHEMA,
        }.get(self.kind)
        if expected_schema is None or self.schema != expected_schema:
            raise PromotionGateError("decision schema does not match decision kind")
        if type(self.identity) is not QualificationIdentity:
            raise PromotionGateError("decision requires exact qualification identity")
        if type(self.profile) is not GateProfile or type(self.status) is not DecisionStatus:
            raise PromotionGateError("decision has an invalid closed status or profile")
        _, expected_blockers = _evaluate_evidence(self.identity, self.profile, self.evidence)
        blockers = _canonical_tokens(self.blockers, "blockers", maximum=MAX_BLOCKERS)
        if blockers != expected_blockers:
            raise PromotionGateError("decision blockers do not match evidence")
        if (self.status is DecisionStatus.PERMITTED) != (not blockers):
            raise PromotionGateError("decision status does not match blockers")
        if self.kind is DecisionKind.PROMOTION and self.rollback_target is not None:
            raise PromotionGateError("promotion decision cannot include a rollback target")
        if self.kind is DecisionKind.ROLLBACK:
            if type(self.rollback_target) is not QualificationIdentity:
                raise PromotionGateError("rollback decision requires an exact predecessor target")
            _validate_rollback_target(self.identity, self.rollback_target)
        elif self.rollback_target is not None:
            raise PromotionGateError("only rollback decisions may include rollback targets")

    @property
    def decision_id(self) -> str:
        return _decision_prefix(self.kind) + ":" + content_identity(
            self.to_dict(include_identity=False)
        )

    @property
    def permitted(self) -> bool:
        return self.status is DecisionStatus.PERMITTED

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": self.schema, "kind": self.kind.value,
            "identity": self.identity.to_dict(), "profile": self.profile.value,
            "status": self.status.value, "evidence": [item.to_dict() for item in self.evidence],
            "blockers": list(self.blockers),
            "authoritative_state_changed": False, "authority_created": False,
            "completion_created": False, "upstream_reverification_required": True,
        }
        if self.rollback_target is not None:
            value["rollback_target"] = self.rollback_target.to_dict()
        if include_identity:
            value["decision_id"] = self.decision_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GateDecision:
        base = {
            "schema", "kind", "identity", "profile", "status", "evidence", "blockers",
            "authoritative_state_changed", "authority_created", "completion_created",
            "upstream_reverification_required", "decision_id",
        }
        if not isinstance(value, Mapping):
            raise PromotionGateError("decision must be an object")
        fields = frozenset(base | ({"rollback_target"} if "rollback_target" in value else set()))
        value = _closed_mapping(value, fields, "gate decision")
        if (
            value["authoritative_state_changed"] is not False
            or value["authority_created"] is not False
            or value["completion_created"] is not False
            or value["upstream_reverification_required"] is not True
        ):
            raise FederationAuthorityError("decision has unsafe authority flags")
        try:
            kind = DecisionKind(value["kind"])
            result = cls(
                kind=kind, identity=QualificationIdentity.from_dict(value["identity"]),
                profile=GateProfile(value["profile"]), status=DecisionStatus(value["status"]),
                evidence=tuple(GateEvidence.from_dict(item) for item in value["evidence"]),
                blockers=tuple(value["blockers"]),
                rollback_target=(QualificationIdentity.from_dict(value["rollback_target"])
                                 if "rollback_target" in value else None),
                schema=value["schema"],
            )
        except (KeyError, TypeError, ValueError, PromotionGateError) as exc:
            raise PromotionGateError("gate decision is malformed") from exc
        if value["decision_id"] != result.decision_id:
            raise PromotionGateError("gate decision identity mismatches")
        return result


def _validate_rollback_target(active: QualificationIdentity, predecessor: QualificationIdentity) -> None:
    if active.tenant_id != predecessor.tenant_id or active.repository_id != predecessor.repository_id:
        raise PromotionGateError("rollback target crosses tenant or repository boundary")
    if active.revision == predecessor.revision or active.tree_id == predecessor.tree_id:
        raise PromotionGateError("rollback target must be a distinct predecessor")
    if active.generation_id == predecessor.generation_id:
        raise PromotionGateError("rollback target must restore a predecessor generation")
    if active.fence_id == predecessor.fence_id:
        raise PromotionGateError("rollback target must use a distinct fenced authority")


def _decision(
    kind: DecisionKind,
    identity: QualificationIdentity,
    profile: GateProfile,
    evidence: tuple[GateEvidence, ...],
    *,
    rollback_target: QualificationIdentity | None = None,
    extra_blockers: tuple[str, ...] = (),
) -> GateDecision:
    _, evidence_blockers = _evaluate_evidence(identity, profile, evidence)
    blockers = tuple(sorted(set(evidence_blockers + extra_blockers)))
    return GateDecision(
        kind=kind, identity=identity, profile=profile,
        status=DecisionStatus.PERMITTED if not blockers else DecisionStatus.BLOCKED,
        evidence=evidence, blockers=blockers, rollback_target=rollback_target,
        schema={
            DecisionKind.PROMOTION: PROMOTION_DECISION_SCHEMA,
            DecisionKind.ROLLBACK: ROLLBACK_DECISION_SCHEMA,
            DecisionKind.QUARANTINE: QUARANTINE_DECISION_SCHEMA,
        }[kind],
    )


class FederationPromotionGate:
    """Pure evaluator for exact-current promotion, rollback, and quarantine gates."""

    INTERFACE: ClassVar[str] = FEDERATION_PROMOTION_GATE_INTERFACE

    @staticmethod
    def promote(
        identity: QualificationIdentity, profile: GateProfile, evidence: tuple[GateEvidence, ...]
    ) -> GateDecision:
        if type(identity) is not QualificationIdentity:
            raise PromotionGateError("promotion requires exact qualification identity")
        return _decision(DecisionKind.PROMOTION, identity, profile, evidence)

    @staticmethod
    def rollback(
        active: QualificationIdentity,
        predecessor: QualificationIdentity,
        profile: GateProfile,
        evidence: tuple[GateEvidence, ...],
    ) -> GateDecision:
        if type(active) is not QualificationIdentity or type(predecessor) is not QualificationIdentity:
            raise PromotionGateError("rollback requires exact active and predecessor identities")
        _validate_rollback_target(active, predecessor)
        return _decision(
            DecisionKind.ROLLBACK, active, profile, evidence, rollback_target=predecessor
        )

    @staticmethod
    def quarantine(
        identity: QualificationIdentity, profile: GateProfile, evidence: tuple[GateEvidence, ...]
    ) -> GateDecision:
        if type(identity) is not QualificationIdentity:
            raise PromotionGateError("quarantine requires exact qualification identity")
        # Quarantine is admitted only after an observed blocker.  A clean
        # population cannot be used to quarantine a healthy target.
        _, blockers = _evaluate_evidence(identity, profile, evidence)
        if not blockers:
            raise PromotionGateError("quarantine requires an observed qualification blocker")
        return _decision(DecisionKind.QUARANTINE, identity, profile, evidence)


def validate_current_decision(
    decision: GateDecision,
    *,
    current_revision: str,
    current_tree_id: str,
    current_generation_id: str,
    current_fence_id: str,
    require_permitted: bool = False,
) -> Mapping[str, Any]:
    """Recheck freshness before a state owner considers the recommendation.

    This check intentionally does not apply the requested transition; it is a
    compact input validator for the registered typed state-owner operation.
    """

    if type(decision) is not GateDecision:
        raise StaleQualificationEvidenceError("decision must be an exact gate decision")
    if decision.identity.revision != _oid(current_revision, "current_revision"):
        raise StaleQualificationEvidenceError("decision is bound to a stale revision")
    if decision.identity.tree_id != _oid(current_tree_id, "current_tree_id"):
        raise StaleQualificationEvidenceError("decision is bound to a stale tree")
    if decision.identity.generation_id != _token(current_generation_id, "current_generation_id"):
        raise StaleQualificationEvidenceError("decision is bound to a stale generation")
    if decision.identity.fence_id != _token(current_fence_id, "current_fence_id"):
        raise StaleQualificationEvidenceError("decision is bound to a stale fence")
    if require_permitted and not decision.permitted:
        raise MissingQualificationCapabilityError("decision remains blocked")
    if decision.decision_id != _decision_prefix(decision.kind) + ":" + content_identity(
        decision.to_dict(include_identity=False)
    ):
        raise StaleQualificationEvidenceError("decision identity is invalid")
    return MappingProxyType(
        {
            "schema": "casf/promotion-decision-validation@1",
            "decision_id": decision.decision_id,
            "current_revision_bound": True,
            "current_tree_bound": True,
            "current_generation_bound": True,
            "current_fence_bound": True,
            "permitted": decision.permitted,
            "authoritative_state_changed": False,
            "upstream_reverification_required": True,
        }
    )


def evaluate_promotion(
    identity: QualificationIdentity, profile: GateProfile, evidence: tuple[GateEvidence, ...]
) -> GateDecision:
    """Functional form of :meth:`FederationPromotionGate.promote`."""

    return FederationPromotionGate.promote(identity, profile, evidence)


__all__ = [
    "DecisionKind", "DecisionStatus", "EvidenceOrigin", "FEDERATION_PROMOTION_GATE_INTERFACE",
    "FederationPromotionGate", "GATE_EVIDENCE_SCHEMA", "GateDecision", "GateEvidence",
    "GateProfile", "GateStatus", "MissingQualificationCapabilityError", "PROMOTION_DECISION_SCHEMA",
    "PromotionGate", "PromotionGateError", "QUALIFICATION_IDENTITY_SCHEMA", "QUARANTINE_DECISION_SCHEMA",
    "QualificationIdentity", "ROLLBACK_DECISION_SCHEMA", "StaleQualificationEvidenceError",
    "evaluate_promotion", "required_gates", "validate_current_decision",
]
