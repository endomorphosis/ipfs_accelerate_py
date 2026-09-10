"""Exact state, transition, procedure, and proof reuse gates (SAWM-016).

Interfaces: ``ProgramWorldReuseGate@1``, ``evaluate_program_world_reuse``,
``explain_program_world_reuse``.

Evidence: ``sawm/exact-reuse@1``, ``sawm/reuse-admission@1``.

Accelerator operational reuse admission.  Datasets remains identity/relation
authority; kit remains verified storage; landed proof/receipt-cache and
procedure authorities remain the evidence owners.  This module never
recomputes semantic CIDs, never treats ANN/similarity as reuse, never
validates its own proof/relation/procedure/root, and never admits its own
output.  A cache hit never suppresses current admission or mandatory
validation.

Only current exact identity or independently proof-backed scoped evidence
can grant reuse.  Stale, unsafe, incomplete, or environment-incompatible
evidence fails closed and explains which binding did not hold.

Importing this module performs no I/O, starts no threads, and does not
import datasets, kit, or proof-cache implementations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessError,
    _bool,
    _closed,
    _text,
    validate_opaque_cid,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_receipts import (
    ADAPTER_ID,
    ADAPTER_OWNED_AUTHORITIES,
    ANN_REASON_CODES,
    EVENT_AUTHORITY,
    FORBIDDEN_IDENTITY_FIELDS,
    MERGE_AUTHORITY,
    OPERATIONAL_ACCEPTANCE_AUTHORITIES,
    PROGRAM_WORLD_REUSE_DECISION_INTERFACE,
    VALIDATION_AUTHORITY,
    FreshnessState,
    ProgramWorldAdmissionError,
    ProgramWorldReuseDecision,
    ReuseVerdict,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload


# ---------------------------------------------------------------------------
# Interfaces / schemas / closed vocabularies
# ---------------------------------------------------------------------------

PROGRAM_WORLD_REUSE_GATE_INTERFACE: Final[str] = "ProgramWorldReuseGate@1"
PROGRAM_WORLD_REUSE_KEY_INTERFACE: Final[str] = "ProgramWorldReuseKey@1"
PROGRAM_WORLD_REUSE_REJECTION_INTERFACE: Final[str] = "ProgramWorldReuseRejection@1"
NEGATIVE_MEMORY_RECORD_INTERFACE: Final[str] = "NegativeMemoryRecord@1"
PROGRAM_WORLD_REUSE_EVALUATION_INTERFACE: Final[str] = (
    "ProgramWorldReuseEvaluation@1"
)
PROGRAM_WORLD_REUSE_EXPLANATION_INTERFACE: Final[str] = (
    "ProgramWorldReuseExplanation@1"
)

SAWM_EXACT_REUSE_EVIDENCE: Final[str] = "sawm/exact-reuse@1"
SAWM_REUSE_ADMISSION_EVIDENCE: Final[str] = "sawm/reuse-admission@1"

PROGRAM_WORLD_REUSE_KEY_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-reuse-key@1"
)
PROGRAM_WORLD_REUSE_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-reuse-candidate@1"
)
PROGRAM_WORLD_REUSE_REJECTION_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-reuse-rejection@1"
)
NEGATIVE_MEMORY_RECORD_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-negative-memory@1"
)
PROGRAM_WORLD_REUSE_EVALUATION_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-reuse-evaluation@1"
)
PROGRAM_WORLD_REUSE_EXPLANATION_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-reuse-explanation@1"
)
BINDING_COMPARISON_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-reuse-binding-comparison@1"
)

REUSE_GATE_ID: Final[str] = "ipfs-accelerate.semantic-state.program-world-reuse"
PRODUCER_ID: Final[str] = "program-world-reuse@1"
GATE_VERSION: Final[str] = "1"

MAX_TEXT_CHARS: Final[int] = 512
MAX_COLLECTION_ITEMS: Final[int] = 10_000
MAX_DIAGNOSTIC_CHARS: Final[int] = 1_024
UNBOUND_BINDING_TEXT: Final[str] = "unbound"
EMPTY_COLLECTION_BINDING_TEXT: Final[str] = "empty"

GATE_OWNED_AUTHORITIES: Final[frozenset[str]] = frozenset(
    {
        *ADAPTER_OWNED_AUTHORITIES,
        ADAPTER_ID,
        REUSE_GATE_ID,
        PROGRAM_WORLD_REUSE_GATE_INTERFACE,
        "ProgramWorldReuseGate",
        "evaluate_program_world_reuse",
        "explain_program_world_reuse",
        PRODUCER_ID,
    }
)

BINDING_FIELDS: Final[tuple[str, ...]] = (
    "state_cid",
    "goal_cid",
    "policy_cid",
    "environment_cid",
    "toolchain_cid",
    "obligation_cids",
    "selection_cids",
    "procedure_revision_cid",
    "validation_dependency_cids",
)

SCALAR_BINDING_FIELDS: Final[tuple[str, ...]] = (
    "state_cid",
    "goal_cid",
    "policy_cid",
    "environment_cid",
    "toolchain_cid",
    "procedure_revision_cid",
)

COLLECTION_BINDING_FIELDS: Final[tuple[str, ...]] = (
    "obligation_cids",
    "selection_cids",
    "validation_dependency_cids",
)

REQUIRED_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "state_cid",
    "goal_cid",
    "policy_cid",
    "environment_cid",
    "toolchain_cid",
)
COMPLETE_SCALAR_FIELDS: Final[tuple[str, ...]] = REQUIRED_IDENTITY_FIELDS + (
    "procedure_revision_cid",
)

BINDING_MISMATCH_REASON: Final[Mapping[str, str]] = {
    "state_cid": "state_mismatch",
    "goal_cid": "goal_mismatch",
    "policy_cid": "policy_mismatch",
    "environment_cid": "environment_mismatch",
    "toolchain_cid": "toolchain_mismatch",
    "obligation_cids": "obligation_mismatch",
    "selection_cids": "selection_mismatch",
    "procedure_revision_cid": "procedure_revision_mismatch",
    "validation_dependency_cids": "validation_dependency_mismatch",
}

STANDING_LIMITATIONS: Final[tuple[str, ...]] = (
    "cache_hit_never_suppresses_current_admission",
    "gate_cannot_admit_own_output",
    "reuse_is_proposal_until_supervisor_admission",
    "similarity_is_not_reuse",
)


class ReuseGateDisposition(str, Enum):
    REUSE = "reuse"
    REJECT = "reject"
    ABSTAIN = "abstain"
    UNAVAILABLE = "unavailable"
    RAW_FALLBACK = "raw_fallback"


class ReuseEvidenceKind(str, Enum):
    EXACT_IDENTITY = "exact_identity"
    PROOF_BACKED_SCOPE = "proof_backed_scope"
    PROOF_CACHE = "proof_cache"
    ABSTRACT_STATE = "abstract_state"
    SIMILARITY = "similarity"
    RAW_SOURCE = "raw_source"
    UNAVAILABLE = "unavailable"


class AbstractionKind(str, Enum):
    RAW = "raw"
    ABSTRACT = "abstract"
    UNSPECIFIED = "unspecified"


class NegativeMemoryStatus(str, Enum):
    NONE = "none"
    RECORDED = "recorded"
    HIT = "hit"
    INVALIDATED = "invalidated"


class ReuseReasonCode(str, Enum):
    EXACT_IDENTITY_MATCH = "exact_identity_match"
    PROOF_BACKED_SCOPED_IDENTITY = "proof_backed_scoped_identity"
    NO_EXACT_IDENTITY_MATCH = "no_exact_identity_match"
    BINDING_MISMATCH = "binding_mismatch"
    STATE_MISMATCH = "state_mismatch"
    GOAL_MISMATCH = "goal_mismatch"
    POLICY_MISMATCH = "policy_mismatch"
    ENVIRONMENT_MISMATCH = "environment_mismatch"
    TOOLCHAIN_MISMATCH = "toolchain_mismatch"
    OBLIGATION_MISMATCH = "obligation_mismatch"
    SELECTION_MISMATCH = "selection_mismatch"
    PROCEDURE_REVISION_MISMATCH = "procedure_revision_mismatch"
    VALIDATION_DEPENDENCY_MISMATCH = "validation_dependency_mismatch"
    INCOMPLETE_BINDINGS = "incomplete_bindings"
    INCOMPLETE_CANDIDATE = "incomplete_candidate"
    STALE_BINDING = "stale_binding"
    FRESHNESS_UNKNOWN = "freshness_unknown"
    FRESHNESS_UNAVAILABLE = "freshness_unavailable"
    UNSAFE_ABSTRACTION = "unsafe_abstraction"
    RAW_SOURCE_FALLBACK = "raw_source_fallback"
    SIMILARITY_IS_NOT_REUSE = "similarity_is_not_reuse"
    ANN_NOT_AUTHORITATIVE = "ann_not_authoritative"
    PROOF_CACHE_REQUIRES_FRESH_ADMISSION = "proof_cache_requires_fresh_admission"
    PROOF_SELF_ADMISSION_FORBIDDEN = "proof_self_admission_forbidden"
    ENVIRONMENT_INCOMPATIBLE = "environment_incompatible"
    NEGATIVE_MEMORY_HIT = "negative_memory_hit"
    SIMULATED_NOT_LIVE = "simulated_not_live"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    SCOPE_ASSUMPTION_MISMATCH = "scope_assumption_mismatch"
    RELATION_SCOPE_MISMATCH = "relation_scope_mismatch"
    PROOF_BACKING_INCOMPLETE = "proof_backing_incomplete"
    IMPORT_FAILED = "import_failed"
    DATASETS_UNAVAILABLE = "datasets_unavailable"
    PROOF_AUTHORITY_UNAVAILABLE = "proof_authority_unavailable"
    PROCEDURE_AUTHORITY_UNAVAILABLE = "procedure_authority_unavailable"


REUSE_GATE_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    item.value for item in ReuseGateDisposition
)
REUSE_EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in ReuseEvidenceKind
)
ABSTRACTION_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in AbstractionKind
)
NEGATIVE_MEMORY_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in NegativeMemoryStatus
)
REUSE_REASON_CODES: Final[frozenset[str]] = frozenset(
    item.value for item in ReuseReasonCode
)
FRESHNESS_STATES: Final[frozenset[str]] = frozenset(
    item.value for item in FreshnessState
)

_FRESHNESS_REJECTION: Final[Mapping[str, str]] = {
    FreshnessState.STALE.value: ReuseReasonCode.STALE_BINDING.value,
    FreshnessState.UNKNOWN.value: ReuseReasonCode.FRESHNESS_UNKNOWN.value,
    FreshnessState.UNAVAILABLE.value: ReuseReasonCode.FRESHNESS_UNAVAILABLE.value,
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProgramWorldReuseError(HarnessError):
    """Closed reuse-gate contract violation."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bounded_text(value: Any, name: str, *, limit: int = MAX_TEXT_CHARS) -> str:
    text = _text(value, name)
    if len(text) > limit:
        raise ProgramWorldReuseError(f"{name} exceeds {limit} characters")
    return text


def _optional_bounded_text(
    value: Any, name: str, *, limit: int = MAX_TEXT_CHARS
) -> str | None:
    if value is None or value == "":
        return None
    return _bounded_text(value, name, limit=limit)


def _optional_cid(value: Any, name: str) -> str | None:
    if value is None or value == "":
        return None
    return validate_opaque_cid(value, name)


def _enum_value(value: Any, allowed: frozenset[str], name: str) -> str:
    text = _bounded_text(getattr(value, "value", value), name)
    if text not in allowed:
        raise ProgramWorldReuseError(f"{name} must be one of {sorted(allowed)}")
    return text


def _unique_sorted_cids(values: Iterable[Any], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ProgramWorldReuseError(f"{name} must be a sequence of CIDs")
    items = [validate_opaque_cid(item, name) for item in values]
    if len(items) > MAX_COLLECTION_ITEMS:
        raise ProgramWorldReuseError(f"{name} exceeds its item bound")
    ordered = tuple(sorted(items))
    if len(ordered) != len(set(ordered)):
        raise ProgramWorldReuseError(f"{name} must not contain duplicates")
    return ordered


def _unique_sorted_texts(values: Iterable[Any], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ProgramWorldReuseError(f"{name} must be a sequence")
    items = [_bounded_text(item, name) for item in values]
    if len(items) > MAX_COLLECTION_ITEMS:
        raise ProgramWorldReuseError(f"{name} exceeds its item bound")
    ordered = tuple(sorted(items))
    if len(ordered) != len(set(ordered)):
        raise ProgramWorldReuseError(f"{name} must not contain duplicates")
    return ordered


def _reject_forbidden_identity_fields(mapping: Mapping[str, Any], name: str) -> None:
    if not isinstance(mapping, Mapping):
        raise ProgramWorldReuseError(f"{name} must be an object")
    forbidden = set(mapping) & FORBIDDEN_IDENTITY_FIELDS
    if forbidden:
        raise ProgramWorldReuseError(
            f"{name} rejects ANN/score identity fields {sorted(forbidden)}"
        )


def _receipt_cid(payload: Mapping[str, Any]) -> str:
    _reject_forbidden_identity_fields(payload, "identity_payload")
    return cid_for_payload(payload)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProgramWorldReuseError(f"{name} must be an object")
    _reject_forbidden_identity_fields(value, name)
    return value


def _limitations_for(
    extra: Iterable[str] = (),
    *,
    relation: bool = False,
    raw_fallback: bool = False,
) -> tuple[str, ...]:
    items = list(STANDING_LIMITATIONS)
    items.extend(extra)
    if relation:
        items.append("relation_reuse_requires_explicit_scope")
    if raw_fallback:
        items.append("raw_source_fallback")
    return _unique_sorted_texts(items, "limitation")


def _mismatch_reason(fields: Sequence[str]) -> str:
    if len(fields) == 1:
        return BINDING_MISMATCH_REASON[fields[0]]
    return ReuseReasonCode.BINDING_MISMATCH.value


def _forbid_self_authority(authority: str | None, *, role: str) -> None:
    if authority is None:
        return
    if authority in GATE_OWNED_AUTHORITIES:
        raise ProgramWorldAdmissionError(
            f"{role} {authority!r} cannot admit or prove reuse for this gate"
        )


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseKey:
    """Exact reuse identity over current operational bindings.

    Completeness requires state, goal, policy, environment, toolchain,
    procedure revision, and the obligation/selection/validation collections
    (collections may be empty; scalar CIDs may not).
    """

    state_cid: str
    goal_cid: str
    policy_cid: str
    environment_cid: str
    toolchain_cid: str
    procedure_revision_cid: str | None = None
    obligation_cids: Sequence[str] = ()
    selection_cids: Sequence[str] = ()
    validation_dependency_cids: Sequence[str] = ()
    scope_cid: str | None = None
    assumption_cids: Sequence[str] = ()

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_REUSE_KEY_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_REUSE_KEY_INTERFACE
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "state_cid",
            "goal_cid",
            "policy_cid",
            "environment_cid",
            "toolchain_cid",
            "procedure_revision_cid",
            "obligation_cids",
            "selection_cids",
            "validation_dependency_cids",
            "scope_cid",
            "assumption_cids",
            "key_cid",
        }
    )

    def __post_init__(self) -> None:
        for name in REQUIRED_IDENTITY_FIELDS:
            object.__setattr__(
                self, name, validate_opaque_cid(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "procedure_revision_cid",
            _optional_cid(self.procedure_revision_cid, "procedure_revision_cid"),
        )
        object.__setattr__(
            self,
            "obligation_cids",
            _unique_sorted_cids(self.obligation_cids, "obligation_cids"),
        )
        object.__setattr__(
            self,
            "selection_cids",
            _unique_sorted_cids(self.selection_cids, "selection_cids"),
        )
        object.__setattr__(
            self,
            "validation_dependency_cids",
            _unique_sorted_cids(
                self.validation_dependency_cids, "validation_dependency_cids"
            ),
        )
        object.__setattr__(self, "scope_cid", _optional_cid(self.scope_cid, "scope_cid"))
        object.__setattr__(
            self,
            "assumption_cids",
            _unique_sorted_cids(self.assumption_cids, "assumption_cids"),
        )

    @property
    def complete(self) -> bool:
        return all(getattr(self, name) for name in COMPLETE_SCALAR_FIELDS)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "state_cid": self.state_cid,
            "goal_cid": self.goal_cid,
            "policy_cid": self.policy_cid,
            "environment_cid": self.environment_cid,
            "toolchain_cid": self.toolchain_cid,
            "procedure_revision_cid": self.procedure_revision_cid,
            "obligation_cids": list(self.obligation_cids),
            "selection_cids": list(self.selection_cids),
            "validation_dependency_cids": list(self.validation_dependency_cids),
            "scope_cid": self.scope_cid,
            "assumption_cids": list(self.assumption_cids),
        }

    @property
    def key_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def binding_value(self, field: str) -> Any:
        if field not in BINDING_FIELDS:
            raise ProgramWorldReuseError(f"unknown binding field {field!r}")
        return getattr(self, field)

    def mismatched_bindings(self, other: "ProgramWorldReuseKey") -> tuple[str, ...]:
        mismatched: list[str] = []
        for field in BINDING_FIELDS:
            if self.binding_value(field) != other.binding_value(field):
                mismatched.append(field)
        return tuple(mismatched)

    def replace(self, **overrides: Any) -> "ProgramWorldReuseKey":
        payload = {
            "state_cid": self.state_cid,
            "goal_cid": self.goal_cid,
            "policy_cid": self.policy_cid,
            "environment_cid": self.environment_cid,
            "toolchain_cid": self.toolchain_cid,
            "procedure_revision_cid": self.procedure_revision_cid,
            "obligation_cids": self.obligation_cids,
            "selection_cids": self.selection_cids,
            "validation_dependency_cids": self.validation_dependency_cids,
            "scope_cid": self.scope_cid,
            "assumption_cids": self.assumption_cids,
        }
        payload.update(overrides)
        return ProgramWorldReuseKey(**payload)

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["key_cid"] = self.key_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgramWorldReuseKey":
        payload = _closed(_mapping(data, "ProgramWorldReuseKey"), cls._FIELDS, cls.__name__)
        claimed = payload.pop("key_cid")
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReuseError("unsupported reuse-key schema")
        if payload.pop("interface") != cls.INTERFACE:
            raise ProgramWorldReuseError("unsupported reuse-key interface")
        result = cls(**payload)
        if result.key_cid != claimed:
            raise ProgramWorldReuseError("key_cid does not rehash")
        return result

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProgramWorldReuseKey":
        if "key_cid" in data and "schema" in data:
            return cls.from_dict(data)
        _reject_forbidden_identity_fields(data, "ProgramWorldReuseKey")
        allowed = {
            "state_cid",
            "goal_cid",
            "policy_cid",
            "environment_cid",
            "toolchain_cid",
            "procedure_revision_cid",
            "obligation_cids",
            "selection_cids",
            "validation_dependency_cids",
            "scope_cid",
            "assumption_cids",
        }
        extra = set(data) - allowed
        if extra:
            raise ProgramWorldReuseError(
                f"ProgramWorldReuseKey unknown fields {sorted(extra)}"
            )
        return cls(**{str(key): data[key] for key in data})


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseCandidate:
    """Prior evidence offered for reuse against a current key."""

    key: ProgramWorldReuseKey
    evidence_kind: ReuseEvidenceKind | str = ReuseEvidenceKind.EXACT_IDENTITY
    freshness: FreshnessState | str = FreshnessState.FRESH
    proof_cache_hit: bool = False
    proof_freshly_admitted: bool = False
    proof_receipt_cid: str | None = None
    proof_authority: str | None = None
    independently_proof_backed: bool = False
    scoped_identity_proved: bool = False
    relation_claim_cid: str | None = None
    abstraction_kind: AbstractionKind | str = AbstractionKind.UNSPECIFIED
    abstraction_safe: bool = True
    raw_source_cid: str | None = None
    environment_compatible: bool = True
    simulated: bool = False
    similarity_only: bool = False
    capability_available: bool = True
    capability_reason_code: str = "import_failed"
    capability_surface: str = "datasets"

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_REUSE_CANDIDATE_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "key",
            "evidence_kind",
            "freshness",
            "proof_cache_hit",
            "proof_freshly_admitted",
            "proof_receipt_cid",
            "proof_authority",
            "independently_proof_backed",
            "scoped_identity_proved",
            "relation_claim_cid",
            "abstraction_kind",
            "abstraction_safe",
            "raw_source_cid",
            "environment_compatible",
            "simulated",
            "similarity_only",
            "capability_available",
            "capability_reason_code",
            "capability_surface",
            "candidate_cid",
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.key, ProgramWorldReuseKey):
            if isinstance(self.key, Mapping):
                object.__setattr__(self, "key", ProgramWorldReuseKey.from_mapping(self.key))
            else:
                raise ProgramWorldReuseError("candidate key must be a ProgramWorldReuseKey")
        object.__setattr__(
            self,
            "evidence_kind",
            _enum_value(self.evidence_kind, REUSE_EVIDENCE_KINDS, "evidence_kind"),
        )
        object.__setattr__(
            self, "freshness", _enum_value(self.freshness, FRESHNESS_STATES, "freshness")
        )
        object.__setattr__(
            self,
            "abstraction_kind",
            _enum_value(self.abstraction_kind, ABSTRACTION_KINDS, "abstraction_kind"),
        )
        for flag in (
            "proof_cache_hit",
            "proof_freshly_admitted",
            "independently_proof_backed",
            "scoped_identity_proved",
            "abstraction_safe",
            "environment_compatible",
            "simulated",
            "similarity_only",
            "capability_available",
        ):
            object.__setattr__(self, flag, _bool(getattr(self, flag), flag))
        object.__setattr__(
            self,
            "proof_receipt_cid",
            _optional_cid(self.proof_receipt_cid, "proof_receipt_cid"),
        )
        object.__setattr__(
            self,
            "proof_authority",
            _optional_bounded_text(self.proof_authority, "proof_authority"),
        )
        object.__setattr__(
            self,
            "relation_claim_cid",
            _optional_cid(self.relation_claim_cid, "relation_claim_cid"),
        )
        object.__setattr__(
            self, "raw_source_cid", _optional_cid(self.raw_source_cid, "raw_source_cid")
        )
        object.__setattr__(
            self,
            "capability_reason_code",
            _bounded_text(self.capability_reason_code, "capability_reason_code"),
        )
        object.__setattr__(
            self,
            "capability_surface",
            _bounded_text(self.capability_surface, "capability_surface"),
        )
        if self.evidence_kind == ReuseEvidenceKind.SIMILARITY.value:
            object.__setattr__(self, "similarity_only", True)
        if self.evidence_kind == ReuseEvidenceKind.PROOF_CACHE.value:
            object.__setattr__(self, "proof_cache_hit", True)
        if self.evidence_kind == ReuseEvidenceKind.UNAVAILABLE.value:
            object.__setattr__(self, "capability_available", False)
        if self.evidence_kind == ReuseEvidenceKind.ABSTRACT_STATE.value:
            object.__setattr__(self, "abstraction_kind", AbstractionKind.ABSTRACT.value)
        if self.proof_cache_hit and not self.independently_proof_backed:
            # A cache hit is proof-shaped evidence; it still cannot self-admit.
            object.__setattr__(self, "independently_proof_backed", True)
        if self.similarity_only and self.evidence_kind != ReuseEvidenceKind.SIMILARITY.value:
            object.__setattr__(self, "evidence_kind", ReuseEvidenceKind.SIMILARITY.value)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "key": self.key.to_dict(),
            "evidence_kind": self.evidence_kind,
            "freshness": self.freshness,
            "proof_cache_hit": self.proof_cache_hit,
            "proof_freshly_admitted": self.proof_freshly_admitted,
            "proof_receipt_cid": self.proof_receipt_cid,
            "proof_authority": self.proof_authority,
            "independently_proof_backed": self.independently_proof_backed,
            "scoped_identity_proved": self.scoped_identity_proved,
            "relation_claim_cid": self.relation_claim_cid,
            "abstraction_kind": self.abstraction_kind,
            "abstraction_safe": self.abstraction_safe,
            "raw_source_cid": self.raw_source_cid,
            "environment_compatible": self.environment_compatible,
            "simulated": self.simulated,
            "similarity_only": self.similarity_only,
            "capability_available": self.capability_available,
            "capability_reason_code": self.capability_reason_code,
            "capability_surface": self.capability_surface,
        }

    @property
    def candidate_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def replace(self, **overrides: Any) -> "ProgramWorldReuseCandidate":
        payload = {
            "key": self.key,
            "evidence_kind": self.evidence_kind,
            "freshness": self.freshness,
            "proof_cache_hit": self.proof_cache_hit,
            "proof_freshly_admitted": self.proof_freshly_admitted,
            "proof_receipt_cid": self.proof_receipt_cid,
            "proof_authority": self.proof_authority,
            "independently_proof_backed": self.independently_proof_backed,
            "scoped_identity_proved": self.scoped_identity_proved,
            "relation_claim_cid": self.relation_claim_cid,
            "abstraction_kind": self.abstraction_kind,
            "abstraction_safe": self.abstraction_safe,
            "raw_source_cid": self.raw_source_cid,
            "environment_compatible": self.environment_compatible,
            "simulated": self.simulated,
            "similarity_only": self.similarity_only,
            "capability_available": self.capability_available,
            "capability_reason_code": self.capability_reason_code,
            "capability_surface": self.capability_surface,
        }
        payload.update(overrides)
        return ProgramWorldReuseCandidate(**payload)

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["candidate_cid"] = self.candidate_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgramWorldReuseCandidate":
        payload = _closed(
            _mapping(data, "ProgramWorldReuseCandidate"), cls._FIELDS, cls.__name__
        )
        claimed = payload.pop("candidate_cid")
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReuseError("unsupported reuse-candidate schema")
        payload["key"] = ProgramWorldReuseKey.from_dict(payload["key"])
        result = cls(**payload)
        if result.candidate_cid != claimed:
            raise ProgramWorldReuseError("candidate_cid does not rehash")
        return result


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseRejection:
    """Typed fail-closed rejection of a reuse candidate."""

    reason_code: str
    diagnostic: str
    current_key_cid: str
    mismatched_bindings: Sequence[str] = ()
    candidate_key_cid: str | None = None
    candidate_cid: str | None = None
    fallback: str = "none"
    raw_source_cid: str | None = None
    negative_memory_cid: str | None = None

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_REUSE_REJECTION_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_REUSE_REJECTION_INTERFACE
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "reason_code",
            "diagnostic",
            "current_key_cid",
            "mismatched_bindings",
            "candidate_key_cid",
            "candidate_cid",
            "fallback",
            "raw_source_cid",
            "negative_memory_cid",
            "rejection_cid",
        }
    )
    FALLBACKS: ClassVar[frozenset[str]] = frozenset(
        {"none", "raw_source", "unavailable"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reason_code", _bounded_text(self.reason_code, "reason_code")
        )
        if self.reason_code not in REUSE_REASON_CODES and self.reason_code not in ANN_REASON_CODES:
            raise ProgramWorldReuseError(
                f"reason_code must be one of {sorted(REUSE_REASON_CODES)}"
            )
        object.__setattr__(
            self,
            "diagnostic",
            _bounded_text(self.diagnostic, "diagnostic", limit=MAX_DIAGNOSTIC_CHARS),
        )
        object.__setattr__(
            self,
            "current_key_cid",
            validate_opaque_cid(self.current_key_cid, "current_key_cid"),
        )
        bindings = _unique_sorted_texts(self.mismatched_bindings, "mismatched_bindings")
        extra = set(bindings) - set(BINDING_FIELDS)
        if extra:
            raise ProgramWorldReuseError(
                f"mismatched_bindings contains unknown fields {sorted(extra)}"
            )
        object.__setattr__(self, "mismatched_bindings", bindings)
        object.__setattr__(
            self,
            "candidate_key_cid",
            _optional_cid(self.candidate_key_cid, "candidate_key_cid"),
        )
        object.__setattr__(
            self, "candidate_cid", _optional_cid(self.candidate_cid, "candidate_cid")
        )
        object.__setattr__(
            self, "fallback", _enum_value(self.fallback, self.FALLBACKS, "fallback")
        )
        object.__setattr__(
            self, "raw_source_cid", _optional_cid(self.raw_source_cid, "raw_source_cid")
        )
        object.__setattr__(
            self,
            "negative_memory_cid",
            _optional_cid(self.negative_memory_cid, "negative_memory_cid"),
        )
        if self.fallback == "raw_source" and self.raw_source_cid is None:
            raise ProgramWorldReuseError("raw_source fallback requires raw_source_cid")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "reason_code": self.reason_code,
            "diagnostic": self.diagnostic,
            "current_key_cid": self.current_key_cid,
            "mismatched_bindings": list(self.mismatched_bindings),
            "candidate_key_cid": self.candidate_key_cid,
            "candidate_cid": self.candidate_cid,
            "fallback": self.fallback,
            "raw_source_cid": self.raw_source_cid,
            "negative_memory_cid": self.negative_memory_cid,
        }

    @property
    def rejection_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["rejection_cid"] = self.rejection_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgramWorldReuseRejection":
        payload = _closed(
            _mapping(data, "ProgramWorldReuseRejection"), cls._FIELDS, cls.__name__
        )
        claimed = payload.pop("rejection_cid")
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReuseError("unsupported reuse-rejection schema")
        if payload.pop("interface") != cls.INTERFACE:
            raise ProgramWorldReuseError("unsupported reuse-rejection interface")
        result = cls(**payload)
        if result.rejection_cid != claimed:
            raise ProgramWorldReuseError("rejection_cid does not rehash")
        return result


@dataclass(frozen=True, slots=True)
class NegativeMemoryRecord:
    """Scoped rejection memory. Bindings that change invalidate the record."""

    key_cid: str
    candidate_cid: str
    reason_code: str
    mismatched_bindings: Sequence[str] = ()
    scope_cid: str | None = None
    recorded_key: ProgramWorldReuseKey | None = None
    invalidated: bool = False
    invalidation_reason: str | None = None

    SCHEMA: ClassVar[str] = NEGATIVE_MEMORY_RECORD_SCHEMA
    INTERFACE: ClassVar[str] = NEGATIVE_MEMORY_RECORD_INTERFACE
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "key_cid",
            "candidate_cid",
            "reason_code",
            "mismatched_bindings",
            "scope_cid",
            "recorded_key",
            "invalidated",
            "invalidation_reason",
            "memory_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_cid", validate_opaque_cid(self.key_cid, "key_cid"))
        object.__setattr__(
            self, "candidate_cid", validate_opaque_cid(self.candidate_cid, "candidate_cid")
        )
        object.__setattr__(
            self, "reason_code", _bounded_text(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self,
            "mismatched_bindings",
            _unique_sorted_texts(self.mismatched_bindings, "mismatched_bindings"),
        )
        object.__setattr__(self, "scope_cid", _optional_cid(self.scope_cid, "scope_cid"))
        if self.recorded_key is not None and not isinstance(
            self.recorded_key, ProgramWorldReuseKey
        ):
            if isinstance(self.recorded_key, Mapping):
                object.__setattr__(
                    self,
                    "recorded_key",
                    ProgramWorldReuseKey.from_dict(self.recorded_key)
                    if "key_cid" in self.recorded_key
                    else ProgramWorldReuseKey.from_mapping(self.recorded_key),
                )
            else:
                raise ProgramWorldReuseError(
                    "recorded_key must be a ProgramWorldReuseKey"
                )
        if self.recorded_key is not None and self.recorded_key.key_cid != self.key_cid:
            raise ProgramWorldReuseError("recorded_key does not match key_cid")
        object.__setattr__(self, "invalidated", _bool(self.invalidated, "invalidated"))
        object.__setattr__(
            self,
            "invalidation_reason",
            _optional_bounded_text(self.invalidation_reason, "invalidation_reason"),
        )
        if self.invalidated and self.invalidation_reason is None:
            raise ProgramWorldReuseError(
                "invalidated negative memory requires invalidation_reason"
            )
        if not self.invalidated and self.invalidation_reason is not None:
            raise ProgramWorldReuseError(
                "active negative memory cannot carry invalidation_reason"
            )

    @property
    def active(self) -> bool:
        return not self.invalidated

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "key_cid": self.key_cid,
            "candidate_cid": self.candidate_cid,
            "reason_code": self.reason_code,
            "mismatched_bindings": list(self.mismatched_bindings),
            "scope_cid": self.scope_cid,
            "recorded_key": None if self.recorded_key is None else self.recorded_key.to_dict(),
            "invalidated": self.invalidated,
            "invalidation_reason": self.invalidation_reason,
        }

    @property
    def memory_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def invalidate(self, reason: str) -> "NegativeMemoryRecord":
        if self.invalidated:
            return self
        return NegativeMemoryRecord(
            key_cid=self.key_cid,
            candidate_cid=self.candidate_cid,
            reason_code=self.reason_code,
            mismatched_bindings=self.mismatched_bindings,
            scope_cid=self.scope_cid,
            recorded_key=self.recorded_key,
            invalidated=True,
            invalidation_reason=_bounded_text(reason, "invalidation_reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["memory_cid"] = self.memory_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NegativeMemoryRecord":
        payload = _closed(
            _mapping(data, "NegativeMemoryRecord"), cls._FIELDS, cls.__name__
        )
        claimed = payload.pop("memory_cid")
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReuseError("unsupported negative-memory schema")
        if payload.pop("interface") != cls.INTERFACE:
            raise ProgramWorldReuseError("unsupported negative-memory interface")
        recorded = payload.get("recorded_key")
        if recorded is not None:
            payload["recorded_key"] = ProgramWorldReuseKey.from_dict(recorded)
        result = cls(**payload)
        if result.memory_cid != claimed:
            raise ProgramWorldReuseError("memory_cid does not rehash")
        return result


@dataclass(frozen=True, slots=True)
class BindingComparison:
    """One current-versus-candidate binding comparison."""

    field: str
    equal: bool
    current_value: str
    candidate_value: str | None
    reason: str

    SCHEMA: ClassVar[str] = BINDING_COMPARISON_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "field",
            "equal",
            "current_value",
            "candidate_value",
            "reason",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "field", _bounded_text(self.field, "field"))
        if self.field not in BINDING_FIELDS:
            raise ProgramWorldReuseError(f"unknown binding field {self.field!r}")
        object.__setattr__(self, "equal", _bool(self.equal, "equal"))
        object.__setattr__(
            self, "current_value", _bounded_text(self.current_value, "current_value")
        )
        object.__setattr__(
            self,
            "candidate_value",
            _optional_bounded_text(self.candidate_value, "candidate_value"),
        )
        object.__setattr__(self, "reason", _bounded_text(self.reason, "reason"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "field": self.field,
            "equal": self.equal,
            "current_value": self.current_value,
            "candidate_value": self.candidate_value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BindingComparison":
        payload = _closed(
            _mapping(data, "BindingComparison"), cls._FIELDS, cls.__name__
        )
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReuseError("unsupported binding-comparison schema")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseExplanation:
    """Deterministic explanation of a reuse grant or fail-closed rejection."""

    verdict: ReuseGateDisposition | str
    reason_code: str
    diagnostic: str
    current_key_cid: str
    bindings: Sequence[BindingComparison]
    similarity_authoritative: bool = False
    proof_cache_hit: bool = False
    proof_freshly_admitted: bool = False
    negative_memory_status: NegativeMemoryStatus | str = NegativeMemoryStatus.NONE
    raw_source_fallback: bool = False
    raw_source_cid: str | None = None
    mismatched_bindings: Sequence[str] = ()
    limitations: Sequence[str] = ()

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_REUSE_EXPLANATION_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_REUSE_EXPLANATION_INTERFACE
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "verdict",
            "reason_code",
            "diagnostic",
            "current_key_cid",
            "bindings",
            "similarity_authoritative",
            "proof_cache_hit",
            "proof_freshly_admitted",
            "negative_memory_status",
            "raw_source_fallback",
            "raw_source_cid",
            "mismatched_bindings",
            "limitations",
            "explanation_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "verdict", _enum_value(self.verdict, REUSE_GATE_DISPOSITIONS, "verdict")
        )
        object.__setattr__(
            self, "reason_code", _bounded_text(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self,
            "diagnostic",
            _bounded_text(self.diagnostic, "diagnostic", limit=MAX_DIAGNOSTIC_CHARS),
        )
        object.__setattr__(
            self,
            "current_key_cid",
            validate_opaque_cid(self.current_key_cid, "current_key_cid"),
        )
        bindings = tuple(self.bindings)
        normalized: list[BindingComparison] = []
        for item in bindings:
            if isinstance(item, BindingComparison):
                normalized.append(item)
            elif isinstance(item, Mapping):
                normalized.append(BindingComparison.from_dict(item))
            else:
                raise ProgramWorldReuseError("bindings must be BindingComparison records")
        if tuple(item.field for item in normalized) != BINDING_FIELDS:
            raise ProgramWorldReuseError(
                "explanation must compare every reuse-key binding in order"
            )
        object.__setattr__(self, "bindings", tuple(normalized))
        object.__setattr__(
            self,
            "similarity_authoritative",
            _bool(self.similarity_authoritative, "similarity_authoritative"),
        )
        if self.similarity_authoritative:
            raise ProgramWorldReuseError("similarity cannot be authoritative for reuse")
        object.__setattr__(
            self, "proof_cache_hit", _bool(self.proof_cache_hit, "proof_cache_hit")
        )
        object.__setattr__(
            self,
            "proof_freshly_admitted",
            _bool(self.proof_freshly_admitted, "proof_freshly_admitted"),
        )
        object.__setattr__(
            self,
            "negative_memory_status",
            _enum_value(
                self.negative_memory_status,
                NEGATIVE_MEMORY_STATUSES,
                "negative_memory_status",
            ),
        )
        object.__setattr__(
            self,
            "raw_source_fallback",
            _bool(self.raw_source_fallback, "raw_source_fallback"),
        )
        object.__setattr__(
            self, "raw_source_cid", _optional_cid(self.raw_source_cid, "raw_source_cid")
        )
        object.__setattr__(
            self,
            "mismatched_bindings",
            _unique_sorted_texts(self.mismatched_bindings, "mismatched_bindings"),
        )
        object.__setattr__(
            self, "limitations", _unique_sorted_texts(self.limitations, "limitation")
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "verdict": self.verdict,
            "reason_code": self.reason_code,
            "diagnostic": self.diagnostic,
            "current_key_cid": self.current_key_cid,
            "bindings": [item.to_dict() for item in self.bindings],
            "similarity_authoritative": self.similarity_authoritative,
            "proof_cache_hit": self.proof_cache_hit,
            "proof_freshly_admitted": self.proof_freshly_admitted,
            "negative_memory_status": self.negative_memory_status,
            "raw_source_fallback": self.raw_source_fallback,
            "raw_source_cid": self.raw_source_cid,
            "mismatched_bindings": list(self.mismatched_bindings),
            "limitations": list(self.limitations),
        }

    @property
    def explanation_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["explanation_cid"] = self.explanation_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgramWorldReuseExplanation":
        payload = _closed(
            _mapping(data, "ProgramWorldReuseExplanation"), cls._FIELDS, cls.__name__
        )
        claimed = payload.pop("explanation_cid")
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReuseError("unsupported reuse-explanation schema")
        if payload.pop("interface") != cls.INTERFACE:
            raise ProgramWorldReuseError("unsupported reuse-explanation interface")
        payload["bindings"] = [
            BindingComparison.from_dict(item) for item in payload["bindings"]
        ]
        result = cls(**payload)
        if result.explanation_cid != claimed:
            raise ProgramWorldReuseError("explanation_cid does not rehash")
        return result


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseEvaluation:
    """Composite reuse-gate result. Cannot self-admit."""

    verdict: ReuseGateDisposition | str
    reason_code: str
    current_key: ProgramWorldReuseKey
    decision: ProgramWorldReuseDecision
    explanation: ProgramWorldReuseExplanation
    candidate: ProgramWorldReuseCandidate | None = None
    rejection: ProgramWorldReuseRejection | None = None
    negative_memory: NegativeMemoryRecord | None = None
    mismatched_bindings: Sequence[str] = ()
    raw_source_cid: str | None = None
    limitations: Sequence[str] = ()

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_REUSE_EVALUATION_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_REUSE_EVALUATION_INTERFACE
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "verdict",
            "reason_code",
            "current_key",
            "decision",
            "explanation",
            "candidate",
            "rejection",
            "negative_memory",
            "mismatched_bindings",
            "raw_source_cid",
            "limitations",
            "evaluation_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "verdict", _enum_value(self.verdict, REUSE_GATE_DISPOSITIONS, "verdict")
        )
        object.__setattr__(
            self, "reason_code", _bounded_text(self.reason_code, "reason_code")
        )
        if not isinstance(self.current_key, ProgramWorldReuseKey):
            raise ProgramWorldReuseError("current_key must be a ProgramWorldReuseKey")
        if not isinstance(self.decision, ProgramWorldReuseDecision):
            raise ProgramWorldReuseError("decision must be a ProgramWorldReuseDecision")
        if not isinstance(self.explanation, ProgramWorldReuseExplanation):
            raise ProgramWorldReuseError(
                "explanation must be a ProgramWorldReuseExplanation"
            )
        if self.candidate is not None and not isinstance(
            self.candidate, ProgramWorldReuseCandidate
        ):
            raise ProgramWorldReuseError(
                "candidate must be a ProgramWorldReuseCandidate"
            )
        if self.rejection is not None and not isinstance(
            self.rejection, ProgramWorldReuseRejection
        ):
            raise ProgramWorldReuseError(
                "rejection must be a ProgramWorldReuseRejection"
            )
        if self.negative_memory is not None and not isinstance(
            self.negative_memory, NegativeMemoryRecord
        ):
            raise ProgramWorldReuseError(
                "negative_memory must be a NegativeMemoryRecord"
            )
        object.__setattr__(
            self,
            "mismatched_bindings",
            _unique_sorted_texts(self.mismatched_bindings, "mismatched_bindings"),
        )
        object.__setattr__(
            self, "raw_source_cid", _optional_cid(self.raw_source_cid, "raw_source_cid")
        )
        object.__setattr__(
            self, "limitations", _unique_sorted_texts(self.limitations, "limitation")
        )
        if self.verdict == ReuseGateDisposition.REUSE.value:
            if self.decision.verdict != ReuseVerdict.REUSE.value:
                raise ProgramWorldReuseError("reuse evaluation requires reuse decision")
            if self.rejection is not None:
                raise ProgramWorldReuseError("reuse evaluation cannot carry a rejection")
        else:
            if self.decision.verdict == ReuseVerdict.REUSE.value:
                raise ProgramWorldReuseError("non-reuse evaluation cannot carry reuse")
        if self.decision.ann_authoritative:
            raise ProgramWorldReuseError("ANN cannot be authoritative")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "verdict": self.verdict,
            "reason_code": self.reason_code,
            "current_key": self.current_key.to_dict(),
            "decision": self.decision.to_dict(),
            "explanation": self.explanation.to_dict(),
            "candidate": None if self.candidate is None else self.candidate.to_dict(),
            "rejection": None if self.rejection is None else self.rejection.to_dict(),
            "negative_memory": (
                None if self.negative_memory is None else self.negative_memory.to_dict()
            ),
            "mismatched_bindings": list(self.mismatched_bindings),
            "raw_source_cid": self.raw_source_cid,
            "limitations": list(self.limitations),
        }

    @property
    def evaluation_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    @property
    def granted(self) -> bool:
        return self.verdict == ReuseGateDisposition.REUSE.value

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["evaluation_cid"] = self.evaluation_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgramWorldReuseEvaluation":
        payload = _closed(
            _mapping(data, "ProgramWorldReuseEvaluation"), cls._FIELDS, cls.__name__
        )
        claimed = payload.pop("evaluation_cid")
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReuseError("unsupported reuse-evaluation schema")
        if payload.pop("interface") != cls.INTERFACE:
            raise ProgramWorldReuseError("unsupported reuse-evaluation interface")
        payload["current_key"] = ProgramWorldReuseKey.from_dict(payload["current_key"])
        payload["decision"] = ProgramWorldReuseDecision.from_dict(payload["decision"])
        payload["explanation"] = ProgramWorldReuseExplanation.from_dict(
            payload["explanation"]
        )
        if payload["candidate"] is not None:
            payload["candidate"] = ProgramWorldReuseCandidate.from_dict(
                payload["candidate"]
            )
        if payload["rejection"] is not None:
            payload["rejection"] = ProgramWorldReuseRejection.from_dict(
                payload["rejection"]
            )
        if payload["negative_memory"] is not None:
            payload["negative_memory"] = NegativeMemoryRecord.from_dict(
                payload["negative_memory"]
            )
        result = cls(**payload)
        if result.evaluation_cid != claimed:
            raise ProgramWorldReuseError("evaluation_cid does not rehash")
        return result


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def _binding_text(value: Any) -> str:
    # BindingComparison rejects empty strings; missing/empty bindings stay named.
    if value is None:
        return UNBOUND_BINDING_TEXT
    if isinstance(value, tuple):
        if not value:
            return EMPTY_COLLECTION_BINDING_TEXT
        text = ",".join(value)
        if not text:
            return EMPTY_COLLECTION_BINDING_TEXT
        if len(text) > MAX_TEXT_CHARS:
            return cid_for_payload({"bindings": list(value)})
        return text
    text = str(value)
    if not text:
        return UNBOUND_BINDING_TEXT
    return text


def _incomplete_scalar_fields(key: ProgramWorldReuseKey) -> tuple[str, ...]:
    return tuple(name for name in COMPLETE_SCALAR_FIELDS if not getattr(key, name))


def _binding_reason(
    field: str,
    *,
    equal: bool,
    current_raw: Any,
    candidate_raw: Any,
    candidate_absent: bool = False,
) -> str:
    if candidate_absent:
        if field in COMPLETE_SCALAR_FIELDS and not current_raw:
            return "incomplete"
        return "candidate_absent"
    if field in COMPLETE_SCALAR_FIELDS and (not current_raw or not candidate_raw):
        return "incomplete"
    return "equal" if equal else BINDING_MISMATCH_REASON[field]


def _compare_bindings(
    current: ProgramWorldReuseKey,
    candidate: ProgramWorldReuseKey | None,
) -> tuple[BindingComparison, ...]:
    items: list[BindingComparison] = []
    for field in BINDING_FIELDS:
        current_raw = current.binding_value(field)
        current_value = _binding_text(current_raw)
        if candidate is None:
            items.append(
                BindingComparison(
                    field=field,
                    equal=False,
                    current_value=current_value,
                    candidate_value=None,
                    reason=_binding_reason(
                        field,
                        equal=False,
                        current_raw=current_raw,
                        candidate_raw=None,
                        candidate_absent=True,
                    ),
                )
            )
            continue
        candidate_raw = candidate.binding_value(field)
        candidate_value = _binding_text(candidate_raw)
        equal = current_raw == candidate_raw
        items.append(
            BindingComparison(
                field=field,
                equal=equal,
                current_value=current_value,
                candidate_value=candidate_value,
                reason=_binding_reason(
                    field,
                    equal=equal,
                    current_raw=current_raw,
                    candidate_raw=candidate_raw,
                ),
            )
        )
    return tuple(items)


def _scope_matches(current: ProgramWorldReuseKey, candidate: ProgramWorldReuseKey) -> str | None:
    if current.scope_cid != candidate.scope_cid:
        return ReuseReasonCode.RELATION_SCOPE_MISMATCH.value
    if current.assumption_cids != candidate.assumption_cids:
        return ReuseReasonCode.SCOPE_ASSUMPTION_MISMATCH.value
    return None


def _proof_backing_error(candidate: ProgramWorldReuseCandidate) -> str | None:
    if not candidate.independently_proof_backed and not candidate.proof_cache_hit:
        return None
    _forbid_self_authority(candidate.proof_authority, role="proof_authority")
    if candidate.proof_authority is None or candidate.proof_receipt_cid is None:
        return ReuseReasonCode.PROOF_BACKING_INCOMPLETE.value
    if not candidate.proof_freshly_admitted:
        return ReuseReasonCode.PROOF_CACHE_REQUIRES_FRESH_ADMISSION.value
    return None


def _diagnostic(reason_code: str, mismatched: Sequence[str]) -> str:
    if reason_code == ReuseReasonCode.EXACT_IDENTITY_MATCH.value:
        return (
            "Current exact state, goal, policy, environment, toolchain, "
            "obligation, selection, procedure, and validation bindings match."
        )
    if reason_code == ReuseReasonCode.PROOF_BACKED_SCOPED_IDENTITY.value:
        return (
            "Independently proof-backed scoped equality holds under the current "
            "policy, environment, and assumptions."
        )
    if reason_code == ReuseReasonCode.SIMILARITY_IS_NOT_REUSE.value:
        return "Similarity/ANN is context-only and cannot grant reuse."
    if reason_code == ReuseReasonCode.ANN_NOT_AUTHORITATIVE.value:
        return "ANN candidates are advisory and cannot grant reuse."
    if reason_code == ReuseReasonCode.STALE_BINDING.value:
        return "Candidate evidence is stale; current freshness admission revoked reuse."
    if reason_code == ReuseReasonCode.FRESHNESS_UNKNOWN.value:
        return "Candidate freshness is unknown; reuse fails closed."
    if reason_code == ReuseReasonCode.FRESHNESS_UNAVAILABLE.value:
        return "Candidate freshness is unavailable; reuse fails closed."
    if reason_code == ReuseReasonCode.UNSAFE_ABSTRACTION.value:
        return "Abstract-state reuse is unsafe; raw-source fallback remains available if cited."
    if reason_code == ReuseReasonCode.RAW_SOURCE_FALLBACK.value:
        return "Abstract reuse rejected; raw source is the remaining fail-closed fallback."
    if reason_code == ReuseReasonCode.PROOF_CACHE_REQUIRES_FRESH_ADMISSION.value:
        return (
            "Proof-cache presence is not authority; current independent admission "
            "and mandatory validation remain required."
        )
    if reason_code == ReuseReasonCode.ENVIRONMENT_INCOMPATIBLE.value:
        return "Candidate environment or toolchain is incompatible with the current bindings."
    if reason_code == ReuseReasonCode.NEGATIVE_MEMORY_HIT.value:
        return "An active scoped negative-memory record rejects retrying this exact candidate."
    if reason_code == ReuseReasonCode.SIMULATED_NOT_LIVE.value:
        return "Simulated evidence cannot be reused as live authority."
    if reason_code == ReuseReasonCode.INCOMPLETE_BINDINGS.value:
        return "Reuse keys require complete state/goal/policy/environment/toolchain/procedure bindings."
    if reason_code == ReuseReasonCode.INCOMPLETE_CANDIDATE.value:
        return "No candidate evidence was supplied; exact reuse cannot be granted."
    if reason_code == ReuseReasonCode.CAPABILITY_UNAVAILABLE.value:
        return "A required identity, proof, or procedure surface is unavailable."
    if reason_code == ReuseReasonCode.RELATION_SCOPE_MISMATCH.value:
        return "Relation-based reuse is outside the current explicit scope."
    if reason_code == ReuseReasonCode.SCOPE_ASSUMPTION_MISMATCH.value:
        return "Relation-based reuse assumptions do not match the current scope."
    if reason_code == ReuseReasonCode.PROOF_BACKING_INCOMPLETE.value:
        return "Proof-backed reuse requires an independent proof authority and receipt CID."
    if mismatched:
        labels = ", ".join(mismatched)
        return f"Reuse rejected because binding(s) differ: {labels}."
    if reason_code == ReuseReasonCode.NO_EXACT_IDENTITY_MATCH.value:
        return "No exact current identity match and no independently proof-backed scoped evidence."
    return f"Reuse failed closed with reason {reason_code}."


def _decision_verdict(disposition: str) -> str:
    if disposition == ReuseGateDisposition.REUSE.value:
        return ReuseVerdict.REUSE.value
    if disposition == ReuseGateDisposition.UNAVAILABLE.value:
        return ReuseVerdict.UNAVAILABLE.value
    if disposition == ReuseGateDisposition.ABSTAIN.value:
        return ReuseVerdict.ABSTAIN.value
    return ReuseVerdict.REJECT.value


class ProgramWorldReuseGate:
    """Fail-closed exact and proof-backed reuse admission."""

    INTERFACE: Final[str] = PROGRAM_WORLD_REUSE_GATE_INTERFACE

    def __init__(
        self,
        *,
        negative_memory: Sequence[NegativeMemoryRecord] = (),
    ) -> None:
        self._negative_memory: list[NegativeMemoryRecord] = [
            item
            if isinstance(item, NegativeMemoryRecord)
            else NegativeMemoryRecord.from_dict(item)
            for item in negative_memory
        ]

    @property
    def negative_memory(self) -> tuple[NegativeMemoryRecord, ...]:
        return tuple(self._negative_memory)

    def active_negative_memory(self) -> tuple[NegativeMemoryRecord, ...]:
        return tuple(item for item in self._negative_memory if item.active)

    def _find_active_memory(
        self, key_cid: str, candidate_cid: str
    ) -> NegativeMemoryRecord | None:
        for item in self._negative_memory:
            if item.active and item.key_cid == key_cid and item.candidate_cid == candidate_cid:
                return item
        return None

    def invalidate_negative_memory(
        self,
        current: ProgramWorldReuseKey,
        *,
        reason: str = "binding_changed",
    ) -> tuple[NegativeMemoryRecord, ...]:
        current_cid = current.key_cid
        updated: list[NegativeMemoryRecord] = []
        changed: list[NegativeMemoryRecord] = []
        for item in self._negative_memory:
            if item.active and item.key_cid != current_cid:
                invalidated = item.invalidate(reason)
                updated.append(invalidated)
                changed.append(invalidated)
            else:
                updated.append(item)
        self._negative_memory = updated
        return tuple(changed)

    def _invalidate_key(
        self, key_cid: str, *, reason: str
    ) -> tuple[NegativeMemoryRecord, ...]:
        updated: list[NegativeMemoryRecord] = []
        changed: list[NegativeMemoryRecord] = []
        for item in self._negative_memory:
            if item.active and item.key_cid == key_cid:
                invalidated = item.invalidate(reason)
                updated.append(invalidated)
                changed.append(invalidated)
            else:
                updated.append(item)
        self._negative_memory = updated
        return tuple(changed)

    def _remember(
        self,
        current: ProgramWorldReuseKey,
        candidate: ProgramWorldReuseCandidate,
        reason_code: str,
        mismatched: Sequence[str],
    ) -> NegativeMemoryRecord:
        existing = self._find_active_memory(current.key_cid, candidate.candidate_cid)
        if existing is not None:
            return existing
        record = NegativeMemoryRecord(
            key_cid=current.key_cid,
            candidate_cid=candidate.candidate_cid,
            reason_code=reason_code,
            mismatched_bindings=mismatched,
            scope_cid=current.scope_cid,
            recorded_key=current,
        )
        self._negative_memory.append(record)
        return record

    def evaluate(
        self,
        current: ProgramWorldReuseKey | Mapping[str, Any],
        candidate: ProgramWorldReuseCandidate
        | ProgramWorldReuseKey
        | Mapping[str, Any]
        | None = None,
        *,
        admission_authority: str | None = None,
        admission_evidence_cid: str | None = None,
        remember: bool = True,
    ) -> ProgramWorldReuseEvaluation:
        current_key = (
            current
            if isinstance(current, ProgramWorldReuseKey)
            else ProgramWorldReuseKey.from_mapping(current)
        )
        parsed_candidate = _parse_candidate(candidate)
        _forbid_self_authority(admission_authority, role="admission_authority")

        disposition, reason_code, mismatched, memory_status, memory = self._classify(
            current_key, parsed_candidate
        )
        raw_source_cid = (
            None if parsed_candidate is None else parsed_candidate.raw_source_cid
        )
        relation = bool(
            parsed_candidate is not None
            and (
                parsed_candidate.relation_claim_cid is not None
                or parsed_candidate.independently_proof_backed
                or parsed_candidate.scoped_identity_proved
            )
        )
        raw_fallback = disposition == ReuseGateDisposition.RAW_FALLBACK.value
        extra_limitations: list[str] = []
        if disposition == ReuseGateDisposition.UNAVAILABLE.value:
            extra_limitations.append("typed_unavailable")
        if parsed_candidate is not None and parsed_candidate.proof_cache_hit:
            extra_limitations.append("proof_cache_is_not_authority")
        limitations = _limitations_for(
            extra_limitations, relation=relation, raw_fallback=raw_fallback
        )
        diagnostic = _diagnostic(reason_code, mismatched)

        rejection: ProgramWorldReuseRejection | None = None
        if disposition != ReuseGateDisposition.REUSE.value:
            fallback = "none"
            if disposition == ReuseGateDisposition.UNAVAILABLE.value:
                fallback = "unavailable"
            elif raw_fallback:
                fallback = "raw_source"
            if (
                remember
                and parsed_candidate is not None
                and disposition
                in {
                    ReuseGateDisposition.REJECT.value,
                    ReuseGateDisposition.RAW_FALLBACK.value,
                }
                and reason_code != ReuseReasonCode.NEGATIVE_MEMORY_HIT.value
            ):
                memory = self._remember(
                    current_key, parsed_candidate, reason_code, mismatched
                )
                if memory_status == NegativeMemoryStatus.NONE.value:
                    memory_status = NegativeMemoryStatus.RECORDED.value
            rejection = ProgramWorldReuseRejection(
                reason_code=reason_code,
                diagnostic=diagnostic,
                current_key_cid=current_key.key_cid,
                mismatched_bindings=mismatched,
                candidate_key_cid=(
                    None if parsed_candidate is None else parsed_candidate.key.key_cid
                ),
                candidate_cid=(
                    None if parsed_candidate is None else parsed_candidate.candidate_cid
                ),
                fallback=fallback,
                raw_source_cid=raw_source_cid if raw_fallback else None,
                negative_memory_cid=None if memory is None else memory.memory_cid,
            )
        else:
            self._invalidate_key(
                current_key.key_cid, reason="superseded_by_current_reuse"
            )

        explanation = ProgramWorldReuseExplanation(
            verdict=disposition,
            reason_code=reason_code,
            diagnostic=diagnostic,
            current_key_cid=current_key.key_cid,
            bindings=_compare_bindings(
                current_key, None if parsed_candidate is None else parsed_candidate.key
            ),
            similarity_authoritative=False,
            proof_cache_hit=bool(
                parsed_candidate is not None and parsed_candidate.proof_cache_hit
            ),
            proof_freshly_admitted=bool(
                parsed_candidate is not None and parsed_candidate.proof_freshly_admitted
            ),
            negative_memory_status=memory_status,
            raw_source_fallback=raw_fallback,
            raw_source_cid=raw_source_cid if raw_fallback else None,
            mismatched_bindings=mismatched,
            limitations=limitations,
        )
        decision = _build_decision(
            current_key,
            parsed_candidate,
            disposition=disposition,
            reason_code=reason_code,
            limitations=limitations,
            admission_authority=admission_authority,
            admission_evidence_cid=admission_evidence_cid,
            exact_match=disposition == ReuseGateDisposition.REUSE.value,
        )
        return ProgramWorldReuseEvaluation(
            verdict=disposition,
            reason_code=reason_code,
            current_key=current_key,
            candidate=parsed_candidate,
            decision=decision,
            explanation=explanation,
            rejection=rejection,
            negative_memory=memory,
            mismatched_bindings=mismatched,
            raw_source_cid=raw_source_cid if raw_fallback else None,
            limitations=limitations,
        )

    def _classify(
        self,
        current: ProgramWorldReuseKey,
        candidate: ProgramWorldReuseCandidate | None,
    ) -> tuple[str, str, tuple[str, ...], str, NegativeMemoryRecord | None]:
        if candidate is None:
            return (
                ReuseGateDisposition.ABSTAIN.value,
                ReuseReasonCode.NO_EXACT_IDENTITY_MATCH.value,
                (),
                NegativeMemoryStatus.NONE.value,
                None,
            )
        if candidate.similarity_only or candidate.evidence_kind in {
            ReuseEvidenceKind.SIMILARITY.value,
        }:
            return (
                ReuseGateDisposition.REJECT.value,
                ReuseReasonCode.SIMILARITY_IS_NOT_REUSE.value,
                (),
                NegativeMemoryStatus.NONE.value,
                None,
            )
        if candidate.simulated:
            return (
                ReuseGateDisposition.REJECT.value,
                ReuseReasonCode.SIMULATED_NOT_LIVE.value,
                (),
                NegativeMemoryStatus.NONE.value,
                None,
            )
        if not candidate.capability_available:
            unavailable_reason = candidate.capability_reason_code
            if unavailable_reason not in REUSE_REASON_CODES:
                unavailable_reason = ReuseReasonCode.CAPABILITY_UNAVAILABLE.value
            return (
                ReuseGateDisposition.UNAVAILABLE.value,
                unavailable_reason,
                (),
                NegativeMemoryStatus.NONE.value,
                None,
            )
        incomplete_fields = set(_incomplete_scalar_fields(current)) | set(
            _incomplete_scalar_fields(candidate.key)
        )
        incomplete = tuple(
            field for field in BINDING_FIELDS if field in incomplete_fields
        )
        if incomplete:
            return (
                ReuseGateDisposition.REJECT.value,
                ReuseReasonCode.INCOMPLETE_BINDINGS.value,
                incomplete,
                NegativeMemoryStatus.NONE.value,
                None,
            )
        freshness_reason = _FRESHNESS_REJECTION.get(candidate.freshness)
        if freshness_reason is not None:
            return (
                ReuseGateDisposition.REJECT.value,
                freshness_reason,
                (),
                NegativeMemoryStatus.NONE.value,
                None,
            )
        if not candidate.environment_compatible:
            return (
                ReuseGateDisposition.REJECT.value,
                ReuseReasonCode.ENVIRONMENT_INCOMPATIBLE.value,
                ("environment_cid",),
                NegativeMemoryStatus.NONE.value,
                None,
            )
        unsafe_abstract = (
            candidate.abstraction_kind == AbstractionKind.ABSTRACT.value
            and not candidate.abstraction_safe
        ) or (
            candidate.evidence_kind == ReuseEvidenceKind.ABSTRACT_STATE.value
            and not candidate.abstraction_safe
        )
        if unsafe_abstract:
            if candidate.raw_source_cid is not None:
                return (
                    ReuseGateDisposition.RAW_FALLBACK.value,
                    ReuseReasonCode.UNSAFE_ABSTRACTION.value,
                    (),
                    NegativeMemoryStatus.NONE.value,
                    None,
                )
            return (
                ReuseGateDisposition.REJECT.value,
                ReuseReasonCode.UNSAFE_ABSTRACTION.value,
                (),
                NegativeMemoryStatus.NONE.value,
                None,
            )
        proof_error = _proof_backing_error(candidate)
        if proof_error is not None:
            return (
                ReuseGateDisposition.REJECT.value,
                proof_error,
                (),
                NegativeMemoryStatus.NONE.value,
                None,
            )
        hit = self._find_active_memory(current.key_cid, candidate.candidate_cid)
        if hit is not None:
            return (
                ReuseGateDisposition.REJECT.value,
                ReuseReasonCode.NEGATIVE_MEMORY_HIT.value,
                hit.mismatched_bindings,
                NegativeMemoryStatus.HIT.value,
                hit,
            )
        mismatched = current.mismatched_bindings(candidate.key)
        scope_error = _scope_matches(current, candidate.key)
        if (
            mismatched == ("state_cid",)
            and candidate.independently_proof_backed
            and candidate.scoped_identity_proved
            and candidate.proof_freshly_admitted
            and scope_error is None
        ):
            return (
                ReuseGateDisposition.REUSE.value,
                ReuseReasonCode.PROOF_BACKED_SCOPED_IDENTITY.value,
                (),
                NegativeMemoryStatus.NONE.value,
                None,
            )
        if mismatched:
            return (
                ReuseGateDisposition.REJECT.value,
                _mismatch_reason(mismatched),
                mismatched,
                NegativeMemoryStatus.NONE.value,
                None,
            )
        if scope_error is not None and (
            candidate.independently_proof_backed
            or candidate.relation_claim_cid is not None
            or candidate.scoped_identity_proved
        ):
            return (
                ReuseGateDisposition.REJECT.value,
                scope_error,
                (),
                NegativeMemoryStatus.NONE.value,
                None,
            )
        if candidate.independently_proof_backed and candidate.proof_freshly_admitted:
            return (
                ReuseGateDisposition.REUSE.value,
                ReuseReasonCode.PROOF_BACKED_SCOPED_IDENTITY.value
                if candidate.scoped_identity_proved or candidate.relation_claim_cid
                else ReuseReasonCode.EXACT_IDENTITY_MATCH.value,
                (),
                NegativeMemoryStatus.NONE.value,
                None,
            )
        return (
            ReuseGateDisposition.REUSE.value,
            ReuseReasonCode.EXACT_IDENTITY_MATCH.value,
            (),
            NegativeMemoryStatus.NONE.value,
            None,
        )


_KEY_INPUT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "state_cid",
        "goal_cid",
        "policy_cid",
        "environment_cid",
        "toolchain_cid",
        "procedure_revision_cid",
        "obligation_cids",
        "selection_cids",
        "validation_dependency_cids",
        "scope_cid",
        "assumption_cids",
    }
)
_CANDIDATE_INPUT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "evidence_kind",
        "freshness",
        "proof_cache_hit",
        "proof_freshly_admitted",
        "proof_receipt_cid",
        "proof_authority",
        "independently_proof_backed",
        "scoped_identity_proved",
        "relation_claim_cid",
        "abstraction_kind",
        "abstraction_safe",
        "raw_source_cid",
        "environment_compatible",
        "simulated",
        "similarity_only",
        "capability_available",
        "capability_reason_code",
        "capability_surface",
    }
)


def _parse_candidate(
    candidate: ProgramWorldReuseCandidate
    | ProgramWorldReuseKey
    | Mapping[str, Any]
    | None,
) -> ProgramWorldReuseCandidate | None:
    if candidate is None:
        return None
    if isinstance(candidate, ProgramWorldReuseCandidate):
        return candidate
    if isinstance(candidate, ProgramWorldReuseKey):
        return ProgramWorldReuseCandidate(key=candidate)
    mapping = _mapping(candidate, "candidate")
    if "candidate_cid" in mapping and "schema" in mapping:
        return ProgramWorldReuseCandidate.from_dict(mapping)
    if "key" in mapping:
        key_value = mapping["key"]
        key = (
            key_value
            if isinstance(key_value, ProgramWorldReuseKey)
            else ProgramWorldReuseKey.from_mapping(key_value)
        )
        extra = {
            str(name): mapping[name]
            for name in mapping
            if name in _CANDIDATE_INPUT_FIELDS
        }
        return ProgramWorldReuseCandidate(key=key, **extra)
    key_fields = {
        field: mapping[field] for field in mapping if field in _KEY_INPUT_FIELDS
    }
    extra = {
        str(name): mapping[name]
        for name in mapping
        if name in _CANDIDATE_INPUT_FIELDS
    }
    return ProgramWorldReuseCandidate(
        key=ProgramWorldReuseKey.from_mapping(key_fields), **extra
    )


def _build_decision(
    current: ProgramWorldReuseKey,
    candidate: ProgramWorldReuseCandidate | None,
    *,
    disposition: str,
    reason_code: str,
    limitations: Sequence[str],
    admission_authority: str | None,
    admission_evidence_cid: str | None,
    exact_match: bool,
) -> ProgramWorldReuseDecision:
    verdict = _decision_verdict(disposition)
    admitted = admission_authority is not None
    fallback = verdict == ReuseVerdict.UNAVAILABLE.value or disposition == (
        ReuseGateDisposition.RAW_FALLBACK.value
    )
    relation_cid = None if candidate is None else candidate.relation_claim_cid
    return ProgramWorldReuseDecision(
        verdict=verdict,
        state_cid=current.state_cid,
        goal_cid=current.goal_cid,
        policy_cid=current.policy_cid,
        environment_cid=current.environment_cid,
        toolchain_cid=current.toolchain_cid,
        reason_code=reason_code,
        relation_claim_cid=relation_cid,
        procedure_revision_cid=current.procedure_revision_cid,
        exact_match=exact_match and verdict == ReuseVerdict.REUSE.value,
        proposal_only=not admitted,
        admitted=admitted,
        admission_authority=admission_authority,
        admission_evidence_cid=admission_evidence_cid,
        operational_acceptance_authorities=OPERATIONAL_ACCEPTANCE_AUTHORITIES,
        fallback=fallback,
        limitations=limitations,
    )


def evaluate_program_world_reuse(
    current: ProgramWorldReuseKey | Mapping[str, Any],
    candidate: ProgramWorldReuseCandidate
    | ProgramWorldReuseKey
    | Mapping[str, Any]
    | None = None,
    *,
    gate: ProgramWorldReuseGate | None = None,
    admission_authority: str | None = None,
    admission_evidence_cid: str | None = None,
    remember: bool = True,
) -> ProgramWorldReuseEvaluation:
    """Evaluate exact or independently proof-backed scoped reuse."""

    owned = gate if gate is not None else ProgramWorldReuseGate()
    return owned.evaluate(
        current,
        candidate,
        admission_authority=admission_authority,
        admission_evidence_cid=admission_evidence_cid,
        remember=remember,
    )


def explain_program_world_reuse(
    evaluation: ProgramWorldReuseEvaluation | Mapping[str, Any],
) -> ProgramWorldReuseExplanation:
    """Explain why reuse was granted or failed closed."""

    if isinstance(evaluation, ProgramWorldReuseEvaluation):
        return evaluation.explanation
    if isinstance(evaluation, ProgramWorldReuseExplanation):
        return evaluation
    if isinstance(evaluation, Mapping):
        if "explanation_cid" in evaluation and evaluation.get("interface") == (
            PROGRAM_WORLD_REUSE_EXPLANATION_INTERFACE
        ):
            return ProgramWorldReuseExplanation.from_dict(evaluation)
        return ProgramWorldReuseEvaluation.from_dict(evaluation).explanation
    raise ProgramWorldReuseError(
        "explain_program_world_reuse requires a reuse evaluation or explanation"
    )


__all__ = [
    "ABSTRACTION_KINDS",
    "BINDING_FIELDS",
    "BINDING_MISMATCH_REASON",
    "EMPTY_COLLECTION_BINDING_TEXT",
    "EVENT_AUTHORITY",
    "GATE_OWNED_AUTHORITIES",
    "MERGE_AUTHORITY",
    "NEGATIVE_MEMORY_RECORD_INTERFACE",
    "OPERATIONAL_ACCEPTANCE_AUTHORITIES",
    "PROGRAM_WORLD_REUSE_DECISION_INTERFACE",
    "PROGRAM_WORLD_REUSE_EVALUATION_INTERFACE",
    "PROGRAM_WORLD_REUSE_EXPLANATION_INTERFACE",
    "PROGRAM_WORLD_REUSE_GATE_INTERFACE",
    "PROGRAM_WORLD_REUSE_KEY_INTERFACE",
    "PROGRAM_WORLD_REUSE_REJECTION_INTERFACE",
    "REUSE_GATE_ID",
    "SAWM_EXACT_REUSE_EVIDENCE",
    "SAWM_REUSE_ADMISSION_EVIDENCE",
    "UNBOUND_BINDING_TEXT",
    "VALIDATION_AUTHORITY",
    "AbstractionKind",
    "BindingComparison",
    "NegativeMemoryRecord",
    "NegativeMemoryStatus",
    "ProgramWorldAdmissionError",
    "ProgramWorldReuseCandidate",
    "ProgramWorldReuseError",
    "ProgramWorldReuseEvaluation",
    "ProgramWorldReuseExplanation",
    "ProgramWorldReuseGate",
    "ProgramWorldReuseKey",
    "ProgramWorldReuseRejection",
    "ReuseEvidenceKind",
    "ReuseGateDisposition",
    "ReuseReasonCode",
    "evaluate_program_world_reuse",
    "explain_program_world_reuse",
]
