"""Exact state, transition, procedure, and proof reuse gates (SAWM-016).

Interface: ``ProgramWorldReuseGate@1``
Evidence: ``sawm/exact-reuse@1``, ``sawm/reuse-admission@1``

Operational reuse admission for the semantic-addressed world model.  Only
current exact identity matches or independently proof-backed scoped evidence
can grant reuse.  Similarity is context-only.  Stale, unsafe, incomplete,
or environment-incompatible evidence fails closed and explains why.

This gate is accelerator operational authority.  It cannot validate its own
proof, relation, procedure, or root; it consumes landed caches and procedure
authorities rather than duplicating them.  A cache hit never suppresses
current admission or mandatory validation.

Importing this module performs no I/O, starts no threads, and does not import
datasets or kit implementations.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    _bool,
    _closed,
    validate_opaque_cid,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_receipts import (
    ANN_REASON_CODES,
    DATASETS_IDENTITY_AUTHORITY,
    DATASETS_RELATION_AUTHORITY,
    FORBIDDEN_IDENTITY_FIELDS,
    FRESHNESS_STATES,
    OPERATIONAL_ACCEPTANCE_AUTHORITIES,
    VALIDATION_AUTHORITY,
    FreshnessState,
    ProgramWorldAdmissionError,
    ProgramWorldReceiptError,
    ProgramWorldReuseDecision,
    ReuseVerdict,
    _bounded_text,
    _enum_value,
    _optional_cid,
    _reject_forbidden_identity_fields,
    _unique_sorted_bounded_texts,
    _unique_sorted_cids,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload


# ---------------------------------------------------------------------------
# Interfaces / schemas / closed vocabularies
# ---------------------------------------------------------------------------

PROGRAM_WORLD_REUSE_GATE_INTERFACE: Final[str] = "ProgramWorldReuseGate@1"
PROGRAM_WORLD_REUSE_KEY_INTERFACE: Final[str] = "ProgramWorldReuseKey@1"
PROGRAM_WORLD_REUSE_REJECTION_INTERFACE: Final[str] = "ProgramWorldReuseRejection@1"
NEGATIVE_MEMORY_RECORD_INTERFACE: Final[str] = "NegativeMemoryRecord@1"
PROGRAM_WORLD_REUSE_EXPLANATION_INTERFACE: Final[str] = (
    "ProgramWorldReuseExplanation@1"
)
PROGRAM_WORLD_REUSE_EVALUATION_INTERFACE: Final[str] = (
    "ProgramWorldReuseEvaluation@1"
)

PROGRAM_WORLD_REUSE_KEY_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-reuse-key@1"
)
PROGRAM_WORLD_REUSE_REJECTION_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-reuse-rejection@1"
)
NEGATIVE_MEMORY_RECORD_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-negative-memory@1"
)
PROGRAM_WORLD_REUSE_EXPLANATION_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-reuse-explanation@1"
)
PROGRAM_WORLD_REUSE_EVALUATION_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-reuse-evaluation@1"
)
PROGRAM_WORLD_REUSE_REQUEST_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-reuse-request@1"
)

SAWM_EXACT_REUSE_EVIDENCE: Final[str] = "sawm/exact-reuse@1"
SAWM_REUSE_ADMISSION_EVIDENCE: Final[str] = "sawm/reuse-admission@1"

GATE_ID: Final[str] = "ipfs-accelerate.semantic-state.program-world-reuse"

DATASETS_ABSTRACTION_AUTHORITY: Final[str] = (
    "ipfs_datasets_py.logic.software_contracts.semantic_state.program_abstraction"
)
RECEIPT_CACHE_AUTHORITY: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.verification.receipt_cache"
)
TEST_PROOF_CACHE_AUTHORITY: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.proof.test_proof_cache"
)
INCREMENTAL_SEALER_AUTHORITY: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.sealer"
)
PROOF_TEST_REUSE_GATE_AUTHORITY: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_current_tree_gate"
)
DECISION_RUNTIME_AUTHORITY: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.context.decision_runtime"
)

LOCATED_LANDED_AUTHORITIES: Final[tuple[str, ...]] = (
    DATASETS_IDENTITY_AUTHORITY,
    DATASETS_RELATION_AUTHORITY,
    DATASETS_ABSTRACTION_AUTHORITY,
    RECEIPT_CACHE_AUTHORITY,
    TEST_PROOF_CACHE_AUTHORITY,
    INCREMENTAL_SEALER_AUTHORITY,
    PROOF_TEST_REUSE_GATE_AUTHORITY,
    DECISION_RUNTIME_AUTHORITY,
    VALIDATION_AUTHORITY,
)

LANDED_PROOF_AUTHORITIES: Final[frozenset[str]] = frozenset(
    {
        DATASETS_RELATION_AUTHORITY,
        DATASETS_ABSTRACTION_AUTHORITY,
        RECEIPT_CACHE_AUTHORITY,
        TEST_PROOF_CACHE_AUTHORITY,
        INCREMENTAL_SEALER_AUTHORITY,
        PROOF_TEST_REUSE_GATE_AUTHORITY,
        VALIDATION_AUTHORITY,
    }
)

GATE_OWNED_AUTHORITIES: Final[frozenset[str]] = frozenset(
    {
        GATE_ID,
        PROGRAM_WORLD_REUSE_GATE_INTERFACE,
        "ProgramWorldReuseGate",
        "ProgramWorldReuseKey",
        "ProgramWorldReuseRejection",
        "NegativeMemoryRecord",
        "evaluate_program_world_reuse",
        "explain_program_world_reuse",
    }
)

SCALAR_BINDING_FIELDS: Final[tuple[str, ...]] = (
    "state_cid",
    "goal_cid",
    "policy_cid",
    "environment_cid",
    "toolchain_cid",
    "procedure_revision_cid",
)
TUPLE_BINDING_FIELDS: Final[tuple[str, ...]] = (
    "obligation_cids",
    "selection_cids",
    "validation_dependency_cids",
)
EXACT_BINDING_FIELDS: Final[tuple[str, ...]] = (
    SCALAR_BINDING_FIELDS + TUPLE_BINDING_FIELDS
)
PROOF_BACKED_SUBSTITUTABLE_BINDINGS: Final[frozenset[str]] = frozenset(
    {"state_cid", "procedure_revision_cid"}
)

PROOF_BACKED_RELATION_KINDS: Final[frozenset[str]] = frozenset(
    {
        "equality",
        "refinement",
        "alpha_equivalence",
        "structural_equivalence",
        "logical_equivalence",
        "behavioral_equivalence",
        "observational_equivalence",
    }
)
INDEPENDENT_RELATION_STATUSES: Final[frozenset[str]] = frozenset(
    {"validated", "proved"}
)
STALE_RELATION_STATUSES: Final[frozenset[str]] = frozenset(
    {"stale", "superseded"}
)
CONTRADICTION_RELATION_KINDS: Final[frozenset[str]] = frozenset({"contradiction"})

UNAVAILABLE_SURFACES: Final[frozenset[str]] = frozenset(
    {
        "datasets_identity",
        "kit_verified_store",
        "proof_cache",
        "procedure_authority",
    }
)

BANNED_DUPLICATE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "SemanticObjectEnvelope",
        "ProgramRelationClaim",
        "ProgramTransitionQuery",
        "ContextCompiler",
        "DurableCoordinationStore",
        "VerifiedSemanticBlockStore",
        "VerificationReceiptCache",
        "TestProofCache",
        "StateAbstractionProfile",
        "ProofCarryingProcedureCompiler",
    }
)


class ProgramWorldReuseError(ProgramWorldReceiptError):
    """Closed reuse-gate contract violation."""


class ReuseEvidenceKind(str, Enum):
    EXACT = "exact"
    PROOF_BACKED = "proof_backed"
    SIMILARITY = "similarity"
    UNAVAILABLE = "unavailable"
    RAW_SOURCE = "raw_source"
    UNSPECIFIED = "unspecified"


class ReuseReasonCode(str, Enum):
    EXACT_IDENTITY_MATCH = "exact_identity_match"
    INDEPENDENTLY_PROOF_BACKED = "independently_proof_backed"
    BINDING_MISMATCH = "binding_mismatch"
    INCOMPLETE_BINDING = "incomplete_binding"
    STALE_EVIDENCE = "stale_evidence"
    UNSAFE_ABSTRACTION = "unsafe_abstraction"
    SIMILARITY_IS_NOT_REUSE = "similarity_is_not_reuse"
    ENVIRONMENT_INCOMPATIBLE = "environment_incompatible"
    PROOF_CACHE_REQUIRES_FRESH_ADMISSION = "proof_cache_requires_fresh_admission"
    RELATION_SCOPE_EXCEEDED = "relation_scope_exceeded"
    NEGATIVE_MEMORY_INVALIDATION = "negative_memory_invalidation"
    TYPED_UNAVAILABLE = "typed_unavailable"
    NO_EXACT_IDENTITY_MATCH = "no_exact_identity_match"
    CONTRADICTORY_PREMISES = "contradictory_premises"
    RAW_SOURCE_FALLBACK = "raw_source_fallback"


REUSE_EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in ReuseEvidenceKind
)
REUSE_REASON_CODES: Final[frozenset[str]] = frozenset(
    item.value for item in ReuseReasonCode
) | set(ANN_REASON_CODES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _optional_bool(value: Any, name: str) -> bool | None:
    if value is None:
        return None
    return _bool(value, name)


def _cid_tuple(values: Iterable[Any] | None, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    return _unique_sorted_cids(values, name)


def _reason(value: Any) -> str:
    text = _bounded_text(getattr(value, "value", value), "reason_code")
    if text not in REUSE_REASON_CODES:
        raise ProgramWorldReuseError(
            f"reason_code must be one of {sorted(REUSE_REASON_CODES)}"
        )
    return text


def _limitations(*parts: str | Sequence[str]) -> tuple[str, ...]:
    items: list[str] = []
    for part in parts:
        if isinstance(part, str):
            if part:
                items.append(part)
        else:
            items.extend(item for item in part if item)
    return _unique_sorted_bounded_texts(items, "limitation")


def _payload_cid(payload: Mapping[str, Any]) -> str:
    _reject_forbidden_identity_fields(payload, "identity_payload")
    return cid_for_payload(payload)


def _forbid_gate_self_approval(authority: str | None, *, role: str) -> None:
    if authority is None:
        return
    if authority in GATE_OWNED_AUTHORITIES:
        raise ProgramWorldAdmissionError(
            f"program-world reuse gate cannot {role} its own output"
        )


def _independent_proof_authority(authority: str | None) -> str | None:
    if authority is None or authority == "":
        return None
    text = _bounded_text(authority, "proof_admission_authority")
    _forbid_gate_self_approval(text, role="validate")
    if text not in LANDED_PROOF_AUTHORITIES:
        raise ProgramWorldReuseError(
            "proof_admission_authority must be a landed proof, relation, "
            f"or validation authority, not {text!r}"
        )
    return text


def _mismatched_bindings(
    query: "ProgramWorldReuseKey", candidate: "ProgramWorldReuseKey"
) -> tuple[str, ...]:
    mismatches: list[str] = []
    for field in SCALAR_BINDING_FIELDS:
        if getattr(query, field) != getattr(candidate, field):
            mismatches.append(field)
    for field in TUPLE_BINDING_FIELDS:
        if getattr(query, field) != getattr(candidate, field):
            mismatches.append(field)
    return tuple(mismatches)


def _surface_reason(surface: str) -> str:
    return {
        "datasets_identity": "datasets_identity_surface_unavailable",
        "kit_verified_store": "kit_verified_store_unavailable",
        "proof_cache": "proof_cache_unavailable",
        "procedure_authority": "procedure_authority_unavailable",
    }.get(surface, "typed_unavailable")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseKey:
    """Exact reuse identity over every operational binding that can admit reuse."""

    state_cid: str
    goal_cid: str
    policy_cid: str
    environment_cid: str
    toolchain_cid: str
    obligation_cids: Sequence[str] = ()
    selection_cids: Sequence[str] = ()
    procedure_revision_cid: str | None = None
    validation_dependency_cids: Sequence[str] = ()

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
            "obligation_cids",
            "selection_cids",
            "procedure_revision_cid",
            "validation_dependency_cids",
            "key_cid",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "state_cid",
            "goal_cid",
            "policy_cid",
            "environment_cid",
            "toolchain_cid",
        ):
            object.__setattr__(self, name, validate_opaque_cid(getattr(self, name), name))
        object.__setattr__(
            self,
            "procedure_revision_cid",
            _optional_cid(self.procedure_revision_cid, "procedure_revision_cid"),
        )
        object.__setattr__(
            self, "obligation_cids", _cid_tuple(self.obligation_cids, "obligation_cid")
        )
        object.__setattr__(
            self, "selection_cids", _cid_tuple(self.selection_cids, "selection_cid")
        )
        object.__setattr__(
            self,
            "validation_dependency_cids",
            _cid_tuple(self.validation_dependency_cids, "validation_dependency_cid"),
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "state_cid": self.state_cid,
            "goal_cid": self.goal_cid,
            "policy_cid": self.policy_cid,
            "environment_cid": self.environment_cid,
            "toolchain_cid": self.toolchain_cid,
            "obligation_cids": list(self.obligation_cids),
            "selection_cids": list(self.selection_cids),
            "procedure_revision_cid": self.procedure_revision_cid,
            "validation_dependency_cids": list(self.validation_dependency_cids),
        }

    @property
    def key_cid(self) -> str:
        return _payload_cid(self.identity_payload())

    def binding_map(self) -> dict[str, Any]:
        return {
            "state_cid": self.state_cid,
            "goal_cid": self.goal_cid,
            "policy_cid": self.policy_cid,
            "environment_cid": self.environment_cid,
            "toolchain_cid": self.toolchain_cid,
            "obligation_cids": self.obligation_cids,
            "selection_cids": self.selection_cids,
            "procedure_revision_cid": self.procedure_revision_cid,
            "validation_dependency_cids": self.validation_dependency_cids,
        }

    def replace(self, **overrides: Any) -> "ProgramWorldReuseKey":
        payload = self.binding_map()
        payload.update(overrides)
        return ProgramWorldReuseKey(**payload)

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["key_cid"] = self.key_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgramWorldReuseKey":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("key_cid")
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReuseError("unsupported reuse-key schema")
        if payload.pop("interface") != cls.INTERFACE:
            raise ProgramWorldReuseError("unsupported reuse-key interface")
        result = cls(**payload)
        if result.key_cid != claimed:
            raise ProgramWorldReuseError("key_cid does not rehash")
        return result


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseRequest:
    """Candidate evidence presented to the reuse gate. Scores never admit reuse."""

    key: ProgramWorldReuseKey
    candidate_key: ProgramWorldReuseKey | None = None
    freshness: FreshnessState | str = FreshnessState.FRESH
    evidence_kind: ReuseEvidenceKind | str = ReuseEvidenceKind.UNSPECIFIED
    environment_compatible: bool = True
    complete: bool = True
    abstraction_sound: bool | None = None
    abstraction_soundness_authority: str | None = None
    abstraction_profile_cid: str | None = None
    candidate_abstraction_profile_cid: str | None = None
    unproved_omission: bool = False
    unavailable_dimensions: Sequence[str] = ()
    relation_claim_cid: str | None = None
    relation_scope_cid: str | None = None
    relation_kind: str | None = None
    relation_status: str | None = None
    relation_environment_cid: str | None = None
    relation_policy_cid: str | None = None
    relation_subject_cids: Sequence[str] = ()
    relation_assumption_cids: Sequence[str] = ()
    relation_assumptions_hold: bool = True
    independently_proof_backed: bool = False
    proof_cache_hit: bool = False
    proof_cache_fresh: bool = False
    proof_admitted: bool = False
    proof_admission_authority: str | None = None
    proof_admission_evidence_cid: str | None = None
    current_world_root_cid: str | None = None
    candidate_world_root_cid: str | None = None
    source_cid: str | None = None
    raw_source_available: bool = False
    unavailable_surfaces: Sequence[str] = ()
    ann_candidates: Sequence[Any] = ()
    similarity_score: float | None = None
    admission_authority: str | None = None
    admission_evidence_cid: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, ProgramWorldReuseKey):
            raise ProgramWorldReuseError("key must be a ProgramWorldReuseKey")
        if self.candidate_key is not None and not isinstance(
            self.candidate_key, ProgramWorldReuseKey
        ):
            raise ProgramWorldReuseError(
                "candidate_key must be a ProgramWorldReuseKey or null"
            )
        object.__setattr__(
            self, "freshness", _enum_value(self.freshness, FRESHNESS_STATES, "freshness")
        )
        object.__setattr__(
            self,
            "evidence_kind",
            _enum_value(self.evidence_kind, REUSE_EVIDENCE_KINDS, "evidence_kind"),
        )
        object.__setattr__(
            self,
            "environment_compatible",
            _bool(self.environment_compatible, "environment_compatible"),
        )
        object.__setattr__(self, "complete", _bool(self.complete, "complete"))
        object.__setattr__(
            self, "abstraction_sound", _optional_bool(self.abstraction_sound, "abstraction_sound")
        )
        object.__setattr__(
            self,
            "abstraction_soundness_authority",
            None
            if self.abstraction_soundness_authority in (None, "")
            else _bounded_text(
                self.abstraction_soundness_authority, "abstraction_soundness_authority"
            ),
        )
        _forbid_gate_self_approval(
            self.abstraction_soundness_authority, role="certify abstraction soundness of"
        )
        object.__setattr__(
            self,
            "abstraction_profile_cid",
            _optional_cid(self.abstraction_profile_cid, "abstraction_profile_cid"),
        )
        object.__setattr__(
            self,
            "candidate_abstraction_profile_cid",
            _optional_cid(
                self.candidate_abstraction_profile_cid, "candidate_abstraction_profile_cid"
            ),
        )
        object.__setattr__(
            self, "unproved_omission", _bool(self.unproved_omission, "unproved_omission")
        )
        object.__setattr__(
            self,
            "unavailable_dimensions",
            _unique_sorted_bounded_texts(
                self.unavailable_dimensions, "unavailable_dimension"
            ),
        )
        object.__setattr__(
            self,
            "relation_claim_cid",
            _optional_cid(self.relation_claim_cid, "relation_claim_cid"),
        )
        object.__setattr__(
            self,
            "relation_scope_cid",
            _optional_cid(self.relation_scope_cid, "relation_scope_cid"),
        )
        object.__setattr__(
            self,
            "relation_kind",
            None
            if self.relation_kind in (None, "")
            else _bounded_text(self.relation_kind, "relation_kind"),
        )
        object.__setattr__(
            self,
            "relation_status",
            None
            if self.relation_status in (None, "")
            else _bounded_text(self.relation_status, "relation_status"),
        )
        object.__setattr__(
            self,
            "relation_environment_cid",
            _optional_cid(self.relation_environment_cid, "relation_environment_cid"),
        )
        object.__setattr__(
            self,
            "relation_policy_cid",
            _optional_cid(self.relation_policy_cid, "relation_policy_cid"),
        )
        object.__setattr__(
            self,
            "relation_subject_cids",
            _cid_tuple(self.relation_subject_cids, "relation_subject_cid"),
        )
        object.__setattr__(
            self,
            "relation_assumption_cids",
            _cid_tuple(self.relation_assumption_cids, "relation_assumption_cid"),
        )
        object.__setattr__(
            self,
            "relation_assumptions_hold",
            _bool(self.relation_assumptions_hold, "relation_assumptions_hold"),
        )
        object.__setattr__(
            self,
            "independently_proof_backed",
            _bool(self.independently_proof_backed, "independently_proof_backed"),
        )
        object.__setattr__(
            self, "proof_cache_hit", _bool(self.proof_cache_hit, "proof_cache_hit")
        )
        object.__setattr__(
            self, "proof_cache_fresh", _bool(self.proof_cache_fresh, "proof_cache_fresh")
        )
        object.__setattr__(
            self, "proof_admitted", _bool(self.proof_admitted, "proof_admitted")
        )
        object.__setattr__(
            self,
            "proof_admission_authority",
            _independent_proof_authority(self.proof_admission_authority),
        )
        object.__setattr__(
            self,
            "proof_admission_evidence_cid",
            _optional_cid(
                self.proof_admission_evidence_cid, "proof_admission_evidence_cid"
            ),
        )
        if self.proof_admitted and (
            self.proof_admission_authority is None
            or self.proof_admission_evidence_cid is None
        ):
            raise ProgramWorldAdmissionError(
                "independently admitted proof evidence requires a landed "
                "proof/relation/validation authority and evidence CID"
            )
        object.__setattr__(
            self,
            "current_world_root_cid",
            _optional_cid(self.current_world_root_cid, "current_world_root_cid"),
        )
        object.__setattr__(
            self,
            "candidate_world_root_cid",
            _optional_cid(self.candidate_world_root_cid, "candidate_world_root_cid"),
        )
        object.__setattr__(
            self, "source_cid", _optional_cid(self.source_cid, "source_cid")
        )
        object.__setattr__(
            self, "raw_source_available", _bool(self.raw_source_available, "raw_source_available")
        )
        surfaces = _unique_sorted_bounded_texts(
            self.unavailable_surfaces, "unavailable_surface"
        )
        extra = set(surfaces) - UNAVAILABLE_SURFACES
        if extra:
            raise ProgramWorldReuseError(
                f"unknown unavailable surfaces {sorted(extra)}"
            )
        object.__setattr__(self, "unavailable_surfaces", surfaces)
        if self.ann_candidates and not isinstance(self.ann_candidates, Sequence):
            raise ProgramWorldReuseError("ann_candidates must be a sequence")
        object.__setattr__(self, "ann_candidates", tuple(self.ann_candidates))
        if self.similarity_score is not None and type(self.similarity_score) not in {
            int,
            float,
        }:
            raise ProgramWorldReuseError("similarity_score must be a number or null")
        if self.similarity_score is not None and isinstance(self.similarity_score, bool):
            raise ProgramWorldReuseError("similarity_score must be a number or null")
        object.__setattr__(
            self,
            "admission_authority",
            None
            if self.admission_authority in (None, "")
            else _bounded_text(self.admission_authority, "admission_authority"),
        )
        object.__setattr__(
            self,
            "admission_evidence_cid",
            _optional_cid(self.admission_evidence_cid, "admission_evidence_cid"),
        )
        _forbid_gate_self_approval(self.admission_authority, role="admit")
        if self.relation_kind is not None:
            lowered = self.relation_kind.lower()
            if lowered in FORBIDDEN_IDENTITY_FIELDS or lowered in ANN_REASON_CODES:
                raise ProgramWorldReuseError(
                    "ANN/similarity cannot be encoded as a reuse relation kind"
                )

    @property
    def similarity_present(self) -> bool:
        return bool(self.ann_candidates) or self.similarity_score is not None or (
            self.evidence_kind == ReuseEvidenceKind.SIMILARITY.value
        )

    @property
    def uses_abstraction(self) -> bool:
        return (
            self.abstraction_profile_cid is not None
            or self.candidate_abstraction_profile_cid is not None
            or self.abstraction_sound is not None
        )


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseRejection:
    """Typed fail-closed rejection. Always explains why reuse was refused."""

    key_cid: str
    reason_code: str
    diagnostic: str
    verdict: ReuseVerdict | str = ReuseVerdict.REJECT
    mismatched_bindings: Sequence[str] = ()
    invalidator_cids: Sequence[str] = ()
    limitations: Sequence[str] = ()
    raw_source_fallback: bool = False
    context_only: bool = False
    candidate_key_cid: str | None = None

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_REUSE_REJECTION_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_REUSE_REJECTION_INTERFACE
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "key_cid",
            "candidate_key_cid",
            "verdict",
            "reason_code",
            "diagnostic",
            "mismatched_bindings",
            "invalidator_cids",
            "limitations",
            "raw_source_fallback",
            "context_only",
            "rejection_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_cid", validate_opaque_cid(self.key_cid, "key_cid"))
        object.__setattr__(
            self,
            "candidate_key_cid",
            _optional_cid(self.candidate_key_cid, "candidate_key_cid"),
        )
        object.__setattr__(
            self, "verdict", _enum_value(self.verdict, {item.value for item in ReuseVerdict}, "verdict")
        )
        object.__setattr__(self, "reason_code", _reason(self.reason_code))
        object.__setattr__(
            self, "diagnostic", _bounded_text(self.diagnostic, "diagnostic")
        )
        object.__setattr__(
            self,
            "mismatched_bindings",
            _unique_sorted_bounded_texts(self.mismatched_bindings, "mismatched_binding"),
        )
        object.__setattr__(
            self,
            "invalidator_cids",
            _cid_tuple(self.invalidator_cids, "invalidator_cid"),
        )
        object.__setattr__(
            self, "limitations", _unique_sorted_bounded_texts(self.limitations, "limitation")
        )
        object.__setattr__(
            self, "raw_source_fallback", _bool(self.raw_source_fallback, "raw_source_fallback")
        )
        object.__setattr__(self, "context_only", _bool(self.context_only, "context_only"))
        if self.verdict == ReuseVerdict.REUSE.value:
            raise ProgramWorldReuseError("rejection cannot carry a reuse verdict")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "key_cid": self.key_cid,
            "candidate_key_cid": self.candidate_key_cid,
            "verdict": self.verdict,
            "reason_code": self.reason_code,
            "diagnostic": self.diagnostic,
            "mismatched_bindings": list(self.mismatched_bindings),
            "invalidator_cids": list(self.invalidator_cids),
            "limitations": list(self.limitations),
            "raw_source_fallback": self.raw_source_fallback,
            "context_only": self.context_only,
        }

    @property
    def rejection_cid(self) -> str:
        return _payload_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["rejection_cid"] = self.rejection_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgramWorldReuseRejection":
        payload = _closed(data, cls._FIELDS, cls.__name__)
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
    """Scoped rejection memory. Does not delete history; invalidation is explicit."""

    key_cid: str
    reason_code: str
    diagnostic: str
    mismatched_bindings: Sequence[str] = ()
    invalidator_cids: Sequence[str] = ()
    candidate_key_cid: str | None = None
    rejection_cid: str | None = None
    proof_admission_evidence_cid: str | None = None
    active: bool = True

    SCHEMA: ClassVar[str] = NEGATIVE_MEMORY_RECORD_SCHEMA
    INTERFACE: ClassVar[str] = NEGATIVE_MEMORY_RECORD_INTERFACE
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "key_cid",
            "candidate_key_cid",
            "reason_code",
            "diagnostic",
            "mismatched_bindings",
            "invalidator_cids",
            "rejection_cid",
            "proof_admission_evidence_cid",
            "active",
            "record_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_cid", validate_opaque_cid(self.key_cid, "key_cid"))
        object.__setattr__(
            self,
            "candidate_key_cid",
            _optional_cid(self.candidate_key_cid, "candidate_key_cid"),
        )
        object.__setattr__(self, "reason_code", _reason(self.reason_code))
        object.__setattr__(
            self, "diagnostic", _bounded_text(self.diagnostic, "diagnostic")
        )
        object.__setattr__(
            self,
            "mismatched_bindings",
            _unique_sorted_bounded_texts(self.mismatched_bindings, "mismatched_binding"),
        )
        object.__setattr__(
            self,
            "invalidator_cids",
            _cid_tuple(self.invalidator_cids, "invalidator_cid"),
        )
        object.__setattr__(
            self, "rejection_cid", _optional_cid(self.rejection_cid, "rejection_cid")
        )
        object.__setattr__(
            self,
            "proof_admission_evidence_cid",
            _optional_cid(
                self.proof_admission_evidence_cid, "proof_admission_evidence_cid"
            ),
        )
        object.__setattr__(self, "active", _bool(self.active, "active"))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "key_cid": self.key_cid,
            "candidate_key_cid": self.candidate_key_cid,
            "reason_code": self.reason_code,
            "diagnostic": self.diagnostic,
            "mismatched_bindings": list(self.mismatched_bindings),
            "invalidator_cids": list(self.invalidator_cids),
            "rejection_cid": self.rejection_cid,
            "proof_admission_evidence_cid": self.proof_admission_evidence_cid,
            "active": self.active,
        }

    @property
    def record_cid(self) -> str:
        return _payload_cid(self.identity_payload())

    def deactivated(self) -> "NegativeMemoryRecord":
        return NegativeMemoryRecord(
            key_cid=self.key_cid,
            reason_code=self.reason_code,
            diagnostic=self.diagnostic,
            mismatched_bindings=self.mismatched_bindings,
            invalidator_cids=self.invalidator_cids,
            candidate_key_cid=self.candidate_key_cid,
            rejection_cid=self.rejection_cid,
            proof_admission_evidence_cid=self.proof_admission_evidence_cid,
            active=False,
        )

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["record_cid"] = self.record_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NegativeMemoryRecord":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("record_cid")
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReuseError("unsupported negative-memory schema")
        if payload.pop("interface") != cls.INTERFACE:
            raise ProgramWorldReuseError("unsupported negative-memory interface")
        result = cls(**payload)
        if result.record_cid != claimed:
            raise ProgramWorldReuseError("record_cid does not rehash")
        return result


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseExplanation:
    """Deterministic explanation of a reuse grant or fail-closed refusal."""

    verdict: ReuseVerdict | str
    reason_code: str
    why: str
    exact_match: bool = False
    independently_proof_backed: bool = False
    mismatched_bindings: Sequence[str] = ()
    limitations: Sequence[str] = ()
    context_only: bool = False
    raw_source_fallback: bool = False
    required_admission: bool = True
    negative_memory_cid: str | None = None
    decision_cid: str | None = None
    rejection_cid: str | None = None

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_REUSE_EXPLANATION_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_REUSE_EXPLANATION_INTERFACE
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "verdict",
            "reason_code",
            "why",
            "exact_match",
            "independently_proof_backed",
            "mismatched_bindings",
            "limitations",
            "context_only",
            "raw_source_fallback",
            "required_admission",
            "negative_memory_cid",
            "decision_cid",
            "rejection_cid",
            "explanation_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "verdict", _enum_value(self.verdict, {item.value for item in ReuseVerdict}, "verdict")
        )
        object.__setattr__(self, "reason_code", _reason(self.reason_code))
        object.__setattr__(self, "why", _bounded_text(self.why, "why"))
        object.__setattr__(self, "exact_match", _bool(self.exact_match, "exact_match"))
        object.__setattr__(
            self,
            "independently_proof_backed",
            _bool(self.independently_proof_backed, "independently_proof_backed"),
        )
        object.__setattr__(
            self,
            "mismatched_bindings",
            _unique_sorted_bounded_texts(self.mismatched_bindings, "mismatched_binding"),
        )
        object.__setattr__(
            self, "limitations", _unique_sorted_bounded_texts(self.limitations, "limitation")
        )
        object.__setattr__(self, "context_only", _bool(self.context_only, "context_only"))
        object.__setattr__(
            self, "raw_source_fallback", _bool(self.raw_source_fallback, "raw_source_fallback")
        )
        object.__setattr__(
            self, "required_admission", _bool(self.required_admission, "required_admission")
        )
        object.__setattr__(
            self,
            "negative_memory_cid",
            _optional_cid(self.negative_memory_cid, "negative_memory_cid"),
        )
        object.__setattr__(
            self, "decision_cid", _optional_cid(self.decision_cid, "decision_cid")
        )
        object.__setattr__(
            self, "rejection_cid", _optional_cid(self.rejection_cid, "rejection_cid")
        )
        if self.verdict == ReuseVerdict.REUSE.value and not (
            self.exact_match or self.independently_proof_backed
        ):
            raise ProgramWorldReuseError(
                "reuse explanations require exact or independently proof-backed evidence"
            )
        if self.context_only and self.verdict == ReuseVerdict.REUSE.value and not self.exact_match and not self.independently_proof_backed:
            raise ProgramWorldReuseError("similarity cannot be the reuse grant path")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "verdict": self.verdict,
            "reason_code": self.reason_code,
            "why": self.why,
            "exact_match": self.exact_match,
            "independently_proof_backed": self.independently_proof_backed,
            "mismatched_bindings": list(self.mismatched_bindings),
            "limitations": list(self.limitations),
            "context_only": self.context_only,
            "raw_source_fallback": self.raw_source_fallback,
            "required_admission": self.required_admission,
            "negative_memory_cid": self.negative_memory_cid,
            "decision_cid": self.decision_cid,
            "rejection_cid": self.rejection_cid,
        }

    @property
    def explanation_cid(self) -> str:
        return _payload_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["explanation_cid"] = self.explanation_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgramWorldReuseExplanation":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("explanation_cid")
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReuseError("unsupported reuse-explanation schema")
        if payload.pop("interface") != cls.INTERFACE:
            raise ProgramWorldReuseError("unsupported reuse-explanation interface")
        result = cls(**payload)
        if result.explanation_cid != claimed:
            raise ProgramWorldReuseError("explanation_cid does not rehash")
        return result


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseEvaluation:
    """Complete reuse-gate result: decision, optional rejection, and explanation."""

    key: ProgramWorldReuseKey
    decision: ProgramWorldReuseDecision
    explanation: ProgramWorldReuseExplanation
    rejection: ProgramWorldReuseRejection | None = None
    negative_memory: NegativeMemoryRecord | None = None

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_REUSE_EVALUATION_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_REUSE_EVALUATION_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.key, ProgramWorldReuseKey):
            raise ProgramWorldReuseError("evaluation key must be a ProgramWorldReuseKey")
        if not isinstance(self.decision, ProgramWorldReuseDecision):
            raise ProgramWorldReuseError("evaluation decision must be a ProgramWorldReuseDecision")
        if not isinstance(self.explanation, ProgramWorldReuseExplanation):
            raise ProgramWorldReuseError(
                "evaluation explanation must be a ProgramWorldReuseExplanation"
            )
        if self.rejection is not None and not isinstance(
            self.rejection, ProgramWorldReuseRejection
        ):
            raise ProgramWorldReuseError("rejection must be a ProgramWorldReuseRejection")
        if self.negative_memory is not None and not isinstance(
            self.negative_memory, NegativeMemoryRecord
        ):
            raise ProgramWorldReuseError("negative_memory must be a NegativeMemoryRecord")
        if self.decision.verdict == ReuseVerdict.REUSE.value and self.rejection is not None:
            raise ProgramWorldReuseError("granted reuse cannot carry a rejection record")
        if self.decision.ann_authoritative:
            raise ProgramWorldReuseError("ANN cannot be authoritative for reuse")

    @property
    def verdict(self) -> str:
        return str(self.decision.verdict)

    @property
    def granted(self) -> bool:
        return self.decision.verdict == ReuseVerdict.REUSE.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "key": self.key.to_dict(),
            "decision": self.decision.to_dict(),
            "explanation": self.explanation.to_dict(),
            "rejection": None if self.rejection is None else self.rejection.to_dict(),
            "negative_memory": None
            if self.negative_memory is None
            else self.negative_memory.to_dict(),
        }


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class ProgramWorldReuseGate:
    """Fail-closed exact and independently proof-backed reuse admission."""

    INTERFACE: Final[str] = PROGRAM_WORLD_REUSE_GATE_INTERFACE
    GATE_ID: Final[str] = GATE_ID

    def __init__(self) -> None:
        self._negative_memory: dict[str, NegativeMemoryRecord] = {}

    def negative_memory_records(self) -> tuple[NegativeMemoryRecord, ...]:
        return tuple(
            record
            for record_cid in sorted(self._negative_memory)
            for record in (self._negative_memory[record_cid],)
        )

    def invalidate_negative_memory(
        self, record_cid: str | None = None, *, key_cid: str | None = None
    ) -> tuple[NegativeMemoryRecord, ...]:
        invalidated: list[NegativeMemoryRecord] = []
        targets = list(self._negative_memory.items())
        for current_cid, record in targets:
            if record_cid is not None and current_cid != record_cid:
                continue
            if key_cid is not None and record.key_cid != key_cid:
                continue
            if not record.active:
                continue
            successor = record.deactivated()
            del self._negative_memory[current_cid]
            self._negative_memory[successor.record_cid] = successor
            invalidated.append(successor)
        return tuple(invalidated)

    def evaluate(self, request: ProgramWorldReuseRequest) -> ProgramWorldReuseEvaluation:
        if not isinstance(request, ProgramWorldReuseRequest):
            raise ProgramWorldReuseError("evaluate requires a ProgramWorldReuseRequest")
        self._forbid_operational_self_admission(request)
        key = request.key
        outcome = (
            self._check_unavailable(request, key)
            or self._check_negative_memory(request, key)
            or self._check_incomplete(request, key)
            or self._check_similarity_only(request, key)
            or self._check_environment(request, key)
            or self._check_freshness(request, key)
            or self._check_abstraction(request, key)
            or self._check_relation_contradiction(request, key)
            or self._check_bindings_and_proof(request, key)
        )
        if outcome is None:
            return self._grant(request, key)
        return self._close(request, key, outcome)

    def explain(
        self,
        target: ProgramWorldReuseEvaluation | ProgramWorldReuseRequest | ProgramWorldReuseDecision,
    ) -> ProgramWorldReuseExplanation:
        if isinstance(target, ProgramWorldReuseEvaluation):
            return target.explanation
        if isinstance(target, ProgramWorldReuseRequest):
            return self.evaluate(target).explanation
        if isinstance(target, ProgramWorldReuseDecision):
            return ProgramWorldReuseExplanation(
                verdict=target.verdict,
                reason_code=target.reason_code,
                why=self._why_from_decision(target),
                exact_match=target.exact_match,
                independently_proof_backed=target.reason_code
                == ReuseReasonCode.INDEPENDENTLY_PROOF_BACKED.value,
                limitations=target.limitations,
                context_only=target.reason_code in ANN_REASON_CODES,
                raw_source_fallback=target.fallback
                and target.verdict != ReuseVerdict.REUSE.value,
                required_admission=not target.admitted,
                decision_cid=target.decision_cid,
            )
        raise ProgramWorldReuseError(
            "explain_program_world_reuse requires an evaluation, request, or decision"
        )

    def _forbid_operational_self_admission(self, request: ProgramWorldReuseRequest) -> None:
        _forbid_gate_self_approval(request.admission_authority, role="admit")
        if request.admission_authority is None:
            return
        if request.admission_authority not in OPERATIONAL_ACCEPTANCE_AUTHORITIES:
            raise ProgramWorldAdmissionError(
                "admission_authority is not an existing supervisor validation, "
                "merge, or event authority"
            )
        if request.admission_evidence_cid is None:
            raise ProgramWorldAdmissionError(
                "naming an admission authority requires admission_evidence_cid"
            )

    def _check_unavailable(
        self, request: ProgramWorldReuseRequest, key: ProgramWorldReuseKey
    ) -> ProgramWorldReuseRejection | None:
        if not request.unavailable_surfaces and request.evidence_kind != (
            ReuseEvidenceKind.UNAVAILABLE.value
        ):
            return None
        surfaces = request.unavailable_surfaces or ("datasets_identity",)
        surface = surfaces[0]
        return self._rejection(
            request,
            key,
            verdict=ReuseVerdict.UNAVAILABLE,
            reason_code=ReuseReasonCode.TYPED_UNAVAILABLE,
            diagnostic=(
                f"reuse surface {surface} is typed-unavailable; "
                "capability is not simulated"
            ),
            limitations=(
                _surface_reason(surface),
                "typed_unavailable",
                "raw_source_fallback_available" if request.raw_source_available else "",
            ),
            fallback=request.raw_source_available,
        )

    def _check_negative_memory(
        self, request: ProgramWorldReuseRequest, key: ProgramWorldReuseKey
    ) -> ProgramWorldReuseRejection | None:
        candidate_cid = None if request.candidate_key is None else request.candidate_key.key_cid
        current_evidence = self._current_evidence_cids(request)
        for record in self._negative_memory.values():
            if not record.active or record.key_cid != key.key_cid:
                continue
            if record.candidate_key_cid != candidate_cid:
                continue
            if self._invalidators_lifted(record, request, current_evidence):
                continue
            return self._rejection(
                request,
                key,
                verdict=ReuseVerdict.REJECT,
                reason_code=ReuseReasonCode.NEGATIVE_MEMORY_INVALIDATION,
                diagnostic=(
                    "scoped negative memory still applies for "
                    f"{record.reason_code}: {record.diagnostic}"
                ),
                mismatched_bindings=record.mismatched_bindings,
                invalidator_cids=record.invalidator_cids,
                limitations=("negative_memory_active", record.reason_code),
                fallback=request.raw_source_available,
            )
        return None

    def _check_incomplete(
        self, request: ProgramWorldReuseRequest, key: ProgramWorldReuseKey
    ) -> ProgramWorldReuseRejection | None:
        if request.complete and request.candidate_key is not None:
            return None
        if request.candidate_key is None and request.similarity_present:
            return None
        if request.candidate_key is None and request.unavailable_surfaces:
            return None
        if request.complete and request.candidate_key is None:
            return None
        return self._rejection(
            request,
            key,
            verdict=ReuseVerdict.REJECT,
            reason_code=ReuseReasonCode.INCOMPLETE_BINDING,
            diagnostic="reuse evidence is incomplete; missing required exact bindings",
            limitations=(
                "incomplete_evidence",
                "raw_source_fallback_available" if request.raw_source_available else "",
            ),
            fallback=request.raw_source_available,
        )

    def _check_similarity_only(
        self, request: ProgramWorldReuseRequest, key: ProgramWorldReuseKey
    ) -> ProgramWorldReuseRejection | None:
        if not request.similarity_present:
            return None
        candidate = request.candidate_key
        exact = candidate is not None and candidate.key_cid == key.key_cid
        if exact:
            return None
        if (
            request.evidence_kind == ReuseEvidenceKind.PROOF_BACKED.value
            or request.independently_proof_backed
            or request.proof_admitted
        ):
            return None
        return self._rejection(
            request,
            key,
            verdict=ReuseVerdict.REJECT,
            reason_code=ReuseReasonCode.SIMILARITY_IS_NOT_REUSE,
            diagnostic=(
                "similarity, ANN, and embedding neighbors are context-only "
                "and cannot grant reuse"
            ),
            limitations=("similarity_is_context_only", "ann_advisory_only"),
            context_only=True,
            fallback=request.raw_source_available,
        )

    def _check_environment(
        self, request: ProgramWorldReuseRequest, key: ProgramWorldReuseKey
    ) -> ProgramWorldReuseRejection | None:
        # CID mismatch is BINDING_MISMATCH; this reason is the explicit flag only.
        if request.environment_compatible:
            return None
        candidate = request.candidate_key
        env_mismatch = candidate is not None and candidate.environment_cid != key.environment_cid
        return self._rejection(
            request,
            key,
            verdict=ReuseVerdict.REJECT,
            reason_code=ReuseReasonCode.ENVIRONMENT_INCOMPATIBLE,
            diagnostic="candidate environment binding is incompatible with the current key",
            mismatched_bindings=("environment_cid",) if env_mismatch else (),
            limitations=("environment_must_match_exactly",),
            fallback=request.raw_source_available,
        )

    def _check_freshness(
        self, request: ProgramWorldReuseRequest, key: ProgramWorldReuseKey
    ) -> ProgramWorldReuseRejection | None:
        stale_root = (
            request.current_world_root_cid is not None
            and request.candidate_world_root_cid is not None
            and request.current_world_root_cid != request.candidate_world_root_cid
        )
        stale_status = request.relation_status in STALE_RELATION_STATUSES
        if (
            request.freshness == FreshnessState.FRESH.value
            and not stale_root
            and not stale_status
        ):
            return None
        if request.freshness == FreshnessState.UNAVAILABLE.value:
            return self._rejection(
                request,
                key,
                verdict=ReuseVerdict.UNAVAILABLE,
                reason_code=ReuseReasonCode.TYPED_UNAVAILABLE,
                diagnostic="freshness authority is typed-unavailable; reuse is not simulated",
                limitations=("freshness_unavailable",),
                fallback=request.raw_source_available,
            )
        return self._rejection(
            request,
            key,
            verdict=ReuseVerdict.REJECT,
            reason_code=ReuseReasonCode.STALE_EVIDENCE,
            diagnostic="stale or unknown freshness revokes reuse even on an exact key hit",
            invalidator_cids=tuple(
                cid
                for cid in (
                    request.candidate_world_root_cid,
                    request.current_world_root_cid,
                    request.relation_claim_cid,
                )
                if cid is not None
            ),
            limitations=("stale_authoritative_reuse_forbidden", "freshness_revocation"),
            fallback=request.raw_source_available,
        )

    def _check_abstraction(
        self, request: ProgramWorldReuseRequest, key: ProgramWorldReuseKey
    ) -> ProgramWorldReuseRejection | None:
        if not request.uses_abstraction:
            return None
        profile_mismatch = (
            request.abstraction_profile_cid is not None
            and request.candidate_abstraction_profile_cid is not None
            and request.abstraction_profile_cid != request.candidate_abstraction_profile_cid
        )
        unsafe = (
            request.abstraction_sound is False
            or request.unproved_omission
            or bool(request.unavailable_dimensions)
            or profile_mismatch
            or (
                request.uses_abstraction
                and request.abstraction_sound is not True
            )
        )
        if not unsafe:
            if request.abstraction_soundness_authority not in {
                DATASETS_ABSTRACTION_AUTHORITY,
                DATASETS_IDENTITY_AUTHORITY,
            }:
                return self._rejection(
                    request,
                    key,
                    verdict=ReuseVerdict.REJECT,
                    reason_code=ReuseReasonCode.UNSAFE_ABSTRACTION,
                    diagnostic=(
                        "abstraction soundness must be independently certified by "
                        "datasets abstraction authority"
                    ),
                    limitations=("datasets_owns_abstraction_soundness",),
                    fallback=True if request.raw_source_available or request.source_cid else False,
                )
            return None
        return self._rejection(
            request,
            key,
            verdict=ReuseVerdict.REJECT,
            reason_code=ReuseReasonCode.UNSAFE_ABSTRACTION,
            diagnostic=(
                "unsafe or unproved abstraction cannot admit reuse; "
                "raw-source fallback remains available when present"
            ),
            mismatched_bindings=("state_cid",) if profile_mismatch else (),
            limitations=(
                "unsafe_abstraction_reuse_forbidden",
                "unproved_omission_cannot_admit_reuse" if request.unproved_omission else "",
                "unknown_relevance_widens" if request.unavailable_dimensions else "",
                "raw_source_fallback_available"
                if request.raw_source_available or request.source_cid
                else "",
            ),
            fallback=True if request.raw_source_available or request.source_cid else False,
        )

    def _check_relation_contradiction(
        self, request: ProgramWorldReuseRequest, key: ProgramWorldReuseKey
    ) -> ProgramWorldReuseRejection | None:
        kind = (request.relation_kind or "").lower()
        if kind not in CONTRADICTION_RELATION_KINDS:
            return None
        return self._rejection(
            request,
            key,
            verdict=ReuseVerdict.ABSTAIN,
            reason_code=ReuseReasonCode.CONTRADICTORY_PREMISES,
            diagnostic="contradictory admitted premises abstain; they never grant reuse",
            limitations=("no_ex_falso_reuse",),
        )

    def _check_bindings_and_proof(
        self, request: ProgramWorldReuseRequest, key: ProgramWorldReuseKey
    ) -> ProgramWorldReuseRejection | None:
        candidate = request.candidate_key
        if candidate is None:
            return self._rejection(
                request,
                key,
                verdict=ReuseVerdict.ABSTAIN,
                reason_code=ReuseReasonCode.NO_EXACT_IDENTITY_MATCH,
                diagnostic="no candidate evidence presents an exact reuse key",
                limitations=(
                    "exact_reuse_required",
                    "raw_source_fallback_available" if request.raw_source_available else "",
                ),
                fallback=request.raw_source_available,
            )
        mismatches = _mismatched_bindings(key, candidate)
        if not mismatches:
            return self._proof_cache_freshness_for_exact(request, key)
        return self._proof_backed_or_mismatch(request, key, candidate, mismatches)

    def _proof_cache_freshness_for_exact(
        self, request: ProgramWorldReuseRequest, key: ProgramWorldReuseKey
    ) -> ProgramWorldReuseRejection | None:
        if not request.proof_cache_hit:
            return None
        if request.proof_cache_fresh and request.proof_admitted:
            return None
        if request.evidence_kind in {
            ReuseEvidenceKind.EXACT.value,
            ReuseEvidenceKind.UNSPECIFIED.value,
        }:
            return None
        return self._rejection(
            request,
            key,
            verdict=ReuseVerdict.REJECT,
            reason_code=ReuseReasonCode.PROOF_CACHE_REQUIRES_FRESH_ADMISSION,
            diagnostic=(
                "a proof-cache hit never grants reuse without current independent "
                "proof admission and freshness"
            ),
            limitations=(
                "cache_hit_never_suppresses_current_admission",
                "mandatory_validation_required",
            ),
            fallback=request.raw_source_available,
        )

    def _proof_backed_or_mismatch(
        self,
        request: ProgramWorldReuseRequest,
        key: ProgramWorldReuseKey,
        candidate: ProgramWorldReuseKey,
        mismatches: tuple[str, ...],
    ) -> ProgramWorldReuseRejection | None:
        extra = tuple(
            field for field in mismatches if field not in PROOF_BACKED_SUBSTITUTABLE_BINDINGS
        )
        wants_proof = request.evidence_kind == ReuseEvidenceKind.PROOF_BACKED.value or (
            request.independently_proof_backed or request.proof_admitted
        )
        if extra or not wants_proof:
            return self._rejection(
                request,
                key,
                verdict=ReuseVerdict.REJECT,
                reason_code=ReuseReasonCode.BINDING_MISMATCH,
                diagnostic=(
                    "single or multiple exact-binding mismatches refuse reuse: "
                    + ", ".join(mismatches)
                ),
                mismatched_bindings=mismatches,
                limitations=("exact_bindings_required",),
                fallback=request.raw_source_available,
            )
        return self._scoped_proof_rejection(request, key, candidate, mismatches)

    def _scoped_proof_rejection(
        self,
        request: ProgramWorldReuseRequest,
        key: ProgramWorldReuseKey,
        candidate: ProgramWorldReuseKey,
        mismatches: tuple[str, ...],
    ) -> ProgramWorldReuseRejection | None:
        if request.proof_cache_hit and not (
            request.proof_cache_fresh and request.proof_admitted
        ):
            return self._rejection(
                request,
                key,
                verdict=ReuseVerdict.REJECT,
                reason_code=ReuseReasonCode.PROOF_CACHE_REQUIRES_FRESH_ADMISSION,
                diagnostic=(
                    "proof-cache evidence still requires current independent "
                    "admission; a cache hit is not authority"
                ),
                mismatched_bindings=mismatches,
                limitations=(
                    "cache_hit_never_suppresses_current_admission",
                    "mandatory_validation_required",
                ),
                fallback=request.raw_source_available,
            )
        if not request.proof_admitted or request.proof_admission_evidence_cid is None:
            return self._rejection(
                request,
                key,
                verdict=ReuseVerdict.REJECT,
                reason_code=ReuseReasonCode.PROOF_CACHE_REQUIRES_FRESH_ADMISSION,
                diagnostic=(
                    "relation-based reuse requires independently admitted, "
                    "current proof evidence from a landed authority"
                ),
                mismatched_bindings=mismatches,
                limitations=("independently_proof_backed_required",),
                fallback=request.raw_source_available,
            )
        kind = (request.relation_kind or "").lower()
        if kind not in PROOF_BACKED_RELATION_KINDS:
            return self._rejection(
                request,
                key,
                verdict=ReuseVerdict.REJECT,
                reason_code=ReuseReasonCode.RELATION_SCOPE_EXCEEDED,
                diagnostic=(
                    "relation kind is outside the scoped equality/refinement/"
                    "equivalence families that may close a substitutable binding"
                ),
                mismatched_bindings=mismatches,
                limitations=("relation_kind_not_substitutable",),
            )
        if request.relation_status not in INDEPENDENT_RELATION_STATUSES:
            return self._rejection(
                request,
                key,
                verdict=ReuseVerdict.REJECT,
                reason_code=ReuseReasonCode.PROOF_CACHE_REQUIRES_FRESH_ADMISSION,
                diagnostic="relation status is not independently validated or proved",
                mismatched_bindings=mismatches,
                limitations=("relation_not_independently_admitted",),
            )
        if (
            request.relation_claim_cid is None
            or request.relation_scope_cid is None
            or not request.relation_assumptions_hold
        ):
            return self._rejection(
                request,
                key,
                verdict=ReuseVerdict.REJECT,
                reason_code=ReuseReasonCode.RELATION_SCOPE_EXCEEDED,
                diagnostic="relation-based reuse remains within explicit scope and assumptions",
                mismatched_bindings=mismatches,
                limitations=("explicit_scope_required", "assumptions_must_hold"),
            )
        if (
            request.relation_environment_cid is not None
            and request.relation_environment_cid != key.environment_cid
        ) or (
            request.relation_policy_cid is not None
            and request.relation_policy_cid != key.policy_cid
        ):
            return self._rejection(
                request,
                key,
                verdict=ReuseVerdict.REJECT,
                reason_code=ReuseReasonCode.RELATION_SCOPE_EXCEEDED,
                diagnostic="relation scope environment or policy does not match the current key",
                mismatched_bindings=mismatches,
                limitations=("scope_environment_and_policy_must_match",),
            )
        subjects = set(request.relation_subject_cids)
        for field in mismatches:
            query_value = getattr(key, field)
            candidate_value = getattr(candidate, field)
            needed = {value for value in (query_value, candidate_value) if value is not None}
            if needed - subjects:
                return self._rejection(
                    request,
                    key,
                    verdict=ReuseVerdict.REJECT,
                    reason_code=ReuseReasonCode.RELATION_SCOPE_EXCEEDED,
                    diagnostic=(
                        f"mismatched {field} values are not both inside the relation subject set"
                    ),
                    mismatched_bindings=mismatches,
                    limitations=("subject_cids_must_cover_substitutable_bindings",),
                )
        return None

    def _grant(
        self, request: ProgramWorldReuseRequest, key: ProgramWorldReuseKey
    ) -> ProgramWorldReuseEvaluation:
        candidate = request.candidate_key
        assert candidate is not None
        mismatches = _mismatched_bindings(key, candidate)
        exact = not mismatches
        proof_backed = not exact
        reason = (
            ReuseReasonCode.EXACT_IDENTITY_MATCH.value
            if exact
            else ReuseReasonCode.INDEPENDENTLY_PROOF_BACKED.value
        )
        limitations = [
            "reuse_is_proposal_until_supervisor_admission",
            "mandatory_validation_required",
            "cache_hit_never_suppresses_current_admission",
        ]
        if request.similarity_present:
            limitations.append("similarity_is_context_only")
        if request.proof_cache_hit:
            limitations.append("proof_cache_is_not_operational_admission")
        if proof_backed:
            limitations.append("relation_reuse_scoped_to_assumptions")
        admitted = request.admission_authority is not None
        decision = ProgramWorldReuseDecision(
            verdict=ReuseVerdict.REUSE,
            state_cid=key.state_cid,
            goal_cid=key.goal_cid,
            policy_cid=key.policy_cid,
            environment_cid=key.environment_cid,
            toolchain_cid=key.toolchain_cid,
            relation_claim_cid=request.relation_claim_cid,
            procedure_revision_cid=key.procedure_revision_cid,
            exact_match=True,
            reason_code=reason,
            proposal_only=not admitted,
            admitted=admitted,
            admission_authority=request.admission_authority,
            admission_evidence_cid=request.admission_evidence_cid,
            limitations=_limitations(*limitations),
        )
        why = (
            "current exact identity match on every reuse binding"
            if exact
            else (
                "independently proof-backed scoped evidence closes substitutable "
                "bindings within explicit assumptions"
            )
        )
        if request.similarity_present:
            why = why + "; similarity remains context-only"
        explanation = ProgramWorldReuseExplanation(
            verdict=ReuseVerdict.REUSE,
            reason_code=reason,
            why=why,
            exact_match=exact,
            independently_proof_backed=proof_backed,
            limitations=decision.limitations,
            context_only=request.similarity_present,
            required_admission=not admitted,
            decision_cid=decision.decision_cid,
        )
        return ProgramWorldReuseEvaluation(
            key=key, decision=decision, explanation=explanation
        )

    def _close(
        self,
        request: ProgramWorldReuseRequest,
        key: ProgramWorldReuseKey,
        rejection: ProgramWorldReuseRejection,
    ) -> ProgramWorldReuseEvaluation:
        memory = None
        if (
            rejection.verdict == ReuseVerdict.REJECT.value
            and rejection.reason_code != ReuseReasonCode.NEGATIVE_MEMORY_INVALIDATION.value
        ):
            memory = self._remember(request, rejection)
        elif rejection.reason_code == ReuseReasonCode.NEGATIVE_MEMORY_INVALIDATION.value:
            memory = self._matching_memory(key, request)
        fallback = rejection.raw_source_fallback or (
            rejection.reason_code == ReuseReasonCode.TYPED_UNAVAILABLE.value
        )
        reason = rejection.reason_code
        if (
            rejection.raw_source_fallback
            and rejection.verdict != ReuseVerdict.REUSE.value
            and request.source_cid is not None
            and reason
            in {
                ReuseReasonCode.UNSAFE_ABSTRACTION.value,
                ReuseReasonCode.INCOMPLETE_BINDING.value,
                ReuseReasonCode.TYPED_UNAVAILABLE.value,
            }
        ):
            limitations = _limitations(rejection.limitations, "raw_source_fallback")
        else:
            limitations = rejection.limitations
        decision = ProgramWorldReuseDecision(
            verdict=rejection.verdict,
            state_cid=key.state_cid,
            goal_cid=key.goal_cid,
            policy_cid=key.policy_cid,
            environment_cid=key.environment_cid,
            toolchain_cid=key.toolchain_cid,
            relation_claim_cid=request.relation_claim_cid,
            procedure_revision_cid=key.procedure_revision_cid,
            exact_match=False,
            reason_code=reason,
            fallback=fallback,
            limitations=limitations,
        )
        explanation = ProgramWorldReuseExplanation(
            verdict=rejection.verdict,
            reason_code=reason,
            why=rejection.diagnostic,
            exact_match=False,
            independently_proof_backed=False,
            mismatched_bindings=rejection.mismatched_bindings,
            limitations=decision.limitations,
            context_only=rejection.context_only,
            raw_source_fallback=rejection.raw_source_fallback,
            required_admission=True,
            negative_memory_cid=None if memory is None else memory.record_cid,
            decision_cid=decision.decision_cid,
            rejection_cid=rejection.rejection_cid,
        )
        return ProgramWorldReuseEvaluation(
            key=key,
            decision=decision,
            explanation=explanation,
            rejection=rejection,
            negative_memory=memory,
        )

    def _rejection(
        self,
        request: ProgramWorldReuseRequest,
        key: ProgramWorldReuseKey,
        *,
        verdict: ReuseVerdict,
        reason_code: ReuseReasonCode | str,
        diagnostic: str,
        mismatched_bindings: Sequence[str] = (),
        invalidator_cids: Sequence[str] = (),
        limitations: Sequence[str] = (),
        fallback: bool = False,
        context_only: bool = False,
    ) -> ProgramWorldReuseRejection:
        return ProgramWorldReuseRejection(
            key_cid=key.key_cid,
            candidate_key_cid=None
            if request.candidate_key is None
            else request.candidate_key.key_cid,
            verdict=verdict,
            reason_code=reason_code,
            diagnostic=diagnostic,
            mismatched_bindings=mismatched_bindings,
            invalidator_cids=invalidator_cids,
            limitations=_limitations(*limitations),
            raw_source_fallback=bool(fallback),
            context_only=context_only,
        )

    def _remember(
        self, request: ProgramWorldReuseRequest, rejection: ProgramWorldReuseRejection
    ) -> NegativeMemoryRecord:
        record = NegativeMemoryRecord(
            key_cid=rejection.key_cid,
            candidate_key_cid=rejection.candidate_key_cid,
            reason_code=rejection.reason_code,
            diagnostic=rejection.diagnostic,
            mismatched_bindings=rejection.mismatched_bindings,
            invalidator_cids=rejection.invalidator_cids or self._current_evidence_cids(request),
            rejection_cid=rejection.rejection_cid,
            proof_admission_evidence_cid=request.proof_admission_evidence_cid,
            active=True,
        )
        self._negative_memory[record.record_cid] = record
        return record

    def _matching_memory(
        self, key: ProgramWorldReuseKey, request: ProgramWorldReuseRequest
    ) -> NegativeMemoryRecord | None:
        candidate_cid = None if request.candidate_key is None else request.candidate_key.key_cid
        for record in self._negative_memory.values():
            if (
                record.active
                and record.key_cid == key.key_cid
                and record.candidate_key_cid == candidate_cid
            ):
                return record
        return None

    def _current_evidence_cids(self, request: ProgramWorldReuseRequest) -> tuple[str, ...]:
        return tuple(
            cid
            for cid in (
                request.proof_admission_evidence_cid,
                request.relation_claim_cid,
                request.candidate_world_root_cid,
                request.current_world_root_cid,
                request.abstraction_profile_cid,
                request.source_cid,
                None if request.candidate_key is None else request.candidate_key.key_cid,
            )
            if cid is not None
        )

    def _invalidators_lifted(
        self,
        record: NegativeMemoryRecord,
        request: ProgramWorldReuseRequest,
        current_evidence: tuple[str, ...],
    ) -> bool:
        if request.freshness == FreshnessState.FRESH.value and record.reason_code == (
            ReuseReasonCode.STALE_EVIDENCE.value
        ):
            if (
                request.current_world_root_cid is None
                or request.candidate_world_root_cid is None
                or request.current_world_root_cid == request.candidate_world_root_cid
            ) and request.relation_status not in STALE_RELATION_STATUSES:
                return True
        if record.reason_code == ReuseReasonCode.UNSAFE_ABSTRACTION.value:
            if (
                request.abstraction_sound is True
                and not request.unproved_omission
                and not request.unavailable_dimensions
            ):
                return True
        if record.reason_code == ReuseReasonCode.PROOF_CACHE_REQUIRES_FRESH_ADMISSION.value:
            if (
                request.proof_admitted
                and request.proof_admission_evidence_cid is not None
                and (
                    record.proof_admission_evidence_cid is None
                    or request.proof_admission_evidence_cid
                    != record.proof_admission_evidence_cid
                )
                and (not request.proof_cache_hit or request.proof_cache_fresh)
            ):
                return True
        if record.proof_admission_evidence_cid is not None:
            if (
                request.proof_admission_evidence_cid is not None
                and request.proof_admission_evidence_cid != record.proof_admission_evidence_cid
                and request.proof_admitted
            ):
                return True
        if record.invalidator_cids and set(record.invalidator_cids).isdisjoint(
            current_evidence
        ):
            if request.freshness == FreshnessState.FRESH.value:
                return True
        return False

    @staticmethod
    def _why_from_decision(decision: ProgramWorldReuseDecision) -> str:
        if decision.verdict == ReuseVerdict.REUSE.value:
            if decision.exact_match:
                return "current exact identity match on every reuse binding"
            return "independently proof-backed scoped evidence granted reuse"
        return decision.reason_code.replace("_", " ")


def evaluate_program_world_reuse(
    request: ProgramWorldReuseRequest | None = None,
    /,
    *,
    gate: ProgramWorldReuseGate | None = None,
    **kwargs: Any,
) -> ProgramWorldReuseEvaluation:
    """Evaluate exact or independently proof-backed reuse. Never self-admits."""

    active = gate if gate is not None else ProgramWorldReuseGate()
    if request is None:
        request = _request_from_kwargs(**kwargs)
    elif kwargs:
        raise ProgramWorldReuseError("pass a request object or keyword fields, not both")
    return active.evaluate(request)


def explain_program_world_reuse(
    target: ProgramWorldReuseEvaluation | ProgramWorldReuseRequest | ProgramWorldReuseDecision,
    /,
    *,
    gate: ProgramWorldReuseGate | None = None,
) -> ProgramWorldReuseExplanation:
    """Explain why reuse was granted or refused. Similarity stays context-only."""

    active = gate if gate is not None else ProgramWorldReuseGate()
    return active.explain(target)


def load_program_world_reuse_gate() -> ProgramWorldReuseGate:
    return ProgramWorldReuseGate()


def _request_from_kwargs(**kwargs: Any) -> ProgramWorldReuseRequest:
    if "key" in kwargs and isinstance(kwargs["key"], ProgramWorldReuseKey):
        key = kwargs.pop("key")
    else:
        required = {
            name: kwargs.pop(name)
            for name in (
                "state_cid",
                "goal_cid",
                "policy_cid",
                "environment_cid",
                "toolchain_cid",
            )
            if name in kwargs
        }
        optional = {
            name: kwargs.pop(name)
            for name in (
                "obligation_cids",
                "selection_cids",
                "procedure_revision_cid",
                "validation_dependency_cids",
            )
            if name in kwargs
        }
        key = ProgramWorldReuseKey(**required, **optional)
    candidate = kwargs.pop("candidate_key", None)
    if candidate is None and "candidate_state_cid" in kwargs:
        candidate = key.replace(state_cid=kwargs.pop("candidate_state_cid"))
    return ProgramWorldReuseRequest(key=key, candidate_key=candidate, **kwargs)


def program_world_reuse_module_source() -> str:
    """Return this module's source for AST audits. Performs no I/O besides read."""

    from pathlib import Path

    return Path(__file__).read_text(encoding="utf-8")


def assert_no_duplicate_reuse_authority(source: str | None = None) -> None:
    """Fail if this module redefines datasets, kit, proof-cache, or compiler types."""

    tree = ast.parse(source or program_world_reuse_module_source())
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    overlap = defined & BANNED_DUPLICATE_TYPES
    if overlap:
        raise ProgramWorldReuseError(
            "reuse gate must not duplicate landed identity, cache, or compiler "
            f"types {sorted(overlap)}"
        )


__all__ = [
    "DATASETS_ABSTRACTION_AUTHORITY",
    "EXACT_BINDING_FIELDS",
    "GATE_ID",
    "GATE_OWNED_AUTHORITIES",
    "LANDED_PROOF_AUTHORITIES",
    "LOCATED_LANDED_AUTHORITIES",
    "NEGATIVE_MEMORY_RECORD_INTERFACE",
    "PROGRAM_WORLD_REUSE_EVALUATION_INTERFACE",
    "PROGRAM_WORLD_REUSE_EXPLANATION_INTERFACE",
    "PROGRAM_WORLD_REUSE_GATE_INTERFACE",
    "PROGRAM_WORLD_REUSE_KEY_INTERFACE",
    "PROGRAM_WORLD_REUSE_REJECTION_INTERFACE",
    "PROOF_BACKED_SUBSTITUTABLE_BINDINGS",
    "RECEIPT_CACHE_AUTHORITY",
    "SAWM_EXACT_REUSE_EVIDENCE",
    "SAWM_REUSE_ADMISSION_EVIDENCE",
    "SCALAR_BINDING_FIELDS",
    "TEST_PROOF_CACHE_AUTHORITY",
    "TUPLE_BINDING_FIELDS",
    "NegativeMemoryRecord",
    "ProgramWorldReuseError",
    "ProgramWorldReuseEvaluation",
    "ProgramWorldReuseExplanation",
    "ProgramWorldReuseGate",
    "ProgramWorldReuseKey",
    "ProgramWorldReuseRejection",
    "ProgramWorldReuseRequest",
    "ReuseEvidenceKind",
    "ReuseReasonCode",
    "assert_no_duplicate_reuse_authority",
    "evaluate_program_world_reuse",
    "explain_program_world_reuse",
    "load_program_world_reuse_gate",
]
