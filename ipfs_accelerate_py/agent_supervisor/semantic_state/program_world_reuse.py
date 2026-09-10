"""Exact state, transition, procedure, and proof reuse gates (SAWM-016).

Interface: ``ProgramWorldReuseGate@1``
Evidence: ``sawm/exact-reuse@1``, ``sawm/reuse-admission@1``

Accelerator operational reuse admission.  The gate binds an exact reuse key
over state, goal, policy, environment, toolchain, obligations, selections,
procedure revision, and validation dependencies.  Only a current exact hit or
independently proof-backed scoped evidence can grant reuse.  Similarity is
context-only.  Stale, unsafe, incomplete, or environment-incompatible
evidence fails closed and explains why.

This module does not validate proofs, relations, procedures, or world roots.
It cites landed cache/procedure/proof authorities and never admits its own
output.  A cache hit never suppresses current admission or mandatory
validation.  Relation reuse stays inside explicit scope and assumptions.
Raw-source fallback remains available when a surface is typed-unavailable.

Importing this module performs no I/O, starts no threads, and does not import
datasets or kit implementations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    UnavailableResult,
    _bool,
    _closed,
    _text,
    validate_opaque_cid,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_receipts import (
    ADAPTER_OWNED_AUTHORITIES,
    ANN_REASON_CODES,
    EVENT_AUTHORITY,
    FORBIDDEN_IDENTITY_FIELDS,
    MERGE_AUTHORITY,
    OPERATIONAL_ACCEPTANCE_AUTHORITIES,
    PROGRAM_WORLD_REUSE_DECISION_INTERFACE,
    VALIDATION_AUTHORITY,
    FreshnessState,
    ProgramWorldReceiptError,
    ProgramWorldReuseDecision,
    ReuseVerdict,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload


PROGRAM_WORLD_REUSE_GATE_INTERFACE: Final[str] = "ProgramWorldReuseGate@1"
PROGRAM_WORLD_REUSE_KEY_INTERFACE: Final[str] = "ProgramWorldReuseKey@1"
PROGRAM_WORLD_REUSE_DECISION_INTERFACE_REF: Final[str] = (
    PROGRAM_WORLD_REUSE_DECISION_INTERFACE
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
REUSE_STAGE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-reuse-stage-receipt@1"
)

GATE_ID: Final[str] = "ipfs-accelerate.semantic-state.program-world-reuse-gate"
PRODUCER_ID: Final[str] = "program-world-reuse@1"
GATE_VERSION: Final[str] = "1"

MAX_TEXT_CHARS: Final[int] = 512
MAX_COLLECTION_ITEMS: Final[int] = 10_000
MAX_STAGE_RECEIPTS: Final[int] = 64

GATE_OWNED_AUTHORITIES: Final[frozenset[str]] = frozenset(
    {
        GATE_ID,
        PROGRAM_WORLD_REUSE_GATE_INTERFACE,
        "ProgramWorldReuseGate",
        "ProgramWorldReuseKey",
        "evaluate_program_world_reuse",
        "explain_program_world_reuse",
        PRODUCER_ID,
    }
)

SCALAR_BINDINGS: Final[tuple[str, ...]] = (
    "state_cid",
    "goal_cid",
    "policy_cid",
    "environment_cid",
    "toolchain_cid",
    "procedure_revision_cid",
)
SET_BINDINGS: Final[tuple[str, ...]] = (
    "obligation_cids",
    "selection_cids",
    "validation_dependency_cids",
)
REUSE_KEY_BINDINGS: Final[tuple[str, ...]] = SCALAR_BINDINGS + SET_BINDINGS

BINDING_MISMATCH_REASONS: Final[Mapping[str, str]] = {
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
PROOF_BACKED_AUTHORITY_STATUSES: Final[frozenset[str]] = frozenset(
    {"proved", "validated"}
)
UNSAFE_ABSTRACTION_CLAIMS: Final[frozenset[str]] = frozenset(
    {"unknown", "under_approximation", "unsound"}
)
STALE_AUTHORITY_STATUSES: Final[frozenset[str]] = frozenset(
    {"stale", "superseded", "refuted", "unknown", "candidate", "asserted"}
)


class ProgramWorldReuseError(ProgramWorldReceiptError):
    """Closed reuse-gate contract violation."""


class ReuseEvidenceKind(str, Enum):
    EXACT = "exact"
    PROOF_BACKED = "proof_backed"
    PROCEDURE = "procedure"
    CACHE = "cache"
    ABSTRACT = "abstract"
    SIMILARITY = "similarity"
    UNAVAILABLE = "unavailable"


class ReuseStage(str, Enum):
    CAPABILITY = "capability"
    COMPLETENESS = "completeness"
    SIMILARITY = "similarity"
    NEGATIVE_MEMORY = "negative_memory"
    BINDING_IDENTITY = "binding_identity"
    FRESHNESS = "freshness"
    ENVIRONMENT = "environment"
    ABSTRACTION = "abstraction"
    PROOF_ADMISSION = "proof_admission"
    SCOPE = "scope"
    VALIDATION = "validation"
    RAW_FALLBACK = "raw_fallback"


class ReuseStageDisposition(str, Enum):
    SELECTED = "selected"
    SKIPPED = "skipped"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


class ReuseReasonCode(str, Enum):
    EXACT_IDENTITY_MATCH = "exact_identity_match"
    PROOF_BACKED_SCOPED_MATCH = "proof_backed_scoped_match"
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
    STALE_BINDING = "stale_binding"
    FRESHNESS_UNKNOWN = "freshness_unknown"
    UNSAFE_ABSTRACTION = "unsafe_abstraction"
    INCOMPLETE_BINDINGS = "incomplete_bindings"
    ENVIRONMENT_INCOMPATIBLE = "environment_incompatible"
    SIMILARITY_IS_NOT_REUSE = "similarity_is_not_reuse"
    ANN_NOT_AUTHORITATIVE = "ann_not_authoritative"
    PROOF_CACHE_REQUIRES_FRESH_ADMISSION = "proof_cache_requires_fresh_admission"
    CACHE_HIT_DOES_NOT_SUPPRESS_VALIDATION = "cache_hit_does_not_suppress_validation"
    TYPED_UNAVAILABLE = "typed_unavailable"
    RAW_SOURCE_FALLBACK = "raw_source_fallback"
    NEGATIVE_MEMORY = "negative_memory"
    SCOPE_MISMATCH = "scope_mismatch"
    ASSUMPTION_MISMATCH = "assumption_mismatch"
    RELATION_NOT_PROVED = "relation_not_proved"
    SELF_ADMISSION_FORBIDDEN = "self_admission_forbidden"
    UNKNOWN_WIDENED = "unknown_widened"


REUSE_EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in ReuseEvidenceKind
)
REUSE_STAGES: Final[frozenset[str]] = frozenset(item.value for item in ReuseStage)
REUSE_STAGE_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    item.value for item in ReuseStageDisposition
)
REUSE_REASON_CODES: Final[frozenset[str]] = frozenset(
    item.value for item in ReuseReasonCode
) | frozenset(BINDING_MISMATCH_REASONS.values())
FRESHNESS_STATES: Final[frozenset[str]] = frozenset(
    item.value for item in FreshnessState
)
STAGE_ORDER: Final[tuple[ReuseStage, ...]] = tuple(ReuseStage)

_REASON_DIAGNOSTICS: Final[Mapping[str, str]] = {
    "exact_identity_match": (
        "current exact key bindings match; reuse is granted without treating "
        "similarity as authority"
    ),
    "proof_backed_scoped_match": (
        "independently proof-backed scoped evidence covers the single binding "
        "difference inside explicit scope and assumptions"
    ),
    "no_exact_identity_match": (
        "no current exact or independently proof-backed scoped evidence is present"
    ),
    "binding_mismatch": (
        "one or more reuse-key bindings differ; similarity cannot close the gap"
    ),
    "state_mismatch": "state_cid differs from the current exact key",
    "goal_mismatch": "goal_cid differs from the current exact key",
    "policy_mismatch": "policy_cid differs from the current exact key",
    "environment_mismatch": "environment_cid differs from the current exact key",
    "toolchain_mismatch": "toolchain_cid differs from the current exact key",
    "obligation_mismatch": "obligation_cids differ from the current exact key",
    "selection_mismatch": "selection_cids differ from the current exact key",
    "procedure_revision_mismatch": (
        "procedure_revision_cid differs from the current exact key"
    ),
    "validation_dependency_mismatch": (
        "validation_dependency_cids differ from the current exact key"
    ),
    "stale_binding": (
        "stale or superseded bindings revoke reuse; current freshness is required"
    ),
    "freshness_unknown": (
        "unknown freshness widens rather than granting reuse"
    ),
    "unsafe_abstraction": (
        "unsafe abstraction: unknown, under-approximate, or unproved omission "
        "cannot admit reuse"
    ),
    "incomplete_bindings": (
        "incomplete evidence fails closed; missing dimensions are not absence"
    ),
    "environment_incompatible": (
        "environment-incompatible evidence fails closed and cannot be reused"
    ),
    "similarity_is_not_reuse": (
        "similarity is context-only and cannot grant exact or proof-backed reuse"
    ),
    "ann_not_authoritative": (
        "ANN/nearest-neighbor scores are advisory and cannot admit reuse"
    ),
    "proof_cache_requires_fresh_admission": (
        "a proof-cache hit still requires current independent fresh admission evidence"
    ),
    "cache_hit_does_not_suppress_validation": (
        "a cache hit never suppresses current admission or mandatory validation"
    ),
    "typed_unavailable": (
        "named capability is typed-unavailable; raw-source fallback remains open"
    ),
    "raw_source_fallback": (
        "authoritative reuse is unavailable; raw-source fallback remains available"
    ),
    "negative_memory": (
        "scoped negative memory still invalidates this candidate until its "
        "invalidators are cleared by new evidence"
    ),
    "scope_mismatch": (
        "relation-based reuse remains inside explicit scope; this scope does not cover the query"
    ),
    "assumption_mismatch": (
        "relation assumptions are not a subset of the current query assumptions"
    ),
    "relation_not_proved": (
        "relation authority is not independently proved or validated for this scope"
    ),
    "self_admission_forbidden": (
        "the reuse gate cannot admit its own output or validate its own proof"
    ),
    "unknown_widened": (
        "unknown dynamic or relevance dimensions widen and fail closed"
    ),
}


def _bounded_text(value: Any, name: str) -> str:
    text = _text(value, name)
    if len(text) > MAX_TEXT_CHARS:
        raise ProgramWorldReuseError(f"{name} exceeds {MAX_TEXT_CHARS} characters")
    return text


def _optional_bounded_text(value: Any, name: str) -> str | None:
    if value is None or value == "":
        return None
    return _bounded_text(value, name)


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


def _reject_forbidden_fields(mapping: Mapping[str, Any], name: str) -> None:
    if not isinstance(mapping, Mapping):
        raise ProgramWorldReuseError(f"{name} must be an object")
    forbidden = set(mapping) & FORBIDDEN_IDENTITY_FIELDS
    if forbidden:
        raise ProgramWorldReuseError(
            f"{name} rejects ANN/score identity fields {sorted(forbidden)}"
        )


def _payload_cid(payload: Mapping[str, Any]) -> str:
    _reject_forbidden_fields(payload, "identity_payload")
    return cid_for_payload(payload)


def _is_gate_owned_authority(authority: str | None) -> bool:
    if authority is None:
        return False
    return authority in GATE_OWNED_AUTHORITIES or authority in ADAPTER_OWNED_AUTHORITIES


def _independent_admission(authority: str | None, evidence_cid: str | None) -> bool:
    if authority is None or evidence_cid is None:
        return False
    if _is_gate_owned_authority(authority):
        return False
    return authority in OPERATIONAL_ACCEPTANCE_AUTHORITIES


def diagnostic_for(reason_code: str, *, extra: str | None = None) -> str:
    base = _REASON_DIAGNOSTICS.get(reason_code, reason_code.replace("_", " "))
    if extra:
        text = f"{base}: {extra}"
    else:
        text = base
    return text[:MAX_TEXT_CHARS]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseKey:
    """Exact reuse identity over every declared operational binding."""

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
            "reuse_key_cid",
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
            "obligation_cids",
            _unique_sorted_cids(self.obligation_cids, "obligation_cid"),
        )
        object.__setattr__(
            self,
            "selection_cids",
            _unique_sorted_cids(self.selection_cids, "selection_cid"),
        )
        object.__setattr__(
            self,
            "procedure_revision_cid",
            _optional_cid(self.procedure_revision_cid, "procedure_revision_cid"),
        )
        object.__setattr__(
            self,
            "validation_dependency_cids",
            _unique_sorted_cids(
                self.validation_dependency_cids, "validation_dependency_cid"
            ),
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
    def reuse_key_cid(self) -> str:
        return _payload_cid(self.identity_payload())

    def binding(self, name: str) -> Any:
        if name not in REUSE_KEY_BINDINGS:
            raise ProgramWorldReuseError(f"unknown reuse-key binding {name!r}")
        return getattr(self, name)

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["reuse_key_cid"] = self.reuse_key_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgramWorldReuseKey":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("reuse_key_cid")
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReuseError("unsupported reuse-key schema")
        if payload.pop("interface") != cls.INTERFACE:
            raise ProgramWorldReuseError("unsupported reuse-key interface")
        result = cls(**payload)
        if result.reuse_key_cid != claimed:
            raise ProgramWorldReuseError("reuse_key_cid does not rehash")
        return result


def _coerce_key(value: ProgramWorldReuseKey | Mapping[str, Any]) -> ProgramWorldReuseKey:
    if isinstance(value, ProgramWorldReuseKey):
        return value
    if isinstance(value, Mapping):
        if "reuse_key_cid" in value:
            return ProgramWorldReuseKey.from_dict(value)
        fields = {
            name: value[name]
            for name in (
                "state_cid",
                "goal_cid",
                "policy_cid",
                "environment_cid",
                "toolchain_cid",
            )
        }
        optional = {
            "obligation_cids": tuple(value.get("obligation_cids") or ()),
            "selection_cids": tuple(value.get("selection_cids") or ()),
            "procedure_revision_cid": value.get("procedure_revision_cid"),
            "validation_dependency_cids": tuple(
                value.get("validation_dependency_cids") or ()
            ),
        }
        return ProgramWorldReuseKey(**fields, **optional)
    raise ProgramWorldReuseError("reuse key must be ProgramWorldReuseKey or object")


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseCandidate:
    """Evidence offered for reuse against a current exact key."""

    key: ProgramWorldReuseKey
    evidence_kind: ReuseEvidenceKind | str = ReuseEvidenceKind.EXACT
    freshness: FreshnessState | str = FreshnessState.FRESH
    proof_cache_hit: bool = False
    proof_admission_authority: str | None = None
    proof_admission_evidence_cid: str | None = None
    relation_claim_cid: str | None = None
    relation_kind: str | None = None
    relation_authority_status: str | None = None
    relation_scope_cid: str | None = None
    relation_assumption_cids: Sequence[str] = ()
    relation_environment_cid: str | None = None
    relation_policy_cid: str | None = None
    relation_left_cid: str | None = None
    relation_right_cid: str | None = None
    query_scope_cid: str | None = None
    query_assumption_cids: Sequence[str] = ()
    abstraction_soundness_claim: str | None = None
    abstraction_reuse_admitted: bool | None = None
    unknown_relevance_dimensions: Sequence[str] = ()
    ann_candidates: Sequence[Any] = ()
    similarity_only: bool = False
    capability_available: bool = True
    capability_reason_code: str | None = None
    capability_surface: str | None = None
    raw_source_cid: str | None = None
    incomplete: bool = False
    suppress_validation: bool = False
    invalidator_cids: Sequence[str] = ()
    environment_compatible: bool = True
    candidate_evidence_cid: str | None = None

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_REUSE_CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _coerce_key(self.key))
        object.__setattr__(
            self,
            "evidence_kind",
            _enum_value(self.evidence_kind, REUSE_EVIDENCE_KINDS, "evidence_kind"),
        )
        object.__setattr__(
            self,
            "freshness",
            _enum_value(self.freshness, FRESHNESS_STATES, "freshness"),
        )
        object.__setattr__(
            self, "proof_cache_hit", _bool(self.proof_cache_hit, "proof_cache_hit")
        )
        object.__setattr__(
            self,
            "proof_admission_authority",
            _optional_bounded_text(
                self.proof_admission_authority, "proof_admission_authority"
            ),
        )
        object.__setattr__(
            self,
            "proof_admission_evidence_cid",
            _optional_cid(
                self.proof_admission_evidence_cid, "proof_admission_evidence_cid"
            ),
        )
        for name in (
            "relation_claim_cid",
            "relation_scope_cid",
            "relation_environment_cid",
            "relation_policy_cid",
            "relation_left_cid",
            "relation_right_cid",
            "query_scope_cid",
            "raw_source_cid",
            "candidate_evidence_cid",
        ):
            object.__setattr__(self, name, _optional_cid(getattr(self, name), name))
        for name in ("relation_kind", "relation_authority_status", "capability_surface"):
            object.__setattr__(
                self, name, _optional_bounded_text(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "abstraction_soundness_claim",
            _optional_bounded_text(
                self.abstraction_soundness_claim, "abstraction_soundness_claim"
            ),
        )
        if self.abstraction_reuse_admitted is not None:
            object.__setattr__(
                self,
                "abstraction_reuse_admitted",
                _bool(self.abstraction_reuse_admitted, "abstraction_reuse_admitted"),
            )
        object.__setattr__(
            self,
            "relation_assumption_cids",
            _unique_sorted_cids(self.relation_assumption_cids, "relation_assumption_cid"),
        )
        object.__setattr__(
            self,
            "query_assumption_cids",
            _unique_sorted_cids(self.query_assumption_cids, "query_assumption_cid"),
        )
        object.__setattr__(
            self,
            "unknown_relevance_dimensions",
            _unique_sorted_texts(
                self.unknown_relevance_dimensions, "unknown_relevance_dimension"
            ),
        )
        object.__setattr__(
            self,
            "invalidator_cids",
            _unique_sorted_cids(self.invalidator_cids, "invalidator_cid"),
        )
        if not isinstance(self.ann_candidates, Sequence) or isinstance(
            self.ann_candidates, (str, bytes, bytearray)
        ):
            raise ProgramWorldReuseError("ann_candidates must be a sequence")
        object.__setattr__(self, "ann_candidates", tuple(self.ann_candidates))
        object.__setattr__(
            self, "similarity_only", _bool(self.similarity_only, "similarity_only")
        )
        object.__setattr__(
            self,
            "capability_available",
            _bool(self.capability_available, "capability_available"),
        )
        object.__setattr__(
            self,
            "capability_reason_code",
            _optional_bounded_text(
                self.capability_reason_code, "capability_reason_code"
            ),
        )
        object.__setattr__(self, "incomplete", _bool(self.incomplete, "incomplete"))
        object.__setattr__(
            self,
            "suppress_validation",
            _bool(self.suppress_validation, "suppress_validation"),
        )
        object.__setattr__(
            self,
            "environment_compatible",
            _bool(self.environment_compatible, "environment_compatible"),
        )
        object.__setattr__(
            self, "candidate_evidence_cid", _payload_cid(self.identity_payload())
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "key_cid": self.key.reuse_key_cid,
            "evidence_kind": self.evidence_kind,
            "freshness": self.freshness,
            "proof_cache_hit": self.proof_cache_hit,
            "proof_admission_authority": self.proof_admission_authority,
            "proof_admission_evidence_cid": self.proof_admission_evidence_cid,
            "relation_claim_cid": self.relation_claim_cid,
            "relation_kind": self.relation_kind,
            "relation_authority_status": self.relation_authority_status,
            "relation_scope_cid": self.relation_scope_cid,
            "relation_assumption_cids": list(self.relation_assumption_cids),
            "relation_environment_cid": self.relation_environment_cid,
            "relation_policy_cid": self.relation_policy_cid,
            "relation_left_cid": self.relation_left_cid,
            "relation_right_cid": self.relation_right_cid,
            "query_scope_cid": self.query_scope_cid,
            "query_assumption_cids": list(self.query_assumption_cids),
            "abstraction_soundness_claim": self.abstraction_soundness_claim,
            "abstraction_reuse_admitted": self.abstraction_reuse_admitted,
            "unknown_relevance_dimensions": list(self.unknown_relevance_dimensions),
            "ann_candidate_count": len(self.ann_candidates),
            "similarity_only": self.similarity_only,
            "capability_available": self.capability_available,
            "capability_reason_code": self.capability_reason_code,
            "capability_surface": self.capability_surface,
            "raw_source_cid": self.raw_source_cid,
            "incomplete": self.incomplete,
            "suppress_validation": self.suppress_validation,
            "invalidator_cids": list(self.invalidator_cids),
            "environment_compatible": self.environment_compatible,
        }


def _coerce_candidate(
    value: ProgramWorldReuseCandidate | Mapping[str, Any] | None,
    *,
    default_key: ProgramWorldReuseKey | None = None,
) -> ProgramWorldReuseCandidate | None:
    if value is None:
        return None
    if isinstance(value, ProgramWorldReuseCandidate):
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        if "key" not in payload:
            if default_key is None:
                raise ProgramWorldReuseError("candidate requires a reuse key")
            payload["key"] = default_key
        return ProgramWorldReuseCandidate(**payload)
    raise ProgramWorldReuseError(
        "candidate must be ProgramWorldReuseCandidate, object, or omitted"
    )


@dataclass(frozen=True, slots=True)
class ReuseStageReceipt:
    """One recorded reuse-gate stage outcome."""

    stage: ReuseStage | str
    disposition: ReuseStageDisposition | str
    reason_code: str
    diagnostic: str = ""

    SCHEMA: ClassVar[str] = REUSE_STAGE_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stage", _enum_value(self.stage, REUSE_STAGES, "stage")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum_value(self.disposition, REUSE_STAGE_DISPOSITIONS, "disposition"),
        )
        object.__setattr__(
            self, "reason_code", _bounded_text(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self,
            "diagnostic",
            _optional_bounded_text(self.diagnostic, "diagnostic") or "",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "stage": self.stage,
            "disposition": self.disposition,
            "reason_code": self.reason_code,
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseRejection:
    """Fail-closed reuse rejection with an explicit reason and mismatched bindings."""

    reason_code: str
    diagnostic: str
    mismatched_bindings: Sequence[str] = ()
    verdict: ReuseVerdict | str = ReuseVerdict.REJECT
    limitations: Sequence[str] = ()
    raw_fallback: bool = False
    invalidator_cids: Sequence[str] = ()

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_REUSE_REJECTION_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "reason_code",
            "diagnostic",
            "mismatched_bindings",
            "verdict",
            "limitations",
            "raw_fallback",
            "invalidator_cids",
            "rejection_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reason_code", _bounded_text(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self, "diagnostic", _bounded_text(self.diagnostic, "diagnostic")
        )
        object.__setattr__(
            self,
            "mismatched_bindings",
            _unique_sorted_texts(self.mismatched_bindings, "mismatched_binding"),
        )
        extra = set(self.mismatched_bindings) - set(REUSE_KEY_BINDINGS)
        if extra:
            raise ProgramWorldReuseError(
                f"mismatched_bindings contains unknown bindings {sorted(extra)}"
            )
        object.__setattr__(
            self,
            "verdict",
            _enum_value(
                self.verdict,
                frozenset(item.value for item in ReuseVerdict),
                "verdict",
            ),
        )
        if self.verdict == ReuseVerdict.REUSE.value:
            raise ProgramWorldReuseError("reuse cannot be recorded as a rejection")
        object.__setattr__(
            self,
            "limitations",
            _unique_sorted_texts(self.limitations, "limitation"),
        )
        object.__setattr__(self, "raw_fallback", _bool(self.raw_fallback, "raw_fallback"))
        object.__setattr__(
            self,
            "invalidator_cids",
            _unique_sorted_cids(self.invalidator_cids, "invalidator_cid"),
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "reason_code": self.reason_code,
            "diagnostic": self.diagnostic,
            "mismatched_bindings": list(self.mismatched_bindings),
            "verdict": self.verdict,
            "limitations": list(self.limitations),
            "raw_fallback": self.raw_fallback,
            "invalidator_cids": list(self.invalidator_cids),
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
        result = cls(**payload)
        if result.rejection_cid != claimed:
            raise ProgramWorldReuseError("rejection_cid does not rehash")
        return result


@dataclass(frozen=True, slots=True)
class NegativeMemoryRecord:
    """Scoped rejection memory. It never grants reuse and is independently invalidatable."""

    key_cid: str
    reason_code: str
    diagnostic: str
    mismatched_bindings: Sequence[str] = ()
    invalidator_cids: Sequence[str] = ()
    candidate_evidence_cid: str | None = None
    rejection_cid: str | None = None

    SCHEMA: ClassVar[str] = NEGATIVE_MEMORY_RECORD_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "key_cid",
            "reason_code",
            "diagnostic",
            "mismatched_bindings",
            "invalidator_cids",
            "candidate_evidence_cid",
            "rejection_cid",
            "record_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_cid", validate_opaque_cid(self.key_cid, "key_cid"))
        object.__setattr__(
            self, "reason_code", _bounded_text(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self, "diagnostic", _bounded_text(self.diagnostic, "diagnostic")
        )
        object.__setattr__(
            self,
            "mismatched_bindings",
            _unique_sorted_texts(self.mismatched_bindings, "mismatched_binding"),
        )
        object.__setattr__(
            self,
            "invalidator_cids",
            _unique_sorted_cids(self.invalidator_cids, "invalidator_cid"),
        )
        object.__setattr__(
            self,
            "candidate_evidence_cid",
            _optional_cid(self.candidate_evidence_cid, "candidate_evidence_cid"),
        )
        object.__setattr__(
            self, "rejection_cid", _optional_cid(self.rejection_cid, "rejection_cid")
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "key_cid": self.key_cid,
            "reason_code": self.reason_code,
            "diagnostic": self.diagnostic,
            "mismatched_bindings": list(self.mismatched_bindings),
            "invalidator_cids": list(self.invalidator_cids),
            "candidate_evidence_cid": self.candidate_evidence_cid,
            "rejection_cid": self.rejection_cid,
        }

    @property
    def record_cid(self) -> str:
        return _payload_cid(self.identity_payload())

    def still_in_force(
        self, query: ProgramWorldReuseKey, candidate: ProgramWorldReuseCandidate
    ) -> bool:
        if self.key_cid != query.reuse_key_cid:
            return False
        if (
            self.candidate_evidence_cid is not None
            and candidate.candidate_evidence_cid != self.candidate_evidence_cid
        ):
            return False
        if not self.invalidator_cids:
            return True
        current = set(candidate.invalidator_cids)
        return set(self.invalidator_cids) <= current

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
        result = cls(**payload)
        if result.record_cid != claimed:
            raise ProgramWorldReuseError("record_cid does not rehash")
        return result


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseEvaluation:
    """Closed operational evaluation of one reuse query against one candidate."""

    query: ProgramWorldReuseKey
    verdict: ReuseVerdict | str
    reason_code: str
    diagnostic: str
    decision: ProgramWorldReuseDecision
    candidate: ProgramWorldReuseCandidate | None = None
    rejection: ProgramWorldReuseRejection | None = None
    negative_memory: NegativeMemoryRecord | None = None
    mismatched_bindings: Sequence[str] = ()
    stages: Sequence[ReuseStageReceipt] = ()
    raw_fallback: bool = False
    granted: bool = False
    proof_backed: bool = False
    similarity_context_only: bool = False

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_REUSE_EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", _coerce_key(self.query))
        object.__setattr__(
            self,
            "verdict",
            _enum_value(
                self.verdict,
                frozenset(item.value for item in ReuseVerdict),
                "verdict",
            ),
        )
        object.__setattr__(
            self, "reason_code", _bounded_text(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self, "diagnostic", _bounded_text(self.diagnostic, "diagnostic")
        )
        if not isinstance(self.decision, ProgramWorldReuseDecision):
            raise ProgramWorldReuseError("evaluation requires ProgramWorldReuseDecision")
        if self.candidate is not None and not isinstance(
            self.candidate, ProgramWorldReuseCandidate
        ):
            raise ProgramWorldReuseError("candidate must be ProgramWorldReuseCandidate")
        if self.rejection is not None and not isinstance(
            self.rejection, ProgramWorldReuseRejection
        ):
            raise ProgramWorldReuseError("rejection must be ProgramWorldReuseRejection")
        if self.negative_memory is not None and not isinstance(
            self.negative_memory, NegativeMemoryRecord
        ):
            raise ProgramWorldReuseError(
                "negative_memory must be NegativeMemoryRecord"
            )
        object.__setattr__(
            self,
            "mismatched_bindings",
            _unique_sorted_texts(self.mismatched_bindings, "mismatched_binding"),
        )
        stages = tuple(self.stages)
        if len(stages) > MAX_STAGE_RECEIPTS:
            raise ProgramWorldReuseError("stage receipts exceed bound")
        for receipt in stages:
            if not isinstance(receipt, ReuseStageReceipt):
                raise ProgramWorldReuseError("stages must contain ReuseStageReceipt")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "raw_fallback", _bool(self.raw_fallback, "raw_fallback"))
        object.__setattr__(self, "granted", _bool(self.granted, "granted"))
        object.__setattr__(self, "proof_backed", _bool(self.proof_backed, "proof_backed"))
        object.__setattr__(
            self,
            "similarity_context_only",
            _bool(self.similarity_context_only, "similarity_context_only"),
        )
        if self.granted and self.verdict != ReuseVerdict.REUSE.value:
            raise ProgramWorldReuseError("granted reuse requires verdict=reuse")
        if self.granted and self.rejection is not None:
            raise ProgramWorldReuseError("granted reuse cannot carry a rejection")
        if self.decision.verdict != self.verdict:
            raise ProgramWorldReuseError("evaluation verdict must match decision verdict")
        if self.decision.reason_code != self.reason_code:
            raise ProgramWorldReuseError(
                "evaluation reason_code must match decision reason_code"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": PROGRAM_WORLD_REUSE_GATE_INTERFACE,
            "query_cid": self.query.reuse_key_cid,
            "verdict": self.verdict,
            "reason_code": self.reason_code,
            "diagnostic": self.diagnostic,
            "decision_cid": self.decision.decision_cid,
            "candidate_evidence_cid": (
                None if self.candidate is None else self.candidate.candidate_evidence_cid
            ),
            "rejection_cid": None if self.rejection is None else self.rejection.rejection_cid,
            "negative_memory_cid": (
                None if self.negative_memory is None else self.negative_memory.record_cid
            ),
            "mismatched_bindings": list(self.mismatched_bindings),
            "stages": [item.to_dict() for item in self.stages],
            "raw_fallback": self.raw_fallback,
            "granted": self.granted,
            "proof_backed": self.proof_backed,
            "similarity_context_only": self.similarity_context_only,
        }

    @property
    def evaluation_cid(self) -> str:
        return _payload_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["evaluation_cid"] = self.evaluation_cid
        value["query"] = self.query.to_dict()
        value["decision"] = self.decision.to_dict()
        value["rejection"] = None if self.rejection is None else self.rejection.to_dict()
        value["negative_memory"] = (
            None if self.negative_memory is None else self.negative_memory.to_dict()
        )
        return value

    def to_unavailable_result(self) -> UnavailableResult:
        return UnavailableResult(
            operation="evaluate_program_world_reuse",
            adapter_id=GATE_ID,
            reason_code=self.reason_code,
            retryable=False,
            diagnostic=self.diagnostic,
        )


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseExplanation:
    """Deterministic explanation of why reuse was granted or failed closed."""

    evaluation_cid: str
    verdict: str
    reason_code: str
    diagnostic: str
    mismatched_bindings: Sequence[str]
    granted: bool
    raw_fallback: bool
    similarity_context_only: bool
    negative_memory_applied: bool
    proof_backed: bool
    limitations: Sequence[str]
    stages: Sequence[Mapping[str, Any]]

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_REUSE_EXPLANATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluation_cid",
            validate_opaque_cid(self.evaluation_cid, "evaluation_cid"),
        )
        object.__setattr__(self, "verdict", _bounded_text(self.verdict, "verdict"))
        object.__setattr__(
            self, "reason_code", _bounded_text(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self, "diagnostic", _bounded_text(self.diagnostic, "diagnostic")
        )
        object.__setattr__(
            self,
            "mismatched_bindings",
            _unique_sorted_texts(self.mismatched_bindings, "mismatched_binding"),
        )
        object.__setattr__(self, "granted", _bool(self.granted, "granted"))
        object.__setattr__(self, "raw_fallback", _bool(self.raw_fallback, "raw_fallback"))
        object.__setattr__(
            self,
            "similarity_context_only",
            _bool(self.similarity_context_only, "similarity_context_only"),
        )
        object.__setattr__(
            self,
            "negative_memory_applied",
            _bool(self.negative_memory_applied, "negative_memory_applied"),
        )
        object.__setattr__(self, "proof_backed", _bool(self.proof_backed, "proof_backed"))
        object.__setattr__(
            self, "limitations", _unique_sorted_texts(self.limitations, "limitation")
        )
        object.__setattr__(self, "stages", tuple(dict(item) for item in self.stages))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "evaluation_cid": self.evaluation_cid,
            "verdict": self.verdict,
            "reason_code": self.reason_code,
            "diagnostic": self.diagnostic,
            "mismatched_bindings": list(self.mismatched_bindings),
            "granted": self.granted,
            "raw_fallback": self.raw_fallback,
            "similarity_context_only": self.similarity_context_only,
            "negative_memory_applied": self.negative_memory_applied,
            "proof_backed": self.proof_backed,
            "limitations": list(self.limitations),
            "stages": [dict(item) for item in self.stages],
        }
        payload["explanation_cid"] = _payload_cid(payload)
        return payload


# ---------------------------------------------------------------------------
# Binding / proof helpers
# ---------------------------------------------------------------------------


def binding_mismatches(
    query: ProgramWorldReuseKey, candidate_key: ProgramWorldReuseKey
) -> tuple[str, ...]:
    mismatched: list[str] = []
    for name in REUSE_KEY_BINDINGS:
        if query.binding(name) != candidate_key.binding(name):
            mismatched.append(name)
    return tuple(mismatched)


def _primary_mismatch_reason(mismatched: Sequence[str]) -> str:
    if len(mismatched) == 1:
        return BINDING_MISMATCH_REASONS[mismatched[0]]
    if mismatched:
        return ReuseReasonCode.BINDING_MISMATCH.value
    return ReuseReasonCode.NO_EXACT_IDENTITY_MATCH.value


def _relation_covers_mismatch(
    query: ProgramWorldReuseKey,
    candidate: ProgramWorldReuseCandidate,
    mismatched: Sequence[str],
) -> bool:
    if len(mismatched) != 1:
        return False
    name = mismatched[0]
    if name not in SCALAR_BINDINGS:
        return False
    left = candidate.relation_left_cid
    right = candidate.relation_right_cid
    if left is None or right is None:
        return False
    query_value = query.binding(name)
    candidate_value = candidate.key.binding(name)
    pair = {left, right}
    return pair == {query_value, candidate_value}


def _proof_backed_scope_ok(
    query: ProgramWorldReuseKey, candidate: ProgramWorldReuseCandidate
) -> str | None:
    status = candidate.relation_authority_status
    if status not in PROOF_BACKED_AUTHORITY_STATUSES:
        return ReuseReasonCode.RELATION_NOT_PROVED.value
    kind = candidate.relation_kind
    if kind not in PROOF_BACKED_RELATION_KINDS:
        return ReuseReasonCode.RELATION_NOT_PROVED.value
    if candidate.relation_claim_cid is None:
        return ReuseReasonCode.RELATION_NOT_PROVED.value
    if (
        candidate.query_scope_cid is not None
        and candidate.relation_scope_cid is not None
        and candidate.query_scope_cid != candidate.relation_scope_cid
    ):
        return ReuseReasonCode.SCOPE_MISMATCH.value
    if set(candidate.relation_assumption_cids) - set(candidate.query_assumption_cids):
        return ReuseReasonCode.ASSUMPTION_MISMATCH.value
    if (
        candidate.relation_environment_cid is not None
        and candidate.relation_environment_cid != query.environment_cid
    ):
        return ReuseReasonCode.ENVIRONMENT_INCOMPATIBLE.value
    if (
        candidate.relation_policy_cid is not None
        and candidate.relation_policy_cid != query.policy_cid
    ):
        return ReuseReasonCode.POLICY_MISMATCH.value
    return None


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class ProgramWorldReuseGate:
    """Fail-closed exact and independently proof-backed reuse admission."""

    INTERFACE: Final[str] = PROGRAM_WORLD_REUSE_GATE_INTERFACE

    def __init__(self) -> None:
        self._negative_memory: dict[tuple[str, str], NegativeMemoryRecord] = {}

    def _memory_key(
        self, key_cid: str, candidate_evidence_cid: str | None
    ) -> tuple[str, str]:
        return (key_cid, candidate_evidence_cid or key_cid)

    def negative_memory_records(self) -> tuple[NegativeMemoryRecord, ...]:
        return tuple(self._negative_memory.values())

    def record_negative_memory(
        self, record: NegativeMemoryRecord
    ) -> NegativeMemoryRecord:
        self._negative_memory[
            self._memory_key(record.key_cid, record.candidate_evidence_cid)
        ] = record
        return record

    def invalidate_negative_memory(
        self,
        key_cid: str,
        *,
        candidate_evidence_cid: str | None = None,
        invalidator_cids: Sequence[str] = (),
    ) -> tuple[NegativeMemoryRecord, ...]:
        key_cid = validate_opaque_cid(key_cid, "key_cid")
        cleared: list[NegativeMemoryRecord] = []
        remaining: dict[tuple[str, str], NegativeMemoryRecord] = {}
        invalidators = set(_unique_sorted_cids(invalidator_cids, "invalidator_cid"))
        for index, record in self._negative_memory.items():
            if record.key_cid != key_cid:
                remaining[index] = record
                continue
            if (
                candidate_evidence_cid is not None
                and record.candidate_evidence_cid != candidate_evidence_cid
            ):
                remaining[index] = record
                continue
            if invalidators and not (set(record.invalidator_cids) & invalidators):
                remaining[index] = record
                continue
            cleared.append(record)
        self._negative_memory = remaining
        return tuple(cleared)

    def lookup_negative_memory(
        self, query: ProgramWorldReuseKey, candidate: ProgramWorldReuseCandidate
    ) -> NegativeMemoryRecord | None:
        record = self._negative_memory.get(
            self._memory_key(query.reuse_key_cid, candidate.candidate_evidence_cid)
        )
        if record is None:
            return None
        if record.still_in_force(query, candidate):
            return record
        self.invalidate_negative_memory(
            query.reuse_key_cid,
            candidate_evidence_cid=candidate.candidate_evidence_cid,
        )
        return None

    def evaluate(
        self,
        query: ProgramWorldReuseKey | Mapping[str, Any],
        candidate: ProgramWorldReuseCandidate | Mapping[str, Any] | None = None,
        *,
        admission_authority: str | None = None,
        admission_evidence_cid: str | None = None,
    ) -> ProgramWorldReuseEvaluation:
        query_key = _coerce_key(query)
        offered = _coerce_candidate(candidate, default_key=query_key)
        admission_authority = _optional_bounded_text(
            admission_authority, "admission_authority"
        )
        admission_evidence_cid = _optional_cid(
            admission_evidence_cid, "admission_evidence_cid"
        )
        return self._evaluate(
            query_key,
            offered,
            admission_authority=admission_authority,
            admission_evidence_cid=admission_evidence_cid,
        )

    def explain(
        self,
        evaluation: ProgramWorldReuseEvaluation
        | ProgramWorldReuseKey
        | Mapping[str, Any],
        candidate: ProgramWorldReuseCandidate | Mapping[str, Any] | None = None,
        *,
        admission_authority: str | None = None,
        admission_evidence_cid: str | None = None,
    ) -> ProgramWorldReuseExplanation:
        if not isinstance(evaluation, ProgramWorldReuseEvaluation):
            evaluation = self.evaluate(
                evaluation,
                candidate,
                admission_authority=admission_authority,
                admission_evidence_cid=admission_evidence_cid,
            )
        limitations = list(evaluation.decision.limitations)
        if evaluation.rejection is not None:
            limitations.extend(evaluation.rejection.limitations)
        return ProgramWorldReuseExplanation(
            evaluation_cid=evaluation.evaluation_cid,
            verdict=evaluation.verdict,
            reason_code=evaluation.reason_code,
            diagnostic=evaluation.diagnostic,
            mismatched_bindings=evaluation.mismatched_bindings,
            granted=evaluation.granted,
            raw_fallback=evaluation.raw_fallback,
            similarity_context_only=evaluation.similarity_context_only,
            negative_memory_applied=evaluation.negative_memory is not None,
            proof_backed=evaluation.proof_backed,
            limitations=tuple(dict.fromkeys(limitations)),
            stages=tuple(item.to_dict() for item in evaluation.stages),
        )

    def _evaluate(
        self,
        query: ProgramWorldReuseKey,
        candidate: ProgramWorldReuseCandidate | None,
        *,
        admission_authority: str | None,
        admission_evidence_cid: str | None,
    ) -> ProgramWorldReuseEvaluation:
        stages: list[ReuseStageReceipt] = []
        if _is_gate_owned_authority(admission_authority):
            rejection = _rejection(
                ReuseReasonCode.SELF_ADMISSION_FORBIDDEN.value,
                verdict=ReuseVerdict.REJECT,
                limitations=("gate_cannot_admit_own_output",),
            )
            return self._finish(
                query,
                candidate,
                rejection,
                stages,
                admission_authority=None,
                admission_evidence_cid=None,
            )

        if candidate is None:
            stages.append(
                _stage(
                    ReuseStage.BINDING_IDENTITY,
                    ReuseStageDisposition.REJECTED,
                    ReuseReasonCode.NO_EXACT_IDENTITY_MATCH.value,
                )
            )
            _skip_remaining(
                stages, after=ReuseStage.BINDING_IDENTITY, raw_fallback=True
            )
            rejection = _rejection(
                ReuseReasonCode.NO_EXACT_IDENTITY_MATCH.value,
                verdict=ReuseVerdict.ABSTAIN,
                limitations=("exact_reuse_required", "raw_source_fallback_available"),
                raw_fallback=True,
            )
            return self._finish(
                query,
                None,
                rejection,
                stages,
                admission_authority=admission_authority,
                admission_evidence_cid=admission_evidence_cid,
            )

        fail = self._first_failure(query, candidate, stages)
        if fail is not None:
            reason, verdict, mismatches, limitations, raw_fallback, memory = fail
            if reason == ReuseReasonCode.NEGATIVE_MEMORY.value and memory is not None:
                rejection = ProgramWorldReuseRejection(
                    reason_code=memory.reason_code
                    if memory.reason_code
                    else ReuseReasonCode.NEGATIVE_MEMORY.value,
                    diagnostic=diagnostic_for(
                        ReuseReasonCode.NEGATIVE_MEMORY.value,
                        extra=memory.diagnostic,
                    ),
                    mismatched_bindings=memory.mismatched_bindings,
                    verdict=ReuseVerdict.REJECT,
                    limitations=("negative_memory_in_force",) + tuple(limitations),
                    raw_fallback=raw_fallback,
                    invalidator_cids=memory.invalidator_cids,
                )
                return self._finish(
                    query,
                    candidate,
                    rejection,
                    stages,
                    admission_authority=admission_authority,
                    admission_evidence_cid=admission_evidence_cid,
                    negative_memory=memory,
                    mismatched_bindings=tuple(memory.mismatched_bindings),
                )
            rejection = _rejection(
                reason,
                verdict=verdict,
                mismatched_bindings=mismatches,
                limitations=limitations,
                raw_fallback=raw_fallback,
                invalidator_cids=candidate.invalidator_cids,
            )
            memory_record = None
            if verdict == ReuseVerdict.REJECT.value:
                memory_record = self.record_negative_memory(
                    NegativeMemoryRecord(
                        key_cid=query.reuse_key_cid,
                        reason_code=reason,
                        diagnostic=rejection.diagnostic,
                        mismatched_bindings=mismatches,
                        invalidator_cids=candidate.invalidator_cids,
                        candidate_evidence_cid=candidate.candidate_evidence_cid,
                        rejection_cid=rejection.rejection_cid,
                    )
                )
            return self._finish(
                query,
                candidate,
                rejection,
                stages,
                admission_authority=admission_authority,
                admission_evidence_cid=admission_evidence_cid,
                negative_memory=memory_record,
                mismatched_bindings=mismatches,
            )

        mismatched = binding_mismatches(query, candidate.key)
        proof_backed = bool(mismatched)
        self.invalidate_negative_memory(query.reuse_key_cid)
        if proof_backed:
            reason = ReuseReasonCode.PROOF_BACKED_SCOPED_MATCH.value
            limitations = (
                "relation_reuse_scoped_to_assumptions",
                "cache_hit_does_not_suppress_validation",
                "reuse_is_proposal_until_supervisor_admission",
            )
        else:
            reason = ReuseReasonCode.EXACT_IDENTITY_MATCH.value
            limitations = (
                "cache_hit_does_not_suppress_validation",
                "reuse_is_proposal_until_supervisor_admission",
            )
        if candidate.ann_candidates:
            limitations = limitations + ("similarity_is_context_only",)
        admitted = _independent_admission(admission_authority, admission_evidence_cid)
        decision = _decision(
            query,
            candidate,
            verdict=ReuseVerdict.REUSE,
            reason_code=reason,
            exact_match=True,
            proposal_only=not admitted,
            admitted=admitted,
            admission_authority=admission_authority if admitted else None,
            admission_evidence_cid=admission_evidence_cid if admitted else None,
            limitations=limitations,
            fallback=False,
        )
        stages = _complete_stages(stages)
        return ProgramWorldReuseEvaluation(
            query=query,
            verdict=ReuseVerdict.REUSE,
            reason_code=reason,
            diagnostic=diagnostic_for(reason),
            decision=decision,
            candidate=candidate,
            rejection=None,
            negative_memory=None,
            mismatched_bindings=(),
            stages=tuple(stages),
            raw_fallback=False,
            granted=True,
            proof_backed=proof_backed,
            similarity_context_only=bool(candidate.ann_candidates),
        )

    def _first_failure(
        self,
        query: ProgramWorldReuseKey,
        candidate: ProgramWorldReuseCandidate,
        stages: list[ReuseStageReceipt],
    ) -> (
        tuple[
            str,
            str,
            tuple[str, ...],
            tuple[str, ...],
            bool,
            NegativeMemoryRecord | None,
        ]
        | None
    ):
        raw_fallback = candidate.raw_source_cid is not None

        if not candidate.capability_available or candidate.evidence_kind == (
            ReuseEvidenceKind.UNAVAILABLE.value
        ):
            reason = candidate.capability_reason_code or (
                ReuseReasonCode.TYPED_UNAVAILABLE.value
            )
            stages.append(
                _stage(
                    ReuseStage.CAPABILITY,
                    ReuseStageDisposition.UNAVAILABLE,
                    reason,
                    extra=candidate.capability_surface,
                )
            )
            _skip_remaining(stages, after=ReuseStage.CAPABILITY, raw_fallback=raw_fallback)
            return (
                ReuseReasonCode.TYPED_UNAVAILABLE.value,
                ReuseVerdict.UNAVAILABLE.value,
                (),
                (
                    "typed_unavailable",
                    "raw_source_fallback_available" if raw_fallback else "raw_source_fallback_required",
                ),
                True,
                None,
            )
        stages.append(
            _stage(
                ReuseStage.CAPABILITY,
                ReuseStageDisposition.SELECTED,
                "available",
            )
        )

        if candidate.incomplete:
            stages.append(
                _stage(
                    ReuseStage.COMPLETENESS,
                    ReuseStageDisposition.REJECTED,
                    ReuseReasonCode.INCOMPLETE_BINDINGS.value,
                )
            )
            _skip_remaining(stages, after=ReuseStage.COMPLETENESS, raw_fallback=raw_fallback)
            return (
                ReuseReasonCode.INCOMPLETE_BINDINGS.value,
                ReuseVerdict.REJECT.value,
                (),
                ("incomplete_fails_closed", "unknown_widens"),
                raw_fallback,
                None,
            )
        stages.append(
            _stage(
                ReuseStage.COMPLETENESS,
                ReuseStageDisposition.SELECTED,
                "complete",
            )
        )

        if (
            candidate.similarity_only
            or candidate.evidence_kind == ReuseEvidenceKind.SIMILARITY.value
            or (
                candidate.ann_candidates
                and candidate.evidence_kind != ReuseEvidenceKind.EXACT.value
                and candidate.evidence_kind != ReuseEvidenceKind.PROOF_BACKED.value
                and candidate.evidence_kind != ReuseEvidenceKind.CACHE.value
                and candidate.evidence_kind != ReuseEvidenceKind.PROCEDURE.value
                and candidate.evidence_kind != ReuseEvidenceKind.ABSTRACT.value
            )
        ):
            reason = (
                ReuseReasonCode.ANN_NOT_AUTHORITATIVE.value
                if candidate.ann_candidates
                else ReuseReasonCode.SIMILARITY_IS_NOT_REUSE.value
            )
            stages.append(
                _stage(
                    ReuseStage.SIMILARITY,
                    ReuseStageDisposition.REJECTED,
                    reason,
                )
            )
            _skip_remaining(stages, after=ReuseStage.SIMILARITY, raw_fallback=raw_fallback)
            return (
                reason,
                ReuseVerdict.REJECT.value,
                (),
                ("similarity_is_context_only", "ann_advisory_only"),
                raw_fallback,
                None,
            )
        stages.append(
            _stage(
                ReuseStage.SIMILARITY,
                ReuseStageDisposition.SELECTED
                if not candidate.ann_candidates
                else ReuseStageDisposition.SKIPPED,
                "similarity_is_context_only"
                if candidate.ann_candidates
                else "no_similarity_authority",
            )
        )

        memory = self.lookup_negative_memory(query, candidate)
        if memory is not None:
            stages.append(
                _stage(
                    ReuseStage.NEGATIVE_MEMORY,
                    ReuseStageDisposition.REJECTED,
                    ReuseReasonCode.NEGATIVE_MEMORY.value,
                )
            )
            _skip_remaining(
                stages, after=ReuseStage.NEGATIVE_MEMORY, raw_fallback=raw_fallback
            )
            return (
                ReuseReasonCode.NEGATIVE_MEMORY.value,
                ReuseVerdict.REJECT.value,
                tuple(memory.mismatched_bindings),
                ("negative_memory_in_force",),
                raw_fallback,
                memory,
            )
        stages.append(
            _stage(
                ReuseStage.NEGATIVE_MEMORY,
                ReuseStageDisposition.SELECTED,
                "no_negative_memory",
            )
        )

        mismatched = binding_mismatches(query, candidate.key)
        proof_backed = False
        if not mismatched:
            stages.append(
                _stage(
                    ReuseStage.BINDING_IDENTITY,
                    ReuseStageDisposition.SELECTED,
                    ReuseReasonCode.EXACT_IDENTITY_MATCH.value,
                )
            )
        else:
            scope_reason = None
            if candidate.evidence_kind in {
                ReuseEvidenceKind.PROOF_BACKED.value,
                ReuseEvidenceKind.CACHE.value,
            } or candidate.relation_claim_cid is not None:
                scope_reason = _proof_backed_scope_ok(query, candidate)
                if scope_reason is None and _relation_covers_mismatch(
                    query, candidate, mismatched
                ):
                    proof_backed = True
            if not proof_backed:
                reason = scope_reason or _primary_mismatch_reason(mismatched)
                stages.append(
                    _stage(
                        ReuseStage.BINDING_IDENTITY,
                        ReuseStageDisposition.REJECTED,
                        reason,
                    )
                )
                _skip_remaining(
                    stages, after=ReuseStage.BINDING_IDENTITY, raw_fallback=raw_fallback
                )
                limitations = ("exact_reuse_required",)
                if scope_reason is not None:
                    limitations = (
                        "relation_reuse_scoped_to_assumptions",
                        scope_reason,
                    )
                return (
                    reason,
                    ReuseVerdict.REJECT.value,
                    mismatched,
                    limitations,
                    raw_fallback,
                    None,
                )
            stages.append(
                _stage(
                    ReuseStage.BINDING_IDENTITY,
                    ReuseStageDisposition.SELECTED,
                    ReuseReasonCode.PROOF_BACKED_SCOPED_MATCH.value,
                )
            )

        if candidate.freshness == FreshnessState.STALE.value:
            stages.append(
                _stage(
                    ReuseStage.FRESHNESS,
                    ReuseStageDisposition.REJECTED,
                    ReuseReasonCode.STALE_BINDING.value,
                )
            )
            _skip_remaining(stages, after=ReuseStage.FRESHNESS, raw_fallback=raw_fallback)
            return (
                ReuseReasonCode.STALE_BINDING.value,
                ReuseVerdict.REJECT.value,
                mismatched if not proof_backed else (),
                ("stale_bindings_revoke",),
                raw_fallback,
                None,
            )
        if candidate.freshness in {
            FreshnessState.UNKNOWN.value,
            FreshnessState.UNAVAILABLE.value,
        }:
            reason = (
                ReuseReasonCode.FRESHNESS_UNKNOWN.value
                if candidate.freshness == FreshnessState.UNKNOWN.value
                else ReuseReasonCode.TYPED_UNAVAILABLE.value
            )
            stages.append(
                _stage(
                    ReuseStage.FRESHNESS,
                    ReuseStageDisposition.REJECTED
                    if candidate.freshness == FreshnessState.UNKNOWN.value
                    else ReuseStageDisposition.UNAVAILABLE,
                    reason,
                )
            )
            _skip_remaining(stages, after=ReuseStage.FRESHNESS, raw_fallback=raw_fallback)
            verdict = (
                ReuseVerdict.REJECT.value
                if candidate.freshness == FreshnessState.UNKNOWN.value
                else ReuseVerdict.UNAVAILABLE.value
            )
            return (
                reason,
                verdict,
                (),
                ("unknown_widens", "stale_bindings_revoke"),
                True,
                None,
            )
        stages.append(
            _stage(
                ReuseStage.FRESHNESS,
                ReuseStageDisposition.SELECTED,
                "fresh",
            )
        )

        env_mismatch = (
            not candidate.environment_compatible
            or query.environment_cid != candidate.key.environment_cid
        )
        if env_mismatch and not (
            proof_backed
            and mismatched == ("environment_cid",)
            and _relation_covers_mismatch(query, candidate, mismatched)
        ):
            if not candidate.environment_compatible or (
                query.environment_cid != candidate.key.environment_cid and not proof_backed
            ):
                stages.append(
                    _stage(
                        ReuseStage.ENVIRONMENT,
                        ReuseStageDisposition.REJECTED,
                        ReuseReasonCode.ENVIRONMENT_INCOMPATIBLE.value,
                    )
                )
                _skip_remaining(
                    stages, after=ReuseStage.ENVIRONMENT, raw_fallback=raw_fallback
                )
                return (
                    ReuseReasonCode.ENVIRONMENT_INCOMPATIBLE.value,
                    ReuseVerdict.REJECT.value,
                    ("environment_cid",)
                    if query.environment_cid != candidate.key.environment_cid
                    else (),
                    ("environment_incompatible_fails_closed",),
                    raw_fallback,
                    None,
                )
        stages.append(
            _stage(
                ReuseStage.ENVIRONMENT,
                ReuseStageDisposition.SELECTED,
                "environment_compatible",
            )
        )

        unsafe = False
        if candidate.unknown_relevance_dimensions:
            unsafe = True
        if candidate.abstraction_soundness_claim in UNSAFE_ABSTRACTION_CLAIMS:
            unsafe = True
        if candidate.abstraction_reuse_admitted is False:
            unsafe = True
        if candidate.evidence_kind == ReuseEvidenceKind.ABSTRACT.value and (
            candidate.abstraction_reuse_admitted is not True
        ):
            unsafe = True
        if unsafe:
            stages.append(
                _stage(
                    ReuseStage.ABSTRACTION,
                    ReuseStageDisposition.REJECTED,
                    ReuseReasonCode.UNSAFE_ABSTRACTION.value,
                )
            )
            _skip_remaining(stages, after=ReuseStage.ABSTRACTION, raw_fallback=raw_fallback)
            return (
                ReuseReasonCode.UNSAFE_ABSTRACTION.value,
                ReuseVerdict.REJECT.value,
                (),
                ("unsafe_abstraction_fails_closed", "unknown_widens"),
                raw_fallback,
                None,
            )
        stages.append(
            _stage(
                ReuseStage.ABSTRACTION,
                ReuseStageDisposition.SKIPPED
                if candidate.abstraction_soundness_claim is None
                else ReuseStageDisposition.SELECTED,
                "abstraction_safe"
                if candidate.abstraction_reuse_admitted is True
                else "no_unsafe_abstraction",
            )
        )

        needs_proof_admission = bool(
            candidate.proof_cache_hit
            or candidate.evidence_kind
            in {ReuseEvidenceKind.CACHE.value, ReuseEvidenceKind.PROOF_BACKED.value}
            or proof_backed
        )
        if needs_proof_admission and not _independent_admission(
            candidate.proof_admission_authority,
            candidate.proof_admission_evidence_cid,
        ):
            stages.append(
                _stage(
                    ReuseStage.PROOF_ADMISSION,
                    ReuseStageDisposition.REJECTED,
                    ReuseReasonCode.PROOF_CACHE_REQUIRES_FRESH_ADMISSION.value,
                )
            )
            _skip_remaining(
                stages, after=ReuseStage.PROOF_ADMISSION, raw_fallback=raw_fallback
            )
            return (
                ReuseReasonCode.PROOF_CACHE_REQUIRES_FRESH_ADMISSION.value,
                ReuseVerdict.REJECT.value,
                (),
                (
                    "proof_cache_requires_fresh_admission",
                    "cache_hit_does_not_suppress_validation",
                ),
                raw_fallback,
                None,
            )
        stages.append(
            _stage(
                ReuseStage.PROOF_ADMISSION,
                ReuseStageDisposition.SELECTED
                if needs_proof_admission
                else ReuseStageDisposition.SKIPPED,
                "fresh_proof_admission"
                if needs_proof_admission
                else "proof_admission_not_required",
            )
        )

        if proof_backed:
            stages.append(
                _stage(
                    ReuseStage.SCOPE,
                    ReuseStageDisposition.SELECTED,
                    ReuseReasonCode.PROOF_BACKED_SCOPED_MATCH.value,
                )
            )
        else:
            stages.append(
                _stage(
                    ReuseStage.SCOPE,
                    ReuseStageDisposition.SKIPPED,
                    "scope_not_required_for_exact_hit",
                )
            )

        if candidate.suppress_validation:
            stages.append(
                _stage(
                    ReuseStage.VALIDATION,
                    ReuseStageDisposition.REJECTED,
                    ReuseReasonCode.CACHE_HIT_DOES_NOT_SUPPRESS_VALIDATION.value,
                )
            )
            _skip_remaining(stages, after=ReuseStage.VALIDATION, raw_fallback=raw_fallback)
            return (
                ReuseReasonCode.CACHE_HIT_DOES_NOT_SUPPRESS_VALIDATION.value,
                ReuseVerdict.REJECT.value,
                (),
                ("mandatory_validation_retained",),
                raw_fallback,
                None,
            )
        stages.append(
            _stage(
                ReuseStage.VALIDATION,
                ReuseStageDisposition.SELECTED,
                "mandatory_validation_retained",
            )
        )
        stages.append(
            _stage(
                ReuseStage.RAW_FALLBACK,
                ReuseStageDisposition.SKIPPED,
                "raw_fallback_not_required",
            )
        )
        return None

    def _finish(
        self,
        query: ProgramWorldReuseKey,
        candidate: ProgramWorldReuseCandidate | None,
        rejection: ProgramWorldReuseRejection,
        stages: Sequence[ReuseStageReceipt],
        *,
        admission_authority: str | None,
        admission_evidence_cid: str | None,
        negative_memory: NegativeMemoryRecord | None = None,
        mismatched_bindings: Sequence[str] = (),
    ) -> ProgramWorldReuseEvaluation:
        del admission_authority, admission_evidence_cid
        raw_fallback = rejection.raw_fallback or (
            candidate is not None and candidate.raw_source_cid is not None
        )
        limitations = tuple(rejection.limitations)
        if raw_fallback and "raw_source_fallback_available" not in limitations:
            limitations = limitations + ("raw_source_fallback_available",)
        decision = _decision(
            query,
            candidate,
            verdict=ReuseVerdict(rejection.verdict),
            reason_code=rejection.reason_code
            if rejection.reason_code != ReuseReasonCode.NEGATIVE_MEMORY.value
            else (
                negative_memory.reason_code
                if negative_memory is not None
                else rejection.reason_code
            ),
            exact_match=False,
            proposal_only=True,
            admitted=False,
            admission_authority=None,
            admission_evidence_cid=None,
            limitations=limitations,
            fallback=rejection.verdict == ReuseVerdict.UNAVAILABLE.value or raw_fallback,
        )
        reason = decision.reason_code
        return ProgramWorldReuseEvaluation(
            query=query,
            verdict=rejection.verdict,
            reason_code=reason,
            diagnostic=rejection.diagnostic,
            decision=decision,
            candidate=candidate,
            rejection=rejection,
            negative_memory=negative_memory,
            mismatched_bindings=mismatched_bindings or rejection.mismatched_bindings,
            stages=tuple(_complete_stages(stages, raw_fallback=raw_fallback)),
            raw_fallback=raw_fallback,
            granted=False,
            proof_backed=False,
            similarity_context_only=reason
            in {
                ReuseReasonCode.SIMILARITY_IS_NOT_REUSE.value,
                ReuseReasonCode.ANN_NOT_AUTHORITATIVE.value,
            }
            or reason in ANN_REASON_CODES,
        )


def _stage(
    stage: ReuseStage,
    disposition: ReuseStageDisposition,
    reason_code: str,
    extra: str | None = None,
) -> ReuseStageReceipt:
    return ReuseStageReceipt(
        stage=stage,
        disposition=disposition,
        reason_code=reason_code,
        diagnostic=diagnostic_for(reason_code, extra=extra),
    )


def _skip_remaining(
    stages: list[ReuseStageReceipt],
    *,
    after: ReuseStage,
    raw_fallback: bool = False,
) -> None:
    seen = {item.stage for item in stages}
    passed = False
    for stage in STAGE_ORDER:
        if stage == after:
            passed = True
            continue
        if not passed or stage.value in seen:
            continue
        if stage == ReuseStage.RAW_FALLBACK:
            stages.append(
                _stage(
                    stage,
                    ReuseStageDisposition.SELECTED
                    if raw_fallback
                    else ReuseStageDisposition.SKIPPED,
                    ReuseReasonCode.RAW_SOURCE_FALLBACK.value
                    if raw_fallback
                    else "raw_fallback_not_selected",
                )
            )
            continue
        stages.append(
            _stage(
                stage,
                ReuseStageDisposition.SKIPPED,
                "skipped_after_fail_closed",
            )
        )


def _complete_stages(
    stages: Sequence[ReuseStageReceipt], *, raw_fallback: bool = False
) -> list[ReuseStageReceipt]:
    by_stage = {item.stage: item for item in stages}
    completed: list[ReuseStageReceipt] = []
    for stage in STAGE_ORDER:
        existing = by_stage.get(stage.value)
        if existing is not None:
            completed.append(existing)
            continue
        if stage == ReuseStage.RAW_FALLBACK and raw_fallback:
            completed.append(
                _stage(
                    stage,
                    ReuseStageDisposition.SELECTED,
                    ReuseReasonCode.RAW_SOURCE_FALLBACK.value,
                )
            )
            continue
        completed.append(
            _stage(
                stage,
                ReuseStageDisposition.SKIPPED,
                "skipped_after_fail_closed",
            )
        )
    return completed


def _rejection(
    reason_code: str,
    *,
    verdict: ReuseVerdict | str,
    mismatched_bindings: Sequence[str] = (),
    limitations: Sequence[str] = (),
    raw_fallback: bool = False,
    invalidator_cids: Sequence[str] = (),
    extra: str | None = None,
) -> ProgramWorldReuseRejection:
    return ProgramWorldReuseRejection(
        reason_code=reason_code,
        diagnostic=diagnostic_for(reason_code, extra=extra),
        mismatched_bindings=mismatched_bindings,
        verdict=verdict,
        limitations=limitations,
        raw_fallback=raw_fallback,
        invalidator_cids=invalidator_cids,
    )


def _decision(
    query: ProgramWorldReuseKey,
    candidate: ProgramWorldReuseCandidate | None,
    *,
    verdict: ReuseVerdict,
    reason_code: str,
    exact_match: bool,
    proposal_only: bool,
    admitted: bool,
    admission_authority: str | None,
    admission_evidence_cid: str | None,
    limitations: Sequence[str],
    fallback: bool,
) -> ProgramWorldReuseDecision:
    relation_cid = None if candidate is None else candidate.relation_claim_cid
    procedure_cid = query.procedure_revision_cid
    return ProgramWorldReuseDecision(
        verdict=verdict,
        state_cid=query.state_cid,
        goal_cid=query.goal_cid,
        policy_cid=query.policy_cid,
        environment_cid=query.environment_cid,
        toolchain_cid=query.toolchain_cid,
        reason_code=reason_code,
        relation_claim_cid=relation_cid,
        procedure_revision_cid=procedure_cid,
        exact_match=exact_match,
        ann_authoritative=False,
        proposal_only=proposal_only,
        admitted=admitted,
        admission_authority=admission_authority,
        admission_evidence_cid=admission_evidence_cid,
        operational_acceptance_authorities=OPERATIONAL_ACCEPTANCE_AUTHORITIES,
        fallback=fallback,
        limitations=limitations,
    )


def evaluate_program_world_reuse(
    query: ProgramWorldReuseKey | Mapping[str, Any],
    candidate: ProgramWorldReuseCandidate | Mapping[str, Any] | None = None,
    *,
    gate: ProgramWorldReuseGate | None = None,
    admission_authority: str | None = None,
    admission_evidence_cid: str | None = None,
) -> ProgramWorldReuseEvaluation:
    """Evaluate exact or independently proof-backed reuse against the current key."""

    owner = gate if gate is not None else ProgramWorldReuseGate()
    return owner.evaluate(
        query,
        candidate,
        admission_authority=admission_authority,
        admission_evidence_cid=admission_evidence_cid,
    )


def explain_program_world_reuse(
    evaluation: ProgramWorldReuseEvaluation
    | ProgramWorldReuseKey
    | Mapping[str, Any],
    candidate: ProgramWorldReuseCandidate | Mapping[str, Any] | None = None,
    *,
    gate: ProgramWorldReuseGate | None = None,
    admission_authority: str | None = None,
    admission_evidence_cid: str | None = None,
) -> ProgramWorldReuseExplanation:
    """Explain a reuse grant or fail-closed rejection without re-validating proofs."""

    if isinstance(evaluation, ProgramWorldReuseEvaluation):
        owner = gate if gate is not None else ProgramWorldReuseGate()
        return owner.explain(evaluation)
    owner = gate if gate is not None else ProgramWorldReuseGate()
    return owner.explain(
        evaluation,
        candidate,
        admission_authority=admission_authority,
        admission_evidence_cid=admission_evidence_cid,
    )


def program_world_reuse_module_source() -> str:
    """Return this module's source for AST audits. Performs no I/O besides read."""

    from pathlib import Path

    return Path(__file__).read_text(encoding="utf-8")


def assert_no_duplicate_reuse_authority(source: str | None = None) -> None:
    """Fail if this module redefines datasets/kit/proof/procedure/root types."""

    import ast

    tree = ast.parse(source or program_world_reuse_module_source())
    banned = {
        "SemanticObjectEnvelope",
        "ProgramRelationClaim",
        "ProgramTransitionQuery",
        "ContextCompiler",
        "DurableCoordinationStore",
        "VerifiedSemanticBlockStore",
        "SemanticWorldArtifactStore",
        "ProofCarryingProcedureCompiler",
        "VerificationReceiptCache",
        "StateAbstractionProfile",
    }
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }
    overlap = defined & banned
    if overlap:
        raise ProgramWorldReuseError(
            "reuse gate must not duplicate proof/relation/procedure/root types "
            + str(sorted(overlap))
        )


__all__ = [
    "BINDING_MISMATCH_REASONS",
    "EVENT_AUTHORITY",
    "GATE_ID",
    "GATE_OWNED_AUTHORITIES",
    "MERGE_AUTHORITY",
    "NEGATIVE_MEMORY_RECORD_SCHEMA",
    "OPERATIONAL_ACCEPTANCE_AUTHORITIES",
    "PROGRAM_WORLD_REUSE_DECISION_INTERFACE_REF",
    "PROGRAM_WORLD_REUSE_GATE_INTERFACE",
    "PROGRAM_WORLD_REUSE_KEY_INTERFACE",
    "REUSE_KEY_BINDINGS",
    "SAWM_EXACT_REUSE_EVIDENCE",
    "SAWM_REUSE_ADMISSION_EVIDENCE",
    "SCALAR_BINDINGS",
    "SET_BINDINGS",
    "STAGE_ORDER",
    "VALIDATION_AUTHORITY",
    "NegativeMemoryRecord",
    "ProgramWorldReuseCandidate",
    "ProgramWorldReuseError",
    "ProgramWorldReuseEvaluation",
    "ProgramWorldReuseExplanation",
    "ProgramWorldReuseGate",
    "ProgramWorldReuseKey",
    "ProgramWorldReuseRejection",
    "ReuseEvidenceKind",
    "ReuseReasonCode",
    "ReuseStage",
    "ReuseStageDisposition",
    "ReuseStageReceipt",
    "assert_no_duplicate_reuse_authority",
    "binding_mismatches",
    "evaluate_program_world_reuse",
    "explain_program_world_reuse",
    "program_world_reuse_module_source",
]
