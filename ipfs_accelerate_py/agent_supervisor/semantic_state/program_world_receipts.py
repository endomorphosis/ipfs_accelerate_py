"""Operational receipts for accelerator semantic-world adapters.

Interfaces: ``SemanticWorldOperationalAdapters@1``,
``ProgramWorldReuseDecision@1``, ``ProgramWorldContextReceipt@1``.

These records are accelerate-owned operational evidence.  They cite datasets
identities and kit verified results; they never recompute semantic CIDs, never
treat ANN/similarity as authority, and never admit their own output.
Operational acceptance remains the existing supervisor validation, merge, and
event authorities.

SCH / SAWM-015 / sawm/context-receipt@1
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final, Iterable, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessError,
    UnavailableResult,
    _bool,
    _closed,
    _text,
    validate_opaque_cid,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload


# ---------------------------------------------------------------------------
# Interfaces / schemas / closed vocabularies
# ---------------------------------------------------------------------------

SEMANTIC_WORLD_OPERATIONAL_ADAPTERS_INTERFACE: Final[str] = (
    "SemanticWorldOperationalAdapters@1"
)
PROGRAM_WORLD_REUSE_DECISION_INTERFACE: Final[str] = "ProgramWorldReuseDecision@1"
PROGRAM_WORLD_CONTEXT_RECEIPT_INTERFACE: Final[str] = "ProgramWorldContextReceipt@1"
PROGRAM_WORLD_CAPABILITY_RECEIPT_INTERFACE: Final[str] = (
    "ProgramWorldCapabilityReceipt@1"
)
EXECUTION_TRANSITION_COMPILATION_INTERFACE: Final[str] = (
    "ExecutionTransitionCompilation@1"
)
OPERATIONAL_WORLD_ROOT_PUBLICATION_REQUEST_INTERFACE: Final[str] = (
    "OperationalWorldRootPublicationRequest@1"
)

PROGRAM_WORLD_REUSE_DECISION_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-reuse-decision@1"
)
PROGRAM_WORLD_CONTEXT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-context-receipt@1"
)
PROGRAM_WORLD_CAPABILITY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-capability-receipt@1"
)
EXECUTION_TRANSITION_COMPILATION_SCHEMA: Final[str] = (
    "ipfs-accelerate.execution-transition-compilation@1"
)
OPERATIONAL_WORLD_ROOT_PUBLICATION_REQUEST_SCHEMA: Final[str] = (
    "ipfs-accelerate.operational-world-root-publication-request@1"
)
PROGRAM_WORLD_CONTEXT_ITEM_SCHEMA: Final[str] = (
    "ipfs-accelerate.program-world-context-item@1"
)

ADAPTER_ID: Final[str] = "ipfs-accelerate.semantic-state.program-world-adapters"

DATASETS_IDENTITY_AUTHORITY: Final[str] = (
    "ipfs_datasets_py.logic.software_contracts.semantic_state.program_identity"
)
DATASETS_RELATION_AUTHORITY: Final[str] = (
    "ipfs_datasets_py.logic.software_contracts.semantic_state.program_relations"
)
DATASETS_TRANSITION_AUTHORITY: Final[str] = (
    "ipfs_datasets_py.logic.software_contracts.semantic_state.program_transition"
)
DATASETS_DOMAIN_ADAPTER_AUTHORITY: Final[str] = (
    "ipfs_datasets_py.logic.software_contracts.semantic_state.program_domain_adapters"
)
KIT_VERIFIED_STORE_AUTHORITY: Final[str] = (
    "ipfs_kit_py.semantic_world_store.verified_store"
)
VALIDATION_AUTHORITY: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.validation.validation_runtime"
)
MERGE_AUTHORITY: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.merge.merge_resolver"
)
EVENT_AUTHORITY: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state"
)

OPERATIONAL_ACCEPTANCE_AUTHORITIES: Final[tuple[str, ...]] = (
    VALIDATION_AUTHORITY,
    MERGE_AUTHORITY,
    EVENT_AUTHORITY,
)

ADAPTER_OWNED_AUTHORITIES: Final[frozenset[str]] = frozenset(
    {
        ADAPTER_ID,
        SEMANTIC_WORLD_OPERATIONAL_ADAPTERS_INTERFACE,
        "DatasetsProgramWorldAdapter",
        "KitProgramWorldAdapter",
        "ExecutionTransitionCompiler",
        "OperationalWorldRootPublisher",
        "SemanticWorldOperationalAdapters",
    }
)

MAX_TEXT_CHARS: Final[int] = 512
MAX_COLLECTION_ITEMS: Final[int] = 10_000

FORBIDDEN_IDENTITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "ann_score",
        "cosine",
        "distance",
        "embedding",
        "embedding_score",
        "embeddings",
        "knn",
        "nearest",
        "rank",
        "score",
        "scores",
        "similarity",
        "vector",
        "vectors",
    }
)

ANN_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "ann_advisory_only",
        "ann_not_authoritative",
        "embedding_cannot_omit_required",
        "similarity_is_not_reuse",
        "projection_is_advisory",
    }
)


class ProgramWorldReceiptError(HarnessError):
    """Closed operational-receipt contract violation."""


class ProgramWorldAdmissionError(ProgramWorldReceiptError):
    """Raised when a proposal is offered as self-admitted operational acceptance."""


class ReuseVerdict(str, Enum):
    REUSE = "reuse"
    REJECT = "reject"
    ABSTAIN = "abstain"
    UNAVAILABLE = "unavailable"


class ContextItemDisposition(str, Enum):
    INCLUDED = "included"
    OMITTED = "omitted"
    RAW_FALLBACK = "raw_fallback"
    UNAVAILABLE = "unavailable"


class CapabilityStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    FALLBACK = "fallback"


class CompilationStatus(str, Enum):
    COMPILED = "compiled"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


class PublicationStatus(str, Enum):
    REQUESTED = "requested"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


class FreshnessState(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


REUSE_VERDICTS: Final[frozenset[str]] = frozenset(item.value for item in ReuseVerdict)
CONTEXT_ITEM_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    item.value for item in ContextItemDisposition
)
CAPABILITY_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in CapabilityStatus
)
COMPILATION_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in CompilationStatus
)
PUBLICATION_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in PublicationStatus
)
FRESHNESS_STATES: Final[frozenset[str]] = frozenset(item.value for item in FreshnessState)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bounded_text(value: Any, name: str) -> str:
    text = _text(value, name)
    if len(text) > MAX_TEXT_CHARS:
        raise ProgramWorldReceiptError(f"{name} exceeds {MAX_TEXT_CHARS} characters")
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
        raise ProgramWorldReceiptError(f"{name} must be one of {sorted(allowed)}")
    return text


def _unique_sorted_cids(values: Iterable[Any], name: str) -> tuple[str, ...]:
    items = [validate_opaque_cid(item, name) for item in values]
    if len(items) > MAX_COLLECTION_ITEMS:
        raise ProgramWorldReceiptError(f"{name} exceeds its item bound")
    ordered = tuple(sorted(items))
    if len(ordered) != len(set(ordered)):
        raise ProgramWorldReceiptError(f"{name} must not contain duplicates")
    return ordered


def _unique_sorted_bounded_texts(values: Iterable[Any], name: str) -> tuple[str, ...]:
    items = [_bounded_text(item, name) for item in values]
    if len(items) > MAX_COLLECTION_ITEMS:
        raise ProgramWorldReceiptError(f"{name} exceeds its item bound")
    ordered = tuple(sorted(items))
    if len(ordered) != len(set(ordered)):
        raise ProgramWorldReceiptError(f"{name} must not contain duplicates")
    return ordered


def _reject_forbidden_identity_fields(mapping: Mapping[str, Any], name: str) -> None:
    if not isinstance(mapping, Mapping):
        raise ProgramWorldReceiptError(f"{name} must be an object")
    forbidden = set(mapping) & FORBIDDEN_IDENTITY_FIELDS
    if forbidden:
        raise ProgramWorldReceiptError(
            f"{name} rejects ANN/score identity fields {sorted(forbidden)}"
        )


def _false_flag(value: Any, name: str) -> bool:
    flag = _bool(value, name)
    if flag:
        raise ProgramWorldReceiptError(f"{name} must remain false")
    return False


def _operational_authorities(values: Iterable[Any]) -> tuple[str, ...]:
    authorities = _unique_sorted_bounded_texts(values, "operational_acceptance_authority")
    if not authorities:
        raise ProgramWorldReceiptError(
            "operational acceptance must name existing supervisor authorities"
        )
    extra = set(authorities) - set(OPERATIONAL_ACCEPTANCE_AUTHORITIES)
    if extra:
        raise ProgramWorldReceiptError(
            "unknown operational acceptance authorities "
            f"{sorted(extra)}; use validation/merge/event only"
        )
    missing = [item for item in OPERATIONAL_ACCEPTANCE_AUTHORITIES if item not in authorities]
    if missing:
        raise ProgramWorldReceiptError(
            "operational acceptance authorities missing "
            f"{missing}; existing supervisor validation/merge/event remain required"
        )
    return tuple(OPERATIONAL_ACCEPTANCE_AUTHORITIES)


def _forbid_self_admission(authority: str | None, *, admitted: bool) -> None:
    if authority is None:
        if admitted:
            raise ProgramWorldAdmissionError(
                "admitted receipts require an existing supervisor admission authority"
            )
        return
    if authority in ADAPTER_OWNED_AUTHORITIES:
        raise ProgramWorldAdmissionError(
            "adapters, compilers, and publishers cannot admit their own output"
        )
    if authority not in OPERATIONAL_ACCEPTANCE_AUTHORITIES:
        raise ProgramWorldAdmissionError(
            f"admission_authority {authority!r} is not an existing supervisor authority"
        )
    if not admitted:
        raise ProgramWorldReceiptError(
            "naming an admission authority requires admitted=true and evidence"
        )


def _require_proposal_or_admitted(
    *,
    proposal_only: bool,
    admitted: bool,
    admission_authority: str | None,
    admission_evidence_cid: str | None,
) -> tuple[bool, bool, str | None, str | None]:
    proposal_only = _bool(proposal_only, "proposal_only")
    admitted = _bool(admitted, "admitted")
    admission_authority = _optional_bounded_text(admission_authority, "admission_authority")
    admission_evidence_cid = _optional_cid(
        admission_evidence_cid, "admission_evidence_cid"
    )
    if admitted and proposal_only:
        raise ProgramWorldAdmissionError(
            "admitted operational evidence cannot remain proposal_only"
        )
    if proposal_only:
        if admitted or admission_authority is not None or admission_evidence_cid is not None:
            raise ProgramWorldAdmissionError(
                "proposal_only receipts cannot carry admission authority or evidence"
            )
        return True, False, None, None
    _forbid_self_admission(admission_authority, admitted=admitted)
    if admitted and admission_evidence_cid is None:
        raise ProgramWorldAdmissionError(
            "admitted operational evidence requires admission_evidence_cid "
            "from existing supervisor validation/merge/event authority"
        )
    return False, admitted, admission_authority, admission_evidence_cid


def _receipt_cid(payload: Mapping[str, Any]) -> str:
    _reject_forbidden_identity_fields(payload, "identity_payload")
    return cid_for_payload(payload)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProgramWorldCapabilityReceipt:
    """Typed capability/unavailability/fallback witness for one surface."""

    surface: str
    status: CapabilityStatus | str
    reason_code: str
    diagnostic: str
    operations: Sequence[str] = ()
    fallback: bool = False
    retryable: bool = False
    adapter_id: str = ADAPTER_ID

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_CAPABILITY_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_CAPABILITY_RECEIPT_INTERFACE
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "surface",
            "status",
            "reason_code",
            "diagnostic",
            "operations",
            "fallback",
            "retryable",
            "adapter_id",
            "capability_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "surface", _bounded_text(self.surface, "surface"))
        object.__setattr__(
            self, "status", _enum_value(self.status, CAPABILITY_STATUSES, "status")
        )
        object.__setattr__(
            self, "reason_code", _bounded_text(self.reason_code, "reason_code")
        )
        diagnostic = _bounded_text(self.diagnostic, "diagnostic")
        object.__setattr__(self, "diagnostic", diagnostic)
        object.__setattr__(
            self,
            "operations",
            _unique_sorted_bounded_texts(self.operations, "operation"),
        )
        fallback = _bool(self.fallback, "fallback")
        object.__setattr__(self, "fallback", fallback)
        object.__setattr__(self, "retryable", _bool(self.retryable, "retryable"))
        object.__setattr__(self, "adapter_id", _bounded_text(self.adapter_id, "adapter_id"))
        if self.status == CapabilityStatus.AVAILABLE.value:
            if fallback:
                raise ProgramWorldReceiptError(
                    "available capabilities cannot be recorded as fallback"
                )
            if self.reason_code != "available":
                raise ProgramWorldReceiptError(
                    "available capability receipts use reason_code 'available'"
                )
        else:
            if self.status == CapabilityStatus.FALLBACK.value and not fallback:
                raise ProgramWorldReceiptError("fallback status requires fallback=true")
            if self.status == CapabilityStatus.UNAVAILABLE.value and not fallback:
                object.__setattr__(self, "fallback", True)
            if not self.reason_code or self.reason_code == "available":
                raise ProgramWorldReceiptError(
                    "unavailable/fallback receipts require a typed reason_code"
                )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "surface": self.surface,
            "status": self.status,
            "reason_code": self.reason_code,
            "diagnostic": self.diagnostic,
            "operations": list(self.operations),
            "fallback": self.fallback,
            "retryable": self.retryable,
            "adapter_id": self.adapter_id,
        }

    @property
    def capability_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    @property
    def available(self) -> bool:
        return self.status == CapabilityStatus.AVAILABLE.value

    def to_unavailable_result(self, operation: str) -> UnavailableResult:
        return UnavailableResult(
            operation=_bounded_text(operation, "operation"),
            adapter_id=self.adapter_id,
            reason_code=self.reason_code,
            retryable=self.retryable,
            diagnostic=self.diagnostic,
        )

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["capability_cid"] = self.capability_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgramWorldCapabilityReceipt":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("capability_cid")
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReceiptError("unsupported capability receipt schema")
        if payload.pop("interface") != cls.INTERFACE:
            raise ProgramWorldReceiptError("unsupported capability receipt interface")
        result = cls(**payload)
        if result.capability_cid != claimed:
            raise ProgramWorldReceiptError("capability_cid does not rehash")
        return result


@dataclass(frozen=True, slots=True)
class ProgramWorldReuseDecision:
    """Exact reuse proposal. ANN never admits reuse; this record cannot self-admit."""

    verdict: ReuseVerdict | str
    state_cid: str
    goal_cid: str
    policy_cid: str
    environment_cid: str
    toolchain_cid: str
    reason_code: str
    relation_claim_cid: str | None = None
    procedure_revision_cid: str | None = None
    exact_match: bool = False
    ann_authoritative: bool = False
    proposal_only: bool = True
    admitted: bool = False
    admission_authority: str | None = None
    admission_evidence_cid: str | None = None
    operational_acceptance_authorities: Sequence[str] = OPERATIONAL_ACCEPTANCE_AUTHORITIES
    fallback: bool = False
    limitations: Sequence[str] = ()

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_REUSE_DECISION_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_REUSE_DECISION_INTERFACE
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "verdict",
            "state_cid",
            "goal_cid",
            "policy_cid",
            "environment_cid",
            "toolchain_cid",
            "relation_claim_cid",
            "procedure_revision_cid",
            "exact_match",
            "ann_authoritative",
            "proposal_only",
            "admitted",
            "admission_authority",
            "admission_evidence_cid",
            "operational_acceptance_authorities",
            "fallback",
            "reason_code",
            "limitations",
            "decision_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "verdict", _enum_value(self.verdict, REUSE_VERDICTS, "verdict")
        )
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
            "relation_claim_cid",
            _optional_cid(self.relation_claim_cid, "relation_claim_cid"),
        )
        object.__setattr__(
            self,
            "procedure_revision_cid",
            _optional_cid(self.procedure_revision_cid, "procedure_revision_cid"),
        )
        object.__setattr__(self, "exact_match", _bool(self.exact_match, "exact_match"))
        object.__setattr__(
            self, "ann_authoritative", _false_flag(self.ann_authoritative, "ann_authoritative")
        )
        proposal_only, admitted, authority, evidence = _require_proposal_or_admitted(
            proposal_only=self.proposal_only,
            admitted=self.admitted,
            admission_authority=self.admission_authority,
            admission_evidence_cid=self.admission_evidence_cid,
        )
        object.__setattr__(self, "proposal_only", proposal_only)
        object.__setattr__(self, "admitted", admitted)
        object.__setattr__(self, "admission_authority", authority)
        object.__setattr__(self, "admission_evidence_cid", evidence)
        object.__setattr__(
            self,
            "operational_acceptance_authorities",
            _operational_authorities(self.operational_acceptance_authorities),
        )
        object.__setattr__(self, "fallback", _bool(self.fallback, "fallback"))
        object.__setattr__(
            self, "reason_code", _bounded_text(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self,
            "limitations",
            _unique_sorted_bounded_texts(self.limitations, "limitation"),
        )
        if self.verdict == ReuseVerdict.REUSE.value:
            if not self.exact_match:
                raise ProgramWorldReceiptError("reuse requires exact identity match")
            if self.fallback:
                raise ProgramWorldReceiptError("reuse cannot be a capability fallback")
            if self.reason_code in ANN_REASON_CODES:
                raise ProgramWorldReceiptError("ANN/similarity cannot admit reuse")
        if self.verdict == ReuseVerdict.UNAVAILABLE.value and not self.fallback:
            object.__setattr__(self, "fallback", True)
        if admitted and self.verdict == ReuseVerdict.UNAVAILABLE.value:
            raise ProgramWorldReceiptError("unavailable reuse cannot be admitted")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "verdict": self.verdict,
            "state_cid": self.state_cid,
            "goal_cid": self.goal_cid,
            "policy_cid": self.policy_cid,
            "environment_cid": self.environment_cid,
            "toolchain_cid": self.toolchain_cid,
            "relation_claim_cid": self.relation_claim_cid,
            "procedure_revision_cid": self.procedure_revision_cid,
            "exact_match": self.exact_match,
            "ann_authoritative": self.ann_authoritative,
            "proposal_only": self.proposal_only,
            "admitted": self.admitted,
            "admission_authority": self.admission_authority,
            "admission_evidence_cid": self.admission_evidence_cid,
            "operational_acceptance_authorities": list(
                self.operational_acceptance_authorities
            ),
            "fallback": self.fallback,
            "reason_code": self.reason_code,
            "limitations": list(self.limitations),
        }

    @property
    def decision_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["decision_cid"] = self.decision_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgramWorldReuseDecision":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("decision_cid")
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReceiptError("unsupported reuse-decision schema")
        if payload.pop("interface") != cls.INTERFACE:
            raise ProgramWorldReceiptError("unsupported reuse-decision interface")
        result = cls(**payload)
        if result.decision_cid != claimed:
            raise ProgramWorldReceiptError("decision_cid does not rehash")
        return result


@dataclass(frozen=True, slots=True)
class ProgramWorldContextItem:
    """One included, omitted, or raw-fallback context citation."""

    identity_cid: str
    kind: str
    reason: str
    authority: str
    freshness: FreshnessState | str
    disposition: ContextItemDisposition | str
    required: bool = False
    source_cid: str | None = None

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_CONTEXT_ITEM_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "identity_cid",
            "kind",
            "reason",
            "authority",
            "freshness",
            "disposition",
            "required",
            "source_cid",
            "item_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "identity_cid", validate_opaque_cid(self.identity_cid, "identity_cid")
        )
        object.__setattr__(self, "kind", _bounded_text(self.kind, "kind"))
        object.__setattr__(self, "reason", _bounded_text(self.reason, "reason"))
        object.__setattr__(self, "authority", _bounded_text(self.authority, "authority"))
        object.__setattr__(
            self, "freshness", _enum_value(self.freshness, FRESHNESS_STATES, "freshness")
        )
        object.__setattr__(
            self,
            "disposition",
            _enum_value(self.disposition, CONTEXT_ITEM_DISPOSITIONS, "disposition"),
        )
        object.__setattr__(self, "required", _bool(self.required, "required"))
        object.__setattr__(self, "source_cid", _optional_cid(self.source_cid, "source_cid"))
        if self.required and self.disposition == ContextItemDisposition.OMITTED.value:
            raise ProgramWorldReceiptError(
                "required context material cannot be omitted"
            )
        lowered = f"{self.reason} {self.kind}".lower()
        if self.required and any(
            marker in lowered
            for marker in ("embedding", "ann", "similar", "knn", "cosine")
        ):
            raise ProgramWorldReceiptError(
                "embeddings cannot suppress required context material"
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "identity_cid": self.identity_cid,
            "kind": self.kind,
            "reason": self.reason,
            "authority": self.authority,
            "freshness": self.freshness,
            "disposition": self.disposition,
            "required": self.required,
            "source_cid": self.source_cid,
        }

    @property
    def item_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["item_cid"] = self.item_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgramWorldContextItem":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("item_cid")
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReceiptError("unsupported context-item schema")
        result = cls(**payload)
        if result.item_cid != claimed:
            raise ProgramWorldReceiptError("item_cid does not rehash")
        return result


def _parse_context_items(
    values: Sequence[Any], *, expected_disposition: str
) -> tuple[ProgramWorldContextItem, ...]:
    items: list[ProgramWorldContextItem] = []
    seen: set[str] = set()
    for raw in values:
        if isinstance(raw, ProgramWorldContextItem):
            item = raw
        elif isinstance(raw, Mapping):
            _reject_forbidden_identity_fields(raw, "context_item")
            item = ProgramWorldContextItem.from_dict(raw) if "item_cid" in raw else ProgramWorldContextItem(**dict(raw))
        else:
            raise ProgramWorldReceiptError("context items must be mappings or records")
        if item.disposition != expected_disposition:
            raise ProgramWorldReceiptError(
                f"context item disposition {item.disposition!r} does not match "
                f"{expected_disposition!r}"
            )
        if item.item_cid in seen:
            raise ProgramWorldReceiptError("context items must not contain duplicates")
        seen.add(item.item_cid)
        items.append(item)
    if len(items) > MAX_COLLECTION_ITEMS:
        raise ProgramWorldReceiptError("context items exceed their bound")
    return tuple(items)


@dataclass(frozen=True, slots=True)
class ProgramWorldContextReceipt:
    """Inclusion/omission/raw-fallback receipt. Embeddings cannot omit required material."""

    included: Sequence[ProgramWorldContextItem]
    omitted: Sequence[ProgramWorldContextItem] = ()
    raw_fallbacks: Sequence[ProgramWorldContextItem] = ()
    unresolved_questions: Sequence[str] = ()
    token_budget: int | None = None
    embeddings_suppressed_required: bool = False
    proposal_only: bool = True
    admitted: bool = False
    admission_authority: str | None = None
    admission_evidence_cid: str | None = None
    operational_acceptance_authorities: Sequence[str] = OPERATIONAL_ACCEPTANCE_AUTHORITIES
    fallback: bool = False
    reason_code: str = "context_recorded"

    SCHEMA: ClassVar[str] = PROGRAM_WORLD_CONTEXT_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = PROGRAM_WORLD_CONTEXT_RECEIPT_INTERFACE
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "included",
            "omitted",
            "raw_fallbacks",
            "unresolved_questions",
            "token_budget",
            "embeddings_suppressed_required",
            "proposal_only",
            "admitted",
            "admission_authority",
            "admission_evidence_cid",
            "operational_acceptance_authorities",
            "fallback",
            "reason_code",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "included",
            _parse_context_items(self.included, expected_disposition="included"),
        )
        object.__setattr__(
            self,
            "omitted",
            _parse_context_items(self.omitted, expected_disposition="omitted"),
        )
        object.__setattr__(
            self,
            "raw_fallbacks",
            _parse_context_items(self.raw_fallbacks, expected_disposition="raw_fallback"),
        )
        object.__setattr__(
            self,
            "unresolved_questions",
            _unique_sorted_bounded_texts(self.unresolved_questions, "unresolved_question"),
        )
        if self.token_budget is not None:
            if type(self.token_budget) is not int or isinstance(self.token_budget, bool):
                raise ProgramWorldReceiptError("token_budget must be an integer or null")
            if self.token_budget < 0:
                raise ProgramWorldReceiptError("token_budget must be nonnegative")
        object.__setattr__(
            self,
            "embeddings_suppressed_required",
            _false_flag(
                self.embeddings_suppressed_required, "embeddings_suppressed_required"
            ),
        )
        proposal_only, admitted, authority, evidence = _require_proposal_or_admitted(
            proposal_only=self.proposal_only,
            admitted=self.admitted,
            admission_authority=self.admission_authority,
            admission_evidence_cid=self.admission_evidence_cid,
        )
        object.__setattr__(self, "proposal_only", proposal_only)
        object.__setattr__(self, "admitted", admitted)
        object.__setattr__(self, "admission_authority", authority)
        object.__setattr__(self, "admission_evidence_cid", evidence)
        object.__setattr__(
            self,
            "operational_acceptance_authorities",
            _operational_authorities(self.operational_acceptance_authorities),
        )
        object.__setattr__(self, "fallback", _bool(self.fallback, "fallback"))
        object.__setattr__(
            self, "reason_code", _bounded_text(self.reason_code, "reason_code")
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "included": [item.to_dict() for item in self.included],
            "omitted": [item.to_dict() for item in self.omitted],
            "raw_fallbacks": [item.to_dict() for item in self.raw_fallbacks],
            "unresolved_questions": list(self.unresolved_questions),
            "token_budget": self.token_budget,
            "embeddings_suppressed_required": self.embeddings_suppressed_required,
            "proposal_only": self.proposal_only,
            "admitted": self.admitted,
            "admission_authority": self.admission_authority,
            "admission_evidence_cid": self.admission_evidence_cid,
            "operational_acceptance_authorities": list(
                self.operational_acceptance_authorities
            ),
            "fallback": self.fallback,
            "reason_code": self.reason_code,
        }

    @property
    def receipt_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["receipt_cid"] = self.receipt_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgramWorldContextReceipt":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("receipt_cid")
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReceiptError("unsupported context-receipt schema")
        if payload.pop("interface") != cls.INTERFACE:
            raise ProgramWorldReceiptError("unsupported context-receipt interface")
        result = cls(**payload)
        if result.receipt_cid != claimed:
            raise ProgramWorldReceiptError("receipt_cid does not rehash")
        return result


@dataclass(frozen=True, slots=True)
class ExecutionTransitionCompilation:
    """Compiled operational transition. Proposal-only unless supervisor-admitted."""

    status: CompilationStatus | str
    query_cid: str
    subject_cid: str
    environment_cid: str
    policy_cid: str
    reason_code: str
    candidate_cid: str | None = None
    prediction_cid: str | None = None
    observation_cid: str | None = None
    datasets_transition_cid: str | None = None
    prediction_authoritative: bool = False
    proposal_only: bool = True
    admitted: bool = False
    admission_authority: str | None = None
    admission_evidence_cid: str | None = None
    operational_acceptance_authorities: Sequence[str] = OPERATIONAL_ACCEPTANCE_AUTHORITIES
    fallback: bool = False
    limitations: Sequence[str] = ()

    SCHEMA: ClassVar[str] = EXECUTION_TRANSITION_COMPILATION_SCHEMA
    INTERFACE: ClassVar[str] = EXECUTION_TRANSITION_COMPILATION_INTERFACE
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "status",
            "query_cid",
            "subject_cid",
            "environment_cid",
            "policy_cid",
            "candidate_cid",
            "prediction_cid",
            "observation_cid",
            "datasets_transition_cid",
            "prediction_authoritative",
            "proposal_only",
            "admitted",
            "admission_authority",
            "admission_evidence_cid",
            "operational_acceptance_authorities",
            "fallback",
            "reason_code",
            "limitations",
            "compilation_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _enum_value(self.status, COMPILATION_STATUSES, "status")
        )
        for name in ("query_cid", "subject_cid", "environment_cid", "policy_cid"):
            object.__setattr__(self, name, validate_opaque_cid(getattr(self, name), name))
        object.__setattr__(
            self, "candidate_cid", _optional_cid(self.candidate_cid, "candidate_cid")
        )
        object.__setattr__(
            self, "prediction_cid", _optional_cid(self.prediction_cid, "prediction_cid")
        )
        object.__setattr__(
            self,
            "observation_cid",
            _optional_cid(self.observation_cid, "observation_cid"),
        )
        object.__setattr__(
            self,
            "datasets_transition_cid",
            _optional_cid(self.datasets_transition_cid, "datasets_transition_cid"),
        )
        object.__setattr__(
            self,
            "prediction_authoritative",
            _false_flag(self.prediction_authoritative, "prediction_authoritative"),
        )
        proposal_only, admitted, authority, evidence = _require_proposal_or_admitted(
            proposal_only=self.proposal_only,
            admitted=self.admitted,
            admission_authority=self.admission_authority,
            admission_evidence_cid=self.admission_evidence_cid,
        )
        object.__setattr__(self, "proposal_only", proposal_only)
        object.__setattr__(self, "admitted", admitted)
        object.__setattr__(self, "admission_authority", authority)
        object.__setattr__(self, "admission_evidence_cid", evidence)
        object.__setattr__(
            self,
            "operational_acceptance_authorities",
            _operational_authorities(self.operational_acceptance_authorities),
        )
        object.__setattr__(self, "fallback", _bool(self.fallback, "fallback"))
        object.__setattr__(
            self, "reason_code", _bounded_text(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self,
            "limitations",
            _unique_sorted_bounded_texts(self.limitations, "limitation"),
        )
        if (
            self.prediction_cid is not None
            and self.observation_cid is None
            and admitted
            and self.admission_evidence_cid is None
        ):
            raise ProgramWorldAdmissionError(
                "predictions cannot self-admit; operational admission requires "
                "observation or existing supervisor validation evidence"
            )
        if self.status == CompilationStatus.UNAVAILABLE.value and not self.fallback:
            object.__setattr__(self, "fallback", True)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "status": self.status,
            "query_cid": self.query_cid,
            "subject_cid": self.subject_cid,
            "environment_cid": self.environment_cid,
            "policy_cid": self.policy_cid,
            "candidate_cid": self.candidate_cid,
            "prediction_cid": self.prediction_cid,
            "observation_cid": self.observation_cid,
            "datasets_transition_cid": self.datasets_transition_cid,
            "prediction_authoritative": self.prediction_authoritative,
            "proposal_only": self.proposal_only,
            "admitted": self.admitted,
            "admission_authority": self.admission_authority,
            "admission_evidence_cid": self.admission_evidence_cid,
            "operational_acceptance_authorities": list(
                self.operational_acceptance_authorities
            ),
            "fallback": self.fallback,
            "reason_code": self.reason_code,
            "limitations": list(self.limitations),
        }

    @property
    def compilation_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["compilation_cid"] = self.compilation_cid
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionTransitionCompilation":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("compilation_cid")
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReceiptError("unsupported compilation schema")
        if payload.pop("interface") != cls.INTERFACE:
            raise ProgramWorldReceiptError("unsupported compilation interface")
        result = cls(**payload)
        if result.compilation_cid != claimed:
            raise ProgramWorldReceiptError("compilation_cid does not rehash")
        return result


@dataclass(frozen=True, slots=True)
class OperationalWorldRootPublicationRequest:
    """Generation-bearing publication *request*. Never mutates the current root."""

    status: PublicationStatus | str
    semantic_world_root_cid: str
    expected_generation: int
    reason_code: str
    expected_root_cid: str | None = None
    candidate_operational_manifest_cid: str | None = None
    kit_verified: bool = False
    changes_current_root: bool = False
    proposal_only: bool = True
    admitted: bool = False
    admission_authority: str | None = None
    admission_evidence_cid: str | None = None
    operational_acceptance_authorities: Sequence[str] = OPERATIONAL_ACCEPTANCE_AUTHORITIES
    fallback: bool = False
    limitations: Sequence[str] = ()

    SCHEMA: ClassVar[str] = OPERATIONAL_WORLD_ROOT_PUBLICATION_REQUEST_SCHEMA
    INTERFACE: ClassVar[str] = OPERATIONAL_WORLD_ROOT_PUBLICATION_REQUEST_INTERFACE
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "status",
            "semantic_world_root_cid",
            "expected_generation",
            "expected_root_cid",
            "candidate_operational_manifest_cid",
            "kit_verified",
            "changes_current_root",
            "proposal_only",
            "admitted",
            "admission_authority",
            "admission_evidence_cid",
            "operational_acceptance_authorities",
            "fallback",
            "reason_code",
            "limitations",
            "request_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _enum_value(self.status, PUBLICATION_STATUSES, "status")
        )
        object.__setattr__(
            self,
            "semantic_world_root_cid",
            validate_opaque_cid(self.semantic_world_root_cid, "semantic_world_root_cid"),
        )
        if (
            type(self.expected_generation) is not int
            or isinstance(self.expected_generation, bool)
            or self.expected_generation < 0
        ):
            raise ProgramWorldReceiptError(
                "expected_generation must be a nonnegative integer"
            )
        object.__setattr__(
            self,
            "expected_root_cid",
            _optional_cid(self.expected_root_cid, "expected_root_cid"),
        )
        object.__setattr__(
            self,
            "candidate_operational_manifest_cid",
            _optional_cid(
                self.candidate_operational_manifest_cid,
                "candidate_operational_manifest_cid",
            ),
        )
        object.__setattr__(self, "kit_verified", _bool(self.kit_verified, "kit_verified"))
        object.__setattr__(
            self,
            "changes_current_root",
            _false_flag(self.changes_current_root, "changes_current_root"),
        )
        proposal_only, admitted, authority, evidence = _require_proposal_or_admitted(
            proposal_only=self.proposal_only,
            admitted=self.admitted,
            admission_authority=self.admission_authority,
            admission_evidence_cid=self.admission_evidence_cid,
        )
        object.__setattr__(self, "proposal_only", proposal_only)
        object.__setattr__(self, "admitted", admitted)
        object.__setattr__(self, "admission_authority", authority)
        object.__setattr__(self, "admission_evidence_cid", evidence)
        object.__setattr__(
            self,
            "operational_acceptance_authorities",
            _operational_authorities(self.operational_acceptance_authorities),
        )
        object.__setattr__(self, "fallback", _bool(self.fallback, "fallback"))
        object.__setattr__(
            self, "reason_code", _bounded_text(self.reason_code, "reason_code")
        )
        object.__setattr__(
            self,
            "limitations",
            _unique_sorted_bounded_texts(self.limitations, "limitation"),
        )
        if self.status == PublicationStatus.UNAVAILABLE.value and not self.fallback:
            object.__setattr__(self, "fallback", True)
        if self.status == PublicationStatus.REQUESTED.value and admitted:
            raise ProgramWorldReceiptError(
                "publication requests remain proposals; existing merge/event "
                "authority publishes the current root"
            )
        if "cannot_change_current_root" not in self.limitations:
            object.__setattr__(
                self,
                "limitations",
                tuple(
                    sorted(set(self.limitations) | {"cannot_change_current_root"})
                ),
            )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "status": self.status,
            "semantic_world_root_cid": self.semantic_world_root_cid,
            "expected_generation": self.expected_generation,
            "expected_root_cid": self.expected_root_cid,
            "candidate_operational_manifest_cid": (
                self.candidate_operational_manifest_cid
            ),
            "kit_verified": self.kit_verified,
            "changes_current_root": self.changes_current_root,
            "proposal_only": self.proposal_only,
            "admitted": self.admitted,
            "admission_authority": self.admission_authority,
            "admission_evidence_cid": self.admission_evidence_cid,
            "operational_acceptance_authorities": list(
                self.operational_acceptance_authorities
            ),
            "fallback": self.fallback,
            "reason_code": self.reason_code,
            "limitations": list(self.limitations),
        }

    @property
    def request_cid(self) -> str:
        return _receipt_cid(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        value = self.identity_payload()
        value["request_cid"] = self.request_cid
        return value

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any]
    ) -> "OperationalWorldRootPublicationRequest":
        payload = _closed(data, cls._FIELDS, cls.__name__)
        claimed = payload.pop("request_cid")
        if payload.pop("schema") != cls.SCHEMA:
            raise ProgramWorldReceiptError("unsupported publication-request schema")
        if payload.pop("interface") != cls.INTERFACE:
            raise ProgramWorldReceiptError(
                "unsupported publication-request interface"
            )
        result = cls(**payload)
        if result.request_cid != claimed:
            raise ProgramWorldReceiptError("request_cid does not rehash")
        return result


def available_capability(
    surface: str, *, operations: Sequence[str] = ()
) -> ProgramWorldCapabilityReceipt:
    return ProgramWorldCapabilityReceipt(
        surface=surface,
        status=CapabilityStatus.AVAILABLE,
        reason_code="available",
        diagnostic=f"{surface} is available in the bound current tree",
        operations=operations,
        fallback=False,
        retryable=False,
    )


def unavailable_capability(
    surface: str,
    *,
    reason_code: str,
    diagnostic: str,
    operations: Sequence[str] = (),
    retryable: bool = False,
) -> ProgramWorldCapabilityReceipt:
    return ProgramWorldCapabilityReceipt(
        surface=surface,
        status=CapabilityStatus.UNAVAILABLE,
        reason_code=reason_code,
        diagnostic=diagnostic,
        operations=operations,
        fallback=True,
        retryable=retryable,
    )


def fallback_capability(
    surface: str,
    *,
    reason_code: str,
    diagnostic: str,
    operations: Sequence[str] = (),
    retryable: bool = False,
) -> ProgramWorldCapabilityReceipt:
    return ProgramWorldCapabilityReceipt(
        surface=surface,
        status=CapabilityStatus.FALLBACK,
        reason_code=reason_code,
        diagnostic=diagnostic,
        operations=operations,
        fallback=True,
        retryable=retryable,
    )


__all__ = [
    "ADAPTER_ID",
    "ADAPTER_OWNED_AUTHORITIES",
    "ANN_REASON_CODES",
    "CAPABILITY_STATUSES",
    "COMPILATION_STATUSES",
    "CONTEXT_ITEM_DISPOSITIONS",
    "DATASETS_DOMAIN_ADAPTER_AUTHORITY",
    "DATASETS_IDENTITY_AUTHORITY",
    "DATASETS_RELATION_AUTHORITY",
    "DATASETS_TRANSITION_AUTHORITY",
    "EVENT_AUTHORITY",
    "EXECUTION_TRANSITION_COMPILATION_INTERFACE",
    "EXECUTION_TRANSITION_COMPILATION_SCHEMA",
    "FORBIDDEN_IDENTITY_FIELDS",
    "KIT_VERIFIED_STORE_AUTHORITY",
    "MERGE_AUTHORITY",
    "OPERATIONAL_ACCEPTANCE_AUTHORITIES",
    "OPERATIONAL_WORLD_ROOT_PUBLICATION_REQUEST_INTERFACE",
    "OPERATIONAL_WORLD_ROOT_PUBLICATION_REQUEST_SCHEMA",
    "PROGRAM_WORLD_CAPABILITY_RECEIPT_INTERFACE",
    "PROGRAM_WORLD_CAPABILITY_RECEIPT_SCHEMA",
    "PROGRAM_WORLD_CONTEXT_RECEIPT_INTERFACE",
    "PROGRAM_WORLD_CONTEXT_RECEIPT_SCHEMA",
    "PROGRAM_WORLD_REUSE_DECISION_INTERFACE",
    "PROGRAM_WORLD_REUSE_DECISION_SCHEMA",
    "PUBLICATION_STATUSES",
    "REUSE_VERDICTS",
    "SEMANTIC_WORLD_OPERATIONAL_ADAPTERS_INTERFACE",
    "VALIDATION_AUTHORITY",
    "CapabilityStatus",
    "CompilationStatus",
    "ContextItemDisposition",
    "ExecutionTransitionCompilation",
    "FreshnessState",
    "OperationalWorldRootPublicationRequest",
    "ProgramWorldAdmissionError",
    "ProgramWorldCapabilityReceipt",
    "ProgramWorldContextItem",
    "ProgramWorldContextReceipt",
    "ProgramWorldReceiptError",
    "ProgramWorldReuseDecision",
    "PublicationStatus",
    "ReuseVerdict",
    "available_capability",
    "fallback_capability",
    "unavailable_capability",
]
