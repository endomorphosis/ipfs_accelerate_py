"""Bounded, proof-gated contracts for contract-repair target selection.

These records are deliberately *references*, never source containers.  They
form the narrow authority boundary between a broken-call analysis and a later
repair packet: nominations may be broad, but only an admitted decision may
grant a write path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


CONTRACT_REPAIR_VERSION: Final[int] = 1
MAX_RECORD_BYTES: Final[int] = 262_144
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_PATH_BYTES: Final[int] = 1_024
MAX_REFERENCE_COUNT: Final[int] = 256
MAX_CANDIDATE_COUNT: Final[int] = 256
MAX_SPAN_OFFSET: Final[int] = 2**63 - 1

AUTHORITY_ROOTS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair/authority-roots@1"
)
SOURCE_SPAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair/source-span@1"
)
EVIDENCE_REFERENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-repair/evidence-reference@1"
)
BROKEN_CONTRACT_TRACE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/broken-contract-trace@1"
)
CALL_REQUIREMENT_CONTRACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/call-requirement-contract@1"
)
MEMORY_SAFETY_FACET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/memory-safety-facet@1"
)
REPAIR_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-candidate@1"
)
REPAIR_TARGET_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-target-decision@1"
)
CANDIDATE_SET_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-candidate-set@1"
)


class ContractRepairError(ContractValidationError):
    """Base class for contract-repair schema failures."""


class ContractRepairBoundsError(ContractRepairError):
    """A record attempted to exceed its declared compactness bounds."""


class ForgedContractRepairIdentityError(ContractRepairError):
    """A stored content identity did not match the canonical preimage."""


class ContractRepairAuthorityError(ContractRepairError):
    """Authority roots, paths, or candidate bindings did not match exactly."""


class TraceDisposition(str, Enum):
    """Closed outcomes of conservative call resolution."""

    RESOLVED_MISMATCH = "resolved_mismatch"
    MISSING_LOCAL = "missing_local"
    LIKELY_REFACTOR = "likely_refactor"
    ADAPTER_REQUIRED = "adapter_required"
    EXTERNAL = "external"
    DYNAMIC = "dynamic"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"


class RepairStrategy(str, Enum):
    """Closed write-placement strategies; nomination is not authorization."""

    RENAME_SUBSTITUTION = "rename_substitution"
    ADAPTER = "adapter"
    IMPLEMENT_EXISTING_DECLARATION = "implement_existing_declaration"
    NEW_IMPLEMENTATION = "new_implementation"
    REJECT = "reject"
    AMBIGUOUS = "ambiguous"


class DecisionDisposition(str, Enum):
    ADMITTED = "admitted"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


class MemorySafetyDisposition(str, Enum):
    """Evidence state, intentionally independent of resource quantities."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    EMPIRICAL = "empirical"
    PROVED = "proved"
    STALE = "stale"
    ERROR = "error"


_BODY_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "body", "source", "source_body", "source_text", "contents",
        "content", "snippet", "code", "file_text", "raw_ast", "ast_body",
    }
)


def _text(value: Any, field_name: str, *, required: bool = False,
          limit: int = MAX_TEXT_BYTES) -> str:
    if not isinstance(value, str):
        raise ContractRepairError(f"{field_name} must be a string")
    value = value.strip()
    if required and not value:
        raise ContractRepairError(f"{field_name} is required")
    if len(value.encode("utf-8")) > limit:
        raise ContractRepairBoundsError(f"{field_name} exceeds its byte bound")
    return value


def _identifier(value: Any, field_name: str) -> str:
    value = _text(value, field_name, required=True)
    if any(char.isspace() for char in value):
        raise ContractRepairError(f"{field_name} must be an opaque compact identifier")
    return value


def _bounded_int(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractRepairError(f"{field_name} must be a finite integer")
    if value < minimum or value > MAX_SPAN_OFFSET:
        raise ContractRepairBoundsError(f"{field_name} is outside the supported bound")
    return value


def _path(value: Any, field_name: str) -> str:
    path = _text(value, field_name, required=True, limit=MAX_PATH_BYTES)
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or path in {".", ""}:
        raise ContractRepairAuthorityError(f"{field_name} must be a relative repository path")
    return candidate.as_posix()


def _enum(value: Any, enum: type[Enum], field_name: str) -> Enum:
    try:
        return value if isinstance(value, enum) else enum(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum)
        raise ContractRepairError(f"{field_name} must be one of: {allowed}") from exc


def _ids(values: Any, field_name: str, *, required: bool = False,
         limit: int = MAX_REFERENCE_COUNT) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str) or not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        raise ContractRepairError(f"{field_name} must be a sequence of identifiers")
    else:
        raw = values
    if len(raw) > limit:
        raise ContractRepairBoundsError(f"{field_name} exceeds its item bound")
    result = tuple(sorted({_identifier(value, field_name) for value in raw}))
    if required and not result:
        raise ContractRepairError(f"{field_name} must not be empty")
    return result


def _paths(values: Any, field_name: str, *, limit: int = MAX_REFERENCE_COUNT) -> tuple[str, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, str) or not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        raise ContractRepairError(f"{field_name} must be a sequence of paths")
    else:
        raw = values
    if len(raw) > limit:
        raise ContractRepairBoundsError(f"{field_name} exceeds its item bound")
    return tuple(sorted({_path(value, field_name) for value in raw}))


def _assert_body_free(value: Any, field_name: str = "record") -> None:
    """Reject source bodies even when smuggled through an opaque mapping."""
    if isinstance(value, float):
        raise ContractRepairError(f"{field_name} may not contain floating-point values")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractRepairError(f"{field_name} has a non-string key")
            normalized = key.lower().replace("-", "_").strip()
            if normalized in _BODY_MARKERS:
                raise ContractRepairError(f"{field_name} may not contain source bodies")
            _assert_body_free(item, field_name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_body_free(item, field_name)
    elif isinstance(value, (bytes, bytearray)):
        raise ContractRepairError(f"{field_name} may not contain binary bodies")


def _bounded(record: CanonicalContract, name: str) -> None:
    _assert_body_free(record.to_dict(), name)
    if len(canonical_json_bytes(record.to_dict())) > MAX_RECORD_BYTES:
        raise ContractRepairBoundsError(f"{name} exceeds its serialized byte bound")


def _verify_identity(payload: Mapping[str, Any], record: CanonicalContract) -> None:
    supplied = payload.get("content_id", payload.get("cid", ""))
    if supplied not in (None, ""):
        if not isinstance(supplied, str) or supplied != record.content_id:
            raise ForgedContractRepairIdentityError(
                "stored content identity does not match the canonical record"
            )


def _decode_fields(
    payload: Mapping[str, Any], schema: str, fields: Sequence[str], name: str
) -> dict[str, Any]:
    """Fail-closed decoder shared by every externally supplied record."""
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise ContractRepairError(f"{name} has an unsupported schema")
    if payload.get("contract_version") not in (None, CONTRACT_REPAIR_VERSION):
        raise ContractRepairError(f"{name} has an unsupported contract version")
    allowed = set(fields) | {"schema", "contract_version", "content_id", "cid"}
    if set(payload).difference(allowed):
        raise ContractRepairError(f"{name} contains unsupported fields")
    _assert_body_free(payload, name)
    try:
        return {field_name: payload[field_name] for field_name in fields if field_name in payload}
    except KeyError as exc:
        raise ContractRepairError(f"{name} omits a required field") from exc


@dataclass(frozen=True)
class AuthorityRoots(CanonicalContract):
    """Every root whose drift invalidates a repair record."""

    SCHEMA: ClassVar[str] = AUTHORITY_ROOTS_SCHEMA

    repository_id: str
    forest_id: str
    tree_id: str
    graph_id: str
    index_id: str
    model_id: str
    config_id: str
    translator_id: str
    toolchain_id: str
    policy_id: str

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            if field_name != "SCHEMA":
                object.__setattr__(self, field_name, _identifier(getattr(self, field_name), field_name))
        _bounded(self, "authority roots")

    def _payload(self) -> dict[str, Any]:
        return {"contract_version": CONTRACT_REPAIR_VERSION, **{
            name: getattr(self, name) for name in self.__dataclass_fields__ if name != "SCHEMA"
        }}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorityRoots":
        names = tuple(name for name in cls.__dataclass_fields__ if name != "SCHEMA")
        value = cls(**_decode_fields(payload, cls.SCHEMA, names, "authority roots"))
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True, order=True)
class SourceSpan(CanonicalContract):
    """Exact, half-open byte offsets in one repository-relative path."""

    SCHEMA: ClassVar[str] = SOURCE_SPAN_SCHEMA

    path: str
    start: int
    end: int
    artifact_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _path(self.path, "path"))
        object.__setattr__(self, "start", _bounded_int(self.start, "start"))
        object.__setattr__(self, "end", _bounded_int(self.end, "end"))
        object.__setattr__(self, "artifact_id", _identifier(self.artifact_id, "artifact_id"))
        if self.end < self.start:
            raise ContractRepairError("span end must be at or after span start")
        _bounded(self, "source span")

    def _payload(self) -> dict[str, Any]:
        return {"contract_version": CONTRACT_REPAIR_VERSION, "path": self.path,
                "start": self.start, "end": self.end, "artifact_id": self.artifact_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceSpan":
        value = cls(**_decode_fields(payload, cls.SCHEMA, ("path", "start", "end", "artifact_id"), "source span"))
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class EvidenceReference(CanonicalContract):
    """A typed, content-addressed pointer; it cannot carry evidence bodies."""

    SCHEMA: ClassVar[str] = EVIDENCE_REFERENCE_SCHEMA

    kind: str
    artifact_id: str
    locator: str = ""
    producer_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _identifier(self.kind, "kind"))
        if self.kind.lower().replace("-", "_") in _BODY_MARKERS:
            raise ContractRepairError("evidence references may not name source bodies")
        object.__setattr__(self, "artifact_id", _identifier(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "locator", _text(self.locator, "locator"))
        object.__setattr__(self, "producer_id", _text(self.producer_id, "producer_id"))
        _bounded(self, "evidence reference")

    def _payload(self) -> dict[str, Any]:
        return {"contract_version": CONTRACT_REPAIR_VERSION, "kind": self.kind,
                "artifact_id": self.artifact_id, "locator": self.locator,
                "producer_id": self.producer_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceReference":
        value = cls(**_decode_fields(payload, cls.SCHEMA, ("kind", "artifact_id", "locator", "producer_id"), "evidence reference"))
        _verify_identity(payload, value)
        return value


def _references(values: Any, field_name: str, *, required: bool = False) -> tuple[EvidenceReference, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        raw = values
    else:
        raise ContractRepairError(f"{field_name} must be a sequence of evidence references")
    if len(raw) > MAX_REFERENCE_COUNT:
        raise ContractRepairBoundsError(f"{field_name} exceeds its item bound")
    refs: list[EvidenceReference] = []
    for item in raw:
        if isinstance(item, EvidenceReference):
            ref = item
        elif isinstance(item, Mapping):
            ref = EvidenceReference.from_dict(item) if "schema" in item else EvidenceReference(
                **{key: item[key] for key in ("kind", "artifact_id", "locator", "producer_id") if key in item}
            )
        else:
            raise ContractRepairError(f"{field_name} contains an invalid evidence reference")
        if ref not in refs:
            refs.append(ref)
    result = tuple(sorted(refs, key=lambda ref: ref.content_id))
    if required and not result:
        raise ContractRepairError(f"{field_name} must not be empty")
    return result


def _roots(value: Any) -> AuthorityRoots:
    if isinstance(value, AuthorityRoots):
        return value
    if isinstance(value, Mapping):
        return AuthorityRoots.from_dict(value) if "schema" in value else AuthorityRoots(**value)
    raise ContractRepairError("roots must be AuthorityRoots")


def _span(value: Any, field_name: str) -> SourceSpan:
    if isinstance(value, SourceSpan):
        return value
    if isinstance(value, Mapping):
        return SourceSpan.from_dict(value) if "schema" in value else SourceSpan(**value)
    raise ContractRepairError(f"{field_name} must be a SourceSpan")


@dataclass(frozen=True)
class BrokenContractTrace(CanonicalContract):
    """Compact facts about one unresolved or mismatched caller-to-receiver edge."""

    SCHEMA: ClassVar[str] = BROKEN_CONTRACT_TRACE_SCHEMA

    roots: AuthorityRoots
    caller_span: SourceSpan
    caller_symbol_id: str
    receiver_reference: str
    disposition: TraceDisposition
    target_span: SourceSpan | None = None
    evidence_refs: tuple[EvidenceReference, ...] = ()
    proof_refs: tuple[EvidenceReference, ...] = ()
    graph_frontier_refs: tuple[str, ...] = ()
    excluded_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.roots, AuthorityRoots) or not isinstance(self.caller_span, SourceSpan):
            raise ContractRepairError("roots and caller_span must be typed contracts")
        if self.target_span is not None and not isinstance(self.target_span, SourceSpan):
            raise ContractRepairError("target_span must be a SourceSpan")
        object.__setattr__(self, "caller_symbol_id", _identifier(self.caller_symbol_id, "caller_symbol_id"))
        object.__setattr__(self, "receiver_reference", _text(self.receiver_reference, "receiver_reference", required=True))
        object.__setattr__(self, "disposition", _enum(self.disposition, TraceDisposition, "disposition"))
        object.__setattr__(self, "evidence_refs", _references(self.evidence_refs, "evidence_refs", required=True))
        object.__setattr__(self, "proof_refs", _references(self.proof_refs, "proof_refs"))
        object.__setattr__(self, "graph_frontier_refs", _ids(self.graph_frontier_refs, "graph_frontier_refs"))
        object.__setattr__(self, "excluded_refs", _ids(self.excluded_refs, "excluded_refs"))
        # A resolver cannot claim a resolved target from dynamic/unsupported evidence.
        if self.disposition in {TraceDisposition.DYNAMIC, TraceDisposition.EXTERNAL, TraceDisposition.UNSUPPORTED} and self.target_span is not None:
            raise ContractRepairError("unresolvable trace dispositions cannot name a target span")
        if self.disposition is TraceDisposition.RESOLVED_MISMATCH and self.target_span is None:
            raise ContractRepairError("resolved mismatch requires the resolved target span")
        _bounded(self, "broken contract trace")

    def _payload(self) -> dict[str, Any]:
        return {"contract_version": CONTRACT_REPAIR_VERSION, "roots": self.roots.to_dict(),
                "caller_span": self.caller_span.to_dict(), "caller_symbol_id": self.caller_symbol_id,
                "receiver_reference": self.receiver_reference, "disposition": self.disposition.value,
                "target_span": self.target_span.to_dict() if self.target_span else None,
                "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
                "proof_refs": [ref.to_dict() for ref in self.proof_refs],
                "graph_frontier_refs": list(self.graph_frontier_refs), "excluded_refs": list(self.excluded_refs)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BrokenContractTrace":
        fields = ("roots", "caller_span", "caller_symbol_id", "receiver_reference", "disposition", "target_span", "evidence_refs", "proof_refs", "graph_frontier_refs", "excluded_refs")
        values = _decode_fields(payload, cls.SCHEMA, fields, "broken contract trace")
        values["roots"] = _roots(values["roots"])
        values["caller_span"] = _span(values["caller_span"], "caller_span")
        if values.get("target_span") is not None:
            values["target_span"] = _span(values["target_span"], "target_span")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class CallRequirementContract(CanonicalContract):
    """Sender requirement joined with a receiver contract by later proof work."""

    SCHEMA: ClassVar[str] = CALL_REQUIREMENT_CONTRACT_SCHEMA

    roots: AuthorityRoots
    trace_id: str
    caller_span: SourceSpan
    requirement_refs: tuple[EvidenceReference, ...]
    receiver_contract_refs: tuple[EvidenceReference, ...] = ()
    evidence_refs: tuple[EvidenceReference, ...] = ()
    proof_refs: tuple[EvidenceReference, ...] = ()
    unsupported_clause_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.roots, AuthorityRoots) or not isinstance(self.caller_span, SourceSpan):
            raise ContractRepairError("roots and caller_span must be typed contracts")
        object.__setattr__(self, "trace_id", _identifier(self.trace_id, "trace_id"))
        object.__setattr__(self, "requirement_refs", _references(self.requirement_refs, "requirement_refs", required=True))
        object.__setattr__(self, "receiver_contract_refs", _references(self.receiver_contract_refs, "receiver_contract_refs"))
        object.__setattr__(self, "evidence_refs", _references(self.evidence_refs, "evidence_refs", required=True))
        object.__setattr__(self, "proof_refs", _references(self.proof_refs, "proof_refs"))
        object.__setattr__(self, "unsupported_clause_refs", _ids(self.unsupported_clause_refs, "unsupported_clause_refs"))
        _bounded(self, "call requirement contract")

    def _payload(self) -> dict[str, Any]:
        return {"contract_version": CONTRACT_REPAIR_VERSION, "roots": self.roots.to_dict(),
                "trace_id": self.trace_id, "caller_span": self.caller_span.to_dict(),
                "requirement_refs": [item.to_dict() for item in self.requirement_refs],
                "receiver_contract_refs": [item.to_dict() for item in self.receiver_contract_refs],
                "evidence_refs": [item.to_dict() for item in self.evidence_refs],
                "proof_refs": [item.to_dict() for item in self.proof_refs],
                "unsupported_clause_refs": list(self.unsupported_clause_refs)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CallRequirementContract":
        fields = ("roots", "trace_id", "caller_span", "requirement_refs", "receiver_contract_refs", "evidence_refs", "proof_refs", "unsupported_clause_refs")
        values = _decode_fields(payload, cls.SCHEMA, fields, "call requirement contract")
        values["roots"] = _roots(values["roots"])
        values["caller_span"] = _span(values["caller_span"], "caller_span")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class MemorySafetyFacet(CanonicalContract):
    """Separate memory-safety evidence; resource limits never appear here."""

    SCHEMA: ClassVar[str] = MEMORY_SAFETY_FACET_SCHEMA

    roots: AuthorityRoots
    subject_span: SourceSpan
    language_runtime: str
    disposition: MemorySafetyDisposition
    evidence_refs: tuple[EvidenceReference, ...] = ()
    proof_refs: tuple[EvidenceReference, ...] = ()
    unsupported_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.roots, AuthorityRoots) or not isinstance(self.subject_span, SourceSpan):
            raise ContractRepairError("roots and subject_span must be typed contracts")
        object.__setattr__(self, "language_runtime", _identifier(self.language_runtime, "language_runtime"))
        object.__setattr__(self, "disposition", _enum(self.disposition, MemorySafetyDisposition, "disposition"))
        object.__setattr__(self, "evidence_refs", _references(self.evidence_refs, "evidence_refs"))
        object.__setattr__(self, "proof_refs", _references(self.proof_refs, "proof_refs"))
        object.__setattr__(self, "unsupported_refs", _ids(self.unsupported_refs, "unsupported_refs"))
        if self.disposition is MemorySafetyDisposition.PROVED and not self.proof_refs:
            raise ContractRepairError("proved memory safety requires proof references")
        if self.disposition is MemorySafetyDisposition.EMPIRICAL and not self.evidence_refs:
            raise ContractRepairError("empirical memory safety requires evidence references")
        if self.disposition is MemorySafetyDisposition.UNSUPPORTED and not self.unsupported_refs:
            raise ContractRepairError("unsupported memory safety requires unsupported references")
        if self.disposition is MemorySafetyDisposition.UNSUPPORTED and self.proof_refs:
            raise ContractRepairError("unsupported memory safety cannot carry proof references")
        _bounded(self, "memory safety facet")

    def _payload(self) -> dict[str, Any]:
        return {"contract_version": CONTRACT_REPAIR_VERSION, "roots": self.roots.to_dict(),
                "subject_span": self.subject_span.to_dict(), "language_runtime": self.language_runtime,
                "disposition": self.disposition.value, "evidence_refs": [item.to_dict() for item in self.evidence_refs],
                "proof_refs": [item.to_dict() for item in self.proof_refs], "unsupported_refs": list(self.unsupported_refs)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MemorySafetyFacet":
        fields = ("roots", "subject_span", "language_runtime", "disposition", "evidence_refs", "proof_refs", "unsupported_refs")
        values = _decode_fields(payload, cls.SCHEMA, fields, "memory safety facet")
        values["roots"] = _roots(values["roots"])
        values["subject_span"] = _span(values["subject_span"], "subject_span")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


@dataclass(frozen=True)
class RepairCandidate(CanonicalContract):
    """A nominated target carrying no write authority of its own."""

    SCHEMA: ClassVar[str] = REPAIR_CANDIDATE_SCHEMA

    roots: AuthorityRoots
    trace_id: str
    strategy: RepairStrategy
    target_span: SourceSpan
    evidence_refs: tuple[EvidenceReference, ...]
    proof_refs: tuple[EvidenceReference, ...] = ()
    permitted_read_paths: tuple[str, ...] = ()
    candidate_write_paths: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.roots, AuthorityRoots) or not isinstance(self.target_span, SourceSpan):
            raise ContractRepairError("roots and target_span must be typed contracts")
        object.__setattr__(self, "trace_id", _identifier(self.trace_id, "trace_id"))
        object.__setattr__(self, "strategy", _enum(self.strategy, RepairStrategy, "strategy"))
        object.__setattr__(self, "evidence_refs", _references(self.evidence_refs, "evidence_refs", required=True))
        object.__setattr__(self, "proof_refs", _references(self.proof_refs, "proof_refs"))
        object.__setattr__(self, "permitted_read_paths", _paths(self.permitted_read_paths, "permitted_read_paths"))
        object.__setattr__(self, "candidate_write_paths", _paths(self.candidate_write_paths, "candidate_write_paths"))
        object.__setattr__(self, "rejection_reasons", _ids(self.rejection_reasons, "rejection_reasons"))
        if self.strategy in {RepairStrategy.REJECT, RepairStrategy.AMBIGUOUS} and self.candidate_write_paths:
            raise ContractRepairAuthorityError("reject and ambiguous candidates cannot propose writes")
        _bounded(self, "repair candidate")

    def _payload(self) -> dict[str, Any]:
        return {"contract_version": CONTRACT_REPAIR_VERSION, "roots": self.roots.to_dict(),
                "trace_id": self.trace_id, "strategy": self.strategy.value, "target_span": self.target_span.to_dict(),
                "evidence_refs": [item.to_dict() for item in self.evidence_refs],
                "proof_refs": [item.to_dict() for item in self.proof_refs],
                "permitted_read_paths": list(self.permitted_read_paths),
                "candidate_write_paths": list(self.candidate_write_paths),
                "rejection_reasons": list(self.rejection_reasons)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairCandidate":
        fields = ("roots", "trace_id", "strategy", "target_span", "evidence_refs", "proof_refs", "permitted_read_paths", "candidate_write_paths", "rejection_reasons")
        values = _decode_fields(payload, cls.SCHEMA, fields, "repair candidate")
        values["roots"] = _roots(values["roots"])
        values["target_span"] = _span(values["target_span"], "target_span")
        value = cls(**values)
        _verify_identity(payload, value)
        return value


def candidate_set_identity(candidates: Sequence[RepairCandidate]) -> str:
    """Derive the identity of the complete, deterministically ordered set."""
    if not candidates or len(candidates) > MAX_CANDIDATE_COUNT:
        raise ContractRepairBoundsError("candidate set must contain a bounded nonempty set")
    ids = tuple(sorted(candidate.content_id for candidate in candidates))
    if len(set(ids)) != len(ids):
        raise ContractRepairError("candidate set contains duplicate candidates")
    return content_identity({"schema": CANDIDATE_SET_SCHEMA, "candidate_ids": list(ids)})


@dataclass(frozen=True)
class RepairTargetDecision(CanonicalContract):
    """The only record that may turn an admitted candidate into write authority."""

    SCHEMA: ClassVar[str] = REPAIR_TARGET_DECISION_SCHEMA

    roots: AuthorityRoots
    candidates: tuple[RepairCandidate, ...]
    candidate_set_id: str
    disposition: DecisionDisposition
    strategy: RepairStrategy
    selected_candidate_id: str = ""
    permitted_read_paths: tuple[str, ...] = ()
    permitted_write_paths: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceReference, ...] = ()
    proof_refs: tuple[EvidenceReference, ...] = ()
    invalidation_refs: tuple[str, ...] = ()

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        """Canonical identities of every candidate considered by this decision."""

        return tuple(candidate.content_id for candidate in self.candidates)

    @property
    def read_paths(self) -> tuple[str, ...]:
        """Exact read authority granted by this decision."""

        return self.permitted_read_paths

    @property
    def write_paths(self) -> tuple[str, ...]:
        """Exact write authority granted by this decision (empty when abstaining)."""

        return self.permitted_write_paths

    def __post_init__(self) -> None:
        if not isinstance(self.roots, AuthorityRoots):
            raise ContractRepairError("roots must be AuthorityRoots")
        if not isinstance(self.candidates, Sequence) or isinstance(self.candidates, (str, bytes, bytearray)):
            raise ContractRepairError("candidates must be a sequence")
        if not self.candidates or len(self.candidates) > MAX_CANDIDATE_COUNT:
            raise ContractRepairBoundsError("candidates must be a bounded nonempty set")
        candidates = tuple(sorted(self.candidates, key=lambda candidate: candidate.content_id))
        if any(not isinstance(candidate, RepairCandidate) for candidate in candidates):
            raise ContractRepairError("candidates must contain RepairCandidate records")
        if len({candidate.content_id for candidate in candidates}) != len(candidates):
            raise ContractRepairError("candidates must be unique")
        for candidate in candidates:
            if candidate.roots != self.roots:
                raise ContractRepairAuthorityError("all candidates must bind the decision authority roots")
        object.__setattr__(self, "candidates", candidates)
        expected_set_id = candidate_set_identity(candidates)
        if _identifier(self.candidate_set_id, "candidate_set_id") != expected_set_id:
            raise ForgedContractRepairIdentityError("candidate_set_id must identify the complete candidate set")
        object.__setattr__(self, "disposition", _enum(self.disposition, DecisionDisposition, "disposition"))
        object.__setattr__(self, "strategy", _enum(self.strategy, RepairStrategy, "strategy"))
        object.__setattr__(self, "selected_candidate_id", _text(self.selected_candidate_id, "selected_candidate_id"))
        object.__setattr__(self, "permitted_read_paths", _paths(self.permitted_read_paths, "permitted_read_paths"))
        object.__setattr__(self, "permitted_write_paths", _paths(self.permitted_write_paths, "permitted_write_paths"))
        object.__setattr__(self, "evidence_refs", _references(self.evidence_refs, "evidence_refs", required=True))
        object.__setattr__(self, "proof_refs", _references(self.proof_refs, "proof_refs"))
        object.__setattr__(self, "invalidation_refs", _ids(self.invalidation_refs, "invalidation_refs", required=True))
        selected = next((item for item in candidates if item.content_id == self.selected_candidate_id), None)
        if self.disposition is DecisionDisposition.ADMITTED:
            if self.strategy in {RepairStrategy.REJECT, RepairStrategy.AMBIGUOUS} or selected is None:
                raise ContractRepairAuthorityError("an admitted decision requires one admitted candidate strategy")
            if selected.strategy is not self.strategy:
                raise ContractRepairAuthorityError("decision strategy must equal selected candidate strategy")
            if not self.proof_refs:
                raise ContractRepairAuthorityError("an admitted decision requires proof references")
            if not set(self.permitted_write_paths).issubset(selected.candidate_write_paths):
                raise ContractRepairAuthorityError("write paths must be derived from the selected candidate")
        elif self.selected_candidate_id or self.permitted_write_paths:
            raise ContractRepairAuthorityError("non-admitted decisions cannot select a target or grant writes")
        if self.disposition is not DecisionDisposition.ADMITTED and self.strategy not in {RepairStrategy.REJECT, RepairStrategy.AMBIGUOUS}:
            raise ContractRepairError("non-admitted decisions must use reject or ambiguous strategy")
        _bounded(self, "repair target decision")

    def _payload(self) -> dict[str, Any]:
        return {"contract_version": CONTRACT_REPAIR_VERSION, "roots": self.roots.to_dict(),
                "candidates": [item.to_dict() for item in self.candidates], "candidate_set_id": self.candidate_set_id,
                "disposition": self.disposition.value, "strategy": self.strategy.value,
                "selected_candidate_id": self.selected_candidate_id,
                "permitted_read_paths": list(self.permitted_read_paths),
                "permitted_write_paths": list(self.permitted_write_paths),
                "evidence_refs": [item.to_dict() for item in self.evidence_refs],
                "proof_refs": [item.to_dict() for item in self.proof_refs],
                "invalidation_refs": list(self.invalidation_refs)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairTargetDecision":
        fields = ("roots", "candidates", "candidate_set_id", "disposition", "strategy", "selected_candidate_id", "permitted_read_paths", "permitted_write_paths", "evidence_refs", "proof_refs", "invalidation_refs")
        values = _decode_fields(payload, cls.SCHEMA, fields, "repair target decision")
        values["roots"] = _roots(values["roots"])
        raw_candidates = values.get("candidates")
        if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes, bytearray)):
            raise ContractRepairError("candidates must be a sequence")
        values["candidates"] = tuple(
            item if isinstance(item, RepairCandidate) else RepairCandidate.from_dict(item)
            for item in raw_candidates
        )
        value = cls(**values)
        _verify_identity(payload, value)
        return value


__all__ = [
    "AUTHORITY_ROOTS_SCHEMA", "BROKEN_CONTRACT_TRACE_SCHEMA", "CALL_REQUIREMENT_CONTRACT_SCHEMA",
    "CANDIDATE_SET_SCHEMA", "CONTRACT_REPAIR_VERSION", "ContractRepairAuthorityError",
    "ContractRepairBoundsError", "ContractRepairError", "DecisionDisposition", "EvidenceReference",
    "ForgedContractRepairIdentityError", "MAX_CANDIDATE_COUNT", "MAX_RECORD_BYTES",
    "MemorySafetyDisposition", "MemorySafetyFacet", "RepairCandidate", "RepairStrategy",
    "RepairTargetDecision", "SourceSpan", "TraceDisposition", "AuthorityRoots", "BrokenContractTrace",
    "CallRequirementContract", "candidate_set_identity",
]
