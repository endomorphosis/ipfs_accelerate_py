"""Evidence-bound, profile-driven program contract profiles.

Domain vocabulary (operations, invariants, error codes, public surfaces, schema
identity, goal identifiers) is supplied by an immutable
:class:`ContractVocabulary` and compiled into a
:class:`ProgramContractProfile`.  The generic engine never embeds a domain
product vocabulary: profiles for storage facades, RPC/key-value services, or
any other bounded interface are data, not module constants.

Every resolved expectation and every canonical vector must cite independent
source evidence with explicit authority.  Missing and conflicting expectations
remain unresolved; the compiler never selects a popular implementation to
resolve them.  Profiles are comparison contracts only: they are not completion
evidence, correctness evidence, or repair authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.proof.program_contracts import (
    SOURCE_PRECEDENCE,
    ContractSourceKind,
)

# Authority bounds: a profile is a comparison contract, never proof or permission.
PROFILE_IS_COMPLETION_EVIDENCE: Final[bool] = False
PROFILE_IS_CORRECTNESS_EVIDENCE: Final[bool] = False
PROFILE_AUTHORIZES_REPAIR: Final[bool] = False

# Neutral default schema identity.  Domain profiles may override these fields;
# generic defaults contain no product-domain product names.
PROGRAM_CONTRACT_PROFILE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/program-contract-profile@1"
)
PROGRAM_CONTRACT_OPERATION_MATRIX_SCHEMA: Final[str] = (
    "program-contract/canonical-operation-matrix@1"
)
PROGRAM_CONTRACT_PROFILE_VERSION: Final[str] = "program-contract-profile/v1"

# Closed generic bounds for fail-closed construction.
MAX_TEXT_BYTES: Final[int] = 64 * 1024
MAX_COLLECTION_ITEMS: Final[int] = 4096
MAX_IDENTIFIER_BYTES: Final[int] = 512

_DEFAULT_DATA_MODES: Final[tuple[str, ...]] = (
    "bytes",
    "text",
    "metadata",
    "handle",
    "none",
)
_DEFAULT_EXECUTION_MODES: Final[tuple[str, ...]] = ("sync", "async")


class ProgramContractProfileError(ValueError):
    """Raised when a profile could silently lose, invent, or unbound an expectation."""

    def __init__(self, message: str, *, reason_codes: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.reason_codes = tuple(str(code) for code in reason_codes if str(code))


class ExpectationState(str, Enum):
    """Resolution state of an expectation."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    CONFLICTING = "conflicting"


class OperationSupport(str, Enum):
    """An explicit operation entry in a public-surface matrix."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"
    CONFLICTING = "conflicting"

    @property
    def is_resolved(self) -> bool:
        return self in {OperationSupport.SUPPORTED, OperationSupport.UNSUPPORTED}


class IssueKind(str, Enum):
    MISSING = "missing"
    CONFLICT = "conflict"


class DataMode(str, Enum):
    BYTES = "bytes"
    TEXT = "text"
    METADATA = "metadata"
    HANDLE = "handle"
    NONE = "none"


class ExecutionMode(str, Enum):
    SYNC = "sync"
    ASYNC = "async"


class FacadeCompatibility(str, Enum):
    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNRESOLVED = "unresolved"


def _require_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ProgramContractProfileError(
            f"{field_name} must be a non-empty string",
            reason_codes=("identifier_empty",),
        )
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_IDENTIFIER_BYTES:
        raise ProgramContractProfileError(
            f"{field_name} exceeds identifier bound of {MAX_IDENTIFIER_BYTES} bytes",
            reason_codes=("identifier_unbounded",),
        )


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ProgramContractProfileError(
            f"{field_name} must be a non-empty string",
            reason_codes=("text_empty",),
        )
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_TEXT_BYTES:
        raise ProgramContractProfileError(
            f"{field_name} exceeds text bound of {MAX_TEXT_BYTES} bytes",
            reason_codes=("text_unbounded",),
        )


def _require_unique(values: Iterable[str], field_name: str) -> None:
    items = tuple(values)
    if len(items) > MAX_COLLECTION_ITEMS:
        raise ProgramContractProfileError(
            f"{field_name} exceeds collection bound of {MAX_COLLECTION_ITEMS}",
            reason_codes=("collection_unbounded",),
        )
    if len(items) != len(set(items)):
        raise ProgramContractProfileError(
            f"{field_name} must be unique",
            reason_codes=("duplicate_entry",),
        )


def _require_bounded_collection(values: Sequence[Any], field_name: str) -> None:
    if len(values) > MAX_COLLECTION_ITEMS:
        raise ProgramContractProfileError(
            f"{field_name} exceeds collection bound of {MAX_COLLECTION_ITEMS}",
            reason_codes=("collection_unbounded",),
        )


def _enum_values(values: Sequence[Enum]) -> list[str]:
    return [value.value for value in values]


def _json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 32:
        raise ProgramContractProfileError(
            "JSON value nesting exceeds bound",
            reason_codes=("json_unbounded",),
        )
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise ProgramContractProfileError(
                "JSON mapping exceeds collection bound",
                reason_codes=("collection_unbounded",),
            )
        return {
            str(key): _json_value(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise ProgramContractProfileError(
                "JSON sequence exceeds collection bound",
                reason_codes=("collection_unbounded",),
            )
        return [_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ProgramContractProfileError(
                "JSON string exceeds text bound",
                reason_codes=("text_unbounded",),
            )
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    raise ProgramContractProfileError(
        f"contract values must be JSON-compatible, got {type(value).__name__}",
        reason_codes=("json_incompatible",),
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _content_id(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _member_of(value: str, vocabulary: Sequence[str], field_name: str) -> None:
    if value not in vocabulary:
        raise ProgramContractProfileError(
            f"{field_name} {value!r} is not in the closed vocabulary",
            reason_codes=("unknown_vocabulary_entry",),
        )


@dataclass(frozen=True)
class ContractVocabulary:
    """Immutable closed vocabularies for a domain profile.

    Semantic identity is the ordered content of these vocabularies and the
    content-addressed :meth:`identity`.  Module paths are never part of that
    identity.
    """

    operations: tuple[str, ...]
    invariant_kinds: tuple[str, ...]
    error_codes: tuple[str, ...]
    surfaces: tuple[str, ...]
    data_modes: tuple[str, ...] = _DEFAULT_DATA_MODES
    execution_modes: tuple[str, ...] = _DEFAULT_EXECUTION_MODES
    max_text_bytes: int = MAX_TEXT_BYTES
    max_collection_items: int = MAX_COLLECTION_ITEMS

    def __post_init__(self) -> None:
        for field_name, values in (
            ("operations", self.operations),
            ("invariant_kinds", self.invariant_kinds),
            ("error_codes", self.error_codes),
            ("surfaces", self.surfaces),
            ("data_modes", self.data_modes),
            ("execution_modes", self.execution_modes),
        ):
            _require_bounded_collection(values, field_name)
            if not values:
                raise ProgramContractProfileError(
                    f"vocabulary.{field_name} must be non-empty",
                    reason_codes=("vocabulary_empty",),
                )
            _require_unique(values, f"vocabulary.{field_name}")
            for item in values:
                _require_identifier(item, f"vocabulary.{field_name} entry")
        if self.max_text_bytes <= 0 or self.max_collection_items <= 0:
            raise ProgramContractProfileError(
                "vocabulary bounds must be positive",
                reason_codes=("vocabulary_bounds_invalid",),
            )
        if self.max_text_bytes > MAX_TEXT_BYTES:
            raise ProgramContractProfileError(
                "vocabulary.max_text_bytes exceeds engine bound",
                reason_codes=("vocabulary_bounds_invalid",),
            )
        if self.max_collection_items > MAX_COLLECTION_ITEMS:
            raise ProgramContractProfileError(
                "vocabulary.max_collection_items exceeds engine bound",
                reason_codes=("vocabulary_bounds_invalid",),
            )

    def identity(self) -> str:
        return _content_id(
            {
                "operations": list(self.operations),
                "invariant_kinds": list(self.invariant_kinds),
                "error_codes": list(self.error_codes),
                "surfaces": list(self.surfaces),
                "data_modes": list(self.data_modes),
                "execution_modes": list(self.execution_modes),
                "max_text_bytes": self.max_text_bytes,
                "max_collection_items": self.max_collection_items,
            }
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "operations": list(self.operations),
            "invariant_kinds": list(self.invariant_kinds),
            "error_codes": list(self.error_codes),
            "surfaces": list(self.surfaces),
            "data_modes": list(self.data_modes),
            "execution_modes": list(self.execution_modes),
            "max_text_bytes": self.max_text_bytes,
            "max_collection_items": self.max_collection_items,
            "vocabulary_identity": self.identity(),
        }


@dataclass(frozen=True)
class SourceContract:
    """Provenance for an expectation, with authority stated explicitly."""

    source_id: str
    kind: ContractSourceKind
    locator: str
    revision: str
    summary: str
    reviewed: bool
    available: bool = True
    expectation_authority: bool = True

    def __post_init__(self) -> None:
        _require_identifier(self.source_id, "source_id")
        _require_identifier(self.locator, "locator")
        _require_identifier(self.revision, "revision")
        _require_text(self.summary, "summary")
        if (
            type(self.reviewed) is not bool
            or type(self.available) is not bool
            or type(self.expectation_authority) is not bool
        ):
            raise ProgramContractProfileError(
                "source authority flags must be booleans",
                reason_codes=("authority_type",),
            )
        # Self-authority is forbidden: a source cannot claim identity from the
        # profile module or a self-referential locator.
        lowered_locator = self.locator.strip().lower()
        if lowered_locator in {"self", "self://", "self://authority"} or (
            lowered_locator.startswith("self:")
        ):
            raise ProgramContractProfileError(
                f"source {self.source_id!r} cannot use self-authority",
                reason_codes=("self_authority",),
            )
        if self.source_id.strip().lower() in {"self", "profile", "this"}:
            raise ProgramContractProfileError(
                f"source_id {self.source_id!r} is reserved (self-authority)",
                reason_codes=("self_authority",),
            )
        if self.expectation_authority:
            if not self.available or not self.reviewed:
                raise ProgramContractProfileError(
                    f"authoritative source {self.source_id!r} must be available and reviewed",
                    reason_codes=("authority_unavailable",),
                )
            if not self.kind.may_define_expectation:
                raise ProgramContractProfileError(
                    f"observation-only source {self.source_id!r} cannot define expectations",
                    reason_codes=("observation_authority",),
                )

    @property
    def precedence(self) -> int:
        return self.kind.rank

    def to_record(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "kind": self.kind.value,
            "precedence": self.precedence,
            "locator": self.locator,
            "revision": self.revision,
            "summary": self.summary,
            "reviewed": self.reviewed,
            "available": self.available,
            "expectation_authority": self.expectation_authority,
        }


@dataclass(frozen=True)
class InvariantContract:
    invariant_id: str
    kind: str
    statement: str
    applies_to: tuple[str, ...]
    source_contract_ids: tuple[str, ...]
    state: ExpectationState = ExpectationState.RESOLVED
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    error_codes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.invariant_id, "invariant_id")
        _require_identifier(self.kind, "kind")
        _require_text(self.statement, "statement")
        _require_unique(self.applies_to, "applies_to")
        _require_unique(self.source_contract_ids, "source_contract_ids")
        _require_unique(self.error_codes, "error_codes")
        _require_bounded_collection(self.preconditions, "preconditions")
        _require_bounded_collection(self.postconditions, "postconditions")
        _require_bounded_collection(self.notes, "notes")
        if not self.applies_to:
            raise ProgramContractProfileError(
                f"invariant {self.invariant_id!r} must apply to an operation",
                reason_codes=("invariant_empty_applies_to",),
            )
        if self.state is ExpectationState.RESOLVED and not self.source_contract_ids:
            raise ProgramContractProfileError(
                f"resolved invariant {self.invariant_id!r} needs a source contract",
                reason_codes=("resolved_without_source",),
            )
        if (
            self.state is ExpectationState.CONFLICTING
            and len(self.source_contract_ids) < 2
        ):
            raise ProgramContractProfileError(
                f"conflicting invariant {self.invariant_id!r} needs two sources",
                reason_codes=("conflict_needs_sources",),
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "kind": self.kind,
            "statement": self.statement,
            "applies_to": list(self.applies_to),
            "source_contract_ids": list(self.source_contract_ids),
            "state": self.state.value,
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "error_codes": list(self.error_codes),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class OperationContract:
    operation: str
    summary: str
    input_modes: tuple[DataMode, ...]
    output_modes: tuple[DataMode, ...]
    execution_modes: tuple[ExecutionMode, ...]
    invariant_ids: tuple[str, ...]
    error_codes: tuple[str, ...]
    source_contract_ids: tuple[str, ...]
    mutates: bool
    idempotent: bool | None
    state: ExpectationState = ExpectationState.RESOLVED

    def __post_init__(self) -> None:
        _require_identifier(self.operation, "operation")
        _require_text(self.summary, "summary")
        _require_unique((item.value for item in self.input_modes), "input_modes")
        _require_unique((item.value for item in self.output_modes), "output_modes")
        _require_unique(
            (item.value for item in self.execution_modes), "execution_modes"
        )
        _require_unique(self.invariant_ids, "invariant_ids")
        _require_unique(self.error_codes, "error_codes")
        _require_unique(self.source_contract_ids, "source_contract_ids")
        if type(self.mutates) is not bool:
            raise ProgramContractProfileError(
                "mutates must be a boolean",
                reason_codes=("mutates_type",),
            )
        if self.idempotent is not None and type(self.idempotent) is not bool:
            raise ProgramContractProfileError(
                "idempotent must be a boolean or null",
                reason_codes=("idempotent_type",),
            )
        if not self.execution_modes:
            raise ProgramContractProfileError(
                f"operation {self.operation!r} needs an execution mode",
                reason_codes=("execution_mode_required",),
            )
        if self.state is ExpectationState.RESOLVED and not self.source_contract_ids:
            raise ProgramContractProfileError(
                f"resolved operation {self.operation!r} needs a source contract",
                reason_codes=("resolved_without_source",),
            )
        if (
            self.state is ExpectationState.CONFLICTING
            and len(self.source_contract_ids) < 2
        ):
            raise ProgramContractProfileError(
                f"conflicting operation {self.operation!r} needs two sources",
                reason_codes=("conflict_needs_sources",),
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "summary": self.summary,
            "input_modes": _enum_values(self.input_modes),
            "output_modes": _enum_values(self.output_modes),
            "execution_modes": _enum_values(self.execution_modes),
            "invariant_ids": list(self.invariant_ids),
            "error_codes": list(self.error_codes),
            "source_contract_ids": list(self.source_contract_ids),
            "mutates": self.mutates,
            "idempotent": self.idempotent,
            "state": self.state.value,
        }


@dataclass(frozen=True)
class SurfaceOperationContract:
    operation: str
    support: OperationSupport
    source_contract_ids: tuple[str, ...]
    entrypoint: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        _require_identifier(self.operation, "operation")
        _require_unique(self.source_contract_ids, "source_contract_ids")
        if self.support.is_resolved and not self.source_contract_ids:
            raise ProgramContractProfileError(
                f"resolved surface operation {self.operation!r} needs a source",
                reason_codes=("resolved_without_source",),
            )
        if (
            self.support is OperationSupport.CONFLICTING
            and len(self.source_contract_ids) < 2
        ):
            raise ProgramContractProfileError(
                f"conflicting surface operation {self.operation!r} needs two sources",
                reason_codes=("conflict_needs_sources",),
            )
        if self.entrypoint is not None:
            _require_identifier(self.entrypoint, "entrypoint")
        if self.note:
            _require_text(self.note if self.note.strip() else "x", "note")
            if len(self.note.encode("utf-8")) > MAX_TEXT_BYTES:
                raise ProgramContractProfileError(
                    "note exceeds text bound",
                    reason_codes=("text_unbounded",),
                )

    def to_record(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "support": self.support.value,
            "source_contract_ids": list(self.source_contract_ids),
            "entrypoint": self.entrypoint,
            "note": self.note,
        }


@dataclass(frozen=True)
class PublicSurfaceContract:
    surface: str
    contract_name: str
    execution_modes: tuple[ExecutionMode, ...]
    operations: tuple[SurfaceOperationContract, ...]
    source_contract_ids: tuple[str, ...]
    transport_error_mapping_required: bool = True

    def __post_init__(self) -> None:
        _require_identifier(self.surface, "surface")
        _require_identifier(self.contract_name, "contract_name")
        _require_unique(
            (item.value for item in self.execution_modes), "execution_modes"
        )
        _require_unique(
            (item.operation for item in self.operations),
            f"{self.surface} operations",
        )
        _require_unique(self.source_contract_ids, "source_contract_ids")
        if not self.execution_modes:
            raise ProgramContractProfileError(
                f"{self.surface} needs an execution mode",
                reason_codes=("execution_mode_required",),
            )
        if not self.source_contract_ids:
            raise ProgramContractProfileError(
                f"{self.surface} needs a source contract",
                reason_codes=("resolved_without_source",),
            )
        if type(self.transport_error_mapping_required) is not bool:
            raise ProgramContractProfileError(
                "transport_error_mapping_required must be a boolean",
                reason_codes=("transport_flag_type",),
            )

    @property
    def supported_operations(self) -> tuple[str, ...]:
        return tuple(
            item.operation
            for item in self.operations
            if item.support is OperationSupport.SUPPORTED
        )

    @property
    def unresolved_operations(self) -> tuple[str, ...]:
        return tuple(
            item.operation
            for item in self.operations
            if item.support
            in {OperationSupport.UNRESOLVED, OperationSupport.CONFLICTING}
        )

    def support_for(self, operation: str) -> SurfaceOperationContract:
        for item in self.operations:
            if item.operation == operation:
                return item
        raise KeyError(operation)

    def to_record(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "contract_name": self.contract_name,
            "execution_modes": _enum_values(self.execution_modes),
            "operations": [item.to_record() for item in self.operations],
            "supported_operations": list(self.supported_operations),
            "unresolved_operations": list(self.unresolved_operations),
            "source_contract_ids": list(self.source_contract_ids),
            "transport_error_mapping_required": self.transport_error_mapping_required,
        }


@dataclass(frozen=True)
class ExpectationIssue:
    """A missing or conflicting expectation which must remain unresolved."""

    issue_id: str
    kind: IssueKind
    subject: str
    source_contract_ids: tuple[str, ...]
    positions: tuple[str, ...]
    state: ExpectationState
    resolution: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.issue_id, "issue_id")
        _require_text(self.subject, "subject")
        _require_unique(self.source_contract_ids, "source_contract_ids")
        _require_bounded_collection(self.positions, "positions")
        if self.kind is IssueKind.MISSING:
            if self.state is not ExpectationState.UNRESOLVED:
                raise ProgramContractProfileError(
                    "missing expectations must stay unresolved",
                    reason_codes=("missing_must_unresolved",),
                )
            if self.resolution is not None:
                raise ProgramContractProfileError(
                    "missing expectations cannot have a resolution",
                    reason_codes=("missing_has_resolution",),
                )
        else:
            if self.state is not ExpectationState.CONFLICTING:
                raise ProgramContractProfileError(
                    "conflicting expectations must stay conflicting",
                    reason_codes=("conflict_must_conflict",),
                )
            if len(self.source_contract_ids) < 2 or len(self.positions) < 2:
                raise ProgramContractProfileError(
                    "a conflict needs at least two sources and two positions",
                    reason_codes=("conflict_needs_sources",),
                )
            if self.resolution is not None:
                raise ProgramContractProfileError(
                    "conflicting expectations cannot select a resolution",
                    reason_codes=("conflict_has_resolution",),
                )

    def to_record(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "kind": self.kind.value,
            "subject": self.subject,
            "source_contract_ids": list(self.source_contract_ids),
            "positions": list(self.positions),
            "state": self.state.value,
            "resolution": self.resolution,
        }


@dataclass(frozen=True)
class CanonicalVector:
    vector_id: str
    operation: str
    description: str
    request: Mapping[str, Any]
    expected: Mapping[str, Any]
    invariant_ids: tuple[str, ...]
    source_contract_ids: tuple[str, ...]
    state: ExpectationState = ExpectationState.RESOLVED
    # Exact semantic tag: free-form exactness note required for resolved vectors.
    exact_semantics: str = ""

    def __post_init__(self) -> None:
        _require_identifier(self.vector_id, "vector_id")
        _require_identifier(self.operation, "operation")
        _require_text(self.description, "description")
        _json_value(self.request)
        _json_value(self.expected)
        _require_unique(self.invariant_ids, "invariant_ids")
        _require_unique(self.source_contract_ids, "source_contract_ids")
        if self.state is ExpectationState.RESOLVED:
            if not self.source_contract_ids:
                raise ProgramContractProfileError(
                    f"resolved vector {self.vector_id!r} needs a source contract",
                    reason_codes=("resolved_without_source",),
                )
            if not self.exact_semantics or not self.exact_semantics.strip():
                raise ProgramContractProfileError(
                    f"resolved vector {self.vector_id!r} needs exact semantics",
                    reason_codes=("vector_missing_semantics",),
                )
            _require_text(self.exact_semantics, "exact_semantics")
        if (
            self.state is ExpectationState.CONFLICTING
            and len(self.source_contract_ids) < 2
        ):
            raise ProgramContractProfileError(
                f"conflicting vector {self.vector_id!r} needs two sources",
                reason_codes=("conflict_needs_sources",),
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "operation": self.operation,
            "description": self.description,
            "request": _json_value(self.request),
            "expected": _json_value(self.expected),
            "invariant_ids": list(self.invariant_ids),
            "source_contract_ids": list(self.source_contract_ids),
            "state": self.state.value,
            "exact_semantics": self.exact_semantics,
        }


@dataclass(frozen=True)
class FacadeExample:
    example_id: str
    surface: str
    compatibility: FacadeCompatibility
    description: str
    operation: str
    example: Mapping[str, Any]
    rationale: str
    source_contract_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.example_id, "example_id")
        _require_identifier(self.surface, "surface")
        _require_identifier(self.operation, "operation")
        _require_text(self.description, "description")
        _require_text(self.rationale, "rationale")
        _json_value(self.example)
        _require_unique(self.source_contract_ids, "source_contract_ids")
        if (
            self.compatibility is not FacadeCompatibility.UNRESOLVED
            and not self.source_contract_ids
        ):
            raise ProgramContractProfileError(
                f"classified example {self.example_id!r} needs a source contract",
                reason_codes=("resolved_without_source",),
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "surface": self.surface,
            "compatibility": self.compatibility.value,
            "description": self.description,
            "operation": self.operation,
            "example": _json_value(self.example),
            "rationale": self.rationale,
            "source_contract_ids": list(self.source_contract_ids),
        }


@dataclass(frozen=True)
class ProgramContractProfile:
    """Immutable, content-identified evidence-bound program contract profile."""

    vocabulary: ContractVocabulary
    sources: tuple[SourceContract, ...]
    invariants: tuple[InvariantContract, ...]
    operations: tuple[OperationContract, ...]
    surfaces: tuple[PublicSurfaceContract, ...]
    issues: tuple[ExpectationIssue, ...]
    vectors: tuple[CanonicalVector, ...]
    facade_examples: tuple[FacadeExample, ...]
    schema: str = PROGRAM_CONTRACT_PROFILE_SCHEMA
    operation_matrix_schema: str = PROGRAM_CONTRACT_OPERATION_MATRIX_SCHEMA
    contract_version: str = PROGRAM_CONTRACT_PROFILE_VERSION
    goal_id: str = ""
    profile_id: str = ""

    def __post_init__(self) -> None:
        _require_identifier(self.schema, "schema")
        _require_identifier(self.operation_matrix_schema, "operation_matrix_schema")
        _require_identifier(self.contract_version, "contract_version")
        if self.goal_id:
            _require_identifier(self.goal_id, "goal_id")
        if self.profile_id:
            _require_identifier(self.profile_id, "profile_id")

        # Schema identity is content of the profile, not a module path.
        if "/" in self.schema and self.schema.endswith(".py"):
            raise ProgramContractProfileError(
                "schema must not be a module path",
                reason_codes=("schema_module_path",),
            )
        if self.schema.endswith(".py") or self.operation_matrix_schema.endswith(".py"):
            raise ProgramContractProfileError(
                "schema identity must not be a module path",
                reason_codes=("schema_module_path",),
            )

        _require_unique((item.source_id for item in self.sources), "source ids")
        _require_unique(
            (item.invariant_id for item in self.invariants), "invariant ids"
        )
        _require_unique((item.operation for item in self.operations), "operations")
        _require_unique((item.surface for item in self.surfaces), "surfaces")
        _require_unique((item.issue_id for item in self.issues), "issue ids")
        _require_unique((item.vector_id for item in self.vectors), "vector ids")
        _require_unique(
            (item.example_id for item in self.facade_examples), "example ids"
        )

        vocab = self.vocabulary
        op_set = set(vocab.operations)
        inv_kinds = set(vocab.invariant_kinds)
        err_set = set(vocab.error_codes)
        surface_set = set(vocab.surfaces)
        data_mode_values = set(vocab.data_modes)
        exec_mode_values = set(vocab.execution_modes)

        if {item.operation for item in self.operations} != op_set:
            missing = sorted(op_set - {item.operation for item in self.operations})
            extra = sorted({item.operation for item in self.operations} - op_set)
            raise ProgramContractProfileError(
                f"operation contracts must cover the vocabulary; "
                f"missing={missing}, extra={extra}",
                reason_codes=("operation_coverage",),
            )
        if {item.kind for item in self.invariants} != inv_kinds:
            missing = sorted(inv_kinds - {item.kind for item in self.invariants})
            extra = sorted({item.kind for item in self.invariants} - inv_kinds)
            raise ProgramContractProfileError(
                f"invariants must cover the vocabulary; missing={missing}, extra={extra}",
                reason_codes=("invariant_coverage",),
            )
        if {item.surface for item in self.surfaces} != surface_set:
            missing = sorted(surface_set - {item.surface for item in self.surfaces})
            extra = sorted({item.surface for item in self.surfaces} - surface_set)
            raise ProgramContractProfileError(
                f"surface contracts must cover the vocabulary; "
                f"missing={missing}, extra={extra}",
                reason_codes=("surface_coverage",),
            )

        for surface in self.surfaces:
            found = {item.operation for item in surface.operations}
            if found != op_set:
                missing = sorted(op_set - found)
                extra = sorted(found - op_set)
                raise ProgramContractProfileError(
                    f"{surface.surface} operation matrix is incomplete; "
                    f"missing={missing}, extra={extra}",
                    reason_codes=("surface_matrix_incomplete",),
                )

        for item in self.invariants:
            for operation in item.applies_to:
                _member_of(operation, vocab.operations, "invariant.applies_to")
            for error in item.error_codes:
                _member_of(error, vocab.error_codes, "invariant.error_codes")
        for item in self.operations:
            _member_of(item.operation, vocab.operations, "operation")
            for mode in item.input_modes:
                if mode.value not in data_mode_values:
                    raise ProgramContractProfileError(
                        f"unknown input mode {mode.value!r}",
                        reason_codes=("unknown_vocabulary_entry",),
                    )
            for mode in item.output_modes:
                if mode.value not in data_mode_values:
                    raise ProgramContractProfileError(
                        f"unknown output mode {mode.value!r}",
                        reason_codes=("unknown_vocabulary_entry",),
                    )
            for mode in item.execution_modes:
                if mode.value not in exec_mode_values:
                    raise ProgramContractProfileError(
                        f"unknown execution mode {mode.value!r}",
                        reason_codes=("unknown_vocabulary_entry",),
                    )
            for error in item.error_codes:
                _member_of(error, vocab.error_codes, "operation.error_codes")
        for surface in self.surfaces:
            _member_of(surface.surface, vocab.surfaces, "surface")
            for mode in surface.execution_modes:
                if mode.value not in exec_mode_values:
                    raise ProgramContractProfileError(
                        f"unknown execution mode {mode.value!r}",
                        reason_codes=("unknown_vocabulary_entry",),
                    )
            for binding in surface.operations:
                _member_of(binding.operation, vocab.operations, "surface.operation")
        for vector in self.vectors:
            _member_of(vector.operation, vocab.operations, "vector.operation")
        for example in self.facade_examples:
            _member_of(example.surface, vocab.surfaces, "facade.surface")
            _member_of(example.operation, vocab.operations, "facade.operation")

        self._validate_references_and_authority()

    def _validate_references_and_authority(self) -> None:
        sources = {item.source_id: item for item in self.sources}
        invariants = {item.invariant_id: item for item in self.invariants}

        def check_sources(
            owner: str, source_ids: Sequence[str], resolved: bool
        ) -> None:
            unknown = sorted(set(source_ids) - sources.keys())
            if unknown:
                raise ProgramContractProfileError(
                    f"{owner} references unknown source contracts: {unknown}",
                    reason_codes=("unknown_source",),
                )
            if resolved and not any(
                sources[source_id].expectation_authority for source_id in source_ids
            ):
                raise ProgramContractProfileError(
                    f"{owner} resolves an expectation without reviewed authority",
                    reason_codes=("resolved_without_authority",),
                )

        for item in self.invariants:
            check_sources(
                item.invariant_id,
                item.source_contract_ids,
                item.state is ExpectationState.RESOLVED,
            )
        for item in self.operations:
            unknown = sorted(set(item.invariant_ids) - invariants.keys())
            if unknown:
                raise ProgramContractProfileError(
                    f"operation {item.operation} has unknown invariants: {unknown}",
                    reason_codes=("unknown_invariant",),
                )
            applicable = {
                invariant.invariant_id
                for invariant in self.invariants
                if item.operation in invariant.applies_to
            }
            if set(item.invariant_ids) != applicable:
                missing = sorted(applicable - set(item.invariant_ids))
                extra = sorted(set(item.invariant_ids) - applicable)
                raise ProgramContractProfileError(
                    f"operation {item.operation} invariant coverage differs; "
                    f"missing={missing}, extra={extra}",
                    reason_codes=("operation_invariant_coverage",),
                )
            inherited_errors = {
                error
                for invariant_id in item.invariant_ids
                for error in invariants[invariant_id].error_codes
            }
            if not inherited_errors.issubset(item.error_codes):
                missing_errors = sorted(inherited_errors - set(item.error_codes))
                raise ProgramContractProfileError(
                    f"operation {item.operation} omits invariant errors: {missing_errors}",
                    reason_codes=("operation_error_coverage",),
                )
            check_sources(
                item.operation,
                item.source_contract_ids,
                item.state is ExpectationState.RESOLVED,
            )
        for surface in self.surfaces:
            check_sources(surface.surface, surface.source_contract_ids, False)
            for binding in surface.operations:
                check_sources(
                    f"{surface.surface}:{binding.operation}",
                    binding.source_contract_ids,
                    binding.support.is_resolved,
                )
        for issue in self.issues:
            check_sources(issue.issue_id, issue.source_contract_ids, False)
        for vector in self.vectors:
            unknown = sorted(set(vector.invariant_ids) - invariants.keys())
            if unknown:
                raise ProgramContractProfileError(
                    f"vector {vector.vector_id} has unknown invariants: {unknown}",
                    reason_codes=("unknown_invariant",),
                )
            inapplicable = sorted(
                invariant_id
                for invariant_id in vector.invariant_ids
                if vector.operation not in invariants[invariant_id].applies_to
            )
            if inapplicable:
                raise ProgramContractProfileError(
                    f"vector {vector.vector_id} uses inapplicable invariants: "
                    f"{inapplicable}",
                    reason_codes=("inapplicable_invariant",),
                )
            check_sources(
                vector.vector_id,
                vector.source_contract_ids,
                vector.state is ExpectationState.RESOLVED,
            )
        for example in self.facade_examples:
            check_sources(
                example.example_id,
                example.source_contract_ids,
                example.compatibility is not FacadeCompatibility.UNRESOLVED,
            )

    def operation_contract(self, operation: str) -> OperationContract:
        for item in self.operations:
            if item.operation == operation:
                return item
        raise KeyError(operation)

    def invariant_contract(self, kind: str) -> InvariantContract:
        for item in self.invariants:
            if item.kind == kind:
                return item
        raise KeyError(kind)

    def surface_contract(self, surface: str) -> PublicSurfaceContract:
        for item in self.surfaces:
            if item.surface == surface:
                return item
        raise KeyError(surface)

    @property
    def unresolved_expectations(self) -> tuple[ExpectationIssue, ...]:
        return tuple(
            item
            for item in self.issues
            if item.state
            in {ExpectationState.UNRESOLVED, ExpectationState.CONFLICTING}
        )

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema": self.schema,
            "operation_matrix_schema": self.operation_matrix_schema,
            "contract_version": self.contract_version,
            "goal_id": self.goal_id,
            "profile_id": self.profile_id,
            "vocabulary": self.vocabulary.to_record(),
            "authority": {
                "completion_evidence": PROFILE_IS_COMPLETION_EVIDENCE,
                "correctness_evidence": PROFILE_IS_CORRECTNESS_EVIDENCE,
                "authorizes_repair": PROFILE_AUTHORIZES_REPAIR,
            },
            "source_precedence": [item.value for item in SOURCE_PRECEDENCE],
            "sources": [item.to_record() for item in self.sources],
            "invariants": [item.to_record() for item in self.invariants],
            "operations": [item.to_record() for item in self.operations],
            "surfaces": [item.to_record() for item in self.surfaces],
            "issues": [item.to_record() for item in self.issues],
            "vectors": [item.to_record() for item in self.vectors],
            "facade_examples": [item.to_record() for item in self.facade_examples],
        }
        record["content_id"] = _content_id(record)
        return record

    @property
    def content_id(self) -> str:
        return self.to_record()["content_id"]

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_record(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        )


class ProgramContractProfileCompiler:
    """Compile vocabulary + evidence-bound records into a validated profile."""

    def __init__(
        self,
        *,
        schema: str = PROGRAM_CONTRACT_PROFILE_SCHEMA,
        operation_matrix_schema: str = PROGRAM_CONTRACT_OPERATION_MATRIX_SCHEMA,
        contract_version: str = PROGRAM_CONTRACT_PROFILE_VERSION,
        goal_id: str = "",
        profile_id: str = "",
    ) -> None:
        self.schema = schema
        self.operation_matrix_schema = operation_matrix_schema
        self.contract_version = contract_version
        self.goal_id = goal_id
        self.profile_id = profile_id

    def compile(
        self,
        vocabulary: ContractVocabulary,
        *,
        sources: Sequence[SourceContract],
        invariants: Sequence[InvariantContract],
        operations: Sequence[OperationContract],
        surfaces: Sequence[PublicSurfaceContract],
        issues: Sequence[ExpectationIssue] = (),
        vectors: Sequence[CanonicalVector] = (),
        facade_examples: Sequence[FacadeExample] = (),
        expected_content_id: str | None = None,
    ) -> ProgramContractProfile:
        """Build a fail-closed profile.

        Parameters
        ----------
        expected_content_id:
            When provided, the compiled content id must match exactly.  This
            rejects forged identity claims without using module paths as
            semantic identity.
        """

        profile = ProgramContractProfile(
            vocabulary=vocabulary,
            sources=tuple(sources),
            invariants=tuple(invariants),
            operations=tuple(operations),
            surfaces=tuple(surfaces),
            issues=tuple(issues),
            vectors=tuple(vectors),
            facade_examples=tuple(facade_examples),
            schema=self.schema,
            operation_matrix_schema=self.operation_matrix_schema,
            contract_version=self.contract_version,
            goal_id=self.goal_id,
            profile_id=self.profile_id,
        )
        if expected_content_id is not None:
            actual = profile.content_id
            if actual != expected_content_id:
                raise ProgramContractProfileError(
                    f"forged or drifted content_id: expected {expected_content_id!r}, "
                    f"got {actual!r}",
                    reason_codes=("forged_content_id",),
                )
        return profile


def assert_contract_profile_complete(profile: ProgramContractProfile) -> None:
    """Re-run the fail-closed construction checks for an existing profile."""

    profile.__post_init__()


def publish_contract_profile(
    output_path: str | os.PathLike[str],
    profile: ProgramContractProfile,
) -> Path:
    """Atomically publish canonical JSON and return the resolved destination."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = profile.to_json(indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def discover_profile_schemas() -> tuple[str, ...]:
    """Return the neutral default schema identifiers (domain-free)."""

    return (
        PROGRAM_CONTRACT_PROFILE_SCHEMA,
        PROGRAM_CONTRACT_OPERATION_MATRIX_SCHEMA,
    )


__all__ = [
    "MAX_COLLECTION_ITEMS",
    "MAX_IDENTIFIER_BYTES",
    "MAX_TEXT_BYTES",
    "PROFILE_AUTHORIZES_REPAIR",
    "PROFILE_IS_COMPLETION_EVIDENCE",
    "PROFILE_IS_CORRECTNESS_EVIDENCE",
    "PROGRAM_CONTRACT_OPERATION_MATRIX_SCHEMA",
    "PROGRAM_CONTRACT_PROFILE_SCHEMA",
    "PROGRAM_CONTRACT_PROFILE_VERSION",
    "SOURCE_PRECEDENCE",
    "CanonicalVector",
    "ContractSourceKind",
    "ContractVocabulary",
    "DataMode",
    "ExecutionMode",
    "ExpectationIssue",
    "ExpectationState",
    "FacadeCompatibility",
    "FacadeExample",
    "InvariantContract",
    "IssueKind",
    "OperationContract",
    "OperationSupport",
    "ProgramContractProfile",
    "ProgramContractProfileCompiler",
    "ProgramContractProfileError",
    "PublicSurfaceContract",
    "SourceContract",
    "SurfaceOperationContract",
    "assert_contract_profile_complete",
    "discover_profile_schemas",
    "publish_contract_profile",
]
