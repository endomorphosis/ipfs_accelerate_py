"""Canonical, evidence-bound contract pack for the public VFS.

The contract pack is intentionally separate from implementation discovery.
It describes the operation and invariant vocabulary that public facades must
map to, records the source contract for every resolved expectation, and keeps
missing or conflicting expectations unresolved.

Nothing in this module is completion evidence and no record authorizes a
repair.  Consumers are expected to compare observations with this pack rather
than treating implementation behaviour as a source of new expectations.
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

from .program_contracts import ContractSourceKind, SOURCE_PRECEDENCE


VFS_CONTRACT_PACK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vfs-contract-pack@1"
)
VFS_CANONICAL_OPERATION_MATRIX_SCHEMA: Final[str] = (
    "vfs/canonical-operation-matrix@1"
)
VFS_CONTRACT_PACK_VERSION: Final[str] = "vfs-contract-pack/v1"
VFS_CONTRACT_PACK_GOAL_ID: Final[str] = "VFS-026"

# Authority bounds: this is a comparison contract, not proof of repository
# correctness or permission to select/modify an implementation.
CONTRACT_PACK_IS_COMPLETION_EVIDENCE: Final[bool] = False
CONTRACT_PACK_IS_CORRECTNESS_EVIDENCE: Final[bool] = False
CONTRACT_PACK_AUTHORIZES_REPAIR: Final[bool] = False


class VfsContractPackError(ValueError):
    """Raised when a pack could silently lose or invent an expectation."""


class VfsOperation(str, Enum):
    """Closed canonical operation vocabulary shared by every facade."""

    PATH_RESOLVE = "path.resolve"
    MOUNT = "mount"
    READ = "read"
    WRITE = "write"
    OPEN = "open"
    CLOSE = "close"
    SEEK = "seek"
    STAT = "stat"
    LIST = "list"
    MKDIR = "mkdir"
    REMOVE = "remove"
    RENAME = "rename"
    COPY = "copy"


class VfsInvariantKind(str, Enum):
    """Semantic dimensions which a conforming implementation must preserve."""

    VERSIONED_PATH = "versioned_path"
    UNICODE = "unicode"
    ROOT = "root"
    TRAVERSAL = "traversal"
    MOUNT = "mount"
    READ_WRITE = "read_write"
    HANDLE_LIFECYCLE = "handle_lifecycle"
    SEEK = "seek"
    STAT_LIST = "stat_list"
    DIRECTORY_MUTATION = "directory_mutation"
    NAMESPACE_MUTATION = "namespace_mutation"
    BYTES_TEXT = "bytes_text"
    SYNC_ASYNC = "sync_async"
    ERROR = "error"
    CID_SIZE = "cid_size"
    ATOMICITY = "atomicity"
    JOURNAL_REPLAY = "journal_replay"
    VERSIONING = "versioning"
    CACHE_PIN_COHERENCE = "cache_pin_coherence"
    BACKEND_NEGOTIATION = "backend_negotiation"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    DEGRADATION = "degradation"


class PublicSurface(str, Enum):
    """Required public facade families."""

    PYTHON = "python"
    CLI = "cli"
    MCP = "mcp"
    MCP_PLUS_PLUS = "mcp++"
    HTTP = "http"
    LIBP2P = "libp2p"


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


class VfsErrorCode(str, Enum):
    """Transport-neutral stable error taxonomy."""

    INVALID_ARGUMENT = "invalid_argument"
    INVALID_PATH = "invalid_path"
    TRAVERSAL_DENIED = "traversal_denied"
    NOT_FOUND = "not_found"
    ALREADY_EXISTS = "already_exists"
    NOT_A_FILE = "not_a_file"
    NOT_A_DIRECTORY = "not_a_directory"
    DIRECTORY_NOT_EMPTY = "directory_not_empty"
    PERMISSION_DENIED = "permission_denied"
    AUTHENTICATION_REQUIRED = "authentication_required"
    CONFLICT = "conflict"
    STALE_VERSION = "stale_version"
    UNSUPPORTED = "unsupported"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    INTEGRITY_FAILURE = "integrity_failure"
    IO_FAILURE = "io_failure"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"


def _require_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise VfsContractPackError(f"{field_name} must be a non-empty string")


def _require_unique(values: Iterable[str], field_name: str) -> None:
    items = tuple(values)
    if len(items) != len(set(items)):
        raise VfsContractPackError(f"{field_name} must be unique")


def _enum_values(values: Sequence[Enum]) -> list[str]:
    return [value.value for value in values]


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise VfsContractPackError(
        f"contract values must be JSON-compatible, got {type(value).__name__}"
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
        _require_identifier(self.summary, "summary")
        if (
            type(self.reviewed) is not bool
            or type(self.available) is not bool
            or type(self.expectation_authority) is not bool
        ):
            raise VfsContractPackError(
                "source authority flags must be booleans"
            )
        if self.expectation_authority:
            if not self.available or not self.reviewed:
                raise VfsContractPackError(
                    f"authoritative source {self.source_id!r} must be available and reviewed"
                )
            if not self.kind.may_define_expectation:
                raise VfsContractPackError(
                    f"observation-only source {self.source_id!r} cannot define expectations"
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
    kind: VfsInvariantKind
    statement: str
    applies_to: tuple[VfsOperation, ...]
    source_contract_ids: tuple[str, ...]
    state: ExpectationState = ExpectationState.RESOLVED
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    error_codes: tuple[VfsErrorCode, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.invariant_id, "invariant_id")
        _require_identifier(self.statement, "statement")
        _require_unique((item.value for item in self.applies_to), "applies_to")
        _require_unique(self.source_contract_ids, "source_contract_ids")
        _require_unique((item.value for item in self.error_codes), "error_codes")
        if not self.applies_to:
            raise VfsContractPackError(
                f"invariant {self.invariant_id!r} must apply to an operation"
            )
        if self.state is ExpectationState.RESOLVED and not self.source_contract_ids:
            raise VfsContractPackError(
                f"resolved invariant {self.invariant_id!r} needs a source contract"
            )
        if (
            self.state is ExpectationState.CONFLICTING
            and len(self.source_contract_ids) < 2
        ):
            raise VfsContractPackError(
                f"conflicting invariant {self.invariant_id!r} needs two sources"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "kind": self.kind.value,
            "statement": self.statement,
            "applies_to": _enum_values(self.applies_to),
            "source_contract_ids": list(self.source_contract_ids),
            "state": self.state.value,
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "error_codes": _enum_values(self.error_codes),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class OperationContract:
    operation: VfsOperation
    summary: str
    input_modes: tuple[DataMode, ...]
    output_modes: tuple[DataMode, ...]
    execution_modes: tuple[ExecutionMode, ...]
    invariant_ids: tuple[str, ...]
    error_codes: tuple[VfsErrorCode, ...]
    source_contract_ids: tuple[str, ...]
    mutates: bool
    idempotent: bool | None
    state: ExpectationState = ExpectationState.RESOLVED

    def __post_init__(self) -> None:
        _require_identifier(self.summary, "summary")
        _require_unique((item.value for item in self.input_modes), "input_modes")
        _require_unique((item.value for item in self.output_modes), "output_modes")
        _require_unique(
            (item.value for item in self.execution_modes), "execution_modes"
        )
        _require_unique(self.invariant_ids, "invariant_ids")
        _require_unique((item.value for item in self.error_codes), "error_codes")
        _require_unique(self.source_contract_ids, "source_contract_ids")
        if type(self.mutates) is not bool:
            raise VfsContractPackError("mutates must be a boolean")
        if self.idempotent is not None and type(self.idempotent) is not bool:
            raise VfsContractPackError("idempotent must be a boolean or null")
        if not self.execution_modes:
            raise VfsContractPackError(
                f"operation {self.operation.value!r} needs an execution mode"
            )
        if self.state is ExpectationState.RESOLVED and not self.source_contract_ids:
            raise VfsContractPackError(
                f"resolved operation {self.operation.value!r} needs a source contract"
            )
        if (
            self.state is ExpectationState.CONFLICTING
            and len(self.source_contract_ids) < 2
        ):
            raise VfsContractPackError(
                f"conflicting operation {self.operation.value!r} needs two sources"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "summary": self.summary,
            "input_modes": _enum_values(self.input_modes),
            "output_modes": _enum_values(self.output_modes),
            "execution_modes": _enum_values(self.execution_modes),
            "invariant_ids": list(self.invariant_ids),
            "error_codes": _enum_values(self.error_codes),
            "source_contract_ids": list(self.source_contract_ids),
            "mutates": self.mutates,
            "idempotent": self.idempotent,
            "state": self.state.value,
        }


@dataclass(frozen=True)
class SurfaceOperationContract:
    operation: VfsOperation
    support: OperationSupport
    source_contract_ids: tuple[str, ...]
    entrypoint: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        _require_unique(self.source_contract_ids, "source_contract_ids")
        if self.support.is_resolved and not self.source_contract_ids:
            raise VfsContractPackError(
                f"resolved surface operation {self.operation.value!r} needs a source"
            )
        if (
            self.support is OperationSupport.CONFLICTING
            and len(self.source_contract_ids) < 2
        ):
            raise VfsContractPackError(
                f"conflicting surface operation {self.operation.value!r} "
                "needs two sources"
            )
        if self.entrypoint is not None:
            _require_identifier(self.entrypoint, "entrypoint")

    def to_record(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "support": self.support.value,
            "source_contract_ids": list(self.source_contract_ids),
            "entrypoint": self.entrypoint,
            "note": self.note,
        }


@dataclass(frozen=True)
class PublicSurfaceContract:
    surface: PublicSurface
    contract_name: str
    execution_modes: tuple[ExecutionMode, ...]
    operations: tuple[SurfaceOperationContract, ...]
    source_contract_ids: tuple[str, ...]
    transport_error_mapping_required: bool = True

    def __post_init__(self) -> None:
        _require_identifier(self.contract_name, "contract_name")
        _require_unique(
            (item.value for item in self.execution_modes), "execution_modes"
        )
        _require_unique(
            (item.operation.value for item in self.operations),
            f"{self.surface.value} operations",
        )
        _require_unique(self.source_contract_ids, "source_contract_ids")
        if not self.execution_modes:
            raise VfsContractPackError(
                f"{self.surface.value} needs an execution mode"
            )
        if not self.source_contract_ids:
            raise VfsContractPackError(
                f"{self.surface.value} needs a source contract"
            )
        if type(self.transport_error_mapping_required) is not bool:
            raise VfsContractPackError(
                "transport_error_mapping_required must be a boolean"
            )
        found = {item.operation for item in self.operations}
        required = set(VfsOperation)
        if found != required:
            missing = sorted(item.value for item in required - found)
            extra = sorted(item.value for item in found - required)
            raise VfsContractPackError(
                f"{self.surface.value} operation matrix is incomplete; "
                f"missing={missing}, extra={extra}"
            )

    @property
    def supported_operations(self) -> tuple[VfsOperation, ...]:
        return tuple(
            item.operation
            for item in self.operations
            if item.support is OperationSupport.SUPPORTED
        )

    @property
    def unresolved_operations(self) -> tuple[VfsOperation, ...]:
        return tuple(
            item.operation
            for item in self.operations
            if item.support
            in {OperationSupport.UNRESOLVED, OperationSupport.CONFLICTING}
        )

    def support_for(self, operation: VfsOperation) -> SurfaceOperationContract:
        for item in self.operations:
            if item.operation is operation:
                return item
        raise KeyError(operation.value)

    def to_record(self) -> dict[str, Any]:
        return {
            "surface": self.surface.value,
            "contract_name": self.contract_name,
            "execution_modes": _enum_values(self.execution_modes),
            "operations": [item.to_record() for item in self.operations],
            "supported_operations": _enum_values(self.supported_operations),
            "unresolved_operations": _enum_values(self.unresolved_operations),
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
        _require_identifier(self.subject, "subject")
        _require_unique(self.source_contract_ids, "source_contract_ids")
        if self.kind is IssueKind.MISSING:
            if self.state is not ExpectationState.UNRESOLVED:
                raise VfsContractPackError("missing expectations must stay unresolved")
            if self.resolution is not None:
                raise VfsContractPackError("missing expectations cannot have a resolution")
        else:
            if self.state is not ExpectationState.CONFLICTING:
                raise VfsContractPackError(
                    "conflicting expectations must stay conflicting"
                )
            if len(self.source_contract_ids) < 2 or len(self.positions) < 2:
                raise VfsContractPackError(
                    "a conflict needs at least two sources and two positions"
                )
            if self.resolution is not None:
                raise VfsContractPackError(
                    "conflicting expectations cannot select a resolution"
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
    operation: VfsOperation
    description: str
    request: Mapping[str, Any]
    expected: Mapping[str, Any]
    invariant_ids: tuple[str, ...]
    source_contract_ids: tuple[str, ...]
    state: ExpectationState = ExpectationState.RESOLVED

    def __post_init__(self) -> None:
        _require_identifier(self.vector_id, "vector_id")
        _require_identifier(self.description, "description")
        _json_value(self.request)
        _json_value(self.expected)
        _require_unique(self.invariant_ids, "invariant_ids")
        _require_unique(self.source_contract_ids, "source_contract_ids")
        if self.state is ExpectationState.RESOLVED and not self.source_contract_ids:
            raise VfsContractPackError(
                f"resolved vector {self.vector_id!r} needs a source contract"
            )
        if (
            self.state is ExpectationState.CONFLICTING
            and len(self.source_contract_ids) < 2
        ):
            raise VfsContractPackError(
                f"conflicting vector {self.vector_id!r} needs two sources"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "operation": self.operation.value,
            "description": self.description,
            "request": _json_value(self.request),
            "expected": _json_value(self.expected),
            "invariant_ids": list(self.invariant_ids),
            "source_contract_ids": list(self.source_contract_ids),
            "state": self.state.value,
        }


@dataclass(frozen=True)
class FacadeExample:
    example_id: str
    surface: PublicSurface
    compatibility: FacadeCompatibility
    description: str
    operation: VfsOperation
    example: Mapping[str, Any]
    rationale: str
    source_contract_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.example_id, "example_id")
        _require_identifier(self.description, "description")
        _require_identifier(self.rationale, "rationale")
        _json_value(self.example)
        _require_unique(self.source_contract_ids, "source_contract_ids")
        if (
            self.compatibility is not FacadeCompatibility.UNRESOLVED
            and not self.source_contract_ids
        ):
            raise VfsContractPackError(
                f"classified example {self.example_id!r} needs a source contract"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "surface": self.surface.value,
            "compatibility": self.compatibility.value,
            "description": self.description,
            "operation": self.operation.value,
            "example": _json_value(self.example),
            "rationale": self.rationale,
            "source_contract_ids": list(self.source_contract_ids),
        }


@dataclass(frozen=True)
class VfsContractPack:
    sources: tuple[SourceContract, ...]
    invariants: tuple[InvariantContract, ...]
    operations: tuple[OperationContract, ...]
    surfaces: tuple[PublicSurfaceContract, ...]
    issues: tuple[ExpectationIssue, ...]
    vectors: tuple[CanonicalVector, ...]
    facade_examples: tuple[FacadeExample, ...]
    schema: str = VFS_CONTRACT_PACK_SCHEMA
    operation_matrix_schema: str = VFS_CANONICAL_OPERATION_MATRIX_SCHEMA
    contract_version: str = VFS_CONTRACT_PACK_VERSION
    goal_id: str = VFS_CONTRACT_PACK_GOAL_ID

    def __post_init__(self) -> None:
        expected_identity = (
            (self.schema, VFS_CONTRACT_PACK_SCHEMA, "schema"),
            (
                self.operation_matrix_schema,
                VFS_CANONICAL_OPERATION_MATRIX_SCHEMA,
                "operation_matrix_schema",
            ),
            (
                self.contract_version,
                VFS_CONTRACT_PACK_VERSION,
                "contract_version",
            ),
            (self.goal_id, VFS_CONTRACT_PACK_GOAL_ID, "goal_id"),
        )
        for actual, expected, field_name in expected_identity:
            if actual != expected:
                raise VfsContractPackError(
                    f"{field_name} must be {expected!r}, got {actual!r}"
                )
        _require_unique((item.source_id for item in self.sources), "source ids")
        _require_unique(
            (item.invariant_id for item in self.invariants), "invariant ids"
        )
        _require_unique(
            (item.operation.value for item in self.operations), "operations"
        )
        _require_unique((item.surface.value for item in self.surfaces), "surfaces")
        _require_unique((item.issue_id for item in self.issues), "issue ids")
        _require_unique((item.vector_id for item in self.vectors), "vector ids")
        _require_unique(
            (item.example_id for item in self.facade_examples), "example ids"
        )
        if {item.operation for item in self.operations} != set(VfsOperation):
            raise VfsContractPackError(
                "operation contracts must cover the complete VfsOperation vocabulary"
            )
        if {item.kind for item in self.invariants} != set(VfsInvariantKind):
            raise VfsContractPackError(
                "invariants must cover the complete VfsInvariantKind vocabulary"
            )
        if {item.surface for item in self.surfaces} != set(PublicSurface):
            raise VfsContractPackError(
                "surface contracts must cover Python, CLI, MCP, MCP++, HTTP, and libp2p"
            )
        self._validate_references_and_authority()

    def _validate_references_and_authority(self) -> None:
        sources = {item.source_id: item for item in self.sources}
        invariants = {item.invariant_id: item for item in self.invariants}

        def check_sources(
            owner: str, source_ids: Sequence[str], resolved: bool
        ) -> None:
            unknown = sorted(set(source_ids) - sources.keys())
            if unknown:
                raise VfsContractPackError(
                    f"{owner} references unknown source contracts: {unknown}"
                )
            if resolved and not any(
                sources[source_id].expectation_authority
                for source_id in source_ids
            ):
                raise VfsContractPackError(
                    f"{owner} resolves an expectation without reviewed authority"
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
                raise VfsContractPackError(
                    f"operation {item.operation.value} has unknown invariants: {unknown}"
                )
            applicable = {
                invariant.invariant_id
                for invariant in self.invariants
                if item.operation in invariant.applies_to
            }
            if set(item.invariant_ids) != applicable:
                missing = sorted(applicable - set(item.invariant_ids))
                extra = sorted(set(item.invariant_ids) - applicable)
                raise VfsContractPackError(
                    f"operation {item.operation.value} invariant coverage differs; "
                    f"missing={missing}, extra={extra}"
                )
            inherited_errors = {
                error
                for invariant_id in item.invariant_ids
                for error in invariants[invariant_id].error_codes
            }
            if not inherited_errors.issubset(item.error_codes):
                missing_errors = sorted(
                    error.value for error in inherited_errors - set(item.error_codes)
                )
                raise VfsContractPackError(
                    f"operation {item.operation.value} omits invariant errors: "
                    f"{missing_errors}"
                )
            check_sources(
                item.operation.value,
                item.source_contract_ids,
                item.state is ExpectationState.RESOLVED,
            )
        for surface in self.surfaces:
            check_sources(surface.surface.value, surface.source_contract_ids, False)
            for binding in surface.operations:
                check_sources(
                    f"{surface.surface.value}:{binding.operation.value}",
                    binding.source_contract_ids,
                    binding.support.is_resolved,
                )
        for issue in self.issues:
            check_sources(issue.issue_id, issue.source_contract_ids, False)
        for vector in self.vectors:
            unknown = sorted(set(vector.invariant_ids) - invariants.keys())
            if unknown:
                raise VfsContractPackError(
                    f"vector {vector.vector_id} has unknown invariants: {unknown}"
                )
            inapplicable = sorted(
                invariant_id
                for invariant_id in vector.invariant_ids
                if vector.operation not in invariants[invariant_id].applies_to
            )
            if inapplicable:
                raise VfsContractPackError(
                    f"vector {vector.vector_id} uses inapplicable invariants: "
                    f"{inapplicable}"
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

    def operation_contract(self, operation: VfsOperation) -> OperationContract:
        for item in self.operations:
            if item.operation is operation:
                return item
        raise KeyError(operation.value)

    def invariant_contract(self, kind: VfsInvariantKind) -> InvariantContract:
        for item in self.invariants:
            if item.kind is kind:
                return item
        raise KeyError(kind.value)

    def surface_contract(self, surface: PublicSurface) -> PublicSurfaceContract:
        for item in self.surfaces:
            if item.surface is surface:
                return item
        raise KeyError(surface.value)

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
            "authority": {
                "completion_evidence": CONTRACT_PACK_IS_COMPLETION_EVIDENCE,
                "correctness_evidence": CONTRACT_PACK_IS_CORRECTNESS_EVIDENCE,
                "authorizes_repair": CONTRACT_PACK_AUTHORIZES_REPAIR,
            },
            "source_precedence": [item.value for item in SOURCE_PRECEDENCE],
            "sources": [item.to_record() for item in self.sources],
            "invariants": [item.to_record() for item in self.invariants],
            "operations": [item.to_record() for item in self.operations],
            "surfaces": [item.to_record() for item in self.surfaces],
            "issues": [item.to_record() for item in self.issues],
            "vectors": [item.to_record() for item in self.vectors],
            "facade_examples": [
                item.to_record() for item in self.facade_examples
            ],
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


_ALL = tuple(VfsOperation)
_PATH_OPS = (
    VfsOperation.PATH_RESOLVE,
    VfsOperation.MOUNT,
    VfsOperation.READ,
    VfsOperation.WRITE,
    VfsOperation.OPEN,
    VfsOperation.STAT,
    VfsOperation.LIST,
    VfsOperation.MKDIR,
    VfsOperation.REMOVE,
    VfsOperation.RENAME,
    VfsOperation.COPY,
)
_MUTATIONS = (
    VfsOperation.MOUNT,
    VfsOperation.WRITE,
    VfsOperation.MKDIR,
    VfsOperation.REMOVE,
    VfsOperation.RENAME,
    VfsOperation.COPY,
)
_CONTENT_OPS = (
    VfsOperation.READ,
    VfsOperation.WRITE,
    VfsOperation.OPEN,
    VfsOperation.STAT,
    VfsOperation.COPY,
)
_HANDLE_OPS = (VfsOperation.OPEN, VfsOperation.CLOSE, VfsOperation.SEEK)
_NAMESPACE_OPS = (
    VfsOperation.MKDIR,
    VfsOperation.REMOVE,
    VfsOperation.RENAME,
    VfsOperation.COPY,
)


def _invariant(
    kind: VfsInvariantKind,
    statement: str,
    applies_to: tuple[VfsOperation, ...],
    *,
    preconditions: tuple[str, ...] = (),
    postconditions: tuple[str, ...] = (),
    errors: tuple[VfsErrorCode, ...] = (),
) -> InvariantContract:
    return InvariantContract(
        invariant_id=f"invariant:{kind.value}",
        kind=kind,
        statement=statement,
        applies_to=applies_to,
        source_contract_ids=("source:vfs-026-acceptance",),
        preconditions=preconditions,
        postconditions=postconditions,
        error_codes=errors,
    )


def _canonical_invariants() -> tuple[InvariantContract, ...]:
    """Build the complete invariant contract without backend assumptions."""

    return (
        _invariant(
            VfsInvariantKind.VERSIONED_PATH,
            "A path is an absolute normalized path plus an optional immutable "
            "version selector; normalization never changes the selector.",
            _PATH_OPS,
            postconditions=("Equivalent path spellings resolve to one canonical path.",),
            errors=(VfsErrorCode.INVALID_PATH,),
        ),
        _invariant(
            VfsInvariantKind.UNICODE,
            "Path components and text payloads use valid Unicode normalized to NFC; "
            "invalid encodings fail and are never replaced silently.",
            _PATH_OPS,
            errors=(VfsErrorCode.INVALID_PATH, VfsErrorCode.INVALID_ARGUMENT),
        ),
        _invariant(
            VfsInvariantKind.ROOT,
            "The canonical root is '/', has no parent, and cannot be removed or renamed.",
            _PATH_OPS,
            errors=(VfsErrorCode.INVALID_PATH, VfsErrorCode.PERMISSION_DENIED),
        ),
        _invariant(
            VfsInvariantKind.TRAVERSAL,
            "Dot components are eliminated, repeated separators collapse, and '..' "
            "must never escape the selected mount root.",
            _PATH_OPS,
            errors=(VfsErrorCode.TRAVERSAL_DENIED,),
        ),
        _invariant(
            VfsInvariantKind.MOUNT,
            "Mount selection is component-boundary longest-prefix matching; cross-mount "
            "operations require an explicitly negotiated capability.",
            _PATH_OPS,
            errors=(
                VfsErrorCode.CAPABILITY_UNAVAILABLE,
                VfsErrorCode.UNSUPPORTED,
            ),
        ),
        _invariant(
            VfsInvariantKind.READ_WRITE,
            "Reads return exactly the selected byte range; successful writes persist "
            "exactly the supplied bytes and return their committed byte count.",
            (VfsOperation.READ, VfsOperation.WRITE),
            errors=(VfsErrorCode.IO_FAILURE, VfsErrorCode.INTEGRITY_FAILURE),
        ),
        _invariant(
            VfsInvariantKind.HANDLE_LIFECYCLE,
            "Open returns a scoped opaque handle, close invalidates it exactly once, "
            "and operations on an invalid handle fail explicitly.",
            _HANDLE_OPS + (VfsOperation.READ, VfsOperation.WRITE),
            errors=(VfsErrorCode.INVALID_ARGUMENT, VfsErrorCode.CONFLICT),
        ),
        _invariant(
            VfsInvariantKind.SEEK,
            "Seek computes a non-negative byte offset from start/current/end and returns "
            "the resulting offset without reading or writing content.",
            (VfsOperation.SEEK,),
            errors=(VfsErrorCode.INVALID_ARGUMENT,),
        ),
        _invariant(
            VfsInvariantKind.STAT_LIST,
            "Stat and list use the same typed metadata schema; list identifies its "
            "ordering/pagination and never invents absent children.",
            (VfsOperation.STAT, VfsOperation.LIST),
            errors=(VfsErrorCode.NOT_FOUND, VfsErrorCode.NOT_A_DIRECTORY),
        ),
        _invariant(
            VfsInvariantKind.DIRECTORY_MUTATION,
            "Mkdir states parent/recursive/existence policy; remove states recursive "
            "policy and refuses a non-empty directory unless recursive was requested.",
            (VfsOperation.MKDIR, VfsOperation.REMOVE),
            errors=(
                VfsErrorCode.ALREADY_EXISTS,
                VfsErrorCode.DIRECTORY_NOT_EMPTY,
            ),
        ),
        _invariant(
            VfsInvariantKind.NAMESPACE_MUTATION,
            "Remove, rename, and copy make source/destination/overwrite semantics "
            "explicit and never report success for a partial namespace mutation.",
            (VfsOperation.REMOVE, VfsOperation.RENAME, VfsOperation.COPY),
            errors=(
                VfsErrorCode.NOT_FOUND,
                VfsErrorCode.ALREADY_EXISTS,
                VfsErrorCode.CONFLICT,
            ),
        ),
        _invariant(
            VfsInvariantKind.BYTES_TEXT,
            "Bytes are canonical. Text is an explicit adapter requiring an encoding "
            "and error policy; byte counts, sizes, offsets, and CIDs refer to encoded bytes.",
            (VfsOperation.READ, VfsOperation.WRITE, VfsOperation.OPEN, VfsOperation.SEEK),
            errors=(VfsErrorCode.INVALID_ARGUMENT,),
        ),
        _invariant(
            VfsInvariantKind.SYNC_ASYNC,
            "Sync and async entrypoints have identical values, errors, effects, and "
            "cancellation boundaries; async entrypoints do not hide blocking I/O.",
            _ALL,
            errors=(VfsErrorCode.CANCELLED, VfsErrorCode.DEADLINE_EXCEEDED),
        ),
        _invariant(
            VfsInvariantKind.ERROR,
            "All facades preserve the stable VFS error code and structured details; "
            "transport status, exception type, or exit code is only a lossless mapping.",
            _ALL,
        ),
        _invariant(
            VfsInvariantKind.CID_SIZE,
            "When content is committed, size is its byte length and CID is computed from "
            "those exact bytes; unknown CID or size is explicit rather than fabricated.",
            _CONTENT_OPS,
            errors=(VfsErrorCode.INTEGRITY_FAILURE,),
        ),
        _invariant(
            VfsInvariantKind.ATOMICITY,
            "Atomicity is capability-bound. A successful atomic mutation is all-or-nothing; "
            "unsupported atomicity fails before effects rather than silently weakening.",
            _MUTATIONS,
            errors=(VfsErrorCode.CAPABILITY_UNAVAILABLE,),
        ),
        _invariant(
            VfsInvariantKind.JOURNAL_REPLAY,
            "A durable mutation is journaled before visibility; replay is ordered, "
            "idempotent by operation identity, and cannot duplicate committed effects.",
            _MUTATIONS,
            errors=(VfsErrorCode.INTEGRITY_FAILURE, VfsErrorCode.CONFLICT),
        ),
        _invariant(
            VfsInvariantKind.VERSIONING,
            "A version selector reads an immutable snapshot; mutations require an "
            "explicit base version and return a new version or a stale-version conflict.",
            _PATH_OPS,
            errors=(VfsErrorCode.STALE_VERSION, VfsErrorCode.CONFLICT),
        ),
        _invariant(
            VfsInvariantKind.CACHE_PIN_COHERENCE,
            "Cache entries are keyed by canonical identity and version; commit invalidates "
            "stale mutable aliases, and pin state reflects the committed CID only.",
            tuple(dict.fromkeys(_CONTENT_OPS + _MUTATIONS)),
            errors=(VfsErrorCode.INTEGRITY_FAILURE,),
        ),
        _invariant(
            VfsInvariantKind.BACKEND_NEGOTIATION,
            "The selected backend advertises versioned capabilities before dispatch; "
            "required unsupported capabilities fail without effects.",
            _ALL,
            errors=(
                VfsErrorCode.CAPABILITY_UNAVAILABLE,
                VfsErrorCode.UNSUPPORTED,
            ),
        ),
        _invariant(
            VfsInvariantKind.AUTHORIZATION,
            "Authentication and authorization occur on canonical path, operation, mount, "
            "and version before observable data or mutation effects.",
            _ALL,
            errors=(
                VfsErrorCode.AUTHENTICATION_REQUIRED,
                VfsErrorCode.PERMISSION_DENIED,
            ),
        ),
        _invariant(
            VfsInvariantKind.RESOURCE,
            "Requests declare or inherit bounded bytes, entries, handles, time, and "
            "concurrency; exceeding a bound fails with no unreported truncation.",
            _ALL,
            errors=(
                VfsErrorCode.RESOURCE_EXHAUSTED,
                VfsErrorCode.DEADLINE_EXCEEDED,
            ),
        ),
        _invariant(
            VfsInvariantKind.DEGRADATION,
            "Fallback and degraded execution are explicit in the result, preserve safety "
            "invariants, and never substitute mock, placeholder, stale, or partial success.",
            _ALL,
            errors=(
                VfsErrorCode.CAPABILITY_UNAVAILABLE,
                VfsErrorCode.IO_FAILURE,
            ),
        ),
    )


_COMMON_ERRORS = (
    VfsErrorCode.INVALID_ARGUMENT,
    VfsErrorCode.PERMISSION_DENIED,
    VfsErrorCode.CAPABILITY_UNAVAILABLE,
    VfsErrorCode.RESOURCE_EXHAUSTED,
    VfsErrorCode.IO_FAILURE,
)


def _canonical_operations(
    invariants: Sequence[InvariantContract],
) -> tuple[OperationContract, ...]:
    invariant_map = {
        operation: tuple(
            item.invariant_id for item in invariants if operation in item.applies_to
        )
        for operation in VfsOperation
    }
    error_map = {
        operation: tuple(
            dict.fromkeys(
                _COMMON_ERRORS
                + tuple(
                    error
                    for invariant in invariants
                    if operation in invariant.applies_to
                    for error in invariant.error_codes
                )
            )
        )
        for operation in VfsOperation
    }
    spec: dict[
        VfsOperation,
        tuple[str, tuple[DataMode, ...], tuple[DataMode, ...], bool, bool | None],
    ] = {
        VfsOperation.PATH_RESOLVE: (
            "Canonicalize and resolve a versioned path without content effects.",
            (DataMode.METADATA,),
            (DataMode.METADATA,),
            False,
            True,
        ),
        VfsOperation.MOUNT: (
            "Bind a canonical mount path to a negotiated backend.",
            (DataMode.METADATA,),
            (DataMode.METADATA,),
            True,
            None,
        ),
        VfsOperation.READ: (
            "Read a byte range from a path or open handle.",
            (DataMode.METADATA, DataMode.HANDLE),
            (DataMode.BYTES,),
            False,
            True,
        ),
        VfsOperation.WRITE: (
            "Commit bytes to a path or writable handle.",
            (DataMode.BYTES, DataMode.HANDLE, DataMode.METADATA),
            (DataMode.METADATA,),
            True,
            None,
        ),
        VfsOperation.OPEN: (
            "Open a scoped byte-oriented file handle.",
            (DataMode.METADATA,),
            (DataMode.HANDLE, DataMode.METADATA),
            False,
            False,
        ),
        VfsOperation.CLOSE: (
            "Close and invalidate an open handle.",
            (DataMode.HANDLE,),
            (DataMode.NONE,),
            True,
            False,
        ),
        VfsOperation.SEEK: (
            "Move a handle's byte offset.",
            (DataMode.HANDLE, DataMode.METADATA),
            (DataMode.METADATA,),
            True,
            False,
        ),
        VfsOperation.STAT: (
            "Return typed metadata for one canonical path.",
            (DataMode.METADATA,),
            (DataMode.METADATA,),
            False,
            True,
        ),
        VfsOperation.LIST: (
            "Return a bounded page of typed directory entries.",
            (DataMode.METADATA,),
            (DataMode.METADATA,),
            False,
            True,
        ),
        VfsOperation.MKDIR: (
            "Create a directory under explicit parent/existence policy.",
            (DataMode.METADATA,),
            (DataMode.METADATA,),
            True,
            None,
        ),
        VfsOperation.REMOVE: (
            "Remove a file or directory under explicit recursive policy.",
            (DataMode.METADATA,),
            (DataMode.NONE,),
            True,
            None,
        ),
        VfsOperation.RENAME: (
            "Move one namespace entry to an explicit destination.",
            (DataMode.METADATA,),
            (DataMode.METADATA,),
            True,
            None,
        ),
        VfsOperation.COPY: (
            "Copy content or a tree to an explicit destination.",
            (DataMode.METADATA,),
            (DataMode.METADATA,),
            True,
            None,
        ),
    }
    return tuple(
        OperationContract(
            operation=operation,
            summary=spec[operation][0],
            input_modes=spec[operation][1],
            output_modes=spec[operation][2],
            execution_modes=(ExecutionMode.SYNC, ExecutionMode.ASYNC),
            invariant_ids=invariant_map[operation],
            error_codes=error_map[operation],
            source_contract_ids=("source:vfs-026-acceptance",),
            mutates=spec[operation][3],
            idempotent=spec[operation][4],
        )
        for operation in VfsOperation
    )


def _surface_source_id(surface: PublicSurface) -> str:
    return f"source:vfs-026-surface:{surface.value}"


def _canonical_sources() -> tuple[SourceContract, ...]:
    acceptance = SourceContract(
        source_id="source:vfs-026-acceptance",
        kind=ContractSourceKind.REVIEWED_INTERFACE,
        locator="task://VFS-026/acceptance",
        revision="baguqeerauopqqkevmksjwate5nvprfxnz3bgci3kmqfcx2ckbmv6al66xfeq",
        summary="Reviewed VFS-026 operation and invariant acceptance contract.",
        reviewed=True,
    )
    surface_sources = tuple(
        SourceContract(
            source_id=_surface_source_id(surface),
            kind=ContractSourceKind.REVIEWED_INTERFACE,
            locator=f"task://VFS-026/acceptance#surface-{surface.value}",
            revision=acceptance.revision,
            summary=(
                f"Reviewed requirement to map the {surface.value} public facade "
                "to the canonical operation contract."
            ),
            reviewed=True,
        )
        for surface in PublicSurface
    )
    missing_backend = SourceContract(
        source_id="source:missing-backend-atomicity-contract",
        kind=ContractSourceKind.REVIEWED_INTERFACE,
        locator="missing://backend/atomicity-capability-contract",
        revision="unavailable",
        summary=(
            "Required backend-specific atomicity capability contract was not "
            "provided to this pack."
        ),
        reviewed=False,
        available=False,
        expectation_authority=False,
    )
    return (acceptance,) + surface_sources + (missing_backend,)


def _canonical_surfaces() -> tuple[PublicSurfaceContract, ...]:
    # CLI and ordinary HTTP requests do not expose persistent handles.  This is
    # an explicit unsupported mapping, not an omitted or presumed operation.
    no_handles = {VfsOperation.OPEN, VfsOperation.CLOSE, VfsOperation.SEEK}
    stateful = {
        PublicSurface.PYTHON,
        PublicSurface.MCP,
        PublicSurface.MCP_PLUS_PLUS,
        PublicSurface.LIBP2P,
    }
    modes = {
        PublicSurface.PYTHON: (ExecutionMode.SYNC, ExecutionMode.ASYNC),
        PublicSurface.CLI: (ExecutionMode.SYNC,),
        PublicSurface.MCP: (ExecutionMode.ASYNC,),
        PublicSurface.MCP_PLUS_PLUS: (ExecutionMode.ASYNC,),
        PublicSurface.HTTP: (ExecutionMode.ASYNC,),
        PublicSurface.LIBP2P: (ExecutionMode.ASYNC,),
    }
    names = {
        PublicSurface.PYTHON: "Python VFS API",
        PublicSurface.CLI: "VFS command line",
        PublicSurface.MCP: "MCP VFS tools",
        PublicSurface.MCP_PLUS_PLUS: "MCP++ VFS tools",
        PublicSurface.HTTP: "HTTP VFS API",
        PublicSurface.LIBP2P: "libp2p VFS protocol",
    }
    surfaces: list[PublicSurfaceContract] = []
    for surface in PublicSurface:
        source_id = _surface_source_id(surface)
        operations = tuple(
            SurfaceOperationContract(
                operation=operation,
                support=(
                    OperationSupport.SUPPORTED
                    if surface in stateful or operation not in no_handles
                    else OperationSupport.UNSUPPORTED
                ),
                source_contract_ids=(source_id,),
                entrypoint=f"{surface.value}:{operation.value}",
                note=(
                    ""
                    if surface in stateful or operation not in no_handles
                    else "Persistent handle lifecycle is outside this stateless facade."
                ),
            )
            for operation in VfsOperation
        )
        surfaces.append(
            PublicSurfaceContract(
                surface=surface,
                contract_name=names[surface],
                execution_modes=modes[surface],
                operations=operations,
                source_contract_ids=(source_id,),
            )
        )
    return tuple(surfaces)


def _canonical_vectors() -> tuple[CanonicalVector, ...]:
    source = ("source:vfs-026-acceptance",)
    return (
        CanonicalVector(
            "vector:path:nfc-dot-segments",
            VfsOperation.PATH_RESOLVE,
            "Unicode and dot segments canonicalize without changing identity.",
            {"path": "/cafe\u0301//draft/../data", "version": "v7"},
            {"path": "/café/data", "version": "v7"},
            (
                "invariant:versioned_path",
                "invariant:unicode",
                "invariant:traversal",
            ),
            source,
        ),
        CanonicalVector(
            "vector:path:root-traversal-denied",
            VfsOperation.PATH_RESOLVE,
            "Traversal above the selected root is rejected.",
            {"path": "/../../etc/passwd"},
            {
                "error": {
                    "code": VfsErrorCode.TRAVERSAL_DENIED.value,
                    "effects": "none",
                }
            },
            ("invariant:root", "invariant:traversal", "invariant:error"),
            source,
        ),
        CanonicalVector(
            "vector:mount:component-boundary",
            VfsOperation.MOUNT,
            "Longest mount prefix matches only complete path components.",
            {
                "mounts": ["/", "/data", "/database"],
                "path": "/data/report",
            },
            {"selected_mount": "/data"},
            ("invariant:mount", "invariant:backend_negotiation"),
            source,
        ),
        CanonicalVector(
            "vector:write:utf8-byte-accounting",
            VfsOperation.WRITE,
            "Explicit UTF-8 text adapter reports encoded byte size.",
            {"path": "/café.txt", "text": "é", "encoding": "utf-8"},
            {"committed_bytes_hex": "c3a9", "size": 2, "written": 2},
            (
                "invariant:bytes_text",
                "invariant:read_write",
                "invariant:cid_size",
            ),
            source,
        ),
        CanonicalVector(
            "vector:seek:byte-offset",
            VfsOperation.SEEK,
            "Seek offsets count bytes, not decoded characters.",
            {"handle": "h1", "offset": -2, "whence": "end", "size": 9},
            {"offset": 7, "content_effects": "none"},
            ("invariant:seek", "invariant:bytes_text"),
            source,
        ),
        CanonicalVector(
            "vector:stat:cid-size",
            VfsOperation.STAT,
            "Metadata binds CID and size to the same committed bytes.",
            {"path": "/hello", "version": "v2"},
            {
                "type": "file",
                "size": 5,
                "cid_input_bytes_hex": "68656c6c6f",
                "version": "v2",
            },
            ("invariant:stat_list", "invariant:cid_size"),
            source,
        ),
        CanonicalVector(
            "vector:remove:non-empty",
            VfsOperation.REMOVE,
            "Non-recursive removal cannot partially remove a non-empty directory.",
            {"path": "/dir", "recursive": False},
            {
                "error": {
                    "code": VfsErrorCode.DIRECTORY_NOT_EMPTY.value,
                    "effects": "none",
                }
            },
            (
                "invariant:directory_mutation",
                "invariant:namespace_mutation",
                "invariant:atomicity",
            ),
            source,
        ),
        CanonicalVector(
            "vector:journal:duplicate-replay",
            VfsOperation.RENAME,
            "Replaying one committed operation identity does not duplicate effects.",
            {"operation_id": "op-17", "replay_count": 2},
            {"commits": 1, "destination_entries": 1},
            ("invariant:journal_replay", "invariant:atomicity"),
            source,
        ),
        CanonicalVector(
            "vector:version:stale-write",
            VfsOperation.WRITE,
            "A stale base version fails without changing the current version.",
            {"path": "/x", "base_version": "v1", "current_version": "v2"},
            {
                "error": {
                    "code": VfsErrorCode.STALE_VERSION.value,
                    "effects": "none",
                },
                "current_version": "v2",
            },
            ("invariant:versioning", "invariant:atomicity"),
            source,
        ),
        CanonicalVector(
            "vector:auth:precedes-cache",
            VfsOperation.READ,
            "An unauthorized cache hit returns no content or metadata.",
            {"path": "/secret", "cache": "hit", "authorized": False},
            {
                "error": VfsErrorCode.PERMISSION_DENIED.value,
                "bytes_exposed": 0,
                "metadata_exposed": False,
            },
            ("invariant:authorization", "invariant:cache_pin_coherence"),
            source,
        ),
        CanonicalVector(
            "vector:resource:list-limit",
            VfsOperation.LIST,
            "List respects its entry bound and returns an explicit continuation.",
            {"path": "/many", "max_entries": 2},
            {"entry_count": 2, "continuation_required": True},
            ("invariant:resource", "invariant:stat_list"),
            source,
        ),
        CanonicalVector(
            "vector:degradation:no-silent-fallback",
            VfsOperation.COPY,
            "Required atomic cross-backend copy cannot silently degrade.",
            {"source": "/a/x", "destination": "/b/x", "atomic": True},
            {
                "error": VfsErrorCode.CAPABILITY_UNAVAILABLE.value,
                "degraded": False,
                "effects": "none",
            },
            (
                "invariant:backend_negotiation",
                "invariant:atomicity",
                "invariant:degradation",
            ),
            source,
        ),
    )


def _canonical_facade_examples() -> tuple[FacadeExample, ...]:
    acceptance = ("source:vfs-026-acceptance",)
    return (
        FacadeExample(
            "facade:python:compatible-bytes",
            PublicSurface.PYTHON,
            FacadeCompatibility.COMPATIBLE,
            "Python returns bytes and preserves the stable metadata fields.",
            VfsOperation.READ,
            {"call": "await vfs.read('/x')", "result": {"type": "bytes"}},
            "The async facade preserves canonical byte semantics.",
            acceptance,
        ),
        FacadeExample(
            "facade:cli:compatible-text",
            PublicSurface.CLI,
            FacadeCompatibility.COMPATIBLE,
            "CLI text output names its decoding.",
            VfsOperation.READ,
            {
                "argv": ["vfs", "read", "/x", "--text", "--encoding", "utf-8"],
                "exit": 0,
            },
            "Text is an explicit adapter and the exit status remains transport-only.",
            acceptance,
        ),
        FacadeExample(
            "facade:mcp++:compatible-error",
            PublicSurface.MCP_PLUS_PLUS,
            FacadeCompatibility.COMPATIBLE,
            "MCP++ retains the canonical error code in its typed result.",
            VfsOperation.STAT,
            {"result": {"ok": False, "error": {"code": "not_found"}}},
            "The transport representation is lossless.",
            acceptance,
        ),
        FacadeExample(
            "facade:http:incompatible-success-error",
            PublicSurface.HTTP,
            FacadeCompatibility.INCOMPATIBLE,
            "HTTP reports success while hiding an error in an untyped string.",
            VfsOperation.WRITE,
            {"status": 200, "body": {"message": "write failed"}},
            "A failed mutation must have a stable error code and cannot report success.",
            acceptance,
        ),
        FacadeExample(
            "facade:libp2p:incompatible-traversal",
            PublicSurface.LIBP2P,
            FacadeCompatibility.INCOMPATIBLE,
            "libp2p forwards an above-root path to a backend.",
            VfsOperation.READ,
            {"path": "/../../secret", "backend_dispatched": True},
            "Canonical traversal rejection and authorization must precede dispatch.",
            acceptance,
        ),
        FacadeExample(
            "facade:mcp:incompatible-implicit-text",
            PublicSurface.MCP,
            FacadeCompatibility.INCOMPATIBLE,
            "MCP decodes bytes with an implicit replacement policy.",
            VfsOperation.READ,
            {"result": {"text": "�", "encoding": None, "errors": None}},
            "Implicit decoding loses byte identity and suppresses an encoding error.",
            acceptance,
        ),
        FacadeExample(
            "facade:http:unresolved-backend-atomicity",
            PublicSurface.HTTP,
            FacadeCompatibility.UNRESOLVED,
            "HTTP claims cross-backend atomic copy without a reviewed backend contract.",
            VfsOperation.COPY,
            {"request": {"atomic": True}, "response": {"atomic": True}},
            "The facade cannot be classified until backend capability evidence exists.",
            ("source:missing-backend-atomicity-contract",),
        ),
    )


def build_vfs_contract_pack() -> VfsContractPack:
    """Return the canonical VFS-026 pack.

    Construction performs all referential, coverage, and authority checks, so
    callers cannot receive a partially mapped canonical pack.
    """

    sources = _canonical_sources()
    invariants = _canonical_invariants()
    operations = _canonical_operations(invariants)
    issues = (
        ExpectationIssue(
            issue_id="issue:backend-specific-atomicity",
            kind=IssueKind.MISSING,
            subject=(
                "Backend-specific atomicity strength and cross-backend transaction "
                "support are absent until a reviewed capability contract is supplied."
            ),
            source_contract_ids=("source:missing-backend-atomicity-contract",),
            positions=(),
            state=ExpectationState.UNRESOLVED,
        ),
    )
    return VfsContractPack(
        sources=sources,
        invariants=invariants,
        operations=operations,
        surfaces=_canonical_surfaces(),
        issues=issues,
        vectors=_canonical_vectors(),
        facade_examples=_canonical_facade_examples(),
    )


def canonical_vfs_contract_pack() -> VfsContractPack:
    """Compatibility spelling for :func:`build_vfs_contract_pack`."""

    return build_vfs_contract_pack()


def assert_vfs_contract_pack_complete(pack: VfsContractPack) -> None:
    """Re-run the fail-closed construction checks for an existing pack."""

    pack.__post_init__()


def publish_vfs_contract_pack(
    output_path: str | os.PathLike[str],
    pack: VfsContractPack | None = None,
) -> Path:
    """Atomically publish canonical JSON and return the resolved destination."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (pack or build_vfs_contract_pack()).to_json(indent=2) + "\n"
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


# Explicit alias for consumers which use the goal's "canonical pack" wording.
CanonicalVfsContractPack = VfsContractPack


__all__ = [
    "CONTRACT_PACK_AUTHORIZES_REPAIR",
    "CONTRACT_PACK_IS_COMPLETION_EVIDENCE",
    "CONTRACT_PACK_IS_CORRECTNESS_EVIDENCE",
    "CanonicalVector",
    "CanonicalVfsContractPack",
    "ContractSourceKind",
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
    "PublicSurface",
    "PublicSurfaceContract",
    "SOURCE_PRECEDENCE",
    "SourceContract",
    "SurfaceOperationContract",
    "VFS_CANONICAL_OPERATION_MATRIX_SCHEMA",
    "VFS_CONTRACT_PACK_GOAL_ID",
    "VFS_CONTRACT_PACK_SCHEMA",
    "VFS_CONTRACT_PACK_VERSION",
    "VfsContractPack",
    "VfsContractPackError",
    "VfsErrorCode",
    "VfsInvariantKind",
    "VfsOperation",
    "assert_vfs_contract_pack_complete",
    "build_vfs_contract_pack",
    "canonical_vfs_contract_pack",
    "publish_vfs_contract_pack",
]
