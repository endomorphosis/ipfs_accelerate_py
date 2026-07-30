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

from .program_contracts import SOURCE_PRECEDENCE, ContractSourceKind

VFS_CONTRACT_PACK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/vfs-contract-pack@1"
)
VFS_CANONICAL_OPERATION_MATRIX_SCHEMA: Final[str] = (
    "vfs/canonical-operation-matrix@1"
)
VFS_DIFFERENTIAL_CONTRACT_WITNESS_SCHEMA: Final[str] = (
    "vfs/differential-contract-witness@1"
)
VFS_CANONICAL_OPERATION_MATRIX_CLAIM_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/"
    "vfs-canonical-operation-matrix-claim@1"
)
VFS_DRIFT_INVENTORY_SCHEMA: Final[str] = "vfs/drift-inventory@1"
VFS_CONTRACT_PACK_VERSION: Final[str] = "vfs-contract-pack/v1"
VFS_CONTRACT_PACK_GOAL_ID: Final[str] = "VFS-026"
VFS_CANONICAL_OPERATION_MATRIX_GOAL_ID: Final[str] = "VFS-G158"
VFS_CANONICAL_OPERATION_MATRIX_TASK_ID: Final[str] = "VFS-073"
VFS_CANONICAL_OPERATION_MATRIX_PARENT_GOAL_ID: Final[str] = "VFS-G090"
VFS_CANONICAL_OPERATION_MATRIX_OBJECTIVE_REVISION: Final[str] = (
    "baguqeeramjx4cofpxl4tvz57mno5f3hx6nfxkwp65ydb7il6vjw5hirigdaa"
)
VFS_CANONICAL_OPERATION_MATRIX_GOAL_PACKET_ID: Final[str] = (
    "goal_packet/vfs_drift/ipfs_accelerate_py/1ad8c79bee6a"
)
VFS_CANONICAL_OPERATION_MATRIX_PACKET_GOAL_IDS: Final[tuple[str, ...]] = (
    "VFS-G091",
    VFS_CANONICAL_OPERATION_MATRIX_GOAL_ID,
)
VFS_CANONICAL_OPERATION_MATRIX_PACKET_TASK_IDS: Final[tuple[str, ...]] = (
    "VFS-077",
    VFS_CANONICAL_OPERATION_MATRIX_TASK_ID,
)
VFS_CANONICAL_OPERATION_MATRIX_EVIDENCE_TERMS: Final[tuple[str, ...]] = (
    VFS_CANONICAL_OPERATION_MATRIX_SCHEMA,
)
VFS_CANONICAL_OPERATION_MATRIX_PACKET_EVIDENCE_TERMS: Final[
    tuple[str, ...]
] = (
    VFS_DIFFERENTIAL_CONTRACT_WITNESS_SCHEMA,
    VFS_CANONICAL_OPERATION_MATRIX_SCHEMA,
)
VFS_DRIFT_INVENTORY_GOAL_ID: Final[str] = "VFS-G090"
VFS_DRIFT_INVENTORY_TASK_ID: Final[str] = "VFS-045"
VFS_DRIFT_INVENTORY_OBJECTIVE_REVISION: Final[str] = (
    "baguqeerahsnzkm2u6e6qvh6hnjyrwwwhyf6usdlocisaibw5zyk4ujektotq"
)
VFS_DRIFT_INVENTORY_SOURCE_REVISION: Final[str] = (
    "git:f6a574375febbcf9a46fcd24bbc7bc5cfb551de5"
)

# Authority bounds: this is a comparison contract, not proof of repository
# correctness or permission to select/modify an implementation.
CONTRACT_PACK_IS_COMPLETION_EVIDENCE: Final[bool] = False
CONTRACT_PACK_IS_CORRECTNESS_EVIDENCE: Final[bool] = False
CONTRACT_PACK_AUTHORIZES_REPAIR: Final[bool] = False
DRIFT_INVENTORY_IS_COMPLETION_EVIDENCE: Final[bool] = False
DRIFT_INVENTORY_IS_CORRECTNESS_EVIDENCE: Final[bool] = False
DRIFT_INVENTORY_AUTHORIZES_REPAIR: Final[bool] = False
DRIFT_INVENTORY_VARIANT_PRESENCE_IS_DEFECT: Final[bool] = False

# Exact-text discovery anchors bind the implementation surface to the
# supervisor-fed objective heap without changing contract-pack content identity.
assert VFS_CANONICAL_OPERATION_MATRIX_SCHEMA == "vfs/canonical-operation-matrix@1"
assert (
    VFS_DIFFERENTIAL_CONTRACT_WITNESS_SCHEMA
    == "vfs/differential-contract-witness@1"
)
assert VFS_CANONICAL_OPERATION_MATRIX_GOAL_ID == "VFS-G158"
assert VFS_CANONICAL_OPERATION_MATRIX_TASK_ID == "VFS-073"
assert VFS_CANONICAL_OPERATION_MATRIX_PARENT_GOAL_ID == "VFS-G090"
assert VFS_CANONICAL_OPERATION_MATRIX_PACKET_GOAL_IDS == (
    "VFS-G091",
    "VFS-G158",
)
assert VFS_CANONICAL_OPERATION_MATRIX_PACKET_TASK_IDS == ("VFS-077", "VFS-073")
assert VFS_CANONICAL_OPERATION_MATRIX_EVIDENCE_TERMS == (
    "vfs/canonical-operation-matrix@1",
)
assert VFS_CANONICAL_OPERATION_MATRIX_PACKET_EVIDENCE_TERMS == (
    "vfs/differential-contract-witness@1",
    "vfs/canonical-operation-matrix@1",
)


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


VFS_CANONICAL_OPERATION_MATRIX_REQUIRED_INVARIANTS: Final[
    tuple[VfsInvariantKind, ...]
] = tuple(VfsInvariantKind)


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


class DriftSurfaceKind(str, Enum):
    """Closed inventory vocabulary required by objective VFS-G090."""

    VFS = "vfs"
    FSSPEC = "fsspec"
    VFS_MANAGER = "vfs_manager"
    BUCKET = "bucket"
    JOURNAL = "journal"
    VERSION = "version"
    BACKEND = "backend"
    MCP_HANDLER = "mcp_handler"
    ENDPOINT = "endpoint"
    TOOL = "tool"
    SDK_MANIFEST = "sdk_manifest"
    VARIANT = "variant"


class DriftFindingKind(str, Enum):
    """What an inventory finding reports, without selecting a repair."""

    SURFACE = "surface"
    CONTRACT_DRIFT = "contract_drift"
    MANIFEST_DRIFT = "manifest_drift"
    DUPLICATE_CANDIDATE = "duplicate_candidate"
    VARIANT_PRESENCE = "variant_presence"


class DriftAssessment(str, Enum):
    """Evidence state of a finding, not a defect or repair verdict."""

    OBSERVED = "observed"
    DRIFT = "drift"
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
class VfsDriftEvidence:
    """One reviewed implementation observation bound to an exact Git blob."""

    evidence_id: str
    locator: str
    revision: str
    summary: str
    observed_symbols: tuple[str, ...]
    reviewed: bool = True
    available: bool = True
    expectation_authority: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.evidence_id, "evidence_id")
        _require_identifier(self.locator, "locator")
        _require_identifier(self.revision, "revision")
        _require_identifier(self.summary, "summary")
        _require_unique(self.observed_symbols, "observed_symbols")
        if not self.observed_symbols:
            raise VfsContractPackError(
                f"drift evidence {self.evidence_id!r} must name an observation"
            )
        if (
            type(self.reviewed) is not bool
            or type(self.available) is not bool
            or type(self.expectation_authority) is not bool
        ):
            raise VfsContractPackError("drift evidence authority flags must be booleans")
        if not self.reviewed or not self.available:
            raise VfsContractPackError(
                f"drift evidence {self.evidence_id!r} must be available and reviewed"
            )
        if self.expectation_authority:
            raise VfsContractPackError(
                "implementation observations cannot define canonical expectations"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_kind": ContractSourceKind.IMPLEMENTATION_OBSERVATION.value,
            "locator": self.locator,
            "revision": self.revision,
            "summary": self.summary,
            "observed_symbols": list(self.observed_symbols),
            "reviewed": self.reviewed,
            "available": self.available,
            "expectation_authority": self.expectation_authority,
        }


@dataclass(frozen=True)
class VfsDriftFinding:
    """Inventory result mapped to reviewed operations, never a repair decision."""

    finding_id: str
    kind: DriftFindingKind
    assessment: DriftAssessment
    surface_kinds: tuple[DriftSurfaceKind, ...]
    summary: str
    evidence_ids: tuple[str, ...]
    canonical_operations: tuple[VfsOperation, ...]
    source_contract_ids: tuple[str, ...]
    variant_presence_only: bool = False
    defect_label: None = None
    repair_decision: None = None

    def __post_init__(self) -> None:
        _require_identifier(self.finding_id, "finding_id")
        _require_identifier(self.summary, "summary")
        _require_unique(
            (item.value for item in self.surface_kinds), "surface_kinds"
        )
        _require_unique(self.evidence_ids, "evidence_ids")
        _require_unique(
            (item.value for item in self.canonical_operations),
            "canonical_operations",
        )
        _require_unique(self.source_contract_ids, "source_contract_ids")
        if not self.surface_kinds:
            raise VfsContractPackError(
                f"drift finding {self.finding_id!r} needs a surface kind"
            )
        if not self.evidence_ids:
            raise VfsContractPackError(
                f"drift finding {self.finding_id!r} needs observation evidence"
            )
        if not self.canonical_operations or not self.source_contract_ids:
            raise VfsContractPackError(
                f"drift finding {self.finding_id!r} needs a reviewed operation mapping"
            )
        if type(self.variant_presence_only) is not bool:
            raise VfsContractPackError("variant_presence_only must be a boolean")
        if self.kind is DriftFindingKind.VARIANT_PRESENCE:
            if (
                not self.variant_presence_only
                or self.assessment is not DriftAssessment.OBSERVED
            ):
                raise VfsContractPackError(
                    "variant presence must remain an observation, not a drift verdict"
                )
        elif self.variant_presence_only:
            raise VfsContractPackError(
                "variant_presence_only is reserved for variant-presence findings"
            )
        if self.defect_label is not None:
            raise VfsContractPackError(
                "inventory findings cannot assign a defect label"
            )
        if self.repair_decision is not None:
            raise VfsContractPackError(
                "inventory findings cannot contain repair decisions"
            )

    def to_record(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "kind": self.kind.value,
            "assessment": self.assessment.value,
            "surface_kinds": _enum_values(self.surface_kinds),
            "summary": self.summary,
            "evidence_ids": list(self.evidence_ids),
            "canonical_operations": _enum_values(self.canonical_operations),
            "source_contract_ids": list(self.source_contract_ids),
            "variant_presence_only": self.variant_presence_only,
            "defect_label": self.defect_label,
            "repair_decision": self.repair_decision,
        }


@dataclass(frozen=True)
class VfsDriftInventory:
    """Revision-bound VFS surface findings mapped to the canonical contract."""

    contract_pack_id: str
    contract_version: str
    source_revision: str
    evidence: tuple[VfsDriftEvidence, ...]
    findings: tuple[VfsDriftFinding, ...]
    schema: str = VFS_DRIFT_INVENTORY_SCHEMA
    goal_id: str = VFS_DRIFT_INVENTORY_GOAL_ID
    task_id: str = VFS_DRIFT_INVENTORY_TASK_ID
    objective_revision: str = VFS_DRIFT_INVENTORY_OBJECTIVE_REVISION

    def __post_init__(self) -> None:
        expected_identity = (
            (self.schema, VFS_DRIFT_INVENTORY_SCHEMA, "schema"),
            (self.goal_id, VFS_DRIFT_INVENTORY_GOAL_ID, "goal_id"),
            (self.task_id, VFS_DRIFT_INVENTORY_TASK_ID, "task_id"),
            (
                self.objective_revision,
                VFS_DRIFT_INVENTORY_OBJECTIVE_REVISION,
                "objective_revision",
            ),
            (
                self.source_revision,
                VFS_DRIFT_INVENTORY_SOURCE_REVISION,
                "source_revision",
            ),
            (self.contract_version, VFS_CONTRACT_PACK_VERSION, "contract_version"),
        )
        for actual, expected, field_name in expected_identity:
            if actual != expected:
                raise VfsContractPackError(
                    f"{field_name} must be {expected!r}, got {actual!r}"
                )
        _require_identifier(self.contract_pack_id, "contract_pack_id")
        _require_unique((item.evidence_id for item in self.evidence), "evidence ids")
        _require_unique((item.finding_id for item in self.findings), "finding ids")
        if not self.evidence or not self.findings:
            raise VfsContractPackError("drift inventory cannot be empty")

        evidence_ids = {item.evidence_id for item in self.evidence}
        referenced_evidence = {
            evidence_id
            for finding in self.findings
            for evidence_id in finding.evidence_ids
        }
        if referenced_evidence != evidence_ids:
            missing = sorted(referenced_evidence - evidence_ids)
            unused = sorted(evidence_ids - referenced_evidence)
            raise VfsContractPackError(
                f"drift evidence coverage differs; missing={missing}, unused={unused}"
            )

        covered_surfaces = {
            surface_kind
            for finding in self.findings
            for surface_kind in finding.surface_kinds
        }
        if covered_surfaces != set(DriftSurfaceKind):
            missing = sorted(
                item.value for item in set(DriftSurfaceKind) - covered_surfaces
            )
            extra = sorted(
                item.value for item in covered_surfaces - set(DriftSurfaceKind)
            )
            raise VfsContractPackError(
                f"drift surface coverage differs; missing={missing}, extra={extra}"
            )

        covered_operations = {
            operation
            for finding in self.findings
            for operation in finding.canonical_operations
        }
        if covered_operations != set(VfsOperation):
            missing = sorted(
                item.value for item in set(VfsOperation) - covered_operations
            )
            raise VfsContractPackError(
                f"drift operation mapping is incomplete; missing={missing}"
            )
        if not any(
            item.kind is DriftFindingKind.MANIFEST_DRIFT for item in self.findings
        ):
            raise VfsContractPackError(
                "drift inventory must contain an evidence-backed manifest finding"
            )
        if not any(
            item.kind is DriftFindingKind.VARIANT_PRESENCE for item in self.findings
        ):
            raise VfsContractPackError(
                "drift inventory must record variant presence separately"
            )
        if not any(
            item.kind is DriftFindingKind.DUPLICATE_CANDIDATE
            for item in self.findings
        ):
            raise VfsContractPackError(
                "drift inventory must record duplicate candidates separately"
            )

    @property
    def drift_findings(self) -> tuple[VfsDriftFinding, ...]:
        return tuple(
            item
            for item in self.findings
            if item.assessment is DriftAssessment.DRIFT
        )

    @property
    def repair_decisions(self) -> tuple[Any, ...]:
        """Repair selection belongs to a separate authority-bearing workflow."""

        return ()

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema": self.schema,
            "evidence_kinds": [VFS_DRIFT_INVENTORY_SCHEMA],
            "goal_id": self.goal_id,
            "task_id": self.task_id,
            "objective_revision": self.objective_revision,
            "contract_pack_id": self.contract_pack_id,
            "contract_version": self.contract_version,
            "source_revision": self.source_revision,
            "authority": {
                "completion_evidence": DRIFT_INVENTORY_IS_COMPLETION_EVIDENCE,
                "correctness_evidence": DRIFT_INVENTORY_IS_CORRECTNESS_EVIDENCE,
                "authorizes_repair": DRIFT_INVENTORY_AUTHORIZES_REPAIR,
                "variant_presence_is_defect": (
                    DRIFT_INVENTORY_VARIANT_PRESENCE_IS_DEFECT
                ),
            },
            "evidence": [item.to_record() for item in self.evidence],
            "inventory_findings": [item.to_record() for item in self.findings],
            "repair_decisions": list(self.repair_decisions),
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


def _canonical_drift_evidence() -> tuple[VfsDriftEvidence, ...]:
    """Reviewed observations from the pinned IPFS Kit source revision."""

    return (
        VfsDriftEvidence(
            evidence_id="evidence:vfs-core-fsspec",
            locator="ipfs_kit_py/ipfs_kit_py/ipfs_fsspec.py",
            revision="git-blob:8c968bac7497f342fd7fbbb454b145910b4ba41e",
            summary=(
                "The fsspec module defines both the registered IPFS filesystem "
                "facade and the broad VFS core."
            ),
            observed_symbols=(
                "IPFSFSSpecFileSystem",
                "VFSCore",
                "fsspec.register_implementation",
                "get_vfs",
            ),
        ),
        VfsDriftEvidence(
            evidence_id="evidence:vfs-manager",
            locator="ipfs_kit_py/ipfs_kit_py/vfs_manager.py",
            revision="git-blob:dd679f5e3ab15ed0873473719fc962389abf17fc",
            summary=(
                "The VFS manager exposes a generic operation dispatcher and "
                "namespace helpers."
            ),
            observed_symbols=(
                "VFSManager",
                "execute_vfs_operation",
                "list_files",
                "create_folder",
                "delete_item",
                "rename_item",
                "move_item",
            ),
        ),
        VfsDriftEvidence(
            evidence_id="evidence:bucket-vfs",
            locator="ipfs_kit_py/ipfs_kit_py/bucket_vfs_manager.py",
            revision="git-blob:d8504612125b8e3cf4be19b9984f82890274a85b",
            summary=(
                "BucketVFSManager and BucketVFS expose a competing bucket-oriented "
                "file namespace."
            ),
            observed_symbols=(
                "BucketVFSManager",
                "BucketVFS",
                "add_file",
                "cat_file",
                "list_files",
                "remove_file",
            ),
        ),
        VfsDriftEvidence(
            evidence_id="evidence:filesystem-journal",
            locator="ipfs_kit_py/ipfs_kit_py/filesystem_journal.py",
            revision="git-blob:5287ae3eb0cb06e062eb4ab7a15f837be10864ae",
            summary=(
                "The filesystem journal records, replays, and directly applies "
                "namespace mutations."
            ),
            observed_symbols=(
                "FilesystemJournal",
                "FilesystemJournalManager",
                "record_operation",
                "recover",
                "write_file",
                "delete",
                "rename",
                "mount",
            ),
        ),
        VfsDriftEvidence(
            evidence_id="evidence:vfs-version",
            locator="ipfs_kit_py/ipfs_kit_py/vfs_version_tracker.py",
            revision="git-blob:5c05e99b810e60ac79b932571137acfd47d744eb",
            summary=(
                "The version tracker scans filesystem state and exposes snapshots, "
                "history, status, and checkout."
            ),
            observed_symbols=(
                "VFSVersionTracker",
                "scan_filesystem",
                "create_version_snapshot",
                "get_version_history",
                "checkout_version",
            ),
        ),
        VfsDriftEvidence(
            evidence_id="evidence:iroh-vfs-backend",
            locator="ipfs_kit_py/ipfs_kit_py/iroh_vfs.py",
            revision="git-blob:aa0d65976501d389e7406af195f182c7700c457c",
            summary=(
                "The Iroh backend adapter exposes its own path, byte, metadata, "
                "and namespace operations."
            ),
            observed_symbols=(
                "IrohVFSAdapter",
                "resolve",
                "read_bytes",
                "write_bytes",
                "mkdir",
                "remove",
                "info",
                "list",
            ),
        ),
        VfsDriftEvidence(
            evidence_id="evidence:mcp-vfs-handler",
            locator=(
                "ipfs_kit_py/ipfs_kit_py/mcp/handlers/"
                "mcp_vfs_action_handler.py"
            ),
            revision="git-blob:921bf3a40a4548d934b688ae13fcfa49c3737379",
            summary=(
                "The MCP vfs.action handler returns a success envelope around an "
                "explicit placeholder implementation."
            ),
            observed_symbols=(
                "McpVfsActionHandler",
                "handle",
                "_execute_mcp_vfs_action_controller",
                "Comprehensive feature implementation in progress",
            ),
        ),
        VfsDriftEvidence(
            evidence_id="evidence:vfs-endpoints",
            locator="ipfs_kit_py/ipfs_kit_py/mcp/ipfs_kit/api/vfs_endpoints.py",
            revision="git-blob:28187edb12e3ac7a6f27edea7f9f60b448f64360",
            summary=(
                "The primary HTTP endpoint class exposes list/create/delete/rename/"
                "upload/move namespace methods."
            ),
            observed_symbols=(
                "VFSEndpoints",
                "list_files",
                "create_folder",
                "delete_item",
                "rename_item",
                "upload_file",
                "move_item",
            ),
        ),
        VfsDriftEvidence(
            evidence_id="evidence:vfs-endpoints-fixed",
            locator=(
                "ipfs_kit_py/ipfs_kit_py/mcp/ipfs_kit/api/"
                "vfs_endpoints_fixed.py"
            ),
            revision="git-blob:7f473fd50cc22851fa6dfc34ce8752760d78af78",
            summary="A separately named fixed endpoint variant is present.",
            observed_symbols=("VFSEndpoints", "vfs_endpoints_fixed.py"),
        ),
        VfsDriftEvidence(
            evidence_id="evidence:vfs-endpoints-optimized",
            locator=(
                "ipfs_kit_py/ipfs_kit_py/mcp/ipfs_kit/api/"
                "vfs_endpoints_optimized.py"
            ),
            revision="git-blob:2144f72856847266b861d2fbd45cc7f3bf406312",
            summary="A separately named optimized endpoint variant is present.",
            observed_symbols=("vfs_endpoints_optimized.py",),
        ),
        VfsDriftEvidence(
            evidence_id="evidence:mcp-vfs-tools",
            locator=(
                "ipfs_kit_py/ipfs_kit_py/mcp/ipfs_kit/mcp_tools/vfs_tools.py"
            ),
            revision="git-blob:c439f866bb240202a663cc06872f7a7f2fc75423",
            summary=(
                "The Python MCP VFS tool surface exposes statistics, cache, vector "
                "index, and knowledge-base methods."
            ),
            observed_symbols=(
                "VFSTools",
                "get_vfs_statistics",
                "get_vfs_cache",
                "get_vfs_vector_index",
                "get_vfs_knowledge_base",
            ),
        ),
        VfsDriftEvidence(
            evidence_id="evidence:mcp-js-tools-manifest",
            locator=(
                "ipfs_kit_py/ipfs_kit_py/mcp_server/js_sdk/"
                "tools-manifest.json"
            ),
            revision="git-blob:af7a5cffeefab111f9f16bd3ba1d7426ffd6e45f",
            summary=(
                "The JS SDK manifest registers files_* operations but none of the "
                "observed Python VFSTools get_vfs_* names."
            ),
            observed_symbols=(
                "files_ls",
                "files_mkdir",
                "files_stat",
                "files_write",
                "files_read",
                "files_rm",
                "absence:get_vfs_*",
            ),
        ),
        VfsDriftEvidence(
            evidence_id="evidence:ipfs-fsspec-clean-variant",
            locator="ipfs_kit_py/ipfs_kit_py/ipfs_fsspec.py.clean",
            revision="git-blob:700beedf6f20d6b91682c125322231cd004216d3",
            summary="A .clean fsspec variant is present in the source tree.",
            observed_symbols=("ipfs_fsspec.py.clean",),
        ),
        VfsDriftEvidence(
            evidence_id="evidence:ipfs-fsspec-full-variant",
            locator="ipfs_kit_py/ipfs_kit_py/ipfs_fsspec.py.full",
            revision="git-blob:92dfb961574d6cefd3b7cc76ee8160f599f664b8",
            summary="A .full fsspec variant is present in the source tree.",
            observed_symbols=("ipfs_fsspec.py.full",),
        ),
    )


def _canonical_drift_findings() -> tuple[VfsDriftFinding, ...]:
    """Map every reviewed inventory family to canonical operation contracts."""

    contract = ("source:vfs-026-acceptance",)
    return (
        VfsDriftFinding(
            finding_id="finding:vfs-core-fsspec",
            kind=DriftFindingKind.SURFACE,
            assessment=DriftAssessment.OBSERVED,
            surface_kinds=(DriftSurfaceKind.VFS, DriftSurfaceKind.FSSPEC),
            summary=(
                "VFSCore and IPFSFSSpecFileSystem are reviewed against the complete "
                "canonical operation vocabulary."
            ),
            evidence_ids=("evidence:vfs-core-fsspec",),
            canonical_operations=_ALL,
            source_contract_ids=contract,
        ),
        VfsDriftFinding(
            finding_id="finding:vfs-manager",
            kind=DriftFindingKind.SURFACE,
            assessment=DriftAssessment.OBSERVED,
            surface_kinds=(DriftSurfaceKind.VFS_MANAGER,),
            summary="VFSManager namespace helpers map to canonical namespace operations.",
            evidence_ids=("evidence:vfs-manager",),
            canonical_operations=(
                VfsOperation.LIST,
                VfsOperation.MKDIR,
                VfsOperation.REMOVE,
                VfsOperation.RENAME,
                VfsOperation.COPY,
            ),
            source_contract_ids=contract,
        ),
        VfsDriftFinding(
            finding_id="finding:bucket-vfs",
            kind=DriftFindingKind.SURFACE,
            assessment=DriftAssessment.OBSERVED,
            surface_kinds=(DriftSurfaceKind.BUCKET,),
            summary="Bucket operations map to the same canonical byte and namespace model.",
            evidence_ids=("evidence:bucket-vfs",),
            canonical_operations=(
                VfsOperation.READ,
                VfsOperation.WRITE,
                VfsOperation.STAT,
                VfsOperation.LIST,
                VfsOperation.MKDIR,
                VfsOperation.REMOVE,
                VfsOperation.COPY,
            ),
            source_contract_ids=contract,
        ),
        VfsDriftFinding(
            finding_id="finding:filesystem-journal",
            kind=DriftFindingKind.SURFACE,
            assessment=DriftAssessment.OBSERVED,
            surface_kinds=(DriftSurfaceKind.JOURNAL,),
            summary=(
                "Journal mutation and replay entrypoints map to canonical mutation, "
                "atomicity, and replay contracts."
            ),
            evidence_ids=("evidence:filesystem-journal",),
            canonical_operations=(
                VfsOperation.MOUNT,
                VfsOperation.WRITE,
                VfsOperation.STAT,
                VfsOperation.MKDIR,
                VfsOperation.REMOVE,
                VfsOperation.RENAME,
            ),
            source_contract_ids=contract,
        ),
        VfsDriftFinding(
            finding_id="finding:vfs-version",
            kind=DriftFindingKind.SURFACE,
            assessment=DriftAssessment.OBSERVED,
            surface_kinds=(DriftSurfaceKind.VERSION,),
            summary=(
                "Snapshot, history, and checkout entrypoints map to canonical "
                "versioned path and content operations."
            ),
            evidence_ids=("evidence:vfs-version",),
            canonical_operations=(
                VfsOperation.PATH_RESOLVE,
                VfsOperation.READ,
                VfsOperation.WRITE,
                VfsOperation.STAT,
                VfsOperation.LIST,
            ),
            source_contract_ids=contract,
        ),
        VfsDriftFinding(
            finding_id="finding:iroh-vfs-backend",
            kind=DriftFindingKind.SURFACE,
            assessment=DriftAssessment.OBSERVED,
            surface_kinds=(DriftSurfaceKind.BACKEND,),
            summary=(
                "The Iroh adapter is a backend surface subject to canonical path, "
                "byte, metadata, and namespace semantics."
            ),
            evidence_ids=("evidence:iroh-vfs-backend",),
            canonical_operations=(
                VfsOperation.PATH_RESOLVE,
                VfsOperation.READ,
                VfsOperation.WRITE,
                VfsOperation.STAT,
                VfsOperation.LIST,
                VfsOperation.MKDIR,
                VfsOperation.REMOVE,
            ),
            source_contract_ids=contract,
        ),
        VfsDriftFinding(
            finding_id="finding:mcp-vfs-handler-placeholder",
            kind=DriftFindingKind.CONTRACT_DRIFT,
            assessment=DriftAssessment.DRIFT,
            surface_kinds=(DriftSurfaceKind.MCP_HANDLER,),
            summary=(
                "The generic MCP VFS action surface can emit success for a named "
                "placeholder; this is an inventory finding, not a repair selection."
            ),
            evidence_ids=("evidence:mcp-vfs-handler",),
            canonical_operations=_ALL,
            source_contract_ids=contract,
        ),
        VfsDriftFinding(
            finding_id="finding:vfs-http-endpoints",
            kind=DriftFindingKind.SURFACE,
            assessment=DriftAssessment.OBSERVED,
            surface_kinds=(DriftSurfaceKind.ENDPOINT,),
            summary=(
                "HTTP VFS namespace methods map to canonical list, create, remove, "
                "rename, write, and copy operations."
            ),
            evidence_ids=("evidence:vfs-endpoints",),
            canonical_operations=(
                VfsOperation.LIST,
                VfsOperation.MKDIR,
                VfsOperation.REMOVE,
                VfsOperation.RENAME,
                VfsOperation.WRITE,
                VfsOperation.COPY,
            ),
            source_contract_ids=contract,
        ),
        VfsDriftFinding(
            finding_id="finding:mcp-vfs-tools",
            kind=DriftFindingKind.SURFACE,
            assessment=DriftAssessment.OBSERVED,
            surface_kinds=(DriftSurfaceKind.TOOL,),
            summary=(
                "Python VFS observability tools are mapped to canonical stat and "
                "bounded list result semantics."
            ),
            evidence_ids=("evidence:mcp-vfs-tools",),
            canonical_operations=(VfsOperation.STAT, VfsOperation.LIST),
            source_contract_ids=contract,
        ),
        VfsDriftFinding(
            finding_id="finding:mcp-sdk-manifest-drift",
            kind=DriftFindingKind.MANIFEST_DRIFT,
            assessment=DriftAssessment.DRIFT,
            surface_kinds=(DriftSurfaceKind.SDK_MANIFEST,),
            summary=(
                "The JS SDK manifest and Python VFS tool namespace differ at the "
                "pinned revision; the observation does not decide whether to repair "
                "either surface."
            ),
            evidence_ids=(
                "evidence:mcp-vfs-tools",
                "evidence:mcp-js-tools-manifest",
            ),
            canonical_operations=(
                VfsOperation.READ,
                VfsOperation.WRITE,
                VfsOperation.STAT,
                VfsOperation.LIST,
                VfsOperation.MKDIR,
                VfsOperation.REMOVE,
            ),
            source_contract_ids=contract,
        ),
        VfsDriftFinding(
            finding_id="finding:vfs-endpoint-duplicate-candidates",
            kind=DriftFindingKind.DUPLICATE_CANDIDATE,
            assessment=DriftAssessment.UNRESOLVED,
            surface_kinds=(DriftSurfaceKind.ENDPOINT, DriftSurfaceKind.VARIANT),
            summary=(
                "Base, fixed, and optimized endpoint modules are duplicate candidates; "
                "presence and naming do not establish a defect or preferred module."
            ),
            evidence_ids=(
                "evidence:vfs-endpoints",
                "evidence:vfs-endpoints-fixed",
                "evidence:vfs-endpoints-optimized",
            ),
            canonical_operations=(
                VfsOperation.LIST,
                VfsOperation.MKDIR,
                VfsOperation.REMOVE,
                VfsOperation.RENAME,
                VfsOperation.WRITE,
                VfsOperation.COPY,
            ),
            source_contract_ids=contract,
        ),
        VfsDriftFinding(
            finding_id="finding:ipfs-fsspec-variant-presence",
            kind=DriftFindingKind.VARIANT_PRESENCE,
            assessment=DriftAssessment.OBSERVED,
            surface_kinds=(DriftSurfaceKind.FSSPEC, DriftSurfaceKind.VARIANT),
            summary=(
                "The base, .clean, and .full fsspec paths are recorded without "
                "classifying any variant as broken, canonical, or repair-worthy."
            ),
            evidence_ids=(
                "evidence:vfs-core-fsspec",
                "evidence:ipfs-fsspec-clean-variant",
                "evidence:ipfs-fsspec-full-variant",
            ),
            canonical_operations=_ALL,
            source_contract_ids=contract,
            variant_presence_only=True,
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


def assert_vfs_drift_inventory_complete(
    inventory: VfsDriftInventory,
    contract_pack: VfsContractPack | None = None,
) -> None:
    """Validate coverage and reviewed mapping authority for a drift inventory."""

    inventory.__post_init__()
    pack = contract_pack or build_vfs_contract_pack()
    if inventory.contract_pack_id != pack.content_id:
        raise VfsContractPackError(
            "drift inventory contract_pack_id does not match the reviewed pack"
        )

    sources = {item.source_id: item for item in pack.sources}
    for finding in inventory.findings:
        unknown_sources = sorted(set(finding.source_contract_ids) - sources.keys())
        if unknown_sources:
            raise VfsContractPackError(
                f"{finding.finding_id} references unknown contract sources: "
                f"{unknown_sources}"
            )
        if not all(
            sources[source_id].expectation_authority
            for source_id in finding.source_contract_ids
        ):
            raise VfsContractPackError(
                f"{finding.finding_id} lacks reviewed mapping authority"
            )
        for operation in finding.canonical_operations:
            operation_contract = pack.operation_contract(operation)
            if operation_contract.state is not ExpectationState.RESOLVED:
                raise VfsContractPackError(
                    f"{finding.finding_id} maps to unresolved operation "
                    f"{operation.value!r}"
                )


def build_vfs_drift_inventory(
    contract_pack: VfsContractPack | None = None,
) -> VfsDriftInventory:
    """Return the revision-bound ``vfs/drift-inventory@1`` evidence artifact."""

    pack = contract_pack or build_vfs_contract_pack()
    inventory = VfsDriftInventory(
        contract_pack_id=pack.content_id,
        contract_version=pack.contract_version,
        source_revision=VFS_DRIFT_INVENTORY_SOURCE_REVISION,
        evidence=_canonical_drift_evidence(),
        findings=_canonical_drift_findings(),
    )
    assert_vfs_drift_inventory_complete(inventory, pack)
    return inventory


def canonical_vfs_drift_inventory() -> VfsDriftInventory:
    """Compatibility spelling for :func:`build_vfs_drift_inventory`."""

    return build_vfs_drift_inventory()


def canonical_operation_matrix_evidence() -> str:
    """Return the closed VFS-G158 canonical-operation-matrix evidence term."""

    return VFS_CANONICAL_OPERATION_MATRIX_SCHEMA


def canonical_operation_matrix_evidence_terms() -> tuple[str, ...]:
    """Return only the domain evidence authored by this contract-pack module.

    The sibling differential runtime witness remains owned by VFS-G091 and is
    exposed separately through :func:`packet_evidence_terms`.  Keeping those
    surfaces distinct prevents a structural matrix claim from being mistaken
    for runtime conformance evidence.
    """

    return VFS_CANONICAL_OPERATION_MATRIX_EVIDENCE_TERMS


def covered_evidence_terms() -> tuple[str, ...]:
    """Return the VFS-G158 objective evidence proved by this module."""

    return canonical_operation_matrix_evidence_terms()


def packet_evidence_terms() -> tuple[str, ...]:
    """Return the shared VFS-G091/VFS-G158 goal-packet evidence vocabulary.

    The ordered pair aligns the packet with the objective heap: differential
    runtime evidence first, then its canonical operation-matrix dependency.
    This is discovery metadata only and never enters contract-pack identity.
    """

    return VFS_CANONICAL_OPERATION_MATRIX_PACKET_EVIDENCE_TERMS


def all_covered_evidence_terms() -> tuple[str, ...]:
    """Return packet-wide domain terms for cross-module discovery scanners."""

    return packet_evidence_terms()


def assert_vfs_canonical_operation_matrix_complete(
    pack: VfsContractPack,
    inventory: VfsDriftInventory | None = None,
) -> None:
    """Fail closed unless *pack* proves the complete VFS-G158 matrix.

    This strengthens the generic pack checks with the objective-specific
    requirements: every canonical operation and semantic dimension is a
    reviewed resolved expectation, every public surface makes an explicit
    resolved support decision, and the revision-bound inventory backs
    duplicate, variant, and manifest findings without selecting a defect or
    repair.  Explicit unresolved capability issues remain visible.
    """

    if not isinstance(pack, VfsContractPack):
        raise TypeError("pack must be a VfsContractPack")
    pack.__post_init__()

    unresolved_operations = sorted(
        item.operation.value
        for item in pack.operations
        if item.state is not ExpectationState.RESOLVED
    )
    if unresolved_operations:
        raise VfsContractPackError(
            "canonical operation matrix has unresolved operations: "
            f"{unresolved_operations}"
        )

    unresolved_invariants = sorted(
        item.kind.value
        for item in pack.invariants
        if item.kind in VFS_CANONICAL_OPERATION_MATRIX_REQUIRED_INVARIANTS
        and item.state is not ExpectationState.RESOLVED
    )
    if unresolved_invariants:
        raise VfsContractPackError(
            "canonical operation matrix has unresolved semantic dimensions: "
            f"{unresolved_invariants}"
        )

    unresolved_surface_bindings = sorted(
        f"{surface.surface.value}:{binding.operation.value}"
        for surface in pack.surfaces
        for binding in surface.operations
        if not binding.support.is_resolved
    )
    if unresolved_surface_bindings:
        raise VfsContractPackError(
            "canonical operation matrix has unresolved surface bindings: "
            f"{unresolved_surface_bindings}"
        )

    selected_inventory = inventory or build_vfs_drift_inventory(pack)
    if not isinstance(selected_inventory, VfsDriftInventory):
        raise TypeError("inventory must be a VfsDriftInventory")
    assert_vfs_drift_inventory_complete(selected_inventory, pack)

    if selected_inventory.repair_decisions:
        raise VfsContractPackError(
            "canonical operation evidence cannot contain repair decisions"
        )
    if any(
        finding.defect_label is not None or finding.repair_decision is not None
        for finding in selected_inventory.findings
    ):
        raise VfsContractPackError(
            "inventory findings cannot be promoted to defect or repair decisions"
        )
    variants = tuple(
        finding
        for finding in selected_inventory.findings
        if finding.kind is DriftFindingKind.VARIANT_PRESENCE
    )
    if not variants or any(
        not finding.variant_presence_only
        or finding.assessment is not DriftAssessment.OBSERVED
        for finding in variants
    ):
        raise VfsContractPackError(
            "variant presence must remain evidence-backed observation only"
        )


def vfs_canonical_operation_matrix_satisfies_objective(
    pack: VfsContractPack,
    inventory: VfsDriftInventory | None = None,
) -> bool:
    """Return whether the matrix and drift mapping satisfy VFS-G158."""

    try:
        assert_vfs_canonical_operation_matrix_complete(pack, inventory)
    except (AttributeError, KeyError, TypeError, VfsContractPackError):
        return False
    return True


def prove_vfs_canonical_operation_matrix(
    pack: VfsContractPack | None = None,
    inventory: VfsDriftInventory | None = None,
) -> dict[str, Any]:
    """Emit a deterministic, portable VFS-G158 structural evidence claim.

    The claim binds the matrix to the exact contract-pack and drift-inventory
    content IDs.  Its ``satisfied`` flag is structural evidence only: authority
    remains false and the VFS-G091 runtime witness is explicitly left to the
    hermetic differential harness.
    """

    selected_pack = pack or build_vfs_contract_pack()
    if not isinstance(selected_pack, VfsContractPack):
        raise TypeError("pack must be a VfsContractPack")
    selected_inventory = inventory or build_vfs_drift_inventory(selected_pack)
    if not isinstance(selected_inventory, VfsDriftInventory):
        raise TypeError("inventory must be a VfsDriftInventory")

    satisfied = vfs_canonical_operation_matrix_satisfies_objective(
        selected_pack, selected_inventory
    )
    finding_kinds = tuple(
        sorted({item.kind.value for item in selected_inventory.findings})
    )
    drift_surface_kinds = tuple(
        item.value
        for item in DriftSurfaceKind
        if any(
            item in finding.surface_kinds
            for finding in selected_inventory.findings
        )
    )
    record: dict[str, Any] = {
        "schema": VFS_CANONICAL_OPERATION_MATRIX_CLAIM_SCHEMA,
        "evidence": VFS_CANONICAL_OPERATION_MATRIX_SCHEMA,
        "evidence_terms": list(canonical_operation_matrix_evidence_terms()),
        "requirement_id": VFS_CANONICAL_OPERATION_MATRIX_SCHEMA,
        "goal_id": VFS_CANONICAL_OPERATION_MATRIX_GOAL_ID,
        "parent_goal_id": VFS_CANONICAL_OPERATION_MATRIX_PARENT_GOAL_ID,
        "task_id": VFS_CANONICAL_OPERATION_MATRIX_TASK_ID,
        "objective_revision": (
            VFS_CANONICAL_OPERATION_MATRIX_OBJECTIVE_REVISION
        ),
        "goal_packet_id": VFS_CANONICAL_OPERATION_MATRIX_GOAL_PACKET_ID,
        "packet_goal_ids": list(
            VFS_CANONICAL_OPERATION_MATRIX_PACKET_GOAL_IDS
        ),
        "packet_task_ids": list(
            VFS_CANONICAL_OPERATION_MATRIX_PACKET_TASK_IDS
        ),
        "packet_evidence_terms": list(packet_evidence_terms()),
        "bindings": {
            "contract_pack_content_id": selected_pack.content_id,
            "drift_inventory_content_id": selected_inventory.content_id,
            "contract_version": selected_pack.contract_version,
        },
        "coverage": {
            "operations": _enum_values(tuple(VfsOperation)),
            "operation_count": len(selected_pack.operations),
            "public_surfaces": _enum_values(tuple(PublicSurface)),
            "public_surface_count": len(selected_pack.surfaces),
            "required_invariant_kinds": _enum_values(
                VFS_CANONICAL_OPERATION_MATRIX_REQUIRED_INVARIANTS
            ),
            "resolved_invariant_kinds": _enum_values(
                tuple(
                    item.kind
                    for item in selected_pack.invariants
                    if item.state is ExpectationState.RESOLVED
                )
            ),
            "execution_modes": _enum_values(tuple(ExecutionMode)),
            "drift_surface_kinds": list(drift_surface_kinds),
            "drift_finding_kinds": list(finding_kinds),
            "unresolved_issue_ids": [
                item.issue_id for item in selected_pack.unresolved_expectations
            ],
            "variant_presence_is_defect": False,
            "repair_decision_count": len(selected_inventory.repair_decisions),
            "matrix_complete": satisfied,
        },
        "sibling_evidence_requirements": [
            {
                "evidence": VFS_DIFFERENTIAL_CONTRACT_WITNESS_SCHEMA,
                "goal_id": "VFS-G091",
                "task_id": "VFS-077",
                "status": "external_runtime_witness_required",
            }
        ],
        "satisfied": satisfied,
        "claim_level": "structural_contract",
        "claims_runtime_conformance": False,
        "authoritative": False,
        "completion_authoritative": False,
        "semantic_authority": False,
        "authorizes_repair": False,
    }
    record["content_id"] = _content_id(record)
    return record


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


def publish_vfs_drift_inventory(
    output_path: str | os.PathLike[str],
    inventory: VfsDriftInventory | None = None,
) -> Path:
    """Atomically publish the drift inventory without publishing a repair plan."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (inventory or build_vfs_drift_inventory()).to_json(indent=2) + "\n"
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
    "DRIFT_INVENTORY_AUTHORIZES_REPAIR",
    "DRIFT_INVENTORY_IS_COMPLETION_EVIDENCE",
    "DRIFT_INVENTORY_IS_CORRECTNESS_EVIDENCE",
    "DRIFT_INVENTORY_VARIANT_PRESENCE_IS_DEFECT",
    "SOURCE_PRECEDENCE",
    "VFS_CANONICAL_OPERATION_MATRIX_CLAIM_SCHEMA",
    "VFS_CANONICAL_OPERATION_MATRIX_EVIDENCE_TERMS",
    "VFS_CANONICAL_OPERATION_MATRIX_GOAL_ID",
    "VFS_CANONICAL_OPERATION_MATRIX_GOAL_PACKET_ID",
    "VFS_CANONICAL_OPERATION_MATRIX_OBJECTIVE_REVISION",
    "VFS_CANONICAL_OPERATION_MATRIX_PACKET_EVIDENCE_TERMS",
    "VFS_CANONICAL_OPERATION_MATRIX_PACKET_GOAL_IDS",
    "VFS_CANONICAL_OPERATION_MATRIX_PACKET_TASK_IDS",
    "VFS_CANONICAL_OPERATION_MATRIX_PARENT_GOAL_ID",
    "VFS_CANONICAL_OPERATION_MATRIX_REQUIRED_INVARIANTS",
    "VFS_CANONICAL_OPERATION_MATRIX_SCHEMA",
    "VFS_CANONICAL_OPERATION_MATRIX_TASK_ID",
    "VFS_CONTRACT_PACK_GOAL_ID",
    "VFS_CONTRACT_PACK_SCHEMA",
    "VFS_CONTRACT_PACK_VERSION",
    "VFS_DIFFERENTIAL_CONTRACT_WITNESS_SCHEMA",
    "VFS_DRIFT_INVENTORY_GOAL_ID",
    "VFS_DRIFT_INVENTORY_OBJECTIVE_REVISION",
    "VFS_DRIFT_INVENTORY_SCHEMA",
    "VFS_DRIFT_INVENTORY_SOURCE_REVISION",
    "VFS_DRIFT_INVENTORY_TASK_ID",
    "CanonicalVector",
    "CanonicalVfsContractPack",
    "ContractSourceKind",
    "DataMode",
    "DriftAssessment",
    "DriftFindingKind",
    "DriftSurfaceKind",
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
    "SourceContract",
    "SurfaceOperationContract",
    "VfsContractPack",
    "VfsContractPackError",
    "VfsDriftEvidence",
    "VfsDriftFinding",
    "VfsDriftInventory",
    "VfsErrorCode",
    "VfsInvariantKind",
    "VfsOperation",
    "all_covered_evidence_terms",
    "assert_vfs_canonical_operation_matrix_complete",
    "assert_vfs_contract_pack_complete",
    "assert_vfs_drift_inventory_complete",
    "build_vfs_contract_pack",
    "build_vfs_drift_inventory",
    "canonical_operation_matrix_evidence",
    "canonical_operation_matrix_evidence_terms",
    "canonical_vfs_contract_pack",
    "canonical_vfs_drift_inventory",
    "covered_evidence_terms",
    "packet_evidence_terms",
    "prove_vfs_canonical_operation_matrix",
    "publish_vfs_contract_pack",
    "publish_vfs_drift_inventory",
    "vfs_canonical_operation_matrix_satisfies_objective",
]
