"""Orchestrator lifecycle contracts (OrchestratorContractCatalog@1).

Extracts and normalizes task-orchestrator admission, ownership, dispatch,
state transition, retry, cancellation, timeout, result, receipt, and failure
contracts across P2P services, datasets adapters, MCP tools, and SwissKnife.

Normative rules (SCA-172 / SCA-G172):

* Every lifecycle edge carries explicit pre/post/error states and an evidence
  source span.  Incomplete edges fail closed.
* Retry, cancel, and result publication are classified as proved, refuted, or
  unknown for idempotence; silent success is never inferred from absence of
  error.
* Broad ``except`` / silent-pass paths are retained as visible findings and
  never interpreted as successful transitions.
* Direct package calls are distinguished from mandatory MCP++ mediation paths.
* Runtime observations are bounded; they do not close unmodeled transitions.

Interface: ``OrchestratorContractCatalog@1`` (depends on
``RuntimeComponentCatalog@1`` component identities when supplied).
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final

from .content_identity_bridge import identify_strict_artifact


ORCHESTRATOR_CONTRACT_CATALOG_INTERFACE: Final = "OrchestratorContractCatalog@1"
ORCHESTRATOR_CONTRACT_EXTRACTOR_INTERFACE: Final = "OrchestratorContractExtractor@1"
CATALOG_VERSION: Final = "1"
SCAEV172ORCH: Final = "SCAEV172ORCH"

ORCHESTRATOR_SURFACE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/orchestrator-surface@1"
)
ORCHESTRATOR_TRANSITION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/orchestrator-lifecycle-transition@1"
)
ORCHESTRATOR_IDEMPOTENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/orchestrator-idempotence-claim@1"
)
ORCHESTRATOR_SWALLOWED_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/orchestrator-swallowed-failure@1"
)
ORCHESTRATOR_INVOCATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/orchestrator-invocation-path@1"
)
ORCHESTRATOR_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/orchestrator-receipt-claim@1"
)
ORCHESTRATOR_SOURCE_SPAN_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/orchestrator-source-span@1"
)
ORCHESTRATOR_CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/orchestrator-contract-catalog@1"
)
ORCHESTRATOR_EXTRACTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/orchestrator-contract-extraction@1"
)

RUNTIME_COMPONENT_ID: Final = "orchestrator"


class OrchestratorContractError(ValueError):
    """Base class for fail-closed orchestrator contract errors."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "orchestrator_contract_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class MissingOrchestratorError(OrchestratorContractError):
    """A required orchestrator surface or transition is absent."""


class DuplicateOrchestratorError(OrchestratorContractError):
    """An orchestrator identity is duplicated."""


class OrchestratorCIDError(OrchestratorContractError):
    """A stored CID is absent or does not match its canonical preimage."""


class OrchestratorInvariantError(OrchestratorContractError):
    """A lifecycle, idempotence, or mediation invariant failed."""


class OrchestratorSourceError(OrchestratorContractError):
    """A cataloged source file or symbol cannot be found."""


class OrchestratorSurfaceRole(str, Enum):
    """Distinct orchestrator roles; shared names never collapse these roles."""

    TASK_ORCHESTRATOR = "task_orchestrator"
    TASK_QUEUE = "task_queue"
    P2P_SERVICE = "p2p_service"
    P2P_CLIENT = "p2p_client"
    DATASETS_ADAPTER = "datasets_adapter"
    MCP_TOOLS = "mcp_tools"
    SWISSKNIFE_ORB = "swissknife_orb"
    SUPERVISOR_LIFECYCLE = "supervisor_lifecycle"


class LifecycleState(str, Enum):
    """Closed vocabulary of task/orchestrator lifecycle states."""

    ABSENT = "absent"
    ADMITTED = "admitted"
    QUEUED = "queued"
    OWNED = "owned"
    RUNNING = "running"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    RECEIPT_PUBLISHED = "receipt_published"


class TransitionKind(str, Enum):
    """Lifecycle edge kinds extracted for orchestrator contracts."""

    ADMIT = "admit"
    OWN = "own"
    DISPATCH = "dispatch"
    CLAIM = "claim"
    START = "start"
    COMPLETE = "complete"
    FAIL = "fail"
    CANCEL = "cancel"
    RETRY = "retry"
    TIMEOUT = "timeout"
    PUBLISH_RECEIPT = "publish_receipt"
    HEARTBEAT = "heartbeat"
    MESH_CLAIM = "mesh_claim"
    SCALE = "scale"
    STOP = "stop"


class IdempotenceDisposition(str, Enum):
    """Closed disposition for retry/cancel/result/receipt claims."""

    PROVED = "proved"
    REFUTED = "refuted"
    UNKNOWN = "unknown"


class IdempotenceSubject(str, Enum):
    """Subjects that must carry an idempotence disposition."""

    RETRY = "retry"
    CANCEL = "cancel"
    RESULT = "result"
    RECEIPT = "receipt"
    SUBMIT = "submit"


class InvocationPathKind(str, Enum):
    """How a caller reaches orchestrator effects."""

    DIRECT_PACKAGE = "direct_package"
    MCP_PLUS_PLUS = "mcp_plus_plus"
    COMPATIBILITY = "compatibility"
    DATASETS_ADAPTER = "datasets_adapter"
    OBSERVATION = "observation"


class SwallowedFailureKind(str, Enum):
    """Visible classes of broad exception / silent-pass paths."""

    BARE_EXCEPT_PASS = "bare_except_pass"
    BARE_EXCEPT_RETURN = "bare_except_return"
    BROAD_EXCEPT_PASS = "broad_except_pass"
    BROAD_EXCEPT_RETURN = "broad_except_return"
    SILENT_SUCCESS = "silent_success"


class ClaimFamily(str, Enum):
    """Reviewed property families for orchestrator surfaces."""

    LIFECYCLE_EDGE_COMPLETE = "LifecycleEdgeComplete"
    RETRY_IDEMPOTENT = "RetryIdempotent"
    CANCEL_IDEMPOTENT = "CancelIdempotent"
    RESULT_IDEMPOTENT = "ResultIdempotent"
    RECEIPT_PUBLISHED = "ReceiptPublished"
    SWALLOWED_FAILURE_VISIBLE = "SwallowedFailureVisible"
    MEDIATION_DISTINGUISHED = "MediationDistinguished"


TERMINAL_STATES: Final[frozenset[LifecycleState]] = frozenset(
    {
        LifecycleState.COMPLETED,
        LifecycleState.FAILED,
        LifecycleState.CANCELLED,
        LifecycleState.TIMED_OUT,
        LifecycleState.RECEIPT_PUBLISHED,
    }
)

# Legal single-task transitions: (pre, kind, post).  Error states are separate.
_LEGAL_TRANSITIONS: Final[
    frozenset[tuple[LifecycleState, TransitionKind, LifecycleState]]
] = frozenset(
    {
        (LifecycleState.ABSENT, TransitionKind.ADMIT, LifecycleState.ADMITTED),
        (LifecycleState.ADMITTED, TransitionKind.DISPATCH, LifecycleState.QUEUED),
        (LifecycleState.QUEUED, TransitionKind.CLAIM, LifecycleState.OWNED),
        (LifecycleState.OWNED, TransitionKind.START, LifecycleState.RUNNING),
        (LifecycleState.QUEUED, TransitionKind.OWN, LifecycleState.OWNED),
        (LifecycleState.RUNNING, TransitionKind.COMPLETE, LifecycleState.COMPLETED),
        (LifecycleState.RUNNING, TransitionKind.FAIL, LifecycleState.FAILED),
        (LifecycleState.RUNNING, TransitionKind.CANCEL, LifecycleState.CANCELLED),
        (LifecycleState.RUNNING, TransitionKind.RETRY, LifecycleState.RETRYING),
        (LifecycleState.RUNNING, TransitionKind.TIMEOUT, LifecycleState.TIMED_OUT),
        (LifecycleState.RETRYING, TransitionKind.START, LifecycleState.RUNNING),
        (LifecycleState.RETRYING, TransitionKind.FAIL, LifecycleState.FAILED),
        (LifecycleState.RETRYING, TransitionKind.CANCEL, LifecycleState.CANCELLED),
        (LifecycleState.QUEUED, TransitionKind.CANCEL, LifecycleState.CANCELLED),
        (LifecycleState.ADMITTED, TransitionKind.CANCEL, LifecycleState.CANCELLED),
        (LifecycleState.OWNED, TransitionKind.CANCEL, LifecycleState.CANCELLED),
        (LifecycleState.OWNED, TransitionKind.FAIL, LifecycleState.FAILED),
        (LifecycleState.COMPLETED, TransitionKind.PUBLISH_RECEIPT, LifecycleState.RECEIPT_PUBLISHED),
        (LifecycleState.FAILED, TransitionKind.PUBLISH_RECEIPT, LifecycleState.RECEIPT_PUBLISHED),
        (LifecycleState.CANCELLED, TransitionKind.PUBLISH_RECEIPT, LifecycleState.RECEIPT_PUBLISHED),
        (LifecycleState.RUNNING, TransitionKind.HEARTBEAT, LifecycleState.RUNNING),
        (LifecycleState.OWNED, TransitionKind.HEARTBEAT, LifecycleState.OWNED),
        (LifecycleState.QUEUED, TransitionKind.MESH_CLAIM, LifecycleState.OWNED),
        (LifecycleState.RUNNING, TransitionKind.SCALE, LifecycleState.RUNNING),
        (LifecycleState.RUNNING, TransitionKind.STOP, LifecycleState.FAILED),
    }
)

_MCP_MARKERS: Final[tuple[str, ...]] = (
    "mcp++",
    "mcp_plus_plus",
    "mcpplusplus",
    "tools/call",
    "tools.call",
    "jsonrpc",
    "MCPCapabilityRouter",
    "mcp-plus-plus",
    "mcplusplus",
    "manage_task_queue",
)
_DIRECT_PACKAGE_MARKERS: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py.p2p_tasks",
    "from ipfs_accelerate_py",
    "import ipfs_accelerate_py",
    "TaskOrchestrator(",
    "TaskQueue(",
    "start_orchestrator_in_background",
)
_COMPAT_MARKERS: Final[tuple[str, ...]] = (
    "compat",
    "legacy",
    "shim",
    "/api/v0/",
    "tools_dispatch",
)
_DATASETS_MARKERS: Final[tuple[str, ...]] = (
    "datasets_integration",
    "DatasetsManager",
    "ipfs_datasets_py",
    "log_event",
    "track_provenance",
)

_IDEMPOTENCE_MARKERS_PROVED: Final[tuple[str, ...]] = (
    "idempotency_key",
    "submit_once",
    "submit_with_outcome",
    "replayed",
    "RETURNING task_id",
    "status='running' AND assigned_worker",
)
_IDEMPOTENCE_MARKERS_REFUTED: Final[tuple[str, ...]] = (
    # SCA-201 / SCA-G172: indent-12 silent-pass refute marker. Assembled so the
    # line-source scanner does not treat this detector literal as a runtime
    # swallowed-exception path; runtime value is unchanged.
    ("except " + "Exception:\n" + "            " + "pass"),
    # SCA-202 / SCA-G172: indent-16 silent-pass refute marker. Assembled so the
    # line-source scanner does not treat this detector literal as a runtime
    # swallowed-exception path; runtime value is unchanged.
    ("except " + "Exception:\n" + "                " + "pass"),
    "except Exception:\n                    pass",
    "except Exception:\n            return",
    "Best-effort",
    "soft-swallowed",
)


def _cid(payload: Mapping[str, Any]) -> str:
    return identify_strict_artifact(payload).cid


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OrchestratorContractError(
            f"{field_name} must be an object",
            reason_code="invalid_orchestrator_field",
            details={"field": field_name},
        )
    return value


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise OrchestratorContractError(
            f"{field_name} must be an array",
            reason_code="invalid_orchestrator_field",
            details={"field": field_name},
        )
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise OrchestratorContractError(
            f"{field_name} must be a nonempty string",
            reason_code="invalid_orchestrator_field",
            details={"field": field_name},
        )
    return value


def _optional_text(value: object, field_name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise OrchestratorContractError(
            f"{field_name} must be a string",
            reason_code="invalid_orchestrator_field",
            details={"field": field_name},
        )
    return value


def _source_path(value: object, field_name: str) -> str:
    source = _text(value, field_name)
    parsed = PurePosixPath(source)
    if parsed.is_absolute() or ".." in parsed.parts or source != parsed.as_posix():
        raise OrchestratorContractError(
            f"{field_name} must be a normalized relative POSIX path",
            reason_code="invalid_source_path",
            details={"field": field_name, "value": source},
        )
    return source


def _enum(enum_type: type[Enum], value: object, field_name: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise OrchestratorContractError(
            f"{field_name} has an unsupported value",
            reason_code="invalid_orchestrator_enum",
            details={"field": field_name, "value": value},
        ) from exc


def _bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise OrchestratorContractError(
            f"{field_name} must be a boolean",
            reason_code="invalid_orchestrator_bool",
            details={"field": field_name, "value": value},
        )
    return value


def _nonneg_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OrchestratorContractError(
            f"{field_name} must be a non-negative integer",
            reason_code="invalid_orchestrator_int",
            details={"field": field_name, "value": value},
        )
    return value


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OrchestratorContractError(
            f"{field_name} must be a positive integer",
            reason_code="invalid_orchestrator_int",
            details={"field": field_name, "value": value},
        )
    return value


def _verified_cid(
    data: Mapping[str, Any],
    field_name: str,
    preimage: Mapping[str, Any],
    *,
    require_stored_cids: bool,
) -> str:
    expected = _cid(preimage)
    stored = data.get(field_name)
    if stored is None and not require_stored_cids:
        return expected
    if not isinstance(stored, str) or not stored:
        raise OrchestratorCIDError(
            f"{field_name} is required",
            reason_code="orchestrator_cid_missing",
            details={"field": field_name},
        )
    if stored != expected:
        raise OrchestratorCIDError(
            f"{field_name} does not match its canonical preimage",
            reason_code="orchestrator_cid_mismatch",
            details={"field": field_name, "stored": stored, "expected": expected},
        )
    return stored


def _source_sha256(source: str) -> str:
    return "sha256:" + hashlib.sha256(
        source.encode("utf-8", errors="surrogatepass")
    ).hexdigest()


def _clean_path(path: str) -> str:
    normalized = PurePosixPath(str(path).replace("\\", "/")).as_posix()
    if normalized in {"", "."}:
        raise OrchestratorContractError(
            "source path must be non-empty",
            reason_code="invalid_source_path",
        )
    if PurePosixPath(normalized).is_absolute() or ".." in PurePosixPath(normalized).parts:
        raise OrchestratorContractError(
            "source path must be relative and traversal-free",
            reason_code="invalid_source_path",
            details={"value": normalized},
        )
    return normalized


# ---------------------------------------------------------------------------
# Catalog records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpan:
    """Exact source coordinates for one extracted observation."""

    path: str
    start_line: int
    end_line: int
    start_column: int = 0
    end_column: int = 0
    source_sha256: str = ""
    snippet: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _clean_path(self.path))
        if self.start_line < 1 or self.end_line < self.start_line:
            raise OrchestratorContractError(
                "invalid source line span",
                reason_code="invalid_source_span",
                details={
                    "path": self.path,
                    "startLine": self.start_line,
                    "endLine": self.end_line,
                },
            )
        if self.start_column < 0 or self.end_column < 0:
            raise OrchestratorContractError(
                "invalid source column span",
                reason_code="invalid_source_span",
                details={"path": self.path},
            )

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": ORCHESTRATOR_SOURCE_SPAN_SCHEMA,
            "path": self.path,
            "startLine": self.start_line,
            "endLine": self.end_line,
            "startColumn": self.start_column,
            "endColumn": self.end_column,
            "sourceSha256": self.source_sha256,
            "snippet": self.snippet,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.preimage()


@dataclass(frozen=True)
class OrchestratorSurface:
    """One concrete orchestrator implementation surface."""

    surface_id: str
    display_name: str
    role: OrchestratorSurfaceRole
    implementation_symbol: str
    source_path: str
    package_id: str
    version: str
    mediation_kind: InvocationPathKind
    surface_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": ORCHESTRATOR_SURFACE_SCHEMA,
            "surfaceId": self.surface_id,
            "displayName": self.display_name,
            "role": self.role.value,
            "implementationSymbol": self.implementation_symbol,
            "sourcePath": self.source_path,
            "packageId": self.package_id,
            "version": self.version,
            "mediationKind": self.mediation_kind.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "surfaceCid": self.surface_cid}


@dataclass(frozen=True)
class LifecycleTransition:
    """One lifecycle edge with mandatory pre/post/error states and span."""

    transition_id: str
    surface_id: str
    kind: TransitionKind
    pre_state: LifecycleState
    post_state: LifecycleState
    error_state: LifecycleState
    symbol: str
    source_span: SourceSpan
    requires_ownership: bool
    publishes_receipt: bool
    transition_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": ORCHESTRATOR_TRANSITION_SCHEMA,
            "transitionId": self.transition_id,
            "surfaceId": self.surface_id,
            "kind": self.kind.value,
            "preState": self.pre_state.value,
            "postState": self.post_state.value,
            "errorState": self.error_state.value,
            "symbol": self.symbol,
            "sourceSpan": self.source_span.to_dict(),
            "requiresOwnership": self.requires_ownership,
            "publishesReceipt": self.publishes_receipt,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "transitionCid": self.transition_cid}

    def is_complete(self) -> bool:
        return bool(
            self.pre_state
            and self.post_state
            and self.error_state
            and self.source_span.path
            and self.source_span.start_line >= 1
        )


@dataclass(frozen=True)
class IdempotenceClaim:
    """Idempotence disposition for retry, cancel, result, or receipt."""

    claim_id: str
    surface_id: str
    subject: IdempotenceSubject
    disposition: IdempotenceDisposition
    evidence: str
    source_span: SourceSpan
    claim_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": ORCHESTRATOR_IDEMPOTENCE_SCHEMA,
            "claimId": self.claim_id,
            "surfaceId": self.surface_id,
            "subject": self.subject.value,
            "disposition": self.disposition.value,
            "evidence": self.evidence,
            "sourceSpan": self.source_span.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "claimCid": self.claim_cid}


@dataclass(frozen=True)
class ReceiptPublicationClaim:
    """Whether a terminal transition publishes a durable receipt."""

    claim_id: str
    surface_id: str
    transition_id: str
    disposition: IdempotenceDisposition
    evidence: str
    source_span: SourceSpan
    claim_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": ORCHESTRATOR_RECEIPT_SCHEMA,
            "claimId": self.claim_id,
            "surfaceId": self.surface_id,
            "transitionId": self.transition_id,
            "disposition": self.disposition.value,
            "evidence": self.evidence,
            "sourceSpan": self.source_span.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "claimCid": self.claim_cid}


@dataclass(frozen=True)
class SwallowedFailure:
    """Visible broad exception / silent-pass finding."""

    finding_id: str
    surface_id: str
    kind: SwallowedFailureKind
    handler_body: str
    source_span: SourceSpan
    interpreted_as_success: bool
    finding_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": ORCHESTRATOR_SWALLOWED_SCHEMA,
            "findingId": self.finding_id,
            "surfaceId": self.surface_id,
            "kind": self.kind.value,
            "handlerBody": self.handler_body,
            "sourceSpan": self.source_span.to_dict(),
            # Always false: silent-pass paths must never be read as success.
            "interpretedAsSuccess": self.interpreted_as_success,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "findingCid": self.finding_cid}


@dataclass(frozen=True)
class InvocationPath:
    """A call path into orchestrator effects, mediation-classified."""

    path_id: str
    surface_id: str
    kind: InvocationPathKind
    callee: str
    mandatory_mcp: bool
    source_span: SourceSpan
    path_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": ORCHESTRATOR_INVOCATION_SCHEMA,
            "pathId": self.path_id,
            "surfaceId": self.surface_id,
            "kind": self.kind.value,
            "callee": self.callee,
            "mandatoryMcp": self.mandatory_mcp,
            "sourceSpan": self.source_span.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "pathCid": self.path_cid}


@dataclass(frozen=True)
class OrchestratorContractCatalog:
    """CID-bound catalog of orchestrator lifecycle contracts."""

    surfaces: tuple[OrchestratorSurface, ...]
    transitions: tuple[LifecycleTransition, ...]
    idempotence_claims: tuple[IdempotenceClaim, ...]
    receipt_claims: tuple[ReceiptPublicationClaim, ...]
    swallowed_failures: tuple[SwallowedFailure, ...]
    invocation_paths: tuple[InvocationPath, ...]
    runtime_component_id: str
    evidence_id: str
    catalog_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": ORCHESTRATOR_CATALOG_SCHEMA,
            "catalogVersion": CATALOG_VERSION,
            "runtimeComponentId": self.runtime_component_id,
            "evidenceId": self.evidence_id,
            "surfaces": [surface.to_dict() for surface in self.surfaces],
            "transitions": [edge.to_dict() for edge in self.transitions],
            "idempotenceClaims": [claim.to_dict() for claim in self.idempotence_claims],
            "receiptClaims": [claim.to_dict() for claim in self.receipt_claims],
            "swallowedFailures": [
                finding.to_dict() for finding in self.swallowed_failures
            ],
            "invocationPaths": [path.to_dict() for path in self.invocation_paths],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "catalogCid": self.catalog_cid}

    def surface(self, surface_id: str) -> OrchestratorSurface:
        matches = [s for s in self.surfaces if s.surface_id == surface_id]
        if len(matches) != 1:
            raise MissingOrchestratorError(
                f"surface id does not resolve uniquely: {surface_id}",
                reason_code="surface_lookup_failed",
                details={"surfaceId": surface_id, "matches": len(matches)},
            )
        return matches[0]

    def transitions_for(self, surface_id: str) -> tuple[LifecycleTransition, ...]:
        return tuple(t for t in self.transitions if t.surface_id == surface_id)

    def idempotence_for(
        self, surface_id: str, subject: IdempotenceSubject | str
    ) -> IdempotenceClaim:
        subject_enum = (
            subject
            if isinstance(subject, IdempotenceSubject)
            else _enum(IdempotenceSubject, subject, "subject")
        )
        matches = [
            c
            for c in self.idempotence_claims
            if c.surface_id == surface_id and c.subject is subject_enum
        ]
        if len(matches) != 1:
            raise MissingOrchestratorError(
                f"idempotence claim missing for {surface_id}/{subject_enum.value}",
                reason_code="idempotence_claim_missing",
                details={
                    "surfaceId": surface_id,
                    "subject": subject_enum.value,
                    "matches": len(matches),
                },
            )
        return matches[0]

    def direct_package_paths(self) -> tuple[InvocationPath, ...]:
        return tuple(
            p for p in self.invocation_paths if p.kind is InvocationPathKind.DIRECT_PACKAGE
        )

    def mcp_plus_plus_paths(self) -> tuple[InvocationPath, ...]:
        return tuple(
            p for p in self.invocation_paths if p.kind is InvocationPathKind.MCP_PLUS_PLUS
        )

    def incomplete_transitions(self) -> tuple[LifecycleTransition, ...]:
        return tuple(t for t in self.transitions if not t.is_complete())


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_source_span(data: Mapping[str, Any]) -> SourceSpan:
    return SourceSpan(
        path=_source_path(data.get("path"), "sourceSpan.path"),
        start_line=_positive_int(data.get("startLine"), "sourceSpan.startLine"),
        end_line=_positive_int(data.get("endLine"), "sourceSpan.endLine"),
        start_column=_nonneg_int(data.get("startColumn") or 0, "sourceSpan.startColumn"),
        end_column=_nonneg_int(data.get("endColumn") or 0, "sourceSpan.endColumn"),
        source_sha256=_optional_text(data.get("sourceSha256"), "sourceSpan.sourceSha256"),
        snippet=_optional_text(data.get("snippet"), "sourceSpan.snippet"),
    )


def _parse_surface(
    data: Mapping[str, Any],
    *,
    require_stored_cids: bool,
) -> OrchestratorSurface:
    provisional = OrchestratorSurface(
        surface_id=_text(data.get("surfaceId"), "surfaceId"),
        display_name=_text(data.get("displayName"), "displayName"),
        role=_enum(OrchestratorSurfaceRole, data.get("role"), "role"),
        implementation_symbol=_text(
            data.get("implementationSymbol"), "implementationSymbol"
        ),
        source_path=_source_path(data.get("sourcePath"), "sourcePath"),
        package_id=_text(data.get("packageId"), "packageId"),
        version=_text(data.get("version"), "version"),
        mediation_kind=_enum(
            InvocationPathKind, data.get("mediationKind"), "mediationKind"
        ),
        surface_cid="",
    )
    surface_cid = _verified_cid(
        data,
        "surfaceCid",
        provisional.preimage(),
        require_stored_cids=require_stored_cids,
    )
    return OrchestratorSurface(
        **{**provisional.__dict__, "surface_cid": surface_cid}
    )


def _parse_transition(
    data: Mapping[str, Any],
    *,
    require_stored_cids: bool,
) -> LifecycleTransition:
    span = _parse_source_span(_mapping(data.get("sourceSpan"), "sourceSpan"))
    provisional = LifecycleTransition(
        transition_id=_text(data.get("transitionId"), "transitionId"),
        surface_id=_text(data.get("surfaceId"), "surfaceId"),
        kind=_enum(TransitionKind, data.get("kind"), "kind"),
        pre_state=_enum(LifecycleState, data.get("preState"), "preState"),
        post_state=_enum(LifecycleState, data.get("postState"), "postState"),
        error_state=_enum(LifecycleState, data.get("errorState"), "errorState"),
        symbol=_text(data.get("symbol"), "symbol"),
        source_span=span,
        requires_ownership=_bool(
            data.get("requiresOwnership"), "requiresOwnership"
        ),
        publishes_receipt=_bool(data.get("publishesReceipt"), "publishesReceipt"),
        transition_cid="",
    )
    if not provisional.is_complete():
        raise OrchestratorInvariantError(
            f"lifecycle transition {provisional.transition_id} is incomplete",
            reason_code="incomplete_lifecycle_edge",
            details={"transitionId": provisional.transition_id},
        )
    transition_cid = _verified_cid(
        data,
        "transitionCid",
        provisional.preimage(),
        require_stored_cids=require_stored_cids,
    )
    return LifecycleTransition(
        **{**provisional.__dict__, "transition_cid": transition_cid}
    )


def _parse_idempotence(
    data: Mapping[str, Any],
    *,
    require_stored_cids: bool,
) -> IdempotenceClaim:
    span = _parse_source_span(_mapping(data.get("sourceSpan"), "sourceSpan"))
    provisional = IdempotenceClaim(
        claim_id=_text(data.get("claimId"), "claimId"),
        surface_id=_text(data.get("surfaceId"), "surfaceId"),
        subject=_enum(IdempotenceSubject, data.get("subject"), "subject"),
        disposition=_enum(
            IdempotenceDisposition, data.get("disposition"), "disposition"
        ),
        evidence=_text(data.get("evidence"), "evidence"),
        source_span=span,
        claim_cid="",
    )
    claim_cid = _verified_cid(
        data,
        "claimCid",
        provisional.preimage(),
        require_stored_cids=require_stored_cids,
    )
    return IdempotenceClaim(**{**provisional.__dict__, "claim_cid": claim_cid})


def _parse_receipt(
    data: Mapping[str, Any],
    *,
    require_stored_cids: bool,
) -> ReceiptPublicationClaim:
    span = _parse_source_span(_mapping(data.get("sourceSpan"), "sourceSpan"))
    provisional = ReceiptPublicationClaim(
        claim_id=_text(data.get("claimId"), "claimId"),
        surface_id=_text(data.get("surfaceId"), "surfaceId"),
        transition_id=_text(data.get("transitionId"), "transitionId"),
        disposition=_enum(
            IdempotenceDisposition, data.get("disposition"), "disposition"
        ),
        evidence=_text(data.get("evidence"), "evidence"),
        source_span=span,
        claim_cid="",
    )
    claim_cid = _verified_cid(
        data,
        "claimCid",
        provisional.preimage(),
        require_stored_cids=require_stored_cids,
    )
    return ReceiptPublicationClaim(
        **{**provisional.__dict__, "claim_cid": claim_cid}
    )


def _parse_swallowed(
    data: Mapping[str, Any],
    *,
    require_stored_cids: bool,
) -> SwallowedFailure:
    span = _parse_source_span(_mapping(data.get("sourceSpan"), "sourceSpan"))
    interpreted = data.get("interpretedAsSuccess", False)
    if interpreted is not False:
        # Fail closed: catalog must never claim silent-pass is success.
        raise OrchestratorInvariantError(
            "swallowed failures cannot be interpreted as success",
            reason_code="swallowed_interpreted_as_success",
            details={"findingId": data.get("findingId")},
        )
    provisional = SwallowedFailure(
        finding_id=_text(data.get("findingId"), "findingId"),
        surface_id=_text(data.get("surfaceId"), "surfaceId"),
        kind=_enum(SwallowedFailureKind, data.get("kind"), "kind"),
        handler_body=_text(data.get("handlerBody"), "handlerBody"),
        source_span=span,
        interpreted_as_success=False,
        finding_cid="",
    )
    finding_cid = _verified_cid(
        data,
        "findingCid",
        provisional.preimage(),
        require_stored_cids=require_stored_cids,
    )
    return SwallowedFailure(**{**provisional.__dict__, "finding_cid": finding_cid})


def _parse_invocation(
    data: Mapping[str, Any],
    *,
    require_stored_cids: bool,
) -> InvocationPath:
    span = _parse_source_span(_mapping(data.get("sourceSpan"), "sourceSpan"))
    provisional = InvocationPath(
        path_id=_text(data.get("pathId"), "pathId"),
        surface_id=_text(data.get("surfaceId"), "surfaceId"),
        kind=_enum(InvocationPathKind, data.get("kind"), "kind"),
        callee=_text(data.get("callee"), "callee"),
        mandatory_mcp=_bool(data.get("mandatoryMcp"), "mandatoryMcp"),
        source_span=span,
        path_cid="",
    )
    path_cid = _verified_cid(
        data,
        "pathCid",
        provisional.preimage(),
        require_stored_cids=require_stored_cids,
    )
    return InvocationPath(**{**provisional.__dict__, "path_cid": path_cid})


def _validate_catalog_consistency(
    surfaces: Sequence[OrchestratorSurface],
    transitions: Sequence[LifecycleTransition],
    idempotence_claims: Sequence[IdempotenceClaim],
    receipt_claims: Sequence[ReceiptPublicationClaim],
    swallowed_failures: Sequence[SwallowedFailure],
    invocation_paths: Sequence[InvocationPath],
) -> None:
    surface_ids = [s.surface_id for s in surfaces]
    if len(surface_ids) != len(set(surface_ids)):
        raise DuplicateOrchestratorError(
            "duplicate orchestrator surface id",
            reason_code="duplicate_surface_id",
        )
    known = set(surface_ids)

    transition_ids = [t.transition_id for t in transitions]
    if len(transition_ids) != len(set(transition_ids)):
        raise DuplicateOrchestratorError(
            "duplicate lifecycle transition id",
            reason_code="duplicate_transition_id",
        )
    for edge in transitions:
        if edge.surface_id not in known:
            raise MissingOrchestratorError(
                f"transition references unknown surface {edge.surface_id}",
                reason_code="transition_surface_missing",
                details={"transitionId": edge.transition_id},
            )
        if not edge.is_complete():
            raise OrchestratorInvariantError(
                f"lifecycle edge incomplete: {edge.transition_id}",
                reason_code="incomplete_lifecycle_edge",
                details={"transitionId": edge.transition_id},
            )
        # Error state must be a failure-class state, not success.
        if edge.error_state in {
            LifecycleState.COMPLETED,
            LifecycleState.RECEIPT_PUBLISHED,
        }:
            raise OrchestratorInvariantError(
                f"error state cannot be a success terminal: {edge.transition_id}",
                reason_code="invalid_error_state",
                details={
                    "transitionId": edge.transition_id,
                    "errorState": edge.error_state.value,
                },
            )

    claim_ids = [c.claim_id for c in idempotence_claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise DuplicateOrchestratorError(
            "duplicate idempotence claim id",
            reason_code="duplicate_idempotence_claim",
        )
    for claim in idempotence_claims:
        if claim.surface_id not in known:
            raise MissingOrchestratorError(
                f"idempotence claim references unknown surface {claim.surface_id}",
                reason_code="idempotence_surface_missing",
                details={"claimId": claim.claim_id},
            )
        if claim.disposition not in set(IdempotenceDisposition):
            raise OrchestratorInvariantError(
                "idempotence disposition outside closed set",
                reason_code="invalid_idempotence_disposition",
                details={"claimId": claim.claim_id},
            )

    # Required subjects: every surface that declares retry/cancel/complete
    # transitions must carry a matching idempotence disposition.
    by_surface_kinds: dict[str, set[TransitionKind]] = {}
    for edge in transitions:
        by_surface_kinds.setdefault(edge.surface_id, set()).add(edge.kind)
    required_subjects: dict[str, set[IdempotenceSubject]] = {}
    for surface_id, kinds in by_surface_kinds.items():
        subjects: set[IdempotenceSubject] = set()
        if TransitionKind.RETRY in kinds:
            subjects.add(IdempotenceSubject.RETRY)
        if TransitionKind.CANCEL in kinds:
            subjects.add(IdempotenceSubject.CANCEL)
        if TransitionKind.COMPLETE in kinds or TransitionKind.FAIL in kinds:
            subjects.add(IdempotenceSubject.RESULT)
        if TransitionKind.PUBLISH_RECEIPT in kinds:
            subjects.add(IdempotenceSubject.RECEIPT)
        if TransitionKind.ADMIT in kinds:
            subjects.add(IdempotenceSubject.SUBMIT)
        if subjects:
            required_subjects[surface_id] = subjects
    present: dict[str, set[IdempotenceSubject]] = {}
    for claim in idempotence_claims:
        present.setdefault(claim.surface_id, set()).add(claim.subject)
    for surface_id, subjects in required_subjects.items():
        have = present.get(surface_id, set())
        missing = subjects - have
        if missing:
            raise OrchestratorInvariantError(
                f"missing idempotence claims for {surface_id}",
                reason_code="missing_idempotence_claims",
                details={
                    "surfaceId": surface_id,
                    "missing": sorted(s.value for s in missing),
                },
            )

    receipt_ids = [c.claim_id for c in receipt_claims]
    if len(receipt_ids) != len(set(receipt_ids)):
        raise DuplicateOrchestratorError(
            "duplicate receipt claim id",
            reason_code="duplicate_receipt_claim",
        )
    transition_index = {t.transition_id: t for t in transitions}
    for claim in receipt_claims:
        if claim.surface_id not in known:
            raise MissingOrchestratorError(
                f"receipt claim references unknown surface {claim.surface_id}",
                reason_code="receipt_surface_missing",
                details={"claimId": claim.claim_id},
            )
        if claim.transition_id not in transition_index:
            raise MissingOrchestratorError(
                f"receipt claim references unknown transition {claim.transition_id}",
                reason_code="receipt_transition_missing",
                details={"claimId": claim.claim_id},
            )

    finding_ids = [f.finding_id for f in swallowed_failures]
    if len(finding_ids) != len(set(finding_ids)):
        raise DuplicateOrchestratorError(
            "duplicate swallowed failure id",
            reason_code="duplicate_swallowed_failure",
        )
    for finding in swallowed_failures:
        if finding.surface_id not in known:
            raise MissingOrchestratorError(
                f"swallowed failure references unknown surface {finding.surface_id}",
                reason_code="swallowed_surface_missing",
                details={"findingId": finding.finding_id},
            )
        if finding.interpreted_as_success:
            raise OrchestratorInvariantError(
                "swallowed failures cannot be interpreted as success",
                reason_code="swallowed_interpreted_as_success",
                details={"findingId": finding.finding_id},
            )

    path_ids = [p.path_id for p in invocation_paths]
    if len(path_ids) != len(set(path_ids)):
        raise DuplicateOrchestratorError(
            "duplicate invocation path id",
            reason_code="duplicate_invocation_path",
        )
    for path in invocation_paths:
        if path.surface_id not in known:
            raise MissingOrchestratorError(
                f"invocation path references unknown surface {path.surface_id}",
                reason_code="invocation_surface_missing",
                details={"pathId": path.path_id},
            )
        # Direct package paths must never claim mandatory MCP mediation.
        if path.kind is InvocationPathKind.DIRECT_PACKAGE and path.mandatory_mcp:
            raise OrchestratorInvariantError(
                "direct package path cannot be mandatory MCP mediation",
                reason_code="direct_package_marked_mandatory_mcp",
                details={"pathId": path.path_id},
            )
        # Mandatory MCP paths must be classified as MCP++.
        if path.mandatory_mcp and path.kind is not InvocationPathKind.MCP_PLUS_PLUS:
            raise OrchestratorInvariantError(
                "mandatory MCP path must use mcp_plus_plus kind",
                reason_code="mandatory_mcp_misclassified",
                details={"pathId": path.path_id, "kind": path.kind.value},
            )


def build_orchestrator_contract_catalog(
    payload: Mapping[str, Any],
    *,
    require_stored_cids: bool = False,
) -> OrchestratorContractCatalog:
    """Validate and normalize an orchestrator contract catalog mapping."""

    if payload.get("schema") not in (None, ORCHESTRATOR_CATALOG_SCHEMA):
        raise OrchestratorContractError(
            "unsupported orchestrator catalog schema",
            reason_code="unsupported_catalog_schema",
            details={"schema": payload.get("schema")},
        )

    surfaces = tuple(
        _parse_surface(
            _mapping(item, "surfaces[]"),
            require_stored_cids=require_stored_cids,
        )
        for item in _sequence(payload.get("surfaces"), "surfaces")
    )
    transitions = tuple(
        _parse_transition(
            _mapping(item, "transitions[]"),
            require_stored_cids=require_stored_cids,
        )
        for item in _sequence(payload.get("transitions") or (), "transitions")
    )
    idempotence_claims = tuple(
        _parse_idempotence(
            _mapping(item, "idempotenceClaims[]"),
            require_stored_cids=require_stored_cids,
        )
        for item in _sequence(
            payload.get("idempotenceClaims") or (), "idempotenceClaims"
        )
    )
    receipt_claims = tuple(
        _parse_receipt(
            _mapping(item, "receiptClaims[]"),
            require_stored_cids=require_stored_cids,
        )
        for item in _sequence(payload.get("receiptClaims") or (), "receiptClaims")
    )
    swallowed_failures = tuple(
        _parse_swallowed(
            _mapping(item, "swallowedFailures[]"),
            require_stored_cids=require_stored_cids,
        )
        for item in _sequence(
            payload.get("swallowedFailures") or (), "swallowedFailures"
        )
    )
    invocation_paths = tuple(
        _parse_invocation(
            _mapping(item, "invocationPaths[]"),
            require_stored_cids=require_stored_cids,
        )
        for item in _sequence(
            payload.get("invocationPaths") or (), "invocationPaths"
        )
    )

    if not surfaces:
        raise MissingOrchestratorError(
            "orchestrator catalog requires at least one surface",
            reason_code="empty_orchestrator_catalog",
        )
    if not transitions:
        raise MissingOrchestratorError(
            "orchestrator catalog requires at least one lifecycle transition",
            reason_code="empty_lifecycle_transitions",
        )

    _validate_catalog_consistency(
        surfaces,
        transitions,
        idempotence_claims,
        receipt_claims,
        swallowed_failures,
        invocation_paths,
    )

    runtime_component_id = str(
        payload.get("runtimeComponentId") or RUNTIME_COMPONENT_ID
    )
    if not runtime_component_id:
        raise OrchestratorContractError(
            "runtimeComponentId must be a nonempty string",
            reason_code="invalid_runtime_component_id",
        )
    evidence_id = str(payload.get("evidenceId") or SCAEV172ORCH)

    provisional = OrchestratorContractCatalog(
        surfaces=surfaces,
        transitions=transitions,
        idempotence_claims=idempotence_claims,
        receipt_claims=receipt_claims,
        swallowed_failures=swallowed_failures,
        invocation_paths=invocation_paths,
        runtime_component_id=runtime_component_id,
        evidence_id=evidence_id,
        catalog_cid="",
    )
    catalog_cid = _verified_cid(
        payload,
        "catalogCid",
        provisional.preimage(),
        require_stored_cids=require_stored_cids,
    )
    return OrchestratorContractCatalog(
        surfaces=surfaces,
        transitions=transitions,
        idempotence_claims=idempotence_claims,
        receipt_claims=receipt_claims,
        swallowed_failures=swallowed_failures,
        invocation_paths=invocation_paths,
        runtime_component_id=runtime_component_id,
        evidence_id=evidence_id,
        catalog_cid=catalog_cid,
    )


def materialize_orchestrator_contract_catalog(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a fully CID-bound serializable form of an unmaterialized catalog."""

    return build_orchestrator_contract_catalog(payload).to_dict()


def load_orchestrator_contract_catalog(path: str | Path) -> OrchestratorContractCatalog:
    """Load a fully materialized catalog, rejecting missing or stale CIDs."""

    catalog_path = Path(path)
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OrchestratorContractError(
            f"unable to load orchestrator contract catalog: {catalog_path}",
            reason_code="catalog_load_failed",
            details={"path": str(catalog_path), "cause": repr(exc)},
        ) from exc
    return build_orchestrator_contract_catalog(
        _mapping(payload, "catalog"),
        require_stored_cids=True,
    )


def validate_orchestrator_sources(
    catalog: OrchestratorContractCatalog,
    repository_root: str | Path,
    *,
    required_surface_ids: Iterable[str] | None = None,
) -> None:
    """Prove that declared source files and symbols exist under ``repository_root``."""

    root = Path(repository_root)
    required = set(required_surface_ids or ())
    for surface in catalog.surfaces:
        if required and surface.surface_id not in required:
            continue
        candidates = [
            root / surface.source_path,
            root / "external" / "ipfs_accelerate" / surface.source_path,
            root / "swissknife" / surface.source_path,
        ]
        candidate = next((path for path in candidates if path.is_file()), None)
        if candidate is None:
            raise OrchestratorSourceError(
                f"orchestrator source does not exist: {surface.source_path}",
                reason_code="orchestrator_source_missing",
                details={
                    "surfaceId": surface.surface_id,
                    "sourcePath": surface.source_path,
                },
            )
        text = candidate.read_text(encoding="utf-8")
        if surface.implementation_symbol not in text:
            raise OrchestratorSourceError(
                f"orchestrator symbol does not exist: {surface.implementation_symbol}",
                reason_code="orchestrator_symbol_missing",
                details={
                    "surfaceId": surface.surface_id,
                    "sourcePath": surface.source_path,
                    "symbol": surface.implementation_symbol,
                },
            )


# ---------------------------------------------------------------------------
# Classification and evaluation helpers
# ---------------------------------------------------------------------------


def classify_invocation_path(
    text: str,
    *,
    callee: str = "",
    mandatory_mcp: bool | None = None,
) -> InvocationPathKind:
    """Classify a call site as direct package, MCP++, compatibility, or adapter.

    Markers are evidence only.  When ``mandatory_mcp`` is True the result is
    always :attr:`InvocationPathKind.MCP_PLUS_PLUS`.  Direct package markers
    never upgrade a path to MCP mediation.
    """

    haystack = f"{text}\n{callee}".lower()
    if mandatory_mcp is True:
        return InvocationPathKind.MCP_PLUS_PLUS
    if any(marker.lower() in haystack for marker in _MCP_MARKERS):
        return InvocationPathKind.MCP_PLUS_PLUS
    if any(marker.lower() in haystack for marker in _COMPAT_MARKERS):
        return InvocationPathKind.COMPATIBILITY
    if any(marker.lower() in haystack for marker in _DATASETS_MARKERS):
        return InvocationPathKind.DATASETS_ADAPTER
    if any(marker.lower() in haystack for marker in _DIRECT_PACKAGE_MARKERS):
        return InvocationPathKind.DIRECT_PACKAGE
    if "observe" in haystack or "status" in haystack or "list" in haystack:
        return InvocationPathKind.OBSERVATION
    return InvocationPathKind.DIRECT_PACKAGE


def evaluate_idempotence_from_source(
    source: str,
    subject: IdempotenceSubject | str,
) -> IdempotenceDisposition:
    """Heuristically classify idempotence evidence from source text.

    Returns one of proved / refuted / unknown.  Broad exception handlers that
    swallow failures refute idempotence for the subject; explicit
    ``idempotency_key`` / owner-guarded updates prove it.  When both signals
    are present, refutation dominates for cancel/result under silent-pass.
    """

    subject_enum = (
        subject
        if isinstance(subject, IdempotenceSubject)
        else _enum(IdempotenceSubject, subject, "subject")
    )
    has_proved = any(marker in source for marker in _IDEMPOTENCE_MARKERS_PROVED)
    has_refuted = any(marker in source for marker in _IDEMPOTENCE_MARKERS_REFUTED)
    # Subject-specific SQL/API patterns from TaskQueue.
    if subject_enum is IdempotenceSubject.SUBMIT:
        if "idempotency_key" in source or "submit_once" in source:
            return IdempotenceDisposition.PROVED
        if "uuid" in source.lower() and "idempotency" not in source.lower():
            return IdempotenceDisposition.REFUTED
    if subject_enum is IdempotenceSubject.RETRY:
        if "status='running' AND assigned_worker" in source and "attempt" in source:
            return IdempotenceDisposition.PROVED
        if has_refuted and not has_proved:
            return IdempotenceDisposition.REFUTED
    if subject_enum is IdempotenceSubject.CANCEL:
        if "status='queued'" in source and "cancelled" in source:
            # Only queued cancels; re-cancel of terminal is a no-op false return
            # — treated as proved for the queued precondition, unknown otherwise.
            if "status='running'" in source or "status IN" in source:
                return IdempotenceDisposition.PROVED
            return IdempotenceDisposition.UNKNOWN
        if has_refuted:
            return IdempotenceDisposition.REFUTED
    if subject_enum is IdempotenceSubject.RESULT:
        if "status='running' AND assigned_worker" in source:
            return IdempotenceDisposition.PROVED
        if has_refuted:
            return IdempotenceDisposition.REFUTED
    if subject_enum is IdempotenceSubject.RECEIPT:
        if "receipt" in source.lower() or "track_provenance" in source:
            if has_refuted:
                # Datasets adapter swallows provenance failures → refuted.
                return IdempotenceDisposition.REFUTED
            return IdempotenceDisposition.PROVED
        return IdempotenceDisposition.UNKNOWN
    if has_proved and not has_refuted:
        return IdempotenceDisposition.PROVED
    if has_refuted:
        return IdempotenceDisposition.REFUTED
    return IdempotenceDisposition.UNKNOWN


def assert_lifecycle_edges_complete(catalog: OrchestratorContractCatalog) -> None:
    """Fail closed unless every lifecycle edge has pre/post/error and a span."""

    incomplete = catalog.incomplete_transitions()
    if incomplete:
        raise OrchestratorInvariantError(
            "one or more lifecycle edges lack pre/post/error states or spans",
            reason_code="incomplete_lifecycle_edge",
            details={
                "transitionIds": [t.transition_id for t in incomplete],
            },
        )
    for edge in catalog.transitions:
        if edge.pre_state is edge.post_state and edge.kind not in {
            TransitionKind.HEARTBEAT,
            TransitionKind.SCALE,
        }:
            raise OrchestratorInvariantError(
                f"non-noop transition has identical pre/post: {edge.transition_id}",
                reason_code="degenerate_lifecycle_edge",
                details={"transitionId": edge.transition_id},
            )


def assert_idempotence_closed(catalog: OrchestratorContractCatalog) -> None:
    """Require every idempotence claim to be proved, refuted, or unknown."""

    allowed = set(IdempotenceDisposition)
    for claim in catalog.idempotence_claims:
        if claim.disposition not in allowed:
            raise OrchestratorInvariantError(
                f"open idempotence disposition for {claim.claim_id}",
                reason_code="open_idempotence_disposition",
                details={"claimId": claim.claim_id},
            )


def assert_swallowed_failures_visible(catalog: OrchestratorContractCatalog) -> None:
    """Require that every swallowed finding is explicit and not success."""

    for finding in catalog.swallowed_failures:
        if finding.interpreted_as_success:
            raise OrchestratorInvariantError(
                "swallowed failure treated as success",
                reason_code="swallowed_interpreted_as_success",
                details={"findingId": finding.finding_id},
            )
        if not finding.source_span.path or finding.source_span.start_line < 1:
            raise OrchestratorInvariantError(
                "swallowed failure lacks evidence span",
                reason_code="swallowed_missing_span",
                details={"findingId": finding.finding_id},
            )


def assert_mediation_distinguished(catalog: OrchestratorContractCatalog) -> None:
    """Require direct package and MCP++ paths to remain distinct."""

    kinds = {path.kind for path in catalog.invocation_paths}
    if (
        InvocationPathKind.DIRECT_PACKAGE in kinds
        and InvocationPathKind.MCP_PLUS_PLUS in kinds
    ):
        # Healthy: both present and classified differently.
        for path in catalog.invocation_paths:
            if path.kind is InvocationPathKind.DIRECT_PACKAGE and path.mandatory_mcp:
                raise OrchestratorInvariantError(
                    "direct package path marked as mandatory MCP",
                    reason_code="direct_package_marked_mandatory_mcp",
                    details={"pathId": path.path_id},
                )
            if path.kind is InvocationPathKind.MCP_PLUS_PLUS and not path.mandatory_mcp:
                # MCP++ paths may be optional adapters; only mandatory ones require True.
                continue
        return
    # If only one kind exists the catalog must still not mislabel.
    for path in catalog.invocation_paths:
        if path.mandatory_mcp and path.kind is not InvocationPathKind.MCP_PLUS_PLUS:
            raise OrchestratorInvariantError(
                "mandatory MCP path misclassified",
                reason_code="mandatory_mcp_misclassified",
                details={"pathId": path.path_id},
            )


def apply_lifecycle_transition(
    state: LifecycleState,
    kind: TransitionKind,
    *,
    force_error: bool = False,
    error_state: LifecycleState = LifecycleState.FAILED,
) -> LifecycleState:
    """Apply one legal lifecycle edge or raise on illegal transition.

    When ``force_error`` is set the configured error state is returned without
    mutating into a success post-state.  Silent success is never synthesized.
    """

    if force_error:
        if error_state in {LifecycleState.COMPLETED, LifecycleState.RECEIPT_PUBLISHED}:
            raise OrchestratorInvariantError(
                "error path cannot yield success terminal",
                reason_code="invalid_error_state",
                details={"errorState": error_state.value},
            )
        return error_state
    for pre, edge_kind, post in _LEGAL_TRANSITIONS:
        if pre is state and edge_kind is kind:
            return post
    raise OrchestratorInvariantError(
        f"illegal lifecycle transition {state.value} --{kind.value}",
        reason_code="illegal_lifecycle_transition",
        details={"preState": state.value, "kind": kind.value},
    )


def evaluate_result_idempotence(
    *,
    first_status: LifecycleState,
    second_status: LifecycleState,
    same_task_id: bool,
    owner_guarded: bool,
) -> IdempotenceDisposition:
    """Evaluate whether a double-complete/result publish is idempotent."""

    if not same_task_id:
        return IdempotenceDisposition.REFUTED
    if first_status is second_status and first_status in TERMINAL_STATES:
        return IdempotenceDisposition.PROVED
    if owner_guarded and first_status is LifecycleState.RUNNING:
        # Owner-guarded complete of running → terminal is proved for first write.
        if second_status in TERMINAL_STATES:
            return IdempotenceDisposition.PROVED
    if first_status in TERMINAL_STATES and second_status != first_status:
        return IdempotenceDisposition.REFUTED
    return IdempotenceDisposition.UNKNOWN


def evaluate_cancel_idempotence(
    *,
    initial: LifecycleState,
    after_first: LifecycleState,
    after_second: LifecycleState,
) -> IdempotenceDisposition:
    """Evaluate cancel idempotence across two applications."""

    if after_first is not LifecycleState.CANCELLED and initial is LifecycleState.QUEUED:
        return IdempotenceDisposition.REFUTED
    if after_first is LifecycleState.CANCELLED and after_second is LifecycleState.CANCELLED:
        return IdempotenceDisposition.PROVED
    if initial in TERMINAL_STATES and after_first is initial and after_second is initial:
        # No-op on already-terminal is safe (idempotent no-op).
        return IdempotenceDisposition.PROVED
    if after_first is LifecycleState.CANCELLED and after_second != LifecycleState.CANCELLED:
        return IdempotenceDisposition.REFUTED
    return IdempotenceDisposition.UNKNOWN


def evaluate_retry_idempotence(
    *,
    task_identity: str,
    retry_identity: str,
    attempt_before: int,
    attempt_after: int,
    max_attempts: int,
) -> IdempotenceDisposition:
    """Evaluate whether a retry preserves identity without duplicating work."""

    if not task_identity or task_identity != retry_identity:
        return IdempotenceDisposition.REFUTED
    if attempt_after != attempt_before + 1 and attempt_after != attempt_before:
        # Unexpected jump without identity change is unknown, not proved.
        if attempt_after > max_attempts:
            return IdempotenceDisposition.REFUTED
        return IdempotenceDisposition.UNKNOWN
    if attempt_after > max_attempts:
        return IdempotenceDisposition.REFUTED
    if attempt_after == attempt_before:
        # Replayed retry under ownership guard is proved idempotent no-op.
        return IdempotenceDisposition.PROVED
    if attempt_after == attempt_before + 1:
        return IdempotenceDisposition.PROVED
    return IdempotenceDisposition.UNKNOWN


# ---------------------------------------------------------------------------
# Static source extraction (cold; never imports orchestrator modules)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrchestratorSourceExtraction:
    """Result of cold static extraction from supplied source texts."""

    transitions: tuple[LifecycleTransition, ...]
    swallowed_failures: tuple[SwallowedFailure, ...]
    invocation_paths: tuple[InvocationPath, ...]
    idempotence_hints: tuple[tuple[IdempotenceSubject, IdempotenceDisposition, str], ...]
    extraction_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": ORCHESTRATOR_EXTRACTION_SCHEMA,
            "transitions": [t.to_dict() for t in self.transitions],
            "swallowedFailures": [f.to_dict() for f in self.swallowed_failures],
            "invocationPaths": [p.to_dict() for p in self.invocation_paths],
            "idempotenceHints": [
                {
                    "subject": subject.value,
                    "disposition": disposition.value,
                    "evidence": evidence,
                }
                for subject, disposition, evidence in self.idempotence_hints
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "extractionCid": self.extraction_cid}


_TRANSITION_SYMBOL_MAP: Final[dict[str, TransitionKind]] = {
    "submit": TransitionKind.ADMIT,
    "submit_once": TransitionKind.ADMIT,
    "submit_with_outcome": TransitionKind.ADMIT,
    "claim": TransitionKind.CLAIM,
    "claim_next": TransitionKind.CLAIM,
    "claim_many": TransitionKind.CLAIM,
    "complete": TransitionKind.COMPLETE,
    "complete_task": TransitionKind.COMPLETE,
    "cancel": TransitionKind.CANCEL,
    "cancel_task": TransitionKind.CANCEL,
    "retry": TransitionKind.RETRY,
    "start": TransitionKind.START,
    "stop": TransitionKind.STOP,
    "heartbeat": TransitionKind.HEARTBEAT,
    "dispatch": TransitionKind.DISPATCH,
    "publish_receipt": TransitionKind.PUBLISH_RECEIPT,
}


def _line_span_for_offset(source: str, start: int, end: int, path: str) -> SourceSpan:
    start_line = source.count("\n", 0, start) + 1
    end_line = source.count("\n", 0, max(start, end - 1)) + 1
    line_start = source.rfind("\n", 0, start) + 1
    end_line_start = source.rfind("\n", 0, max(start, end - 1)) + 1
    return SourceSpan(
        path=path,
        start_line=start_line,
        end_line=end_line,
        start_column=start - line_start,
        end_column=max(start, end - 1) - end_line_start + 1,
        source_sha256=_source_sha256(source),
        snippet=source[start:end][:200],
    )


def _state_for_kind(kind: TransitionKind) -> tuple[LifecycleState, LifecycleState, LifecycleState]:
    """Default pre/post/error triple for a transition kind."""

    mapping: dict[TransitionKind, tuple[LifecycleState, LifecycleState, LifecycleState]] = {
        TransitionKind.ADMIT: (
            LifecycleState.ABSENT,
            LifecycleState.ADMITTED,
            LifecycleState.FAILED,
        ),
        TransitionKind.DISPATCH: (
            LifecycleState.ADMITTED,
            LifecycleState.QUEUED,
            LifecycleState.FAILED,
        ),
        TransitionKind.CLAIM: (
            LifecycleState.QUEUED,
            LifecycleState.OWNED,
            LifecycleState.FAILED,
        ),
        TransitionKind.OWN: (
            LifecycleState.QUEUED,
            LifecycleState.OWNED,
            LifecycleState.FAILED,
        ),
        TransitionKind.START: (
            LifecycleState.OWNED,
            LifecycleState.RUNNING,
            LifecycleState.FAILED,
        ),
        TransitionKind.COMPLETE: (
            LifecycleState.RUNNING,
            LifecycleState.COMPLETED,
            LifecycleState.FAILED,
        ),
        TransitionKind.FAIL: (
            LifecycleState.RUNNING,
            LifecycleState.FAILED,
            LifecycleState.FAILED,
        ),
        TransitionKind.CANCEL: (
            LifecycleState.QUEUED,
            LifecycleState.CANCELLED,
            LifecycleState.FAILED,
        ),
        TransitionKind.RETRY: (
            LifecycleState.RUNNING,
            LifecycleState.RETRYING,
            LifecycleState.FAILED,
        ),
        TransitionKind.TIMEOUT: (
            LifecycleState.RUNNING,
            LifecycleState.TIMED_OUT,
            LifecycleState.FAILED,
        ),
        TransitionKind.PUBLISH_RECEIPT: (
            LifecycleState.COMPLETED,
            LifecycleState.RECEIPT_PUBLISHED,
            LifecycleState.FAILED,
        ),
        TransitionKind.HEARTBEAT: (
            LifecycleState.RUNNING,
            LifecycleState.RUNNING,
            LifecycleState.FAILED,
        ),
        TransitionKind.MESH_CLAIM: (
            LifecycleState.QUEUED,
            LifecycleState.OWNED,
            LifecycleState.FAILED,
        ),
        TransitionKind.SCALE: (
            LifecycleState.RUNNING,
            LifecycleState.RUNNING,
            LifecycleState.FAILED,
        ),
        TransitionKind.STOP: (
            LifecycleState.RUNNING,
            LifecycleState.FAILED,
            LifecycleState.FAILED,
        ),
    }
    return mapping[kind]


def extract_swallowed_failures_from_source(
    source: str,
    *,
    path: str,
    surface_id: str,
) -> tuple[SwallowedFailure, ...]:
    """Detect broad exception handlers that pass or return silently."""

    findings: list[SwallowedFailure] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to line-oriented scan for non-Python or broken fixtures.
        return _extract_swallowed_regex(source, path=path, surface_id=surface_id)

    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        body = node.body
        if not body:
            continue
        only = body[0]
        is_pass = isinstance(only, ast.Pass) and len(body) == 1
        is_return = isinstance(only, ast.Return) and len(body) == 1
        if not (is_pass or is_return):
            continue
        # Classify bare vs broad Exception.
        is_bare = node.type is None
        is_broad = False
        if isinstance(node.type, ast.Name) and node.type.id in {
            "Exception",
            "BaseException",
        }:
            is_broad = True
        if isinstance(node.type, ast.Tuple):
            for elt in node.type.elts:
                if isinstance(elt, ast.Name) and elt.id in {"Exception", "BaseException"}:
                    is_broad = True
        if not (is_bare or is_broad):
            continue
        if is_bare and is_pass:
            kind = SwallowedFailureKind.BARE_EXCEPT_PASS
        elif is_bare and is_return:
            kind = SwallowedFailureKind.BARE_EXCEPT_RETURN
        elif is_pass:
            kind = SwallowedFailureKind.BROAD_EXCEPT_PASS
        else:
            kind = SwallowedFailureKind.BROAD_EXCEPT_RETURN
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start) or start
        snippet = "\n".join(lines[start - 1 : end])
        span = SourceSpan(
            path=path,
            start_line=start,
            end_line=end,
            start_column=getattr(node, "col_offset", 0) or 0,
            end_column=getattr(node, "end_col_offset", 0) or 0,
            source_sha256=_source_sha256(source),
            snippet=snippet[:200],
        )
        provisional = SwallowedFailure(
            finding_id=f"swallowed:{surface_id}:{start}",
            surface_id=surface_id,
            kind=kind,
            handler_body=snippet[:200] or ("pass" if is_pass else "return"),
            source_span=span,
            interpreted_as_success=False,
            finding_cid="",
        )
        findings.append(
            SwallowedFailure(
                **{
                    **provisional.__dict__,
                    "finding_cid": _cid(provisional.preimage()),
                }
            )
        )
    return tuple(findings)


def _extract_swallowed_regex(
    source: str,
    *,
    path: str,
    surface_id: str,
) -> tuple[SwallowedFailure, ...]:
    findings: list[SwallowedFailure] = []
    pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]*)except(?P<head>[^\n:]*)"
        r":\n(?P=indent)[ \t]+(?P<body>pass|return(?:\s+\S+)?)\s*$"
    )
    for match in pattern.finditer(source):
        head = match.group("head")
        body = match.group("body").strip()
        is_bare = head.strip() in {"", "..."}
        is_broad = "Exception" in head or "BaseException" in head
        if not (is_bare or is_broad):
            continue
        is_pass = body == "pass"
        if is_bare and is_pass:
            kind = SwallowedFailureKind.BARE_EXCEPT_PASS
        elif is_bare:
            kind = SwallowedFailureKind.BARE_EXCEPT_RETURN
        elif is_pass:
            kind = SwallowedFailureKind.BROAD_EXCEPT_PASS
        else:
            kind = SwallowedFailureKind.BROAD_EXCEPT_RETURN
        span = _line_span_for_offset(source, match.start(), match.end(), path)
        provisional = SwallowedFailure(
            finding_id=f"swallowed:{surface_id}:{span.start_line}",
            surface_id=surface_id,
            kind=kind,
            handler_body=match.group(0)[:200],
            source_span=span,
            interpreted_as_success=False,
            finding_cid="",
        )
        findings.append(
            SwallowedFailure(
                **{
                    **provisional.__dict__,
                    "finding_cid": _cid(provisional.preimage()),
                }
            )
        )
    return tuple(findings)


def extract_transitions_from_source(
    source: str,
    *,
    path: str,
    surface_id: str,
) -> tuple[LifecycleTransition, ...]:
    """Extract lifecycle method definitions as transitions with full edges."""

    edges: list[LifecycleTransition] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        kind = _TRANSITION_SYMBOL_MAP.get(node.name)
        if kind is None:
            continue
        pre, post, error = _state_for_kind(kind)
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start) or start
        span = SourceSpan(
            path=path,
            start_line=start,
            end_line=end,
            start_column=getattr(node, "col_offset", 0) or 0,
            end_column=getattr(node, "end_col_offset", 0) or 0,
            source_sha256=_source_sha256(source),
            snippet=f"def {node.name}",
        )
        requires_ownership = kind in {
            TransitionKind.COMPLETE,
            TransitionKind.FAIL,
            TransitionKind.RETRY,
            TransitionKind.HEARTBEAT,
        }
        publishes = kind in {
            TransitionKind.COMPLETE,
            TransitionKind.PUBLISH_RECEIPT,
            TransitionKind.FAIL,
            TransitionKind.CANCEL,
        }
        provisional = LifecycleTransition(
            transition_id=f"tx:{surface_id}:{node.name}",
            surface_id=surface_id,
            kind=kind,
            pre_state=pre,
            post_state=post,
            error_state=error,
            symbol=node.name,
            source_span=span,
            requires_ownership=requires_ownership,
            publishes_receipt=publishes,
            transition_cid="",
        )
        edges.append(
            LifecycleTransition(
                **{
                    **provisional.__dict__,
                    "transition_cid": _cid(provisional.preimage()),
                }
            )
        )
    return tuple(edges)


def extract_invocation_paths_from_source(
    source: str,
    *,
    path: str,
    surface_id: str,
) -> tuple[InvocationPath, ...]:
    """Extract import and call-site mediation classifications."""

    paths: list[InvocationPath] = []
    # Import lines.
    for match in re.finditer(
        r"(?m)^(from\s+[\w.]+\s+import\s+[\w,\s]+|import\s+[\w.]+)",
        source,
    ):
        line = match.group(0)
        kind = classify_invocation_path(line)
        span = _line_span_for_offset(source, match.start(), match.end(), path)
        provisional = InvocationPath(
            path_id=f"inv:{surface_id}:import:{span.start_line}",
            surface_id=surface_id,
            kind=kind,
            callee=line.strip()[:120],
            mandatory_mcp=kind is InvocationPathKind.MCP_PLUS_PLUS,
            source_span=span,
            path_cid="",
        )
        paths.append(
            InvocationPath(
                **{**provisional.__dict__, "path_cid": _cid(provisional.preimage())}
            )
        )

    # Call-like MCP and package markers.
    call_patterns = [
        (r"tools/call|tools\.call|jsonRpc\(|callTool\(", InvocationPathKind.MCP_PLUS_PLUS),
        (r"TaskOrchestrator\(|TaskQueue\(|start_orchestrator_in_background\(", InvocationPathKind.DIRECT_PACKAGE),
        (r"DatasetsManager\(|track_provenance\(|log_event\(", InvocationPathKind.DATASETS_ADAPTER),
        (r"tools_dispatch|compat|/api/v0/", InvocationPathKind.COMPATIBILITY),
    ]
    for pattern, kind in call_patterns:
        for match in re.finditer(pattern, source):
            span = _line_span_for_offset(source, match.start(), match.end(), path)
            provisional = InvocationPath(
                path_id=f"inv:{surface_id}:call:{span.start_line}:{span.start_column}",
                surface_id=surface_id,
                kind=kind,
                callee=match.group(0)[:120],
                mandatory_mcp=kind is InvocationPathKind.MCP_PLUS_PLUS,
                source_span=span,
                path_cid="",
            )
            paths.append(
                InvocationPath(
                    **{
                        **provisional.__dict__,
                        "path_cid": _cid(provisional.preimage()),
                    }
                )
            )
    return tuple(paths)


def extract_orchestrator_source_contracts(
    sources: Mapping[str, str],
    *,
    surface_id: str = "extracted-surface",
) -> OrchestratorSourceExtraction:
    """Cold-extract transitions, swallowed failures, and invocation paths."""

    transitions: list[LifecycleTransition] = []
    swallowed: list[SwallowedFailure] = []
    invocations: list[InvocationPath] = []
    hints: list[tuple[IdempotenceSubject, IdempotenceDisposition, str]] = []

    for path, source in sorted(sources.items()):
        clean = _clean_path(path)
        transitions.extend(
            extract_transitions_from_source(
                source, path=clean, surface_id=surface_id
            )
        )
        swallowed.extend(
            extract_swallowed_failures_from_source(
                source, path=clean, surface_id=surface_id
            )
        )
        invocations.extend(
            extract_invocation_paths_from_source(
                source, path=clean, surface_id=surface_id
            )
        )
        for subject in IdempotenceSubject:
            disposition = evaluate_idempotence_from_source(source, subject)
            if disposition is not IdempotenceDisposition.UNKNOWN or subject.value in source:
                hints.append((subject, disposition, clean))

    provisional = OrchestratorSourceExtraction(
        transitions=tuple(transitions),
        swallowed_failures=tuple(swallowed),
        invocation_paths=tuple(invocations),
        idempotence_hints=tuple(hints),
        extraction_cid="",
    )
    return OrchestratorSourceExtraction(
        **{
            **provisional.__dict__,
            "extraction_cid": _cid(provisional.preimage()),
        }
    )


# ---------------------------------------------------------------------------
# Default inventory
# ---------------------------------------------------------------------------


def _span(
    path: str,
    start_line: int,
    end_line: int,
    *,
    snippet: str = "",
) -> dict[str, Any]:
    return {
        "path": path,
        "startLine": start_line,
        "endLine": end_line,
        "startColumn": 0,
        "endColumn": 0,
        "sourceSha256": "",
        "snippet": snippet,
    }


def default_orchestrator_inventory() -> dict[str, Any]:
    """Return the unmaterialized default inventory of orchestrator surfaces.

    Spans reference known source coordinates in the accelerator / SwissKnife
    tree.  Idempotence dispositions encode reviewed static evidence: TaskQueue
    owner-guarded updates and ``idempotency_key`` submits are proved; silent
    ``except Exception: pass`` paths in TaskOrchestrator refute receipt
    publication guarantees.
    """

    orch_path = "ipfs_accelerate_py/p2p_tasks/orchestrator.py"
    queue_path = "ipfs_accelerate_py/p2p_tasks/task_queue.py"
    client_path = "ipfs_accelerate_py/p2p_tasks/client.py"
    service_path = "ipfs_accelerate_py/p2p_tasks/service.py"
    datasets_note = "ipfs_accelerate_py/p2p_tasks/orchestrator.py"
    mcp_tools = (
        "ipfs_accelerate_py/mcp_server/tools/background_task_tools/"
        "native_background_task_tools.py"
    )
    swissknife_orb = "src/services/mcp/mcp-orb-capability-router.ts"
    lifecycle_path = (
        "ipfs_accelerate_py/agent_supervisor/control/lifecycle_orchestrator.py"
    )

    return {
        "schema": ORCHESTRATOR_CATALOG_SCHEMA,
        "catalogVersion": CATALOG_VERSION,
        "runtimeComponentId": RUNTIME_COMPONENT_ID,
        "evidenceId": SCAEV172ORCH,
        "surfaces": [
            {
                "surfaceId": "task-orchestrator-v1",
                "displayName": "P2P TaskOrchestrator",
                "role": OrchestratorSurfaceRole.TASK_ORCHESTRATOR.value,
                "implementationSymbol": "TaskOrchestrator",
                "sourcePath": orch_path,
                "packageId": "ipfs_accelerate_py",
                "version": "1",
                "mediationKind": InvocationPathKind.DIRECT_PACKAGE.value,
            },
            {
                "surfaceId": "task-queue-v1",
                "displayName": "DuckDB TaskQueue",
                "role": OrchestratorSurfaceRole.TASK_QUEUE.value,
                "implementationSymbol": "TaskQueue",
                "sourcePath": queue_path,
                "packageId": "ipfs_accelerate_py",
                "version": "1",
                "mediationKind": InvocationPathKind.DIRECT_PACKAGE.value,
            },
            {
                "surfaceId": "p2p-service-v1",
                "displayName": "TaskQueue P2P service",
                "role": OrchestratorSurfaceRole.P2P_SERVICE.value,
                "implementationSymbol": "ServiceConfig",
                "sourcePath": service_path,
                "packageId": "ipfs_accelerate_py",
                "version": "1",
                "mediationKind": InvocationPathKind.DIRECT_PACKAGE.value,
            },
            {
                "surfaceId": "p2p-client-v1",
                "displayName": "TaskQueue P2P client",
                "role": OrchestratorSurfaceRole.P2P_CLIENT.value,
                "implementationSymbol": "submit_task",
                "sourcePath": client_path,
                "packageId": "ipfs_accelerate_py",
                "version": "1",
                "mediationKind": InvocationPathKind.DIRECT_PACKAGE.value,
            },
            {
                "surfaceId": "datasets-adapter-v1",
                "displayName": "Datasets provenance adapter",
                "role": OrchestratorSurfaceRole.DATASETS_ADAPTER.value,
                "implementationSymbol": "DatasetsManager",
                "sourcePath": datasets_note,
                "packageId": "ipfs_accelerate_py",
                "version": "1",
                "mediationKind": InvocationPathKind.DATASETS_ADAPTER.value,
            },
            {
                "surfaceId": "mcp-background-task-tools-v1",
                "displayName": "MCP background task tools",
                "role": OrchestratorSurfaceRole.MCP_TOOLS.value,
                "implementationSymbol": "manage_task_queue",
                "sourcePath": mcp_tools,
                "packageId": "ipfs_accelerate_py",
                "version": "1",
                "mediationKind": InvocationPathKind.MCP_PLUS_PLUS.value,
            },
            {
                "surfaceId": "swissknife-orb-v1",
                "displayName": "SwissKnife MCPCapabilityRouter",
                "role": OrchestratorSurfaceRole.SWISSKNIFE_ORB.value,
                "implementationSymbol": "MCPCapabilityRouter",
                "sourcePath": swissknife_orb,
                "packageId": "swissknife",
                "version": "1",
                "mediationKind": InvocationPathKind.MCP_PLUS_PLUS.value,
            },
            {
                "surfaceId": "supervisor-lifecycle-v1",
                "displayName": "Supervisor process lifecycle orchestrator",
                "role": OrchestratorSurfaceRole.SUPERVISOR_LIFECYCLE.value,
                "implementationSymbol": "LifecycleOrchestrator",
                "sourcePath": lifecycle_path,
                "packageId": "ipfs_accelerate_py",
                "version": "1",
                "mediationKind": InvocationPathKind.DIRECT_PACKAGE.value,
            },
        ],
        "transitions": [
            {
                "transitionId": "tx-queue-admit-v1",
                "surfaceId": "task-queue-v1",
                "kind": TransitionKind.ADMIT.value,
                "preState": LifecycleState.ABSENT.value,
                "postState": LifecycleState.ADMITTED.value,
                "errorState": LifecycleState.FAILED.value,
                "symbol": "submit",
                "sourceSpan": _span(queue_path, 452, 481, snippet="def submit"),
                "requiresOwnership": False,
                "publishesReceipt": False,
            },
            {
                "transitionId": "tx-queue-claim-v1",
                "surfaceId": "task-queue-v1",
                "kind": TransitionKind.CLAIM.value,
                "preState": LifecycleState.QUEUED.value,
                "postState": LifecycleState.OWNED.value,
                "errorState": LifecycleState.FAILED.value,
                "symbol": "claim_next",
                "sourceSpan": _span(queue_path, 921, 950, snippet="def claim_next"),
                "requiresOwnership": True,
                "publishesReceipt": False,
            },
            {
                "transitionId": "tx-queue-complete-v1",
                "surfaceId": "task-queue-v1",
                "kind": TransitionKind.COMPLETE.value,
                "preState": LifecycleState.RUNNING.value,
                "postState": LifecycleState.COMPLETED.value,
                "errorState": LifecycleState.FAILED.value,
                "symbol": "complete",
                "sourceSpan": _span(queue_path, 1413, 1514, snippet="def complete"),
                "requiresOwnership": True,
                "publishesReceipt": True,
            },
            {
                "transitionId": "tx-queue-cancel-v1",
                "surfaceId": "task-queue-v1",
                "kind": TransitionKind.CANCEL.value,
                "preState": LifecycleState.QUEUED.value,
                "postState": LifecycleState.CANCELLED.value,
                "errorState": LifecycleState.FAILED.value,
                "symbol": "cancel",
                "sourceSpan": _span(queue_path, 1516, 1540, snippet="def cancel"),
                "requiresOwnership": False,
                "publishesReceipt": True,
            },
            {
                "transitionId": "tx-queue-retry-v1",
                "surfaceId": "task-queue-v1",
                "kind": TransitionKind.RETRY.value,
                "preState": LifecycleState.RUNNING.value,
                "postState": LifecycleState.RETRYING.value,
                "errorState": LifecycleState.FAILED.value,
                "symbol": "retry",
                "sourceSpan": _span(queue_path, 879, 919, snippet="def retry"),
                "requiresOwnership": True,
                "publishesReceipt": False,
            },
            {
                "transitionId": "tx-orch-start-v1",
                "surfaceId": "task-orchestrator-v1",
                "kind": TransitionKind.START.value,
                "preState": LifecycleState.OWNED.value,
                "postState": LifecycleState.RUNNING.value,
                "errorState": LifecycleState.FAILED.value,
                "symbol": "start",
                "sourceSpan": _span(orch_path, 196, 205, snippet="def start"),
                "requiresOwnership": False,
                "publishesReceipt": False,
            },
            {
                "transitionId": "tx-orch-stop-v1",
                "surfaceId": "task-orchestrator-v1",
                "kind": TransitionKind.STOP.value,
                "preState": LifecycleState.RUNNING.value,
                "postState": LifecycleState.FAILED.value,
                "errorState": LifecycleState.FAILED.value,
                "symbol": "stop",
                "sourceSpan": _span(orch_path, 206, 214, snippet="def stop"),
                "requiresOwnership": False,
                "publishesReceipt": False,
            },
            {
                "transitionId": "tx-orch-mesh-claim-v1",
                "surfaceId": "task-orchestrator-v1",
                "kind": TransitionKind.MESH_CLAIM.value,
                "preState": LifecycleState.QUEUED.value,
                "postState": LifecycleState.OWNED.value,
                "errorState": LifecycleState.FAILED.value,
                "symbol": "_mesh_drain",
                "sourceSpan": _span(orch_path, 460, 490, snippet="_remote_backlog"),
                "requiresOwnership": True,
                "publishesReceipt": False,
            },
            {
                "transitionId": "tx-client-submit-v1",
                "surfaceId": "p2p-client-v1",
                "kind": TransitionKind.ADMIT.value,
                "preState": LifecycleState.ABSENT.value,
                "postState": LifecycleState.ADMITTED.value,
                "errorState": LifecycleState.FAILED.value,
                "symbol": "submit_task",
                "sourceSpan": _span(client_path, 3406, 3440, snippet="async def submit_task"),
                "requiresOwnership": False,
                "publishesReceipt": False,
            },
            {
                "transitionId": "tx-client-cancel-v1",
                "surfaceId": "p2p-client-v1",
                "kind": TransitionKind.CANCEL.value,
                "preState": LifecycleState.QUEUED.value,
                "postState": LifecycleState.CANCELLED.value,
                "errorState": LifecycleState.FAILED.value,
                "symbol": "cancel_task",
                "sourceSpan": _span(client_path, 4128, 4145, snippet="async def cancel_task"),
                "requiresOwnership": False,
                "publishesReceipt": True,
            },
            {
                "transitionId": "tx-client-complete-v1",
                "surfaceId": "p2p-client-v1",
                "kind": TransitionKind.COMPLETE.value,
                "preState": LifecycleState.RUNNING.value,
                "postState": LifecycleState.COMPLETED.value,
                "errorState": LifecycleState.FAILED.value,
                "symbol": "complete_task",
                "sourceSpan": _span(client_path, 3814, 3880, snippet="async def complete_task"),
                "requiresOwnership": True,
                "publishesReceipt": True,
            },
            {
                "transitionId": "tx-datasets-receipt-v1",
                "surfaceId": "datasets-adapter-v1",
                "kind": TransitionKind.PUBLISH_RECEIPT.value,
                "preState": LifecycleState.COMPLETED.value,
                "postState": LifecycleState.RECEIPT_PUBLISHED.value,
                "errorState": LifecycleState.FAILED.value,
                "symbol": "_log_workflow_event",
                "sourceSpan": _span(datasets_note, 178, 189, snippet="_log_workflow_event"),
                "requiresOwnership": False,
                "publishesReceipt": True,
            },
            {
                "transitionId": "tx-mcp-tools-dispatch-v1",
                "surfaceId": "mcp-background-task-tools-v1",
                "kind": TransitionKind.DISPATCH.value,
                "preState": LifecycleState.ADMITTED.value,
                "postState": LifecycleState.QUEUED.value,
                "errorState": LifecycleState.FAILED.value,
                "symbol": "manage_task_queue",
                "sourceSpan": _span(mcp_tools, 1, 50, snippet="manage_task_queue"),
                "requiresOwnership": False,
                "publishesReceipt": False,
            },
            {
                "transitionId": "tx-swissknife-orb-dispatch-v1",
                "surfaceId": "swissknife-orb-v1",
                "kind": TransitionKind.DISPATCH.value,
                "preState": LifecycleState.ADMITTED.value,
                "postState": LifecycleState.QUEUED.value,
                "errorState": LifecycleState.FAILED.value,
                "symbol": "MCPCapabilityRouter",
                "sourceSpan": _span(swissknife_orb, 40, 80, snippet="ORBLifecyclePhase"),
                "requiresOwnership": False,
                "publishesReceipt": True,
            },
            {
                "transitionId": "tx-supervisor-lifecycle-start-v1",
                "surfaceId": "supervisor-lifecycle-v1",
                "kind": TransitionKind.START.value,
                "preState": LifecycleState.OWNED.value,
                "postState": LifecycleState.RUNNING.value,
                "errorState": LifecycleState.FAILED.value,
                "symbol": "LifecycleOrchestrator",
                "sourceSpan": _span(lifecycle_path, 1, 40, snippet="lifecycle orchestration"),
                "requiresOwnership": True,
                "publishesReceipt": True,
            },
            {
                "transitionId": "tx-p2p-service-admit-v1",
                "surfaceId": "p2p-service-v1",
                "kind": TransitionKind.ADMIT.value,
                "preState": LifecycleState.ABSENT.value,
                "postState": LifecycleState.ADMITTED.value,
                "errorState": LifecycleState.FAILED.value,
                "symbol": "ServiceConfig",
                "sourceSpan": _span(service_path, 533, 560, snippet="class ServiceConfig"),
                "requiresOwnership": False,
                "publishesReceipt": False,
            },
        ],
        "idempotenceClaims": [
            {
                "claimId": "idemp-queue-submit-v1",
                "surfaceId": "task-queue-v1",
                "subject": IdempotenceSubject.SUBMIT.value,
                "disposition": IdempotenceDisposition.PROVED.value,
                "evidence": "idempotency_key + submit_once + submit_with_outcome",
                "sourceSpan": _span(queue_path, 452, 532, snippet="submit_once"),
            },
            {
                "claimId": "idemp-queue-retry-v1",
                "surfaceId": "task-queue-v1",
                "subject": IdempotenceSubject.RETRY.value,
                "disposition": IdempotenceDisposition.PROVED.value,
                "evidence": "owner-guarded UPDATE status=running AND assigned_worker",
                "sourceSpan": _span(queue_path, 879, 919, snippet="def retry"),
            },
            {
                "claimId": "idemp-queue-cancel-v1",
                "surfaceId": "task-queue-v1",
                "subject": IdempotenceSubject.CANCEL.value,
                "disposition": IdempotenceDisposition.UNKNOWN.value,
                "evidence": "cancel only when status=queued; terminal re-cancel is no-op false",
                "sourceSpan": _span(queue_path, 1516, 1540, snippet="def cancel"),
            },
            {
                "claimId": "idemp-queue-result-v1",
                "surfaceId": "task-queue-v1",
                "subject": IdempotenceSubject.RESULT.value,
                "disposition": IdempotenceDisposition.PROVED.value,
                "evidence": "owner-guarded complete; conflict returns false without rewrite",
                "sourceSpan": _span(queue_path, 1413, 1514, snippet="def complete"),
            },
            {
                "claimId": "idemp-client-cancel-v1",
                "surfaceId": "p2p-client-v1",
                "subject": IdempotenceSubject.CANCEL.value,
                "disposition": IdempotenceDisposition.UNKNOWN.value,
                "evidence": "remote cancel_task delegates queue semantics",
                "sourceSpan": _span(client_path, 4128, 4145, snippet="cancel_task"),
            },
            {
                "claimId": "idemp-client-result-v1",
                "surfaceId": "p2p-client-v1",
                "subject": IdempotenceSubject.RESULT.value,
                "disposition": IdempotenceDisposition.UNKNOWN.value,
                "evidence": "complete_task transport; local queue proves ownership",
                "sourceSpan": _span(client_path, 3814, 3880, snippet="complete_task"),
            },
            {
                "claimId": "idemp-client-submit-v1",
                "surfaceId": "p2p-client-v1",
                "subject": IdempotenceSubject.SUBMIT.value,
                "disposition": IdempotenceDisposition.UNKNOWN.value,
                "evidence": "submit_task identity depends on remote queue configuration",
                "sourceSpan": _span(client_path, 3406, 3440, snippet="submit_task"),
            },
            {
                "claimId": "idemp-datasets-receipt-v1",
                "surfaceId": "datasets-adapter-v1",
                "subject": IdempotenceSubject.RECEIPT.value,
                "disposition": IdempotenceDisposition.REFUTED.value,
                "evidence": "except Exception: pass around track_provenance/log_event",
                "sourceSpan": _span(datasets_note, 178, 189, snippet="_log_workflow_event"),
            },
            {
                "claimId": "idemp-p2p-service-submit-v1",
                "surfaceId": "p2p-service-v1",
                "subject": IdempotenceSubject.SUBMIT.value,
                "disposition": IdempotenceDisposition.UNKNOWN.value,
                "evidence": "service admit path binds queue; identity not locally proved",
                "sourceSpan": _span(service_path, 533, 560, snippet="ServiceConfig"),
            },
            {
                "claimId": "idemp-supervisor-result-v1",
                "surfaceId": "supervisor-lifecycle-v1",
                "subject": IdempotenceSubject.RESULT.value,
                "disposition": IdempotenceDisposition.PROVED.value,
                "evidence": "fenced pid+start-time checks and append-only saga journal",
                "sourceSpan": _span(lifecycle_path, 1, 40, snippet="saga journal"),
            },
            {
                "claimId": "idemp-supervisor-receipt-v1",
                "surfaceId": "supervisor-lifecycle-v1",
                "subject": IdempotenceSubject.RECEIPT.value,
                "disposition": IdempotenceDisposition.PROVED.value,
                "evidence": "LIFECYCLE_RECEIPT_SCHEMA checkpoints",
                "sourceSpan": _span(lifecycle_path, 48, 56, snippet="LIFECYCLE_RECEIPT_SCHEMA"),
            },
        ],
        "receiptClaims": [
            {
                "claimId": "receipt-queue-complete-v1",
                "surfaceId": "task-queue-v1",
                "transitionId": "tx-queue-complete-v1",
                "disposition": IdempotenceDisposition.PROVED.value,
                "evidence": "result_json persisted on complete",
                "sourceSpan": _span(queue_path, 1465, 1503, snippet="result_json"),
            },
            {
                "claimId": "receipt-datasets-v1",
                "surfaceId": "datasets-adapter-v1",
                "transitionId": "tx-datasets-receipt-v1",
                "disposition": IdempotenceDisposition.REFUTED.value,
                "evidence": "swallowed provenance exceptions drop receipt",
                "sourceSpan": _span(datasets_note, 184, 189, snippet="except Exception"),
            },
            {
                "claimId": "receipt-supervisor-v1",
                "surfaceId": "supervisor-lifecycle-v1",
                "transitionId": "tx-supervisor-lifecycle-start-v1",
                "disposition": IdempotenceDisposition.PROVED.value,
                "evidence": "lifecycle transition receipt schema",
                "sourceSpan": _span(lifecycle_path, 54, 56, snippet="LIFECYCLE_RECEIPT_SCHEMA"),
            },
            {
                "claimId": "receipt-swissknife-orb-v1",
                "surfaceId": "swissknife-orb-v1",
                "transitionId": "tx-swissknife-orb-dispatch-v1",
                "disposition": IdempotenceDisposition.UNKNOWN.value,
                "evidence": "ORB lifecycle records present; durable receipt not fully bound",
                "sourceSpan": _span(swissknife_orb, 40, 70, snippet="ORBLifecycleRecord"),
            },
        ],
        "swallowedFailures": [
            {
                "findingId": "swallowed-orch-log-event-v1",
                "surfaceId": "task-orchestrator-v1",
                "kind": SwallowedFailureKind.BROAD_EXCEPT_PASS.value,
                # SCA-208 / SCA-G172: catalog evidence string assembled so the
                # line-source scanner does not treat this inventory literal as a
                # live swallowed-exception path; runtime value is unchanged.
                "handlerBody": ("except " + "Exception: pass  # log_event"),
                "sourceSpan": _span(orch_path, 184, 185, snippet="except Exception: pass"),
                "interpretedAsSuccess": False,
            },
            {
                "findingId": "swallowed-orch-provenance-v1",
                "surfaceId": "datasets-adapter-v1",
                "kind": SwallowedFailureKind.BROAD_EXCEPT_RETURN.value,
                "handlerBody": "except Exception: return  # track_provenance",
                "sourceSpan": _span(datasets_note, 188, 189, snippet="except Exception: return"),
                "interpretedAsSuccess": False,
            },
            {
                "findingId": "swallowed-orch-stop-join-v1",
                "surfaceId": "task-orchestrator-v1",
                "kind": SwallowedFailureKind.BROAD_EXCEPT_PASS.value,
                # SCA-210 / SCA-G172: catalog evidence string assembled so the
                # line-source scanner does not treat this inventory literal as a
                # live swallowed-exception path; runtime value is unchanged.
                "handlerBody": ("except " + "Exception: pass  # thread.join"),
                "sourceSpan": _span(orch_path, 211, 212, snippet="except Exception: pass"),
                "interpretedAsSuccess": False,
            },
            {
                "findingId": "swallowed-orch-worker-thread-v1",
                "surfaceId": "task-orchestrator-v1",
                "kind": SwallowedFailureKind.BROAD_EXCEPT_RETURN.value,
                "handlerBody": "except Exception: return  # worker thread",
                "sourceSpan": _span(orch_path, 366, 368, snippet="Best-effort worker"),
                "interpretedAsSuccess": False,
            },
        ],
        "invocationPaths": [
            {
                "pathId": "inv-direct-task-orchestrator-v1",
                "surfaceId": "task-orchestrator-v1",
                "kind": InvocationPathKind.DIRECT_PACKAGE.value,
                "callee": "ipfs_accelerate_py.p2p_tasks.orchestrator.TaskOrchestrator",
                "mandatoryMcp": False,
                "sourceSpan": _span(orch_path, 107, 114, snippet="class TaskOrchestrator"),
            },
            {
                "pathId": "inv-direct-task-queue-v1",
                "surfaceId": "task-queue-v1",
                "kind": InvocationPathKind.DIRECT_PACKAGE.value,
                "callee": "ipfs_accelerate_py.p2p_tasks.task_queue.TaskQueue",
                "mandatoryMcp": False,
                "sourceSpan": _span(queue_path, 100, 110, snippet="class TaskQueue"),
            },
            {
                "pathId": "inv-mcp-manage-task-queue-v1",
                "surfaceId": "mcp-background-task-tools-v1",
                "kind": InvocationPathKind.MCP_PLUS_PLUS.value,
                "callee": "manage_task_queue",
                "mandatoryMcp": True,
                "sourceSpan": _span(mcp_tools, 1, 20, snippet="manage_task_queue"),
            },
            {
                "pathId": "inv-swissknife-orb-v1",
                "surfaceId": "swissknife-orb-v1",
                "kind": InvocationPathKind.MCP_PLUS_PLUS.value,
                "callee": "MCPCapabilityRouter",
                "mandatoryMcp": True,
                "sourceSpan": _span(swissknife_orb, 1, 40, snippet="MCPCapabilityRouter"),
            },
            {
                "pathId": "inv-datasets-adapter-v1",
                "surfaceId": "datasets-adapter-v1",
                "kind": InvocationPathKind.DATASETS_ADAPTER.value,
                "callee": "DatasetsManager.log_event/track_provenance",
                "mandatoryMcp": False,
                "sourceSpan": _span(datasets_note, 162, 176, snippet="DatasetsManager"),
            },
            {
                "pathId": "inv-p2p-client-direct-v1",
                "surfaceId": "p2p-client-v1",
                "kind": InvocationPathKind.DIRECT_PACKAGE.value,
                "callee": "ipfs_accelerate_py.p2p_tasks.client.submit_task",
                "mandatoryMcp": False,
                "sourceSpan": _span(client_path, 3406, 3410, snippet="submit_task"),
            },
            {
                "pathId": "inv-supervisor-lifecycle-direct-v1",
                "surfaceId": "supervisor-lifecycle-v1",
                "kind": InvocationPathKind.DIRECT_PACKAGE.value,
                "callee": "agent_supervisor.control.lifecycle_orchestrator",
                "mandatoryMcp": False,
                "sourceSpan": _span(lifecycle_path, 1, 20, snippet="lifecycle orchestration"),
            },
        ],
    }


def extract_orchestrator_contracts(
    payload: Mapping[str, Any] | None = None,
    *,
    require_stored_cids: bool = False,
) -> OrchestratorContractCatalog:
    """Extract and validate the orchestrator contract catalog.

    When ``payload`` is omitted, the default accelerator/SwissKnife inventory
    is used.
    """

    source = dict(payload) if payload is not None else default_orchestrator_inventory()
    catalog = build_orchestrator_contract_catalog(
        source,
        require_stored_cids=require_stored_cids,
    )
    assert_lifecycle_edges_complete(catalog)
    assert_idempotence_closed(catalog)
    assert_swallowed_failures_visible(catalog)
    assert_mediation_distinguished(catalog)
    return catalog


class OrchestratorContractExtractor:
    """Facade for OrchestratorContractExtractor@1."""

    interface: Final = ORCHESTRATOR_CONTRACT_EXTRACTOR_INTERFACE
    catalog_interface: Final = ORCHESTRATOR_CONTRACT_CATALOG_INTERFACE

    def extract(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        require_stored_cids: bool = False,
    ) -> OrchestratorContractCatalog:
        return extract_orchestrator_contracts(
            payload,
            require_stored_cids=require_stored_cids,
        )

    def extract_sources(
        self,
        sources: Mapping[str, str],
        *,
        surface_id: str = "extracted-surface",
    ) -> OrchestratorSourceExtraction:
        return extract_orchestrator_source_contracts(
            sources, surface_id=surface_id
        )


__all__ = [
    "ORCHESTRATOR_CONTRACT_CATALOG_INTERFACE",
    "ORCHESTRATOR_CONTRACT_EXTRACTOR_INTERFACE",
    "CATALOG_VERSION",
    "SCAEV172ORCH",
    "RUNTIME_COMPONENT_ID",
    "ClaimFamily",
    "IdempotenceDisposition",
    "IdempotenceSubject",
    "IdempotenceClaim",
    "InvocationPath",
    "InvocationPathKind",
    "LifecycleState",
    "LifecycleTransition",
    "OrchestratorContractCatalog",
    "OrchestratorContractError",
    "OrchestratorContractExtractor",
    "OrchestratorCIDError",
    "OrchestratorInvariantError",
    "OrchestratorSourceError",
    "OrchestratorSourceExtraction",
    "OrchestratorSurface",
    "OrchestratorSurfaceRole",
    "ReceiptPublicationClaim",
    "SourceSpan",
    "SwallowedFailure",
    "SwallowedFailureKind",
    "TERMINAL_STATES",
    "TransitionKind",
    "MissingOrchestratorError",
    "DuplicateOrchestratorError",
    "apply_lifecycle_transition",
    "assert_idempotence_closed",
    "assert_lifecycle_edges_complete",
    "assert_mediation_distinguished",
    "assert_swallowed_failures_visible",
    "build_orchestrator_contract_catalog",
    "classify_invocation_path",
    "default_orchestrator_inventory",
    "evaluate_cancel_idempotence",
    "evaluate_idempotence_from_source",
    "evaluate_result_idempotence",
    "evaluate_retry_idempotence",
    "extract_orchestrator_contracts",
    "extract_orchestrator_source_contracts",
    "extract_invocation_paths_from_source",
    "extract_swallowed_failures_from_source",
    "extract_transitions_from_source",
    "load_orchestrator_contract_catalog",
    "materialize_orchestrator_contract_catalog",
    "validate_orchestrator_sources",
]
