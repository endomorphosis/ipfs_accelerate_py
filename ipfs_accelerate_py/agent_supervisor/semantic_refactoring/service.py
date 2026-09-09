"""SPAR-044 typed Python control surface and diagnostics.

This module extends current supervisor operational authority with
``SemanticRefactoringService@1``.  It publishes deterministic
non-authoritative diagnostics and a closed catalog of narrow authorized
operations.  Thin CLI/MCP adapters decode only; this service owns every
policy decision.  MCP never shells out.

The service is nomination-only.  Diagnostics cannot admit a transition,
completion, merge, or competing authority.  Vector, model, and heuristic
evidence cannot admit an operation.  Observational metadata is excluded
from identity.  Dry-run is deterministic and never mutates.  Network is
denied.  Independent supervisor validation and merge authority remain
separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Final, Mapping, Sequence
import hashlib
import json
import unicodedata

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

from .partition_generators import (
    IDENTITY_EXCLUDED_FIELDS as SPAR013_IDENTITY_EXCLUDED_FIELDS,
)


TASK_ID: Final[str] = "SPAR-044"
GOAL_ID: Final[str] = "SPAR-G071"
PROGRAM: Final[str] = "semantic-preserving-autonomous-remodularization-v1"
AUTHORITY: Final[str] = "operational refactoring authority"
AUTHORITY_OWNER: Final[str] = "ipfs_accelerate_py"
ANALYZER_ID: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_refactoring.service@1"
)
PREDECESSOR_TASK_IDS: Final[tuple[str, ...]] = ("SPAR-035", "SPAR-043")

SEMANTIC_REFACTORING_SERVICE_INTERFACE: Final[str] = "SemanticRefactoringService@1"
CONTROL_REQUEST_INTERFACE: Final[str] = "SparControlRequest@1"
CONTROL_RESULT_INTERFACE: Final[str] = "SparControlResult@1"
CONTROL_RECEIPT_INTERFACE: Final[str] = "SparControlReceipt@1"
DIAGNOSTIC_REPORT_INTERFACE: Final[str] = "SparDiagnosticReport@1"
TYPED_TERMINAL_INTERFACE: Final[str] = "TypedTerminal@1"

SEMANTIC_REFACTORING_SERVICE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/semantic-refactoring-service@1"
)
CONTROL_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/spar-control-request@1"
)
CONTROL_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/spar-control-result@1"
)
CONTROL_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/spar-control-receipt@1"
)
DIAGNOSTIC_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/spar-diagnostic-report@1"
)
TYPED_TERMINAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/spar-control-typed-terminal@1"
)
CONTROL_AUDIT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/spar-control-audit@1"
)

CONTROL_CONTRACT_VERSION: Final[str] = "1"

SERVICE_CAN_AUTHORIZE_TRANSITION: Final[bool] = False
SERVICE_CAN_AUTHORIZE_COMPLETION: Final[bool] = False
SERVICE_CAN_CREATE_AUTHORITY: Final[bool] = False
SERVICE_CAN_WEAKEN_VALIDATION: Final[bool] = False
SERVICE_WRITES_REPOSITORY: Final[bool] = False
SERVICE_CAN_CHANGE_MODE: Final[bool] = False
VECTOR_SIMILARITY_IS_AUTHORITY: Final[bool] = False
PROJECTION_CLUSTERING_IS_AUTHORITY: Final[bool] = False
MODEL_OUTPUT_IS_PROPOSAL_ONLY: Final[bool] = True
TEST_PASS_IS_NOT_COMPLETION: Final[bool] = True
MARKDOWN_IS_NOT_COMPLETION: Final[bool] = True
WORKER_SELF_APPROVAL: Final[bool] = False
DUCKLAKE_IS_AUTHORITY: Final[bool] = False
SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS: Final[bool] = True
SERVICE_IS_NOMINATION_ONLY: Final[bool] = True
RAW_SOURCE_REQUIRED: Final[bool] = True
NETWORK_DENIED: Final[bool] = True
NETWORK_DENY: Final[str] = "deny"
DRY_RUN_IS_DETERMINISTIC: Final[bool] = True
DRY_RUN_MUTATES: Final[bool] = False
NEGATIVE_EVIDENCE_RETAINED: Final[bool] = True
MCP_NEVER_SHELLS: Final[bool] = True
DIAGNOSTICS_ARE_AUTHORITATIVE: Final[bool] = False
INDEPENDENT_VALIDATION_REQUIRED: Final[bool] = True
CONTEXT_COMPILER_REMAINS_AUTHORITY: Final[bool] = True
REQUIRED_ROLLOUT_CONSUMES_CONTEXT: Final[bool] = True
REQUIRED_ROLLOUT_EMITS_RECEIPT_FLOOR: Final[bool] = True

MAX_TEXT_CHARS: Final[int] = 16_384
MAX_PATH_CHARS: Final[int] = 1_024
MAX_COMMAND_CHARS: Final[int] = 1_024
MAX_WRITE_PATHS: Final[int] = 64
MAX_COMMANDS: Final[int] = 64
MAX_EVIDENCE_CIDS: Final[int] = 1_024
MAX_PARAMETERS_BYTES: Final[int] = 32_768
MAX_AUDITS: Final[int] = 4_096

MUTATION_SCOPE: Final[str] = "spar.mutate"

IDENTITY_EXCLUDED_FIELDS: Final[frozenset[str]] = SPAR013_IDENTITY_EXCLUDED_FIELDS

READ_OPERATIONS: Final[tuple[str, ...]] = (
    "spar.capabilities",
    "spar.status",
    "spar.diagnose",
    "spar.get_receipt",
    "spar.metrics",
    "spar.rollout_status",
    "spar.context_status",
)
MUTATION_OPERATIONS: Final[tuple[str, ...]] = (
    "spar.nominate",
    "spar.request_review",
    "spar.request_merge",
    "spar.rollback",
)
ALL_OPERATIONS: Final[tuple[str, ...]] = READ_OPERATIONS + MUTATION_OPERATIONS

DECLARED_READ_OPERATIONS: Final[frozenset[str]] = frozenset(READ_OPERATIONS)
DECLARED_MUTATION_OPERATIONS: Final[frozenset[str]] = frozenset(MUTATION_OPERATIONS)
DECLARED_OPERATIONS: Final[frozenset[str]] = frozenset(ALL_OPERATIONS)

REQUIRED_RECEIPT_FLOOR: Final[tuple[str, ...]] = (
    "pre_world_root_cid",
    "program_graph_snapshot_cid",
    "partition_candidate_cid",
    "boundary_contract_set_cid",
    "transformation_packet_cid",
    "context_receipt_cid",
    "route_decision_cid",
    "refactor_transition_cid",
)
DECLARED_RECEIPT_FLOOR: Final[frozenset[str]] = frozenset(REQUIRED_RECEIPT_FLOOR)

EXISTING_ADAPTER_AUTHORITIES: Final[tuple[str, ...]] = (
    "datasets_semantic",
    "kit_storage",
    "kit_vfs",
    "accelerator_supervisor",
    "accelerator_runtime",
    "spar_narrow",
)

_FORBIDDEN_CAPSULE_TYPE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "FunctionSemanticCapsule",
        "MethodSemanticCapsule",
        "ClassSemanticCapsule",
        "TopLevelBlockCapsule",
        "ModuleSemanticCapsule",
        "PackageSemanticCapsule",
        "CallsiteSemanticCapsule",
        "StateOwnerCapsule",
        "RegistrationCapsule",
        "ResourceLifecycleCapsule",
    }
)

_NON_ADMITTING_EVIDENCE: Final[frozenset[str]] = frozenset(
    {
        "vector_candidate",
        "model_hypothesis",
        "heuristic",
    }
)

_AUTHORITY_FLAG_NAMES: Final[tuple[str, ...]] = (
    "can_authorize_transition",
    "can_authorize_completion",
    "can_create_authority",
    "can_weaken_validation",
    "can_change_mode",
    "projection_is_authority",
    "writes_repository",
    "worker_self_approval",
)

_FORBIDDEN_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "ast",
        "body",
        "code",
        "contents",
        "content",
        "file_content",
        "file_contents",
        "file_text",
        "full_ast",
        "full_graph",
        "full_source",
        "full_task_body",
        "full_task_dump",
        "prompt",
        "repository_dump",
        "source",
        "source_body",
        "source_code",
        "source_text",
        "task_prose",
        "transcript",
        "snippet",
    }
)

FORBIDDEN_CONTROL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "authorize_completion",
        "shell_mcp",
        "subprocess_dispatch",
        "weaken_validation",
        "admit_by_similarity",
        "open_network",
        "self_approve",
        "replace_context_compiler",
        "change_rollout_mode",
        "suppress_raw_source",
    }
)

MISSING_RECEIPT_FLOOR_MESSAGE: Final[str] = (
    "subsequent program tasks must emit exact pre/post roots, partition, "
    "boundary, packet, route, validation, and transition receipts"
)
MISSING_CONTEXT_MESSAGE: Final[str] = (
    "subsequent program tasks must consume refactoring context"
)


class ControlSurfaceError(ValueError):
    """Fail-closed violation of a SPAR-044 control-surface contract."""

    def __init__(self, message: str, *, reason_code: str = "malformed") -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)


class ControlStatus(str, Enum):
    OK = "ok"
    DIAGNOSED = "diagnosed"
    NOMINATED = "nominated"
    DRY_RUN = "dry_run"
    CONFLICT = "conflict"
    TYPED_TERMINAL = "typed_terminal"
    INVALID = "invalid"


class TerminalKind(str, Enum):
    UNSUPPORTED = "unsupported"
    HUMAN_REVIEW = "human_review"
    MISSING_RECEIPT_FLOOR = "missing_receipt_floor"
    MISSING_CONTEXT = "missing_context"
    STALE_FENCE = "stale_fence"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"


DECLARED_CONTROL_STATUSES: Final[frozenset[str]] = frozenset(
    item.value for item in ControlStatus
)
DECLARED_TERMINAL_KINDS: Final[frozenset[str]] = frozenset(
    item.value for item in TerminalKind
)


def _text(value: Any, name: str, *, empty: bool = False) -> str:
    if type(value) is not str:
        raise ControlSurfaceError(f"{name} must be a string")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise ControlSurfaceError(f"{name} must be trimmed NFC text")
    if not empty and not value:
        raise ControlSurfaceError(f"{name} must be a nonempty string")
    if any(not char.isprintable() for char in value):
        raise ControlSurfaceError(f"{name} contains invalid text")
    if len(value) > MAX_TEXT_CHARS:
        raise ControlSurfaceError(f"{name} exceeds text bound")
    return value


def _cid(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        return validate_cid(text)
    except Exception as exc:
        raise ControlSurfaceError(f"{name} must be a valid CID") from exc


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ControlSurfaceError(f"{name} must be a boolean")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ControlSurfaceError(f"{name} must be a non-negative integer")
    return value


def _reject_excluded(payload: Mapping[str, Any], name: str) -> None:
    present = IDENTITY_EXCLUDED_FIELDS & set(payload)
    if present:
        raise ControlSurfaceError(
            f"{name} identity excludes observational fields: {sorted(present)}"
        )


def _reject_body(payload: Mapping[str, Any], name: str) -> None:
    present = _FORBIDDEN_BODY_KEYS & set(payload)
    if present:
        raise ControlSurfaceError(
            f"{name} must remain body-free; forbidden keys: {sorted(present)}"
        )


def _pop_authority_flags(payload: dict[str, Any], name: str) -> None:
    for flag in _AUTHORITY_FLAG_NAMES:
        if flag not in payload:
            continue
        if payload.pop(flag) is not False:
            raise ControlSurfaceError(f"{name} cannot claim {flag}")


def _network_value(value: Any) -> str:
    text = _text(value, "network")
    if text != NETWORK_DENY:
        raise ControlSurfaceError("network is denied")
    return NETWORK_DENY


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def control_surface_cid_profile() -> dict[str, str]:
    return {
        "profile_id": "ipfs_accelerate_py.cid-utils@1",
        "codec": "dag-json",
        "rule": (
            "CID identifies exact canonical bytes under declared codec/profile, "
            "not universal meaning"
        ),
    }


def control_surface_descriptor() -> dict[str, Any]:
    return {
        "schema": SEMANTIC_REFACTORING_SERVICE_SCHEMA,
        "interface": SEMANTIC_REFACTORING_SERVICE_INTERFACE,
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "analyzer_id": ANALYZER_ID,
        "predecessor_task_ids": list(PREDECESSOR_TASK_IDS),
        "authority_owner": AUTHORITY_OWNER,
        "nomination_only": True,
        "writes_repository": False,
        "diagnostics_are_authoritative": False,
        "mcp_never_shells": True,
        "independent_validation_required": True,
        "context_compiler_remains_authority": True,
        "required_rollout_consumes_context": True,
        "required_rollout_emits_receipt_floor": True,
        "network": NETWORK_DENY,
        "model_output_is_proposal_only": True,
        "can_weaken_validation": False,
        "can_authorize_completion": False,
        "can_authorize_transition": False,
        "read_operations": list(READ_OPERATIONS),
        "mutation_operations": list(MUTATION_OPERATIONS),
        "receipt_floor": list(REQUIRED_RECEIPT_FLOOR),
        "forbids": list(sorted(FORBIDDEN_CONTROL_NAMES)),
    }


@dataclass(frozen=True)
class ControlAuthorization:
    """Explicit transport-neutral authority for a SPAR mutation."""

    subject: str
    permitted: bool
    scopes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject", _text(self.subject, "authorization subject"))
        if type(self.permitted) is not bool:
            raise ControlSurfaceError("authorization permitted must be boolean")
        if isinstance(self.scopes, (str, bytes)):
            raise ControlSurfaceError("authorization scopes must be an array")
        object.__setattr__(
            self,
            "scopes",
            tuple(sorted({_text(item, "authorization scope") for item in self.scopes})),
        )

    @property
    def allows_mutation(self) -> bool:
        return self.permitted and MUTATION_SCOPE in self.scopes

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "permitted": self.permitted,
            "scopes": list(self.scopes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlAuthorization":
        if not isinstance(payload, Mapping) or set(payload).difference(
            {"subject", "permitted", "scopes"}
        ):
            raise ControlSurfaceError("authorization must be a closed object")
        return cls(
            payload.get("subject", ""),
            payload.get("permitted"),
            tuple(payload.get("scopes") or ()),
        )


@dataclass(frozen=True)
class ControlBudget:
    """Bounded control-plane work; it never grants merge or completion."""

    max_units: int
    requested_units: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_units", _non_negative_int(self.max_units, "budget max_units")
        )
        object.__setattr__(
            self,
            "requested_units",
            _non_negative_int(self.requested_units, "budget requested_units"),
        )
        if self.requested_units > self.max_units:
            raise ControlSurfaceError("budget requested_units exceeds max_units")

    def to_dict(self) -> dict[str, int]:
        return {"max_units": self.max_units, "requested_units": self.requested_units}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlBudget":
        if not isinstance(payload, Mapping) or set(payload) != {
            "max_units",
            "requested_units",
        }:
            raise ControlSurfaceError("budget must contain max_units and requested_units")
        return cls(payload["max_units"], payload["requested_units"])


@dataclass(frozen=True)
class ControlRequest:
    """One complete SPAR control request submitted to the service."""

    operation: str
    target_id: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    idempotency_key: str = ""
    authorization: ControlAuthorization | None = None
    lease_id: str = ""
    fencing_epoch: int | None = None
    budget: ControlBudget | None = None
    tree_id: str = ""

    def __post_init__(self) -> None:
        operation = _text(self.operation, "operation")
        if operation not in DECLARED_OPERATIONS:
            raise ControlSurfaceError(f"unknown SPAR operation: {operation}")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(
            self, "target_id", _text(self.target_id, "target_id", empty=True)
        )
        if not isinstance(self.parameters, Mapping):
            raise ControlSurfaceError("parameters must be an object")
        _reject_body(self.parameters, "parameters")
        try:
            encoded = json.dumps(
                dict(self.parameters), sort_keys=True, separators=(",", ":")
            )
        except (TypeError, ValueError) as exc:
            raise ControlSurfaceError("parameters must be JSON-compatible") from exc
        if len(encoded.encode("utf-8")) > MAX_PARAMETERS_BYTES:
            raise ControlSurfaceError("parameters exceed 32768 bytes")
        object.__setattr__(self, "parameters", dict(self.parameters))
        if type(self.dry_run) is not bool:
            raise ControlSurfaceError("dry_run must be boolean")
        if self.authorization is not None and not isinstance(
            self.authorization, ControlAuthorization
        ):
            if not isinstance(self.authorization, Mapping):
                raise ControlSurfaceError("authorization must be typed")
            object.__setattr__(
                self, "authorization", ControlAuthorization.from_dict(self.authorization)
            )
        if self.budget is not None and not isinstance(self.budget, ControlBudget):
            if not isinstance(self.budget, Mapping):
                raise ControlSurfaceError("budget must be typed")
            object.__setattr__(self, "budget", ControlBudget.from_dict(self.budget))
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id", empty=True))
        if operation in DECLARED_MUTATION_OPERATIONS:
            object.__setattr__(
                self, "idempotency_key", _text(self.idempotency_key, "idempotency_key")
            )
            object.__setattr__(self, "lease_id", _text(self.lease_id, "lease_id"))
            if (
                isinstance(self.fencing_epoch, bool)
                or not isinstance(self.fencing_epoch, int)
                or self.fencing_epoch < 0
            ):
                raise ControlSurfaceError(
                    "mutation requires a non-negative fencing_epoch"
                )
            if self.authorization is None or not self.authorization.allows_mutation:
                raise ControlSurfaceError(
                    "mutation requires SPAR mutation authorization"
                )
            if self.budget is None:
                raise ControlSurfaceError("mutation requires a resource budget")
        elif any(
            (
                self.idempotency_key,
                self.lease_id,
                self.fencing_epoch is not None,
                self.budget is not None,
            )
        ):
            raise ControlSurfaceError("read operations cannot carry mutation bindings")

    @property
    def is_mutation(self) -> bool:
        return self.operation in DECLARED_MUTATION_OPERATIONS

    @property
    def fingerprint(self) -> str:
        body = {
            "operation": self.operation,
            "target_id": self.target_id,
            "parameters": self.parameters,
            "dry_run": self.dry_run,
            "authorization": (
                self.authorization.to_dict() if self.authorization else None
            ),
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "budget": self.budget.to_dict() if self.budget else None,
            "tree_id": self.tree_id,
        }
        return _fingerprint(body)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "operation": self.operation,
            "target_id": self.target_id,
            "parameters": dict(self.parameters),
            "dry_run": self.dry_run,
            "tree_id": self.tree_id,
        }
        if self.is_mutation:
            result.update(
                {
                    "idempotency_key": self.idempotency_key,
                    "authorization": (
                        self.authorization.to_dict() if self.authorization else None
                    ),
                    "lease_id": self.lease_id,
                    "fencing_epoch": self.fencing_epoch,
                    "budget": self.budget.to_dict() if self.budget else None,
                }
            )
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlRequest":
        allowed = {
            "operation",
            "target_id",
            "parameters",
            "dry_run",
            "idempotency_key",
            "authorization",
            "lease_id",
            "fencing_epoch",
            "budget",
            "tree_id",
        }
        if not isinstance(payload, Mapping) or set(payload).difference(allowed):
            raise ControlSurfaceError("SPAR control request contains unknown fields")
        return cls(**dict(payload))


@dataclass(frozen=True)
class ControlResult:
    """Canonical SPAR control result shared by Python, CLI, and MCP."""

    operation: str
    ok: bool
    status: str
    audit_id: str
    payload: Mapping[str, Any]
    idempotent_replay: bool = False
    receipt_cid: str = ""

    interface: ClassVar[str] = CONTROL_RESULT_INTERFACE
    schema: ClassVar[str] = CONTROL_RESULT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _text(self.operation, "result operation"))
        if self.operation not in DECLARED_OPERATIONS:
            raise ControlSurfaceError(f"unknown SPAR operation: {self.operation}")
        if type(self.ok) is not bool:
            raise ControlSurfaceError("result ok must be boolean")
        status = _text(self.status, "result status")
        if status not in DECLARED_CONTROL_STATUSES:
            raise ControlSurfaceError(f"unsupported status {status!r}")
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self, "audit_id", _text(self.audit_id, "audit_id", empty=True)
        )
        if not isinstance(self.payload, Mapping):
            raise ControlSurfaceError("payload must be an object")
        _reject_body(self.payload, "payload")
        object.__setattr__(self, "payload", _jsonable(dict(self.payload)))
        if type(self.idempotent_replay) is not bool:
            raise ControlSurfaceError("idempotent_replay must be boolean")
        object.__setattr__(
            self, "receipt_cid", _text(self.receipt_cid, "receipt_cid", empty=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "ok": self.ok,
            "status": self.status,
            "audit_id": self.audit_id,
            "payload": dict(self.payload),
            "idempotent_replay": self.idempotent_replay,
            "receipt_cid": self.receipt_cid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlResult":
        required = {"operation", "ok", "status", "audit_id", "payload"}
        if not isinstance(payload, Mapping) or not required.issubset(payload):
            raise ControlSurfaceError("SPAR control service returned an invalid result")
        return cls(
            _text(payload["operation"], "result operation"),
            bool(payload["ok"]),
            _text(payload["status"], "result status"),
            _text(payload["audit_id"], "audit_id", empty=True),
            dict(payload["payload"])
            if isinstance(payload["payload"], Mapping)
            else {},
            bool(payload.get("idempotent_replay", False)),
            _text(payload.get("receipt_cid", ""), "receipt_cid", empty=True),
        )


@dataclass(frozen=True)
class ControlReceipt:
    """Content-addressed SPAR-044 control receipt. Nomination-only."""

    operation: str
    status: str
    payload: Mapping[str, Any]
    analyzer_id: str = ANALYZER_ID
    adapter_is_nomination_only: bool = True
    mutated: bool = False
    deterministic: bool = True
    network: str = NETWORK_DENY
    diagnostics_are_authoritative: bool = False
    independent_validation_required: bool = True
    mcp_never_shells: bool = True

    interface: ClassVar[str] = CONTROL_RECEIPT_INTERFACE
    schema: ClassVar[str] = CONTROL_RECEIPT_SCHEMA
    _FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema",
            "interface",
            "operation",
            "status",
            "payload",
            "analyzer_id",
            "adapter_is_nomination_only",
            "mutated",
            "deterministic",
            "network",
            "diagnostics_are_authoritative",
            "independent_validation_required",
            "mcp_never_shells",
            "can_authorize_transition",
            "can_authorize_completion",
            "can_create_authority",
            "can_weaken_validation",
            "receipt_cid",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _text(self.operation, "operation"))
        if self.operation not in DECLARED_OPERATIONS:
            raise ControlSurfaceError(f"unknown SPAR operation: {self.operation}")
        status = _text(self.status, "status")
        if status not in DECLARED_CONTROL_STATUSES:
            raise ControlSurfaceError(f"unsupported status {status!r}")
        object.__setattr__(self, "status", status)
        if not isinstance(self.payload, Mapping):
            raise ControlSurfaceError("payload must be an object")
        _reject_excluded(self.payload, "payload")
        _reject_body(self.payload, "payload")
        object.__setattr__(self, "payload", _jsonable(dict(self.payload)))
        analyzer = _text(self.analyzer_id, "analyzer_id")
        if analyzer != ANALYZER_ID:
            raise ControlSurfaceError("receipt analyzer_id must remain SPAR-044")
        object.__setattr__(self, "analyzer_id", ANALYZER_ID)
        if _bool(self.adapter_is_nomination_only, "adapter_is_nomination_only") is not True:
            raise ControlSurfaceError("adapter must remain nomination_only")
        if _bool(self.mutated, "mutated") is not False:
            raise ControlSurfaceError("dry-run/receipt must not mutate")
        if _bool(self.deterministic, "deterministic") is not True:
            raise ControlSurfaceError("control receipt must remain deterministic")
        object.__setattr__(self, "network", _network_value(self.network))
        if (
            _bool(self.diagnostics_are_authoritative, "diagnostics_are_authoritative")
            is not False
        ):
            raise ControlSurfaceError("diagnostics are not authoritative")
        if (
            _bool(
                self.independent_validation_required,
                "independent_validation_required",
            )
            is not True
        ):
            raise ControlSurfaceError("independent validation is required")
        if _bool(self.mcp_never_shells, "mcp_never_shells") is not True:
            raise ControlSurfaceError("MCP never shells out")
        object.__setattr__(self, "adapter_is_nomination_only", True)
        object.__setattr__(self, "mutated", False)
        object.__setattr__(self, "deterministic", True)
        object.__setattr__(self, "diagnostics_are_authoritative", False)
        object.__setattr__(self, "independent_validation_required", True)
        object.__setattr__(self, "mcp_never_shells", True)

    @property
    def can_authorize_transition(self) -> bool:
        return False

    @property
    def can_authorize_completion(self) -> bool:
        return False

    @property
    def can_create_authority(self) -> bool:
        return False

    @property
    def can_weaken_validation(self) -> bool:
        return False

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "operation": self.operation,
            "status": self.status,
            "payload": _jsonable(dict(self.payload)),
            "analyzer_id": self.analyzer_id,
            "adapter_is_nomination_only": True,
            "mutated": False,
            "deterministic": True,
            "network": NETWORK_DENY,
            "diagnostics_are_authoritative": False,
            "independent_validation_required": True,
            "mcp_never_shells": True,
            "can_authorize_transition": False,
            "can_authorize_completion": False,
            "can_create_authority": False,
            "can_weaken_validation": False,
        }

    @property
    def receipt_cid(self) -> str:
        return cid_for_dag_json(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["receipt_cid"] = self.receipt_cid
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlReceipt":
        if not isinstance(payload, Mapping):
            raise ControlSurfaceError("control receipt must be an object")
        data = dict(payload)
        _reject_excluded(data, "control receipt")
        claimed = data.pop("receipt_cid", "")
        _pop_authority_flags(data, "control receipt")
        extra = set(data) - (cls._FIELDS - {"receipt_cid"})
        if extra:
            raise ControlSurfaceError(
                f"unknown control receipt field: {sorted(extra)}"
            )
        receipt = cls(
            operation=data.get("operation", ""),
            status=data.get("status", ""),
            payload=data.get("payload") or {},
            analyzer_id=data.get("analyzer_id", ANALYZER_ID),
            adapter_is_nomination_only=data.get("adapter_is_nomination_only", True),
            mutated=data.get("mutated", False),
            deterministic=data.get("deterministic", True),
            network=data.get("network", NETWORK_DENY),
            diagnostics_are_authoritative=data.get(
                "diagnostics_are_authoritative", False
            ),
            independent_validation_required=data.get(
                "independent_validation_required", True
            ),
            mcp_never_shells=data.get("mcp_never_shells", True),
        )
        if claimed and claimed != receipt.receipt_cid:
            raise ControlSurfaceError("receipt_cid does not verify")
        return receipt


def encode_canonical_receipt(receipt: ControlReceipt) -> dict[str, Any]:
    return receipt.to_dict()


def decode_canonical_receipt(payload: Mapping[str, Any]) -> ControlReceipt:
    return ControlReceipt.from_dict(payload)


def compile_control_receipt(
    *,
    operation: str,
    status: str,
    payload: Mapping[str, Any],
) -> ControlReceipt:
    return ControlReceipt(operation=operation, status=status, payload=dict(payload))


class ControlStateStore:
    """In-process SPAR control state. Not completion or merge authority."""

    def __init__(self) -> None:
        self.applied: dict[str, ControlResult] = {}
        self.fingerprints: dict[str, str] = {}
        self.fences: dict[str, int] = {}
        self.receipts: dict[str, dict[str, Any]] = {}
        self.audits: list[dict[str, Any]] = []

    def record_audit(self, record: Mapping[str, Any]) -> None:
        if len(self.audits) >= MAX_AUDITS:
            raise ControlSurfaceError("audit log exceeds maximum length")
        self.audits.append(dict(record))


def _catalog() -> dict[str, Any]:
    return {
        "read": list(READ_OPERATIONS),
        "mutation": list(MUTATION_OPERATIONS),
        "all": list(ALL_OPERATIONS),
    }


def _status_payload() -> dict[str, Any]:
    descriptor = control_surface_descriptor()
    return {
        "task_id": TASK_ID,
        "goal_id": GOAL_ID,
        "interface": SEMANTIC_REFACTORING_SERVICE_INTERFACE,
        "nomination_only": True,
        "network": NETWORK_DENY,
        "mcp_never_shells": True,
        "diagnostics_are_authoritative": False,
        "independent_validation_required": True,
        "predecessor_task_ids": list(PREDECESSOR_TASK_IDS),
        "existing_adapter_authorities": list(EXISTING_ADAPTER_AUTHORITIES),
        "descriptor": descriptor,
    }


def _diagnostic_payload() -> dict[str, Any]:
    return {
        "schema": DIAGNOSTIC_REPORT_SCHEMA,
        "interface": DIAGNOSTIC_REPORT_INTERFACE,
        "authoritative": False,
        "can_authorize_completion": False,
        "can_authorize_transition": False,
        "catalog": _catalog(),
        "predecessors": {
            "SPAR-035": {
                "role": "context-routing",
                "context_compiler_remains_authority": True,
                "adapter_is_nomination_only": True,
            },
            "SPAR-043": {
                "role": "required-rollout",
                "consumes_context": True,
                "emits_receipt_floor": True,
                "receipt_floor": list(REQUIRED_RECEIPT_FLOOR),
            },
        },
        "network": NETWORK_DENY,
        "mcp_never_shells": True,
        "cid_profile": control_surface_cid_profile(),
    }


def _context_status_payload() -> dict[str, Any]:
    return {
        "predecessor": "SPAR-035",
        "context_compiler_remains_authority": True,
        "adapter_is_nomination_only": True,
        "one_residual_general_model_question": True,
        "independent_validation_required": True,
        "authoritative": False,
        "missing_context_message": MISSING_CONTEXT_MESSAGE,
    }


def _rollout_status_payload() -> dict[str, Any]:
    return {
        "predecessor": "SPAR-043",
        "nominated_mode": "required",
        "worker_may_change_mode": False,
        "consumes_context": True,
        "emits_receipt_floor": True,
        "receipt_floor": list(REQUIRED_RECEIPT_FLOOR),
        "authoritative": False,
        "missing_receipt_floor_message": MISSING_RECEIPT_FLOOR_MESSAGE,
    }


def _require_cids(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ControlSurfaceError(f"{name} must be a list of CIDs")
    ordered = tuple(_cid(item, name) for item in values)
    if len(ordered) != len(set(ordered)):
        raise ControlSurfaceError(f"{name} must not contain duplicates")
    if len(ordered) > MAX_EVIDENCE_CIDS:
        raise ControlSurfaceError(f"{name} exceeds maximum length")
    return ordered


def _exact_path(value: Any, name: str = "write_paths") -> str:
    raw = _text(value, name, empty=False)
    if len(raw) > MAX_PATH_CHARS:
        raise ControlSurfaceError(f"{name} exceeds path bound")
    normalized = raw.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or normalized in {".", ""}
        or normalized.startswith("./")
        or any(char in normalized for char in "*?[]{}")
        or "//" in normalized
        or normalized.endswith("/")
    ):
        raise ControlSurfaceError(
            f"{name} must be an exact repository-relative path; unrestricted scope is rejected"
        )
    if normalized != candidate.as_posix():
        raise ControlSurfaceError(
            f"{name} must be a normalized repository-relative path"
        )
    return normalized


def _exact_paths(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ControlSurfaceError(f"{name} must be a list of exact paths")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        path = _exact_path(item, name)
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    if not ordered:
        raise ControlSurfaceError(
            f"{name} must not be empty; unrestricted scope is rejected"
        )
    if len(ordered) > MAX_WRITE_PATHS:
        raise ControlSurfaceError(f"{name} exceeds path bound")
    return tuple(ordered)


def _commands(values: Any, name: str = "validation_commands") -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ControlSurfaceError(f"{name} must be a list of commands")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = _text(item, name)
        if len(text) > MAX_COMMAND_CHARS:
            raise ControlSurfaceError(f"{name} exceeds command bound")
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    if not ordered:
        raise ControlSurfaceError(f"{name} must not be empty")
    if len(ordered) > MAX_COMMANDS:
        raise ControlSurfaceError(f"{name} exceeds maximum length")
    return tuple(ordered)


def _reject_non_admitting(payload: Mapping[str, Any], name: str) -> None:
    authority = payload.get("source_authority") or payload.get("evidence_class")
    if isinstance(authority, str) and authority in _NON_ADMITTING_EVIDENCE:
        raise ControlSurfaceError(f"{name} cannot admit {authority}")
    for key in ("vector_candidate", "model_hypothesis", "heuristic"):
        if payload.get(key):
            raise ControlSurfaceError(f"{name} cannot admit {key}")


def _receipt_floor_from(parameters: Mapping[str, Any]) -> dict[str, str]:
    missing = [name for name in REQUIRED_RECEIPT_FLOOR if not parameters.get(name)]
    if missing:
        raise ControlSurfaceError(
            f"{MISSING_RECEIPT_FLOOR_MESSAGE}: {missing}",
            reason_code=TerminalKind.MISSING_RECEIPT_FLOOR.value,
        )
    floor: dict[str, str] = {}
    for name in REQUIRED_RECEIPT_FLOOR:
        floor[name] = _cid(parameters[name], name)
    return floor


def _context_binding(parameters: Mapping[str, Any]) -> str:
    context_cid = parameters.get("context_receipt_cid") or parameters.get(
        "context_cid"
    )
    if not context_cid:
        raise ControlSurfaceError(
            MISSING_CONTEXT_MESSAGE,
            reason_code=TerminalKind.MISSING_CONTEXT.value,
        )
    return _cid(context_cid, "context_receipt_cid")


class SemanticRefactoringService:
    """Canonical SPAR-044 Python service. Adapters decode only."""

    interface: ClassVar[str] = SEMANTIC_REFACTORING_SERVICE_INTERFACE
    schema: ClassVar[str] = SEMANTIC_REFACTORING_SERVICE_SCHEMA

    def __init__(self, store: ControlStateStore | None = None) -> None:
        self._store = store or ControlStateStore()

    @property
    def store(self) -> ControlStateStore:
        return self._store

    def operation_catalog(self) -> dict[str, tuple[str, ...]]:
        return {"read": READ_OPERATIONS, "mutation": MUTATION_OPERATIONS}

    def execute(self, request: ControlRequest | Mapping[str, Any]) -> ControlResult:
        if not isinstance(request, ControlRequest):
            request = ControlRequest.from_dict(request)
        if request.is_mutation:
            return self._execute_mutation(request)
        return self._execute_read(request)

    def dry_run(self, request: ControlRequest | Mapping[str, Any]) -> ControlResult:
        if not isinstance(request, ControlRequest):
            request = ControlRequest.from_dict(request)
        payload = request.to_dict()
        payload["dry_run"] = True
        return self.execute(ControlRequest.from_dict(payload))

    def _execute_read(self, request: ControlRequest) -> ControlResult:
        if request.operation == "spar.capabilities":
            payload: dict[str, Any] = _catalog()
            status = ControlStatus.OK.value
        elif request.operation == "spar.status":
            payload = _status_payload()
            status = ControlStatus.OK.value
        elif request.operation == "spar.diagnose":
            payload = _diagnostic_payload()
            status = ControlStatus.DIAGNOSED.value
        elif request.operation == "spar.get_receipt":
            receipt_cid = _cid(
                request.parameters.get("receipt_cid") or request.target_id,
                "receipt_cid",
            )
            stored = self._store.receipts.get(receipt_cid)
            if stored is None:
                payload = {
                    "found": False,
                    "receipt_cid": receipt_cid,
                    "reason_code": "receipt_not_found",
                }
                status = ControlStatus.OK.value
            else:
                payload = {"found": True, "stored_receipt": dict(stored)}
                status = ControlStatus.OK.value
        elif request.operation == "spar.metrics":
            payload = {
                "applied": len(self._store.applied),
                "audits": len(self._store.audits),
                "receipts": len(self._store.receipts),
                "authoritative": False,
            }
            status = ControlStatus.OK.value
        elif request.operation == "spar.rollout_status":
            payload = _rollout_status_payload()
            status = ControlStatus.OK.value
        elif request.operation == "spar.context_status":
            payload = _context_status_payload()
            status = ControlStatus.OK.value
        else:
            raise ControlSurfaceError(f"unknown SPAR operation: {request.operation}")
        receipt = compile_control_receipt(
            operation=request.operation, status=status, payload=payload
        )
        result = ControlResult(
            operation=request.operation,
            ok=True,
            status=status,
            audit_id="",
            payload=payload,
            receipt_cid=receipt.receipt_cid,
        )
        self._store.receipts[receipt.receipt_cid] = receipt.to_dict()
        return result

    def _mutation_payload(self, request: ControlRequest) -> dict[str, Any]:
        _reject_non_admitting(request.parameters, "parameters")
        if request.operation == "spar.nominate":
            write_paths = _exact_paths(
                request.parameters.get("write_paths") or (), "write_paths"
            )
            validation = _commands(
                request.parameters.get("validation_commands") or (),
                "validation_commands",
            )
            evidence = _require_cids(
                request.parameters.get("evidence_cids") or (),
                "evidence_cids",
            ) if request.parameters.get("evidence_cids") else ()
            context_cid = _context_binding(request.parameters)
            return {
                "applied": True,
                "nominated": True,
                "write_paths": list(write_paths),
                "validation_commands": list(validation),
                "evidence_cids": list(evidence),
                "context_receipt_cid": context_cid,
                "can_authorize_completion": False,
                "can_authorize_transition": False,
            }
        if request.operation == "spar.request_review":
            context_cid = _context_binding(request.parameters)
            return {
                "applied": True,
                "nominated": True,
                "review_requested": True,
                "context_receipt_cid": context_cid,
                "independent_validation_required": True,
                "can_authorize_completion": False,
            }
        if request.operation == "spar.request_merge":
            floor = _receipt_floor_from(request.parameters)
            context_cid = floor["context_receipt_cid"]
            return {
                "applied": True,
                "nominated": True,
                "merge_requested": True,
                "merge_authorized": False,
                "receipt_floor": floor,
                "context_receipt_cid": context_cid,
                "current_authority_required": True,
                "can_authorize_completion": False,
                "can_authorize_transition": False,
            }
        if request.operation == "spar.rollback":
            context_cid = _context_binding(request.parameters)
            return {
                "applied": True,
                "nominated": True,
                "rollback_nominated": True,
                "rollback_applied": False,
                "context_receipt_cid": context_cid,
                "can_authorize_completion": False,
            }
        raise ControlSurfaceError(f"unknown SPAR operation: {request.operation}")

    def _execute_mutation(self, request: ControlRequest) -> ControlResult:
        fingerprint = request.fingerprint
        prior = self._store.applied.get(request.idempotency_key)
        if prior is not None:
            if self._store.fingerprints.get(request.idempotency_key) != fingerprint:
                receipt = compile_control_receipt(
                    operation=request.operation,
                    status=ControlStatus.CONFLICT.value,
                    payload={
                        "reason_code": TerminalKind.IDEMPOTENCY_CONFLICT.value,
                        "applied": False,
                    },
                )
                return ControlResult(
                    operation=request.operation,
                    ok=False,
                    status=ControlStatus.CONFLICT.value,
                    audit_id=prior.audit_id,
                    payload={
                        "reason_code": TerminalKind.IDEMPOTENCY_CONFLICT.value,
                        "applied": False,
                    },
                    receipt_cid=receipt.receipt_cid,
                )
            replayed = ControlResult.from_dict(
                {**prior.to_dict(), "idempotent_replay": True}
            )
            self._store.record_audit(
                {
                    "audit_id": replayed.audit_id,
                    "operation": request.operation,
                    "status": replayed.status,
                    "dry_run": request.dry_run,
                    "idempotency_key": request.idempotency_key,
                    "idempotent_replay": True,
                }
            )
            return replayed
        last_fence = self._store.fences.get(request.lease_id)
        if last_fence is not None and int(request.fencing_epoch) <= last_fence:
            payload = {
                "reason_code": "spar_stale_fence",
                "applied": False,
                "lease_id": request.lease_id,
                "fencing_epoch": request.fencing_epoch,
            }
            receipt = compile_control_receipt(
                operation=request.operation,
                status=ControlStatus.CONFLICT.value,
                payload={"reason_code": "spar_stale_fence", "applied": False},
            )
            audit_id = cid_for_dag_json(
                {
                    "schema": CONTROL_AUDIT_SCHEMA,
                    "operation": request.operation,
                    "idempotency_key": request.idempotency_key,
                    "status": ControlStatus.CONFLICT.value,
                    "dry_run": request.dry_run,
                    "fingerprint": fingerprint,
                }
            )
            self._store.record_audit(
                {
                    "audit_id": audit_id,
                    "operation": request.operation,
                    "status": ControlStatus.CONFLICT.value,
                    "dry_run": request.dry_run,
                    "idempotency_key": request.idempotency_key,
                }
            )
            return ControlResult(
                operation=request.operation,
                ok=False,
                status=ControlStatus.CONFLICT.value,
                audit_id=audit_id,
                payload=payload,
                receipt_cid=receipt.receipt_cid,
            )
        try:
            nominated = self._mutation_payload(request)
        except ControlSurfaceError as exc:
            if exc.reason_code in DECLARED_TERMINAL_KINDS:
                payload = {
                    "applied": False,
                    "nominated": False,
                    "reason_code": exc.reason_code,
                    "error": str(exc),
                    "typed_terminal": True,
                }
                status = ControlStatus.TYPED_TERMINAL.value
                receipt = compile_control_receipt(
                    operation=request.operation, status=status, payload=payload
                )
                audit_id = cid_for_dag_json(
                    {
                        "schema": CONTROL_AUDIT_SCHEMA,
                        "operation": request.operation,
                        "idempotency_key": request.idempotency_key,
                        "status": status,
                        "dry_run": request.dry_run,
                        "fingerprint": fingerprint,
                    }
                )
                self._store.record_audit(
                    {
                        "audit_id": audit_id,
                        "operation": request.operation,
                        "status": status,
                        "dry_run": request.dry_run,
                        "idempotency_key": request.idempotency_key,
                    }
                )
                return ControlResult(
                    operation=request.operation,
                    ok=False,
                    status=status,
                    audit_id=audit_id,
                    payload=payload,
                    receipt_cid=receipt.receipt_cid,
                )
            raise
        if request.dry_run:
            payload = {**nominated, "applied": False, "dry_run": True}
            status = ControlStatus.DRY_RUN.value
            receipt = compile_control_receipt(
                operation=request.operation, status=status, payload=payload
            )
            audit_id = cid_for_dag_json(
                {
                    "schema": CONTROL_AUDIT_SCHEMA,
                    "operation": request.operation,
                    "idempotency_key": request.idempotency_key,
                    "status": status,
                    "dry_run": True,
                    "fingerprint": fingerprint,
                }
            )
            self._store.record_audit(
                {
                    "audit_id": audit_id,
                    "operation": request.operation,
                    "status": status,
                    "dry_run": True,
                    "idempotency_key": request.idempotency_key,
                }
            )
            result = ControlResult(
                operation=request.operation,
                ok=True,
                status=status,
                audit_id=audit_id,
                payload=payload,
                receipt_cid=receipt.receipt_cid,
            )
            self._store.receipts[receipt.receipt_cid] = receipt.to_dict()
            return result
        payload = {**nominated, "applied": True, "dry_run": False}
        status = ControlStatus.NOMINATED.value
        receipt = compile_control_receipt(
            operation=request.operation, status=status, payload=payload
        )
        audit_id = cid_for_dag_json(
            {
                "schema": CONTROL_AUDIT_SCHEMA,
                "operation": request.operation,
                "idempotency_key": request.idempotency_key,
                "status": status,
                "dry_run": False,
                "fingerprint": fingerprint,
            }
        )
        result = ControlResult(
            operation=request.operation,
            ok=True,
            status=status,
            audit_id=audit_id,
            payload=payload,
            receipt_cid=receipt.receipt_cid,
        )
        self._store.applied[request.idempotency_key] = result
        self._store.fingerprints[request.idempotency_key] = fingerprint
        self._store.fences[request.lease_id] = int(request.fencing_epoch)
        self._store.receipts[receipt.receipt_cid] = receipt.to_dict()
        self._store.record_audit(
            {
                "audit_id": audit_id,
                "operation": request.operation,
                "status": status,
                "dry_run": False,
                "idempotency_key": request.idempotency_key,
            }
        )
        return result


def register_operations() -> dict[str, tuple[str, ...]]:
    """Publish the exact canonical SPAR operation catalog."""

    return {"read": READ_OPERATIONS, "mutation": MUTATION_OPERATIONS}


def dry_run_control(
    request: ControlRequest | Mapping[str, Any],
    *,
    service: SemanticRefactoringService | None = None,
) -> ControlResult:
    return (service or SemanticRefactoringService()).dry_run(request)


def execute_control(
    request: ControlRequest | Mapping[str, Any],
    *,
    service: SemanticRefactoringService | None = None,
) -> ControlResult:
    return (service or SemanticRefactoringService()).execute(request)


def provider_free_exports() -> tuple[str, ...]:
    return tuple(sorted(__all__))


def assert_not_competing_capsule_family() -> None:
    names = set(globals())
    overlap = names & _FORBIDDEN_CAPSULE_TYPE_NAMES
    if overlap:
        raise ControlSurfaceError(
            f"control surface must not define capsule types: {sorted(overlap)}"
        )


__all__ = [
    "ALL_OPERATIONS",
    "ANALYZER_ID",
    "AUTHORITY",
    "AUTHORITY_OWNER",
    "CONTEXT_COMPILER_REMAINS_AUTHORITY",
    "CONTROL_CONTRACT_VERSION",
    "CONTROL_RECEIPT_INTERFACE",
    "CONTROL_REQUEST_INTERFACE",
    "CONTROL_RESULT_INTERFACE",
    "DECLARED_MUTATION_OPERATIONS",
    "DECLARED_OPERATIONS",
    "DECLARED_READ_OPERATIONS",
    "DECLARED_RECEIPT_FLOOR",
    "DECLARED_TERMINAL_KINDS",
    "DIAGNOSTIC_REPORT_INTERFACE",
    "DIAGNOSTICS_ARE_AUTHORITATIVE",
    "DRY_RUN_IS_DETERMINISTIC",
    "DRY_RUN_MUTATES",
    "DUCKLAKE_IS_AUTHORITY",
    "EXISTING_ADAPTER_AUTHORITIES",
    "FORBIDDEN_CONTROL_NAMES",
    "GOAL_ID",
    "IDENTITY_EXCLUDED_FIELDS",
    "INDEPENDENT_VALIDATION_REQUIRED",
    "MARKDOWN_IS_NOT_COMPLETION",
    "MCP_NEVER_SHELLS",
    "MISSING_CONTEXT_MESSAGE",
    "MISSING_RECEIPT_FLOOR_MESSAGE",
    "MODEL_OUTPUT_IS_PROPOSAL_ONLY",
    "MUTATION_OPERATIONS",
    "MUTATION_SCOPE",
    "NEGATIVE_EVIDENCE_RETAINED",
    "NETWORK_DENIED",
    "NETWORK_DENY",
    "PREDECESSOR_TASK_IDS",
    "PROGRAM",
    "PROJECTION_CLUSTERING_IS_AUTHORITY",
    "RAW_SOURCE_REQUIRED",
    "READ_OPERATIONS",
    "REQUIRED_RECEIPT_FLOOR",
    "REQUIRED_ROLLOUT_CONSUMES_CONTEXT",
    "REQUIRED_ROLLOUT_EMITS_RECEIPT_FLOOR",
    "SEMANTIC_REFACTORING_SERVICE_INTERFACE",
    "SERVICE_CAN_AUTHORIZE_COMPLETION",
    "SERVICE_CAN_AUTHORIZE_TRANSITION",
    "SERVICE_CAN_CHANGE_MODE",
    "SERVICE_CAN_CREATE_AUTHORITY",
    "SERVICE_CAN_WEAKEN_VALIDATION",
    "SERVICE_IS_NOMINATION_ONLY",
    "SERVICE_WRITES_REPOSITORY",
    "SOFT_SIGNALS_CANNOT_OVERRIDE_HARD_CONSTRAINTS",
    "TASK_ID",
    "TEST_PASS_IS_NOT_COMPLETION",
    "TYPED_TERMINAL_INTERFACE",
    "VECTOR_SIMILARITY_IS_AUTHORITY",
    "WORKER_SELF_APPROVAL",
    "ControlAuthorization",
    "ControlBudget",
    "ControlReceipt",
    "ControlRequest",
    "ControlResult",
    "ControlStateStore",
    "ControlStatus",
    "ControlSurfaceError",
    "SemanticRefactoringService",
    "TerminalKind",
    "assert_not_competing_capsule_family",
    "compile_control_receipt",
    "control_surface_cid_profile",
    "control_surface_descriptor",
    "decode_canonical_receipt",
    "dry_run_control",
    "encode_canonical_receipt",
    "execute_control",
    "provider_free_exports",
    "register_operations",
]
