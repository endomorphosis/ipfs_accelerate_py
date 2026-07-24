"""Canonical control-plane contracts for the agent supervisor.

This module is intentionally transport-neutral.  Python, CLI, and MCP
adapters can all exchange the same immutable records without acquiring
different authority semantics.

Operation names are closed and each name has a fixed maximum authority.
Mutation requests fail closed unless they carry explicit roots, an exact
authorization decision, lease/fencing data, expected effects, and a scoped
idempotency key.  Result effect claims are checked against both the operation
authority and, when available, the originating request.
"""

from __future__ import annotations

import json
import posixpath
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


CONTROL_CONTRACT_VERSION = 1
CONTRACT_VERSION = CONTROL_CONTRACT_VERSION
SCHEMA_VERSION = CONTROL_CONTRACT_VERSION

CONTROL_BOUNDS_SCHEMA = "ipfs_accelerate_py/agent-supervisor/control-bounds@1"
EXPECTED_EFFECT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/expected-effect@1"
EFFECT_CLAIM_SCHEMA = "ipfs_accelerate_py/agent-supervisor/effect-claim@1"
IDEMPOTENCY_KEY_SCHEMA = "ipfs_accelerate_py/agent-supervisor/idempotency-key@1"
AUTHORIZATION_DECISION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/control-authorization-decision@1"
)
OPERATION_REQUEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/operation-request@1"
)
OPERATION_ERROR_SCHEMA = "ipfs_accelerate_py/agent-supervisor/operation-error@1"
DRY_RUN_PREVIEW_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/dry-run-preview@1"
)
OPERATION_RESULT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/operation-result@1"
OPERATION_CAPABILITY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/operation-capability@1"
)
CAPABILITY_REPORT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/capability-report@1"
)
CONTROL_DISCOVERY_MANIFEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/control-discovery-manifest@1"
)
CONTROL_DISCOVERY_RUNTIME_STATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/control-discovery-runtime-state@1"
)
CONTROL_DISCOVERY_OBSERVATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/control-discovery-observation@1"
)
CONTROL_DISCOVERY_SAFETY_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/control-discovery-safety-evidence@1"
)
LIFECYCLE_COMMAND_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/lifecycle-command@1"
)
CONTROL_SURFACE_PARITY_CASE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/control-surface-parity-case@1"
)
CONTROL_SURFACE_PARITY_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/control-surface-parity-evidence@1"
)
MUTATION_GUARD_REJECTION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/mutation-guard-rejection@1"
)
CONTROL_MUTATION_GUARD_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/control-mutation-guard-evidence@2"
)
CONTROL_MUTATION_RUNTIME_STATE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/control-mutation-runtime-state@1"
)
MUTATION_GUARD_EXECUTION_OBSERVATION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/mutation-guard-execution-observation@1"
)

# ASI-G070/ASI-G103: the requirement is emitted only by a validated
# ControlSurfaceParityEvidence record.  Merely mentioning this opaque ID is
# intentionally not completion evidence.
CONTROL_SURFACE_PARITY_REQUIREMENT_ID: Final[str] = (
    "031486194157679117987393491870400400279"
)
CONTROL_MUTATION_GUARD_REQUIREMENT_ID: Final[str] = (
    "184125100306462690646212311073240043804"
)
CONTROL_DISCOVERY_SAFETY_REQUIREMENT_ID: Final[str] = (
    "186773143401179107362964063059661378722"
)
# Compatibility spelling for callers which describe this boundary as
# discovery isolation rather than discovery safety.
CONTROL_DISCOVERY_ISOLATION_REQUIREMENT_ID: Final[str] = (
    CONTROL_DISCOVERY_SAFETY_REQUIREMENT_ID
)

ABSOLUTE_MAX_CONTROL_BYTES = 1_048_576
ABSOLUTE_MAX_CONTROL_ITEMS = 4_096
ABSOLUTE_MAX_CONTROL_DEPTH = 32
ABSOLUTE_MAX_CONTROL_TEXT_BYTES = 65_536


class ControlContractError(ContractValidationError):
    """Base exception for invalid control contracts."""


class UnknownOperationError(ControlContractError):
    """Raised when an operation is not in the closed operation vocabulary."""


class PathEscapeError(ControlContractError):
    """Raised when a root or repository-relative path can escape its scope."""


class MissingIdempotencyError(ControlContractError):
    """Raised when a mutation is not bound to an idempotency key."""


class AuthorizationBindingError(ControlContractError):
    """Raised when mutation authorization is absent, denied, or mismatched."""


class AuthorityViolationError(ControlContractError):
    """Raised when a request or result claims more authority than allowed."""


class ControlBoundsError(ControlContractError):
    """Raised when a control record exceeds a count, byte, or depth bound."""


class _ControlCanonicalContract(CanonicalContract):
    """Canonical mixin whose decoding failures retain the control error type."""

    @classmethod
    def from_json(cls, payload: str) -> "_ControlCanonicalContract":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ControlContractError("control contract JSON is malformed") from exc
        if not isinstance(value, Mapping):
            raise ControlContractError(
                "control contract JSON must contain an object"
            )
        decoder = getattr(cls, "from_dict", None)
        if decoder is None:
            raise ControlContractError(
                f"{cls.__name__} does not support from_dict"
            )
        return decoder(value)


class OperationAuthority(str, Enum):
    """Maximum semantic authority of an operation or effect."""

    READ = "read"
    PROPOSAL = "proposal"
    MUTATION = "mutation"

    @property
    def rank(self) -> int:
        return {
            OperationAuthority.READ: 0,
            OperationAuthority.PROPOSAL: 1,
            OperationAuthority.MUTATION: 2,
        }[self]

    def allows(self, other: "OperationAuthority | str") -> bool:
        return self.rank >= _authority(other).rank


Authority = OperationAuthority


class ControlSurface(str, Enum):
    """The independently invoked public supervisor control surfaces."""

    PYTHON = "python"
    CLI = "cli"
    MCP = "mcp"


class ControlBehaviorClass(str, Enum):
    """Behavior classes required before cross-surface parity can qualify."""

    READ_SUCCESS = "read_success"
    PROPOSAL_SUCCESS = "proposal_success"
    STABLE_FAILURE = "stable_failure"
    MUTATION_SUCCESS = "mutation_success"


class Operation(str, Enum):
    """Closed set of supervisor operations shared by all control surfaces."""

    CAPABILITIES = "capabilities"
    STATUS = "status"
    HEALTH = "health"
    METRICS = "metrics"
    GOALS = "goals"
    TASKS = "tasks"
    BUNDLES = "bundles"
    LANES = "lanes"
    EVENTS = "events"
    RECEIPTS = "receipts"
    CACHE_INSPECT = "cache_inspect"
    ARTIFACT_QUERY = "artifact_query"

    OBJECTIVE_PREVIEW = "objective_preview"
    PLAN = "plan"

    OBJECTIVE_REFINE = "objective_refine"
    OBJECTIVE_RECONCILE = "objective_reconcile"
    BACKLOG_REFILL = "backlog_refill"
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    DRAIN = "drain"
    STOP = "stop"
    RETRY = "retry"
    CANCEL = "cancel"
    QUARANTINE = "quarantine"
    VALIDATION_REPLAY = "validation_replay"

    @property
    def authority(self) -> OperationAuthority:
        return OPERATION_AUTHORITIES[self]

    @property
    def mutating(self) -> bool:
        return self.authority is OperationAuthority.MUTATION


READ_OPERATIONS: Final[frozenset[Operation]] = frozenset(
    {
        Operation.CAPABILITIES,
        Operation.STATUS,
        Operation.HEALTH,
        Operation.METRICS,
        Operation.GOALS,
        Operation.TASKS,
        Operation.BUNDLES,
        Operation.LANES,
        Operation.EVENTS,
        Operation.RECEIPTS,
        Operation.CACHE_INSPECT,
        Operation.ARTIFACT_QUERY,
    }
)
PROPOSAL_OPERATIONS: Final[frozenset[Operation]] = frozenset(
    {Operation.OBJECTIVE_PREVIEW, Operation.PLAN}
)
MUTATION_OPERATIONS: Final[frozenset[Operation]] = frozenset(
    set(Operation).difference(READ_OPERATIONS).difference(PROPOSAL_OPERATIONS)
)
OPERATION_AUTHORITIES: Final[Mapping[Operation, OperationAuthority]] = (
    MappingProxyType(
        {
            **{item: OperationAuthority.READ for item in READ_OPERATIONS},
            **{item: OperationAuthority.PROPOSAL for item in PROPOSAL_OPERATIONS},
            **{item: OperationAuthority.MUTATION for item in MUTATION_OPERATIONS},
        }
    )
)


class EffectKind(str, Enum):
    OBSERVE = "observe"
    PROPOSE = "propose"
    WRITE_STATE = "write_state"
    WRITE_REPOSITORY = "write_repository"
    DELETE_STATE = "delete_state"
    DELETE_REPOSITORY = "delete_repository"
    LIFECYCLE_TRANSITION = "lifecycle_transition"
    START_PROCESS = "start_process"
    STOP_PROCESS = "stop_process"
    EXECUTE_VALIDATION = "execute_validation"
    EMIT_AUDIT = "emit_audit"

    @property
    def authority(self) -> OperationAuthority:
        if self is EffectKind.OBSERVE:
            return OperationAuthority.READ
        if self is EffectKind.PROPOSE:
            return OperationAuthority.PROPOSAL
        return OperationAuthority.MUTATION


class AuthorizationVerdict(str, Enum):
    PERMIT = "permit"
    DENY = "deny"


class OperationStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    CONFLICT = "conflict"
    NOT_FOUND = "not_found"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    UNAVAILABLE = "unavailable"

    @property
    def successful(self) -> bool:
        return self is OperationStatus.SUCCEEDED


class ErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    UNKNOWN_OPERATION = "unknown_operation"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    STALE_TREE = "stale_tree"
    STALE_LEASE = "stale_lease"
    BOUNDS_EXCEEDED = "bounds_exceeded"
    PATH_ESCAPE = "path_escape"
    IDEMPOTENCY_REQUIRED = "idempotency_required"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    AUTHORITY_VIOLATION = "authority_violation"
    INVALID_LIFECYCLE_TRANSITION = "invalid_lifecycle_transition"
    UNAVAILABLE = "unavailable"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


class LifecycleAction(str, Enum):
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    DRAIN = "drain"
    STOP = "stop"
    RETRY = "retry"
    CANCEL = "cancel"
    QUARANTINE = "quarantine"

    @property
    def operation(self) -> Operation:
        return Operation(self.value)


LifecycleOperation = LifecycleAction


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = ABSOLUTE_MAX_CONTROL_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise ControlContractError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise ControlContractError(f"{name} must not be empty")
    if "\x00" in result:
        raise ControlContractError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > max_bytes:
        raise ControlBoundsError(f"{name} exceeds {max_bytes} UTF-8 bytes")
    return result


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ControlContractError(f"{name} must be a non-negative integer")
    return value


def _positive(value: Any, name: str) -> int:
    result = _nonnegative(value, name)
    if result < 1:
        raise ControlContractError(f"{name} must be at least 1")
    return result


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in kind)
        if kind is Operation:
            raise UnknownOperationError(
                f"unknown operation {value!r}; allowed operations: {allowed}"
            ) from exc
        raise ControlContractError(f"{name} must be one of: {allowed}") from exc


def _operation(value: Any) -> Operation:
    return _enum(value, Operation, "operation")


def _authority(value: Any) -> OperationAuthority:
    return _enum(value, OperationAuthority, "authority")


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    if not isinstance(payload, Mapping):
        raise ControlContractError("control contract payload must be an object")
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise ControlContractError(
            f"unsupported control schema {supplied!r}; expected {expected}"
        )
    version = payload.get("contract_version", payload.get("schema_version"))
    if version not in (None, CONTROL_CONTRACT_VERSION):
        raise ControlContractError("unsupported control contract version")


def _reject_unknown(
    payload: Mapping[str, Any], allowed: Iterable[str], noun: str
) -> None:
    if set(payload).difference(allowed):
        raise ControlContractError(
            f"{noun} contains unsupported fields; rebuild its canonical payload"
        )


def _identity(payload: Mapping[str, Any], actual: str, noun: str) -> None:
    claimed = payload.get("content_id")
    if claimed not in (None, "") and claimed != actual:
        raise ControlContractError(f"{noun} identity does not match payload")


def _absolute_root(value: Any, name: str) -> str:
    result = _text(value, name).replace("\\", "/")
    if not result.startswith("/"):
        raise PathEscapeError(f"{name} must be an absolute path")
    if ".." in PurePosixPath(result).parts:
        raise PathEscapeError(f"{name} must not traverse a parent")
    normalized = posixpath.normpath(result)
    if normalized == "/":
        raise PathEscapeError(f"{name} must not be the filesystem root")
    return normalized


def _relative_path(value: Any, name: str, *, required: bool = True) -> str:
    result = _text(value, name, required=required).replace("\\", "/")
    if not result:
        return ""
    candidate = PurePosixPath(result)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or (candidate.parts and candidate.parts[0].endswith(":"))
    ):
        raise PathEscapeError(f"{name} must be repository-relative")
    normalized = candidate.as_posix().removeprefix("./")
    if normalized in ("", "."):
        if required:
            raise PathEscapeError(f"{name} must not be empty")
        return ""
    return normalized


def _strings(
    values: Any,
    name: str,
    *,
    required: bool = False,
    maximum: int = ABSOLUTE_MAX_CONTROL_ITEMS,
) -> tuple[str, ...]:
    if values is None:
        source: Any = ()
    elif isinstance(values, str):
        source = (values,)
    elif isinstance(values, Sequence) and not isinstance(
        values, (bytes, bytearray, memoryview)
    ):
        source = values
    else:
        raise ControlContractError(f"{name} must be a sequence of strings")
    result = tuple(sorted({_text(item, name) for item in source}))
    if required and not result:
        raise ControlContractError(f"{name} must not be empty")
    if len(result) > maximum:
        raise ControlBoundsError(f"{name} exceeds its count bound")
    return result


def _paths(values: Any, name: str) -> tuple[str, ...]:
    return tuple(
        sorted(_relative_path(item, name) for item in _strings(values, name))
    )


_PATH_KEYS: Final[frozenset[str]] = frozenset(
    {
        "path",
        "paths",
        "repository_path",
        "repository_paths",
        "target_path",
        "target_paths",
        "artifact_path",
        "artifact_paths",
        "state_path",
        "state_paths",
        "worktree_path",
        "worktree_paths",
    }
)


def _freeze_value(
    value: Any,
    *,
    name: str,
    max_depth: int,
    max_items: int,
    max_text_bytes: int,
) -> Any:
    """Validate, path-check, and deeply freeze a canonical open value."""

    seen = 0

    def visit(item: Any, depth: int, key_name: str = "") -> Any:
        nonlocal seen
        seen += 1
        if seen > max_items:
            raise ControlBoundsError(f"{name} exceeds its item-count bound")
        if depth > max_depth:
            raise ControlBoundsError(f"{name} exceeds its nesting-depth bound")
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            return item
        if isinstance(item, str):
            text = _text(
                item, name, required=False, max_bytes=max_text_bytes
            )
            if key_name in _PATH_KEYS and not key_name.endswith("paths"):
                return _relative_path(text, key_name, required=False)
            return text
        if isinstance(item, Enum):
            return visit(item.value, depth, key_name)
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise ControlContractError(f"{name} object keys must be strings")
            frozen: dict[str, Any] = {}
            for key in sorted(item):
                normalized_key = _text(
                    key, f"{name} key", max_bytes=max_text_bytes
                )
                raw = item[key]
                if normalized_key in _PATH_KEYS and normalized_key.endswith("paths"):
                    if isinstance(raw, str) or not isinstance(raw, Sequence):
                        raise PathEscapeError(
                            f"{normalized_key} must be a sequence of paths"
                        )
                    frozen[normalized_key] = tuple(
                        _relative_path(member, normalized_key) for member in raw
                    )
                else:
                    frozen[normalized_key] = visit(
                        raw, depth + 1, normalized_key
                    )
            return MappingProxyType(frozen)
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray, memoryview)
        ):
            return tuple(visit(member, depth + 1, key_name) for member in item)
        raise ControlContractError(
            f"{name} contains unsupported value type {type(item).__name__}"
        )

    return visit(value, 0)


def _coerce_tuple(
    value: Any,
    kind: type,
    decoder: Any,
    name: str,
    *,
    maximum: int = ABSOLUTE_MAX_CONTROL_ITEMS,
) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray, memoryview)) or not isinstance(
        value, Sequence
    ):
        raise ControlContractError(f"{name} must be a sequence")
    if len(value) > maximum:
        raise ControlBoundsError(f"{name} exceeds its count bound")
    return tuple(
        item if isinstance(item, kind) else decoder(item) for item in value
    )


def _bounded_record(
    value: CanonicalContract,
    noun: str,
    *,
    maximum: int = ABSOLUTE_MAX_CONTROL_BYTES,
) -> None:
    if len(value.canonical_bytes()) > maximum:
        raise ControlBoundsError(f"{noun} exceeds its serialized-byte bound")


@dataclass(frozen=True)
class ControlBounds(_ControlCanonicalContract):
    """Limits carried by every operation request and result."""

    SCHEMA: ClassVar[str] = CONTROL_BOUNDS_SCHEMA

    max_items: int = 256
    max_serialized_bytes: int = 262_144
    max_depth: int = 8
    max_text_bytes: int = 8_192
    max_paths: int = 128
    max_effects: int = 64
    timeout_ms: int = 30_000

    def __post_init__(self) -> None:
        for name in (
            "max_items",
            "max_serialized_bytes",
            "max_depth",
            "max_text_bytes",
            "max_paths",
            "max_effects",
            "timeout_ms",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.max_items > ABSOLUTE_MAX_CONTROL_ITEMS:
            raise ControlBoundsError("max_items exceeds the absolute limit")
        if self.max_serialized_bytes > ABSOLUTE_MAX_CONTROL_BYTES:
            raise ControlBoundsError(
                "max_serialized_bytes exceeds the absolute limit"
            )
        if self.max_depth > ABSOLUTE_MAX_CONTROL_DEPTH:
            raise ControlBoundsError("max_depth exceeds the absolute limit")
        if self.max_text_bytes > ABSOLUTE_MAX_CONTROL_TEXT_BYTES:
            raise ControlBoundsError("max_text_bytes exceeds the absolute limit")
        if self.max_paths > self.max_items or self.max_effects > self.max_items:
            raise ControlBoundsError(
                "path and effect limits cannot exceed max_items"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "max_items": self.max_items,
            "max_serialized_bytes": self.max_serialized_bytes,
            "max_depth": self.max_depth,
            "max_text_bytes": self.max_text_bytes,
            "max_paths": self.max_paths,
            "max_effects": self.max_effects,
            "timeout_ms": self.timeout_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlBounds":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "max_items",
                "max_serialized_bytes",
                "max_bytes",
                "max_depth",
                "max_text_bytes",
                "max_paths",
                "max_effects",
                "timeout_ms",
                "content_id",
            },
            "control bounds",
        )
        defaults = cls()
        result = cls(
            max_items=payload.get("max_items", defaults.max_items),
            max_serialized_bytes=payload.get(
                "max_serialized_bytes",
                payload.get("max_bytes", defaults.max_serialized_bytes),
            ),
            max_depth=payload.get("max_depth", defaults.max_depth),
            max_text_bytes=payload.get(
                "max_text_bytes", defaults.max_text_bytes
            ),
            max_paths=payload.get("max_paths", defaults.max_paths),
            max_effects=payload.get("max_effects", defaults.max_effects),
            timeout_ms=payload.get("timeout_ms", defaults.timeout_ms),
        )
        _identity(payload, result.content_id, "control bounds")
        return result


@dataclass(frozen=True)
class ExpectedEffect(_ControlCanonicalContract):
    """An explicit effect a caller expects an operation to produce."""

    SCHEMA: ClassVar[str] = EXPECTED_EFFECT_SCHEMA

    effect_id: str
    kind: EffectKind
    resource: str
    paths: tuple[str, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect_id", _text(self.effect_id, "effect_id"))
        object.__setattr__(self, "kind", _enum(self.kind, EffectKind, "kind"))
        object.__setattr__(self, "resource", _text(self.resource, "resource"))
        object.__setattr__(self, "paths", _paths(self.paths, "paths"))
        object.__setattr__(
            self,
            "description",
            _text(self.description, "description", required=False),
        )
        _bounded_record(self, "expected effect")

    @property
    def authority(self) -> OperationAuthority:
        return self.kind.authority

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "effect_id": self.effect_id,
            "kind": self.kind,
            "authority": self.authority,
            "resource": self.resource,
            "paths": self.paths,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpectedEffect":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "effect_id",
                "kind",
                "authority",
                "resource",
                "paths",
                "description",
                "content_id",
            },
            "expected effect",
        )
        result = cls(
            effect_id=payload.get("effect_id", ""),
            kind=payload.get("kind", ""),
            resource=payload.get("resource", ""),
            paths=payload.get("paths", ()),
            description=payload.get("description", ""),
        )
        claimed_authority = payload.get("authority")
        if claimed_authority not in (None, "") and _authority(
            claimed_authority
        ) is not result.authority:
            raise AuthorityViolationError(
                "expected effect authority does not match its kind"
            )
        _identity(payload, result.content_id, "expected effect")
        return result


@dataclass(frozen=True)
class EffectClaim(_ControlCanonicalContract):
    """A bounded claim about an effect observed in an operation result."""

    SCHEMA: ClassVar[str] = EFFECT_CLAIM_SCHEMA

    effect_id: str
    kind: EffectKind
    resource: str
    paths: tuple[str, ...] = ()
    applied: bool = False
    receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect_id", _text(self.effect_id, "effect_id"))
        object.__setattr__(self, "kind", _enum(self.kind, EffectKind, "kind"))
        object.__setattr__(self, "resource", _text(self.resource, "resource"))
        object.__setattr__(self, "paths", _paths(self.paths, "paths"))
        if not isinstance(self.applied, bool):
            raise ControlContractError("applied must be a boolean")
        if self.applied and self.kind.authority is not OperationAuthority.MUTATION:
            raise AuthorityViolationError(
                "only mutation effects may be claimed as applied"
            )
        object.__setattr__(
            self,
            "receipt_id",
            _text(self.receipt_id, "receipt_id", required=False),
        )
        if self.applied and not self.receipt_id:
            raise ControlContractError(
                "an applied effect claim requires an audit receipt identity"
            )
        _bounded_record(self, "effect claim")

    @property
    def authority(self) -> OperationAuthority:
        return self.kind.authority

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "effect_id": self.effect_id,
            "kind": self.kind,
            "authority": self.authority,
            "resource": self.resource,
            "paths": self.paths,
            "applied": self.applied,
            "receipt_id": self.receipt_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EffectClaim":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "effect_id",
                "kind",
                "authority",
                "resource",
                "paths",
                "applied",
                "receipt_id",
                "content_id",
            },
            "effect claim",
        )
        result = cls(
            effect_id=payload.get("effect_id", ""),
            kind=payload.get("kind", ""),
            resource=payload.get("resource", ""),
            paths=payload.get("paths", ()),
            applied=payload.get("applied", False),
            receipt_id=payload.get("receipt_id", ""),
        )
        claimed_authority = payload.get("authority")
        if claimed_authority not in (None, "") and _authority(
            claimed_authority
        ) is not result.authority:
            raise AuthorityViolationError(
                "effect claim authority does not match its kind"
            )
        _identity(payload, result.content_id, "effect claim")
        return result


@dataclass(frozen=True)
class IdempotencyKey(_ControlCanonicalContract):
    """A caller-scoped mutation replay key."""

    SCHEMA: ClassVar[str] = IDEMPOTENCY_KEY_SCHEMA

    key: str
    operation: Operation
    caller: str
    repository_id: str
    objective_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _text(self.key, "idempotency key"))
        if len(self.key.encode("utf-8")) > 256:
            raise ControlBoundsError("idempotency key exceeds 256 UTF-8 bytes")
        object.__setattr__(self, "operation", _operation(self.operation))
        for name in ("caller", "repository_id", "objective_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        _bounded_record(self, "idempotency key")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "key": self.key,
            "operation": self.operation,
            "caller": self.caller,
            "repository_id": self.repository_id,
            "objective_id": self.objective_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IdempotencyKey":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "key",
                "operation",
                "caller",
                "repository_id",
                "objective_id",
                "content_id",
            },
            "idempotency key",
        )
        result = cls(
            key=payload.get("key", ""),
            operation=payload.get("operation", ""),
            caller=payload.get("caller", ""),
            repository_id=payload.get("repository_id", ""),
            objective_id=payload.get("objective_id", ""),
        )
        _identity(payload, result.content_id, "idempotency key")
        return result


@dataclass(frozen=True)
class AuthorizationDecision(_ControlCanonicalContract):
    """An exact, policy-produced authorization for one operation binding."""

    SCHEMA: ClassVar[str] = AUTHORIZATION_DECISION_SCHEMA

    verdict: AuthorizationVerdict
    operation: Operation
    granted_authority: OperationAuthority | None
    repository_root: str
    state_root: str
    repository_id: str
    tree_id: str
    objective_id: str
    objective_revision: str
    policy_id: str
    policy_revision: str
    caller: str
    lease_id: str = ""
    fencing_epoch: int | None = None
    authorized_effect_ids: tuple[str, ...] = ()
    reason_code: str = ""
    grant_ids: tuple[str, ...] = ()
    evaluated_at_ms: int = 0
    expires_at_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "verdict",
            _enum(self.verdict, AuthorizationVerdict, "verdict"),
        )
        object.__setattr__(self, "operation", _operation(self.operation))
        if self.granted_authority is not None:
            object.__setattr__(
                self,
                "granted_authority",
                _authority(self.granted_authority),
            )
        object.__setattr__(
            self,
            "repository_root",
            _absolute_root(self.repository_root, "repository_root"),
        )
        object.__setattr__(
            self, "state_root", _absolute_root(self.state_root, "state_root")
        )
        for name in (
            "repository_id",
            "tree_id",
            "objective_id",
            "objective_revision",
            "policy_id",
            "policy_revision",
            "caller",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self, "lease_id", _text(self.lease_id, "lease_id", required=False)
        )
        if self.fencing_epoch is not None:
            object.__setattr__(
                self,
                "fencing_epoch",
                _nonnegative(self.fencing_epoch, "fencing_epoch"),
            )
        object.__setattr__(
            self,
            "authorized_effect_ids",
            _strings(
                self.authorized_effect_ids,
                "authorized_effect_ids",
                required=self.verdict is AuthorizationVerdict.PERMIT
                and self.operation.mutating,
            ),
        )
        object.__setattr__(
            self,
            "reason_code",
            _text(
                self.reason_code,
                "reason_code",
                required=self.verdict is AuthorizationVerdict.DENY,
            ),
        )
        object.__setattr__(
            self, "grant_ids", _strings(self.grant_ids, "grant_ids")
        )
        object.__setattr__(
            self,
            "evaluated_at_ms",
            _nonnegative(self.evaluated_at_ms, "evaluated_at_ms"),
        )
        if self.expires_at_ms is not None:
            object.__setattr__(
                self,
                "expires_at_ms",
                _nonnegative(self.expires_at_ms, "expires_at_ms"),
            )
            if self.expires_at_ms <= self.evaluated_at_ms:
                raise ControlContractError(
                    "expires_at_ms must follow evaluated_at_ms"
                )
        if self.verdict is AuthorizationVerdict.PERMIT:
            if self.granted_authority is None:
                raise AuthorizationBindingError(
                    "permit decisions require granted_authority"
                )
            if not self.operation.authority.allows(self.granted_authority):
                raise AuthorityViolationError(
                    "authorization grants more authority than the operation"
                )
        elif self.granted_authority is not None:
            raise AuthorizationBindingError(
                "deny decisions must not grant authority"
            )
        _bounded_record(self, "authorization decision")

    @property
    def permitted(self) -> bool:
        return self.verdict is AuthorizationVerdict.PERMIT

    @property
    def decision_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "verdict": self.verdict,
            "operation": self.operation,
            "granted_authority": self.granted_authority,
            "repository_root": self.repository_root,
            "state_root": self.state_root,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "caller": self.caller,
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "authorized_effect_ids": self.authorized_effect_ids,
            "reason_code": self.reason_code,
            "grant_ids": self.grant_ids,
            "evaluated_at_ms": self.evaluated_at_ms,
            "expires_at_ms": self.expires_at_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorizationDecision":
        _schema(payload, cls.SCHEMA)
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "verdict",
            "operation",
            "granted_authority",
            "repository_root",
            "state_root",
            "repository_id",
            "tree_id",
            "objective_id",
            "objective_revision",
            "policy_id",
            "policy_revision",
            "caller",
            "lease_id",
            "fencing_epoch",
            "authorized_effect_ids",
            "reason_code",
            "grant_ids",
            "evaluated_at_ms",
            "expires_at_ms",
            "content_id",
        }
        _reject_unknown(payload, allowed, "authorization decision")
        result = cls(
            verdict=payload.get("verdict", ""),
            operation=payload.get("operation", ""),
            granted_authority=payload.get("granted_authority"),
            repository_root=payload.get("repository_root", ""),
            state_root=payload.get("state_root", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            objective_id=payload.get("objective_id", ""),
            objective_revision=payload.get("objective_revision", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            caller=payload.get("caller", ""),
            lease_id=payload.get("lease_id", ""),
            fencing_epoch=payload.get("fencing_epoch"),
            authorized_effect_ids=payload.get("authorized_effect_ids", ()),
            reason_code=payload.get("reason_code", ""),
            grant_ids=payload.get("grant_ids", ()),
            evaluated_at_ms=payload.get("evaluated_at_ms", 0),
            expires_at_ms=payload.get("expires_at_ms"),
        )
        _identity(payload, result.content_id, "authorization decision")
        return result


@dataclass(frozen=True)
class OperationRequest(_ControlCanonicalContract):
    """One fully bound request to the shared supervisor control service."""

    SCHEMA: ClassVar[str] = OPERATION_REQUEST_SCHEMA

    operation: Operation
    repository_root: str
    state_root: str
    repository_id: str
    tree_id: str
    objective_id: str
    objective_revision: str
    policy_id: str
    policy_revision: str
    caller: str
    bounds: ControlBounds = dataclass_field(default_factory=ControlBounds)
    expected_effects: tuple[ExpectedEffect, ...] = ()
    parameters: Mapping[str, Any] = dataclass_field(default_factory=dict)
    dry_run: bool = False
    idempotency: IdempotencyKey | None = None
    authorization: AuthorizationDecision | None = None
    lease_id: str = ""
    fencing_epoch: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _operation(self.operation))
        object.__setattr__(
            self,
            "repository_root",
            _absolute_root(self.repository_root, "repository_root"),
        )
        object.__setattr__(
            self, "state_root", _absolute_root(self.state_root, "state_root")
        )
        for name in (
            "repository_id",
            "tree_id",
            "objective_id",
            "objective_revision",
            "policy_id",
            "policy_revision",
            "caller",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        bounds = self.bounds
        if not isinstance(bounds, ControlBounds):
            if not isinstance(bounds, Mapping):
                raise ControlContractError("bounds must be ControlBounds")
            bounds = ControlBounds.from_dict(bounds)
        object.__setattr__(self, "bounds", bounds)
        effects = _coerce_tuple(
            self.expected_effects,
            ExpectedEffect,
            ExpectedEffect.from_dict,
            "expected_effects",
        )
        if len(effects) > bounds.max_effects:
            raise ControlBoundsError("request exceeds its effect-count bound")
        if sum(len(item.paths) for item in effects) > bounds.max_paths:
            raise ControlBoundsError("request exceeds its path-count bound")
        effect_ids = [item.effect_id for item in effects]
        if len(effect_ids) != len(set(effect_ids)):
            raise ControlContractError("expected effect IDs must be unique")
        # A dry-run mutation is allowed to *describe* mutation-shaped expected
        # effects.  Its result remains proposal-only and cannot claim any such
        # effect was applied.
        maximum_authority = self.operation.authority
        for effect in effects:
            if not maximum_authority.allows(effect.authority):
                raise AuthorityViolationError(
                    "expected effect lies outside the request authority"
                )
        if self.operation.mutating and not self.dry_run and not effects:
            raise AuthorityViolationError(
                "mutation requests must declare expected effects"
            )
        if (
            self.operation.mutating
            and not self.dry_run
            and not any(
                effect.authority is OperationAuthority.MUTATION
                for effect in effects
            )
        ):
            raise AuthorityViolationError(
                "mutation requests must declare a mutation effect"
            )
        object.__setattr__(
            self,
            "expected_effects",
            tuple(sorted(effects, key=lambda item: item.effect_id)),
        )
        if not isinstance(self.parameters, Mapping):
            raise ControlContractError("parameters must be a mapping")
        frozen_parameters = _freeze_value(
            self.parameters,
            name="parameters",
            max_depth=bounds.max_depth,
            max_items=bounds.max_items,
            max_text_bytes=bounds.max_text_bytes,
        )
        object.__setattr__(self, "parameters", frozen_parameters)
        if not isinstance(self.dry_run, bool):
            raise ControlContractError("dry_run must be a boolean")
        object.__setattr__(
            self, "lease_id", _text(self.lease_id, "lease_id", required=False)
        )
        if self.fencing_epoch is not None:
            object.__setattr__(
                self,
                "fencing_epoch",
                _nonnegative(self.fencing_epoch, "fencing_epoch"),
            )
        idempotency = self.idempotency
        if idempotency is not None and not isinstance(idempotency, IdempotencyKey):
            if not isinstance(idempotency, Mapping):
                raise ControlContractError(
                    "idempotency must be an IdempotencyKey"
                )
            idempotency = IdempotencyKey.from_dict(idempotency)
        object.__setattr__(self, "idempotency", idempotency)
        authorization = self.authorization
        if authorization is not None and not isinstance(
            authorization, AuthorizationDecision
        ):
            if not isinstance(authorization, Mapping):
                raise ControlContractError(
                    "authorization must be an AuthorizationDecision"
                )
            authorization = AuthorizationDecision.from_dict(authorization)
        object.__setattr__(self, "authorization", authorization)
        if self.operation.mutating and not self.dry_run:
            self._validate_mutation_bindings()
        else:
            if idempotency is not None:
                self._validate_idempotency(idempotency)
            if authorization is not None:
                self._validate_optional_authorization(authorization)
        if len(self.canonical_bytes()) > bounds.max_serialized_bytes:
            raise ControlBoundsError(
                "operation request exceeds its serialized-byte bound"
            )

    @property
    def authority(self) -> OperationAuthority:
        return self.operation.authority

    @property
    def effective_authority(self) -> OperationAuthority:
        if self.dry_run and self.operation.mutating:
            return OperationAuthority.PROPOSAL
        return self.operation.authority

    @property
    def request_id(self) -> str:
        return self.content_id

    @property
    def idempotency_key(self) -> str:
        return self.idempotency.key if self.idempotency else ""

    def _validate_idempotency(self, key: IdempotencyKey) -> None:
        expected = (
            self.operation,
            self.caller,
            self.repository_id,
            self.objective_id,
        )
        actual = (
            key.operation,
            key.caller,
            key.repository_id,
            key.objective_id,
        )
        if actual != expected:
            raise MissingIdempotencyError(
                "idempotency key scope does not match the request"
            )

    def _validate_mutation_bindings(self) -> None:
        if self.idempotency is None:
            raise MissingIdempotencyError(
                "mutation requests require an idempotency key"
            )
        self._validate_idempotency(self.idempotency)
        if not self.lease_id or self.fencing_epoch is None:
            raise AuthorizationBindingError(
                "mutation requests require lease_id and fencing_epoch"
            )
        decision = self.authorization
        if decision is None or not decision.permitted:
            raise AuthorizationBindingError(
                "mutation requests require a permit authorization decision"
            )
        if (
            decision.granted_authority is None
            or not decision.granted_authority.allows(OperationAuthority.MUTATION)
        ):
            raise AuthorizationBindingError(
                "mutation authorization does not grant mutation authority"
            )
        binding_names = (
            "operation",
            "repository_root",
            "state_root",
            "repository_id",
            "tree_id",
            "objective_id",
            "objective_revision",
            "policy_id",
            "policy_revision",
            "caller",
            "lease_id",
            "fencing_epoch",
        )
        if any(
            getattr(decision, name) != getattr(self, name)
            for name in binding_names
        ):
            raise AuthorizationBindingError(
                "authorization decision binding does not match the request"
            )
        expected_ids = {item.effect_id for item in self.expected_effects}
        allowed_ids = set(decision.authorized_effect_ids)
        if "*" not in allowed_ids and not expected_ids.issubset(allowed_ids):
            raise AuthorizationBindingError(
                "authorization does not cover every expected effect"
            )

    def _validate_optional_authorization(
        self, decision: AuthorizationDecision
    ) -> None:
        """Validate a supplied decision even when the operation does not need it."""

        if not decision.permitted or decision.granted_authority is None:
            raise AuthorizationBindingError(
                "a supplied authorization decision must permit the request"
            )
        binding_names = (
            "operation",
            "repository_root",
            "state_root",
            "repository_id",
            "tree_id",
            "objective_id",
            "objective_revision",
            "policy_id",
            "policy_revision",
            "caller",
        )
        if any(
            getattr(decision, name) != getattr(self, name)
            for name in binding_names
        ):
            raise AuthorizationBindingError(
                "authorization decision binding does not match the request"
            )
        if not decision.granted_authority.allows(self.effective_authority):
            raise AuthorizationBindingError(
                "authorization does not grant the request authority"
            )
        if self.lease_id and decision.lease_id != self.lease_id:
            raise AuthorizationBindingError(
                "authorization lease does not match the request"
            )
        if (
            self.fencing_epoch is not None
            and decision.fencing_epoch != self.fencing_epoch
        ):
            raise AuthorizationBindingError(
                "authorization fencing epoch does not match the request"
            )
        expected_ids = {item.effect_id for item in self.expected_effects}
        allowed_ids = set(decision.authorized_effect_ids)
        if (
            expected_ids
            and "*" not in allowed_ids
            and not expected_ids.issubset(allowed_ids)
        ):
            raise AuthorizationBindingError(
                "authorization does not cover every expected effect"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "operation": self.operation,
            "authority": self.authority,
            "effective_authority": self.effective_authority,
            "repository_root": self.repository_root,
            "state_root": self.state_root,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "caller": self.caller,
            "bounds": self.bounds.to_record(),
            "expected_effects": tuple(
                item.to_record() for item in self.expected_effects
            ),
            "parameters": self.parameters,
            "dry_run": self.dry_run,
            "idempotency": (
                self.idempotency.to_record() if self.idempotency else None
            ),
            "authorization": (
                self.authorization.to_record() if self.authorization else None
            ),
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperationRequest":
        _schema(payload, cls.SCHEMA)
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "operation",
            "authority",
            "effective_authority",
            "repository_root",
            "state_root",
            "repository_id",
            "tree_id",
            "objective_id",
            "objective_revision",
            "policy_id",
            "policy_revision",
            "caller",
            "bounds",
            "expected_effects",
            "parameters",
            "dry_run",
            "idempotency",
            "idempotency_key",
            "authorization",
            "lease_id",
            "fencing_epoch",
            "content_id",
        }
        _reject_unknown(payload, allowed, "operation request")
        operation = _operation(payload.get("operation", ""))
        raw_idempotency = payload.get("idempotency")
        if raw_idempotency is None and payload.get("idempotency_key"):
            raw_idempotency = IdempotencyKey(
                key=payload["idempotency_key"],
                operation=operation,
                caller=payload.get("caller", ""),
                repository_id=payload.get("repository_id", ""),
                objective_id=payload.get("objective_id", ""),
            )
        result = cls(
            operation=operation,
            repository_root=payload.get("repository_root", ""),
            state_root=payload.get("state_root", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            objective_id=payload.get("objective_id", ""),
            objective_revision=payload.get("objective_revision", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            caller=payload.get("caller", ""),
            bounds=payload.get("bounds") or ControlBounds(),
            expected_effects=payload.get("expected_effects", ()),
            parameters=payload.get("parameters") or {},
            dry_run=payload.get("dry_run", False),
            idempotency=raw_idempotency,
            authorization=payload.get("authorization"),
            lease_id=payload.get("lease_id", ""),
            fencing_epoch=payload.get("fencing_epoch"),
        )
        for name, actual in (
            ("authority", result.authority),
            ("effective_authority", result.effective_authority),
        ):
            claimed = payload.get(name)
            if claimed not in (None, "") and _authority(claimed) is not actual:
                raise AuthorityViolationError(
                    f"request {name} does not match its operation"
                )
        _identity(payload, result.content_id, "operation request")
        return result


@dataclass(frozen=True)
class OperationError(_ControlCanonicalContract):
    """Stable machine-readable error returned by every control surface."""

    SCHEMA: ClassVar[str] = OPERATION_ERROR_SCHEMA

    code: ErrorCode
    message: str
    retryable: bool = False
    field: str = ""
    details: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _enum(self.code, ErrorCode, "code"))
        object.__setattr__(
            self, "message", _text(self.message, "message", max_bytes=2_048)
        )
        if not isinstance(self.retryable, bool):
            raise ControlContractError("retryable must be a boolean")
        object.__setattr__(
            self, "field", _text(self.field, "field", required=False)
        )
        if not isinstance(self.details, Mapping):
            raise ControlContractError("error details must be a mapping")
        details = _freeze_value(
            self.details,
            name="error details",
            max_depth=4,
            max_items=64,
            max_text_bytes=2_048,
        )
        if len(canonical_json_bytes(details)) > 16_384:
            raise ControlBoundsError("error details exceed 16384 bytes")
        object.__setattr__(self, "details", details)
        _bounded_record(self, "operation error", maximum=32_768)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "field": self.field,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperationError":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "code",
                "message",
                "retryable",
                "field",
                "details",
                "content_id",
            },
            "operation error",
        )
        result = cls(
            code=payload.get("code", ""),
            message=payload.get("message", ""),
            retryable=payload.get("retryable", False),
            field=payload.get("field", ""),
            details=payload.get("details") or {},
        )
        _identity(payload, result.content_id, "operation error")
        return result


@dataclass(frozen=True)
class DryRunPreview(_ControlCanonicalContract):
    """Non-authoritative preview of the effects of one dry-run request."""

    SCHEMA: ClassVar[str] = DRY_RUN_PREVIEW_SCHEMA

    request_id: str
    operation: Operation
    repository_id: str
    tree_id: str
    objective_id: str
    policy_id: str
    caller: str
    expected_effects: tuple[ExpectedEffect, ...] = ()
    checks: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    would_change: bool = False

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "repository_id",
            "tree_id",
            "objective_id",
            "policy_id",
            "caller",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "operation", _operation(self.operation))
        effects = _coerce_tuple(
            self.expected_effects,
            ExpectedEffect,
            ExpectedEffect.from_dict,
            "expected_effects",
        )
        if any(
            effect.authority is OperationAuthority.READ for effect in effects
        ):
            raise AuthorityViolationError(
                "a dry-run preview cannot present observations as changes"
            )
        object.__setattr__(
            self,
            "expected_effects",
            tuple(sorted(effects, key=lambda item: item.effect_id)),
        )
        object.__setattr__(self, "checks", _strings(self.checks, "checks"))
        object.__setattr__(
            self, "warnings", _strings(self.warnings, "warnings")
        )
        if not isinstance(self.would_change, bool):
            raise ControlContractError("would_change must be a boolean")
        _bounded_record(self, "dry-run preview")

    @property
    def authority(self) -> OperationAuthority:
        return OperationAuthority.PROPOSAL

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "request_id": self.request_id,
            "operation": self.operation,
            "authority": self.authority,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "caller": self.caller,
            "expected_effects": tuple(
                item.to_record() for item in self.expected_effects
            ),
            "checks": self.checks,
            "warnings": self.warnings,
            "would_change": self.would_change,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DryRunPreview":
        _schema(payload, cls.SCHEMA)
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "request_id",
            "operation",
            "authority",
            "repository_id",
            "tree_id",
            "objective_id",
            "policy_id",
            "caller",
            "expected_effects",
            "checks",
            "warnings",
            "would_change",
            "content_id",
        }
        _reject_unknown(payload, allowed, "dry-run preview")
        result = cls(
            request_id=payload.get("request_id", ""),
            operation=payload.get("operation", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            objective_id=payload.get("objective_id", ""),
            policy_id=payload.get("policy_id", ""),
            caller=payload.get("caller", ""),
            expected_effects=payload.get("expected_effects", ()),
            checks=payload.get("checks", ()),
            warnings=payload.get("warnings", ()),
            would_change=payload.get("would_change", False),
        )
        claimed = payload.get("authority")
        if claimed not in (None, "") and _authority(
            claimed
        ) is not OperationAuthority.PROPOSAL:
            raise AuthorityViolationError(
                "dry-run previews have proposal authority only"
            )
        _identity(payload, result.content_id, "dry-run preview")
        return result


@dataclass(frozen=True)
class OperationResult(_ControlCanonicalContract):
    """Bounded result with effect claims constrained by operation authority."""

    SCHEMA: ClassVar[str] = OPERATION_RESULT_SCHEMA

    request_id: str
    operation: Operation
    authority: OperationAuthority
    status: OperationStatus
    repository_id: str
    tree_id: str
    objective_id: str
    policy_id: str
    caller: str
    bounds: ControlBounds = dataclass_field(default_factory=ControlBounds)
    data: Mapping[str, Any] = dataclass_field(default_factory=dict)
    effects: tuple[EffectClaim, ...] = ()
    error: OperationError | None = None
    preview: DryRunPreview | None = None
    idempotency_key: str = ""
    audit_receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id"))
        object.__setattr__(self, "operation", _operation(self.operation))
        object.__setattr__(self, "authority", _authority(self.authority))
        if not self.operation.authority.allows(self.authority):
            raise AuthorityViolationError(
                "result authority lies outside the operation authority"
            )
        object.__setattr__(
            self, "status", _enum(self.status, OperationStatus, "status")
        )
        for name in (
            "repository_id",
            "tree_id",
            "objective_id",
            "policy_id",
            "caller",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        bounds = self.bounds
        if not isinstance(bounds, ControlBounds):
            if not isinstance(bounds, Mapping):
                raise ControlContractError("bounds must be ControlBounds")
            bounds = ControlBounds.from_dict(bounds)
        object.__setattr__(self, "bounds", bounds)
        if not isinstance(self.data, Mapping):
            raise ControlContractError("result data must be a mapping")
        object.__setattr__(
            self,
            "data",
            _freeze_value(
                self.data,
                name="result data",
                max_depth=bounds.max_depth,
                max_items=bounds.max_items,
                max_text_bytes=bounds.max_text_bytes,
            ),
        )
        effects = _coerce_tuple(
            self.effects, EffectClaim, EffectClaim.from_dict, "effects"
        )
        if len(effects) > bounds.max_effects:
            raise ControlBoundsError("result exceeds its effect-count bound")
        if sum(len(item.paths) for item in effects) > bounds.max_paths:
            raise ControlBoundsError("result exceeds its path-count bound")
        effect_ids = [item.effect_id for item in effects]
        if len(effect_ids) != len(set(effect_ids)):
            raise ControlContractError("result effect IDs must be unique")
        for effect in effects:
            if not self.authority.allows(effect.authority):
                raise AuthorityViolationError(
                    "result effect claim lies outside result authority"
                )
            if effect.applied and self.authority is not OperationAuthority.MUTATION:
                raise AuthorityViolationError(
                    "non-mutation results cannot claim applied effects"
                )
        object.__setattr__(
            self, "effects", tuple(sorted(effects, key=lambda item: item.effect_id))
        )
        error = self.error
        if error is not None and not isinstance(error, OperationError):
            if not isinstance(error, Mapping):
                raise ControlContractError("error must be an OperationError")
            error = OperationError.from_dict(error)
        object.__setattr__(self, "error", error)
        if self.status.successful and error is not None:
            raise ControlContractError("successful results must not contain an error")
        if not self.status.successful and error is None:
            raise ControlContractError("unsuccessful results require a typed error")
        preview = self.preview
        if preview is not None and not isinstance(preview, DryRunPreview):
            if not isinstance(preview, Mapping):
                raise ControlContractError("preview must be a DryRunPreview")
            preview = DryRunPreview.from_dict(preview)
        object.__setattr__(self, "preview", preview)
        if preview is not None:
            if (
                self.authority is not OperationAuthority.PROPOSAL
                or preview.request_id != self.request_id
                or preview.operation is not self.operation
            ):
                raise AuthorityViolationError(
                    "preview does not match the proposal result"
                )
        for name in ("idempotency_key", "audit_receipt_id"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, required=False),
            )
        if any(effect.applied for effect in effects) and not self.audit_receipt_id:
            raise ControlContractError(
                "applied mutation results require an audit receipt"
            )
        if any(
            effect.applied and effect.receipt_id != self.audit_receipt_id
            for effect in effects
        ):
            raise ControlContractError(
                "applied effect receipt must match the result audit receipt"
            )
        if len(self.canonical_bytes()) > bounds.max_serialized_bytes:
            raise ControlBoundsError(
                "operation result exceeds its serialized-byte bound"
            )

    @property
    def result_id(self) -> str:
        return self.content_id

    @property
    def succeeded(self) -> bool:
        return self.status.successful

    def validate_against(self, request: OperationRequest) -> None:
        """Fail closed unless this result is an exact projection of ``request``."""

        if not isinstance(request, OperationRequest):
            raise ControlContractError("request must be an OperationRequest")
        comparisons = (
            (self.request_id, request.request_id),
            (self.operation, request.operation),
            (self.repository_id, request.repository_id),
            (self.tree_id, request.tree_id),
            (self.objective_id, request.objective_id),
            (self.policy_id, request.policy_id),
            (self.caller, request.caller),
            (self.authority, request.effective_authority),
            (self.bounds, request.bounds),
        )
        if any(actual != expected for actual, expected in comparisons):
            raise AuthorityViolationError("result binding does not match request")
        expected = {item.effect_id: item for item in request.expected_effects}
        for claim in self.effects:
            declared = expected.get(claim.effect_id)
            if declared is None:
                raise AuthorityViolationError(
                    "result claims an effect not declared by the request"
                )
            if (
                claim.kind is not declared.kind
                or claim.resource != declared.resource
                or claim.paths != declared.paths
            ):
                raise AuthorityViolationError(
                    "result effect claim exceeds its declared shape"
                )
        if request.dry_run and any(item.applied for item in self.effects):
            raise AuthorityViolationError(
                "dry-run results cannot claim applied effects"
            )
        if request.operation.mutating and not request.dry_run:
            if self.idempotency_key != request.idempotency_key:
                raise AuthorityViolationError(
                    "mutation result idempotency key does not match request"
                )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "request_id": self.request_id,
            "operation": self.operation,
            "authority": self.authority,
            "status": self.status,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "caller": self.caller,
            "bounds": self.bounds.to_record(),
            "data": self.data,
            "effects": tuple(item.to_record() for item in self.effects),
            "error": self.error.to_record() if self.error else None,
            "preview": self.preview.to_record() if self.preview else None,
            "idempotency_key": self.idempotency_key,
            "audit_receipt_id": self.audit_receipt_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperationResult":
        _schema(payload, cls.SCHEMA)
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "request_id",
            "operation",
            "authority",
            "status",
            "repository_id",
            "tree_id",
            "objective_id",
            "policy_id",
            "caller",
            "bounds",
            "data",
            "effects",
            "error",
            "preview",
            "idempotency_key",
            "audit_receipt_id",
            "content_id",
        }
        _reject_unknown(payload, allowed, "operation result")
        result = cls(
            request_id=payload.get("request_id", ""),
            operation=payload.get("operation", ""),
            authority=payload.get("authority", ""),
            status=payload.get("status", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            objective_id=payload.get("objective_id", ""),
            policy_id=payload.get("policy_id", ""),
            caller=payload.get("caller", ""),
            bounds=payload.get("bounds") or ControlBounds(),
            data=payload.get("data") or {},
            effects=payload.get("effects", ()),
            error=payload.get("error"),
            preview=payload.get("preview"),
            idempotency_key=payload.get("idempotency_key", ""),
            audit_receipt_id=payload.get("audit_receipt_id", ""),
        )
        _identity(payload, result.content_id, "operation result")
        return result


@dataclass(frozen=True)
class OperationCapability(_ControlCanonicalContract):
    """Advertised support and bounds for one operation."""

    SCHEMA: ClassVar[str] = OPERATION_CAPABILITY_SCHEMA

    operation: Operation
    authority: OperationAuthority
    bounds: ControlBounds = dataclass_field(default_factory=ControlBounds)
    supports_dry_run: bool = False
    requires_idempotency: bool = False
    requires_authorization: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _operation(self.operation))
        object.__setattr__(self, "authority", _authority(self.authority))
        if self.authority is not self.operation.authority:
            raise AuthorityViolationError(
                "capability authority must match the operation registry"
            )
        if not isinstance(self.bounds, ControlBounds):
            if not isinstance(self.bounds, Mapping):
                raise ControlContractError("bounds must be ControlBounds")
            object.__setattr__(
                self, "bounds", ControlBounds.from_dict(self.bounds)
            )
        for name in (
            "supports_dry_run",
            "requires_idempotency",
            "requires_authorization",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ControlContractError(f"{name} must be a boolean")
        if self.operation.mutating:
            if not (
                self.supports_dry_run
                and self.requires_idempotency
                and self.requires_authorization
            ):
                raise ControlContractError(
                    "mutation capabilities must advertise dry-run, "
                    "idempotency, and authorization"
                )
        elif self.requires_idempotency:
            raise ControlContractError(
                "non-mutation capabilities must not require idempotency"
            )
        _bounded_record(self, "operation capability")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "operation": self.operation,
            "authority": self.authority,
            "bounds": self.bounds.to_record(),
            "supports_dry_run": self.supports_dry_run,
            "requires_idempotency": self.requires_idempotency,
            "requires_authorization": self.requires_authorization,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperationCapability":
        _schema(payload, cls.SCHEMA)
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "operation",
            "authority",
            "bounds",
            "supports_dry_run",
            "requires_idempotency",
            "requires_authorization",
            "content_id",
        }
        _reject_unknown(payload, allowed, "operation capability")
        result = cls(
            operation=payload.get("operation", ""),
            authority=payload.get("authority", ""),
            bounds=payload.get("bounds") or ControlBounds(),
            supports_dry_run=payload.get("supports_dry_run", False),
            requires_idempotency=payload.get("requires_idempotency", False),
            requires_authorization=payload.get(
                "requires_authorization", False
            ),
        )
        _identity(payload, result.content_id, "operation capability")
        return result


@dataclass(frozen=True)
class CapabilityReport(_ControlCanonicalContract):
    """Side-effect-free capability handshake for a control implementation."""

    SCHEMA: ClassVar[str] = CAPABILITY_REPORT_SCHEMA

    service_id: str
    service_version: str
    capabilities: tuple[OperationCapability, ...]
    contract_versions: tuple[int, ...] = (CONTROL_CONTRACT_VERSION,)
    optional_providers_loaded: bool = False
    processes_started: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "service_id", _text(self.service_id, "service_id"))
        object.__setattr__(
            self,
            "service_version",
            _text(self.service_version, "service_version"),
        )
        capabilities = _coerce_tuple(
            self.capabilities,
            OperationCapability,
            OperationCapability.from_dict,
            "capabilities",
        )
        operations = [item.operation for item in capabilities]
        if len(operations) != len(set(operations)):
            raise ControlContractError(
                "capability report contains duplicate operations"
            )
        object.__setattr__(
            self,
            "capabilities",
            tuple(sorted(capabilities, key=lambda item: item.operation.value)),
        )
        versions = tuple(
            sorted(
                {
                    _positive(item, "contract version")
                    for item in self.contract_versions
                }
            )
        )
        if not versions:
            raise ControlContractError("contract_versions must not be empty")
        object.__setattr__(self, "contract_versions", versions)
        for name in ("optional_providers_loaded", "processes_started"):
            if not isinstance(getattr(self, name), bool):
                raise ControlContractError(f"{name} must be a boolean")
        _bounded_record(self, "capability report")

    @property
    def supported_operations(self) -> tuple[Operation, ...]:
        return tuple(item.operation for item in self.capabilities)

    def supports(self, operation: Operation | str) -> bool:
        selected = _operation(operation)
        return selected in self.supported_operations

    def capability_for(
        self, operation: Operation | str
    ) -> OperationCapability | None:
        selected = _operation(operation)
        return next(
            (
                capability
                for capability in self.capabilities
                if capability.operation is selected
            ),
            None,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "service_id": self.service_id,
            "service_version": self.service_version,
            "capabilities": tuple(
                item.to_record() for item in self.capabilities
            ),
            "contract_versions": self.contract_versions,
            "optional_providers_loaded": self.optional_providers_loaded,
            "processes_started": self.processes_started,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilityReport":
        _schema(payload, cls.SCHEMA)
        allowed = {
            "schema",
            "schema_version",
            "contract_version",
            "service_id",
            "service_version",
            "capabilities",
            "contract_versions",
            "optional_providers_loaded",
            "processes_started",
            "content_id",
        }
        _reject_unknown(payload, allowed, "capability report")
        result = cls(
            service_id=payload.get("service_id", ""),
            service_version=payload.get("service_version", ""),
            capabilities=payload.get("capabilities", ()),
            contract_versions=payload.get(
                "contract_versions", (CONTROL_CONTRACT_VERSION,)
            ),
            optional_providers_loaded=payload.get(
                "optional_providers_loaded", False
            ),
            processes_started=payload.get("processes_started", False),
        )
        _identity(payload, result.content_id, "capability report")
        return result


@dataclass(frozen=True)
class ControlDiscoveryManifest(_ControlCanonicalContract):
    """Canonical contract exposed by one side-effect-free discovery surface.

    Request and result schema identities are derived from the authoritative
    schema producers.  A transport therefore cannot qualify by advertising a
    caller-supplied digest or a partial operation vocabulary.
    """

    SCHEMA: ClassVar[str] = CONTROL_DISCOVERY_MANIFEST_SCHEMA

    surface: ControlSurface
    operations: tuple[Operation, ...] = tuple(
        sorted(Operation, key=lambda item: item.value)
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "surface",
            _enum(self.surface, ControlSurface, "surface"),
        )
        operations = tuple(
            sorted(
                {_operation(item) for item in self.operations},
                key=lambda item: item.value,
            )
        )
        expected = tuple(sorted(Operation, key=lambda item: item.value))
        if operations != expected:
            raise ControlContractError(
                "discovery manifest must bind the complete operation vocabulary"
            )
        object.__setattr__(self, "operations", operations)
        _bounded_record(self, "control discovery manifest")

    @property
    def request_schema_ids(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                operation.value: content_identity(
                    operation_request_json_schema(operation)
                )
                for operation in self.operations
            }
        )

    @property
    def result_schema_ids(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                operation.value: content_identity(
                    operation_result_json_schema(operation)
                )
                for operation in self.operations
            }
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "surface": self.surface,
            "operations": self.operations,
            "request_schema_ids": dict(self.request_schema_ids),
            "result_schema_ids": dict(self.result_schema_ids),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ControlDiscoveryManifest":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "surface",
                "operations",
                "request_schema_ids",
                "result_schema_ids",
                "content_id",
            },
            "control discovery manifest",
        )
        result = cls(
            surface=payload.get("surface", ""),
            operations=payload.get("operations", ()),
        )
        for name, actual in (
            ("request_schema_ids", result.request_schema_ids),
            ("result_schema_ids", result.result_schema_ids),
        ):
            claimed = payload.get(name)
            if claimed not in (None, ""):
                if not isinstance(claimed, Mapping) or dict(claimed) != dict(
                    actual
                ):
                    raise ControlContractError(
                        f"discovery manifest {name} does not match shared schemas"
                    )
        _identity(payload, result.content_id, "control discovery manifest")
        return result


@dataclass(frozen=True)
class ControlDiscoveryRuntimeState(_ControlCanonicalContract):
    """Independently observed state surrounding a discovery-only action.

    The three counters are cumulative instrumentation readings.  Comparing
    before and after states catches even a provider load or short-lived child
    process which is absent by the time the final module/process inventory is
    sampled.
    """

    SCHEMA: ClassVar[str] = CONTROL_DISCOVERY_RUNTIME_STATE_SCHEMA

    optional_provider_modules: tuple[str, ...] = ()
    child_process_ids: tuple[int, ...] = ()
    service_resolution_count: int = 0
    optional_provider_load_count: int = 0
    process_start_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "optional_provider_modules",
            _strings(
                self.optional_provider_modules,
                "optional_provider_modules",
            ),
        )
        if isinstance(self.child_process_ids, (str, bytes, bytearray)):
            raise ControlContractError(
                "child_process_ids must be a sequence of integers"
            )
        try:
            process_ids = tuple(
                sorted(
                    {
                        _positive(item, "child process ID")
                        for item in self.child_process_ids
                    }
                )
            )
        except TypeError as exc:
            raise ControlContractError(
                "child_process_ids must be a sequence of integers"
            ) from exc
        if len(process_ids) > ABSOLUTE_MAX_CONTROL_ITEMS:
            raise ControlBoundsError(
                "child_process_ids exceeds its count bound"
            )
        object.__setattr__(self, "child_process_ids", process_ids)
        for name in (
            "service_resolution_count",
            "optional_provider_load_count",
            "process_start_count",
        ):
            object.__setattr__(
                self, name, _nonnegative(getattr(self, name), name)
            )
        _bounded_record(self, "control discovery runtime state")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "optional_provider_modules": self.optional_provider_modules,
            "child_process_ids": self.child_process_ids,
            "service_resolution_count": self.service_resolution_count,
            "optional_provider_load_count": self.optional_provider_load_count,
            "process_start_count": self.process_start_count,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ControlDiscoveryRuntimeState":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "optional_provider_modules",
                "child_process_ids",
                "service_resolution_count",
                "optional_provider_load_count",
                "process_start_count",
                "content_id",
            },
            "control discovery runtime state",
        )
        result = cls(
            optional_provider_modules=payload.get(
                "optional_provider_modules", ()
            ),
            child_process_ids=payload.get("child_process_ids", ()),
            service_resolution_count=payload.get(
                "service_resolution_count", 0
            ),
            optional_provider_load_count=payload.get(
                "optional_provider_load_count", 0
            ),
            process_start_count=payload.get("process_start_count", 0),
        )
        _identity(payload, result.content_id, "control discovery runtime state")
        return result


@dataclass(frozen=True)
class ControlDiscoveryObservation(_ControlCanonicalContract):
    """A repeated deterministic discovery observed between runtime snapshots."""

    SCHEMA: ClassVar[str] = CONTROL_DISCOVERY_OBSERVATION_SCHEMA

    surface: ControlSurface
    first_manifest: ControlDiscoveryManifest | Mapping[str, Any]
    second_manifest: ControlDiscoveryManifest | Mapping[str, Any]
    before: ControlDiscoveryRuntimeState | Mapping[str, Any]
    after: ControlDiscoveryRuntimeState | Mapping[str, Any]

    def __post_init__(self) -> None:
        surface = _enum(self.surface, ControlSurface, "surface")
        object.__setattr__(self, "surface", surface)
        for name, kind, decoder in (
            (
                "first_manifest",
                ControlDiscoveryManifest,
                ControlDiscoveryManifest.from_dict,
            ),
            (
                "second_manifest",
                ControlDiscoveryManifest,
                ControlDiscoveryManifest.from_dict,
            ),
            (
                "before",
                ControlDiscoveryRuntimeState,
                ControlDiscoveryRuntimeState.from_dict,
            ),
            (
                "after",
                ControlDiscoveryRuntimeState,
                ControlDiscoveryRuntimeState.from_dict,
            ),
        ):
            value = getattr(self, name)
            if not isinstance(value, kind):
                if not isinstance(value, Mapping):
                    raise ControlContractError(
                        f"{name} must be a {kind.__name__}"
                    )
                value = decoder(value)
            object.__setattr__(self, name, value)
        assert isinstance(self.first_manifest, ControlDiscoveryManifest)
        assert isinstance(self.second_manifest, ControlDiscoveryManifest)
        assert isinstance(self.before, ControlDiscoveryRuntimeState)
        assert isinstance(self.after, ControlDiscoveryRuntimeState)
        if (
            self.first_manifest.surface is not surface
            or self.second_manifest.surface is not surface
        ):
            raise ControlContractError(
                "discovery manifest surface does not match its observation"
            )
        if (
            self.first_manifest.to_record()
            != self.second_manifest.to_record()
        ):
            raise ControlContractError(
                "repeated control discovery is not deterministic"
            )
        for name in (
            "optional_provider_modules",
            "child_process_ids",
            "service_resolution_count",
            "optional_provider_load_count",
            "process_start_count",
        ):
            if getattr(self.before, name) != getattr(self.after, name):
                raise ControlContractError(
                    "control discovery changed observed runtime state: " + name
                )
        _bounded_record(self, "control discovery observation")

    @property
    def manifest(self) -> ControlDiscoveryManifest:
        assert isinstance(self.first_manifest, ControlDiscoveryManifest)
        return self.first_manifest

    @property
    def side_effect_free(self) -> bool:
        return True

    def _payload(self) -> dict[str, Any]:
        assert isinstance(self.first_manifest, ControlDiscoveryManifest)
        assert isinstance(self.second_manifest, ControlDiscoveryManifest)
        assert isinstance(self.before, ControlDiscoveryRuntimeState)
        assert isinstance(self.after, ControlDiscoveryRuntimeState)
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "surface": self.surface,
            "side_effect_free": True,
            "first_manifest": self.first_manifest.to_record(),
            "second_manifest": self.second_manifest.to_record(),
            "before": self.before.to_record(),
            "after": self.after.to_record(),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ControlDiscoveryObservation":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "surface",
                "side_effect_free",
                "first_manifest",
                "second_manifest",
                "before",
                "after",
                "content_id",
            },
            "control discovery observation",
        )
        if payload.get("side_effect_free") not in (None, True):
            raise ControlContractError(
                "discovery observation cannot claim a failed safety result"
            )
        result = cls(
            surface=payload.get("surface", ""),
            first_manifest=payload.get("first_manifest") or {},
            second_manifest=payload.get("second_manifest") or {},
            before=payload.get("before") or {},
            after=payload.get("after") or {},
        )
        _identity(payload, result.content_id, "control discovery observation")
        return result


@dataclass(frozen=True)
class ControlDiscoverySafetyEvidence(_ControlCanonicalContract):
    """Tamper-evident proof that discovery is deterministic and isolated."""

    SCHEMA: ClassVar[str] = CONTROL_DISCOVERY_SAFETY_EVIDENCE_SCHEMA

    repository_tree: str
    objective_id: str
    policy_id: str
    policy_revision: str
    capability_report: CapabilityReport | Mapping[str, Any]
    observations: tuple[ControlDiscoveryObservation, ...]
    requirement_id: str = CONTROL_DISCOVERY_SAFETY_REQUIREMENT_ID

    def __post_init__(self) -> None:
        for name in (
            "repository_tree",
            "objective_id",
            "policy_id",
            "policy_revision",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.requirement_id != CONTROL_DISCOVERY_SAFETY_REQUIREMENT_ID:
            raise ControlContractError(
                "discovery evidence requirement_id is not the ASI-G105 requirement"
            )
        report = self.capability_report
        if not isinstance(report, CapabilityReport):
            if not isinstance(report, Mapping):
                raise ControlContractError(
                    "capability_report must be a CapabilityReport"
                )
            report = CapabilityReport.from_dict(report)
        expected_operations = tuple(
            sorted(Operation, key=lambda item: item.value)
        )
        if report.supported_operations != expected_operations:
            raise ControlContractError(
                "discovery evidence requires complete capabilities"
            )
        if report.optional_providers_loaded or report.processes_started:
            raise ControlContractError(
                "capability discovery reports provider or process side effects"
            )
        object.__setattr__(self, "capability_report", report)
        observations = _coerce_tuple(
            self.observations,
            ControlDiscoveryObservation,
            ControlDiscoveryObservation.from_dict,
            "observations",
        )
        surfaces = tuple(
            sorted(
                (item.surface for item in observations),
                key=lambda item: item.value,
            )
        )
        expected_surfaces = tuple(
            sorted(ControlSurface, key=lambda item: item.value)
        )
        if surfaces != expected_surfaces:
            raise ControlContractError(
                "discovery evidence requires one Python, CLI, and MCP observation"
            )
        object.__setattr__(
            self,
            "observations",
            tuple(sorted(observations, key=lambda item: item.surface.value)),
        )
        _bounded_record(
            self,
            "control discovery safety evidence",
            maximum=ABSOLUTE_MAX_CONTROL_BYTES,
        )

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (CONTROL_DISCOVERY_SAFETY_REQUIREMENT_ID,)

    def _payload(self) -> dict[str, Any]:
        assert isinstance(self.capability_report, CapabilityReport)
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "requirement_id": self.requirement_id,
            "repository_tree": self.repository_tree,
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "capability_report": self.capability_report.to_record(),
            "observations": tuple(
                item.to_record() for item in self.observations
            ),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ControlDiscoverySafetyEvidence":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "requirement_id",
                "repository_tree",
                "objective_id",
                "policy_id",
                "policy_revision",
                "capability_report",
                "observations",
                "content_id",
            },
            "control discovery safety evidence",
        )
        result = cls(
            requirement_id=payload.get("requirement_id", ""),
            repository_tree=payload.get("repository_tree", ""),
            objective_id=payload.get("objective_id", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            capability_report=payload.get("capability_report") or {},
            observations=payload.get("observations", ()),
        )
        _identity(payload, result.content_id, "control discovery safety evidence")
        return result


# Alternate terminology retained for integrations which use "isolation".
ControlDiscoveryIsolationEvidence = ControlDiscoverySafetyEvidence


@dataclass(frozen=True)
class ControlSurfaceParityCase(_ControlCanonicalContract):
    """One independently invoked Python/CLI/MCP behavior comparison.

    Full canonical records are retained rather than caller-supplied booleans
    or digests.  Construction re-decodes every record, validates each result
    against the request, and requires byte-for-byte equality across surfaces.
    """

    SCHEMA: ClassVar[str] = CONTROL_SURFACE_PARITY_CASE_SCHEMA

    scenario: str
    request: OperationRequest | Mapping[str, Any]
    python_result: OperationResult | Mapping[str, Any]
    cli_result: OperationResult | Mapping[str, Any]
    mcp_result: OperationResult | Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scenario", _text(self.scenario, "scenario", max_bytes=128)
        )
        request = self.request
        if not isinstance(request, OperationRequest):
            if not isinstance(request, Mapping):
                raise ControlContractError(
                    "parity case request must be an OperationRequest"
                )
            request = OperationRequest.from_dict(request)
        object.__setattr__(self, "request", request)

        decoded: list[OperationResult] = []
        for field_name in ("python_result", "cli_result", "mcp_result"):
            result = getattr(self, field_name)
            if not isinstance(result, OperationResult):
                if not isinstance(result, Mapping):
                    raise ControlContractError(
                        f"{field_name} must be an OperationResult"
                    )
                result = OperationResult.from_dict(result)
            result.validate_against(request)
            object.__setattr__(self, field_name, result)
            decoded.append(result)
        records = [item.to_record() for item in decoded]
        if records[1:] != records[:-1]:
            raise ControlContractError(
                "Python, CLI, and MCP results are not canonically identical"
            )
        _bounded_record(
            self,
            "control surface parity case",
            maximum=ABSOLUTE_MAX_CONTROL_BYTES,
        )

    @property
    def operation(self) -> Operation:
        assert isinstance(self.request, OperationRequest)
        return self.request.operation

    @property
    def status(self) -> OperationStatus:
        assert isinstance(self.python_result, OperationResult)
        return self.python_result.status

    @property
    def result_id(self) -> str:
        assert isinstance(self.python_result, OperationResult)
        return self.python_result.result_id

    @property
    def behavior_class(self) -> ControlBehaviorClass:
        """Derive, rather than trust, the qualifying behavior class."""

        assert isinstance(self.request, OperationRequest)
        assert isinstance(self.python_result, OperationResult)
        if not self.python_result.succeeded:
            if self.python_result.error is None:
                raise ControlContractError(
                    "failed parity result must carry a stable typed error"
                )
            return ControlBehaviorClass.STABLE_FAILURE
        if self.request.operation.mutating and not self.request.dry_run:
            if (
                not self.python_result.audit_receipt_id
                or not any(
                    effect.applied for effect in self.python_result.effects
                )
            ):
                raise ControlContractError(
                    "successful mutation parity case must be audited and applied"
                )
            return ControlBehaviorClass.MUTATION_SUCCESS
        if self.request.effective_authority is OperationAuthority.PROPOSAL:
            if self.python_result.preview is None:
                raise ControlContractError(
                    "successful proposal parity case must carry a preview"
                )
            return ControlBehaviorClass.PROPOSAL_SUCCESS
        return ControlBehaviorClass.READ_SUCCESS

    def _payload(self) -> dict[str, Any]:
        assert isinstance(self.request, OperationRequest)
        assert isinstance(self.python_result, OperationResult)
        assert isinstance(self.cli_result, OperationResult)
        assert isinstance(self.mcp_result, OperationResult)
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "scenario": self.scenario,
            "behavior_class": self.behavior_class,
            "operation": self.operation,
            "status": self.status,
            "request": self.request.to_record(),
            "python_result": self.python_result.to_record(),
            "cli_result": self.cli_result.to_record(),
            "mcp_result": self.mcp_result.to_record(),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ControlSurfaceParityCase":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "scenario",
                "behavior_class",
                "operation",
                "status",
                "request",
                "python_result",
                "cli_result",
                "mcp_result",
                "content_id",
            },
            "control surface parity case",
        )
        result = cls(
            scenario=payload.get("scenario", ""),
            request=payload.get("request") or {},
            python_result=payload.get("python_result") or {},
            cli_result=payload.get("cli_result") or {},
            mcp_result=payload.get("mcp_result") or {},
        )
        claimed_operation = payload.get("operation")
        if claimed_operation not in (None, "") and _operation(
            claimed_operation
        ) is not result.operation:
            raise ControlContractError(
                "parity case operation does not match its request"
            )
        claimed_behavior = payload.get("behavior_class")
        if claimed_behavior not in (None, "") and _enum(
            claimed_behavior,
            ControlBehaviorClass,
            "behavior_class",
        ) is not result.behavior_class:
            raise ControlContractError(
                "parity case behavior_class does not match its result"
            )
        claimed_status = payload.get("status")
        if claimed_status not in (None, "") and _enum(
            claimed_status, OperationStatus, "status"
        ) is not result.status:
            raise ControlContractError(
                "parity case status does not match its result"
            )
        _identity(payload, result.content_id, "control surface parity case")
        return result


@dataclass(frozen=True)
class ControlSurfaceParityEvidence(_ControlCanonicalContract):
    """Tamper-evident proof that all public control surfaces share one contract."""

    SCHEMA: ClassVar[str] = CONTROL_SURFACE_PARITY_EVIDENCE_SCHEMA

    repository_tree: str
    objective_id: str
    policy_id: str
    policy_revision: str
    capability_report: CapabilityReport | Mapping[str, Any]
    cases: tuple[ControlSurfaceParityCase, ...]
    operations: tuple[Operation, ...] = tuple(
        sorted(Operation, key=lambda item: item.value)
    )
    surfaces: tuple[ControlSurface, ...] = tuple(ControlSurface)
    requirement_id: str = CONTROL_SURFACE_PARITY_REQUIREMENT_ID

    def __post_init__(self) -> None:
        for name in (
            "repository_tree",
            "objective_id",
            "policy_id",
            "policy_revision",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.requirement_id != CONTROL_SURFACE_PARITY_REQUIREMENT_ID:
            raise ControlContractError(
                "parity evidence requirement_id is not the ASI-G103 requirement"
            )
        report = self.capability_report
        if not isinstance(report, CapabilityReport):
            if not isinstance(report, Mapping):
                raise ControlContractError(
                    "capability_report must be a CapabilityReport"
                )
            report = CapabilityReport.from_dict(report)
        object.__setattr__(self, "capability_report", report)

        cases = _coerce_tuple(
            self.cases,
            ControlSurfaceParityCase,
            ControlSurfaceParityCase.from_dict,
            "cases",
        )
        if not cases:
            raise ControlContractError(
                "parity evidence requires independently invoked cases"
            )
        scenarios = [item.scenario for item in cases]
        if len(scenarios) != len(set(scenarios)):
            raise ControlContractError("parity case scenarios must be unique")
        behavior_classes = {item.behavior_class for item in cases}
        required_behavior_classes = set(ControlBehaviorClass)
        if behavior_classes != required_behavior_classes:
            missing = sorted(
                item.value
                for item in required_behavior_classes - behavior_classes
            )
            raise ControlContractError(
                "parity evidence requires the complete behavior matrix"
                + (": missing " + ", ".join(missing) if missing else "")
            )
        for item in cases:
            assert isinstance(item.request, OperationRequest)
            if (
                item.request.tree_id != self.repository_tree
                or item.request.objective_id != self.objective_id
                or item.request.policy_id != self.policy_id
                or item.request.policy_revision != self.policy_revision
            ):
                raise ControlContractError(
                    "parity case is stale or bound to another objective/policy"
                )
        object.__setattr__(self, "cases", tuple(cases))

        operations = tuple(
            sorted(
                {_operation(item) for item in self.operations},
                key=lambda item: item.value,
            )
        )
        expected_operations = tuple(
            sorted(Operation, key=lambda item: item.value)
        )
        if operations != expected_operations:
            raise ControlContractError(
                "parity evidence must bind the complete operation vocabulary"
            )
        if report.supported_operations != expected_operations:
            raise ControlContractError(
                "capability report must support the complete operation vocabulary"
            )
        object.__setattr__(self, "operations", operations)

        surfaces = tuple(
            sorted(
                {_enum(item, ControlSurface, "surface") for item in self.surfaces},
                key=lambda item: item.value,
            )
        )
        expected_surfaces = tuple(
            sorted(ControlSurface, key=lambda item: item.value)
        )
        if surfaces != expected_surfaces:
            raise ControlContractError(
                "parity evidence requires Python, CLI, and MCP observations"
            )
        object.__setattr__(self, "surfaces", surfaces)
        _bounded_record(
            self,
            "control surface parity evidence",
            maximum=ABSOLUTE_MAX_CONTROL_BYTES,
        )

    @property
    def request_schema_id(self) -> str:
        return content_identity(operation_request_json_schema())

    @property
    def result_schema_id(self) -> str:
        return content_identity(operation_result_json_schema())

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (CONTROL_SURFACE_PARITY_REQUIREMENT_ID,)

    def _payload(self) -> dict[str, Any]:
        assert isinstance(self.capability_report, CapabilityReport)
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "requirement_id": self.requirement_id,
            "repository_tree": self.repository_tree,
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "surfaces": self.surfaces,
            "operations": self.operations,
            "request_schema_id": self.request_schema_id,
            "result_schema_id": self.result_schema_id,
            "capability_report": self.capability_report.to_record(),
            "cases": tuple(item.to_record() for item in self.cases),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ControlSurfaceParityEvidence":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "requirement_id",
                "repository_tree",
                "objective_id",
                "policy_id",
                "policy_revision",
                "surfaces",
                "operations",
                "request_schema_id",
                "result_schema_id",
                "capability_report",
                "cases",
                "content_id",
            },
            "control surface parity evidence",
        )
        result = cls(
            requirement_id=payload.get("requirement_id", ""),
            repository_tree=payload.get("repository_tree", ""),
            objective_id=payload.get("objective_id", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            surfaces=payload.get("surfaces", ()),
            operations=payload.get("operations", ()),
            capability_report=payload.get("capability_report") or {},
            cases=payload.get("cases", ()),
        )
        for name, actual in (
            ("request_schema_id", result.request_schema_id),
            ("result_schema_id", result.result_schema_id),
        ):
            claimed = payload.get(name)
            if claimed not in (None, "") and claimed != actual:
                raise ControlContractError(
                    f"parity evidence {name} does not match the shared schema"
                )
        _identity(payload, result.content_id, "control surface parity evidence")
        return result


@dataclass(frozen=True)
class MutationGuardRejection(_ControlCanonicalContract):
    """A malformed mutation replayed through the authoritative request parser."""

    SCHEMA: ClassVar[str] = MUTATION_GUARD_REJECTION_SCHEMA

    scenario: str
    request_payload: Mapping[str, Any]
    error_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario", _text(self.scenario, "scenario"))
        if not isinstance(self.request_payload, Mapping):
            raise ControlContractError("request_payload must be a mapping")
        payload = dict(self.request_payload)
        payload.pop("content_id", None)
        object.__setattr__(
            self,
            "request_payload",
            _freeze_value(
                payload,
                name="rejected request payload",
                max_depth=ABSOLUTE_MAX_CONTROL_DEPTH,
                max_items=ABSOLUTE_MAX_CONTROL_ITEMS,
                max_text_bytes=ABSOLUTE_MAX_CONTROL_TEXT_BYTES,
            ),
        )
        error_type = _text(self.error_type, "error_type")
        try:
            OperationRequest.from_dict(payload)
        except (ControlContractError, ValueError) as exc:
            actual = type(exc).__name__
        else:
            raise ControlContractError(
                "mutation guard rejection payload was accepted"
            )
        if actual != error_type:
            raise ControlContractError(
                "mutation guard rejection error_type does not match replay"
            )
        object.__setattr__(self, "error_type", error_type)
        _bounded_record(self, "mutation guard rejection")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "scenario": self.scenario,
            "request_payload": self.request_payload,
            "error_type": self.error_type,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MutationGuardRejection":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "scenario",
                "request_payload",
                "error_type",
                "content_id",
            },
            "mutation guard rejection",
        )
        result = cls(
            scenario=payload.get("scenario", ""),
            request_payload=payload.get("request_payload") or {},
            error_type=payload.get("error_type", ""),
        )
        _identity(payload, result.content_id, "mutation guard rejection")
        return result


@dataclass(frozen=True)
class ControlMutationRuntimeState(_ControlCanonicalContract):
    """Observed mutation dispatch and durable-audit population at one instant."""

    SCHEMA: ClassVar[str] = CONTROL_MUTATION_RUNTIME_STATE_SCHEMA

    dispatch_count: int = 0
    audit_receipt_count: int = 0
    last_dispatch_request_id: str = ""
    last_audit_receipt_id: str = ""

    def __post_init__(self) -> None:
        for name in ("dispatch_count", "audit_receipt_count"):
            object.__setattr__(
                self, name, _nonnegative(getattr(self, name), name)
            )
        for name in (
            "last_dispatch_request_id",
            "last_audit_receipt_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, required=False),
            )
        if bool(self.dispatch_count) != bool(self.last_dispatch_request_id):
            raise ControlContractError(
                "dispatch count and latest request identity must both be set"
            )
        if bool(self.audit_receipt_count) != bool(
            self.last_audit_receipt_id
        ):
            raise ControlContractError(
                "audit count and latest receipt identity must both be set"
            )
        _bounded_record(self, "control mutation runtime state")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "dispatch_count": self.dispatch_count,
            "audit_receipt_count": self.audit_receipt_count,
            "last_dispatch_request_id": self.last_dispatch_request_id,
            "last_audit_receipt_id": self.last_audit_receipt_id,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ControlMutationRuntimeState":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "dispatch_count",
                "audit_receipt_count",
                "last_dispatch_request_id",
                "last_audit_receipt_id",
                "content_id",
            },
            "control mutation runtime state",
        )
        result = cls(
            dispatch_count=payload.get("dispatch_count", 0),
            audit_receipt_count=payload.get("audit_receipt_count", 0),
            last_dispatch_request_id=payload.get(
                "last_dispatch_request_id", ""
            ),
            last_audit_receipt_id=payload.get(
                "last_audit_receipt_id", ""
            ),
        )
        _identity(payload, result.content_id, "control mutation runtime state")
        return result


@dataclass(frozen=True)
class MutationGuardExecutionObservation(_ControlCanonicalContract):
    """Three runtime snapshots proving one dispatch and a dispatch-free replay."""

    SCHEMA: ClassVar[str] = MUTATION_GUARD_EXECUTION_OBSERVATION_SCHEMA

    request_id: str
    result_id: str
    audit_receipt_id: str
    before: ControlMutationRuntimeState | Mapping[str, Any]
    after_result: ControlMutationRuntimeState | Mapping[str, Any]
    after_replay: ControlMutationRuntimeState | Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("request_id", "result_id", "audit_receipt_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        states: dict[str, ControlMutationRuntimeState] = {}
        for name in ("before", "after_result", "after_replay"):
            item = getattr(self, name)
            if not isinstance(item, ControlMutationRuntimeState):
                if not isinstance(item, Mapping):
                    raise ControlContractError(
                        f"{name} must be a ControlMutationRuntimeState"
                    )
                item = ControlMutationRuntimeState.from_dict(item)
            object.__setattr__(self, name, item)
            states[name] = item

        before = states["before"]
        after_result = states["after_result"]
        after_replay = states["after_replay"]
        if (
            after_result.dispatch_count != before.dispatch_count + 1
            or after_result.last_dispatch_request_id != self.request_id
        ):
            raise ControlContractError(
                "mutation execution must add exactly one bound backend dispatch"
            )
        if (
            after_replay.dispatch_count != after_result.dispatch_count
            or after_replay.last_dispatch_request_id
            != after_result.last_dispatch_request_id
        ):
            raise ControlContractError(
                "idempotent replay must not dispatch the backend"
            )
        if (
            after_result.audit_receipt_count
            != before.audit_receipt_count + 1
            or after_result.last_audit_receipt_id
            != self.audit_receipt_id
        ):
            raise ControlContractError(
                "mutation execution must add exactly one bound audit receipt"
            )
        if (
            after_replay.audit_receipt_count
            != after_result.audit_receipt_count
            or after_replay.last_audit_receipt_id
            != after_result.last_audit_receipt_id
        ):
            raise ControlContractError(
                "idempotent replay must not append another audit receipt"
            )
        _bounded_record(self, "mutation guard execution observation")

    def _payload(self) -> dict[str, Any]:
        assert isinstance(self.before, ControlMutationRuntimeState)
        assert isinstance(self.after_result, ControlMutationRuntimeState)
        assert isinstance(self.after_replay, ControlMutationRuntimeState)
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "request_id": self.request_id,
            "result_id": self.result_id,
            "audit_receipt_id": self.audit_receipt_id,
            "before": self.before.to_record(),
            "after_result": self.after_result.to_record(),
            "after_replay": self.after_replay.to_record(),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "MutationGuardExecutionObservation":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "request_id",
                "result_id",
                "audit_receipt_id",
                "before",
                "after_result",
                "after_replay",
                "content_id",
            },
            "mutation guard execution observation",
        )
        result = cls(
            request_id=payload.get("request_id", ""),
            result_id=payload.get("result_id", ""),
            audit_receipt_id=payload.get("audit_receipt_id", ""),
            before=payload.get("before") or {},
            after_result=payload.get("after_result") or {},
            after_replay=payload.get("after_replay") or {},
        )
        _identity(
            payload, result.content_id, "mutation guard execution observation"
        )
        return result


@dataclass(frozen=True)
class ControlMutationGuardEvidence(_ControlCanonicalContract):
    """Evidence that an applied mutation was guarded, audited, and replay-safe."""

    SCHEMA: ClassVar[str] = CONTROL_MUTATION_GUARD_EVIDENCE_SCHEMA

    repository_tree: str
    objective_id: str
    policy_id: str
    policy_revision: str
    request: OperationRequest | Mapping[str, Any]
    result: OperationResult | Mapping[str, Any]
    replay_result: OperationResult | Mapping[str, Any]
    execution: MutationGuardExecutionObservation | Mapping[str, Any]
    rejections: tuple[MutationGuardRejection, ...]
    requirement_id: str = CONTROL_MUTATION_GUARD_REQUIREMENT_ID

    def __post_init__(self) -> None:
        for name in (
            "repository_tree",
            "objective_id",
            "policy_id",
            "policy_revision",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.requirement_id != CONTROL_MUTATION_GUARD_REQUIREMENT_ID:
            raise ControlContractError(
                "mutation evidence requirement_id is not the ASI-G104 requirement"
            )
        request = self.request
        if not isinstance(request, OperationRequest):
            if not isinstance(request, Mapping):
                raise ControlContractError("request must be an OperationRequest")
            request = OperationRequest.from_dict(request)
        if not request.operation.mutating or request.dry_run:
            raise ControlContractError(
                "mutation guard evidence requires a real mutation request"
            )
        if (
            request.tree_id != self.repository_tree
            or request.objective_id != self.objective_id
            or request.policy_id != self.policy_id
            or request.policy_revision != self.policy_revision
        ):
            raise ControlContractError(
                "mutation guard evidence request binding is stale"
            )
        object.__setattr__(self, "request", request)

        decoded_results: list[OperationResult] = []
        for name in ("result", "replay_result"):
            item = getattr(self, name)
            if not isinstance(item, OperationResult):
                if not isinstance(item, Mapping):
                    raise ControlContractError(
                        f"{name} must be an OperationResult"
                    )
                item = OperationResult.from_dict(item)
            item.validate_against(request)
            if not item.succeeded:
                raise ControlContractError(
                    "mutation guard evidence requires successful results"
                )
            object.__setattr__(self, name, item)
            decoded_results.append(item)
        if (
            decoded_results[0].to_record() != decoded_results[1].to_record()
            or not decoded_results[0].audit_receipt_id
            or not any(effect.applied for effect in decoded_results[0].effects)
        ):
            raise ControlContractError(
                "mutation result is not audited, applied, and exactly replayed"
            )

        execution = self.execution
        if not isinstance(execution, MutationGuardExecutionObservation):
            if not isinstance(execution, Mapping):
                raise ControlContractError(
                    "execution must be a MutationGuardExecutionObservation"
                )
            execution = MutationGuardExecutionObservation.from_dict(execution)
        if (
            execution.request_id != request.request_id
            or execution.result_id != decoded_results[0].result_id
            or execution.audit_receipt_id
            != decoded_results[0].audit_receipt_id
        ):
            raise ControlContractError(
                "mutation execution observation is detached from its request "
                "or result"
            )
        object.__setattr__(self, "execution", execution)

        rejections = _coerce_tuple(
            self.rejections,
            MutationGuardRejection,
            MutationGuardRejection.from_dict,
            "rejections",
        )
        scenarios = {item.scenario for item in rejections}
        required = {
            "missing_authorization",
            "missing_idempotency",
            "missing_lease_or_fence",
        }
        if scenarios != required or len(rejections) != len(required):
            raise ControlContractError(
                "mutation evidence requires authorization, idempotency, and "
                "lease/fence rejection replays"
            )
        object.__setattr__(
            self,
            "rejections",
            tuple(sorted(rejections, key=lambda item: item.scenario)),
        )
        _bounded_record(
            self,
            "control mutation guard evidence",
            maximum=ABSOLUTE_MAX_CONTROL_BYTES,
        )

    @property
    def proved_requirement_ids(self) -> tuple[str, ...]:
        return (CONTROL_MUTATION_GUARD_REQUIREMENT_ID,)

    def _payload(self) -> dict[str, Any]:
        assert isinstance(self.request, OperationRequest)
        assert isinstance(self.result, OperationResult)
        assert isinstance(self.replay_result, OperationResult)
        assert isinstance(self.execution, MutationGuardExecutionObservation)
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "requirement_id": self.requirement_id,
            "repository_tree": self.repository_tree,
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "request": self.request.to_record(),
            "result": self.result.to_record(),
            "replay_result": self.replay_result.to_record(),
            "execution": self.execution.to_record(),
            "rejections": tuple(item.to_record() for item in self.rejections),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ControlMutationGuardEvidence":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "requirement_id",
                "repository_tree",
                "objective_id",
                "policy_id",
                "policy_revision",
                "request",
                "result",
                "replay_result",
                "execution",
                "rejections",
                "content_id",
            },
            "control mutation guard evidence",
        )
        result = cls(
            requirement_id=payload.get("requirement_id", ""),
            repository_tree=payload.get("repository_tree", ""),
            objective_id=payload.get("objective_id", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            request=payload.get("request") or {},
            result=payload.get("result") or {},
            replay_result=payload.get("replay_result") or {},
            execution=payload.get("execution") or {},
            rejections=payload.get("rejections", ()),
        )
        _identity(payload, result.content_id, "control mutation guard evidence")
        return result


@dataclass(frozen=True)
class LifecycleCommand(_ControlCanonicalContract):
    """A typed lifecycle intent, suitable for conversion to a request."""

    SCHEMA: ClassVar[str] = LIFECYCLE_COMMAND_SCHEMA

    action: LifecycleAction
    target_id: str
    reason: str
    requested_state: str = ""
    dry_run: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "action", _enum(self.action, LifecycleAction, "action")
        )
        object.__setattr__(self, "target_id", _text(self.target_id, "target_id"))
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        object.__setattr__(
            self,
            "requested_state",
            _text(self.requested_state, "requested_state", required=False),
        )
        if not isinstance(self.dry_run, bool):
            raise ControlContractError("dry_run must be a boolean")
        _bounded_record(self, "lifecycle command")

    @property
    def operation(self) -> Operation:
        return self.action.operation

    @property
    def authority(self) -> OperationAuthority:
        return (
            OperationAuthority.PROPOSAL
            if self.dry_run
            else OperationAuthority.MUTATION
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTROL_CONTRACT_VERSION,
            "action": self.action,
            "operation": self.operation,
            "authority": self.authority,
            "target_id": self.target_id,
            "reason": self.reason,
            "requested_state": self.requested_state,
            "dry_run": self.dry_run,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LifecycleCommand":
        _schema(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            {
                "schema",
                "schema_version",
                "contract_version",
                "action",
                "operation",
                "authority",
                "target_id",
                "reason",
                "requested_state",
                "dry_run",
                "content_id",
            },
            "lifecycle command",
        )
        result = cls(
            action=payload.get("action", payload.get("operation", "")),
            target_id=payload.get("target_id", ""),
            reason=payload.get("reason", ""),
            requested_state=payload.get("requested_state", ""),
            dry_run=payload.get("dry_run", False),
        )
        claimed_operation = payload.get("operation")
        if claimed_operation not in (None, "") and _operation(
            claimed_operation
        ) is not result.operation:
            raise ControlContractError(
                "lifecycle operation does not match its action"
            )
        claimed_authority = payload.get("authority")
        if claimed_authority not in (None, "") and _authority(
            claimed_authority
        ) is not result.authority:
            raise AuthorityViolationError(
                "lifecycle authority does not match dry_run"
            )
        _identity(payload, result.content_id, "lifecycle command")
        return result


def operation_request_json_schema(
    operation: Operation | str | None = None,
) -> dict[str, Any]:
    """Return the shared JSON Schema advertised by CLI and MCP adapters.

    JSON Schema is an early validation aid, not an authorization decision.
    :class:`OperationRequest` remains the authoritative parser and performs
    cross-field identity, effect, idempotency, authorization, and lease checks.
    """

    selected = _operation(operation) if operation is not None else None
    operation_values = (
        [selected.value]
        if selected is not None
        else [item.value for item in sorted(Operation, key=lambda item: item.value)]
    )
    string_id = {"type": "string", "minLength": 1, "maxLength": 65536}
    root = {
        "type": "string",
        "minLength": 1,
        "maxLength": 65536,
        "pattern": "^/",
    }
    properties: dict[str, Any] = {
        "schema": {"const": OPERATION_REQUEST_SCHEMA},
        "schema_version": {"type": "integer", "const": CONTROL_CONTRACT_VERSION},
        "contract_version": {
            "type": "integer",
            "const": CONTROL_CONTRACT_VERSION,
        },
        "operation": (
            {"const": selected.value}
            if selected is not None
            else {"type": "string", "enum": operation_values}
        ),
        "authority": {
            "type": "string",
            "enum": [item.value for item in OperationAuthority],
        },
        "effective_authority": {
            "type": "string",
            "enum": [item.value for item in OperationAuthority],
        },
        "repository_root": root,
        "state_root": root,
        "repository_id": string_id,
        "tree_id": string_id,
        "objective_id": string_id,
        "objective_revision": string_id,
        "policy_id": string_id,
        "policy_revision": string_id,
        "caller": string_id,
        "bounds": {"type": "object"},
        "expected_effects": {"type": "array", "maxItems": 64},
        "parameters": {"type": "object"},
        "dry_run": {"type": "boolean"},
        "idempotency": {"type": ["object", "null"]},
        "idempotency_key": {"type": "string"},
        "authorization": {"type": ["object", "null"]},
        "lease_id": {"type": "string"},
        "fencing_epoch": {"type": ["integer", "null"], "minimum": 0},
        "content_id": {"type": "string"},
    }
    required = [
        "operation",
        "repository_root",
        "state_root",
        "repository_id",
        "tree_id",
        "objective_id",
        "objective_revision",
        "policy_id",
        "policy_revision",
        "caller",
    ]
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": (
            f"{OPERATION_REQUEST_SCHEMA}#{selected.value}"
            if selected is not None
            else OPERATION_REQUEST_SCHEMA
        ),
        "title": "Agent supervisor operation request",
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    if selected is not None and selected.mutating:
        schema["allOf"] = [
            {
                "if": {
                    "properties": {"dry_run": {"const": False}},
                },
                "then": {
                    "required": [
                        "expected_effects",
                        "idempotency",
                        "authorization",
                        "lease_id",
                        "fencing_epoch",
                    ]
                },
            }
        ]
    return schema


def operation_result_json_schema(
    operation: Operation | str | None = None,
) -> dict[str, Any]:
    """Return the stable result-envelope JSON Schema for every surface."""

    selected = _operation(operation) if operation is not None else None
    operation_property: dict[str, Any] = (
        {"const": selected.value}
        if selected is not None
        else {
            "type": "string",
            "enum": [
                item.value
                for item in sorted(Operation, key=lambda item: item.value)
            ],
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": (
            f"{OPERATION_RESULT_SCHEMA}#{selected.value}"
            if selected is not None
            else OPERATION_RESULT_SCHEMA
        ),
        "title": "Agent supervisor operation result",
        "type": "object",
        "properties": {
            "schema": {"const": OPERATION_RESULT_SCHEMA},
            "schema_version": {
                "type": "integer",
                "const": CONTROL_CONTRACT_VERSION,
            },
            "contract_version": {
                "type": "integer",
                "const": CONTROL_CONTRACT_VERSION,
            },
            "request_id": {"type": "string", "minLength": 1},
            "operation": operation_property,
            "authority": {
                "type": "string",
                "enum": [item.value for item in OperationAuthority],
            },
            "status": {
                "type": "string",
                "enum": [item.value for item in OperationStatus],
            },
            "repository_id": {"type": "string", "minLength": 1},
            "tree_id": {"type": "string", "minLength": 1},
            "objective_id": {"type": "string", "minLength": 1},
            "policy_id": {"type": "string", "minLength": 1},
            "caller": {"type": "string", "minLength": 1},
            "bounds": {"type": "object"},
            "data": {"type": "object"},
            "effects": {"type": "array"},
            "error": {"type": ["object", "null"]},
            "preview": {"type": ["object", "null"]},
            "idempotency_key": {"type": "string"},
            "audit_receipt_id": {"type": "string"},
            "content_id": {"type": "string", "minLength": 1},
        },
        "required": [
            "schema",
            "contract_version",
            "request_id",
            "operation",
            "authority",
            "status",
            "repository_id",
            "tree_id",
            "objective_id",
            "policy_id",
            "caller",
            "bounds",
            "data",
            "effects",
            "error",
            "preview",
            "idempotency_key",
            "audit_receipt_id",
            "content_id",
        ],
        "additionalProperties": False,
    }


def canonical_control_json_bytes(value: Any) -> bytes:
    """Return canonical DAG-JSON bytes for a control value."""

    return canonical_json_bytes(value)


def operation_authority(operation: Operation | str) -> OperationAuthority:
    """Return the registry authority for an operation, rejecting unknown IDs."""

    return _operation(operation).authority


ControlLimits = ControlBounds
RequestBounds = ControlBounds
OperationName = Operation
ControlOperation = Operation
AuthorityLevel = OperationAuthority
ControlError = OperationError
TypedOperationError = OperationError
AuthorizationResult = AuthorizationDecision
IdempotencyContract = IdempotencyKey
OperationEffect = ExpectedEffect
OperationEffectClaim = EffectClaim
ControlContractValidationError = ControlContractError


__all__ = [
    "AUTHORIZATION_DECISION_SCHEMA",
    "CAPABILITY_REPORT_SCHEMA",
    "CONTRACT_VERSION",
    "CONTROL_BOUNDS_SCHEMA",
    "CONTROL_CONTRACT_VERSION",
    "CONTROL_DISCOVERY_ISOLATION_REQUIREMENT_ID",
    "CONTROL_DISCOVERY_MANIFEST_SCHEMA",
    "CONTROL_DISCOVERY_OBSERVATION_SCHEMA",
    "CONTROL_DISCOVERY_RUNTIME_STATE_SCHEMA",
    "CONTROL_DISCOVERY_SAFETY_EVIDENCE_SCHEMA",
    "CONTROL_DISCOVERY_SAFETY_REQUIREMENT_ID",
    "CONTROL_MUTATION_GUARD_EVIDENCE_SCHEMA",
    "CONTROL_MUTATION_GUARD_REQUIREMENT_ID",
    "CONTROL_MUTATION_RUNTIME_STATE_SCHEMA",
    "CONTROL_SURFACE_PARITY_CASE_SCHEMA",
    "CONTROL_SURFACE_PARITY_EVIDENCE_SCHEMA",
    "CONTROL_SURFACE_PARITY_REQUIREMENT_ID",
    "DRY_RUN_PREVIEW_SCHEMA",
    "EFFECT_CLAIM_SCHEMA",
    "EXPECTED_EFFECT_SCHEMA",
    "IDEMPOTENCY_KEY_SCHEMA",
    "LIFECYCLE_COMMAND_SCHEMA",
    "OPERATION_CAPABILITY_SCHEMA",
    "OPERATION_ERROR_SCHEMA",
    "OPERATION_AUTHORITIES",
    "OPERATION_REQUEST_SCHEMA",
    "OPERATION_RESULT_SCHEMA",
    "SCHEMA_VERSION",
    "PROPOSAL_OPERATIONS",
    "READ_OPERATIONS",
    "MUTATION_OPERATIONS",
    "Authority",
    "AuthorityLevel",
    "AuthorityViolationError",
    "AuthorizationBindingError",
    "AuthorizationDecision",
    "AuthorizationResult",
    "AuthorizationVerdict",
    "CapabilityReport",
    "ControlBehaviorClass",
    "ControlBounds",
    "ControlBoundsError",
    "ControlContractError",
    "ControlContractValidationError",
    "ControlError",
    "ControlDiscoveryIsolationEvidence",
    "ControlDiscoveryManifest",
    "ControlDiscoveryObservation",
    "ControlDiscoveryRuntimeState",
    "ControlDiscoverySafetyEvidence",
    "ControlLimits",
    "ControlMutationGuardEvidence",
    "ControlMutationRuntimeState",
    "ControlOperation",
    "ControlSurface",
    "ControlSurfaceParityCase",
    "ControlSurfaceParityEvidence",
    "DryRunPreview",
    "EffectClaim",
    "EffectKind",
    "ErrorCode",
    "ExpectedEffect",
    "IdempotencyContract",
    "IdempotencyKey",
    "LifecycleAction",
    "LifecycleCommand",
    "LifecycleOperation",
    "MissingIdempotencyError",
    "MUTATION_GUARD_REJECTION_SCHEMA",
    "MUTATION_GUARD_EXECUTION_OBSERVATION_SCHEMA",
    "MutationGuardExecutionObservation",
    "MutationGuardRejection",
    "Operation",
    "OperationAuthority",
    "OperationCapability",
    "OperationError",
    "OperationEffect",
    "OperationEffectClaim",
    "OperationName",
    "OperationRequest",
    "OperationResult",
    "OperationStatus",
    "PathEscapeError",
    "RequestBounds",
    "TypedOperationError",
    "UnknownOperationError",
    "canonical_control_json_bytes",
    "operation_authority",
    "operation_request_json_schema",
    "operation_result_json_schema",
]
