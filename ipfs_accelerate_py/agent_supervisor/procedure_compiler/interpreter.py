"""Fail-closed deterministic interpreter for proof-carrying procedures.

``ProcedureIR`` is data.  It never gains authority merely by being parsed or
content addressed.  This module is the deliberately small runtime boundary
that combines that data with *injected* trusted operation implementations and
independent admission producers.

The interpreter has no built-in shell, Python, network, filesystem, policy,
certificate, or completion implementation.  In particular, callbacks live
only in :class:`TrustedOperationCatalog`; neither callbacks nor their return
bodies are serialized into procedure artifacts or checkpoints.

The P0 runtime makes interruption ambiguity explicit.  A checkpoint is
persisted immediately before dispatch.  If recovery finds that checkpoint
without an independently observed result, execution terminates with
``unknown_external_outcome``.  It never guesses that the operation was safe to
replay, including when the procedure labels the step idempotent.
"""

from __future__ import annotations

import fnmatch
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Final, Protocol

from ..proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from .contracts import (
    ProcedureCertificate,
    ProcedureExecutionTrace,
    ProcedureFailure,
    ProcedureInvocation,
    ProcedureInvocationReceipt,
    ProcedureOutcome,
    ProcedureOutcomeStatus,
    ProcedureSpec,
    ProcedureTraceEntry,
    RiskClass,
    StepOperation,
    TraceEventStatus,
    TraceState,
    ValueType,
)
from .procedure_ir import ProcedureIRParser, validate_procedure_spec

MAX_RUNTIME_TEXT_BYTES: Final[int] = 16_384
MAX_RUNTIME_COLLECTION_ITEMS: Final[int] = 4_096
MAX_RUNTIME_DEPTH: Final[int] = 16
MAX_RUNTIME_STEPS: Final[int] = 16_384
MAX_RUNTIME_RETRIES: Final[int] = 32
MAX_RUNTIME_LOOP_ITERATIONS: Final[int] = 1_024
MAX_RUNTIME_OUTPUT_BYTES: Final[int] = 1_048_576


class ProcedureInterpreterError(ValueError):
    """A procedure could not be executed without violating a hard gate."""

    def __init__(self, message: str, *, reason_code: str = "invalid_runtime_input") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class ExecutionMode(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Closed runtime modes; only ``live`` may perform admitted mutation."""

    LIVE = "live"
    SHADOW = "shadow"
    TEST = "test"


class AdmissionKind(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """What an external admission decision independently establishes."""

    PRODUCTION_CERTIFICATE = "production_certificate"
    TEST_CERTIFICATE = "test_certificate"
    AUTHORITY = "authority"
    PRECONDITION = "precondition"
    INVARIANT = "invariant"
    OBSERVATION = "observation"
    POSTCONDITION = "postcondition"
    VALIDATION = "validation"
    ROLLBACK = "rollback"
    COMPATIBILITY = "compatibility"
    CHECKPOINT = "checkpoint"


class RuntimeFailureCode(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Closed, machine-actionable terminal and recovery reasons."""

    INVALID_PROCEDURE = "invalid_procedure"
    INVALID_INVOCATION = "invalid_invocation"
    CERTIFICATE_REQUIRED = "certificate_required"
    CERTIFICATE_REJECTED = "certificate_rejected"
    STALE_CERTIFICATE = "stale_certificate"
    REVOKED_PROCEDURE = "revoked_procedure"
    BINDING_MISMATCH = "binding_mismatch"
    REGISTRY_DRIFT = "registry_drift"
    OPERATION_CONTRACT_DRIFT = "operation_contract_drift"
    AUTHORITY_REJECTED = "authority_rejected"
    STALE_FENCING = "stale_fencing"
    ISOLATION_ACQUISITION_FAILED = "isolation_acquisition_failed"
    ISOLATION_RELEASE_FAILED = "isolation_release_failed"
    PRECONDITION_FAILED = "precondition_failed"
    INVARIANT_FAILED = "invariant_failed"
    RESOURCE_RESERVATION_FAILED = "resource_reservation_failed"
    TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"
    RESOURCE_BUDGET_EXHAUSTED = "resource_budget_exhausted"
    TIME_BUDGET_EXHAUSTED = "time_budget_exhausted"
    UNKNOWN_OPERATION = "unknown_operation"
    OPERATION_FAILED = "operation_failed"
    EFFECT_VIOLATION = "effect_violation"
    SCOPE_ESCAPE = "scope_escape"
    READ_SCOPE_ESCAPE = "read_scope_escape"
    OBSERVATION_FAILED = "observation_failed"
    VALIDATION_FAILED = "validation_failed"
    POSTCONDITION_FAILED = "postcondition_failed"
    RETRY_EXHAUSTED = "retry_exhausted"
    CONTROL_FLOW_BOUNDS = "control_flow_bounds"
    CHECKPOINT_INVALID = "checkpoint_invalid"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    UNKNOWN_EXTERNAL_OUTCOME = "unknown_external_outcome"
    ROLLBACK_FAILED = "rollback_failed"
    FALLBACK_FAILED = "fallback_failed"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


class CheckpointPhase(str, Enum):  # noqa: UP042 - package supports Python 3.8
    READY = "ready"
    STEP_STARTED = "step_started"
    STEP_OBSERVED = "step_observed"
    ROLLING_BACK = "rolling_back"
    TERMINAL = "terminal"


class RuntimeOutcomeStatus(str, Enum):  # noqa: UP042 - package supports Python 3.8
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    BLOCKED = "blocked"
    INCOMPLETE = "incomplete"
    UNKNOWN_EXTERNAL_OUTCOME = "unknown_external_outcome"
    ESCALATED = "escalated"
    QUARANTINED = "quarantined"
    REFUSED = "refused"
    CANCELLED = "cancelled"


class TraceEventKind(str, Enum):  # noqa: UP042 - package supports Python 3.8
    INVOCATION_ADMITTED = "invocation_admitted"
    ISOLATION_ACQUIRED = "isolation_acquired"
    RESOURCE_RESERVED = "resource_reserved"
    PRECONDITION_ADMITTED = "precondition_admitted"
    INVARIANT_ADMITTED = "invariant_admitted"
    STEP_STARTED = "step_started"
    STEP_OBSERVED = "step_observed"
    STEP_RETRY = "step_retry"
    BRANCH_SELECTED = "branch_selected"
    LOOP_CHECKED = "loop_checked"
    FALLBACK_SELECTED = "fallback_selected"
    VALIDATION_ADMITTED = "validation_admitted"
    POSTCONDITION_ADMITTED = "postcondition_admitted"
    ROLLBACK_STARTED = "rollback_started"
    ROLLBACK_OBSERVED = "rollback_observed"
    TERMINAL = "terminal"


class CertificateClass(str, Enum):  # noqa: UP042 - package supports Python 3.8
    PRODUCTION = "production"
    TEST = "test"


def _enum_value(value: Any) -> str:
    return str(value.value if isinstance(value, Enum) else value)


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _first(value: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        found = _get(value, name, None)
        if found is not None:
            return found
    return default


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return tuple(value)
    raise ProcedureInterpreterError(
        "runtime collection must be a sequence", reason_code="invalid_runtime_input"
    )


def _canonical(value: Any, *, depth: int = 0) -> Any:
    """Return bounded JSON data suitable for a durable runtime checkpoint."""

    if depth > MAX_RUNTIME_DEPTH:
        raise ProcedureInterpreterError(
            "runtime value exceeds maximum nesting",
            reason_code="runtime_value_bounds",
        )
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        if "\x00" in value or len(value.encode("utf-8")) > MAX_RUNTIME_TEXT_BYTES:
            raise ProcedureInterpreterError(
                "runtime text is invalid or unbounded",
                reason_code="runtime_value_bounds",
            )
        return value
    if isinstance(value, float):
        raise ProcedureInterpreterError(
            "floating-point runtime values are not in the canonical profile",
            reason_code="runtime_value_bounds",
        )
    if isinstance(value, Enum):
        return _canonical(value.value, depth=depth + 1)
    if isinstance(value, Mapping):
        if len(value) > MAX_RUNTIME_COLLECTION_ITEMS:
            raise ProcedureInterpreterError(
                "runtime mapping exceeds bound", reason_code="runtime_value_bounds"
            )
        if any(not isinstance(key, str) for key in value):
            raise ProcedureInterpreterError(
                "runtime mapping keys must be strings",
                reason_code="runtime_value_bounds",
            )
        return {key: _canonical(item, depth=depth + 1) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        if len(value) > MAX_RUNTIME_COLLECTION_ITEMS:
            raise ProcedureInterpreterError(
                "runtime sequence exceeds bound", reason_code="runtime_value_bounds"
            )
        return [_canonical(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _canonical(to_dict(), depth=depth + 1)
    to_record = getattr(value, "to_record", None)
    if callable(to_record):
        return _canonical(to_record(), depth=depth + 1)
    if is_dataclass(value):
        return _canonical(
            {item.name: getattr(value, item.name) for item in fields(value)},
            depth=depth + 1,
        )
    raise ProcedureInterpreterError(
        "runtime value contains non-serializable object",
        reason_code="runtime_value_bounds",
    )


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = canonical_json_bytes(_canonical(value))
    except Exception as exc:
        raise ProcedureInterpreterError(
            "runtime value is outside the canonical DAG-JSON profile",
            reason_code="runtime_value_bounds",
        ) from exc
    if len(encoded) > MAX_RUNTIME_OUTPUT_BYTES:
        raise ProcedureInterpreterError(
            "runtime artifact exceeds maximum bytes",
            reason_code="runtime_value_bounds",
        )
    return encoded


def _content_id(namespace: str, value: Any) -> str:
    # Namespace separation is inside the canonical DAG-JSON value.  The
    # returned identity remains the repository's CIDv1 DAG-JSON identity.
    body = {"namespace": namespace, "value": _canonical(value)}
    _canonical_bytes(body)
    return content_identity(body)


def _artifact_id(value: Any, namespace: str) -> str:
    for name in ("content_id", "cid", "procedure_cid", "invocation_cid"):
        candidate = _get(value, name, "")
        if isinstance(candidate, str) and candidate:
            return candidate
    return _content_id(namespace, value)


def _identifier(value: Any, *names: str) -> str:
    candidate = _first(value, names, "")
    if isinstance(candidate, Enum):
        candidate = candidate.value
    if not isinstance(candidate, str):
        return ""
    return candidate


def _path(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ProcedureInterpreterError(
            "{} must be a non-empty relative path".format(field_name),
            reason_code=RuntimeFailureCode.SCOPE_ESCAPE.value,
        )
    if "\x00" in value or PureWindowsPath(value).is_absolute():
        raise ProcedureInterpreterError(
            "{} escapes procedure scope".format(field_name),
            reason_code=RuntimeFailureCode.SCOPE_ESCAPE.value,
        )
    normalized = value.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or ".." in candidate.parts or normalized.startswith("//"):
        raise ProcedureInterpreterError(
            "{} escapes procedure scope".format(field_name),
            reason_code=RuntimeFailureCode.SCOPE_ESCAPE.value,
        )
    return candidate.as_posix()


def _path_is_within(path: str, prefixes: Sequence[str]) -> bool:
    normalized = _path(path, field_name="observed path")
    candidate = PurePosixPath(normalized)
    for prefix in prefixes:
        declared = _path(prefix, field_name="declared scope")
        if declared == ".":
            return True
        if any(marker in declared for marker in "*?["):
            if _path_pattern_matches(normalized, declared):
                return True
            continue
        root = PurePosixPath(declared)
        if candidate == root or root in candidate.parents:
            return True
    return False


def _path_pattern_matches(path: str, pattern: str) -> bool:
    """Match repository paths without allowing ``*`` to cross a segment.

    A segment containing ordinary glob syntax is matched with
    :func:`fnmatch.fnmatchcase`; an exact ``**`` segment is the only construct
    allowed to consume multiple path segments.  The iterative state set is
    bounded by the already-bounded path lengths and avoids regex/glob engine
    ambiguity.
    """

    path_parts = PurePosixPath(path).parts
    pattern_parts = PurePosixPath(pattern).parts
    if any("**" in item and item != "**" for item in pattern_parts):
        return False
    states: set[tuple[int, int]] = {(0, 0)}
    visited: set[tuple[int, int]] = set()
    while states:
        pattern_index, path_index = states.pop()
        marker = (pattern_index, path_index)
        if marker in visited:
            continue
        visited.add(marker)
        if pattern_index == len(pattern_parts):
            if path_index == len(path_parts):
                return True
            continue
        segment = pattern_parts[pattern_index]
        if segment == "**":
            states.add((pattern_index + 1, path_index))
            if path_index < len(path_parts):
                states.add((pattern_index, path_index + 1))
            continue
        if path_index < len(path_parts) and fnmatch.fnmatchcase(path_parts[path_index], segment):
            states.add((pattern_index + 1, path_index + 1))
    return False


_RUNTIME_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/+\-]*$")


def _reject_embedded_path_escape(value: Any, *, field_name: str) -> None:
    """Reject absolute/traversal strings even inside structured values.

    Mapping keys are data too: operation adapters commonly interpret them as
    paths, symbols, or field selectors.  Validate keys and values recursively
    so a structured value cannot bypass the same boundary enforced for an
    ordinary string value.
    """

    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        candidate = PurePosixPath(normalized)
        if (
            PureWindowsPath(value).is_absolute()
            or candidate.is_absolute()
            or ".." in candidate.parts
            or normalized.startswith("//")
        ):
            raise ProcedureInterpreterError(
                f"{field_name} contains an absolute or traversal value",
                reason_code=RuntimeFailureCode.SCOPE_ESCAPE.value,
            )
        return
    if isinstance(value, Mapping):
        if len(value) > MAX_RUNTIME_COLLECTION_ITEMS:
            raise ProcedureInterpreterError(
                f"{field_name} exceeds its mapping bound",
                reason_code="runtime_value_bounds",
            )
        for key, item in value.items():
            if (
                not isinstance(key, str)
                or "\x00" in key
                or len(key.encode("utf-8")) > MAX_RUNTIME_TEXT_BYTES
            ):
                raise ProcedureInterpreterError(
                    f"{field_name} contains an invalid mapping key",
                    reason_code="runtime_value_bounds",
                )
            _reject_embedded_path_escape(key, field_name=f"{field_name} key")
            _reject_embedded_path_escape(item, field_name=field_name)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        if len(value) > MAX_RUNTIME_COLLECTION_ITEMS:
            raise ProcedureInterpreterError(
                f"{field_name} exceeds its sequence bound",
                reason_code="runtime_value_bounds",
            )
        for item in value:
            _reject_embedded_path_escape(item, field_name=field_name)


def _validate_typed_value(
    value: Any,
    value_type: ValueType | str,
    *,
    field_name: str,
    scope_paths: Sequence[str],
    allowed_values: Sequence[Any] = (),
    allow_none: bool = False,
) -> Any:
    """Validate one runtime value against the closed ProcedureIR type set."""

    if value is None and allow_none:
        return None
    try:
        expected = ValueType(value_type)
    except (TypeError, ValueError) as exc:
        raise ProcedureInterpreterError(
            f"{field_name} has an unknown value type",
            reason_code=RuntimeFailureCode.INVALID_PROCEDURE.value,
        ) from exc
    canonical = _canonical(value)
    _reject_embedded_path_escape(canonical, field_name=field_name)
    valid = True
    if expected is ValueType.STRING:
        valid = type(canonical) is str
    elif expected is ValueType.INTEGER:
        valid = type(canonical) is int
    elif expected is ValueType.BOOLEAN:
        valid = type(canonical) is bool
    elif expected is ValueType.IDENTIFIER:
        valid = (
            type(canonical) is str
            and bool(canonical)
            and bool(_RUNTIME_IDENTIFIER_RE.fullmatch(canonical))
        )
    elif expected is ValueType.CID:
        valid = type(canonical) is str and bool(canonical) and canonical == canonical.strip()
    elif expected is ValueType.RELATIVE_PATH:
        valid = type(canonical) is str
        if valid:
            canonical = _path(canonical, field_name=field_name)
            valid = bool(scope_paths) and _path_is_within(canonical, scope_paths)
    elif expected is ValueType.ENUM:
        valid = bool(allowed_values) and canonical in tuple(allowed_values)
    elif expected is ValueType.STRING_SEQUENCE:
        valid = isinstance(canonical, list) and all(type(item) is str for item in canonical)
    elif expected is ValueType.CID_SEQUENCE:
        valid = isinstance(canonical, list) and all(
            type(item) is str and bool(item) for item in canonical
        )
    elif expected is ValueType.STRUCTURED:
        valid = isinstance(canonical, (dict, list))
    if not valid:
        reason = (
            RuntimeFailureCode.SCOPE_ESCAPE
            if expected is ValueType.RELATIVE_PATH
            else RuntimeFailureCode.INVALID_INVOCATION
        )
        raise ProcedureInterpreterError(
            f"{field_name} does not satisfy {expected.value}",
            reason_code=reason.value,
        )
    return canonical


def _value_type_map(
    value: Mapping[str, ValueType | str], field_name: str
) -> Mapping[str, ValueType]:
    if not isinstance(value, Mapping) or len(value) > MAX_RUNTIME_COLLECTION_ITEMS:
        raise ProcedureInterpreterError(f"{field_name} must be a bounded mapping")
    normalized: dict[str, ValueType] = {}
    for name, value_type in sorted(value.items()):
        if not isinstance(name, str) or not name:
            raise ProcedureInterpreterError(f"{field_name} keys must be identifiers")
        try:
            normalized[name] = ValueType(value_type)
        except (TypeError, ValueError) as exc:
            raise ProcedureInterpreterError(f"{field_name} contains an unknown type") from exc
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class RuntimeIdentity:
    """Trusted current state supplied by the host, not by ProcedureIR."""

    repository_id: str
    repository_commit: str
    tree_id: str
    objective_id: str
    task_id: str
    contract_revision: str
    policy_revision: str
    environment_id: str
    registry_revision: str
    operation_catalog_revision: str
    now_ms: int
    active_lease_id: str = ""
    fencing_token: int = 0

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "repository_commit",
            "tree_id",
            "objective_id",
            "task_id",
            "contract_revision",
            "policy_revision",
            "environment_id",
            "registry_revision",
            "operation_catalog_revision",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ProcedureInterpreterError("RuntimeIdentity.{} is invalid".format(name))
        if isinstance(self.now_ms, bool) or not isinstance(self.now_ms, int) or self.now_ms < 0:
            raise ProcedureInterpreterError("RuntimeIdentity.now_ms must be non-negative")
        if not isinstance(self.active_lease_id, str):
            raise ProcedureInterpreterError("RuntimeIdentity.active_lease_id must be text")
        if (
            isinstance(self.fencing_token, bool)
            or not isinstance(self.fencing_token, int)
            or self.fencing_token < 0
        ):
            raise ProcedureInterpreterError("RuntimeIdentity.fencing_token must be non-negative")
        if bool(self.active_lease_id) != bool(self.fencing_token):
            raise ProcedureInterpreterError(
                "RuntimeIdentity lease and fencing token must be bound together"
            )

    def binding_values(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "repository_id": self.repository_id,
                "repository_commit": self.repository_commit,
                "tree_id": self.tree_id,
                "objective_id": self.objective_id,
                "task_id": self.task_id,
                "contract_revision": self.contract_revision,
                "policy_revision": self.policy_revision,
                "environment_id": self.environment_id,
            }
        )


@dataclass(frozen=True)
class AdmissionDecision:
    """One external producer's bounded, auditable admission result."""

    admitted: bool
    kind: AdmissionKind
    receipt_cids: tuple[str, ...] = ()
    reason_code: str = ""
    observed_at_ms: int = 0
    predicate_value: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", AdmissionKind(self.kind))
        receipts = tuple(self.receipt_cids)
        if self.admitted and not receipts:
            raise ProcedureInterpreterError(
                "an admitted decision requires independent receipt evidence",
                reason_code="missing_admission_receipt",
            )
        if any(not isinstance(item, str) or not item for item in receipts):
            raise ProcedureInterpreterError("admission receipt CIDs are invalid")
        object.__setattr__(self, "receipt_cids", receipts)
        if not isinstance(self.reason_code, str):
            raise ProcedureInterpreterError("admission reason_code must be text")
        if self.predicate_value is not None and type(self.predicate_value) is not bool:
            raise ProcedureInterpreterError("admission predicate_value must be boolean")


@dataclass(frozen=True)
class AdmissionRequest:
    """Context given to an independent admission producer."""

    kind: AdmissionKind
    subject: Any
    procedure: ProcedureSpec
    invocation: ProcedureInvocation
    certificate: ProcedureCertificate
    runtime: RuntimeIdentity
    mode: ExecutionMode
    step_id: str = ""
    variables: Mapping[str, Any] = field(default_factory=dict)
    evidence_cids: tuple[str, ...] = ()
    operation_result: Any = None


class AdmissionProducer(Protocol):
    """Fail-closed port to existing certificate/evidence/authority producers."""

    def admit(self, request: AdmissionRequest) -> AdmissionDecision: ...


@dataclass(frozen=True)
class InterpreterAdmissionPorts:
    """Explicit trust boundaries required by the interpreter.

    A caller may route several fields to the same existing service adapter,
    but every field must be populated.  There is intentionally no permissive
    default and no adapter that accepts assertions from the procedure itself.
    """

    certificate: AdmissionProducer
    compatibility: AdmissionProducer
    authority: AdmissionProducer
    evidence: AdmissionProducer

    def __post_init__(self) -> None:
        for name in ("certificate", "compatibility", "authority", "evidence"):
            producer = getattr(self, name)
            if not callable(getattr(producer, "admit", None)):
                raise ProcedureInterpreterError(
                    "admission port {} is not configured".format(name),
                    reason_code="missing_admission_port",
                )


class BudgetReservationPort(Protocol):
    """Host-owned reservation boundary; procedures cannot reserve themselves."""

    def reserve(self, request: BudgetReservationRequest) -> BudgetReservation: ...

    def release(self, reservation: BudgetReservation, *, consumed: RuntimeCost) -> None: ...


class IsolationReservationPort(Protocol):
    """Adapter to existing lease/worktree owners; no implementation lives here."""

    def acquire(self, request: IsolationReservationRequest) -> IsolationReservation: ...

    def compensate(self, reservation: IsolationReservation, *, reason_code: str) -> str: ...

    def release(self, reservation: IsolationReservation) -> None: ...


@dataclass(frozen=True)
class IsolationReservationRequest:
    invocation_cid: str
    procedure_cid: str
    repository_id: str
    tree_id: str
    lease_id: str
    fencing_token: int
    scope_paths: tuple[str, ...]
    worktree_required: bool
    read_only: bool


@dataclass(frozen=True)
class IsolationReservation:
    reservation_id: str
    lease_id: str
    fencing_token: int
    scope_paths: tuple[str, ...]
    worktree_id: str
    read_only: bool
    receipt_cid: str

    def __post_init__(self) -> None:
        if not self.reservation_id or not self.receipt_cid:
            raise ProcedureInterpreterError("isolation reservation lacks admitted evidence")
        if not isinstance(self.lease_id, str):
            raise ProcedureInterpreterError("isolation lease_id must be text")
        if (
            isinstance(self.fencing_token, bool)
            or not isinstance(self.fencing_token, int)
            or self.fencing_token < 0
        ):
            raise ProcedureInterpreterError("isolation fencing token is invalid")
        scopes = tuple(_path(item, field_name="isolation scope") for item in self.scope_paths)
        if not scopes:
            raise ProcedureInterpreterError("isolation scope must not be empty")
        object.__setattr__(self, "scope_paths", scopes)
        if type(self.read_only) is not bool:
            raise ProcedureInterpreterError("isolation read_only must be boolean")
        if not self.read_only and (
            not self.lease_id or self.fencing_token <= 0 or not self.worktree_id
        ):
            raise ProcedureInterpreterError(
                "effectful isolation requires lease, fence, and worktree"
            )


@dataclass(frozen=True)
class RuntimeCost:
    token_count: int = 0
    resource_units: int = 0
    elapsed_ms: int = 0

    def __post_init__(self) -> None:
        for name in ("token_count", "resource_units", "elapsed_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProcedureInterpreterError("RuntimeCost.{} must be non-negative".format(name))

    def plus(self, other: RuntimeCost) -> RuntimeCost:
        return RuntimeCost(
            token_count=self.token_count + other.token_count,
            resource_units=self.resource_units + other.resource_units,
            elapsed_ms=self.elapsed_ms + other.elapsed_ms,
        )


@dataclass(frozen=True)
class BudgetReservationRequest:
    invocation_cid: str
    procedure_cid: str
    token_limit: int
    resource_limit: int
    time_limit_ms: int
    envelope: Any


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: str
    token_limit: int
    resource_limit: int
    time_limit_ms: int
    receipt_cid: str

    def __post_init__(self) -> None:
        if not self.reservation_id or not self.receipt_cid:
            raise ProcedureInterpreterError("budget reservation lacks identity/evidence")
        for name in ("token_limit", "resource_limit", "time_limit_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ProcedureInterpreterError("budget reservation is invalid")


@dataclass(frozen=True)
class OperationRequest:
    """Bounded request passed to one trusted, injected operation callback."""

    invocation_cid: str
    procedure_cid: str
    step_id: str
    operation: str
    operation_contract: str
    inputs: Mapping[str, Any]
    idempotency_key: str
    timeout_ms: int
    attempt: int
    deadline_ms: int
    mode: ExecutionMode
    dry_run: bool
    scope_paths: tuple[str, ...]
    lease_id: str
    fencing_token: int


@dataclass(frozen=True)
class OperationResult:
    """Observed result from a trusted operation implementation.

    ``success`` is operational status only.  It does not establish a
    procedure observation, postcondition, proof, validation, or completion.
    Those are admitted through :class:`AdmissionProducer`.
    """

    success: bool
    outputs: Mapping[str, Any] = field(default_factory=dict)
    observed_effect_ids: tuple[str, ...] = ()
    read_paths: tuple[str, ...] = ()
    changed_paths: tuple[str, ...] = ()
    evidence_cids: tuple[str, ...] = ()
    cost: RuntimeCost = field(default_factory=RuntimeCost)
    failure_code: str = ""
    retryable: bool = False
    external_outcome_observed: bool = True
    new_evidence: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise ProcedureInterpreterError("OperationResult.success must be boolean")
        if not isinstance(self.outputs, Mapping):
            raise ProcedureInterpreterError("OperationResult.outputs must be a mapping")
        canonical_outputs = _canonical(self.outputs)
        object.__setattr__(self, "outputs", MappingProxyType(canonical_outputs))
        for name in ("observed_effect_ids", "read_paths", "changed_paths", "evidence_cids"):
            values = tuple(getattr(self, name))
            if any(not isinstance(item, str) or not item for item in values):
                raise ProcedureInterpreterError("OperationResult.{} is invalid".format(name))
            object.__setattr__(self, name, values)
        if not isinstance(self.cost, RuntimeCost):
            raise ProcedureInterpreterError("OperationResult.cost must be RuntimeCost")
        if not isinstance(self.failure_code, str):
            raise ProcedureInterpreterError("OperationResult.failure_code must be text")
        if self.success and self.failure_code:
            raise ProcedureInterpreterError("successful operation cannot carry failure_code")
        if not self.external_outcome_observed:
            # No output/effect assertion is trusted when the producer says the
            # actual external result was not observed.
            if self.outputs or self.observed_effect_ids or self.changed_paths:
                raise ProcedureInterpreterError(
                    "unobserved external outcome cannot claim outputs or effects"
                )


class OperationExecutionError(RuntimeError):
    """A trusted adapter's typed failure before a complete observation."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: str = RuntimeFailureCode.OPERATION_FAILED.value,
        outcome_observed: bool = False,
        retryable: bool = False,
        evidence_cids: Sequence[str] = (),
        new_evidence: bool = False,
    ) -> None:
        super().__init__(message)
        self.failure_code = str(failure_code)
        self.outcome_observed = bool(outcome_observed)
        self.retryable = bool(retryable)
        self.evidence_cids = tuple(evidence_cids)
        self.new_evidence = bool(new_evidence)


OperationHandler = Callable[[OperationRequest], OperationResult]


@dataclass(frozen=True)
class TrustedOperation:
    """Runtime-only binding between an allowlisted contract and callback.

    Handlers are synchronous trusted adapters.  They must enforce their own
    operation timeout at the external boundary; the interpreter checks elapsed
    time and refuses later dispatch, but deliberately does not use unsafe
    worker threads or process termination to interrupt an in-flight callback.
    """

    operation: str
    operation_contract: str
    handler: OperationHandler = field(repr=False, compare=False)
    allowed_effect_ids: tuple[str, ...] = ()
    read_only: bool = True
    supports_dry_run: bool = False
    maximum_timeout_ms: int = 3_600_000
    maximum_retries: int = 0
    input_types: Mapping[str, ValueType | str] = field(default_factory=dict)
    output_types: Mapping[str, ValueType | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operation = _enum_value(self.operation)
        if not operation or not self.operation_contract:
            raise ProcedureInterpreterError("trusted operation identity is incomplete")
        if not callable(self.handler):
            raise ProcedureInterpreterError("trusted operation handler must be callable")
        if operation in {
            "ARBITRARY_SHELL",
            "ARBITRARY_PYTHON",
            "ARBITRARY_NETWORK_REQUEST",
            "ARBITRARY_FILESYSTEM_PATH",
            "DISABLE_VALIDATION",
            "MODIFY_AUTHORITY_POLICY",
            "MODIFY_TRUSTED_KEYS",
            "CLAIM_COMPLETION",
        }:
            raise ProcedureInterpreterError(
                "forbidden operation cannot enter trusted catalog",
                reason_code=RuntimeFailureCode.UNKNOWN_OPERATION.value,
            )
        object.__setattr__(self, "operation", operation)
        allowed = tuple(sorted(set(self.allowed_effect_ids)))
        if any(not isinstance(item, str) or not item for item in allowed):
            raise ProcedureInterpreterError("trusted operation effects are invalid")
        object.__setattr__(self, "allowed_effect_ids", allowed)
        object.__setattr__(
            self, "input_types", _value_type_map(self.input_types, "trusted operation input_types")
        )
        object.__setattr__(
            self,
            "output_types",
            _value_type_map(self.output_types, "trusted operation output_types"),
        )
        if (
            isinstance(self.maximum_timeout_ms, bool)
            or not isinstance(self.maximum_timeout_ms, int)
            or self.maximum_timeout_ms <= 0
        ):
            raise ProcedureInterpreterError("trusted operation timeout is invalid")
        if (
            isinstance(self.maximum_retries, bool)
            or not isinstance(self.maximum_retries, int)
            or self.maximum_retries < 0
            or self.maximum_retries > MAX_RUNTIME_RETRIES
        ):
            raise ProcedureInterpreterError("trusted operation retry bound is invalid")


class TrustedOperationCatalog:
    """Immutable dispatch catalog.  Its callbacks are intentionally not data."""

    def __init__(self, revision: str, operations: Sequence[TrustedOperation]) -> None:
        if not isinstance(revision, str) or not revision:
            raise ProcedureInterpreterError("operation catalog revision is required")
        by_name: dict[str, TrustedOperation] = {}
        for operation in operations:
            if not isinstance(operation, TrustedOperation):
                raise ProcedureInterpreterError("catalog entries must be TrustedOperation")
            if operation.operation in by_name:
                raise ProcedureInterpreterError("duplicate trusted operation")
            by_name[operation.operation] = operation
        self._revision = revision
        self._operations = MappingProxyType(by_name)

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def operation_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._operations))

    def resolve(self, operation: Any, operation_contract: str) -> TrustedOperation:
        name = _enum_value(operation)
        registered = self._operations.get(name)
        if registered is None:
            raise ProcedureInterpreterError(
                "operation {!r} is not in the trusted catalog".format(name),
                reason_code=RuntimeFailureCode.UNKNOWN_OPERATION.value,
            )
        if registered.operation_contract != operation_contract:
            raise ProcedureInterpreterError(
                "operation contract revision does not match trusted catalog",
                reason_code=RuntimeFailureCode.OPERATION_CONTRACT_DRIFT.value,
            )
        return registered


@dataclass(frozen=True)
class RuntimeTraceEntry:
    sequence: int
    event: TraceEventKind
    step_id: str = ""
    attempt: int = 0
    operation: str = ""
    evidence_cids: tuple[str, ...] = ()
    observed_effect_ids: tuple[str, ...] = ()
    cost: RuntimeCost = field(default_factory=RuntimeCost)
    reason_code: str = ""
    detail: str = ""
    input_digest: str = ""
    output_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event": self.event.value,
            "step_id": self.step_id,
            "attempt": self.attempt,
            "operation": self.operation,
            "evidence_cids": list(self.evidence_cids),
            "observed_effect_ids": list(self.observed_effect_ids),
            "cost": {
                "token_count": self.cost.token_count,
                "resource_units": self.cost.resource_units,
                "elapsed_ms": self.cost.elapsed_ms,
            },
            "reason_code": self.reason_code,
            "detail": self.detail[:1024],
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
        }


@dataclass(frozen=True)
class InterpreterCheckpoint:
    invocation_cid: str
    procedure_cid: str
    idempotency_key: str
    phase: CheckpointPhase
    current_node_id: str
    started_step_id: str
    attempt: int
    variables: Mapping[str, Any]
    trace_entries: tuple[RuntimeTraceEntry, ...]
    observed_effect_ids: tuple[str, ...]
    evidence_cids: tuple[str, ...]
    validation_receipt_cids: tuple[str, ...]
    satisfied_postcondition_ids: tuple[str, ...]
    rollback_receipt_cids: tuple[str, ...]
    executed_step_ids: tuple[str, ...]
    admitted_observation_ids: tuple[str, ...]
    loop_counts: Mapping[str, int]
    cost: RuntimeCost
    terminal_at_ms: int = 0
    failure_code: str = ""
    status: RuntimeOutcomeStatus = RuntimeOutcomeStatus.INCOMPLETE
    changed_paths: tuple[str, ...] = ()

    @property
    def checkpoint_cid(self) -> str:
        return _content_id("procedure-checkpoint", self.to_dict(include_cid=False))

    def to_dict(self, *, include_cid: bool = True) -> dict[str, Any]:
        value = {
            "invocation_cid": self.invocation_cid,
            "procedure_cid": self.procedure_cid,
            "idempotency_key": self.idempotency_key,
            "phase": self.phase.value,
            "current_node_id": self.current_node_id,
            "started_step_id": self.started_step_id,
            "attempt": self.attempt,
            "variables": dict(self.variables),
            "trace_entries": [item.to_dict() for item in self.trace_entries],
            "observed_effect_ids": list(self.observed_effect_ids),
            "evidence_cids": list(self.evidence_cids),
            "validation_receipt_cids": list(self.validation_receipt_cids),
            "satisfied_postcondition_ids": list(self.satisfied_postcondition_ids),
            "rollback_receipt_cids": list(self.rollback_receipt_cids),
            "executed_step_ids": list(self.executed_step_ids),
            "admitted_observation_ids": list(self.admitted_observation_ids),
            "loop_counts": dict(self.loop_counts),
            "cost": {
                "token_count": self.cost.token_count,
                "resource_units": self.cost.resource_units,
                "elapsed_ms": self.cost.elapsed_ms,
            },
            "terminal_at_ms": self.terminal_at_ms,
            "failure_code": self.failure_code,
            "status": self.status.value,
            "changed_paths": list(self.changed_paths),
        }
        if include_cid:
            value["checkpoint_cid"] = self.checkpoint_cid
        return value


class CheckpointStore(Protocol):
    def load(self, invocation_cid: str) -> InterpreterCheckpoint | None: ...

    def save(self, checkpoint: InterpreterCheckpoint) -> None: ...

    def execution_lock(self, invocation_cid: str) -> AbstractContextManager[None]: ...


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._values: dict[str, InterpreterCheckpoint] = {}
        self._lock = threading.RLock()
        self._execution_locks: dict[str, threading.Lock] = {}

    def load(self, invocation_cid: str) -> InterpreterCheckpoint | None:
        with self._lock:
            return self._values.get(invocation_cid)

    def save(self, checkpoint: InterpreterCheckpoint) -> None:
        if not isinstance(checkpoint, InterpreterCheckpoint):
            raise ProcedureInterpreterError("checkpoint store received wrong type")
        # Force bounded canonical serialization before admitting the checkpoint.
        _canonical_bytes(checkpoint.to_dict())
        with self._lock:
            self._values[checkpoint.invocation_cid] = checkpoint

    @contextmanager
    def execution_lock(self, invocation_cid: str):
        """Store-owned single-flight lock shared by interpreter instances."""

        if not isinstance(invocation_cid, str) or not invocation_cid:
            raise ProcedureInterpreterError("execution lock requires an invocation CID")
        with self._lock:
            lock = self._execution_locks.get(invocation_cid)
            if lock is None:
                if len(self._execution_locks) >= MAX_RUNTIME_COLLECTION_ITEMS:
                    raise ProcedureInterpreterError(
                        "checkpoint execution-lock bound exhausted",
                        reason_code=RuntimeFailureCode.RESOURCE_BUDGET_EXHAUSTED.value,
                    )
                lock = threading.Lock()
                self._execution_locks[invocation_cid] = lock
        lock.acquire()
        try:
            yield
        finally:
            lock.release()


class IdempotencyStore(Protocol):
    def claim(self, key: str, invocation_cid: str) -> bool: ...


class InMemoryIdempotencyStore:
    """Atomic exact-key claim; a key never aliases a different invocation."""

    def __init__(self) -> None:
        self._claims: dict[str, str] = {}
        self._lock = threading.RLock()

    def claim(self, key: str, invocation_cid: str) -> bool:
        if not key or not invocation_cid:
            raise ProcedureInterpreterError("idempotency claim is incomplete")
        with self._lock:
            previous = self._claims.get(key)
            if previous is None:
                self._claims[key] = invocation_cid
                return True
            return previous == invocation_cid


class RuntimeClock(Protocol):
    def now_ms(self) -> int: ...

    def monotonic_ms(self) -> int: ...


class SystemRuntimeClock:
    def now_ms(self) -> int:
        return int(time.time() * 1000)

    def monotonic_ms(self) -> int:
        return int(time.monotonic() * 1000)


@dataclass(frozen=True)
class ProcedureExecution:
    """Public interpreter return value, retaining typed canonical artifacts."""

    trace: ProcedureExecutionTrace
    outcome: ProcedureOutcome
    receipt: ProcedureInvocationReceipt
    failure: ProcedureFailure | None
    cost: RuntimeCost
    resumed: bool
    checkpoint_cid: str


@dataclass
class _RunState:
    invocation_cid: str
    procedure_cid: str
    idempotency_key: str
    current_node_id: str
    variables: dict[str, Any]
    trace_entries: list[RuntimeTraceEntry] = field(default_factory=list)
    observed_effect_ids: set[str] = field(default_factory=set)
    changed_paths: set[str] = field(default_factory=set)
    evidence_cids: set[str] = field(default_factory=set)
    validation_receipt_cids: set[str] = field(default_factory=set)
    satisfied_postcondition_ids: set[str] = field(default_factory=set)
    rollback_receipt_cids: set[str] = field(default_factory=set)
    executed_step_ids: set[str] = field(default_factory=set)
    admitted_observation_ids: set[str] = field(default_factory=set)
    loop_counts: dict[str, int] = field(default_factory=dict)
    cost: RuntimeCost = field(default_factory=RuntimeCost)
    terminal_at_ms: int = 0
    started_step_id: str = ""
    attempt: int = 0
    failure_code: str = ""
    status: RuntimeOutcomeStatus = RuntimeOutcomeStatus.INCOMPLETE

    @classmethod
    def from_checkpoint(cls, checkpoint: InterpreterCheckpoint) -> _RunState:
        return cls(
            invocation_cid=checkpoint.invocation_cid,
            procedure_cid=checkpoint.procedure_cid,
            idempotency_key=checkpoint.idempotency_key,
            current_node_id=checkpoint.current_node_id,
            variables=dict(checkpoint.variables),
            trace_entries=list(checkpoint.trace_entries),
            observed_effect_ids=set(checkpoint.observed_effect_ids),
            changed_paths=set(checkpoint.changed_paths),
            evidence_cids=set(checkpoint.evidence_cids),
            validation_receipt_cids=set(checkpoint.validation_receipt_cids),
            satisfied_postcondition_ids=set(checkpoint.satisfied_postcondition_ids),
            rollback_receipt_cids=set(checkpoint.rollback_receipt_cids),
            executed_step_ids=set(checkpoint.executed_step_ids),
            admitted_observation_ids=set(checkpoint.admitted_observation_ids),
            loop_counts=dict(checkpoint.loop_counts),
            cost=checkpoint.cost,
            terminal_at_ms=checkpoint.terminal_at_ms,
            started_step_id=checkpoint.started_step_id,
            attempt=checkpoint.attempt,
            failure_code=checkpoint.failure_code,
            status=checkpoint.status,
        )

    def trace(
        self,
        event: TraceEventKind,
        *,
        step_id: str = "",
        attempt: int = 0,
        operation: str = "",
        evidence_cids: Sequence[str] = (),
        effects: Sequence[str] = (),
        cost: RuntimeCost | None = None,
        reason_code: str = "",
        detail: str = "",
        input_digest: str = "",
        output_digest: str = "",
    ) -> None:
        self.trace_entries.append(
            RuntimeTraceEntry(
                sequence=len(self.trace_entries),
                event=event,
                step_id=step_id,
                attempt=attempt,
                operation=operation,
                evidence_cids=tuple(evidence_cids),
                observed_effect_ids=tuple(effects),
                cost=cost or RuntimeCost(),
                reason_code=reason_code,
                detail=detail,
                input_digest=input_digest,
                output_digest=output_digest,
            )
        )

    def checkpoint(self, phase: CheckpointPhase) -> InterpreterCheckpoint:
        return InterpreterCheckpoint(
            invocation_cid=self.invocation_cid,
            procedure_cid=self.procedure_cid,
            idempotency_key=self.idempotency_key,
            phase=phase,
            current_node_id=self.current_node_id,
            started_step_id=self.started_step_id,
            attempt=self.attempt,
            variables=MappingProxyType(_canonical(self.variables)),
            trace_entries=tuple(self.trace_entries),
            observed_effect_ids=tuple(sorted(self.observed_effect_ids)),
            evidence_cids=tuple(sorted(self.evidence_cids)),
            validation_receipt_cids=tuple(sorted(self.validation_receipt_cids)),
            satisfied_postcondition_ids=tuple(sorted(self.satisfied_postcondition_ids)),
            rollback_receipt_cids=tuple(sorted(self.rollback_receipt_cids)),
            executed_step_ids=tuple(sorted(self.executed_step_ids)),
            admitted_observation_ids=tuple(sorted(self.admitted_observation_ids)),
            loop_counts=MappingProxyType(dict(sorted(self.loop_counts.items()))),
            cost=self.cost,
            terminal_at_ms=self.terminal_at_ms,
            failure_code=self.failure_code,
            status=self.status,
            changed_paths=tuple(sorted(self.changed_paths)),
        )


@dataclass(frozen=True)
class _StepExecutionResult:
    succeeded: bool
    next_node_id: str = ""
    failure_code: str = ""
    failure_transition: str = "abort"
    failure_target: str = ""
    unknown_external_outcome: bool = False


class ProcedureInterpreter:
    """Execute verified ``ProcedureIR`` through external trusted authorities.

    Every constructor dependency is a runtime object supplied by the host.
    There are no production-accepting defaults.  The same instance may be used
    concurrently when its injected ports and stores implement their declared
    atomic interfaces.
    """

    def __init__(
        self,
        *,
        operation_catalog: TrustedOperationCatalog,
        admissions: InterpreterAdmissionPorts,
        isolation: IsolationReservationPort,
        budget_reservations: BudgetReservationPort,
        checkpoints: CheckpointStore,
        idempotency: IdempotencyStore,
        clock: RuntimeClock | None = None,
    ) -> None:
        if not isinstance(operation_catalog, TrustedOperationCatalog):
            raise ProcedureInterpreterError("a TrustedOperationCatalog is required")
        if not isinstance(admissions, InterpreterAdmissionPorts):
            raise ProcedureInterpreterError("InterpreterAdmissionPorts are required")
        if not callable(getattr(budget_reservations, "reserve", None)) or not callable(
            getattr(budget_reservations, "release", None)
        ):
            raise ProcedureInterpreterError("budget reservation port is required")
        for name in ("acquire", "compensate", "release"):
            if not callable(getattr(isolation, name, None)):
                raise ProcedureInterpreterError(
                    "isolation reservation port is required",
                    reason_code="missing_isolation_port",
                )
        if not callable(getattr(checkpoints, "load", None)) or not callable(
            getattr(checkpoints, "save", None)
        ):
            raise ProcedureInterpreterError("checkpoint store is required")
        if not callable(getattr(checkpoints, "execution_lock", None)):
            raise ProcedureInterpreterError(
                "checkpoint store must provide an atomic execution lock",
                reason_code=RuntimeFailureCode.IDEMPOTENCY_CONFLICT.value,
            )
        if not callable(getattr(idempotency, "claim", None)):
            raise ProcedureInterpreterError("idempotency store is required")
        self._catalog = operation_catalog
        self._admissions = admissions
        self._isolation = isolation
        self._budget_reservations = budget_reservations
        self._checkpoints = checkpoints
        self._idempotency = idempotency
        self._clock = clock or SystemRuntimeClock()
        self._invocation_lock_guard = threading.RLock()
        self._invocation_locks: dict[str, threading.Lock] = {}

    @property
    def operation_catalog_revision(self) -> str:
        return self._catalog.revision

    def execute(
        self,
        procedure: ProcedureSpec | Mapping[str, Any],
        invocation: ProcedureInvocation,
        certificate: ProcedureCertificate | None,
        runtime: RuntimeIdentity,
        *,
        mode: ExecutionMode = ExecutionMode.LIVE,
    ) -> ProcedureExecution:
        """Single-flight an exact invocation without serializing other work."""

        invocation_cid = _artifact_id(invocation, "procedure-invocation")
        with self._invocation_lock_guard:
            lock = self._invocation_locks.get(invocation_cid)
            if lock is None:
                if len(self._invocation_locks) >= MAX_RUNTIME_COLLECTION_ITEMS:
                    raise ProcedureInterpreterError(
                        "concurrent invocation lock bound exhausted",
                        reason_code=RuntimeFailureCode.RESOURCE_BUDGET_EXHAUSTED.value,
                    )
                lock = threading.Lock()
                self._invocation_locks[invocation_cid] = lock
        with lock:
            execution_lock = self._checkpoints.execution_lock(invocation_cid)
            if not isinstance(execution_lock, AbstractContextManager):
                raise ProcedureInterpreterError(
                    "checkpoint store returned an invalid execution lock",
                    reason_code=RuntimeFailureCode.IDEMPOTENCY_CONFLICT.value,
                )
            with execution_lock:
                return self._execute_once(procedure, invocation, certificate, runtime, mode=mode)

    def _execute_once(
        self,
        procedure: ProcedureSpec | Mapping[str, Any],
        invocation: ProcedureInvocation,
        certificate: ProcedureCertificate | None,
        runtime: RuntimeIdentity,
        *,
        mode: ExecutionMode = ExecutionMode.LIVE,
    ) -> ProcedureExecution:
        """Validate, admit, execute, observe, and seal one invocation.

        Operational failure is represented by the returned typed outcome.
        Malformed caller input and a missing trust port raise
        :class:`ProcedureInterpreterError` before dispatch.
        """

        mode = ExecutionMode(mode)
        if isinstance(procedure, Mapping):
            procedure = ProcedureIRParser().parse(procedure)
        if not isinstance(procedure, ProcedureSpec):
            raise ProcedureInterpreterError(
                "procedure must be a ProcedureSpec",
                reason_code=RuntimeFailureCode.INVALID_PROCEDURE.value,
            )
        try:
            validate_procedure_spec(procedure)
        except Exception as exc:
            raise ProcedureInterpreterError(
                "ProcedureIR validation failed: {}".format(exc),
                reason_code=RuntimeFailureCode.INVALID_PROCEDURE.value,
            ) from exc
        if not isinstance(invocation, ProcedureInvocation):
            raise ProcedureInterpreterError(
                "invocation must be a ProcedureInvocation",
                reason_code=RuntimeFailureCode.INVALID_INVOCATION.value,
            )
        if not isinstance(runtime, RuntimeIdentity):
            raise ProcedureInterpreterError("trusted RuntimeIdentity is required")

        procedure_cid = _artifact_id(procedure, "procedure")
        invocation_cid = _artifact_id(invocation, "procedure-invocation")
        idempotency_key = _identifier(invocation, "idempotency_key")
        if not idempotency_key:
            raise ProcedureInterpreterError(
                "invocation idempotency key is required",
                reason_code=RuntimeFailureCode.INVALID_INVOCATION.value,
            )

        steps = {_identifier(item, "step_id"): item for item in procedure.steps}
        branches = {_identifier(item, "branch_id"): item for item in procedure.branches}
        loops = {_identifier(item, "loop_id"): item for item in procedure.loops}
        observations = {
            _identifier(item, "observation_id"): item for item in procedure.observations
        }
        operations = {
            step_id: self._catalog.resolve(step.operation, step.operation_contract)
            for step_id, step in steps.items()
        }
        self._validate_operation_value_contracts(procedure, steps, operations)
        allowed_operations = {_enum_value(item) for item in procedure.authority.allowed_operations}
        if any(_enum_value(step.operation) not in allowed_operations for step in steps.values()):
            raise ProcedureInterpreterError(
                "ProcedureIR step is outside its authority operation envelope",
                reason_code=RuntimeFailureCode.AUTHORITY_REJECTED.value,
            )
        effectful = any(not operation.read_only for operation in operations.values())
        if effectful and (
            not _identifier(invocation, "lease_id")
            or int(_get(invocation, "fencing_token", 0)) <= 0
        ):
            raise ProcedureInterpreterError(
                "effectful invocation requires an exact lease and fencing token",
                reason_code=RuntimeFailureCode.AUTHORITY_REJECTED.value,
            )
        if effectful and (
            invocation.lease_id != runtime.active_lease_id
            or invocation.fencing_token != runtime.fencing_token
        ):
            raise ProcedureInterpreterError(
                "effectful invocation lease or fencing token is stale",
                reason_code=RuntimeFailureCode.STALE_FENCING.value,
            )

        self._validate_exact_bindings(procedure, invocation, certificate, runtime)
        self._validate_identity_links(procedure, procedure_cid, invocation, certificate, runtime)
        self._validate_scope(procedure, invocation)
        self._validate_lifecycle(procedure, certificate, mode, effectful, runtime)

        certificate_decision = self._admit(
            self._admissions.certificate,
            AdmissionRequest(
                kind=AdmissionKind.PRODUCTION_CERTIFICATE
                if mode is ExecutionMode.LIVE
                else AdmissionKind.TEST_CERTIFICATE,
                subject=certificate,
                procedure=procedure,
                invocation=invocation,
                certificate=certificate,  # type: ignore[arg-type]
                runtime=runtime,
                mode=mode,
            ),
            expected=(
                (AdmissionKind.PRODUCTION_CERTIFICATE,)
                if mode is ExecutionMode.LIVE or effectful
                else (
                    AdmissionKind.PRODUCTION_CERTIFICATE,
                    AdmissionKind.TEST_CERTIFICATE,
                )
            ),
            failure=RuntimeFailureCode.CERTIFICATE_REJECTED,
        )
        compatibility = self._admit(
            self._admissions.compatibility,
            AdmissionRequest(
                kind=AdmissionKind.COMPATIBILITY,
                subject=runtime,
                procedure=procedure,
                invocation=invocation,
                certificate=certificate,  # type: ignore[arg-type]
                runtime=runtime,
                mode=mode,
                evidence_cids=certificate_decision.receipt_cids,
            ),
            expected=(AdmissionKind.COMPATIBILITY,),
            failure=RuntimeFailureCode.BINDING_MISMATCH,
        )

        if not self._idempotency.claim(idempotency_key, invocation_cid):
            raise ProcedureInterpreterError(
                "idempotency key is already bound to a different invocation",
                reason_code=RuntimeFailureCode.IDEMPOTENCY_CONFLICT.value,
            )

        saved = self._checkpoints.load(invocation_cid)
        resumed = saved is not None
        recovery_phase: CheckpointPhase | None = None
        recovered_terminal_status: RuntimeOutcomeStatus | None = None
        recovered_rollback_observation = False
        if saved is not None:
            self._validate_checkpoint(
                saved,
                procedure,
                invocation,
                procedure_cid,
                invocation_cid,
                idempotency_key,
            )
            checkpoint_decision = self._admit(
                self._admissions.evidence,
                AdmissionRequest(
                    kind=AdmissionKind.CHECKPOINT,
                    subject=saved,
                    procedure=procedure,
                    invocation=invocation,
                    certificate=certificate,  # type: ignore[arg-type]
                    runtime=runtime,
                    mode=mode,
                    evidence_cids=tuple(sorted(saved.evidence_cids)),
                ),
                expected=(AdmissionKind.CHECKPOINT,),
                failure=RuntimeFailureCode.CHECKPOINT_INVALID,
            )
            state = _RunState.from_checkpoint(saved)
            state.evidence_cids.update(certificate_decision.receipt_cids)
            state.evidence_cids.update(compatibility.receipt_cids)
            state.evidence_cids.update(checkpoint_decision.receipt_cids)
            recovery_phase = saved.phase
            if saved.phase is CheckpointPhase.STEP_STARTED:
                # Dispatch crossed the external boundary, but no trusted result
                # was persisted.  Never infer idempotency or replay from IR.
                state.failure_code = RuntimeFailureCode.UNKNOWN_EXTERNAL_OUTCOME.value
                state.status = RuntimeOutcomeStatus.UNKNOWN_EXTERNAL_OUTCOME
                state.trace(
                    TraceEventKind.TERMINAL,
                    step_id=saved.started_step_id,
                    attempt=saved.attempt,
                    reason_code=state.failure_code,
                    detail="recovered started step without an observed external result",
                )
            elif saved.phase is CheckpointPhase.ROLLING_BACK:
                # Even a pre-dispatch rollback checkpoint is never replayed:
                # the recovery process cannot prove that the external
                # compensation boundary was not crossed after persistence.
                state.failure_code = RuntimeFailureCode.ROLLBACK_FAILED.value
                state.status = RuntimeOutcomeStatus.FAILED
                state.trace(
                    TraceEventKind.TERMINAL,
                    step_id=saved.started_step_id or saved.current_node_id,
                    reason_code=state.failure_code,
                    detail="rollback recovery refused ambiguous compensation replay",
                )
            elif saved.phase is CheckpointPhase.TERMINAL:
                recovered_terminal_status = saved.status
                if saved.status is RuntimeOutcomeStatus.SUCCEEDED:
                    self._prepare_terminal_success_recovery(state)
            elif (
                saved.phase is CheckpointPhase.STEP_OBSERVED
                and saved.trace_entries[-1].event is TraceEventKind.ROLLBACK_OBSERVED
            ):
                recovered_rollback_observation = True
                state.status = RuntimeOutcomeStatus.ROLLED_BACK
                state.trace(
                    TraceEventKind.TERMINAL,
                    reason_code=state.failure_code,
                    detail="recovered independently admitted rollback observation",
                )
        else:
            state = _RunState(
                invocation_cid=invocation_cid,
                procedure_cid=procedure_cid,
                idempotency_key=idempotency_key,
                current_node_id=procedure.entry_step_id,
                variables=self._initial_variables(procedure, invocation),
            )
            state.evidence_cids.update(certificate_decision.receipt_cids)
            state.evidence_cids.update(compatibility.receipt_cids)
            state.trace(
                TraceEventKind.INVOCATION_ADMITTED,
                evidence_cids=tuple(
                    sorted(set(certificate_decision.receipt_cids) | set(compatibility.receipt_cids))
                ),
            )

        isolation_request = IsolationReservationRequest(
            invocation_cid=invocation_cid,
            procedure_cid=procedure_cid,
            repository_id=runtime.repository_id,
            tree_id=runtime.tree_id,
            lease_id=invocation.lease_id,
            fencing_token=invocation.fencing_token,
            scope_paths=tuple(invocation.requested_scope),
            worktree_required=effectful,
            read_only=not effectful,
        )
        isolation_reservation: IsolationReservation | None = None
        request = self._budget_request(procedure, invocation_cid, procedure_cid)
        reservation: BudgetReservation | None = None
        try:
            try:
                isolation_reservation = self._isolation.acquire(isolation_request)
                self._validate_isolation_reservation(isolation_request, isolation_reservation)
            except Exception as exc:
                return self._fail_without_dispatch(
                    procedure,
                    invocation,
                    certificate,  # type: ignore[arg-type]
                    state,
                    runtime,
                    RuntimeFailureCode.ISOLATION_ACQUISITION_FAILED,
                    "lease/worktree acquisition failed: {}".format(exc),
                    resumed=resumed,
                )
            state.evidence_cids.add(isolation_reservation.receipt_cid)
            state.trace(
                TraceEventKind.ISOLATION_ACQUIRED,
                evidence_cids=(isolation_reservation.receipt_cid,),
                detail=isolation_reservation.worktree_id,
            )
            try:
                reservation = self._budget_reservations.reserve(request)
            except Exception as exc:
                return self._fail_without_dispatch(
                    procedure,
                    invocation,
                    certificate,  # type: ignore[arg-type]
                    state,
                    runtime,
                    RuntimeFailureCode.RESOURCE_RESERVATION_FAILED,
                    "resource reservation failed: {}".format(exc),
                    resumed=resumed,
                )
            self._validate_reservation(request, reservation)
            state.evidence_cids.add(reservation.receipt_cid)
            state.trace(
                TraceEventKind.RESOURCE_RESERVED,
                evidence_cids=(reservation.receipt_cid,),
            )

            # Every continuation re-admits current authority and every
            # precondition.  A READY or TERMINAL checkpoint is historical
            # evidence, never proof that current gates still hold.
            authority_subject = _first(procedure, ("authority", "authority_envelope"), ())
            authority = self._admit(
                self._admissions.authority,
                AdmissionRequest(
                    kind=AdmissionKind.AUTHORITY,
                    subject=authority_subject,
                    procedure=procedure,
                    invocation=invocation,
                    certificate=certificate,  # type: ignore[arg-type]
                    runtime=runtime,
                    mode=mode,
                    variables=MappingProxyType(dict(state.variables)),
                    evidence_cids=tuple(_as_tuple(_get(invocation, "authority_receipt_cids", ()))),
                ),
                expected=(AdmissionKind.AUTHORITY,),
                failure=RuntimeFailureCode.AUTHORITY_REJECTED,
            )
            state.evidence_cids.update(authority.receipt_cids)
            for condition in procedure.preconditions:
                decision = self._admit_predicate(
                    AdmissionKind.PRECONDITION,
                    condition,
                    procedure,
                    invocation,
                    certificate,  # type: ignore[arg-type]
                    runtime,
                    mode,
                    state,
                    failure=RuntimeFailureCode.PRECONDITION_FAILED,
                )
                state.evidence_cids.update(decision.receipt_cids)
                state.trace(
                    TraceEventKind.PRECONDITION_ADMITTED,
                    evidence_cids=decision.receipt_cids,
                    detail=_identifier(condition, "condition_id"),
                )
            if (
                recovery_phase
                in {
                    None,
                    CheckpointPhase.READY,
                    CheckpointPhase.STEP_OBSERVED,
                }
                and not state.failure_code
            ):
                self._save(state, CheckpointPhase.READY)

            if recovery_phase in {
                CheckpointPhase.STEP_STARTED,
                CheckpointPhase.ROLLING_BACK,
            }:
                execution = self._finish(
                    procedure,
                    invocation,
                    certificate,  # type: ignore[arg-type]
                    state,
                    runtime,
                    resumed=True,
                )
            elif (
                recovery_phase is CheckpointPhase.TERMINAL
                and recovered_terminal_status is not RuntimeOutcomeStatus.SUCCEEDED
            ):
                execution = self._finish(
                    procedure,
                    invocation,
                    certificate,  # type: ignore[arg-type]
                    state,
                    runtime,
                    resumed=True,
                )
            elif recovered_rollback_observation:
                execution = self._finish(
                    procedure,
                    invocation,
                    certificate,  # type: ignore[arg-type]
                    state,
                    runtime,
                    resumed=True,
                )
            elif recovery_phase is CheckpointPhase.STEP_OBSERVED and state.failure_code:
                observed_step = steps[state.started_step_id]
                failure_code = (
                    RuntimeFailureCode(state.failure_code)
                    if state.failure_code in {item.value for item in RuntimeFailureCode}
                    else RuntimeFailureCode.OPERATION_FAILED
                )
                execution = self._terminal_failure(
                    procedure,
                    invocation,
                    certificate,  # type: ignore[arg-type]
                    state,
                    runtime,
                    failure_code,
                    "recovered an observed failed operation without replay",
                    resumed=True,
                    steps=steps,
                    operations=operations,
                    mode=mode,
                    rollback_target=(
                        observed_step.failure_target
                        if _enum_value(observed_step.failure_transition) == "rollback"
                        else ""
                    ),
                )
            else:
                execution = self._run_graph(
                    procedure,
                    invocation,
                    certificate,  # type: ignore[arg-type]
                    runtime,
                    mode,
                    reservation,
                    state,
                    steps,
                    branches,
                    loops,
                    observations,
                    operations,
                    resumed=resumed,
                )
            return self._compensate_and_reseal(
                execution,
                procedure,
                invocation,
                certificate,  # type: ignore[arg-type]
                state,
                runtime,
                isolation_reservation,
                resumed=resumed,
            )
        finally:
            cleanup_error: ProcedureInterpreterError | None = None
            if reservation is not None:
                try:
                    self._budget_reservations.release(reservation, consumed=state.cost)
                except Exception as exc:
                    cleanup_error = ProcedureInterpreterError(
                        "budget reservation release failed: {}".format(exc),
                        reason_code=RuntimeFailureCode.RESOURCE_RESERVATION_FAILED.value,
                    )
            if isolation_reservation is not None:
                try:
                    self._isolation.release(isolation_reservation)
                except Exception as exc:
                    cleanup_error = cleanup_error or ProcedureInterpreterError(
                        "lease/worktree release or compensation failed: {}".format(exc),
                        reason_code=RuntimeFailureCode.ISOLATION_RELEASE_FAILED.value,
                    )
            if cleanup_error is not None:
                raise cleanup_error

    invoke = execute

    def _validate_exact_bindings(
        self,
        procedure: ProcedureSpec,
        invocation: ProcedureInvocation,
        certificate: ProcedureCertificate | None,
        runtime: RuntimeIdentity,
    ) -> None:
        if certificate is None:
            raise ProcedureInterpreterError(
                "an externally admitted certificate is required",
                reason_code=RuntimeFailureCode.CERTIFICATE_REQUIRED.value,
            )
        if not isinstance(certificate, ProcedureCertificate):
            raise ProcedureInterpreterError(
                "certificate must be a ProcedureCertificate",
                reason_code=RuntimeFailureCode.CERTIFICATE_REJECTED.value,
            )
        for label, binding in (
            ("procedure", procedure.bindings),
            ("invocation", invocation.bindings),
            ("certificate", certificate.bindings),
        ):
            for name, current in runtime.binding_values().items():
                if _get(binding, name, None) != current:
                    raise ProcedureInterpreterError(
                        "{} {} does not match current runtime".format(label, name),
                        reason_code=RuntimeFailureCode.BINDING_MISMATCH.value,
                    )

    def _validate_identity_links(
        self,
        procedure: ProcedureSpec,
        procedure_cid: str,
        invocation: ProcedureInvocation,
        certificate: ProcedureCertificate | None,
        runtime: RuntimeIdentity,
    ) -> None:
        assert certificate is not None
        if _identifier(invocation, "procedure_cid") != procedure_cid:
            raise ProcedureInterpreterError(
                "invocation does not bind the exact procedure",
                reason_code=RuntimeFailureCode.BINDING_MISMATCH.value,
            )
        if _identifier(certificate, "procedure_cid") != procedure_cid:
            raise ProcedureInterpreterError(
                "certificate does not bind the exact procedure",
                reason_code=RuntimeFailureCode.CERTIFICATE_REJECTED.value,
            )
        certificate_cid = _artifact_id(certificate, "procedure-certificate")
        if _identifier(invocation, "certificate_cid") != certificate_cid:
            raise ProcedureInterpreterError(
                "invocation does not bind the exact certificate",
                reason_code=RuntimeFailureCode.CERTIFICATE_REJECTED.value,
            )
        if certificate.procedure_version != procedure.version:
            raise ProcedureInterpreterError(
                "certificate procedure version differs from ProcedureIR",
                reason_code=RuntimeFailureCode.CERTIFICATE_REJECTED.value,
            )
        if certificate.task_family_cid != procedure.task_family_id:
            raise ProcedureInterpreterError(
                "certificate task family differs from ProcedureIR",
                reason_code=RuntimeFailureCode.CERTIFICATE_REJECTED.value,
            )
        invocation_registry = _identifier(invocation, "registry_revision")
        if invocation_registry != runtime.registry_revision:
            raise ProcedureInterpreterError(
                "invocation registry revision is stale",
                reason_code=RuntimeFailureCode.REGISTRY_DRIFT.value,
            )
        certificate_catalog = _identifier(
            certificate, "operation_catalog_revision", "operation_catalog_cid"
        )
        if certificate_catalog != runtime.operation_catalog_revision:
            raise ProcedureInterpreterError(
                "certificate operation catalog revision is stale",
                reason_code=RuntimeFailureCode.OPERATION_CONTRACT_DRIFT.value,
            )
        if self._catalog.revision != runtime.operation_catalog_revision:
            raise ProcedureInterpreterError(
                "injected catalog does not match current runtime identity",
                reason_code=RuntimeFailureCode.OPERATION_CONTRACT_DRIFT.value,
            )

    def _validate_lifecycle(
        self,
        procedure: ProcedureSpec,
        certificate: ProcedureCertificate | None,
        mode: ExecutionMode,
        effectful: bool,
        runtime: RuntimeIdentity,
    ) -> None:
        assert certificate is not None
        procedure_state = _enum_value(_get(procedure, "state", ""))
        certificate_state = _enum_value(_get(certificate, "state", ""))
        forbidden = {"stale", "revoked", "superseded", "rejected", "degraded"}
        if procedure_state in forbidden or certificate_state in forbidden:
            reason = (
                RuntimeFailureCode.REVOKED_PROCEDURE
                if "revoked" in {procedure_state, certificate_state}
                else RuntimeFailureCode.STALE_CERTIFICATE
            )
            raise ProcedureInterpreterError(
                "procedure or certificate is not current", reason_code=reason.value
            )
        if mode is ExecutionMode.LIVE and certificate_state not in {"verified", "promoted"}:
            raise ProcedureInterpreterError(
                "live execution requires a current externally verified certificate",
                reason_code=RuntimeFailureCode.CERTIFICATE_REJECTED.value,
            )
        if effectful and mode is not ExecutionMode.LIVE:
            raise ProcedureInterpreterError(
                "test and shadow execution are strictly read-only",
                reason_code=RuntimeFailureCode.AUTHORITY_REJECTED.value,
            )
        risk_order = {
            RiskClass.OBSERVATION_ONLY.value: 0,
            RiskClass.REVERSIBLE_LOCAL.value: 1,
            RiskClass.REPOSITORY_WRITE.value: 2,
            RiskClass.PUBLIC_CONTRACT.value: 3,
            RiskClass.AUTHORITY_OR_SECURITY.value: 4,
        }
        procedure_risk = _enum_value(procedure.authority.risk_ceiling)
        certificate_risk = _enum_value(certificate.risk_ceiling)
        if (
            procedure_risk not in risk_order
            or certificate_risk not in risk_order
            or risk_order[certificate_risk] < risk_order[procedure_risk]
        ):
            raise ProcedureInterpreterError(
                "certificate risk ceiling is below ProcedureIR authority risk",
                reason_code=RuntimeFailureCode.CERTIFICATE_REJECTED.value,
            )
        expiry = _first(certificate, ("expires_at_ms", "expiry_ms", "review_horizon_ms"), 0)
        if isinstance(expiry, int) and expiry and expiry < runtime.now_ms:
            raise ProcedureInterpreterError(
                "procedure certificate expired",
                reason_code=RuntimeFailureCode.STALE_CERTIFICATE.value,
            )

    def _validate_scope(self, procedure: ProcedureSpec, invocation: ProcedureInvocation) -> None:
        declared = tuple(
            _path(item, field_name="procedure scope") for item in procedure.scope_paths
        )
        requested = tuple(
            _path(item, field_name="invocation scope")
            for item in _as_tuple(_get(invocation, "requested_scope", ()))
        )
        if not declared:
            raise ProcedureInterpreterError(
                "procedure has no declared scope",
                reason_code=RuntimeFailureCode.SCOPE_ESCAPE.value,
            )
        if not requested:
            raise ProcedureInterpreterError(
                "invocation has no exact requested scope",
                reason_code=RuntimeFailureCode.SCOPE_ESCAPE.value,
            )
        if any(not _path_is_within(item, declared) for item in requested):
            raise ProcedureInterpreterError(
                "invocation broadens procedure scope",
                reason_code=RuntimeFailureCode.SCOPE_ESCAPE.value,
            )

    def _validate_checkpoint(
        self,
        checkpoint: InterpreterCheckpoint,
        procedure: ProcedureSpec,
        invocation: ProcedureInvocation,
        procedure_cid: str,
        invocation_cid: str,
        idempotency_key: str,
    ) -> None:
        if not isinstance(checkpoint, InterpreterCheckpoint):
            raise ProcedureInterpreterError(
                "checkpoint has an unsupported type",
                reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
            )
        if (
            checkpoint.procedure_cid != procedure_cid
            or checkpoint.invocation_cid != invocation_cid
            or checkpoint.idempotency_key != idempotency_key
        ):
            raise ProcedureInterpreterError(
                "checkpoint identity does not match invocation",
                reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
            )
        if not isinstance(checkpoint.phase, CheckpointPhase) or not isinstance(
            checkpoint.status, RuntimeOutcomeStatus
        ):
            raise ProcedureInterpreterError(
                "checkpoint uses an unknown phase or status",
                reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
            )
        steps = {item.step_id: item for item in procedure.steps}
        branches = {item.branch_id: item for item in procedure.branches}
        loops = {item.loop_id: item for item in procedure.loops}
        observations = {item.observation_id: item for item in procedure.observations}
        nodes = set(steps) | set(branches) | set(loops)
        if checkpoint.current_node_id and checkpoint.current_node_id not in nodes:
            raise ProcedureInterpreterError(
                "checkpoint control-flow node is not in ProcedureIR",
                reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
            )
        if not set(checkpoint.executed_step_ids).issubset(steps):
            raise ProcedureInterpreterError(
                "checkpoint claims unknown executed steps",
                reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
            )
        if not set(checkpoint.admitted_observation_ids).issubset(observations):
            raise ProcedureInterpreterError(
                "checkpoint claims unknown observations",
                reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
            )
        for loop_id, count in checkpoint.loop_counts.items():
            if (
                loop_id not in loops
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                or count > loops[loop_id].max_iterations
            ):
                raise ProcedureInterpreterError(
                    "checkpoint loop state violates ProcedureIR bounds",
                    reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                )
        self._validate_checkpoint_variables(checkpoint, procedure, invocation)
        requested_scope = tuple(invocation.requested_scope)
        for changed_path in checkpoint.changed_paths:
            try:
                if not _path_is_within(changed_path, requested_scope):
                    raise ProcedureInterpreterError("changed path is outside invocation scope")
            except ProcedureInterpreterError as exc:
                raise ProcedureInterpreterError(
                    "checkpoint contains an out-of-scope changed path",
                    reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                ) from exc

        if len(checkpoint.trace_entries) > MAX_RUNTIME_STEPS * 4:
            raise ProcedureInterpreterError(
                "checkpoint trace exceeds its runtime bound",
                reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
            )
        for sequence, entry in enumerate(checkpoint.trace_entries):
            if not isinstance(entry, RuntimeTraceEntry) or entry.sequence != sequence:
                raise ProcedureInterpreterError(
                    "checkpoint trace is not contiguous and typed",
                    reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                )
            if not isinstance(entry.event, TraceEventKind):
                raise ProcedureInterpreterError(
                    "checkpoint trace contains an unknown event",
                    reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                )
            if entry.step_id and entry.operation:
                step = steps.get(entry.step_id)
                if step is None or _enum_value(step.operation) != entry.operation:
                    raise ProcedureInterpreterError(
                        "checkpoint trace operation does not match ProcedureIR",
                        reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                    )
                if not set(entry.observed_effect_ids).issubset(set(step.declared_effect_ids)):
                    raise ProcedureInterpreterError(
                        "checkpoint trace claims undeclared effects",
                        reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                    )

        last = checkpoint.trace_entries[-1] if checkpoint.trace_entries else None
        nonterminal = checkpoint.phase is not CheckpointPhase.TERMINAL
        if nonterminal and (
            checkpoint.status is not RuntimeOutcomeStatus.INCOMPLETE or checkpoint.terminal_at_ms
        ):
            raise ProcedureInterpreterError(
                "nonterminal checkpoint claims a terminal outcome",
                reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
            )
        if checkpoint.phase is CheckpointPhase.STEP_STARTED:
            if (
                checkpoint.started_step_id not in steps
                or checkpoint.current_node_id != checkpoint.started_step_id
                or checkpoint.attempt <= 0
                or last is None
                or last.event is not TraceEventKind.STEP_STARTED
                or last.step_id != checkpoint.started_step_id
                or last.attempt != checkpoint.attempt
            ):
                raise ProcedureInterpreterError(
                    "started checkpoint is not bound to its dispatch trace",
                    reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                )
        elif checkpoint.phase is CheckpointPhase.STEP_OBSERVED:
            if last is None or last.event not in {
                TraceEventKind.STEP_OBSERVED,
                TraceEventKind.ROLLBACK_OBSERVED,
            }:
                raise ProcedureInterpreterError(
                    "observed checkpoint lacks its terminal operation observation",
                    reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                )
            if last.event is TraceEventKind.ROLLBACK_OBSERVED:
                if (
                    not checkpoint.failure_code
                    or checkpoint.started_step_id
                    or not checkpoint.rollback_receipt_cids
                ):
                    raise ProcedureInterpreterError(
                        "rollback observation checkpoint lacks admitted compensation",
                        reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                    )
            elif checkpoint.failure_code:
                if (
                    checkpoint.started_step_id != last.step_id
                    or last.step_id in checkpoint.executed_step_ids
                ):
                    raise ProcedureInterpreterError(
                        "failed observed checkpoint has inconsistent execution state",
                        reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                    )
            elif (
                checkpoint.started_step_id
                or last.step_id not in checkpoint.executed_step_ids
                or checkpoint.current_node_id == last.step_id
            ):
                raise ProcedureInterpreterError(
                    "successful observed checkpoint did not durably advance",
                    reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                )
        elif checkpoint.phase is CheckpointPhase.ROLLING_BACK:
            if (
                not checkpoint.failure_code
                or last is None
                or last.event is not TraceEventKind.ROLLBACK_STARTED
            ):
                raise ProcedureInterpreterError(
                    "rollback checkpoint lacks a typed failure transition",
                    reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                )
        elif checkpoint.phase is CheckpointPhase.READY:
            if checkpoint.started_step_id or checkpoint.failure_code:
                raise ProcedureInterpreterError(
                    "ready checkpoint retains in-flight operation state",
                    reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                )
        elif checkpoint.phase is CheckpointPhase.TERMINAL:
            if (
                checkpoint.status is RuntimeOutcomeStatus.INCOMPLETE
                or checkpoint.terminal_at_ms <= 0
                or last is None
                or last.event is not TraceEventKind.TERMINAL
            ):
                raise ProcedureInterpreterError(
                    "terminal checkpoint is structurally incomplete",
                    reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                )
            if checkpoint.status is RuntimeOutcomeStatus.SUCCEEDED:
                required_posts = {item.condition_id for item in procedure.postconditions}
                if (
                    checkpoint.current_node_id
                    or checkpoint.failure_code
                    or not checkpoint.validation_receipt_cids
                    or set(checkpoint.satisfied_postcondition_ids) != required_posts
                    or not set(procedure.validation.required_step_ids).issubset(
                        checkpoint.executed_step_ids
                    )
                    or not set(procedure.validation.required_observation_ids).issubset(
                        checkpoint.admitted_observation_ids
                    )
                ):
                    raise ProcedureInterpreterError(
                        "successful checkpoint does not establish structural hard gates",
                        reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                    )
            elif not checkpoint.failure_code:
                raise ProcedureInterpreterError(
                    "terminal non-success checkpoint lacks a failure code",
                    reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                )
        _canonical_bytes(checkpoint.to_dict())

    def _validate_checkpoint_variables(
        self,
        checkpoint: InterpreterCheckpoint,
        procedure: ProcedureSpec,
        invocation: ProcedureInvocation,
    ) -> None:
        declarations = {
            item.name: (item.value_type, item.allowed_values, not item.required)
            for item in procedure.parameters
        }
        declarations.update({item.name: (item.value_type, (), True) for item in procedure.locals})
        if set(checkpoint.variables) != set(declarations):
            raise ProcedureInterpreterError(
                "checkpoint variable set differs from ProcedureIR",
                reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
            )
        for name, (value_type, allowed, allow_none) in declarations.items():
            try:
                _validate_typed_value(
                    checkpoint.variables[name],
                    value_type,
                    field_name=f"checkpoint variable {name}",
                    scope_paths=invocation.requested_scope,
                    allowed_values=allowed,
                    allow_none=allow_none,
                )
            except ProcedureInterpreterError as exc:
                raise ProcedureInterpreterError(
                    "checkpoint contains an invalid typed variable",
                    reason_code=RuntimeFailureCode.CHECKPOINT_INVALID.value,
                ) from exc

    def _validate_operation_value_contracts(
        self,
        procedure: ProcedureSpec,
        steps: Mapping[str, Any],
        operations: Mapping[str, TrustedOperation],
    ) -> None:
        declarations = {
            item.name: item.value_type for item in (*procedure.parameters, *procedure.locals)
        }
        local_types = {item.name: item.value_type for item in procedure.locals}
        for step_id, step in steps.items():
            operation = operations[step_id]
            input_bindings = dict(step.input_bindings)
            output_bindings = dict(step.output_bindings)
            if set(operation.input_types) != set(input_bindings):
                raise ProcedureInterpreterError(
                    "trusted operation input type contract is not exact",
                    reason_code=RuntimeFailureCode.OPERATION_CONTRACT_DRIFT.value,
                )
            if set(operation.output_types) != set(output_bindings):
                raise ProcedureInterpreterError(
                    "trusted operation output type contract is not exact",
                    reason_code=RuntimeFailureCode.OPERATION_CONTRACT_DRIFT.value,
                )
            for input_name, source in input_bindings.items():
                variable_name = source
                for prefix in ("parameter:", "param:", "local:"):
                    if source.startswith(prefix):
                        variable_name = source[len(prefix) :]
                        break
                if source.startswith(("literal:", "binding:")):
                    continue
                if declarations.get(variable_name) != operation.input_types[input_name]:
                    raise ProcedureInterpreterError(
                        "trusted operation input type differs from ProcedureIR binding",
                        reason_code=RuntimeFailureCode.OPERATION_CONTRACT_DRIFT.value,
                    )
            for output_name, target in output_bindings.items():
                variable_name = target[len("local:") :] if target.startswith("local:") else target
                if local_types.get(variable_name) != operation.output_types[output_name]:
                    raise ProcedureInterpreterError(
                        "trusted operation output type differs from ProcedureIR local",
                        reason_code=RuntimeFailureCode.OPERATION_CONTRACT_DRIFT.value,
                    )
                if operation.output_types[output_name] is ValueType.ENUM:
                    raise ProcedureInterpreterError(
                        "enum local output lacks a closed ProcedureIR value set",
                        reason_code=RuntimeFailureCode.OPERATION_CONTRACT_DRIFT.value,
                    )

    def _initial_variables(
        self, procedure: ProcedureSpec, invocation: ProcedureInvocation
    ) -> dict[str, Any]:
        supplied = _get(invocation, "parameters", {})
        if not isinstance(supplied, Mapping):
            raise ProcedureInterpreterError(
                "invocation parameters must be a mapping",
                reason_code=RuntimeFailureCode.INVALID_INVOCATION.value,
            )
        result = dict(_canonical(supplied))
        declared = {item.name: item for item in procedure.parameters}
        unknown = set(result).difference(declared)
        if unknown:
            raise ProcedureInterpreterError(
                "invocation has undeclared parameters: {}".format(sorted(unknown)),
                reason_code=RuntimeFailureCode.INVALID_INVOCATION.value,
            )
        for name, parameter in declared.items():
            if name not in result:
                default = _get(parameter, "default_value", None)
                if default is None and bool(_get(parameter, "required", True)):
                    raise ProcedureInterpreterError(
                        "required parameter {} is missing".format(name),
                        reason_code=RuntimeFailureCode.INVALID_INVOCATION.value,
                    )
                result[name] = _canonical(default)
            allowed = tuple(_as_tuple(_get(parameter, "allowed_values", ())))
            if allowed and result[name] not in allowed:
                raise ProcedureInterpreterError(
                    "parameter {} is outside its closed value set".format(name),
                    reason_code=RuntimeFailureCode.INVALID_INVOCATION.value,
                )
            result[name] = _validate_typed_value(
                result[name],
                parameter.value_type,
                field_name=f"parameter {name}",
                scope_paths=tuple(invocation.requested_scope),
                allowed_values=allowed,
                allow_none=not parameter.required,
            )
        for local in procedure.locals:
            result[local.name] = None
        return result

    def _admit(
        self,
        producer: AdmissionProducer,
        request: AdmissionRequest,
        *,
        expected: Sequence[AdmissionKind],
        failure: RuntimeFailureCode,
        require_true_predicate: bool = False,
    ) -> AdmissionDecision:
        try:
            decision = producer.admit(request)
        except Exception as exc:
            raise ProcedureInterpreterError(
                "independent admission producer failed: {}".format(exc),
                reason_code=failure.value,
            ) from exc
        if not isinstance(decision, AdmissionDecision):
            raise ProcedureInterpreterError(
                "admission producer returned an untyped decision",
                reason_code=failure.value,
            )
        if decision.kind not in tuple(expected) or not decision.admitted:
            raise ProcedureInterpreterError(
                "independent admission rejected {}".format(request.kind.value),
                reason_code=decision.reason_code or failure.value,
            )
        if require_true_predicate and decision.predicate_value is not True:
            raise ProcedureInterpreterError(
                "independently evaluated predicate does not hold",
                reason_code=decision.reason_code or failure.value,
            )
        return decision

    def _admit_predicate(
        self,
        kind: AdmissionKind,
        subject: Any,
        procedure: ProcedureSpec,
        invocation: ProcedureInvocation,
        certificate: ProcedureCertificate,
        runtime: RuntimeIdentity,
        mode: ExecutionMode,
        state: _RunState,
        *,
        failure: RuntimeFailureCode,
        step_id: str = "",
        operation_result: OperationResult | None = None,
        require_true: bool = True,
    ) -> AdmissionDecision:
        return self._admit(
            self._admissions.evidence,
            AdmissionRequest(
                kind=kind,
                subject=subject,
                procedure=procedure,
                invocation=invocation,
                certificate=certificate,
                runtime=runtime,
                mode=mode,
                step_id=step_id,
                variables=MappingProxyType(dict(state.variables)),
                evidence_cids=tuple(sorted(state.evidence_cids)),
                operation_result=operation_result,
            ),
            expected=(kind,),
            failure=failure,
            require_true_predicate=require_true,
        )

    def _budget_request(
        self, procedure: ProcedureSpec, invocation_cid: str, procedure_cid: str
    ) -> BudgetReservationRequest:
        envelope = _first(procedure, ("resources", "resource_envelope"), None)
        token_limit = int(
            _first(
                envelope,
                (
                    "model_token_limit",
                    "max_tokens",
                    "token_limit",
                    "token_budget",
                    "maximum_tokens",
                ),
                0,
            )
            or 0
        )
        resource_limit = int(
            _first(
                envelope,
                (
                    "cpu_time_ms",
                    "max_resource_units",
                    "resource_limit",
                    "maximum_resource_units",
                ),
                0,
            )
            or 0
        )
        time_limit = int(
            _first(
                envelope,
                (
                    "wall_time_ms",
                    "max_wall_time_ms",
                    "time_limit_ms",
                    "maximum_wall_time_ms",
                ),
                0,
            )
            or 0
        )
        return BudgetReservationRequest(
            invocation_cid=invocation_cid,
            procedure_cid=procedure_cid,
            token_limit=token_limit,
            resource_limit=resource_limit,
            time_limit_ms=time_limit,
            envelope=envelope,
        )

    def _validate_reservation(
        self, request: BudgetReservationRequest, reservation: BudgetReservation
    ) -> None:
        if not isinstance(reservation, BudgetReservation):
            raise ProcedureInterpreterError(
                "reservation port returned an untyped value",
                reason_code=RuntimeFailureCode.RESOURCE_RESERVATION_FAILED.value,
            )
        for requested, admitted in (
            (request.token_limit, reservation.token_limit),
            (request.resource_limit, reservation.resource_limit),
            (request.time_limit_ms, reservation.time_limit_ms),
        ):
            if requested > admitted:
                raise ProcedureInterpreterError(
                    "reservation is smaller than procedure envelope",
                    reason_code=RuntimeFailureCode.RESOURCE_RESERVATION_FAILED.value,
                )

    def _validate_isolation_reservation(
        self,
        request: IsolationReservationRequest,
        reservation: IsolationReservation,
    ) -> None:
        if not isinstance(reservation, IsolationReservation):
            raise ProcedureInterpreterError(
                "isolation port returned an untyped reservation",
                reason_code=RuntimeFailureCode.ISOLATION_ACQUISITION_FAILED.value,
            )
        if (
            reservation.lease_id != request.lease_id
            or reservation.fencing_token != request.fencing_token
            or reservation.scope_paths != request.scope_paths
            or reservation.read_only != request.read_only
        ):
            raise ProcedureInterpreterError(
                "isolation reservation does not exactly bind invocation",
                reason_code=RuntimeFailureCode.ISOLATION_ACQUISITION_FAILED.value,
            )
        if request.worktree_required and not reservation.worktree_id:
            raise ProcedureInterpreterError(
                "effectful invocation did not acquire an isolated worktree",
                reason_code=RuntimeFailureCode.ISOLATION_ACQUISITION_FAILED.value,
            )

    def _save(self, state: _RunState, phase: CheckpointPhase) -> InterpreterCheckpoint:
        checkpoint = state.checkpoint(phase)
        self._checkpoints.save(checkpoint)
        return checkpoint

    def _prepare_terminal_success_recovery(self, state: _RunState) -> None:
        """Discard historical completion claims before current requalification."""

        while state.trace_entries and state.trace_entries[-1].event in {
            TraceEventKind.TERMINAL,
            TraceEventKind.POSTCONDITION_ADMITTED,
            TraceEventKind.VALIDATION_ADMITTED,
        }:
            state.trace_entries.pop()
        state.validation_receipt_cids.clear()
        state.satisfied_postcondition_ids.clear()
        state.status = RuntimeOutcomeStatus.INCOMPLETE
        state.failure_code = ""
        state.terminal_at_ms = 0
        state.started_step_id = ""
        state.attempt = 0
        state.current_node_id = ""

    def _compensate_and_reseal(
        self,
        execution: ProcedureExecution,
        procedure: ProcedureSpec,
        invocation: ProcedureInvocation,
        certificate: ProcedureCertificate,
        state: _RunState,
        runtime: RuntimeIdentity,
        isolation: IsolationReservation,
        *,
        resumed: bool,
    ) -> ProcedureExecution:
        """Compensate known mutation before exposing the final receipt."""

        if not state.changed_paths or state.status in {
            RuntimeOutcomeStatus.SUCCEEDED,
            RuntimeOutcomeStatus.ROLLED_BACK,
        }:
            return execution
        try:
            compensation_cid = self._isolation.compensate(
                isolation,
                reason_code=state.failure_code or RuntimeFailureCode.INTERNAL_ERROR.value,
            )
        except Exception as exc:
            raise ProcedureInterpreterError(
                f"lease/worktree compensation failed: {exc}",
                reason_code=RuntimeFailureCode.ISOLATION_RELEASE_FAILED.value,
            ) from exc
        if not isinstance(compensation_cid, str) or not compensation_cid:
            raise ProcedureInterpreterError(
                "isolation compensation returned no receipt",
                reason_code=RuntimeFailureCode.ISOLATION_RELEASE_FAILED.value,
            )
        state.rollback_receipt_cids.add(compensation_cid)
        state.evidence_cids.add(compensation_cid)
        state.status = RuntimeOutcomeStatus.ROLLED_BACK
        state.trace(
            TraceEventKind.ROLLBACK_OBSERVED,
            step_id=state.started_step_id or state.current_node_id or procedure.entry_step_id,
            evidence_cids=(compensation_cid,),
            reason_code=state.failure_code,
            detail="external isolation owner admitted worktree compensation",
        )
        state.trace(
            TraceEventKind.TERMINAL,
            step_id=state.started_step_id or state.current_node_id,
            reason_code=state.failure_code,
            detail="known changed paths compensated before receipt emission",
        )
        return self._finish(
            procedure,
            invocation,
            certificate,
            state,
            runtime,
            resumed=resumed,
        )

    def _run_graph(
        self,
        procedure: ProcedureSpec,
        invocation: ProcedureInvocation,
        certificate: ProcedureCertificate,
        runtime: RuntimeIdentity,
        mode: ExecutionMode,
        reservation: BudgetReservation,
        state: _RunState,
        steps: Mapping[str, Any],
        branches: Mapping[str, Any],
        loops: Mapping[str, Any],
        observations: Mapping[str, Any],
        operations: Mapping[str, TrustedOperation],
        *,
        resumed: bool,
    ) -> ProcedureExecution:
        start_ms = self._clock.monotonic_ms()
        transitions = 0
        fallback_visits: dict[str, int] = {}

        while state.current_node_id:
            transitions += 1
            if transitions > min(MAX_RUNTIME_STEPS, max(1, len(steps) * 64 + 128)):
                return self._terminal_failure(
                    procedure,
                    invocation,
                    certificate,
                    state,
                    runtime,
                    RuntimeFailureCode.CONTROL_FLOW_BOUNDS,
                    "procedure exceeded the runtime transition bound",
                    resumed=resumed,
                    steps=steps,
                    operations=operations,
                    mode=mode,
                )
            budget_failure = self._budget_failure(state, reservation, start_ms)
            if budget_failure is not None:
                return self._terminal_failure(
                    procedure,
                    invocation,
                    certificate,
                    state,
                    runtime,
                    budget_failure,
                    "procedure exhausted its admitted budget",
                    resumed=resumed,
                    steps=steps,
                    operations=operations,
                    mode=mode,
                )

            node_id = state.current_node_id
            if node_id in branches:
                branch = branches[node_id]
                observation_id = _identifier(branch, "observation_id")
                observation = observations.get(observation_id)
                if observation is None:
                    return self._terminal_failure(
                        procedure,
                        invocation,
                        certificate,
                        state,
                        runtime,
                        RuntimeFailureCode.OBSERVATION_FAILED,
                        "branch observation is missing",
                        resumed=resumed,
                        steps=steps,
                        operations=operations,
                        mode=mode,
                    )
                try:
                    decision = self._admit_predicate(
                        AdmissionKind.OBSERVATION,
                        observation,
                        procedure,
                        invocation,
                        certificate,
                        runtime,
                        mode,
                        state,
                        failure=RuntimeFailureCode.OBSERVATION_FAILED,
                        require_true=False,
                    )
                except ProcedureInterpreterError as exc:
                    return self._terminal_failure(
                        procedure,
                        invocation,
                        certificate,
                        state,
                        runtime,
                        RuntimeFailureCode.OBSERVATION_FAILED,
                        str(exc),
                        resumed=resumed,
                        steps=steps,
                        operations=operations,
                        mode=mode,
                    )
                if decision.predicate_value is None:
                    return self._terminal_failure(
                        procedure,
                        invocation,
                        certificate,
                        state,
                        runtime,
                        RuntimeFailureCode.OBSERVATION_FAILED,
                        "branch producer did not return a typed boolean observation",
                        resumed=resumed,
                        steps=steps,
                        operations=operations,
                        mode=mode,
                    )
                state.evidence_cids.update(decision.receipt_cids)
                state.admitted_observation_ids.add(observation_id)
                state.current_node_id = (
                    branch.true_step_id if decision.predicate_value else branch.false_step_id
                )
                state.trace(
                    TraceEventKind.BRANCH_SELECTED,
                    step_id=node_id,
                    evidence_cids=decision.receipt_cids,
                    detail="true" if decision.predicate_value else "false",
                )
                self._save(state, CheckpointPhase.READY)
                continue

            if node_id in loops:
                loop = loops[node_id]
                observation_id = _identifier(loop, "condition_observation_id")
                observation = observations.get(observation_id)
                if observation is None:
                    return self._terminal_failure(
                        procedure,
                        invocation,
                        certificate,
                        state,
                        runtime,
                        RuntimeFailureCode.OBSERVATION_FAILED,
                        "loop condition observation is missing",
                        resumed=resumed,
                        steps=steps,
                        operations=operations,
                        mode=mode,
                    )
                try:
                    decision = self._admit_predicate(
                        AdmissionKind.OBSERVATION,
                        observation,
                        procedure,
                        invocation,
                        certificate,
                        runtime,
                        mode,
                        state,
                        failure=RuntimeFailureCode.OBSERVATION_FAILED,
                        require_true=False,
                    )
                except ProcedureInterpreterError as exc:
                    return self._terminal_failure(
                        procedure,
                        invocation,
                        certificate,
                        state,
                        runtime,
                        RuntimeFailureCode.OBSERVATION_FAILED,
                        str(exc),
                        resumed=resumed,
                        steps=steps,
                        operations=operations,
                        mode=mode,
                    )
                if decision.predicate_value is None:
                    return self._terminal_failure(
                        procedure,
                        invocation,
                        certificate,
                        state,
                        runtime,
                        RuntimeFailureCode.OBSERVATION_FAILED,
                        "loop producer did not return a typed boolean observation",
                        resumed=resumed,
                        steps=steps,
                        operations=operations,
                        mode=mode,
                    )
                count = state.loop_counts.get(node_id, 0)
                if decision.predicate_value:
                    maximum = min(int(loop.max_iterations), MAX_RUNTIME_LOOP_ITERATIONS)
                    if count >= maximum:
                        return self._terminal_failure(
                            procedure,
                            invocation,
                            certificate,
                            state,
                            runtime,
                            RuntimeFailureCode.CONTROL_FLOW_BOUNDS,
                            "loop remained true at its declared bound",
                            resumed=resumed,
                            steps=steps,
                            operations=operations,
                            mode=mode,
                        )
                    state.loop_counts[node_id] = count + 1
                    state.current_node_id = loop.body_step_id
                else:
                    state.current_node_id = loop.exit_step_id
                state.evidence_cids.update(decision.receipt_cids)
                state.admitted_observation_ids.add(observation_id)
                state.trace(
                    TraceEventKind.LOOP_CHECKED,
                    step_id=node_id,
                    attempt=count + (1 if decision.predicate_value else 0),
                    evidence_cids=decision.receipt_cids,
                    detail="continue" if decision.predicate_value else "exit",
                )
                self._save(state, CheckpointPhase.READY)
                continue

            step = steps.get(node_id)
            if step is None:
                return self._terminal_failure(
                    procedure,
                    invocation,
                    certificate,
                    state,
                    runtime,
                    RuntimeFailureCode.INVALID_PROCEDURE,
                    "control flow targets an unknown node",
                    resumed=resumed,
                    steps=steps,
                    operations=operations,
                    mode=mode,
                )

            try:
                for invariant in procedure.invariants:
                    decision = self._admit_predicate(
                        AdmissionKind.INVARIANT,
                        invariant,
                        procedure,
                        invocation,
                        certificate,
                        runtime,
                        mode,
                        state,
                        failure=RuntimeFailureCode.INVARIANT_FAILED,
                        step_id=node_id,
                    )
                    state.evidence_cids.update(decision.receipt_cids)
                    state.trace(
                        TraceEventKind.INVARIANT_ADMITTED,
                        step_id=node_id,
                        evidence_cids=decision.receipt_cids,
                        detail=_identifier(invariant, "condition_id"),
                    )
            except ProcedureInterpreterError as exc:
                return self._terminal_failure(
                    procedure,
                    invocation,
                    certificate,
                    state,
                    runtime,
                    RuntimeFailureCode.INVARIANT_FAILED,
                    str(exc),
                    resumed=resumed,
                    steps=steps,
                    operations=operations,
                    mode=mode,
                )

            execution = self._execute_step(
                procedure,
                invocation,
                certificate,
                runtime,
                mode,
                reservation,
                state,
                step,
                operations[node_id],
                observations,
                start_ms,
            )
            if execution.succeeded:
                if node_id in set(procedure.terminal_step_ids):
                    state.current_node_id = ""
                else:
                    state.current_node_id = execution.next_node_id
                self._save(state, CheckpointPhase.READY)
                continue

            if execution.unknown_external_outcome:
                state.failure_code = RuntimeFailureCode.UNKNOWN_EXTERNAL_OUTCOME.value
                state.status = RuntimeOutcomeStatus.UNKNOWN_EXTERNAL_OUTCOME
                state.trace(
                    TraceEventKind.TERMINAL,
                    step_id=node_id,
                    reason_code=state.failure_code,
                )
                return self._finish(
                    procedure,
                    invocation,
                    certificate,
                    state,
                    runtime,
                    resumed=resumed,
                )

            transition = execution.failure_transition
            if transition == "fallback" and execution.failure_target:
                fallback = next(
                    (
                        item
                        for item in procedure.fallback
                        if item.fallback_id == execution.failure_target
                    ),
                    None,
                )
                count = fallback_visits.get(execution.failure_target, 0) + 1
                fallback_visits[execution.failure_target] = count
                if (
                    fallback is None
                    or execution.failure_code not in fallback.trigger_failure_codes
                    or count > fallback.maximum_uses
                ):
                    return self._terminal_failure(
                        procedure,
                        invocation,
                        certificate,
                        state,
                        runtime,
                        RuntimeFailureCode.FALLBACK_FAILED,
                        "fallback target repeated without new evidence",
                        resumed=resumed,
                        steps=steps,
                        operations=operations,
                        mode=mode,
                    )
                state.current_node_id = fallback.entry_step_id
                state.trace(
                    TraceEventKind.FALLBACK_SELECTED,
                    step_id=node_id,
                    reason_code=execution.failure_code,
                    detail=fallback.entry_step_id,
                )
                self._save(state, CheckpointPhase.READY)
                continue
            if transition == "escalate":
                state.failure_code = execution.failure_code
                state.status = RuntimeOutcomeStatus.ESCALATED
                return self._finish(
                    procedure, invocation, certificate, state, runtime, resumed=resumed
                )
            if transition == "quarantine":
                state.failure_code = execution.failure_code
                state.status = RuntimeOutcomeStatus.QUARANTINED
                return self._finish(
                    procedure, invocation, certificate, state, runtime, resumed=resumed
                )
            return self._terminal_failure(
                procedure,
                invocation,
                certificate,
                state,
                runtime,
                RuntimeFailureCode(execution.failure_code)
                if execution.failure_code in {item.value for item in RuntimeFailureCode}
                else RuntimeFailureCode.OPERATION_FAILED,
                "step failed after its bounded retry policy",
                resumed=resumed,
                steps=steps,
                operations=operations,
                mode=mode,
                rollback_target=(execution.failure_target if transition == "rollback" else ""),
            )

        # No terminal success assertion is trusted.  Every required validation
        # and postcondition is independently re-evaluated against current state.
        gate_failure = RuntimeFailureCode.VALIDATION_FAILED
        try:
            validation = procedure.validation
            missing_steps = set(validation.required_step_ids).difference(state.executed_step_ids)
            missing_observations = set(validation.required_observation_ids).difference(
                state.admitted_observation_ids
            )
            if missing_steps or missing_observations:
                raise ProcedureInterpreterError(
                    "required validation coverage is incomplete",
                    reason_code=RuntimeFailureCode.VALIDATION_FAILED.value,
                )
            decision = self._admit_predicate(
                AdmissionKind.VALIDATION,
                validation,
                procedure,
                invocation,
                certificate,
                runtime,
                mode,
                state,
                failure=RuntimeFailureCode.VALIDATION_FAILED,
            )
            state.validation_receipt_cids.update(decision.receipt_cids)
            state.evidence_cids.update(decision.receipt_cids)
            state.trace(
                TraceEventKind.VALIDATION_ADMITTED,
                evidence_cids=decision.receipt_cids,
                detail=_artifact_id(validation, "procedure-validation-plan"),
            )
            gate_failure = RuntimeFailureCode.POSTCONDITION_FAILED
            for postcondition in procedure.postconditions:
                decision = self._admit_predicate(
                    AdmissionKind.POSTCONDITION,
                    postcondition,
                    procedure,
                    invocation,
                    certificate,
                    runtime,
                    mode,
                    state,
                    failure=RuntimeFailureCode.POSTCONDITION_FAILED,
                )
                condition_id = _identifier(postcondition, "condition_id")
                state.satisfied_postcondition_ids.add(condition_id)
                state.evidence_cids.update(decision.receipt_cids)
                state.trace(
                    TraceEventKind.POSTCONDITION_ADMITTED,
                    evidence_cids=decision.receipt_cids,
                    detail=condition_id,
                )
        except ProcedureInterpreterError as exc:
            return self._terminal_failure(
                procedure,
                invocation,
                certificate,
                state,
                runtime,
                gate_failure,
                str(exc),
                resumed=resumed,
                steps=steps,
                operations=operations,
                mode=mode,
            )

        state.status = RuntimeOutcomeStatus.SUCCEEDED
        state.failure_code = ""
        state.trace(TraceEventKind.TERMINAL, detail="all external hard gates admitted")
        return self._finish(procedure, invocation, certificate, state, runtime, resumed=resumed)

    def _execute_step(
        self,
        procedure: ProcedureSpec,
        invocation: ProcedureInvocation,
        certificate: ProcedureCertificate,
        runtime: RuntimeIdentity,
        mode: ExecutionMode,
        reservation: BudgetReservation,
        state: _RunState,
        step: Any,
        operation: TrustedOperation,
        observations: Mapping[str, Any],
        run_start_ms: int,
    ) -> _StepExecutionResult:
        step_id = _identifier(step, "step_id")
        declared_effects = set(_as_tuple(_get(step, "declared_effect_ids", ())))
        procedure_effects = {
            _identifier(item, "effect_id"): item for item in procedure.declared_effects
        }
        if not declared_effects.issubset(procedure_effects):
            return _StepExecutionResult(
                False,
                failure_code=RuntimeFailureCode.EFFECT_VIOLATION.value,
                failure_transition=_enum_value(step.failure_transition),
                failure_target=step.failure_target,
            )
        if not declared_effects.issubset(set(operation.allowed_effect_ids)):
            return _StepExecutionResult(
                False,
                failure_code=RuntimeFailureCode.EFFECT_VIOLATION.value,
                failure_transition=_enum_value(step.failure_transition),
                failure_target=step.failure_target,
            )
        dry_run = bool(_get(invocation, "dry_run", False))
        if not operation.read_only and dry_run and not operation.supports_dry_run:
            return _StepExecutionResult(
                False,
                failure_code=RuntimeFailureCode.AUTHORITY_REJECTED.value,
                failure_transition=_enum_value(step.failure_transition),
                failure_target=step.failure_target,
            )
        if not operation.read_only and mode is not ExecutionMode.LIVE:
            return _StepExecutionResult(
                False,
                failure_code=RuntimeFailureCode.AUTHORITY_REJECTED.value,
                failure_transition=_enum_value(step.failure_transition),
                failure_target=step.failure_target,
            )

        try:
            authority = self._admit(
                self._admissions.authority,
                AdmissionRequest(
                    kind=AdmissionKind.AUTHORITY,
                    subject=tuple(_as_tuple(_get(step, "required_authority_ids", ()))),
                    procedure=procedure,
                    invocation=invocation,
                    certificate=certificate,
                    runtime=runtime,
                    mode=mode,
                    step_id=step_id,
                    variables=MappingProxyType(dict(state.variables)),
                    evidence_cids=tuple(_as_tuple(_get(invocation, "authority_receipt_cids", ()))),
                ),
                expected=(AdmissionKind.AUTHORITY,),
                failure=RuntimeFailureCode.AUTHORITY_REJECTED,
            )
        except ProcedureInterpreterError:
            return _StepExecutionResult(
                False,
                failure_code=RuntimeFailureCode.AUTHORITY_REJECTED.value,
                failure_transition=_enum_value(step.failure_transition),
                failure_target=step.failure_target,
            )
        state.evidence_cids.update(authority.receipt_cids)

        retry = step.retry_policy
        max_attempts = int(_get(retry, "max_attempts", 1))
        if max_attempts - 1 > operation.maximum_retries:
            return _StepExecutionResult(
                False,
                failure_code=RuntimeFailureCode.OPERATION_CONTRACT_DRIFT.value,
                failure_transition=_enum_value(step.failure_transition),
                failure_target=step.failure_target,
            )
        timeout_ms = int(step.timeout_ms)
        if timeout_ms > operation.maximum_timeout_ms:
            return _StepExecutionResult(
                False,
                failure_code=RuntimeFailureCode.OPERATION_CONTRACT_DRIFT.value,
                failure_transition=_enum_value(step.failure_transition),
                failure_target=step.failure_target,
            )
        retryable_codes = set(_as_tuple(_get(retry, "retryable_failure_codes", ())))

        for attempt in range(1, max_attempts + 1):
            inputs_or_error = self._resolve_inputs(
                step,
                state,
                procedure.bindings,
                procedure,
                operation,
                invocation.requested_scope,
            )
            if isinstance(inputs_or_error, str):
                return _StepExecutionResult(
                    False,
                    failure_code=RuntimeFailureCode.INVALID_PROCEDURE.value,
                    failure_transition=_enum_value(step.failure_transition),
                    failure_target=step.failure_target,
                )
            inputs = inputs_or_error
            request = OperationRequest(
                invocation_cid=state.invocation_cid,
                procedure_cid=state.procedure_cid,
                step_id=step_id,
                operation=operation.operation,
                operation_contract=operation.operation_contract,
                inputs=MappingProxyType(inputs),
                idempotency_key=_content_id(
                    "procedure-step-idempotency",
                    {
                        "invocation_cid": state.invocation_cid,
                        "step_id": step_id,
                        "attempt": attempt,
                    },
                ),
                timeout_ms=timeout_ms,
                attempt=attempt,
                deadline_ms=runtime.now_ms + timeout_ms,
                mode=mode,
                dry_run=dry_run,
                scope_paths=tuple(_as_tuple(_get(invocation, "requested_scope", ()))),
                lease_id=_identifier(invocation, "lease_id"),
                fencing_token=int(_get(invocation, "fencing_token", 0)),
            )
            # The prior attempt and any declared backoff may have consumed the
            # remaining wall budget.  No external callback is crossed without
            # a check immediately adjacent to dispatch.
            budget_failure = self._budget_failure(state, reservation, run_start_ms)
            if budget_failure is not None:
                return _StepExecutionResult(
                    False,
                    failure_code=budget_failure.value,
                    failure_transition=_enum_value(step.failure_transition),
                    failure_target=step.failure_target,
                )
            state.started_step_id = step_id
            state.attempt = attempt
            state.trace(
                TraceEventKind.STEP_STARTED,
                step_id=step_id,
                attempt=attempt,
                operation=operation.operation,
                evidence_cids=authority.receipt_cids,
                input_digest=_content_id("procedure-step-input", inputs),
            )
            self._save(state, CheckpointPhase.STEP_STARTED)
            call_started = self._clock.monotonic_ms()
            try:
                result = operation.handler(request)
            except OperationExecutionError as exc:
                elapsed = max(0, self._clock.monotonic_ms() - call_started)
                if not exc.outcome_observed:
                    return _StepExecutionResult(
                        False,
                        failure_code=RuntimeFailureCode.UNKNOWN_EXTERNAL_OUTCOME.value,
                        failure_transition="abort",
                        unknown_external_outcome=True,
                    )
                result = OperationResult(
                    success=False,
                    evidence_cids=exc.evidence_cids,
                    failure_code=exc.failure_code,
                    retryable=exc.retryable,
                    external_outcome_observed=True,
                    new_evidence=exc.new_evidence,
                    cost=RuntimeCost(elapsed_ms=elapsed),
                )
            except Exception:
                # An adapter exception is not evidence that an external effect
                # did or did not happen.  Mutation-capable dispatch therefore
                # becomes terminally ambiguous.  Read-only callbacks can be
                # classified as a known operational failure because there is
                # no mutation to replay or compensate.
                if not operation.read_only:
                    return _StepExecutionResult(
                        False,
                        failure_code=RuntimeFailureCode.UNKNOWN_EXTERNAL_OUTCOME.value,
                        failure_transition="abort",
                        unknown_external_outcome=True,
                    )
                elapsed = max(0, self._clock.monotonic_ms() - call_started)
                result = OperationResult(
                    success=False,
                    failure_code=RuntimeFailureCode.OPERATION_FAILED.value,
                    retryable=False,
                    cost=RuntimeCost(elapsed_ms=elapsed),
                )
            elapsed = max(0, self._clock.monotonic_ms() - call_started)
            if not isinstance(result, OperationResult):
                if not operation.read_only:
                    return _StepExecutionResult(
                        False,
                        failure_code=RuntimeFailureCode.UNKNOWN_EXTERNAL_OUTCOME.value,
                        failure_transition="abort",
                        unknown_external_outcome=True,
                    )
                result = OperationResult(
                    success=False,
                    failure_code=RuntimeFailureCode.OPERATION_FAILED.value,
                )
            if not result.external_outcome_observed:
                return _StepExecutionResult(
                    False,
                    failure_code=RuntimeFailureCode.UNKNOWN_EXTERNAL_OUTCOME.value,
                    failure_transition="abort",
                    unknown_external_outcome=True,
                )
            measured_cost = RuntimeCost(
                token_count=result.cost.token_count,
                resource_units=result.cost.resource_units,
                elapsed_ms=result.cost.elapsed_ms,
            )
            state.cost = state.cost.plus(measured_cost)
            state.observed_effect_ids.update(result.observed_effect_ids)
            state.changed_paths.update(result.changed_paths)
            state.evidence_cids.update(result.evidence_cids)
            state.trace(
                TraceEventKind.STEP_OBSERVED,
                step_id=step_id,
                attempt=attempt,
                operation=operation.operation,
                evidence_cids=result.evidence_cids,
                effects=result.observed_effect_ids,
                cost=measured_cost,
                reason_code=result.failure_code,
                input_digest=_content_id("procedure-step-input", inputs),
                output_digest=_content_id("procedure-step-output", result.outputs),
            )

            failure = self._enforce_result(
                procedure, invocation, step, operation, result, procedure_effects
            )
            if failure is None and elapsed > timeout_ms:
                failure = RuntimeFailureCode.TIME_BUDGET_EXHAUSTED
            if failure is None:
                failure = self._budget_failure(state, reservation, run_start_ms)
            if failure is not None:
                state.failure_code = failure.value
                self._save(state, CheckpointPhase.STEP_OBSERVED)
                return _StepExecutionResult(
                    False,
                    failure_code=failure.value,
                    failure_transition=_enum_value(step.failure_transition),
                    failure_target=step.failure_target,
                )

            if not result.success:
                code = result.failure_code or RuntimeFailureCode.OPERATION_FAILED.value
                state.failure_code = code
                self._save(state, CheckpointPhase.STEP_OBSERVED)
                can_retry = (
                    attempt < max_attempts
                    and bool(result.retryable)
                    and code in retryable_codes
                    and (
                        not bool(_get(retry, "requires_new_evidence", True))
                        or bool(result.new_evidence)
                    )
                )
                if can_retry:
                    state.trace(
                        TraceEventKind.STEP_RETRY,
                        step_id=step_id,
                        attempt=attempt,
                        operation=operation.operation,
                        evidence_cids=result.evidence_cids,
                        reason_code=code,
                    )
                    state.started_step_id = ""
                    state.failure_code = ""
                    self._save(state, CheckpointPhase.READY)
                    backoff = int(_get(retry, "backoff_ms", 0) or 0)
                    if backoff:
                        wait = getattr(self._clock, "wait_ms", None)
                        if callable(wait):
                            wait(backoff)
                        else:
                            time.sleep(backoff / 1000.0)
                    budget_failure = self._budget_failure(state, reservation, run_start_ms)
                    if budget_failure is not None:
                        return _StepExecutionResult(
                            False,
                            failure_code=budget_failure.value,
                            failure_transition=_enum_value(step.failure_transition),
                            failure_target=step.failure_target,
                        )
                    continue
                return _StepExecutionResult(
                    False,
                    failure_code=code,
                    failure_transition=_enum_value(step.failure_transition),
                    failure_target=step.failure_target,
                )

            try:
                self._apply_outputs(
                    step,
                    result,
                    state,
                    procedure,
                    operation,
                    invocation.requested_scope,
                )
                produced_observations = {
                    evidence_output
                    for evidence_output in step.evidence_outputs
                    if evidence_output in observations
                }
                produced_locals = {
                    (str(value)[len("local:") :] if str(value).startswith("local:") else str(value))
                    for value in step.output_bindings.values()
                }
                produced_observations.update(
                    observation_id
                    for observation_id, observation in observations.items()
                    if _identifier(observation, "producer_contract") == step.operation_contract
                    and (
                        _identifier(observation, "output_binding")[len("local:") :]
                        if _identifier(observation, "output_binding").startswith("local:")
                        else _identifier(observation, "output_binding")
                    )
                    in produced_locals
                )
                for evidence_output in sorted(produced_observations):
                    observation = observations[evidence_output]
                    decision = self._admit_predicate(
                        AdmissionKind.OBSERVATION,
                        observation,
                        procedure,
                        invocation,
                        certificate,
                        runtime,
                        mode,
                        state,
                        failure=RuntimeFailureCode.OBSERVATION_FAILED,
                        step_id=step_id,
                        operation_result=result,
                    )
                    state.evidence_cids.update(decision.receipt_cids)
                    state.admitted_observation_ids.add(evidence_output)
                for invariant in procedure.invariants:
                    decision = self._admit_predicate(
                        AdmissionKind.INVARIANT,
                        invariant,
                        procedure,
                        invocation,
                        certificate,
                        runtime,
                        mode,
                        state,
                        failure=RuntimeFailureCode.INVARIANT_FAILED,
                        step_id=step_id,
                        operation_result=result,
                    )
                    state.evidence_cids.update(decision.receipt_cids)
            except ProcedureInterpreterError as exc:
                failure_code = exc.reason_code
                state.failure_code = failure_code
                self._save(state, CheckpointPhase.STEP_OBSERVED)
                return _StepExecutionResult(
                    False,
                    failure_code=failure_code,
                    failure_transition=_enum_value(step.failure_transition),
                    failure_target=step.failure_target,
                )
            state.executed_step_ids.add(step_id)
            state.started_step_id = ""
            state.failure_code = ""
            state.current_node_id = (
                ""
                if step_id in set(procedure.terminal_step_ids)
                else _identifier(step, "next_step_id")
            )
            self._save(state, CheckpointPhase.STEP_OBSERVED)
            return _StepExecutionResult(True, next_node_id=_identifier(step, "next_step_id"))

        return _StepExecutionResult(
            False,
            failure_code=RuntimeFailureCode.RETRY_EXHAUSTED.value,
            failure_transition=_enum_value(step.failure_transition),
            failure_target=step.failure_target,
        )

    def _resolve_inputs(
        self,
        step: Any,
        state: _RunState,
        exact_bindings: Any,
        procedure: ProcedureSpec,
        operation: TrustedOperation,
        scope_paths: Sequence[str],
    ) -> dict[str, Any] | str:
        bindings = _get(step, "input_bindings", {})
        if not isinstance(bindings, Mapping):
            return "input_bindings must be a mapping"
        resolved: dict[str, Any] = {}
        parameters = {item.name: item for item in procedure.parameters}
        for input_name, variable_name in sorted(bindings.items()):
            if not isinstance(input_name, str) or not isinstance(variable_name, str):
                return "input binding names must be strings"
            if variable_name.startswith("literal:"):
                raw_value = variable_name[len("literal:") :]
                resolved[input_name] = _validate_typed_value(
                    raw_value,
                    operation.input_types[input_name],
                    field_name=f"operation input {input_name}",
                    scope_paths=scope_paths,
                )
                continue
            if variable_name.startswith("binding:"):
                binding_name = variable_name[len("binding:") :]
                binding_value = _get(exact_bindings, binding_name, None)
                if binding_value is None:
                    return "input binding references an unknown exact binding"
                resolved[input_name] = _validate_typed_value(
                    binding_value,
                    operation.input_types[input_name],
                    field_name=f"operation input {input_name}",
                    scope_paths=scope_paths,
                )
                continue
            for prefix in ("parameter:", "param:", "local:"):
                if variable_name.startswith(prefix):
                    variable_name = variable_name[len(prefix) :]
                    break
            if variable_name not in state.variables:
                return "input binding references an uninitialized variable"
            if state.variables[variable_name] is None:
                return "input binding references an uninitialized local"
            parameter = parameters.get(variable_name)
            resolved[input_name] = _validate_typed_value(
                state.variables[variable_name],
                operation.input_types[input_name],
                field_name=f"operation input {input_name}",
                scope_paths=scope_paths,
                allowed_values=parameter.allowed_values if parameter else (),
            )
        return resolved

    def _apply_outputs(
        self,
        step: Any,
        result: OperationResult,
        state: _RunState,
        procedure: ProcedureSpec,
        operation: TrustedOperation,
        scope_paths: Sequence[str],
    ) -> None:
        bindings = _get(step, "output_bindings", {})
        if not isinstance(bindings, Mapping):
            raise ProcedureInterpreterError(
                "output_bindings must be a mapping",
                reason_code=RuntimeFailureCode.INVALID_PROCEDURE.value,
            )
        for output_name, variable_name in sorted(bindings.items()):
            if output_name not in result.outputs:
                raise ProcedureInterpreterError(
                    "operation omitted declared output {}".format(output_name),
                    reason_code=RuntimeFailureCode.OBSERVATION_FAILED.value,
                )
            if variable_name.startswith("local:"):
                variable_name = variable_name[len("local:") :]
            if variable_name not in state.variables:
                raise ProcedureInterpreterError(
                    "output targets an undeclared local",
                    reason_code=RuntimeFailureCode.INVALID_PROCEDURE.value,
                )
            state.variables[variable_name] = _validate_typed_value(
                result.outputs[output_name],
                operation.output_types[output_name],
                field_name=f"operation output {output_name}",
                scope_paths=scope_paths,
            )
        unknown = set(result.outputs).difference(bindings)
        if unknown:
            raise ProcedureInterpreterError(
                "operation returned undeclared outputs: {}".format(sorted(unknown)),
                reason_code=RuntimeFailureCode.OBSERVATION_FAILED.value,
            )

    def _enforce_result(
        self,
        procedure: ProcedureSpec,
        invocation: ProcedureInvocation,
        step: Any,
        operation: TrustedOperation,
        result: OperationResult,
        procedure_effects: Mapping[str, Any],
    ) -> RuntimeFailureCode | None:
        step_effects = set(_as_tuple(_get(step, "declared_effect_ids", ())))
        observed = set(result.observed_effect_ids)
        if not observed.issubset(step_effects):
            return RuntimeFailureCode.EFFECT_VIOLATION
        if not observed.issubset(operation.allowed_effect_ids):
            return RuntimeFailureCode.EFFECT_VIOLATION
        if operation.read_only and result.changed_paths:
            return RuntimeFailureCode.EFFECT_VIOLATION
        if bool(_get(invocation, "dry_run", False)) and result.changed_paths:
            return RuntimeFailureCode.EFFECT_VIOLATION

        requested_scope = tuple(_as_tuple(_get(invocation, "requested_scope", ())))
        for changed in result.changed_paths:
            try:
                if not _path_is_within(changed, requested_scope):
                    return RuntimeFailureCode.SCOPE_ESCAPE
            except ProcedureInterpreterError:
                return RuntimeFailureCode.SCOPE_ESCAPE
            permitted_by_effect = False
            for effect_id in observed:
                effect = procedure_effects.get(effect_id)
                targets = tuple(_as_tuple(_get(effect, "targets", ())))
                if targets and _path_is_within(changed, targets):
                    permitted_by_effect = True
                    break
            if not permitted_by_effect:
                return RuntimeFailureCode.EFFECT_VIOLATION

        declared_reads = tuple(_as_tuple(_get(procedure, "declared_reads", ())))
        for read_path in result.read_paths:
            try:
                if not declared_reads or not _path_is_within(read_path, declared_reads):
                    return RuntimeFailureCode.READ_SCOPE_ESCAPE
                if not _path_is_within(read_path, requested_scope):
                    return RuntimeFailureCode.READ_SCOPE_ESCAPE
            except ProcedureInterpreterError:
                return RuntimeFailureCode.READ_SCOPE_ESCAPE
        if (observed or result.changed_paths) and not result.evidence_cids:
            return RuntimeFailureCode.OBSERVATION_FAILED
        return None

    def _budget_failure(
        self,
        state: _RunState,
        reservation: BudgetReservation,
        run_start_ms: int,
    ) -> RuntimeFailureCode | None:
        elapsed = max(state.cost.elapsed_ms, self._clock.monotonic_ms() - run_start_ms)
        if state.cost.token_count > reservation.token_limit:
            return RuntimeFailureCode.TOKEN_BUDGET_EXHAUSTED
        if state.cost.resource_units > reservation.resource_limit:
            return RuntimeFailureCode.RESOURCE_BUDGET_EXHAUSTED
        if elapsed >= reservation.time_limit_ms:
            return RuntimeFailureCode.TIME_BUDGET_EXHAUSTED
        return None

    def _terminal_failure(
        self,
        procedure: ProcedureSpec,
        invocation: ProcedureInvocation,
        certificate: ProcedureCertificate,
        state: _RunState,
        runtime: RuntimeIdentity,
        code: RuntimeFailureCode,
        detail: str,
        *,
        resumed: bool,
        steps: Mapping[str, Any],
        operations: Mapping[str, TrustedOperation],
        mode: ExecutionMode,
        rollback_target: str = "",
    ) -> ProcedureExecution:
        state.failure_code = code.value
        if state.observed_effect_ids or state.changed_paths:
            rolled_back = self._perform_rollback(
                procedure,
                invocation,
                certificate,
                runtime,
                mode,
                state,
                steps,
                operations,
                rollback_target=rollback_target,
            )
            state.status = (
                RuntimeOutcomeStatus.ROLLED_BACK if rolled_back else RuntimeOutcomeStatus.FAILED
            )
            if not rolled_back and procedure.rollback and state.failure_code == code.value:
                state.failure_code = RuntimeFailureCode.ROLLBACK_FAILED.value
        else:
            state.status = RuntimeOutcomeStatus.FAILED
        state.trace(
            TraceEventKind.TERMINAL,
            step_id=state.current_node_id,
            reason_code=state.failure_code,
            detail=detail,
        )
        return self._finish(procedure, invocation, certificate, state, runtime, resumed=resumed)

    def _perform_rollback(
        self,
        procedure: ProcedureSpec,
        invocation: ProcedureInvocation,
        certificate: ProcedureCertificate,
        runtime: RuntimeIdentity,
        mode: ExecutionMode,
        state: _RunState,
        steps: Mapping[str, Any],
        operations: Mapping[str, TrustedOperation],
        *,
        rollback_target: str,
    ) -> bool:
        rollback_plans = tuple(_as_tuple(_get(procedure, "rollback", ())))
        planned: list[str] = []
        selected_plans: list[Any] = []
        for rollback in rollback_plans:
            triggers = set(_as_tuple(_get(rollback, "trigger_effect_ids", ())))
            configured = _first(
                rollback,
                ("step_ids", "rollback_step_ids", "compensation_step_ids"),
                (),
            )
            configured_ids = tuple(str(item) for item in _as_tuple(configured))
            explicitly_selected = (
                bool(rollback_target) and _identifier(rollback, "rollback_id") == rollback_target
            )
            if explicitly_selected or (
                not rollback_target and triggers.intersection(state.observed_effect_ids)
            ):
                selected_plans.append(rollback)
                for item_id in configured_ids:
                    if item_id and item_id not in planned:
                        planned.append(item_id)
        if not planned or not selected_plans:
            return False
        state.trace(
            TraceEventKind.ROLLBACK_STARTED,
            step_id=planned[0],
            reason_code=state.failure_code,
        )
        self._save(state, CheckpointPhase.ROLLING_BACK)

        for step_id in planned:
            step = steps.get(step_id)
            operation = operations.get(step_id)
            if step is None or operation is None or operation.operation != "ROLLBACK":
                return False
            if operation.read_only:
                return False
            # Rollback dispatch is intentionally single-attempt.  An ambiguous
            # compensation outcome must be surfaced, never replayed.
            inputs = self._resolve_inputs(
                step,
                state,
                procedure.bindings,
                procedure,
                operation,
                invocation.requested_scope,
            )
            if isinstance(inputs, str):
                return False
            request = OperationRequest(
                invocation_cid=state.invocation_cid,
                procedure_cid=state.procedure_cid,
                step_id=step_id,
                operation=operation.operation,
                operation_contract=operation.operation_contract,
                inputs=MappingProxyType(inputs),
                idempotency_key=_content_id(
                    "procedure-rollback-idempotency",
                    {"invocation_cid": state.invocation_cid, "step_id": step_id},
                ),
                timeout_ms=min(int(step.timeout_ms), operation.maximum_timeout_ms),
                attempt=1,
                deadline_ms=runtime.now_ms
                + min(int(step.timeout_ms), operation.maximum_timeout_ms),
                mode=mode,
                dry_run=False,
                scope_paths=tuple(_as_tuple(_get(invocation, "requested_scope", ()))),
                lease_id=_identifier(invocation, "lease_id"),
                fencing_token=int(_get(invocation, "fencing_token", 0)),
            )
            try:
                rollback_authority = self._admit(
                    self._admissions.authority,
                    AdmissionRequest(
                        kind=AdmissionKind.AUTHORITY,
                        subject=tuple(step.required_authority_ids),
                        procedure=procedure,
                        invocation=invocation,
                        certificate=certificate,
                        runtime=runtime,
                        mode=mode,
                        step_id=step_id,
                        variables=MappingProxyType(dict(state.variables)),
                        evidence_cids=tuple(invocation.authority_receipt_cids),
                    ),
                    expected=(AdmissionKind.AUTHORITY,),
                    failure=RuntimeFailureCode.AUTHORITY_REJECTED,
                )
            except ProcedureInterpreterError:
                return False
            state.evidence_cids.update(rollback_authority.receipt_cids)
            state.started_step_id = step_id
            state.attempt = 1
            state.trace(
                TraceEventKind.STEP_STARTED,
                step_id=step_id,
                attempt=1,
                operation=operation.operation,
                evidence_cids=rollback_authority.receipt_cids,
                input_digest=_content_id("procedure-step-input", inputs),
            )
            self._save(state, CheckpointPhase.STEP_STARTED)
            try:
                result = operation.handler(request)
            except Exception:
                return False
            if (
                not isinstance(result, OperationResult)
                or not result.external_outcome_observed
                or not result.success
            ):
                return False
            state.changed_paths.update(result.changed_paths)
            procedure_effects = {
                _identifier(item, "effect_id"): item for item in procedure.declared_effects
            }
            if (
                self._enforce_result(
                    procedure,
                    invocation,
                    step,
                    operation,
                    result,
                    procedure_effects,
                )
                is not None
            ):
                return False
            state.started_step_id = ""
            state.observed_effect_ids.update(result.observed_effect_ids)
            state.evidence_cids.update(result.evidence_cids)
            try:
                self._apply_outputs(
                    step,
                    result,
                    state,
                    procedure,
                    operation,
                    invocation.requested_scope,
                )
            except ProcedureInterpreterError:
                return False
            try:
                matching_plan = next(
                    (
                        item
                        for item in selected_plans
                        if step_id in tuple(_as_tuple(_get(item, "step_ids", ())))
                    ),
                    selected_plans[0],
                )
                decision = self._admit_predicate(
                    AdmissionKind.ROLLBACK,
                    matching_plan,
                    procedure,
                    invocation,
                    certificate,
                    runtime,
                    mode,
                    state,
                    failure=RuntimeFailureCode.ROLLBACK_FAILED,
                    step_id=step_id,
                    operation_result=result,
                )
            except ProcedureInterpreterError:
                return False
            state.rollback_receipt_cids.update(decision.receipt_cids)
            state.evidence_cids.update(decision.receipt_cids)
            state.trace(
                TraceEventKind.ROLLBACK_OBSERVED,
                step_id=step_id,
                operation=operation.operation,
                evidence_cids=tuple(result.evidence_cids) + decision.receipt_cids,
                effects=result.observed_effect_ids,
            )
            self._save(state, CheckpointPhase.STEP_OBSERVED)
        return True

    def _fail_without_dispatch(
        self,
        procedure: ProcedureSpec,
        invocation: ProcedureInvocation,
        certificate: ProcedureCertificate,
        state: _RunState,
        runtime: RuntimeIdentity,
        code: RuntimeFailureCode,
        detail: str,
        *,
        resumed: bool,
    ) -> ProcedureExecution:
        state.failure_code = code.value
        state.status = RuntimeOutcomeStatus.REFUSED
        state.trace(TraceEventKind.TERMINAL, reason_code=code.value, detail=detail)
        return self._finish(procedure, invocation, certificate, state, runtime, resumed=resumed)

    def _finish(
        self,
        procedure: ProcedureSpec,
        invocation: ProcedureInvocation,
        certificate: ProcedureCertificate,
        state: _RunState,
        runtime: RuntimeIdentity,
        *,
        resumed: bool,
        persist: bool = True,
        recovered_trace: bool | None = None,
    ) -> ProcedureExecution:
        if state.terminal_at_ms == 0:
            state.terminal_at_ms = runtime.now_ms
        checkpoint = state.checkpoint(CheckpointPhase.TERMINAL)
        if persist:
            self._checkpoints.save(checkpoint)

        contract_entries: list[ProcedureTraceEntry] = []
        time_cursor = int(_get(invocation, "requested_at_ms", runtime.now_ms))
        for entry in state.trace_entries:
            if not entry.step_id or not entry.operation:
                continue
            try:
                operation = StepOperation(entry.operation)
            except ValueError:
                # This is defensive.  The trusted catalog only resolves the
                # closed StepOperation vocabulary and therefore cannot reach it.
                continue
            if entry.event is TraceEventKind.STEP_STARTED:
                status = TraceEventStatus.STARTED
            elif entry.event is TraceEventKind.STEP_RETRY:
                status = TraceEventStatus.RETRYING
            elif entry.event is TraceEventKind.ROLLBACK_OBSERVED:
                status = TraceEventStatus.ROLLED_BACK
            elif entry.reason_code:
                status = TraceEventStatus.FAILED
            else:
                status = TraceEventStatus.OBSERVED
            started_at = time_cursor
            ended_at = started_at + entry.cost.elapsed_ms
            time_cursor = ended_at
            input_digest = entry.input_digest or _content_id("procedure-step-input", {})
            contract_entries.append(
                ProcedureTraceEntry(
                    sequence=len(contract_entries),
                    step_id=entry.step_id,
                    operation=operation,
                    status=status,
                    attempt=max(1, entry.attempt),
                    started_at_ms=started_at,
                    ended_at_ms=ended_at,
                    input_digest=input_digest,
                    output_digest=entry.output_digest,
                    observed_effect_ids=entry.observed_effect_ids,
                    evidence_cids=entry.evidence_cids,
                    failure_code=(entry.reason_code if status is TraceEventStatus.FAILED else ""),
                )
            )

        trace_state = TraceState.COMPLETE
        if state.status is RuntimeOutcomeStatus.UNKNOWN_EXTERNAL_OUTCOME:
            trace_state = TraceState.INTERRUPTED
        elif state.status not in {
            RuntimeOutcomeStatus.SUCCEEDED,
            RuntimeOutcomeStatus.ROLLED_BACK,
        }:
            trace_state = TraceState.FAILED
        elif resumed if recovered_trace is None else recovered_trace:
            trace_state = TraceState.RECOVERED
        trace = ProcedureExecutionTrace(
            bindings=procedure.bindings,
            invocation_cid=state.invocation_cid,
            procedure_cid=state.procedure_cid,
            entries=tuple(contract_entries),
            checkpoint_cids=(checkpoint.checkpoint_cid,),
            state=trace_state,
        )

        status_map = {
            RuntimeOutcomeStatus.SUCCEEDED: ProcedureOutcomeStatus.SUCCEEDED,
            RuntimeOutcomeStatus.FAILED: ProcedureOutcomeStatus.FAILED,
            RuntimeOutcomeStatus.ROLLED_BACK: ProcedureOutcomeStatus.ROLLED_BACK,
            RuntimeOutcomeStatus.BLOCKED: ProcedureOutcomeStatus.REFUSED,
            RuntimeOutcomeStatus.INCOMPLETE: ProcedureOutcomeStatus.INCOMPLETE,
            RuntimeOutcomeStatus.UNKNOWN_EXTERNAL_OUTCOME: ProcedureOutcomeStatus.INCOMPLETE,
            RuntimeOutcomeStatus.ESCALATED: ProcedureOutcomeStatus.ESCALATED,
            RuntimeOutcomeStatus.QUARANTINED: ProcedureOutcomeStatus.QUARANTINED,
            RuntimeOutcomeStatus.REFUSED: ProcedureOutcomeStatus.REFUSED,
            RuntimeOutcomeStatus.CANCELLED: ProcedureOutcomeStatus.CANCELLED,
        }
        failure_cid = ""
        failure: ProcedureFailure | None = None
        if state.status is not RuntimeOutcomeStatus.SUCCEEDED:
            failure = ProcedureFailure(
                bindings=procedure.bindings,
                invocation_cid=state.invocation_cid,
                procedure_cid=state.procedure_cid,
                step_id=state.started_step_id or state.current_node_id or procedure.entry_step_id,
                failure_code=state.failure_code or RuntimeFailureCode.INTERNAL_ERROR.value,
                retryable=False,
                diagnostic_cids=tuple(sorted(state.evidence_cids)),
                observed_at_ms=state.terminal_at_ms,
            )
            failure_cid = failure.content_id
        outcome = ProcedureOutcome(
            bindings=procedure.bindings,
            invocation_cid=state.invocation_cid,
            procedure_cid=state.procedure_cid,
            status=status_map[state.status],
            observed_effect_ids=tuple(sorted(state.observed_effect_ids)),
            validation_receipt_cids=tuple(sorted(state.validation_receipt_cids)),
            satisfied_postcondition_ids=tuple(sorted(state.satisfied_postcondition_ids)),
            rollback_receipt_cids=tuple(sorted(state.rollback_receipt_cids)),
            trace_cid=trace.content_id,
            terminal_at_ms=state.terminal_at_ms,
            failure_cid=failure_cid,
        )
        receipt = ProcedureInvocationReceipt(
            bindings=procedure.bindings,
            invocation_cid=state.invocation_cid,
            procedure_cid=state.procedure_cid,
            certificate_cid=_artifact_id(certificate, "procedure-certificate"),
            trace_cid=trace.content_id,
            outcome_cid=outcome.content_id,
            admitted_evidence_cids=tuple(sorted(state.evidence_cids)),
            emitted_at_ms=state.terminal_at_ms,
        )
        return ProcedureExecution(
            trace=trace,
            outcome=outcome,
            receipt=receipt,
            failure=failure,
            cost=state.cost,
            resumed=resumed,
            checkpoint_cid=checkpoint.checkpoint_cid,
        )


__all__ = [
    "AdmissionDecision",
    "AdmissionKind",
    "AdmissionProducer",
    "AdmissionRequest",
    "BudgetReservation",
    "BudgetReservationPort",
    "BudgetReservationRequest",
    "CertificateClass",
    "CheckpointPhase",
    "CheckpointStore",
    "ExecutionMode",
    "IdempotencyStore",
    "InMemoryCheckpointStore",
    "InMemoryIdempotencyStore",
    "InterpreterAdmissionPorts",
    "InterpreterCheckpoint",
    "IsolationReservation",
    "IsolationReservationPort",
    "IsolationReservationRequest",
    "OperationExecutionError",
    "OperationRequest",
    "OperationResult",
    "ProcedureExecution",
    "ProcedureInterpreter",
    "ProcedureInterpreterError",
    "RuntimeClock",
    "RuntimeCost",
    "RuntimeFailureCode",
    "RuntimeIdentity",
    "RuntimeOutcomeStatus",
    "RuntimeTraceEntry",
    "SystemRuntimeClock",
    "TraceEventKind",
    "TrustedOperation",
    "TrustedOperationCatalog",
]
