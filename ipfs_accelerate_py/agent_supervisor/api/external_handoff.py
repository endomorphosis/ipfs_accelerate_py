"""Closed in-process External Agent Handoff Python API (EAAEF-110).

This module is the transport-neutral Python surface for handoff, preview,
attach, status, follow, steer, pause, resume, approve, reject, cancel,
explain, doctor, report and export.  Callers pass canonical request dicts or
small dataclasses and receive canonical receipts with content identities.

The API is pure and in-process: it uses an in-memory run registry and does
not talk to live Quack, Docker, or the network.  Preview never admits a
mutating handoff.  ``approve`` requires an independent reviewer principal
distinct from the worker.  Cancel, pause, resume and steer bind the exact
run identity plus the authority id issued at admission.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)


CONTRACT_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = CONTRACT_VERSION

EXTERNAL_HANDOFF_API_INTERFACE: Final[str] = "ExternalHandoffAPI@1"
EXTERNAL_HANDOFF_API_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-handoff-api@1"
)
EXTERNAL_HANDOFF_REQUEST_INTERFACE: Final[str] = "ExternalHandoffOperationRequest@1"
EXTERNAL_HANDOFF_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-handoff-operation-request@1"
)
EXTERNAL_HANDOFF_RECEIPT_INTERFACE: Final[str] = "ExternalHandoffOperationReceipt@1"
EXTERNAL_HANDOFF_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-handoff-operation-receipt@1"
)
EXTERNAL_HANDOFF_RUN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-handoff-run@1"
)
EXTERNAL_HANDOFF_AUTHORITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-handoff-authority@1"
)
EXTERNAL_HANDOFF_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-handoff-event@1"
)
EXTERNAL_HANDOFF_EXPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-handoff-export@1"
)

MAX_ID_BYTES: Final[int] = 256
MAX_REASON_BYTES: Final[int] = 256
MAX_INSTRUCTION_BYTES: Final[int] = 4096
MAX_EVENTS: Final[int] = 4096

HANDOFF_API_OPERATIONS: Final[tuple[str, ...]] = (
    "handoff",
    "preview",
    "attach",
    "status",
    "follow",
    "steer",
    "pause",
    "resume",
    "approve",
    "reject",
    "cancel",
    "explain",
    "doctor",
    "report",
    "export",
)
OPERATIONS: Final[tuple[str, ...]] = HANDOFF_API_OPERATIONS

_CONTROL_OPERATIONS: Final[frozenset[str]] = frozenset(
    {"steer", "pause", "resume", "cancel"}
)
_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset(
    {"cancelled", "approved", "rejected"}
)

_WIRE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "interface",
        "contract_version",
        "schema_version",
        "content_id",
        "cid",
        "identity",
        "canonical_id",
        "request_id",
        "receipt_id",
    }
)
_REQUEST_FIELDS: Final[tuple[str, ...]] = (
    "operation",
    "principal_id",
    "worker_principal_id",
    "reviewer_principal_id",
    "authority_id",
    "run_id",
    "session_id",
    "repository_id",
    "objective_id",
    "idempotency_key",
    "cursor",
    "instruction",
    "reason",
)
_PRIVATE_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "chain_of_thought",
        "cookie",
        "credential",
        "hidden_chain_of_thought",
        "hidden_cot",
        "password",
        "private_key",
        "secret",
        "session_token",
        "thinking",
        "transcript_body",
        "witness",
    }
)


class HandoffApiOperation(str, Enum):
    """Closed Python-API operation vocabulary."""

    HANDOFF = "handoff"
    PREVIEW = "preview"
    ATTACH = "attach"
    STATUS = "status"
    FOLLOW = "follow"
    STEER = "steer"
    PAUSE = "pause"
    RESUME = "resume"
    APPROVE = "approve"
    REJECT = "reject"
    CANCEL = "cancel"
    EXPLAIN = "explain"
    DOCTOR = "doctor"
    REPORT = "report"
    EXPORT = "export"


class HandoffApiVerdict(str, Enum):
    """Admission verdict carried on origin receipts.  Preview is not mutation."""

    ADMITTED = "admitted"
    PREVIEW_ONLY = "preview_only"
    NONE = "none"


class ExternalHandoffAPIError(ContractValidationError):
    """Closed-API failure.  Unknown operations and authority mismatches fail closed."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class ExternalHandoffAuthorityError(ExternalHandoffAPIError):
    """Run identity or authority id did not match the admitted run."""


class WorkerSelfApprovalError(ExternalHandoffAPIError):
    """A worker principal attempted to approve or reject its own work."""


def _normalize_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = MAX_ID_BYTES,
) -> str:
    if value is None:
        result = ""
    elif not isinstance(value, str):
        raise ExternalHandoffAPIError(f"{name} must be a string", reason_code="malformed")
    else:
        result = value.strip()
    if required and not result:
        raise ExternalHandoffAPIError(f"{name} is required", reason_code="malformed")
    if "\x00" in result:
        raise ExternalHandoffAPIError(
            f"{name} must not contain NUL", reason_code="malformed"
        )
    if len(result.encode("utf-8")) > max_bytes:
        raise ExternalHandoffAPIError(
            f"{name} exceeds {max_bytes} UTF-8 bytes", reason_code="bounds"
        )
    return result


def _optional_text(value: Any, name: str, *, max_bytes: int = MAX_ID_BYTES) -> str:
    return _text(value, name, required=False, max_bytes=max_bytes)


def _operation_name(value: Any) -> str:
    if isinstance(value, HandoffApiOperation):
        name = value.value
    else:
        name = _normalize_key(_text(value, "operation"))
        if name == "export_result":
            name = "export"
    if name not in HANDOFF_API_OPERATIONS:
        raise ExternalHandoffAPIError(
            "unknown handoff API operation", reason_code="unknown_operation"
        )
    return name


def _reject_unknown(payload: Mapping[str, Any], allowed: Sequence[str], *, name: str) -> None:
    extra = set(payload).difference(allowed)
    if extra:
        raise ExternalHandoffAPIError(
            f"{name} contains unsupported fields; rebuild its canonical payload",
            reason_code="malformed",
        )


def _reject_private_material(value: Any, *, name: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = _normalize_key(raw_key)
            if key in _PRIVATE_MARKERS or any(
                key.endswith("_" + marker) or marker in key for marker in _PRIVATE_MARKERS
            ):
                raise ExternalHandoffAPIError(
                    f"{name} must not contain private material or hidden chain-of-thought",
                    reason_code="private_material",
                )
            _reject_private_material(item, name=name)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _reject_private_material(item, name=name)


def _require_schema(
    payload: Mapping[str, Any],
    expected_schema: str,
    expected_interface: str,
    *,
    artifact_name: str,
) -> None:
    schema = payload.get("schema")
    if schema not in (None, "", expected_schema):
        raise ExternalHandoffAPIError(
            f"unsupported {artifact_name} schema",
            reason_code="unsupported_version",
        )
    interface = payload.get("interface")
    if interface not in (None, "", expected_interface):
        raise ExternalHandoffAPIError(
            f"unsupported {artifact_name} interface",
            reason_code="unsupported_version",
        )
    for key in ("contract_version", "schema_version"):
        version = payload.get(key)
        if version not in (None, "", CONTRACT_VERSION):
            raise ExternalHandoffAPIError(
                f"unsupported {artifact_name} contract version",
                reason_code="unsupported_version",
            )


def _claimed_identity(
    payload: Mapping[str, Any],
    actual: str,
    *,
    artifact_name: str,
    names: Sequence[str] = ("content_id", "cid", "identity", "canonical_id"),
) -> None:
    for name in names:
        claimed = payload.get(name)
        if claimed not in (None, "") and claimed != actual:
            raise ExternalHandoffAPIError(
                f"{artifact_name} content identity does not match payload",
                reason_code="identity_mismatch",
            )


def _ids(values: Any, name: str) -> tuple[str, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ExternalHandoffAPIError(
            f"{name} must be a sequence of strings", reason_code="malformed"
        )
    else:
        items = values
    if len(items) > MAX_EVENTS:
        raise ExternalHandoffAPIError(f"{name} exceeds its item-count limit", reason_code="bounds")
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _text(item, name)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


def _freeze_identities(value: Any) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ExternalHandoffAPIError("identities must be an object", reason_code="malformed")
    frozen: dict[str, str] = {}
    for key in sorted(value):
        frozen[_text(str(key), "identities key")] = _optional_text(
            value[key], "identities value"
        )
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class ExternalHandoffRequest(CanonicalContract):
    """Canonical in-process request for one closed handoff-API operation."""

    SCHEMA: ClassVar[str] = EXTERNAL_HANDOFF_REQUEST_SCHEMA

    operation: str
    principal_id: str
    worker_principal_id: str = ""
    reviewer_principal_id: str = ""
    authority_id: str = ""
    run_id: str = ""
    session_id: str = ""
    repository_id: str = ""
    objective_id: str = ""
    idempotency_key: str = ""
    cursor: str = ""
    instruction: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _operation_name(self.operation))
        object.__setattr__(self, "principal_id", _text(self.principal_id, "principal_id"))
        object.__setattr__(
            self,
            "worker_principal_id",
            _optional_text(self.worker_principal_id, "worker_principal_id"),
        )
        object.__setattr__(
            self,
            "reviewer_principal_id",
            _optional_text(self.reviewer_principal_id, "reviewer_principal_id"),
        )
        for name in (
            "authority_id",
            "run_id",
            "session_id",
            "repository_id",
            "objective_id",
            "idempotency_key",
            "cursor",
        ):
            object.__setattr__(self, name, _optional_text(getattr(self, name), name))
        object.__setattr__(
            self,
            "instruction",
            _optional_text(self.instruction, "instruction", max_bytes=MAX_INSTRUCTION_BYTES),
        )
        object.__setattr__(
            self,
            "reason",
            _optional_text(self.reason, "reason", max_bytes=MAX_REASON_BYTES),
        )
        _reject_private_material(self._payload(), name="external handoff request")

    @property
    def request_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": EXTERNAL_HANDOFF_REQUEST_INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "operation": self.operation,
            "principal_id": self.principal_id,
            "worker_principal_id": self.worker_principal_id,
            "reviewer_principal_id": self.reviewer_principal_id,
            "authority_id": self.authority_id,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "repository_id": self.repository_id,
            "objective_id": self.objective_id,
            "idempotency_key": self.idempotency_key,
            "cursor": self.cursor,
            "instruction": self.instruction,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalHandoffRequest":
        if not isinstance(payload, Mapping):
            raise ExternalHandoffAPIError(
                "handoff request payload must be an object", reason_code="malformed"
            )
        _require_schema(
            payload,
            cls.SCHEMA,
            EXTERNAL_HANDOFF_REQUEST_INTERFACE,
            artifact_name="external handoff request",
        )
        _reject_unknown(
            payload,
            tuple(_WIRE_FIELDS.union(_REQUEST_FIELDS)),
            name="external handoff request",
        )
        result = cls(
            operation=payload.get("operation", ""),
            principal_id=payload.get("principal_id", ""),
            worker_principal_id=payload.get("worker_principal_id", ""),
            reviewer_principal_id=payload.get("reviewer_principal_id", ""),
            authority_id=payload.get("authority_id", ""),
            run_id=payload.get("run_id", ""),
            session_id=payload.get("session_id", ""),
            repository_id=payload.get("repository_id", ""),
            objective_id=payload.get("objective_id", ""),
            idempotency_key=payload.get("idempotency_key", ""),
            cursor=payload.get("cursor", ""),
            instruction=payload.get("instruction", ""),
            reason=payload.get("reason", ""),
        )
        _claimed_identity(
            payload,
            result.content_id,
            artifact_name="external handoff request",
            names=("content_id", "cid", "identity", "canonical_id", "request_id"),
        )
        return result


@dataclass(frozen=True)
class ExternalHandoffReceipt(CanonicalContract):
    """Canonical receipt for one closed handoff-API operation."""

    SCHEMA: ClassVar[str] = EXTERNAL_HANDOFF_RECEIPT_SCHEMA

    operation: str
    run_id: str
    request_id: str
    authority_id: str
    principal_id: str
    status: str = "ok"
    run_status: str = "running"
    verdict: str = HandoffApiVerdict.NONE.value
    worker_principal_id: str = ""
    reviewer_principal_id: str = ""
    session_id: str = ""
    repository_id: str = ""
    cursor: str = ""
    reason_code: str = "bound"
    event_ids: tuple[str, ...] = ()
    export_id: str = ""
    identities: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _operation_name(self.operation))
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id"))
        object.__setattr__(self, "authority_id", _text(self.authority_id, "authority_id"))
        object.__setattr__(self, "principal_id", _text(self.principal_id, "principal_id"))
        object.__setattr__(self, "status", _text(self.status, "status"))
        object.__setattr__(self, "run_status", _text(self.run_status, "run_status"))
        verdict = self.verdict
        if isinstance(verdict, HandoffApiVerdict):
            verdict = verdict.value
        object.__setattr__(self, "verdict", _text(verdict, "verdict"))
        object.__setattr__(
            self,
            "worker_principal_id",
            _optional_text(self.worker_principal_id, "worker_principal_id"),
        )
        object.__setattr__(
            self,
            "reviewer_principal_id",
            _optional_text(self.reviewer_principal_id, "reviewer_principal_id"),
        )
        object.__setattr__(self, "session_id", _optional_text(self.session_id, "session_id"))
        object.__setattr__(
            self, "repository_id", _optional_text(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "cursor", _optional_text(self.cursor, "cursor"))
        object.__setattr__(
            self,
            "reason_code",
            _text(self.reason_code, "reason_code", max_bytes=MAX_REASON_BYTES),
        )
        object.__setattr__(self, "event_ids", _ids(self.event_ids, "event_ids"))
        object.__setattr__(self, "export_id", _optional_text(self.export_id, "export_id"))
        identities = _freeze_identities(self.identities)
        if not identities:
            identities = MappingProxyType(
                {
                    "run_id": self.run_id,
                    "request_id": self.request_id,
                    "authority_id": self.authority_id,
                    "session_id": self.session_id,
                }
            )
        object.__setattr__(self, "identities", identities)
        _reject_private_material(self._payload(), name="external handoff receipt")

    @property
    def receipt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": EXTERNAL_HANDOFF_RECEIPT_INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "operation": self.operation,
            "status": self.status,
            "run_status": self.run_status,
            "verdict": self.verdict,
            "run_id": self.run_id,
            "request_id": self.request_id,
            "authority_id": self.authority_id,
            "principal_id": self.principal_id,
            "worker_principal_id": self.worker_principal_id,
            "reviewer_principal_id": self.reviewer_principal_id,
            "session_id": self.session_id,
            "repository_id": self.repository_id,
            "cursor": self.cursor,
            "reason_code": self.reason_code,
            "event_ids": list(self.event_ids),
            "export_id": self.export_id,
            "identities": dict(self.identities),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalHandoffReceipt":
        if not isinstance(payload, Mapping):
            raise ExternalHandoffAPIError(
                "handoff receipt payload must be an object", reason_code="malformed"
            )
        _require_schema(
            payload,
            cls.SCHEMA,
            EXTERNAL_HANDOFF_RECEIPT_INTERFACE,
            artifact_name="external handoff receipt",
        )
        _reject_unknown(
            payload,
            tuple(
                _WIRE_FIELDS.union(
                    {
                        "operation",
                        "status",
                        "run_status",
                        "verdict",
                        "run_id",
                        "request_id",
                        "authority_id",
                        "principal_id",
                        "worker_principal_id",
                        "reviewer_principal_id",
                        "session_id",
                        "repository_id",
                        "cursor",
                        "reason_code",
                        "event_ids",
                        "export_id",
                        "identities",
                    }
                )
            ),
            name="external handoff receipt",
        )
        result = cls(
            operation=payload.get("operation", ""),
            status=payload.get("status", "ok"),
            run_status=payload.get("run_status", "running"),
            verdict=payload.get("verdict", HandoffApiVerdict.NONE.value),
            run_id=payload.get("run_id", ""),
            request_id=payload.get("request_id", ""),
            authority_id=payload.get("authority_id", ""),
            principal_id=payload.get("principal_id", ""),
            worker_principal_id=payload.get("worker_principal_id", ""),
            reviewer_principal_id=payload.get("reviewer_principal_id", ""),
            session_id=payload.get("session_id", ""),
            repository_id=payload.get("repository_id", ""),
            cursor=payload.get("cursor", ""),
            reason_code=payload.get("reason_code", "bound"),
            event_ids=payload.get("event_ids", ()),
            export_id=payload.get("export_id", ""),
            identities=payload.get("identities") or {},
        )
        _claimed_identity(
            payload,
            result.content_id,
            artifact_name="external handoff receipt",
            names=("content_id", "cid", "identity", "canonical_id", "receipt_id"),
        )
        return result


RequestLike = ExternalHandoffRequest | Mapping[str, Any]


@dataclass
class _RunRecord:
    """In-memory run.  Not a public contract and never serialized as authority."""

    run_id: str
    authority_id: str
    principal_id: str
    worker_principal_id: str
    session_id: str
    repository_id: str
    objective_id: str
    origin_operation: str
    admitted: bool
    run_status: str
    cursor: str
    events: list[dict[str, str]] = field(default_factory=list)
    idempotency_key: str = ""


def _mint_run_id(request: ExternalHandoffRequest) -> str:
    return content_identity(
        {
            "schema": EXTERNAL_HANDOFF_RUN_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "operation": request.operation,
            "principal_id": request.principal_id,
            "worker_principal_id": request.worker_principal_id,
            "session_id": request.session_id,
            "repository_id": request.repository_id,
            "objective_id": request.objective_id,
            "idempotency_key": request.idempotency_key,
        }
    )


def _mint_authority_id(*, principal_id: str, run_id: str) -> str:
    return content_identity(
        {
            "schema": EXTERNAL_HANDOFF_AUTHORITY_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "principal_id": principal_id,
            "run_id": run_id,
        }
    )


def _event_id(*, run_id: str, sequence: int, kind: str, detail: str = "") -> str:
    return content_identity(
        {
            "schema": EXTERNAL_HANDOFF_EVENT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "sequence": sequence,
            "kind": kind,
            "detail": detail,
        }
    )


def _export_id(*, run_id: str, request_id: str) -> str:
    return content_identity(
        {
            "schema": EXTERNAL_HANDOFF_EXPORT_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "run_id": run_id,
            "request_id": request_id,
            "disclosure_class": "public_projection",
        }
    )


def _from_handoff_family(value: Any, *, operation: str | None) -> ExternalHandoffRequest | None:
    mode = getattr(value, "mode", None)
    caller = getattr(value, "caller_principal_id", None)
    session_id = getattr(value, "session_id", None)
    if caller is None or session_id is None:
        return None
    mode_value = str(getattr(mode, "value", mode) or "").strip().lower()
    inferred = {
        "preview": "preview",
        "attach": "attach",
        "continue": "handoff",
        "import_only": "preview",
    }.get(mode_value, "handoff")
    return ExternalHandoffRequest(
        operation=operation or inferred,
        principal_id=str(caller),
        session_id=str(session_id),
        repository_id=str(getattr(value, "repository_id", "") or ""),
        objective_id=str(getattr(value, "objective_id", "") or ""),
        idempotency_key=str(getattr(value, "idempotency_key", "") or ""),
    )


def coerce_request(value: RequestLike, *, operation: str | None = None) -> ExternalHandoffRequest:
    """Accept a canonical request dict, dataclass, or EAAEF-010 handoff request."""

    if isinstance(value, ExternalHandoffRequest):
        if operation is not None and value.operation != operation:
            raise ExternalHandoffAPIError(
                "request operation does not match the invoked API method",
                reason_code="operation_mismatch",
            )
        return value
    if isinstance(value, Mapping):
        payload = dict(value)
        if operation is not None:
            supplied = payload.get("operation")
            if supplied in (None, ""):
                payload["operation"] = operation
            elif _operation_name(supplied) != operation:
                raise ExternalHandoffAPIError(
                    "request operation does not match the invoked API method",
                    reason_code="operation_mismatch",
                )
        elif payload.get("operation") in (None, ""):
            raise ExternalHandoffAPIError("operation is required", reason_code="malformed")
        return ExternalHandoffRequest.from_dict(payload)
    mapped = _from_handoff_family(value, operation=operation)
    if mapped is not None:
        if operation is not None and mapped.operation != operation:
            raise ExternalHandoffAPIError(
                "request operation does not match the invoked API method",
                reason_code="operation_mismatch",
            )
        return mapped
    raise ExternalHandoffAPIError(
        "handoff request must be a mapping or ExternalHandoffRequest",
        reason_code="malformed",
    )


class ExternalHandoffAPI:
    """In-process handoff API with an isolated in-memory run registry."""

    def __init__(self) -> None:
        self._runs: dict[str, _RunRecord] = {}
        self._idempotency: dict[tuple[str, str, str], str] = {}

    def invoke(self, request: RequestLike) -> ExternalHandoffReceipt:
        req = coerce_request(request)
        method = getattr(self, req.operation)
        return method(req)

    def handoff(self, request: RequestLike) -> ExternalHandoffReceipt:
        return self._origin(coerce_request(request, operation="handoff"), admitted=True)

    def preview(self, request: RequestLike) -> ExternalHandoffReceipt:
        return self._origin(coerce_request(request, operation="preview"), admitted=False)

    def attach(self, request: RequestLike) -> ExternalHandoffReceipt:
        req = coerce_request(request, operation="attach")
        run = self._lookup(req, require_authority=True)
        self._append(run, "attach")
        return self._receipt(req, run, reason_code="attached")

    def status(self, request: RequestLike) -> ExternalHandoffReceipt:
        req = coerce_request(request, operation="status")
        run = self._lookup(req, require_authority=False)
        return self._receipt(req, run, reason_code="status")

    def follow(self, request: RequestLike) -> ExternalHandoffReceipt:
        req = coerce_request(request, operation="follow")
        run = self._lookup(req, require_authority=False)
        event_ids = tuple(item["content_id"] for item in run.events)
        if req.cursor:
            try:
                index = event_ids.index(req.cursor) + 1
            except ValueError as exc:
                raise ExternalHandoffAPIError(
                    "follow cursor does not match the run event stream",
                    reason_code="unknown_cursor",
                ) from exc
            event_ids = event_ids[index:]
        cursor = event_ids[-1] if event_ids else run.cursor
        return self._receipt(
            req, run, reason_code="followed", event_ids=event_ids, cursor=cursor
        )

    def steer(self, request: RequestLike) -> ExternalHandoffReceipt:
        req = coerce_request(request, operation="steer")
        if not req.instruction:
            raise ExternalHandoffAPIError("instruction is required", reason_code="malformed")
        run = self._require_control(req)
        if run.run_status != "running":
            raise ExternalHandoffAPIError(
                "steer requires a running admitted run", reason_code="not_running"
            )
        self._append(run, "steer", req.instruction)
        return self._receipt(req, run, reason_code="steered")

    def pause(self, request: RequestLike) -> ExternalHandoffReceipt:
        req = coerce_request(request, operation="pause")
        run = self._require_control(req)
        if run.run_status != "running":
            raise ExternalHandoffAPIError(
                "pause requires a running run", reason_code="not_running"
            )
        run.run_status = "paused"
        self._append(run, "pause")
        return self._receipt(req, run, reason_code="paused")

    def resume(self, request: RequestLike) -> ExternalHandoffReceipt:
        req = coerce_request(request, operation="resume")
        run = self._require_control(req)
        if run.run_status != "paused":
            raise ExternalHandoffAPIError(
                "resume requires a paused run", reason_code="not_paused"
            )
        run.run_status = "running"
        self._append(run, "resume")
        return self._receipt(req, run, reason_code="resumed")

    def approve(self, request: RequestLike) -> ExternalHandoffReceipt:
        req = coerce_request(request, operation="approve")
        run = self._lookup(req, require_authority=False)
        reviewer = self._require_independent_reviewer(req, run)
        if run.run_status in _TERMINAL_STATUSES:
            raise ExternalHandoffAPIError(
                "run is already terminal", reason_code="terminal_run"
            )
        run.run_status = "approved"
        self._append(run, "approve", reviewer)
        return self._receipt(req, run, reason_code="approved", reviewer=reviewer)

    def reject(self, request: RequestLike) -> ExternalHandoffReceipt:
        req = coerce_request(request, operation="reject")
        run = self._lookup(req, require_authority=False)
        reviewer = self._require_independent_reviewer(req, run)
        if run.run_status in _TERMINAL_STATUSES:
            raise ExternalHandoffAPIError(
                "run is already terminal", reason_code="terminal_run"
            )
        run.run_status = "rejected"
        self._append(run, "reject", reviewer)
        return self._receipt(req, run, reason_code="rejected", reviewer=reviewer)

    def cancel(self, request: RequestLike) -> ExternalHandoffReceipt:
        req = coerce_request(request, operation="cancel")
        run = self._require_control(req)
        if run.run_status in _TERMINAL_STATUSES:
            raise ExternalHandoffAPIError(
                "run is already terminal", reason_code="terminal_run"
            )
        run.run_status = "cancelled"
        self._append(run, "cancel")
        return self._receipt(req, run, reason_code="cancelled")

    def explain(self, request: RequestLike) -> ExternalHandoffReceipt:
        req = coerce_request(request, operation="explain")
        run = self._lookup(req, require_authority=False)
        return self._receipt(req, run, reason_code="explained")

    def doctor(self, request: RequestLike) -> ExternalHandoffReceipt:
        req = coerce_request(request, operation="doctor")
        run = self._lookup(req, require_authority=False)
        return self._receipt(req, run, reason_code="diagnosed")

    def report(self, request: RequestLike) -> ExternalHandoffReceipt:
        req = coerce_request(request, operation="report")
        run = self._lookup(req, require_authority=False)
        return self._receipt(req, run, reason_code="reported")

    def export(self, request: RequestLike) -> ExternalHandoffReceipt:
        req = coerce_request(request, operation="export")
        run = self._lookup(req, require_authority=False)
        export_id = _export_id(run_id=run.run_id, request_id=req.request_id)
        return self._receipt(req, run, reason_code="exported", export_id=export_id)

    export_result = export

    def _origin(self, request: ExternalHandoffRequest, *, admitted: bool) -> ExternalHandoffReceipt:
        if request.idempotency_key:
            key = (request.operation, request.principal_id, request.idempotency_key)
            existing_id = self._idempotency.get(key)
            if existing_id is not None:
                return self._receipt(request, self._runs[existing_id], reason_code="idempotent")
        run_id = request.run_id or _mint_run_id(request)
        if run_id in self._runs:
            return self._receipt(request, self._runs[run_id], reason_code="idempotent")
        authority_id = request.authority_id or _mint_authority_id(
            principal_id=request.principal_id, run_id=run_id
        )
        run = _RunRecord(
            run_id=run_id,
            authority_id=authority_id,
            principal_id=request.principal_id,
            worker_principal_id=request.worker_principal_id,
            session_id=request.session_id,
            repository_id=request.repository_id,
            objective_id=request.objective_id,
            origin_operation=request.operation,
            admitted=admitted,
            run_status="running" if admitted else "preview_only",
            cursor="",
            idempotency_key=request.idempotency_key,
        )
        self._append(run, request.operation)
        self._runs[run_id] = run
        if request.idempotency_key:
            self._idempotency[(request.operation, request.principal_id, request.idempotency_key)] = (
                run_id
            )
        reason = "admitted" if admitted else "preview_only"
        return self._receipt(request, run, reason_code=reason)

    def _lookup(self, request: ExternalHandoffRequest, *, require_authority: bool) -> _RunRecord:
        if not request.run_id:
            raise ExternalHandoffAPIError("run_id is required", reason_code="malformed")
        run = self._runs.get(request.run_id)
        if run is None:
            raise ExternalHandoffAPIError("unknown run identity", reason_code="unknown_run")
        if require_authority:
            if not request.authority_id:
                raise ExternalHandoffAuthorityError(
                    "authority_id is required", reason_code="authority_mismatch"
                )
            if request.authority_id != run.authority_id:
                raise ExternalHandoffAuthorityError(
                    "run identity and authority id must match the admitted run",
                    reason_code="authority_mismatch",
                )
        return run

    def _require_control(self, request: ExternalHandoffRequest) -> _RunRecord:
        run = self._lookup(request, require_authority=True)
        if request.operation in _CONTROL_OPERATIONS and request.run_id != run.run_id:
            raise ExternalHandoffAuthorityError(
                "run identity and authority id must match the admitted run",
                reason_code="authority_mismatch",
            )
        return run

    def _require_independent_reviewer(self, request: ExternalHandoffRequest, run: _RunRecord) -> str:
        worker = request.worker_principal_id or run.worker_principal_id
        reviewer = request.reviewer_principal_id
        if not reviewer:
            raise WorkerSelfApprovalError(
                "approve and reject require an independent reviewer principal",
                reason_code="missing_reviewer",
            )
        if not worker:
            raise WorkerSelfApprovalError(
                "worker principal is required to bind independent review",
                reason_code="missing_worker",
            )
        if reviewer == worker or request.principal_id == worker:
            raise WorkerSelfApprovalError(
                "worker self-approval is forbidden",
                reason_code="worker_self_approval",
            )
        return reviewer

    def _append(self, run: _RunRecord, kind: str, detail: str = "") -> str:
        if len(run.events) >= MAX_EVENTS:
            raise ExternalHandoffAPIError("run event stream exceeds its bound", reason_code="bounds")
        sequence = len(run.events)
        event_id = _event_id(run_id=run.run_id, sequence=sequence, kind=kind, detail=detail)
        run.events.append({"content_id": event_id, "kind": kind})
        run.cursor = event_id
        return event_id

    def _receipt(
        self,
        request: ExternalHandoffRequest,
        run: _RunRecord,
        *,
        reason_code: str,
        event_ids: tuple[str, ...] | None = None,
        cursor: str | None = None,
        export_id: str = "",
        reviewer: str = "",
    ) -> ExternalHandoffReceipt:
        if run.origin_operation == "preview":
            verdict = HandoffApiVerdict.PREVIEW_ONLY.value
        elif run.admitted:
            verdict = HandoffApiVerdict.ADMITTED.value
        else:
            verdict = HandoffApiVerdict.NONE.value
        identities = {
            "run_id": run.run_id,
            "request_id": request.request_id,
            "authority_id": run.authority_id,
            "session_id": run.session_id,
        }
        if export_id:
            identities["export_id"] = export_id
        return ExternalHandoffReceipt(
            operation=request.operation,
            status="ok",
            run_status=run.run_status,
            verdict=verdict,
            run_id=run.run_id,
            request_id=request.request_id,
            authority_id=run.authority_id,
            principal_id=request.principal_id,
            worker_principal_id=run.worker_principal_id,
            reviewer_principal_id=reviewer or request.reviewer_principal_id,
            session_id=run.session_id,
            repository_id=run.repository_id,
            cursor=run.cursor if cursor is None else cursor,
            reason_code=reason_code,
            event_ids=event_ids if event_ids is not None else tuple(item["content_id"] for item in run.events),
            export_id=export_id,
            identities=identities,
        )


_DEFAULT_API: ExternalHandoffAPI | None = None


def get_default_api() -> ExternalHandoffAPI:
    """Return the process-local in-memory API registry."""

    global _DEFAULT_API
    if _DEFAULT_API is None:
        _DEFAULT_API = ExternalHandoffAPI()
    return _DEFAULT_API


def reset_default_api() -> ExternalHandoffAPI:
    """Replace the process-local registry.  Tests use a dedicated instance."""

    global _DEFAULT_API
    _DEFAULT_API = ExternalHandoffAPI()
    return _DEFAULT_API


def invoke(request: RequestLike, *, api: ExternalHandoffAPI | None = None) -> ExternalHandoffReceipt:
    return (api or get_default_api()).invoke(request)


def handoff(request: RequestLike, *, api: ExternalHandoffAPI | None = None) -> ExternalHandoffReceipt:
    return (api or get_default_api()).handoff(request)


def preview(request: RequestLike, *, api: ExternalHandoffAPI | None = None) -> ExternalHandoffReceipt:
    return (api or get_default_api()).preview(request)


def attach(request: RequestLike, *, api: ExternalHandoffAPI | None = None) -> ExternalHandoffReceipt:
    return (api or get_default_api()).attach(request)


def status(request: RequestLike, *, api: ExternalHandoffAPI | None = None) -> ExternalHandoffReceipt:
    return (api or get_default_api()).status(request)


def follow(request: RequestLike, *, api: ExternalHandoffAPI | None = None) -> ExternalHandoffReceipt:
    return (api or get_default_api()).follow(request)


def steer(request: RequestLike, *, api: ExternalHandoffAPI | None = None) -> ExternalHandoffReceipt:
    return (api or get_default_api()).steer(request)


def pause(request: RequestLike, *, api: ExternalHandoffAPI | None = None) -> ExternalHandoffReceipt:
    return (api or get_default_api()).pause(request)


def resume(request: RequestLike, *, api: ExternalHandoffAPI | None = None) -> ExternalHandoffReceipt:
    return (api or get_default_api()).resume(request)


def approve(request: RequestLike, *, api: ExternalHandoffAPI | None = None) -> ExternalHandoffReceipt:
    return (api or get_default_api()).approve(request)


def reject(request: RequestLike, *, api: ExternalHandoffAPI | None = None) -> ExternalHandoffReceipt:
    return (api or get_default_api()).reject(request)


def cancel(request: RequestLike, *, api: ExternalHandoffAPI | None = None) -> ExternalHandoffReceipt:
    return (api or get_default_api()).cancel(request)


def explain(request: RequestLike, *, api: ExternalHandoffAPI | None = None) -> ExternalHandoffReceipt:
    return (api or get_default_api()).explain(request)


def doctor(request: RequestLike, *, api: ExternalHandoffAPI | None = None) -> ExternalHandoffReceipt:
    return (api or get_default_api()).doctor(request)


def report(request: RequestLike, *, api: ExternalHandoffAPI | None = None) -> ExternalHandoffReceipt:
    return (api or get_default_api()).report(request)


def export(request: RequestLike, *, api: ExternalHandoffAPI | None = None) -> ExternalHandoffReceipt:
    return (api or get_default_api()).export(request)


export_result = export


def discover_external_handoff_api() -> dict[str, Any]:
    """Return the frozen in-process operation catalog without side effects."""

    return {
        "schema": EXTERNAL_HANDOFF_API_SCHEMA,
        "interface": EXTERNAL_HANDOFF_API_INTERFACE,
        "contract_version": CONTRACT_VERSION,
        "operations": list(HANDOFF_API_OPERATIONS),
        "preview_is_handoff": False,
        "self_approval": False,
        "live_quack": False,
        "live_docker": False,
        "registry": "in_memory",
    }


__all__ = (
    "CONTRACT_VERSION",
    "EXTERNAL_HANDOFF_API_INTERFACE",
    "EXTERNAL_HANDOFF_API_SCHEMA",
    "EXTERNAL_HANDOFF_RECEIPT_INTERFACE",
    "EXTERNAL_HANDOFF_RECEIPT_SCHEMA",
    "EXTERNAL_HANDOFF_REQUEST_INTERFACE",
    "EXTERNAL_HANDOFF_REQUEST_SCHEMA",
    "HANDOFF_API_OPERATIONS",
    "OPERATIONS",
    "SCHEMA_VERSION",
    "ExternalHandoffAPI",
    "ExternalHandoffAPIError",
    "ExternalHandoffAuthorityError",
    "ExternalHandoffReceipt",
    "ExternalHandoffRequest",
    "HandoffApiOperation",
    "HandoffApiVerdict",
    "WorkerSelfApprovalError",
    "approve",
    "attach",
    "cancel",
    "coerce_request",
    "discover_external_handoff_api",
    "doctor",
    "explain",
    "export",
    "export_result",
    "follow",
    "get_default_api",
    "handoff",
    "invoke",
    "pause",
    "preview",
    "reject",
    "report",
    "reset_default_api",
    "resume",
    "status",
    "steer",
)
