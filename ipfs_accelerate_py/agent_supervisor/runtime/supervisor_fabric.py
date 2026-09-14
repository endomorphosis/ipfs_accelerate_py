"""Supervisor capability and fenced coordination contracts.

Sibling-supervisor event validation, the sibling-supervisor capability
registry, and cross-supervisor receipts are bindings of this fabric, not a
second event log, bus, registry, receipt store, database, or state owner.
Sibling supervisors exchange canonical event envelopes and receipts. They
never write DuckDB or DuckLake, never consume ``DatabaseEventLog@1``, and
never terminalize tasks. A worker or model assertion cannot admit a sibling
event, a sibling capability, or a cross-supervisor receipt. The capability
registry is an in-process admitted catalogue only; it is not a competing subsystem
and grants no completion, mutation, write, or proof authority.
Cross-supervisor receipts are in-process admitted envelopes bound to
``receipt-exchange``; they are not a second receipt log and grant no
completion, mutation, write, or proof authority.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final


class SupervisorFabricError(ValueError):
    """A fenced coordination contract was violated."""


class SiblingSupervisorEventValidationError(SupervisorFabricError):
    """A sibling-supervisor event failed operational admission."""

    def __init__(self, message: str, *, code: str = "sibling_event_invalid") -> None:
        super().__init__(message)
        self.code = code


class SiblingSupervisorCapabilityRegistryError(SupervisorFabricError):
    """A sibling-supervisor capability failed registry admission or lookup."""

    def __init__(self, message: str, *, code: str = "sibling_capability_invalid") -> None:
        super().__init__(message)
        self.code = code


class CrossSupervisorReceiptError(SupervisorFabricError):
    """A cross-supervisor receipt failed operational admission."""

    def __init__(self, message: str, *, code: str = "receipt_invalid") -> None:
        super().__init__(message)
        self.code = code


SUPERVISOR_FABRIC_INTERFACE: Final[str] = "SupervisorFabric@1"
SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING: Final[str] = (
    "SiblingSupervisorEventValidation@1"
)
SIBLING_SUPERVISOR_EVENT_VALIDATION_INTERFACE: Final[str] = (
    SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING
)
SIBLING_SUPERVISOR_EVENT_VALIDATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/sibling-supervisor-event-validation@1"
)
CANONICAL_EVENT_INTERFACE: Final[str] = "CanonicalEvent@1"
CANONICAL_EVENT_SCHEMA_ID: Final[str] = (
    "ipfs_datasets_py/logic/ir-core/canonical-event@1"
)
DATABASE_EVENT_LOG_INTERFACE: Final[str] = "DatabaseEventLog@1"
EVENT_CURSOR_INTERFACE: Final[str] = "EventCursor@1"

SIBLING_SUPERVISOR_EVENT_VALIDATION_CONSUMES: Final[tuple[str, ...]] = (
    SUPERVISOR_FABRIC_INTERFACE,
    CANONICAL_EVENT_INTERFACE,
    DATABASE_EVENT_LOG_INTERFACE,
)
SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING: Final[str] = (
    "SiblingSupervisorCapabilityRegistry@1"
)
SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_INTERFACE: Final[str] = (
    SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING
)
SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/sibling-supervisor-capability-registry@1"
)
SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_CONSUMES: Final[tuple[str, ...]] = (
    SUPERVISOR_FABRIC_INTERFACE,
    SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING,
)
CROSS_SUPERVISOR_RECEIPT_BINDING: Final[str] = "CrossSupervisorReceipt@1"
CROSS_SUPERVISOR_RECEIPT_INTERFACE: Final[str] = CROSS_SUPERVISOR_RECEIPT_BINDING
CROSS_SUPERVISOR_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/cross-supervisor-receipt@1"
)
CROSS_SUPERVISOR_RECEIPT_CAPABILITY: Final[str] = "receipt-exchange"
CROSS_SUPERVISOR_RECEIPT_CARRIER: Final[str] = "event"
CROSS_SUPERVISOR_RECEIPT_CONSUMES: Final[tuple[str, ...]] = (
    SUPERVISOR_FABRIC_INTERFACE,
    SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING,
    SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING,
)

CANONICAL_EVENT_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "event_id",
        "event_type",
        "stream_id",
        "causal_parent_ids",
        "correlation_id",
        "causation_id",
        "payload",
    }
)
CANONICAL_EVENT_FORBIDDEN_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "authorization_decision",
        "completion_decision",
        "lease_id",
        "fencing_epoch",
        "policy_id",
        "policy_revision",
    }
)

ALLOWED_SIBLING_EFFECTS: Final[frozenset[str]] = frozenset(
    {"", "none", "read_only", "event_exchange"}
)
ALLOWED_SIBLING_CAPABILITIES: Final[frozenset[str]] = frozenset(
    {"event-exchange", "task-request", "receipt-exchange"}
)
FORBIDDEN_SIBLING_CAPABILITIES: Final[frozenset[str]] = frozenset(
    {
        "completion-authority",
        "database-write",
        "direct-state-write",
        "duckdb-write",
        "ducklake-write",
        "owner-mutation",
        "policy-pointer",
        "terminalize-task",
    }
)
_MAX_SIBLING_CAPABILITY_RECORDS: Final[int] = 128
FORBIDDEN_SIBLING_WRITE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "database_path",
        "database_write",
        "direct_state_write",
        "duckdb_path",
        "ducklake_path",
        "owner_mutation",
        "policy_pointer",
        "sql",
        "state_write",
        "terminalize_task",
        "write_database",
    }
)
FORBIDDEN_SIBLING_LOG_MUTATIONS: Final[frozenset[str]] = frozenset(
    {
        "append_event",
        "consume",
        "mark_consumed",
        "save_consumer_checkpoint",
    }
)
CROSS_SUPERVISOR_RECEIPT_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "receipt_id",
        "task_id",
        "request_id",
        "carrier_event_id",
        "outcome",
        "evidence_digest",
        "payload",
    }
)
CROSS_SUPERVISOR_RECEIPT_FORBIDDEN_FIELDS: Final[frozenset[str]] = (
    CANONICAL_EVENT_FORBIDDEN_FIELDS
    | FORBIDDEN_SIBLING_WRITE_KEYS
    | FORBIDDEN_SIBLING_LOG_MUTATIONS
    | {
        "completion_authoritative",
        "database_write",
        "direct_state_write",
        "terminalize_task",
    }
)
CROSS_SUPERVISOR_RECEIPT_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"admitted", "rejected", "unknown"}
)
_MAX_IDENTIFIER_CHARS: Final[int] = 512
_MAX_SIBLING_RECEIPT_PAYLOAD_BYTES: Final[int] = 65536


def issue_fence(record: Mapping[str, Any]) -> Mapping[str, Any]:
    if not record.get("supervisor_id") or not record.get("capability"):
        raise SupervisorFabricError("supervisor capability is required")
    if record.get("stale_epoch"):
        raise SupervisorFabricError("stale fence epoch")
    return MappingProxyType(
        {
            "supervisor_id": record["supervisor_id"],
            "capability": record["capability"],
            "epoch": int(record.get("epoch") or 1),
            "fenced": True,
        }
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = True,
    error_cls: type[SupervisorFabricError] = SiblingSupervisorEventValidationError,
    code: str = "sibling_identity_mismatch",
) -> str:
    if value is None:
        if required:
            raise error_cls(f"{field_name} is required", code=code)
        return ""
    if not isinstance(value, str):
        raise error_cls(f"{field_name} must be a string", code=code)
    if not value:
        if required:
            raise error_cls(f"{field_name} must not be empty", code=code)
        return ""
    if value != value.strip():
        raise error_cls(
            f"{field_name} must be exact and contain no surrounding whitespace",
            code=code,
        )
    if len(value) > _MAX_IDENTIFIER_CHARS:
        raise error_cls(
            f"{field_name} must not exceed {_MAX_IDENTIFIER_CHARS} characters",
            code=code,
        )
    if any(character.isspace() or not character.isprintable() for character in value):
        raise error_cls(
            f"{field_name} must contain only printable, non-whitespace characters",
            code=code,
        )
    return value


def _event_identifier(value: Any, field_name: str) -> str:
    try:
        return _text(value, field_name)
    except SiblingSupervisorEventValidationError as error:
        raise SiblingSupervisorEventValidationError(
            str(error),
            code="canonical_event_invalid",
        ) from error


def _reject_direct_state_writes(
    record: Mapping[str, Any],
    *,
    error_cls: type[SupervisorFabricError] = SiblingSupervisorEventValidationError,
    subject: str = "sibling events",
) -> None:
    present = sorted(key for key in FORBIDDEN_SIBLING_WRITE_KEYS if record.get(key))
    if present:
        raise error_cls(
            "sibling supervisors exchange events, never database writes: "
            + ", ".join(present),
            code="direct_state_write",
        )
    mutations = sorted(
        key for key in FORBIDDEN_SIBLING_LOG_MUTATIONS if record.get(key)
    )
    if mutations:
        raise error_cls(
            "sibling events cannot mutate DatabaseEventLog@1: " + ", ".join(mutations),
            code="direct_state_write",
        )
    effect = record.get("effect")
    if effect is None:
        return
    if not isinstance(effect, str) or effect not in ALLOWED_SIBLING_EFFECTS:
        raise error_cls(
            f"{subject} effect {effect!r} is not admitted",
            code="forbidden_effect",
        )


def _canonical_event_wire(event: Any) -> dict[str, Any]:
    if not isinstance(event, Mapping):
        raise SiblingSupervisorEventValidationError(
            "canonical event must be an object",
            code="canonical_event_invalid",
        )
    fields = set(event)
    forbidden = fields & CANONICAL_EVENT_FORBIDDEN_FIELDS
    if forbidden:
        raise SiblingSupervisorEventValidationError(
            "canonical event contains operational authority field(s): "
            + ", ".join(sorted(forbidden)),
            code="operational_authority_field",
        )
    missing = CANONICAL_EVENT_REQUIRED_FIELDS - fields
    extra = fields - CANONICAL_EVENT_REQUIRED_FIELDS
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unknown " + ", ".join(sorted(extra)))
        raise SiblingSupervisorEventValidationError(
            "canonical event fields: " + "; ".join(details),
            code="canonical_event_invalid",
        )
    schema = event["schema"]
    if schema != CANONICAL_EVENT_SCHEMA_ID:
        raise SiblingSupervisorEventValidationError(
            f"unsupported canonical event schema {schema!r}",
            code="canonical_event_schema",
        )
    event_id = _event_identifier(event["event_id"], "event_id")
    parents = event["causal_parent_ids"]
    if not isinstance(parents, (list, tuple)):
        raise SiblingSupervisorEventValidationError(
            "causal_parent_ids must be an array",
            code="canonical_event_invalid",
        )
    parent_ids = tuple(
        _event_identifier(parent_id, "causal_parent_id") for parent_id in parents
    )
    if len(parent_ids) != len(set(parent_ids)):
        raise SiblingSupervisorEventValidationError(
            "causal_parent_ids must be unique",
            code="canonical_event_invalid",
        )
    if event_id in parent_ids:
        raise SiblingSupervisorEventValidationError(
            "an event cannot be its own causal parent",
            code="canonical_event_invalid",
        )
    payload = event["payload"]
    if not isinstance(payload, Mapping):
        raise SiblingSupervisorEventValidationError(
            "payload must be an object",
            code="canonical_event_invalid",
        )
    try:
        detached_payload = json.loads(_canonical_json(dict(payload)))
    except (TypeError, ValueError) as error:
        raise SiblingSupervisorEventValidationError(
            f"invalid event payload: {error}",
            code="canonical_event_invalid",
        ) from error
    if not isinstance(detached_payload, dict):
        raise SiblingSupervisorEventValidationError(
            "payload must be an object",
            code="canonical_event_invalid",
        )
    return {
        "causal_parent_ids": list(parent_ids),
        "causation_id": _event_identifier(event["causation_id"], "causation_id"),
        "correlation_id": _event_identifier(event["correlation_id"], "correlation_id"),
        "event_id": event_id,
        "event_type": _event_identifier(event["event_type"], "event_type"),
        "payload": detached_payload,
        "schema": CANONICAL_EVENT_SCHEMA_ID,
        "stream_id": _event_identifier(event["stream_id"], "stream_id"),
    }


@dataclass(frozen=True)
class SiblingSupervisorEventAdmission:
    """Non-authoritative admission of one sibling-supervisor event envelope."""

    local_supervisor_id: str
    sibling_supervisor_id: str
    event_id: str
    event_type: str
    stream_id: str
    epoch: int
    capability: str
    event_digest: str
    effect: str = "none"
    schema: str = SIBLING_SUPERVISOR_EVENT_VALIDATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIBLING_SUPERVISOR_EVENT_VALIDATION_SCHEMA:
            raise SiblingSupervisorEventValidationError(
                f"unsupported sibling event admission schema {self.schema!r}",
                code="admission_schema",
            )
        object.__setattr__(
            self,
            "local_supervisor_id",
            _text(self.local_supervisor_id, "local_supervisor_id"),
        )
        object.__setattr__(
            self,
            "sibling_supervisor_id",
            _text(self.sibling_supervisor_id, "sibling_supervisor_id"),
        )
        if self.local_supervisor_id == self.sibling_supervisor_id:
            raise SiblingSupervisorEventValidationError(
                "a supervisor is not its own sibling",
                code="not_a_sibling",
            )
        object.__setattr__(self, "event_id", _event_identifier(self.event_id, "event_id"))
        object.__setattr__(
            self, "event_type", _event_identifier(self.event_type, "event_type")
        )
        object.__setattr__(
            self, "stream_id", _event_identifier(self.stream_id, "stream_id")
        )
        object.__setattr__(self, "capability", _text(self.capability, "capability"))
        object.__setattr__(self, "event_digest", _text(self.event_digest, "event_digest"))
        epoch = int(self.epoch)
        if epoch < 1:
            raise SiblingSupervisorEventValidationError(
                "epoch must be >= 1",
                code="stale_fence_epoch",
            )
        object.__setattr__(self, "epoch", epoch)
        effect = self.effect or "none"
        if effect not in ALLOWED_SIBLING_EFFECTS:
            raise SiblingSupervisorEventValidationError(
                f"sibling event effect {effect!r} is not admitted",
                code="forbidden_effect",
            )
        object.__setattr__(self, "effect", effect)

    @property
    def logical_once_key(self) -> str:
        return f"{self.sibling_supervisor_id}:{self.event_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "binding": SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING,
            "interface": SIBLING_SUPERVISOR_EVENT_VALIDATION_INTERFACE,
            "carrier": SUPERVISOR_FABRIC_INTERFACE,
            "consumes": {
                "supervisor_fabric": SUPERVISOR_FABRIC_INTERFACE,
                "canonical_event": CANONICAL_EVENT_INTERFACE,
                "database_event_log": DATABASE_EVENT_LOG_INTERFACE,
            },
            "local_supervisor_id": self.local_supervisor_id,
            "sibling_supervisor_id": self.sibling_supervisor_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "stream_id": self.stream_id,
            "epoch": self.epoch,
            "capability": self.capability,
            "effect": self.effect,
            "event_digest": self.event_digest,
            "logical_once_key": self.logical_once_key,
            "fenced": True,
            "admitted": True,
            "database_write": False,
            "completion_authoritative": False,
            "worker_assertion_is_authority": False,
            "worker_completion_insufficient": True,
        }


def validate_sibling_supervisor_event(
    record: Mapping[str, Any],
) -> SiblingSupervisorEventAdmission:
    """Admit one sibling event envelope without writing or consuming state."""

    if not isinstance(record, Mapping):
        raise SiblingSupervisorEventValidationError(
            "sibling event record must be an object",
            code="record_invalid",
        )
    _reject_direct_state_writes(record)
    if record.get("completion_authoritative"):
        raise SiblingSupervisorEventValidationError(
            "sibling event admission is not completion authority",
            code="completion_not_authoritative",
        )
    local_supervisor_id = _text(record.get("local_supervisor_id"), "local_supervisor_id")
    sibling_supervisor_id = _text(
        record.get("sibling_supervisor_id") or record.get("supervisor_id"),
        "sibling_supervisor_id",
    )
    if sibling_supervisor_id == local_supervisor_id:
        raise SiblingSupervisorEventValidationError(
            "a supervisor is not its own sibling",
            code="not_a_sibling",
        )
    known = record.get("known_sibling_ids")
    if known is not None:
        if not isinstance(known, Sequence) or isinstance(known, (str, bytes)):
            raise SiblingSupervisorEventValidationError(
                "known_sibling_ids must be a sequence of supervisor ids",
                code="unknown_sibling",
            )
        if sibling_supervisor_id not in tuple(known):
            raise SiblingSupervisorEventValidationError(
                f"unknown sibling supervisor {sibling_supervisor_id!r}",
                code="unknown_sibling",
            )
    fence = issue_fence(
        {
            "supervisor_id": sibling_supervisor_id,
            "capability": record.get("capability"),
            "epoch": record.get("epoch"),
            "stale_epoch": record.get("stale_epoch"),
        }
    )
    event = _canonical_event_wire(record.get("event"))
    admission = SiblingSupervisorEventAdmission(
        local_supervisor_id=local_supervisor_id,
        sibling_supervisor_id=str(fence["supervisor_id"]),
        event_id=str(event["event_id"]),
        event_type=str(event["event_type"]),
        stream_id=str(event["stream_id"]),
        epoch=int(fence["epoch"]),
        capability=str(fence["capability"]),
        event_digest=_sha256_hex(_canonical_json(event).encode("utf-8")),
        effect=str(record.get("effect") or "none"),
    )
    # Worker assertions are recorded as non-authority; they never change admission.
    _ = bool(record.get("worker_assertion"))
    return admission


def _capability_text(value: Any, field_name: str, *, required: bool = True) -> str:
    return _text(
        value,
        field_name,
        required=required,
        error_cls=SiblingSupervisorCapabilityRegistryError,
        code="sibling_identity_mismatch",
    )


def _admit_sibling_effect(
    value: Any,
    *,
    error_cls: type[SupervisorFabricError],
) -> str:
    effect = value or "none"
    if not isinstance(effect, str) or effect not in ALLOWED_SIBLING_EFFECTS:
        raise error_cls(
            f"sibling event effect {effect!r} is not admitted",
            code="forbidden_effect",
        )
    return effect or "none"


@dataclass(frozen=True)
class SiblingSupervisorCapabilityAdmission:
    """Non-authoritative admission of one sibling-supervisor capability."""

    local_supervisor_id: str
    sibling_supervisor_id: str
    capability: str
    epoch: int
    record_digest: str
    effect: str = "none"
    schema: str = SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA:
            raise SiblingSupervisorCapabilityRegistryError(
                f"unsupported sibling capability admission schema {self.schema!r}",
                code="admission_schema",
            )
        object.__setattr__(
            self,
            "local_supervisor_id",
            _capability_text(self.local_supervisor_id, "local_supervisor_id"),
        )
        object.__setattr__(
            self,
            "sibling_supervisor_id",
            _capability_text(self.sibling_supervisor_id, "sibling_supervisor_id"),
        )
        if self.local_supervisor_id == self.sibling_supervisor_id:
            raise SiblingSupervisorCapabilityRegistryError(
                "a supervisor is not its own sibling",
                code="not_a_sibling",
            )
        capability = _capability_text(self.capability, "capability")
        if capability in FORBIDDEN_SIBLING_CAPABILITIES:
            raise SiblingSupervisorCapabilityRegistryError(
                f"sibling capability {capability!r} is forbidden",
                code="forbidden_capability",
            )
        if capability not in ALLOWED_SIBLING_CAPABILITIES:
            raise SiblingSupervisorCapabilityRegistryError(
                f"unknown sibling capability {capability!r}",
                code="unknown_capability",
            )
        object.__setattr__(self, "capability", capability)
        object.__setattr__(
            self, "record_digest", _capability_text(self.record_digest, "record_digest")
        )
        epoch = int(self.epoch)
        if epoch < 1:
            raise SiblingSupervisorCapabilityRegistryError(
                "epoch must be >= 1",
                code="stale_fence_epoch",
            )
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(
            self,
            "effect",
            _admit_sibling_effect(
                self.effect,
                error_cls=SiblingSupervisorCapabilityRegistryError,
            ),
        )

    @property
    def registry_key(self) -> str:
        return f"{self.sibling_supervisor_id}:{self.capability}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "binding": SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING,
            "interface": SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_INTERFACE,
            "carrier": SUPERVISOR_FABRIC_INTERFACE,
            "consumes": {
                "supervisor_fabric": SUPERVISOR_FABRIC_INTERFACE,
                "sibling_event_validation": SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING,
            },
            "local_supervisor_id": self.local_supervisor_id,
            "sibling_supervisor_id": self.sibling_supervisor_id,
            "capability": self.capability,
            "epoch": self.epoch,
            "effect": self.effect,
            "record_digest": self.record_digest,
            "registry_key": self.registry_key,
            "fenced": True,
            "admitted": True,
            "database_write": False,
            "completion_authoritative": False,
            "worker_assertion_is_authority": False,
            "worker_completion_insufficient": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SiblingSupervisorCapabilityAdmission:
        if not isinstance(payload, Mapping):
            raise SiblingSupervisorCapabilityRegistryError(
                "sibling capability admission must be an object",
                code="record_invalid",
            )
        return cls(
            local_supervisor_id=str(payload.get("local_supervisor_id") or ""),
            sibling_supervisor_id=str(payload.get("sibling_supervisor_id") or ""),
            capability=str(payload.get("capability") or ""),
            epoch=int(payload.get("epoch") or 0),
            record_digest=str(payload.get("record_digest") or ""),
            effect=str(payload.get("effect") or "none"),
            schema=str(
                payload.get("schema") or SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA
            ),
        )


@dataclass(frozen=True)
class SiblingSupervisorCapabilityRegistry:
    """Admitted in-process sibling capabilities. Not a database, bus, or owner."""

    local_supervisor_id: str
    epoch: int
    records: tuple[SiblingSupervisorCapabilityAdmission, ...] = ()
    schema: str = SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA:
            raise SiblingSupervisorCapabilityRegistryError(
                f"unsupported sibling capability registry schema {self.schema!r}",
                code="admission_schema",
            )
        object.__setattr__(
            self,
            "local_supervisor_id",
            _capability_text(self.local_supervisor_id, "local_supervisor_id"),
        )
        epoch = int(self.epoch)
        if epoch < 1:
            raise SiblingSupervisorCapabilityRegistryError(
                "epoch must be >= 1",
                code="stale_fence_epoch",
            )
        object.__setattr__(self, "epoch", epoch)
        admitted: list[SiblingSupervisorCapabilityAdmission] = []
        seen: dict[str, SiblingSupervisorCapabilityAdmission] = {}
        for item in self.records:
            if not isinstance(item, SiblingSupervisorCapabilityAdmission):
                raise SiblingSupervisorCapabilityRegistryError(
                    "registry records must be capability admissions",
                    code="record_invalid",
                )
            if item.local_supervisor_id != self.local_supervisor_id:
                raise SiblingSupervisorCapabilityRegistryError(
                    "capability record local supervisor does not match the registry",
                    code="sibling_identity_mismatch",
                )
            if item.epoch < self.epoch:
                raise SiblingSupervisorCapabilityRegistryError(
                    "stale fence epoch",
                    code="stale_fence_epoch",
                )
            existing = seen.get(item.registry_key)
            if existing is not None:
                if existing.record_digest != item.record_digest:
                    raise SiblingSupervisorCapabilityRegistryError(
                        f"conflicting sibling capability {item.registry_key!r}",
                        code="capability_conflict",
                    )
                continue
            seen[item.registry_key] = item
            admitted.append(item)
        if len(admitted) > _MAX_SIBLING_CAPABILITY_RECORDS:
            raise SiblingSupervisorCapabilityRegistryError(
                "sibling capability registry exceeds the admitted bound",
                code="registry_bound",
            )
        object.__setattr__(
            self,
            "records",
            tuple(sorted(admitted, key=lambda item: item.registry_key)),
        )

    @property
    def sibling_supervisor_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(item.sibling_supervisor_id for item in self.records)
        )

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(sorted({item.capability for item in self.records}))

    def lookup(
        self, sibling_supervisor_id: str, capability: str
    ) -> SiblingSupervisorCapabilityAdmission:
        sibling_supervisor_id = _capability_text(
            sibling_supervisor_id, "sibling_supervisor_id"
        )
        capability = _capability_text(capability, "capability")
        key = f"{sibling_supervisor_id}:{capability}"
        for item in self.records:
            if item.registry_key == key:
                if item.epoch < self.epoch:
                    raise SiblingSupervisorCapabilityRegistryError(
                        "stale fence epoch",
                        code="stale_fence_epoch",
                    )
                return item
        raise SiblingSupervisorCapabilityRegistryError(
            f"unknown sibling capability {key!r}",
            code="unknown_capability",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "binding": SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING,
            "interface": SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_INTERFACE,
            "carrier": SUPERVISOR_FABRIC_INTERFACE,
            "consumes": {
                "supervisor_fabric": SUPERVISOR_FABRIC_INTERFACE,
                "sibling_event_validation": SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING,
            },
            "local_supervisor_id": self.local_supervisor_id,
            "epoch": self.epoch,
            "records": [item.to_dict() for item in self.records],
            "sibling_supervisor_ids": list(self.sibling_supervisor_ids),
            "capabilities": list(self.capabilities),
            "fenced": True,
            "admitted": True,
            "database_write": False,
            "completion_authoritative": False,
            "worker_assertion_is_authority": False,
            "worker_completion_insufficient": True,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> SiblingSupervisorCapabilityRegistry:
        if not isinstance(payload, Mapping):
            raise SiblingSupervisorCapabilityRegistryError(
                "sibling capability registry must be an object",
                code="record_invalid",
            )
        records = payload.get("records") or ()
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise SiblingSupervisorCapabilityRegistryError(
                "registry records must be a sequence",
                code="record_invalid",
            )
        return cls(
            local_supervisor_id=str(payload.get("local_supervisor_id") or ""),
            epoch=int(payload.get("epoch") or 0),
            records=tuple(
                SiblingSupervisorCapabilityAdmission.from_dict(item)
                if not isinstance(item, SiblingSupervisorCapabilityAdmission)
                else item
                for item in records
            ),
            schema=str(
                payload.get("schema") or SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA
            ),
        )


def register_sibling_supervisor_capability(
    record: Mapping[str, Any],
) -> SiblingSupervisorCapabilityAdmission:
    """Admit one sibling capability without writing or consuming state."""

    if not isinstance(record, Mapping):
        raise SiblingSupervisorCapabilityRegistryError(
            "sibling capability record must be an object",
            code="record_invalid",
        )
    _reject_direct_state_writes(
        record,
        error_cls=SiblingSupervisorCapabilityRegistryError,
        subject="sibling capabilities",
    )
    if record.get("completion_authoritative"):
        raise SiblingSupervisorCapabilityRegistryError(
            "sibling capability admission is not completion authority",
            code="completion_not_authoritative",
        )
    local_supervisor_id = _capability_text(
        record.get("local_supervisor_id"), "local_supervisor_id"
    )
    sibling_supervisor_id = _capability_text(
        record.get("sibling_supervisor_id") or record.get("supervisor_id"),
        "sibling_supervisor_id",
    )
    if sibling_supervisor_id == local_supervisor_id:
        raise SiblingSupervisorCapabilityRegistryError(
            "a supervisor is not its own sibling",
            code="not_a_sibling",
        )
    known = record.get("known_sibling_ids")
    if known is not None:
        if not isinstance(known, Sequence) or isinstance(known, (str, bytes)):
            raise SiblingSupervisorCapabilityRegistryError(
                "known_sibling_ids must be a sequence of supervisor ids",
                code="unknown_sibling",
            )
        if sibling_supervisor_id not in tuple(known):
            raise SiblingSupervisorCapabilityRegistryError(
                f"unknown sibling supervisor {sibling_supervisor_id!r}",
                code="unknown_sibling",
            )
    capability = record.get("capability")
    if isinstance(capability, str) and capability in FORBIDDEN_SIBLING_CAPABILITIES:
        raise SiblingSupervisorCapabilityRegistryError(
            f"sibling capability {capability!r} is forbidden",
            code="forbidden_capability",
        )
    if isinstance(capability, str) and capability and capability not in ALLOWED_SIBLING_CAPABILITIES:
        raise SiblingSupervisorCapabilityRegistryError(
            f"unknown sibling capability {capability!r}",
            code="unknown_capability",
        )
    current_epoch = record.get("current_epoch")
    if current_epoch is not None:
        try:
            current = int(current_epoch)
        except (TypeError, ValueError) as error:
            raise SiblingSupervisorCapabilityRegistryError(
                "current_epoch must be an integer >= 1",
                code="stale_fence_epoch",
            ) from error
        if current < 1:
            raise SiblingSupervisorCapabilityRegistryError(
                "current_epoch must be >= 1",
                code="stale_fence_epoch",
            )
        try:
            offered = int(record.get("epoch") or 0)
        except (TypeError, ValueError) as error:
            raise SiblingSupervisorCapabilityRegistryError(
                "epoch must be an integer >= 1",
                code="stale_fence_epoch",
            ) from error
        if offered < current:
            raise SiblingSupervisorCapabilityRegistryError(
                "stale fence epoch",
                code="stale_fence_epoch",
            )
    fence = issue_fence(
        {
            "supervisor_id": sibling_supervisor_id,
            "capability": capability,
            "epoch": record.get("epoch"),
            "stale_epoch": record.get("stale_epoch"),
        }
    )
    admitted_capability = str(fence["capability"])
    if admitted_capability in FORBIDDEN_SIBLING_CAPABILITIES:
        raise SiblingSupervisorCapabilityRegistryError(
            f"sibling capability {admitted_capability!r} is forbidden",
            code="forbidden_capability",
        )
    if admitted_capability not in ALLOWED_SIBLING_CAPABILITIES:
        raise SiblingSupervisorCapabilityRegistryError(
            f"unknown sibling capability {admitted_capability!r}",
            code="unknown_capability",
        )
    canonical = {
        "capability": admitted_capability,
        "effect": _admit_sibling_effect(
            record.get("effect"),
            error_cls=SiblingSupervisorCapabilityRegistryError,
        ),
        "epoch": int(fence["epoch"]),
        "local_supervisor_id": local_supervisor_id,
        "sibling_supervisor_id": str(fence["supervisor_id"]),
    }
    admission = SiblingSupervisorCapabilityAdmission(
        local_supervisor_id=canonical["local_supervisor_id"],
        sibling_supervisor_id=canonical["sibling_supervisor_id"],
        capability=canonical["capability"],
        epoch=canonical["epoch"],
        record_digest=_sha256_hex(_canonical_json(canonical).encode("utf-8")),
        effect=canonical["effect"],
    )
    _ = bool(record.get("worker_assertion"))
    return admission


def build_sibling_supervisor_capability_registry(
    records: Sequence[Mapping[str, Any] | SiblingSupervisorCapabilityAdmission] = (),
    *,
    local_supervisor_id: str,
    epoch: int = 1,
    known_sibling_ids: Sequence[str] = (),
) -> SiblingSupervisorCapabilityRegistry:
    """Build an admitted in-process registry without writing state."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise SiblingSupervisorCapabilityRegistryError(
            "registry records must be a sequence",
            code="record_invalid",
        )
    admitted: list[SiblingSupervisorCapabilityAdmission] = []
    for item in records:
        if isinstance(item, SiblingSupervisorCapabilityAdmission):
            admitted.append(item)
            continue
        payload: dict[str, Any] = dict(item)
        payload.setdefault("local_supervisor_id", local_supervisor_id)
        payload.setdefault("epoch", epoch)
        payload.setdefault("current_epoch", epoch)
        if known_sibling_ids and "known_sibling_ids" not in payload:
            payload["known_sibling_ids"] = known_sibling_ids
        admitted.append(register_sibling_supervisor_capability(payload))
    return SiblingSupervisorCapabilityRegistry(
        local_supervisor_id=local_supervisor_id,
        epoch=epoch,
        records=tuple(admitted),
    )


def lookup_sibling_supervisor_capability(
    registry: SiblingSupervisorCapabilityRegistry,
    sibling_supervisor_id: str,
    capability: str,
) -> SiblingSupervisorCapabilityAdmission:
    """Look up one admitted sibling capability. Missing is fail-closed."""

    if not isinstance(registry, SiblingSupervisorCapabilityRegistry):
        raise SiblingSupervisorCapabilityRegistryError(
            "lookup requires SiblingSupervisorCapabilityRegistry@1",
            code="record_invalid",
        )
    return registry.lookup(sibling_supervisor_id, capability)


def _receipt_text(value: Any, field_name: str, *, required: bool = True) -> str:
    return _text(
        value,
        field_name,
        required=required,
        error_cls=CrossSupervisorReceiptError,
        code="receipt_invalid",
    )


def _receipt_digest_text(value: Any, field_name: str) -> str:
    digest = _receipt_text(value, field_name)
    prefix = "sha256:"
    if not digest.startswith(prefix):
        raise CrossSupervisorReceiptError(
            f"{field_name} must be a sha256 digest",
            code="receipt_invalid",
        )
    hex_part = digest[len(prefix) :]
    if len(hex_part) != 64 or any(
        character not in "0123456789abcdef" for character in hex_part
    ):
        raise CrossSupervisorReceiptError(
            f"{field_name} must be a lowercase sha256 hex digest",
            code="receipt_invalid",
        )
    return digest


def _cross_supervisor_receipt_wire(receipt: Any) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise CrossSupervisorReceiptError(
            "cross-supervisor receipt must be an object",
            code="receipt_invalid",
        )
    fields = set(receipt)
    forbidden = fields & CROSS_SUPERVISOR_RECEIPT_FORBIDDEN_FIELDS
    if forbidden:
        raise CrossSupervisorReceiptError(
            "cross-supervisor receipt contains operational authority field(s): "
            + ", ".join(sorted(forbidden)),
            code="operational_authority_field",
        )
    missing = CROSS_SUPERVISOR_RECEIPT_REQUIRED_FIELDS - fields
    extra = fields - CROSS_SUPERVISOR_RECEIPT_REQUIRED_FIELDS
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unknown " + ", ".join(sorted(extra)))
        raise CrossSupervisorReceiptError(
            "cross-supervisor receipt fields: " + "; ".join(details),
            code="receipt_invalid",
        )
    schema = receipt["schema"]
    if schema != CROSS_SUPERVISOR_RECEIPT_SCHEMA:
        raise CrossSupervisorReceiptError(
            f"unsupported cross-supervisor receipt schema {schema!r}",
            code="receipt_schema",
        )
    outcome = _receipt_text(receipt["outcome"], "outcome")
    if outcome not in CROSS_SUPERVISOR_RECEIPT_OUTCOMES:
        raise CrossSupervisorReceiptError(
            f"unsupported cross-supervisor receipt outcome {outcome!r}",
            code="unknown_outcome",
        )
    payload = receipt["payload"]
    if not isinstance(payload, Mapping):
        raise CrossSupervisorReceiptError(
            "payload must be an object",
            code="receipt_invalid",
        )
    try:
        detached_payload = json.loads(_canonical_json(dict(payload)))
    except (TypeError, ValueError) as error:
        raise CrossSupervisorReceiptError(
            f"invalid receipt payload: {error}",
            code="receipt_invalid",
        ) from error
    if not isinstance(detached_payload, dict):
        raise CrossSupervisorReceiptError(
            "payload must be an object",
            code="receipt_invalid",
        )
    encoded_payload = _canonical_json(detached_payload).encode("utf-8")
    if len(encoded_payload) > _MAX_SIBLING_RECEIPT_PAYLOAD_BYTES:
        raise CrossSupervisorReceiptError(
            "receipt payload exceeds the admitted bound",
            code="receipt_invalid",
        )
    return {
        "carrier_event_id": _receipt_text(
            receipt["carrier_event_id"], "carrier_event_id"
        ),
        "evidence_digest": _receipt_digest_text(
            receipt["evidence_digest"], "evidence_digest"
        ),
        "outcome": outcome,
        "payload": detached_payload,
        "receipt_id": _receipt_text(receipt["receipt_id"], "receipt_id"),
        "request_id": _receipt_text(
            receipt["request_id"], "request_id", required=False
        ),
        "schema": CROSS_SUPERVISOR_RECEIPT_SCHEMA,
        "task_id": _receipt_text(receipt["task_id"], "task_id"),
    }


@dataclass(frozen=True)
class CrossSupervisorReceiptAdmission:
    """Non-authoritative admission of one cross-supervisor receipt envelope.

    This binding is not a second receipt log, bus, registry, or state owner.
    Sibling supervisors exchange receipts as events; admission never writes
    DuckDB or DuckLake, never consumes ``DatabaseEventLog@1``, and never
    terminalizes a task.
    """

    local_supervisor_id: str
    sibling_supervisor_id: str
    receipt_id: str
    task_id: str
    carrier_event_id: str
    outcome: str
    evidence_digest: str
    receipt_digest: str
    epoch: int
    request_id: str = ""
    capability: str = CROSS_SUPERVISOR_RECEIPT_CAPABILITY
    effect: str = "event_exchange"
    schema: str = CROSS_SUPERVISOR_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CROSS_SUPERVISOR_RECEIPT_SCHEMA:
            raise CrossSupervisorReceiptError(
                f"unsupported cross-supervisor receipt schema {self.schema!r}",
                code="receipt_schema",
            )
        object.__setattr__(
            self,
            "local_supervisor_id",
            _receipt_text(self.local_supervisor_id, "local_supervisor_id"),
        )
        object.__setattr__(
            self,
            "sibling_supervisor_id",
            _receipt_text(self.sibling_supervisor_id, "sibling_supervisor_id"),
        )
        if self.local_supervisor_id == self.sibling_supervisor_id:
            raise CrossSupervisorReceiptError(
                "a supervisor is not its own sibling",
                code="not_a_sibling",
            )
        object.__setattr__(self, "receipt_id", _receipt_text(self.receipt_id, "receipt_id"))
        object.__setattr__(self, "task_id", _receipt_text(self.task_id, "task_id"))
        object.__setattr__(
            self,
            "request_id",
            _receipt_text(self.request_id, "request_id", required=False),
        )
        object.__setattr__(
            self,
            "carrier_event_id",
            _receipt_text(self.carrier_event_id, "carrier_event_id"),
        )
        outcome = _receipt_text(self.outcome, "outcome")
        if outcome not in CROSS_SUPERVISOR_RECEIPT_OUTCOMES:
            raise CrossSupervisorReceiptError(
                f"unsupported cross-supervisor receipt outcome {outcome!r}",
                code="unknown_outcome",
            )
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(
            self,
            "evidence_digest",
            _receipt_digest_text(self.evidence_digest, "evidence_digest"),
        )
        object.__setattr__(
            self,
            "receipt_digest",
            _receipt_digest_text(self.receipt_digest, "receipt_digest"),
        )
        capability = _receipt_text(self.capability, "capability")
        if capability in FORBIDDEN_SIBLING_CAPABILITIES:
            raise CrossSupervisorReceiptError(
                f"sibling capability {capability!r} is forbidden",
                code="forbidden_capability",
            )
        if capability != CROSS_SUPERVISOR_RECEIPT_CAPABILITY:
            raise CrossSupervisorReceiptError(
                "cross-supervisor receipts require "
                f"{CROSS_SUPERVISOR_RECEIPT_CAPABILITY!r}",
                code="unknown_capability",
            )
        object.__setattr__(self, "capability", capability)
        epoch = int(self.epoch)
        if epoch < 1:
            raise CrossSupervisorReceiptError(
                "epoch must be >= 1",
                code="stale_fence_epoch",
            )
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(
            self,
            "effect",
            _admit_sibling_effect(
                self.effect,
                error_cls=CrossSupervisorReceiptError,
            ),
        )

    @property
    def logical_once_key(self) -> str:
        return f"{self.sibling_supervisor_id}:{self.receipt_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "binding": CROSS_SUPERVISOR_RECEIPT_BINDING,
            "interface": CROSS_SUPERVISOR_RECEIPT_INTERFACE,
            "carrier": SUPERVISOR_FABRIC_INTERFACE,
            "consumes": {
                "supervisor_fabric": SUPERVISOR_FABRIC_INTERFACE,
                "sibling_event_validation": SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING,
                "sibling_capability_registry": (
                    SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING
                ),
            },
            "local_supervisor_id": self.local_supervisor_id,
            "sibling_supervisor_id": self.sibling_supervisor_id,
            "receipt_id": self.receipt_id,
            "task_id": self.task_id,
            "request_id": self.request_id,
            "carrier_event_id": self.carrier_event_id,
            "outcome": self.outcome,
            "evidence_digest": self.evidence_digest,
            "receipt_digest": self.receipt_digest,
            "epoch": self.epoch,
            "capability": self.capability,
            "effect": self.effect,
            "logical_once_key": self.logical_once_key,
            "fenced": True,
            "admitted": True,
            "database_write": False,
            "direct_state_write": False,
            "terminalize_task": False,
            "completion_authoritative": False,
            "worker_assertion_is_authority": False,
            "worker_completion_insufficient": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CrossSupervisorReceiptAdmission:
        if not isinstance(payload, Mapping):
            raise CrossSupervisorReceiptError(
                "cross-supervisor receipt admission must be an object",
                code="record_invalid",
            )
        return cls(
            local_supervisor_id=str(payload.get("local_supervisor_id") or ""),
            sibling_supervisor_id=str(payload.get("sibling_supervisor_id") or ""),
            receipt_id=str(payload.get("receipt_id") or ""),
            task_id=str(payload.get("task_id") or ""),
            request_id=str(payload.get("request_id") or ""),
            carrier_event_id=str(payload.get("carrier_event_id") or ""),
            outcome=str(payload.get("outcome") or ""),
            evidence_digest=str(payload.get("evidence_digest") or ""),
            receipt_digest=str(payload.get("receipt_digest") or ""),
            epoch=int(payload.get("epoch") or 0),
            capability=str(
                payload.get("capability") or CROSS_SUPERVISOR_RECEIPT_CAPABILITY
            ),
            effect=str(payload.get("effect") or "event_exchange"),
            schema=str(payload.get("schema") or CROSS_SUPERVISOR_RECEIPT_SCHEMA),
        )


def admit_cross_supervisor_receipt(
    record: Mapping[str, Any],
) -> CrossSupervisorReceiptAdmission:
    """Admit one sibling receipt envelope without writing or consuming state."""

    if not isinstance(record, Mapping):
        raise CrossSupervisorReceiptError(
            "cross-supervisor receipt record must be an object",
            code="record_invalid",
        )
    _reject_direct_state_writes(
        record,
        error_cls=CrossSupervisorReceiptError,
        subject="sibling receipts",
    )
    if record.get("completion_authoritative"):
        raise CrossSupervisorReceiptError(
            "cross-supervisor receipt admission is not completion authority",
            code="completion_not_authoritative",
        )
    local_supervisor_id = _receipt_text(
        record.get("local_supervisor_id"), "local_supervisor_id"
    )
    sibling_supervisor_id = _receipt_text(
        record.get("sibling_supervisor_id") or record.get("supervisor_id"),
        "sibling_supervisor_id",
    )
    if sibling_supervisor_id == local_supervisor_id:
        raise CrossSupervisorReceiptError(
            "a supervisor is not its own sibling",
            code="not_a_sibling",
        )
    known = record.get("known_sibling_ids")
    if known is not None:
        if not isinstance(known, Sequence) or isinstance(known, (str, bytes)):
            raise CrossSupervisorReceiptError(
                "known_sibling_ids must be a sequence of supervisor ids",
                code="unknown_sibling",
            )
        if sibling_supervisor_id not in tuple(known):
            raise CrossSupervisorReceiptError(
                f"unknown sibling supervisor {sibling_supervisor_id!r}",
                code="unknown_sibling",
            )
    if "capability" in record:
        capability = record.get("capability")
    else:
        capability = CROSS_SUPERVISOR_RECEIPT_CAPABILITY
    if isinstance(capability, str) and capability in FORBIDDEN_SIBLING_CAPABILITIES:
        raise CrossSupervisorReceiptError(
            f"sibling capability {capability!r} is forbidden",
            code="forbidden_capability",
        )
    if isinstance(capability, str) and capability and capability != CROSS_SUPERVISOR_RECEIPT_CAPABILITY:
        raise CrossSupervisorReceiptError(
            "cross-supervisor receipts require "
            f"{CROSS_SUPERVISOR_RECEIPT_CAPABILITY!r}",
            code="unknown_capability",
        )
    current_epoch = record.get("current_epoch")
    if current_epoch is not None:
        try:
            current = int(current_epoch)
        except (TypeError, ValueError) as error:
            raise CrossSupervisorReceiptError(
                "current_epoch must be an integer >= 1",
                code="stale_fence_epoch",
            ) from error
        if current < 1:
            raise CrossSupervisorReceiptError(
                "current_epoch must be >= 1",
                code="stale_fence_epoch",
            )
        try:
            offered = int(record.get("epoch") or 0)
        except (TypeError, ValueError) as error:
            raise CrossSupervisorReceiptError(
                "epoch must be an integer >= 1",
                code="stale_fence_epoch",
            ) from error
        if offered < current:
            raise CrossSupervisorReceiptError(
                "stale fence epoch",
                code="stale_fence_epoch",
            )
    fence = issue_fence(
        {
            "supervisor_id": sibling_supervisor_id,
            "capability": capability,
            "epoch": record.get("epoch"),
            "stale_epoch": record.get("stale_epoch"),
        }
    )
    admitted_capability = str(fence["capability"])
    if admitted_capability in FORBIDDEN_SIBLING_CAPABILITIES:
        raise CrossSupervisorReceiptError(
            f"sibling capability {admitted_capability!r} is forbidden",
            code="forbidden_capability",
        )
    if admitted_capability != CROSS_SUPERVISOR_RECEIPT_CAPABILITY:
        raise CrossSupervisorReceiptError(
            "cross-supervisor receipts require "
            f"{CROSS_SUPERVISOR_RECEIPT_CAPABILITY!r}",
            code="unknown_capability",
        )
    envelope = _cross_supervisor_receipt_wire(record.get("receipt"))
    admission = CrossSupervisorReceiptAdmission(
        local_supervisor_id=local_supervisor_id,
        sibling_supervisor_id=str(fence["supervisor_id"]),
        receipt_id=str(envelope["receipt_id"]),
        task_id=str(envelope["task_id"]),
        request_id=str(envelope["request_id"]),
        carrier_event_id=str(envelope["carrier_event_id"]),
        outcome=str(envelope["outcome"]),
        evidence_digest=str(envelope["evidence_digest"]),
        receipt_digest=_sha256_hex(_canonical_json(envelope).encode("utf-8")),
        epoch=int(fence["epoch"]),
        capability=admitted_capability,
        effect=_admit_sibling_effect(
            record.get("effect") or "event_exchange",
            error_cls=CrossSupervisorReceiptError,
        ),
    )
    _ = bool(record.get("worker_assertion"))
    return admission


class SupervisorFabric:
    """Fenced coordination carrier for sibling-supervisor event and receipt admission."""

    INTERFACE: Final[str] = SUPERVISOR_FABRIC_INTERFACE
    SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING: Final[str] = (
        SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING
    )
    SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING: Final[str] = (
        SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING
    )
    CROSS_SUPERVISOR_RECEIPT_BINDING: Final[str] = CROSS_SUPERVISOR_RECEIPT_BINDING

    def __init__(
        self,
        *,
        supervisor_id: str,
        epoch: int = 1,
        capability: str = "event-exchange",
        known_sibling_ids: Sequence[str] = (),
        sibling_capabilities: Sequence[
            Mapping[str, Any] | SiblingSupervisorCapabilityAdmission
        ] = (),
    ) -> None:
        self._supervisor_id = _text(supervisor_id, "supervisor_id")
        self._epoch = int(epoch or 1)
        if self._epoch < 1:
            raise SupervisorFabricError("stale fence epoch")
        self._capability = _text(capability, "capability")
        self._known_sibling_ids = tuple(
            _text(item, "known_sibling_id") for item in known_sibling_ids
        )
        self._capability_registry: dict[str, SiblingSupervisorCapabilityAdmission] = {}
        for item in sibling_capabilities:
            self.register_sibling_capability(item)

    @property
    def supervisor_id(self) -> str:
        return self._supervisor_id

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def capability(self) -> str:
        return self._capability

    @property
    def known_sibling_ids(self) -> tuple[str, ...]:
        return self._known_sibling_ids

    @property
    def sibling_capability_registry(self) -> SiblingSupervisorCapabilityRegistry:
        return SiblingSupervisorCapabilityRegistry(
            local_supervisor_id=self._supervisor_id,
            epoch=self._epoch,
            records=tuple(self._capability_registry.values()),
        )

    def issue_fence(self, record: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        payload: dict[str, Any] = dict(record or {})
        payload.setdefault("supervisor_id", self._supervisor_id)
        payload.setdefault("capability", self._capability)
        payload.setdefault("epoch", self._epoch)
        return issue_fence(payload)

    def register_sibling_capability(
        self, record: Mapping[str, Any] | SiblingSupervisorCapabilityAdmission
    ) -> SiblingSupervisorCapabilityAdmission:
        if isinstance(record, SiblingSupervisorCapabilityAdmission):
            admission = record
            if admission.local_supervisor_id != self._supervisor_id:
                raise SiblingSupervisorCapabilityRegistryError(
                    "capability record local supervisor does not match the fabric",
                    code="sibling_identity_mismatch",
                )
        else:
            payload: dict[str, Any] = dict(record)
            payload.setdefault("local_supervisor_id", self._supervisor_id)
            payload.setdefault("epoch", record.get("epoch", self._epoch))
            payload.setdefault("current_epoch", self._epoch)
            if self._known_sibling_ids and "known_sibling_ids" not in payload:
                payload["known_sibling_ids"] = self._known_sibling_ids
            admission = register_sibling_supervisor_capability(payload)
        if admission.epoch < self._epoch:
            raise SiblingSupervisorCapabilityRegistryError(
                "stale fence epoch",
                code="stale_fence_epoch",
            )
        existing = self._capability_registry.get(admission.registry_key)
        if existing is not None:
            if existing.record_digest != admission.record_digest:
                raise SiblingSupervisorCapabilityRegistryError(
                    f"conflicting sibling capability {admission.registry_key!r}",
                    code="capability_conflict",
                )
            return existing
        if len(self._capability_registry) >= _MAX_SIBLING_CAPABILITY_RECORDS:
            raise SiblingSupervisorCapabilityRegistryError(
                "sibling capability registry exceeds the admitted bound",
                code="registry_bound",
            )
        self._capability_registry[admission.registry_key] = admission
        return admission

    def lookup_sibling_capability(
        self, sibling_supervisor_id: str, capability: str
    ) -> SiblingSupervisorCapabilityAdmission:
        return self.sibling_capability_registry.lookup(
            sibling_supervisor_id, capability
        )

    def list_sibling_capabilities(
        self, sibling_supervisor_id: str | None = None
    ) -> tuple[SiblingSupervisorCapabilityAdmission, ...]:
        records = self.sibling_capability_registry.records
        if sibling_supervisor_id is None:
            return records
        sibling_supervisor_id = _capability_text(
            sibling_supervisor_id, "sibling_supervisor_id"
        )
        return tuple(
            item
            for item in records
            if item.sibling_supervisor_id == sibling_supervisor_id
        )

    def validate_sibling_event(
        self, record: Mapping[str, Any]
    ) -> SiblingSupervisorEventAdmission:
        payload: dict[str, Any] = dict(record)
        payload.setdefault("local_supervisor_id", self._supervisor_id)
        payload.setdefault("epoch", record.get("epoch", self._epoch))
        if self._known_sibling_ids and "known_sibling_ids" not in payload:
            payload["known_sibling_ids"] = self._known_sibling_ids
        if self._capability_registry:
            sibling_supervisor_id = _capability_text(
                payload.get("sibling_supervisor_id") or payload.get("supervisor_id"),
                "sibling_supervisor_id",
            )
            capability = _capability_text(payload.get("capability"), "capability")
            self.lookup_sibling_capability(sibling_supervisor_id, capability)
        return validate_sibling_supervisor_event(payload)

    def admit_cross_supervisor_receipt(
        self, record: Mapping[str, Any]
    ) -> CrossSupervisorReceiptAdmission:
        payload: dict[str, Any] = dict(record)
        payload.setdefault("local_supervisor_id", self._supervisor_id)
        payload.setdefault("epoch", record.get("epoch", self._epoch))
        payload.setdefault("capability", CROSS_SUPERVISOR_RECEIPT_CAPABILITY)
        payload.setdefault("current_epoch", self._epoch)
        if self._known_sibling_ids and "known_sibling_ids" not in payload:
            payload["known_sibling_ids"] = self._known_sibling_ids
        if self._capability_registry:
            sibling_supervisor_id = _receipt_text(
                payload.get("sibling_supervisor_id") or payload.get("supervisor_id"),
                "sibling_supervisor_id",
            )
            capability = _receipt_text(
                payload.get("capability") or CROSS_SUPERVISOR_RECEIPT_CAPABILITY,
                "capability",
            )
            self.lookup_sibling_capability(sibling_supervisor_id, capability)
        return admit_cross_supervisor_receipt(payload)


__all__ = [
    "ALLOWED_SIBLING_CAPABILITIES",
    "ALLOWED_SIBLING_EFFECTS",
    "CANONICAL_EVENT_FORBIDDEN_FIELDS",
    "CANONICAL_EVENT_INTERFACE",
    "CANONICAL_EVENT_REQUIRED_FIELDS",
    "CANONICAL_EVENT_SCHEMA_ID",
    "CROSS_SUPERVISOR_RECEIPT_BINDING",
    "CROSS_SUPERVISOR_RECEIPT_CAPABILITY",
    "CROSS_SUPERVISOR_RECEIPT_CARRIER",
    "CROSS_SUPERVISOR_RECEIPT_CONSUMES",
    "CROSS_SUPERVISOR_RECEIPT_FORBIDDEN_FIELDS",
    "CROSS_SUPERVISOR_RECEIPT_INTERFACE",
    "CROSS_SUPERVISOR_RECEIPT_OUTCOMES",
    "CROSS_SUPERVISOR_RECEIPT_REQUIRED_FIELDS",
    "CROSS_SUPERVISOR_RECEIPT_SCHEMA",
    "DATABASE_EVENT_LOG_INTERFACE",
    "EVENT_CURSOR_INTERFACE",
    "FORBIDDEN_SIBLING_CAPABILITIES",
    "SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING",
    "SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_CONSUMES",
    "SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_INTERFACE",
    "SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA",
    "SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING",
    "SIBLING_SUPERVISOR_EVENT_VALIDATION_CONSUMES",
    "SIBLING_SUPERVISOR_EVENT_VALIDATION_INTERFACE",
    "SIBLING_SUPERVISOR_EVENT_VALIDATION_SCHEMA",
    "SUPERVISOR_FABRIC_INTERFACE",
    "CrossSupervisorReceiptAdmission",
    "CrossSupervisorReceiptError",
    "SiblingSupervisorCapabilityAdmission",
    "SiblingSupervisorCapabilityRegistry",
    "SiblingSupervisorCapabilityRegistryError",
    "SiblingSupervisorEventAdmission",
    "SiblingSupervisorEventValidationError",
    "SupervisorFabric",
    "SupervisorFabricError",
    "admit_cross_supervisor_receipt",
    "build_sibling_supervisor_capability_registry",
    "issue_fence",
    "lookup_sibling_supervisor_capability",
    "register_sibling_supervisor_capability",
    "validate_sibling_supervisor_event",
]
