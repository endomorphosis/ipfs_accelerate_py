"""Supervisor capability and fenced coordination contracts.

Sibling-supervisor event validation and the sibling-supervisor capability
registry are bindings of this fabric, not a second event log, bus, coordinator,
or state owner. Sibling supervisors exchange canonical event envelopes and
receipts. They never write DuckDB or DuckLake, never consume
``DatabaseEventLog@1``, and never terminalize tasks. A worker or model
assertion cannot admit a sibling event or grant a sibling capability.

The capability registry is a closed catalog plus fenced advertisements. It is
not a competing subsystem.

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
    """A sibling-supervisor capability advertisement failed admission."""

    def __init__(self, message: str, *, code: str = "sibling_capability_invalid") -> None:
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
SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING: Final[str] = (
    "SiblingSupervisorCapabilityRegistry@1"
)
SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_INTERFACE: Final[str] = (
    SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING
)
SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/sibling-supervisor-capability-registry@1"
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
SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_CONSUMES: Final[tuple[str, ...]] = (
    SUPERVISOR_FABRIC_INTERFACE,
    SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING,
    CANONICAL_EVENT_INTERFACE,
    DATABASE_EVENT_LOG_INTERFACE,
)

ADMITTED_SIBLING_CAPABILITIES: Final[tuple[str, ...]] = (
    "event-exchange",
    "incremental-reassessment",
    "receipt-exchange",
    "task-request",
)
SIBLING_CAPABILITY_DEFAULT_EFFECT: Final[str] = "event_exchange"
FORBIDDEN_SIBLING_CAPABILITIES: Final[frozenset[str]] = frozenset(
    {
        "database-write",
        "direct-state-write",
        "duckdb-write",
        "ducklake-write",
        "owner-mutation",
        "policy-pointer-mutation",
        "terminalize-task",
        "write-database",
    }
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
_MAX_IDENTIFIER_CHARS: Final[int] = 512


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


def _text(value: Any, field_name: str, *, required: bool = True) -> str:
    if value is None:
        if required:
            raise SiblingSupervisorEventValidationError(
                f"{field_name} is required",
                code="sibling_identity_mismatch",
            )
        return ""
    if not isinstance(value, str):
        raise SiblingSupervisorEventValidationError(
            f"{field_name} must be a string",
            code="sibling_identity_mismatch",
        )
    if not value:
        if required:
            raise SiblingSupervisorEventValidationError(
                f"{field_name} must not be empty",
                code="sibling_identity_mismatch",
            )
        return ""
    if value != value.strip():
        raise SiblingSupervisorEventValidationError(
            f"{field_name} must be exact and contain no surrounding whitespace",
            code="sibling_identity_mismatch",
        )
    if len(value) > _MAX_IDENTIFIER_CHARS:
        raise SiblingSupervisorEventValidationError(
            f"{field_name} must not exceed {_MAX_IDENTIFIER_CHARS} characters",
            code="sibling_identity_mismatch",
        )
    if any(character.isspace() or not character.isprintable() for character in value):
        raise SiblingSupervisorEventValidationError(
            f"{field_name} must contain only printable, non-whitespace characters",
            code="sibling_identity_mismatch",
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


def _reject_direct_state_writes(record: Mapping[str, Any]) -> None:
    present = sorted(key for key in FORBIDDEN_SIBLING_WRITE_KEYS if record.get(key))
    if present:
        raise SiblingSupervisorEventValidationError(
            "sibling supervisors exchange events, never database writes: "
            + ", ".join(present),
            code="direct_state_write",
        )
    mutations = sorted(
        key for key in FORBIDDEN_SIBLING_LOG_MUTATIONS if record.get(key)
    )
    if mutations:
        raise SiblingSupervisorEventValidationError(
            "sibling events cannot mutate DatabaseEventLog@1: " + ", ".join(mutations),
            code="direct_state_write",
        )
    effect = record.get("effect")
    if effect is None:
        return
    if not isinstance(effect, str) or effect not in ALLOWED_SIBLING_EFFECTS:
        raise SiblingSupervisorEventValidationError(
            f"sibling event effect {effect!r} is not admitted",
            code="forbidden_effect",
        )


def _capability_text(value: Any, field_name: str, *, required: bool = True) -> str:
    try:
        return _text(value, field_name, required=required)
    except SiblingSupervisorEventValidationError as error:
        raise SiblingSupervisorCapabilityRegistryError(
            str(error),
            code="capability_identity_invalid",
        ) from error


def _reject_capability_state_writes(record: Mapping[str, Any]) -> None:
    try:
        _reject_direct_state_writes(record)
    except SiblingSupervisorEventValidationError as error:
        raise SiblingSupervisorCapabilityRegistryError(
            str(error),
            code=error.code,
        ) from error


def sibling_supervisor_capability_catalog() -> tuple[Mapping[str, Any], ...]:
    """Return the closed sibling-capability catalog. It is not runtime-extensible."""

    return tuple(
        MappingProxyType(
            {
                "capability": name,
                "effect": SIBLING_CAPABILITY_DEFAULT_EFFECT,
                "database_write": False,
                "completion_authoritative": False,
                "worker_assertion_is_authority": False,
                "worker_completion_insufficient": True,
            }
        )
        for name in ADMITTED_SIBLING_CAPABILITIES
    )


def _admitted_sibling_capability(value: Any) -> str:
    capability = _capability_text(value, "capability")
    if capability in FORBIDDEN_SIBLING_CAPABILITIES:
        raise SiblingSupervisorCapabilityRegistryError(
            f"sibling capability {capability!r} is forbidden",
            code="forbidden_capability",
        )
    if capability not in ADMITTED_SIBLING_CAPABILITIES:
        raise SiblingSupervisorCapabilityRegistryError(
            f"unknown sibling capability {capability!r}",
            code="unknown_capability",
        )
    return capability


def _admitted_sibling_effect(value: Any) -> str:
    if value is None or value == "":
        return SIBLING_CAPABILITY_DEFAULT_EFFECT
    if not isinstance(value, str) or value not in ALLOWED_SIBLING_EFFECTS:
        raise SiblingSupervisorCapabilityRegistryError(
            f"sibling capability effect {value!r} is not admitted",
            code="forbidden_effect",
        )
    if value in {"", "none"}:
        return SIBLING_CAPABILITY_DEFAULT_EFFECT
    return value


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
    registry = record.get("capability_registry")
    if registry is not None:
        lookup_sibling_supervisor_capability(
            registry,
            sibling_supervisor_id,
            str(fence["capability"]),
            local_supervisor_id=local_supervisor_id,
            epoch=int(fence["epoch"]),
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


@dataclass(frozen=True)
class SiblingSupervisorCapabilityRecord:
    """Non-authoritative advertisement of one sibling-supervisor capability."""

    local_supervisor_id: str
    sibling_supervisor_id: str
    capability: str
    epoch: int
    effect: str = SIBLING_CAPABILITY_DEFAULT_EFFECT
    advertised: bool = True
    schema: str = SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA:
            raise SiblingSupervisorCapabilityRegistryError(
                f"unsupported sibling capability schema {self.schema!r}",
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
        object.__setattr__(
            self, "capability", _admitted_sibling_capability(self.capability)
        )
        object.__setattr__(self, "effect", _admitted_sibling_effect(self.effect))
        epoch = int(self.epoch)
        if epoch < 1:
            raise SiblingSupervisorCapabilityRegistryError(
                "epoch must be >= 1",
                code="stale_fence_epoch",
            )
        object.__setattr__(self, "epoch", epoch)
        if not isinstance(self.advertised, bool) or self.advertised is not True:
            raise SiblingSupervisorCapabilityRegistryError(
                "sibling capabilities must be explicitly advertised",
                code="capability_not_advertised",
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
                "canonical_event": CANONICAL_EVENT_INTERFACE,
                "database_event_log": DATABASE_EVENT_LOG_INTERFACE,
            },
            "local_supervisor_id": self.local_supervisor_id,
            "sibling_supervisor_id": self.sibling_supervisor_id,
            "capability": self.capability,
            "epoch": self.epoch,
            "effect": self.effect,
            "advertised": self.advertised,
            "registry_key": self.registry_key,
            "fenced": True,
            "admitted": True,
            "database_write": False,
            "completion_authoritative": False,
            "worker_assertion_is_authority": False,
            "worker_completion_insufficient": True,
        }


@dataclass(frozen=True)
class SiblingSupervisorCapabilityRegistrySnapshot:
    """Closed, fenced snapshot of sibling-supervisor capability advertisements."""

    local_supervisor_id: str
    epoch: int
    records: tuple[SiblingSupervisorCapabilityRecord, ...]
    registry_digest: str
    schema: str = SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA:
            raise SiblingSupervisorCapabilityRegistryError(
                f"unsupported sibling capability schema {self.schema!r}",
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
        ordered = tuple(
            sorted(
                self.records,
                key=lambda item: (item.sibling_supervisor_id, item.capability),
            )
        )
        object.__setattr__(self, "records", ordered)
        seen: dict[str, SiblingSupervisorCapabilityRecord] = {}
        for record in ordered:
            if record.local_supervisor_id != self.local_supervisor_id:
                raise SiblingSupervisorCapabilityRegistryError(
                    "capability record local supervisor does not match registry",
                    code="sibling_identity_mismatch",
                )
            if record.epoch > self.epoch:
                raise SiblingSupervisorCapabilityRegistryError(
                    "capability record epoch is ahead of registry epoch",
                    code="stale_fence_epoch",
                )
            existing = seen.get(record.registry_key)
            if existing is not None and existing.to_dict() != record.to_dict():
                raise SiblingSupervisorCapabilityRegistryError(
                    f"conflicting sibling capability {record.registry_key!r}",
                    code="capability_conflict",
                )
            seen[record.registry_key] = record
        digest = _sha256_hex(
            _canonical_json(
                {
                    "epoch": self.epoch,
                    "local_supervisor_id": self.local_supervisor_id,
                    "records": [
                        {
                            "capability": item.capability,
                            "effect": item.effect,
                            "epoch": item.epoch,
                            "sibling_supervisor_id": item.sibling_supervisor_id,
                        }
                        for item in ordered
                    ],
                    "schema": self.schema,
                }
            ).encode("utf-8")
        )
        supplied = _capability_text(self.registry_digest, "registry_digest")
        if supplied != digest:
            raise SiblingSupervisorCapabilityRegistryError(
                "sibling capability registry digest does not match contents",
                code="registry_digest_mismatch",
            )

    def lookup(
        self, sibling_supervisor_id: str, capability: str
    ) -> SiblingSupervisorCapabilityRecord:
        key = (
            f"{_capability_text(sibling_supervisor_id, 'sibling_supervisor_id')}:"
            f"{_admitted_sibling_capability(capability)}"
        )
        for record in self.records:
            if record.registry_key == key:
                return record
        raise SiblingSupervisorCapabilityRegistryError(
            f"unregistered sibling capability {key!r}",
            code="unregistered_capability",
        )

    def advertised_for(self, sibling_supervisor_id: str) -> tuple[str, ...]:
        sibling_id = _capability_text(sibling_supervisor_id, "sibling_supervisor_id")
        return tuple(
            record.capability
            for record in self.records
            if record.sibling_supervisor_id == sibling_id
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
                "canonical_event": CANONICAL_EVENT_INTERFACE,
                "database_event_log": DATABASE_EVENT_LOG_INTERFACE,
            },
            "local_supervisor_id": self.local_supervisor_id,
            "epoch": self.epoch,
            "records": [record.to_dict() for record in self.records],
            "registry_digest": self.registry_digest,
            "catalog": [item["capability"] for item in sibling_supervisor_capability_catalog()],
            "fenced": True,
            "admitted": True,
            "database_write": False,
            "completion_authoritative": False,
            "worker_assertion_is_authority": False,
            "worker_completion_insufficient": True,
        }


def _registry_digest_for(
    *,
    local_supervisor_id: str,
    epoch: int,
    records: Sequence[SiblingSupervisorCapabilityRecord],
) -> str:
    ordered = sorted(
        records, key=lambda item: (item.sibling_supervisor_id, item.capability)
    )
    return _sha256_hex(
        _canonical_json(
            {
                "epoch": epoch,
                "local_supervisor_id": local_supervisor_id,
                "records": [
                    {
                        "capability": item.capability,
                        "effect": item.effect,
                        "epoch": item.epoch,
                        "sibling_supervisor_id": item.sibling_supervisor_id,
                    }
                    for item in ordered
                ],
                "schema": SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA,
            }
        ).encode("utf-8")
    )


def register_sibling_supervisor_capability(
    record: Mapping[str, Any],
) -> SiblingSupervisorCapabilityRecord:
    """Admit one sibling capability advertisement without writing or consuming state."""

    if not isinstance(record, Mapping):
        raise SiblingSupervisorCapabilityRegistryError(
            "sibling capability record must be an object",
            code="record_invalid",
        )
    _reject_capability_state_writes(record)
    if record.get("completion_authoritative"):
        raise SiblingSupervisorCapabilityRegistryError(
            "sibling capability admission is not completion authority",
            code="completion_not_authoritative",
        )
    if record.get("catalog_override") or record.get("self_granted"):
        raise SiblingSupervisorCapabilityRegistryError(
            "sibling capability catalog is closed and not self-granted",
            code="catalog_closed",
        )
    if record.get("advertised") is False:
        raise SiblingSupervisorCapabilityRegistryError(
            "sibling capabilities must be explicitly advertised",
            code="capability_not_advertised",
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
    capability = _admitted_sibling_capability(record.get("capability"))
    fence = issue_fence(
        {
            "supervisor_id": sibling_supervisor_id,
            "capability": capability,
            "epoch": record.get("epoch"),
            "stale_epoch": record.get("stale_epoch"),
        }
    )
    admitted = SiblingSupervisorCapabilityRecord(
        local_supervisor_id=local_supervisor_id,
        sibling_supervisor_id=str(fence["supervisor_id"]),
        capability=str(fence["capability"]),
        epoch=int(fence["epoch"]),
        effect=_admitted_sibling_effect(record.get("effect")),
        advertised=True,
    )
    # Worker assertions are recorded as non-authority; they never grant a capability.
    _ = bool(record.get("worker_assertion"))
    return admitted


def _iter_capability_records(
    registry: Any,
    *,
    local_supervisor_id: str | None = None,
    epoch: int | None = None,
) -> tuple[SiblingSupervisorCapabilityRecord, ...]:
    if isinstance(registry, SiblingSupervisorCapabilityRegistrySnapshot):
        return registry.records
    if isinstance(registry, SiblingSupervisorCapabilityRecord):
        return (registry,)
    if isinstance(registry, Mapping):
        if "capability" in registry and (
            "sibling_supervisor_id" in registry or "supervisor_id" in registry
        ):
            payload = dict(registry)
            if local_supervisor_id:
                payload.setdefault("local_supervisor_id", local_supervisor_id)
            if epoch is not None:
                payload.setdefault("epoch", epoch)
            return (register_sibling_supervisor_capability(payload),)
        if "records" in registry or "siblings" in registry or "local_supervisor_id" in registry:
            snapshot = admit_sibling_supervisor_capability_registry(
                {
                    **dict(registry),
                    **(
                        {"local_supervisor_id": local_supervisor_id}
                        if local_supervisor_id and "local_supervisor_id" not in registry
                        else {}
                    ),
                    **({"epoch": epoch} if epoch is not None and "epoch" not in registry else {}),
                }
            )
            return snapshot.records
        records: list[SiblingSupervisorCapabilityRecord] = []
        for sibling_id, capabilities in registry.items():
            if isinstance(capabilities, Mapping):
                capability_items: Sequence[Any] = (capabilities,)
            elif isinstance(capabilities, Sequence) and not isinstance(
                capabilities, (str, bytes)
            ):
                capability_items = capabilities
            else:
                capability_items = (capabilities,)
            for item in capability_items:
                payload: dict[str, Any]
                if isinstance(item, Mapping):
                    payload = dict(item)
                else:
                    payload = {"capability": item}
                payload.setdefault("sibling_supervisor_id", sibling_id)
                if local_supervisor_id:
                    payload.setdefault("local_supervisor_id", local_supervisor_id)
                if epoch is not None:
                    payload.setdefault("epoch", epoch)
                records.append(register_sibling_supervisor_capability(payload))
        return tuple(records)
    if isinstance(registry, Sequence) and not isinstance(registry, (str, bytes)):
        records = []
        for item in registry:
            if isinstance(item, SiblingSupervisorCapabilityRecord):
                records.append(item)
                continue
            if not isinstance(item, Mapping):
                raise SiblingSupervisorCapabilityRegistryError(
                    "capability registry records must be objects",
                    code="record_invalid",
                )
            payload = dict(item)
            if local_supervisor_id:
                payload.setdefault("local_supervisor_id", local_supervisor_id)
            if epoch is not None:
                payload.setdefault("epoch", epoch)
            records.append(register_sibling_supervisor_capability(payload))
        return tuple(records)
    raise SiblingSupervisorCapabilityRegistryError(
        "capability registry must be a snapshot, mapping, or record sequence",
        code="record_invalid",
    )


def lookup_sibling_supervisor_capability(
    registry: Any,
    sibling_supervisor_id: str,
    capability: str,
    *,
    local_supervisor_id: str | None = None,
    epoch: int | None = None,
) -> SiblingSupervisorCapabilityRecord:
    """Look up one advertised sibling capability without writing state."""

    sibling_id = _capability_text(sibling_supervisor_id, "sibling_supervisor_id")
    capability_id = _admitted_sibling_capability(capability)
    if isinstance(registry, SiblingSupervisorCapabilityRegistrySnapshot):
        if local_supervisor_id and registry.local_supervisor_id != _capability_text(
            local_supervisor_id, "local_supervisor_id"
        ):
            raise SiblingSupervisorCapabilityRegistryError(
                "capability record local supervisor does not match registry",
                code="sibling_identity_mismatch",
            )
        if epoch is not None and int(epoch) < registry.epoch:
            raise SiblingSupervisorCapabilityRegistryError(
                "stale fence epoch",
                code="stale_fence_epoch",
            )
        return registry.lookup(sibling_id, capability_id)
    records = _iter_capability_records(
        registry, local_supervisor_id=local_supervisor_id, epoch=epoch
    )
    matches = [
        item
        for item in records
        if item.sibling_supervisor_id == sibling_id and item.capability == capability_id
    ]
    if not matches:
        raise SiblingSupervisorCapabilityRegistryError(
            f"unregistered sibling capability {sibling_id}:{capability_id}",
            code="unregistered_capability",
        )
    first = matches[0]
    for item in matches[1:]:
        if item.to_dict() != first.to_dict():
            raise SiblingSupervisorCapabilityRegistryError(
                f"conflicting sibling capability {sibling_id}:{capability_id}",
                code="capability_conflict",
            )
    if local_supervisor_id and first.local_supervisor_id != _capability_text(
        local_supervisor_id, "local_supervisor_id"
    ):
        raise SiblingSupervisorCapabilityRegistryError(
            "capability record local supervisor does not match registry",
            code="sibling_identity_mismatch",
        )
    if epoch is not None and int(epoch) < first.epoch:
        raise SiblingSupervisorCapabilityRegistryError(
            "stale fence epoch",
            code="stale_fence_epoch",
        )
    return first


def admit_sibling_supervisor_capability_registry(
    record: Mapping[str, Any],
) -> SiblingSupervisorCapabilityRegistrySnapshot:
    """Admit a closed sibling-capability snapshot without writing state."""

    if not isinstance(record, Mapping):
        raise SiblingSupervisorCapabilityRegistryError(
            "sibling capability registry must be an object",
            code="record_invalid",
        )
    _reject_capability_state_writes(record)
    if record.get("completion_authoritative"):
        raise SiblingSupervisorCapabilityRegistryError(
            "sibling capability admission is not completion authority",
            code="completion_not_authoritative",
        )
    if record.get("catalog_override") or record.get("self_granted"):
        raise SiblingSupervisorCapabilityRegistryError(
            "sibling capability catalog is closed and not self-granted",
            code="catalog_closed",
        )
    local_supervisor_id = _capability_text(
        record.get("local_supervisor_id"), "local_supervisor_id"
    )
    epoch = int(record.get("epoch") or 1)
    if epoch < 1:
        raise SiblingSupervisorCapabilityRegistryError(
            "epoch must be >= 1",
            code="stale_fence_epoch",
        )
    if record.get("stale_epoch"):
        raise SiblingSupervisorCapabilityRegistryError(
            "stale fence epoch",
            code="stale_fence_epoch",
        )
    known = record.get("known_sibling_ids")
    collected: list[SiblingSupervisorCapabilityRecord] = []
    raw_records = record.get("records")
    if raw_records is None:
        raw_records = ()
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        raise SiblingSupervisorCapabilityRegistryError(
            "records must be a sequence",
            code="record_invalid",
        )
    for item in raw_records:
        if isinstance(item, SiblingSupervisorCapabilityRecord):
            payload = item.to_dict()
        elif isinstance(item, Mapping):
            payload = dict(item)
        else:
            raise SiblingSupervisorCapabilityRegistryError(
                "capability registry records must be objects",
                code="record_invalid",
            )
        payload.setdefault("local_supervisor_id", local_supervisor_id)
        payload.setdefault("epoch", epoch)
        if known is not None:
            payload.setdefault("known_sibling_ids", known)
        collected.append(register_sibling_supervisor_capability(payload))
    siblings = record.get("siblings")
    if siblings is not None:
        if not isinstance(siblings, Mapping):
            raise SiblingSupervisorCapabilityRegistryError(
                "siblings must be a mapping of supervisor id to capabilities",
                code="record_invalid",
            )
        collected.extend(
            _iter_capability_records(
                siblings, local_supervisor_id=local_supervisor_id, epoch=epoch
            )
        )
    unique: dict[str, SiblingSupervisorCapabilityRecord] = {}
    for item in collected:
        existing = unique.get(item.registry_key)
        if existing is not None and existing.to_dict() != item.to_dict():
            raise SiblingSupervisorCapabilityRegistryError(
                f"conflicting sibling capability {item.registry_key!r}",
                code="capability_conflict",
            )
        unique[item.registry_key] = item
    collected = list(unique.values())
    _ = bool(record.get("worker_assertion"))
    digest = _registry_digest_for(
        local_supervisor_id=local_supervisor_id, epoch=epoch, records=collected
    )
    supplied = record.get("registry_digest")
    if supplied is not None and supplied != digest:
        raise SiblingSupervisorCapabilityRegistryError(
            "sibling capability registry digest does not match contents",
            code="registry_digest_mismatch",
        )
    return SiblingSupervisorCapabilityRegistrySnapshot(
        local_supervisor_id=local_supervisor_id,
        epoch=epoch,
        records=tuple(collected),
        registry_digest=digest,
    )


class SupervisorFabric:
    """Fenced coordination carrier for sibling-supervisor event admission."""

    INTERFACE: Final[str] = SUPERVISOR_FABRIC_INTERFACE
    SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING: Final[str] = (
        SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING
    )
    SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING: Final[str] = (
        SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING
    )

    def __init__(
        self,
        *,
        supervisor_id: str,
        epoch: int = 1,
        capability: str = "event-exchange",
        known_sibling_ids: Sequence[str] = (),
        sibling_capabilities: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self._supervisor_id = _text(supervisor_id, "supervisor_id")
        self._epoch = int(epoch or 1)
        if self._epoch < 1:
            raise SupervisorFabricError("stale fence epoch")
        self._capability = _text(capability, "capability")
        self._known_sibling_ids = tuple(
            _text(item, "known_sibling_id") for item in known_sibling_ids
        )
        self._sibling_capabilities: dict[
            tuple[str, str], SiblingSupervisorCapabilityRecord
        ] = {}
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

    def issue_fence(self, record: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        payload: dict[str, Any] = dict(record or {})
        payload.setdefault("supervisor_id", self._supervisor_id)
        payload.setdefault("capability", self._capability)
        payload.setdefault("epoch", self._epoch)
        return issue_fence(payload)

    def register_sibling_capability(
        self, record: Mapping[str, Any]
    ) -> SiblingSupervisorCapabilityRecord:
        payload: dict[str, Any] = dict(record)
        payload.setdefault("local_supervisor_id", self._supervisor_id)
        payload.setdefault("epoch", record.get("epoch", self._epoch))
        if self._known_sibling_ids and "known_sibling_ids" not in payload:
            payload["known_sibling_ids"] = self._known_sibling_ids
        admitted = register_sibling_supervisor_capability(payload)
        if admitted.epoch < self._epoch:
            raise SiblingSupervisorCapabilityRegistryError(
                "stale fence epoch",
                code="stale_fence_epoch",
            )
        key = (admitted.sibling_supervisor_id, admitted.capability)
        existing = self._sibling_capabilities.get(key)
        if existing is not None:
            if (
                existing.effect != admitted.effect
                or existing.local_supervisor_id != admitted.local_supervisor_id
            ):
                raise SiblingSupervisorCapabilityRegistryError(
                    "conflicting sibling capability advertisement",
                    code="capability_conflict",
                )
            if admitted.epoch < existing.epoch:
                raise SiblingSupervisorCapabilityRegistryError(
                    "stale fence epoch",
                    code="stale_fence_epoch",
                )
        self._sibling_capabilities[key] = admitted
        return admitted

    def lookup_sibling_capability(
        self, sibling_supervisor_id: str, capability: str
    ) -> SiblingSupervisorCapabilityRecord:
        return lookup_sibling_supervisor_capability(
            self.sibling_capability_snapshot(),
            sibling_supervisor_id,
            capability,
            local_supervisor_id=self._supervisor_id,
            epoch=self._epoch,
        )

    def sibling_capability_snapshot(self) -> SiblingSupervisorCapabilityRegistrySnapshot:
        records = tuple(self._sibling_capabilities.values())
        return SiblingSupervisorCapabilityRegistrySnapshot(
            local_supervisor_id=self._supervisor_id,
            epoch=self._epoch,
            records=records,
            registry_digest=_registry_digest_for(
                local_supervisor_id=self._supervisor_id,
                epoch=self._epoch,
                records=records,
            ),
        )

    def validate_sibling_event(
        self, record: Mapping[str, Any]
    ) -> SiblingSupervisorEventAdmission:
        payload: dict[str, Any] = dict(record)
        payload.setdefault("local_supervisor_id", self._supervisor_id)
        payload.setdefault("epoch", record.get("epoch", self._epoch))
        if self._known_sibling_ids and "known_sibling_ids" not in payload:
            payload["known_sibling_ids"] = self._known_sibling_ids
        if self._sibling_capabilities and "capability_registry" not in payload:
            payload["capability_registry"] = self.sibling_capability_snapshot().to_dict()
        return validate_sibling_supervisor_event(payload)


__all__ = [
    "ADMITTED_SIBLING_CAPABILITIES",
    "ALLOWED_SIBLING_EFFECTS",
    "CANONICAL_EVENT_FORBIDDEN_FIELDS",
    "CANONICAL_EVENT_INTERFACE",
    "CANONICAL_EVENT_REQUIRED_FIELDS",
    "CANONICAL_EVENT_SCHEMA_ID",
    "DATABASE_EVENT_LOG_INTERFACE",
    "EVENT_CURSOR_INTERFACE",
    "FORBIDDEN_SIBLING_CAPABILITIES",
    "SIBLING_CAPABILITY_DEFAULT_EFFECT",
    "SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_BINDING",
    "SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_CONSUMES",
    "SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_INTERFACE",
    "SIBLING_SUPERVISOR_CAPABILITY_REGISTRY_SCHEMA",
    "SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING",
    "SIBLING_SUPERVISOR_EVENT_VALIDATION_CONSUMES",
    "SIBLING_SUPERVISOR_EVENT_VALIDATION_INTERFACE",
    "SIBLING_SUPERVISOR_EVENT_VALIDATION_SCHEMA",
    "SUPERVISOR_FABRIC_INTERFACE",
    "SiblingSupervisorCapabilityRecord",
    "SiblingSupervisorCapabilityRegistryError",
    "SiblingSupervisorCapabilityRegistrySnapshot",
    "SiblingSupervisorEventAdmission",
    "SiblingSupervisorEventValidationError",
    "SupervisorFabric",
    "SupervisorFabricError",
    "admit_sibling_supervisor_capability_registry",
    "issue_fence",
    "lookup_sibling_supervisor_capability",
    "register_sibling_supervisor_capability",
    "sibling_supervisor_capability_catalog",
    "validate_sibling_supervisor_event",
]
