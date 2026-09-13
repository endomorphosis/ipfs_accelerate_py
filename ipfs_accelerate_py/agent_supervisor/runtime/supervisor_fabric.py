"""Supervisor capability and fenced coordination contracts.

Sibling-supervisor event validation is a binding of this fabric, not a second
event log, bus, or state owner. Sibling supervisors exchange canonical event
envelopes and receipts. They never write DuckDB or DuckLake, never consume
``DatabaseEventLog@1``, and never terminalize tasks. A worker or model
assertion cannot admit a sibling event.

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


class SupervisorFabric:
    """Fenced coordination carrier for sibling-supervisor event admission."""

    INTERFACE: Final[str] = SUPERVISOR_FABRIC_INTERFACE
    SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING: Final[str] = (
        SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING
    )

    def __init__(
        self,
        *,
        supervisor_id: str,
        epoch: int = 1,
        capability: str = "event-exchange",
        known_sibling_ids: Sequence[str] = (),
    ) -> None:
        self._supervisor_id = _text(supervisor_id, "supervisor_id")
        self._epoch = int(epoch or 1)
        if self._epoch < 1:
            raise SupervisorFabricError("stale fence epoch")
        self._capability = _text(capability, "capability")
        self._known_sibling_ids = tuple(
            _text(item, "known_sibling_id") for item in known_sibling_ids
        )

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

    def validate_sibling_event(
        self, record: Mapping[str, Any]
    ) -> SiblingSupervisorEventAdmission:
        payload: dict[str, Any] = dict(record)
        payload.setdefault("local_supervisor_id", self._supervisor_id)
        payload.setdefault("epoch", record.get("epoch", self._epoch))
        if self._known_sibling_ids and "known_sibling_ids" not in payload:
            payload["known_sibling_ids"] = self._known_sibling_ids
        return validate_sibling_supervisor_event(payload)


__all__ = [
    "ALLOWED_SIBLING_EFFECTS",
    "CANONICAL_EVENT_FORBIDDEN_FIELDS",
    "CANONICAL_EVENT_INTERFACE",
    "CANONICAL_EVENT_REQUIRED_FIELDS",
    "CANONICAL_EVENT_SCHEMA_ID",
    "DATABASE_EVENT_LOG_INTERFACE",
    "EVENT_CURSOR_INTERFACE",
    "SIBLING_SUPERVISOR_EVENT_VALIDATION_BINDING",
    "SIBLING_SUPERVISOR_EVENT_VALIDATION_CONSUMES",
    "SIBLING_SUPERVISOR_EVENT_VALIDATION_INTERFACE",
    "SIBLING_SUPERVISOR_EVENT_VALIDATION_SCHEMA",
    "SUPERVISOR_FABRIC_INTERFACE",
    "SiblingSupervisorEventAdmission",
    "SiblingSupervisorEventValidationError",
    "SupervisorFabric",
    "SupervisorFabricError",
    "issue_fence",
    "validate_sibling_supervisor_event",
]
