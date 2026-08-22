"""Typed lifecycle and assurance events for external-agent runs (EAAEF-130).

Frozen ``ExternalLifecycleEvent@1`` records are the public observability
boundary from handoff through terminal state.  Every event binds exact
run/task/attempt/fence/artifact identities and a strictly increasing sequence
plus continuation cursor.  Transcript bodies, secrets, and hidden
chain-of-thought are rejected; public events carry identities only.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final, TypeVar

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
)


CONTRACT_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = CONTRACT_VERSION

EXTERNAL_LIFECYCLE_EVENT_INTERFACE: Final[str] = "ExternalLifecycleEvent@1"
EXTERNAL_LIFECYCLE_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-lifecycle-event@1"
)

MAX_ID_BYTES: Final[int] = 256
MAX_EVENTS: Final[int] = 4_096
MAX_RECORD_BYTES: Final[int] = 16_384

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_CIDV1_RE: Final[re.Pattern[str]] = re.compile(r"^b[a-z2-7]{20,}$")
_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/=@+-]*$"
)

_HIDDEN_CHAIN_OF_THOUGHT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "chain_of_thought",
        "cot",
        "hidden_chain_of_thought",
        "hidden_cot",
        "hidden_reasoning",
        "hidden_thoughts",
        "internal_monologue",
        "model_thoughts",
        "private_reasoning",
        "private_thinking",
        "scratchpad",
        "thinking",
        "thinking_blocks",
        "thinking_private",
        "thinking_text",
    }
)
_PRIVATE_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "hidden_witness",
        "password",
        "private_key",
        "private_premise",
        "private_witness",
        "refresh_token",
        "secret",
        "session_token",
        "transcript_body",
        "witness",
    }
)
_TRANSCRIPT_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "full_transcript",
        "raw_bytes",
        "raw_export",
        "raw_transcript",
        "transcript",
        "transcript_body",
        "transcript_text",
    }
)

TEnum = TypeVar("TEnum", bound=Enum)

REQUIRED_IDENTITIES: Final[tuple[str, ...]] = (
    "run_id",
    "task_id",
    "attempt_id",
    "fence_token",
    "artifact_cid",
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
        "event_id",
        "kind",
        "sequence",
        "continuation_cursor",
        "run_id",
        "task_id",
        "attempt_id",
        "fence_token",
        "artifact_cid",
        "created_at_ms",
    }
)


class LifecycleEventError(ContractValidationError):
    """Malformed or privacy-unsafe external lifecycle event."""


class LifecycleIdentityError(LifecycleEventError):
    """A required run/task/attempt/fence/artifact identity is missing."""


class LifecycleOrderError(LifecycleEventError):
    """Sequence, continuation cursor, or kind order is not strictly advancing."""


class LifecyclePrivacyError(LifecycleEventError):
    """Transcript bodies, secrets, or hidden reasoning appeared on an event."""


class LifecycleEventKind(str, Enum):
    """Closed lifecycle and assurance kinds from handoff through terminal."""

    HANDOFF_ACCEPTED = "handoff_accepted"
    CLAIMED = "claimed"
    LEASED = "leased"
    LAUNCHED = "launched"
    CHECKPOINTED = "checkpointed"
    VERIFIED = "verified"
    MERGE_PROPOSED = "merge_proposed"
    MERGE_ACCEPTED = "merge_accepted"
    TERMINAL_COMPLETED = "terminal_completed"
    TERMINAL_CANCELLED = "terminal_cancelled"
    TERMINAL_FAILED = "terminal_failed"
    TERMINAL_QUARANTINED = "terminal_quarantined"

    @property
    def is_terminal(self) -> bool:
        return self.value.startswith("terminal_")


CANONICAL_SUCCESS_KINDS: Final[tuple[LifecycleEventKind, ...]] = (
    LifecycleEventKind.HANDOFF_ACCEPTED,
    LifecycleEventKind.CLAIMED,
    LifecycleEventKind.LEASED,
    LifecycleEventKind.LAUNCHED,
    LifecycleEventKind.CHECKPOINTED,
    LifecycleEventKind.VERIFIED,
    LifecycleEventKind.MERGE_PROPOSED,
    LifecycleEventKind.MERGE_ACCEPTED,
    LifecycleEventKind.TERMINAL_COMPLETED,
)

TERMINAL_KINDS: Final[frozenset[LifecycleEventKind]] = frozenset(
    kind for kind in LifecycleEventKind if kind.is_terminal
)

_KIND_RANK: Final[Mapping[LifecycleEventKind, int]] = {
    LifecycleEventKind.HANDOFF_ACCEPTED: 0,
    LifecycleEventKind.CLAIMED: 1,
    LifecycleEventKind.LEASED: 2,
    LifecycleEventKind.LAUNCHED: 3,
    LifecycleEventKind.CHECKPOINTED: 4,
    LifecycleEventKind.VERIFIED: 5,
    LifecycleEventKind.MERGE_PROPOSED: 6,
    LifecycleEventKind.MERGE_ACCEPTED: 7,
    LifecycleEventKind.TERMINAL_COMPLETED: 8,
    LifecycleEventKind.TERMINAL_CANCELLED: 8,
    LifecycleEventKind.TERMINAL_FAILED: 8,
    LifecycleEventKind.TERMINAL_QUARANTINED: 8,
}

_REPEATABLE_KINDS: Final[frozenset[LifecycleEventKind]] = frozenset(
    {LifecycleEventKind.CHECKPOINTED}
)


def _normalize_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _key_is_forbidden(key: str) -> str | None:
    normalized = _normalize_key(key)
    if normalized in _HIDDEN_CHAIN_OF_THOUGHT_KEYS:
        return "hidden_chain_of_thought"
    if normalized in _TRANSCRIPT_BODY_KEYS:
        return "transcript_body"
    if any(
        normalized == marker or normalized.endswith("_" + marker) or marker in normalized
        for marker in _PRIVATE_FIELD_MARKERS
    ):
        return "private_material"
    return None


def _reject_forbidden_keys(value: Any, *, name: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            reason = _key_is_forbidden(str(raw_key))
            if reason == "hidden_chain_of_thought":
                raise LifecyclePrivacyError(
                    f"{name} must not represent hidden chain-of-thought"
                )
            if reason == "transcript_body":
                raise LifecyclePrivacyError(
                    f"{name} must not embed transcript bodies; "
                    "use content-addressed artifact identities"
                )
            if reason == "private_material":
                raise LifecyclePrivacyError(
                    f"{name} must not contain secrets or private material"
                )
            _reject_forbidden_keys(item, name=name)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for item in value:
            _reject_forbidden_keys(item, name=name)


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = MAX_ID_BYTES,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if value is None:
        result = ""
    elif not isinstance(value, str):
        raise LifecycleEventError(f"{name} must be a string")
    else:
        result = value.strip()
    if required and not result:
        raise LifecycleIdentityError(f"{name} is required")
    if "\x00" in result:
        raise LifecycleEventError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > max_bytes:
        raise LifecycleEventError(f"{name} exceeds {max_bytes} UTF-8 bytes")
    if result and pattern is not None and pattern.fullmatch(result) is None:
        raise LifecycleIdentityError(f"{name} is not a permitted identity")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LifecycleEventError(f"{name} must be a non-negative integer")
    return value


def _enum(value: Any, enum_type: type[TEnum], name: str) -> TEnum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise LifecycleEventError(f"{name} must be one of: {allowed}") from exc


def _identity(value: Any, name: str) -> str:
    text = _text(value, name)
    if _SHA256_RE.fullmatch(text) or _CIDV1_RE.fullmatch(text) or _ID_RE.fullmatch(text):
        return text
    raise LifecycleIdentityError(f"{name} is not a permitted identity")


def _artifact_cid(value: Any) -> str:
    text = _text(value, "artifact_cid")
    if _SHA256_RE.fullmatch(text) or _CIDV1_RE.fullmatch(text) or _ID_RE.fullmatch(text):
        return text
    raise LifecycleIdentityError("artifact_cid is not a permitted identity")


def _reject_unknown(payload: Mapping[str, Any]) -> None:
    extra = set(payload).difference(_WIRE_FIELDS)
    if extra:
        raise LifecycleEventError(
            "external lifecycle event contains unsupported fields; "
            "rebuild its canonical payload"
        )


def _require_schema(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise LifecycleEventError("external lifecycle event payload must be an object")
    schema = payload.get("schema")
    if schema not in (None, "", EXTERNAL_LIFECYCLE_EVENT_SCHEMA):
        raise LifecycleEventError(
            f"unsupported schema {schema!r}; expected {EXTERNAL_LIFECYCLE_EVENT_SCHEMA}"
        )
    interface = payload.get("interface")
    if interface not in (None, "", EXTERNAL_LIFECYCLE_EVENT_INTERFACE):
        raise LifecycleEventError(
            f"unsupported interface {interface!r}; expected {EXTERNAL_LIFECYCLE_EVENT_INTERFACE}"
        )
    for key in ("contract_version", "schema_version"):
        version = payload.get(key)
        if version not in (None, "", CONTRACT_VERSION):
            raise LifecycleEventError(
                "unsupported contract version; rebuild with ExternalLifecycleEvent@1"
            )


def _claimed_identity(payload: Mapping[str, Any], actual: str) -> None:
    for name in ("content_id", "cid", "identity", "canonical_id", "event_id"):
        claimed = payload.get(name)
        if claimed not in (None, "") and claimed != actual:
            raise LifecycleIdentityError(
                "external lifecycle event content identity does not match payload"
            )


def _kind_rank(kind: LifecycleEventKind) -> int:
    return _KIND_RANK[kind]


@dataclass(frozen=True)
class ExternalLifecycleEvent(CanonicalContract):
    """Frozen, content-addressed lifecycle event with exact identities @1."""

    SCHEMA: ClassVar[str] = EXTERNAL_LIFECYCLE_EVENT_SCHEMA
    INTERFACE: ClassVar[str] = EXTERNAL_LIFECYCLE_EVENT_INTERFACE

    kind: LifecycleEventKind | str
    sequence: int
    run_id: str
    task_id: str
    attempt_id: str
    fence_token: str
    artifact_cid: str
    continuation_cursor: int = -1
    created_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, LifecycleEventKind, "kind")
        )
        object.__setattr__(
            self, "sequence", _nonnegative_int(self.sequence, "sequence")
        )
        expected_cursor = self.sequence + 1
        if self.continuation_cursor < 0:
            object.__setattr__(self, "continuation_cursor", expected_cursor)
        else:
            object.__setattr__(
                self,
                "continuation_cursor",
                _nonnegative_int(self.continuation_cursor, "continuation_cursor"),
            )
            if self.continuation_cursor != expected_cursor:
                raise LifecycleOrderError(
                    "continuation_cursor must be sequence + 1"
                )
        object.__setattr__(self, "run_id", _identity(self.run_id, "run_id"))
        object.__setattr__(self, "task_id", _identity(self.task_id, "task_id"))
        object.__setattr__(
            self, "attempt_id", _identity(self.attempt_id, "attempt_id")
        )
        object.__setattr__(
            self, "fence_token", _identity(self.fence_token, "fence_token")
        )
        object.__setattr__(self, "artifact_cid", _artifact_cid(self.artifact_cid))
        object.__setattr__(
            self,
            "created_at_ms",
            _nonnegative_int(self.created_at_ms, "created_at_ms"),
        )
        _reject_forbidden_keys(self.to_dict(), name="external lifecycle event")
        if len(self.canonical_bytes()) > MAX_RECORD_BYTES:
            raise LifecycleEventError(
                f"external lifecycle event exceeds {MAX_RECORD_BYTES} bytes"
            )

    @property
    def event_id(self) -> str:
        return self.content_id

    @property
    def is_terminal(self) -> bool:
        kind = self.kind
        assert isinstance(kind, LifecycleEventKind)
        return kind.is_terminal

    def _payload(self) -> dict[str, Any]:
        kind = self.kind
        assert isinstance(kind, LifecycleEventKind)
        return {
            "interface": self.INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "kind": kind.value,
            "sequence": self.sequence,
            "continuation_cursor": self.continuation_cursor,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "fence_token": self.fence_token,
            "artifact_cid": self.artifact_cid,
            "created_at_ms": self.created_at_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalLifecycleEvent":
        if isinstance(payload, ExternalLifecycleEvent):
            return payload
        if not isinstance(payload, Mapping):
            raise LifecycleEventError(
                "external lifecycle event payload must be an object"
            )
        _reject_forbidden_keys(payload, name="external lifecycle event")
        _require_schema(payload)
        _reject_unknown(payload)
        result = cls(
            kind=payload.get("kind"),
            sequence=payload.get("sequence"),
            continuation_cursor=payload.get("continuation_cursor", -1),
            run_id=payload.get("run_id"),
            task_id=payload.get("task_id"),
            attempt_id=payload.get("attempt_id"),
            fence_token=payload.get("fence_token"),
            artifact_cid=payload.get("artifact_cid"),
            created_at_ms=payload.get("created_at_ms", 0),
        )
        _claimed_identity(payload, result.content_id)
        return result

    @classmethod
    def from_json(cls, payload: str) -> "ExternalLifecycleEvent":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise LifecycleEventError(
                "external lifecycle event JSON is malformed"
            ) from exc
        return cls.from_dict(value)


def emit_lifecycle_event(
    kind: LifecycleEventKind | str,
    *,
    sequence: int,
    run_id: str,
    task_id: str,
    attempt_id: str,
    fence_token: str,
    artifact_cid: str,
    created_at_ms: int = 0,
) -> ExternalLifecycleEvent:
    """Construct one frozen lifecycle event with bound identities."""

    return ExternalLifecycleEvent(
        kind=kind,
        sequence=sequence,
        run_id=run_id,
        task_id=task_id,
        attempt_id=attempt_id,
        fence_token=fence_token,
        artifact_cid=artifact_cid,
        created_at_ms=created_at_ms,
    )


def validate_lifecycle_sequence(
    events: Sequence[ExternalLifecycleEvent],
) -> tuple[str, ...]:
    """Require strictly increasing sequence/cursor and advancing kind order."""

    if len(events) > MAX_EVENTS:
        raise LifecycleOrderError("lifecycle event sequence exceeds the event limit")
    identities: list[str] = []
    seen: set[str] = set()
    previous_sequence = -1
    previous_cursor = 0
    previous_kind: LifecycleEventKind | None = None
    bound: dict[str, str] | None = None
    for event in events:
        if not isinstance(event, ExternalLifecycleEvent):
            raise LifecycleEventError("lifecycle sequence must contain ExternalLifecycleEvent@1")
        if event.sequence <= previous_sequence:
            raise LifecycleOrderError("event sequence must be strictly increasing")
        if event.continuation_cursor <= previous_cursor:
            raise LifecycleOrderError(
                "continuation cursor must be strictly increasing"
            )
        if event.continuation_cursor != event.sequence + 1:
            raise LifecycleOrderError("continuation_cursor must be sequence + 1")
        kind = event.kind
        assert isinstance(kind, LifecycleEventKind)
        if previous_kind is not None:
            if previous_kind.is_terminal:
                raise LifecycleOrderError(
                    "no lifecycle event may follow a terminal state"
                )
            previous_rank = _kind_rank(previous_kind)
            next_rank = _kind_rank(kind)
            if next_rank < previous_rank:
                raise LifecycleOrderError(
                    f"{kind.value} cannot follow {previous_kind.value}"
                )
            if next_rank == previous_rank and kind not in _REPEATABLE_KINDS:
                raise LifecycleOrderError(
                    f"{kind.value} cannot follow {previous_kind.value}"
                )
        current_bound = {
            "run_id": event.run_id,
            "task_id": event.task_id,
            "attempt_id": event.attempt_id,
            "fence_token": event.fence_token,
            "artifact_cid": event.artifact_cid,
        }
        if bound is None:
            bound = current_bound
        elif bound != current_bound:
            raise LifecycleIdentityError(
                "lifecycle events must keep run/task/attempt/fence/artifact identities bound"
            )
        identity = event.event_id
        if identity in seen:
            raise LifecycleIdentityError(
                "event sequence must not contain duplicate identities"
            )
        seen.add(identity)
        identities.append(identity)
        previous_sequence = event.sequence
        previous_cursor = event.continuation_cursor
        previous_kind = kind
    return tuple(identities)


class ExternalLifecycleEventStream:
    """Ordered emitter of privacy-safe lifecycle events for one bound attempt."""

    def __init__(
        self,
        *,
        run_id: str,
        task_id: str,
        attempt_id: str,
        fence_token: str,
        artifact_cid: str,
        created_at_ms: int = 0,
    ) -> None:
        self._run_id = _identity(run_id, "run_id")
        self._task_id = _identity(task_id, "task_id")
        self._attempt_id = _identity(attempt_id, "attempt_id")
        self._fence_token = _identity(fence_token, "fence_token")
        self._artifact_cid = _artifact_cid(artifact_cid)
        self._created_at_ms = _nonnegative_int(created_at_ms, "created_at_ms")
        self._events: list[ExternalLifecycleEvent] = []

    @property
    def events(self) -> tuple[ExternalLifecycleEvent, ...]:
        return tuple(self._events)

    @property
    def continuation_cursor(self) -> int:
        if not self._events:
            return 0
        return self._events[-1].continuation_cursor

    @property
    def identities(self) -> Mapping[str, str]:
        return {
            "run_id": self._run_id,
            "task_id": self._task_id,
            "attempt_id": self._attempt_id,
            "fence_token": self._fence_token,
            "artifact_cid": self._artifact_cid,
        }

    def emit(
        self,
        kind: LifecycleEventKind | str,
        *,
        created_at_ms: int | None = None,
    ) -> ExternalLifecycleEvent:
        event = emit_lifecycle_event(
            kind,
            sequence=len(self._events),
            run_id=self._run_id,
            task_id=self._task_id,
            attempt_id=self._attempt_id,
            fence_token=self._fence_token,
            artifact_cid=self._artifact_cid,
            created_at_ms=self._created_at_ms if created_at_ms is None else created_at_ms,
        )
        validate_lifecycle_sequence((*self._events, event))
        self._events.append(event)
        return event

    def emit_canonical_success_path(
        self,
        kinds: Sequence[LifecycleEventKind | str] = CANONICAL_SUCCESS_KINDS,
    ) -> tuple[ExternalLifecycleEvent, ...]:
        """Emit handoff through terminal kinds in canonical order."""

        for kind in kinds:
            self.emit(kind)
        return self.events

    def after(self, cursor: int) -> tuple[ExternalLifecycleEvent, ...]:
        """Resume strictly after an exact continuation cursor."""

        position = _nonnegative_int(cursor, "continuation_cursor")
        return tuple(
            event for event in self._events if event.sequence >= position
        )


def decode_lifecycle_event(
    payload: Mapping[str, Any] | ExternalLifecycleEvent,
) -> ExternalLifecycleEvent:
    """Decode one strictly versioned ExternalLifecycleEvent@1 record."""

    if isinstance(payload, ExternalLifecycleEvent):
        return payload
    return ExternalLifecycleEvent.from_dict(payload)


__all__ = (
    "CANONICAL_SUCCESS_KINDS",
    "CONTRACT_VERSION",
    "EXTERNAL_LIFECYCLE_EVENT_INTERFACE",
    "EXTERNAL_LIFECYCLE_EVENT_SCHEMA",
    "REQUIRED_IDENTITIES",
    "SCHEMA_VERSION",
    "TERMINAL_KINDS",
    "ExternalLifecycleEvent",
    "ExternalLifecycleEventStream",
    "LifecycleEventError",
    "LifecycleEventKind",
    "LifecycleIdentityError",
    "LifecycleOrderError",
    "LifecyclePrivacyError",
    "decode_lifecycle_event",
    "emit_lifecycle_event",
    "validate_lifecycle_sequence",
)
