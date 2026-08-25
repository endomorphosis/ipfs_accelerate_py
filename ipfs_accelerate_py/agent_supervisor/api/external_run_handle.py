"""Durable detach/reconnect handle for admitted external-agent runs (EAAEF-114).

``ExternalRunHandle`` is the client-held continuation of EAAEF-110.  Its public
identity is ``run_id``, ``cursor`` and ``authority_id``.  ``serialize`` /
``deserialize`` reattach after a client process restart.  Host-supervisor
restart is survived by restoring the snapshot onto a new in-process
``ExternalHandoffAPI``.  ``resume_from`` returns only events after the supplied
cursor.  ``steer`` and ``cancel`` require the admitted authority id.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import CanonicalContract
from .external_handoff import (
    CONTRACT_VERSION,
    MAX_EVENTS,
    ExternalHandoffAPI,
    ExternalHandoffAPIError,
    ExternalHandoffAuthorityError,
    ExternalHandoffReceipt,
    RequestLike,
    _RunRecord,
    _optional_text,
    _reject_private_material,
    _reject_unknown,
    _require_schema,
    _text,
)

SCHEMA_VERSION: Final[int] = CONTRACT_VERSION

EXTERNAL_RUN_HANDLE_INTERFACE: Final[str] = "ExternalRunHandle@1"
EXTERNAL_RUN_HANDLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-run-handle@1"
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
        "snapshot_id",
    }
)
_STATE_FIELDS: Final[tuple[str, ...]] = (
    "run_id",
    "cursor",
    "authority_id",
    "principal_id",
    "worker_principal_id",
    "session_id",
    "repository_id",
    "objective_id",
    "origin_operation",
    "admitted",
    "run_status",
    "idempotency_key",
    "events",
)


def _event_record(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ExternalHandoffAPIError(
            "run event must be an object", reason_code="malformed"
        )
    content_id = _text(value.get("content_id"), "event content_id")
    kind = _optional_text(value.get("kind", ""), "event kind")
    extra = set(value).difference({"content_id", "kind"})
    if extra:
        raise ExternalHandoffAPIError(
            "run event contains unsupported fields; rebuild its canonical payload",
            reason_code="malformed",
        )
    record = {"content_id": content_id}
    if kind:
        record["kind"] = kind
    return record


def _events(values: Any) -> tuple[dict[str, str], ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ExternalHandoffAPIError(
            "events must be a sequence of event objects", reason_code="malformed"
        )
    else:
        items = values
    if len(items) > MAX_EVENTS:
        raise ExternalHandoffAPIError(
            "run event stream exceeds its bound", reason_code="bounds"
        )
    return tuple(_event_record(item) for item in items)


def _bool(value: Any, name: str, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ExternalHandoffAPIError(f"{name} must be a boolean", reason_code="malformed")
    return value


def _decode_payload(payload: Mapping[str, Any] | str | bytes) -> Mapping[str, Any]:
    if isinstance(payload, (str, bytes, bytearray)):
        try:
            decoded = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ExternalHandoffAPIError(
                "run handle JSON is malformed", reason_code="malformed"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise ExternalHandoffAPIError(
                "run handle payload must be an object", reason_code="malformed"
            )
        return decoded
    if not isinstance(payload, Mapping):
        raise ExternalHandoffAPIError(
            "run handle payload must be an object", reason_code="malformed"
        )
    return payload


@dataclass(frozen=True)
class ExternalRunEvents:
    """Events strictly after a continuation cursor."""

    run_id: str
    cursor: str
    events: tuple[Mapping[str, str], ...]

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(item["content_id"] for item in self.events)

    def __iter__(self) -> Iterator[Mapping[str, str]]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)


@dataclass(frozen=True)
class ExternalRunHandleState(CanonicalContract):
    """Canonical serialized reconnect state.  Not itself mutation authority."""

    SCHEMA: ClassVar[str] = EXTERNAL_RUN_HANDLE_SCHEMA

    run_id: str
    cursor: str
    authority_id: str
    principal_id: str = ""
    worker_principal_id: str = ""
    session_id: str = ""
    repository_id: str = ""
    objective_id: str = ""
    origin_operation: str = "handoff"
    admitted: bool = True
    run_status: str = "running"
    idempotency_key: str = ""
    events: tuple[Mapping[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        object.__setattr__(self, "cursor", _optional_text(self.cursor, "cursor"))
        object.__setattr__(self, "authority_id", _text(self.authority_id, "authority_id"))
        object.__setattr__(self, "principal_id", _optional_text(self.principal_id, "principal_id"))
        object.__setattr__(
            self,
            "worker_principal_id",
            _optional_text(self.worker_principal_id, "worker_principal_id"),
        )
        object.__setattr__(self, "session_id", _optional_text(self.session_id, "session_id"))
        object.__setattr__(
            self, "repository_id", _optional_text(self.repository_id, "repository_id")
        )
        object.__setattr__(
            self, "objective_id", _optional_text(self.objective_id, "objective_id")
        )
        origin = _optional_text(self.origin_operation, "origin_operation") or "handoff"
        object.__setattr__(self, "origin_operation", origin)
        object.__setattr__(self, "admitted", _bool(self.admitted, "admitted", default=True))
        object.__setattr__(
            self,
            "run_status",
            _optional_text(self.run_status, "run_status") or "running",
        )
        object.__setattr__(
            self, "idempotency_key", _optional_text(self.idempotency_key, "idempotency_key")
        )
        frozen_events = tuple(MappingProxyType(dict(item)) for item in _events(self.events))
        object.__setattr__(self, "events", frozen_events)
        _reject_private_material(self._payload(), name="external run handle")

    @property
    def snapshot_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": EXTERNAL_RUN_HANDLE_INTERFACE,
            "contract_version": CONTRACT_VERSION,
            "run_id": self.run_id,
            "cursor": self.cursor,
            "authority_id": self.authority_id,
            "principal_id": self.principal_id,
            "worker_principal_id": self.worker_principal_id,
            "session_id": self.session_id,
            "repository_id": self.repository_id,
            "objective_id": self.objective_id,
            "origin_operation": self.origin_operation,
            "admitted": self.admitted,
            "run_status": self.run_status,
            "idempotency_key": self.idempotency_key,
            "events": [dict(item) for item in self.events],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalRunHandleState":
        if not isinstance(payload, Mapping):
            raise ExternalHandoffAPIError(
                "run handle payload must be an object", reason_code="malformed"
            )
        _require_schema(
            payload,
            cls.SCHEMA,
            EXTERNAL_RUN_HANDLE_INTERFACE,
            artifact_name="external run handle",
        )
        _reject_unknown(
            payload,
            tuple(_WIRE_FIELDS.union(_STATE_FIELDS)),
            name="external run handle",
        )
        result = cls(
            run_id=payload.get("run_id", ""),
            cursor=payload.get("cursor", ""),
            authority_id=payload.get("authority_id", ""),
            principal_id=payload.get("principal_id", ""),
            worker_principal_id=payload.get("worker_principal_id", ""),
            session_id=payload.get("session_id", ""),
            repository_id=payload.get("repository_id", ""),
            objective_id=payload.get("objective_id", ""),
            origin_operation=payload.get("origin_operation", "handoff"),
            admitted=_bool(payload.get("admitted"), "admitted", default=True),
            run_status=payload.get("run_status", "running"),
            idempotency_key=payload.get("idempotency_key", ""),
            events=payload.get("events", ()),
        )
        claimed = None
        for name in ("content_id", "cid", "identity", "canonical_id", "snapshot_id"):
            value = payload.get(name)
            if value not in (None, ""):
                claimed = value
                break
        if claimed not in (None, "") and claimed != result.content_id:
            raise ExternalHandoffAPIError(
                "run handle content identity does not match payload",
                reason_code="identity_mismatch",
            )
        return result


@dataclass
class ExternalRunHandle:
    """Live reconnect handle bound to an in-process EAAEF-110 API."""

    run_id: str
    cursor: str
    authority_id: str
    principal_id: str = ""
    worker_principal_id: str = ""
    session_id: str = ""
    repository_id: str = ""
    objective_id: str = ""
    origin_operation: str = "handoff"
    admitted: bool = True
    run_status: str = "running"
    idempotency_key: str = ""
    events: tuple[Mapping[str, str], ...] = ()
    api: ExternalHandoffAPI | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.run_id = _text(self.run_id, "run_id")
        self.cursor = _optional_text(self.cursor, "cursor")
        self.authority_id = _text(self.authority_id, "authority_id")
        self.principal_id = _optional_text(self.principal_id, "principal_id")
        self.worker_principal_id = _optional_text(
            self.worker_principal_id, "worker_principal_id"
        )
        self.session_id = _optional_text(self.session_id, "session_id")
        self.repository_id = _optional_text(self.repository_id, "repository_id")
        self.objective_id = _optional_text(self.objective_id, "objective_id")
        self.origin_operation = (
            _optional_text(self.origin_operation, "origin_operation") or "handoff"
        )
        self.admitted = _bool(self.admitted, "admitted", default=True)
        self.run_status = _optional_text(self.run_status, "run_status") or "running"
        self.idempotency_key = _optional_text(self.idempotency_key, "idempotency_key")
        self.events = tuple(MappingProxyType(dict(item)) for item in _events(self.events))
        if self.api is not None:
            self._restore_if_missing()
            self._pull()

    @classmethod
    def from_receipt(
        cls,
        receipt: ExternalHandoffReceipt,
        *,
        api: ExternalHandoffAPI,
    ) -> "ExternalRunHandle":
        """Bind a durable handle to an admitted (or preview) receipt."""

        return cls(
            run_id=receipt.run_id,
            cursor=receipt.cursor,
            authority_id=receipt.authority_id,
            principal_id=receipt.principal_id,
            worker_principal_id=receipt.worker_principal_id,
            session_id=receipt.session_id,
            repository_id=receipt.repository_id,
            origin_operation=receipt.operation,
            admitted=receipt.verdict == "admitted",
            run_status=receipt.run_status,
            events=tuple({"content_id": event_id} for event_id in receipt.event_ids),
            api=api,
        )

    @classmethod
    def handoff(
        cls,
        request: RequestLike,
        *,
        api: ExternalHandoffAPI | None = None,
    ) -> "ExternalRunHandle":
        """Admit a run through EAAEF-110 and return its durable handle."""

        bound = api if api is not None else ExternalHandoffAPI()
        return cls.from_receipt(bound.handoff(request), api=bound)

    @classmethod
    def from_state(
        cls,
        state: ExternalRunHandleState,
        *,
        api: ExternalHandoffAPI | None = None,
    ) -> "ExternalRunHandle":
        return cls(
            run_id=state.run_id,
            cursor=state.cursor,
            authority_id=state.authority_id,
            principal_id=state.principal_id,
            worker_principal_id=state.worker_principal_id,
            session_id=state.session_id,
            repository_id=state.repository_id,
            objective_id=state.objective_id,
            origin_operation=state.origin_operation,
            admitted=state.admitted,
            run_status=state.run_status,
            idempotency_key=state.idempotency_key,
            events=tuple(dict(item) for item in state.events),
            api=api,
        )

    def state(self) -> ExternalRunHandleState:
        return ExternalRunHandleState(
            run_id=self.run_id,
            cursor=self.cursor,
            authority_id=self.authority_id,
            principal_id=self.principal_id,
            worker_principal_id=self.worker_principal_id,
            session_id=self.session_id,
            repository_id=self.repository_id,
            objective_id=self.objective_id,
            origin_operation=self.origin_operation,
            admitted=self.admitted,
            run_status=self.run_status,
            idempotency_key=self.idempotency_key,
            events=tuple(dict(item) for item in self.events),
        )

    def serialize(self) -> dict[str, Any]:
        """Return canonical reconnect state a restarted client can reattach."""

        record = self.state().to_record()
        record["snapshot_id"] = record["content_id"]
        return record

    def to_dict(self) -> dict[str, Any]:
        return self.serialize()

    def to_json(self) -> str:
        return self.state().to_json()

    @classmethod
    def deserialize(
        cls,
        payload: Mapping[str, Any] | str | bytes,
        *,
        api: ExternalHandoffAPI | None = None,
    ) -> "ExternalRunHandle":
        """Reattach a handle after client or host-supervisor restart."""

        return cls.from_state(
            ExternalRunHandleState.from_dict(_decode_payload(payload)),
            api=api,
        )

    from_dict = deserialize

    @classmethod
    def from_json(
        cls,
        payload: str,
        *,
        api: ExternalHandoffAPI | None = None,
    ) -> "ExternalRunHandle":
        return cls.deserialize(payload, api=api)

    def detach(self) -> dict[str, Any]:
        """Snapshot continuation state and drop the live host binding."""

        payload = self.serialize()
        self.api = None
        return payload

    def reconnect(self, api: ExternalHandoffAPI | None = None) -> "ExternalRunHandle":
        """Bind this handle to a (possibly restarted) host supervisor."""

        self.api = api if api is not None else ExternalHandoffAPI()
        self._restore_if_missing()
        self._pull()
        return self

    def resume_from(self, cursor: str) -> ExternalRunEvents:
        """Return events after ``cursor`` only.  The cursor event is excluded."""

        cursor_text = _optional_text(cursor, "cursor")
        if self.api is not None:
            self._restore_if_missing()
            self._pull()
        window = self._events_after(cursor_text)
        if window:
            self.cursor = window[-1]["content_id"]
        elif cursor_text:
            self.cursor = cursor_text
        return ExternalRunEvents(run_id=self.run_id, cursor=self.cursor, events=window)

    def steer(
        self,
        instruction: str,
        *,
        authority_id: str | None = None,
    ) -> ExternalHandoffReceipt:
        presented = self._require_authority(authority_id)
        api = self._require_api()
        receipt = api.steer(self._control_request(presented, instruction=instruction))
        self._ingest_receipt(receipt)
        return receipt

    def cancel(self, *, authority_id: str | None = None) -> ExternalHandoffReceipt:
        presented = self._require_authority(authority_id)
        api = self._require_api()
        receipt = api.cancel(self._control_request(presented))
        self._ingest_receipt(receipt)
        return receipt

    def _events_after(self, cursor: str) -> tuple[Mapping[str, str], ...]:
        if not cursor:
            return tuple(self.events)
        ids = [item["content_id"] for item in self.events]
        try:
            index = ids.index(cursor) + 1
        except ValueError as exc:
            raise ExternalHandoffAPIError(
                "resume cursor does not match the run event stream",
                reason_code="unknown_cursor",
            ) from exc
        return tuple(self.events[index:])

    def _require_api(self) -> ExternalHandoffAPI:
        if self.api is None:
            raise ExternalHandoffAPIError(
                "run handle is detached; reconnect before steering or cancelling",
                reason_code="detached",
            )
        self._restore_if_missing()
        return self.api

    def _host_run(self) -> _RunRecord | None:
        if self.api is None:
            return None
        return self.api._runs.get(self.run_id)

    def _require_authority(self, authority_id: str | None) -> str:
        presented = (
            self.authority_id
            if authority_id in (None, "")
            else _text(authority_id, "authority_id")
        )
        if not presented:
            raise ExternalHandoffAuthorityError(
                "authority_id is required", reason_code="authority_mismatch"
            )
        run = self._host_run()
        admitted = run.authority_id if run is not None else self.authority_id
        if presented != admitted or presented != self.authority_id:
            raise ExternalHandoffAuthorityError(
                "run identity and authority id must match the admitted run",
                reason_code="authority_mismatch",
            )
        return presented

    def _control_request(
        self,
        authority_id: str,
        *,
        instruction: str = "",
    ) -> dict[str, str]:
        payload = {
            "principal_id": self.principal_id or "principal:handle",
            "worker_principal_id": self.worker_principal_id,
            "run_id": self.run_id,
            "authority_id": authority_id,
            "session_id": self.session_id,
        }
        if instruction:
            payload["instruction"] = instruction
        return payload

    def _restore_if_missing(self) -> None:
        if self.api is None or self.run_id in self.api._runs:
            return
        events = [dict(item) for item in self.events]
        tip = events[-1]["content_id"] if events else self.cursor
        self.api._runs[self.run_id] = _RunRecord(
            run_id=self.run_id,
            authority_id=self.authority_id,
            principal_id=self.principal_id,
            worker_principal_id=self.worker_principal_id,
            session_id=self.session_id,
            repository_id=self.repository_id,
            objective_id=self.objective_id,
            origin_operation=self.origin_operation,
            admitted=self.admitted,
            run_status=self.run_status,
            cursor=tip,
            events=events,
            idempotency_key=self.idempotency_key,
        )
        if self.idempotency_key and self.principal_id:
            self.api._idempotency[
                (self.origin_operation, self.principal_id, self.idempotency_key)
            ] = self.run_id

    def _pull(self) -> None:
        run = self._host_run()
        if run is None:
            return
        self.events = tuple(
            MappingProxyType(_event_record(item)) for item in run.events
        )
        self.run_status = run.run_status
        self.principal_id = self.principal_id or run.principal_id
        self.worker_principal_id = self.worker_principal_id or run.worker_principal_id
        self.session_id = self.session_id or run.session_id
        self.repository_id = self.repository_id or run.repository_id
        self.objective_id = self.objective_id or run.objective_id
        self.origin_operation = self.origin_operation or run.origin_operation
        self.admitted = run.admitted
        self.idempotency_key = self.idempotency_key or run.idempotency_key
        if not self.cursor:
            self.cursor = run.cursor

    def _ingest_receipt(self, receipt: ExternalHandoffReceipt) -> None:
        self.run_status = receipt.run_status
        self.cursor = receipt.cursor
        if receipt.event_ids:
            known = {item["content_id"]: dict(item) for item in self.events}
            refreshed: list[dict[str, str]] = []
            for event_id in receipt.event_ids:
                existing = known.get(event_id)
                refreshed.append(existing if existing is not None else {"content_id": event_id})
            self.events = tuple(MappingProxyType(item) for item in refreshed)
        self._pull()


def serialize_run_handle(handle: ExternalRunHandle) -> dict[str, Any]:
    return handle.serialize()


def deserialize_run_handle(
    payload: Mapping[str, Any] | str | bytes,
    *,
    api: ExternalHandoffAPI | None = None,
) -> ExternalRunHandle:
    return ExternalRunHandle.deserialize(payload, api=api)


__all__ = (
    "CONTRACT_VERSION",
    "EXTERNAL_RUN_HANDLE_INTERFACE",
    "EXTERNAL_RUN_HANDLE_SCHEMA",
    "SCHEMA_VERSION",
    "ExternalRunEvents",
    "ExternalRunHandle",
    "ExternalRunHandleState",
    "deserialize_run_handle",
    "serialize_run_handle",
)
