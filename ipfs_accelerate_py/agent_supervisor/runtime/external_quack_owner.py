"""In-memory fenced DuckDB/Quack owner (EAAEF-093).

Quack supplies bounded authenticated multi-reader/multi-writer transport.
Exactly one local owner validates bound envelopes and serializes private
DuckDB transactions. ``failover()`` advances the epoch; a stale owner with
an old epoch is rejected. Remote UPDATE and arbitrary SQL are refused.

This module is process-local. It does not bind, start, or stop live Quack
on :19495, overlay runtime CAS, or issue cryptographic signatures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity


CONTRACT_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = CONTRACT_VERSION

EXTERNAL_QUACK_OWNER_INTERFACE: Final[str] = "ExternalQuackOwner@1"
EXTERNAL_QUACK_OWNER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-quack-owner@1"
)
OWNER_LEASE_INTERFACE: Final[str] = "ExternalQuackOwnerLease@1"
OWNER_LEASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-quack-owner-lease@1"
)
ENVELOPE_INTERFACE: Final[str] = "ExternalQuackEnvelope@1"
ENVELOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-quack-envelope@1"
)
APPLY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-quack-apply-receipt@1"
)
TRANSPORT_INTERFACE: Final[str] = "BoundedQuackTransport@1"
TRANSPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/bounded-quack-transport@1"
)

INITIAL_EPOCH: Final[int] = 1
INITIAL_FENCE: Final[int] = 1
LIVE_QUACK_PORT: Final[int] = 19495
REMOTE_CAPABILITIES: Final[frozenset[str]] = frozenset({"append", "read"})
ALLOWED_OPERATIONS: Final[frozenset[str]] = frozenset({"put", "increment"})
ALLOWED_TRANSPORT_ROLES: Final[frozenset[str]] = frozenset(
    {"reader", "writer", "readwrite"}
)
SQL_OPERATIONS: Final[frozenset[str]] = frozenset(
    {"sql", "execute_sql", "remote_update_sql", "update", "query"}
)

_IDENTITY_KEYS: Final[frozenset[str]] = frozenset(
    {"content_id", "cid", "identity", "canonical_id"}
)


class ExternalQuackOwnerError(ValueError):
    """Fenced owner or transport operation failed closed."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class StaleOwnerError(ExternalQuackOwnerError):
    """Caller epoch or owner id is not the current fenced owner."""


class DuplicateOwnerError(ExternalQuackOwnerError):
    """A second owner attempted to claim the same epoch without failover."""


class RemoteSqlRefusedError(ExternalQuackOwnerError):
    """Remote UPDATE or arbitrary SQL was refused."""


class UnsignedEnvelopeError(ExternalQuackOwnerError):
    """Envelope content identity was missing or did not match the payload."""


class TransportAuthError(ExternalQuackOwnerError):
    """Transport attach, append, or read lacked an admitted session."""


def _text(value: object, name: str, *, required: bool = True) -> str:
    text = "" if value is None else str(value).strip()
    if required and not text:
        raise ExternalQuackOwnerError(f"{name} is required", reason_code="malformed")
    if "\x00" in text:
        raise ExternalQuackOwnerError(
            f"{name} must not contain NUL", reason_code="malformed"
        )
    return text


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise ExternalQuackOwnerError(f"{name} must be an object", reason_code="malformed")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ExternalQuackOwnerError(
            f"{name} must be a positive integer", reason_code="malformed"
        )
    return int(value)


def _envelope_payload(envelope: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): envelope[key]
        for key in envelope
        if str(key) not in _IDENTITY_KEYS
    }


def issue_envelope(
    *,
    operation: str,
    key: str,
    value: Mapping[str, Any] | None = None,
    principal_id: str,
    idempotency_key: str = "",
) -> dict[str, Any]:
    """Issue a bound envelope whose content identity covers the payload."""

    op = _text(operation, "operation")
    payload = {
        "schema": ENVELOPE_SCHEMA,
        "interface": ENVELOPE_INTERFACE,
        "operation": op,
        "key": _text(key, "key"),
        "value": dict(_mapping(value or {}, "value")),
        "principal_id": _text(principal_id, "principal_id"),
        "idempotency_key": _text(idempotency_key, "idempotency_key", required=False),
    }
    envelope = dict(payload)
    envelope["content_id"] = content_identity(payload)
    return envelope


def verify_envelope(envelope: object) -> dict[str, Any]:
    """Fail closed on a missing, unsigned, or forged bound envelope."""

    if envelope is None:
        raise UnsignedEnvelopeError(
            "signed envelope is missing", reason_code="unsigned_envelope"
        )
    body = _mapping(envelope, "envelope")
    claimed = _text(
        body.get("content_id") or body.get("cid"),
        "envelope content identity",
        required=False,
    )
    if not claimed:
        raise UnsignedEnvelopeError(
            "signed envelope is missing", reason_code="unsigned_envelope"
        )
    expected = content_identity(_envelope_payload(body))
    if claimed != expected:
        raise UnsignedEnvelopeError(
            "forged envelope rejected", reason_code="forged_envelope"
        )
    operation = _text(body.get("operation"), "operation")
    if operation in SQL_OPERATIONS:
        raise RemoteSqlRefusedError(
            "arbitrary SQL is refused", reason_code="remote_sql_refused"
        )
    if operation not in ALLOWED_OPERATIONS:
        raise ExternalQuackOwnerError(
            f"unknown envelope operation: {operation}", reason_code="malformed"
        )
    _text(body.get("key"), "key")
    _text(body.get("principal_id"), "principal_id")
    value = body.get("value") or {}
    _mapping(value, "value")
    return dict(body)


@dataclass(frozen=True)
class OwnerLease:
    """Current single-owner epoch and fence."""

    owner_id: str
    epoch: int
    fence: int
    shard_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", _text(self.owner_id, "owner_id"))
        object.__setattr__(self, "epoch", _positive_int(self.epoch, "epoch"))
        object.__setattr__(self, "fence", _positive_int(self.fence, "fence"))
        object.__setattr__(self, "shard_id", _text(self.shard_id, "shard_id"))

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": OWNER_LEASE_SCHEMA,
                "interface": OWNER_LEASE_INTERFACE,
                "owner_id": self.owner_id,
                "epoch": self.epoch,
                "fence": self.fence,
                "shard_id": self.shard_id,
            }
        )


@dataclass(frozen=True)
class TransportSession:
    """Authenticated append/read session. Never a DuckDB handle."""

    client_id: str
    role: str
    token: str

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "client_id", _text(self.client_id, "client_id"))
            role = _text(self.role, "role")
            token = _text(self.token, "token")
        except ExternalQuackOwnerError as exc:
            raise TransportAuthError(str(exc), reason_code="transport_auth") from exc
        if role not in ALLOWED_TRANSPORT_ROLES:
            raise TransportAuthError(
                f"unknown transport role: {role}", reason_code="transport_auth"
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "token", token)


class BoundedQuackTransport:
    """In-memory authenticated multi-reader/multi-writer append/read transport.

    Remote clients receive append and read only. The operational table is never
    exposed for UPDATE or arbitrary SQL. No loopback port is bound.
    """

    INTERFACE: Final[str] = TRANSPORT_INTERFACE
    SCHEMA: Final[str] = TRANSPORT_SCHEMA

    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, TransportSession] = {}
        self._log: list[Mapping[str, Any]] = []
        self._cursor = 0

    @property
    def bound_port(self) -> int | None:
        return None

    @property
    def listen_uri(self) -> str:
        return ""

    @property
    def remote_capabilities(self) -> frozenset[str]:
        return REMOTE_CAPABILITIES

    @property
    def length(self) -> int:
        with self._lock:
            return len(self._log)

    def attach(
        self,
        client_id: str,
        *,
        role: str,
        token: str,
    ) -> TransportSession:
        session = TransportSession(client_id=client_id, role=role, token=token)
        with self._lock:
            existing = self._sessions.get(session.client_id)
            if existing is not None and existing.token != session.token:
                raise TransportAuthError(
                    "client already attached with a different token",
                    reason_code="transport_auth",
                )
            self._sessions[session.client_id] = session
        return session

    def _require(
        self,
        session: TransportSession,
        *,
        write: bool = False,
        read: bool = False,
    ) -> TransportSession:
        admitted = self._sessions.get(session.client_id)
        if (
            admitted is None
            or admitted.token != session.token
            or admitted.role != session.role
        ):
            raise TransportAuthError(
                "transport session is not authenticated",
                reason_code="transport_auth",
            )
        if write and admitted.role not in {"writer", "readwrite"}:
            raise TransportAuthError(
                "reader sessions cannot append", reason_code="transport_auth"
            )
        if read and admitted.role not in {"reader", "readwrite"}:
            raise TransportAuthError(
                "writer sessions cannot read", reason_code="transport_auth"
            )
        return admitted

    def append(self, session: TransportSession, envelope: Mapping[str, Any]) -> int:
        verified = verify_envelope(envelope)
        with self._lock:
            self._require(session, write=True)
            self._cursor += 1
            self._log.append(
                MappingProxyType({"ordinal": self._cursor, "envelope": dict(verified)})
            )
            return self._cursor

    def read(
        self,
        session: TransportSession,
        *,
        after: int = 0,
    ) -> tuple[Mapping[str, Any], ...]:
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ExternalQuackOwnerError(
                "after must be a non-negative integer", reason_code="malformed"
            )
        with self._lock:
            self._require(session, read=True)
            return tuple(row for row in self._log if int(row["ordinal"]) > after)

    def owner_drain(self) -> tuple[Mapping[str, Any], ...]:
        """Local owner snapshot of appended envelopes. Not a remote table."""

        with self._lock:
            return tuple(dict(row["envelope"]) for row in self._log)

    def remote_update_sql(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        del sql, args, kwargs
        raise RemoteSqlRefusedError(
            "operational tables are not exposed for remote UPDATE",
            reason_code="remote_sql_refused",
        )

    def execute_sql(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        del sql, args, kwargs
        raise RemoteSqlRefusedError(
            "arbitrary SQL is refused", reason_code="remote_sql_refused"
        )


class ExternalQuackOwner:
    """Sole local owner of one in-memory operational shard."""

    INTERFACE: Final[str] = EXTERNAL_QUACK_OWNER_INTERFACE
    SCHEMA: Final[str] = EXTERNAL_QUACK_OWNER_SCHEMA

    def __init__(
        self,
        owner_id: str,
        *,
        shard_id: str = "disposable-test-shard",
        transport: BoundedQuackTransport | None = None,
    ) -> None:
        self._lock = Lock()
        self._owner_id = _text(owner_id, "owner_id")
        self._shard_id = _text(shard_id, "shard_id")
        self._epoch = INITIAL_EPOCH
        self._fence = INITIAL_FENCE
        self._rows: dict[str, dict[str, Any]] = {}
        self._idempotency: dict[str, Mapping[str, Any]] = {}
        self._claimed = True
        self._transport = transport if transport is not None else BoundedQuackTransport()

    @property
    def owner_id(self) -> str:
        return self._owner_id

    @property
    def shard_id(self) -> str:
        return self._shard_id

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def fence(self) -> int:
        return self._fence

    @property
    def transport(self) -> BoundedQuackTransport:
        return self._transport

    @property
    def bound_port(self) -> int | None:
        return self._transport.bound_port

    @property
    def remote_capabilities(self) -> frozenset[str]:
        return REMOTE_CAPABILITIES

    @property
    def operational_table_exposed(self) -> bool:
        return False

    def lease(self) -> OwnerLease:
        with self._lock:
            return OwnerLease(
                owner_id=self._owner_id,
                epoch=self._epoch,
                fence=self._fence,
                shard_id=self._shard_id,
            )

    def _require_owner(self, owner_id: str, epoch: int) -> None:
        claimed_owner = _text(owner_id, "owner_id")
        claimed_epoch = _positive_int(epoch, "epoch")
        if claimed_epoch != self._epoch or claimed_owner != self._owner_id:
            raise StaleOwnerError(
                "stale owner rejected", reason_code="stale_owner"
            )

    def claim(self, owner_id: str, *, epoch: int) -> OwnerLease:
        """Idempotent claim of the current owner epoch. A second owner fails."""

        with self._lock:
            claimed_owner = _text(owner_id, "owner_id")
            claimed_epoch = _positive_int(epoch, "epoch")
            if claimed_epoch != self._epoch:
                raise StaleOwnerError(
                    "stale owner rejected", reason_code="stale_owner"
                )
            if claimed_owner != self._owner_id:
                raise DuplicateOwnerError(
                    "second owner refused without failover",
                    reason_code="duplicate_owner",
                )
            return OwnerLease(
                owner_id=self._owner_id,
                epoch=self._epoch,
                fence=self._fence,
                shard_id=self._shard_id,
            )

    def failover(self, new_owner_id: str | None = None) -> OwnerLease:
        """Advance epoch (and fence). The previous owner lease is stale."""

        with self._lock:
            replacement = (
                _text(new_owner_id, "new_owner_id")
                if new_owner_id is not None
                else self._owner_id
            )
            self._epoch += 1
            self._fence += 1
            self._owner_id = replacement
            return OwnerLease(
                owner_id=self._owner_id,
                epoch=self._epoch,
                fence=self._fence,
                shard_id=self._shard_id,
            )

    def get(self, key: str) -> Mapping[str, Any] | None:
        row_key = _text(key, "key")
        with self._lock:
            row = self._rows.get(row_key)
            return None if row is None else MappingProxyType(dict(row))

    def rows(self) -> Mapping[str, Mapping[str, Any]]:
        with self._lock:
            return MappingProxyType(
                {key: MappingProxyType(dict(value)) for key, value in self._rows.items()}
            )

    def _apply_unlocked(self, envelope: Mapping[str, Any]) -> Mapping[str, Any]:
        verified = verify_envelope(envelope)
        idempotency_key = _text(
            verified.get("idempotency_key"), "idempotency_key", required=False
        )
        if idempotency_key:
            existing = self._idempotency.get(idempotency_key)
            if existing is not None:
                return existing
        key = _text(verified.get("key"), "key")
        operation = _text(verified.get("operation"), "operation")
        value = dict(_mapping(verified.get("value") or {}, "value"))
        if operation == "put":
            self._rows[key] = value
        elif operation == "increment":
            current = self._rows.get(key, {})
            amount = value.get("n", 1)
            if isinstance(amount, bool) or not isinstance(amount, int):
                raise ExternalQuackOwnerError(
                    "increment n must be an integer", reason_code="malformed"
                )
            base = current.get("n", 0)
            if isinstance(base, bool) or not isinstance(base, int):
                base = 0
            self._rows[key] = {"n": int(base) + int(amount)}
        else:
            raise RemoteSqlRefusedError(
                "arbitrary SQL is refused", reason_code="remote_sql_refused"
            )
        receipt = MappingProxyType(
            {
                "schema": APPLY_RECEIPT_SCHEMA,
                "status": "applied",
                "owner_id": self._owner_id,
                "epoch": self._epoch,
                "fence": self._fence,
                "shard_id": self._shard_id,
                "operation": operation,
                "key": key,
                "row": dict(self._rows[key]),
                "idempotency_key": idempotency_key,
                "envelope_content_id": verified["content_id"],
            }
        )
        if idempotency_key:
            self._idempotency[idempotency_key] = receipt
        return receipt

    def apply(
        self,
        envelope: Mapping[str, Any],
        *,
        owner_id: str,
        epoch: int,
    ) -> Mapping[str, Any]:
        """Validate a bound envelope and serialize one private DuckDB transaction."""

        with self._lock:
            self._require_owner(owner_id, epoch)
            return self._apply_unlocked(envelope)

    def apply_from_transport(
        self,
        *,
        owner_id: str,
        epoch: int,
        envelopes: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Owner-only drain: apply appended envelopes as private transactions."""

        pending = (
            tuple(envelopes)
            if envelopes is not None
            else self._transport.owner_drain()
        )
        receipts: list[Mapping[str, Any]] = []
        with self._lock:
            self._require_owner(owner_id, epoch)
            seen: set[str] = set()
            for envelope in pending:
                content_id = str(_mapping(envelope, "envelope").get("content_id") or "")
                if content_id and content_id in seen:
                    continue
                if content_id:
                    seen.add(content_id)
                receipts.append(self._apply_unlocked(envelope))
        return tuple(receipts)

    def remote_update_sql(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        del sql, args, kwargs
        raise RemoteSqlRefusedError(
            "operational tables are not exposed for remote UPDATE",
            reason_code="remote_sql_refused",
        )

    def execute_sql(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        del sql, args, kwargs
        raise RemoteSqlRefusedError(
            "arbitrary SQL is refused", reason_code="remote_sql_refused"
        )


__all__ = (
    "ALLOWED_OPERATIONS",
    "APPLY_RECEIPT_SCHEMA",
    "BoundedQuackTransport",
    "CONTRACT_VERSION",
    "DuplicateOwnerError",
    "ENVELOPE_INTERFACE",
    "ENVELOPE_SCHEMA",
    "EXTERNAL_QUACK_OWNER_INTERFACE",
    "EXTERNAL_QUACK_OWNER_SCHEMA",
    "ExternalQuackOwner",
    "ExternalQuackOwnerError",
    "INITIAL_EPOCH",
    "LIVE_QUACK_PORT",
    "OWNER_LEASE_INTERFACE",
    "OWNER_LEASE_SCHEMA",
    "OwnerLease",
    "REMOTE_CAPABILITIES",
    "RemoteSqlRefusedError",
    "StaleOwnerError",
    "TRANSPORT_INTERFACE",
    "TRANSPORT_SCHEMA",
    "TransportAuthError",
    "TransportSession",
    "UnsignedEnvelopeError",
    "issue_envelope",
    "verify_envelope",
)
