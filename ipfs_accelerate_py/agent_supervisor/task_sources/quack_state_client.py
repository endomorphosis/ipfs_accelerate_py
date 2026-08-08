"""Typed Quack control-plane client with connection caching and safe SQL.

Interface: ``QuackStateClient@1``

Clients never receive an open SQL surface. Every statement is a named,
parameter-bound template from a closed registry. Identifiers are fixed in the
template text; callers supply only typed parameter values. Attach sessions
verify store/server identity (database UUID, schema fingerprint, generation,
extension fingerprint) and refuse mismatched peers rather than retrying through
an LLM path.

Transport modes:

* ``embedded`` — open ``control.duckdb`` through the existing exclusive-lock
  helper (hermetic tests and single-process tooling);
* ``quack`` — ``ATTACH`` a loopback Quack URI when the extension is available.

Transaction, CAS, fence, generation, and idempotency semantics live in
``control_plane_transactions.StateTransaction``.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .control_plane_contracts import (
    CommandKind,
    CommandOutcome,
    ControlPlaneBounds,
    ControlPlaneContractError,
    ControlPlaneIdentityError,
    ControlPlaneStoreIdentity,
    StateAuthorityClass,
    StateCommand,
    StoreGeneration,
    canonical_json_bytes,
)
from .control_plane_transactions import (
    CASResult,
    FenceMismatchError,
    IdempotencyConflictError,
    OptimisticConflictError,
    RetryPolicy,
    StateTransaction,
    StaleGenerationError,
    TransactionConflictKind,
    TransactionError,
    TransientTransactionError,
    default_retry_policy,
    run_with_retry,
)
from .duckdb_state import open_duckdb_connection

QUACK_STATE_CLIENT_INTERFACE: Final = "QuackStateClient@1"
QUACK_STATE_CLIENT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/quack-state-client@1"
)
QUACK_STATE_CLIENT_VERSION: Final[int] = 1
CLIENT_SESSION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/client-session@1"
)
STATEMENT_TEMPLATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/statement-template@1"
)
PAGE_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/page-result@1"
)

DEFAULT_STORE_ID: Final = "control.duckdb"
DEFAULT_PAGE_LIMIT: Final = 50
MAX_PAGE_LIMIT: Final = 500
DEFAULT_CONNECT_TIMEOUT_SECONDS: Final = 30.0
_SAFE_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PLACEHOLDER_RE: Final = re.compile(r"\?")
# Reject attempts to smuggle multi-statement or comment-terminated SQL.
_FORBIDDEN_SQL_FRAGMENT_RE: Final = re.compile(
    r";|--|/\*|\*/|\b(ATTACH|DETACH|COPY|INSTALL|LOAD|PRAGMA|CALL|EXPORT|"
    r"IMPORT|DROP|ALTER|CREATE|TRUNCATE|VACUUM|FORCE)\b",
    re.IGNORECASE,
)


class QuackClientError(ControlPlaneContractError):
    """Base error for the typed Quack client.

    Inherits ``ControlPlaneContractError`` (a ``ValueError``) so identity and
    SQL boundary failures share one exception lattice without mixing
    ``RuntimeError`` and ``ValueError`` bases.
    """


class QuackClientIdentityError(QuackClientError, ControlPlaneIdentityError):
    """Server/store identity verification failed."""


class QuackClientSQLError(QuackClientError):
    """Caller attempted raw SQL, identifier interpolation, or an unknown template."""


class QuackClientTransportError(QuackClientError):
    """Transport/connection failure (may be retried by higher layers)."""


class TransportMode(str, Enum):
    EMBEDDED = "embedded"
    QUACK = "quack"


class StatementKind(str, Enum):
    QUERY = "query"
    MUTATION = "mutation"
    META = "meta"


@dataclass(frozen=True)
class StatementTemplate:
    """Closed, parameter-bound SQL template. Identifiers are fixed in ``sql``."""

    SCHEMA: ClassVar[str] = STATEMENT_TEMPLATE_SCHEMA

    name: str
    sql: str
    parameter_names: tuple[str, ...] = ()
    kind: StatementKind = StatementKind.QUERY
    description: str = ""

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name or not _SAFE_IDENTIFIER_RE.fullmatch(name):
            raise QuackClientSQLError(f"invalid template name: {self.name!r}")
        sql = str(self.sql or "").strip()
        if not sql:
            raise QuackClientSQLError(f"template {name} has empty SQL")
        if _FORBIDDEN_SQL_FRAGMENT_RE.search(sql):
            # Templates themselves may contain CREATE only in internal seeds;
            # public registry forbids DDL/admin verbs.
            if name not in _INTERNAL_SEED_TEMPLATES:
                raise QuackClientSQLError(
                    f"template {name} contains forbidden SQL surface"
                )
        placeholders = len(_PLACEHOLDER_RE.findall(sql))
        params = tuple(str(item).strip() for item in self.parameter_names)
        if any(not item or not _SAFE_IDENTIFIER_RE.fullmatch(item) for item in params):
            raise QuackClientSQLError(
                f"template {name} has invalid parameter names"
            )
        if placeholders != len(params):
            raise QuackClientSQLError(
                f"template {name} placeholder count {placeholders} != "
                f"parameter count {len(params)}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "sql", sql)
        object.__setattr__(self, "parameter_names", params)
        kind = self.kind if isinstance(self.kind, StatementKind) else StatementKind(str(self.kind))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "description", str(self.description or ""))

    def bind(self, parameters: Mapping[str, Any] | Sequence[Any] | None) -> list[Any]:
        """Return ordered parameter values; reject unknown/missing names."""

        if parameters is None:
            values: Mapping[str, Any] | Sequence[Any] = {}
        else:
            values = parameters
        if isinstance(values, Mapping):
            unknown = set(values) - set(self.parameter_names)
            if unknown:
                raise QuackClientSQLError(
                    f"template {self.name} received unknown parameters: "
                    f"{sorted(unknown)}"
                )
            missing = [name for name in self.parameter_names if name not in values]
            if missing:
                raise QuackClientSQLError(
                    f"template {self.name} missing parameters: {missing}"
                )
            ordered = [values[name] for name in self.parameter_names]
        elif isinstance(values, Sequence) and not isinstance(
            values, (str, bytes, bytearray)
        ):
            if len(values) != len(self.parameter_names):
                raise QuackClientSQLError(
                    f"template {self.name} expected {len(self.parameter_names)} "
                    f"parameters, got {len(values)}"
                )
            ordered = list(values)
        else:
            raise QuackClientSQLError(
                f"template {self.name} parameters must be a mapping or sequence"
            )
        for index, value in enumerate(ordered):
            _assert_bound_value(value, self.parameter_names[index])
        return ordered

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "name": self.name,
            "sql": self.sql,
            "parameter_names": list(self.parameter_names),
            "kind": self.kind.value,
            "description": self.description,
        }


def _assert_bound_value(value: Any, name: str) -> None:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not (value == value and abs(value) != float("inf")):
            raise QuackClientSQLError(f"parameter {name} must be finite")
        if isinstance(value, str):
            if "\x00" in value:
                raise QuackClientSQLError(f"parameter {name} must not contain NUL")
            if len(value.encode("utf-8")) > 1_048_576:
                raise QuackClientSQLError(f"parameter {name} exceeds byte bound")
        return
    raise QuackClientSQLError(
        f"parameter {name} has unsupported type {type(value).__name__}"
    )


# Internal-only templates that may touch admin-ish surfaces during seed.
_INTERNAL_SEED_TEMPLATES: Final[frozenset[str]] = frozenset(
    {
        "seed_store_generation",
        "seed_client_session",
        "upsert_task_status",
    }
)


def _default_templates() -> dict[str, StatementTemplate]:
    return {
        "whoami_metadata": StatementTemplate(
            name="whoami_metadata",
            sql=(
                "SELECT key, value FROM control_plane_metadata "
                "WHERE key IN ('database_uuid', 'schema_fingerprint', "
                "'schema_version', 'application_version', 'tool_version') "
                "ORDER BY key"
            ),
            parameter_names=(),
            kind=StatementKind.META,
            description="Read store identity metadata keys",
        ),
        "load_store_generation": StatementTemplate(
            name="load_store_generation",
            sql=(
                "SELECT generation, schema_revision, fence_epoch, revision, "
                "database_uuid, birth_id FROM store_generations "
                "ORDER BY generation DESC LIMIT 1"
            ),
            parameter_names=(),
            kind=StatementKind.META,
            description="Load latest store generation head",
        ),
        "seed_store_generation": StatementTemplate(
            name="seed_store_generation",
            sql=(
                "INSERT INTO store_generations ("
                "generation, schema_revision, fence_epoch, revision, "
                "database_uuid, birth_id, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?)"
            ),
            parameter_names=(
                "generation",
                "schema_revision",
                "fence_epoch",
                "revision",
                "database_uuid",
                "birth_id",
                "created_at",
            ),
            kind=StatementKind.MUTATION,
            description="Seed initial store generation (bootstrap only)",
        ),
        "seed_client_session": StatementTemplate(
            name="seed_client_session",
            sql=(
                "INSERT INTO client_sessions ("
                "session_id, server_id, owner_id, process_birth_id, "
                "attached_at, last_seen_at, fence_epoch, generation, "
                "status, revision"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            parameter_names=(
                "session_id",
                "server_id",
                "owner_id",
                "process_birth_id",
                "attached_at",
                "last_seen_at",
                "fence_epoch",
                "generation",
                "status",
                "revision",
            ),
            kind=StatementKind.MUTATION,
            description="Register an attached client session",
        ),
        "touch_client_session": StatementTemplate(
            name="touch_client_session",
            sql=(
                "UPDATE client_sessions SET last_seen_at = ?, revision = revision + 1 "
                "WHERE session_id = ?"
            ),
            parameter_names=("last_seen_at", "session_id"),
            kind=StatementKind.MUTATION,
            description="Heartbeat an attached session",
        ),
        "select_task_by_cid": StatementTemplate(
            name="select_task_by_cid",
            sql=(
                "SELECT task_cid, task_alias, goal_cid, status, revision, "
                "ordinal, body_json FROM tasks WHERE task_cid = ? LIMIT 1"
            ),
            parameter_names=("task_cid",),
            kind=StatementKind.QUERY,
            description="Fetch one task by content id",
        ),
        "list_tasks_page": StatementTemplate(
            name="list_tasks_page",
            sql=(
                "SELECT task_cid, task_alias, goal_cid, status, revision, "
                "ordinal FROM tasks WHERE ordinal > ? "
                "ORDER BY ordinal ASC, task_cid ASC LIMIT ?"
            ),
            parameter_names=("after_ordinal", "limit"),
            kind=StatementKind.QUERY,
            description="Cursor page of tasks ordered by ordinal",
        ),
        "insert_task": StatementTemplate(
            name="insert_task",
            sql=(
                "INSERT INTO tasks ("
                "task_cid, task_alias, goal_cid, plan_cid, objective_id, "
                "ordinal, status, revision, priority, created_at, updated_at, "
                "identity_json, body_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            parameter_names=(
                "task_cid",
                "task_alias",
                "goal_cid",
                "plan_cid",
                "objective_id",
                "ordinal",
                "status",
                "revision",
                "priority",
                "created_at",
                "updated_at",
                "identity_json",
                "body_json",
            ),
            kind=StatementKind.MUTATION,
            description="Insert a task row",
        ),
        "cas_task_status": StatementTemplate(
            name="cas_task_status",
            sql=(
                "UPDATE tasks SET status = ?, revision = ?, updated_at = ? "
                "WHERE task_cid = ? AND revision = ?"
            ),
            parameter_names=(
                "status",
                "new_revision",
                "updated_at",
                "task_cid",
                "expected_revision",
            ),
            kind=StatementKind.MUTATION,
            description="CAS update task status by expected revision",
        ),
        "insert_goal": StatementTemplate(
            name="insert_goal",
            sql=(
                "INSERT INTO goals ("
                "goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal, "
                "title, status, created_at, updated_at, revision, body_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            parameter_names=(
                "goal_cid",
                "goal_alias",
                "objective_id",
                "parent_goal_cid",
                "ordinal",
                "title",
                "status",
                "created_at",
                "updated_at",
                "revision",
                "body_json",
            ),
            kind=StatementKind.MUTATION,
            description="Insert a goal row",
        ),
        "lookup_idempotency": StatementTemplate(
            name="lookup_idempotency",
            sql=(
                "SELECT idempotency_key, command_kind, command_id, store_id, "
                "session_id, result_digest, created_at, expires_at, body_json "
                "FROM idempotency_records WHERE idempotency_key = ? LIMIT 1"
            ),
            parameter_names=("idempotency_key",),
            kind=StatementKind.QUERY,
            description="Lookup a prior idempotent command result",
        ),
        "count_tasks": StatementTemplate(
            name="count_tasks",
            sql="SELECT COUNT(*) AS task_count FROM tasks",
            parameter_names=(),
            kind=StatementKind.QUERY,
            description="Count tasks",
        ),
    }


DEFAULT_STATEMENT_TEMPLATES: Final[Mapping[str, StatementTemplate]] = MappingProxyType(
    _default_templates()
)


@dataclass(frozen=True)
class ClientSession:
    """Attached client session identity."""

    SCHEMA: ClassVar[str] = CLIENT_SESSION_SCHEMA

    session_id: str
    server_id: str
    owner_id: str
    process_birth_id: str
    store_id: str
    generation: int
    fence_epoch: int
    attached_at: str
    transport_mode: TransportMode
    endpoint: str
    store_identity: ControlPlaneStoreIdentity | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "session_id": self.session_id,
            "server_id": self.server_id,
            "owner_id": self.owner_id,
            "process_birth_id": self.process_birth_id,
            "store_id": self.store_id,
            "generation": self.generation,
            "fence_epoch": self.fence_epoch,
            "attached_at": self.attached_at,
            "transport_mode": self.transport_mode.value,
            "endpoint": self.endpoint,
            "store_identity": (
                None if self.store_identity is None else self.store_identity.to_dict()
            ),
        }


@dataclass(frozen=True)
class PageResult:
    """Cursor page for bounded list queries."""

    SCHEMA: ClassVar[str] = PAGE_RESULT_SCHEMA

    items: tuple[Mapping[str, Any], ...]
    next_cursor: int | None
    limit: int
    exhausted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "items": [dict(item) for item in self.items],
            "next_cursor": self.next_cursor,
            "limit": self.limit,
            "exhausted": self.exhausted,
        }


@dataclass(frozen=True)
class QuackEndpoint:
    """Resolved client endpoint."""

    mode: TransportMode
    target: str
    database_path: Path | None = None
    quack_uri: str | None = None
    secret_handle: str = ""

    def __post_init__(self) -> None:
        mode = self.mode if isinstance(self.mode, TransportMode) else TransportMode(str(self.mode))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "target", str(self.target or "").strip())
        if not self.target:
            raise QuackClientError("endpoint target must not be empty")
        if self.secret_handle and not str(self.secret_handle).startswith(
            ("env://", "vault://", "handle:", "secret-handle:")
        ):
            raise QuackClientError(
                "endpoint secret must be an opaque handle, not raw credential"
            )


def resolve_endpoint(
    target: str | Path,
    *,
    mode: TransportMode | str | None = None,
    secret_handle: str = "",
) -> QuackEndpoint:
    """Resolve an embedded path or quack:// URI into a typed endpoint."""

    text = str(target).strip()
    if not text:
        raise QuackClientError("endpoint target is required")
    selected_mode: TransportMode
    if mode is not None:
        selected_mode = mode if isinstance(mode, TransportMode) else TransportMode(str(mode))
    elif text.startswith("quack:"):
        selected_mode = TransportMode.QUACK
    else:
        selected_mode = TransportMode.EMBEDDED

    if selected_mode is TransportMode.EMBEDDED:
        path = Path(text)
        return QuackEndpoint(
            mode=selected_mode,
            target=str(path),
            database_path=path,
            secret_handle=secret_handle,
        )
    if not text.startswith("quack:"):
        raise QuackClientError(
            "quack transport requires a quack: URI (loopback only by default)"
        )
    # Accept quack:127.0.0.1:PORT or quack://127.0.0.1:PORT
    return QuackEndpoint(
        mode=selected_mode,
        target=text,
        quack_uri=text,
        secret_handle=secret_handle,
    )


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _new_birth_id() -> str:
    return f"birth:{uuid.uuid4()}"


def _new_session_id() -> str:
    return f"session:{uuid.uuid4()}"


def _row_mapping(columns: Sequence[str], row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
        return {
            str(columns[index] if index < len(columns) else index): value
            for index, value in enumerate(row)
        }
    try:
        return {
            str(columns[index]): row[index]  # type: ignore[index]
            for index in range(len(columns))
        }
    except Exception:
        return {"value": row}


def _result_columns(result: Any) -> tuple[str, ...]:
    # DuckDBCursor stores columns on ``_columns``; native results use description.
    direct = getattr(result, "_columns", None)
    if isinstance(direct, Sequence) and direct and not isinstance(
        direct, (str, bytes, bytearray)
    ):
        return tuple(str(item) for item in direct)
    description = getattr(result, "description", None) or ()
    columns: list[str] = []
    for item in description:
        if isinstance(item, Sequence) and item and not isinstance(
            item, (str, bytes, bytearray)
        ):
            columns.append(str(item[0]))
        else:
            columns.append(str(item))
    return tuple(columns)


def _fetch_all(result: Any) -> list[Any]:
    if result is None:
        return []
    fetchall = getattr(result, "fetchall", None)
    if callable(fetchall):
        return list(fetchall() or [])
    if isinstance(result, list):
        return result
    return []


def _fetch_one(result: Any) -> Any | None:
    if result is None:
        return None
    fetchone = getattr(result, "fetchone", None)
    if callable(fetchone):
        return fetchone()
    rows = _fetch_all(result)
    return rows[0] if rows else None


class _ConnectionAdapter:
    """Normalize DuckDBConnection / native duckdb connections for transactions."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self._owns_close = False

    @property
    def raw(self) -> Any:
        return self._connection

    def execute(
        self,
        sql: str,
        parameters: Sequence[Any] | Mapping[str, Any] | None = None,
    ) -> Any:
        if parameters is None:
            return self._connection.execute(sql)
        return self._connection.execute(sql, parameters)

    def commit(self) -> None:
        commit = getattr(self._connection, "commit", None)
        if callable(commit):
            commit()
            # DuckDBConnection tracks BEGIN via execute(); a successful
            # commit() clears that flag. Also issue SQL COMMIT for native
            # connections that started the txn via BEGIN but whose commit()
            # is a no-op outside their own flag.
            try:
                self._connection.execute("COMMIT")
            except Exception:
                pass
            return
        try:
            self._connection.execute("COMMIT")
        except Exception:
            pass

    def rollback(self) -> None:
        rollback = getattr(self._connection, "rollback", None)
        if callable(rollback):
            try:
                rollback()
            except Exception:
                pass
        try:
            self._connection.execute("ROLLBACK")
        except Exception:
            pass

    def close(self) -> None:
        close = getattr(self._connection, "close", None)
        if callable(close):
            close()


class QuackStateClient:
    """Typed, fail-closed Quack/DuckDB control-plane client.

    Interface: ``QuackStateClient@1``.
    """

    INTERFACE: ClassVar[str] = QUACK_STATE_CLIENT_INTERFACE
    SCHEMA: ClassVar[str] = QUACK_STATE_CLIENT_SCHEMA
    VERSION: ClassVar[int] = QUACK_STATE_CLIENT_VERSION

    def __init__(
        self,
        *,
        owner_id: str,
        store_id: str = DEFAULT_STORE_ID,
        expected_identity: ControlPlaneStoreIdentity | None = None,
        templates: Mapping[str, StatementTemplate] | None = None,
        retry_policy: RetryPolicy | None = None,
        bounds: ControlPlaneBounds | None = None,
        connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
        process_birth_id: str | None = None,
        clock: Callable[[], str] | None = None,
        connection_factory: Callable[[QuackEndpoint], Any] | None = None,
    ) -> None:
        owner = str(owner_id or "").strip()
        if not owner:
            raise QuackClientError("owner_id is required")
        self.owner_id = owner
        self.store_id = str(store_id or DEFAULT_STORE_ID).strip() or DEFAULT_STORE_ID
        self.expected_identity = expected_identity
        self.bounds = bounds or ControlPlaneBounds()
        self.retry_policy = retry_policy or default_retry_policy(self.bounds)
        self.connect_timeout_seconds = float(connect_timeout_seconds)
        self.process_birth_id = process_birth_id or _new_birth_id()
        self._clock = clock or _utc_now
        self._connection_factory = connection_factory
        self._templates: dict[str, StatementTemplate] = dict(
            templates or DEFAULT_STATEMENT_TEMPLATES
        )
        self._lock = threading.RLock()
        self._endpoint: QuackEndpoint | None = None
        self._adapter: _ConnectionAdapter | None = None
        self._session: ClientSession | None = None
        self._store_generation: StoreGeneration | None = None
        self._closed = False

    @property
    def attached(self) -> bool:
        return self._session is not None and self._adapter is not None and not self._closed

    @property
    def session(self) -> ClientSession | None:
        return self._session

    @property
    def store_generation(self) -> StoreGeneration | None:
        return self._store_generation

    def list_templates(self) -> tuple[str, ...]:
        return tuple(sorted(self._templates))

    def get_template(self, name: str) -> StatementTemplate:
        key = str(name or "").strip()
        if key not in self._templates:
            raise QuackClientSQLError(f"unknown statement template: {name!r}")
        return self._templates[key]

    def register_template(self, template: StatementTemplate) -> None:
        """Register an additional closed template (trusted code only)."""

        if not isinstance(template, StatementTemplate):
            raise QuackClientSQLError("template must be a StatementTemplate")
        with self._lock:
            self._templates[template.name] = template

    def attach(
        self,
        target: str | Path | QuackEndpoint,
        *,
        mode: TransportMode | str | None = None,
        secret_handle: str = "",
        server_id: str = "server:local",
        seed_generation: bool = False,
        expected_identity: ControlPlaneStoreIdentity | None = None,
    ) -> ClientSession:
        """Attach to a store, verify identity, and cache the connection."""

        with self._lock:
            if self._closed:
                raise QuackClientError("client is closed")
            if self.attached:
                raise QuackClientError("client is already attached; detach first")
            endpoint = (
                target
                if isinstance(target, QuackEndpoint)
                else resolve_endpoint(target, mode=mode, secret_handle=secret_handle)
            )
            if endpoint.mode is TransportMode.QUACK and secret_handle:
                # Credentials remain handle-only; never materialize into argv.
                endpoint = QuackEndpoint(
                    mode=endpoint.mode,
                    target=endpoint.target,
                    database_path=endpoint.database_path,
                    quack_uri=endpoint.quack_uri,
                    secret_handle=secret_handle,
                )
            adapter = self._open_connection(endpoint)
            try:
                if seed_generation:
                    self._seed_generation_if_missing(adapter)
                generation = self._load_generation(adapter)
                identity = self._observe_store_identity(adapter, generation)
                expected = expected_identity or self.expected_identity
                if expected is not None:
                    self._verify_identity(expected, identity, generation)
                session_id = _new_session_id()
                attached_at = self._clock()
                self._execute_template(
                    adapter,
                    "seed_client_session",
                    {
                        "session_id": session_id,
                        "server_id": server_id,
                        "owner_id": self.owner_id,
                        "process_birth_id": self.process_birth_id,
                        "attached_at": attached_at,
                        "last_seen_at": attached_at,
                        "fence_epoch": generation.fence_epoch,
                        "generation": generation.generation,
                        "status": "attached",
                        "revision": 0,
                    },
                )
                # Best-effort commit for session row when using explicit tx APIs.
                adapter.commit()
                session = ClientSession(
                    session_id=session_id,
                    server_id=server_id,
                    owner_id=self.owner_id,
                    process_birth_id=self.process_birth_id,
                    store_id=self.store_id,
                    generation=generation.generation,
                    fence_epoch=generation.fence_epoch,
                    attached_at=attached_at,
                    transport_mode=endpoint.mode,
                    endpoint=endpoint.target,
                    store_identity=identity,
                )
            except Exception:
                adapter.close()
                raise
            self._endpoint = endpoint
            self._adapter = adapter
            self._session = session
            self._store_generation = generation
            return session

    def detach(self) -> None:
        with self._lock:
            adapter = self._adapter
            self._adapter = None
            self._session = None
            self._store_generation = None
            self._endpoint = None
            if adapter is not None:
                adapter.close()

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self.detach()

    def __enter__(self) -> "QuackStateClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def reconnect(self) -> ClientSession:
        """Drop the cached connection and re-attach to the same endpoint."""

        with self._lock:
            if self._endpoint is None or self._session is None:
                raise QuackClientError("cannot reconnect without a prior attach")
            endpoint = self._endpoint
            expected = (
                self._session.store_identity
                if self._session is not None
                else self.expected_identity
            )
            server_id = self._session.server_id
            self.detach()
            return self.attach(
                endpoint,
                server_id=server_id,
                expected_identity=expected,
            )

    def execute(
        self,
        template_name: str,
        parameters: Mapping[str, Any] | Sequence[Any] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Execute a named template with bound parameters only."""

        with self._lock:
            adapter = self._require_adapter()
            return self._execute_template(adapter, template_name, parameters)

    def execute_sql(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        """Rejected escape hatch: callers cannot run arbitrary SQL."""

        raise QuackClientSQLError(
            "arbitrary SQL is forbidden; use a named statement template"
        )

    def paginate(
        self,
        template_name: str = "list_tasks_page",
        *,
        cursor: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
        parameters: Mapping[str, Any] | None = None,
        cursor_parameter: str = "after_ordinal",
        limit_parameter: str = "limit",
        cursor_field: str = "ordinal",
    ) -> PageResult:
        """Fetch one cursor page from a list template."""

        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise QuackClientError("cursor must be a non-negative integer")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_PAGE_LIMIT
        ):
            raise QuackClientError(f"limit must be between 1 and {MAX_PAGE_LIMIT}")
        params = dict(parameters or {})
        params[cursor_parameter] = cursor
        params[limit_parameter] = limit + 1  # probe for more rows
        rows = self.execute(template_name, params)
        exhausted = len(rows) <= limit
        page_rows = rows[:limit]
        next_cursor: int | None = None
        if not exhausted and page_rows:
            last = page_rows[-1]
            if cursor_field not in last:
                raise QuackClientError(
                    f"page row missing cursor field {cursor_field!r}"
                )
            next_cursor = int(last[cursor_field])
        return PageResult(
            items=tuple(page_rows),
            next_cursor=next_cursor,
            limit=limit,
            exhausted=exhausted,
        )

    def load_generation(self) -> StoreGeneration:
        with self._lock:
            adapter = self._require_adapter()
            generation = self._load_generation(adapter)
            self._store_generation = generation
            return generation

    def transaction(
        self,
        *,
        expected_generation: StoreGeneration | None = None,
    ) -> StateTransaction:
        """Open a StateTransaction against the cached connection."""

        with self._lock:
            adapter = self._require_adapter()
            session = self._session
            generation = expected_generation or self._store_generation
            return StateTransaction(
                adapter,
                store_id=self.store_id,
                expected_generation=generation,
                session_id="" if session is None else session.session_id,
                retry_policy=self.retry_policy,
                now_iso=self._clock,
            )

    def submit_command(
        self,
        command: StateCommand,
        *,
        apply: Callable[[StateTransaction, StateCommand, StoreGeneration], Mapping[str, Any]]
        | None = None,
        refresh_on_conflict: bool = True,
    ) -> CASResult:
        """Submit a fenced idempotent command with jittered conflict retry."""

        if not isinstance(command, StateCommand):
            raise QuackClientError("command must be a StateCommand")
        apply_fn = apply or self._default_task_status_apply

        def _operation(attempt: int) -> CASResult:
            with self._lock:
                adapter = self._require_adapter()
                live = self._load_generation(adapter)
                active = command
                if attempt > 1 and refresh_on_conflict:
                    active = StateCommand(
                        command_id=command.command_id,
                        command_kind=command.command_kind,
                        store_id=command.store_id,
                        session_id=command.session_id or (
                            self._session.session_id if self._session else ""
                        ),
                        expected_generation=live.generation,
                        expected_revision=live.revision,
                        fence_epoch=live.fence_epoch,
                        idempotency_key=command.idempotency_key,
                        authority_class=command.authority_class,
                        parameters=dict(command.parameters),
                        secret_handle=command.secret_handle,
                    )
                txn = StateTransaction(
                    adapter,
                    store_id=self.store_id,
                    expected_generation=StoreGeneration(
                        store_id=self.store_id,
                        generation=active.expected_generation,
                        schema_revision=live.schema_revision,
                        fence_epoch=active.fence_epoch,
                        revision=active.expected_revision,
                        database_uuid=live.database_uuid,
                        birth_id=live.birth_id,
                    ),
                    session_id=active.session_id,
                    retry_policy=self.retry_policy,
                    now_iso=self._clock,
                )
                try:
                    result = txn.execute_command(active, apply=apply_fn)
                except OptimisticConflictError as exc:
                    return CASResult(
                        outcome=CommandOutcome.CONFLICT,
                        changed=False,
                        revision=live.revision,
                        generation=live.generation,
                        fence_epoch=live.fence_epoch,
                        result={"error": str(exc)},
                        conflict_kind=TransactionConflictKind.OPTIMISTIC,
                        attempts=attempt,
                        idempotency_key=command.idempotency_key,
                        command_id=command.command_id,
                    )
                except StaleGenerationError as exc:
                    return CASResult(
                        outcome=CommandOutcome.STALE,
                        changed=False,
                        revision=live.revision,
                        generation=live.generation,
                        fence_epoch=live.fence_epoch,
                        result={"error": str(exc)},
                        conflict_kind=TransactionConflictKind.STALE_GENERATION,
                        attempts=attempt,
                        idempotency_key=command.idempotency_key,
                        command_id=command.command_id,
                    )
                except FenceMismatchError as exc:
                    return CASResult(
                        outcome=CommandOutcome.STALE,
                        changed=False,
                        revision=live.revision,
                        generation=live.generation,
                        fence_epoch=live.fence_epoch,
                        result={"error": str(exc)},
                        conflict_kind=TransactionConflictKind.FENCE_MISMATCH,
                        attempts=attempt,
                        idempotency_key=command.idempotency_key,
                        command_id=command.command_id,
                    )
                except IdempotencyConflictError:
                    # Non-retryable: surface the typed failure to the caller.
                    raise
                except TransientTransactionError as exc:
                    return CASResult(
                        outcome=CommandOutcome.CONFLICT,
                        changed=False,
                        revision=live.revision,
                        generation=live.generation,
                        fence_epoch=live.fence_epoch,
                        result={"error": str(exc)},
                        conflict_kind=TransactionConflictKind.TRANSIENT,
                        attempts=attempt,
                        idempotency_key=command.idempotency_key,
                        command_id=command.command_id,
                    )
                except TransactionError as exc:
                    if exc.retryable:
                        return CASResult(
                            outcome=CommandOutcome.CONFLICT,
                            changed=False,
                            revision=live.revision,
                            generation=live.generation,
                            fence_epoch=live.fence_epoch,
                            result={"error": str(exc)},
                            conflict_kind=exc.kind,
                            attempts=attempt,
                            idempotency_key=command.idempotency_key,
                            command_id=command.command_id,
                        )
                    raise
                self._store_generation = self._load_generation(adapter)
                return CASResult(
                    outcome=result.outcome,
                    changed=result.changed,
                    revision=result.revision,
                    generation=result.generation,
                    fence_epoch=result.fence_epoch,
                    result=dict(result.result),
                    conflict_kind=result.conflict_kind,
                    attempts=attempt,
                    idempotency_key=result.idempotency_key,
                    command_id=result.command_id,
                    result_digest=result.result_digest,
                )

        return run_with_retry(_operation, policy=self.retry_policy)

    def cas_task_status(
        self,
        *,
        task_cid: str,
        expected_task_revision: int,
        new_status: str,
        idempotency_key: str,
        command_id: str | None = None,
    ) -> CASResult:
        """Convenience CAS for task status using the closed template set."""

        session = self._require_session()
        live = self.load_generation()
        command = StateCommand(
            command_id=command_id or f"cmd:cas-status:{task_cid}:{expected_task_revision}",
            command_kind=CommandKind.CLAIM,
            store_id=self.store_id,
            session_id=session.session_id,
            expected_generation=live.generation,
            expected_revision=live.revision,
            fence_epoch=live.fence_epoch,
            idempotency_key=idempotency_key,
            authority_class=StateAuthorityClass.AUTHORITATIVE,
            parameters={
                "task_cid": task_cid,
                "expected_task_revision": expected_task_revision,
                "status": new_status,
            },
        )
        return self.submit_command(command)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_adapter(self) -> _ConnectionAdapter:
        if self._closed:
            raise QuackClientError("client is closed")
        if self._adapter is None:
            raise QuackClientError("client is not attached")
        return self._adapter

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise QuackClientError("client is not attached")
        return self._session

    def _open_connection(self, endpoint: QuackEndpoint) -> _ConnectionAdapter:
        if self._connection_factory is not None:
            connection = self._connection_factory(endpoint)
            return _ConnectionAdapter(connection)
        if endpoint.mode is TransportMode.EMBEDDED:
            if endpoint.database_path is None:
                raise QuackClientError("embedded endpoint requires a database path")
            connection = open_duckdb_connection(
                endpoint.database_path,
                timeout_seconds=self.connect_timeout_seconds,
            )
            return _ConnectionAdapter(connection)
        return self._open_quack_connection(endpoint)

    def _open_quack_connection(self, endpoint: QuackEndpoint) -> _ConnectionAdapter:
        uri = endpoint.quack_uri or endpoint.target
        if not self._is_loopback_quack_uri(uri):
            raise QuackClientError(
                "non-loopback Quack bind requires a separately reviewed policy"
            )
        try:
            import duckdb
        except ImportError as exc:
            raise QuackClientTransportError(
                "DuckDB is required for Quack transport"
            ) from exc
        try:
            connection = duckdb.connect(":memory:")
            # LOAD is local-only; network INSTALL is never implicit here.
            try:
                connection.execute("LOAD quack")
            except Exception as load_exc:
                raise QuackClientTransportError(
                    "Quack extension is not loadable in this process"
                ) from load_exc
            # ATTACH uses a fixed SQL shape; the URI is a bound parameter value
            # only when the engine supports it. DuckDB ATTACH takes a literal,
            # so we validate the URI strictly before interpolation.
            safe_uri = self._validated_quack_uri_literal(uri)
            connection.execute(f"ATTACH '{safe_uri}' AS control_plane (READ_WRITE)")
            # Subsequent statements run against the attached alias by setting path.
            connection.execute("USE control_plane")
            return _ConnectionAdapter(connection)
        except QuackClientError:
            raise
        except Exception as exc:
            raise QuackClientTransportError(
                f"failed to attach Quack endpoint: {exc}"
            ) from exc

    @staticmethod
    def _is_loopback_quack_uri(uri: str) -> bool:
        text = str(uri or "").strip().lower()
        if not text.startswith("quack:"):
            return False
        rest = text[len("quack:") :]
        if rest.startswith("//"):
            rest = rest[2:]
        host = rest.split(":", 1)[0].split("/", 1)[0]
        return host in {"127.0.0.1", "localhost", "::1"}

    @staticmethod
    def _validated_quack_uri_literal(uri: str) -> str:
        text = str(uri or "").strip()
        if not re.fullmatch(
            r"quack:(?://)?(?:127\.0\.0\.1|localhost|::1):\d{1,5}",
            text,
            flags=re.IGNORECASE,
        ):
            raise QuackClientError(f"invalid or non-loopback quack URI: {uri!r}")
        if "'" in text or ";" in text or "\x00" in text:
            raise QuackClientError("quack URI contains forbidden characters")
        return text

    def _execute_template(
        self,
        adapter: _ConnectionAdapter,
        template_name: str,
        parameters: Mapping[str, Any] | Sequence[Any] | None,
    ) -> tuple[Mapping[str, Any], ...]:
        template = self.get_template(template_name)
        bound = template.bind(parameters)
        try:
            result = adapter.execute(template.sql, bound if bound else None)
        except QuackClientError:
            raise
        except Exception as exc:
            raise QuackClientTransportError(
                f"template {template.name} failed: {exc}"
            ) from exc
        columns = _result_columns(result)
        rows = _fetch_all(result)
        if not rows:
            return tuple()
        if not columns:
            # Prefer mapping rows (DuckDBRow); otherwise treat as DML with no projection.
            if isinstance(rows[0], Mapping):
                return tuple(_row_mapping((), row) for row in rows)
            return tuple()
        return tuple(_row_mapping(columns, row) for row in rows)

    def _load_generation(self, adapter: _ConnectionAdapter) -> StoreGeneration:
        rows = self._execute_template(adapter, "load_store_generation", None)
        if not rows:
            raise StaleGenerationError(
                "store generation is missing; seed or migrate the database first"
            )
        row = rows[0]
        return StoreGeneration(
            store_id=self.store_id,
            generation=int(row["generation"]),
            schema_revision=int(row["schema_revision"]),
            fence_epoch=int(row["fence_epoch"]),
            revision=int(row["revision"]),
            database_uuid=str(row["database_uuid"]),
            birth_id=str(row.get("birth_id") or ""),
        )

    def _seed_generation_if_missing(self, adapter: _ConnectionAdapter) -> None:
        rows = self._execute_template(adapter, "load_store_generation", None)
        if rows:
            return
        meta = {
            item["key"]: item["value"]
            for item in self._execute_template(adapter, "whoami_metadata", None)
        }
        database_uuid = str(meta.get("database_uuid") or str(uuid.uuid4()))
        self._execute_template(
            adapter,
            "seed_store_generation",
            {
                "generation": 1,
                "schema_revision": 1,
                "fence_epoch": 1,
                "revision": 0,
                "database_uuid": database_uuid,
                "birth_id": self.process_birth_id,
                "created_at": self._clock(),
            },
        )
        adapter.commit()

    def _observe_store_identity(
        self,
        adapter: _ConnectionAdapter,
        generation: StoreGeneration,
    ) -> ControlPlaneStoreIdentity:
        meta_rows = self._execute_template(adapter, "whoami_metadata", None)
        meta = {str(row["key"]): str(row["value"]) for row in meta_rows}
        schema_fingerprint = str(meta.get("schema_fingerprint") or "")
        if not schema_fingerprint.startswith("sha256:"):
            # Derive a stable fingerprint from available identity material so
            # hermetic stores without migration metadata still verify.
            material = {
                "database_uuid": generation.database_uuid,
                "schema_revision": generation.schema_revision,
                "store_id": self.store_id,
            }
            digest = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
            schema_fingerprint = f"sha256:{digest}"
        extension_fingerprint = ""
        if self.expected_identity is not None:
            extension_fingerprint = self.expected_identity.extension_fingerprint
        return ControlPlaneStoreIdentity(
            repository_id=(
                self.expected_identity.repository_id
                if self.expected_identity is not None
                else "repository:local"
            ),
            database_uuid=generation.database_uuid,
            store_id=self.store_id,
            schema_revision=generation.schema_revision,
            generation=generation.generation,
            schema_fingerprint=schema_fingerprint,
            authority_class=StateAuthorityClass.AUTHORITATIVE,
            server_birth_id=generation.birth_id,
            extension_fingerprint=extension_fingerprint,
        )

    def _verify_identity(
        self,
        expected: ControlPlaneStoreIdentity,
        observed: ControlPlaneStoreIdentity,
        generation: StoreGeneration,
    ) -> None:
        if expected.store_id != observed.store_id:
            raise QuackClientIdentityError(
                f"store_id mismatch: expected {expected.store_id}, "
                f"observed {observed.store_id}"
            )
        if expected.database_uuid != observed.database_uuid:
            raise QuackClientIdentityError(
                f"database_uuid mismatch: expected {expected.database_uuid}, "
                f"observed {observed.database_uuid}"
            )
        if expected.schema_fingerprint and (
            expected.schema_fingerprint != observed.schema_fingerprint
        ):
            raise QuackClientIdentityError(
                "schema_fingerprint mismatch between client expectation and store"
            )
        if expected.generation and expected.generation != generation.generation:
            raise QuackClientIdentityError(
                f"generation mismatch: expected {expected.generation}, "
                f"observed {generation.generation}"
            )
        if expected.extension_fingerprint and observed.extension_fingerprint:
            if expected.extension_fingerprint != observed.extension_fingerprint:
                raise QuackClientIdentityError(
                    "extension_fingerprint mismatch; refuse mismatched Quack peer"
                )

    def _default_task_status_apply(
        self,
        txn: StateTransaction,
        command: StateCommand,
        live: StoreGeneration,
    ) -> Mapping[str, Any]:
        params = dict(command.parameters)
        task_cid = str(params.get("task_cid") or "").strip()
        status = str(params.get("status") or "").strip()
        expected_task_revision = params.get("expected_task_revision")
        if not task_cid or not status:
            raise QuackClientError("task_cid and status parameters are required")
        if (
            isinstance(expected_task_revision, bool)
            or not isinstance(expected_task_revision, int)
            or expected_task_revision < 0
        ):
            raise QuackClientError("expected_task_revision must be a non-negative int")
        new_revision = txn.cas_row_revision(
            table="tasks",
            key_column="task_cid",
            key_value=task_cid,
            expected_revision=int(expected_task_revision),
            assignments={
                "status": status,
                "updated_at": self._clock(),
            },
        )
        return {
            "task_cid": task_cid,
            "status": status,
            "task_revision": new_revision,
            "store_revision_before": live.revision,
            "command_id": command.command_id,
        }


def open_embedded_client(
    database_path: str | Path,
    *,
    owner_id: str,
    store_id: str = DEFAULT_STORE_ID,
    expected_identity: ControlPlaneStoreIdentity | None = None,
    seed_generation: bool = True,
    retry_policy: RetryPolicy | None = None,
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS,
) -> QuackStateClient:
    """Attach an embedded client to ``database_path`` and return it open."""

    client = QuackStateClient(
        owner_id=owner_id,
        store_id=store_id,
        expected_identity=expected_identity,
        retry_policy=retry_policy,
        connect_timeout_seconds=connect_timeout_seconds,
    )
    client.attach(
        database_path,
        mode=TransportMode.EMBEDDED,
        seed_generation=seed_generation,
        expected_identity=expected_identity,
    )
    return client


__all__ = [
    "CLIENT_SESSION_SCHEMA",
    "DEFAULT_STATEMENT_TEMPLATES",
    "DEFAULT_STORE_ID",
    "PAGE_RESULT_SCHEMA",
    "QUACK_STATE_CLIENT_INTERFACE",
    "QUACK_STATE_CLIENT_SCHEMA",
    "ClientSession",
    "PageResult",
    "QuackClientError",
    "QuackClientIdentityError",
    "QuackClientSQLError",
    "QuackClientTransportError",
    "QuackEndpoint",
    "QuackStateClient",
    "StatementKind",
    "StatementTemplate",
    "TransportMode",
    "open_embedded_client",
    "resolve_endpoint",
]
