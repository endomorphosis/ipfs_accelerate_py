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
* ``quack`` — connect to the loopback owner's typed Unix-socket command
  gateway; generic SQL ``ATTACH`` is intentionally unavailable to clients.

Transaction, CAS, fence, generation, and idempotency semantics live in
``control_plane_transactions.StateTransaction``.
"""

# Python 3.8 compatibility requires ``str, Enum`` rather than ``StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

import base64
import hashlib
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

from ..federation.event_wait import (
    AdaptiveLongPollEventWaitClient,
    EventSource,
    EventWaitError,
)
from ..federation.events import EventBatch, EventWaitRequest
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
    StaleGenerationError,
    StateTransaction,
    TransactionConflictKind,
    TransactionError,
    TransientTransactionError,
    default_retry_policy,
    run_with_retry,
)
from .duckdb_state import open_duckdb_connection
from .typed_state_owner import (
    TypedStateOwnerError,
    open_typed_state_owner_connection,
)

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
    r"IMPORT|DROP|ALTER|CREATE|TRUNCATE|VACUUM|FORCE|"
    r"READ_CSV(?:_AUTO)?|READ_JSON(?:_AUTO)?|READ_PARQUET|PYTHON_EVAL)\b|"
    r"['\"]/|['\"]\.\./",
    re.IGNORECASE,
)


def _schema_fingerprint_digest(value: str) -> str:
    """Return the SHA-256 identity carried by a canonical schema CID.

    Migration metadata stores the canonical DAG-JSON CID, while the closed
    control-plane store identity contract admits digest strings.  The state
    owner performs the same lossless conversion before publishing identity;
    attached clients must compare that digest instead of an unrelated
    fallback derived from store coordinates.
    """

    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("sha256:"):
        digest = text.removeprefix("sha256:")
        if len(digest) == 64:
            try:
                bytes.fromhex(digest)
            except ValueError:
                pass
            else:
                return f"sha256:{digest.lower()}"
    if text.startswith("b"):
        try:
            encoded = text[1:].upper()
            encoded += "=" * ((8 - len(encoded) % 8) % 8)
            raw = base64.b32decode(encoded)
        except (ValueError, TypeError):
            raw = b""
        dag_json_sha256_prefix = b"\x01\xa9\x02\x12\x20"
        if raw.startswith(dag_json_sha256_prefix) and len(raw) == (
            len(dag_json_sha256_prefix) + 32
        ):
            return f"sha256:{raw[len(dag_json_sha256_prefix):].hex()}"
    return ""


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
        "list_ready_task_aliases": StatementTemplate(
            name="list_ready_task_aliases",
            sql=(
                "SELECT t.task_alias FROM tasks AS t "
                "WHERE lower(t.status) IN ('admitted','pending','proposed','queued',"
                "'ready','retrying','todo','unstarted') "
                "AND NOT EXISTS ("
                "SELECT 1 FROM task_dependencies AS d "
                "JOIN tasks AS prerequisite "
                "ON prerequisite.task_cid = d.dependency_task_cid "
                "WHERE d.task_cid = t.task_cid "
                "AND lower(prerequisite.status) NOT IN "
                "('complete','completed','done','skipped')"
                ") ORDER BY t.ordinal ASC, t.task_cid ASC LIMIT 500"
            ),
            parameter_names=(),
            kind=StatementKind.QUERY,
            description="Bounded configured-board ready frontier",
        ),
        "max_event_watermark": StatementTemplate(
            name="max_event_watermark",
            sql=(
                "SELECT COALESCE(MAX(global_sequence), 0) AS event_watermark "
                "FROM domain_events"
            ),
            parameter_names=(),
            kind=StatementKind.QUERY,
            description="Latest authoritative domain-event watermark",
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

    def execute_operation(
        self,
        operation: str,
        parameters: Sequence[Any] | None = None,
    ) -> Any:
        execute = getattr(self._connection, "execute_operation", None)
        if callable(execute):
            return execute(operation, parameters)
        # Native/embedded connections intentionally stay on the existing SQL
        # path. The caller supplies the trusted template object separately.
        raise AttributeError("connection has no typed owner operation surface")

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
        self._templates_sealed = False
        self._lock = threading.RLock()
        self._endpoint: QuackEndpoint | None = None
        self._adapter: _ConnectionAdapter | None = None
        self._session: ClientSession | None = None
        self._store_generation: StoreGeneration | None = None
        self._closed = False
        self._event_wait_source: EventSource | None = None
        self._event_wait_owner_boundary: Any | None = None
        self._event_wait_minimum_interval_seconds = 0.25
        self._event_wait_maximum_interval_seconds = 5.0
        self._event_wait_backoff_multiplier = 2.0

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
            if self._templates_sealed:
                raise QuackClientSQLError(
                    "statement template catalog is sealed for this client"
                )
            self._templates[template.name] = template

    def seal_templates(self) -> tuple[str, ...]:
        """Prevent runtime enlargement of the named statement catalog.

        Federation state-owner facades call this after installing their closed,
        trusted templates.  The method is monotonic and idempotent; it never
        removes an existing template or changes SQL text.
        """

        with self._lock:
            self._templates_sealed = True
            return tuple(sorted(self._templates))

    @property
    def templates_sealed(self) -> bool:
        with self._lock:
            return bool(self._templates_sealed)

    # ------------------------------------------------------------------
    # Typed event wait boundary
    # ------------------------------------------------------------------

    def bind_event_wait_source(
        self,
        source: EventSource,
        *,
        owner_boundary: Any | None = None,
        minimum_interval_seconds: float = 0.25,
        maximum_interval_seconds: float = 5.0,
        backoff_multiplier: float = 2.0,
    ) -> Mapping[str, object]:
        """Bind a closed event source and optional owner-local wait service.

        ``owner_boundary`` is normally :class:`QuackStateServer` in the state
        owner process and provides the real shared condition.  A remote Quack
        client cannot receive server push with the current extension; without
        that boundary this method enables only the bounded, backing-off,
        explicitly unqualified compatibility path.
        """

        if not callable(getattr(source, "events_for_subscription", None)) or not callable(
            getattr(source, "store_generation", None)
        ):
            raise QuackClientError(
                "event source must expose the closed subscription and generation interfaces"
            )
        if owner_boundary is not None:
            if not callable(getattr(owner_boundary, "wait_for_events", None)) or not callable(
                getattr(owner_boundary, "event_wait_capability", None)
            ):
                raise QuackClientError(
                    "owner event boundary does not expose the typed wait interface"
                )
        try:
            minimum = float(minimum_interval_seconds)
            maximum = float(maximum_interval_seconds)
            multiplier = float(backoff_multiplier)
            # Reuse the compatibility implementation's closed bound checks.
            AdaptiveLongPollEventWaitClient(
                lambda request: self._fetch_event_batch(source, request),
                minimum_interval_seconds=minimum,
                maximum_interval_seconds=maximum,
                backoff_multiplier=multiplier,
            )
        except (TypeError, ValueError, EventWaitError) as exc:
            raise QuackClientError("adaptive event wait bounds are invalid") from exc
        with self._lock:
            if not self.attached:
                raise QuackClientError(
                    "event wait source requires an attached state client"
                )
            if self._event_wait_source is not None and self._event_wait_source is not source:
                raise QuackClientError(
                    "event wait source is already bound for this client session"
                )
            if (
                self._event_wait_owner_boundary is not None
                and self._event_wait_owner_boundary is not owner_boundary
            ):
                raise QuackClientError(
                    "event wait owner boundary is already bound for this client session"
                )
            self._event_wait_source = source
            self._event_wait_owner_boundary = owner_boundary
            self._event_wait_minimum_interval_seconds = minimum
            self._event_wait_maximum_interval_seconds = maximum
            self._event_wait_backoff_multiplier = multiplier
        return MappingProxyType(self.event_wait_capability())

    def clear_event_wait_binding(self) -> None:
        """Release client-side references without changing owner state."""

        with self._lock:
            self._event_wait_source = None
            self._event_wait_owner_boundary = None

    def _fetch_event_batch(
        self,
        source: EventSource,
        request: EventWaitRequest,
    ) -> EventBatch:
        events = source.events_for_subscription(
            consumer_id=request.consumer_id,
            subscription_id=request.subscription_id,
            subscription_revision=request.subscription_revision,
            after_cursor=request.after_cursor,
            maximum_events=request.maximum_events,
        )
        return EventBatch(
            consumer_id=request.consumer_id,
            subscription_id=request.subscription_id,
            subscription_revision=request.subscription_revision,
            after_cursor=request.after_cursor,
            next_cursor=(events[-1].global_sequence if events else request.after_cursor),
            store_generation=source.store_generation(),
            events=events,
            timed_out=False,
            cancelled=False,
            server_shutdown=False,
        )

    @staticmethod
    def _validate_event_batch(
        request: EventWaitRequest,
        batch: EventBatch,
        *,
        expected_store_generation: int,
    ) -> EventBatch:
        if not isinstance(batch, EventBatch):
            raise QuackClientError("event wait boundary returned an untyped batch")
        if (
            batch.consumer_id != request.consumer_id
            or batch.subscription_id != request.subscription_id
            or batch.subscription_revision != request.subscription_revision
            or batch.after_cursor != request.after_cursor
            or len(batch.events) > request.maximum_events
            or batch.store_generation != expected_store_generation
        ):
            raise QuackClientIdentityError(
                "event wait batch differs from the bounded request identity"
            )
        return batch

    def wait_for_events(self, request: EventWaitRequest) -> EventBatch:
        """Wait through the owner condition or explicit adaptive fallback."""

        if not isinstance(request, EventWaitRequest):
            raise QuackClientError("event wait requires EventWaitRequest")
        with self._lock:
            session = self._require_session()
            adapter = self._require_adapter()
            source = self._event_wait_source
            owner_boundary = self._event_wait_owner_boundary
            minimum = self._event_wait_minimum_interval_seconds
            maximum = self._event_wait_maximum_interval_seconds
            multiplier = self._event_wait_backoff_multiplier
        remote_wait = getattr(adapter.raw, "wait_for_events", None)
        if (
            session.transport_mode is TransportMode.QUACK
            and callable(remote_wait)
            and bool(getattr(adapter.raw, "supports_event_wait", False))
        ):
            return self._validate_event_batch(
                request,
                remote_wait(request),
                expected_store_generation=session.generation,
            )
        if source is None:
            raise QuackClientError("typed event wait source is not bound")
        if owner_boundary is not None:
            return self._validate_event_batch(
                request,
                owner_boundary.wait_for_events(request),
                expected_store_generation=session.generation,
            )
        if session.transport_mode is not TransportMode.QUACK:
            raise QuackClientError(
                "embedded event waits require the server-owned condition boundary"
            )
        compatibility = AdaptiveLongPollEventWaitClient(
            lambda candidate: self._fetch_event_batch(source, candidate),
            minimum_interval_seconds=minimum,
            maximum_interval_seconds=maximum,
            backoff_multiplier=multiplier,
        )
        return self._validate_event_batch(
            request,
            compatibility.wait_for_events(request),
            expected_store_generation=session.generation,
        )

    def cancel_event_wait(self, consumer_id: str) -> None:
        """Cancel through the owner boundary; adaptive fallback has no push."""

        consumer = str(consumer_id or "").strip()
        if not consumer:
            raise QuackClientError("consumer_id is required")
        with self._lock:
            adapter = self._adapter
            owner_boundary = self._event_wait_owner_boundary
        cancel = getattr(owner_boundary, "cancel_event_wait", None)
        if not callable(cancel) and adapter is not None:
            remote_cancel = getattr(adapter.raw, "cancel_event_wait", None)
            if callable(remote_cancel) and bool(
                getattr(adapter.raw, "supports_event_wait", False)
            ):
                remote_cancel(consumer)
                return
        if not callable(cancel):
            raise QuackClientError(
                "remote adaptive event wait cancellation is unavailable"
            )
        cancel(consumer)

    def clear_event_wait_cancellation(self, consumer_id: str) -> None:
        """Clear an owner-side cancellation before a later wait."""

        consumer = str(consumer_id or "").strip()
        if not consumer:
            raise QuackClientError("consumer_id is required")
        with self._lock:
            adapter = self._adapter
            owner_boundary = self._event_wait_owner_boundary
        clear = getattr(owner_boundary, "clear_event_wait_cancellation", None)
        if not callable(clear) and adapter is not None:
            remote_clear = getattr(
                adapter.raw,
                "clear_event_wait_cancellation",
                None,
            )
            if callable(remote_clear) and bool(
                getattr(adapter.raw, "supports_event_wait", False)
            ):
                remote_clear(consumer)
                return
        if not callable(clear):
            raise QuackClientError(
                "remote adaptive event wait cancellation is unavailable"
            )
        clear(consumer)

    def event_wait_capability(self) -> dict[str, object]:
        """Describe the selected wait path without claiming promotion."""

        with self._lock:
            source = self._event_wait_source
            owner_boundary = self._event_wait_owner_boundary
            session = self._session
            adapter = self._adapter
        if (
            session is not None
            and session.transport_mode is TransportMode.QUACK
            and adapter is not None
            and callable(getattr(adapter.raw, "wait_for_events", None))
            and bool(getattr(adapter.raw, "supports_event_wait", False))
        ):
            return {
                "available": True,
                "interface": "TypedStateOwnerEventWait@1",
                "client_interface": "QuackStateClientEventWait@1",
                "transport": "typed_state_owner_bounded_long_wait",
                "server_owned": True,
                "blocking_condition": True,
                "adaptive_polling": False,
                "event_driven_qualified": True,
            }
        if source is None:
            return {
                "available": False,
                "interface": "QuackStateClientEventWait@1",
                "event_driven_qualified": False,
                "reason": "typed event source is not bound",
            }
        if owner_boundary is not None:
            capability = dict(owner_boundary.event_wait_capability())
            capability.update(
                {
                    "client_interface": "QuackStateClientEventWait@1",
                    "transport": "owner_local_condition",
                    "event_driven_qualified": False,
                }
            )
            return capability
        if session is not None and session.transport_mode is TransportMode.QUACK:
            capability = dict(AdaptiveLongPollEventWaitClient.capability())
            capability.update(
                {
                    "available": True,
                    "client_interface": "QuackStateClientEventWait@1",
                    "transport": "quack_adaptive_long_poll",
                    "event_driven_qualified": False,
                }
            )
            return capability
        return {
            "available": False,
            "interface": "QuackStateClientEventWait@1",
            "event_driven_qualified": False,
            "reason": "embedded mode requires an owner-local wait boundary",
        }

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
                if endpoint.mode is TransportMode.QUACK:
                    owner_identity = getattr(adapter.raw, "identity", None)
                    observed_server_id = (
                        str(owner_identity.get("server_id") or "")
                        if isinstance(owner_identity, Mapping)
                        else ""
                    )
                    if (
                        not observed_server_id
                        and self._connection_factory is not None
                    ):
                        # Hermetic tests may inject an in-memory DB-API object;
                        # it is never the default Quack authority path.
                        observed_server_id = str(server_id or "")
                    if not observed_server_id:
                        raise QuackClientIdentityError(
                            "typed state-owner handshake returned no server identity"
                        )
                    if server_id not in {"", "server:local", observed_server_id}:
                        raise QuackClientIdentityError(
                            "requested server identity differs from the typed owner"
                        )
                    server_id = observed_server_id
                if seed_generation:
                    self._seed_generation_if_missing(adapter)
                generation = self._load_generation(adapter)
                identity = self._observe_store_identity(adapter, generation)
                expected = expected_identity or self.expected_identity
                if expected is not None:
                    self._verify_identity(expected, identity, generation)
                owner_session_id = str(
                    getattr(adapter.raw, "session_id", "") or ""
                )
                session_id = owner_session_id or _new_session_id()
                attached_at = self._clock()
                if not owner_session_id:
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
                    # Embedded/test adapters retain their existing session
                    # registration path. Quack sessions are server-issued.
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
        self.clear_event_wait_binding()

    def __enter__(self) -> QuackStateClient:
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
                    prepare = getattr(adapter.raw, "prepare_command", None)
                    if callable(prepare):
                        prepare(active)
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
                "operation": "task.status.cas",
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
            # The Quack state owner exposes a closed server-side operation
            # catalog over its authenticated local command socket.  Clients do
            # not receive a generic READ_WRITE ATTACH and cannot submit SQL.
            connection = open_typed_state_owner_connection(
                store_id=self.store_id,
                client_id=self.owner_id,
                process_birth_id=self.process_birth_id,
                timeout_seconds=self.connect_timeout_seconds,
            )
            return _ConnectionAdapter(connection)
        except (OSError, TypedStateOwnerError) as exc:
            raise QuackClientTransportError(
                "failed to attach the typed Quack state-owner boundary "
                f"({type(exc).__name__})"
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
            execute_operation = getattr(adapter.raw, "execute_operation", None)
            if callable(execute_operation):
                result = execute_operation(template.name, bound)
            else:
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
            # Prefer mapping rows (DuckDBRow).  Tuple rows without a
            # projection used to be discarded as DML, which made live
            # lookups look empty.  Keep positional keys so callers such as
            # load_store_generation can still read the generation head.
            if isinstance(rows[0], Mapping):
                return tuple(_row_mapping((), row) for row in rows)
            return tuple(
                {
                    str(index): value
                    for index, value in enumerate(row)
                }
                if isinstance(row, Sequence)
                and not isinstance(row, (str, bytes, bytearray))
                else {"value": row}
                for row in rows
            )
        return tuple(_row_mapping(columns, row) for row in rows)

    def _load_generation(self, adapter: _ConnectionAdapter) -> StoreGeneration:
        rows = self._execute_template(adapter, "load_store_generation", None)
        if not rows:
            raise StaleGenerationError(
                "store generation is missing; seed or migrate the database first"
            )
        row = rows[0]
        names = (
            "generation",
            "schema_revision",
            "fence_epoch",
            "revision",
            "database_uuid",
            "birth_id",
        )
        values = []
        for index, name in enumerate(names):
            if name in row:
                values.append(row[name])
                continue
            if str(index) in row:
                values.append(row[str(index)])
                continue
            if isinstance(row, Sequence) and not isinstance(
                row, (str, bytes, bytearray)
            ) and index < len(row):
                values.append(row[index])
                continue
            raise StaleGenerationError(
                "store generation row is missing field " + name
            )
        return StoreGeneration(
            store_id=self.store_id,
            generation=int(values[0]),
            schema_revision=int(values[1]),
            fence_epoch=int(values[2]),
            revision=int(values[3]),
            database_uuid=str(values[4]),
            birth_id=str(values[5] or ""),
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
        try:
            schema_revision = int(meta.get("schema_version") or 1)
        except (TypeError, ValueError) as exc:
            raise QuackClientIdentityError(
                "control-plane schema_version metadata is not an integer"
            ) from exc
        if schema_revision < 1:
            raise QuackClientIdentityError(
                "control-plane schema_version metadata must be positive"
            )
        self._execute_template(
            adapter,
            "seed_store_generation",
            {
                "generation": 1,
                "schema_revision": schema_revision,
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
        schema_fingerprint = _schema_fingerprint_digest(
            str(meta.get("schema_fingerprint") or "")
        )
        if not schema_fingerprint:
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
