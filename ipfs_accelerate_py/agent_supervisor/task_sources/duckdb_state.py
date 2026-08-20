"""Shared DuckDB primitives for durable agent-supervisor state.

DuckDB permits only one external writer process. Supervisor stores therefore
use short-lived connections protected by a process-shared file lock. Legacy
SQLite databases are copied table-by-table into the new DuckDB file and are
left untouched as rollback evidence unless strict DuckDB-only mode is enabled.
"""

from __future__ import annotations

import fcntl
import hmac
import hashlib
import json
import os
import re
import sqlite3
import stat
import threading
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from .control_plane_contracts import (
    canonical_json_bytes,
    content_identity,
    is_secret_handle,
)

DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0
DEFAULT_MEMORY_LIMIT = "256MB"
DUCKDB_ONLY_ENV = "IPFS_ACCELERATE_DUCKDB_ONLY"
SQLITE_MAGIC = b"SQLite format 3\0"
# Loopback Quack URIs are the multi-writer control-plane transport. File
# connections remain one-writer; they must not be used as a silent fallback.
_QUACK_TRANSPORT_URI_RE = re.compile(
    r"^quack:(?://)?(?:127\.0\.0\.1|localhost|::1):\d{1,5}$",
    re.IGNORECASE,
)

# These settings are connection-birth policy, not mutable query preferences.
# ``lock_configuration`` is deliberately supplied in the same connect call and
# inserted last: no caller SQL can observe or restore DuckDB's permissive
# defaults between connection birth and policy verification.  The policy
# denies dynamic extension bytes and external filesystem/network access.
# Statically linked modules remain part of the separately reviewed native
# DuckDB payload and do not cross that byte boundary.
DUCKDB_CONNECTION_POLICY_SETTINGS = (
    ("autoinstall_known_extensions", "false", False),
    ("autoload_known_extensions", "false", False),
    ("enable_external_access", "false", False),
    ("allow_unsigned_extensions", "false", False),
    ("lock_configuration", "true", True),
)
DUCKDB_CONNECTION_POLICY_TUNING_KEYS = frozenset({"threads", "memory_limit"})
DUCKDB_CONNECTION_POLICY_MAX_THREADS = 256
DUCKDB_CONNECTION_POLICY_MIN_MEMORY_BYTES = 1_000_000
DUCKDB_CONNECTION_POLICY_MAX_MEMORY_BYTES = 256_000_000
_DUCKDB_MEMORY_LIMIT = re.compile(r"([1-9][0-9]{0,9})(B|KB|MB|GB)", re.ASCII)
_DUCKDB_MEMORY_MULTIPLIERS = {
    "B": 1,
    "KB": 1_000,
    "MB": 1_000_000,
    "GB": 1_000_000_000,
}

_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class DuckDBConnectionPolicyError(RuntimeError):
    """A DuckDB connection did not enforce the supervisor's sealed policy."""


def _connection_tuning(
    configuration: Mapping[str, Any] | None,
) -> dict[str, str]:
    if configuration is None:
        return {}
    if not isinstance(configuration, Mapping):
        raise TypeError("DuckDB connection configuration must be a mapping")
    tuning: dict[str, str] = {}
    protected = {
        name
        for name, _configured, _expected in DUCKDB_CONNECTION_POLICY_SETTINGS
    }
    for raw_name, raw_value in configuration.items():
        if not isinstance(raw_name, str):
            raise TypeError("DuckDB connection configuration keys must be strings")
        name = raw_name
        if name in protected:
            raise ValueError(
                f"DuckDB supervisor policy setting {name!r} cannot be overridden"
            )
        if name not in DUCKDB_CONNECTION_POLICY_TUNING_KEYS:
            raise ValueError(
                f"unsupported DuckDB supervisor connection setting: {raw_name!r}"
            )
        if name in tuning:
            raise ValueError(f"duplicate DuckDB connection setting: {name!r}")
        if name == "threads":
            if type(raw_value) is not int:
                raise TypeError("DuckDB threads must be an integer")
            if not 1 <= raw_value <= DUCKDB_CONNECTION_POLICY_MAX_THREADS:
                raise ValueError(
                    "DuckDB threads must be between 1 and "
                    f"{DUCKDB_CONNECTION_POLICY_MAX_THREADS}"
                )
            tuning[name] = str(raw_value)
            continue
        if type(raw_value) is not str:
            raise TypeError("DuckDB memory_limit must be a string")
        memory_limit = raw_value
        match = _DUCKDB_MEMORY_LIMIT.fullmatch(memory_limit)
        if match is None:
            raise ValueError(
                "DuckDB memory_limit must be an integer B, KB, MB, or GB value"
            )
        memory_bytes = int(match.group(1)) * _DUCKDB_MEMORY_MULTIPLIERS[
            match.group(2)
        ]
        if not (
            DUCKDB_CONNECTION_POLICY_MIN_MEMORY_BYTES
            <= memory_bytes
            <= DUCKDB_CONNECTION_POLICY_MAX_MEMORY_BYTES
        ):
            raise ValueError(
                "DuckDB memory_limit must be between "
                f"{DUCKDB_CONNECTION_POLICY_MIN_MEMORY_BYTES} and "
                f"{DUCKDB_CONNECTION_POLICY_MAX_MEMORY_BYTES} bytes"
            )
        tuning[name] = memory_limit
    return tuning


def _verify_duckdb_connection_policy(connection: Any) -> None:
    setting_names = tuple(
        name for name, _configured, _expected in DUCKDB_CONNECTION_POLICY_SETTINGS
    )
    expressions = ", ".join(
        f"current_setting('{name}')" for name in setting_names
    )
    try:
        row = connection.execute(f"SELECT {expressions}").fetchone()
    except Exception as exc:
        raise DuckDBConnectionPolicyError(
            "could not verify DuckDB supervisor connection policy"
        ) from exc
    expected = tuple(
        value for _name, _configured, value in DUCKDB_CONNECTION_POLICY_SETTINGS
    )
    if (
        not isinstance(row, tuple)
        or len(row) != len(expected)
        or any(type(value) is not bool for value in row)
        or row != expected
    ):
        raise DuckDBConnectionPolicyError(
            "DuckDB supervisor connection policy verification failed"
        )


def connect_duckdb_with_policy(
    duckdb_module: Any,
    database: Path | str,
    *,
    read_only: bool = False,
    configuration: Mapping[str, Any] | None = None,
) -> Any:
    """Open and verify one configuration-locked supervisor connection.

    The four dynamic-extension/external-access settings and configuration lock
    are passed to ``duckdb.connect`` atomically.  Only the bounded, canonical
    ``threads`` and ``memory_limit`` tuning keys may be supplied by internal
    callers; names and values are never normalized or coerced, and policy
    settings and all other DuckDB settings are not caller-overridable.
    """

    if type(read_only) is not bool:
        raise TypeError("DuckDB read_only must be a boolean")
    tuning = {
        "threads": "1",
        "memory_limit": DEFAULT_MEMORY_LIMIT,
    }
    tuning.update(_connection_tuning(configuration))
    connect_config: dict[str, str] = {
        name: configured
        for name, configured, _expected in DUCKDB_CONNECTION_POLICY_SETTINGS
        if name != "lock_configuration"
    }
    connect_config.update(tuning)
    # Keep the lock last in insertion order so DuckDB applies every selected
    # tuning and denial before sealing the connection configuration.
    connect_config["lock_configuration"] = "true"
    connection = duckdb_module.connect(
        str(database),
        read_only=read_only,
        config=connect_config,
    )
    try:
        _verify_duckdb_connection_policy(connection)
    except BaseException:
        connection.close()
        raise
    return connection


def duckdb_only_enabled() -> bool:
    """Return whether legacy SQLite discovery and migration are disabled."""

    return os.environ.get(DUCKDB_ONLY_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class DuckDBRow(Mapping[str, Any]):
    """Small ``sqlite3.Row``-compatible view over a DuckDB result row."""

    def __init__(self, columns: Iterable[str], values: Iterable[Any]) -> None:
        self._columns = tuple(str(column) for column in columns)
        self._values = tuple(values)
        self._positions = {column: index for index, column in enumerate(self._columns)}

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return self._values[self._positions[str(key)]]

    def __iter__(self) -> Iterator[str]:
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)


class DuckDBCursor:
    """Materialize a result before another statement reuses the connection."""

    def __init__(self, connection: Any, *, dml: bool = False) -> None:
        description = connection.description or ()
        self._columns = tuple(str(item[0]) for item in description)
        self._rows = list(connection.fetchall()) if description else []
        self._offset = 0
        self.rowcount = -1
        if (
            dml
            and len(self._columns) == 1
            and self._columns[0].lower() == "count"
            and len(self._rows) == 1
            and isinstance(self._rows[0][0], int)
        ):
            self.rowcount = int(self._rows[0][0])
            self._rows = []

    def fetchone(self) -> DuckDBRow | None:
        if self._offset >= len(self._rows):
            return None
        values = self._rows[self._offset]
        self._offset += 1
        return DuckDBRow(self._columns, values)

    def fetchall(self) -> list[DuckDBRow]:
        rows = [DuckDBRow(self._columns, values) for values in self._rows[self._offset :]]
        self._offset = len(self._rows)
        return rows

    def __iter__(self) -> Iterator[DuckDBRow]:
        return iter(self.fetchall())


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def exclusive_file_lock(
    lock_path: Path,
    *,
    timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> Iterator[None]:
    """Take a bounded thread/process lock for one DuckDB file."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    thread_lock = _thread_lock(lock_path)
    deadline = time.monotonic() + float(timeout_seconds)
    if not thread_lock.acquire(timeout=max(0.0, float(timeout_seconds))):
        raise TimeoutError(f"timed out acquiring DuckDB thread lock: {lock_path}")
    handle = lock_path.open("a+b")
    acquired = False
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out acquiring DuckDB process lock: {lock_path}")
                time.sleep(0.01)
        yield
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        thread_lock.release()


def is_sqlite_database(path: Path | str) -> bool:
    candidate = Path(path)
    if not candidate.is_file():
        return False
    try:
        with candidate.open("rb") as stream:
            return stream.read(len(SQLITE_MAGIC)) == SQLITE_MAGIC
    except OSError:
        return False


def resolve_duckdb_path(
    path: str | os.PathLike[str] | None,
    *,
    default_filename: str,
    temporary_prefix: str,
) -> tuple[Path, Path | None]:
    """Resolve a DuckDB target and its optional legacy SQLite sibling."""

    if not default_filename.endswith(".duckdb"):
        raise ValueError("default_filename must end in .duckdb")
    legacy_filename = f"{Path(default_filename).stem}.sqlite3"
    strict_duckdb_only = duckdb_only_enabled()
    if path is None:
        import tempfile

        root = Path(tempfile.mkdtemp(prefix=temporary_prefix))
        return root / default_filename, None

    supplied = Path(path)
    suffix = supplied.suffix.lower()
    if suffix in {".sqlite", ".sqlite3", ".db"}:
        target = supplied.with_suffix(".duckdb")
        legacy = None if strict_duckdb_only else supplied if is_sqlite_database(supplied) else None
        return target, legacy
    if suffix == ".duckdb":
        if strict_duckdb_only:
            return supplied, None
        legacy_candidate = supplied.with_suffix(".sqlite3")
        return supplied, (legacy_candidate if is_sqlite_database(legacy_candidate) else None)
    target = supplied / default_filename
    if strict_duckdb_only:
        return target, None
    legacy_candidate = supplied / legacy_filename
    return target, (legacy_candidate if is_sqlite_database(legacy_candidate) else None)


class DuckDBConnection:
    """Lock-owning compatibility adapter for existing SQLite-style code."""

    def __init__(
        self,
        path: Path | str,
        *,
        timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        threads: int = 1,
        transaction_on_context: bool = False,
    ) -> None:
        if is_quack_transport_target(path):
            raise DuckDBConnectionPolicyError(
                "quack transport URIs cannot be opened as DuckDB files; "
                "use open_quack_transport_connection"
            )
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if is_sqlite_database(self.path):
            raise ValueError(f"legacy SQLite database must be migrated before opening: {self.path}")
        self._transaction_active = False
        self._transaction_on_context = bool(transaction_on_context)
        self._context_depth = 0
        self._closed = False
        self._default_catalog = None
        self._quack_mutation_binding: dict[str, Any] | None = None
        self._quack_mutation_token = ""
        self._quack_pending_mutations: list[dict[str, Any]] = []
        self._lock_context = exclusive_file_lock(
            self.path.with_name(f".{self.path.name}.lock"),
            timeout_seconds=timeout_seconds,
        )
        self._lock_context.__enter__()
        try:
            import duckdb

            self._connection = connect_duckdb_with_policy(
                duckdb,
                self.path,
                configuration={
                    "threads": threads,
                    "memory_limit": memory_limit,
                },
            )
        except BaseException:
            self._lock_context.__exit__(None, None, None)
            raise

    @classmethod
    def wrap(
        cls,
        connection: Any,
        *,
        transaction_on_context: bool = False,
    ) -> DuckDBConnection:
        """Wrap an already configured connection without taking another lock."""

        instance = cls.__new__(cls)
        instance.path = None
        instance._connection = connection
        instance._transaction_active = False
        instance._transaction_on_context = bool(transaction_on_context)
        instance._context_depth = 0
        instance._closed = False
        instance._lock_context = None
        instance._default_catalog = None
        instance._quack_mutation_binding = None
        instance._quack_mutation_token = ""
        instance._quack_pending_mutations = []
        return instance

    @property
    def in_transaction(self) -> bool:
        return self._transaction_active

    def execute(
        self,
        sql: str,
        parameters: Iterable[Any] | Mapping[str, Any] | None = None,
    ) -> DuckDBCursor:
        statement = str(sql)
        normalized = " ".join(statement.strip().upper().split())
        if normalized == "BEGIN IMMEDIATE":
            statement = "BEGIN TRANSACTION"
            normalized = statement.upper()
        if normalized.startswith("PRAGMA BUSY_TIMEOUT"):
            return DuckDBCursor(self._connection)
        if normalized in {"PRAGMA FOREIGN_KEYS=ON", "PRAGMA JOURNAL_MODE=WAL"}:
            return DuckDBCursor(self._connection)
        catalog = getattr(self, "_default_catalog", None)
        if catalog and normalized.startswith("BEGIN"):
            if self._transaction_active or self._quack_pending_mutations:
                raise DuckDBConnectionPolicyError(
                    "quack owner mutation transaction is already active"
                )
        if catalog and normalized == "ROLLBACK":
            self._quack_pending_mutations = []
        if catalog and normalized == "COMMIT" and self._quack_pending_mutations:
            pending = list(self._quack_pending_mutations)
            self._quack_pending_mutations = []
            # This attached transaction contains reads only. End its snapshot
            # before the exclusive state owner applies the admitted bundle in
            # one local transaction.
            self._connection.execute("ROLLBACK")
            _consume_duckdb_result(self._connection)
            self._transaction_active = False
            return _execute_quack_owner_mutation_bundle(
                pending,
                binding=self._quack_mutation_binding,
                token=self._quack_mutation_token,
            )
        is_dml = normalized.startswith(
            ("INSERT ", "UPDATE ", "DELETE ", "MERGE ")
        )
        if catalog and is_dml:
            if not self._transaction_active:
                raise DuckDBConnectionPolicyError(
                    "quack owner mutation requires an explicit transaction"
                )
            template_id = _QUACK_OWNER_MUTATION_SQL_TO_TEMPLATE.get(normalized)
            if template_id is None:
                raise DuckDBConnectionPolicyError(
                    "quack owner mutation SQL is not in the closed template catalog"
                )
            if parameters is None or isinstance(parameters, Mapping):
                raise DuckDBConnectionPolicyError(
                    "quack owner mutation templates require positional parameters"
                )
            bound = list(parameters)
            _validate_quack_mutation_parameters(bound)
            if len(self._quack_pending_mutations) >= QUACK_OWNER_MUTATION_MAX_STEPS:
                raise DuckDBConnectionPolicyError(
                    "quack owner mutation bundle exceeds its step bound"
                )
            self._quack_pending_mutations.append(
                {"template_id": template_id, "parameters": bound}
            )
            return _empty_duckdb_cursor()
        if catalog and not normalized.startswith("USE "):
            self._connection.execute(f"USE {catalog}")
            _consume_duckdb_result(self._connection)
        if parameters is None:
            self._connection.execute(statement)
        else:
            self._connection.execute(statement, parameters)
        if normalized.startswith("BEGIN"):
            self._transaction_active = True
        elif normalized in {"COMMIT", "ROLLBACK"}:
            self._transaction_active = False
        dml = normalized.startswith(("INSERT ", "UPDATE ", "DELETE "))
        return DuckDBCursor(self._connection, dml=dml)

    def executemany(
        self,
        sql: str,
        parameters: Iterable[Iterable[Any]],
    ) -> DuckDBCursor:
        if getattr(self, "_default_catalog", None) and self._transaction_active:
            raise DuckDBConnectionPolicyError(
                "quack owner mutation bundles do not admit executemany"
            )
        self._connection.executemany(sql, parameters)
        return DuckDBCursor(self._connection, dml=True)

    def executescript(self, sql: str) -> DuckDBCursor:
        if getattr(self, "_default_catalog", None) and self._transaction_active:
            raise DuckDBConnectionPolicyError(
                "quack owner mutation bundles do not admit executescript"
            )
        self._connection.execute(sql)
        return DuckDBCursor(self._connection)

    def commit(self) -> None:
        if self._quack_pending_mutations and not self._transaction_active:
            raise DuckDBConnectionPolicyError(
                "quack owner mutation bundle exists outside an active transaction"
            )
        if self._transaction_active:
            # Keep the method API and SQL API on the same path.  For a Quack
            # attachment this ends the read-only snapshot and dispatches the
            # authenticated owner-side bundle; it must never silently commit
            # the attached snapshot while dropping buffered mutations.
            self.execute("COMMIT")

    def rollback(self) -> None:
        self._quack_pending_mutations = []
        if self._transaction_active:
            self._connection.rollback()
            self._transaction_active = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.rollback()
            self._connection.close()
        finally:
            if self._lock_context is not None:
                self._lock_context.__exit__(None, None, None)

    def __enter__(self) -> DuckDBConnection:
        if (
            self._transaction_on_context
            and self._context_depth == 0
            and not self._transaction_active
        ):
            self.execute("BEGIN TRANSACTION")
        self._context_depth += 1
        return self

    def __exit__(self, exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self._context_depth = max(0, self._context_depth - 1)
        if self._context_depth:
            return
        try:
            if self._transaction_active:
                if exc_type is None:
                    self.commit()
                else:
                    self.rollback()
        finally:
            if self._lock_context is not None:
                self.close()


def quack_transport_uri(target: object) -> str:
    """Return a loopback ``quack:`` URI, recovering Path.absolute() prefixes.

    ``Path('quack:127.0.0.1:41307').absolute()`` becomes
    ``<cwd>/quack:127.0.0.1:41307``. ``Path('quack://host:port')`` also
    collapses the double slash. Those are still transport URIs, never files.
    """

    text = str(target or "").strip()
    if not text:
        return ""
    if _QUACK_TRANSPORT_URI_RE.fullmatch(text):
        return text
    name = Path(text).name
    if _QUACK_TRANSPORT_URI_RE.fullmatch(name):
        return name
    collapsed = re.search(
        r"(?i)(?:^|/)(quack:/+(?:127\.0\.0\.1|localhost|::1):\d{1,5})$",
        text.replace("\\", "/"),
    )
    if collapsed is None:
        return ""
    recovered = re.sub(r"(?i)^quack:/+", "quack://", collapsed.group(1))
    if _QUACK_TRANSPORT_URI_RE.fullmatch(recovered):
        return recovered
    return ""


def is_quack_transport_target(target: object) -> bool:
    """Return whether ``target`` is a loopback ``quack:`` control-plane URI."""

    return bool(quack_transport_uri(target))


_QUACK_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{8,}$")
_QUACK_TOKEN_FILE_SUFFIX = ".quack-token"
_QUACK_STATUS_FILENAME = "quack-state-server.status.json"
_QUACK_CONTROL_CATALOG = "control_plane"
QUACK_OWNER_MUTATION_REQUEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-mutation-request@2"
)
QUACK_OWNER_MUTATION_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-mutation-result@2"
)
QUACK_OWNER_MUTATION_PROTOCOL_REVISION = 2
QUACK_OWNER_MUTATION_MAX_REQUEST_BYTES = 1_048_576
QUACK_OWNER_MUTATION_MAX_PARAMETER_BYTES = 262_144
QUACK_OWNER_MUTATION_MAX_STEPS = 5
QUACK_OWNER_MUTATION_REQUEST_TTL_MS = 30_000
QUACK_OWNER_MUTATION_SETTLEMENT_MS = 30_000
QUACK_OWNER_MUTATION_MAX_CLOCK_SKEW_MS = 5_000
QUACK_TRANSPORT_REFRESH_RETRY_SECONDS = 3.0

QUACK_MUTATION_TASK_STATUS_CAS = "task_status_cas@1"
QUACK_MUTATION_TASK_REVISION_INSERT = "task_revision_insert@1"
QUACK_MUTATION_COMPLETION_RECEIPT_INSERT = "completion_receipt_insert@1"
QUACK_MUTATION_DOMAIN_EVENT_INSERT = "domain_event_insert@1"
QUACK_MUTATION_VALIDATION_RUN_INSERT = "validation_run_insert@1"
QUACK_MUTATION_VALIDATION_RESULT_INSERT = "validation_result_insert@1"
QUACK_MUTATION_EVIDENCE_DELETE = "evidence_delete@1"
QUACK_MUTATION_EVIDENCE_INSERT = "evidence_insert@1"

QUACK_MUTATION_TASK_STATUS_TRANSITION = "task_status_transition@1"
QUACK_MUTATION_VALIDATION_RECORD = "validation_record@1"


def _normalize_quack_mutation_sql(sql: str) -> str:
    return " ".join(str(sql).strip().upper().split())


_QUACK_OWNER_MUTATION_SQL_TO_TEMPLATE = {
    _normalize_quack_mutation_sql(
        """
        UPDATE tasks SET status = ?, revision = ?, updated_at = ?, body_json = ?
        WHERE task_cid = ? AND revision = ?
        """
    ): QUACK_MUTATION_TASK_STATUS_CAS,
    _normalize_quack_mutation_sql(
        """
        INSERT INTO task_revisions (
            task_cid, revision, status, body_json, recorded_at
        ) VALUES (?, ?, ?, ?, ?)
        """
    ): QUACK_MUTATION_TASK_REVISION_INSERT,
    _normalize_quack_mutation_sql(
        """
        INSERT INTO completion_receipts (
            receipt_cid, task_cid, goal_cid, attempt_id, claim_cid,
            fencing_token, completed_at, validation_run_id,
            evidence_digest, body_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
    ): QUACK_MUTATION_COMPLETION_RECEIPT_INSERT,
    _normalize_quack_mutation_sql(
        """
        INSERT INTO domain_events (
            event_id, stream_id, sequence, global_sequence, event_type,
            task_cid, attempt_id, session_id, recorded_at, body_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
    ): QUACK_MUTATION_DOMAIN_EVENT_INSERT,
    _normalize_quack_mutation_sql(
        """
        INSERT INTO validation_runs (
            run_id, task_cid, attempt_id, started_at, finished_at,
            status, command_digest, body_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
    ): QUACK_MUTATION_VALIDATION_RUN_INSERT,
    _normalize_quack_mutation_sql(
        """
        INSERT INTO validation_results (
            result_id, run_id, task_cid, ordinal, outcome,
            evidence_digest, body_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
    ): QUACK_MUTATION_VALIDATION_RESULT_INSERT,
    _normalize_quack_mutation_sql(
        "DELETE FROM evidence_nodes WHERE evidence_id = ?"
    ): QUACK_MUTATION_EVIDENCE_DELETE,
    _normalize_quack_mutation_sql(
        """
        INSERT INTO evidence_nodes (
            evidence_id, parent_evidence_id, task_cid, evidence_kind,
            digest, created_at, body_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
    ): QUACK_MUTATION_EVIDENCE_INSERT,
}


class DuckDBQuackMutationConflictError(DuckDBConnectionPolicyError):
    """A closed owner-side compare-and-set observed a stale revision."""


class DuckDBQuackMutationUnknownOutcomeError(DuckDBConnectionPolicyError):
    """The writer committed but fresh transport settlement was not proved."""


def _quack_mutation_json(value: Mapping[str, Any]) -> bytes:
    try:
        return canonical_json_bytes(dict(value))
    except Exception as exc:
        raise DuckDBConnectionPolicyError(
            "quack owner mutation payload is not canonical JSON"
        ) from exc


def quack_owner_mutation_content_id(value: Mapping[str, Any]) -> str:
    try:
        return content_identity(dict(value))
    except Exception as exc:
        raise DuckDBConnectionPolicyError(
            "quack owner mutation payload has no canonical content identity"
        ) from exc


def quack_owner_mutation_mac(value: Mapping[str, Any], token: str) -> str:
    secret = str(token or "").strip()
    if not _QUACK_TOKEN_RE.fullmatch(secret):
        raise DuckDBConnectionPolicyError(
            "quack owner mutation requires an authenticated transport token"
        )
    return hmac.new(
        secret.encode("utf-8"),
        _quack_mutation_json(value),
        hashlib.sha256,
    ).hexdigest()


def quack_owner_mutation_dir(store_id: object = "") -> Path | None:
    """Resolve the inbox below the accepted repository root, or fail closed.

    Lanes may run in child worktrees, so cwd is never an authority for a
    relative store identity.  The lifecycle repository-root marker is the
    shared, accepted anchor used by every lane.
    """

    root_text = str(
        os.environ.get("IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT", "") or ""
    ).strip()
    if not root_text:
        return None
    root = Path(root_text).expanduser().resolve()
    store = str(
        store_id
        or os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "")
        or ""
    ).strip()
    if not store:
        return None
    configured = str(
        os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "") or ""
    ).strip()
    if configured and configured != store:
        return None
    path = Path(store).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    if path.suffix.lower() in {".duckdb", ".ddb"}:
        inbox = (path.parent / "quack-owner" / "mutations").resolve()
        try:
            inbox.relative_to(root)
        except ValueError:
            return None
        explicit = str(
            os.environ.get("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", "") or ""
        ).strip()
        if explicit and Path(explicit).expanduser().resolve() != inbox:
            return None
        return inbox
    return None


def quack_owner_mutation_write_lock_path(store_id: object = "") -> Path | None:
    inbox = quack_owner_mutation_dir(store_id)
    return None if inbox is None else inbox.parent / "write-transaction.lock"


def _empty_duckdb_cursor(*, rowcount: int = -1) -> DuckDBCursor:
    cursor = DuckDBCursor.__new__(DuckDBCursor)
    cursor._columns = ()
    cursor._rows = []
    cursor._offset = 0
    cursor.rowcount = int(rowcount)
    return cursor


def _validate_quack_mutation_parameters(parameters: Sequence[Any]) -> None:
    if len(parameters) > 16:
        raise DuckDBConnectionPolicyError(
            "quack owner mutation parameter count exceeds its bound"
        )
    for value in parameters:
        if value is None:
            continue
        if type(value) is int:
            if not -(2**63) <= value < 2**63:
                raise DuckDBConnectionPolicyError(
                    "quack owner mutation integer is outside int64"
                )
            continue
        if isinstance(value, str):
            if len(value.encode("utf-8")) > QUACK_OWNER_MUTATION_MAX_PARAMETER_BYTES:
                raise DuckDBConnectionPolicyError(
                    "quack owner mutation string exceeds its byte bound"
                )
            continue
        raise DuckDBConnectionPolicyError(
            "quack owner mutation parameters must be bounded JSON scalars"
        )


def _quack_mutation_operation(steps: Sequence[Mapping[str, Any]]) -> str:
    templates = tuple(str(item.get("template_id") or "") for item in steps)
    if templates in {
        (
            QUACK_MUTATION_TASK_STATUS_CAS,
            QUACK_MUTATION_TASK_REVISION_INSERT,
            QUACK_MUTATION_DOMAIN_EVENT_INSERT,
        ),
        (
            QUACK_MUTATION_TASK_STATUS_CAS,
            QUACK_MUTATION_TASK_REVISION_INSERT,
            QUACK_MUTATION_COMPLETION_RECEIPT_INSERT,
            QUACK_MUTATION_DOMAIN_EVENT_INSERT,
        ),
    }:
        return QUACK_MUTATION_TASK_STATUS_TRANSITION
    if templates in {
        (
            QUACK_MUTATION_VALIDATION_RUN_INSERT,
            QUACK_MUTATION_VALIDATION_RESULT_INSERT,
            QUACK_MUTATION_DOMAIN_EVENT_INSERT,
        ),
        (
            QUACK_MUTATION_VALIDATION_RUN_INSERT,
            QUACK_MUTATION_VALIDATION_RESULT_INSERT,
            QUACK_MUTATION_EVIDENCE_DELETE,
            QUACK_MUTATION_EVIDENCE_INSERT,
            QUACK_MUTATION_DOMAIN_EVENT_INSERT,
        ),
    }:
        return QUACK_MUTATION_VALIDATION_RECORD
    raise DuckDBConnectionPolicyError(
        "quack owner mutation bundle does not match an admitted operation"
    )


def _atomic_write_quack_request(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _quack_mutation_json(payload) + b"\n"
    if len(encoded) > QUACK_OWNER_MUTATION_MAX_REQUEST_BYTES:
        raise DuckDBConnectionPolicyError(
            "quack owner mutation request exceeds its byte bound"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _validate_quack_mutation_result(
    payload: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    token: str,
) -> int:
    expected_fields = {
        "schema",
        "protocol_revision",
        "request_id",
        "request_cid",
        "issued_at_ms",
        "expires_at_ms",
        "operation",
        "binding",
        "ok",
        "error_code",
        "rowcounts",
        "observed",
        "result_cid",
        "result_mac",
    }
    if set(payload) != expected_fields:
        raise DuckDBConnectionPolicyError(
            "quack owner mutation result has unknown or missing fields"
        )
    result_cid = payload.get("result_cid")
    result_mac = payload.get("result_mac")
    unsigned = dict(payload)
    unsigned.pop("result_mac", None)
    unsigned.pop("result_cid", None)
    authenticated = {**unsigned, "result_cid": result_cid}
    if (
        payload.get("schema") != QUACK_OWNER_MUTATION_RESULT_SCHEMA
        or payload.get("protocol_revision")
        != QUACK_OWNER_MUTATION_PROTOCOL_REVISION
        or payload.get("request_id") != request.get("request_id")
        or payload.get("request_cid") != request.get("request_cid")
        or payload.get("issued_at_ms") != request.get("issued_at_ms")
        or payload.get("expires_at_ms") != request.get("expires_at_ms")
        or payload.get("operation") != request.get("operation")
        or not isinstance(result_cid, str)
        or result_cid != quack_owner_mutation_content_id(unsigned)
        or not isinstance(result_mac, str)
        or not hmac.compare_digest(
            result_mac,
            quack_owner_mutation_mac(authenticated, token),
        )
    ):
        raise DuckDBConnectionPolicyError(
            "quack owner mutation returned an invalid result receipt"
        )
    binding = request.get("binding")
    if payload.get("binding") != binding:
        raise DuckDBConnectionPolicyError(
            "quack owner mutation result binding changed"
        )
    if payload.get("ok") is not True:
        error_code = str(payload.get("error_code") or "unknown")
        if error_code in {"cas_conflict", "event_head_conflict"}:
            raise DuckDBQuackMutationConflictError(
                "quack owner mutation compare-and-set conflicted"
            )
        if error_code in {
            "read_replica_refresh_unknown_outcome",
            "unknown_external_outcome",
        }:
            raise DuckDBQuackMutationUnknownOutcomeError(
                "quack owner mutation has an unknown external outcome; "
                "reconnect and reconcile the exact semantic request"
            )
        raise DuckDBConnectionPolicyError(
            f"quack owner mutation failed: {error_code}"
        )
    rowcounts = payload.get("rowcounts")
    if (
        not isinstance(rowcounts, list)
        or len(rowcounts) > QUACK_OWNER_MUTATION_MAX_STEPS
        or any(type(item) is not int for item in rowcounts)
    ):
        raise DuckDBConnectionPolicyError(
            "quack owner mutation result rowcounts are malformed"
        )
    observed = payload.get("observed")
    read_replica = (
        observed.get("read_replica") if isinstance(observed, Mapping) else None
    )
    replica_fields = {
        "schema",
        "authority",
        "path",
        "source_database_path",
        "server_id",
        "database_uuid",
        "generation",
        "schema_revision",
        "schema_fingerprint",
        "storage_schema_fingerprint",
        "sha256",
        "size_bytes",
        "refresh_sequence",
        "refreshed_at_ms",
        "live",
    }
    binding = payload.get("binding")
    store_id = str(binding.get("store_id") or "") if isinstance(binding, Mapping) else ""
    store_path = Path(store_id).expanduser() if store_id else Path()
    expected_replica_path = (
        store_path.with_name(
            f"{store_path.stem}.read-replica{store_path.suffix}"
        )
        if store_path.is_absolute() and store_path.name
        else None
    )
    if (
        not isinstance(observed, Mapping)
        or not isinstance(binding, Mapping)
        or not isinstance(read_replica, Mapping)
        or set(read_replica) != replica_fields
        or read_replica.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/read-replica-observation@1"
        or read_replica.get("authority")
        != "non_authoritative_read_replica"
        or read_replica.get("server_id") != binding.get("server_id")
        or read_replica.get("database_uuid") != binding.get("database_uuid")
        or read_replica.get("generation") != binding.get("generation")
        or read_replica.get("schema_revision") != binding.get("schema_revision")
        or read_replica.get("storage_schema_fingerprint")
        != binding.get("schema_fingerprint")
        or not isinstance(read_replica.get("schema_fingerprint"), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", read_replica["schema_fingerprint"])
        is None
        or not isinstance(read_replica.get("sha256"), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", read_replica["sha256"])
        is None
        or type(read_replica.get("size_bytes")) is not int
        or not 0 < read_replica["size_bytes"] <= 8 * 1024 * 1024 * 1024
        or type(read_replica.get("refresh_sequence")) is not int
        or read_replica["refresh_sequence"] < 1
        or type(read_replica.get("refreshed_at_ms")) is not int
        or read_replica["refreshed_at_ms"] < 1
        or read_replica.get("live") is not True
        or (
            expected_replica_path is not None
            and (
                read_replica.get("path") != str(expected_replica_path)
                or read_replica.get("source_database_path") != str(store_path)
            )
        )
    ):
        raise DuckDBConnectionPolicyError(
            "quack owner mutation result lacks exact fresh read-replica proof"
        )
    return int(rowcounts[0] if rowcounts else -1)


def _read_quack_result(path: Path) -> Mapping[str, Any] | None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > QUACK_OWNER_MUTATION_MAX_REQUEST_BYTES:
            raise DuckDBConnectionPolicyError(
                "quack owner mutation result is not a bounded regular file"
            )
        raw = os.read(descriptor, QUACK_OWNER_MUTATION_MAX_REQUEST_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > QUACK_OWNER_MUTATION_MAX_REQUEST_BYTES:
        raise DuckDBConnectionPolicyError(
            "quack owner mutation result exceeds its byte bound"
        )
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_json_pairs)
    except (json.JSONDecodeError, ValueError) as exc:
        raise DuckDBConnectionPolicyError(
            "quack owner mutation result is not canonical JSON"
        ) from exc
    if not isinstance(payload, Mapping) or _quack_mutation_json(payload).strip() != raw.strip():
        raise DuckDBConnectionPolicyError(
            "quack owner mutation result is not a canonical object"
        )
    return payload


def _resolve_quack_token_handle(*, uri: str) -> tuple[str, dict[str, Any]]:
    """Resolve a trusted supervisor handle from the owner-only state mount.

    The path is derived solely from the admitted repository root and exact
    state-store identity.  The public owner status must bind the handle, URI,
    database, and canonical owner directory before any token bytes are read.
    Provider subprocesses receive neither these state-program bindings nor the
    state mount.
    """

    handle = str(
        os.environ.get(
            "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE", ""
        )
        or ""
    ).strip()
    if not handle:
        return "", {}
    if not is_secret_handle(handle):
        raise DuckDBConnectionPolicyError(
            "quack endpoint secret handle is not an admitted opaque handle"
        )
    root_text = str(
        os.environ.get("IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT", "") or ""
    ).strip()
    store_id = str(
        os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "") or ""
    ).strip()
    if not root_text or not store_id:
        raise DuckDBConnectionPolicyError(
            "quack handle resolution requires repository-root and store identity"
        )
    root = Path(root_text).expanduser().resolve()
    store_path = Path(store_id).expanduser()
    if not store_path.is_absolute():
        store_path = root / store_path
    store_path = store_path.resolve()
    try:
        store_path.relative_to(root)
    except ValueError as exc:
        raise DuckDBConnectionPolicyError(
            "quack handle store identity escapes the admitted repository root"
        ) from exc
    owner_state = (store_path.parent / "quack-owner").resolve()
    try:
        owner_state.relative_to(root)
    except ValueError as exc:
        raise DuckDBConnectionPolicyError(
            "quack owner state escapes the admitted repository root"
        ) from exc
    status_path = owner_state / _QUACK_STATUS_FILENAME
    status = _read_quack_result(status_path)
    identity = status.get("identity") if isinstance(status, Mapping) else None
    read_replica = (
        status.get("read_replica") if isinstance(status, Mapping) else None
    )
    replica_path = store_path.with_name(
        f"{store_path.stem}.read-replica{store_path.suffix}"
    )
    if (
        not isinstance(status, Mapping)
        or status.get("lifecycle") != "ready"
        or str(status.get("database_path") or "") != str(store_path)
        or str(status.get("state_dir") or "") != str(owner_state)
        or status.get("store_id") != store_id
        or status.get("secret_handle") != handle
        or not isinstance(status.get("storage_schema_fingerprint"), str)
        or not status.get("storage_schema_fingerprint")
        or not isinstance(identity, Mapping)
        or identity.get("status") != "ready"
        or identity.get("store_id") != store_id
        or identity.get("listen_uri") != uri
        or identity.get("secret_handle") != handle
        or not isinstance(read_replica, Mapping)
        or read_replica.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/read-replica-observation@1"
        or read_replica.get("authority")
        != "non_authoritative_read_replica"
        or read_replica.get("path") != str(replica_path)
        or read_replica.get("source_database_path") != str(store_path)
        or read_replica.get("server_id") != identity.get("server_id")
        or read_replica.get("database_uuid") != identity.get("database_uuid")
        or read_replica.get("generation") != identity.get("generation")
        or read_replica.get("schema_revision")
        != identity.get("schema_revision")
        or read_replica.get("schema_fingerprint")
        != identity.get("schema_fingerprint")
        or read_replica.get("storage_schema_fingerprint")
        != status.get("storage_schema_fingerprint")
        or not isinstance(read_replica.get("sha256"), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", read_replica["sha256"])
        is None
        or type(read_replica.get("size_bytes")) is not int
        or not 0 < read_replica["size_bytes"] <= 8 * 1024 * 1024 * 1024
        or type(read_replica.get("refresh_sequence")) is not int
        or read_replica["refresh_sequence"] < 1
        or type(read_replica.get("refreshed_at_ms")) is not int
        or read_replica["refreshed_at_ms"] < 1
        or read_replica.get("live") is not True
    ):
        raise DuckDBConnectionPolicyError(
            "quack handle is not bound to the exact ready owner status"
        )
    admitted_identity = dict(identity)
    admitted_identity["schema_fingerprint"] = status["storage_schema_fingerprint"]
    admitted_identity["read_replica"] = dict(read_replica)
    token_path = owner_state / (
        handle.replace(":", "_").replace("/", "_") + _QUACK_TOKEN_FILE_SUFFIX
    )
    try:
        descriptor = os.open(
            token_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise DuckDBConnectionPolicyError(
            "quack endpoint token is unavailable for its admitted handle"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
            or not 8 <= info.st_size <= 512
        ):
            raise DuckDBConnectionPolicyError(
                "quack endpoint token file is not owner-only and bounded"
            )
        raw = os.read(descriptor, 513)
    finally:
        os.close(descriptor)
    try:
        token = raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise DuckDBConnectionPolicyError(
            "quack endpoint token is not an opaque ASCII secret"
        ) from exc
    if not _QUACK_TOKEN_RE.fullmatch(token):
        raise DuckDBConnectionPolicyError(
            "quack endpoint token is not an opaque url-safe secret"
        )
    return token, admitted_identity


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _execute_quack_owner_mutation_bundle(
    steps: Sequence[Mapping[str, Any]],
    *,
    binding: Mapping[str, Any] | None,
    token: str,
) -> DuckDBCursor:
    if not isinstance(binding, Mapping):
        raise DuckDBConnectionPolicyError(
            "quack owner mutation lacks an authenticated server binding"
        )
    operation = _quack_mutation_operation(steps)
    target = quack_owner_mutation_dir(binding.get("store_id"))
    if target is None:
        raise DuckDBConnectionPolicyError(
            "quack owner mutation store does not resolve to a bounded inbox"
        )
    semantic = {
        "schema": "ipfs_accelerate_py/agent-supervisor/quack-owner-mutation-semantic@1",
        "protocol_revision": QUACK_OWNER_MUTATION_PROTOCOL_REVISION,
        "operation": operation,
        "binding": dict(binding),
        "steps": [dict(item) for item in steps],
    }
    request_id = quack_owner_mutation_content_id(semantic)
    request_path = target / f"{request_id}.request.json"
    processing_path = target / f"{request_id}.processing.json"
    done_path = target / f"{request_id}.done.json"

    def build_request(issued_at_ms: int, expires_at_ms: int) -> dict[str, Any]:
        if (
            type(issued_at_ms) is not int
            or type(expires_at_ms) is not int
            or expires_at_ms - issued_at_ms != QUACK_OWNER_MUTATION_REQUEST_TTL_MS
        ):
            raise DuckDBConnectionPolicyError(
                "quack owner mutation request lifetime is malformed"
            )
        unsigned = {
            "schema": QUACK_OWNER_MUTATION_REQUEST_SCHEMA,
            "protocol_revision": QUACK_OWNER_MUTATION_PROTOCOL_REVISION,
            "request_id": request_id,
            "issued_at_ms": issued_at_ms,
            "expires_at_ms": expires_at_ms,
            "operation": operation,
            "binding": dict(binding),
            "steps": [dict(item) for item in steps],
        }
        request_cid = quack_owner_mutation_content_id(unsigned)
        authenticated = {**unsigned, "request_cid": request_cid}
        return {
            **authenticated,
            "auth_mac": quack_owner_mutation_mac(authenticated, token),
        }

    existing = _read_quack_result(done_path)
    if existing is not None:
        request = build_request(
            existing.get("issued_at_ms"),
            existing.get("expires_at_ms"),
        )
        rowcount = _validate_quack_mutation_result(existing, request=request, token=token)
        return _empty_duckdb_cursor(rowcount=rowcount)
    issued_at_ms = int(time.time() * 1000)
    request = build_request(
        issued_at_ms,
        issued_at_ms + QUACK_OWNER_MUTATION_REQUEST_TTL_MS,
    )
    _atomic_write_quack_request(request_path, request)
    deadline = time.monotonic() + (QUACK_OWNER_MUTATION_REQUEST_TTL_MS / 1000.0)
    while time.monotonic() < deadline:
        payload = _read_quack_result(done_path)
        if payload is None:
            time.sleep(0.05)
            continue
        rowcount = _validate_quack_mutation_result(
            payload,
            request=request,
            token=token,
        )
        request_path.unlink(missing_ok=True)
        return _empty_duckdb_cursor(rowcount=rowcount)
    cancelled = target / f"{request_id}.cancelled.json"
    try:
        os.replace(request_path, cancelled)
    except FileNotFoundError:
        # The owner claimed it.  Do not return while it can still commit: wait
        # for the authenticated terminal receipt (or report unknown outcome).
        settlement = time.monotonic() + (
            QUACK_OWNER_MUTATION_SETTLEMENT_MS / 1000.0
        )
        while time.monotonic() < settlement:
            payload = _read_quack_result(done_path)
            if payload is not None:
                rowcount = _validate_quack_mutation_result(
                    payload, request=request, token=token
                )
                return _empty_duckdb_cursor(rowcount=rowcount)
            if not processing_path.exists() and request_path.exists():
                # A recoverer requeued the exact semantic request.
                time.sleep(0.05)
                continue
            time.sleep(0.05)
        raise DuckDBConnectionPolicyError(
            "quack owner mutation has an unknown external outcome; exact "
            f"semantic replay is required ({request_id})"
        )
    else:
        cancelled.unlink(missing_ok=True)
    raise DuckDBConnectionPolicyError(
        "quack owner mutation timed out before owner claim; no effect occurred"
    )


def _consume_duckdb_result(connection: Any) -> None:
    try:
        connection.fetchall()
    except Exception:
        pass


def _open_quack_transport_connection_once(
    uri: str,
    *,
    token: str = "",
) -> DuckDBConnection:
    """Attach to the exclusive Quack state-owner (multi-reader/multi-writer).

    This is a transport connection, not a direct file open. The sealed
    one-writer file policy does not apply: Quack ATTACH requires a process
    that can reach the loopback state-owner.
    """

    text = quack_transport_uri(uri)
    if not text:
        raise DuckDBConnectionPolicyError(
            f"invalid or non-loopback quack URI: {uri!r}"
        )
    if "'" in text or ";" in text or "\x00" in text:
        raise DuckDBConnectionPolicyError("quack URI contains forbidden characters")
    try:
        import duckdb
    except ImportError as exc:
        raise DuckDBConnectionPolicyError(
            "DuckDB is required for Quack transport"
        ) from exc
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("LOAD quack")
        # Remote clients are read-only at the Quack SQL surface.  Every
        # intended mutation is a closed, authenticated owner-inbox bundle;
        # token possession alone must not retain arbitrary remote DML power.
        attach = f"ATTACH '{text}' AS {_QUACK_CONTROL_CATALOG} (READ_ONLY"
        secret = str(
            token or os.environ.get("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "") or ""
        ).strip()
        admitted_handle_binding: dict[str, Any] = {}
        if not secret:
            secret, admitted_handle_binding = _resolve_quack_token_handle(uri=text)
        if secret:
            if not _QUACK_TOKEN_RE.fullmatch(secret):
                raise DuckDBConnectionPolicyError(
                    "quack attach token must be an opaque url-safe secret"
                )
            attach += f", TOKEN '{secret}'"
        attach += ")"
        attached = connection.execute(attach)
        _consume_duckdb_result(attached)
        used = connection.execute(f"USE {_QUACK_CONTROL_CATALOG}")
        _consume_duckdb_result(used)
        # Prove the attached control catalog is visible on this connection.
        probed = connection.execute(
            f"SELECT count(*) FROM {_QUACK_CONTROL_CATALOG}.tasks"
        )
        _consume_duckdb_result(probed)
        identity_row = connection.execute(
            f"""
            SELECT server_id, store_id, database_uuid, schema_revision,
                   generation, process_birth_id, listen_uri,
                   extension_fingerprint
            FROM {_QUACK_CONTROL_CATALOG}.state_servers
            WHERE listen_uri = ? AND status = 'ready' AND stopped_at IS NULL
            ORDER BY generation DESC, started_at DESC
            LIMIT 1
            """,
            [text],
        ).fetchone()
        if identity_row is None or len(identity_row) != 8:
            raise DuckDBConnectionPolicyError(
                "quack transport did not publish a complete live server binding"
            )
        binding = {
            "server_id": str(identity_row[0]),
            "store_id": str(identity_row[1]),
            "database_uuid": str(identity_row[2]),
            "schema_revision": int(identity_row[3]),
            "generation": int(identity_row[4]),
            "process_birth_id": str(identity_row[5]),
            "listen_uri": str(identity_row[6]),
            "extension_fingerprint": str(identity_row[7]),
        }
        expected_store = str(
            os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "") or ""
        ).strip()
        expected_generation = str(
            os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_LIVE_GENERATION", "") or ""
        ).strip()
        expected_schema = str(
            os.environ.get("IPFS_ACCELERATE_AGENT_STATE_LIVE_SCHEMA_REVISION", "") or ""
        ).strip()
        if expected_store and binding["store_id"] != expected_store:
            raise DuckDBConnectionPolicyError(
                "quack transport store identity differs from the admitted environment"
            )
        if expected_generation and binding["generation"] != int(expected_generation):
            raise DuckDBConnectionPolicyError(
                "quack transport generation differs from the admitted environment"
            )
        if expected_schema and binding["schema_revision"] != int(expected_schema):
            raise DuckDBConnectionPolicyError(
                "quack transport schema revision differs from the admitted environment"
            )
        schema_row = connection.execute(
            f"""
            SELECT value
            FROM {_QUACK_CONTROL_CATALOG}.control_plane_metadata
            WHERE key = 'schema_fingerprint'
            """
        ).fetchone()
        binding["schema_fingerprint"] = str(schema_row[0] if schema_row else "")
        generation_row = connection.execute(
            f"""
            SELECT schema_revision, database_uuid, birth_id
            FROM {_QUACK_CONTROL_CATALOG}.store_generations
            WHERE generation = ?
            """,
            [binding["generation"]],
        ).fetchone()
        if (
            generation_row is None
            or int(generation_row[0]) != binding["schema_revision"]
            or str(generation_row[1]) != binding["database_uuid"]
            or str(generation_row[2]) != binding["process_birth_id"]
            or not binding["schema_fingerprint"]
        ):
            raise DuckDBConnectionPolicyError(
                "quack transport store generation does not match server binding"
            )
        if admitted_handle_binding:
            exact_handle_binding = {
                "server_id": binding["server_id"],
                "store_id": binding["store_id"],
                "database_uuid": binding["database_uuid"],
                "schema_revision": binding["schema_revision"],
                "schema_fingerprint": binding["schema_fingerprint"],
                "generation": binding["generation"],
                "process_birth_id": binding["process_birth_id"],
                "listen_uri": binding["listen_uri"],
                "extension_fingerprint": binding["extension_fingerprint"],
            }
            mismatched = [
                name
                for name, value in exact_handle_binding.items()
                if admitted_handle_binding.get(name) != value
            ]
            if mismatched:
                raise DuckDBConnectionPolicyError(
                    "quack live binding differs from the admitted owner status: "
                    + ", ".join(mismatched)
                )
    except Exception:
        try:
            connection.close()
        except Exception:
            pass
        raise
    wrapped = DuckDBConnection.wrap(connection)
    wrapped._default_catalog = _QUACK_CONTROL_CATALOG
    wrapped._quack_mutation_binding = binding
    wrapped._quack_mutation_token = secret
    return wrapped


def open_quack_transport_connection(
    uri: str,
    *,
    token: str = "",
) -> DuckDBConnection:
    """Open after a bounded retry across fail-closed replica refresh gaps."""

    deadline = time.monotonic() + QUACK_TRANSPORT_REFRESH_RETRY_SECONDS
    while True:
        try:
            return _open_quack_transport_connection_once(uri, token=token)
        except DuckDBConnectionPolicyError:
            raise
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise DuckDBConnectionPolicyError(
                    "quack transport remained unavailable after bounded "
                    "read-replica refresh retry"
                ) from exc
            time.sleep(0.02)


def open_duckdb_connection(
    path: Path | str,
    *,
    timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    threads: int = 1,
) -> DuckDBConnection:
    if is_quack_transport_target(path):
        return open_quack_transport_connection(path)
    return DuckDBConnection(
        path,
        timeout_seconds=timeout_seconds,
        memory_limit=memory_limit,
        threads=threads,
    )


def initialize_duckdb_database(
    path: Path | str,
    *,
    schema_sql: str,
    table_names: Sequence[str],
    legacy_sqlite_path: Path | str | None = None,
    timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    value_transform: (Callable[[str, str, Any], Any] | None) = None,
) -> None:
    """Initialize a store and idempotently import a legacy SQLite database."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    legacy = (
        Path(legacy_sqlite_path)
        if legacy_sqlite_path is not None and not duckdb_only_enabled()
        else None
    )
    if legacy is not None and not is_sqlite_database(legacy):
        legacy = None

    connection = open_duckdb_connection(
        target,
        timeout_seconds=timeout_seconds,
    )
    try:
        connection.execute("BEGIN TRANSACTION")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS agent_supervisor_store_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        connection.executescript(schema_sql)
        migration_key = f"sqlite_migration:{legacy.resolve()}" if legacy is not None else ""
        migrated = False
        if migration_key:
            migrated = (
                connection.execute(
                    "SELECT 1 FROM agent_supervisor_store_metadata WHERE key=?",
                    (migration_key,),
                ).fetchone()
                is not None
            )
        if legacy is not None and not migrated:
            source = sqlite3.connect(
                f"file:{legacy.resolve()}?mode=ro",
                uri=True,
                timeout=timeout_seconds,
            )
            source.row_factory = sqlite3.Row
            try:
                available = {
                    str(row[0])
                    for row in source.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                for table_name in table_names:
                    if table_name not in available:
                        continue
                    columns = [
                        str(row[1])
                        for row in source.execute(f'PRAGMA table_info("{table_name}")').fetchall()
                    ]
                    if not columns:
                        continue
                    quoted_columns = ", ".join(f'"{column}"' for column in columns)
                    placeholders = ", ".join("?" for _ in columns)
                    insert_sql = (
                        f'INSERT INTO "{table_name}" ({quoted_columns}) '
                        f"VALUES ({placeholders}) ON CONFLICT DO NOTHING"
                    )
                    cursor = source.execute(f'SELECT {quoted_columns} FROM "{table_name}"')
                    while True:
                        rows = cursor.fetchmany(256)
                        if not rows:
                            break
                        values = []
                        for row in rows:
                            values.append(
                                tuple(
                                    value_transform(
                                        table_name,
                                        column,
                                        row[column],
                                    )
                                    if value_transform is not None
                                    else row[column]
                                    for column in columns
                                )
                            )
                        connection.executemany(insert_sql, values)
            finally:
                source.close()
            connection.execute(
                """
                INSERT INTO agent_supervisor_store_metadata(key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (migration_key, str(int(time.time() * 1000))),
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
