"""Shared DuckDB primitives for durable agent-supervisor state.

DuckDB permits only one external writer process. Supervisor stores therefore
use short-lived connections protected by a process-shared file lock. Legacy
SQLite databases are copied table-by-table into the new DuckDB file and are
left untouched as rollback evidence unless strict DuckDB-only mode is enabled.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import random
import re
import sqlite3
import stat
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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
    protected = {name for name, _configured, _expected in DUCKDB_CONNECTION_POLICY_SETTINGS}
    for raw_name, raw_value in configuration.items():
        if not isinstance(raw_name, str):
            raise TypeError("DuckDB connection configuration keys must be strings")
        name = raw_name
        if name in protected:
            raise ValueError(f"DuckDB supervisor policy setting {name!r} cannot be overridden")
        if name not in DUCKDB_CONNECTION_POLICY_TUNING_KEYS:
            raise ValueError(f"unsupported DuckDB supervisor connection setting: {raw_name!r}")
        if name in tuning:
            raise ValueError(f"duplicate DuckDB connection setting: {name!r}")
        if name == "threads":
            if type(raw_value) is not int:
                raise TypeError("DuckDB threads must be an integer")
            if not 1 <= raw_value <= DUCKDB_CONNECTION_POLICY_MAX_THREADS:
                raise ValueError(
                    f"DuckDB threads must be between 1 and {DUCKDB_CONNECTION_POLICY_MAX_THREADS}"
                )
            tuning[name] = str(raw_value)
            continue
        if type(raw_value) is not str:
            raise TypeError("DuckDB memory_limit must be a string")
        memory_limit = raw_value
        match = _DUCKDB_MEMORY_LIMIT.fullmatch(memory_limit)
        if match is None:
            raise ValueError("DuckDB memory_limit must be an integer B, KB, MB, or GB value")
        memory_bytes = int(match.group(1)) * _DUCKDB_MEMORY_MULTIPLIERS[match.group(2)]
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
    expressions = ", ".join(f"current_setting('{name}')" for name in setting_names)
    try:
        row = connection.execute(f"SELECT {expressions}").fetchone()
    except Exception as exc:
        raise DuckDBConnectionPolicyError(
            "could not verify DuckDB supervisor connection policy"
        ) from exc
    expected = tuple(value for _name, _configured, value in DUCKDB_CONNECTION_POLICY_SETTINGS)
    if (
        not isinstance(row, tuple)
        or len(row) != len(expected)
        or any(type(value) is not bool for value in row)
        or row != expected
    ):
        raise DuckDBConnectionPolicyError("DuckDB supervisor connection policy verification failed")


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
                    raise TimeoutError(
                        f"timed out acquiring DuckDB process lock: {lock_path}"
                    ) from None
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
        if catalog and (
            "\x00" in statement
            or "--" in statement
            or "/*" in statement
            or ";" in normalized.rstrip(";")
        ):
            raise DuckDBConnectionPolicyError(
                "Quack transport accepts one comment-free read statement"
            )
        if catalog and not normalized.startswith(
            (
                "SELECT ",
                "SHOW ",
                "DESCRIBE ",
                "BEGIN",
                "COMMIT",
                "ROLLBACK",
            )
        ):
            raise DuckDBConnectionPolicyError(
                "Quack transport SQL is read-only; use a closed typed owner "
                "command for DatabaseTaskSource mutations"
            )
        if catalog and not normalized.startswith("USE "):
            statement = _qualify_quack_statement(statement, catalog)
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
        if getattr(self, "_default_catalog", None):
            raise DuckDBConnectionPolicyError(
                "Quack transport executemany is disabled; use a closed typed "
                "owner command for DatabaseTaskSource mutations"
            )
        self._connection.executemany(sql, parameters)
        return DuckDBCursor(self._connection, dml=True)

    def executescript(self, sql: str) -> DuckDBCursor:
        if getattr(self, "_default_catalog", None):
            raise DuckDBConnectionPolicyError(
                "Quack transport SQL scripts are disabled; use a closed typed "
                "owner command for DatabaseTaskSource mutations"
            )
        self._connection.execute(sql)
        return DuckDBCursor(self._connection)

    def commit(self) -> None:
        if self._transaction_active:
            self._connection.commit()
            self._transaction_active = False

    def rollback(self) -> None:
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
_QUACK_ATTACH_TOKEN_ENV = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
_QUACK_SECRET_HANDLE_ENV = "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE"
_QUACK_CONTROL_CATALOG = "control_plane"


def resolve_quack_attach_token(
    token: str = "",
    *,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Return the admitted attach token without logging it.

    Resolution order: explicit argument, ``IPFS_ACCELERATE_AGENT_QUACK_TOKEN``,
    then the environment variable named by an ``env://`` secret handle.
    """

    source = os.environ if environment is None else environment
    secret = str(token or source.get(_QUACK_ATTACH_TOKEN_ENV, "") or "").strip()
    if secret:
        return secret
    handle = str(source.get(_QUACK_SECRET_HANDLE_ENV, "") or "").strip()
    if handle.startswith("env://"):
        target = handle[len("env://") :].strip()
        if target:
            secret = str(source.get(target, "") or "").strip()
    return secret


def _quack_token_fingerprint(token: str) -> str:
    secret = str(token or "").strip()
    if not secret:
        return "none"
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:16]


QUACK_OWNER_COMMAND_REQUEST_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-command-request@1"
)
QUACK_OWNER_COMMAND_RESPONSE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/quack-owner-command-response@1"
)
QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS = "compare_and_set_status"
QUACK_OWNER_COMMAND_COMPARE_AND_SET_GOAL_STATUS = "compare_and_set_goal_status"
QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK = "rearm_blocked_task"
QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET = (
    "recover_typed_deferral_budget"
)
QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF = "record_queue_backoff"
QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF_AND_CAS_STATUS = (
    "record_queue_backoff_and_cas_status"
)
QUACK_OWNER_COMMAND_RECORD_QUEUE_RETRY = "record_queue_retry"
QUACK_OWNER_COMMAND_RECORD_EVIDENCE = "record_evidence"
QUACK_OWNER_COMMAND_RECORD_VALIDATION_RESULT = "record_validation_result"
QUACK_OWNER_COMMANDS = frozenset(
    {
        QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS,
        QUACK_OWNER_COMMAND_COMPARE_AND_SET_GOAL_STATUS,
        QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK,
        QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET,
        QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF,
        QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF_AND_CAS_STATUS,
        QUACK_OWNER_COMMAND_RECORD_QUEUE_RETRY,
        QUACK_OWNER_COMMAND_RECORD_EVIDENCE,
        QUACK_OWNER_COMMAND_RECORD_VALIDATION_RESULT,
    }
)
QUACK_OWNER_COMMAND_MAX_BYTES = 262_144
QUACK_OWNER_COMMAND_MAX_AGE_MS = 10_000
_QUACK_OWNER_REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_QUACK_OWNER_WRITER_RE = re.compile(r"^supervisor-process:[1-9][0-9]{0,19}$")

_QUACK_OWNER_COMMAND_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS: (
        frozenset({"task_cid_or_alias", "expected_revision", "status"}),
        frozenset(
            {
                "receipt",
                "expected_control_receipt",
                "evidence_digests",
            }
        ),
    ),
    QUACK_OWNER_COMMAND_COMPARE_AND_SET_GOAL_STATUS: (
        frozenset({"goal_cid_or_alias", "expected_revision", "status"}),
        frozenset({"receipt"}),
    ),
    QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK: (
        frozenset({"task_cid_or_alias"}),
        frozenset({"receipt"}),
    ),
    QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET: (
        frozenset({"task_cid_or_alias", "repair_head", "repair_tree"}),
        frozenset(),
    ),
    QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF: (
        frozenset({"task_cid", "delay_ms"}),
        frozenset({"reason", "selection_penalty"}),
    ),
    QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF_AND_CAS_STATUS: (
        frozenset(
            {
                "task_cid",
                "expected_revision",
                "expected_control_receipt",
                "status",
                "receipt",
                "delay_ms",
                "reason",
            }
        ),
        frozenset({"selection_penalty"}),
    ),
    QUACK_OWNER_COMMAND_RECORD_QUEUE_RETRY: (
        frozenset({"task_cid"}),
        frozenset(),
    ),
    QUACK_OWNER_COMMAND_RECORD_EVIDENCE: (
        frozenset({"task_cid", "evidence_kind", "digest"}),
        frozenset({"body"}),
    ),
    QUACK_OWNER_COMMAND_RECORD_VALIDATION_RESULT: (
        frozenset({"task_cid", "outcome", "evidence_digest"}),
        frozenset({"argv", "attempt_id", "body"}),
    ),
}


class QuackOwnerCommandRemoteError(DuckDBConnectionPolicyError):
    """A typed owner command was rejected by the exclusive state owner."""

    def __init__(self, code: str, message: str, *, request_id: str = "") -> None:
        self.code = str(code or "owner_error")
        self.message = str(message or "quack owner command rejected")
        self.request_id = str(request_id or "")
        super().__init__(self.message)


def _owner_command_text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise DuckDBConnectionPolicyError(
            f"quack owner command field {field!r} must be a non-empty string"
        )
    if len(value.encode("utf-8")) > 16_384:
        raise DuckDBConnectionPolicyError(f"quack owner command field {field!r} exceeds byte bound")
    return value


def validate_quack_owner_command(
    command: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and copy one command from the closed owner vocabulary.

    The request carries domain arguments only.  SQL, statement templates, and
    executable callbacks are deliberately absent; the owner maps the command
    name back to a canonical :class:`IntentRepository` method.
    """

    if type(command) is not str or command not in QUACK_OWNER_COMMANDS:
        raise DuckDBConnectionPolicyError(f"unsupported quack owner command: {command!r}")
    if not isinstance(payload, Mapping):
        raise DuckDBConnectionPolicyError("quack owner command payload must be a mapping")
    required, optional = _QUACK_OWNER_COMMAND_FIELDS[command]
    fields = frozenset(payload)
    if not required.issubset(fields) or not fields.issubset(required | optional):
        raise DuckDBConnectionPolicyError(
            f"quack owner command {command!r} fields do not match its closed schema"
        )
    copied = dict(payload)
    text_fields = {
        "task_cid_or_alias",
        "goal_cid_or_alias",
        "task_cid",
        "status",
        "reason",
        "evidence_kind",
        "digest",
        "outcome",
        "evidence_digest",
        "repair_head",
        "repair_tree",
    }
    for field in text_fields & fields:
        _owner_command_text(copied[field], field=field)
    if "attempt_id" in fields:
        attempt_id = copied["attempt_id"]
        if type(attempt_id) is not str or "\x00" in attempt_id:
            raise DuckDBConnectionPolicyError(
                "quack owner command field 'attempt_id' must be a string"
            )
    for field in {"expected_revision", "delay_ms", "selection_penalty"} & fields:
        value = copied[field]
        if type(value) is not int or value < 0:
            raise DuckDBConnectionPolicyError(
                f"quack owner command field {field!r} must be a non-negative integer"
            )
    for field in {"receipt", "expected_control_receipt", "body"} & fields:
        value = copied[field]
        if value is not None and not isinstance(value, Mapping):
            raise DuckDBConnectionPolicyError(
                f"quack owner command field {field!r} must be a mapping or null"
            )
        if isinstance(value, Mapping):
            copied[field] = dict(value)
    for field in {"argv", "evidence_digests"} & fields:
        value = copied[field]
        if value is None:
            continue
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise DuckDBConnectionPolicyError(
                f"quack owner command field {field!r} must be a sequence or null"
            )
        items = list(value)
        if len(items) > 4_096 or any(type(item) is not str for item in items):
            raise DuckDBConnectionPolicyError(
                f"quack owner command field {field!r} has invalid items"
            )
        copied[field] = items
    try:
        encoded = json.dumps(
            copied,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DuckDBConnectionPolicyError(
            "quack owner command payload is not canonical JSON"
        ) from exc
    if len(encoded) > QUACK_OWNER_COMMAND_MAX_BYTES:
        raise DuckDBConnectionPolicyError("quack owner command payload exceeds byte bound")
    return copied


def quack_owner_command_signature(
    payload: Mapping[str, Any],
    token: str,
) -> str:
    """Return the HMAC for one exact typed owner-command envelope.

    The raw Quack token never enters the request file.  The state owner verifies
    this signature, the request identity, store binding, generation, and
    freshness before it maps a closed command to repository code.
    """

    secret = str(token or "").strip()
    if not _QUACK_TOKEN_RE.fullmatch(secret):
        raise DuckDBConnectionPolicyError(
            "quack owner command requires the admitted opaque transport token"
        )
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    try:
        encoded = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DuckDBConnectionPolicyError(
            "quack owner command envelope is not canonical JSON"
        ) from exc
    return hmac.new(secret.encode("utf-8"), encoded, hashlib.sha256).hexdigest()


def validate_quack_owner_command_request(
    request: Mapping[str, Any],
    *,
    token: str,
    expected_request_id: str,
    expected_store_id: str,
    expected_store_generation: str,
    now_ms: int | None = None,
    max_age_ms: int = QUACK_OWNER_COMMAND_MAX_AGE_MS,
) -> tuple[str, dict[str, Any]]:
    """Authenticate and bind one owner-side request before dispatch.

    Callers remain responsible for same-user, regular-file, size, and replay
    checks around the inbox.  This helper seals the content-level identity,
    HMAC, freshness, store generation, and closed command payload.
    """

    if not isinstance(request, Mapping):
        raise DuckDBConnectionPolicyError("quack owner command request must be a mapping")
    expected_fields = {
        "schema",
        "request_id",
        "issued_at_ms",
        "writer_identity",
        "store_id",
        "store_generation",
        "command",
        "payload",
        "signature",
    }
    if set(request) != expected_fields:
        raise DuckDBConnectionPolicyError(
            "quack owner command request fields do not match the closed envelope"
        )
    request_id = str(request.get("request_id") or "")
    if (
        request.get("schema") != QUACK_OWNER_COMMAND_REQUEST_SCHEMA
        or _QUACK_OWNER_REQUEST_ID_RE.fullmatch(request_id) is None
        or request_id != expected_request_id
    ):
        raise DuckDBConnectionPolicyError("quack owner command request identity is invalid")
    issued_at_ms = request.get("issued_at_ms")
    if type(issued_at_ms) is not int:
        raise DuckDBConnectionPolicyError("quack owner command issued_at_ms must be an integer")
    if type(max_age_ms) is not int or not 1 <= max_age_ms <= 60_000:
        raise DuckDBConnectionPolicyError("quack owner command freshness bound is invalid")
    observed_now = int(time.time() * 1000) if now_ms is None else now_ms
    if type(observed_now) is not int:
        raise DuckDBConnectionPolicyError("quack owner command current time must be an integer")
    age_ms = observed_now - issued_at_ms
    if age_ms < -5_000 or age_ms > max_age_ms:
        raise DuckDBConnectionPolicyError("quack owner command request is stale or future-dated")
    if _QUACK_OWNER_WRITER_RE.fullmatch(str(request.get("writer_identity") or "")) is None:
        raise DuckDBConnectionPolicyError("quack owner command writer identity is invalid")
    if (
        request.get("store_id") != expected_store_id
        or request.get("store_generation") != expected_store_generation
        or not expected_store_id
        or not expected_store_generation
    ):
        raise DuckDBConnectionPolicyError(
            "quack owner command store or generation binding is stale"
        )
    observed_signature = str(request.get("signature") or "")
    expected_signature = quack_owner_command_signature(request, token)
    if not hmac.compare_digest(observed_signature, expected_signature):
        raise DuckDBConnectionPolicyError("quack owner command authorization is invalid")
    command = str(request.get("command") or "")
    payload = request.get("payload")
    if not isinstance(payload, Mapping):
        raise DuckDBConnectionPolicyError("quack owner command payload must be a mapping")
    return command, validate_quack_owner_command(command, payload)


def quack_owner_command_response(
    request: Mapping[str, Any],
    *,
    token: str,
    result: Mapping[str, Any] | None = None,
    error_code: str = "",
    error_message: str = "",
) -> dict[str, Any]:
    """Build and owner-sign the exact response consumed by command clients."""

    common: dict[str, Any] = {
        "schema": QUACK_OWNER_COMMAND_RESPONSE_SCHEMA,
        "request_id": str(request.get("request_id") or ""),
        "command": str(request.get("command") or ""),
        "store_id": str(request.get("store_id") or ""),
        "store_generation": str(request.get("store_generation") or ""),
    }
    if result is not None and not error_code and not error_message:
        response = {**common, "ok": True, "result": dict(result)}
    else:
        code = str(error_code or "owner_error")
        message = str(error_message or "typed owner command rejected")
        response = {
            **common,
            "ok": False,
            "error_code": code,
            "error_message": message,
        }
    response["signature"] = quack_owner_command_signature(response, token)
    return response


def quack_attach_lock_path(uri: str = "") -> Path:
    """Return the process-shared lock that serializes Quack ATTACH storms."""

    command_dir = quack_owner_command_dir()
    if command_dir is not None:
        return command_dir.parent / "attach.lock"
    digest = hashlib.sha256(
        str(uri or os.environ.get("IPFS_ACCELERATE_AGENT_QUACK_ENDPOINT", "")).encode(
            "utf-8"
        )
    ).hexdigest()[:16]
    return Path(os.environ.get("TMPDIR", "/tmp")) / (
        f"ipfs-accelerate-quack-attach-{digest}.lock"
    )


def _is_transient_quack_attach_error(detail: str) -> bool:
    """Return whether ATTACH may retry without wedging the exclusive owner.

    ``Authentication failed`` is not transient: eight retries from each lane
    fill Quack's small listen backlog and turn a live token into a board-wide
    stall. Retry only connect-time transport failures while the owner is
    still binding its port.
    """

    text = str(detail or "")
    lowered = text.lower()
    if "Authentication failed" in text or "authentication token" in lowered:
        return False
    return (
        "connection refused" in lowered
        or "connection reset" in lowered
        or "timed out" in lowered
        or "timeout" in lowered
        or "could not connect" in lowered
    )


def _is_quack_session_dead(exc: BaseException) -> bool:
    """Return whether a Quack SQL error means the ATTACH session is gone.

    Query-level ``Authorization failed`` is not session death: dropping the
    attached connection forces a new ATTACH, which is what wedges the
    owner and turns later ticks into ``quack_attach_failed``.
    """

    detail = str(exc)
    lowered = detail.lower()
    return (
        "Authentication failed" in detail
        or "connection refused" in lowered
        or "connection reset" in lowered
        or "connection closed" in lowered
        or "not connected" in lowered
        or "could not connect" in lowered
    )


def quack_session_is_live(connection: Any) -> bool:
    """Probe one attached Quack session without opening another ATTACH."""

    if connection is None or getattr(connection, "_closed", False):
        return False
    catalog = str(getattr(connection, "_default_catalog", "") or _QUACK_CONTROL_CATALOG)
    raw = getattr(connection, "_connection", connection)
    execute = getattr(raw, "execute", None)
    if not callable(execute):
        return False
    try:
        probed = execute(f"SELECT count(*) FROM {catalog}.tasks")
        _consume_duckdb_result(probed)
    except Exception:
        return False
    return True


def quack_owner_command_dir(store_id: object = "") -> Path | None:
    """Return the exclusive owner's local typed-command inbox, if configured."""

    explicit = str(os.environ.get("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", "") or "").strip()
    if explicit:
        return Path(explicit)
    store = str(
        store_id or os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "") or ""
    ).strip()
    if not store:
        return None
    path = Path(store)
    if path.suffix.lower() in {".duckdb", ".ddb"}:
        return path.expanduser().resolve().parent / "quack-owner" / "mutations"
    return None


def _read_quack_owner_command_response(path: Path) -> Mapping[str, Any]:
    """Read one same-UID, bounded, regular response without following links."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise DuckDBConnectionPolicyError(
            "this platform cannot safely open Quack owner command responses"
        )
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | nofollow)
    except OSError as exc:
        raise DuckDBConnectionPolicyError(
            "quack owner command response is not a safe regular file"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_size <= 0
            or metadata.st_size > QUACK_OWNER_COMMAND_MAX_BYTES
        ):
            raise DuckDBConnectionPolicyError(
                "quack owner command response owner, type, or size is invalid"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            encoded = handle.read(QUACK_OWNER_COMMAND_MAX_BYTES + 1)
        if len(encoded) != metadata.st_size:
            raise DuckDBConnectionPolicyError(
                "quack owner command response changed while being read"
            )
        try:
            payload = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DuckDBConnectionPolicyError(
                "quack owner command response is not valid JSON"
            ) from exc
        if not isinstance(payload, Mapping):
            raise DuckDBConnectionPolicyError("quack owner command response must be a mapping")
        return payload
    finally:
        os.close(descriptor)


def submit_quack_owner_command(
    command: str,
    payload: Mapping[str, Any],
    *,
    timeout_seconds: float = 15.0,
) -> Mapping[str, Any]:
    """Submit one typed command and return its typed result mapping.

    This filesystem rendezvous exists only because the currently admitted
    Quack build cannot update attached base tables.  It is not a SQL tunnel.
    """

    import uuid

    command_name = str(command or "")
    command_payload = validate_quack_owner_command(command_name, payload)
    maximum_timeout = (
        660.0
        if command_name == QUACK_OWNER_COMMAND_RECOVER_TYPED_DEFERRAL_BUDGET
        else 60.0
    )
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not 0 < float(timeout_seconds) <= maximum_timeout
    ):
        raise DuckDBConnectionPolicyError(
            "quack owner command timeout exceeds its closed command bound"
        )
    target = quack_owner_command_dir()
    if target is None:
        raise DuckDBConnectionPolicyError(
            "quack ATTACH cannot mutate remote base tables; set "
            "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR or "
            "IPFS_ACCELERATE_AGENT_STATE_STORE_ID so the state-owner can "
            "apply a typed command"
        )
    target.mkdir(parents=True, exist_ok=True)
    os.chmod(target, 0o700)
    request_id = uuid.uuid4().hex
    request_path = target / f"{request_id}.request.json"
    done_path = target / f"{request_id}.done.json"
    token = resolve_quack_attach_token()
    request_payload: dict[str, Any] = {
        "schema": QUACK_OWNER_COMMAND_REQUEST_SCHEMA,
        "request_id": request_id,
        "issued_at_ms": int(time.time() * 1000),
        "writer_identity": f"supervisor-process:{os.getpid()}",
        "store_id": str(os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "") or "").strip(),
        "store_generation": str(
            os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", "") or ""
        ).strip(),
        "command": command_name,
        "payload": command_payload,
    }
    if not request_payload["store_id"] or not request_payload["store_generation"]:
        raise DuckDBConnectionPolicyError(
            "quack owner command requires exact store and generation bindings"
        )
    request_payload["signature"] = quack_owner_command_signature(
        request_payload,
        token,
    )
    encoded_request = (
        json.dumps(
            request_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary_path = target / f".{request_id}.request.tmp"
    descriptor = os.open(
        temporary_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded_request)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, request_path)
        os.chmod(request_path, 0o600)
    except BaseException:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        if done_path.is_file():
            response = _read_quack_owner_command_response(done_path)
            common = {
                "schema",
                "request_id",
                "command",
                "store_id",
                "store_generation",
                "ok",
                "signature",
            }
            variant = (
                {"result"}
                if response.get("ok") is True
                else {
                    "error_code",
                    "error_message",
                }
            )
            if (
                set(response) != common | variant
                or response.get("schema") != QUACK_OWNER_COMMAND_RESPONSE_SCHEMA
                or response.get("request_id") != request_id
                or response.get("command") != command_name
                or response.get("store_id") != request_payload["store_id"]
                or response.get("store_generation") != request_payload["store_generation"]
                or type(response.get("ok")) is not bool
            ):
                raise DuckDBConnectionPolicyError("quack owner command response binding is invalid")
            observed_signature = str(response.get("signature") or "")
            expected_signature = quack_owner_command_signature(response, token)
            if not hmac.compare_digest(observed_signature, expected_signature):
                raise DuckDBConnectionPolicyError(
                    "quack owner command response authorization is invalid"
                )
            # Only an owner-authenticated response may retire the request.
            # A forged same-UID completion file can cause a closed failure but
            # cannot cancel a command that the owner may still recover.
            try:
                request_path.unlink(missing_ok=True)
                done_path.unlink(missing_ok=True)
            except OSError:
                pass
            if response["ok"] is not True:
                raise QuackOwnerCommandRemoteError(
                    str(response.get("error_code") or "owner_error"),
                    str(response.get("error_message") or "owner command rejected"),
                    request_id=request_id,
                )
            result = response.get("result")
            if not isinstance(result, Mapping):
                raise DuckDBConnectionPolicyError("quack owner command result must be a mapping")
            return dict(result)
        time.sleep(0.05)
    raise DuckDBConnectionPolicyError(
        "timed out waiting for quack state-owner to apply typed command"
    )


# The directory environment variable remains stable for deployed supervisors;
# only the contents changed from SQL mutation requests to typed commands.
quack_owner_mutation_dir = quack_owner_command_dir


def _consume_duckdb_result(connection: Any) -> None:
    try:
        connection.fetchall()
    except Exception:
        pass


_QUACK_RELATION_RE = re.compile(
    r"\b(FROM|JOIN)\s+(?![\w]+\.)([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)


def _qualify_quack_statement(sql: str, catalog: str) -> str:
    """Prefix unqualified FROM/JOIN tables with the attached Quack catalog.

    TOKEN sessions accept ``control_plane.tasks`` at ATTACH probe time but
    reject later unqualified ``FROM tasks`` with Authorization failed even
    after USE. Qualification keeps the attached catalog explicit.
    """

    def _replace(match: re.Match[str]) -> str:
        return f"{match.group(1)} {catalog}.{match.group(2)}"

    return _QUACK_RELATION_RE.sub(_replace, str(sql))


def open_quack_transport_connection(
    uri: str,
    *,
    token: str = "",
) -> DuckDBConnection:
    """Attach to the exclusive Quack state-owner (multi-reader/multi-writer).

    This is a transport connection, not a direct file open. The sealed
    one-writer file policy does not apply: Quack ATTACH requires a process
    that can reach the loopback state-owner.

    ATTACH is serialized across supervisor processes so the native Quack
    listen backlog (a handful of pending TCP handshakes) is not stormed.
    Retry sleeps happen *outside* that lock; holding it across backoff
    makes sibling lanes time out on ``attach.lock`` and die. Widening
    Quack listen parallelism is not required once ATTACH syscalls are
    sequential.
    """

    text = quack_transport_uri(uri)
    if not text:
        raise DuckDBConnectionPolicyError(f"invalid or non-loopback quack URI: {uri!r}")
    if "'" in text or ";" in text or "\x00" in text:
        raise DuckDBConnectionPolicyError("quack URI contains forbidden characters")
    try:
        import duckdb
    except ImportError as exc:
        raise DuckDBConnectionPolicyError("DuckDB is required for Quack transport") from exc
    secret = resolve_quack_attach_token(token)
    if secret and not _QUACK_TOKEN_RE.fullmatch(secret):
        raise DuckDBConnectionPolicyError(
            "quack attach token must be an opaque url-safe secret"
        )
    attach = f"ATTACH '{text}' AS {_QUACK_CONTROL_CATALOG} (READ_WRITE"
    if secret:
        attach += f", TOKEN '{secret}'"
    attach += ")"
    last_error: Exception | None = None
    attempts = 8
    lock_path = quack_attach_lock_path(text)
    lock_timeout = max(DEFAULT_LOCK_TIMEOUT_SECONDS, 45.0)
    for attempt in range(attempts):
        connection = duckdb.connect(":memory:")
        try:
            connection.execute("LOAD quack")
            with exclusive_file_lock(lock_path, timeout_seconds=lock_timeout):
                attached = connection.execute(attach)
                _consume_duckdb_result(attached)
            used = connection.execute(f"USE {_QUACK_CONTROL_CATALOG}")
            _consume_duckdb_result(used)
            probed = connection.execute(
                f"SELECT count(*) FROM {_QUACK_CONTROL_CATALOG}.tasks"
            )
            _consume_duckdb_result(probed)
            wrapped = DuckDBConnection.wrap(connection)
            wrapped._default_catalog = _QUACK_CONTROL_CATALOG
            return wrapped
        except Exception as exc:
            last_error = exc
            try:
                connection.close()
            except Exception:
                pass
            if (
                not _is_transient_quack_attach_error(str(exc))
                or attempt == attempts - 1
            ):
                break
            time.sleep(0.5 * (attempt + 1) + random.random() * 0.4)
    assert last_error is not None
    detail = str(last_error)
    lowered = detail.lower()
    if (
        _is_transient_quack_attach_error(detail)
        or "Authentication failed" in detail
        or "authentication token" in lowered
    ):
        raise DuckDBConnectionPolicyError(
            "quack attach authentication failed "
            f"uri={text!r} token_present={bool(secret)} "
            f"token_sha16={_quack_token_fingerprint(secret)}"
        ) from last_error
    raise last_error


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
