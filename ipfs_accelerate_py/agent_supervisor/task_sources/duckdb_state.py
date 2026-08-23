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
import re
import sqlite3
import stat
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .control_plane_contracts import (
    canonical_json_bytes,
    content_identity,
    is_secret_handle,
)
from .quack_owner_mutation import (
    QuackOwnerMutationEnvelopeError,
    build_mutation_request,
    mutation_envelope_exists_at,
    open_mutation_inbox_directory,
    parse_mutation_result,
    read_envelope_at,
    unlink_mutation_envelope_at,
    write_envelope_atomic_at,
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
    ("allow_persistent_secrets", "false", False),
    ("lock_configuration", "true", True),
)
# Quack 1.5.5+c154811 creates an internal connection for each authenticated
# request.  That worker requires both extension autoload and external access
# even after ``quack`` itself has been loaded; disabling either makes every
# remote request fail with HTTP 500.  This is therefore a distinct service
# policy, not a relaxation of ``DUCKDB_CONNECTION_POLICY_SETTINGS``.  It is
# admitted only for the loopback, single-owner transport process.  Automatic
# installation and unsigned bytes stay disabled and the configuration is
# immutable before the connection is returned.  The raw capability token and
# mutation inbox are stripped from every provider subprocess.
QUACK_OWNER_CONNECTION_POLICY_SETTINGS = (
    ("autoinstall_known_extensions", "false", False),
    ("autoload_known_extensions", "true", True),
    ("enable_external_access", "true", True),
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


class QuackTransportContentionError(DuckDBConnectionPolicyError):
    """Loopback Quack ATTACH failed because the owner was busy or contended."""


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


def _verify_connection_policy(
    connection: Any,
    *,
    settings: Sequence[tuple[str, str, bool]],
    policy_name: str,
) -> None:
    setting_names = tuple(
        name for name, _configured, _expected in settings
    )
    expressions = ", ".join(
        f"current_setting('{name}')" for name in setting_names
    )
    try:
        row = connection.execute(f"SELECT {expressions}").fetchone()
    except Exception as exc:
        raise DuckDBConnectionPolicyError(
            f"could not verify {policy_name}"
        ) from exc
    expected = tuple(
        value for _name, _configured, value in settings
    )
    if (
        not isinstance(row, tuple)
        or len(row) != len(expected)
        or any(type(value) is not bool for value in row)
        or row != expected
    ):
        raise DuckDBConnectionPolicyError(
            f"{policy_name} verification failed"
        )


def _verify_duckdb_connection_policy(connection: Any) -> None:
    _verify_connection_policy(
        connection,
        settings=DUCKDB_CONNECTION_POLICY_SETTINGS,
        policy_name="DuckDB supervisor connection policy",
    )


def _verify_quack_owner_connection_policy(connection: Any) -> None:
    _verify_connection_policy(
        connection,
        settings=QUACK_OWNER_CONNECTION_POLICY_SETTINGS,
        policy_name="Quack owner connection policy",
    )


def connect_duckdb_with_policy(
    duckdb_module: Any,
    database: Path | str,
    *,
    read_only: bool = False,
    configuration: Mapping[str, Any] | None = None,
) -> Any:
    """Open and verify one configuration-locked supervisor connection.

    The dynamic-extension, external-access, and persistent-secret settings plus
    the configuration lock
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


def connect_duckdb_with_quack_owner_policy(
    duckdb_module: Any,
    database: Path | str,
    *,
    configuration: Mapping[str, Any] | None = None,
) -> Any:
    """Open the locked, loopback-only Quack state-owner connection.

    The Quack request worker currently requires autoload and external access.
    Those settings remain enabled only in this exclusive transport authority;
    ordinary supervisor connections continue to use the deny-by-default
    policy.  No network installation is permitted, the exact installed Quack
    extension is loaded with a literal statement, its required surface is
    verified, and configuration is locked atomically at connection birth.
    """

    tuning = {
        "threads": "1",
        "memory_limit": DEFAULT_MEMORY_LIMIT,
    }
    tuning.update(_connection_tuning(configuration))
    connect_config = {
        name: configured
        for name, configured, _expected in QUACK_OWNER_CONNECTION_POLICY_SETTINGS
        if name != "lock_configuration"
    }
    connect_config.update(tuning)
    connect_config["lock_configuration"] = "true"
    connection = duckdb_module.connect(str(database), config=connect_config)
    try:
        connection.execute("LOAD quack")
        functions = connection.execute(
            """
            SELECT count(DISTINCT function_name)
            FROM duckdb_functions()
            WHERE function_name IN ('quack_serve', 'quack_query')
            """
        ).fetchone()
        if not isinstance(functions, tuple) or int(functions[0]) != 2:
            raise DuckDBConnectionPolicyError(
                "preloaded Quack extension lacks its required functions"
            )
        _verify_quack_owner_connection_policy(connection)
    except BaseException:
        connection.close()
        raise
    return connection


def connect_duckdb_quack_owner_with_policy(
    duckdb_module: Any,
    database: Path | str,
    *,
    configuration: Mapping[str, Any] | None = None,
) -> Any:
    """Birth the exclusive typed state-owner connection with Quack preloaded.

    Ordinary supervisor connections deny extension loading from connection
    birth.  The one typed Quack state owner loads the already-installed,
    signed ``quack`` extension without installation, then disables external
    access and seals configuration before exposing the connection.
    """

    tuning = {
        "threads": "1",
        "memory_limit": DEFAULT_MEMORY_LIMIT,
    }
    tuning.update(_connection_tuning(configuration))
    birth_config: dict[str, str] = {
        "autoinstall_known_extensions": "false",
        "autoload_known_extensions": "false",
        "enable_external_access": "true",
        "allow_unsigned_extensions": "false",
        "allow_persistent_secrets": "false",
        **tuning,
    }
    connection = duckdb_module.connect(
        str(database),
        read_only=False,
        config=birth_config,
    )
    try:
        connection.execute("LOAD httpfs")
        connection.execute("LOAD quack")
        connection.execute("SET enable_external_access=false")
        connection.execute("SET lock_configuration=true")
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


def _is_quack_invalid_connection(exc: BaseException) -> bool:
    """Return whether Quack dropped the ATTACH query handle."""

    return "invalid connection id" in str(exc).lower()


def _result_description_columns(result: Any) -> tuple[str, ...]:
    description = getattr(result, "description", None) or ()
    columns: list[str] = []
    for item in description:
        if (
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes, bytearray))
            and len(item) > 0
        ):
            columns.append(str(item[0]))
        else:
            columns.append(str(item))
    return tuple(columns)


def _materialize_duckdb_result(result: Any) -> tuple[tuple[str, ...], list[Any]]:
    """Fetch rows before reading description or columns.

    Quack remote results consume the cursor when ``description`` is read
    first.  The next ``execute`` then fails with ``Invalid connection id``.
    """

    fetchall = getattr(result, "fetchall", None)
    if not callable(fetchall):
        return _result_description_columns(result), []
    try:
        rows = list(fetchall() or [])
    except Exception as exc:
        if _is_quack_invalid_connection(exc):
            raise
        rows = []
    columns = _result_description_columns(result)
    if not columns and rows:
        first = rows[0]
        if isinstance(first, Mapping):
            columns = tuple(
                str(key)
                for key in first
                if isinstance(key, str) and key and not key.isdigit()
            )
        elif isinstance(first, Sequence) and not isinstance(
            first, (str, bytes, bytearray)
        ):
            columns = tuple(str(index) for index in range(len(first)))
    return columns, rows


class DuckDBCursor:
    """Materialize a result before another statement reuses the connection."""

    def __init__(self, connection: Any, *, dml: bool = False) -> None:
        columns, rows = _materialize_duckdb_result(connection)
        self._columns = columns
        self._rows = rows
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


def _sql_statement_token_shapes(sql: str) -> tuple[tuple[str, ...], ...]:
    """Tokenize statement shapes without confusing comments or string literals."""

    import duckdb

    text = str(sql)
    encoded = text.encode("utf-8")
    statements: list[tuple[str, ...]] = []
    current: list[str] = []
    for offset, token_type in duckdb.tokenize(text):
        if token_type.name == "operator" and encoded[offset : offset + 1] == b";":
            if current:
                statements.append(tuple(current))
                current = []
            continue
        if len(current) >= 6:
            continue
        if token_type.name == "keyword":
            match = re.match(rb"[A-Za-z_][A-Za-z0-9_]*", encoded[offset:])
            current.append(
                match.group(0).decode("ascii").upper()
                if match is not None
                else "<TOKEN>"
            )
        else:
            current.append("<TOKEN>")
    if current:
        statements.append(tuple(current))
    return tuple(statements)


def _classify_transaction_sql(sql: str) -> tuple[str, str]:
    """Classify one admitted transaction statement and reject ambiguous forms."""

    statements = _sql_statement_token_shapes(sql)
    transaction_prefixes = {
        "ABORT",
        "BEGIN",
        "COMMIT",
        "END",
        "RELEASE",
        "ROLLBACK",
        "SAVEPOINT",
        "START",
    }
    transaction_statements = [
        statement
        for statement in statements
        if statement and statement[0] in transaction_prefixes
    ]
    if transaction_statements and len(statements) != 1:
        raise DuckDBConnectionPolicyError(
            "transaction control must be one standalone SQL statement"
        )
    if not transaction_statements:
        return "", ""
    words = transaction_statements[0]
    if words in {
        ("BEGIN",),
        ("BEGIN", "TRANSACTION"),
        ("BEGIN", "DEFERRED"),
        ("BEGIN", "DEFERRED", "TRANSACTION"),
        ("BEGIN", "EXCLUSIVE"),
        ("BEGIN", "EXCLUSIVE", "TRANSACTION"),
        ("BEGIN", "IMMEDIATE"),
        ("BEGIN", "IMMEDIATE", "TRANSACTION"),
        ("START", "TRANSACTION"),
    }:
        return "begin", "BEGIN TRANSACTION"
    if words in {
        ("COMMIT",),
        ("COMMIT", "TRANSACTION"),
        ("COMMIT", "WORK"),
        ("END",),
        ("END", "TRANSACTION"),
        ("END", "WORK"),
    }:
        return "commit", "COMMIT"
    if words in {
        ("ABORT",),
        ("ABORT", "TRANSACTION"),
        ("ABORT", "WORK"),
        ("ROLLBACK",),
        ("ROLLBACK", "TRANSACTION"),
        ("ROLLBACK", "WORK"),
    }:
        return "rollback", "ROLLBACK"
    raise DuckDBConnectionPolicyError(
        "unsupported or ambiguous transaction control statement"
    )


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
        quack_owner: bool = False,
        _preload_quack_for_state_owner: bool = False,
    ) -> None:
        if type(quack_owner) is not bool:
            raise TypeError("quack_owner must be a boolean")
        if type(_preload_quack_for_state_owner) is not bool:
            raise TypeError("_preload_quack_for_state_owner must be a boolean")
        if is_quack_transport_target(path):
            raise DuckDBConnectionPolicyError(
                "quack transport URIs cannot be opened as DuckDB files; "
                "use open_quack_transport_connection"
            )
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if is_sqlite_database(self.path):
            raise ValueError(f"legacy SQLite database must be migrated before opening: {self.path}")
        self._execution_lock = threading.RLock()
        self._execution_condition = threading.Condition(self._execution_lock)
        self._transaction_active = False
        self._transaction_lock_owner = 0
        self._transaction_on_context = bool(transaction_on_context)
        self._context_depth = 0
        self._context_owner = 0
        self._context_finalizing = False
        self._closed = False
        self._closing_owner = 0
        self._poisoned = False
        self._default_catalog = None
        self._quack_mutation_binding: dict[str, Any] | None = None
        self._quack_mutation_token = ""
        self._quack_pending_mutations: list[dict[str, Any]] = []
        self._pooled = False
        self._quack_uri = ""
        self._active_catalog = None
        self._raw_wrapper_key = 0
        self._lock_context = exclusive_file_lock(
            self.path.with_name(f".{self.path.name}.lock"),
            timeout_seconds=timeout_seconds,
        )
        self._lock_context.__enter__()
        try:
            import duckdb

            if _preload_quack_for_state_owner:
                connector = connect_duckdb_quack_owner_with_policy
            elif quack_owner:
                connector = connect_duckdb_with_quack_owner_policy
            else:
                connector = connect_duckdb_with_policy
            self._connection = connector(
                duckdb,
                self.path,
                configuration={"threads": threads, "memory_limit": memory_limit},
            )
            _register_duckdb_wrapper(self, self._connection)
        except BaseException:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
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
        instance._execution_lock = threading.RLock()
        instance._execution_condition = threading.Condition(instance._execution_lock)
        instance._transaction_active = False
        instance._transaction_lock_owner = 0
        instance._transaction_on_context = bool(transaction_on_context)
        instance._context_depth = 0
        instance._context_owner = 0
        instance._context_finalizing = False
        instance._closed = False
        instance._closing_owner = 0
        instance._poisoned = False
        instance._lock_context = None
        instance._default_catalog = None
        instance._quack_mutation_binding = None
        instance._quack_mutation_token = ""
        instance._quack_pending_mutations = []
        instance._pooled = False
        instance._quack_uri = ""
        instance._active_catalog = None
        instance._raw_wrapper_key = 0
        _register_duckdb_wrapper(instance, connection)
        return instance

    @property
    def in_transaction(self) -> bool:
        with self._execution_condition:
            return self._transaction_active

    def _wait_for_transaction_turn_locked(self) -> int:
        """Wait until this thread may use the shared native connection."""

        thread_id = threading.get_ident()
        while (
            not self._closed
            and not self._poisoned
            and (
                (
                    self._transaction_active
                    and self._transaction_lock_owner != thread_id
                )
                or (
                    self._context_finalizing
                    and self._context_owner != thread_id
                )
            )
        ):
            self._execution_condition.wait()
        return thread_id

    def _transaction_finished_locked(self) -> None:
        self._transaction_active = False
        self._transaction_lock_owner = 0
        self._execution_condition.notify_all()

    def _require_usable_locked(self) -> None:
        thread_id = threading.get_ident()
        if (
            self._closed
            or (
                self._closing_owner
                and self._closing_owner != thread_id
            )
            or self._poisoned
            or self._connection is None
        ):
            raise DuckDBConnectionPolicyError(
                "DuckDB connection is unusable after an uncertain transaction"
            )

    def _poison_locked(self) -> str:
        """Close an uncertain native handle and wake every excluded peer."""

        pooled = bool(self._pooled)
        uri = str(self._quack_uri or "") if pooled else ""
        raw = self._connection
        self._connection = None
        self._quack_pending_mutations = []
        self._poisoned = True
        self._pooled = False
        if pooled:
            self._closed = True
        self._transaction_finished_locked()
        if raw is not None:
            try:
                raw.close()
            except Exception:
                pass
            else:
                _unregister_duckdb_wrapper(self, raw)
        return uri

    def _evict_poisoned_pool_entry(self, uri: str) -> None:
        if not uri:
            return
        with _QUACK_ATTACH_LOCK:
            if _QUACK_TRANSPORT_CACHE.get(uri) is self:
                _QUACK_TRANSPORT_CACHE.pop(uri, None)

    def _discard_pooled_connection(self) -> None:
        """Administratively invalidate a cached handle regardless of API owner."""

        with self._execution_condition:
            uri = self._poison_locked()
            self._closed = True
            self._closing_owner = 0
            self._execution_condition.notify_all()
        self._evict_poisoned_pool_entry(uri)

    def execute(
        self,
        sql: str,
        parameters: Iterable[Any] | Mapping[str, Any] | None = None,
    ) -> DuckDBCursor:
        statement = str(sql)
        transaction_kind, canonical = _classify_transaction_sql(statement)
        if transaction_kind:
            statement = canonical
        normalized = " ".join(statement.strip().upper().split())
        begins_transaction = transaction_kind == "begin"
        ends_transaction = transaction_kind in {"commit", "rollback"}
        evict_uri = ""
        try:
            with self._execution_condition:
                thread_id = threading.get_ident()
                if ends_transaction:
                    if (
                        self._transaction_active
                        and self._transaction_lock_owner != thread_id
                    ):
                        raise DuckDBConnectionPolicyError(
                            "transaction termination is owned by another thread"
                        )
                    if not self._transaction_active:
                        raise DuckDBConnectionPolicyError(
                            "transaction termination requires an active transaction"
                        )
                else:
                    thread_id = self._wait_for_transaction_turn_locked()
                self._require_usable_locked()
                if begins_transaction and self._transaction_active:
                    raise DuckDBConnectionPolicyError(
                        "transaction is already active on this connection"
                    )
                try:
                    result = self._execute_locked(statement, normalized, parameters)
                except BaseException:
                    if begins_transaction:
                        self._quack_pending_mutations = []
                        try:
                            self._connection.rollback()
                        except Exception:
                            evict_uri = self._poison_locked()
                        else:
                            self._transaction_finished_locked()
                    elif ends_transaction:
                        evict_uri = self._poison_locked()
                    raise
                if begins_transaction and self._transaction_active:
                    self._transaction_lock_owner = thread_id
                elif ends_transaction and not self._transaction_active:
                    self._transaction_finished_locked()
                return result
        except BaseException:
            self._evict_poisoned_pool_entry(evict_uri)
            raise

    def _execute_locked(
        self,
        statement: str,
        normalized: str,
        parameters: Iterable[Any] | Mapping[str, Any] | None,
    ) -> DuckDBCursor:
        """Execute and materialize one result while owning the connection lock."""

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
            template_id = _QUACK_OWNER_MUTATION_SQL_TO_TEMPLATE.get(normalized)
            if template_id is None:
                if (
                    not self._transaction_active
                    and not self._quack_pending_mutations
                    and normalized.startswith(
                        (
                            "UPDATE ",
                            "DELETE ",
                            "MERGE ",
                            "INSERT OR REPLACE ",
                            "INSERT OR IGNORE ",
                        )
                    )
                ):
                    return _execute_quack_owner_mutation(statement, parameters)
                raise DuckDBConnectionPolicyError(
                    "quack owner mutation SQL is not in the closed template catalog"
                )
            if not self._transaction_active:
                raise DuckDBConnectionPolicyError(
                    "quack owner mutation requires an explicit transaction"
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
            if getattr(self, "_active_catalog", None) != catalog:
                used = self._connection.execute(f"USE {catalog}")
                _consume_duckdb_result(used)
                self._active_catalog = catalog
        if parameters is None:
            executed = self._connection.execute(statement)
        else:
            executed = self._connection.execute(statement, parameters)
        if executed is None:
            executed = self._connection
        if normalized.startswith("BEGIN"):
            self._transaction_active = True
        elif normalized in {"COMMIT", "ROLLBACK"}:
            self._transaction_active = False
        elif normalized.startswith("USE "):
            self._active_catalog = catalog or getattr(self, "_active_catalog", None)
        dml = normalized.startswith(("INSERT ", "UPDATE ", "DELETE "))
        return DuckDBCursor(executed, dml=dml)

    def executemany(
        self,
        sql: str,
        parameters: Iterable[Iterable[Any]],
    ) -> DuckDBCursor:
        transaction_kind, _canonical = _classify_transaction_sql(sql)
        if transaction_kind:
            raise DuckDBConnectionPolicyError(
                "executemany does not admit transaction control"
            )
        with self._execution_condition:
            self._wait_for_transaction_turn_locked()
            self._require_usable_locked()
            if getattr(self, "_default_catalog", None):
                raise DuckDBConnectionPolicyError(
                    "quack transport does not admit executemany"
                )
            self._connection.executemany(sql, parameters)
            return DuckDBCursor(self._connection, dml=True)

    def executescript(self, sql: str) -> DuckDBCursor:
        transaction_kind, _canonical = _classify_transaction_sql(sql)
        if transaction_kind:
            raise DuckDBConnectionPolicyError(
                "executescript does not admit transaction control"
            )
        with self._execution_condition:
            self._wait_for_transaction_turn_locked()
            self._require_usable_locked()
            if getattr(self, "_default_catalog", None):
                raise DuckDBConnectionPolicyError(
                    "quack transport does not admit scripts"
                )
            self._connection.execute(sql)
            return DuckDBCursor(self._connection)

    def commit(self) -> None:
        with self._execution_condition:
            thread_id = threading.get_ident()
            if self._context_depth and self._context_owner != thread_id:
                raise DuckDBConnectionPolicyError(
                    "DuckDB connection context is owned by another thread"
                )
            if (
                self._transaction_active
                and self._transaction_lock_owner != thread_id
            ):
                raise DuckDBConnectionPolicyError(
                    "transaction termination is owned by another thread"
                )
            self._require_usable_locked()
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
        evict_uri = ""
        try:
            with self._execution_condition:
                thread_id = threading.get_ident()
                if self._context_depth and self._context_owner != thread_id:
                    raise DuckDBConnectionPolicyError(
                        "DuckDB connection context is owned by another thread"
                    )
                if (
                    self._transaction_active
                    and self._transaction_lock_owner != thread_id
                ):
                    raise DuckDBConnectionPolicyError(
                        "transaction termination is owned by another thread"
                    )
                self._require_usable_locked()
                self._quack_pending_mutations = []
                if self._transaction_active:
                    try:
                        self._connection.rollback()
                    except BaseException:
                        evict_uri = self._poison_locked()
                        raise
                self._transaction_finished_locked()
        except BaseException:
            self._evict_poisoned_pool_entry(evict_uri)
            raise

    def close(self) -> None:
        with self._execution_condition:
            thread_id = threading.get_ident()
            if self._closed:
                return
            if self._closing_owner:
                if self._closing_owner == thread_id:
                    return
                raise DuckDBConnectionPolicyError(
                    "DuckDB connection is being closed by another thread"
                )
            if self._context_depth and self._context_owner != thread_id:
                raise DuckDBConnectionPolicyError(
                    "DuckDB connection context is owned by another thread"
                )
            if (
                self._transaction_active
                and self._transaction_lock_owner != thread_id
            ):
                raise DuckDBConnectionPolicyError(
                    "DuckDB transaction is owned by another thread"
                )
            if getattr(self, "_pooled", False):
                if self._transaction_active:
                    try:
                        self.rollback()
                    except Exception:
                        pass
                return
            self._closing_owner = thread_id
        try:
            try:
                self.rollback()
            except Exception:
                pass
            with self._execution_condition:
                raw = self._connection
                self._connection = None
                self._quack_pending_mutations = []
                self._transaction_finished_locked()
            if raw is not None:
                try:
                    raw.close()
                except Exception:
                    self._poisoned = True
                else:
                    _unregister_duckdb_wrapper(self, raw)
        finally:
            try:
                if self._lock_context is not None:
                    lock_context = self._lock_context
                    self._lock_context = None
                    lock_context.__exit__(None, None, None)
            finally:
                with self._execution_condition:
                    self._closed = True
                    self._closing_owner = 0
                    self._execution_condition.notify_all()

    def _release_quack_attach_session(self) -> None:
        raise DuckDBConnectionPolicyError(
            "transferring a live Quack session out of its synchronized wrapper "
            "is not admitted"
        )

    def _restore_quack_attach_session(self, uri: str) -> None:
        del uri
        raise DuckDBConnectionPolicyError(
            "transplanting a Quack session between synchronized wrappers is not admitted"
        )

    def __enter__(self) -> DuckDBConnection:
        with self._execution_condition:
            self._require_usable_locked()
            thread_id = threading.get_ident()
            if self._context_depth and self._context_owner != thread_id:
                raise DuckDBConnectionPolicyError(
                    "DuckDB connection context is owned by another thread"
                )
            if (
                self._transaction_active
                and self._transaction_lock_owner != thread_id
            ):
                raise DuckDBConnectionPolicyError(
                    "DuckDB transaction context is owned by another thread"
                )
            if self._context_depth == 0:
                self._context_owner = thread_id
                if self._transaction_on_context and not self._transaction_active:
                    try:
                        self.execute("BEGIN TRANSACTION")
                    except BaseException:
                        self._context_owner = 0
                        raise
            self._context_depth += 1
            return self

    def __exit__(self, exc_type: Any, _exc: Any, _traceback: Any) -> None:
        with self._execution_condition:
            thread_id = threading.get_ident()
            if self._context_depth <= 0 or self._context_owner != thread_id:
                raise DuckDBConnectionPolicyError(
                    "DuckDB connection context exit is owned by another thread"
                )
            if self._context_depth > 1:
                self._context_depth -= 1
                return
            self._context_finalizing = True
        try:
            if self._transaction_active:
                if exc_type is None:
                    self.commit()
                else:
                    self.rollback()
        finally:
            try:
                if self._lock_context is not None:
                    self.close()
            finally:
                with self._execution_condition:
                    self._context_depth = 0
                    self._context_owner = 0
                    self._context_finalizing = False
                    self._execution_condition.notify_all()


_RAW_WRAPPER_GUARD = threading.Lock()
_RAW_WRAPPERS: dict[
    int,
    tuple[Any, weakref.ReferenceType[DuckDBConnection]],
] = {}


def _register_duckdb_wrapper(wrapper: DuckDBConnection, raw: Any) -> None:
    """Fail closed instead of assigning independent locks to one raw handle."""

    key = id(raw)
    with _RAW_WRAPPER_GUARD:
        existing = _RAW_WRAPPERS.get(key)
        if existing is not None:
            raise DuckDBConnectionPolicyError(
                "native DuckDB connection is already owned by a synchronized wrapper"
            )

        def close_abandoned(
            reference: weakref.ReferenceType[DuckDBConnection],
        ) -> None:
            with _RAW_WRAPPER_GUARD:
                registered = _RAW_WRAPPERS.get(key)
                if (
                    registered is None
                    or registered[0] is not raw
                    or registered[1] is not reference
                ):
                    return
            close = getattr(raw, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    return
            with _RAW_WRAPPER_GUARD:
                registered = _RAW_WRAPPERS.get(key)
                if (
                    registered is not None
                    and registered[0] is raw
                    and registered[1] is reference
                ):
                    _RAW_WRAPPERS.pop(key, None)

        reference = weakref.ref(wrapper, close_abandoned)
        _RAW_WRAPPERS[key] = (raw, reference)
        wrapper._raw_wrapper_key = key


def _unregister_duckdb_wrapper(wrapper: DuckDBConnection, raw: Any) -> None:
    key = int(getattr(wrapper, "_raw_wrapper_key", 0) or 0)
    if not key:
        return
    with _RAW_WRAPPER_GUARD:
        existing = _RAW_WRAPPERS.get(key)
        if (
            existing is not None
            and existing[0] is raw
            and existing[1]() is wrapper
        ):
            _RAW_WRAPPERS.pop(key, None)
    wrapper._raw_wrapper_key = 0


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
_QUACK_OWNER_DML_PREFIXES = (
    "UPDATE ",
    "DELETE ",
    "MERGE ",
    "INSERT OR REPLACE",
    "INSERT OR IGNORE",
)
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
QUACK_OWNER_MUTATION_MAX_PARAMETERS = 17
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
QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT = "lease_queue_backoff_insert@1"
QUACK_MUTATION_LEASE_QUEUE_BACKOFF_UPDATE = "lease_queue_backoff_update@1"

QUACK_MUTATION_TASK_STATUS_TRANSITION = "task_status_transition@1"
QUACK_MUTATION_VALIDATION_RECORD = "validation_record@1"
QUACK_MUTATION_QUEUE_BACKOFF = "queue_backoff@1"


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
    _normalize_quack_mutation_sql(
        """
        INSERT INTO leases (
            task_cid, claim_cid, resolution_cid, claimant_did,
            logical_epoch, fencing_token, expires_at_ms, attempt,
            state, started_at_ms, release_reason, retry_not_before_ms,
            owner_session_id, fence_epoch, revision, extension_schema,
            extension_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
    ): QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT,
    _normalize_quack_mutation_sql(
        """
        UPDATE leases SET
            attempt = ?, retry_not_before_ms = ?,
            release_reason = ?, state = 'released',
            extension_schema = ?, extension_json = ?,
            revision = revision + 1
        WHERE task_cid = ?
        """
    ): QUACK_MUTATION_LEASE_QUEUE_BACKOFF_UPDATE,
}


class DuckDBQuackMutationConflictError(DuckDBConnectionPolicyError):
    """A closed owner-side compare-and-set observed a stale revision."""


class DuckDBQuackMutationTransitionError(DuckDBConnectionPolicyError):
    """Owner rejected a status CAS outside the closed transition matrix."""


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


_QUACK_MUTATION_DIR_ENV = "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR"
_RUNTIME_REGISTRY_PATH_ENV = "IPFS_ACCELERATE_AGENT_RUNTIME_REGISTRY_PATH"
_STATE_AUTHORITY_MODE_ENV = "IPFS_ACCELERATE_AGENT_STATE_AUTHORITY_MODE"
# Longer than implementation_max_timeout (14400s) so a live Grok run is not
# stolen. Shorter than a multi-day freeze so a dead in_progress gate unblocks.
STALE_IN_PROGRESS_UNSTALL_SECONDS = 16_200
_QUACK_ATTACH_LOCK = threading.RLock()
_QUACK_TRANSPORT_CACHE: dict[str, DuckDBConnection] = {}
QUACK_ATTACH_ATTEMPTS = 8
QUACK_ATTACH_BACKOFF_SECONDS: tuple[float, ...] = (
    0.05,
    0.1,
    0.2,
    0.4,
    0.8,
    1.6,
    3.2,
)
_QUACK_ATTACH_CONTENTION_MARKERS = (
    "authentication failed",
    "could not set lock",
    "conflicting lock",
    "lock timeout",
    "database is locked",
    "connection refused",
    "connection reset",
    "connection timed out",
    "temporarily unavailable",
    "resource temporarily unavailable",
    "too many clients",
    "too many connections",
    "broken pipe",
    "timeout",
    "busy",
    "locked",
    "contention",
)


_QUACK_ATTACH_TOKEN_ENV = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
_QUACK_SECRET_HANDLE_ENV = "IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE"

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
QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK = "rearm_blocked_task"
QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF = "record_queue_backoff"
QUACK_OWNER_COMMAND_RECORD_QUEUE_RETRY = "record_queue_retry"
QUACK_OWNER_COMMAND_RECORD_EVIDENCE = "record_evidence"
QUACK_OWNER_COMMAND_RECORD_VALIDATION_RESULT = "record_validation_result"
QUACK_OWNER_COMMANDS = frozenset(
    {
        QUACK_OWNER_COMMAND_COMPARE_AND_SET_STATUS,
        QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK,
        QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF,
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
        frozenset({"receipt", "evidence_digests"}),
    ),
    QUACK_OWNER_COMMAND_REARM_BLOCKED_TASK: (
        frozenset({"task_cid_or_alias"}),
        frozenset({"receipt"}),
    ),
    QUACK_OWNER_COMMAND_RECORD_QUEUE_BACKOFF: (
        frozenset({"task_cid", "delay_ms"}),
        frozenset({"reason", "selection_penalty"}),
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
        "task_cid",
        "status",
        "reason",
        "evidence_kind",
        "digest",
        "outcome",
        "evidence_digest",
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
    for field in {"receipt", "body"} & fields:
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
        or _is_quack_invalid_connection(exc)
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
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or not 0 < float(timeout_seconds) <= 60
    ):
        raise DuckDBConnectionPolicyError("quack owner command timeout must be in (0, 60] seconds")
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

def quack_owner_mutation_inbox_path(
    runtime_registry_path: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Bind one owner/worker mutation inbox to the runtime registry path."""

    text = str(runtime_registry_path or "").strip()
    if not text or "\x00" in text:
        raise DuckDBConnectionPolicyError(
            "quack owner mutation inbox requires a runtime registry path"
        )
    registry = Path(text).expanduser()
    root: Path | None = None
    if not registry.is_absolute() and repository_root is None:
        raise DuckDBConnectionPolicyError(
            "relative runtime registry requires an explicit repository root"
        )
    if repository_root is not None:
        root_text = str(repository_root or "").strip()
        if not root_text or "\x00" in root_text:
            raise DuckDBConnectionPolicyError(
                "quack owner mutation inbox requires a valid repository root"
            )
        root_path = Path(root_text).expanduser()
        if not root_path.is_absolute():
            raise DuckDBConnectionPolicyError(
                "quack owner mutation inbox requires an absolute repository root"
            )
        root = root_path.resolve()
        if not registry.is_absolute():
            registry = root / registry
    registry = registry.resolve()
    if root is not None:
        try:
            registry.relative_to(root)
        except ValueError as exc:
            raise DuckDBConnectionPolicyError(
                "quack owner mutation inbox escapes the repository root"
            ) from exc
    # Do not resolve the final component.  The owner opens it with O_NOFOLLOW
    # so a replaced ``mutations`` directory cannot redirect trusted writes.
    return registry / "mutations"


def quack_owner_mutation_dir(store_id: object = "") -> Path | None:
    """Resolve the inbox below the accepted repository root, or fail closed.

    Quack authority requires an explicit inbox that matches the bound runtime
    registry.  Lanes may run in child worktrees, so cwd is never an authority
    for a relative store identity; the lifecycle repository-root marker is the
    shared, accepted anchor used by every lane.
    """

    explicit = str(
        os.environ.get(_QUACK_MUTATION_DIR_ENV, "") or ""
    ).strip()
    registry = str(
        os.environ.get(_RUNTIME_REGISTRY_PATH_ENV, "") or ""
    ).strip()
    authority_mode = str(
        os.environ.get(_STATE_AUTHORITY_MODE_ENV, "") or ""
    ).strip().lower().replace("-", "_")
    if authority_mode == "quack" and not registry:
        raise DuckDBConnectionPolicyError(
            "quack owner mutation inbox requires a bound runtime registry"
        )
    if authority_mode == "quack" and not explicit:
        raise DuckDBConnectionPolicyError(
            "quack owner mutation inbox requires an explicit mutation binding"
        )
    registry_inbox = (
        quack_owner_mutation_inbox_path(registry) if registry else None
    )
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if not explicit_path.is_absolute():
            raise DuckDBConnectionPolicyError(
                "quack owner mutation inbox must be an absolute path"
            )
        # Resolve only the parent.  The final inbox component must remain
        # available to the O_NOFOLLOW admission check below.
        explicit_inbox = explicit_path.parent.resolve() / explicit_path.name
        if registry_inbox is not None and explicit_inbox != registry_inbox:
            raise DuckDBConnectionPolicyError(
                "quack owner mutation inbox does not match the bound runtime "
                "registry"
            )
        return explicit_inbox
    if registry_inbox is not None:
        return registry_inbox
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
    root_text = str(
        os.environ.get("IPFS_ACCELERATE_LIFECYCLE_REPOSITORY_ROOT", "") or ""
    ).strip()
    root: Path | None = None
    if root_text:
        root = Path(root_text).expanduser().resolve()
    if not path.is_absolute():
        if root is None:
            return None
        path = root / path
    path = path.resolve()
    if root is not None:
        try:
            path.relative_to(root)
        except ValueError:
            return None
    if path.suffix.lower() in {".duckdb", ".ddb"}:
        inbox = (path.parent / "quack-owner" / "mutations").resolve()
        if root is not None:
            try:
                inbox.relative_to(root)
            except ValueError:
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
    if len(parameters) > QUACK_OWNER_MUTATION_MAX_PARAMETERS:
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
    if templates in {
        (
            QUACK_MUTATION_LEASE_QUEUE_BACKOFF_INSERT,
            QUACK_MUTATION_DOMAIN_EVENT_INSERT,
        ),
        (
            QUACK_MUTATION_LEASE_QUEUE_BACKOFF_UPDATE,
            QUACK_MUTATION_DOMAIN_EVENT_INSERT,
        ),
    }:
        return QUACK_MUTATION_QUEUE_BACKOFF
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
        if error_code == "transition_invalid":
            raise DuckDBQuackMutationTransitionError(
                "quack owner mutation failed: transition_invalid"
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


def _row_tuple(row: Any) -> tuple[Any, ...]:
    values = getattr(row, "_values", None)
    if values is not None:
        return tuple(values)
    if isinstance(row, Mapping):
        return tuple(row.values())
    try:
        return tuple(row)
    except TypeError:
        return (row,)


def _parse_task_updated_at(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _quote_duckdb_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _drop_task_status_indexes(connection: Any) -> list[str]:
    """Drop status-bearing task indexes that can fatal status UPDATEs.

    DuckDB ART indexes on ``tasks.status`` have failed closed with
    ``Failed to delete all rows from index`` after a leftover in_progress
    CAS. Drop them for the unstall UPDATE, then recreate from the saved
    DDL.
    """

    try:
        rows = connection.execute(
            "SELECT index_name, sql FROM duckdb_indexes() WHERE table_name = 'tasks'"
        ).fetchall()
    except Exception:
        return []
    statements: list[str] = []
    names: list[str] = []
    for row in rows:
        name, sql = _row_tuple(row)[:2]
        text = str(sql or "").strip()
        if "status" not in text.lower():
            continue
        names.append(str(name))
        if text.upper().startswith("CREATE "):
            statements.append(text)
    for name in names:
        connection.execute("DROP INDEX IF EXISTS " + _quote_duckdb_ident(name))
    return statements


def _restore_task_status_indexes(connection: Any, statements: Sequence[str]) -> None:
    for sql in statements:
        text = str(sql or "").strip()
        if not text:
            continue
        connection.execute(text)


def unstall_stale_in_progress_tasks(
    connection: Any,
    *,
    now: datetime | None = None,
    stale_seconds: int = STALE_IN_PROGRESS_UNSTALL_SECONDS,
) -> dict[str, Any]:
    """Return in_progress tasks that have been idle longer than a live attempt.

    Used by the exclusive Quack owner so a dead gate (attach contention,
    crashed implementer, leftover CAS) cannot freeze the rest of the board.
    Live implementations heartbeat ``updated_at`` on claim; a run still under
    ``implementation_max_timeout`` is left alone.
    """

    if stale_seconds <= 0:
        raise ValueError("stale_seconds must be positive")
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    rows = connection.execute(
        "SELECT task_cid, task_alias, status, revision, updated_at "
        "FROM tasks WHERE status = 'in_progress'"
    ).fetchall()
    unstalled: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    pending: list[tuple[Any, ...]] = []
    for row in rows:
        task_cid, task_alias, status, revision, updated_at = _row_tuple(row)[:5]
        updated = _parse_task_updated_at(updated_at)
        if updated is None:
            skipped.append(
                {
                    "task_cid": str(task_cid),
                    "task_alias": str(task_alias),
                    "reason": "updated_at_unparseable",
                }
            )
            continue
        age = (clock - updated).total_seconds()
        if age < float(stale_seconds):
            skipped.append(
                {
                    "task_cid": str(task_cid),
                    "task_alias": str(task_alias),
                    "reason": "still_within_live_attempt_window",
                    "age_seconds": int(age),
                }
            )
            continue
        pending.append((task_cid, task_alias, status, revision, int(age)))
    index_sql: list[str] = []
    if pending:
        index_sql = _drop_task_status_indexes(connection)
    try:
        for task_cid, task_alias, status, revision, age in pending:
            new_revision = int(revision) + 1
            stamp = clock.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            connection.execute(
                "UPDATE tasks SET status = ?, revision = ?, updated_at = ? "
                "WHERE task_cid = ? AND revision = ? AND status = 'in_progress'",
                ["retrying", new_revision, stamp, str(task_cid), int(revision)],
            )
            unstalled.append(
                {
                    "task_cid": str(task_cid),
                    "task_alias": str(task_alias),
                    "previous_revision": int(revision),
                    "revision": new_revision,
                    "previous_status": str(status),
                    "status": "retrying",
                    "age_seconds": int(age),
                }
            )
    finally:
        if index_sql:
            _restore_task_status_indexes(connection, index_sql)
    return {
        "unstalled": unstalled,
        "skipped": skipped,
        "stale_seconds": int(stale_seconds),
        "status_indexes_rebuilt": list(index_sql),
    }


BOARD_UNSTALL_OWNER_OP = "board_unstall"
BOARD_UNSTALL_BOUNCE_NAME = "board-unstall.bounce"
OWNER_BOARD_UNSTALL_BOUNCE_MIN_AGE_SECONDS = 15.0
_OWNER_INBOX_DML_PREFIXES = _QUACK_OWNER_DML_PREFIXES + ("INSERT ",)


def apply_owner_command_payload(
    connection: Any,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply one owner-inbox command on the exclusive writer connection.

    ATTACH clients cannot UPDATE/DELETE base tables, and ``quack_serve``
    occupies the listen handle, so the owner applies DML here. Structured
    ``board_unstall`` commands recover leftover ``in_progress`` gates without
    the client issuing SQL.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("mutation request must be an object")
    op = str(payload.get("op") or "").strip()
    if op == BOARD_UNSTALL_OWNER_OP:
        stale_raw = payload.get("stale_seconds", STALE_IN_PROGRESS_UNSTALL_SECONDS)
        stale_seconds = int(stale_raw)
        result = unstall_stale_in_progress_tasks(
            connection, stale_seconds=stale_seconds
        )
        return {
            "ok": True,
            "rowcount": len(result.get("unstalled") or []),
            "board_unstall": result,
        }
    sql = str(payload.get("sql") or "")
    normalized = " ".join(sql.strip().upper().split())
    if not normalized.startswith(_OWNER_INBOX_DML_PREFIXES):
        raise ValueError("mutation inbox accepts only owner DML")
    if ";" in normalized.rstrip(";"):
        raise ValueError("mutation inbox accepts exactly one SQL statement")
    parameters = payload.get("parameters")
    result = (
        connection.execute(sql)
        if parameters is None
        else connection.execute(sql, parameters)
    )
    rowcount = -1
    try:
        if getattr(result, "description", None):
            result.fetchall()
        elif hasattr(result, "rowcount"):
            rowcount = int(result.rowcount)
    except Exception:
        pass
    return {"ok": True, "rowcount": rowcount}


def request_owner_board_unstall(
    *,
    stale_seconds: int = STALE_IN_PROGRESS_UNSTALL_SECONDS,
    wait: bool = True,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    """Ask the exclusive owner to retry leftover in_progress gates.

    Used when Quack ATTACH is contended: the daemon cannot SELECT or UPDATE,
    but it can still write a mutation-inbox request. ``wait=False`` is the
    attach-deferral path so the process does not block on a dead owner.
    """

    if int(stale_seconds) <= 0:
        raise ValueError("stale_seconds must be positive")
    target = quack_owner_mutation_dir()
    if target is None:
        return {"ok": False, "requested": False, "error": "no_mutation_dir"}
    import uuid

    target.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex
    request_path = target / f"{request_id}.request.json"
    done_path = target / f"{request_id}.done.json"
    request_path.write_text(
        json.dumps(
            {
                "op": BOARD_UNSTALL_OWNER_OP,
                "stale_seconds": int(stale_seconds),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    bounce_path = target / BOARD_UNSTALL_BOUNCE_NAME
    bounce_path.write_text(
        json.dumps(
            {
                "op": BOARD_UNSTALL_OWNER_OP,
                "request_id": request_id,
                "stale_seconds": int(stale_seconds),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not wait:
        return {
            "ok": True,
            "requested": True,
            "waited": False,
            "request_id": request_id,
        }
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        if done_path.is_file():
            reply = json.loads(done_path.read_text(encoding="utf-8"))
            try:
                request_path.unlink(missing_ok=True)
                done_path.unlink(missing_ok=True)
            except OSError:
                pass
            if not isinstance(reply, dict):
                raise DuckDBConnectionPolicyError(
                    "malformed board unstall owner reply"
                )
            return reply
        time.sleep(0.05)
    raise DuckDBConnectionPolicyError(
        "timed out waiting for quack state-owner to unstall the task board"
    )


def owner_should_recycle_for_board_unstall(
    mutation_dir: object = None,
    *,
    min_age_seconds: float = OWNER_BOARD_UNSTALL_BOUNCE_MIN_AGE_SECONDS,
) -> bool:
    """Return whether the exclusive owner should restart to apply board unstall.

    ``quack_serve`` occupies the listen connection, so leftover in_progress
    UPDATEs only land in the pre-listen writer window. A bounce marker older
    than ``min_age_seconds`` means the live owner could not apply it.
    """

    if mutation_dir is not None:
        target = Path(str(mutation_dir))
    else:
        target = quack_owner_mutation_dir()
    if target is None:
        return False
    bounce = target / BOARD_UNSTALL_BOUNCE_NAME
    if not bounce.is_file():
        return False
    try:
        age = time.time() - bounce.stat().st_mtime
    except OSError:
        return False
    return age >= float(min_age_seconds)


def clear_owner_board_unstall_bounce(mutation_dir: object = None) -> None:
    """Drop the recycle marker after a pre-listen unstall has run."""

    if mutation_dir is not None:
        target = Path(str(mutation_dir))
    else:
        target = quack_owner_mutation_dir()
    if target is None:
        return
    bounce = target / BOARD_UNSTALL_BOUNCE_NAME
    try:
        bounce.unlink(missing_ok=True)
    except OSError:
        pass


def _quack_owner_mutation_required(normalized: str) -> bool:
    return normalized.startswith(_QUACK_OWNER_DML_PREFIXES)


def _execute_quack_owner_mutation(
    statement: str,
    parameters: Iterable[Any] | Mapping[str, Any] | None = None,
) -> DuckDBCursor:
    """Apply UPDATE/DELETE on the exclusive owner connection.

    This Quack ATTACH build can SELECT/INSERT new rows but cannot UPDATE or
    DELETE attached base tables. Mutations stay on the state-owner that
    already holds the exclusive file connection.
    """

    target = quack_owner_mutation_dir()
    if target is None:
        raise DuckDBConnectionPolicyError(
            "quack owner mutation store does not resolve to a bounded inbox"
        )
    try:
        inbox_fd = open_mutation_inbox_directory(target)
    except QuackOwnerMutationEnvelopeError as exc:
        raise DuckDBConnectionPolicyError(
            "quack owner mutation inbox is not a safe owner directory"
        ) from exc
    request_id = uuid.uuid4().hex
    request_name = f"{request_id}.request.json"
    done_name = f"{request_id}.done.json"
    if parameters is None:
        bound: Any = None
    elif isinstance(parameters, Mapping):
        bound = dict(parameters)
    else:
        bound = list(parameters)
    token = str(os.environ.get("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "") or "")
    store_id = str(
        os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "") or ""
    )
    generation_raw = str(
        os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_GENERATION", "") or ""
    )
    try:
        try:
            generation = int(generation_raw)
            envelope = build_mutation_request(
                request_id=request_id,
                store_id=store_id,
                generation=generation,
                sql=statement,
                parameters=bound,
                token=token,
            )
            write_envelope_atomic_at(
                inbox_fd,
                request_name,
                envelope,
                replace=False,
            )
        except (OSError, ValueError, QuackOwnerMutationEnvelopeError) as exc:
            raise DuckDBConnectionPolicyError(
                "could not create authenticated Quack owner mutation: "
                f"{type(exc).__name__}"
            ) from exc
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if mutation_envelope_exists_at(inbox_fd, done_name):
                try:
                    payload = parse_mutation_result(
                        read_envelope_at(inbox_fd, done_name),
                        token=token,
                        expected_request_id=request_id,
                        expected_store_id=store_id,
                        expected_generation=generation,
                    )
                except (OSError, ValueError, QuackOwnerMutationEnvelopeError) as exc:
                    raise DuckDBConnectionPolicyError(
                        "Quack owner mutation result was not admissible: "
                        f"{type(exc).__name__}"
                    ) from exc
                finally:
                    try:
                        unlink_mutation_envelope_at(
                            inbox_fd,
                            request_name,
                            missing_ok=True,
                        )
                        unlink_mutation_envelope_at(
                            inbox_fd,
                            done_name,
                            missing_ok=True,
                        )
                    except OSError:
                        pass
                if payload.get("ok") is not True:
                    raise DuckDBConnectionPolicyError(
                        "quack owner mutation failed: "
                        + str(payload.get("error_code") or "unknown")
                    )
                cursor = DuckDBCursor.__new__(DuckDBCursor)
                cursor._columns = tuple(
                    str(item) for item in payload.get("columns") or ()
                )
                cursor._rows = [
                    tuple(item) for item in payload.get("rows") or ()
                ]
                cursor._offset = 0
                cursor.rowcount = int(payload.get("rowcount") or -1)
                return cursor
            time.sleep(0.05)
        raise DuckDBConnectionPolicyError(
            "timed out waiting for quack state-owner to apply mutation; "
            "outcome is unknown and must not be replayed blindly"
        )
    finally:
        os.close(inbox_fd)


def _consume_duckdb_result(connection: Any) -> None:
    try:
        connection.fetchall()
    except Exception:
        pass


def _quack_store_id() -> str:
    store = str(os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "") or "").strip()
    if store:
        return store
    raw = str(os.environ.get("IPFS_ACCELERATE_AGENT_DATABASE_PROGRAM_JSON", "") or "").strip()
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("store_id") or "").strip()


def quack_token_vault_path() -> Path | None:
    """Return the owner token-vault path when the process can locate it."""

    explicit = str(
        os.environ.get("IPFS_ACCELERATE_AGENT_QUACK_TOKEN_FILE", "") or ""
    ).strip()
    if explicit:
        return Path(explicit)
    store = _quack_store_id()
    if not store:
        return None
    handle = str(
        os.environ.get("IPFS_ACCELERATE_AGENT_STATE_ENDPOINT_SECRET_HANDLE", "")
        or "env://IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
    ).strip()
    owner_dir = Path(store).expanduser().resolve().parent / "quack-owner"
    safe = handle.replace(":", "_").replace("/", "_")
    return owner_dir / f"{safe}{'.quack-token'}"


def _read_quack_token_vault(path: Path) -> str:
    try:
        metadata = os.lstat(path)
    except OSError:
        return ""
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        return ""
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not _QUACK_TOKEN_RE.fullmatch(token):
        return ""
    return token


def persist_quack_attach_token_vault(token: str = "") -> Path | None:
    """Re-materialize the 0600 owner vault when it is missing.

    The operator used to unlink the vault after supervisor launch. Owner
    recycle, operator status, and later ATTACH then fail closed even though
    a live daemon still holds the credential. Write the vault from a trusted
    env or explicit token only when the file is absent or empty so remaining
    board drain can keep attaching.
    """

    secret = str(
        token or os.environ.get("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "") or ""
    ).strip()
    if not secret or not _QUACK_TOKEN_RE.fullmatch(secret):
        return None
    vault = quack_token_vault_path()
    if vault is None:
        return None
    existing = _read_quack_token_vault(vault)
    if existing:
        return vault
    try:
        vault.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        return None
    tmp = vault.with_name(f".{vault.name}.{os.getpid()}.tmp")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        fd = os.open(tmp, flags, 0o600)
        try:
            os.write(fd, f"{secret}\n".encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, vault)
        os.chmod(vault, 0o600)
        dir_fd = os.open(vault.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return None
    if _read_quack_token_vault(vault) != secret:
        return None
    return vault


def resolve_quack_attach_token(token: str = "") -> str:
    """Resolve the current owner attach token.

    The vault file is preferred over a process environment value so a
    restarted owner generation is not blocked by a stale supervisor env.
    When the vault is missing, persist the live env token so owner recycle
    and operator status can keep draining the board.
    """

    explicit = str(token or "").strip()
    if explicit:
        if not _QUACK_TOKEN_RE.fullmatch(explicit):
            raise DuckDBConnectionPolicyError(
                "quack attach token must be an opaque url-safe secret"
            )
        return explicit
    vault = quack_token_vault_path()
    if vault is not None:
        material = _read_quack_token_vault(vault)
        if material:
            return material
    secret = str(os.environ.get("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "") or "").strip()
    if secret and not _QUACK_TOKEN_RE.fullmatch(secret):
        raise DuckDBConnectionPolicyError(
            "quack attach token must be an opaque url-safe secret"
        )
    if secret:
        persist_quack_attach_token_vault(secret)
    return secret


def quack_attach_error_is_contention(exc: BaseException) -> bool:
    """Return whether a failed ATTACH is owner-side contention, not a policy bug."""

    text = " ".join(str(exc).lower().split())
    return any(marker in text for marker in _QUACK_ATTACH_CONTENTION_MARKERS)


def reset_quack_transport_cache() -> None:
    """Drop cached loopback Quack attachments (tests and owner restart)."""

    with _QUACK_ATTACH_LOCK:
        cached = list(_QUACK_TRANSPORT_CACHE.items())
        _QUACK_TRANSPORT_CACHE.clear()
    for _uri, connection in cached:
        try:
            connection._discard_pooled_connection()
        except Exception:
            pass


def _probe_quack_connection(connection: Any) -> None:
    probed = connection.execute("SELECT 1")
    _consume_duckdb_result(probed)


def _attach_quack_once(uri: str, secret: str) -> Any:
    import duckdb

    connection = duckdb.connect(":memory:")
    try:
        connection.execute("LOAD quack")
        try:
            connection.execute("SET httpfs_connection_caching = true")
        except Exception:
            pass
        attach = (
            f"ATTACH '{uri}' AS {_QUACK_CONTROL_CATALOG} "
            "(READ_WRITE, DISABLE_SSL true"
        )
        if secret:
            attach += f", TOKEN '{secret}'"
        attach += ")"
        attached = connection.execute(attach)
        _consume_duckdb_result(attached)
        used = connection.execute(f"USE {_QUACK_CONTROL_CATALOG}")
        _consume_duckdb_result(used)
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
            [uri],
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
        return connection, binding
    except Exception:
        try:
            connection.close()
        except Exception:
            pass
        raise


def _open_quack_transport_connection_once(
    uri: str,
    *,
    token: str = "",
) -> DuckDBConnection:
    """Attach to the exclusive Quack state-owner (multi-reader/multi-writer).

    This is a transport connection, not a direct file open. The sealed
    one-writer file policy does not apply: Quack ATTACH requires a process
    that can reach the loopback state-owner.

    Attach is serialized and retried: the owner DuckDB connection is the
    single writer, the serve backlog is small, and a new ATTACH per query
    otherwise loses the handshake to contention (often reported as
    ``Authentication failed``). Live attachments are reused in-process.
    Closed catalog mutations bind the live server identity onto the wrapper.
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
    del duckdb

    while True:
        with _QUACK_ATTACH_LOCK:
            cached = _QUACK_TRANSPORT_CACHE.get(text)
            if cached is not None and getattr(cached, "_closed", False):
                _QUACK_TRANSPORT_CACHE.pop(text, None)
                cached = None
        if cached is not None:
            try:
                _probe_quack_connection(cached)
            except Exception:
                with _QUACK_ATTACH_LOCK:
                    if _QUACK_TRANSPORT_CACHE.get(text) is cached:
                        _QUACK_TRANSPORT_CACHE.pop(text, None)
                try:
                    cached._discard_pooled_connection()
                except Exception:
                    pass
            else:
                with _QUACK_ATTACH_LOCK:
                    if (
                        _QUACK_TRANSPORT_CACHE.get(text) is cached
                        and not getattr(cached, "_closed", False)
                    ):
                        return cached
                continue

        with _QUACK_ATTACH_LOCK:
            cached = _QUACK_TRANSPORT_CACHE.get(text)
            if cached is not None and not getattr(cached, "_closed", False):
                continue
            if cached is not None:
                _QUACK_TRANSPORT_CACHE.pop(text, None)

            last_error: BaseException | None = None
            for attempt in range(QUACK_ATTACH_ATTEMPTS):
                secret = resolve_quack_attach_token(token)
                admitted_handle_binding: dict[str, Any] = {}
                if not secret:
                    secret, admitted_handle_binding = _resolve_quack_token_handle(
                        uri=text
                    )
                try:
                    attached = _attach_quack_once(text, secret)
                    if (
                        isinstance(attached, tuple)
                        and len(attached) == 2
                    ):
                        raw, binding = attached
                    else:
                        raw = attached
                        binding = getattr(raw, "_quack_live_binding", None)
                except BaseException as exc:
                    last_error = exc
                    if (
                        attempt + 1 >= QUACK_ATTACH_ATTEMPTS
                        or not quack_attach_error_is_contention(exc)
                    ):
                        break
                    delay = QUACK_ATTACH_BACKOFF_SECONDS[
                        min(attempt, len(QUACK_ATTACH_BACKOFF_SECONDS) - 1)
                    ]
                    time.sleep(delay)
                    continue
                wrapped = DuckDBConnection.wrap(raw)
                wrapped._default_catalog = _QUACK_CONTROL_CATALOG
                wrapped._active_catalog = _QUACK_CONTROL_CATALOG
                wrapped._pooled = True
                wrapped._quack_uri = text
                wrapped._quack_mutation_binding = (
                    dict(binding) if isinstance(binding, Mapping) else None
                )
                wrapped._quack_mutation_token = secret
                if admitted_handle_binding:
                    live = wrapped._quack_mutation_binding or {}
                    exact_handle_binding = {
                        "server_id": live.get("server_id"),
                        "store_id": live.get("store_id"),
                        "database_uuid": live.get("database_uuid"),
                        "schema_revision": live.get("schema_revision"),
                        "schema_fingerprint": live.get("schema_fingerprint"),
                        "generation": live.get("generation"),
                        "process_birth_id": live.get("process_birth_id"),
                        "listen_uri": live.get("listen_uri"),
                        "extension_fingerprint": live.get("extension_fingerprint"),
                    }
                    mismatched = [
                        name
                        for name, value in exact_handle_binding.items()
                        if admitted_handle_binding.get(name) != value
                    ]
                    if mismatched:
                        wrapped._discard_pooled_connection()
                        raise DuckDBConnectionPolicyError(
                            "quack live binding differs from the admitted owner status: "
                            + ", ".join(mismatched)
                        )
                _QUACK_TRANSPORT_CACHE[text] = wrapped
                return wrapped

            if last_error is not None and quack_attach_error_is_contention(last_error):
                raise QuackTransportContentionError(
                    "quack control-plane attach contended: " + str(last_error)
                ) from last_error
            if last_error is not None:
                raise last_error
            raise DuckDBConnectionPolicyError("quack control-plane attach failed")


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
    quack_owner: bool = False,
) -> DuckDBConnection:
    if is_quack_transport_target(path):
        if quack_owner:
            raise DuckDBConnectionPolicyError(
                "Quack transport clients cannot assume the owner policy"
            )
        return open_quack_transport_connection(path)
    return DuckDBConnection(
        path,
        timeout_seconds=timeout_seconds,
        memory_limit=memory_limit,
        threads=threads,
        quack_owner=quack_owner,
    )


def open_quack_state_owner_connection(
    path: Path | str,
    *,
    timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    threads: int = 1,
) -> DuckDBConnection:
    """Open the sole file-owning connection with signed Quack preloaded."""

    if is_quack_transport_target(path):
        raise DuckDBConnectionPolicyError(
            "the Quack state owner requires an exact DuckDB file path"
        )
    return DuckDBConnection(
        path,
        timeout_seconds=timeout_seconds,
        memory_limit=memory_limit,
        threads=threads,
        _preload_quack_for_state_owner=True,
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
