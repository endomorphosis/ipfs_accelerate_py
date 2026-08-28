"""Shared DuckDB primitives for durable agent-supervisor state.

DuckDB permits only one external writer process. Supervisor stores therefore
use short-lived connections protected by a process-shared file lock. Legacy
SQLite databases are copied table-by-table into the new DuckDB file and are
left untouched as rollback evidence unless strict DuckDB-only mode is enabled.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .quack_owner_mutation import (
    QUACK_OWNER_MUTATION_MAX_STEPS,
    QuackOwnerMutationError,
    execute_mutation_bundle,
    mutation_step,
    validate_mutation_binding,
)

DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0
DEFAULT_MEMORY_LIMIT = "256MB"
DUCKDB_ONLY_ENV = "IPFS_ACCELERATE_DUCKDB_ONLY"
QUACK_ENDPOINT_ENV = "IPFS_ACCELERATE_AGENT_QUACK_ENDPOINT"
QUACK_TOKEN_ENV = "IPFS_ACCELERATE_AGENT_QUACK_TOKEN"
QUACK_REQUIRE_ENV = "IPFS_ACCELERATE_AGENT_QUACK_REQUIRE"
QUACK_PREFER_ENV = "IPFS_ACCELERATE_AGENT_QUACK_PREFER"
QUACK_STORE_ID_ENV = "IPFS_ACCELERATE_AGENT_STATE_STORE_ID"
QUACK_MUTATION_DIR_ENV = "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR"
QUACK_MUTATION_BINDING_ENV = "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_BINDING"
CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV = (
    "IPFS_ACCELERATE_AGENT_DUCKDB_EXTENSION_DIRECTORY"
)
QUACK_LIVE_OWNER_FILE_FALLBACK_TIMEOUT_SECONDS = 1.0
_LOGGER = logging.getLogger(__name__)
SQLITE_MAGIC = b"SQLite format 3\0"
# Loopback Quack URIs are the multi-writer control-plane transport. File
# connections remain one-writer. Clients prefer a live owner advertised next
# to the store; a file open is a logged fallback, not a silent one.
# IPFS_ACCELERATE_AGENT_QUACK_REQUIRE restores fail-closed (no file fallback).
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
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out acquiring DuckDB process lock: {lock_path}"
                    ) from exc
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
        self._quack_mutation_binding = None
        self._quack_mutation_token = ""
        self._quack_mutation_inbox = None
        self._quack_pending_mutations: list[dict[str, Any]] = []
        self._quack_uri = ""
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
        instance._quack_mutation_inbox = None
        instance._quack_pending_mutations = []
        instance._quack_uri = ""
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
            # The Quack attachment is a read-only snapshot. End it before the
            # exclusive owner applies the complete mutation bundle atomically.
            self._connection.execute("ROLLBACK")
            _consume_duckdb_result(self._connection)
            self._transaction_active = False
            binding = self._quack_mutation_binding
            inbox = self._quack_mutation_inbox
            if not isinstance(binding, Mapping) or inbox is None:
                raise DuckDBConnectionPolicyError(
                    "quack mutation lacks an exact live owner binding"
                )
            try:
                rowcount = execute_mutation_bundle(
                    pending,
                    binding=binding,
                    token=self._quack_mutation_token,
                    inbox=inbox,
                )
            except QuackOwnerMutationError as exc:
                raise DuckDBConnectionPolicyError(
                    f"quack owner mutation failed: {exc.code}"
                ) from exc
            self._reattach_quack_transport()
            return _empty_duckdb_cursor(rowcount=rowcount)
        is_dml = normalized.startswith(("INSERT ", "UPDATE ", "DELETE ", "MERGE "))
        if catalog and is_dml:
            if not self._transaction_active:
                raise DuckDBConnectionPolicyError(
                    "quack owner mutation requires an explicit transaction"
                )
            if parameters is None or isinstance(parameters, Mapping):
                raise DuckDBConnectionPolicyError(
                    "quack owner mutation templates require positional parameters"
                )
            try:
                step = mutation_step(statement, list(parameters))
            except QuackOwnerMutationError as exc:
                raise DuckDBConnectionPolicyError(
                    f"quack owner mutation rejected: {exc.code}"
                ) from exc
            if len(self._quack_pending_mutations) >= QUACK_OWNER_MUTATION_MAX_STEPS:
                raise DuckDBConnectionPolicyError(
                    "quack owner mutation bundle exceeds its step bound"
                )
            self._quack_pending_mutations.append(step)
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
        if getattr(self, "_default_catalog", None):
            raise DuckDBConnectionPolicyError(
                "quack transport does not admit executemany"
            )
        self._connection.executemany(sql, parameters)
        return DuckDBCursor(self._connection, dml=True)

    def executescript(self, sql: str) -> DuckDBCursor:
        if getattr(self, "_default_catalog", None):
            raise DuckDBConnectionPolicyError(
                "quack transport does not admit scripts"
            )
        self._connection.execute(sql)
        return DuckDBCursor(self._connection)

    def commit(self) -> None:
        if self._quack_pending_mutations and not self._transaction_active:
            raise DuckDBConnectionPolicyError(
                "quack mutation bundle exists outside an active transaction"
            )
        if self._transaction_active:
            self.execute("COMMIT")

    def rollback(self) -> None:
        self._quack_pending_mutations = []
        if self._transaction_active:
            self._connection.rollback()
            self._transaction_active = False

    def _reattach_quack_transport(self) -> None:
        uri = str(self._quack_uri or "")
        if not uri:
            raise DuckDBConnectionPolicyError("quack transport URI is unavailable")
        try:
            self._connection.close()
        except Exception:
            pass
        fresh = open_quack_transport_connection(
            uri, token=self._quack_mutation_token
        )
        self._connection = fresh._connection
        self._default_catalog = fresh._default_catalog
        self._quack_mutation_binding = fresh._quack_mutation_binding
        self._quack_mutation_token = fresh._quack_mutation_token
        self._quack_mutation_inbox = fresh._quack_mutation_inbox
        self._quack_uri = fresh._quack_uri
        fresh._connection = None
        fresh._closed = True

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
_QUACK_STATUS_FILENAME = "quack-state-server.status.json"
_QUACK_CONTROL_CATALOG = "control_plane"


class QuackEndpointDiscovery:
    """Lookup of a live loopback Quack owner for one DuckDB file."""

    def __init__(
        self,
        *,
        uri: str = "",
        token: str = "",
        source: str = "none",
        reason: str = "",
        status_path: str = "",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.uri = str(uri or "")
        self.token = str(token or "")
        self.source = str(source or "none")
        self.reason = str(reason or "")
        self.status_path = str(status_path or "")
        self.details = dict(details or {})

    @property
    def found(self) -> bool:
        return bool(self.uri)


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _resolve_store_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        return candidate.resolve()
    except OSError:
        return candidate


def _owner_process_alive(identity: Mapping[str, Any] | None) -> bool:
    if not isinstance(identity, Mapping):
        return False
    birth = identity.get("process_birth")
    if not isinstance(birth, Mapping):
        return False
    pid = birth.get("pid")
    if type(pid) is not int or pid <= 1:
        return False
    return Path(f"/proc/{pid}").exists()


def _read_json_object(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, Mapping) else None


def _candidate_quack_status_paths(database: Path) -> tuple[Path, ...]:
    parent = database.parent
    name = _QUACK_STATUS_FILENAME
    return (
        parent / "quack-owner" / name,
        parent / "live" / "state" / "quack-owner" / name,
        parent / "live" / "state" / "quack-owner-v2" / name,
    )


def _read_quack_client_token(
    status_dir: Path,
    status: Mapping[str, Any] | None = None,
) -> str:
    env_token = str(os.environ.get(QUACK_TOKEN_ENV, "") or "").strip()
    if env_token:
        return env_token
    handle = ""
    if isinstance(status, Mapping):
        handle = str(status.get("secret_handle") or "").strip()
        identity = status.get("identity")
        if not handle and isinstance(identity, Mapping):
            handle = str(identity.get("secret_handle") or "").strip()
    if handle:
        safe = handle.replace(":", "_").replace("/", "_")
        vault = status_dir / f"{safe}.quack-token"
        try:
            raw = vault.read_text(encoding="ascii").strip()
        except OSError:
            raw = ""
        except UnicodeDecodeError:
            raw = ""
        if raw and _QUACK_TOKEN_RE.fullmatch(raw):
            return raw
    return ""


def _status_listen_uri(status: Mapping[str, Any]) -> str:
    identity = status.get("identity")
    if isinstance(identity, Mapping):
        uri = quack_transport_uri(identity.get("listen_uri") or "")
        if uri:
            return uri
    return quack_transport_uri(status.get("listen_uri") or "")


def _status_database_path(status: Mapping[str, Any]) -> str:
    return str(status.get("database_path") or "").strip()


def _same_database(status_database: str, requested: Path) -> bool:
    if not status_database:
        return False
    candidate = Path(status_database).expanduser()
    if not candidate.is_absolute():
        candidate = requested.parent / candidate
    try:
        return candidate.resolve() == requested
    except OSError:
        return str(candidate) == str(requested)


def _reject_status(
    status: Mapping[str, Any] | None,
    requested: Path,
    status_path: Path,
) -> str:
    if status is None:
        return "status_unreadable"
    if str(status.get("lifecycle") or "") != "ready":
        return f"lifecycle_{status.get('lifecycle') or 'missing'}"
    if not _same_database(_status_database_path(status), requested):
        return "database_path_mismatch"
    identity = status.get("identity")
    if isinstance(identity, Mapping) and str(identity.get("status") or "") not in {
        "",
        "ready",
    }:
        return f"identity_status_{identity.get('status')}"
    uri = _status_listen_uri(status)
    if not uri:
        return "listen_uri_missing"
    if isinstance(identity, Mapping):
        birth = identity.get("process_birth")
        if isinstance(birth, Mapping) and type(birth.get("pid")) is int:
            if not _owner_process_alive(identity):
                return "owner_process_not_alive"
    del status_path
    return ""


def discover_live_quack_endpoint(path: Path | str) -> QuackEndpointDiscovery:
    """Find a live loopback Quack owner bound to this exact DuckDB file."""

    requested = _resolve_store_path(path)
    env_uri = quack_transport_uri(os.environ.get(QUACK_ENDPOINT_ENV, "") or "")
    store_id = str(os.environ.get(QUACK_STORE_ID_ENV, "") or "").strip()
    env_store_matches = False
    if store_id:
        env_store_matches = _same_database(store_id, requested) or _same_database(
            str(_resolve_store_path(store_id)), requested
        )
    rejections: list[str] = []
    for status_path in _candidate_quack_status_paths(requested):
        if not status_path.is_file():
            continue
        status = _read_json_object(status_path)
        if status is None:
            rejections.append(f"{status_path}:status_unreadable")
            continue
        if not _same_database(_status_database_path(status), requested):
            continue
        reason = _reject_status(status, requested, status_path)
        if reason:
            rejections.append(f"{status_path}:{reason}")
            continue
        assert status is not None
        uri = _status_listen_uri(status)
        if env_uri and env_uri != uri:
            rejections.append(f"{status_path}:env_uri_mismatch")
            continue
        return QuackEndpointDiscovery(
            uri=uri,
            token=_read_quack_client_token(status_path.parent, status),
            source="status_file",
            reason="ready_owner",
            status_path=str(status_path),
            details={
                "lifecycle": str(status.get("lifecycle") or ""),
                "database_path": _status_database_path(status),
            },
        )
    if env_uri and env_store_matches:
        return QuackEndpointDiscovery(
            uri=env_uri,
            token=str(os.environ.get(QUACK_TOKEN_ENV, "") or "").strip(),
            source="env",
            reason="env_endpoint_store_bound",
            details={"store_id": store_id},
        )
    if env_uri:
        rejections.append(f"{QUACK_ENDPOINT_ENV}:unbound_to_database")
    reason = "no_live_owner"
    if rejections:
        reason = "owner_status_rejected"
    return QuackEndpointDiscovery(
        reason=reason,
        details={"rejections": "; ".join(rejections) if rejections else "status_missing"},
    )


def _format_quack_prefer_log(
    *,
    action: str,
    database: Path | str,
    discovery: QuackEndpointDiscovery,
    extra: Mapping[str, Any] | None = None,
) -> str:
    parts = [
        "agent-supervisor quack prefer:",
        action,
        f"database={database}",
        f"reason={discovery.reason}",
        f"source={discovery.source}",
    ]
    if discovery.uri:
        parts.append(f"uri={discovery.uri}")
    if discovery.status_path:
        parts.append(f"status_path={discovery.status_path}")
    if discovery.token:
        parts.append("token_present=true")
    details = dict(discovery.details)
    if extra:
        details.update({str(key): extra[key] for key in extra})
    for key, value in details.items():
        if value in (None, "", (), [], {}):
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts)


def _open_file_duckdb_connection(
    path: Path | str,
    *,
    timeout_seconds: float,
    memory_limit: str,
    threads: int,
) -> DuckDBConnection:
    connection = DuckDBConnection(
        path,
        timeout_seconds=timeout_seconds,
        memory_limit=memory_limit,
        threads=threads,
    )
    connection._transport_mode = "file"
    return connection


def quack_owner_mutation_dir(store_id: object = "") -> Path | None:
    """Return the exclusive owner's local mutation inbox, if configured."""

    explicit = str(
        os.environ.get(QUACK_MUTATION_DIR_ENV, "") or ""
    ).strip()
    if explicit:
        return Path(explicit)
    store = str(
        store_id
        or os.environ.get("IPFS_ACCELERATE_AGENT_STATE_STORE_ID", "")
        or ""
    ).strip()
    if not store:
        return None
    path = Path(store)
    if path.suffix.lower() in {".duckdb", ".ddb"}:
        return path.expanduser().resolve().parent / "quack-owner" / "mutations"
    return None


def _empty_duckdb_cursor(*, rowcount: int = -1) -> DuckDBCursor:
    cursor = DuckDBCursor.__new__(DuckDBCursor)
    cursor._columns = ()
    cursor._rows = []
    cursor._offset = 0
    cursor.rowcount = int(rowcount)
    return cursor


def _quack_mutation_binding_from_environment() -> Mapping[str, Any] | None:
    raw = str(os.environ.get(QUACK_MUTATION_BINDING_ENV, "") or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return validate_mutation_binding(payload)
    except (TypeError, ValueError, json.JSONDecodeError, QuackOwnerMutationError) as exc:
        raise DuckDBConnectionPolicyError(
            "quack mutation binding environment is invalid"
        ) from exc


def _consume_duckdb_result(connection: Any) -> None:
    try:
        connection.fetchall()
    except Exception:
        pass


def open_quack_transport_connection(
    uri: str,
    *,
    token: str = "",
) -> DuckDBConnection:
    """Attach read-only to the state owner's verified Quack replica.

    Canonical mutations never traverse the Quack SQL surface. They are
    buffered into closed protocol-2 bundles and executed by the exclusive
    DuckDB writer after exact owner/generation authentication.
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
    # Quack must already be present in the reviewed local extension cache.
    # Disable DuckDB's implicit installer and autoloader before ``LOAD`` so a
    # missing client extension is a typed launch failure, never a download.
    extension_directory = str(
        os.environ.get(CONFIGURED_BOARD_EXTENSION_DIRECTORY_ENV, "") or ""
    ).strip()
    connection_config = {
        "autoinstall_known_extensions": "false",
        "autoload_known_extensions": "false",
        "allow_unsigned_extensions": "false",
    }
    if extension_directory:
        extension_path = Path(extension_directory)
        try:
            extension_path = extension_path.resolve(strict=True)
        except OSError as exc:
            raise DuckDBConnectionPolicyError(
                "configured-board extension projection is unavailable"
            ) from exc
        if (
            not extension_path.is_dir()
            or extension_path.name != "extensions"
            or extension_path.parent.name != ".duckdb"
        ):
            raise DuckDBConnectionPolicyError(
                "configured-board extension projection is invalid"
            )
        connection_config["extension_directory"] = str(extension_path)
    connection = duckdb.connect(
        ":memory:",
        config=connection_config,
    )
    try:
        connection.execute("LOAD quack")
        attach = f"ATTACH '{text}' AS {_QUACK_CONTROL_CATALOG} (READ_ONLY"
        secret = str(
            token or os.environ.get("IPFS_ACCELERATE_AGENT_QUACK_TOKEN", "") or ""
        ).strip()
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
    except Exception:
        try:
            connection.close()
        except Exception:
            pass
        raise
    wrapped = DuckDBConnection.wrap(connection)
    wrapped._default_catalog = _QUACK_CONTROL_CATALOG
    wrapped._quack_uri = text
    wrapped._quack_mutation_token = secret
    wrapped._quack_mutation_binding = _quack_mutation_binding_from_environment()
    wrapped._quack_mutation_inbox = quack_owner_mutation_dir(
        wrapped._quack_mutation_binding.get("store_id")
        if isinstance(wrapped._quack_mutation_binding, Mapping)
        else ""
    )
    if isinstance(wrapped._quack_mutation_binding, Mapping):
        if wrapped._quack_mutation_binding.get("listen_uri") != text:
            wrapped.close()
            raise DuckDBConnectionPolicyError(
                "quack mutation binding does not match attached endpoint"
            )
    return wrapped


def open_duckdb_connection(
    path: Path | str,
    *,
    timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    threads: int = 1,
    prefer_quack: bool | None = None,
) -> DuckDBConnection:
    if is_quack_transport_target(path):
        connection = open_quack_transport_connection(path)
        connection._transport_mode = "quack"
        return connection
    if prefer_quack is None:
        prefer_quack = _env_flag(QUACK_PREFER_ENV, default=True)
    if not prefer_quack:
        return _open_file_duckdb_connection(
            path,
            timeout_seconds=timeout_seconds,
            memory_limit=memory_limit,
            threads=threads,
        )
    discovery = discover_live_quack_endpoint(path)
    require = _env_flag(QUACK_REQUIRE_ENV, default=False)
    if discovery.found:
        try:
            connection = open_quack_transport_connection(
                discovery.uri, token=discovery.token
            )
        except Exception as exc:
            message = _format_quack_prefer_log(
                action="attach_failed",
                database=path,
                discovery=discovery,
                extra={
                    "error_type": type(exc).__name__,
                    "error": exc,
                },
            )
            if require:
                _LOGGER.error("%s; file fallback disabled by %s", message, QUACK_REQUIRE_ENV)
                raise DuckDBConnectionPolicyError(message) from exc
            fallback_timeout = min(
                float(timeout_seconds),
                QUACK_LIVE_OWNER_FILE_FALLBACK_TIMEOUT_SECONDS,
            )
            _LOGGER.warning(
                "%s; falling back to exclusive DuckDB file lock_timeout=%s",
                message,
                fallback_timeout,
            )
            try:
                return _open_file_duckdb_connection(
                    path,
                    timeout_seconds=fallback_timeout,
                    memory_limit=memory_limit,
                    threads=threads,
                )
            except Exception as fallback_exc:
                _LOGGER.error(
                    "%s; file fallback also failed error_type=%s error=%s",
                    message,
                    type(fallback_exc).__name__,
                    fallback_exc,
                )
                raise DuckDBConnectionPolicyError(
                    message + f"; file fallback failed: {fallback_exc}"
                ) from exc
        connection._transport_mode = "quack"
        _LOGGER.info(
            _format_quack_prefer_log(
                action="attached",
                database=path,
                discovery=discovery,
            )
        )
        return connection
    message = _format_quack_prefer_log(
        action="no_live_owner",
        database=path,
        discovery=discovery,
    )
    if require:
        _LOGGER.error("%s; file fallback disabled by %s", message, QUACK_REQUIRE_ENV)
        raise DuckDBConnectionPolicyError(message)
    if discovery.reason == "owner_status_rejected":
        _LOGGER.warning("%s; falling back to exclusive DuckDB file", message)
    else:
        _LOGGER.info("%s; falling back to exclusive DuckDB file", message)
    return _open_file_duckdb_connection(
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
