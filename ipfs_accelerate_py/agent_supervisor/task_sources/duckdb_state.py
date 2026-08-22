"""Shared DuckDB primitives for durable agent-supervisor state.

DuckDB permits only one external writer process. Supervisor stores therefore
use short-lived connections protected by a process-shared file lock. Legacy
SQLite databases are copied table-by-table into the new DuckDB file and are
left untouched as rollback evidence unless strict DuckDB-only mode is enabled.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import sqlite3
import stat
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
        self._pooled = False
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
        instance._pooled = False
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
        if catalog and _quack_owner_mutation_required(normalized):
            return _execute_quack_owner_mutation(
                statement,
                parameters,
                dml=True,
            )
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
        self._connection.executemany(sql, parameters)
        return DuckDBCursor(self._connection, dml=True)

    def executescript(self, sql: str) -> DuckDBCursor:
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
        if getattr(self, "_pooled", False):
            if self._transaction_active:
                try:
                    self.rollback()
                except Exception:
                    pass
            return
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
_QUACK_CONTROL_CATALOG = "control_plane"
_QUACK_OWNER_DML_PREFIXES = (
    "UPDATE ",
    "DELETE ",
    "MERGE ",
    "INSERT OR REPLACE",
    "INSERT OR IGNORE",
)
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


def quack_owner_mutation_dir(store_id: object = "") -> Path | None:
    """Return the exclusive owner's local mutation inbox, if configured."""

    explicit = str(
        os.environ.get("IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR", "") or ""
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
    return {
        "unstalled": unstalled,
        "skipped": skipped,
        "stale_seconds": int(stale_seconds),
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
    parameters: Iterable[Any] | Mapping[str, Any] | None,
    *,
    dml: bool,
) -> DuckDBCursor:
    """Apply UPDATE/DELETE on the exclusive owner connection.

    This Quack ATTACH build can SELECT/INSERT new rows but cannot UPDATE or
    DELETE attached base tables. Mutations stay on the state-owner that
    already holds the exclusive file connection.
    """

    import json
    import uuid

    target = quack_owner_mutation_dir()
    if target is None:
        raise DuckDBConnectionPolicyError(
            "quack ATTACH cannot UPDATE/DELETE remote base tables; set "
            "IPFS_ACCELERATE_AGENT_QUACK_MUTATION_DIR or "
            "IPFS_ACCELERATE_AGENT_STATE_STORE_ID so the state-owner can "
            "apply the mutation"
        )
    target.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex
    request_path = target / f"{request_id}.request.json"
    done_path = target / f"{request_id}.done.json"
    if parameters is None:
        bound: Any = None
    elif isinstance(parameters, Mapping):
        bound = dict(parameters)
    else:
        bound = list(parameters)
    request_path.write_text(
        json.dumps({"sql": statement, "parameters": bound}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if done_path.is_file():
            payload = json.loads(done_path.read_text(encoding="utf-8"))
            try:
                request_path.unlink(missing_ok=True)
                done_path.unlink(missing_ok=True)
            except OSError:
                pass
            if payload.get("ok") is not True:
                raise DuckDBConnectionPolicyError(
                    "quack owner mutation failed: "
                    + str(payload.get("error") or "unknown")
                )
            cursor = DuckDBCursor.__new__(DuckDBCursor)
            cursor._columns = ()
            cursor._rows = []
            cursor._offset = 0
            cursor.rowcount = int(payload.get("rowcount") or -1)
            return cursor
        time.sleep(0.05)
    raise DuckDBConnectionPolicyError(
        "timed out waiting for quack state-owner to apply mutation"
    )


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


def resolve_quack_attach_token(token: str = "") -> str:
    """Resolve the current owner attach token.

    The vault file is preferred over a process environment value so a
    restarted owner generation is not blocked by a stale supervisor env.
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
                connection._pooled = False
                connection.close()
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
    except Exception:
        try:
            connection.close()
        except Exception:
            pass
        raise
    return connection


def open_quack_transport_connection(
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

    with _QUACK_ATTACH_LOCK:
        cached = _QUACK_TRANSPORT_CACHE.get(text)
        if cached is not None and not getattr(cached, "_closed", False):
            try:
                _probe_quack_connection(cached._connection)
                return cached
            except Exception:
                _QUACK_TRANSPORT_CACHE.pop(text, None)
                try:
                    cached._pooled = False
                    cached.close()
                except Exception:
                    pass

        last_error: BaseException | None = None
        for attempt in range(QUACK_ATTACH_ATTEMPTS):
            secret = resolve_quack_attach_token(token)
            try:
                raw = _attach_quack_once(text, secret)
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
            wrapped._pooled = True
            _QUACK_TRANSPORT_CACHE[text] = wrapped
            return wrapped

        if last_error is not None and quack_attach_error_is_contention(last_error):
            raise QuackTransportContentionError(
                "quack control-plane attach contended: " + str(last_error)
            ) from last_error
        if last_error is not None:
            raise last_error
        raise DuckDBConnectionPolicyError("quack control-plane attach failed")


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
