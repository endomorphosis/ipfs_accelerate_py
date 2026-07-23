"""Shared DuckDB primitives for durable agent-supervisor state.

DuckDB permits only one external writer process. Supervisor stores therefore
use short-lived connections protected by a process-shared file lock. Legacy
SQLite databases are copied table-by-table into the new DuckDB file and are
left untouched as rollback evidence.
"""

from __future__ import annotations

import fcntl
import os
import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0
DEFAULT_MEMORY_LIMIT = "256MB"
SQLITE_MAGIC = b"SQLite format 3\0"

_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class DuckDBRow(Mapping[str, Any]):
    """Small ``sqlite3.Row``-compatible view over a DuckDB result row."""

    def __init__(self, columns: Iterable[str], values: Iterable[Any]) -> None:
        self._columns = tuple(str(column) for column in columns)
        self._values = tuple(values)
        self._positions = {
            column: index for index, column in enumerate(self._columns)
        }

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
            and
            len(self._columns) == 1
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
        rows = [
            DuckDBRow(self._columns, values)
            for values in self._rows[self._offset :]
        ]
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
                    )
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
    if path is None:
        import tempfile

        root = Path(tempfile.mkdtemp(prefix=temporary_prefix))
        return root / default_filename, None

    supplied = Path(path)
    suffix = supplied.suffix.lower()
    if suffix in {".sqlite", ".sqlite3", ".db"}:
        target = supplied.with_suffix(".duckdb")
        legacy = supplied if is_sqlite_database(supplied) else None
        return target, legacy
    if suffix == ".duckdb":
        legacy_candidate = supplied.with_suffix(".sqlite3")
        return supplied, (
            legacy_candidate if is_sqlite_database(legacy_candidate) else None
        )
    target = supplied / default_filename
    legacy_candidate = supplied / legacy_filename
    return target, (
        legacy_candidate if is_sqlite_database(legacy_candidate) else None
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
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if is_sqlite_database(self.path):
            raise ValueError(
                f"legacy SQLite database must be migrated before opening: {self.path}"
            )
        self._transaction_active = False
        self._transaction_on_context = bool(transaction_on_context)
        self._context_depth = 0
        self._closed = False
        self._lock_context = exclusive_file_lock(
            self.path.with_name(f".{self.path.name}.lock"),
            timeout_seconds=timeout_seconds,
        )
        self._lock_context.__enter__()
        try:
            import duckdb

            self._connection = duckdb.connect(str(self.path))
            self._connection.execute(f"SET threads={max(1, int(threads))}")
            self._connection.execute(
                "SET memory_limit=?",
                [str(memory_limit)],
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


def open_duckdb_connection(
    path: Path | str,
    *,
    timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    threads: int = 1,
) -> DuckDBConnection:
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
    value_transform: (
        Callable[[str, str, Any], Any] | None
    ) = None,
) -> None:
    """Initialize a store and idempotently import a legacy SQLite database."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    legacy = Path(legacy_sqlite_path) if legacy_sqlite_path is not None else None
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
        migration_key = (
            f"sqlite_migration:{legacy.resolve()}" if legacy is not None else ""
        )
        migrated = False
        if migration_key:
            migrated = connection.execute(
                "SELECT 1 FROM agent_supervisor_store_metadata WHERE key=?",
                (migration_key,),
            ).fetchone() is not None
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
                        for row in source.execute(
                            f'PRAGMA table_info("{table_name}")'
                        ).fetchall()
                    ]
                    if not columns:
                        continue
                    quoted_columns = ", ".join(f'"{column}"' for column in columns)
                    placeholders = ", ".join("?" for _ in columns)
                    insert_sql = (
                        f'INSERT INTO "{table_name}" ({quoted_columns}) '
                        f"VALUES ({placeholders}) ON CONFLICT DO NOTHING"
                    )
                    cursor = source.execute(
                        f'SELECT {quoted_columns} FROM "{table_name}"'
                    )
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
