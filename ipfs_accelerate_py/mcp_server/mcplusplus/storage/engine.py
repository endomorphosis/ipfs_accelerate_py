"""DuckDB / Quack / DuckLake-first SQL engine for MCP++ persistence.

MCP++ single-authority state, durable journals, and Profile H execution
records use this helper. The default engine is DuckDB. Quack and DuckLake
are loaded when already present locally (``LOAD`` only — never ``INSTALL``
from the network). SQLite is a fallback when DuckDB cannot be imported or
when ``MCPPLUSPLUS_SQL_ENGINE=sqlite`` is set.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

DEFAULT_ENGINE = "duckdb"
SUPPORTED_ENGINES = frozenset({"duckdb", "sqlite"})
QUACK_EXTENSION = "quack"
DUCKLAKE_EXTENSION = "ducklake"
PRIMARY_EXTENSIONS = (QUACK_EXTENSION, DUCKLAKE_EXTENSION)
ENGINE_ENV = "MCPPLUSPLUS_SQL_ENGINE"
DUCKLAKE_CATALOG_ENV = "MCPPLUSPLUS_DUCKLAKE_CATALOG"

PathLike = Union[str, os.PathLike[str]]


class EngineError(RuntimeError):
    """Raised for engine open / statement failures."""

    code = "mcpplusplus_sql_engine_error"


class NamedRow:
    """sqlite3.Row-compatible mapping over a result tuple."""

    def __init__(self, columns: Sequence[str], values: Sequence[Any]) -> None:
        self._columns = tuple(str(col) for col in columns)
        self._values = tuple(values)
        self._map = {
            column: value for column, value in zip(self._columns, self._values)
        }

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return self._map[key]

    def keys(self) -> tuple[str, ...]:
        return self._columns


class EngineCursor:
    """Minimal cursor: ``fetchone`` / ``fetchall`` with named rows."""

    def __init__(
        self,
        rows: Sequence[Sequence[Any]],
        columns: Sequence[str],
    ) -> None:
        self._rows = [NamedRow(columns, row) for row in rows]
        self._index = 0
        self.description = tuple((name, None, None, None, None, None, None) for name in columns)

    def fetchone(self) -> Optional[NamedRow]:
        if self._index >= len(self._rows):
            return None
        row = self._rows[self._index]
        self._index += 1
        return row

    def fetchall(self) -> list[NamedRow]:
        remaining = list(self._rows[self._index :])
        self._index = len(self._rows)
        return remaining


class EngineConnection:
    """Long-lived SQL connection used by MCP++ durable/state stores."""

    def __init__(
        self,
        *,
        engine: str,
        path: Path,
        raw: Any,
        loaded: tuple[str, ...],
        ducklake_catalog: Optional[str] = None,
    ) -> None:
        self.engine = engine
        self.path = path
        self._raw = raw
        self.loaded_extensions = loaded
        self.ducklake_catalog = ducklake_catalog
        self._lock = threading.RLock()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> EngineCursor:
        statement = _normalize_sql(_adapt_schema_sql(sql, self.engine), self.engine)
        bound: Optional[list[Any]]
        if params is None:
            bound = None
        elif isinstance(params, (str, bytes)):
            bound = [params]
        else:
            bound = list(params)
        with self._lock:
            try:
                if self.engine == "duckdb":
                    result = (
                        self._raw.execute(statement, bound)
                        if bound is not None
                        else self._raw.execute(statement)
                    )
                    columns = _duckdb_columns(result)
                    rows = result.fetchall() if result is not None else []
                    return EngineCursor(rows, columns)
                cursor = (
                    self._raw.execute(statement, tuple(bound))
                    if bound is not None
                    else self._raw.execute(statement)
                )
                columns = [str(item[0]) for item in (cursor.description or ())]
                rows = cursor.fetchall()
                return EngineCursor(rows, columns)
            except EngineError:
                raise
            except Exception as exc:
                raise EngineError(f"{self.engine} execute failed: {exc}") from exc

    def executescript(self, script: str) -> EngineCursor:
        last = EngineCursor((), ())
        adapted = _adapt_schema_sql(script, self.engine)
        for statement in _split_sql_script(adapted):
            last = self.execute(statement)
        return last

    def commit(self) -> None:
        with self._lock:
            if self.engine == "sqlite":
                self._raw.commit()
                return
            try:
                self._raw.commit()
            except Exception:
                self.execute("COMMIT")

    def rollback(self) -> None:
        with self._lock:
            if self.engine == "sqlite":
                self._raw.rollback()
                return
            try:
                self._raw.rollback()
            except Exception:
                self.execute("ROLLBACK")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                if self.engine == "duckdb":
                    try:
                        self._raw.execute("CHECKPOINT")
                    except Exception:
                        pass
                else:
                    try:
                        self._raw.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    except Exception:
                        pass
                self._raw.close()
            except Exception:
                pass

    def __enter__(self) -> "EngineConnection":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def resolve_sql_engine(requested: Optional[str] = None) -> str:
    """Return ``duckdb`` (default) or ``sqlite`` fallback."""

    raw = (requested or os.environ.get(ENGINE_ENV) or DEFAULT_ENGINE).strip().lower()
    if raw in {"duck", "duckdb", "quack", "ducklake", "duckdb-quack"}:
        return "duckdb"
    if raw in {"sqlite", "sqlite3"}:
        return "sqlite"
    raise EngineError(f"unknown MCP++ SQL engine {raw!r}")


def connect_sql_engine(
    path: PathLike,
    *,
    engine: Optional[str] = None,
    load_extensions: bool = True,
) -> EngineConnection:
    """Open the primary MCP++ SQL store (DuckDB unless fallback is required)."""

    target = Path(path)
    if target.parent and str(target.parent) not in {"", "."}:
        target.parent.mkdir(parents=True, exist_ok=True)
    chosen = resolve_sql_engine(engine)
    if chosen == "duckdb":
        try:
            return _open_duckdb(target, load_extensions=load_extensions)
        except EngineError:
            raise
        except Exception as exc:
            if engine is None and os.environ.get(ENGINE_ENV, "").strip() == "":
                return _open_sqlite(target)
            raise EngineError(f"DuckDB open failed for {target}: {exc}") from exc
    return _open_sqlite(target)


def loaded_extensions(connection: EngineConnection) -> tuple[str, ...]:
    return connection.loaded_extensions


def _open_duckdb(path: Path, *, load_extensions: bool) -> EngineConnection:
    try:
        import duckdb
    except ImportError as exc:
        raise EngineError("duckdb is not installed") from exc

    raw = duckdb.connect(str(path))
    loaded: list[str] = []
    catalog: Optional[str] = None
    try:
        if load_extensions:
            loaded.extend(_load_local_extensions(raw))
            catalog = _attach_optional_ducklake(raw)
        return EngineConnection(
            engine="duckdb",
            path=path,
            raw=raw,
            loaded=tuple(loaded),
            ducklake_catalog=catalog,
        )
    except Exception:
        raw.close()
        raise


def _open_sqlite(path: Path) -> EngineConnection:
    raw = sqlite3.connect(
        str(path),
        timeout=30,
        check_same_thread=False,
        isolation_level=None,
    )
    raw.row_factory = sqlite3.Row
    try:
        raw.execute("PRAGMA journal_mode=WAL")
        raw.execute("PRAGMA synchronous=FULL")
        raw.execute("PRAGMA foreign_keys=ON")
        raw.execute("PRAGMA busy_timeout=30000")
    except sqlite3.Error:
        pass
    return EngineConnection(
        engine="sqlite",
        path=path,
        raw=raw,
        loaded=(),
    )


def _load_local_extensions(raw: Any) -> list[str]:
    """``LOAD`` Quack then DuckLake when already installed. Never ``INSTALL``."""

    loaded: list[str] = []
    for name in PRIMARY_EXTENSIONS:
        try:
            raw.execute(f"LOAD {name}")
        except Exception:
            continue
        loaded.append(name)
    return loaded


def _attach_optional_ducklake(raw: Any) -> Optional[str]:
    catalog = os.environ.get(DUCKLAKE_CATALOG_ENV, "").strip()
    if not catalog:
        return None
    try:
        raw.execute(
            f"ATTACH 'ducklake:{catalog}' AS mcpp_ducklake (READ_ONLY)"
        )
    except Exception:
        return None
    return catalog


def _duckdb_columns(result: Any) -> tuple[str, ...]:
    description = getattr(result, "description", None)
    if not description:
        return ()
    return tuple(str(item[0]) for item in description)


def _adapt_schema_sql(script: str, engine: str) -> str:
    """Adapt SQLite DDL so DuckDB can host the same tables.

    DuckDB INTEGER is INT32 (epoch-ms needs BIGINT). DuckDB foreign keys
    reject ``INSERT OR REPLACE`` on parent rows still referenced by children,
    so declared FKs are dropped; the journal/state stores already enforce
    parent/child linkage in Python.
    """

    if engine != "duckdb":
        return script
    text = script.replace("INTEGER", "BIGINT")
    cleaned: list[str] = []
    for line in text.splitlines():
        if line.strip().upper().startswith("FOREIGN KEY"):
            if cleaned:
                cleaned[-1] = cleaned[-1].rstrip().rstrip(",")
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


_UPSERT_CONFLICT_TARGETS = {
    "metadata": "key",
    "method_idempotency": "execution_id, scope, idempotency_key",
    "start_idempotency": "idempotency_key",
    "write_ops": "operation_id",
}


def _rewrite_insert_or_replace(sql: str) -> str:
    import re

    match = re.match(
        r"INSERT\s+OR\s+REPLACE\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\((.+)\)\s*$",
        sql.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return sql
    table, columns_raw, values_raw = match.group(1), match.group(2), match.group(3)
    target = _UPSERT_CONFLICT_TARGETS.get(table.lower())
    if not target:
        return sql
    columns = [part.strip() for part in columns_raw.split(",") if part.strip()]
    conflict = {part.strip() for part in target.split(",")}
    assignments = [
        f"{column} = excluded.{column}"
        for column in columns
        if column not in conflict
    ]
    if not assignments:
        return (
            f"INSERT INTO {table} ({columns_raw}) VALUES ({values_raw}) "
            f"ON CONFLICT ({target}) DO NOTHING"
        )
    return (
        f"INSERT INTO {table} ({columns_raw}) VALUES ({values_raw}) "
        f"ON CONFLICT ({target}) DO UPDATE SET {', '.join(assignments)}"
    )


def _normalize_sql(sql: str, engine: str) -> str:
    text = sql.strip()
    if engine != "duckdb":
        return text
    upper = text.upper()
    if upper.startswith("INSERT OR REPLACE"):
        return _rewrite_insert_or_replace(text)
    if upper.startswith("BEGIN IMMEDIATE"):
        return "BEGIN TRANSACTION"
    if upper.startswith("PRAGMA JOURNAL_MODE") or upper.startswith(
        "PRAGMA WAL_CHECKPOINT"
    ):
        return "SELECT 'duckdb' AS journal_mode"
    if upper.startswith("PRAGMA "):
        return "SELECT 1"
    return text


def _split_sql_script(script: str) -> Iterable[str]:
    statement: list[str] = []
    in_single = False
    for char in script:
        if char == "'" and not in_single:
            in_single = True
            statement.append(char)
            continue
        if char == "'" and in_single:
            in_single = False
            statement.append(char)
            continue
        if char == ";" and not in_single:
            piece = "".join(statement).strip()
            if piece:
                yield piece
            statement = []
            continue
        statement.append(char)
    piece = "".join(statement).strip()
    if piece:
        yield piece
