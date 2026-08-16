"""Checksum-bound control-plane migration catalog and transactional runner.

One catalog owns ordered schema creation for ``control.duckdb``. Migrations are
versioned SQL units with content checksums, optional application/tool version
bounds, and pre/postconditions. The runner records every apply attempt as a
receipt (version, checksum, application/tool versions, start/end/outcome,
schema fingerprint) and refuses drift, gaps, downgrades, partial application,
and runtime ad-hoc DDL outside an explicit compatibility path.

Schema-domain SQL files land under ``task_sources/sql/`` in later tasks; this
module only installs the migration bookkeeping tables and applies whatever the
catalog supplies.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Final

from .duckdb_state import DuckDBConnection, exclusive_file_lock, open_duckdb_connection
from .task_identity import canonical_content_cid


CONTROL_PLANE_MIGRATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-migration@1"
)
MIGRATION_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-migration-receipt@1"
)
MIGRATION_CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-migration-catalog@1"
)
MIGRATION_RUN_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/control-plane-migration-run@1"
)

SQL_DIRECTORY_NAME: Final = "sql"
SQL_FILENAME_RE: Final = re.compile(
    r"^(?P<version>\d{4})_(?P<slug>[a-z0-9]+(?:_[a-z0-9]+)*)\.sql$"
)

# Outcomes recorded on MigrationReceipt.
OUTCOME_APPLIED: Final = "applied"
OUTCOME_REPLAYED: Final = "replayed"
OUTCOME_FAILED: Final = "failed"
OUTCOME_REFUSED: Final = "refused"

# Ownership lease for single-owner migration application.
OWNERSHIP_KEY: Final = "migration_owner"
OWNERSHIP_LEASE_SECONDS: Final = 120.0

# Metadata keys stored in control_plane_metadata.
META_SCHEMA_VERSION: Final = "schema_version"
META_SCHEMA_FINGERPRINT: Final = "schema_fingerprint"
META_CATALOG_FINGERPRINT: Final = "catalog_fingerprint"
META_APPLICATION_VERSION: Final = "application_version"
META_TOOL_VERSION: Final = "tool_version"
META_DATABASE_UUID: Final = "database_uuid"

# Statements that count as DDL for the ad-hoc guard.
_DDL_PREFIXES: Final = (
    "CREATE ",
    "ALTER ",
    "DROP ",
    "ATTACH ",
    "DETACH ",
    "COPY ",
    "EXPORT ",
    "IMPORT ",
    "INSTALL ",
    "LOAD ",
    "PRAGMA ",
    "SET ",
    "CALL ",
    "VACUUM ",
    "CHECKPOINT ",
    "FORCE ",
    "TRUNCATE ",
)

_BOOKKEEPING_SQL: Final = """
CREATE TABLE IF NOT EXISTS control_plane_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    migration_id VARCHAR NOT NULL UNIQUE,
    checksum VARCHAR NOT NULL,
    application_version VARCHAR NOT NULL,
    tool_version VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    schema_fingerprint VARCHAR NOT NULL,
    receipt_cid VARCHAR NOT NULL UNIQUE,
    body_json VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS schema_migration_attempts (
    attempt_id VARCHAR PRIMARY KEY,
    version INTEGER NOT NULL,
    migration_id VARCHAR NOT NULL,
    checksum VARCHAR NOT NULL,
    application_version VARCHAR NOT NULL,
    tool_version VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    finished_at VARCHAR,
    outcome VARCHAR NOT NULL,
    schema_fingerprint VARCHAR,
    error_text VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL
);
"""


class ControlPlaneMigrationError(RuntimeError):
    """Base class for fail-closed migration errors."""


class MigrationCatalogError(ControlPlaneMigrationError, ValueError):
    """The catalog is malformed, gapped, or contains duplicate identities."""


class MigrationDriftError(ControlPlaneMigrationError):
    """An applied migration no longer matches the catalog checksum."""


class MigrationGapError(ControlPlaneMigrationError):
    """Applied or requested versions skip a required intermediate version."""


class MigrationDowngradeError(ControlPlaneMigrationError):
    """A caller requested a schema version lower than the applied head."""


class MigrationPartialError(ControlPlaneMigrationError):
    """A prior migration attempt left the database in a partial state."""


class MigrationOwnershipError(ControlPlaneMigrationError):
    """Another owner holds the exclusive migration lease."""


class MigrationAdHocDDLError(ControlPlaneMigrationError):
    """Runtime DDL was attempted outside the migration runner path."""


class MigrationPreconditionError(ControlPlaneMigrationError):
    """A migration precondition failed before SQL was applied."""


class MigrationPostconditionError(ControlPlaneMigrationError):
    """A migration postcondition failed after SQL was applied."""


class DuckDBUnavailableError(ControlPlaneMigrationError):
    """DuckDB is required for hermetic migration tests/runs but is missing."""


def duckdb_available() -> bool:
    """Return whether the optional duckdb package can be imported."""

    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _require_duckdb() -> Any:
    try:
        import duckdb  # type: ignore
    except ImportError as exc:
        raise DuckDBUnavailableError(
            "DuckDB is required for control-plane migrations; install the "
            "optional duckdb dependency"
        ) from exc
    return duckdb


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utc_iso(value: datetime | None = None) -> str:
    moment = value or _utc_now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def checksum_sql(sql_text: str) -> str:
    """Return a stable sha256 checksum for migration SQL text."""

    normalized = sql_text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.endswith("\n") and normalized:
        normalized = normalized + "\n"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _package_sql_directory() -> Path:
    return Path(__file__).resolve().parent / SQL_DIRECTORY_NAME


def default_application_version() -> str:
    """Best-effort application version for receipt attestation."""

    try:
        from importlib import metadata

        return metadata.version("ipfs_accelerate_py")
    except Exception:
        return "unknown"


def default_tool_version() -> str:
    """Best-effort DuckDB tool version for receipt attestation."""

    if not duckdb_available():
        return "unavailable"
    duckdb = _require_duckdb()
    version = getattr(duckdb, "__version__", None)
    if version:
        return str(version)
    try:
        connection = duckdb.connect(":memory:")
        try:
            row = connection.execute("SELECT version()").fetchone()
            return str(row[0]) if row else "unknown"
        finally:
            connection.close()
    except Exception:
        return "unknown"


def _is_ddl_statement(sql: str) -> bool:
    stripped = sql.strip()
    if not stripped or stripped.startswith("--") or stripped.startswith("/*"):
        # Strip leading comments for classification.
        lines = []
        in_block = False
        for line in stripped.splitlines():
            text = line.strip()
            if in_block:
                if "*/" in text:
                    in_block = False
                continue
            if text.startswith("/*"):
                if "*/" not in text:
                    in_block = True
                continue
            if text.startswith("--") or not text:
                continue
            lines.append(text)
        stripped = " ".join(lines)
    if not stripped:
        return False
    upper = " ".join(stripped.upper().split())
    return any(upper.startswith(prefix) for prefix in _DDL_PREFIXES)


def _split_sql_statements(sql_text: str) -> list[str]:
    """Split SQL text on semicolons outside quotes/comments."""

    statements: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(sql_text)
    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False
    while i < n:
        ch = sql_text[i]
        nxt = sql_text[i + 1] if i + 1 < n else ""
        if in_line_comment:
            buf.append(ch)
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            buf.append(ch)
            if ch == "*" and nxt == "/":
                buf.append(nxt)
                i += 2
                in_block_comment = False
                continue
            i += 1
            continue
        if not in_single and not in_double:
            if ch == "-" and nxt == "-":
                buf.append(ch)
                buf.append(nxt)
                i += 2
                in_line_comment = True
                continue
            if ch == "/" and nxt == "*":
                buf.append(ch)
                buf.append(nxt)
                i += 2
                in_block_comment = True
                continue
            if ch == ";":
                statement = "".join(buf).strip()
                if statement:
                    statements.append(statement)
                buf = []
                i += 1
                continue
        if ch == "'" and not in_double:
            # Handle escaped single quotes.
            if in_single and nxt == "'":
                buf.append(ch)
                buf.append(nxt)
                i += 2
                continue
            in_single = not in_single
            buf.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            i += 1
            continue
        buf.append(ch)
        i += 1
    trailing = "".join(buf).strip()
    if trailing:
        statements.append(trailing)
    return statements


@dataclass(frozen=True)
class ControlPlaneMigration:
    """One ordered, checksum-bound schema migration unit."""

    version: int
    migration_id: str
    sql_text: str
    checksum: str
    description: str = ""
    depends_on: tuple[int, ...] = ()
    min_application_version: str | None = None
    max_application_version: str | None = None
    min_tool_version: str | None = None
    max_tool_version: str | None = None
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    source_path: str | None = None
    schema: str = CONTROL_PLANE_MIGRATION_SCHEMA

    def __post_init__(self) -> None:
        if int(self.version) < 1:
            raise MigrationCatalogError("migration version must be >= 1")
        if not str(self.migration_id).strip():
            raise MigrationCatalogError("migration_id is required")
        if not str(self.sql_text).strip():
            raise MigrationCatalogError(
                f"migration {self.migration_id} has empty SQL"
            )
        expected = checksum_sql(self.sql_text)
        if self.checksum != expected:
            raise MigrationCatalogError(
                f"migration {self.migration_id} checksum mismatch: "
                f"declared {self.checksum}, computed {expected}"
            )
        for dep in self.depends_on:
            if int(dep) >= int(self.version):
                raise MigrationCatalogError(
                    f"migration {self.migration_id} depends_on must be "
                    f"strictly lower than its version"
                )

    @classmethod
    def from_sql(
        cls,
        *,
        version: int,
        migration_id: str,
        sql_text: str,
        description: str = "",
        depends_on: Sequence[int] = (),
        min_application_version: str | None = None,
        max_application_version: str | None = None,
        min_tool_version: str | None = None,
        max_tool_version: str | None = None,
        preconditions: Sequence[str] = (),
        postconditions: Sequence[str] = (),
        source_path: str | None = None,
    ) -> ControlPlaneMigration:
        text = str(sql_text)
        return cls(
            version=int(version),
            migration_id=str(migration_id).strip(),
            sql_text=text,
            checksum=checksum_sql(text),
            description=str(description or ""),
            depends_on=tuple(int(item) for item in depends_on),
            min_application_version=min_application_version,
            max_application_version=max_application_version,
            min_tool_version=min_tool_version,
            max_tool_version=max_tool_version,
            preconditions=tuple(str(item) for item in preconditions),
            postconditions=tuple(str(item) for item in postconditions),
            source_path=source_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": int(self.version),
            "migration_id": self.migration_id,
            "checksum": self.checksum,
            "description": self.description,
            "depends_on": list(self.depends_on),
            "min_application_version": self.min_application_version,
            "max_application_version": self.max_application_version,
            "min_tool_version": self.min_tool_version,
            "max_tool_version": self.max_tool_version,
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "source_path": self.source_path,
            "sql_sha256": self.checksum,
        }


@dataclass(frozen=True)
class MigrationReceipt:
    """Durable attestation for one migration apply/replay attempt."""

    receipt_cid: str
    version: int
    migration_id: str
    checksum: str
    application_version: str
    tool_version: str
    started_at: str
    finished_at: str
    outcome: str
    schema_fingerprint: str
    error_text: str = ""
    schema: str = MIGRATION_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "receipt_cid": self.receipt_cid,
            "version": int(self.version),
            "migration_id": self.migration_id,
            "checksum": self.checksum,
            "application_version": self.application_version,
            "tool_version": self.tool_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "outcome": self.outcome,
            "schema_fingerprint": self.schema_fingerprint,
            "error_text": self.error_text,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MigrationReceipt:
        return cls(
            receipt_cid=str(payload["receipt_cid"]),
            version=int(payload["version"]),
            migration_id=str(payload["migration_id"]),
            checksum=str(payload["checksum"]),
            application_version=str(payload["application_version"]),
            tool_version=str(payload["tool_version"]),
            started_at=str(payload["started_at"]),
            finished_at=str(payload["finished_at"]),
            outcome=str(payload["outcome"]),
            schema_fingerprint=str(payload["schema_fingerprint"]),
            error_text=str(payload.get("error_text") or ""),
        )


@dataclass(frozen=True)
class MigrationRunReport:
    """Summary of a runner invocation to a target version."""

    from_version: int
    to_version: int
    receipts: tuple[MigrationReceipt, ...]
    schema_fingerprint: str
    catalog_fingerprint: str
    changed: bool
    schema: str = MIGRATION_RUN_REPORT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "from_version": int(self.from_version),
            "to_version": int(self.to_version),
            "receipts": [receipt.to_dict() for receipt in self.receipts],
            "schema_fingerprint": self.schema_fingerprint,
            "catalog_fingerprint": self.catalog_fingerprint,
            "changed": bool(self.changed),
        }


@dataclass(frozen=True)
class MigrationCatalog:
    """Ordered, gap-free, dependency-aware catalog of control-plane migrations."""

    migrations: tuple[ControlPlaneMigration, ...]
    schema: str = MIGRATION_CATALOG_SCHEMA

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not self.migrations:
            # Empty catalogs are valid: bookkeeping-only bootstrap.
            return
        versions = [int(item.version) for item in self.migrations]
        ids = [item.migration_id for item in self.migrations]
        if sorted(versions) != versions:
            raise MigrationCatalogError(
                "migrations must be supplied in strictly ascending version order"
            )
        if len(set(versions)) != len(versions):
            raise MigrationCatalogError("duplicate migration versions are refused")
        if len(set(ids)) != len(ids):
            raise MigrationCatalogError("duplicate migration_id values are refused")
        expected = list(range(1, len(versions) + 1))
        if versions != expected:
            raise MigrationCatalogError(
                f"migration versions must be contiguous from 1; got {versions}"
            )
        known = set(versions)
        for migration in self.migrations:
            for dep in migration.depends_on:
                if int(dep) not in known:
                    raise MigrationCatalogError(
                        f"migration {migration.migration_id} depends on "
                        f"missing version {dep}"
                    )

    @classmethod
    def from_migrations(
        cls,
        migrations: Sequence[ControlPlaneMigration],
    ) -> MigrationCatalog:
        ordered = tuple(
            sorted(migrations, key=lambda item: int(item.version))
        )
        return cls(migrations=ordered)

    @classmethod
    def from_sql_directory(
        cls,
        directory: Path | str | None = None,
        *,
        extra_migrations: Sequence[ControlPlaneMigration] = (),
    ) -> MigrationCatalog:
        """Load ``NNNN_slug.sql`` files from the package sql directory."""

        root = Path(directory) if directory is not None else _package_sql_directory()
        loaded: list[ControlPlaneMigration] = []
        if root.is_dir():
            for path in sorted(root.iterdir()):
                if not path.is_file() or path.suffix.lower() != ".sql":
                    continue
                if path.name.upper() == "README.SQL":
                    continue
                match = SQL_FILENAME_RE.match(path.name)
                if match is None:
                    raise MigrationCatalogError(
                        f"migration filename must match "
                        f"NNNN_slug.sql; got {path.name!r}"
                    )
                version = int(match.group("version"))
                slug = match.group("slug")
                sql_text = path.read_text(encoding="utf-8")
                loaded.append(
                    ControlPlaneMigration.from_sql(
                        version=version,
                        migration_id=f"{version:04d}_{slug}",
                        sql_text=sql_text,
                        description=slug.replace("_", " "),
                        depends_on=tuple(range(1, version)),
                        source_path=str(path),
                    )
                )
        if extra_migrations:
            loaded.extend(extra_migrations)
        return cls.from_migrations(loaded)

    @property
    def latest_version(self) -> int:
        if not self.migrations:
            return 0
        return int(self.migrations[-1].version)

    def get(self, version: int) -> ControlPlaneMigration:
        for migration in self.migrations:
            if int(migration.version) == int(version):
                return migration
        raise MigrationCatalogError(f"migration version {version} is not in catalog")

    def fingerprint(self) -> str:
        material = [
            {
                "version": int(item.version),
                "migration_id": item.migration_id,
                "checksum": item.checksum,
                "depends_on": list(item.depends_on),
                "min_application_version": item.min_application_version,
                "max_application_version": item.max_application_version,
                "min_tool_version": item.min_tool_version,
                "max_tool_version": item.max_tool_version,
                "preconditions": list(item.preconditions),
                "postconditions": list(item.postconditions),
            }
            for item in self.migrations
        ]
        return canonical_content_cid(
            {"schema": self.schema, "migrations": material}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "latest_version": self.latest_version,
            "fingerprint": self.fingerprint(),
            "migrations": [item.to_dict() for item in self.migrations],
        }

    def __iter__(self) -> Iterator[ControlPlaneMigration]:
        return iter(self.migrations)

    def __len__(self) -> int:
        return len(self.migrations)


def compute_schema_fingerprint(connection: Any) -> str:
    """Canonical information-schema fingerprint for empty-to-latest proofs."""

    rows = connection.execute(
        """
        SELECT
            table_schema,
            table_name,
            column_name,
            data_type,
            is_nullable,
            column_default,
            ordinal_position
        FROM information_schema.columns
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
        ORDER BY table_schema, table_name, ordinal_position, column_name
        """
    ).fetchall()
    columns = []
    for row in rows:
        if isinstance(row, Mapping):
            columns.append(
                {
                    "table_schema": str(row["table_schema"]),
                    "table_name": str(row["table_name"]),
                    "column_name": str(row["column_name"]),
                    "data_type": str(row["data_type"]),
                    "is_nullable": str(row["is_nullable"]),
                    "column_default": (
                        None
                        if row["column_default"] is None
                        else str(row["column_default"])
                    ),
                    "ordinal_position": int(row["ordinal_position"]),
                }
            )
        else:
            columns.append(
                {
                    "table_schema": str(row[0]),
                    "table_name": str(row[1]),
                    "column_name": str(row[2]),
                    "data_type": str(row[3]),
                    "is_nullable": str(row[4]),
                    "column_default": None if row[5] is None else str(row[5]),
                    "ordinal_position": int(row[6]),
                }
            )
    table_rows = connection.execute(
        """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
        ORDER BY table_schema, table_name
        """
    ).fetchall()
    tables = []
    for row in table_rows:
        if isinstance(row, Mapping):
            tables.append(
                {
                    "table_schema": str(row["table_schema"]),
                    "table_name": str(row["table_name"]),
                    "table_type": str(row["table_type"]),
                }
            )
        else:
            tables.append(
                {
                    "table_schema": str(row[0]),
                    "table_name": str(row[1]),
                    "table_type": str(row[2]),
                }
            )
    material = {
        "schema": "ipfs_accelerate_py/agent-supervisor/schema-fingerprint@1",
        "tables": tables,
        "columns": columns,
    }
    return canonical_content_cid(material)


def _sql_predicate_holds(connection: Any, sql: str) -> bool:
    """Return True when a predicate SQL statement yields a truthy first cell."""

    row = connection.execute(sql).fetchone()
    if row is None:
        return False
    try:
        value = row[0]
    except Exception:
        if isinstance(row, Mapping):
            value = next(iter(row.values()), None)
        else:
            return False
    return bool(value)


def _meta_get(connection: Any, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM control_plane_metadata WHERE key = ?",
        [str(key)],
    ).fetchone()
    if row is None:
        return None
    if isinstance(row, Mapping):
        return str(row["value"])
    return str(row[0])


def _meta_set(connection: Any, key: str, value: str) -> None:
    connection.execute(
        """
        INSERT INTO control_plane_metadata (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT (key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
        """,
        [str(key), str(value), _utc_iso()],
    )


def _meta_delete(connection: Any, key: str) -> None:
    connection.execute(
        "DELETE FROM control_plane_metadata WHERE key = ?",
        [str(key)],
    )


def _table_exists(connection: Any, table_name: str) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'main' AND table_name = ?
        LIMIT 1
        """,
        [table_name],
    ).fetchone()
    return row is not None


def _compare_versions(left: str, right: str) -> int | None:
    """Compare dotted numeric versions; return None if incomparable."""

    def parts(value: str) -> list[int] | None:
        text = str(value).strip()
        if not text or text in {"unknown", "unavailable"}:
            return None
        chunks = text.split(".")
        out: list[int] = []
        for chunk in chunks:
            match = re.match(r"^(\d+)", chunk)
            if match is None:
                return None
            out.append(int(match.group(1)))
        return out

    left_parts = parts(left)
    right_parts = parts(right)
    if left_parts is None or right_parts is None:
        return None
    width = max(len(left_parts), len(right_parts))
    left_parts = left_parts + [0] * (width - len(left_parts))
    right_parts = right_parts + [0] * (width - len(right_parts))
    if left_parts < right_parts:
        return -1
    if left_parts > right_parts:
        return 1
    return 0


def _version_in_range(
    current: str,
    *,
    minimum: str | None,
    maximum: str | None,
    noun: str,
) -> None:
    if minimum is not None:
        cmp = _compare_versions(current, minimum)
        if cmp is not None and cmp < 0:
            raise ControlPlaneMigrationError(
                f"{noun} version {current} is below minimum {minimum}"
            )
    if maximum is not None:
        cmp = _compare_versions(current, maximum)
        if cmp is not None and cmp > 0:
            raise ControlPlaneMigrationError(
                f"{noun} version {current} is above maximum {maximum}"
            )


class GuardedDuckDBConnection:
    """Connection wrapper that refuses ad-hoc DDL outside migrations."""

    def __init__(
        self,
        connection: DuckDBConnection | Any,
        *,
        allow_ddl: bool = False,
        compatibility_path: bool = False,
    ) -> None:
        self._connection = connection
        self._allow_ddl = bool(allow_ddl)
        self._compatibility_path = bool(compatibility_path)

    @property
    def allow_ddl(self) -> bool:
        return self._allow_ddl or self._compatibility_path

    def enable_compatibility_path(self) -> None:
        """Explicit opt-in for bounded compatibility DDL outside migrations."""

        self._compatibility_path = True

    def disable_compatibility_path(self) -> None:
        self._compatibility_path = False

    def _guard(self, sql: str) -> None:
        if self.allow_ddl:
            return
        for statement in _split_sql_statements(sql):
            if _is_ddl_statement(statement):
                raise MigrationAdHocDDLError(
                    "runtime ad-hoc DDL is refused outside the migration "
                    "runner or an explicit compatibility path"
                )

    def execute(
        self,
        sql: str,
        parameters: Iterable[Any] | Mapping[str, Any] | None = None,
    ) -> Any:
        self._guard(str(sql))
        if parameters is None:
            return self._connection.execute(sql)
        return self._connection.execute(sql, parameters)

    def executescript(self, sql: str) -> Any:
        self._guard(str(sql))
        if hasattr(self._connection, "executescript"):
            return self._connection.executescript(sql)
        return self._connection.execute(sql)

    def commit(self) -> None:
        if hasattr(self._connection, "commit"):
            self._connection.commit()

    def rollback(self) -> None:
        if hasattr(self._connection, "rollback"):
            self._connection.rollback()

    def close(self) -> None:
        if hasattr(self._connection, "close"):
            self._connection.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


@dataclass
class ControlPlaneMigrationRunner:
    """Transactional, single-owner runner over a :class:`MigrationCatalog`."""

    catalog: MigrationCatalog
    database_path: Path
    application_version: str = field(default_factory=default_application_version)
    tool_version: str = field(default_factory=default_tool_version)
    owner_id: str = field(default_factory=lambda: f"migration-owner:{uuid.uuid4()}")
    ownership_lease_seconds: float = OWNERSHIP_LEASE_SECONDS
    _thread_guard: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        self.database_path = Path(self.database_path)
        if not isinstance(self.catalog, MigrationCatalog):
            raise TypeError("catalog must be a MigrationCatalog")

    @classmethod
    def for_database(
        cls,
        database_path: Path | str,
        *,
        catalog: MigrationCatalog | None = None,
        sql_directory: Path | str | None = None,
        application_version: str | None = None,
        tool_version: str | None = None,
        owner_id: str | None = None,
    ) -> ControlPlaneMigrationRunner:
        resolved_catalog = catalog or MigrationCatalog.from_sql_directory(
            sql_directory
        )
        return cls(
            catalog=resolved_catalog,
            database_path=Path(database_path),
            application_version=(
                application_version
                if application_version is not None
                else default_application_version()
            ),
            tool_version=(
                tool_version if tool_version is not None else default_tool_version()
            ),
            owner_id=owner_id or f"migration-owner:{uuid.uuid4()}",
        )

    def _lock_path(self) -> Path:
        return self.database_path.with_name(f".{self.database_path.name}.migration.lock")

    @contextmanager
    def _exclusive(self) -> Iterator[DuckDBConnection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_guard:
            with exclusive_file_lock(self._lock_path()):
                connection = open_duckdb_connection(self.database_path)
                try:
                    yield connection
                finally:
                    connection.close()

    def ensure_bookkeeping(self, connection: Any | None = None) -> None:
        """Install migration metadata tables (idempotent)."""

        if connection is None:
            with self._exclusive() as owned:
                self._ensure_bookkeeping(owned)
            return
        self._ensure_bookkeeping(connection)

    def _ensure_bookkeeping(self, connection: Any) -> None:
        # Bookkeeping DDL is part of the migration runner path, not ad-hoc.
        if hasattr(connection, "executescript"):
            connection.executescript(_BOOKKEEPING_SQL)
        else:
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
        if _meta_get(connection, META_DATABASE_UUID) is None:
            _meta_set(connection, META_DATABASE_UUID, str(uuid.uuid4()))
        if _meta_get(connection, META_SCHEMA_VERSION) is None:
            _meta_set(connection, META_SCHEMA_VERSION, "0")

    def current_version(self, connection: Any | None = None) -> int:
        if connection is None:
            with self._exclusive() as owned:
                self._ensure_bookkeeping(owned)
                return self._current_version(owned)
        self._ensure_bookkeeping(connection)
        return self._current_version(connection)

    def _current_version(self, connection: Any) -> int:
        raw = _meta_get(connection, META_SCHEMA_VERSION) or "0"
        return int(raw)

    def list_receipts(self, connection: Any | None = None) -> list[MigrationReceipt]:
        if connection is None:
            with self._exclusive() as owned:
                self._ensure_bookkeeping(owned)
                return self._list_receipts(owned)
        self._ensure_bookkeeping(connection)
        return self._list_receipts(connection)

    def _list_receipts(self, connection: Any) -> list[MigrationReceipt]:
        rows = connection.execute(
            """
            SELECT body_json
            FROM schema_migrations
            ORDER BY version ASC
            """
        ).fetchall()
        receipts: list[MigrationReceipt] = []
        for row in rows:
            body = row["body_json"] if isinstance(row, Mapping) else row[0]
            payload = json.loads(str(body))
            receipts.append(MigrationReceipt.from_dict(payload))
        return receipts

    def open_guarded_connection(
        self,
        *,
        compatibility_path: bool = False,
    ) -> GuardedDuckDBConnection:
        """Open a runtime connection that refuses ad-hoc DDL by default."""

        connection = open_duckdb_connection(self.database_path)
        self._ensure_bookkeeping(connection)
        return GuardedDuckDBConnection(
            connection,
            allow_ddl=False,
            compatibility_path=compatibility_path,
        )

    def schema_fingerprint(self, connection: Any | None = None) -> str:
        if connection is None:
            with self._exclusive() as owned:
                self._ensure_bookkeeping(owned)
                return compute_schema_fingerprint(owned)
        self._ensure_bookkeeping(connection)
        return compute_schema_fingerprint(connection)

    def inspect(self) -> dict[str, Any]:
        """Return an inspectable snapshot of catalog and applied state."""

        with self._exclusive() as connection:
            self._ensure_bookkeeping(connection)
            current = self._current_version(connection)
            receipts = self._list_receipts(connection)
            fingerprint = compute_schema_fingerprint(connection)
            partial = self._detect_partial(connection)
            return {
                "database_path": str(self.database_path),
                "database_uuid": _meta_get(connection, META_DATABASE_UUID),
                "current_version": current,
                "latest_version": self.catalog.latest_version,
                "catalog_fingerprint": self.catalog.fingerprint(),
                "schema_fingerprint": fingerprint,
                "partial_application": partial,
                "receipts": [receipt.to_dict() for receipt in receipts],
                "pending_versions": [
                    int(item.version)
                    for item in self.catalog
                    if int(item.version) > current
                ],
            }

    def _detect_partial(self, connection: Any) -> bool:
        if not _table_exists(connection, "schema_migration_attempts"):
            return False
        row = connection.execute(
            """
            SELECT attempt_id
            FROM schema_migration_attempts
            WHERE outcome = ?
            LIMIT 1
            """,
            [OUTCOME_FAILED],
        ).fetchone()
        # A failed attempt that left no successful receipt at that version is
        # partial only when schema_version still points at a lower version and
        # residual non-bookkeeping state exists. Transactional rollback means
        # residual state should not exist; we still refuse if a failed attempt
        # is unrepaired and the version was claimed without a receipt.
        if row is None:
            return False
        # Look for failed attempts whose version has no success receipt.
        failed = connection.execute(
            """
            SELECT a.version
            FROM schema_migration_attempts a
            LEFT JOIN schema_migrations m ON m.version = a.version
            WHERE a.outcome = ? AND m.version IS NULL
            ORDER BY a.started_at DESC
            LIMIT 1
            """,
            [OUTCOME_FAILED],
        ).fetchone()
        return failed is not None and self._has_residual_partial_state(connection)

    def _has_residual_partial_state(self, connection: Any) -> bool:
        # Marker set only when a non-transactional partial path is observed.
        return _meta_get(connection, "partial_application") == "1"

    def _acquire_ownership(self, connection: Any) -> None:
        raw = _meta_get(connection, OWNERSHIP_KEY)
        now = _utc_now()
        if raw:
            try:
                payload = json.loads(raw)
                owner = str(payload.get("owner_id") or "")
                expires_at = _parse_utc(str(payload.get("expires_at")))
            except Exception:
                owner = ""
                expires_at = now
            if owner and owner != self.owner_id and expires_at > now:
                raise MigrationOwnershipError(
                    f"migration ownership held by {owner} until "
                    f"{expires_at.isoformat()}"
                )
        expires = datetime.fromtimestamp(
            now.timestamp() + float(self.ownership_lease_seconds),
            tz=timezone.utc,
        )
        lease = {
            "owner_id": self.owner_id,
            "acquired_at": _utc_iso(now),
            "expires_at": _utc_iso(expires),
        }
        _meta_set(connection, OWNERSHIP_KEY, json.dumps(lease, sort_keys=True))

    def _release_ownership(self, connection: Any) -> None:
        raw = _meta_get(connection, OWNERSHIP_KEY)
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except Exception:
            _meta_delete(connection, OWNERSHIP_KEY)
            return
        if str(payload.get("owner_id") or "") == self.owner_id:
            _meta_delete(connection, OWNERSHIP_KEY)

    def verify_applied_checksums(self, connection: Any | None = None) -> None:
        """Refuse checksum drift between catalog and applied receipts."""

        if connection is None:
            with self._exclusive() as owned:
                self._ensure_bookkeeping(owned)
                self._verify_applied_checksums(owned)
            return
        self._ensure_bookkeeping(connection)
        self._verify_applied_checksums(connection)

    def _verify_applied_checksums(self, connection: Any) -> None:
        receipts = self._list_receipts(connection)
        if not receipts:
            return
        versions = [int(item.version) for item in receipts]
        expected = list(range(1, len(versions) + 1))
        if versions != expected:
            raise MigrationGapError(
                f"applied migration versions have gaps: {versions}"
            )
        for receipt in receipts:
            try:
                migration = self.catalog.get(int(receipt.version))
            except MigrationCatalogError as exc:
                raise MigrationDriftError(
                    f"applied migration version {receipt.version} is absent "
                    "from the current catalog"
                ) from exc
            if receipt.checksum != migration.checksum:
                raise MigrationDriftError(
                    f"checksum drift at version {receipt.version}: "
                    f"applied {receipt.checksum}, catalog {migration.checksum}"
                )
            if receipt.migration_id != migration.migration_id:
                raise MigrationDriftError(
                    f"migration_id drift at version {receipt.version}: "
                    f"applied {receipt.migration_id}, "
                    f"catalog {migration.migration_id}"
                )

    def apply(
        self,
        *,
        target_version: int | None = None,
        allow_partial_repair: bool = False,
        fault_injector: Callable[[str, ControlPlaneMigration], None] | None = None,
    ) -> MigrationRunReport:
        """Apply pending migrations up to ``target_version`` (default: latest)."""

        target = (
            self.catalog.latest_version
            if target_version is None
            else int(target_version)
        )
        if target < 0:
            raise MigrationCatalogError("target_version must be >= 0")
        if target > self.catalog.latest_version:
            raise MigrationCatalogError(
                f"target_version {target} exceeds catalog latest "
                f"{self.catalog.latest_version}"
            )

        with self._exclusive() as connection:
            self._ensure_bookkeeping(connection)
            current = self._current_version(connection)
            if target < current:
                raise MigrationDowngradeError(
                    f"refusing downgrade from schema version {current} to {target}"
                )
            if self._detect_partial(connection) and not allow_partial_repair:
                raise MigrationPartialError(
                    "refusing to continue over a partial migration application"
                )
            self._verify_applied_checksums(connection)
            self._acquire_ownership(connection)
            receipts: list[MigrationReceipt] = []
            started_from = current
            try:
                if target == current:
                    fingerprint = compute_schema_fingerprint(connection)
                    return MigrationRunReport(
                        from_version=current,
                        to_version=current,
                        receipts=tuple(self._list_receipts(connection)),
                        schema_fingerprint=fingerprint,
                        catalog_fingerprint=self.catalog.fingerprint(),
                        changed=False,
                    )
                # Reject gaps between current head and target relative to catalog.
                needed = [
                    migration
                    for migration in self.catalog
                    if current < int(migration.version) <= target
                ]
                expected_versions = list(range(current + 1, target + 1))
                actual_versions = [int(item.version) for item in needed]
                if actual_versions != expected_versions:
                    raise MigrationGapError(
                        f"catalog cannot fill versions {expected_versions}; "
                        f"available pending {actual_versions}"
                    )
                for migration in needed:
                    receipt = self._apply_one(
                        connection,
                        migration,
                        fault_injector=fault_injector,
                    )
                    receipts.append(receipt)
                    if receipt.outcome not in {OUTCOME_APPLIED, OUTCOME_REPLAYED}:
                        raise ControlPlaneMigrationError(
                            f"migration {migration.migration_id} ended with "
                            f"outcome {receipt.outcome}"
                        )
                fingerprint = compute_schema_fingerprint(connection)
                _meta_set(connection, META_SCHEMA_FINGERPRINT, fingerprint)
                _meta_set(
                    connection,
                    META_CATALOG_FINGERPRINT,
                    self.catalog.fingerprint(),
                )
                _meta_set(
                    connection,
                    META_APPLICATION_VERSION,
                    self.application_version,
                )
                _meta_set(connection, META_TOOL_VERSION, self.tool_version)
                return MigrationRunReport(
                    from_version=started_from,
                    to_version=self._current_version(connection),
                    receipts=tuple(receipts),
                    schema_fingerprint=fingerprint,
                    catalog_fingerprint=self.catalog.fingerprint(),
                    changed=bool(receipts),
                )
            finally:
                self._release_ownership(connection)

    def _apply_one(
        self,
        connection: DuckDBConnection,
        migration: ControlPlaneMigration,
        *,
        fault_injector: Callable[[str, ControlPlaneMigration], None] | None,
    ) -> MigrationReceipt:
        current = self._current_version(connection)
        if int(migration.version) != current + 1:
            raise MigrationGapError(
                f"next migration must be version {current + 1}, "
                f"got {migration.version}"
            )
        existing = connection.execute(
            "SELECT body_json FROM schema_migrations WHERE version = ?",
            [int(migration.version)],
        ).fetchone()
        if existing is not None:
            body = existing["body_json"] if isinstance(existing, Mapping) else existing[0]
            receipt = MigrationReceipt.from_dict(json.loads(str(body)))
            if receipt.checksum != migration.checksum:
                raise MigrationDriftError(
                    f"checksum drift at version {migration.version}"
                )
            return receipt

        _version_in_range(
            self.application_version,
            minimum=migration.min_application_version,
            maximum=migration.max_application_version,
            noun="application",
        )
        _version_in_range(
            self.tool_version,
            minimum=migration.min_tool_version,
            maximum=migration.max_tool_version,
            noun="tool",
        )

        started_at = _utc_iso()
        attempt_id = f"attempt:{uuid.uuid4()}"
        attempt_body = {
            "attempt_id": attempt_id,
            "version": int(migration.version),
            "migration_id": migration.migration_id,
            "checksum": migration.checksum,
            "application_version": self.application_version,
            "tool_version": self.tool_version,
            "started_at": started_at,
            "outcome": "started",
        }
        connection.execute(
            """
            INSERT INTO schema_migration_attempts (
                attempt_id, version, migration_id, checksum,
                application_version, tool_version, started_at, finished_at,
                outcome, schema_fingerprint, error_text, body_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?)
            """,
            [
                attempt_id,
                int(migration.version),
                migration.migration_id,
                migration.checksum,
                self.application_version,
                self.tool_version,
                started_at,
                "started",
                "",
                json.dumps(attempt_body, sort_keys=True, separators=(",", ":")),
            ],
        )

        try:
            # Evaluate preconditions before opening the schema transaction.
            for sql in migration.preconditions:
                if not _sql_predicate_holds(connection, sql):
                    raise MigrationPreconditionError(
                        f"precondition failed for {migration.migration_id}: {sql}"
                    )

            if fault_injector is not None:
                fault_injector("before_sql", migration)

            connection.execute("BEGIN TRANSACTION")
            try:
                for statement in _split_sql_statements(migration.sql_text):
                    connection.execute(statement)
                if fault_injector is not None:
                    fault_injector("before_commit", migration)
                for sql in migration.postconditions:
                    if not _sql_predicate_holds(connection, sql):
                        raise MigrationPostconditionError(
                            f"postcondition failed for {migration.migration_id}: {sql}"
                        )
                fingerprint = compute_schema_fingerprint(connection)
                finished_at = _utc_iso()
                receipt_body = {
                    "schema": MIGRATION_RECEIPT_SCHEMA,
                    "version": int(migration.version),
                    "migration_id": migration.migration_id,
                    "checksum": migration.checksum,
                    "application_version": self.application_version,
                    "tool_version": self.tool_version,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "outcome": OUTCOME_APPLIED,
                    "schema_fingerprint": fingerprint,
                    "error_text": "",
                }
                receipt_cid = canonical_content_cid(receipt_body)
                receipt_body["receipt_cid"] = receipt_cid
                # Recompute identity including receipt_cid for durable body.
                # Keep receipt_cid content-bound to the attestation fields only.
                encoded = json.dumps(
                    receipt_body, sort_keys=True, separators=(",", ":")
                )
                connection.execute(
                    """
                    INSERT INTO schema_migrations (
                        version, migration_id, checksum, application_version,
                        tool_version, started_at, finished_at, outcome,
                        schema_fingerprint, receipt_cid, body_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        int(migration.version),
                        migration.migration_id,
                        migration.checksum,
                        self.application_version,
                        self.tool_version,
                        started_at,
                        finished_at,
                        OUTCOME_APPLIED,
                        fingerprint,
                        receipt_cid,
                        encoded,
                    ],
                )
                _meta_set(connection, META_SCHEMA_VERSION, str(int(migration.version)))
                _meta_set(connection, META_SCHEMA_FINGERPRINT, fingerprint)
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

            if fault_injector is not None:
                fault_injector("after_commit", migration)

            finished_at = _utc_iso()
            connection.execute(
                """
                UPDATE schema_migration_attempts
                SET finished_at = ?,
                    outcome = ?,
                    schema_fingerprint = ?,
                    error_text = ?,
                    body_json = ?
                WHERE attempt_id = ?
                """,
                [
                    finished_at,
                    OUTCOME_APPLIED,
                    fingerprint,
                    "",
                    json.dumps(
                        {
                            **attempt_body,
                            "finished_at": finished_at,
                            "outcome": OUTCOME_APPLIED,
                            "schema_fingerprint": fingerprint,
                            "receipt_cid": receipt_cid,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    attempt_id,
                ],
            )
            return MigrationReceipt.from_dict(receipt_body)
        except Exception as exc:
            finished_at = _utc_iso()
            error_text = f"{type(exc).__name__}: {exc}"
            # Attempt row update is best-effort outside the rolled-back txn.
            try:
                connection.execute(
                    """
                    UPDATE schema_migration_attempts
                    SET finished_at = ?,
                        outcome = ?,
                        error_text = ?,
                        body_json = ?
                    WHERE attempt_id = ?
                    """,
                    [
                        finished_at,
                        OUTCOME_FAILED,
                        error_text[:2000],
                        json.dumps(
                            {
                                **attempt_body,
                                "finished_at": finished_at,
                                "outcome": OUTCOME_FAILED,
                                "error_text": error_text[:2000],
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        attempt_id,
                    ],
                )
            except Exception:
                pass
            raise

    def prove_empty_to_latest_equivalence(
        self,
        *,
        other_database_path: Path | str,
    ) -> dict[str, Any]:
        """Apply this catalog to two fresh DBs and compare fingerprints."""

        left = ControlPlaneMigrationRunner(
            catalog=self.catalog,
            database_path=self.database_path,
            application_version=self.application_version,
            tool_version=self.tool_version,
            owner_id=f"{self.owner_id}:left",
        )
        right = ControlPlaneMigrationRunner(
            catalog=self.catalog,
            database_path=Path(other_database_path),
            application_version=self.application_version,
            tool_version=self.tool_version,
            owner_id=f"{self.owner_id}:right",
        )
        left_report = left.apply()
        right_report = right.apply()
        if left_report.schema_fingerprint != right_report.schema_fingerprint:
            raise ControlPlaneMigrationError(
                "empty-to-latest schema fingerprints diverge: "
                f"{left_report.schema_fingerprint} != "
                f"{right_report.schema_fingerprint}"
            )
        return {
            "schema_fingerprint": left_report.schema_fingerprint,
            "catalog_fingerprint": left_report.catalog_fingerprint,
            "to_version": left_report.to_version,
            "equivalent": True,
        }


def load_default_catalog(
    sql_directory: Path | str | None = None,
) -> MigrationCatalog:
    """Load the package SQL catalog (may be empty before domain SQL lands)."""

    return MigrationCatalog.from_sql_directory(sql_directory)


__all__ = [
    "CONTROL_PLANE_MIGRATION_SCHEMA",
    "ControlPlaneMigration",
    "ControlPlaneMigrationError",
    "ControlPlaneMigrationRunner",
    "DuckDBUnavailableError",
    "GuardedDuckDBConnection",
    "MIGRATION_CATALOG_SCHEMA",
    "MIGRATION_RECEIPT_SCHEMA",
    "MIGRATION_RUN_REPORT_SCHEMA",
    "MigrationAdHocDDLError",
    "MigrationCatalog",
    "MigrationCatalogError",
    "MigrationDowngradeError",
    "MigrationDriftError",
    "MigrationGapError",
    "MigrationOwnershipError",
    "MigrationPartialError",
    "MigrationPostconditionError",
    "MigrationPreconditionError",
    "MigrationReceipt",
    "MigrationRunReport",
    "OUTCOME_APPLIED",
    "OUTCOME_FAILED",
    "OUTCOME_REPLAYED",
    "OUTCOME_REFUSED",
    "SQL_DIRECTORY_NAME",
    "checksum_sql",
    "compute_schema_fingerprint",
    "default_application_version",
    "default_tool_version",
    "duckdb_available",
    "load_default_catalog",
]
