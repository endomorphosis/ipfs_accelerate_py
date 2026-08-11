"""DuckDB-backed authoritative artifact, dataset, and edge store.

DQP-025 / DatabaseArtifactStore@1
=================================

:class:`DatabaseArtifactStore` is the durable authority for artifact
metadata, provenance edges, dataset records, and digest-bound external blob
references. JSON, Parquet, and filesystem freshness are export adapters only:
deleting or tampering with an export has no authority effect.

Large immutable bodies remain CAS by digest. The database commits metadata
and edges first; optional exports are receipts bound to the admitted
snapshot. Cache and file age never promote assurance or authority.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ..task_sources.control_plane_contracts import (
    REDACTION_MARKER,
    redact_mapping,
)
from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.task_identity import canonical_json_bytes


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_ARTIFACT_STORE_INTERFACE: Final[str] = "DatabaseArtifactStore@1"
DATABASE_ARTIFACT_STORE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-artifact-store@1"
)
ARTIFACT_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-artifact-record@1"
)
ARTIFACT_EDGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-artifact-edge@1"
)
DATASET_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-dataset-record@1"
)
BLOB_REFERENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-blob-reference@1"
)
ARTIFACT_EXPORT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-artifact-export-receipt@1"
)
PROJECTION_REBUILD_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-artifact-projection-rebuild@1"
)

DEFAULT_SNAPSHOT_ID: Final[str] = "snapshot:database-artifact-store"
AUTHORITY_CLASS: Final[str] = "database_authority"
EXPORT_AUTHORITY: Final[str] = "export_adapter_only"

MAX_BODY_BYTES: Final[int] = 262_144
MAX_RECURSION_DEPTH: Final[int] = 8
MAX_TEXT_BYTES: Final[int] = 8_192
MAX_ID_BYTES: Final[int] = 512
DEFAULT_MAX_ARTIFACTS: Final[int] = 16_384
DEFAULT_MAX_EDGES: Final[int] = 65_536
DEFAULT_MAX_DATASETS: Final[int] = 4_096
DEFAULT_MAX_TOTAL_BLOB_BYTES: Final[int] = 512 * 1024 * 1024
DEFAULT_MAX_BLOB_BYTES: Final[int] = 64 * 1024 * 1024
DEFAULT_MAX_GRAPH_DEGREE: Final[int] = 1_024

_DIGEST_RE_COMPILED = re.compile(r"^sha256:[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS artifact_store_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id VARCHAR PRIMARY KEY,
    kind VARCHAR NOT NULL,
    digest VARCHAR NOT NULL,
    media_type VARCHAR NOT NULL DEFAULT '',
    size_bytes BIGINT NOT NULL DEFAULT 0,
    provenance_json VARCHAR NOT NULL DEFAULT '{}',
    metadata_json VARCHAR NOT NULL DEFAULT '{}',
    redacted BOOLEAN NOT NULL DEFAULT FALSE,
    admitted BOOLEAN NOT NULL DEFAULT TRUE,
    snapshot_id VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS artifacts_kind_idx
    ON artifacts(kind, created_at);
CREATE INDEX IF NOT EXISTS artifacts_digest_idx
    ON artifacts(digest);

CREATE TABLE IF NOT EXISTS artifact_edges (
    edge_id VARCHAR PRIMARY KEY,
    source_artifact_id VARCHAR NOT NULL,
    target_artifact_id VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL,
    reason VARCHAR NOT NULL DEFAULT '',
    snapshot_id VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS artifact_edges_source_idx
    ON artifact_edges(source_artifact_id, edge_kind);
CREATE INDEX IF NOT EXISTS artifact_edges_target_idx
    ON artifact_edges(target_artifact_id, edge_kind);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    digest VARCHAR NOT NULL,
    row_count BIGINT NOT NULL DEFAULT 0,
    byte_count BIGINT NOT NULL DEFAULT 0,
    media_type VARCHAR NOT NULL DEFAULT 'application/x-ndjson',
    provenance_json VARCHAR NOT NULL DEFAULT '{}',
    metadata_json VARCHAR NOT NULL DEFAULT '{}',
    redacted BOOLEAN NOT NULL DEFAULT FALSE,
    admitted BOOLEAN NOT NULL DEFAULT TRUE,
    snapshot_id VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS datasets_name_idx
    ON datasets(name, created_at);

CREATE TABLE IF NOT EXISTS blob_references (
    blob_id VARCHAR PRIMARY KEY,
    digest VARCHAR NOT NULL UNIQUE,
    size_bytes BIGINT NOT NULL,
    media_type VARCHAR NOT NULL DEFAULT 'application/octet-stream',
    external_path VARCHAR NOT NULL DEFAULT '',
    verified BOOLEAN NOT NULL DEFAULT FALSE,
    admitted BOOLEAN NOT NULL DEFAULT TRUE,
    snapshot_id VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS blob_references_digest_idx
    ON blob_references(digest);

CREATE TABLE IF NOT EXISTS artifact_exports (
    export_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    target_path VARCHAR NOT NULL,
    artifact_count BIGINT NOT NULL DEFAULT 0,
    dataset_count BIGINT NOT NULL DEFAULT 0,
    edge_count BIGINT NOT NULL DEFAULT 0,
    export_digest VARCHAR NOT NULL,
    authority VARCHAR NOT NULL DEFAULT 'export_adapter_only',
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS artifact_projections (
    projection_id VARCHAR PRIMARY KEY,
    snapshot_id VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    digest VARCHAR NOT NULL,
    row_count BIGINT NOT NULL DEFAULT 0,
    rebuilt_from VARCHAR NOT NULL DEFAULT 'admitted_evidence',
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS artifact_projections_snapshot_idx
    ON artifact_projections(snapshot_id, kind);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseArtifactStoreError(RuntimeError):
    """Base error for database artifact store failures."""


class DatabaseArtifactStoreNotOpenError(DatabaseArtifactStoreError):
    """Operation requires an open artifact store."""


class DatabaseArtifactStoreIntegrityError(DatabaseArtifactStoreError, ValueError):
    """Identity, digest, or payload integrity failure."""


class DatabaseArtifactStoreBoundsError(DatabaseArtifactStoreError, ValueError):
    """A resource or payload bound was exceeded."""


class DatabaseArtifactStoreConflictError(DatabaseArtifactStoreError):
    """Duplicate identity with a conflicting payload."""


class DatabaseArtifactStoreQuotaError(DatabaseArtifactStoreError):
    """Size or graph quota exceeded."""


class DuckDBUnavailableError(DatabaseArtifactStoreError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ArtifactKind(str, Enum):
    BLOB = "blob"
    PROJECTION = "projection"
    RECEIPT = "receipt"
    DATASET = "dataset"
    ATTESTATION = "attestation"
    GRAPH = "graph"
    GENERIC = "generic"

    @classmethod
    def coerce(cls, value: Any) -> "ArtifactKind":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases = {
            "blob": cls.BLOB,
            "projection": cls.PROJECTION,
            "receipt": cls.RECEIPT,
            "dataset": cls.DATASET,
            "attestation": cls.ATTESTATION,
            "graph": cls.GRAPH,
            "generic": cls.GENERIC,
            "artifact": cls.GENERIC,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise DatabaseArtifactStoreIntegrityError(
                f"unsupported artifact kind: {value!r}"
            ) from exc


class EdgeKind(str, Enum):
    DERIVES_FROM = "derives_from"
    PROVES = "proves"
    ATTESTS = "attests"
    CONTAINS = "contains"
    REFERENCES = "references"
    SUPERSEDES = "supersedes"
    INVALIDATES = "invalidates"
    PROVENANCE = "provenance"

    @classmethod
    def coerce(cls, value: Any) -> "EdgeKind":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases = {
            "derives_from": cls.DERIVES_FROM,
            "derived_from": cls.DERIVES_FROM,
            "proves": cls.PROVES,
            "attests": cls.ATTESTS,
            "contains": cls.CONTAINS,
            "references": cls.REFERENCES,
            "ref": cls.REFERENCES,
            "supersedes": cls.SUPERSEDES,
            "invalidates": cls.INVALIDATES,
            "provenance": cls.PROVENANCE,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise DatabaseArtifactStoreIntegrityError(
                f"unsupported edge kind: {value!r}"
            ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def duckdb_available() -> bool:
    """Return whether the optional duckdb package can be imported."""

    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise DatabaseArtifactStoreIntegrityError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseArtifactStoreIntegrityError(f"{name} is required")
    if len(text.encode("utf-8")) > MAX_ID_BYTES and name.endswith("_id"):
        raise DatabaseArtifactStoreBoundsError(
            f"{name} exceeds {MAX_ID_BYTES} bytes"
        )
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatabaseArtifactStoreBoundsError(
            f"{name} must be a non-negative integer"
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatabaseArtifactStoreBoundsError(
            f"{name} must be a positive integer"
        )
    return value


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json_bytes(value).decode("utf-8")
    except ValueError:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        )


def _sha256_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _require_digest(value: Any, name: str = "digest") -> str:
    text = _text(value, name)
    if not _DIGEST_RE_COMPILED.fullmatch(text):
        raise DatabaseArtifactStoreIntegrityError(
            f"{name} must be sha256:<64 lowercase hex>"
        )
    return text


def _bounded_mapping(
    body: Mapping[str, Any] | None,
    *,
    redact: bool,
    depth: int = 0,
    name: str = "body",
) -> dict[str, Any]:
    if depth > MAX_RECURSION_DEPTH:
        raise DatabaseArtifactStoreBoundsError(
            f"{name} exceeds recursion depth {MAX_RECURSION_DEPTH}"
        )
    raw = dict(body or {})
    cleaned = redact_mapping(raw) if redact else raw
    if not isinstance(cleaned, dict):
        raise DatabaseArtifactStoreIntegrityError(
            f"{name} must project to an object"
        )
    encoded = _canonical_json(cleaned).encode("utf-8")
    if len(encoded) > MAX_BODY_BYTES:
        raise DatabaseArtifactStoreBoundsError(
            f"{name} exceeds the {MAX_BODY_BYTES}-byte bound"
        )
    return cleaned


def _identity(prefix: str, value: Any) -> str:
    return f"{prefix}:{_sha256_digest(_canonical_json(value).encode('utf-8'))}"


def _row_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    try:
        keys = list(row.keys())  # type: ignore[attr-defined]
    except Exception:
        return {}
    return {str(key): row[key] for key in keys}


def _split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    for chunk in str(sql_text).split(";"):
        statement = chunk.strip()
        if not statement:
            continue
        lines = [
            line
            for line in statement.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if lines:
            statements.append("\n".join(lines))
    return statements


def _load_json_object(text: Any) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(str(text))
    except (TypeError, ValueError) as exc:
        raise DatabaseArtifactStoreIntegrityError(
            "stored JSON is corrupted"
        ) from exc
    if not isinstance(value, dict):
        raise DatabaseArtifactStoreIntegrityError(
            "stored JSON must be an object"
        )
    return value


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlobReference:
    """Digest-bound external blob; body is never authority."""

    blob_id: str
    digest: str
    size_bytes: int
    media_type: str = "application/octet-stream"
    external_path: str = ""
    verified: bool = False
    admitted: bool = True
    snapshot_id: str = DEFAULT_SNAPSHOT_ID
    created_at: str = ""
    schema: str = BLOB_REFERENCE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "blob_id": self.blob_id,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "external_path": self.external_path,
            "verified": self.verified,
            "admitted": self.admitted,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ArtifactRecord:
    """Database-authoritative artifact metadata."""

    artifact_id: str
    kind: str
    digest: str
    media_type: str = ""
    size_bytes: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    redacted: bool = False
    admitted: bool = True
    snapshot_id: str = DEFAULT_SNAPSHOT_ID
    created_at: str = ""
    schema: str = ARTIFACT_RECORD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "digest": self.digest,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
            "redacted": self.redacted,
            "admitted": self.admitted,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class ArtifactEdge:
    """Directed provenance or dependency edge between artifacts."""

    edge_id: str
    source_artifact_id: str
    target_artifact_id: str
    edge_kind: str
    reason: str = ""
    snapshot_id: str = DEFAULT_SNAPSHOT_ID
    created_at: str = ""
    schema: str = ARTIFACT_EDGE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "edge_id": self.edge_id,
            "source_artifact_id": self.source_artifact_id,
            "target_artifact_id": self.target_artifact_id,
            "edge_kind": self.edge_kind,
            "reason": self.reason,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class DatasetRecord:
    """Database-authoritative dataset metadata."""

    dataset_id: str
    name: str
    digest: str
    row_count: int = 0
    byte_count: int = 0
    media_type: str = "application/x-ndjson"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    redacted: bool = False
    admitted: bool = True
    snapshot_id: str = DEFAULT_SNAPSHOT_ID
    created_at: str = ""
    schema: str = DATASET_RECORD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "digest": self.digest,
            "row_count": self.row_count,
            "byte_count": self.byte_count,
            "media_type": self.media_type,
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
            "redacted": self.redacted,
            "admitted": self.admitted,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class ArtifactExportReceipt:
    """Non-authoritative export receipt; database remains authority."""

    export_id: str
    snapshot_id: str
    target_path: str
    artifact_count: int
    dataset_count: int
    edge_count: int
    export_digest: str
    authority: str = EXPORT_AUTHORITY
    created_at: str = ""
    schema: str = ARTIFACT_EXPORT_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "export_id": self.export_id,
            "snapshot_id": self.snapshot_id,
            "target_path": self.target_path,
            "artifact_count": self.artifact_count,
            "dataset_count": self.dataset_count,
            "edge_count": self.edge_count,
            "export_digest": self.export_digest,
            "authority": self.authority,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ProjectionRebuildReceipt:
    """Receipt proving a projection was rebuilt from admitted evidence."""

    projection_id: str
    snapshot_id: str
    kind: str
    digest: str
    row_count: int
    rebuilt_from: str = "admitted_evidence"
    created_at: str = ""
    schema: str = PROJECTION_REBUILD_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "projection_id": self.projection_id,
            "snapshot_id": self.snapshot_id,
            "kind": self.kind,
            "digest": self.digest,
            "row_count": self.row_count,
            "rebuilt_from": self.rebuilt_from,
            "created_at": self.created_at,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class ArtifactStoreQuotas:
    """Hard size and graph quotas for the store."""

    max_artifacts: int = DEFAULT_MAX_ARTIFACTS
    max_edges: int = DEFAULT_MAX_EDGES
    max_datasets: int = DEFAULT_MAX_DATASETS
    max_total_blob_bytes: int = DEFAULT_MAX_TOTAL_BLOB_BYTES
    max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES
    max_graph_degree: int = DEFAULT_MAX_GRAPH_DEGREE


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DatabaseArtifactStore:
    """DuckDB authority for artifacts, datasets, edges, and blob references."""

    INTERFACE: Final[str] = DATABASE_ARTIFACT_STORE_INTERFACE

    def __init__(
        self,
        database_path: Path | str,
        *,
        snapshot_id: str = DEFAULT_SNAPSHOT_ID,
        auto_redact: bool = True,
        quotas: ArtifactStoreQuotas | None = None,
        blob_root: Path | str | None = None,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseArtifactStore; install the "
                "optional duckdb dependency"
            )
        self._path = Path(database_path)
        self._snapshot_id = _text(snapshot_id, "snapshot_id")
        self._auto_redact = bool(auto_redact)
        self._quotas = quotas or ArtifactStoreQuotas()
        self._blob_root = (
            Path(blob_root)
            if blob_root is not None
            else self._path.parent / f"{self._path.stem}.blobs"
        )
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    @property
    def blob_root(self) -> Path:
        return self._blob_root

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "DatabaseArtifactStore":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._blob_root.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
            for key, value in (
                ("interface", DATABASE_ARTIFACT_STORE_INTERFACE),
                ("schema", DATABASE_ARTIFACT_STORE_SCHEMA),
                ("snapshot_id", self._snapshot_id),
                ("authority", AUTHORITY_CLASS),
            ):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO artifact_store_metadata(key, value)
                    VALUES (?, ?)
                    """,
                    [key, value],
                )
            self._connection = connection
            self._closed = False
            return self

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            self._connection = None
            self._closed = True
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def __enter__(self) -> "DatabaseArtifactStore":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise DatabaseArtifactStoreNotOpenError(
                "DatabaseArtifactStore is not open"
            )
        return self._connection

    def _commit_if_idle(self, connection: Any) -> None:
        if getattr(connection, "in_transaction", False):
            return
        commit = getattr(connection, "commit", None)
        if callable(commit):
            try:
                commit()
            except Exception:
                pass

    # -- quotas --------------------------------------------------------------

    def _count(self, connection: Any, table: str) -> int:
        row = connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
        mapping = _row_mapping(row)
        return int(mapping.get("n") or mapping.get("count") or 0)

    def _total_blob_bytes(self, connection: Any) -> int:
        row = connection.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) AS total FROM blob_references"
        ).fetchone()
        mapping = _row_mapping(row)
        return int(mapping.get("total") or 0)

    def _degree(self, connection: Any, artifact_id: str) -> int:
        row = connection.execute(
            """
            SELECT COUNT(*) AS n FROM artifact_edges
            WHERE source_artifact_id = ? OR target_artifact_id = ?
            """,
            [artifact_id, artifact_id],
        ).fetchone()
        mapping = _row_mapping(row)
        return int(mapping.get("n") or 0)

    # -- blobs ---------------------------------------------------------------

    def _blob_path(self, digest: str) -> Path:
        hex_part = digest.removeprefix("sha256:")
        return self._blob_root / hex_part[:2] / f"{hex_part}.blob"

    def put_blob(
        self,
        data: bytes | bytearray | memoryview | str,
        *,
        media_type: str = "application/octet-stream",
        external_path: str = "",
    ) -> BlobReference:
        """Admit a digest-bound blob. Database metadata is authority."""

        if isinstance(data, str):
            payload = data.encode("utf-8")
        else:
            payload = bytes(data)
        size = len(payload)
        if size > self._quotas.max_blob_bytes:
            raise DatabaseArtifactStoreQuotaError(
                f"blob exceeds max_blob_bytes={self._quotas.max_blob_bytes}"
            )
        digest = _sha256_digest(payload)
        blob_id = f"blob:{digest}"
        stamp = _utc_iso()
        media = _text(media_type, "media_type", required=False) or (
            "application/octet-stream"
        )

        with self._lock:
            connection = self._require()
            existing = connection.execute(
                "SELECT * FROM blob_references WHERE digest = ?",
                [digest],
            ).fetchone()
            if existing is not None:
                record = self._blob_from_row(_row_mapping(existing))
                if record.size_bytes != size:
                    raise DatabaseArtifactStoreIntegrityError(
                        "blob digest collides with a different size"
                    )
                path = self._blob_path(digest)
                if path.is_file():
                    on_disk = path.read_bytes()
                    if _sha256_digest(on_disk) != digest:
                        raise DatabaseArtifactStoreIntegrityError(
                            "stored blob is corrupted"
                        )
                return record

            if self._count(connection, "blob_references") >= (
                self._quotas.max_artifacts
            ):
                raise DatabaseArtifactStoreQuotaError(
                    "blob reference quota exceeded"
                )
            if self._total_blob_bytes(connection) + size > (
                self._quotas.max_total_blob_bytes
            ):
                raise DatabaseArtifactStoreQuotaError(
                    "total blob byte quota exceeded"
                )

            path = self._blob_path(digest)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(payload)
                tmp.replace(path)
            # Verify on write so corruption fails closed immediately.
            if _sha256_digest(path.read_bytes()) != digest:
                raise DatabaseArtifactStoreIntegrityError(
                    "blob failed post-write digest verification"
                )

            body = {
                "digest": digest,
                "size_bytes": size,
                "media_type": media,
            }
            connection.execute(
                """
                INSERT INTO blob_references(
                    blob_id, digest, size_bytes, media_type, external_path,
                    verified, admitted, snapshot_id, created_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    blob_id,
                    digest,
                    size,
                    media,
                    _text(external_path, "external_path", required=False),
                    True,
                    True,
                    self._snapshot_id,
                    stamp,
                    _canonical_json(body),
                ],
            )
            self._commit_if_idle(connection)
            return BlobReference(
                blob_id=blob_id,
                digest=digest,
                size_bytes=size,
                media_type=media,
                external_path=_text(external_path, "external_path", required=False),
                verified=True,
                admitted=True,
                snapshot_id=self._snapshot_id,
                created_at=stamp,
            )

    def get_blob(self, digest_or_id: str) -> BlobReference | None:
        key = _text(digest_or_id, "digest_or_id")
        with self._lock:
            connection = self._require()
            if key.startswith("blob:"):
                row = connection.execute(
                    "SELECT * FROM blob_references WHERE blob_id = ?",
                    [key],
                ).fetchone()
            else:
                digest = _require_digest(key)
                row = connection.execute(
                    "SELECT * FROM blob_references WHERE digest = ?",
                    [digest],
                ).fetchone()
            if row is None:
                return None
            return self._blob_from_row(_row_mapping(row))

    def verify_blob(self, digest_or_id: str) -> bytes:
        """Load and verify an external blob by digest. Fail closed on mismatch."""

        reference = self.get_blob(digest_or_id)
        if reference is None or not reference.admitted:
            raise DatabaseArtifactStoreIntegrityError(
                "blob reference is missing or not admitted"
            )
        path = self._blob_path(reference.digest)
        if not path.is_file():
            raise DatabaseArtifactStoreIntegrityError(
                "external blob body is missing"
            )
        payload = path.read_bytes()
        actual = _sha256_digest(payload)
        if actual != reference.digest or len(payload) != reference.size_bytes:
            raise DatabaseArtifactStoreIntegrityError(
                "blob digest or size verification failed"
            )
        return payload

    def _blob_from_row(self, row: Mapping[str, Any]) -> BlobReference:
        return BlobReference(
            blob_id=str(row["blob_id"]),
            digest=str(row["digest"]),
            size_bytes=int(row["size_bytes"]),
            media_type=str(row.get("media_type") or ""),
            external_path=str(row.get("external_path") or ""),
            verified=bool(row.get("verified")),
            admitted=bool(row.get("admitted", True)),
            snapshot_id=str(row.get("snapshot_id") or self._snapshot_id),
            created_at=str(row.get("created_at") or ""),
        )

    # -- artifacts -----------------------------------------------------------

    def put_artifact(
        self,
        *,
        kind: ArtifactKind | str = ArtifactKind.GENERIC,
        digest: str | None = None,
        body: Mapping[str, Any] | None = None,
        media_type: str = "",
        size_bytes: int = 0,
        provenance: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        artifact_id: str | None = None,
        blob_digest: str | None = None,
        redact: bool | None = None,
    ) -> ArtifactRecord:
        """Commit artifact metadata before any optional export."""

        do_redact = self._auto_redact if redact is None else bool(redact)
        selected_kind = ArtifactKind.coerce(kind)
        meta = _bounded_mapping(
            metadata, redact=do_redact, name="metadata"
        )
        prov = _bounded_mapping(
            provenance, redact=do_redact, name="provenance"
        )
        payload = _bounded_mapping(body, redact=do_redact, name="body")
        stamp = _utc_iso()

        if blob_digest is not None:
            blob_ref = self.get_blob(blob_digest)
            if blob_ref is None:
                raise DatabaseArtifactStoreIntegrityError(
                    "blob_digest is not admitted"
                )
            # Verify on use.
            self.verify_blob(blob_ref.digest)
            content_digest = blob_ref.digest
            size = blob_ref.size_bytes
            media = media_type or blob_ref.media_type
        else:
            if digest is not None:
                content_digest = _require_digest(digest)
            else:
                content_digest = _sha256_digest(
                    _canonical_json(
                        {
                            "kind": selected_kind.value,
                            "body": payload,
                            "metadata": meta,
                            "provenance": prov,
                        }
                    ).encode("utf-8")
                )
            size = _nonneg_int(size_bytes, "size_bytes")
            media = _text(media_type, "media_type", required=False)

        identity_material = {
            "kind": selected_kind.value,
            "digest": content_digest,
            "snapshot_id": self._snapshot_id,
            "metadata": meta,
            "provenance": prov,
            "body": payload,
        }
        computed_id = _identity("artifact", identity_material)
        selected_id = _text(artifact_id or computed_id, "artifact_id")

        with self._lock:
            connection = self._require()
            existing = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                [selected_id],
            ).fetchone()
            if existing is not None:
                current = self._artifact_from_row(_row_mapping(existing))
                if (
                    current.digest != content_digest
                    or current.kind != selected_kind.value
                ):
                    raise DatabaseArtifactStoreConflictError(
                        "artifact_id already exists with a different payload"
                    )
                return current

            if self._count(connection, "artifacts") >= self._quotas.max_artifacts:
                raise DatabaseArtifactStoreQuotaError(
                    "artifact quota exceeded"
                )

            connection.execute(
                """
                INSERT INTO artifacts(
                    artifact_id, kind, digest, media_type, size_bytes,
                    provenance_json, metadata_json, redacted, admitted,
                    snapshot_id, created_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    selected_id,
                    selected_kind.value,
                    content_digest,
                    media,
                    size,
                    _canonical_json(prov),
                    _canonical_json(meta),
                    do_redact,
                    True,
                    self._snapshot_id,
                    stamp,
                    _canonical_json(payload),
                ],
            )
            self._commit_if_idle(connection)
            return ArtifactRecord(
                artifact_id=selected_id,
                kind=selected_kind.value,
                digest=content_digest,
                media_type=media,
                size_bytes=size,
                provenance=MappingProxyType(prov),
                metadata=MappingProxyType(meta),
                redacted=do_redact,
                admitted=True,
                snapshot_id=self._snapshot_id,
                created_at=stamp,
            )

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        selected = _text(artifact_id, "artifact_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                [selected],
            ).fetchone()
            if row is None:
                return None
            return self._artifact_from_row(_row_mapping(row))

    def list_artifacts(
        self,
        *,
        kind: ArtifactKind | str | None = None,
        admitted_only: bool = True,
        limit: int = 256,
    ) -> list[ArtifactRecord]:
        bound = min(_positive_int(limit, "limit"), 4_096)
        clauses = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(ArtifactKind.coerce(kind).value)
        if admitted_only:
            clauses.append("admitted = TRUE")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                f"""
                SELECT * FROM artifacts
                {where}
                ORDER BY created_at, artifact_id
                LIMIT ?
                """,
                [*params, bound],
            ).fetchall()
            return [self._artifact_from_row(_row_mapping(row)) for row in rows]

    def _artifact_from_row(self, row: Mapping[str, Any]) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=str(row["artifact_id"]),
            kind=str(row["kind"]),
            digest=str(row["digest"]),
            media_type=str(row.get("media_type") or ""),
            size_bytes=int(row.get("size_bytes") or 0),
            provenance=MappingProxyType(
                _load_json_object(row.get("provenance_json"))
            ),
            metadata=MappingProxyType(
                _load_json_object(row.get("metadata_json"))
            ),
            redacted=bool(row.get("redacted")),
            admitted=bool(row.get("admitted", True)),
            snapshot_id=str(row.get("snapshot_id") or self._snapshot_id),
            created_at=str(row.get("created_at") or ""),
        )

    # -- edges ---------------------------------------------------------------

    def put_edge(
        self,
        source_artifact_id: str,
        target_artifact_id: str,
        edge_kind: EdgeKind | str,
        *,
        reason: str = "",
        edge_id: str | None = None,
    ) -> ArtifactEdge:
        source = _text(source_artifact_id, "source_artifact_id")
        target = _text(target_artifact_id, "target_artifact_id")
        kind = EdgeKind.coerce(edge_kind)
        stamp = _utc_iso()
        reason_text = str(reason or "")[:MAX_TEXT_BYTES]
        material = {
            "source": source,
            "target": target,
            "edge_kind": kind.value,
            "reason": reason_text,
            "snapshot_id": self._snapshot_id,
        }
        selected_id = _text(
            edge_id or _identity("edge", material), "edge_id"
        )

        with self._lock:
            connection = self._require()
            # Endpoints must be admitted when present as artifacts.
            for endpoint in (source, target):
                row = connection.execute(
                    "SELECT admitted FROM artifacts WHERE artifact_id = ?",
                    [endpoint],
                ).fetchone()
                if row is not None and not bool(_row_mapping(row).get("admitted", True)):
                    raise DatabaseArtifactStoreIntegrityError(
                        f"endpoint is not admitted: {endpoint}"
                    )

            if self._count(connection, "artifact_edges") >= self._quotas.max_edges:
                raise DatabaseArtifactStoreQuotaError("edge quota exceeded")
            if self._degree(connection, source) >= self._quotas.max_graph_degree:
                raise DatabaseArtifactStoreQuotaError(
                    "source graph degree quota exceeded"
                )
            if self._degree(connection, target) >= self._quotas.max_graph_degree:
                raise DatabaseArtifactStoreQuotaError(
                    "target graph degree quota exceeded"
                )

            existing = connection.execute(
                "SELECT * FROM artifact_edges WHERE edge_id = ?",
                [selected_id],
            ).fetchone()
            if existing is not None:
                current = self._edge_from_row(_row_mapping(existing))
                if (
                    current.source_artifact_id != source
                    or current.target_artifact_id != target
                    or current.edge_kind != kind.value
                ):
                    raise DatabaseArtifactStoreConflictError(
                        "edge_id already exists with a different payload"
                    )
                return current

            connection.execute(
                """
                INSERT INTO artifact_edges(
                    edge_id, source_artifact_id, target_artifact_id,
                    edge_kind, reason, snapshot_id, created_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    selected_id,
                    source,
                    target,
                    kind.value,
                    reason_text,
                    self._snapshot_id,
                    stamp,
                    _canonical_json(material),
                ],
            )
            self._commit_if_idle(connection)
            return ArtifactEdge(
                edge_id=selected_id,
                source_artifact_id=source,
                target_artifact_id=target,
                edge_kind=kind.value,
                reason=reason_text,
                snapshot_id=self._snapshot_id,
                created_at=stamp,
            )

    def list_edges(
        self,
        *,
        artifact_id: str | None = None,
        edge_kind: EdgeKind | str | None = None,
        limit: int = 256,
    ) -> list[ArtifactEdge]:
        bound = min(_positive_int(limit, "limit"), 4_096)
        clauses: list[str] = []
        params: list[Any] = []
        if artifact_id is not None:
            selected = _text(artifact_id, "artifact_id")
            clauses.append(
                "(source_artifact_id = ? OR target_artifact_id = ?)"
            )
            params.extend([selected, selected])
        if edge_kind is not None:
            clauses.append("edge_kind = ?")
            params.append(EdgeKind.coerce(edge_kind).value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                f"""
                SELECT * FROM artifact_edges
                {where}
                ORDER BY created_at, edge_id
                LIMIT ?
                """,
                [*params, bound],
            ).fetchall()
            return [self._edge_from_row(_row_mapping(row)) for row in rows]

    def _edge_from_row(self, row: Mapping[str, Any]) -> ArtifactEdge:
        return ArtifactEdge(
            edge_id=str(row["edge_id"]),
            source_artifact_id=str(row["source_artifact_id"]),
            target_artifact_id=str(row["target_artifact_id"]),
            edge_kind=str(row["edge_kind"]),
            reason=str(row.get("reason") or ""),
            snapshot_id=str(row.get("snapshot_id") or self._snapshot_id),
            created_at=str(row.get("created_at") or ""),
        )

    # -- datasets ------------------------------------------------------------

    def put_dataset(
        self,
        *,
        name: str,
        digest: str | None = None,
        rows: Sequence[Mapping[str, Any]] | None = None,
        row_count: int | None = None,
        byte_count: int = 0,
        media_type: str = "application/x-ndjson",
        provenance: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        dataset_id: str | None = None,
        redact: bool | None = None,
    ) -> DatasetRecord:
        """Admit dataset metadata. File/Parquet freshness is not authority."""

        do_redact = self._auto_redact if redact is None else bool(redact)
        selected_name = _text(name, "name")
        meta = _bounded_mapping(metadata, redact=do_redact, name="metadata")
        prov = _bounded_mapping(
            provenance, redact=do_redact, name="provenance"
        )
        stamp = _utc_iso()

        material_rows: list[dict[str, Any]] = []
        if rows is not None:
            for index, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    raise DatabaseArtifactStoreIntegrityError(
                        f"rows[{index}] must be an object"
                    )
                material_rows.append(
                    _bounded_mapping(
                        row, redact=do_redact, name=f"rows[{index}]"
                    )
                )
            computed_count = len(material_rows)
            content_digest = _sha256_digest(
                _canonical_json(material_rows).encode("utf-8")
            )
            size = len(_canonical_json(material_rows).encode("utf-8"))
        else:
            computed_count = _nonneg_int(
                row_count if row_count is not None else 0, "row_count"
            )
            if digest is None:
                raise DatabaseArtifactStoreIntegrityError(
                    "digest is required when rows are not supplied"
                )
            content_digest = _require_digest(digest)
            size = _nonneg_int(byte_count, "byte_count")

        identity_material = {
            "name": selected_name,
            "digest": content_digest,
            "snapshot_id": self._snapshot_id,
            "row_count": computed_count,
            "metadata": meta,
            "provenance": prov,
        }
        selected_id = _text(
            dataset_id or _identity("dataset", identity_material),
            "dataset_id",
        )

        with self._lock:
            connection = self._require()
            existing = connection.execute(
                "SELECT * FROM datasets WHERE dataset_id = ?",
                [selected_id],
            ).fetchone()
            if existing is not None:
                current = self._dataset_from_row(_row_mapping(existing))
                if current.digest != content_digest:
                    raise DatabaseArtifactStoreConflictError(
                        "dataset_id already exists with a different digest"
                    )
                return current

            if self._count(connection, "datasets") >= self._quotas.max_datasets:
                raise DatabaseArtifactStoreQuotaError(
                    "dataset quota exceeded"
                )

            body = {
                "row_preview_count": min(len(material_rows), 8),
                "has_inline_rows": bool(material_rows),
            }
            connection.execute(
                """
                INSERT INTO datasets(
                    dataset_id, name, digest, row_count, byte_count,
                    media_type, provenance_json, metadata_json, redacted,
                    admitted, snapshot_id, created_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    selected_id,
                    selected_name,
                    content_digest,
                    computed_count,
                    size,
                    _text(media_type, "media_type", required=False)
                    or "application/x-ndjson",
                    _canonical_json(prov),
                    _canonical_json(meta),
                    do_redact,
                    True,
                    self._snapshot_id,
                    stamp,
                    _canonical_json(body),
                ],
            )
            self._commit_if_idle(connection)
            return DatasetRecord(
                dataset_id=selected_id,
                name=selected_name,
                digest=content_digest,
                row_count=computed_count,
                byte_count=size,
                media_type=_text(media_type, "media_type", required=False)
                or "application/x-ndjson",
                provenance=MappingProxyType(prov),
                metadata=MappingProxyType(meta),
                redacted=do_redact,
                admitted=True,
                snapshot_id=self._snapshot_id,
                created_at=stamp,
            )

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        selected = _text(dataset_id, "dataset_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM datasets WHERE dataset_id = ?",
                [selected],
            ).fetchone()
            if row is None:
                return None
            return self._dataset_from_row(_row_mapping(row))

    def _dataset_from_row(self, row: Mapping[str, Any]) -> DatasetRecord:
        return DatasetRecord(
            dataset_id=str(row["dataset_id"]),
            name=str(row["name"]),
            digest=str(row["digest"]),
            row_count=int(row.get("row_count") or 0),
            byte_count=int(row.get("byte_count") or 0),
            media_type=str(row.get("media_type") or ""),
            provenance=MappingProxyType(
                _load_json_object(row.get("provenance_json"))
            ),
            metadata=MappingProxyType(
                _load_json_object(row.get("metadata_json"))
            ),
            redacted=bool(row.get("redacted")),
            admitted=bool(row.get("admitted", True)),
            snapshot_id=str(row.get("snapshot_id") or self._snapshot_id),
            created_at=str(row.get("created_at") or ""),
        )

    # -- export / rebuild ----------------------------------------------------

    def export_snapshot(self, target_path: Path | str) -> ArtifactExportReceipt:
        """Write a non-authoritative JSON export bound to the DB snapshot."""

        target = Path(target_path)
        with self._lock:
            connection = self._require()
            artifacts = [
                self._artifact_from_row(_row_mapping(row)).to_dict()
                for row in connection.execute(
                    "SELECT * FROM artifacts WHERE admitted = TRUE "
                    "ORDER BY created_at, artifact_id"
                ).fetchall()
            ]
            datasets = [
                self._dataset_from_row(_row_mapping(row)).to_dict()
                for row in connection.execute(
                    "SELECT * FROM datasets WHERE admitted = TRUE "
                    "ORDER BY created_at, dataset_id"
                ).fetchall()
            ]
            edges = [
                self._edge_from_row(_row_mapping(row)).to_dict()
                for row in connection.execute(
                    "SELECT * FROM artifact_edges "
                    "ORDER BY created_at, edge_id"
                ).fetchall()
            ]
            payload = {
                "schema": ARTIFACT_EXPORT_RECEIPT_SCHEMA,
                "snapshot_id": self._snapshot_id,
                "authority": EXPORT_AUTHORITY,
                "artifacts": artifacts,
                "datasets": datasets,
                "edges": edges,
            }
            encoded = _canonical_json(payload).encode("utf-8")
            export_digest = _sha256_digest(encoded)
            stamp = _utc_iso()
            export_id = _identity(
                "export",
                {
                    "snapshot_id": self._snapshot_id,
                    "digest": export_digest,
                    "created_at": stamp,
                },
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(encoded)
            connection.execute(
                """
                INSERT INTO artifact_exports(
                    export_id, snapshot_id, target_path, artifact_count,
                    dataset_count, edge_count, export_digest, authority,
                    created_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    export_id,
                    self._snapshot_id,
                    str(target),
                    len(artifacts),
                    len(datasets),
                    len(edges),
                    export_digest,
                    EXPORT_AUTHORITY,
                    stamp,
                    _canonical_json(
                        {
                            "target_path": str(target),
                            "export_digest": export_digest,
                        }
                    ),
                ],
            )
            self._commit_if_idle(connection)
            return ArtifactExportReceipt(
                export_id=export_id,
                snapshot_id=self._snapshot_id,
                target_path=str(target),
                artifact_count=len(artifacts),
                dataset_count=len(datasets),
                edge_count=len(edges),
                export_digest=export_digest,
                authority=EXPORT_AUTHORITY,
                created_at=stamp,
            )

    def rebuild_projection(
        self,
        kind: str = "admitted_artifacts",
    ) -> ProjectionRebuildReceipt:
        """Rebuild a projection strictly from admitted database evidence."""

        selected_kind = _text(kind, "kind")
        with self._lock:
            connection = self._require()
            if selected_kind == "admitted_datasets":
                rows = connection.execute(
                    "SELECT * FROM datasets WHERE admitted = TRUE "
                    "ORDER BY created_at, dataset_id"
                ).fetchall()
                material = [
                    self._dataset_from_row(_row_mapping(row)).to_dict()
                    for row in rows
                ]
            elif selected_kind == "admitted_edges":
                rows = connection.execute(
                    "SELECT * FROM artifact_edges "
                    "ORDER BY created_at, edge_id"
                ).fetchall()
                material = [
                    self._edge_from_row(_row_mapping(row)).to_dict()
                    for row in rows
                ]
            else:
                rows = connection.execute(
                    "SELECT * FROM artifacts WHERE admitted = TRUE "
                    "ORDER BY created_at, artifact_id"
                ).fetchall()
                material = [
                    self._artifact_from_row(_row_mapping(row)).to_dict()
                    for row in rows
                ]
            digest = _sha256_digest(_canonical_json(material).encode("utf-8"))
            stamp = _utc_iso()
            projection_id = _identity(
                "projection",
                {
                    "kind": selected_kind,
                    "snapshot_id": self._snapshot_id,
                    "digest": digest,
                },
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO artifact_projections(
                    projection_id, snapshot_id, kind, digest, row_count,
                    rebuilt_from, created_at, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    projection_id,
                    self._snapshot_id,
                    selected_kind,
                    digest,
                    len(material),
                    "admitted_evidence",
                    stamp,
                    _canonical_json(
                        {
                            "kind": selected_kind,
                            "digest": digest,
                            "row_count": len(material),
                        }
                    ),
                ],
            )
            self._commit_if_idle(connection)
            return ProjectionRebuildReceipt(
                projection_id=projection_id,
                snapshot_id=self._snapshot_id,
                kind=selected_kind,
                digest=digest,
                row_count=len(material),
                rebuilt_from="admitted_evidence",
                created_at=stamp,
            )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            connection = self._require()
            return {
                "interface": self.INTERFACE,
                "snapshot_id": self._snapshot_id,
                "authority": AUTHORITY_CLASS,
                "artifact_count": self._count(connection, "artifacts"),
                "dataset_count": self._count(connection, "datasets"),
                "edge_count": self._count(connection, "artifact_edges"),
                "blob_count": self._count(connection, "blob_references"),
                "total_blob_bytes": self._total_blob_bytes(connection),
            }


def open_database_artifact_store(
    database_path: Path | str,
    *,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    auto_redact: bool = True,
    quotas: ArtifactStoreQuotas | None = None,
    blob_root: Path | str | None = None,
) -> DatabaseArtifactStore:
    """Open and initialize a DatabaseArtifactStore."""

    return DatabaseArtifactStore(
        database_path,
        snapshot_id=snapshot_id,
        auto_redact=auto_redact,
        quotas=quotas,
        blob_root=blob_root,
    ).open()


__all__ = (
    "ARTIFACT_EDGE_SCHEMA",
    "ARTIFACT_EXPORT_RECEIPT_SCHEMA",
    "ARTIFACT_RECORD_SCHEMA",
    "AUTHORITY_CLASS",
    "ArtifactEdge",
    "ArtifactExportReceipt",
    "ArtifactKind",
    "ArtifactRecord",
    "ArtifactStoreQuotas",
    "BLOB_REFERENCE_SCHEMA",
    "BlobReference",
    "DATABASE_ARTIFACT_STORE_INTERFACE",
    "DATABASE_ARTIFACT_STORE_SCHEMA",
    "DATASET_RECORD_SCHEMA",
    "DEFAULT_SNAPSHOT_ID",
    "DatabaseArtifactStore",
    "DatabaseArtifactStoreBoundsError",
    "DatabaseArtifactStoreConflictError",
    "DatabaseArtifactStoreError",
    "DatabaseArtifactStoreIntegrityError",
    "DatabaseArtifactStoreNotOpenError",
    "DatabaseArtifactStoreQuotaError",
    "DatasetRecord",
    "DuckDBUnavailableError",
    "EXPORT_AUTHORITY",
    "EdgeKind",
    "PROJECTION_REBUILD_RECEIPT_SCHEMA",
    "ProjectionRebuildReceipt",
    "REDACTION_MARKER",
    "duckdb_available",
    "open_database_artifact_store",
)
