"""DuckDB-backed database-first artifact and dataset authority.

DQP-025 / DatabaseArtifactStore@1
=================================

:class:`DatabaseArtifactStore` is the durable authority for artifact
metadata, dataset descriptors, graph edges, blob references, provenance,
and size/graph quotas. JSON, Parquet, and other filesystem renders are
export adapters only: deleting, aging, or tampering with an export has no
authority effect.

Large immutable bodies may remain on a content-addressed filesystem CAS
keyed solely by digest. Every blob is digest-bound at admission and
re-verified on use; a stale, missing, or poisoned body fails closed and
never promotes an export or cache hit into authority.

Database projections rebuild from admitted evidence rows, not from file
mtime or export freshness.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Iterator

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
    "ipfs_accelerate_py/agent-supervisor/artifact-record@1"
)
DATASET_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/dataset-record@1"
)
BLOB_REFERENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/artifact-blob-reference@1"
)
ARTIFACT_EDGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/artifact-edge@1"
)
ARTIFACT_EXPORT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/artifact-export-receipt@1"
)
ARTIFACT_PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/artifact-projection@1"
)
ADMITTED_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/artifact-admitted-evidence@1"
)

DEFAULT_SNAPSHOT_ID: Final[str] = "snapshot:database-artifact-store"
DEFAULT_MAX_BLOB_BYTES: Final[int] = 64 * 1024 * 1024
DEFAULT_MAX_ARTIFACTS: Final[int] = 16_384
DEFAULT_MAX_EDGES: Final[int] = 65_536
DEFAULT_MAX_DATASETS: Final[int] = 4_096
DEFAULT_MAX_METADATA_BYTES: Final[int] = 262_144
DEFAULT_MAX_GRAPH_DEGREE: Final[int] = 1_024
MAX_RECURSION_DEPTH: Final[int] = 8

_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_AUTHORITY_DATABASE: Final[str] = "database"
_AUTHORITY_EXPORT_ONLY: Final[str] = "export_only"
_AUTHORITY_CAS: Final[str] = "digest_bound_cas"

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS artifact_store_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS admitted_evidence (
    evidence_id VARCHAR PRIMARY KEY,
    evidence_kind VARCHAR NOT NULL,
    subject_id VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    admitted_at VARCHAR NOT NULL,
    sequence BIGINT NOT NULL,
    body_json VARCHAR NOT NULL,
    provenance_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS admitted_evidence_seq_uidx
    ON admitted_evidence(sequence);
CREATE INDEX IF NOT EXISTS admitted_evidence_subject_idx
    ON admitted_evidence(subject_id, sequence);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id VARCHAR PRIMARY KEY,
    kind VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    media_type VARCHAR NOT NULL DEFAULT 'application/json',
    size_bytes BIGINT NOT NULL DEFAULT 0,
    blob_digest VARCHAR NOT NULL DEFAULT '',
    retention_class VARCHAR NOT NULL DEFAULT 'routine',
    outcome VARCHAR NOT NULL DEFAULT 'successful',
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    provenance_json VARCHAR NOT NULL DEFAULT '{}',
    metadata_json VARCHAR NOT NULL DEFAULT '{}',
    redacted BOOLEAN NOT NULL DEFAULT FALSE,
    admitted_sequence BIGINT NOT NULL DEFAULT 0,
    authority VARCHAR NOT NULL DEFAULT 'database'
);
CREATE INDEX IF NOT EXISTS artifacts_kind_idx
    ON artifacts(kind, created_at);
CREATE INDEX IF NOT EXISTS artifacts_digest_idx
    ON artifacts(content_digest);

CREATE TABLE IF NOT EXISTS artifact_edges (
    edge_id VARCHAR PRIMARY KEY,
    source_id VARCHAR NOT NULL,
    target_id VARCHAR NOT NULL,
    edge_kind VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    admitted_sequence BIGINT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS artifact_edges_source_idx
    ON artifact_edges(source_id, edge_kind);
CREATE INDEX IF NOT EXISTS artifact_edges_target_idx
    ON artifact_edges(target_id, edge_kind);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    schema_id VARCHAR NOT NULL DEFAULT '',
    row_count BIGINT NOT NULL DEFAULT 0,
    blob_digest VARCHAR NOT NULL DEFAULT '',
    created_at VARCHAR NOT NULL,
    updated_at VARCHAR NOT NULL,
    provenance_json VARCHAR NOT NULL DEFAULT '{}',
    metadata_json VARCHAR NOT NULL DEFAULT '{}',
    redacted BOOLEAN NOT NULL DEFAULT FALSE,
    admitted_sequence BIGINT NOT NULL DEFAULT 0,
    authority VARCHAR NOT NULL DEFAULT 'database'
);
CREATE INDEX IF NOT EXISTS datasets_name_idx
    ON datasets(name, created_at);

CREATE TABLE IF NOT EXISTS blob_references (
    blob_digest VARCHAR PRIMARY KEY,
    size_bytes BIGINT NOT NULL,
    media_type VARCHAR NOT NULL DEFAULT 'application/octet-stream',
    cas_path VARCHAR NOT NULL DEFAULT '',
    created_at VARCHAR NOT NULL,
    last_verified_at VARCHAR NOT NULL DEFAULT '',
    reference_count BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS artifact_projections (
    projection_id VARCHAR PRIMARY KEY,
    projection_kind VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    rebuilt_from_sequence BIGINT NOT NULL,
    rebuilt_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    authority VARCHAR NOT NULL DEFAULT 'database'
);

CREATE TABLE IF NOT EXISTS artifact_exports (
    export_id VARCHAR PRIMARY KEY,
    export_path VARCHAR NOT NULL,
    export_format VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    subject_count BIGINT NOT NULL,
    recorded_at VARCHAR NOT NULL,
    authority VARCHAR NOT NULL DEFAULT 'export_only',
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseArtifactStoreError(RuntimeError):
    """Base error for database artifact-store failures."""


class DatabaseArtifactStoreNotOpenError(DatabaseArtifactStoreError):
    """Operation requires an open artifact store."""


class DatabaseArtifactStoreConflictError(DatabaseArtifactStoreError):
    """Identity conflict, digest mismatch, or stale authority claim."""


class DatabaseArtifactStoreIntegrityError(DatabaseArtifactStoreError):
    """Digest, CAS, or projection integrity failure (fail closed)."""


class DatabaseArtifactStoreBoundsError(DatabaseArtifactStoreError, ValueError):
    """Size, graph, or payload bound exceeded."""


class DatabaseArtifactStoreQuotaError(DatabaseArtifactStoreBoundsError):
    """Store quota would be exceeded by the write."""


class DuckDBUnavailableError(DatabaseArtifactStoreError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class ArtifactOutcome(str, Enum):
    SUCCESSFUL = "successful"
    PARTIAL = "partial"
    FAILED = "failed"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"

    @classmethod
    def coerce(cls, value: Any) -> "ArtifactOutcome":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().casefold().replace("-", "_")
        aliases = {
            "success": cls.SUCCESSFUL,
            "ok": cls.SUCCESSFUL,
            "complete": cls.SUCCESSFUL,
            "error": cls.FAILED,
            "failure": cls.FAILED,
        }
        try:
            return aliases.get(text, cls(text))
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise DatabaseArtifactStoreError(
                f"artifact outcome must be one of: {choices}"
            ) from exc


class RetentionClass(str, Enum):
    EPHEMERAL = "ephemeral"
    NEGATIVE = "negative"
    ROUTINE = "routine"
    CHECKPOINT = "checkpoint"
    AUTHORITATIVE = "authoritative"
    PINNED = "pinned"

    @classmethod
    def coerce(cls, value: Any) -> "RetentionClass":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().casefold().replace("-", "_")
        try:
            return cls(text)
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise DatabaseArtifactStoreError(
                f"retention_class must be one of: {choices}"
            ) from exc


class EvidenceKind(str, Enum):
    ARTIFACT = "artifact"
    DATASET = "dataset"
    EDGE = "edge"
    BLOB = "blob"
    INVALIDATION = "invalidation"
    PROJECTION = "projection"


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
        raise DatabaseArtifactStoreError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseArtifactStoreError(f"{name} is required")
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


def _sha256_hex(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


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


def _bounded_mapping(
    body: Mapping[str, Any] | None,
    *,
    redact: bool,
    maximum: int = DEFAULT_MAX_METADATA_BYTES,
    depth: int = 0,
) -> dict[str, Any]:
    if depth > MAX_RECURSION_DEPTH:
        raise DatabaseArtifactStoreBoundsError(
            f"metadata exceeds recursion depth {MAX_RECURSION_DEPTH}"
        )
    raw = dict(body or {})
    cleaned = redact_mapping(raw) if redact else raw
    if not isinstance(cleaned, dict):
        raise DatabaseArtifactStoreError("metadata must project to an object")
    encoded = _canonical_json(cleaned).encode("utf-8")
    if len(encoded) > maximum:
        raise DatabaseArtifactStoreBoundsError(
            f"metadata exceeds the {maximum}-byte bound"
        )
    return cleaned


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
        if not statement or statement.startswith("--"):
            continue
        lines = [
            line
            for line in statement.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if lines:
            statements.append("\n".join(lines))
    return statements


def _require_digest(value: Any, name: str = "digest") -> str:
    text = _text(value, name)
    if not _SHA256_DIGEST.fullmatch(text):
        raise DatabaseArtifactStoreIntegrityError(
            f"{name} must be sha256:<64-hex>"
        )
    return text


def _blob_cas_path(root: Path, digest: str) -> Path:
    hex_digest = digest.removeprefix("sha256:")
    return root / "blobs" / "sha256" / hex_digest[:2] / f"{hex_digest}.blob"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlobReference:
    """Digest-bound reference to an external immutable body."""

    blob_digest: str
    size_bytes: int
    media_type: str = "application/octet-stream"
    cas_path: str = ""
    schema: str = BLOB_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "blob_digest", _require_digest(self.blob_digest, "blob_digest")
        )
        object.__setattr__(
            self, "size_bytes", _nonneg_int(self.size_bytes, "size_bytes")
        )
        object.__setattr__(
            self,
            "media_type",
            _text(self.media_type or "application/octet-stream", "media_type"),
        )
        object.__setattr__(
            self, "cas_path", _text(self.cas_path, "cas_path", required=False)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "blob_digest": self.blob_digest,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "cas_path": self.cas_path,
            "authority": _AUTHORITY_CAS,
        }


@dataclass(frozen=True)
class ArtifactRecord:
    """Authoritative artifact metadata projection."""

    artifact_id: str
    kind: str
    content_digest: str
    size_bytes: int = 0
    media_type: str = "application/json"
    blob_digest: str = ""
    retention_class: str = RetentionClass.ROUTINE.value
    outcome: str = ArtifactOutcome.SUCCESSFUL.value
    created_at: str = ""
    updated_at: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    redacted: bool = False
    admitted_sequence: int = 0
    authority: str = _AUTHORITY_DATABASE
    schema: str = ARTIFACT_RECORD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": DATABASE_ARTIFACT_STORE_INTERFACE,
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "content_digest": self.content_digest,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "blob_digest": self.blob_digest,
            "retention_class": self.retention_class,
            "outcome": self.outcome,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
            "redacted": self.redacted,
            "admitted_sequence": self.admitted_sequence,
            "authority": self.authority,
        }


@dataclass(frozen=True)
class DatasetRecord:
    """Authoritative dataset descriptor projection."""

    dataset_id: str
    name: str
    content_digest: str
    schema_id: str = ""
    row_count: int = 0
    blob_digest: str = ""
    created_at: str = ""
    updated_at: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    redacted: bool = False
    admitted_sequence: int = 0
    authority: str = _AUTHORITY_DATABASE
    schema: str = DATASET_RECORD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "content_digest": self.content_digest,
            "schema_id": self.schema_id,
            "row_count": self.row_count,
            "blob_digest": self.blob_digest,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
            "redacted": self.redacted,
            "admitted_sequence": self.admitted_sequence,
            "authority": self.authority,
        }


@dataclass(frozen=True)
class ArtifactEdge:
    """Directed relationship between two admitted artifacts or datasets."""

    edge_id: str
    source_id: str
    target_id: str
    edge_kind: str
    created_at: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)
    admitted_sequence: int = 0
    schema: str = ARTIFACT_EDGE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_kind": self.edge_kind,
            "created_at": self.created_at,
            "body": dict(self.body),
            "admitted_sequence": self.admitted_sequence,
        }


@dataclass(frozen=True)
class ArtifactExportReceipt:
    """Non-authoritative receipt for a JSON/Parquet export render."""

    export_id: str
    export_path: str
    export_format: str
    content_digest: str
    subject_count: int
    recorded_at: str
    authority: str = _AUTHORITY_EXPORT_ONLY
    schema: str = ARTIFACT_EXPORT_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "export_id": self.export_id,
            "export_path": self.export_path,
            "export_format": self.export_format,
            "content_digest": self.content_digest,
            "subject_count": self.subject_count,
            "recorded_at": self.recorded_at,
            "authority": self.authority,
            "authoritative": False,
        }


@dataclass(frozen=True)
class ArtifactProjection:
    """Database projection rebuilt solely from admitted evidence."""

    projection_id: str
    projection_kind: str
    content_digest: str
    rebuilt_from_sequence: int
    rebuilt_at: str
    body: Mapping[str, Any]
    authority: str = _AUTHORITY_DATABASE
    schema: str = ARTIFACT_PROJECTION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "projection_id": self.projection_id,
            "projection_kind": self.projection_kind,
            "content_digest": self.content_digest,
            "rebuilt_from_sequence": self.rebuilt_from_sequence,
            "rebuilt_at": self.rebuilt_at,
            "body": dict(self.body),
            "authority": self.authority,
        }


@dataclass(frozen=True)
class ArtifactQuotaPolicy:
    """Hard bounds for artifact, dataset, edge, and blob admission."""

    max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES
    max_artifacts: int = DEFAULT_MAX_ARTIFACTS
    max_edges: int = DEFAULT_MAX_EDGES
    max_datasets: int = DEFAULT_MAX_DATASETS
    max_metadata_bytes: int = DEFAULT_MAX_METADATA_BYTES
    max_graph_degree: int = DEFAULT_MAX_GRAPH_DEGREE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_blob_bytes",
            _positive_int(self.max_blob_bytes, "max_blob_bytes"),
        )
        object.__setattr__(
            self,
            "max_artifacts",
            _positive_int(self.max_artifacts, "max_artifacts"),
        )
        object.__setattr__(
            self, "max_edges", _positive_int(self.max_edges, "max_edges")
        )
        object.__setattr__(
            self,
            "max_datasets",
            _positive_int(self.max_datasets, "max_datasets"),
        )
        object.__setattr__(
            self,
            "max_metadata_bytes",
            _positive_int(self.max_metadata_bytes, "max_metadata_bytes"),
        )
        object.__setattr__(
            self,
            "max_graph_degree",
            _positive_int(self.max_graph_degree, "max_graph_degree"),
        )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DatabaseArtifactStore:
    """DuckDB authority for artifacts, datasets, edges, and digest-bound CAS."""

    INTERFACE: Final[str] = DATABASE_ARTIFACT_STORE_INTERFACE

    def __init__(
        self,
        database_path: Path | str,
        *,
        cas_root: Path | str | None = None,
        snapshot_id: str = DEFAULT_SNAPSHOT_ID,
        quotas: ArtifactQuotaPolicy | Mapping[str, Any] | None = None,
        auto_redact: bool = True,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseArtifactStore; install the "
                "optional duckdb dependency"
            )
        self._path = Path(database_path)
        self._cas_root = Path(cas_root) if cas_root is not None else (
            self._path.parent / f"{self._path.stem}.cas"
        )
        self._snapshot_id = _text(snapshot_id, "snapshot_id")
        self._auto_redact = bool(auto_redact)
        if quotas is None:
            self._quotas = ArtifactQuotaPolicy()
        elif isinstance(quotas, ArtifactQuotaPolicy):
            self._quotas = quotas
        else:
            self._quotas = ArtifactQuotaPolicy(**dict(quotas))
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True
        self._sequence = 0

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def cas_root(self) -> Path:
        return self._cas_root

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    @property
    def quotas(self) -> ArtifactQuotaPolicy:
        return self._quotas

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "DatabaseArtifactStore":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._cas_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            connection = open_duckdb_connection(self._path)
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
            for key, value in (
                ("interface", DATABASE_ARTIFACT_STORE_INTERFACE),
                ("schema", DATABASE_ARTIFACT_STORE_SCHEMA),
                ("snapshot_id", self._snapshot_id),
                ("authority", _AUTHORITY_DATABASE),
            ):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO artifact_store_metadata(key, value)
                    VALUES (?, ?)
                    """,
                    [key, value],
                )
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS max_seq FROM admitted_evidence"
            ).fetchone()
            mapping = _row_mapping(row) if row is not None else {}
            self._sequence = int(mapping.get("max_seq") or 0)
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

    def _next_sequence(self, connection: Any) -> int:
        self._sequence += 1
        return self._sequence

    def _admit_evidence(
        self,
        connection: Any,
        *,
        evidence_kind: EvidenceKind | str,
        subject_id: str,
        content_digest: str,
        body: Mapping[str, Any],
        provenance: Mapping[str, Any],
        sequence: int | None = None,
    ) -> int:
        seq = sequence if sequence is not None else self._next_sequence(connection)
        kind = (
            evidence_kind.value
            if isinstance(evidence_kind, EvidenceKind)
            else _text(evidence_kind, "evidence_kind")
        )
        stamp = _utc_iso()
        material = {
            "evidence_kind": kind,
            "subject_id": subject_id,
            "content_digest": content_digest,
            "sequence": seq,
            "body": dict(body),
            "provenance": dict(provenance),
        }
        evidence_id = "evidence:" + _sha256_hex(
            _canonical_json(material).encode("utf-8")
        )
        connection.execute(
            """
            INSERT INTO admitted_evidence(
                evidence_id, evidence_kind, subject_id, content_digest,
                admitted_at, sequence, body_json, provenance_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                evidence_id,
                kind,
                subject_id,
                content_digest,
                stamp,
                seq,
                _canonical_json(body),
                _canonical_json(provenance),
            ],
        )
        return seq

    # -- blob CAS ------------------------------------------------------------

    def put_blob(
        self,
        data: bytes | bytearray | str | Mapping[str, Any] | Sequence[Any] | Any,
        *,
        media_type: str | None = None,
    ) -> BlobReference:
        """Admit one digest-bound external body into CAS and the database."""

        if isinstance(data, (bytes, bytearray)):
            payload = bytes(data)
            inferred = "application/octet-stream"
        elif isinstance(data, str):
            payload = data.encode("utf-8")
            inferred = "text/plain; charset=utf-8"
        elif isinstance(data, Mapping):
            payload = _canonical_json(dict(data)).encode("utf-8")
            inferred = "application/json"
        elif isinstance(data, Sequence):
            payload = _canonical_json(list(data)).encode("utf-8")
            inferred = "application/json"
        else:
            payload = _canonical_json(data).encode("utf-8")
            inferred = "application/json"
        if len(payload) > self._quotas.max_blob_bytes:
            raise DatabaseArtifactStoreQuotaError(
                f"blob exceeds {self._quotas.max_blob_bytes}-byte bound"
            )
        digest = _sha256_hex(payload)
        selected_media = _text(
            media_type or inferred, "media_type"
        )
        cas_path = _blob_cas_path(self._cas_root, digest)
        with self._lock:
            connection = self._require()
            existing = connection.execute(
                "SELECT * FROM blob_references WHERE blob_digest=?",
                [digest],
            ).fetchone()
            if existing is not None:
                row = _row_mapping(existing)
                path = Path(str(row.get("cas_path") or cas_path))
                if not path.is_file():
                    raise DatabaseArtifactStoreIntegrityError(
                        "blob reference exists but CAS body is missing"
                    )
                observed = _sha256_hex(path.read_bytes())
                if observed != digest:
                    raise DatabaseArtifactStoreIntegrityError(
                        "CAS body is poisoned relative to its digest"
                    )
                connection.execute(
                    """
                    UPDATE blob_references
                    SET last_verified_at=?, reference_count=reference_count+1
                    WHERE blob_digest=?
                    """,
                    [_utc_iso(), digest],
                )
                self._commit_if_idle(connection)
                return BlobReference(
                    blob_digest=digest,
                    size_bytes=int(row["size_bytes"]),
                    media_type=str(row["media_type"]),
                    cas_path=str(row.get("cas_path") or path),
                )
            _atomic_write_bytes(cas_path, payload)
            stamp = _utc_iso()
            connection.execute("BEGIN TRANSACTION")
            try:
                connection.execute(
                    """
                    INSERT INTO blob_references(
                        blob_digest, size_bytes, media_type, cas_path,
                        created_at, last_verified_at, reference_count
                    ) VALUES (?, ?, ?, ?, ?, ?, 1)
                    """,
                    [
                        digest,
                        len(payload),
                        selected_media,
                        str(cas_path),
                        stamp,
                        stamp,
                    ],
                )
                self._admit_evidence(
                    connection,
                    evidence_kind=EvidenceKind.BLOB,
                    subject_id=f"blob:{digest}",
                    content_digest=digest,
                    body={
                        "size_bytes": len(payload),
                        "media_type": selected_media,
                    },
                    provenance={"cas_path": str(cas_path)},
                )
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            self._commit_if_idle(connection)
            return BlobReference(
                blob_digest=digest,
                size_bytes=len(payload),
                media_type=selected_media,
                cas_path=str(cas_path),
            )

    def verify_blob(self, blob_digest: str) -> bytes:
        """Load and verify one external blob; fail closed on mismatch."""

        digest = _require_digest(blob_digest, "blob_digest")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM blob_references WHERE blob_digest=?",
                [digest],
            ).fetchone()
            if row is None:
                raise DatabaseArtifactStoreIntegrityError(
                    f"unknown blob digest {digest}"
                )
            mapping = _row_mapping(row)
            path = Path(str(mapping.get("cas_path") or ""))
            if not path.is_file():
                # Path may have been relocated relative to cas_root.
                path = _blob_cas_path(self._cas_root, digest)
            if not path.is_file():
                raise DatabaseArtifactStoreIntegrityError(
                    f"CAS body missing for {digest}"
                )
            payload = path.read_bytes()
            observed = _sha256_hex(payload)
            if observed != digest:
                raise DatabaseArtifactStoreIntegrityError(
                    f"CAS body poisoned for {digest}"
                )
            if int(mapping.get("size_bytes") or -1) != len(payload):
                raise DatabaseArtifactStoreIntegrityError(
                    f"CAS size mismatch for {digest}"
                )
            connection.execute(
                """
                UPDATE blob_references SET last_verified_at=?
                WHERE blob_digest=?
                """,
                [_utc_iso(), digest],
            )
            self._commit_if_idle(connection)
            return payload

    def get_blob_reference(self, blob_digest: str) -> BlobReference | None:
        digest = _require_digest(blob_digest, "blob_digest")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM blob_references WHERE blob_digest=?",
                [digest],
            ).fetchone()
            if row is None:
                return None
            mapping = _row_mapping(row)
            return BlobReference(
                blob_digest=str(mapping["blob_digest"]),
                size_bytes=int(mapping["size_bytes"]),
                media_type=str(mapping["media_type"]),
                cas_path=str(mapping.get("cas_path") or ""),
            )

    # -- artifacts -----------------------------------------------------------

    def admit_artifact(
        self,
        kind: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        body: bytes | bytearray | str | Mapping[str, Any] | Sequence[Any] | Any | None = None,
        blob_digest: str = "",
        retention_class: RetentionClass | str = RetentionClass.ROUTINE,
        outcome: ArtifactOutcome | str = ArtifactOutcome.SUCCESSFUL,
        media_type: str = "application/json",
        artifact_id: str | None = None,
        redact: bool | None = None,
    ) -> ArtifactRecord:
        """Commit artifact metadata (and optional digest-bound body) as authority."""

        selected_kind = _text(kind, "kind")
        do_redact = self._auto_redact if redact is None else bool(redact)
        meta = _bounded_mapping(
            metadata,
            redact=do_redact,
            maximum=self._quotas.max_metadata_bytes,
        )
        prov = _bounded_mapping(
            provenance,
            redact=do_redact,
            maximum=self._quotas.max_metadata_bytes,
        )
        selected_retention = RetentionClass.coerce(retention_class)
        selected_outcome = ArtifactOutcome.coerce(outcome)
        selected_media = _text(media_type, "media_type")
        selected_blob = ""
        size_bytes = 0
        if body is not None:
            blob_ref = self.put_blob(body, media_type=selected_media)
            selected_blob = blob_ref.blob_digest
            size_bytes = blob_ref.size_bytes
            selected_media = blob_ref.media_type
        elif blob_digest:
            selected_blob = _require_digest(blob_digest, "blob_digest")
            # Verify on use at admission time as well.
            payload = self.verify_blob(selected_blob)
            size_bytes = len(payload)
            existing = self.get_blob_reference(selected_blob)
            if existing is not None:
                selected_media = existing.media_type

        content_material = {
            "kind": selected_kind,
            "metadata": meta,
            "blob_digest": selected_blob,
            "retention_class": selected_retention.value,
            "outcome": selected_outcome.value,
            "media_type": selected_media,
        }
        content_digest = _sha256_hex(
            _canonical_json(content_material).encode("utf-8")
        )
        computed_id = f"artifact:{content_digest}"
        selected_id = _text(artifact_id or computed_id, "artifact_id")
        stamp = _utc_iso()

        with self._lock:
            connection = self._require()
            existing = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id=?",
                [selected_id],
            ).fetchone()
            if existing is not None:
                prior = self._row_to_artifact(_row_mapping(existing))
                if prior.content_digest != content_digest:
                    raise DatabaseArtifactStoreConflictError(
                        "artifact_id already exists with a different content identity"
                    )
                return prior

            count_row = connection.execute(
                "SELECT COUNT(*) AS n FROM artifacts"
            ).fetchone()
            count = int(_row_mapping(count_row).get("n") or 0)
            if count >= self._quotas.max_artifacts:
                raise DatabaseArtifactStoreQuotaError(
                    f"artifact count exceeds quota {self._quotas.max_artifacts}"
                )

            connection.execute("BEGIN TRANSACTION")
            try:
                seq = self._next_sequence(connection)
                connection.execute(
                    """
                    INSERT INTO artifacts(
                        artifact_id, kind, content_digest, media_type,
                        size_bytes, blob_digest, retention_class, outcome,
                        created_at, updated_at, provenance_json, metadata_json,
                        redacted, admitted_sequence, authority
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        selected_id,
                        selected_kind,
                        content_digest,
                        selected_media,
                        size_bytes,
                        selected_blob,
                        selected_retention.value,
                        selected_outcome.value,
                        stamp,
                        stamp,
                        _canonical_json(prov),
                        _canonical_json(meta),
                        do_redact,
                        seq,
                        _AUTHORITY_DATABASE,
                    ],
                )
                self._admit_evidence(
                    connection,
                    evidence_kind=EvidenceKind.ARTIFACT,
                    subject_id=selected_id,
                    content_digest=content_digest,
                    body={
                        "kind": selected_kind,
                        "blob_digest": selected_blob,
                        "size_bytes": size_bytes,
                        "metadata": meta,
                    },
                    provenance=prov,
                    sequence=seq,
                )
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            self._commit_if_idle(connection)
            return ArtifactRecord(
                artifact_id=selected_id,
                kind=selected_kind,
                content_digest=content_digest,
                size_bytes=size_bytes,
                media_type=selected_media,
                blob_digest=selected_blob,
                retention_class=selected_retention.value,
                outcome=selected_outcome.value,
                created_at=stamp,
                updated_at=stamp,
                provenance=MappingProxyType(prov),
                metadata=MappingProxyType(meta),
                redacted=do_redact,
                admitted_sequence=seq,
            )

    def get_artifact(
        self,
        artifact_id: str,
        *,
        verify_blob: bool = True,
    ) -> ArtifactRecord | None:
        """Load one artifact; optionally re-verify its external body."""

        selected_id = _text(artifact_id, "artifact_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id=?",
                [selected_id],
            ).fetchone()
            if row is None:
                return None
            record = self._row_to_artifact(_row_mapping(row))
            if verify_blob and record.blob_digest:
                self.verify_blob(record.blob_digest)
            return record

    def list_artifacts(
        self,
        *,
        kind: str | None = None,
        limit: int = 256,
    ) -> list[ArtifactRecord]:
        limit = min(_positive_int(limit, "limit"), 4_096)
        with self._lock:
            connection = self._require()
            if kind is not None:
                rows = connection.execute(
                    """
                    SELECT * FROM artifacts
                    WHERE kind=?
                    ORDER BY admitted_sequence ASC
                    LIMIT ?
                    """,
                    [_text(kind, "kind"), limit],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM artifacts
                    ORDER BY admitted_sequence ASC
                    LIMIT ?
                    """,
                    [limit],
                ).fetchall()
            return [self._row_to_artifact(_row_mapping(row)) for row in rows]

    @staticmethod
    def _row_to_artifact(mapping: Mapping[str, Any]) -> ArtifactRecord:
        try:
            metadata = json.loads(str(mapping.get("metadata_json") or "{}"))
            provenance = json.loads(str(mapping.get("provenance_json") or "{}"))
        except json.JSONDecodeError as exc:
            raise DatabaseArtifactStoreIntegrityError(
                "poisoned artifact metadata envelope"
            ) from exc
        if not isinstance(metadata, dict) or not isinstance(provenance, dict):
            raise DatabaseArtifactStoreIntegrityError(
                "poisoned artifact metadata envelope"
            )
        return ArtifactRecord(
            artifact_id=str(mapping["artifact_id"]),
            kind=str(mapping["kind"]),
            content_digest=str(mapping["content_digest"]),
            size_bytes=int(mapping.get("size_bytes") or 0),
            media_type=str(mapping.get("media_type") or "application/json"),
            blob_digest=str(mapping.get("blob_digest") or ""),
            retention_class=str(
                mapping.get("retention_class") or RetentionClass.ROUTINE.value
            ),
            outcome=str(
                mapping.get("outcome") or ArtifactOutcome.SUCCESSFUL.value
            ),
            created_at=str(mapping.get("created_at") or ""),
            updated_at=str(mapping.get("updated_at") or ""),
            provenance=MappingProxyType(provenance),
            metadata=MappingProxyType(metadata),
            redacted=bool(mapping.get("redacted")),
            admitted_sequence=int(mapping.get("admitted_sequence") or 0),
            authority=str(mapping.get("authority") or _AUTHORITY_DATABASE),
        )

    # -- datasets ------------------------------------------------------------

    def admit_dataset(
        self,
        name: str,
        *,
        schema_id: str = "",
        row_count: int = 0,
        metadata: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
        body: bytes | bytearray | str | Mapping[str, Any] | Sequence[Any] | Any | None = None,
        blob_digest: str = "",
        dataset_id: str | None = None,
        redact: bool | None = None,
    ) -> DatasetRecord:
        """Commit one dataset descriptor as database authority."""

        selected_name = _text(name, "name")
        do_redact = self._auto_redact if redact is None else bool(redact)
        meta = _bounded_mapping(
            metadata,
            redact=do_redact,
            maximum=self._quotas.max_metadata_bytes,
        )
        prov = _bounded_mapping(
            provenance,
            redact=do_redact,
            maximum=self._quotas.max_metadata_bytes,
        )
        selected_rows = _nonneg_int(row_count, "row_count")
        selected_schema = _text(schema_id, "schema_id", required=False)
        selected_blob = ""
        if body is not None:
            blob_ref = self.put_blob(body)
            selected_blob = blob_ref.blob_digest
        elif blob_digest:
            selected_blob = _require_digest(blob_digest, "blob_digest")
            self.verify_blob(selected_blob)

        content_material = {
            "name": selected_name,
            "schema_id": selected_schema,
            "row_count": selected_rows,
            "metadata": meta,
            "blob_digest": selected_blob,
        }
        content_digest = _sha256_hex(
            _canonical_json(content_material).encode("utf-8")
        )
        computed_id = f"dataset:{content_digest}"
        selected_id = _text(dataset_id or computed_id, "dataset_id")
        stamp = _utc_iso()

        with self._lock:
            connection = self._require()
            existing = connection.execute(
                "SELECT * FROM datasets WHERE dataset_id=?",
                [selected_id],
            ).fetchone()
            if existing is not None:
                prior = self._row_to_dataset(_row_mapping(existing))
                if prior.content_digest != content_digest:
                    raise DatabaseArtifactStoreConflictError(
                        "dataset_id already exists with a different content identity"
                    )
                return prior

            count_row = connection.execute(
                "SELECT COUNT(*) AS n FROM datasets"
            ).fetchone()
            count = int(_row_mapping(count_row).get("n") or 0)
            if count >= self._quotas.max_datasets:
                raise DatabaseArtifactStoreQuotaError(
                    f"dataset count exceeds quota {self._quotas.max_datasets}"
                )

            connection.execute("BEGIN TRANSACTION")
            try:
                seq = self._next_sequence(connection)
                connection.execute(
                    """
                    INSERT INTO datasets(
                        dataset_id, name, content_digest, schema_id, row_count,
                        blob_digest, created_at, updated_at, provenance_json,
                        metadata_json, redacted, admitted_sequence, authority
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        selected_id,
                        selected_name,
                        content_digest,
                        selected_schema,
                        selected_rows,
                        selected_blob,
                        stamp,
                        stamp,
                        _canonical_json(prov),
                        _canonical_json(meta),
                        do_redact,
                        seq,
                        _AUTHORITY_DATABASE,
                    ],
                )
                self._admit_evidence(
                    connection,
                    evidence_kind=EvidenceKind.DATASET,
                    subject_id=selected_id,
                    content_digest=content_digest,
                    body={
                        "name": selected_name,
                        "schema_id": selected_schema,
                        "row_count": selected_rows,
                        "blob_digest": selected_blob,
                        "metadata": meta,
                    },
                    provenance=prov,
                    sequence=seq,
                )
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            self._commit_if_idle(connection)
            return DatasetRecord(
                dataset_id=selected_id,
                name=selected_name,
                content_digest=content_digest,
                schema_id=selected_schema,
                row_count=selected_rows,
                blob_digest=selected_blob,
                created_at=stamp,
                updated_at=stamp,
                provenance=MappingProxyType(prov),
                metadata=MappingProxyType(meta),
                redacted=do_redact,
                admitted_sequence=seq,
            )

    def get_dataset(
        self,
        dataset_id: str,
        *,
        verify_blob: bool = True,
    ) -> DatasetRecord | None:
        selected_id = _text(dataset_id, "dataset_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM datasets WHERE dataset_id=?",
                [selected_id],
            ).fetchone()
            if row is None:
                return None
            record = self._row_to_dataset(_row_mapping(row))
            if verify_blob and record.blob_digest:
                self.verify_blob(record.blob_digest)
            return record

    def list_datasets(self, *, limit: int = 256) -> list[DatasetRecord]:
        limit = min(_positive_int(limit, "limit"), 4_096)
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT * FROM datasets
                ORDER BY admitted_sequence ASC
                LIMIT ?
                """,
                [limit],
            ).fetchall()
            return [self._row_to_dataset(_row_mapping(row)) for row in rows]

    @staticmethod
    def _row_to_dataset(mapping: Mapping[str, Any]) -> DatasetRecord:
        try:
            metadata = json.loads(str(mapping.get("metadata_json") or "{}"))
            provenance = json.loads(str(mapping.get("provenance_json") or "{}"))
        except json.JSONDecodeError as exc:
            raise DatabaseArtifactStoreIntegrityError(
                "poisoned dataset metadata envelope"
            ) from exc
        if not isinstance(metadata, dict) or not isinstance(provenance, dict):
            raise DatabaseArtifactStoreIntegrityError(
                "poisoned dataset metadata envelope"
            )
        return DatasetRecord(
            dataset_id=str(mapping["dataset_id"]),
            name=str(mapping["name"]),
            content_digest=str(mapping["content_digest"]),
            schema_id=str(mapping.get("schema_id") or ""),
            row_count=int(mapping.get("row_count") or 0),
            blob_digest=str(mapping.get("blob_digest") or ""),
            created_at=str(mapping.get("created_at") or ""),
            updated_at=str(mapping.get("updated_at") or ""),
            provenance=MappingProxyType(provenance),
            metadata=MappingProxyType(metadata),
            redacted=bool(mapping.get("redacted")),
            admitted_sequence=int(mapping.get("admitted_sequence") or 0),
            authority=str(mapping.get("authority") or _AUTHORITY_DATABASE),
        )

    # -- edges ---------------------------------------------------------------

    def link(
        self,
        source_id: str,
        target_id: str,
        edge_kind: str,
        *,
        body: Mapping[str, Any] | None = None,
    ) -> ArtifactEdge:
        """Admit one directed edge after both endpoints exist in authority."""

        source = _text(source_id, "source_id")
        target = _text(target_id, "target_id")
        kind = _text(edge_kind, "edge_kind")
        payload = _bounded_mapping(
            body, redact=self._auto_redact, maximum=self._quotas.max_metadata_bytes
        )
        stamp = _utc_iso()
        material = {
            "source_id": source,
            "target_id": target,
            "edge_kind": kind,
            "body": payload,
        }
        edge_id = "edge:" + _sha256_hex(_canonical_json(material).encode("utf-8"))

        with self._lock:
            connection = self._require()
            if not self._subject_exists(connection, source):
                raise DatabaseArtifactStoreConflictError(
                    f"source endpoint is not admitted: {source}"
                )
            if not self._subject_exists(connection, target):
                raise DatabaseArtifactStoreConflictError(
                    f"target endpoint is not admitted: {target}"
                )
            existing = connection.execute(
                "SELECT * FROM artifact_edges WHERE edge_id=?",
                [edge_id],
            ).fetchone()
            if existing is not None:
                return self._row_to_edge(_row_mapping(existing))

            edge_count = int(
                _row_mapping(
                    connection.execute(
                        "SELECT COUNT(*) AS n FROM artifact_edges"
                    ).fetchone()
                ).get("n")
                or 0
            )
            if edge_count >= self._quotas.max_edges:
                raise DatabaseArtifactStoreQuotaError(
                    f"edge count exceeds quota {self._quotas.max_edges}"
                )
            degree = int(
                _row_mapping(
                    connection.execute(
                        """
                        SELECT COUNT(*) AS n FROM artifact_edges
                        WHERE source_id=? OR target_id=?
                        """,
                        [source, source],
                    ).fetchone()
                ).get("n")
                or 0
            )
            if degree >= self._quotas.max_graph_degree:
                raise DatabaseArtifactStoreQuotaError(
                    f"graph degree exceeds quota {self._quotas.max_graph_degree}"
                )

            connection.execute("BEGIN TRANSACTION")
            try:
                seq = self._next_sequence(connection)
                connection.execute(
                    """
                    INSERT INTO artifact_edges(
                        edge_id, source_id, target_id, edge_kind,
                        created_at, body_json, admitted_sequence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        edge_id,
                        source,
                        target,
                        kind,
                        stamp,
                        _canonical_json(payload),
                        seq,
                    ],
                )
                self._admit_evidence(
                    connection,
                    evidence_kind=EvidenceKind.EDGE,
                    subject_id=edge_id,
                    content_digest=_sha256_hex(
                        _canonical_json(material).encode("utf-8")
                    ),
                    body=material,
                    provenance={},
                    sequence=seq,
                )
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            self._commit_if_idle(connection)
            return ArtifactEdge(
                edge_id=edge_id,
                source_id=source,
                target_id=target,
                edge_kind=kind,
                created_at=stamp,
                body=MappingProxyType(payload),
                admitted_sequence=seq,
            )

    def list_edges(
        self,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        limit: int = 256,
    ) -> list[ArtifactEdge]:
        limit = min(_positive_int(limit, "limit"), 4_096)
        with self._lock:
            connection = self._require()
            clauses: list[str] = []
            params: list[Any] = []
            if source_id is not None:
                clauses.append("source_id=?")
                params.append(_text(source_id, "source_id"))
            if target_id is not None:
                clauses.append("target_id=?")
                params.append(_text(target_id, "target_id"))
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            params.append(limit)
            rows = connection.execute(
                f"""
                SELECT * FROM artifact_edges
                {where}
                ORDER BY admitted_sequence ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return [self._row_to_edge(_row_mapping(row)) for row in rows]

    def _subject_exists(self, connection: Any, subject_id: str) -> bool:
        for table, column in (
            ("artifacts", "artifact_id"),
            ("datasets", "dataset_id"),
        ):
            row = connection.execute(
                f"SELECT 1 AS ok FROM {table} WHERE {column}=? LIMIT 1",
                [subject_id],
            ).fetchone()
            if row is not None:
                return True
        return False

    @staticmethod
    def _row_to_edge(mapping: Mapping[str, Any]) -> ArtifactEdge:
        try:
            body = json.loads(str(mapping.get("body_json") or "{}"))
        except json.JSONDecodeError as exc:
            raise DatabaseArtifactStoreIntegrityError(
                "poisoned edge body envelope"
            ) from exc
        if not isinstance(body, dict):
            raise DatabaseArtifactStoreIntegrityError(
                "poisoned edge body envelope"
            )
        return ArtifactEdge(
            edge_id=str(mapping["edge_id"]),
            source_id=str(mapping["source_id"]),
            target_id=str(mapping["target_id"]),
            edge_kind=str(mapping["edge_kind"]),
            created_at=str(mapping.get("created_at") or ""),
            body=MappingProxyType(body),
            admitted_sequence=int(mapping.get("admitted_sequence") or 0),
        )

    # -- projections / rebuild -----------------------------------------------

    def rebuild_projection(
        self,
        projection_kind: str = "catalog",
    ) -> ArtifactProjection:
        """Rebuild a projection solely from admitted evidence rows."""

        kind = _text(projection_kind, "projection_kind")
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT * FROM admitted_evidence
                ORDER BY sequence ASC
                """
            ).fetchall()
            events: list[dict[str, Any]] = []
            max_seq = 0
            for row in rows:
                mapping = _row_mapping(row)
                try:
                    body = json.loads(str(mapping.get("body_json") or "{}"))
                    provenance = json.loads(
                        str(mapping.get("provenance_json") or "{}")
                    )
                except json.JSONDecodeError as exc:
                    raise DatabaseArtifactStoreIntegrityError(
                        "poisoned admitted evidence envelope"
                    ) from exc
                seq = int(mapping.get("sequence") or 0)
                max_seq = max(max_seq, seq)
                events.append(
                    {
                        "evidence_id": str(mapping["evidence_id"]),
                        "evidence_kind": str(mapping["evidence_kind"]),
                        "subject_id": str(mapping["subject_id"]),
                        "content_digest": str(mapping["content_digest"]),
                        "sequence": seq,
                        "body": body if isinstance(body, dict) else {},
                        "provenance": (
                            provenance if isinstance(provenance, dict) else {}
                        ),
                    }
                )

            # Materialize catalog projections from evidence, not from export files.
            artifacts: list[dict[str, Any]] = []
            datasets: list[dict[str, Any]] = []
            edges: list[dict[str, Any]] = []
            blobs: list[dict[str, Any]] = []
            for event in events:
                kind_name = event["evidence_kind"]
                if kind_name == EvidenceKind.ARTIFACT.value:
                    artifacts.append(
                        {
                            "artifact_id": event["subject_id"],
                            "content_digest": event["content_digest"],
                            "sequence": event["sequence"],
                            **event["body"],
                        }
                    )
                elif kind_name == EvidenceKind.DATASET.value:
                    datasets.append(
                        {
                            "dataset_id": event["subject_id"],
                            "content_digest": event["content_digest"],
                            "sequence": event["sequence"],
                            **event["body"],
                        }
                    )
                elif kind_name == EvidenceKind.EDGE.value:
                    edges.append(
                        {
                            "edge_id": event["subject_id"],
                            "content_digest": event["content_digest"],
                            "sequence": event["sequence"],
                            **event["body"],
                        }
                    )
                elif kind_name == EvidenceKind.BLOB.value:
                    blobs.append(
                        {
                            "blob_digest": event["content_digest"],
                            "sequence": event["sequence"],
                            **event["body"],
                        }
                    )

            body = {
                "projection_kind": kind,
                "snapshot_id": self._snapshot_id,
                "artifact_count": len(artifacts),
                "dataset_count": len(datasets),
                "edge_count": len(edges),
                "blob_count": len(blobs),
                "artifacts": artifacts,
                "datasets": datasets,
                "edges": edges,
                "blobs": blobs,
                "rebuilt_from_admitted_evidence": True,
                "authority": _AUTHORITY_DATABASE,
            }
            content_digest = _sha256_hex(
                _canonical_json(body).encode("utf-8")
            )
            stamp = _utc_iso()
            projection_id = f"projection:{kind}:{content_digest}"
            connection.execute(
                """
                INSERT OR REPLACE INTO artifact_projections(
                    projection_id, projection_kind, content_digest,
                    rebuilt_from_sequence, rebuilt_at, body_json, authority
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    projection_id,
                    kind,
                    content_digest,
                    max_seq,
                    stamp,
                    _canonical_json(body),
                    _AUTHORITY_DATABASE,
                ],
            )
            self._commit_if_idle(connection)
            return ArtifactProjection(
                projection_id=projection_id,
                projection_kind=kind,
                content_digest=content_digest,
                rebuilt_from_sequence=max_seq,
                rebuilt_at=stamp,
                body=MappingProxyType(body),
            )

    def get_projection(self, projection_id: str) -> ArtifactProjection | None:
        selected = _text(projection_id, "projection_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM artifact_projections WHERE projection_id=?",
                [selected],
            ).fetchone()
            if row is None:
                return None
            mapping = _row_mapping(row)
            try:
                body = json.loads(str(mapping.get("body_json") or "{}"))
            except json.JSONDecodeError as exc:
                raise DatabaseArtifactStoreIntegrityError(
                    "poisoned projection envelope"
                ) from exc
            if not isinstance(body, dict):
                raise DatabaseArtifactStoreIntegrityError(
                    "poisoned projection envelope"
                )
            return ArtifactProjection(
                projection_id=str(mapping["projection_id"]),
                projection_kind=str(mapping["projection_kind"]),
                content_digest=str(mapping["content_digest"]),
                rebuilt_from_sequence=int(
                    mapping.get("rebuilt_from_sequence") or 0
                ),
                rebuilt_at=str(mapping.get("rebuilt_at") or ""),
                body=MappingProxyType(body),
                authority=str(mapping.get("authority") or _AUTHORITY_DATABASE),
            )

    # -- exports (non-authoritative) -----------------------------------------

    def export_json(
        self,
        export_path: Path | str,
        *,
        include_datasets: bool = True,
    ) -> ArtifactExportReceipt:
        """Render a JSON export. File freshness never grants authority."""

        path = Path(export_path)
        with self._lock:
            artifacts = [
                record.to_dict() for record in self.list_artifacts(limit=4_096)
            ]
            datasets = (
                [record.to_dict() for record in self.list_datasets(limit=4_096)]
                if include_datasets
                else []
            )
            edges = [edge.to_dict() for edge in self.list_edges(limit=4_096)]
            document = {
                "schema": ARTIFACT_EXPORT_RECEIPT_SCHEMA,
                "authority": _AUTHORITY_EXPORT_ONLY,
                "authoritative": False,
                "snapshot_id": self._snapshot_id,
                "artifacts": artifacts,
                "datasets": datasets,
                "edges": edges,
            }
            encoded = (
                json.dumps(
                    document,
                    sort_keys=True,
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
            content_digest = _sha256_hex(encoded)
            _atomic_write_bytes(path, encoded)
            stamp = _utc_iso()
            export_id = "export:" + content_digest
            connection = self._require()
            connection.execute(
                """
                INSERT OR REPLACE INTO artifact_exports(
                    export_id, export_path, export_format, content_digest,
                    subject_count, recorded_at, authority, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    export_id,
                    str(path),
                    "json",
                    content_digest,
                    len(artifacts) + len(datasets),
                    stamp,
                    _AUTHORITY_EXPORT_ONLY,
                    _canonical_json(
                        {
                            "artifact_count": len(artifacts),
                            "dataset_count": len(datasets),
                            "edge_count": len(edges),
                        }
                    ),
                ],
            )
            self._commit_if_idle(connection)
            return ArtifactExportReceipt(
                export_id=export_id,
                export_path=str(path),
                export_format="json",
                content_digest=content_digest,
                subject_count=len(artifacts) + len(datasets),
                recorded_at=stamp,
            )

    def authority_unaffected_by_export_deletion(
        self, export_path: Path | str
    ) -> bool:
        """Delete an export file and confirm database authority is unchanged."""

        path = Path(export_path)
        with self._lock:
            before_artifacts = [
                record.to_dict() for record in self.list_artifacts(limit=4_096)
            ]
            before_datasets = [
                record.to_dict() for record in self.list_datasets(limit=4_096)
            ]
            if path.exists():
                path.unlink()
            after_artifacts = [
                record.to_dict() for record in self.list_artifacts(limit=4_096)
            ]
            after_datasets = [
                record.to_dict() for record in self.list_datasets(limit=4_096)
            ]
            return (
                before_artifacts == after_artifacts
                and before_datasets == after_datasets
                and not path.exists()
            )

    def file_freshness_is_non_authoritative(self) -> bool:
        """Document that JSON/Parquet mtime never selects authority."""

        return True

    def admitted_evidence_count(self) -> int:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM admitted_evidence"
            ).fetchone()
            return int(_row_mapping(row).get("n") or 0)

    def list_admitted_evidence(
        self, *, limit: int = 1_024
    ) -> list[dict[str, Any]]:
        limit = min(_positive_int(limit, "limit"), 16_384)
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT * FROM admitted_evidence
                ORDER BY sequence ASC
                LIMIT ?
                """,
                [limit],
            ).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                mapping = _row_mapping(row)
                try:
                    body = json.loads(str(mapping.get("body_json") or "{}"))
                    provenance = json.loads(
                        str(mapping.get("provenance_json") or "{}")
                    )
                except json.JSONDecodeError as exc:
                    raise DatabaseArtifactStoreIntegrityError(
                        "poisoned admitted evidence envelope"
                    ) from exc
                results.append(
                    {
                        "schema": ADMITTED_EVIDENCE_SCHEMA,
                        "evidence_id": str(mapping["evidence_id"]),
                        "evidence_kind": str(mapping["evidence_kind"]),
                        "subject_id": str(mapping["subject_id"]),
                        "content_digest": str(mapping["content_digest"]),
                        "admitted_at": str(mapping.get("admitted_at") or ""),
                        "sequence": int(mapping.get("sequence") or 0),
                        "body": body if isinstance(body, dict) else {},
                        "provenance": (
                            provenance if isinstance(provenance, dict) else {}
                        ),
                    }
                )
            return results


def open_database_artifact_store(
    database_path: Path | str,
    *,
    cas_root: Path | str | None = None,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    quotas: ArtifactQuotaPolicy | Mapping[str, Any] | None = None,
    auto_redact: bool = True,
) -> DatabaseArtifactStore:
    """Open and return a :class:`DatabaseArtifactStore`."""

    store = DatabaseArtifactStore(
        database_path,
        cas_root=cas_root,
        snapshot_id=snapshot_id,
        quotas=quotas,
        auto_redact=auto_redact,
    )
    return store.open()


__all__ = [
    "ADMITTED_EVIDENCE_SCHEMA",
    "ARTIFACT_EDGE_SCHEMA",
    "ARTIFACT_EXPORT_RECEIPT_SCHEMA",
    "ARTIFACT_PROJECTION_SCHEMA",
    "ARTIFACT_RECORD_SCHEMA",
    "BLOB_REFERENCE_SCHEMA",
    "DATABASE_ARTIFACT_STORE_INTERFACE",
    "DATABASE_ARTIFACT_STORE_SCHEMA",
    "DATASET_RECORD_SCHEMA",
    "DEFAULT_MAX_ARTIFACTS",
    "DEFAULT_MAX_BLOB_BYTES",
    "DEFAULT_MAX_DATASETS",
    "DEFAULT_MAX_EDGES",
    "DEFAULT_MAX_GRAPH_DEGREE",
    "DEFAULT_MAX_METADATA_BYTES",
    "DEFAULT_SNAPSHOT_ID",
    "ArtifactEdge",
    "ArtifactExportReceipt",
    "ArtifactOutcome",
    "ArtifactProjection",
    "ArtifactQuotaPolicy",
    "ArtifactRecord",
    "BlobReference",
    "DatabaseArtifactStore",
    "DatabaseArtifactStoreBoundsError",
    "DatabaseArtifactStoreConflictError",
    "DatabaseArtifactStoreError",
    "DatabaseArtifactStoreIntegrityError",
    "DatabaseArtifactStoreNotOpenError",
    "DatabaseArtifactStoreQuotaError",
    "DatasetRecord",
    "DuckDBUnavailableError",
    "EvidenceKind",
    "REDACTION_MARKER",
    "RetentionClass",
    "duckdb_available",
    "open_database_artifact_store",
]
