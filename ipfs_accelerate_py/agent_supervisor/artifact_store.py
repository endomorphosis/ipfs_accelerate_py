"""Dual JSON/DuckDB storage and bounded queries for supervisor artifacts.

JSON remains the portable interchange format.  Each write also materializes a
normalized DuckDB sidecar so schedulers and operators can inspect a small,
typed projection without loading a complete planning graph into a prompt.
"""

from __future__ import annotations

import argparse
import base64
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Iterator, Mapping, Sequence

from .supervisor_v2_contracts import MAX_PROJECTION_BYTES, MAX_RECEIPT_BYTES

BUNDLE_INDEX_KIND = "bundle_planning_index"
SCHEDULER_MANIFEST_KIND = "scheduler_manifest"
CODE_EVIDENCE_GRAPH_KIND = "code_evidence_graph"
EVIDENCE_GRAPH_KIND = CODE_EVIDENCE_GRAPH_KIND
PROOF_METRICS_KIND = "proof_metrics"
PROOF_ATTESTATION_KIND = "proof_attestations"

# These fields remain available in the bundle-index DuckDB tables for bounded
# evidence queries. The scheduler does not need to materialize their repeated
# multi-megabyte values to rebuild dependency and conflict plans.
BUNDLE_PLANNING_BUNDLE_OMIT_FIELDS = (
    "conflict_graph",
    "conflict_planning_decisions",
    "dependency_dag",
    "task_conflict_graph",
    "task_dependency_graph",
    "task_planning_graph",
    "todo_vector_summary",
)
BUNDLE_PLANNING_TASK_OMIT_FIELDS = (
    "conflict_decisions",
    "conflict_edges",
    "conflict_surface",
    "coverage_inputs",
    "dependency_dag",
    "task_conflict_graph",
    "task_dependency_graph",
    "task_planning_graph",
)
PROOF_ATTESTATION_STORE_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.proof-attestation-store@1"
)
PROOF_ATTESTATIONS_KIND = PROOF_ATTESTATION_KIND
PROOF_ATTESTATION_ARTIFACT_KIND = PROOF_ATTESTATION_KIND
PROOF_ATTESTATION_ARTIFACT_SCHEMA = PROOF_ATTESTATION_STORE_SCHEMA
QUERY_SCHEMA = "ipfs_accelerate_py.agent_supervisor.queryable_artifact@2"
MAX_QUERY_ROWS = 1_000
MAX_GRAPH_QUERY_HOPS = 8
ARTIFACT_LOCK_TIMEOUT_SECONDS = 300.0
DUCKDB_ARTIFACT_THREADS = 2
DUCKDB_ARTIFACT_MEMORY_LIMIT = "1GB"
MAX_INLINE_GRAPH_ITEMS = 128
MAX_INLINE_COVERAGE_TASKS = 128
MAX_ADAPTER_READ_FIELDS = 64
MAX_ADAPTER_READ_BYTES = 256 * 1024
MAX_ADAPTER_QUERY_BYTES = 1024 * 1024

BOUNDED_ARTIFACT_STORE_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.bounded-artifact-store@1"
)
BOUNDED_ARTIFACT_MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.bounded-artifact-manifest@1"
)
BOUNDED_BLOB_REFERENCE_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.bounded-blob-reference@1"
)
BOUNDED_PROJECTION_SCHEMA: Final = (
    "ipfs_accelerate_py.agent_supervisor.bounded-projection@1"
)
DEFAULT_ARTIFACT_STORE_MAX_BYTES: Final = 512 * 1024 * 1024
DEFAULT_ARTIFACT_STORE_MAX_BLOBS: Final = 16_384
DEFAULT_ARTIFACT_STORE_MAX_PROJECTIONS: Final = 4_096
DEFAULT_ARTIFACT_BLOB_MAX_BYTES: Final = 64 * 1024 * 1024
DEFAULT_ARTIFACT_COMPACTION_BATCH: Final = 128
DEFAULT_NEGATIVE_TTL_SECONDS: Final = 5 * 60
DEFAULT_INCONCLUSIVE_TTL_SECONDS: Final = 5 * 60
DEFAULT_MAX_ARTIFACT_TTL_SECONDS: Final = 30 * 24 * 60 * 60
MAX_ROUTINE_PROJECTION_BYTES: Final = MAX_PROJECTION_BYTES
RECEIPT_MAX_BYTES: Final = MAX_RECEIPT_BYTES
ROUTINE_PROJECTION_MAX_BYTES: Final = MAX_PROJECTION_BYTES

_EMBEDDED_BODY_FIELDS = frozenset(
    {
        "body",
        "bytes",
        "checkpoint",
        "checkpoints",
        "decoded_model_text",
        "decoded_text",
        "full_source",
        "model_output_text",
        "model_text",
        "nested_artifact_graph",
        "nested_artifact_graphs",
        "proof_trace",
        "proof_traces",
        "source_bodies",
        "source_body",
        "source_text",
    }
)
_GRAPH_BODY_FIELDS = frozenset(
    {"artifact_graph", "artifact_graphs", "evidence_graph", "nested_graph"}
)
_REFERENCE_BODY_FIELDS = frozenset(
    {
        "body",
        "bytes",
        "content",
        "contents",
        "data",
        "payload",
        "source",
        "text",
    }
)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_READ_ONLY_SQL = re.compile(r"^(?:select|with|describe|show)\b", re.IGNORECASE)
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class QueryArtifactPaths:
    """The portable and queryable representations of one artifact."""

    json_path: Path
    duckdb_path: Path


@dataclass(frozen=True)
class QueryableArtifactReference:
    """Verified, body-free identity for a paired queryable artifact.

    ``digest`` addresses the portable artifact's logical JSON content and is
    deliberately independent of its filesystem location. ``source_sha256``
    binds the reference to one exact on-disk JSON generation, including its
    query-store descriptor, so an adapter can reject source replacement before
    returning projected data.
    """

    artifact_id: str
    digest: str
    path: str
    kind: str
    schema: str
    size_bytes: int
    source_sha256: str
    duckdb_path: str

    def __post_init__(self) -> None:
        if not _SHA256_DIGEST.fullmatch(self.digest):
            raise ValueError("queryable artifact digest must be sha256:<hex>")
        if self.artifact_id != f"queryable-artifact:{self.digest}":
            raise ValueError("queryable artifact identity does not match its digest")
        if not _SHA256_HEX.fullmatch(self.source_sha256):
            raise ValueError("queryable artifact source_sha256 must be lowercase hex")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 1
        ):
            raise ValueError("queryable artifact size_bytes must be positive")
        for name in ("path", "duckdb_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"queryable artifact {name} is required")
            if not Path(value).is_absolute():
                raise ValueError(f"queryable artifact {name} must be absolute")
        for name in ("kind", "schema"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"queryable artifact {name} is required")

    def to_dict(self) -> dict[str, Any]:
        """Return complete verification metadata without the artifact body."""

        return {
            "artifact_id": self.artifact_id,
            "digest": self.digest,
            "path": self.path,
            "kind": self.kind,
            "schema": self.schema,
            "size_bytes": self.size_bytes,
            "source_sha256": self.source_sha256,
            "duckdb_path": self.duckdb_path,
        }

    def to_artifact_reference(self) -> dict[str, Any]:
        """Return the shallow shape accepted by common cache envelopes."""

        return {
            "artifact_id": self.artifact_id,
            "digest": self.digest,
            "path": self.path,
            "kind": self.kind,
            "schema": self.schema,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QueryableArtifactReference":
        if not isinstance(value, Mapping):
            raise ValueError("queryable artifact reference must be an object")
        required = {
            "artifact_id",
            "digest",
            "path",
            "kind",
            "schema",
            "size_bytes",
            "source_sha256",
            "duckdb_path",
        }
        missing = sorted(required.difference(value))
        unknown = sorted(set(value).difference(required))
        if missing:
            raise ValueError(
                "queryable artifact reference is missing fields: "
                + ", ".join(missing)
            )
        if unknown:
            raise ValueError(
                "queryable artifact reference has unknown fields: "
                + ", ".join(unknown)
            )
        return cls(
            artifact_id=value["artifact_id"],
            digest=value["digest"],
            path=value["path"],
            kind=value["kind"],
            schema=value["schema"],
            size_bytes=value["size_bytes"],
            source_sha256=value["source_sha256"],
            duckdb_path=value["duckdb_path"],
        )


class BoundedPersistenceError(RuntimeError):
    """Base error for bounded persistence failures."""


class ArtifactPayloadTooLarge(BoundedPersistenceError, ValueError):
    """A receipt, projection, or individual blob exceeded its hard bound."""


class ArtifactQuotaExceeded(BoundedPersistenceError):
    """The configured quota cannot admit an object without unsafe eviction."""


class ArtifactBlobIntegrityError(BoundedPersistenceError, ValueError):
    """A referenced immutable blob is absent, truncated, or corrupt."""


class RetentionClass(str, Enum):
    """Eviction class, ordered from easiest to hardest to discard."""

    EPHEMERAL = "ephemeral"
    NEGATIVE = "negative"
    ROUTINE = "routine"
    CHECKPOINT = "checkpoint"
    AUTHORITATIVE = "authoritative"
    PINNED = "pinned"

    TRANSIENT = "ephemeral"
    FAILED = "negative"
    DURABLE = "authoritative"


class ArtifactOutcome(str, Enum):
    SUCCESSFUL = "successful"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"

    @classmethod
    def coerce(cls, value: "ArtifactOutcome | str") -> "ArtifactOutcome":
        if isinstance(value, cls):
            return value
        normalized = str(value or "").strip().casefold().replace("-", "_")
        aliases = {
            "complete": cls.SUCCESSFUL,
            "completed": cls.SUCCESSFUL,
            "success": cls.SUCCESSFUL,
            "succeeded": cls.SUCCESSFUL,
            "ok": cls.SUCCESSFUL,
            "failed": cls.NEGATIVE,
            "failure": cls.NEGATIVE,
            "error": cls.NEGATIVE,
            "timed_out": cls.NEGATIVE,
            "timeout": cls.NEGATIVE,
            "partial": cls.INCONCLUSIVE,
            "unknown": cls.INCONCLUSIVE,
        }
        try:
            return aliases.get(normalized, cls(normalized))
        except ValueError as exc:
            raise ValueError(
                "outcome must be successful, negative, or inconclusive"
            ) from exc

    @property
    def can_complete(self) -> bool:
        return self is ArtifactOutcome.SUCCESSFUL


@dataclass(frozen=True)
class ArtifactQuotaPolicy:
    """Aggregate and per-object policy for the durable artifact store."""

    max_bytes: int = DEFAULT_ARTIFACT_STORE_MAX_BYTES
    max_blobs: int = DEFAULT_ARTIFACT_STORE_MAX_BLOBS
    max_projections: int = DEFAULT_ARTIFACT_STORE_MAX_PROJECTIONS
    max_blob_bytes: int = DEFAULT_ARTIFACT_BLOB_MAX_BYTES
    max_receipt_bytes: int = MAX_RECEIPT_BYTES
    max_projection_bytes: int = MAX_PROJECTION_BYTES
    min_free_bytes: int = 0
    compaction_batch_size: int = DEFAULT_ARTIFACT_COMPACTION_BATCH
    negative_ttl_seconds: int = DEFAULT_NEGATIVE_TTL_SECONDS
    inconclusive_ttl_seconds: int = DEFAULT_INCONCLUSIVE_TTL_SECONDS
    max_ttl_seconds: int = DEFAULT_MAX_ARTIFACT_TTL_SECONDS

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            minimum = 0 if name == "min_free_bytes" else 1
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum
            ):
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer")
        if self.max_blob_bytes > self.max_bytes:
            raise ValueError("max_blob_bytes cannot exceed max_bytes")
        if self.max_receipt_bytes > MAX_RECEIPT_BYTES:
            raise ValueError(
                f"max_receipt_bytes cannot exceed {MAX_RECEIPT_BYTES}"
            )
        if self.max_projection_bytes > MAX_PROJECTION_BYTES:
            raise ValueError(
                f"max_projection_bytes cannot exceed {MAX_PROJECTION_BYTES}"
            )
        if self.max_receipt_bytes > self.max_projection_bytes:
            raise ValueError(
                "max_receipt_bytes cannot exceed max_projection_bytes"
            )
        for name in ("negative_ttl_seconds", "inconclusive_ttl_seconds"):
            if getattr(self, name) > self.max_ttl_seconds:
                raise ValueError(f"{name} cannot exceed max_ttl_seconds")

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


ArtifactStoreQuota = ArtifactQuotaPolicy
PersistenceQuotaPolicy = ArtifactQuotaPolicy
ArtifactStoreConfig = ArtifactQuotaPolicy


@dataclass(frozen=True)
class BlobReference:
    """Shallow, location-independent reference to one immutable blob."""

    artifact_id: str
    digest: str
    size_bytes: int
    kind: str
    media_type: str = "application/octet-stream"
    schema: str = BOUNDED_BLOB_REFERENCE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != BOUNDED_BLOB_REFERENCE_SCHEMA:
            raise ArtifactBlobIntegrityError(
                "unsupported bounded blob reference schema"
            )
        if not _SHA256_DIGEST.fullmatch(str(self.digest)):
            raise ArtifactBlobIntegrityError(
                "blob digest must be sha256:<lowercase hex>"
            )
        if self.artifact_id != f"blob:{self.digest}":
            raise ArtifactBlobIntegrityError(
                "blob artifact_id does not match its digest"
            )
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise ArtifactBlobIntegrityError(
                "blob size_bytes must be a non-negative integer"
            )
        for name in ("kind", "media_type"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ArtifactBlobIntegrityError(f"blob {name} is required")

    @property
    def blob_id(self) -> str:
        return self.artifact_id

    @property
    def cid(self) -> str:
        return self.artifact_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "kind": self.kind,
            "media_type": self.media_type,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BlobReference":
        if not isinstance(value, Mapping):
            raise ArtifactBlobIntegrityError("blob reference must be an object")
        allowed = {
            "schema",
            "artifact_id",
            "blob_id",
            "digest",
            "size_bytes",
            "kind",
            "media_type",
            "retention_class",
            "outcome",
            "created_at_ms",
            "last_accessed_at_ms",
            "expires_at_ms",
            "references",
        }
        if set(value).difference(allowed):
            raise ArtifactBlobIntegrityError(
                "blob reference contains unsupported fields"
            )
        artifact_id = value.get("artifact_id") or value.get("blob_id") or ""
        return cls(
            schema=str(value.get("schema") or BOUNDED_BLOB_REFERENCE_SCHEMA),
            artifact_id=str(artifact_id),
            digest=str(value.get("digest") or ""),
            size_bytes=value.get("size_bytes", -1),
            kind=str(value.get("kind") or "artifact"),
            media_type=str(
                value.get("media_type") or "application/octet-stream"
            ),
        )


ArtifactBlobReference = BlobReference
ContentReference = BlobReference


@dataclass(frozen=True)
class ProjectionReference:
    """Body-free identity of one bounded receipt or routine projection."""

    artifact_id: str
    digest: str
    size_bytes: int
    projection_kind: str
    retention_class: RetentionClass
    outcome: ArtifactOutcome
    created_at_ms: int
    expires_at_ms: int | None
    artifact_references: tuple[BlobReference, ...] = ()
    schema: str = BOUNDED_PROJECTION_SCHEMA

    @property
    def can_complete(self) -> bool:
        return self.outcome.can_complete

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "projection_kind": self.projection_kind,
            "retention_class": self.retention_class.value,
            "outcome": self.outcome.value,
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "artifact_references": [
                item.to_dict() for item in self.artifact_references
            ],
            "can_complete": self.can_complete,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProjectionReference":
        if not isinstance(value, Mapping):
            raise ArtifactBlobIntegrityError(
                "projection reference must be an object"
            )
        return cls(
            schema=str(value.get("schema") or BOUNDED_PROJECTION_SCHEMA),
            artifact_id=str(value.get("artifact_id") or ""),
            digest=str(value.get("digest") or ""),
            size_bytes=value.get("size_bytes", -1),
            projection_kind=str(value.get("projection_kind") or ""),
            retention_class=RetentionClass(
                str(value.get("retention_class") or RetentionClass.ROUTINE.value)
            ),
            outcome=ArtifactOutcome.coerce(
                str(value.get("outcome") or ArtifactOutcome.SUCCESSFUL.value)
            ),
            created_at_ms=value.get("created_at_ms", -1),
            expires_at_ms=value.get("expires_at_ms"),
            artifact_references=tuple(
                BlobReference.from_dict(item)
                for item in value.get("artifact_references", ())
            ),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.retention_class, RetentionClass):
            object.__setattr__(
                self,
                "retention_class",
                RetentionClass(str(self.retention_class)),
            )
        if not isinstance(self.outcome, ArtifactOutcome):
            object.__setattr__(
                self, "outcome", ArtifactOutcome.coerce(self.outcome)
            )
        if self.schema != BOUNDED_PROJECTION_SCHEMA:
            raise ArtifactBlobIntegrityError(
                "unsupported bounded projection schema"
            )
        if not _SHA256_DIGEST.fullmatch(str(self.digest)):
            raise ArtifactBlobIntegrityError(
                "projection digest must be sha256:<lowercase hex>"
            )
        if self.artifact_id != f"projection:{self.digest}":
            raise ArtifactBlobIntegrityError(
                "projection artifact_id does not match its digest"
            )
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 1
        ):
            raise ArtifactBlobIntegrityError(
                "projection size_bytes must be positive"
            )
        if (
            isinstance(self.created_at_ms, bool)
            or not isinstance(self.created_at_ms, int)
            or self.created_at_ms < 0
        ):
            raise ArtifactBlobIntegrityError(
                "projection created_at_ms must be non-negative"
            )
        if self.expires_at_ms is not None and (
            isinstance(self.expires_at_ms, bool)
            or not isinstance(self.expires_at_ms, int)
            or self.expires_at_ms <= self.created_at_ms
        ):
            raise ArtifactBlobIntegrityError(
                "projection expires_at_ms must follow created_at_ms"
            )
        if not self.outcome.can_complete and self.expires_at_ms is None:
            raise ArtifactBlobIntegrityError(
                "negative and inconclusive projections require finite expiry"
            )
        if not isinstance(self.projection_kind, str) or not self.projection_kind:
            raise ArtifactBlobIntegrityError("projection_kind is required")


BoundedProjectionReference = ProjectionReference
ArtifactRetentionClass = RetentionClass
PersistenceOutcome = ArtifactOutcome


@dataclass(frozen=True)
class ArtifactStoreMetrics:
    writes: int = 0
    blob_writes: int = 0
    deduplicated_blob_writes: int = 0
    projection_writes: int = 0
    reads: int = 0
    compactions: int = 0
    scanned: int = 0
    evictions: int = 0
    evicted_bytes: int = 0
    expired_evictions: int = 0
    quota_evictions: int = 0
    quota_rejections: int = 0
    disk_pressure_rejections: int = 0
    corruption_recoveries: int = 0
    manifest_recoveries: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class CompactionResult:
    scanned: int
    evicted: int
    evicted_bytes: int
    expired: int
    quota_evicted: int
    cursor: int
    quota_satisfied: bool
    evicted_artifact_ids: tuple[str, ...] = ()

    @property
    def evictions(self) -> int:
        return self.evicted

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "evicted": self.evicted,
            "evicted_bytes": self.evicted_bytes,
            "expired": self.expired,
            "quota_evicted": self.quota_evicted,
            "cursor": self.cursor,
            "quota_satisfied": self.quota_satisfied,
            "evicted_artifact_ids": list(self.evicted_artifact_ids),
        }


def _bounded_canonical_bytes(
    value: Any,
    *,
    maximum: int,
    label: str,
) -> bytes:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ArtifactBlobIntegrityError(
            f"{label} must contain canonical JSON values"
        ) from exc
    if len(payload) > maximum:
        raise ArtifactPayloadTooLarge(
            f"{label} exceeds {maximum} bytes"
        )
    return payload


def enforce_receipt_bound(value: Any) -> bytes:
    """Return canonical receipt bytes after enforcing the 256 KiB ceiling."""

    return _bounded_canonical_bytes(
        value, maximum=MAX_RECEIPT_BYTES, label="receipt"
    )


def enforce_projection_bound(value: Any) -> bytes:
    """Return canonical routine projection bytes after enforcing 1 MiB."""

    return _bounded_canonical_bytes(
        value, maximum=MAX_PROJECTION_BYTES, label="routine projection"
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory = -1
        if directory >= 0:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


_BOUNDED_STORE_LOCKS: dict[str, threading.RLock] = {}
_BOUNDED_STORE_LOCKS_GUARD = threading.Lock()


def _bounded_store_lock(path: Path) -> threading.RLock:
    identity = str(path.absolute())
    with _BOUNDED_STORE_LOCKS_GUARD:
        return _BOUNDED_STORE_LOCKS.setdefault(identity, threading.RLock())


class BoundedArtifactStore:
    """Crash-recoverable CAS for bounded projections and referenced bodies.

    Projection files contain only compact summaries and shallow blob
    references. Blob identities are derived solely from bytes, so deduplication
    and compaction never rewrite a surviving reference.
    """

    _retention_rank = {
        RetentionClass.EPHEMERAL: 0,
        RetentionClass.NEGATIVE: 1,
        RetentionClass.ROUTINE: 2,
        RetentionClass.CHECKPOINT: 3,
        RetentionClass.AUTHORITATIVE: 4,
        RetentionClass.PINNED: 5,
    }

    def __init__(
        self,
        path: Path | str,
        *,
        quotas: ArtifactQuotaPolicy | Mapping[str, Any] | None = None,
        quota: ArtifactQuotaPolicy | Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.time,
        eviction_observer: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if quotas is not None and quota is not None:
            raise ValueError("pass quotas or quota, not both")
        selected_quota = quotas if quotas is not None else quota
        self.quotas = (
            selected_quota
            if isinstance(selected_quota, ArtifactQuotaPolicy)
            else ArtifactQuotaPolicy(**dict(selected_quota or {}))
        )
        self.path = Path(path)
        self.blobs_path = self.path / "blobs" / "sha256"
        self.projections_path = self.path / "projections"
        self.manifest_path = self.path / "manifest.json"
        self.previous_manifest_path = self.path / "manifest.previous.json"
        self.eviction_log_path = self.path / "evictions.jsonl"
        self.lock_path = self.path / ".bounded-store.lock"
        for directory in (self.path, self.blobs_path, self.projections_path):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._clock = clock
        self._eviction_observer = eviction_observer
        self._thread_lock = _bounded_store_lock(self.lock_path)
        self._metrics_lock = threading.Lock()
        self._metric_values = {
            name: 0 for name in ArtifactStoreMetrics.__dataclass_fields__
        }
        self._closed = False
        with self._locked():
            self._manifest = self._load_or_recover_manifest()

    def __enter__(self) -> "BoundedArtifactStore":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _increment(self, name: str, amount: int = 1) -> None:
        with self._metrics_lock:
            self._metric_values[name] += amount

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            handle = self.lock_path.open("a+b")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()

    @staticmethod
    def _manifest_digest(value: Mapping[str, Any]) -> str:
        body = dict(value)
        body.pop("manifest_digest", None)
        return "sha256:" + hashlib.sha256(
            _bounded_canonical_bytes(
                body,
                maximum=DEFAULT_ARTIFACT_BLOB_MAX_BYTES,
                label="artifact manifest",
            )
        ).hexdigest()

    def _empty_manifest(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": BOUNDED_ARTIFACT_MANIFEST_SCHEMA,
            "generation": 0,
            "updated_at_ms": self._now_ms(),
            "compaction_cursor": 0,
            "blobs": {},
            "projections": {},
        }
        value["manifest_digest"] = self._manifest_digest(value)
        return value

    def _decode_manifest(self, path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if (
            not isinstance(value, dict)
            or value.get("schema") != BOUNDED_ARTIFACT_MANIFEST_SCHEMA
            or value.get("manifest_digest") != self._manifest_digest(value)
            or not isinstance(value.get("blobs"), dict)
            or not isinstance(value.get("projections"), dict)
        ):
            return None
        return value

    def _load_or_recover_manifest(self) -> dict[str, Any]:
        manifest = self._decode_manifest(self.manifest_path)
        if manifest is None:
            manifest = self._decode_manifest(self.previous_manifest_path)
            if manifest is not None:
                self._increment("manifest_recoveries")
        if manifest is None:
            manifest = self._empty_manifest()
            if self.manifest_path.exists() or self.previous_manifest_path.exists():
                self._increment("manifest_recoveries")
        changed = self._reconcile_files(manifest)
        if changed or not self.manifest_path.exists():
            self._write_manifest(manifest, preserve_previous=False)
        return manifest

    def _reconcile_files(self, manifest: dict[str, Any]) -> bool:
        changed = False
        projections = manifest["projections"]
        blobs = manifest["blobs"]
        for path in self.projections_path.glob("*/*.json"):
            try:
                wrapper = json.loads(path.read_text(encoding="utf-8"))
                reference = ProjectionReference.from_dict(wrapper["reference"])
                payload_bytes = _bounded_canonical_bytes(
                    wrapper["payload"],
                    maximum=self.quotas.max_projection_bytes,
                    label="stored projection",
                )
                if self._projection_digest(
                    wrapper["payload"],
                    projection_kind=reference.projection_kind,
                    retention_class=reference.retention_class,
                    outcome=reference.outcome,
                ) != reference.digest:
                    raise ArtifactBlobIntegrityError(
                        "stored projection digest mismatch"
                    )
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                try:
                    path.unlink()
                except OSError:
                    pass
                self._increment("corruption_recoveries")
                changed = True
                continue
            if reference.artifact_id not in projections:
                projections[reference.artifact_id] = self._projection_metadata(
                    reference
                )
                changed = True
            for blob_ref in reference.artifact_references:
                metadata = blobs.get(blob_ref.artifact_id)
                if metadata is not None:
                    references = set(metadata.get("references", ()))
                    if reference.artifact_id not in references:
                        references.add(reference.artifact_id)
                        metadata["references"] = sorted(references)
                        changed = True
        for artifact_id in tuple(projections):
            try:
                reference = ProjectionReference.from_dict(projections[artifact_id])
            except (TypeError, ValueError):
                projections.pop(artifact_id, None)
                changed = True
                continue
            if not self._projection_path(reference).exists():
                projections.pop(artifact_id, None)
                for metadata in blobs.values():
                    references = set(metadata.get("references", ()))
                    if artifact_id in references:
                        references.discard(artifact_id)
                        metadata["references"] = sorted(references)
                changed = True
        for path in self.blobs_path.glob("*/*.blob"):
            digest = "sha256:" + path.stem
            artifact_id = f"blob:{digest}"
            if artifact_id not in blobs:
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                blobs[artifact_id] = {
                    "schema": BOUNDED_BLOB_REFERENCE_SCHEMA,
                    "artifact_id": artifact_id,
                    "digest": digest,
                    "size_bytes": size,
                    "kind": "recovered",
                    "media_type": "application/octet-stream",
                    "retention_class": RetentionClass.ROUTINE.value,
                    "outcome": ArtifactOutcome.SUCCESSFUL.value,
                    "created_at_ms": self._now_ms(),
                    "last_accessed_at_ms": self._now_ms(),
                    "expires_at_ms": None,
                    "references": [],
                }
                changed = True
        for artifact_id in tuple(blobs):
            try:
                reference = BlobReference.from_dict(blobs[artifact_id])
            except (TypeError, ValueError):
                blobs.pop(artifact_id, None)
                changed = True
                continue
            if not self._blob_path(reference).exists():
                blobs.pop(artifact_id, None)
                changed = True
                continue
            owners = set(blobs[artifact_id].get("references", ()))
            live_owners = owners.intersection(projections)
            if owners != live_owners:
                blobs[artifact_id]["references"] = sorted(live_owners)
                changed = True
        return changed

    def _write_manifest(
        self,
        manifest: dict[str, Any],
        *,
        preserve_previous: bool = True,
    ) -> None:
        manifest["generation"] = int(manifest.get("generation", 0)) + 1
        manifest["updated_at_ms"] = self._now_ms()
        manifest["manifest_digest"] = self._manifest_digest(manifest)
        encoded = _bounded_canonical_bytes(
            manifest,
            maximum=DEFAULT_ARTIFACT_BLOB_MAX_BYTES,
            label="artifact manifest",
        ) + b"\n"
        if preserve_previous and self.manifest_path.exists():
            current = self._decode_manifest(self.manifest_path)
            if current is not None:
                _atomic_write_bytes(
                    self.previous_manifest_path,
                    self.manifest_path.read_bytes(),
                )
        _atomic_write_bytes(self.manifest_path, encoded)

    @staticmethod
    def _digest_hex(value: str, prefix: str) -> str:
        if not isinstance(value, str) or not value.startswith(prefix):
            raise ArtifactBlobIntegrityError("artifact identity is not canonical")
        digest = value.removeprefix(prefix)
        if len(digest) != 64 or not _SHA256_HEX.fullmatch(digest):
            raise ArtifactBlobIntegrityError("artifact identity is not canonical")
        return digest

    def _blob_path(self, reference: BlobReference | str) -> Path:
        artifact_id = (
            reference.artifact_id
            if isinstance(reference, BlobReference)
            else reference
        )
        digest = self._digest_hex(artifact_id, "blob:sha256:")
        return self.blobs_path / digest[:2] / f"{digest}.blob"

    def _projection_path(
        self, reference: ProjectionReference | str
    ) -> Path:
        artifact_id = (
            reference.artifact_id
            if isinstance(reference, ProjectionReference)
            else reference
        )
        digest = self._digest_hex(artifact_id, "projection:sha256:")
        return self.projections_path / digest[:2] / f"{digest}.json"

    def _coerce_retention(
        self, value: RetentionClass | str
    ) -> RetentionClass:
        return value if isinstance(value, RetentionClass) else RetentionClass(str(value))

    def _expiry(
        self,
        outcome: ArtifactOutcome,
        ttl_seconds: int | None,
        now_ms: int,
    ) -> int | None:
        if ttl_seconds is not None and (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, int)
            or ttl_seconds < 1
        ):
            raise ValueError("ttl_seconds must be a positive integer or None")
        if outcome is ArtifactOutcome.NEGATIVE:
            ttl_seconds = ttl_seconds or self.quotas.negative_ttl_seconds
        elif outcome is ArtifactOutcome.INCONCLUSIVE:
            ttl_seconds = ttl_seconds or self.quotas.inconclusive_ttl_seconds
        if ttl_seconds is None:
            return None
        return now_ms + min(ttl_seconds, self.quotas.max_ttl_seconds) * 1000

    @staticmethod
    def _blob_bytes(value: Any) -> tuple[bytes, str]:
        if isinstance(value, bytes):
            return value, "application/octet-stream"
        if isinstance(value, bytearray):
            return bytes(value), "application/octet-stream"
        if isinstance(value, str):
            return value.encode("utf-8"), "text/plain; charset=utf-8"
        return (
            _bounded_canonical_bytes(
                value,
                maximum=DEFAULT_ARTIFACT_BLOB_MAX_BYTES,
                label="artifact blob",
            ),
            "application/json",
        )

    def _assert_open(self) -> None:
        if self._closed:
            raise BoundedPersistenceError("artifact store is closed")

    def _disk_pressure(self, additional_bytes: int) -> bool:
        if self.quotas.min_free_bytes <= 0:
            return False
        try:
            free = shutil.disk_usage(self.path).free
        except OSError:
            return False
        return free - additional_bytes < self.quotas.min_free_bytes

    def _usage(self) -> tuple[int, int, int]:
        blobs = self._manifest["blobs"]
        projections = self._manifest["projections"]
        total = sum(
            int(item.get("size_bytes", 0)) for item in blobs.values()
        ) + sum(
            int(item.get("size_bytes", 0)) for item in projections.values()
        )
        return total, len(blobs), len(projections)

    def _has_capacity(
        self,
        *,
        additional_bytes: int = 0,
        additional_blobs: int = 0,
        additional_projections: int = 0,
    ) -> bool:
        total, blobs, projections = self._usage()
        return (
            total + additional_bytes <= self.quotas.max_bytes
            and blobs + additional_blobs <= self.quotas.max_blobs
            and projections + additional_projections
            <= self.quotas.max_projections
            and not self._disk_pressure(additional_bytes)
        )

    def _ensure_capacity(
        self,
        *,
        additional_bytes: int = 0,
        additional_blobs: int = 0,
        additional_projections: int = 0,
    ) -> None:
        if self._has_capacity(
            additional_bytes=additional_bytes,
            additional_blobs=additional_blobs,
            additional_projections=additional_projections,
        ):
            return
        if self._disk_pressure(additional_bytes):
            # Disk-reserve pressure is not evidence that live supervisor state
            # should be discarded. Reclaim one bounded batch of expired
            # records, then degrade the new write if the reserve is still low.
            self._compact_locked(
                max_items=self.quotas.compaction_batch_size,
                force_quota=False,
                reserve_bytes=additional_bytes,
                reserve_blobs=additional_blobs,
                reserve_projections=additional_projections,
            )
            if self._disk_pressure(additional_bytes):
                self._increment("disk_pressure_rejections")
                raise ArtifactQuotaExceeded(
                    "artifact write rejected by the configured disk-free reserve"
                )
        remaining = self.quotas.compaction_batch_size
        while remaining > 0:
            result = self._compact_locked(
                max_items=remaining,
                force_quota=True,
                reserve_bytes=additional_bytes,
                reserve_blobs=additional_blobs,
                reserve_projections=additional_projections,
            )
            remaining -= max(1, result.scanned)
            if result.quota_satisfied or result.evicted == 0:
                break
        if self._has_capacity(
            additional_bytes=additional_bytes,
            additional_blobs=additional_blobs,
            additional_projections=additional_projections,
        ):
            return
        if self._disk_pressure(additional_bytes):
            self._increment("disk_pressure_rejections")
            raise ArtifactQuotaExceeded(
                "artifact write rejected by the configured disk-free reserve"
            )
        self._increment("quota_rejections")
        raise ArtifactQuotaExceeded(
            "artifact write exceeds aggregate persistence quota"
        )

    def put_blob(
        self,
        value: Any,
        *,
        kind: str = "artifact",
        retention_class: RetentionClass | str = RetentionClass.ROUTINE,
        retention: RetentionClass | str | None = None,
        outcome: ArtifactOutcome | str = ArtifactOutcome.SUCCESSFUL,
        ttl_seconds: int | None = None,
        media_type: str | None = None,
    ) -> BlobReference:
        """Store bytes once and return a shallow content-addressed reference."""

        self._assert_open()
        data, inferred_media_type = self._blob_bytes(value)
        if len(data) > self.quotas.max_blob_bytes:
            raise ArtifactPayloadTooLarge(
                f"artifact blob exceeds {self.quotas.max_blob_bytes} bytes"
            )
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        reference = BlobReference(
            artifact_id=f"blob:{digest}",
            digest=digest,
            size_bytes=len(data),
            kind=str(kind or "artifact"),
            media_type=str(media_type or inferred_media_type),
        )
        record_outcome = ArtifactOutcome.coerce(outcome)
        selected_retention = self._coerce_retention(
            retention if retention is not None else retention_class
        )
        if not record_outcome.can_complete and selected_retention not in {
            RetentionClass.EPHEMERAL,
            RetentionClass.NEGATIVE,
        }:
            selected_retention = RetentionClass.NEGATIVE
        now_ms = self._now_ms()
        expires_at_ms = self._expiry(record_outcome, ttl_seconds, now_ms)
        with self._locked():
            existing = self._manifest["blobs"].get(reference.artifact_id)
            if existing is not None:
                current = BlobReference.from_dict(existing)
                if (
                    current.digest != reference.digest
                    or current.size_bytes != reference.size_bytes
                    or not self.verify_blob(current)
                ):
                    raise ArtifactBlobIntegrityError(
                        "existing blob does not match its content identity"
                    )
                existing["last_accessed_at_ms"] = now_ms
                self._increment("deduplicated_blob_writes")
                return current
            self._ensure_capacity(
                additional_bytes=len(data), additional_blobs=1
            )
            try:
                _atomic_write_bytes(self._blob_path(reference), data)
            except OSError as exc:
                if exc.errno in {errno.ENOSPC, errno.EDQUOT}:
                    self._increment("disk_pressure_rejections")
                    raise ArtifactQuotaExceeded(
                        "artifact blob could not be persisted under disk pressure"
                    ) from exc
                raise
            self._manifest["blobs"][reference.artifact_id] = {
                **reference.to_dict(),
                "retention_class": selected_retention.value,
                "outcome": record_outcome.value,
                "created_at_ms": now_ms,
                "last_accessed_at_ms": now_ms,
                "expires_at_ms": expires_at_ms,
                "references": [],
            }
            self._write_manifest(self._manifest)
            self._increment("writes")
            self._increment("blob_writes")
        return reference

    store_blob = put_blob
    write_blob = put_blob

    @staticmethod
    def _is_shallow_reference(value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        return (
            ("artifact_id" in value or "blob_id" in value or "cid" in value)
            and "digest" in value
        )

    @staticmethod
    def _validate_shallow_reference(value: Mapping[str, Any]) -> dict[str, Any]:
        body_fields = set(value).intersection(_REFERENCE_BODY_FIELDS)
        if body_fields:
            raise ArtifactBlobIntegrityError(
                "artifact references cannot recursively embed bodies: "
                + ", ".join(sorted(body_fields))
            )
        for item in value.values():
            if isinstance(item, Mapping) and set(item).intersection(
                _REFERENCE_BODY_FIELDS
            ):
                raise ArtifactBlobIntegrityError(
                    "artifact references cannot contain nested bodies"
                )
        return dict(value)

    def project_payload(
        self,
        payload: Any,
        *,
        retention_class: RetentionClass | str = RetentionClass.ROUTINE,
        outcome: ArtifactOutcome | str = ArtifactOutcome.SUCCESSFUL,
        ttl_seconds: int | None = None,
    ) -> tuple[Any, tuple[BlobReference, ...]]:
        """Externalize known large-body fields and deduplicate their bytes."""

        references: dict[str, BlobReference] = {}
        active: set[int] = set()

        def visit(value: Any, field_name: str = "") -> Any:
            if self._is_shallow_reference(value):
                return self._validate_shallow_reference(value)
            if field_name in _EMBEDDED_BODY_FIELDS or field_name in _GRAPH_BODY_FIELDS:
                reference = self.put_blob(
                    value,
                    kind=field_name or "artifact",
                    retention_class=retention_class,
                    outcome=outcome,
                    ttl_seconds=ttl_seconds,
                )
                references[reference.artifact_id] = reference
                return {"artifact_ref": reference.to_dict()}
            if isinstance(value, Mapping):
                identity = id(value)
                if identity in active:
                    raise ArtifactBlobIntegrityError(
                        "recursive artifact payloads are not supported"
                    )
                active.add(identity)
                try:
                    return {
                        str(key): visit(item, str(key))
                        for key, item in value.items()
                    }
                finally:
                    active.remove(identity)
            if isinstance(value, (list, tuple)):
                identity = id(value)
                if identity in active:
                    raise ArtifactBlobIntegrityError(
                        "recursive artifact payloads are not supported"
                    )
                active.add(identity)
                try:
                    return [visit(item, field_name) for item in value]
                finally:
                    active.remove(identity)
            if isinstance(value, bytearray):
                return base64.b64encode(bytes(value)).decode("ascii")
            if isinstance(value, bytes):
                return base64.b64encode(value).decode("ascii")
            if value is None or isinstance(value, (str, bool, int, float)):
                return value
            converter = getattr(value, "to_dict", None)
            if callable(converter):
                return visit(converter(), field_name)
            raise ArtifactBlobIntegrityError(
                f"unsupported projection value: {type(value).__name__}"
            )

        projected = visit(payload)
        return projected, tuple(
            references[key] for key in sorted(references)
        )

    externalize_payload = project_payload

    @staticmethod
    def _projection_metadata(
        reference: ProjectionReference,
    ) -> dict[str, Any]:
        return reference.to_dict()

    @staticmethod
    def _projection_digest(
        payload: Any,
        *,
        projection_kind: str,
        retention_class: RetentionClass,
        outcome: ArtifactOutcome,
    ) -> str:
        identity = {
            "payload": payload,
            "projection_kind": projection_kind,
            "retention_class": retention_class.value,
            "outcome": outcome.value,
        }
        encoded = _bounded_canonical_bytes(
            identity,
            maximum=MAX_PROJECTION_BYTES + MAX_RECEIPT_BYTES,
            label="projection identity",
        )
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def store_projection(
        self,
        payload: Any,
        *,
        projection_kind: str = "routine",
        kind: str | None = None,
        retention_class: RetentionClass | str = RetentionClass.ROUTINE,
        retention: RetentionClass | str | None = None,
        outcome: ArtifactOutcome | str = ArtifactOutcome.SUCCESSFUL,
        ttl_seconds: int | None = None,
    ) -> ProjectionReference:
        """Persist a bounded projection whose large bodies are blob references."""

        self._assert_open()
        selected_kind = str(kind or projection_kind or "routine")
        record_outcome = ArtifactOutcome.coerce(outcome)
        selected_retention = self._coerce_retention(
            retention if retention is not None else retention_class
        )
        if not record_outcome.can_complete:
            selected_retention = RetentionClass.NEGATIVE
        projected, references = self.project_payload(
            payload,
            retention_class=selected_retention,
            outcome=record_outcome,
            ttl_seconds=ttl_seconds,
        )
        maximum = (
            self.quotas.max_receipt_bytes
            if "receipt" in selected_kind.casefold()
            else self.quotas.max_projection_bytes
        )
        encoded = _bounded_canonical_bytes(
            projected,
            maximum=maximum,
            label=(
                "receipt"
                if maximum <= self.quotas.max_receipt_bytes
                else "routine projection"
            ),
        )
        digest = self._projection_digest(
            projected,
            projection_kind=selected_kind,
            retention_class=selected_retention,
            outcome=record_outcome,
        )
        now_ms = self._now_ms()
        reference = ProjectionReference(
            artifact_id=f"projection:{digest}",
            digest=digest,
            size_bytes=len(encoded),
            projection_kind=selected_kind,
            retention_class=selected_retention,
            outcome=record_outcome,
            created_at_ms=now_ms,
            expires_at_ms=self._expiry(
                record_outcome, ttl_seconds, now_ms
            ),
            artifact_references=references,
        )
        wrapper = {
            "schema": BOUNDED_ARTIFACT_STORE_SCHEMA,
            "reference": reference.to_dict(),
            "payload": projected,
        }
        wrapper_bytes = _bounded_canonical_bytes(
            wrapper,
            maximum=self.quotas.max_projection_bytes
            + self.quotas.max_receipt_bytes,
            label="stored projection envelope",
        ) + b"\n"
        with self._locked():
            existing = self._manifest["projections"].get(
                reference.artifact_id
            )
            if existing is not None:
                existing_reference = ProjectionReference.from_dict(existing)
                if (
                    existing_reference.expires_at_ms is None
                    or now_ms < existing_reference.expires_at_ms
                ):
                    return existing_reference
                self._evict_projection(
                    existing_reference.artifact_id, reason="expired"
                )
            staged_blob_metadata: list[dict[str, Any]] = []
            for blob_reference in references:
                metadata = self._manifest["blobs"].get(
                    blob_reference.artifact_id
                )
                if metadata is None:
                    raise ArtifactBlobIntegrityError(
                        "projection references a missing blob"
                    )
                owners = set(metadata.get("references", ()))
                owners.add(reference.artifact_id)
                metadata["references"] = sorted(owners)
                staged_blob_metadata.append(metadata)
            try:
                self._ensure_capacity(
                    additional_bytes=len(encoded), additional_projections=1
                )
                try:
                    _atomic_write_bytes(
                        self._projection_path(reference), wrapper_bytes
                    )
                except OSError as exc:
                    if exc.errno in {errno.ENOSPC, errno.EDQUOT}:
                        self._increment("disk_pressure_rejections")
                        raise ArtifactQuotaExceeded(
                            "projection could not be persisted under disk pressure"
                        ) from exc
                    raise
            except BaseException:
                for metadata in staged_blob_metadata:
                    owners = set(metadata.get("references", ()))
                    owners.discard(reference.artifact_id)
                    metadata["references"] = sorted(owners)
                raise
            self._manifest["projections"][reference.artifact_id] = (
                self._projection_metadata(reference)
            )
            self._write_manifest(self._manifest)
            self._increment("writes")
            self._increment("projection_writes")
        return reference

    persist = store_projection
    write_projection = store_projection
    put = store_projection

    def store_receipt(self, payload: Any, **kwargs: Any) -> ProjectionReference:
        """Persist one receipt while always applying the 256 KiB bound."""

        supplied = kwargs.pop("projection_kind", "receipt")
        if "receipt" not in str(supplied).casefold():
            supplied = f"{supplied}_receipt"
        return self.store_projection(
            payload, projection_kind=str(supplied), **kwargs
        )

    def store_routine_projection(
        self, payload: Any, **kwargs: Any
    ) -> ProjectionReference:
        """Persist one ordinary projection under the 1 MiB bound."""

        return self.store_projection(
            payload,
            projection_kind=str(kwargs.pop("projection_kind", "routine")),
            **kwargs,
        )

    def _coerce_blob_reference(
        self, value: BlobReference | Mapping[str, Any] | str
    ) -> BlobReference:
        if isinstance(value, BlobReference):
            return value
        if isinstance(value, Mapping):
            return BlobReference.from_dict(value)
        metadata = self._manifest["blobs"].get(str(value))
        if metadata is None:
            raise ArtifactBlobIntegrityError("blob reference is unknown")
        return BlobReference.from_dict(metadata)

    def verify_blob(
        self, value: BlobReference | Mapping[str, Any] | str
    ) -> bool:
        try:
            reference = self._coerce_blob_reference(value)
            data = self._blob_path(reference).read_bytes()
        except (OSError, TypeError, ValueError):
            return False
        return (
            len(data) == reference.size_bytes
            and "sha256:" + hashlib.sha256(data).hexdigest()
            == reference.digest
        )

    verify = verify_blob

    def read_blob(
        self,
        value: BlobReference | Mapping[str, Any] | str,
        *,
        decode: bool = False,
    ) -> Any:
        self._assert_open()
        with self._locked():
            reference = self._coerce_blob_reference(value)
            try:
                data = self._blob_path(reference).read_bytes()
            except OSError as exc:
                raise ArtifactBlobIntegrityError(
                    "referenced blob is missing"
                ) from exc
            if (
                len(data) != reference.size_bytes
                or "sha256:" + hashlib.sha256(data).hexdigest()
                != reference.digest
            ):
                self._increment("corruption_recoveries")
                raise ArtifactBlobIntegrityError(
                    "referenced blob failed content integrity verification"
                )
            metadata = self._manifest["blobs"].get(reference.artifact_id)
            if metadata is not None:
                metadata["last_accessed_at_ms"] = self._now_ms()
            self._increment("reads")
        if not decode:
            return data
        if reference.media_type.startswith("text/"):
            return data.decode("utf-8")
        if reference.media_type == "application/json":
            return json.loads(data)
        return data

    get_blob = read_blob
    load_blob = read_blob

    def _coerce_projection_reference(
        self, value: ProjectionReference | Mapping[str, Any] | str
    ) -> ProjectionReference:
        if isinstance(value, ProjectionReference):
            return value
        if isinstance(value, Mapping):
            return ProjectionReference.from_dict(value)
        metadata = self._manifest["projections"].get(str(value))
        if metadata is None:
            raise ArtifactBlobIntegrityError("projection reference is unknown")
        return ProjectionReference.from_dict(metadata)

    def read_projection(
        self,
        value: ProjectionReference | Mapping[str, Any] | str,
        *,
        verify_blobs: bool = True,
    ) -> Any:
        self._assert_open()
        with self._locked():
            reference = self._coerce_projection_reference(value)
            if (
                reference.expires_at_ms is not None
                and self._now_ms() >= reference.expires_at_ms
            ):
                raise ArtifactBlobIntegrityError("projection has expired")
            try:
                wrapper = json.loads(
                    self._projection_path(reference).read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArtifactBlobIntegrityError(
                    "stored projection is missing or corrupt"
                ) from exc
            encoded = _bounded_canonical_bytes(
                wrapper.get("payload"),
                maximum=self.quotas.max_projection_bytes,
                label="stored projection",
            )
            if (
                len(encoded) != reference.size_bytes
                or self._projection_digest(
                    wrapper.get("payload"),
                    projection_kind=reference.projection_kind,
                    retention_class=reference.retention_class,
                    outcome=reference.outcome,
                )
                != reference.digest
            ):
                raise ArtifactBlobIntegrityError(
                    "stored projection failed content integrity verification"
                )
            if verify_blobs:
                for blob_reference in reference.artifact_references:
                    if not self.verify_blob(blob_reference):
                        raise ArtifactBlobIntegrityError(
                            "stored projection contains a corrupt blob reference"
                        )
            self._increment("reads")
            return wrapper["payload"]

    get = read_projection
    load = read_projection

    def _emit_eviction(
        self,
        *,
        artifact_id: str,
        size_bytes: int,
        reason: str,
        retention_class: str,
    ) -> None:
        event = {
            "type": "artifact_evicted",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "artifact_id": artifact_id,
            "size_bytes": size_bytes,
            "reason": reason,
            "retention_class": retention_class,
        }
        encoded = _bounded_canonical_bytes(
            event, maximum=MAX_RECEIPT_BYTES, label="eviction event"
        ) + b"\n"
        self.eviction_log_path.parent.mkdir(
            parents=True, exist_ok=True, mode=0o700
        )
        with self.eviction_log_path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if self._eviction_observer is not None:
            self._eviction_observer(dict(event))

    def _evict_projection(
        self, artifact_id: str, *, reason: str
    ) -> tuple[int, str]:
        metadata = self._manifest["projections"].pop(artifact_id)
        reference = ProjectionReference.from_dict(metadata)
        try:
            self._projection_path(reference).unlink()
        except FileNotFoundError:
            pass
        for blob_reference in reference.artifact_references:
            blob_metadata = self._manifest["blobs"].get(
                blob_reference.artifact_id
            )
            if blob_metadata is not None:
                owners = set(blob_metadata.get("references", ()))
                owners.discard(artifact_id)
                blob_metadata["references"] = sorted(owners)
        self._emit_eviction(
            artifact_id=artifact_id,
            size_bytes=reference.size_bytes,
            reason=reason,
            retention_class=reference.retention_class.value,
        )
        return reference.size_bytes, reference.retention_class.value

    def _evict_blob(
        self, artifact_id: str, *, reason: str
    ) -> tuple[int, str]:
        metadata = self._manifest["blobs"].pop(artifact_id)
        reference = BlobReference.from_dict(metadata)
        try:
            self._blob_path(reference).unlink()
        except FileNotFoundError:
            pass
        retention = str(
            metadata.get("retention_class") or RetentionClass.ROUTINE.value
        )
        self._emit_eviction(
            artifact_id=artifact_id,
            size_bytes=reference.size_bytes,
            reason=reason,
            retention_class=retention,
        )
        return reference.size_bytes, retention

    def _quota_satisfied(
        self,
        reserve_bytes: int,
        reserve_blobs: int,
        reserve_projections: int,
    ) -> bool:
        return self._has_capacity(
            additional_bytes=reserve_bytes,
            additional_blobs=reserve_blobs,
            additional_projections=reserve_projections,
        )

    def _compact_locked(
        self,
        *,
        max_items: int,
        force_quota: bool,
        reserve_bytes: int = 0,
        reserve_blobs: int = 0,
        reserve_projections: int = 0,
    ) -> CompactionResult:
        if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items < 1:
            raise ValueError("max_items must be a positive integer")
        now_ms = self._now_ms()
        candidates: list[tuple[int, int, str, str, Mapping[str, Any]]] = []
        for artifact_id, metadata in self._manifest["projections"].items():
            retention = RetentionClass(
                str(
                    metadata.get("retention_class")
                    or RetentionClass.ROUTINE.value
                )
            )
            expires_at = metadata.get("expires_at_ms")
            expired = isinstance(expires_at, int) and now_ms >= expires_at
            if expired or (
                force_quota and retention is not RetentionClass.PINNED
            ):
                candidates.append(
                    (
                        0 if expired else 1,
                        self._retention_rank[retention],
                        artifact_id,
                        "projection",
                        metadata,
                    )
                )
        for artifact_id, metadata in self._manifest["blobs"].items():
            if metadata.get("references"):
                continue
            retention = RetentionClass(
                str(
                    metadata.get("retention_class")
                    or RetentionClass.ROUTINE.value
                )
            )
            expires_at = metadata.get("expires_at_ms")
            expired = isinstance(expires_at, int) and now_ms >= expires_at
            if expired or (
                force_quota and retention is not RetentionClass.PINNED
            ):
                candidates.append(
                    (
                        0 if expired else 1,
                        self._retention_rank[retention],
                        artifact_id,
                        "blob",
                        metadata,
                    )
                )
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                int(
                    item[4].get("last_accessed_at_ms")
                    or item[4].get("created_at_ms")
                    or 0
                ),
                item[2],
            )
        )
        cursor = int(self._manifest.get("compaction_cursor", 0))
        if candidates:
            cursor %= len(candidates)
            ordered = candidates[cursor:] + candidates[:cursor]
        else:
            ordered = []
            cursor = 0
        scanned = evicted = evicted_bytes = expired_count = quota_count = 0
        evicted_ids: list[str] = []
        for expiry_rank, _retention_rank, artifact_id, kind, _metadata in ordered:
            if scanned >= max_items:
                break
            scanned += 1
            if (
                expiry_rank != 0
                and self._quota_satisfied(
                    reserve_bytes, reserve_blobs, reserve_projections
                )
            ):
                continue
            if kind == "projection":
                size, _retention = self._evict_projection(
                    artifact_id,
                    reason="expired" if expiry_rank == 0 else "quota",
                )
            else:
                size, _retention = self._evict_blob(
                    artifact_id,
                    reason="expired" if expiry_rank == 0 else "quota",
                )
            evicted += 1
            evicted_bytes += size
            evicted_ids.append(artifact_id)
            if expiry_rank == 0:
                expired_count += 1
            else:
                quota_count += 1
        self._manifest["compaction_cursor"] = (
            (cursor + scanned) % max(1, len(candidates))
        )
        if evicted or scanned:
            self._write_manifest(self._manifest)
        self._increment("compactions")
        self._increment("scanned", scanned)
        self._increment("evictions", evicted)
        self._increment("evicted_bytes", evicted_bytes)
        self._increment("expired_evictions", expired_count)
        self._increment("quota_evictions", quota_count)
        return CompactionResult(
            scanned=scanned,
            evicted=evicted,
            evicted_bytes=evicted_bytes,
            expired=expired_count,
            quota_evicted=quota_count,
            cursor=int(self._manifest["compaction_cursor"]),
            quota_satisfied=self._quota_satisfied(
                reserve_bytes, reserve_blobs, reserve_projections
            ),
            evicted_artifact_ids=tuple(evicted_ids),
        )

    def compact(
        self,
        *,
        max_items: int | None = None,
        limit: int | None = None,
        force_quota: bool = False,
    ) -> CompactionResult:
        """Inspect at most one batch and atomically checkpoint the GC cursor."""

        self._assert_open()
        if max_items is not None and limit is not None:
            raise ValueError("pass max_items or limit, not both")
        with self._locked():
            return self._compact_locked(
                max_items=max_items
                or limit
                or self.quotas.compaction_batch_size,
                force_quota=force_quota,
            )

    incremental_compact = compact
    gc = compact

    def manifest(self) -> dict[str, Any]:
        """Return a detached manifest snapshot for restart diagnostics."""

        with self._locked():
            return json.loads(json.dumps(self._manifest))

    def eviction_events(self) -> list[dict[str, Any]]:
        try:
            lines = self.eviction_log_path.read_text(
                encoding="utf-8"
            ).splitlines()
        except OSError:
            return []
        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def metrics(self) -> ArtifactStoreMetrics:
        with self._metrics_lock:
            return ArtifactStoreMetrics(**self._metric_values)

    stats = metrics

    def usage(self) -> dict[str, Any]:
        """Return the current logical quota projection without scanning bodies."""

        with self._locked():
            blob_bytes = sum(
                int(item.get("size_bytes", 0))
                for item in self._manifest["blobs"].values()
            )
            projection_bytes = sum(
                int(item.get("size_bytes", 0))
                for item in self._manifest["projections"].values()
            )
            try:
                disk_free_bytes = shutil.disk_usage(self.path).free
            except OSError:
                disk_free_bytes = None
            return {
                "total_bytes": blob_bytes + projection_bytes,
                "blob_bytes": blob_bytes,
                "projection_bytes": projection_bytes,
                "blob_count": len(self._manifest["blobs"]),
                "projection_count": len(self._manifest["projections"]),
                "disk_free_bytes": disk_free_bytes,
                "quotas": self.quotas.to_dict(),
            }

    def close(
        self,
        *,
        timeout_seconds: float = 1.0,
        timeout: float | None = None,
    ) -> bool:
        """Synchronously checkpoint state; no unbounded background drain exists."""

        if timeout is not None:
            if timeout_seconds != 1.0:
                raise ValueError("pass timeout_seconds or timeout, not both")
            timeout_seconds = timeout
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        if self._closed:
            return True
        deadline = time.monotonic() + float(timeout_seconds)
        with self._locked():
            if time.monotonic() >= deadline:
                return False
            self._write_manifest(self._manifest)
            self._closed = True
        return time.monotonic() <= deadline

    shutdown = close


BoundedPersistenceStore = BoundedArtifactStore
ContentAddressedArtifactStore = BoundedArtifactStore
ArtifactBlobStore = BoundedArtifactStore
ArtifactStore = BoundedArtifactStore


def query_artifact_paths(path: Path | str) -> QueryArtifactPaths:
    """Resolve either a JSON or DuckDB path to both artifact representations."""

    resolved = Path(path).resolve()
    suffix = resolved.suffix.lower()
    if suffix == ".duckdb":
        return QueryArtifactPaths(
            json_path=resolved.with_suffix(".json"), duckdb_path=resolved
        )
    if suffix == ".json":
        return QueryArtifactPaths(
            json_path=resolved, duckdb_path=resolved.with_suffix(".duckdb")
        )
    raise ValueError(f"queryable artifacts require a .json or .duckdb path: {resolved}")


def _duckdb_module() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - declared runtime dependency
        raise RuntimeError(
            "DuckDB is required for queryable supervisor artifacts"
        ) from exc
    return duckdb


def _configure_duckdb_connection(connection: Any) -> Any:
    """Bound storage work so planning leaves CPU and memory for worker lanes."""

    connection.execute(f"SET threads={DUCKDB_ARTIFACT_THREADS}")
    connection.execute(f"SET memory_limit='{DUCKDB_ARTIFACT_MEMORY_LIMIT}'")
    return connection


@contextmanager
def _artifact_write_lock(database_path: Path) -> Iterator[None]:
    """Serialize paired JSON/DuckDB generations across supervisor processes."""

    lock_path = database_path.with_name(f".{database_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    acquired = False
    deadline = time.monotonic() + ARTIFACT_LOCK_TIMEOUT_SECONDS
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out acquiring query artifact lock: {lock_path}"
                    )
                time.sleep(0.01)
        yield
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_value(value: str) -> Any:
    return json.loads(value)


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return bool(value)


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_kind(payload: Mapping[str, Any]) -> str:
    schema = str(payload.get("schema") or "")
    if schema == PROOF_ATTESTATION_STORE_SCHEMA and isinstance(
        payload.get("attestations"), list
    ):
        return PROOF_ATTESTATION_KIND
    if schema.startswith(
        "ipfs_accelerate_py.agent_supervisor.proof-metrics@"
    ) or all(
        key in payload
        for key in ("obligations", "attempts", "receipts", "cache_outcomes", "metrics")
    ):
        # This check must precede the generic scheduler ``counts`` check so a
        # JSON-only proof artifact can rebuild its DuckDB sidecar correctly.
        return PROOF_METRICS_KIND
    if schema == "ipfs_accelerate_py.agent_supervisor.code-evidence-graph@1" or (
        isinstance(payload.get("nodes"), list)
        and isinstance(payload.get("edges"), list)
        and str(payload.get("graph_id") or "").startswith("graph-")
    ):
        return CODE_EVIDENCE_GRAPH_KIND
    if isinstance(payload.get("bundles"), Mapping):
        return BUNDLE_INDEX_KIND
    if any(key in payload for key in ("lanes", "tasks", "scheduler_state", "counts")):
        return SCHEDULER_MANIFEST_KIND
    raise ValueError("could not infer supervisor artifact kind")


def _query_descriptor(kind: str, paths: QueryArtifactPaths) -> dict[str, Any]:
    return {
        "schema": QUERY_SCHEMA,
        "artifact_kind": kind,
        "duckdb_path": paths.duckdb_path.name,
        "catalog_table": "artifact_catalog",
    }


def _common_schema(connection: Any) -> None:
    connection.execute("""
        CREATE TABLE artifact_catalog (
            artifact_kind VARCHAR NOT NULL,
            schema_version VARCHAR NOT NULL,
            source_path VARCHAR NOT NULL,
            generated_at VARCHAR,
            source_sha256 VARCHAR NOT NULL,
            database_payload_sha256 VARCHAR NOT NULL,
            source_size BIGINT NOT NULL,
            source_mtime_ns BIGINT NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE artifact_fields (
            field_name VARCHAR PRIMARY KEY,
            value_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE artifact_tables (
            table_name VARCHAR PRIMARY KEY,
            description VARCHAR NOT NULL
        )
        """)


def _bundle_schema(connection: Any) -> None:
    connection.execute("""
        CREATE TABLE bundles (
            bundle_key VARCHAR PRIMARY KEY,
            shard_path VARCHAR,
            parallel_lane VARCHAR,
            bundle_strategy VARCHAR,
            conflict_policy VARCHAR,
            task_count BIGINT NOT NULL,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE bundle_tasks (
            bundle_key VARCHAR NOT NULL,
            task_ordinal BIGINT NOT NULL,
            task_id VARCHAR,
            canonical_task_cid VARCHAR,
            goal_id VARCHAR,
            parent_goal_id VARCHAR,
            subgoal_id VARCHAR,
            status VARCHAR,
            priority VARCHAR,
            title VARCHAR,
            payload_json VARCHAR NOT NULL,
            PRIMARY KEY (bundle_key, task_ordinal)
        )
        """)
    connection.execute("""
        CREATE TABLE bundle_task_dependencies (
            bundle_key VARCHAR NOT NULL,
            task_id VARCHAR,
            dependency_kind VARCHAR NOT NULL,
            dependency_id VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE dependency_edges (
            edge_ordinal BIGINT NOT NULL,
            source_task_cid VARCHAR,
            target_task_cid VARCHAR,
            edge_kind VARCHAR,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE conflict_edges (
            edge_ordinal BIGINT NOT NULL,
            left_task_cid VARCHAR,
            right_task_cid VARCHAR,
            reason VARCHAR,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE planning_decisions (
            decision_ordinal BIGINT NOT NULL,
            left_task_cid VARCHAR,
            right_task_cid VARCHAR,
            decision VARCHAR,
            reason VARCHAR,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("CREATE INDEX bundle_tasks_task_id_idx ON bundle_tasks(task_id)")
    connection.execute(
        "CREATE INDEX bundle_tasks_cid_idx ON bundle_tasks(canonical_task_cid)"
    )
    connection.execute("""
        CREATE VIEW open_bundle_tasks AS
        SELECT * FROM bundle_tasks
        WHERE lower(coalesce(status, 'todo')) NOT IN
              ('complete', 'completed', 'done', 'succeeded', 'blocked')
        """)


def _manifest_schema(connection: Any) -> None:
    connection.execute("""
        CREATE TABLE manifest_tasks (
            task_ordinal BIGINT NOT NULL,
            task_cid VARCHAR,
            task_id VARCHAR,
            bundle_key VARCHAR,
            state VARCHAR,
            lease_state VARCHAR,
            attempt BIGINT,
            claimant_did VARCHAR,
            updated_at_ms BIGINT,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE manifest_lanes (
            lane_ordinal BIGINT NOT NULL,
            bundle_key VARCHAR,
            parallel_lane VARCHAR,
            task_cid VARCHAR,
            state VARCHAR,
            pid BIGINT,
            claimable BOOLEAN,
            conflict_color BIGINT,
            schedule_rank BIGINT,
            log_path VARCHAR,
            task_ids_json VARCHAR NOT NULL,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE manifest_decisions (
            decision_ordinal BIGINT NOT NULL,
            task_cid VARCHAR,
            bundle_key VARCHAR,
            decision VARCHAR,
            reason VARCHAR,
            snapshot_id VARCHAR,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE manifest_conflict_edges (
            edge_ordinal BIGINT NOT NULL,
            left_task_id VARCHAR,
            right_task_id VARCHAR,
            blocks_concurrency BOOLEAN,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE manifest_conflict_decisions (
            decision_ordinal BIGINT NOT NULL,
            left_task_cid VARCHAR,
            right_task_cid VARCHAR,
            action VARCHAR,
            weight DOUBLE,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE scheduler_task_states (
            state_ordinal BIGINT NOT NULL,
            task_cid VARCHAR,
            task_id VARCHAR,
            goal_cid VARCHAR,
            subgoal_cid VARCHAR,
            lane_id VARCHAR,
            provider_id VARCHAR,
            phase VARCHAR,
            status VARCHAR,
            last_event_at VARCHAR,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE scheduler_metrics (
            metric_ordinal BIGINT NOT NULL,
            task_cid VARCHAR,
            goal_cid VARCHAR,
            subgoal_cid VARCHAR,
            lane_id VARCHAR,
            provider_id VARCHAR,
            repository_tree_id VARCHAR,
            template_id VARCHAR,
            resource_class VARCHAR,
            queue_latency_ms BIGINT,
            solver_latency_ms BIGINT,
            kernel_latency_ms BIGINT,
            model_latency_ms BIGINT,
            validation_latency_ms BIGINT,
            merge_latency_ms BIGINT,
            cancellation_latency_ms BIGINT,
            cache_latency_ms BIGINT,
            queue_wait_seconds DOUBLE,
            implementation_duration_seconds DOUBLE,
            validation_duration_seconds DOUBLE,
            merge_wait_seconds DOUBLE,
            retries BIGINT,
            conflicts BIGINT,
            completions BIGINT,
            total_tokens BIGINT,
            total_cost_usd DOUBLE,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE scheduler_phases (
            phase VARCHAR PRIMARY KEY,
            task_count BIGINT NOT NULL
        )
        """)
    connection.execute(
        "CREATE INDEX manifest_tasks_cid_idx ON manifest_tasks(task_cid)"
    )
    connection.execute("CREATE INDEX manifest_tasks_state_idx ON manifest_tasks(state)")
    connection.execute(
        "CREATE INDEX manifest_lanes_key_idx ON manifest_lanes(bundle_key)"
    )
    connection.execute(
        "CREATE INDEX scheduler_task_states_cid_idx ON scheduler_task_states(task_cid)"
    )
    connection.execute(
        "CREATE VIEW ready_tasks AS SELECT * FROM manifest_tasks WHERE state = 'ready'"
    )
    connection.execute("""
        CREATE VIEW blocked_tasks AS
        SELECT * FROM manifest_tasks AS task
        WHERE task.state = 'blocked'
           OR (task.state = 'accepted' AND NOT EXISTS (
               SELECT 1 FROM manifest_lanes AS lane WHERE lane.task_cid = task.task_cid
           ))
        """)
    connection.execute(
        "CREATE VIEW completed_tasks AS SELECT * FROM manifest_tasks WHERE state = 'completed'"
    )
    connection.execute(
        "CREATE VIEW active_lanes AS SELECT * FROM manifest_lanes WHERE state = 'running'"
    )


def _code_evidence_graph_schema(connection: Any) -> None:
    connection.execute("""
        CREATE TABLE evidence_nodes (
            node_id VARCHAR PRIMARY KEY,
            node_kind VARCHAR NOT NULL,
            record_key VARCHAR NOT NULL,
            provenance VARCHAR NOT NULL,
            authoritative BOOLEAN NOT NULL,
            task_id VARCHAR,
            tree_id VARCHAR,
            symbol VARCHAR,
            obligation_id VARCHAR,
            assurance VARCHAR,
            freshness VARCHAR,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE evidence_edges (
            edge_id VARCHAR PRIMARY KEY,
            source_node_id VARCHAR NOT NULL,
            target_node_id VARCHAR NOT NULL,
            edge_kind VARCHAR NOT NULL,
            provenance VARCHAR NOT NULL,
            provenance_record_id VARCHAR NOT NULL,
            authoritative BOOLEAN NOT NULL,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute("""
        CREATE TABLE graph_records (
            record_type VARCHAR NOT NULL,
            record_id VARCHAR NOT NULL,
            record_ordinal BIGINT NOT NULL,
            payload_json VARCHAR NOT NULL,
            PRIMARY KEY (record_type, record_id)
        )
        """)
    for statement in (
        "CREATE INDEX evidence_nodes_kind_idx ON evidence_nodes(node_kind)",
        "CREATE INDEX evidence_nodes_task_idx ON evidence_nodes(task_id)",
        "CREATE INDEX evidence_nodes_tree_idx ON evidence_nodes(tree_id)",
        "CREATE INDEX evidence_nodes_symbol_idx ON evidence_nodes(symbol)",
        "CREATE INDEX evidence_nodes_obligation_idx ON evidence_nodes(obligation_id)",
        "CREATE INDEX evidence_nodes_assurance_idx ON evidence_nodes(assurance)",
        "CREATE INDEX evidence_nodes_freshness_idx ON evidence_nodes(freshness)",
        "CREATE INDEX evidence_edges_source_idx ON evidence_edges(source_node_id)",
        "CREATE INDEX evidence_edges_target_idx ON evidence_edges(target_node_id)",
        "CREATE INDEX evidence_edges_kind_idx ON evidence_edges(edge_kind)",
    ):
        connection.execute(statement)

    # Stable, narrow query surfaces.  Both the concise ``*_index`` spellings
    # and explicit ``code_evidence_*`` aliases are kept for callers composing
    # SQL across multiple supervisor artifact kinds.
    connection.execute(
        "CREATE VIEW task_index AS SELECT * FROM evidence_nodes "
        "WHERE node_kind = 'task'"
    )
    connection.execute(
        "CREATE VIEW tree_index AS SELECT * FROM evidence_nodes "
        "WHERE node_kind = 'tree'"
    )
    connection.execute(
        "CREATE VIEW symbol_index AS SELECT * FROM evidence_nodes "
        "WHERE node_kind = 'symbol'"
    )
    connection.execute(
        "CREATE VIEW obligation_index AS SELECT * FROM evidence_nodes "
        "WHERE node_kind = 'obligation'"
    )
    connection.execute(
        "CREATE VIEW assurance_index AS SELECT node_id, node_kind, task_id, "
        "obligation_id, assurance, authoritative, payload_json FROM evidence_nodes "
        "WHERE assurance <> ''"
    )
    connection.execute(
        "CREATE VIEW freshness_index AS SELECT node_id, node_kind, task_id, "
        "obligation_id, freshness, authoritative, payload_json FROM evidence_nodes "
        "WHERE freshness <> ''"
    )
    connection.execute(
        "CREATE VIEW dependency_index AS SELECT * FROM evidence_edges "
        "WHERE edge_kind = 'depends_on'"
    )
    connection.execute(
        "CREATE VIEW authoritative_evidence_edges AS SELECT * FROM evidence_edges "
        "WHERE authoritative"
    )
    for alias, source in (
        ("graph_nodes", "evidence_nodes"),
        ("graph_edges", "evidence_edges"),
        ("tasks", "task_index"),
        ("trees", "tree_index"),
        ("symbols", "symbol_index"),
        ("obligations", "obligation_index"),
        ("assurances", "assurance_index"),
        ("freshness", "freshness_index"),
        ("dependencies", "dependency_index"),
        ("graph_tasks", "task_index"),
        ("graph_trees", "tree_index"),
        ("graph_symbols", "symbol_index"),
        ("graph_obligations", "obligation_index"),
        ("graph_assurance", "assurance_index"),
        ("graph_freshness", "freshness_index"),
        ("graph_dependencies", "dependency_index"),
        ("code_evidence_tasks", "task_index"),
        ("code_evidence_trees", "tree_index"),
        ("code_evidence_symbols", "symbol_index"),
        ("code_evidence_obligations", "obligation_index"),
        ("code_evidence_assurance", "assurance_index"),
        ("code_evidence_freshness", "freshness_index"),
        ("code_evidence_dependencies", "dependency_index"),
    ):
        connection.execute(f"CREATE VIEW {alias} AS SELECT * FROM {source}")


def _proof_attestation_schema(connection: Any) -> None:
    connection.execute("""
        CREATE TABLE proof_attestations (
            record_id VARCHAR PRIMARY KEY,
            proof_receipt_id VARCHAR NOT NULL,
            kernel_receipt_id VARCHAR,
            envelope_id VARCHAR NOT NULL,
            verification_id VARCHAR NOT NULL,
            statement_id VARCHAR NOT NULL,
            public_input_digest VARCHAR NOT NULL,
            formal_policy_id VARCHAR NOT NULL,
            backend_policy_id VARCHAR NOT NULL,
            backend_id VARCHAR NOT NULL,
            backend_version VARCHAR NOT NULL,
            circuit_id VARCHAR NOT NULL,
            circuit_version VARCHAR NOT NULL,
            public_input_schema_id VARCHAR NOT NULL,
            public_input_schema_version VARCHAR NOT NULL,
            verification_key_id VARCHAR NOT NULL,
            verification_key_version VARCHAR NOT NULL,
            verification_key_expires_at VARCHAR,
            backend_health_id VARCHAR NOT NULL,
            proof_artifact_id VARCHAR NOT NULL,
            proof_digest VARCHAR NOT NULL,
            verifier_id VARCHAR NOT NULL,
            verdict VARCHAR NOT NULL,
            independent BOOLEAN NOT NULL,
            authoritative BOOLEAN NOT NULL,
            created_at VARCHAR NOT NULL,
            expires_at VARCHAR NOT NULL,
            ipfs_cid VARCHAR,
            payload_json VARCHAR NOT NULL
        )
        """)
    connection.execute(
        "CREATE INDEX proof_attestations_receipt_idx "
        "ON proof_attestations(proof_receipt_id)"
    )
    connection.execute(
        "CREATE INDEX proof_attestations_expiry_idx "
        "ON proof_attestations(expires_at)"
    )


def _proof_metrics_schema(connection: Any) -> None:
    """Create the compact, public proof observability query schema."""

    dimensions = """
        goal_cid VARCHAR NOT NULL,
        subgoal_cid VARCHAR NOT NULL,
        task_cid VARCHAR NOT NULL,
        repository_tree_id VARCHAR NOT NULL,
        provider_id VARCHAR NOT NULL,
        template_id VARCHAR NOT NULL,
        resource_class VARCHAR NOT NULL
    """
    connection.execute(f"""
        CREATE TABLE proof_obligations (
            {dimensions},
            obligation_id VARCHAR,
            plan_id VARCHAR,
            invariant_class VARCHAR,
            required_assurance VARCHAR,
            status VARCHAR,
            ast_scope_ids_json VARCHAR NOT NULL,
            premise_count BIGINT NOT NULL,
            fallback_check_count BIGINT NOT NULL
        )
        """)
    connection.execute(f"""
        CREATE TABLE proof_attempts (
            {dimensions},
            attempt_id VARCHAR,
            plan_id VARCHAR,
            step_id VARCHAR,
            obligation_id VARCHAR,
            stage VARCHAR,
            status VARCHAR,
            started_at VARCHAR,
            finished_at VARCHAR,
            duration_ms BIGINT NOT NULL,
            input_count BIGINT NOT NULL,
            output_count BIGINT NOT NULL,
            evidence_count BIGINT NOT NULL,
            error_code VARCHAR,
            claimed_assurance VARCHAR,
            authoritative_assurance VARCHAR,
            cpu_milliseconds BIGINT NOT NULL,
            memory_peak_bytes BIGINT NOT NULL,
            input_token_count BIGINT NOT NULL,
            output_token_count BIGINT NOT NULL,
            token_count BIGINT NOT NULL
        )
        """)
    connection.execute(f"""
        CREATE TABLE proof_receipts (
            {dimensions},
            receipt_id VARCHAR,
            plan_id VARCHAR,
            attempt_id VARCHAR,
            obligation_id VARCHAR,
            repository_id VARCHAR,
            verdict VARCHAR,
            assurance VARCHAR,
            authoritative BOOLEAN NOT NULL,
            freshness VARCHAR,
            policy_id VARCHAR,
            translator_id VARCHAR,
            solver_id VARCHAR,
            kernel_id VARCHAR,
            toolchain_id VARCHAR,
            theorem_registry_id VARCHAR,
            started_at VARCHAR,
            finished_at VARCHAR,
            duration_ms BIGINT NOT NULL,
            scope_count BIGINT NOT NULL,
            premise_count BIGINT NOT NULL,
            evidence_count BIGINT NOT NULL,
            assurance_reason_codes_json VARCHAR NOT NULL
        )
        """)
    connection.execute(f"""
        CREATE TABLE proof_dependencies (
            {dimensions},
            plan_id VARCHAR,
            source_step_id VARCHAR,
            target_step_id VARCHAR,
            obligation_id VARCHAR,
            dependency_kind VARCHAR,
            satisfied BOOLEAN
        )
        """)
    connection.execute(f"""
        CREATE TABLE proof_cache_outcomes (
            {dimensions},
            cache_key VARCHAR,
            obligation_id VARCHAR,
            receipt_id VARCHAR,
            outcome VARCHAR,
            lookup_latency_ms BIGINT NOT NULL,
            required_assurance VARCHAR,
            actual_assurance VARCHAR,
            fresh BOOLEAN,
            reason_codes_json VARCHAR NOT NULL,
            observed_at VARCHAR
        )
        """)
    connection.execute(f"""
        CREATE TABLE proof_resource_samples (
            {dimensions},
            observed_at_ms BIGINT NOT NULL,
            cpu_percent BIGINT NOT NULL,
            memory_percent BIGINT NOT NULL,
            disk_percent BIGINT NOT NULL,
            memory_used_bytes BIGINT NOT NULL,
            memory_available_bytes BIGINT NOT NULL,
            disk_used_bytes BIGINT NOT NULL,
            disk_available_bytes BIGINT NOT NULL,
            active_workers BIGINT NOT NULL,
            available_worker_capacity BIGINT NOT NULL,
            provider_latency_ms BIGINT NOT NULL,
            provider_quota_remaining BIGINT NOT NULL,
            provider_token_budget_remaining BIGINT NOT NULL
        )
        """)
    connection.execute(f"""
        CREATE TABLE proof_assurance_counts (
            {dimensions},
            assurance VARCHAR NOT NULL,
            receipt_count BIGINT NOT NULL,
            authoritative_count BIGINT NOT NULL
        )
        """)
    connection.execute(f"""
        CREATE TABLE proof_metrics (
            {dimensions},
            obligation_count BIGINT NOT NULL,
            attempt_count BIGINT NOT NULL,
            successful_attempt_count BIGINT NOT NULL,
            failed_attempt_count BIGINT NOT NULL,
            receipt_count BIGINT NOT NULL,
            authoritative_receipt_count BIGINT NOT NULL,
            dependency_count BIGINT NOT NULL,
            cache_hit_count BIGINT NOT NULL,
            cache_miss_count BIGINT NOT NULL,
            cache_rejection_count BIGINT NOT NULL,
            resource_sample_count BIGINT NOT NULL,
            cancellation_count BIGINT NOT NULL,
            availability_check_count BIGINT NOT NULL,
            availability_success_count BIGINT NOT NULL,
            availability_failure_count BIGINT NOT NULL,
            schema_validation_count BIGINT NOT NULL,
            schema_acceptance_count BIGINT NOT NULL,
            schema_rejection_count BIGINT NOT NULL,
            proof_closure_count BIGINT NOT NULL,
            fallback_count BIGINT NOT NULL,
            repair_attempt_count BIGINT NOT NULL,
            repair_convergence_count BIGINT NOT NULL,
            repair_exhaustion_count BIGINT NOT NULL,
            input_token_count BIGINT NOT NULL,
            output_token_count BIGINT NOT NULL,
            token_count BIGINT NOT NULL,
            unsupported_semantics_count BIGINT NOT NULL,
            false_completion_prevention_count BIGINT NOT NULL,
            queue_latency_ms BIGINT NOT NULL,
            solver_latency_ms BIGINT NOT NULL,
            kernel_latency_ms BIGINT NOT NULL,
            model_latency_ms BIGINT NOT NULL,
            validation_latency_ms BIGINT NOT NULL,
            merge_latency_ms BIGINT NOT NULL,
            cancellation_latency_ms BIGINT NOT NULL,
            cache_latency_ms BIGINT NOT NULL,
            queue_latency_seconds DOUBLE NOT NULL,
            solver_latency_seconds DOUBLE NOT NULL,
            kernel_latency_seconds DOUBLE NOT NULL,
            model_latency_seconds DOUBLE NOT NULL,
            validation_latency_seconds DOUBLE NOT NULL,
            merge_latency_seconds DOUBLE NOT NULL,
            cancellation_latency_seconds DOUBLE NOT NULL,
            cache_latency_seconds DOUBLE NOT NULL,
            availability_rate DOUBLE NOT NULL,
            schema_acceptance_rate DOUBLE NOT NULL,
            proof_closure_rate DOUBLE NOT NULL,
            fallback_rate DOUBLE NOT NULL,
            repair_convergence_rate DOUBLE NOT NULL,
            cache_hit_rate DOUBLE NOT NULL
        )
        """)
    for table in (
        "proof_obligations",
        "proof_attempts",
        "proof_receipts",
        "proof_dependencies",
        "proof_cache_outcomes",
        "proof_resource_samples",
        "proof_assurance_counts",
        "proof_metrics",
    ):
        connection.execute(
            f"CREATE INDEX {table}_identity_idx ON {table}"
            "(goal_cid, subgoal_cid, task_cid, repository_tree_id, "
            "provider_id, template_id, resource_class)"
        )
    for table, identifier in (
        ("proof_obligations", "obligation_id"),
        ("proof_attempts", "attempt_id"),
        ("proof_receipts", "receipt_id"),
    ):
        connection.execute(
            f"CREATE INDEX {table}_{identifier}_idx ON {table}({identifier})"
        )
    for alias, source in (
        ("obligations", "proof_obligations"),
        ("attempts", "proof_attempts"),
        ("receipts", "proof_receipts"),
        ("dependencies", "proof_dependencies"),
        ("cache_outcomes", "proof_cache_outcomes"),
        ("resource_samples", "proof_resource_samples"),
        ("assurance_counts", "proof_assurance_counts"),
        ("latency_metrics", "proof_metrics"),
        ("proof_latency_metrics", "proof_metrics"),
        ("proof_metric_aggregates", "proof_metrics"),
    ):
        connection.execute(f"CREATE VIEW {alias} AS SELECT * FROM {source}")


def _top_level_fields(
    payload: Mapping[str, Any], kind: str
) -> Iterable[tuple[str, str]]:
    if kind == PROOF_ATTESTATION_KIND:
        for key, value in payload.items():
            if key != "attestations":
                yield str(key), _json_text(value)
        return
    if kind == PROOF_METRICS_KIND:
        # Proof query databases expose a deliberately closed catalog.  Unknown
        # extension fields must not become an accidental side channel for a
        # witness, transcript, prompt, or provider diagnostic.
        allowed = {
            "schema",
            "schema_version",
            "generated_at",
            "snapshot_id",
            "authoritative",
            "bounded",
            "contains_hidden_witnesses",
            "contains_proof_transcripts",
            "plan_id",
            "plan_ids",
            "totals",
            "source_counts",
            "query_store",
        }
        for key, value in payload.items():
            if key in allowed:
                yield str(key), _json_text(value)
        return
    excluded = (
        {"bundles"}
        if kind == BUNDLE_INDEX_KIND
        else {"nodes", "edges"}
        if kind == CODE_EVIDENCE_GRAPH_KIND
        else {
            "obligations",
            "attempts",
            "receipts",
            "dependencies",
            "cache_outcomes",
            "resource_samples",
            "assurance_counts",
            "metrics",
            "latency_metrics",
        }
        if kind == PROOF_METRICS_KIND
        else {
            "blocked",
            "completed",
            "lanes",
            "ready",
            "scheduler_decisions",
            "tasks",
        }
    )
    for key, value in payload.items():
        if key not in excluded:
            yield str(key), _json_text(value)


def _graph_mapping(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _mapping_items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _populate_bundle_tables(connection: Any, payload: Mapping[str, Any]) -> None:
    bundles = (
        payload.get("bundles") if isinstance(payload.get("bundles"), Mapping) else {}
    )
    bundle_rows: list[tuple[Any, ...]] = []
    task_rows: list[tuple[Any, ...]] = []
    dependency_rows: list[tuple[Any, ...]] = []
    for bundle_key, raw_bundle in sorted(bundles.items()):
        if not isinstance(raw_bundle, Mapping):
            continue
        tasks = _mapping_items(raw_bundle.get("tasks"))
        bundle_payload = {
            key: value for key, value in raw_bundle.items() if key != "tasks"
        }
        bundle_rows.append(
            (
                str(bundle_key),
                str(raw_bundle.get("shard_path") or ""),
                str(raw_bundle.get("parallel_lane") or ""),
                str(raw_bundle.get("bundle_strategy") or ""),
                str(raw_bundle.get("conflict_policy") or ""),
                len(tasks),
                _json_text(bundle_payload),
            )
        )
        for ordinal, task in enumerate(tasks):
            task_id = str(task.get("task_id") or "")
            task_rows.append(
                (
                    str(bundle_key),
                    ordinal,
                    task_id,
                    str(task.get("canonical_task_cid") or task.get("task_cid") or ""),
                    str(task.get("goal_id") or ""),
                    str(task.get("parent_goal_id") or ""),
                    str(task.get("subgoal_id") or ""),
                    str(task.get("status") or ""),
                    str(task.get("priority") or ""),
                    str(task.get("title") or ""),
                    _json_text(task),
                )
            )
            for dependency_kind in (
                "depends_on",
                "dependency_task_cids",
                "blocking_task_cids",
            ):
                dependency_rows.extend(
                    (str(bundle_key), task_id, dependency_kind, dependency_id)
                    for dependency_id in _string_values(task.get(dependency_kind))
                )
    if bundle_rows:
        connection.executemany(
            "INSERT INTO bundles VALUES (?, ?, ?, ?, ?, ?, ?)", bundle_rows
        )
    if task_rows:
        connection.executemany(
            "INSERT INTO bundle_tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            task_rows,
        )
    if dependency_rows:
        connection.executemany(
            "INSERT INTO bundle_task_dependencies VALUES (?, ?, ?, ?)", dependency_rows
        )

    dependency_graph = _graph_mapping(
        payload, "task_dependency_graph", "dependency_dag"
    )
    dependency_edges = _mapping_items(dependency_graph.get("edges"))
    if dependency_edges:
        connection.executemany(
            "INSERT INTO dependency_edges VALUES (?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(edge.get("source_task_cid") or edge.get("source") or ""),
                    str(edge.get("target_task_cid") or edge.get("target") or ""),
                    str(edge.get("kind") or edge.get("edge_kind") or ""),
                    _json_text(edge),
                )
                for ordinal, edge in enumerate(dependency_edges)
            ],
        )

    conflict_graph = _graph_mapping(payload, "task_conflict_graph", "conflict_graph")
    conflict_edges = _mapping_items(conflict_graph.get("edges"))
    if conflict_edges:
        connection.executemany(
            "INSERT INTO conflict_edges VALUES (?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(edge.get("left_task_cid") or edge.get("left") or ""),
                    str(edge.get("right_task_cid") or edge.get("right") or ""),
                    str(edge.get("reason") or edge.get("conflict_reason") or ""),
                    _json_text(edge),
                )
                for ordinal, edge in enumerate(conflict_edges)
            ],
        )
    decisions = _mapping_items(
        payload.get("conflict_planning_decisions") or conflict_graph.get("decisions")
    )
    if decisions:
        connection.executemany(
            "INSERT INTO planning_decisions VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(item.get("left_task_cid") or item.get("left") or ""),
                    str(item.get("right_task_cid") or item.get("right") or ""),
                    str(item.get("decision") or ""),
                    str(item.get("reason") or ""),
                    _json_text(item),
                )
                for ordinal, item in enumerate(decisions)
            ],
        )


def _populate_manifest_tables(connection: Any, payload: Mapping[str, Any]) -> None:
    tasks = _mapping_items(payload.get("tasks"))
    if tasks:
        connection.executemany(
            "INSERT INTO manifest_tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(item.get("task_cid") or ""),
                    str(item.get("task_id") or ""),
                    str(item.get("bundle_key") or ""),
                    str(item.get("state") or ""),
                    str(item.get("lease_state") or ""),
                    _as_int(item.get("attempt")),
                    str(item.get("claimant_did") or ""),
                    _as_int(item.get("updated_at_ms")),
                    _json_text(item),
                )
                for ordinal, item in enumerate(tasks)
            ],
        )
    lanes = _mapping_items(payload.get("lanes"))
    if lanes:
        connection.executemany(
            "INSERT INTO manifest_lanes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(item.get("bundle_key") or ""),
                    str(item.get("parallel_lane") or ""),
                    str(item.get("task_cid") or ""),
                    str(item.get("state") or ""),
                    _as_int(item.get("pid")),
                    _as_bool(item.get("claimable")),
                    _as_int(item.get("conflict_color")),
                    _as_int(item.get("schedule_rank")),
                    str(item.get("log_path") or ""),
                    _json_text(_string_values(item.get("task_ids"))),
                    _json_text(item),
                )
                for ordinal, item in enumerate(lanes)
            ],
        )
    decisions = _mapping_items(payload.get("scheduler_decisions"))
    if decisions:
        connection.executemany(
            "INSERT INTO manifest_decisions VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(item.get("task_cid") or ""),
                    str(item.get("bundle_key") or ""),
                    str(item.get("decision") or ""),
                    str(item.get("reason") or ""),
                    str(item.get("snapshot_id") or ""),
                    _json_text(item),
                )
                for ordinal, item in enumerate(decisions)
            ],
        )
    conflict_graph = payload.get("conflict_graph")
    if not isinstance(conflict_graph, Mapping):
        conflict_graph = {}
    conflict_edges = _mapping_items(conflict_graph.get("edges"))
    if conflict_edges:
        connection.executemany(
            "INSERT INTO manifest_conflict_edges VALUES (?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(item.get("left_task_id") or item.get("left_task_cid") or ""),
                    str(item.get("right_task_id") or item.get("right_task_cid") or ""),
                    _as_bool(item.get("blocks_concurrency")),
                    _json_text(item),
                )
                for ordinal, item in enumerate(conflict_edges)
            ],
        )
    conflict_decisions = _mapping_items(conflict_graph.get("decisions"))
    if conflict_decisions:
        connection.executemany(
            "INSERT INTO manifest_conflict_decisions VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(item.get("left_task_cid") or ""),
                    str(item.get("right_task_cid") or ""),
                    str(item.get("action") or item.get("decision") or ""),
                    _as_float(item.get("weight")),
                    _json_text(item),
                )
                for ordinal, item in enumerate(conflict_decisions)
            ],
        )

    scheduler_snapshot = payload.get("scheduler_snapshot")
    if not isinstance(scheduler_snapshot, Mapping):
        scheduler_snapshot = {}
    task_states = _mapping_items(scheduler_snapshot.get("task_states"))
    if task_states:
        connection.executemany(
            "INSERT INTO scheduler_task_states VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    ordinal,
                    str(item.get("task_cid") or item.get("canonical_task_cid") or ""),
                    str(item.get("task_id") or ""),
                    str(item.get("goal_cid") or item.get("canonical_goal_id") or ""),
                    str(
                        item.get("subgoal_cid")
                        or item.get("canonical_subgoal_id")
                        or ""
                    ),
                    str(item.get("lane_id") or item.get("canonical_lane_id") or ""),
                    str(
                        item.get("provider_id")
                        or item.get("canonical_provider_id")
                        or ""
                    ),
                    str(item.get("phase") or ""),
                    str(item.get("status") or ""),
                    str(item.get("last_event_at") or ""),
                    _json_text(item),
                )
                for ordinal, item in enumerate(task_states)
            ],
        )
    metrics = _mapping_items(scheduler_snapshot.get("metrics"))
    if metrics:
        connection.executemany(
            "INSERT INTO scheduler_metrics VALUES ("
            + ", ".join("?" for _ in range(27))
            + ")",
            [
                (
                    ordinal,
                    str(item.get("task_cid") or item.get("canonical_task_cid") or ""),
                    str(item.get("goal_cid") or item.get("canonical_goal_id") or ""),
                    str(
                        item.get("subgoal_cid")
                        or item.get("canonical_subgoal_id")
                        or ""
                    ),
                    str(item.get("lane_id") or item.get("canonical_lane_id") or ""),
                    str(
                        item.get("provider_id")
                        or item.get("canonical_provider_id")
                        or ""
                    ),
                    str(
                        item.get("repository_tree_id")
                        or item.get("tree_id")
                        or item.get("canonical_tree_id")
                        or "unknown"
                    ),
                    str(
                        item.get("template_id")
                        or item.get("canonical_template_id")
                        or "unknown"
                    ),
                    str(
                        item.get("resource_class")
                        or item.get("canonical_resource_class")
                        or "unknown"
                    ),
                    _as_int(item.get("queue_latency_ms")) or 0,
                    _as_int(item.get("solver_latency_ms")) or 0,
                    _as_int(item.get("kernel_latency_ms")) or 0,
                    _as_int(item.get("model_latency_ms")) or 0,
                    _as_int(item.get("validation_latency_ms")) or 0,
                    _as_int(item.get("merge_latency_ms")) or 0,
                    _as_int(item.get("cancellation_latency_ms")) or 0,
                    _as_int(item.get("cache_latency_ms")) or 0,
                    _as_float(item.get("queue_wait_seconds")),
                    _as_float(item.get("implementation_duration_seconds")),
                    _as_float(item.get("validation_duration_seconds")),
                    _as_float(item.get("merge_wait_seconds")),
                    _as_int(item.get("retries")),
                    _as_int(item.get("conflicts")),
                    _as_int(item.get("completions")),
                    _as_int(item.get("total_tokens", item.get("tokens"))),
                    _as_float(item.get("total_cost_usd", item.get("cost_usd"))),
                    _json_text(item),
                )
                for ordinal, item in enumerate(metrics)
            ],
        )
    phases = scheduler_snapshot.get("phases")
    if isinstance(phases, Mapping):
        phase_rows = [
            (
                str(phase),
                _as_int(value.get("count")) or 0 if isinstance(value, Mapping) else 0,
            )
            for phase, value in sorted(phases.items())
        ]
        if phase_rows:
            connection.executemany(
                "INSERT INTO scheduler_phases VALUES (?, ?)", phase_rows
            )


def _populate_code_evidence_graph_tables(
    connection: Any, payload: Mapping[str, Any]
) -> None:
    # Decode through the graph contract before persistence.  This rejects
    # forged identities and any enrichment-originated authoritative edge.
    from .code_evidence_graph import CodeEvidenceGraph

    graph = CodeEvidenceGraph.from_dict(payload)
    node_rows: list[tuple[Any, ...]] = []
    graph_rows: list[tuple[Any, ...]] = []
    for ordinal, node in enumerate(graph.nodes):
        item = node.to_dict()
        text = _json_text(item)
        node_rows.append(
            (
                node.node_id,
                node.kind.value,
                node.record_key,
                node.provenance.value,
                node.authoritative,
                node.task_id,
                node.tree_id,
                node.symbol,
                node.obligation_id,
                node.assurance,
                node.freshness,
                text,
            )
        )
        graph_rows.append(("node", node.node_id, ordinal, text))
    edge_rows: list[tuple[Any, ...]] = []
    for ordinal, edge in enumerate(graph.edges):
        item = edge.to_dict()
        text = _json_text(item)
        edge_rows.append(
            (
                edge.edge_id,
                edge.source,
                edge.target,
                edge.kind.value,
                edge.provenance.value,
                edge.provenance_record_id,
                edge.authoritative,
                text,
            )
        )
        graph_rows.append(("edge", edge.edge_id, ordinal, text))
    if node_rows:
        connection.executemany(
            "INSERT INTO evidence_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            node_rows,
        )
    if edge_rows:
        connection.executemany(
            "INSERT INTO evidence_edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            edge_rows,
        )
    if graph_rows:
        connection.executemany(
            "INSERT INTO graph_records VALUES (?, ?, ?, ?)", graph_rows
        )


def _table_descriptions(kind: str) -> dict[str, str]:
    common = {
        "artifact_catalog": "Artifact identity, source freshness, and schema version.",
        "artifact_fields": "Individually queryable top-level JSON fields.",
        "artifact_tables": "Descriptions of the normalized query tables.",
    }
    if kind == BUNDLE_INDEX_KIND:
        return {
            **common,
            "bundles": "One row per objective bundle without embedded member tasks.",
            "bundle_tasks": "One typed row per bundle member with full JSON in payload_json.",
            "bundle_task_dependencies": "Normalized declared and computed task dependencies.",
            "dependency_edges": "Normalized dependency graph edges.",
            "conflict_edges": "Normalized conflict graph edges.",
            "planning_decisions": "Normalized conflict-planning decisions.",
            "open_bundle_tasks": "View of bundle tasks that are not terminal or blocked.",
        }
    if kind == CODE_EVIDENCE_GRAPH_KIND:
        return {
            **common,
            "evidence_nodes": "Canonical graph nodes with indexed task, tree, symbol, obligation, assurance, and freshness fields.",
            "evidence_edges": "Canonical provenance edges and their derived authority.",
            "graph_records": "Lossless canonical node and edge records used for projection round trips.",
            "task_index": "Task nodes.",
            "tree_index": "Repository tree nodes.",
            "symbol_index": "Qualified AST symbol nodes.",
            "obligation_index": "Code proof obligation nodes.",
            "assurance_index": "Evidence records with an assurance projection.",
            "freshness_index": "Evidence records with a freshness projection.",
            "dependency_index": "Task and proof dependency edges.",
            "authoritative_evidence_edges": "Gate-relevant edges derived from trusted record boundaries.",
            "graph_nodes": "Compatibility alias for evidence_nodes.",
            "graph_edges": "Compatibility alias for evidence_edges.",
            "tasks": "Compatibility alias for task_index.",
            "trees": "Compatibility alias for tree_index.",
            "symbols": "Compatibility alias for symbol_index.",
            "obligations": "Compatibility alias for obligation_index.",
            "assurances": "Compatibility alias for assurance_index.",
            "freshness": "Compatibility alias for freshness_index.",
            "dependencies": "Compatibility alias for dependency_index.",
            "graph_tasks": "Compatibility alias for task_index.",
            "graph_trees": "Compatibility alias for tree_index.",
            "graph_symbols": "Compatibility alias for symbol_index.",
            "graph_obligations": "Compatibility alias for obligation_index.",
            "graph_assurance": "Compatibility alias for assurance_index.",
            "graph_freshness": "Compatibility alias for freshness_index.",
            "graph_dependencies": "Compatibility alias for dependency_index.",
            "code_evidence_tasks": "Compatibility alias for task_index.",
            "code_evidence_trees": "Compatibility alias for tree_index.",
            "code_evidence_symbols": "Compatibility alias for symbol_index.",
            "code_evidence_obligations": "Compatibility alias for obligation_index.",
            "code_evidence_assurance": "Compatibility alias for assurance_index.",
            "code_evidence_freshness": "Compatibility alias for freshness_index.",
            "code_evidence_dependencies": "Compatibility alias for dependency_index.",
        }
    if kind == PROOF_ATTESTATION_KIND:
        return {
            **common,
            "proof_attestations": (
                "Public, receipt-bound ZKP verification sidecars with backend, "
                "circuit, key, policy, expiration, and optional IPFS identities."
            ),
        }
    if kind == PROOF_METRICS_KIND:
        descriptions = {
            **common,
            "proof_obligations": "Bounded obligation identities and assurance requirements.",
            "proof_attempts": "Provider attempt status, timing, counts, and numeric resource use.",
            "proof_receipts": "Public receipt verdict, freshness, and derived assurance projection.",
            "proof_dependencies": "Normalized proof-plan dependency edges.",
            "proof_cache_outcomes": "Trust-aware cache hit, miss, rejection, and lookup latency.",
            "proof_resource_samples": "Bounded host and provider resource measurements.",
            "proof_assurance_counts": "Receipt counts grouped by canonical dimensions and assurance.",
            "proof_metrics": "Wide proof latency and throughput aggregates with all dimensions.",
            "obligations": "Compatibility alias for proof_obligations.",
            "attempts": "Compatibility alias for proof_attempts.",
            "receipts": "Compatibility alias for proof_receipts.",
            "dependencies": "Compatibility alias for proof_dependencies.",
            "cache_outcomes": "Compatibility alias for proof_cache_outcomes.",
            "resource_samples": "Compatibility alias for proof_resource_samples.",
            "assurance_counts": "Compatibility alias for proof_assurance_counts.",
            "latency_metrics": "Compatibility alias for proof_metrics.",
            "proof_latency_metrics": "Compatibility alias for proof_metrics.",
            "proof_metric_aggregates": "Compatibility alias for proof_metrics.",
        }
        return descriptions
    return {
        **common,
        "manifest_tasks": "Current scheduler task projection, one task per row.",
        "manifest_lanes": "Current worker lane projection, one lane per row.",
        "manifest_decisions": "Scheduler admission and deferral decisions.",
        "manifest_conflict_edges": "Normalized scheduler bundle-conflict edges.",
        "manifest_conflict_decisions": "Normalized scheduler conflict-coloring decisions.",
        "scheduler_task_states": "Per-task lifecycle states from the authoritative scheduler snapshot.",
        "scheduler_metrics": "Per-task timing, retry, conflict, token, and cost metrics.",
        "scheduler_phases": "Task counts grouped by scheduler lifecycle phase.",
        "ready_tasks": "View of tasks currently ready for a lease.",
        "blocked_tasks": "View of tasks currently blocked.",
        "completed_tasks": "View of completed tasks.",
        "active_lanes": "View of currently running lanes.",
    }


def _proof_dimensions(row: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(row.get(name) or "unknown")
        for name in (
            "goal_cid",
            "subgoal_cid",
            "task_cid",
            "repository_tree_id",
            "provider_id",
            "template_id",
            "resource_class",
        )
    )


def _populate_proof_attestation_tables(
    connection: Any, payload: Mapping[str, Any]
) -> None:
    from .proof_attestation import PersistedAttestationRecord

    for raw in payload.get("attestations") or ():
        if not isinstance(raw, Mapping):
            raise ValueError("proof attestation rows must be objects")
        record = PersistedAttestationRecord.from_dict(raw)
        rendered = record.to_public_artifact()
        statement = record.envelope.statement
        verification = record.verification
        connection.execute(
            "INSERT INTO proof_attestations VALUES ("
            + ", ".join("?" for _ in range(29))
            + ")",
            (
                record.record_id,
                record.proof_receipt_id,
                record.kernel_receipt_id,
                record.envelope_id,
                record.verification_id,
                record.statement_id,
                record.public_input_digest,
                statement.policy_id,
                statement.backend_policy_id,
                statement.backend_id,
                statement.backend_version,
                statement.circuit_id,
                statement.circuit_version,
                statement.public_input_schema_id,
                statement.public_input_schema_version,
                statement.verification_key_id,
                statement.verification_key_version,
                record.backend_policy.verification_key_expires_at,
                record.envelope.backend_health_id,
                record.envelope.proof_artifact_id,
                record.envelope.proof_digest,
                verification.verifier_id,
                verification.verdict.value,
                verification.independent,
                verification.authoritative,
                record.created_at,
                record.expires_at,
                str(raw.get("ipfs_cid") or ""),
                _json_text(rendered),
            ),
        )


def _populate_proof_metrics_tables(
    connection: Any, payload: Mapping[str, Any]
) -> None:
    """Populate proof tables from allowlisted public projection fields only."""

    for row in _mapping_items(payload.get("obligations")):
        connection.execute(
            "INSERT INTO proof_obligations VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                *_proof_dimensions(row),
                str(row.get("obligation_id") or ""),
                str(row.get("plan_id") or ""),
                str(row.get("invariant_class") or ""),
                str(row.get("required_assurance") or "unverified"),
                str(row.get("status") or "planned"),
                _json_text(_string_values(row.get("ast_scope_ids"))),
                _as_int(row.get("premise_count")) or 0,
                _as_int(row.get("fallback_check_count")) or 0,
            ),
        )
    for row in _mapping_items(payload.get("attempts")):
        connection.execute(
            "INSERT INTO proof_attempts VALUES ("
            + ", ".join("?" for _ in range(27))
            + ")",
            (
                *_proof_dimensions(row),
                str(row.get("attempt_id") or ""),
                str(row.get("plan_id") or ""),
                str(row.get("step_id") or ""),
                str(row.get("obligation_id") or ""),
                str(row.get("stage") or "unknown"),
                str(row.get("status") or "unknown"),
                str(row.get("started_at") or ""),
                str(row.get("finished_at") or ""),
                _as_int(row.get("duration_ms")) or 0,
                _as_int(row.get("input_count")) or 0,
                _as_int(row.get("output_count")) or 0,
                _as_int(row.get("evidence_count")) or 0,
                str(row.get("error_code") or ""),
                str(row.get("claimed_assurance") or "unverified"),
                str(row.get("authoritative_assurance") or "unverified"),
                _as_int(row.get("cpu_milliseconds")) or 0,
                _as_int(row.get("memory_peak_bytes")) or 0,
                _as_int(row.get("input_token_count")) or 0,
                _as_int(row.get("output_token_count")) or 0,
                _as_int(row.get("token_count")) or 0,
            ),
        )
    for row in _mapping_items(payload.get("receipts")):
        connection.execute(
            "INSERT INTO proof_receipts VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                *_proof_dimensions(row),
                str(row.get("receipt_id") or ""),
                str(row.get("plan_id") or ""),
                str(row.get("attempt_id") or ""),
                str(row.get("obligation_id") or ""),
                str(row.get("repository_id") or ""),
                str(row.get("verdict") or "inconclusive"),
                str(row.get("assurance") or "unverified"),
                bool(row.get("authoritative")),
                str(row.get("freshness") or "unknown"),
                str(row.get("policy_id") or ""),
                str(row.get("translator_id") or ""),
                str(row.get("solver_id") or ""),
                str(row.get("kernel_id") or ""),
                str(row.get("toolchain_id") or ""),
                str(row.get("theorem_registry_id") or ""),
                str(row.get("started_at") or ""),
                str(row.get("finished_at") or ""),
                _as_int(row.get("duration_ms")) or 0,
                _as_int(row.get("scope_count")) or 0,
                _as_int(row.get("premise_count")) or 0,
                _as_int(row.get("evidence_count")) or 0,
                _json_text(_string_values(row.get("assurance_reason_codes"))),
            ),
        )
    for row in _mapping_items(payload.get("dependencies")):
        connection.execute(
            "INSERT INTO proof_dependencies VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                *_proof_dimensions(row),
                str(row.get("plan_id") or ""),
                str(row.get("source_step_id") or ""),
                str(row.get("target_step_id") or ""),
                str(row.get("obligation_id") or ""),
                str(row.get("dependency_kind") or "requires"),
                _as_bool(row.get("satisfied")),
            ),
        )
    for row in _mapping_items(payload.get("cache_outcomes")):
        connection.execute(
            "INSERT INTO proof_cache_outcomes VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                *_proof_dimensions(row),
                str(row.get("cache_key") or ""),
                str(row.get("obligation_id") or ""),
                str(row.get("receipt_id") or ""),
                str(row.get("outcome") or "miss"),
                _as_int(row.get("lookup_latency_ms")) or 0,
                str(row.get("required_assurance") or "unverified"),
                str(row.get("actual_assurance") or "unverified"),
                _as_bool(row.get("fresh")),
                _json_text(_string_values(row.get("reason_codes"))),
                str(row.get("observed_at") or ""),
            ),
        )
    for row in _mapping_items(payload.get("resource_samples")):
        connection.execute(
            "INSERT INTO proof_resource_samples VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                *_proof_dimensions(row),
                _as_int(row.get("observed_at_ms")) or 0,
                _as_int(row.get("cpu_percent")) or 0,
                _as_int(row.get("memory_percent")) or 0,
                _as_int(row.get("disk_percent")) or 0,
                _as_int(row.get("memory_used_bytes")) or 0,
                _as_int(row.get("memory_available_bytes")) or 0,
                _as_int(row.get("disk_used_bytes")) or 0,
                _as_int(row.get("disk_available_bytes")) or 0,
                _as_int(row.get("active_workers")) or 0,
                _as_int(row.get("available_worker_capacity")) or 0,
                _as_int(row.get("provider_latency_ms")) or 0,
                _as_int(row.get("provider_quota_remaining")) or 0,
                _as_int(row.get("provider_token_budget_remaining")) or 0,
            ),
        )
    for row in _mapping_items(payload.get("assurance_counts")):
        connection.execute(
            "INSERT INTO proof_assurance_counts VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                *_proof_dimensions(row),
                str(row.get("assurance") or "unverified"),
                _as_int(row.get("receipt_count")) or 0,
                _as_int(row.get("authoritative_count")) or 0,
            ),
        )
    count_fields = (
        "obligation_count",
        "attempt_count",
        "successful_attempt_count",
        "failed_attempt_count",
        "receipt_count",
        "authoritative_receipt_count",
        "dependency_count",
        "cache_hit_count",
        "cache_miss_count",
        "cache_rejection_count",
        "resource_sample_count",
        "cancellation_count",
        "availability_check_count",
        "availability_success_count",
        "availability_failure_count",
        "schema_validation_count",
        "schema_acceptance_count",
        "schema_rejection_count",
        "proof_closure_count",
        "fallback_count",
        "repair_attempt_count",
        "repair_convergence_count",
        "repair_exhaustion_count",
        "input_token_count",
        "output_token_count",
        "token_count",
        "unsupported_semantics_count",
        "false_completion_prevention_count",
    )
    latency_fields = (
        "queue_latency",
        "solver_latency",
        "kernel_latency",
        "model_latency",
        "validation_latency",
        "merge_latency",
        "cancellation_latency",
        "cache_latency",
    )
    rate_fields = (
        "availability_rate",
        "schema_acceptance_rate",
        "proof_closure_rate",
        "fallback_rate",
        "repair_convergence_rate",
        "cache_hit_rate",
    )
    for row in _mapping_items(payload.get("metrics") or payload.get("latency_metrics")):
        connection.execute(
            "INSERT INTO proof_metrics VALUES ("
            + ", ".join("?" for _ in range(57))
            + ")",
            (
                *_proof_dimensions(row),
                *(_as_int(row.get(name)) or 0 for name in count_fields),
                *(_as_int(row.get(f"{name}_ms")) or 0 for name in latency_fields),
                *(_as_float(row.get(f"{name}_seconds")) or 0.0 for name in latency_fields),
                *(_as_float(row.get(name)) or 0.0 for name in rate_fields),
            ),
        )


def _write_duckdb(
    path: Path,
    payload: Mapping[str, Any],
    *,
    kind: str,
    source_path: Path,
    source_sha256: str,
    source_size: int,
    source_mtime_ns: int,
) -> None:
    if kind == PROOF_METRICS_KIND:
        from .proof_metrics import ProofMetricsSnapshot

        # Rebuilding a sidecar from JSON is another trust boundary.  Validate
        # here as well as in the public writer so a hand-edited or stale JSON
        # file cannot promote private proof material into DuckDB.
        payload = ProofMetricsSnapshot(payload).to_dict()
    duckdb = _duckdb_module()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.unlink(missing_ok=True)
    Path(f"{temporary}.wal").unlink(missing_ok=True)
    payload_text = _json_text(payload)
    try:
        connection = _configure_duckdb_connection(
            duckdb.connect(str(temporary))
        )
        try:
            connection.execute("BEGIN TRANSACTION")
            try:
                _common_schema(connection)
                if kind == BUNDLE_INDEX_KIND:
                    _bundle_schema(connection)
                    _populate_bundle_tables(connection, payload)
                elif kind == SCHEDULER_MANIFEST_KIND:
                    _manifest_schema(connection)
                    _populate_manifest_tables(connection, payload)
                elif kind == CODE_EVIDENCE_GRAPH_KIND:
                    _code_evidence_graph_schema(connection)
                    _populate_code_evidence_graph_tables(connection, payload)
                elif kind == PROOF_ATTESTATION_KIND:
                    _proof_attestation_schema(connection)
                    _populate_proof_attestation_tables(connection, payload)
                elif kind == PROOF_METRICS_KIND:
                    _proof_metrics_schema(connection)
                    _populate_proof_metrics_tables(connection, payload)
                else:
                    raise ValueError(f"unsupported query artifact kind: {kind}")
                connection.execute(
                    "INSERT INTO artifact_catalog VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        kind,
                        QUERY_SCHEMA,
                        str(source_path),
                        str(payload.get("generated_at") or ""),
                        source_sha256,
                        hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
                        source_size,
                        source_mtime_ns,
                    ),
                )
                fields = list(_top_level_fields(payload, kind))
                if fields:
                    connection.executemany(
                        "INSERT INTO artifact_fields VALUES (?, ?)", fields
                    )
                connection.executemany(
                    "INSERT INTO artifact_tables VALUES (?, ?)",
                    sorted(_table_descriptions(kind).items()),
                )
                connection.execute("COMMIT")
            except BaseException:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            connection.execute("CHECKPOINT")
        finally:
            connection.close()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        Path(f"{temporary}.wal").unlink(missing_ok=True)


def write_queryable_artifact(
    path: Path | str,
    payload: Mapping[str, Any],
    *,
    kind: str | None = None,
    database_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write equivalent JSON and normalized DuckDB artifacts."""

    paths = query_artifact_paths(path)
    resolved_kind = kind or _artifact_kind(payload)
    rendered = dict(payload)
    rendered["query_store"] = _query_descriptor(resolved_kind, paths)
    database_rendered = dict(database_payload or rendered)
    database_rendered["query_store"] = dict(rendered["query_store"])
    source_text = json.dumps(rendered, indent=2, sort_keys=True) + "\n"
    with _artifact_write_lock(paths.duckdb_path):
        _atomic_write_text(paths.json_path, source_text)
        source_stat = paths.json_path.stat()
        _write_duckdb(
            paths.duckdb_path,
            database_rendered,
            kind=resolved_kind,
            source_path=paths.json_path,
            source_sha256=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            source_size=source_stat.st_size,
            source_mtime_ns=source_stat.st_mtime_ns,
        )
    return rendered


def _compact_bundle_conflict_surface(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    surface = dict(value)
    ast_records = surface.pop("ast_records", None)
    metadata = surface.pop("metadata", None)
    if isinstance(ast_records, list):
        surface["ast_record_count"] = len(ast_records)
    else:
        surface.setdefault("ast_record_count", 0)
    if isinstance(metadata, Mapping):
        surface["metadata_field_count"] = len(metadata)
    else:
        surface.setdefault("metadata_field_count", 0)
    return surface


def _compact_bundle_task(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    task = dict(value)
    task_id = str(task.get("task_id") or "")
    task_cid = str(task.get("canonical_task_cid") or task.get("task_cid") or "")
    for field_name, count_name in (
        ("conflict_decisions", "conflict_decision_count"),
        ("conflict_edges", "conflict_edge_count"),
    ):
        records = task.pop(field_name, None)
        if isinstance(records, list):
            task[count_name] = len(records)
    coverage = task.pop("coverage_inputs", None)
    if isinstance(coverage, Mapping):
        task["coverage_input_field_count"] = len(coverage)
        task.setdefault(
            "coverage_input_ref",
            {
                "field": "todo_coverage_inputs",
                "task_id": task_id,
                "todo_vector_key": str(task.get("todo_vector_key") or ""),
            },
        )
    surface = task.get("conflict_surface")
    if isinstance(surface, Mapping):
        task["conflict_surface"] = _compact_bundle_conflict_surface(surface)
    for field_name in (
        "conflict_graph",
        "conflict_planning_decisions",
        "dependency_dag",
        "task_conflict_graph",
        "task_dependency_graph",
        "task_planning_graph",
        "todo_coverage_inputs",
        "todo_vector_summary",
    ):
        task.pop(field_name, None)
    if task_cid and (
        task.get("conflict_decision_count") or task.get("conflict_edge_count")
    ):
        task.setdefault(
            "conflict_evidence_ref",
            {
                "field": "task_conflict_graph",
                "task_cid": task_cid,
                "tables": ["conflict_edges", "planning_decisions"],
            },
        )
    return task


def _compact_dependency_graph(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    graph = dict(value)
    nodes = graph.get("nodes")
    if isinstance(nodes, Mapping):
        compact_nodes: dict[str, Any] = {}
        for task_cid, raw_node in nodes.items():
            if not isinstance(raw_node, Mapping):
                continue
            node = dict(raw_node)
            if isinstance(node.get("metadata"), Mapping):
                node["metadata"] = _compact_bundle_task(node["metadata"])
            compact_nodes[str(task_cid)] = node
        graph["nodes"] = compact_nodes
    return graph


def _stored_collection_count(
    value: Mapping[str, Any],
    field_name: str,
    count_name: str,
) -> int:
    collection = value.get(field_name)
    count = len(collection) if isinstance(collection, (list, Mapping)) else 0
    stored_count = value.get(count_name)
    if isinstance(stored_count, int) and stored_count >= 0:
        count = max(count, stored_count)
    return count


def compact_conflict_graph_projection(
    value: Any,
    *,
    max_inline_items: int = MAX_INLINE_GRAPH_ITEMS,
) -> dict[str, Any]:
    """Return an inline graph or a bounded query-store projection."""

    if not isinstance(value, Mapping):
        return {}
    graph = dict(value)
    edge_count = _stored_collection_count(graph, "edges", "edge_count")
    decision_count = max(
        _stored_collection_count(graph, "decisions", "planning_decision_count"),
        _stored_collection_count(
            graph,
            "planning_decisions",
            "planning_decision_count",
        ),
    )
    surface_count = _stored_collection_count(graph, "surfaces", "surface_count")
    assignment_count = _stored_collection_count(
        graph,
        "assignments",
        "assignment_count",
    )
    lane_count = _stored_collection_count(graph, "lanes", "lane_count")
    if max(
        edge_count,
        decision_count,
        surface_count,
        assignment_count,
        lane_count,
    ) > max_inline_items:
        return {
            "schema": str(graph.get("schema") or ""),
            "history": dict(graph.get("history") or {})
            if isinstance(graph.get("history"), Mapping)
            else {},
            "edge_count": edge_count,
            "planning_decision_count": decision_count,
            "surface_count": surface_count,
            "assignment_count": assignment_count,
            "lane_count": lane_count,
            "compacted": True,
            "planning_evidence_ref": {
                "field": "task_conflict_graph",
                "tables": ["conflict_edges", "planning_decisions"],
            },
        }
    surfaces = graph.get("surfaces")
    if isinstance(surfaces, Mapping):
        graph["surfaces"] = {
            str(task_cid): _compact_bundle_conflict_surface(surface)
            for task_cid, surface in surfaces.items()
        }
    return graph


def compact_coverage_inputs_projection(
    value: Any,
    *,
    max_inline_tasks: int = MAX_INLINE_COVERAGE_TASKS,
) -> dict[str, Any]:
    """Return coverage inputs inline until they require bounded retrieval."""

    if not isinstance(value, Mapping):
        return {}
    coverage = dict(value)
    task_count = _stored_collection_count(coverage, "by_task", "task_count")
    goal_count = _stored_collection_count(coverage, "by_goal", "goal_count")
    criterion_count = _stored_collection_count(
        coverage,
        "criteria",
        "criterion_count",
    )
    edge_count = _stored_collection_count(coverage, "edges", "edge_count")
    if task_count <= max_inline_tasks and max(criterion_count, edge_count) <= (
        max_inline_tasks * 4
    ):
        return coverage
    return {
        "schema": str(coverage.get("schema") or ""),
        "fingerprint": str(coverage.get("fingerprint") or ""),
        "goal_ids": list(coverage.get("goal_ids") or [])
        if isinstance(coverage.get("goal_ids"), list)
        else [],
        "unmapped_bucket": str(coverage.get("unmapped_bucket") or ""),
        "unmapped_task_ids": list(coverage.get("unmapped_task_ids") or [])
        if isinstance(coverage.get("unmapped_task_ids"), list)
        else [],
        "task_count": task_count,
        "goal_count": goal_count,
        "criterion_count": criterion_count,
        "edge_count": edge_count,
        "compacted": True,
        "coverage_evidence_ref": {
            "field": "todo_coverage_inputs",
            "table": "artifact_fields",
        },
    }


def _compact_task_planning_graph(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    decisions = value.get("planning_decisions")
    decision_count = len(decisions) if isinstance(decisions, list) else 0
    stored_decision_count = value.get("planning_decision_count")
    if isinstance(stored_decision_count, int) and stored_decision_count >= 0:
        decision_count = max(decision_count, stored_decision_count)
    decisions_truncated = bool(value.get("planning_decisions_truncated"))
    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.task_planning_projection@1",
        "claimable_task_cids": _string_values(value.get("claimable_task_cids")),
        "lanes": dict(value.get("lanes") or {})
        if isinstance(value.get("lanes"), Mapping)
        else {},
        "lane_assignments": list(value.get("lane_assignments") or [])
        if isinstance(value.get("lane_assignments"), list)
        else [],
        "planning_decisions": (
            [dict(item) for item in decisions if isinstance(item, Mapping)]
            if isinstance(decisions, list) and len(decisions) <= 128
            else []
        ),
        "planning_decision_count": decision_count,
        "planning_decisions_truncated": decisions_truncated or decision_count > 128,
        "planning_evidence_ref": {
            "dependency_field": "task_dependency_graph",
            "conflict_field": "task_conflict_graph",
            "tables": [
                "dependency_edges",
                "conflict_edges",
                "planning_decisions",
            ],
        },
    }


def _compact_bundle_index_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    rendered = dict(payload)
    bundles = rendered.get("bundles")
    if isinstance(bundles, Mapping):
        compact_bundles: dict[str, Any] = {}
        for bundle_key, raw_bundle in bundles.items():
            if not isinstance(raw_bundle, Mapping):
                continue
            bundle = dict(raw_bundle)
            tasks = bundle.get("tasks")
            if isinstance(tasks, list):
                bundle["tasks"] = [
                    _compact_bundle_task(task)
                    for task in tasks
                    if isinstance(task, Mapping)
                ]
            summary = bundle.get("todo_vector_summary")
            if isinstance(summary, Mapping):
                compact_summary = dict(summary)
                decisions = compact_summary.pop("conflict_decisions", None)
                if isinstance(decisions, list):
                    compact_summary["conflict_decision_count"] = len(decisions)
                compact_summary.setdefault(
                    "conflict_graph_ref",
                    {
                        "field": "task_conflict_graph",
                        "bundle_key": str(bundle_key),
                        "tables": ["conflict_edges", "planning_decisions"],
                    },
                )
                bundle["todo_vector_summary"] = compact_summary
            for field_name in {
                "conflict_graph",
                "conflict_planning_decisions",
                "dependency_dag",
                "task_conflict_graph",
                "task_dependency_graph",
                "task_planning_graph",
            }:
                bundle.pop(field_name, None)
            compact_bundles[str(bundle_key)] = bundle
        rendered["bundles"] = compact_bundles

    dependency_graph = _compact_dependency_graph(
        rendered.get("task_dependency_graph") or rendered.get("dependency_dag")
    )
    if dependency_graph:
        rendered["task_dependency_graph"] = dependency_graph
        rendered["dependency_dag"] = dependency_graph

    conflict_graph = compact_conflict_graph_projection(
        rendered.get("task_conflict_graph") or rendered.get("conflict_graph")
    )
    if conflict_graph:
        rendered["task_conflict_graph"] = conflict_graph
        if isinstance(conflict_graph.get("history"), Mapping):
            rendered.setdefault("conflict_history", dict(conflict_graph["history"]))
        rendered["conflict_graph"] = (
            dict(conflict_graph)
            if conflict_graph.get("compacted")
            else {
                "schema": str(conflict_graph.get("schema") or ""),
                "history": dict(conflict_graph.get("history") or {})
                if isinstance(conflict_graph.get("history"), Mapping)
                else {},
                "planning_evidence_ref": {
                    "field": "task_conflict_graph",
                    "tables": ["conflict_edges", "planning_decisions"],
                },
            }
        )

    planning_graph = _compact_task_planning_graph(rendered.get("task_planning_graph"))
    if planning_graph:
        rendered["task_planning_graph"] = planning_graph
    coverage_inputs = compact_coverage_inputs_projection(
        rendered.get("todo_coverage_inputs")
    )
    if coverage_inputs:
        rendered["todo_coverage_inputs"] = coverage_inputs
    return rendered


def write_bundle_index_artifact(
    path: Path | str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    portable_payload = _compact_bundle_index_payload(payload)
    database_payload = dict(payload)
    if isinstance(portable_payload.get("bundles"), Mapping):
        database_payload["bundles"] = portable_payload["bundles"]
    return write_queryable_artifact(
        path,
        portable_payload,
        kind=BUNDLE_INDEX_KIND,
        database_payload=database_payload,
    )


def write_scheduler_manifest_artifact(
    path: Path | str,
    payload: Mapping[str, Any],
    *,
    database_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return write_queryable_artifact(
        path,
        payload,
        kind=SCHEDULER_MANIFEST_KIND,
        database_payload=database_payload,
    )


def write_proof_metrics_artifact(
    path: Path | str, payload: Mapping[str, Any] | Any
) -> dict[str, Any]:
    """Write a bounded proof metrics JSON document and DuckDB sidecar.

    Typed :class:`ProofMetricsSnapshot` values and already-projected mappings
    are accepted.  Arbitrary proof contracts must first pass through
    ``build_proof_metrics_snapshot`` so raw evidence cannot accidentally be
    promoted into this public query plane.
    """

    rendered = (
        payload.to_dict()
        if not isinstance(payload, Mapping) and callable(getattr(payload, "to_dict", None))
        else dict(payload)
    )
    schema = str(rendered.get("schema") or "")
    if not schema.startswith("ipfs_accelerate_py.agent_supervisor.proof-metrics@"):
        raise ValueError("proof metrics artifacts require a bounded proof-metrics snapshot")
    from .proof_metrics import ProofMetricsSnapshot

    # Validation is repeated here because callers may use artifact_store
    # directly with a mapping instead of the typed snapshot wrapper.
    ProofMetricsSnapshot(rendered)
    return write_queryable_artifact(path, rendered, kind=PROOF_METRICS_KIND)


def _attestation_records(value: Any) -> tuple[Any, ...]:
    from .proof_attestation import PersistedAttestationRecord

    raw_values = (
        value
        if isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray, Mapping))
        else (value,)
    )
    records = []
    for item in raw_values:
        records.append(
            item
            if isinstance(item, PersistedAttestationRecord)
            else PersistedAttestationRecord.from_dict(item)
        )
    if not records:
        raise ValueError("at least one proof attestation record is required")
    identities = [record.record_id for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate proof attestation records are not allowed")
    return tuple(records)


def _ipfs_publish(
    backend: Any,
    payload: bytes,
    *,
    record_id: str,
) -> str:
    raw_block = False
    if callable(backend):
        result = backend(payload)
    elif callable(getattr(backend, "block_put", None)):
        raw_block = True
        result = backend.block_put(payload, codec="raw")
    elif callable(getattr(backend, "store", None)):
        result = backend.store(
            payload,
            filename=f"{record_id}.json",
            pin=True,
        )
    else:
        raise TypeError("IPFS publisher must be callable or provide block_put/store")
    if isinstance(result, Mapping):
        result = result.get("cid") or result.get("Hash") or result.get("hash")
    cid = str(result or "").strip()
    if not cid:
        raise ValueError("IPFS publisher returned an empty CID")
    if raw_block and cid != raw_ipfs_cid(payload):
        raise ValueError("IPFS publisher returned a CID for different raw content")
    return cid


def raw_ipfs_cid(payload: bytes) -> str:
    """Return the CIDv1/base32 identity of one raw SHA-256 IPFS block."""

    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("raw IPFS content must be bytes")
    # CIDv1, raw codec (0x55), sha2-256 multihash (0x12, 32 bytes).
    binary = b"\x01\x55\x12\x20" + hashlib.sha256(bytes(payload)).digest()
    return "b" + base64.b32encode(binary).decode("ascii").lower().rstrip("=")


def write_proof_attestation_artifact(
    path: Path | str,
    records: Any,
    *,
    ipfs_backend: Any | None = None,
    ipfs_publisher: Callable[[bytes], Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Write queryable public attestation records and optionally publish them.

    IPFS publication is best-effort and never affects the canonical local
    artifact.  Each publisher receives only a validated public record; private
    proving requests and witnesses cannot cross this boundary.
    """

    from .formal_verification_contracts import canonical_json_bytes

    typed = _attestation_records(records)
    publisher = ipfs_publisher if ipfs_publisher is not None else ipfs_backend
    rows: list[dict[str, Any]] = []
    for record in typed:
        row = record.to_public_artifact()
        cid = ""
        publication_error = ""
        if publisher is not None:
            try:
                cid = _ipfs_publish(
                    publisher,
                    canonical_json_bytes(row),
                    record_id=record.record_id,
                )
            except Exception as exc:
                # Exception messages can contain backend request bodies.  A
                # stable type-only code keeps this public projection secret-free.
                publication_error = f"ipfs_publication_{type(exc).__name__.lower()}"
        row["ipfs_cid"] = cid
        row["ipfs_publication_error"] = publication_error
        rows.append(row)
    payload = {
        "schema": PROOF_ATTESTATION_STORE_SCHEMA,
        "generated_at": generated_at
        or max(record.created_at for record in typed),
        # A portable artifact is evidence to replay, never a live trust root.
        "authoritative": False,
        "contains_hidden_witnesses": False,
        "attestation_count": len(rows),
        "ipfs_record_count": sum(bool(row["ipfs_cid"]) for row in rows),
        "attestations": rows,
    }
    return write_queryable_artifact(
        path,
        payload,
        kind=PROOF_ATTESTATION_KIND,
    )


def _ipfs_read(backend: Any, cid: str) -> bytes:
    if callable(getattr(backend, "block_get", None)):
        result = backend.block_get(cid)
    elif callable(getattr(backend, "retrieve", None)):
        result = backend.retrieve(cid)
    elif callable(getattr(backend, "cat", None)):
        result = backend.cat(cid)
    elif callable(backend):
        result = backend(cid)
    else:
        raise TypeError("IPFS reader must be callable or provide block_get/retrieve/cat")
    if isinstance(result, str):
        return result.encode("utf-8")
    if not isinstance(result, (bytes, bytearray)):
        raise ValueError("IPFS reader returned a non-byte payload")
    return bytes(result)


def read_proof_attestation_artifact(
    path_or_cid: Path | str,
    *,
    ipfs_backend: Any | None = None,
    verifier: Callable[[Any], bool] | None = None,
    checked_at: str | None = None,
) -> dict[str, Any]:
    """Read and revalidate public records from JSON, DuckDB path, or IPFS CID.

    Supplying ``verifier`` additionally reproduces every stored verdict.  A
    rejected, errored, or expired replay raises instead of trusting serialized
    assurance claims.
    """

    from .proof_attestation import (
        PersistedAttestationRecord,
        reproduce_attestation_verification,
    )

    requested = Path(path_or_cid)
    if requested.exists() or requested.suffix.lower() in {".json", ".duckdb"}:
        paths = query_artifact_paths(requested)
        payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or _artifact_kind(payload) != PROOF_ATTESTATION_KIND
        ):
            raise ValueError(f"not a proof attestation artifact: {paths.json_path}")
        raw_rows = payload.get("attestations")
        if not isinstance(raw_rows, list):
            raise ValueError("proof attestation artifact rows must be an array")
    else:
        if ipfs_backend is None:
            raise ValueError("an IPFS backend is required to read an attestation CID")
        raw = _ipfs_read(ipfs_backend, str(path_or_cid))
        if callable(getattr(ipfs_backend, "block_get", None)):
            if raw_ipfs_cid(raw) != str(path_or_cid):
                raise ValueError("IPFS raw block does not match its requested CID")
        decoded = json.loads(raw.decode("utf-8"))
        if not isinstance(decoded, Mapping):
            raise ValueError("IPFS attestation record must contain an object")
        raw_rows = [dict(decoded, ipfs_cid=str(path_or_cid))]
        payload = {
            "schema": PROOF_ATTESTATION_STORE_SCHEMA,
            "generated_at": str(decoded.get("created_at") or ""),
            "authoritative": False,
            "contains_hidden_witnesses": False,
            "attestation_count": 1,
            "ipfs_record_count": 1,
            "attestations": raw_rows,
        }

    validated_rows = []
    attested_count = 0
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("proof attestation row must be an object")
        record = PersistedAttestationRecord.from_dict(raw_row)
        current = (
            record.is_current_at(checked_at)
            if checked_at is not None
            else None
        )
        reproduced_authoritative = False
        if verifier is not None:
            if checked_at is None:
                raise ValueError("checked_at is required when reproducing verification")
            reproduced = reproduce_attestation_verification(
                record,
                verifier=verifier,
                checked_at=checked_at,
            )
            if not reproduced.authoritative:
                raise ValueError("persisted attestation failed independent reverification")
            reproduced_authoritative = True
            attested_count += 1
        rendered = record.to_public_artifact()
        rendered["ipfs_cid"] = str(raw_row.get("ipfs_cid") or "")
        rendered["ipfs_publication_error"] = str(
            raw_row.get("ipfs_publication_error") or ""
        )
        rendered["attestation_current"] = current
        rendered["reproduced_authoritative"] = reproduced_authoritative
        rendered["effective_assurance"] = (
            reproduced.authoritative_assurance.value
            if reproduced_authoritative
            else record.receipt.authoritative_assurance.value
        )
        validated_rows.append(rendered)
    claimed_count = payload.get("attestation_count")
    if claimed_count not in (None, len(validated_rows)):
        raise ValueError("proof attestation count does not match artifact rows")
    actual_ipfs_count = sum(bool(row["ipfs_cid"]) for row in validated_rows)
    claimed_ipfs_count = payload.get("ipfs_record_count")
    if claimed_ipfs_count not in (None, actual_ipfs_count):
        raise ValueError("IPFS attestation count does not match artifact rows")
    if payload.get("contains_hidden_witnesses") not in (None, False):
        raise ValueError("proof attestation artifacts cannot contain hidden witnesses")
    if payload.get("authoritative") not in (None, False):
        raise ValueError("proof attestation authority label is inconsistent")
    result = dict(payload)
    result["authoritative"] = False
    result["attestations"] = validated_rows
    result["attestation_count"] = len(validated_rows)
    result["attested_record_count"] = attested_count
    result["attested_assurance_available"] = bool(
        validated_rows and attested_count == len(validated_rows)
    )
    return result


def query_proof_attestations(path: Path | str, **query: Any) -> dict[str, Any]:
    """Execute a bounded query against the public attestation projection."""

    supplied_kind = query.pop("kind", PROOF_ATTESTATION_KIND)
    if supplied_kind != PROOF_ATTESTATION_KIND:
        raise ValueError("proof attestation queries require proof_attestations kind")
    query.setdefault("table", "proof_attestations")
    return query_artifact(path, kind=PROOF_ATTESTATION_KIND, **query)


write_attestation_artifact = write_proof_attestation_artifact
read_attestation_artifact = read_proof_attestation_artifact
query_proof_attestation_artifact = query_proof_attestations
write_proof_attestation_store = write_proof_attestation_artifact
read_proof_attestation_store = read_proof_attestation_artifact


def read_proof_metrics_artifact(path: Path | str) -> dict[str, Any]:
    """Read and validate the portable JSON representation of proof metrics."""

    paths = query_artifact_paths(path)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or _artifact_kind(payload) != PROOF_METRICS_KIND:
        raise ValueError(f"not a proof metrics artifact: {paths.json_path}")
    from .proof_metrics import ProofMetricsSnapshot

    return ProofMetricsSnapshot(payload).to_dict()


def query_proof_metrics(path: Path | str, **query: Any) -> dict[str, Any]:
    """Execute one bounded query against a proof metrics artifact."""

    supplied_kind = query.pop("kind", PROOF_METRICS_KIND)
    if supplied_kind != PROOF_METRICS_KIND:
        raise ValueError("proof metrics queries require proof_metrics kind")
    return query_artifact(path, kind=PROOF_METRICS_KIND, **query)


def write_code_evidence_graph_artifact(
    path: Path | str, payload: Any
) -> dict[str, Any]:
    """Write a validated code-evidence graph to paired JSON and DuckDB files."""

    from .code_evidence_graph import CodeEvidenceGraph

    graph = (
        payload
        if isinstance(payload, CodeEvidenceGraph)
        else CodeEvidenceGraph.from_dict(payload)
    )
    return write_queryable_artifact(
        path, graph.to_dict(), kind=CODE_EVIDENCE_GRAPH_KIND
    )


def _stable_file_identity(path: Path) -> tuple[os.stat_result, str] | None:
    """Return a stable stat/digest pair, or ``None`` during replacement."""

    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError:
        return None
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        return None
    return after, digest.hexdigest()


def _bundle_tables_match_source(
    connection: Any,
    payload: Mapping[str, Any],
) -> bool:
    """Prove scheduler-consumed bundle/task rows equal the paired JSON."""

    raw_bundles = payload.get("bundles")
    if not isinstance(raw_bundles, Mapping):
        return False
    expected: dict[str, dict[str, Any]] = {}
    for bundle_key, raw_bundle in raw_bundles.items():
        if not isinstance(raw_bundle, Mapping):
            return False
        bundle = dict(raw_bundle)
        tasks = bundle.pop("tasks", [])
        if not isinstance(tasks, list) or not all(
            isinstance(task, Mapping) for task in tasks
        ):
            return False
        bundle["tasks"] = [dict(task) for task in tasks]
        expected[str(bundle_key)] = bundle
    try:
        bundle_rows = connection.execute(
            "SELECT bundle_key, payload_json FROM bundles ORDER BY bundle_key"
        ).fetchall()
        task_rows = connection.execute(
            "SELECT bundle_key, payload_json "
            "FROM bundle_tasks ORDER BY bundle_key, task_ordinal"
        ).fetchall()
    except Exception:
        return False
    observed = {
        str(bundle_key): {
            **_json_value(str(payload_json)),
            "tasks": [],
        }
        for bundle_key, payload_json in bundle_rows
    }
    for bundle_key, payload_json in task_rows:
        bundle = observed.get(str(bundle_key))
        if bundle is None:
            return False
        bundle["tasks"].append(_json_value(str(payload_json)))
    return observed == expected


def _database_fresh(database_path: Path, source_path: Path, kind: str | None) -> bool:
    if not database_path.exists() or not source_path.exists():
        return False
    source_identity = _stable_file_identity(source_path)
    if source_identity is None:
        return False
    source_stat, source_sha256 = source_identity
    duckdb = _duckdb_module()
    try:
        connection = duckdb.connect(str(database_path), read_only=True)
        try:
            row = connection.execute(
                "SELECT artifact_kind, schema_version, source_sha256, "
                "source_size, source_mtime_ns "
                "FROM artifact_catalog LIMIT 1"
            ).fetchone()
            basic_match = bool(
                row
                and (kind is None or str(row[0]) == kind)
                and str(row[1]) == QUERY_SCHEMA
                and str(row[2]) == source_sha256
                and int(row[3]) == source_stat.st_size
                and int(row[4]) == source_stat.st_mtime_ns
            )
            if basic_match and str(row[0]) == BUNDLE_INDEX_KIND:
                source_payload, verified_stat, verified_sha256 = (
                    _read_stable_json(source_path)
                )
                basic_match = bool(
                    verified_sha256 == source_sha256
                    and verified_stat.st_size == source_stat.st_size
                    and verified_stat.st_mtime_ns == source_stat.st_mtime_ns
                    and _bundle_tables_match_source(
                        connection,
                        source_payload,
                    )
                )
        finally:
            connection.close()
    except Exception:
        return False
    return basic_match


def _read_stable_json(
    path: Path,
) -> tuple[Mapping[str, Any], os.stat_result, str]:
    """Read one atomic JSON generation even when another process is replacing it."""

    for _attempt in range(3):
        before = path.stat()
        text = path.read_text(encoding="utf-8")
        after = path.stat()
        if (
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            continue
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError(f"artifact JSON must contain an object: {path}")
        source_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return payload, after, source_sha256
    raise RuntimeError(f"artifact changed repeatedly while being read: {path}")


def ensure_query_database(path: Path | str, *, kind: str | None = None) -> Path:
    """Return a current DuckDB representation for a JSON or DuckDB artifact."""

    requested = Path(path).resolve()
    paths = query_artifact_paths(requested)
    if requested.suffix.lower() == ".duckdb":
        if not requested.exists():
            raise FileNotFoundError(requested)
        duckdb = _duckdb_module()
        connection = duckdb.connect(str(requested), read_only=True)
        try:
            row = connection.execute(
                "SELECT artifact_kind, schema_version, source_sha256, "
                "source_size FROM artifact_catalog LIMIT 1"
            ).fetchone()
            if row and paths.json_path.exists():
                source_payload, source_stat, source_sha256 = _read_stable_json(
                    paths.json_path
                )
                if (
                    str(row[2]) != source_sha256
                    or int(row[3]) != source_stat.st_size
                ):
                    raise ValueError(
                        "paired JSON/DuckDB source digest mismatch: "
                        f"{requested}"
                    )
                observed_kind = str(row[0])
                if (
                    observed_kind == BUNDLE_INDEX_KIND
                    and not _bundle_tables_match_source(
                        connection,
                        source_payload,
                    )
                ):
                    raise ValueError(
                        "paired JSON/DuckDB bundle projection mismatch: "
                        f"{requested}"
                    )
        finally:
            connection.close()
        if (
            row
            and str(row[1]) == QUERY_SCHEMA
            and (kind is None or str(row[0]) == kind)
        ):
            return requested
        if paths.json_path.exists():
            return ensure_query_database(paths.json_path, kind=kind)
        actual_kind = str(row[0]) if row else "unknown"
        actual_schema = str(row[1]) if row else "unknown"
        raise ValueError(
            f"expected {kind or actual_kind} {QUERY_SCHEMA} DuckDB artifact, "
            f"found {actual_kind} {actual_schema}: {requested}"
        )
    with _artifact_write_lock(paths.duckdb_path):
        if _database_fresh(paths.duckdb_path, paths.json_path, kind):
            return paths.duckdb_path
        payload, source_stat, source_sha256 = _read_stable_json(paths.json_path)
        resolved_kind = kind or _artifact_kind(payload)
        _write_duckdb(
            paths.duckdb_path,
            payload,
            kind=resolved_kind,
            source_path=paths.json_path,
            source_sha256=source_sha256,
            source_size=source_stat.st_size,
            source_mtime_ns=source_stat.st_mtime_ns,
        )
    return paths.duckdb_path


def read_artifact_fields(
    path: Path | str,
    field_names: Sequence[str],
    *,
    kind: str | None = None,
) -> dict[str, Any]:
    """Read selected top-level fields without decoding the full artifact."""

    if not field_names:
        return {}
    database_path = ensure_query_database(path, kind=kind)
    duckdb = _duckdb_module()
    connection = _configure_duckdb_connection(
        duckdb.connect(str(database_path), read_only=True)
    )
    try:
        placeholders = ", ".join("?" for _ in field_names)
        rows = connection.execute(
            f"SELECT field_name, value_json FROM artifact_fields WHERE field_name IN ({placeholders})",
            list(field_names),
        ).fetchall()
    finally:
        connection.close()
    return {str(name): _json_value(str(value)) for name, value in rows}


def read_bundle_index_projection(
    path: Path | str,
    *,
    field_names: Sequence[str] = ("source_todo",),
    bundle_omit_fields: Sequence[str] = (),
    task_omit_fields: Sequence[str] = (),
) -> dict[str, Any]:
    """Read bundle rows plus only the requested top-level planning fields.

    Optional omissions are applied inside DuckDB with JSON merge patches, so
    callers can avoid transferring and decoding fields irrelevant to planning.
    """

    database_path = ensure_query_database(path, kind=BUNDLE_INDEX_KIND)
    duckdb = _duckdb_module()
    connection = _configure_duckdb_connection(
        duckdb.connect(str(database_path), read_only=True)
    )
    try:
        bundle_expression = "payload_json"
        bundle_parameters: list[str] = []
        if bundle_omit_fields:
            bundle_expression = "json_merge_patch(payload_json, ?)"
            bundle_parameters.append(
                _json_text(
                    {
                        str(field): None
                        for field in dict.fromkeys(bundle_omit_fields)
                        if str(field).strip()
                    }
                )
            )
        bundle_rows = connection.execute(
            f"SELECT bundle_key, {bundle_expression} "
            "FROM bundles ORDER BY bundle_key",
            bundle_parameters,
        ).fetchall()
        task_expression = "payload_json"
        task_parameters: list[str] = []
        if task_omit_fields:
            task_expression = "json_merge_patch(payload_json, ?)"
            task_parameters.append(
                _json_text(
                    {
                        str(field): None
                        for field in dict.fromkeys(task_omit_fields)
                        if str(field).strip()
                    }
                )
            )
        task_rows = connection.execute(
            f"SELECT bundle_key, {task_expression} "
            "FROM bundle_tasks ORDER BY bundle_key, task_ordinal",
            task_parameters,
        ).fetchall()
        fields: dict[str, Any] = {}
        if field_names:
            placeholders = ", ".join("?" for _ in field_names)
            for name, value in connection.execute(
                f"SELECT field_name, value_json FROM artifact_fields WHERE field_name IN ({placeholders})",
                list(field_names),
            ).fetchall():
                fields[str(name)] = _json_value(str(value))
    finally:
        connection.close()
    bundles = {str(key): _json_value(str(value)) for key, value in bundle_rows}
    for bundle_key, value in task_rows:
        bundles.setdefault(str(bundle_key), {})
        bundles[str(bundle_key)].setdefault("tasks", []).append(_json_value(str(value)))
    return {**fields, "bundles": bundles}


def read_bundle_index_planning_projection(
    path: Path | str,
    *,
    field_names: Sequence[str] = ("source_todo",),
) -> dict[str, Any]:
    """Read the bounded task fields required to rebuild a scheduler plan."""

    return read_bundle_index_projection(
        path,
        field_names=field_names,
        bundle_omit_fields=BUNDLE_PLANNING_BUNDLE_OMIT_FIELDS,
        task_omit_fields=BUNDLE_PLANNING_TASK_OMIT_FIELDS,
    )


def read_bundle_index_artifact(path: Path | str) -> dict[str, Any]:
    """Reconstruct a complete bundle index from either representation."""

    database_path = ensure_query_database(path, kind=BUNDLE_INDEX_KIND)
    duckdb = _duckdb_module()
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        field_names = [
            str(row[0])
            for row in connection.execute(
                "SELECT field_name FROM artifact_fields ORDER BY field_name"
            ).fetchall()
        ]
    finally:
        connection.close()
    return read_bundle_index_projection(path, field_names=field_names)


def read_code_evidence_graph_projection(path: Path | str) -> dict[str, Any]:
    """Reconstruct canonical graph records from either artifact representation."""

    database_path = ensure_query_database(path, kind=CODE_EVIDENCE_GRAPH_KIND)
    duckdb = _duckdb_module()
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        rows = connection.execute(
            "SELECT record_type, payload_json FROM graph_records "
            "ORDER BY record_type DESC, record_ordinal, record_id"
        ).fetchall()
        fields = {
            str(name): _json_value(str(value))
            for name, value in connection.execute(
                "SELECT field_name, value_json FROM artifact_fields"
            ).fetchall()
        }
    finally:
        connection.close()
    nodes: list[Any] = []
    edges: list[Any] = []
    for record_type, value in rows:
        (nodes if str(record_type) == "node" else edges).append(_json_value(str(value)))
    from .code_evidence_graph import CodeEvidenceGraph

    graph = CodeEvidenceGraph.from_dict({**fields, "nodes": nodes, "edges": edges})
    return graph.to_dict()


def read_code_evidence_graph_artifact(path: Path | str) -> dict[str, Any]:
    """Compatibility spelling for the lossless graph projection reader."""

    return read_code_evidence_graph_projection(path)


def read_code_evidence_graph(path: Path | str) -> Any:
    """Return a typed graph reconstructed from JSON or DuckDB."""

    from .code_evidence_graph import CodeEvidenceGraph

    return CodeEvidenceGraph.from_dict(read_code_evidence_graph_projection(path))


def canonical_code_evidence_graph_records(
    path: Path | str,
) -> dict[str, list[dict[str, Any]]]:
    """Read only the canonical records used to compare graph projections."""

    return read_code_evidence_graph(path).canonical_records()


# Concise compatibility spellings for callers whose artifact type is already
# clear from context.
write_evidence_graph_artifact = write_code_evidence_graph_artifact
read_evidence_graph_artifact = read_code_evidence_graph_artifact
read_evidence_graph_projection = read_code_evidence_graph_projection
canonical_evidence_graph_records = canonical_code_evidence_graph_records


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    return value


def _validated_identifier(value: str, *, label: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def query_artifact(
    path: Path | str,
    *,
    table: str | None = None,
    columns: Sequence[str] = ("*",),
    where: str = "",
    sql: str = "",
    limit: int = 50,
    kind: str | None = None,
) -> dict[str, Any]:
    """Execute one read-only, row-bounded query against either artifact format."""

    row_limit = max(1, min(int(limit), MAX_QUERY_ROWS))
    database_path = ensure_query_database(path, kind=kind)
    if sql:
        statement = sql.strip().rstrip(";").strip()
        if ";" in statement or not _READ_ONLY_SQL.match(statement):
            raise ValueError(
                "only one read-only SELECT/WITH/DESCRIBE/SHOW query is allowed"
            )
        if re.match(r"^(?:select|with)\b", statement, re.IGNORECASE):
            statement = f"SELECT * FROM ({statement}) AS bounded_artifact_query LIMIT {row_limit + 1}"
    else:
        selected_table = _validated_identifier(
            table or "artifact_catalog", label="table name"
        )
        if columns == ("*",) or list(columns) == ["*"]:
            selected_columns = "*"
        else:
            selected_columns = ", ".join(
                _validated_identifier(column, label="column name") for column in columns
            )
        if ";" in where:
            raise ValueError("where clauses may not contain statement separators")
        statement = f"SELECT {selected_columns} FROM {selected_table}"
        if where.strip():
            statement += f" WHERE {where.strip()}"
        statement += f" LIMIT {row_limit + 1}"

    duckdb = _duckdb_module()
    connection = duckdb.connect(str(database_path), read_only=True)
    try:
        cursor = connection.execute(statement)
        names = [str(item[0]) for item in cursor.description or ()]
        values = cursor.fetchmany(row_limit + 1)
    finally:
        connection.close()
    truncated = len(values) > row_limit
    rows = [
        {name: _jsonable(value) for name, value in zip(names, row)}
        for row in values[:row_limit]
    ]
    return {
        "schema": QUERY_SCHEMA,
        "duckdb_path": str(database_path),
        "columns": names,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "limit": row_limit,
    }


def _logical_artifact_digest(payload: Mapping[str, Any]) -> str:
    """Address portable content without its location-dependent sidecar hint."""

    logical_payload = dict(payload)
    logical_payload.pop("query_store", None)
    try:
        encoded = json.dumps(
            logical_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "queryable artifact must contain canonical JSON values"
        ) from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _queryable_artifact_snapshot(
    path: Path | str,
    *,
    kind: str | None,
) -> QueryableArtifactReference:
    """Capture one verified paired generation, rebuilding its sidecar if needed."""

    paths = query_artifact_paths(path)
    if not paths.json_path.exists():
        raise FileNotFoundError(
            "a paired JSON source is required to verify a queryable artifact: "
            f"{paths.json_path}"
        )
    for _attempt in range(3):
        # Always enter through JSON.  In addition to creating a missing
        # sidecar, this replaces a corrupt DuckDB database from its portable
        # source under the existing cross-process artifact lock.
        database_path = ensure_query_database(paths.json_path, kind=kind)
        payload, source_stat, source_sha256 = _read_stable_json(paths.json_path)
        resolved_kind = kind or _artifact_kind(payload)
        if not _database_fresh(
            database_path, paths.json_path, resolved_kind
        ):
            continue

        duckdb = _duckdb_module()
        try:
            connection = duckdb.connect(str(database_path), read_only=True)
            try:
                rows = connection.execute(
                    "SELECT artifact_kind, schema_version, source_sha256, "
                    "source_size, source_mtime_ns "
                    "FROM artifact_catalog"
                ).fetchall()
            finally:
                connection.close()
        except Exception:
            # A sidecar can be replaced or damaged after the freshness check.
            # The next pass asks ensure_query_database to recover it.
            continue
        if len(rows) != 1:
            raise ValueError(
                "queryable artifact catalog must contain exactly one record"
            )
        catalog = rows[0]
        if (
            str(catalog[0]) != resolved_kind
            or str(catalog[1]) != QUERY_SCHEMA
            or str(catalog[2]) != source_sha256
            or int(catalog[3]) != source_stat.st_size
            or int(catalog[4]) != source_stat.st_mtime_ns
        ):
            continue
        final_identity = _stable_file_identity(paths.json_path)
        if final_identity is None:
            continue
        final_stat, final_sha256 = final_identity
        if (
            final_sha256 != source_sha256
            or final_stat.st_size != source_stat.st_size
            or final_stat.st_mtime_ns != source_stat.st_mtime_ns
        ):
            continue

        digest = _logical_artifact_digest(payload)
        return QueryableArtifactReference(
            artifact_id=f"queryable-artifact:{digest}",
            digest=digest,
            path=str(paths.json_path),
            kind=resolved_kind,
            schema=str(payload.get("schema") or QUERY_SCHEMA),
            size_bytes=source_stat.st_size,
            source_sha256=source_sha256,
            duckdb_path=str(database_path),
        )
    raise RuntimeError(
        f"queryable artifact changed repeatedly while being verified: {paths.json_path}"
    )


def queryable_artifact_reference(
    path: Path | str,
    *,
    kind: str | None = None,
) -> QueryableArtifactReference:
    """Return a canonical, body-free reference to an existing artifact."""

    return _queryable_artifact_snapshot(path, kind=kind)


def _adapter_reference_constraints(
    reference: QueryableArtifactReference | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(reference, QueryableArtifactReference):
        return reference.to_dict()
    if not isinstance(reference, Mapping):
        raise ValueError("queryable artifact reference must be an object")
    allowed = {
        "artifact_id",
        "digest",
        "path",
        "kind",
        "schema",
        "size_bytes",
        "source_sha256",
        "duckdb_path",
    }
    unknown = sorted(set(reference).difference(allowed))
    if unknown:
        raise ValueError(
            "queryable artifact reference has unsupported fields: "
            + ", ".join(unknown)
        )
    required = {"artifact_id", "digest", "path"}
    missing = sorted(required.difference(reference))
    if missing:
        raise ValueError(
            "queryable artifact reference is missing identity fields: "
            + ", ".join(missing)
        )
    constraints = dict(reference)
    if not _SHA256_DIGEST.fullmatch(str(constraints["digest"])):
        raise ValueError("queryable artifact digest must be sha256:<hex>")
    if constraints["artifact_id"] != (
        f"queryable-artifact:{constraints['digest']}"
    ):
        raise ValueError("queryable artifact identity does not match its digest")
    supplied_path = Path(str(constraints["path"]))
    if not supplied_path.is_absolute():
        raise ValueError("queryable artifact path must be absolute")
    constraints["path"] = str(supplied_path.resolve())
    if "duckdb_path" in constraints:
        supplied_database = Path(str(constraints["duckdb_path"]))
        if not supplied_database.is_absolute():
            raise ValueError("queryable artifact duckdb_path must be absolute")
        constraints["duckdb_path"] = str(supplied_database.resolve())
    return constraints


def _bounded_adapter_bytes(value: Any, *, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return min(value, maximum)


class QueryableArtifactCASAdapter:
    """Read-only CAS adapter over the existing JSON/DuckDB artifact pair.

    The adapter does not assign cache authority and never returns a complete
    artifact body.  Runtime stores can retain its shallow reference while
    namespace-specific code continues to query the existing normalized tables.
    """

    def __init__(self, path: Path | str, *, kind: str | None = None) -> None:
        self.paths = query_artifact_paths(path)
        self.kind = kind

    def reference(self) -> QueryableArtifactReference:
        """Return the currently verified source generation."""

        return _queryable_artifact_snapshot(
            self.paths.json_path,
            kind=self.kind,
        )

    def verify(
        self,
        reference: QueryableArtifactReference | Mapping[str, Any],
    ) -> bool:
        """Return whether a body-free reference still names the exact source."""

        try:
            constraints = _adapter_reference_constraints(reference)
            current = self.reference().to_dict()
        except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
            return False
        return all(current.get(name) == value for name, value in constraints.items())

    def _verified_reference(
        self,
        reference: QueryableArtifactReference | Mapping[str, Any] | None,
    ) -> QueryableArtifactReference:
        expected = self.reference() if reference is None else reference
        if not self.verify(expected):
            raise ValueError(
                "queryable artifact reference does not match the current source"
            )
        return (
            expected
            if isinstance(expected, QueryableArtifactReference)
            else self.reference()
        )

    def read(
        self,
        reference: QueryableArtifactReference | Mapping[str, Any] | None = None,
        *,
        field_names: Sequence[str] = (),
        fields: Sequence[str] | None = None,
        max_bytes: int = MAX_ADAPTER_READ_BYTES,
    ) -> dict[str, Any]:
        """Read selected top-level fields with hard field and byte bounds.

        With no requested fields this returns verification metadata only.  It
        intentionally has no spelling that loads the complete JSON body.
        """

        selected_fields = tuple(field_names)
        if fields is not None:
            if selected_fields:
                raise ValueError("use either field_names or fields, not both")
            selected_fields = tuple(fields)
        if len(selected_fields) > MAX_ADAPTER_READ_FIELDS:
            raise ValueError("queryable artifact field selection exceeds its bound")
        if len(set(selected_fields)) != len(selected_fields):
            raise ValueError("queryable artifact fields must be unique")
        if any(
            not isinstance(field, str) or not field.strip()
            for field in selected_fields
        ):
            raise ValueError("queryable artifact fields must be nonempty strings")
        byte_limit = _bounded_adapter_bytes(
            max_bytes,
            maximum=MAX_ADAPTER_READ_BYTES,
            label="max_bytes",
        )
        verified = self._verified_reference(reference)
        if not selected_fields:
            return verified.to_dict()
        result = read_artifact_fields(
            verified.path,
            selected_fields,
            kind=verified.kind,
        )
        if len(_json_text(result).encode("utf-8")) > byte_limit:
            raise ValueError("queryable artifact field projection exceeds max_bytes")
        if not self.verify(verified):
            raise RuntimeError("queryable artifact changed during bounded read")
        return result

    def query(
        self,
        reference: QueryableArtifactReference | Mapping[str, Any] | None = None,
        *,
        max_bytes: int = MAX_ADAPTER_QUERY_BYTES,
        **query: Any,
    ) -> dict[str, Any]:
        """Run an existing row-bounded query with an additional byte bound."""

        byte_limit = _bounded_adapter_bytes(
            max_bytes,
            maximum=MAX_ADAPTER_QUERY_BYTES,
            label="max_bytes",
        )
        verified = self._verified_reference(reference)
        supplied_kind = query.pop("kind", verified.kind)
        if supplied_kind != verified.kind:
            raise ValueError("query kind does not match the artifact reference")
        result = query_artifact(
            verified.path,
            kind=verified.kind,
            **query,
        )
        if len(_json_text(result).encode("utf-8")) > byte_limit:
            raise ValueError("queryable artifact query result exceeds max_bytes")
        if not self.verify(verified):
            raise RuntimeError("queryable artifact changed during bounded query")
        return result


def queryable_artifact_adapter(
    path: Path | str,
    *,
    kind: str | None = None,
) -> QueryableArtifactCASAdapter:
    """Construct a read-only adapter for an existing queryable artifact."""

    return QueryableArtifactCASAdapter(path, kind=kind)


# Compatibility spellings for callers that name the adapted subsystem or the
# consuming runtime first.
ArtifactStoreCASAdapter = QueryableArtifactCASAdapter
RuntimeCASArtifactStoreAdapter = QueryableArtifactCASAdapter


def query_code_evidence_graph(
    path: Path | str,
    **query: Any,
) -> dict[str, Any]:
    """Execute one bounded query against a code-evidence graph artifact."""

    supplied_kind = query.pop("kind", CODE_EVIDENCE_GRAPH_KIND)
    if supplied_kind != CODE_EVIDENCE_GRAPH_KIND:
        raise ValueError("code evidence graph queries require code_evidence_graph kind")
    return query_artifact(path, kind=CODE_EVIDENCE_GRAPH_KIND, **query)


def _exact_strings(values: Sequence[str] | str, *, label: str) -> tuple[str, ...]:
    """Normalize exact-match selectors without accepting wildcard spellings."""

    raw_values: Sequence[str]
    if isinstance(values, str):
        raw_values = (values,)
    else:
        raw_values = values
    result = tuple(
        sorted({str(value).strip() for value in raw_values if str(value).strip()})
    )
    if any(value in {"*", "%"} for value in result):
        raise ValueError(f"{label} selectors must be exact identifiers")
    return result


def query_code_evidence_neighborhood(
    path: Path | str,
    *,
    task_id: str,
    symbols: Sequence[str] | str = (),
    dependency_task_ids: Sequence[str] | str = (),
    obligation_ids: Sequence[str] | str = (),
    receipt_ids: Sequence[str] | str = (),
    contradiction_ids: Sequence[str] | str = (),
    max_hops: int = 2,
    limit: int = 100,
) -> dict[str, Any]:
    """Return a deterministic, exact, row-bounded proof neighborhood.

    This is intentionally narrower than arbitrary graph traversal.  Repository
    tree and AST-blob nodes are never traversed, receipt/transcript children are
    terminal, and only proof-relevant edge directions may expand a seed.  The
    resulting query is therefore safe to feed into a context reducer without
    first materializing the complete evidence graph.
    """

    exact_task = str(task_id or "").strip()
    if not exact_task or exact_task in {"*", "%"}:
        raise ValueError("task_id must be one exact identifier")
    exact_symbols = _exact_strings(symbols, label="symbol")
    exact_dependencies = _exact_strings(dependency_task_ids, label="dependency task")
    exact_obligations = _exact_strings(obligation_ids, label="obligation")
    exact_receipts = _exact_strings(receipt_ids, label="receipt")
    exact_contradictions = _exact_strings(contradiction_ids, label="contradiction")
    hop_limit = int(max_hops)
    if hop_limit < 0 or hop_limit > MAX_GRAPH_QUERY_HOPS:
        raise ValueError(f"max_hops must be between 0 and {MAX_GRAPH_QUERY_HOPS}")
    row_limit = max(1, min(int(limit), MAX_QUERY_ROWS))
    database_path = ensure_query_database(path, kind=CODE_EVIDENCE_GRAPH_KIND)
    duckdb = _duckdb_module()

    seed_clauses = [
        "(node_kind = 'task' AND task_id = ?)",
        # Enrichments deliberately carry no authoritative task index.  Select
        # them by their exact declared target, not by graph alias inference.
        "(node_kind = 'enrichment' AND ("
        "json_extract_string(payload_json, '$.record.target') = ? OR "
        "json_extract_string(payload_json, '$.record.target_id') = ? OR "
        "json_contains(json_extract(payload_json, '$.record.targets'), ?) OR "
        "json_contains(json_extract(payload_json, '$.record.target_ids'), ?)"
        "))",
    ]
    parameters: list[Any] = [
        exact_task,
        exact_task,
        exact_task,
        json.dumps(exact_task),
        json.dumps(exact_task),
    ]

    def add_in(clause: str, values: tuple[str, ...]) -> None:
        if not values:
            return
        placeholders = ", ".join("?" for _ in values)
        seed_clauses.append(clause.format(placeholders=placeholders))
        parameters.extend(values)

    add_in("(node_kind = 'symbol' AND symbol IN ({placeholders}))", exact_symbols)
    add_in(
        "(node_kind = 'task' AND task_id IN ({placeholders}))",
        exact_dependencies,
    )
    add_in(
        "(node_kind = 'obligation' AND obligation_id IN ({placeholders}))",
        exact_obligations,
    )
    add_in(
        "(node_kind IN ('proof', 'validation', 'merge') "
        "AND record_key IN ({placeholders}))",
        exact_receipts,
    )
    if exact_contradictions:
        placeholders = ", ".join("?" for _ in exact_contradictions)
        seed_clauses.append(
            "("
            f"record_key IN ({placeholders}) OR "
            f"json_extract_string(payload_json, '$.record.contradiction_id') IN ({placeholders}) OR "
            f"json_extract_string(payload_json, '$.record.source_receipt_id') IN ({placeholders})"
            ")"
        )
        parameters.extend(exact_contradictions)
        parameters.extend(exact_contradictions)
        parameters.extend(exact_contradictions)

    node_columns = (
        "node_id, node_kind, record_key, provenance, authoritative, task_id, "
        "tree_id, symbol, obligation_id, assurance, freshness, payload_json"
    )

    def node_dict(row: Sequence[Any]) -> dict[str, Any]:
        names = (
            "node_id",
            "node_kind",
            "record_key",
            "provenance",
            "authoritative",
            "task_id",
            "tree_id",
            "symbol",
            "obligation_id",
            "assurance",
            "freshness",
            "payload_json",
        )
        value = {name: _jsonable(item) for name, item in zip(names, row)}
        value["payload"] = _json_value(str(value.pop("payload_json")))
        return value

    def edge_dict(row: Sequence[Any]) -> dict[str, Any]:
        names = (
            "edge_id",
            "source_node_id",
            "target_node_id",
            "edge_kind",
            "provenance",
            "provenance_record_id",
            "authoritative",
            "payload_json",
        )
        value = {name: _jsonable(item) for name, item in zip(names, row)}
        value["payload"] = _json_value(str(value.pop("payload_json")))
        return value

    connection = duckdb.connect(str(database_path), read_only=True)
    truncated = False
    try:
        seed_rows = connection.execute(
            f"SELECT {node_columns} FROM evidence_nodes WHERE "
            + " OR ".join(seed_clauses)
            + " ORDER BY node_kind, record_key, node_id "
            + f"LIMIT {row_limit + 1}",
            parameters,
        ).fetchall()
        if len(seed_rows) > row_limit:
            truncated = True
            seed_rows = seed_rows[:row_limit]
        selected = {str(row[0]): node_dict(row) for row in seed_rows}
        seed_ids = frozenset(selected)
        frontier = set(seed_ids)

        # Legal directional expansions by current node kind.  In particular,
        # TARGETS_TREE/CONTAINS/DEFINES_SYMBOL never expand a context query.
        allowed: dict[tuple[str, str, str], frozenset[str]] = {
            ("task", "out", "depends_on"): frozenset({"task"}),
            ("task", "out", "has_obligation"): frozenset({"obligation"}),
            ("task", "in", "validates"): frozenset({"validation"}),
            ("task", "in", "merged"): frozenset({"merge"}),
            ("task", "in", "completes"): frozenset({"merge"}),
            ("task", "in", "mentions"): frozenset({"enrichment"}),
            ("task", "in", "suggests"): frozenset({"enrichment"}),
            ("task", "in", "related_to"): frozenset({"enrichment"}),
            ("obligation", "out", "depends_on"): frozenset({"obligation"}),
            ("obligation", "out", "covers"): frozenset({"symbol"}),
            ("obligation", "in", "proves"): frozenset({"proof"}),
            ("obligation", "in", "derived_from"): frozenset({"proof"}),
            ("obligation", "in", "covers"): frozenset({"validation"}),
            ("symbol", "in", "covers"): frozenset({"obligation"}),
        }
        # An explicitly selected receipt may lead back to its exact subject,
        # but receipts discovered during traversal remain terminal.
        receipt_seed_expansions = {
            ("proof", "out", "proves"): frozenset({"obligation"}),
            ("proof", "out", "derived_from"): frozenset({"obligation"}),
            ("validation", "out", "covers"): frozenset({"obligation"}),
            ("validation", "out", "validates"): frozenset({"task"}),
            ("merge", "out", "merged"): frozenset({"task"}),
            ("merge", "out", "completes"): frozenset({"task"}),
            ("enrichment", "out", "mentions"): frozenset(
                {"task", "obligation", "symbol"}
            ),
            ("enrichment", "out", "suggests"): frozenset(
                {"task", "obligation", "symbol"}
            ),
            ("enrichment", "out", "related_to"): frozenset(
                {"task", "obligation", "symbol"}
            ),
        }
        candidate_edges: dict[str, dict[str, Any]] = {}
        for _hop in range(hop_limit):
            if not frontier or len(selected) >= row_limit:
                truncated = truncated or bool(frontier)
                break
            placeholders = ", ".join("?" for _ in frontier)
            edge_rows = connection.execute(
                "SELECT edge_id, source_node_id, target_node_id, edge_kind, "
                "provenance, provenance_record_id, authoritative, payload_json "
                "FROM evidence_edges "
                f"WHERE source_node_id IN ({placeholders}) "
                f"OR target_node_id IN ({placeholders}) "
                "ORDER BY edge_id "
                f"LIMIT {MAX_QUERY_ROWS + 1}",
                [*sorted(frontier), *sorted(frontier)],
            ).fetchall()
            if len(edge_rows) > MAX_QUERY_ROWS:
                truncated = True
                edge_rows = edge_rows[:MAX_QUERY_ROWS]
            neighbor_ids = sorted(
                {
                    str(row[index])
                    for row in edge_rows
                    for index in (1, 2)
                    if str(row[index]) not in selected
                }
            )
            if not neighbor_ids:
                break
            node_placeholders = ", ".join("?" for _ in neighbor_ids)
            neighbor_rows = connection.execute(
                f"SELECT {node_columns} FROM evidence_nodes "
                f"WHERE node_id IN ({node_placeholders}) ORDER BY node_id "
                f"LIMIT {MAX_QUERY_ROWS + 1}",
                neighbor_ids,
            ).fetchall()
            neighbor_map = {
                str(row[0]): node_dict(row) for row in neighbor_rows[:MAX_QUERY_ROWS]
            }
            accepted: set[str] = set()
            for row in edge_rows:
                edge = edge_dict(row)
                source_id = str(row[1])
                target_id = str(row[2])
                edge_kind = str(row[3])
                if source_id in selected and target_id in selected:
                    candidate_edges[str(row[0])] = edge
                for current_id, other_id, direction in (
                    (source_id, target_id, "out"),
                    (target_id, source_id, "in"),
                ):
                    if current_id not in frontier or other_id not in neighbor_map:
                        continue
                    current_kind = str(selected[current_id]["node_kind"])
                    next_kind = str(neighbor_map[other_id]["node_kind"])
                    permitted = allowed.get((current_kind, direction, edge_kind))
                    if current_id in seed_ids:
                        permitted = permitted or receipt_seed_expansions.get(
                            (current_kind, direction, edge_kind)
                        )
                    if permitted and next_kind in permitted:
                        accepted.add(other_id)
                        candidate_edges[str(row[0])] = edge
            frontier = set()
            for node_id in sorted(accepted):
                if len(selected) >= row_limit:
                    truncated = True
                    break
                selected[node_id] = neighbor_map[node_id]
                frontier.add(node_id)

        remaining = max(0, row_limit - len(selected))
        edges = [
            edge
            for _, edge in sorted(candidate_edges.items())
            if edge["source_node_id"] in selected and edge["target_node_id"] in selected
        ]
        if len(edges) > remaining:
            truncated = True
            edges = edges[:remaining]
    finally:
        connection.close()

    nodes = [selected[node_id] for node_id in sorted(selected)]
    return {
        "schema": QUERY_SCHEMA,
        "artifact_kind": CODE_EVIDENCE_GRAPH_KIND,
        "duckdb_path": str(database_path),
        "query": {
            "task_id": exact_task,
            "symbols": list(exact_symbols),
            "dependency_task_ids": list(exact_dependencies),
            "obligation_ids": list(exact_obligations),
            "receipt_ids": list(exact_receipts),
            "contradiction_ids": list(exact_contradictions),
        },
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "row_count": len(nodes) + len(edges),
        "truncated": truncated,
        "limit": row_limit,
        "max_hops": hop_limit,
    }


# Compatibility spelling for callers where the graph kind is implicit.
query_evidence_neighborhood = query_code_evidence_neighborhood


def artifact_schema(path: Path | str) -> dict[str, Any]:
    """Return typed table/column metadata without returning artifact rows."""

    return query_artifact(
        path,
        sql=(
            "SELECT table_name, column_name, data_type, is_nullable "
            "FROM information_schema.columns "
            "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
        ),
        limit=MAX_QUERY_ROWS,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded queries against supervisor JSON or DuckDB artifacts."
    )
    parser.add_argument("artifact_path", type=Path)
    parser.add_argument("--table", default="artifact_catalog")
    parser.add_argument("--columns", default="*")
    parser.add_argument("--where", default="")
    parser.add_argument("--sql", default="")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--schema", action="store_true", help="Return table and column metadata"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.schema:
        result = artifact_schema(args.artifact_path)
    else:
        columns = tuple(
            item.strip() for item in args.columns.split(",") if item.strip()
        ) or ("*",)
        result = query_artifact(
            args.artifact_path,
            table=args.table,
            columns=columns,
            where=args.where,
            sql=args.sql,
            limit=args.limit,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
