"""DuckDB-backed database-first proof, attestation, and cache authority.

DQP-025 / DatabaseEvidenceStore@1
=================================

:class:`DatabaseEvidenceStore` is the durable authority for validation and
proof receipts, attestations, analysis/proof cache keys, invalidations,
single-flight leases, and use outcomes. Cache rows are never a trust root:
every hit re-evaluates applicability, and stale or poisoned entries fail
closed without promoting assurance.

JSON/Parquet/file freshness never selects authority. Optional exports are
render adapters only. Large external bodies remain digest-bound CAS and are
re-verified on use. Database projections rebuild from admitted evidence.

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, TypeVar

from ..task_sources.control_plane_contracts import (
    REDACTION_MARKER,
    redact_mapping,
)
from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.task_identity import canonical_json_bytes


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_EVIDENCE_STORE_INTERFACE: Final[str] = "DatabaseEvidenceStore@1"

DATABASE_EVIDENCE_STORE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-evidence-store@1"
)
EVIDENCE_KEY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-evidence-key@1"
)
EVIDENCE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-evidence-receipt@1"
)
ATTESTATION_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-attestation-record@1"
)
CACHE_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-cache-entry@1"
)
INVALIDATION_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-invalidation@1"
)
USE_OUTCOME_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-use-outcome@1"
)
FLIGHT_LEASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-flight-lease@1"
)
EVIDENCE_PROJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-evidence-projection@1"
)
EVIDENCE_EXPORT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-evidence-export-receipt@1"
)

DEFAULT_SNAPSHOT_ID: Final[str] = "snapshot:database-evidence-store"
DEFAULT_RECEIPT_TTL_SECONDS: Final[int] = 24 * 60 * 60
DEFAULT_CACHE_TTL_SECONDS: Final[int] = 60 * 60
DEFAULT_NEGATIVE_TTL_SECONDS: Final[int] = 5 * 60
DEFAULT_FLIGHT_LEASE_SECONDS: Final[int] = 5 * 60
DEFAULT_FLIGHT_WAIT_SECONDS: Final[float] = 10 * 60
DEFAULT_FLIGHT_OUTCOME_TTL_SECONDS: Final[int] = 10 * 60
DEFAULT_MAX_BODY_BYTES: Final[int] = 262_144
MAX_RECURSION_DEPTH: Final[int] = 8

_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_AUTHORITY_DATABASE: Final[str] = "database"
_AUTHORITY_EXPORT_ONLY: Final[str] = "export_only"
_AUTHORITY_CACHE: Final[str] = "cache_non_authoritative"

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS evidence_store_metadata (
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
CREATE UNIQUE INDEX IF NOT EXISTS evidence_admitted_seq_uidx
    ON admitted_evidence(sequence);
CREATE INDEX IF NOT EXISTS evidence_admitted_subject_idx
    ON admitted_evidence(subject_id, sequence);

CREATE TABLE IF NOT EXISTS evidence_receipts (
    receipt_id VARCHAR PRIMARY KEY,
    key_id VARCHAR NOT NULL,
    key_json VARCHAR NOT NULL,
    receipt_json VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    assurance_level VARCHAR NOT NULL,
    verdict VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    created_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    blob_digest VARCHAR NOT NULL DEFAULT '',
    redacted BOOLEAN NOT NULL DEFAULT FALSE,
    admitted_sequence BIGINT NOT NULL DEFAULT 0,
    authority VARCHAR NOT NULL DEFAULT 'database'
);
CREATE INDEX IF NOT EXISTS evidence_receipts_key_idx
    ON evidence_receipts(key_id, created_at_ms DESC);

CREATE TABLE IF NOT EXISTS attestations (
    attestation_id VARCHAR PRIMARY KEY,
    receipt_id VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    backend VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    created_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    admitted_sequence BIGINT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS attestations_receipt_idx
    ON attestations(receipt_id, created_at_ms DESC);

CREATE TABLE IF NOT EXISTS cache_entries (
    cache_id VARCHAR PRIMARY KEY,
    key_id VARCHAR NOT NULL,
    domain VARCHAR NOT NULL,
    key_json VARCHAR NOT NULL,
    entry_json VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    assurance_level VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    created_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL,
    is_negative BOOLEAN NOT NULL DEFAULT FALSE,
    admitted_sequence BIGINT NOT NULL DEFAULT 0,
    authority VARCHAR NOT NULL DEFAULT 'cache_non_authoritative'
);
CREATE INDEX IF NOT EXISTS cache_entries_key_idx
    ON cache_entries(key_id, created_at_ms DESC);
CREATE INDEX IF NOT EXISTS cache_entries_domain_idx
    ON cache_entries(domain, key_id);

CREATE TABLE IF NOT EXISTS invalidations (
    invalidation_id VARCHAR PRIMARY KEY,
    subject_id VARCHAR NOT NULL,
    subject_kind VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    invalidated_at_ms BIGINT NOT NULL,
    invalidated_by VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL DEFAULT '{}',
    admitted_sequence BIGINT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS invalidations_subject_idx
    ON invalidations(subject_id, invalidated_at_ms);

CREATE TABLE IF NOT EXISTS use_outcomes (
    outcome_id VARCHAR PRIMARY KEY,
    subject_id VARCHAR NOT NULL,
    subject_kind VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    reason_codes_json VARCHAR NOT NULL DEFAULT '[]',
    observed_at_ms BIGINT NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}',
    admitted_sequence BIGINT NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS use_outcomes_subject_idx
    ON use_outcomes(subject_id, observed_at_ms);

CREATE TABLE IF NOT EXISTS flight_leases (
    key_id VARCHAR PRIMARY KEY,
    owner_id VARCHAR NOT NULL,
    token VARCHAR NOT NULL,
    fencing_token BIGINT NOT NULL,
    acquired_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS flight_outcomes (
    key_id VARCHAR PRIMARY KEY,
    fencing_token BIGINT NOT NULL,
    status VARCHAR NOT NULL,
    outcome_json VARCHAR NOT NULL,
    outcome_digest VARCHAR NOT NULL,
    created_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_projections (
    projection_id VARCHAR PRIMARY KEY,
    projection_kind VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    rebuilt_from_sequence BIGINT NOT NULL,
    rebuilt_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    authority VARCHAR NOT NULL DEFAULT 'database'
);

CREATE TABLE IF NOT EXISTS evidence_exports (
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


class DatabaseEvidenceStoreError(RuntimeError):
    """Base error for database evidence-store failures."""


class DatabaseEvidenceStoreNotOpenError(DatabaseEvidenceStoreError):
    """Operation requires an open evidence store."""


class DatabaseEvidenceStoreConflictError(DatabaseEvidenceStoreError):
    """Identity conflict or authority violation."""


class DatabaseEvidenceStoreIntegrityError(DatabaseEvidenceStoreError):
    """Digest, envelope, or CAS integrity failure (fail closed)."""


class DatabaseEvidenceStoreBoundsError(DatabaseEvidenceStoreError, ValueError):
    """Payload or time-bound exceeded."""


class DuckDBUnavailableError(DatabaseEvidenceStoreError):
    """Optional DuckDB dependency is not installed."""


class SingleFlightError(DatabaseEvidenceStoreError):
    """Base error for single-flight coordination failures."""


class SingleFlightTimeout(SingleFlightError, TimeoutError):
    """No leader outcome was observed before the caller's deadline."""


class SingleFlightExecutionError(SingleFlightError):
    """The single-flight owner failed while producing the shared result."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class AssuranceLevel(str, Enum):
    """Closed assurance vocabulary. Cache hits never promote these levels."""

    NONE = "none"
    MODEL_ONLY = "model_only"
    HEURISTIC = "heuristic"
    SOLVER_CHECKED = "solver_checked"
    KERNEL_CHECKED = "kernel_checked"
    ATTESTED = "attested"

    @classmethod
    def coerce(cls, value: Any) -> "AssuranceLevel":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().casefold().replace("-", "_")
        aliases = {
            "model": cls.MODEL_ONLY,
            "solver": cls.SOLVER_CHECKED,
            "kernel": cls.KERNEL_CHECKED,
            "proof": cls.KERNEL_CHECKED,
        }
        try:
            return aliases.get(text, cls(text))
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise DatabaseEvidenceStoreError(
                f"assurance_level must be one of: {choices}"
            ) from exc

    def rank(self) -> int:
        order = {
            self.NONE: 0,
            self.MODEL_ONLY: 1,
            self.HEURISTIC: 2,
            self.SOLVER_CHECKED: 3,
            self.KERNEL_CHECKED: 4,
            self.ATTESTED: 5,
        }
        return order[self]

    def satisfies(self, required: "AssuranceLevel") -> bool:
        return self.rank() >= required.rank()


class EvidenceVerdict(str, Enum):
    PROVED = "proved"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: Any) -> "EvidenceVerdict":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().casefold().replace("-", "_")
        aliases = {
            "success": cls.PROVED,
            "pass": cls.PROVED,
            "passed": cls.PROVED,
            "fail": cls.REFUTED,
            "failed": cls.REFUTED,
        }
        try:
            return aliases.get(text, cls(text))
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise DatabaseEvidenceStoreError(
                f"verdict must be one of: {choices}"
            ) from exc


class EvidenceOutcome(str, Enum):
    SUCCESSFUL = "successful"
    PARTIAL = "partial"
    FAILED = "failed"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"
    TIMED_OUT = "timed_out"

    @classmethod
    def coerce(cls, value: Any) -> "EvidenceOutcome":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().casefold().replace("-", "_")
        aliases = {
            "success": cls.SUCCESSFUL,
            "ok": cls.SUCCESSFUL,
            "error": cls.FAILED,
            "failure": cls.FAILED,
            "timeout": cls.TIMED_OUT,
        }
        try:
            return aliases.get(text, cls(text))
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise DatabaseEvidenceStoreError(
                f"outcome must be one of: {choices}"
            ) from exc

    @property
    def is_completion_evidence(self) -> bool:
        return self is EvidenceOutcome.SUCCESSFUL

    @property
    def is_negative(self) -> bool:
        return self in {
            EvidenceOutcome.FAILED,
            EvidenceOutcome.NEGATIVE,
            EvidenceOutcome.INCONCLUSIVE,
            EvidenceOutcome.TIMED_OUT,
        }


class LookupStatus(str, Enum):
    HIT = "hit"
    MISS = "miss"
    REJECTED = "rejected"


class RejectionReason(str, Enum):
    CACHE_MISS = "cache_miss"
    MALFORMED = "malformed_receipt"
    POISONED = "poisoned_receipt"
    STALE = "stale_receipt"
    INVALIDATED = "invalidated_receipt"
    INSUFFICIENT_ASSURANCE = "required_assurance_not_satisfied"
    INCONCLUSIVE = "inconclusive_result"
    NEGATIVE_NOT_PROMOTABLE = "negative_not_promotable"
    FRESHNESS_NOT_SATISFIED = "freshness_requirement_not_satisfied"
    BINDING_MISMATCH = "cache_binding_mismatch"
    KEY_MISMATCH = "stale_key"
    BLOB_CORRUPTION = "blob_corruption"
    CACHE_NON_AUTHORITATIVE = "cache_never_promotes_assurance"


class CacheDomain(str, Enum):
    ANALYSIS = "analysis"
    PROOF = "proof"
    ATTESTATION = "attestation"
    GENERIC = "generic"


class EvidenceKind(str, Enum):
    RECEIPT = "receipt"
    ATTESTATION = "attestation"
    CACHE = "cache"
    INVALIDATION = "invalidation"
    USE_OUTCOME = "use_outcome"
    FLIGHT = "flight"


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
        raise DatabaseEvidenceStoreError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseEvidenceStoreError(f"{name} is required")
    return text


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DatabaseEvidenceStoreBoundsError(
            f"{name} must be a non-negative integer"
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatabaseEvidenceStoreBoundsError(
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
    maximum: int = DEFAULT_MAX_BODY_BYTES,
    depth: int = 0,
) -> dict[str, Any]:
    if depth > MAX_RECURSION_DEPTH:
        raise DatabaseEvidenceStoreBoundsError(
            f"body exceeds recursion depth {MAX_RECURSION_DEPTH}"
        )
    raw = dict(body or {})
    cleaned = redact_mapping(raw) if redact else raw
    if not isinstance(cleaned, dict):
        raise DatabaseEvidenceStoreError("body must project to an object")
    encoded = _canonical_json(cleaned).encode("utf-8")
    if len(encoded) > maximum:
        raise DatabaseEvidenceStoreBoundsError(
            f"body exceeds the {maximum}-byte bound"
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
        raise DatabaseEvidenceStoreIntegrityError(
            f"{name} must be sha256:<64-hex>"
        )
    return text


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
class EvidenceKey:
    """Content identity of one proof/analysis request.

    Every dimension that can change a result is part of the key. A stale key
    never matches a stored receipt or cache entry.
    """

    domain: str
    subject_id: str
    repository_tree_id: str
    policy_digest: str
    schema_version: str
    tool_versions: Mapping[str, Any] = field(default_factory=dict)
    configuration_digest: str = ""
    query_digest: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)
    schema: str = EVIDENCE_KEY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _text(self.domain, "domain"))
        object.__setattr__(
            self, "subject_id", _text(self.subject_id, "subject_id")
        )
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(self.repository_tree_id, "repository_tree_id"),
        )
        object.__setattr__(
            self, "policy_digest", _text(self.policy_digest, "policy_digest")
        )
        object.__setattr__(
            self, "schema_version", _text(self.schema_version, "schema_version")
        )
        tools = dict(self.tool_versions or {})
        if not tools:
            raise DatabaseEvidenceStoreError(
                "tool_versions must not be empty; unknown versions are not allowed"
            )
        object.__setattr__(self, "tool_versions", MappingProxyType(tools))
        object.__setattr__(
            self,
            "configuration_digest",
            _text(self.configuration_digest, "configuration_digest", required=False),
        )
        object.__setattr__(
            self,
            "query_digest",
            _text(self.query_digest, "query_digest", required=False),
        )
        object.__setattr__(
            self, "extra", MappingProxyType(dict(self.extra or {}))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "domain": self.domain,
            "subject_id": self.subject_id,
            "repository_tree_id": self.repository_tree_id,
            "policy_digest": self.policy_digest,
            "schema_version": self.schema_version,
            "tool_versions": dict(self.tool_versions),
            "configuration_digest": self.configuration_digest,
            "query_digest": self.query_digest,
            "extra": dict(self.extra),
        }

    @property
    def key_id(self) -> str:
        return "evidence-key:" + _sha256_hex(
            _canonical_json(self.to_dict()).encode("utf-8")
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceKey":
        return cls(
            domain=str(value.get("domain") or ""),
            subject_id=str(value.get("subject_id") or ""),
            repository_tree_id=str(value.get("repository_tree_id") or ""),
            policy_digest=str(value.get("policy_digest") or ""),
            schema_version=str(value.get("schema_version") or ""),
            tool_versions=dict(value.get("tool_versions") or {}),
            configuration_digest=str(value.get("configuration_digest") or ""),
            query_digest=str(value.get("query_digest") or ""),
            extra=dict(value.get("extra") or {}),
        )


@dataclass(frozen=True)
class EvidenceReceipt:
    """Authoritative validation/proof receipt committed in the database."""

    receipt_id: str
    key: EvidenceKey
    content_digest: str
    assurance_level: str
    verdict: str
    outcome: str
    created_at_ms: int
    expires_at_ms: int
    body: Mapping[str, Any] = field(default_factory=dict)
    blob_digest: str = ""
    redacted: bool = False
    admitted_sequence: int = 0
    authority: str = _AUTHORITY_DATABASE
    schema: str = EVIDENCE_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": DATABASE_EVIDENCE_STORE_INTERFACE,
            "receipt_id": self.receipt_id,
            "key": self.key.to_dict(),
            "key_id": self.key.key_id,
            "content_digest": self.content_digest,
            "assurance_level": self.assurance_level,
            "verdict": self.verdict,
            "outcome": self.outcome,
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "body": dict(self.body),
            "blob_digest": self.blob_digest,
            "redacted": self.redacted,
            "admitted_sequence": self.admitted_sequence,
            "authority": self.authority,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceReceipt":
        key_value = value.get("key")
        if not isinstance(key_value, Mapping):
            raise DatabaseEvidenceStoreIntegrityError("receipt missing key")
        return cls(
            receipt_id=str(value.get("receipt_id") or ""),
            key=EvidenceKey.from_dict(key_value),
            content_digest=str(value.get("content_digest") or ""),
            assurance_level=str(value.get("assurance_level") or ""),
            verdict=str(value.get("verdict") or ""),
            outcome=str(value.get("outcome") or ""),
            created_at_ms=int(value.get("created_at_ms") or 0),
            expires_at_ms=int(value.get("expires_at_ms") or 0),
            body=dict(value.get("body") or {}),
            blob_digest=str(value.get("blob_digest") or ""),
            redacted=bool(value.get("redacted")),
            admitted_sequence=int(value.get("admitted_sequence") or 0),
            authority=str(value.get("authority") or _AUTHORITY_DATABASE),
        )


@dataclass(frozen=True)
class AttestationRecord:
    """Attestation bound to one admitted receipt."""

    attestation_id: str
    receipt_id: str
    content_digest: str
    backend: str
    status: str
    created_at_ms: int
    expires_at_ms: int
    body: Mapping[str, Any] = field(default_factory=dict)
    admitted_sequence: int = 0
    schema: str = ATTESTATION_RECORD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "attestation_id": self.attestation_id,
            "receipt_id": self.receipt_id,
            "content_digest": self.content_digest,
            "backend": self.backend,
            "status": self.status,
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "body": dict(self.body),
            "admitted_sequence": self.admitted_sequence,
        }


@dataclass(frozen=True)
class CacheEntry:
    """Non-authoritative cache projection. Never promotes assurance."""

    cache_id: str
    key: EvidenceKey
    domain: str
    content_digest: str
    assurance_level: str
    outcome: str
    created_at_ms: int
    expires_at_ms: int
    body: Mapping[str, Any] = field(default_factory=dict)
    is_negative: bool = False
    admitted_sequence: int = 0
    authority: str = _AUTHORITY_CACHE
    schema: str = CACHE_ENTRY_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cache_id": self.cache_id,
            "key": self.key.to_dict(),
            "key_id": self.key.key_id,
            "domain": self.domain,
            "content_digest": self.content_digest,
            "assurance_level": self.assurance_level,
            "outcome": self.outcome,
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "body": dict(self.body),
            "is_negative": self.is_negative,
            "admitted_sequence": self.admitted_sequence,
            "authority": self.authority,
            "promotes_assurance": False,
        }


@dataclass(frozen=True)
class LookupResult:
    """Result of a fail-closed receipt or cache lookup."""

    status: LookupStatus
    key: EvidenceKey
    receipt: EvidenceReceipt | None = None
    cache_entry: CacheEntry | None = None
    reason_codes: tuple[str, ...] = ()

    @property
    def is_hit(self) -> bool:
        return self.status is LookupStatus.HIT

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "key_id": self.key.key_id,
            "reason_codes": list(self.reason_codes),
            "receipt": self.receipt.to_dict() if self.receipt else None,
            "cache_entry": (
                self.cache_entry.to_dict() if self.cache_entry else None
            ),
        }


@dataclass(frozen=True)
class InvalidationRecord:
    invalidation_id: str
    subject_id: str
    subject_kind: str
    reason: str
    invalidated_at_ms: int
    invalidated_by: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)
    admitted_sequence: int = 0
    schema: str = INVALIDATION_RECORD_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "invalidation_id": self.invalidation_id,
            "subject_id": self.subject_id,
            "subject_kind": self.subject_kind,
            "reason": self.reason,
            "invalidated_at_ms": self.invalidated_at_ms,
            "invalidated_by": self.invalidated_by,
            "body": dict(self.body),
            "admitted_sequence": self.admitted_sequence,
        }


@dataclass(frozen=True)
class UseOutcome:
    outcome_id: str
    subject_id: str
    subject_kind: str
    status: str
    reason_codes: tuple[str, ...]
    observed_at_ms: int
    body: Mapping[str, Any] = field(default_factory=dict)
    admitted_sequence: int = 0
    schema: str = USE_OUTCOME_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "outcome_id": self.outcome_id,
            "subject_id": self.subject_id,
            "subject_kind": self.subject_kind,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "observed_at_ms": self.observed_at_ms,
            "body": dict(self.body),
            "admitted_sequence": self.admitted_sequence,
        }


@dataclass(frozen=True)
class SingleFlightResult:
    value: Any
    owner: bool
    fencing_token: int

    @property
    def shared(self) -> bool:
        return not self.owner


@dataclass(frozen=True)
class EvidenceProjection:
    projection_id: str
    projection_kind: str
    content_digest: str
    rebuilt_from_sequence: int
    rebuilt_at: str
    body: Mapping[str, Any]
    authority: str = _AUTHORITY_DATABASE
    schema: str = EVIDENCE_PROJECTION_SCHEMA

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
class EvidenceExportReceipt:
    export_id: str
    export_path: str
    export_format: str
    content_digest: str
    subject_count: int
    recorded_at: str
    authority: str = _AUTHORITY_EXPORT_ONLY
    schema: str = EVIDENCE_EXPORT_RECEIPT_SCHEMA

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


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DatabaseEvidenceStore:
    """DuckDB authority for receipts, attestations, caches, and single-flight."""

    INTERFACE: Final[str] = DATABASE_EVIDENCE_STORE_INTERFACE

    def __init__(
        self,
        database_path: Path | str,
        *,
        cas_root: Path | str | None = None,
        snapshot_id: str = DEFAULT_SNAPSHOT_ID,
        default_receipt_ttl_seconds: int = DEFAULT_RECEIPT_TTL_SECONDS,
        default_cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        default_negative_ttl_seconds: int = DEFAULT_NEGATIVE_TTL_SECONDS,
        auto_redact: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseEvidenceStore; install the "
                "optional duckdb dependency"
            )
        self._path = Path(database_path)
        self._cas_root = Path(cas_root) if cas_root is not None else (
            self._path.parent / f"{self._path.stem}.cas"
        )
        self._snapshot_id = _text(snapshot_id, "snapshot_id")
        self._default_receipt_ttl = _positive_int(
            default_receipt_ttl_seconds, "default_receipt_ttl_seconds"
        )
        self._default_cache_ttl = _positive_int(
            default_cache_ttl_seconds, "default_cache_ttl_seconds"
        )
        self._default_negative_ttl = _positive_int(
            default_negative_ttl_seconds, "default_negative_ttl_seconds"
        )
        self._auto_redact = bool(auto_redact)
        self._clock = clock
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
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "DatabaseEvidenceStore":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._cas_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            connection = open_duckdb_connection(self._path)
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
            for key, value in (
                ("interface", DATABASE_EVIDENCE_STORE_INTERFACE),
                ("schema", DATABASE_EVIDENCE_STORE_SCHEMA),
                ("snapshot_id", self._snapshot_id),
                ("authority", _AUTHORITY_DATABASE),
                ("cache_promotes_assurance", "false"),
            ):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO evidence_store_metadata(key, value)
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

    def __enter__(self) -> "DatabaseEvidenceStore":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise DatabaseEvidenceStoreNotOpenError(
                "DatabaseEvidenceStore is not open"
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

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _next_sequence(self) -> int:
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
        provenance: Mapping[str, Any] | None = None,
        sequence: int | None = None,
    ) -> int:
        seq = sequence if sequence is not None else self._next_sequence()
        kind = (
            evidence_kind.value
            if isinstance(evidence_kind, EvidenceKind)
            else _text(evidence_kind, "evidence_kind")
        )
        stamp = _utc_iso()
        prov = dict(provenance or {})
        material = {
            "evidence_kind": kind,
            "subject_id": subject_id,
            "content_digest": content_digest,
            "sequence": seq,
            "body": dict(body),
            "provenance": prov,
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
                _canonical_json(prov),
            ],
        )
        return seq

    def _is_invalidated(self, connection: Any, subject_id: str) -> bool:
        row = connection.execute(
            "SELECT 1 AS ok FROM invalidations WHERE subject_id=? LIMIT 1",
            [subject_id],
        ).fetchone()
        return row is not None

    def _record_use_outcome(
        self,
        connection: Any,
        *,
        subject_id: str,
        subject_kind: str,
        status: str,
        reason_codes: Sequence[str],
        body: Mapping[str, Any] | None = None,
    ) -> UseOutcome:
        now = self._now_ms()
        reasons = tuple(sorted({str(item) for item in reason_codes if item}))
        payload = _bounded_mapping(body, redact=False)
        material = {
            "subject_id": subject_id,
            "subject_kind": subject_kind,
            "status": status,
            "reason_codes": list(reasons),
            "observed_at_ms": now,
            "body": payload,
        }
        outcome_id = "use:" + _sha256_hex(
            _canonical_json(material).encode("utf-8")
        )
        seq = self._next_sequence()
        connection.execute(
            """
            INSERT INTO use_outcomes(
                outcome_id, subject_id, subject_kind, status,
                reason_codes_json, observed_at_ms, body_json, admitted_sequence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                outcome_id,
                subject_id,
                subject_kind,
                status,
                _canonical_json(list(reasons)),
                now,
                _canonical_json(payload),
                seq,
            ],
        )
        self._admit_evidence(
            connection,
            evidence_kind=EvidenceKind.USE_OUTCOME,
            subject_id=outcome_id,
            content_digest=_sha256_hex(
                _canonical_json(material).encode("utf-8")
            ),
            body=material,
            sequence=seq,
        )
        return UseOutcome(
            outcome_id=outcome_id,
            subject_id=subject_id,
            subject_kind=subject_kind,
            status=status,
            reason_codes=reasons,
            observed_at_ms=now,
            body=MappingProxyType(payload),
            admitted_sequence=seq,
        )

    # -- blob helpers --------------------------------------------------------

    def _blob_path(self, digest: str) -> Path:
        hex_digest = digest.removeprefix("sha256:")
        return self._cas_root / "blobs" / "sha256" / hex_digest[:2] / f"{hex_digest}.blob"

    def put_blob(
        self,
        data: bytes | bytearray | str | Mapping[str, Any] | Sequence[Any] | Any,
        *,
        media_type: str = "application/octet-stream",
    ) -> str:
        """Store a digest-bound external body; return its digest."""

        if isinstance(data, (bytes, bytearray)):
            payload = bytes(data)
        elif isinstance(data, str):
            payload = data.encode("utf-8")
        elif isinstance(data, Mapping):
            payload = _canonical_json(dict(data)).encode("utf-8")
        elif isinstance(data, Sequence):
            payload = _canonical_json(list(data)).encode("utf-8")
        else:
            payload = _canonical_json(data).encode("utf-8")
        if len(payload) > DEFAULT_MAX_BODY_BYTES * 64:
            raise DatabaseEvidenceStoreBoundsError("blob too large")
        digest = _sha256_hex(payload)
        path = self._blob_path(digest)
        if path.is_file():
            observed = _sha256_hex(path.read_bytes())
            if observed != digest:
                raise DatabaseEvidenceStoreIntegrityError(
                    "CAS body is poisoned relative to its digest"
                )
            return digest
        _atomic_write_bytes(path, payload)
        # media_type retained for future metadata tables; digest is authority.
        _ = media_type
        return digest

    def verify_blob(self, blob_digest: str) -> bytes:
        digest = _require_digest(blob_digest, "blob_digest")
        path = self._blob_path(digest)
        if not path.is_file():
            raise DatabaseEvidenceStoreIntegrityError(
                f"CAS body missing for {digest}"
            )
        payload = path.read_bytes()
        observed = _sha256_hex(payload)
        if observed != digest:
            raise DatabaseEvidenceStoreIntegrityError(
                f"CAS body poisoned for {digest}"
            )
        return payload

    # -- receipts ------------------------------------------------------------

    def put_receipt(
        self,
        key: EvidenceKey | Mapping[str, Any],
        *,
        assurance_level: AssuranceLevel | str,
        verdict: EvidenceVerdict | str,
        outcome: EvidenceOutcome | str = EvidenceOutcome.SUCCESSFUL,
        body: Mapping[str, Any] | None = None,
        blob: bytes | bytearray | str | Mapping[str, Any] | None = None,
        blob_digest: str = "",
        ttl_seconds: int | None = None,
        redact: bool | None = None,
    ) -> EvidenceReceipt:
        """Admit one validation/proof receipt as database authority."""

        cache_key = (
            key if isinstance(key, EvidenceKey) else EvidenceKey.from_dict(key)
        )
        assurance = AssuranceLevel.coerce(assurance_level)
        selected_verdict = EvidenceVerdict.coerce(verdict)
        selected_outcome = EvidenceOutcome.coerce(outcome)
        do_redact = self._auto_redact if redact is None else bool(redact)
        payload = _bounded_mapping(body, redact=do_redact)
        selected_blob = ""
        if blob is not None:
            selected_blob = self.put_blob(blob)
        elif blob_digest:
            selected_blob = _require_digest(blob_digest, "blob_digest")
            self.verify_blob(selected_blob)

        ttl = (
            self._default_receipt_ttl
            if ttl_seconds is None
            else _positive_int(ttl_seconds, "ttl_seconds")
        )
        now = self._now_ms()
        content_material = {
            "key": cache_key.to_dict(),
            "assurance_level": assurance.value,
            "verdict": selected_verdict.value,
            "outcome": selected_outcome.value,
            "body": payload,
            "blob_digest": selected_blob,
        }
        content_digest = _sha256_hex(
            _canonical_json(content_material).encode("utf-8")
        )
        receipt_id = f"receipt:{content_digest}"
        with self._lock:
            connection = self._require()
            existing = connection.execute(
                "SELECT * FROM evidence_receipts WHERE receipt_id=?",
                [receipt_id],
            ).fetchone()
            if existing is not None:
                return self._decode_receipt(_row_mapping(existing))

            connection.execute("BEGIN TRANSACTION")
            try:
                seq = self._next_sequence()
                receipt = EvidenceReceipt(
                    receipt_id=receipt_id,
                    key=cache_key,
                    content_digest=content_digest,
                    assurance_level=assurance.value,
                    verdict=selected_verdict.value,
                    outcome=selected_outcome.value,
                    created_at_ms=now,
                    expires_at_ms=now + ttl * 1000,
                    body=MappingProxyType(payload),
                    blob_digest=selected_blob,
                    redacted=do_redact,
                    admitted_sequence=seq,
                )
                connection.execute(
                    """
                    INSERT INTO evidence_receipts(
                        receipt_id, key_id, key_json, receipt_json,
                        content_digest, assurance_level, verdict, outcome,
                        created_at_ms, expires_at_ms, blob_digest, redacted,
                        admitted_sequence, authority
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        receipt.receipt_id,
                        cache_key.key_id,
                        _canonical_json(cache_key.to_dict()),
                        _canonical_json(receipt.to_dict()),
                        receipt.content_digest,
                        receipt.assurance_level,
                        receipt.verdict,
                        receipt.outcome,
                        receipt.created_at_ms,
                        receipt.expires_at_ms,
                        receipt.blob_digest,
                        receipt.redacted,
                        seq,
                        _AUTHORITY_DATABASE,
                    ],
                )
                self._admit_evidence(
                    connection,
                    evidence_kind=EvidenceKind.RECEIPT,
                    subject_id=receipt.receipt_id,
                    content_digest=receipt.content_digest,
                    body=receipt.to_dict(),
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
            return receipt

    def _decode_receipt(self, mapping: Mapping[str, Any]) -> EvidenceReceipt:
        try:
            value = json.loads(str(mapping.get("receipt_json") or "{}"))
            key_value = json.loads(str(mapping.get("key_json") or "{}"))
        except json.JSONDecodeError as exc:
            raise DatabaseEvidenceStoreIntegrityError(
                "poisoned receipt envelope"
            ) from exc
        if not isinstance(value, dict) or not isinstance(key_value, dict):
            raise DatabaseEvidenceStoreIntegrityError(
                "poisoned receipt envelope"
            )
        receipt = EvidenceReceipt.from_dict(value)
        if (
            receipt.receipt_id != str(mapping.get("receipt_id") or "")
            or receipt.key.key_id != str(mapping.get("key_id") or "")
            or receipt.content_digest != str(mapping.get("content_digest") or "")
            or receipt.key.to_dict() != key_value
        ):
            raise DatabaseEvidenceStoreIntegrityError(
                "poisoned durable receipt envelope"
            )
        return receipt

    def lookup_receipt(
        self,
        key: EvidenceKey | Mapping[str, Any],
        *,
        required_assurance: AssuranceLevel | str = AssuranceLevel.SOLVER_CHECKED,
        max_age_seconds: int | None = None,
        verify_blob: bool = True,
        record_use: bool = True,
    ) -> LookupResult:
        """Fail-closed receipt lookup with re-derived applicability."""

        cache_key = (
            key if isinstance(key, EvidenceKey) else EvidenceKey.from_dict(key)
        )
        required = AssuranceLevel.coerce(required_assurance)
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT * FROM evidence_receipts
                WHERE key_id=?
                ORDER BY created_at_ms DESC, receipt_id DESC
                """,
                [cache_key.key_id],
            ).fetchall()
            if not rows:
                if record_use:
                    self._record_use_outcome(
                        connection,
                        subject_id=cache_key.key_id,
                        subject_kind="receipt_key",
                        status=LookupStatus.MISS.value,
                        reason_codes=(RejectionReason.CACHE_MISS.value,),
                    )
                    self._commit_if_idle(connection)
                return LookupResult(
                    LookupStatus.MISS,
                    cache_key,
                    reason_codes=(RejectionReason.CACHE_MISS.value,),
                )

            accumulated: set[str] = set()
            now = self._now_ms()
            for row in rows:
                mapping = _row_mapping(row)
                try:
                    receipt = self._decode_receipt(mapping)
                except (
                    DatabaseEvidenceStoreError,
                    TypeError,
                    ValueError,
                    KeyError,
                    json.JSONDecodeError,
                ):
                    accumulated.add(RejectionReason.POISONED.value)
                    continue

                reasons: set[str] = set()
                if receipt.key.key_id != cache_key.key_id:
                    reasons.add(RejectionReason.KEY_MISMATCH.value)
                if receipt.key.to_dict() != cache_key.to_dict():
                    reasons.add(RejectionReason.BINDING_MISMATCH.value)
                if self._is_invalidated(connection, receipt.receipt_id):
                    reasons.add(RejectionReason.INVALIDATED.value)
                if now >= receipt.expires_at_ms:
                    reasons.add(RejectionReason.STALE.value)
                    reasons.add(RejectionReason.FRESHNESS_NOT_SATISFIED.value)
                if max_age_seconds is not None and (
                    now - receipt.created_at_ms > max_age_seconds * 1000
                ):
                    reasons.add(RejectionReason.STALE.value)
                    reasons.add(RejectionReason.FRESHNESS_NOT_SATISFIED.value)

                try:
                    stored_assurance = AssuranceLevel.coerce(
                        receipt.assurance_level
                    )
                    stored_outcome = EvidenceOutcome.coerce(receipt.outcome)
                    stored_verdict = EvidenceVerdict.coerce(receipt.verdict)
                except DatabaseEvidenceStoreError:
                    accumulated.add(RejectionReason.POISONED.value)
                    continue
                if not stored_assurance.satisfies(required):
                    reasons.add(RejectionReason.INSUFFICIENT_ASSURANCE.value)

                if stored_outcome.is_negative:
                    reasons.add(RejectionReason.NEGATIVE_NOT_PROMOTABLE.value)
                if stored_verdict is not EvidenceVerdict.PROVED:
                    reasons.add(RejectionReason.INCONCLUSIVE.value)

                if verify_blob and receipt.blob_digest:
                    try:
                        self.verify_blob(receipt.blob_digest)
                    except DatabaseEvidenceStoreIntegrityError:
                        reasons.add(RejectionReason.BLOB_CORRUPTION.value)
                        reasons.add(RejectionReason.POISONED.value)

                if not reasons:
                    if record_use:
                        self._record_use_outcome(
                            connection,
                            subject_id=receipt.receipt_id,
                            subject_kind="receipt",
                            status=LookupStatus.HIT.value,
                            reason_codes=(),
                            body={"key_id": cache_key.key_id},
                        )
                        self._commit_if_idle(connection)
                    return LookupResult(
                        LookupStatus.HIT, cache_key, receipt=receipt
                    )
                accumulated.update(reasons)

            if record_use:
                self._record_use_outcome(
                    connection,
                    subject_id=cache_key.key_id,
                    subject_kind="receipt_key",
                    status=LookupStatus.REJECTED.value,
                    reason_codes=tuple(sorted(accumulated)),
                )
                self._commit_if_idle(connection)
            return LookupResult(
                LookupStatus.REJECTED,
                cache_key,
                reason_codes=tuple(sorted(accumulated)),
            )

    def get_receipt(self, receipt_id: str) -> EvidenceReceipt | None:
        selected = _text(receipt_id, "receipt_id")
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM evidence_receipts WHERE receipt_id=?",
                [selected],
            ).fetchone()
            if row is None:
                return None
            return self._decode_receipt(_row_mapping(row))

    # -- attestations --------------------------------------------------------

    def put_attestation(
        self,
        receipt_id: str,
        *,
        backend: str,
        status: str = "verified",
        body: Mapping[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> AttestationRecord:
        selected_receipt = _text(receipt_id, "receipt_id")
        selected_backend = _text(backend, "backend")
        selected_status = _text(status, "status")
        payload = _bounded_mapping(body, redact=self._auto_redact)
        ttl = (
            self._default_receipt_ttl
            if ttl_seconds is None
            else _positive_int(ttl_seconds, "ttl_seconds")
        )
        now = self._now_ms()

        with self._lock:
            connection = self._require()
            receipt_row = connection.execute(
                "SELECT receipt_id FROM evidence_receipts WHERE receipt_id=?",
                [selected_receipt],
            ).fetchone()
            if receipt_row is None:
                raise DatabaseEvidenceStoreConflictError(
                    f"receipt is not admitted: {selected_receipt}"
                )
            material = {
                "receipt_id": selected_receipt,
                "backend": selected_backend,
                "status": selected_status,
                "body": payload,
            }
            content_digest = _sha256_hex(
                _canonical_json(material).encode("utf-8")
            )
            attestation_id = f"attestation:{content_digest}"
            existing = connection.execute(
                "SELECT * FROM attestations WHERE attestation_id=?",
                [attestation_id],
            ).fetchone()
            if existing is not None:
                mapping = _row_mapping(existing)
                try:
                    body_value = json.loads(str(mapping.get("body_json") or "{}"))
                except json.JSONDecodeError as exc:
                    raise DatabaseEvidenceStoreIntegrityError(
                        "poisoned attestation envelope"
                    ) from exc
                return AttestationRecord(
                    attestation_id=str(mapping["attestation_id"]),
                    receipt_id=str(mapping["receipt_id"]),
                    content_digest=str(mapping["content_digest"]),
                    backend=str(mapping["backend"]),
                    status=str(mapping["status"]),
                    created_at_ms=int(mapping["created_at_ms"]),
                    expires_at_ms=int(mapping["expires_at_ms"]),
                    body=MappingProxyType(
                        body_value if isinstance(body_value, dict) else {}
                    ),
                    admitted_sequence=int(mapping.get("admitted_sequence") or 0),
                )

            connection.execute("BEGIN TRANSACTION")
            try:
                seq = self._next_sequence()
                connection.execute(
                    """
                    INSERT INTO attestations(
                        attestation_id, receipt_id, content_digest, backend,
                        status, body_json, created_at_ms, expires_at_ms,
                        admitted_sequence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        attestation_id,
                        selected_receipt,
                        content_digest,
                        selected_backend,
                        selected_status,
                        _canonical_json(payload),
                        now,
                        now + ttl * 1000,
                        seq,
                    ],
                )
                self._admit_evidence(
                    connection,
                    evidence_kind=EvidenceKind.ATTESTATION,
                    subject_id=attestation_id,
                    content_digest=content_digest,
                    body=material,
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
            return AttestationRecord(
                attestation_id=attestation_id,
                receipt_id=selected_receipt,
                content_digest=content_digest,
                backend=selected_backend,
                status=selected_status,
                created_at_ms=now,
                expires_at_ms=now + ttl * 1000,
                body=MappingProxyType(payload),
                admitted_sequence=seq,
            )

    def list_attestations(
        self, receipt_id: str, *, limit: int = 64
    ) -> list[AttestationRecord]:
        selected = _text(receipt_id, "receipt_id")
        limit = min(_positive_int(limit, "limit"), 1_024)
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT * FROM attestations
                WHERE receipt_id=?
                ORDER BY created_at_ms DESC
                LIMIT ?
                """,
                [selected, limit],
            ).fetchall()
            results: list[AttestationRecord] = []
            for row in rows:
                mapping = _row_mapping(row)
                try:
                    body_value = json.loads(str(mapping.get("body_json") or "{}"))
                except json.JSONDecodeError as exc:
                    raise DatabaseEvidenceStoreIntegrityError(
                        "poisoned attestation envelope"
                    ) from exc
                results.append(
                    AttestationRecord(
                        attestation_id=str(mapping["attestation_id"]),
                        receipt_id=str(mapping["receipt_id"]),
                        content_digest=str(mapping["content_digest"]),
                        backend=str(mapping["backend"]),
                        status=str(mapping["status"]),
                        created_at_ms=int(mapping["created_at_ms"]),
                        expires_at_ms=int(mapping["expires_at_ms"]),
                        body=MappingProxyType(
                            body_value if isinstance(body_value, dict) else {}
                        ),
                        admitted_sequence=int(
                            mapping.get("admitted_sequence") or 0
                        ),
                    )
                )
            return results

    # -- cache (never promotes assurance) ------------------------------------

    def put_cache_entry(
        self,
        key: EvidenceKey | Mapping[str, Any],
        *,
        outcome: EvidenceOutcome | str,
        assurance_level: AssuranceLevel | str = AssuranceLevel.NONE,
        body: Mapping[str, Any] | None = None,
        ttl_seconds: int | None = None,
        domain: CacheDomain | str | None = None,
    ) -> CacheEntry:
        """Store a non-authoritative cache entry.

        Cache rows never promote assurance: stored ``assurance_level`` is
        retained for audit only and is re-derived (never trusted) on lookup.
        Negative and inconclusive entries receive short TTLs.
        """

        cache_key = (
            key if isinstance(key, EvidenceKey) else EvidenceKey.from_dict(key)
        )
        selected_outcome = EvidenceOutcome.coerce(outcome)
        # Cache never promotes; clamp claimed assurance for storage audit only.
        claimed = AssuranceLevel.coerce(assurance_level)
        stored_assurance = AssuranceLevel.NONE
        if claimed is not AssuranceLevel.NONE:
            # Persist the claim for audit but mark authority as non-promotable.
            stored_assurance = claimed
        payload = _bounded_mapping(body, redact=self._auto_redact)
        selected_domain = (
            domain.value
            if isinstance(domain, CacheDomain)
            else _text(domain or cache_key.domain, "domain")
        )
        if selected_outcome.is_negative:
            ttl = (
                self._default_negative_ttl
                if ttl_seconds is None
                else min(
                    _positive_int(ttl_seconds, "ttl_seconds"),
                    self._default_negative_ttl,
                )
            )
        else:
            ttl = (
                self._default_cache_ttl
                if ttl_seconds is None
                else _positive_int(ttl_seconds, "ttl_seconds")
            )
        now = self._now_ms()
        content_material = {
            "key": cache_key.to_dict(),
            "domain": selected_domain,
            "outcome": selected_outcome.value,
            "assurance_level": stored_assurance.value,
            "body": payload,
            "authority": _AUTHORITY_CACHE,
            "promotes_assurance": False,
        }
        content_digest = _sha256_hex(
            _canonical_json(content_material).encode("utf-8")
        )
        cache_id = f"cache:{content_digest}"
        with self._lock:
            connection = self._require()
            existing = connection.execute(
                "SELECT * FROM cache_entries WHERE cache_id=?",
                [cache_id],
            ).fetchone()
            if existing is not None:
                return self._decode_cache_entry(_row_mapping(existing))

            connection.execute("BEGIN TRANSACTION")
            try:
                seq = self._next_sequence()
                entry = CacheEntry(
                    cache_id=cache_id,
                    key=cache_key,
                    domain=selected_domain,
                    content_digest=content_digest,
                    assurance_level=stored_assurance.value,
                    outcome=selected_outcome.value,
                    created_at_ms=now,
                    expires_at_ms=now + ttl * 1000,
                    body=MappingProxyType(payload),
                    is_negative=selected_outcome.is_negative,
                    admitted_sequence=seq,
                )
                connection.execute(
                    """
                    INSERT INTO cache_entries(
                        cache_id, key_id, domain, key_json, entry_json,
                        content_digest, assurance_level, outcome,
                        created_at_ms, expires_at_ms, is_negative,
                        admitted_sequence, authority
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        entry.cache_id,
                        cache_key.key_id,
                        selected_domain,
                        _canonical_json(cache_key.to_dict()),
                        _canonical_json(entry.to_dict()),
                        entry.content_digest,
                        entry.assurance_level,
                        entry.outcome,
                        entry.created_at_ms,
                        entry.expires_at_ms,
                        entry.is_negative,
                        seq,
                        _AUTHORITY_CACHE,
                    ],
                )
                self._admit_evidence(
                    connection,
                    evidence_kind=EvidenceKind.CACHE,
                    subject_id=entry.cache_id,
                    content_digest=entry.content_digest,
                    body=entry.to_dict(),
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
            return entry

    def _decode_cache_entry(self, mapping: Mapping[str, Any]) -> CacheEntry:
        try:
            value = json.loads(str(mapping.get("entry_json") or "{}"))
            key_value = json.loads(str(mapping.get("key_json") or "{}"))
        except json.JSONDecodeError as exc:
            raise DatabaseEvidenceStoreIntegrityError(
                "poisoned cache entry envelope"
            ) from exc
        if not isinstance(value, dict) or not isinstance(key_value, dict):
            raise DatabaseEvidenceStoreIntegrityError(
                "poisoned cache entry envelope"
            )
        key = EvidenceKey.from_dict(key_value)
        if (
            str(mapping.get("cache_id") or "") != str(value.get("cache_id") or "")
            or str(mapping.get("key_id") or "") != key.key_id
            or str(mapping.get("content_digest") or "")
            != str(value.get("content_digest") or "")
        ):
            raise DatabaseEvidenceStoreIntegrityError(
                "poisoned durable cache envelope"
            )
        return CacheEntry(
            cache_id=str(value.get("cache_id") or ""),
            key=key,
            domain=str(value.get("domain") or mapping.get("domain") or ""),
            content_digest=str(value.get("content_digest") or ""),
            assurance_level=str(value.get("assurance_level") or ""),
            outcome=str(value.get("outcome") or ""),
            created_at_ms=int(value.get("created_at_ms") or 0),
            expires_at_ms=int(value.get("expires_at_ms") or 0),
            body=dict(value.get("body") or {}),
            is_negative=bool(value.get("is_negative")),
            admitted_sequence=int(mapping.get("admitted_sequence") or 0),
            authority=_AUTHORITY_CACHE,
        )

    def lookup_cache(
        self,
        key: EvidenceKey | Mapping[str, Any],
        *,
        required_assurance: AssuranceLevel | str = AssuranceLevel.NONE,
        max_age_seconds: int | None = None,
        allow_negative: bool = False,
        record_use: bool = True,
    ) -> LookupResult:
        """Fail-closed cache lookup that never promotes assurance.

        Even a successful structural hit returns ``REJECTED`` with
        ``cache_never_promotes_assurance`` when the caller requests an
        assurance level above ``none``. Negative entries never count as
        completion evidence.
        """

        cache_key = (
            key if isinstance(key, EvidenceKey) else EvidenceKey.from_dict(key)
        )
        required = AssuranceLevel.coerce(required_assurance)
        with self._lock:
            connection = self._require()
            rows = connection.execute(
                """
                SELECT * FROM cache_entries
                WHERE key_id=?
                ORDER BY created_at_ms DESC, cache_id DESC
                """,
                [cache_key.key_id],
            ).fetchall()
            if not rows:
                if record_use:
                    self._record_use_outcome(
                        connection,
                        subject_id=cache_key.key_id,
                        subject_kind="cache_key",
                        status=LookupStatus.MISS.value,
                        reason_codes=(RejectionReason.CACHE_MISS.value,),
                    )
                    self._commit_if_idle(connection)
                return LookupResult(
                    LookupStatus.MISS,
                    cache_key,
                    reason_codes=(RejectionReason.CACHE_MISS.value,),
                )

            accumulated: set[str] = set()
            now = self._now_ms()
            for row in rows:
                mapping = _row_mapping(row)
                try:
                    entry = self._decode_cache_entry(mapping)
                except (
                    DatabaseEvidenceStoreError,
                    TypeError,
                    ValueError,
                    KeyError,
                    json.JSONDecodeError,
                ):
                    accumulated.add(RejectionReason.POISONED.value)
                    continue

                reasons: set[str] = set()
                if entry.key.to_dict() != cache_key.to_dict():
                    reasons.add(RejectionReason.KEY_MISMATCH.value)
                    reasons.add(RejectionReason.BINDING_MISMATCH.value)
                if self._is_invalidated(connection, entry.cache_id):
                    reasons.add(RejectionReason.INVALIDATED.value)
                if now >= entry.expires_at_ms:
                    reasons.add(RejectionReason.STALE.value)
                    reasons.add(RejectionReason.FRESHNESS_NOT_SATISFIED.value)
                if max_age_seconds is not None and (
                    now - entry.created_at_ms > max_age_seconds * 1000
                ):
                    reasons.add(RejectionReason.STALE.value)
                if entry.is_negative and not allow_negative:
                    reasons.add(RejectionReason.NEGATIVE_NOT_PROMOTABLE.value)
                # Cache never promotes assurance above none.
                if required is not AssuranceLevel.NONE:
                    reasons.add(RejectionReason.CACHE_NON_AUTHORITATIVE.value)
                    reasons.add(RejectionReason.INSUFFICIENT_ASSURANCE.value)
                # Even for required=NONE, never claim the stored level as proof.
                if entry.authority != _AUTHORITY_CACHE:
                    reasons.add(RejectionReason.POISONED.value)

                if not reasons:
                    if record_use:
                        self._record_use_outcome(
                            connection,
                            subject_id=entry.cache_id,
                            subject_kind="cache",
                            status=LookupStatus.HIT.value,
                            reason_codes=(
                                RejectionReason.CACHE_NON_AUTHORITATIVE.value,
                            ),
                            body={
                                "promotes_assurance": False,
                                "key_id": cache_key.key_id,
                            },
                        )
                        self._commit_if_idle(connection)
                    # Structural hit is allowed only as non-assurance metadata.
                    return LookupResult(
                        LookupStatus.HIT,
                        cache_key,
                        cache_entry=entry,
                        reason_codes=(
                            RejectionReason.CACHE_NON_AUTHORITATIVE.value,
                        ),
                    )
                accumulated.update(reasons)

            if record_use:
                self._record_use_outcome(
                    connection,
                    subject_id=cache_key.key_id,
                    subject_kind="cache_key",
                    status=LookupStatus.REJECTED.value,
                    reason_codes=tuple(sorted(accumulated)),
                )
                self._commit_if_idle(connection)
            return LookupResult(
                LookupStatus.REJECTED,
                cache_key,
                reason_codes=tuple(sorted(accumulated)),
            )

    def cache_promotes_assurance(self) -> bool:
        """Caches never promote assurance."""

        return False

    # -- invalidations -------------------------------------------------------

    def invalidate(
        self,
        subject_id: str,
        *,
        reason: str,
        subject_kind: str = "receipt",
        invalidated_by: str = "",
        body: Mapping[str, Any] | None = None,
    ) -> InvalidationRecord:
        selected = _text(subject_id, "subject_id")
        selected_reason = _text(reason, "reason")
        selected_kind = _text(subject_kind, "subject_kind")
        selected_by = _text(invalidated_by, "invalidated_by", required=False)
        payload = _bounded_mapping(body, redact=False)
        now = self._now_ms()
        material = {
            "subject_id": selected,
            "subject_kind": selected_kind,
            "reason": selected_reason,
            "invalidated_at_ms": now,
            "invalidated_by": selected_by,
            "body": payload,
        }
        invalidation_id = "invalidation:" + _sha256_hex(
            _canonical_json(material).encode("utf-8")
        )

        with self._lock:
            connection = self._require()
            existing = connection.execute(
                "SELECT * FROM invalidations WHERE invalidation_id=?",
                [invalidation_id],
            ).fetchone()
            if existing is not None:
                mapping = _row_mapping(existing)
                try:
                    body_value = json.loads(str(mapping.get("body_json") or "{}"))
                except json.JSONDecodeError as exc:
                    raise DatabaseEvidenceStoreIntegrityError(
                        "poisoned invalidation envelope"
                    ) from exc
                return InvalidationRecord(
                    invalidation_id=str(mapping["invalidation_id"]),
                    subject_id=str(mapping["subject_id"]),
                    subject_kind=str(mapping["subject_kind"]),
                    reason=str(mapping["reason"]),
                    invalidated_at_ms=int(mapping["invalidated_at_ms"]),
                    invalidated_by=str(mapping.get("invalidated_by") or ""),
                    body=MappingProxyType(
                        body_value if isinstance(body_value, dict) else {}
                    ),
                    admitted_sequence=int(mapping.get("admitted_sequence") or 0),
                )

            connection.execute("BEGIN TRANSACTION")
            try:
                seq = self._next_sequence()
                connection.execute(
                    """
                    INSERT INTO invalidations(
                        invalidation_id, subject_id, subject_kind, reason,
                        invalidated_at_ms, invalidated_by, body_json,
                        admitted_sequence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        invalidation_id,
                        selected,
                        selected_kind,
                        selected_reason,
                        now,
                        selected_by,
                        _canonical_json(payload),
                        seq,
                    ],
                )
                # Drop reusable flight outcomes for the subject key when present.
                connection.execute(
                    "DELETE FROM flight_outcomes WHERE key_id=?",
                    [selected],
                )
                self._admit_evidence(
                    connection,
                    evidence_kind=EvidenceKind.INVALIDATION,
                    subject_id=invalidation_id,
                    content_digest=_sha256_hex(
                        _canonical_json(material).encode("utf-8")
                    ),
                    body=material,
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
            return InvalidationRecord(
                invalidation_id=invalidation_id,
                subject_id=selected,
                subject_kind=selected_kind,
                reason=selected_reason,
                invalidated_at_ms=now,
                invalidated_by=selected_by,
                body=MappingProxyType(payload),
                admitted_sequence=seq,
            )

    # -- single flight -------------------------------------------------------

    def _claim_flight(
        self,
        key_id: str,
        *,
        owner_id: str,
        lease_seconds: int,
    ) -> tuple[bool, str, int]:
        now = self._now_ms()
        expires = now + lease_seconds * 1000
        token = uuid.uuid4().hex
        with self._lock:
            connection = self._require()
            connection.execute("BEGIN TRANSACTION")
            try:
                row = connection.execute(
                    "SELECT * FROM flight_leases WHERE key_id=?",
                    [key_id],
                ).fetchone()
                if row is not None:
                    mapping = _row_mapping(row)
                    if int(mapping["expires_at_ms"]) > now:
                        connection.execute("COMMIT")
                        return (
                            False,
                            str(mapping["token"]),
                            int(mapping["fencing_token"]),
                        )
                    fencing = int(mapping["fencing_token"]) + 1
                    connection.execute(
                        """
                        UPDATE flight_leases
                        SET owner_id=?, token=?, fencing_token=?,
                            acquired_at_ms=?, expires_at_ms=?
                        WHERE key_id=?
                        """,
                        [owner_id, token, fencing, now, expires, key_id],
                    )
                else:
                    fencing = 1
                    connection.execute(
                        """
                        INSERT INTO flight_leases(
                            key_id, owner_id, token, fencing_token,
                            acquired_at_ms, expires_at_ms
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        [key_id, owner_id, token, fencing, now, expires],
                    )
                connection.execute("COMMIT")
            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            self._commit_if_idle(connection)
            return True, token, fencing

    def _renew_flight(
        self,
        key_id: str,
        *,
        owner_id: str,
        token: str,
        fencing_token: int,
        lease_seconds: int,
    ) -> None:
        now = self._now_ms()
        expires = now + lease_seconds * 1000
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT * FROM flight_leases
                WHERE key_id=? AND owner_id=? AND token=? AND fencing_token=?
                """,
                [key_id, owner_id, token, fencing_token],
            ).fetchone()
            if row is None:
                raise SingleFlightError("single-flight lease was fenced")
            connection.execute(
                """
                UPDATE flight_leases SET expires_at_ms=?
                WHERE key_id=? AND token=? AND fencing_token=?
                """,
                [expires, key_id, token, fencing_token],
            )
            self._commit_if_idle(connection)

    def _publish_flight(
        self,
        key_id: str,
        *,
        token: str,
        fencing_token: int,
        status: str,
        value: Any,
        outcome_ttl_seconds: int,
    ) -> None:
        now = self._now_ms()
        encoded = _canonical_json(value)
        digest = _sha256_hex(encoded.encode("utf-8"))
        with self._lock:
            connection = self._require()
            row = connection.execute(
                """
                SELECT * FROM flight_leases
                WHERE key_id=? AND token=? AND fencing_token=?
                """,
                [key_id, token, fencing_token],
            ).fetchone()
            if row is None:
                raise SingleFlightError(
                    "cannot publish flight outcome without ownership"
                )
            connection.execute(
                """
                INSERT OR REPLACE INTO flight_outcomes(
                    key_id, fencing_token, status, outcome_json,
                    outcome_digest, created_at_ms, expires_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    key_id,
                    fencing_token,
                    status,
                    encoded,
                    digest,
                    now,
                    now + outcome_ttl_seconds * 1000,
                ],
            )
            connection.execute(
                "DELETE FROM flight_leases WHERE key_id=?",
                [key_id],
            )
            self._commit_if_idle(connection)

    def _flight_outcome(
        self, key_id: str
    ) -> tuple[Any, int] | None:
        now = self._now_ms()
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT * FROM flight_outcomes WHERE key_id=?",
                [key_id],
            ).fetchone()
            if row is None:
                return None
            mapping = _row_mapping(row)
            if int(mapping["expires_at_ms"]) <= now:
                return None
            if str(mapping["status"]) != "ok":
                return None
            try:
                value = json.loads(str(mapping["outcome_json"]))
            except json.JSONDecodeError as exc:
                raise DatabaseEvidenceStoreIntegrityError(
                    "poisoned flight outcome"
                ) from exc
            observed = _sha256_hex(
                str(mapping["outcome_json"]).encode("utf-8")
            )
            if observed != str(mapping["outcome_digest"]):
                # Recompute with canonical form for resilience.
                recomputed = _sha256_hex(
                    _canonical_json(value).encode("utf-8")
                )
                if recomputed != str(mapping["outcome_digest"]):
                    raise DatabaseEvidenceStoreIntegrityError(
                        "poisoned flight outcome digest"
                    )
            return value, int(mapping["fencing_token"])

    def single_flight(
        self,
        key: EvidenceKey | Mapping[str, Any],
        producer: Callable[[], T],
        *,
        owner_id: str | None = None,
        lease_seconds: int = DEFAULT_FLIGHT_LEASE_SECONDS,
        wait_timeout_seconds: float = DEFAULT_FLIGHT_WAIT_SECONDS,
        poll_interval_seconds: float = 0.02,
        outcome_ttl_seconds: int = DEFAULT_FLIGHT_OUTCOME_TTL_SECONDS,
    ) -> SingleFlightResult:
        """Run one producer for all equivalent active supervisors.

        Outcomes are coordination results only; they never promote assurance
        and never replace admitted receipts.
        """

        cache_key = (
            key if isinstance(key, EvidenceKey) else EvidenceKey.from_dict(key)
        )
        if (
            lease_seconds <= 0
            or wait_timeout_seconds <= 0
            or poll_interval_seconds <= 0
            or outcome_ttl_seconds <= 0
        ):
            raise DatabaseEvidenceStoreBoundsError(
                "single-flight time bounds must be positive"
            )
        existing = self._flight_outcome(cache_key.key_id)
        if existing is not None:
            return SingleFlightResult(existing[0], False, existing[1])

        owner = owner_id or (
            f"{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex}"
        )
        deadline = time.monotonic() + float(wait_timeout_seconds)
        while time.monotonic() < deadline:
            acquired, token, fencing_token = self._claim_flight(
                cache_key.key_id,
                owner_id=owner,
                lease_seconds=lease_seconds,
            )
            if acquired:
                heartbeat_stop = threading.Event()
                heartbeat_errors: list[BaseException] = []

                def heartbeat() -> None:
                    interval = max(0.05, lease_seconds / 3)
                    while not heartbeat_stop.wait(interval):
                        try:
                            self._renew_flight(
                                cache_key.key_id,
                                owner_id=owner,
                                token=token,
                                fencing_token=fencing_token,
                                lease_seconds=lease_seconds,
                            )
                        except BaseException as exc:  # pragma: no cover
                            heartbeat_errors.append(exc)
                            return

                heartbeat_thread = threading.Thread(
                    target=heartbeat,
                    name=f"database-evidence-flight-{fencing_token}",
                    daemon=True,
                )
                heartbeat_thread.start()
                try:
                    raw = producer()
                    # Canonicalize JSON-compatible values only.
                    if isinstance(raw, Mapping):
                        value: Any = dict(raw)
                    elif isinstance(raw, (str, int, bool)) or raw is None:
                        value = raw
                    elif isinstance(raw, Sequence) and not isinstance(
                        raw, (str, bytes, bytearray)
                    ):
                        value = list(raw)
                    else:
                        value = {"repr": str(raw), "type": type(raw).__name__}
                    # Round-trip for stability.
                    value = json.loads(_canonical_json(value))
                    heartbeat_stop.set()
                    heartbeat_thread.join(timeout=1.0)
                    if heartbeat_errors:
                        raise SingleFlightError(
                            "single-flight lease heartbeat was fenced"
                        ) from heartbeat_errors[0]
                    self._publish_flight(
                        cache_key.key_id,
                        token=token,
                        fencing_token=fencing_token,
                        status="ok",
                        value=value,
                        outcome_ttl_seconds=outcome_ttl_seconds,
                    )
                    return SingleFlightResult(value, True, fencing_token)
                except BaseException as exc:
                    heartbeat_stop.set()
                    heartbeat_thread.join(timeout=1.0)
                    try:
                        self._publish_flight(
                            cache_key.key_id,
                            token=token,
                            fencing_token=fencing_token,
                            status="error",
                            value={
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                            outcome_ttl_seconds=outcome_ttl_seconds,
                        )
                    except BaseException:
                        pass
                    raise
                finally:
                    heartbeat_stop.set()
                    try:
                        heartbeat_thread.join(timeout=1.0)
                    except Exception:
                        pass

            observed_fence = fencing_token
            while time.monotonic() < deadline:
                outcome = self._flight_outcome(cache_key.key_id)
                if outcome is not None and outcome[1] >= observed_fence:
                    return SingleFlightResult(outcome[0], False, outcome[1])
                with self._lock:
                    connection = self._require()
                    active = connection.execute(
                        """
                        SELECT fencing_token, expires_at_ms
                        FROM flight_leases WHERE key_id=?
                        """,
                        [cache_key.key_id],
                    ).fetchone()
                if (
                    active is None
                    or int(_row_mapping(active)["expires_at_ms"])
                    <= self._now_ms()
                    or int(_row_mapping(active)["fencing_token"])
                    != observed_fence
                ):
                    break
                time.sleep(
                    min(
                        poll_interval_seconds,
                        max(0.0, deadline - time.monotonic()),
                    )
                )
        raise SingleFlightTimeout(
            f"timed out waiting for evidence flight {cache_key.key_id}"
        )

    run_single_flight = single_flight
    execute_single_flight = single_flight

    # -- projections / rebuild -----------------------------------------------

    def rebuild_projection(
        self,
        projection_kind: str = "evidence_catalog",
    ) -> EvidenceProjection:
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
                    raise DatabaseEvidenceStoreIntegrityError(
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

            receipts = [
                event
                for event in events
                if event["evidence_kind"] == EvidenceKind.RECEIPT.value
            ]
            attestations = [
                event
                for event in events
                if event["evidence_kind"] == EvidenceKind.ATTESTATION.value
            ]
            caches = [
                event
                for event in events
                if event["evidence_kind"] == EvidenceKind.CACHE.value
            ]
            invalidations = [
                event
                for event in events
                if event["evidence_kind"] == EvidenceKind.INVALIDATION.value
            ]
            use_outcomes = [
                event
                for event in events
                if event["evidence_kind"] == EvidenceKind.USE_OUTCOME.value
            ]
            body = {
                "projection_kind": kind,
                "snapshot_id": self._snapshot_id,
                "receipt_count": len(receipts),
                "attestation_count": len(attestations),
                "cache_count": len(caches),
                "invalidation_count": len(invalidations),
                "use_outcome_count": len(use_outcomes),
                "receipts": receipts,
                "attestations": attestations,
                "caches": caches,
                "invalidations": invalidations,
                "use_outcomes": use_outcomes,
                "rebuilt_from_admitted_evidence": True,
                "cache_promotes_assurance": False,
                "authority": _AUTHORITY_DATABASE,
            }
            content_digest = _sha256_hex(
                _canonical_json(body).encode("utf-8")
            )
            stamp = _utc_iso()
            projection_id = f"projection:{kind}:{content_digest}"
            connection.execute(
                """
                INSERT OR REPLACE INTO evidence_projections(
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
            return EvidenceProjection(
                projection_id=projection_id,
                projection_kind=kind,
                content_digest=content_digest,
                rebuilt_from_sequence=max_seq,
                rebuilt_at=stamp,
                body=MappingProxyType(body),
            )

    # -- exports -------------------------------------------------------------

    def export_json(self, export_path: Path | str) -> EvidenceExportReceipt:
        """Render a JSON export. File freshness never grants authority."""

        path = Path(export_path)
        with self._lock:
            connection = self._require()
            receipt_rows = connection.execute(
                "SELECT receipt_json FROM evidence_receipts ORDER BY created_at_ms"
            ).fetchall()
            receipts: list[dict[str, Any]] = []
            for row in receipt_rows:
                mapping = _row_mapping(row)
                try:
                    value = json.loads(str(mapping.get("receipt_json") or "{}"))
                except json.JSONDecodeError as exc:
                    raise DatabaseEvidenceStoreIntegrityError(
                        "poisoned receipt envelope"
                    ) from exc
                if isinstance(value, dict):
                    receipts.append(value)
            document = {
                "schema": EVIDENCE_EXPORT_RECEIPT_SCHEMA,
                "authority": _AUTHORITY_EXPORT_ONLY,
                "authoritative": False,
                "snapshot_id": self._snapshot_id,
                "cache_promotes_assurance": False,
                "receipts": receipts,
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
            connection.execute(
                """
                INSERT OR REPLACE INTO evidence_exports(
                    export_id, export_path, export_format, content_digest,
                    subject_count, recorded_at, authority, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    export_id,
                    str(path),
                    "json",
                    content_digest,
                    len(receipts),
                    stamp,
                    _AUTHORITY_EXPORT_ONLY,
                    _canonical_json({"receipt_count": len(receipts)}),
                ],
            )
            self._commit_if_idle(connection)
            return EvidenceExportReceipt(
                export_id=export_id,
                export_path=str(path),
                export_format="json",
                content_digest=content_digest,
                subject_count=len(receipts),
                recorded_at=stamp,
            )

    def authority_unaffected_by_export_deletion(
        self, export_path: Path | str
    ) -> bool:
        path = Path(export_path)
        with self._lock:
            connection = self._require()
            before = connection.execute(
                "SELECT COUNT(*) AS n FROM evidence_receipts"
            ).fetchone()
            before_count = int(_row_mapping(before).get("n") or 0)
            if path.exists():
                path.unlink()
            after = connection.execute(
                "SELECT COUNT(*) AS n FROM evidence_receipts"
            ).fetchone()
            after_count = int(_row_mapping(after).get("n") or 0)
            return before_count == after_count and not path.exists()

    def file_freshness_is_non_authoritative(self) -> bool:
        return True

    def list_use_outcomes(
        self, *, subject_id: str | None = None, limit: int = 256
    ) -> list[UseOutcome]:
        limit = min(_positive_int(limit, "limit"), 4_096)
        with self._lock:
            connection = self._require()
            if subject_id is not None:
                rows = connection.execute(
                    """
                    SELECT * FROM use_outcomes
                    WHERE subject_id=?
                    ORDER BY observed_at_ms ASC
                    LIMIT ?
                    """,
                    [_text(subject_id, "subject_id"), limit],
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM use_outcomes
                    ORDER BY observed_at_ms ASC
                    LIMIT ?
                    """,
                    [limit],
                ).fetchall()
            results: list[UseOutcome] = []
            for row in rows:
                mapping = _row_mapping(row)
                try:
                    reasons = json.loads(
                        str(mapping.get("reason_codes_json") or "[]")
                    )
                    body = json.loads(str(mapping.get("body_json") or "{}"))
                except json.JSONDecodeError as exc:
                    raise DatabaseEvidenceStoreIntegrityError(
                        "poisoned use-outcome envelope"
                    ) from exc
                results.append(
                    UseOutcome(
                        outcome_id=str(mapping["outcome_id"]),
                        subject_id=str(mapping["subject_id"]),
                        subject_kind=str(mapping["subject_kind"]),
                        status=str(mapping["status"]),
                        reason_codes=tuple(
                            str(item) for item in reasons
                        )
                        if isinstance(reasons, list)
                        else (),
                        observed_at_ms=int(mapping["observed_at_ms"]),
                        body=MappingProxyType(
                            body if isinstance(body, dict) else {}
                        ),
                        admitted_sequence=int(
                            mapping.get("admitted_sequence") or 0
                        ),
                    )
                )
            return results

    def admitted_evidence_count(self) -> int:
        with self._lock:
            connection = self._require()
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM admitted_evidence"
            ).fetchone()
            return int(_row_mapping(row).get("n") or 0)


def open_database_evidence_store(
    database_path: Path | str,
    *,
    cas_root: Path | str | None = None,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    default_receipt_ttl_seconds: int = DEFAULT_RECEIPT_TTL_SECONDS,
    default_cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    default_negative_ttl_seconds: int = DEFAULT_NEGATIVE_TTL_SECONDS,
    auto_redact: bool = True,
    clock: Callable[[], float] | None = None,
) -> DatabaseEvidenceStore:
    """Open and return a :class:`DatabaseEvidenceStore`."""

    store = DatabaseEvidenceStore(
        database_path,
        cas_root=cas_root,
        snapshot_id=snapshot_id,
        default_receipt_ttl_seconds=default_receipt_ttl_seconds,
        default_cache_ttl_seconds=default_cache_ttl_seconds,
        default_negative_ttl_seconds=default_negative_ttl_seconds,
        auto_redact=auto_redact,
        clock=clock or time.time,
    )
    return store.open()


__all__ = [
    "ATTESTATION_RECORD_SCHEMA",
    "CACHE_ENTRY_SCHEMA",
    "DATABASE_EVIDENCE_STORE_INTERFACE",
    "DATABASE_EVIDENCE_STORE_SCHEMA",
    "DEFAULT_CACHE_TTL_SECONDS",
    "DEFAULT_FLIGHT_LEASE_SECONDS",
    "DEFAULT_FLIGHT_OUTCOME_TTL_SECONDS",
    "DEFAULT_FLIGHT_WAIT_SECONDS",
    "DEFAULT_NEGATIVE_TTL_SECONDS",
    "DEFAULT_RECEIPT_TTL_SECONDS",
    "DEFAULT_SNAPSHOT_ID",
    "EVIDENCE_EXPORT_RECEIPT_SCHEMA",
    "EVIDENCE_KEY_SCHEMA",
    "EVIDENCE_PROJECTION_SCHEMA",
    "EVIDENCE_RECEIPT_SCHEMA",
    "INVALIDATION_RECORD_SCHEMA",
    "USE_OUTCOME_SCHEMA",
    "AssuranceLevel",
    "AttestationRecord",
    "CacheDomain",
    "CacheEntry",
    "DatabaseEvidenceStore",
    "DatabaseEvidenceStoreBoundsError",
    "DatabaseEvidenceStoreConflictError",
    "DatabaseEvidenceStoreError",
    "DatabaseEvidenceStoreIntegrityError",
    "DatabaseEvidenceStoreNotOpenError",
    "DuckDBUnavailableError",
    "EvidenceExportReceipt",
    "EvidenceKey",
    "EvidenceKind",
    "EvidenceOutcome",
    "EvidenceProjection",
    "EvidenceReceipt",
    "EvidenceVerdict",
    "InvalidationRecord",
    "LookupResult",
    "LookupStatus",
    "REDACTION_MARKER",
    "RejectionReason",
    "SingleFlightError",
    "SingleFlightExecutionError",
    "SingleFlightResult",
    "SingleFlightTimeout",
    "UseOutcome",
    "duckdb_available",
    "open_database_evidence_store",
]
